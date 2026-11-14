//! Checkpoint-native geometry for block dSpark drafters.
//! This is admission metadata, not a claim that a matching GPU export exists.
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

/// The native Markov kernel tiles independent sequences over fixed 32-sequence arrays.
pub const MAX_SEQUENCES: usize = cuteafd_core::concurrency::MAX_CONCURRENCY as usize;
pub const fn supports_sequences(sequences: usize) -> bool {
    sequences > 0 && sequences <= MAX_SEQUENCES
}

#[derive(Debug, thiserror::Error)]
pub enum DsparkConfigError {
    #[error(transparent)]
    Topk(#[from] super::dflash::DflashTopkVocabError),
    #[error("reading dSpark config: {0}")]
    Io(#[from] std::io::Error),
    #[error("parsing dSpark config: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported dSpark checkpoint: {0}")]
    Unsupported(String),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "rope_type")]
pub enum DsparkRope {
    #[serde(rename = "default")]
    Default { rope_theta: f64 },
    #[serde(rename = "yarn")]
    Yarn {
        rope_theta: f64,
        factor: f64,
        original_max_position_embeddings: usize,
        #[serde(default = "default_beta_fast")]
        beta_fast: f64,
        #[serde(default = "default_beta_slow")]
        beta_slow: f64,
        #[serde(default)]
        mscale: Option<f64>,
        #[serde(default)]
        mscale_all_dim: Option<f64>,
    },
}

fn default_beta_fast() -> f64 { 32.0 }
fn default_beta_slow() -> f64 { 1.0 }

/// Immutable draft-only rotary inputs. The legacy 64-wide default route
/// computes its frequencies in the kernel instead of consuming this table.
#[derive(Debug, Clone, PartialEq)]
pub struct DsparkRotaryPlan {
    pub inv_freq: Vec<f32>,
    /// Magnitude applied to rotary query/key coordinates.
    pub attention_factor: f32,
    /// Additional MLA softmax multiplier, separate from rotary magnitude.
    pub softmax_factor: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsparkNorm { Rms, Gemma }

/// Arithmetic belongs to the checkpoint, independently of the draft lifecycle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DsparkBackbone {
    pub norm: DsparkNorm,
    pub rotary_dim: usize,
    pub activation: cuteafd_core::FfnActivation,
}

impl DsparkBackbone {
    pub fn from_layers(raw: &Value, head_dim: usize) -> Result<Self, DsparkConfigError> {
        use DsparkConfigError::Unsupported;
        let norm = match raw.get("use_gemma_norm") {
            None | Some(Value::Bool(false)) => DsparkNorm::Rms,
            Some(Value::Bool(true)) => DsparkNorm::Gemma,
            _ => return Err(Unsupported("use_gemma_norm must be boolean".into())),
        };
        let rotary_dim = match raw.get("rotary_dim") {
            None => head_dim,
            Some(value) => value.as_u64().and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Unsupported("invalid rotary_dim".into()))?,
        };
        if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
            return Err(Unsupported("rotary_dim must be positive, even and no wider than a head".into()));
        }
        if let Some(value) = raw.get("partial_rotary_factor") {
            let factor = value.as_f64().filter(|n| n.is_finite())
                .ok_or_else(|| Unsupported("invalid partial_rotary_factor".into()))?;
            if (factor * head_dim as f64 - rotary_dim as f64).abs() > 1e-9 {
                return Err(Unsupported("partial_rotary_factor disagrees with rotary_dim".into()));
            }
        }
        let activation = match raw["hidden_act"].as_str() {
            Some("silu") => cuteafd_core::FfnActivation::SiluBf16Product { limit: None },
            Some("swigluoai") => {
                let number = |name: &str| raw[name].as_f64().map(|n| n as f32)
                    .ok_or_else(|| Unsupported(format!("missing or invalid {name}")));
                cuteafd_core::FfnActivation::SwigluOai {
                    alpha: number("swiglu_alpha")?, beta: 1.0, limit: number("swiglu_limit")?,
                }
            }
            _ => return Err(Unsupported("unsupported dSpark FFN activation".into())),
        }.validate().map_err(|error| Unsupported(error.into()))?;
        Ok(Self { norm, rotary_dim, activation })
    }

    pub fn is_qwen3(self, head_dim: usize) -> bool {
        self.norm == DsparkNorm::Rms && self.rotary_dim == head_dim
            && self.activation == (cuteafd_core::FfnActivation::SiluBf16Product { limit: None })
    }
}

impl DsparkRope {
    pub fn plan(&self, head_dim: usize) -> Result<DsparkRotaryPlan, DsparkConfigError> {
        use DsparkConfigError::Unsupported;
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(Unsupported(
                "rotary head dimension must be positive and even".into(),
            ));
        }
        let (theta, yarn) = match *self {
            Self::Default { rope_theta } => (rope_theta, None),
            Self::Yarn {
                rope_theta,
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                mscale,
                mscale_all_dim,
            } => {
                if !factor.is_finite()
                    || factor < 1.0
                    || original_max_position_embeddings == 0
                    || !beta_fast.is_finite()
                    || !beta_slow.is_finite()
                    || beta_fast <= beta_slow
                    || beta_slow <= 0.0
                    || [mscale, mscale_all_dim].into_iter().flatten()
                        .any(|scale| !scale.is_finite() || scale < 0.0)
                {
                    return Err(Unsupported("invalid YaRN rotary parameters".into()));
                }
                (
                    rope_theta,
                    Some((
                        factor,
                        original_max_position_embeddings,
                        beta_fast,
                        beta_slow,
                    )),
                )
            }
        };
        if !theta.is_finite() || theta <= 1.0 || !(theta as f32).is_finite() {
            return Err(Unsupported(
                "rotary theta must be finite and greater than one in FP32".into(),
            ));
        }
        let (attention_factor, softmax_factor) = match *self {
            Self::Yarn { factor, mscale, mscale_all_dim, .. } => {
                let magnitude = |scale: f64| if factor <= 1.0 { 1.0 }
                    else { 1.0 + 0.1 * scale * factor.ln() };
                // MLA separates the rotary-coordinate magnitude from the
                // all-dimension query scale. GQA without these fields keeps HF YaRN.
                let all = magnitude(mscale_all_dim.unwrap_or(0.0));
                ((magnitude(mscale.unwrap_or(1.0)) / all) as f32, (all * all) as f32)
            }
            Self::Default { .. } => (1.0, 1.0),
        };
        let correction = yarn.map(|(_, original, fast, slow)| {
            let dimension = |rotations: f64| {
                head_dim as f64 * (original as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                    / (2.0 * theta.ln())
            };
            let low = dimension(fast).floor().max(0.0) as f32;
            let mut high = dimension(slow).ceil().min((head_dim - 1) as f64) as f32;
            if low == high {
                high += 0.001;
            }
            (low, high)
        });
        let inv_freq = (0..head_dim / 2)
            .map(|i| {
                // Preserve the reference's FP32 operation boundaries, including
                // interpolation before blending (not division of extrapolation).
                let frequency = (theta as f32).powf((2 * i) as f32 / head_dim as f32);
                let extrapolation = 1.0f32 / frequency;
                match (yarn, correction) {
                    (Some((factor, _, _, _)), Some((low, high))) => {
                        let weight = 1.0 - ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                        let interpolation = 1.0f32 / (factor as f32 * frequency);
                        interpolation * (1.0 - weight) + extrapolation * weight
                    }
                    _ => extrapolation,
                }
            })
            .collect::<Vec<_>>();
        if !attention_factor.is_finite() || !softmax_factor.is_finite()
            || inv_freq.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(Unsupported(
                "rotary plan is not representable in FP32".into(),
            ));
        }
        Ok(DsparkRotaryPlan {
            inv_freq,
            attention_factor,
            softmax_factor,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockDsparkConfig {
    pub backbone: DsparkBackbone,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f64,
    pub rope: DsparkRope,
    pub block: usize,
    pub mask_token: u32,
    pub rank: usize,
    /// Zero-based target post-block outputs, in projector concatenation order.
    pub taps: Vec<usize>,
    pub vocab: usize,
    /// None means full context, never a silently truncated sliding ring.
    pub window: Option<usize>,
    pub causal: bool,
}

#[derive(Deserialize)]
struct Layers {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_parameters: DsparkRope,
    vocab_size: usize,
    hidden_act: String,
    layer_types: Vec<String>,
    sliding_window: Option<usize>,
}

impl BlockDsparkConfig {
    /// Full layers share the target token budget; windowed layers keep their
    /// original slot-local rings. Both use the shared draft cache inventory.
    pub fn paged_cache_layout(&self, slots: usize, context: usize, target_tokens: usize,
        kv: crate::families::qwen4::Qwen4KvCache) -> Result<super::dflash::DraftCacheLayout, DsparkConfigError> {
        let invalid = || DsparkConfigError::Unsupported("invalid dSpark paged cache admission".into());
        if slots == 0 || context == 0 || target_tokens == 0
            || context.checked_add(self.block).is_none_or(|n| n > i32::MAX as usize)
            || self.window.is_some_and(|window| window != context) { return Err(invalid()); }
        let pages = target_tokens.div_ceil(64);
        let rows = pages.checked_mul(64).ok_or_else(invalid)?;
        if rows > i32::MAX as usize { return Err(invalid()); }
        let plane = kv.plane_record_bytes(self.kv_heads, self.head_dim);
        let record_bytes = plane.checked_mul(2).ok_or_else(invalid)?;
        let full = self.window.is_none();
        let cache_rows = if full { rows } else { slots.checked_mul(context).ok_or_else(invalid)? };
        let layer_bytes = cache_rows.checked_mul(record_bytes).ok_or_else(invalid)?;
        let total = layer_bytes.checked_mul(self.layers).ok_or_else(invalid)? as u64;
        let page_table_bytes = if full { slots.checked_mul(context.div_ceil(64))
            .and_then(|n| n.checked_mul(4)).ok_or_else(invalid)? as u64 } else { 0 };
        Ok(super::dflash::DraftCacheLayout { target_tokens, page_rows: 64, pages, record_bytes,
            full_layers: if full { self.layers } else { 0 },
            window_bytes: if full { 0 } else { total }, full_bytes: if full { total } else { 0 },
            page_table_bytes, layer_bytes: vec![layer_bytes; self.layers] })
    }

    pub fn read(snapshot: &Path) -> Result<Self, DsparkConfigError> {
        Self::from_json(&serde_json::from_slice(&std::fs::read(
            snapshot.join("config.json"),
        )?)?)
    }

    pub fn from_json(raw: &Value) -> Result<Self, DsparkConfigError> {
        use DsparkConfigError::Unsupported;
        let speculators = raw["speculators_model_type"] == "dspark";
        // Speculators also advertises DSparkDraftModel; its explicit dialect wins.
        let specforge = !speculators
            && raw["architectures"]
                .as_array()
                .is_some_and(|a| a.iter().any(|v| v == "DSparkDraftModel"));
        if !speculators && !specforge {
            return Err(Unsupported(
                "expected DSparkDraftModel or speculators_model_type=dspark".into(),
            ));
        }
        let layer_config = if speculators { &raw["transformer_layer_config"] } else { raw };
        let t: Layers = serde_json::from_value(layer_config.clone())?;
        let backbone = DsparkBackbone::from_layers(layer_config, t.head_dim)?;
        let d = if speculators {
            raw
        } else {
            &raw["dflash_config"]
        };
        let integer = |value: &Value, name: &str| {
            value
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| Unsupported(format!("missing or invalid {name}")))
        };
        let block = integer(&raw["block_size"], "block_size")?;
        let rank = integer(&raw["markov_rank"], "markov_rank")?;
        let mask_token = u32::try_from(integer(&d["mask_token_id"], "mask_token_id")?)
            .map_err(|_| Unsupported("mask_token_id exceeds u32".into()))?;
        let layer_ids: Vec<usize> = serde_json::from_value(
            d[if speculators {
                "aux_hidden_state_layer_ids"
            } else {
                "target_layer_ids"
            }]
            .clone(),
        )?;
        let taps = if speculators {
            layer_ids
                .into_iter()
                .map(|id| {
                    id.checked_sub(1).ok_or_else(|| {
                        Unsupported("embedding aux id 0 is not a post-block tap".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            layer_ids
        };
        if t.hidden_size == 0
            || t.intermediate_size == 0
            || t.num_hidden_layers == 0
            || t.num_key_value_heads == 0
            || t.num_attention_heads == 0
            || t.num_attention_heads % t.num_key_value_heads != 0
            || t.head_dim == 0
            || t.head_dim % 2 != 0
            || block == 0
            || rank == 0
            || taps.is_empty()
            || t.vocab_size == 0
            || mask_token as usize >= t.vocab_size
            || !t.rms_norm_eps.is_finite()
            || t.rms_norm_eps <= 0.0
        {
            return Err(Unsupported(
                "invalid dimensions, GQA grouping, epsilon, taps or mask token".into(),
            ));
        }
        if raw["markov_head_type"] != "vanilla"
            || raw["enable_confidence_head"] != true
            || raw["confidence_head_with_markov"] != true
            || t.hidden_act != "silu"
            || !backbone.is_qwen3(t.head_dim)
            || (speculators
                && (raw["sample_from_anchor"] != true
                    || raw["draft_vocab_size"].as_u64() != Some(t.vocab_size as u64)))
            || (specforge && d["projector_type"] != "dspark")
        {
            return Err(Unsupported("requires SiLU, vanilla Markov, Markov confidence, anchor sampling and the full target vocabulary".into()));
        }
        let sliding = t.layer_types.len() == t.num_hidden_layers
            && t.layer_types.iter().all(|v| v == "sliding_attention");
        let full = t.layer_types.len() == t.num_hidden_layers
            && t.layer_types.iter().all(|v| v == "full_attention");
        let window = match (sliding, full, t.sliding_window) {
            (true, false, Some(n)) if n > 0 => Some(n),
            (false, true, None) => None,
            _ => {
                return Err(Unsupported(
                    "requires uniform full attention or a positive uniform sliding window".into(),
                ))
            }
        };
        if t.rope_parameters.plan(t.head_dim)?.softmax_factor != 1.0 {
            return Err(Unsupported("GQA dSpark has no all-dimension YaRN softmax scale".into()));
        }
        if let Some(target_layers) = raw["num_target_layers"].as_u64() {
            if taps.iter().any(|&l| l as u64 >= target_layers) {
                return Err(Unsupported("tap exceeds num_target_layers".into()));
            }
        }
        let mut unique = taps.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != taps.len() {
            return Err(Unsupported("duplicate target tap".into()));
        }
        Ok(Self {
            backbone,
            hidden: t.hidden_size,
            intermediate: t.intermediate_size,
            layers: t.num_hidden_layers,
            heads: t.num_attention_heads,
            kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            eps: t.rms_norm_eps,
            rope: t.rope_parameters,
            block,
            mask_token,
            rank,
            taps,
            vocab: t.vocab_size,
            window,
            causal: sliding && raw["sliding_window_non_causal"] != true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backbone_descriptor_preserves_qwen_and_reads_native_minimax() {
        use cuteafd_core::FfnActivation;
        for raw in [qwen(), glm_flash_speculators()["transformer_layer_config"].clone()] {
            let head = raw["head_dim"].as_u64().unwrap() as usize;
            let descriptor = DsparkBackbone::from_layers(&raw, head).unwrap();
            assert!(descriptor.is_qwen3(head));
            assert_eq!(descriptor.norm, DsparkNorm::Rms);
            let old = FfnActivation::SiluBf16Product { limit: None };
            for gate in -128..128 {
                for up in [-19.0, -1.125, 0.0, 0.03125, 7.0, 19.0] {
                    let gate = gate as f32 / 16.0;
                    assert_eq!(descriptor.activation.evaluate(gate, up).to_bits(), old.evaluate(gate, up).to_bits());
                }
            }
        }
        let raw: Value = serde_json::from_str(include_str!("../families/minimax_m3/fixtures/dspark-config.json")).unwrap();
        let descriptor = DsparkBackbone::from_layers(&raw["text_config"], 128).unwrap();
        assert_eq!(descriptor, DsparkBackbone { norm: DsparkNorm::Gemma, rotary_dim: 64,
            activation: FfnActivation::SwigluOai { alpha: 1.702, beta: 1.0, limit: 7.0 } });
        assert!(!descriptor.is_qwen3(128));
        // Parsing arithmetic does not claim that runtime bindings exist yet.
        assert!(BlockDsparkConfig::from_json(&raw).is_err());
    }

    #[test]
    fn backbone_descriptor_rejects_ambiguous_arithmetic() {
        for (key, value) in [("use_gemma_norm", json!(1)), ("rotary_dim", json!(0)),
            ("rotary_dim", json!(127)), ("rotary_dim", json!(130)), ("rotary_dim", json!(1.5)),
            ("partial_rotary_factor", json!(0.5)), ("hidden_act", json!("gelu"))] {
            let mut raw = qwen();
            raw[key] = value;
            assert!(DsparkBackbone::from_layers(&raw, 128).is_err(), "{key}");
        }
        for key in ["swiglu_alpha", "swiglu_limit"] {
            let mut raw = json!({"hidden_act":"swigluoai", "swiglu_alpha":1.702, "swiglu_limit":7.0});
            raw[key] = json!(0.0);
            assert!(DsparkBackbone::from_layers(&raw, 128).is_err(), "{key}");
        }
    }

    #[test]
    fn arena_requests_follow_the_shared_recipe_for_both_representations() {
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        use crate::plan::testing::{t, write_safetensors};
        let snapshot = tempfile::tempdir().unwrap();
        let raw = serde_json::json!({"architectures":["DSparkDraftModel"], "speculators_model_type":"dspark",
            "aux_hidden_state_layer_ids":[1,2], "block_size":2, "confidence_head_with_markov":true,
            "draft_vocab_size":16, "enable_confidence_head":true, "markov_head_type":"vanilla", "markov_rank":256,
            "mask_token_id":0, "sample_from_anchor":true, "sliding_window_non_causal":false,
            "transformer_layer_config":{"attention_bias":false, "head_dim":128, "hidden_act":"silu",
                "hidden_size":256, "intermediate_size":256, "layer_types":["full_attention"],
                "max_position_embeddings":64, "model_type":"qwen3", "num_attention_heads":2, "num_hidden_layers":1,
                "num_key_value_heads":1, "rms_norm_eps":1e-5, "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
                "sliding_window":null, "vocab_size":16}});
        std::fs::write(snapshot.path().join("config.json"), serde_json::to_vec(&raw).unwrap()).unwrap();
        let cfg = BlockDsparkConfig::read(snapshot.path()).unwrap();
        let shapes = std::cell::RefCell::new(Vec::new());
        block_dspark_weight_recipe(&cfg, |name, shape| { shapes.borrow_mut().push(t(name, "BF16", shape)); Ok::<_, ()>(()) },
            |parts, cols| { for (name, rows) in parts { shapes.borrow_mut().push(t(*name, "BF16", &[*rows, cols])); } Ok(()) })
            .unwrap();
        write_safetensors(&snapshot.path().join("model.safetensors"), &shapes.into_inner());
        let bf16 = block_dspark_arena_requests(snapshot.path(), GlmDraftRepresentation::Bf16Only).unwrap();
        let qkv = "layers.0.self_attn.q_proj.weight+layers.0.self_attn.k_proj.weight+layers.0.self_attn.v_proj.weight";
        assert_eq!(bf16.iter().find(|(n, _)| n == qkv).unwrap().1, (256 + 128 + 128) * 256 * 2);
        assert_eq!(bf16.iter().find(|(n, _)| n == "layers.0.self_attn.q_norm.weight").unwrap().1, 256);
        assert_eq!(bf16.iter().find(|(n, _)| n == "fc.weight").unwrap().1, 256 * 512 * 2);
        assert_eq!(bf16[bf16.len() - 2], ("markov_norms".to_string(), 16 * 4));
        assert_eq!(bf16.last().unwrap(), &("inv_freq".to_string(), 256));
        let fp8 = block_dspark_arena_requests(snapshot.path(), GlmDraftRepresentation::Fp8Only).unwrap();
        let payload = (256 + 128 + 128) * 256;
        assert!(fp8.contains(&(format!("{qkv}.payload"), payload)) && fp8.contains(&(format!("{qkv}.scales"), payload / 128 * 4)));
        // Every matrix splits into one payload and one scales plane; tensors keep their single request.
        assert_eq!(fp8.len(), bf16.len() + 5);
        let names = bf16.iter().map(|(name, _)| name).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), bf16.len(), "request names must be unique");
    }
    use serde_json::json;

    fn qwen() -> Value {
        json!({"architectures":["DSparkDraftModel"], "hidden_size":5120, "intermediate_size":10240,
            "num_hidden_layers":5,"num_attention_heads":40,"num_key_value_heads":8,"head_dim":128,
            "rms_norm_eps":1e-6,"rope_parameters":{"rope_type":"yarn","rope_theta":10000000.0,
                "factor":32.0,"original_max_position_embeddings":8192,"beta_fast":32.0,"beta_slow":1.0},
            "vocab_size":248320,"hidden_act":"silu","layer_types":["full_attention","full_attention",
                "full_attention","full_attention","full_attention"],"sliding_window":null,"block_size":7,
            "markov_rank":256,"markov_head_type":"vanilla","enable_confidence_head":true,
            "confidence_head_with_markov":true,"num_target_layers":64,
            "dflash_config":{"projector_type":"dspark","mask_token_id":248077,"target_layer_ids":[4,16,28,40,52]}})
    }

    // RedHatAI/GLM-5.3-Flash-speculator.dspark-preview, revision 1972f1f0:
    // both dialect markers are present in the checkpoint's own config.
    fn glm_flash_speculators() -> Value {
        json!({"architectures":["DSparkDraftModel"],
            "speculators_model_type":"dspark","target_hidden_size":null,
            "aux_hidden_state_layer_ids":[20,28,32,36,40,44],"block_size":8,
            "confidence_head_with_markov":true,"draft_vocab_size":154880,
            "enable_confidence_head":true,"markov_head_type":"vanilla","markov_rank":256,
            "mask_token_id":154856,"sample_from_anchor":true,"sliding_window_non_causal":false,
            "transformer_layer_config":{"attention_bias":false,"head_dim":64,
                "hidden_act":"silu","hidden_size":4096,"intermediate_size":12288,
                "layer_types":["sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention"],"max_position_embeddings":1048576,
                "model_type":"qwen3","num_attention_heads":64,"num_hidden_layers":5,
                "num_key_value_heads":64,"rms_norm_eps":1e-5,
                "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
                "sliding_window":2048,"use_sliding_window":true,"vocab_size":154880}})
    }

    #[test]
    fn speculators_architecture_marker_does_not_require_specforge_projector() {
        let c = BlockDsparkConfig::from_json(&glm_flash_speculators()).unwrap();
        assert_eq!((c.hidden, c.intermediate, c.layers), (4096, 12288, 5));
        assert_eq!((c.heads, c.kv_heads, c.head_dim), (64, 64, 64));
        assert_eq!(
            (c.block, c.rank, c.vocab, c.mask_token),
            (8, 256, 154880, 154856)
        );
        assert_eq!(c.taps, vec![19, 27, 31, 35, 39, 43]);
        assert_eq!(c.window, Some(2048));
        assert!(c.causal);
        assert_eq!(
            c.rope,
            DsparkRope::Default {
                rope_theta: 10000.0
            }
        );

        // Dialect selection must not bypass Speculators' semantic checks.
        for (field, value) in [
            ("sample_from_anchor", json!(false)),
            ("draft_vocab_size", json!(154879)),
            ("confidence_head_with_markov", json!(false)),
        ] {
            let mut raw = glm_flash_speculators();
            raw[field] = value;
            assert!(BlockDsparkConfig::from_json(&raw).is_err(), "{field}");
        }
        let mut raw = glm_flash_speculators();
        raw["transformer_layer_config"]["hidden_act"] = json!("gelu");
        assert!(BlockDsparkConfig::from_json(&raw).is_err());
    }

    #[test]
    fn immutable_yarn_plan_uses_reference_fp32_blending() {
        let c = BlockDsparkConfig::from_json(&qwen()).unwrap();
        let plan = c.rope.plan(c.head_dim).unwrap();
        assert_eq!(plan.inv_freq.len(), 64);
        assert!((plan.attention_factor - (1.0f64 + 0.1 * 32.0f64.ln()) as f32).abs() < 1e-7);
        assert_eq!(plan.inv_freq[0], 1.0);
        // Checkpoint correction dimensions floor/ceil to [14, 29].
        for i in [0usize, 14, 20, 29, 63] {
            let frequency = 10_000_000f32.powf((2 * i) as f32 / 128.0);
            let weight = 1.0 - ((i as f32 - 14.0) / 15.0).clamp(0.0, 1.0);
            let expected = (1.0 / (32.0 * frequency)) * (1.0 - weight) + (1.0 / frequency) * weight;
            assert_eq!(plan.inv_freq[i], expected, "frequency {i}");
        }
        // Pinned Qwen3RotaryEmbedding CPU FP32 table, full checkpoint geometry.
        let reference_bits = [
            1065353216u32,
            1061618021,
            1058714411,
            1055949884,
            1052440603,
            1049712609,
            1046607925,
            1043310893,
            1040747895,
            1037323628,
            1034226006,
            1031818025,
            1028093503,
            1025183233,
            1022431607,
            1018121107,
            1014872157,
            1010211608,
            1006802083,
            1002002126,
            998397281,
            993346669,
            989370984,
            984051384,
            979118631,
            973860955,
            967379848,
            960192872,
            950929914,
            934311859,
            931737090,
            928335622,
            925223776,
            922804734,
            919101633,
            916177998,
            913452258,
            909918774,
            907171966,
            904103724,
            900783954,
            898203279,
            894813249,
            891694265,
            889269675,
            885577321,
            882646980,
            879934230,
            876392641,
            873639533,
            870583494,
            867256108,
            864669514,
            861290950,
            858164812,
            855734661,
            852053079,
            849116016,
            846416285,
            842866572,
            840107150,
            837063342,
            833728324,
            831135798,
        ];
        for (&actual, expected) in plan.inv_freq.iter().zip(reference_bits) {
            assert!(
                actual.to_bits().abs_diff(expected) <= 2,
                "frequency {actual} vs {}",
                f32::from_bits(expected)
            );
        }
        assert_eq!(plan, c.rope.plan(c.head_dim).unwrap());
        assert!(plan.inv_freq.windows(2).all(|p| p[0] > p[1]));
    }

    #[test]
    fn mla_yarn_separates_rotary_magnitude_and_softmax_scale() {
        let raw = json!({"rope_type":"yarn", "rope_theta":50000.0,
            "factor":32.0, "original_max_position_embeddings":32768,
            "mscale":1.0, "mscale_all_dim":1.0});
        let rope: DsparkRope = serde_json::from_value(raw.clone()).unwrap();
        let plan = rope.plan(64).unwrap();
        assert_eq!(plan.attention_factor, 1.0);
        assert_eq!(plan.softmax_factor, (1.0 + 0.1 * 32.0f64.ln()).powi(2) as f32);
        let mut explicit = raw.clone();
        explicit["beta_fast"] = json!(32.0);
        explicit["beta_slow"] = json!(1.0);
        assert_eq!(plan, serde_json::from_value::<DsparkRope>(explicit).unwrap().plan(64).unwrap());
        let mut scaled = raw.clone();
        scaled["mscale"] = json!(0.5);
        let scaled = serde_json::from_value::<DsparkRope>(scaled).unwrap().plan(64).unwrap();
        assert_eq!(scaled.inv_freq, plan.inv_freq);
        assert_eq!(scaled.softmax_factor, plan.softmax_factor);
        assert_eq!(scaled.attention_factor,
            ((1.0 + 0.05 * 32.0f64.ln()) / (1.0 + 0.1 * 32.0f64.ln())) as f32);
        for field in ["mscale", "mscale_all_dim"] {
            let mut invalid = raw.clone();
            invalid[field] = json!(-1.0);
            assert!(serde_json::from_value::<DsparkRope>(invalid).unwrap().plan(64).is_err());
        }
        let mut unit = raw;
        unit["factor"] = json!(1.0);
        let unit = serde_json::from_value::<DsparkRope>(unit).unwrap().plan(64).unwrap();
        assert_eq!((unit.attention_factor, unit.softmax_factor), (1.0, 1.0));
    }

    #[test]
    fn rotary_plan_rejects_invalid_direct_construction() {
        assert!(DsparkRope::Default {
            rope_theta: f64::NAN
        }
        .plan(128)
        .is_err());
        assert!(DsparkRope::Default {
            rope_theta: 10000.0
        }
        .plan(127)
        .is_err());
        let plan = DsparkRope::Default {
            rope_theta: 10000.0,
        }
        .plan(64)
        .unwrap();
        assert_eq!(plan.attention_factor, 1.0);
        assert_eq!(plan.inv_freq[0], 1.0);
    }

    #[test]
    fn shared_paged_inventory_is_token_budget_not_slot_context_product() {
        use crate::families::qwen4::Qwen4KvCache;
        let cfg = BlockDsparkConfig::from_json(&qwen()).unwrap();
        for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
            let one = cfg.paged_cache_layout(1, 262144, 8193, kv).unwrap();
            let many = cfg.paged_cache_layout(16, 262144, 8193, kv).unwrap();
            assert_eq!(one.full_bytes, many.full_bytes);
            assert_eq!(one.pages, 129);
            assert_eq!(one.full_bytes, (129 * 64 * 2 * kv.plane_record_bytes(8, 128) * 5) as u64);
            assert_eq!(many.page_table_bytes, 16 * 4096 * 4);
            assert_eq!(many.window_bytes, 0);
            assert!(cfg.paged_cache_layout(16, 262144, 0, kv).is_err());
        }
    }

    #[test]
    fn specforge_post_block_gqa_full_yarn() {
        let c = BlockDsparkConfig::from_json(&qwen()).unwrap();
        assert_eq!(c.taps, vec![4, 16, 28, 40, 52]);
        assert_eq!((c.head_dim, c.heads / c.kv_heads * c.block), (128, 35));
        assert_eq!(c.window, None);
        assert!(!c.causal);
    }

    #[test]
    fn speculators_input_ids_become_post_block_ids() {
        let q = qwen();
        let mut t = q.clone();
        t["sliding_window"] = json!(2048);
        t["layer_types"] = json!(vec!["sliding_attention"; 5]);
        t["rope_parameters"] = json!({"rope_type":"default","rope_theta":10000.0});
        let c = BlockDsparkConfig::from_json(&json!({"speculators_model_type":"dspark",
            "transformer_layer_config":t,"aux_hidden_state_layer_ids":[20,28,32,36,40,44],
            "block_size":8,"markov_rank":256,"markov_head_type":"vanilla","sample_from_anchor":true,
            "enable_confidence_head":true,"confidence_head_with_markov":true,"draft_vocab_size":248320,
            "mask_token_id":248077})).unwrap();
        assert_eq!(c.taps, vec![19, 27, 31, 35, 39, 43]);
        assert_eq!(c.window, Some(2048));
        assert!(c.causal);
    }

    #[test]
    fn rejects_invalid_geometry_and_semantics() {
        for (field, value) in [
            ("num_key_value_heads", json!(0)),
            ("num_key_value_heads", json!(7)),
            ("block_size", json!(0)),
            ("head_dim", json!(127)),
            ("markov_head_type", json!("other")),
            ("enable_confidence_head", json!(false)),
            ("layer_types", json!(["full_attention"])),
        ] {
            let mut c = qwen();
            c[field] = value;
            assert!(BlockDsparkConfig::from_json(&c).is_err(), "{field}");
        }
        let mut c = qwen();
        c["dflash_config"]["target_layer_ids"] = json!([4, 64]);
        assert!(BlockDsparkConfig::from_json(&c).is_err());
    }
}

/// One draft layer's checkpoint operands, in loader order.
pub struct BlockDsparkLayer<T, M> {
    pub input_norm: T, pub post_norm: T, pub qkv: M, pub q_norm: T, pub k_norm: T,
    pub o: M, pub gate_up: M, pub down: M,
}

/// Every checkpoint-sourced drafter weight. `tensor` binds one tensor of a
/// shape; `matrix` binds row-concatenated parts sharing `cols`.
pub struct BlockDsparkWeights<T, M> {
    pub layers: Vec<BlockDsparkLayer<T, M>>, pub fc: M, pub hidden_norm: T, pub norm: T,
    pub markov_w1: T, pub markov_w2: T, pub confidence_w: T, pub confidence_b: T,
}

/// The single recipe for runtime loading and admission: names, shapes and the
/// order in which the loader requests each owner.
pub fn block_dspark_weight_recipe<T, M, E>(cfg: &BlockDsparkConfig,
    mut tensor: impl FnMut(&str, &[usize]) -> Result<T, E>,
    mut matrix: impl FnMut(&[(&str, usize)], usize) -> Result<M, E>) -> Result<BlockDsparkWeights<T, M>, E> {
    let (h, inter) = (cfg.hidden, cfg.intermediate);
    let (q, kv) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
    let layers = (0..cfg.layers).map(|l| -> Result<_, E> {
        let p = format!("layers.{l}");
        let a = format!("{p}.self_attn");
        Ok(BlockDsparkLayer {
            input_norm: tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
            post_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
            qkv: matrix(&[(&format!("{a}.q_proj.weight"), q), (&format!("{a}.k_proj.weight"), kv),
                (&format!("{a}.v_proj.weight"), kv)], h)?,
            q_norm: tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim])?,
            k_norm: tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim])?,
            o: matrix(&[(&format!("{a}.o_proj.weight"), h)], q)?,
            gate_up: matrix(&[(&format!("{p}.mlp.gate_proj.weight"), inter),
                (&format!("{p}.mlp.up_proj.weight"), inter)], h)?,
            down: matrix(&[(&format!("{p}.mlp.down_proj.weight"), h)], inter)?,
        })
    }).collect::<Result<Vec<_>, E>>()?;
    let markov_w2 = tensor("markov_head.markov_w2.weight", &[cfg.vocab, cfg.rank])?;
    Ok(BlockDsparkWeights { layers, markov_w2,
        fc: matrix(&[("fc.weight", h)], cfg.taps.len() * h)?,
        hidden_norm: tensor("hidden_norm.weight", &[h])?, norm: tensor("norm.weight", &[h])?,
        markov_w1: tensor("markov_head.markov_w1.weight", &[cfg.vocab, cfg.rank])?,
        confidence_w: tensor("confidence_head.proj.weight", &[1, h + cfg.rank])?,
        confidence_b: tensor("confidence_head.proj.bias", &[1])?,
    })
}

/// Final device owners the loader binds, in order, from checkpoint headers:
/// one request per tensor, row-concatenated BF16 matrix, or FP8 payload/scales
/// plane pair, plus the Markov norms and the RoPE table. A runtime arena and
/// admission both size from exactly this list.
pub fn block_dspark_arena_requests(snapshot: &Path,
    representation: crate::families::glm5::draft_representation::GlmDraftRepresentation)
    -> anyhow::Result<Vec<(String, usize)>> {
    use crate::families::glm5::draft_representation::GlmDraftRepresentation;
    use anyhow::{ensure, Context};
    let cfg = BlockDsparkConfig::read(snapshot)?;
    let headers = crate::read_safetensors_metadata(&snapshot.join("model.safetensors"))?
        .into_iter().map(|t| (t.name.clone(), t)).collect::<std::collections::HashMap<_, _>>();
    let bf16 = |name: &str, shape: &[usize]| -> anyhow::Result<usize> {
        let tensor = headers.get(name).with_context(|| format!("drafter checkpoint has no {name}"))?;
        ensure!(tensor.dtype == cuteafd_core::DType::Bf16, "{name}: drafter weights must be BF16, found {:?}", tensor.dtype);
        let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
            .with_context(|| format!("{name}: BF16 tensor byte count overflow"))?;
        ensure!(tensor.shape == shape && tensor.byte_length == bytes as u64,
            "{name}: shape {:?}/{} bytes, expected BF16 {shape:?}/{bytes} bytes", tensor.shape, tensor.byte_length);
        Ok(bytes)
    };
    let requests = std::cell::RefCell::new(Vec::new());
    block_dspark_weight_recipe(&cfg, |name, shape| {
        requests.borrow_mut().push((name.to_owned(), bf16(name, shape)?.max(256)));
        Ok::<_, anyhow::Error>(())
    }, |parts, cols| {
        let bytes = parts.iter().try_fold(0usize, |n, (name, rows)| -> anyhow::Result<usize> {
            n.checked_add(bf16(name, &[*rows, cols])?).context("block dSpark matrix extent overflow")
        })?;
        let name = parts.iter().map(|(name, _)| *name).collect::<Vec<_>>().join("+");
        if representation == GlmDraftRepresentation::Bf16Only {
            requests.borrow_mut().push((name, bytes.max(256)));
        } else {
            let rows = parts.iter().try_fold(0usize, |n, (_, rows)| n.checked_add(*rows))
                .context("block dSpark matrix rows overflow")?;
            ensure!(rows > 0 && rows % 16 == 0 && cols > 0 && cols % 128 == 0,
                "block dSpark FP8 shape {rows}/{cols}");
            let payload = bytes / 2;
            requests.borrow_mut().extend([(format!("{name}.payload"), payload),
                (format!("{name}.scales"), payload / 128 * 4)]);
        }
        Ok(())
    })?;
    let mut requests = requests.into_inner();
    requests.push(("markov_norms".into(), cfg.vocab.checked_mul(4).context("block dSpark norms extent overflow")?));
    if cfg.head_dim != 64 {
        let plan = cfg.rope.plan(cfg.head_dim)?;
        requests.push(("inv_freq".into(), std::mem::size_of_val(plan.inv_freq.as_slice()).max(256)));
    }
    Ok(requests)
}
