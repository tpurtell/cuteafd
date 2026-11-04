//! Checkpoint-native geometry for Qwen3-backbone block dSpark drafters.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockDraftLayout {
    pub owners: Vec<super::dflash::MlaDraftOwner>,
    pub packing_sources: Vec<super::dflash::MlaDraftOwner>,
    pub cache: super::dflash::DraftCacheLayout,
    pub sync_upload_bytes: usize,
    pub loading_host_bytes: u64,
}

impl BlockDraftLayout {
    /// Cache row indices are signed 32-bit; capacity contains complete 64-row pages.
    pub const MAX_POOL_TOKENS: usize = i32::MAX as usize / 64 * 64;

    #[allow(clippy::too_many_arguments)]
    pub fn read(snapshot: &Path, slots: usize, sequences: usize, context: usize, tokens: usize,
        sms: usize, representation: crate::families::glm5::draft_representation::GlmDraftRepresentation,
        mode: u8, kv: crate::families::qwen4::Qwen4KvCache) -> Result<Self, DsparkConfigError> {
        let cfg = BlockDsparkConfig::read(snapshot)?;
        let mut layout = cfg.runtime_layout(slots, sequences, context, tokens, sms, representation, mode, kv)?;
        let headers = crate::read_safetensors_metadata(&snapshot.join("model.safetensors"))
            .map_err(|error| DsparkConfigError::Unsupported(format!("dSpark headers: {error}")))?;
        layout.loading_host_bytes = Self::loading_host_bytes(&headers, cfg.hidden * 2)?;
        Ok(layout)
    }

    pub fn device_bytes(&self) -> Result<u64, DsparkConfigError> {
        self.owners.iter().filter(|o| o.domain == super::dflash::MlaDraftDomain::Device && !o.on_use)
            .try_fold(0u64, |sum, o| sum.checked_add(o.bytes as u64)
                .ok_or_else(|| DsparkConfigError::Unsupported("dSpark owner sum overflow".into())))
    }

    pub fn resident_weights(&self) -> Result<crate::placement::ResidentWeights, DsparkConfigError> {
        self.owners.iter().try_fold(crate::placement::ResidentWeights::default(), |sum, owner|
            sum.checked_add(owner.resident_weights)
                .map_err(|_| DsparkConfigError::Unsupported("dSpark resident-weight sum overflow".into())))
    }

    pub fn scratch_bytes(&self) -> Result<u64, DsparkConfigError> {
        self.owners.iter().filter(|o| matches!(o.id.as_str(), "fp8_scratch" | "attention_workspace"))
            .try_fold(0u64, |sum, o| sum.checked_add(o.bytes as u64)
                .ok_or_else(|| DsparkConfigError::Unsupported("dSpark scratch sum overflow".into())))
    }

    pub fn page_bytes(&self) -> Result<u64, DsparkConfigError> {
        (self.cache.record_bytes as u64).checked_mul(self.cache.page_rows as u64)
            .and_then(|n| n.checked_mul(self.cache.full_layers as u64))
            .ok_or_else(|| DsparkConfigError::Unsupported("dSpark page extent overflow".into()))
    }

    pub fn with_pool_tokens(&self, tokens: usize) -> Result<Self, DsparkConfigError> {
        let invalid = || DsparkConfigError::Unsupported("invalid dSpark pool extent".into());
        let rows = tokens.div_ceil(64).checked_mul(64).ok_or_else(invalid)?;
        if tokens == 0 || rows > Self::MAX_POOL_TOKENS { return Err(invalid()); }
        let mut next = self.clone();
        next.cache.target_tokens = tokens;
        next.cache.pages = rows / 64;
        if next.cache.full_layers > 0 {
            let bytes = rows.checked_mul(next.cache.record_bytes).ok_or_else(invalid)?;
            next.cache.full_bytes = (bytes as u64).checked_mul(next.cache.full_layers as u64).ok_or_else(invalid)?;
            next.cache.layer_bytes.fill(bytes);
            for owner in &mut next.owners {
                if owner.id.ends_with(".k_ring") || owner.id.ends_with(".v_ring") {
                    owner.bytes = (bytes / 2).max(256);
                    owner.shape = vec![owner.bytes];
                }
            }
        }
        Ok(next)
    }

    /// Prefetch retains every payload except the two borrowed target tables.
    pub fn loading_host_bytes(headers: &[crate::SafetensorsTensorMetadata], mask_bytes: usize)
        -> Result<u64, DsparkConfigError> {
        headers.iter().filter(|t| !matches!(t.name.as_str(), "embed_tokens.weight" | "lm_head.weight"))
            .try_fold(mask_bytes as u64, |sum, t| sum.checked_add(t.byte_length)
                .ok_or_else(|| DsparkConfigError::Unsupported("dSpark prefetch heap overflow".into())))
    }
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
        if rows > BlockDraftLayout::MAX_POOL_TOKENS { return Err(invalid()); }
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

    #[allow(clippy::too_many_arguments)]
    pub fn runtime_layout(&self, slots: usize, sequences: usize, context: usize, tokens: usize,
        sms: usize, representation: crate::families::glm5::draft_representation::GlmDraftRepresentation,
        mode: u8, kv_cache: crate::families::qwen4::Qwen4KvCache) -> Result<BlockDraftLayout, DsparkConfigError> {
        use super::dflash::{MlaDraftDomain as Domain, MlaDraftDtype as Dtype, MlaDraftLifetime, MlaDraftOwner};
        use crate::families::glm5::draft_representation::{draft_fp8_scratch_bytes, GlmDraftRepresentation};
        let invalid = || DsparkConfigError::Unsupported("invalid dSpark runtime geometry/capacity".into());
        if !supports_sequences(sequences) || sequences > slots || sms == 0 || sms > 4096 || mode > 2
            || [self.hidden, self.intermediate, self.vocab, self.block, self.taps.len()].iter().any(|&n| n == 0 || n > 1_000_000)
            || self.layers == 0 || self.layers > 1024 || self.heads == 0 || self.heads > 4096
            || self.kv_heads == 0 || self.kv_heads > 4096 || self.heads % self.kv_heads != 0
            || !matches!(self.head_dim, 64 | 128) || self.rank != 256 || self.block > 128
            || self.taps.len() > 1024 || slots.checked_mul(context).is_none_or(|n| n > i32::MAX as usize) {
            return Err(invalid());
        }
        if self.head_dim == 64 && !matches!(self.rope, DsparkRope::Default { .. }) {
            return Err(DsparkConfigError::Unsupported("64-wide dSpark requires default RoPE".into()));
        }
        self.rope.plan(self.head_dim)?;
        let cache = self.paged_cache_layout(slots, context, tokens, kv_cache)?;
        let (h, inter, q, kv) = (self.hidden, self.intermediate, self.heads * self.head_dim, self.kv_heads * self.head_dim);
        let rows = sequences * self.block;
        let fp8 = representation == GlmDraftRepresentation::Fp8Only;
        let mut owners = Vec::new();
        let mut packing_sources = Vec::new();
        let mut sync_upload_bytes = 0usize;
        let mut add_weight = |id: String, shape: Vec<usize>, matrix: bool, upload: Option<usize>| -> Result<(), DsparkConfigError> {
            let source = MlaDraftOwner::new(format!("{id}.source"), Domain::Device, Dtype::Bf16, shape.clone(), 256)?;
            let payload = shape.iter().product::<usize>() * 2;
            sync_upload_bytes = sync_upload_bytes.max(upload.unwrap_or(payload));
            if matrix && fp8 {
                if shape.len() != 2 || shape[0] % 16 != 0 || shape[1] % 128 != 0 { return Err(invalid()); }
                owners.push(MlaDraftOwner::new(format!("{id}.values"), Domain::Device, Dtype::Bytes, vec![payload / 2], 0)?.weight());
                owners.push(MlaDraftOwner::new(format!("{id}.scales"), Domain::Device, Dtype::F32, vec![shape[0], shape[1] / 128], 0)?.weight());
                packing_sources.push(MlaDraftOwner { dtype: Dtype::Bytes, shape: vec![source.bytes], ..source });
            } else { owners.push(MlaDraftOwner { id, ..source }.weight()); }
            Ok(())
        };
        for layer in 0..self.layers {
            let p = format!("weight.layer.{layer}");
            for leaf in ["input_norm", "post_norm"] { add_weight(format!("{p}.{leaf}"), vec![h], false, None)?; }
            add_weight(format!("{p}.qkv"), vec![q + 2 * kv, h], true, Some(q.max(kv) * h * 2))?;
            for leaf in ["q_norm", "k_norm"] { add_weight(format!("{p}.{leaf}"), vec![self.head_dim], false, None)?; }
            add_weight(format!("{p}.o"), vec![h, q], true, None)?;
            add_weight(format!("{p}.gate_up"), vec![2 * inter, h], true, Some(inter * h * 2))?;
            add_weight(format!("{p}.down"), vec![h, inter], true, None)?;
        }
        add_weight("weight.markov_w2".into(), vec![self.vocab, self.rank], false, None)?;
        add_weight("weight.fc".into(), vec![h, self.taps.len() * h], true, None)?;
        for leaf in ["hidden_norm", "norm"] { add_weight(format!("weight.{leaf}"), vec![h], false, None)?; }
        add_weight("weight.markov_w1".into(), vec![self.vocab, self.rank], false, None)?;
        add_weight("weight.confidence_w".into(), vec![1, h + self.rank], false, None)?;
        add_weight("weight.confidence_b".into(), vec![1], false, None)?;
        let mut add = |id: String, dtype, shape: Vec<usize>, minimum| -> Result<(), DsparkConfigError> {
            owners.push(MlaDraftOwner::new(id, Domain::Device, dtype, shape, minimum)?); Ok(())
        };
        for layer in 0..self.layers {
            for leaf in ["k_ring", "v_ring"] {
                add(format!("layer.{layer}.{leaf}"), Dtype::Bytes, vec![cache.layer_bytes[layer] / 2], 256)?;
            }
        }
        add("markov_norms".into(), Dtype::F32, vec![self.vocab], 0)?;
        if self.head_dim == 128 {
            add("inverse_frequencies".into(), Dtype::F32, vec![self.head_dim / 2], 256)?;
            sync_upload_bytes = sync_upload_bytes.max(self.head_dim * 2);
        }
        if self.window.is_none() {
            add("page_table".into(), Dtype::I32, vec![slots, context.div_ceil(64)], 256)?;
        }
        for (id, dtype, shape) in [
            ("taps", Dtype::Bf16, vec![2048, self.taps.len(), h]),
            ("fused", Dtype::Bf16, vec![2048, h]), ("fused_norm", Dtype::Bf16, vec![2048, h]),
            ("context_kv", Dtype::Bf16, vec![2048, 2 * kv]),
            ("context_positions", Dtype::I64, vec![2048]), ("context_slots", Dtype::I32, vec![2048]),
            ("head_workspace", Dtype::Bytes, vec![4 << 20]),
            ("h", Dtype::Bf16, vec![rows, h]), ("n", Dtype::Bf16, vec![rows, h]),
            ("qkv", Dtype::Bf16, vec![rows, q + 2 * kv]), ("q", Dtype::Bf16, vec![rows, q]),
            ("k", Dtype::Bf16, vec![rows, kv]), ("v", Dtype::Bf16, vec![rows, kv]),
            ("attn", Dtype::Bf16, vec![rows, q]), ("delta", Dtype::Bf16, vec![rows, h]),
            ("gate_up", Dtype::Bf16, vec![rows, 2 * inter]), ("act", Dtype::Bf16, vec![rows, inter]),
            ("logits", Dtype::F32, vec![rows, self.vocab]), ("anchors", Dtype::U32, vec![sequences]),
            ("ids", Dtype::U32, vec![rows]), ("tokens", Dtype::U32, vec![rows]),
            ("confidence", Dtype::F32, vec![rows]), ("positions", Dtype::I64, vec![rows]),
            ("tables", Dtype::I32, vec![3, sequences]),
        ] { add(id.into(), dtype, shape, 256)?; }
        let attention = crate::serving_capacity::draft_attention_workspace_bytes(sequences as u64, self.kv_heads as u64,
            (self.heads / self.kv_heads * self.block).div_ceil(32) as u64, (context + self.block) as u64,
            self.head_dim as u64, sms as u64).map_err(|_| invalid())?;
        add("attention_workspace".into(), Dtype::Bytes, vec![usize::try_from(attention).map_err(|_| invalid())?.max(256)], 256)?;
        add("markov_workspace".into(), Dtype::Bytes, vec![(sequences * 296 * 8 + rows * 4).max(256)], 256)?;
        if fp8 {
            let scratch = [(self.taps.len() * h, h), (h, q + 2 * kv), (h, 2 * kv), (q, h), (h, 2 * inter), (inter, h)]
                .iter().map(|&(k, n)| draft_fp8_scratch_bytes(rows.max(2048) as u64, k as u64, n as u64, sms as u64, mode))
                .max().unwrap_or(256).max(256);
            add("fp8_scratch".into(), Dtype::Bytes, vec![usize::try_from(scratch).map_err(|_| invalid())?], 256)?;
        }
        owners.push(MlaDraftOwner::new("landing", Domain::HostPinned, Dtype::Bytes, vec![rows * 8], 0)?);
        for destination in owners.iter().filter(|o| o.domain == Domain::Device &&
            matches!(o.id.as_str(), "context_positions" | "context_slots" | "page_table" | "positions" | "tables" | "anchors" | "taps" | "h"))
            .cloned().collect::<Vec<_>>() {
            let mut upload = MlaDraftOwner::new(format!("{}.upload", destination.id), Domain::HostPinned,
                Dtype::Bytes, vec![destination.bytes], 0)?;
            upload.lifetime = MlaDraftLifetime::Library;
            upload.staging_for = Some(destination.id.clone());
            upload.on_use = matches!(destination.id.as_str(), "taps" | "h");
            owners.push(upload);
        }
        Ok(BlockDraftLayout { owners, packing_sources, cache, sync_upload_bytes, loading_host_bytes: 0 })
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
        let t: Layers = serde_json::from_value(if speculators {
            raw["transformer_layer_config"].clone()
        } else {
            raw.clone()
        })?;
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
    fn named_block_owners_preserve_frozen_totals_and_exact_sources() {
        use crate::families::{glm5::draft_representation::GlmDraftRepresentation as Rep, qwen4::Qwen4KvCache as Kv};
        for (raw, old_bf16, old_fp8, old_bf16_compact, old_fp8_compact, scratch, fp8_scratch, delta) in [
            (glm_flash_speculators(), 6_168_387_072u64, 5_014_429_184, 4_595_523_072, 3_441_565_184,
                77_856_768u64, 109_462_528, 1280u64),
            (qwen(), 3_843_803_264, 2_650_228_864, 3_437_480_064, 2_243_905_664,
                102_236_160, 135_158_784, 0),
        ] {
            let cfg = BlockDsparkConfig::from_json(&raw).unwrap();
            for (rep, kv, before, scratch) in [
                (Rep::Bf16Only, Kv::Bf16, old_bf16, scratch),
                (Rep::Fp8Only, Kv::Bf16, old_fp8, fp8_scratch),
                (Rep::Bf16Only, Kv::Fp8, old_bf16_compact, scratch),
                (Rep::Fp8Only, Kv::Fp8, old_fp8_compact, fp8_scratch),
            ] {
                let layout = cfg.runtime_layout(20, 16, 2048, 20 * 2048, 188, rep, 2, kv).unwrap();
                assert_eq!(layout.scratch_bytes().unwrap(), scratch);
                // The old 64-wide norm aggregate missed actual min-256 padding:
                // two 128-byte norm payloads per layer, five layers, +1280 B.
                assert_eq!(layout.device_bytes().unwrap() - scratch, before + delta);
                let ids: std::collections::BTreeSet<_> = layout.owners.iter().map(|o| &o.id).collect();
                assert_eq!(ids.len(), layout.owners.len());
                assert!(!layout.owners.iter().any(|o| o.id.contains("embed_tokens") || o.id.contains("lm_head")));
                assert_eq!(layout.owners.iter().find(|o| o.id == "landing").unwrap().bytes, 16 * cfg.block * 8);
                assert_eq!(layout.owners.iter().filter(|o| o.staging_for.is_some() && !o.on_use).count(),
                    if cfg.window.is_none() { 6 } else { 5 });
                for id in ["taps.upload", "h.upload"] {
                    assert!(layout.owners.iter().find(|o| o.id == id).unwrap().on_use);
                }
                if rep == Rep::Fp8Only {
                    assert_eq!(layout.packing_sources.len(), cfg.layers * 4 + 1);
                    assert_eq!(layout.packing_sources[0].id, "weight.layer.0.qkv.source");
                    assert_eq!(layout.packing_sources[2].id, "weight.layer.0.gate_up.source");
                    assert_eq!(layout.packing_sources.last().unwrap().id, "weight.fc.source");
                    assert_eq!(layout.packing_sources[2].bytes, 2 * cfg.intermediate * cfg.hidden * 2);
                    assert!(layout.packing_sources.iter().all(|o| o.shape == [o.bytes]));
                } else { assert!(layout.packing_sources.is_empty()); }
                assert!(layout.with_pool_tokens(BlockDraftLayout::MAX_POOL_TOKENS).is_ok());
                assert!(layout.with_pool_tokens(BlockDraftLayout::MAX_POOL_TOKENS + 1).is_err());
                let larger = layout.with_pool_tokens(80 * 2048).unwrap();
                for (a, b) in layout.owners.iter().zip(&larger.owners) {
                    if cfg.window.is_none() && (a.id.ends_with(".k_ring") || a.id.ends_with(".v_ring")) {
                        assert_eq!(b.bytes, a.bytes * 4);
                        assert_eq!(b.shape, [b.bytes]);
                    } else { assert_eq!(a, b); }
                }
            }
        }
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
