//! Shared Hy4 family primitives (`hy4` and `hy4_flash`).
//!
//! Holds the genuinely identical building blocks shared across the Hy4 family
//! (576-dim absorbed MLA KV cache record, DFly2 drafter & ViT vision config
//! parsing, core projection split classification, packed-arena matrix packing,
//! and routed-expert storage detection). `hy4` uses these directly; `hy4_flash`
//! will migrate onto them once its in-flight branches (#2618, #2620,
//! `work/hy4f-dfly2`) merge to `dev`.
use std::collections::BTreeMap;

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use crate::formats::dense_trellis::TrellisSide;
use crate::formats::native_attention::{Fp8Role, NativeLayout};
use crate::formats::trellis_core::{CoreProjection, CoreSplit, CoreWeightLayout};
use crate::plan::checkpoint::Checkpoint;

pub const HEAD_STEM: &str = "lm_head";
pub const HEAD_WEIGHT: &str = "lm_head.weight";
pub const SPARSE_ATTENTION: &str = "deepseek_sparse_attention";
pub const PAGE_ROWS: usize = 64;

/// Paged KV cache precision for Hy4 / Hy4-flash absorbed MLA layers
/// (576-dim record: 512 normalized latent + 64 interleaved-RoPE key).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hy4KvCache {
    #[default]
    Bf16,
    Fp8,
}

impl Hy4KvCache {
    pub const fn record_bytes(self) -> usize {
        match self {
            // 576 BF16 elements (512 latent + 64 RoPE).
            Self::Bf16 => 1152,
            // 512 E4M3 latent + 4x F32 block-128 scales (16 B) + 64 BF16 RoPE (128 B).
            Self::Fp8 => 656,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        }
    }
}

impl std::str::FromStr for Hy4KvCache {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => Err("Hy4 KV format must be bf16 or fp8".into()),
        }
    }
}

/// Routed-expert storage representation detected from checkpoint headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Hy4ExpertStorage {
    /// Per-expert EXL3 trellis (`experts.{e}.{gate_proj,up_proj,down_proj}.{trellis,suh,svh,mul1|mcg}`).
    Exl3 { bits: usize, mul1: bool },
    /// Per-expert plain BF16 (`experts.{e}.{gate_proj,up_proj,down_proj}.weight`).
    Bf16PerExpert,
    /// Fused official BF16 (`experts.gate_up_proj` `[E, 2*I, H]` + `experts.down_proj` `[E, H, I]`).
    Bf16Fused,
}

/// Bundled 5-layer DFly2 drafter config (`dfly_*` top-level fields).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hy4DflyConfig {
    pub layers: usize,
    pub target_kv_layers: Vec<usize>,
    pub target_index_layers: Vec<usize>,
    pub target_index_topk: usize,
    pub mask_token_id: u32,
    pub enable_hidden_correction: bool,
    pub enable_ihc: bool,
    pub attn_causal: bool,
    pub block_size: Option<usize>,
    pub mlp_type: String,
    pub mtp_fusion: String,
    pub num_speculative_tokens: usize,
}

/// Bundled ViT vision config (`vision_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Hy4VisionConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub layers: usize,
    pub out_hidden: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub rms_norm_eps: f64,
}

pub(crate) fn json_int(value: &Value, family: &str, key: &str) -> Result<usize> {
    let n = value[key]
        .as_u64()
        .with_context(|| format!("{family} config lacks {key}"))?;
    ensure!(n > 0 && n <= (1 << 24), "invalid {family} {key}: {n}");
    Ok(n as usize)
}

pub(crate) fn json_float(value: &Value, family: &str, key: &str) -> Result<f64> {
    let x = value[key]
        .as_f64()
        .filter(|x| x.is_finite())
        .with_context(|| format!("{family} config lacks {key}"))?;
    ensure!(x > 0.0, "{family} {key} must be positive, found {x}");
    Ok(x)
}

pub(crate) fn json_bool(value: &Value, family: &str, key: &str, expected: bool) -> Result<()> {
    ensure!(
        value[key].as_bool() == Some(expected),
        "{family} {key} must be {expected}, found {}",
        value[key]
    );
    Ok(())
}

pub(crate) fn json_token_id(value: &Value, family: &str, key: &str, vocab: usize) -> Result<u32> {
    let id = value[key]
        .as_u64()
        .with_context(|| format!("{family} config lacks {key}"))?;
    ensure!(id < vocab as u64, "{family} {key} {id} is outside vocabulary {vocab}");
    Ok(u32::try_from(id)?)
}

pub(crate) fn json_token_ids(value: &Value, family: &str, vocab: usize, key: &str) -> Result<Vec<u32>> {
    let one = |id: &Value| -> Result<u32> {
        let id = id
            .as_u64()
            .with_context(|| format!("{family} {key} must be a token id"))?;
        ensure!(id < vocab as u64, "{family} {key} {id} is outside vocabulary {vocab}");
        Ok(u32::try_from(id)?)
    };
    match value {
        Value::Array(ids) if !ids.is_empty() => ids.iter().map(one).collect(),
        Value::Array(_) => bail!("{family} {key} must not be empty"),
        other => Ok(vec![one(other)?]),
    }
}

impl Hy4DflyConfig {
    pub fn from_hf(root: &Value, family: &str, target_layers: usize, vocab_size: usize) -> Result<Option<Self>> {
        let Some(layers) = root.get("dfly_num_hidden_layers").and_then(Value::as_u64) else {
            return Ok(None);
        };
        let layers = usize::try_from(layers)?;
        ensure!(layers > 0 && layers <= 16, "invalid {family} dfly_num_hidden_layers {layers}");
        let parse_layers = |key: &str| -> Result<Vec<usize>> {
            let arr = root[key]
                .as_array()
                .with_context(|| format!("{family} lacks {key}"))?;
            ensure!(arr.len() == layers, "{family} {key} length mismatch");
            arr.iter()
                .map(|entry| {
                    let id = entry
                        .as_u64()
                        .filter(|&id| (id as usize) < target_layers)
                        .with_context(|| format!("invalid {family} {key} entry {entry}"))?;
                    Ok(id as usize)
                })
                .collect()
        };
        let target_kv_layers = parse_layers("dfly_target_kv_layers")?;
        let target_index_layers = parse_layers("dfly_target_index_layer")?;
        let target_index_topk = json_int(root, family, "dfly_target_index_topk")?;
        let mask_token_id = json_token_id(root, family, "dfly_mask_token_id", vocab_size)?;
        let enable_hidden_correction = root["dfly_enable_hidden_correction"]
            .as_bool()
            .with_context(|| format!("{family} lacks dfly_enable_hidden_correction"))?;
        let enable_ihc = root
            .get("dfly_enable_ihc")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let attn_causal = root
            .get("dfly_attn_causal")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let block_size = root
            .get("dfly_block_size")
            .and_then(Value::as_u64)
            .map(|n| n as usize);
        let mlp_type = root
            .get("dfly_mlp_type")
            .and_then(Value::as_str)
            .unwrap_or("dense")
            .to_owned();
        let mtp_fusion = root["dfly_mtp_fusion"]
            .as_str()
            .with_context(|| format!("{family} lacks dfly_mtp_fusion"))?
            .to_owned();
        let num_speculative_tokens = root
            .get("speculative_config")
            .and_then(|s| s.get("num_speculative_tokens"))
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .or(block_size)
            .unwrap_or(layers);
        Ok(Some(Self {
            layers,
            target_kv_layers,
            target_index_layers,
            target_index_topk,
            mask_token_id,
            enable_hidden_correction,
            enable_ihc,
            attn_causal,
            block_size,
            mlp_type,
            mtp_fusion,
            num_speculative_tokens,
        }))
    }
}

impl Hy4VisionConfig {
    pub fn from_hf(root: &Value, family: &str) -> Result<Option<Self>> {
        let Some(v) = root.get("vision_config").filter(|v| v.is_object()) else {
            return Ok(None);
        };
        Ok(Some(Self {
            hidden: json_int(v, family, "hidden_size")?,
            intermediate: json_int(v, family, "intermediate_size")?,
            heads: json_int(v, family, "num_attention_heads")?,
            layers: json_int(v, family, "num_hidden_layers")?,
            out_hidden: json_int(v, family, "out_hidden_size")?,
            patch_size: json_int(v, family, "patch_size")?,
            spatial_merge_size: json_int(v, family, "spatial_merge_size")?,
            rms_norm_eps: json_float(v, family, "rms_norm_eps")?,
        }))
    }
}

/// Detect routed-expert storage at `layer` from checkpoint safetensors headers.
pub fn detect_expert_storage(checkpoint: &Checkpoint, family: &str, layer: usize) -> Result<Hy4ExpertStorage> {
    let trellis_name = format!("model.layers.{layer}.mlp.experts.0.gate_proj.trellis");
    if let Some(header) = checkpoint.header(&trellis_name) {
        let bits = header
            .shape
            .get(2)
            .map(|rows| rows / 16)
            .filter(|bits| (1..=8).contains(bits))
            .ok_or_else(|| {
                anyhow::anyhow!("{trellis_name}: EXL3 trellis shape {:?} encodes no width", header.shape)
            })?;
        let mul1 = checkpoint.contains_tensor(&format!("model.layers.{layer}.mlp.experts.0.gate_proj.mul1"));
        return Ok(Hy4ExpertStorage::Exl3 { bits, mul1 });
    }
    let fused_name = format!("model.layers.{layer}.mlp.experts.gate_up_proj");
    if let Some(header) = checkpoint.header(&fused_name) {
        ensure!(
            header.dtype == cuteafd_core::DType::Bf16,
            "{fused_name}: unsupported {family} fused routed expert dtype {:?}",
            header.dtype
        );
        return Ok(Hy4ExpertStorage::Bf16Fused);
    }
    let per_expert_name = format!("model.layers.{layer}.mlp.experts.0.gate_proj.weight");
    if let Some(header) = checkpoint.header(&per_expert_name) {
        ensure!(
            header.dtype == cuteafd_core::DType::Bf16,
            "{per_expert_name}: unsupported {family} routed expert dtype {:?}",
            header.dtype
        );
        return Ok(Hy4ExpertStorage::Bf16PerExpert);
    }
    bail!(
        "{family} checkpoint lacks routed expert headers at layer {layer} \
         ({trellis_name}, {fused_name}, or {per_expert_name})"
    )
}

/// Classify whether a Hy4 / Hy4-flash core projection splits across two
/// coordinator GPUs under `HeadSplit` / `ContextSplit`.
pub fn core_split(stem: &str) -> CoreSplit {
    if stem == HEAD_STEM {
        return CoreSplit::Output;
    }
    if let Some(leaf) = stem.rsplit_once('.').map(|(_, l)| l) {
        if stem.contains(".self_attn.") {
            return match leaf {
                "q_b_proj" | "kv_b_proj" | "linear_gate" => CoreSplit::Output,
                "o_proj" => CoreSplit::Input,
                _ => CoreSplit::Whole,
            };
        }
        if stem.contains(".mlp.shared_experts.")
            || (stem.contains(".mlp.")
                && !stem.contains(".experts.")
                && !stem.contains("routed_expert_")
                && !stem.ends_with(".mlp.gate")
                && !stem.contains(".mlp.gate."))
        {
            return match leaf {
                "gate_proj" | "up_proj" => CoreSplit::Output,
                "down_proj" => CoreSplit::Input,
                _ => CoreSplit::Whole,
            };
        }
    }
    CoreSplit::Whole
}

/// Add one linear matrix (EXL3 compact core, runtime-derived block-FP8, or BF16)
/// to a coordinator rank's packed target arena view list.
#[allow(clippy::too_many_arguments)]
pub fn add_arena_matrix(
    checkpoint: &Checkpoint,
    core_projections: &BTreeMap<String, CoreProjection>,
    fp8_role: fn(&str) -> Fp8Role,
    stem: &str,
    shape: [usize; 2],
    split: CoreSplit,
    rank: usize,
    ranks: usize,
    views: &mut Vec<(String, usize)>,
    core_map: &mut BTreeMap<String, CoreWeightLayout>,
    fp8_map: &mut BTreeMap<String, NativeLayout>,
) -> Result<()> {
    let effective_ranks = if split == CoreSplit::Whole { 1 } else { ranks };
    let effective_rank = if split == CoreSplit::Whole { 0 } else { rank };

    if let Some(proj) = core_projections.get(stem) {
        let side = if effective_ranks > 1 && split == CoreSplit::Input {
            TrellisSide::Input
        } else {
            TrellisSide::Output
        };
        let layout = if stem == HEAD_STEM && effective_ranks == 2 {
            CoreWeightLayout::output_shard(proj, effective_rank)?
        } else {
            CoreWeightLayout::new(proj, side, effective_rank, effective_ranks)?
        };
        views.extend(layout.arena_tensors(stem)?);
        core_map.insert(stem.to_owned(), layout);
        return Ok(());
    }

    let weight_name = format!("{stem}.weight");
    let [mut n, mut k] = shape;
    if effective_ranks == 2 {
        match split {
            CoreSplit::Output => {
                if stem == HEAD_STEM {
                    n = crate::shared::vocabulary::shard(n, 128, effective_rank)?.len();
                } else {
                    ensure!(n % 2 == 0, "{stem}: output dim {n} does not divide TP2");
                    n /= 2;
                }
            }
            CoreSplit::Input => {
                ensure!(k % 2 == 0, "{stem}: input dim {k} does not divide TP2");
                k /= 2;
            }
            CoreSplit::Whole => {}
        }
    }

    let role = fp8_role(&weight_name);
    if checkpoint.converts_to_fp8(&weight_name) && role.supported {
        let [stored_n, stored_k] = role.stored_shape([n, k])?;
        let payload = stored_n * stored_k;
        let scales = ((stored_n / 128) * (stored_k / 128) * 4).max(256);
        views.push((format!("{stem}.payload"), payload));
        views.push((format!("{stem}.scales"), scales));
        fp8_map.insert(
            stem.to_owned(),
            NativeLayout {
                stem: stem.to_owned(),
                shape: [stored_n, stored_k],
                fp8: true,
                rank: effective_rank,
                ranks: effective_ranks,
                split,
            },
        );
        return Ok(());
    }

    views.push((
        weight_name,
        n.checked_mul(k)
            .and_then(|v| v.checked_mul(2))
            .with_context(|| format!("{stem}: BF16 weight size overflow"))?,
    ));
    Ok(())
}
