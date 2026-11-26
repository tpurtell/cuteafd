//! DeepSeek V4.1 external DFly2 drafter (`DFlyDeepseekV41DraftModel`) config,
//! safetensors catalog, packed dense arena layout, and MXFP4 routed-expert
//! adapter for the shared `OfficialV41Catalog` expert loader.

use crate::serving_capacity::WeightArenaLayout;
use crate::{
    read_safetensors_metadata, OfficialV41Catalog, OfficialV41Config, SafetensorsTensorMetadata,
    V41Tensor, V41TensorPlacement,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: String,
    dfly_config: RawDflyConfig,
    text_config: RawTextConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct RawDflyConfig {
    block_size: usize,
    #[serde(default)]
    causal_block: bool,
    target_kv_layers: Vec<usize>,
    target_index_layers: Vec<usize>,
    index_topk: usize,
    mask_token_id: u32,
    mtp_fusion: String,
    #[serde(default)]
    dense_dtype: String,
    hidden_correction: RawHiddenCorrection,
    attentive_correction: RawAttentiveCorrection,
}

#[derive(Debug, Clone, Deserialize)]
struct RawHiddenCorrection {
    intermediate_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct RawAttentiveCorrection {
    source_layer: i32,
    block_keys: bool,
    block_mode: String,
    block_kv_in: String,
    max_block_slots: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct RawTextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    head_dim: usize,
    qk_rope_head_dim: usize,
    q_lora_rank: usize,
    o_lora_rank: usize,
    o_groups: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    #[serde(default = "default_target_routed_experts")]
    n_routed_experts: usize,
    #[serde(default = "default_target_experts_per_tok")]
    num_experts_per_tok: usize,
    #[serde(default = "default_nextn_layers")]
    num_nextn_predict_layers: usize,
    #[serde(default = "default_moe_intermediate")]
    moe_intermediate_size: usize,
    #[serde(default = "default_hc_mult")]
    hc_mult: usize,
    #[serde(default = "default_hc_iters")]
    hc_sinkhorn_iters: usize,
    #[serde(default = "default_swiglu_limit")]
    swiglu_limit: f32,
}

fn default_target_routed_experts() -> usize { 384 }
fn default_target_experts_per_tok() -> usize { 6 }
fn default_nextn_layers() -> usize { 3 }
fn default_moe_intermediate() -> usize { 2304 }
fn default_hc_mult() -> usize { 4 }
fn default_hc_iters() -> usize { 20 }
fn default_swiglu_limit() -> f32 { 10.0 }

/// Validated DeepSeek V4.1 DFly2 drafter configuration.
#[derive(Debug, Clone)]
pub struct V41Dfly2Config {
    /// Trained block size in the checkpoint (6 = 1 anchor + 5 drafts).
    pub trained_block_size: usize,
    /// Maximum block slots supported by `block_rel_bias` / `block_gate_bias` (16).
    pub max_block_slots: usize,
    /// Target layers whose window rings and indexer picks feed the 3 draft layers.
    pub target_kv_layers: [usize; 3],
    pub target_index_layers: [usize; 3],
    pub index_topk: usize,
    pub mask_token_id: u32,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub num_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub hidden_correction_intermediate: usize,
    pub swiglu_limit: f32,
}

impl V41Dfly2Config {
    pub fn read(snapshot: &Path) -> Result<Self> {
        let path = snapshot.join("config.json");
        let raw: RawConfig = serde_json::from_slice(
            &std::fs::read(&path)
                .with_context(|| format!("reading DFly2 config {}", path.display()))?,
        )
        .with_context(|| format!("parsing DFly2 config {}", path.display()))?;
        ensure!(
            raw.architectures.iter().any(|a| a == "DFlyDeepseekV41DraftModel")
                || raw.model_type == "dfly_deepseek_v41",
            "unsupported V4.1 DFly2 architecture {:?} / model_type {:?}",
            raw.architectures,
            raw.model_type
        );
        let d = &raw.dfly_config;
        let t = &raw.text_config;
        ensure!(!d.causal_block, "V4.1 DFly2 requires bidirectional block attention (causal_block=false)");
        ensure!(d.mtp_fusion == "slot0", "V4.1 DFly2 requires mtp_fusion=slot0, got {}", d.mtp_fusion);
        ensure!(
            d.dense_dtype.is_empty() || d.dense_dtype == "bfloat16",
            "V4.1 DFly2 requires bfloat16 dense weights, got {}",
            d.dense_dtype
        );
        ensure!(
            t.hidden_size == 5120
                && t.num_hidden_layers == 40
                && t.num_nextn_predict_layers == 3
                && t.num_attention_heads == 64
                && t.head_dim == 512
                && t.qk_rope_head_dim == 64
                && t.q_lora_rank == 1280
                && t.o_lora_rank == 1024
                && t.o_groups == 8
                && t.vocab_size == 129280
                && t.n_routed_experts == 384
                && t.num_experts_per_tok == 6
                && t.moe_intermediate_size == 2304
                && t.hc_mult == 4
                && t.hc_sinkhorn_iters == 20,
            "V4.1 DFly2 text_config geometry mismatch"
        );
        ensure!(
            (t.rms_norm_eps - 1e-20).abs() <= 1e-24,
            "V4.1 DFly2 requires rms_norm_eps=1e-20, got {}",
            t.rms_norm_eps
        );
        ensure!(
            d.target_kv_layers == [28, 32, 36] && d.target_index_layers == [28, 32, 36],
            "V4.1 DFly2 requires target_kv_layers/target_index_layers=[28,32,36], got {:?}/{:?}",
            d.target_kv_layers,
            d.target_index_layers
        );
        ensure!(d.index_topk == 512, "V4.1 DFly2 requires index_topk=512, got {}", d.index_topk);
        ensure!(
            d.hidden_correction.intermediate_size == 5120,
            "V4.1 DFly2 requires hidden_correction.intermediate_size=5120"
        );
        let ac = &d.attentive_correction;
        ensure!(
            ac.source_layer == -1
                && ac.block_keys
                && ac.block_mode == "gated"
                && ac.block_kv_in == "hidden_embed"
                && ac.max_block_slots == 16,
            "V4.1 DFly2 attentive_correction contract mismatch: {ac:?}"
        );
        ensure!(
            (2..=ac.max_block_slots).contains(&d.block_size),
            "V4.1 DFly2 invalid block_size {}",
            d.block_size
        );
        ensure!(
            (d.mask_token_id as usize) < t.vocab_size,
            "V4.1 DFly2 mask_token_id {} out of vocab range",
            d.mask_token_id
        );
        Ok(Self {
            trained_block_size: d.block_size,
            max_block_slots: ac.max_block_slots,
            target_kv_layers: [28, 32, 36],
            target_index_layers: [28, 32, 36],
            index_topk: d.index_topk,
            mask_token_id: d.mask_token_id,
            hidden_size: t.hidden_size,
            num_layers: d.target_kv_layers.len(),
            vocab_size: t.vocab_size,
            num_routed_experts: 128,
            num_experts_per_tok: 3,
            moe_intermediate_size: t.moe_intermediate_size,
            hidden_correction_intermediate: d.hidden_correction.intermediate_size,
            swiglu_limit: t.swiglu_limit,
        })
    }

    pub fn validate_target(&self, target: &OfficialV41Config) -> Result<()> {
        let t = target.text();
        ensure!(t.vocab_size == self.vocab_size, "DFly2 vocab {} != target {}", self.vocab_size, t.vocab_size);
        ensure!(t.hidden_size == self.hidden_size, "DFly2 hidden {} != target {}", self.hidden_size, t.hidden_size);
        ensure!(t.num_hidden_layers >= 40, "DFly2 requires >= 40 target layers");
        for &layer in &self.target_index_layers {
            ensure!(
                t.index_source_layer_ids.contains(&layer),
                "target missing DFly2 index layer {layer}"
            );
        }
        ensure!(
            t.compress_ratios.get(20).copied() == Some(1),
            "DFly2 requires target source layer 20 compress_ratio=1"
        );
        Ok(())
    }

    /// Resolve an optional block-size override (`--dfly2-block-size`).
    /// Currently V4.1 verify budgets support up to 7 drafts (block size 8).
    pub fn resolve_block_size(&self, requested: Option<usize>) -> Result<usize> {
        let block = requested.unwrap_or(self.trained_block_size);
        ensure!(
            matches!(block, 6 | 8) && block <= self.max_block_slots,
            "V4.1 DFly2 block_size must be 6 (trained) or 8 (overdrive), got {block}"
        );
        Ok(block)
    }
}

#[derive(Debug, Clone)]
pub struct V41Dfly2Tensor {
    pub shard: String,
    pub metadata: SafetensorsTensorMetadata,
}

#[derive(Debug, Clone)]
pub struct V41Dfly2Catalog {
    snapshot: PathBuf,
    config: V41Dfly2Config,
    dense: BTreeMap<String, V41Dfly2Tensor>,
    experts: Vec<V41Tensor>,
}

#[derive(Deserialize)]
struct IndexFile {
    weight_map: BTreeMap<String, String>,
}

/// Every required dense (non-routed-expert) tensor name and its expected (dtype, shape).
fn expected_dense_specs() -> Vec<(String, DType, Vec<usize>)> {
    let mut specs = vec![
        ("model.final_norm.weight".into(), DType::Bf16, vec![5120]),
        ("model.mtp_fusion.enorm.weight".into(), DType::Bf16, vec![5120]),
        ("model.mtp_fusion.hnorm.weight".into(), DType::Bf16, vec![5120]),
        ("model.mtp_fusion.eh_proj.weight".into(), DType::Bf16, vec![5120, 10240]),
        ("model.mtp_fusion.gate".into(), DType::Bf16, vec![5120]),
        ("model.hidden_correction.hidden_norm.weight".into(), DType::Bf16, vec![5120]),
        ("model.hidden_correction.embed_norm.weight".into(), DType::Bf16, vec![5120]),
        ("model.hidden_correction.gate_proj.weight".into(), DType::Bf16, vec![5120, 10240]),
        ("model.hidden_correction.up_proj.weight".into(), DType::Bf16, vec![5120, 10240]),
        ("model.hidden_correction.down_proj.weight".into(), DType::Bf16, vec![5120, 5120]),
        ("model.hidden_correction.attention.q_in.weight".into(), DType::Bf16, vec![1280, 10240]),
        ("model.hidden_correction.attention.q_norm.weight".into(), DType::Bf16, vec![1280]),
        ("model.hidden_correction.attention.wq_b.weight".into(), DType::Bf16, vec![32768, 1280]),
        ("model.hidden_correction.attention.kv_in.weight".into(), DType::Bf16, vec![512, 10240]),
        ("model.hidden_correction.attention.kv_norm.weight".into(), DType::Bf16, vec![512]),
        ("model.hidden_correction.attention.wo_a.weight".into(), DType::Bf16, vec![8192, 4096]),
        ("model.hidden_correction.attention.wo_b.weight".into(), DType::Bf16, vec![5120, 8192]),
        ("model.hidden_correction.attention.attn_sink".into(), DType::F32, vec![64]),
        ("model.hidden_correction.attention.block_sink".into(), DType::F32, vec![64]),
        ("model.hidden_correction.attention.block_rel_bias".into(), DType::F32, vec![64, 16]),
        ("model.hidden_correction.attention.block_gate.weight".into(), DType::Bf16, vec![64, 10240]),
        ("model.hidden_correction.attention.block_gate_bias".into(), DType::F32, vec![64, 16]),
    ];
    for layer in 0..3 {
        let p = format!("model.layers.{layer}");
        for kind in ["attn", "ffn"] {
            specs.push((format!("{p}.hc_{kind}_base"), DType::F32, vec![24]));
            specs.push((format!("{p}.hc_{kind}_fn"), DType::F32, vec![24, 20480]));
            specs.push((format!("{p}.hc_{kind}_scale"), DType::F32, vec![3]));
            specs.push((format!("{p}.{kind}_norm.weight"), DType::Bf16, vec![5120]));
        }
        specs.extend([
            (format!("{p}.attn.wq_a.weight"), DType::Bf16, vec![1280, 5120]),
            (format!("{p}.attn.q_norm.weight"), DType::Bf16, vec![1280]),
            (format!("{p}.attn.wq_b.weight"), DType::Bf16, vec![32768, 1280]),
            (format!("{p}.attn.wkv.weight"), DType::Bf16, vec![512, 5120]),
            (format!("{p}.attn.kv_norm.weight"), DType::Bf16, vec![512]),
            (format!("{p}.attn.wo_a.weight"), DType::Bf16, vec![8192, 4096]),
            (format!("{p}.attn.wo_b.weight"), DType::Bf16, vec![5120, 8192]),
            (format!("{p}.attn.attn_sink"), DType::F32, vec![64]),
            (format!("{p}.ffn.gate.weight"), DType::Bf16, vec![128, 5120]),
            (format!("{p}.ffn.gate.bias"), DType::F32, vec![128]),
            (format!("{p}.ffn.gate.bias_vl"), DType::F32, vec![128]),
            (format!("{p}.ffn.shared_experts.w1.weight"), DType::Bf16, vec![2304, 5120]),
            (format!("{p}.ffn.shared_experts.w3.weight"), DType::Bf16, vec![2304, 5120]),
            (format!("{p}.ffn.shared_experts.w2.weight"), DType::Bf16, vec![5120, 2304]),
        ]);
    }
    specs
}

pub fn read_v41_dfly2_catalog(snapshot: &Path, target: &OfficialV41Config) -> Result<V41Dfly2Catalog> {
    let config = V41Dfly2Config::read(snapshot)?;
    config.validate_target(target)?;

    let index_path = snapshot.join("model.safetensors.index.json");
    let shard_files: Vec<String> = if index_path.is_file() {
        let index: IndexFile = serde_json::from_slice(&std::fs::read(&index_path)?)?;
        index.weight_map.into_values().collect::<BTreeSet<_>>().into_iter().collect()
    } else {
        let single = snapshot.join("model.safetensors");
        ensure!(single.is_file(), "DFly2 snapshot missing model.safetensors[.index.json]");
        vec!["model.safetensors".to_string()]
    };

    let mut dense = BTreeMap::new();
    let mut experts = Vec::with_capacity(3 * 128 * 6);

    for shard in &shard_files {
        let path = snapshot.join(shard);
        for meta in read_safetensors_metadata(&path)? {
            if let Some(rest) = meta.name.strip_prefix("model.layers.") {
                if let Some((stage_str, tail)) = rest.split_once(".ffn.experts.") {
                    let stage: usize = stage_str.parse().context("invalid DFly2 expert stage")?;
                    ensure!(stage < 3, "DFly2 expert stage {stage} out of range");
                    let new_name = format!("mtp.{stage}.ffn.experts.{tail}");
                    let mut aliased = meta;
                    aliased.name = new_name;
                    experts.push(V41Tensor {
                        shard: shard.clone(),
                        metadata: aliased,
                        placement: V41TensorPlacement::CoordinatorRtx,
                    });
                    continue;
                }
            }
            let prev = dense.insert(
                meta.name.clone(),
                V41Dfly2Tensor { shard: shard.clone(), metadata: meta },
            );
            ensure!(prev.is_none(), "duplicate DFly2 dense tensor");
        }
    }

    for (name, dtype, shape) in expected_dense_specs() {
        let tensor = dense
            .get(&name)
            .with_context(|| format!("missing DFly2 dense tensor {name}"))?;
        ensure!(
            tensor.metadata.dtype == dtype && tensor.metadata.shape == shape,
            "DFly2 tensor {name} geometry mismatch: expected {:?}/{:?}, got {:?}/{:?}",
            dtype,
            shape,
            tensor.metadata.dtype,
            tensor.metadata.shape
        );
    }

    ensure!(
        experts.len() == 3 * 128 * 6,
        "DFly2 expected {} routed expert tensors, found {}",
        3 * 128 * 6,
        experts.len()
    );
    experts.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));

    Ok(V41Dfly2Catalog {
        snapshot: snapshot.to_path_buf(),
        config,
        dense,
        experts,
    })
}

impl V41Dfly2Catalog {
    pub fn snapshot(&self) -> &Path { &self.snapshot }
    pub fn config(&self) -> &V41Dfly2Config { &self.config }

    /// Packed dense weight arena layout (`v41/dfly2/dense`): ONE device
    /// allocation holding every BF16/F32 DFly2 dense tensor at 256-byte
    /// alignment. Routed experts live in their own packed `ExpertWeights` arena.
    pub fn dense_layout(&self) -> Result<WeightArenaLayout> {
        let requests = self
            .dense
            .iter()
            .map(|(name, t)| Ok((name.clone(), usize::try_from(t.metadata.byte_length)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        WeightArenaLayout::new("v41/dfly2/dense", requests)
    }

    pub fn dense_tensors(&self) -> &BTreeMap<String, V41Dfly2Tensor> { &self.dense }

    /// Read one dense tensor's raw bytes from its safetensors shard.
    pub fn read_dense_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self
            .dense
            .get(name)
            .with_context(|| format!("unknown DFly2 dense tensor {name}"))?;
        let bytes = usize::try_from(tensor.metadata.byte_length)?;
        let mut buf = vec![0u8; bytes];
        let file = File::open(self.snapshot.join(&tensor.shard))
            .with_context(|| format!("opening DFly2 shard {}", tensor.shard))?;
        file.read_exact_at(&mut buf, tensor.metadata.byte_offset)
            .with_context(|| format!("reading DFly2 tensor {name}"))?;
        Ok(buf)
    }

    /// Build an in-memory `OfficialV41Catalog` adapter exposing the 3×128
    /// MXFP4 routed experts as `mtp.{0..3}.ffn.experts.*` so `ExpertWeights`
    /// and TP2 draft expert loaders reuse their native MXFP4 packing verbatim.
    pub fn expert_catalog(&self, target: &OfficialV41Config) -> OfficialV41Catalog {
        OfficialV41Catalog::from_dfly2_experts(
            &self.snapshot,
            target.clone(),
            self.experts.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_resolution_accepts_6_and_8_only() -> Result<()> {
        let cfg = V41Dfly2Config {
            trained_block_size: 6,
            max_block_slots: 16,
            target_kv_layers: [28, 32, 36],
            target_index_layers: [28, 32, 36],
            index_topk: 512,
            mask_token_id: 128799,
            hidden_size: 5120,
            num_layers: 3,
            vocab_size: 129280,
            num_routed_experts: 128,
            num_experts_per_tok: 3,
            moe_intermediate_size: 2304,
            hidden_correction_intermediate: 5120,
            swiglu_limit: 10.0,
        };
        assert_eq!(cfg.resolve_block_size(None)?, 6);
        assert_eq!(cfg.resolve_block_size(Some(6))?, 6);
        assert_eq!(cfg.resolve_block_size(Some(8))?, 8);
        assert!(cfg.resolve_block_size(Some(4)).is_err());
        assert!(cfg.resolve_block_size(Some(10)).is_err());
        Ok(())
    }
}
