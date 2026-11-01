//! Dense Qwen checkpoint contract. Header-only; no runtime or weight conversion.
pub mod config;
pub use config::Qwen35Config;
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;
use crate::families::qwen::HybridAttention;
use crate::plan::{checkpoint::Checkpoint, format::{self, Encoding, QuantOperand, ScaleEncoding}, Component, TensorRole};

#[derive(Debug, Clone)]
pub struct TensorContract {
    pub role: TensorRole,
    pub shape: Vec<usize>,
    pub fp8: bool,
    pub fp32: bool,
}

pub struct Qwen35Checkpoint {
    pub config: Qwen35Config,
    /// Exact names, including bias/scales only when the checkpoint declares them.
    pub tensors: BTreeMap<String, TensorContract>,
}

impl Qwen35Checkpoint {
    pub fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let config = Qwen35Config::from_hf(&checkpoint.config)?;
        let mut tensors = BTreeMap::new();
        let h = config.hidden;
        add(&mut tensors, "model.language_model.embed_tokens.weight", Component::Embedding, None, &[config.vocab, h], false, false);
        if !config.tied_embeddings {
            add(&mut tensors, "lm_head.weight", Component::LmHead, None, &[config.vocab, h], false, false);
        }
        add(&mut tensors, "model.language_model.norm.weight", Component::Norm, None, &[h], false, false);
        for (layer, &kind) in config.attention.iter().enumerate() {
            block(&mut tensors, &format!("model.language_model.layers.{layer}"), &config, kind, Some(layer), false);
        }
        if config.mtp_layers > 0 {
            for name in ["norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding"] {
                add(&mut tensors, &format!("mtp.{name}.weight"), Component::Speculator, None, &[h], false, false);
            }
            add(&mut tensors, "mtp.fc.weight", Component::Speculator, None, &[h, 2 * h], false, false);
            block(&mut tensors, "mtp.layers.0", &config, HybridAttention::Full, None, true);
        }
        if let Some(vision) = checkpoint.config.get("vision_config") {
            for (key, value) in [("depth", 27), ("hidden_size", 1152), ("intermediate_size", 4304),
                ("num_heads", 16), ("out_hidden_size", h), ("patch_size", 16), ("temporal_patch_size", 2),
                ("spatial_merge_size", 2), ("num_position_embeddings", 2304), ("in_channels", 3)] {
                ensure!(vision[key].as_u64() == Some(value as u64), "unsupported vision_config.{key}: {}", vision[key]);
            }
            ensure!(vision["hidden_act"] == "gelu_pytorch_tanh"
                && vision["deepstack_visual_indexes"].as_array().is_some_and(Vec::is_empty), "unsupported dense Qwen vision activation/deepstack");
            let mut tower = |name: &str, shape: &[usize], bias: bool| {
                add(&mut tensors, &format!("model.visual.{name}.weight"), Component::Vision, None, shape, false, false);
                if bias { add(&mut tensors, &format!("model.visual.{name}.bias"), Component::Vision, None, &[shape[0]], false, false); }
            };
            tower("patch_embed.proj", &[1152, 3, 2, 16, 16], true);
            tower("pos_embed", &[2304, 1152], false);
            tower("merger.norm", &[1152], true);
            tower("merger.linear_fc1", &[4608, 4608], true);
            tower("merger.linear_fc2", &[h, 4608], true);
            for layer in 0..27 {
                for (name, shape) in [("norm1", vec![1152]), ("norm2", vec![1152]),
                    ("attn.qkv", vec![3456, 1152]), ("attn.proj", vec![1152, 1152]),
                    ("mlp.linear_fc1", vec![4304, 1152]), ("mlp.linear_fc2", vec![1152, 4304])] {
                    tower(&format!("blocks.{layer}.{name}"), &shape, true);
                }
            }
        }
        for name in tensors.keys() {
            ensure!(checkpoint.contains_tensor(name), "dense Qwen checkpoint lacks required tensor {name}");
        }
        Ok(Self { config, tensors })
    }

    pub fn accepts(&self, stem: &str, operand: &QuantOperand) -> Result<(), String> {
        let contract = self.tensors.get(&format!("{stem}.weight")).or_else(|| self.tensors.get(stem))
            .ok_or_else(|| format!("no dense Qwen tensor contract for {stem}"))?;
        if operand.logical != contract.shape {
            return Err(format!("{stem}: expected {:?}, found {:?}", contract.shape, operand.logical));
        }
        let plain = operand.is_plain(if contract.fp32 { &[Encoding::Bf16, Encoding::F32] } else { &[Encoding::Bf16] });
        let fp8 = contract.fp8 && operand.is_fp8_block(128, &[ScaleEncoding::Bf16, ScaleEncoding::F32]);
        if !plain && !fp8 {
            return Err(format!("{stem}: expected BF16{}{}, found {}", if contract.fp32 { "/FP32" } else { "" },
                if contract.fp8 { " or native E4M3 + 128x128 BF16/FP32 multiplier scales" } else { "" }, operand.label()));
        }
        let has_bias = self.tensors.contains_key(&format!("{stem}.bias"));
        if operand.companions.iter().any(|c| c != "bias") || operand.companions.iter().any(|c| c == "bias") != has_bias {
            return Err(format!("{stem}: unexpected or missing bias/companion tensor"));
        }
        Ok(())
    }

    /// Validate every readable tensor header, independently of execution readiness.
    pub fn validate_headers(&self, checkpoint: &Checkpoint) -> Result<()> {
        for tensor in &checkpoint.tensors {
            if let Some(contract) = self.tensors.get(&tensor.meta.name) {
                ensure!(tensor.meta.shape == contract.shape, "{}: expected {:?}, found {:?}",
                    tensor.meta.name, contract.shape, tensor.meta.shape);
                if tensor.meta.name.ends_with(".bias") {
                    ensure!(tensor.meta.dtype == cuteafd_core::DType::Bf16, "{}: bias must remain BF16", tensor.meta.name);
                }
            } else {
                let (stem, suffix) = format::split_stem(&tensor.meta.name);
                ensure!(suffix == "weight_scale_inv" && self.tensors.get(&format!("{stem}.weight")).is_some_and(|c| c.fp8),
                    "unrecognized dense Qwen tensor {}", tensor.meta.name);
            }
        }
        for (stem, members) in format::group_by_stem(&checkpoint.tensors) {
            let operand = format::detect(&members)?;
            self.accepts(&stem, &operand).map_err(anyhow::Error::msg)?;
        }
        ensure!(checkpoint.missing_shards.is_empty(), "dense Qwen checkpoint has unreadable shards: {:?}", checkpoint.missing_shards);
        for name in self.tensors.keys() {
            ensure!(checkpoint.tensors.binary_search_by(|t| t.meta.name.cmp(name)).is_ok(), "missing dense Qwen header {name}");
        }
        Ok(())
    }

    pub fn read(snapshot: &std::path::Path) -> Result<Self> {
        let checkpoint = Checkpoint::open(snapshot)?;
        let model = Self::open(&checkpoint)?;
        model.validate_headers(&checkpoint).context("dense Qwen header contract")?;
        Ok(model)
    }
}

fn add(out: &mut BTreeMap<String, TensorContract>, name: &str, component: Component, layer: Option<usize>,
    shape: &[usize], fp8: bool, fp32: bool) {
    out.insert(name.into(), TensorContract { role: TensorRole { component, layer, expert: None }, shape: shape.into(), fp8, fp32 });
}

fn block(out: &mut BTreeMap<String, TensorContract>, prefix: &str, cfg: &Qwen35Config,
    kind: HybridAttention, layer: Option<usize>, mtp: bool) {
    let h = cfg.hidden;
    let g = cfg.gdn;
    let mut tensor = |name: &str, component: Component, shape: &[usize], fp8: bool, fp32: bool| {
        add(out, &format!("{prefix}.{name}"), if mtp { Component::Speculator } else { component }, layer, shape, fp8, fp32);
    };
    for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
        tensor(name, Component::Norm, &[h], false, false);
    }
    for name in ["gate_proj", "up_proj", "down_proj"] {
        let shape = if name == "down_proj" { [h, cfg.intermediate] } else { [cfg.intermediate, h] };
        tensor(&format!("mlp.{name}.weight"), Component::DenseFfn, &shape, true, false);
    }
    match kind {
        HybridAttention::Gdn => {
            for (name, shape, fp8, fp32) in [
                ("in_proj_qkv.weight", vec![g.conv_width(), h], true, false),
                ("in_proj_z.weight", vec![g.value_width(), h], true, false),
                ("in_proj_a.weight", vec![g.value_heads, h], false, false),
                ("in_proj_b.weight", vec![g.value_heads, h], false, false),
                ("out_proj.weight", vec![h, g.value_width()], true, false),
                ("conv1d.weight", vec![g.conv_width(), 1, g.conv_kernel], false, false),
                ("norm.weight", vec![g.value_dim], false, true),
                ("A_log", vec![g.value_heads], false, true),
                ("dt_bias", vec![g.value_heads], false, false)] {
                tensor(&format!("linear_attn.{name}"), Component::Attention, &shape, fp8, fp32);
            }
        }
        HybridAttention::Full => {
            for (name, shape, fp8) in [
                ("q_proj.weight", vec![2 * cfg.heads * cfg.head_dim, h], true),
                ("k_proj.weight", vec![cfg.kv_heads * cfg.head_dim, h], true),
                ("v_proj.weight", vec![cfg.kv_heads * cfg.head_dim, h], true),
                ("o_proj.weight", vec![h, cfg.heads * cfg.head_dim], true),
                ("q_norm.weight", vec![cfg.head_dim], false), ("k_norm.weight", vec![cfg.head_dim], false)] {
                tensor(&format!("self_attn.{name}"), Component::Attention, &shape, fp8, false);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
