//! Step-5 release contracts. Whole-model serving remains fail-closed.
use std::collections::BTreeMap;
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, FfnActivation};
use crate::plan::{checkpoint::Checkpoint, family::*, format::QuantOperand, names::indexed, spec::*};

mod placement;
pub mod admission;
pub mod exl3;
pub mod native;
pub mod coordinator;
pub mod ownership;
#[cfg(test)]
mod tests;

pub struct Step5;
pub static STEP5: Step5 = Step5;

#[derive(Debug, Clone)]
pub struct Config {
    pub norm_eps: f32,
    pub hidden: usize,
    pub vocab: usize,
    pub layers: usize,
    pub mtp: usize,
    pub dense: usize,
    pub experts: usize,
    pub intermediate: usize,
    pub shared: usize,
    pub topk: usize,
    pub heads: usize,
    pub groups: usize,
    pub dim: usize,
    pub window: usize,
    pub max_context: u64,
    pub full: Vec<bool>,
    pub moe: Vec<bool>,
    pub theta: Vec<f64>,
    pub rotary: Vec<usize>,
    pub rope_scaling: Option<crate::plan::rope::Llama3>,
    pub scale_full: bool,
    pub scale_sliding: bool,
    pub activation: Vec<FfnActivation>,
    pub shared_activation: Vec<FfnActivation>,
}

impl Config {
    pub fn parse(checkpoint: &Checkpoint) -> Result<Self> {
        let v = checkpoint.config.get("text_config").unwrap_or(&checkpoint.config);
        ensure!(v["model_type"] == "step5", "expected nested text_config.model_type=step5");
        let int = |key: &str| -> Result<usize> {
            let n = v[key].as_u64().with_context(|| format!("Step-5 requires {key}"))?;
            ensure!(n > 0 && n <= 16_777_216, "invalid Step-5 {key}: {n}");
            Ok(n as usize)
        };
        let layers = int("num_hidden_layers")?;
        let mtp = v["num_nextn_predict_layers"].as_u64().context("missing num_nextn_predict_layers")? as usize;
        ensure!(layers == 92 && mtp == 3, "Step-5 release requires 92 target layers and 3 MTP stages");
        let total = layers + mtp;
        let array = |key: &str| -> Result<&Vec<serde_json::Value>> {
            let a = v[key].as_array().with_context(|| format!("Step-5 requires {key}"))?;
            ensure!(a.len() == total, "Step-5 {key} must retain all {total} target/MTP entries, found {}", a.len());
            Ok(a)
        };
        let full = array("layer_types")?.iter().enumerate().map(|(layer, kind)| {
            ensure!(kind == "full_attention" || kind == "sliding_attention", "layer_types[{layer}]: {kind}");
            Ok(kind == "full_attention")
        }).collect::<Result<Vec<_>>>()?;
        let theta = array("rope_theta")?.iter().map(|v| v.as_f64().filter(|v| v.is_finite() && *v > 0.0)
            .context("invalid Step-5 rope_theta")).collect::<Result<Vec<_>>>()?;
        let dim = int("head_dim")?;
        let rotary = array("partial_rotary_factors")?.iter().map(|v| {
            let fraction = v.as_f64().context("invalid partial_rotary_factors")?;
            let dims = fraction * dim as f64;
            ensure!(dims.is_finite() && dims > 0.0 && dims <= dim as f64 && (dims - dims.round()).abs() < 1e-6
                && dims.round() as usize % 2 == 0, "invalid partial rotary width {dims}");
            Ok(dims.round() as usize)
        }).collect::<Result<Vec<_>>>()?;
        let activations = |key: &str| -> Result<Vec<FfnActivation>> {
            array(key)?.iter().map(|v| {
                let limit = v.as_f64().context("invalid SwiGLU limit")? as f32;
                FfnActivation::SiluPostClampFp32Product { limit }.validate().map_err(anyhow::Error::msg)
            }).collect()
        };
        let mut moe = vec![false; total];
        for layer in v["moe_layer_list"].as_array().context("missing moe_layer_list")? {
            let layer = layer.as_u64().context("invalid moe_layer_list entry")? as usize;
            ensure!(layer < layers && !moe[layer], "invalid/duplicate routed layer {layer}");
            moe[layer] = true;
        }
        ensure!(moe.iter().enumerate().all(|(l, routed)| *routed == (3..91).contains(&l)), "unsupported Step-5 routed layer map");
        ensure!(v["fp32_residual_connection"] == true && v["need_fp32_gate"] == true
            && v["use_head_wise_attn_gate"] == true, "Step-5 requires FP32 residual/router and headwise gate");
        let sparse = &v["sparse_config"];
        ensure!(sparse["enabled"] == true, "Step-5 sparse indexer must be enabled");
        for (key, expected) in [("proxy_dim", 256), ("sparse_indexer_rope_dim", 32), ("sparse_indexer_num_heads", 16),
            ("sparse_indexer_num_k_heads", 1), ("num_provider_groups", 4), ("topk", 512), ("region_block_size", 8)] {
            ensure!(sparse[key].as_u64() == Some(expected), "unsupported Step-5 sparse_config.{key}: {}", sparse[key]);
        }
        let norm_eps = v["rms_norm_eps"].as_f64().filter(|v| v.is_finite() && *v > 0.0)
            .context("Step-5 requires positive finite rms_norm_eps")? as f32;
        ensure!(norm_eps.is_finite() && norm_eps > 0.0 && v["norm_dtype"] == "float32",
            "Step-5 requires representable positive epsilon and norm_dtype=float32");
        let rope_scaling = v.get("rope_scaling").filter(|v| !v.is_null())
            .map(crate::plan::rope::Llama3::parse).transpose()?;
        let types = v.get("yarn_only_types").filter(|v| !v.is_null())
            .map(|v| v.as_array().context("Step-5 yarn_only_types must be an array")).transpose()?;
        if let Some(types) = types {
            ensure!(types.iter().all(|v| v == "full_attention" || v == "sliding_attention"),
                "unsupported Step-5 yarn_only_types");
        }
        let enabled = |kind: &str| types.is_none_or(|types| types.is_empty()
            || types.iter().any(|v| v == kind));
        let cfg = Self { norm_eps, hidden: int("hidden_size")?, vocab: int("vocab_size")?, layers, mtp,
            dense: int("intermediate_size")?, experts: int("moe_num_experts")?, intermediate: int("moe_intermediate_size")?,
            shared: int("share_expert_dim")?, topk: int("moe_top_k")?, heads: int("num_attention_heads")?,
            groups: int("num_attention_groups")?, dim, window: int("sliding_window")?,
            max_context: int("max_position_embeddings")? as u64, full, moe, theta, rotary,
            rope_scaling, scale_full: enabled("full_attention"), scale_sliding: enabled("sliding_attention"),
            activation: activations("swiglu_limits")?, shared_activation: activations("swiglu_limits_shared")? };
        ensure!((cfg.hidden, cfg.experts, cfg.intermediate, cfg.heads, cfg.groups, cfg.dim, cfg.window)
            == (4096, 352, 1536, 64, 4, 192, 512), "unsupported Step-5 release geometry");
        for (key, actual, expected) in [("vocab_size", cfg.vocab, 128896),
            ("intermediate_size", cfg.dense, 13824), ("share_expert_dim", cfg.shared, 1536),
            ("moe_top_k", cfg.topk, 8), ("max_position_embeddings", cfg.max_context as usize, 1048576)] {
            ensure!(actual == expected, "unsupported Step-5 {key}: {actual}, expected {expected}");
        }
        for (key, expected) in [("head_dim_full_attention", 192), ("head_dim_sliding_attention", 192),
            ("num_attention_groups_full_attention", 4), ("num_attention_groups_sliding_attention", 4),
            ("num_sliding_attention_heads", 64), ("num_experts_per_tok", 8)] {
            ensure!(v[key].as_u64() == Some(expected), "unsupported Step-5 {key}: {}", v[key]);
        }
        ensure!(v["norm_expert_weight"] == true && v["use_moe_router_bias"] == true
            && v["moe_router_scaling_factor"].as_f64() == Some(3.0),
            "Step-5 requires normalized expert weights, selection-only router bias and moe_router_scaling_factor=3");
        ensure!(cfg.full[..layers].iter().filter(|v| **v).count() == 23
            && cfg.full.iter().enumerate().all(|(l, full)| *full == (l < layers && l % 4 == 3)), "unsupported Step-5 full/SWA layer map");
        Ok(cfg)
    }

    pub fn rope_recipe(&self, layer: usize, indexer: bool) -> Result<crate::plan::rope::Recipe> {
        ensure!(layer < self.layers + self.mtp, "Step-5 rotary layer out of range");
        ensure!(!indexer || self.full[layer], "Step-5 indexer requires full attention");
        let scaled = if self.full[layer] { self.scale_full } else { self.scale_sliding };
        crate::plan::rope::Recipe::new(if indexer { 32 } else { self.rotary[layer] }, self.theta[layer],
            if scaled { self.rope_scaling } else { None })
    }

    pub fn rope_tables(&self) -> Result<crate::plan::rope::Tables> {
        let mut recipes = Vec::new();
        for layer in 0..self.layers + self.mtp {
            recipes.push(self.rope_recipe(layer, false)?);
            if self.full[layer] { recipes.push(self.rope_recipe(layer, true)?); }
        }
        crate::plan::rope::Tables::new(self.max_context as usize, recipes)
    }
}

#[derive(Clone)]
struct Tensor {
    role: TensorRole,
    shape: Vec<usize>,
    dtype: DType,
}

pub(super) struct Model {
    cfg: Config,
    spec: ModelSpec,
    tensors: BTreeMap<String, Tensor>,
    /// Header byte counts, never the index's potentially stale total_size.
    bytes: BTreeMap<Component, u64>,
    complete: bool,
    formats: Vec<String>,
    /// Header-native EXL3 dense core (empty for the official FP8/NVFP4 releases).
    pub(super) core: exl3::CoreInventory,
    /// Per routed target layer, the EXL3 width its experts are stored at.
    pub(super) expert_bits: BTreeMap<usize, usize>,
}

impl Model {
    /// Routed EXL3 widths as one two-tier decoder family (`exl3-k46`). A
    /// uniform checkpoint takes the empty adjacent tier, as the kernels expect.
    pub(super) fn exl3_tiers(&self) -> Option<Vec<usize>> {
        let mut bits = self.expert_bits.values().copied().collect::<Vec<_>>();
        bits.sort_unstable();
        bits.dedup();
        match bits.as_slice() {
            [] => None,
            [only] => Some(if *only == 6 { vec![5, 6] } else { vec![*only, only + 1] }),
            [low, high] => Some(vec![*low, *high]),
            _ => None,
        }
    }

    pub(super) fn exl3_package(&self) -> Option<String> {
        Some(format!("exl3-k{}", self.exl3_tiers()?.iter().map(usize::to_string).collect::<String>()))
    }

    pub fn is_exl3(&self) -> bool { !self.expert_bits.is_empty() }

    fn accepts_exl3(&self, role: &TensorRole, stem: &str, operand: &QuantOperand, bits: usize) -> Result<(), String> {
        let trellis = self.tensors.get(&format!("{stem}.trellis"))
            .ok_or_else(|| format!("{stem}: Step-5 keeps this tensor native; EXL3 storage is outside the recipe"))?;
        if trellis.role != *role {
            return Err(format!("{stem}: EXL3 operand role {:?} differs from its contract {:?}", role, trellis.role));
        }
        if operand.stored != trellis.shape || trellis.shape.get(2) != Some(&(16 * bits)) {
            return Err(format!("{stem}: EXL3 K{bits} storage {:?}, expected {:?}", operand.stored, trellis.shape));
        }
        if role.component == Component::RoutedExpert {
            let layer = role.layer.ok_or("routed EXL3 operand without a layer")?;
            if self.expert_bits.get(&layer) != Some(&bits) {
                return Err(format!("{stem}: layer {layer} routed experts mix EXL3 widths (K{bits} vs K{:?})",
                    self.expert_bits.get(&layer)));
            }
            return Ok(());
        }
        let projection = self.core.get(stem).ok_or_else(|| format!("{stem}: not in the native dense core inventory"))?;
        if projection.bits != bits { return Err(format!("{stem}: core K{} disagrees with headers K{bits}", projection.bits)); }
        Ok(())
    }

    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = Config::parse(checkpoint)?;
        let exl3 = exl3::is_exl3(checkpoint);
        let core = exl3::CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        let mut tensors = BTreeMap::new();
        // A header-native EXL3 projection replaces the `.weight` contract of its
        // stem with the standard trellis/suh/svh/mcg set at its own width.
        let mut add = |name: String, component, layer, shape: Vec<usize>, dtype| {
            let role = TensorRole { component, layer, expert: None };
            if let Some(projection) = name.strip_suffix(".weight").and_then(|stem| core.get(stem)) {
                let stem = &projection.stem;
                for (suffix, dtype, shape) in [
                    ("trellis", DType::I16, vec![shape[1] / 16, shape[0] / 16, 16 * projection.bits]),
                    ("suh", DType::F16, vec![shape[1]]), ("svh", DType::F16, vec![shape[0]]),
                    ("mcg", DType::I32, vec![])] {
                    tensors.insert(format!("{stem}.{suffix}"), Tensor { role: role.clone(), shape, dtype });
                }
            } else {
                tensors.insert(name, Tensor { role, shape, dtype });
            }
        };
        let (h, v, d, e, i) = (cfg.hidden, cfg.vocab, cfg.dim, cfg.experts, cfg.intermediate);
        add("model.embed_tokens.weight".into(), Component::Embedding, None, vec![v, h], DType::Bf16);
        add("lm_head.weight".into(), Component::LmHead, None, vec![v, h], DType::Bf16);
        add("model.norm.weight".into(), Component::Norm, None, vec![h], DType::F32);
        let mut formats = Vec::new();
        let mut expert_bits = BTreeMap::new();
        let mut experts: Vec<(String, Tensor)> = Vec::new();
        for layer in 0..cfg.layers + cfg.mtp {
            let base = format!("model.layers.{layer}");
            let draft = layer >= cfg.layers;
            let role = |component| if draft { Component::Speculator } else { component };
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                add(format!("{base}.{norm}.weight"), role(Component::Norm), Some(layer), vec![h], DType::F32);
            }
            for norm in ["q_norm", "k_norm"] {
                add(format!("{base}.self_attn.{norm}.weight"), role(Component::Attention), Some(layer), vec![d], DType::F32);
            }
            for (proj, n, k) in [("q", cfg.heads*d, h), ("k", cfg.groups*d, h), ("v", cfg.groups*d, h),
                ("o", h, cfg.heads*d), ("g", cfg.heads, h)] {
                add(format!("{base}.self_attn.{proj}_proj.weight"), role(Component::Attention), Some(layer), vec![n,k], DType::Bf16);
            }
            if cfg.full[layer] {
                for (name, shape, dtype) in [("sparse_indexer_q.weight", vec![4096,h], DType::Bf16),
                    ("sparse_indexer_k.weight", vec![256,h], DType::Bf16), ("sparse_indexer_z.weight", vec![256,h], DType::Bf16),
                    ("sparse_indexer_w.weight", vec![16,h], DType::F32), ("sparse_indexer_q_norm.weight", vec![256], DType::F32),
                    ("sparse_indexer_k_norm.weight", vec![256], DType::F32), ("sparse_indexer_k_norm.bias", vec![256], DType::F32),
                    ("ssmax_s", vec![64], DType::F32)] {
                    add(format!("{base}.self_attn.{name}"), Component::Indexer, Some(layer), shape, dtype);
                }
            }
            if cfg.moe[layer] {
                add(format!("{base}.moe.gate.weight"), Component::Router, Some(layer), vec![e,h], DType::Bf16);
                add(format!("{base}.moe.router_bias"), Component::Router, Some(layer), vec![e], DType::F32);
            }
            for (prefix, inter, component, stacked) in if cfg.moe[layer] {
                vec![("moe", i, Component::RoutedExpert, true), ("share_expert", cfg.shared, Component::SharedExpert, false)]
            } else { vec![("mlp", cfg.dense, role(Component::DenseFfn), false)] } {
                if stacked && exl3 {
                    // Per-expert standard EXL3 modules; the width is the checkpoint's own.
                    let bits = exl3_expert_bits(checkpoint, &format!("{base}.moe.experts.0.gate_proj"))?;
                    ensure!(exl3::declared_expert_bits(checkpoint, layer).is_none_or(|declared| declared == bits),
                        "{base}: routed experts are K{bits}; quantization_config declares K{:?}",
                        exl3::declared_expert_bits(checkpoint, layer));
                    expert_bits.insert(layer, bits);
                    let label = format!("exl3-k{bits}");
                    if !formats.contains(&label) { formats.push(label); }
                    for expert in 0..e {
                        for (proj,n,k) in [("gate",inter,h), ("up",inter,h), ("down",h,inter)] {
                            let stem = format!("{base}.moe.experts.{expert}.{proj}_proj");
                            for (suffix, dtype, shape) in [("trellis", DType::I16, vec![k/16, n/16, 16*bits]),
                                ("suh", DType::F16, vec![k]), ("svh", DType::F16, vec![n]), ("mcg", DType::I32, vec![])] {
                                experts.push((format!("{stem}.{suffix}"), Tensor { role: TensorRole::expert(component, layer, expert), shape, dtype }));
                            }
                        }
                    }
                    continue;
                }
                for (proj,n,k) in [("gate",inter,h), ("up",inter,h), ("down",h,inter)] {
                    let stem = format!("{base}.{prefix}.{proj}_proj");
                    let name = format!("{stem}.weight");
                    let actual = header(checkpoint, &name);
                    let dtype = if stacked { actual.map(|t| t.meta.dtype.clone()).unwrap_or_else(|| {
                        if checkpoint.contains_tensor(&format!("{stem}.weight_scale_inv")) { DType::F8E4M3 }
                        else if checkpoint.contains_tensor(&format!("{stem}.weight_scale_2")) { DType::U8 }
                        else { DType::Bf16 }
                    }) } else { DType::Bf16 };
                    ensure!(!stacked || matches!(dtype, DType::Bf16 | DType::F8E4M3 | DType::U8), "{name}: unsupported routed dtype {dtype:?}");
                    ensure!(!stacked || layer < 88 || dtype == DType::Bf16, "{name}: release tail layers 88-90 must remain BF16");
                    if stacked {
                        ensure!((dtype == DType::F8E4M3) == checkpoint.contains_tensor(&format!("{stem}.weight_scale_inv")),
                            "{name}: FP8 storage dtype and weight_scale_inv presence disagree");
                        ensure!((dtype == DType::U8) == checkpoint.contains_tensor(&format!("{stem}.weight_scale_2")),
                            "{name}: NVFP4 storage dtype and weight_scale_2 presence disagree");
                    }
                    let shape = if stacked { vec![e,n,if dtype == DType::U8 { k/2 } else { k }] } else { vec![n,k] };
                    add(name, component, Some(layer), shape, dtype.clone());
                    if stacked {
                        let label = match dtype { DType::F8E4M3 => "fp8-block128", DType::U8 => "nvfp4-g16-w4a4", _ => "bf16" };
                        if !formats.iter().any(|s| s == label) { formats.push(label.to_owned()); }
                        if dtype == DType::F8E4M3 {
                            add(format!("{stem}.weight_scale_inv"), component, Some(layer), vec![e,n.div_ceil(128),k.div_ceil(128)], DType::F32);
                        } else if dtype == DType::U8 {
                            add(format!("{stem}.weight_scale"), component, Some(layer), vec![e,n,k/16], DType::F8E4M3);
                            for suffix in ["weight_scale_2", "input_scale"] {
                                add(format!("{stem}.{suffix}"), component, Some(layer), vec![e], DType::F32);
                            }
                        }
                    }
                }
            }
            if draft {
                for norm in ["enorm", "hnorm", "transformer.shared_head.norm"] {
                    add(format!("{base}.{norm}.weight"), Component::Speculator, Some(layer), vec![h], DType::F32);
                }
                add(format!("{base}.eh_proj.weight"), Component::Speculator, Some(layer), vec![h,2*h], DType::Bf16);
                add(format!("{base}.transformer.shared_head.output.weight"), Component::Speculator, Some(layer), vec![v,h], DType::Bf16);
            }
        }
        if let Some(vision) = checkpoint.config.get("vision_config") {
            ensure!(vision["model_type"] == "perception_encoder" && vision["width"] == 1536
                && vision["layers"] == 47 && vision["patch_size"] == 14 && vision["image_size"] == 728
                && vision["hidden_act"] == "quick_gelu", "unsupported Step-5 official vision geometry");
            let mut vision_tensor = |name: String, shape| add(name, Component::Vision, None, shape, DType::Bf16);
            vision_tensor("vision_model.conv1.weight".into(), vec![1536,3,14,14]);
            vision_tensor("vision_model.positional_embedding".into(), vec![2704,1536]);
            for suffix in ["weight", "bias"] { vision_tensor(format!("vision_model.ln_pre.{suffix}"), vec![1536]); }
            for layer in 0..47 {
                let base = format!("vision_model.transformer.resblocks.{layer}");
                for (name,shape) in [("attn.in_proj_weight",vec![4608,1536]), ("attn.in_proj_bias",vec![4608]),
                    ("attn.out_proj.weight",vec![1536,1536]), ("attn.out_proj.bias",vec![1536]),
                    ("mlp.c_fc.weight",vec![8960,1536]), ("mlp.c_fc.bias",vec![8960]),
                    ("mlp.c_proj.weight",vec![1536,8960]), ("mlp.c_proj.bias",vec![1536])] {
                    vision_tensor(format!("{base}.{name}"), shape);
                }
                for norm in ["ln_1", "ln_2"] { for suffix in ["weight", "bias"] {
                    vision_tensor(format!("{base}.{norm}.{suffix}"), vec![1536]);
                }}
                for ls in ["ls_1", "ls_2"] { vision_tensor(format!("{base}.{ls}.gamma"), vec![1536]); }
            }
            for (index,n,k) in [(1,3072,1536), (2,6144,3072)] {
                vision_tensor(format!("vision_model.vit_downsampler{index}.weight"), vec![n,k,3,3]);
                vision_tensor(format!("vision_model.vit_downsampler{index}.bias"), vec![n]);
            }
            vision_tensor("vit_large_projector.weight".into(), vec![h,6144]);
        }
        tensors.extend(experts);
        for (name, contract) in &tensors {
            ensure!(checkpoint.contains_tensor(name), "Step-5 checkpoint lacks required tensor {name}");
            if let Some(actual) = header(checkpoint, name) {
                ensure!(actual.meta.shape == contract.shape && actual.meta.dtype == contract.dtype,
                    "{name}: expected {:?} {:?}, found {:?} {:?}", contract.dtype, contract.shape, actual.meta.dtype, actual.meta.shape);
            }
        }
        // A layer has one expert storage format across all three projections.
        for layer in (3..91).filter(|_| !exl3) {
            let dtype = &tensors[&format!("model.layers.{layer}.moe.gate_proj.weight")].dtype;
            for proj in ["up", "down"] {
                let name = format!("model.layers.{layer}.moe.{proj}_proj.weight");
                ensure!(&tensors[&name].dtype == dtype, "{name}: routed projection format differs from gate_proj");
            }
        }
        let complete = checkpoint.missing_shards.is_empty() && tensors.keys().all(|name|
            header(checkpoint, name).is_some());
        let mut bytes = BTreeMap::new();
        for tensor in &checkpoint.tensors {
            let contract = tensors.get(&tensor.meta.name)
                .with_context(|| format!("unrecognized Step-5 tensor {}", tensor.meta.name))?;
            *bytes.entry(contract.role.component).or_default() += tensor.meta.byte_length;
        }
        let spec = ModelSpec { family: "step5", architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
            hidden: h, vocab: v, layers: (0..cfg.layers).map(|l| LayerSpec {
                attention: if cfg.full[l] { AttentionKind::Gqa { heads: cfg.heads, kv_heads: cfg.groups, head_dim: d } }
                    else { AttentionKind::SlidingGqa { heads: cfg.heads, kv_heads: cfg.groups, head_dim: d, window: cfg.window, sinks: false } },
                ffn: if cfg.moe[l] { FfnKind::Moe } else { FfnKind::Dense { intermediate: cfg.dense } },
                rope: Some(RopeSpec { dims: cfg.rotary[l], theta: cfg.theta[l] }),
            }).collect(), moe: Some(MoeSpec { experts: e, top_k: cfg.topk, intermediate: i, shared_experts: 1,
                shared_intermediate: cfg.shared, scoring: "sigmoid; FP32, bias for selection only; renormalize unbiased top-k".into(),
                routed_scaling: Some(3.0), groups: None }), speculator: Some(SpeculatorSpec::NativeMtp { layers: cfg.mtp }),
            tables: vec![], vision: checkpoint.config.get("vision_config").is_some(), notes: vec![
                format!("92 target layers (69 SWA512, 23 sparse GQA192); 3 separate MTP stages; routed formats: {}", formats.join(", ")),
                "CSA compresses index keys only: one E4M3 proxy256 summary per eight tokens, original GQA K/V retained; provider-group top-512 regions".into(),
                "FP32 residual and zero-centered norms; post-SiLU clamp + FP32 product, one BF16 round, for routed/shared/dense FFNs".into(),
                "ssmax_s [64] FP32 is retained and validated, unused by released kernels; ordinary attention scale 1/sqrt(192)".into(),
                "Eager native admission on qualified WIP cards; native MTP defaults to depth 3, with exact predictor header contracts; no release or media qualification".into(),
            ] };
        let mut spec = spec;
        if complete {
            if let Ok(index) = crate::plan::checkpoint::read_json(&checkpoint.snapshot.join("model.safetensors.index.json")) {
                if let Some(advertised) = index["metadata"]["total_size"].as_u64() {
                    let headers: u64 = bytes.values().sum();
                    if advertised != headers {
                        spec.notes.push(format!("stale index metadata.total_size {advertised} bytes; authoritative safetensors header data_offsets sum {headers} bytes"));
                    }
                }
            }
        }
        Ok(Self { cfg, spec, tensors, bytes, complete, formats, core, expert_bits })
    }
}

/// `Checkpoint::tensors` is sorted by name; EXL3 publications carry ~376k headers.
fn header<'c>(checkpoint: &'c Checkpoint, name: &str) -> Option<&'c crate::plan::checkpoint::CheckpointTensor> {
    checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name)).ok().map(|at| &checkpoint.tensors[at])
}

/// A routed EXL3 module's width from its own trellis header (index-only
/// inventories read that one header).
fn exl3_expert_bits(checkpoint: &Checkpoint, stem: &str) -> Result<usize> {
    let name = format!("{stem}.trellis");
    let shape = match header(checkpoint, &name) {
        Some(t) => t.meta.shape.clone(),
        None => {
            let shard = checkpoint.weight_map.get(&name)
                .with_context(|| format!("Step-5 EXL3 checkpoint lacks {name}"))?;
            crate::read_safetensors_metadata(&checkpoint.snapshot.join(shard))?.into_iter()
                .find(|m| m.name == name).with_context(|| format!("{shard} lacks indexed {name}"))?.shape
        }
    };
    ensure!(shape.len() == 3 && shape[2] % 16 == 0 && (2..=6).contains(&(shape[2] / 16)),
        "{name}: EXL3 trellis shape {shape:?} is not K2..K6");
    Ok(shape[2] / 16)
}

impl Family for Step5 {
    fn decode_lanes(&self) -> crate::plan::decode_lanes::LaneSupport {
        crate::plan::decode_lanes::LaneSupport::One {
            reason: "a second lane needs its own decode link, separate from the shared prefill link; low priority (#744)" }
    }
    fn spark_decode(&self) -> Option<crate::plan::spark_executions::SparkDecode> {
        // Step-5 decode/verify steps send at most 64 rows (`rows.min(64)`).
        Some(crate::plan::spark_executions::SparkDecode { rows: 64 })
    }
    fn id(&self) -> &'static str { "step5" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Serving }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|a| matches!(a.as_str(), "Step5ForConditionalGeneration" | "Step5ForCausalLM"))
            || matches!(checkpoint.model_type(), Some("step5_vl" | "step5"))
    }
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        Ok(Box::new(Model::open(checkpoint).map_err(ConfigError::from_anyhow)?))
    }
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        if name.starts_with("vision_model.") || name.starts_with("vit_large_projector.") { return Some(TensorRole::new(Component::Vision)); }
        for (prefix,component) in [("model.embed_tokens.",Component::Embedding),("lm_head.",Component::LmHead),("model.norm.",Component::Norm)] {
            if name.starts_with(prefix) { return Some(TensorRole::new(component)); }
        }
        let (layer, rest) = indexed(name, "model.layers.")?;
        if layer >= spec.layers.len() {
            return (layer < spec.layers.len()+3).then(|| TensorRole::layer(Component::Speculator, layer));
        }
        if let Some((expert, _)) = indexed(rest, "moe.experts.") {
            return Some(TensorRole::expert(Component::RoutedExpert, layer, expert));
        }
        let component = if rest.starts_with("moe.gate_proj.") || rest.starts_with("moe.up_proj.") || rest.starts_with("moe.down_proj.") { Component::RoutedExpert }
            else if rest.starts_with("moe.gate.") || rest == "moe.router_bias" { Component::Router }
            else if rest.starts_with("share_expert.") { Component::SharedExpert }
            else if rest.starts_with("mlp.") { Component::DenseFfn }
            else if rest.starts_with("self_attn.sparse_indexer_") || rest == "self_attn.ssmax_s" { Component::Indexer }
            else if rest.starts_with("self_attn.") { Component::Attention }
            else if rest.starts_with("input_layernorm.") || rest.starts_with("post_attention_layernorm.") { Component::Norm }
            else { return None; };
        Some(TensorRole::layer(component, layer))
    }
    fn component_hint(&self, component: Component) -> Option<Hint> {
        match component {
            Component::Vision => Some(Hint { what: "Step-5 bundled vision is not qualified".into(),
                how: "Use text-only serving; qualify the official bundled encoder before enabling media.".into() }),
            _ => None,
        }
    }
    fn optional(&self, component: Component) -> bool {
        matches!(component, Component::Vision | Component::Speculator)
    }
    fn expert_catalog(&self) -> bool { true }

}

impl FamilyModel for Model {
    fn spec(&self) -> &ModelSpec { &self.spec }
    fn native_serving_admission(&self) -> bool { true }
    fn vision_mode(&self, mode: crate::plan::MediaMode) -> crate::plan::MediaMode {
        if mode == crate::plan::MediaMode::Auto { crate::plan::MediaMode::Off } else { mode }
    }
    fn cache_geometry(&self, options: crate::serving_capacity::CacheOptions)
        -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        let layout = crate::serving_capacity::step5::Step5CacheLayout::new(
            &self.cfg, options.coordinator_ranks, self.cfg.max_context, options.native_mtp_layers, options.prefill_rows.max(64), options.step5_kv)?;
        Ok(Some(layout.geometry))
    }
    fn activation(&self, layer: usize, shared: bool) -> Option<FfnActivation> {
        if shared { self.cfg.shared_activation.get(layer) } else { self.cfg.activation.get(layer) }.copied()
    }
    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        if !self.tensors.contains_key(stem) && !self.tensors.contains_key(&format!("{stem}.weight"))
            && !self.tensors.contains_key(&format!("{stem}.trellis")) {
            return Err(format!("{stem}: unknown Step-5 tensor; add a validated header contract"));
        }
        if let Some(bits) = operand.exl3_bits() {
            return self.accepts_exl3(role, stem, operand, bits);
        }
        if role.component == Component::RoutedExpert {
            let contract = self.tensors.get(&format!("{stem}.weight"));
            let shape_matches = contract.is_some_and(|tensor| tensor.role == *role
                && tensor.shape == operand.logical && tensor.shape == operand.stored);
            if shape_matches && role.layer.is_some_and(|layer| (3..88).contains(&layer))
                && contract.is_some_and(|tensor| tensor.dtype == DType::F8E4M3)
                && operand.is_fp8_block(128, &[crate::plan::format::ScaleEncoding::F32]) {
                return Ok(());
            }
            if contract.is_some_and(|tensor| tensor.role == *role && tensor.dtype == DType::U8
                    && tensor.shape == operand.stored)
                && role.layer.is_some_and(|layer| (3..88).contains(&layer))
                && operand.is_nvfp4() && operand.input_scale.is_some_and(|scale|
                    scale.encoding == crate::plan::format::ScaleEncoding::F32)
                && operand.companions.is_empty()
                && operand.stored.len() == 3 && operand.logical.len() == 3
                && operand.logical[..2] == operand.stored[..2]
                && operand.stored[2].checked_mul(2) == Some(operand.logical[2]) {
                return Ok(());
            }
            if shape_matches && role.layer.is_some_and(|layer| (88..91).contains(&layer))
                && contract.is_some_and(|tensor| tensor.dtype == DType::Bf16)
                && operand.encoding == crate::plan::format::Encoding::Bf16
                && operand.scale.is_none() && operand.global_scale.is_none()
                && operand.input_scale.is_none() && operand.companions.is_empty() {
                return Ok(());
            }
        }
        // Predictor headers in this same contract are the three qualified SWA/dense
        // stages, including their private EH projection, norms and shared head.
        if !matches!(role.component, Component::RoutedExpert | Component::Vision) {
            if let Some(tensor) = self.tensors.get(stem).or_else(|| self.tensors.get(&format!("{stem}.weight"))) {
                let encoding = match tensor.dtype {
                    DType::Bf16 => Some(crate::plan::format::Encoding::Bf16),
                    DType::F32 => Some(crate::plan::format::Encoding::F32),
                    _ => None,
                };
                // The index key norm groups its validated FP32 bias with the weight.
                let companions = operand.companions.is_empty()
                    || (operand.companions == ["bias"]
                        && stem.ends_with(".self_attn.sparse_indexer_k_norm")
                        && self.tensors.get(&format!("{stem}.bias")).is_some_and(|bias|
                            bias.role == *role && bias.dtype == DType::F32 && bias.shape == tensor.shape));
                let container = role.component != Component::Speculator || matches!(
                    (&tensor.dtype, operand.container.as_str()), (DType::Bf16, "bf16") | (DType::F32, "f32"));
                if tensor.role == *role && tensor.shape == operand.logical && tensor.shape == operand.stored
                    && container && encoding == Some(operand.encoding) && operand.scale.is_none()
                    && operand.global_scale.is_none() && operand.input_scale.is_none() && companions {
                    return Ok(());
                }
            }
        }
        Err(format!("{stem}: Step-5 {:?} operand is outside the validated native contract", role.component))
    }
    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        if operand.exl3_bits().is_some() {
            let package = format!("step5:{}", self.exl3_package()?);
            return Some(ExpertContract {
                spark_worlds: crate::plan::experts::spark_worlds(&package, self.cfg.intermediate),
                package, block: 128,
                local: Ok("serve-step5 RTX expert onboarding (EXL3 rtx-tp1/rtx-tp2 package)".into()),
            });
        }
        let package = if operand.is_fp8_block(128, &[crate::plan::format::ScaleEncoding::F32]) {
            "step5:fp8"
        } else if operand.is_nvfp4() && operand.input_scale.is_some_and(|scale|
            scale.encoding == crate::plan::format::ScaleEncoding::F32) {
            "step5:nvfp4"
        } else if operand.encoding == crate::plan::format::Encoding::Bf16
            && operand.scale.is_none() && operand.global_scale.is_none()
            && operand.input_scale.is_none() && operand.companions.is_empty() {
            "step5:bf16"
        } else { return None; };
        Some(ExpertContract {
            package: package.into(), block: 128,
            spark_worlds: crate::plan::experts::spark_worlds(package, self.cfg.intermediate),
            local: Err("Step-5 coordinator-local routed experts are not qualified".into()),
        })
    }
    fn placement(&self, input: &crate::plan::PlanInput, report: &mut crate::plan::PlanReport,
        options: &mut crate::plan::layout::LayoutOptions, requested: crate::plan::ExpertPlacement)
        -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        Some(placement::layout(self, input, report, options, requested))
    }
}
