//! Step-5 release metadata only. Native kernels are deliberately not advertised.
use std::collections::BTreeMap;
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, FfnActivation};
use crate::plan::{checkpoint::Checkpoint, family::*, format::QuantOperand, names::indexed, spec::*};

mod placement;
pub mod coordinator;
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
    pub activation: Vec<FfnActivation>,
    pub shared_activation: Vec<FfnActivation>,
}

impl Config {
    fn parse(checkpoint: &Checkpoint) -> Result<Self> {
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
        let cfg = Self { norm_eps, hidden: int("hidden_size")?, vocab: int("vocab_size")?, layers, mtp,
            dense: int("intermediate_size")?, experts: int("moe_num_experts")?, intermediate: int("moe_intermediate_size")?,
            shared: int("share_expert_dim")?, topk: int("moe_top_k")?, heads: int("num_attention_heads")?,
            groups: int("num_attention_groups")?, dim, window: int("sliding_window")?,
            max_context: int("max_position_embeddings")? as u64, full, moe, theta, rotary,
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
}

impl Model {
    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = Config::parse(checkpoint)?;
        let mut tensors = BTreeMap::new();
        let mut add = |name: String, component, layer, shape: Vec<usize>, dtype| {
            tensors.insert(name, Tensor { role: TensorRole { component, layer, expert: None }, shape, dtype });
        };
        let (h, v, d, e, i) = (cfg.hidden, cfg.vocab, cfg.dim, cfg.experts, cfg.intermediate);
        add("model.embed_tokens.weight".into(), Component::Embedding, None, vec![v, h], DType::Bf16);
        add("lm_head.weight".into(), Component::LmHead, None, vec![v, h], DType::Bf16);
        add("model.norm.weight".into(), Component::Norm, None, vec![h], DType::F32);
        let mut formats = Vec::new();
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
                for (proj,n,k) in [("gate",inter,h), ("up",inter,h), ("down",h,inter)] {
                    let stem = format!("{base}.{prefix}.{proj}_proj");
                    let name = format!("{stem}.weight");
                    let actual = checkpoint.tensors.iter().find(|t| t.meta.name == name);
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
        for (name, contract) in &tensors {
            ensure!(checkpoint.contains_tensor(name), "Step-5 checkpoint lacks required tensor {name}");
            if let Some(actual) = checkpoint.tensors.iter().find(|t| &t.meta.name == name) {
                ensure!(actual.meta.shape == contract.shape && actual.meta.dtype == contract.dtype,
                    "{name}: expected {:?} {:?}, found {:?} {:?}", contract.dtype, contract.shape, actual.meta.dtype, actual.meta.shape);
            }
        }
        // A layer has one expert storage format across all three projections.
        for layer in 3..91 {
            let dtype = &tensors[&format!("model.layers.{layer}.moe.gate_proj.weight")].dtype;
            for proj in ["up", "down"] {
                let name = format!("model.layers.{layer}.moe.{proj}_proj.weight");
                ensure!(&tensors[&name].dtype == dtype, "{name}: routed projection format differs from gate_proj");
            }
        }
        let complete = checkpoint.missing_shards.is_empty() && tensors.keys().all(|name|
            checkpoint.tensors.iter().any(|t| &t.meta.name == name));
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
                "Header-only memory lower bounds; no executable kernels, runtime workspace/prefix/graph admission or serving-fit claim".into(),
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
        Ok(Self { cfg, spec, tensors, bytes, complete, formats })
    }
}

impl Family for Step5 {
    fn id(&self) -> &'static str { "step5" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Planned }
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
        let exports = match component {
            Component::RoutedExpert => "shared stacked FP8 block128 / NVFP4 W4A4 expert-axis reader and native BF16-tail backend; step5 H4096 E352 K8 I1536 TP6(256)/TP4(384) exports; post-SiLU clamp/FP32 product",
            Component::Attention => "BF16 GQA192 projections, Q/K norms, sigmoid head gate, SWA512 + verify slack, provider-group sparse paged GQA192 prefill/decode/verify exports",
            Component::Indexer | Component::Compressor => "CSA-8 summarize/commit/rollback, index norm/RoPE/E4M3 rounding, signed weighted-ReLU scores and deterministic provider top-512 region expansion exports",
            Component::DenseFfn | Component::SharedExpert => "shared BF16 FFN exports with post-SiLU clamp and FP32 product (not existing pre-SiLU clamp kernels)",
            Component::Vision => "official perception encoder QuickGELU/LayerScale, two downsamplers, BF16 projector and crop/token ordering exports",
            Component::Speculator => "three BF16 SWA/dense MTP stage exports and independent heads; shared draft policy and exact rollback/prefix adapter",
            _ => "FP32 residual/zero-centered norm and FP32 sigmoid router with selection-only bias and factor3 primitives",
        };
        Some(Hint { what: format!("Step-5 {} kernels/exports missing", component.label()), how: format!("Add {exports}; gate SM120 + SM121 and exact state before enabling runtime.") })
    }
}

impl FamilyModel for Model {
    fn spec(&self) -> &ModelSpec { &self.spec }
    fn activation(&self, layer: usize, shared: bool) -> Option<FfnActivation> {
        if shared { self.cfg.shared_activation.get(layer) } else { self.cfg.activation.get(layer) }.copied()
    }
    fn accepts(&self, role: &TensorRole, stem: &str, _operand: &mut QuantOperand) -> Result<(), String> {
        if !self.tensors.contains_key(stem) && !self.tensors.contains_key(&format!("{stem}.weight")) {
            return Err(format!("{stem}: unknown Step-5 tensor; add a validated header contract"));
        }
        let hint = STEP5.component_hint(role.component).expect("every component has a hint");
        Err(format!("{stem}: {}: {}", hint.what, hint.how))
    }
    fn placement(&self, _input: &crate::plan::PlanInput, report: &mut crate::plan::PlanReport,
        options: &crate::plan::layout::LayoutOptions, requested: crate::plan::ExpertPlacement)
        -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        Some(Ok(placement::layout(self, report, options, requested)))
    }
}
