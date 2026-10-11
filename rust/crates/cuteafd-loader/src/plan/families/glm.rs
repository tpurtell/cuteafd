//! GLM 5.x: MLA + DeepSeek Sparse Attention indexers (glm_moe_dsa) and the
//! hybrid Kimi-Delta-Attention + DSA variant with mHC (glm5_next). HF names,
//! optionally under `model.language_model.`.
use anyhow::Result;
use serde_json::Value;

use super::{bf16, bf16_or_f32, bf16_or_fp8_block128, describe, fp8_f32_block128, leaf, require};
use crate::families::glm5::{GlmDsaConfig, GlmIndexer};
use crate::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
use crate::plan::checkpoint::{opt_usize_field, usize_field, Checkpoint};
use crate::plan::experts::spark_worlds;
use crate::plan::family::{ConfigError, ExpertContract, Family, FamilyModel, Hint, RuntimeStatus};
use crate::plan::format::{QuantOperand, ScaleEncoding};
use crate::plan::names::indexed;
use crate::plan::spec::*;

pub struct Glm {
    id: &'static str,
    architecture: &'static str,
    runtime: RuntimeStatus,
}

/// GLM 5.x: serve-glm / glm-golden on the glm coordinator programs.
pub static GLM5: Glm =
    Glm { id: "glm5", architecture: "GlmMoeDsaForCausalLM", runtime: RuntimeStatus::Serving };
/// GLM 5.3 Flash: serve-glmf / glmf-golden on the glmf coordinator programs.
pub static GLM5_FLASH: Glm =
    Glm { id: "glm5_flash", architecture: "Glm5NextForConditionalGeneration", runtime: RuntimeStatus::Serving };

fn str_list(config: &Value, key: &str) -> Vec<String> {
    config
        .get(key)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default()
}

fn strip_model_prefix(name: &str) -> Option<&str> {
    name.strip_prefix("model.language_model.").or_else(|| name.strip_prefix("model."))
}

impl Family for Glm {
    fn id(&self) -> &'static str {
        self.id
    }
    fn runtime(&self) -> RuntimeStatus {
        self.runtime
    }

    fn expert_catalog(&self) -> bool {
        true
    }

    fn optional(&self, component: Component) -> bool {
        // Text serving does not require vision or native MTP weights;
        // speculation uses a separate drafter checkpoint.
        matches!(component, Component::Speculator | Component::SpeculatorExpert | Component::Vision)
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|arch| arch == self.architecture)
    }

    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        Ok(Box::new(self.model(checkpoint).map_err(ConfigError::from_anyhow)?))
    }

    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        use Component::*;
        if name == "lm_head.weight" {
            return Some(TensorRole::new(LmHead));
        }
        if name.starts_with("model.visual.") || name.starts_with("visual.") || name.starts_with("model.vision") {
            return Some(TensorRole::new(Vision));
        }
        let name = strip_model_prefix(name)?;
        match name {
            "embed_tokens.weight" => return Some(TensorRole::new(Embedding)),
            "norm.weight" => return Some(TensorRole::new(Norm)),
            _ => {}
        }
        let (layer, rest) = indexed(name, "layers.")?;
        // The layer after the backbone is the native MTP layer.
        if layer >= spec.layers.len() {
            if let Some((expert, _)) = indexed(rest, "mlp.experts.") {
                return Some(TensorRole::expert(SpeculatorExpert, layer, expert));
            }
            return Some(TensorRole::layer(Speculator, layer));
        }
        if let Some((expert, _)) = indexed(rest, "mlp.experts.") {
            return Some(TensorRole::expert(RoutedExpert, layer, expert));
        }
        let component = if rest.starts_with("self_attn.indexer.") {
            Indexer
        } else if rest.starts_with("self_attn.") {
            Attention
        } else if rest.starts_with("mlp.gate.") {
            Router
        } else if rest.starts_with("mlp.shared_experts.") {
            SharedExpert
        } else if rest.starts_with("mlp.") {
            DenseFfn
        } else if rest.starts_with("hc_") {
            HyperConnection
        } else if rest.ends_with("layernorm.weight") {
            Norm
        } else {
            Other
        };
        Some(TensorRole::layer(component, layer))
    }

    fn component_hint_for(&self, _spec: &ModelSpec, component: Component, formats: &[String]) -> Option<Hint> {
        // ModelOpt releases (nvidia/GLM-5.3-NVFP4, nvidia/GLM-5.3-Flash-NVFP4) store the
        // dense parts as NVFP4, per-tensor FP8 or BF16 where the programs read FP8 blocks.
        let modelopt = formats.iter().any(|f| f.starts_with("nvfp4") || f.starts_with("fp8-tensor"));
        if component != Component::RoutedExpert && modelopt {
            return Some(Hint {
                what: format!("{} stored as {} (a ModelOpt release's dense parts)", component.label(), formats.join(", ")),
                how: if self.id == "glm5_flash" {
                    "serve-glmf runs NVFP4 dense MLPs natively; MLA and dense/shared block projections run FP8 \
                     only: the checkpoint's E4M3 with FP32 128x128 scales (or --fp8-snapshot's), else BF16 \
                     quantized to those blocks at load (the only resident copy).".into()
                } else {
                    "PLAN.md Phase 5 S2: serve-glm prefills per-tensor FP8 MLPs as static W8A8 and quantizes BF16 \
                     MLA / shared-expert weights to FP8 blocks at load (CUTEAFD_GLM_BF16=native: the BF16 programs \
                     on the checkpoint's own weights).".into()
                },
            });
        }
        self.component_hint(component)
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        let glmrt = "../glmrt (the GLM-5.3 engine this family ports from)";
        let (what, how) = match (self.id, component) {
            ("glm5_flash", Component::Attention) => (
                "hybrid attention: Kimi Delta Attention (linear) layers plus MLA+DSA layers".to_string(),
                "Runs as the glmf programs (b12x integration/cuteafd/glmf.py: token-sequential KDA \
                 recurrence, no-RoPE MLA over 528-byte FP8 records, pooled indexer) in \
                 cuteafd-daemon src/glmf. Faster prefill: b12x sequence/kda_prefill (chunked) for \
                 the recurrence.".to_string(),
            ),
            ("glm5_flash", Component::Speculator) | ("glm5_flash", Component::SpeculatorExpert) => (
                "native MTP layer 45 (not run); external drafters: dSpark \
                 RedHatAI/GLM-5.3-Flash-speculator.dspark-preview, DFlash2 incoai/GLM-5.3-Flash-DFlash2".to_string(),
                "serve-glmf --draft SNAP drafts with either (run-family picks each checkpoint's measured best); \
                 KDA state rolls back by verify-by-replay of the kept rows (families/glm5_flash).".to_string(),
            ),
            ("glm5_flash", Component::Indexer) => (
                "DSA indexer over 4-token key pools".to_string(),
                "Pool keys are a per-channel softmax over each complete pool of LayerNorm(wk x) weighted by \
                 index_kpool_compress_gate x + ape; score = sum_h w_h relu(q_h . k_pool) / sqrt(128); top \
                 index_topk/kpool pools expand to tokens, plus the open tail pool. Up to index_topk + kpool - 1 \
                 tokens every token is selected, so short contexts are dense causal MLA.".to_string(),
            ),
            ("glm5_flash", Component::RoutedExpert) | ("glm5_flash", Component::SpeculatorExpert) => (
                "top-8 of 288 sigmoid routed experts with SwiGLU clamp 10".to_string(),
                "EXL3 K3/K4 checkpoints: the glm:exl3 recipe at hidden 4096 / inter 2048 / 288 experts with \
                 the clamp enabled (python/tools/aot/package_exl3_aot.py, exl3_cross_sm121.py for Sparks). \
                 FP8 checkpoints: the fp8 routed family (native/cmake/shared/fp8_moe.cmake) at this geometry.".to_string(),
            ),
            (_, Component::Attention) | (_, Component::Indexer) => (
                "MLA with DeepSeek Sparse Attention indexer".to_string(),
                format!("Port from {glmrt}: native/cuda/kernels/dsa_indexer.cu and the real_full \
                 attention/mla.rs + attention/residual/dsa_indexer.rs path. b12x: attention/sparse_mla, \
                 dsa_indexer. Shared indexer layers reuse the previous full layer's selection \
                 (indexer_types)."),
            ),
            (_, Component::RoutedExpert) => (
                "top-8 sigmoid routed experts".to_string(),
                format!("Spark expertd-native is specialized to V4.1 geometry. Export b12x fused_moe for \
                 this geometry (see the model spec) and format; {glmrt} has mixed EXL3 K3/K4 top-8 \
                 dispatch (native/cuda/kernels/b12x_mixed_aot.h, b12x_direct.cu) and loader layouts \
                 (loader exl3_format.rs Glm53Exl3MixedTp4LayerLayout)."),
            ),
            (_, Component::Router) => (
                "sigmoid noaux_tc router with e_score_correction_bias, routed scale 2.5".to_string(),
                format!("{glmrt} native/cuda/kernels/router.cu implements the top-8 path."),
            ),
            (_, Component::Speculator) | (_, Component::SpeculatorExpert) => (
                "native MTP layer; DFlash2 external drafter preferred".to_string(),
                format!("{glmrt} commands/real_full/{{mtp,dflash*}}.rs. DFlash2 (incoai/GLM-5.3-DFlash2) \
                 is the measured best speculator for GLM-5.3."),
            ),
            (_, Component::DenseFfn) | (_, Component::SharedExpert) => (
                "FP8 dense and shared FFN on the coordinator".to_string(),
                "b12x gemm block_fp8_linear covers 128x128 FP8 at M small; reuse V4.1's shared-expert \
                 path with this geometry.".to_string(),
            ),
            (_, Component::HyperConnection) => (
                "mHC hyper-connections".to_string(),
                "V4.1 mHC kernels (v41_hc) apply at this width.".to_string(),
            ),
            _ => return None,
        };
        Some(Hint { what, how })
    }
}

impl Glm {
    /// The spec from the runtime's reader (`GlmDsaConfig` for serve-glm,
    /// `GlmNextConfig` for serve-glmf): layer schedule, MoE and RoPE come from
    /// it; the notes describe the rest of config.json.
    fn model(&self, checkpoint: &Checkpoint) -> Result<GlmModel> {
        let text = checkpoint.text_config();
        let (layer_specs, moe, mtp, cache_cfg) = if self.id == "glm5" {
            let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
            let layers = (0..cfg.layers)
                .map(|layer| LayerSpec {
                    attention: AttentionKind::MlaDsa { indexer: cfg.indexers[layer] == GlmIndexer::Full },
                    ffn: if layer < cfg.first_moe_layer {
                        FfnKind::Dense { intermediate: cfg.dense_intermediate }
                    } else {
                        FfnKind::Moe
                    },
                    rope: Some(RopeSpec { dims: cfg.qk_rope_head_dim, theta: cfg.rope_theta }),
                })
                .collect::<Vec<_>>();
            let moe = MoeSpec {
                experts: cfg.experts,
                top_k: cfg.topk,
                intermediate: cfg.moe_intermediate,
                shared_experts: cfg.shared_experts,
                shared_intermediate: cfg.moe_intermediate,
                scoring: "sigmoid".into(),
                routed_scaling: Some(cfg.routed_scale),
                groups: None,
            };
            let mtp = cfg.mtp_layers;
            (layers, moe, mtp, GlmCacheConfig::Dsa(cfg))
        } else {
            let cfg = GlmNextConfig::from_hf(&checkpoint.config)?;
            let layers = (0..cfg.layers)
                .map(|layer| LayerSpec {
                    attention: match cfg.attention[layer] {
                        GlmNextAttention::Kda => AttentionKind::Kda,
                        GlmNextAttention::Mla => AttentionKind::MlaDsa { indexer: true },
                    },
                    ffn: if cfg.dense[layer] {
                        FfnKind::Dense { intermediate: cfg.dense_intermediate }
                    } else {
                        FfnKind::Moe
                    },
                    rope: None,
                })
                .collect::<Vec<_>>();
            let moe = MoeSpec {
                experts: cfg.experts,
                top_k: cfg.topk,
                intermediate: cfg.moe_intermediate,
                shared_experts: opt_usize_field(text, "n_shared_experts").unwrap_or(0),
                shared_intermediate: cfg.moe_intermediate,
                scoring: "sigmoid".into(),
                routed_scaling: Some(cfg.routed_scale),
                groups: None,
            };
            (layers, moe, opt_usize_field(text, "num_nextn_predict_layers").unwrap_or(0), GlmCacheConfig::Flash(cfg))
        };
        let layer_types = str_list(text, "layer_types");
        let mut notes = Vec::new();
        if let Some(hc) = opt_usize_field(text, "hc_mult") {
            notes.push(format!(
                "mHC width {hc}, Sinkhorn {} iterations, final collapse {}",
                opt_usize_field(text, "hc_sinkhorn_iters").unwrap_or(0),
                if self.id == "glm5_flash" { "unweighted mean" } else { "hc_head" },
            ));
        }
        let rope = opt_usize_field(text, "qk_rope_head_dim").unwrap_or(0);
        let kv_lora = opt_usize_field(text, "kv_lora_rank").unwrap_or(0);
        notes.push(format!(
            "MLA q_lora {} kv_lora {kv_lora} rope {rope}, qk nope {} v {} (absorbed query {}; FP8 latent record \
             {} B: E4M3 + 4 FP32 group scales{})",
            opt_usize_field(text, "q_lora_rank").unwrap_or(0),
            opt_usize_field(text, "qk_nope_head_dim").unwrap_or(0),
            opt_usize_field(text, "v_head_dim").unwrap_or(0),
            kv_lora + rope,
            kv_lora + 4 * (kv_lora / 128) + 2 * rope,
            if rope > 0 { " + BF16 RoPE" } else { ", no RoPE" },
        ));
        let kpool = opt_usize_field(text, "index_kpool").unwrap_or(1);
        notes.push(format!(
            "DSA index top-k {}{}",
            opt_usize_field(text, "index_topk").unwrap_or(0),
            if kpool > 1 {
                format!(
                    " tokens = {} pools of {kpool} (gated softmax pool keys + learned position bias){}; \
                     dense causal up to {} tokens",
                    opt_usize_field(text, "index_topk").unwrap_or(0) / kpool,
                    if text.get("index_kpool_always_select_tail").and_then(Value::as_bool) == Some(true) {
                        ", plus the open tail pool"
                    } else {
                        ""
                    },
                    opt_usize_field(text, "index_topk").unwrap_or(0) + kpool - 1,
                )
            } else {
                String::new()
            },
        ));
        if let Some(linear) = text.get("linear_attn_config") {
            let heads = opt_usize_field(linear, "num_heads").unwrap_or(0);
            let dim = opt_usize_field(linear, "head_dim").unwrap_or(0);
            let kda = layer_types.iter().filter(|t| *t == "linear_attention").count();
            notes.push(format!(
                "KDA {heads} heads x {dim}, short conv {}, gate lower bound {}; recurrent state {:.1} MiB per \
                 sequence ({kda} layers x FP32 {heads}x{dim}x{dim}) plus conv state",
                opt_usize_field(linear, "short_conv_kernel_size").unwrap_or(0),
                linear.get("gate_lower_bound").and_then(Value::as_f64).unwrap_or(0.0),
                (kda * heads * dim * dim * 4) as f64 / (1u64 << 20) as f64,
            ));
        }
        if let Some(limit) = text.get("swiglu_limit").and_then(Value::as_f64) {
            notes.push(format!(
                "SwiGLU clamp {limit} (gate <= {limit}, |up| <= {limit}) in dense, shared and routed experts"
            ));
        }
        let vision = if self.id == "glm5_flash" { glm_flash_vision_resident_weights(checkpoint) }
            else { Err("this GLM family's vision tower is not qualified".into()) };
        Ok(GlmModel { id: self.id, cache_cfg, vision, spec: ModelSpec {
            family: self.id,
            architecture: self.architecture.into(),
            hidden: usize_field(text, "hidden_size")?,
            vocab: usize_field(text, "vocab_size")?,
            layers: layer_specs,
            moe: Some(moe),
            speculator: (mtp > 0).then_some(SpeculatorSpec::NativeMtp { layers: mtp }),
            tables: Vec::new(),
            vision: checkpoint.config.get("vision_config").is_some(),
            notes,
        } })
    }

}

pub(crate) fn glm_flash_vision_tensors() -> Vec<(String, Vec<usize>)> {
    let mut tensors = vec![("patch_embed.proj.weight".into(), vec![1024, 3, 2, 14, 14]),
        ("patch_embed.proj.bias".into(), vec![1024])];
    for layer in 0..24 {
        for (name, shape) in [
            ("attn.qkv.weight", vec![3072, 1024]), ("attn.qkv.bias", vec![3072]),
            ("attn.proj.weight", vec![1024, 1024]), ("attn.proj.bias", vec![1024]),
            ("mlp.gate_proj.weight", vec![4096, 1024]), ("mlp.gate_proj.bias", vec![4096]),
            ("mlp.up_proj.weight", vec![4096, 1024]), ("mlp.up_proj.bias", vec![4096]),
            ("mlp.down_proj.weight", vec![1024, 4096]), ("mlp.down_proj.bias", vec![1024]),
            ("norm1.weight", vec![1024]), ("norm2.weight", vec![1024]),
            ("attn.q_norm.weight", vec![64]), ("attn.k_norm.weight", vec![64]),
        ] { tensors.push((format!("blocks.{layer}.{name}"), shape)); }
    }
    for (name, shape) in [
        ("post_layernorm.weight", vec![1024]), ("downsample.weight", vec![4096, 1024, 2, 2]),
        ("downsample.bias", vec![4096]), ("merger.proj.weight", vec![4096, 4096]),
        ("merger.post_projection_norm.weight", vec![4096]), ("merger.post_projection_norm.bias", vec![4096]),
        ("merger.gate_proj.weight", vec![10240, 4096]), ("merger.up_proj.weight", vec![10240, 4096]),
        ("merger.down_proj.weight", vec![4096, 10240]),
    ] { tensors.push((name.into(), shape)); }
    tensors
}

fn glm_flash_vision_shape(stem: &str) -> Option<Vec<usize>> {
    let name = stem.strip_prefix("model.visual.")?;
    glm_flash_vision_tensors().into_iter().find_map(|(tensor, shape)|
        (tensor.strip_suffix(".weight") == Some(name)).then_some(shape))
}

pub(crate) fn glm_flash_vision_resident_weights(checkpoint: &Checkpoint) -> Result<u64, String> {
    let config = &checkpoint.config;
    let v = &config["vision_config"];
    for (name, expected) in [("depth", 24), ("hidden_size", 1024), ("intermediate_size", 4096),
        ("num_heads", 16), ("out_hidden_size", 4096), ("patch_size", 14), ("temporal_patch_size", 2),
        ("spatial_merge_size", 2), ("projection_intermediate_size", 10240), ("in_channels", 3)] {
        if v[name].as_u64() != Some(expected) { return Err(format!("GLM vision_config.{name} needs {expected}; add a tower kernel")); }
    }
    if config["model_type"] != "glm5_next" || config["text_config"]["hidden_size"] != 4096
        || v["attention_bias"] != true || v["hidden_act"] != "silu"
        || v["rms_norm_eps"].as_f64() != Some(1e-5) || v["swiglu_limit"].as_f64() != Some(10.0)
        || v.get("rope_parameters").is_some_and(|rope| rope["rope_type"] != "axial" || rope["rope_theta"].as_f64() != Some(10000.0)) {
        return Err("GLM Flash vision/LM geometry is not supported by the resident tower".into());
    }
    let tensors: std::collections::BTreeMap<_, _> = checkpoint.tensors.iter()
        .filter_map(|t| t.meta.name.strip_prefix("model.visual.").map(|name| (name, &t.meta))).collect();
    let expected = glm_flash_vision_tensors();
    if tensors.len() != expected.len() { return Err(format!("GLM resident tower needs {} tensors, found {}", expected.len(), tensors.len())); }
    let mut bytes = 0;
    for (name, shape) in expected {
        let tensor = tensors.get(name.as_str()).ok_or_else(|| format!("missing model.visual.{name}"))?;
        if tensor.dtype != cuteafd_core::DType::Bf16 || tensor.shape != shape
            || tensor.byte_length != (shape.iter().product::<usize>() * 2) as u64 {
            return Err(format!("model.visual.{name} needs BF16 {shape:?}"));
        }
        // The native arena promotes vectors to FP32; every supported extent is 256-byte aligned.
        bytes += tensor.byte_length * if shape.len() == 1 { 2 } else { 1 };
    }
    Ok(bytes + 64) // Runtime-generated axial rotary frequencies, not checkpoint data.
}

struct GlmModel {
    id: &'static str,
    spec: ModelSpec,
    cache_cfg: GlmCacheConfig,
    vision: Result<u64, String>,
}

enum GlmCacheConfig {
    Dsa(GlmDsaConfig),
    Flash(GlmNextConfig),
}

/// serve-glm's decode programs read these as FP8 (E4M3 with FP32 128x128
/// scales): the checkpoint's own blocks, a ModelOpt per-tensor FP8 weight
/// under a uniform grid (prefill: static W8A8 on its input_scale), or a
/// ModelOpt release's BF16 weight quantized to blocks at load
/// (`GlmLoader::with_fp8`; CUTEAFD_GLM_BF16=native keeps it BF16 instead).
const GLM5_DECODE_FP8: &[&str] = &["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

impl GlmModel {
    fn glm5(&self, stem: &str, operand: &QuantOperand) -> Result<(), String> {
        let name = leaf(stem);
        let indexer = stem.contains(".indexer.");
        if GLM5_DECODE_FP8.contains(&name) || (indexer && name == "wq_b") {
            // Concatenated FP8 parts (q_a | kv_a, gate | up) must end on whole 128-row blocks.
            let first_of_pair = matches!(name, "q_a_proj" | "gate_proj");
            let per_tensor = operand.encoding == crate::plan::format::Encoding::E4m3
                && operand.scale.as_ref().is_some_and(|s| s.encoding == ScaleEncoding::F32 && s.grid.iter().product::<usize>() == 1);
            let stored = fp8_f32_block128(operand) || per_tensor
                || operand.is_plain(&[crate::plan::format::Encoding::Bf16]);
            return require(stored && operand.matrix().is_some_and(|(rows, cols)| cols % 128 == 0
                && (!first_of_pair || rows % 128 == 0)), || {
                format!("serve-glm's decode programs read {name} as FP8 128x128 blocks (the checkpoint's FP8 blocks, \
                    per-tensor FP8, or BF16 quantized at load{}); found {}",
                    if first_of_pair { "; 128-row multiple" } else { "" }, describe(operand))
            });
        }
        match name {
            "kv_b_proj" | "wk" | "weights_proj" | "lm_head" => bf16_or_fp8_block128(operand, name),
            "gate" => bf16(operand, "the router weight"),
            "e_score_correction_bias" => require(operand.is_plain(&[crate::plan::format::Encoding::F32]),
                || format!("the router bias must be FP32, found {}", describe(operand))),
            _ => bf16(operand, name),
        }
    }

    fn glm5_flash(&self, role: &TensorRole, stem: &str, operand: &QuantOperand) -> Result<(), String> {
        let name = leaf(stem);
        let block_shape = match (&self.cache_cfg, role.layer) {
            (GlmCacheConfig::Flash(cfg), Some(layer)) => match role.component {
                Component::Attention if cfg.attention.get(layer) == Some(&GlmNextAttention::Mla) => match name {
                    "q_a_proj" => Some(vec![cfg.q_lora_rank, cfg.hidden]),
                    "kv_a_proj_with_mqa" => Some(vec![cfg.kv_lora_rank, cfg.hidden]),
                    "q_b_proj" => Some(vec![cfg.heads.checked_mul(cfg.qk_nope_dim)
                        .ok_or_else(|| format!("{stem}: query geometry overflows"))?, cfg.q_lora_rank]),
                    "o_proj" => Some(vec![cfg.hidden, cfg.heads.checked_mul(cfg.v_head_dim)
                        .ok_or_else(|| format!("{stem}: output geometry overflows"))?]),
                    _ => None,
                },
                Component::DenseFfn | Component::SharedExpert => {
                    let inter = if role.component == Component::DenseFfn { cfg.dense_intermediate }
                        else { cfg.moe_intermediate };
                    match name {
                        "gate_proj" | "up_proj" => Some(vec![inter, cfg.hidden]),
                        "down_proj" => Some(vec![cfg.hidden, inter]),
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => None,
        };
        if let Some(shape) = block_shape {
            // The FP8 block programs read the checkpoint's E4M3 blocks, or BF16
            // quantized to 128x128 blocks at load (the only resident copy).
            let bf16_blocks = operand.is_plain(&[crate::plan::format::Encoding::Bf16])
                && shape.iter().all(|&n| n % 128 == 0);
            return require((fp8_f32_block128(operand) || bf16_blocks) && operand.logical == shape, || format!(
                "GLMF {name} runs FP8 128x128 blocks: checkpoint E4M3 with FP32 128x128 scales, or BF16 \
                 quantized to blocks at load, shape {shape:?}; found {}", describe(operand)));
        }
        match name {
            // `absorbed` splits kv_b_proj into w_uk / w_uv from BF16 rows.
            "kv_b_proj" => bf16(operand, "kv_b_proj (absorbed into w_uk / w_uv)"),
            "A_log" | "dt_bias" | "q_conv1d" | "k_conv1d" | "v_conv1d" | "e_score_correction_bias" => {
                bf16_or_f32(operand, name)
            }
            _ if name.starts_with("hc_") => bf16_or_f32(operand, name),
            _ if name.ends_with("norm") || name.ends_with("layernorm") || name == "index_kpool_compress_ape" => {
                bf16(operand, name)
            }
            "gate" => bf16(operand, "the router weight"),
            _ if operand.matrix().is_some() => bf16_or_fp8_block128(operand, name),
            _ => bf16(operand, name),
        }
    }

    fn routed(&self, stem: &str, operand: &QuantOperand) -> Result<(), String> {
        let moe = self.spec.moe.as_ref().ok_or("no MoE geometry")?;
        let shape = super::deepseek::routed_shape(stem, self.spec.hidden, moe.intermediate)
            .ok_or("not a routed projection (gate/up/down_proj)")?;
        require(operand.logical == shape, || format!("experts are {shape:?}, found {}", describe(operand)))?;
        let exl3 = if self.id == "glm5" { 4..=5 } else { 3..=4 };
        require(operand.is_fp8_block(128, &[ScaleEncoding::F32, ScaleEncoding::Bf16])
            || operand.exl3_bits().is_some_and(|bits| exl3.contains(&bits))
            || operand.is_nvfp4() && operand.scale.as_ref().is_some_and(|s| s.cols == 16), || {
            format!("routed experts run E4M3 with 128x128 scales, EXL3 K{}-K{} or ModelOpt NVFP4 (group 16); found {}",
                exl3.start(), exl3.end(), describe(operand))
        })
    }

    fn geometry(&self) -> &'static str {
        if self.id == "glm5" { "glm" } else { "glmf" }
    }
}

impl FamilyModel for GlmModel {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn cache_geometry(&self, options: crate::serving_capacity::CacheOptions)
        -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        use crate::serving_capacity::{glm_cache_geometry, glm_flash_rank_cache_geometry_rows, CacheGeometryError};
        if options.native_mtp_layers > 0 {
            return Err(CacheGeometryError::Unsupported { family: self.id, what: "native MTP is not executed; reserve DFlash separately" });
        }
        match &self.cache_cfg {
            GlmCacheConfig::Dsa(cfg) => glm_cache_geometry(cfg, cfg.layers, options.coordinator_ranks).map(Some),
            GlmCacheConfig::Flash(cfg) => glm_flash_rank_cache_geometry_rows(cfg, cfg.layers, options.coordinator_ranks,
                options.glmf_index, options.kda_state_bytes, options.glmf_decode_rows).map(Some),
        }
    }

    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        match role.component {
            Component::RoutedExpert => self.routed(stem, operand),
            Component::Speculator | Component::SpeculatorExpert => {
                Err("the native MTP layer is not run (speculation uses a DFlash2 drafter)".into())
            }
            Component::Vision => {
                self.vision.clone()?;
                let shape = glm_flash_vision_shape(stem).ok_or_else(|| format!("{stem} is not read by the resident GLM Flash tower"))?;
                require(operand.is_plain(&[crate::plan::format::Encoding::Bf16]) && operand.logical == shape, ||
                    format!("{stem} needs BF16 {shape:?}, found {}", describe(operand)))
            },
            // ModelOpt NVFP4 dense MLPs run natively (serve-glmf: the fp8-glmfdense-nvfp4 package).
            Component::DenseFfn if self.id == "glm5_flash" && operand.is_nvfp4()
                && operand.scale.as_ref().is_some_and(|s| s.cols == 16) => Ok(()),
            _ if self.id == "glm5" => self.glm5(stem, operand),
            _ => self.glm5_flash(role, stem, operand),
        }
    }

    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        let moe = self.spec.moe.as_ref()?;
        let geometry = cuteafd_core::ExpertGeometry {
            hidden: self.spec.hidden as u32,
            experts: moe.experts as u32,
            topk: moe.top_k as u32,
            intermediate: moe.intermediate as u32,
            layers: 0,
        };
        let expected = if self.id == "glm5" { cuteafd_core::ExpertGeometry::GLM5 } else { cuteafd_core::ExpertGeometry::GLM5_FLASH };
        if !geometry.same_shape(&expected) {
            return None;
        }
        let local = if self.id == "glm5" {
            Err("serve-glm has no local expert path (no --local-experts)".to_string())
        } else {
            Ok("serve-glmf --local-experts (TP1 FP8, NVFP4 or EXL3 package)".to_string())
        };
        if operand.is_nvfp4() {
            return Some(ExpertContract {
                package: format!("{}:nvfp4", self.geometry()),
                block: 16,
                spark_worlds: spark_worlds(&format!("{}:nvfp4", self.geometry()), moe.intermediate),
                local,
            });
        }
        if operand.exl3_bits().is_some() {
            let tiers = if self.id == "glm5" { "k45" } else { "k34" };
            return Some(ExpertContract {
                package: format!("{}:exl3-{tiers}", self.geometry()),
                block: 128,
                spark_worlds: spark_worlds(&format!("{}:exl3-{tiers}", self.geometry()), moe.intermediate),
                local,
            });
        }
        operand.is_fp8_block(128, &[ScaleEncoding::F32, ScaleEncoding::Bf16]).then(|| ExpertContract {
            package: format!("{}:fp8", self.geometry()),
            block: 128,
            spark_worlds: spark_worlds(&format!("{}:fp8", self.geometry()), moe.intermediate),
            local,
        })
    }
}
