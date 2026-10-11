//! Checkpoint fixtures for planner tests (this crate's and the daemon's CLI
//! tests): headers with the real geometry of each family over sparse,
//! all-zero payloads, so a fixture costs no disk.
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::Path;

/// (name, safetensors dtype, shape).
pub type Tensor = (String, &'static str, Vec<usize>);

pub fn t(name: impl Into<String>, dtype: &'static str, shape: &[usize]) -> Tensor {
    (name.into(), dtype, shape.to_vec())
}

fn width(dtype: &str) -> u64 {
    match dtype {
        "BF16" | "F16" | "I16" => 2,
        "F32" | "I32" => 4,
        "I64" => 8,
        _ => 1,
    }
}

/// One safetensors file whose payload is a hole of the right length.
pub fn write_safetensors(path: &Path, tensors: &[Tensor]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, dtype, shape) in tensors {
        let bytes = shape.iter().product::<usize>() as u64 * width(dtype);
        header.insert(name.clone(), json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}));
        offset += bytes;
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&header).unwrap();
    file.set_len(8 + header.len() as u64 + offset).unwrap();
}

/// `config.json`, one shard and its index (with `metadata.tp_size` when given).
pub fn write_snapshot(dir: &Path, config: &Value, tensors: &[Tensor], tp_size: Option<usize>) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
    write_safetensors(&dir.join("model-00001-of-00001.safetensors"), tensors);
    let weight_map: serde_json::Map<String, Value> =
        tensors.iter().map(|(name, ..)| (name.clone(), json!("model-00001-of-00001.safetensors"))).collect();
    let mut index = json!({"weight_map": weight_map});
    if let Some(tp) = tp_size {
        index["metadata"] = json!({"tp_size": tp});
    }
    fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&index).unwrap()).unwrap();
}

/// E4M3 `[n, k]` with FP32 128x128 block scales (`grid_rows` scale rows, by
/// default uniform).
pub fn fp8(name: &str, n: usize, k: usize, grid_rows: Option<usize>) -> Vec<Tensor> {
    vec![
        t(format!("{name}.weight"), "F8_E4M3", &[n, k]),
        t(format!("{name}.weight_scale_inv"), "F32", &[grid_rows.unwrap_or(n.div_ceil(128)), k.div_ceil(128)]),
    ]
}

/// MXFP4 `[n, k]`: packed E2M1 U8 `[n, k/2]`, UE8M0 U8 `[n, k/32]`.
pub fn mxfp4(name: &str, n: usize, k: usize) -> Vec<Tensor> {
    vec![t(format!("{name}.weight"), "U8", &[n, k / 2]), t(format!("{name}.weight_scale"), "U8", &[n, k / 32])]
}

/// ModelOpt NVFP4 `[n, k]`: packed E2M1 U8 `[n, k/2]`, E4M3 `[n, k/16]`, FP32
/// `weight_scale_2` and `input_scale` scalars.
pub fn nvfp4(name: &str, n: usize, k: usize) -> Vec<Tensor> {
    vec![
        t(format!("{name}.weight"), "U8", &[n, k / 2]),
        t(format!("{name}.weight_scale"), "F8_E4M3", &[n, k / 16]),
        t(format!("{name}.weight_scale_2"), "F32", &[]),
        t(format!("{name}.input_scale"), "F32", &[]),
    ]
}

/// A complete EXL3 projection `[n, k]` at `bits`.
pub fn exl3(name: &str, n: usize, k: usize, bits: usize) -> Vec<Tensor> {
    vec![
        t(format!("{name}.trellis"), "I16", &[k / 16, n / 16, 16 * bits]),
        t(format!("{name}.suh"), "F16", &[k]),
        t(format!("{name}.svh"), "F16", &[n]),
        t(format!("{name}.mcg"), "I32", &[]),
    ]
}

/// MiMo V2 Flash's config with two layers (full + dense, SWA + MoE) in the
/// hub spelling.
pub fn mimo_flash_config() -> Value {
    let mut config = json!({
        "architectures": ["MiMoV2FlashForCausalLM"], "model_type": "mimo_v2_flash", "vocab_size": 64,
        "hidden_size": 4096, "num_hidden_layers": 2, "max_position_embeddings": 1048576,
        "num_attention_heads": 64, "num_key_value_heads": 4,
        "head_dim": 192, "v_head_dim": 128, "swa_num_attention_heads": 64, "swa_num_key_value_heads": 8,
        "swa_head_dim": 192, "swa_v_head_dim": 128, "partial_rotary_factor": 0.334, "rope_theta": 5000000,
        "swa_rope_theta": 10000, "sliding_window": 128, "sliding_window_size": 128,
        "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1], "add_swa_attention_sink_bias": true,
        "add_full_attention_sink_bias": false, "intermediate_size": 16384, "n_routed_experts": 256,
        "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "routed_scaling_factor": null,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1, "norm_topk_prob": true,
        "layernorm_epsilon": 1e-5, "attention_value_scale": 0.707
    });
    config["quantization_config"] = FP8_BLOCK.clone();
    config
}

fn mimo_common(h: usize, vocab: usize) -> Vec<Tensor> {
    vec![
        t("lm_head.weight", "BF16", &[vocab, h]),
        t("model.embed_tokens.weight", "BF16", &[vocab, h]),
        t("model.norm.weight", "BF16", &[h]),
    ]
}

/// Every tensor serve-mimo stages for `mimo_flash_config` (one expert per layer).
pub fn mimo_flash_tensors() -> Vec<Tensor> {
    let h = 4096;
    let mut out = mimo_common(h, 64);
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        out.push(t(format!("{p}.input_layernorm.weight"), "BF16", &[h]));
        out.push(t(format!("{p}.post_attention_layernorm.weight"), "BF16", &[h]));
        out.extend(fp8(&format!("{p}.self_attn.q_proj"), 64 * 192, h, None));
        out.push(t(format!("{p}.self_attn.o_proj.weight"), "BF16", &[h, 64 * 128]));
        if layer == 0 {
            // Full layer: 4 KV heads; k_proj's grid restarts per 192-row head.
            out.extend(fp8(&format!("{p}.self_attn.k_proj"), 4 * 192, h, Some(8)));
            out.extend(fp8(&format!("{p}.self_attn.v_proj"), 4 * 128, h, None));
            out.extend(fp8(&format!("{p}.mlp.gate_proj"), 16384, h, None));
            out.extend(fp8(&format!("{p}.mlp.up_proj"), 16384, h, None));
            out.extend(fp8(&format!("{p}.mlp.down_proj"), h, 16384, None));
        } else {
            out.extend(fp8(&format!("{p}.self_attn.k_proj"), 8 * 192, h, None));
            out.extend(fp8(&format!("{p}.self_attn.v_proj"), 8 * 128, h, None));
            out.push(t(format!("{p}.self_attn.attention_sink_bias"), "BF16", &[64]));
            out.push(t(format!("{p}.mlp.gate.weight"), "F32", &[256, h]));
            out.push(t(format!("{p}.mlp.gate.e_score_correction_bias"), "F32", &[256]));
            out.extend(fp8(&format!("{p}.mlp.experts.0.gate_proj"), 2048, h, None));
            out.extend(fp8(&format!("{p}.mlp.experts.0.up_proj"), 2048, h, None));
            out.extend(fp8(&format!("{p}.mlp.experts.0.down_proj"), h, 2048, None));
        }
    }
    out
}

/// V2.6 Flash MOPD: Flash shapes with BF16 routing and MXFP4 experts.
pub fn mimo_flash_mopd_config() -> Value {
    let mut config = mimo_flash_config();
    config["architectures"] = json!(["MiMoV2ForCausalLM"]);
    config["model_type"] = json!("mimo_v2");
    config["rope_theta"] = json!(1e7);
    config["layernorm_epsilon"] = json!(1e-6);
    config["moe_router_dtype"] = json!("bfloat16");
    config["attention_projection_layout"] = json!("fused_qkv");
    config["quantization_config"]["store_dtype"] = json!("mxfp4");
    config["quantization_config"]["mxfp4_block_size"] = json!(32);
    config
}

/// TP4 fused QKV: full has one KV head per shard; SWA has two.
pub fn mimo_flash_mopd_tensors() -> Vec<Tensor> {
    let mut out = mimo_flash_tensors();
    out.retain(|(name, ..)| !["q_proj", "k_proj", "v_proj"].iter()
        .any(|proj| name.contains(&format!("self_attn.{proj}."))) && !name.contains(".experts."));
    for layer in 0..2 {
        let (rows, scales) = if layer == 0 { (13568, 108) } else { (14848, 116) };
        out.extend(fp8(&format!("model.layers.{layer}.self_attn.qkv_proj"), rows, 4096, Some(scales)));
    }
    let router = out.iter_mut().find(|(name, ..)| name == "model.layers.1.mlp.gate.weight").unwrap();
    router.1 = "BF16";
    for (proj, n, k) in [("gate_proj", 2048, 4096), ("up_proj", 2048, 4096), ("down_proj", 4096, 2048)] {
        out.extend(mxfp4(&format!("model.layers.1.mlp.experts.0.{proj}"), n, k));
    }
    out
}

/// MiMo V2.6 Pro's config with two layers (full + dense, SWA + MoE).
pub fn mimo_pro_config() -> Value {
    let mut config = json!({
        "architectures": ["MiMoV2ForCausalLM"], "model_type": "mimo_v2", "vocab_size": 64, "hidden_size": 6144,
        "num_hidden_layers": 2, "max_position_embeddings": 1048576,
        "num_attention_heads": 128, "num_key_value_heads": 8, "head_dim": 192,
        "v_head_dim": 128, "swa_num_attention_heads": 128, "swa_num_key_value_heads": 8, "swa_head_dim": 192,
        "swa_v_head_dim": 128, "partial_rotary_factor": 0.334, "rope_theta": 10000000, "swa_rope_theta": 10000,
        "sliding_window": 128, "sliding_window_size": 128, "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1],
        "add_swa_attention_sink_bias": true, "add_full_attention_sink_bias": false, "intermediate_size": 16384,
        "n_routed_experts": 384, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
        "routed_scaling_factor": null, "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1,
        "norm_topk_prob": true, "layernorm_epsilon": 1e-5, "attention_value_scale": 0.612
    });
    config["quantization_config"] = FP8_BLOCK.clone();
    config["quantization_config"]["store_dtype"] = json!("mxfp4");
    config["quantization_config"]["mxfp4_block_size"] = json!(32);
    config
}

/// Every tensor serve-mimo stages for `mimo_pro_config`: fused TP8 qkv
/// (27136 rows over a 216-row grid), MXFP4 experts.
pub fn mimo_pro_tensors() -> Vec<Tensor> {
    let h = 6144;
    let mut out = mimo_common(h, 64);
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        out.push(t(format!("{p}.input_layernorm.weight"), "BF16", &[h]));
        out.push(t(format!("{p}.post_attention_layernorm.weight"), "BF16", &[h]));
        out.extend(fp8(&format!("{p}.self_attn.qkv_proj"), 27136, h, Some(216)));
        out.push(t(format!("{p}.self_attn.o_proj.weight"), "BF16", &[h, 128 * 128]));
        if layer == 0 {
            out.extend(fp8(&format!("{p}.mlp.gate_proj"), 16384, h, None));
            out.extend(fp8(&format!("{p}.mlp.up_proj"), 16384, h, None));
            out.extend(fp8(&format!("{p}.mlp.down_proj"), h, 16384, None));
        } else {
            out.push(t(format!("{p}.self_attn.attention_sink_bias"), "BF16", &[128]));
            out.push(t(format!("{p}.mlp.gate.weight"), "BF16", &[384, h]));
            out.push(t(format!("{p}.mlp.gate.e_score_correction_bias"), "F32", &[384]));
            out.extend(mxfp4(&format!("{p}.mlp.experts.0.gate_proj"), 2048, h));
            out.extend(mxfp4(&format!("{p}.mlp.experts.0.up_proj"), 2048, h));
            out.extend(mxfp4(&format!("{p}.mlp.experts.0.down_proj"), h, 2048));
        }
    }
    out
}

/// GLM 5.3's config with two layers (dense, MoE with a shared indexer).
pub fn glm5_config() -> Value {
    json!({
        "architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "vocab_size": 64,
        "hidden_size": 6144, "num_hidden_layers": 2, "num_attention_heads": 64, "q_lora_rank": 2048,
        "kv_lora_rank": 512, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048, "indexer_types": ["full", "shared"],
        "first_k_dense_replace": 1, "mlp_layer_types": ["dense", "sparse"], "intermediate_size": 12288,
        "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "n_shared_experts": 1,
        "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1,
        "rope_interleave": true, "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"}, "num_nextn_predict_layers": 0,
        "eos_token_id": [154820], "quantization_config": FP8_BLOCK.clone()
    })
}

/// Every tensor serve-glm stages for `glm5_config`; `expert` supplies layer
/// 1's routed expert 0 (`model.layers.1.mlp.experts.0`).
pub fn glm5_tensors(expert: impl Fn(&str, usize, usize) -> Vec<Tensor>) -> Vec<Tensor> {
    let h = 6144;
    let mut out = vec![
        t("lm_head.weight", "BF16", &[64, h]),
        t("model.embed_tokens.weight", "BF16", &[64, h]),
        t("model.norm.weight", "BF16", &[h]),
    ];
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            out.push(t(format!("{p}.{norm}.weight"), "BF16", &[h]));
        }
        out.push(t(format!("{p}.self_attn.q_a_layernorm.weight"), "BF16", &[2048]));
        out.push(t(format!("{p}.self_attn.kv_a_layernorm.weight"), "BF16", &[512]));
        out.extend(fp8(&format!("{p}.self_attn.q_a_proj"), 2048, h, None));
        out.extend(fp8(&format!("{p}.self_attn.kv_a_proj_with_mqa"), 576, h, None));
        out.extend(fp8(&format!("{p}.self_attn.q_b_proj"), 64 * 256, 2048, None));
        out.extend(fp8(&format!("{p}.self_attn.kv_b_proj"), 64 * 448, 512, None));
        out.extend(fp8(&format!("{p}.self_attn.o_proj"), h, 64 * 256, None));
        if layer == 0 {
            out.extend(fp8(&format!("{p}.self_attn.indexer.wq_b"), 4096, 2048, None));
            out.extend(fp8(&format!("{p}.self_attn.indexer.wk"), 128, h, None));
            out.push(t(format!("{p}.self_attn.indexer.weights_proj.weight"), "BF16", &[32, h]));
            out.push(t(format!("{p}.self_attn.indexer.k_norm.weight"), "BF16", &[128]));
            out.push(t(format!("{p}.self_attn.indexer.k_norm.bias"), "BF16", &[128]));
            out.extend(fp8(&format!("{p}.mlp.gate_proj"), 12288, h, None));
            out.extend(fp8(&format!("{p}.mlp.up_proj"), 12288, h, None));
            out.extend(fp8(&format!("{p}.mlp.down_proj"), h, 12288, None));
        } else {
            out.push(t(format!("{p}.mlp.gate.weight"), "BF16", &[256, h]));
            out.push(t(format!("{p}.mlp.gate.e_score_correction_bias"), "F32", &[256]));
            out.extend(fp8(&format!("{p}.mlp.shared_experts.gate_proj"), 2048, h, None));
            out.extend(fp8(&format!("{p}.mlp.shared_experts.up_proj"), 2048, h, None));
            out.extend(fp8(&format!("{p}.mlp.shared_experts.down_proj"), h, 2048, None));
            for (proj, n, k) in [("gate_proj", 2048, h), ("up_proj", 2048, h), ("down_proj", h, 2048)] {
                out.extend(expert(&format!("{p}.mlp.experts.0.{proj}"), n, k));
            }
        }
    }
    out
}

/// GLM 5.3 Flash's config with `layers` layers (KDA / MLA alternating, the
/// first dense).
pub fn glm5_flash_config(layers: usize) -> Value {
    json!({
        "architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next",
        "quantization_config": FP8_BLOCK.clone(), "text_config": {
            "model_type": "glm5_next_text", "vocab_size": 64, "hidden_size": 4096, "num_hidden_layers": layers,
            "layer_types": (0..layers).map(|l| if l % 2 == 1 { "deepseek_sparse_attention" } else { "linear_attention" })
                .collect::<Vec<_>>(),
            "mlp_layer_types": (0..layers).map(|l| if l == 0 { "dense" } else { "sparse" }).collect::<Vec<_>>(),
            "intermediate_size": 12288, "n_routed_experts": 288, "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048, "routed_scaling_factor": 2.5, "swiglu_limit": 10.0, "rms_norm_eps": 1e-5,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "mla_use_nope": true, "qk_rope_head_dim": 0,
            "num_attention_heads": 64, "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 256,
            "v_head_dim": 256, "index_topk": 2048, "index_kpool": 4, "index_kpool_always_select_tail": true,
            "eos_token_id": [154820],
            "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4,
                                   "gate_lower_bound": -5.0}}
    })
}

/// Complete coordinator headers plus one expert per routed layer; payloads stay sparse.
pub fn glm5_flash_tensors(config: &Value) -> Vec<Tensor> {
    use crate::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
    let cfg = GlmNextConfig::from_hf(config).unwrap();
    let h = cfg.hidden;
    let mut out = vec![t("model.language_model.embed_tokens.weight", "BF16", &[cfg.vocab_size, h]),
        t("model.language_model.norm.weight", "BF16", &[h]), t("lm_head.weight", "BF16", &[cfg.vocab_size, h])];
    for layer in 0..cfg.layers {
        let p = format!("model.language_model.layers.{layer}");
        for site in ["attn", "ffn"] { for suffix in ["fn", "scale", "base"] {
            out.push(t(format!("{p}.hc_{site}_{suffix}"), "F32", &[4]));
        } }
        for norm in ["input_layernorm", "post_attention_layernorm"] { out.push(t(format!("{p}.{norm}.weight"), "BF16", &[h])); }
        let a = format!("{p}.self_attn");
        if cfg.attention[layer] == GlmNextAttention::Kda {
            for proj in ["q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj", "o_proj", "f_b_proj", "g_b_proj"] {
                out.push(t(format!("{a}.{proj}.weight"), "BF16", &[h, h]));
            }
            for proj in ["q", "k", "v"] { out.push(t(format!("{a}.{proj}_conv1d.weight"), "F32", &[h, 4])); }
            for suffix in ["A_log", "dt_bias"] { out.push(t(format!("{a}.{suffix}"), "F32", &[cfg.kda_heads])); }
            out.push(t(format!("{a}.o_norm.weight"), "BF16", &[cfg.kda_head_dim]));
        } else {
            for suffix in ["q_a_layernorm.weight", "kv_a_layernorm.weight", "indexer.wq_b.weight", "indexer.wk.weight", "indexer.weights_proj.weight", "indexer.index_kpool_compress_gate", "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.index_kpool_compress_ape"] {
                out.push(t(format!("{a}.{suffix}"), "BF16", &[128]));
            }
            for (proj, n, k) in [("q_a_proj", cfg.q_lora_rank, h), ("kv_a_proj_with_mqa", cfg.kv_lora_rank, h),
                ("q_b_proj", cfg.heads * cfg.qk_nope_dim, cfg.q_lora_rank), ("o_proj", h, cfg.heads * cfg.v_head_dim)] {
                out.extend(fp8(&format!("{a}.{proj}"), n, k, None));
            }
        }
        let mlp = if cfg.dense[layer] { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
        let width = if cfg.dense[layer] { cfg.dense_intermediate } else { cfg.moe_intermediate };
        for proj in ["gate_proj", "up_proj"] { out.extend(fp8(&format!("{mlp}.{proj}"), width, h, None)); }
        out.extend(fp8(&format!("{mlp}.down_proj"), h, width, None));
        if !cfg.dense[layer] {
            out.push(t(format!("{p}.mlp.gate.weight"), "BF16", &[cfg.experts, h]));
            out.push(t(format!("{p}.mlp.gate.e_score_correction_bias"), "F32", &[cfg.experts]));
            for proj in ["gate_proj", "up_proj"] { out.extend(fp8(&format!("{p}.mlp.experts.0.{proj}"), cfg.moe_intermediate, h, None)); }
            out.extend(fp8(&format!("{p}.mlp.experts.0.down_proj"), h, cfg.moe_intermediate, None));
        }
    }
    out
}

/// Qwen 3.8 Flash Next's config with `layers` layers (every fourth full attention).
pub fn qwen4_config(layers: usize) -> Value {
    json!({
        "architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
        "quantization_config": exl3_compact(4), "text_config": {
            "model_type": "qwen4_exp_text", "vocab_size": 64, "hidden_size": 2560, "num_hidden_layers": layers,
            "layer_types": (0..layers).map(|l| if l % 4 == 3 { "full_attention" } else { "linear_attention" })
                .collect::<Vec<_>>(),
            "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640, "rms_norm_eps": 1e-6, "hc_count": 4, "hc_lowrank": 320,
            "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
            "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
            "indexer_n_heads": 4, "indexer_head_dim": 128, "indexer_budget": 2048, "indexer_compress_ratio": 4,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
            "linear_conv_kernel_dim": 4, "eos_token_id": 248044, "output_gate_type": "sigmoid",
            "mtp_num_hidden_layers": 0}
    })
}

/// Every Qwen routed expert of every layer as complete EXL3 projections at
/// `bits`, a BF16 router per layer, and the storage map.
pub fn qwen4_exl3(layers: usize, bits: usize) -> (Vec<Tensor>, Value) {
    let mut out = Vec::new();
    let mut projections = Vec::new();
    for layer in 0..layers {
        out.push(t(format!("model.language_model.layers.{layer}.mlp.gate.weight"), "BF16", &[512, 2560]));
        for expert in 0..512 {
            for (proj, n, k) in [("gate_proj", 640, 2560), ("up_proj", 640, 2560), ("down_proj", 2560, 640)] {
                let name = format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}");
                out.extend(exl3(&name, n, k, bits));
                projections.push((name, bits, k, n));
            }
        }
    }
    (out, exl3_manifest(&exl3_compact(bits), &projections))
}

/// HF block-FP8 quantization (`quant_method: fp8`, 128x128 blocks).
pub static FP8_BLOCK: std::sync::LazyLock<Value> = std::sync::LazyLock::new(|| {
    json!({"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [128, 128]})
});

/// The compact `quantization_config` of a GPTQModel EXL3 publication.
pub fn exl3_compact(bits: usize) -> Value {
    json!({"bits": bits as f64, "checkpoint_format": "exl3", "codebook": "mcg", "desc_act": false, "format": "exl3",
           "group_size": -1, "method": "exl3", "out_scales": "auto", "pack_dtype": "int32", "quant_method": "exl3"})
}

/// `quantize_config.json`: the compact fields plus the storage map of every
/// projection (name, bits, input K, output N).
pub fn exl3_manifest(compact: &Value, projections: &[(String, usize, usize, usize)]) -> Value {
    let mut manifest = compact.clone();
    let storage: serde_json::Map<String, Value> = projections
        .iter()
        .map(|(name, bits, k, n)| {
            (name.clone(), json!({"quant_format": "exl3", "bits_per_weight": bits, "stored_tensors": {
                format!("{name}.trellis"): {"torch_dtype": "int16", "shape": [k / 16, n / 16, 16 * bits]},
                format!("{name}.suh"): {"torch_dtype": "float16", "shape": [k]},
                format!("{name}.svh"): {"torch_dtype": "float16", "shape": [n]},
                format!("{name}.mcg"): {"torch_dtype": "int32", "shape": []}}}))
        })
        .collect();
    manifest["tensor_storage"] = Value::Object(storage);
    manifest
}

pub fn write_quantize_config(dir: &Path, manifest: &Value) {
    fs::write(dir.join("quantize_config.json"), serde_json::to_vec(manifest).unwrap()).unwrap();
}

pub fn write_v4_snapshot(path: &Path) {
    write_v4_snapshot_with_dspark(path, 0);
}

pub fn write_v4_snapshot_with_dspark(path: &Path, stages: usize) {
    // serve-dsv4 reads inference/config.json when the snapshot has one; so does the plan.
    let hf = json!({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"});
    let mut ratios = vec![0, 0];
    ratios.extend([4, 128, 4]);
    let args = json!({
        "vocab_size": 64, "dim": 4096, "moe_inter_dim": 2048, "n_layers": 4, "n_hash_layers": 1, "n_heads": 64,
        "n_routed_experts": 256, "n_shared_experts": 1, "n_activated_experts": 6, "score_func": "sqrtsoftplus",
        "route_scale": 1.5, "swiglu_limit": 10.0, "q_lora_rank": 1024, "head_dim": 512, "rope_head_dim": 64,
        "o_groups": 8, "o_lora_rank": 1024, "window_size": 128, "original_seq_len": 65536, "rope_theta": 10000,
        "rope_factor": 16, "beta_fast": 32, "beta_slow": 1, "index_n_heads": 64, "index_head_dim": 128,
        "index_topk": 512, "hc_mult": 4, "hc_sinkhorn_iters": 20, "compress_rope_theta": 160000,
        "compress_ratios": ratios[1..].to_vec()
    });
    let mut tensors = vec![t("embed.weight", "BF16", &[64, 4096])];
    for layer in 0..4 {
        for expert in 0..256 {
            for (projection, rows, cols) in [("w1", 2048, 4096), ("w2", 4096, 2048), ("w3", 2048, 4096)] {
                let name = format!("layers.{layer}.ffn.experts.{expert}.{projection}");
                tensors.push(t(format!("{name}.weight"), "I8", &[rows, cols / 2]));
                tensors.push(t(format!("{name}.scale"), "F8_E8M0", &[rows, cols / 32]));
            }
        }
    }
    for stage in 0..stages {
        for expert in 0..256 {
            for (projection, rows, cols) in [("w1", 2048, 4096), ("w2", 4096, 2048), ("w3", 2048, 4096)] {
                let name = format!("mtp.{stage}.ffn.experts.{expert}.{projection}");
                tensors.push(t(format!("{name}.weight"), "I8", &[rows, cols / 2]));
                tensors.push(t(format!("{name}.scale"), "F8_E8M0", &[rows, cols / 32]));
            }
        }
    }
    write_snapshot(path, &hf, &tensors, None);
    std::fs::create_dir_all(path.join("inference")).unwrap();
    std::fs::write(path.join("inference/config.json"), serde_json::to_vec(&args).unwrap()).unwrap();
}
