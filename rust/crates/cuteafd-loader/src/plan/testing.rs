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
        "hidden_size": 6144, "num_hidden_layers": 2, "max_position_embeddings": 1048576,
        "num_attention_heads": 64, "q_lora_rank": 2048,
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

/// Qwen's coordinator headers for `layers` target layers (and one native MTP
/// layer with `mtp`): exactly what `checkpoint_resident_bytes` packs, so a
/// fixture's plan charges the serve-qwen4 resident inventory.
pub fn qwen4_coordinator(config: &Value, mtp: bool) -> Vec<Tensor> {
    use crate::families::qwen4::{resident::layer_projections, Qwen4Attention, Qwen4Config};
    let cfg = Qwen4Config::from_hf(config).unwrap();
    let h = cfg.hidden;
    let mut out = vec![t("model.language_model.embed_tokens.weight", "BF16", &[cfg.vocab_size, h]),
        t("lm_head.weight", "BF16", &[cfg.vocab_size, h])];
    let layer = |out: &mut Vec<Tensor>, p: String, attention: Qwen4Attention| {
        // Both selectable projections as one BF16 matrix each, at their packed shapes.
        for projection in layer_projections(&cfg, attention, None) {
            out.push(t(format!("{p}.self_attn.{}.weight", projection.operand), "BF16", &[projection.rows, projection.cols]));
        }
        let s = cfg.shared_intermediate;
        out.extend([t(format!("{p}.mlp.shared_expert.gate_proj.weight"), "BF16", &[s, h]),
            t(format!("{p}.mlp.shared_expert.up_proj.weight"), "BF16", &[s, h]),
            t(format!("{p}.mlp.shared_expert_gate.weight"), "BF16", &[1, h]),
            t(format!("{p}.mlp.shared_expert.down_proj.weight"), "BF16", &[h, s])]);
    };
    for l in 0..cfg.layers {
        layer(&mut out, format!("model.language_model.layers.{l}"), cfg.attention[l]);
    }
    if mtp { layer(&mut out, "mtp.layers.0".into(), Qwen4Attention::Full); }
    out
}

/// TP1 EXL3 capacity manifests for Qwen's routed geometry under `share/exl3`,
/// as images bundle them beside PROGRAMS.json; returns that PROGRAMS.json.
pub fn write_qwen4_exl3_capacities(dir: &Path, tiers: &str) -> std::path::PathBuf {
    let share = dir.join("share");
    for capacity in crate::placement::inventory::exl3_capacities(4096) {
        let path = share.join("exl3").join(format!("exl3-qwen4-k{tiers}")).join(format!("rtx-tp1/m{capacity}"));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&json!({
            "capacity": capacity, "hidden": 2560, "experts": 512, "intermediate": 640, "top_k": 10,
            "input_format": "e4m3_k32", "trellis_lut": {"bytes": 32}, "buffers": {
                "scratch": {"allocation": "scratch", "dtype": "f32", "bytes": capacity * 8, "zero_on_create": false}}
        })).unwrap()).unwrap();
    }
    let programs = share.join("PROGRAMS.json");
    fs::write(&programs, b"{}").unwrap();
    programs
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
        "compress_ratios": ratios[1..].to_vec(), "dspark_block_size": if stages > 0 { 5 } else { 0 },
        "dspark_target_layer_ids": [1, 2, 3], "dspark_markov_rank": 256
    });
    let cfg = crate::families::deepseek_v4::DeepseekV4Config::from_model_args(&args, stages).unwrap();
    let mut tensors = vec![t("embed.weight", "BF16", &[cfg.vocab_size, cfg.dim]),
        t("head.weight", "BF16", &[cfg.vocab_size, cfg.dim]),
        t("hc_head_fn", "F32", &[4, 4 * cfg.dim]), t("hc_head_scale", "F32", &[1]),
        t("hc_head_base", "F32", &[4]), t("norm.weight", "BF16", &[cfg.dim])];
    for layer in 0..cfg.n_layers {
        tensors.extend(v4_block(&cfg, &format!("layers.{layer}"), cfg.compress_ratios[layer], cfg.is_hash_layer(layer)));
    }
    for stage in 0..stages { tensors.extend(v4_block(&cfg, &format!("mtp.{stage}"), 0, false)); }
    if stages > 0 {
        let last = stages - 1;
        tensors.extend([
            t("mtp.0.main_proj.weight", "F8_E4M3", &[cfg.dim, 3 * cfg.dim]),
            t("mtp.0.main_proj.scale", "F8_E8M0", &[cfg.dim / 128, 3 * cfg.dim / 128]),
            t("mtp.0.main_norm.weight", "BF16", &[cfg.dim]),
            t(format!("mtp.{last}.hc_head_fn"), "F32", &[4, 4 * cfg.dim]),
            t(format!("mtp.{last}.hc_head_scale"), "F32", &[1]),
            t(format!("mtp.{last}.hc_head_base"), "F32", &[4]),
            t(format!("mtp.{last}.norm.weight"), "BF16", &[cfg.dim]),
            t(format!("mtp.{last}.markov_head.markov_w1.weight"), "BF16", &[cfg.vocab_size, 256]),
            t(format!("mtp.{last}.markov_head.markov_w2.weight"), "BF16", &[cfg.vocab_size, 256]),
        ]);
    }
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

/// Complete native coordinator operands; routed experts are added separately.
fn v4_block(cfg: &crate::families::deepseek_v4::DeepseekV4Config, prefix: &str, ratio: usize, hash: bool)
    -> Vec<Tensor> {
    let (h, q, heads, groups, rank, inter) =
        (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.o_groups, cfg.o_lora_rank, cfg.moe_inter_dim);
    let mut out = Vec::new();
    for site in ["attn", "ffn"] {
        out.extend([t(format!("{prefix}.hc_{site}_fn"), "F32", &[24, 4 * h]),
            t(format!("{prefix}.hc_{site}_scale"), "F32", &[3]),
            t(format!("{prefix}.hc_{site}_base"), "F32", &[24]),
            t(format!("{prefix}.{site}_norm.weight"), "BF16", &[h])]);
    }
    for (name, n, k) in [("attn.wq_a", q, h), ("attn.wkv", 512, h), ("attn.wq_b", heads * 512, q),
        ("attn.wo_a", groups * rank, heads * 512 / groups), ("attn.wo_b", h, groups * rank),
        ("ffn.shared_experts.w1", inter, h), ("ffn.shared_experts.w3", inter, h),
        ("ffn.shared_experts.w2", h, inter)] {
        out.extend([t(format!("{prefix}.{name}.weight"), "F8_E4M3", &[n, k]),
            t(format!("{prefix}.{name}.scale"), "F8_E8M0", &[n / 128, k / 128])]);
    }
    out.extend([t(format!("{prefix}.attn.q_norm.weight"), "BF16", &[q]),
        t(format!("{prefix}.attn.kv_norm.weight"), "BF16", &[512]),
        t(format!("{prefix}.attn.attn_sink"), "F32", &[heads]),
        t(format!("{prefix}.ffn.gate.weight"), "BF16", &[cfg.n_routed_experts, h])]);
    out.push(if hash { t(format!("{prefix}.ffn.gate.tid2eid"), "I64", &[cfg.vocab_size, cfg.n_activated_experts]) }
        else { t(format!("{prefix}.ffn.gate.bias"), "F32", &[cfg.n_routed_experts]) });
    if ratio > 0 {
        let width = if ratio == 4 { 1024 } else { 512 };
        for projection in ["wkv", "wgate"] {
            out.push(t(format!("{prefix}.attn.compressor.{projection}.weight"), "BF16", &[width, h]));
        }
        out.extend([t(format!("{prefix}.attn.compressor.ape"), "F32", &[ratio, width]),
            t(format!("{prefix}.attn.compressor.norm.weight"), "BF16", &[512])]);
    }
    if ratio == 4 {
        for projection in ["wkv", "wgate"] {
            out.push(t(format!("{prefix}.attn.indexer.compressor.{projection}.weight"), "BF16", &[256, h]));
        }
        let n = cfg.index_n_heads * cfg.index_head_dim;
        out.extend([t(format!("{prefix}.attn.indexer.compressor.ape"), "F32", &[4, 256]),
            t(format!("{prefix}.attn.indexer.compressor.norm.weight"), "BF16", &[cfg.index_head_dim]),
            t(format!("{prefix}.attn.indexer.wq_b.weight"), "F8_E4M3", &[n, q]),
            t(format!("{prefix}.attn.indexer.wq_b.scale"), "F8_E8M0", &[n / 128, q / 128]),
            t(format!("{prefix}.attn.indexer.weights_proj.weight"), "BF16", &[cfg.index_n_heads, h])]);
    }
    out
}

/// The fixed official V4.1 header contract over sparse payloads, split into its
/// 48 required shard names. This never reads HF_HOME or allocates tensor data.
pub fn write_v41_snapshot(dir: &Path) {
    write_v41_snapshot_format(dir, false, None);
}

/// ModelOpt's calibrated W4A4 expert headers with the official non-expert tensors.
pub fn write_v41_nvfp4_snapshot(dir: &Path) {
    write_v41_snapshot_format(dir, true, None);
}

/// Raw routed-only EXL3 publication, with source-precision draft experts.
pub fn write_v41_exl3_snapshot(dir: &Path, bits: usize) {
    assert!((2..=5).contains(&bits));
    write_v41_snapshot_format(dir, false, Some(bits));
}

/// Bounded synthetic published allocation inputs, not performance estimates.
pub fn write_v41_workspaces(dir: &Path) -> std::path::PathBuf {
    let share = dir.join("share");
    fs::create_dir_all(share.join("V41_EXPERT_WORKSPACES")).unwrap();
    let programs = share.join("PROGRAMS.json");
    fs::write(&programs, b"{}").unwrap();
    for ranks in [1, 2] {
        let role = if ranks == 1 { "rtx_backbone" } else { "rtx_tp2" };
        for format in ["mxfp4", "nvfp4"] {
            let manifest = json!({"role":role, "geometry":{"hidden":5120,
                "intermediate":2304/ranks,"kernel_intermediate":2304/ranks,
                "experts":384,"topk":6}, "variants":([1,16,80,256,1024,4096].map(|c|
                    json!({"capacity_rows":c,"core_scratch_nbytes":if c==16 {99999}else{c*7}})))});
            fs::write(share.join("V41_EXPERT_WORKSPACES").join(format!("{format}_{role}.json")),
                serde_json::to_vec(&manifest).unwrap()).unwrap();
        }
        for bits in 2..=5 {
            let tiers = if bits == 5 { "45".to_owned() } else { format!("{bits}{}", bits+1) };
            for capacity in [1,16,80,256,1024,4096] {
                let path = share.join("exl3").join(format!("exl3-k{tiers}"))
                    .join(format!("rtx-tp{ranks}")).join(format!("m{capacity}"));
                fs::create_dir_all(&path).unwrap();
                fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&json!({
                    "capacity":capacity,"hidden":5120,"experts":384,
                    "intermediate":2304/ranks,"top_k":6,"input_format":"e4m3_k32",
                    "trellis_lut":{"bytes":32},"buffers":{
                        "scratch":{"allocation":"scratch","dtype":"f32","bytes":capacity*7,
                            "zero_on_create":false}}})).unwrap()).unwrap();
            }
        }
    }
    let matrices = [(5120,1152),(1152,5120),(5120,1280),(1280,32768),
        (32768,8192),(8192,5120),(5120,2304),(2304,5120),(5120,512),
        (1280,4096),(6144,25600),(15360,5120)];
    let variants = matrices.into_iter().flat_map(|(k,n)|
        [1,16,80,256,1024,4096].map(move |c| json!({"capacity":c,"input_dim":k,
            "output_dim":n,"label":format!("matrix_{k}_{n}_m{c}"),"scratch_bytes":c+1,
            "expected_m":if c<=80 {16}else{c},"split_k_slices":1,"all_rows":c<=80}))).collect::<Vec<_>>();
    fs::write(share.join("V41_FP8_AOT.json"), serde_json::to_vec(&json!({"variants":variants})).unwrap()).unwrap();
    fs::write(share.join("V41_EXPERT_AOT.json"), serde_json::to_vec(&json!({
        "role":"coordinator","input_format":"bf16","geometry":{"hidden":5120,
            "intermediate":2304,"kernel_intermediate":2304,"experts":128,"topk":3},
        "variants":([16,40,80,256].map(|c|json!({"capacity_rows":c,"core_scratch_nbytes":c*7})))
    })).unwrap()).unwrap();
    programs
}

fn write_v41_snapshot_format(dir: &Path, nvfp4: bool, exl3_bits: Option<usize>) {
    let mut config: Value = serde_json::from_str(include_str!(
        "../families/deepseek_v41/official-v41-config.json")).unwrap();
    if nvfp4 {
        let layers: serde_json::Map<String, Value> = (0..40).map(|layer|
            (format!("layers.{layer}.ffn.experts"), json!({"group_size":16,"quant_algo":"NVFP4"}))).collect();
        config["quantization_config"] = json!({
            "quant_method":"fp8", "quant_algo":"MIXED_PRECISION", "moe_quant_algo":"NVFP4",
            "expert_dtype":"fp4", "activation_scheme":"dynamic", "group_size":16,
            "scale_fmt":"ue8m0", "weight_block_size":[32,32], "kv_cache_quant_algo":null,
            "ignore":["*.attn.*","*.ffn.shared_experts.*","head","mtp.*"],
            "producer":{"name":"modelopt","version":"fixture"}, "quantized_layers":layers,
            "config_groups":{"group_0":{"targets":["Linear"],
                "weights":{"dynamic":false,"group_size":16,"num_bits":4,"type":"float"},
                "input_activations":{"dynamic":false,"group_size":16,"num_bits":4,"type":"float"}}}
        });
    }
    if let Some(bits) = exl3_bits {
        let native = config["quantization_config"].clone();
        config["quantization_config"] = json!({"quant_method":"exl3","bits":bits,
            "codebook":"mcg","mtp_experts":"source","mtp_experts_start_layer":40,
            "weight_block_size":[32,32],"non_routed_quantization":{
                "quant_method":"deepseek_v4_fp8","fmt":"e4m3",
                "activation_scheme":native["activation_scheme"],"scale_fmt":native["scale_fmt"],
                "weight_block_size":native["weight_block_size"],"expert_dtype":"fp4"}});
    }
    let templates: Vec<Value> = serde_json::from_str(include_str!(
        "../families/deepseek_v41/official-v41-tensors.json")).unwrap();
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    let mut shards: Vec<Vec<Tensor>> = vec![vec![]; 48];
    let mut weight_map = serde_json::Map::new();
    let mut index = 0;
    for template in templates {
        let mut names = vec![template["pattern"].as_str().unwrap().to_owned()];
        for axis in template["axes"].as_array().unwrap() {
            names = names.into_iter().flat_map(|name| axis.as_array().unwrap().iter()
                .map(move |i| name.replacen("{}", &i.to_string(), 1))).collect();
        }
        let dtype = match template["dtype"].as_str().unwrap() {
            "BF16" => "BF16", "F32" => "F32", "I8" => "I8",
            "F8_E4M3" => "F8_E4M3", "F8_E8M0" => "F8_E8M0",
            other => panic!("unexpected official dtype {other}"),
        };
        let shape = template["shape"].as_array().unwrap().iter()
            .map(|n| n.as_u64().unwrap() as usize).collect::<Vec<_>>();
        for name in names {
            let tensors = if let Some(bits) = exl3_bits.filter(|_| name.starts_with("layers.") && name.contains(".ffn.experts.")) {
                if name.ends_with(".scale") { continue; }
                let prefix = name.strip_suffix(".weight").expect("official expert weight");
                let (n,k) = if prefix.ends_with(".w2") { (5120,2304) } else { (2304,5120) };
                exl3(prefix,n,k,bits)
            } else if nvfp4 && name.starts_with("layers.") && name.contains(".ffn.experts.") {
                if name.ends_with(".scale") { continue; }
                let prefix = name.strip_suffix(".weight").expect("official expert weight");
                let (n,k) = if prefix.ends_with(".w2") { (5120,2304) } else { (2304,5120) };
                self::nvfp4(prefix,n,k)
            } else { vec![(name,dtype,shape.clone())] };
            for tensor in tensors {
                let shard = index % 48;
                weight_map.insert(tensor.0.clone(), json!(format!("model-{:05}-of-00048.safetensors", shard + 1)));
                shards[shard].push(tensor);
                index += 1;
            }
        }
    }
    assert_eq!(index, 96085 + if nvfp4 || exl3_bits.is_some() { 40*384*3*2 } else { 0 });
    for (shard, tensors) in shards.iter().enumerate() {
        write_safetensors(&dir.join(format!("model-{:05}-of-00048.safetensors", shard + 1)), tensors);
    }
    fs::write(dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map":weight_map})).unwrap()).unwrap();
}
