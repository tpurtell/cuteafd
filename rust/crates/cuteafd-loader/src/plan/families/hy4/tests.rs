use super::*;
use crate::families::hy4::execution::{CacheLayout, INDEX_PAGE_BYTES};
use crate::families::hy4::weight_layout::target_arenas;
use crate::families::hy4::{Hy4Config, Hy4KvCache, IndexCacheSchedule};
use crate::families::hy4_common::{core_split, PAGE_ROWS};
use crate::placement::families::hy4::HY4 as HY4_EXECUTOR;
use crate::placement::AttentionPlacement;
use crate::plan::{
    testing::{t, write_snapshot, Tensor},
    PlanInput, PlanOptions,
};

const GIB: u64 = 1 << 30;

fn hy4_bf16_snapshot() -> Option<std::path::PathBuf> {
    let hf = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME")
        .or_else(|| std::env::var_os("HF_HOME"))
        .map(std::path::PathBuf::from)?;
    let snap = hf.join(
        "hub/models--tencent--Hy4/snapshots/c10c0de5b6f850703fb0a90e65d1c32b558b9167",
    );
    snap.is_dir().then_some(snap)
}

fn hy4_config_json(exl3: bool) -> serde_json::Value {
    let indexer_types: Vec<&str> = (0..78)
        .map(|i| if i == 0 || (i >= 1 && (i - 1) % 4 == 0) { "full" } else { "shared" })
        .collect();
    let layer_types: Vec<&str> = vec!["deepseek_sparse_attention"; 78];
    let mlp_layer_types: Vec<&str> = (0..78).map(|i| if i == 0 { "dense" } else { "sparse" }).collect();
    let mut cfg: serde_json::Value = serde_json::from_str(r#"{
        "architectures": ["HYV4VLForConditionalGeneration"],
        "model_type": "hy_v4_vl",
        "vocab_size": 120832,
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "image_token_id": 120120,
        "video_token_id": 120124,
        "eos_token_id": 120025,
        "tie_word_embeddings": false,
        "dfly_enable_ihc": false,
        "dfly_mtp_fusion": "slot0",
        "dfly_target_index_topk": 2048,
        "dfly_context_local_window": null,
        "dfly_enable_hidden_correction": true,
        "dfly_num_hidden_layers": 5,
        "dfly_mlp_type": "dense",
        "dfly_mask_token_id": 120009,
        "dfly_target_index_layer": [61, 65, 69, 73, 77],
        "dfly_target_kv_layers": [61, 65, 69, 73, 77],
        "dfly_attn_causal": true,
        "dfly_block_size": 7,
        "text_config": {
            "model_type": "hy_v4",
            "vocab_size": 120832,
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "intermediate_size": 18432,
            "first_k_dense_replace": 1,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "attention_bias": false,
            "enable_ihc": true,
            "gated_mla": true,
            "gating_type": "elementwise",
            "enable_lm_head_fp32": true,
            "learnable_sink": true,
            "tie_word_embeddings": false,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_magnitude": 2.0,
            "num_attention_heads": 64,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_head_dim": 256,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 2048,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 2.827,
            "swiglu_limit": 10.0,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [12, 10, 10],
                "rope_theta": 10000000,
                "rope_type": "default"
            }
        }
    }"#).expect("valid json");
    cfg["text_config"]["indexer_types"] = serde_json::json!(indexer_types);
    cfg["text_config"]["layer_types"] = serde_json::json!(layer_types);
    cfg["text_config"]["mlp_layer_types"] = serde_json::json!(mlp_layer_types);
    if exl3 {
        cfg["quantization_config"] = serde_json::json!({
            "quant_method": "exl3",
            "format": "exl3",
            "checkpoint_format": "exl3",
            "bits": 4.0,
            "codebook": "mul1",
            "mul1_multiplier": 2212286765u64,
            "expert_bits": 4.0
        });
    }
    cfg
}

/// Build a synthetic header-only 78-layer Hy4 snapshot (either BF16 fused source
/// or EXL3 K4 MUL1 MC with optional compact K6/K8 core).
fn synthetic_hy4_snapshot(exl3: bool, compact_core_bits: Option<usize>) -> tempfile::TempDir {
    let cfg_json = hy4_config_json(exl3);
    let parsed = Hy4Config::from_hf(&cfg_json).unwrap();
    let (h, v, hc) = (parsed.hidden, parsed.vocab_size, parsed.hc_mult);
    let mut tensors: Vec<Tensor> = vec![
        t("model.embed_tokens.weight", "BF16", &[v, h]),
        t("model.norm.weight", "BF16", &[h]),
        t("model.hc_head.hc_head_fn", "F32", &[hc, hc * h]),
        t("model.hc_head.hc_head_base", "F32", &[hc]),
        t("model.hc_head.hc_head_scale", "F32", &[1]),
    ];

    let push_mat = |tensors: &mut Vec<Tensor>, stem: String, n: usize, k: usize, bits: Option<usize>| {
        if let Some(b) = bits {
            let stored_n = n.div_ceil(128) * 128;
            tensors.push(t(format!("{stem}.trellis"), "I16", &[k / 16, stored_n / 16, 16 * b]));
            tensors.push(t(format!("{stem}.suh"), "F16", &[k]));
            tensors.push(t(format!("{stem}.svh"), "F16", &[stored_n]));
            tensors.push(t(format!("{stem}.mul1"), "I32", &[]));
        } else {
            tensors.push(t(format!("{stem}.weight"), "BF16", &[n, k]));
        }
    };

    push_mat(&mut tensors, "lm_head".into(), v, h, compact_core_bits);

    for layer in 0..parsed.layers {
        let base = format!("model.layers.{layer}");
        tensors.push(t(format!("{base}.input_layernorm.weight"), "BF16", &[h]));
        tensors.push(t(format!("{base}.post_attention_layernorm.weight"), "BF16", &[h]));
        for tag in ["hc_attn_layer", "hc_mlp_layer"] {
            let p = format!("{base}.{tag}.hc_pre");
            tensors.push(t(format!("{p}.hc_fn"), "F32", &[2 * hc, hc * h]));
            tensors.push(t(format!("{p}.hc_base"), "F32", &[2 * hc]));
            tensors.push(t(format!("{p}.hc_scale"), "F32", &[2]));
        }
        let a = format!("{base}.self_attn");
        push_mat(&mut tensors, format!("{a}.q_a_proj"), parsed.q_lora_rank, h, compact_core_bits);
        tensors.push(t(format!("{a}.q_a_layernorm.weight"), "BF16", &[parsed.q_lora_rank]));
        push_mat(
            &mut tensors,
            format!("{a}.q_b_proj"),
            parsed.heads * parsed.qk_head_dim(),
            parsed.q_lora_rank,
            compact_core_bits,
        );
        push_mat(
            &mut tensors,
            format!("{a}.kv_a_proj_with_mqa"),
            parsed.kv_lora_rank + parsed.qk_rope_dim,
            h,
            compact_core_bits,
        );
        tensors.push(t(format!("{a}.kv_a_layernorm.weight"), "BF16", &[parsed.kv_lora_rank]));
        push_mat(
            &mut tensors,
            format!("{a}.kv_b_proj"),
            parsed.heads * (parsed.qk_nope_dim + parsed.v_head_dim),
            parsed.kv_lora_rank,
            compact_core_bits,
        );
        push_mat(
            &mut tensors,
            format!("{a}.linear_gate"),
            parsed.heads * parsed.v_head_dim,
            h,
            compact_core_bits,
        );
        push_mat(
            &mut tensors,
            format!("{a}.o_proj"),
            h,
            parsed.heads * parsed.v_head_dim,
            compact_core_bits,
        );
        tensors.push(t(format!("{a}.learnable_sink_param"), "F32", &[parsed.heads]));

        if parsed.is_full_indexer(layer) {
            let (d, n) = (parsed.indexer_head_dim, parsed.indexer_heads);
            let ip = format!("{a}.indexer");
            tensors.push(t(format!("{ip}.wq_b.weight"), "BF16", &[n * d, parsed.q_lora_rank]));
            tensors.push(t(format!("{ip}.wk.weight"), "BF16", &[d, h]));
            tensors.push(t(format!("{ip}.k_norm.weight"), "BF16", &[d]));
            tensors.push(t(format!("{ip}.k_norm.bias"), "BF16", &[d]));
            tensors.push(t(format!("{ip}.weights_proj.weight"), "BF16", &[n, h]));
        }

        if parsed.dense_layer[layer] {
            push_mat(&mut tensors, format!("{base}.mlp.gate_proj"), parsed.intermediate, h, compact_core_bits);
            push_mat(&mut tensors, format!("{base}.mlp.up_proj"), parsed.intermediate, h, compact_core_bits);
            push_mat(&mut tensors, format!("{base}.mlp.down_proj"), h, parsed.intermediate, compact_core_bits);
        } else {
            let shared = parsed.moe_intermediate * parsed.shared_experts;
            tensors.push(t(format!("{base}.mlp.gate.weight"), "BF16", &[parsed.experts, h]));
            tensors.push(t(format!("{base}.mlp.gate.e_score_correction_bias"), "F32", &[parsed.experts]));
            push_mat(
                &mut tensors,
                format!("{base}.mlp.shared_experts.gate_proj"),
                shared,
                h,
                compact_core_bits,
            );
            push_mat(
                &mut tensors,
                format!("{base}.mlp.shared_experts.up_proj"),
                shared,
                h,
                compact_core_bits,
            );
            push_mat(
                &mut tensors,
                format!("{base}.mlp.shared_experts.down_proj"),
                h,
                shared,
                compact_core_bits,
            );
            if exl3 {
                let (eh, ei, bits) = (parsed.hidden, parsed.moe_intermediate, 4usize);
                for expert in 0..parsed.experts {
                    for (proj, n, k) in [("gate_proj", ei, eh), ("up_proj", ei, eh), ("down_proj", eh, ei)] {
                        let stem = format!("{base}.mlp.experts.{expert}.{proj}");
                        tensors.push(t(format!("{stem}.trellis"), "I16", &[k / 16, n / 16, 16 * bits]));
                        tensors.push(t(format!("{stem}.suh"), "F16", &[k]));
                        tensors.push(t(format!("{stem}.svh"), "F16", &[n]));
                        tensors.push(t(format!("{stem}.mul1"), "I32", &[]));
                    }
                }
            } else {
                tensors.push(t(
                    format!("{base}.mlp.experts.gate_up_proj"),
                    "BF16",
                    &[parsed.experts, 2 * parsed.moe_intermediate, parsed.hidden],
                ));
                tensors.push(t(
                    format!("{base}.mlp.experts.down_proj"),
                    "BF16",
                    &[parsed.experts, parsed.hidden, parsed.moe_intermediate],
                ));
            }
        }
    }

    let dir = tempfile::tempdir().unwrap();
    write_snapshot(dir.path(), &cfg_json, &tensors, None);
    if exl3 {
        let mut tensor_storage = serde_json::Map::new();
        for (name, _, shape) in &tensors {
            let Some(stem) = name.strip_suffix(".trellis") else {
                continue;
            };
            let k = shape[0] * 16;
            let n = shape[1] * 16;
            let bits = shape[2] / 16;
            tensor_storage.insert(
                stem.to_string(),
                serde_json::json!({
                    "quant_format": "exl3",
                    "bits_per_weight": bits,
                    "mul1_multiplier": 2212286765u64,
                    "stored_tensors": {
                        format!("{stem}.trellis"): {"dtype": "int16", "shape": [shape[0], shape[1], shape[2]]},
                        format!("{stem}.suh"): {"dtype": "float16", "shape": [k]},
                        format!("{stem}.svh"): {"dtype": "float16", "shape": [n]},
                        format!("{stem}.mul1"): {"dtype": "int32", "shape": []},
                    }
                }),
            );
        }
        let mut quant_cfg = cfg_json["quantization_config"].as_object().unwrap().clone();
        quant_cfg.insert("method".to_string(), serde_json::json!("exl3"));
        quant_cfg.insert("out_scales".to_string(), serde_json::json!("auto"));
        quant_cfg.insert("group_size".to_string(), serde_json::json!(-1));
        quant_cfg.insert("desc_act".to_string(), serde_json::json!(false));
        quant_cfg.insert("lm_head".to_string(), serde_json::json!(compact_core_bits.is_some()));
        quant_cfg.insert("tensor_storage".to_string(), serde_json::Value::Object(tensor_storage));
        std::fs::write(
            dir.path().join("quantize_config.json"),
            serde_json::to_vec(&serde_json::Value::Object(quant_cfg)).unwrap(),
        )
        .unwrap();
    }
    dir
}

#[test]
fn hy4_family_descriptor_invariants() {
    assert_eq!(HY4.id(), "hy4");
    assert_eq!(HY4.runtime(), RuntimeStatus::Serving);
    assert!(HY4.expert_catalog());
    assert!(HY4.optional(Component::Vision));
    assert!(HY4.optional(Component::Speculator));
    assert!(!HY4.optional(Component::Attention));
    assert!(!HY4.optional(Component::Indexer));
    assert!(!HY4.optional(Component::RoutedExpert));
    assert_ne!(
        HY4_EXECUTOR.attention,
        AttentionPlacement::Heads,
        "2-GPU default must not replicate DSA KV"
    );
}

#[test]
fn hy4_config_index_cache_and_activation_contracts() {
    let cfg_json = hy4_config_json(false);
    let parsed = Hy4Config::from_hf(&cfg_json).expect("valid Hy4 config");
    parsed.release_geometry().expect("matches Hy4 release geometry");
    assert_eq!(parsed.qa_kva_eps, 1e-6);
    assert_eq!(parsed.rms_norm_eps, 1e-5);
    assert_eq!(parsed.index_cache.full_count(), 21);
    assert_eq!(parsed.index_cache.shared_count(), 57);
    assert!(parsed.is_full_indexer(0));
    assert!(parsed.is_full_indexer(1));
    assert!(!parsed.is_full_indexer(2));
    assert_eq!(parsed.index_source_layer(0), 0);
    assert_eq!(parsed.index_source_layer(1), 1);
    assert_eq!(parsed.index_source_layer(2), 1);
    assert_eq!(parsed.index_source_layer(3), 1);
    assert_eq!(parsed.index_source_layer(4), 1);
    assert_eq!(parsed.index_source_layer(5), 5);
    assert_eq!(parsed.index_source_layer(76), 73);
    assert_eq!(parsed.index_source_layer(77), 77);
    assert!(parsed.is_full_indexer(77));

    // SwiGLU clamp=10.0 on routed experts only; dense layer 0 and shared experts unclamped.
    assert_eq!(parsed.activation(0, false), cuteafd_core::FfnActivation::SiluBf16Product { limit: None });
    assert_eq!(parsed.activation(1, true), cuteafd_core::FfnActivation::SiluBf16Product { limit: None });
    assert_eq!(parsed.activation(1, false), cuteafd_core::FfnActivation::SiluBf16Product { limit: Some(10.0) });

    // Core splits
    assert_eq!(core_split("lm_head"), crate::formats::trellis_core::CoreSplit::Output);
    assert_eq!(core_split("model.layers.0.mlp.gate_proj"), crate::formats::trellis_core::CoreSplit::Output);
    assert_eq!(core_split("model.layers.0.mlp.down_proj"), crate::formats::trellis_core::CoreSplit::Input);
    assert_eq!(core_split("model.layers.1.mlp.gate"), crate::formats::trellis_core::CoreSplit::Whole);

    // Reject invalid IndexCache schedule starting with "shared".
    assert!(IndexCacheSchedule::from_kinds(vec![crate::families::hy4::Hy4IndexerKind::Shared]).is_err());

    // Reject split-half RoPE (interleaved settled in #2550).
    let mut bad_rope = cfg_json.clone();
    bad_rope["text_config"]["rope_parameters"]["mrope_interleaved"] = serde_json::json!(false);
    assert!(Hy4Config::from_hf(&bad_rope).is_err());
}

#[test]
fn hy4_pages_only_cache_geometry_charges_21_full_indexers_and_zero_marks() {
    let dir = synthetic_hy4_snapshot(false, None);
    let checkpoint = Checkpoint::inventory(dir.path()).unwrap();
    let model = Model::open(&checkpoint).unwrap();

    // 1x RTX BF16 KV: 78 * 64 * 1152 + 21 * 8448 = 5,750,784 + 177,408 = 5,928,192 B/page.
    let geom_1x = model
        .cache_geometry(crate::serving_capacity::CacheOptions {
            coordinator_ranks: 1,
            attention_placement: None,
            mla_kv: cuteafd_core::MlaCacheFormat::Bf16,
            ..Default::default()
        })
        .unwrap()
        .unwrap();
    assert_eq!(geom_1x.logical_unit_rows, PAGE_ROWS as u64);
    assert_eq!(geom_1x.ranks.len(), 1);
    assert_eq!(
        geom_1x.ranks[0].persistent_unit_bytes,
        78 * 64 * 1152 + 21 * INDEX_PAGE_BYTES as u64
    );
    assert_eq!(geom_1x.ranks[0].active_state_per_sequence_bytes, 0);
    assert_eq!(geom_1x.ranks[0].retained_mark_bytes, 0);

    // 2x RTX ContextSplit FP8 KV: logical unit = 128 rows (64/GPU), zero KV replication.
    let geom_2x = model
        .cache_geometry(crate::serving_capacity::CacheOptions {
            coordinator_ranks: 2,
            attention_placement: Some(AttentionPlacement::Context),
            mla_kv: cuteafd_core::MlaCacheFormat::Fp8,
            ..Default::default()
        })
        .unwrap()
        .unwrap();
    assert_eq!(geom_2x.logical_unit_rows, 2 * PAGE_ROWS as u64);
    assert_eq!(geom_2x.placement, crate::serving_capacity::KvPlacement::PartitionedContext);
    let expected_rank = CacheLayout::new(Hy4KvCache::Fp8, 78, 21)
        .unwrap()
        .persistent_unit_bytes()
        .unwrap();
    assert_eq!(geom_2x.ranks[0].persistent_unit_bytes, expected_rank);
    assert_eq!(geom_2x.ranks[1].persistent_unit_bytes, expected_rank);
}

#[test]
fn hy4_bf16_source_contract_and_arena_sizes_match_expectations() {
    let dir = synthetic_hy4_snapshot(false, None);
    let input = PlanInput::open(dir.path()).unwrap();
    let report = input.plan(&PlanOptions::default()).unwrap();
    assert_eq!(report.family.as_deref(), Some("hy4"));
    assert_eq!(report.runtime, Some(RuntimeStatus::Serving));
    assert!(report.unclassified.is_empty(), "unclassified: {:?}", report.unclassified);
    for comp in &report.components {
        if comp.component == Component::RoutedExpert {
            assert_eq!(comp.rejected, 1, "BF16 routed experts should report only the catalog absence: {:?}", comp.rejections);
            assert!(
                comp.rejections[0].reason.contains("BF16 fused experts have no Spark/RTX package"),
                "unexpected routed rejection: {:?}",
                comp.rejections
            );
        } else {
            assert_eq!(
                comp.rejected, 0,
                "component {} rejected: {:?}",
                comp.component.label(),
                comp.rejections
            );
        }
    }

    let checkpoint = Checkpoint::inventory(dir.path()).unwrap();
    let model = Model::open(&checkpoint).unwrap();
    let modes_1x = crate::placement::families::hy4::layer_modes(&model.cfg, 1, AttentionPlacement::Context);
    let arenas_1x = target_arenas(&checkpoint, &model.cfg, &model.core, &modes_1x, 1).unwrap();
    let gib_1x = arenas_1x[0].arena.bytes() as f64 / GIB as f64;
    assert!((48.0..48.5).contains(&gib_1x), "1x BF16 target arena {gib_1x:.2} GiB");

    let modes_2x = crate::placement::families::hy4::layer_modes(&model.cfg, 2, AttentionPlacement::Context);
    let arenas_2x = target_arenas(&checkpoint, &model.cfg, &model.core, &modes_2x, 2).unwrap();
    let gib_2x = arenas_2x[0].arena.bytes() as f64 / GIB as f64;
    assert!((26.0..26.6).contains(&gib_2x), "2x ContextSplit BF16 target arena {gib_2x:.2} GiB");
}

#[test]
fn hy4_exl3_catalog_resolves_hy4_geometry_distinct_from_glm5() {
    // Full Spark EXL3 placement testing lands in Step 3 alongside the
    // spark-package-ranges.json `hy4:exl3` entry and placement-input republish (#2550).
    let dir_orig = synthetic_hy4_snapshot(true, None);
    let input_orig = PlanInput::open(dir_orig.path()).unwrap();
    let cat_geom = input_orig.expert_catalog().unwrap().routed_experts().geometry().unwrap();
    assert_eq!(cat_geom, cuteafd_core::ExpertGeometry::HY4);
    assert_eq!(cat_geom.family(), Some("hy4"));
    assert_eq!(cat_geom.swiglu_limit(), Some(10.0));
    assert!(!cat_geom.same_shape(&cuteafd_core::ExpertGeometry::GLM5));

    // Verify K6 compact core packed target arena shrinks as planned (~20.7 GiB 1x / ~11.9 GiB/GPU 2x).
    let dir_k6 = synthetic_hy4_snapshot(true, Some(6));
    let ckpt_k6 = Checkpoint::inventory(dir_k6.path()).unwrap();
    let model_k6 = Model::open(&ckpt_k6).unwrap();
    let modes_1x = crate::placement::families::hy4::layer_modes(&model_k6.cfg, 1, AttentionPlacement::Context);
    let arenas_1x = target_arenas(&ckpt_k6, &model_k6.cfg, &model_k6.core, &modes_1x, 1).unwrap();
    let gib_1x = arenas_1x[0].arena.bytes() as f64 / GIB as f64;
    assert!((20.4..21.2).contains(&gib_1x), "1x K6 target arena {gib_1x:.2} GiB");
    let modes_2x = crate::placement::families::hy4::layer_modes(&model_k6.cfg, 2, AttentionPlacement::Context);
    let arenas_2x = target_arenas(&ckpt_k6, &model_k6.cfg, &model_k6.core, &modes_2x, 2).unwrap();
    let gib_2x = arenas_2x[0].arena.bytes() as f64 / GIB as f64;
    assert!((11.6..12.3).contains(&gib_2x), "2x ContextSplit K6 target arena {gib_2x:.2} GiB");
}

#[test]
fn hy4_live_bf16_checkpoint_validates_all_2710_tensors() {
    let Some(snapshot) = hy4_bf16_snapshot() else {
        return;
    };
    let input = PlanInput::open(&snapshot).expect("open live tencent/Hy4 snapshot");
    assert_eq!(crate::plan::family::detect(input.checkpoint()).expect("detected family").id(), "hy4");
    let report = input.plan(&PlanOptions::default()).expect("plan live tencent/Hy4");
    assert_eq!(report.family.as_deref(), Some("hy4"));
    assert_eq!(report.runtime, Some(RuntimeStatus::Serving));
    assert!(report.unclassified.is_empty(), "unclassified: {:?}", report.unclassified);
    for comp in &report.components {
        if comp.component == Component::RoutedExpert {
            assert_eq!(
                comp.rejected, 1,
                "live tencent/Hy4 BF16 routed experts should report only the catalog absence: {:?}",
                comp.rejections
            );
            assert!(
                comp.rejections[0].reason.contains("BF16 fused experts have no Spark/RTX package"),
                "unexpected live routed rejection: {:?}",
                comp.rejections
            );
        } else {
            assert_eq!(
                comp.rejected, 0,
                "live tencent/Hy4 component {} rejected: {:?}",
                comp.component.label(),
                comp.rejections
            );
        }
    }
}

pub(crate) fn concurrency_matrix_snapshot() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = synthetic_hy4_snapshot(true, Some(8));
    let manifest = dir.path().join("share/PROGRAMS.json");
    std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    std::fs::write(&manifest, r#"{"programs":[]}"#).unwrap();
    for (tp, intermediate) in [(1usize, 2048usize), (2, 1024)] {
        let root = dir.path().join(format!("lib/exl3/exl3-hy4-k45-mul1/rtx-tp{tp}"));
        for capacity in crate::placement::inventory::exl3_capacities(4096) {
            let path = root.join(format!("m{capacity}"));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(
                path.join("v41_exl3.json"),
                serde_json::to_vec(&serde_json::json!({
                    "capacity": capacity, "hidden": 6144, "experts": 256, "intermediate": intermediate, "top_k": 8,
                    "bits": [4, 5], "codebook": "mul1", "activation": "silu_post_clamp", "swiglu_limit": 10.0,
                    "input_format": "e4m3_k32", "trellis_lut": {"bytes": 4096}, "buffers": {
                        "scratch": {"allocation": "scratch", "dtype": "f32", "bytes": capacity * 64, "zero_on_create": false}}
                }))
                .unwrap(),
            )
            .unwrap();
        }
    }
    (dir, manifest)
}

#[test]
fn hy4_compact_core_trellis_side_is_output_on_tp1_and_input_on_tp2() {
    use crate::formats::dense_trellis::TrellisSide;
    let dir = synthetic_hy4_snapshot(true, Some(8));
    let ckpt = crate::plan::checkpoint::Checkpoint::inventory(dir.path()).unwrap();
    let model = Model::open(&ckpt).unwrap();
    let modes_1x = crate::placement::families::hy4::layer_modes(&model.cfg, 1, AttentionPlacement::Context);
    let modes_2x = crate::placement::families::hy4::layer_modes(&model.cfg, 2, AttentionPlacement::Context);
    let tp1 = target_arenas(&ckpt, &model.cfg, &model.core, &modes_1x, 1).unwrap();
    let tp2 = target_arenas(&ckpt, &model.cfg, &model.core, &modes_2x, 2).unwrap();
    for stem in [
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.shared_experts.down_proj",
    ] {
        assert_eq!(tp1[0].core[stem].side, TrellisSide::Output, "TP1 {stem}");
        assert_eq!(tp2[0].core[stem].side, TrellisSide::Input, "TP2 rank 0 {stem}");
        assert_eq!(tp2[1].core[stem].side, TrellisSide::Input, "TP2 rank 1 {stem}");
    }
}

#[test]
fn hy4_decode_graphs_on_matches_default_and_off_matches_zero_reserve_plan() {
    const CHILD: &str = "CUTEAFD_TEST_HY4_DECODE_GRAPHS_CHILD";
    if let Ok(mode) = std::env::var(CHILD) {
        let (dir, manifest) = concurrency_matrix_snapshot();
        let selected = crate::plan::checkpoint::CheckpointView::select(dir.path(), crate::plan::checkpoint::CoreView::Original).unwrap();
        let input = PlanInput::open_view(&selected).unwrap();
        let mut host = crate::plan::testing::SimulatedHost::raptor();
        host.gpu_count = 1;
        let mut layout = host.layout();
        layout.workspace_manifest = Some(manifest);
        layout.decode_graphs = mode != "off";
        let mut opts = PlanOptions::default();
        opts.placement = crate::plan::ExpertPlacement::from_spark_ranks(4);
        opts.vision = crate::plan::MediaMode::Off;
        opts.layout = Some(layout);
        let report = input.plan(&opts).unwrap();
        assert!(report.placement_supported, "{:?}", report.hints);
        let memory = report.memory_layout.unwrap();
        let reserve: u64 = memory.devices.iter()
            .filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx)
            .flat_map(|d| d.items.iter().filter(|i| i.group.starts_with("graph reserve")).map(|i| i.bytes))
            .sum();
        if mode == "on" {
            let glayout = crate::families::hy4::graph_reserve::Layout::resolve(1, 4, AttentionPlacement::Heads);
            let expected = crate::families::hy4::graph_reserve::reserve_policy(glayout, 1, 0, true).unwrap().bytes;
            assert_eq!(reserve, expected);
            assert!(reserve > 0);
        } else {
            assert_eq!(reserve, 0);
        }
        println!("LAYOUT={}", serde_json::to_string(&memory).unwrap());
        return;
    }
    let run = |mode: &str, reserve_env: Option<&str>| {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["plan::families::hy4::tests::hy4_decode_graphs_on_matches_default_and_off_matches_zero_reserve_plan", "--exact", "--nocapture"])
            .env(CHILD, mode)
            .env_remove("CUTEAFD_GPU_GRAPH_RESERVE")
            .env_remove("CUTEAFD_GPU_GRAPH_RESERVE_EXTRA")
            .env_remove("CUTEAFD_GPU_GRAPH_RESERVE_EXTRA_CAP");
        if let Some(value) = reserve_env { cmd.env("CUTEAFD_GPU_GRAPH_RESERVE", value); }
        let out = cmd.output().unwrap();
        assert!(out.status.success(), "{mode} failed: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap().lines()
            .find_map(|line| line.strip_prefix("LAYOUT=").map(str::to_owned))
            .expect("child layout output")
    };
    let on_layout = run("on", None);
    let off_layout = run("off", None);
    let zero_reserve_layout = run("zero_reserve", Some("0"));
    assert_ne!(on_layout, off_layout);
    assert_eq!(off_layout, zero_reserve_layout, "Hy4 decode_graphs=off must match GPU_GRAPH_RESERVE=0 plan");
}
