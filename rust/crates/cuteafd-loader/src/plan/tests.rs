use super::testing::*;
use super::*;
use crate::formats::exl3_storage::Exl3StorageSource;
use crate::families::mimo_v2::{MimoAttention, MimoV2Config};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn snapshot(config: Value, tensors: &[Tensor]) -> tempfile::TempDir {
    snapshot_tp(config, tensors, None)
}

fn snapshot_tp(config: Value, tensors: &[Tensor], tp: Option<usize>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_snapshot(dir.path(), &config, tensors, tp);
    dir
}

/// One Qwen layer with all 512 EXL3 experts and its storage map.
fn qwen_snapshot(bits: usize) -> tempfile::TempDir {
    let (tensors, manifest) = qwen4_exl3(1, bits);
    let mut config = qwen4_config(1);
    config["quantization_config"] = exl3_compact(bits);
    let dir = snapshot(config, &tensors);
    write_quantize_config(dir.path(), &manifest);
    dir
}

fn sparks(ranks: usize) -> PlanOptions {
    PlanOptions { placement: ExpertPlacement::from_spark_ranks(ranks), ..PlanOptions::default() }
}

fn component(report: &PlanReport, component: Component) -> &ComponentPlan {
    report.components.iter().find(|c| c.component == component).unwrap_or_else(|| panic!("{component:?}"))
}

fn rejected(report: &PlanReport, which: Component) -> Vec<String> {
    component(report, which).rejections.iter().map(|r| format!("{}: {}", r.tensor, r.reason)).collect()
}

/// Detects one group from (suffix, dtype, shape) members.
fn detect_group(stem: &str, members: &[(&str, &'static str, &[usize])]) -> Result<QuantOperand, Malformed> {
    let tensors: Vec<Tensor> = members
        .iter()
        .map(|(suffix, dtype, shape)| t(if suffix.is_empty() { stem.to_string() } else { format!("{stem}.{suffix}") },
            dtype, shape))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    write_safetensors(&dir.path().join("x.safetensors"), &tensors);
    let tensors: Vec<checkpoint::CheckpointTensor> = crate::read_safetensors_metadata(&dir.path().join("x.safetensors"))
        .unwrap()
        .into_iter()
        .map(|meta| checkpoint::CheckpointTensor { shard: "x".into(), meta })
        .collect();
    let groups = format::group_by_stem(&tensors);
    assert_eq!(groups.len(), 1, "{:?}", groups.keys().collect::<Vec<_>>());
    format::detect(groups.values().next().unwrap())
}

fn v41_config() -> Value {
    json!({
        "architectures": ["DeepseekV41ForCausalLM"],
        "text_config": {
            "hidden_size": 5120, "vocab_size": 128, "num_hidden_layers": 2,
            "n_routed_experts": 384, "num_experts_per_tok": 6, "moe_intermediate_size": 2304,
            "n_shared_experts": 1, "compress_ratios": [0, 2], "scoring_func": "sqrtsoftplus",
            "engram_layer_ids": [1]
        }
    })
}

#[test]
fn deepseek_v41_components_formats_and_placement() {
    let dir = snapshot(
        v41_config(),
        &[
            t("embed.weight", "BF16", &[128, 5120]),
            t("head.weight", "BF16", &[128, 5120]),
            t("norm.weight", "BF16", &[5120]),
            t("layers.0.attn.wq_a.weight", "F8_E4M3", &[64, 5120]),
            t("layers.0.attn.wq_a.scale", "F8_E8M0", &[2, 160]),
            t("layers.0.attn.attn_sink", "F32", &[4]),
            t("layers.1.attn.indexer.wq_b.weight", "F8_E4M3", &[64, 64]),
            t("layers.1.attn.indexer.wq_b.scale", "F8_E8M0", &[2, 2]),
            t("layers.0.ffn.gate.weight", "BF16", &[384, 5120]),
            t("layers.0.ffn.experts.0.w1.weight", "I8", &[2304, 2560]),
            t("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", &[2304, 160]),
            t("layers.1.engram.embed.weight", "F8_E4M3", &[16, 64]),
            t("layers.1.engram.embed.scale", "F8_E8M0", &[16, 2]),
            t("mtp.0.ffn.experts.0.w2.weight", "I8", &[5120, 1152]),
            t("mtp.0.ffn.experts.0.w2.scale", "F8_E8M0", &[5120, 72]),
        ],
    );
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    assert_eq!(report.family.as_deref(), Some("deepseek_v41"));
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    assert_eq!(component(&report, Component::RoutedExpert).owner, Owner::SparkSliced);
    assert!(component(&report, Component::RoutedExpert).formats.contains_key("mxfp4-g32"));
    assert!(component(&report, Component::Attention).formats.contains_key("fp8-block32x32/ue8m0"));
    assert_eq!(component(&report, Component::MappedTable).owner, Owner::HostMapped);
    assert!(component(&report, Component::MappedTable).formats.contains_key("fp8-block1x32/ue8m0"));
    assert_eq!(component(&report, Component::SpeculatorExpert).owner, Owner::Rtx);
    assert_eq!(report.experts.as_ref().unwrap().spark_worlds, [2, 3, 4, 6]);
    assert!(report.executable(), "{}", render(&report));
}

#[test]
fn v41_rejects_an_unsupported_expert_format_with_a_hint() {
    let dir = snapshot(
        v41_config(),
        &[t("embed.weight", "BF16", &[128, 5120]), t("layers.0.ffn.experts.0.w1.weight", "BF16", &[2304, 5120])],
    );
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    assert_eq!(component(&report, Component::RoutedExpert).status, Status::MissingKernel);
    assert!(!report.executable());
    assert!(report.hints.iter().any(|hint| hint.what.contains("routed_expert")));
    assert!(rejected(&report, Component::RoutedExpert)[0].starts_with("layers.0.ffn.experts.0.w1: "));
}

#[test]
fn unknown_architecture_is_reported_not_fatal() {
    let dir = snapshot(json!({"architectures": ["SomethingNew"], "model_type": "new"}), &[t("w", "BF16", &[2])]);
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.family.is_none());
    assert!(report.hints[0].what.contains("SomethingNew"));
}

// --- R06: lossless formats and the loaders' tensor contracts ---------------

/// The review's observation: an E5M2 expert and an EXL3 trellis without its
/// companions planned as ready. Both are now unsupported, by tensor name.
#[test]
fn observed_e5m2_and_incomplete_exl3_are_ready() {
    let dir = snapshot(glm5_config(), &glm5_tensors(|name, n, k| {
        if name.ends_with("gate_proj") {
            vec![t(format!("{name}.weight"), "F8_E5M2", &[n, k]), t(format!("{name}.weight_scale_inv"), "F32",
                &[n / 128, k / 128])]
        } else if name.ends_with("up_proj") {
            vec![t(format!("{name}.trellis"), "I16", &[k / 16, n / 16, 64])]
        } else {
            fp8(name, n, k, None)
        }
    }));
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    let experts = component(&report, Component::RoutedExpert);
    assert_eq!(experts.status, Status::MissingKernel, "{}", render(&report));
    // Two operands, and the expert staging's own refusal of the E5M2 weight.
    assert_eq!(experts.rejected, 3);
    let reasons = rejected(&report, Component::RoutedExpert);
    assert!(reasons[0].starts_with("routed experts (read_expert_catalog): ") && reasons[0].contains("expected E4M3"),
        "{reasons:?}");
    assert!(reasons.iter().any(|r| r.starts_with("model.layers.1.mlp.experts.0.gate_proj: ") && r.contains("fp8e5m2")),
        "{reasons:?}");
    assert!(reasons.iter().any(|r| r.starts_with("model.layers.1.mlp.experts.0.up_proj.") && r.contains("missing EXL3")),
        "{reasons:?}");
    assert!(experts.formats.contains_key("fp8e5m2-block128x128/f32") && experts.formats.contains_key("malformed"));
    assert!(!report.executable());
}

#[test]
fn e4m3_and_e5m2_are_distinct_operands() {
    let e4m3 = detect_group("w", &[("weight", "F8_E4M3", &[256, 256]), ("weight_scale_inv", "F32", &[2, 2])]).unwrap();
    let e5m2 = detect_group("w", &[("weight", "F8_E5M2", &[256, 256]), ("weight_scale_inv", "F32", &[2, 2])]).unwrap();
    assert_eq!(e4m3.encoding, Encoding::E4m3);
    assert_eq!(e5m2.encoding, Encoding::E5m2);
    assert_ne!(e4m3.label(), e5m2.label());
    assert!(e4m3.is_fp8_block(128, &[ScaleEncoding::F32]) && !e5m2.is_fp8_block(128, &[ScaleEncoding::F32]));
    // Scale encodings stay distinct too: E8M0 grids are not FP32 grids.
    let ue8m0 = detect_group("w", &[("weight", "F8_E4M3", &[256, 256]), ("scale", "F8_E8M0", &[2, 2])]).unwrap();
    assert!(ue8m0.is_fp8_block(128, &[ScaleEncoding::Ue8m0]) && !ue8m0.is_fp8_block(128, &[ScaleEncoding::F32]));
    // Both rows of a GLM routed expert: E4M3 accepted, E5M2 rejected.
    let dir = snapshot(glm5_config(), &glm5_tensors(|name, n, k| fp8(name, n, k, None)));
    assert!(plan(dir.path(), &PlanOptions::default()).unwrap().executable());
}

#[test]
fn malformed_trellis_and_scales_name_the_tensor() {
    let exl3 = |trellis: (&'static str, &'static [usize]), suh: &'static [usize], mcg: &'static [usize]| {
        detect_group("p", &[("trellis", trellis.0, trellis.1), ("suh", "F16", suh), ("svh", "F16", &[32]),
            ("mcg", "I32", mcg)])
    };
    let good = exl3(("I16", &[4, 2, 48]), &[64], &[]).unwrap();
    assert_eq!((good.encoding, good.logical.clone()), (Encoding::Exl3 { bits: 3 }, vec![32, 64]));
    // exllamav3 writes a scalar MCG marker, other exporters a one-element vector.
    assert_eq!(exl3(("I16", &[4, 2, 48]), &[64], &[1]).unwrap(), good);
    let cases: [(Result<QuantOperand, Malformed>, &str, &str); 10] = [
        (exl3(("I16", &[4, 96]), &[64], &[]), "p.trellis", "rank 2"),
        (exl3(("I32", &[4, 2, 48]), &[64], &[]), "p.trellis", "dtype i32"),
        (exl3(("I16", &[4, 2, 40]), &[64], &[]), "p.trellis", "not 16 x bits"),
        (exl3(("I16", &[4, 2, 48]), &[48], &[]), "p.suh", "expected f16 [64]"),
        (exl3(("I16", &[4, 2, 48]), &[64], &[2]), "p.mcg", "expected i32 [] or [1]"),
        (detect_group("p", &[("trellis", "I16", &[4, 2, 48]), ("suh", "F16", &[64]), ("svh", "F16", &[32])]),
            "p.trellis", "3INST codebook"),
        (detect_group("p", &[("trellis", "I16", &[4, 2, 48]), ("suh", "F16", &[64]), ("svh", "F16", &[32]),
            ("mul1", "I32", &[])]), "p.mul1", "MUL1 codebook"),
        (detect_group("p", &[("trellis", "I16", &[4, 2, 48]), ("su", "I16", &[4]), ("sv", "I16", &[2]),
            ("mcg", "I32", &[])]), "p.su", "packed EXL3 sign bitfield"),
        (detect_group("p", &[("trellis", "I16", &[4, 2, 48]), ("suh", "F16", &[64]), ("mcg", "I32", &[])]),
            "p.svh", "missing EXL3 companion"),
        (detect_group("p", &[("suh", "F16", &[64]), ("svh", "F16", &[32]), ("mcg", "I32", &[1])]),
            "p.mcg", "without a trellis"),
    ];
    for (result, tensor, reason) in cases {
        let error = result.unwrap_err();
        assert_eq!(error.tensor, tensor, "{error}");
        assert!(error.reason.contains(reason), "{error}");
    }
    // Scales: a grid that tiles neither axis, an MXFP4 grid that does not divide K, a BF16 weight with a scale.
    let bad_fp8 = detect_group("w", &[("weight", "F8_E4M3", &[256, 256]), ("weight_scale_inv", "F32", &[2, 3])]);
    assert_eq!(bad_fp8.unwrap_err().tensor, "w.weight_scale_inv");
    let bad_mx = detect_group("e", &[("weight", "U8", &[64, 96]), ("weight_scale", "U8", &[64, 5])]);
    assert!(bad_mx.unwrap_err().reason.contains("does not tile"));
    let bf16_scaled = detect_group("n", &[("weight", "BF16", &[64]), ("scale", "F32", &[1])]);
    assert_eq!(bf16_scaled.unwrap_err().tensor, "n.scale");
    let mx = detect_group("e", &[("weight", "U8", &[64, 96]), ("weight_scale", "U8", &[64, 6])]).unwrap();
    assert!(mx.is_mxfp4(32) && mx.logical == [64, 192]);
}

#[test]
fn nvfp4_operand_carries_its_global_and_input_scales() {
    let nvfp4 = detect_group("e", &[("weight", "U8", &[64, 64]), ("weight_scale", "F8_E4M3", &[64, 8]),
        ("weight_scale_2", "F32", &[]), ("input_scale", "F32", &[])]).unwrap();
    assert!(nvfp4.is_nvfp4() && nvfp4.input_scale.is_some());
    assert_eq!((nvfp4.label(), nvfp4.logical.clone()), ("nvfp4-g16".to_string(), vec![64, 128]));
    // An E4M3 per-16 grid without the FP32 global scale is not NVFP4.
    let partial = detect_group("e", &[("weight", "U8", &[64, 64]), ("weight_scale", "F8_E4M3", &[64, 8])]).unwrap();
    assert!(!partial.is_nvfp4());
}

#[test]
fn segmented_fused_qkv_resolves_checkpoint_shards() {
    let dir = snapshot_tp(mimo_pro_config(), &mimo_pro_tensors(), Some(8));
    let report = plan(dir.path(), &sparks(6)).unwrap();
    let attention = component(&report, Component::Attention);
    assert_eq!(attention.status, Status::Ready, "{}", render(&report));
    // 8 shards x [q | k | v]: one grid segment per whole checkpoint shard.
    assert!(attention.formats.contains_key("fp8-block128x128/f32-segmented8"), "{:?}", attention.formats);
    assert!(report.executable(), "{}", render(&report));
    // The same rows read as TP4 shards do not tile the 216-row grid.
    let dir = snapshot_tp(mimo_pro_config(), &mimo_pro_tensors(), Some(4));
    let report = plan(dir.path(), &sparks(6)).unwrap();
    assert_eq!(component(&report, Component::Attention).status, Status::MissingKernel);
    assert!(rejected(&report, Component::Attention)[0].contains("fused qkv_proj"), "{}", render(&report));
}

/// serve-glm's FP8 decode operands come from FP8 blocks, ModelOpt per-tensor
/// FP8 (a uniform grid) or BF16 quantized at load (nvidia/GLM-5.3-NVFP4); FP32
/// is still refused.
#[test]
fn glm5_coordinator_takes_bf16_and_per_tensor_fp8_where_decode_reads_fp8() {
    let mut tensors = glm5_tensors(|name, n, k| fp8(name, n, k, None));
    tensors.retain(|(name, ..)| !name.contains("layers.1.self_attn.q_b_proj") && !name.contains("layers.1.self_attn.o_proj")
        && !name.contains("layers.1.self_attn.kv_b_proj"));
    tensors.push(t("model.layers.1.self_attn.q_b_proj.weight", "BF16", &[16384, 2048]));
    tensors.push(t("model.layers.1.self_attn.o_proj.weight", "F32", &[6144, 16384]));
    // kv_b_proj is staged through BF16 rows: BF16 is fine there.
    tensors.push(t("model.layers.1.self_attn.kv_b_proj.weight", "BF16", &[28672, 512]));
    let dir = snapshot(glm5_config(), &tensors);
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    let attention = component(&report, Component::Attention);
    assert_eq!((attention.status, attention.rejected), (Status::MissingKernel, 1), "{}", render(&report));
    let reasons = rejected(&report, Component::Attention).join("\n");
    assert!(reasons.contains("model.layers.1.self_attn.o_proj: serve-glm's decode programs read o_proj as FP8 128x128 \
        blocks") && reasons.contains("found f32"), "{reasons}");
    // Per-tensor FP8 (E4M3 + one FP32 weight_scale) is the same bytes under a uniform grid.
    let mut tensors = glm5_tensors(|name, n, k| fp8(name, n, k, None));
    tensors.retain(|(name, ..)| !name.contains("layers.0.mlp.down_proj"));
    tensors.push(t("model.layers.0.mlp.down_proj.weight", "F8_E4M3", &[6144, 12288]));
    tensors.push(t("model.layers.0.mlp.down_proj.weight_scale", "F32", &[]));
    let report = plan(snapshot(glm5_config(), &tensors).path(), &PlanOptions::default()).unwrap();
    assert_eq!(component(&report, Component::DenseFfn).status, Status::Ready, "{}", render(&report));
    // A BF16 router bias (FP32 in every GLM checkpoint) is refused too.
    let mut tensors = glm5_tensors(|name, n, k| fp8(name, n, k, None));
    tensors.retain(|(name, ..)| !name.ends_with("e_score_correction_bias"));
    tensors.push(t("model.layers.1.mlp.gate.e_score_correction_bias", "BF16", &[256]));
    let report = plan(snapshot(glm5_config(), &tensors).path(), &PlanOptions::default()).unwrap();
    assert_eq!(component(&report, Component::Router).status, Status::MissingKernel);
}

#[test]
fn glm5_flash_mixed_exl3_tiers_share_one_package() {
    let mut tensors = vec![t("model.language_model.layers.1.mlp.gate.weight", "BF16", &[288, 4096])];
    let mut projections = Vec::new();
    for expert in 0..288 {
        let bits = 3 + expert % 2;
        for (proj, n, k) in [("gate_proj", 2048, 4096), ("up_proj", 2048, 4096), ("down_proj", 4096, 2048)] {
            let name = format!("model.language_model.layers.1.mlp.experts.{expert}.{proj}");
            tensors.extend(exl3(&name, n, k, bits));
            projections.push((name, bits, k, n));
        }
    }
    let mut config = glm5_flash_config(2);
    config["quantization_config"] = exl3_compact(3);
    let dir = snapshot(config, &tensors);
    write_quantize_config(dir.path(), &exl3_manifest(&exl3_compact(3), &projections));
    let report = plan(dir.path(), &sparks(3)).unwrap();
    let experts = component(&report, Component::RoutedExpert);
    assert_eq!(experts.status, Status::Ready, "{}", render(&report));
    assert_eq!(experts.formats.len(), 2);
    let contract = report.experts.as_ref().unwrap();
    assert_eq!((contract.package.as_str(), contract.spark_worlds.as_slice()), ("glmf:exl3-k34", &[2, 3, 4, 6][..]));
    assert!(report.executable(), "{}", render(&report));
    assert_eq!(report.expert_storage, Some(Exl3StorageSource::GptqModel));
    // A storage map that disagrees with the tensors is refused, by module.
    let mut lying = exl3_manifest(&exl3_compact(3), &projections);
    lying["tensor_storage"]["model.language_model.layers.1.mlp.experts.0.gate_proj"]["bits_per_weight"] = json!(4);
    lying["tensor_storage"]["model.language_model.layers.1.mlp.experts.0.gate_proj"]["stored_tensors"]
        ["model.language_model.layers.1.mlp.experts.0.gate_proj.trellis"]["shape"] = json!([256, 128, 64]);
    write_quantize_config(dir.path(), &lying);
    let report = plan(dir.path(), &sparks(3)).unwrap();
    let reason = &rejected(&report, Component::RoutedExpert)[0];
    assert!(reason.contains("quantize_config.json disagrees with the safetensors headers at \
        model.language_model.layers.1.mlp.experts.0.gate_proj: map K4"), "{reason}");
    assert!(!report.executable());
    // Without its storage map the expert service derives the same layout.
    std::fs::remove_file(dir.path().join("quantize_config.json")).unwrap();
    let report = plan(dir.path(), &sparks(3)).unwrap();
    assert!(report.executable(), "{}", render(&report));
    assert_eq!(report.expert_storage, Some(Exl3StorageSource::Headers));
}

/// A standard exllamav3 GLM 5.3 Flash checkpoint (brandonmusic/GLM-5.3-Flash-
/// tr3-4bpw): the exllamav3 config block, no storage map, uniform K4 with
/// `[1]` MCG markers. Its experts run on the K3/K4 package with an empty K3
/// tier; its BF16 coordinator block projections are quantized to FP8 blocks at load.
#[test]
fn glm5_flash_standard_exllamav3_checkpoint_is_ready() {
    let mut tensors = vec![t("model.language_model.layers.1.mlp.gate.weight", "BF16", &[288, 4096])];
    for expert in 0..288 {
        for (proj, n, k) in [("gate_proj", 2048, 4096), ("up_proj", 2048, 4096), ("down_proj", 4096, 2048)] {
            let name = format!("model.language_model.layers.1.mlp.experts.{expert}.{proj}");
            let mut projection = exl3(&name, n, k, 4);
            projection[3].2 = vec![1];
            tensors.extend(projection);
        }
    }
    let mut config = glm5_flash_config(2);
    config["quantization_config"] = json!({"quant_method": "exl3", "version": "0.0.43", "bits": 4, "head_bits": 16,
        "codebook": "mcg", "scope": "glm53_routed_experts_only", "non_routed_dtype_policy": "official_source_native",
        "serving_reader_qualified": false});
    let dir = snapshot(config.clone(), &tensors);
    let report = plan(dir.path(), &sparks(4)).unwrap();
    assert!(report.executable(), "{}", render(&report));
    assert_eq!(report.expert_storage, Some(Exl3StorageSource::Headers));
    assert!(render(&report).contains("exl3 map   derived from config.json and the tensor headers"), "{}", render(&report));
    assert_eq!(report.experts.as_ref().unwrap().package, "glmf:exl3-k34");
    let catalog = crate::read_expert_catalog(dir.path()).unwrap();
    assert_eq!(catalog.exl3().unwrap().decoder_tiers(), &[3, 4]);
    // The real publication's BF16 coordinator block projections run FP8 blocks
    // quantized at load (one resident copy).
    let mut with_bf16_projection = tensors.clone();
    with_bf16_projection.push(t("model.language_model.layers.0.mlp.gate_proj.weight", "BF16", &[12288, 4096]));
    let direct = plan(snapshot(config.clone(), &with_bf16_projection).path(), &sparks(4)).unwrap();
    assert_eq!(component(&direct, Component::RoutedExpert).status, Status::Ready);
    assert_eq!(component(&direct, Component::DenseFfn).status, Status::Ready, "{}", render(&direct));
    // An unsupported codebook in config.json stays unsupported, by key.
    let mut mul1 = config.clone();
    mul1["quantization_config"]["codebook"] = json!("mul1");
    let report = plan(snapshot(mul1, &tensors).path(), &sparks(4)).unwrap();
    let reason = &rejected(&report, Component::RoutedExpert)[0];
    assert!(reason.contains("quantization_config.codebook=\"mul1\": this build runs the MCG codebook only"), "{reason}");
}

#[test]
fn glm5_flash_block_projection_policy_distinguishes_mla_dense_and_shared_from_kda() {
    // BF16 MLA, dense and shared projections are quantized to FP8 blocks at load.
    let tensors = [
        t("model.language_model.layers.1.self_attn.q_a_proj.weight", "BF16", &[1536, 4096]),
        t("model.language_model.layers.0.self_attn.o_proj.weight", "BF16", &[4096, 8192]),
        t("model.language_model.layers.0.mlp.gate_proj.weight", "BF16", &[12288, 4096]),
        t("model.language_model.layers.1.mlp.shared_experts.down_proj.weight", "BF16", &[4096, 2048]),
    ];
    let report = plan(snapshot(glm5_flash_config(2), &tensors).path(), &PlanOptions::default()).unwrap();
    for which in [Component::Attention, Component::DenseFfn, Component::SharedExpert] {
        assert_eq!(component(&report, which).status, Status::Ready, "{}", render(&report));
    }
    // Another source format, or the wrong geometry, names the block consumer; KDA's
    // BF16 o_proj is not an MLA FP8 operand and stays supported.
    let tensors = [
        t("model.language_model.layers.1.self_attn.q_a_proj.weight", "F16", &[1536, 4096]),
        t("model.language_model.layers.0.self_attn.o_proj.weight", "BF16", &[4096, 8192]),
        t("model.language_model.layers.0.mlp.gate_proj.weight", "BF16", &[4096, 12288]),
        t("model.language_model.layers.1.mlp.shared_experts.down_proj.weight", "F16", &[4096, 2048]),
    ];
    let report = plan(snapshot(glm5_flash_config(2), &tensors).path(), &PlanOptions::default()).unwrap();
    for which in [Component::Attention, Component::DenseFfn, Component::SharedExpert] {
        let section = component(&report, which);
        assert_eq!((section.status, section.rejected), (Status::MissingKernel, 1), "{}", render(&report));
        assert!(rejected(&report, which)[0].contains("runs FP8 128x128 blocks"));
    }
    let attention = component(&report, Component::Attention);
    assert_eq!(attention.tensors - attention.rejected, 1);
}

#[test]
fn glm5_flash_native_block_projections_and_matching_fp32_scales_are_ready() {
    let mut tensors = Vec::new();
    for (name, n, k) in [
        ("model.language_model.layers.1.self_attn.q_a_proj", 1536, 4096),
        ("model.language_model.layers.1.self_attn.kv_a_proj_with_mqa", 512, 4096),
        ("model.language_model.layers.1.self_attn.q_b_proj", 16384, 1536),
        ("model.language_model.layers.1.self_attn.o_proj", 4096, 16384),
        ("model.language_model.layers.0.mlp.gate_proj", 12288, 4096),
        ("model.language_model.layers.0.mlp.up_proj", 12288, 4096),
        ("model.language_model.layers.0.mlp.down_proj", 4096, 12288),
        ("model.language_model.layers.1.mlp.shared_experts.gate_proj", 2048, 4096),
        ("model.language_model.layers.1.mlp.shared_experts.up_proj", 2048, 4096),
        ("model.language_model.layers.1.mlp.shared_experts.down_proj", 4096, 2048),
    ] { tensors.extend(fp8(name, n, k, None)); }
    let report = plan(snapshot(glm5_flash_config(2), &tensors).path(), &PlanOptions::default()).unwrap();
    for which in [Component::Attention, Component::DenseFfn, Component::SharedExpert] {
        assert_eq!(component(&report, which).status, Status::Ready, "{}", render(&report));
    }
}

#[test]
fn glm5_flash_native_block_projection_rejects_wrong_scale_dtype_grid_or_missing_scale() {
    let name = "model.language_model.layers.1.self_attn.q_b_proj";
    for scale in [Some(t(format!("{name}.weight_scale_inv"), "BF16", &[128, 12])),
                  Some(t(format!("{name}.weight_scale_inv"), "F32", &[127, 12])), None] {
        let mut tensors = vec![t(format!("{name}.weight"), "F8_E4M3", &[16384, 1536])];
        tensors.extend(scale);
        let report = plan(snapshot(glm5_flash_config(2), &tensors).path(), &PlanOptions::default()).unwrap();
        assert_ne!(component(&report, Component::Attention).status, Status::Ready, "{}", render(&report));
        assert!(rejected(&report, Component::Attention).iter().any(|reason| reason.contains(name)),
            "{}", render(&report));
    }
}

/// nvidia/GLM-5.3-Flash-NVFP4's shape: ModelOpt NVFP4 routed experts read
/// from the checkpoint's own hf_quant_config.json; the experts run on the
/// fp8_moe NVFP4 packages (`glmf:nvfp4`, 16-value slices: TP3 too).
#[test]
fn glm5_flash_modelopt_nvfp4_experts_are_ready() {
    let mut tensors = vec![t("model.language_model.layers.1.mlp.gate.weight", "BF16", &[288, 4096])];
    for (proj, n, k) in [("gate_proj", 12288, 4096), ("up_proj", 12288, 4096), ("down_proj", 4096, 12288)] {
        tensors.extend(nvfp4(&format!("model.language_model.layers.0.mlp.{proj}"), n, k));
    }
    for expert in 0..288 {
        for (proj, n, k) in [("gate_proj", 2048, 4096), ("up_proj", 2048, 4096), ("down_proj", 4096, 2048)] {
            tensors.extend(nvfp4(&format!("model.language_model.layers.1.mlp.experts.{expert}.{proj}"), n, k));
        }
    }
    let mut config = glm5_flash_config(2);
    config["quantization_config"] = json!({"quant_method": "modelopt", "quant_algo": "NVFP4",
        "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}},
        "ignore": ["lm_head", "model.language_model.layers.1.mlp.gate"], "producer": {"name": "modelopt"}});
    let dir = snapshot(config, &tensors);
    let hf = json!({"producer": {"name": "modelopt", "version": "0.47"}, "quantization": {"quant_algo": "NVFP4",
        "group_size": 16, "kv_cache_quant_algo": "FP8", "exclude_modules": ["lm_head", "model.language_model.layers.1.mlp.gate"]}});
    std::fs::write(dir.path().join("hf_quant_config.json"), serde_json::to_vec(&hf).unwrap()).unwrap();
    let report = plan(dir.path(), &sparks(3)).unwrap();
    assert_eq!(component(&report, Component::RoutedExpert).status, Status::Ready, "{}", render(&report));
    assert_eq!(component(&report, Component::DenseFfn).status, Status::Ready, "{}", render(&report));
    let contract = report.experts.as_ref().unwrap();
    assert_eq!((contract.package.as_str(), contract.block, contract.spark_worlds.as_slice()),
        ("glmf:nvfp4", 16, &[2, 3, 4, 6][..]));
    assert!(contract.local.is_ok());
    assert!(report.executable(), "{}", render(&report));
    assert!(render(&report).contains("quant      modelopt 0.47 (hf_quant_config.json) NVFP4: every Linear NVFP4 g16"),
        "{}", render(&report));
    // TP3 of 128 blocks: 43 blocks (688 rows) stored 768 wide.
    assert!((report.spark_rank_share - 768.0 / 2048.0).abs() < 1e-9);
    let catalog = crate::read_expert_catalog(dir.path()).unwrap();
    let tensors = catalog.fp8().unwrap();
    assert_eq!(tensors.format(), crate::formats::fp8_experts::ExpertFormat::Nvfp4);
    assert_eq!(tensors.rank_range(3, 2).unwrap(), (1376, 672));
    // Metadata that excludes the experts contradicts their NVFP4 tensors.
    let hf = json!({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4", "group_size": 16,
        "exclude_modules": ["model.language_model.layers.1.mlp.experts*"]}});
    std::fs::write(dir.path().join("hf_quant_config.json"), serde_json::to_vec(&hf).unwrap()).unwrap();
    let report = plan(dir.path(), &sparks(3)).unwrap();
    let reason = &rejected(&report, Component::RoutedExpert)[0];
    assert!(reason.contains("hf_quant_config.json excludes model.language_model.layers.1.mlp.experts.0.down_proj \
        (model.language_model.layers.1.mlp.experts*), but its tensors store NVFP4"), "{reason}");
    assert!(!report.executable());
    // An algorithm this build does not read is a typed refusal, not a guess.
    let hf = json!({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"model.language_model.layers.1.mlp.experts": {"quant_algo": "W4A8_AWQ"}}}});
    std::fs::write(dir.path().join("hf_quant_config.json"), serde_json::to_vec(&hf).unwrap()).unwrap();
    let report = plan(dir.path(), &sparks(3)).unwrap();
    assert!(report.config_error.as_deref().is_some_and(|e| e.contains("W4A8_AWQ")), "{}", render(&report));
    assert!(!report.executable());
}

// --- R07: MiMo serves, with a precise contract ------------------------------

/// The review's observation: the MiMo family reported Planned although
/// serve-mimo runs it. It is Serving now.
#[test]
fn observed_mimo_runtime_still_planned() {
    let report = plan(snapshot_tp(mimo_flash_config(), &mimo_flash_tensors(), Some(1)).path(), &sparks(4)).unwrap();
    assert_eq!(report.runtime, Some(RuntimeStatus::Serving));
    assert!(report.components.iter().all(|c| c.status != Status::Planned), "{}", render(&report));
}

#[test]
fn mimo_flash_complete_inventory_is_ready() {
    let dir = snapshot_tp(mimo_flash_config(), &mimo_flash_tensors(), Some(1));
    let report = plan(dir.path(), &sparks(4)).unwrap();
    assert!(report.executable(), "{}", render(&report));
    // The full layer's k_proj grid restarts per 192-row head (4 heads).
    assert!(component(&report, Component::Attention).formats.contains_key("fp8-block128x128/f32-segmented4"));
    let contract = report.experts.as_ref().unwrap();
    assert_eq!((contract.package.as_str(), contract.block), ("mimo:fp8", 128));
    assert_eq!(contract.spark_worlds, [2, 4, 6]);
    assert!(contract.local.is_ok());
    // fp8-mimo has no tp3 layout (16 blocks do not split in three).
    let report = plan(dir.path(), &sparks(3)).unwrap();
    assert!(!report.placement_supported && !report.executable());
    assert!(report.hints.iter().any(|h| h.what.contains("no mimo:fp8 layout for 3 Spark ranks")));
}

#[test]
fn mimo_pro_complete_inventory_needs_its_packaged_worlds() {
    let dir = snapshot_tp(mimo_pro_config(), &mimo_pro_tensors(), Some(8));
    for (ranks, executable) in [(0, true), (2, true), (3, false), (4, false), (6, true)] {
        let report = plan(dir.path(), &sparks(ranks)).unwrap();
        assert_eq!(report.executable(), executable, "{ranks} ranks\n{}", render(&report));
    }
    let report = plan(dir.path(), &sparks(6)).unwrap();
    let contract = report.experts.as_ref().unwrap();
    assert_eq!((contract.package.as_str(), contract.block, contract.spark_worlds.as_slice()),
        ("mimop:fp8 (MXFP4)", 32, &[2, 6][..]));
    // TP6 of 2048 in 32-row blocks: 352 rows, stored 384 wide.
    assert!((report.spark_rank_share - 384.0 / 2048.0).abs() < 1e-12);
}

#[test]
fn mimo_flash_mopd_tp4_qkv_and_mxfp4_are_ready_without_multimodal_towers() {
    let mut config = mimo_flash_mopd_config();
    config["vision_config"] = json!({"hidden_size": 64});
    let mut tensors = mimo_flash_mopd_tensors();
    tensors.extend([t("visual.blocks.0.norm.weight", "BF16", &[64]),
        t("audio_encoder.norm.weight", "BF16", &[64])]);
    let dir = snapshot_tp(config, &tensors, Some(4));
    for (ranks, executable) in [(0, true), (2, true), (3, false), (4, true), (6, false)] {
        let report = plan(dir.path(), &sparks(ranks)).unwrap();
        assert_eq!(report.executable(), executable, "{ranks}: {}", render(&report));
        let experts = report.experts.as_ref().unwrap();
        assert_eq!((experts.package.as_str(), experts.block, experts.spark_worlds.as_slice()),
            ("mimof:fp8 (MXFP4)", 32, &[2, 4][..]));
        assert_eq!(component(&report, Component::Vision).status, Status::Unused);
        assert!(report.spec.as_ref().unwrap().notes.iter().any(|note| note.contains("family mimof")));
        assert!(report.spec.as_ref().unwrap().notes.iter().any(|note| note.contains("follow checkpoint tensors")));
    }
    let mut wrong = mimo_flash_mopd_tensors();
    wrong.iter_mut().find(|(name, ..)| name == "model.layers.1.mlp.gate.weight").unwrap().1 = "F32";
    let report = plan(snapshot_tp(mimo_flash_mopd_config(), &wrong, Some(4)).path(), &sparks(4)).unwrap();
    assert!(rejected(&report, Component::Router)[0].contains("Bf16"));
    let report = plan(snapshot_tp(mimo_flash_mopd_config(), &tensors, None).path(), &sparks(4)).unwrap();
    assert!(rejected(&report, Component::Attention).iter().any(|error| error.contains("metadata.tp_size")));
}

#[test]
fn audio_defaults_off_without_qualified_bundle_and_explicit_placement_refuses() {
    for config in [mimo_flash_mopd_config(), v41_config(), qwen4_config(1),
        json!({"model_type": "glm5_next"}), json!({"model_type": "deepseek_v4"})] {
        let dir = snapshot(config, &[]);
        assert_eq!(resolve_audio(PlanOptions::default().audio, dir.path()).unwrap(), MediaMode::Off);
        assert_eq!(resolve_audio(MediaMode::Off, dir.path()).unwrap(), MediaMode::Off);
        for mode in [MediaMode::Rtx(None), MediaMode::Spark(Some(0))] {
            assert!(resolve_audio(mode, dir.path()).is_err());
        }
    }
    let dir = snapshot_tp(mimo_flash_mopd_config(), &mimo_flash_mopd_tensors(), Some(4));
    let off = plan(dir.path(), &PlanOptions { audio: MediaMode::Off, ..sparks(4) }).unwrap();
    let auto = plan(dir.path(), &sparks(4)).unwrap();
    assert_eq!(auto.audio, MediaMode::Off);
    assert!(auto.audio_encoder.is_none());
    assert_eq!(off.encoder_plan_hash, auto.encoder_plan_hash);
    assert_eq!(off.hints.iter().map(|h| &h.what).collect::<Vec<_>>(), auto.hints.iter().map(|h| &h.what).collect::<Vec<_>>());
}

#[test]
fn mounted_mimo_audio_inventory_is_independently_admitted() {
    use encoder::EncoderKind;
    use cuteafd_core::memory_layout::DeviceKind;
    const GIB: u64 = 1 << 30;
    let hub = std::path::Path::new("/mnt/sparknest/hf-home/hub");
    for (model, revision, ranks, gpus) in [
        ("Flash", "2479e2d0029eca9a34cc7e7f55a121925f81908e", 4, 1),
        ("Pro", "adea8e2c5373181e5a973fa1ecb343cb31af214b", 6, 2),
    ] {
        let snapshot = hub.join(format!("models--XiaomiMiMo--MiMo-V2.6-{model}-MOPD/snapshots/{revision}"));
        if !snapshot.exists() { continue; }
        assert_eq!(resolve_audio(PlanOptions::default().audio, &snapshot).unwrap(), MediaMode::Auto);
        assert_eq!(resolve_audio(MediaMode::Off, &snapshot).unwrap(), MediaMode::Off);
        let options = PlanOptions { vision: MediaMode::Auto,
            layout: Some(layout::LayoutOptions { rtx_bytes: vec![96 * GIB; gpus], spark_bytes: 121 * GIB,
                ..Default::default() }), ..sparks(ranks) };
        let on = plan(&snapshot, &options).unwrap();
        let audio = on.audio_encoder.as_ref().unwrap();
        assert!(matches!(audio.kind, EncoderKind::Spark { .. }), "{}", render(&on));
        assert!(audio.weights > 2 * GIB && audio.scratch > GIB);
        assert_eq!(component(&on, Component::Audio).status, Status::Ready, "{}", render(&on));
        let owner = match audio.kind { EncoderKind::Spark { rank } => format!("spark{rank} audio"), _ => unreachable!() };
        assert_eq!(on.bytes_by_owner[&owner], audio.admitted_bytes());
        let memory = on.memory_layout.as_ref().unwrap();
        assert_eq!(memory.devices.iter().filter(|d| d.kind == DeviceKind::Spark).flat_map(|d| &d.items).filter(|item| item.group == "audio tower")
            .map(|item| item.bytes).sum::<u64>(), audio.weights);
        assert!(!memory.devices.iter().filter(|d| d.kind == DeviceKind::Rtx).flat_map(|d| &d.items).any(|item| item.group == "audio tower" || item.group == "audio"));
        let off = plan(&snapshot, &PlanOptions { audio: MediaMode::Off, ..options }).unwrap();
        assert!(off.audio_encoder.is_none());
        assert_eq!(on.bytes_by_owner.get("rtx"), off.bytes_by_owner.get("rtx"), "Spark audio must not charge RTX weights");
        assert_ne!(on.encoder_plan_hash, off.encoder_plan_hash);
        let constrained = PlanOptions { vision: MediaMode::Off, layout: Some(layout::LayoutOptions {
            rtx_bytes: vec![1; gpus], spark_bytes: 1, ..Default::default()
        }), ..sparks(ranks) };
        let fallback = plan(&snapshot, &constrained).unwrap();
        assert_eq!(fallback.audio, MediaMode::Off);
        assert_eq!(component(&fallback, Component::Audio).status, Status::Disabled);
        assert!(fallback.placement_supported);
        assert!(fallback.audio_encoder.as_ref().unwrap().shortfall > 0);
        let explicit = plan(&snapshot, &PlanOptions { audio: MediaMode::Spark(Some(0)), ..constrained }).unwrap();
        assert!(!explicit.placement_supported);
    }
}

#[test]
fn mimo_vision_accepts_only_resident_tower_geometry_and_bf16() {
    let mut config = mimo_flash_mopd_config();
    config["vision_config"] = json!({"depth":28,"hidden_size":1280,"intermediate_size":4608,
        "num_heads":32,"num_key_value_heads":8,"out_hidden_size":4096,"patch_size":16,
        "temporal_patch_size":2,"spatial_merge_size":2,"hidden_act":"silu"});
    let mut tensors = mimo_flash_mopd_tensors();
    tensors.extend([t("visual.patch_embed.proj.weight", "BF16", &[1280,3,2,16,16]),
        t("visual.merger.mlp.2.weight", "BF16", &[4096,5120]),
        t("visual.blocks.0.norm1.weight", "BF16", &[1280])]);
    let report = plan(snapshot_tp(config.clone(), &tensors, Some(4)).path(), &sparks(2)).unwrap();
    assert_eq!(component(&report, Component::Vision).status, Status::Ready);
    let mut wrong = tensors.clone();
    wrong.last_mut().unwrap().1 = "F16";
    let report = plan(snapshot_tp(config.clone(), &wrong, Some(4)).path(), &sparks(2)).unwrap();
    assert!(rejected(&report, Component::Vision)[0].contains("BF16"));
    config["vision_config"]["patch_size"] = json!(14);
    let report = plan(snapshot_tp(config, &tensors, Some(4)).path(), &sparks(2)).unwrap();
    assert!(rejected(&report, Component::Vision)[0].contains("patch_size"));
}

#[test]
fn mimo_unsupported_inventories_name_the_tensors() {
    // V2 Flash geometry with MXFP4 experts: the mimo package runs E4M3.
    let mut tensors = mimo_flash_tensors();
    tensors.retain(|(name, ..)| !name.contains(".experts."));
    for (proj, n, k) in [("gate_proj", 2048, 4096), ("up_proj", 2048, 4096), ("down_proj", 4096, 2048)] {
        tensors.extend(mxfp4(&format!("model.layers.1.mlp.experts.0.{proj}"), n, k));
    }
    let report = plan(snapshot_tp(mimo_flash_config(), &tensors, Some(1)).path(), &sparks(4)).unwrap();
    assert_eq!(component(&report, Component::RoutedExpert).status, Status::MissingKernel);
    assert!(rejected(&report, Component::RoutedExpert)[0].contains("mimo:fp8 runs E4M3"));
    // Sinks on a full-attention layer.
    let mut tensors = mimo_flash_tensors();
    tensors.push(t("model.layers.0.self_attn.attention_sink_bias", "BF16", &[64]));
    let report = plan(snapshot_tp(mimo_flash_config(), &tensors, Some(1)).path(), &sparks(4)).unwrap();
    assert!(rejected(&report, Component::Attention)[0].contains("sinks on a full-attention layer"));
    // A configuration with full-layer sinks: the programs refuse every coordinator part.
    let mut config = mimo_flash_config();
    config["add_full_attention_sink_bias"] = json!(true);
    let report = plan(snapshot_tp(config, &mimo_flash_tensors(), Some(1)).path(), &sparks(4)).unwrap();
    assert!(!report.executable());
    assert!(rejected(&report, Component::Embedding)[0].contains("sinks on SWA layers only"));
    // A geometry no program family is built for.
    let mut config = mimo_flash_config();
    config["num_attention_heads"] = json!(32);
    config["swa_num_attention_heads"] = json!(32);
    let report = plan(snapshot_tp(config, &mimo_flash_tensors(), Some(1)).path(), &sparks(4)).unwrap();
    assert!(rejected(&report, Component::Embedding)[0].contains("no mimo program for query/full KV/SWA KV heads"), "{}", render(&report));
}

// --- R08: one canonical configuration per family ----------------------------

fn transformers_spelling(mut config: Value) -> Value {
    let object = config.as_object_mut().unwrap();
    let pattern = object.remove("hybrid_layer_pattern").unwrap();
    let freq = object.remove("moe_layer_freq").unwrap();
    object.insert("layer_types".into(), pattern.as_array().unwrap().iter()
        .map(|p| if p == 0 { "full_attention" } else { "sliding_attention" }).collect());
    object.insert("mlp_layer_types".into(), freq.as_array().unwrap().iter()
        .map(|f| if f == 0 { "dense" } else { "sparse" }).collect());
    config
}

/// The planner's layer, compared field by field with the runtime's reading.
fn assert_layers_match_runtime(spec: &ModelSpec, cfg: &MimoV2Config) {
    assert_eq!(spec.layers.len(), cfg.layers);
    for (layer, planned) in spec.layers.iter().enumerate() {
        let attention = cfg.attention[layer];
        let expected = match attention {
            MimoAttention::Full => AttentionKind::Gqa { heads: cfg.heads, kv_heads: cfg.full_kv_heads, head_dim: cfg.head_dim },
            MimoAttention::Sliding => AttentionKind::SlidingGqa { heads: cfg.heads, kv_heads: cfg.swa_kv_heads,
                head_dim: cfg.head_dim, window: cfg.window, sinks: cfg.swa_sinks },
        };
        assert_eq!(planned.attention, expected, "layer {layer}");
        assert_eq!(planned.ffn == FfnKind::Moe, !cfg.dense[layer], "layer {layer}");
        assert_eq!(planned.rope, Some(spec::RopeSpec { dims: cfg.rope_dim, theta: cfg.rope_theta(attention) }));
    }
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.top_k, moe.intermediate, moe.routed_scaling),
        (cfg.experts, cfg.topk, cfg.moe_intermediate, Some(cfg.routed_scale)));
}

/// The review's observation: the runtime read `layer_types` / `mlp_layer_types`
/// while the planner fell back to all-full, all-dense. Both now read the one
/// canonical configuration.
#[test]
fn observed_mimo_planner_disagrees_with_runtime_layer_types() {
    let config = transformers_spelling(mimo_flash_config());
    let report = plan(snapshot_tp(config.clone(), &mimo_flash_tensors(), Some(1)).path(), &sparks(4)).unwrap();
    let spec = report.spec.as_ref().unwrap();
    assert!(matches!(spec.layers[1].attention, AttentionKind::SlidingGqa { sinks: true, window: 128, .. }));
    assert_eq!(spec.layers[1].ffn, FfnKind::Moe);
    assert_layers_match_runtime(spec, &MimoV2Config::from_hf(&config).unwrap());
    assert!(report.executable(), "{}", render(&report));
}

#[test]
fn mimo_spellings_give_identical_planner_and_runtime_geometry() {
    let hub = mimo_flash_config();
    let transformers = transformers_spelling(hub.clone());
    let mut both = hub.clone();
    both["layer_types"] = transformers["layer_types"].clone();
    both["mlp_layer_types"] = transformers["mlp_layer_types"].clone();
    let runtime = MimoV2Config::from_hf(&hub).unwrap();
    let mut specs = Vec::new();
    for config in [hub, transformers, both] {
        assert_eq!(MimoV2Config::from_hf(&config).unwrap(), runtime);
        let report = plan(snapshot_tp(config, &mimo_flash_tensors(), Some(1)).path(), &sparks(4)).unwrap();
        let spec = report.spec.unwrap();
        assert_layers_match_runtime(&spec, &runtime);
        specs.push((spec.layers, spec.moe));
    }
    assert!(specs.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn mimo_partial_or_conflicting_patterns_are_refused_alike() {
    let base = mimo_flash_config();
    let mut short = base.clone();
    short["moe_layer_freq"] = json!([0]);
    let mut scalar = base.clone();
    scalar["moe_layer_freq"] = json!(1);
    let mut conflict = base.clone();
    conflict["layer_types"] = json!(["full_attention", "full_attention"]);
    let mut unknown = base.clone();
    unknown["hybrid_layer_pattern"] = json!([0, 2]);
    for config in [short, scalar, conflict, unknown] {
        let runtime = MimoV2Config::from_hf(&config).map(|_| ()).unwrap_err();
        let report = plan(snapshot_tp(config, &mimo_flash_tensors(), Some(1)).path(), &sparks(4)).unwrap();
        assert_eq!(report.config_error.as_deref(), Some(format!("{runtime:#}").as_str()));
        assert!(!report.executable() && report.components.is_empty());
    }
}

#[test]
fn mimo_defaults_are_the_runtime_defaults() {
    // No patterns and no sink flags: full attention on layer 0 and every sixth
    // layer, layer 0 dense, SWA sinks on (the runtime's defaults).
    let mut config = mimo_flash_config();
    for key in ["hybrid_layer_pattern", "moe_layer_freq", "add_swa_attention_sink_bias", "add_full_attention_sink_bias",
        "sliding_window_size"] {
        config.as_object_mut().unwrap().remove(key);
    }
    let runtime = MimoV2Config::from_hf(&config).unwrap();
    assert!(runtime.swa_sinks && !runtime.full_sinks);
    let report = plan(snapshot_tp(config, &mimo_flash_tensors(), Some(1)).path(), &sparks(4)).unwrap();
    assert_layers_match_runtime(report.spec.as_ref().unwrap(), &runtime);
    assert!(report.executable(), "{}", render(&report));
}

#[test]
fn glm_and_qwen_patterns_follow_their_runtime_readers() {
    // GLM 5.x: dense layers are a prefix; GLM Flash and Qwen: patterns cover exactly every layer.
    let mut gap = glm5_config();
    gap["mlp_layer_types"] = json!(["sparse", "dense"]);
    gap["first_k_dense_replace"] = json!(0);
    let mut short_flash = glm5_flash_config(2);
    short_flash["text_config"]["layer_types"] = json!(["linear_attention"]);
    let mut short_qwen = qwen4_config(4);
    short_qwen["text_config"]["layer_types"] = json!(vec!["linear_attention"; 3]);
    for (config, needle) in [(gap, "prefix"), (short_flash, "layer_types has 1 entries"), (short_qwen, "layer_types has 3")] {
        let report = plan(snapshot(config, &[t("lm_head.weight", "BF16", &[64, 64])]).path(), &sparks(4)).unwrap();
        assert!(report.config_error.as_deref().is_some_and(|e| e.contains(needle)), "{:?}", report.config_error);
    }
    // GLM 5.3's spec is the runtime's: indexers, dense prefix and RoPE.
    let report = plan(snapshot(glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None))).path(), &sparks(4)).unwrap();
    let cfg = crate::families::glm5::GlmDsaConfig::from_hf(&glm5_config()).unwrap();
    let spec = report.spec.unwrap();
    assert_eq!(spec.layers[1].attention, AttentionKind::MlaDsa { indexer: false });
    assert_eq!(spec.layers.iter().filter(|l| l.ffn != FfnKind::Moe).count(), cfg.first_moe_layer);
    assert_eq!(spec.layers[0].rope, Some(spec::RopeSpec { dims: 64, theta: 8e6 }));
}


fn v4_snapshot() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_v4_snapshot(dir.path());
    dir
}

#[test]
fn deepseek_v4_plan_reads_the_runtime_config_source() {
    let dir = v4_snapshot();
    let report = plan(dir.path(), &sparks(4)).unwrap();
    let spec = report.spec.as_ref().expect("planned from inference/config.json");
    let cfg = crate::families::deepseek_v4::DeepseekV4Config::read(dir.path(), 0).unwrap();
    let ratios: Vec<usize> = spec.layers.iter().map(|l| match l.attention {
        AttentionKind::CompressedMla { ratio, .. } => ratio,
        _ => unreachable!(),
    }).collect();
    assert_eq!(ratios, cfg.compress_ratios);
    assert_eq!(spec.layers[1].rope, Some(spec::RopeSpec { dims: 64, theta: 160000.0 }));
}

#[test]
fn v4_workspace_plan_matches_runtime_below_compiled_context() {
    use crate::serving_capacity::{compiled_c128_width, deepseek_v4_workspace_geometry,
        deepseek_v4_workspace_scratch};
    use cuteafd_core::memory_layout::Basis;
    let dir = v4_snapshot();
    let manifest = json!({"capacities": {"decode_rows": 64, "prefill_rows": 4096, "max_context": 131072},
        "programs": [
            {"family": "dsv4f", "name": "dsv4f_sparse_mla_decode_c128_m64", "params": {"indexed_width": 1024}},
            {"family": "dsv4f", "name": "dsv4f_sparse_mla_prefill_c128_m4096", "params": {"indexed_width": 1024}},
            {"name": "dsv4f_index_topk_decode_m64", "scratch_bytes_at_capacity": {"scratch": 8653824}},
            {"name": "dsv4f_index_topk_prefill_m4096", "scratch_bytes_at_capacity": {"scratch": 558007296}}
        ]});
    let path = dir.path().join("PROGRAMS.json");
    std::fs::write(&path, manifest.to_string()).unwrap();
    let cfg = crate::families::deepseek_v4::DeepseekV4Config::read(dir.path(), 0).unwrap();
    let scratch = deepseek_v4_workspace_scratch(&manifest, "dsv4f", false, 4096, 64).unwrap();
    let runtime = deepseek_v4_workspace_geometry(&cfg, 4096, 64,
        compiled_c128_width(&manifest, "dsv4f").unwrap() * 128, 1, scratch).unwrap();
    for gib in [31.8, 95.5] {
        let total = cuteafd_core::serving_capacity::GpuMemoryBudget::from_gib(gib).unwrap().0;
        for context in [16384, 32512, 131072] {
            let report = plan(dir.path(), &PlanOptions { layout: Some(layout::LayoutOptions {
                rtx_bytes: vec![total], context_tokens: context, pool_tokens: Some(32768),
                workspace_manifest: Some(path.clone()), ..Default::default()
            }), ..sparks(2) }).unwrap();
            let mut layout = report.memory_layout.unwrap();
            let notes = layout.notes.clone();
            let device = layout.devices.remove(0);
            let steps = device.items.iter().find(|i| i.group == "steps").unwrap_or_else(|| panic!("{gib} {context}: {notes:?}"));
            let intake = 2 * 2 * 4096 * cfg.dim as u64 * 2;
            assert_eq!((steps.bytes, steps.basis), (runtime[0].fixed_device_bytes + intake, Basis::Formula));
            let graph = layout::family_costs("deepseek_v4").graph_bytes[0];
            // The PRO envelope covers the measured loaded code as well (`V4Inputs::code_bytes`).
            let code = crate::placement::loaded_code("dsv4f", "*", false, 0).unwrap().bytes;
            assert_eq!(total - device.capacity_bytes, crate::serving_capacity::deepseek_v4_headroom_bytes(
                total, 10 << 30, steps.bytes + code, graph));
        }
    }
}

#[test]
fn v4_missing_workspace_manifest_keeps_conservative_small_card_reserve() {
    use cuteafd_core::serving_capacity::GpuMemoryBudget;
    let dir = v4_snapshot();
    let total = GpuMemoryBudget::from_gib(31.8).unwrap().0;
    let report = plan(dir.path(), &PlanOptions { layout: Some(layout::LayoutOptions {
        rtx_bytes: vec![total], workspace_manifest: Some(dir.path().join("absent.json")),
        ..Default::default()
    }), ..sparks(2) }).unwrap();
    let device = report.memory_layout.unwrap().devices.remove(0);
    let workspace = device.items.iter().find(|i| i.group == "steps").unwrap().bytes;
    let graph = layout::family_costs("deepseek_v4").graph_bytes[0];
    assert_eq!(total - device.capacity_bytes,
        (10u64 << 30).saturating_sub(workspace + graph).max(3 << 30));
}

#[test]
fn v4_pool_first_defaults_and_explicit_layer_failure() {
    let dir = v4_snapshot();
    let solve = |bytes, pool_tokens, local_expert_layers| plan(dir.path(), &PlanOptions {
        layout: Some(layout::LayoutOptions { rtx_bytes: vec![bytes], pool_tokens, local_expert_layers,
            onboard: local_expert_layers.is_none().then_some(crate::placement::Onboard::Auto), ..Default::default() }),
        ..sparks(2)
    }).unwrap();
    let large = solve(96 << 30, None, None);
    assert_eq!(large.memory_layout.unwrap().pool_tokens, 2 << 20);
    let small = solve(24 << 30, None, None);
    assert_eq!(small.memory_layout.unwrap().pool_tokens, 1 << 20);
    let explicit = solve(24 << 30, Some(262144), Some(2));
    assert!(explicit.placement_supported);
    assert_eq!(explicit.memory_layout.unwrap().pool_tokens, 262144);
    // Explicit layers that do not fit beside the fixed demands are refused,
    // naming how many would.
    let rejected = solve(16 << 30, Some(2 << 20), Some(4));
    assert!(!rejected.placement_supported);
    assert!(rejected.memory_layout.as_ref().unwrap().notes.iter().any(|n|
        n.contains("4 RTX expert layers do not fit beside the fixed demands")),
        "{:?}", rejected.memory_layout.as_ref().unwrap().notes);
    // A fixed onboard alone makes the pool the output: above the 1M target here.
    let fixed = plan(dir.path(), &PlanOptions { layout: Some(layout::LayoutOptions { rtx_bytes: vec![24 << 30],
        onboard: Some(crate::placement::Onboard::Layers(1)), ..Default::default() }), ..sparks(2) }).unwrap();
    assert!(fixed.placement_supported);
    assert!(fixed.memory_layout.unwrap().pool_tokens > 1 << 20);
}

#[test]
fn launch_descriptions_match_the_shared_fixtures() {
    let fixtures: Vec<Value> =
        serde_json::from_str(include_str!("../../tests/fixtures/launch-families.json")).unwrap();
    assert!(fixtures.len() >= 10);
    for case in fixtures {
        let described = launch::describe(&case["config"]).map(|d| d.line()).ok();
        assert_eq!(described.as_deref(), case["line"].as_str(), "{}", case["name"]);
    }
}

// --- R09: placement options --------------------------------------------------

/// The review's observation: `--spark-ranks 0` divided by zero. Zero ranks is
/// the local-only placement: every routed expert on the coordinator GPU.
#[test]
fn observed_zero_ranks_panics() {
    let dir = qwen_snapshot(4);
    let report = plan(dir.path(), &sparks(0)).unwrap();
    assert_eq!((report.placement, report.spark_ranks), (ExpertPlacement::Local, 0));
    assert!(report.placement_supported && report.fits, "{}", render(&report));
    assert_eq!(component(&report, Component::RoutedExpert).owner.label(report.placement), "rtx (local)");
    // Over the coordinator budget: the shortfall is named.
    let tight = PlanOptions { coordinator_budget_bytes: 1 << 20, ..sparks(0) };
    let report = plan(dir.path(), &tight).unwrap();
    assert!(!report.fits && report.hints.iter().any(|h| h.what.contains("over the 0 GiB budget")), "{}", render(&report));
    // GLM 5.x has no local expert package.
    let report = plan(snapshot(glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None))).path(), &sparks(0)).unwrap();
    assert!(!report.placement_supported && !report.executable());
    assert!(report.hints.iter().any(|h| h.what.contains("unsupported: no local expert package for glm5")));
}

#[test]
fn spark_rank_options_follow_transport_and_packages() {
    let flash = snapshot_tp(mimo_flash_config(), &mimo_flash_tensors(), Some(1));
    let qwen = qwen_snapshot(4);
    let dsv41 = snapshot(v41_config(), &[t("layers.0.ffn.experts.0.w1.weight", "I8", &[2304, 2560]),
        t("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", &[2304, 160])]);
    for ranks in 0..=9 {
        let options = sparks(ranks);
        if ranks > 8 {
            let error = plan(flash.path(), &options).unwrap_err();
            assert!(matches!(error, PlanError::InvalidOption { option: "spark ranks", .. }), "{ranks}: {error}");
            continue;
        }
        // mimo:fp8: local, tp2/tp4/tp6; qwen4:exl3: local, 1/2/3/4; V4.1: Sparks 2/3/4/6 only.
        assert_eq!(plan(flash.path(), &options).unwrap().placement_supported, matches!(ranks, 0 | 2 | 4 | 6), "mimo {ranks}");
        assert_eq!(plan(qwen.path(), &options).unwrap().placement_supported, matches!(ranks, 0..=4), "qwen {ranks}");
        assert_eq!(plan(dsv41.path(), &options).unwrap().placement_supported, matches!(ranks, 2 | 3 | 4 | 6), "v41 {ranks}");
    }
}

#[test]
fn invalid_budgets_are_typed_option_errors() {
    for gib in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0, 1e30, f64::MIN_POSITIVE] {
        assert!(matches!(budget_bytes("--spark-budget-gib", gib), Err(PlanError::InvalidOption { .. })), "{gib}");
    }
    assert_eq!(budget_bytes("--spark-budget-gib", 100.0).unwrap(), 100 << 30);
    let dir = qwen_snapshot(4);
    for options in [PlanOptions { spark_budget_bytes: 0, ..sparks(4) }, PlanOptions { coordinator_budget_bytes: 0, ..sparks(0) }] {
        assert!(matches!(plan(dir.path(), &options), Err(PlanError::InvalidOption { .. })));
    }
}

#[test]
fn capacity_suggests_a_packaged_rank_count() {
    // One layer of Qwen EXL3 K4 experts.
    let dir = qwen_snapshot(4);
    let routed = component(&plan(dir.path(), &sparks(2)).unwrap(), Component::RoutedExpert).bytes;
    // Qwen's 5 blocks over 2, 3, 4 ranks: the widest holds 3/5, 2/5, 2/5 (padded to 128 rows).
    let budget = (routed as f64 * 0.45) as u64;
    let report = plan(dir.path(), &PlanOptions { spark_budget_bytes: budget, ..sparks(2) }).unwrap();
    assert!(!report.fits && report.min_spark_ranks == Some(3), "{}", render(&report));
    let report = plan(dir.path(), &PlanOptions { spark_budget_bytes: budget, ..sparks(3) }).unwrap();
    assert!(report.fits && report.executable(), "{}", render(&report));
}

#[test]
fn capacity_counts_the_widest_rank_of_an_uneven_split() {
    // 2048 = 16 whole 128-blocks: six ranks hold 3, 3, 3, 3, 2, 2 of them, so the
    // widest carries 3/16 of the routed bytes, not 1/6.
    let mut tensors = vec![t("model.language_model.layers.1.mlp.gate.weight", "BF16", &[288, 4096])];
    for proj in ["gate_proj", "up_proj"] {
        tensors.extend(fp8(&format!("model.language_model.layers.1.mlp.experts.0.{proj}"), 2048, 4096, None));
    }
    tensors.extend(fp8("model.language_model.layers.1.mlp.experts.0.down_proj", 4096, 2048, None));
    let dir = snapshot(glm5_flash_config(2), &tensors);
    let routed = component(&plan(dir.path(), &sparks(6)).unwrap(), Component::RoutedExpert).bytes as f64;
    let report = plan(dir.path(), &PlanOptions { spark_budget_bytes: (routed * 0.18) as u64, ..sparks(6) }).unwrap();
    assert!((report.spark_rank_share - 3.0 / 16.0).abs() < 1e-12, "{}", report.spark_rank_share);
    assert!(!report.fits, "3/16 of the routed bytes exceeds 0.18");
    let report = plan(dir.path(), &PlanOptions { spark_budget_bytes: (routed * 0.19) as u64, ..sparks(6) }).unwrap();
    assert!(report.fits && report.min_spark_ranks == Some(6), "{}", render(&report));
}

#[test]
fn glm5_flash_layout_charges_the_engine_step_workspaces_and_headroom() {
    use crate::families::glm5_flash::GlmNextConfig;
    use crate::serving_capacity::{glmf_manifest_scratch, glmf_step_scratch, glmf_step_workspaces, GlmfStepShape};
    use cuteafd_core::memory_layout::{Basis, Category};
    let config = glm5_flash_config(2);
    let dir = snapshot(config.clone(), &[t("model.language_model.layers.0.self_attn.A_log", "F32", &[64])]);
    // The GLM programs' scratch of an export (the engine reads the same manifest).
    let programs: Vec<Value> = [("glmf_mhc_pre", 26_214_400u64), ("glmf_index_producer_m64", 561_152),
        ("glmf_index_topk_decode_m64", 8_653_824), ("glmf_mhc_post_pre_m64", 409_600), ("glmf_kda_m64", 10_526_720),
        ("glmf_mla_producer_m64", 2_359_296), ("glmf_o_m64", 2_097_152), ("glmf_sparse_mla_decode_m64", 8_404_992),
        ("glmf_ffn_i2048_m64", 786_432), ("glmf_ffn_i12288_m64", 4_718_592), ("glmf_index_producer_m4096", 35_913_728),
        ("glmf_index_topk_prefill_m4096", 558_007_296), ("glmf_mhc_post_pre_m4096", 26_214_400),
        ("glmf_kda_m4096", 782_236_672), ("glmf_mla_producer_m4096", 168_296_448), ("glmf_o_m4096", 203_423_744),
        ("glmf_sparse_mla_prefill_m4096", 1_048_576), ("glmf_ffn_i2048_m4096", 67_633_152),
        ("glmf_ffn_i12288_m4096", 353_894_400)].into_iter()
        .map(|(name, bytes)| json!({"name": name, "scratch_bytes_at_capacity": {"scratch": bytes}})).collect();
    let manifest = json!({"capacities": {"decode_rows": 64, "prefill_rows": 4096, "max_context": 131_072},
        "programs": programs});
    let path = dir.path().join("PROGRAMS.json");
    std::fs::write(&path, manifest.to_string()).unwrap();
    let cfg = GlmNextConfig::from_hf(&config).unwrap();
    let lookup = glmf_manifest_scratch(&manifest);
    // Two lanes of 4,096 rows and the default 2 GiB headroom; four lanes of 2,048 and 1 GiB.
    for (lanes, rows, headroom) in [(2u64, 4096u64, 2u64 << 30), (4, 2048, 1 << 30)] {
        let options = PlanOptions { layout: Some(layout::LayoutOptions { rtx_bytes: vec![32 << 30], prefill_lanes: lanes,
            prefill_rows: rows, headroom_bytes: headroom, context_tokens: 131_072, workspace_manifest: Some(path.clone()),
            ..Default::default() }), ..sparks(4) };
        let memory = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
        let gpu = &memory.devices[0];
        assert_eq!(gpu.capacity_bytes, (32 << 30) - headroom);
        // What the engine allocates from the same arithmetic and manifest.
        let shape = GlmfStepShape { lead: true, split: false, local_experts: false, spark: true, partial_bytes: 2,
            output_shard: false, full_prefill_logits: false, table_pages: 2048, table_pool_pages: 512 };
        let engine = glmf_step_workspaces(&cfg, lanes as usize, rows, 64, &shape,
            glmf_step_scratch(&lookup, &cfg, Default::default(), 64, true).unwrap(),
            glmf_step_scratch(&lookup, &cfg, Default::default(), rows, false).unwrap()).device_bytes();
        let steps = gpu.items.iter().find(|i| i.category == Category::Workspace && i.group == "steps").unwrap();
        assert_eq!((steps.bytes, steps.basis), (engine + lanes * 4 * rows * 4096 * 2, Basis::Formula),
            "the step workspaces and every lane's intake planes");
    }
    // Without a manifest: the one-GPU allowance, never below the default lanes' rows in flight.
    let options = |lanes: u64, rows: u64| PlanOptions { layout: Some(layout::LayoutOptions { rtx_bytes: vec![32 << 30],
        prefill_lanes: lanes, prefill_rows: rows, workspace_manifest: Some(dir.path().join("absent.json")),
        ..Default::default() }), ..sparks(4) };
    let steps = |lanes, rows| plan(dir.path(), &options(lanes, rows)).unwrap().memory_layout.unwrap().devices[0].items
        .iter().find(|i| i.group == "steps").unwrap().bytes;
    let allowance = layout::family_costs("glm5_flash").workspace_bytes[0];
    assert_eq!(steps(2, 4096), allowance + 2 * 4 * 4096 * 4096 * 2);
    assert_eq!(steps(4, 2048), allowance + 4 * 4 * 2048 * 4096 * 2);
    assert_eq!(steps(4, 4096), 2 * allowance + 4 * 4 * 4096 * 4096 * 2);
    // A graph budget replaces the graph allowance, as the engine's admission reserves it.
    let mut budgeted = options(2, 4096);
    budgeted.layout.as_mut().unwrap().graph_budget_bytes = Some(512 << 20);
    let gpu = plan(dir.path(), &budgeted).unwrap().memory_layout.unwrap().devices.remove(0);
    let graphs: Vec<_> = gpu.items.iter().filter(|i| i.group.starts_with("graph")).map(|i| (i.group.as_str(), i.bytes))
        .collect();
    assert_eq!(graphs, [("graph growth", 512 << 20)]);
}

/// `--decode-rows 128` in the planner charges what the engine allocates for it: the decode workspace of
/// 128 rows over both program sets' scratch (planned = allocated, from the same arithmetic and
/// manifest), the wide token selector and sampler, and the speculative replay records and commit tables
/// of 128 rows (with `--replay-records shared`, the KDA records of 128 rows in the prefill scratch), and
/// past a narrower prefill lane the Spark intake planes of 128 rows. A manifest without the wide
/// programs, or a head split, is refused.
#[test]
fn glm5_flash_layout_charges_the_wide_decode_rows() {
    use crate::families::glm5_flash::GlmNextConfig;
    use crate::serving_capacity::{glm_flash_kda_replay_bytes_rows, glm_flash_rank_cache_geometry_rows,
        glmf_manifest_scratch, glmf_selector_bytes, glmf_step_scratch, glmf_step_workspaces, GlmfIndexCache,
        GlmfStepShape};
    use cuteafd_core::memory_layout::{Basis, Category};
    let mut config = glm5_flash_config(2);
    config["text_config"]["vocab_size"] = 154_880.into();
    let dir = snapshot(config.clone(), &[t("model.language_model.layers.0.self_attn.A_log", "F32", &[64])]);
    let base: Vec<(&str, u64)> = vec![("glmf_mhc_pre", 26_214_400u64), ("glmf_index_producer_m64", 561_152),
        ("glmf_index_topk_decode_m64", 8_653_824), ("glmf_mhc_post_pre_m64", 409_600), ("glmf_kda_m64", 10_526_720),
        ("glmf_mla_producer_m64", 2_359_296), ("glmf_o_m64", 2_097_152), ("glmf_sparse_mla_decode_m64", 8_404_992),
        ("glmf_ffn_i2048_m64", 786_432), ("glmf_ffn_i12288_m64", 4_718_592), ("glmf_index_producer_m4096", 35_913_728),
        ("glmf_index_topk_prefill_m4096", 558_007_296), ("glmf_mhc_post_pre_m4096", 26_214_400),
        ("glmf_kda_m4096", 782_236_672), ("glmf_mla_producer_m4096", 168_296_448), ("glmf_o_m4096", 203_423_744),
        ("glmf_sparse_mla_prefill_m4096", 1_048_576), ("glmf_ffn_i2048_m4096", 67_633_152),
        ("glmf_ffn_i12288_m4096", 353_894_400)];
    // The offline SM 12.0 export of the 16 wide programs (170 SMs, the sparse MLA at one split).
    let wide = [("glmf_index_producer_m128", 1_122_304u64), ("glmf_index_topk_decode_m128", 17_304_576),
        ("glmf_mhc_post_pre_m128", 819_200), ("glmf_kda_m128", 21_053_440), ("glmf_mla_producer_m128", 4_718_592),
        ("glmf_o_m128", 4_194_304), ("glmf_sparse_mla_decode_m128", 16_809_984), ("glmf_ffn_i2048_m128", 1_572_864),
        ("glmf_ffn_i12288_m128", 9_437_184)];
    let write = |name: &str, programs: &[(&str, u64)]| {
        let programs: Vec<Value> = programs.iter().map(|(name, bytes)|
            json!({"name": name, "scratch_bytes_at_capacity": {"scratch": bytes}})).collect();
        let manifest = json!({"capacities": {"decode_rows": 64, "prefill_rows": 4096, "max_context": 131_072},
            "programs": programs});
        let path = dir.path().join(name);
        std::fs::write(&path, manifest.to_string()).unwrap();
        (path, manifest)
    };
    let (narrow_path, _) = write("narrow.json", &base);
    let (wide_path, manifest) = write("wide.json", &[base.clone(), wide.to_vec()].concat());
    let cfg = GlmNextConfig::from_hf(&config).unwrap();
    let lookup = glmf_manifest_scratch(&manifest);
    let options = |decode_rows: u64, path: &std::path::Path, gpus: usize| PlanOptions { layout: Some(layout::LayoutOptions {
        rtx_bytes: vec![32 << 30; gpus], context_tokens: 131_072, workspace_manifest: Some(path.to_path_buf()),
        glmf_decode_rows: decode_rows, ..Default::default() }), ..sparks(4) };
    let item = |report: &PlanReport, group: &str| report.memory_layout.as_ref().unwrap().devices[0].items
        .iter().find(|i| i.group == group).map(|i| (i.category, i.bytes, i.basis));
    let refused = |report: &PlanReport, why: &str| report.memory_layout.as_ref().unwrap().notes.iter()
        .any(|note| note.contains(why));
    let (wide_build, one_gpu) = ("CUTEAFD_GLMF_WIDE_DECODE_ROWS=128", "a head split takes --decode-rows 64");
    let shape = GlmfStepShape { lead: true, split: false, local_experts: false, spark: true, partial_bytes: 2,
        output_shard: false, full_prefill_logits: false, table_pages: 2048, table_pool_pages: 512 };
    let intake = 2 * 4 * 4096 * 4096 * 2;
    let narrow = plan(dir.path(), &options(64, &wide_path, 1)).unwrap();
    let broad = plan(dir.path(), &options(128, &wide_path, 1)).unwrap();
    assert_eq!(broad.placement_supported, narrow.placement_supported);
    assert!(!refused(&broad, wide_build) && !refused(&broad, one_gpu));
    for (report, rows) in [(&narrow, 64u64), (&broad, 128)] {
        // What the engine allocates from the same arithmetic and manifest.
        let engine = glmf_step_workspaces(&cfg, 2, 4096, rows, &shape,
            glmf_step_scratch(&lookup, &cfg, Default::default(), rows, true).unwrap(),
            glmf_step_scratch(&lookup, &cfg, Default::default(), 4096, false).unwrap()).device_bytes();
        assert_eq!(item(report, "steps"), Some((Category::Workspace, engine + intake, Basis::Formula)), "{rows} rows");
        // The engine's caches: 2 layers (one KDA, one MLA) of records and commit tables of `rows` rows.
        let geometry = glm_flash_rank_cache_geometry_rows(&cfg, 2, 1, GlmfIndexCache::Keys, 4, rows).unwrap();
        let rank = &geometry.ranks[0];
        assert_eq!(item(report, "state").map(|(_, bytes, _)| bytes), Some(rank.fixed_state_bytes
            + rank.active_state_per_sequence_bytes * 8 + rank.speculative_replay_bytes));
    }
    // 128 rows: +64,606,464 B of decode workspace, the wide selector, one KDA layer's 64 more record rows.
    let steps = |report| item(report, "steps").unwrap().1;
    assert_eq!(steps(&broad) - steps(&narrow), 64_606_464);
    assert_eq!(item(&narrow, "wide decode selector"), None);
    assert_eq!(item(&broad, "wide decode selector"),
        Some((Category::Workspace, glmf_selector_bytes(128, 154_880) - glmf_selector_bytes(64, 154_880), Basis::Formula)));
    assert_eq!(item(&broad, "wide decode selector").unwrap().1, 3_209_984);
    let state = |report| item(report, "state").unwrap().1;
    assert_eq!(state(&broad) - state(&narrow), 9_453_568 + 768);
    // A build without the wide programs, and a head split, cannot run 128 rows.
    let missing = plan(dir.path(), &options(128, &narrow_path, 1)).unwrap();
    assert!(!missing.placement_supported && refused(&missing, wide_build));
    assert!(!refused(&plan(dir.path(), &options(64, &narrow_path, 1)).unwrap(), wide_build));
    let split = plan(dir.path(), &options(128, &wide_path, 2)).unwrap();
    assert!(!split.placement_supported && refused(&split, one_gpu));
    assert!(!refused(&plan(dir.path(), &options(64, &wide_path, 2)).unwrap(), one_gpu));
    // Shared replay records at either row count: the state sheds the KDA layer's records of those rows
    // (18,907,136 B at 128), which the 782,236,672-byte prefill scratch holds without growing.
    let shared = |rows: u64| {
        let mut shared = options(rows, &wide_path, 1);
        shared.layout.as_mut().unwrap().glmf_shared_replay = true;
        plan(dir.path(), &shared).unwrap()
    };
    let (narrow_shared, broad_shared) = (shared(64), shared(128));
    for (report, shared, rows) in [(&narrow, &narrow_shared, 64u64), (&broad, &broad_shared, 128)] {
        let records = glm_flash_kda_replay_bytes_rows(&cfg, 2, 1, rows).unwrap();
        assert_eq!(records, rows / 64 * 9_453_568);
        assert_eq!(state(report) - state(shared), records, "{rows} rows");
        assert_eq!(steps(shared), steps(report), "{rows} rows");
    }
    // Compact producers carry their key|gate rows in scratch: planning must use
    // the same program union as runtime, not only the smaller cache geometry.
    for rows in [64, 128] {
        let mut programs = [base.clone(), wide.to_vec()].concat();
        programs.extend([("glmf_index_producer_c_m64", 16 << 20),
            ("glmf_index_producer_c_m128", 32 << 20), ("glmf_index_producer_c_m4096", 64 << 20)]);
        let (path, compact_manifest) = write("compact.json", &programs);
        let mut compact = options(rows, &path, 1);
        compact.layout.as_mut().unwrap().glmf_index = GlmfIndexCache::Compact;
        let report = plan(dir.path(), &compact).unwrap();
        let lookup = glmf_manifest_scratch(&compact_manifest);
        let scratch_options = crate::serving_capacity::GlmfScratchOptions {
            index_compact: true, ..Default::default()
        };
        let engine = glmf_step_workspaces(&cfg, 2, 4096, rows, &shape,
            glmf_step_scratch(&lookup, &cfg, scratch_options, rows, true).unwrap(),
            glmf_step_scratch(&lookup, &cfg, scratch_options, 4096, false).unwrap()).device_bytes();
        assert_eq!(item(&report, "steps"), Some((Category::Workspace, engine + intake, Basis::Formula)));
    }
    // A prefill lane narrower than a verify step (`--prefill-rows 64 --decode-rows 128`): every lane's
    // intake planes hold the widest step's rows, as the engine's Spark transports do; lanes of 4,096
    // rows keep theirs.
    let lane = |prefill_rows: u64, decode_rows: u64| {
        let mut options = options(decode_rows, &wide_path, 1);
        options.layout.as_mut().unwrap().prefill_rows = prefill_rows;
        plan(dir.path(), &options).unwrap()
    };
    for (rows, decode_rows, intake_rows) in [(64u64, 64u64, 64u64), (64, 128, 128), (4096, 128, 4096)] {
        let engine = glmf_step_workspaces(&cfg, 2, rows, decode_rows, &shape,
            glmf_step_scratch(&lookup, &cfg, Default::default(), decode_rows, true).unwrap(),
            glmf_step_scratch(&lookup, &cfg, Default::default(), rows, false).unwrap()).device_bytes();
        let intake = crate::serving_capacity::glmf_spark_intake_bytes(2, 4, intake_rows, 4096);
        assert_eq!(intake, 2 * 4 * intake_rows * 4096 * 2);
        assert_eq!(item(&lane(rows, decode_rows), "steps"), Some((Category::Workspace, engine + intake, Basis::Formula)),
            "{rows} prefill rows, {decode_rows} decode rows");
    }
    // 128 rows past a 64-row lane: the wide decode workspace and 64 more intake rows per lane and Spark.
    assert_eq!(steps(&lane(64, 128)) - steps(&lane(64, 64)), 64_606_464 + 2 * 4 * 64 * 4096 * 2);
}

#[test]
fn glm5_flash_index_layout_pool_matches_runtime_geometry() {
    use crate::families::glm5_flash::GlmNextConfig;
    use crate::serving_capacity::{glm_flash_rank_cache_geometry_rows, GlmfIndexCache};
    use cuteafd_core::memory_layout::Category;
    let config = glm5_flash_config(2);
    let dir = snapshot(config.clone(), &[t("model.language_model.layers.0.self_attn.A_log", "F32", &[64])]);
    let cfg = GlmNextConfig::from_hf(&config).unwrap();
    for gpus in [1, 2] {
        for index in [GlmfIndexCache::Keys, GlmfIndexCache::Compact] {
            for rows in if gpus == 1 { vec![64, 128] } else { vec![64] } {
                for pool_marks in [false, true] {
                    for shared in [false, true] {
                        let options = PlanOptions { layout: Some(layout::LayoutOptions {
                            rtx_bytes: vec![24 << 30; gpus], context_tokens: 131_072,
                            target_pool_tokens: 16_777_216, glmf_index: index, glmf_decode_rows: rows, glmf_pool_marks: pool_marks,
                            glmf_shared_replay: shared, state_slots: Some(8), ..Default::default()
                        }), ..sparks(2) };
                        let report = plan(dir.path(), &options).unwrap();
                        let memory = report.memory_layout.unwrap();
                        let served_index = if gpus == 2 { GlmfIndexCache::Keys } else { index };
                        let geometry = glm_flash_rank_cache_geometry_rows(&cfg, cfg.layers, gpus,
                            served_index, 4, rows).unwrap();
                        let unit = geometry.logical_unit_rows;
                        let mut served_pool = 16_777_216;
                        for (device, rank) in memory.devices.iter().filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx)
                            .zip(&geometry.ranks) {
                            let records = device.items.iter().find(|i| i.category == Category::Kv && i.group == "records")
                                .unwrap().bytes;
                            let bytes = rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes;
                            assert_eq!(records, memory.pool_tokens.div_ceil(unit) * bytes);
                            let available = (device.free_bytes() + records as i64).max(0) as u64;
                            served_pool = served_pool.min(available / bytes.div_ceil(unit) / unit * unit);
                            let state = device.items.iter().find(|i| i.category == Category::Kv && i.group == "state")
                                .unwrap().bytes;
                            let replay = if shared && gpus == 1 {
                                crate::serving_capacity::glm_flash_kda_replay_bytes_rows(&cfg, cfg.layers, 1, rows).unwrap()
                            } else { 0 };
                            assert_eq!(state, rank.fixed_state_bytes + rank.active_state_per_sequence_bytes * 8
                                + rank.speculative_replay_bytes - replay);
                            if pool_marks {
                                let reserve = device.items.iter().find(|i| i.group == "reserved units").unwrap().bytes;
                                assert_eq!(reserve, bytes * crate::serving_capacity::GLMF_POOL_MARK_RESERVED_UNITS);
                            }
                        }
                        assert!(served_pool > 0,
                            "{index:?} gpus={gpus} rows={rows} pool_marks={pool_marks} shared={shared}: {memory:?}");
                        assert_eq!(memory.pool_tokens, served_pool,
                            "{index:?} gpus={gpus} rows={rows} pool_marks={pool_marks} shared={shared}");
                    }
                }
            }
        }
    }
}

#[test]
fn glm5_flash_layout_moves_shared_replay_records_into_the_prefill_scratch() {
    use crate::serving_capacity::glm_flash_kda_replay_bytes;
    use cuteafd_core::memory_layout::Category;
    let config = glm5_flash_config(2);
    let dir = snapshot(config.clone(), &[t("model.language_model.layers.0.self_attn.A_log", "F32", &[64])]);
    // `small`: every prefill program's scratch 1 MiB, under the records.
    let manifest = |small: bool| {
        let prefill = |bytes: u64| if small { 1 << 20 } else { bytes };
        let programs: Vec<Value> = [("glmf_mhc_pre", prefill(26_214_400)), ("glmf_index_producer_m64", 561_152),
            ("glmf_index_topk_decode_m64", 8_653_824), ("glmf_mhc_post_pre_m64", 409_600), ("glmf_kda_m64", 10_526_720),
            ("glmf_mla_producer_m64", 2_359_296), ("glmf_o_m64", 2_097_152), ("glmf_sparse_mla_decode_m64", 8_404_992),
            ("glmf_ffn_i2048_m64", 786_432), ("glmf_ffn_i12288_m64", 4_718_592),
            ("glmf_index_producer_m4096", prefill(35_913_728)), ("glmf_index_topk_prefill_m4096", 558_007_296),
            ("glmf_mhc_post_pre_m4096", prefill(26_214_400)), ("glmf_kda_m4096", prefill(782_236_672)),
            ("glmf_mla_producer_m4096", prefill(168_296_448)), ("glmf_o_m4096", prefill(203_423_744)),
            ("glmf_sparse_mla_prefill_m4096", prefill(1_048_576)), ("glmf_ffn_i2048_m4096", prefill(67_633_152)),
            ("glmf_ffn_i12288_m4096", prefill(353_894_400))].into_iter()
            .map(|(name, bytes)| json!({"name": name, "scratch_bytes_at_capacity": {"scratch": bytes}})).collect();
        json!({"capacities": {"decode_rows": 64, "prefill_rows": 4096, "max_context": 131_072}, "programs": programs})
    };
    let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(&config).unwrap();
    // One KDA layer's records (the fixture has one KDA and one MLA layer).
    let records = glm_flash_kda_replay_bytes(&cfg, 2, 1).unwrap();
    assert_eq!(records, 9_453_568);
    let report = |path: &std::path::Path, shared: bool, pool: Option<u64>| {
        let options = PlanOptions { layout: Some(layout::LayoutOptions { rtx_bytes: vec![32 << 30],
            context_tokens: 131_072, workspace_manifest: Some(path.to_path_buf()), glmf_shared_replay: shared,
            pool_tokens: pool, ..Default::default() }), ..sparks(4) };
        plan(dir.path(), &options).unwrap()
    };
    let item = |path: &std::path::Path, shared: bool, pool: Option<u64>, category: Category, group: &str| {
        report(path, shared, pool).memory_layout.unwrap().devices[0].items.iter()
            .find(|i| i.category == category && i.group == group).unwrap().bytes
    };
    // The export's prefill scratch (782 MB, the 4,096-row KDA programs) holds the records: the state
    // sheds them and the step workspaces do not grow.
    let path = dir.path().join("PROGRAMS.json");
    std::fs::write(&path, manifest(false).to_string()).unwrap();
    assert_eq!(item(&path, false, None, Category::Kv, "state") - item(&path, true, None, Category::Kv, "state"), records);
    assert_eq!(item(&path, false, None, Category::Workspace, "steps"), item(&path, true, None, Category::Workspace, "steps"));
    // A prefill scratch smaller than the records grows to hold them.
    let small = dir.path().join("SMALL.json");
    std::fs::write(&small, manifest(true).to_string()).unwrap();
    assert_eq!(item(&small, true, None, Category::Workspace, "steps"),
        item(&small, false, None, Category::Workspace, "steps") + records - (1 << 20));
    assert_eq!(item(&small, false, None, Category::Kv, "state") - item(&small, true, None, Category::Kv, "state"), records);
    // A fixed pool admits from the planner's costs, where the engine keeps the records of their own.
    let fixed = Some(131_072);
    assert_eq!(item(&path, true, fixed, Category::Kv, "state"), item(&path, false, fixed, Category::Kv, "state"));
    let notes = report(&path, true, fixed).memory_layout.unwrap().notes.join("\n");
    assert!(notes.contains("--replay-records shared needs one GPU"), "{notes}");
}

/// A GLM 5.3 Flash graph budget is kept as the engine's KV admission keeps it: the budget itself on one
/// GPU with Spark experts and an automatic pool (measured), else the budget or the graph allowance,
/// whichever is larger, on every GPU (a head split, a fixed pool, local experts: planned).
#[test]
fn glm5_flash_layout_keeps_the_graph_budget_as_the_admission_does() {
    use cuteafd_core::memory_layout::DeviceKind;
    let dir = snapshot(glm5_flash_config(2), &[t("model.language_model.layers.0.self_attn.A_log", "F32", &[64])]);
    let allowance = layout::family_costs("glm5_flash").graph_bytes;
    // Each coordinator GPU's graph item for `gpus` GPUs, a pool (None: automatic), `ranks` Sparks (0:
    // local experts) and a budget of `mib` MiB.
    let graphs = |gpus: usize, pool: Option<u64>, ranks: usize, mib: u64| -> Vec<(String, u64)> {
        let options = PlanOptions { layout: Some(layout::LayoutOptions { rtx_bytes: vec![96 << 30; gpus],
            pool_tokens: pool, graph_budget_bytes: Some(mib << 20), ..Default::default() }), ..sparks(ranks) };
        let memory = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
        memory.devices.iter().filter(|d| d.kind == DeviceKind::Rtx).flat_map(|d| d.items.iter()
            .filter(|i| i.group.starts_with("graph")).map(|i| (i.group.clone(), i.bytes))).collect()
    };
    let budget = |mib: u64| ("graph growth".to_string(), mib << 20);
    let kept = ("graph growth".to_string(), allowance[0]);
    // Measured: the budget itself, below the allowance or above it.
    assert_eq!(graphs(1, None, 4, 512), [budget(512)]);
    assert_eq!(graphs(1, None, 4, 4096), [budget(4096)]);
    // Planned: a fixed pool, local experts, a head split (both GPUs).
    for (gpus, pool, ranks) in [(1, Some(131_072), 4), (1, None, 0), (2, None, 4), (2, Some(131_072), 4)] {
        assert_eq!(graphs(gpus, pool, ranks, 512), vec![kept.clone(); gpus], "{gpus} {pool:?} {ranks}");
        assert_eq!(graphs(gpus, pool, ranks, 1536), vec![kept.clone(); gpus], "{gpus} {pool:?} {ranks}");
        assert_eq!(graphs(gpus, pool, ranks, 4096), vec![budget(4096); gpus], "{gpus} {pool:?} {ranks}");
    }
}

#[test]
fn glm5_flash_disabled_draft_omits_arenas_and_speculative_graphs() {
    use cuteafd_core::memory_layout::Category;
    let dir = snapshot(glm5_flash_config(2), &[t("model.language_model.layers.0.self_attn.A_log", "F32", &[64])]);
    let options = |disabled| PlanOptions { layout: Some(layout::LayoutOptions {
        glmf_drafter_disabled: disabled, context_tokens: 131_072, ..Default::default() }), ..sparks(4) };
    let enabled = plan(dir.path(), &options(false)).unwrap().memory_layout.unwrap();
    let disabled = plan(dir.path(), &options(true)).unwrap().memory_layout.unwrap();
    assert!(enabled.devices[0].items.iter().any(|i| i.category == Category::Drafter));
    assert!(!disabled.devices[0].items.iter().any(|i| i.category == Category::Drafter));
    let graphs = |layout: &cuteafd_core::memory_layout::MemoryLayout| layout.devices[0].items.iter()
        .find(|i| i.group == "graphs").unwrap().bytes;
    assert!(graphs(&disabled) < graphs(&enabled), "disabled={} enabled={}", graphs(&disabled), graphs(&enabled));
    // The measured loaded code (`placement::inventory::LOADED_CODE`) replaces the per-workspace
    // runtime allowance: it already holds the workspaces' and the drafter's untracked memory.
    let code = crate::placement::loaded_code("glmf", "*", false, 0).unwrap().bytes;
    for layout in [&enabled, &disabled] {
        assert!(!layout.devices[0].items.iter().any(|i| i.group == "workspace runtime overhead"));
        let context = layout.devices[0].items.iter().find(|i| i.group == "context+modules").unwrap().bytes;
        assert_eq!(context, crate::placement::ArchContext::coordinator(crate::placement::inventory::PRO_TOTAL_BYTES,
            None).context_bytes + code);
    }
}

#[test]
fn glm_next_facts_and_dflash2_drafter_are_described() {
    let dir = snapshot(glm5_flash_config(2), &[t("model.language_model.layers.0.self_attn.A_log", "F32", &[64])]);
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    let notes = report.spec.as_ref().unwrap().notes.join("\n");
    assert!(notes.contains("record 528 B"), "{notes}");
    assert!(notes.contains("dense causal up to 2051 tokens"), "{notes}");
    assert!(notes.contains("recurrent state 4.0 MiB"), "{notes}");
    assert!(notes.contains("SwiGLU clamp 10"), "{notes}");

    let draft = snapshot(
        json!({"architectures": ["DFlash2DraftModel"], "hidden_size": 64, "vocab_size": 8,
               "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16,
               "sliding_window": 2048, "intermediate_size": 128, "num_target_layers": 45,
               "dflash_config": {"block_size": 8, "target_layer_ids": [5, 14]}}),
        &[t("fc.weight", "BF16", &[64, 128])],
    );
    let report = plan(draft.path(), &PlanOptions::default()).unwrap();
    assert_eq!(report.family.as_deref(), Some("dflash2"));
    assert!(report.spec.unwrap().notes[0].contains("taps [5, 14] of a 45-layer target"));
}

#[test]
fn formats_count_logical_weights() {
    let dir = snapshot_tp(mimo_flash_config(), &mimo_flash_tensors(), Some(1));
    let report = plan(dir.path(), &sparks(4)).unwrap();
    let total: BTreeMap<Component, usize> =
        report.components.iter().map(|c| (c.component, c.formats.values().sum())).collect();
    // q, k, v, o for two layers, plus one sink.
    assert_eq!(total[&Component::Attention], 9);
}

#[test]
fn layout_places_every_device_and_names_padded_spark_slices() {
    use crate::plan::testing::{mimo_pro_config, mimo_pro_tensors, write_snapshot};
    use cuteafd_core::memory_layout::{Category, DeviceKind};
    let dir = tempfile::tempdir().unwrap();
    write_snapshot(dir.path(), &mimo_pro_config(), &mimo_pro_tensors(), Some(8));
    let gib = 1u64 << 30;
    let options = PlanOptions {
        layout: Some(layout::LayoutOptions { rtx_bytes: vec![96 * gib, 96 * gib], ..Default::default() }),
        ..sparks(6)
    };
    let report = plan(dir.path(), &options).unwrap();
    let layout = report.memory_layout.as_ref().expect("layout requested");
    let rtx = layout.devices.iter().filter(|d| d.kind == DeviceKind::Rtx).count();
    let spark = layout.devices.iter().filter(|d| d.kind == DeviceKind::Spark).count();
    assert_eq!((rtx, spark), (2, 6));
    // Head split: both GPUs hold attention weights; only the lead holds the embedding.
    assert!(layout.devices[1].by_category().get(&Category::Weights).copied().unwrap_or(0) > 0);
    assert_eq!(layout.devices[1].by_category().get(&Category::Embedding), None);
    // The pool fills what the tighter GPU has left, within the target, in whole pages.
    assert!(layout.pool_tokens > 0 && layout.pool_tokens <= cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS);
    assert_eq!(layout.pool_tokens % 64, 0);
    assert!(layout.devices.iter().filter(|d| d.kind == DeviceKind::Rtx).all(|d| d.free_bytes() >= 0));
    // 2048 over six ranks: exact whole-block slices (384 x 4, 256 x 2), no padding.
    let experts = |i: usize| layout.devices[2 + i].by_category()[&Category::Experts];
    assert_eq!(experts(0) * 2, experts(4) * 3);
    assert!(!layout.waste.iter().any(|w| w.what.contains("padded")), "{:?}", layout.waste);
}

#[test]
fn host_embedding_removes_only_the_lead_copy_before_pool_admission() {
    use crate::plan::testing::{mimo_pro_config, mimo_pro_tensors, write_snapshot};
    use cuteafd_core::memory_layout::Category;
    let dir = tempfile::tempdir().unwrap();
    let mut config = mimo_pro_config();
    config["tie_word_embeddings"] = json!(false);
    write_snapshot(dir.path(), &config, &mimo_pro_tensors(), Some(8));
    let mut options = PlanOptions {
        layout: Some(layout::LayoutOptions { rtx_bytes: vec![24 << 30, 96 << 30], force_gpu_embedding: true, ..Default::default() }),
        ..sparks(6)
    };
    let probe = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    let bytes = probe.devices[0].by_category()[&Category::Embedding];
    assert!(bytes > 0);
    // Leave only one table's worth of KV room, below this small fixture's target cap.
    options.layout.as_mut().unwrap().rtx_bytes[0] =
        probe.devices[0].used_bytes()
            - probe.devices[0].items.iter().filter(|i| i.category == Category::Kv && i.group == "records").map(|i| i.bytes).sum::<u64>()
            + bytes + options.layout.as_ref().unwrap().headroom_bytes
                .max(cuteafd_core::serving_capacity::SMALL_CARD_HEADROOM_BYTES);
    let gpu = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert_eq!(gpu.devices[1].by_category().get(&Category::Embedding), None);
    options.layout.as_mut().unwrap().host_embedding = true;
    let report = plan(dir.path(), &options).unwrap();
    let host = report.memory_layout.unwrap();
    assert!(!host.devices.iter().any(|d| d.by_category().contains_key(&Category::Embedding)));
    assert!(host.pool_tokens > gpu.pool_tokens, "host={} GPU={}", host.pool_tokens, gpu.pool_tokens);
    assert_eq!(host.devices[1].by_category().get(&Category::Weights), gpu.devices[1].by_category().get(&Category::Weights));
    assert!(host.notes.iter().any(|n| n.contains(&format!("{bytes} device bytes freed"))));
    config["tie_word_embeddings"] = json!(true);
    write_snapshot(dir.path(), &config, &mimo_pro_tensors(), Some(8));
    let tied = plan(dir.path(), &options).unwrap();
    assert!(!tied.fits);
    assert!(tied.hints.iter().any(|h| h.what.contains("tie_word_embeddings")));
    assert_eq!(tied.memory_layout.unwrap().devices[0].by_category()[&Category::Embedding], bytes);
}

#[test]
fn mimo_small_card_auto_embedding_and_full_context() {
    use cuteafd_core::memory_layout::Category;
    let dir = tempfile::tempdir().unwrap();
    let mut config = mimo_pro_config();
    config["max_position_embeddings"] = json!(1_048_576);
    config["tie_word_embeddings"] = json!(false);
    write_snapshot(dir.path(), &config, &mimo_pro_tensors(), Some(8));
    let mut options = PlanOptions { layout: Some(layout::LayoutOptions {
        rtx_bytes: vec![32 << 30], ..Default::default()
    }), ..sparks(4) };
    let base = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert!(!base.devices[0].by_category().contains_key(&Category::Embedding));
    assert_eq!(base.devices[0].capacity_bytes,
        (32u64 << 30) - cuteafd_core::serving_capacity::SMALL_CARD_HEADROOM_BYTES);
    assert!(base.pool_tokens >= 1_048_576, "{}", base.render());
    assert!(!base.devices[0].items.iter().any(|i| i.group == "DFlash context marks"));
    options.layout.as_mut().unwrap().force_gpu_embedding = true;
    assert!(plan(dir.path(), &options).unwrap().memory_layout.unwrap().devices[0].by_category().contains_key(&Category::Embedding));
}

#[test]
fn mimo_concurrency_default_is_small_card_only_and_overridable() {
    let dir = tempfile::tempdir().unwrap();
    write_snapshot(dir.path(), &mimo_pro_config(), &mimo_pro_tensors(), Some(8));
    for (gib, expected) in [(32, 16), (96, 8)] {
        let mut options = PlanOptions { layout: Some(layout::LayoutOptions {
            rtx_bytes: vec![gib << 30], ..Default::default()
        }), ..sparks(4) };
        let automatic = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
        options.layout.as_mut().unwrap().concurrency = expected;
        let explicit = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
        assert_eq!(automatic.pool_tokens, explicit.pool_tokens);
        assert_eq!(automatic.devices[0].used_bytes(), explicit.devices[0].used_bytes());
        options.layout.as_mut().unwrap().concurrency = if expected == 16 { 8 } else { 16 };
        let overridden = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
        let state = |layout: &cuteafd_core::memory_layout::MemoryLayout| layout.devices[0].items.iter()
            .find(|item| item.group == "state").unwrap().bytes;
        // Both concurrency limits fit the same sixteen physical target rings.
        assert_eq!(state(&automatic), state(&overridden));
        options.layout.as_mut().unwrap().mimo_rings = 32;
        let more_rings = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
        assert!(state(&more_rings) > state(&automatic));
    }
}

#[test]
fn mimo_draft_prefix_ledger_is_opt_in_and_checked() {
    let dir = tempfile::tempdir().unwrap();
    write_snapshot(dir.path(), &mimo_pro_config(), &mimo_pro_tensors(), Some(8));
    let mut options = PlanOptions { layout: Some(layout::LayoutOptions {
        concurrency: 16, ..Default::default()
    }), ..sparks(4) };
    let base = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert!(!base.devices[0].items.iter().any(|i| i.group == "DFlash context marks"));
    let draft = dir.path().join("dflash");
    std::fs::create_dir(&draft).unwrap();
    std::fs::write(draft.join("config.json"), json!({"num_hidden_layers": 5,
        "num_key_value_heads": 8, "head_dim": 128}).to_string()).unwrap();
    let disabled = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert_eq!(disabled.devices[0].used_bytes(), base.devices[0].used_bytes());
    options.layout.as_mut().unwrap().mimo_prefix_draft = true;
    let marked = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    let bytes = marked.devices[0].items.iter().find(|i| i.group == "DFlash context marks").unwrap().bytes;
    assert_eq!(bytes, (20 * (1 << 20) + 8) * 42);
    assert_eq!(marked.devices[0].items.iter().find(|i| i.group == "DFlash valid-floor transfer").unwrap().bytes, 16 * 8);
    options.layout.as_mut().unwrap().draft_context_slots = Some(25);
    let overridden = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert_eq!(overridden.devices[0].items.iter().find(|i| i.group == "DFlash valid-floor transfer").unwrap().bytes, 16 * 8);
    assert!(overridden.notes.iter().any(|note| note.contains("context slots 25")));
    std::fs::write(draft.join("config.json"), "{}").unwrap();
    let malformed = plan(dir.path(), &options).unwrap();
    assert!(!malformed.placement_supported);
    assert!(!malformed.executable());
}

#[test]
fn qwen_layout_reserves_recurrent_state_before_auto_pool_and_leaves_peer_idle() {
    use cuteafd_core::memory_layout::Category;
    let dir = snapshot(qwen4_config(48), &[]);
    let options = PlanOptions {
        layout: Some(layout::LayoutOptions {
            rtx_bytes: vec![12 << 30, 96 << 30], pool_tokens: Some(0), ..Default::default()
        }), ..sparks(0)
    };
    let report = plan(dir.path(), &options).unwrap();
    let memory = report.memory_layout.as_ref().unwrap();
    assert!(memory.pool_tokens > 0);
    assert_eq!(memory.pool_tokens % 256, 0);
    assert!(memory.devices[0].free_bytes() >= 0, "{}", memory.render());
    assert!(memory.devices[0].by_category()[&Category::Kv] > 0);
    assert_eq!(memory.devices[1].used_bytes(), 0);
    assert_eq!(memory.devices[1].kv_tokens, 0);
    let mut implicit = options.clone();
    implicit.layout.as_mut().unwrap().pool_tokens = None;
    assert_eq!(plan(dir.path(), &implicit).unwrap().memory_layout.unwrap().pool_tokens, memory.pool_tokens);
}

#[test]
fn v41_small_card_graph_envelope_preserves_pro_allowance() {
    let dir = snapshot(v41_config(), &[]);
    for (gib, expected) in [(32, 2u64 << 30), (96, (1u64 << 30) * 150 / 100)] {
        let options = PlanOptions {
            layout: Some(layout::LayoutOptions { rtx_bytes: vec![gib << 30], ..Default::default() }),
            ..sparks(4)
        };
        let memory = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
        let graph = memory.devices[0].items.iter().find(|item| item.group == "graph allowance").unwrap();
        assert_eq!(graph.bytes, expected);
    }
}

#[test]
fn v41_auto_layout_honors_occupancy_and_disabled_prefix_arenas() {
    use cuteafd_core::memory_layout::Category;
    let mut config = v41_config();
    let text = &mut config["text_config"];
    text["num_hidden_layers"] = json!(40);
    text["head_dim"] = json!(512);
    text["qk_rope_head_dim"] = json!(64);
    text["sliding_window"] = json!(128);
    text["kv_source_layer_ids"] = json!([2, 8, 14, 20]);
    text["compress_ratios"] = json!((0..40).map(|l| if l < 2 { 0 } else if l < 20 { 2 } else { 1 }).collect::<Vec<_>>());
    let dir = snapshot(config, &[t("embed.weight", "BF16", &[128, 5120])]);
    let mut options = sparks(4);
    options.layout = Some(layout::LayoutOptions { rtx_bytes: vec![96 << 30], pool_tokens: Some(0),
        prefix_slots: Some(0), native_mtp_layers: 0, ..Default::default() });
    let automatic = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert_eq!(automatic.devices[0].capacity_bytes, (96u64 << 30) * 97 / 100 - (3 << 30));
    assert_eq!(automatic.devices[0].by_category().get(&Category::Prefix).copied().unwrap_or(0), 0);
    let state = automatic.devices[0].items.iter().find(|i| i.group == "state").unwrap().bytes;
    let cache = crate::serving_capacity::deepseek_v41_cache_geometry(&serde_json::from_reader::<_, Value>(
        std::fs::File::open(dir.path().join("config.json")).unwrap()).unwrap(), 1).unwrap();
    assert_eq!(state, cache.ranks[0].active_state_per_sequence_bytes * 16);
    options.layout.as_mut().unwrap().pool_tokens = Some(512);
    let explicit = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert_eq!(explicit.devices[0].capacity_bytes, (96u64 << 30) - (3 << 30));
}

#[test]
fn v41_vision_off_removes_only_the_tower_from_fixed_layout() {
    let mut config = v41_config();
    let text = &mut config["text_config"];
    text["num_hidden_layers"] = json!(40);
    text["head_dim"] = json!(512);
    text["qk_rope_head_dim"] = json!(64);
    text["sliding_window"] = json!(128);
    text["kv_source_layer_ids"] = json!([2, 8, 14, 20]);
    text["compress_ratios"] = json!((0..40).map(|l| if l < 2 { 0 } else if l < 20 { 2 } else { 1 }).collect::<Vec<_>>());
    let dir = snapshot(config, &[t("embed.weight", "BF16", &[128, 5120]),
        t("head.weight", "BF16", &[128, 5120]),
        t("vision.patch_embed.proj.weight", "BF16", &[16, 3, 14, 14])]);
    let mut options = sparks(4);
    options.vision = MediaMode::Auto;
    options.layout = Some(layout::LayoutOptions { rtx_bytes: vec![32 << 30], pool_tokens: Some(512),
        local_expert_layers: Some(0), native_mtp_layers: 0, ..Default::default() });
    use super::encoder::EncoderKind;
    let auto_report = plan(dir.path(), &options).unwrap();
    assert_eq!(auto_report.encoder.as_ref().unwrap().kind, EncoderKind::Spark { rank: 0 });
    let auto = auto_report.memory_layout.unwrap();
    let vision_item = |i: &&cuteafd_core::memory_layout::Item| i.group.starts_with("vision ");
    assert!(auto.devices[0].items.iter().all(|i| !i.group.starts_with("vision ")));
    let tower: u64 = auto.devices[1].items.iter().filter(vision_item).map(|i| i.bytes).sum();
    assert!(tower > 0);
    options.vision = MediaMode::Rtx(Some(0));
    let local = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    let local_tower: u64 = local.devices[0].items.iter().filter(vision_item).map(|i| i.bytes).sum();
    assert!(local_tower > 0);
    options.vision = MediaMode::Off;
    let off = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert!(off.devices.iter().all(|d| d.items.iter().all(|i| !i.group.starts_with("vision "))));
    assert_eq!(auto.devices[0].used_bytes(), off.devices[0].used_bytes());
    assert_eq!(auto.devices[1].used_bytes() - off.devices[1].used_bytes(), tower);
    assert_eq!(local.devices[0].used_bytes() - off.devices[0].used_bytes(), local_tower);
    assert_eq!(auto.pool_tokens, off.pool_tokens);
    assert_eq!(local.pool_tokens, off.pool_tokens);
    use cuteafd_core::memory_layout::Category;
    assert!(!auto.devices[0].by_category().contains_key(&Category::Embedding));
    options.layout.as_mut().unwrap().force_gpu_embedding = true;
    let gpu = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert!(gpu.devices[0].by_category()[&Category::Embedding] > 0);
    assert_eq!(gpu.devices[0].used_bytes() - off.devices[0].used_bytes(), 128 * 5120 * 2);
    options.layout.as_mut().unwrap().force_gpu_embedding = false;
    options.layout.as_mut().unwrap().rtx_bytes = vec![96 << 30];
    let pro = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert!(pro.devices[0].by_category()[&Category::Embedding] > 0);
}

#[test]
fn layout_charges_admitted_probe_outputs_before_sizing_kv() {
    let dir = qwen_snapshot(4);
    let mut options = PlanOptions { layout: Some(layout::LayoutOptions {
        rtx_bytes: vec![96 << 30], prefill_rows: 128, pool_tokens: Some(256),
        ..Default::default()
    }), ..sparks(0) };
    let ordinary = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert!(ordinary.devices[0].items.iter().all(|i| i.group != "probe prefill logits"));
    options.layout.as_mut().unwrap().full_prefill_logits = true;
    let diagnostic = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    let output = diagnostic.devices[0].items.iter().find(|i| i.group == "probe prefill logits").unwrap();
    assert_eq!(output.category, cuteafd_core::memory_layout::Category::Workspace);
    assert!(output.bytes > 0);
    assert_eq!(ordinary.devices[0].free_bytes() - diagnostic.devices[0].free_bytes(), output.bytes as i64);
    assert!(diagnostic.devices.iter().skip(1).all(|d| d.items.iter().all(|i| i.group != "probe prefill logits")));
}

#[test]
fn layout_charges_local_routed_experts_to_the_coordinator() {
    use cuteafd_core::memory_layout::Category;
    let dir = qwen_snapshot(4);
    let report = plan(dir.path(), &PlanOptions {
        layout: Some(layout::LayoutOptions { pool_tokens: Some(256), ..Default::default() }), ..sparks(0)
    }).unwrap();
    let routed = component(&report, Component::RoutedExpert).bytes;
    assert!(routed > 0);
    let memory = report.memory_layout.unwrap();
    let expert_weights: u64 = memory.devices[0].items.iter()
        .filter(|item| item.category == Category::Experts && item.group == "routed expert arenas")
        .map(|item| item.bytes).sum();
    assert!(expert_weights >= routed);
    assert_eq!(expert_weights % (2 * 1024 * 1024), 0);
    // Tier-specific rotations, maps and arena padding add a small overhead;
    // charge them without duplicating the trellis payload itself.
    assert!(expert_weights - routed < routed / 50);
    // Resident EXL3 execution arenas also consume the coordinator budget.
    assert!(memory.devices[0].by_category()[&Category::Experts] > routed);
    assert_eq!(memory.devices.len(), 1);
}

#[test]
fn preferred_qwen_experts_require_room_for_serving_and_keep_explicit_layouts() {
    use cuteafd_core::memory_layout::Category;
    let dir = qwen_snapshot(4);
    let ample = PlanOptions {
        layout: Some(layout::LayoutOptions { pool_tokens: Some(32768), ..Default::default() }),
        ..sparks(4)
    };
    assert_eq!(plan_preferred(dir.path(), &PlanOptions::default()).unwrap().placement, ExpertPlacement::Local);
    let preferred = plan_preferred(dir.path(), &ample).unwrap();
    assert_eq!(preferred.placement, ExpertPlacement::Local, "{}", render(&preferred));
    let memory = preferred.memory_layout.as_ref().unwrap();
    assert!(memory.devices[0].by_category()[&Category::Experts] >
        component(&preferred, Component::RoutedExpert).bytes);
    assert!(memory.devices.iter().all(|d| d.free_bytes() >= 0));
    let explicit_local = plan(dir.path(), &PlanOptions {
        placement: ExpertPlacement::Local, ..ample.clone()
    }).unwrap();
    assert_eq!(memory.devices[0].by_category(),
        explicit_local.memory_layout.as_ref().unwrap().devices[0].by_category());
    let unused_peer = PlanOptions {
        layout: Some(layout::LayoutOptions {
            rtx_bytes: vec![96 << 30, 1 << 30], pool_tokens: Some(32768), ..Default::default()
        }),
        ..ample.clone()
    };
    assert_eq!(plan_preferred(dir.path(), &unused_peer).unwrap().placement, ExpertPlacement::Local);
    // Explicit --spark-ranks keeps the requested topology, even when local fits.
    assert_eq!(plan(dir.path(), &ample).unwrap().placement, ExpertPlacement::Sparks { ranks: 4 });
    let tight = PlanOptions {
        layout: Some(layout::LayoutOptions { rtx_bytes: vec![12 << 30], ..Default::default() }),
        ..ample.clone()
    };
    assert_eq!(plan_preferred(dir.path(), &tight).unwrap().placement, ExpertPlacement::Sparks { ranks: 4 });
    // Weight-only fit is insufficient when a pinned KV pool overruns the GPU.
    let oversized_pool = PlanOptions {
        layout: Some(layout::LayoutOptions { pool_tokens: Some(1 << 40), ..Default::default() }),
        ..ample
    };
    assert_eq!(plan_preferred(dir.path(), &oversized_pool).unwrap().placement, ExpertPlacement::Sparks { ranks: 4 });
    let unsupported = qwen_snapshot(3);
    assert_eq!(plan_preferred(unsupported.path(), &sparks(4)).unwrap().placement, ExpertPlacement::Sparks { ranks: 4 });
}

#[test]
fn qwen_spark_layout_keeps_only_native_mtp_experts_on_coordinator() {
    use cuteafd_core::memory_layout::Category;
    let (mut tensors, _) = qwen4_exl3(1, 4);
    let draft: Vec<Tensor> = tensors.iter().map(|(name, dtype, shape)|
        (name.replace("model.language_model.layers.0", "mtp.layers.0"), *dtype, shape.clone())).collect();
    tensors.extend(draft);
    let mut config = qwen4_config(1);
    config["text_config"]["mtp_num_hidden_layers"] = json!(1);
    let dir = snapshot(config, &tensors);
    let mut options = PlanOptions { layout: Some(layout::LayoutOptions {
        rtx_bytes: vec![32 << 30], pool_tokens: Some(32768), native_mtp_layers: 1, ..Default::default()
    }), ..sparks(1) };
    let with_mtp = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    let lead = &with_mtp.devices[0];
    assert!(lead.items.iter().any(|i| i.group == "native MTP expert arena" && i.bytes > 0), "{}", with_mtp.render());
    assert!(!lead.items.iter().any(|i| i.group == "routed expert arenas"));
    assert!(lead.items.iter().any(|i| i.group == "local EXL3 workspace" && i.bytes > 0));
    assert!(with_mtp.notes.iter().any(|n| n.contains("MTP experts stay on rtx0")));
    options.layout.as_mut().unwrap().native_mtp_layers = 0;
    let without_mtp = plan(dir.path(), &options).unwrap().memory_layout.unwrap();
    assert!(!without_mtp.devices[0].items.iter().any(|i| i.group == "native MTP expert arena"));
    assert!(lead.by_category()[&Category::Kv] > without_mtp.devices[0].by_category()[&Category::Kv]);
    assert_eq!(with_mtp.devices[1].by_category()[&Category::Experts],
        without_mtp.devices[1].by_category()[&Category::Experts]);
}

#[test]
fn local_qwen_memory_layout_charges_experts_to_the_lead_gpu() {
    use cuteafd_core::memory_layout::Category;
    let dir = qwen_snapshot(4);
    let report = plan(dir.path(), &PlanOptions {
        layout: Some(layout::LayoutOptions {
            rtx_bytes: vec![96 << 30, 96 << 30], pool_tokens: Some(32768), ..Default::default()
        }),
        ..sparks(0)
    }).unwrap();
    let layout = report.memory_layout.as_ref().unwrap();
    assert!(layout.devices[0].by_category()[&Category::Experts] >
        component(&report, Component::RoutedExpert).bytes);
    assert!(!layout.devices[1].by_category().contains_key(&Category::Experts));
    assert!(layout.devices[1].items.is_empty());
    assert_eq!(layout.devices[1].kv_tokens, 0);
    assert!(!layout.notes.iter().any(|n| n.contains("only one coordinator")));
    let one = plan(dir.path(), &PlanOptions {
        layout: Some(layout::LayoutOptions { pool_tokens: Some(32768), ..Default::default() }),
        ..sparks(0)
    }).unwrap();
    assert_eq!(layout.devices[0].by_category(), one.memory_layout.as_ref().unwrap().devices[0].by_category());
}

#[test]
fn media_off_is_disabled_and_saves_checkpoint_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let config = mimo_flash_config();
    let mut tensors = mimo_flash_tensors();
    tensors.push(("visual.patch_embed.proj.weight".into(), "BF16".into(), vec![1280, 3, 2, 16, 16]));
    write_snapshot(dir.path(), &config, &tensors, None);
    let report = plan(dir.path(), &PlanOptions { vision: MediaMode::Off, ..Default::default() }).unwrap();
    let vision = report.components.iter().find(|c| c.component == Component::Vision).unwrap();
    assert_eq!(vision.status, Status::Disabled);
    assert_eq!(vision.bytes, 0);
    assert!(report.disabled_media_bytes > 0);
}

#[test]
fn v41_encoder_preserves_experts_and_falls_back_at_runtime_budget() {
    use super::encoder::EncoderKind;
    use cuteafd_core::memory_layout::{Category, DeviceKind};
    let dir = snapshot(v41_config(), &[
        t("embed.weight", "BF16", &[128, 5120]),
        t("layers.0.ffn.experts.0.w1.weight", "I8", &[2304, 2560]),
        t("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", &[2304, 160]),
        t("vision.embeddings.patch_embedding.weight", "BF16", &[1152, 3, 14, 14]),
    ]);
    let options = PlanOptions { layout: Some(layout::LayoutOptions {
        rtx_bytes: vec![96 << 30], local_expert_layers: Some(0), target_pool_tokens: 32768,
        spark_allocation_budget_bytes: Some(100 << 30), ..Default::default()
    }), ..Default::default() };
    let auto = plan(dir.path(), &options).unwrap();
    assert_eq!(auto.encoder.as_ref().unwrap().kind, EncoderKind::Spark { rank: 0 });
    assert_eq!(auto.encoder.as_ref().unwrap().scratch,
        617_439_296 + super::encoder::V41_SPARK_CUDA_OVERHEAD_BYTES);
    let local = plan(dir.path(), &PlanOptions { vision: MediaMode::Rtx(Some(0)), ..options.clone() }).unwrap();
    let off = plan(dir.path(), &PlanOptions { vision: MediaMode::Off, ..options.clone() }).unwrap();
    assert_eq!(local.encoder.as_ref().unwrap().scratch, 617_439_296);
    let experts = |report: &PlanReport| report.memory_layout.as_ref().unwrap().devices.iter()
        .filter(|d| d.kind == DeviceKind::Spark).map(|d| d.items.iter()
        .filter(|i| i.category == Category::Experts).map(|i| i.bytes).sum::<u64>()).collect::<Vec<_>>();
    assert_eq!(experts(&auto), experts(&off));
    assert_eq!(experts(&local), experts(&off));
    let mut limited = options;
    limited.layout.as_mut().unwrap().spark_allocation_budget_bytes = Some(1);
    let fallback = plan(dir.path(), &limited).unwrap();
    assert_eq!(fallback.encoder.as_ref().unwrap().kind, EncoderKind::Rtx { gpu: 0 });
    assert_eq!(experts(&fallback), experts(&off));
    assert_ne!(auto.encoder_plan_hash, local.encoder_plan_hash);
    assert_ne!(auto.encoder_plan_hash, off.encoder_plan_hash);
}

#[test]
fn encoder_plan_g9_charges_before_pool_and_hashes_off() {
    use super::encoder::EncoderKind;
    let mut cfg = mimo_flash_config();
    cfg["vision_config"] = json!({"depth":28});
    let mut tensors = mimo_flash_tensors();
    tensors.push(t("visual.patch_embed.proj.weight", "BF16", &[1280,3,2,16,16]));
    let dir = snapshot_tp(cfg, &tensors, Some(1));
    let options = PlanOptions { layout: Some(layout::LayoutOptions { rtx_bytes: vec![96<<30], target_pool_tokens: 32_768, ..Default::default() }), ..Default::default() };
    let spark = plan(dir.path(), &options).unwrap();
    assert!(matches!(spark.encoder.as_ref().unwrap().kind, EncoderKind::Spark { .. }));
    let local = plan(dir.path(), &PlanOptions { vision: MediaMode::Rtx(Some(0)), ..options.clone() }).unwrap();
    assert_eq!(local.encoder.as_ref().unwrap().kind, EncoderKind::Rtx { gpu:0 });
    let memory = local.memory_layout.as_ref().unwrap();
    let tower = memory.devices[0].items.iter().position(|i| i.group == "vision tower").unwrap();
    let pool = memory.devices[0].items.iter().position(|i| i.group == "records").unwrap();
    assert!(tower < pool, "{:?}", memory.devices[0].items);
    let off = plan(dir.path(), &PlanOptions { vision: MediaMode::Off, ..options }).unwrap();
    assert_eq!(off.encoder.as_ref().unwrap().admitted_bytes(),0);
    assert_eq!(off.components.iter().find(|c| c.component == Component::Vision).unwrap().bytes,0);
    assert_ne!(spark.encoder_plan_hash,off.encoder_plan_hash);
    assert_ne!(spark.encoder_plan_hash,local.encoder_plan_hash);
    assert!(off.memory_layout.unwrap().devices.iter().flat_map(|d| &d.items).all(|i| !i.group.starts_with("vision")));
}

#[test]
fn glm_flash_vision_plan_matches_resident_admission_and_mimo_shape() {
    use super::encoder::EncoderKind;
    let mut cfg = glm5_flash_config(2);
    cfg["vision_config"] = json!({"depth":24,"hidden_size":1024,"intermediate_size":4096,
        "num_heads":16,"out_hidden_size":4096,"patch_size":14,"temporal_patch_size":2,
        "spatial_merge_size":2,"projection_intermediate_size":10240,"in_channels":3,
        "attention_bias":true,"hidden_act":"silu","rms_norm_eps":1e-5,"swiglu_limit":10.0});
    let tensors: Vec<_> = super::families::glm::glm_flash_vision_tensors().into_iter()
        .map(|(name, shape)| t(format!("model.visual.{name}"), "BF16", &shape)).collect();
    let dir = snapshot_tp(cfg.clone(), &tensors, Some(4));
    for (gpus, gib, ranks) in [(1, 96, 4), (2, 96, 4), (1, 32, 4), (1, 96, 0)] {
        let options = PlanOptions { layout: Some(layout::LayoutOptions {
            rtx_bytes: vec![gib << 30; gpus], pool_tokens: Some(32768), ..Default::default()
        }), ..sparks(ranks) };
        let report = plan(dir.path(), &options).unwrap();
        let encoder = report.encoder.as_ref().unwrap();
        assert_eq!(encoder.weights, 1_128_026_176);
        assert_eq!(encoder.admitted_bytes(), 1_919_933_760);
        assert_eq!(encoder.kind, if ranks > 0 { EncoderKind::Spark { rank: 0 } } else { EncoderKind::Rtx { gpu: 0 } });
        let vision = report.components.iter().find(|c| c.component == Component::Vision).unwrap();
        assert_eq!(vision.status, Status::Ready);
        assert_eq!(vision.rejected, 0);
        assert_eq!(vision.bytes, 1_127_254_016);
        assert_eq!(vision.formats.keys().collect::<Vec<_>>(), vec!["bf16"]);
        let off = plan(dir.path(), &PlanOptions { vision: MediaMode::Off, ..options }).unwrap();
        assert_eq!(off.encoder.unwrap().admitted_bytes(), 0);
        assert_eq!(off.components.iter().find(|c| c.component == Component::Vision).unwrap().bytes, 0);
    }
    let mut bad = cfg.clone(); bad["vision_config"]["num_heads"] = json!(8);
    let invalid = snapshot_tp(bad, &tensors, Some(4));
    let report = plan(invalid.path(), &sparks(4)).unwrap();
    assert_ne!(report.components.iter().find(|c| c.component == Component::Vision).unwrap().status, Status::Ready);
    assert_eq!(report.encoder.as_ref().unwrap().admitted_bytes(), 0);
    assert!(!report.placement_supported);
    let mut missing = tensors.clone(); missing.pop();
    let incomplete = snapshot_tp(cfg.clone(), &missing, Some(4));
    let report = plan(incomplete.path(), &sparks(4)).unwrap();
    assert_ne!(report.components.iter().find(|c| c.component == Component::Vision).unwrap().status, Status::Ready);
    let mut wrong_dtype = tensors.clone(); wrong_dtype[0].1 = "F16";
    let mut extra_tensor = tensors.clone();
    extra_tensor.push(t("model.visual.unrecognized.weight", "BF16", &[1024]));
    for unsupported in [wrong_dtype, extra_tensor] {
        let invalid = snapshot_tp(cfg.clone(), &unsupported, Some(4));
        let report = plan(invalid.path(), &sparks(4)).unwrap();
        assert_ne!(component(&report, Component::Vision).status, Status::Ready);
        assert_eq!(report.encoder.as_ref().unwrap().admitted_bytes(), 0);
    }
    let mut mimo_config = mimo_flash_mopd_config();
    mimo_config["vision_config"] = json!({"depth":28,"hidden_size":1280,"intermediate_size":4608,
        "num_heads":32,"num_key_value_heads":8,"out_hidden_size":4096,"patch_size":16,
        "temporal_patch_size":2,"spatial_merge_size":2,"hidden_act":"silu"});
    let mut mimo_tensors = mimo_flash_mopd_tensors();
    mimo_tensors.push(t("visual.patch_embed.proj.weight", "BF16", &[1280,3,2,16,16]));
    let mimo = snapshot(mimo_config, &mimo_tensors);
    let report = plan(mimo.path(), &sparks(4)).unwrap();
    let glm = plan(dir.path(), &sparks(4)).unwrap();
    let component = |report: &super::PlanReport| serde_json::to_value(report.components.iter()
        .find(|c| c.component == Component::Vision).unwrap()).unwrap().as_object().unwrap().keys().cloned().collect::<Vec<_>>();
    // Both families expose the same component/encoder schema, not a GLM-only side channel.
    assert_eq!(component(&glm), component(&report));
    let mimo_vision = report.components.iter().find(|c| c.component == Component::Vision).unwrap();
    assert_eq!(mimo_vision.status, Status::Ready);
    assert_eq!(mimo_vision.formats.keys().collect::<Vec<_>>(), vec!["bf16"]);
    let encoder_keys = |report: &super::PlanReport| serde_json::to_value(report.encoder.as_ref().unwrap())
        .unwrap().as_object().unwrap().keys().cloned().collect::<Vec<_>>();
    assert_eq!(encoder_keys(&glm), encoder_keys(&report));
    assert_eq!(glm.encoder.as_ref().unwrap().kind, report.encoder.as_ref().unwrap().kind);
    let rendered = super::render(&glm);
    assert!(rendered.contains("Spark { rank: 0 }") && rendered.contains("1128026176"));
    assert!(!rendered.contains("MiMo key-0"));
}

#[test]
fn explicit_encoder_failure_and_small_pool_admission() {
    use super::encoder::EncoderKind;
    let mut cfg = mimo_flash_config();
    cfg["vision_config"] = json!({"depth":28});
    let mut tensors = mimo_flash_tensors();
    tensors.push(t("visual.patch_embed.proj.weight", "BF16", &[1280,3,2,16,16]));
    let dir = snapshot_tp(cfg, &tensors, Some(1));
    let options = PlanOptions { vision: MediaMode::Rtx(Some(0)), layout: Some(layout::LayoutOptions {
        rtx_bytes: vec![32<<30], pool_tokens: Some(32768), ..Default::default()
    }), ..Default::default() };
    let local = plan(dir.path(), &options).unwrap();
    assert_eq!(local.encoder.as_ref().unwrap().kind, EncoderKind::Rtx { gpu:0 });
    for vision in [MediaMode::Rtx(Some(5)), MediaMode::Spark(Some(5))] {
        let invalid = plan(dir.path(), &PlanOptions { vision, ..options.clone() }).unwrap();
        assert!(!invalid.placement_supported);
        assert!(!invalid.executable());
        assert_ne!(invalid.components.iter().find(|c| c.component == Component::Vision).unwrap().status, Status::Disabled);
    }
}

#[test]
fn qwen_image_cap_is_visible_and_auto_preserves_zero_spark_kv() {
    use super::encoder::EncoderKind;
    let mut cfg = qwen4_config(1);
    cfg["vision_config"] = json!({"depth":27,"hidden_size":1152,"intermediate_size":4304,
        "num_heads":16,"num_position_embeddings":2304,"out_hidden_size":2560,
        "patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
        "hidden_act":"gelu_pytorch_tanh","deepstack_visual_indexes":[]});
    let tensors = [t("model.visual.patch_embed.proj.weight", "BF16", &[1152,3,2,16,16]),
        t("model.visual.patch_embed.proj.bias", "BF16", &[1152]),
        t("model.visual.blocks.0.norm1.weight", "BF16", &[1152]),
        t("model.visual.blocks.0.norm1.bias", "BF16", &[1152])];
    let dir = snapshot(cfg.clone(), &tensors);
    let options = PlanOptions { placement: ExpertPlacement::Local, layout: Some(layout::LayoutOptions {
        rtx_bytes: vec![96<<30], pool_tokens: Some(32768), target_pool_tokens: 32768,
        ..Default::default()
    }), ..Default::default() };
    let auto = plan(dir.path(), &options).unwrap();
    assert_eq!(auto.max_image_tokens, Some(1024));
    assert_eq!(serde_json::to_value(&auto).unwrap()["max_image_tokens"], 1024);
    assert!(render(&auto).contains("image cap  1024 merged tokens per image (detail=low 256)"));
    assert_eq!(auto.spark_ranks, 0);
    assert_eq!(auto.encoder.as_ref().unwrap().kind, EncoderKind::Rtx { gpu: 0 });
    assert_eq!(auto.encoder.as_ref().unwrap().weights, 898_680_904);
    assert_eq!(auto.encoder.as_ref().unwrap().scratch, 447_778_048);
    assert_eq!(auto.encoder.as_ref().unwrap().admitted_bytes(), 1_346_458_952);
    assert_eq!(auto.memory_layout.as_ref().unwrap().pool_tokens, 32768);
    let vision = auto.components.iter().find(|c| c.component == Component::Vision).unwrap();
    assert_eq!(vision.status, Status::Ready);
    let off = plan(dir.path(), &PlanOptions { vision: MediaMode::Off, ..options.clone() }).unwrap();
    assert_eq!(off.max_image_tokens, Some(1024));
    assert_eq!(off.encoder.as_ref().unwrap().kind, EncoderKind::Off);
    assert_eq!(off.encoder.as_ref().unwrap().admitted_bytes(), 0);
    assert_ne!(auto.encoder_plan_hash, off.encoder_plan_hash);
    cfg["vision_config"]["intermediate_size"] = json!(4305);
    let invalid = snapshot(cfg, &tensors);
    let invalid = plan(invalid.path(), &options).unwrap();
    assert!(invalid.components.iter().find(|c| c.component == Component::Vision).unwrap()
        .rejections.iter().any(|r| r.reason.contains("intermediate_size")));
}

/// GLM 5.3 Flash's layout reserves the mark arena its server allocates at the default knobs
/// (`MarkArena::slots_for`: 2C + 2 for its 147.6 MB FP32 marks, where it reserved a flat 18),
/// and its state carries the speculative replay records.
#[test]
fn glm5_flash_layout_reserves_the_mark_arena_its_server_allocates() {
    use cuteafd_core::memory_layout::Category;
    let marks = |memory: &cuteafd_core::memory_layout::MemoryLayout| memory.devices[0].by_category()
        .get(&Category::Prefix).copied().unwrap_or(0);
    let layout = |rtx: u64, concurrency: u64| PlanOptions { layout: Some(layout::LayoutOptions {
        rtx_bytes: vec![rtx], concurrency, pool_tokens: Some(0), ..Default::default() }), ..sparks(4) };
    // GLM 5.3 Flash: 34 KDA layers, 147.6 MB marks: the 2C + 2 floor (18 at C8, 34 at C16).
    let mut config = glm5_flash_config(45);
    config["text_config"]["layer_types"] = json!((0..45)
        .map(|l| if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect::<Vec<_>>());
    let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(&config).unwrap();
    let rank = crate::serving_capacity::glm_flash_cache_geometry(&cfg, 45).unwrap().ranks[0];
    assert_eq!((rank.retained_mark_bytes, rank.speculative_replay_bytes), (147_619_840, 321_421_312));
    let dir = snapshot(config, &[]);
    for (concurrency, slots) in [(1, 14), (8, 18), (16, 34)] {
        let memory = plan(dir.path(), &layout(96 << 30, concurrency)).unwrap().memory_layout.unwrap();
        assert_eq!(cuteafd_core::prefix::mark_slots_for(concurrency, 20, rank.retained_mark_bytes, 2 << 30), slots);
        assert_eq!(marks(&memory), slots * rank.retained_mark_bytes);
        let state = memory.devices[0].items.iter().find(|i| i.group == "state").unwrap().bytes;
        assert_eq!(state, rank.fixed_state_bytes + rank.active_state_per_sequence_bytes * concurrency.max(8)
            + rank.speculative_replay_bytes);
    }
    // An explicit arena (0: none) is taken as given.
    let mut none = layout(96 << 30, 16);
    none.layout.as_mut().unwrap().prefix_slots = Some(0);
    assert_eq!(marks(&plan(dir.path(), &none).unwrap().memory_layout.unwrap()), 0);
    // `--prefix-cache-entries` and `--prefix-cache-mark-mib` size it as they size the server's:
    // 1,971 MiB holds 14 marks at 5 sequences and 6 entries, one or two GPUs (each its half of
    // every mark under a head split, which keeps the token keys planned here); no entries, none.
    let knobs = |rtx: usize, entries: u64| PlanOptions { layout: Some(layout::LayoutOptions {
        rtx_bytes: vec![96 << 30; rtx], concurrency: 5, pool_tokens: Some(0), mimo_prefix_entries: entries,
        mimo_prefix_mark_bytes: 1971 << 20, ..Default::default() }), ..sparks(4) };
    assert_eq!(cuteafd_core::prefix::mark_slots_for(5, 6, rank.retained_mark_bytes, 1971 << 20), 14);
    let one = plan(dir.path(), &knobs(1, 6)).unwrap().memory_layout.unwrap();
    assert_eq!(marks(&one), 14 * rank.retained_mark_bytes);
    let split = plan(dir.path(), &knobs(2, 6)).unwrap().memory_layout.unwrap();
    let half = crate::serving_capacity::glm_flash_rank_cache_geometry(&cfg, 45, 2,
        crate::serving_capacity::GlmfIndexCache::Keys, 4).unwrap().ranks[0].retained_mark_bytes;
    assert_eq!((marks(&split), 2 * half), (14 * half, rank.retained_mark_bytes));
    assert_eq!(marks(&plan(dir.path(), &knobs(1, 0)).unwrap().memory_layout.unwrap()), 0);
}

/// `--prefix-marks pool` for GLM 5.3 Flash: no mark arena, and the units pool marks reserve
/// (`GLMF_POOL_MARK_RESERVED_UNITS`, never handed out) charged beside the pool, so the pool the
/// layout admits is the pool requests can use.
#[test]
fn glm_flash_pool_marks_charge_their_reserved_unit_beside_the_pool() {
    use crate::serving_capacity::GLMF_POOL_MARK_RESERVED_UNITS;
    let mut config = glm5_flash_config(45);
    config["text_config"]["layer_types"] = json!((0..45)
        .map(|l| if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect::<Vec<_>>());
    let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(&config).unwrap();
    let rank = crate::serving_capacity::glm_flash_cache_geometry(&cfg, 45).unwrap().ranks[0];
    let unit = rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes;
    let dir = snapshot(config, &[]);
    let layout = |pool_marks: bool, prefix_slots: Option<u64>| plan(dir.path(), &PlanOptions {
        layout: Some(layout::LayoutOptions { rtx_bytes: vec![48 << 30], concurrency: 16, pool_tokens: Some(0),
            glmf_pool_marks: pool_marks, prefix_slots, ..Default::default() }), ..sparks(4) })
        .unwrap().memory_layout.unwrap();
    let item = |memory: &cuteafd_core::memory_layout::MemoryLayout, group: &str| memory.devices[0].items.iter()
        .filter(|i| i.group == group).map(|i| i.bytes).sum::<u64>();
    let (pool, none) = (layout(true, None), layout(false, Some(0)));
    assert_eq!((item(&pool, "marks"), item(&pool, "reserved units")), (0, GLMF_POOL_MARK_RESERVED_UNITS * unit));
    // No entries, no marks: pool marks keep no unit back, as serve-glmf then keeps none.
    let off = plan(dir.path(), &PlanOptions { layout: Some(layout::LayoutOptions { rtx_bytes: vec![48 << 30],
        concurrency: 16, pool_tokens: Some(0), glmf_pool_marks: true, mimo_prefix_entries: 0, ..Default::default() }),
        ..sparks(4) }).unwrap().memory_layout.unwrap();
    assert_eq!((item(&off, "marks"), item(&off, "reserved units")), (0, 0));
    assert_eq!((item(&none, "marks"), item(&none, "reserved units")), (0, 0));
    // Pool marks ignore an arena request; the reserved unit's bytes come out of the pool.
    assert_eq!(item(&layout(true, Some(34)), "marks"), 0);
    assert!(none.pool_tokens - pool.pool_tokens <= 256 && pool.pool_tokens > 0, "{} {}", none.pool_tokens,
        pool.pool_tokens);
    let fewer = (none.pool_tokens - pool.pool_tokens) as i64 * unit as i64 / 256;
    assert_eq!(none.devices[0].free_bytes() - pool.devices[0].free_bytes(),
        (GLMF_POOL_MARK_RESERVED_UNITS * unit) as i64 - fewer);
    // Against the arena the server would allocate at 16 sequences (34 marks of 147.6 MB), pool marks
    // free the arena's bytes less the reserved unit, for the pool (up to its target) or as free memory.
    let arena = layout(false, None);
    assert_eq!(item(&arena, "marks"), 34 * rank.retained_mark_bytes);
    assert!(pool.pool_tokens >= arena.pool_tokens, "{} {}", pool.pool_tokens, arena.pool_tokens);
    let more = (pool.pool_tokens - arena.pool_tokens) as i64 * unit as i64 / 256;
    assert_eq!(pool.devices[0].free_bytes() - arena.devices[0].free_bytes(),
        (34 * rank.retained_mark_bytes - GLMF_POOL_MARK_RESERVED_UNITS * unit) as i64 - more);
}


#[test]
fn attention_placement_heads_preserves_default_layout_and_modes_refuse_at_plan_time() {
    use crate::placement::AttentionPlacement;
    let snapshots = [v4_snapshot(), qwen_snapshot(4), snapshot(v41_config(), &[]),
        snapshot(glm5_config(), &[]), snapshot(glm5_flash_config(45), &[]),
        snapshot(mimo_flash_config(), &mimo_flash_tensors()), snapshot(mimo_pro_config(), &mimo_pro_tensors())];
    for dir in snapshots {
        for gpus in [1, 2] {
            let options = PlanOptions { layout: Some(layout::LayoutOptions {
                rtx_bytes: vec![crate::placement::inventory::PRO_TOTAL_BYTES; gpus], ..Default::default()
            }), ..sparks(4) };
            let auto = plan(dir.path(), &options).unwrap();
            let heads = plan(dir.path(), &PlanOptions { attention_placement: Some(AttentionPlacement::Heads), ..options.clone() }).unwrap();
            assert_eq!(auto.memory_layout, heads.memory_layout);
            assert_eq!(auto.config_error, heads.config_error);
            for mode in [AttentionPlacement::Context, AttentionPlacement::Layers] {
                let unsupported = plan(dir.path(), &PlanOptions { attention_placement: Some(mode), ..options.clone() }).unwrap();
                let executor = crate::placement::families::executor(auto.family.as_deref().unwrap()).unwrap();
                if executor.check_attention(Some(mode), gpus, gpus == 2).is_ok() {
                    assert!(unsupported.config_error.is_none());
                } else {
                    assert!(!unsupported.executable());
                    let reason = unsupported.config_error.unwrap();
                    assert!(reason.contains(auto.family.as_deref().unwrap()) && reason.contains(&mode.to_string()), "{reason}");
                    assert!(unsupported.memory_layout.is_none());
                }
            }
        }
    }
}
