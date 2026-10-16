use super::*;
use crate::plan::{self, checkpoint::Checkpoint, testing::{self, Tensor as FixtureTensor}, format, *};
use serde_json::Value;

fn fixture(kind: &str) -> (Value, Vec<FixtureTensor>) {
    let source: Value = serde_json::from_str(include_str!("fixtures/official-fp8.json")).unwrap();
    let mut config = source["config"].clone();
    if kind == "nvfp4" {
        config = serde_json::from_str(include_str!("fixtures/official-nvfp4-config.json")).unwrap();
    } else if kind == "bf16" {
        config.as_object_mut().unwrap().remove("quantization_config");
    }
    let mut tensors = Vec::new();
    for (name, value) in source["tensors"].as_object().unwrap() {
        let dtype = value[0].as_str().unwrap();
        let shape: Vec<usize> = value[1].as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as usize).collect();
        let dtype = match dtype { "BF16" => "BF16", "F32" => "F32", "F8_E4M3" => "F8_E4M3", _ => panic!("{dtype}") };
        if kind == "bf16" && name.ends_with("weight_scale_inv") { continue; }
        if kind == "nvfp4" && name.ends_with("weight_scale_inv") { continue; }
        if dtype == "F8_E4M3" && kind == "nvfp4" {
            let stem = name.strip_suffix(".weight").unwrap();
            let [e,n,k]: [usize;3] = shape.try_into().unwrap();
            tensors.push(testing::t(name, "U8", &[e,n,k/2]));
            tensors.push(testing::t(format!("{stem}.weight_scale"), "F8_E4M3", &[e,n,k/16]));
            tensors.push(testing::t(format!("{stem}.weight_scale_2"), "F32", &[e]));
            tensors.push(testing::t(format!("{stem}.input_scale"), "F32", &[e]));
        } else { tensors.push((name.clone(), if kind == "bf16" { "BF16" } else { dtype }, shape)); }
    }
    // Only originally quantized expert matrices widen in the BF16 root;
    // all release FP32 norms/router/index weights stay FP32.
    if kind == "bf16" {
        for (name,dtype,_) in &mut tensors {
            if source["tensors"][name.as_str()][0] == "F32" { *dtype = "F32"; }
        }
    }
    (config,tensors)
}

fn snapshot(kind: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let (config,tensors) = fixture(kind);
    testing::write_snapshot(dir.path(), &config, &tensors, None);
    dir
}

fn report(dir: &std::path::Path, gpus: usize, gib: u64, ranks: usize, pool: Option<u64>) -> PlanReport {
    plan::plan(dir, &PlanOptions { placement: ExpertPlacement::from_spark_ranks(ranks),
        layout: Some(layout::LayoutOptions { rtx_bytes: vec![gib<<30;gpus], pool_tokens: pool, ..Default::default() }),
        ..Default::default() }).unwrap()
}

#[test]
fn official_nested_inventory_separates_target_tail_mtp_vision() {
    let dir = snapshot("fp8");
    // Poisoned sidecar must have no effect on config/header-only planning.
    std::fs::write(dir.path().join("fp8_vit_sidecar_ready.json"), "not JSON").unwrap();
    let cp = Checkpoint::open(dir.path()).unwrap();
    let model = Model::open(&cp).unwrap();
    assert_eq!(model.spec.layers.len(),92);
    assert_eq!(model.spec.moe_layers(),88);
    assert_eq!(model.cfg.full.iter().filter(|v| **v).count(),23);
    assert_eq!(model.cfg.activation.len(),95);
    assert_eq!(model.activation(94,false),Some(FfnActivation::SiluPostClampFp32Product { limit:7.0 }));
    assert_eq!(model.spec.layers[91].ffn,FfnKind::Dense { intermediate:13824 });
    assert_eq!(model.spec.layers[3].rope,Some(RopeSpec { dims:64,theta:10_000_000.0 }));
    assert_eq!(model.spec.layers[0].rope,Some(RopeSpec { dims:192,theta:10_000.0 }));
    assert_eq!(model.bytes.values().sum::<u64>(),641_367_461_632);
    assert_eq!(model.bytes[&Component::RoutedExpert],604_721_627_136);
    assert_eq!(model.bytes[&Component::Speculator],5_031_842_304);
    assert_eq!(model.bytes[&Component::Vision],3_962_029_568);
    for layer in 88..91 {
        assert_eq!(model.tensors[&format!("model.layers.{layer}.moe.gate_proj.weight")].dtype,DType::Bf16);
    }
    let result = report(dir.path(),1,96,6,Some(262_144));
    assert!(result.config_error.is_none(),"{:?}",result.config_error);
    assert!(result.unclassified.is_empty(),"{:?}",result.unclassified);
    assert!(!result.executable());
    assert!(result.components.iter().all(|c| c.status != Status::Ready));
    assert!(result.hints.iter().any(|hint|hint.how.contains("CSA-8")));
    assert!(result.hints.iter().any(|hint|hint.how.contains("BF16-tail")));
    assert_eq!(result.min_spark_ranks,Some(6));
}

#[test]
fn card_memory_conclusions_and_request_vs_pool_extent() {
    let dir = snapshot("fp8");
    for gpus in [1,2] {
        let r = report(dir.path(),gpus,96,6,Some(262_144));
        assert!(r.config_error.is_none(),"{:?}",r.config_error);
        assert!(!r.executable());
        let memory = r.memory_layout.unwrap();
        assert_eq!(memory.pool_tokens,262_144,"{:?}",r.hints);
        assert!(memory.devices.iter().all(|d| d.used_bytes() <= d.capacity_bytes));
    }
    let r = report(dir.path(),1,96,6,Some(1<<20));
    assert!(r.hints.iter().any(|h| h.what.contains("coordinator BF16 pool lower bound refused")),"{:?}",r.hints);
    let r = report(dir.path(),2,96,6,Some(1<<20));
    assert_eq!(r.memory_layout.unwrap().pool_tokens,1<<20);
    let r = report(dir.path(),1,32,6,None);
    assert!(r.hints.iter().any(|h| h.what.contains("5090 refused")));
    let r = report(dir.path(),1,96,4,None);
    assert!(r.hints.iter().any(|h| h.what.contains("TP4 refused")));
    let r = report(dir.path(),2,96,0,None);
    assert!(r.hints.iter().any(|h| h.what.contains("no 0-Spark")));
    let r = report(dir.path(),1,96,6,Some(32_768));
    assert!(r.hints.iter().any(|h| h.what.contains("below 256K")));
    let r = report(dir.path(),1,96,6,None);
    let memory = r.memory_layout.unwrap();
    assert!(memory.pool_tokens >= 262_144 && memory.pool_tokens < 1<<20);
    assert!(memory.notes.iter().any(|n| n.contains("71392 bytes/token")));
    assert!(memory.notes.iter().any(|n| n.contains("36064 bytes/token")));
    assert!(memory.notes.iter().any(|n| n.contains("Maximum request 1048576")));
}

#[test]
fn nvfp4_stacked_scales_and_bf16_root_are_named_not_executed() {
    let dir = snapshot("nvfp4");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let model = Model::open(&cp).unwrap();
    assert_eq!(model.bytes.values().sum::<u64>(),394_164_831_232);
    for (stem,members) in format::group_by_stem(&cp.tensors) {
        let operand = format::detect(&members).unwrap_or_else(|e|panic!("{stem}: {e}"));
        if stem == "model.layers.3.moe.gate_proj" {
            assert!(operand.is_nvfp4());
            assert_eq!(operand.logical,vec![352,1536,4096]);
            assert_eq!(operand.scale.unwrap().grid,vec![352,1536,256]);
        }
    }
    let result = report(dir.path(),1,96,4,Some(262_144));
    assert!(result.config_error.is_none(),"{:?}",result.config_error);
    assert_eq!(result.min_spark_ranks,Some(4));
    assert_eq!(result.memory_layout.unwrap().pool_tokens,262_144);
    let dir = snapshot("bf16");
    let r = report(dir.path(),2,96,6,None);
    assert!(r.config_error.is_none(),"{:?}",r.config_error);
    assert!(r.hints.iter().any(|h| h.what.contains("TP6 refused")));
}

#[test]
fn malformed_headers_missing_tensors_and_arrays_fail_with_names() {
    let (config,tensors) = fixture("fp8");
    for key in ["layer_types","rope_theta","partial_rotary_factors","swiglu_limits","swiglu_limits_shared"] {
        let mut config = config.clone();
        config["text_config"][key].as_array_mut().unwrap().pop();
        let dir = tempfile::tempdir().unwrap();
        testing::write_snapshot(dir.path(),&config,&tensors,None);
        let r = report(dir.path(),1,96,6,None);
        assert!(r.config_error.unwrap().contains(key));
    }
    for (key, value) in [("moe_top_k", serde_json::json!(16)),
        ("head_dim_full_attention", serde_json::json!(128)),
        ("moe_router_scaling_factor", serde_json::json!(1.0))] {
        let mut config = config.clone();
        config["text_config"][key] = value;
        let dir = tempfile::tempdir().unwrap();
        testing::write_snapshot(dir.path(), &config, &tensors, None);
        assert!(report(dir.path(), 1, 96, 6, None).config_error.unwrap().contains(key));
    }
    for (name,change) in [("model.layers.3.moe.gate_proj.weight",0),
        ("model.layers.3.self_attn.ssmax_s",1),("model.layers.92.transformer.shared_head.output.weight",2),
        ("vision_model.vit_downsampler2.weight",2),("model.layers.3.moe.gate_proj.weight_scale_inv",1)] {
        let mut tensors = tensors.clone();
        if change == 2 { tensors.retain(|t| t.0 != name); }
        else { let t = tensors.iter_mut().find(|t|t.0 == name).unwrap();
            if change == 0 { t.1 = "BF16"; } else { t.2[0] += 1; }
        }
        let dir = tempfile::tempdir().unwrap();
        testing::write_snapshot(dir.path(),&config,&tensors,None);
        let r = report(dir.path(),1,96,6,None);
        assert!(r.config_error.unwrap().contains(name),"missing diagnostic for {name}");
    }
}

#[test]
fn shared_format_detector_checks_each_expert_scale_axis() {
    let dir = snapshot("fp8");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let groups = format::group_by_stem(&cp.tensors);
    let fp8 = format::detect(&groups["model.layers.3.moe.gate_proj"]).unwrap();
    assert!(fp8.is_fp8_block(128,&[format::ScaleEncoding::F32]));
    assert_eq!(fp8.logical,vec![352,1536,4096]);
    // Existing arbitrary-rank per-tensor FP8 remains a single global scale.
    let mut global = cp.tensors.clone();
    global.iter_mut().find(|t| t.meta.name == "model.layers.3.moe.gate_proj.weight_scale_inv")
        .unwrap().meta.shape.clear();
    let groups = format::group_by_stem(&global);
    let scalar = format::detect(&groups["model.layers.3.moe.gate_proj"]).unwrap();
    assert_eq!(scalar.logical, vec![352,1536,4096]);
    assert!(scalar.scale.unwrap().grid.is_empty());
    let mut bad = cp.tensors.clone();
    let scale = bad.iter_mut().find(|t| t.meta.name == "model.layers.3.moe.gate_proj.weight_scale_inv").unwrap();
    scale.meta.shape[0] -= 1;
    let groups = format::group_by_stem(&bad);
    let error = format::detect(&groups["model.layers.3.moe.gate_proj"]).unwrap_err();
    assert!(error.tensor.ends_with("weight_scale_inv"));
    assert!(error.reason.contains("expert axis"));
}

#[test]
fn missing_headers_stale_index_and_direct_text_detection() {
    let dir = snapshot("fp8");
    let index_path = dir.path().join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&std::fs::read(&index_path).unwrap()).unwrap();
    index["metadata"] = serde_json::json!({"total_size":644_100_979_280u64});
    std::fs::write(&index_path,serde_json::to_vec(&index).unwrap()).unwrap();
    let cp = Checkpoint::open(dir.path()).unwrap();
    let model = Model::open(&cp).unwrap();
    assert!(model.spec.notes.iter().any(|n|n.contains("stale index") && n.contains("641367461632")));
    let mut config = cp.config["text_config"].clone();
    config["architectures"] = serde_json::json!(["Step5ForCausalLM"]);
    let mut cp = cp;
    cp.config = config;
    cp.tensors.retain(|t| !t.meta.name.starts_with("vision_model.") && !t.meta.name.starts_with("vit_large_projector."));
    cp.weight_map.retain(|name, _| !name.starts_with("vision_model.") && !name.starts_with("vit_large_projector."));
    assert!(STEP5.detect(&cp));
    assert_eq!(Model::open(&cp).unwrap().cfg.layers,92);
    cp.tensors.retain(|t|t.meta.name != "model.layers.3.moe.gate_proj.weight");
    let input = PlanInput::from_checkpoint(cp);
    let result = input.plan(&PlanOptions { placement:ExpertPlacement::Sparks { ranks:6 },
        layout:Some(Default::default()), ..Default::default() }).unwrap();
    assert!(!result.executable());
    assert!(result.hints.iter().any(|h| h.what.contains("missing safetensors headers")));
}

#[test]
fn unknown_tensor_and_context_ownership_fail_closed() {
    let (config,mut tensors) = fixture("fp8");
    tensors.push(testing::t("vision_model.unexpected.weight","BF16",&[1]));
    let dir = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir.path(),&config,&tensors,None);
    assert!(report(dir.path(),1,96,6,None).config_error.unwrap().contains("vision_model.unexpected.weight"));
    let dir = snapshot("fp8");
    let r = plan::plan(dir.path(), &PlanOptions { placement:ExpertPlacement::Sparks { ranks:6 },
        attention_placement:Some(crate::placement::AttentionPlacement::Context),
        layout:Some(layout::LayoutOptions { rtx_bytes:vec![96<<30;2], ..Default::default() }),
        ..Default::default() }).unwrap();
    assert!(!r.placement_supported);
    assert!(r.hints.iter().any(|h|h.what.contains("context/layer ownership")));
}

#[test]
fn routed_catalog_preserves_native_formats_activation_and_target_bounds() {
    use crate::formats::fp8_experts::{ExpertFormat, Fp8Projection};
    for kind in ["fp8", "nvfp4", "bf16"] {
        let dir = snapshot(kind);
        let mut catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let tensors = catalog.fp8().unwrap();
        assert_eq!(catalog.routed_experts().first_layer, 3);
        assert_eq!(catalog.routed_experts().layers, 91);
        assert_eq!(catalog.routed_experts().draft_stages, 0);
        assert!(!tensors.has_layer(91) && !tensors.has_layer(92));
        assert_eq!(tensors.layer_format(3).unwrap(), match kind {
            "fp8" => ExpertFormat::Fp8Block128, "nvfp4" => ExpertFormat::Nvfp4, _ => ExpertFormat::Bf16 });
        for layer in [88,89,90] {
            let tail = tensors.for_layer(layer).unwrap();
            assert_eq!(tail.format(), ExpertFormat::Bf16);
            assert_eq!(tail.slice_bytes(Fp8Projection::Gate,6).unwrap(), (256*4096*2,0));
            assert_eq!(tail.activation(layer), Some(FfnActivation::SiluPostClampFp32Product { limit: 7.0 }));
        }
        catalog.select_draft(crate::plan::spec::DraftOwnership::None).unwrap();
    }
}

#[test]
fn official_nvfp4_layer30_headers_keep_router_and_stacked_parameters_distinct() {
    use crate::formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors};
    use crate::families::deepseek_v41::v41_catalog::RoutedExpertShape;
    use crate::plan::checkpoint::ModuleNumerics;
    let source: Value = serde_json::from_str(include_str!("fixtures/official-nvfp4-layer30.json")).unwrap();
    let config: Value = serde_json::from_str(include_str!("fixtures/official-nvfp4-config.json")).unwrap();
    let tensors: Vec<_> = source["tensors"].as_object().unwrap().iter().map(|(name, meta)| {
        let dtype = match meta["dtype"].as_str().unwrap() {
            "U8" => "U8", "BF16" => "BF16", "F8_E4M3" => "F8_E4M3", "F32" => "F32", other => panic!("{other}")
        };
        testing::t(name, dtype, &meta["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect::<Vec<_>>())
    }).collect();
    let dir = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir.path(), &config, &tensors, None);
    let shape = RoutedExpertShape { layers:91, first_layer:30, experts:352, topk:8,
        hidden:4096, intermediate:1536, draft_stages:0, draft_experts:0 };
    let selected = [30].into_iter().collect();
    let catalog = Fp8ExpertTensors::read_layers(dir.path(), shape, &selected).unwrap();
    assert_eq!(catalog.format(), ExpertFormat::Nvfp4);
    let cp = Checkpoint::open(dir.path()).unwrap();
    assert_eq!(cp.module_numerics("model.layers.30.moe.gate").unwrap(), ModuleNumerics::CheckpointPrecision);
    assert_eq!(cp.module_numerics("model.layers.30.moe.gate_proj").unwrap(), ModuleNumerics::W4A4);
    // Genuine owner exclusions still reject contradictory packed headers.
    let mut excluded = config.clone();
    excluded["quantization_config"]["ignore"] = serde_json::json!(["*layers.30.moe"]);
    std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&excluded).unwrap()).unwrap();
    assert!(Fp8ExpertTensors::read_layers(dir.path(), shape, &selected).unwrap_err().to_string().contains("declared CheckpointPrecision"));
    // A missing companion must not be reinterpreted as checkpoint precision.
    let partial: Vec<_> = tensors.into_iter().filter(|(name, ..)| name != "model.layers.30.moe.gate_proj.input_scale").collect();
    testing::write_snapshot(dir.path(), &config, &partial, None);
    assert!(Fp8ExpertTensors::read_layers(dir.path(), shape, &selected).unwrap_err().to_string().contains("input_scale"));
}
