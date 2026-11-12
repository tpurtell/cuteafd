use super::*;
use crate::plan::{self, testing, PlanOptions};

fn fixture() -> (tempfile::TempDir, serde_json::Value, Vec<testing::Tensor>) {
    let dir = tempfile::tempdir().unwrap();
    let cfg = crate::families::minimax_m3::config::tests::release_config();
    // Use the checkpoint's real shape recipe without writing 89k sparse records per test.
    let mut small = cfg;
    small["text_config"]["num_hidden_layers"] = 4.into();
    small["text_config"]["num_local_experts"] = 4.into();
    for key in ["moe_layer_freq"] {
        small["text_config"][key].as_array_mut().unwrap().truncate(4);
    }
    for key in ["sparse_attention_freq", "sparse_disable_index_value"] {
        small["text_config"]["sparse_attention_config"][key].as_array_mut().unwrap().truncate(4);
    }
    // Construct headers from the independent Python reference's spelling and shapes.
    let c = MiniMaxM3Config::from_hf(&small).unwrap();
    let mut tensors = Vec::new();
    let mut plain = |name: String, dtype, shape: Vec<usize>| tensors.push((name, dtype, shape));
    plain(format!("{TEXT}embed_tokens.weight"), "BF16", vec![c.vocab_size, c.hidden]);
    plain(format!("{HEAD}weight"), "BF16", vec![c.vocab_size, c.hidden]);
    plain(format!("{TEXT}norm.weight"), "BF16", vec![c.hidden]);
    for layer in 0..4 {
        let p = format!("{TEXT}layers.{layer}.");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            plain(format!("{p}{norm}.weight"), "BF16", vec![c.hidden]);
        }
        let mut matrix = |stem: String, n, k| {
            plain(format!("{stem}.weight"), "F8_E4M3", vec![n, k]);
            plain(format!("{stem}.weight_scale_inv"), "U8", vec![n, k / 32]);
        };
        for (name, n, k) in [("q_proj", 8192, 6144), ("k_proj", 512, 6144), ("v_proj", 512, 6144),
            ("o_proj", 6144, 8192), ("index_q_proj", 512, 6144), ("index_k_proj", 128, 6144)] {
            matrix(format!("{p}self_attn.{name}"), n, k);
        }
        let (stem, inter) = if layer == 3 { ("block_sparse_moe.shared_experts", 3072) } else { ("mlp", 12288) };
        for (proj, n, k) in [("gate", inter, 6144), ("up", inter, 6144), ("down", 6144, inter)] {
            matrix(format!("{p}{stem}.{proj}_proj"), n, k);
        }
        for norm in ["q_norm", "k_norm", "index_q_norm", "index_k_norm"] {
            plain(format!("{p}self_attn.{norm}.weight"), "BF16", vec![128]);
        }
        if layer == 3 {
            plain(format!("{p}block_sparse_moe.gate.weight"), "F32", vec![4, 6144]);
            plain(format!("{p}block_sparse_moe.e_score_correction_bias"), "F32", vec![4]);
            for e in 0..4 {
                for (w, n, k) in [("w1", 3072, 6144), ("w3", 3072, 6144), ("w2", 6144, 3072)] {
                    let stem = format!("{p}block_sparse_moe.experts.{e}.{w}");
                    plain(format!("{stem}.weight_packed"), "U8", vec![n, k / 2]);
                    plain(format!("{stem}.weight_scale"), "F8_E4M3", vec![n, k / 16]);
                    for scale in ["weight_global_scale", "input_global_scale"] {
                        plain(format!("{stem}.{scale}"), "F32", vec![]);
                    }
                }
            }
        }
    }
    small.as_object_mut().unwrap().remove("vision_config");
    testing::write_snapshot(dir.path(), &small, &tensors, None);
    (dir, small, tensors)
}

#[test]
fn minimax_native_inventory_is_planned_not_executable() {
    let (dir, _, _) = fixture();
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert_eq!(report.family.as_deref(), Some("minimax_m3"));
    assert_eq!(report.runtime, Some(RuntimeStatus::Planned));
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    assert!(report.components.iter().all(|c| c.rejected == 0), "{:?}", report.components);
    assert!(!report.executable());
    let spec = report.spec.unwrap();
    assert_eq!(spec.moe_layers(), 1);
    assert!(matches!(spec.layers[0].attention, AttentionKind::BlockSparseGqa { block: 128, topk_blocks: 16, .. }));
    assert_eq!(spec.layers[0].rope.unwrap().dims, 64);
}

#[test]
fn minimax_spark_recipe_uses_compressed_divisor_headers_without_opening_serving() {
    use crate::plan::format::{ScaleDirection, detect, group_by_stem};
    let (dir, _, _) = fixture();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let model = MINIMAX_M3.open(&checkpoint).unwrap();
    let stem = format!("{TEXT}layers.3.block_sparse_moe.experts.0.w1");
    let groups = group_by_stem(&checkpoint.tensors);
    let mut operand = detect(&groups[&stem]).unwrap();
    let contract = model.experts(&operand).unwrap();
    assert_eq!(contract.package, "minimax:nvfp4a4");
    assert_eq!(contract.block, 16);
    assert_eq!(contract.spark_worlds, (2..=8).collect::<Vec<_>>());
    assert!(contract.local.is_err());
    for world in 2..=8 {
        assert_eq!(crate::plan::experts::checkpoint_spark_recipe(&checkpoint, world, &[3]).unwrap(),
            crate::plan::experts::spark_recipe(&contract.package, 3072, world).unwrap());
    }
    operand.input_scale = None;
    assert!(model.experts(&operand).is_none());
    operand = detect(&groups[&stem]).unwrap();
    operand.global_scale.as_mut().unwrap().direction = ScaleDirection::Multiply;
    assert!(model.experts(&operand).is_none());
    assert!(model.experts(&detect(&groups[&format!("{TEXT}layers.0.self_attn.q_proj")]).unwrap()).is_none());
    assert!(!plan::plan(dir.path(), &PlanOptions::default()).unwrap().executable());
}

#[test]
fn generic_native_validation_rejects_missing_and_drifted_minimax_tensors_early() {
    let (dir, cfg, mut tensors) = fixture();
    let name = format!("{TEXT}layers.0.self_attn.index_k_proj.weight_scale_inv");
    tensors.iter_mut().find(|t| t.0 == name).unwrap().2[1] += 1;
    testing::write_snapshot(dir.path(), &cfg, &tensors, None);
    let diagnostic = match plan::plan(dir.path(), &PlanOptions::default()).unwrap_err() {
        plan::PlanError::Checkpoint(error) => error.to_string(),
        error => panic!("expected generic checkpoint validation, got {error:?}"),
    };
    assert!(diagnostic.contains(&name), "{diagnostic}");
    tensors.retain(|t| t.0 != name);
    testing::write_snapshot(dir.path(), &cfg, &tensors, None);
    let diagnostic = match plan::plan(dir.path(), &PlanOptions::default()).unwrap_err() {
        plan::PlanError::Checkpoint(error) => error.to_string(),
        error => panic!("expected generic checkpoint validation, got {error:?}"),
    };
    assert!(diagnostic.contains(&name), "{diagnostic}");
}

#[test]
fn coordinator_native_dense_workspace_prices_each_serial_owner_once() {
    use crate::families::minimax_m3::weight_layout::CoordinatorWeights;
    use crate::placement::planner::{ComponentKind, Lifetime};
    let (dir, _, _) = fixture();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config).unwrap();
    let weights = CoordinatorWeights::plan(&checkpoint, &cfg, 1, &[0; 4]).unwrap().remove(0);
    for rows in [1, 16, 17, 2752, 4096] {
        let layout = weights.dense_workspace(rows).unwrap();
        assert_eq!(layout.records, u64::from(rows).div_ceil(16) * (12288 / 32) * 528);
        assert_eq!(layout.output, u64::from(rows) * 12288 * 4);
        let components = layout.components("minimax.dense", 0, Lifetime::TARGET, 1024).unwrap();
        let owners = layout.allocations("minimax.dense", 0, Lifetime::TARGET);
        assert_eq!(owners.iter().map(|a| a.bytes).sum::<u64>(), 8 + layout.records + layout.output);
        let allocations = components.iter().filter_map(|c| match &c.kind {
            ComponentKind::Allocation { template, .. } => Some(template), _ => None }).collect::<Vec<_>>();
        assert_eq!(allocations, owners.iter().collect::<Vec<_>>());
        assert!(!components.iter().any(|c| matches!(c.kind, ComponentKind::ResidentWeight(_))));
        for component in &components {
            if let ComponentKind::AllocationGranularity(tail) = &component.kind {
                assert_eq!(tail.backings.len(), 1);
                let backing = owners.iter().find(|a| a.id == tail.backings[0].backing).unwrap();
                assert_eq!(tail.requests(), [backing.bytes]);
            }
        }
        let input = weights.dense_input(rows).unwrap();
        assert_eq!((input.width, input.bytes), (12288, u64::from(rows) * 12288 * 2));
        let input_components = input.components("minimax.dense", 0, Lifetime::TARGET, 1024).unwrap();
        let owner = input.allocation("minimax.dense", 0, Lifetime::TARGET);
        let input_allocations = input_components.iter().filter_map(|c| match &c.kind {
            ComponentKind::Allocation { template, .. } => Some(template), _ => None }).collect::<Vec<_>>();
        assert_eq!(input_allocations, [&owner]);
        assert!(!owners.iter().any(|a| a.id == owner.id));
        for component in &input_components {
            if let ComponentKind::AllocationGranularity(tail) = &component.kind {
                assert_eq!(tail.backings.len(), 1);
                assert_eq!(tail.backings[0].backing, owner.id);
                assert_eq!(tail.requests(), [owner.bytes]);
            }
        }
        assert!(input.components("", 0, Lifetime::TARGET, 1024).is_err());
        assert!(input.components("dense", 2, Lifetime::TARGET, 1024).is_err());
        assert!(input.components("dense", 0, Lifetime::TARGET, 0).is_err());
        assert!(layout.components("", 0, Lifetime::TARGET, 1024).is_err());
        assert!(layout.components("dense", 2, Lifetime::TARGET, 1024).is_err());
        assert!(layout.components("dense", 0, Lifetime::TARGET, 0).is_err());
    }
    assert!(weights.dense_workspace(0).is_err());
    assert!(weights.dense_workspace(4097).is_err());
    assert!(weights.dense_input(0).is_err());
    assert!(weights.dense_input(4097).is_err());
    let mut empty = weights;
    empty.projections.clear();
    assert!(empty.dense_workspace(1).is_err());
    assert!(empty.dense_input(1).is_err());
}

#[test]
fn coordinator_native_arena_preserves_checkpoint_bytes_and_explicit_owners() {
    use crate::families::minimax_m3::weight_layout::CoordinatorWeights;
    let (dir, _, _) = fixture();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config).unwrap();
    let weights = CoordinatorWeights::plan(&checkpoint, &cfg, 2, &[1, 0, 1, 0]).unwrap();
    assert_eq!(weights[0].layers, [1, 3]);
    assert_eq!(weights[1].layers, [0, 2]);
    for (rank, weight) in weights.iter().enumerate() {
        assert_eq!(weight.arena.id(), format!("minimax/target/{rank}"));
        assert_eq!(weight.projections.len(), 18);
        assert_eq!(weight.records_bytes(17).unwrap(), 2 * (12288 / 32) * 528);
        assert!(weight.records_bytes(0).is_err());
        assert!(weight.records_bytes(4097).is_err());
        assert!(weight.dense_programs().contains("exlr8_dense_mxfp8_k6144_n8192_s1_a8"));
        assert!(weight.dense_programs().contains("exlr8_dense_mxfp8_k12288_n6144_s1_a8"));
        assert_eq!(weight.arena.payload_bytes(), weight.arena.views().iter().map(|(name, view)| {
            assert_eq!(view.offset() % 256, 0);
            let header = checkpoint.header(name).unwrap();
            assert_eq!(view.bytes() as u64, header.byte_length);
            assert!(!name.contains(".experts."));
            header.byte_length as usize
        }).sum::<usize>());
        assert!(weight.arena.view(&format!("{TEXT}norm.weight")).is_some());
        assert_eq!(weight.arena.view(&format!("{TEXT}embed_tokens.weight")).is_some(), rank == 0);
        assert_eq!(weight.arena.view("language_model.lm_head.weight").is_some(), rank == 1);
        let components = weight.weight_components(1024).unwrap();
        use crate::placement::planner::ComponentKind;
        assert_eq!(components.len(), 3);
        let ComponentKind::Allocation { template, .. } = &components[0].kind else { panic!("root") };
        assert_eq!(template.id, weight.arena.id());
        assert_eq!(template.bytes, weight.arena.bytes() as u64);
        assert!(template.view.is_none());
        let ComponentKind::ResidentWeight(resident) = &components[1].kind else { panic!("resident") };
        assert_eq!(resident.id, template.id);
        assert_eq!(resident.weights.0, template.bytes);
        let ComponentKind::AllocationGranularity(tail) = &components[2].kind else { panic!("tail") };
        assert_eq!(tail.requests(), [template.bytes]);
        assert_eq!(tail.backings[0].backing, template.id);
        assert!(weight.weight_components(0).is_err());
        let load = weight.mount_load(&checkpoint).unwrap();
        assert_eq!(load.full_row_cache, 0);
        assert_eq!(load.pinned_tables, 0);
        assert!(load.pinned_source > 0 && load.pinned_source <= crate::shared::dense_read::STEP_BYTES as u64);
        let owners = load.allocations("minimax.mount");
        assert_eq!(owners.len(), 1);
        assert_eq!(owners[0].location, crate::placement::planner::Location::PINNED);
        assert_eq!(owners[0].lifetime, crate::placement::planner::Lifetime {
            first: crate::placement::planner::Phase::Load, last: crate::placement::planner::Phase::Load });
    }
    assert!(weights[0].arena.view(&format!("{TEXT}layers.3.block_sparse_moe.gate.weight")).is_some());
    let single = CoordinatorWeights::plan(&checkpoint, &cfg, 1, &[0; 4]).unwrap();
    // TP2 duplicates only final norm, not layer payloads or global matrices.
    assert_eq!(weights.iter().map(|w| w.arena.payload_bytes()).sum::<usize>(),
        single[0].arena.payload_bytes() + cfg.hidden * 2);
    assert!(CoordinatorWeights::plan(&checkpoint, &cfg, 0, &[0; 4]).is_err());
    assert!(CoordinatorWeights::plan(&checkpoint, &cfg, 2, &[0; 3]).is_err());
    assert!(CoordinatorWeights::plan(&checkpoint, &cfg, 2, &[0, 1, 2, 0]).is_err());
    let mut wrong = cfg.clone();
    wrong.hidden += 128;
    assert!(CoordinatorWeights::plan(&checkpoint, &wrong, 1, &[0; 4]).is_err());
}

fn dense_manifest(weights: &crate::families::minimax_m3::weight_layout::CoordinatorWeights) -> serde_json::Value {
    use serde_json::json;
    let scalars = json!([{"name":"rows","type":"int32"},
        {"name":"rotation_id","type":"int32"}, {"name":"cta_limit","type":"int32"}]);
    let mut entries = std::collections::BTreeMap::new();
    for projection in &weights.projections {
        let (k, n) = (projection.input, projection.output);
        let quant = format!("exlr8_dense_quant_k{k}_bf16_a8");
        entries.insert(quant.clone(), json!({"name":quant,"op":"dense_mxfp8_quant",
            "capacity_rows":4096,"scratch_bytes_at_capacity":{},"scalars":scalars,
            "geometry":{"input_features":k,"input_dtype":"bf16","rotation_count":1,"max_rows":4096,"abi_version":2},
            "pointers":[{"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
                {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"out"}]}));
        entries.insert(projection.program_name(), json!({"name":projection.program_name(),"op":"dense_mxfp8",
            "capacity_rows":4096,"scratch_bytes_at_capacity":{},"scalars":scalars,
            "geometry":{"input_features":k,"output_features":n,"sharded_side":1,"encoding":"mxfp8",
                "input_dtype":"bf16","rotation_count":1,"max_rows":4096,"abi_version":2},
            "pointers":[{"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
                {"name":"payload","dtype":"uint8","shape":format!("[{}]", k*n),"role":"in"},
                {"name":"scales","dtype":"uint8","shape":format!("[{}]", n*k/32),"role":"in"},
                {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"in"},
                {"name":"output","dtype":"float32","shape":format!("[rows,{n}]"),"role":"out"}]}));
    }
    json!({"programs":entries.into_values().collect::<Vec<_>>()})
}

#[test]
fn coordinator_native_program_inventory_validates_full_abi_and_quantizers() {
    use crate::families::minimax_m3::weight_layout::CoordinatorWeights;
    use serde_json::json;
    let (dir, _, _) = fixture();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config).unwrap();
    let weights = CoordinatorWeights::plan(&checkpoint, &cfg, 1, &[0; 4]).unwrap().remove(0);
    let manifest = dense_manifest(&weights);
    assert_eq!(weights.dense_programs().len(), 8);
    assert_eq!(weights.quantizer_programs().len(), 4);
    let programs = weights.resolve_dense_programs(&manifest, 2752).unwrap();
    assert_eq!(programs.len(), 12);
    assert!(programs.windows(2).all(|p| p[0].name < p[1].name));
    for rows in [0, 4097] { assert!(weights.resolve_dense_programs(&manifest, rows).is_err()); }
    assert!(weights.resolve_dense_programs(&json!({}), 1).is_err());
    let mut missing = manifest.clone();
    missing["programs"].as_array_mut().unwrap().pop();
    assert!(weights.resolve_dense_programs(&missing, 1).is_err());
    let mut duplicate = manifest.clone();
    duplicate["programs"].as_array_mut().unwrap().push(manifest["programs"][0].clone());
    assert!(weights.resolve_dense_programs(&duplicate, 1).is_err());
    for index in 0..12 {
        for (field, value) in [("input_dtype", json!("f16")), ("abi_version", json!(null)),
            ("rotation_count", json!(4)), ("input_features", json!(512)), ("max_rows", json!(64))] {
            let mut bad = manifest.clone();
            bad["programs"][index]["geometry"][field] = value;
            assert!(weights.resolve_dense_programs(&bad, 1).is_err(), "{index} {field}");
        }
        for (field, value) in [("dtype", "float16"), ("shape", "[4]"), ("role", "out")] {
            let mut bad = manifest.clone();
            bad["programs"][index]["pointers"][0][field] = json!(value);
            assert!(weights.resolve_dense_programs(&bad, 1).is_err(), "{index} {field}");
        }
        let mut bad = manifest.clone();
        bad["programs"][index]["scalars"][0]["name"] = json!("count");
        assert!(weights.resolve_dense_programs(&bad, 1).is_err());
    }
}

#[test]
fn coordinator_native_arena_refuses_missing_scale_header() {
    use crate::families::minimax_m3::weight_layout::CoordinatorWeights;
    let (dir, _, _) = fixture();
    let mut checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config).unwrap();
    let scale = format!("{TEXT}layers.0.self_attn.q_proj.weight_scale_inv");
    checkpoint.tensors.retain(|tensor| tensor.meta.name != scale);
    let error = CoordinatorWeights::plan(&checkpoint, &cfg, 1, &[0; 4]).unwrap_err().to_string();
    assert!(error.contains(&scale), "{error}");
}

#[test]
#[ignore = "requires the installed vendor checkpoint"]
fn installed_minimax_headers_match_native_contract() {
    // Explicitly ignored: requires the installed checkpoint, but never reads payloads or a GPU.
    let path = std::path::Path::new("/mnt/sparknest/hf-home/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e");
    let checkpoint = Checkpoint::open(path).unwrap();
    let model = Model::open(&checkpoint).unwrap();
    assert_eq!(model.cfg.moe_layers(), 57);
    assert_eq!(checkpoint.tensors.len(), 89_632);
    assert!(checkpoint.spark_input_is_bf16());
    use crate::plan::checkpoint::{ModuleNumerics, ProjectionStorage};
    for module in [format!("{TEXT}layers.0.self_attn.q_proj"),
        format!("{TEXT}layers.59.self_attn.index_k_proj"),
        format!("{TEXT}layers.3.block_sparse_moe.shared_experts.down_proj")] {
        assert_eq!(checkpoint.module_numerics(&module).unwrap(), ModuleNumerics::W8A8, "{module}");
    }
    let expert = format!("{TEXT}layers.3.block_sparse_moe.experts.0.w1");
    assert_eq!(checkpoint.module_numerics(&expert).unwrap(), ModuleNumerics::W4A4);
    assert_eq!(checkpoint.projection(&expert).unwrap().storage,
        ProjectionStorage::CompressedNvfp4 { shape: vec![3072, 6144] });
    assert_eq!(checkpoint.module_numerics(&format!("{TEXT}layers.3.block_sparse_moe.gate")).unwrap(),
        ModuleNumerics::CheckpointPrecision);
    let weights = crate::families::minimax_m3::weight_layout::CoordinatorWeights::plan(
        &checkpoint, &model.cfg, 1, &vec![0; model.cfg.layers]).unwrap();
    let load = weights[0].mount_load(&checkpoint).unwrap();
    assert_eq!(load, crate::shared::dense_read::MountLoad {
        pinned_source: crate::shared::dense_read::STEP_BYTES as u64, full_row_cache: 0, pinned_tables: 0 });
    assert_eq!(weights[0].projections.len(), 60 * 9);
    assert_eq!(weights[0].records_bytes(2752).unwrap(), (2752 / 16) * (12288 / 32) * 528);
    assert!(weights[0].arena.views().keys().all(|name| !name.contains(".experts.") && !name.starts_with("vision_tower.")));
    let report = plan::plan(path, &PlanOptions::default()).unwrap();
    eprintln!("MiniMax installed plan: min_spark_ranks={:?}, spark_budget_bytes={}",
        report.min_spark_ranks, report.spark_budget_bytes);
    // Runtime placement remains gated, so this is not a serving-fit witness.
    assert_eq!(report.min_spark_ranks, None);
    let routed_bytes = checkpoint.tensors.iter()
        .filter(|tensor| tensor.meta.name.contains(".block_sparse_moe.experts."))
        .map(|tensor| tensor.meta.byte_length).sum::<u64>();
    assert_eq!(routed_bytes, 232_381_393_920);
    assert!(routed_bytes > 121 << 30, "one GB10 cannot hold the native experts");
    eprintln!("MiniMax installed routed expert bytes={routed_bytes}; one GB10 is physically insufficient");
    assert_eq!(report.spark_budget_bytes, 100 << 30);
}
