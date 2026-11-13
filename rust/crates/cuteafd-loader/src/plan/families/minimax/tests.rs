use super::*;
use crate::plan::{self, testing, PlanOptions};

fn fixture() -> (tempfile::TempDir, serde_json::Value, Vec<testing::Tensor>) {
    fixture_with_experts(4)
}

fn fixture_with_experts(experts: usize) -> (tempfile::TempDir, serde_json::Value, Vec<testing::Tensor>) {
    let dir = tempfile::tempdir().unwrap();
    let cfg = crate::families::minimax_m3::config::tests::release_config();
    // Use the checkpoint's real shape recipe without writing 89k sparse records per test.
    let mut small = cfg;
    small["text_config"]["num_hidden_layers"] = 4.into();
    small["text_config"]["num_local_experts"] = experts.into();
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
            plain(format!("{p}block_sparse_moe.gate.weight"), "F32", vec![experts, 6144]);
            plain(format!("{p}block_sparse_moe.e_score_correction_bias"), "F32", vec![experts]);
            for e in 0..experts {
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

pub(crate) fn concurrency_matrix_snapshot() -> (tempfile::TempDir, std::path::PathBuf) {
    use crate::families::minimax_m3::weight_layout::CoordinatorWeights;
    let (dir, config, _) = fixture_with_experts(128);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = MiniMaxM3Config::from_hf(&config).unwrap();
    let weights = CoordinatorWeights::plan(&checkpoint, &cfg, 1, &[0; 4]).unwrap();
    let mut programs = dense_manifest(&weights[0])["programs"].as_array().unwrap().clone();
    for name in ["minimax_msa_programs", "minimax_producer_programs", "minimax_forward_programs"] {
        let fixture: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!(
            "{}/../cuteafd-ffi/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        programs.extend(fixture["programs"].as_array().unwrap().clone());
    }
    let manifest = dir.path().join("PROGRAMS.json");
    std::fs::write(&manifest, serde_json::json!({"programs": programs}).to_string()).unwrap();
    (dir, manifest)
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
fn cache_geometry_preserves_native_fp8_pages_and_physical_layer_owners() {
    use crate::serving_capacity::{CacheOptions, KvPlacement};
    use crate::placement::AttentionPlacement;
    let (dir, config, _) = fixture();
    let cfg = MiniMaxM3Config::from_hf(&config).unwrap();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let model = MINIMAX_M3.open(&checkpoint).unwrap();
    let rows = crate::placement::families::minimax_m3::PAGE_ROWS;
    let kv = rows * cfg.kv_heads as u64 * cfg.head_dim as u64;
    let index = rows * (cfg.msa.index_dim as u64 + 4);
    assert_eq!((kv, kv, index), (32768, 32768, 8448));
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../cuteafd-ffi/tests/fixtures/minimax_msa_programs.json")).unwrap();
    let mut cache_operands = 0;
    for pointer in manifest["programs"].as_array().unwrap().iter()
        .flat_map(|program| program["pointers"].as_array().unwrap())
        .filter(|pointer| matches!(pointer["name"].as_str(), Some("key_cache" | "value_cache"))) {
        assert_eq!(pointer["dtype"], "uint8");
        assert_eq!(pointer["shape"], "[pool_pages,64,4,128] E4M3 bytes");
        cache_operands += 1;
    }
    assert_eq!(cache_operands, 4);
    let single = model.cache_geometry(CacheOptions::default()).unwrap().unwrap();
    assert_eq!(single.logical_unit_rows, rows);
    assert_eq!(single.placement, KvPlacement::SingleDevice);
    assert_eq!(single.ranks[0].persistent_unit_bytes, cfg.layers as u64 * (2 * kv + index));
    for gpus in [1, 2] {
        let geometry = model.cache_geometry(CacheOptions { coordinator_ranks: gpus, ..Default::default() }).unwrap().unwrap();
        let owners = crate::placement::families::minimax_m3::layer_owners(cfg.layers, gpus);
        let native = crate::families::minimax_m3::execution::MsaCache::new(cfg.layers as u32, 1, 1, 64).unwrap();
        let native = native.buffers().unwrap();
        for gpu in 0..gpus {
            let owned: u64 = owners.iter().enumerate().filter(|(_, owner)| usize::from(**owner) == gpu)
                .map(|(layer, _)| native.iter().filter(|(name, _)| name.starts_with(&format!("layer.{layer}.")))
                    .map(|(_, bytes)| bytes).sum::<u64>()).sum();
            assert_eq!(geometry.ranks[gpu].persistent_unit_bytes, owned);
            assert_eq!(geometry.ranks[gpu].context_table_bytes_per_token, cfg.rotary_dim as u64 * 4);
        }
        assert_eq!(geometry.placement, if gpus == 1 { KvPlacement::SingleDevice } else { KvPlacement::PartitionedLayers });
    }
    for mode in [AttentionPlacement::Heads, AttentionPlacement::Context] {
        assert!(model.cache_geometry(CacheOptions { coordinator_ranks: 2, attention_placement: Some(mode),
            ..Default::default() }).is_err());
    }
    assert!(model.cache_geometry(CacheOptions { coordinator_ranks: 3, ..Default::default() }).is_err());
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
    // H1: RTX onboarding is an admitted contract owner; serve stays refused.
    assert!(contract.local.as_ref().is_ok_and(|package| package.contains("fp8-minimax-nvfp4a4")));
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

/// M1: open is structural. A consistent but drifted header (here a norm
/// stored F32 instead of BF16) and an extra tensor both open, then report a
/// named census rejection on the exact tensor, instead of failing the whole
/// open with one message.
#[test]
fn structural_open_reports_exact_header_drift_and_unknown_tensors_by_name() {
    let (dir, cfg, mut tensors) = fixture();
    let norm = format!("{TEXT}layers.1.input_layernorm.weight");
    tensors.iter_mut().find(|t| t.0 == norm).unwrap().1 = "F32";
    tensors.push((format!("{TEXT}layers.0.self_attn.extra_norm.weight"), "BF16", vec![128]));
    testing::write_snapshot(dir.path(), &cfg, &tensors, None);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    let rejections = report.components.iter().flat_map(|c| &c.rejections).collect::<Vec<_>>();
    assert!(rejections.iter().any(|r| r.tensor.contains("layers.1.input_layernorm")
        && r.reason.contains("expected Bf16")), "{rejections:?}");
    assert!(rejections.iter().any(|r| r.tensor.contains("extra_norm")
        && r.reason.contains("outside the MiniMax M3 tensor contract")), "{rejections:?}");
    assert!(!report.executable());
    // A missing contract member is still a typed open error naming it.
    tensors.retain(|t| t.0 != norm);
    testing::write_snapshot(dir.path(), &cfg, &tensors, None);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.as_deref().is_some_and(|e| e.contains(&norm)), "{:?}", report.config_error);
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
    // Header-derived package fit is not a serving qualification.
    assert_eq!(report.min_spark_ranks, Some(3));
    assert_eq!(report.runtime, Some(RuntimeStatus::Planned));
    assert!(!report.fits && !report.placement_supported && !report.executable());
    let routed_bytes = checkpoint.tensors.iter()
        .filter(|tensor| tensor.meta.name.contains(".block_sparse_moe.experts."))
        .map(|tensor| tensor.meta.byte_length).sum::<u64>();
    assert_eq!(routed_bytes, 232_381_393_920);
    assert!(routed_bytes > 121 << 30, "one GB10 cannot hold the native experts");
    eprintln!("MiniMax installed routed expert bytes={routed_bytes}; one GB10 is physically insufficient");
    assert_eq!(report.spark_budget_bytes, 100 << 30);
}

fn recipe_inputs(gpus: usize, ranks: usize) -> crate::placement::families::minimax_m3::MiniMaxInputs {
    use crate::families::minimax_m3::weight_layout::CoordinatorWeights;
    use crate::placement::families::minimax_m3::{self as shared, LayerExperts, LocalPackage, MiniMaxInputs, SparkCharge};
    use crate::placement::planner::{Context, PrefillShape, serving};
    let (dir, small, _) = fixture();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = MiniMaxM3Config::from_hf(&small).unwrap();
    let fixture = |name: &str| -> serde_json::Value { serde_json::from_str(&std::fs::read_to_string(
        format!("{}/../cuteafd-ffi/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() };
    let routed = (0..cfg.layers).filter(|&layer| cfg.moe[layer]).collect::<Vec<_>>();
    let owners = shared::layer_owners(cfg.layers, gpus);
    let targets = CoordinatorWeights::plan(&checkpoint, &cfg, gpus,
        &owners.iter().map(|&g| usize::from(g)).collect::<Vec<_>>()).unwrap();
    // Programs: MSA, producers, forward and this layout's dense exports.
    let mut programs = Vec::new();
    for name in ["minimax_msa_programs", "minimax_producer_programs", "minimax_forward_programs"] {
        programs.extend(fixture(name)["programs"].as_array().unwrap().clone());
    }
    let mut dense = std::collections::BTreeMap::new();
    for target in &targets {
        for entry in dense_manifest(target)["programs"].as_array().unwrap() {
            dense.insert(entry["name"].as_str().unwrap().to_owned(), entry.clone());
        }
    }
    programs.extend(dense.into_values());
    let manifest = std::sync::Arc::new(serde_json::json!({"programs": programs}));
    let prefill = PrefillShape { rows: 1024, waves: 3 };
    let charge = |count: usize| (0..ranks).map(|_| if count == routed.len() {
        SparkCharge { experts: 0, owners: 0, runtime: 0 } } else {
        SparkCharge { experts: ((routed.len() - count) as u64) << 30, owners: 1 << 30, runtime: 1 << 30 } }).collect();
    let budgets = (0..gpus).map(|gpu| shared::planned_budget(gpu as u8, 96 << 30, 96 << 30, 1 << 30))
        .chain((0..ranks).map(|rank| shared::spark_budget(rank as u8, 100 << 30)))
        .chain([crate::placement::families::kimi::host_budget(512 << 30, 480 << 30)]).collect();
    // Executor geometry checks are release-exact (E128); the fixture keeps four experts' headers.
    let mut release = cfg.clone(); release.experts = 128;
    MiniMaxInputs { cfg: release, gpus, spark_ranks: ranks, context: Context::automatic(1 << 20),
        maximum_pool_tokens: 4 << 20, prefill, sequences: 16, targets: targets.clone(), owners: owners.clone(),
        manifest, routed: routed.clone(), experts: vec![LayerExperts { whole: 1 << 30, half: [1 << 29; 2] }; routed.len()],
        local: LocalPackage::Installed { workspace: [64 << 20, if gpus == 2 { 64 << 20 } else { 0 }] },
        spark: (0..=routed.len()).map(charge).collect(),
        links: shared::spark_link((gpus - 1) as u8, ranks, prefill, Default::default()).unwrap(),
        spark_reply_formats: Default::default(), drafter: None, vision: None, budgets,
        code: vec![64 << 20; gpus], graph_reserve: vec![500 << 20; gpus], head: serving::BlasHead::Gemm,
        host_prefix: serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }, device_sampling: true,
        expert_usage: None, resident_weight_limit: None,
        mount: targets.iter().map(|t| t.mount_load(&checkpoint).unwrap()).collect() }
}

/// H1 contract: the shared recipe charges each physical owner exactly once on
/// one RTX and on a pair: one target arena backing per rank, MSA KV/index pages
/// only on each layer's attention owner, expert arenas and Spark owners only
/// under their own onboarding count, and the solve never re-derives a policy.
#[test]
fn shared_recipe_charges_every_owner_once_on_one_and_two_rtx() {
    use crate::placement::families::minimax_m3 as shared;
    use crate::placement::planner::{Location, ComponentKind, Recipe};
    for gpus in [1usize, 2] {
        let inputs = recipe_inputs(gpus, 2);
        let (owners, targets, prefill) = (&inputs.owners, &inputs.targets, inputs.prefill);
        let request = shared::request(inputs.context, None, crate::placement::Onboard::Auto, prefill,
            crate::placement::planner::Operation::Serving, None, Default::default());
        let contract = inputs.prepare(&request).unwrap();
        // Every allocation id is unique within one activation class.
        let mut seen = std::collections::BTreeSet::new();
        for component in &contract.components {
            if let ComponentKind::Allocation { template, .. } = &component.kind {
                assert!(seen.insert((template.id.clone(), format!("{:?}", component.activation))), "{} twice", template.id);
            }
        }
        // One target arena backing per rank.
        for gpu in 0..gpus {
            let arenas = contract.components.iter().filter(|c| matches!(&c.kind, ComponentKind::Allocation { template, .. }
                if template.group == "weight arena" && template.location == Location::coordinator(gpu as u8))).count();
            assert_eq!(arenas, 1, "rtx{gpu}");
        }
        // KV/index pages live only on each layer's attention owner.
        for (layer, &owner) in owners.iter().enumerate() {
            let homes = contract.components.iter().filter_map(|c| match &c.kind {
                ComponentKind::Allocation { template, .. } if template.id.starts_with(&format!("minimax.kv.{layer}.")) =>
                    Some(template.location), _ => None }).collect::<std::collections::BTreeSet<_>>();
            assert_eq!(homes, [Location::coordinator(owner)].into(), "layer {layer}");
        }
        let admitted = inputs.admit(request).unwrap();
        assert!(admitted.solved.pool_tokens >= 1 << 20);
        assert!(admitted.placement.layers.iter().all(|l| shared::MINIMAX_M3.runs(l.mode)));
        if gpus == 2 {
            // Whole owners: each layer's weights live on exactly one rank.
            for layer in 0..inputs.cfg.layers {
                let stem = format!("{TEXT}layers.{layer}.input_layernorm.weight");
                let homes = targets.iter().filter(|t| t.arena.view(&stem).is_some()).map(|t| t.rank).collect::<Vec<_>>();
                assert_eq!(homes, [usize::from(owners[layer])], "layer {layer}");
            }
            assert!(admitted.placement.tp2.is_none_or(|t| t.peak_bytes[0] == t.peak_bytes[1]));
        }
    }
}


#[test]
fn shared_recipe_supports_layouts_and_measured_baselines_without_replanning() {
    use crate::placement::families::minimax_m3 as shared;
    use crate::placement::planner::{Baseline, Operation, Phase};
    use crate::placement::{ExpertHome, Onboard};
    for (gpus, ranks) in (2..=8).map(|ranks| (1, ranks)).chain([(2, 4), (2, 6)]) {
        let inputs = recipe_inputs(gpus, ranks);
        let request = shared::request(inputs.context, Some(32768), Onboard::Layers(0), inputs.prefill,
            Operation::Serving, None, Default::default());
        let planned = inputs.admit(request.clone()).unwrap();
        assert_eq!(planned.placement.pool_tokens, 32768);
        assert!(planned.placement.layers.iter().filter(|l| l.experts != ExpertHome::Dense)
            .all(|l| l.experts == ExpertHome::Spark));
        let mut measured = inputs.clone();
        for budget in &mut measured.budgets[..gpus] {
            let Baseline::Forecast { occupied, .. } = budget.baseline else { panic!("forecast") };
            budget.baseline = Baseline::Sampled { free: budget.physical_bytes - occupied,
                phase: Phase::Load, source: "same pre-weight sample".into() };
        }
        let runtime = measured.admit(request).unwrap();
        assert_eq!(planned.solved.pool_tokens, runtime.solved.pool_tokens);
        assert_eq!(planned.solved.decisions, runtime.solved.decisions);
        assert_eq!(planned.solved.facts, runtime.solved.facts);
        for gpu in 0..gpus {
            if gpus == 2 {
                let hops = planned.solved.facts.allocations.iter()
                    .filter(|owner| owner.id == shared::Owner::Hops { gpu }.id()).collect::<Vec<_>>();
                // hop_buffer_bytes owns the slot count; the recipe supplies FP32 residual rows.
                assert!(hops.iter().all(|owner| owner.bytes % (inputs.cfg.hidden as u64 * 4) == 0));
            }
        }
    }
}

#[test]
fn shared_recipe_refuses_invalid_owners_concurrency_and_missing_exports() {
    use crate::placement::families::minimax_m3::{self as shared, LocalPackage};
    use crate::placement::planner::{Operation, Recipe, Refusal};
    let inputs = recipe_inputs(2, 4);
    let request = shared::request(inputs.context, Some(32768), crate::placement::Onboard::Layers(0),
        inputs.prefill, Operation::Serving, None, Default::default());
    for concurrency in [0, 65, 128] {
        let mut bad = inputs.clone();
        bad.sequences = concurrency;
        assert!(matches!(bad.prepare(&request), Err(Refusal::InvalidFact(_))));
    }
    let mut bad = inputs.clone();
    bad.owners.swap(0, 3);
    assert!(bad.prepare(&request).unwrap_err().to_string().contains("contiguous"));
    let mut bad = inputs.clone();
    bad.spark[0].pop();
    assert!(bad.prepare(&request).unwrap_err().to_string().contains("rank inventory"));
    let mut bad = inputs.clone();
    bad.local = LocalPackage::Missing("missing fp8-minimax-nvfp4a4/tp2/libcuteafd_fp8moe.so".into());
    let mut pinned = request.clone();
    pinned.onboard = crate::placement::planner::Pin::Exact(1);
    assert!(bad.prepare(&pinned).unwrap_err().to_string().contains("fp8-minimax-nvfp4a4/tp2"));
    // Missing local exports never prevent a remote-only layout.
    bad.admit(request.clone()).unwrap();
    let mut bad = inputs.clone();
    let manifest = std::sync::Arc::make_mut(&mut bad.manifest);
    manifest["programs"].as_array_mut().unwrap().retain(|p|
        p["name"].as_str() != Some("exlr8_dense_mxfp8_k6144_n8192_s1_a8"));
    let error = bad.prepare(&request).unwrap_err().to_string();
    assert!(error.contains("exlr8_dense_mxfp8_k6144_n8192_s1_a8"), "{error}");
}

#[test]
fn shared_recipe_onboards_whole_layers_or_tp2_halves() {
    use crate::placement::families::minimax_m3 as shared;
    use crate::placement::planner::Operation;
    use crate::placement::{ExpertHome, Onboard};
    for gpus in [1, 2] {
        let inputs = recipe_inputs(gpus, 0);
        let request = shared::request(inputs.context, Some(32768), Onboard::Layers(1), inputs.prefill,
            Operation::Serving, None, Default::default());
        let admitted = inputs.admit(request).unwrap();
        assert_eq!(admitted.onboard, 1);
        assert_eq!(admitted.placement.layers[3].experts,
            if gpus == 1 { ExpertHome::RtxWhole { gpu: 0 } } else { ExpertHome::RtxTp2 });
        assert_eq!(admitted.placement.tp2.is_some(), gpus == 2);
        let bytes = admitted.solved.facts.allocations.iter()
            .filter(|owner| owner.id.starts_with("minimax.experts."))
            .map(|owner| owner.bytes).collect::<Vec<_>>();
        assert_eq!(bytes, vec![if gpus == 1 { (1 << 30) + (64 << 20) }
            else { (1 << 29) + (64 << 20) }; gpus]);
    }
}

#[test]
fn prefix_ordered_uploads_match_each_request_destination() {
    use crate::placement::families::minimax_m3::{self as shared, Owner};
    use crate::placement::planner::{Operation, Location};
    for gpus in [1, 2] {
        for concurrency in [1, 4, 16, 64] {
            let mut inputs = recipe_inputs(gpus, 4);
            inputs.sequences = concurrency;
            let request = shared::request(inputs.context, Some(32768), crate::placement::Onboard::Layers(0),
                inputs.prefill, Operation::Serving, None, Default::default());
            let admitted = inputs.admit(request).unwrap();
            let mut bytes = 0;
            for gpu in 0..gpus {
                for slot in 0..concurrency {
                    for (plane, expected) in [("table", inputs.context.maximum.div_ceil(64) * 4), ("length", 4)] {
                        let owner = Owner::PrefixH2d { gpu, slot, plane };
                        assert_eq!(format!("minimax.{}", owner.inventory_name().unwrap()), owner.id());
                        let allocation = admitted.solved.facts.allocations.iter().find(|a| a.id == owner.id()).unwrap();
                        assert_eq!(allocation.location, Location::PINNED);
                        assert_eq!(allocation.bytes, expected);
                        assert!(allocation.view.is_none());
                        bytes += expected;
                    }
                }
            }
            assert_eq!(bytes, gpus as u64 * concurrency * (4 * inputs.context.maximum.div_ceil(64) + 4));
        }
    }
}

#[test]
fn expert_backing_descriptor_uses_canonical_mount_regions_and_physical_ids() {
    use crate::placement::families::minimax_m3::{ExpertBacking, ExpertStorage};
    let (root, cfg, _) = fixture();
    let cfg = MiniMaxM3Config::from_hf(&cfg).unwrap();
    let catalog = crate::read_expert_catalog(root.path()).unwrap();
    let tensors = catalog.fp8().unwrap();
    let layers = (0..cfg.layers).filter(|&layer| cfg.moe[layer]).collect::<Vec<_>>();
    for gpus in [1, 2] {
        for gpu in 0..gpus {
            for count in 1..=layers.len() {
                let backing = ExpertBacking::plan(tensors, &layers[..count], gpus, gpu, (1 - gpu) as i32).unwrap();
                assert_eq!(backing.storage, ExpertStorage::Mount);
                assert_eq!(backing.layers.len(), count);
                let regions = crate::serving_capacity::fp8_expert_regions(&tensors.for_coordinator(),
                    &layers[..count], gpus, gpu, crate::formats::fp8_experts::Slicing::Blocks(128)).unwrap();
                assert_eq!(backing.arena.id(), crate::serving_capacity::expert_weight_arena_id("fp8",
                    (1 - gpu) as i32, &regions).unwrap());
                for layer in &backing.layers {
                    for (plane, range) in layer.planes.iter().enumerate() {
                        let view = backing.arena.view(&format!("{}.{plane}", layer.layer)).unwrap();
                        assert_eq!(range, &(view.offset()..view.offset() + view.bytes()));
                        assert!(range.start >= layer.region.start && range.end <= layer.region.end);
                    }
                }
                assert_eq!(backing.validation_bytes, 0);
            }
        }
    }
}

#[test]
fn heads_target_slices_match_headers_and_charge_one_arena_per_rank() -> anyhow::Result<()> {
    use crate::families::minimax_m3::weight_layout::{CoordinatorWeights, HeadsWeights};
    use crate::placement::planner::{ComponentKind, EmbeddingHome};
    let (dir, _, _) = fixture();
    let checkpoint = Checkpoint::open(dir.path())?;
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
    for embedding in [EmbeddingHome::Device(0), EmbeddingHome::HostPinned] {
        let heads = HeadsWeights::plan(&checkpoint, &cfg, embedding)?;
        let whole = CoordinatorWeights::plan_with_embedding(&checkpoint, &cfg, 2, &[0, 0, 1, 1], embedding)?;
        let mut replicated = 0;
        let mut unique = std::collections::BTreeMap::new();
        for target in &whole { for name in target.arena.views().keys() {
            unique.insert(name.clone(), checkpoint.header(name).unwrap().byte_length as usize);
        }}
        for (name, bytes) in &unique {
            let slices = heads.iter().filter_map(|h| h.slices.get(name)).collect::<Vec<_>>();
            let payload: usize = slices.iter().map(|s| s.bytes()).sum();
            if slices.len() == 2 && slices.iter().all(|s| s.bytes() == *bytes) {
                replicated += bytes;
                assert_eq!(payload, 2 * bytes, "{name}");
            } else { assert_eq!(payload, *bytes, "{name}"); }
        }
        assert_eq!(heads.iter().map(|h| h.target.arena.payload_bytes()).sum::<usize>(),
            unique.values().sum::<usize>() + replicated);
        for (rank, h) in heads.iter().enumerate() {
            assert_eq!(h.target.rank, rank);
            assert_eq!(h.target.layers, [0, 1, 2, 3]);
            assert_eq!(h.target.arena.bytes(), h.slices.values().map(|s| s.bytes()).sum::<usize>());
            assert_eq!(h.target.arena.padding_bytes(), 0);
            h.mount_load(&checkpoint)?;
            let components = h.target.weight_components(1024)?;
            assert_eq!(components.iter().filter(|c| matches!(&c.kind,
                ComponentKind::Allocation { template, .. } if template.view.is_none())).count(), 1);
            assert_eq!(components.iter().filter(|c| matches!(c.kind,
                ComponentKind::AllocationGranularity(_))).count(), 1);
            for (stem, rows, columns, row_bytes) in [
                ("self_attn.q_proj.weight", rank*4096..(rank+1)*4096, 0..6144, 6144),
                ("self_attn.q_proj.weight_scale_inv", rank*4096..(rank+1)*4096, 0..192, 192),
                ("self_attn.k_proj.weight", rank*256..(rank+1)*256, 0..6144, 6144),
                ("self_attn.index_q_proj.weight", rank*256..(rank+1)*256, 0..6144, 6144),
                ("self_attn.index_k_proj.weight", 0..128, 0..6144, 6144),
                ("self_attn.index_k_proj.weight_scale_inv", 0..128, 0..192, 192),
                ("self_attn.o_proj.weight", 0..6144, rank*4096..(rank+1)*4096, 8192),
                ("self_attn.o_proj.weight_scale_inv", 0..6144, rank*128..(rank+1)*128, 256),
                ("mlp.gate_proj.weight", rank*6144..(rank+1)*6144, 0..6144, 6144),
                ("mlp.down_proj.weight_scale_inv", 0..6144, rank*192..(rank+1)*192, 384),
                ("input_layernorm.weight", 0..1, 0..12288, 12288),
                ("self_attn.q_norm.weight", 0..1, 0..256, 256)] {
                let s = &h.slices[&format!("{TEXT}layers.0.{stem}")];
                assert_eq!((&s.rows, &s.columns, s.row_bytes), (&rows, &columns, row_bytes), "{stem}");
            }
            for (stem, rows, columns) in [
                ("shared_experts.gate_proj.weight", rank*1536..(rank+1)*1536, 0..6144),
                ("shared_experts.down_proj.weight_scale_inv", 0..6144, rank*48..(rank+1)*48),
                ("gate.weight", 0..4, 0..6144*4),
                ("e_score_correction_bias", 0..1, 0..16)] {
                let s = &h.slices[&format!("{TEXT}layers.3.block_sparse_moe.{stem}")];
                assert_eq!((&s.rows, &s.columns), (&rows, &columns), "{stem}");
            }
            for p in &h.target.projections {
                assert_eq!(h.slices[&format!("{}.weight",p.stem)].bytes(),p.input*p.output);
                assert_eq!(h.slices[&format!("{}.weight_scale_inv",p.stem)].bytes(),p.input/32*p.output);
            }
            assert_eq!(h.target.arena.view(&format!("{TEXT}embed_tokens.weight")).is_some(),
                rank == 0 && embedding == EmbeddingHome::Device(0));
            assert_eq!(h.target.arena.view("language_model.lm_head.weight").is_some(),rank == 1);
        }
    }
    Ok(())
}
