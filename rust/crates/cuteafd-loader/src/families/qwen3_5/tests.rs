use super::*;
use crate::plan::{self, testing::{t, write_snapshot, Tensor}, ExpertPlacement, PlanOptions, RuntimeStatus};
use crate::families::qwen::{GdnOutputGate, HybridAttention};
use serde_json::{json, Value};

fn config(hidden: usize, intermediate: usize, heads: usize, values: usize, layers: usize) -> Value {
    json!({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": {
        "model_type": "qwen3_5_text", "hidden_size": hidden, "intermediate_size": intermediate,
        "num_hidden_layers": layers, "vocab_size": 64, "num_attention_heads": heads, "num_key_value_heads": 4,
        "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": values,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "layer_types": (0..layers).map(|l| if l % 4 == 3 { "full_attention" } else { "linear_attention" }).collect::<Vec<_>>(),
        "full_attention_interval": 4, "hidden_act": "silu", "attention_bias": false, "attn_output_gate": true,
        "output_gate_type": "swish", "mamba_ssm_dtype": "float32", "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144, "mtp_num_hidden_layers": 0, "tie_word_embeddings": false,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10], "mrope_interleaved": true}}})
}

fn tensors(fp8: bool) -> Vec<Tensor> {
    let mut out = vec![t("model.language_model.embed_tokens.weight", "BF16", &[64, 4096]),
        t("lm_head.weight", "BF16", &[64, 4096]), t("model.language_model.norm.weight", "BF16", &[4096])];
    for layer in 0..4 {
        let p = format!("model.language_model.layers.{layer}");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            out.push(t(format!("{p}.{name}.weight"), "BF16", &[4096]));
        }
        let mut projection = |name: &str, n: usize, k: usize| {
            out.push(t(format!("{p}.{name}.weight"), if fp8 { "F8_E4M3" } else { "BF16" }, &[n, k]));
            if fp8 { out.push(t(format!("{p}.{name}.weight_scale_inv"), "BF16", &[n.div_ceil(128), k.div_ceil(128)])); }
        };
        projection("mlp.gate_proj", 12288, 4096);
        projection("mlp.up_proj", 12288, 4096);
        projection("mlp.down_proj", 4096, 12288);
        if layer == 3 {
            projection("self_attn.q_proj", 8192, 4096);
            projection("self_attn.k_proj", 1024, 4096);
            projection("self_attn.v_proj", 1024, 4096);
            projection("self_attn.o_proj", 4096, 4096);
            for name in ["q_norm", "k_norm"] { out.push(t(format!("{p}.self_attn.{name}.weight"), "BF16", &[256])); }
        } else {
            projection("linear_attn.in_proj_qkv", 8192, 4096);
            projection("linear_attn.in_proj_z", 4096, 4096);
            projection("linear_attn.out_proj", 4096, 4096);
            for name in ["in_proj_a", "in_proj_b"] { out.push(t(format!("{p}.linear_attn.{name}.weight"), "BF16", &[32, 4096])); }
            out.push(t(format!("{p}.linear_attn.conv1d.weight"), "BF16", &[8192, 1, 4]));
            out.push(t(format!("{p}.linear_attn.A_log"), "F32", &[32]));
            out.push(t(format!("{p}.linear_attn.dt_bias"), "BF16", &[32]));
            out.push(t(format!("{p}.linear_attn.norm.weight"), "F32", &[128]));
        }
    }
    out
}

fn native_planner_fixture() -> (tempfile::TempDir, PlanOptions, crate::placement::families::qwen3_5::NativeBuffers, Value) {
    use crate::placement::families::qwen3_5::NativeBuffers;
    let dir = tempfile::tempdir().unwrap();
    let raw = config(4096, 12288, 16, 32, 4);
    let cfg = Qwen35Config::from_hf(&raw).unwrap();
    write_snapshot(dir.path(), &raw, &tensors(false), None);
    let buffers = NativeBuffers::new(&cfg, 1024, 4, 10).unwrap();
    let mut programs = Vec::new();
    for (op, rows) in [("norm", 0), ("add", 0), ("gdn_commit", 0)].into_iter().chain(
        [64, 4096].into_iter().flat_map(|rows| ["gdn", "attn_producer", "full_gqa", "attn_o", "ffn"].map(|op| (op, rows)))) {
        let mut pointers = match op {
            "norm" => vec!["x", "weight", "out"], "add" => vec!["a", "b", "out"],
            "gdn_commit" => vec!["state", "conv_state", "replay", "tables"],
            "gdn" => vec!["x", "w_in", "conv_w", "a_log", "dt_bias", "norm_w", "w_out", "conv_state", "state", "slots", "seq_first", "out"],
            "attn_producer" => vec!["x", "w_in", "q_norm", "k_norm", "rope_positions", "kv_slots", "kv_cache", "query", "gate", "scratch"],
            "full_gqa" => vec!["query", "kv_cache", "positions", "page_table", "out", "scratch"],
            "attn_o" => vec!["attn", "gate", "w_o", "out", "scratch"],
            "ffn" => vec!["x", "w_gate_up", "w_down", "out", "scratch"], _ => unreachable!(),
        };
        if op == "gdn" { if rows == 64 { pointers.push("replay"); } pointers.push("scratch"); }
        let scalars = match op { "gdn_commit" | "full_gqa" => 3, "gdn" if rows == 64 => 2, _ => 1 };
        let name = if rows == 0 { format!("qwen35_{op}") } else { format!("qwen35_{op}_m{rows}") };
        let in_width = if op == "gdn" { 12352 } else { 10240 };
        programs.push(json!({"name":name, "capacity_rows":rows,
            "geometry":{"hidden":4096,"max_rows":rows,"eps":1e-6,"zero_centered":true,
                "key_heads":16,"value_heads":32,"head_dim":if op == "gdn" {128} else {256},
                "in_width":in_width,"output_gate":"silu","fp8_weights":false,"norm_dtype":"torch.float32",
                "channels":8192,"replay_rows":64,"replay_bytes":buffers.replay_per_layer,
                "heads":16,"kv_heads":4,"record_bytes":4096,"indexed":false,"rope_dim":64,
                "rope_theta":10000000.0,"page_rows":64,"max_splits":32,"causal":true,
                "empty_rows_zero":true,"width":4096,"intermediate":12288},
            "pointers":pointers.iter().map(|name| json!({"name":name})).collect::<Vec<_>>(),
            "scalars":(0..scalars).map(|_| json!({"type":"int32"})).collect::<Vec<_>>(),
            "scratch_bytes_at_capacity":{"scratch":1024}}));
    }
    let manifest = dir.path().join("PROGRAMS.json");
    let exports = json!({"programs":programs});
    std::fs::write(&manifest, serde_json::to_vec(&exports).unwrap()).unwrap();
    let options = PlanOptions { placement: ExpertPlacement::Local, vision: plan::MediaMode::Off,
        audio: plan::MediaMode::Off, layout: Some(plan::layout::LayoutOptions {
            rtx_bytes: vec![96 << 30], concurrency: 4, prefix_slots: Some(10), native_mtp_layers: 0,
            pool_tokens: Some(262144), workspace_manifest: Some(manifest.clone()), ..Default::default() }),
        ..Default::default() };
    (dir, options, buffers, exports)
}

#[test]
fn native_planner_uses_export_contract_and_exact_dense_allocations() {
    let (dir, options, buffers, mut exports) = native_planner_fixture();
    let manifest = options.layout.as_ref().unwrap().workspace_manifest.as_ref().unwrap();
    let report = plan::plan(dir.path(), &options).unwrap();
    assert!(report.fits && report.placement_supported && report.native_serving_admitted);
    assert!(report.wip_launchable());
    assert!(!report.executable());
    let deployment_options = plan::deployment::DeploymentOptions { rtx_gpus: 1, ..Default::default() };
    let deployment = plan::deployment::DeploymentSpec::from_report(&report, &deployment_options).unwrap();
    assert!(deployment.wip_launchable && !deployment.executable);
    let memory = report.memory_layout.unwrap();
    assert_eq!(memory.pool_tokens, 262144, "{:?}", report.hints);
    assert_eq!(memory.devices.len(), 1);
    assert!(memory.devices[0].items.iter().any(|item| item.group == "exact retained prefix marks"
        && item.bytes == buffers.prefix as u64));
    assert!(memory.notes.iter().any(|note| note.contains("no captured graphs")));
    let spark_options = PlanOptions { placement: ExpertPlacement::Sparks { ranks: 2 }, ..options.clone() };
    let refused = plan::plan(dir.path(), &spark_options).unwrap();
    assert!(!refused.fits && !refused.placement_supported && !refused.native_serving_admitted
        && !refused.wip_launchable());
    assert!(refused.hints.iter().any(|hint| hint.what.contains("no Spark layout")));
    assert!(refused.memory_layout.unwrap().devices.is_empty());
    exports["programs"][0]["geometry"]["zero_centered"] = json!(false);
    std::fs::write(&manifest, serde_json::to_vec(&exports).unwrap()).unwrap();
    let refused = plan::plan(dir.path(), &options).unwrap();
    assert!(!refused.fits && !refused.placement_supported && !refused.native_serving_admitted
        && !refused.wip_launchable());
    assert!(refused.hints.iter().any(|hint| hint.what.contains("zero-centered")));
    assert!(refused.memory_layout.unwrap().devices.is_empty());
    exports["programs"][0]["geometry"]["zero_centered"] = json!(true);
    exports["programs"][0]["pointers"][0]["name"] = json!("wrong_input");
    std::fs::write(&manifest, serde_json::to_vec(&exports).unwrap()).unwrap();
    let refused = plan::plan(dir.path(), &options).unwrap();
    assert!(!refused.fits && !refused.placement_supported && !refused.native_serving_admitted
        && !refused.wip_launchable());
    assert!(refused.hints.iter().any(|hint| hint.what.contains("pointer order")));
}

#[test]
fn dense_default_prefix_geometry_releases_retention_with_host() {
    use cuteafd_core::prefix::{mark_slots_for, mark_slots_with_host};
    use crate::placement::families::qwen3_5::NativeBuffers;
    for (name, hidden, intermediate, heads, values, layers, expected_mark) in [
        ("Qwen3.5-9B", 4096, 12288, 16, 32, 32, 51_511_296),
        ("Qwen3.8-27B", 5120, 17408, 24, 48, 64, 153_944_064),
    ] {
        let cfg = Qwen35Config::from_hf(&config(hidden, intermediate, heads, values, layers)).unwrap();
        let mark = NativeBuffers::new(&cfg, 0, 16, 0).unwrap().mark_bytes as u64;
        assert_eq!(mark, expected_mark);
        let old = mark_slots_for(16, 20, mark, 2 << 30);
        let staged = mark_slots_with_host(16, 20, mark, 2 << 30, true);
        assert_eq!((old, staged), (if layers == 32 { 41 } else { 34 }, 2));
        println!("{name}: C16 entries=20 mark_bytes={mark} slots={old}->{staged} arena_GiB={:.6}->{:.6} freed_GiB={:.6}",
            old as f64 * mark as f64 / (1u64 << 30) as f64,
            staged as f64 * mark as f64 / (1u64 << 30) as f64,
            (old - staged) as f64 * mark as f64 / (1u64 << 30) as f64);
    }
}

#[test]
fn native_planner_prefix_marks_follow_shared_host_rule() {
    use cuteafd_core::{memory_layout::Category, prefix::mark_slots_with_host};
    let (dir, mut options, buffers, _) = native_planner_fixture();
    for host in [false, true] {
        for entries in [0, 1, 20] {
            for lanes in [1, 16, 64] {
                let layout = options.layout.as_mut().unwrap();
                layout.prefix_slots = None;
                layout.concurrency = lanes;
                layout.host_cache_bytes = if host { None } else { Some(0) };
                layout.mimo_prefix_entries = entries;
                let expected = mark_slots_with_host(lanes, entries, buffers.mark_bytes as u64,
                    layout.mimo_prefix_mark_bytes, host);
                let report = plan::plan(dir.path(), &options).unwrap();
                let memory = report.memory_layout.unwrap();
                let prefix = memory.devices[0].by_category().get(&Category::Prefix).copied().unwrap_or(0);
                assert_eq!(prefix, expected * buffers.mark_bytes as u64, "host={host}, entries={entries}, lanes={lanes}");
            }
        }
    }
    // Explicit arena overrides are unchanged by the automatic host-aware sizing.
    for host in [false, true] {
        let layout = options.layout.as_mut().unwrap();
        layout.host_cache_bytes = if host { None } else { Some(0) };
        layout.prefix_slots = Some(7);
        let memory = plan::plan(dir.path(), &options).unwrap().memory_layout.unwrap();
        assert_eq!(memory.devices[0].by_category()[&Category::Prefix], 7 * buffers.mark_bytes as u64);
    }
}

#[test]
fn native_text_admission_disables_automatic_bundled_vision() {
    use plan::{MediaMode, Status, spec::Component};
    let (dir, mut options, _, _) = native_planner_fixture();
    let mut raw = config(4096, 12288, 16, 32, 4);
    raw["vision_config"] = json!({"depth":27,"hidden_size":1152,"intermediate_size":4304,
        "num_heads":16,"out_hidden_size":4096,"patch_size":16,"temporal_patch_size":2,
        "spatial_merge_size":2,"num_position_embeddings":2304,"in_channels":3,
        "hidden_act":"gelu_pytorch_tanh","deepstack_visual_indexes":[]});
    let mut weights = tensors(false);
    let mut tower = |name: &str, shape: &[usize], bias: bool| {
        weights.push(t(format!("model.visual.{name}.weight"), "BF16", shape));
        if bias { weights.push(t(format!("model.visual.{name}.bias"), "BF16", &[shape[0]])); }
    };
    tower("patch_embed.proj", &[1152, 3, 2, 16, 16], true);
    tower("pos_embed", &[2304, 1152], false);
    tower("merger.norm", &[1152], true);
    tower("merger.linear_fc1", &[4608, 4608], true);
    tower("merger.linear_fc2", &[4096, 4608], true);
    for layer in 0..27 {
        for (name, shape) in [("norm1", vec![1152]), ("norm2", vec![1152]),
            ("attn.qkv", vec![3456, 1152]), ("attn.proj", vec![1152, 1152]),
            ("mlp.linear_fc1", vec![4304, 1152]), ("mlp.linear_fc2", vec![1152, 4304])] {
            tower(&format!("blocks.{layer}.{name}"), &shape, true);
        }
    }
    write_snapshot(dir.path(), &raw, &weights, None);
    options.vision = MediaMode::Auto;
    let report = plan::plan(dir.path(), &options).unwrap();
    assert_eq!(report.vision, MediaMode::Off);
    assert!(report.wip_launchable() && !report.executable(), "{:?}", report.hints);
    let vision = report.components.iter().find(|c| c.component == Component::Vision).unwrap();
    assert_eq!(vision.status, Status::Disabled);
    assert_eq!(vision.rejected, 0);
    assert_eq!(vision.bytes, 0);
    assert!(report.disabled_media_bytes > 0 && report.encoder.is_none());
    let off = plan::plan(dir.path(), &PlanOptions { vision: MediaMode::Off, ..options.clone() }).unwrap();
    assert_eq!(report.memory_layout.as_ref().unwrap().devices[0].used_bytes(),
        off.memory_layout.as_ref().unwrap().devices[0].used_bytes());
    assert_eq!(report.encoder_plan_hash, off.encoder_plan_hash);
    for vision in [MediaMode::Rtx(None), MediaMode::Spark(None)] {
        let refused = plan::plan(dir.path(), &PlanOptions { vision, ..options.clone() }).unwrap();
        assert!(!refused.wip_launchable() && !refused.native_serving_admitted);
        assert!(refused.hints.iter().any(|h| h.what.contains("media execution is not qualified")));
    }
}

#[test]
fn automatic_spark_count_retains_native_wip_zero_spark_candidate() {
    let (dir, options, _, _) = native_planner_fixture();
    let report = plan::plan_spark_count(dir.path(), &options, 2, None).unwrap();
    assert_eq!(report.spark_ranks, 0);
    assert!(report.native_serving_admitted && report.wip_launchable());
    assert!(!report.executable());
    assert!(report.spark_count_warning.as_ref().unwrap().contains("using 0 Sparks"));
    let explicit = plan::plan_spark_count(dir.path(), &options, 2, Some(2)).unwrap();
    assert!(!explicit.wip_launchable() && !explicit.executable());
    assert!(explicit.hints.iter().any(|h| h.what.contains("counts that work: [0]")));
}

#[test]
fn dense_geometry_and_native_gate_sites() {
    for (h, i, heads, values, layers, kv, state) in [(4096, 12288, 16, 32, 32, 32768, 51511296),
        (5120, 17408, 24, 48, 64, 65536, 153944064)] {
        let cfg = Qwen35Config::from_hf(&config(h, i, heads, values, layers)).unwrap();
        assert_eq!(cfg.gdn.output_gate, GdnOutputGate::Silu);
        assert_eq!(cfg.full_kv_bytes_per_token(), kv);
        assert_eq!(cfg.recurrent_state_bytes(), state);
        assert_eq!(cfg.attention[3], HybridAttention::Full);
    }
}

#[test]
fn malformed_config_is_not_a_flash_next_alias() {
    let base = config(4096, 12288, 16, 32, 4);
    for (field, value) in [("output_gate_type", json!("sigmoid")), ("hidden_act", json!("gelu")),
        ("num_key_value_heads", json!(3)), ("linear_conv_kernel_dim", json!(0)),
        ("linear_value_head_dim", json!(64)), ("attention_bias", json!(true)),
        ("mamba_ssm_dtype", json!("bfloat16")), ("layer_types", json!(["full_attention"])),
        ("mtp_num_hidden_layers", json!(-1)), ("mtp_num_hidden_layers", json!("1")),
        ("attn_output_gate", json!("true")), ("attention_dropout", json!("zero")),
        ("output_gate_type", json!(false)), ("mamba_ssm_dtype", json!(32)),
        ("tie_word_embeddings", json!("false")), ("mtp_use_dedicated_embeddings", json!(null))] {
        let mut bad = base.clone(); bad["text_config"][field] = value;
        assert!(Qwen35Config::from_hf(&bad).is_err(), "{field}");
    }
}

#[test]
fn bf16_and_native_block_fp8_headers_remain_planned_and_zero_spark() {
    for fp8 in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &config(4096, 12288, 16, 32, 4), &tensors(fp8), None);
        Qwen35Checkpoint::read(dir.path()).unwrap();
        let report = plan::plan(dir.path(), &PlanOptions { placement: ExpertPlacement::Local, ..Default::default() }).unwrap();
        assert_eq!(report.family.as_deref(), Some("qwen3_5"));
        assert_eq!(report.runtime, Some(RuntimeStatus::Planned));
        assert!(!report.fits && !report.placement_supported && !report.executable() && !report.wip_launchable());
        assert_eq!(report.spark_ranks, 0);
        assert!(report.experts.is_none() && report.spec.as_ref().unwrap().moe.is_none());
        assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
        assert!(report.components.iter().all(|c| c.rejected == 0), "{:?}", report.components);
    }
}

#[test]
fn missing_or_malformed_projection_is_named() {
    let cfg = config(4096, 12288, 16, 32, 4);
    let dir = tempfile::tempdir().unwrap();
    let mut weights = tensors(false);
    let missing = "model.language_model.layers.0.mlp.up_proj.weight";
    weights.retain(|(name, ..)| name != missing);
    write_snapshot(dir.path(), &cfg, &weights, None);
    assert!(format!("{:#}", Qwen35Checkpoint::read(dir.path()).err().unwrap()).contains(missing));
    let mut weights = tensors(false);
    weights.iter_mut().find(|(name, ..)| name.ends_with("self_attn.q_proj.weight")).unwrap().2 = vec![4096, 4096];
    write_snapshot(dir.path(), &cfg, &weights, None);
    assert!(format!("{:#}", Qwen35Checkpoint::read(dir.path()).err().unwrap()).contains("q_proj.weight"));
    let mut weights = tensors(true);
    weights.iter_mut().find(|(name, ..)| name.ends_with("linear_attn.in_proj_a.weight")).unwrap().1 = "F8_E4M3".into();
    write_snapshot(dir.path(), &cfg, &weights, None);
    assert!(format!("{:#}", Qwen35Checkpoint::read(dir.path()).err().unwrap()).contains("in_proj_a"));
}
