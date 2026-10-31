use crate::plan::{self, testing, *};

const GIB: u64 = 1 << 30;

/// Header-only EXL3 K4 + Core-K6 fixture. Every layer's experts are written so
/// the Spark ledger is exact; the payloads are holes.
fn snapshot(expert_layers: &[usize]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    testing::write_muse_spark_snapshot(dir.path(), expert_layers);
    dir
}

fn all_layers() -> Vec<usize> { (0..62).collect() }

fn with_layout(rtx: usize, ranks: usize) -> PlanOptions {
    PlanOptions { placement: ExpertPlacement::from_spark_ranks(ranks),
        layout: Some(layout::LayoutOptions { host_memory: Some((256 << 30, 256 << 30)),
            rtx_bytes: vec![crate::placement::inventory::PRO_TOTAL_BYTES; rtx], head_split: rtx == 2,
            host_cache_bytes: Some(0), ..Default::default() }), ..Default::default() }
}

#[test]
fn detects_the_checkpoint_and_classifies_every_text_tensor() {
    let dir = snapshot(&[0]);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let checkpoint = input.checkpoint();
    let family = plan::family::detect(checkpoint).unwrap();
    assert_eq!(family.id(), "muse_spark");
    let model = family.open(checkpoint).unwrap();
    let spec = model.spec();
    assert_eq!((spec.hidden, spec.vocab, spec.layers.len()), (8192, 202048, 62));
    assert!(matches!(spec.layers[1].attention, AttentionKind::Gqa { heads: 128, kv_heads: 16, head_dim: 64 }));
    assert!(spec.layers[1].rope.is_none(), "full layers are NoPE");
    assert!(matches!(spec.layers[0].attention, AttentionKind::SlidingGqa { window: 2048, .. }));
    assert_eq!(spec.layers[0].rope.unwrap().theta, 500000.0);
    for (name, component) in [
        ("model.language_model.layers.3.mlp.experts.17.down_proj.trellis", Component::RoutedExpert),
        ("model.language_model.layers.3.mlp.experts.post_expert_norm.weight", Component::RoutedExpert),
        ("model.language_model.layers.3.mlp.pre_expert_proj.trellis", Component::LatentProjection),
        ("model.language_model.layers.3.mlp.gate.e_score_correction_bias", Component::Router),
        ("model.language_model.layers.3.self_attn.gate_proj.weight", Component::Attention),
        ("model.language_model.layers.3.post_feedforward_residual_gate.gate", Component::ResidualMix),
        ("model.language_model.layers.3.post_feedforward_layernorm.weight", Component::Norm),
        ("model.vision_projection.weight", Component::Vision),
        ("model.audio_tower.proj.weight", Component::Audio),
        ("lm_head.trellis", Component::LmHead),
    ] {
        assert_eq!(family.classify(spec, name).map(|role| role.component), Some(component), "{name}");
    }
    let expert = family.classify(spec, "model.language_model.layers.3.mlp.experts.17.down_proj.trellis").unwrap();
    assert_eq!((expert.layer, expert.expert), (Some(3), Some(17)));
}

#[test]
fn header_inventory_accepts_k6_core_and_k4_experts_and_reports_bytes() {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let report = input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(6), ..Default::default() }).unwrap();
    assert_eq!(report.family.as_deref(), Some("muse_spark"));
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    for component in report.components.iter().filter(|c| c.status != Status::Disabled) {
        assert_eq!(component.rejected, 0, "{:?}: {:?}", component.component, component.rejections);
    }
    assert!(report.components.iter().filter(|c| matches!(c.component, Component::Vision | Component::Audio))
        .all(|c| c.status == Status::Disabled), "media stay off");
    assert!(!report.hints.iter().any(|h| h.what.contains("EXL3 publication")), "{:?}", report.hints);
    let routed = report.components.iter().find(|c| c.component == Component::RoutedExpert).unwrap();
    // 62 x 256 x 3 projections of 4096 x 4096 at 4 bits, plus vectors, norms and markers.
    let trellis = 62u64 * 256 * 3 * 4096 * 4096 / 2;
    assert!(routed.bytes > trellis && routed.bytes < trellis + 2 * GIB, "{}", routed.bytes);
    assert!(routed.formats.contains_key("exl3-k4"), "{:?}", routed.formats);
    let experts = report.experts.as_ref().unwrap();
    assert_eq!(experts.package, super::PACKAGE);
    // The K6 head: stored at 202,112 rows (N128), logical 202,048.
    let checkpoint = input.checkpoint();
    let model = crate::families::muse_spark::core::CoreInventory::from_checkpoint(
        &crate::families::muse_spark::MuseSparkConfig::from_hf(&checkpoint.config).unwrap(), checkpoint).unwrap();
    let head = model.get_weight("lm_head").unwrap();
    assert_eq!((head.shape, head.stored_outputs), ([202048, 8192], 202112));
    let o = model.get_weight("model.language_model.layers.0.self_attn.o_proj").unwrap();
    assert_eq!(o.split, crate::families::muse_spark::core::CoreSplit::Input, "o_proj splits its input on TP2");
    assert!(experts.spark_worlds.contains(&4) && experts.spark_worlds.contains(&6));
}

#[test]
fn whole_expert_ranks_split_the_routed_bytes_evenly() {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    for ranks in [4, 6] {
        let report = input.plan(&with_layout(2, ranks)).unwrap();
        let memory = report.memory_layout.as_ref().unwrap();
        let spark: Vec<u64> = memory.devices.iter().filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Spark)
            .map(|d| d.items.iter().filter(|i| i.category == cuteafd_core::memory_layout::Category::Experts)
                .map(|i| i.bytes).sum()).collect();
        assert_eq!(spark.len(), ranks, "{:?}", report.hints);
        let (max, min) = (*spark.iter().max().unwrap(), *spark.iter().min().unwrap());
        // Staggered ownership: ranks differ by at most the rotated remainder.
        assert!(max - min < max / 20, "TP{ranks} expert bytes {spark:?}");
        eprintln!("muse EP{ranks} expert GiB/rank {:?}; onboard {:?}; pool {}", spark.iter()
            .map(|b| format!("{:.1}", *b as f64 / GIB as f64)).collect::<Vec<_>>(), report.onboard_layers, memory.pool_tokens);
    }
}

#[test]
fn layout_picks_its_spark_range_and_onboards_by_default() { plan_matrix(2); }

#[test]
fn layout_plans_every_spark_world_on_one_rtx() { plan_matrix(1); }

/// The plan matrix (#923) of one coordinator column: every Spark world the
/// shared range data gives the width; each packages (no 'no layout' hint),
/// and the worlds at or above the minimum fit.
fn plan_matrix(column: usize) {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let worlds = crate::plan::experts::spark_worlds(super::PACKAGE, 4096);
    assert_eq!(worlds, [2, 3, 4, 5, 6, 7, 8], "range data for muse:exl3-k45");
    for (rtx, ranks) in worlds.iter().map(|&ranks| (column, ranks)) {
        let report = input.plan(&with_layout(rtx, ranks)).unwrap();
        eprintln!("muse {rtx} RTX + {ranks} Sparks: fits {} onboard {:?} pool {:?} hints {:?}", report.fits,
            report.onboard_layers, report.memory_layout.as_ref().map(|m| m.pool_tokens),
            report.hints.iter().map(|h| &h.what).collect::<Vec<_>>());
        assert!(!report.hints.iter().any(|h| h.what.contains("no muse:exl3-k45 layout")), "{rtx} RTX + {ranks}: {:?}",
            report.hints.iter().map(|h| &h.what).collect::<Vec<_>>());
        if ranks >= 4 { assert!(report.fits, "{rtx} RTX + {ranks} must fit (minimum 4): {:?}", report.hints); }
        if (rtx, ranks) == (2, 6) {
            assert!(report.fits && report.placement_supported, "{:?}", report.hints);
            let memory = report.memory_layout.as_ref().unwrap();
            assert!(memory.pool_tokens >= 1 << 20, "pool {}", memory.pool_tokens);
            // Onboarding stays automatic; pool first decides the count.
            assert!(report.onboard_layers.is_some());
            assert!(report.solver_placement.as_ref().unwrap().max_pool_tokens_no_onboard >= memory.pool_tokens);
        }
    }
    // Planned family: the inventory is not a serving claim.
    let report = input.plan(&with_layout(column, 6)).unwrap();
    assert!(!report.executable());
}

/// Each Spark's rings carry the shared Muse endpoint: the worst partial reply
/// per frame (one SEND), depth = prefill waves + the decode step, exactly the
/// bytes the endpoint seals; the frame stays under the NIC message limit.
#[test]
fn spark_rings_carry_the_worst_partial_reply_frame() {
    use crate::families::muse_spark::replies::{max_rank_bytes, Cutoff, WideFormat};
    use crate::placement::families::muse_spark::{spark_endpoint, NIC_MAX_MESSAGE_BYTES};
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let report = input.plan(&with_layout(2, 6)).unwrap();
    let memory = report.memory_layout.as_ref().unwrap();
    let spark = memory.devices.iter().find(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Spark).unwrap();
    let rings = spark.items.iter().find(|item| item.group == "rdma rings").unwrap().bytes;
    let prefill = crate::serving_capacity::prefill_waves::PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 };
    let (endpoint, base) = spark_endpoint(prefill, Cutoff::new(64, 1).unwrap(), 16).unwrap();
    let geometry = cuteafd_core::ExpertGeometry::MUSE_SPARK;
    assert_eq!(rings, endpoint.ring_bytes(geometry, base).unwrap() as u64, "planner charges the sealed endpoint");
    let reply = max_rank_bytes(Cutoff::new(64, 1).unwrap(), 2752, 16, 8, 4096);
    assert_eq!(endpoint.partial_reply_bytes, Some(reply as u64));
    assert_eq!(endpoint.ring_depth, Some(4));
    let frame = endpoint.frame_bytes(geometry, base).unwrap();
    assert!(frame > 64 << 20 && frame < 100 << 20, "NVFP4 worst mix at 16 sequences: {frame}");
    // FP8 above the cutoff: the widest release mix still fits one RC message.
    let fp8 = spark_endpoint(prefill, Cutoff::with_wide(64, 1, WideFormat::Fp8).unwrap(), 16).unwrap().0;
    let fp8_frame = fp8.frame_bytes(geometry, base).unwrap();
    assert!(fp8_frame > frame && fp8_frame < NIC_MAX_MESSAGE_BYTES, "{fp8_frame}");
    eprintln!("muse Spark endpoint: NVFP4 frame {:.1} MiB, FP8 {:.1} MiB, rings {:.0} MiB per Spark",
        frame as f64 / 1048576.0, fp8_frame as f64 / 1048576.0, rings as f64 / 1048576.0);
}

#[test]
fn fused_bf16_experts_name_the_publication_to_serve() {
    // The BF16 release fuses each layer's experts into [256, ...] stacks.
    let dir = tempfile::tempdir().unwrap();
    let mut config = testing::muse_spark_config();
    config.as_object_mut().unwrap().remove("quantization_config");
    let (mut tensors, _) = testing::muse_spark_tensors(&[]);
    tensors.retain(|t| !t.0.ends_with(".trellis") && !t.0.ends_with(".suh") && !t.0.ends_with(".svh") && !t.0.ends_with(".mcg"));
    for layer in 0..62 {
        let p = format!("model.language_model.layers.{layer}");
        for (name, shape) in [("self_attn.q_proj", [8192, 8192]), ("self_attn.k_proj", [1024, 8192]),
            ("self_attn.v_proj", [1024, 8192]), ("self_attn.o_proj", [8192, 8192]),
            ("mlp.pre_expert_proj", [4096, 8192]), ("mlp.post_expert_proj", [8192, 4096])] {
            tensors.push(testing::t(format!("{p}.{name}.weight"), "BF16", &shape));
        }
    }
    tensors.push(testing::t("lm_head.weight", "BF16", &[202048, 8192]));
    let p = "model.language_model.layers.0";
    tensors.push(testing::t(format!("{p}.mlp.experts.gate_up_proj"), "BF16", &[256, 8192, 4096]));
    tensors.push(testing::t(format!("{p}.mlp.experts.down_proj"), "BF16", &[256, 4096, 4096]));
    testing::write_snapshot(dir.path(), &config, &tensors, None);
    let report = testing::plan(dir.path(), &PlanOptions::default()).unwrap();
    let routed = report.components.iter().find(|c| c.component == Component::RoutedExpert).unwrap();
    assert!(routed.rejected > 0);
    assert!(report.hints.iter().any(|h| h.what.contains("EXL3 publication")), "{:?}", report.hints);
}
