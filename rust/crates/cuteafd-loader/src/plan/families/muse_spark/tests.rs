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

/// [`with_layout`] with the coordinator EXL3 package installed beside the
/// manifest (onboarding available, its executor workspace charged).
fn with_package(rtx: usize, ranks: usize, dir: &std::path::Path) -> PlanOptions {
    let mut options = with_layout(rtx, ranks);
    options.layout.as_mut().unwrap().workspace_manifest = Some(testing::write_muse_exl3_capacities(dir));
    options
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
fn rotated_tp_ranks_balance_the_routed_bytes() {
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
        eprintln!("muse TP{ranks} expert GiB/rank {:?}; onboard {:?}; pool {}", spark.iter()
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
    let package = tempfile::tempdir().unwrap();
    for (rtx, ranks) in worlds.iter().map(|&ranks| (column, ranks)) {
        let report = input.plan(&with_package(rtx, ranks, package.path())).unwrap();
        eprintln!("muse {rtx} RTX + {ranks} Sparks: fits {} onboard {:?} pool {:?} hints {:?}", report.fits,
            report.onboard_layers, report.memory_layout.as_ref().map(|m| m.pool_tokens),
            report.hints.iter().map(|h| &h.what).collect::<Vec<_>>());
        assert!(!report.hints.iter().any(|h| h.what.contains("no muse:exl3-k45 layout")), "{rtx} RTX + {ranks}: {:?}",
            report.hints.iter().map(|h| &h.what).collect::<Vec<_>>());
        if ranks >= 4 { assert!(report.fits, "{rtx} RTX + {ranks} must fit (minimum 4): {:?}", report.hints); }
        if matches!((rtx, ranks), (1, 4) | (2, 6)) {
            let memory = report.memory_layout.as_ref().unwrap();
            let spark = memory.devices.iter().find(|device|
                device.kind == cuteafd_core::memory_layout::DeviceKind::Spark).unwrap();
            let owners: u64 = spark.items.iter().filter(|item|
                matches!(item.group.as_str(), "execution owners" | "execution staging"))
                .map(|item| item.bytes).sum();
            eprintln!("muse census {rtx}+{ranks}: owners={owners} free={} pool={} onboard={:?}",
                spark.free_bytes(), memory.pool_tokens, report.onboard_layers);
            let cfg = crate::families::muse_spark::MuseSparkConfig::from_hf(&testing::muse_spark_config()).unwrap();
            let mark_bytes = crate::placement::families::muse_spark::mark_bytes(&cfg, 1);
            let budget = layout::LayoutOptions::default().mimo_prefix_mark_bytes;
            let marks = budget / mark_bytes;
            let devices: Vec<_> = memory.devices.iter().filter(|device|
                device.kind == cuteafd_core::memory_layout::DeviceKind::Rtx).collect();
            let prefix_bytes: u64 = devices.iter().flat_map(|device| &device.items).filter(|item|
                item.category == cuteafd_core::memory_layout::Category::Prefix).map(|item| item.bytes).sum();
            assert_eq!(prefix_bytes, marks * mark_bytes, "host-off marks obey the aggregate byte budget");
            assert!(memory.notes.iter().any(|note|
                note.contains(&format!("host prefix tier off: retains {marks} exact prefix marks"))));
            let record_bytes = (cfg.kv_record_bytes() * cfg.full_layers()) as u64 / rtx as u64;
            let page_rows = crate::families::muse_spark::cache::PAGE_ROWS;
            let pool_from_ledger = devices.iter().map(|device| {
                let fixed: u64 = device.items.iter().filter(|item| item.group != "records").map(|item| item.bytes).sum();
                let context = crate::placement::ArchContext::for_device("sm_120", device.capacity_bytes).context_bytes;
                (device.capacity_bytes - context - fixed) / (record_bytes * page_rows) * page_rows
            }).min().unwrap().min(cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS);
            assert_eq!(memory.pool_tokens, pool_from_ledger, "pool first, derived from the exact ledger");
            if rtx == 2 { assert_eq!(memory.pool_tokens, 2 << 20); }
        }
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

/// Onboarding needs the coordinator EXL3 package: without it automatic
/// placement keeps every layer on the Sparks and an explicit count is refused;
/// with it, a pair's TP2 workspace charges the package's executor workspace
/// beside the partial planes.
#[test]
fn onboarding_follows_the_installed_package() {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let mut auto = with_layout(2, 6);
    auto.layout.as_mut().unwrap().host_cache_bytes = None;
    // An image whose manifest lacks the package beside it: the automatic placement
    // onboards (memory says it fits), so the plan refuses rather than going Spark-only.
    let broken = tempfile::tempdir().unwrap();
    let manifest = broken.path().join("PROGRAMS.json");
    std::fs::write(&manifest, b"{}").unwrap();
    let mut missing = auto.clone();
    missing.layout.as_mut().unwrap().workspace_manifest = Some(manifest);
    let refused = input.plan(&missing).unwrap();
    assert!(refused.solver_placement.is_none(), "a missing package must refuse, not degrade");
    assert!(refused.hints.iter().any(|h| h.what.contains("image lacks exl3-muse-k45/rtx-tp2; rebuild with the Muse \
        coordinator scope")), "{:?}", refused.hints);
    let mut explicit = missing.clone();
    explicit.layout.as_mut().unwrap().onboard = Some(crate::placement::Onboard::Layers(2));
    let refused = input.plan(&explicit).unwrap();
    assert!(refused.hints.iter().any(|h| h.what.contains("image lacks exl3-muse-k45/rtx-tp2")), "{:?}", refused.hints);
    // Explicit 0 is a labelled measurement arm; it needs no package.
    let mut zero = missing.clone();
    zero.layout.as_mut().unwrap().onboard = Some(crate::placement::Onboard::Layers(0));
    assert_eq!(input.plan(&zero).unwrap().onboard_layers, Some(0));
    // Without any image (a bare plan) onboarding is planned as the default.
    let report = input.plan(&auto).unwrap();
    assert!(report.onboard_layers.is_some_and(|n| n > 0), "{:?}", report.hints);
    let package = tempfile::tempdir().unwrap();
    let manifest = testing::write_muse_exl3_capacities(package.path());
    let mut installed = auto.clone();
    installed.layout.as_mut().unwrap().workspace_manifest = Some(manifest.clone());
    let report = input.plan(&installed).unwrap();
    assert!(report.onboard_layers.is_some_and(|n| n > 0), "{:?}", report.hints);
    let catalog = input.expert_catalog().unwrap();
    let (_, workspace) = crate::placement::families::muse_spark::local_package(catalog, &manifest, 2, 2752).unwrap();
    let placed = report.solver_placement.as_ref().unwrap();
    let cfg = crate::families::muse_spark::MuseSparkConfig::from_hf(&testing::muse_spark_config()).unwrap();
    let partials = crate::placement::families::muse_spark::tp2_partial_bytes(&cfg, 2752) as u64;
    let tp2 = placed.tp2.as_ref().unwrap();
    assert!(tp2.peak_bytes.iter().all(|&bytes| bytes >= partials + workspace), "{:?} vs {}", tp2.peak_bytes, partials + workspace);
}

/// Automatic placement lands on 0 RTX layers only when memory says 0: with
/// no room for an onboarded layer, an image without the package still plans
/// (all-Spark), because the package was never the reason.
#[test]
fn missing_package_is_harmless_only_when_memory_onboards_nothing() {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let broken = tempfile::tempdir().unwrap();
    let manifest = broken.path().join("PROGRAMS.json");
    std::fs::write(&manifest, b"{}").unwrap();
    // One RTX at the default automatic context: the pool takes the GPU (onboarding 0).
    let mut one = with_layout(1, 6);
    one.layout.as_mut().unwrap().workspace_manifest = Some(manifest);
    let unpackaged = input.plan(&one).unwrap();
    let bare = input.plan(&with_layout(1, 6)).unwrap();
    assert_eq!(bare.onboard_layers, Some(0), "{:?}", bare.hints);
    assert_eq!(unpackaged.onboard_layers, Some(0), "{:?}", unpackaged.hints);
    // It plans (the manifest only adds the image's loaded-code demand).
    assert!(unpackaged.solver_placement.as_ref().is_some_and(|p| p.pool_tokens > 0), "{:?}", unpackaged.hints);
}

/// The DFlash2 drafter is charged on the head GPU (it borrows the target head
/// there): its shared DFlash2 inventory plus the head scratch its target-head
/// launch owns. A drafter whose head softcap or multiplier differs from the
/// target's is refused by name (its logits would not be the target head's).
#[test]
fn drafter_is_charged_on_the_head_gpu_and_must_match_the_target_head() {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let radixark = include_str!("../../../../../cuteafd-daemon/src/families/muse_spark/fixtures/drafters/radixark.json");
    let drafter = tempfile::tempdir().unwrap();
    std::fs::write(drafter.path().join("config.json"), radixark).unwrap();
    let base = with_layout(2, 6);
    let mut with_draft = base.clone();
    with_draft.layout.as_mut().unwrap().muse_drafter_snapshot = Some(drafter.path().to_path_buf());
    let plain = input.plan(&base).unwrap();
    let drafted = input.plan(&with_draft).unwrap();
    let placed = drafted.solver_placement.as_ref().unwrap_or_else(|| panic!("{:?}", drafted.hints));
    let group = |gpu: usize| placed.items[gpu].iter().filter(|item| item.group == "Drafter").map(|item| item.bytes).sum::<u64>();
    assert_eq!(group(0), 0, "the drafter lives beside the head (GPU1)");
    let cfg = crate::families::muse_spark::MuseSparkConfig::from_hf(&testing::muse_spark_config()).unwrap();
    let config: serde_json::Value = serde_json::from_str(radixark).unwrap();
    let sms = crate::placement::ArchContext::for_device("sm_120", crate::placement::inventory::PRO_TOTAL_BYTES).sms;
    let expected = crate::placement::families::muse_spark::drafter_inventory(&cfg, &config, 16, 16, 202_112,
        u64::from(sms), crate::families::qwen4::Qwen4KvCache::Fp8).unwrap();
    assert_eq!(group(1), expected.owned + expected.scratch + expected.head_scratch + expected.runtime);
    assert_eq!(expected.runtime, (16 * 8 * 4 + 62 * 64 * 8 * 4) as u64, "draft ids + probe ring");
    assert_eq!(expected.head_scratch, (16 * 8 * (202_112 + 202_048) * 4) as u64);
    assert!(placed.pool_tokens <= plain.solver_placement.as_ref().unwrap().pool_tokens);
    // One RTX cannot reach the pool cap: the exact drafter charge must displace KV pages.
    let single = with_layout(1, 6);
    let mut single_draft = single.clone();
    single_draft.layout.as_mut().unwrap().muse_drafter_snapshot = Some(drafter.path().to_path_buf());
    let single_plain = input.plan(&single).unwrap();
    let single_drafted = input.plan(&single_draft).unwrap();
    assert!(single_plain.fits && single_drafted.fits, "{:?}", single_drafted.hints);
    let memory = single_plain.memory_layout.as_ref().unwrap();
    let device = memory.devices.iter().find(|device|
        device.kind == cuteafd_core::memory_layout::DeviceKind::Rtx).unwrap();
    let charged = expected.owned + expected.scratch + expected.head_scratch + expected.runtime;
    let draft_device = single_drafted.memory_layout.as_ref().unwrap().devices.iter().find(|device|
        device.kind == cuteafd_core::memory_layout::DeviceKind::Rtx).unwrap();
    assert_eq!(draft_device.items.iter().filter(|item| item.group == "Drafter").map(|item| item.bytes).sum::<u64>(), charged);
    let fixed = |device: &cuteafd_core::memory_layout::DeviceLayout| device.items.iter()
        .filter(|item| item.group != "records").map(|item| item.bytes).sum::<u64>();
    assert_eq!(fixed(draft_device) - fixed(device), charged);
    let page_rows = crate::families::muse_spark::cache::PAGE_ROWS;
    let page_bytes = (cfg.kv_record_bytes() * cfg.full_layers()) as u64 * page_rows;
    let context = crate::placement::ArchContext::for_device("sm_120", device.capacity_bytes).context_bytes;
    let available = device.capacity_bytes - context - fixed(device);
    assert!(memory.pool_tokens < cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS);
    assert_eq!(memory.pool_tokens, available / page_bytes * page_rows);
    let displaced_pages = available / page_bytes - (available - charged) / page_bytes;
    assert!(displaced_pages > 0);
    assert_eq!(memory.pool_tokens - single_drafted.memory_layout.as_ref().unwrap().pool_tokens,
        displaced_pages * page_rows, "drafter bytes displace exactly the whole-page KV difference");
    // A drafter trained against another head refuses before placement.
    let mut other: serde_json::Value = config.clone();
    other["dflash_config"]["output_multiplier"] = serde_json::json!(0.125);
    std::fs::write(drafter.path().join("config.json"), other.to_string()).unwrap();
    let refused = input.plan(&with_draft).unwrap();
    assert!(refused.hints.iter().any(|h| h.what.contains("softcap/output_multiplier differ")), "{:?}", refused.hints);
}
