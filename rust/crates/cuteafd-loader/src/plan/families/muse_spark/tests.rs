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

/// The min card's declared serving context (examples/configs/muse-spark-exl3-min.config):
/// one RTX PRO 6000 cannot hold the checkpoint's compiled 1M pool together
/// with DFlash2 and the onboarding needed to keep both bundled towers on Sparks.
const MIN_CARD_CONTEXT: u64 = 634_304;

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
fn zero_spark_is_a_typed_planner_result_not_a_launcher_restriction() {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let report = input.plan(&with_package(2, 0, dir.path())).unwrap();
    assert!(!report.executable() && !report.wip_launchable());
    assert!(report.solver_placement.is_none());
    assert!(report.hints.iter().any(|hint| hint.what.contains("joint admission refused")), "{:?}", report.hints);
}

fn check_target_arena_admission(input: &testing::BudgetedPlanInput) {
    use crate::families::muse_spark::{core::CoreInventory,
        weight_layout::{coordinator_arena, GlobalLayout, LayerLayout}, MuseSparkConfig};
    use cuteafd_core::memory_layout::DeviceKind;
    let checkpoint = input.checkpoint();
    let cfg = MuseSparkConfig::from_hf(&checkpoint.config).unwrap();
    let core = CoreInventory::from_checkpoint(&cfg, checkpoint).unwrap();
    let global = GlobalLayout::new(&cfg, &core).unwrap();
    for gpus in [1, 2] {
        let report = input.plan(&with_layout(gpus, 6)).unwrap();
        let placement = report.solver_placement.as_ref().unwrap_or_else(|| panic!("{:?}", report.hints));
        let layouts = (0..cfg.layers).map(|layer| LayerLayout::ranks(&cfg, layer, gpus, &core).unwrap())
            .collect::<Vec<_>>();
        let devices = report.memory_layout.as_ref().unwrap().devices.iter()
            .filter(|device| device.kind == DeviceKind::Rtx).collect::<Vec<_>>();
        assert_eq!(devices.len(), gpus);
        for (gpu, device) in devices.iter().enumerate() {
            let local = layouts.iter().map(|ranks| &ranks[gpu]).collect::<Vec<_>>();
            let arena = coordinator_arena(gpu, gpus, &global, &local).unwrap();
            let bytes = arena.bytes() as u64;
            let tail = bytes.div_ceil(2 << 20) * (2 << 20) - bytes;
            let roots = device.items.iter().filter(|item| item.group == "weight arena").collect::<Vec<_>>();
            assert_eq!(roots.len(), 1, "one target root per physical GPU");
            assert_eq!(roots[0].bytes, bytes);
            assert_eq!(device.items.iter().filter(|item| item.group == "allocation granularity")
                .map(|item| item.bytes).sum::<u64>(), tail);
            assert!(!placement.items[gpu].iter().any(|item| matches!(item.group.as_str(),
                "token embedding" | "lm head (K6)" | "final norm" |
                "Muse native attention values and block scales") || item.group.starts_with("layer weights (")),
                "the old immutable tensor charge must not survive beside its arena");
        }
    }
}

#[test]
fn target_admission_replaces_tensor_charges_with_runtime_arenas() {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    check_target_arena_admission(&input);
}

#[test]
#[ignore = "reads the installed Muse checkpoint headers"]
fn installed_target_admission_matches_runtime_arenas() {
    let hub = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/mnt/sparknest/hf-home"));
    let snapshot = hub.join("hub/models--wrldsuksgo2mars--Muse-Spark-Open-EXL3-K4-Core-K6/snapshots/89998fb177acd9a862ac8bd4fb79bfd50e940c97");
    let input = testing::BudgetedPlanInput::open(&snapshot).unwrap();
    check_target_arena_admission(&input);
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
        ("model.language_model.layers.3.mlp.experts.post_expert_norm.weight", Component::Norm),
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

/// Check the coordinator's header view, not index membership: the latter also
/// contains the Spark-only payload and cannot catch ownership mistakes.
fn check_coordinator_startup_inventory(snapshot: &std::path::Path) -> anyhow::Result<Checkpoint> {
    let inventory = Checkpoint::inventory(snapshot)?;
    let model = super::Model::open(&inventory)?;
    let coordinator = Checkpoint::coordinator(snapshot, false, false)?;
    let headers = coordinator.tensors.iter().map(|t| t.meta.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    for name in model.tensors.keys() {
        if let Some(core) = model.core.get_weight(name) {
            for suffix in ["trellis", "suh", "svh", "mcg"] {
                let member = format!("{}.{suffix}", core.stem);
                anyhow::ensure!(headers.contains(member.as_str()), "coordinator startup lacks {member}");
            }
        } else {
            anyhow::ensure!(headers.contains(name.as_str()), "coordinator startup lacks {name}");
        }
    }
    for layer in 0..model.cfg.layers {
        let name = format!("model.language_model.layers.{layer}.mlp.experts.post_expert_norm.weight");
        anyhow::ensure!(headers.contains(name.as_str()), "coordinator combine lacks {name}");
    }
    Ok(coordinator)
}

#[test]
fn published_native_norms_belong_to_coordinator_not_sparks() -> anyhow::Result<()> {
    let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../tests/fixtures/muse-89998fb1.json"))?;
    // Expand the miniature's layer pattern and router extents to the release
    // geometry required by Model::open; other projection/native classes stay as-is.
    let config = testing::muse_spark_config();
    let cfg = crate::families::muse_spark::MuseSparkConfig::from_hf(&config)?;
    let miniature = crate::families::muse_spark::MuseSparkConfig::from_hf(&fixture["config"])?;
    let mut tensors = Vec::new();
    for (name, header) in fixture["headers"].as_object().unwrap() {
        let dtype = match header["dtype"].as_str().unwrap() {
            "BF16" => "BF16", "F16" => "F16", "F32" => "F32", "I16" => "I16", "I32" => "I32",
            other => anyhow::bail!("unexpected published dtype {other}"),
        };
        let mut shape: Vec<usize> = serde_json::from_value(header["shape"].clone())?;
        if let Some(rest) = name.strip_prefix("model.language_model.layers.") {
            let (source, leaf) = rest.split_once('.').unwrap();
            if matches!(leaf, "mlp.gate.weight" | "mlp.gate.e_score_correction_bias") {
                assert_eq!(shape[0], miniature.experts);
                shape[0] = cfg.experts;
            }
            for layer in (source.parse::<usize>()?..cfg.layers).step_by(miniature.layers) {
                tensors.push(testing::t(format!("model.language_model.layers.{layer}.{leaf}"), dtype, &shape));
            }
        } else { tensors.push(testing::t(name, dtype, &shape)); }
    }
    let dir = tempfile::tempdir()?;
    testing::write_snapshot(dir.path(), &config, &tensors, None);
    let coordinator = check_coordinator_startup_inventory(dir.path())?;
    let inventory = Checkpoint::inventory(dir.path())?;
    let spark = plan::files::required_tensors(&inventory, plan::files::ReadRole::Spark { rank: 0, world: 4 })?;
    for layer in 0..62 {
        let norm = format!("model.language_model.layers.{layer}.mlp.experts.post_expert_norm.weight");
        assert!(coordinator.tensors.iter().any(|t| t.meta.name == norm));
        assert!(!spark.contains(&norm), "Spark must not own coordinator combine norm {norm}");
        assert!(spark.contains(&format!("model.language_model.layers.{layer}.mlp.experts.0.down_proj.trellis")));
    }
    assert!(!coordinator.tensors.iter().any(|t| t.meta.name.contains(".experts.0.")));
    Ok(())
}

#[test]
fn mounted_published_coordinator_contains_every_startup_tensor() -> anyhow::Result<()> {
    let Some(snapshot) = std::env::var_os("CUTEAFD_MUSE_PUBLICATION_SNAPSHOT") else {
        eprintln!("skip full publication receipt: set CUTEAFD_MUSE_PUBLICATION_SNAPSHOT");
        return Ok(());
    };
    let coordinator = check_coordinator_startup_inventory(std::path::Path::new(&snapshot))?;
    eprintln!("Muse full publication coordinator: {} selected headers, {} indexed tensors; all startup tensors present",
        coordinator.tensors.len(), coordinator.weight_map.len());
    Ok(())
}

#[test]
fn header_inventory_accepts_k6_core_and_k4_experts_and_reports_bytes() {
    let dir = tempfile::tempdir().unwrap();
    testing::write_muse_media_snapshot(dir.path(), &all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let report = input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(6), ..Default::default() }).unwrap();
    assert_eq!(report.family.as_deref(), Some("muse_spark"));
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    for component in report.components.iter().filter(|c| c.status != Status::Disabled) {
        assert_eq!(component.rejected, 0, "{:?}: {:?}", component.component, component.rejections);
    }
    assert_eq!(report.vision, MediaMode::Auto, "vision auto is not family-disabled");
    assert_eq!(report.audio, MediaMode::Auto, "bundled PCM audio is not family-disabled");
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
fn bundled_media_inventory_is_header_exact_and_never_hides_declared_errors() {
    use crate::placement::families::muse_spark::MediaOptions;
    let dir = tempfile::tempdir().unwrap();
    testing::write_muse_media_snapshot(dir.path(), &[0]);
    let open = || Checkpoint::open(dir.path()).unwrap();
    let inventory = MediaOptions::from_checkpoint(&open(), MediaMode::Auto, MediaMode::Auto, 2).unwrap();
    assert_eq!(inventory.vision_bytes.0 + inventory.vision_bytes.1, 4_860_356_960);
    assert_eq!(inventory.audio_bytes, (235_680_768, 10_905_600));
    assert_eq!(inventory.replicas, 2);
    let path = dir.path().join("config.json");
    let original: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    for (tower, field, invalid) in [("vision_config", "patch_size", serde_json::json!(16)),
        ("audio_config", "conv_kernel_size", serde_json::json!(9))] {
        let mut config = original.clone();
        config[tower][field] = invalid;
        std::fs::write(&path, config.to_string()).unwrap();
        assert!(MediaOptions::from_checkpoint(&open(), MediaMode::Auto, MediaMode::Auto, 1).is_err(), "{tower}");
        let disabled = MediaOptions::from_checkpoint(&open(), MediaMode::Off, MediaMode::Off, 1).unwrap();
        assert_eq!((disabled.vision_bytes, disabled.audio_bytes), ((0, 0), (0, 0)));
    }
    let mut config = original;
    config.as_object_mut().unwrap().remove("vision_config");
    config.as_object_mut().unwrap().remove("audio_config");
    std::fs::write(&path, config.to_string()).unwrap();
    let missing = MediaOptions::from_checkpoint(&open(), MediaMode::Auto, MediaMode::Auto, 1).unwrap();
    assert_eq!((missing.vision_bytes, missing.audio_bytes), ((0, 0), (0, 0)));
    assert!(MediaOptions::from_checkpoint(&open(), MediaMode::Rtx(None), MediaMode::Off, 1).is_err());
    assert!(MediaOptions::from_checkpoint(&open(), MediaMode::Off, MediaMode::Spark(None), 1).is_err());
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

/// Automatic context follows the admitted pool; an explicit 1M pin still
/// reports the typed capacity shortfall on one RTX.
#[test]
fn automatic_context_fits_one_rtx_and_full_context_pin_reports_shortfall() {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let package = tempfile::tempdir().unwrap();
    let mut options = with_package(1, 4, package.path());
    let report = input.plan(&options).unwrap();
    assert!(report.fits && report.wip_launchable(), "{:?}", report.hints);
    let pool = report.memory_layout.as_ref().unwrap().pool_tokens;
    assert!((262144..1 << 20).contains(&pool));
    assert_eq!(report.phase_ledger.as_ref().unwrap().served_context_tokens, pool);
    options.layout.as_mut().unwrap().context_tokens = 1 << 20;
    let report = input.plan(&options).unwrap();
    assert!(!report.fits && !report.wip_launchable(), "{:?}", report.hints);
    let hint = report.hints.iter().map(|hint| &hint.what).find(|what|
        what.contains("selected context 1048576 cannot fit")).expect("the typed shortfall");
    assert!(hint.contains("maximum feasible pool is"), "{hint}");
    assert!(hint.contains("pass a lower explicit context"), "{hint}");
    // The declared max card context (1M) still plans on the head-split pair.
    let report = input.plan(&with_package(2, 6, package.path())).unwrap();
    assert!(report.fits && report.wip_launchable(), "{:?}", report.hints);
    assert!(report.memory_layout.as_ref().unwrap().pool_tokens >= 1 << 20);
}

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
        let mut options = with_package(rtx, ranks, package.path());
        if column == 1 { options.layout.as_mut().unwrap().context_tokens = MIN_CARD_CONTEXT; }
        let report = input.plan(&options).unwrap();
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
            // The shared pool policy is the only pool rule: 2M on a 96 GiB
            // card, never below the declared serving context, capped by what
            // the admission actually fits.
            assert!(memory.pool_tokens >= if rtx == 1 { MIN_CARD_CONTEXT } else { 1 << 20 });
            assert!(memory.pool_tokens <= cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS);
            if rtx == 2 { assert_eq!(memory.pool_tokens, 2 << 20, "the pooled target, not a fit cap"); }
            else { assert!(memory.pool_tokens < 1 << 20, "one RTX is fit-capped below the compiled context"); }
        }
        if (rtx, ranks) == (2, 6) {
            assert!(report.fits && report.placement_supported, "{:?}", report.hints);
            let memory = report.memory_layout.as_ref().unwrap();
            assert!(memory.pool_tokens >= 1 << 20, "pool {}", memory.pool_tokens);
            // Onboarding stays automatic; pool first decides the count.
            assert!(report.onboard_layers.is_some());
        }
    }
    // The installed inventory supports the serving path at the qualified width.
    let mut serving = with_layout(column, 6);
    if column == 1 { serving.layout.as_mut().unwrap().context_tokens = MIN_CARD_CONTEXT; }
    let report = input.plan(&serving).unwrap();
    assert_eq!(report.runtime, Some(RuntimeStatus::Serving));
    assert_eq!(report.spark_ranks, 6, "the serving plan preserves the requested Spark width");
    assert!(report.executable() && report.fits && report.placement_supported, "{:?}", report.hints);
    assert!(report.memory_layout.as_ref().unwrap().pool_tokens >= 256 << 10,
        "the executable layout must preserve the agentic context floor");
}

/// Zero and unset pool select one recipe (the same request, the same prefill
/// shape), and a pool the checkpoint cannot hold is a typed refusal on both
/// the direct and the count-admission edges, never a silently clamped plan.
#[test]
fn muse_auto_pool_and_count_admission_preserve_one_recipe() {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let package = tempfile::tempdir().unwrap();
    let options = with_package(2, 6, package.path());
    let unset = input.plan(&options).unwrap();
    let mut zero = options.clone();
    zero.layout.as_mut().unwrap().pool_tokens = Some(0);
    let automatic = input.plan(&zero).unwrap();
    assert!(automatic.wip_launchable(), "{}", render(&automatic));
    assert_eq!(automatic.memory_layout.as_ref().unwrap().pool_tokens,
        unset.memory_layout.as_ref().unwrap().pool_tokens);
    assert_eq!(automatic.prefill_unit_rows, unset.prefill_unit_rows);
    zero.layout.as_mut().unwrap().pool_tokens = Some(64);
    let direct = input.plan(&zero).unwrap();
    let counted = input.plan_spark_count(&zero, 6, Some(6)).unwrap();
    assert!(!counted.wip_launchable());
    assert_eq!(direct.prefill_unit_rows, counted.prefill_unit_rows);
    for hint in &direct.hints {
        assert!(counted.hints.iter().any(|other| other.what == hint.what));
    }
    assert!(!counted.hints.iter().any(|hint| hint.what.contains("reserve does not fit")), "{:?}", counted.hints);
}

/// HOST and PINNED share one sampled contract, including lazy retention and
/// Load-only push owners. A failed host contract is never retried generically.
#[test]
fn host_contract_admits_all_phases_and_preserves_host_refusal() {
    use crate::placement::planner::{Location, Phase};
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let package = tempfile::tempdir().unwrap();
    let mut options = with_package(2, 6, package.path());
    options.layout.as_mut().unwrap().host_cache_bytes = Some((1 << 30) + 123);
    let report = input.plan(&options).unwrap();
    assert!(report.fits, "{:?}", report.hints);
    assert_eq!(report.host_budget_bytes, (1 << 30) + 123);
    let host = report.host_ledger.as_ref().unwrap();
    let cfg = crate::families::muse_spark::MuseSparkConfig::from_hf(&input.checkpoint().config).unwrap();
    let chunk = (256u64 << 20).max(crate::placement::families::muse_spark::mark_bytes(&cfg, 1));
    assert_eq!(host.host_budget_bytes, ((1u64 << 30) + 123) / chunk * chunk,
        "only whole retained chunks allocate");
    for group in ["transport/rdma-rings", "transport/egress", "spark-intake", "token selector"] {
        assert!(host.items.iter().any(|item| item.group == group), "missing {group}");
    }
    let ledger = report.phase_ledger.as_ref().unwrap();
    let budget = ledger.budgets.iter().find(|budget| budget.id == "host").unwrap();
    assert_eq!(budget.locations, [Location::HOST, Location::PINNED]);
    let load = ledger.phases.iter().find(|row| row.budget == "host" && row.phase == Phase::Load).unwrap();
    assert!(load.owned > 0 && load.opaque >= 4 << 30, "both estimated dense ranks are Load owners");
    let ready = ledger.phases.iter().find(|row| row.budget == "host" && row.phase == Phase::Ready).unwrap();
    assert_eq!(ready.owned + ready.opaque, host.deferred_bytes + host.host_budget_bytes);
    assert_eq!(ready.opaque, 1536 << 20, "census-rounded ordinary host allowance");
    use crate::placement::planner::{Allocation, Lifetime, PhaseOrder, driver_page_table_bytes};
    use crate::placement::planner::serving::{ServingOwners, SelectorOwners, HostPrefixOwners};
    use crate::families::muse_spark::{core::CoreInventory, weight_layout::{LayerLayout, native_mount_load}};
    use cuteafd_core::memory_layout::Category;
    let opts = options.layout.as_ref().unwrap();
    let slots = opts.resolved_state_slots("muse_spark");
    let prefill = crate::serving_capacity::prefill_waves::PrefillPlan {
        unit_rows: report.prefill_unit_rows.unwrap(), waves: report.prefill_waves.unwrap(),
        span_rows: report.prefill_span_rows.unwrap() };
    let placement = report.solver_placement.as_ref().unwrap();
    let mut pinned = ServingOwners {
        selector: SelectorOwners { location: Location::coordinator(1),
            capacity: crate::families::muse_spark::lane::HEAD_ROWS, vocab: cfg.vocab_size, device_sampling: true },
        prefix: HostPrefixOwners { quota_bytes: host.host_budget_bytes, chunk_bytes: chunk },
    }.allocations().unwrap().into_iter().filter(|owner| owner.location == Location::PINNED).collect::<Vec<_>>();
    let cutoff = crate::families::muse_spark::replies::Cutoff::with_formats(opts.muse_partial_bf16_max_rows,
        crate::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS, opts.spark_reply_formats).unwrap();
    let transport = crate::placement::families::muse_spark::host_transport(&cfg, prefill, cutoff, slots as usize, 6).unwrap();
    for (id, bytes) in [("rings", transport.rings), ("egress", transport.egress), ("intake", transport.intake)] {
        pinned.push(Allocation::buffer(id, Location::PINNED, bytes, Lifetime::RESIDENT, Category::Staging, id));
    }
    let core = CoreInventory::from_checkpoint(&cfg, input.checkpoint()).unwrap();
    let layouts = (0..cfg.layers).map(|layer| LayerLayout::ranks(&cfg, layer, 2, &core).unwrap()).collect::<Vec<_>>();
    for gpu in 0..2 {
        let mount = native_mount_load(layouts.iter().flat_map(|ranks| ranks[gpu].native.values())).unwrap();
        pinned.extend(mount.allocations(&format!("muse.native.load.{gpu}")).into_iter()
            .filter(|owner| owner.location == Location::PINNED));
    }
    let catalog = input.expert_catalog().unwrap();
    let manifest = catalog.exl3().unwrap();
    let validation = (0..2).map(|gpu| (0..placement.onboard_layers).map(|layer|
        manifest.residency(crate::V41Exl3Layer::Backbone(layer), 2, gpu).unwrap().loads.iter()
            .filter(|load| load.tensor.ends_with(".trellis")).count() as u64 * 4).sum()).collect();
    // Dense staging is an estimated allowance, not a physical pinned root.
    let push = crate::push_load::PushHostEnvelope::new(validation, vec![0; 2],
        cuteafd_ffi::push_ring::DEFAULT_SLOTS as u64, cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES as u64).unwrap();
    pinned.extend(push.allocations("muse.push").unwrap().into_iter().filter(|owner| owner.location == Location::PINNED));
    for gpu in 0..2 {
        let id = format!("rtx{gpu}");
        let tails = placement.items[gpu].iter().filter(|item| item.group == "allocation granularity")
            .map(|item| item.bytes).sum::<u64>();
        for row in ledger.phases.iter().filter(|row| row.budget == id) {
            let mappings = pinned.iter().filter(|owner| owner.lifetime.contains(row.phase, PhaseOrder::TARGET_THEN_DRAFT))
                .map(|owner| driver_page_table_bytes(owner.bytes)).sum::<u64>();
            // The retired opaque runtime constant is replaced by the measured
            // driver stream slabs (`muse/driver-stream-slabs/gpuN`): the model
            // stream, the push loader's copy streams and the transient rotary
            // table builder, four live streams per 2 MiB slab (#1561).
            let peak_live = 1 + crate::push_load::COPY_STREAMS as u64 + 1;
            let slabs = crate::placement::planner::cuda_stream_slab_bytes(peak_live).unwrap();
            assert_eq!(row.opaque, slabs + tails + mappings,
                "driver stream slabs, target arena tail and physical pinned mappings at {:?}", row.phase);
        }
        let ready_mappings = pinned.iter().filter(|owner| owner.lifetime.contains(Phase::Ready, PhaseOrder::TARGET_THEN_DRAFT))
            .map(|owner| driver_page_table_bytes(owner.bytes)).sum::<u64>();
        assert_eq!(placement.items[gpu].iter().filter(|item|
            item.group == cuteafd_core::memory_layout::DRIVER_PAGE_TABLES_GROUP).map(|item| item.bytes).sum::<u64>(), ready_mappings);
    }
    options.layout.as_mut().unwrap().host_memory = Some((256 << 30, 28 << 30));
    let direct = input.plan(&options).unwrap();
    let counted = input.plan_spark_count(&options, 6, Some(6)).unwrap();
    assert!(!direct.fits && !counted.fits);
    assert!(direct.hints.iter().any(|hint| hint.what.contains("host")));
    for hint in &direct.hints {
        assert!(counted.hints.iter().any(|other| other.what == hint.what));
    }
    assert!(!counted.hints.iter().any(|hint| hint.what.contains("reserve does not fit")));
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
    config.as_object_mut().unwrap().remove("audio_config");
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
    let mut all = missing.clone();
    all.layout.as_mut().unwrap().onboard = Some(crate::placement::OnboardSetting::All);
    let routed = input.expert_catalog().unwrap().routed_experts().routed_layers();
    assert_eq!(routed, all_layers().len());
    assert_eq!(all.layout.as_ref().unwrap().resolve_onboard(routed), Some(crate::placement::Onboard::Layers(routed)));
    let refused = input.plan(&all).unwrap();
    assert!(refused.hints.iter().any(|h| h.what.contains("image lacks exl3-muse-k45/rtx-tp2")), "{:?}", refused.hints);
    let mut explicit = missing.clone();
    explicit.layout.as_mut().unwrap().onboard = Some(crate::placement::OnboardSetting::Layers(2));
    let refused = input.plan(&explicit).unwrap();
    assert!(refused.hints.iter().any(|h| h.what.contains("image lacks exl3-muse-k45/rtx-tp2")), "{:?}", refused.hints);
    // Explicit 0 is a labelled measurement arm; it needs no package.
    let mut zero = missing.clone();
    zero.layout.as_mut().unwrap().onboard = Some(crate::placement::OnboardSetting::Layers(0));
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
    // One RTX at the min card's declared context: the pool takes the GPU (onboarding 0).
    let mut one = with_layout(1, 6);
    one.layout.as_mut().unwrap().context_tokens = MIN_CARD_CONTEXT;
    one.layout.as_mut().unwrap().workspace_manifest = Some(manifest);
    let unpackaged = input.plan(&one).unwrap();
    let mut plain = with_layout(1, 6);
    plain.layout.as_mut().unwrap().context_tokens = MIN_CARD_CONTEXT;
    let bare = input.plan(&plain).unwrap();
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
    let nonweights = expected.owned - expected.weights.0 + expected.scratch + expected.head_scratch + expected.runtime;
    assert_eq!(group(1), nonweights);
    let arena = expected.arena(0, 1).unwrap();
    let arena_bytes = arena.bytes() as u64;
    let tail = arena_bytes.div_ceil(2 << 20) * (2 << 20) - arena_bytes;
    let root_bytes = |placement: &crate::placement::Placement, gpu: usize|
        placement.items[gpu].iter().filter(|item| item.group == "weight arena").map(|item| item.bytes).sum::<u64>();
    let plain_placed = plain.solver_placement.as_ref().unwrap();
    assert_eq!(root_bytes(placed, 0), root_bytes(plain_placed, 0));
    assert_eq!(root_bytes(placed, 1) - root_bytes(plain_placed, 1), arena_bytes);
    let tails = |placement: &crate::placement::Placement, gpu: usize|
        placement.items[gpu].iter().filter(|item| item.group == "allocation granularity").map(|item| item.bytes).sum::<u64>();
    assert_eq!(tails(placed, 1) - tails(plain_placed, 1), tail);
    assert_eq!(expected.runtime, (16 * 8 * 4 + 62 * 64 * 8 * 4) as u64, "draft ids + probe ring");
    assert_eq!(expected.head_scratch, (16 * 8 * (202_112 + 202_048) * 4) as u64);
    assert!(placed.pool_tokens <= plain.solver_placement.as_ref().unwrap().pool_tokens);
    // One RTX cannot reach the pool cap: the exact drafter charge must displace KV pages.
    let mut single = with_layout(1, 6);
    single.layout.as_mut().unwrap().context_tokens = MIN_CARD_CONTEXT;
    let mut single_draft = single.clone();
    single_draft.layout.as_mut().unwrap().muse_drafter_snapshot = Some(drafter.path().to_path_buf());
    let single_plain = input.plan(&single).unwrap();
    let single_drafted = input.plan(&single_draft).unwrap();
    assert!(single_plain.fits && single_drafted.fits, "{:?}", single_drafted.hints);
    let memory = single_plain.memory_layout.as_ref().unwrap();
    let device = memory.devices.iter().find(|device|
        device.kind == cuteafd_core::memory_layout::DeviceKind::Rtx).unwrap();
    let charged = nonweights + arena_bytes + tail;
    let draft_device = single_drafted.memory_layout.as_ref().unwrap().devices.iter().find(|device|
        device.kind == cuteafd_core::memory_layout::DeviceKind::Rtx).unwrap();
    assert_eq!(draft_device.items.iter().filter(|item| item.group == "Drafter").map(|item| item.bytes).sum::<u64>(), nonweights);
    let fixed = |device: &cuteafd_core::memory_layout::DeviceLayout| device.items.iter()
        .filter(|item| item.group != "full-layer KV (paged K+V records)").map(|item| item.bytes).sum::<u64>();
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
    for window in [serde_json::json!(1024), serde_json::Value::Null] {
        let mut other = config.clone();
        other["sliding_window"] = window;
        std::fs::write(drafter.path().join("config.json"), other.to_string()).unwrap();
        let refused = input.plan(&with_draft).unwrap();
        assert!(refused.hints.iter().any(|h| h.what.contains("ring holds 2048")), "{:?}", refused.hints);
    }
}

/// An explicitly requested tower with no executable device is a first typed
/// refusal that names the tower and the requested placement. Auto never lands
/// here: its domain always closes with the Off option.
#[test]
fn explicit_unplaceable_tower_is_refused_by_name() {
    let dir = tempfile::tempdir().unwrap();
    testing::write_muse_media_snapshot(dir.path(), &all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let case = |vision: MediaMode, audio: MediaMode| {
        let mut options = with_layout(1, 6);
        let layout = options.layout.as_mut().unwrap();
        layout.context_tokens = 4096;
        layout.pool_tokens = Some(4096);
        options.vision = vision;
        options.audio = audio;
        input.plan(&options).unwrap()
    };
    // Eight vision replicas cannot exist on one RTX; a rank-8 audio copy cannot
    // exist across six Sparks.
    let refused = case(MediaMode::Rtx(Some(8)), MediaMode::Off);
    assert!(!refused.fits && !refused.placement_supported, "{:?}", refused.hints);
    assert!(refused.encoder.as_ref().unwrap().reason.contains("not admitted"));
    assert!(refused.audio_encoder.as_ref().unwrap().reason.contains("not admitted"));
    assert!(refused.hints.iter().any(|hint| hint.what
        .contains("requested vision tower (Rtx(Some(8))) has no executable device")), "{:?}", refused.hints);
    let refused = case(MediaMode::Off, MediaMode::Spark(Some(8)));
    assert!(!refused.fits && !refused.placement_supported, "{:?}", refused.hints);
    assert!(refused.hints.iter().any(|hint| hint.what
        .contains("requested audio tower (Spark(Some(8))) has no executable device")), "{:?}", refused.hints);
}

#[test]
fn media_auto_off_disables_component_bytes() {
    let dir = tempfile::tempdir().unwrap();
    testing::write_muse_media_snapshot(dir.path(), &all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    // Force Auto through Off without changing the text inventory: impossible
    // replica count eliminates Sparks; a capacity just above the text-only
    // layout leaves no room for either tower.
    let mut options = with_layout(1, 6);
    options.vision = MediaMode::Off;
    options.audio = MediaMode::Off;
    let opts = options.layout.as_mut().unwrap();
    opts.context_tokens = 4096;
    opts.pool_tokens = Some(4096);
    opts.onboard = Some(crate::placement::OnboardSetting::Layers(0));
    let text = input.plan(&options).unwrap();
    let used = text.memory_layout.as_ref().unwrap().devices[0].used_bytes();
    options.vision = MediaMode::Auto;
    options.audio = MediaMode::Auto;
    // The ledger excludes the baseline CUDA context deducted by GpuBudget.
    let context = crate::placement::ArchContext::for_device("sm_120", used).context_bytes;
    options.layout.as_mut().unwrap().rtx_bytes = vec![used + context + 1024];
    options.layout.as_mut().unwrap().vision_replicas = 9;
    // Audio has one copy, so remove its Spark candidate by retaining only the
    // text's exact Spark allocation budget.
    let spark_used = text.memory_layout.as_ref().unwrap().devices.iter()
        .filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Spark)
        .map(|d| d.used_bytes()).max().unwrap();
    options.layout.as_mut().unwrap().spark_allocation_budget_bytes = Some(spark_used);
    let off = input.plan(&options).unwrap();
    assert!(off.fits, "{:?}", off.hints);
    assert_eq!(off.encoder.as_ref().unwrap().kind, crate::plan::encoder::EncoderKind::Off);
    assert_eq!(off.audio_encoder.as_ref().unwrap().kind, crate::plan::encoder::EncoderKind::Off);
    assert_eq!(off.memory_layout.as_ref().unwrap().pool_tokens, 4096);
    assert!(plan::render(&off).contains("Muse axial/window attention"));
    assert!(off.disabled_media_bytes > 0, "{:?}", off.hints);
    for component in [Component::Vision, Component::Audio] {
        let component = off.components.iter().find(|c| c.component == component).unwrap();
        assert_eq!(component.status, Status::Disabled);
        assert_eq!(component.bytes, 0);
    }
}

#[test]
fn coordinator_media_hash_includes_vision_and_pcm_tensor_headers() {
    let dir = tempfile::tempdir().unwrap();
    testing::write_muse_media_snapshot(dir.path(), &[0]);
    let hash = || {
        let checkpoint = Checkpoint::coordinator(dir.path(), false, false).unwrap();
        assert!(checkpoint.tensors.iter().all(|t| !t.meta.name.starts_with("model.audio_tower.")
            && !crate::media::muse_vision::is_tensor(&t.meta.name)));
        PlanInput::from_checkpoint(checkpoint).plan(&PlanOptions {
            vision: MediaMode::Off, audio: MediaMode::Off,
            placement: ExpertPlacement::from_spark_ranks(6), ..Default::default()
        }).unwrap().encoder_plan_hash
    };
    let mut previous = hash();
    let path = dir.path().join("model-00000.safetensors");
    // Change only a tower header, not config or text tensors. F16 has the same
    // storage extent as BF16; disabled towers do not require runnable numerics.
    for prefix in ["model.vision", "model.audio_tower."] {
        use std::io::{Read, Seek, Write};
        let mut file = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
        let mut length = [0; 8];
        file.read_exact(&mut length).unwrap();
        let mut bytes = vec![0; u64::from_le_bytes(length) as usize];
        file.read_exact(&mut bytes).unwrap();
        let mut header: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let name = header.as_object().unwrap().keys().find(|name| name.starts_with(prefix)).unwrap().clone();
        header[&name]["dtype"] = serde_json::json!("F16");
        let mut updated = serde_json::to_vec(&header).unwrap();
        assert!(updated.len() <= bytes.len());
        updated.resize(bytes.len(), b' ');
        file.seek(std::io::SeekFrom::Start(8)).unwrap();
        file.write_all(&updated).unwrap();
        let next = hash();
        assert_ne!(previous, next, "{prefix}");
        previous = next;
    }
}

#[test]
fn host_transport_uses_resolved_small_and_large_card_slots_only() {
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let cfg = crate::families::muse_spark::MuseSparkConfig::from_hf(&testing::muse_spark_config()).unwrap();
    for (bytes, slots) in [(32 << 30, 8), (crate::placement::inventory::PRO_TOTAL_BYTES, 16)] {
        let mut options = with_layout(1, 6);
        options.layout.as_mut().unwrap().rtx_bytes = vec![bytes];
        options.layout.as_mut().unwrap().context_tokens = 1024;
        options.layout.as_mut().unwrap().pool_tokens = Some(1024);
        let report = input.plan(&options).unwrap();
        let opts = options.layout.as_ref().unwrap();
        assert_eq!(opts.resolved_state_slots("muse_spark"), slots);
        let unit_rows = report.prefill_unit_rows.unwrap();
        let waves = report.prefill_waves.unwrap();
        let shape = crate::serving_capacity::prefill_waves::PrefillPlan { unit_rows, waves, span_rows: unit_rows * waves };
        let cutoff = crate::families::muse_spark::replies::Cutoff::with_formats(opts.muse_partial_bf16_max_rows,
            crate::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS, opts.spark_reply_formats).unwrap();
        let expected = crate::placement::families::muse_spark::host_transport(&cfg, shape, cutoff, slots as usize, 6).unwrap();
        let host = report.host_ledger.as_ref().unwrap_or_else(|| panic!("{bytes} B card: {:?}", report.hints));
        for (name, bytes) in [("transport/rdma-rings", expected.rings), ("transport/egress", expected.egress),
            ("spark-intake", expected.intake)] {
            assert_eq!(host.items.iter().find(|item| item.group == name).unwrap().bytes, bytes, "{name}");
        }
        assert!(host.items.iter().all(|item| !matches!(item.group.as_str(),
            "workspace/router" | "staging/tables-and-sync")));
    }
}

/// The family's epilogue declaration is what sizes the Spark workspace: Muse
/// Spark replies per-expert partials, so its executors never allocate the
/// weighted sum's token output, while every sum-epilogue family still does.
#[test]
fn routed_experts_declare_the_partial_reply_epilogue() {
    use crate::serving_capacity::Exl3Epilogue;
    let dir = snapshot(&all_layers());
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let catalog = input.expert_catalog().unwrap();
    assert!(catalog.exl3().is_some());
    let first = catalog.routed_experts().first_layer;
    assert_eq!(first, all_layers()[0]);
    assert!(catalog.per_expert_output_norm());
    assert_eq!(Exl3Epilogue::for_catalog(catalog), Exl3Epilogue::PartialReplies);
    assert!(!Exl3Epilogue::PartialReplies.allocates("output"));
    assert!(Exl3Epilogue::Sum.allocates("output"));
}
