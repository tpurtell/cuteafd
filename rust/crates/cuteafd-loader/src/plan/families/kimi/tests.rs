use super::*;
use crate::plan::{self, checkpoint::Checkpoint, format, testing::{self, t, kimi_k3_tensors as tensors}, *};
use anyhow::Context;
use serde_json::Value;
use crate::families::kimi_k3::validate_expert_inventory;

fn config() -> Value { crate::families::kimi_k3::config::tests::release_config() }

/// One shard per layer, as the release (a single 990k-entry header would
/// exceed the 64 MiB safetensors header bound).
fn snapshot(all: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    testing::write_kimi_snapshot(dir.path(), all);
    dir
}

fn with_layout(ranks: usize, budget: Option<u64>) -> PlanOptions {
    PlanOptions { attention_placement: Some(crate::placement::AttentionPlacement::Layers), placement: ExpertPlacement::from_spark_ranks(ranks), spark_budget_bytes: budget,
        layout: Some(layout::LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![crate::placement::inventory::PRO_TOTAL_BYTES; 2],
            // The text card retains device snapshots; host spilling is disabled.
            host_cache_bytes: Some(0), spark_allocation_budget_bytes: budget, ..Default::default() }), ..Default::default() }
}

fn installed_inventory(root: &std::path::Path, spark: bool) -> std::path::PathBuf {
    testing::write_kimi_installed_inventory(root, spark)
}

#[test]
fn shared_catalog_rejects_the_same_missing_indexed_headers_as_standalone() {
    let dir = snapshot(false);
    let path = dir.path().join("model-00000.safetensors");
    testing::write_safetensors(&path, &[t("unindexed.weight", "BF16", &[2])]);
    let input = PlanInput::open(dir.path()).unwrap();
    let cached = format!("{:#}", input.expert_catalog().unwrap_err());
    let standalone = format!("{:#}", crate::read_expert_catalog(dir.path()).unwrap_err());
    assert!(cached.contains("missing from shard"), "{cached}");
    assert_eq!(cached, standalone);
}

#[test]
fn planner_publishes_kimi_runtime_default_and_explicit_units() {
    let dir = snapshot(false);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    for (unit, waves, expected) in [(0, 0, (2752, 3, 8256)), (1024, 2, (1024, 2, 2048)),
        (4096, 1, (4096, 1, 4096)), (0, 4, (2048, 4, 8192)), (4096, 2, (4096, 2, 8192))] {
        let mut options = with_layout(6, None);
        let layout = options.layout.as_mut().unwrap();
        layout.prefill_rows = unit;
        layout.prefill_lanes = waves;
        layout.host_memory = Some((128 << 30, 128 << 30));
        let report = input.plan(&options).unwrap();
        assert_eq!((report.prefill_unit_rows.unwrap(), report.prefill_waves.unwrap(), report.prefill_span_rows.unwrap()), expected);
        assert_eq!(report.prefill_transports, Some(1));
    }
}

#[test]
fn planner_resolves_every_spark_count_before_recipe_refusal() {
    let dir = snapshot(false);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    for ranks in 0..=8 {
        let report = input.plan(&with_layout(ranks, None)).unwrap();
        let daemon = crate::serving_capacity::prefill_waves::prefill_shape(ranks,
            crate::plan::Family::prefill_limits(&super::KIMI_K3), Default::default()).unwrap();
        assert_eq!((report.prefill_unit_rows, report.prefill_waves, report.prefill_span_rows),
            (Some(daemon.unit_rows), Some(daemon.waves), Some(daemon.span_rows)), "ranks={ranks}: {:?}", report.hints);
        assert!(!report.fits && !report.placement_supported && !report.wip_launchable());
        assert!(report.hints.iter().any(|hint| hint.what.contains("Kimi header inventory refused: missing projection")),
            "ranks={ranks}: {:?}", report.hints);
        assert!(report.memory_layout.unwrap().devices.is_empty());
    }
}

#[test]
fn complete_checkpoint_preserves_recipe_refusals_at_every_spark_count() {
    let dir = snapshot(true);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    for ranks in 1..=8 {
        let report = input.plan(&with_layout(ranks, Some(108 << 30))).unwrap();
        let expected = crate::plan::experts::spark_recipe("kimik3:exl3-k23", 3072, ranks);
        assert!(!report.hints.iter().any(|hint| hint.what.contains("missing projection")), "{:?}", report.hints);
        if expected.is_err() {
            assert!(!report.fits && !report.placement_supported);
            assert!(report.hints.iter().any(|hint| hint.what.contains("routed experts need")
                && hint.what.contains("over the 108 GiB budget")),
                "TP{ranks}: {:?}", report.hints);
        } else {
            assert!(!report.hints.iter().any(|hint| hint.what.contains("no exported width class")),
                "TP{ranks}: {:?}", report.hints);
        }
    }
    let refusal = crate::plan::experts::spark_recipe("kimik3:exl3-k23", 3072, 5).unwrap_err();
    let report = input.plan(&with_layout(5, Some(256 << 30))).unwrap();
    assert!(!report.placement_supported);
    assert!(report.hints.iter().any(|hint| hint.what.contains(&refusal.to_string())),
        "{:?}", report.hints);
}

#[test]
fn zero_spark_reports_the_missing_local_executor_without_slicing_world_zero() {
    let dir = snapshot(true);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let report = input.plan(&with_layout(0, None)).unwrap();
    assert!(!report.executable() && !report.wip_launchable());
    assert!(report.solver_placement.is_none());
    let refusal = crate::placement::families::kimi::LOCAL_ONLY_REFUSAL.to_string();
    assert!(report.hints.iter().any(|hint| hint.what.contains(&refusal)), "{:?}", report.hints);
    assert!(!report.hints.iter().any(|hint| hint.what.contains("geometry-invalid")), "{:?}", report.hints);
}

#[test]
fn single_owner_header_plan_never_synthesizes_a_peer_or_half_experts() {
    use crate::placement::{ExpertHome, FfnMode, LayerMode};
    let dir = snapshot(true);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let mut options = with_layout(6, Some(110 << 30));
    // One owner runs whole MLA layers; an explicit `layers` pin needs the pair.
    options.attention_placement = None;
    // This BF16 geometry fixture needs an explicit weight cap above the native text owners.
    options.coordinator_budget_bytes = 192 << 30;
    let layout = options.layout.as_mut().unwrap();
    layout.rtx_bytes = vec![192 << 30];
    layout.head_split = false;
    let report = input.plan(&options).unwrap();
    let placed = report.solver_placement.as_ref().unwrap_or_else(|| panic!("{:?}", report.hints));
    assert_eq!(placed.items.len(), 1);
    assert!(placed.items[0].iter().any(|item| item.group == crate::placement::planner::ALLOCATION_GRANULARITY_GROUP
        && item.bytes > 0 && item.basis == cuteafd_core::memory_layout::Basis::Estimated));
    assert!(placed.layers.iter().all(|layer| layer.mode == LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }));
    assert!(placed.layers.iter().all(|layer| matches!(layer.experts,
        ExpertHome::Dense | ExpertHome::Spark | ExpertHome::RtxWhole { gpu: 0 })));
    assert!(placed.tp2.is_none());
    assert_eq!(report.memory_layout.as_ref().unwrap().devices.iter()
        .filter(|device| device.kind == cuteafd_core::memory_layout::DeviceKind::Rtx).count(), 1);
}

#[test]
fn header_and_installed_min_plans_share_one_arena_backing_and_tail() -> Result<()> {
    use crate::{families::kimi_k3::weight_layout::{coordinator_arena, GlobalLayout, LayerLayout},
        placement::{planner::{allocation_granularity_forecast, ALLOCATION_GRANULARITY_GROUP},
            FfnMode, LayerMode, OnboardSetting}};
    use cuteafd_core::memory_layout::Basis;
    let dir = snapshot(true);
    let input = testing::BudgetedPlanInput::open(dir.path())?;
    let model = Model::open(&Checkpoint::open(dir.path())?)?;
    let globals = GlobalLayout::ranks_with_core(model.cfg(), &model.core, 1)?;
    let layers = (0..model.cfg().layers).map(|layer| LayerLayout::ranks_with_core(model.cfg(), layer,
        LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }, &model.core, 1)).collect::<Result<Vec<_>>>()?;
    let arena = coordinator_arena(0, &globals[0], &layers.iter().map(|pair| &pair[0]).collect::<Vec<_>>())?;
    assert_eq!(arena.payload_bytes(), 113_509_532_032);
    assert_eq!(arena.padding_bytes(), 8832);
    let expected = allocation_granularity_forecast(&[arena.bytes() as u64])?;
    assert!(expected > 0);
    let package = tempfile::tempdir()?;
    let manifest = installed_inventory(package.path(), false);
    let mut options = with_layout(6, Some(192 << 30));
    options.attention_placement = None;
    options.coordinator_budget_bytes = 192 << 30;
    let layout = options.layout.as_mut().unwrap();
    layout.rtx_bytes = vec![192 << 30];
    layout.head_split = false;
    layout.context_tokens = 262_144;
    layout.pool_tokens = Some(262_144);
    layout.prefix_slots = Some(1);
    layout.onboard = Some(OnboardSetting::Layers(0));
    let mut charges = Vec::new();
    for installed in [false, true] {
        options.layout.as_mut().unwrap().workspace_manifest = installed.then(|| manifest.clone());
        let report = input.plan(&options)?;
        let placed = report.solver_placement.as_ref().unwrap_or_else(|| panic!("installed={installed}: {:?}", report.hints));
        // One named tail per fixed single-request device owner: the arena, the
        // KDA state/marks, and on an installed layout the lane buffers and code.
        let owners = placed.items[0].iter().filter(|item| item.group == ALLOCATION_GRANULARITY_GROUP).collect::<Vec<_>>();
        assert!(owners.iter().all(|item| item.basis == Basis::Estimated && item.bytes > 0),
            "installed={installed}: {:?}", owners.iter().map(|item| item.bytes).collect::<Vec<_>>());
        let tail = owners.iter().find(|item| item.bytes == expected)
            .unwrap_or_else(|| panic!("installed={installed}: missing weight arena tail {expected}"));
        let roots = placed.items[0].iter().filter(|item| item.group == "weight arena").collect::<Vec<_>>();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].bytes, arena.bytes() as u64);
        assert!(!placed.items[0].iter().any(|item| item.group == "kimi arena padding"
            || item.group == "Kimi global weights" || item.group == "layers" || item.group.starts_with("Kimi layer ")));
        charges.push((roots[0].bytes, tail.bytes));
    }
    assert_eq!(charges[0], charges[1], "the arena request inventory must not depend on PROGRAMS.json");
    Ok(())
}

#[test]
fn single_owner_header_plan_refuses_mandatory_weights_above_default_cap() {
    use crate::placement::{PlacementError, ResidentWeights};
    let dir = snapshot(true);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let mut options = with_layout(6, Some(110 << 30));
    // One owner runs whole MLA layers; an explicit `layers` pin needs the pair.
    options.attention_placement = None;
    assert_eq!(options.coordinator_budget_bytes, ResidentWeights::DEFAULT_LIMIT.0);
    let layout = options.layout.as_mut().unwrap();
    layout.rtx_bytes = vec![192 << 30];
    layout.head_split = false;
    let report = input.plan(&options).unwrap();
    // The resident-weight cap charges B, including exact inter-view padding, but not its driver tail.
    let refusal = PlacementError::ResidentWeightLimit { gpu: 0, required: 113_509_532_032 + 8832,
        limit: ResidentWeights::DEFAULT_LIMIT.0 };
    assert!(!report.fits && !report.placement_supported);
    assert!(report.solver_placement.is_none());
    assert!(report.hints.iter().any(|hint| hint.what == format!("Kimi K3 coordinator admission refused: {refusal}")),
        "{:?}", report.hints);
    assert!(report.memory_layout.unwrap().devices.is_empty());
}

#[test]
fn planner_refuses_shapes_outside_shared_live_reserve() {
    let dir = snapshot(false);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    for (unit, waves, span) in [(4096, 4, None), (4096, 3, None), (2753, 3, None),
        (2752, 3, Some(8192)), (2752, 3, Some(16384))] {
        let mut options = with_layout(6, None);
        let layout = options.layout.as_mut().unwrap();
        layout.prefill_rows = unit;
        layout.prefill_lanes = waves;
        layout.prefill_span_rows = span;
        let report = input.plan(&options).unwrap();
        assert!(!report.fits && !report.placement_supported);
        assert!(report.hints.iter().any(|hint| hint.what.contains("prefill unsupported")), "{:?}", report.hints);
        assert!(report.memory_layout.unwrap().devices.is_empty());
    }
}

#[test]
fn coordinator_inventory_requires_exact_tp2_exports() {
    use serde_json::json;
    for corruption in ["missing-layout", "missing-capacity", "missing-object", "width", "dtype", "capacity", "hidden", "activation"] {
        let dir = tempfile::tempdir().unwrap();
        let path = installed_inventory(dir.path(), false);
        let package = dir.path().join("lib/exl3/exl3-kimik3-k23");
        validate_expert_inventory(&path, &package).unwrap();
        let manifest = package.join("manifest.json");
        let mut inventory: Value = serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
        match corruption {
            "missing-layout" | "missing-capacity" => {
                inventory["variants"].as_array_mut().unwrap().retain(|entry| {
                    let name = entry["directory"].as_str().unwrap();
                    if corruption == "missing-layout" { !name.starts_with("rtx-tp2/") }
                    else { name != "rtx-tp2/m80" }
                });
                std::fs::write(&manifest, inventory.to_string()).unwrap();
            }
            "missing-object" => std::fs::remove_file(package.join("rtx-tp2/m80/libcuteafd_exl3.so")).unwrap(),
            _ => {
                let meta_path = package.join("rtx-tp2/m80/v41_exl3.json");
                let mut meta: Value = serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
                let (key, value) = match corruption {
                    "width" => ("intermediate", json!(3072)),
                    "dtype" => ("output_dtype", json!("bf16")),
                    "capacity" => ("capacity", json!(16)),
                    "hidden" => ("hidden", json!(7168)),
                    "activation" => ("activation", json!("silu")),
                    _ => unreachable!(),
                };
                meta[key] = value;
                std::fs::write(meta_path, meta.to_string()).unwrap();
            }
        }
        assert!(validate_expert_inventory(&path, &package).is_err(), "accepted {corruption}");
    }
}

#[test]
fn scoped_spark_inventory_requires_every_rank_of_each_declared_world() {
    for world in [6, 7, 8] {
        let dir = tempfile::tempdir().unwrap();
        let path = installed_inventory(dir.path(), true);
        let package = dir.path().join("lib/exl3/exl3-kimik3-k23");
        let manifest = package.join("manifest.json");
        let mut inventory: Value = serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
        let prefix = format!("tp{world}-");
        inventory["variants"].as_array_mut().unwrap().retain(|entry|
            entry["directory"].as_str().unwrap().starts_with(&prefix));
        std::fs::write(&manifest, inventory.to_string()).unwrap();
        validate_expert_inventory(&path, &package).unwrap();
        let recipe = crate::plan::experts::spark_recipe("kimik3:exl3-k23", 3072, world).unwrap();
        let missing = format!("{}/m80", recipe.required_layouts[world - 1]);
        inventory["variants"].as_array_mut().unwrap().retain(|entry| entry["directory"] != missing);
        std::fs::write(&manifest, inventory.to_string()).unwrap();
        assert!(validate_expert_inventory(&path, &package).is_err(), "accepted incomplete TP{world}");
    }
}

#[test]
fn header_acceptance_is_independent_of_runtime_inventory() {
    let snapshot = snapshot(false);
    let checkpoint = Checkpoint::open(snapshot.path()).unwrap();
    let trellis = checkpoint.tensors.iter().filter(|t| t.meta.name.contains(".experts.0.w1.")).cloned().collect::<Vec<_>>();
    let mut operand = format::detect(format::group_by_stem(&trellis).values().next().unwrap()).unwrap();
    let stem = format!("{TEXT}layers.1.block_sparse_moe.experts.0.w1");
    let role = TensorRole::layer(Component::RoutedExpert, 1);
    let model = Model::open(&checkpoint).unwrap();
    model.accepts(&role, &stem, &mut operand).unwrap();
    assert_eq!(model.experts(&operand).unwrap().package, PACKAGE);
    let mut malformed = operand.clone();
    malformed.logical[0] += 128;
    assert!(model.accepts(&role, &stem, &mut malformed).is_err());
    let mut unsupported = operand.clone();
    unsupported.encoding = format::Encoding::Exl3 { bits: 3 };
    assert!(model.accepts(&role, &stem, &mut unsupported).is_err());
    assert!(model.experts(&unsupported).is_none());
    for spark in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = installed_inventory(dir.path(), spark);
        let package = dir.path().join("lib/exl3/exl3-kimik3-k23");
        assert!(!path.parent().unwrap().join("exl3").exists());
        validate_expert_inventory(&path, &package).unwrap();
        assert!(validate_expert_inventory(&path, &dir.path().join("absent-package")).is_err());
        model.accepts(&role, &stem, &mut operand).unwrap();
        if spark {
            let meta_path = dir.path().join("lib/exl3/exl3-kimik3-k23/tp6-rank5/m4096/v41_exl3.json");
            let mut meta: Value = serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
            meta["descriptor_bits"] = serde_json::json!(9);
            std::fs::write(&meta_path, meta.to_string()).unwrap();
        } else {
            std::fs::remove_file(dir.path().join("lib/exl3/exl3-kimik3-k23/rtx-tp1/m80/libcuteafd_exl3.so")).unwrap();
        }
        assert!(validate_expert_inventory(&path, &package).is_err());
        model.accepts(&role, &stem, &mut operand).unwrap();
        assert_eq!(model.experts(&operand).unwrap().package, PACKAGE);
    }
    let dir = tempfile::tempdir().unwrap();
    let path = installed_inventory(dir.path(), false);
    let mut inventory: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    inventory["programs"] = serde_json::json!([]);
    std::fs::write(&path, inventory.to_string()).unwrap();
    let package = dir.path().join("lib/exl3/exl3-kimik3-k23");
    assert!(validate_expert_inventory(&path, &package).unwrap_err().to_string().contains("kimik3_router_scores"));
    assert!(validate_expert_inventory(&dir.path().join("absent.json"), &package).is_err());
}

#[test]
fn release_headers_classify_native_text_without_installed_artifacts() {
    let dir = snapshot(false);
    let report = testing::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert_eq!(report.family.as_deref(), Some("kimi_k3"));
    assert_eq!(report.runtime, Some(RuntimeStatus::Serving));
    assert!(report.unclassified.is_empty(), "{:?}", &report.unclassified[..report.unclassified.len().min(5)]);
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    let spec = report.spec.as_ref().unwrap();
    assert_eq!(spec.layers.iter().filter(|l| matches!(l.attention, AttentionKind::Mla { rope: false, output_gate: true })).count(), 24);
    assert_eq!(spec.moe_layers(), 92);
    for component in [Component::Attention, Component::ResidualMix, Component::LatentProjection, Component::Router,
        Component::RoutedExpert, Component::SharedExpert, Component::DenseFfn] {
        let plan = report.components.iter().find(|c| c.component == component).unwrap_or_else(|| panic!("{component:?}"));
        assert_eq!(plan.status, Status::Ready, "{component:?}");
    }
    let experts = report.components.iter().find(|c| c.component == Component::RoutedExpert).unwrap();
    assert_eq!(experts.formats.keys().collect::<Vec<_>>(), vec!["exl3-k2"]);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let model = KIMI_K3.open(&checkpoint).unwrap();
    let trellis = checkpoint.tensors.iter().filter(|t| t.meta.name.contains(".experts.0.w1.")).cloned().collect::<Vec<_>>();
    let operand = format::detect(format::group_by_stem(&trellis).values().next().unwrap()).unwrap();
    let contract = model.experts(&operand).unwrap();
    assert_eq!((contract.package.as_str(), contract.block, contract.spark_worlds.clone()), (PACKAGE, 128, vec![6, 7, 8]));
    assert_eq!(experts.rejected, 0);
    assert_eq!(report.experts.as_ref().unwrap().package, PACKAGE);
    assert!(report.components.iter().filter(|c| c.component != Component::Vision).all(|c| c.rejected == 0));
    assert!(!report.hints.iter().any(|h| h.what.contains("W9") || h.what.contains("Kimi K3 attention")));
    // The family's own Spark budget, shown in the report and its rendering.
    assert_eq!(report.spark_budget_bytes, SPARK_BUDGET_BYTES);
    // Serving status does not make the default single-GPU header plan fit.
    assert!(!report.executable());
    assert!(report.hints.iter().any(|hint| hint.what.contains("Kimi header inventory refused: missing projection")),
        "{:?}", report.hints);
    assert_eq!(model.vision_mode(MediaMode::Auto), MediaMode::Off);
    let vision_report = testing::plan(dir.path(), &PlanOptions { vision: MediaMode::Rtx(None), ..Default::default() }).unwrap();
    assert!(!vision_report.wip_launchable());
    assert!(vision_report.hints.iter().any(|h| h.what.contains("vision execution is not implemented")));
}

#[test]
fn planned_headers_supply_range_geometry_without_claiming_fit() {
    let dir = snapshot(false);
    let columns = PlanInput::open(dir.path()).unwrap().spark_ranges(&PlanOptions { layout: Some(layout::LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), ..Default::default() }), ..Default::default() }).unwrap();
    assert_eq!(columns.len(), 2);
    for column in columns {
        assert_eq!(column.package, PACKAGE);
        assert_eq!((column.intermediate, column.unit_columns, column.policy.width_cap), (3072, 128, 12));
        assert!(column.routed_expert_bytes > 0);
        assert!(column.range.is_none() && column.executable_minimum.is_none());
    }
}

#[test]
fn auto_pair_requires_context_programs_but_explicit_layers_does_not() {
    let dir = snapshot(false);
    let mut options = with_layout(6, None);
    options.attention_placement = Some(crate::placement::AttentionPlacement::Context);
    let report = testing::plan(dir.path(), &options).unwrap();
    assert!(!report.placement_supported && !report.fits && !report.executable());
    assert!(report.hints.iter().any(|h| h.what.contains("Kimi context program admission refused")
        && h.how.contains("kimik3_mla_partial") && h.how.contains("kimik3_lse_combine2")), "{:?}", report.hints);
    options.attention_placement = None;
    let report = testing::plan(dir.path(), &options).unwrap();
    assert_eq!(report.attention_placement, crate::placement::AttentionPlacement::Context);
    assert!(report.hints.iter().any(|h| h.what.contains("context program")));
    options.attention_placement = Some(crate::placement::AttentionPlacement::Layers);
    let report = testing::plan(dir.path(), &options).unwrap();
    assert_eq!(report.attention_placement, crate::placement::AttentionPlacement::Layers);
    assert!(!report.hints.iter().any(|h| h.what.contains("context program")));
}

#[test]
fn mla_headers_leave_fixed_tp2_residency_exactly_once() {
    let dir = snapshot(false);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let model = Model::open(&checkpoint).unwrap();
    let mut mla_layers = 0;
    for (layer, r) in model.mla.iter().enumerate() {
        if model.cfg.attention[layer] == KimiAttention::Mla {
            mla_layers += 1;
            assert_eq!(r.replicated, 30_281_728);
            assert_eq!(r.split + r.replicated, 464_392_192);
        } else { assert_eq!((r.split, r.replicated), (0, 0)); }
    }
    assert_eq!(mla_layers, 24);
    let components: u64 = model.bytes.iter().filter(|(c, _)| !matches!(c,
        Component::Embedding | Component::LmHead | Component::RoutedExpert | Component::Vision)).map(|(_, n)| n).sum();
    assert_eq!(model.coordinator.split + model.coordinator.replicated
        + model.mla.iter().map(|r| r.split + r.replicated).sum::<u64>(), components);
}

#[test]
fn header_contract_refuses_other_shapes_and_dtypes() {
    for (name, dtype, shape) in [("self_attn.A_log", "F32", vec![96usize]), ("self_attn.o_norm.weight", "BF16", vec![128])] {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors: Vec<_> = tensors(false).into_iter().filter(|(n, ..)| !n.contains(".experts.")).collect();
        let full = format!("{TEXT}layers.1.{name}");
        let entry = tensors.iter_mut().find(|(n, ..)| *n == full).unwrap();
        *entry = t(full.clone(), dtype, &shape);
        testing::write_snapshot(dir.path(), &config(), &tensors, None);
        let report = testing::plan(dir.path(), &PlanOptions::default()).unwrap();
        let error = report.config_error.unwrap_or_default();
        assert!(error.contains(&full) && error.contains("release contract"), "{error}");
    }
}

fn device<'m>(memory: &'m cuteafd_core::memory_layout::MemoryLayout, kind: cuteafd_core::memory_layout::DeviceKind)
    -> Vec<&'m cuteafd_core::memory_layout::DeviceLayout> { memory.devices.iter().filter(|d| d.kind == kind).collect() }

fn category(d: &cuteafd_core::memory_layout::DeviceLayout, c: cuteafd_core::memory_layout::Category) -> u64 {
    d.items.iter().filter(|i| i.category == c).map(|i| i.bytes).sum()
}

#[test]
fn direct_layout_uses_shared_host_mark_rule_and_preserves_overrides() {
    use cuteafd_core::memory_layout::{Category, DeviceKind};
    let dir = snapshot(true);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let mark_bytes = 449_372_160;
    for (host, entries, explicit) in [
        (Some(0), 20, None), (None, 20, None), (Some(32 << 30), 20, None),
        (None, 0, None), (None, 20, Some(4)),
    ] {
        let mut options = with_layout(6, Some(108 << 30));
        let layout = options.layout.as_mut().unwrap();
        layout.rtx_bytes = vec![256 << 30; 2];
        layout.context_tokens = 262_144;
        layout.pool_tokens = Some(524_288);
        layout.concurrency = 8;
        layout.host_cache_bytes = host;
        layout.host_memory = Some((128 << 30, 128 << 30));
        layout.mimo_prefix_entries = entries;
        layout.prefix_slots = explicit;
        let expected = explicit.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(
            layout.concurrency, entries, mark_bytes, layout.mimo_prefix_mark_bytes, host != Some(0)));
        let report = input.plan(&options).unwrap();
        assert!(report.fits, "host={host:?} entries={entries} explicit={explicit:?}: {:?}", report.hints);
        let reserved = device(report.memory_layout.as_ref().unwrap(), DeviceKind::Rtx)
            .into_iter().map(|gpu| category(gpu, Category::Prefix)).sum::<u64>();
        assert_eq!(reserved, expected * mark_bytes,
            "host={host:?} entries={entries} explicit={explicit:?}");
    }
}

#[test]
fn max_card_jointly_selects_tp2_layers_marks_and_complete_spark_budget() {
    use cuteafd_core::memory_layout::{Category, DeviceKind};
    let dir = snapshot(true);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let mut options = with_layout(6, Some(108 << 30));
    let layout = options.layout.as_mut().unwrap();
    layout.host_cache_bytes = None;
    layout.host_memory = Some((128 << 30, 128 << 30));
    layout.concurrency = 8;
    layout.pool_tokens = Some(1_048_576);
    layout.prefix_slots = Some(2);
    layout.context_tokens = 262144;
    let report = input.plan(&options).unwrap();
    assert!(report.fits && report.placement_supported, "{:#?}", report.hints);
    let memory = report.memory_layout.as_ref().unwrap();
    assert_eq!(memory.pool_tokens, 1_048_576);
    assert!(memory.notes.iter().any(|note| note.contains("API max context: 262144")
        && note.contains("compiled max context: 1048576")), "{:?}", memory.notes);
    assert!(memory.notes.iter().any(|note| note.contains("selected RTX expert layers:")
        && note.contains("retained marks:")), "{:?}", memory.notes);
    assert!(memory.notes.iter().any(|note| note.contains("runtime residual")
        && note.contains("typed execution owners")), "{:?}", memory.notes);
    let sparks = device(memory, DeviceKind::Spark);
    assert_eq!(sparks.len(), 6);
    for spark in &sparks {
        assert!(spark.used_bytes() <= 108 << 30);
        assert!(category(spark, Category::Runtime) > 0 && category(spark, Category::Transport) > 0);
    }
    let rtx = device(memory, DeviceKind::Rtx);
    assert_eq!(rtx.len(), 2);
    for gpu in &rtx {
        assert!(category(gpu, Category::Experts) > 0);
        // The Kimi family reserve constant is 0: no reserved item.
        assert_eq!(category(gpu, Category::Reserved), crate::placement::reserve::gpu("kimi_k3"));
    }
    assert_eq!(category(rtx[0], Category::Experts), category(rtx[1], Category::Experts));
    // Numeric overrides are strict, never silently increased to satisfy the workers.
    options.layout.as_mut().unwrap().onboard = Some(crate::placement::OnboardSetting::Layers(0));
    let refused = input.plan(&options).unwrap();
    assert!(!refused.fits);
    // The shared planner names the budget the pin cannot fit instead of the old
    // minimum-layers message: an explicit 0 still refuses rather than clamping.
    assert!(refused.hints.iter().any(|hint| hint.what.contains("memory budget 'spark0'")
        && hint.what.contains("available")), "{:#?}", refused.hints);
    // An impossible Spark ceiling must refuse instead of shrinking the 1M extent.
    options.layout.as_mut().unwrap().onboard = None;
    options.spark_budget_bytes = Some(1);
    let refused = input.plan(&options).unwrap();
    assert!(!refused.fits);
    // Other card columns are refused by name.
    let refused = input.plan(&with_layout(4, None)).unwrap();
    assert!(!refused.fits);
    assert!(refused.hints.iter().any(|h| h.what.contains("budget")), "{:?}", refused.hints);
}

#[test]
#[ignore = "requires CUTEAFD_KIMI_EXL3_SNAPSHOT; reads real headers only in a manifest-free process"]
fn mounted_release_plan_is_manifest_independent_and_serving() -> anyhow::Result<()> {
    let snapshot = std::env::var_os("CUTEAFD_KIMI_EXL3_SNAPSHOT").context("Kimi snapshot required")?;
    ensure!(std::env::var_os("CUTEAFD_PROGRAM_MANIFEST").is_none(), "run with no installed manifest environment");
    let checkpoint = Checkpoint::open(std::path::Path::new(&snapshot))?;
    let stem = format!("{TEXT}layers.1.block_sparse_moe.experts.0.w1");
    let members = checkpoint.tensors.iter().filter(|t| t.meta.name.starts_with(&format!("{stem}.")))
        .cloned().collect::<Vec<_>>();
    let mut operand = format::detect(format::group_by_stem(&members).values().next().context("real routed header")?)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    assert_eq!(operand.exl3_bits(), Some(2));
    assert_eq!(operand.logical, [3072, 3584]);
    let model = Model::open(&checkpoint)?;
    model.accepts(&TensorRole::expert(Component::RoutedExpert, 1, 0), &stem, &mut operand)
        .map_err(anyhow::Error::msg)?;
    assert_eq!(model.experts(&operand).context("K2 package")?.package, PACKAGE);
    drop(checkpoint);
    let mut options = with_layout(6, Some(108 << 30));
    options.vision = MediaMode::Off;
    options.audio = MediaMode::Off;
    options.draft_ownership = Some(spec::DraftOwnership::None);
    let layout = options.layout.as_mut().unwrap();
    layout.onboard = Some(crate::placement::OnboardSetting::Layers(5));
    // Match the serving card's automatic pool, not a forced 2M allocation.
    layout.pool_tokens = None;
    let report = testing::plan(std::path::Path::new(&snapshot), &options)?;
    assert_eq!(report.experts.as_ref().context("deployment expert package")?.package, PACKAGE);
    assert!(report.fits && report.placement_supported && report.wip_launchable(), "{:#?}", report.hints);
    assert!(report.executable(), "{:#?}", report.hints);
    let deployment_options = crate::plan::deployment::DeploymentOptions { rtx_gpus: 2, speculator: "off".into(), ..Default::default() };
    let deployment = crate::plan::deployment::DeploymentSpec::from_report(&report, &deployment_options);
    assert!(deployment.is_ok(), "{deployment:?}");
    Ok(())
}

#[test]
fn cache_geometry_preserves_actual_physical_owners_and_mla_copies() {
    use crate::{placement::AttentionPlacement, serving_capacity::{CacheOptions, KvPlacement}};
    let dir = snapshot(false);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let model = KIMI_K3.open(&checkpoint).unwrap();
    let single = model.cache_geometry(CacheOptions { coordinator_ranks: 1, ..Default::default() }).unwrap().unwrap();
    assert_eq!(single.ranks.len(), 1);
    assert_eq!(single.ranks[0].retained_mark_bytes, 2 * 224_686_080);
    assert_eq!(single.ranks[0].persistent_unit_bytes,
        (24 * 64 * CacheOptions::default().mla_kv.record_bytes()) as u64);
    assert!(model.cache_geometry(CacheOptions { coordinator_ranks: 3, ..Default::default() }).is_err());
    for (attention, format) in [AttentionPlacement::Layers, AttentionPlacement::Heads].into_iter()
        .flat_map(|attention| [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8]
            .map(|format| (attention, format))) {
        let cache = model.cache_geometry(CacheOptions { coordinator_ranks: 2, mla_kv: format,
            attention_placement: Some(attention), ..Default::default() }).unwrap().unwrap();
        assert_eq!(cache.placement, KvPlacement::PartitionedLayers);
        assert_eq!(cache.logical_unit_rows, 64);
        assert_eq!(cache.ranks.len(), 2);
        assert_eq!(cache.ranks[0].persistent_unit_bytes,
            (12 * 64 * format.record_bytes()) as u64 * if attention == AttentionPlacement::Heads { 2 } else { 1 });
        assert_eq!(cache.ranks[0], cache.ranks[1]);
        assert_eq!(cache.ranks[0].retained_mark_bytes, 224_686_080);
    }
    let context = model.cache_geometry(CacheOptions { coordinator_ranks: 2,
        attention_placement: Some(AttentionPlacement::Context), ..Default::default() }).unwrap().unwrap();
    let auto = model.cache_geometry(CacheOptions { coordinator_ranks: 2, ..Default::default() }).unwrap().unwrap();
    assert_eq!(auto, context);
    assert_eq!(context.placement, KvPlacement::PartitionedContext);
    assert_eq!(context.ranks[0].persistent_unit_bytes, single.ranks[0].persistent_unit_bytes / 2);
    assert_eq!(context.ranks[0], context.ranks[1]);
    assert!(model.cache_geometry(CacheOptions { coordinator_ranks: 1,
        attention_placement: Some(AttentionPlacement::Context), ..Default::default() }).is_err());
}

#[test]
fn core_k6_headers_charge_native_bytes_and_admit_native_operands() {
    let dir = tempfile::tempdir().unwrap();
    testing::write_kimi_core_snapshot(dir.path(), false);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let model = Model::open(&checkpoint).unwrap();
    assert_eq!(model.core.projections.len(), 926);
    assert!(model.native_serving_admission());
    assert!(!dir.path().join("quantize_config.json").exists());
    let source: u64 = model.core.projections.values().map(|p| p.source_bytes).sum();
    assert_eq!(source, 40_323_110_520);
    let kv_a = &model.core.projections[&format!("{TEXT}layers.3.self_attn.kv_a_proj_with_mqa")];
    assert_eq!(kv_a.shape, [576, 7168]);
    assert_eq!(kv_a.stored_outputs, 640);
    let original = snapshot(false);
    let bf16 = Model::open(&Checkpoint::open(original.path()).unwrap()).unwrap();
    let weights = |model: &Model, gpu: usize| {
        model.coordinator.split / 2 + model.coordinator.replicated - model.frontends[1 - gpu]
            + model.mla.iter().enumerate().filter(|(layer, _)| model.cfg.attention[*layer] == KimiAttention::Mla)
                .enumerate().filter(|(ordinal, _)| ordinal % 2 == gpu).map(|(_, (_, r))| r.split + r.replicated).sum::<u64>()
            + if gpu == 0 { model.bytes[&Component::Embedding] } else {
                model.core.projections.get("language_model.lm_head").map_or(model.bytes[&Component::LmHead],
                    |p| p.resident_bytes(1).unwrap()) }
    };
    let freed = [weights(&bf16, 0) - weights(&model, 0), weights(&bf16, 1) - weights(&model, 1)];
    assert_eq!(freed, [32_815_041_536, 34_282_705_920]);
    for (stem, projection) in &model.core.projections {
        let members = checkpoint.tensors.iter().filter(|t| format::split_stem(&t.meta.name).0 == stem)
            .cloned().collect::<Vec<_>>();
        let mut operand = format::detect(format::group_by_stem(&members).values().next().unwrap()).unwrap();
        assert_eq!(operand.exl3_bits(), Some(6));
        assert_eq!(operand.logical, [projection.stored_outputs, projection.shape[1]]);
        let role = KIMI_K3.classify(&model.spec, &format!("{stem}.trellis")).unwrap();
        model.accepts(&role, stem, &mut operand).unwrap();
    }
}

#[test]
fn core_index_only_native_inventory_reads_headers_and_admits_native_core() {
    let dir = tempfile::tempdir().unwrap();
    testing::write_kimi_core_snapshot(dir.path(), false);
    let checkpoint = Checkpoint::inventory(dir.path()).unwrap();
    assert!(checkpoint.tensors.is_empty());
    let model = Model::open(&checkpoint).unwrap();
    assert_eq!(model.core.projections.len(), 926);
    assert!(model.native_serving_admission());
    assert_eq!(model.core.projections.values().map(|p| p.source_bytes).sum::<u64>(), 40_323_110_520);
    model.core.validate_runtime_groups(&model.cfg).unwrap();
    let names = files::required_tensors(&checkpoint,
        files::ReadRole::Coordinator { local_experts: false, speculator: false }).unwrap();
    assert!(names.contains("language_model.lm_head.trellis"));
    assert!(!names.contains("language_model.lm_head.weight"));
    // Already-loaded core headers must not reopen a shard on the readiness path.
    let loaded = Checkpoint::open(dir.path()).unwrap();
    for shard in loaded.weight_map.values().collect::<std::collections::BTreeSet<_>>() {
        std::fs::remove_file(dir.path().join(shard)).unwrap();
    }
    assert_eq!(Model::open(&loaded).unwrap().core.projections.len(), 926);
}

#[test]
fn core_index_only_mixed_inventory_keeps_bf16_kda_and_opens_spark_role() {
    let dir = tempfile::tempdir().unwrap();
    let mut mixed = Vec::new();
    for tensor in tensors(false) {
        let native = tensor.0.starts_with("language_model.model.layers.1.block_sparse_moe.shared_experts.")
            || tensor.0.starts_with("language_model.model.layers.1.block_sparse_moe.routed_expert_")
                && !tensor.0.contains("routed_expert_norm");
        if native {
            let stem = tensor.0.strip_suffix(".weight").unwrap();
            mixed.extend(testing::exl3(stem, tensor.2[0], tensor.2[1], 6));
        } else { mixed.push(tensor); }
    }
    testing::write_snapshot(dir.path(), &config(), &mixed, None);
    let checkpoint = Checkpoint::inventory(dir.path()).unwrap();
    assert!(checkpoint.tensors.is_empty());
    let model = Model::open(&checkpoint).unwrap();
    assert_eq!(model.core.projections.len(), 5);
    model.core.validate_runtime_groups(&model.cfg).unwrap();
    assert!(model.core.get_weight("language_model.model.layers.1.self_attn.q_proj.weight").is_none());
    let role = files::ReadRole::Spark { rank: 0, world: 6 };
    let spark = files::open_role(dir.path(), role).unwrap();
    assert!(spark.tensors.iter().all(|tensor| tensor.meta.name.contains(".experts.")));
    assert!(!spark.tensors.is_empty());
    // Spark-only headers still discover core formats from the authoritative index.
    assert_eq!(Model::open(&spark).unwrap().core.projections.len(), 5);
}

#[test]
fn core_header_validation_rejects_wrong_bits_keep_list_and_duplicate_bf16() {
    let dir = tempfile::tempdir().unwrap();
    testing::write_kimi_core_snapshot(dir.path(), false);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    for corruption in ["bits", "duplicate", "keep-list", "missing-scale", "shape", "codebook"] {
        let mut changed = Checkpoint::open(dir.path()).unwrap();
        let stem = format!("{TEXT}layers.0.self_attn.q_proj");
        let trellis = changed.tensors.iter_mut().find(|t| t.meta.name == format!("{stem}.trellis")).unwrap();
        match corruption {
            "bits" => { trellis.meta.shape[2] = 80; trellis.meta.byte_length = trellis.meta.byte_length / 6 * 5; }
            "duplicate" => {
                let mut tensor = checkpoint.tensors[0].clone();
                tensor.meta.name = format!("{stem}.weight");
                changed.weight_map.insert(tensor.meta.name.clone(), tensor.shard.clone());
                changed.tensors.push(tensor);
            }
            "keep-list" => {
                for tensor in &mut changed.tensors {
                    if tensor.meta.name.starts_with(&format!("{stem}.")) {
                        tensor.meta.name = tensor.meta.name.replace("self_attn.q_proj", "self_attn.f_a_proj");
                    }
                }
            }
            "missing-scale" => changed.tensors.retain(|t| t.meta.name != format!("{stem}.svh")),
            "shape" => trellis.meta.shape[0] -= 1,
            "codebook" => {
                let mut marker = changed.tensors.iter().find(|t| t.meta.name == format!("{stem}.mcg")).unwrap().clone();
                marker.meta.name = format!("{stem}.mul1");
                changed.tensors.push(marker);
            }
            _ => unreachable!(),
        }
        assert!(Model::open(&changed).is_err(), "accepted {corruption}");
    }
}

#[test]
fn header_native_k6_core_is_admitted_by_default() -> Result<()> {
    let dir=tempfile::tempdir()?;
    testing::write_kimi_core_snapshot(dir.path(),false);
    let checkpoint=Checkpoint::open(dir.path())?;
    let mut model=Model::open(&checkpoint)?;
    model.configure().map_err(anyhow::Error::msg)?;
    assert!(model.native_serving_admission());
    for (stem,projection) in &model.core.projections {
        let members=checkpoint.tensors.iter().filter(|t|format::split_stem(&t.meta.name).0==stem)
            .cloned().collect::<Vec<_>>();
        let mut operand=format::detect(format::group_by_stem(&members).values().next().unwrap())
            .map_err(|error|anyhow::anyhow!("{error:?}"))?;
        let role=KIMI_K3.classify(&model.spec,&format!("{stem}.trellis")).unwrap();
        model.accepts(&role,stem,&mut operand).map_err(anyhow::Error::msg)?;
        assert_eq!(operand.logical,[projection.stored_outputs,projection.shape[1]]);
    }
    Ok(())
}

#[test]
fn automatic_context_uses_model_maximum_for_every_core() -> Result<()> {
    let k6 = tempfile::tempdir()?;
    testing::write_kimi_core_snapshot(k6.path(), false);
    let bf16 = snapshot(false);
    for dir in [k6.path(), bf16.path()] {
        let checkpoint = Checkpoint::open(dir)?;
        let expected = Model::open(&checkpoint)?.cfg.max_position;
        // The planner's unset context selects the same extent; an explicit one wins.
        let report = testing::plan(dir, &PlanOptions { placement: ExpertPlacement::Sparks { ranks: 6 },
            layout: Some(plan::layout::LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; 2], ..Default::default() }), ..Default::default() })?;
        let notes = report.memory_layout.as_ref().map(|m| m.notes.join("\n")).unwrap_or_default();
        assert!(notes.contains(&format!("Kimi API max context: {expected} tokens")), "{notes}");
        let explicit = testing::plan(dir, &PlanOptions { placement: ExpertPlacement::Sparks { ranks: 6 },
            layout: Some(plan::layout::LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; 2], context_tokens: 262_144, ..Default::default() }),
            ..Default::default() })?;
        let notes = explicit.memory_layout.as_ref().map(|m| m.notes.join("\n")).unwrap_or_default();
        assert!(notes.contains("Kimi API max context: 262144 tokens"), "{notes}");
    }
    Ok(())
}

#[test]
fn k6_max_context_plan_charges_parity_pool_and_each_shared_allocation() -> Result<()> {
    use crate::{families::kimi_k3::{context::{self, ContextLayout}, core::CoreSplit},
        placement::{AttentionPlacement, LayerMode}};
    use cuteafd_core::{MlaCacheFormat, memory_layout::{Category, DeviceKind}};
    let dir = tempfile::tempdir()?;
    testing::write_kimi_core_snapshot(dir.path(), true);
    let checkpoint = Checkpoint::open(dir.path())?;
    let model = Model::open(&checkpoint)?;
    let package = tempfile::tempdir()?;
    let path = installed_inventory(package.path(), false);
    let mut manifest: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    context::add_manifest_programs(&mut manifest);
    let mut policies = std::collections::BTreeMap::new();
    for p in model.core.projections.values() {
        for world in [1, 2] {
            let (mut n, mut k) = (p.stored_outputs, p.shape[1]);
            let side = if world == 1 || p.split != CoreSplit::Input { 1 } else { 2 };
            if world == 2 && p.split != CoreSplit::Whole {
                if side == 1 { n /= 2; } else { k /= 2; }
            }
            let book = match p.codebook {
                crate::formats::exl3_storage::Exl3Codebook::Mcg => "mcg",
                crate::formats::exl3_storage::Exl3Codebook::Mul1 => "mul1",
            };
            policies.insert((k, n, side, book), serde_json::json!({"input_features":k,
                "output_features":n,"sharded_side":side,"codebook":book}));
        }
    }
    manifest["families"] = serde_json::json!({"dense_trellis_k6":{"abi_version":1,
        "row_policies":policies.into_values().collect::<Vec<_>>()}});
    for capacity in [64, 4096] {
        let mut names = vec![format!("kda_core_h48_m{capacity}")];
        for heads in [48, 96] { for stride in [576, 640] { for cache in ["", "_fp8"] {
            names.push(format!("mla_rank_norm_h{heads}_kv{stride}{cache}_m{capacity}"));
        } } }
        manifest["programs"].as_array_mut().unwrap().extend(names.into_iter().map(|name|
            serde_json::json!({"name":name,"scratch_bytes_at_capacity":{"scratch":256}})));
    }
    std::fs::write(&path, manifest.to_string())?;
    let input = testing::BudgetedPlanInput::open(dir.path())?;
    let mut options = with_layout(6, Some(110 << 30));
    let layout = options.layout.as_mut().unwrap();
    layout.workspace_manifest = Some(path);
    layout.context_tokens = 262_144;
    layout.pool_tokens = Some(2_097_152);
    layout.target_pool_tokens = 2_097_152;
    layout.mla_kv = MlaCacheFormat::Fp8;
    layout.prefix_slots = Some(1);
    options.attention_placement = None;
    let auto = input.plan(&options)?;
    assert_eq!(auto.attention_placement, AttentionPlacement::Context);
    options.attention_placement = Some(AttentionPlacement::Layers);
    let layers = input.plan(&options)?;
    assert_eq!(layers.attention_placement, AttentionPlacement::Layers);
    options.attention_placement = Some(AttentionPlacement::Context);
    let report = input.plan(&options)?;
    let placed = report.solver_placement.as_ref().unwrap_or_else(|| panic!("{:?}", report.hints));
    assert_eq!(report.attention_placement, AttentionPlacement::Context);
    assert_eq!(report.memory_layout, auto.memory_layout);
    assert_eq!(report.solver_placement, auto.solver_placement);
    assert_eq!(placed.pool_tokens, layers.solver_placement.as_ref().unwrap().pool_tokens);
    assert_eq!(placed.pool_tokens, 2_097_152);
    assert!(placed.onboard_layers > 0);
    for (layer, assignment) in placed.layers.iter().enumerate() {
        assert_eq!(assignment.mode, if model.cfg.attention[layer] == KimiAttention::Mla {
            LayerMode::ContextSplit } else { LayerMode::HeadSplit });
    }
    let expected = ContextLayout::from_manifest(&manifest, MlaCacheFormat::Fp8, 262_144,
        report.prefill_unit_rows.unwrap() as usize, report.prefill_waves.unwrap() as usize)?.buffers();
    let demands = expected.demands()?;
    let memory = report.memory_layout.as_ref().unwrap();
    for device in memory.devices.iter().filter(|d| d.kind == DeviceKind::Rtx) {
        for demand in demands.iter().filter(|d| u32::from(d.gpu) == device.index) {
            assert_eq!(device.items.iter().filter(|i| i.group == demand.group).map(|i| i.bytes).sum::<u64>(), demand.bytes);
        }
        let kv = device.items.iter().filter(|i| i.category == Category::Kv && i.group == "records")
            .map(|i| i.bytes).sum::<u64>();
        assert_eq!(kv, placed.pool_tokens * model.cfg.mla_layers() as u64 * MlaCacheFormat::Fp8.record_bytes() as u64 / 2);
    }
    assert!(memory.notes.iter().any(|n| n.contains("configured 262144 tokens = 343932928 B")
        && n.contains("compiled cap 1048576 tokens = 1375731712 B")), "{:?}", memory.notes);
    Ok(())
}
