use super::*;
use crate::plan::{self, testing, Component, ExpertPlacement, PlanOptions, Status};

/// Routed experts of a dense-boundary layer, a NoPE layer, a shared-KV reader
/// and the last layer: enough to exercise every routed naming path.
const EXPERT_LAYERS: &[usize] = &[1, 3, 39, 71];

fn snapshot(exl3: bool, layers: Option<&[usize]>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    testing::write_mistral_large4_snapshot(dir.path(), layers, exl3);
    dir
}

fn component(report: &plan::PlanReport, component: Component) -> &plan::ComponentPlan {
    report.components.iter().find(|c| c.component == component).unwrap()
}

#[test]
fn official_fp8_release_inventory_is_complete_and_classified() {
    // The full release header set (56,563 tensors, @f12dabdc) from the templates.
    let tensors = testing::mistral_large4_tensors(None, false);
    assert_eq!(tensors.len(), 56_563);
    let bytes: u64 = tensors.iter().map(|(_, dtype, shape)| shape.iter().product::<usize>() as u64
        * match *dtype { "BF16" => 2, "F32" => 4, _ => 1 }).sum();
    assert_eq!(bytes, 1_060_548_427_328);
    let dir = snapshot(false, Some(EXPERT_LAYERS));
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert_eq!(report.family.as_deref(), Some("mistral_large4"));
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert!(report.unclassified.is_empty(), "{:?}", &report.unclassified[..report.unclassified.len().min(5)]);
    for c in &report.components {
        // The FP8 release is the quantization source: the Spark service names it unserved.
        let expected = usize::from(c.component == Component::RoutedExpert);
        assert_eq!(c.rejected, expected, "{:?}: {:?}", c.component, c.rejections);
        assert_eq!(c.status, if matches!(c.component, Component::Vision | Component::Speculator) { Status::Disabled }
            else { Status::Planned }, "{:?}", c.component);
    }
    assert!(component(&report, Component::RoutedExpert).rejections[0].reason.contains("EXL3 K4 + Core-K6"));
    // Official FP8 core forms as the reference loads them.
    let attention = component(&report, Component::Attention);
    assert_eq!(attention.formats.get("fp8-block128x128/f32"), Some(&144), "{:?}", attention.formats);
    assert_eq!(component(&report, Component::Indexer).formats.get("fp8-block128x128/f32"), Some(&44));
    assert_eq!(component(&report, Component::RoutedExpert).formats.get("fp8-block128x128/f32"), Some(&(4 * 128 * 3)));
    // Planned family: an inventory, not a serving fit.
    assert!(!report.fits && !report.executable());
}

#[test]
fn exl3_publication_form_passes_the_contract() {
    // Every routed layer: the expert catalog requires the complete storage map.
    let dir = snapshot(true, None);
    let report = testing::BudgetedPlanInput::open(dir.path()).unwrap().plan(&PlanOptions::default()).unwrap();
    assert!(report.unclassified.is_empty());
    for c in &report.components {
        assert_eq!(c.rejected, 0, "{:?}: {:?}", c.component, c.rejections);
    }
    let attention = component(&report, Component::Attention);
    // wq + wo on 72 layers, wkv_a on 63 owners; wkv_b stays BF16.
    assert_eq!(attention.formats.get("exl3-k6"), Some(&(72 * 2 + 63)), "{:?}", attention.formats);
    assert_eq!(component(&report, Component::SharedExpert).formats.get("exl3-k6"), Some(&(71 * 3)));
    assert_eq!(component(&report, Component::DenseFfn).formats.get("exl3-k6"), Some(&3));
    assert_eq!(component(&report, Component::LmHead).formats.get("exl3-k6"), Some(&1));
    assert_eq!(component(&report, Component::RoutedExpert).formats.get("exl3-k4"), Some(&(71 * 128 * 3)));
    let experts = report.experts.as_ref().expect("an expert package contract");
    assert_eq!(experts.package, PACKAGE);
    assert_eq!(experts.spark_worlds, [2, 3, 4, 6]);
}

#[test]
fn exl3_expert_catalog_reads_consolidated_naming() {
    let dir = snapshot(true, None);
    let catalog = crate::read_expert_catalog(dir.path()).unwrap();
    let manifest = catalog.exl3().unwrap();
    let shape = catalog.routed_experts();
    assert_eq!((shape.first_layer, shape.layers, shape.experts), (1, 72, 128));
    assert_eq!(manifest.projections.len(), 71 * 128 * 3);
    assert_eq!(manifest.decoder_tiers(), [4, 5]);
    assert!(manifest.projections.contains_key("layers.71.experts.127.w2"));
}

#[test]
fn contract_names_recipe_drift() {
    let dir = tempfile::tempdir().unwrap();
    let mut tensors = testing::mistral_large4_tensors(Some(&[1]), true);
    // A K6 wkv_b (the absorbed MLA needs BF16 rows) and a K5 shared expert.
    tensors.retain(|t| !t.0.starts_with("layers.4.attention.wkv_b.") && !t.0.starts_with("layers.5.shared_experts.w1."));
    tensors.extend(testing::exl3("layers.4.attention.wkv_b", 12288, 512, 6));
    tensors.extend(testing::exl3("layers.5.shared_experts.w1", 6144, 6144, 5));
    let (compact, manifest) = testing::mistral_large4_exl3_config(&tensors);
    let mut config = crate::families::mistral_large4::config::tests::official();
    config["quantization_config"] = compact;
    // write_snapshot writes the standard index name.
    config.as_object_mut().unwrap().remove("transformers_weights");
    testing::write_quantize_config(dir.path(), &manifest);
    testing::write_snapshot(dir.path(), &config, &tensors, None);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    let attention = component(&report, Component::Attention);
    assert_eq!(attention.rejected, 1);
    assert!(attention.rejections[0].reason.contains("expected BF16 [12288, 512]"), "{:?}", attention.rejections);
    let shared = component(&report, Component::SharedExpert);
    assert_eq!(shared.rejected, 1);
    assert!(shared.rejections[0].reason.contains("K6 MCG core or the official E4M3"), "{:?}", shared.rejections);
}

#[test]
fn spec_carries_nope_kv_sharing_and_indexer_schedule() {
    let dir = snapshot(false, Some(&[1]));
    let report = plan::plan(dir.path(), &PlanOptions { placement: ExpertPlacement::Sparks { ranks: 6 }, ..Default::default() }).unwrap();
    let spec = report.spec.as_ref().unwrap();
    assert_eq!(spec.layers.len(), 72);
    assert_eq!(spec.layers[3].rope, None);
    assert!(spec.layers[4].rope.is_some());
    assert_eq!(spec.layers[17].attention, AttentionKind::MlaDsa { indexer: false });
    assert_eq!(spec.layers[0].ffn, FfnKind::Dense { intermediate: 24576 });
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.top_k, moe.intermediate, moe.scoring.as_str()), (128, 4, 6144, "softmax"));
    assert!(spec.notes.iter().any(|n| n.contains("readers 39<-35 43<-35")), "{:?}", spec.notes);
    let launch = report.launch.as_ref().unwrap();
    assert_eq!(launch.line(), "mistral_large4 mistral_large4 1 71");
    let cache = report.cache_requirements.as_ref().unwrap();
    // 54 RoPE owners x 656 + 9 NoPE owners x 528 (the other 9 NoPE layers read
    // layer 35's latent) + 44 indexers x 132 bytes per token.
    let unit = cache.target_only_layouts[0].ranks[0].persistent_unit_bytes;
    assert_eq!(unit, 64 * (54 * 656 + 9 * 528 + 44 * 132));
}

#[test]
fn declared_index_file_is_the_inventory() {
    let dir = snapshot(false, Some(&[1]));
    assert_eq!(crate::plan::checkpoint::index_name(dir.path()).unwrap(), "consolidated.safetensors.index.json");
    let manifest = crate::plan::files::manifest(dir.path(),
        crate::plan::files::ReadRole::Coordinator { local_experts: false, speculator: false }).unwrap();
    assert!(manifest.files.iter().any(|f| f.path == "consolidated.safetensors.index.json"));
    let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.path().join("config.json")).unwrap()).unwrap();
    for bad in ["../x.index.json", "model.safetensors", ".hidden.index.json"] {
        config["transformers_weights"] = bad.into();
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        assert!(crate::plan::checkpoint::index_name(dir.path()).is_err(), "{bad}");
    }
}

fn layout_options(ranks: usize, gpus: usize) -> PlanOptions {
    PlanOptions {
        placement: ExpertPlacement::from_spark_ranks(ranks),
        mapped_scale_host_budget_bytes: Some(64 << 30),
        layout: Some(plan::layout::LayoutOptions {
            rtx_bytes: vec![crate::placement::inventory::PRO_TOTAL_BYTES; gpus],
            host_memory: Some((128 << 30, 128 << 30)), host_cache_bytes: Some(0), prefix_slots: Some(0),
            glm5_drafter_disabled: true, ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn exl3_publication_lays_out_on_the_glm_engine() {
    let dir = snapshot(true, None);
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    for (ranks, gpus) in [(6, 1), (6, 2), (4, 2)] {
        let report = input.plan(&layout_options(ranks, gpus)).unwrap();
        let memory = report.memory_layout.as_ref().expect("a memory layout");
        assert!(!memory.devices.is_empty(), "{ranks}+{gpus}: {:?}", report.hints);
        assert_eq!(report.prefill_waves, Some(3), "{ranks}+{gpus}");
        let rtx = memory.devices.iter().filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx).count();
        assert_eq!(rtx, gpus);
        // No head-split programs: whole attention, one GPU's heads or two GPUs' layer owners.
        assert_eq!(report.attention_placement, if gpus == 2 { crate::placement::AttentionPlacement::Layers }
            else { crate::placement::AttentionPlacement::Heads }, "{ranks}+{gpus}");
        let groups: std::collections::BTreeSet<_> = memory.devices[0].items.iter().map(|item| item.group.clone()).collect();
        for group in ["attention", "indexer", "embedding", "lm_head"] {
            assert!(groups.iter().any(|g| g.contains(group)), "{ranks}+{gpus}: no {group} in {groups:?}");
        }
        println!("ml4 {ranks} Sparks + {gpus} RTX: pool {} tokens, onboard {:?}, fits {}",
            memory.pool_tokens, report.onboard_layers, report.fits);
        // Before the build: the K6 core charges need the built PROGRAMS.json and
        // the Spark scratch its package census; the planner names both.
        assert!(memory.notes.iter().any(|n| n.contains("no calibrated Spark EXL3 scratch for mistral_large4")));
    }
}
