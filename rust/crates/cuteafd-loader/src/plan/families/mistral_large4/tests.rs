use super::*;
use crate::plan::{self, testing, Component, ExpertPlacement, MediaMode, PlanOptions, Status};

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
        // The engine serves the EXL3 publication: the official FP8 release's routed experts
        // have no kernel; every other component, the BF16 Pixtral tower included, is ready.
        assert_eq!(c.status, if c.component == Component::Speculator { Status::Disabled }
            else if c.component == Component::RoutedExpert { Status::MissingKernel } else { Status::Ready },
            "{:?}", c.component);
    }
    assert!(component(&report, Component::RoutedExpert).rejections[0].reason.contains("EXL3 K4 + Core-K6"));
    // Official FP8 core forms as the reference loads them.
    let attention = component(&report, Component::Attention);
    assert_eq!(attention.formats.get("fp8-block128x128/f32"), Some(&144), "{:?}", attention.formats);
    assert_eq!(component(&report, Component::Indexer).formats.get("fp8-block128x128/f32"), Some(&44));
    assert_eq!(component(&report, Component::RoutedExpert).formats.get("fp8-block128x128/f32"), Some(&(4 * 128 * 3)));
    // The official FP8 release is the quantization source, never served: not executable.
    assert!(!report.executable());
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
    // The width-class package range (spark-package-ranges.json `ml4:exl3`).
    assert_eq!(experts.spark_worlds, [5, 6, 7, 8]);
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
        // auto: one GPU's heads, or two GPUs' whole-layer owners until context measures faster.
        assert_eq!(report.attention_placement, if gpus == 2 { crate::placement::AttentionPlacement::Layers }
            else { crate::placement::AttentionPlacement::Heads }, "{ranks}+{gpus}");
        let groups: std::collections::BTreeSet<_> = memory.devices[0].items.iter().map(|item| item.group.clone()).collect();
        for group in ["attention", "indexer", "embedding", "lm_head"] {
            assert!(groups.iter().any(|g| g.contains(group)), "{ranks}+{gpus}: no {group} in {groups:?}");
        }
        println!("ml4 {ranks} Sparks + {gpus} RTX: pool {} tokens, onboard {:?}, fits {}",
            memory.pool_tokens, report.onboard_layers, report.fits);
        // The Spark scratch is its TP6 package census (spark_slot_bytes), never unknown.
        assert!(!memory.notes.iter().any(|n| n.contains("no calibrated Spark EXL3 scratch")), "{:?}", memory.notes);
    }
}


#[test]
fn bundled_drafter_is_admitted_on_the_glm_layout() {
    let dir = snapshot(true, None);
    // The planner's GLM drafter admission reads the release's own params.json mtp
    // block (never the GLM DFlash2 snapshot).
    let layout = crate::placement::families::glm5::ml4_draft_layout(dir.path(), 20, 16, 1 << 20, 188).unwrap();
    let bytes = layout.resident_bytes().unwrap();
    // BF16 weights (3 dense MLA layers + target projection + the 204800 x 512 Markov tables) and arenas.
    assert!(bytes > 4 << 30 && bytes < 8 << 30, "{bytes}");
    assert_eq!((layout.capacity().block_rows, layout.target_taps()), (48, &[1, 18, 36, 54, 71][..]));
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let mut options = layout_options(6, 1);
    options.layout.as_mut().unwrap().glm5_drafter_disabled = false;
    let report = input.plan(&options).unwrap();
    assert!(!report.hints.iter().any(|h| h.what.contains("drafter config unavailable")), "{:?}", report.hints);
    // SPECULATOR=dspark owns the checkpoint's own mtp_dspark.* tensors; off leaves them disabled.
    for (speculator, status) in [("dspark", Status::Ready), ("off", Status::Disabled)] {
        let mut options = layout_options(6, 1);
        options.draft_ownership = Some(plan::spec::DraftOwnership::selected("mistral_large4", speculator));
        let report = input.plan(&options).unwrap();
        assert_eq!(component(&report, Component::Speculator).status, status, "{speculator}");
    }
}

/// The complete header witness for plan matrices: every release tensor (56,563, all 72 layers'
/// experts) shaped to the EXL3 K4 + Core-K6 contract, as sparse safetensors plus config.json,
/// quantize_config.json and params.json. Writes into `CUTEAFD_ML4_WITNESS_DIR` (must be empty).
#[test]
#[ignore = "writes the full ML4 EXL3 header witness to CUTEAFD_ML4_WITNESS_DIR"]
fn write_header_witness() {
    let dir = std::path::PathBuf::from(std::env::var("CUTEAFD_ML4_WITNESS_DIR").expect("CUTEAFD_ML4_WITNESS_DIR"));
    std::fs::create_dir_all(&dir).unwrap();
    assert!(std::fs::read_dir(&dir).unwrap().next().is_none(), "{} is not empty", dir.display());
    testing::write_mistral_large4_snapshot(&dir, None, true);
    let checkpoint = crate::plan::Checkpoint::open(&dir).unwrap();
    assert!(checkpoint.missing_shards.is_empty());
    assert_eq!(checkpoint.tensors.len(), testing::mistral_large4_tensors(None, true).len());
}

#[test]
#[ignore = "prints the RTX resident accounting of the EXL3 K4 + Core-K6 publication form"]
fn print_resident_fit() {
    let dir = snapshot(true, None);
    let checkpoint = crate::plan::Checkpoint::open(dir.path()).unwrap();
    for ranks in [1, 2] {
        let groups = crate::plan::layout::glm5::resident_owned(&checkpoint, ranks, 72, None).unwrap();
        for (rank, items) in groups.iter().enumerate() {
            let total: u64 = items.iter().map(|(_, _, b)| b).sum();
            let mut by: std::collections::BTreeMap<String, u64> = Default::default();
            for (group, format, bytes) in items { *by.entry(format!("{group}/{format}")).or_default() += bytes; }
            println!("ranks={ranks} rank{rank}: total {:.2} GiB {:?}", total as f64 / (1u64 << 30) as f64,
                by.iter().map(|(k, v)| format!("{k} {:.2}", *v as f64 / (1u64 << 30) as f64)).collect::<Vec<_>>());
        }
    }
}

/// TP2 RTX onboarding reads Mistral Large 4's own geometry (6144 / 6144 / 128 / top-4), its
/// `exl3-ml4-k45` package and its consolidated router names (`layers.L.gate`, no bias),
/// never GLM's constants, package or `mlp.gate.e_score_correction_bias`.
#[test]
fn tp2_onboarding_inventory_reads_the_checkpoint_geometry_package_and_router() {
    use crate::placement::families::glm5 as glm;
    // The catalog reads every routed layer's headers; the inventory walks the first two.
    let dir = snapshot(true, None);
    let checkpoint = crate::plan::Checkpoint::open(dir.path()).unwrap();
    let cfg = crate::families::glm5::GlmDsaConfig::from_hf(&checkpoint.config).unwrap();
    let catalog = crate::read_expert_catalog(dir.path()).unwrap();
    let shape = catalog.routed_experts().clone();
    assert_eq!((shape.hidden, shape.intermediate, shape.experts, shape.topk), (6144, 6144, 128, 4));
    let tiers = catalog.exl3().unwrap().decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let root = tempfile::tempdir().unwrap();
    let manifest = root.path().join("PROGRAMS.json");
    std::fs::write(&manifest, b"{\"programs\":[]}").unwrap();
    let write_package = |family: &str| for capacity in cuteafd_core::EXL3_CAPACITIES {
        let package = root.path().join(format!("exl3/exl3-{family}-k{tiers}/rtx-tp2/m{capacity}"));
        std::fs::create_dir_all(package.join("routes")).unwrap();
        std::fs::write(package.join("v41_exl3.json"), serde_json::to_vec(&serde_json::json!({
            "hidden": 6144, "intermediate": 3072, "experts": 128, "top_k": 4, "capacity": capacity,
            "bits": catalog.exl3().unwrap().decoder_tiers(), "output_dtype": "fp32", "input_format": "e4m3_k32",
            "direct": false, "route_preparation": {"manifest": "routes/v41_exl3_routes.json"},
            "trellis_lut": {"file": "lut.bin", "bytes": 32}, "buffers": {
                "scratch": {"allocation": "scratch", "bytes": capacity * 64, "dtype": "f32", "zero_on_create": false},
                "state": {"allocation": "state", "bytes": 0, "dtype": "i32", "zero_on_create": true}}})).unwrap()).unwrap();
        for file in ["libcuteafd_exl3.so", "lut.bin", "routes/v41_exl3_routes.json", "routes/libv41_exl3_routes.so"] {
            std::fs::write(package.join(file), b"").unwrap();
        }
    };
    // A GLM-named package is not this checkpoint's.
    write_package("glm");
    let missing = glm::local_inventory(&cfg, &catalog, &manifest, 3, 64, 3, false).unwrap_err();
    assert!(missing.to_string().contains(&format!("exl3-ml4-k{tiers}/rtx-tp2")), "{missing:#}");
    write_package("ml4");
    let local = glm::local_inventory(&cfg, &catalog, &manifest, 3, 64, 3, false).unwrap();
    assert!(local.package.ends_with(format!("exl3-ml4-k{tiers}/rtx-tp2")));
    assert_eq!(local.experts.len(), 2);
    let exl3 = catalog.exl3().unwrap();
    for (index, layer) in [1, 2].into_iter().enumerate() {
        let half = |rank| exl3.residency(crate::V41Exl3Layer::Backbone(layer), 2, rank).unwrap()
            .device_arena_layout().unwrap().1 as u64;
        // Rank 1 holds the router: BF16 [128, 6144], no correction bias.
        assert_eq!(local.experts[index].half[0].resident, half(0));
        assert_eq!(local.experts[index].half[1].resident, half(1) + 128 * 6144 * 2);
    }
    // The catalog and config must agree on the routed geometry.
    let mut other = cfg.clone();
    other.topk = 8;
    assert!(glm::local_inventory(&other, &catalog, &manifest, 3, 64, 3, false).is_err());
}

/// The Pixtral tower on the 2-RTX column (full release headers, native K6
/// programs): wherever it resolves to an RTX (`--vision rtx`, or auto when no
/// Spark holds tower + scratch + 1 GiB, as at n=5) it is reserved before the
/// pool and onboard solve and the plan holds, never a refused encoder; where a
/// Spark holds it (auto at n=6..8) the RTX working set is untouched. That the
/// reserved bytes shed onboarded layers is the solver's own contract
/// (`placement::families::glm5` `reserved_tower_sheds_onboarded_layers`); this
/// manifest carries no local expert package, so nothing onboards here.
#[test]
fn vision_on_two_rtx_reserves_the_tower_before_the_solve() {
    use crate::plan::encoder::EncoderKind;
    let dir = snapshot(true, None);
    let checkpoint = crate::plan::Checkpoint::open(dir.path()).unwrap();
    let cfg = crate::families::glm5::GlmDsaConfig::from_hf(&checkpoint.config).unwrap();
    let core = crate::families::glm5::core::CoreInventory::from_checkpoint(&cfg, &checkpoint).unwrap();
    let manifest = dir.path().join("PROGRAMS.json");
    std::fs::write(&manifest, testing::glm_k6_programs(&cfg, &core).unwrap().to_string()).unwrap();
    let input = testing::BudgetedPlanInput::open(dir.path()).unwrap();
    let plan = |ranks: usize, vision: MediaMode| {
        let mut options = layout_options(ranks, 2);
        options.vision = vision;
        options.layout.as_mut().unwrap().workspace_manifest = Some(manifest.clone());
        input.plan(&options).unwrap()
    };
    let tower = crate::families::mistral_large4::vision::resident_weight_bytes()
        + crate::plan::layout::mistral_scratch_bytes(55 * 55);
    for ranks in 5..=8 {
        let off = plan(ranks, MediaMode::Off);
        assert!(off.placement_supported, "n={ranks}: {:?}", off.hints);
        let rtx_used = |report: &plan::PlanReport| report.memory_layout.as_ref().unwrap().devices.iter()
            .filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx).map(|d| d.used_bytes()).sum::<u64>();
        for vision in [MediaMode::Rtx(None), MediaMode::Auto] {
            let report = plan(ranks, vision);
            assert!(report.placement_supported, "n={ranks} {vision:?}: {:?}", report.hints);
            let encoder = report.encoder.as_ref().expect("an encoder placement");
            let on_rtx = matches!(encoder.kind, EncoderKind::Rtx { .. });
            assert_eq!(on_rtx, vision != MediaMode::Auto || ranks == 5, "n={ranks} {vision:?}: {:?}", encoder.kind);
            assert_eq!(encoder.weights + encoder.scratch, tower);
            let memory = report.memory_layout.as_ref().unwrap();
            assert!(memory.devices.iter().all(|d| d.free_bytes() >= 0), "n={ranks} {vision:?}");
            if on_rtx {
                // Reserved first: the pool gave way, the tower was not refused.
                assert!(memory.devices.iter().any(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx
                    && d.items.iter().any(|i| i.group == "vision tower")));
                assert!(memory.pool_tokens <= off.memory_layout.as_ref().unwrap().pool_tokens);
            } else {
                assert_eq!(rtx_used(&report), rtx_used(&off), "n={ranks}: a Spark-hosted tower leaves the RTX set");
            }
            println!("ml4 n={ranks} 2 RTX {vision:?}: encoder {:?}, pool {} (off {})", encoder.kind,
                memory.pool_tokens, off.memory_layout.as_ref().unwrap().pool_tokens);
        }
    }
}
