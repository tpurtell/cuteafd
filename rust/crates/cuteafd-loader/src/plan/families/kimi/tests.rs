use super::*;
use crate::plan::{self, checkpoint::Checkpoint, format, testing::{self, t, Tensor as FixtureTensor}, *};
use serde_json::Value;

fn config() -> Value { crate::families::kimi_k3::config::tests::release_config() }

/// The release's tensor headers (sparse payloads). `experts`: routed layers
/// whose 896 experts are written; the rest are omitted from the shard and the
/// index alike (a partial snapshot) unless `all`.
fn tensors(all: bool) -> Vec<FixtureTensor> {
    let cfg = KimiK3Config::from_hf(&config()).unwrap();
    let mut out = Vec::new();
    let p = TEXT;
    out.push(t(format!("{p}embed_tokens.weight"), "BF16", &[163840, 7168]));
    out.push(t(format!("{HEAD}weight"), "BF16", &[163840, 7168]));
    for name in ["norm", "output_attn_res_norm"] { out.push(t(format!("{p}{name}.weight"), "BF16", &[7168])); }
    out.push(t(format!("{p}output_attn_res_proj.weight"), "BF16", &[1, 7168]));
    for layer in 0..93 {
        let b = format!("{p}layers.{layer}");
        for n in ["input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"] {
            out.push(t(format!("{b}.{n}.weight"), "BF16", &[7168]));
        }
        for n in ["self_attention_res_proj", "mlp_res_proj"] { out.push(t(format!("{b}.{n}.weight"), "BF16", &[1, 7168])); }
        let a = format!("{b}.self_attn");
        out.push(t(format!("{a}.g_proj.weight"), "BF16", &[12288, 7168]));
        out.push(t(format!("{a}.o_proj.weight"), "BF16", &[7168, 12288]));
        if cfg.attention[layer] == KimiAttention::Kda {
            for proj in ["q", "k", "v"] {
                out.push(t(format!("{a}.{proj}_proj.weight"), "BF16", &[12288, 7168]));
                out.push(t(format!("{a}.{proj}_conv1d.weight"), "F32", &[12288, 1, 4]));
            }
            out.push(t(format!("{a}.f_a_proj.weight"), "BF16", &[128, 7168]));
            out.push(t(format!("{a}.f_b_proj.weight"), "BF16", &[12288, 128]));
            out.push(t(format!("{a}.b_proj.weight"), "BF16", &[96, 7168]));
            out.push(t(format!("{a}.A_log"), "F32", &[128]));
            out.push(t(format!("{a}.dt_bias"), "F32", &[12288]));
            out.push(t(format!("{a}.o_norm.weight"), "F32", &[128]));
        } else {
            out.push(t(format!("{a}.q_a_proj.weight"), "BF16", &[1536, 7168]));
            out.push(t(format!("{a}.q_a_layernorm.weight"), "BF16", &[1536]));
            out.push(t(format!("{a}.q_b_proj.weight"), "BF16", &[18432, 1536]));
            out.push(t(format!("{a}.kv_a_proj_with_mqa.weight"), "BF16", &[576, 7168]));
            out.push(t(format!("{a}.kv_a_layernorm.weight"), "BF16", &[512]));
            out.push(t(format!("{a}.kv_b_proj.weight"), "BF16", &[24576, 512]));
        }
        if layer == 0 {
            for (proj, n, k) in [("gate", 33792, 7168), ("up", 33792, 7168), ("down", 7168, 33792)] {
                out.push(t(format!("{b}.mlp.{proj}_proj.weight"), "BF16", &[n, k]));
            }
            continue;
        }
        let m = format!("{b}.block_sparse_moe");
        out.push(t(format!("{m}.gate.weight"), "BF16", &[896, 7168]));
        out.push(t(format!("{m}.gate.e_score_correction_bias"), "F32", &[896]));
        out.push(t(format!("{m}.routed_expert_down_proj.weight"), "BF16", &[3584, 7168]));
        out.push(t(format!("{m}.routed_expert_up_proj.weight"), "BF16", &[7168, 3584]));
        out.push(t(format!("{m}.routed_expert_norm.weight"), "BF16", &[3584]));
        for (proj, n, k) in [("gate", 6144, 7168), ("up", 6144, 7168), ("down", 7168, 6144)] {
            out.push(t(format!("{m}.shared_experts.{proj}_proj.weight"), "BF16", &[n, k]));
        }
        if all || layer == 1 {
            for e in 0..896 {
                for (w, n, k) in [("w1", 3072, 3584), ("w3", 3072, 3584), ("w2", 3584, 3072)] {
                    out.extend(testing::exl3(&format!("{m}.experts.{e}.{w}"), n, k, 2));
                }
            }
        }
    }
    out.push(t("vision_tower.encoder.blocks.0.wqkv.weight", "BF16", &[1536, 1024]));
    out.push(t("mm_projector.proj.0.weight", "BF16", &[4096, 4096]));
    out
}

/// One shard per layer, as the release (a single 990k-entry header would
/// exceed the 64 MiB safetensors header bound).
fn snapshot(all: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let tensors = tensors(all);
    let shard = |name: &str| indexed(name.strip_prefix(TEXT).unwrap_or(""), "layers.").map_or(0, |(l, _)| l + 1);
    let mut shards: BTreeMap<usize, Vec<FixtureTensor>> = BTreeMap::new();
    for tensor in tensors { shards.entry(shard(&tensor.0)).or_default().push(tensor); }
    let mut weight_map = serde_json::Map::new();
    for (index, members) in &shards {
        let file = format!("model-{index:05}.safetensors");
        testing::write_safetensors(&dir.path().join(&file), members);
        for (name, ..) in members { weight_map.insert(name.clone(), Value::String(file.clone())); }
    }
    std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&config()).unwrap()).unwrap();
    std::fs::write(dir.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap()).unwrap();
    dir
}

fn with_layout(ranks: usize, budget: Option<u64>) -> PlanOptions {
    PlanOptions { placement: ExpertPlacement::from_spark_ranks(ranks), spark_budget_bytes: budget,
        layout: Some(layout::LayoutOptions { rtx_bytes: vec![crate::placement::inventory::PRO_TOTAL_BYTES; 2],
            spark_allocation_budget_bytes: budget, ..Default::default() }), ..Default::default() }
}

fn installed_inventory(root: &std::path::Path, spark: bool) -> std::path::PathBuf {
    use serde_json::json;
    let lock: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../../third_party/sparkinfer.lock.json"))).unwrap();
    let revision = lock["revision"].as_str().unwrap();
    let mut programs = Vec::new();
    for (name, geometry) in [
        ("kimik3_router_scores", json!({"hidden":7168,"experts":896,"top_k":16})),
        ("kimik3_expert_input_quant", json!({"hidden":3584})),
        ("kimik3_latent_down", json!({"width":7168,"outputs":3584})),
        ("kimik3_latent_up", json!({"width":3584,"outputs":7168})),
        ("kimik3_rmsnorm_h3584", json!({"hidden":3584})),
    ] {
        if !spark { programs.push(json!({"name":name,"geometry":geometry,"object_sha256":"b".repeat(64)})); }
    }
    let path = root.join("PROGRAMS.json");
    std::fs::write(&path, json!({"schema":1,"sparkinfer_revision":revision,"programs":programs}).to_string()).unwrap();
    let package = root.join("exl3/exl3-kimik3-k23");
    let layouts = if spark { (0..6).map(|r| format!("tp6-rank{r}")).collect::<Vec<_>>() }
        else { vec!["rtx-tp1".into()] };
    let mut variants = Vec::new();
    for layout in layouts {
        for capacity in [1, 80, 256, 1024, 4096] {
            let directory = format!("{layout}/m{capacity}");
            let meta = json!({"directory":directory,"capacity":capacity,"intermediate":if spark {512} else {3072},
                "experts":896,"top_k":16,"bits":[2,3],"output_dtype":if spark {"bf16"} else {"fp32"},
                "hidden":3584,"activation":"situ","descriptor_bits":10,"sparkinfer_revision":revision});
            let target = package.join(&directory);
            std::fs::create_dir_all(target.join("routes")).unwrap();
            std::fs::write(target.join("v41_exl3.json"), meta.to_string()).unwrap();
            for file in ["libcuteafd_exl3.so", "routes/libv41_exl3_routes.so", "trellis_lut.bin"] {
                std::fs::write(target.join(file), []).unwrap();
            }
            variants.push(meta);
        }
    }
    std::fs::write(package.join("manifest.json"), json!({"schema":"cuteafd.exl3-package.v1",
        "geometry":"kimik3","role":if spark {"spark"} else {"coordinator"},
        "sparkinfer_revision":revision,"variants":variants}).to_string()).unwrap();
    path
}

#[test]
fn routed_acceptance_requires_the_installed_role_inventory() {
    let snapshot = snapshot(false);
    let checkpoint = Checkpoint::open(snapshot.path()).unwrap();
    let trellis = checkpoint.tensors.iter().filter(|t| t.meta.name.contains(".experts.0.w1.")).cloned().collect::<Vec<_>>();
    let mut operand = format::detect(format::group_by_stem(&trellis).values().next().unwrap()).unwrap();
    let stem = format!("{TEXT}layers.1.block_sparse_moe.experts.0.w1");
    let role = TensorRole::layer(Component::RoutedExpert, 1);
    let mut model = Model::open(&checkpoint).unwrap();
    for spark in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = installed_inventory(dir.path(), spark);
        model.installed_experts = validate_expert_inventory(&path).map_err(|e| format!("{e:#}"));
        model.accepts(&role, &stem, &mut operand).unwrap();
        if spark {
            let meta_path = dir.path().join("exl3/exl3-kimik3-k23/tp6-rank5/m4096/v41_exl3.json");
            let mut meta: Value = serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
            meta["descriptor_bits"] = serde_json::json!(9);
            std::fs::write(&meta_path, meta.to_string()).unwrap();
        } else {
            std::fs::remove_file(dir.path().join("exl3/exl3-kimik3-k23/rtx-tp1/m80/libcuteafd_exl3.so")).unwrap();
        }
        model.installed_experts = validate_expert_inventory(&path).map_err(|e| format!("{e:#}"));
        assert!(model.accepts(&role, &stem, &mut operand).unwrap_err().contains("artifacts unavailable"));
    }
    let dir = tempfile::tempdir().unwrap();
    let path = installed_inventory(dir.path(), false);
    let mut inventory: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    inventory["programs"] = serde_json::json!([]);
    std::fs::write(&path, inventory.to_string()).unwrap();
    assert!(validate_expert_inventory(&path).unwrap_err().to_string().contains("kimik3_router_scores"));
    assert!(validate_expert_inventory(&dir.path().join("absent.json")).is_err());
}

#[test]
fn release_headers_classify_and_name_every_missing_kernel() {
    let dir = snapshot(false);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert_eq!(report.family.as_deref(), Some("kimi_k3"));
    assert_eq!(report.runtime, Some(RuntimeStatus::Planned));
    assert!(report.unclassified.is_empty(), "{:?}", &report.unclassified[..report.unclassified.len().min(5)]);
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    let spec = report.spec.as_ref().unwrap();
    assert_eq!(spec.layers.iter().filter(|l| matches!(l.attention, AttentionKind::Mla { rope: false, output_gate: true })).count(), 24);
    assert_eq!(spec.moe_layers(), 92);
    for component in [Component::Attention, Component::ResidualMix, Component::LatentProjection, Component::Router,
        Component::RoutedExpert, Component::SharedExpert, Component::DenseFfn] {
        let plan = report.components.iter().find(|c| c.component == component).unwrap_or_else(|| panic!("{component:?}"));
        assert_eq!(plan.status, Status::Planned, "{component:?}");
    }
    let experts = report.components.iter().find(|c| c.component == Component::RoutedExpert).unwrap();
    assert_eq!(experts.formats.keys().collect::<Vec<_>>(), vec!["exl3-k2"]);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let model = KIMI_K3.open(&checkpoint).unwrap();
    let trellis = checkpoint.tensors.iter().filter(|t| t.meta.name.contains(".experts.0.w1.")).cloned().collect::<Vec<_>>();
    let operand = format::detect(format::group_by_stem(&trellis).values().next().unwrap()).unwrap();
    let contract = model.experts(&operand).unwrap();
    assert_eq!((contract.package.as_str(), contract.block, contract.spark_worlds.clone()), (PACKAGE, 128, vec![6]));
    assert!(report.hints.iter().any(|h| h.what.contains("Kimi K3 attention") && h.how.contains("dense_mla")));
    // The family's own Spark budget, shown in the report and its rendering.
    assert_eq!(report.spark_budget_bytes, SPARK_BUDGET_BYTES);
    assert!(!report.executable());
}

#[test]
fn context_refuses_the_missing_executor_instead_of_claiming_fit() {
    let dir = snapshot(false);
    let mut options = with_layout(6, None);
    options.attention_placement = Some(crate::placement::AttentionPlacement::Context);
    let report = plan::plan(dir.path(), &options).unwrap();
    assert!(!report.placement_supported && !report.fits && !report.executable());
    let reason = report.config_error.as_deref().unwrap();
    assert!(reason.contains("kimi_k3") && reason.contains("context")
        && reason.contains("executor not implemented"), "{reason}");
    assert!(report.memory_layout.is_none());
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
        let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
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
fn max_card_charges_spark_runtime_and_onboards_layers_as_the_fallback() {
    use cuteafd_core::memory_layout::{Category, DeviceKind};
    let dir = snapshot(true);
    let gib = |b: u64| b as f64 / super::GIB as f64;
    // Default: no routed expert on RTX. Weights fit the 108 GiB family budget,
    // but weights + runtime residual + latent I/O + EXL3 workspace do not fit
    // the device: an honest refusal naming the shortfall.
    let report = plan::plan(dir.path(), &with_layout(6, None)).unwrap();
    let memory = report.memory_layout.as_ref().unwrap();
    let sparks = device(memory, DeviceKind::Spark);
    assert_eq!(sparks.len(), 6);
    assert!((gib(category(sparks[0], Category::Experts)) - 107.544).abs() < 0.01);
    assert!(category(sparks[0], Category::Runtime) > 0 && category(sparks[0], Category::Transport) > 0);
    assert!(!report.hints.iter().any(|h| h.what.contains("over the 108.00 GiB Spark budget")), "{:#?}", report.hints);
    assert!(report.hints.iter().any(|h| h.what.starts_with("spark0 full memory layout needs")), "{:#?}", report.hints);
    assert_eq!(report.attention_placement, crate::placement::AttentionPlacement::Layers);
    let rtx = device(memory, DeviceKind::Rtx);
    for gpu in &rtx {
        assert_eq!(category(gpu, Category::Experts), 0);
        assert!(gib(category(gpu, Category::Weights) + category(gpu, Category::Embedding)) > 55.0);
        assert_eq!(category(gpu, Category::Reserved), 2 * super::GIB);
        assert!(gpu.free_bytes() >= 0);
        assert_eq!(gpu.items.iter().find(|i| i.group == "layers").unwrap().bytes, 12 * 464_392_192);
    }
    assert_eq!(rtx[0].used_bytes(), rtx[1].used_bytes());
    assert_eq!(category(rtx[0], Category::Kv), category(rtx[1], Category::Kv));
    for gpu in 0..2 {
        let owners: Vec<_> = (0..24).filter(|ordinal| ordinal % 2 == gpu)
            .map(|ordinal| if ordinal == 23 { 92 } else { ordinal * 4 + 3 }).collect();
        assert!(memory.notes.iter().any(|n| n.starts_with(&format!("RTX{gpu}: owned MLA layers {owners:?};"))), "{:?}", memory.notes);
    }
    // Explicit heads keeps both full latent caches and replicates only low-rank operands.
    let mut heads = with_layout(6, None);
    heads.attention_placement = Some(crate::placement::AttentionPlacement::Heads);
    heads.layout.as_mut().unwrap().pool_tokens = Some(262_144);
    let heads_report = plan::plan(dir.path(), &heads).unwrap();
    assert_eq!(heads_report.attention_placement, crate::placement::AttentionPlacement::Heads);
    let head_gpus = device(heads_report.memory_layout.as_ref().unwrap(), DeviceKind::Rtx);
    for gpu in 0..2 {
        assert_eq!(category(head_gpus[gpu], Category::Weights) - category(rtx[gpu], Category::Weights), 363_380_736);
        assert_eq!(head_gpus[gpu].items.iter().find(|i| i.group == "layers").unwrap().bytes, 5_936_087_040);
    }
    let mut owned = with_layout(6, None);
    owned.attention_placement = Some(crate::placement::AttentionPlacement::Layers);
    owned.layout.as_mut().unwrap().pool_tokens = Some(262_144);
    let owned_report = plan::plan(dir.path(), &owned).unwrap();
    let owned_gpus = device(owned_report.memory_layout.as_ref().unwrap(), DeviceKind::Rtx);
    for gpu in 0..2 {
        assert_eq!(category(head_gpus[gpu], Category::Kv) - category(owned_gpus[gpu], Category::Kv), 12 * 1152 * 262_144);
    }
    // Four onboarded layers (two per GPU, 6.92 GiB each) bring the Sparks under the device and keep >= 256K KV.
    let mut options = with_layout(6, None);
    options.layout.as_mut().unwrap().onboard = Some(crate::placement::Onboard::Layers(4));
    let report = plan::plan(dir.path(), &options).unwrap();
    let memory = report.memory_layout.as_ref().unwrap();
    assert!(!report.hints.iter().any(|h| h.what.contains("full memory layout needs")), "{:#?}", report.hints);
    for rtx in device(memory, DeviceKind::Rtx) { assert!((gib(category(rtx, Category::Experts)) - 13.848).abs() < 0.01); }
    assert!((gib(category(device(memory, DeviceKind::Spark)[0], Category::Experts)) - 102.869).abs() < 0.01);
    assert!(memory.pool_tokens >= 262_144, "pool {}", memory.pool_tokens);
    // The shared 100 GiB budget refuses with the exact per-rank shortfall.
    let report = plan::plan(dir.path(), &with_layout(6, Some(100 << 30))).unwrap();
    assert!(!report.fits);
    assert!(report.hints.iter().any(|h| h.what.contains("107.54 GiB/rank, 7.54 GiB over the 100.00 GiB")), "{:#?}", report.hints);
    // Explicit ceiling replaces the old 108 GiB device estimate too; runtime
    // remains unqualified even when this offline ledger fits.
    let report = plan::plan(dir.path(), &with_layout(6, Some(112 << 30))).unwrap();
    assert_eq!(report.spark_budget_bytes, 112 << 30);
    let memory = report.memory_layout.as_ref().unwrap();
    assert!(device(memory, DeviceKind::Spark).iter().all(|d| d.capacity_bytes == 112 << 30 && d.free_bytes() > 0));
    assert!(!report.hints.iter().any(|h| h.what.contains("full memory layout needs")));
    assert!(!report.fits && !report.placement_supported, "header fit is not W9 qualification");
    assert!(memory.notes.iter().any(|n| n.contains("offline estimates")));
    // Other card columns are refused by name.
    let report = plan::plan(dir.path(), &with_layout(4, None)).unwrap();
    assert!(report.hints.iter().any(|h| h.what.contains("6 Spark ranks")));
}
