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

#[test]
fn max_card_places_experts_at_the_family_budget_and_refuses_the_shared_default() {
    let dir = snapshot(true);
    let report = plan::plan(dir.path(), &with_layout(6, None)).unwrap();
    let memory = report.memory_layout.as_ref().unwrap();
    let sparks: Vec<_> = memory.devices.iter().filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Spark).collect();
    assert_eq!(sparks.len(), 6);
    let per_rank = sparks[0].items.iter().map(|i| i.bytes).sum::<u64>();
    // 107.54 GiB: 1/6 of trellis and 3072-wide rotations, whole 3584-wide rotations.
    assert!((per_rank as f64 / super::GIB as f64 - 107.544).abs() < 0.01, "{per_rank}");
    assert!(per_rank <= SPARK_BUDGET_BYTES);
    assert!(!report.hints.iter().any(|h| h.what.contains("over the")), "{:#?}", report.hints);
    // No routed expert is resident on the coordinators by default.
    for rtx in memory.devices.iter().filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx) {
        assert!(rtx.items.iter().all(|i| i.category != cuteafd_core::memory_layout::Category::Experts));
    }
    assert!(memory.pool_tokens >= 262_144, "pool {}", memory.pool_tokens);
    // The shared 100 GiB budget refuses with the exact per-rank shortfall.
    let report = plan::plan(dir.path(), &with_layout(6, Some(100 << 30))).unwrap();
    assert!(!report.fits);
    assert!(report.hints.iter().any(|h| h.what.contains("107.54 GiB/rank, 7.54 GiB over the 100.00 GiB")), "{:#?}", report.hints);
    // Other card columns are refused by name.
    let report = plan::plan(dir.path(), &with_layout(4, None)).unwrap();
    assert!(report.hints.iter().any(|h| h.what.contains("6 Spark ranks")));
}
