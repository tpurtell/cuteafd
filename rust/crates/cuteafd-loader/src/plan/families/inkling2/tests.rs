use super::*;
use crate::plan::{self, testing, PlanOptions};

/// Which routed-expert storage the fixture writes.
#[derive(Clone, Copy, PartialEq)]
enum Fixture { Mxfp8, Mixed, Exl3 }

/// A four-layer release-shaped snapshot (layers 0..3 local, layer 3 global so
/// both attention kinds, both conv kinds and a three-vector residual query
/// appear). Expert storage follows `fixture` per layer, as the three target
/// checkpoints store them.
fn fixture(kind: Fixture) -> (tempfile::TempDir, serde_json::Value, Vec<testing::Tensor>) {
    let dir = tempfile::tempdir().unwrap();
    let mut small = crate::families::inkling2::config::tests::release_config();
    small["text_config"]["num_hidden_layers"] = 4.into();
    small["text_config"]["local_layer_ids"] = serde_json::json!([0, 1, 2]);
    small["text_config"]["n_routed_experts"] = 4.into();
    small["text_config"]["num_experts_per_tok"] = 2.into();
    small["text_config"]["n_layers_per_attn_res_block"] = 2.into();
    small["mtp_config"]["target_layer_ids"] = serde_json::json!([1, 4]);
    let c = Inkling2Config::from_hf(&small).unwrap();
    let (h, d, v, e, i, k) = (c.hidden, c.head_dim, c.vocab_size, c.experts, c.intermediate, c.sconv_kernel);
    let mut tensors = Vec::new();
    let mut plain = |name: String, dtype: &'static str, shape: Vec<usize>| tensors.push((name, dtype, shape));
    plain(format!("{TEXT}embed.weight"), "BF16", vec![v, h]);
    plain(format!("{TEXT}unembed.weight"), "BF16", vec![v, h]);
    plain(format!("{TEXT}embed_norm.weight"), "BF16", vec![h]);
    plain(format!("{TEXT}norm.weight"), "BF16", vec![h]);
    plain(format!("{TEXT}attn_res.output_query"), "F32", vec![h]);
    for layer in 0..4 {
        let local = c.local[layer];
        let p = format!("{TEXT}layers.{layer}.");
        for (name, n, kk) in [("wq_du", c.heads * d, h), ("wr_du", c.heads * c.d_rel, h), ("wk_dv", c.kv_heads * d, h),
            ("wv_dv", c.kv_heads * d, h), ("gate_attn", c.heads * d, h), ("attn_sink_proj", c.heads, h), ("wo_ud", h, c.heads * d)] {
            plain(format!("{p}attn.{name}.weight"), "BF16", vec![n, kk]);
        }
        plain(format!("{p}attn.q_l2_clip.scale"), "BF16", vec![c.heads, d]);
        plain(format!("{p}attn.rel_l2_clip.scale"), "BF16", vec![c.heads, c.d_rel]);
        plain(format!("{p}attn.rel_logits_proj.proj"), "BF16", vec![c.d_rel, if local { c.window } else { c.rel_extent }]);
        for name in ["k", "v"] {
            if local { plain(format!("{p}attn.{name}_sconv.weight"), "BF16", vec![c.kv_heads * d, 1, k]); }
            else {
                plain(format!("{p}attn.{name}_sconv.static_w"), "BF16", vec![k, c.kv_heads * d]);
                plain(format!("{p}attn.{name}_sconv.weight_proj"), "BF16", vec![c.kv_heads * k, h]);
            }
        }
        for name in ["attn_sconv", "mlp_sconv"] { plain(format!("{p}{name}.weight"), "BF16", vec![h, 1, k]); }
        let norms: &[&str] = if local { &["attn_norm"] } else { &["q_input_norm", "k_input_norm", "v_input_norm"] };
        for norm in norms.iter().chain(["mlp_norm"].iter()) { plain(format!("{p}{norm}.weight"), "BF16", vec![h]); }
        plain(format!("{p}mlp.gate.weight"), "BF16", vec![e, h]);
        plain(format!("{p}mlp.gate.bias"), "F32", vec![e]);
        plain(format!("{TEXT}attn_res.mlp_queries.{layer}"), "F32", vec![h]);
        if layer > 0 { plain(format!("{TEXT}attn_res.attn_queries.{}", layer - 1), "F32", if local { vec![h] } else { vec![3, h] }); }
        let storage = match kind {
            Fixture::Mxfp8 => ExpertStorage::Mxfp8,
            Fixture::Mixed => if layer < 3 { ExpertStorage::Nvfp4 } else { ExpertStorage::Mxfp8 },
            Fixture::Exl3 => ExpertStorage::Exl3 { bits: 4 },
        };
        match storage {
            ExpertStorage::Mxfp8 => {
                plain(format!("{p}mlp.experts.w13_weight"), "F8_E4M3", vec![e, 2 * i, h]);
                plain(format!("{p}mlp.experts.w13_weight_scale"), "U8", vec![e, 2 * i, h / 32]);
                plain(format!("{p}mlp.experts.w2_weight"), "F8_E4M3", vec![e, h, i]);
                plain(format!("{p}mlp.experts.w2_weight_scale"), "U8", vec![e, h, i / 32]);
            }
            ExpertStorage::Nvfp4 => {
                for (proj, n, kk) in [("w13", 2 * i, h), ("w2", h, i)] {
                    plain(format!("{p}mlp.experts.{proj}_weight"), "U8", vec![e, n, kk / 2]);
                    plain(format!("{p}mlp.experts.{proj}_weight_scale"), "F8_E4M3", vec![e, n, kk / 16]);
                    plain(format!("{p}mlp.experts.{proj}_weight_amax"), "F32", vec![e]);
                    plain(format!("{p}mlp.experts.{proj}_weight_outer_scale_denominator"), "F32", vec![1]);
                }
            }
            ExpertStorage::Exl3 { bits } => {
                for expert in 0..e {
                    for (proj, n, kk) in [("gate_proj", i, h), ("up_proj", i, h), ("down_proj", h, i)] {
                        for (name, dtype, shape) in testing::exl3(&format!("{p}mlp.experts.{expert}.{proj}"), n, kk, bits) {
                            plain(name, dtype, shape);
                        }
                    }
                }
            }
        }
    }
    if kind == Fixture::Exl3 {
        small["quantization_config"] = testing::exl3_compact(4);
    }
    testing::write_snapshot(dir.path(), &small, &tensors, None);
    // The fixtures carry the checkpoint's quantization metadata beside the
    // headers, as the release does; the loader cross-checks and never requires it.
    std::fs::write(dir.path().join("hf_quant_config.json"),
        serde_json::to_vec(&crate::families::inkling2::config::tests::release_quant_config()).unwrap()).unwrap();
    (dir, small, tensors)
}

#[test]
fn inkling_mxfp8_inventory_is_planned_not_executable() {
    let (dir, _, _) = fixture(Fixture::Mxfp8);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert_eq!(report.family.as_deref(), Some("inkling2"));
    assert_eq!(report.runtime, Some(RuntimeStatus::Planned));
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    assert!(report.components.iter().all(|c| c.rejected == 0), "{:?}", report.components);
    assert!(!report.executable());
    let spec = report.spec.unwrap();
    assert_eq!(spec.moe_layers(), 4);
    assert!(matches!(spec.layers[0].attention, AttentionKind::SlidingGqa { window: 1024, sinks: true, .. }));
    assert!(matches!(spec.layers[3].attention, AttentionKind::Gqa { heads: 48, kv_heads: 8, head_dim: 128 }));
    assert!(spec.notes.iter().any(|note| note.contains("MXFP8 W8A8 layers 0..=3")), "{:?}", spec.notes);
    let experts = report.experts.expect("routed expert contract");
    assert_eq!(experts.package, "inkling2:mxfp8");
    assert!(experts.spark_worlds.is_empty(), "no Inkling 2 Spark package ships yet: {:?}", experts.spark_worlds);
}

#[test]
fn mixed_native_checkpoint_reads_each_layers_storage_from_its_headers() {
    let (dir, _, _) = fixture(Fixture::Mixed);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    assert_eq!((0..4).map(|layer| expert_storage(&checkpoint, layer).unwrap()).collect::<Vec<_>>(),
        [ExpertStorage::Nvfp4, ExpertStorage::Nvfp4, ExpertStorage::Nvfp4, ExpertStorage::Mxfp8]);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.components.iter().all(|c| c.rejected == 0), "{:?}", report.components);
    let spec = report.spec.unwrap();
    assert!(spec.notes.iter().any(|note| note.contains("native NVFP4 W4A4 (per-token amax, fouroversix) layers 0..=2")
        && note.contains("MXFP8 W8A8 layers 3..=3")), "{:?}", spec.notes);
    // Two packages, one contract: the plan names both and keeps the common worlds.
    assert_eq!(report.experts.unwrap().package, "inkling2:mxfp8 + inkling2:nvfp4a4");
    let cfg = Inkling2Config::from_hf(&checkpoint.config).unwrap();
    let experts = crate::families::inkling2::experts::layer_experts(&checkpoint, &cfg).unwrap();
    // NVFP4 holds half the codes plus twice the scales of MXFP8; both halves cover the whole.
    assert!(experts[0].whole < experts[3].whole, "{:?} vs {:?}", experts[0], experts[3]);
    for layer in &experts {
        assert!(layer.half[0] + layer.half[1] >= layer.whole - 2 * 4 * 2, "{layer:?}");
        assert!(layer.half[0] >= layer.half[1], "{layer:?}");
    }
    assert_eq!(experts[3].whole, 4 * (2 * 2304 * 4608 + 4608 * 2304) + 4 * (2 * 2304 * 144 + 4608 * 72));
}

#[test]
fn exl3_checkpoint_accepts_per_expert_k4_modules_split_by_row_parity() {
    let (dir, _, _) = fixture(Fixture::Exl3);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    assert!(report.components.iter().all(|c| c.rejected == 0), "{:?}", report.components);
    assert_eq!(report.experts.unwrap().package, "inkling2:exl3-k4");
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    assert_eq!(expert_storage(&checkpoint, 0).unwrap(), ExpertStorage::Exl3 { bits: 4 });
}

#[test]
fn structural_open_names_missing_tensors_and_accepts_reports_drift_per_tensor() {
    let (dir, small, tensors) = fixture(Fixture::Mxfp8);
    // A missing conv tensor is a typed open error naming it.
    let pruned = tensors.iter().filter(|t| !t.0.ends_with("layers.1.attn_sconv.weight")).cloned().collect::<Vec<_>>();
    let dir2 = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir2.path(), &small, &pruned, None);
    let error = match Model::open(&Checkpoint::open(dir2.path()).unwrap()) {
        Ok(_) => panic!("open accepted a checkpoint missing layers.1.attn_sconv.weight"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("layers.1.attn_sconv.weight"), "{error}");
    // A drifted dtype is an `accepts` rejection, not an open failure.
    let drifted = tensors.iter().map(|t| if t.0.ends_with("layers.2.mlp.gate.weight") { (t.0.clone(), "F32", t.2.clone()) } else { t.clone() })
        .collect::<Vec<_>>();
    let dir3 = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir3.path(), &small, &drifted, None);
    let report = plan::plan(dir3.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    let router = report.components.iter().find(|c| c.component == Component::Router).unwrap();
    assert_eq!(router.rejected, 1, "{router:?}");
    assert!(router.rejections[0].tensor.ends_with("layers.2.mlp.gate"), "{:?}", router.rejections);
    // An unknown tensor is a named census rejection.
    let mut extra = tensors.clone();
    extra.push((format!("{TEXT}layers.0.attn.extra.weight"), "BF16", vec![8]));
    let dir4 = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir4.path(), &small, &extra, None);
    let report = plan::plan(dir4.path(), &PlanOptions::default()).unwrap();
    let attention = report.components.iter().find(|c| c.component == Component::Attention).unwrap();
    assert!(attention.rejections.iter().any(|r| r.tensor.contains("attn.extra") && r.reason.contains("outside the Inkling 2 tensor contract")),
        "{:?}", attention.rejections);
    drop(dir);
}

#[test]
fn shared_recipe_charges_every_owner_once_on_one_and_two_rtx() {
    use crate::families::inkling2::{experts::layer_experts, weight_layout::CoordinatorWeights};
    use crate::placement::families::inkling2::{self as shared, InklingInputs, LocalPackage, SparkCharge};
    use crate::placement::planner::{Context, Location, PrefillShape, ComponentKind, Recipe, serving};
    let (dir, small, _) = fixture(Fixture::Mxfp8);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = Inkling2Config::from_hf(&small).unwrap();
    let routed = (0..cfg.layers).collect::<Vec<_>>();
    for gpus in [1usize, 2] {
        let owners = shared::layer_owners(cfg.layers, gpus);
        let targets = CoordinatorWeights::plan(&checkpoint, &cfg, gpus,
            &owners.iter().map(|&g| usize::from(g)).collect::<Vec<_>>()).unwrap();
        let experts = layer_experts(&checkpoint, &cfg).unwrap();
        let ranks = 2;
        let prefill = PrefillShape { rows: 1024, waves: 3 };
        let charge = |count: usize| (0..ranks).map(|_| if count == routed.len() {
            SparkCharge { experts: 0, owners: 0, runtime: 0 } } else {
            SparkCharge { experts: ((routed.len() - count) as u64) << 30, owners: 1 << 30, runtime: 1 << 30 } }).collect();
        let budgets = (0..gpus).map(|gpu| shared::planned_budget(gpu as u8, 96 << 30, 96 << 30, 1 << 30))
            .chain((0..ranks).map(|rank| shared::spark_budget(rank as u8, 100 << 30)))
            .chain([crate::placement::families::kimi::host_budget(512 << 30, 480 << 30)]).collect();
        let inputs = InklingInputs { cfg: cfg.clone(), gpus, spark_ranks: ranks, context: Context::automatic(1 << 20),
            maximum_pool_tokens: 4 << 20, prefill, sequences: 16, targets: targets.clone(), owners: owners.clone(),
            routed: routed.clone(), experts,
            local: LocalPackage::Installed { workspace: [64 << 20, if gpus == 2 { 64 << 20 } else { 0 }] },
            spark: (0..=routed.len()).map(charge).collect(),
            links: shared::spark_link((gpus - 1) as u8, ranks, prefill, Default::default(), cuteafd_core::SparkIngress::Fp8K32).unwrap(),
            spark_reply_formats: Default::default(), drafter: None, vision: None, budgets,
            code: vec![64 << 20; gpus], graph_reserve: vec![500 << 20; gpus], head: serving::BlasHead::Gemm,
            host_prefix: serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }, device_sampling: true,
            expert_usage: None, resident_weight_limit: None,
            mount: targets.iter().map(|t| t.mount_load(&checkpoint).unwrap()).collect() };
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
        // KV pages exist for the global layer only, on its attention owner; local layers have none.
        for (layer, &owner) in owners.iter().enumerate() {
            let homes = contract.components.iter().filter_map(|c| match &c.kind {
                ComponentKind::Allocation { template, .. } if template.id.starts_with(&format!("inkling2.kv.{layer}.")) =>
                    Some(template.location), _ => None }).collect::<std::collections::BTreeSet<_>>();
            if cfg.local[layer] { assert!(homes.is_empty(), "layer {layer} is sliding-window"); }
            else { assert_eq!(homes, [Location::coordinator(owner)].into(), "layer {layer}"); }
        }
        // The per-sequence ring/lag marks exist on every rank that owns a layer.
        for gpu in 0..gpus {
            assert!(contract.components.iter().any(|c| matches!(&c.kind, ComponentKind::Allocation { template, .. }
                if template.id == format!("inkling2.marks.{gpu}"))), "rtx{gpu} marks");
        }
        let admitted = inputs.admit(request).unwrap();
        assert!(admitted.solved.pool_tokens >= 1 << 20);
        assert!(admitted.placement.layers.iter().all(|l| shared::INKLING2.runs(l.mode)));
        if gpus == 2 {
            for layer in 0..cfg.layers {
                let stem = format!("{TEXT}layers.{layer}.mlp_norm.weight");
                let homes = targets.iter().filter(|t| t.arena.view(&stem).is_some()).map(|t| t.rank).collect::<Vec<_>>();
                assert_eq!(homes, [usize::from(owners[layer])], "layer {layer}");
            }
            assert!(admitted.placement.tp2.is_none_or(|t| t.peak_bytes[0] >= t.peak_bytes[1]));
        }
    }
}
