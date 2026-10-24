use super::*;
use crate::plan::{self, checkpoint::Checkpoint, testing::{self, Tensor as FixtureTensor}, format, *};
use serde_json::Value;

fn fixture(kind: &str) -> (Value, Vec<FixtureTensor>) {
    let source: Value = serde_json::from_str(include_str!("fixtures/official-fp8.json")).unwrap();
    let mut config = source["config"].clone();
    if kind == "nvfp4" {
        config = serde_json::from_str(include_str!("fixtures/official-nvfp4-config.json")).unwrap();
    } else if kind == "bf16" {
        config.as_object_mut().unwrap().remove("quantization_config");
    }
    let mut tensors = Vec::new();
    for (name, value) in source["tensors"].as_object().unwrap() {
        let dtype = value[0].as_str().unwrap();
        let shape: Vec<usize> = value[1].as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as usize).collect();
        let dtype = match dtype { "BF16" => "BF16", "F32" => "F32", "F8_E4M3" => "F8_E4M3", _ => panic!("{dtype}") };
        if kind == "bf16" && name.ends_with("weight_scale_inv") { continue; }
        if kind == "nvfp4" && name.ends_with("weight_scale_inv") { continue; }
        if dtype == "F8_E4M3" && kind == "nvfp4" {
            let stem = name.strip_suffix(".weight").unwrap();
            let [e,n,k]: [usize;3] = shape.try_into().unwrap();
            tensors.push(testing::t(name, "U8", &[e,n,k/2]));
            tensors.push(testing::t(format!("{stem}.weight_scale"), "F8_E4M3", &[e,n,k/16]));
            tensors.push(testing::t(format!("{stem}.weight_scale_2"), "F32", &[e]));
            tensors.push(testing::t(format!("{stem}.input_scale"), "F32", &[e]));
        } else { tensors.push((name.clone(), if kind == "bf16" { "BF16" } else { dtype }, shape)); }
    }
    // Only originally quantized expert matrices widen in the BF16 root;
    // all release FP32 norms/router/index weights stay FP32.
    if kind == "bf16" {
        for (name,dtype,_) in &mut tensors {
            if source["tensors"][name.as_str()][0] == "F32" { *dtype = "F32"; }
        }
    }
    (config,tensors)
}

pub(super) fn snapshot(kind: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let (config,tensors) = fixture(kind);
    testing::write_snapshot(dir.path(), &config, &tensors, None);
    dir
}

fn native_manifest(dir: &std::path::Path) -> std::path::PathBuf {
    let ops = ["hidden_norm", "final_norm", "residual_bf16", "residual_f32", "swa_attention",
        "csa_attention", "region_compress", "region_scores", "region_selection", "region_commit",
        "sliding_commit", "attn_q", "attn_kv", "attn_gate", "attn_out", "router",
        "dense_gate_up", "dense_down", "shared_gate_up", "shared_down"];
    let programs = ["step5", "step52"].iter().flat_map(|prefix| ops.iter().map(move |op|
        serde_json::json!({"name": format!("{prefix}_{op}"), "capacity_rows": 4096}))).collect::<Vec<_>>();
    let path = dir.join("PROGRAMS.json");
    std::fs::write(&path, serde_json::to_vec(&serde_json::json!({"programs": programs,
        "capacities": {"max_context": 1048576}})).unwrap()).unwrap();
    path
}

fn report(dir: &std::path::Path, gpus: usize, gib: u64, ranks: usize, pool: Option<u64>) -> PlanReport {
    report_context(dir, gpus, gib, ranks, pool, 0)
}

fn report_context(dir: &std::path::Path, gpus: usize, gib: u64, ranks: usize, pool: Option<u64>, context: u64) -> PlanReport {
    let manifest = native_manifest(dir);
    plan::plan(dir, &PlanOptions { placement: ExpertPlacement::from_spark_ranks(ranks),
        draft_ownership: Some(spec::DraftOwnership::None),
        layout: Some(layout::LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![gib<<30;gpus], pool_tokens: pool, context_tokens: context,
            workspace_manifest: Some(manifest), ..Default::default() }),
        ..Default::default() }).unwrap()
}

#[test]
fn coordinator_headers_include_native_mtp_only_when_selected() -> anyhow::Result<()> {
    use std::collections::BTreeSet;
    let fixture = snapshot("fp8");
    let check = |path: &std::path::Path| -> anyhow::Result<()> {
        let off = Checkpoint::coordinator(path, false, false)?;
        let on = Checkpoint::coordinator(path, false, true)?;
        let inventory = Checkpoint::inventory(path)?;
        let model = Model::open(&inventory)?;
        let names = |cp: &Checkpoint| cp.tensors.iter()
            .map(|tensor| tensor.meta.name.clone()).collect::<BTreeSet<_>>();
        let off_names = names(&off);
        let on_names = names(&on);
        let expected_off = inventory.tensor_names().filter(|name| {
            model.tensors.get(*name).is_some_and(|tensor| !matches!(tensor.role.component,
                Component::Speculator | Component::SpeculatorExpert | Component::RoutedExpert
                    | Component::Vision | Component::Audio))
        }).map(str::to_owned).collect::<BTreeSet<_>>();
        assert_eq!(off_names, expected_off, "MTP-off target resident set changed");
        let predictors = inventory.tensor_names().filter(|name| {
            (92..95).any(|layer| name.starts_with(&format!("model.layers.{layer}.")))
        }).map(str::to_owned).collect::<BTreeSet<_>>();
        assert!(!predictors.is_empty());
        assert_eq!(on_names.difference(&off_names).cloned().collect::<BTreeSet<_>>(), predictors);
        assert!(off_names.is_subset(&on_names));
        assert_eq!(off.weight_map, on.weight_map);
        for stage in 0..3 {
            let layout = coordinator::RankWeightLayout::for_mtp_stage(&on, stage)?;
            assert_eq!(layout.operands.len(), 16);
            for operand in &layout.operands {
                assert!(!operand.sources.is_empty(), "native operand must retain source headers");
                for source in &operand.sources {
                    let header = on.tensors.iter().find(|tensor| tensor.meta.name == source.meta.name)
                        .expect("indexed predictor source must resolve to its own shard header");
                    assert_eq!(on.weight_map.get(&source.meta.name), Some(&header.shard));
                    assert!(!off_names.contains(&source.meta.name));
                }
            }
        }
        assert_eq!(names(&Checkpoint::coordinator(path, false, false)?), off_names);
        eprintln!("Step-5 coordinator headers: target={} predictor={} stages=3 index_bound=true mtp_off_unchanged=true",
            off_names.len(), predictors.len());
        Ok(())
    };
    check(fixture.path())?;
    if let Some(snapshot) = std::env::var_os("CUTEAFD_STEP5_HEADER_SNAPSHOT") {
        check(std::path::Path::new(&snapshot))?;
    }
    Ok(())
}

#[test]
fn native_mtp_contract_accepts_only_validated_predictor_headers() -> anyhow::Result<()> {
    let fixture = snapshot("fp8");
    let check = |path: &std::path::Path| -> anyhow::Result<()> {
        let cp = Checkpoint::coordinator(path, false, true)?;
        let model = Model::open(&cp)?;
        let groups = format::group_by_stem(&cp.tensors);
        let scale = format::TensorScale { encoding: format::ScaleEncoding::F32 };
        let mut count = 0;
        for (stem, members) in &groups {
            let Some(role) = STEP5.classify(model.spec(), stem) else { continue; };
            if role.component != Component::Speculator { continue; }
            let operand = format::detect(members)?;
            assert!(model.accepts(&role, stem, &mut operand.clone()).is_ok(), "{stem}");
            for mutation in 0..8 {
                let mut wrong = operand.clone();
                match mutation {
                    0 => wrong.logical[0] -= 1,
                    1 => wrong.stored[0] -= 1,
                    2 => wrong.encoding = format::Encoding::F16,
                    3 => wrong.global_scale = Some(scale),
                    4 => wrong.input_scale = Some(scale),
                    5 => wrong.scale = Some(format::BlockScale {
                        tensor: "weight_scale_inv".into(), encoding: format::ScaleEncoding::F32,
                        direction: format::ScaleDirection::Multiply,
                        rows: format::RowTiling::Uniform { rows: 128 }, cols: 128, grid: vec![1, 1],
                    }),
                    6 => wrong.container = "F16".into(),
                    _ => wrong.companions.push("bias".into()),
                }
                assert!(model.accepts(&role, stem, &mut wrong).is_err(), "{stem} mutation {mutation}");
            }
            for wrong in [TensorRole::layer(Component::Speculator, 91),
                TensorRole::layer(Component::Speculator, 95), TensorRole::layer(Component::Attention, role.layer.unwrap())] {
                assert!(model.accepts(&wrong, stem, &mut operand.clone()).is_err(), "{stem} {wrong:?}");
            }
            let unknown = format!("model.layers.{}.unvalidated", role.layer.unwrap());
            assert!(model.accepts(&role, &unknown, &mut operand.clone()).is_err());
            count += 1;
        }
        assert_eq!(count, 51, "only the qualified 17 operands per stage may be admitted");
        assert!(STEP5.component_hint(Component::Speculator).is_none());
        Ok(())
    };
    check(fixture.path())?;
    if let Some(snapshot) = std::env::var_os("CUTEAFD_STEP5_HEADER_SNAPSHOT") {
        check(std::path::Path::new(&snapshot))?;
    }
    Ok(())
}

#[test]
fn full_context_mtp_forecast_charges_private_predictors_before_allocation() -> anyhow::Result<()> {
    use crate::placement::{inventory::{self, ArchContext}, *};
    use super::admission::{self, MarkRule, RuntimeInputs};
    let fixture = snapshot("fp8");
    let checkpoint_path = std::env::var_os("CUTEAFD_STEP5_FORECAST_SNAPSHOT")
        .map(std::path::PathBuf::from).unwrap_or_else(|| fixture.path().to_path_buf());
    let cp = Checkpoint::open(&checkpoint_path)?;
    let manifest_path = std::env::var_os("CUTEAFD_STEP5_FORECAST_MANIFEST")
        .map(std::path::PathBuf::from).unwrap_or_else(|| native_manifest(fixture.path()));
    let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    let programs = ProgramSet::from_manifest(&manifest, &["step5", "step52"]);
    let capacity = inventory::PRO_TOTAL_BYTES;
    let arch = ArchContext::coordinator(capacity, None);
    for kv_format in [crate::families::qwen4::Qwen4KvCache::Bf16,
        crate::families::qwen4::Qwen4KvCache::Fp8] {
        let inputs = RuntimeInputs {
            kv_format,
            inventory: Inventory { gpus: vec![GpuBudget { capacity_bytes: capacity,
                baseline: Baseline::Planned { context_bytes: arch.context_bytes, loaded_bytes: 0 },
                headroom_bytes: 1 << 30 }; 2], spark_ranks: 6, peer_access: true },
            rows: 256, waves: 1, full_prefill_logits: false, sequences: 16, context: 1_048_576,
            requested_pool: Some(1_048_576),
            marks: MarkRule { lanes: 16, entries: 20, budget_bytes: 2 << 30, host: true },
            pending_code: (0..2).map(|rank| inventory::loaded_code_allowance(
                "step5", "step5:fp8", true, rank, &arch, Some(&programs)).bytes).collect(),
            peer_diagnostics: false, startup: vec![],
        };
        for depth in 0..=3 {
            let admitted = admission::forecast_mtp(&cp, &inputs, depth)?;
            assert_eq!(admitted.cache.context, 1_048_576);
            assert_eq!(admitted.placement.pool_tokens, 1_048_576);
            assert_eq!(admitted.marks, 2);
            assert_eq!(admitted.cache.ranks[0].mtp_stage_hidden.len(), depth);
            assert!(admitted.cache.ranks[1].mtp_stage_hidden.is_empty());
            let weights = coordinator::RankWeightLayout::for_mtp(&cp, depth)?.resident_bytes;
            for rank in 0..2 {
                let target = native::rank_admission(&cp, &admitted.ownership, rank, inputs.unit_rows(), inputs.rows, inputs.logit_rows())?;
                let item = |group| admitted.fixed.iter().find(|item| item.gpu == rank as u8 && item.group == group)
                    .expect("native charge").bytes;
                assert_eq!(item("Step-5 native target and head"), target.immutable + if rank == 0 { weights } else { 0 });
                let workspace = if rank == 0 {
                    target.workspace.max(native::mtp_forecast_workspace(inputs.rows, depth, kv_format)?)
                } else { target.workspace };
                assert_eq!(item("Step-5 native workspace"), workspace);
                assert_eq!(item("Step-5 exact retained marks"), admitted.reservations[rank].marks as u64);
                assert_eq!(item("Step-5 active sequence state"), admitted.reservations[rank].active as u64);
            }
            let planned = plan::plan(&checkpoint_path, &PlanOptions {
                placement: ExpertPlacement::from_spark_ranks(6),
                draft_ownership: if depth == 0 { Some(spec::DraftOwnership::None) } else { None },
                layout: Some(layout::LayoutOptions { host_memory: Some((128 << 30, 128 << 30)),
                    rtx_bytes: vec![capacity; 2], pool_tokens: inputs.requested_pool, step5_kv: kv_format,
                    context_tokens: inputs.context, concurrency: inputs.sequences as u64,
                    prefill_rows: inputs.rows as u64, prefill_lanes: inputs.waves as u64, headroom_bytes: 1 << 30,
                    native_mtp_layers: depth, workspace_manifest: Some(manifest_path.clone()),
                    mimo_prefix_entries: inputs.marks.entries as u64,
                    mimo_prefix_mark_bytes: inputs.marks.budget_bytes,
                    ..Default::default()
                }), ..Default::default()
            })?;
            assert!(planned.wip_launchable(), "depth {depth}: {:?}", planned.hints);
            let memory = planned.memory_layout.as_ref().expect("depth-aware Step-5 layout");
            assert_eq!(memory.pool_tokens, admitted.placement.pool_tokens);
            for rank in 0..2 {
                let device = &memory.devices[rank];
                for category in [cuteafd_core::memory_layout::Category::Weights,
                    cuteafd_core::memory_layout::Category::Kv,
                    cuteafd_core::memory_layout::Category::Prefix,
                    cuteafd_core::memory_layout::Category::Workspace] {
                    let expected = admitted.placement.items[rank].iter()
                        .filter(|item| item.category == category).map(|item| item.bytes).sum::<u64>();
                    let actual = device.items.iter().filter(|item| item.category == category)
                        .map(|item| item.bytes).sum::<u64>();
                    assert_eq!(actual, expected, "depth {depth}, rank {rank}, {category:?}");
                }
            }
            let charged = |a: &admission::Admission, rank: usize| a.placement.items[rank].iter()
                .map(|item| item.bytes).sum::<u64>();
            let by_category = (0..2).map(|rank| {
                let mut categories = std::collections::BTreeMap::new();
                for item in &admitted.placement.items[rank] {
                    *categories.entry(format!("{:?}", item.category)).or_insert(0u64) += item.bytes;
                }
                serde_json::json!({"rank": rank, "charged": charged(&admitted, rank),
                    "context": arch.context_bytes, "headroom": 1u64 << 30,
                    "remaining": capacity - arch.context_bytes - (1 << 30) - charged(&admitted, rank),
                    "categories": categories})
            }).collect::<Vec<_>>();
            eprintln!("Step-5 full-context forecast {}", serde_json::json!({
                "depth": depth, "context": admitted.cache.context, "pool": admitted.placement.pool_tokens,
                "sequences": inputs.sequences, "rows": inputs.rows,
                "marks": admitted.marks, "predictor_weights": weights,
                "owners": admitted.ownership.execution().modes().iter().map(|mode| format!("{mode:?}")).collect::<Vec<_>>(),
                "ranks": by_category }));
        }
        assert!(admission::forecast_mtp(&cp, &inputs, 4).is_err());
    }
    Ok(())
}

#[test]
fn execution_owners_bind_native_weights_cache_and_exact_snapshots() {
    use crate::placement::{FfnMode, LayerMode};
    use crate::serving_capacity::{step5::Step5CacheLayout, KvPlacement};
    use super::{coordinator::RankWeightLayout, ownership::{self, ExecutionOwnership}};
    for quant in ["fp8", "nvfp4"] {
        let dir = snapshot(quant);
        let cp = Checkpoint::open(dir.path()).unwrap();
        let cfg = Config::parse(&cp).unwrap();
        let diagnostic = Step5CacheLayout::new(&cfg, 1, 1024, 0, 64, crate::families::qwen4::Qwen4KvCache::Bf16).unwrap();
        // Exercise unequal owner counts, a rank with no CSA, and owned SWA.
        for (ranks, policy) in [(1, 0), (2, 0), (2, 1), (2, 2)] {
            let modes: Vec<_> = (0..cfg.layers).map(|l| {
                if ranks == 2 && !cfg.full[l] && policy != 2 { LayerMode::HeadSplit }
                else { LayerMode::Whole { gpu: if ranks == 1 || policy != 0 { 0 }
                    else { ((l / 4) % 2) as u8 }, ffn: FfnMode::Owner } }
            }).collect();
            if ranks == 2 && policy == 2 {
                let alternating: Vec<_> = (0..cfg.layers).map(|layer|
                    LayerMode::Whole { gpu: ((layer / 4) % 2) as u8, ffn: FfnMode::Owner }).collect();
                let error = ExecutionOwnership::new(&cfg, ranks, ownership::plan(&alternating, 0, 1))
                    .unwrap_err().to_string();
                assert!(error.contains("without an intervening head exchange"), "{error}");
            }
            let head = if ranks == 1 { 0 } else { 1 };
            let owners = ExecutionOwnership::new(&cfg, ranks, ownership::plan(&modes, 0, head)).unwrap();
            assert_eq!(owners.routed_endpoints().unwrap(), 1);
            let layout = Step5CacheLayout::for_execution(&cfg, &owners, 1024, 0, 64, crate::families::qwen4::Qwen4KvCache::Bf16).unwrap();
            assert_eq!(layout.geometry.placement, if ranks == 1 { KvPlacement::SingleDevice }
                else { KvPlacement::PartitionedLayers });
            assert_eq!(layout.geometry.ranks.iter().map(|r| r.persistent_unit_bytes).sum::<u64>(),
                diagnostic.geometry.ranks[0].persistent_unit_bytes);
            let weights: Vec<_> = (0..ranks).map(|rank| RankWeightLayout::for_execution(&cp, rank, &owners).unwrap()).collect();
            for rank in 0..ranks {
                let rows = owners.execution().spec().rows as usize;
                let attention = (0..cfg.layers).filter_map(|layer| {
                    let local = owners.layer(layer, rank).unwrap();
                    (local.query_heads > 0).then(|| native::attention_bytes(
                        rows, cfg.full[layer], local.query_heads).unwrap())
                }).collect::<Vec<_>>();
                let layer_bytes = attention.iter().map(|bytes| bytes + native::ffn_region_bytes(rows, rows).unwrap())
                    .collect::<Vec<_>>();
                let logits = native::head_rows(rows, false);
                let head_bytes = if rank == usize::from(head) { native::plane_bytes(&native::head_planes(logits).unwrap()).unwrap() } else { 0 };
                let expected = layer_bytes.iter().copied().max().unwrap_or(0).max(head_bytes);
                assert_eq!(native::rank_admission(&cp, &owners, rank, rows, rows, logits).unwrap().workspace, expected);
                if layer_bytes.len() > 1 { assert!(expected < layer_bytes.iter().sum::<u64>() + head_bytes); }
            }
            for (layer, &mode) in modes.iter().enumerate() {
                let mut attention = Vec::new();
                let mut ffns = Vec::new();
                for rank in 0..ranks {
                    let local = owners.layer(layer, rank).unwrap();
                    let state = &layout.ranks[rank];
                    let prefix = format!("model.layers.{layer}.");
                    let operands: Vec<_> = weights[rank].operands.iter()
                        .filter(|o| o.name.starts_with(&prefix)).collect();
                    let q = operands.iter().find(|o| o.name.ends_with("self_attn.q_proj.weight"));
                    assert_eq!(q.is_some(), local.query_heads > 0);
                    if let Some(q) = q {
                        attention.push(rank as u8);
                        assert_eq!(q.shape, [local.query_heads * 192, 4096]);
                    }
                    let ffn = operands.iter().any(|o| o.name.ends_with("post_attention_layernorm.weight"));
                    assert_eq!(ffn, local.ffn);
                    assert_eq!(local.routed_wave, cfg.moe[layer] && match mode {
                        LayerMode::Whole { gpu, .. } => rank == gpu as usize,
                        LayerMode::HeadSplit => rank == 0,
                        LayerMode::ContextSplit => unreachable!(),
                    });
                    if ffn { ffns.push(rank); }
                    let branch = if cfg.moe[layer] { "share_expert" } else { "mlp" };
                    assert_eq!(operands.iter().any(|o| o.name.ends_with(&format!("{branch}.gate_up_proj.weight"))), local.ffn);
                    assert_eq!(operands.iter().any(|o| o.name.ends_with("moe.gate.weight")), local.ffn && cfg.moe[layer]);
                    if cfg.full[layer] {
                        let region = state.regions.iter().find(|r| r.layer == layer);
                        assert_eq!(region.is_some(), local.query_heads > 0);
                        if let Some(region) = region {
                            assert_eq!(region.query_heads, 64);
                            assert_eq!(region.page_bytes, 64 * 4 * 192 * 2);
                            assert_eq!(region.summaries.unwrap().bytes, 128 * 256);
                            assert!(operands.iter().any(|o| o.name.ends_with("sparse_indexer_q.weight")));
                            assert!(operands.iter().all(|o| o.slice == coordinator::WeightSlice::Replicated));
                        } else { assert!(operands.is_empty()); }
                    } else {
                        let ring = state.sliding.iter().find(|s| s.layer == layer);
                        assert_eq!(ring.is_some(), local.query_heads > 0);
                        if let Some(ring) = ring {
                            assert_eq!(ring.query_heads, local.query_heads);
                            assert_eq!(ring.key.bytes, layout.ring.rows * local.kv_heads as u64 * 192 * 2);
                        }
                    }
                }
                owners.execution().validate_state_owners(layer, &attention).unwrap();
                assert_eq!(ffns, match mode { LayerMode::Whole { gpu, .. } => vec![gpu as usize], _ => vec![0, 1] });
            }
            for rank in 0..ranks {
                let state = &layout.ranks[rank];
                let cost = layout.geometry.ranks[rank];
                assert_eq!(cost.persistent_unit_bytes, state.regions.iter().map(|r| r.page_bytes * 2).sum::<u64>());
                assert_eq!(cost.active_state_per_sequence_bytes, state.bytes);
                assert_eq!(cost.retained_mark_bytes, state.bytes);
                assert_eq!(weights[rank].resident_bytes, weights[rank].operands.iter().map(|o| o.allocation_bytes).sum::<u64>());
                assert_eq!(weights[rank].operands.iter().any(|o| o.name == "model.embed_tokens.weight"), rank == 0);
                assert_eq!(weights[rank].operands.iter().any(|o| o.name == "lm_head.weight"), rank == head as usize);
                for frontier in [0, 63, 64, 65, 575, 576, 1024] {
                    let spans = layout.snapshot_spans(rank, frontier, &vec![0; state.mtp_lengths.len()]).unwrap();
                    let expected = 8 + state.sliding.iter().map(|s| 2 * frontier.min(512) * s.key.bytes / layout.ring.rows).sum::<u64>()
                        + state.regions.len() as u64 * (2 * 8 * 256 * 2 + frontier / 8 * 256);
                    assert_eq!(spans.iter().map(|s| s.bytes).sum::<u64>(), expected);
                    let copies = layout.snapshot_copies(rank, frontier, &[], 3, 2, 4, 3).unwrap();
                    assert_eq!(copies.iter().map(|s| s.active.bytes).sum::<u64>(), expected);
                    assert!(copies.iter().all(|c| c.active.offset + c.active.bytes <= state.bytes * 3
                        && c.mark.offset >= state.bytes * 3 && c.mark.offset + c.mark.bytes <= state.bytes * 4));
                    assert_eq!(layout.csa_tail_bytes(rank, frontier % 64).unwrap(), cost.persistent_unit_bytes / 64 * (frontier % 64));
                }
            }
            let mtp = Step5CacheLayout::for_execution(&cfg, &owners, 1024, 3, 64, crate::families::qwen4::Qwen4KvCache::Bf16).unwrap();
            for (rank, state) in mtp.ranks.iter().enumerate() {
                assert_eq!(state.mtp_hidden.is_some(), rank == head as usize);
                assert_eq!(state.mtp_stage_hidden.len(), if rank == head as usize { 3 } else { 0 });
                assert_eq!(state.sliding.iter().filter(|s| s.mtp).count(), if rank == head as usize { 3 } else { 0 });
            }
        }
    }
}

#[test]
fn execution_ownership_refuses_unbound_modes_and_checkpoint_changes() {
    use crate::placement::{FfnMode, LayerMode};
    use super::ownership::{self, ExecutionOwnership};
    let dir = snapshot("fp8");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let cfg = Config::parse(&cp).unwrap();
    let whole = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
    let modes = vec![whole; 92];
    let owners = ExecutionOwnership::new(&cfg, 2, ownership::plan(&modes, 0, 0)).unwrap();
    assert!(owners.layer(92, 0).is_err());
    assert!(owners.layer(0, 2).is_err());
    for ranks in [0, 3] { assert!(ExecutionOwnership::new(&cfg, ranks, ownership::plan(&modes, 0, 0)).is_err()); }
    assert!(ExecutionOwnership::new(&cfg, 2, ownership::plan(&modes[..91], 0, 0)).is_err());
    for (layer, mode, ranks) in [(3, LayerMode::HeadSplit, 2), (0, LayerMode::HeadSplit, 1),
        (0, LayerMode::ContextSplit, 2), (3, LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, 2),
        (0, LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }, 1)] {
        let mut changed = modes.clone();
        changed[layer] = mode;
        assert!(ExecutionOwnership::new(&cfg, ranks, ownership::plan(&changed, 0, 0)).is_err());
    }
    assert!(ExecutionOwnership::new(&cfg, 1, ownership::plan(&modes, 0, 1)).is_err());
    let mut changed = cfg.clone();
    changed.full[3] = false;
    assert!(owners.validate_config(&changed).is_err());
    assert!(crate::serving_capacity::step5::Step5CacheLayout::for_execution(&changed, &owners, 1024, 0, 64, crate::families::qwen4::Qwen4KvCache::Bf16).is_err());
}

#[test]
fn routed_replication_admission_requires_a_consumer_ordered_head_exchange() {
    use crate::placement::{FfnMode, LayerMode, LayerPlanError, RoutedReplication};
    use super::ownership;
    let whole = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
    let copy = |slot, storage| Some(RoutedReplication { source: 0, slot, storage });
    let check = |modes: &[LayerMode], copies: &[Option<RoutedReplication>]| {
        ownership::plan(modes, 0, 0).validate_routed_replications(copies)
    };
    // Different slots do not protect an aliased allocation; different buffers
    // do not protect a publication flag whose next value can pass its reader.
    for copies in [[copy(4, 1), copy(4, 2)], [copy(4, 1), copy(5, 1)]] {
        assert!(matches!(check(&[whole, whole], &copies), Err(LayerPlanError::ReplicationReuse { layer: 1, previous: 0 })));
    }
    // Two unique destinations/flags still race on the next request, even when
    // the planner supplies an exit hop. Context and one-way hops are no proof.
    for modes in [vec![whole, whole], vec![whole, LayerMode::ContextSplit]] {
        assert!(matches!(check(&modes, &[copy(4, 1), copy(5, 2)]), Err(LayerPlanError::ReplicationReuse { .. })));
    }
    assert!(check(&[whole], &[copy(4, 1)]).is_err());
    assert!(check(&[whole, LayerMode::HeadSplit, whole], &[copy(4, 1), None, copy(4, 1)]).is_err(),
        "in-pass exchange cannot excuse unsafe wraparound reuse");
    check(&[LayerMode::HeadSplit, whole, LayerMode::HeadSplit, whole],
        &[None, copy(4, 1), None, copy(4, 1)]).unwrap();
    check(&[LayerMode::HeadSplit, whole, whole], &[None, copy(4, 1), copy(5, 2)]).unwrap();
    let plan = ownership::plan(&[whole], 0, 0);
    for invalid in [vec![], vec![copy(usize::MAX, 1)], vec![copy(4, 0)],
        vec![Some(RoutedReplication { source: 2, slot: 4, storage: 1 })]] {
        assert_eq!(plan.validate_routed_replications(&invalid), Err(LayerPlanError::ReplicationBindings));
    }
}

#[test]
fn step5_routed_replication_bindings_are_single_wave_and_disjoint_from_heads() {
    use crate::placement::{FfnMode, LayerMode};
    use super::ownership::{self, ExecutionOwnership};
    let dir = snapshot("fp8");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let cfg = Config::parse(&cp).unwrap();
    let modes: Vec<_> = (0..cfg.layers).map(|layer| if cfg.full[layer] {
        LayerMode::Whole { gpu: ((layer / 4) % 2) as u8, ffn: FfnMode::Owner }
    } else { LayerMode::HeadSplit }).collect();
    let ownership = ExecutionOwnership::new(&cfg, 2, ownership::plan(&modes, 0, 0)).unwrap();
    let copies = ownership.routed_replications().unwrap();
    assert_eq!(copies.iter().flatten().count(), 77);
    for (layer, copy) in copies.iter().enumerate().filter_map(|(l, c)| c.map(|c| (l, c))) {
        assert_eq!(copy.source, 0);
        assert_eq!(copy.slot, if cfg.full[layer] { 5 } else { 4 });
        assert_eq!(copy.storage, layer + 1);
        assert!(cfg.moe[layer]);
    }
    ownership.execution().validate_routed_replications(&copies).unwrap();
    assert_eq!(ownership.routed_endpoints().unwrap(), 1);
}

#[test]
fn predictor_histories_are_private_charged_and_snapshotted_without_append_slack() {
    use crate::serving_capacity::step5::Step5CacheLayout;
    let dir = snapshot("fp8");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let cfg = Config::parse(&cp).unwrap();
    for kv_format in [crate::families::qwen4::Qwen4KvCache::Bf16, crate::families::qwen4::Qwen4KvCache::Fp8] {
    for depth in 0..=3 {
        let layout = Step5CacheLayout::new(&cfg, 2, 1 << 20, depth, 256, kv_format).unwrap();
        let state = &layout.ranks[0];
        assert_eq!(state.mtp_stage_hidden.len(), depth);
        assert!(layout.ranks[1].mtp_stage_hidden.is_empty());
        if depth == 0 {
            assert_eq!(layout.mtp_hidden_rows, 0);
            assert!(state.mtp_hidden.is_none());
            continue;
        }
        assert_eq!(layout.mtp_hidden_rows, 832);
        let histories = state.mtp_stage_hidden.iter().copied().chain(state.mtp_hidden)
            .collect::<Vec<_>>();
        assert_eq!(histories.iter().map(|span| span.bytes).sum::<u64>(),
            832 * 4096 * (4 * depth as u64 + 2));
        for (index, span) in histories.iter().enumerate() {
            assert!(histories[index + 1..].iter().all(|other|
                span.offset + span.bytes <= other.offset || other.offset + other.bytes <= span.offset));
            assert!(span.offset + span.bytes <= state.bytes);
        }
        for frontier in [0u64, 511, 512, 831, 832, 1 << 20] {
            let stages: Vec<_> = (0..depth).map(|stage| frontier.saturating_sub(stage as u64 + 1)).collect();
            let copies = layout.snapshot_copies(0, frontier, &stages, 16, 15, 32, 31).unwrap();
            for (component, end) in std::iter::once((state.mtp_hidden.unwrap(), frontier))
                .chain(state.mtp_stage_hidden.iter().copied().zip(stages.iter().copied())) {
                let at = component.offset * 16 + component.bytes * 15;
                let retained = copies.iter().filter(|c| c.active.offset >= at
                    && c.active.offset < at + component.bytes).map(|c| c.active.bytes).sum::<u64>();
                assert_eq!(retained, end.min(512 + depth as u64 + 1) * component.bytes / layout.mtp_hidden_rows);
                assert!(retained < component.bytes, "proposal slack must not enter a mark");
            }
            assert!(layout.snapshot_spans(0, frontier, &stages).is_ok());
        }
        // An entire proposed wave must not overwrite any history needed by
        // the previous frontier, including all shifted teacher-forced inputs.
        let retained = 512 + depth as u64 + 1;
        for base in [retained, 831, 832, 2 * 832, 1 << 20] {
            let old = (base - retained..base).map(|p| p % layout.mtp_hidden_rows)
                .collect::<std::collections::BTreeSet<_>>();
            assert!((base..base + 256).all(|p| !old.contains(&(p % layout.mtp_hidden_rows))));
        }
    }
    }
}

#[test]
fn predictor_marks_validate_independent_frontiers_and_ignore_rejected_rows() {
    use crate::serving_capacity::step5::Step5CacheLayout;
    let dir = snapshot("fp8");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let cfg = Config::parse(&cp).unwrap();
    for kv_format in [crate::families::qwen4::Qwen4KvCache::Bf16, crate::families::qwen4::Qwen4KvCache::Fp8] {
    let layout = Step5CacheLayout::new(&cfg, 1, 1024, 3, 64, kv_format).unwrap();
    let state = &layout.ranks[0];
    for stages in [[600, 0, 0], [600, 598, 0], [600, 598, 597]] {
        let copies = layout.snapshot_copies(0, 603, &stages, 1, 0, 1, 0).unwrap();
        for (stage, &end) in stages.iter().enumerate() {
            let ring = state.sliding.iter().find(|s| s.layer == 92 + stage).unwrap();
            for (component, rows, keep) in [(ring.key, layout.ring.rows, 512),
                (ring.value, layout.ring.rows, 512),
                (state.mtp_stage_hidden[stage], layout.mtp_hidden_rows, 516)] {
                let stride = component.bytes / rows;
                let mut retained = Vec::new();
                for copy in copies.iter().filter(|c| c.active.offset >= component.offset
                    && c.active.offset < component.offset + component.bytes) {
                    assert_eq!(copy.active.bytes % stride, 0);
                    for row in 0..copy.active.bytes / stride {
                        let source = (copy.active.offset - component.offset) / stride + row;
                        let canonical = (copy.mark.offset - component.offset) / stride + row;
                        retained.push((canonical, source));
                    }
                }
                retained.sort_unstable();
                let expected: Vec<_> = (end.saturating_sub(keep)..end).enumerate()
                    .map(|(canonical, logical)| (canonical as u64, logical % rows)).collect();
                assert_eq!(retained, expected);
                let proposed: std::collections::BTreeSet<_> = (end..end + 64).map(|p| p % rows).collect();
                assert!(retained.iter().all(|(_, source)| !proposed.contains(source)),
                    "rejected rows may never enter a canonical mark");
            }
        }
    }
    for invalid in [vec![], vec![600], vec![603, 0, 0], vec![600, 601, 0], vec![600, 598, 599]] {
        assert!(layout.snapshot_spans(0, 603, &invalid).is_err());
    }
    }
}

#[test]
fn exact_cache_layout_charges_dense_summaries_and_every_retained_byte() {
    use crate::serving_capacity::step5::{Step5CacheLayout, PAGE_ROWS, REGION_ROWS};
    let dir = snapshot("fp8");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let cfg = Model::open(&cp).unwrap().cfg;
    for kv_format in [crate::families::qwen4::Qwen4KvCache::Bf16, crate::families::qwen4::Qwen4KvCache::Fp8] {
    for ranks in [1, 2] {
        let layout = Step5CacheLayout::new(&cfg, ranks, 1 << 20, 3, 64, kv_format).unwrap();
        assert_eq!(layout.mtp_hidden_rows, 640);
        assert_eq!(layout.ranks[0].mtp_stage_hidden.len(), 3);
        assert!(layout.ranks[0].mtp_stage_hidden.iter().all(|span| span.bytes == 640 * 4096 * 4));
        assert_eq!(layout.window(), 512);
        assert_eq!(layout.kv_format, kv_format);
        for state in &layout.ranks {
            for ring in &state.sliding {
                let groups = if ring.mtp { 4 } else { 4 / ranks };
                let row = kv_format.plane_record_bytes(groups, 192) as u64;
                assert_eq!(ring.key.bytes, layout.ring.rows * row);
                assert_eq!(ring.value.bytes, layout.ring.rows * row);
            }
            assert!(state.regions.iter().all(|r|
                r.page_bytes == PAGE_ROWS * kv_format.plane_record_bytes(4 / ranks, 192) as u64));
        }
        assert_eq!(layout.capture_reach(), 0);
        assert_eq!(layout.geometry.logical_unit_rows, PAGE_ROWS);
        let summaries: u64 = layout.ranks[0].regions.iter().map(|r| r.summaries.unwrap().bytes).sum();
        assert_eq!(summaries, 736 << 20);
        assert_eq!(layout.ranks[0].sliding.len(), 72);
        assert_eq!(layout.ranks[0].regions.len(), 23);
        if ranks == 2 {
            assert_eq!(layout.ranks[1].sliding.len(), 69);
            assert!(layout.ranks[1].regions.iter().all(|r| r.summaries.is_none()));
            assert!(layout.ranks[1].mtp_hidden.is_none());
        }
        for (rank, state) in layout.ranks.iter().enumerate() {
            let geometry = layout.geometry.ranks[rank];
            assert_eq!(geometry.retained_mark_bytes, state.bytes);
            assert_eq!(geometry.active_state_per_sequence_bytes, state.bytes);
            for frontier in [0, 1, 7, 8, 63, 64, 65, 511, 512, 513, 1 << 20] {
                let spans = layout.snapshot_spans(rank, frontier, &vec![0; state.mtp_lengths.len()]).unwrap();
                assert!(spans.iter().all(|s| s.offset.checked_add(s.bytes).unwrap() <= state.bytes));
                let summary_bytes = if rank == 0 { 23 * (frontier / REGION_ROWS) * 256 } else { 0 };
                let maximum: u64 = layout.snapshot_spans(rank, 1 << 20, &vec![0; state.mtp_lengths.len()]).unwrap().iter().map(|s| s.bytes).sum();
                let missing_rings: u64 = state.sliding.iter().filter(|s| !s.mtp)
                    .map(|s| 2 * (512 - frontier.min(512)) * s.key.bytes / layout.ring.rows).sum();
                let missing_history = state.mtp_hidden.map_or(0, |span|
                    (516 - frontier.min(516)) * span.bytes / layout.mtp_hidden_rows);
                let expected = maximum - if rank == 0 { summaries - summary_bytes } else { 0 } - missing_rings - missing_history;
                assert_eq!(spans.iter().map(|s| s.bytes).sum::<u64>(), expected);
                assert!(state.regions.iter().all(|region| spans.contains(&region.pending_key)
                    && spans.contains(&region.pending_z)));
                for ring in &state.sliding {
                    for component in [ring.key, ring.value] {
                        let retained: u64 = spans.iter().filter(|s| s.offset >= component.offset
                            && s.offset < component.offset + component.bytes).map(|s| s.bytes).sum();
                        assert_eq!(retained, if ring.mtp { 0 }
                            else { frontier.min(512) * component.bytes / layout.ring.rows });
                    }
                }
                for region in &state.regions {
                    assert!(!spans.contains(&region.next_key) && !spans.contains(&region.next_z));
                }
                assert_eq!(layout.csa_tail_bytes(rank, frontier % PAGE_ROWS).unwrap(),
                    geometry.persistent_unit_bytes / PAGE_ROWS * (frontier % PAGE_ROWS));
            }
        }
        let (slots, arena) = layout.prefix_arena_bytes(16, 20, 2 << 30).unwrap();
        assert_eq!(slots, 34);
        assert!(arena.iter().sum::<u64>() > 2 << 30);
        assert_eq!(arena[0], slots * layout.ranks[0].bytes);
        assert_eq!(layout.prefix_arena_bytes(16, 0, 2 << 30).unwrap(), (0, vec![0; ranks]));
        assert!(layout.snapshot_spans(ranks, 0, &[]).is_err());
        assert!(layout.snapshot_spans(0, (1 << 20) + 1, &[0; 3]).is_err());
        assert!(layout.csa_tail_bytes(0, 65).is_err());
        let mut occupied = Vec::new();
        for sequence in 0..3 {
            let copies = layout.snapshot_copies(0, 65, &[0; 3], 3, sequence, 4, sequence).unwrap();
            let spans = layout.snapshot_spans(0, 65, &[0; 3]).unwrap();
            assert_eq!(copies.len(), spans.len());
            for (copy, span) in copies.iter().zip(spans) {
                assert!(copy.mark.offset >= sequence * layout.ranks[0].bytes
                    && copy.mark.offset + copy.mark.bytes <= (sequence + 1) * layout.ranks[0].bytes);
                assert_eq!(copy.active.bytes, span.bytes);
                assert_eq!(copy.mark.bytes, span.bytes);
                occupied.push((copy.active.offset, copy.active.offset + copy.active.bytes));
            }
            let summary = layout.ranks[0].regions[0].summaries.unwrap();
            let copy = copies.iter().find(|c| c.mark.offset == sequence * layout.ranks[0].bytes + summary.offset).unwrap();
            assert_eq!(copy.active.offset, summary.offset * 3 + sequence * summary.bytes);
            assert_eq!(copy.active.bytes, 8 * 256); // Never copy the unfinished ninth region.
        }
        occupied.sort_unstable();
        assert!(occupied.windows(2).all(|pair| pair[0].1 <= pair[1].0));
        assert!(layout.snapshot_copies(0, 0, &[0; 3], 3, 3, 4, 0).is_err());
        assert!(layout.snapshot_copies(0, 0, &[0; 3], 3, 0, 4, 4).is_err());
        assert!(layout.snapshot_copies(0, 0, &[0; 3], u64::MAX, 0, 4, 0).is_err());
    }
    assert!(Step5CacheLayout::new(&cfg, 3, 1024, 0, 64, kv_format).is_err());
    assert!(Step5CacheLayout::new(&cfg, 1, 0, 0, 64, kv_format).is_err());
    assert!(Step5CacheLayout::new(&cfg, 1, 1024, 4, 64, kv_format).is_err());
    }
}

#[test]
fn official_nested_inventory_separates_target_tail_mtp_vision() {
    let dir = snapshot("fp8");
    // Poisoned sidecar must have no effect on config/header-only planning.
    std::fs::write(dir.path().join("fp8_vit_sidecar_ready.json"), "not JSON").unwrap();
    let cp = Checkpoint::open(dir.path()).unwrap();
    let model = Model::open(&cp).unwrap();
    assert_eq!(model.spec.layers.len(),92);
    assert_eq!(model.spec.moe_layers(),88);
    assert_eq!(model.cfg.full.iter().filter(|v| **v).count(),23);
    assert_eq!(model.cfg.activation.len(),95);
    assert_eq!(model.activation(94,false),Some(FfnActivation::SiluPostClampFp32Product { limit:7.0 }));
    assert_eq!(model.spec.layers[91].ffn,FfnKind::Dense { intermediate:13824 });
    assert_eq!(model.spec.layers[3].rope,Some(RopeSpec { dims:64,theta:10_000_000.0 }));
    assert_eq!(model.spec.layers[0].rope,Some(RopeSpec { dims:192,theta:10_000.0 }));
    assert_eq!(model.bytes.values().sum::<u64>(),641_367_461_632);
    assert_eq!(model.bytes[&Component::RoutedExpert],604_721_627_136);
    assert_eq!(model.bytes[&Component::Speculator],5_031_842_304);
    assert_eq!(model.bytes[&Component::Vision],3_962_029_568);
    for layer in 88..91 {
        assert_eq!(model.tensors[&format!("model.layers.{layer}.moe.gate_proj.weight")].dtype,DType::Bf16);
    }
    let result = report(dir.path(),1,96,6,Some(262_144));
    assert!(result.config_error.is_none(),"{:?}",result.config_error);
    assert!(result.unclassified.is_empty(),"{:?}",result.unclassified);
    assert!(!result.executable() && !result.wip_launchable() && !result.native_serving_admitted);
    assert!(result.components.iter().all(|c| c.status != Status::Ready));
    assert!(result.hints.iter().any(|hint|hint.what.contains("unqualified")));
    assert_eq!(result.min_spark_ranks,Some(6));
}

#[test]
fn card_memory_conclusions_and_request_vs_pool_extent() {
    let dir = snapshot("fp8");
    let r = report_context(dir.path(), 2, 96, 6, Some(262_144), 262_144);
    assert!(r.wip_launchable(), "{:?}", r.hints);
    assert!(!r.executable());
    assert!(r.native_serving_admitted);
    assert_eq!(r.memory_layout.as_ref().unwrap().pool_tokens, 262_144);
    assert!(r.memory_layout.as_ref().unwrap().devices.iter().all(|d| d.used_bytes() <= d.capacity_bytes));
    let r = report(dir.path(), 2, 96, 6, Some(262_144));
    assert!(!r.wip_launchable());
    assert!(r.hints.iter().any(|h| h.what.contains("compiled/max-context floor of 1048576")), "{:?}", r.hints);
    for (gpus, gib, ranks) in [(1,96,6), (1,32,6), (2,96,4), (2,96,0)] {
        let r = report(dir.path(), gpus, gib, ranks, None);
        assert!(!r.wip_launchable() && !r.native_serving_admitted);
        assert!(r.hints.iter().any(|h| h.what.contains("unqualified")), "{:?}", r.hints);
    }
    let r = report(dir.path(), 2, 96, 6, Some(32_768));
    assert!(!r.wip_launchable());
    assert!(r.hints.iter().any(|h| h.what.contains("at least 256K")));
    let r = report(dir.path(), 2, 96, 6, None);
    let memory = r.memory_layout.unwrap();
    assert!(memory.pool_tokens >= 262_144);
    assert!(memory.notes.iter().any(|n| n.contains("zero graphs")));
    assert!(memory.notes.iter().any(|n| n.contains("compiled extent 1048576")));

}

#[test]
fn nvfp4_min_card_admits_native_w4a4_and_bf16_root_stays_reference_only() {
    let dir = snapshot("nvfp4");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let model = Model::open(&cp).unwrap();
    assert_eq!(model.bytes.values().sum::<u64>(),394_164_831_232);
    for (stem,members) in format::group_by_stem(&cp.tensors) {
        let operand = format::detect(&members).unwrap_or_else(|e|panic!("{stem}: {e}"));
        if stem == "model.layers.3.moe.gate_proj" {
            assert!(operand.is_nvfp4());
            assert_eq!(operand.logical,vec![352,1536,4096]);
            assert_eq!(operand.scale.unwrap().grid,vec![352,1536,256]);
        }
    }
    for (context, pool) in [(262_144, Some(262_144)), (1_048_576, None)] {
        let result = report_context(dir.path(), 1, 96, 4, pool, context);
        assert!(result.config_error.is_none(), "{:?}", result.config_error);
        assert!(result.wip_launchable() && result.native_serving_admitted, "{:?}", result.hints);
        assert!(!result.executable());
        assert_eq!(result.min_spark_ranks, Some(4));
        let memory = result.memory_layout.unwrap();
        assert_eq!(memory.devices.len(), 5);
        if let Some(tokens) = pool {
            assert_eq!(memory.pool_tokens, tokens);
        } else {
            let target = crate::placement::PoolPolicy::resolve(&[96 << 30], context, None, 64, false).target;
            assert!((context..=target).contains(&memory.pool_tokens));
            // More free capacity must grow the same default-format pool to its cap.
            let roomy = report_context(dir.path(), 1, 128, 4, None, context);
            assert!(roomy.wip_launchable(), "{:?}", roomy.hints);
            assert_eq!(roomy.memory_layout.unwrap().pool_tokens, target);
        }
        assert!(memory.devices.iter().all(|d| d.used_bytes() <= d.capacity_bytes));
        // The endpoint holds the planner's whole intake span.
        let rows = result.prefill_unit_rows.expect("planned prefill unit rows");
        assert!([2048, 1024, 512].contains(&rows), "{rows}");
        let span = result.prefill_span_rows.expect("planned prefill span rows");
        assert_eq!(span, rows * result.prefill_waves.expect("planned prefill waves"));
        let endpoint = cuteafd_core::SparkEndpoint { ring_depth: None, capacity: span as u32, ingress: cuteafd_core::SparkIngress::Bf16 };
        let rings = cuteafd_core::SparkRings { max_frame_bytes: 64 << 20,
            depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
        let lane_rings = endpoint.ring_bytes(cuteafd_core::ExpertGeometry::STEP5, rings).unwrap();
        for device in &memory.devices[1..] {
            assert!(device.items.iter().any(|item| item.format == "W4A4"));
            let mapped = device.items.iter().find(|item| item.group == "rdma rings").unwrap();
            assert_eq!(mapped.bytes, lane_rings as u64, "one main transport endpoint");
        }
    }
    for (gpus, gib, ranks) in [(2, 96, 4), (1, 32, 4), (1, 96, 6), (1, 96, 0)] {
        let result = report(dir.path(), gpus, gib, ranks, None);
        assert!(!result.wip_launchable() && !result.native_serving_admitted);
        assert!(result.hints.iter().any(|h| h.what.contains("unqualified")), "{:?}", result.hints);
    }
    let dir = snapshot("bf16");
    let r = report(dir.path(),2,96,6,None);
    assert!(r.config_error.is_none(),"{:?}",r.config_error);
    assert!(r.hints.iter().any(|h| h.what.contains("BF16 target format")));
}

#[test]
fn malformed_headers_missing_tensors_and_arrays_fail_with_names() {
    let (config,tensors) = fixture("fp8");
    for key in ["layer_types","rope_theta","partial_rotary_factors","swiglu_limits","swiglu_limits_shared"] {
        let mut config = config.clone();
        config["text_config"][key].as_array_mut().unwrap().pop();
        let dir = tempfile::tempdir().unwrap();
        testing::write_snapshot(dir.path(),&config,&tensors,None);
        let r = report(dir.path(),1,96,6,None);
        assert!(r.config_error.unwrap().contains(key));
    }
    for (key, value) in [("moe_top_k", serde_json::json!(16)),
        ("head_dim_full_attention", serde_json::json!(128)),
        ("moe_router_scaling_factor", serde_json::json!(1.0))] {
        let mut config = config.clone();
        config["text_config"][key] = value;
        let dir = tempfile::tempdir().unwrap();
        testing::write_snapshot(dir.path(), &config, &tensors, None);
        assert!(report(dir.path(), 1, 96, 6, None).config_error.unwrap().contains(key));
    }
    for (name,change) in [("model.layers.3.moe.gate_proj.weight",0),
        ("model.layers.3.self_attn.ssmax_s",1),("model.layers.92.transformer.shared_head.output.weight",2),
        ("vision_model.vit_downsampler2.weight",2),("model.layers.3.moe.gate_proj.weight_scale_inv",1)] {
        let mut tensors = tensors.clone();
        if change == 2 { tensors.retain(|t| t.0 != name); }
        else { let t = tensors.iter_mut().find(|t|t.0 == name).unwrap();
            if change == 0 { t.1 = "BF16"; } else { t.2[0] += 1; }
        }
        let dir = tempfile::tempdir().unwrap();
        testing::write_snapshot(dir.path(),&config,&tensors,None);
        let r = report(dir.path(),1,96,6,None);
        assert!(r.config_error.unwrap().contains(name),"missing diagnostic for {name}");
    }
}

#[test]
fn shared_format_detector_checks_each_expert_scale_axis() {
    let dir = snapshot("fp8");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let groups = format::group_by_stem(&cp.tensors);
    let fp8 = format::detect(&groups["model.layers.3.moe.gate_proj"]).unwrap();
    assert!(fp8.is_fp8_block(128,&[format::ScaleEncoding::F32]));
    assert_eq!(fp8.logical,vec![352,1536,4096]);
    // Existing arbitrary-rank per-tensor FP8 remains a single global scale.
    let mut global = cp.tensors.clone();
    global.iter_mut().find(|t| t.meta.name == "model.layers.3.moe.gate_proj.weight_scale_inv")
        .unwrap().meta.shape.clear();
    let groups = format::group_by_stem(&global);
    let scalar = format::detect(&groups["model.layers.3.moe.gate_proj"]).unwrap();
    assert_eq!(scalar.logical, vec![352,1536,4096]);
    assert!(scalar.scale.unwrap().grid.is_empty());
    let mut bad = cp.tensors.clone();
    let scale = bad.iter_mut().find(|t| t.meta.name == "model.layers.3.moe.gate_proj.weight_scale_inv").unwrap();
    scale.meta.shape[0] -= 1;
    let groups = format::group_by_stem(&bad);
    let error = format::detect(&groups["model.layers.3.moe.gate_proj"]).unwrap_err();
    assert!(error.tensor.ends_with("weight_scale_inv"));
    assert!(error.reason.contains("expert axis"));
}

#[test]
fn missing_headers_stale_index_and_direct_text_detection() {
    let dir = snapshot("fp8");
    let index_path = dir.path().join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&std::fs::read(&index_path).unwrap()).unwrap();
    index["metadata"] = serde_json::json!({"total_size":644_100_979_280u64});
    std::fs::write(&index_path,serde_json::to_vec(&index).unwrap()).unwrap();
    let cp = Checkpoint::open(dir.path()).unwrap();
    let model = Model::open(&cp).unwrap();
    assert!(model.spec.notes.iter().any(|n|n.contains("stale index") && n.contains("641367461632")));
    let mut config = cp.config["text_config"].clone();
    config["architectures"] = serde_json::json!(["Step5ForCausalLM"]);
    let mut cp = cp;
    cp.config = config;
    cp.tensors.retain(|t| !t.meta.name.starts_with("vision_model.") && !t.meta.name.starts_with("vit_large_projector."));
    cp.weight_map.retain(|name, _| !name.starts_with("vision_model.") && !name.starts_with("vit_large_projector."));
    assert!(STEP5.detect(&cp));
    assert_eq!(Model::open(&cp).unwrap().cfg.layers,92);
    cp.tensors.retain(|t|t.meta.name != "model.layers.3.moe.gate_proj.weight");
    let input = PlanInput::from_checkpoint(cp);
    let result = input.plan(&PlanOptions { placement:ExpertPlacement::Sparks { ranks:6 },
        layout:Some(layout::LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), ..Default::default() }), ..Default::default() }).unwrap();
    assert!(!result.executable() && !result.wip_launchable() && !result.native_serving_admitted);
    assert!(result.hints.iter().any(|h| h.what.contains("missing safetensors headers")));
}

#[test]
fn unknown_tensor_and_context_ownership_fail_closed() {
    let (config,mut tensors) = fixture("fp8");
    tensors.push(testing::t("vision_model.unexpected.weight","BF16",&[1]));
    let dir = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir.path(),&config,&tensors,None);
    assert!(report(dir.path(),1,96,6,None).config_error.unwrap().contains("vision_model.unexpected.weight"));
    let dir = snapshot("fp8");
    let r = plan::plan(dir.path(), &PlanOptions { placement:ExpertPlacement::Sparks { ranks:6 },
        attention_placement:Some(crate::placement::AttentionPlacement::Context),
        layout:Some(layout::LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes:vec![96<<30;2], ..Default::default() }),
        ..Default::default() }).unwrap();
    assert!(!r.wip_launchable() && !r.native_serving_admitted);
    assert!(r.config_error.as_deref().is_some_and(|e| e.contains("context")));
    assert!(r.hints.iter().any(|h|h.what.contains("executor not implemented")));
}

#[test]
fn planner_rejects_all_whole_swa_before_serving_admission() {
    use crate::placement::AttentionPlacement;
    let dir = snapshot("fp8");
    let manifest = native_manifest(dir.path());
    for (attention, head_split) in [(Some(AttentionPlacement::Layers), true), (None, false)] {
        let result = plan::plan(dir.path(), &PlanOptions {
            placement: ExpertPlacement::Sparks { ranks: 6 },
            draft_ownership: Some(spec::DraftOwnership::None),
            attention_placement: attention,
            layout: Some(layout::LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; 2],
                head_split, workspace_manifest: Some(manifest.clone()), ..Default::default() }),
            ..Default::default()
        }).unwrap();
        assert!(!result.wip_launchable() && !result.native_serving_admitted);
        assert!(result.hints.iter().any(|hint| hint.what.contains(
            "two-GPU execution requires head-split SWA with whole-owned CSA")),
            "{:?}", result.hints);
    }
}

#[test]
fn routed_catalog_preserves_native_formats_activation_and_target_bounds() {
    use crate::formats::fp8_experts::{ExpertFormat, Fp8Projection};
    for kind in ["fp8", "nvfp4", "bf16"] {
        let dir = snapshot(kind);
        let mut catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let tensors = catalog.fp8().unwrap();
        assert_eq!(catalog.routed_experts().first_layer, 3);
        assert_eq!(catalog.routed_experts().layers, 91);
        assert_eq!(catalog.routed_experts().draft_stages, 0);
        assert!(!tensors.has_layer(91) && !tensors.has_layer(92));
        assert_eq!(tensors.layer_format(3).unwrap(), match kind {
            "fp8" => ExpertFormat::Fp8Block128, "nvfp4" => ExpertFormat::Nvfp4, _ => ExpertFormat::Bf16 });
        for layer in [88,89,90] {
            let tail = tensors.for_layer(layer).unwrap();
            assert_eq!(tail.format(), ExpertFormat::Bf16);
            assert_eq!(tail.slice_bytes(Fp8Projection::Gate,6).unwrap(), (256*4096*2,0));
            assert_eq!(tail.activation(layer), Some(FfnActivation::SiluPostClampFp32Product { limit: 7.0 }));
        }
        catalog.select_draft(crate::plan::spec::DraftOwnership::None).unwrap();
    }
}

#[test]
fn mixed_native_assignment_matches_each_worker_groups_exact_units() {
    use crate::formats::fp8_experts::Slicing;
    use crate::placement::{SparkAssignment, SparkResidentLayout, SparkWorkingSet};
    use cuteafd_core::{SparkEndpoint, SparkIngress, SparkRings};
    let endpoints = [SparkEndpoint { ring_depth: None, capacity: 64, ingress: SparkIngress::Bf16 }];
    let rings = SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
    for kind in ["fp8", "nvfp4", "bf16"] {
        let dir = snapshot(kind);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let tensors = catalog.fp8().unwrap();
        for world in [4, 6] {
            let working = SparkWorkingSet { spark_layers: vec![3, 88, 89, 90],
                spark_ranks: world, prefill_lanes: 1, ..Default::default() };
            // Exercise either exact unit chosen from a mixed checkpoint's headers.
            for block in [16, 128] {
                for rank in 0..world {
                    let assignment = SparkAssignment::from_catalog(&catalog, &working, rank,
                        SparkResidentLayout::Fp8 { world, rank, slicing: Slicing::Blocks(block) },
                        &endpoints, rings).unwrap();
                    let mut resident = 0u64;
                    for slice in &assignment.slices {
                        let view = tensors.for_layer(slice.layer as usize).unwrap();
                        let slicing = view.slicing_for_directory(std::path::Path::new(&format!("tp{world}")), world);
                        assert_eq!(slicing, Slicing::Blocks(view.block()));
                        let map = view.unit_map(world, rank, slicing).unwrap();
                        assert_eq!(slice.unit_map, Some(map.fingerprint()),
                            "{kind} TP{world} rank{rank} layer{} inputK{block}", slice.layer);
                        let (first, width) = view.rank_range_with(world, rank, slicing).unwrap();
                        assert_eq!((slice.first as usize, slice.width as usize, slice.stored_width as usize),
                            (first, width, view.rank_width(world, rank, slicing).unwrap()));
                        resident += (view.weight_read_bytes_with(world, rank, slicing).unwrap()
                            * view.shape().experts) as u64;
                    }
                    assert_eq!(assignment.resident_bytes, resident);
                }
            }
        }
    }
}

#[test]
fn official_nvfp4_layer30_headers_keep_router_and_stacked_parameters_distinct() {
    use crate::formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors};
    use crate::families::deepseek_v41::v41_catalog::RoutedExpertShape;
    use crate::plan::checkpoint::ModuleNumerics;
    let source: Value = serde_json::from_str(include_str!("fixtures/official-nvfp4-layer30.json")).unwrap();
    let config: Value = serde_json::from_str(include_str!("fixtures/official-nvfp4-config.json")).unwrap();
    let tensors: Vec<_> = source["tensors"].as_object().unwrap().iter().map(|(name, meta)| {
        let dtype = match meta["dtype"].as_str().unwrap() {
            "U8" => "U8", "BF16" => "BF16", "F8_E4M3" => "F8_E4M3", "F32" => "F32", other => panic!("{other}")
        };
        testing::t(name, dtype, &meta["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect::<Vec<_>>())
    }).collect();
    let dir = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir.path(), &config, &tensors, None);
    let shape = RoutedExpertShape { layers:91, first_layer:30, experts:352, topk:8,
        hidden:4096, intermediate:1536, draft_stages:0, draft_experts:0 };
    let selected = [30].into_iter().collect();
    let catalog = Fp8ExpertTensors::read_layers(dir.path(), shape, &selected).unwrap();
    assert_eq!(catalog.format(), ExpertFormat::Nvfp4);
    let cp = Checkpoint::open(dir.path()).unwrap();
    assert_eq!(cp.module_numerics("model.layers.30.moe.gate").unwrap(), ModuleNumerics::CheckpointPrecision);
    assert_eq!(cp.module_numerics("model.layers.30.moe.gate_proj").unwrap(), ModuleNumerics::W4A4);
    // Genuine owner exclusions still reject contradictory packed headers.
    let mut excluded = config.clone();
    excluded["quantization_config"]["ignore"] = serde_json::json!(["*layers.30.moe"]);
    std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&excluded).unwrap()).unwrap();
    assert!(Fp8ExpertTensors::read_layers(dir.path(), shape, &selected).unwrap_err().to_string().contains("declared CheckpointPrecision"));
    // A missing companion must not be reinterpreted as checkpoint precision.
    let partial: Vec<_> = tensors.into_iter().filter(|(name, ..)| name != "model.layers.30.moe.gate_proj.input_scale").collect();
    testing::write_snapshot(dir.path(), &config, &partial, None);
    assert!(Fp8ExpertTensors::read_layers(dir.path(), shape, &selected).unwrap_err().to_string().contains("input_scale"));
}

#[test]
fn native_rope_plan_honors_checkpoint_layer_policy_and_factor_one() -> Result<()> {
    for quant in ["fp8", "nvfp4"] {
        let dir = snapshot(quant);
        let cp = Checkpoint::open(dir.path())?;
        let cfg = Config::parse(&cp)?;
        let tables = cfg.rope_tables()?;
        assert_eq!(tables.recipes.len(), 3);
        assert_eq!(tables.extent, 1048576);
        assert_eq!(tables.allocation_bytes, 576 << 20);
        let contract = coordinator::CoordinatorContract::from_checkpoint(&cp)?;
        assert_eq!(contract.rope_tables, tables);
        for layer in 0..cfg.layers + cfg.mtp {
            let recipe = cfg.rope_recipe(layer, false)?;
            assert!(tables.recipes.contains(&recipe));
            assert_eq!(recipe.span, if cfg.full[layer] { 64 } else { 192 });
            assert_eq!(recipe.theta, if cfg.full[layer] { 10000000.0 } else { 10000.0 });
            assert_eq!(recipe.scaling.is_some(), cfg.full[layer]);
            if let Some(scaling) = recipe.scaling { assert_eq!(scaling.factor, 1.0); }
            if cfg.full[layer] {
                let indexer = cfg.rope_recipe(layer, true)?;
                assert_eq!(indexer.span, 32);
                assert_eq!(indexer.scaling, recipe.scaling);
                assert!(tables.recipes.contains(&indexer));
            } else { assert!(cfg.rope_recipe(layer, true).is_err()); }
        }
        for policy in [serde_json::json!([]), serde_json::json!(["sliding_attention"]),
                       serde_json::json!(["full_attention", "sliding_attention"])] {
            let mut changed = Checkpoint::open(dir.path())?;
            changed.config["text_config"]["yarn_only_types"] = policy.clone();
            let cfg = Config::parse(&changed)?;
            assert!(cfg.rope_recipe(0, false)?.scaling.is_some());
            assert_eq!(cfg.rope_recipe(3, false)?.scaling.is_some(), policy.as_array().unwrap().len() != 1);
        }
        for invalid in [serde_json::json!({"rope_type":"unknown"}),
                        serde_json::json!({"rope_type":"llama3", "factor":0})] {
            let mut changed = Checkpoint::open(dir.path())?;
            changed.config["text_config"]["rope_scaling"] = invalid;
            assert!(Config::parse(&changed).is_err());
        }
    }
    Ok(())
}

#[test]
fn spark_admission_is_scoped_to_qualified_native_layers() {
    use crate::plan::experts::{checkpoint_spark_recipe, spark_recipe};
    for kind in ["fp8", "nvfp4", "bf16"] {
        let dir = snapshot(kind);
        let cp = Checkpoint::open(dir.path()).unwrap();
        if kind != "bf16" {
            for world in [4, 6] {
                assert_eq!(checkpoint_spark_recipe(&cp, world, &[3]).unwrap(), spark_recipe(&format!("step5:{kind}"), 1536, world).unwrap());
                assert!(checkpoint_spark_recipe(&cp, world, &(3..88).collect::<Vec<_>>()).is_ok());
            }
            for world in [1, 2, 3, 5, 7, 8] {
                assert!(checkpoint_spark_recipe(&cp, world, &[3]).is_err(), "TP{world}");
            }
        } else {
            for world in [4, 6] {
                assert!(checkpoint_spark_recipe(&cp, world, &[3]).is_err(), "{kind}");
            }
        }
        for world in [4, 6] {
            for layers in [vec![88], vec![89], vec![90], (88..91).collect()] {
                assert_eq!(checkpoint_spark_recipe(&cp, world, &layers).unwrap(),
                    spark_recipe("step5:bf16", 1536, world).unwrap());
            }
            if kind != "bf16" {
                assert!(checkpoint_spark_recipe(&cp, world, &(3..91).collect::<Vec<_>>()).is_ok());
            } else {
                assert!(checkpoint_spark_recipe(&cp, world, &(3..91).collect::<Vec<_>>()).is_err());
            }
            for layers in [vec![], vec![3, 3], vec![4, 3], vec![0], vec![91], vec![95]] {
                assert!(checkpoint_spark_recipe(&cp, world, &layers).is_err());
            }
        }
        let r = report(dir.path(), 2, 96, 6, None);
        assert!(!r.executable());
        assert_eq!(r.wip_launchable(), kind == "fp8", "{:?}", r.hints);
        assert_eq!(r.native_serving_admitted, kind == "fp8");
    }
}

#[test]
fn routed_admission_rechecks_operand_shape_scale_and_role() {
    let dir = snapshot("fp8");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let model = Model::open(&cp).unwrap();
    let groups = format::group_by_stem(&cp.tensors);
    let stem = "model.layers.3.moe.gate_proj";
    let operand = format::detect(&groups[stem]).unwrap();
    let role = TensorRole::layer(Component::RoutedExpert, 3);
    assert!(model.accepts(&role, stem, &mut operand.clone()).is_ok());
    for mutation in 0..5 {
        let mut wrong = operand.clone();
        match mutation {
            0 => wrong.logical[0] -= 1,
            1 => wrong.stored[0] -= 1,
            2 => wrong.scale = None,
            3 => wrong.scale.as_mut().unwrap().encoding = format::ScaleEncoding::Bf16,
            _ => wrong.scale.as_mut().unwrap().cols = 64,
        }
        assert!(model.accepts(&role, stem, &mut wrong).is_err());
    }
    assert!(model.accepts(&TensorRole::layer(Component::RoutedExpert, 4), stem, &mut operand.clone()).is_err());
    assert!(model.accepts(&TensorRole::layer(Component::DenseFfn, 3), stem, &mut operand.clone()).is_err());
    for layer in 88..91 {
        let stem = format!("model.layers.{layer}.moe.gate_proj");
        let operand = format::detect(&groups[&stem]).unwrap();
        let role = TensorRole::layer(Component::RoutedExpert, layer);
        assert!(model.accepts(&role, &stem, &mut operand.clone()).is_ok());
        for mutation in 0..5 {
            let mut wrong = operand.clone();
            match mutation {
                0 => wrong.logical[0] -= 1,
                1 => wrong.stored[0] -= 1,
                2 => wrong.encoding = format::Encoding::F16,
                3 => wrong.companions.push("bias".into()),
                _ => wrong.scale = format::detect(&groups["model.layers.3.moe.gate_proj"]).unwrap().scale,
            }
            assert!(model.accepts(&role, &stem, &mut wrong).is_err());
        }
        assert!(model.accepts(&TensorRole::layer(Component::RoutedExpert, 3), &stem, &mut operand.clone()).is_err());
    }
}

#[test]
fn nvfp4_routed_admission_rechecks_packing_scales_and_role() {
    let dir = snapshot("nvfp4");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let model = Model::open(&cp).unwrap();
    let groups = format::group_by_stem(&cp.tensors);
    for projection in ["gate", "up", "down"] {
        let stem = format!("model.layers.3.moe.{projection}_proj");
        let operand = format::detect(&groups[&stem]).unwrap();
        let role = TensorRole::layer(Component::RoutedExpert, 3);
        assert!(model.accepts(&role, &stem, &mut operand.clone()).is_ok());
        assert_eq!(model.experts(&operand).unwrap().package, "step5:nvfp4");
        for mutation in 0..14 {
            let mut wrong = operand.clone();
            match mutation {
                0 => wrong.logical[0] -= 1,
                1 => wrong.logical[2] -= 1,
                2 => wrong.stored[0] -= 1,
                3 => wrong.encoding = format::Encoding::Int { bits: 8 },
                4 => wrong.scale = None,
                5 => wrong.scale.as_mut().unwrap().encoding = format::ScaleEncoding::Bf16,
                6 => wrong.scale.as_mut().unwrap().cols = 32,
                7 => wrong.scale.as_mut().unwrap().rows = format::RowTiling::Uniform { rows: 2 },
                8 => wrong.global_scale = None,
                9 => wrong.global_scale.as_mut().unwrap().encoding = format::ScaleEncoding::Bf16,
                10 => wrong.input_scale = None,
                11 => wrong.input_scale.as_mut().unwrap().encoding = format::ScaleEncoding::Bf16,
                12 => wrong.companions.push("bias".into()),
                _ => { wrong.logical.pop(); }
            }
            assert!(model.accepts(&role, &stem, &mut wrong).is_err(), "{projection}: mutation {mutation}");
        }
        for role in [TensorRole::layer(Component::RoutedExpert, 4),
            TensorRole::layer(Component::RoutedExpert, 88), TensorRole::layer(Component::DenseFfn, 3)] {
            assert!(model.accepts(&role, &stem, &mut operand.clone()).is_err());
        }
    }
}

/// Startup validates that the peer's replicated routed reply never lands in
/// storage a previous replication used without a head exchange in between.
/// The shared FFN region must keep that true for every owner pattern the
/// planner can produce (contiguous and alternating CSA owners).
#[test]
fn shared_ffn_region_keeps_replicated_replies_apart() {
    use crate::placement::{FfnMode, LayerMode};
    use super::ownership::{self, ExecutionOwnership};
    let dir = snapshot("fp8");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let cfg = Config::parse(&cp).unwrap();
    let csa: Vec<usize> = (0..cfg.layers).filter(|&l| cfg.full[l]).collect();
    for contiguous in [true, false] {
        let modes: Vec<_> = (0..cfg.layers).map(|layer| if !cfg.full[layer] { LayerMode::HeadSplit } else {
            let ordinal = csa.iter().position(|&l| l == layer).unwrap();
            let gpu = if contiguous { u8::from(ordinal >= csa.len() / 2) } else { (ordinal % 2) as u8 };
            LayerMode::Whole { gpu, ffn: FfnMode::Owner }
        }).collect();
        let owners = ExecutionOwnership::new(&cfg, 2, ownership::plan(&modes, 0, 0)).unwrap();
        for rows in [256, 2048] {
            let mut copies = owners.routed_replications().unwrap();
            for (layer, copy) in copies.iter_mut().enumerate() {
                if let Some(copy) = copy {
                    let destination = 1 - usize::from(copy.source);
                    let base = native::rank_attention_bytes(&cfg, &owners, destination, rows).unwrap() as usize;
                    // Live binding: the destination layer's ROUTED plane in its arena.
                    copy.storage = 1 + base + native::routed_layout(rows, rows, cfg.full[layer]).unwrap()[9].0;
                }
            }
            owners.execution().validate_routed_replications(&copies)
                .unwrap_or_else(|error| panic!("contiguous={contiguous} rows={rows}: {error}"));
            // A single reply plane would reuse storage across a CSA/SWA boundary.
            let mut collapsed = copies.clone();
            for copy in collapsed.iter_mut().flatten() { copy.storage = 1; }
            assert!(owners.execution().validate_routed_replications(&collapsed).is_err());
        }
    }
}

