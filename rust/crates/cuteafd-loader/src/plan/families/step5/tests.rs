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

fn report(dir: &std::path::Path, gpus: usize, gib: u64, ranks: usize, pool: Option<u64>) -> PlanReport {
    plan::plan(dir, &PlanOptions { placement: ExpertPlacement::from_spark_ranks(ranks),
        layout: Some(layout::LayoutOptions { rtx_bytes: vec![gib<<30;gpus], pool_tokens: pool, ..Default::default() }),
        ..Default::default() }).unwrap()
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
        let diagnostic = Step5CacheLayout::new(&cfg, 1, 1024, 0, 64).unwrap();
        // Exercise unequal owner counts, a rank with no CSA, and owned SWA.
        for (ranks, policy) in [(1, 0), (2, 0), (2, 1), (2, 2)] {
            let modes: Vec<_> = (0..cfg.layers).map(|l| {
                if ranks == 2 && !cfg.full[l] && policy != 2 { LayerMode::HeadSplit }
                else { LayerMode::Whole { gpu: if ranks == 1 || policy == 1 { 0 }
                    else { ((l / 4) % 2) as u8 }, ffn: FfnMode::Owner } }
            }).collect();
            let head = if ranks == 1 { 0 } else { 1 };
            let owners = ExecutionOwnership::new(&cfg, ranks, ownership::plan(&modes, 0, head)).unwrap();
            assert_eq!(owners.routed_wave_ranks().unwrap(),
                if ranks == 1 || policy == 1 { vec![0] } else { vec![0, 1] });
            let layout = Step5CacheLayout::for_execution(&cfg, &owners, 1024, 0, 64).unwrap();
            assert_eq!(layout.geometry.placement, if ranks == 1 { KvPlacement::SingleDevice }
                else { KvPlacement::PartitionedLayers });
            assert_eq!(layout.geometry.ranks.iter().map(|r| r.persistent_unit_bytes).sum::<u64>(),
                diagnostic.geometry.ranks[0].persistent_unit_bytes);
            let weights: Vec<_> = (0..ranks).map(|rank| RankWeightLayout::for_execution(&cp, rank, &owners).unwrap()).collect();
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
                    let spans = layout.snapshot_spans(rank, frontier).unwrap();
                    let expected = 8 + state.sliding.iter().map(|s| 2 * frontier.min(512) * s.key.bytes / layout.ring.rows).sum::<u64>()
                        + state.regions.len() as u64 * (2 * 8 * 256 * 2 + frontier / 8 * 256);
                    assert_eq!(spans.iter().map(|s| s.bytes).sum::<u64>(), expected);
                    let copies = layout.snapshot_copies(rank, frontier, 3, 2, 4, 3).unwrap();
                    assert_eq!(copies.iter().map(|s| s.active.bytes).sum::<u64>(), expected);
                    assert!(copies.iter().all(|c| c.active.offset + c.active.bytes <= state.bytes * 3
                        && c.mark.offset >= state.bytes * 3 && c.mark.offset + c.mark.bytes <= state.bytes * 4));
                    assert_eq!(layout.csa_tail_bytes(rank, frontier % 64).unwrap(), cost.persistent_unit_bytes / 64 * (frontier % 64));
                }
            }
            let mtp = Step5CacheLayout::for_execution(&cfg, &owners, 1024, 3, 64).unwrap();
            for (rank, state) in mtp.ranks.iter().enumerate() {
                assert_eq!(state.mtp_hidden.is_some(), rank == head as usize);
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
    assert!(crate::serving_capacity::step5::Step5CacheLayout::for_execution(&changed, &owners, 1024, 0, 64).is_err());
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
    let copies = ownership.routed_replications(0).unwrap();
    assert_eq!(copies.iter().flatten().count(), 66);
    for (ordinal, (layer, copy)) in copies.iter().enumerate().filter_map(|(l, c)| c.map(|c| (l, c))).enumerate() {
        assert_eq!(copy.source, 0);
        assert_eq!(copy.slot, 4 + ordinal % 2);
        assert_eq!(copy.storage, layer + 1);
        assert!(cfg.moe[layer] && !cfg.full[layer]);
    }
    ownership.execution().validate_routed_replications(&copies).unwrap();
    assert!(ownership.routed_replications(1).is_err());
    let mut spec = ownership.execution().spec();
    spec.lanes = 2;
    let mut placement = crate::placement::Placement {
        pool_tokens: 0, pool_limits: crate::placement::PoolLimits(vec![]),
        attention_placement: crate::placement::AttentionPlacement::Layers, attention_by_kind: vec![],
        peer_row_bytes: 0, onboard_layers: 0,
        layers: modes.iter().map(|&mode| crate::placement::LayerAssignment {
            mode, experts: crate::placement::ExpertHome::Spark }).collect(),
        movables: vec![], expert_ranges: vec![], residual: ownership.execution().homes().to_vec(),
        hops: vec![], tp2: None, items: vec![],
    };
    placement.hops = crate::placement::plan_hops(&modes, &spec).unwrap();
    let two = ExecutionOwnership::new(&cfg, 2,
        crate::placement::LayerExecutionPlan::new(&placement, spec).unwrap()).unwrap();
    let lanes = [two.routed_replications(0).unwrap(), two.routed_replications(1).unwrap()];
    for copy in lanes[0].iter().flatten() {
        assert!(lanes[1].iter().flatten().all(|peer| peer.slot != copy.slot && peer.storage != copy.storage));
    }
    assert!(two.routed_replications(2).is_err());
}

#[test]
fn exact_cache_layout_charges_dense_summaries_and_every_retained_byte() {
    use crate::serving_capacity::step5::{Step5CacheLayout, PAGE_ROWS, REGION_ROWS};
    let dir = snapshot("fp8");
    let cp = Checkpoint::open(dir.path()).unwrap();
    let cfg = Model::open(&cp).unwrap().cfg;
    for ranks in [1, 2] {
        let layout = Step5CacheLayout::new(&cfg, ranks, 1 << 20, 3, 64).unwrap();
        assert_eq!(layout.mtp_hidden_rows, 516);
        assert_eq!(layout.window(), 512);
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
                let spans = layout.snapshot_spans(rank, frontier).unwrap();
                assert!(spans.iter().all(|s| s.offset.checked_add(s.bytes).unwrap() <= state.bytes));
                let summary_bytes = if rank == 0 { 23 * (frontier / REGION_ROWS) * 256 } else { 0 };
                let maximum: u64 = layout.snapshot_spans(rank, 1 << 20).unwrap().iter().map(|s| s.bytes).sum();
                let missing_rings: u64 = state.sliding.iter().filter(|s| !s.mtp)
                    .map(|s| 2 * (512 - frontier.min(512)) * s.key.bytes / layout.ring.rows).sum();
                let expected = maximum - if rank == 0 { summaries - summary_bytes } else { 0 } - missing_rings;
                assert_eq!(spans.iter().map(|s| s.bytes).sum::<u64>(), expected);
                assert!(state.regions.iter().all(|region| spans.contains(&region.pending_key)
                    && spans.contains(&region.pending_z)));
                for ring in &state.sliding {
                    for component in [ring.key, ring.value] {
                        let retained: u64 = spans.iter().filter(|s| s.offset >= component.offset
                            && s.offset < component.offset + component.bytes).map(|s| s.bytes).sum();
                        assert_eq!(retained, if ring.mtp { component.bytes }
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
        assert!(layout.snapshot_spans(ranks, 0).is_err());
        assert!(layout.snapshot_spans(0, (1 << 20) + 1).is_err());
        assert!(layout.csa_tail_bytes(0, 65).is_err());
        let mut occupied = Vec::new();
        for sequence in 0..3 {
            let copies = layout.snapshot_copies(0, 65, 3, sequence, 4, sequence).unwrap();
            let spans = layout.snapshot_spans(0, 65).unwrap();
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
        assert!(layout.snapshot_copies(0, 0, 3, 3, 4, 0).is_err());
        assert!(layout.snapshot_copies(0, 0, 3, 0, 4, 4).is_err());
        assert!(layout.snapshot_copies(0, 0, u64::MAX, 0, 4, 0).is_err());
    }
    assert!(Step5CacheLayout::new(&cfg, 3, 1024, 0, 64).is_err());
    assert!(Step5CacheLayout::new(&cfg, 1, 0, 0, 64).is_err());
    assert!(Step5CacheLayout::new(&cfg, 1, 1024, 4, 64).is_err());
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
    assert!(result.hints.iter().any(|hint|hint.how.contains("CSA-8")));
    assert!(result.hints.iter().any(|hint|hint.how.contains("BF16-tail")));
    assert_eq!(result.min_spark_ranks,Some(6));
}

#[test]
fn card_memory_conclusions_and_request_vs_pool_extent() {
    let dir = snapshot("fp8");
    for gpus in [1,2] {
        let r = report(dir.path(),gpus,96,6,Some(262_144));
        assert!(r.config_error.is_none(),"{:?}",r.config_error);
        assert!(!r.executable());
        let memory = r.memory_layout.unwrap();
        assert_eq!(memory.pool_tokens,262_144,"{:?}",r.hints);
        assert!(memory.devices.iter().all(|d| d.used_bytes() <= d.capacity_bytes));
    }
    let r = report(dir.path(),1,96,6,Some(1<<20));
    assert!(r.hints.iter().any(|h| h.what.contains("coordinator BF16 pool lower bound refused")),"{:?}",r.hints);
    let r = report(dir.path(),2,96,6,Some(1<<20));
    assert_eq!(r.memory_layout.unwrap().pool_tokens,1<<20);
    let r = report(dir.path(),1,32,6,None);
    assert!(r.hints.iter().any(|h| h.what.contains("5090 refused")));
    let r = report(dir.path(),1,96,4,None);
    assert!(r.hints.iter().any(|h| h.what.contains("TP4 refused")));
    let r = report(dir.path(),2,96,0,None);
    assert!(r.hints.iter().any(|h| h.what.contains("no 0-Spark")));
    let r = report(dir.path(),1,96,6,Some(32_768));
    assert!(r.hints.iter().any(|h| h.what.contains("below 256K")));
    let r = report(dir.path(),1,96,6,None);
    let memory = r.memory_layout.unwrap();
    assert!(memory.pool_tokens >= 262_144 && memory.pool_tokens < 1<<20);
    assert!(memory.notes.iter().any(|n| n.contains("71392 bytes/token")));
    assert!(memory.notes.iter().any(|n| n.contains("36064 bytes/token")));
    assert!(memory.notes.iter().any(|n| n.contains("Maximum request 1048576")));
    assert!(memory.notes.iter().any(|n| n.contains("14155776 bytes per admitted sequence")));
}

#[test]
fn nvfp4_stacked_scales_and_bf16_root_are_named_not_executed() {
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
    let result = report(dir.path(),1,96,4,Some(262_144));
    assert!(result.config_error.is_none(),"{:?}",result.config_error);
    assert_eq!(result.min_spark_ranks,Some(4));
    assert_eq!(result.memory_layout.unwrap().pool_tokens,262_144);
    let dir = snapshot("bf16");
    let r = report(dir.path(),2,96,6,None);
    assert!(r.config_error.is_none(),"{:?}",r.config_error);
    assert!(r.hints.iter().any(|h| h.what.contains("TP6 refused")));
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
        layout:Some(Default::default()), ..Default::default() }).unwrap();
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
        layout:Some(layout::LayoutOptions { rtx_bytes:vec![96<<30;2], ..Default::default() }),
        ..Default::default() }).unwrap();
    assert!(!r.placement_supported);
    assert!(r.hints.iter().any(|h|h.what.contains("context/layer ownership")));
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
        if kind == "fp8" {
            for world in [4, 6] {
                assert_eq!(checkpoint_spark_recipe(&cp, world, &[3]).unwrap(), spark_recipe("step5:fp8", 1536, world).unwrap());
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
            if kind == "fp8" {
                assert!(checkpoint_spark_recipe(&cp, world, &(3..91).collect::<Vec<_>>()).is_ok());
            } else {
                assert!(checkpoint_spark_recipe(&cp, world, &(3..91).collect::<Vec<_>>()).is_err());
            }
            for layers in [vec![], vec![3, 3], vec![4, 3], vec![0], vec![91], vec![95]] {
                assert!(checkpoint_spark_recipe(&cp, world, &layers).is_err());
            }
        }
        let r = report(dir.path(), 2, 96, 6, Some(262_144));
        assert!(!r.executable() && !r.wip_launchable() && !r.native_serving_admitted);
        for component in ["dense_ffn", "attention", "routed_expert"] {
            assert!(r.hints.iter().any(|hint| hint.what.contains(component)), "{component}: {:?}", r.hints);
        }
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
