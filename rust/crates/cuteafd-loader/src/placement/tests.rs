use super::*;

const GIB: u64 = 1 << 30;
const UNIT: u64 = 4 << 20;

/// Two equal GPUs measured with `free` bytes each, `layers` MoE layers of
/// 1 GiB (+256 MiB staging), 4 MiB per pool unit of 256 tokens over all
/// layers (2M tokens: 8192 units, 32 GiB).
fn request(gpus: usize, free: u64, layers: usize, sparks: usize, onboard: Onboard) -> PlacementRequest {
    PlacementRequest {
        attention_placement: None,
        context_buffers: ContextBuffers::default(),
        layers_first_gpu: 0,
        inventory: Inventory {
            gpus: vec![GpuBudget { capacity_bytes: 96 * GIB, headroom_bytes: 0,
                baseline: Baseline::Measured { free_bytes: free } }; gpus],
            spark_ranks: sparks, peer_access: gpus == 2 },
        pool: PoolPolicy::resolve(&vec![96 * GIB; gpus], 131_072, None, 256, sparks == 0),
        layers: (0..layers).map(|_| LayerDemand { kind: AttentionClass::Csa, weights: ModeBytes::default(),
            kv_unit: ModeBytes::default().into(), colocate: None,
            fixed_bytes: ModeBytes::default(), context_indexer: false,
            experts: Some(ExpertCost { whole: Bytes2 { resident: GIB, staging: GIB / 4 }, half: [Bytes2::default(); 2], tp2: false, spark_ok: true }),
            modes: vec![LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] }).collect(),
        pool_overhead: vec![UNIT; gpus],
        fixed: Vec::new(),
        movables: Vec::new(),
        expert_workspace: GIB / 4,
        tp2_workspace: [0; 2],
        onboard,
        expert_gpus: gpus,
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit], by_kind: Vec::new() },
        hops: HopSpec::default(),
        executor: families::DEEPSEEK_V4,
    }
}

#[test]
fn v4_defaults_to_pool_first_and_retains_explicit_max() {
    assert_eq!(families::deepseek_v4::default_onboard(), Onboard::Auto);
    assert_eq!("max".parse::<Onboard>(), Ok(Onboard::ExpertsFirst { pool_floor: EXPERTS_FIRST_POOL_FLOOR }));
}

#[test]
fn pool_precedes_contiguous_layers_on_both_gpus() {
    // 2M tokens = 8192 units x 4 MiB = 32 GiB per GPU.
    let placement = solve(&request(2, 44 * GIB, 60, 4, Onboard::Auto)).unwrap();
    assert_eq!(placement.pool_tokens, 2 << 20);
    // 12 GiB left: 0.25 workspace + n GiB + 0.25 staging <= 12 -> 11 layers per GPU.
    assert_eq!(placement.expert_ranges.iter().map(|r| (r.first, r.layers)).collect::<Vec<_>>(), [(0, 11), (11, 11)]);
    assert_eq!(placement.onboard_layers, 22);
    assert!(placement.layers.iter().all(|l| l.mode == LayerMode::HeadSplit));
    assert_eq!(placement.layers[11].experts, ExpertHome::RtxWhole { gpu: 1 });
    assert_eq!(placement.layers[22].experts, ExpertHome::Spark);
}

#[test]
fn one_gpu_runs_whole_layers_and_shrinks_the_pool_before_failing() {
    let placement = solve(&request(1, 12 * GIB, 60, 4, Onboard::Auto)).unwrap();
    assert!(placement.layers.iter().all(|l| l.mode == LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }));
    assert_eq!(placement.pool_tokens, 12 * GIB / UNIT * 256);
    assert_eq!(placement.onboard_layers, 0);
}

#[test]
fn fixed_onboard_makes_the_pool_the_output() {
    let placement = solve(&request(2, 44 * GIB, 60, 4, Onboard::Layers(10))).unwrap();
    assert_eq!(placement.onboard_layers, 10);
    // The split that leaves the most pool: 5 + 5 (each GPU 0.25 + 5 + 0.25 GiB).
    assert_eq!(placement.expert_ranges.iter().map(|r| r.layers).collect::<Vec<_>>(), [5, 5]);
    assert_eq!(placement.pool_tokens, (44 * GIB - (5 * GIB + GIB / 2)) / UNIT * 256);
    assert!(placement.pool_tokens > 2 << 20);
    // Fraction rounds to the nearest layer; 0 keeps every expert on the Sparks.
    assert_eq!(solve(&request(2, 44 * GIB, 60, 4, Onboard::Fraction(0.25))).unwrap().onboard_layers, 15);
    let none = solve(&request(2, 44 * GIB, 60, 4, Onboard::Layers(0))).unwrap();
    assert_eq!((none.onboard_layers, none.expert_ranges[0].peak_bytes), (0, 0));
    // More than fits beside the fixed demands is a refusal that names how many fit.
    assert!(matches!(solve(&request(2, 44 * GIB, 100, 4, Onboard::Layers(90))),
        Err(PlacementError::ExpertLayers { requested: 90, placed: 86 })));
    // Fitting the layers but not the floor (the compiled context) is refused too.
    assert!(matches!(solve(&request(1, 33 * GIB, 60, 4, Onboard::Layers(32))),
        Err(PlacementError::BelowFloor { floor: 131_072, layers: 32, .. })));
}

#[test]
fn spark_free_layouts_need_every_layer_and_the_agentic_floor() {
    let placement = solve(&request(2, 44 * GIB, 30, 0, Onboard::Auto)).unwrap();
    assert_eq!(placement.onboard_layers, 30);
    assert!(placement.pool_tokens >= pool::AGENTIC_FLOOR_TOKENS);
    assert!(matches!(solve(&request(2, 44 * GIB, 100, 0, Onboard::Auto)), Err(PlacementError::SparkFree { .. })));
    assert!(matches!(solve(&request(2, 44 * GIB, 30, 0, Onboard::Layers(10))), Err(PlacementError::SparkFree { .. })));
    // All routed layers with no pool room left is below the floor.
    assert!(matches!(solve(&request(2, 18 * GIB, 30, 0, Onboard::Auto)), Err(PlacementError::BelowFloor { .. })
        | Err(PlacementError::Mandatory { .. })));
}

#[test]
fn explicit_pool_is_strict_and_movables_share_the_arena() {
    let mut req = request(2, 44 * GIB, 60, 4, Onboard::Auto);
    req.pool.requested = Some(4 << 20);
    assert!(matches!(solve(&req), Err(PlacementError::PoolDoesNotFit { requested: 4194304, .. })));
    req.pool.requested = Some(1 << 20);
    req.movables.push(Movable { id: MovableId::DsparkExperts, allowed: vec![0], expert_arena: true,
        parts: vec![Bytes2 { resident: 2 * GIB, staging: GIB / 4 }; 3] });
    let placement = solve(&req).unwrap();
    assert_eq!(placement.pool_tokens, 1 << 20);
    assert_eq!(placement.movables, [(MovableId::DsparkExperts, 0)]);
    // GPU0 holds the 6 GiB of stage experts in its arena: 6 fewer layers.
    assert_eq!(placement.expert_ranges[0].layers + 6, placement.expert_ranges[1].layers);
    // A movable that cannot fit at all is a refusal naming its GPU.
    req.movables[0].parts = vec![Bytes2 { resident: 50 * GIB, staging: 0 }];
    assert!(matches!(solve(&req), Err(PlacementError::Mandatory { gpu: 0, .. })));
}

#[test]
fn experts_stay_on_gpu0_when_the_peer_cannot_run_them() {
    let mut req = request(2, 44 * GIB, 60, 4, Onboard::Auto);
    req.expert_gpus = 1;
    let placement = solve(&req).unwrap();
    assert_eq!(placement.pool_tokens, 2 << 20);
    assert_eq!((placement.expert_ranges[0].layers, placement.expert_ranges[1].layers), (11, 0));
    req.onboard = Onboard::Layers(5);
    let fixed = solve(&req).unwrap();
    assert_eq!((fixed.expert_ranges[0].layers, fixed.expert_ranges[1].layers), (5, 0));
    // Without a pool reserved first GPU0 alone holds 43, never GPU1.
    req.onboard = Onboard::Layers(50);
    assert!(matches!(solve(&req), Err(PlacementError::ExpertLayers { requested: 50, placed: 43 })));
}

#[test]
fn experts_first_fills_layers_above_the_pool_floor() {
    let req = request(2, 44 * GIB, 60, 4, Onboard::ExpertsFirst { pool_floor: 262_144 });
    let placement = solve(&req).unwrap();
    // 262K tokens = 1024 units x 4 MiB = 4 GiB; 40 GiB of arena -> 39 layers per GPU (0.25 + 39 + 0.25 <= 40).
    assert_eq!(placement.onboard_layers, 60);
    assert!(placement.pool_tokens >= 262_144 && placement.pool_tokens <= 2 << 20);
    let mut one = request(1, 44 * GIB, 60, 4, Onboard::ExpertsFirst { pool_floor: 262_144 });
    one.expert_gpus = 1;
    let single = solve(&one).unwrap();
    assert_eq!(single.onboard_layers, 39);
    assert!(single.pool_tokens >= 262_144);
    let auto = solve(&request(1, 44 * GIB, 60, 4, Onboard::Auto)).unwrap();
    assert!(auto.onboard_layers < single.onboard_layers && auto.pool_tokens > single.pool_tokens);
    // An explicit pool is exact: the most layers that still leave it.
    one.pool.requested = Some(1 << 20);
    let explicit = solve(&one).unwrap();
    assert_eq!(explicit.pool_tokens, 1 << 20);
    assert_eq!(explicit.onboard_layers, 27);
}

#[test]
fn onboard_parses_the_launcher_spellings() {
    assert_eq!("auto".parse::<Onboard>(), Ok(Onboard::Auto));
    assert_eq!("12".parse::<Onboard>(), Ok(Onboard::Layers(12)));
    assert_eq!("50%".parse::<Onboard>(), Ok(Onboard::Fraction(0.5)));
    assert_eq!("all".parse::<Onboard>(), Ok(Onboard::Fraction(1.0)));
    assert_eq!("max".parse::<Onboard>(), Ok(Onboard::ExpertsFirst { pool_floor: 262_144 }));
    assert!("150%".parse::<Onboard>().is_err() && "x".parse::<Onboard>().is_err());
    assert_eq!(Onboard::Fraction(0.5).layers(43), Some(22));
    assert_eq!(Onboard::Layers(99).layers(43), Some(43));
    assert_eq!(Onboard::Layers(12).to_string(), "12");
}

const W0: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
const W1: LayerMode = LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner };
const S0: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Split };
const S1: LayerMode = LayerMode::Whole { gpu: 1, ffn: FfnMode::Split };
/// V4/V4.1-sized mHC hop: `[4096,4,4096]` BF16 per lane slot = 128 MiB.
const MHC: HopSpec = HopSpec { row_bytes: 4 * 4096 * 2, rows: 4096, lanes: 2, entry_gpu: 0, head_gpu: 0 };
const ANY: ExecutorModes = ExecutorModes { family: "test", modes: &[LayerMode::HeadSplit, W0, W1, S0, S1], hops: true };

#[test]
fn residual_transitions_follow_the_section_3_table() {
    use ResidualHome::*;
    let t = |from: ResidualHome, next| { let t = from.transition(next); (t.before, t.after_attention, t.home) };
    // Into a head split: an owned residual is broadcast first; the split leaves it replicated.
    assert_eq!(t(Replicated, LayerMode::HeadSplit), (None, None, Replicated));
    assert_eq!(t(Owned(0), LayerMode::HeadSplit), (Some((0, 1, HopKind::Broadcast)), None, Replicated));
    assert_eq!(t(Owned(1), LayerMode::HeadSplit), (Some((1, 0, HopKind::Broadcast)), None, Replicated));
    // A split FFN needs its input on the peer after attention; the all-reduce replicates.
    assert_eq!(t(Replicated, S1), (None, Some((1, 0)), Replicated));
    assert_eq!(t(Owned(1), S1), (None, Some((1, 0)), Replicated));
    assert_eq!(t(Owned(0), S1), (Some((0, 1, HopKind::Boundary)), Some((1, 0)), Replicated));
    // An owner FFN keeps the residual on its GPU; crossing owners is one boundary hop.
    assert_eq!(t(Replicated, W1), (None, None, Owned(1)));
    assert_eq!(t(Owned(1), W1), (None, None, Owned(1)));
    assert_eq!(t(Owned(0), W1), (Some((0, 1, HopKind::Boundary)), None, Owned(1)));
}

#[test]
fn layer_ranges_hop_once_per_boundary_and_back_to_the_head() {
    // V4.1: layers 0-19 on GPU0, 20-39 on GPU1, head on GPU0.
    let modes: Vec<_> = (0..40).map(|l| if l < 20 { W0 } else { W1 }).collect();
    let hops = plan_hops(&modes, &MHC);
    assert_eq!(hops.iter().map(|h| (h.at, h.from, h.to, h.kind)).collect::<Vec<_>>(), [
        (HopPoint::BeforeLayer(20), 0, 1, HopKind::Boundary),
        (HopPoint::Exit, 1, 0, HopKind::Boundary),
    ]);
    // One receive pair per lane on each GPU: 2 lanes x 1 hop x 128 MiB.
    assert_eq!(hop_buffer_bytes(&hops, &MHC, 2), Some(vec![256 << 20, 256 << 20]));
    // Head on GPU1 with the last range: no exit hop.
    let tail = plan_hops(&modes, &HopSpec { head_gpu: 1, ..MHC });
    assert_eq!(tail.len(), 1);
    assert_eq!(hop_buffer_bytes(&tail, &MHC, 2), Some(vec![0, 256 << 20]));
}

#[test]
fn head_split_stacks_only_hop_at_entry_and_charge_nothing() {
    // Today's V4/GLM/MiMo head split: the embedding broadcast lands in the step input.
    let hops = plan_hops(&[LayerMode::HeadSplit; 61], &MHC);
    assert_eq!(hops.len(), 1);
    assert_eq!((hops[0].at, hops[0].kind, hops[0].charged()), (HopPoint::Entry, HopKind::Broadcast, false));
    assert_eq!(hop_buffer_bytes(&hops, &MHC, 2), Some(vec![0, 0]));
}

#[test]
fn mixed_split_and_whole_layers_broadcast_into_split_ffns() {
    // GLM Flash's mixed policy: KDA head split, DSA layers whole on alternating GPUs with a split FFN.
    let modes = [LayerMode::HeadSplit, S0, LayerMode::HeadSplit, S1, LayerMode::HeadSplit];
    let hops = plan_hops(&modes, &MHC);
    assert_eq!(hops.iter().map(|h| (h.at, h.from, h.to)).collect::<Vec<_>>(), [
        (HopPoint::Entry, 0, 1), (HopPoint::AfterAttention(1), 0, 1), (HopPoint::AfterAttention(3), 1, 0),
    ]);
    // Each GPU receives one charged hop per lane.
    assert_eq!(hop_buffer_bytes(&hops, &MHC, 2), Some(vec![256 << 20, 256 << 20]));
    // Many hops into one GPU alternate between two slots per lane.
    let many = plan_hops(&[S0, S0, S0, S0], &MHC);
    assert_eq!(many.len(), 4);
    assert_eq!(hop_buffer_bytes(&many, &MHC, 2), Some(vec![0, 512 << 20]));
}

#[test]
fn placement_records_residual_homes_and_charges_hops_as_fixed_demands() {
    let mut req = request(2, 44 * GIB, 4, 4, Onboard::Layers(0));
    req.hops = MHC;
    req.executor = ANY;
    req.policy.default = vec![W0];
    for (index, layer) in req.layers.iter_mut().enumerate() {
        layer.modes = vec![if index < 2 { W0 } else { W1 }];
    }
    let ranges = solve(&req).unwrap();
    assert_eq!(ranges.residual, [ResidualHome::Owned(0), ResidualHome::Owned(0), ResidualHome::Owned(0),
        ResidualHome::Owned(1), ResidualHome::Owned(1)]);
    assert_eq!(ranges.hops.len(), 2);
    for gpu in 0..2 {
        assert!(ranges.items[gpu].iter().any(|i| i.group == "residual hops" && i.bytes == 256 << 20), "gpu{gpu}");
    }
    assert!(ranges.summary().ends_with("; 2 residual hops"));
    // The hop buffers come out of the pool: 256 MiB = 64 units of 4 MiB fewer.
    let mut split = req.clone();
    split.hops = HopSpec::default();
    assert_eq!(solve(&split).unwrap().pool_tokens - ranges.pool_tokens, 64 * 256);
    // Today's V4 request: head split, the residual replicated after layer 0, nothing charged.
    let v4 = solve(&request(2, 44 * GIB, 4, 4, Onboard::Auto)).unwrap();
    assert_eq!(v4.residual[0], ResidualHome::Owned(0));
    assert!(v4.residual[1..].iter().all(|h| *h == ResidualHome::Replicated));
    assert_eq!(v4.hops.iter().map(|h| h.at).collect::<Vec<_>>(), [HopPoint::Entry]);
    assert!(!v4.items.iter().flatten().any(|i| i.group == "residual hops"));
    // One GPU never hops.
    let one = solve(&request(1, 44 * GIB, 4, 4, Onboard::Auto)).unwrap();
    assert!(one.hops.is_empty() && one.residual.iter().all(|h| *h == ResidualHome::Owned(0)));
}

#[test]
fn executors_refuse_modes_they_cannot_run_at_plan_time() {
    // V4 runs head split or whole-on-GPU0 only: a layer that allows only GPU1 ownership is refused.
    let mut req = request(2, 44 * GIB, 4, 4, Onboard::Auto);
    req.layers[2].modes = vec![W1];
    assert_eq!(solve(&req), Err(PlacementError::NoMode { layer: 2, allowed: vec![W1] }));
    // The policy's preference falls through to a mode the executor runs.
    let mut req = request(2, 44 * GIB, 4, 4, Onboard::Auto);
    req.policy.default = vec![W1, LayerMode::HeadSplit];
    for layer in &mut req.layers { layer.modes = vec![W1, LayerMode::HeadSplit]; }
    assert!(solve(&req).unwrap().layers.iter().all(|l| l.mode == LayerMode::HeadSplit));
    // An executor without hop points refuses modes that need them (ranges need a boundary hop).
    let mut req = request(2, 44 * GIB, 4, 4, Onboard::Layers(0));
    req.executor = ExecutorModes { family: "nohops", modes: &[W0, W1], hops: false };
    req.policy.default = vec![W0];
    req.layers[3].modes = vec![W1];
    assert!(matches!(solve(&req), Err(PlacementError::UnsupportedHop { family: "nohops",
        hop: Hop { at: HopPoint::BeforeLayer(3), from: 0, to: 1, .. } })));
    // Without peer access, GPU1-owned and split layers are not executable at all.
    let mut req = request(2, 44 * GIB, 4, 4, Onboard::Auto);
    req.inventory.peer_access = false;
    req.executor = ANY;
    req.policy.default = vec![S1, W1, W0];
    for layer in &mut req.layers { layer.modes = vec![S1, W1, W0]; }
    assert!(solve(&req).unwrap().layers.iter().all(|l| l.mode == W0));
    // Engines check the placement they are handed against the same set.
    let mut handed = solve(&request(2, 44 * GIB, 4, 4, Onboard::Auto)).unwrap();
    assert_eq!(families::DEEPSEEK_V4.check(&handed), Ok(()));
    handed.layers[1].mode = S1;
    assert_eq!(families::DEEPSEEK_V4.check(&handed),
        Err(PlacementError::UnsupportedMode { family: "deepseek_v4", layer: 1, mode: S1 }));
    assert!(families::DEEPSEEK_V41.runs(W1) && !families::QWEN4.runs(LayerMode::HeadSplit));
    assert_eq!(families::executor("glm5_flash").map(|e| e.hops), Some(false));
}

fn tp2_request(free: [u64; 2], onboard: Onboard) -> PlacementRequest {
    let mut req = request(2, free[0], 60, 4, onboard);
    req.inventory.gpus[1].baseline = Baseline::Measured { free_bytes: free[1] };
    req.tp2_workspace = [GIB / 2, GIB / 4];
    for layer in &mut req.layers {
        let cost = layer.experts.as_mut().unwrap();
        cost.tp2 = true;
        cost.half = [Bytes2 { resident: GIB / 2, staging: GIB / 8 }; 2];
    }
    req
}

#[test]
fn tp2_auto_fixed_max_use_the_tighter_rank() {
    for free in [[44 * GIB; 2], [48 * GIB, 44 * GIB], [44 * GIB, 48 * GIB]] {
        for onboard in [Onboard::Auto, Onboard::Layers(10), Onboard::Fraction(1.0 / 6.0),
            Onboard::ExpertsFirst { pool_floor: 262_144 }] {
            let req = tp2_request(free, onboard);
            let placement = solve(&req).unwrap();
            assert!(placement.expert_ranges.iter().all(|r| r.layers == 0 && r.peak_bytes == 0));
            assert!(placement.layers.iter().all(|l| !matches!(l.experts, ExpertHome::RtxWhole { .. })));
            let t = placement.tp2.unwrap();
            assert_eq!(t.layers, placement.onboard_layers);
            assert!(placement.layers[..t.layers].iter().all(|l| l.experts == ExpertHome::RtxTp2));
            assert_eq!(t.peak_bytes, [GIB / 2 + t.layers as u64 * GIB / 2 + GIB / 8,
                GIB / 4 + t.layers as u64 * GIB / 2 + GIB / 8]);
            if onboard == Onboard::Auto {
                assert_eq!(placement.pool_tokens, 2 << 20);
                assert_eq!(t.layers, if free[0] == 44 * GIB { 22 } else { 23 });
            } else if onboard.layers(60).is_some() {
                assert_eq!(t.layers, 10);
                let fit = (0..2).map(|g| (free[g] - t.peak_bytes[g]) / UNIT * 256).min().unwrap();
                assert_eq!(placement.pool_tokens, fit);
            } else { assert_eq!(t.layers, 60); }
        }
    }
    let mut req = tp2_request([12 * GIB, 48 * GIB], Onboard::Auto);
    let p = solve(&req).unwrap();
    assert_eq!(p.pool_tokens, 12 * GIB / UNIT * 256);
    assert!(p.tp2.is_none());
    req.onboard = Onboard::Layers(30);
    assert!(matches!(solve(&req), Err(PlacementError::ExpertLayers { .. })));
    req = tp2_request([31 * GIB; 2], Onboard::Layers(60));
    assert!(matches!(solve(&req), Err(PlacementError::BelowFloor { .. })));
}

#[test]
fn tp2_keeps_dspark_tp1_arena_separate() {
    let mut req = tp2_request([44 * GIB; 2], Onboard::Auto);
    req.movables.push(Movable { id: MovableId::DsparkExperts, allowed: vec![0], expert_arena: true,
        parts: vec![Bytes2 { resident: 2 * GIB, staging: GIB / 4 }; 3] });
    let p = solve(&req).unwrap();
    assert_eq!(p.movables, [(MovableId::DsparkExperts, 0)]);
    assert_eq!(p.expert_ranges[0].peak_bytes, 6 * GIB + GIB / 2);
    assert_eq!(p.expert_ranges[1].peak_bytes, 0);
    let t = p.tp2.unwrap();
    assert_eq!(t.layers, 9);
    for g in 0..2 {
        let charge: u64 = p.items[g].iter().filter(|i| i.category == cuteafd_core::memory_layout::Category::Experts).map(|i| i.bytes).sum();
        assert_eq!(charge, t.peak_bytes[g] + p.expert_ranges[g].peak_bytes);
    }
}

#[test]
fn one_gpu_placements_ignore_tp2_costs_byte_for_byte() {
    for onboard in [Onboard::Auto, Onboard::Layers(10), Onboard::Fraction(0.25),
        Onboard::ExpertsFirst { pool_floor: 262_144 }] {
        let req = request(1, 44 * GIB, 60, 4, onboard);
        let expected = solve(&req).unwrap();
        let mut half = req;
        half.tp2_workspace = [u64::MAX; 2];
        for layer in &mut half.layers {
            layer.experts.as_mut().unwrap().tp2 = true;
            layer.experts.as_mut().unwrap().half = [Bytes2 { resident: u64::MAX, staging: u64::MAX }; 2];
        }
        assert_eq!(solve(&half).unwrap(), expected);
    }
}

#[test]
fn tp2_staging_peak_is_reserved_beside_every_half() {
    let mut req = tp2_request([44 * GIB; 2], Onboard::Layers(2));
    req.layers[0].experts.as_mut().unwrap().half[0].staging = 3 * GIB;
    let t = solve(&req).unwrap().tp2.unwrap();
    assert_eq!(t.peak_bytes[0], GIB / 2 + GIB + 3 * GIB);
}

#[test]
fn experts_first_falls_back_to_a_spark_pool_below_the_context() {
    // A card that cannot hold the 1M context's pool even without RTX layers: v2 served it with no
    // RTX layers and the largest pool above 262K (the context clamped to it).
    let mut small = request(1, 8 * GIB, 60, 2, Onboard::ExpertsFirst { pool_floor: EXPERTS_FIRST_POOL_FLOOR });
    small.pool = PoolPolicy::resolve(&[32 * GIB], 1 << 20, None, 256, false);
    let placement = solve(&small).unwrap();
    assert_eq!(placement.onboard_layers, 0);
    assert!((EXPERTS_FIRST_POOL_FLOOR..1 << 20).contains(&placement.pool_tokens), "{}", placement.pool_tokens);
    // Below v2's floor it is still a refusal.
    small.inventory.gpus[0].baseline = Baseline::Measured { free_bytes: 3 * GIB };
    assert!(solve(&small).is_err());
    // Spark-free layouts never fall back.
    let mut spark_free = request(1, 8 * GIB, 60, 0, Onboard::ExpertsFirst { pool_floor: EXPERTS_FIRST_POOL_FLOOR });
    spark_free.pool = PoolPolicy::resolve(&[32 * GIB], 1 << 20, None, 256, true);
    assert!(solve(&spark_free).is_err());
}


#[test]
fn attention_selectors_are_strict_until_family_executors_land() {
    for executor in families::EXECUTORS {
        assert_eq!(executor.check_attention(None, 2, true).unwrap(), AttentionPlacement::Heads);
        for mode in [AttentionPlacement::Context, AttentionPlacement::Layers] {
            assert!(matches!(executor.check_attention(Some(mode), 2, true),
                Err(PlacementError::AttentionPlacement { family, mode: refused, .. }) if family == executor.family && refused == mode));
        }
    }
    assert_eq!(attention::parse("auto"), Ok(None));
    assert_eq!(attention::parse("context"), Ok(Some(AttentionPlacement::Context)));
    assert!(attention::parse("CONTEXT").is_err());
}

const CONTEXT: ExecutorModes = ExecutorModes { family: "fixture", modes: &[
    LayerMode::HeadSplit, LayerMode::ContextSplit, S0, S1, W0], hops: true };

#[test]
fn context_residuals_and_buffers_match_heads_and_section_6() {
    for from in [ResidualHome::Replicated, ResidualHome::Owned(0), ResidualHome::Owned(1)] {
        assert_eq!(from.transition(LayerMode::ContextSplit), from.transition(LayerMode::HeadSplit));
    }
    for (token, q, p, c, stage) in [(788, 36_864, 32_896, 16_384, 1.54),
        (179, 32_768, 32_896, 4_096, 0.35), (179, 65_536, 65_792, 8_192, 0.35),
        (561, 32_768, 32_896, 4_096, 1.10)] {
        let buffers = ContextBuffers { staging_unit_bytes: token, staging_unit_rows: 1, query_row_bytes: q, partial_row_bytes: p,
            candidate_row_bytes: c, compiled_extent: 1 << 20, decode_rows: 64, lanes: 2 };
        let demands = buffers.demands().unwrap();
        assert!((demands[0].bytes as f64 / GIB as f64 - stage).abs() < 0.005);
        assert_eq!(demands[1].bytes, 2 * 2 * 64 * (q + p + c));
        assert_eq!(demands[0].bytes, demands[2].bytes);
    }
    assert!(ContextBuffers { staging_unit_bytes: u64::MAX, compiled_extent: 2, ..Default::default() }.demands().is_err());
}

#[test]
fn context_halves_units_but_keeps_other_layers_and_fixed_bytes_replicated() {
    let mut req = request(2, 44 * GIB, 2, 4, Onboard::Auto);
    req.pool_overhead = vec![0; 2];
    req.executor = CONTEXT;
    req.attention_placement = Some(AttentionPlacement::Context);
    req.layers[0].kv_unit = KvDemand { unit_bytes_whole: UNIT, unit_bytes_split: [UNIT; 2], unit_bytes_context: Some([UNIT / 2; 2]) };
    req.layers[0].modes.push(LayerMode::ContextSplit);
    req.layers[0].fixed_bytes = ModeBytes::replicated(GIB);
    req.layers[1].kv_unit = ModeBytes::replicated(UNIT / 8).into();
    let plan = solve(&req).unwrap();
    assert_eq!(plan.layers[0].mode, LayerMode::ContextSplit);
    assert_eq!(plan.layers[1].mode, LayerMode::HeadSplit);
    for items in plan.items {
        assert_eq!(items.iter().find(|i| i.group == "records").unwrap().bytes, (UNIT / 2 + UNIT / 8) * 8192);
        assert_eq!(items.iter().find(|i| i.group == "layer state and marks").unwrap().bytes, GIB);
    }
}

#[test]
fn only_auto_uses_the_memory_lever_and_colocate_groups_keep_their_owner() {
    let mut req = request(2, 20 * GIB, 4, 4, Onboard::Auto);
    req.pool_overhead = vec![0; 2];
    req.executor = CONTEXT;
    for (index, layer) in req.layers.iter_mut().enumerate() {
        layer.kv_unit = KvDemand { unit_bytes_whole: UNIT / 4, unit_bytes_split: [UNIT / 4; 2],
            unit_bytes_context: Some([UNIT / 8; 2]) };
        layer.modes = vec![LayerMode::HeadSplit, LayerMode::ContextSplit, S0, S1];
        layer.colocate = Some((index / 2) as u16);
    }
    req.attention_placement = Some(AttentionPlacement::Heads);
    let heads = solve(&req).unwrap();
    assert!(heads.pool_tokens < req.pool.target);
    req.attention_placement = None;
    let context = solve(&req).unwrap();
    assert_eq!(context.pool_tokens, req.pool.target);
    assert_eq!(context.attention_placement, AttentionPlacement::Context);
    req.attention_placement = Some(AttentionPlacement::Layers);
    let layers = solve(&req).unwrap();
    assert_eq!(layers.layers.iter().map(|l| l.mode).collect::<Vec<_>>(), [S0, S0, S1, S1]);
    req.attention_placement = Some(AttentionPlacement::Context);
    req.inventory.peer_access = false;
    assert!(solve(&req).is_err());
}

#[test]
fn section_6_glm_memory_uses_one_latent_and_charges_every_exchange_slot() {
    // P2's fixed demands, taken from the design's plan JSON after records.
    // Full indexers on 0/1/2 and each fourth layer thereafter: 21 in 78.
    let fixed = [30_180_346_250u64, 18_353_380_746];
    let buffers = ContextBuffers { staging_unit_bytes: 788, staging_unit_rows: 1, query_row_bytes: 36_864,
        partial_row_bytes: 32_896, candidate_row_bytes: 16_384,
        compiled_extent: 1 << 20, decode_rows: 64, lanes: 2 };
    for nvfp4 in [false, true] {
        let mut req = request(2, inventory::PRO_TOTAL_BYTES - 2 * GIB, 78, 6, Onboard::Auto);
        req.pool_overhead = vec![0; 2];
        req.context_buffers = buffers;
        req.layers_first_gpu = 1;
        req.executor = CONTEXT;
        for (layer, demand) in req.layers.iter_mut().enumerate() {
            let indexer = layer <= 2 || (layer >= 6 && (layer - 6) % 4 == 0);
            let token = 656 + if indexer { 132 } else { 0 };
            demand.kind = AttentionClass::Mla;
            demand.experts = None;
            demand.context_indexer = indexer;
            demand.kv_unit = KvDemand { unit_bytes_whole: token * 64, unit_bytes_split: [token * 64; 2],
                unit_bytes_context: Some([token * 32; 2]) };
            demand.colocate = Some(if layer < 2 { layer as u16 } else { (2 + (layer - 2) / 4) as u16 });
            demand.modes = vec![LayerMode::HeadSplit, LayerMode::ContextSplit, S1, S0];
        }
        req.pool.unit_rows = 64;
        req.fixed = fixed.into_iter().enumerate().map(|(gpu, bytes)| Demand::new(gpu as u8, Category::Workspace,
            "P2 fixed", bytes + if nvfp4 { 8_685_388_318 } else { 0 }, Basis::Estimated)).collect();
        req.hops = HopSpec { row_bytes: 12_288, rows: 4096, lanes: 4, entry_gpu: 0, head_gpu: 0 };
        req.attention_placement = Some(AttentionPlacement::Heads);
        let heads = solve(&req).unwrap();
        print_attention_fixture(if nvfp4 { "GLM 5.3 NVFP4" } else { "GLM 5.3 EXL3" }, AttentionPlacement::Heads, &heads);
        req.attention_placement = Some(AttentionPlacement::Context);
        let context = solve(&req).unwrap();
        print_attention_fixture(if nvfp4 { "GLM 5.3 NVFP4" } else { "GLM 5.3 EXL3" }, AttentionPlacement::Context, &context);
        let mut max_req = req.clone();
        max_req.onboard = Onboard::Layers(0);
        let max_pool = solve(&max_req).unwrap().pool_tokens;
        assert!((max_pool as f64 / 1e6 - if nvfp4 { 2.20 } else { 2.52 }).abs() < 0.005);
        eprintln!("K0 GLM context max_pool={max_pool}");
        assert_eq!(context.pool_tokens, 2 << 20);
        assert_eq!(context.peer_row_bytes, 78 * (36_864 + 32_896) + 21 * 16_384);
        let used = context.items.iter().map(|items| items.iter().map(|i| i.bytes).sum::<u64>() as f64 / GIB as f64).collect::<Vec<_>>();
        let expected = if nvfp4 { [90.4, 79.4] } else { [82.3, 71.3] };
        for gpu in 0..2 { assert!((used[gpu] - expected[gpu]).abs() < 0.07, "NVFP4={nvfp4} GPU{gpu}: {}", used[gpu]); }
        assert!((heads.pool_tokens as f64 / 1e6 - if nvfp4 { 1.13 } else { 1.29 }).abs() < 0.015);
        req.attention_placement = Some(AttentionPlacement::Layers);
        // The design drops 1.386 GiB replicated operands, half from each GPU.
        for d in &mut req.fixed { d.bytes -= (1_287_508_185 + 200_991_168) / 2; }
        let layers = solve(&req).unwrap();
        print_attention_fixture(if nvfp4 { "GLM 5.3 NVFP4" } else { "GLM 5.3 EXL3" }, AttentionPlacement::Layers, &layers);
        let mut max_req = req.clone();
        max_req.onboard = Onboard::Layers(0);
        let max_pool = solve(&max_req).unwrap().pool_tokens;
        assert!((max_pool as f64 / 1e6 - if nvfp4 { 2.22 } else { 2.54 }).abs() < 0.005);
        eprintln!("K0 GLM layers max_pool={max_pool}");
        let records = layers.items.iter().map(|items| items.iter().find(|i| i.group == "records").unwrap().bytes / (2 << 20)).collect::<Vec<_>>();
        let k = layers.layers.iter().position(|l| l.mode == S0).unwrap();
        eprintln!("K0 GLM layers switch={k} hops={}", layers.hops.len());
        assert!(layers.layers[..k].iter().all(|l| l.mode == S1));
        assert!(layers.layers[k..].iter().all(|l| l.mode == S0));
        eprintln!("K0 GLM layers records={records:?}");
        let used = layers.items.iter().map(|items| items.iter().map(|i| i.bytes).sum::<u64>() as f64 / GIB as f64).collect::<Vec<_>>();
        let expected = if nvfp4 { [89.7, 76.4] } else { [81.6, 68.3] };
        for gpu in 0..2 { assert!((used[gpu] - expected[gpu]).abs() < 0.07, "NVFP4={nvfp4} layers GPU{gpu}: {}", used[gpu]); }
    }
}


#[test]
fn section_6_v4_and_glm_flash_memory_savings() {
    let records = (2u64 << 20) * 11 * (528 + 512 + 33);
    let compact = (2u64 << 20) * 11 * (528 + 33);
    assert!((records as f64 / GIB as f64 - 23.05).abs() < 0.01);
    let staging = ContextBuffers { staging_unit_bytes: 256 * 528 + 64 * 132,
        staging_unit_rows: 256, compiled_extent: 1 << 20, ..Default::default() }.demands().unwrap()[0].bytes;
    let exchange = 4 * 64 * (32_768 + 32_896 + 4_096);
    let freed = (records - compact / 2 - staging - exchange) as f64 / GIB as f64;
    assert!((freed - 15.9).abs() < 0.05);
    let hop = 2 * 2 * 4096 * 32_784;
    let layers_freed = [(records - compact * 5 / 11 - hop) as f64 / GIB as f64,
        (records - compact * 6 / 11 - hop) as f64 / GIB as f64];
    assert!((layers_freed[0] - 17.1).abs() < 0.05 && (layers_freed[1] - 16.0).abs() < 0.05);
    assert!((98.6 - freed - 82.7).abs() < 0.07 && (99.4 - freed - 83.5).abs() < 0.07);
}


fn print_attention_fixture(family: &str, mode: AttentionPlacement, placement: &Placement) {
    let fixed = placement.items.iter().map(|items| items.iter()
        .filter(|i| i.group != "records" && i.category != Category::Experts)
        .map(|i| i.bytes).sum::<u64>()).collect::<Vec<_>>();
    eprintln!("K0 {family} {mode}: pool={} onboard={} fixed={fixed:?}", placement.pool_tokens, placement.onboard_layers);
}

#[test]
fn section_6_tp2_counts_use_exact_pages_and_real_layer_ownership() {
    // Calibrated section-six admission model, not a runtime executor: the P4
    // table supplies per-half bytes and slack; geometry supplies exact pages.
    for (name, count, c4, dim, state, half, pairs, slack, expected) in [
        ("V4 Flash", 43usize, 21usize, 4096u64, 2.048f64, 1.594f64, 36usize, 0.074110f64, [36, 38, 38]),
        ("V4 Pro", 61, 30, 7168, 2.487, 2.9921875, 10, 0.734058, [10, 11, 12]),
    ] {
        let available = inventory::PRO_TOTAL_BYTES - 2 * GIB;
        let mut req = request(2, available, count, 4, Onboard::Auto);
        req.executor = CONTEXT;
        req.pool_overhead = vec![0; 2];
        req.context_buffers = ContextBuffers { staging_unit_bytes: 45_888, staging_unit_rows: 256,
            query_row_bytes: if dim == 4096 { 32_768 } else { 65_536 },
            partial_row_bytes: if dim == 4096 { 32_896 } else { 65_792 },
            candidate_row_bytes: if dim == 4096 { 4_096 } else { 8_192 },
            compiled_extent: 1 << 20, decode_rows: 64, lanes: 2 };
        req.hops = HopSpec { row_bytes: 4 * dim * 2 + 16, rows: 4096, lanes: 2, entry_gpu: 0, head_gpu: 0 };
        let half = (half * GIB as f64) as u64;
        let state_unit = (state * GIB as f64 / count as f64) as u64;
        for (i, layer) in req.layers.iter_mut().enumerate() {
            // Official configs: C4 layers 2,4,...; Pro begins with two C128.
            let ratio = if i >= 2 && i % 2 == 0 { 4 } else if name == "V4 Pro" || i >= 2 { 128 } else { 0 };
            let unit = crate::serving_capacity::deepseek_v4_layer_unit_bytes(ratio);
            layer.kv_unit = KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit; 2],
                unit_bytes_context: (ratio == 4).then_some([unit / 2; 2]) };
            layer.context_indexer = ratio == 4;
            layer.fixed_bytes = ModeBytes::replicated(state_unit);
            layer.modes = vec![LayerMode::HeadSplit, LayerMode::ContextSplit, S0, S1];
            layer.experts = (i >= 2).then_some(ExpertCost { whole: Bytes2::default(),
                half: [Bytes2 { resident: half, staging: 0 }; 2], tp2: true, spark_ok: true });
        }
        assert_eq!(req.layers.iter().filter(|l| l.context_indexer).count(), c4);
        let records = req.layers.iter().map(|l| l.kv_unit.unit_bytes_whole).sum::<u64>() * 8192;
        let base = available - records - state_unit * count as u64 - pairs as u64 * half - (slack * GIB as f64) as u64;
        req.fixed = (0..2).map(|gpu| Demand::new(gpu, Category::Workspace, "P4 adjusted fixed", base, Basis::Estimated)).collect();
        for (mode, layers) in [AttentionPlacement::Heads, AttentionPlacement::Context, AttentionPlacement::Layers].into_iter().zip(expected) {
            req.attention_placement = Some(mode);
            let p = solve(&req).unwrap();
            print_attention_fixture(name, mode, &p);
            assert_eq!(p.pool_tokens, 2 << 20);
            assert_eq!(p.onboard_layers, layers, "{name} {mode}");
            let kv = p.items.iter().map(|items| items.iter().filter(|i| i.group == "records").map(|i| i.bytes).sum::<u64>()).collect::<Vec<_>>();
            eprintln!("K0 {name} {mode}: kv={kv:?} hops={}", p.hops.len());
            if mode == AttentionPlacement::Layers {
                let k = p.layers.iter().position(|l| l.mode == S1).unwrap_or(p.layers.len());
                eprintln!("K0 {name} layers switch={k}");
                assert!(p.layers[..k].iter().all(|l| l.mode == S0));
                assert!(p.layers[k..].iter().all(|l| l.mode == S1));
            }
        }
    }
    // GLM Flash P6 hypothetical TP2 admission: current production has no
    // such executor. Compact pages on 11 MLA owners; KDA remains split.
    let mut req = request(2, inventory::PRO_TOTAL_BYTES - 2 * GIB, 45, 4, Onboard::Auto);
    req.executor = CONTEXT;
    req.layers_first_gpu = 1;
    req.pool_overhead = vec![0; 2];
    req.context_buffers = ContextBuffers { staging_unit_bytes: 256 * 528 + 64 * 132, staging_unit_rows: 256,
        query_row_bytes: 32_768, partial_row_bytes: 32_896, candidate_row_bytes: 4_096,
        compiled_extent: 1 << 20, decode_rows: 64, lanes: 2 };
    req.hops = HopSpec { row_bytes: 32_784, rows: 4096, lanes: 2, entry_gpu: 0, head_gpu: 0 };
    for (i, layer) in req.layers.iter_mut().enumerate() {
        let mla = i % 4 == 3;
        layer.kind = if mla { AttentionClass::Mla } else { AttentionClass::Kda };
        layer.kv_unit = if mla { KvDemand { unit_bytes_whole: 256 * 561, unit_bytes_split: [256 * 1073; 2],
            unit_bytes_context: Some([256 * 561 / 2; 2]) } } else { KvDemand::default() };
        layer.context_indexer = mla;
        layer.modes = vec![LayerMode::HeadSplit, LayerMode::ContextSplit, S0, S1];
        layer.experts = (i >= 3).then_some(ExpertCost { whole: Bytes2::default(),
            half: [Bytes2 { resident: (115.59 * GIB as f64 / 42.0 / 2.0) as u64, staging: 0 }; 2], tp2: true, spark_ok: true });
    }
    req.fixed = [20_100_569_073, 15_815_582_073].into_iter().enumerate().map(|(gpu, bytes)|
        Demand::new(gpu as u8, Category::Workspace, "P2 fixed", bytes, Basis::Estimated)).collect();
    let spark_req = req.clone();
    for (mode, layers) in [(AttentionPlacement::Heads, 37), (AttentionPlacement::Context, 42), (AttentionPlacement::Layers, 42)] {
        req.attention_placement = Some(mode);
        let p = solve(&req).unwrap();
        print_attention_fixture("GLM Flash", mode, &p);
        let kv = p.items.iter().map(|items| items.iter().filter(|i| i.group == "records").map(|i| i.bytes).sum::<u64>()).collect::<Vec<_>>();
        eprintln!("K0 GLM Flash {mode}: kv={kv:?} hops={}", p.hops.len());
        assert_eq!(p.pool_tokens, 2 << 20);
        assert_eq!(p.onboard_layers, layers);
        if mode == AttentionPlacement::Layers {
            let k = p.layers.iter().position(|l| l.mode == S0).unwrap();
            eprintln!("K0 GLM Flash layers switch={k}");
        }
    }
    let mut free_req = spark_req;
    free_req.inventory.spark_ranks = 0;
    for mode in [AttentionPlacement::Context, AttentionPlacement::Layers] {
        free_req.attention_placement = Some(mode);
        let p = solve(&free_req).unwrap();
        assert_eq!(p.onboard_layers, 42);
        assert!(p.pool_tokens >= 2 << 20);
        print_attention_fixture("GLM Flash Spark-free", mode, &p);
    }
}


#[test]
fn context_pool_rounding_never_crosses_floor_and_explicit_units_are_even() {
    let mut req = request(2, GIB, 1, 4, Onboard::Layers(0));
    req.executor = CONTEXT;
    req.attention_placement = Some(AttentionPlacement::Context);
    req.pool_overhead = vec![0; 2];
    req.pool.unit_rows = 1;
    req.pool.floor = 3;
    req.pool.target = 3;
    req.pool.ceiling = 3;
    req.layers[0].kv_unit = KvDemand { unit_bytes_whole: 2, unit_bytes_split: [2; 2], unit_bytes_context: Some([1; 2]) };
    req.layers[0].modes.push(LayerMode::ContextSplit);
    req.layers[0].experts = None;
    assert!(matches!(solve(&req), Err(PlacementError::BelowFloor { pool: 2, .. })));
    req.pool.floor = 1;
    req.pool.requested = Some(3);
    assert!(matches!(solve(&req), Err(PlacementError::Inventory(_))));
}


#[test]
fn auto_cannot_flip_to_an_unqualified_executor_mode() {
    for executor in families::EXECUTORS {
        let mut req = request(2, 20 * GIB, 2, 4, Onboard::Auto);
        req.executor = executor;
        for layer in &mut req.layers {
            layer.modes = vec![LayerMode::HeadSplit, LayerMode::ContextSplit, S0, S1, W0, W1];
            layer.kv_unit = KvDemand { unit_bytes_whole: UNIT, unit_bytes_split: [UNIT; 2], unit_bytes_context: Some([UNIT / 2; 2]) };
        }
        req.pool_overhead = vec![0; 2];
        let auto = solve(&req).unwrap();
        req.attention_placement = Some(AttentionPlacement::Heads);
        assert_eq!(auto, solve(&req).unwrap(), "{}", executor.family);
        assert!(auto.layers.iter().all(|l| executor.runs(l.mode)));
    }
}

#[test]
fn mixed_memory_flips_are_explicit_in_plan_and_summary() {
    let mut req = request(2, 7 * UNIT / 2, 3, 4, Onboard::Auto);
    req.pool_overhead = vec![0; 2];
    req.executor = CONTEXT;
    req.layers[0].kind = AttentionClass::Mla;
    req.layers[0].kv_unit = KvDemand { unit_bytes_whole: UNIT, unit_bytes_split: [UNIT; 2], unit_bytes_context: Some([UNIT / 2; 2]) };
    req.layers[0].modes = vec![LayerMode::HeadSplit, LayerMode::ContextSplit];
    req.layers[1].kind = AttentionClass::Dsa;
    req.layers[1].kv_unit = KvDemand { unit_bytes_whole: UNIT / 4, unit_bytes_split: [UNIT / 4; 2], unit_bytes_context: None };
    req.layers[1].modes = vec![LayerMode::HeadSplit, S0, S1];
    req.layers[2] = req.layers[1].clone();
    req.pool.target = 1024;
    req.pool.floor = 1;
    for layer in &mut req.layers { layer.experts = None; }
    let mixed = solve(&req).unwrap();
    assert_eq!(mixed.pool_tokens, 1024);
    assert_eq!(mixed.attention_by_kind, [(AttentionClass::Mla, AttentionPlacement::Context), (AttentionClass::Dsa, AttentionPlacement::Layers)]);
    assert!(mixed.summary().contains("mixed (Mla=context, Dsa=layers)"));
}


#[test]
fn contiguous_layer_switch_keeps_groups_and_breaks_equal_bytes_by_pool() {
    let mut req = request(2, 32 * GIB, 4, 4, Onboard::Auto);
    req.executor = CONTEXT;
    req.attention_placement = Some(AttentionPlacement::Layers);
    req.pool_overhead = vec![0; 2];
    for layer in &mut req.layers {
        layer.experts = None;
        layer.modes = vec![LayerMode::HeadSplit, S0, S1];
        layer.kv_unit = ModeBytes::replicated(UNIT / 4).into();
    }
    req.layers[1].colocate = Some(7);
    req.layers[2].colocate = Some(7);
    // k=1 and k=3 have equal largest owned bytes; the asymmetric fixed
    // demand makes GPU1's larger suffix worse, so choose k=3.
    req.fixed = vec![Demand::new(1, Category::Workspace, "rank1 fixed", 8 * GIB, Basis::Exact)];
    let p = solve(&req).unwrap();
    assert_eq!(p.layers.iter().map(|l| l.mode).collect::<Vec<_>>(), [S0, S0, S0, S1]);
    req.fixed.clear();
    let p = solve(&req).unwrap();
    assert_eq!(p.layers.iter().map(|l| l.mode).collect::<Vec<_>>(), [S0, S1, S1, S1]);
}
