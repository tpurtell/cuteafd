use super::*;

const GIB: u64 = 1 << 30;
const UNIT: u64 = 4 << 20;

/// Two equal GPUs measured with `free` bytes each, `layers` MoE layers of
/// 1 GiB (+256 MiB staging), 4 MiB per pool unit of 256 tokens over all
/// layers (2M tokens: 8192 units, 32 GiB).
fn request(gpus: usize, free: u64, layers: usize, sparks: usize, onboard: Onboard) -> PlacementRequest {
    PlacementRequest {
        inventory: Inventory {
            gpus: vec![GpuBudget { capacity_bytes: 96 * GIB, headroom_bytes: 0,
                baseline: Baseline::Measured { free_bytes: free } }; gpus],
            spark_ranks: sparks, peer_access: gpus == 2 },
        pool: PoolPolicy::resolve(&vec![96 * GIB; gpus], 131_072, None, 256, sparks == 0),
        layers: (0..layers).map(|_| LayerDemand { kind: AttentionClass::Csa, weights: ModeBytes::default(),
            kv_unit: ModeBytes::default(),
            experts: Some(ExpertCost { whole: Bytes2 { resident: GIB, staging: GIB / 4 }, half: [Bytes2::default(); 2], tp2: false,
                spark_ok: true }),
            modes: vec![LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] }).collect(),
        pool_overhead: vec![UNIT; gpus],
        fixed: Vec::new(),
        movables: Vec::new(),
        expert_workspace: GIB / 4,
        tp2_workspace: [0; 2],
        onboard,
        expert_gpus: gpus,
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit], by_kind: Vec::new() },
    }
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
