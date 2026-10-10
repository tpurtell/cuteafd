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
