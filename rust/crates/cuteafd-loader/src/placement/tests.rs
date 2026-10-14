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
            fixed_bytes: ModeBytes::default(), extra: Vec::new(), context_indexer: false,
            experts: Some(ExpertCost { whole: Bytes2 { resident: GIB, staging: GIB / 4 }, half: [Bytes2::default(); 2], tp2: false, spark_ok: true }),
            modes: vec![LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] }).collect(),
        pool_overhead: vec![UNIT; gpus],
        fixed: Vec::new(),
        movables: Vec::new(),
        expert_workspace: GIB / 4,
        tp2_workspace: [0; 2],
        onboard,
        expert_gpus: gpus,
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit], by_kind: Vec::new(), ..LayerPolicy::default() },
        hops: HopSpec { row_bytes: 0, rows: 0, lanes: 0, entry_gpu: 0, head_gpu: 0,
            terminal: if gpus == 2 { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } },
        executor: V4_HEADS,
    }
}

/// V4 as `deepseek_v4::request` builds it without the C4 context programs: heads or GPU0 only.
const V4_HEADS: ExecutorModes = ExecutorModes { modes: &[LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }],
    ..families::DEEPSEEK_V4 };

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
    req.movables.push(Movable { id: MovableId::DsparkExperts, allowed: vec![0], expert_arena: true, conditional: vec![],
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
const MHC: HopSpec = HopSpec { row_bytes: 4 * 4096 * 2, rows: 4096, lanes: 2, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner };
const ANY: ExecutorModes = ExecutorModes { family: "test", modes: &[LayerMode::HeadSplit, W0, W1, S0, S1], hops: true, attention: AttentionPlacement::Heads };

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
    let hops = plan_hops(&modes, &MHC).unwrap();
    assert_eq!(hops.iter().map(|h| (h.at, h.from, h.to, h.kind)).collect::<Vec<_>>(), [
        (HopPoint::BeforeLayer(20), 0, 1, HopKind::Boundary),
        (HopPoint::Exit, 1, 0, HopKind::Boundary),
    ]);
    // One receive pair per lane on each GPU: 2 lanes x 1 hop x 128 MiB.
    assert_eq!(hop_buffer_bytes(&hops, &MHC, 2), Some(vec![256 << 20, 256 << 20]));
    // Head on GPU1 with the last range: no exit hop.
    let tail = plan_hops(&modes, &HopSpec { head_gpu: 1, ..MHC }).unwrap();
    assert_eq!(tail.len(), 1);
    assert_eq!(hop_buffer_bytes(&tail, &MHC, 2), Some(vec![0, 256 << 20]));
}

#[test]
fn head_split_stacks_only_hop_at_entry_and_charge_nothing() {
    // Today's V4/GLM/MiMo head split: the embedding broadcast lands in the step input.
    let hops = plan_hops(&[LayerMode::HeadSplit; 61], &HopSpec { terminal: TerminalHome::Replicated, ..MHC }).unwrap();
    assert_eq!(hops.len(), 1);
    assert_eq!((hops[0].at, hops[0].kind, hops[0].charged()), (HopPoint::Entry, HopKind::Broadcast, false));
    assert_eq!(hop_buffer_bytes(&hops, &MHC, 2), Some(vec![0, 0]));
}

#[test]
fn mixed_split_and_whole_layers_broadcast_into_split_ffns() {
    // GLM Flash's mixed policy: KDA head split, DSA layers whole on alternating GPUs with a split FFN.
    let modes = [LayerMode::HeadSplit, S0, LayerMode::HeadSplit, S1, LayerMode::HeadSplit];
    let hops = plan_hops(&modes, &HopSpec { terminal: TerminalHome::Replicated, ..MHC }).unwrap();
    assert_eq!(hops.iter().map(|h| (h.at, h.from, h.to)).collect::<Vec<_>>(), [
        (HopPoint::Entry, 0, 1), (HopPoint::AfterAttention(1), 0, 1), (HopPoint::AfterAttention(3), 1, 0),
    ]);
    // Each GPU receives one charged hop per lane.
    assert_eq!(hop_buffer_bytes(&hops, &MHC, 2), Some(vec![256 << 20, 256 << 20]));
    // Many hops into one GPU alternate between two slots per lane.
    let many = plan_hops(&[S0, S0, S0, S0], &HopSpec { terminal: TerminalHome::Replicated, ..MHC }).unwrap();
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
    split.hops = HopSpec { row_bytes: 0, rows: 0, lanes: 0, ..req.hops };
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
    req.hops.terminal = TerminalHome::AttentionOwner;
    req.executor = ExecutorModes { family: "nohops", modes: &[W0, W1], hops: false, attention: AttentionPlacement::Heads };
    req.policy.default = vec![W0];
    req.layers[3].modes = vec![W1];
    assert!(matches!(solve(&req), Err(PlacementError::UnsupportedHop { family: "nohops",
        hop: Hop { at: HopPoint::BeforeLayer(3), from: 0, to: 1, .. } })));
    // Without peer access, GPU1-owned and split layers are not executable at all.
    let mut req = request(2, 44 * GIB, 4, 4, Onboard::Auto);
    req.inventory.peer_access = false;
    req.hops.terminal = TerminalHome::AttentionOwner;
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
    req.movables.push(Movable { id: MovableId::DsparkExperts, allowed: vec![0], expert_arena: true, conditional: vec![],
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
fn attention_selectors_follow_declared_executor_capabilities() {
    for executor in families::EXECUTORS {
        // Auto is the recorded default on two peer GPUs (GLM 5.3: context, K3; GLM Flash: context,
        // #57), heads on one.
        let recorded = if executor.family == "qwen4" { AttentionPlacement::Layers }
            else if ["glm5", "glm5_flash"].contains(&executor.family) { AttentionPlacement::Context }
            else { AttentionPlacement::Heads };
        assert_eq!(executor.check_attention(None, 2, true).unwrap(), recorded);
        assert_eq!(executor.check_attention(None, 1, false).unwrap(), AttentionPlacement::Heads);
        for mode in [AttentionPlacement::Context, AttentionPlacement::Layers] {
            let supported = match mode {
                AttentionPlacement::Context => executor.runs(LayerMode::ContextSplit),
                AttentionPlacement::Layers => (0..2).all(|gpu| executor.whole_mode(gpu).is_some()),
                AttentionPlacement::Heads => unreachable!(),
            };
            let result = executor.check_attention(Some(mode), 2, true);
            if supported { assert_eq!(result.unwrap(), mode); }
            else { assert!(matches!(result,
                Err(PlacementError::AttentionPlacement { family, mode: refused, .. }) if family == executor.family && refused == mode)); }
        }
    }
    assert_eq!(attention::parse("auto"), Ok(None));
    assert_eq!(attention::parse("context"), Ok(Some(AttentionPlacement::Context)));
    assert!(attention::parse("CONTEXT").is_err());
}

const CONTEXT: ExecutorModes = ExecutorModes { family: "fixture", modes: &[
    LayerMode::HeadSplit, LayerMode::ContextSplit, S0, S1, W0], hops: true, attention: AttentionPlacement::Heads };

#[test]
fn context_residuals_and_buffers_match_heads_and_section_6() {
    for from in [ResidualHome::Replicated, ResidualHome::Owned(0), ResidualHome::Owned(1)] {
        assert_eq!(from.transition(LayerMode::ContextSplit), from.transition(LayerMode::HeadSplit));
    }
    for (token, q, p, c, stage) in [(788, 36_864, 32_896, 16_384, 1.54),
        (179, 32_768, 32_896, 4_096, 0.35), (179, 65_536, 65_792, 8_192, 0.35),
        (561, 32_768, 32_896, 4_096, 1.10)] {
        let buffers = ContextBuffers { staging_unit_bytes: token, staging_unit_rows: 1, query_row_bytes: q, partial_row_bytes: p,
            candidate_row_bytes: c, compiled_extent: 1 << 20, decode_rows: 64, lanes: 2, workspace_bytes: 0 };
        let demands = buffers.demands().unwrap();
        assert!((demands[0].bytes as f64 / GIB as f64 - stage).abs() < 0.005);
        // Two parities x two lanes of [64 rows x (both query halves + partial + both
        // candidate lists)], plus three 16-byte flags per slot: what ContextExchange allocates.
        assert_eq!(demands[1].bytes, 2 * 2 * 64 * (2 * q + p + 2 * c) + 2 * 2 * 3 * 16);
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
fn layers_selector_supports_gqa_gdn_owner_reduction_and_prefers_split() {
    const OWNERS: ExecutorModes = ExecutorModes { family: "owner", modes: &[W0, W1], hops: true, attention: AttentionPlacement::Heads };
    let mut req = request(2, 96 * GIB, 4, 0, Onboard::Auto);
    req.executor = OWNERS;
    req.attention_placement = Some(AttentionPlacement::Layers);
    req.pool_overhead = vec![0; 2];
    req.hops = MHC;
    for (i, layer) in req.layers.iter_mut().enumerate() {
        layer.kv_unit = ModeBytes::replicated(UNIT / 4).into();
        layer.kind = if i % 2 == 0 { AttentionClass::Gqa } else { AttentionClass::Gdn };
        layer.modes = vec![W0, W1];
        layer.experts = None;
    }
    assert_eq!(OWNERS.check_attention(req.attention_placement, 2, true).unwrap(), AttentionPlacement::Layers);
    let plan = solve(&req).unwrap();
    assert_eq!(plan.layers.iter().map(|l| l.mode).collect::<Vec<_>>(), [W0, W0, W1, W1]);
    assert_eq!(plan.hops.len(), 2);
    assert_eq!(plan.hops[0].at, HopPoint::BeforeLayer(2));
    assert_eq!(plan.hops[1].at, HopPoint::Exit);
    assert!(OWNERS.check_attention(req.attention_placement, 1, true).is_err());
    req.executor = ExecutorModes { family: "both", modes: &[W0, W1, S0, S1], hops: true, attention: AttentionPlacement::Heads };
    for layer in &mut req.layers { layer.modes = vec![W0, W1, S0, S1]; }
    assert_eq!(req.executor.whole_mode(0), Some(S0));
    req.hops.terminal = TerminalHome::Replicated;
    assert_eq!(solve(&req).unwrap().layers.iter().map(|l| l.mode).collect::<Vec<_>>(), [S0, S0, S1, S1]);
    // A kind with no owner-capable layer kernels stays split even though
    // other kinds in the same executor support whole ownership.
    req.layers[1].modes = vec![LayerMode::HeadSplit];
    req.executor = CONTEXT;
    assert_eq!(solve(&req).unwrap().layers[1].mode, LayerMode::HeadSplit);
}

#[test]
fn section_6_glm_memory_uses_one_latent_and_charges_every_exchange_slot() {
    // P2's fixed demands, taken from the design's plan JSON after records.
    // Full indexers on 0/1/2 and each fourth layer thereafter: 21 in 78.
    let fixed = [30_180_346_250u64, 18_353_380_746];
    let buffers = ContextBuffers { staging_unit_bytes: 788, staging_unit_rows: 1, query_row_bytes: 36_864,
        partial_row_bytes: 32_896, candidate_row_bytes: 16_384,
        compiled_extent: 1 << 20, decode_rows: 64, lanes: 2, workspace_bytes: 0 };
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
        req.hops = HopSpec { row_bytes: 12_288, rows: 4096, lanes: 4, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::Replicated };
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
    let exchange = 4 * 64 * (2 * 32_768 + 32_896 + 2 * 4_096) + 4 * 3 * 16;
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
            compiled_extent: 1 << 20, decode_rows: 64, lanes: 2, workspace_bytes: 0 };
        req.hops = HopSpec { row_bytes: 4 * dim * 2 + 16, rows: 4096, lanes: 2, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::Replicated };
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
        compiled_extent: 1 << 20, decode_rows: 64, lanes: 2, workspace_bytes: 0 };
    req.hops = HopSpec { row_bytes: 32_784, rows: 4096, lanes: 2, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::Replicated };
    for (i, layer) in req.layers.iter_mut().enumerate() {
        let mla = i % 4 == 3;
        layer.kind = if mla { AttentionClass::Mla } else { AttentionClass::Kda };
        layer.kv_unit = if mla { KvDemand { unit_bytes_whole: 256 * 561, unit_bytes_split: [256 * 1073; 2],
            unit_bytes_context: Some([256 * 561 / 2; 2]) } } else { KvDemand::default() };
        layer.context_indexer = mla;
        layer.modes = if mla { vec![LayerMode::HeadSplit, LayerMode::ContextSplit, S0, S1] }
            else { vec![LayerMode::HeadSplit] };
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
        req.hops.terminal = if executor.runs(LayerMode::HeadSplit) { TerminalHome::Replicated } else { TerminalHome::AttentionOwner };
        for layer in &mut req.layers {
            layer.modes = vec![LayerMode::HeadSplit, LayerMode::ContextSplit, S0, S1, W0, W1];
            layer.kv_unit = KvDemand { unit_bytes_whole: UNIT, unit_bytes_split: [UNIT; 2], unit_bytes_context: Some([UNIT / 2; 2]) };
        }
        req.pool_overhead = vec![0; 2];
        // A family whose context mode is not yet its recorded default pins auto to heads in its
        // request (V4 until #58's gates pass: `deepseek_v4::request`).
        if executor.family == "deepseek_v4" {
            req.policy.attention = vec![(AttentionClass::Csa, AttentionPlacement::Heads)];
        }
        let auto = solve(&req).unwrap();
        req.attention_placement = Some(AttentionPlacement::Heads);
        let heads = solve(&req).unwrap();
        req.policy.attention.clear();
        if executor.attention_default() != AttentionPlacement::Heads {
            // A recorded non-heads default is not a flip: auto runs it outright.
            req.attention_placement = Some(executor.attention_default());
            assert_eq!(auto, solve(&req).unwrap(), "{}", executor.family);
        } else if executor.check_attention(Some(AttentionPlacement::Layers), 2, true).is_err() {
            assert_eq!(auto, heads, "{}", executor.family);
        } else {
            assert!(auto.pool_tokens >= heads.pool_tokens);
        }
        assert!(auto.layers.iter().all(|l| executor.runs(l.mode)));
        assert!(auto.attention_by_kind.iter().all(|(_, mode)|
            executor.check_attention(Some(*mode), 2, true).is_ok()));
    }
}

#[test]
fn mixed_memory_flips_are_explicit_in_plan_and_summary() {
    let mut req = request(2, 15 * UNIT / 4, 3, 4, Onboard::Auto);
    req.pool_overhead = vec![0; 2];
    req.executor = CONTEXT;
    req.context_buffers = ContextBuffers { query_row_bytes: 64, partial_row_bytes: 128,
        decode_rows: 1, lanes: 1, ..ContextBuffers::default() };
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
    assert_eq!(mixed.peer_row_bytes, 192);
    // If the Dsa flip alone reaches an odd-unit target, pruning the Mla
    // context stepping stone recovers the unit and removes its peer traffic.
    req.pool.target = 768;
    let pruned = solve(&req).unwrap();
    assert_eq!(pruned.pool_tokens, 768);
    assert_eq!(pruned.attention_by_kind, [(AttentionClass::Mla, AttentionPlacement::Heads), (AttentionClass::Dsa, AttentionPlacement::Layers)]);
    assert_eq!(pruned.peer_row_bytes, 0);
    // Beyond the available pool, removing context leaves the final four-unit
    // pool unchanged; the equal-pool stepping stone must still be pruned.
    for gpu in &mut req.inventory.gpus {
        gpu.baseline = Baseline::Measured { free_bytes: 5 * UNIT };
    }
    req.pool.target = 1280;
    let pruned = solve(&req).unwrap();
    assert_eq!(pruned.pool_tokens, 1024);
    assert_eq!(pruned.attention_by_kind, [(AttentionClass::Mla, AttentionPlacement::Heads), (AttentionClass::Dsa, AttentionPlacement::Layers)]);
    assert_eq!(pruned.peer_row_bytes, 0);
}


#[test]
fn measured_per_kind_policy_interleaves_owned_mla_and_partitioned_kda() {
    let mut req = request(2, 32 * GIB, 8, 4, Onboard::Auto);
    req.executor = CONTEXT;
    req.pool_overhead = vec![0; 2];
    req.policy.attention = vec![(AttentionClass::Mla, AttentionPlacement::Layers),
        (AttentionClass::Kda, AttentionPlacement::Heads)];
    req.policy.ownership = LayerOwnership::Alternating;
    req.layers_first_gpu = 1;
    for (index, layer) in req.layers.iter_mut().enumerate() {
        layer.experts = None;
        layer.kind = if index % 2 == 0 { AttentionClass::Mla } else { AttentionClass::Kda };
        layer.kv_unit = if index % 2 == 0 { ModeBytes::replicated(UNIT).into() } else { KvDemand::default() };
        layer.modes = if index % 2 == 0 { vec![LayerMode::HeadSplit, S0, S1] } else { vec![LayerMode::HeadSplit] };
    }
    req.layers[0].colocate = Some(7);
    req.layers[4].colocate = Some(7);
    let placed = solve(&req).unwrap();
    assert_eq!(placed.layers.iter().map(|layer| layer.mode).collect::<Vec<_>>(),
        [S1, LayerMode::HeadSplit, S0, LayerMode::HeadSplit, S1, LayerMode::HeadSplit, S1, LayerMode::HeadSplit]);
    assert_eq!(placed.attention_placement, AttentionPlacement::Layers);
    assert_eq!(placed.attention_by_kind, req.policy.attention);
    assert!(placed.hops.iter().filter(|hop| hop.at != HopPoint::Entry)
        .all(|hop| matches!(hop.at, HopPoint::AfterAttention(index) if index % 2 == 0)));
    // A policy is still strict when memory is short, not silently changed to context/heads.
    req.inventory.gpus.iter_mut().for_each(|gpu| gpu.baseline = Baseline::Measured { free_bytes: 2 * UNIT });
    assert!(solve(&req).is_err());
    // Explicit heads bypasses the recorded auto policy for reference measurements.
    req.attention_placement = Some(AttentionPlacement::Heads);
    req.inventory.gpus.iter_mut().for_each(|gpu| gpu.baseline = Baseline::Measured { free_bytes: 32 * GIB });
    assert!(solve(&req).unwrap().layers.iter().all(|layer| layer.mode == LayerMode::HeadSplit));
}

#[test]
fn measured_kind_does_not_flip_even_when_context_would_grow_the_pool() {
    let mut req = request(2, GIB, 1, 4, Onboard::Auto);
    req.executor = CONTEXT;
    req.pool_overhead = vec![0; 2];
    req.pool.target = 2 << 20;
    req.pool.floor = 256;
    req.layers[0].experts = None;
    req.layers[0].kind = AttentionClass::Mla;
    req.layers[0].modes = vec![LayerMode::HeadSplit, LayerMode::ContextSplit];
    req.layers[0].kv_unit = KvDemand { unit_bytes_whole: UNIT, unit_bytes_split: [UNIT; 2],
        unit_bytes_context: Some([UNIT / 2; 2]) };
    req.policy.attention = vec![(AttentionClass::Mla, AttentionPlacement::Heads)];
    let measured = solve(&req).unwrap();
    assert_eq!(measured.pool_tokens, 65_536);
    assert_eq!(measured.layers[0].mode, LayerMode::HeadSplit);
    req.policy.attention.clear();
    let memory_driven = solve(&req).unwrap();
    assert!(memory_driven.pool_tokens > measured.pool_tokens);
    assert_eq!(memory_driven.layers[0].mode, LayerMode::ContextSplit);
}

#[test]
fn per_kind_attention_policy_is_strict_only_on_peer_tp2() {
    let mut req = request(2, 32 * GIB, 1, 4, Onboard::Auto);
    req.policy.attention = vec![(AttentionClass::Mla, AttentionPlacement::Context)];
    req.layers[0].kind = AttentionClass::Mla;
    assert!(matches!(solve(&req), Err(PlacementError::AttentionPlacement { .. })));
    req.inventory.peer_access = false;
    req.hops.terminal = TerminalHome::AttentionOwner;
    assert!(solve(&req).is_ok(), "no-peer layout retains the legacy single-device policy");
    req.inventory.peer_access = true;
    req.policy.attention.push(req.policy.attention[0]);
    assert!(matches!(solve(&req), Err(PlacementError::Inventory(_))));
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
    // Explicit paired declaration keeps these asymmetric bytes in secondary fit.
    req.fixed = [0, 8 * GIB].into_iter().enumerate().map(|(gpu, bytes)|
        Demand::new(gpu as u8, Category::Workspace, "rank fixed", bytes, Basis::Exact)).collect();
    let p = solve(&req).unwrap();
    assert_eq!(p.layers.iter().map(|l| l.mode).collect::<Vec<_>>(), [S0, S0, S0, S1]);
    req.fixed.clear();
    let p = solve(&req).unwrap();
    assert_eq!(p.layers.iter().map(|l| l.mode).collect::<Vec<_>>(), [S0, S1, S1, S1]);
}


fn conditional_request() -> PlacementRequest {
    let mut req = request(2, 44 * GIB, 0, 4, Onboard::Auto);
    req.movables.push(Movable { id: MovableId::Drafter, parts: vec![Bytes2 { resident: 3 * GIB, staging: 0 }],
        allowed: vec![0, 1], expert_arena: false, conditional: vec![MovableDemands { placement_gpu: 1,
            demands: vec![Demand::new(0, Category::Workspace, "bridge", GIB, Basis::Formula),
                Demand::new(1, Category::Transport, "controls", 256, Basis::Exact)] }] });
    req
}

fn group_bytes(p: &Placement, gpu: usize, group: &str) -> u64 {
    p.items[gpu].iter().filter(|i| i.group == group).map(|i| i.bytes).sum()
}

#[test]
fn conditional_movable_charges_only_selected_home_and_all_affected_devices() {
    let mut req = conditional_request();
    req.movables[0].allowed = vec![0]; // Dormant GPU1 inventory is legal.
    let local = solve(&req).unwrap();
    assert_eq!(local.movables, [(MovableId::Drafter, 0)]);
    assert_eq!(group_bytes(&local, 0, "bridge"), 0);
    assert_eq!(group_bytes(&local, 1, "controls"), 0);
    req.movables[0].allowed = vec![0, 1];
    req.inventory.gpus[1].baseline = Baseline::Measured { free_bytes: 60 * GIB };
    assert_eq!(solve(&req).unwrap().movables, [(MovableId::Drafter, 1)]);
    req.movables[0].allowed = vec![1];
    let remote = solve(&req).unwrap();
    assert_eq!(remote.movables, [(MovableId::Drafter, 1)]);
    assert_eq!(group_bytes(&remote, 0, "bridge"), GIB);
    assert_eq!(group_bytes(&remote, 1, "controls"), 256);
    assert_eq!(group_bytes(&remote, 1, "Drafter"), 3 * GIB);
}

#[test]
fn conditional_movable_ties_are_deterministic_and_following_costs_accumulate_once() {
    let mut req = conditional_request();
    req.movables[0].conditional = vec![MovableDemands { placement_gpu: 0, demands: vec![] },
        MovableDemands { placement_gpu: 1, demands: vec![] }];
    let first = solve(&req).unwrap();
    assert_eq!(first.movables, [(MovableId::Drafter, 0)]);
    req.movables[0].allowed.reverse();
    assert_eq!(solve(&req).unwrap(), first);
    req.movables[0].conditional.clear();
    assert_eq!(solve(&req).unwrap(), first);
    let mut another = conditional_request().movables.remove(0);
    another.id = MovableId::Vision;
    another.allowed = vec![1];
    req.movables.push(another);
    let placed = solve(&req).unwrap();
    assert_eq!(placed.movables, [(MovableId::Drafter, 0), (MovableId::Vision, 1)]);
    assert_eq!(group_bytes(&placed, 0, "bridge"), GIB);
    assert_eq!(group_bytes(&placed, 1, "controls"), 256);
}

#[test]
fn conditional_movable_bridge_pressure_falls_back_before_pool_admission() {
    let mut req = conditional_request();
    req.inventory.gpus[1].baseline = Baseline::Measured { free_bytes: 60 * GIB };
    req.movables[0].conditional[0].demands[0].bytes = 43 * GIB;
    let placed = solve(&req).unwrap();
    assert_eq!(placed.movables, [(MovableId::Drafter, 0)]);
    assert_eq!(group_bytes(&placed, 0, "bridge"), 0);
    req.movables[0].allowed = vec![1];
    assert!(matches!(solve(&req), Err(PlacementError::BelowFloor { .. })));
    req.movables[0].allowed = vec![0, 1];
    req.pool.requested = Some(2 << 20);
    assert_eq!(solve(&req).unwrap().movables, [(MovableId::Drafter, 0)]);
    req.movables[0].allowed = vec![1];
    assert!(matches!(solve(&req), Err(PlacementError::PoolDoesNotFit { .. })));
    req.movables[0].conditional[0].demands[0].bytes = 45 * GIB;
    assert!(matches!(solve(&req), Err(PlacementError::Mandatory { gpu: 0, what }) if what.contains("bridge")));
    req.movables[0].allowed = vec![0, 1];
    req.movables[0].parts[0].resident = 61 * GIB;
    assert!(matches!(solve(&req), Err(PlacementError::Mandatory { .. })));
}

#[test]
fn conditional_movable_validates_keys_destinations_and_checked_sums() {
    let req = conditional_request();
    let mut invalid = req.clone();
    let duplicate = invalid.movables[0].conditional[0].clone();
    invalid.movables[0].conditional.push(duplicate);
    assert!(matches!(solve(&invalid), Err(PlacementError::Inventory(_))));
    invalid = req.clone();
    invalid.movables[0].conditional[0].placement_gpu = 2;
    assert!(matches!(solve(&invalid), Err(PlacementError::Inventory(_))));
    invalid = req.clone();
    invalid.movables[0].conditional[0].demands[0].gpu = 2;
    assert!(matches!(solve(&invalid), Err(PlacementError::Inventory(_))));
    invalid = req;
    invalid.movables[0].allowed = vec![1];
    invalid.movables[0].conditional[0].demands[0].bytes = u64::MAX;
    invalid.movables[0].conditional[0].demands.push(Demand::new(0, Category::Workspace, "overflow", 1, Basis::Exact));
    assert!(matches!(solve(&invalid), Err(PlacementError::Overflow(_))));
}

#[test]
fn conditional_movable_preserves_arena_peak_and_strict_pool() {
    let mut req = conditional_request();
    req.movables[0].allowed = vec![1];
    req.movables[0].expert_arena = true;
    req.movables[0].parts = vec![Bytes2 { resident: 3 * GIB, staging: GIB }];
    req.pool.requested = Some(2 << 20);
    let placed = solve(&req).unwrap();
    assert_eq!(placed.pool_tokens, 2 << 20);
    assert_eq!(placed.expert_ranges[1].peak_bytes, 4 * GIB + GIB / 4);
    assert_eq!(group_bytes(&placed, 1, "Drafter"), 0);
    req.pool.requested = Some(4 << 20);
    assert!(matches!(solve(&req), Err(PlacementError::PoolDoesNotFit { .. })));
}

#[test]
fn conditional_movable_selected_device_scratch_and_planned_measured_agree() {
    let mut req = conditional_request();
    // Synthetic per-SM scratch: common residency plus SM170 minimum, then
    // the exact selected-device delta (real families supply their own formula).
    req.movables[0].parts[0].resident += 170 * 1024;
    req.movables[0].conditional.insert(0, MovableDemands { placement_gpu: 0,
        demands: vec![Demand::new(0, Category::Workspace, "SM188 scratch delta", (188 - 170) * 1024, Basis::Formula)] });
    req.movables[0].conditional[1].demands.push(Demand::new(1, Category::Workspace,
        "SM170 scratch delta", 0, Basis::Formula));
    for home in [0, 1] {
        req.movables[0].allowed = vec![home];
        let measured = solve(&req).unwrap();
        let mut planned = req.clone();
        for gpu in &mut planned.inventory.gpus {
            gpu.baseline = Baseline::Planned { context_bytes: 52 * GIB, loaded_bytes: 0 };
        }
        assert_eq!(solve(&planned).unwrap(), measured);
        assert_eq!(group_bytes(&measured, home as usize, if home == 0 { "SM188 scratch delta" }
            else { "SM170 scratch delta" }), u64::from(if home == 0 { 18u32 } else { 0u32 }) * 1024);
        assert_eq!(group_bytes(&measured, home as usize, "Drafter"), 3 * GIB + 170 * 1024);
    }
}

fn owner_balance_request() -> PlacementRequest {
    let mut req = request(2, 96 * GIB, 4, 4, Onboard::Auto);
    req.executor = CONTEXT;
    req.attention_placement = Some(AttentionPlacement::Layers);
    req.pool_overhead = vec![0; 2];
    for layer in &mut req.layers {
        layer.experts = None;
        layer.modes = vec![LayerMode::HeadSplit, S0, S1];
        // 2 GiB per layer at the wanted 2M pool.
        layer.kv_unit = ModeBytes::replicated(UNIT / 16).into();
    }
    req
}

fn owner_cut(req: &PlacementRequest) -> usize {
    let p = solve(req).unwrap();
    p.layers.iter().take_while(|layer| layer.mode == S0).count()
}

#[test]
fn nonempty_layer_ranges_keep_both_owners_under_extreme_entry_skew() {
    let mut req = owner_balance_request();
    req.fixed.push(Demand::new(0, Category::Weights, "entry weights", 16 * GIB, Basis::Exact));
    assert_eq!(owner_cut(&req), 0);
    req.policy.ownership = LayerOwnership::NonemptyRanges;
    assert_eq!(owner_cut(&req), 1);
    req.fixed[0].gpu = 1;
    assert_eq!(owner_cut(&req), 3);
    req.layers.truncate(1);
    assert!(matches!(solve(&req), Err(PlacementError::Inventory("no legal layer ownership boundary"))));
}

#[test]
fn layer_extra_rows_follow_the_cutover_and_reduce_available_pool() {
    let mut req = owner_balance_request();
    req.layers[1].extra = vec![LayerExtraDemand::new(Category::Tables, "side table", GIB,
        LayerResidency::Owner, Basis::Exact)];
    let first = solve(&req).unwrap();
    assert_eq!(owner_cut(&req), 2);
    assert!(first.items[0].iter().any(|i| i.category == Category::Tables && i.group == "side table" && i.bytes == GIB));
    assert!(!first.items[1].iter().any(|i| i.group == "side table"));
    req.fixed.push(Demand::new(0, Category::Weights, "entry weights", 4 * GIB, Basis::Exact));
    let moved = solve(&req).unwrap();
    assert_eq!(owner_cut(&req), 1);
    assert!(moved.items[1].iter().any(|i| i.group == "side table" && i.bytes == GIB));
    assert!(!moved.items[0].iter().any(|i| i.group == "side table"));
    req.layers[1].extra[0].bytes = 8 * GIB;
    req.fixed.clear();
    // A large side table participates in the primary owner balance, not just
    // the final charge: k2 has 12/4 GiB, better than k1's 2/14 or k0's 0/16.
    assert_eq!(owner_cut(&req), 2);
    req.layers[0].extra = std::mem::take(&mut req.layers[1].extra);
    assert_eq!(owner_cut(&req), 1);
    req.layers[1].extra = std::mem::take(&mut req.layers[0].extra);
    req.attention_placement = Some(AttentionPlacement::Heads);
    req.layers[1].extra[0].bytes = GIB;
    assert!(matches!(solve(&req), Err(PlacementError::OwnerDemand { layer: 1, .. })));
    req.layers[1].extra[0].residency = LayerResidency::Replicated;
    let split = solve(&req).unwrap();
    assert!(split.items.iter().all(|rank| rank.iter().any(|i| i.group == "side table" && i.bytes == GIB)));
    req.inventory.gpus.iter_mut().for_each(|gpu| gpu.baseline = Baseline::Measured { free_bytes: 2 * GIB });
    req.pool.target = 2 << 20;
    req.pool.floor = 1;
    let limited = solve(&req).unwrap();
    assert_eq!(limited.pool_tokens, GIB / (UNIT / 4) * 256);
    req.layers[1].extra[0].bytes = u64::MAX;
    req.layers[1].extra.push(LayerExtraDemand::new(Category::Tables, "scale", 256,
        LayerResidency::Replicated, Basis::Formula));
    assert!(matches!(solve(&req), Err(PlacementError::Overflow(_))));
}

#[test]
fn layer_switch_balances_owner_only_head_mtp_and_layer_weights() {
    let mut req = owner_balance_request();
    assert_eq!(owner_cut(&req), 2);
    req.fixed.push(Demand::new(0, Category::Weights, "Qwen head", 3 * GIB, Basis::Exact));
    req.fixed.push(Demand::new(0, Category::Drafter, "Qwen MTP", GIB, Basis::Exact));
    assert_eq!(owner_cut(&req), 1);
    // Same names on both ranks are paired, even when the byte totals differ.
    req.fixed.push(Demand::new(1, Category::Weights, "Qwen head", 0, Basis::Exact));
    req.fixed.push(Demand::new(1, Category::Drafter, "Qwen MTP", 0, Basis::Exact));
    assert_eq!(owner_cut(&req), 2);
    // Category is part of the key: an unrelated workspace cannot pair a head.
    req.fixed.retain(|d| d.gpu == 0 || d.category != Category::Weights);
    req.fixed.push(Demand::new(1, Category::Workspace, "Qwen head", 0, Basis::Exact));
    assert_eq!(owner_cut(&req), 1);
    req.fixed.clear();
    req.layers[0].weights.whole = 8 * GIB;
    // k1 holds 10/6 GiB, better than k0's 0/16 GiB or k2's 12/4 GiB.
    assert_eq!(owner_cut(&req), 1);
}

#[test]
fn layer_switch_balances_mandatory_unequal_tp2_not_optional_residency() {
    let mut req = owner_balance_request();
    for layer in &mut req.layers {
        layer.experts = Some(ExpertCost { whole: Bytes2::default(),
            half: [Bytes2 { resident: 3 * GIB, staging: 0 }, Bytes2 { resident: 2 * GIB, staging: 0 }],
            tp2: true, spark_ok: true });
    }
    // 384/256 resident ratio must not become mandatory under Auto with Sparks.
    assert_eq!(owner_cut(&req), 2);
    req.onboard = Onboard::Layers(4);
    assert_eq!(owner_cut(&req), 1);
    req.onboard = Onboard::Fraction(1.0);
    assert_eq!(owner_cut(&req), 1);
    req.onboard = Onboard::Auto;
    req.inventory.spark_ranks = 0;
    assert_eq!(owner_cut(&req), 1);
    // An unequal mandatory workspace participates in the same primary score.
    req.tp2_workspace = [0, 8 * GIB];
    assert_eq!(owner_cut(&req), 3);
}

#[test]
fn qwen_refuses_shared_solver_empty_owner_chain() {
    let mut req = owner_balance_request();
    req.fixed.push(Demand::new(1, Category::Workspace, "owner1 fixed", 8 * GIB, Basis::Exact));
    assert_eq!(owner_cut(&req), 4);
    let p = solve(&req).unwrap();
    let owners = p.layers.iter().map(|layer| match layer.mode {
        LayerMode::Whole { gpu, .. } => usize::from(gpu), _ => unreachable!(),
    }).collect::<Vec<_>>();
    assert!(families::qwen4::check_dual_layer_owners(&owners).is_err());
    req.fixed[0].gpu = 0;
    assert_eq!(owner_cut(&req), 0);
    assert!(families::qwen4::check_dual_layer_owners(&[1; 4]).is_err());
    assert!(families::qwen4::check_dual_layer_owners(&[]).is_err());
    families::qwen4::check_dual_layer_owners(&[0, 0, 1, 1]).unwrap();
}


#[test]
fn frozen_admission_keeps_loaded_owners_and_still_checks_memory_and_executor() {
    let req = request(2, 44 * GIB, 8, 4, Onboard::Auto);
    let original = solve(&req).unwrap();
    let checked = solve_frozen(&req, &original).unwrap();
    assert_eq!(checked.layers, original.layers);
    assert_eq!(checked.hops, original.hops);
    assert_eq!(checked.pool_tokens, original.pool_tokens);
    let mut invalid = original.clone();
    invalid.layers.pop();
    assert!(matches!(solve_frozen(&req, &invalid), Err(PlacementError::Inventory(_))));
    invalid = original.clone();
    invalid.layers[0].mode = LayerMode::Whole { gpu: 1, ffn: FfnMode::Split };
    assert!(matches!(solve_frozen(&req, &invalid), Err(PlacementError::UnsupportedMode { .. })));
    let mut insufficient = req;
    insufficient.inventory.gpus[0].baseline = Baseline::Measured { free_bytes: 1 };
    insufficient.pool.requested = Some(original.pool_tokens);
    assert!(solve_frozen(&insufficient, &original).is_err());
}


#[test]
fn frozen_owned_admission_does_not_rechoose_after_measured_rank_balance_changes() {
    let mut req = request(2, 32 * GIB, 3, 4, Onboard::Layers(0));
    req.pool.requested = Some(2 << 20);
    req.executor = CONTEXT;
    req.attention_placement = Some(AttentionPlacement::Layers);
    req.pool_overhead = vec![0; 2];
    req.hops = HopSpec { terminal: TerminalHome::Replicated, ..MHC };
    for layer in &mut req.layers {
        layer.experts = None;
        layer.modes = vec![LayerMode::HeadSplit, S0, S1];
        layer.weights = ModeBytes { whole: GIB, split: [GIB / 2; 2] };
        layer.kv_unit = ModeBytes::replicated(UNIT / 4).into();
    }
    let original = solve(&req).unwrap();
    req.inventory.gpus[0].baseline = Baseline::Measured { free_bytes: 48 * GIB };
    req.inventory.gpus[1].baseline = Baseline::Measured { free_bytes: 20 * GIB };
    req.pool.requested = Some(original.pool_tokens);
    req.pool.floor = original.pool_tokens;
    assert_ne!(solve(&req).unwrap().layers, original.layers);
    let frozen = solve_frozen(&req, &original).unwrap();
    assert_eq!(frozen.layers, original.layers);
    assert_eq!(frozen.hops, original.hops);
    assert_eq!(frozen.residual, original.residual);
    assert_eq!(frozen.pool_tokens, original.pool_tokens);
    let mut changed = req.clone();
    changed.attention_placement = Some(AttentionPlacement::Heads);
    assert!(matches!(solve_frozen(&changed, &original), Err(PlacementError::Inventory(_))));
    changed = req.clone();
    changed.attention_placement = None;
    changed.policy.attention = vec![(AttentionClass::Csa, AttentionPlacement::Layers); 2];
    assert!(matches!(solve_frozen(&changed, &original), Err(PlacementError::Inventory(_))));
    let mut invalid = original.clone();
    invalid.attention_by_kind.push(invalid.attention_by_kind[0]);
    assert!(matches!(solve_frozen(&req, &invalid), Err(PlacementError::Inventory(_))));
    invalid = original.clone();
    invalid.layers[0].mode = LayerMode::HeadSplit;
    assert!(matches!(solve_frozen(&req, &invalid), Err(PlacementError::Inventory(_))));
}

#[test]
fn terminal_home_is_explicit_and_validated_for_every_final_mode() {
    use ResidualHome::{Owned, Replicated};
    for mode in [LayerMode::HeadSplit, LayerMode::ContextSplit, S0, S1] {
        let replicated = HopSpec { terminal: TerminalHome::Replicated, ..MHC };
        let head = HopSpec { terminal: TerminalHome::Head, ..MHC };
        assert_eq!(residual_homes(&[mode], &replicated).unwrap().last(), Some(&Replicated));
        assert_eq!(residual_homes(&[mode], &head).unwrap().last(), Some(&Owned(0)));
        assert_eq!(plan_hops(&[mode], &head), plan_hops(&[mode], &replicated),
            "a head-only terminal post needs no extra residual publication");
        assert!(matches!(residual_homes(&[mode], &MHC), Err(LayerPlanError::Terminal { .. })));
    }
    assert_eq!(residual_homes(&[W0, W1], &MHC).unwrap().last(), Some(&Owned(1)));
    assert!(matches!(residual_homes(&[W1], &HopSpec { terminal: TerminalHome::Head, ..MHC }),
        Err(LayerPlanError::Terminal { .. })));
    assert!(matches!(residual_homes(&[W0], &HopSpec { terminal: TerminalHome::Replicated, ..MHC }),
        Err(LayerPlanError::Terminal { .. })));
    let mut req = request(2, 44 * GIB, 4, 4, Onboard::Layers(0));
    req.hops = HopSpec { terminal: TerminalHome::Head, ..MHC };
    let mut p = solve(&req).unwrap();
    assert_eq!(p.residual.last(), Some(&Owned(0)));
    assert!(p.residual[1..p.residual.len() - 1].iter().all(|home| *home == Replicated));
    LayerExecutionPlan::new(&p, req.hops).unwrap();
    let replicated = HopSpec { terminal: TerminalHome::Replicated, ..req.hops };
    assert_eq!(LayerExecutionPlan::new(&p, replicated), Err(LayerPlanError::Residual));
    p.residual.last_mut().map(|home| *home = Replicated);
    assert_eq!(LayerExecutionPlan::new(&p, req.hops), Err(LayerPlanError::Residual));
    req.hops.head_gpu = 2;
    assert!(matches!(solve(&req), Err(PlacementError::Inventory(_))));
    let mut one = request(1, 44 * GIB, 4, 4, Onboard::Layers(0));
    one.hops.terminal = TerminalHome::Replicated;
    assert!(matches!(solve(&one), Err(PlacementError::Inventory(_))));
    one.hops.terminal = TerminalHome::Head;
    let p = solve(&one).unwrap();
    assert!(p.residual.iter().all(|home| *home == Owned(0)));
}
