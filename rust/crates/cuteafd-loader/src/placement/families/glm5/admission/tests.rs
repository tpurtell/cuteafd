use super::*;
use crate::placement::{FfnMode, planner};

/// The searched extent: an independent cudaMalloc request rounds to its own
/// driver block; sub-block requests stay exact and share blocks.
fn searched(exact: u64) -> u64 {
    if planner::independent_request(exact) {
        exact + planner::allocation_granularity_forecast(&[exact]).unwrap()
    } else { exact }
}
use cuteafd_core::memory_layout::Category;

fn cfg() -> GlmDsaConfig {
    GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap()
}

fn production_link(cfg: &GlmDsaConfig, ranks: usize, shape: PrefillShape, prefill: bool)
    -> crate::placement::PreparedSparkLink {
    use crate::placement::{PreparedSparkLink, SparkEndpointOwners, SparkIntakeCandidates};
    let geometry = cuteafd_core::ExpertGeometry { hidden: cfg.hidden as u32,
        experts: cfg.experts as u32, topk: cfg.topk as u32,
        intermediate: cfg.moe_intermediate as u32, layers: cfg.layers as u32 };
    PreparedSparkLink::endpoint(if prefill { "test.prefill" } else { "test.decode" }, 0,
        SparkEndpointOwners { ranks, geometry,
            endpoint: cuteafd_core::SparkEndpoint { capacity: shape.rows as u32,
                ingress: cuteafd_core::SparkIngress::Bf16, ring_depth: None, partial_reply_bytes: None },
            rings: cuteafd_core::SparkRings { max_frame_bytes: 64 << 20, depth: 8,
                slot_bytes: 8 << 20, alignment: 4096 } },
        shape.waves as usize, prefill, true, SparkIntakeCandidates::Probed,
        cuteafd_core::SparkReplyFormat::Bf16, Lifetime::TARGET).unwrap()
}

fn production_input(cfg: &GlmDsaConfig, gpus: usize, ranks: usize, prefill: PrefillShape)
    -> ProductionInput<'_> {
    let mut budgets = budgets();
    for rank in 2..ranks {
        budgets.push(Budget { id: format!("spark.{rank}"),
            locations: vec![Location { device: Device::Spark(rank as u8), domain: Domain::Managed }],
            physical_bytes: 1 << 40, logical_bytes: 1 << 40,
            baseline: Baseline::Forecast { occupied: 0, source: "test physical witness".into(), assumed: false },
            headroom: 0 });
    }
    ProductionInput { expert_usage: false, cfg, layers: cfg.layers, context: 4096, maximum_pool_tokens: 8192,
        attention: if gpus == 2 { AttentionPlacement::Context } else { AttentionPlacement::Heads },
        gpus, prefill, blas_head: serving::BlasHead::TensorLt, blas: vec![],
        target: TargetFacts {
            weights: vec![(EmbeddingHome::Device(0), WeightInventory { ranks: gpus, owners: vec![],
                core_loading: Default::default(), kv_b_loading: Default::default(), row_loading: Default::default(), attention_mount: Default::default() })],
            hooks: (0..gpus).map(|rank| HookBuffers { device: rank as u8, maximum_pages: 1,
                native: None, core: None, shared_workspace: None, full_prefill_logits: false }).collect(),
            embedding_read: None, native_load: None,
        },
        draft: DraftFacts::default(), scratch: Default::default(),
        context_scratch: (gpus == 2).then(super::super::GlmContext::default),
        exchange_element_bytes: 2, peer_diagnostics: false, local: None, spark_ranks: ranks,
        prefix_host: gpus == 1,
        decode_link: (ranks > 0).then(|| production_link(cfg, ranks, PrefillShape { rows: 64, waves: 1 }, false)),
        prefill_link: (ranks > 0).then(|| production_link(cfg, ranks, prefill, true)),
        skip_routed_experts: false, operation: Operation::Serving,
        sync_upload_bytes: 0, owners: vec![], budgets }
}

#[test]
fn blas_target_and_bridge_share_root_while_media_owns_its_thread() {
    use crate::plan::encoder::{EncoderKind, EncoderPlacement};
    let cfg = cfg();
    for head in [serving::BlasHead::Native, serving::BlasHead::TensorLt, serving::BlasHead::Gemm] {
        for draft in [false, true] {
            let mut input = production_input(&cfg, 2, 2, PrefillShape { rows: 64, waves: 1 });
            input.blas_head = head;
            if draft { input.draft = DraftFacts::default().with_window_marks(1024); }
            let mut prepared = input.prepare().unwrap();
            prepared.recipe.encoders = vec![EncoderPlacement { kind: EncoderKind::Rtx { gpu: 1 },
                weights: 1024, scratch: 4096, replicas: vec![], reason: "test".into(), shortfall: 0 }];
            let contract = prepared.recipe.prepare(&request()).unwrap();
            let selected = contract.facts(&vec![0; contract.variables.len()], 4).unwrap();
            let handles = |rank| selected.allowances.iter().filter(|owner|
                owner.location == Location::coordinator(rank) && owner.source.starts_with("cuBLAS handles:")
                    && owner.lifetime.contains(Phase::Ready, contract.phase_order))
                .map(|owner| owner.bytes).sum::<u64>();
            assert_eq!(handles(0), u64::from(head.gemm() || (head.bf16() && draft)) * (64 << 20));
            assert_eq!(handles(1), 64 << 20);
        }
    }
}

#[test]
fn target_checkpoint_facts_use_exact_inventory_and_preserve_embedding_alternatives() {
    use crate::plan::{testing, Checkpoint};
    let dir = tempfile::tempdir().unwrap();
    let (config, tensors) = testing::glm5_compact_weights();
    testing::write_snapshot(dir.path(), &config, &tensors, None);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = GlmDsaConfig::from_hf(&config).unwrap();
    let homes = [EmbeddingHome::Device(0), EmbeddingHome::HostPinned];
    let target = TargetFacts::checkpoint(&checkpoint, cfg.layers, 2, None, false, &homes, None).unwrap();
    assert_eq!(target.weights.len(), homes.len());
    for ((home, actual), expected_home) in target.weights.iter().zip(homes) {
        assert_eq!(*home, expected_home);
        let expected = WeightInventory::from_checkpoint(&checkpoint, 2, cfg.layers, None,
            false, false, expected_home).unwrap();
        assert_eq!(actual.owners.iter().map(|owner| &owner.allocation).collect::<Vec<_>>(),
            expected.owners.iter().map(|owner| &owner.allocation).collect::<Vec<_>>());
        assert_eq!(actual.core_loading, expected.core_loading);
        assert_eq!(actual.kv_b_loading, expected.kv_b_loading);
        assert_eq!(actual.row_loading, expected.row_loading);
    }
    assert_eq!(target.hooks.iter().map(|hook| hook.device).collect::<Vec<_>>(), [0, 1]);
    assert_eq!(target.embedding_read.unwrap().pinned_bytes(),
        crate::shared::dense_read::ReadBatch::new(cfg.vocab_size, cfg.hidden * 2, cfg.hidden * 2,
            crate::shared::dense_read::STEP_BYTES).unwrap().pinned_bytes());
    assert!(target.native_load.is_none());
    for homes in [vec![], vec![EmbeddingHome::Device(0); 2]] {
        assert!(TargetFacts::checkpoint(&checkpoint, cfg.layers, 2, None, false, &homes, None).is_err());
    }
}

fn gqa_layout(representation: crate::families::glm5::draft_representation::GlmDraftRepresentation)
    -> crate::families::glm5::draft_physical::GlmDflashPhysicalLayout {
    use crate::families::glm5::{draft_physical::{GlmDflashConfig, GlmDflashPhysicalLayout},
        draft_representation::GlmDraftCapacity};
    let cfg = GlmDflashConfig { hidden: 256, intermediate: 512, layers: 2, heads: 4,
        kv_heads: 2, head_dim: 64, eps: 1e-5, theta: 10000., block: 4, group: 2,
        mask_token: 0, rank: 256, taps: vec![0, 1], vocab: 64, window: 2048,
        row_window: false, sample_from_anchor: false };
    GlmDflashPhysicalLayout::new(&cfg, representation, GlmDraftCapacity::new(20, 16, cfg.block).unwrap(),
        188, 0, crate::families::qwen4::Qwen4KvCache::Bf16).unwrap()
}

#[test]
fn packed_draft_is_charged_once_during_push_target_load() {
    use crate::families::glm5::{draft_physical::{GlmDflashConfig, GlmDflashPhysicalLayout, gqa_arena_requests},
        draft_representation::{GlmDraftCapacity, GlmDraftRepresentation}};
    let draft = GlmDflashConfig { hidden: 256, intermediate: 512, layers: 2, heads: 4,
        kv_heads: 2, head_dim: 64, eps: 1e-5, theta: 10000., block: 4, group: 2,
        mask_token: 0, rank: 256, taps: vec![0, 1], vocab: 64, window: 2048,
        row_window: false, sample_from_anchor: false };
    for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
        let layout = GlmDflashPhysicalLayout::new(&draft, representation,
            GlmDraftCapacity::new(20, 16, draft.block).unwrap(), 188, 0,
            crate::families::qwen4::Qwen4KvCache::Bf16).unwrap();
        let mut facts = DraftFacts::from_gqa(&layout).unwrap();
        let old = facts.weights.iter().map(|owner| owner.weights.0).sum::<u64>();
        facts.pack_arena(gqa_arena_requests(&draft, representation, 0, 1).unwrap()).unwrap();
        let arena = facts.arena.as_ref().unwrap();
        assert_eq!(old, arena.payload_bytes() as u64);
        assert!(facts.weights.is_empty());
        assert!(!facts.owners.iter().any(|owner| layout.weight_owners().any(|weight|
            owner.id == format!("glm.draft.{}", weight.id))));
        let cfg = cfg();
        let mut input = production_input(&cfg, 1, 4, PrefillShape { rows: 128, waves: 3 });
        input.draft = facts.clone();
        let solved = input.prepare().unwrap().admit(request()).unwrap().solved;
        let root = solved.facts.allocations.iter().find(|owner| owner.id == "glm/draft/0").unwrap();
        assert_eq!(root.bytes, arena.bytes() as u64);
        assert_eq!(root.lifetime, Lifetime::RESIDENT);
        let tail = solved.facts.allocation_granularity.iter().find(|owner|
            owner.backings.iter().any(|backing| backing.backing == root.id)).unwrap();
        assert_eq!(tail.lifetime, root.lifetime);
        assert_eq!(tail.backings.len(), 1);
        assert_eq!(tail.backings[0].requests, [root.bytes]);
        let load = solved.phases.iter().find(|row| row.phase == Phase::Load
            && solved.budgets.iter().any(|budget| budget.id == row.budget
                && budget.locations.contains(&Location::coordinator(0)))).unwrap();
        let mut without = solved.facts.clone();
        without.allocations.retain(|owner| owner.id != root.id);
        without.resident_weights.retain(|owner| owner.id != root.id);
        without.allocation_granularity.retain(|owner| owner.id != tail.id);
        let baseline = planner::evaluate(&without, &solved.budgets).unwrap().into_iter()
            .find(|row| row.phase == Phase::Load && row.budget == load.budget).unwrap();
        assert_eq!(load.owned - baseline.owned, root.bytes);
        assert_eq!(load.opaque - baseline.opaque, tail.bytes().unwrap());
    }
}

#[test]
fn draft_facts_preserve_owner_ids_resources_and_serial_fp8_sources() {
    use crate::families::glm5::draft_representation::GlmDraftRepresentation::{Bf16Only, Fp8Only};
    for representation in [Bf16Only, Fp8Only] {
        let layout = gqa_layout(representation);
        let facts = DraftFacts::from_gqa(&layout).unwrap();
        assert_eq!(facts.owners, layout.allocations("glm.draft", Location::coordinator(0)));
        let ids = facts.owners.iter().map(|owner| owner.id.as_str()).collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), facts.owners.len());
        for weight in &facts.weights {
            let owner = facts.owners.iter().find(|owner| owner.id == weight.id).unwrap();
            assert_eq!(weight.location, owner.location);
            assert_eq!(weight.weights.0, owner.bytes);
        }
        assert_eq!(facts.weights.len(), layout.weight_owners().count());
        let values = layout.weight_owners().filter(|owner| owner.id.ends_with(".values"))
            .collect::<Vec<_>>();
        if representation == Bf16Only {
            assert!(facts.loading.is_empty());
        } else {
            assert_eq!(facts.loading.len(), 1);
            assert_eq!(facts.loading[0].members.len(), values.len());
            assert_eq!(facts.loading[0].peak(), values.iter().map(|owner| owner.bytes * 2).max().unwrap());
            assert!(facts.loading[0].members.iter().all(|owner| owner.location == Location::coordinator(0)
                && owner.lifetime == Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad }));
        }
        assert!(facts.components(0, 0).is_err());
        let cfg = cfg();
        let mut input = production_input(&cfg, 1, 4, PrefillShape { rows: 128, waves: 3 });
        input.draft = facts.clone();
        let solved = input.prepare().unwrap().admit(request()).unwrap().solved;
        for weight in &facts.weights {
            assert!(solved.facts.resident_weights.contains(weight));
        }
        for owner in facts.owners.iter().filter(|owner| owner.location != Location::HOST) {
            assert!(solved.facts.allocations.contains(owner));
        }
        assert_eq!(solved.facts.allocations.iter().find(|owner| owner.id == "glm.load.sync_h2d").unwrap().bytes,
            facts.sync_upload_bytes);
        assert_eq!(solved.facts.sequential.iter().filter(|group| group.id == "glm.draft.load.sources").count(),
            usize::from(representation == Fp8Only));
    }
}

#[test]
fn draft_home_is_one_solve_choice_that_balances_and_charges_its_bridge() {
    use crate::families::glm5::draft_representation::GlmDraftRepresentation::Fp8Only;
    let cfg = cfg();
    let facts = DraftFacts::from_gqa(&gqa_layout(Fp8Only)).unwrap();
    assert!(facts.bridge().is_empty() && facts.homes(2) == [0], "GPU1 needs the bridge inventory");
    let mut facts = facts;
    facts.bridge = DraftFacts::bridge_demands(256, 64, 2, 64).unwrap();
    assert_eq!((facts.homes(1), facts.homes(2)), (vec![0], vec![0, 1]));
    assert!(DraftFacts::default().homes(2).is_empty());
    assert!(facts.components(1, 64).is_ok() && facts.components(2, 64).is_err());
    let draft_bytes = facts.owners.iter().filter(|owner| owner.location == Location::coordinator(0))
        .map(|owner| owner.bytes).sum::<u64>();
    let bridge0 = facts.bridge().iter().filter(|demand| demand.gpu == 0).map(|demand| demand.bytes).sum::<u64>();
    assert!(draft_bytes > bridge0, "fixture drafter must outweigh its GPU0 bridge staging");
    let solve = |gpu: u8| {
        let mut input = production_input(&cfg, 2, 2, PrefillShape { rows: 128, waves: 3 });
        input.draft = facts.clone();
        input.owners.push(Component::allocation(Allocation::buffer("test.pinned", Location::coordinator(gpu),
            1 << 30, Lifetime::TARGET, Category::Weights, "single-copy owner pinned to one GPU"),
            ByteCurve::fixed(1 << 30, 64), false, 64));
        input.prepare().unwrap().admit(request()).unwrap()
    };
    // GPU1 already binds: the drafter stays beside its target head, no bridge.
    let local = solve(1);
    assert_eq!(local.solved.decisions.draft_home, Some(0));
    assert_eq!(local.placement.movables, vec![(crate::placement::MovableId::Drafter, 0)]);
    assert!(!local.solved.facts.allocations.iter().any(|owner| owner.id.starts_with("glm.draft.bridge.")));
    // A GPU0-pinned owner makes GPU1 the home that lowers the binding GPU.
    let remote = solve(0);
    assert_eq!(remote.solved.decisions.draft_home, Some(1));
    assert_eq!(remote.placement.movables, vec![(crate::placement::MovableId::Drafter, 1)]);
    for owner in facts.owners.iter().filter(|owner| owner.location == Location::coordinator(0)) {
        let moved = remote.solved.facts.allocations.iter().find(|a| a.id == owner.id).unwrap();
        assert_eq!(moved.location, Location::coordinator(1));
    }
    for weight in &facts.weights {
        assert!(remote.solved.facts.resident_weights.iter().any(|w| w.id == weight.id
            && w.location == Location::coordinator(1) && w.weights == weight.weights));
    }
    let bridge = remote.solved.facts.allocations.iter().filter(|owner| owner.id.starts_with("glm.draft.bridge."))
        .map(|owner| (owner.location, owner.bytes)).collect::<Vec<_>>();
    assert_eq!(bridge, facts.bridge().iter().map(|demand| (Location::coordinator(demand.gpu), demand.bytes))
        .collect::<Vec<_>>());
    assert!(bridge.iter().any(|(location, _)| *location == Location::coordinator(1)), "controls charged on both GPUs");
}

#[test]
fn ml4_draft_facts_preserve_native_owners_and_share_the_library_upload_bank() {
    use crate::formats::{block_dspark::DsparkRotaryPlan, dflash::MlaDflashConfig, mla::MlaGeometry};
    use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation::Bf16Only};
    let cfg = MlaDflashConfig { mla: MlaGeometry { hidden: 6144, heads: 64, q_rank: 0,
        kv_rank: 512, nope: 64, rope: 64, value: 128 }, intermediate: 24576, eps: 1e-5,
        block: 3, group: 0, selector_rank: 512, selector_top_k: 0, taps: vec![1, 18, 36, 54, 71],
        vocab: 204800, mask_token: 0, max_context: 1 << 20, windows: vec![Some(2048); 3],
        rotary: DsparkRotaryPlan { inv_freq: vec![1.; 32], attention_factor: 1., softmax_factor: 1. },
        kind: cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark, markov_rank: 512 };
    let layout = cfg.runtime_layout(Bf16Only, GlmDraftCapacity::new(20, 16, 3).unwrap(),
        1 << 20, 1 << 21, 188, 0, cuteafd_core::MlaCacheFormat::Fp8).unwrap();
    let facts = DraftFacts::ml4(&layout).unwrap();
    let retained = layout.owners().iter().filter(|owner| !layout.is_immutable_buffer(&owner.id)
        && !owner.on_use && owner.shared_key.as_deref() != Some("library/sync-h2d")).collect::<Vec<_>>();
    assert_eq!(facts.owners.len(), retained.len() + 1);
    for expected in retained {
        let owner = facts.owners.iter().find(|owner| owner.id == format!("glm.draft.{}", expected.id)).unwrap();
        assert_eq!(owner.bytes, expected.bytes as u64);
        assert_eq!(owner.alignment, expected.alignment as u64);
        assert_eq!(owner.shape, expected.shape.iter().map(|&n| n as u64).collect::<Vec<_>>());
    }
    assert!(facts.weights.is_empty());
    assert_eq!(facts.arena_weights, layout.weights().resident_bytes().unwrap());
    let arena = facts.arena.as_ref().unwrap();
    assert_eq!(arena.views().len(), layout.weight_destinations().len() + 1);
    assert!(arena.view("markov_norms").is_some());
    let components = facts.components(0, 64).unwrap();
    let roots = components.iter().filter_map(|component| match &component.kind {
        ComponentKind::Allocation { template, .. } if template.id == "glm/draft/0" => Some(template), _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].bytes, arena.bytes() as u64);
    assert_eq!(roots[0].lifetime, Lifetime::RESIDENT);
    assert_eq!(facts.sync_upload_bytes, layout.owners().iter()
        .filter(|owner| owner.shared_key.as_deref() == Some("library/sync-h2d"))
        .map(|owner| owner.bytes as u64).max().unwrap());
    let mount = facts.owners.iter().find(|owner| owner.id == "glm.draft.load.mount").unwrap();
    assert_eq!(mount.location, Location::PINNED);
    assert_eq!(mount.bytes, crate::shared::dense_read::STEP_BYTES as u64);
    assert_eq!(mount.lifetime, Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad });
    assert!(facts.loading.is_empty(), "ML4 BF16 storage must not invent GQA FP8 source matrices");
}

#[test]
fn native_load_keeps_exact_banks_and_serializes_ranks_without_host_cache_double_charge() {
    use crate::plan::exlr8::{dense::DensePlan, residency::{ExpertSet, Residency}};
    use super::super::local_physical::LocalStorage;
    let dense = (0..2).map(|rank| DensePlan { rank, world: 2, tensors: vec![],
        resident_bytes: 0, pinned_staging_bytes: (rank + 1) * 1_048_576,
        cached_rows_bytes: (rank + 1) * 8_388_608,
        read_staging_bytes: (rank + 1) * 9_437_184 }).collect::<Vec<_>>();
    let vocabulary = DensePlan { rank: 0, world: 2, tensors: vec![], resident_bytes: 0,
        pinned_staging_bytes: 32 << 20, cached_rows_bytes: 0, read_staging_bytes: 32 << 20 };
    let plans = (0..2).map(|rank| (1..3).map(|layer| Residency {
        layer, world: 2, rank, hidden: 64, experts: 1, units: 2, set: ExpertSet::Shared,
        set_id: String::new(), manifest: String::new(), loads: vec![], signs: [vec![], vec![]],
        signs_offsets: [0, 0], pointers_offset: 0, bits_offset: 0, compact_offsets: [0; 2],
        resident_bytes: 4096, payload_bytes: 0, staging_bytes: (rank + layer) * 32,
    }).collect::<Vec<_>>()).collect::<Vec<_>>();
    let refs = dense.iter().collect::<Vec<_>>();
    assert_eq!(DensePlan::mount_staging_bytes(&refs).unwrap(), (2 << 20, 16 << 20));
    assert!(DensePlan::mount_staging_bytes(&[&dense[1], &dense[0]]).is_err());
    let mount = NativeLoad::new(&refs, &vocabulary, &plans, LocalStorage::Mount).unwrap();
    mount.validate_vocabulary(0).unwrap();
    assert!(mount.validate_vocabulary(1).is_err());
    for plan in &plans { mount.validate_shared(plan).unwrap(); }
    let mut changed = plans[0].clone();
    changed[0].staging_bytes += 1;
    assert!(mount.validate_shared(&changed).is_err());
    changed[0].staging_bytes -= 1;
    changed[0].world = 1;
    assert!(mount.validate_shared(&changed).is_err());
    assert_eq!(mount.groups.len(), 33);
    assert_eq!(mount.groups[0].members.len(), 1);
    assert_eq!(mount.groups[0].peak(), 2 << 20,
        "one reused Mount bank, not two ranks or the mixed-domain aggregate");
    assert_eq!(mount.sync_upload_bytes, 1 * 2 * 3 * 3 * 8);
    assert_eq!(mount.coarse_overcharge_bytes, 32 * 96);
    let mut ids = BTreeSet::new();
    for group in &mount.groups[1..] {
        assert_eq!(group.members.len(), 4);
        assert_eq!(group.peak(), 96);
        for owner in &group.members {
            assert!(ids.insert(owner.id.clone()));
            assert_eq!(owner.location, Location::PINNED);
            assert_eq!(owner.lifetime, Lifetime { first: Phase::Load, last: Phase::Load });
        }
    }
    assert_eq!(ids.len(), 128);
    let push = NativeLoad::new(&refs, &vocabulary, &plans, LocalStorage::Push).unwrap();
    assert_eq!(push.groups.len(), 32,
        "Push does not also create the native Mount bank; shared experts still load separately");
    assert_eq!(push.coarse_overcharge_bytes, 0);
    let components = mount.components(1);
    assert_eq!(components.iter().filter(|component|
        matches!(component.kind, ComponentKind::Sequential(_))).count(), 33);
}

#[test]
fn production_native_sync_bank_is_one_maximum_not_one_copy_per_rank_or_layer() {
    let cfg = cfg();
    let prefill = PrefillShape { rows: 128, waves: 3 };
    let mut input = production_input(&cfg, 2, 4, prefill);
    input.sync_upload_bytes = 4096;
    input.target.native_load = Some(NativeLoad { gpus: 2, sync_upload_bytes: 8192,
        coarse_overcharge_bytes: 0, groups: vec![] });
    let prepared = input.prepare().unwrap();
    let solved = prepared.admit(request()).unwrap().solved;
    let banks = solved.facts.allocations.iter().filter(|owner| owner.id == "glm.load.sync_h2d")
        .collect::<Vec<_>>();
    assert_eq!(banks.len(), 1);
    assert_eq!(banks[0].bytes, 8192);
}

#[test]
fn production_min_and_max_keep_devs_spark_prefill_shape() {
    use crate::serving_capacity::prefill_waves::{prefill_shape, PrefillLimits, PrefillOverrides};
    let cfg = cfg();
    // Both example cards leave rows/waves unpinned. The launcher's shared depth
    // resolver selects three waves; the old solver never transitions to one.
    for (gpus, ranks) in [(1, 4), (2, 6)] {
        let dev = prefill_shape(ranks, PrefillLimits::default(), PrefillOverrides::default()).unwrap();
        assert_eq!((dev.unit_rows, dev.waves), (2752, 3));
        let prefill = PrefillShape { rows: dev.unit_rows, waves: dev.waves };
        let prepared = production_input(&cfg, gpus, ranks, prefill).prepare().unwrap();
        assert_eq!(prepared.recipe.prefill, vec![prefill]);
        let admitted = prepared.admit(request()).unwrap();
        assert_eq!(admitted.solved.decisions.prefill, prefill);
        assert_eq!(admitted.spark.prefill_lanes, dev.waves);
        assert_eq!(admitted.spark.spark_ranks, ranks);
        assert!(admitted.decode_link.is_some() && admitted.prefill_link.is_some());
    }
}

#[test]
fn production_count_domain_preserves_executable_wave_capabilities() {
    assert_eq!(production_counts(3, false, 3, false), [0]);
    assert_eq!(production_counts(3, true, 1, false), [0, 1, 2, 3]);
    assert_eq!(production_counts(3, true, 3, false), [0, 1, 2]);
    assert_eq!(production_counts(0, true, 3, false), [0]);
    assert_eq!(production_counts(3, true, 3, true), [0]);
    let cfg = cfg();
    let prefill = PrefillShape { rows: 128, waves: 3 };
    let mut prepared = production_input(&cfg, 2, 4, prefill).prepare().unwrap();
    prepared.recipe.experts.as_mut().unwrap().counts = production_counts(1, true, 3, false);
    let mut exact = request();
    exact.onboard = Pin::Exact(1);
    assert!(planner::solve(exact, prepared.budgets, &prepared.recipe).is_err(),
        "an unavailable full count must refuse, not clamp or change shape");
}

#[test]
fn production_rejects_misindexed_hooks_and_spark_free_multiwave() {
    let cfg = cfg();
    let mut input = production_input(&cfg, 2, 4, PrefillShape { rows: 128, waves: 3 });
    input.target.hooks.swap(0, 1);
    assert!(input.prepare().is_err());
    let input = production_input(&cfg, 2, 0, PrefillShape { rows: 128, waves: 3 });
    assert!(input.prepare().is_err());
}

#[test]
fn production_whole_local_has_no_pair_owners_and_rejects_wrong_world() {
    use super::super::local_physical::{LocalPhysicalInput, LocalPhysicalSource, LocalStorage};
    let mut cfg = cfg();
    // Keep a remote suffix: full onboarding is deliberately unavailable at three waves.
    cfg.layers += 1;
    cfg.indexers.push(crate::families::glm5::GlmIndexer::Shared);
    let prefill = PrefillShape { rows: 128, waves: 3 };
    let source = LocalPhysicalSource::test_source(1, cfg.first_moe_layer..cfg.layers, 128);
    let physical = LocalPhysicalInput { layers: (cfg.first_moe_layer..cfg.layers).collect(),
        max_rows: 128, lanes: 3, storage: LocalStorage::Mount, route_checks: true,
        peer_diagnostics: true, activations: crate::plan::exlr8::PrefillActivations::A16 };
    let mut input = production_input(&cfg, 1, 4, prefill);
    input.local = Some((&source, physical.clone()));
    let mut req = request();
    req.onboard = Pin::Exact(1);
    let admitted = input.prepare().unwrap().admit(req).unwrap();
    assert_eq!(admitted.placement.onboard_layers, 1);
    assert!(admitted.solved.facts.allocations.iter().any(|owner|
        owner.id == format!("glm.expert.routed.rank.0.weight.{}.backing", cfg.first_moe_layer)));
    assert!(admitted.solved.facts.allocations.iter().all(|owner|
        owner.location != Location::coordinator(1)
            && !owner.id.contains(".peer.") && !owner.id.contains(".checks.")
            && !owner.id.ends_with(".rows.payload")));
    assert!(admitted.graphs.layers[cfg.first_moe_layer].local);
    let mut input = production_input(&cfg, 2, 4, prefill);
    input.local = Some((&source, physical));
    assert!(matches!(input.prepare(), Err(Refusal::InvalidFact(_))));
}

#[test]
fn glm_zero_demand_omits_charge_and_positive_demand_has_one_exact_item() {
    let cfg = cfg();
    for bytes in [0, 768] {
        let mut recipe = recipe(&cfg);
        let owner = Allocation::buffer("glm.optional.draft_marks", Location::coordinator(0), bytes,
            Lifetime::TARGET, Category::Prefix, "optional draft marks");
        recipe.owners.push(Case { when: vec![], value: vec![Component::allocation(owner,
            ByteCurve::fixed(bytes, recipe.maximum_pages), false, recipe.maximum_pages)] });
        let resolved = planner::solve(request(), budgets(), &recipe).unwrap();
        let owners: Vec<_> = resolved.facts.allocations.iter()
            .filter(|a| a.id == "glm.optional.draft_marks").collect();
        let items = resolved.items(Location::coordinator(0)).unwrap();
        let charges: Vec<_> = items.iter().filter(|item| item.group == "optional draft marks").collect();
        if bytes == 0 {
            assert!(owners.is_empty());
            assert!(charges.is_empty(), "zero demand has no manufactured charge");
        } else {
            assert_eq!(owners.len(), 1);
            assert_eq!(owners[0].bytes, bytes);
            assert_eq!(charges.len(), 1);
            assert_eq!(charges[0].bytes, bytes);
        }
    }
}

#[test]
fn glm_draft_first_peak_excludes_target_kv_and_changes_tight_pool_admission() {
    struct Prepared(Contract);
    impl Recipe for Prepared {
        fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
            Ok(self.0.clone())
        }
    }

    // Synthetic retained/load draft owners isolate chronology; this is not a
    // native drafter census or a complete card forecast.
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    let owners = [
        Allocation::buffer(
            "draft.retained",
            Location::coordinator(0),
            1 << 30,
            Lifetime {
                first: Phase::DraftLoad,
                last: Phase::Growth,
            },
            Category::Drafter,
            "synthetic retained draft",
        ),
        Allocation::buffer(
            "draft.load",
            Location::coordinator(0),
            1 << 30,
            Lifetime {
                first: Phase::DraftLoad,
                last: Phase::DraftLoad,
            },
            Category::Staging,
            "synthetic draft load",
        ),
    ];
    recipe.owners.push(Case {
        when: vec![],
        value: owners
            .into_iter()
            .map(|owner| {
                let bytes = ByteCurve::fixed(owner.bytes, recipe.maximum_pages);
                Component::allocation(owner, bytes, false, recipe.maximum_pages)
            })
            .collect(),
    });
    let req = Request {
        pool: Pin::Auto,
        onboard: Pin::Exact(0),
        prefill: Pin::Exact(recipe.prefill[0]),
        ..request()
    };
    let actual = recipe.prepare(&req).unwrap();
    assert_eq!(actual.phase_order, PhaseOrder::DRAFT_THEN_TARGET);
    let mut target_first = actual.clone();
    target_first.phase_order = PhaseOrder::TARGET_THEN_DRAFT;
    let selected = vec![0; actual.variables.len()];
    let pages = req.context.maximum / recipe.page_tokens;
    let facts = actual.facts(&selected, pages).unwrap();
    let target_bytes: u64 = facts
        .allocations
        .iter()
        .filter(|a| {
            a.location == Location::coordinator(0)
                && a.lifetime.first == Phase::PostTargetKv
                && a.view.is_none()
        })
        .map(|a| a.bytes)
        .sum();
    assert!(target_bytes > 0);
    let room = budgets();
    let after = planner::evaluate(&facts, &room).unwrap();
    let before = planner::evaluate(&target_first.facts(&selected, pages).unwrap(), &room).unwrap();
    let draft_peak = |rows: &[PhaseBudget]| {
        rows.iter()
            .find(|row| row.budget == "physical.0" && row.phase == Phase::DraftLoad)
            .unwrap()
            .owned
    };
    assert_eq!(draft_peak(&before) - draft_peak(&after), target_bytes);
    for rows in [&before, &after] {
        let peak = rows
            .iter()
            .filter(|row| row.budget == "physical.0")
            .max_by_key(|row| row.owned + row.opaque)
            .unwrap();
        assert_eq!(peak.phase, Phase::DraftLoad);
    }
    let mut tight = room;
    tight[0].physical_bytes = draft_peak(&before);
    tight[0].logical_bytes = draft_peak(&before);
    let old = planner::solve(req.clone(), tight.clone(), &Prepared(target_first)).unwrap();
    let new = planner::solve(req, tight, &Prepared(actual)).unwrap();
    assert_eq!(old.pool_tokens, 4096);
    assert_eq!(new.pool_tokens, 8192);
    eprintln!(
        "GLM bounded owner fixture: pool {} -> {}; DraftLoad peak {} -> {} B; target overlap {} B",
        old.pool_tokens,
        new.pool_tokens,
        draft_peak(&before),
        draft_peak(&after),
        target_bytes
    );
}

fn request() -> Request {
    Request {
        context: (4096).into(),
        pool: Pin::Exact(4096),
        onboard: Pin::Auto,
        prefill: Pin::Auto,
        spark_reply_formats: SparkReplyFormats::default(),
        operation: Operation::Serving,
        resident_weight_limit: None,
    }
}

fn recipe(cfg: &GlmDsaConfig) -> GlmRecipe<'_> {
    let prefill = vec![
        PrefillShape { rows: 64, waves: 1 },
        PrefillShape {
            rows: 128,
            waves: 3,
        },
    ];
    let mut recipe = GlmRecipe {
        expert_usage: None,
        cfg,
        context: 4096,
        page_tokens: 128,
        maximum_pages: 64,
        attention: vec![LayerMode::ContextSplit; cfg.layers],
        gpus: 2,
        spark_links: true,
        prefill,
        lanes: vec![1],
        encoders: vec![], media_cache_bytes: 0, blas: vec![],
        embedding: vec![EmbeddingHome::Device(0), EmbeddingHome::HostPinned],
        draft_homes: vec![],
        experts: Some(ExpertProfile {
            layers: vec![1],
            destination: vec![0, 1],
            counts: vec![0, 1],
            rank_sets: vec![vec![0, 1]],
        }),
        targets: vec![],
        hooks: vec![],
        exchanges: vec![],
        weights: vec![],
        graphs: vec![],
        host: None,
        owners: vec![],
    };
    for &shape in &recipe.prefill {
        for local in [false, true] {
            let when = vec![
                When::Prefill(shape),
                if local {
                    When::LocalLayer(1)
                } else {
                    When::RemoteLayer(1)
                },
            ];
            for device in 0..2 {
                recipe.targets.push(Case {
                    when: when.clone(),
                    value: TargetBuffers {
                        device,
                        lead: device == 0,
                        page_tokens: recipe.page_tokens,
                        maximum_pages: recipe.maximum_pages,
                        context: recipe.context,
                        partitioned: true,
                        owned_layers: vec![true; cfg.layers],
                        split_heads: true,
                        router: local,
                        native_head: false,
                        decode: Some(super::super::GlmScratch {
                            programs: 1024,
                            topk: 4096,
                        }),
                        prefill: Some((
                            shape,
                            super::super::GlmScratch {
                                programs: 2048,
                                topk: 8192,
                            },
                        )),
                        full_prefill_logits: false,
                        payload_row_bytes: if local { cfg.hidden as u64 * 4 } else { 0 },
                        core_rows: false, attention_partial_bytes: 2,
                    },
                });
            }
        }
        recipe.exchanges.push(Case {
            when: vec![When::Prefill(shape)],
            value: ExchangeBuffers {
                physical_devices: [0, 1],
                maximum_pages: recipe.maximum_pages,
                receive_bytes: vec![shape.rows * cfg.hidden as u64 * 2; (4 * shape.waves) as usize],
                terminal_abort: true,
                diagnostics: true,
                context: Some(super::super::exchanges::ContextOwners {
                    extent: recipe.context,
                    prefill: shape,
                    scratch: super::super::GlmContext {
                        topk_scratch: 512,
                        partial_scratch: 1024,
                    },
                }),
            },
        });
    }
    for home in &recipe.embedding {
        recipe.weights.push(Case {
            when: vec![When::Embedding(*home)],
            value: WeightInventory {
                ranks: 2,
                core_loading: super::super::weights::CoreLoading::default(),
                kv_b_loading: super::super::weights::KvBLoading::default(),
                row_loading: super::super::weights::RowLoading::default(), attention_mount: Default::default(),
                owners: vec![super::super::weights::WeightOwner {
                    allocation: Allocation::buffer(
                        "glm.weights.embedding",
                        home.location(),
                        8192,
                        Lifetime::RESIDENT,
                        Category::Embedding,
                        "test physical embedding",
                    ),
                    group: "embedding".into(),
                    format: "bf16".into(),
                }],
            },
        });
    }
    for gpu in 0..2 {
        let owner = Allocation::buffer(
            format!("expert.local.{gpu}.1"),
            Location::coordinator(gpu),
            16384,
            Lifetime::RESIDENT,
            Category::Weights,
            "test local expert arena",
        );
        let bytes = ByteCurve::fixed(owner.bytes, recipe.maximum_pages);
        recipe.owners.push(Case {
            when: vec![When::LocalLayer(1)],
            value: vec![
                Component::allocation(owner.clone(), bytes.clone(), false, recipe.maximum_pages),
                Component::resident_weight(
                    ResidentWeight::from_allocation(
                        &owner,
                        crate::placement::ResidentWeights(owner.bytes),
                    ),
                    bytes,
                ),
            ],
        });
    }
    for rank in 0..2 {
        recipe.owners.push(Case {
            when: vec![When::RemoteLayer(1), When::SparkRanks(vec![0, 1])],
            value: vec![Component::allocation(
                Allocation::buffer(
                    format!("expert.spark.{rank}.1"),
                    Location {
                        device: Device::Spark(rank),
                        domain: Domain::Managed,
                    },
                    12288,
                    Lifetime::RESIDENT,
                    Category::Weights,
                    "test Spark expert arena",
                ),
                ByteCurve::fixed(12288, recipe.maximum_pages),
                false,
                recipe.maximum_pages,
            )],
        });
    }
    recipe
}

fn budgets() -> Vec<Budget> {
    vec![
        vec![Location::coordinator(0)],
        vec![Location::coordinator(1)],
        vec![Location::HOST, Location::PINNED],
        vec![Location {
            device: Device::Spark(0),
            domain: Domain::Managed,
        }],
        vec![Location {
            device: Device::Spark(1),
            domain: Domain::Managed,
        }],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, locations)| Budget {
        id: format!("physical.{i}"),
        locations,
        physical_bytes: 1 << 40,
        logical_bytes: 1 << 40,
        baseline: Baseline::Forecast {
            occupied: 0,
            source: "test physical witness".into(),
            assumed: false,
        },
        headroom: 0,
    })
    .collect()
}

fn local_execution_inventory() -> super::super::local_physical::LocalPhysicalInventory {
    use super::super::local_physical::LocalPhysicalInventory;
    let mut inventory = LocalPhysicalInventory::default();
    for rank in 0..2 {
        let location = Location::coordinator(rank);
        for (suffix, bytes) in [("weight.1.backing", 100 + u64::from(rank)),
            ("execution.scratch", 13 + 4 * u64::from(rank))] {
            inventory.owners.push(Allocation::buffer(format!("glm.expert.routed.rank.{rank}.{suffix}"),
                location, bytes, Lifetime::TARGET, Category::Experts, "local backend"));
        }
        inventory.owners.push(Allocation::buffer(format!("glm.expert.routed.routes.rank.{rank}.peer.recv.0"),
            location, 201 + u64::from(rank), Lifetime::TARGET, Category::Experts, "local routes"));
        let mut view = Allocation::buffer(format!("glm.expert.routed.rank.{rank}.weight.1"),
            location, 50, Lifetime::TARGET, Category::Experts, "local backend");
        view.view = Some(View { backing: format!("glm.expert.routed.rank.{rank}.weight.1.backing"), offset: 0 });
        inventory.views.push(view);
    }
    inventory.owners.push(Allocation::buffer("glm.expert.routed.checks.0.1.0", Location::PINNED,
        81, Lifetime::TARGET, Category::Experts, "local route checks"));
    inventory
}

#[test]
fn execution_adapter_uses_resolved_homes_and_unique_tp2_prefix_and_backend_bytes() {
    use crate::placement::{ExpertHome, ResidualHome};
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    let local = local_execution_inventory();
    recipe.owners.retain(|case| !case.when.iter().any(|when| matches!(when, When::LocalLayer(_))));
    recipe.owners.push(Case { when: vec![When::LocalLayer(1)], value: local.owners.iter()
        .chain(&local.views).cloned().map(|owner| {
            let bytes = ByteCurve::fixed(owner.bytes, recipe.maximum_pages);
            Component::allocation(owner, bytes, false, recipe.maximum_pages)
        }).collect() });
    for onboard in [0, 1] {
        let resolved = planner::solve(Request { onboard: Pin::Exact(onboard), ..request() }, budgets(), &recipe).unwrap();
        let inventory = if onboard == 0 { Default::default() } else { local.clone() };
        let (placement, working) = recipe.execution(&resolved, &inventory).unwrap();
        assert_eq!(placement.pool_tokens, resolved.pool_tokens);
        assert!(placement.pool_limits.0.is_empty());
        assert_eq!(placement.max_pool_tokens_no_onboard, None);
        assert_eq!(placement.onboard_layers, onboard as usize);
        assert_eq!(placement.layers[0].experts, ExpertHome::Dense);
        assert_eq!(working.prefill_lanes, resolved.decisions.prefill.waves);
        assert_eq!(placement.residual, vec![ResidualHome::Owned(0), ResidualHome::Replicated, ResidualHome::Replicated]);
        let geometry = super::super::context_buffers(&cfg, recipe.context,
            resolved.decisions.prefill.rows, resolved.decisions.prefill.waves, Default::default());
        assert_eq!(cfg.indexers, vec![crate::families::glm5::GlmIndexer::Full,
            crate::families::glm5::GlmIndexer::Shared]);
        // Shared layers reuse indices: candidates cross the peer only for the full indexer.
        assert_eq!(placement.peer_row_bytes, 2 * (geometry.query_row_bytes + geometry.partial_row_bytes)
            + geometry.candidate_row_bytes);
        for rank in 0..2 {
            // Traffic is per layer; candidate storage is one reusable owner per rank.
            let candidates = resolved.facts.allocations.iter().filter(|owner|
                owner.id == format!("glm.exchange.rank.{rank}.context.candidates")).collect::<Vec<_>>();
            assert_eq!(candidates.len(), 1);
            assert_eq!(candidates[0].bytes, super::super::DECODE_ROWS * geometry.candidate_row_bytes);
            assert_eq!(placement.items[rank], resolved.items(Location::coordinator(rank as u8)).unwrap());
            assert_eq!(placement.expert_ranges[rank].layers, 0);
            assert_eq!(placement.expert_ranges[rank].peak_bytes, 0);
        }
        if onboard == 0 {
            assert_eq!(placement.layers[1].experts, ExpertHome::Spark);
            assert_eq!(working.rtx_layers, Vec::<usize>::new());
            assert_eq!(working.spark_layers, vec![1]);
            assert_eq!(working.selected_ranks().collect::<Vec<_>>(), vec![0, 1]);
            assert!(placement.tp2.is_none());
        } else {
            assert_eq!(resolved.decisions.onboard_layers, vec![vec![1], vec![1]]);
            assert_eq!(placement.layers[1].experts, ExpertHome::RtxTp2);
            assert_eq!(working.rtx_layers, vec![1]);
            assert!(working.spark_layers.is_empty());
            assert!(working.selected_ranks().next().is_none());
            let range = placement.tp2.as_ref().unwrap();
            assert_eq!((range.first, range.layers, range.peak_bytes), (1, 1, [113, 118]));
            assert!(placement.summary().contains("TP2"));
            assert_eq!(placement.items[0].iter().find(|item| item.group == "local routes").unwrap().bytes, 201);
            assert_eq!(placement.items[0].iter().find(|item| item.group == "local backend").unwrap().bytes, 113);
        }
    }
}

#[test]
fn execution_adapter_preserves_whole_expert_range_and_shared_hop_geometry() {
    use crate::placement::{ExpertHome, HopSpec, HopStorage, TerminalHome, plan_hops, residual_homes};
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    recipe.attention = vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
        LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }];
    recipe.targets.clear();
    recipe.exchanges.clear();
    recipe.experts.as_mut().unwrap().destination = vec![1];
    let mut local = local_execution_inventory();
    local.owners.retain(|owner| owner.location == Location::coordinator(1));
    local.views.retain(|owner| owner.location == Location::coordinator(1));
    let resolved = planner::solve(Request { onboard: Pin::Exact(1), ..request() }, budgets(), &recipe).unwrap();
    let (placement, working) = recipe.execution(&resolved, &local).unwrap();
    assert_eq!(placement.attention_placement, AttentionPlacement::Layers);
    assert_eq!(placement.layers[1].experts, ExpertHome::RtxWhole { gpu: 1 });
    assert_eq!(resolved.decisions.onboard_layers, vec![vec![], vec![1]]);
    assert!(placement.tp2.is_none());
    assert_eq!(placement.expert_ranges[0].layers, 0);
    let range = &placement.expert_ranges[1];
    assert_eq!((range.first, range.layers, range.peak_bytes), (1, 1, 118));
    assert_eq!(placement.peer_row_bytes, 0);
    assert_eq!(working.rtx_layers, vec![1]);
    let spec = HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2,
        rows: resolved.decisions.prefill.rows.max(64), lanes: resolved.decisions.prefill.waves,
        entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::Replicated };
    assert_eq!(placement.residual, residual_homes(&recipe.attention, &spec).unwrap());
    assert_eq!(placement.hops, plan_hops(&recipe.attention, &spec).unwrap());
}

#[test]
fn execution_adapter_refuses_nonprefix_sparks_without_renumbering_or_clamping() {
    let cfg = cfg();
    let recipe = recipe(&cfg);
    let resolved = planner::solve(Request { onboard: Pin::Exact(0), ..request() }, budgets(), &recipe).unwrap();
    let mut nonprefix = resolved.clone();
    nonprefix.decisions.expert_homes.insert(1, ExpertDestination::Sparks { ranks: vec![2, 3] });
    let error = recipe.execution(&nonprefix, &Default::default()).unwrap_err();
    assert!(matches!(error, Refusal::InvalidFact(ref detail)
        if detail.contains("non-prefix rank sets cannot be renumbered")));
    let mut wrong_prefix = resolved.clone();
    wrong_prefix.decisions.expert_homes.insert(1, ExpertDestination::Coordinator { gpus: vec![0, 1] });
    assert!(matches!(recipe.execution(&wrong_prefix, &Default::default()), Err(Refusal::InvalidFact(_))));
    assert!(matches!(planner::solve(Request { onboard: Pin::Exact(2), ..request() }, budgets(), &recipe),
        Err(Refusal::PinConflict)));
}

#[test]
fn phase_reply_resolution_is_immutable_and_absent_without_links() {
    use cuteafd_core::SparkReplyFormat::{Bf16, Fp8, Nvfp4};
    let cfg = cfg();
    for links in [false, true] {
        let mut recipe = recipe(&cfg);
        recipe.spark_links = links;
        if !links {
            recipe.experts = None;
            recipe.owners.clear();
            recipe.targets.clear();
            recipe.hooks.clear();
        }
        for prefill in [None, Some(Bf16), Some(Fp8), Some(Nvfp4)] {
            for decode in [None, Some(Bf16), Some(Fp8), Some(Nvfp4)] {
                let req = Request {
                    spark_reply_formats: SparkReplyFormats { prefill, decode },
                    ..request()
                };
                let contract = recipe.prepare(&req).unwrap();
                let expected = links.then_some(SparkReplyFormats {
                    prefill: Some(prefill.unwrap_or(Bf16)),
                    decode: Some(decode.unwrap_or(Bf16)),
                });
                assert_eq!(contract.spark_reply_formats, expected);
                assert_eq!(
                    contract
                        .decisions(&vec![0; contract.variables.len()])
                        .unwrap()
                        .spark_reply_formats,
                    expected
                );
                let plan = planner::solve(req, budgets(), &recipe);
                if !links && (prefill.is_some() || decode.is_some()) {
                    assert!(matches!(plan, Err(Refusal::PinConflict)));
                } else {
                    assert_eq!(plan.unwrap().decisions.spark_reply_formats, expected);
                }
            }
        }
    }
}

#[test]
fn warmup_upload_is_one_exact_successor_per_selected_shape_not_per_rank() {
    let cfg = cfg();
    for gpus in [1, 2] {
        let mut recipe = recipe(&cfg);
        recipe.gpus = gpus;
        recipe.attention = vec![
            LayerMode::Whole {
                gpu: 0,
                ffn: FfnMode::Split
            };
            cfg.layers
        ];
        recipe.experts = None;
        recipe.targets.clear();
        recipe.hooks.clear();
        recipe.exchanges.clear();
        recipe.weights.clear();
        recipe.owners.clear();
        let contract = recipe.prepare(&request()).unwrap();
        for (selected, shape) in recipe.prefill.iter().enumerate() {
            let mut choices = vec![0; contract.variables.len()];
            choices[recipe.attention.len()] = selected;
            let facts = contract.facts(&choices, 32).unwrap();
            assert_eq!(facts.allocations.len(), 1);
            let owner = &facts.allocations[0];
            assert_eq!(owner.id, "glm.upload.sync-successor");
            assert_eq!(owner.bytes, (shape.rows * shape.waves).max(64) * 4);
            assert_eq!(owner.location, Location::PINNED);
            assert_eq!(
                owner.lifetime,
                Lifetime {
                    first: Phase::Warmup,
                    last: Phase::Growth
                }
            );
            assert!(facts.reservations.is_empty());
            let phases = planner::evaluate(&facts, &budgets()).unwrap();
            for row in phases.iter().filter(|p| p.budget == "physical.2") {
                assert_eq!(
                    row.owned,
                    if matches!(row.phase, Phase::Warmup | Phase::Ready | Phase::Growth) {
                        owner.bytes
                    } else {
                        0
                    }
                );
            }
        }
    }
}

#[test]
fn warmup_upload_requires_an_independent_sufficient_host_budget() {
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    recipe.embedding = vec![EmbeddingHome::Device(0)];
    recipe.experts = None;
    recipe.targets.clear();
    recipe.hooks.clear();
    recipe.exchanges.clear();
    recipe.weights.clear();
    recipe.owners.clear();
    let req = Request {
        prefill: Pin::Exact(recipe.prefill[0]),
        ..request()
    };
    let mut room = budgets();
    room.retain(|budget| !budget.locations.contains(&Location::PINNED));
    assert!(matches!(planner::solve(req.clone(), room, &recipe),
        Err(Refusal::InvalidFact(detail)) if detail == "component has no physical budget"));
    let required = super::super::buffers::sync_h2d_successor(recipe.prefill[0])
        .unwrap()
        .bytes;
    let mut room = budgets();
    room[2].physical_bytes = required - 1;
    room[2].logical_bytes = required - 1;
    assert!(matches!(planner::solve(req.clone(), room.clone(), &recipe),
        Err(Refusal::Capacity { budget, phase: Phase::Warmup, required: actual, available, .. })
            if budget == "physical.2" && actual == required && available == required - 1));
    room[2].physical_bytes = required;
    room[2].logical_bytes = required;
    let plan = planner::solve(req.clone(), room, &recipe).unwrap();
    assert_eq!(plan.request, req);
    assert_eq!(plan.pool_tokens, 4096);
}

#[test]
fn dflash_prefetch_refuses_missing_empty_nonfile_and_zero_pool_extent() {
    let root = tempfile::tempdir().unwrap();
    assert!(DflashPrefetch::from_snapshot(root.path()).is_err());
    let path = root.path().join("model.safetensors");
    std::fs::File::create(&path).unwrap();
    assert!(DflashPrefetch::from_snapshot(root.path()).is_err());
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    assert!(DflashPrefetch::from_snapshot(root.path()).is_err());
    std::fs::remove_dir(&path).unwrap();
    crate::plan::testing::write_safetensors(&path, &[("fc.weight".into(), "BF16", vec![128, 128])]);
    let prefetch = DflashPrefetch::from_snapshot(root.path()).unwrap();
    assert!(matches!(
        prefetch.components(0),
        Err(Refusal::InvalidFact(_))
    ));
    let components = prefetch.components(64).unwrap();
    assert_eq!(components.len(), 1);
    assert!(matches!(components[0].kind, ComponentKind::Allowance(_)));
    assert_eq!(
        components[0].bytes.at(1).unwrap(),
        std::fs::metadata(&path).unwrap().len()
    );
}

#[test]
fn serialized_heap_max_follows_weight_case_and_adds_concurrent_prefetch() {
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    for (index, case) in recipe.weights.iter_mut().enumerate() {
        case.value.core_loading = super::super::weights::CoreLoading {
            heap_payload_bytes: 4096 * (index as u64 + 1),
            sync_upload_bytes: 1024,
        };
        case.value.kv_b_loading = super::super::weights::KvBLoading {
            heap_payload_bytes: 6144,
            sync_upload_bytes: 512,
        };
        case.value.row_loading = super::super::weights::RowLoading {
            heap_payload_bytes: 7168,
            sync_upload_bytes: 256,
        };
    }
    // The current loader reads the whole sparse file, including header and any
    // unused tensor payload. This is not an installed checkpoint measurement.
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("model.safetensors");
    crate::plan::testing::write_safetensors(
        &path,
        &[
            ("fc.weight".into(), "BF16", vec![128, 128]),
            ("unused.weight".into(), "BF16", vec![1024, 128]),
        ],
    );
    let file_bytes = std::fs::metadata(&path).unwrap().len();
    assert!(file_bytes > (128 * 128 + 1024 * 128) * 2);
    recipe.owners.push(Case {
        when: vec![],
        value: DflashPrefetch::from_snapshot(root.path())
            .unwrap()
            .components(recipe.maximum_pages)
            .unwrap(),
    });
    let contract = recipe.prepare(&request()).unwrap();
    for embedding in 0..2 {
        let facts = contract.facts(&[0, 0, 0, 0, embedding, 0, 0], 32).unwrap();
        assert_eq!(facts.allowances.len(), 2);
        assert!(
            facts
                .allowances
                .iter()
                .all(|a| a.location == Location::HOST && a.assumed)
        );
        let budgets = budgets();
        let phases = planner::evaluate(&facts, &budgets).unwrap();
        let opaque = |phase| {
            phases
                .iter()
                .find(|p| p.budget == "physical.2" && p.phase == phase)
                .unwrap()
                .opaque
        };
        assert_eq!(
            opaque(Phase::Load),
            file_bytes + (4096 * (embedding as u64 + 1)).max(7168)
        );
        assert_eq!(opaque(Phase::DraftLoad), file_bytes);
        assert!(facts.allowances.iter().any(|a| a.bytes == file_bytes
            && a.source.contains("metadata.len()")
            && a.source.contains("unmeasured")
            && a.source.contains("excludes allocator/tensor-map metadata")));
        assert_eq!(opaque(Phase::PostTargetKv), 0);
        assert_eq!(opaque(Phase::Warmup), 0);
        // The later successor is distinct from the still-uncomposed load maximum.
        let uploads = facts
            .allocations
            .iter()
            .filter(|a| a.id.contains("upload.sync"))
            .collect::<Vec<_>>();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].lifetime.first, Phase::Warmup);
        let peak = phases
            .iter()
            .find(|p| p.budget == "physical.2" && p.phase == Phase::Load)
            .unwrap();
        let mut tight = budgets;
        tight[2].physical_bytes = peak.owned + peak.opaque - 1;
        tight[2].logical_bytes = tight[2].physical_bytes;
        assert!(matches!(
            planner::evaluate(&facts, &tight),
            Err(Refusal::Capacity { .. })
        ));
    }
}

#[test]
fn one_contract_activates_physical_owners_by_shape_home_and_unique_target_count() {
    let cfg = cfg();
    let recipe = recipe(&cfg);
    let contract = recipe.prepare(&request()).unwrap();
    for prefill in 0..2 {
        for embedding in 0..2 {
            for onboard in 0..2 {
                let selected = [0, 0, prefill, 0, embedding, onboard, 0];
                let facts = contract.facts(&selected, 32).unwrap();
                planner::evaluate(&facts, &budgets()).unwrap();
                let decisions = contract.decisions(&selected).unwrap();
                assert_eq!(decisions.prefill, recipe.prefill[prefill]);
                assert_eq!(decisions.embedding, Some(recipe.embedding[embedding]));
                assert_eq!(
                    decisions.onboard_layers,
                    if onboard == 0 {
                        vec![vec![], vec![]]
                    } else {
                        vec![vec![1], vec![1]]
                    }
                );
                let has = |name: &str| facts.allocations.iter().any(|owner| owner.id == name);
                assert_eq!(has("expert.local.0.1"), onboard != 0);
                assert_eq!(has("expert.local.1.1"), onboard != 0);
                assert_eq!(has("expert.spark.0.1"), onboard == 0);
                assert_eq!(has("expert.spark.1.1"), onboard == 0);
                let owner = |name: &str| facts.allocations.iter().find(|a| a.id == name).unwrap();
                assert_eq!(
                    owner("glm.weights.embedding").location,
                    recipe.embedding[embedding].location()
                );
                assert_eq!(
                    owner("glm.target.gpu.0.kv.0").bytes,
                    searched(33 * super::super::RECORD_PAGE_BYTES)
                );
                assert_eq!(
                    owner("glm.target.gpu.0.prefill.rows.h").bytes,
                    recipe.prefill[prefill].rows
                        * recipe.prefill[prefill].waves
                        * cfg.hidden as u64
                        * 2
                );
                assert_eq!(has("glm.target.gpu.0.prefill.rows.payload"), onboard != 0);
                assert_eq!(has("glm.target.gpu.1.prefill.slot.2.h"), prefill == 1);
            }
        }
    }
    // The common solver, not a family count loop, enforces the exact pins.
    let mut exact = request();
    exact.onboard = Pin::Exact(1);
    exact.prefill = Pin::Exact(recipe.prefill[0]);
    let plan = planner::solve(exact.clone(), budgets(), &recipe).unwrap();
    assert_eq!(plan.decisions.onboard_layers, vec![vec![1], vec![1]]);
    exact.onboard = Pin::Exact(2);
    assert!(matches!(
        planner::solve(exact, budgets(), &recipe),
        Err(Refusal::PinConflict)
    ));
}

#[test]
fn shared_serving_owners_compose_without_family_private_geometry_or_policy() {
    use planner::serving::{HostPrefixOwners, SelectorOwners, ServingOwners};
    let cfg = cfg();
    for device_sampling in [false, true] {
        let mut recipe = recipe(&cfg);
        let serving = ServingOwners {
            selector: SelectorOwners {
                location: Location::coordinator(0),
                capacity: super::super::DECODE_ROWS as usize,
                vocab: cfg.vocab_size,
                device_sampling,
            },
            prefix: HostPrefixOwners {
                quota_bytes: 2 * 4096 + 17,
                chunk_bytes: 4096,
            },
        };
        let expected = serving.allocations().unwrap();
        recipe.owners.push(Case {
            when: vec![],
            value: serving.components(recipe.maximum_pages).unwrap(),
        });
        let contract = recipe.prepare(&request()).unwrap();
        for prefill in 0..2 {
            for embedding in 0..2 {
                for onboard in 0..2 {
                    let selected = [0, 0, prefill, 0, embedding, onboard, 0];
                    let facts = contract.facts(&selected, 32).unwrap();
                    planner::evaluate(&facts, &budgets()).unwrap();
                    assert_eq!(
                        facts
                            .allocations
                            .iter()
                            .filter(|a| a.id.starts_with("selector.")
                                || a.id.starts_with("host.prefix."))
                            .cloned()
                            .collect::<Vec<_>>(),
                        expected
                    );
                    assert!(!facts.allocations.iter().any(|a| a.id == "host.prefix.2"));
                }
            }
        }
    }
}

fn graph_layers(owners: &[u8], tp2: &[bool]) -> Vec<GlmGraphLayer> {
    owners.iter().zip(tp2).enumerate().map(|(index, (&owner, &tp2))| GlmGraphLayer {
        dense: index == 0, local: tp2, attention_owner: Some(owner),
    }).collect()
}

#[test]
fn graph_figures_are_keyed_by_the_measured_fingerprint_and_estimate_other_layouts() {
    disallow_synthetic_graphs();
    let min_layers: Vec<_> = (0..78).map(|layer| GlmGraphLayer {
        dense: layer < 3, local: false, attention_owner: None,
    }).collect();
    let mut max_layers = min_layers.clone();
    for layer in &mut max_layers[3..9] { layer.local = true; }
    let min = StartupGraphs::new(1 << 20, AttentionPlacement::Heads, 1, &min_layers, 4).unwrap();
    let max = StartupGraphs::new(1 << 20, AttentionPlacement::Context, 2, &max_layers, 6).unwrap();
    assert_eq!(min.declared()[0].fingerprint(0).unwrap(), MIN_GRAPHS);
    assert_eq!(max.declared()[0].fingerprint(0).unwrap(), MAX_GRAPHS[0]);
    assert_eq!(max.declared()[1].fingerprint(1).unwrap(), MAX_GRAPHS[1]);
    assert_eq!(min.ranks()[0].bytes, 2_984_247_296);
    assert_eq!(max.ranks().iter().map(|rank| rank.bytes).collect::<Vec<_>>(),
        [3_042_967_552, 2_971_664_384]);
    for (rank, graph) in min.ranks().iter().enumerate() {
        assert_eq!(graph.margin, 0);
        assert_eq!(graph.basis, Basis::Calibrated);
        assert_eq!(graph.executables as usize, min.declared()[rank].len());
    }
    assert_eq!(min.figures()[0].attribution, FigureAttribution::Measured);
    assert_eq!(min.figures()[0].identity.declared_graphs, MIN_GRAPHS);
    for (rank, component) in max.components(32768).unwrap().iter().enumerate() {
        let ComponentKind::Allowance(allowance) = &component.kind else {
            panic!("graphs are opaque, not physical allocations")
        };
        assert_eq!(allowance.location, Location::coordinator(rank as u8));
        assert_eq!(allowance.lifetime, Lifetime { first: Phase::Warmup, last: Phase::Growth });
        assert_eq!(allowance.bytes, max.ranks()[rank].bytes);
        assert!(!allowance.assumed, "a measured row is never labelled assumed");
        assert!(allowance.source.contains("startup graphs"));
    }
    // The declared key set, not the context alone, is the identity. Each of
    // these produces a fingerprint with no measured row; #1605 serves it on the
    // best measured row for its rank (the default-context row upper-bounds a
    // lower context), labelled Estimated with one warning, and never scales or
    // refuses.
    let mut owned = max_layers.clone();
    for (layer, spec) in owned.iter_mut().enumerate() { spec.attention_owner = Some((layer % 2) as u8); }
    let estimated = [
        StartupGraphs::new(1 << 19, AttentionPlacement::Heads, 1, &min_layers, 4),
        StartupGraphs::new(1 << 20, AttentionPlacement::Heads, 2, &max_layers, 6),
        StartupGraphs::new(1 << 20, AttentionPlacement::Layers, 2, &owned, 6),
        StartupGraphs::new(1 << 20, AttentionPlacement::Heads, 1, &min_layers, 8),
        StartupGraphs::new(1 << 20, AttentionPlacement::Heads, 1, &min_layers[..77], 4),
    ];
    for graphs in estimated {
        let graphs = graphs.expect("an unmeasured declared graph set is served, not refused");
        for (rank, graph) in graphs.ranks().iter().enumerate() {
            assert_eq!(graph.basis, Basis::Estimated);
            assert_eq!(graph.margin, 0);
            // Same rank first, identically-keyed first within it.
            let same_rank = super::MEASURED_GLM_GRAPHS.iter().filter(|row| row.rank == rank as u8);
            let expected = same_rank.clone().filter(|row| row.keys == graph.executables as usize)
                .map(|row| row.bytes).max()
                .or_else(|| same_rank.map(|row| row.bytes).max())
                .expect("the family has a measured row for every rank");
            assert_eq!(graph.bytes, expected);
            let figure = &graphs.figures()[rank];
            assert!(matches!(figure.attribution, FigureAttribution::Estimated { .. }));
            // The figure keeps the provenance of the row it borrowed.
            assert!(!figure.identity.declared_graphs.is_empty());
            assert_eq!(figure.bytes, expected);
        }
        for component in graphs.components(32768).unwrap() {
            let ComponentKind::Allowance(allowance) = &component.kind else {
                panic!("graphs are opaque, not physical allocations")
            };
            assert!(allowance.assumed, "an estimated graph row is labelled assumed");
        }
    }
    assert!(StartupGraphs::new(0, AttentionPlacement::Heads, 1, &min_layers, 4).is_err());
    assert!(StartupGraphs::new(1 << 20, AttentionPlacement::Heads, 1, &min_layers, 4)
        .unwrap().components(0).is_err());
}

#[test]
fn a_live_capture_replaces_the_row_only_at_the_row_fingerprint() {
    let min_layers: Vec<_> = (0..78).map(|layer| GlmGraphLayer {
        dense: layer < 3, local: false, attention_owner: None,
    }).collect();
    let min = StartupGraphs::new(1 << 20, AttentionPlacement::Heads, 1, &min_layers, 4).unwrap();
    let set = min.set();
    assert_eq!(set.lifetime, crate::placement::Lifetime::Startup);
    assert_eq!(set.shapes, glm_graph_shapes(1 << 20, (1 << 20) / PAGE_ROWS as usize).len() as u64);
    assert_eq!(set.ranks, min.ranks());
    assert_eq!(min.capture_note(&set), None);
    let mut moved = set.clone();
    moved.ranks[0].bytes += 2 << 20;
    let note = min.capture_note(&moved).unwrap();
    assert!(note.contains("update MEASURED_GLM_GRAPHS"), "{note}");
    assert!(note.contains(&format!("rank 0: row {} -> captured {}", set.ranks[0].bytes, moved.ranks[0].bytes)));
}

#[test]
fn stream_inventory_counts_long_lived_serving_streams_and_shared_delivery() {
    use crate::placement::planner::cuda_stream_slab_bytes;
    let copies = crate::push_load::COPY_STREAMS as u64;
    // Engine + prefix copy + delivery copies are the floor of every layout.
    assert_eq!(startup_peak_live_streams(0, false, false), 2 + copies);
    assert_eq!(startup_peak_live_streams(1, false, false), 2 + copies);
    // The Spark intakes' copy streams live on the rank that owns their planes.
    assert_eq!(startup_peak_live_streams(0, true, false), 4 + copies);
    assert_eq!(startup_peak_live_streams(1, true, false), 2 + copies);
    // The host-prefix copy engine is per rank; the head split never admits it.
    assert_eq!(startup_peak_live_streams(0, true, true), 6 + copies);
    assert_eq!(startup_peak_live_streams(1, true, true), 4 + copies);
    // One-coordinator serving with Sparks and the host tier needs three slabs;
    // a head split's ranks need two each.
    assert_eq!(cuda_stream_slab_bytes(startup_peak_live_streams(0, true, true)).unwrap(), 3 << 21);
    assert_eq!(cuda_stream_slab_bytes(startup_peak_live_streams(0, true, false)).unwrap(), 2 << 21);
    assert_eq!(cuda_stream_slab_bytes(startup_peak_live_streams(1, true, false)).unwrap(), 2 << 21);
}

#[test]
fn device_stage_rows_partition_each_reference_census_delta_exactly_once() {
    // #1242 raw untracked samples per phase transition. Each row is the
    // difference between adjacent samples (with the measured instantiate sum
    // removed from the final sample). The pool's cudaMalloc backing is no
    // longer a stage row: the selected pool owners carry the sizes-derived
    // SM120 granularity tail instead, so it moves with the pool extent.
    let cases: [(usize, usize, u64, &[(&str, u64)]); 3] = [
        (1, 0, 598_999_040, &[
            ("target weight load device staging", 143_391_104),
            ("drafter load device staging", 123_296_064),
            ("transport device staging", 90_144_768),
            ("startup prewarm module first touch", 72_615_168),
            ("first ready-step CUDA module touch", 69_045_248)]),
        (2, 0, 594_804_736, &[
            ("target weight load device staging", 269_002_112),
            ("drafter load device staging", 77_117_248),
            ("transport device staging", 169_868_800),
            ("first ready-step CUDA module touch", 69_045_248)]),
        (2, 1, 603_193_344, &[
            ("target weight load device staging", 140_074_880),
            ("transport device staging", 27_965_952),
            ("first ready-step CUDA module touch", 71_303_168)]),
    ];
    for (gpus, rank, context, rows) in cases {
        let figures = DeviceFigures::new(gpus, false, false).unwrap();
        assert_eq!(figures.context()[rank].bytes, context);
        assert_eq!(figures.context()[rank].attribution, FigureAttribution::Measured);
        let actual: Vec<_> = figures.stages().iter().filter(|row| row.rank == rank)
            .map(|row| (row.label, row.bytes)).collect();
        assert_eq!(actual, rows);
        assert_eq!(figures.components(32768).unwrap().len(), 2 * gpus + figures.stages().len());
        for component in figures.components(32768).unwrap() {
            let ComponentKind::Allowance(owner) = &component.kind else {
                panic!("a census stage is not a physical allocation")
            };
            assert!(!owner.assumed);
            assert!(owner.source.contains("census") || owner.source.contains("GLM context")
                || owner.source.contains("stream-slabs"));
            assert_eq!(component.bytes.at(1).unwrap(), owner.bytes);
        }
        let phase_order = PhaseOrder::DRAFT_THEN_TARGET;
        for (index, row) in figures.stages().iter().enumerate() {
            let ComponentKind::Allowance(owner) = &figures.components(32768).unwrap()[2 * gpus + index].kind else {
                panic!("stage rows stay allowance owners")
            };
            assert_eq!(owner.bytes, row.bytes);
            assert_eq!(owner.location, Location::coordinator(row.rank as u8));
            assert_eq!(owner.lifetime, Lifetime { first: row.first, last: row.last });
            assert!(owner.lifetime.contains(row.first, phase_order) && owner.lifetime.contains(row.last, phase_order));
        }
        // Stream-slot slabs are their own owner and come out of the context row.
        let slabs = figures.slabs();
        for rank in 0..gpus {
            let peak = startup_peak_live_streams(rank, false, false);
            assert_eq!(slabs[rank], crate::placement::planner::cuda_stream_slab_bytes(peak).unwrap());
            let ComponentKind::Allowance(owner) = &figures.components(32768).unwrap()[gpus + rank].kind else {
                panic!("stream slabs stay an allowance owner")
            };
            assert_eq!(owner.bytes, slabs[rank]);
            assert_eq!(owner.lifetime, Lifetime::RESIDENT);
            assert!(!owner.assumed);
            assert!(owner.source.contains(&format!("{peak} peak live streams")));
            let ComponentKind::Allowance(context) = &figures.components(32768).unwrap()[rank].kind else {
                panic!("the context row stays an allowance owner")
            };
            assert_eq!(context.bytes, figures.context()[rank].bytes - slabs[rank]);
        }
    }
    for gpus in [0, 3, usize::MAX] { assert!(DeviceFigures::new(gpus, false, false).is_err()); }
    assert!(DeviceFigures::new(1, false, false).unwrap().components(0).is_err());
}

#[test]
fn host_figures_keep_reference_receipts_and_estimate_uncovered_layouts_without_scaling() {
    use crate::placement::inventory::{FigureUse, census_recommendation};
    for (gpus, layout, load, steady) in [
        (1, MIN_LAYOUT, GLM_MIN_HOST_LOAD, GLM_MIN_HOST_STEADY),
        (2, MAX_LAYOUT, GLM_MAX_HOST_LOAD, GLM_MAX_HOST_STEADY),
    ] {
        let figures = HostFigures::for_layout(gpus, layout).unwrap();
        assert_eq!(figures.load(), load);
        assert_eq!(figures.steady(), steady);
        assert_eq!(load.attribution, FigureAttribution::Measured);
        let recommendation = census_recommendation([
            FigureUse { name: "host load", figure: load, current: load.identity },
            FigureUse { name: "host steady", figure: steady, current: steady.identity },
        ]);
        if gpus == 1 {
            assert_eq!(load.bytes, 5_759_569_920);
            assert_eq!(steady.bytes, 40_403_427_328 - 36_902_883_328);
            let FigureAttribution::Estimated { derivation } = steady.attribution else {
                panic!("incomplete min Ready must never become a measured receipt")
            };
            assert!(derivation.contains("ordinary_rss_bytes missing and complete=false"));
            let recommendation = recommendation.unwrap();
            assert!(recommendation.contains("census recommended"));
            assert!(recommendation.contains("host steady"));
            assert!(!recommendation.contains("host load"));
        } else {
            assert_eq!(load.bytes, 8_439_906_304);
            assert_eq!(steady.bytes, 8_439_906_304);
            assert_eq!(steady.attribution, FigureAttribution::Measured);
            assert!(recommendation.is_none());
        }
        for sparks in [0, 1, 4, 6, 8] {
            let uncovered = HostFigures::for_layout(gpus, &format!("other layout; sparks/{sparks}")).unwrap();
            for (source, derived) in [(load, uncovered.load()), (steady, uncovered.steady())] {
                assert_eq!(derived.bytes, source.bytes);
                assert_eq!(derived.evidence, source.evidence);
                assert_eq!(derived.identity, source.identity);
                assert_eq!(derived.producing_commit, source.producing_commit);
                let FigureAttribution::Estimated { derivation } = derived.attribution else {
                    panic!("an uncovered layout is estimated")
                };
                assert!(derivation.contains("no Spark-count scaling"));
                assert!(derivation.contains(if gpus == 1 {
                    "glm53-exl3-min-census-r1"
                } else { "glm53-exl3-max-census-r1" }));
                assert!(derivation.contains("#1242 retained coordinator log"));
            }
        }
        assert!(figures.components(0).is_err());
    }
    for gpus in [0, 3, usize::MAX] {
        assert!(HostFigures::for_layout(gpus, MIN_LAYOUT).is_err());
    }
    assert!(HostFigures::for_layout(1, " ").is_err());
}

#[test]
fn host_envelopes_cover_every_startup_phase_exactly_once_including_warmup() {
    let order = PhaseOrder::DRAFT_THEN_TARGET;
    let phases = order.phases();
    let ready = phases.iter().position(|phase| *phase == Phase::Ready).unwrap();
    assert_eq!(phases[ready - 1], Phase::Warmup);
    for (gpus, layout) in [(1, MIN_LAYOUT), (2, MAX_LAYOUT)] {
        let figures = HostFigures::for_layout(gpus, layout).unwrap();
        for maximum in [1, 32768] {
            let components = figures.components(maximum).unwrap();
            assert_eq!(components.len(), 2, "process RSS is never multiplied by GPU count");
            for &phase in phases {
                let active: Vec<_> = components.iter().filter_map(|component| {
                    assert_eq!(component.activation, Activation::always());
                    let ComponentKind::Allowance(allowance) = &component.kind else {
                        panic!("RSS is opaque, not an invented allocation owner")
                    };
                    assert_eq!(allowance.location, Location::HOST);
                    assert_eq!(component.bytes.at(1).unwrap(), allowance.bytes);
                    assert_eq!(component.bytes.at(maximum).unwrap(), allowance.bytes);
                    allowance.lifetime.contains(phase, order).then_some(allowance)
                }).collect();
                assert_eq!(active.len(), 1, "{gpus} GPUs, phase {phase:?}");
                let expected = if order.at_or_before(phase, Phase::Warmup) {
                    figures.load()
                } else { figures.steady() };
                assert_eq!(active[0].bytes, expected.bytes);
                assert_eq!(active[0].assumed,
                    matches!(expected.attribution, FigureAttribution::Estimated { .. }));
            }
        }
    }
}

#[test]
fn host_figures_subsume_only_ordinary_heap_and_preserve_pinned_device_and_weight_facts() {
    let cfg = cfg();
    let mut without = recipe(&cfg);
    for case in &mut without.weights {
        case.value.core_loading.heap_payload_bytes = 54321;
    }
    let maximum = without.maximum_pages;
    fn owner(id: impl Into<String>, location: Location, bytes: u64, lifetime: Lifetime) -> Allocation {
        Allocation::buffer(id, location, bytes, lifetime, Category::Staging, "domain preservation")
    }
    let heap = owner("test.host.retained", Location::HOST, 512, Lifetime::RESIDENT);
    let mut components = DflashPrefetch { file_bytes: 98765 }.components(maximum).unwrap();
    for (id, location, bytes, deferred) in [
        ("test.host.retained", Location::HOST, 512, false),
        ("test.host.deferred", Location::HOST, 256, true),
        ("glm.media.cache", Location::HOST, 4096, false),
        ("test.pinned.retained", Location::PINNED, 1024, false),
        ("test.device.retained", Location::coordinator(1), 2048, false),
    ] {
        components.push(Component::allocation(owner(id, location, bytes, Lifetime::RESIDENT),
            ByteCurve::fixed(bytes, maximum), deferred, maximum));
    }
    components.push(Component::resident_weight(
        ResidentWeight::from_allocation(&heap, crate::placement::ResidentWeights(heap.bytes)),
        ByteCurve::fixed(heap.bytes, maximum)));
    for (name, location) in [("host", Location::HOST), ("pinned", Location::PINNED),
        ("device", Location::coordinator(0))] {
        let group = Sequential { id: format!("test.{name}.sequential"),
            members: [64, 128].into_iter().enumerate().map(|(index, bytes)| owner(
                format!("test.{name}.sequential.{index}"), location, bytes,
                Lifetime { first: Phase::Load, last: Phase::Load })).collect() };
        components.push(Component { activation: Activation::always(),
            bytes: ByteCurve::fixed(group.peak(), maximum), kind: ComponentKind::Sequential(group) });
    }
    components.push(Component { activation: Activation::always(),
        bytes: ByteCurve::fixed(4096, maximum), kind: ComponentKind::Reservation(Reservation {
            id: "test.pinned.load.reserve".into(), location: Location::PINNED, bytes: 4096,
            lifetime: Lifetime { first: Phase::Load, last: Phase::Load },
            source: "synthetic bounded pinned load pool".into(),
        }) });
    without.owners.push(Case { when: vec![], value: components });
    let mut with = without.clone();
    let host = HostFigures::for_layout(2, MAX_LAYOUT).unwrap();
    with.host = Some(host.clone());
    for shape in without.prefill.clone() {
        for count in [0, 1] {
            let request = Request { onboard: Pin::Exact(count), prefill: Pin::Exact(shape), ..request() };
            let before = planner::solve(request.clone(), budgets(), &without).unwrap();
            let after = planner::solve(request, budgets(), &with).unwrap();
            assert_eq!(after.decisions, before.decisions);
            assert_eq!(after.pool_tokens, before.pool_tokens);
            assert_eq!(after.facts.resident_weights, before.facts.resident_weights);
            assert!(after.facts.resident_weights.iter().any(|fact| fact.id == heap.id));
            let nonhost = |facts: &Facts| {
                let mut facts = facts.clone();
                facts.allocations.retain(|a| a.location != Location::HOST);
                facts.deferred.retain(|a| a.location != Location::HOST);
                facts.sequential.retain(|g| g.members[0].location != Location::HOST);
                facts.allowances.retain(|a| a.location != Location::HOST);
                facts
            };
            assert_eq!(nonhost(&after.facts), nonhost(&before.facts),
                "every PINNED/device owner, view, reservation and byte total is unchanged");
            assert!(before.facts.allowances.iter().any(|a|
                a.source.contains("serialized target payload heap subset")));
            assert!(before.facts.allowances.iter().any(|a| a.source.contains("DFlash whole-file")));
            let retained_host = after.facts.allocations.iter().chain(&after.facts.deferred)
                .filter(|a| a.location == Location::HOST).collect::<Vec<_>>();
            assert_eq!(retained_host.len(), 1);
            assert_eq!((retained_host[0].id.as_str(), retained_host[0].bytes), ("glm.media.cache", 4096),
                "future media capacity is not covered by the text-only RSS census");
            assert!(after.facts.sequential.iter().all(|g| g.members[0].location != Location::HOST));
            assert!(after.facts.reservations.iter().all(|r| r.location != Location::HOST));
            let allowances: Vec<_> = after.facts.allowances.iter()
                .filter(|a| a.location == Location::HOST).collect();
            assert_eq!(allowances.len(), 2);
            assert_eq!(allowances.iter().map(|a| a.bytes).sum::<u64>(),
                host.load().bytes + host.steady().bytes);
            for phase in after.facts.phase_order.phases() {
                let active: Vec<_> = allowances.iter().filter(|a|
                    a.lifetime.contains(*phase, after.facts.phase_order)).collect();
                assert_eq!(active.len(), 1);
                let expected = if after.facts.phase_order.at_or_before(*phase, Phase::Warmup) {
                    host.load().bytes
                } else { host.steady().bytes };
                assert_eq!(active[0].bytes, expected);
            }
        }
    }
    with.host = Some(HostFigures::for_layout(1, MIN_LAYOUT).unwrap());
    assert!(matches!(with.prepare(&request()), Err(Refusal::InvalidFact(_))));
}

#[test]
fn fixed_prefix_quota_uses_joint_host_ram_after_steady_and_pinned_owners() {
    use crate::placement::planner::serving::{HostPrefixOwners, ServingOwners};
    use cuteafd_core::prefix::HostCacheGeometry;

    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    let host = HostFigures::for_layout(2, MAX_LAYOUT).unwrap();
    recipe.host = Some(host.clone());
    // Fix embedding placement so the shortage cannot move its backing to a GPU.
    recipe.embedding = vec![EmbeddingHome::HostPinned];
    recipe.weights.retain(|case|
        matches!(case.when.as_slice(), [When::Embedding(EmbeddingHome::HostPinned)]));
    let request = Request { onboard: Pin::Exact(0), prefill: Pin::Exact(recipe.prefill[0]),
        ..request() };
    let without_prefix = planner::solve(request.clone(), budgets(), &recipe).unwrap();
    let steady = without_prefix.phases.iter().find(|phase|
        phase.budget == "physical.2" && phase.phase == Phase::Ready).unwrap();
    assert_eq!(steady.opaque, host.steady().bytes);
    let pinned: u64 = without_prefix.facts.allocations.iter().filter(|owner|
        owner.location == Location::PINNED && owner.view.is_none()
            && owner.lifetime.contains(Phase::Ready, without_prefix.facts.phase_order))
        .map(|owner| owner.bytes).sum();
    assert!(pinned > 0);
    assert_eq!(steady.owned, pinned);
    // Serving prefill lanes allocate their pinned H2B staging on the first live
    // prefill, after Ready, so the host budget carries them at Growth even
    // though the Ready phase does not.
    let deferred: u64 = without_prefix.facts.allocations.iter().filter(|owner|
        owner.location == Location::PINNED && owner.view.is_none()
            && !owner.lifetime.contains(Phase::Ready, without_prefix.facts.phase_order)
            && owner.lifetime.contains(Phase::Growth, without_prefix.facts.phase_order))
        .map(|owner| owner.bytes).sum();
    assert!(deferred > 0);

    // Use the existing resolver, not a new cutover default or free-RAM clamp.
    let geometry = HostCacheGeometry { unit_rows: 64, page_bytes: 1 << 20,
        mark_bytes: 0, draft_bytes: 0 };
    let chunk = geometry.chunk_bytes();
    let host_total = 128 << 30;
    for explicit in [None, Some(40 << 30)] {
        let quota = geometry.resolve(explicit, 1 << 20, 16, host_total, chunk).unwrap();
        assert_eq!(quota, explicit.unwrap_or(32 << 30));
        let mut with_prefix = recipe.clone();
        let prefix = HostPrefixOwners { quota_bytes: quota, chunk_bytes: chunk }
            .allocations("glm.host.prefix", ServingOwners::LIFETIME).unwrap();
        assert_eq!(prefix.iter().map(|owner| owner.bytes).sum::<u64>(), quota);
        with_prefix.owners.push(Case { when: vec![], value: prefix.into_iter().map(|owner| {
            let bytes = ByteCurve::fixed(owner.bytes, with_prefix.maximum_pages);
            Component::allocation(owner, bytes, false, with_prefix.maximum_pages)
        }).collect() });
        let mut joint = budgets();
        let occupied = 128 << 20;
        let headroom = 256 << 20;
        let available = host.steady().bytes + pinned + deferred + quota;
        joint[2] = Budget { id: "physical.2".into(),
            locations: vec![Location::HOST, Location::PINNED],
            physical_bytes: available + occupied + headroom,
            logical_bytes: available + occupied + headroom,
            baseline: Baseline::Forecast { occupied, source: "external host consumers".into(),
                assumed: false }, headroom };
        assert_eq!(joint[2].available().unwrap() - host.steady().bytes - pinned,
            quota + deferred);
        let admitted = planner::solve(request.clone(), joint.clone(), &with_prefix).unwrap();
        let ready = admitted.phases.iter().find(|phase|
            phase.budget == "physical.2" && phase.phase == Phase::Ready).unwrap();
        assert_eq!(ready.opaque, host.steady().bytes);
        assert_eq!(ready.owned, pinned + quota);
        assert_eq!(admitted.pool_tokens, without_prefix.pool_tokens);
        assert_eq!(admitted.decisions, without_prefix.decisions);
        assert_eq!(admitted.facts.allocations.iter().filter(|owner|
            owner.id.starts_with("glm.host.prefix.")).map(|owner| owner.bytes).sum::<u64>(), quota);
        joint[2].logical_bytes -= 1;
        assert!(matches!(planner::solve(request.clone(), joint, &with_prefix),
            Err(Refusal::Capacity { budget, required, available: free, .. })
                if budget == "physical.2" && required == available && free == available - 1),
            "a fixed quota refuses instead of silently shrinking into remaining RAM");
    }
}

#[test]
fn auto_embedding_can_move_without_shrinking_fixed_prefix_quota_or_pool() {
    use crate::placement::planner::serving::{HostPrefixOwners, ServingOwners};
    use cuteafd_core::prefix::HostCacheGeometry;

    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    let host = HostFigures::for_layout(2, MAX_LAYOUT).unwrap();
    recipe.host = Some(host.clone());
    let request = Request { onboard: Pin::Exact(0), prefill: Pin::Exact(recipe.prefill[0]),
        ..request() };
    let without_prefix = planner::solve(request.clone(), budgets(), &recipe).unwrap();
    assert_eq!(without_prefix.decisions.embedding, Some(EmbeddingHome::HostPinned));
    let embedding = without_prefix.facts.allocations.iter()
        .find(|owner| owner.id == "glm.weights.embedding").unwrap();
    assert_eq!(embedding.location, Location::PINNED);
    assert_eq!(embedding.bytes, 8192);
    let pinned = without_prefix.phases.iter().find(|phase|
        phase.budget == "physical.2" && phase.phase == Phase::Ready).unwrap().owned;
    // Serving prefill lanes allocate their pinned H2B staging on the first live
    // prefill, after Ready, so the host budget carries them at Growth even
    // though the Ready phase does not.
    let deferred: u64 = without_prefix.facts.allocations.iter().filter(|owner|
        owner.location == Location::PINNED && owner.view.is_none()
            && !owner.lifetime.contains(Phase::Ready, without_prefix.facts.phase_order)
            && owner.lifetime.contains(Phase::Growth, without_prefix.facts.phase_order))
        .map(|owner| owner.bytes).sum();
    assert!(deferred > 0);
    let geometry = HostCacheGeometry { unit_rows: 64, page_bytes: 1 << 20,
        mark_bytes: 0, draft_bytes: 0 };
    let chunk = geometry.chunk_bytes();
    for explicit in [None, Some(40 << 30)] {
        let quota = geometry.resolve(explicit, 1 << 20, 16, 128 << 30, chunk).unwrap();
        let mut with_prefix = recipe.clone();
        let prefix = HostPrefixOwners { quota_bytes: quota, chunk_bytes: chunk }
            .allocations("glm.host.prefix", ServingOwners::LIFETIME).unwrap();
        with_prefix.owners.push(Case { when: vec![], value: prefix.into_iter().map(|owner| {
            let bytes = ByteCurve::fixed(owner.bytes, with_prefix.maximum_pages);
            Component::allocation(owner, bytes, false, with_prefix.maximum_pages)
        }).collect() });
        let available = host.steady().bytes + pinned + deferred + quota;
        let mut joint = budgets();
        joint[2].physical_bytes = available;
        joint[2].logical_bytes = available;
        let exact = planner::solve(request.clone(), joint.clone(), &with_prefix).unwrap();
        assert_eq!(exact.decisions, without_prefix.decisions);
        joint[2].logical_bytes -= 1;
        let moved = planner::solve(request.clone(), joint, &with_prefix).unwrap();
        assert_eq!(moved.decisions.embedding, Some(EmbeddingHome::Device(0)));
        let mut expected = exact.decisions.clone();
        expected.embedding = Some(EmbeddingHome::Device(0));
        assert_eq!(moved.decisions, expected);
        assert_eq!(moved.pool_tokens, without_prefix.pool_tokens);
        assert_eq!(moved.facts.allocations.iter().filter(|owner|
            owner.id.starts_with("glm.host.prefix.")).map(|owner| owner.bytes).sum::<u64>(), quota);
        let ready = moved.phases.iter().find(|phase|
            phase.budget == "physical.2" && phase.phase == Phase::Ready).unwrap();
        assert_eq!(ready.opaque, host.steady().bytes);
        assert_eq!(ready.owned, pinned + quota - embedding.bytes);
        // The unpinned host budget still carries the Growth-only prefill
        // staging, so Ready keeps exactly that much slack plus the 1-byte hole.
        assert_eq!(ready.available - ready.owned - ready.opaque,
            embedding.bytes - 1 + deferred);
        let relocated = moved.facts.allocations.iter()
            .find(|owner| owner.id == embedding.id).unwrap();
        assert_eq!(relocated.location, Location::coordinator(0));
        assert_eq!(relocated.bytes, embedding.bytes);
    }
}

fn owned_graph_recipe(cfg: &GlmDsaConfig, context: u64) -> GlmRecipe<'_> {
    let mut recipe = recipe(cfg);
    recipe.context = context;
    recipe.page_tokens = 64;
    recipe.maximum_pages = 32768;
    recipe.embedding = vec![EmbeddingHome::Device(0)];
    recipe.weights.retain(|case| {
        matches!(
            case.when.as_slice(),
            [When::Embedding(EmbeddingHome::Device(0))]
        )
    });
    recipe.attention = (0..cfg.layers)
        .map(|layer| LayerMode::Whole {
            gpu: (layer % 2) as u8,
            ffn: FfnMode::Split,
        })
        .collect();
    for case in &mut recipe.targets {
        let target = &mut case.value;
        target.context = context;
        target.page_tokens = recipe.page_tokens;
        target.maximum_pages = recipe.maximum_pages;
        target.partitioned = false;
        target.split_heads = false;
        target.owned_layers = (0..cfg.layers)
            .map(|layer| layer % 2 == usize::from(target.device))
            .collect();
    }
    for case in &mut recipe.exchanges {
        case.value.maximum_pages = recipe.maximum_pages;
        case.value.context = None;
    }
    for case in &mut recipe.owners {
        for component in &mut case.value {
            let bytes = component.bytes.at(1).unwrap();
            component.bytes = ByteCurve::fixed(bytes, recipe.maximum_pages);
            if let ComponentKind::Allocation { template, .. } = &component.kind {
                *component = Component::allocation(
                    template.clone(),
                    ByteCurve::fixed(bytes, recipe.maximum_pages),
                    false,
                    recipe.maximum_pages,
                );
            }
        }
    }
    // Synthetic expert owners remain fixed; target buffers and graph keys use
    // their real shared runtime geometry. This is not a complete card forecast.
    for count in 0..=1 {
        recipe.graphs.push(Case {
            when: vec![When::Onboard {
                first: count,
                last: count,
            }],
            value: StartupGraphs::new(context, AttentionPlacement::Layers, 2,
                &graph_layers(&[0, 1], &[false, count != 0]), 4).unwrap(),
        });
    }
    recipe
}

#[test]
fn full_context_graph_boundary_keeps_original_pool_domain_and_exact_pins() {
    let cfg = cfg();
    let context = 131136;
    let recipe = owned_graph_recipe(&cfg, context);
    let mut request = request();
    request.context = context.into();
    request.pool = Pin::Auto;
    request.onboard = Pin::Exact(0);
    request.prefill = Pin::Exact(recipe.prefill[0]);
    let mut contract = recipe.prepare(&request).unwrap();
    contract.add_driver_page_tables();
    let selected = [0, 0, 0, 0, 0, 0, 0];
    let wanted = context + 128 * 64;
    let facts = contract.facts(&selected, wanted / 64).unwrap();
    let mut limits = budgets();
    let phases = planner::evaluate(&facts, &limits).unwrap();
    for (rank, budget) in limits.iter_mut().take(2).enumerate() {
        budget.logical_bytes = phases
            .iter()
            .filter(|phase| phase.budget == budget.id)
            .map(|phase| phase.owned + phase.opaque)
            .max()
            .unwrap();
        assert!(budget.logical_bytes > 0, "rank {rank}");
    }
    let plan = planner::solve(request.clone(), limits.clone(), &recipe).unwrap();
    assert_eq!(plan.pool_tokens, wanted);
    assert_eq!(plan.request.context.maximum, context);
    assert_eq!(contract.maximum_pages, 32768);
    // A smaller context changes the declaration; it cannot make the full request fit.
    let smaller = StartupGraphs::new(context - 64, AttentionPlacement::Layers, 2,
        &graph_layers(&[0, 1], &[false, false]), 4).unwrap();
    assert_ne!(smaller.declared()[0].fingerprint(0).unwrap(),
        recipe.graphs[0].value.declared()[0].fingerprint(0).unwrap());
    request.pool = Pin::Exact(wanted);
    assert_eq!(
        planner::solve(request.clone(), limits.clone(), &recipe)
            .unwrap()
            .pool_tokens,
        wanted
    );
    request.pool = Pin::Exact(wanted + 64);
    assert!(matches!(
        planner::solve(request.clone(), limits.clone(), &recipe),
        Err(Refusal::Capacity { .. })
    ));
    request.pool = Pin::Exact(context - 64);
    assert!(matches!(
        planner::solve(request.clone(), limits.clone(), &recipe),
        Err(Refusal::PoolBelowContext { .. })
    ));
    let floor = contract.facts(&selected, context / 64).unwrap();
    let phases = planner::evaluate(&floor, &budgets()).unwrap();
    for budget in limits.iter_mut().take(2) {
        budget.logical_bytes = phases
            .iter()
            .filter(|phase| phase.budget == budget.id)
            .map(|phase| phase.owned + phase.opaque)
            .max()
            .unwrap()
            - 1;
    }
    request.pool = Pin::Auto;
    assert!(matches!(
        planner::solve(request.clone(), limits, &recipe),
        Err(Refusal::FullContextShortfall { .. })
    ));
    request.context.maximum -= 64;
    assert!(matches!(
        recipe.prepare(&request),
        Err(Refusal::InvalidFact(_))
    ));
}

#[test]
fn declared_graphs_must_match_every_activated_onboard_count_and_attention_owner() {
    let cfg = cfg();
    let mut recipe = owned_graph_recipe(&cfg, 4096);
    let request = request();
    recipe.prepare(&request).unwrap();
    recipe.graphs[0].when.clear();
    assert!(matches!(
        recipe.prepare(&request),
        Err(Refusal::InvalidFact(_))
    ));
    recipe.graphs[0].when = vec![When::Onboard { first: 0, last: 0 }];
    recipe.graphs[0].value = StartupGraphs::new(4096, AttentionPlacement::Layers, 2,
        &graph_layers(&[1, 0], &[false, false]), 4).unwrap();
    assert!(matches!(
        recipe.prepare(&request),
        Err(Refusal::InvalidFact(_))
    ));
    recipe.graphs[0].value = StartupGraphs::new(2048, AttentionPlacement::Layers, 2,
        &graph_layers(&[0, 1], &[false, false]), 4).unwrap();
    assert!(matches!(
        recipe.prepare(&request),
        Err(Refusal::InvalidFact(_))
    ));
}

#[test]
fn named_conditions_intersect_count_domains_and_reject_private_indexes() {
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    recipe.owners.push(Case {
        when: vec![When::Onboard { first: 0, last: 1 }, When::LocalLayer(1)],
        value: vec![],
    });
    recipe.prepare(&request()).unwrap();
    recipe
        .owners
        .last_mut()
        .unwrap()
        .when
        .push(When::RemoteLayer(1));
    assert!(matches!(
        recipe.prepare(&request()),
        Err(Refusal::InvalidFact(_))
    ));
    recipe.owners.pop();
    recipe.owners[0].value[0].activation = Activation::choice(99, 0);
    assert!(matches!(
        recipe.prepare(&request()),
        Err(Refusal::InvalidFact(_))
    ));
}

#[test]
fn composition_refuses_mismatched_geometry_domains_and_non_target_experts() {
    let cfg = cfg();
    let base = recipe(&cfg);
    let refuses = |recipe: GlmRecipe<'_>| {
        assert!(matches!(
            recipe.prepare(&request()),
            Err(Refusal::InvalidFact(_))
        ))
    };
    let mut bad = base.clone();
    bad.targets[0].value.page_tokens = 64;
    refuses(bad);
    let mut bad = base.clone();
    bad.targets[0].value.owned_layers[0] = false;
    refuses(bad);
    let mut bad = base.clone();
    bad.exchanges[0].value.context.as_mut().unwrap().extent = 2048;
    refuses(bad);
    let mut bad = base.clone();
    bad.experts.as_mut().unwrap().layers = vec![0];
    refuses(bad);
    let mut bad = base.clone();
    bad.experts.as_mut().unwrap().layers = vec![2];
    refuses(bad);
    let mut bad = base.clone();
    bad.experts.as_mut().unwrap().rank_sets.clear();
    refuses(bad);
    let mut bad = base.clone();
    bad.prefill.push(bad.prefill[0]);
    refuses(bad);
    let mut bad = base.clone();
    bad.weights[0].when = vec![When::Embedding(EmbeddingHome::Device(1))];
    refuses(bad);
    let mut bad = base.clone();
    bad.targets[0].value.full_prefill_logits = true;
    refuses(bad);
    let mut bad = base.clone();
    bad.attention = vec![
        LayerMode::Whole {
            gpu: 0,
            ffn: FfnMode::Split
        };
        cfg.layers
    ];
    refuses(bad);
    let mut beyond = request();
    beyond.context = 8192.into();
    assert!(matches!(
        base.prepare(&beyond),
        Err(Refusal::InvalidFact(_))
    ));
}

#[test]
fn resident_cap_counts_target_draft_and_asymmetric_expert_owners_not_storage() {
    use crate::placement::ResidentWeights;
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    recipe.embedding = vec![EmbeddingHome::Device(0)];
    recipe.weights.retain(|case| {
        matches!(
            case.when.as_slice(),
            [When::Embedding(EmbeddingHome::Device(0))]
        )
    });
    let draft = Allocation::buffer(
        "draft.weights",
        Location::coordinator(0),
        4096,
        Lifetime::RESIDENT,
        Category::Drafter,
        "native draft weights",
    );
    recipe.owners.push(Case {
        when: vec![],
        value: vec![
            Component::allocation(
                draft.clone(),
                ByteCurve::fixed(4096, recipe.maximum_pages),
                false,
                recipe.maximum_pages,
            ),
            Component::resident_weight(
                ResidentWeight::from_allocation(&draft, ResidentWeights(4096)),
                ByteCurve::fixed(4096, recipe.maximum_pages),
            ),
            Component::resident_weight(
                ResidentWeight {
                    id: "expert.rank1.extra".into(),
                    location: Location::coordinator(1),
                    weights: ResidentWeights(1024),
                },
                ByteCurve::fixed(1024, recipe.maximum_pages),
            ),
        ],
    });
    let mut req = Request {
        resident_weight_limit: Some(ResidentWeights(12288)),
        ..request()
    };
    let plan = planner::solve(req.clone(), budgets(), &recipe).unwrap();
    assert_eq!(plan.decisions.onboard_layers, vec![Vec::<usize>::new(); 2]);
    assert_eq!(plan.pool_tokens, 4096);
    assert!(
        !plan
            .facts
            .allocations
            .iter()
            .any(|a| a.id == "expert.rank1.extra")
    );
    assert_eq!(
        plan.facts
            .resident_weights
            .iter()
            .filter(|a| a.location == Location::coordinator(0))
            .map(|a| a.weights.0)
            .sum::<u64>(),
        8192 + 4096
    );
    req.onboard = Pin::Exact(1);
    assert!(matches!(
        planner::solve(req.clone(), budgets(), &recipe),
        Err(Refusal::ResidentWeightLimit { .. })
    ));
    req.resident_weight_limit = Some(ResidentWeights(8192 + 4096 + 16384));
    let admitted = planner::solve(req.clone(), budgets(), &recipe).unwrap();
    assert_eq!(admitted.decisions.onboard_layers, vec![vec![1], vec![1]]);
    assert_eq!(admitted.pool_tokens, plan.pool_tokens);
    assert_eq!(
        admitted
            .facts
            .resident_weights
            .iter()
            .filter(|a| a.location == Location::coordinator(1))
            .map(|a| a.weights.0)
            .sum::<u64>(),
        16384 + 1024
    );
    req.resident_weight_limit = Some(ResidentWeights(8192 + 4096 + 16384 - 1));
    assert!(matches!(
        planner::solve(req, budgets(), &recipe),
        Err(Refusal::ResidentWeightLimit {
            gpu: 0,
            required: 28672,
            limit: 28671
        })
    ));
}

#[test]
fn bf16_target_receipt_matches_selected_physical_representation() {
    use crate::plan::{testing, Checkpoint};
    let (config, tensors) = testing::glm5_compact_weights();
    let tensors = tensors.into_iter().filter(|t| !t.0.ends_with("_scale_inv"))
        .map(|(name, dtype, shape)| (name, if dtype == "F8_E4M3" { "BF16" } else { dtype }, shape)).collect::<Vec<_>>();
    let dir = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir.path(), &config, &tensors, None);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let mut totals = Vec::new();
    for bf16 in [false, true] {
        let target = TargetFacts::checkpoint(&checkpoint, 2, 2, None, bf16,
            &[EmbeddingHome::Device(0), EmbeddingHome::HostPinned], None).unwrap();
        let device = target.device_weight_bytes(EmbeddingHome::Device(0)).unwrap();
        let host = target.device_weight_bytes(EmbeddingHome::HostPinned).unwrap();
        assert_eq!(device[0] - host[0], 64 * 256 * 2);
        assert_eq!(device[1], host[1]);
        assert!(target.device_weight_bytes(EmbeddingHome::Device(1)).is_err());
        let groups = target.weights[0].1.resident_groups().unwrap();
        assert_eq!(device, groups.iter().map(|rank| rank.iter().map(|entry| entry.2).sum()).collect::<Vec<u64>>());
        totals.push(device.into_iter().sum::<u64>());
    }
    assert!(totals[1] > totals[0], "native BF16 must not use the FP8 admission");
}

#[test]
fn scoring_host_output_is_additional_to_text_only_rss() {
    let cfg = cfg();
    let shape = PrefillShape { rows: 128, waves: 3 };
    let mut input = production_input(&cfg, 2, 4, shape);
    input.operation = Operation::Scoring;
    let mut scoring = request();
    scoring.operation = Operation::Scoring;
    let admitted = input.prepare().unwrap().admit(scoring).unwrap();
    let owner = admitted.solved.facts.allocations.iter().find(|owner| owner.id == "glm.scoring.host_logits").unwrap();
    assert_eq!(owner.location, Location::HOST);
    assert_eq!(owner.bytes, cuteafd_core::memory_layout::host_logits_bytes(128, 128, cfg.vocab_size as u64).unwrap());
}

#[test]
fn optional_encoder_is_selected_jointly_and_allows_tower_only_spark() {
    use crate::plan::encoder::{EncoderKind, EncoderPlacement};
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    let off = EncoderPlacement { kind: EncoderKind::Off, weights: 0, scratch: 0, replicas: vec![], reason: "off".into(), shortfall: 0 };
    recipe.encoders = vec![EncoderPlacement { kind: EncoderKind::Rtx { gpu: 1 }, weights: 1 << 50, scratch: 0,
        ..off.clone() }, off.clone()];
    let solved = planner::solve(request(), budgets(), &recipe).unwrap();
    assert_eq!(solved.decisions.encoder, Some(1));
    assert_eq!(solved.encoder_diagnostics.len(), 1);
    assert!(solved.encoder_diagnostics[0].0.contains(&format!("during {}", Phase::Load)), "{:?}", solved.encoder_diagnostics);
    assert!(solved.encoder_diagnostics[0].1 > 0);
    assert!(!solved.facts.allocations.iter().any(|owner| owner.id.starts_with("glm.encoder")));
    recipe.encoders.pop();
    assert!(planner::solve(request(), budgets(), &recipe).is_err(), "explicit oversized encoder must refuse");
    recipe.encoders = vec![EncoderPlacement { kind: EncoderKind::Spark { rank: 0 }, weights: 256, ..off.clone() }, off];
    let mut pinned = request();
    pinned.onboard = Pin::Exact(recipe.experts.as_ref().unwrap().layers.len() as u64);
    let solved = planner::solve(pinned, budgets(), &recipe).unwrap();
    assert_eq!(solved.decisions.encoder, Some(0));
    let owners = solved.facts.allocations.iter().filter(|owner| owner.id.starts_with("glm.encoder")).collect::<Vec<_>>();
    assert_eq!(owners.iter().map(|owner| owner.bytes).sum::<u64>(), 256);
    assert!(owners.iter().all(|owner| owner.location.device == Device::Spark(0)));
}

#[test]
fn expert_usage_recipe_is_canonical_fixed_and_off_or_skip_has_no_owners() {
    use cuteafd_core::expert_usage::{device_owner, Geometry};
    use crate::placement::planner::ComponentKind;
    let cfg = cfg();
    let shape = PrefillShape { rows: 64, waves: 3 };
    for enabled in [false, true] {
        let mut input = production_input(&cfg, 2, 4, shape);
        input.expert_usage = enabled;
        let prepared = input.prepare().unwrap();
        let owners = prepared.recipe.owners.iter().flat_map(|case| &case.value)
            .filter_map(|component| if let ComponentKind::Allocation { template, .. } = &component.kind {
                (template.id.starts_with("expert-usage.") || template.id.starts_with("glm.usage.")).then_some(template)
            } else { None }).collect::<Vec<_>>();
        if !enabled {
            assert!(prepared.recipe.expert_usage.is_none());
            assert!(owners.is_empty());
            continue;
        }
        let geometry = Geometry { layers: cfg.layers, experts: cfg.experts, topk: cfg.topk,
            routed_layers: (cfg.first_moe_layer..cfg.layers).collect() };
        assert_eq!(prepared.recipe.expert_usage, Some(geometry.clone()));
        let table = geometry.table_bytes().unwrap() as u64;
        for suffix in ["counters", "snapshot", "pinned"] {
            let owner = owners.iter().find(|owner| owner.id == device_owner(0, suffix)).unwrap();
            assert_eq!(owner.bytes, table);
            assert_eq!(owner.location, if suffix == "pinned" { Location::PINNED } else { Location::coordinator(0) });
        }
        assert!(!owners.iter().any(|owner| owner.id.starts_with("expert-usage.gpu1")));
        assert_eq!(owners.iter().find(|owner| owner.id == "expert-usage.history").unwrap().bytes,
            geometry.host_bytes(1).unwrap() as u64);
        assert_eq!(owners.iter().find(|owner| owner.id == "glm.usage.lookup").unwrap().bytes,
            super::super::USAGE_LOOKUP_BYTES as u64);
        assert_eq!(owners.iter().filter(|owner| owner.location == Location::coordinator(0)
            && owner.id.starts_with("glm.usage.")).count(), 4);
    }
    let mut input = production_input(&cfg, 2, 0, PrefillShape { rows: 64, waves: 1 });
    input.expert_usage = true;
    input.skip_routed_experts = true;
    let prepared = input.prepare().unwrap();
    assert!(prepared.recipe.expert_usage.is_none());
    assert!(!prepared.recipe.owners.iter().flat_map(|case| &case.value).any(|component|
        matches!(&component.kind, ComponentKind::Allocation { template, .. }
            if template.id.starts_with("expert-usage.") || template.id.starts_with("glm.usage."))));
}
