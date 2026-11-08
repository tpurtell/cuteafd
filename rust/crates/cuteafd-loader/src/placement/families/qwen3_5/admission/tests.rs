use super::*;
use crate::families::qwen3_5::tests::native_planner_fixture;
use crate::plan::checkpoint::Checkpoint;
use crate::placement::planner::{self, Operation, Pin, Request};
use crate::placement::ResidentWeights;

fn fixture() -> (Qwen35Config, NativeWeights, NativeBuffers, crate::serving_capacity::WeightArenaLayout) {
    let (dir, _, buffers, _) = native_planner_fixture();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = Qwen35Config::from_hf(&checkpoint.config).unwrap();
    let weights = NativeWeights::native(&checkpoint, &cfg, cuteafd_core::DType::F32).unwrap();
    let arena = crate::families::qwen3_5::weight_layout::target_arena(&checkpoint, &cfg,
        cuteafd_core::DType::F32, crate::families::qwen3_5::weight_layout::DenseFormat::Bf16).unwrap();
    (cfg, weights, buffers, arena)
}

fn budgets() -> Vec<Budget> {
    forecast_budgets(&crate::placement::ARCH_CONTEXTS[1], 96 << 30, 0,
        (128 << 30, 128 << 30)).unwrap()
}

fn request() -> Request {
    Request { context: (262144).into(), pool: Pin::Auto, onboard: Pin::Auto, prefill: Pin::Auto,
        operation: Operation::Serving, resident_weight_limit: None,
        spark_reply_formats: Default::default() }
}

fn facts<'a>(cfg: &'a Qwen35Config, weights: &'a NativeWeights, buffers: &'a NativeBuffers,
    target_arena: &'a crate::serving_capacity::WeightArenaLayout) -> DenseFacts<'a> {
    DenseFacts { cfg, weights, target_arena, buffers, slots: 4, record_bytes: 4096, drafter: None, blas_head: serving::BlasHead::TensorLt, serving: None,
        host_prefix: serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }, verify_check: false }
}

#[test]
fn blas_heads_share_one_pool_only_for_genuine_bf16_drafts() {
    let (cfg, weights, buffers, arena) = fixture();
    for head in [serving::BlasHead::Native, serving::BlasHead::TensorLt, serving::BlasHead::Gemm] {
        for draft in [None, Some(false), Some(true)] {
            let mut input = facts(&cfg, &weights, &buffers, &arena);
            input.blas_head = head;
            input.drafter = draft.map(|gemm_head| DrafterBytes { gemm_head, ..Default::default() });
            let contract = input.prepare(&request()).unwrap();
            let selected = contract.facts(&vec![0; contract.variables.len()], 4).unwrap();
            for phase in [Phase::Load, Phase::DraftLoad, Phase::Ready] {
                let handles = selected.allowances.iter().filter(|owner|
                    owner.source.starts_with("cuBLAS handles:")
                        && owner.lifetime.contains(phase, contract.phase_order))
                    .map(|owner| owner.bytes).sum::<u64>();
                assert_eq!(handles, u64::from(phase != Phase::Load
                    && (head.gemm() || draft == Some(true))) * (64 << 20),
                    "head {head:?}, draft {draft:?}, phase {phase:?}");
            }
        }
    }
}

#[test]
fn auto_pool_matches_the_declared_owners_and_never_shrinks_context() {
    let (cfg, weights, buffers, arena) = fixture();
    let plan = admit(request(), budgets(), facts(&cfg, &weights, &buffers, &arena)).unwrap();
    assert_eq!(plan.pool_tokens, 2 << 20);
    assert_eq!(plan.pool_tokens % PAGE_ROWS, 0);
    assert_eq!(plan.decisions.embedding, Some(planner::EmbeddingHome::Device(0)));
    assert!(plan.decisions.onboard_layers.iter().all(Vec::is_empty));
    assert!(plan.decisions.expert_homes.is_empty());
    assert_eq!(plan.decisions.prefill, PrefillShape { rows: PREFILL_ROWS as u64, waves: 1 });
    assert_eq!(plan.decisions.lanes, 1);
    // The admitted plan's owners are exactly the physical declarations at the
    // admitted page count, not a second estimate.
    let contract = facts(&cfg, &weights, &buffers, &arena).prepare(&plan.request).unwrap();
    let declared = contract.facts(&vec![0; contract.variables.len()], plan.pool_tokens / PAGE_ROWS).unwrap();
    let map = |facts: &Facts| {
        facts.allocations.iter().map(|a| (a.id.clone(), (a.bytes, a.shape.clone(), a.view.clone(), a.lifetime)))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let left = map(&plan.facts);
    let right = map(&declared);
    // Staging is a named physical owner (or nothing); no pinned allowance
    // stands in for an unmeasured upload buffer any more.
    assert!(!plan.facts.allowances.iter().any(|allowance| allowance.location == Location::PINNED));
    // The retained driver stream slab has a second live stream only when the
    // host prefix cache exists; this plan has no host cache and pays none.
    assert!(!plan.facts.allowances.iter()
        .any(|allowance| allowance.source.starts_with("dense Qwen driver stream slabs")));
    assert_eq!(left.len(), plan.facts.allocations.len(), "duplicate owner id");
    assert_eq!(left.len(), right.len());
    for id in left.keys().chain(right.keys()).collect::<std::collections::BTreeSet<_>>() {
        assert_eq!(left.get(id), right.get(id), "{id}");
    }
    let residency = |facts: &Facts| facts.resident_weights.iter()
        .map(|owner| (owner.id.clone(), owner.weights.0)).collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(residency(&plan.facts), residency(&declared));
    assert_eq!(plan.facts.resident_weights.iter().map(|owner| owner.weights.0).sum::<u64>(), arena.bytes() as u64);
    assert_arena_charge(&plan.facts, arena.id(), arena.bytes() as u64, Lifetime::RESIDENT);
    assert!(!plan.facts.allocations.iter().any(|owner| owner.location == Location::coordinator(0)
        && owner.id.starts_with("weights.")));
    let pages = plan.pool_tokens / PAGE_ROWS;
    let kv: u64 = plan.facts.allocations.iter()
        .filter(|owner| owner.id.starts_with("kv.") && owner.view.is_none())
        .map(|owner| owner.bytes).sum();
    let full = cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Full).count() as u64;
    // One cudaMalloc per full-attention layer, each rounded to the SM120
    // allocation granularity.
    let per_layer = (pages * PAGE_ROWS * 4096).div_ceil(SM120_ALLOCATION_GRANULARITY_BYTES)
        * SM120_ALLOCATION_GRANULARITY_BYTES;
    assert_eq!(kv, full * per_layer);
    // No drafter: the only compiled extent is the 64-row page table itself.
    let contract = facts(&cfg, &weights, &buffers, &arena).prepare(&plan.request).unwrap();
    assert_eq!(contract.maximum_pages, u32::MAX as u64 / PAGE_ROWS);
}

#[test]
fn drafter_owners_track_pages_and_lower_the_compiled_extent() {
    let (cfg, weights, buffers, arena) = fixture();
    let drafter = DrafterBytes { owned: 4096, weights: ResidentWeights(1024),
        weight_arena: Some(("qwen3_5/mtp/gpu0", 1280)), scratch: 512, pools: vec![4096],
        workspace_requests: vec![2048, 1024], page_table_bytes: 0, landing: 0, staging: vec![],
        host_prefetch: 2_720_000_000, gemm_head: false };
    let mut input = facts(&cfg, &weights, &buffers, &arena);
    input.drafter = Some(drafter);
    let contract = input.prepare(&request()).unwrap();
    assert_eq!(contract.maximum_pages, i32::MAX as u64 / PAGE_ROWS);
    let plan = admit(request(), budgets(), input.clone()).unwrap();
    let pages = plan.pool_tokens / PAGE_ROWS;
    let owner = |id: &str| plan.facts.allocations.iter().find(|owner| owner.id == id).unwrap();
    assert_eq!(owner("draft.kv.0").bytes,
        (pages * 4096).div_ceil(SM120_ALLOCATION_GRANULARITY_BYTES) * SM120_ALLOCATION_GRANULARITY_BYTES);
    assert_eq!(plan.decisions.draft_home, Some(0));
    assert_eq!(owner("draft.prefetch").bytes, 2_720_000_000);
    assert_eq!(owner("draft.prefetch").location, Location::HOST);
    assert_eq!(owner("draft.prefetch").lifetime, Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad });
    assert_eq!(owner("qwen3_5/mtp/gpu0").bytes, 1280);
    assert!(!plan.facts.allocations.iter().any(|owner| owner.id == "draft.weights"));
    assert_eq!(owner("draft.runtime").bytes, owned_rest());
    assert_eq!(owner("draft.scratch").bytes, 512);
    assert_eq!(owner("draft.kv.0").lifetime, Lifetime { first: Phase::DraftLoad, last: Phase::Growth });
    assert!(plan.facts.resident_weights.contains(&ResidentWeight { id: "qwen3_5/mtp/gpu0".into(),
        location: Location::coordinator(0), weights: ResidentWeights(1280) }));
    // Draft KV is a page-linear extent, so a smaller explicit pool pays for
    // fewer pages rather than a fixed full-context reservation; its one
    // cudaMalloc still rounds to the SM120 backing granularity.
    let mut smaller = request();
    smaller.context = (4 * PAGE_ROWS).into();
    smaller.pool = Pin::Exact(4 * PAGE_ROWS);
    let small = admit(smaller, budgets(), input).unwrap();
    assert_eq!(small.pool_tokens, 4 * PAGE_ROWS);
    assert_eq!(small.facts.allocations.iter().find(|owner| owner.id == "draft.kv.0").unwrap().bytes,
        (4 * 4096u64).div_ceil(SM120_ALLOCATION_GRANULARITY_BYTES) * SM120_ALLOCATION_GRANULARITY_BYTES);
}

fn owned_rest() -> u64 { 4096 - 1024 }

fn assert_arena_charge(facts: &Facts, id: &str, bytes: u64, lifetime: Lifetime) {
    let roots: Vec<_> = facts.allocations.iter().filter(|owner| owner.id == id).collect();
    assert_eq!(roots.len(), 1);
    let root = roots[0];
    assert_eq!((root.bytes, root.location, root.lifetime), (bytes, Location::coordinator(0), lifetime));
    assert!(root.view.is_none());
    assert_eq!(root.group, "weight arena");
    assert_eq!(facts.resident_weights.iter().filter(|owner| owner.id == id)
        .map(|owner| owner.weights.0).collect::<Vec<_>>(), [bytes]);
    let tails: Vec<_> = facts.allocation_granularity.iter().filter(|tail| tail.id == format!("{id}.tail")).collect();
    assert_eq!(tails.len(), 1);
    let tail = tails[0];
    assert_eq!((tail.location, tail.lifetime), (root.location, lifetime));
    assert_eq!(tail.backings, [GranularityBacking { backing: id.into(), requests: vec![bytes] }]);
    assert_eq!(tail.bytes().unwrap(), allocation_granularity_forecast(&[bytes]).unwrap());
    assert!(matches!(tail.evidence, GranularityEvidence::Forecast));
}

#[test]
fn padded_weight_backings_replace_payload_and_activate_with_their_single_tails() {
    let (cfg, weights, buffers, _) = fixture();
    let payload = weights.layers.iter().sum::<u64>() + weights.norm + weights.head + weights.embedding;
    let arena = crate::serving_capacity::WeightArenaLayout::new("qwen3_5/target/gpu0",
        [("first".into(), 257), ("rest".into(), payload as usize - 257)]).unwrap();
    assert_eq!(arena.padding_bytes(), 255);
    let mut input = facts(&cfg, &weights, &buffers, &arena);
    input.drafter = Some(DrafterBytes { owned: 4096, weights: ResidentWeights(1024),
        weight_arena: Some(("qwen3_5/mtp/gpu0", 1280)), pools: vec![4096],
        workspace_requests: vec![2048, 1024], ..Default::default() });
    let contract = input.prepare(&request()).unwrap();
    let home = contract.variables.iter().position(|v| matches!(v, Variable::DraftHome(_))).unwrap();
    let components = |id: &str| contract.components.iter().filter(|component| match &component.kind {
        ComponentKind::Allocation { template, .. } => template.id == id,
        ComponentKind::ResidentWeight(owner) => owner.id == id,
        ComponentKind::AllocationGranularity(tail) => tail.id == format!("{id}.tail"),
        _ => false,
    }).collect::<Vec<_>>();
    let target = components(arena.id());
    assert_eq!(target.len(), 3);
    assert!(target.iter().all(|component| component.activation == Activation::always()));
    let draft = components("qwen3_5/mtp/gpu0");
    assert_eq!(draft.len(), 3);
    assert!(draft.iter().all(|component| component.activation == Activation::choice(home, 0)));
    let plan = admit(request(), budgets(), input.clone()).unwrap();
    assert_eq!(plan.facts.phase_order, PhaseOrder::TARGET_THEN_DRAFT);
    assert_arena_charge(&plan.facts, arena.id(), arena.bytes() as u64, Lifetime::RESIDENT);
    assert_arena_charge(&plan.facts, "qwen3_5/mtp/gpu0", 1280,
        Lifetime { first: Phase::DraftLoad, last: Phase::Growth });
    // Two weight-arena tails plus one tail for each side's fixed runtime
    // owners; pool-shaped KV owners price their own tail in `pool_curve`.
    assert_eq!(plan.facts.allocation_granularity.len(), 4);
    assert!(plan.facts.allocation_granularity.iter().any(|tail| tail.id == "dense.runtime.tail"));
    assert!(plan.facts.allocation_granularity.iter().any(|tail| tail.id == "draft.runtime.tail"));
    assert_eq!(plan.facts.allocations.iter().filter(|owner| owner.location == Location::coordinator(0)
        && owner.category == Category::Weights).map(|owner| owner.bytes).sum::<u64>(), arena.bytes() as u64);
    assert_eq!(plan.facts.allocations.iter().find(|owner| owner.id == "qwen3_5/mtp/gpu0").unwrap().category,
        cuteafd_core::memory_layout::MemoryKind::DrafterWeights.category());
    assert_eq!(plan.facts.allocations.iter().find(|owner| owner.id == "draft.runtime").unwrap().bytes, 3072);
    let mut missing = input;
    missing.drafter.as_mut().unwrap().weight_arena = None;
    assert!(matches!(admit(request(), budgets(), missing), Err(Refusal::InvalidFact(_))));
}

#[test]
fn exact_pools_report_typed_refusals_instead_of_shrinking_the_context() {
    let (cfg, weights, buffers, arena) = fixture();
    let input = facts(&cfg, &weights, &buffers, &arena);
    for (pool, expected) in [
        (Pin::Exact(PAGE_ROWS), "below context"),
        (Pin::Exact(3 * PAGE_ROWS + 1), "misaligned"),
        (Pin::Exact((32 << 20) * PAGE_ROWS), "beyond memory"),
    ] {
        let mut ask = request();
        ask.pool = pool;
        let refusal = admit(ask, budgets(), input.clone()).unwrap_err();
        match expected {
            "below context" => assert!(matches!(&refusal, Refusal::PoolBelowContext { .. }), "{refusal}"),
            "misaligned" => assert!(matches!(&refusal, Refusal::MisalignedPool { .. }), "{refusal}"),
            _ => assert!(matches!(&refusal, Refusal::Capacity { budget, .. } if budget == "coordinator.0"),
                "{refusal}"),
        }
    }
    // The compiled extent is a real ceiling, refused before any arithmetic.
    let mut ask = request();
    ask.pool = Pin::Exact((u32::MAX as u64 / PAGE_ROWS) * PAGE_ROWS + PAGE_ROWS);
    assert!(matches!(admit(ask, budgets(), input).unwrap_err(), Refusal::InvalidPin(_)));
}

#[test]
fn serving_selector_and_sampler_owners_keep_their_domains() {
    let (cfg, weights, buffers, arena) = fixture();
    for device_sampling in [false, true] {
        let mut input = facts(&cfg, &weights, &buffers, &arena);
        input.serving = Some(ServingFacts { capacity: 64, device_sampling });
        let plan = admit(request(), budgets(), input.clone()).unwrap();
        let selector: Vec<_> = plan.facts.allocations.iter()
            .filter(|owner| owner.id.starts_with("selector")).collect();
        assert_eq!(selector.len(), if device_sampling { 24 } else { 2 });
        assert_eq!(selector[0].id, "selector.output");
        assert_eq!(selector[0].bytes, 64 * 12);
        assert_eq!(selector[0].location, Location::coordinator(0));
        assert_eq!(selector[1].id, "selector.landing");
        assert_eq!(selector[1].location, Location::PINNED);
        assert!(selector.iter().all(|owner| owner.lifetime == Lifetime { first: Phase::Warmup, last: Phase::Growth }));
        let ready = plan.items(Location::coordinator(0)).unwrap();
        assert!(ready.iter().any(|item| item.group == "token selector"));
        // The pinned landing never charges the device budget.
        let pinned: u64 = selector.iter().filter(|owner| owner.location == Location::PINNED)
            .map(|owner| owner.bytes).sum();
        let host = plan.phases.iter().filter(|row| row.budget == "host")
            .map(|row| row.owned).max().unwrap();
        assert!(host >= pinned);
    }
}

#[test]
fn gdn_slots_and_prefix_marks_are_exact_partitions_of_their_backings() {
    let (cfg, weights, buffers, arena) = fixture();
    let plan = admit(request(), budgets(), facts(&cfg, &weights, &buffers, &arena)).unwrap();
    let layers = cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Gdn).count() as u64;
    let slots = 4u64;
    let conv = (cfg.gdn.conv_kernel - 1) as u64 * cfg.gdn.conv_width() as u64 * 2;
    for (backing, prefix, per_slot) in [
        ("gdn.conv", "gdn.conv.", conv),
        ("gdn.state", "gdn.state.", buffers.state_per_layer as u64 - conv),
    ] {
        let owner = plan.facts.allocations.iter().find(|owner| owner.id == backing).unwrap();
        let mut views: Vec<_> = plan.facts.allocations.iter()
            .filter(|view| view.id.starts_with(prefix)).collect();
        views.sort_by_key(|view| view.view.as_ref().unwrap().offset);
        assert_eq!(views.len() as u64, layers * slots);
        let mut cursor = 0;
        for view in &views {
            assert_eq!(view.view.as_ref().unwrap().backing, backing);
            assert_eq!(view.view.as_ref().unwrap().offset, cursor);
            assert_eq!(view.bytes, per_slot);
            assert_eq!(view.slot, Some((cursor / per_slot % slots) as u32));
            assert!(view.lifetime == Lifetime::TARGET);
            cursor += view.bytes;
        }
        assert_eq!(cursor, owner.bytes);
    }
    let marks: Vec<_> = plan.facts.allocations.iter().filter(|view| view.id.starts_with("prefix.mark."))
        .collect();
    assert_eq!(marks.len(), buffers.marks);
    let prefix = plan.facts.allocations.iter().find(|owner| owner.id == "prefix").unwrap();
    let end = marks.iter().filter_map(|view| view.view.as_ref().map(|view| view.offset + buffers.mark_bytes as u64))
        .max().unwrap();
    assert_eq!(end, prefix.bytes);
    assert!(marks.iter().all(|view| view.view.as_ref().unwrap().backing == "prefix"));
    // Every view's backing is a plain allocation of the same device domain.
    for view in plan.facts.allocations.iter().filter(|owner| owner.view.is_some()) {
        let backing = plan.facts.allocations.iter()
            .find(|owner| owner.id == view.view.as_ref().unwrap().backing).unwrap();
        assert_eq!(backing.location, view.location);
        assert!(backing.view.is_none());
    }
}

#[test]
fn host_memory_is_a_required_fact_and_live_samples_replace_the_forecast() {
    let (_, _, _, _) = fixture();
    assert!(matches!(forecast_budgets(&crate::placement::ARCH_CONTEXTS[1], 96 << 30, 0, (0, 0)),
        Err(Refusal::InvalidFact(_))));
    // No CUDA total reported yet: the nominal 32 GiB class is explicit.
    let forecast = forecast_budgets(&crate::placement::ARCH_CONTEXTS[0], 96 << 30, 0,
        (128 << 30, 64 << 30)).unwrap();
    assert_eq!(forecast[0].physical_bytes, 32 << 30);
    assert!(matches!(forecast[0].baseline, Baseline::Forecast { assumed: true, .. }));
    assert_eq!(forecast[1].locations, [Location::HOST, Location::PINNED]);
    let sampled = sampled_budgets(90 << 30, 96 << 30, None, (128 << 30, 64 << 30)).unwrap();
    assert_eq!(sampled[0].physical_bytes, 96 << 30);
    assert_eq!(sampled[0].logical_bytes, 96 << 30);
    assert!(matches!(sampled[0].baseline, Baseline::Sampled { free, phase: Phase::Load, .. }
        if free == 90 << 30));
    assert!(sampled_budgets(200 << 30, 96 << 30, None, (128 << 30, 64 << 30)).is_err());
}

#[test]
fn onboarding_pins_are_auto_zero_and_refuse_random_layer_counts() {
    let (cfg, weights, buffers, arena) = fixture();
    for onboard in [Pin::Auto, Pin::Exact(0)] {
        let mut ask = request();
        ask.onboard = onboard;
        let plan = admit(ask, budgets(), facts(&cfg, &weights, &buffers, &arena)).unwrap();
        assert!(plan.decisions.onboard_layers.iter().all(Vec::is_empty));
        assert!(plan.decisions.expert_homes.is_empty());
    }
    for layers in [1, 2, u64::MAX] {
        let mut ask = request();
        ask.onboard = Pin::Exact(layers);
        assert!(matches!(admit(ask, budgets(), facts(&cfg, &weights, &buffers, &arena)).unwrap_err(),
            Refusal::InvalidPin("model has 0 routed-expert layers")));
    }
    // A resident-weight limit is a pure policy refusal, not a silent trim.
    let mut ask = request();
    ask.resident_weight_limit = Some(ResidentWeights(1));
    assert!(matches!(admit(ask, budgets(), facts(&cfg, &weights, &buffers, &arena)).unwrap_err(),
        Refusal::ResidentWeightLimit { gpu: 0, .. }));
}

#[test]
fn operations_must_match_the_allocated_head_rows() {
    let (cfg, weights, buffers, arena) = fixture();
    let mut ask = request();
    ask.operation = Operation::Scoring;
    assert!(matches!(admit(ask, budgets(), facts(&cfg, &weights, &buffers, &arena)).unwrap_err(),
        Refusal::InvalidFact(_)));
    let mut input = facts(&cfg, &weights, &buffers, &arena);
    input.slots = 0;
    assert!(matches!(admit(request(), budgets(), input).unwrap_err(), Refusal::InvalidFact(_)));
    let mut input = facts(&cfg, &weights, &buffers, &arena);
    input.record_bytes = 0;
    assert!(matches!(admit(request(), budgets(), input).unwrap_err(), Refusal::InvalidFact(_)));
}

#[test]
fn solved_allowances_are_the_only_estimates_in_the_ready_ledger() {
    let (cfg, weights, buffers, arena) = fixture();
    let mut input = facts(&cfg, &weights, &buffers, &arena);
    input.serving = Some(ServingFacts { capacity: 64, device_sampling: true });
    let plan = admit(request(), budgets(), input).unwrap();
    let ready = plan.items(Location::coordinator(0)).unwrap();
    assert!(ready.iter().any(|item| item.basis == cuteafd_core::memory_layout::Basis::Estimated));
    let calibrated = plan.facts.allowances.iter().filter(|allowance| !allowance.assumed).collect::<Vec<_>>();
    assert!(!calibrated.is_empty());
    assert!(calibrated.iter().all(|allowance| allowance.source.starts_with("BLAS device initialization")));
    // Exact items are physical owners; calibrated BLAS residency is an allowance.
    let exact: u64 = ready.iter().filter(|item| item.basis == cuteafd_core::memory_layout::Basis::Exact)
        .map(|item| item.bytes).sum();
    let declared: u64 = plan.facts.allocations.iter()
        .filter(|owner| owner.view.is_none() && owner.location == Location::coordinator(0)
            && owner.lifetime.contains(Phase::Ready, plan.facts.phase_order))
        .map(|owner| owner.bytes).sum();
    assert_eq!(exact, declared);
    // Phase evaluation reconciles against the sample, never a second solve.
    let row = planner::evaluate(&plan.facts, &plan.budgets).unwrap();
    assert_eq!(row, plan.phases);
}

#[test]
fn host_chunks_and_diagnostic_staging_are_physical_owners_not_allowances() {
    let (cfg, weights, buffers, arena) = fixture();
    let mut input = facts(&cfg, &weights, &buffers, &arena);
    input.host_prefix = serving::HostPrefixOwners { quota_bytes: (1 << 30) + 17, chunk_bytes: 256 << 20 };
    input.verify_check = true;
    let plan = admit(request(), budgets(), input.clone()).unwrap();
    let owners = &plan.facts.allocations;
    let chunks: Vec<_> = owners.iter().filter(|owner| owner.id.starts_with("host.prefix.")).collect();
    assert_eq!(chunks.len(), 4);
    assert!(chunks.iter().all(|owner| owner.location == Location::PINNED && owner.bytes == 256 << 20));
    let reset: u64 = owners.iter().filter(|owner| owner.group == "diagnostic GDN reset staging")
        .map(|owner| owner.bytes).sum();
    let gdn = cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Gdn).count();
    assert_eq!(reset, (buffers.state_per_layer * input.slots * gdn) as u64);
    assert!(!owners.iter().any(|owner| owner.id.starts_with("selector")));
    input.host_prefix.quota_bytes = 0;
    input.host_prefix.chunk_bytes = 0;
    input.verify_check = false;
    let plain = admit(request(), budgets(), input).unwrap();
    assert!(!plain.facts.allocations.iter().any(|owner|
        owner.id.starts_with("host.prefix.") || owner.group == "diagnostic GDN reset staging"));
}
