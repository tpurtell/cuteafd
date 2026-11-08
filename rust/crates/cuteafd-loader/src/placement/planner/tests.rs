use super::*;
use cuteafd_core::SparkReplyFormat;
use crate::placement::FfnMode;

fn granular_owner(id: &str, backing: &str, requests: Vec<u64>, lifetime: Lifetime) -> AllocationGranularity {
    AllocationGranularity { id: id.into(), location: Location::coordinator(0), lifetime,
        backings: vec![GranularityBacking { backing: backing.into(), requests }], evidence: GranularityEvidence::Forecast }
}

#[test]
fn allocation_granularity_preserves_activation_lifetimes_serde_search_and_fixed_replay() {
    let maximum = 8;
    let mut contract = Linear { maximum: maximum * 64, ..Linear::new() }.prepare(&request()).unwrap();
    let lifetime = Lifetime { first: Phase::Ready, last: Phase::Growth };
    let owner = granular_owner("weights", "payload", vec![3 << 20, 24 << 10], lifetime);
    let payload = owner.requests().iter().sum();
    let option = contract.variables.len();
    contract.variables.push(Variable::Encoder(vec![0, 1]));
    let mut allocation = Component::allocation(Allocation::buffer("payload", owner.location,
        payload, lifetime, Category::Weights, "weights"), ByteCurve::fixed(payload, maximum), false, maximum);
    allocation.activation = Activation::choice(option, 0);
    contract.components.push(allocation);
    let mut component = allocation_granularity_component(owner.clone(), maximum).unwrap();
    component.activation = Activation::choice(option, 0);
    contract.components.push(component);
    // An exact multiple still retains its typed inventory when the slack is zero.
    let zero = granular_owner("zero", "aligned", vec![2 << 20], Lifetime::RESIDENT);
    contract.components.push(Component::allocation(Allocation::buffer("aligned", zero.location,
        2 << 20, zero.lifetime, Category::Weights, "aligned"), ByteCurve::fixed(2 << 20, maximum), false, maximum));
    contract.components.push(allocation_granularity_component(zero, maximum).unwrap());
    let budgets = vec![budget(16 << 20)];
    let compiled = search::Compiled::new(&contract, &budgets).unwrap();
    for selected in [&[0, 0, 0, 0][..], &[0, 0, 0, 1][..]] {
        for pages in 1..=maximum {
            let owners = contract.facts(selected, pages).unwrap();
            let phases = evaluate(&owners, &budgets).unwrap();
            assert_eq!(compiled.phases(&budgets, selected, pages).unwrap(), phases);
            assert_eq!(owners.allocation_granularity.len(), if selected[option] == 0 { 2 } else { 1 });
            for row in phases {
                assert_eq!(row.opaque, if selected[option] == 0 && lifetime.contains(row.phase, owners.phase_order) {
                    owner.bytes().unwrap()
                } else { 0 });
            }
            let encoded = serde_json::to_vec(&owners).unwrap();
            let restored: Facts = serde_json::from_slice(&encoded).unwrap();
            let mut replay = Contract::fixed(64, 1, 1, restored);
            replay.variables = vec![Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]), Variable::Lanes(vec![1])];
            assert_eq!(replay.facts(&[0, 0], 1).unwrap(), owners);
        }
    }
    cuteafd_core::memory_policy::MemoryPolicy { gpu_reserve: Some(0), ..Default::default() }.scoped(|| {
        let mut overridden = contract.clone();
        let mut budgets = budgets.clone();
        search::apply_reserve_override(&mut overridden, &mut budgets);
        assert_eq!(overridden.components.iter().filter(|c| matches!(c.kind, ComponentKind::AllocationGranularity(_))).count(), 2);
    });
}

#[test]
fn allocation_granularity_refuses_duplicate_alias_stale_and_wrong_domain_backings() {
    let owner = granular_owner("weights", "payload", vec![3 << 20], Lifetime::TARGET);
    let mut owners = facts(vec![Allocation::buffer("payload", owner.location, 3 << 20,
        owner.lifetime, Category::Weights, "weights")]);
    owners.allocation_granularity.push(owner);
    let budgets = vec![budget(16 << 20)];
    evaluate(&owners, &budgets).unwrap();
    let mut invalid = Vec::new();
    let mut duplicate = owners.clone();
    duplicate.allocation_granularity.push(duplicate.allocation_granularity[0].clone());
    invalid.push(duplicate);
    let mut duplicate_backing = owners.clone();
    let mut other = duplicate_backing.allocation_granularity[0].clone();
    other.id = "other".into();
    duplicate_backing.allocation_granularity.push(other);
    invalid.push(duplicate_backing);
    let mut alias = owners.clone();
    let mut view = alias.allocations[0].clone();
    view.id = "alias".into();
    view.view = Some(View { backing: "payload".into(), offset: 0 });
    alias.allocations.push(view);
    alias.allocation_granularity[0].backings[0].backing = "alias".into();
    invalid.push(alias);
    let mut stale = owners.clone();
    stale.allocation_granularity[0].backings[0].requests[0] += 1;
    invalid.push(stale);
    let mut missing = owners.clone();
    missing.allocation_granularity[0].backings[0].backing = "missing".into();
    invalid.push(missing);
    let mut wrong_domain = owners.clone();
    wrong_domain.allocation_granularity[0].location = Location { device: Device::Spark(0), domain: Domain::Managed };
    invalid.push(wrong_domain);
    let mut wrong_lifetime = owners.clone();
    wrong_lifetime.allocation_granularity[0].lifetime = Lifetime::RESIDENT;
    invalid.push(wrong_lifetime);
    let mut overflow = owners.clone();
    overflow.allocation_granularity[0].backings[0].requests = vec![u64::MAX];
    invalid.push(overflow);
    for owners in invalid {
        assert!(evaluate(&owners, &budgets).is_err());
        let mut replay = Contract::fixed(64, 1, 1, owners);
        replay.variables = vec![Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]), Variable::Lanes(vec![1])];
        assert!(solve(request(), budgets.clone(), &Prepared(replay)).is_err());
    }
}

#[test]
fn driver_page_tables_preserve_backings_aliases_activation_and_lifetimes() {
    let maximum = 32;
    let mut contract = Linear { maximum: maximum * 64, ..Linear::new() }.prepare(&request()).unwrap();
    contract.gpus = 2;
    let ready = Lifetime { first: Phase::Ready, last: Phase::Growth };
    let mut backing = Allocation::buffer("prefix.0", Location::PINNED, 1, ready, Category::Prefix, "host prefix");
    let mut component = Component::allocation(backing.clone(), ByteCurve::linear(1, 513, maximum), true, maximum);
    component.byte_extent(ByteCurve::linear(1, 513, maximum)).unwrap();
    let option = contract.variables.len();
    contract.variables.push(Variable::Encoder(vec![0, 1]));
    component.activation = Activation::choice(option, 0);
    contract.components.push(component);
    backing.id = "prefix.alias".into();
    backing.view = Some(View { backing: "prefix.0".into(), offset: 0 });
    let mut alias = Component::allocation(backing, ByteCurve::fixed(1, maximum), true, maximum);
    alias.activation = Activation::choice(option, 0);
    contract.components.push(alias);
    let load = Lifetime { first: Phase::Load, last: Phase::Load };
    let group = Sequential { id: "readers".into(), members: vec![
        Allocation::buffer("read.a", Location::PINNED, 1024, load, Category::Staging, "reader"),
        Allocation::buffer("read.b", Location::PINNED, 4096, load, Category::Staging, "reader"),
    ] };
    contract.components.push(Component { activation: Activation::always(), bytes: ByteCurve::fixed(group.peak(), maximum),
        kind: ComponentKind::Sequential(group) });
    contract.add_driver_page_tables();
    contract.add_driver_page_tables();
    assert_eq!(contract.components.iter().filter(|c| matches!(c.kind, ComponentKind::DriverPageTables { .. })).count(), 4);
    let mut gpu1 = budget(1 << 30);
    gpu1.id = "gpu1".into();
    gpu1.locations = vec![Location::coordinator(1)];
    let mut host = budget(1 << 30);
    host.id = "host".into();
    host.locations = vec![Location::HOST, Location::PINNED];
    let budgets = vec![budget(1 << 30), gpu1, host];
    let compiled = search::Compiled::new(&contract, &budgets).unwrap();
    for selected in [&[0, 0, 0, 0][..], &[0, 0, 0, 1][..]] {
        for pages in 1..=maximum {
            let owners = contract.facts(selected, pages).unwrap();
            assert_eq!(compiled.phases(&budgets, selected, pages).unwrap(), evaluate(&owners, &budgets).unwrap());
            let json = serde_json::to_vec(&owners).unwrap();
            let restored: Facts = serde_json::from_slice(&json).unwrap();
            let mut replay = Contract::fixed(64, 1, 2, restored);
            replay.add_driver_page_tables();
            replay.add_driver_page_tables();
            // Contract decisions need explicit physical execution geometry.
            replay.variables = vec![Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]), Variable::Lanes(vec![1])];
            let replayed = replay.facts(&[0, 0], 1).unwrap();
            assert_eq!(replayed, owners);
            assert_eq!(evaluate(&replayed, &budgets).unwrap(), evaluate(&owners, &budgets).unwrap());
            let mut stale = owners.clone();
            stale.driver_page_tables[0].mapped_bytes += 1;
            assert!(matches!(evaluate(&stale, &budgets), Err(Refusal::InvalidFact(_))));
            let mut duplicate = owners.clone();
            duplicate.driver_page_tables.push(duplicate.driver_page_tables[0].clone());
            assert!(matches!(evaluate(&duplicate, &budgets), Err(Refusal::InvalidFact(_))));
            for gpu in 0..2 {
                let rows: Vec<_> = owners.driver_page_tables.iter().filter(|a| a.location == Location::coordinator(gpu)).collect();
                assert_eq!(rows.len(), if selected[option] == 0 { 2 } else { 1 });
                let reader = rows.iter().find(|a| a.backing == "readers").unwrap();
                assert_eq!((reader.bytes(), reader.lifetime), (8, load));
                if selected[option] == 0 {
                    let prefix = rows.iter().find(|a| a.backing == "prefix.0").unwrap();
                    assert_eq!((prefix.bytes(), prefix.lifetime), (driver_page_table_bytes(1 + (pages - 1) * 513), ready));
                }
            }
        }
    }
    for (bytes, expected) in [(0, 0), (1, 1), (512, 1), (513, 2), (32 << 30, 64 << 20)] {
        assert_eq!(driver_page_table_bytes(bytes), expected);
    }
    assert_eq!(driver_page_table_bytes(u64::MAX), u64::MAX / 512 + 1);
    cuteafd_core::memory_policy::MemoryPolicy { gpu_reserve: Some(0), ..Default::default() }.scoped(|| {
        let mut overridden = contract.clone();
        let mut overridden_budgets = budgets.clone();
        search::apply_reserve_override(&mut overridden, &mut overridden_budgets);
        assert_eq!(overridden.components.iter().filter(|component|
            matches!(component.kind, ComponentKind::DriverPageTables { .. })).count(), 4);
    });
}

#[test]
fn driver_page_table_staircase_limits_pool_without_rounding_slope() {
    let maximum = 32;
    let mut contract = Linear { maximum: maximum * 64, ..Linear::new() }.prepare(&request()).unwrap();
    let pinned = Allocation::buffer("mapped", Location::PINNED, 1, Lifetime::TARGET, Category::Workspace, "mapped");
    let mut component = Component::allocation(pinned, ByteCurve::linear(1, 513, maximum), false, maximum);
    component.byte_extent(ByteCurve::linear(1, 513, maximum)).unwrap();
    contract.components.push(component);
    contract.add_driver_page_tables();
    let mut host = budget(1 << 30);
    host.id = "host".into();
    host.locations = vec![Location::PINNED];
    for bytes in [1039, 1040, 1041] {
        let expected = (1..=maximum).filter(|pages| pages * 64 + driver_page_table_bytes(1 + (pages - 1) * 513) <= bytes).max().unwrap();
        let plan = solve(request(), vec![budget(bytes), host.clone()], &Prepared(contract.clone())).unwrap();
        assert_eq!(plan.pool_tokens, expected * 64);
        assert!(plan.facts.driver_page_tables.iter().any(|a| a.backing == "mapped"));
    }
}

#[test]
fn driver_models_preserve_physical_backings_opaque_costs_and_fixed_replay() {
    let maximum = 8;
    let spark = Location { device: Device::Spark(0), domain: Domain::Managed };
    let mut contract = Linear { maximum: maximum * 64, ..Linear::new() }.prepare(&request()).unwrap();
    let option = contract.variables.len();
    contract.variables.push(Variable::Encoder(vec![0, 1]));
    let curve = ByteCurve::linear(1, 4097, maximum);
    let arena = Allocation::buffer("arena", spark, 1, Lifetime::RESIDENT, Category::Experts, "experts");
    let mut owner = Component::allocation(arena.clone(), curve.clone(), false, maximum);
    owner.byte_extent(curve.clone()).unwrap();
    owner.activation = Activation::choice(option, 0);
    contract.components.push(owner);
    let mut alias = arena;
    alias.id = "arena.alias".into();
    alias.view = Some(View { backing: "arena".into(), offset: 0 });
    let mut alias = Component::allocation(alias, ByteCurve::fixed(1, maximum), false, maximum);
    alias.activation = Activation::choice(option, 0);
    contract.components.push(alias);
    contract.components.push(Component {
        activation: Activation::choice(option, 0), bytes: curve,
        kind: ComponentKind::DriverPageTables { model: DriverPageTableModel::Sm121Arena,
            location: spark, lifetime: Lifetime::RESIDENT, backing: "arena".into() },
    });
    contract.components.push(Component::allocation(
        Allocation::buffer("portable", Location::PINNED, 513, Lifetime::TARGET, Category::Prefix, "prefix"),
        ByteCurve::fixed(513, maximum), false, maximum));
    contract.add_driver_page_tables();
    contract.add_driver_page_tables();
    let mut host = budget(1 << 30);
    host.id = "host".into();
    host.locations = vec![Location::PINNED];
    let mut worker = budget(1 << 30);
    worker.id = "spark".into();
    worker.locations = vec![spark];
    let budgets = vec![budget(1 << 30), host, worker];
    let compiled = search::Compiled::new(&contract, &budgets).unwrap();
    for selected in [&[0, 0, 0, 0][..], &[0, 0, 0, 1][..]] {
        for pages in 1..=maximum {
            let owners = contract.facts(selected, pages).unwrap();
            let phases = evaluate(&owners, &budgets).unwrap();
            assert_eq!(compiled.phases(&budgets, selected, pages).unwrap(), phases);
            let json = serde_json::to_vec(&owners).unwrap();
            let restored: Facts = serde_json::from_slice(&json).unwrap();
            let mut replay = Contract::fixed(64, 1, 1, restored);
            replay.add_driver_page_tables();
            replay.add_driver_page_tables();
            replay.variables = vec![Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]), Variable::Lanes(vec![1])];
            assert_eq!(replay.facts(&[0, 0], 1).unwrap(), owners);
            for phase in phases.iter().filter(|row| row.budget == "spark") {
                let bytes = if selected[option] == 0 { 1 + (pages - 1) * 4097 } else { 0 };
                assert_eq!((phase.owned, phase.opaque), (bytes, DriverPageTableModel::Sm121Arena.bytes(bytes)));
            }
            for row in &owners.driver_page_tables {
                let mut wrong_model = owners.clone();
                wrong_model.driver_page_tables.iter_mut().find(|item| item.backing == row.backing).unwrap().model =
                    match row.model {
                        DriverPageTableModel::Sm120Portable => DriverPageTableModel::Sm121Arena,
                        DriverPageTableModel::Sm121Arena => DriverPageTableModel::Sm120Portable,
                    };
                assert!(matches!(evaluate(&wrong_model, &budgets), Err(Refusal::InvalidFact(_))));
            }
            if selected[option] == 0 {
                let mut alias = owners.clone();
                alias.driver_page_tables.iter_mut().find(|row| row.backing == "arena").unwrap().backing = "arena.alias".into();
                assert!(matches!(evaluate(&alias, &budgets), Err(Refusal::InvalidFact(_))));
                let mut stale = owners.clone();
                stale.driver_page_tables.iter_mut().find(|row| row.backing == "arena").unwrap().mapped_bytes += 1;
                assert!(matches!(evaluate(&stale, &budgets), Err(Refusal::InvalidFact(_))));
                let mut wrong_owner = owners.clone();
                wrong_owner.allocations.iter_mut().find(|owner| owner.id == "arena").unwrap().category = Category::Workspace;
                assert!(matches!(evaluate(&wrong_owner, &budgets), Err(Refusal::InvalidFact(_))));
            }
        }
    }
}

#[test]
fn spark_driver_staircase_matches_exact_search_and_affine_pricing() {
    let maximum = 8;
    let spark = Location { device: Device::Spark(0), domain: Domain::Managed };
    for slope in [4096, 4097] {
        let mut contract = Linear { maximum: maximum * 64, ..Linear::new() }.prepare(&request()).unwrap();
        let curve = ByteCurve::linear(1, slope, maximum);
        let mut owner = Component::allocation(
            Allocation::buffer("arena", spark, 1, Lifetime::RESIDENT, Category::Experts, "experts"),
            curve.clone(), false, maximum);
        owner.byte_extent(curve.clone()).unwrap();
        contract.components.push(owner);
        contract.components.push(Component { activation: Activation::always(), bytes: curve,
            kind: ComponentKind::DriverPageTables { model: DriverPageTableModel::Sm121Arena,
                location: spark, lifetime: Lifetime::RESIDENT, backing: "arena".into() } });
        for bytes in [4144, 4145, 8241, 8242, 8243, 12400] {
            let expected = (1..=maximum).filter(|pages| {
                let arena = 1 + (pages - 1) * slope;
                arena + DriverPageTableModel::Sm121Arena.bytes(arena) <= bytes
            }).max().unwrap();
            let mut worker = budget(bytes);
            worker.id = "spark".into();
            worker.locations = vec![spark];
            let plan = solve(request(), vec![budget(1 << 30), worker], &Prepared(contract.clone())).unwrap();
            assert_eq!(plan.pool_tokens, expected * 64, "slope={slope}, budget={bytes}");
            assert_eq!(plan.facts.driver_page_tables.len(), 1);
        }
    }
}

#[test]
fn coordinator_devices_preserve_logical_rank_for_reversed_and_sparse_ordinals() {
    for physical in [[0, 1], [1, 0], [7, 3]] {
        let devices = CoordinatorDevices::new(&physical).unwrap();
        for (rank, device) in physical.into_iter().enumerate() {
            assert_eq!(devices.logical_rank(device).unwrap(), rank as u8);
            assert_eq!(devices.location(device).unwrap(), Location::coordinator(rank as u8));
            assert_eq!(devices.physical_device(rank as u8).unwrap(), device);
        }
        assert!(devices.logical_rank(9).is_err());
        assert!(devices.physical_device(2).is_err());
    }
    for physical in [&[][..], &[0, 0], &[-1, 0], &[0, 1, 2]] {
        assert!(CoordinatorDevices::new(physical).is_err());
    }
    assert_eq!(CoordinatorDevices::new(&[1]).unwrap().location(1).unwrap(), Location::coordinator(0));
}

#[test]
fn startup_graph_capture_replaces_opaque_estimate_without_reselection() {
    use crate::placement::inventory::{CapturedGraphRank, GraphRank, GraphSet, Lifetime as GraphLifetime};
    let rank = GraphRank::estimated(2, 10);
    let owner = serving::startup_graph_component("test.0", Location::coordinator(0), rank, 1).unwrap();
    let ComponentKind::Allowance(owner) = owner.kind else { panic!("graph is not an opaque allowance") };
    let mut physical = facts(vec![allocation("kv", 100)]);
    physical.allowances.push(owner);
    let budgets = vec![budget(1000)];
    let mut plan = ResolvedPlan { encoder_diagnostics: vec![], request: request(), decisions: decisions(1),
        pool_tokens: 64, phases: evaluate(&physical, &budgets).unwrap(), facts: physical, budgets };
    let original = plan.clone();
    let graphs = GraphSet { ranks: vec![rank], shapes: 1, lifetime: GraphLifetime::Startup };
    let declared: Vec<String> = vec!["exact keys".into()];
    let capture = CapturedGraphRank { declared_graphs: declared[0].clone(), executables: 2, bytes: 200 };
    let measured = serving::reconcile_startup_graphs(&mut plan, "test", &graphs, &declared, &[capture.clone()]).unwrap();
    assert_eq!(measured.ranks[0].bytes, 200);
    assert_eq!(measured.ranks[0].basis, Basis::Calibrated);
    assert_eq!(plan.request, original.request);
    assert_eq!(plan.decisions, original.decisions);
    assert_eq!(plan.pool_tokens, original.pool_tokens);
    assert_eq!(plan.budgets, original.budgets);
    assert_eq!(plan.facts.allocations, original.facts.allocations);
    assert_eq!(plan.facts.allowances[0].bytes, 200);
    assert!(!plan.facts.allowances[0].assumed);
    for phase in plan.phases.iter().filter(|phase| matches!(phase.phase, Phase::Warmup | Phase::Ready | Phase::Growth)) {
        assert_eq!((phase.owned, phase.opaque), (100, 200));
    }
    let good = plan.clone();
    for bad in [CapturedGraphRank { bytes: 1000, ..capture.clone() },
        CapturedGraphRank { declared_graphs: "wrong keys".into(), ..capture.clone() },
        CapturedGraphRank { executables: 3, ..capture.clone() }] {
        assert!(serving::reconcile_startup_graphs(&mut plan, "test", &graphs, &declared, &[bad]).is_err());
        assert_eq!(plan, good);
    }
    let mut duplicate = good.clone();
    duplicate.facts.allowances.push(duplicate.facts.allowances[0].clone());
    let unchanged = duplicate.clone();
    assert!(serving::reconcile_startup_graphs(&mut duplicate, "test", &graphs, &declared, &[capture.clone()]).is_err());
    assert_eq!(duplicate, unchanged);
    let mut missing = good.clone();
    missing.facts.allowances.clear();
    assert!(serving::reconcile_startup_graphs(&mut missing, "test", &graphs, &declared, &[capture]).is_err());
    assert!(serving::startup_graph_component("", Location::coordinator(0), rank, 1).is_err());
    assert!(serving::startup_graph_component("test", Location::coordinator(0), rank, 0).is_err());
    assert!(serving::startup_graph_component("test", Location::coordinator(0), GraphRank { margin: 1, ..rank }, 1).is_err());
}

#[test]
fn shared_arena_reconciliation_maps_physical_ids_without_changing_the_plan() -> anyhow::Result<()> {
    use crate::serving_capacity::{expert_weight_arena_id, WeightArenaLayout};
    for physical in [&[1][..], &[0, 1], &[1, 0]] {
        let devices = CoordinatorDevices::new(physical)?;
        let mut planned = Vec::new();
        let mut runtime = Vec::new();
        for (rank, &ordinal) in physical.iter().enumerate() {
            let location = Location::coordinator(rank as u8);
            for (selection, label) in [(vec![("layer.3".into(), 257), ("layer.7".into(), 17)], "target"),
                (vec![("draft.0".into(), 19)], "draft")] {
                let layout = WeightArenaLayout::new(expert_weight_arena_id("exl3", rank as i32, &selection)?, selection.clone())?;
                let physical_id = expert_weight_arena_id("exl3", ordinal, &selection)?;
                let root = Allocation::buffer(layout.id(), location, layout.bytes() as u64,
                    Lifetime::TARGET, Category::Weights, "weight arena");
                let address = 0x10000 + rank as u64 * 0x10000 + if label == "draft" { 0x8000 } else { 0 };
                planned.push(root.clone());
                runtime.push(Receipt { allocation: Allocation { id: physical_id.clone(), ..root }, address });
                for (name, view) in layout.views() {
                    let mut alias = Allocation::buffer(format!("{label}.rank.{rank}.{name}"), location, view.bytes() as u64,
                        Lifetime::TARGET, Category::Experts, "resident expert owners");
                    alias.view = Some(View { backing: layout.id().into(), offset: view.offset() as u64 });
                    planned.push(alias.clone());
                    alias.view.as_mut().unwrap().backing = physical_id.clone();
                    runtime.push(Receipt { allocation: alias, address: address + view.offset() as u64 });
                }
            }
        }
        let mut budgets = vec![budget(4096)];
        if physical.len() == 2 {
            let mut peer = budget(4096);
            peer.id = "peer".into();
            peer.locations = vec![Location::coordinator(1)];
            budgets.push(peer);
        }
        let mut facts = facts(planned.clone());
        facts.resident_weights = facts.allocations.iter().filter(|a| a.view.is_none())
            .map(|a| ResidentWeight::from_allocation(a, ResidentWeights(a.bytes))).collect();
        let plan = ResolvedPlan { encoder_diagnostics: vec![], request: request(), decisions: decisions(0),
            pool_tokens: 64, phases: evaluate(&facts, &budgets)?, budgets, facts };
        let original = plan.clone();
        let mapped = devices.remap_plan_weight_arenas(&plan)?;
        assert_eq!(plan, original, "identity translation mutates the admitted plan");
        assert_eq!(mapped.decisions, plan.decisions);
        assert_eq!(mapped.phases, plan.phases);
        assert_eq!(mapped.budgets, plan.budgets);
        assert_eq!(mapped.facts.allocations, runtime.iter().map(|r| r.allocation.clone()).collect::<Vec<_>>());
        for resident in &mapped.facts.resident_weights {
            assert!(mapped.facts.allocations.iter().any(|a| a.id == resident.id && a.location == resident.location));
        }
        assert_eq!(devices.remap_weight_arenas(&planned)?, mapped.facts.allocations);
        devices.reconcile(&plan, Phase::Ready, &runtime)?;
        let history = runtime.iter().cloned().map(|receipt| Observation {
            created: receipt.allocation.lifetime.first, freed: None, receipt }).collect::<Vec<_>>();
        devices.reconcile_history(&plan, Phase::Ready, &history)?;
        let events = runtime.iter().filter(|r| r.allocation.view.is_none()).cloned()
            .map(|receipt| OwnerEvent::Allocated { phase: receipt.allocation.lifetime.first, receipt }).collect::<Vec<_>>();
        let views = history.iter().filter(|o| o.receipt.allocation.view.is_some()).cloned().collect::<Vec<_>>();
        devices.reconcile_events(&plan, Phase::Ready, &events, &views)?;
        devices.reconcile_startup(&plan, Phase::Ready, &events, &views, &[])?;
        assert!(devices.reconcile(&plan, Phase::Ready, &runtime[1..]).is_err());
        let mut wrong = runtime.clone();
        wrong[0].allocation.bytes += 1;
        assert!(devices.reconcile(&plan, Phase::Ready, &wrong).is_err());
        wrong = runtime.clone();
        wrong[1].allocation.view.as_mut().unwrap().offset += 1;
        assert!(devices.reconcile(&plan, Phase::Ready, &wrong).is_err());
        if physical != [0, 1] { assert!(reconcile(&plan, Phase::Ready, &runtime).is_err()); }
        let mut invalid = planned.clone();
        invalid[0].id = "experts.exl3.gpu.0.bad-sha".into();
        assert!(devices.remap_weight_arenas(&invalid).is_err());
        assert!(devices.remap_weight_arenas(&planned[1..]).is_err(), "partial rootless inventory accepted");
    }
    let mut ordinary = allocation("core/logical/0", 64);
    ordinary.category = Category::Weights;
    ordinary.group = "weight arena".into();
    assert_eq!(CoordinatorDevices::new(&[1])?.remap_weight_arenas(&[ordinary.clone()])?, [ordinary]);
    Ok(())
}

#[test]
fn shared_arena_remap_preserves_derived_backings_and_all_owner_scopes() -> anyhow::Result<()> {
    use crate::serving_capacity::expert_weight_arena_id;
    for physical in [&[1][..], &[0, 1], &[1, 0]] {
        for order in [PhaseOrder::TARGET_THEN_DRAFT, PhaseOrder::DRAFT_THEN_TARGET] {
            let devices = CoordinatorDevices::new(physical)?;
            let mut planned = facts(vec![]);
            let mut expected = facts(vec![]);
            for inventory in [&mut planned, &mut expected] {
                inventory.phase_order = order;
                inventory.allocations.push(Allocation::buffer("mapped.prefix", Location::PINNED,
                    513, Lifetime::RESIDENT, Category::Prefix, "host prefix"));
            }
            let draft = Lifetime { first: Phase::DraftLoad, last: Phase::Growth };
            let load = Lifetime { first: Phase::Load, last: Phase::Load };
            for (rank, &ordinal) in physical.iter().enumerate() {
                let location = Location::coordinator(rank as u8);
                // Build physical expectations independently, including derived references.
                for (inventory, ordinal) in [(&mut planned, rank as i32), (&mut expected, ordinal)] {
                    let mut resident_backings = Vec::new();
                    let mut sequential = Vec::new();
                    for (scope, label, bytes, lifetime) in [
                        (0, "target", (3 << 20) + 257, Lifetime::RESIDENT),
                        (1, "deferred", (2 << 20) + 17, Lifetime::RESIDENT),
                        (1, "draft", (3 << 20) + 33, draft),
                        (2, "load.a", 513, load),
                        (2, "load.b", 1025, load),
                    ] {
                        let selection = vec![(label.into(), bytes as usize)];
                        let id = expert_weight_arena_id("exl3", ordinal, &selection)?;
                        let root = Allocation::buffer(&id, location, bytes, lifetime,
                            Category::Weights, "weight arena");
                        if scope == 2 {
                            sequential.push(root);
                            continue;
                        }
                        inventory.resident_weights.push(ResidentWeight::from_allocation(&root, ResidentWeights(bytes)));
                        let mut alias = Allocation::buffer(format!("{label}.rank.{rank}.view"), location,
                            17, lifetime, Category::Experts, "resident expert owners");
                        alias.view = Some(View { backing: id.clone(), offset: 256 });
                        let allocations = if scope == 0 { &mut inventory.allocations } else { &mut inventory.deferred };
                        allocations.extend([root, alias]);
                        let backing = GranularityBacking { backing: id, requests: vec![bytes] };
                        if label == "draft" {
                            inventory.allocation_granularity.push(AllocationGranularity {
                                id: format!("draft.rank.{rank}.tail"), location, lifetime,
                                backings: vec![backing], evidence: GranularityEvidence::Forecast,
                            });
                        } else {
                            resident_backings.push(backing);
                        }
                    }
                    inventory.allocation_granularity.push(AllocationGranularity {
                        id: format!("resident.rank.{rank}.tails"), location, lifetime: Lifetime::RESIDENT,
                        backings: resident_backings, evidence: GranularityEvidence::Forecast,
                    });
                    inventory.sequential.push(Sequential { id: format!("load.rank.{rank}"), members: sequential });
                    // SM120 maps pinned storage, not device weight arenas.
                    inventory.driver_page_tables.push(DriverPageTable { model: DriverPageTableModel::Sm120Portable,
                        location, lifetime: Lifetime::RESIDENT, backing: "mapped.prefix".into(), mapped_bytes: 513 });
                }
            }
            let mut budgets = (0..physical.len()).map(|rank| {
                let mut row = budget(64 << 20);
                row.id = format!("gpu{rank}");
                row.locations = vec![Location::coordinator(rank as u8)];
                row
            }).collect::<Vec<_>>();
            let mut host = budget(4096);
            host.id = "host".into();
            host.locations = vec![Location::PINNED];
            budgets.push(host);
            let plan = ResolvedPlan { encoder_diagnostics: vec![], request: request(), decisions: decisions(0),
                pool_tokens: 64, phases: evaluate(&planned, &budgets)?, budgets, facts: planned };
            let original = plan.clone();
            let mut expected_plan = plan.clone();
            expected_plan.facts = expected;
            assert_eq!(evaluate(&expected_plan.facts, &expected_plan.budgets)?, plan.phases);
            let mapped = devices.remap_plan_weight_arenas(&plan)?;
            assert_eq!(mapped, expected_plan);
            assert_eq!(plan, original, "translation mutates the admitted plan");
            assert_eq!(mapped.facts.driver_page_tables, plan.facts.driver_page_tables);
            for (before, after) in plan.facts.allocation_granularity.iter().zip(&mapped.facts.allocation_granularity) {
                assert_eq!(before.bytes()?, after.bytes()?);
                assert_eq!(before.requests(), after.requests());
            }
            let mut invalid = plan.clone();
            invalid.facts.allocation_granularity[0].backings[0].backing = "missing".into();
            let unchanged = invalid.clone();
            assert!(devices.remap_plan_weight_arenas(&invalid).is_err());
            assert_eq!(invalid, unchanged);
        }
    }
    Ok(())
}

fn budget(bytes: u64) -> Budget {
    Budget {
        id: "gpu".into(),
        locations: vec![Location::coordinator(0)],
        physical_bytes: bytes,
        logical_bytes: bytes,
        baseline: Baseline::Forecast {
            occupied: 0,
            source: "test witness".into(),
            assumed: false,
        },
        headroom: 0,
    }
}
fn allocation(id: &str, bytes: u64) -> Allocation {
    Allocation::buffer(
        id,
        Location::coordinator(0),
        bytes,
        Lifetime::TARGET,
        Category::Workspace,
        "test",
    )
}
fn facts(allocations: Vec<Allocation>) -> Facts {
    Facts { allocation_granularity: vec![], driver_page_tables: vec![], resident_weights: vec![],
        phase_order: crate::placement::planner::PhaseOrder::TARGET_THEN_DRAFT,
        allocations,
        deferred: vec![],
        sequential: vec![],
        reservations: vec![], allowances: vec![],
    }
}
fn request() -> Request {
    Request {
        context: (64).into(),
        pool: Pin::Auto,
        onboard: Pin::Auto,
        prefill: Pin::Auto,
        spark_reply_formats: Default::default(),
        operation: Operation::Serving, resident_weight_limit: None,
    }
}
fn decisions(onboard: u64) -> Decisions {
    Decisions {
        encoder: None,
        spark_reply_formats: None,
        attention: vec![LayerMode::Whole {
            gpu: 0,
            ffn: FfnMode::Owner,
        }],
        onboard_layers: vec![(0..onboard as usize).collect()],
        expert_homes: (0..onboard as usize).map(|layer|
            (layer, ExpertDestination::Coordinator { gpus: vec![0] })).collect(),
        embedding: None,
        draft_home: None,
        head_home: None,
        intake_home: None,
        prefill: PrefillShape {
            rows: 4096,
            waves: 1,
        },
        lanes: 1,
    }
}
struct Linear {
    onboard: u64,
    maximum: u64,
    fixed: u64,
}
impl Linear {
    fn new() -> Self {
        Self {
            onboard: 0,
            maximum: 2 << 20,
            fixed: 0,
        }
    }
}
impl Recipe for Linear {
    fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
        let maximum = self.maximum / 64;
        let mut contract = Contract::fixed(
            64,
            maximum,
            1,
            facts(vec![allocation("kv", 64 + self.fixed)]),
        );
        contract.set_byte_extent("kv", ByteCurve::linear(64 + self.fixed, 64, maximum))?;
        contract.variables = vec![
            Variable::Attention {
                layer: 0,
                values: decisions(0).attention,
            },
            Variable::Prefill(vec![decisions(0).prefill]),
            Variable::Lanes(vec![1]),
        ];
        if self.onboard != 0 {
            let variable = contract.variables.len();
            contract.variables.push(Variable::Onboard {
                layers: (0..self.onboard as usize).collect(),
                destination: vec![0],
                counts: (0..=self.onboard as usize).collect(),
            });
            for layer in 0..self.onboard as usize {
                let mut component = Component::allocation(
                    allocation(&format!("expert.{layer}"), 64),
                    ByteCurve::fixed(64, maximum),
                    false,
                    maximum,
                );
                component.activation = Activation {
                    clauses: (layer + 1..=self.onboard as usize)
                        .map(|option| vec![Choice { variable, option }])
                        .collect(),
                };
                contract.components.push(component);
            }
        }
        Ok(contract)
    }
}
fn plan(allocations: Vec<Allocation>) -> ResolvedPlan {
    let budgets = vec![budget(4096)];
    let facts = facts(allocations);
    let phases = evaluate(&facts, &budgets).unwrap();
    ResolvedPlan {
        encoder_diagnostics: vec![],
        request: request(),
        decisions: decisions(0),
        pool_tokens: 64,
        budgets,
        facts,
        phases,
    }
}
#[test]
fn ready_items_include_opaque_attribution_but_not_startup_peaks_or_views() {
    let owner = allocation("owner", 100);
    let mut view = allocation("view", 16);
    view.view = Some(View { backing: owner.id.clone(), offset: 0 });
    let mut startup = allocation("startup", 500);
    startup.lifetime = Lifetime { first: Phase::Load, last: Phase::Load };
    let mut p = plan(vec![owner, view, startup]);
    p.facts.allowances = vec![
        Allowance { location: Location::coordinator(0), bytes: 37,
            lifetime: Lifetime::RESIDENT, source: "measured context".into(), assumed: false },
        Allowance { location: Location::coordinator(0), bytes: 19,
            lifetime: Lifetime::TARGET, source: "estimated runtime".into(), assumed: true },
        Allowance { location: Location::coordinator(0), bytes: 11,
            lifetime: Lifetime { first: Phase::Load, last: Phase::Warmup },
            source: "load only".into(), assumed: true },
        Allowance { location: Location::coordinator(0), bytes: 13,
            lifetime: Lifetime { first: Phase::Growth, last: Phase::Growth },
            source: "growth only".into(), assumed: false },
        Allowance { location: Location::coordinator(0), bytes: 0,
            lifetime: Lifetime::RESIDENT, source: "empty".into(), assumed: true },
    ];
    p.phases = evaluate(&p.facts, &p.budgets).unwrap();
    let items = p.items(Location::coordinator(0)).unwrap();
    assert_eq!(items.len(), 3);
    assert_eq!(items.iter().map(|item| item.bytes).sum::<u64>(), 156);
    assert!(items.iter().all(|item|
        item.lifetime == cuteafd_core::memory_layout::ItemLifetime::ReadyResident));
    for (name, bytes, basis) in [("measured context", 37, Basis::Calibrated),
        ("estimated runtime", 19, Basis::Estimated)]
    {
        let item = items.iter().find(|item| item.group == name).unwrap();
        assert_eq!((item.bytes, item.category, item.basis), (bytes, Category::Runtime, basis));
    }
    assert_eq!(items.iter().filter(|item| item.basis == Basis::Exact)
        .map(|item| item.bytes).sum::<u64>(), 100);
    let ready = p.phases.iter().find(|row| row.phase == Phase::Ready).unwrap();
    assert_eq!(ready.owned + ready.opaque, 156);
    let load = p.phases.iter().find(|row| row.phase == Phase::Load).unwrap();
    assert_eq!(load.owned + load.opaque, 548);
}

#[test]
fn capacity_refusal_names_binding_budget_phase_and_exact_bytes() {
    let refusal = Refusal::Capacity { budget: "coordinator GPU 0".into(),
        phase: Phase::Warmup, required: 1025, available: 1024, owners: vec![] };
    let text = refusal.to_string();
    assert!(text.contains("memory budget 'coordinator GPU 0' during decode warm-up"), "{text}");
    assert!(text.contains("needs 1025 bytes, has 1024 bytes available"), "{text}");
}

fn receipts(plan: &ResolvedPlan) -> Vec<Receipt> {
    plan.facts
        .allocations
        .iter()
        .enumerate()
        .map(|(index, a)| Receipt {
            allocation: a.clone(),
            address: 4096 + index as u64 * 4096,
        })
        .collect()
}
fn errors(result: Result<(), Refusal>) -> Vec<Mismatch> {
    match result {
        Err(Refusal::Reconciliation(errors)) => errors,
        other => panic!("expected reconciliation refusal: {other:?}"),
    }
}

struct Prepared(Contract);
impl Recipe for Prepared {
    fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
        Ok(self.0.clone())
    }
}

#[test]
fn pool_page_multiple_aligns_automatic_pool_floor_extent_and_explicit_pins() {
    let mut contract = Linear { maximum: 64 * 9, ..Linear::new() }.prepare(&request()).unwrap();
    assert_eq!(contract.pool_page_multiple, 1);
    assert_eq!(solve(request(), vec![budget(64 * 7)], &Prepared(contract.clone())).unwrap().pool_tokens, 64 * 7);
    contract.pool_page_multiple = 2;
    for (capacity, pages) in [(9, 8), (8, 8), (7, 6), (3, 2)] {
        let result = solve(request(), vec![budget(64 * capacity)], &Prepared(contract.clone())).unwrap();
        assert_eq!(result.pool_tokens, 64 * pages);
        assert_eq!(result.facts.allocations.iter().find(|owner| owner.id == "kv").unwrap().bytes, 64 * pages);
    }
    let mut pinned = request();
    pinned.pool = Pin::Exact(64 * 7);
    assert!(matches!(solve(pinned, vec![budget(1024)], &Prepared(contract.clone())),
        Err(Refusal::MisalignedPool { tokens: 448, page_rows: 128 })));
    let mut floor = request();
    floor.context = (64 * 3).into();
    assert_eq!(solve(floor.clone(), vec![budget(64 * 5)], &Prepared(contract.clone())).unwrap().pool_tokens, 64 * 4);
    assert!(matches!(solve(floor, vec![budget(64 * 3)], &Prepared(contract.clone())),
        Err(Refusal::FullContextShortfall { tokens: 128, .. })));
    for multiple in [0, u64::MAX] {
        contract.pool_page_multiple = multiple;
        assert!(matches!(solve(request(), vec![budget(1024)], &Prepared(contract.clone())), Err(Refusal::InvalidFact(_))));
    }
}

#[test]
fn strided_pool_search_matches_enumeration_across_nonmonotonic_breakpoints() {
    let mut contract = Linear { maximum: 64 * 13, ..Linear::new() }.prepare(&request()).unwrap();
    let curve = ByteCurve(vec![
        BytePiece { first: 1, last: 3, bytes_at_first: 10, bytes_per_page: 10 },
        BytePiece { first: 4, last: 6, bytes_at_first: 100, bytes_per_page: 0 },
        BytePiece { first: 7, last: 7, bytes_at_first: 1, bytes_per_page: 0 },
        BytePiece { first: 8, last: 11, bytes_at_first: 20, bytes_per_page: 10 },
        BytePiece { first: 12, last: 13, bytes_at_first: 80, bytes_per_page: 5 },
    ]);
    contract.set_byte_extent("kv", curve.clone()).unwrap();
    for multiple in [1, 2, 3, 4] {
        contract.pool_page_multiple = multiple;
        for capacity in 1..=110 {
            let expected = (1..=13).filter(|pages| pages % multiple == 0)
                .filter(|&pages| curve.at(pages).unwrap() <= capacity).max();
            let result = solve(request(), vec![budget(capacity)], &Prepared(contract.clone()));
            if let Some(pages) = expected {
                assert_eq!(result.unwrap().pool_tokens, pages * 64, "multiple {multiple}, capacity {capacity}");
            } else {
                assert!(result.is_err(), "multiple {multiple}, capacity {capacity}");
            }
        }
    }
}

#[test]
fn immutable_reply_formats_flow_to_decisions_and_each_explicit_phase_conflicts() {
    use SparkReplyFormat::{Bf16, Fp8, Nvfp4};
    let mut contracts = vec![None];
    for prefill in [Bf16, Fp8, Nvfp4] {
        for decode in [Bf16, Fp8, Nvfp4] {
            contracts.push(Some(SparkReplyFormats { prefill: Some(prefill), decode: Some(decode) }));
        }
    }
    for resolved in contracts {
        let mut contract = Linear::new().prepare(&request()).unwrap();
        contract.spark_reply_formats = resolved;
        let selected = vec![0; contract.variables.len()];
        assert_eq!(contract.decisions(&selected).unwrap().spark_reply_formats, resolved);
        let recipe = Prepared(contract);
        for prefill in [None, Some(Bf16), Some(Fp8), Some(Nvfp4)] {
            for decode in [None, Some(Bf16), Some(Fp8), Some(Nvfp4)] {
                let requested = SparkReplyFormats { prefill, decode };
                let conflict = [(prefill, resolved.and_then(|formats| formats.prefill)),
                    (decode, resolved.and_then(|formats| formats.decode))]
                    .into_iter().any(|(requested, resolved)| requested.is_some() && requested != resolved);
                let req = Request { spark_reply_formats: requested, ..request() };
                match solve(req, vec![budget(1 << 40)], &recipe) {
                    Ok(plan) => {
                        assert!(!conflict, "{requested:?} vs {resolved:?}");
                        assert_eq!(plan.request.spark_reply_formats, requested);
                        assert_eq!(plan.decisions.spark_reply_formats, resolved);
                        let encoded = serde_json::to_value(&plan).unwrap();
                        assert_eq!(encoded["request"]["spark_reply_formats"]["prefill"], serde_json::to_value(prefill).unwrap());
                        assert_eq!(encoded["request"]["spark_reply_formats"]["decode"], serde_json::to_value(decode).unwrap());
                        let decoded: ResolvedPlan = serde_json::from_value(encoded).unwrap();
                        assert_eq!(decoded, plan);
                    }
                    Err(Refusal::PinConflict) => assert!(conflict, "{requested:?} vs {resolved:?}"),
                    other => panic!("unexpected reply-format admission: {other:?}"),
                }
            }
        }
    }
}

fn assert_curve_census(contract: &Contract, budgets: &[Budget], selected: &[usize]) {
    contract.validate().unwrap();
    let costs = search::Compiled::new(contract, budgets).unwrap();
    let mut pages = BTreeSet::from([1, contract.maximum_pages]);
    for component in &contract.components {
        for piece in &component.bytes.0 {
            for boundary in [piece.first, piece.last] {
                for page in boundary.saturating_sub(1)..=boundary.saturating_add(1) {
                    if page >= 1 && page <= contract.maximum_pages {
                        pages.insert(page);
                    }
                }
            }
        }
    }
    for pages in pages {
        let exact = evaluate(&contract.facts(selected, pages).unwrap(), budgets).unwrap();
        assert_eq!(
            costs.phases(budgets, selected, pages).unwrap(),
            exact,
            "page {pages}"
        );
    }
}

#[test]
fn piecewise_cost_vectors_match_exact_owners_at_every_breakpoint_neighbor() {
    let mut contract = Linear {
        maximum: 64 * 20,
        fixed: 37,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    let curve = ByteCurve(vec![
        BytePiece {
            first: 1,
            last: 4,
            bytes_at_first: 0,
            bytes_per_page: 16,
        },
        BytePiece {
            first: 5,
            last: 8,
            bytes_at_first: 96,
            bytes_per_page: 0,
        },
        BytePiece {
            first: 9,
            last: 20,
            bytes_at_first: 128,
            bytes_per_page: 3,
        },
    ]);
    let mut bounded = Component::allocation(allocation("index", 16), curve.clone(), false, 20);
    bounded.byte_extent(curve).unwrap();
    contract.components.push(bounded);
    let mut backing = allocation("backing", 512);
    backing.lifetime = Lifetime::RESIDENT;
    contract.components.push(Component::allocation(
        backing,
        ByteCurve::fixed(512, 20),
        false,
        20,
    ));
    let mut view = allocation("view", 64);
    view.view = Some(View {
        backing: "backing".into(),
        offset: 0,
    });
    contract.components.push(Component::allocation(
        view,
        ByteCurve::fixed(64, 20),
        false,
        20,
    ));
    let mut staging = allocation("load.staging", 65);
    staging.location = Location::PINNED;
    staging.lifetime = Lifetime {
        first: Phase::Load,
        last: Phase::Load,
    };
    contract.components.push(Component::allocation(
        staging,
        ByteCurve::fixed(65, 20),
        false,
        20,
    ));
    contract.components.push(Component {
        activation: Activation::always(),
        bytes: ByteCurve::linear(2, 1, 20),
        kind: ComponentKind::Allowance(Allowance {
            location: Location::HOST,
            bytes: 2,
            lifetime: Lifetime::TARGET,
            source: "test allowance".into(),
            assumed: true,
        }),
    });
    let mut host = budget(1 << 30);
    host.id = "host".into();
    host.locations = vec![Location::HOST, Location::PINNED];
    assert_curve_census(&contract, &[budget(1 << 30), host], &[0, 0, 0]);
}

#[test]
fn analytic_pool_solves_discontinuous_nonmonotone_and_zero_slope_pieces() {
    let mut contract = Linear {
        maximum: 64 * 12,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    contract
        .set_byte_extent(
            "kv",
            ByteCurve(vec![
                BytePiece {
                    first: 1,
                    last: 3,
                    bytes_at_first: 100,
                    bytes_per_page: 10,
                },
                BytePiece {
                    first: 4,
                    last: 6,
                    bytes_at_first: 1000,
                    bytes_per_page: 0,
                },
                BytePiece {
                    first: 7,
                    last: 12,
                    bytes_at_first: 50,
                    bytes_per_page: 10,
                },
            ]),
        )
        .unwrap();
    assert_curve_census(&contract, &[budget(1 << 30)], &[0, 0, 0]);
    let result = solve(request(), vec![budget(75)], &Prepared(contract.clone())).unwrap();
    assert_eq!(result.pool_tokens, 9 * 64);
    assert!(matches!(
        solve(
            Request {
                pool: Pin::Exact(10 * 64),
                ..request()
            },
            vec![budget(75)],
            &Prepared(contract)
        ),
        Err(Refusal::Capacity { .. })
    ));
    let mut fixed = Linear {
        maximum: 1024,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    fixed
        .set_byte_extent("kv", ByteCurve::fixed(75, 16))
        .unwrap();
    assert_eq!(
        solve(request(), vec![budget(75)], &Prepared(fixed))
            .unwrap()
            .pool_tokens,
        1024
    );
}

#[test]
fn exact_pins_propagate_compatibility_and_fixed_capacity_before_branching() {
    let mut contract = Linear {
        maximum: 256,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    contract.variables[0] = Variable::Attention {
        layer: 0,
        values: vec![
            LayerMode::Whole {
                gpu: 0,
                ffn: FfnMode::Owner,
            },
            LayerMode::Whole {
                gpu: 0,
                ffn: FfnMode::Split,
            },
        ],
    };
    contract.variables[1] = Variable::Prefill(vec![
        decisions(0).prefill,
        PrefillShape {
            rows: 2048,
            waves: 2,
        },
    ]);
    contract.variables[2] = Variable::Lanes(vec![1, 2]);
    contract.requires = vec![
        Requires {
            when: Choice {
                variable: 1,
                option: 1,
            },
            any_of: vec![Choice {
                variable: 2,
                option: 1,
            }],
        },
        Requires {
            when: Choice {
                variable: 2,
                option: 1,
            },
            any_of: vec![Choice {
                variable: 0,
                option: 1,
            }],
        },
    ];
    let mut impossible = Component::allocation(
        allocation("impossible", 4096),
        ByteCurve::fixed(4096, 4),
        false,
        4,
    );
    impossible.activation = Activation::choice(0, 0);
    contract.components.push(impossible);
    let result = solve(
        Request {
            prefill: Pin::Exact(PrefillShape {
                rows: 2048,
                waves: 2,
            }),
            ..request()
        },
        vec![budget(1024)],
        &Prepared(contract),
    )
    .unwrap();
    assert_eq!(
        (result.decisions.prefill.rows, result.decisions.lanes),
        (2048, 2)
    );
    assert!(matches!(
        result.decisions.attention[0],
        LayerMode::Whole {
            ffn: FfnMode::Split,
            ..
        }
    ));
    assert!(!result
        .facts
        .allocations
        .iter()
        .any(|a| a.id == "impossible"));
}

#[test]
fn analytical_solve_prepares_once_and_matches_exact_small_domain_oracle() {
    struct Counted<'a>(&'a std::cell::Cell<u64>, Contract);
    impl Recipe for Counted<'_> {
        fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
            self.0.set(self.0.get() + 1);
            Ok(self.1.clone())
        }
    }
    let count = std::cell::Cell::new(0);
    let contract = Linear {
        maximum: 64 * 8,
        fixed: 3,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    for available in 60..600 {
        let result = solve(
            request(),
            vec![budget(available)],
            &Counted(&count, contract.clone()),
        );
        let oracle = (1..=8)
            .filter(|&p| {
                evaluate(
                    &contract.facts(&[0, 0, 0], p).unwrap(),
                    &[budget(available)],
                )
                .is_ok()
            })
            .max();
        assert_eq!(
            result.ok().map(|p| p.pool_tokens / 64),
            oracle,
            "budget {available}"
        );
    }
    assert_eq!(count.get(), 540);
}

fn two_gpus(first: u64, second: u64) -> Vec<Budget> {
    let mut peer = budget(second);
    peer.id = "peer".into();
    peer.locations = vec![Location::coordinator(1)];
    vec![budget(first), peer]
}
fn profile_contract(maximum: u64) -> Contract {
    let mut contract = Contract::fixed(64, maximum, 2, facts(vec![]));
    contract.variables = vec![
        Variable::Attention {
            layer: 0,
            values: decisions(0).attention,
        },
        Variable::Prefill(vec![decisions(0).prefill]),
        Variable::Lanes(vec![1]),
    ];
    contract
}
fn add_profile(
    contract: &mut Contract,
    layers: Vec<usize>,
    destination: Vec<u8>,
    counts: Vec<usize>,
    bytes: &[u64],
) -> usize {
    assert_eq!(counts.len(), bytes.len());
    let variable = contract.variables.len();
    for &gpu in &destination {
        for (option, &bytes) in bytes.iter().enumerate().filter(|(_, bytes)| **bytes != 0) {
            let mut owner = allocation(&format!("profile.{variable}.{option}.{gpu}"), bytes);
            owner.location = Location::coordinator(gpu);
            let mut component = Component::allocation(
                owner,
                ByteCurve::fixed(bytes, contract.maximum_pages),
                false,
                contract.maximum_pages,
            );
            component.activation = Activation::choice(variable, option);
            contract.components.push(component);
        }
    }
    contract.variables.push(Variable::Onboard {
        layers,
        destination,
        counts,
    });
    variable
}

fn home_contract() -> (Contract, Vec<Budget>) {
    let mut contract = profile_contract(4);
    let remote = contract.variables.len();
    contract.variables.push(Variable::ExpertHome {
        layers: vec![3, 4, 5, 6],
        rank_sets: vec![vec![0, 1], vec![2, 3]],
    });
    let local = add_profile(&mut contract, vec![3, 4, 5], vec![0, 1],
        vec![0, 1, 2, 3], &[0, 32, 64, 96]);
    for gpu in 0..2 {
        let mut owner = allocation(&format!("kv.{gpu}"), 16);
        owner.location = Location::coordinator(gpu);
        let curve = ByteCurve::linear(16, 16, 4);
        let mut component = Component::allocation(owner, curve.clone(), false, 4);
        component.byte_extent(curve).unwrap();
        contract.components.push(component);
    }
    for (option, ranks) in [vec![0, 1], vec![2, 3]].into_iter().enumerate() {
        for count in 0..=3 {
            for &rank in &ranks {
                let mut owner = allocation(&format!("remote.{option}.{count}.{rank}"), (4 - count) * 32);
                owner.location = Location { device: Device::Spark(rank), domain: Domain::Device };
                let mut component = Component::allocation(owner.clone(), ByteCurve::fixed(owner.bytes, 4), false, 4);
                component.activation = Activation { clauses: vec![vec![
                    Choice { variable: remote, option },
                    Choice { variable: local, option: count as usize },
                ]] };
                contract.components.push(component);
            }
        }
    }
    let embedding = contract.variables.len();
    contract.variables.push(Variable::Embedding(vec![EmbeddingHome::Device(0), EmbeddingHome::HostPinned]));
    for (option, home) in [EmbeddingHome::Device(0), EmbeddingHome::HostPinned].into_iter().enumerate() {
        let mut owner = allocation(&format!("embedding.{option}"), 64);
        owner.location = home.location();
        owner.lifetime = Lifetime::RESIDENT;
        let mut component = Component::allocation(owner, ByteCurve::fixed(64, 4), false, 4);
        component.activation = Activation::choice(embedding, option);
        contract.components.push(component);
    }
    let mut budgets = two_gpus(160, 160);
    for budget in &mut budgets { budget.physical_bytes = 96 << 30; }
    for rank in 0..4 {
        let mut b = budget(if rank < 2 { 50 } else { 100 });
        b.id = format!("spark.{rank}");
        b.locations = vec![Location { device: Device::Spark(rank), domain: Domain::Device }];
        budgets.push(b);
    }
    let mut host = budget(100);
    host.id = "host".into();
    host.locations = vec![Location::PINNED];
    budgets.push(host);
    (contract, budgets)
}

#[test]
fn single_copy_homes_are_selected_jointly_with_pool_and_per_gpu_headroom() {
    for kind in 0..3 {
        let domain = |values| match kind {
            0 => Variable::DraftHome(values),
            1 => Variable::HeadHome(values),
            _ => Variable::IntakeHome(values),
        };
        let home = |decisions: &Decisions| match kind {
            0 => decisions.draft_home,
            1 => decisions.head_home,
            _ => decisions.intake_home,
        };
        let mut contract = Linear::new().prepare(&request()).unwrap();
        contract.gpus = 2;
        let mut peer = allocation("peer.kv", 64);
        peer.location = Location::coordinator(1);
        let curve = ByteCurve::linear(64, 64, contract.maximum_pages);
        let mut component = Component::allocation(peer, curve.clone(), false, contract.maximum_pages);
        component.byte_extent(curve).unwrap();
        contract.components.push(component);
        let variable = contract.variables.len();
        contract.variables.push(domain(vec![0, 1]));
        for (option, gpu) in [0, 1].into_iter().enumerate() {
            let mut owner = allocation("draft", 64);
            owner.location = Location::coordinator(gpu);
            let mut component = Component::allocation(owner, ByteCurve::fixed(64, contract.maximum_pages), false, contract.maximum_pages);
            component.activation = Activation::choice(variable, option);
            contract.components.push(component);
        }
        let budgets = two_gpus(100, 1000);
        let selected = solve(request(), budgets.clone(), &Prepared(contract.clone())).unwrap();
        assert_eq!(home(&selected.decisions), Some(1));
        let equal = solve(request(), two_gpus(1000, 1000), &Prepared(contract.clone())).unwrap();
        assert_eq!(home(&equal.decisions), Some(0));
        for values in [vec![], vec![2], vec![0, 0]] {
            let mut invalid = contract.clone();
            invalid.variables[variable] = domain(values);
            assert!(solve(request(), budgets.clone(), &Prepared(invalid)).is_err());
        }
        let mut duplicate = contract.clone();
        duplicate.variables.push(domain(vec![0]));
        assert!(matches!(solve(request(), budgets, &Prepared(duplicate)), Err(Refusal::InvalidFact(_))));
        assert!(matches!(solve(request(), vec![budget(1000)], &Prepared(contract)), Err(Refusal::InvalidFact(_))));
    }
}

#[test]
fn single_copy_homes_balance_jointly_and_charge_cross_gpu_owners() {
    let mut contract = Linear::new().prepare(&request()).unwrap();
    contract.gpus = 2;
    contract.components.clear();
    let first = contract.variables.len();
    contract.variables.extend([Variable::DraftHome(vec![0, 1]), Variable::HeadHome(vec![0, 1]),
        Variable::IntakeHome(vec![0, 1])]);
    for (index, bytes) in [60, 30, 20].into_iter().enumerate() {
        for (option, gpu) in [0, 1].into_iter().enumerate() {
            let mut owner = allocation(&format!("owner.{index}"), bytes);
            owner.location = Location::coordinator(gpu);
            let mut component = Component::allocation(owner, ByteCurve::fixed(bytes, contract.maximum_pages),
                false, contract.maximum_pages);
            component.activation = Activation::choice(first + index, option);
            contract.components.push(component);
        }
    }
    let selected = solve(request(), two_gpus(100, 100), &Prepared(contract.clone())).unwrap();
    assert_eq!((selected.decisions.draft_home, selected.decisions.head_home, selected.decisions.intake_home),
        (Some(0), Some(1), Some(1)));
    // A costly remote bridge is a physical owner, not a post-solve adjustment.
    let mut bridge = allocation("remote.head.bridge", 60);
    bridge.location = Location::coordinator(1);
    let mut component = Component::allocation(bridge, ByteCurve::fixed(60, contract.maximum_pages),
        false, contract.maximum_pages);
    component.activation = Activation::choice(first + 1, 1);
    contract.components.push(component);
    let selected = solve(request(), two_gpus(100, 100), &Prepared(contract)).unwrap();
    assert_eq!((selected.decisions.draft_home, selected.decisions.head_home, selected.decisions.intake_home),
        (Some(1), Some(0), Some(0)));
}

#[test]
fn embedding_and_remote_rank_domains_share_pool_first_constraints_and_exact_profiles() {
    let (mut contract, budgets) = home_contract();
    contract.add_driver_page_tables();
    let recipe = Prepared(contract.clone());
    let req = Request { onboard: Pin::Exact(2), ..request() };
    let resolved = solve(req.clone(), budgets.clone(), &recipe).unwrap();
    assert_eq!(resolved.pool_tokens, 256);
    assert_eq!(resolved.decisions.embedding, Some(EmbeddingHome::HostPinned));
    assert_eq!(resolved.decisions.onboard_layers, vec![vec![3, 4]; 2]);
    assert_eq!(resolved.decisions.expert_homes, BTreeMap::from([
        (3, ExpertDestination::Coordinator { gpus: vec![0, 1] }),
        (4, ExpertDestination::Coordinator { gpus: vec![0, 1] }),
        (5, ExpertDestination::Sparks { ranks: vec![2, 3] }),
        (6, ExpertDestination::Sparks { ranks: vec![2, 3] }),
    ]));
    assert_eq!(resolved.decisions.onboard(), 2);
    assert_eq!(resolved.remaining("spark.2", Phase::Ready), Some(36));
    let automatic = solve(request(), budgets.clone(), &recipe).unwrap();
    // Three onboard layers + KV exactly consumed 160 B before the portable
    // host embedding's 1 B/context mapping cost; that choice now needs 161 B.
    assert_eq!((automatic.pool_tokens, automatic.decisions.onboard()), (256, 2));
    let mut mapped = contract.clone();
    mapped.add_driver_page_tables();
    assert!(matches!(evaluate(&mapped.facts(&[0, 0, 0, 1, 3, 1], 4).unwrap(), &budgets),
        Err(Refusal::Capacity { required: 161, available: 160, .. })));
    // Fixing a home is an initial compatibility restriction. An exact pool
    // refuses rather than silently changing the home or dropping a pool page.
    let mut restricted = contract.clone();
    restricted.requires.push(Requires { when: Choice { variable: 5, option: 1 }, any_of: vec![] });
    assert!(matches!(solve(Request { pool: Pin::Exact(256), ..req.clone() },
        budgets.clone(), &Prepared(restricted.clone())), Err(Refusal::Capacity { .. })));
    let smaller = solve(req.clone(), budgets.clone(), &Prepared(restricted)).unwrap();
    assert_eq!((smaller.pool_tokens, smaller.decisions.onboard()), (128, 2));
    let mut restricted = contract.clone();
    restricted.requires.push(Requires { when: Choice { variable: 3, option: 1 }, any_of: vec![] });
    assert!(matches!(solve(req, budgets.clone(), &Prepared(restricted)),
        Err(Refusal::FullContextShortfall { tokens: 0, limiting, .. })
            if matches!(*limiting, Refusal::Capacity { .. })));
    // Resolved local homes override remote fallback independent of variable order.
    let mut reversed = contract.clone();
    reversed.variables.swap(3, 4);
    let selected = [0, 0, 0, 1, 2, 1];
    let reordered = [0, 0, 0, 2, 1, 1];
    assert_eq!(contract.decisions(&selected).unwrap(), reversed.decisions(&reordered).unwrap());
    let encoded = serde_json::to_string(&resolved.decisions).unwrap();
    assert_eq!(serde_json::from_str::<Decisions>(&encoded).unwrap(), resolved.decisions);
}

#[test]
fn home_domains_refuse_duplicate_destinations_and_unbudgeted_physical_locations() {
    let (contract, budgets) = home_contract();
    for malformed in 0..11 {
        let mut bad = contract.clone();
        match malformed {
            0 => bad.variables[3] = Variable::ExpertHome { layers: vec![3, 3], rank_sets: vec![vec![0]] },
            1 => bad.variables[3] = Variable::ExpertHome { layers: vec![3], rank_sets: vec![vec![]] },
            2 => bad.variables[3] = Variable::ExpertHome { layers: vec![3], rank_sets: vec![vec![1, 0]] },
            3 => bad.variables[3] = Variable::ExpertHome { layers: vec![3], rank_sets: vec![vec![0], vec![0]] },
            4 => bad.variables[5] = Variable::Embedding(vec![EmbeddingHome::Device(2)]),
            5 => bad.variables[5] = Variable::Embedding(vec![EmbeddingHome::HostPinned, EmbeddingHome::HostPinned]),
            6 => bad.variables.push(Variable::Embedding(vec![EmbeddingHome::Device(0)])),
            7 => bad.variables.push(Variable::ExpertHome { layers: vec![6], rank_sets: vec![vec![0]] }),
            8 => bad.variables[4] = Variable::Onboard { layers: vec![3], destination: vec![1, 0], counts: vec![1] },
            9 => bad.variables[3] = Variable::ExpertHome { layers: vec![3], rank_sets: vec![vec![0, 0]] },
            10 => bad.variables[4] = Variable::Onboard { layers: vec![3], destination: vec![0, 0], counts: vec![1] },
            _ => unreachable!(),
        }
        assert!(matches!(bad.validate(), Err(Refusal::InvalidFact(_))), "case {malformed}");
    }
    for missing in ["host", "spark.3", "peer"] {
        let mut bad = budgets.clone();
        bad.retain(|b| b.id != missing);
        assert!(matches!(solve(request(), bad, &Prepared(contract.clone())), Err(Refusal::InvalidFact(_))));
    }
}

#[test]
fn shared_onboarding_balances_ordered_profiles_and_charges_workspace_once() {
    let mut contract = profile_contract(4);
    for (gpu, layers) in [(0, vec![0, 1, 2, 3]), (1, vec![3, 2, 1, 0])] {
        let variable = add_profile(
            &mut contract,
            layers,
            vec![gpu],
            (0..=4).collect(),
            &[0, 32, 64, 96, 128],
        );
        let mut owner = allocation(&format!("workspace.{gpu}"), 16);
        owner.location = Location::coordinator(gpu);
        let mut term = Component::allocation(owner, ByteCurve::fixed(16, 4), false, 4);
        term.activation = Activation {
            clauses: (1..=4)
                .map(|option| vec![Choice { variable, option }])
                .collect(),
        };
        contract.components.push(term);
    }
    let req = Request {
        pool: Pin::Exact(256),
        ..request()
    };
    let result = solve(req.clone(), two_gpus(128, 128), &Prepared(contract.clone())).unwrap();
    assert_eq!(
        result.decisions.onboard_layers,
        vec![vec![0, 1], vec![2, 3]]
    );
    assert_eq!(result.decisions.onboard(), 4);
    assert_eq!(
        result
            .facts
            .allocations
            .iter()
            .filter(|a| a.id.starts_with("workspace"))
            .count(),
        2
    );
    add_profile(&mut contract, vec![4], vec![0, 1], vec![1], &[32]);
    let result = solve(
        Request {
            onboard: Pin::Exact(3),
            ..req
        },
        two_gpus(128, 128),
        &Prepared(contract),
    )
    .unwrap();
    assert_eq!(result.decisions.onboard(), 3);
    assert!(result
        .decisions
        .onboard_layers
        .iter()
        .all(|layers| layers.contains(&4)));
}

#[test]
fn required_onboarding_domains_survive_target_propagation() {
    let mut contract = profile_contract(8);
    let variable = add_profile(&mut contract, vec![7], vec![1], vec![0, 1], &[0, 32]);
    let mut remote = allocation("remote.experts", 300);
    remote.location = Location {
        device: Device::Spark(0),
        domain: Domain::Device,
    };
    let mut remote = Component::allocation(remote, ByteCurve::fixed(300, 8), false, 8);
    remote.activation = Activation::choice(variable, 0);
    contract.components.push(remote);
    contract.requires.push(Requires {
        when: Choice {
            variable: 0,
            option: 0,
        },
        any_of: vec![Choice {
            variable,
            option: 1,
        }],
    });
    let mut spark = budget(128);
    spark.id = "spark".into();
    spark.locations = vec![Location {
        device: Device::Spark(0),
        domain: Domain::Device,
    }];
    let mut budgets = two_gpus(128, 128);
    budgets.push(spark);
    let result = solve(request(), budgets, &Prepared(contract)).unwrap();
    assert_eq!(result.pool_tokens, 8 * 64);
    assert_eq!(result.decisions.onboard_layers, vec![vec![], vec![7]]);
    assert!(!result
        .facts
        .allocations
        .iter()
        .any(|a| a.id == "remote.experts"));
}

#[test]
fn below_target_onboards_on_nonbinding_device_without_spending_one_page() {
    let mut contract = profile_contract(8);
    let mut kv =
        Component::allocation(allocation("kv", 64), ByteCurve::linear(64, 64, 8), false, 8);
    kv.byte_extent(ByteCurve::linear(64, 64, 8)).unwrap();
    contract.components.push(kv);
    add_profile(
        &mut contract,
        vec![0, 1],
        vec![0],
        vec![0, 1, 2],
        &[0, 1, 2],
    );
    add_profile(
        &mut contract,
        vec![2, 3],
        vec![1],
        vec![0, 1, 2],
        &[0, 17, 49],
    );
    let budgets = two_gpus(256, 50);
    let automatic = solve(request(), budgets.clone(), &Prepared(contract.clone())).unwrap();
    let pinned = solve(
        Request {
            pool: Pin::Exact(256),
            ..request()
        },
        budgets,
        &Prepared(contract),
    )
    .unwrap();
    assert_eq!(automatic.pool_tokens, 256);
    assert_eq!(automatic.decisions.onboard_layers, vec![vec![], vec![2, 3]]);
    assert_eq!(automatic.decisions, pinned.decisions);
}

#[test]
fn exact_onboarding_reaches_target_before_maximizing_physical_capacity() {
    const TARGET: u64 = 2 << 20;
    let maximum = TARGET * 2 / 64;
    let mut contract = profile_contract(maximum);
    contract.variables[0] = Variable::Attention {
        layer: 0,
        values: vec![
            LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
            LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner },
        ],
    };
    for (option, slopes) in [[64, 192], [64, 64]].into_iter().enumerate() {
        for (gpu, slope) in slopes.into_iter().enumerate() {
            let mut owner = allocation(&format!("kv.{option}.{gpu}"), slope);
            owner.location = Location::coordinator(gpu as u8);
            let curve = ByteCurve::linear(slope, slope, maximum);
            let mut component = Component::allocation(owner, curve.clone(), false, maximum);
            component.byte_extent(curve).unwrap();
            component.activation = Activation::choice(0, option);
            contract.components.push(component);
        }
    }
    add_profile(&mut contract, vec![0, 1], vec![0, 1], vec![0, 1, 2],
        &[0, 1 << 20, 2 << 20]);
    let mut budgets = two_gpus(5 << 20, 5 << 20);
    for budget in &mut budgets {
        budget.physical_bytes = 96 << 30;
    }
    let recipe = Prepared(contract);
    let pinned = Request { onboard: Pin::Exact(1), ..request() };
    let automatic = solve(pinned.clone(), budgets.clone(), &recipe).unwrap();
    let explicit = solve(Request { pool: Pin::Exact(TARGET), ..pinned.clone() },
        budgets.clone(), &recipe).unwrap();
    assert_eq!(automatic.pool_tokens, TARGET);
    assert_eq!(automatic.decisions.onboard(), 1);
    assert_eq!(automatic.decisions, explicit.decisions);
    assert_eq!(automatic.decisions.attention,
        vec![LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }]);
    assert_eq!(solve(request(), budgets.clone(), &recipe).unwrap().pool_tokens, TARGET);
    // The target-fitting owner also fits above the target. That is spare
    // capacity, not a reason to overfill or discard its automatic-pool plan.
    assert_eq!(solve(Request { pool: Pin::Exact(3 << 20), ..pinned }, budgets, &recipe)
        .unwrap().pool_tokens, 3 << 20);
}

#[test]
fn step_exact_fifteen_inventory_replay_preserves_feasible_two_million_target() {
    // Frozen CPU plans from #776 comment 6587874033, source 140e7f802,
    // Step EXL3 K4+Core-K6 revision 83d47a384eabc07ea6fcfaac4779b0e5f013ce89:
    // 2 PRO + 6 Sparks, C16, MTP3, 1M context, prefill 2752x3. This replays
    // three observed inventories, not a complete Step recipe or live receipts.
    const TARGET: u64 = 2 << 20;
    const PINNED_EXPERTS: u64 = 26_371_720_568;
    const AUTO_EXPERTS: u64 = 31_417_468_280;
    let witnesses = [
        (7, 1_996_608, 15, [81_448_123_784, 99_238_178_520], [702_464, 1_605_632]),
        (9, TARGET, 18, [95_671_429_768, 98_732_386_776], [903_168, 1_404_928]),
        (10, TARGET, 15, [94_662_675_720, 89_649_645_400], [1_003_520, 1_304_576]),
    ];
    let maximum = TARGET * 2 / 64;
    let mut contract = Contract::fixed(64, maximum, 2, facts(vec![]));
    for layer in 0..23 {
        let whole = |gpu| LayerMode::Whole { gpu, ffn: FfnMode::Owner };
        contract.variables.push(Variable::Attention {
            layer,
            values: if layer < 7 { vec![whole(0)] }
                else if layer < 10 { vec![whole(0), whole(1)] }
                else { vec![whole(1)] },
        });
    }
    // Keep precisely the observed contiguous cuts 7, 9 and 10, while leaving
    // ownership independent of the onboarding pin and the pool request.
    for (when, required) in [(7, 8), (8, 7), (9, 8)] {
        contract.requires.push(Requires {
            when: Choice { variable: when, option: 0 },
            any_of: vec![Choice { variable: required, option: 0 }],
        });
    }
    contract.variables.push(Variable::Prefill(vec![PrefillShape { rows: 2752, waves: 3 }]));
    contract.variables.push(Variable::Lanes(vec![1]));
    add_profile(&mut contract, (3..91).collect(), vec![0, 1], vec![15, 18],
        &[PINNED_EXPERTS, AUTO_EXPERTS]);
    for &(cut, tokens, onboard, totals, slopes) in &witnesses {
        let activation = match cut {
            7 => Activation::choice(7, 1),
            9 => Activation { clauses: vec![vec![
                Choice { variable: 7, option: 0 }, Choice { variable: 9, option: 1 },
            ]] },
            10 => Activation::choice(9, 0),
            _ => unreachable!(),
        };
        for gpu in 0..2 {
            let expert = if onboard == 15 { PINNED_EXPERTS } else { AUTO_EXPERTS };
            let fixed = totals[gpu] - tokens / 64 * slopes[gpu] - expert;
            let curve = ByteCurve::linear(fixed + slopes[gpu], slopes[gpu], maximum);
            let mut owner = allocation(&format!("step.frozen.{cut}.{gpu}"), fixed + slopes[gpu]);
            owner.location = Location::coordinator(gpu as u8);
            let mut component = Component::allocation(owner, curve.clone(), false, maximum);
            component.byte_extent(curve).unwrap();
            component.activation = activation.clone();
            contract.components.push(component);
        }
    }
    let mut budgets = two_gpus(101_973_491_712, 101_973_491_712);
    for budget in &mut budgets {
        budget.baseline = Baseline::Forecast {
            occupied: 586_416_128,
            source: "frozen Step CPU plan pre-weight baseline".into(),
            assumed: false,
        };
        budget.headroom = 2 << 30;
    }
    for &(cut, tokens, onboard, totals, _) in &witnesses {
        let mut selected = (0..23).map(|layer|
            usize::from((7..10).contains(&layer) && layer >= cut)).collect::<Vec<_>>();
        selected.extend([0, 0, usize::from(onboard == 18)]);
        let phases = evaluate(&contract.facts(&selected, tokens / 64).unwrap(), &budgets).unwrap();
        let owned = phases.iter().filter(|p| p.phase == Phase::Ready)
            .map(|p| p.owned).collect::<Vec<_>>();
        assert_eq!(owned, totals, "CSA cut {cut} preserves the real plan's rank totals");
        if cut == 7 {
            assert!(matches!(evaluate(&contract.facts(&selected, tokens / 64 + 1).unwrap(), &budgets),
                Err(Refusal::Capacity { .. })));
        } else {
            evaluate(&contract.facts(&selected, TARGET / 64 + 1).unwrap(), &budgets).unwrap();
        }
    }
    let recipe = Prepared(contract);
    let req = Request { context: (1 << 20).into(), onboard: Pin::Exact(15), ..request() };
    let pinned = solve(req.clone(), budgets.clone(), &recipe).unwrap();
    let explicit = solve(Request { pool: Pin::Exact(TARGET), ..req.clone() },
        budgets.clone(), &recipe).unwrap();
    let automatic = solve(Request { onboard: Pin::Auto, ..req }, budgets, &recipe).unwrap();
    assert_eq!((pinned.pool_tokens, pinned.decisions.onboard()), (TARGET, 15));
    assert_eq!((automatic.pool_tokens, automatic.decisions.onboard()), (TARGET, 18));
    assert_eq!(pinned.decisions, explicit.decisions);
    // Active-phase balance may improve upon the old explicit-2M witness's cut
    // 10. The required invariant is the feasible target, not its old owner cut.
    assert_eq!(pinned.decisions.attention.iter()
        .filter(|mode| matches!(mode, LayerMode::Whole { gpu: 0, .. })).count(), 9);
    assert_eq!(pinned.pool_tokens, automatic.pool_tokens);
}

#[test]
fn flash_spark_free_auto_all_inventory_replay_stops_at_target_and_retains_slack() {
    // Real requested-zero-Spark plan from #776 comment 6589996421, a2a6de7d0:
    // Flash EXLR8 k4 revision 867580abd28d142bb4c56218bb81a752b7dea988,
    // two PROs, spec/media off, context 1M, prefill 4096x1. Preserve this
    // inventory, not the legacy policy that fills past the automatic target.
    const TARGET: u64 = 2 << 20;
    const LEGACY_POOL: u64 = 2_761_216;
    const SLOPE: u64 = 789_892;
    const EXPERTS: u64 = 79_800_852_992;
    const CAPACITY: u64 = 99_826_008_064;
    const TOTALS: [u64; 2] = [99_825_322_576, 99_489_193_600];
    let maximum = (4 << 20) / 256;
    let mut contract = Contract::fixed(256, maximum, 2, facts(vec![]));
    for layer in 0..45 {
        contract.variables.push(Variable::Attention { layer, values: vec![
            if layer >= 3 && (layer - 3) % 4 == 0 { LayerMode::ContextSplit }
            else { LayerMode::HeadSplit },
        ] });
    }
    contract.variables.push(Variable::Prefill(vec![PrefillShape { rows: 4096, waves: 1 }]));
    contract.variables.push(Variable::Lanes(vec![1]));
    // Spark-free removes the remote home from the domain; Auto is still the
    // same pool-first objective, with the only legal profile owning all 42.
    add_profile(&mut contract, (3..45).collect(), vec![0, 1], vec![42], &[EXPERTS]);
    for (gpu, total) in TOTALS.into_iter().enumerate() {
        let fixed = total - LEGACY_POOL / 256 * SLOPE - EXPERTS;
        let curve = ByteCurve::linear(fixed + SLOPE, SLOPE, maximum);
        let mut owner = allocation(&format!("flash.frozen.{gpu}"), fixed + SLOPE);
        owner.location = Location::coordinator(gpu as u8);
        let mut component = Component::allocation(owner, curve.clone(), false, maximum);
        component.byte_extent(curve).unwrap();
        contract.components.push(component);
    }
    let mut budgets = two_gpus(CAPACITY, CAPACITY);
    for budget in &mut budgets { budget.physical_bytes = 96 << 30; }
    let selected = vec![0; contract.variables.len()];
    let legacy = evaluate(&contract.facts(&selected, LEGACY_POOL / 256).unwrap(), &budgets).unwrap();
    assert_eq!(legacy.iter().filter(|p| p.phase == Phase::Ready).map(|p| p.owned)
        .collect::<Vec<_>>(), TOTALS);
    assert!(matches!(evaluate(&contract.facts(&selected, LEGACY_POOL / 256 + 1).unwrap(), &budgets),
        Err(Refusal::Capacity { .. })));
    let recipe = Prepared(contract);
    let req = Request { context: (1 << 20).into(), ..request() };
    let automatic = solve(req.clone(), budgets.clone(), &recipe).unwrap();
    let pinned = solve(Request { onboard: Pin::Exact(42), ..req.clone() }, budgets.clone(), &recipe).unwrap();
    let explicit = solve(Request { pool: Pin::Exact(TARGET), ..req.clone() }, budgets.clone(), &recipe).unwrap();
    assert_eq!(automatic.pool_tokens, TARGET);
    assert_eq!(automatic.decisions.onboard(), 42);
    assert_eq!(automatic.decisions.onboard_layers, vec![(3..45).collect::<Vec<_>>(); 2]);
    assert_eq!(automatic.decisions, pinned.decisions);
    assert_eq!(automatic.decisions, explicit.decisions);
    assert_eq!(pinned.pool_tokens, TARGET);
    for (gpu, slack) in [2_049_665_336, 2_385_794_312].into_iter().enumerate() {
        assert_eq!(automatic.remaining(&budgets[gpu].id, Phase::Ready), Some(slack));
    }
    assert_eq!(solve(Request { pool: Pin::Exact(LEGACY_POOL), ..req }, budgets, &recipe)
        .unwrap().pool_tokens, LEGACY_POOL);
}

#[test]
fn exact_completion_prefers_two_whole_layers_over_one_tp2_layer() {
    let mut contract = profile_contract(4);
    add_profile(&mut contract, vec![0], vec![0, 1], vec![0, 1], &[0, 50]);
    add_profile(&mut contract, vec![0], vec![0], vec![0, 1], &[0, 60]);
    add_profile(&mut contract, vec![1], vec![1], vec![0, 1], &[0, 60]);
    let result = solve(request(), two_gpus(100, 100), &Prepared(contract)).unwrap();
    assert_eq!(result.decisions.onboard_layers, vec![vec![0], vec![1]]);
    assert_eq!(result.decisions.onboard(), 2);
}

#[test]
fn active_gpu_balance_is_not_masked_by_host_or_load_slack() {
    let mut contract = profile_contract(4);
    add_profile(
        &mut contract,
        vec![0, 1, 2, 3],
        vec![0],
        (0..=4).collect(),
        &[0, 32, 64, 96, 128],
    );
    add_profile(
        &mut contract,
        vec![3, 2, 1, 0],
        vec![1],
        (0..=4).collect(),
        &[0, 32, 64, 96, 128],
    );
    for gpu in 0..2 {
        let mut load = allocation(&format!("load.{gpu}"), 128);
        load.location = Location::coordinator(gpu);
        load.lifetime = Lifetime {
            first: Phase::Load,
            last: Phase::Load,
        };
        contract.components.push(Component::allocation(
            load,
            ByteCurve::fixed(128, 4),
            false,
            4,
        ));
    }
    let mut budgets = two_gpus(128, 128);
    let mut host = budget(1);
    host.id = "host".into();
    host.locations = vec![Location::HOST, Location::PINNED];
    budgets.push(host);
    let first = solve(request(), budgets.clone(), &Prepared(contract.clone())).unwrap();
    assert_eq!(first.decisions.onboard_layers, vec![vec![0, 1], vec![2, 3]]);
    for _ in 0..3 {
        assert_eq!(
            solve(request(), budgets.clone(), &Prepared(contract.clone())).unwrap(),
            first
        );
    }
}

#[test]
fn capacity_shaving_keeps_resource_failure_distinct_from_pin_conflict() {
    let mut contract = profile_contract(1);
    contract.variables[2] = Variable::Lanes(vec![1, 2]);
    for gpu in 0..2 {
        let mut owner = allocation(&format!("choice.{gpu}"), 65);
        owner.location = Location::coordinator(gpu);
        let mut component = Component::allocation(owner, ByteCurve::fixed(65, 1), false, 1);
        component.activation = Activation::choice(2, gpu as usize);
        contract.components.push(component);
    }
    assert!(matches!(
        solve(request(), two_gpus(64, 64), &Prepared(contract.clone())),
        Err(Refusal::FullContextShortfall { tokens: 0, limiting, .. })
            if matches!(*limiting, Refusal::Capacity { .. })
    ));
    assert_eq!(
        solve(
            Request {
                onboard: Pin::Exact(1),
                ..request()
            },
            two_gpus(64, 64),
            &Prepared(contract)
        )
        .unwrap_err(),
        Refusal::PinConflict
    );
}

#[test]
fn correlated_count_profiles_match_exhaustive_full_score_oracle() {
    let mut contract = profile_contract(6);
    contract.variables[0] = Variable::Attention {
        layer: 0,
        values: vec![
            LayerMode::Whole {
                gpu: 0,
                ffn: FfnMode::Owner,
            },
            LayerMode::Whole {
                gpu: 1,
                ffn: FfnMode::Owner,
            },
        ],
    };
    let left = add_profile(
        &mut contract,
        vec![0, 1],
        vec![0],
        vec![0, 1, 2],
        &[0, 31, 79],
    );
    let right = add_profile(
        &mut contract,
        vec![1, 0],
        vec![1],
        vec![0, 1, 2],
        &[0, 23, 63],
    );
    let tp2 = add_profile(&mut contract, vec![2], vec![0, 1], vec![0, 1], &[0, 17]);
    contract.requires.push(Requires {
        when: Choice {
            variable: 0,
            option: 1,
        },
        any_of: vec![
            Choice {
                variable: right,
                option: 1,
            },
            Choice {
                variable: right,
                option: 2,
            },
        ],
    });
    let curve = ByteCurve(vec![
        BytePiece {
            first: 1,
            last: 3,
            bytes_at_first: 20,
            bytes_per_page: 19,
        },
        BytePiece {
            first: 4,
            last: 6,
            bytes_at_first: 12,
            bytes_per_page: 13,
        },
    ]);
    for gpu in 0..2 {
        let mut owner = allocation("kv", 20);
        owner.location = Location::coordinator(gpu);
        let mut component = Component::allocation(owner, curve.clone(), false, 6);
        component.byte_extent(curve.clone()).unwrap();
        component.activation = Activation::choice(0, gpu as usize);
        contract.components.push(component);
    }
    let mut joint = Component::allocation(
        allocation("joint.scratch", 27),
        ByteCurve::fixed(27, 6),
        false,
        6,
    );
    joint.activation = Activation {
        clauses: vec![vec![
            Choice {
                variable: left,
                option: 1,
            },
            Choice {
                variable: tp2,
                option: 1,
            },
        ]],
    };
    contract.components.push(joint);
    for a in [40, 70, 100] {
        for b in [35, 65, 95] {
            for onboard in [Pin::Auto, Pin::Exact(0), Pin::Exact(2), Pin::Exact(3)] {
                for pool in [Pin::Auto, Pin::Exact(128), Pin::Exact(384)] {
                    let budgets = two_gpus(a, b);
                    let req = Request {
                        pool,
                        onboard,
                        ..request()
                    };
                    let mut oracle = None;
                    for attention in 0..2 {
                        for left_count in 0..3 {
                            for right_count in 0..3 {
                                for tp_count in 0..2 {
                                    let assignment =
                                        vec![attention, 0, 0, left_count, right_count, tp_count];
                                    if contract.requires.iter().any(|r| {
                                        assignment[r.when.variable] == r.when.option
                                            && !r
                                                .any_of
                                                .iter()
                                                .any(|c| assignment[c.variable] == c.option)
                                    }) {
                                        continue;
                                    }
                                    let Ok(decisions) = contract.decisions(&assignment) else {
                                        continue;
                                    };
                                    if matches!(onboard, Pin::Exact(n) if decisions.onboard() != n)
                                    {
                                        continue;
                                    }
                                    for pages in 1..=6 {
                                        if matches!(pool, Pin::Exact(n) if pages * 64 != n) {
                                            continue;
                                        }
                                        let Ok(phases) = evaluate(
                                            &contract.facts(&assignment, pages).unwrap(),
                                            &budgets,
                                        ) else {
                                            continue;
                                        };
                                        let active = phases
                                            .iter()
                                            .filter(|p| matches!(p.phase, Phase::Warmup | Phase::Ready | Phase::Growth))
                                            .map(|p| p.available - p.owned - p.opaque)
                                            .min()
                                            .unwrap();
                                        let global = phases
                                            .iter()
                                            .map(|p| p.available - p.owned - p.opaque)
                                            .min()
                                            .unwrap();
                                        let score = (pages, decisions.onboard(), active, global);
                                        if oracle.as_ref().is_none_or(
                                            |(old, choices): &(
                                                (u64, u64, u64, u64),
                                                Vec<usize>,
                                            )| {
                                                score > *old
                                                    || (score == *old && assignment < *choices)
                                            },
                                        ) {
                                            oracle = Some((score, assignment.clone()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let result = solve(req.clone(), budgets, &Prepared(contract.clone()));
                    match oracle {
                        Some((score, assignment)) => {
                            let result =
                                result.unwrap_or_else(|e| panic!("{a}/{b} {req:?}: {e:?}"));
                            assert_eq!(result.pool_tokens / 64, score.0, "{a}/{b} {req:?}");
                            assert_eq!(
                                result.decisions,
                                contract.decisions(&assignment).unwrap(),
                                "{a}/{b} {req:?}"
                            );
                        }
                        None => assert!(result.is_err(), "{a}/{b} {req:?}"),
                    }
                }
            }
        }
    }
}

#[test]
fn malformed_curves_and_unrepresentable_recipes_are_typed_refusals() {
    let mut contract = Linear {
        maximum: 256,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    contract
        .set_byte_extent(
            "kv",
            ByteCurve(vec![BytePiece {
                first: 2,
                last: 4,
                bytes_at_first: 1,
                bytes_per_page: 1,
            }]),
        )
        .unwrap();
    assert!(matches!(
        solve(request(), vec![budget(4096)], &Prepared(contract)),
        Err(Refusal::InvalidFact(_))
    ));
    struct Unsupported;
    impl Recipe for Unsupported {
        fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
            Err(Refusal::Unsupported)
        }
    }
    assert_eq!(
        solve(request(), vec![budget(4096)], &Unsupported).unwrap_err(),
        Refusal::Unsupported
    );
}

#[test]
fn pool_pins_are_exact_and_context_is_never_clamped() {
    let recipe = Linear::new();
    for (pin, expected) in [
        (
            65,
            Refusal::MisalignedPool {
                tokens: 65,
                page_rows: 64,
            },
        ),
        (
            0,
            Refusal::PoolBelowContext {
                tokens: 0,
                context: 64,
            },
        ),
    ] {
        assert_eq!(
            solve(
                Request {
                    pool: Pin::Exact(pin),
                    ..request()
                },
                vec![budget(1 << 30)],
                &recipe
            )
            .unwrap_err(),
            expected
        );
    }
    let exact = solve(
        Request {
            pool: Pin::Exact(256),
            ..request()
        },
        vec![budget(1 << 30)],
        &recipe,
    )
    .unwrap();
    assert_eq!(exact.pool_tokens, 256);
    assert!(matches!(
        solve(
            Request {
                pool: Pin::Exact(256),
                ..request()
            },
            vec![budget(255)],
            &recipe
        ),
        Err(Refusal::Capacity { .. })
    ));
    let automatic = solve(
        Request {
            context: (129).into(),
            ..request()
        },
        vec![budget(192)],
        &recipe,
    )
    .unwrap();
    assert_eq!(
        (automatic.request.context.maximum, automatic.pool_tokens),
        (129, 192)
    );
    for (available, context, tokens) in [(192, 193, 192), (192, 2 << 20, 192), (63, 64, 0)] {
        let refusal = solve(Request { context: context.into(), ..request() }, vec![budget(available)], &recipe).unwrap_err();
        let text = refusal.to_string();
        assert!(text.contains(&format!("maximum feasible pool is {tokens} tokens")), "{text}");
        assert!(text.contains("lower explicit context"), "{text}");
        let Refusal::FullContextShortfall { context: actual, tokens: maximum, limiting } = refusal else {
            panic!("expected a full-context refusal");
        };
        assert_eq!((actual, maximum), (context, tokens));
        assert!(matches!(*limiting, Refusal::Capacity { phase: Phase::PostTargetKv, .. }));
    }
}

#[test]
fn full_context_refusal_prices_best_feasible_assignment_not_pruned_alternative() {
    let mut contract = Linear::new().prepare(&request()).unwrap();
    let variable = add_profile(&mut contract, vec![0], vec![0], vec![0, 1], &[0, 64]);
    let mut remote = allocation("remote.experts", 1000);
    remote.location = Location { device: Device::Spark(0), domain: Domain::Device };
    let mut component = Component::allocation(remote, ByteCurve::fixed(1000, contract.maximum_pages),
        false, contract.maximum_pages);
    component.activation = Activation::choice(variable, 0);
    contract.components.push(component);
    let mut spark = budget(100);
    spark.id = "spark0".into();
    spark.locations = vec![Location { device: Device::Spark(0), domain: Domain::Device }];
    let refusal = solve(Request { context: (256).into(), ..request() }, vec![budget(192), spark],
        &Prepared(contract)).unwrap_err();
    let Refusal::FullContextShortfall { tokens, limiting, .. } = refusal else {
        panic!("expected full-context refusal");
    };
    assert_eq!(tokens, 128);
    let Refusal::Capacity { budget, required, available, owners, .. } = *limiting else {
        panic!("expected capacity witness");
    };
    assert_eq!((budget.as_str(), required, available), ("gpu", 256, 192));
    assert!(!owners.is_empty(), "refusal must retain evaluated owner names");
    assert!(!owners.iter().any(|owner| owner.contains("remote.experts")));
}

#[test]
fn automatic_target_uses_physical_card_not_logical_cap() {
    let recipe = Linear::new();
    let mut card = budget(96 << 30);
    card.logical_bytes = 24 << 30;
    assert_eq!(
        solve(request(), vec![card], &recipe).unwrap().pool_tokens,
        2 << 20
    );
    assert_eq!(
        solve(request(), vec![budget(32 << 30)], &recipe)
            .unwrap()
            .pool_tokens,
        1 << 20
    );
    let context = 3 << 20;
    let recipe = Linear {
        maximum: context,
        ..Linear::new()
    };
    assert_eq!(
        solve(
            Request {
                context: context.into(),
                ..request()
            },
            vec![budget(32 << 30)],
            &recipe
        )
        .unwrap()
        .pool_tokens,
        context
    );
}

#[test]
fn pool_first_then_onboarding_and_exact_constraints() {
    let target = 1 << 20;
    let recipe = Linear {
        onboard: 2,
        ..Linear::new()
    };
    let result = solve(request(), vec![budget(target + 64)], &recipe).unwrap();
    assert_eq!(
        (result.pool_tokens, result.decisions.onboard()),
        (target, 1)
    );
    let result = solve(request(), vec![budget(target - 64)], &recipe).unwrap();
    assert_eq!(
        (result.pool_tokens, result.decisions.onboard()),
        (target - 64, 0)
    );
    let result = solve(
        Request {
            onboard: Pin::Exact(2),
            ..request()
        },
        vec![budget(target)],
        &recipe,
    )
    .unwrap();
    assert_eq!(
        (result.pool_tokens, result.decisions.onboard()),
        (target - 128, 2)
    );
    let exact = Request {
        pool: Pin::Exact(256),
        ..request()
    };
    assert_eq!(
        solve(exact, vec![budget(384)], &recipe)
            .unwrap()
            .decisions
            .onboard(),
        2
    );
    for req in [
        Request {
            onboard: Pin::Exact(3),
            ..request()
        },
        Request {
            prefill: Pin::Exact(PrefillShape {
                rows: 2048,
                waves: 1,
            }),
            ..request()
        },
    ] {
        assert_eq!(
            solve(req, vec![budget(target)], &recipe).unwrap_err(),
            Refusal::PinConflict
        );
    }
}

#[test]
fn compiled_ceiling_uses_spare_memory_for_onboarding() {
    let recipe = Linear {
        maximum: 256,
        onboard: 1,
        fixed: 0,
    };
    let result = solve(request(), vec![budget(4096)], &recipe).unwrap();
    assert_eq!((result.pool_tokens, result.decisions.onboard()), (256, 1));
}

#[test]
fn per_device_balance_uses_the_tightest_complete_cost() {
    struct Balanced;
    impl Recipe for Balanced {
        fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
            let mut contract = Contract::fixed(64, 4, 2, facts(vec![]));
            contract.variables = vec![
                Variable::Attention {
                    layer: 0,
                    values: [0, 1]
                        .map(|gpu| LayerMode::Whole {
                            gpu,
                            ffn: FfnMode::Split,
                        })
                        .to_vec(),
                },
                Variable::Prefill(vec![decisions(0).prefill]),
                Variable::Lanes(vec![1]),
            ];
            for gpu in 0..2 {
                let mut a = allocation("attention+kv", 256);
                a.location = Location::coordinator(gpu);
                let mut component =
                    Component::allocation(a, ByteCurve::linear(256, 64, 4), false, 4);
                component.byte_extent(ByteCurve::linear(256, 64, 4))?;
                component.activation = Activation::choice(0, gpu as usize);
                contract.components.push(component);
            }
            Ok(contract)
        }
    }
    let mut peer = budget(512);
    peer.id = "peer".into();
    peer.locations = vec![Location::coordinator(1)];
    let result = solve(request(), vec![budget(256), peer], &Balanced).unwrap();
    assert_eq!(result.pool_tokens, 256);
    assert_eq!(
        result.decisions.attention,
        vec![LayerMode::Whole {
            gpu: 1,
            ffn: FfnMode::Split
        }]
    );
}

#[test]
fn loading_and_ready_peaks_are_separate_and_host_domains_share_one_budget() {
    let mut load = allocation("loading", 100);
    load.lifetime = Lifetime {
        first: Phase::Load,
        last: Phase::Load,
    };
    let mut pinned = allocation("pinned", 60);
    pinned.location = Location::PINNED;
    let mut host = allocation("host", 50);
    host.location = Location::HOST;
    let mut host_budget = budget(109);
    host_budget.id = "host".into();
    host_budget.locations = vec![Location::HOST, Location::PINNED];
    let allocations = facts(vec![load, allocation("ready", 100), host, pinned]);
    assert!(
        matches!(evaluate(&allocations, &[budget(100), host_budget.clone()]), Err(Refusal::Capacity { budget, phase: Phase::PostTargetKv, required: 110, .. }) if budget == "host")
    );
    host_budget.physical_bytes = 110;
    host_budget.logical_bytes = 110;
    let rows = evaluate(&allocations, &[budget(100), host_budget]).unwrap();
    assert!(rows
        .iter()
        .filter(|r| r.budget == "gpu")
        .all(|r| r.owned == 100));
}

#[test]
fn shared_backing_views_charge_once_and_enforce_slots_alignment_and_bounds() {
    let mut backing = allocation("arena", 1024);
    backing.alignment = 256;
    let mut a = allocation("slot.0", 128);
    a.slot = Some(0);
    a.view = Some(View {
        backing: "arena".into(),
        offset: 0,
    });
    let mut b = allocation("slot.1", 128);
    b.slot = Some(1);
    b.view = Some(View {
        backing: "arena".into(),
        offset: 128,
    });
    let mut p = plan(vec![backing, a, b]);
    assert_eq!(
        p.phases
            .iter()
            .find(|p| p.phase == Phase::Ready)
            .unwrap()
            .owned,
        1024
    );
    let r = |p: &ResolvedPlan| {
        p.facts
            .allocations
            .iter()
            .map(|a| Receipt {
                address: 4096 + a.view.as_ref().map_or(0, |v| v.offset),
                allocation: a.clone(),
            })
            .collect::<Vec<_>>()
    };
    reconcile(&p, Phase::Ready, &r(&p)).unwrap();
    p.facts.allocations[2].view.as_mut().unwrap().offset = 64;
    assert!(errors(reconcile(&p, Phase::Ready, &r(&p)))
        .iter()
        .any(|e| matches!(e, Mismatch::Address(_))));
    p.facts.allocations[2].view.as_mut().unwrap().offset = 1000;
    assert!(matches!(
        evaluate(&p.facts, &p.budgets),
        Err(Refusal::InvalidFact(_))
    ));
    let p = plan(vec![allocation("owner", 100)]);
    let mut r = receipts(&p);
    r[0].address = u64::MAX;
    assert!(
        errors(reconcile(&p, Phase::Ready, &r))
            .iter()
            .any(|e| matches!(e, Mismatch::Address(_))),
        "overflowing pointer extents must fail"
    );
}

#[test]
fn missing_extra_wrong_bytes_and_equal_total_mismatches_never_cancel() {
    let p = plan(vec![allocation("a", 100), allocation("b", 100)]);
    let r = receipts(&p);
    reconcile(&p, Phase::Ready, &r).unwrap();
    assert!(errors(reconcile(&p, Phase::Ready, &r[..1])).contains(&Mismatch::Missing("b".into())));
    let mut extra = r.clone();
    let mut owner = allocation("extra", 100);
    owner.location = Location::coordinator(0);
    extra.push(Receipt {
        allocation: owner,
        address: 16384,
    });
    assert!(errors(reconcile(&p, Phase::Ready, &extra)).contains(&Mismatch::Extra("extra".into())));
    let mut wrong = r.clone();
    wrong[0].allocation.bytes += 1;
    wrong[1].allocation.bytes -= 1;
    let e = errors(reconcile(&p, Phase::Ready, &wrong));
    assert!(
        e.contains(&Mismatch::Contract("a".into())) && e.contains(&Mismatch::Contract("b".into()))
    );
    let mut wrong_device = r.clone();
    wrong_device[0].allocation.location = Location::coordinator(1);
    assert!(errors(reconcile(&p, Phase::Ready, &wrong_device))
        .contains(&Mismatch::Contract("a".into())));
    let mut duplicate = r.clone();
    duplicate.push(r[0].clone());
    assert!(
        errors(reconcile(&p, Phase::Ready, &duplicate)).contains(&Mismatch::Duplicate("a".into()))
    );
}

#[test]
fn observed_lifetimes_require_loading_owner_release_before_ready() {
    let mut staging = allocation("load", 100);
    staging.lifetime = Lifetime {
        first: Phase::Load,
        last: Phase::Load,
    };
    let p = plan(vec![staging, allocation("target", 100)]);
    let r = receipts(&p);
    let mut history = vec![
        Observation {
            receipt: r[0].clone(),
            created: Phase::Load,
            freed: Some(Phase::Load),
        },
        Observation {
            receipt: r[1].clone(),
            created: Phase::PostTargetKv,
            freed: None,
        },
    ];
    reconcile_history(&p, Phase::Ready, &history).unwrap();
    history[0].freed = None;
    assert!(errors(reconcile_history(&p, Phase::Ready, &history))
        .contains(&Mismatch::Lifetime("load".into())));
    history[0].freed = Some(Phase::PostTargetKv);
    assert!(errors(reconcile_history(&p, Phase::Ready, &history))
        .contains(&Mismatch::Lifetime("load".into())));
    history[0].freed = Some(Phase::Load);
    history[1].created = Phase::Load;
    assert!(errors(reconcile_history(&p, Phase::Ready, &history))
        .contains(&Mismatch::Lifetime("target".into())));
}

fn packing_plan() -> ResolvedPlan {
    let mut p = plan(vec![allocation("target", 100)]);
    let mut a = allocation("pack.a", 300);
    a.lifetime = Lifetime {
        first: Phase::DraftLoad,
        last: Phase::DraftLoad,
    };
    let mut b = a.clone();
    b.id = "pack.b".into();
    b.bytes = 200;
    b.shape = vec![200];
    p.facts.sequential.push(Sequential {
        id: "packing".into(),
        members: vec![a, b],
    });
    p.phases = evaluate(&p.facts, &p.budgets).unwrap();
    p
}

fn packing_history(p: &ResolvedPlan) -> (Vec<Observation>, Vec<OwnerEvent>) {
    let ordinary = vec![Observation {
        receipt: receipts(p)[0].clone(),
        created: Phase::PostTargetKv,
        freed: None,
    }];
    let mut edges = vec![OwnerEvent::Allocated {
        receipt: ordinary[0].receipt.clone(),
        phase: Phase::PostTargetKv,
    }];
    for member in &p.facts.sequential[0].members {
        let receipt = Receipt {
            allocation: member.clone(),
            address: 32768,
        };
        edges.push(OwnerEvent::Allocated {
            receipt: receipt.clone(),
            phase: Phase::DraftLoad,
        });
        edges.push(OwnerEvent::Freed {
            receipt,
            phase: Phase::DraftLoad,
        });
    }
    (ordinary, edges)
}

#[test]
fn load_pool_reservations_charge_bounds_and_replay_independent_actual_peaks() {
    let mut exact = allocation("exact", 80);
    exact.location = Location::PINNED;
    exact.lifetime = Lifetime { first: Phase::Load, last: Phase::Growth };
    let mut p = plan(vec![]);
    p.facts.allocations = vec![exact.clone()];
    p.facts.reservations = vec![Reservation { id: "dense.0".into(), location: Location::PINNED,
        bytes: 300, lifetime: Lifetime { first: Phase::Load, last: Phase::Load },
        source: "checked ring/step bound".into() }];
    p.budgets[0].locations = vec![Location::PINNED];
    p.phases = evaluate(&p.facts, &p.budgets).unwrap();
    assert_eq!(p.phases.iter().find(|r| r.phase == Phase::Load).unwrap().owned, 380);
    assert_eq!(p.phases.iter().find(|r| r.phase == Phase::Ready).unwrap().owned, 80);
    let mut contract = Contract::fixed(64, 1, 1, p.facts.clone());
    contract.variables = vec![Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]), Variable::Lanes(vec![1])];
    assert_curve_census(&contract, &p.budgets, &[0, 0]);
    assert_eq!(contract.facts(&[0, 0], 1).unwrap(), p.facts);
    let pool = &p.facts.reservations[0];
    let edge = |id: &str, bytes, address, allocate| OwnerEvent::Reserved {
        reservation: pool.id.clone(), receipt: Receipt {
            allocation: Allocation::buffer(id, pool.location, bytes, pool.lifetime, Category::Staging, "pool"), address },
        phase: Phase::Load, allocate,
    };
    let exact_event = OwnerEvent::Allocated { receipt: Receipt { allocation: exact.clone(), address: 4096 }, phase: Phase::Load };
    let check = |events: &[OwnerEvent], reports: &[ReservationReceipt]| reconcile_startup(&p, Phase::Ready, events, &[], reports);
    for peak in [200, 300] {
        let events = vec![exact_event.clone(), edge("buffer.a", peak, 8192, true), edge("buffer.a", peak, 8192, false)];
        let report = ReservationReceipt { reservation: pool.clone(), peak };
        check(&events, &[report.clone()]).unwrap();
        assert!(check(&events, &[]).is_err());
        assert!(check(&events, &[report.clone(), report.clone()]).is_err());
        let mut wrong = report.clone(); wrong.peak -= 1;
        assert!(check(&events, &[wrong]).is_err());
        let mut wrong = report.clone(); wrong.reservation.bytes += 1;
        assert!(check(&events, &[wrong]).is_err());
        let mut extra = report.clone(); extra.reservation.id = "unknown".into();
        assert!(check(&events, &[report.clone(), extra]).is_err());
        let mut wrong = events.clone();
        if let OwnerEvent::Allocated { receipt, .. } = &mut wrong[0] { receipt.allocation.bytes += 1; }
        assert!(check(&wrong, &[report.clone()]).is_err(), "unused reservation cannot cancel an exact mismatch");
        assert!(check(&events[..2], &[report.clone()]).is_err());
        for mutation in 0..9 {
            let mut wrong = events.clone();
            if let OwnerEvent::Reserved { reservation, receipt, phase, .. } = &mut wrong[1] {
                match mutation {
                    0 => *reservation = "unknown".into(),
                    1 => receipt.allocation.location = Location::HOST,
                    2 => receipt.allocation.lifetime.last = Phase::Ready,
                    3 => receipt.address = 4096,
                    4 => receipt.allocation.id = "exact".into(),
                    5 => *phase = Phase::DraftLoad,
                    6 => receipt.allocation.dtype = "bf16".into(),
                    7 => receipt.allocation.shape = vec![1],
                    _ => receipt.allocation.alignment = 0,
                }
            }
            assert!(check(&wrong, &[report.clone()]).is_err(), "mutation {mutation}");
        }
        let mut wrong = events.clone();
        if let OwnerEvent::Reserved { receipt, .. } = &mut wrong[2] { receipt.address += 16; }
        assert!(check(&wrong, &[report.clone()]).is_err());
        let ordinary = [Observation { receipt: Receipt { allocation: exact.clone(), address: 4096 }, created: Phase::Load, freed: None }];
        assert!(reconcile_history(&p, Phase::Ready, &ordinary).is_err());
        assert!(reconcile_events(&p, Phase::Ready, &events, &[]).is_err());
    }
    let events = vec![exact_event.clone(), edge("buffer.a", 301, 8192, true), edge("buffer.a", 301, 8192, false)];
    assert!(check(&events, &[ReservationReceipt { reservation: pool.clone(), peak: 301 }]).is_err());
    let events = vec![exact_event, edge("buffer.a", 200, 8192, true), edge("buffer.b", 101, 12288, true),
        edge("buffer.a", 200, 8192, false), edge("buffer.b", 101, 12288, false)];
    assert!(check(&events, &[ReservationReceipt { reservation: pool.clone(), peak: 301 }]).is_err());
    let mut small = p.budgets.clone(); small[0].physical_bytes = 379; small[0].logical_bytes = 379;
    assert!(matches!(evaluate(&p.facts, &small), Err(Refusal::Capacity { owners, .. }) if owners.contains(&pool.id)));
}

#[test]
fn sequential_sources_charge_maximum_but_require_every_actual_pair() {
    let p = packing_plan();
    let at = |phase| p.phases.iter().find(|r| r.phase == phase).unwrap().owned;
    assert_eq!(at(Phase::DraftLoad), 400);
    assert_eq!(at(Phase::Ready), 100);
    let (ordinary, edges) = packing_history(&p);
    reconcile_events(&p, Phase::Ready, &edges, &[]).unwrap();
    assert!(errors(reconcile_history(&p, Phase::Ready, &ordinary))
        .contains(&Mismatch::Missing("pack.a".into())));
    let mut missing = edges.clone();
    missing.truncate(3);
    assert!(errors(reconcile_events(&p, Phase::Ready, &missing, &[]))
        .contains(&Mismatch::Missing("pack.b".into())));
    let mut retained = edges.clone();
    retained.pop();
    assert!(errors(reconcile_events(&p, Phase::Ready, &retained, &[]))
        .contains(&Mismatch::Lifetime("pack.b".into())));
    let overlap = [
        edges[0].clone(),
        edges[1].clone(),
        edges[3].clone(),
        edges[2].clone(),
        edges[4].clone(),
    ];
    assert!(errors(reconcile_events(&p, Phase::Ready, &overlap, &[]))
        .contains(&Mismatch::Lifetime("packing".into())));
    let wrong = edges
        .iter()
        .cloned()
        .map(|mut event| {
            let receipt = match &mut event {
                OwnerEvent::Allocated { receipt, .. } | OwnerEvent::Freed { receipt, .. }
                                | OwnerEvent::Reserved { receipt, .. } => {
                    receipt
                }
            };
            receipt.allocation.bytes += 1;
            event
        })
        .collect::<Vec<_>>();
    let errors = errors(reconcile_events(&p, Phase::Ready, &wrong, &[]));
    assert!(errors.contains(&Mismatch::Contract("pack.a".into())));
    assert!(errors.contains(&Mismatch::Contract("pack.b".into())));
}

#[test]
fn sequential_sources_reject_duplicate_extra_wrong_device_address_and_phase() {
    let p = packing_plan();
    let (_, edges) = packing_history(&p);
    for mutation in 0..8 {
        let mut bad = edges.clone();
        match mutation {
            0 => bad.extend_from_slice(&edges[1..3]),
            1 => {
                if let OwnerEvent::Allocated { receipt, .. } = &mut bad[1] {
                    receipt.allocation.id = "unplanned".into();
                }
            }
            2 => {
                if let OwnerEvent::Allocated { receipt, .. } = &mut bad[1] {
                    receipt.allocation.location = Location::coordinator(1);
                }
            }
            3 => {
                if let OwnerEvent::Allocated { receipt, .. } = &mut bad[1] {
                    receipt.address = u64::MAX;
                }
            }
            4 => {
                if let OwnerEvent::Freed { phase, .. } = &mut bad[2] {
                    *phase = Phase::Warmup;
                }
            }
            5 => {
                if let OwnerEvent::Freed { receipt, .. } = &mut bad[2] {
                    receipt.address += 1;
                }
            }
            6 => bad.insert(3, edges[2].clone()),
            7 => {
                bad.remove(0);
            }
            _ => unreachable!(),
        }
        assert!(
            reconcile_events(&p, Phase::Ready, &bad, &[]).is_err(),
            "mutation {mutation}"
        );
    }
}

#[test]
fn sequential_history_proves_same_phase_owner_order_and_address_reuse() {
    let mut p = packing_plan();
    p.facts.allocations[0].lifetime.first = Phase::DraftLoad;
    p.phases = evaluate(&p.facts, &p.budgets).unwrap();
    let (_, mut events) = packing_history(&p);
    let mut owner = events.remove(0);
    if let OwnerEvent::Allocated { receipt, phase } = &mut owner {
        *phase = Phase::DraftLoad;
        receipt.address = 32768;
    }
    // A retained destination can reuse the source address only after its free.
    events.push(owner.clone());
    reconcile_events(&p, Phase::Ready, &events, &[]).unwrap();
    events.pop();
    events.insert(1, owner);
    assert!(errors(reconcile_events(&p, Phase::Ready, &events, &[]))
        .iter()
        .any(|e| matches!(e, Mismatch::Address(_))));
    let (ordinary, _) = packing_history(&p);
    assert!(
        errors(reconcile(&p, Phase::Ready, &[ordinary[0].receipt.clone()]))
            .contains(&Mismatch::Missing("pack.a".into()))
    );
}

#[test]
fn sequential_groups_may_coexist_but_their_live_addresses_cannot_overlap() {
    let mut p = packing_plan();
    let mut other = p.facts.sequential[0].members[0].clone();
    other.id = "other.source".into();
    p.facts.sequential.push(Sequential {
        id: "other.packing".into(),
        members: vec![other.clone()],
    });
    p.phases = evaluate(&p.facts, &p.budgets).unwrap();
    let (_, original) = packing_history(&p);
    for address in [65536, 32768] {
        let r = Receipt {
            allocation: other.clone(),
            address,
        };
        let events = [
            original[0].clone(),
            original[1].clone(),
            OwnerEvent::Allocated {
                receipt: r.clone(),
                phase: Phase::DraftLoad,
            },
            OwnerEvent::Freed {
                receipt: r,
                phase: Phase::DraftLoad,
            },
            original[2].clone(),
            original[3].clone(),
            original[4].clone(),
        ];
        let result = reconcile_events(&p, Phase::Ready, &events, &[]);
        if address == 65536 {
            result.unwrap();
        } else {
            assert!(errors(result)
                .iter()
                .any(|e| matches!(e, Mismatch::Address(_))));
        }
    }
}

#[test]
fn sequential_contract_costs_match_exact_evaluator_and_capacity() {
    let p = packing_plan();
    let mut contract = Contract::fixed(64, 1, 1, p.facts.clone());
    contract.variables = vec![
        Variable::Prefill(vec![decisions(0).prefill]),
        Variable::Lanes(vec![1]),
    ];
    let mut req = request();
    req.pool = Pin::Exact(64);
    let mut b = budget(400);
    let solved = solve(req.clone(), vec![b.clone()], &Prepared(contract.clone())).unwrap();
    assert_eq!(solved.facts, p.facts);
    assert_eq!(solved.phases, evaluate(&p.facts, &[b.clone()]).unwrap());
    b.physical_bytes = 399;
    b.logical_bytes = 399;
    assert!(matches!(
        solve(req, vec![b], &Prepared(contract)),
        Err(Refusal::Capacity {
            phase: Phase::DraftLoad,
            ..
        })
    ));
    for malformed in 0..4 {
        let mut bad = p.facts.clone();
        match malformed {
            0 => bad.sequential[0].members.clear(),
            1 => bad.sequential[0].members[1].location = Location::PINNED,
            2 => bad.sequential[0].members[1].lifetime.last = Phase::Ready,
            3 => bad.sequential[0].members[1].id = "target".into(),
            _ => unreachable!(),
        }
        assert!(matches!(
            evaluate(&bad, &p.budgets),
            Err(Refusal::InvalidFact(_))
        ));
    }
}

#[test]
fn deferred_reservations_cannot_authorize_readiness_and_headroom_is_not_opaque() {
    let mut p = plan(vec![allocation("target", 100)]);
    p.facts.deferred.push(allocation("draft.pending", 100));
    let r = receipts(&p);
    reconcile(&p, Phase::PostTargetKv, &r).unwrap();
    assert!(errors(reconcile(&p, Phase::Ready, &r))
        .contains(&Mismatch::Missing("draft.pending".into())));
    reconcile_usage(&p, "gpu", Phase::Ready, 100).unwrap();
    assert!(matches!(
        reconcile_usage(&p, "gpu", Phase::Ready, 101),
        Err(Refusal::Capacity { .. })
    ));
}

#[test]
fn legacy_mla_bridge_only_proves_startup_owners_and_keeps_plan_budget_validation() {
    let mut a = allocation("draft.mla.owner", 100);
    a.lifetime = Lifetime {
        first: Phase::Load,
        last: Phase::Growth,
    };
    let f = facts(vec![a.clone()]);
    let event = OwnerEvent::Allocated {
        receipt: Receipt {
            allocation: a,
            address: 4096,
        },
        phase: Phase::Load,
    };
    reconcile_legacy_mla_events(&f, Phase::Load, &[event.clone()]).unwrap();
    reconcile_legacy_mla_events(&f, Phase::Warmup, &[event.clone()]).unwrap();
    assert!(matches!(
        reconcile_legacy_mla_events(&f, Phase::Ready, &[event.clone()]),
        Err(Refusal::InvalidFact(_))
    ));
    assert!(reconcile_legacy_mla_events(&f, Phase::Load, &[]).is_err());
    let mut wrong = event.clone();
    if let OwnerEvent::Allocated { receipt, .. } = &mut wrong {
        receipt.allocation.bytes += 1;
    }
    assert!(reconcile_legacy_mla_events(&f, Phase::Load, &[wrong]).is_err());
    let mut invalid = f.clone();
    invalid.deferred.push(allocation("pending", 1));
    assert!(matches!(
        reconcile_legacy_mla_events(&invalid, Phase::Load, &[event.clone()]),
        Err(Refusal::InvalidFact(_))
    ));
    let mut p = plan(f.allocations);
    p.budgets.clear();
    assert!(matches!(
        reconcile_events(&p, Phase::Load, &[event], &[]),
        Err(Refusal::InvalidFact(_))
    ));
}

#[test]
fn invalid_inputs_return_typed_errors() {
    let p = plan(vec![allocation("a", 100)]);
    let mut budgets = p.budgets.clone();
    budgets[0].baseline = Baseline::Sampled {
        free: 5000,
        phase: Phase::Load,
        source: "sample".into(),
    };
    assert!(matches!(
        evaluate(&p.facts, &budgets),
        Err(Refusal::InvalidFact(_))
    ));
    let mut invalid = p.facts.clone();
    invalid.allocations.push(invalid.allocations[0].clone());
    assert!(matches!(
        evaluate(&invalid, &p.budgets),
        Err(Refusal::InvalidFact(_))
    ));
    let mut duplicate_budget = p.budgets[0].clone();
    duplicate_budget.id = "duplicate".into();
    assert!(matches!(
        evaluate(&p.facts, &[p.budgets[0].clone(), duplicate_budget]),
        Err(Refusal::InvalidFact(_))
    ));
    let mut invalid = p.facts.clone();
    invalid.allocations[0].alignment = 3;
    assert!(matches!(
        evaluate(&invalid, &p.budgets),
        Err(Refusal::InvalidFact(_))
    ));
    let mut invalid = p.facts.clone();
    invalid.allocations[0].bytes = u64::MAX;
    invalid.allocations.push(allocation("b", 1));
    assert!(matches!(
        evaluate(&invalid, &p.budgets),
        Err(Refusal::Overflow(_))
    ));
}


#[test]
fn family_phase_order_changes_packing_peak_pool_and_event_validation_together() {
    struct Ordered(PhaseOrder);
    impl Recipe for Ordered {
        fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
            let mut target = allocation("kv", 64);
            target.lifetime = Lifetime::TARGET;
            let mut weights = allocation("weights", 100);
            weights.lifetime = Lifetime::RESIDENT;
            let mut draft = allocation("draft", 100);
            draft.lifetime = Lifetime { first: Phase::DraftLoad, last: Phase::Growth };
            let mut pack = allocation("pack", 600);
            pack.lifetime = Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad };
            let mut owners = facts(vec![weights, target, draft]);
            owners.phase_order = self.0;
            owners.sequential.push(Sequential { id: "packing".into(), members: vec![pack] });
            let mut contract = Contract::fixed(64, 16, 1, owners);
            contract.set_byte_extent("kv", ByteCurve::linear(64, 64, 16))?;
            contract.variables = vec![Variable::Attention { layer: 0, values: decisions(0).attention },
                Variable::Prefill(vec![decisions(0).prefill]), Variable::Lanes(vec![1])];
            Ok(contract)
        }
    }
    for (order, expected_pool, peak_phase, peak_bytes) in [
        (PhaseOrder::TARGET_THEN_DRAFT, 192, Phase::DraftLoad, 992),
        (PhaseOrder::DRAFT_THEN_TARGET, 768, Phase::PostTargetKv, 968),
    ] {
        let recipe = Ordered(order);
        let plan = solve(request(), vec![budget(1000)], &recipe).unwrap();
        assert_eq!(plan.facts.phase_order, order);
        assert_eq!(plan.pool_tokens, expected_pool);
        assert_eq!(plan.phases.iter().max_by_key(|row| row.owned).unwrap().owned, peak_bytes);
        assert_eq!(plan.phases.iter().find(|row| row.phase == peak_phase).unwrap().owned, peak_bytes);
        let contract = recipe.prepare(&request()).unwrap();
        let roomy = [budget(10000)];
        let compiled = search::Compiled::new(&contract, &roomy).unwrap();
        for pages in 1..=16 {
            let facts = contract.facts(&[0, 0, 0], pages).unwrap();
            assert_eq!(compiled.phases(&roomy, &[0, 0, 0], pages).unwrap(), evaluate(&facts, &roomy).unwrap());
        }
        let receipts = plan.facts.allocations.iter().chain(&plan.facts.sequential[0].members)
            .enumerate().map(|(i, allocation)| Receipt { allocation: allocation.clone(),
                address: (i as u64 + 1) * 4096 }).collect::<Vec<_>>();
        let mut events = Vec::new();
        for &phase in order.phases() {
            for receipt in receipts.iter().filter(|r| r.allocation.lifetime.first == phase) {
                events.push(OwnerEvent::Allocated { receipt: receipt.clone(), phase });
            }
            for receipt in receipts.iter().filter(|r| r.allocation.lifetime.last == phase && phase == Phase::DraftLoad) {
                events.push(OwnerEvent::Freed { receipt: receipt.clone(), phase });
            }
            if phase == Phase::Ready { break; }
        }
        reconcile_events(&plan, Phase::Ready, &events, &[]).unwrap();
        let mut wrong = plan.clone();
        wrong.facts.phase_order = if order == PhaseOrder::TARGET_THEN_DRAFT {
            PhaseOrder::DRAFT_THEN_TARGET
        } else { PhaseOrder::TARGET_THEN_DRAFT };
        assert!(reconcile_events(&wrong, Phase::Ready, &events, &[]).is_err());
    }
    let pinned = Request { pool: Pin::Exact(512), ..request() };
    assert!(matches!(solve(pinned.clone(), vec![budget(1000)], &Ordered(PhaseOrder::TARGET_THEN_DRAFT)),
        Err(Refusal::Capacity { phase: Phase::DraftLoad, .. })));
    assert!(solve(pinned, vec![budget(1000)], &Ordered(PhaseOrder::DRAFT_THEN_TARGET)).is_ok());
}

#[test]
fn family_phase_order_validates_lifetimes_views_and_serialized_phase_lists() {
    for order in [PhaseOrder::TARGET_THEN_DRAFT, PhaseOrder::DRAFT_THEN_TARGET] {
        let value = serde_json::to_value(order).unwrap();
        assert_eq!(serde_json::from_value::<PhaseOrder>(value).unwrap(), order);
        let first = order.phases()[1];
        let last = order.phases()[2];
        let mut backing = allocation("backing", 100);
        backing.lifetime = Lifetime { first, last };
        let mut view = allocation("view", 20);
        view.lifetime = Lifetime { first: last, last };
        view.view = Some(View { backing: "backing".into(), offset: 0 });
        let mut owners = facts(vec![backing, view]);
        owners.phase_order = order;
        evaluate(&owners, &[budget(100)]).unwrap();
        owners.allocations[0].lifetime = Lifetime { first: last, last: first };
        assert!(matches!(evaluate(&owners, &[budget(100)]), Err(Refusal::InvalidFact(_))));
        owners.allocations[0].lifetime = Lifetime { first, last: first };
        assert!(matches!(evaluate(&owners, &[budget(100)]), Err(Refusal::InvalidFact(_))));
    }
    let mut invalid = *PhaseOrder::TARGET_THEN_DRAFT.phases();
    invalid[2] = Phase::PostTargetKv;
    assert!(PhaseOrder::try_from(invalid).is_err());
    assert!(serde_json::from_value::<PhaseOrder>(serde_json::to_value(invalid).unwrap()).is_err());
}


fn resource(id: &str, gpu: u8, bytes: u64, maximum: u64) -> Component {
    Component::resident_weight(ResidentWeight { id: id.into(), location: Location::coordinator(gpu),
        weights: ResidentWeights(bytes) }, ByteCurve::fixed(bytes, maximum))
}
fn capped(limit: u64) -> Request {
    Request { resident_weight_limit: Some(ResidentWeights(limit)), ..request() }
}
fn gpu_budget(gpu: u8, bytes: u64) -> Budget {
    Budget { id: format!("rtx{gpu}"), locations: vec![Location::coordinator(gpu)], ..budget(bytes) }
}

#[test]
fn analytical_weight_cap_counts_loaded_owners_without_allocation_or_category_inference() {
    let mut contract = Linear::new().prepare(&request()).unwrap();
    contract.components.push(resource("loaded.target", 0, 80, contract.maximum_pages));
    let accepted = solve(capped(80), vec![budget(1 << 30)], &Prepared(contract.clone())).unwrap();
    let uncapped = solve(request(), vec![budget(1 << 30)], &Prepared(contract.clone())).unwrap();
    assert_eq!(accepted.pool_tokens, uncapped.pool_tokens);
    assert_eq!(accepted.phases, uncapped.phases);
    assert_eq!(accepted.facts.resident_weights[0].id, "loaded.target");
    assert!(!accepted.facts.allocations.iter().any(|a| a.id == "loaded.target"));
    assert!(matches!(solve(capped(79), vec![budget(1 << 30)], &Prepared(contract)),
        Err(Refusal::ResidentWeightLimit { gpu: 0, required: 80, limit: 79 })));

    let mut contract = Linear::new().prepare(&request()).unwrap();
    let mut owner = allocation("display.weight", 64);
    owner.category = Category::Weights;
    contract.components.push(Component::allocation(owner, ByteCurve::fixed(64, contract.maximum_pages), false, contract.maximum_pages));
    assert!(solve(capped(0), vec![budget(1 << 30)], &Prepared(contract)).is_ok());
}

#[test]
fn analytical_weight_cap_prunes_conditional_peer_cost_before_home_selection() {
    let mut contract = Linear::new().prepare(&request()).unwrap();
    contract.gpus = 2;
    contract.variables.push(Variable::Embedding(vec![EmbeddingHome::Device(0), EmbeddingHome::Device(1)]));
    let home = contract.variables.len() - 1;
    contract.components.push(resource("loaded.peer", 1, 75, contract.maximum_pages));
    for (option, gpu, bytes) in [(0, 1, 6), (1, 0, 6)] {
        let mut owner = resource("mixed.draft.weight", gpu, bytes, contract.maximum_pages);
        owner.activation = Activation::choice(home, option);
        contract.components.push(owner);
    }
    let budgets = vec![gpu_budget(0, 1 << 30), gpu_budget(1, 1 << 30)];
    let plan = solve(capped(80), budgets, &Prepared(contract)).unwrap();
    assert_eq!(plan.decisions.embedding, Some(EmbeddingHome::Device(1)));
    assert_eq!(resident_weight_totals(&plan.facts, 2).unwrap(), [ResidentWeights(6), ResidentWeights(75)]);
}

#[test]
fn analytical_weight_cap_stops_tp2_onboarding_without_reducing_pool_or_mandatory_draft() {
    let mut contract = Linear { onboard: 2, ..Linear::new() }.prepare(&request()).unwrap();
    contract.gpus = 2;
    let variable = contract.variables.len() - 1;
    if let Variable::Onboard { destination, .. } = &mut contract.variables[variable] { *destination = vec![0, 1]; }
    contract.components.push(resource("target.0", 0, 60, contract.maximum_pages));
    contract.components.push(resource("target.1", 1, 60, contract.maximum_pages));
    contract.components.push(resource("tp1.draft", 0, 10, contract.maximum_pages));
    for layer in 0..2 {
        for (gpu, bytes) in [(0, 4), (1, 12)] {
            let mut owner = resource(&format!("expert.{layer}.rank.{gpu}"), gpu, bytes, contract.maximum_pages);
            owner.activation = Activation { clauses: (layer + 1..=2).map(|option| vec![Choice { variable, option }]).collect() };
            contract.components.push(owner);
        }
    }
    let budgets = vec![gpu_budget(0, 1 << 30), gpu_budget(1, 1 << 30)];
    let plan = solve(capped(80), budgets.clone(), &Prepared(contract.clone())).unwrap();
    let uncapped = solve(request(), budgets.clone(), &Prepared(contract.clone())).unwrap();
    assert_eq!(plan.pool_tokens, uncapped.pool_tokens);
    assert_eq!(plan.decisions.onboard_layers, [vec![0], vec![0]]);
    assert_eq!(resident_weight_totals(&plan.facts, 2).unwrap(), [ResidentWeights(74), ResidentWeights(72)]);
    assert!(matches!(solve(Request { onboard: Pin::Exact(2), ..capped(80) }, budgets, &Prepared(contract)),
        Err(Refusal::ResidentWeightLimit { gpu: 1, required: 84, limit: 80 })));
}

#[test]
fn analytical_weight_cap_prices_every_finite_piece_and_exact_bound() {
    let mut contract = Linear { maximum: 64 * 12, ..Linear::new() }.prepare(&request()).unwrap();
    let mut owner = resource("piecewise.weight", 0, 100, 12);
    owner.bytes = ByteCurve(vec![
        BytePiece { first: 1, last: 3, bytes_at_first: 100, bytes_per_page: 0 },
        BytePiece { first: 4, last: 8, bytes_at_first: 10, bytes_per_page: 10 },
        BytePiece { first: 9, last: 12, bytes_at_first: 100, bytes_per_page: 0 },
    ]);
    contract.components.push(owner);
    let plan = solve(capped(35), vec![budget(1 << 30)], &Prepared(contract.clone())).unwrap();
    assert_eq!(plan.pool_tokens, 6 * 64);
    assert_eq!(plan.facts.resident_weights[0].weights, ResidentWeights(30));
    let costs = search::Compiled::new(&Contract { resident_weight_limit: Some(ResidentWeights(35)), ..contract.clone() }, &[budget(1 << 30)]).unwrap();
    assert_eq!(costs.limit(&[budget(1 << 30)], &[0, 0, 0], 1, 12).unwrap(), Some(6));
    assert!(matches!(solve(Request { pool: Pin::Exact(7 * 64), ..capped(35) }, vec![budget(1 << 30)], &Prepared(contract)),
        Err(Refusal::ResidentWeightLimit { gpu: 0, required: 40, limit: 35 })));
}

#[test]
fn analytical_weight_cap_rejects_alias_duplicate_unknown_home_and_overflow() {
    let base = Linear::new().prepare(&request()).unwrap();
    let mut duplicate = base.clone();
    duplicate.components.push(resource("loaded", 0, 1, base.maximum_pages));
    duplicate.components.push(resource("loaded", 0, 1, base.maximum_pages));
    assert!(matches!(solve(capped(0), vec![budget(1 << 30)], &Prepared(duplicate)), Err(Refusal::InvalidFact(_))));
    let mut alias = base.clone();
    let mut view = allocation("alias", 64);
    view.view = Some(View { backing: "kv".into(), offset: 0 });
    alias.components.push(Component::allocation(view, ByteCurve::fixed(64, base.maximum_pages), false, base.maximum_pages));
    alias.components.push(resource("alias", 0, 1, base.maximum_pages));
    assert!(matches!(solve(capped(0), vec![budget(1 << 30)], &Prepared(alias)), Err(Refusal::InvalidFact(_))));
    let mut unknown = base.clone();
    unknown.components.push(resource("unknown", 1, 1, base.maximum_pages));
    assert!(matches!(solve(capped(0), vec![budget(1 << 30)], &Prepared(unknown)), Err(Refusal::InvalidFact(_))));
    let mut overflow = base;
    overflow.components.push(resource("loaded.a", 0, u64::MAX, overflow.maximum_pages));
    overflow.components.push(resource("loaded.b", 0, 1, overflow.maximum_pages));
    assert!(matches!(solve(request(), vec![budget(1 << 30)], &Prepared(overflow)), Err(Refusal::Overflow(_))));
}

#[test]
fn floor_fallback_preserves_optional_owner_before_maximizing_pool() {
    struct Optional;
    impl Recipe for Optional {
        fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
            let mut base = Linear::new();
            base.maximum = 256;
            let mut contract = base.prepare(request)?;
            let variable = contract.variables.len();
            contract.variables.push(Variable::Encoder(vec![0, 1]));
            contract.floor_fallback = Some(Choice { variable, option: 1 });
            let mut tower = Component::allocation(allocation("tower", 64),
                ByteCurve::fixed(64, 4), false, 4);
            tower.activation = Activation::choice(variable, 0);
            contract.components.push(tower);
            Ok(contract)
        }
    }
    let on = solve(request(), vec![budget(256)], &Optional).unwrap();
    assert_eq!((on.pool_tokens, on.decisions.encoder), (192, Some(0)));
    let off = solve(request(), vec![budget(64)], &Optional).unwrap();
    assert_eq!((off.pool_tokens, off.decisions.encoder), (64, Some(1)));
    let exact = solve(Request { pool: Pin::Exact(256), ..request() },
        vec![budget(256)], &Optional).unwrap();
    assert_eq!((exact.pool_tokens, exact.decisions.encoder), (256, Some(1)));
    assert!(matches!(solve(request(), vec![budget(32)], &Optional),
        Err(Refusal::FullContextShortfall { .. })));
}

#[test]
fn encoder_priority_follows_pool_and_onboarding_before_slack() {
    struct WithEncoder;
    impl Recipe for WithEncoder {
        fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
            let mut base = Linear::new();
            base.maximum = 128;
            base.onboard = 1;
            let mut contract = base.prepare(request)?;
            let variable = contract.variables.len();
            contract.variables.push(Variable::Encoder(vec![0, 1]));
            let mut tower = Component::allocation(allocation("tower", 64),
                ByteCurve::fixed(64, 2), false, 2);
            tower.activation = Activation::choice(variable, 0);
            contract.components.push(tower);
            Ok(contract)
        }
    }
    let roomy = solve(request(), vec![budget(256)], &WithEncoder).unwrap();
    assert_eq!(roomy.pool_tokens, 128);
    assert_eq!(roomy.decisions.onboard_layers, vec![vec![0]]);
    assert_eq!(roomy.decisions.encoder, Some(0), "encoder wins over extra slack");
    let tight = solve(request(), vec![budget(192)], &WithEncoder).unwrap();
    assert_eq!(tight.pool_tokens, 128);
    assert_eq!(tight.decisions.onboard_layers, vec![vec![0]]);
    assert_eq!(tight.decisions.encoder, Some(1), "onboarding wins over optional encoder");
    assert_eq!(tight.encoder_diagnostics[0].1, 64);
    let tighter = solve(request(), vec![budget(128)], &WithEncoder).unwrap();
    assert_eq!(tighter.pool_tokens, 128, "pool wins over optional encoder");
    assert_eq!(tighter.decisions.encoder, Some(1));
}

#[test]
fn many_singletons_preserve_branching_winner_and_typed_refusals() {
    let request = Request { pool: Pin::Exact(128), ..request() };
    let compact = Linear { maximum: 256, onboard: 2, fixed: 0 }.prepare(&request).unwrap();
    let mut padded = compact.clone();
    for layer in 1..64 {
        padded.variables.push(Variable::Attention { layer,
            values: vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] });
    }
    for capacity in [128, 192, 256] {
        let expected = solve(request.clone(), vec![budget(capacity)], &Prepared(compact.clone())).unwrap();
        let actual = solve(request.clone(), vec![budget(capacity)], &Prepared(padded.clone())).unwrap();
        assert_eq!(actual.pool_tokens, 128);
        assert_eq!(actual.decisions.onboard_layers, expected.decisions.onboard_layers);
        assert_eq!(actual.decisions.onboard_layers[0].len(), ((capacity - 128) / 64) as usize);
        assert_eq!(actual.decisions.attention.len(), 64);
        assert_eq!(actual.facts, expected.facts);
        assert_eq!(actual.phases, expected.phases);
    }
    let expected = solve(request.clone(), vec![budget(127)], &Prepared(compact)).unwrap_err();
    let actual = solve(request.clone(), vec![budget(127)], &Prepared(padded)).unwrap_err();
    assert_eq!(actual, expected);
    assert!(matches!(actual, Refusal::Capacity { .. }));
}

#[test]
fn infeasible_singletons_are_refused_without_an_option_probe() {
    let request = Request { pool: Pin::Exact(64), ..request() };
    let mut contract = Linear { maximum: 64, ..Linear::new() }.prepare(&request).unwrap();
    for layer in 1..64 {
        contract.variables.push(Variable::Attention { layer,
            values: vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] });
    }
    let refusal = solve(request.clone(), vec![budget(63)], &Prepared(contract)).unwrap_err();
    assert!(matches!(refusal, Refusal::Capacity { required: 64, available: 63, .. }));
}

/// A Linear contract (64 B per 64-token page) with built-in GPU reserve
/// components: an opaque runtime allowance and a Runtime-category allocation.
fn with_reserve_components(maximum_pages: u64) -> Contract {
    let mut contract = Linear { maximum: 64 * maximum_pages, ..Linear::new() }.prepare(&request()).unwrap();
    contract.components.push(Component { activation: Activation::always(),
        bytes: ByteCurve::fixed(1000, maximum_pages), kind: ComponentKind::Allowance(Allowance {
            location: Location::coordinator(0), bytes: 1000, lifetime: Lifetime::RESIDENT,
            source: "estimated runtime growth".into(), assumed: true }) });
    let graphs = Allocation::buffer("graphs", Location::coordinator(0), 500, Lifetime::RESIDENT,
        Category::Runtime, "graphs");
    contract.components.push(Component::allocation(graphs, ByteCurve::fixed(500, maximum_pages), false, maximum_pages));
    contract
}

#[test]
fn gpu_reserve_override_replaces_every_builtin_reserve_component() {
    use cuteafd_core::memory_policy::MemoryPolicy;
    let contract = with_reserve_components(1 << 15);
    // Default: both components charged; 64 * 100 B of pool beside 1500 B.
    let auto = solve(request(), vec![budget(1500 + 64 * 100)], &Prepared(contract.clone())).unwrap();
    assert_eq!(auto.pool_tokens, 64 * 100);
    assert!(auto.facts.allocations.iter().any(|owner| owner.id == "graphs"));
    assert_eq!(auto.facts.allowances.len(), 1);
    // GPU_RESERVE 200 B: components dropped, the budget keeps 200 B free.
    let overridden = MemoryPolicy { gpu_reserve: Some(200), ..Default::default() }
        .scoped(|| solve(request(), vec![budget(1500 + 64 * 100)], &Prepared(contract.clone()))).unwrap();
    assert!(!overridden.facts.allocations.iter().any(|owner| owner.id == "graphs"));
    assert!(overridden.facts.allowances.is_empty());
    assert_eq!(overridden.budgets[0].headroom, 200);
    assert_eq!(overridden.pool_tokens, 64 * ((1500 + 64 * 100 - 200) / 64));
    // A Spark override leaves GPU components alone.
    let spark = MemoryPolicy { spark_reserve: Some(200), ..Default::default() }
        .scoped(|| solve(request(), vec![budget(1500 + 64 * 100)], &Prepared(contract.clone()))).unwrap();
    assert_eq!((spark.pool_tokens, spark.budgets[0].headroom), (auto.pool_tokens, 0));
}

#[test]
fn family_reserve_adds_to_the_measured_components() {
    use cuteafd_core::memory_policy::MemoryPolicy;
    let contract = with_reserve_components(1 << 15);
    let solved = MemoryPolicy { family_reserve: Some(640), ..Default::default() }.scoped(|| {
        let mut budgets = vec![budget(1500 + 64 * 100)];
        budgets[0].headroom = crate::placement::reserve::gpu("glm5");
        solve(request(), budgets, &Prepared(contract.clone()))
    }).unwrap();
    // The components stay charged and the named reserve comes off on top.
    assert!(solved.facts.allocations.iter().any(|owner| owner.id == "graphs"));
    assert_eq!(solved.facts.allowances.len(), 1);
    assert_eq!(solved.budgets[0].headroom, 640);
    assert_eq!(solved.pool_tokens, 64 * 90);
}

#[test]
fn kv_min_refuses_and_kv_max_grows_an_automatic_pool_past_the_target() {
    use cuteafd_core::memory_policy::MemoryPolicy;
    let contract = Linear { maximum: 64 * 1000, ..Linear::new() }.prepare(&request()).unwrap();
    let solve_with = |policy: MemoryPolicy, bytes: u64| policy.scoped(||
        solve(request(), vec![budget(bytes)], &Prepared(contract.clone())));
    // Target 64 pages: the layout is sized for it; without a max the pool stops there.
    let target = MemoryPolicy { kv_target: Some(64 * 64), kv_min: Some(64), ..Default::default() };
    assert_eq!(solve_with(target, 64 * 500).unwrap().pool_tokens, 64 * 64);
    // KV max 640 pages: the pool grows into what is left, up to the max.
    let grow = MemoryPolicy { kv_max: Some(64 * 640), ..target };
    assert_eq!(solve_with(grow, 64 * 500).unwrap().pool_tokens, 64 * 500);
    assert_eq!(solve_with(grow, 64 * 900).unwrap().pool_tokens, 64 * 640);
    // KV min above what fits is refused, never published.
    let minimum = MemoryPolicy { kv_min: Some(64 * 64), ..target };
    assert!(solve_with(minimum, 64 * 32).is_err());
    assert_eq!(solve_with(minimum, 64 * 64).unwrap().pool_tokens, 64 * 64);
}

#[test]
fn automatic_context_uses_shared_floor_and_exact_admitted_extent() {
    let maximum = 1 << 20;
    let recipe = Linear { maximum: 2 << 20, ..Linear::new() };
    let automatic = Request { context: Context::automatic(maximum), ..request() };
    for pool in [262_144, 760_768, maximum, 2 << 20] {
        let mut available = budget(pool);
        available.physical_bytes = 96 << 30;
        let solved = solve(automatic.clone(), vec![available], &recipe).unwrap();
        assert_eq!(solved.pool_tokens, pool);
        assert_eq!(solved.served_context(), maximum.min(pool));
        assert_eq!(solved.request.context.maximum, maximum);
    }
    assert!(solve(automatic.clone(), vec![budget(262_144 - 64)], &recipe).is_err());
    let pinned = Request { context: Context::pinned(maximum), ..automatic.clone() };
    assert!(solve(pinned, vec![budget(760_768)], &recipe).is_err());
    let small_pool = Request { pool: Pin::Exact(64), ..automatic };
    let solved = solve(small_pool, vec![budget(64)], &recipe).unwrap();
    assert_eq!(solved.served_context(), 64);
    let short = solve(Request { context: Context::automatic(128), ..request() },
        vec![budget(128)], &recipe).unwrap();
    assert_eq!(short.served_context(), 128);
    assert!(solve(Request { context: Context::resolve(128, Some(256)), ..request() },
        vec![budget(512)], &recipe).is_err());
}

#[test]
fn pinned_context_keeps_the_operator_limit_with_a_larger_pool() {
    let solved = solve(Request { context: Context::resolve(1 << 20, Some(131_072)), ..request() },
        vec![budget(2 << 20)], &Linear::new()).unwrap();
    assert_eq!(solved.served_context(), 131_072);
    assert_eq!(solved.request.context.maximum, 1 << 20);
}

#[test]
fn automatic_floor_is_bounded_by_an_explicit_target() {
    cuteafd_core::memory_policy::MemoryPolicy { kv_target: Some(131_072), ..Default::default() }
        .scoped(|| {
            let solved = solve(Request { context: Context::automatic(1 << 20), ..request() },
                vec![budget(2 << 20)], &Linear::new()).unwrap();
            assert_eq!((solved.pool_tokens, solved.served_context()), (131_072, 131_072));
        });
}
