use super::*;
use crate::placement::FfnMode;

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
    Facts { resident_weights: vec![],
        phase_order: crate::placement::planner::PhaseOrder::TARGET_THEN_DRAFT,
        allocations,
        deferred: vec![],
        sequential: vec![],
        reservations: vec![], allowances: vec![],
    }
}
fn request() -> Request {
    Request {
        context: 64,
        pool: Pin::Auto,
        onboard: Pin::Auto,
        prefill: Pin::Auto,
        spark_reply_format: None,
        operation: Operation::Serving, resident_weight_limit: None,
    }
}
fn decisions(onboard: u64) -> Decisions {
    Decisions {
        spark_reply_format: None,
        attention: vec![LayerMode::Whole {
            gpu: 0,
            ffn: FfnMode::Owner,
        }],
        onboard_layers: vec![(0..onboard as usize).collect()],
        expert_homes: (0..onboard as usize).map(|layer|
            (layer, ExpertDestination::Coordinator { gpus: vec![0] })).collect(),
        embedding: None,
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
        request: request(),
        decisions: decisions(0),
        pool_tokens: 64,
        budgets,
        facts,
        phases,
    }
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
fn immutable_reply_format_flows_to_decisions_and_explicit_conflicts_refuse() {
    for resolved in [None, Some(SparkReplyFormat::Bf16), Some(SparkReplyFormat::Fp8),
        Some(SparkReplyFormat::Nvfp4)] {
        let mut contract = Linear::new().prepare(&request()).unwrap();
        contract.spark_reply_format = resolved;
        let selected = vec![0; contract.variables.len()];
        assert_eq!(contract.decisions(&selected).unwrap().spark_reply_format, resolved);
        let recipe = Prepared(contract);
        for requested in [None, Some(SparkReplyFormat::Bf16), Some(SparkReplyFormat::Fp8),
            Some(SparkReplyFormat::Nvfp4)] {
            let req = Request { spark_reply_format: requested, ..request() };
            match solve(req.clone(), vec![budget(1 << 40)], &recipe) {
                Ok(plan) => {
                    assert!(requested.is_none() || requested == resolved);
                    assert_eq!(plan.request.spark_reply_format, requested);
                    assert_eq!(plan.decisions.spark_reply_format, resolved);
                    let encoded = serde_json::to_value(&plan).unwrap();
                    let decoded: ResolvedPlan = serde_json::from_value(encoded).unwrap();
                    assert_eq!(decoded, plan);
                }
                Err(Refusal::PinConflict) => assert!(requested.is_some() && requested != resolved),
                other => panic!("unexpected reply-format admission: {other:?}"),
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
fn embedding_and_remote_rank_domains_share_pool_first_constraints_and_exact_profiles() {
    let (contract, budgets) = home_contract();
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
    assert_eq!((automatic.pool_tokens, automatic.decisions.onboard()), (256, 3));
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
    assert!(matches!(solve(req, budgets.clone(), &Prepared(restricted)), Err(Refusal::Capacity { .. })));
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
    let req = Request { context: 1 << 20, onboard: Pin::Exact(15), ..request() };
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
    let req = Request { context: 1 << 20, ..request() };
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
        Err(Refusal::Capacity { .. })
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
            context: 129,
            ..request()
        },
        vec![budget(192)],
        &recipe,
    )
    .unwrap();
    assert_eq!(
        (automatic.request.context, automatic.pool_tokens),
        (129, 192)
    );
    assert!(matches!(
        solve(
            Request {
                context: 193,
                ..request()
            },
            vec![budget(192)],
            &recipe
        ),
        Err(Refusal::Capacity { .. })
    ));
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
                context,
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
