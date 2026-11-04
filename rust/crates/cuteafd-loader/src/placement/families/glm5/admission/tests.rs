use super::*;
use crate::placement::{FfnMode, planner};
use cuteafd_core::memory_layout::Category;

fn cfg() -> GlmDsaConfig {
    GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap()
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
    let pages = req.context / recipe.page_tokens;
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
        context: 4096,
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
        cfg,
        context: 4096,
        page_tokens: 128,
        maximum_pages: 64,
        attention: vec![LayerMode::ContextSplit; cfg.layers],
        gpus: 2,
        spark_links: true,
        prefill,
        lanes: vec![1],
        embedding: vec![EmbeddingHome::Device(0), EmbeddingHome::HostPinned],
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
        returns: vec![],
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
                        core_rows: false,
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
                row_loading: super::super::weights::RowLoading::default(),
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
                    33 * super::super::RECORD_PAGE_BYTES
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

#[test]
fn owned_graph_census_is_fixed_above_full_context_and_separates_opaque_growth() {
    use crate::serving_capacity::glm_graphs::glm_owned_startup_graphs;
    let owners = [0, 1];
    let tp2 = [false, true];
    for context in [64, 1024, 1025, 131072, 131073, 262144, 262145, 1 << 20] {
        let graphs = OwnedGraphs::new(context, &owners, &tp2).unwrap();
        let floor = (context as usize).div_ceil(super::super::PAGE_ROWS as usize);
        for pages in [floor, floor + 1, floor * 2, 32768] {
            let actual = glm_owned_startup_graphs(context as usize, pages, &owners, &tp2).unwrap();
            assert_eq!(&actual, graphs.startup());
        }
        let components = graphs.components(32768).unwrap();
        assert_eq!(components.len(), 4);
        for rank in 0..2 {
            for (index, at) in [Phase::Warmup, Phase::Growth].into_iter().enumerate() {
                let component = &components[2 * rank + index];
                let ComponentKind::Allowance(allowance) = &component.kind else {
                    panic!("graphs are opaque, not physical allocations")
                };
                assert_eq!(allowance.location, Location::coordinator(rank as u8));
                assert_eq!(
                    allowance.lifetime,
                    Lifetime {
                        first: at,
                        last: Phase::Growth
                    }
                );
                assert!(allowance.assumed);
                assert!(allowance.source.contains("5090 unmeasured"));
                let expected = if index == 0 {
                    graphs.startup().inventory.at_ready(rank)
                } else {
                    graphs.startup().inventory.growth(rank)
                };
                assert_eq!(allowance.bytes, expected);
                for pages in [1, floor as u64, 32768] {
                    assert_eq!(component.bytes.at(pages).unwrap(), expected);
                }
            }
        }
    }
    assert!(OwnedGraphs::new(0, &owners, &tp2).is_err());
    assert!(OwnedGraphs::new(64, &[2, 0], &tp2).is_err());
    assert!(OwnedGraphs::new(64, &owners, &[false]).is_err());
    assert!(
        OwnedGraphs::new(64, &owners, &tp2)
            .unwrap()
            .components(0)
            .is_err()
    );
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
            value: OwnedGraphs::new(context, &[0, 1], &[false, count != 0]).unwrap(),
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
    request.context = context;
    request.pool = Pin::Auto;
    request.onboard = Pin::Exact(0);
    request.prefill = Pin::Exact(recipe.prefill[0]);
    let contract = recipe.prepare(&request).unwrap();
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
    assert_eq!(plan.request.context, context);
    assert_eq!(contract.maximum_pages, 32768);
    // Actual startup census differs below the full-context floor; those keys
    // may not be used to make an otherwise infeasible full request appear to fit.
    let smaller = OwnedGraphs::new(context - 64, &[0, 1], &[false, false]).unwrap();
    assert_ne!(
        smaller.startup().keys,
        recipe.graphs[0].value.startup().keys
    );
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
        Err(Refusal::Capacity { .. })
    ));
    request.context -= 64;
    assert!(matches!(
        recipe.prepare(&request),
        Err(Refusal::InvalidFact(_))
    ));
}

#[test]
fn owned_graph_census_must_match_every_activated_onboard_count_and_attention_owner() {
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
    recipe.graphs[0].value = OwnedGraphs::new(4096, &[1, 0], &[false, false]).unwrap();
    assert!(matches!(
        recipe.prepare(&request),
        Err(Refusal::InvalidFact(_))
    ));
    recipe.graphs[0].value = OwnedGraphs::new(2048, &[0, 1], &[false, false]).unwrap();
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
    beyond.context = 8192;
    assert!(matches!(
        base.prepare(&beyond),
        Err(Refusal::InvalidFact(_))
    ));
}

#[test]
fn phase_returns_compose_per_physical_link_and_retain_bf16_fallback() {
    use cuteafd_core::SparkReplyFormat::{Bf16, Fp8, Nvfp4};
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    for (link, ranks) in [vec![0, 1], vec![2, 3]].into_iter().enumerate() {
        recipe.returns.push(Case {
            when: vec![],
            value: ReplyReturns {
                id: format!("returns.decode.{link}"),
                device: 0,
                ranks: ranks.clone(),
                phase: ReplyPhase::Decode,
            },
        });
        for &shape in &recipe.prefill {
            recipe.returns.push(Case {
                when: vec![When::Prefill(shape)],
                value: ReplyReturns {
                    id: format!("returns.prefill.{link}"),
                    device: 0,
                    ranks: ranks.clone(),
                    phase: ReplyPhase::Prefill(shape),
                },
            });
        }
    }
    for prefill in [Bf16, Fp8, Nvfp4] {
        for decode in [Bf16, Fp8, Nvfp4] {
            for &shape in &recipe.prefill {
                let request = Request {
                    onboard: Pin::Exact(1),
                    prefill: Pin::Exact(shape),
                    spark_reply_formats: SparkReplyFormats {
                        prefill: Some(prefill),
                        decode: Some(decode),
                    },
                    ..request()
                };
                let plan = planner::solve(request, budgets(), &recipe).unwrap();
                assert_eq!(
                    plan.decisions.spark_reply_formats,
                    Some(SparkReplyFormats {
                        prefill: Some(prefill),
                        decode: Some(decode)
                    })
                );
                let returns: Vec<_> = plan
                    .facts
                    .allocations
                    .iter()
                    .filter(|a| a.id.starts_with("returns."))
                    .collect();
                assert_eq!(
                    returns.len(),
                    8,
                    "two links per phase, two rank reserves per link"
                );
                for phase in ["decode", "prefill"] {
                    let rows = if phase == "decode" {
                        64
                    } else {
                        shape.rows * shape.waves
                    };
                    let expected = cuteafd_core::serving_capacity::spark_intake_allocation_bytes(
                        rows * cfg.hidden as u64 * 2,
                    )
                    .unwrap();
                    let owners: Vec<_> = returns
                        .iter()
                        .filter(|a| a.id.starts_with(&format!("returns.{phase}.")))
                        .collect();
                    assert_eq!(owners.len(), 4);
                    assert!(owners.iter().all(|a| a.bytes == expected
                        && a.location == Location::coordinator(0)
                        && a.view.is_none()));
                    assert_eq!(owners.iter().map(|a| a.bytes).sum::<u64>(), 4 * expected);
                }
                assert_eq!(plan.decisions.onboard_layers, vec![vec![1], vec![1]]);
                assert!(
                    !plan
                        .facts
                        .allocations
                        .iter()
                        .any(|a| a.id.contains("codec"))
                );
            }
        }
    }
}

#[test]
fn return_owners_refuse_wrong_domains_links_and_duplicate_backings() {
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    let shape = recipe.prefill[0];
    let good = Case {
        when: vec![When::Prefill(shape)],
        value: ReplyReturns {
            id: "returns.prefill".into(),
            device: 0,
            ranks: vec![0, 1],
            phase: ReplyPhase::Prefill(shape),
        },
    };
    for bad in [
        Case {
            when: vec![],
            ..good.clone()
        },
        Case {
            value: ReplyReturns {
                device: 2,
                ..good.value.clone()
            },
            ..good.clone()
        },
        Case {
            value: ReplyReturns {
                ranks: vec![1, 1],
                ..good.value.clone()
            },
            ..good.clone()
        },
        Case {
            value: ReplyReturns {
                ranks: vec![8],
                ..good.value.clone()
            },
            ..good.clone()
        },
        Case {
            value: ReplyReturns {
                ranks: vec![],
                ..good.value.clone()
            },
            ..good.clone()
        },
        Case {
            value: ReplyReturns {
                id: String::new(),
                ..good.value.clone()
            },
            ..good.clone()
        },
    ] {
        recipe.returns = vec![bad];
        assert!(matches!(
            recipe.prepare(&request()),
            Err(Refusal::InvalidFact(_))
        ));
    }
    recipe.returns = vec![good.clone()];
    recipe.spark_links = false;
    assert!(matches!(
        recipe.prepare(&request()),
        Err(Refusal::InvalidFact(_))
    ));
    recipe.spark_links = true;
    recipe.returns = vec![good.clone(), good];
    assert!(matches!(
        planner::solve(request(), budgets(), &recipe),
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
