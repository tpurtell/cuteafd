use super::*;
use crate::families::qwen3_5::tests::native_planner_fixture;
use crate::formats::block_dspark::{BlockDraftLayout, BlockDsparkConfig};

fn block_layout(fp8: bool) -> BlockDraftLayout {
    use crate::families::{glm5::draft_representation::GlmDraftRepresentation, qwen4::Qwen4KvCache};
    BlockDsparkConfig { hidden: 256, intermediate: 512, layers: 2, heads: 4, kv_heads: 2, head_dim: 64,
        eps: 1e-6, rope: crate::formats::block_dspark::DsparkRope::Default { rope_theta: 10000. },
        block: 8, mask_token: 0, rank: 256, taps: vec![0, 1], vocab: 512, window: None, causal: false }
        .runtime_layout(4, 4, 262144, 64, 188, if fp8 { GlmDraftRepresentation::Fp8Only }
            else { GlmDraftRepresentation::Bf16Only }, 2, Qwen4KvCache::Bf16).unwrap()
}
use crate::plan::checkpoint::Checkpoint;

#[test]
fn qwen_target_first_peak_keeps_kv_live_during_block_draft_packing() {
    use crate::placement::planner;

    let (cfg, weights, buffers) = fixture();
    let draft = block_layout(true);
    let mut recipe = facts(&cfg, &weights, &buffers);
    recipe.drafter = Some(DenseDraft::Block(&draft));
    let req = request();
    let contract = recipe.prepare(&req).unwrap();
    assert_eq!(contract.phase_order, PhaseOrder::TARGET_THEN_DRAFT);
    let selected = vec![0; contract.variables.len()];
    let pages = req.context / PAGE_ROWS;
    let actual = contract.facts(&selected, pages).unwrap();
    let expected = recipe.owner_templates(req.context).unwrap();
    assert_eq!(actual, expected);
    let target_bytes: u64 = actual.allocations.iter()
        .filter(|a| a.location == Location::coordinator(0)
            && a.lifetime.first == Phase::PostTargetKv && a.view.is_none())
        .map(|a| a.bytes).sum();
    let packing = draft.packing_sources.iter().map(|a| a.bytes as u64).max().unwrap();
    let mut room = budgets();
    for budget in &mut room {
        budget.physical_bytes = 1 << 50;
        budget.logical_bytes = 1 << 50;
    }
    let before = planner::evaluate(&expected, &room).unwrap();
    let after = planner::evaluate(&actual, &room).unwrap();
    assert_eq!(before, after);
    let peak = after.iter().filter(|row| row.budget == "coordinator.0")
        .max_by_key(|row| row.owned + row.opaque).unwrap();
    assert_eq!(peak.phase, Phase::DraftLoad);
    let ready = after.iter().find(|row|
        row.budget == "coordinator.0" && row.phase == Phase::Ready).unwrap();
    assert_eq!(peak.owned - ready.owned, packing);
    let mut wrong = actual.clone();
    wrong.phase_order = PhaseOrder::DRAFT_THEN_TARGET;
    let wrong_rows = planner::evaluate(&wrong, &room).unwrap();
    let wrong_load = wrong_rows.iter().find(|row|
        row.budget == "coordinator.0" && row.phase == Phase::DraftLoad).unwrap();
    assert_eq!(peak.owned - wrong_load.owned, target_bytes);
    let old = planner::solve(req.clone(), budgets(), &recipe).unwrap();
    let new = admit(req, budgets(), recipe).unwrap();
    assert_eq!(old, new);
    eprintln!("Qwen bounded native/block owner fixture: pool {} -> {}; DraftLoad peak {} -> {} B; retained target overlap {} B; packing {} B",
        old.pool_tokens, new.pool_tokens, peak.owned + peak.opaque,
        peak.owned + peak.opaque, target_bytes, packing);
}

fn request() -> Request {
    Request {
        context: 262144,
        pool: Pin::Exact(262144),
        onboard: Pin::Auto,
        prefill: Pin::Auto,
        spark_reply_format: None,
        operation: Operation::Serving,
    }
}
fn fixture() -> (Qwen35Config, NativeWeights, NativeBuffers) {
    let (dir, _, buffers, _) = native_planner_fixture();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = Qwen35Config::from_hf(&checkpoint.config).unwrap();
    let weights = NativeWeights::native(&checkpoint, &cfg, cuteafd_core::DType::F32).unwrap();
    (cfg, weights, buffers)
}
fn budgets() -> Vec<Budget> {
    forecast_budgets(
        &crate::placement::ARCH_CONTEXTS[1],
        96 << 30,
        0,
        (128 << 30, 128 << 30),
    )
    .unwrap()
}
fn facts<'a>(
    cfg: &'a Qwen35Config,
    weights: &'a NativeWeights,
    buffers: &'a NativeBuffers,
) -> DenseFacts<'a> {
    DenseFacts {
        cfg,
        weights,
        buffers,
        slots: 4,
        record_bytes: 4096,
        drafter: None,
        serving: None,
    }
}

#[test]
fn dense_onboarding_all_is_exact_zero_and_positive_pins_are_refused() {
    use crate::placement::{Onboard, OnboardSetting};

    let (cfg, weights, buffers) = fixture();
    for setting in [
        OnboardSetting::Auto,
        OnboardSetting::All,
        OnboardSetting::Layers(0),
    ] {
        let onboard = match setting.resolve(0) {
            Onboard::Auto => Pin::Auto,
            Onboard::Layers(n) => Pin::Exact(n as u64),
        };
        let plan = admit(
            Request {
                onboard,
                ..request()
            },
            budgets(),
            facts(&cfg, &weights, &buffers),
        )
        .unwrap();
        assert!(plan.decisions.onboard_layers.iter().all(Vec::is_empty));
        assert!(plan.decisions.expert_homes.is_empty());
        assert_eq!(plan.decisions.embedding, Some(EmbeddingHome::Device(0)));
        if setting == OnboardSetting::All {
            assert_eq!(plan.request.onboard, Pin::Exact(0));
        }
    }
    for layers in [1, 2, usize::MAX] {
        let Onboard::Layers(layers) = OnboardSetting::Layers(layers).resolve(0) else {
            panic!("explicit pin lost");
        };
        let refusal = admit(
            Request {
                onboard: Pin::Exact(layers as u64),
                ..request()
            },
            budgets(),
            facts(&cfg, &weights, &buffers),
        )
        .unwrap_err();
        assert!(matches!(
            refusal,
            Refusal::InvalidPin("model has 0 routed-expert layers")
        ));
    }
}

#[test]
fn dense_curves_match_independent_exact_inventory_and_monotone_page_neighbors() {
    let (cfg, weights, buffers) = fixture();
    let bf16 = block_layout(false);
    let fp8 = block_layout(true);
    for record_bytes in [2048, 4096, 8192] {
        for draft in [None, Some(&bf16), Some(&fp8)] {
            for sampling in [false, true] {
                let mut recipe = facts(&cfg, &weights, &buffers);
                recipe.record_bytes = record_bytes;
                recipe.drafter = draft.map(DenseDraft::Block);
                recipe.serving = Some(
                    ServingFacts::new(
                        &cfg,
                        &buffers,
                        record_bytes,
                        0,
                        ServingRequest {
                            concurrency: 4,
                            prefix_entries: 8,
                            host_cache_bytes: Some(512 << 20),
                            host_headroom: 8 << 30,
                            device_sampling: sampling,
                        },
                        128 << 30,
                    )
                    .unwrap(),
                );
                let contract = recipe.prepare(&request()).unwrap();
                let selected = vec![0; contract.variables.len()];
                let mut pages = std::collections::BTreeSet::from([
                    1,
                    2,
                    4095,
                    4096,
                    4097,
                    8191,
                    8192,
                    8193,
                    32767,
                    32768,
                    32769,
                    contract.maximum_pages - 1,
                    contract.maximum_pages,
                ]);
                for component in &contract.components {
                    for piece in &component.bytes.0 {
                        for boundary in [piece.first, piece.last] {
                            for p in boundary.saturating_sub(1)..=boundary.saturating_add(1) {
                                if p > 0 && p <= contract.maximum_pages {
                                    pages.insert(p);
                                }
                            }
                        }
                    }
                }
                let mut generous = budgets();
                for budget in &mut generous {
                    budget.physical_bytes = 1 << 60;
                    budget.logical_bytes = 1 << 60;
                    budget.baseline = Baseline::Forecast {
                        occupied: 0,
                        source: "curve oracle".into(),
                        assumed: false,
                    };
                    budget.headroom = 0;
                }
                recipe
                    .serving
                    .unwrap()
                    .apply_budgets(&mut generous)
                    .unwrap();
                for pages in pages {
                    let actual = contract.facts(&selected, pages).unwrap();
                    let expected = recipe.owner_templates(pages * PAGE_ROWS).unwrap();
                    assert_eq!(
                        actual, expected,
                        "page {pages}, record {record_bytes}, sampling {sampling}"
                    );
                    let rows = evaluate(&actual, &generous).unwrap();
                    if pages < contract.maximum_pages {
                        let next = contract.facts(&selected, pages + 1).unwrap();
                        let next_rows = evaluate(&next, &generous).unwrap();
                        assert!(rows
                            .iter()
                            .zip(next_rows)
                            .all(|(p, next)| next.owned + next.opaque >= p.owned + p.opaque));
                    }
                    if pages >= request().context / PAGE_ROWS {
                        let resolved = admit(
                            Request {
                                pool: Pin::Exact(pages * PAGE_ROWS),
                                ..request()
                            },
                            generous.clone(),
                            DenseFacts { ..recipe },
                        )
                        .unwrap();
                        assert_eq!(resolved.phases, rows);
                    }
                }
            }
        }
    }
}

#[test]
fn logical_thirty_gib_cap_preserves_pro_overheads_and_real_5090_identity() {
    let pro = &crate::placement::ARCH_CONTEXTS[1];
    let small = &crate::placement::ARCH_CONTEXTS[0];
    let host = (128 << 30, 128 << 30);
    let modules = 37 << 20;
    let capped = forecast_budgets(pro, 30 << 30, modules, host).unwrap();
    assert_eq!(capped[0].physical_bytes, pro.total_bytes);
    assert_eq!(capped[0].logical_bytes, 30 << 30);
    assert_eq!(
        capped[0].baseline,
        Baseline::Forecast {
            occupied: pro.context_bytes + modules,
            source: pro.source.into(),
            assumed: false,
        }
    );
    let live = sampled_budgets(
        pro.total_bytes - pro.context_bytes,
        pro.total_bytes,
        Some(30 << 30),
        host,
    )
    .unwrap();
    assert_eq!(live[0].physical_bytes, pro.total_bytes);
    assert_eq!(
        live[0].available().unwrap(),
        (30 << 30) - pro.context_bytes - (1 << 30)
    );
    assert!(matches!(&live[0].baseline, Baseline::Sampled { source, .. }
        if source.contains(pro.source) && !source.contains("5090")));
    let total = 32 << 30;
    let actual_5090 =
        sampled_budgets(total - small.context_bytes, total, Some(30 << 30), host).unwrap();
    assert_eq!(actual_5090[0].physical_bytes, total);
    assert!(
        matches!(&actual_5090[0].baseline, Baseline::Sampled { source, .. }
        if source.contains(small.source) && source.contains("5090"))
    );
    let (cfg, weights, buffers) = fixture();
    let mut req = request();
    req.pool = Pin::Auto;
    let large = admit(req.clone(), live, facts(&cfg, &weights, &buffers)).unwrap();
    let small = admit(req, actual_5090, facts(&cfg, &weights, &buffers)).unwrap();
    assert_eq!((large.pool_tokens, small.pool_tokens), (2 << 20, 1 << 20));
}

#[test]
fn loading_heap_allowance_is_host_only_and_cannot_shrink_the_pool() {
    let (cfg, mut weights, buffers) = fixture();
    let mut req = request();
    req.pool = Pin::Auto;
    let first = admit(req.clone(), budgets(), facts(&cfg, &weights, &buffers)).unwrap();
    weights.loading_host_bytes += 8 << 30;
    let larger = admit(req, budgets(), facts(&cfg, &weights, &buffers)).unwrap();
    assert_eq!(larger.pool_tokens, first.pool_tokens);
    assert_eq!(
        larger
            .phases
            .iter()
            .filter(|p| p.budget == "coordinator.0")
            .collect::<Vec<_>>(),
        first
            .phases
            .iter()
            .filter(|p| p.budget == "coordinator.0")
            .collect::<Vec<_>>()
    );
    for row in larger.phases.iter().filter(|p| p.budget == "host") {
        assert_eq!(
            row.opaque,
            if row.phase == Phase::Load {
                weights.loading_host_bytes
            } else {
                0
            }
        );
    }
    assert!(larger
        .facts
        .allowances
        .iter()
        .any(|a| a.location == Location::HOST
            && a.assumed
            && a.lifetime
                == Lifetime {
                    first: Phase::Load,
                    last: Phase::Load
                }));
}

#[test]
fn block_loading_heap_allowance_matches_curves_at_zero_and_nonzero_bytes() {
    let (cfg, mut weights, buffers) = fixture();
    weights.loading_host_bytes = 0;
    for bytes in [0, 1, 512 << 20] {
        let mut layout = block_layout(true);
        layout.loading_host_bytes = bytes;
        let mut recipe = facts(&cfg, &weights, &buffers);
        recipe.drafter = Some(DenseDraft::Block(&layout));
        let contract = recipe.prepare(&request()).unwrap();
        let selected = vec![0; contract.variables.len()];
        for pages in [1, 4096, contract.maximum_pages] {
            let exact = recipe.owner_templates(pages * PAGE_ROWS).unwrap();
            let actual = contract.facts(&selected, pages).unwrap();
            assert_eq!(actual, exact, "pages {pages}, heap {bytes}");
            assert!(exact.allowances.iter().all(|allowance| allowance.bytes > 0));
            let heap: Vec<_> = exact.allowances.iter().filter(|allowance|
                allowance.lifetime.first == Phase::DraftLoad).collect();
            assert_eq!(heap.len(), usize::from(bytes > 0));
            if bytes > 0 {
                assert_eq!(heap[0].location, Location::HOST);
                assert_eq!(heap[0].bytes, bytes);
                assert_eq!(heap[0].lifetime.last, Phase::DraftLoad);
                assert!(heap[0].assumed && heap[0].source.starts_with("block dSpark headers:"));
            }
        }
    }
}

#[test]
fn block_drafter_signed_row_ceiling_bounds_contract_and_refuses_larger_exact_pins() {
    let (cfg, weights, buffers) = fixture();
    let no_draft = facts(&cfg, &weights, &buffers).prepare(&request()).unwrap();
    assert_eq!(no_draft.maximum_pages, u32::MAX as u64 / PAGE_ROWS);
    let capacity = BlockDraftLayout::MAX_POOL_TOKENS as u64;
    for fp8 in [false, true] {
        let layout = block_layout(fp8);
        let mut recipe = facts(&cfg, &weights, &buffers);
        recipe.drafter = Some(DenseDraft::Block(&layout));
        let contract = recipe.prepare(&request()).unwrap();
        assert_eq!(contract.maximum_pages * PAGE_ROWS, capacity);
        assert_eq!(contract.facts(&vec![0; contract.variables.len()], contract.maximum_pages).unwrap(),
            recipe.owner_templates(capacity).unwrap());
        let mut generous = budgets();
        for budget in &mut generous {
            budget.physical_bytes = 1 << 60;
            budget.logical_bytes = 1 << 60;
        }
        let plan = admit(Request { pool: Pin::Exact(capacity), ..request() },
            generous.clone(), recipe).unwrap();
        assert_eq!(plan.pool_tokens, capacity);
        let refusal = admit(Request { pool: Pin::Exact(capacity + PAGE_ROWS), ..request() },
            generous, recipe).unwrap_err();
        assert!(matches!(refusal, Refusal::InvalidPin("pool exceeds compiled extent")));
    }
}

#[test]
fn native_scratch_prefix_and_pinned_staging_are_individual_owners() {
    let (cfg, weights, buffers) = fixture();
    let plan = admit(request(), budgets(), facts(&cfg, &weights, &buffers)).unwrap();
    let owner = |id: &str| plan.facts.allocations.iter().find(|a| a.id == id).unwrap();
    assert_eq!(owner("planes.6").bytes, 1024);
    assert_eq!(owner("prefix").bytes, buffers.prefix as u64);
    assert_eq!(
        owner("upload.sync").bytes,
        weights
            .allocations
            .iter()
            .map(|a| a.upload_bytes)
            .max()
            .unwrap()
    );
    for index in 7..13 {
        assert_eq!(
            owner(&format!("planes.{index}.upload")).bytes,
            buffers.planes[index] as u64
        );
        assert_eq!(
            owner(&format!("planes.{index}.upload")).location,
            Location::PINNED
        );
    }
    assert_eq!(owner("ids.upload").bytes, buffers.ids as u64);
    assert_eq!(
        owner("embedding.upload").lifetime,
        Lifetime {
            first: Phase::Load,
            last: Phase::Load
        }
    );
    assert!(plan
        .facts
        .allowances
        .iter()
        .any(|a| a.bytes == 512 << 20 && a.assumed));
    let conv_view = owner("gdn.conv.0.slot.1");
    assert_eq!(conv_view.slot, Some(1));
    assert_eq!(conv_view.view.as_ref().unwrap().backing, "gdn.conv");
    assert_eq!(conv_view.view.as_ref().unwrap().offset, conv_view.bytes);
    assert!(owner("prefix.mark.0").view.is_some());
    let mut larger = buffers.clone();
    larger.planes[6] += 1 << 30;
    let result = admit(request(), budgets(), facts(&cfg, &weights, &larger)).unwrap();
    let ready = |p: &ResolvedPlan| {
        p.phases
            .iter()
            .find(|r| r.budget == "coordinator.0" && r.phase == Phase::Ready)
            .unwrap()
            .owned
    };
    assert_eq!(ready(&result) - ready(&plan), 1 << 30);
}

#[test]
fn block_drafter_pages_sources_and_ready_owners_are_exact() {
    let (cfg, weights, buffers) = fixture();
    let layout = block_layout(true);
    let mut recipe = facts(&cfg, &weights, &buffers);
    recipe.drafter = Some(DenseDraft::Block(&layout));
    let contract = recipe.prepare(&request()).unwrap();
    let selected = vec![0; contract.variables.len()];
    let small = contract.facts(&selected, 262144 / PAGE_ROWS).unwrap();
    let larger = contract.facts(&selected, 524288 / PAGE_ROWS).unwrap();
    assert!(small.deferred.is_empty());
    assert_eq!(small.sequential.len(), 1);
    assert_eq!(small.sequential[0].members.len(), 9);
    for (a, b) in small.allocations.iter().zip(&larger.allocations) {
        let delta = if a.id.ends_with(".k_ring") || a.id.ends_with(".v_ring") {
            262144 * layout.cache.record_bytes as u64 / 2
        } else if a.id.starts_with("kv.") {
            assert!(a.view.is_none());
            262144 * recipe.record_bytes
        } else { 0 };
        assert_eq!(b.bytes - a.bytes, delta, "{}", a.id);
    }
    assert!(!small.allocations.iter().any(|o| o.id.ends_with("taps.upload") || o.id.ends_with("h.upload")));
    let plan = admit(request(), budgets(), recipe).unwrap();
    assert!(matches!(reconcile(&plan, Phase::Ready, &[]), Err(Refusal::Reconciliation(errors))
        if errors.contains(&Mismatch::Missing("draft.block.weight.fc.values".into()))));
}

#[test]
fn malformed_native_inventory_and_operation_fail_before_admission() {
    let (cfg, mut weights, buffers) = fixture();
    assert!(matches!(
        admit(
            Request {
                spark_reply_format: None,
                operation: Operation::Scoring,
                ..request()
            },
            budgets(),
            facts(&cfg, &weights, &buffers)
        ),
        Err(Refusal::InvalidFact(_))
    ));
    weights.layers.pop();
    assert!(matches!(
        admit(request(), budgets(), facts(&cfg, &weights, &buffers)),
        Err(Refusal::InvalidFact(_))
    ));
}

#[test]
fn serving_host_quota_charges_full_context_draft_pages_before_admission() {
    let (cfg, _, buffers) = fixture();
    let request = ServingRequest {
        concurrency: 4,
        prefix_entries: 8,
        host_cache_bytes: None,
        host_headroom: 8 << 30,
        device_sampling: false,
    };
    let plain = ServingFacts::new(&cfg, &buffers, 4096, 0, request, 128 << 30).unwrap();
    let draft =
        ServingFacts::new(&cfg, &buffers, 4096, 64 * 4096 + 8192, request, 128 << 30).unwrap();
    let full_layers = cfg
        .attention
        .iter()
        .filter(|&&a| a == HybridAttention::Full)
        .count() as u64;
    let geometry = cuteafd_core::prefix::HostCacheGeometry {
        unit_rows: PAGE_ROWS,
        page_bytes: PAGE_ROWS * 4096 * full_layers,
        mark_bytes: buffers.mark_bytes as u64,
        draft_bytes: 0,
        draft_page_bytes: 64 * 4096 + 8192,
    };
    assert_eq!(
        draft.host_quota,
        geometry
            .resolve(
                None,
                cfg.max_context as u64,
                4,
                128 << 30,
                geometry.chunk_bytes()
            )
            .unwrap()
    );
    assert!(draft.host_quota > plain.host_quota);
}

#[test]
fn serving_ready_requires_exact_scheduler_sampler_and_whole_host_chunks() {
    let (cfg, weights, buffers) = fixture();
    let chunk = 256 << 20;
    let serving = ServingFacts::new(
        &cfg,
        &buffers,
        4096,
        0,
        ServingRequest {
            concurrency: 4,
            prefix_entries: 8,
            host_cache_bytes: Some(2 * chunk + 17),
            host_headroom: 8 << 30,
            device_sampling: true,
        },
        128 << 30,
    )
    .unwrap();
    assert_eq!(
        (serving.host_quota, serving.host_chunk),
        (2 * chunk + 17, chunk)
    );
    let mut recipe = facts(&cfg, &weights, &buffers);
    recipe.serving = Some(serving);
    let plan = admit(request(), budgets(), recipe).unwrap();
    let warm = plan
        .facts
        .allocations
        .iter()
        .filter(|a| a.lifetime.first == Phase::Warmup)
        .collect::<Vec<_>>();
    assert_eq!(warm.len(), 2 + 22 + 2);
    assert_eq!(
        warm.iter()
            .filter(|a| a.group == "retained host prefix chunk")
            .map(|a| a.bytes)
            .sum::<u64>(),
        2 * chunk
    );
    let row = |phase| {
        plan.phases
            .iter()
            .find(|p| p.budget == "host" && p.phase == phase)
            .unwrap()
    };
    assert_eq!(row(Phase::Ready).opaque, serving.host_runtime);
    assert_eq!(row(Phase::Load).opaque, weights.loading_host_bytes);
    let mut addresses = std::collections::BTreeMap::new();
    let mut cursor = 4096u64;
    let mut all = Vec::new();
    for allocation in plan.facts.allocations.iter().filter(|a| a.view.is_none()) {
        addresses.insert(allocation.id.as_str(), cursor);
        all.push(Receipt {
            allocation: allocation.clone(),
            address: cursor,
        });
        cursor += allocation.bytes + 4096;
    }
    for allocation in plan.facts.allocations.iter().filter(|a| a.view.is_some()) {
        let view = allocation.view.as_ref().unwrap();
        all.push(Receipt {
            allocation: allocation.clone(),
            address: addresses[view.backing.as_str()] + view.offset,
        });
    }
    let receipts = all
        .iter()
        .filter(|r| r.allocation.lifetime.contains(Phase::Ready, plan.facts.phase_order))
        .cloned()
        .collect::<Vec<_>>();
    let target = all
        .iter()
        .filter(|r| r.allocation.lifetime.contains(Phase::PostTargetKv, plan.facts.phase_order))
        .cloned()
        .collect::<Vec<_>>();
    reconcile(&plan, Phase::PostTargetKv, &target).unwrap();
    reconcile(&plan, Phase::Ready, &receipts).unwrap();
    let observations = all
        .iter()
        .map(|receipt| Observation {
            receipt: receipt.clone(),
            created: receipt.allocation.lifetime.first,
            freed: (!plan.facts.phase_order.at_or_before(Phase::Ready, receipt.allocation.lifetime.last))
                .then_some(receipt.allocation.lifetime.last),
        })
        .collect::<Vec<_>>();
    reconcile_history(&plan, Phase::Ready, &observations).unwrap();
    for id in [
        "selector.output",
        "selector.landing",
        "selector.sampler.mask_pinned",
        "selector.sampler.bitmap",
        "host.prefix.1",
    ] {
        let missing = receipts
            .iter()
            .filter(|r| r.allocation.id != id)
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            matches!(reconcile(&plan, Phase::Ready, &missing), Err(Refusal::Reconciliation(errors))
            if errors.contains(&Mismatch::Missing(id.into())))
        );
        for domain in [false, true] {
            let mut wrong = receipts.clone();
            let owner = wrong.iter_mut().find(|r| r.allocation.id == id).unwrap();
            if domain {
                owner.allocation.location = Location::HOST;
            } else {
                owner.allocation.bytes += 1;
            }
            assert!(
                matches!(reconcile(&plan, Phase::Ready, &wrong), Err(Refusal::Reconciliation(errors))
                if errors.contains(&Mismatch::Contract(id.into())))
            );
        }
    }
    let mut premature = observations;
    premature
        .iter_mut()
        .find(|o| o.receipt.allocation.id == "selector.output")
        .unwrap()
        .created = Phase::PostTargetKv;
    assert!(
        matches!(reconcile_history(&plan, Phase::Ready, &premature), Err(Refusal::Reconciliation(errors))
        if errors.contains(&Mismatch::Lifetime("selector.output".into())))
    );
    assert!(reconcile(&plan, Phase::Ready, &target).is_err());
}

#[test]
fn host_selector_and_disabled_prefix_have_no_sampler_or_chunk_owners() {
    let (cfg, weights, buffers) = fixture();
    for (entries, bytes) in [(0, None), (4, Some(0))] {
        let serving = ServingFacts::new(
            &cfg,
            &buffers,
            4096,
            0,
            ServingRequest {
                concurrency: 4,
                prefix_entries: entries,
                host_cache_bytes: bytes,
                host_headroom: 8 << 30,
                device_sampling: false,
            },
            128 << 30,
        )
        .unwrap();
        let mut recipe = facts(&cfg, &weights, &buffers);
        recipe.serving = Some(serving);
        let plan = admit(request(), budgets(), recipe).unwrap();
        assert_eq!(
            plan.facts
                .allocations
                .iter()
                .filter(|a| a.lifetime.first == Phase::Warmup)
                .count(),
            2
        );
        assert!(!plan
            .facts
            .allocations
            .iter()
            .any(|a| a.id.starts_with("host.prefix.") || a.id.starts_with("selector.sampler.")));
    }
}
