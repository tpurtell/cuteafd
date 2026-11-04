use super::*;
use crate::placement::planner::{self, Budget, Contract, Facts, Variable};

pub(super) fn cfg() -> GlmDsaConfig {
    GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap()
}

pub(super) fn target(device: u8) -> TargetBuffers {
    TargetBuffers {
        device,
        lead: device == 0,
        page_tokens: 128,
        maximum_pages: 128,
        context: 4096,
        partitioned: true,
        owned_layers: vec![true, false],
        split_heads: true,
        router: false,
        native_head: false,
        decode: Some(GlmScratch {
            programs: 81,
            topk: 1024,
        }),
        prefill: Some((
            PrefillShape { rows: 64, waves: 3 },
            GlmScratch {
                programs: 2048,
                topk: 7,
            },
        )),
        full_prefill_logits: false,
        payload_row_bytes: 0,
        core_rows: false,
    }
}

fn facts(target: &TargetBuffers, pages: u64) -> Facts {
    let mut contract = Contract::fixed(
        target.page_tokens,
        target.maximum_pages,
        2,
        Facts { resident_weights: vec![],
            phase_order: crate::placement::planner::PhaseOrder::DRAFT_THEN_TARGET,
            allocations: vec![],
            deferred: vec![],
            sequential: vec![],
            reservations: vec![], allowances: vec![],
        },
    );
    contract.variables = vec![
        Variable::Prefill(vec![target.prefill.as_ref().unwrap().0]),
        Variable::Lanes(vec![1]),
    ];
    contract.components = target.components(&cfg()).unwrap();
    contract.facts(&[0, 0], pages).unwrap()
}

fn budgets() -> Vec<Budget> {
    [
        Location::coordinator(0),
        Location::coordinator(1),
        Location::PINNED,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, location)| Budget {
        id: format!("test.{i}"),
        locations: vec![location],
        physical_bytes: 1 << 40,
        logical_bytes: 1 << 40,
        baseline: planner::Baseline::Forecast {
            occupied: 0,
            source: "test only".into(),
            assumed: false,
        },
        headroom: 0,
    })
    .collect()
}

#[test]
fn synchronous_upload_successor_uses_one_checked_workspace_span_and_decode_floor() {
    for (rows, waves, expected) in [(1, 1, 256), (64, 1, 256), (128, 3, 1536),
        (2752, 3, 33024), (4096, 4, 65536)] {
        let owner = sync_h2d_successor(PrefillShape { rows, waves }).unwrap();
        assert_eq!(owner.bytes, expected);
        assert_eq!(owner.shape, [expected]);
        assert_eq!(owner.id, "glm.upload.sync-successor");
        assert_eq!(owner.location, Location::PINNED);
        assert_eq!(owner.category, Category::Staging);
        assert_eq!(owner.lifetime, Lifetime { first: Phase::Warmup, last: Phase::Growth });
        assert!(owner.view.is_none());
    }
    for shape in [PrefillShape { rows: 0, waves: 1 }, PrefillShape { rows: 64, waves: 0 }] {
        assert!(matches!(sync_h2d_successor(shape), Err(Refusal::InvalidFact(_))));
    }
    for shape in [PrefillShape { rows: u64::MAX, waves: 2 },
        PrefillShape { rows: u64::MAX / 4 + 1, waves: 1 }] {
        assert!(matches!(sync_h2d_successor(shape), Err(Refusal::Overflow(_))));
    }
}

#[test]
fn prefix_backings_charge_tail_and_window_marks_but_no_target_or_draft_aliases() {
    let life = Lifetime { first: Phase::Warmup, last: Phase::Growth };
    for device in [0, 1] {
        for windows in [None, Some(0), Some(1), Some(8192)] {
            for slots in [0, 1, 3] {
                let spec = PrefixBuffers { device, draft_windows: windows, mark_slots: slots };
                let owners = spec.allocations(life).unwrap();
                assert_eq!(owners.len(), 1 + usize::from(windows.is_some() && slots > 0));
                assert_eq!(owners[0].id, format!("glm.prefix.gpu.{device}.tail-scratch"));
                assert_eq!(owners[0].bytes, 256);
                if let Some(windows) = windows.filter(|_| slots > 0) {
                    assert_eq!(owners[1].bytes, if windows == 0 {256} else {((windows + 16) * slots).max(256)});
                    assert_eq!(owners[1].id, format!("glm.prefix.gpu.{device}.draft-marks"));
                }
                assert!(owners.iter().all(|a| a.view.is_none() && a.lifetime == life
                    && a.location == Location::coordinator(device) && a.category == Category::Prefix));
                let mut contract = Contract::fixed(64, 16, 2, Facts { resident_weights: vec![],
                    phase_order: planner::PhaseOrder::DRAFT_THEN_TARGET, allocations: vec![],
                    deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![] });
                contract.variables = vec![Variable::Prefill(vec![PrefillShape {rows: 64, waves: 1}]), Variable::Lanes(vec![1])];
                contract.components = spec.components(16).unwrap();
                for pages in [1, 8, 16] {
                    assert_eq!(contract.facts(&[0, 0], pages).unwrap().allocations, owners);
                }
                assert!(spec.components(0).is_err());
            }
        }
    }
    for spec in [
        PrefixBuffers {device: 2, draft_windows: None, mark_slots: 0},
        PrefixBuffers {device: 0, draft_windows: Some(u64::MAX), mark_slots: 1},
        PrefixBuffers {device: 0, draft_windows: Some(1), mark_slots: u64::MAX},
    ] { assert!(spec.allocations(life).is_err()); }
}

#[test]
fn named_target_buffers_match_independent_pool_geometry_and_aliases() {
    for device in [0, 1] {
        let target = target(device);
        for pages in [1, 2, 31, 32, 33, 128] {
            let facts = facts(&target, pages);
            planner::evaluate(&facts, &budgets()).unwrap();
            let prefix = format!("glm.target.gpu.{device}");
            let get = |name: &str| {
                facts
                    .allocations
                    .iter()
                    .find(|a| a.id == format!("{prefix}.{name}"))
                    .unwrap()
            };
            assert_eq!(get("kv.0").bytes, (pages + 1) * 41984);
            assert_eq!(get("kv.1").bytes, 256);
            assert_eq!(get("index.0").bytes, (pages + 1) * 8448);
            assert!(!facts
                .allocations
                .iter()
                .any(|a| a.id == format!("{prefix}.index.1")));
            assert_eq!(get("rope").bytes, 4096 * 64 * 4);
            assert_eq!(get("decode.page_table.0").bytes, 64 * pages * 2 * 4);
            assert_eq!(get("prefill.page_table.2").bytes, (pages * 2 * 4).max(256));
            assert_eq!(
                get("prefill.upload.2.page_table").bytes,
                (pages * 2 * 4).max(256)
            );
            assert_eq!(get("prefill.rows.h").bytes, 192 * 6144 * 2);
            assert_eq!(
                get("prefill.slot.2.h").view.as_ref().unwrap().offset,
                128 * 6144 * 2
            );
            assert_eq!(get("prefill.slot.2.h").slot, Some(2));
            assert_eq!(
                get("prefill.logits").bytes,
                if device == 0 { 64 * 64 * 4 } else { 256 }
            );
            assert_eq!(get("prefill.slot.2.logits").slot, None);
            if device == 1 {
                assert_eq!(get("prefill.rows.wire").bytes, 256);
                assert_eq!(get("prefill.slot.2.wire").slot, None);
                assert!(!facts
                    .allocations
                    .iter()
                    .any(|a| a.id == format!("{prefix}.decode.upload.0.ids")));
            }
            // Views do not charge a second physical arena or shared logits allocation.
            let phase = planner::evaluate(&facts, &budgets()).unwrap();
            let physical: u64 = facts
                .allocations
                .iter()
                .filter(|a| a.view.is_none() && a.location == Location::coordinator(device))
                .map(|a| a.bytes)
                .sum();
            assert_eq!(
                phase
                    .iter()
                    .find(|r| r.budget == format!("test.{device}") && r.phase == Phase::Ready)
                    .unwrap()
                    .owned,
                physical
            );
        }
    }
}

#[test]
fn target_buffer_curves_cover_floor_breaks_and_reject_invalid_geometry() {
    let mut target = target(1);
    target.partitioned = false;
    target.page_tokens = 64;
    target.core_rows = true;
    target.payload_row_bytes = 6144 * 2;
    target.prefill.as_mut().unwrap().0.rows = 128;
    for pages in [1, 63, 64, 65, 128] {
        let facts = facts(&target, pages);
        planner::evaluate(&facts, &budgets()).unwrap();
        let get = |suffix: &str| {
            facts
                .allocations
                .iter()
                .find(|a| a.id.ends_with(suffix))
                .unwrap()
        };
        assert_eq!(get("prefill.page_table.0").bytes, (pages * 4).max(256));
        assert_eq!(get("prefill.rows.core.reduction").bytes, 384 * 6144 * 4);
        assert_eq!(
            get("prefill.slot.2.payload").view.as_ref().unwrap().offset,
            256 * 6144 * 2
        );
    }
    target.page_tokens = 65;
    assert!(matches!(
        target.components(&cfg()),
        Err(Refusal::InvalidFact(_))
    ));
    target.page_tokens = 64;
    target.partitioned = true;
    assert!(matches!(
        target.components(&cfg()),
        Err(Refusal::InvalidFact(_))
    ));
    target.page_tokens = 128;
    target.prefill.as_mut().unwrap().0.waves = 5;
    assert!(matches!(
        target.components(&cfg()),
        Err(Refusal::InvalidFact(_))
    ));
    target.prefill.as_mut().unwrap().0.waves = 3;
    target.context = u64::MAX;
    assert!(matches!(
        target.components(&cfg()),
        Err(Refusal::Overflow(_))
    ));
}
