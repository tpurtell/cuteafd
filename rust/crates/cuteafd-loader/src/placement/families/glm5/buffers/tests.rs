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
        Facts {
            phase_order: crate::placement::planner::PhaseOrder::DRAFT_THEN_TARGET,
            allocations: vec![],
            deferred: vec![],
            sequential: vec![],
            allowances: vec![],
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
