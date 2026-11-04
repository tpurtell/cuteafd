use super::*;
use crate::families::glm5::{GlmDsaConfig, core::CoreStageLayout};
use crate::placement::planner::{self, Baseline, Budget, Contract, Facts, PrefillShape, Variable};
use crate::plan::exlr8::dense::scratch::LinearScratchLayout;

fn layouts() -> HookBuffers {
    let cfg = GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap();
    let native = |rows| {
        ScratchLayout::new(
            &cfg,
            2,
            128,
            rows,
            LinearScratchLayout {
                rotated: rows * 128 * 2,
                table: 256,
                block_scales: 256,
                partial: 1024,
                records: (rows > 64).then_some(640),
                acc: rows * 256 * 4,
            },
        )
        .unwrap()
    };
    HookBuffers {
        device: 1,
        maximum_pages: 16,
        native: Some([native(64), native(128)]),
        core: Some(CoreRankLayout {
            body: vec![],
            stages: Some([
                CoreStageLayout::new(&cfg, 64, 2, false).unwrap(),
                CoreStageLayout::new(&cfg, 128, 2, false).unwrap(),
            ]),
            decode: Some(ProjectionScratchLayout::new(128, 256, 64).unwrap()),
            prefill: Some(ProjectionScratchLayout::new(128, 256, 128).unwrap()),
            target_head: Some(ProjectionScratchLayout::new(128, 256, 64).unwrap()),
            draft_head: Some(ProjectionScratchLayout::new(128, 256, 96).unwrap()),
            retained_rows: 448,
            receive_suffix: 7,
        }),
        shared_workspace: Some(crate::plan::exlr8::ExecutionWorkspace {
            ranges: std::array::from_fn(|index| ((index + usize::from(index > 9)) * 256,
                if index == 9 { 512 } else { 256 })),
            bytes: 3072, rows: 64, top_k: 1, max_groups: 4, hidden: 128, units: 16,
            unit_limit: 16, intermediate_bytes: 256, compact: false, unit_tables: false,
        }),
    }
}

#[test]
fn native_hooks_name_physical_owners_and_k6_route_aliases_once() {
    let hooks = layouts();
    let mut contract = Contract::fixed(
        64,
        16,
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
        Variable::Prefill(vec![PrefillShape {
            rows: 128,
            waves: 3,
        }]),
        Variable::Lanes(vec![1]),
    ];
    contract.components = hooks.components().unwrap();
    let facts = contract.facts(&[0, 0], 8).unwrap();
    let get = |suffix: &str| {
        facts
            .allocations
            .iter()
            .find(|a| a.id.ends_with(suffix))
            .unwrap()
    };
    assert_eq!(get("decode.native.linear.rotated").bytes, 64 * 128 * 2);
    assert_eq!(get("prefill.native.linear.records").bytes, 640);
    assert!(
        !facts
            .allocations
            .iter()
            .any(|a| a.id.ends_with("decode.native.linear.records"))
    );
    assert_eq!(get("decode.native.stage.qkv").bytes, 64 * 2624 * 2);
    assert_eq!(get("prefill.native.stage.values").bytes, 128 * 2 * 256 * 2);
    assert_eq!(get("decode.core.projection.storage.table").bytes, 8);
    assert_eq!(get("decode.core.projection.storage.scales").bytes, 2);
    let partial = get("decode.core.projection.view.partial");
    assert_eq!(partial.bytes, 64 * 256 * 4);
    assert_eq!(
        partial.view.as_ref().unwrap().backing,
        "glm.target.gpu.1.decode.core.projection.storage.route_arena"
    );
    assert_eq!(
        get("prefill.core.projection.storage.route_arena").bytes,
        128 * 256 * 4
    );
    assert_eq!(get("core.draft_head.storage.rotated").bytes, 96 * 128 * 2);
    assert_eq!(
        get("core.draft_head.storage.rotated").category,
        Category::Drafter
    );
    assert_eq!(
        get("core.target_head.storage.rotated").lifetime,
        Lifetime::TARGET
    );
    assert!(
        facts
            .allocations
            .iter()
            .all(|a| a.location == Location::coordinator(1))
    );
    // Retained row planes and peer suffix belong to their own recipes, not these hooks.
    let expected: u64 = hooks
        .native
        .as_ref()
        .unwrap()
        .iter()
        .map(|l| l.bytes().unwrap() as u64)
        .sum::<u64>()
        + hooks
            .core
            .as_ref()
            .unwrap()
            .decode
            .as_ref()
            .unwrap()
            .bytes()
            .unwrap() as u64
        + hooks
            .core
            .as_ref()
            .unwrap()
            .prefill
            .as_ref()
            .unwrap()
            .bytes()
            .unwrap() as u64
        + hooks
            .core
            .as_ref()
            .unwrap()
            .target_head
            .as_ref()
            .unwrap()
            .bytes()
            .unwrap() as u64
        + hooks
            .core
            .as_ref()
            .unwrap()
            .draft_head
            .as_ref()
            .unwrap()
            .bytes()
            .unwrap() as u64
        + hooks
            .core
            .as_ref()
            .unwrap()
            .stages
            .as_ref()
            .unwrap()
            .iter()
            .map(|s| s.bytes().unwrap() as u64)
            .sum::<u64>()
        + 2 * hooks.shared_workspace.as_ref().unwrap().bytes as u64;
    for kind in ["decode", "prefill"] {
        assert_eq!(get(&format!("{kind}.native.shared.arena")).bytes, 3072);
        assert_eq!(get(&format!("{kind}.native.shared.fc2")).view.as_ref().unwrap().backing,
            format!("glm.target.gpu.1.{kind}.native.shared.arena"));
    }
    let budget = Budget {
        id: "physical".into(),
        locations: vec![Location::coordinator(1)],
        physical_bytes: 1 << 40,
        logical_bytes: 1 << 40,
        baseline: Baseline::Forecast {
            occupied: 0,
            source: "test".into(),
            assumed: false,
        },
        headroom: 0,
    };
    let phases = planner::evaluate(&facts, &[budget]).unwrap();
    assert_eq!(
        phases
            .iter()
            .find(|p| p.phase == Phase::Ready)
            .unwrap()
            .owned,
        expected
    );
    let early = phases
        .iter()
        .find(|p| p.phase == Phase::PostTargetKv)
        .unwrap()
        .owned;
    let head = hooks.core.as_ref().unwrap();
    assert_eq!(
        early,
        (head.target_head.as_ref().unwrap().bytes().unwrap()
            + head.draft_head.as_ref().unwrap().bytes().unwrap()) as u64
    );
}

#[test]
fn hook_facts_reject_empty_physical_owners_and_invalid_capacity() {
    let mut hooks = layouts();
    hooks.device = 2;
    assert!(hooks.components().is_err());
    hooks.device = 0;
    hooks.maximum_pages = 0;
    assert!(hooks.components().is_err());
    hooks.maximum_pages = 16;
    let shared = hooks.shared_workspace.clone().unwrap();
    for defect in 0..6 {
        let mut invalid = shared.clone();
        match defect {
            0 => invalid.bytes = 0,
            1 => invalid.compact = true,
            2 => invalid.unit_tables = true,
            3 => invalid.top_k = 8,
            4 => invalid.rows = 128,
            5 => invalid.ranges[1].0 = 0,
            _ => unreachable!(),
        }
        hooks.shared_workspace = Some(invalid);
        assert!(hooks.components().is_err(), "shared defect {defect}");
    }
    hooks.shared_workspace = None;
    hooks.native.as_mut().unwrap()[0].linear.table = 1;
    assert!(hooks.components().is_err());
    hooks.native = None;
    hooks
        .core
        .as_mut()
        .unwrap()
        .decode
        .as_mut()
        .unwrap()
        .extents
        .remove("partial");
    assert!(hooks.components().is_err());
}
