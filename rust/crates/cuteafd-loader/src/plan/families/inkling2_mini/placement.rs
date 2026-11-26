//! Inkling 2 Mini (`inkling2_mini`) `cuteafd plan --layout` over the shared contract planner.
//!
//! Calls the exact same [`admission::gather`] + [`MiniInputs::admit`] solver as serving
//! startup so plan == runtime admission across every named owner.
use super::Model;
use crate::families::inkling2::draft::Selection;
use crate::families::inkling2_mini::admission::{gather, Coordinator, FactsRequest};
use crate::placement::families::inkling2_mini::{self as shared, LocalPackage};
use crate::placement::planner::{Operation, PrefillShape};
use crate::placement::{ArchContext, AttentionPlacement, ResidentWeights};
use crate::plan::{ExpertPlacement, PlanReport};
use cuteafd_core::memory_layout::{DeviceKind, DeviceLayout, MemoryLayout};

fn empty(notes: Vec<String>) -> MemoryLayout {
    MemoryLayout {
        devices: Vec::new(),
        pool_tokens: 0,
        waste: Vec::new(),
        notes,
    }
}

pub(super) fn layout(
    model: &Model,
    input: &crate::plan::PlanInput,
    report: &mut PlanReport,
    opts: &mut crate::plan::layout::LayoutOptions,
    requested: ExpertPlacement,
) -> MemoryLayout {
    let cfg = &model.cfg;
    let gpus = opts.rtx_bytes.len();
    let ranks = requested.spark_ranks();
    report.bytes_by_owner.clear();

    let mut notes = vec![
        "Inkling 2 Mini shared contract: TP1 on 1x RTX, 0 Sparks, 48 distinct KV + short-conv cache slots (8 global paged + 40 SWA-1024)".to_string(),
    ];
    if gpus != 1 {
        report.refuse_placement(
            format!("Inkling 2 Mini is TP1-only (1 coordinator GPU), not {gpus}"),
            "Use --rtx 1 --sparks 0.",
        );
        return empty(notes);
    }
    if ranks != 0 {
        report.refuse_placement(
            format!("Inkling 2 Mini runs locally on 1x RTX with 0 Sparks, not {ranks} Spark ranks"),
            "Use --rtx 1 --sparks 0.",
        );
        return empty(notes);
    }
    if opts.host_embedding {
        report.refuse_placement(
            "Inkling 2 Mini keeps embed on GPU; --embedding-placement host is unsupported".into(),
            "Use --embedding-placement gpu.",
        );
        return empty(notes);
    }
    report.attention_placement = AttentionPlacement::Heads;

    let limits = crate::plan::Family::prefill_limits(&super::INKLING2_MINI);
    let prefill = match crate::plan::layout::resolve_prefill(input, report, opts, 0, limits) {
        Ok(p) => p,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    report.prefill_reserve_rows = Some(prefill.span_rows);
    let shape = PrefillShape {
        rows: prefill.unit_rows,
        waves: prefill.waves,
    };

    let manifest_path = opts.workspace_manifest.clone().or_else(|| {
        let path = std::path::PathBuf::from("/opt/cuteafd/share/PROGRAMS.json");
        path.is_file().then_some(path)
    });
    let manifest = match manifest_path.as_deref().map(|path| input.manifest(path)).transpose() {
        Ok(m) => m,
        Err(error) => {
            report.refuse_placement(
                format!("Inkling 2 Mini program manifest refused: {error:#}"),
                "Supply a readable PROGRAMS.json.",
            );
            return empty(notes);
        }
    };
    let host = match crate::plan::layout::planner_host_memory(
        opts.host_memory,
        crate::plan::layout::HostMemorySample::Raw,
    ) {
        Ok(memory) => memory,
        Err(error) => {
            report.refuse_placement(
                format!("Inkling 2 Mini host inventory refused: {error:#}"),
                "Provide a host memory sample.",
            );
            return empty(notes);
        }
    };
    let card = crate::placement::CoordinatorCard::planned(opts.rtx_bytes[0], opts.physical_sms);
    let sequences = opts.resolved_concurrency("inkling2_mini");
    let catalog = match input.expert_catalog() {
        Ok(c) => c,
        Err(error) => {
            report.refuse_placement(
                format!("Inkling 2 Mini shared expert catalog refused: {error:#}"),
                "Provide complete NVFP4/MXFP8 expert headers.",
            );
            return empty(notes);
        }
    };
    let drafter = (opts.inkling_dspark && model.has_mtp).then_some(Selection {
        kv_cache: opts.draft_kv_cache,
        sms: opts
            .physical_sms
            .unwrap_or(ArchContext::for_device("sm_120", card.physical_bytes).sms) as usize,
    });

    let request = FactsRequest {
        vision: report.vision,
        vision_replicas: opts.vision_replicas,
        media_cache_bytes: opts.media_cache_bytes,
        checkpoint: input.checkpoint(),
        catalog,
        cfg,
        kv_format: opts.inkling_kv,
        drafter,
        dflash: opts.inkling_mini_dflash.as_deref().map(|snapshot| (snapshot, Selection {
            kv_cache: opts.draft_kv_cache,
            sms: opts.physical_sms.unwrap_or(ArchContext::for_device("sm_120", card.physical_bytes).sms) as usize,
        })),
        manifest_path: manifest_path.as_deref(),
        manifest,
        coordinator: Coordinator::Planned { card },
        prefill: shape,
        context_tokens: opts.context_tokens,
        pool_tokens: opts.pool_tokens,
        target_pool_tokens: opts.target_pool_tokens,
        sequences,
        host,
        host_cache_bytes: opts.host_cache_bytes,
        prefix_entries: opts.prefix_entries("inkling2_mini"),
        prefix_mark_budget: opts.mimo_prefix_mark_bytes,
        head: opts.bf16_target_head,
        expert_usage: opts.expert_usage,
        resident_weight_limit: ResidentWeights(opts.weight_budget_bytes),
    };

    let inputs = match gather(&request) {
        Ok(i) => i,
        Err(refused) => {
            report.refuse_placement(refused.what, refused.how);
            return empty(notes);
        }
    };
    if let LocalPackage::Missing(reason) = &inputs.local {
        notes.push(format!("Inkling 2 Mini RTX expert package unavailable: {reason}"));
    }

    let solve_request = shared::request(
        inputs.context,
        opts.pool_tokens,
        inputs.routed.len(),
        shape,
        Operation::Serving,
        inputs.resident_weight_limit,
    );
    let admitted = match inputs.admit(solve_request) {
        Ok(a) => a,
        Err(error) => {
            report.refuse_placement(
                format!("Inkling 2 Mini TP1 admission refused: {error}"),
                "Check --gpu-memory-cap, --kv-min-tokens, --kv-cache fp8, or installed fp8-inkling2_mini-* packages.",
            );
            return empty(notes);
        }
    };

    report.encoder = inputs.vision.as_ref().and_then(|v| v.admitted(&admitted.solved));
    report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&admitted.solved));
    report.host_budget_bytes = inputs.host_prefix.quota_bytes;
    let placed = admitted.placement;

    report.bytes_by_owner.insert(
        "coordinator target arena".into(),
        inputs.target.execution_arena().map_or(0, |a| a.bytes() as u64),
    );
    report.bytes_by_owner.insert("routed expert weights".into(), model.routed_bytes);
    notes.push(format!(
        "rtx0: target arena {} B ({} views); routed experts 24/24 resident (NVFP4 0..20 + MXFP8 20..24); \
         global KV pool ({} global slots {:?}, {}) {} tokens; 40 SWA-1024 slots + 48x3 conv lags; unit {} x {} waves; C{sequences}",
        inputs.target.execution_arena().map_or(0, |a| a.bytes()),
        inputs.target.execution_arena().map_or(0, |a| a.views().len()),
        cfg.global_cache_slots().len(),
        cfg.global_cache_slots(),
        opts.inkling_kv.name(),
        placed.pool_tokens,
        shape.rows,
        shape.waves,
    ));

    let mut device = DeviceLayout {
        kind: DeviceKind::Rtx,
        index: 0,
        capacity_bytes: card.physical_bytes.min(card.logical_bytes),
        worker_budget_bytes: 0,
        items: placed.items[0].clone(),
        kv_tokens: placed.pool_tokens,
    };
    device.items.extend(crate::placement::reserve::gpu_item("inkling2_mini"));
    report.solver_placement = Some(placed);
    MemoryLayout {
        devices: vec![device],
        pool_tokens: report.solver_placement.as_ref().map_or(0, |p| p.pool_tokens),
        waste: Vec::new(),
        notes,
    }
}
