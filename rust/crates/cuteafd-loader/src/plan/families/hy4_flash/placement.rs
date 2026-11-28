//! Hy4-flash `cuteafd plan --layout` over the shared contract planner (`Hy4Inputs`).
//!
//! Calls the same `families::hy4_flash::admission::gather` + `Hy4Inputs::admit`
//! path that `serve-hy4f` runs at startup so plan and serve produce the exact
//! same KV pool, onboarding count, and owner bytes.
use super::Model;
use crate::families::hy4_flash::admission::{self, Coordinator, FactsRequest};
use crate::families::hy4_flash::Hy4KvCache;
use crate::placement::families::hy4_flash::{self, LocalPackage, HY4_FLASH};
use crate::placement::planner::{Device, Domain, Location, Operation, PrefillShape};
use crate::placement::{Onboard, ResidentWeights};
use crate::plan::{ExpertPlacement, PlanReport};
use cuteafd_core::memory_layout::{DeviceKind, DeviceLayout, MemoryLayout};

const GIB: f64 = (1u64 << 30) as f64;

fn empty(notes: Vec<String>) -> MemoryLayout {
    MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes }
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
        "Hy4-flash shared contract: header-exact packed arenas over named physical owners (#2478)".to_string(),
    ];
    if !matches!(gpus, 1 | 2) {
        report.refuse_placement(
            format!("Hy4-flash needs one or two coordinator GPUs, not {gpus}"),
            "Use --rtx 1 or --rtx 2.",
        );
        return empty(notes);
    }
    if opts.host_embedding {
        report.refuse_placement(
            "Hy4-flash keeps embed_tokens on coordinator GPU; --embedding-placement host is unsupported".into(),
            "Use --embedding-placement gpu.",
        );
        return empty(notes);
    }

    let catalog = match input.expert_catalog() {
        Ok(catalog) => catalog,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4-flash expert catalog refused: {error:#}"),
                "Use an EXL3 K4 MUL1 Hy4-flash checkpoint.",
            );
            return empty(notes);
        }
    };

    let limits = crate::plan::Family::prefill_limits(&super::HY4_FLASH);
    let prefill = match crate::plan::layout::resolve_prefill(input, report, opts, ranks, limits) {
        Ok(prefill) => prefill,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    report.prefill_reserve_rows = Some(prefill.span_rows);
    let shape = PrefillShape { rows: prefill.unit_rows, waves: prefill.waves };

    let attention = match HY4_FLASH.check_attention(report.requested_attention, gpus, gpus == 2 && opts.head_split) {
        Ok(mode) => mode,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4-flash attention placement refused: {error}"),
                "Use ATTENTION_PLACEMENT=auto, context, layers, or heads.",
            );
            return empty(notes);
        }
    };
    report.attention_placement = attention;

    let kv_format = match opts.mla_kv {
        cuteafd_core::MlaCacheFormat::Bf16 => Hy4KvCache::Bf16,
        cuteafd_core::MlaCacheFormat::Fp8 => Hy4KvCache::Fp8,
    };
    let sequences = opts.resolved_concurrency("hy4_flash");
    let manifest_path = opts.workspace_manifest.clone().or_else(|| {
        let path = std::path::PathBuf::from("/opt/cuteafd/share/PROGRAMS.json");
        path.is_file().then_some(path)
    });
    let host = match crate::plan::layout::planner_host_memory(
        opts.host_memory,
        crate::plan::layout::HostMemorySample::Raw,
    ) {
        Ok(host) => host,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4-flash host inventory refused: {error:#}"),
                "Provide a host memory sample.",
            );
            return empty(notes);
        }
    };
    let spark_bytes: Vec<u64> = (0..ranks).map(|rank| opts.spark_capacity(rank)).collect();

    let tp2_dflash = opts.hy4_dfly2 && gpus == 2 && opts.tp2_dflash;
    if opts.tp2_dflash && (!opts.hy4_dfly2 || gpus != 2) {
        report.refuse_placement(
            "Hy4-flash TP2_DFLASH requires two coordinator GPUs and SPECULATOR=dfly2".into(),
            "Use --rtx 2 with --speculator dfly2, or set TP2_DFLASH=off.",
        );
        return empty(notes);
    }
    report.tp2_dflash = Some(tp2_dflash);

    let facts_req = FactsRequest {
        checkpoint: input.checkpoint(),
        catalog,
        cfg,
        model_core: &model.core,
        dfly2: opts.hy4_dfly2,
        tp2_dflash,
        kv_format,
        manifest_path: manifest_path.as_deref(),
        coordinator: Coordinator::Planned { rtx_bytes: opts.rtx_bytes.clone() },
        attention,
        spark_ranks: ranks,
        spark_bytes,
        prefill: shape,
        decode_rows: report.decode_rows,
        context_tokens: opts.context_tokens,
        pool_tokens: opts.pool_tokens,
        target_pool_tokens: opts.target_pool_tokens,
        sequences,
        host,
        host_cache_bytes: opts.host_cache_bytes,
        prefix_entries: opts.prefix_entries("hy4_flash"),
        prefix_mark_budget: opts.mimo_prefix_mark_bytes,
        prefix_slots: opts.prefix_slots,
        replies: opts.spark_reply_formats,
        decode_graphs: opts.decode_graphs,
        expert_usage: opts.expert_usage,
        resident_weight_limit: Some(ResidentWeights(opts.weight_budget_bytes)),
        vision: report.vision,
        vision_replicas: opts.vision_replicas,
        media_cache_bytes: opts.media_cache_bytes,
    };
    let inputs = match admission::gather(&facts_req) {
        Ok(inputs) => inputs,
        Err(refused) => {
            report.refuse_placement(refused.what, refused.how);
            return empty(notes);
        }
    };
    if let LocalPackage::Missing(reason) = &inputs.local {
        notes.push(format!("Hy4-flash RTX expert onboarding unavailable: {reason}"));
    }

    let routed = model.routed();
    let onboard = opts.resolve_onboard(routed.len()).unwrap_or(Onboard::Auto);
    let solve_req = hy4_flash::request(
        inputs.context,
        opts.pool_tokens,
        onboard,
        shape,
        Operation::Serving,
        inputs.resident_weight_limit,
        opts.spark_reply_formats,
    );
    let admitted = match inputs.admit(solve_req) {
        Ok(admitted) => admitted,
        Err(error) => {
            let detail = match &inputs.local {
                LocalPackage::Missing(reason) if ranks == 0 => {
                    format!("Hy4-flash 0-Spark placement requires installed RTX expert package: {reason} ({error})")
                }
                _ => format!("Hy4-flash joint admission refused: {error}"),
            };
            report.refuse_placement(
                detail,
                "Install exl3-hy4f-k45-mul1 beside PROGRAMS.json, use a compact core view, or add Spark ranks.",
            );
            return empty(notes);
        }
    };

    report.encoder = inputs.vision.as_ref().and_then(|vision| vision.admitted(&admitted.solved));
    if let Some(vision) = &inputs.vision {
        notes.push(format!("Hy4-flash bundled ViT: packed arena {:.2} GiB + scratch {:.2} GiB ({} merged rows), owner {:?}",
            vision.inventory.arena.bytes() as f64 / GIB, vision.inventory.scratch as f64 / GIB,
            vision.inventory.max_output_rows, report.encoder.as_ref().map(|e| &e.kind)));
    }
    let host_quota = inputs.host_prefix.quota_bytes;
    report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&admitted.solved));
    report.host_budget_bytes = host_quota;
    report.host_ledger = match admitted.host_ledger(host_quota) {
        Ok(ledger) => Some(ledger),
        Err(error) => {
            report.refuse_placement(format!("Hy4-flash host ledger refused: {error}"), "Check host RAM budget.");
            return empty(notes);
        }
    };
    let placed = admitted.placement;
    let onboarded = placed.onboard_layers;
    report.bytes_by_owner.insert(
        "coordinator target arenas".into(),
        inputs.targets.iter().map(|t| t.arena.bytes() as u64).sum(),
    );
    report.bytes_by_owner.insert("routed expert weights".into(), model.routed_bytes);
    report.bytes_by_owner.insert(
        "mapped-tables/preloaded (coordinator local)".into(),
        inputs.engram_mapped.preloaded_bytes,
    );

    notes.push(format!(
        "Hy4-flash attention {}: pool {} tokens (12 DSA layers paged, 30 SWA-512 fixed per-sequence rings); \
         {} of {} routed layers onboarded ({}); Engram engram_fp8 61.5 GiB ({}); unit {} x {} waves; C{}",
        placed.attention_summary(),
        placed.pool_tokens,
        onboarded,
        routed.len(),
        if gpus == 2 { "TP2 768+768" } else { "whole 1536" },
        inputs.engram_mapped.reason,
        shape.rows,
        shape.waves,
        sequences
    ));

    let mut devices = Vec::new();
    for gpu in 0..gpus {
        let mut device = DeviceLayout {
            kind: DeviceKind::Rtx,
            index: gpu as u32,
            capacity_bytes: opts.rtx_bytes[gpu],
            worker_budget_bytes: 0,
            items: placed.items[gpu].clone(),
            kv_tokens: placed.pool_tokens,
        };
        device.items.extend(crate::placement::reserve::gpu_item("hy4_flash"));
        notes.push(format!(
            "rtx{gpu}: target arena {:.2} GiB ({} views)",
            inputs.targets[gpu].arena.bytes() as f64 / GIB,
            inputs.targets[gpu].arena.views().len()
        ));
        devices.push(device);
    }
    let mut widest = 0u64;
    for rank in 0..ranks {
        let charge = inputs.spark[onboarded][rank];
        widest = widest.max(charge.experts);
        let location = Location { device: Device::Spark(rank as u8), domain: Domain::Managed };
        let items = match admitted.solved.items(location) {
            Ok(items) => items,
            Err(error) => {
                report.refuse_placement(format!("Hy4-flash Spark ledger refused: {error}"), "Check Spark budgets.");
                return empty(notes);
            }
        };
        devices.push(DeviceLayout {
            kind: DeviceKind::Spark,
            index: rank as u32,
            capacity_bytes: opts.spark_capacity(rank),
            worker_budget_bytes: opts.spark_capacity(rank),
            kv_tokens: 0,
            items,
        });
    }
    report.spark_rank_share = if model.routed_bytes == 0 {
        0.0
    } else {
        widest as f64 / model.routed_bytes as f64
    };
    if ranks > 0 && onboarded < routed.len() {
        notes.push(format!(
            "Spark: TP{ranks} slices of routed layers {}..={}; widest rank {:.2} GiB",
            routed[onboarded],
            cfg.layers - 1,
            widest as f64 / GIB
        ));
    }
    report.solver_placement = Some(placed);
    MemoryLayout {
        devices,
        pool_tokens: report.solver_placement.as_ref().map_or(0, |p| p.pool_tokens),
        waste: Vec::new(),
        notes,
    }
}
