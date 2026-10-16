//! Header-only inventory for the same pre-allocation Qwen dual request as runtime.
use super::*;
use anyhow::{Context, Result};
use crate::placement::{families::qwen4, Baseline, GpuBudget, Onboard, Placement};
use crate::families::qwen4::{resident::checkpoint_resident_bytes, Qwen4Config};
use crate::serving_capacity::qwen_graphs::QwenAdmissionInputs;

pub(super) fn layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &LayoutOptions, rows: u64, concurrency: u64,
    context: u64, mtp: bool, manifest: Option<&serde_json::Value>) -> MemoryLayout {
    let checkpoint = input.checkpoint();
    let mut devices = options.rtx_bytes.iter().enumerate().map(|(rank, &capacity)| {
        let arch = crate::placement::ArchContext::for_device("sm_120", capacity);
        let modules = manifest.map(|m| crate::placement::ProgramSet::from_manifest(m, &["qwen4"]).module_bytes())
            .unwrap_or(34 * MIB);
        DeviceLayout { kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: capacity,
            items: vec![Item::new(Category::Runtime, "context+modules", "SM120",
                arch.context_bytes + arch.cublas_bytes + arch.peer_context_bytes + modules, Basis::Formula)], kv_tokens: 0 }
    }).collect::<Vec<_>>();
    let mut notes = vec![format!("resolved context {context} tokens"),
        "Qwen whole-layer attention ranges; all routed experts native TP2; head and MTP on rtx0".into()];
    if report.requested_attention.is_some_and(|mode| mode != crate::placement::AttentionPlacement::Layers) {
        report.placement_supported = false;
        notes.push("Qwen dual supports whole-layer attention only, not head or context shards".into());
        return MemoryLayout { devices, pool_tokens: 0, waste: Vec::new(), notes };
    }
    report.attention_placement = crate::placement::AttentionPlacement::Layers;
    if report.encoder.is_none() {
        let mut media = devices.clone();
        let mut reserved = options.clone();
        reserved.pool_tokens = Some(options.pool_tokens.filter(|&n| n > 0).unwrap_or_else(||
            crate::placement::PoolPolicy::resolve(&options.rtx_bytes, context, None, 256, true).target));
        match apply(input, &reserved, &mut media, rows, concurrency, context, mtp, manifest) {
            Ok(_) => {
                resolve_encoder(report, &qwen_towers(report), &mut media, &mut [], &[0; 2], options, &mut notes);
                for (device, admitted) in devices.iter_mut().zip(media) {
                    device.items.extend(admitted.items.into_iter().filter(|item|
                        item.group.starts_with("vision") || item.group.starts_with("audio")));
                }
            }
            Err(error) => { report.placement_supported = false; notes.push(format!("Qwen media admission: {error:#}")); }
        }
    }
    let pool_tokens = match apply(input, options, &mut devices, rows, concurrency, context, mtp, manifest) {
        Ok(placement) => { report.solver_placement = Some(placement.clone()); notes.push(format!("placement: {}", placement.summary())); placement.pool_tokens },
        Err(error) => {
            report.placement_supported = false;
            report.hints.push(super::super::Hint { what: format!("Qwen dual admission: {error:#}"),
                how: "Supply complete checkpoint headers and exact unequal TP2 plus MTP package manifests.".into() });
            notes.push(format!("Qwen dual admission: {error:#}")); 0
        }
    };
    MemoryLayout { devices, pool_tokens, waste: Vec::new(), notes }
}

#[allow(clippy::too_many_arguments)]
fn apply(input: &super::super::PlanInput, options: &LayoutOptions, devices: &mut [DeviceLayout],
    rows: u64, concurrency: u64, context: u64, mtp: bool, manifest: Option<&serde_json::Value>) -> Result<Placement> {
    let checkpoint = input.checkpoint();
    let cfg = Qwen4Config::from_hf(&checkpoint.config)?;
    let mtp = mtp && cfg.mtp_layers > 0;
    let selected = options.qwen_representation;
    let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(64,
        mtp.then_some(options.native_mtp_layers), options.qwen_attention_plan_rows)?;
    let sms = [0, 1].map(|rank| options.physical_sms.unwrap_or_else(||
        crate::placement::ArchContext::for_device("sm_120", options.rtx_bytes[rank]).sms));
    let selected_manifests = manifest.map(|m| qwen4::dual_program_manifests(m, sms, plan)).transpose()?;
    let owner_manifests = [0, 1].map(|rank| selected_manifests.as_ref().map(|m| &m[rank]));
    let manifest = owner_manifests[0];
    let resident = checkpoint_resident_bytes(checkpoint, &cfg, cfg.layers, mtp, selected)?;
    let catalog = input.expert_catalog()?;
    let costs = qwen4::dual_expert_costs(&catalog, cfg.layers)?;
    let lib = crate::placement::inventory::image_lib(options.workspace_manifest.as_deref()).join("libcuteafd_native.so");
    let packages = qwen4::dual_packages(&catalog, &lib)?;
    let workspace = qwen4::dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), rows.max(64))?;
    let host = options.host_embedding;
    let mut fixed = qwen4::dual_owner0_weights(&resident, host);
    fixed.extend(qwen4::dual_projection_load_staging(&cfg, cfg.layers, mtp, selected));
    fixed.extend(qwen4::QwenTp2Rows::new(cfg.hidden as u64, cfg.topk as u64, rows.max(64),
        catalog.exl3().is_some())?.demands()?);
    fixed.extend(qwen4::dual_peer_diagnostics(
        std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").is_ok_and(|v| v == "1")));
    if mtp { fixed.extend(qwen4::dual_draft_demands(&catalog, &lib, rows)?); }
    let extra = qwen4::dual_ple_demands(&cfg, cfg.layers, &resident)?;
    let geometry = crate::serving_capacity::qwen_cache_geometry(&cfg, cfg.layers, mtp, options.qwen_kv)?;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_for(concurrency,
        options.mimo_prefix_entries, geometry.ranks[0].retained_mark_bytes, options.mimo_prefix_mark_bytes));
    let ple = checkpoint.tensors.iter().find(|t| t.meta.name.contains("ngram_embedding.shard_"))
        .map(|t| (resident.ple_row_bytes, t.meta.dtype == cuteafd_core::DType::F8E4M3));
    let headroom = options.headroom_bytes.max(3 * GIB);
    let inputs = qwen4::QwenDualInputs {
        owner_manifests,
        admission: QwenAdmissionInputs { cfg: &cfg, layers: cfg.layers, mtp, kv_format: options.qwen_kv,
            manifest, prefill_rows: rows, slots: options.state_slots.unwrap_or(concurrency),
            mark_bytes: marks * geometry.ranks[0].retained_mark_bytes,
            full_prefill_logits: if options.full_prefill_logits { full_prefill_logits_bytes("qwen4", rows, cfg.vocab_size as u64) } else { 0 },
            ple, future_expert_bytes: 0, headroom },
        gpus: [0, 1].map(|rank| GpuBudget { capacity_bytes: options.rtx_bytes[rank], headroom_bytes: headroom,
            baseline: Baseline::Planned { context_bytes: 0, loaded_bytes: devices[rank].used_bytes() } }),
        layer_weights: &resident.layer_bytes, layer_experts: &costs, layer_extra: &extra, fixed,
        tp2_workspace: workspace, pending_code_bytes: [0; 2], mark_slots: marks, max_context: context,
        requested_pool: options.pool_tokens.filter(|&n| n > 0), spark_ranks: 0,
        onboard: options.onboard.unwrap_or(Onboard::Auto), startup_graph_modes: Some((concurrency.min(16) as usize, true)),
    };
    let placement = qwen4::dual_placement(&inputs).context("solving Qwen owner ranges")?;
    for (rank, device) in devices.iter_mut().enumerate() {
        device.items.extend(placement.items[rank].iter().cloned());
        device.kv_tokens = placement.pool_tokens;
    }
    Ok(placement)
}
