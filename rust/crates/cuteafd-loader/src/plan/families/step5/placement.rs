//! Offline counterpart of the target's exact live allocation recipe.
use super::{
    admission::{self, MarkRule, RuntimeInputs},
    Model,
};
use crate::{
    placement::*,
    plan::{layout::LayoutOptions, ExpertPlacement, MediaMode, PlanInput, PlanReport},
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

#[derive(Debug, thiserror::Error)]
#[error("Step-5 layout is unqualified: {0}; use one or two RTX PRO 6000 for EXL3, two for FP8 or one for NVFP4, with a covered Spark width and enough memory")]
struct Unqualified(&'static str);

pub(super) fn layout(
    model: &Model,
    input: &PlanInput,
    report: &mut PlanReport,
    opts: &mut LayoutOptions,
    requested: ExpertPlacement,
) -> Result<MemoryLayout> {
    ensure!(
        model.complete,
        "Step-5 admission requires complete safetensors headers; missing safetensors headers"
    );
    let nvfp4 = model.formats.iter().any(|f| f == "nvfp4-g16-w4a4");
    let exl3 = model.is_exl3();
    // Coordinator GPUs are a placement qualification: EXL3 runs on one or two
    // RTX; the official releases keep their qualified GPU counts. Spark worlds
    // come from the shared width-class package ranges for every format, and a
    // frozen format admits only the worlds it shipped on.
    let coordinators = if exl3 { opts.rtx_bytes.len() } else if nvfp4 { 1 } else { 2 };
    let spark_ranks = requested.spark_ranks();
    let format = if exl3 { "exl3" } else if nvfp4 { "nvfp4" } else { "fp8" };
    let package = if exl3 { format!("step5:{}", model.exl3_package().context("Step-5 EXL3 expert tiers")?) }
        else { format!("step5:{format}") };
    let worlds = cuteafd_core::tp_units::frozen_admitted_worlds("step5", format)
        .unwrap_or_else(|| cuteafd_core::tp_units::spark_package_worlds("step5", format));
    if let Some(refusal) = crate::plan::experts::frozen_refusal(&package, spark_ranks) {
        anyhow::bail!("Step-5 layout is unqualified: {refusal}");
    }
    crate::plan::experts::spark_recipe(&package, 1536, spark_ranks)?;
    ensure!(
        exl3 || nvfp4 || model.formats.iter().any(|f| f == "fp8-block128"),
        Unqualified("BF16 target format")
    );
    ensure!(
        matches!(coordinators, 1 | 2) && opts.rtx_bytes.len() == coordinators
            && worlds.contains(&spark_ranks)
            && opts.rtx_bytes.iter().all(|&bytes| bytes > 32 << 30),
        Unqualified("coordinator/Spark count or 5090")
    );
    ensure!(
        opts.head_split
            && report
                .requested_attention
                .is_none_or(|a| a == AttentionPlacement::Heads),
        Unqualified("two-GPU execution requires head-split SWA with whole-owned CSA; all-whole SWA has no routed reply reuse proof")
    );
    ensure!(
        !opts.host_embedding
            && opts.drafter_bytes == 0
            && report.vision == MediaMode::Off
            && report.audio == MediaMode::Off,
        Unqualified("host embedding, draft or media owners")
    );
    // Every format onboards through the shared RTX expert path (pool first by
    // default); only all-local routed experts are unqualified.
    ensure!(opts.local_expert_layers.is_none_or(|n| n == 0), Unqualified("coordinator-local routed experts"));
    let shape = crate::plan::layout::resolve_prefill(report, opts, spark_ranks,
        crate::plan::Family::prefill_limits(&super::STEP5))?;
    let checked_rows = usize::try_from(shape.unit_rows)?;
    // Hop lanes, return slots and Spark owner units.
    let waves = usize::try_from(shape.waves)?;

    let sequences = usize::try_from(if opts.concurrency == 0 {
        16
    } else {
        opts.concurrency
    })?;
    ensure!(
        opts.state_slots.is_none_or(|n| n == sequences as u64),
        Unqualified("sequence-slot mismatch")
    );
    ensure!(
        opts.prefix_slots.is_none(),
        Unqualified("explicit prefix slots; use mark entries/budget")
    );
    let context = if opts.context_tokens == 0 {
        model.cfg.max_context
    } else {
        opts.context_tokens
    };
    let path = opts
        .workspace_manifest
        .as_deref()
        .unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path)
            .context("Step-5 native admission requires its matching PROGRAMS.json")?,
    )?;
    ensure!(
        manifest["capacities"]["max_context"] == model.cfg.max_context,
        "Step-5 exports require the checkpoint's 1M compiled extent"
    );
    let prefixes: &[&str] = if coordinators == 2 { &["step5", "step52"] } else { &["step5"] };
    let programs = ProgramSet::from_manifest(&manifest, prefixes);
    let entries = manifest["programs"]
        .as_array()
        .context("Step-5 program manifest lacks entries")?;
    for prefix in prefixes {
        for op in [
            "hidden_norm",
            "final_norm",
            "residual_bf16",
            "residual_f32",
            "swa_attention",
            "csa_attention",
            "region_compress",
            "region_scores",
            "region_selection",
            "region_commit",
            "sliding_commit",
            "attn_q",
            "attn_kv",
            "attn_gate",
            "attn_out",
            "router",
            "dense_gate_up",
            "dense_down",
            "shared_gate_up",
            "shared_down",
        ] {
            let name = format!("{prefix}_{op}");
            let entry = entries
                .iter()
                .find(|p| p["name"] == name)
                .with_context(|| format!("Step-5 missing native export {name}"))?;
            ensure!(
                entry["capacity_rows"]
                    .as_u64()
                    .is_some_and(|n| n >= checked_rows as u64),
                "Step-5 export {name} cannot hold live rows {checked_rows}"
            );
        }
    }
    let catalog = input.expert_catalog()?;
    let package = match catalog.exl3() {
        Some(manifest) => format!("exl3-step5-k{}",
            manifest.decoder_tiers().iter().map(usize::to_string).collect::<String>()),
        None => catalog.fp8().context("Step-5 requires native FP8, NVFP4 or EXL3 expert headers")?
            .format().package_name("step5"),
    };
    let mut code = Vec::new();
    let inventory = Inventory {
        gpus: opts
            .rtx_bytes
            .iter()
            .enumerate()
            .map(|(rank, &capacity_bytes)| {
                let arch = ArchContext::coordinator(capacity_bytes, None);
                let allowance = inventory::loaded_code_allowance(
                    "step5",
                    &package,
                    coordinators == 2,
                    rank as u8,
                    &arch,
                    Some(&programs),
                );
                code.push(allowance.bytes);
                GpuBudget {
                    capacity_bytes,
                    headroom_bytes: opts.headroom_bytes,
                    baseline: Baseline::Planned {
                        context_bytes: arch.context_bytes,
                        loaded_bytes: 0,
                    },
                }
            })
            .collect(),
        spark_ranks,
        peer_access: coordinators == 2,
    };
    let inputs = RuntimeInputs {
                kv_format: opts.step5_kv,
        inventory,
        rows: usize::try_from(shape.span_rows)?,
        waves,
        unit: checked_rows,
        full_prefill_logits: opts.full_prefill_logits,
        sequences,
        context,
        requested_pool: opts.pool_tokens,
        marks: MarkRule {
            lanes: sequences,
            entries: usize::try_from(opts.mimo_prefix_entries)?,
            budget_bytes: opts.mimo_prefix_mark_bytes,
            host: opts.host_cache_bytes != Some(0),
        },
        peer_diagnostics: false,
        pending_code: code,
        startup: vec![],
        onboard: super::admission::OnboardInputs::new(input.expert_catalog()?, opts.workspace_manifest.as_deref(),
            checked_rows, coordinators, waves, opts.onboard.unwrap_or(Onboard::Auto))?,
    };
    let admitted = admission::forecast_mtp(input.checkpoint(), &inputs, opts.native_mtp_layers)?;
    report.attention_placement = admitted.placement.attention_placement;
    report.bytes_by_owner.clear();
    report.bytes_by_owner.insert(
        "coordinator text weights".into(),
        admitted
            .fixed
            .iter()
            .filter(|d| matches!(d.category, Category::Weights | Category::Embedding))
            .map(|d| d.bytes)
            .sum(),
    );
    let mut memory = MemoryLayout { devices: Vec::new(), pool_tokens: admitted.placement.pool_tokens,
        waste: vec![], notes: vec![format!("WIP eager Step-5: exact native allocation owners; zero graphs; {} native MTP stages; no media", opts.native_mtp_layers),
            "Offline CUDA/code baseline is estimated; serving must re-admit against pre-weight live samples and native export ABIs".into(),
            format!("{}; {sequences} sequences, request context {}, compiled extent {}",
                if coordinators == 2 { "SWA head split, complete-cost contiguous CSA owners" } else { "Single-device SWA and CSA ownership" },
                admitted.cache.context, model.cfg.max_context)] };
    let csa_owners = (0..coordinators).map(|rank| admitted.placement.layers.iter().enumerate()
        .filter(|(layer, item)| model.cfg.full[*layer]
            && matches!(item.mode, LayerMode::Whole { gpu, .. } if usize::from(gpu) == rank)).count())
        .collect::<Vec<_>>();
    memory.notes.push(format!("Step-5 admitted CSA owners: {csa_owners:?}"));
    for rank in 0..coordinators {
        let mut items = admitted.placement.items[rank].clone();
        items.push(Item::new(
            Category::Runtime,
            "CUDA pre-weight baseline",
            "sm_120",
            ArchContext::coordinator(opts.rtx_bytes[rank], None).context_bytes,
            Basis::Calibrated,
        ));
        items.push(
            Item::new(
                Category::Reserved,
                "coordinator headroom",
                "",
                opts.headroom_bytes,
                Basis::Formula,
            )
            .with_lifetime(cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve),
        );
        memory.devices.push(DeviceLayout {
            kind: DeviceKind::Rtx,
            index: rank as u32,
            capacity_bytes: opts.rtx_bytes[rank],
            items,
            kv_tokens: memory.pool_tokens,
        });
    }
    // The solver's homes are the plan: onboarded layers leave the Spark working set.
    let rtx_layers = admitted.placement.layers.iter().enumerate()
        .filter(|(_, l)| matches!(l.experts, ExpertHome::RtxWhole { .. } | ExpertHome::RtxTp2))
        .map(|(layer, _)| layer).collect::<Vec<_>>();
    let working = SparkWorkingSet {
        spark_layers: (3..91).filter(|layer| !rtx_layers.contains(layer)).collect(),
        rtx_layers,
        spark_ranks,
        prefill_lanes: 1,
    };
    if let Some(reason) = super::admission::onboarding_unavailable(input.expert_catalog()?, coordinators) {
        memory.notes.push(reason.into());
    }
    if !working.rtx_layers.is_empty() {
        memory.notes.push(format!("Step-5 RTX onboarding: {} routed layers {} {:?}", working.rtx_layers.len(),
            if coordinators == 2 { "split TP2 over both RTX" } else { "whole on the RTX" }, working.rtx_layers));
    }
    let capacity = u32::try_from(shape.unit_rows)?;
    let endpoints = (0..admitted.ownership.routed_endpoints()?)
        .map(|_| cuteafd_core::SparkEndpoint { ring_depth: None,
            capacity,
            ingress: if nvfp4 && !exl3 { cuteafd_core::SparkIngress::Bf16 } else { cuteafd_core::SparkIngress::Native }
        })
        .collect::<Vec<_>>();
    let rings = cuteafd_core::SparkRings {
        max_frame_bytes: 64 << 20,
        depth: 8,
        slot_bytes: 8 << 20,
        alignment: 4096,
    };
    let recipe = crate::plan::experts::checkpoint_spark_recipe(
        input.checkpoint(),
        spark_ranks,
        &working.spark_layers,
    )?;
    for rank in 0..spark_ranks {
        let layout = if exl3 { SparkResidentLayout::Exl3 { world: spark_ranks, rank } }
            else { SparkResidentLayout::Fp8 { world: spark_ranks, rank, slicing: recipe.slicing } };
        let assignment = SparkAssignment::from_catalog(&catalog, &working, rank, layout, &endpoints, rings)?;
        ensure!(
            assignment.resident_bytes
                <= opts
                    .spark_allocation_budget_bytes
                    .unwrap_or(opts.spark_bytes),
            "Step-5 {package} TP{spark_ranks} Spark rank {rank} weights exceed allocation budget"
        );
        *report
            .bytes_by_owner
            .entry("routed expert weights".into())
            .or_default() += assignment.resident_bytes;
        let io = assignment.io;
        let mut device = DeviceLayout {
            kind: DeviceKind::Spark,
            index: rank as u32,
            capacity_bytes: opts.spark_bytes,
            kv_tokens: 0,
            items: vec![
                Item::new(
                    Category::Experts,
                    if exl3 { "EXL3 K4 experts and K6 tails" } else if nvfp4 { "native NVFP4 TP4 and BF16 tails" }
                        else { "native FP8 TP6 and BF16 tails" },
                    if exl3 { "MCG W4A16" } else if nvfp4 { "W4A4" } else { "W8A16" },
                    assignment.resident_bytes,
                    Basis::Exact,
                ),
                cuteafd_core::memory_layout::spark_runtime_residual(),
            ],
        };
        for (group, bytes) in [
            ("expert intake", io.intake),
            ("host expert exchange", io.host_exchange),
            ("expert row indices", io.row_indices),
            ("endpoint response prefixes", io.endpoint_staging),
            ("rdma rings", io.mapped_rings),
        ] {
            device.items.push(Item::new(
                Category::Transport,
                group,
                "BF16 ingress",
                bytes,
                Basis::Formula,
            ));
        }
        // Unlike the legacy family wave allowances, Step-5 has no first-slot scratch reserve.
        let backend = crate::plan::spark_executions::SparkBackend::for_catalog(input.expert_catalog()?)?;
        let executions = crate::plan::spark_executions::admit_family("step5", &backend.exports(), None,
            shape.unit_rows, shape.waves)?;
        crate::plan::layout::spark_slots::charge_executions(&mut device, input, &working, "step5", &executions)?;
        memory.devices.push(device);
    }
    report.solver_placement = Some(admitted.placement);
    Ok(memory)
}
