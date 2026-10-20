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
#[error("Step-5 layout is unqualified: {0}; use WIP target-only FP8 on 2 RTX PRO 6000 + 6 Sparks")]
struct Unqualified(&'static str);

pub(super) fn layout(
    model: &Model,
    input: &PlanInput,
    report: &mut PlanReport,
    opts: &LayoutOptions,
    requested: ExpertPlacement,
) -> Result<MemoryLayout> {
    ensure!(
        model.complete,
        "Step-5 admission requires complete safetensors headers; missing safetensors headers"
    );
    report.min_spark_ranks = Some(6);
    ensure!(
        model.formats.iter().any(|f| f == "fp8-block128")
            && !model.formats.iter().any(|f| f == "nvfp4-g16-w4a4"),
        Unqualified("NVFP4/BF16 target format")
    );
    ensure!(
        opts.rtx_bytes.len() == 2
            && requested.spark_ranks() == 6
            && opts.rtx_bytes.iter().all(|&bytes| bytes > 32 << 30),
        Unqualified("coordinator/Spark count or 5090")
    );
    ensure!(
        opts.head_split
            && report
                .requested_attention
                .is_none_or(|a| a == AttentionPlacement::Heads),
        Unqualified("context or non-native attention ownership")
    );
    ensure!(
        !opts.host_embedding
            && opts.native_mtp_layers == 0
            && opts.drafter_bytes == 0
            && report.vision == MediaMode::Off
            && report.audio == MediaMode::Off,
        Unqualified("host embedding, draft or media owners")
    );
    ensure!(
        opts.onboard
            .is_none_or(|onboard| matches!(onboard, Onboard::Layers(0) | Onboard::Auto))
            && opts.local_expert_layers.is_none_or(|n| n == 0),
        Unqualified("coordinator-local routed experts")
    );
    let rows = usize::try_from(if opts.prefill_rows == 0 {
        256
    } else {
        opts.prefill_rows
    })?;
    let lanes = usize::try_from(if opts.prefill_lanes == 0 {
        2
    } else {
        opts.prefill_lanes
    })?;
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
    let programs = ProgramSet::from_manifest(&manifest, &["step5", "step52"]);
    let entries = manifest["programs"]
        .as_array()
        .context("Step-5 program manifest lacks entries")?;
    for prefix in ["step5", "step52"] {
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
                    .is_some_and(|n| n >= rows as u64),
                "Step-5 export {name} cannot hold live rows {rows}"
            );
        }
    }
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
                    "step5:fp8",
                    true,
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
        spark_ranks: 6,
        peer_access: true,
    };
    let inputs = RuntimeInputs {
        inventory,
        rows,
        sequences,
        lanes,
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
    };
    let admitted = admission::admit(input.checkpoint(), &inputs)?;
    report.prefill_rows = Some(rows as u64);
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
        waste: vec![], notes: vec!["WIP target-only eager Step-5: exact native allocation owners; zero graphs; no MTP or media".into(),
            "Offline CUDA/code baseline is estimated; serving must re-admit against pre-weight live samples and native export ABIs".into(),
            format!("SWA head split, alternating whole-owned CSA; {lanes} lanes, {sequences} sequences, request context {}, compiled extent {}",
                admitted.cache.context, model.cfg.max_context)] };
    for rank in 0..2 {
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
    let working = SparkWorkingSet {
        rtx_layers: vec![],
        spark_layers: (3..91).collect(),
        spark_ranks: 6,
        prefill_lanes: lanes as u64,
    };
    let endpoints = (0..lanes * admitted.ownership.routed_endpoints()?)
        .map(|_| cuteafd_core::SparkEndpoint {
            capacity: u32::try_from(rows).unwrap_or(u32::MAX),
            ingress: cuteafd_core::SparkIngress::Bf16,
        })
        .collect::<Vec<_>>();
    let rings = cuteafd_core::SparkRings {
        max_frame_bytes: 64 << 20,
        depth: 8,
        slot_bytes: 8 << 20,
        alignment: 4096,
    };
    let catalog = input.expert_catalog()?;
    let recipe = crate::plan::experts::checkpoint_spark_recipe(
        input.checkpoint(),
        6,
        &working.spark_layers,
    )?;
    for rank in 0..6 {
        let assignment = SparkAssignment::from_catalog(
            &catalog,
            &working,
            rank,
            SparkResidentLayout::Fp8 {
                world: 6,
                rank,
                slicing: recipe.slicing,
            },
            &endpoints,
            rings,
        )?;
        ensure!(
            assignment.resident_bytes
                <= opts
                    .spark_allocation_budget_bytes
                    .unwrap_or(opts.spark_bytes),
            "Step-5 FP8 TP6 Spark rank {rank} weights exceed allocation budget"
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
                    "native FP8 TP6 and BF16 tails",
                    "W8A16",
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
        memory.devices.push(device);
    }
    report.solver_placement = Some(admitted.placement);
    Ok(memory)
}
