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
#[error("Step-5 layout is unqualified: {0}; use WIP FP8 on 2 RTX PRO 6000 + 6 Sparks or NVFP4 on 1 RTX PRO 6000 + 4 Sparks")]
struct Unqualified(&'static str);

/// Prefill units (PLAN "Prefill contract"): 2048 rows with two in flight by
/// default, stepping down 2048 -> 1024 -> 512 when the larger unit would cost
/// pool capacity. Explicit --prefill-rows / --prefill-waves are never
/// changed. Two units in flight everywhere: NVFP4 min's 8K prefill gained
/// 1.47x at 512 x 2 for a smaller pool (measured 2026-10-23).
const DEFAULT_ROWS: [usize; 3] = [2048, 1024, 512];
const DEFAULT_WAVES: usize = 2;

fn default_rows(inputs: &mut RuntimeInputs, waves: usize,
    mut admit: impl FnMut(&RuntimeInputs) -> Result<admission::Admission>) -> Result<(usize, admission::Admission)> {
    // The smallest unit's pool is the reference: a larger unit is taken only
    // when it admits the same pool (its extra rows cost no KV capacity).
    let smallest = *DEFAULT_ROWS.last().expect("default prefill units");
    inputs.waves = waves;
    inputs.rows = smallest * waves;
    let reference = admit(inputs)?;
    for unit in DEFAULT_ROWS {
        if unit == smallest { break; }
        inputs.rows = unit * waves;
        if inputs.rows > 4096 { continue; }
        if let Ok(admitted) = admit(inputs) {
            if admitted.placement.pool_tokens >= reference.placement.pool_tokens {
                return Ok((unit, admitted));
            }
        }
    }
    inputs.rows = smallest * waves;
    Ok((smallest, reference))
}

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
    let (coordinators, spark_ranks) = if nvfp4 { (1, 4) } else { (2, 6) };
    report.min_spark_ranks = Some(spark_ranks);
    ensure!(
        nvfp4 || model.formats.iter().any(|f| f == "fp8-block128"),
        Unqualified("BF16 target format")
    );
    ensure!(
        opts.rtx_bytes.len() == coordinators
            && requested.spark_ranks() == spark_ranks
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
    ensure!(
        opts.onboard
            .is_none_or(|onboard| matches!(onboard, Onboard::Layers(0) | Onboard::Auto))
            && opts.local_expert_layers.is_none_or(|n| n == 0),
        Unqualified("coordinator-local routed experts")
    );
    let explicit_rows = usize::try_from(opts.prefill_rows)?;
    let mut waves = if opts.prefill_lanes == 0 { DEFAULT_WAVES } else { usize::try_from(opts.prefill_lanes)? };
    ensure!((1..=4).contains(&waves), "Step-5 prefill waves must be 1..=4");
    let checked_rows = if explicit_rows == 0 { DEFAULT_ROWS[0] } else { explicit_rows };
    ensure!(checked_rows * waves <= 4096, "Step-5 prefill span {checked_rows} x {waves} exceeds 4096 rows");

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
    let format = catalog.fp8().context("Step-5 requires native expert headers")?.format();
    let package = format.package_name("step5");
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
    let mut inputs = RuntimeInputs {
                kv_format: opts.step5_kv,
        inventory,
        rows: checked_rows * waves,
        waves,
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
    };
    let admitted = if explicit_rows != 0 {
        admission::forecast_mtp(input.checkpoint(), &inputs, opts.native_mtp_layers)?
    } else {
        // Default waves step down to one when the in-flight span does not admit
        // (e.g. device-only prefix retention at full context); explicit waves never do.
        let mut selected = default_rows(&mut inputs, waves, |inputs|
            admission::forecast_mtp(input.checkpoint(), inputs, opts.native_mtp_layers));
        if selected.is_err() && opts.prefill_lanes == 0 && waves > 1 {
            waves = 1;
            selected = default_rows(&mut inputs, waves, |inputs|
                admission::forecast_mtp(input.checkpoint(), inputs, opts.native_mtp_layers));
        }
        let (unit, admitted) = selected?;
        report.prefill_selection = Some(format!("Step-5 automatic prefill selected {unit} rows x {waves} waves"));
        admitted
    };
    let rows = inputs.rows;
    report.prefill_span_rows = Some(rows as u64);
    report.prefill_transports = Some(1);
    opts.prefill_unit_rows = Some(inputs.unit_rows() as u64);
    opts.prefill_waves = waves as u64;
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
    let working = SparkWorkingSet {
        rtx_layers: vec![],
        spark_layers: (3..91).collect(),
        spark_ranks,
        prefill_lanes: 1,
    };
    let endpoints = (0..admitted.ownership.routed_endpoints()?)
        .map(|_| cuteafd_core::SparkEndpoint { ring_depth: None,
            capacity: u32::try_from(rows).unwrap_or(u32::MAX),
            ingress: cuteafd_core::SparkIngress::Bf16, execution_lane: 0 
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
        let assignment = SparkAssignment::from_catalog(
            &catalog,
            &working,
            rank,
            SparkResidentLayout::Fp8 {
                world: spark_ranks,
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
                    if nvfp4 { "native NVFP4 TP4 and BF16 tails" } else { "native FP8 TP6 and BF16 tails" },
                    if nvfp4 { "W4A4" } else { "W8A16" },
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
