//! Header-only inventory for the same pre-allocation Qwen dual request as runtime.
use super::*;
use anyhow::{Context, Result};
use crate::placement::{families::qwen4, Baseline, GpuBudget, Onboard, Placement};
use crate::families::qwen4::{resident::checkpoint_resident_bytes, Qwen4Config};
use crate::serving_capacity::qwen_graphs::QwenAdmissionInputs;

/// Spark ranks at 4096 capacity rows: the observed host footprint and the
/// non-I/O wave residual (P11 replaces both with the worker's measurement).
const SPARK_HOST_BYTES: u64 = 13 * GIB;

/// Qwen plan adapter: a dual 0-Spark layout runs whole-layer TP2 owners
/// (`dual_layout`); every other layout serves on rtx0 (`single_layout`).
pub(crate) fn layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &mut LayoutOptions) -> MemoryLayout {
    let routed = model.spec().moe_layers();
    let all_onboarded = routed > 0
        && options.resolve_onboard(routed) == Some(Onboard::Layers(routed));
    let local = report.placement == ExpertPlacement::Local || all_onboarded;
    let expert_world = if local { 0 } else { report.spark_ranks };
    let prefill = match resolve_prefill(input, report, options, expert_world,
        crate::families::qwen4::prefill_limits(local)) {
        Ok(shape) => shape,
        Err(error) => return unsupported_prefill(report, error),
    };
    let shape = Shape::resolve(input, report, model, options, ShapePolicy::manifest_bounded());
    report.prefill_span_rows = Some(prefill.span_rows);
    if options.rtx_bytes.len() == 2 && options.head_split {
        return dual_layout(report, model, input, options, shape.prefill_rows, shape.concurrency,
            shape.context_tokens, shape.native_layers > 0, shape.manifest.as_deref());
    }
    single_layout(report, model, input, options, shape)
}

/// Exact resident weights on rtx0 (`checkpoint_resident_bytes`).
fn resident(checkpoint: &crate::plan::Checkpoint, mtp: bool,
    representation: crate::families::qwen4::resident::Qwen4Representation) -> Option<Vec<Vec<(String, String, u64)>>> {
    let cfg = Qwen4Config::from_hf(&checkpoint.config).ok()?;
    let mtp = mtp && cfg.mtp_layers > 0;
    let bytes = checkpoint_resident_bytes(checkpoint, &cfg, cfg.layers, mtp, representation).ok()?;
    // The loader allocates its arenas from these same headers; a checkpoint
    // missing any tensor they name cannot serve, so it does not describe the
    // resident layout (refused below, never charged as a partial payload).
    crate::families::qwen4::weight_layout::single_arena_bytes(checkpoint, &cfg, cfg.layers, mtp,
        representation, 0).ok()?;
    Some(vec![vec![
        ("target incl PLE projections".into(), if representation.fp8_projections { "fp8+bf16+f32" } else { "bf16+f32" }.into(), bytes.target_bytes),
        ("embedding".into(), "bf16".into(), bytes.embedding_bytes),
        ("lm_head".into(), if representation.fp8_head { "fp8-row128" } else { "bf16" }.into(), bytes.head_bytes),
        ("speculator".into(), if representation.fp8_projections { "fp8+bf16+f32" } else { "bf16+f32" }.into(), bytes.mtp_bytes),
    ]])
}

fn single_layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &LayoutOptions, mut shape: Shape) -> MemoryLayout {
    let checkpoint = input.checkpoint();
    let gpus = options.rtx_bytes.len().clamp(1, 2);
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: bytes.saturating_sub(crate::placement::reserve::gpu("qwen4")), worker_budget_bytes: 0, items: Vec::new(), kv_tokens: 0 })
        .collect();
    let mut waste = Vec::new();
    let mut notes = vec![format!("resolved context {} tokens (checkpoint {})", shape.context_tokens, shape.checkpoint_context),
        format!("attention placement {}", report.attention_placement)];
    shape.select_card_plans(report, options, &mut notes);
    let mtp = shape.native_layers > 0;
    let routed = model.spec().moe_layers();
    let all_onboarded = routed > 0
        && options.resolve_onboard(routed) == Some(Onboard::Layers(routed));
    let local = report.placement == ExpertPlacement::Local || all_onboarded;
    let exl3 = report.experts.as_ref().is_some_and(|e| e.package.contains("exl3"));
    let package = report.experts.as_ref().map(|e| e.package.as_str());
    let reference = gpus == 1 && (94 * GIB..=96 * GIB).contains(&options.rtx_bytes[0])
        && mtp && options.onboard.is_none_or(|mode| mode == crate::placement::OnboardSetting::Auto) && report.placement == ExpertPlacement::Local && package == Some("qwen4:exl3-k45")
        && model.spec().hidden == 2560 && model.spec().layers.len() == 48
        && shape.prefill_rows == 4096 && shape.concurrency == 8 && matches!(options.prefix_slots, None | Some(18))
        && matches!(shape.context_tokens, 0 | 32768);
    let basis = if reference { Basis::Calibrated } else { Basis::Estimated };
    if reference {
        notes.push("Runtime allowances calibrated at the natural-minimum reference layout after an 8K prefill and C4 decode; other layouts remain estimates".into());
    } else {
        // Retain estimated-allowance warnings pending census proof coverage: https://github.com/tpurtell/cuteafd/issues/388.
        notes.push("qwen4: workspace/runtime/Spark allowances are unqualified; validate against the allocation ledger before using this layout for admission".into());
    }
    if gpus == 2 {
        notes.push("Qwen serves on rtx0; rtx1 is idle and contributes no KV capacity".into());
    }
    mapped_table_note(report, &mut notes);
    match resident(checkpoint, mtp, options.qwen_representation) {
        Some(ranks) => {
            push_resident(&mut devices, &ranks, |group| match group {
                "embedding" => Category::Embedding,
                // The native MTP layer loads with the target (ledger scope `weights`).
                _ => Category::Weights,
            });
            // FP8/NVFP4 local experts load at their checkpoint bytes; EXL3 below as arenas.
            if local && !exl3 {
                for component in report.components.iter().filter(|c| c.component == Component::RoutedExpert
                    && c.status != Status::Unused && c.status != Status::Disabled) {
                    devices[0].items.push(Item::new(Category::Experts, component.component.label(),
                        component.formats.keys().cloned().collect::<Vec<_>>().join("+"), component.bytes, Basis::Exact));
                }
            }
        }
        None => {
            let what = "Qwen resident weights: checkpoint headers do not describe the serve-qwen4 resident layout".to_string();
            report.refuse_placement(what.clone(), "Supply the checkpoint's complete coordinator tensor headers.");
            notes.push(what);
        }
    }
    if options.host_embedding {
        place_host_embedding(report, checkpoint, &mut devices, &mut waste, &mut notes);
    }
    if let Some((backbone, draft)) = exl3_arenas(input, mtp) {
        if local {
            devices[0].items.push(Item::new(Category::Experts, "routed expert arenas", "exl3", backbone, Basis::Formula));
        }
        if draft > 0 {
            devices[0].items.push(Item::new(Category::Experts, "native MTP expert arena", "exl3", draft, Basis::Formula));
        }
    } else if mtp {
        let native: u64 = report.components.iter().filter(|c| c.component == Component::SpeculatorExpert && c.status == Status::Unused)
            .map(|c| c.bytes).sum();
        if native > 0 {
            devices[0].items.push(Item::new(Category::Experts, "native MTP experts", "native", native, Basis::Exact));
        }
    }
    if exl3 && (local || mtp) {
        // The EXL3 window retains its shared capacity arenas beside the trellis
        // bytes, from the same capacity manifests the loader reads.
        match expert_workspace(report, model, input, options.workspace_manifest.as_deref(), shape.prefill_rows) {
            Some(bytes) => devices[0].items.push(Item::new(Category::Experts, "local EXL3 workspace", "", bytes, Basis::Formula)),
            // A header-only inventory names no image: its workspace is the layout's question.
            None if shape.manifest.is_none() => notes.push("local EXL3 workspace not charged: no program manifest \
                (--workspace-manifest or an image's PROGRAMS.json) to read its capacity manifests from".into()),
            None => {
                let what = "Qwen local EXL3 workspace: no matching rtx-tp1/m*/v41_exl3.json capacity manifests".to_string();
                report.refuse_placement(what.clone(), "Images bundle them; or export the exl3 tree \
                    alongside PROGRAMS.json (--workspace-manifest).");
                notes.push(what);
            }
        }
    }
    if !local && mtp {
        notes.push("Native Qwen MTP experts stay on rtx0; Spark ranks serve backbone experts only".into());
    }
    if options.drafter_bytes > 0 {
        devices[0].items.push(Item::new(Category::Drafter, "drafter", "", options.drafter_bytes, basis));
    }
    let pkg = if report.placement == ExpertPlacement::Local { local_package(report) }
        else if !local { "none" } else if exl3 { "exl3" } else { "fp8" };
    devices[0].items.push(runtime_item(shape.manifest.as_deref(), &program_families("qwen4", false),
        pkg, false, 0, options.rtx_bytes[0]));
    let spark_ranks = if local { 0 } else { report.placement.spark_ranks() };
    let mut working = crate::placement::SparkWorkingSet { rtx_layers: Vec::new(),
        spark_layers: if spark_ranks > 0 { (0..model.spec().layers.len()).collect() } else { Vec::new() },
        spark_ranks, prefill_lanes: 1 };
    let charges = SparkCharges { host_bytes: Some(SPARK_HOST_BYTES),
        wave_bytes: crate::placement::families::qwen4::SPARK_WORKSPACE_BYTES, basis };
    let endpoints = spark_endpoints("qwen4", 1);
    let mut sparks = if local {
        let mut devices = Vec::new();
        for rank in 0..report.placement.spark_ranks() {
            ensure_tower_spark(&mut devices, rank, options);
        }
        devices
    } else {
        spark_devices(report, model, input, options, charges, &working, endpoints, &mut waste, &mut notes)
    };
    // The vision tower is one choice of the shared solve (no pre-solve pass).
    let encoders = match qwen_encoders(report, options, &sparks, 1) {
        Ok(encoders) => encoders,
        Err(error) => {
            report.refuse_placement(format!("Qwen vision: {error:#}"), "Use the bundled tower or --vision=off.");
            qwen4::QwenEncoders::default()
        }
    };
    // Host owners are stated before the device solve, so a refusal still
    // reports its host demand, as the shared host pass did.
    let host = Qwen4Config::from_hf(&checkpoint.config).map_err(anyhow::Error::from).and_then(|cfg| {
        let span = if options.prefill_waves > 1 {
            crate::serving_capacity::PrefillShape { unit_rows: if options.prefill_rows > 0 { options.prefill_rows } else { 4096 },
                waves: options.prefill_waves }.validate()?.slot_rows()
        } else { shape.prefill_rows };
        host_owners(report, options, input, &cfg, 1, span, mtp)
    });
    let mut pool_tokens = 0;
    match host.as_ref().map_err(|error| anyhow::anyhow!("Qwen host owners: {error:#}")).and_then(|host|
        placement(input, report, options, &mut devices[0], shape.manifest.as_deref(),
            PlanShape { prefill_rows: shape.prefill_rows, concurrency: shape.concurrency, context_tokens: shape.context_tokens, mtp },
            host.clone(), encoders.clone(), &mut sparks, &mut notes)) {
        Ok(tokens) => pool_tokens = tokens,
        Err(error) => {
            let reason = format!("Qwen admission: {error:#}");
            report.refuse_placement(reason.clone(),
                "Supply complete native checkpoint owners within the device and resident-weight budgets.");
            notes.push(reason);
            if let Ok(host) = &host {
                refused_host(report, options, host, &encoders, shape.context_tokens);
            }
        }
    }
    refused_encoder(report, &encoders);
    if pool_tokens < shape.context_tokens {
        notes.push(format!("full-context admission shortfall: context {} tokens, pool {pool_tokens} tokens, shortfall {} tokens",
            shape.context_tokens, shape.context_tokens - pool_tokens));
    }
    if let Some(placement) = &report.solver_placement {
        working.rtx_layers = placement.layers.iter().enumerate().filter_map(|(layer, assignment)|
            (assignment.experts != crate::placement::ExpertHome::Spark).then_some(layer)).collect();
        working.spark_layers = placement.layers.iter().enumerate().filter_map(|(layer, assignment)|
            (assignment.experts == crate::placement::ExpertHome::Spark).then_some(layer)).collect();
        report.onboard_layers = Some(placement.onboard_layers);
    }
    charge_spark_devices(report, model, input, options, charges, &working, endpoints, &mut sparks,
        &mut notes, |device| format!("{} I/O admission", device.name()));
    if local {
        let media_ranks: std::collections::BTreeSet<usize> = report.encoder.as_ref()
            .map_or_else(Vec::new, super::super::encoder::EncoderPlacement::spark_ranks).into_iter().collect();
        sparks.retain(|device| media_ranks.contains(&(device.index as usize)));
    }
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens, waste, notes }
}

/// The planner's Qwen shape inputs, beside `LayoutOptions`.
struct PlanShape {
    prefill_rows: u64,
    concurrency: u64,
    context_tokens: u64,
    mtp: bool,
}

/// serve-qwen4's admission on the planned GPU: the shared fixed items
/// (`qwen_admission`), its startup graph set and the largest pool both fit
/// beside, over what the planner already laid out (context, weights, embedding,
/// head, drafter, resident experts and their package scratch, encoder).
fn placement(input: &super::super::PlanInput, report: &mut PlanReport, options: &LayoutOptions,
    device: &mut DeviceLayout, manifest: Option<&serde_json::Value>, shape: PlanShape,
    host: qwen4::QwenHost, encoders: qwen4::QwenEncoders, sparks: &mut [DeviceLayout], notes: &mut Vec<String>) -> anyhow::Result<u64> {
    use crate::serving_capacity::qwen_graphs::*;
    let checkpoint = input.checkpoint();
    let cfg = crate::families::qwen4::Qwen4Config::from_hf(&checkpoint.config)?;
    let mtp = shape.mtp && cfg.mtp_layers > 0;
    let geometry = crate::serving_capacity::qwen_cache_geometry(&cfg, cfg.layers, mtp, options.qwen_kv)?;
    let mark = geometry.ranks[0].retained_mark_bytes;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(shape.concurrency,
        options.prefix_entries("qwen4"), mark, options.mimo_prefix_mark_bytes, options.host_cache_bytes != Some(0))) * mark;
    let prefill_shape = if options.prefill_waves > 1 {
        anyhow::ensure!(matches!(report.placement, ExpertPlacement::Sparks { .. }),
            "Qwen prefill waves require single-owner Spark experts");
        let unit = if options.prefill_rows > 0 { options.prefill_rows } else { 4096 };
        let units = crate::serving_capacity::PrefillShape { unit_rows: unit, waves: options.prefill_waves }.validate()?;
        Some(units)
    } else { None };
    let span = prefill_shape.map_or(shape.prefill_rows, |s| s.slot_rows());
    // The PLE n-gram table is mapped (each step's rows uploaded): row width ple_dim / ple_rows, one byte
    // per value in E4M3, two in BF16 (`ple::PleTable::load`).
    let ple = checkpoint.tensors.iter().find(|t| t.meta.name.contains("ngram_embedding.shard_"))
        .map(|t| ((cfg.ple_dim / cfg.ple_rows().max(1)) as u64
            * if t.meta.dtype == cuteafd_core::DType::F8E4M3 { 1 } else { 2 }, t.meta.dtype == cuteafd_core::DType::F8E4M3));
    let host_embedding = options.host_embedding && checkpoint.tensors.iter()
        .filter(|tensor| tensor.meta.name.ends_with("embed_tokens.weight")).count() == 1
        && checkpoint.require_untied_embedding("model.language_model.embed_tokens.weight").is_ok();
    let all_onboarded = cfg.layers > 0 && options.resolve_onboard(cfg.layers) == Some(Onboard::Layers(cfg.layers));
    let spark_ranks = if report.placement == ExpertPlacement::Local || all_onboarded { 0 }
        else { report.placement.spark_ranks() };
    let lib = crate::placement::inventory::image_lib(options.workspace_manifest.as_deref()).join("libcuteafd_native.so");
    let catalog = input.expert_catalog()?;
    let experts = qwen4::QwenExperts::read(catalog, &lib, cfg.layers, 1, shape.prefill_rows.max(64))?;
    let embedding = if host_embedding { 0 } else { crate::families::qwen4::weight_layout::embedding_table_bytes(checkpoint)? };
    let arenas = crate::families::qwen4::weight_layout::coordinator_arenas(checkpoint, &cfg,
        &vec![0; cfg.layers], 1, mtp, options.qwen_representation, embedding)?
        .into_iter().map(|(_, arena)| arena).collect();
    let mut future = qwen4::dual_projection_load_staging(options.qwen_representation);
    future.retain(|demand| demand.gpu == 0);
    if mtp { future.extend(qwen4::dual_draft_demands(catalog, &lib, shape.prefill_rows.max(64))?); }
    let loaded_bytes = device.items.iter().filter(|item| item.group == "context+modules")
        .map(|item| item.bytes).sum::<u64>();
    let inputs = qwen4::QwenInputs {
        experts: Some(experts), onboard: options.resolve_onboard(cfg.layers).unwrap_or_default(),
        arenas, expert_arenas: vec![], preallocation: true, future,
        blas_head: if options.qwen_representation.fp8_head { crate::placement::planner::serving::BlasHead::Native } else { options.bf16_target_head },
        loaded_weights: crate::placement::ResidentWeights(options.drafter_bytes),
        resident_weight_limit: Some(crate::placement::ResidentWeights(options.weight_budget_bytes)),
        spark_reply_format: options.resolved_spark_reply_format(false),
        context: crate::placement::planner::Context::resolve(shape.context_tokens, (options.context_tokens > 0).then_some(options.context_tokens)),
        admission: QwenAdmissionInputs { prefill_shape, cfg: &cfg, layers: cfg.layers, mtp, kv_format: options.qwen_kv, manifest,
            prefill_rows: span, slots: options.resolved_state_slots("qwen4"), mark_bytes: marks,
            ple, future_expert_bytes: 0, headroom: crate::placement::reserve::gpu("qwen4") },
        capacity_bytes: options.rtx_bytes[0], baseline: Baseline::Planned { context_bytes: 0, loaded_bytes },
        pending_code_bytes: 0, max_context: shape.context_tokens,
        requested_pool: options.pool_tokens.filter(|&n| n > 0), spark_ranks, host, encoders,
    };
    let (placement, solved) = qwen4::admitted(&inputs)?;
    device.items.retain(|item| item.group == "context+modules");
    publish_host(report, options, &inputs.host, &solved)?;
    adopt_encoder(report, &inputs.encoders, &solved, sparks, notes)?;
    report.prefill_span_rows = Some(span);
    device.items.extend(placement.items[0].iter().cloned());
    device.kv_tokens = placement.pool_tokens;
    let pool_tokens = placement.pool_tokens;
    report.solver_placement = Some(placement);
    Ok(pool_tokens)
}

/// The coordinator host sample before this process allocates (the shared
/// recipe convention); an inventory admits no host owners, so it solves
/// device placement against an unconstrained host.
fn host_sample(options: &LayoutOptions) -> Result<(u64, u64)> {
    if options.inventory && options.host_memory.is_none() { return Ok((u64::MAX >> 2, u64::MAX >> 2)); }
    super::recipe_host_memory(options)
}

/// Coordinator host owners of one Qwen route, from the plan options and the
/// same formulas the runtime allocates with. `owners` names the attention
/// owners that each hold a prefill and a decode upload arena.
fn host_owners(report: &mut PlanReport, options: &LayoutOptions, input: &super::super::PlanInput,
    cfg: &Qwen4Config, owners: usize, span_rows: u64, mtp: bool) -> Result<qwen4::QwenHost> {
    use crate::placement::families::qwen4::host::{HostMemory, PleHost, QwenHost, UploadInputs,
        mtp_pinned_logits_bytes, pinned_logits_bytes, scoring_logits_bytes};
    let checkpoint = input.checkpoint();
    let (total, available) = host_sample(options)?;
    let all_onboarded = cfg.layers > 0
        && options.resolve_onboard(cfg.layers) == Some(Onboard::Layers(cfg.layers));
    let ranks = if all_onboarded || report.placement == ExpertPlacement::Local { 0 } else { report.spark_ranks };
    let unit_rows = options.prefill_unit_rows.unwrap_or(span_rows);
    let waves = options.prefill_waves.max(1);
    let mapped = cfg.ple_layers.first().is_some_and(|&layer| layer < cfg.layers);
    let ple = if mapped {
        let tensor = checkpoint.tensors.iter().find(|t| t.meta.name.contains("ngram_embedding.shard_"))
            .context("Qwen mapped PLE layer without table headers")?;
        anyhow::ensure!(tensor.meta.shape.len() == 2 && tensor.meta.shape[0] > 0, "Qwen PLE table is not a nonempty matrix");
        Some(PleHost::new(cfg, unit_rows, waves, tensor.meta.byte_length / tensor.meta.shape[0] as u64)?)
    } else { None };
    let uploads = UploadInputs { prefill_rows: usize::try_from(unit_rows)?, waves: usize::try_from(waves)?,
        spark: ranks > 0, mtp: mtp && cfg.mtp_layers > 0 && options.native_mtp_layers > 0, ple_ids: !mapped };
    let (bounds, mtp_upload) = uploads.bounds(cfg)?;
    let spark = if ranks > 0 {
        let intake_rows = if waves == 1 { unit_rows } else {
            crate::serving_capacity::PrefillShape { unit_rows, waves }.validate()?.slot_rows() };
        super::spark_host_transport(report, options, input, false, unit_rows, intake_rows, 1)?
            .map(|bytes| bytes * owners as u64)
    } else { [0; 3] };
    let host_embedding = if options.host_embedding || report.memory_layout.as_ref().is_some_and(|m|
        m.notes.iter().any(|n| n.starts_with("embedding placement: host"))) {
        report.components.iter().filter(|c| c.component == Component::Embedding)
            .try_fold(0u64, |n, c| n.checked_add(c.bytes)).context("host embedding overflow")?
    } else { 0 };
    // As `charge_host` did: an unsupported host prefix geometry refuses with
    // its own hint, without erasing the device placement.
    let (prefix_quota, prefix_chunk) = match if options.inventory { Ok((0, 0)) } else { host_prefix(report, options, input, total) } {
        Ok(quota) => quota,
        Err(error) => {
            report.fits = false;
            report.placement_supported = false;
            report.hints.push(super::super::Hint {
                what: format!("qwen4 host prefix cache unsupported (requested {}): {error:#}",
                    options.host_cache_bytes.map_or_else(|| "auto bytes".into(), |n| format!("{n} bytes"))),
                how: "Add exact retained page and mark geometry and a host restore implementation for this family/shape, or disable the host tier with --host-cache-bytes 0.".into(),
            });
            (options.host_cache_bytes.unwrap_or(0), options.host_cache_bytes.unwrap_or(0))
        }
    };
    Ok(QwenHost {
        memory: HostMemory { total, available },
        ple, uploads: vec![bounds; owners], mtp_upload,
        pinned_logits: pinned_logits_bytes(owners, cfg.vocab_size as u64)?,
        mtp_pinned_logits: mtp_pinned_logits_bytes(waves > 1, uploads.mtp, cfg.vocab_size as u64)?,
        scoring_logits: scoring_logits_bytes(span_rows, cfg.vocab_size as u64)?,
        prefix_quota, prefix_chunk, loaded_media_cache: 0,
        host_embedding,
        table_reactor: crate::formats::mapped_table::table_reactor_workspace_bytes(report.mapped_placement.mapped) as u64,
        preloaded_tables: report.mapped_placement.preloaded_bytes,
        spark, push: Vec::new(),
    })
}

/// The host prefix quota and chunk, from the single-owner cache geometry (the
/// dual executor's snapshot concatenates owners into the same aggregate).
fn host_prefix(report: &mut PlanReport, options: &LayoutOptions, input: &super::super::PlanInput, total: u64)
    -> Result<(u64, u64)> {
    if options.prefix_entries("qwen4") == 0 || options.host_cache_bytes == Some(0) { return Ok((0, 0)); }
    let model = input.family_model().map_err(anyhow::Error::from)?;
    let (geometry, context, concurrency) = super::selected_host_geometry(report, options, input.checkpoint(), model, input)?;
    let chunk = geometry.chunk_bytes();
    let bytes = geometry.resolve(options.host_cache_bytes, context, concurrency, total, chunk)
        .with_context(|| format!("invalid host snapshot geometry {geometry:?}, context {context}, concurrency {concurrency}"))?;
    anyhow::ensure!(bytes > 0, "host prefix quota resolves to zero; use a positive quota or explicitly disable host retention");
    Ok((bytes, chunk.min(bytes)))
}

/// Vision tower candidates in preference order (Spark, then RTX, then Off for
/// Auto), for the recipe's encoder choice. `gpus` coordinators may hold an
/// RTX tower; Spark candidates need a Spark worker with tower headroom.
fn qwen_encoders(report: &mut PlanReport, options: &LayoutOptions, sparks: &[DeviceLayout], gpus: usize)
    -> Result<qwen4::QwenEncoders> {
    use super::super::encoder::{candidates, encoder_placement, EncoderGpuBudget, EncoderHardware,
        EncoderKind, EncoderPlacement, EncoderRequest, EncoderSparkBudget};
    use super::super::MediaMode;
    let towers = qwen_towers(report);
    let (weights, scratch) = match towers.vision {
        Ok(bytes) => bytes,
        Err(_) if report.vision == MediaMode::Off => (0, 0),
        Err(error) => anyhow::bail!("Qwen vision tower unavailable: {error}"),
    };
    // Every candidate is priced by the solve; the hardware view only filters
    // Spark workers and names devices, never a KV target.
    let hardware = EncoderHardware { v41: false,
        gpus: (0..gpus).map(|gpu| EncoderGpuBudget { free_bytes: options.rtx_bytes[gpu], kv_target_bytes: 0 }).collect(),
        sparks: sparks.iter().map(|device| EncoderSparkBudget { rank: device.index as usize,
            host: format!("spark{}", device.index), idle: false,
            expert_bytes: device.items.iter().filter(|item| item.category == Category::Experts).map(|item| item.bytes).sum(),
            free_bytes: super::spark_encoder_headroom(device, options.spark_allocation_budget_bytes) }).collect() };
    let request = EncoderRequest { mode: report.vision, weights, scratch, copies: options.vision_replicas };
    let mut encoders = candidates(&request, &hardware);
    if report.vision == MediaMode::Auto && weights > 0 {
        if !encoders.iter().any(|encoder| matches!(encoder.kind, EncoderKind::Rtx { .. })) {
            encoders.extend(candidates(&EncoderRequest { mode: MediaMode::Rtx(None), weights, scratch,
                copies: options.vision_replicas }, &hardware));
        }
        encoders.push(encoder_placement(MediaMode::Off, &hardware, 0, 0, 1));
    }
    if encoders.is_empty() && matches!(report.vision, MediaMode::Spark(_)) && sparks.is_empty() {
        anyhow::bail!("vision-only Spark workers are not supported; set VISION=rtx or RTX_EXPERT_LAYERS");
    }
    anyhow::ensure!(!encoders.is_empty() || report.vision == MediaMode::Off,
        "requested Qwen vision placement has no executable device");
    use crate::placement::planner::{Baseline, Budget, Device, Domain, Location};
    // The media solve owns only candidate workers; full layouts check all ranks separately.
    let media_ranks: std::collections::BTreeSet<usize> = encoders.iter()
        .flat_map(EncoderPlacement::spark_ranks).collect();
    let spark_budgets = sparks.iter().filter(|device| media_ranks.contains(&(device.index as usize))).map(|device| Budget {
        id: format!("spark{}", device.index),
        locations: vec![Location { device: Device::Spark(device.index as u8), domain: Domain::Managed }],
        physical_bytes: device.capacity_bytes,
        logical_bytes: device.used_bytes().saturating_add(
            super::spark_encoder_headroom(device, options.spark_allocation_budget_bytes)),
        baseline: Baseline::Forecast { occupied: device.used_bytes(),
            source: "Qwen Spark worker before vision tower and BLAS".into(), assumed: false },
        headroom: 0,
    }).collect();
    Ok(qwen4::QwenEncoders { candidates: encoders, media_cache: options.media_cache_bytes.unwrap_or_else(||
        (8u64 << 30).min(host_sample(options).map_or(0, |(total, _)| total) / 20)), spark_budgets })
}

/// Adopt the solve's encoder: the chosen candidate, its device items, and the
/// component bookkeeping `resolve_encoder` kept for an Off tower.
fn adopt_encoder(report: &mut PlanReport, encoders: &qwen4::QwenEncoders,
    solved: &crate::placement::planner::ResolvedPlan, sparks: &mut [DeviceLayout], notes: &mut Vec<String>) -> Result<()> {
    use super::super::encoder::{EncoderKind, EncoderPlacement};
    let encoder = encoders.admitted(solved).unwrap_or_else(|| EncoderPlacement { kind: EncoderKind::Off,
        weights: 0, scratch: 0, replicas: Vec::new(), reason: "disabled or no checkpoint tower".into(), shortfall: 0 });
    if encoder.kind == EncoderKind::Off {
        if let Some(c) = report.components.iter_mut().find(|c| c.component == Component::Vision)
            .filter(|c| c.status != Status::Unused) {
            report.disabled_media_bytes += c.bytes;
            c.bytes = 0;
            c.status = Status::Disabled;
        }
        if matches!(report.vision, super::super::MediaMode::Rtx(_) | super::super::MediaMode::Spark(_)) {
            report.placement_supported = false;
            report.hints.push(super::super::Hint { what: format!("requested vision placement unavailable: {}", encoder.reason),
                how: format!("Select an available encoder device with tower/scratch headroom (shortfall {} bytes), or explicitly use --vision=off.", encoder.shortfall) });
        }
    }
    notes.push(format!("vision {:?}: {}; {} bytes admitted; shortfall {} bytes", encoder.kind, encoder.reason,
        encoder.admitted_bytes(), encoder.shortfall));
    report.encoder = Some(encoder);
    for device in sparks {
        device.items.extend(solved.items(crate::placement::planner::Location {
            device: crate::placement::planner::Device::Spark(device.index as u8),
            domain: crate::placement::planner::Domain::Managed })?);
    }
    Ok(())
}

/// A refused solve still names its vision decision: the requested tower
/// (its first candidate), or Off when vision was not requested.
fn refused_encoder(report: &mut PlanReport, encoders: &qwen4::QwenEncoders) {
    use super::super::encoder::{EncoderKind, EncoderPlacement};
    if report.encoder.is_some() { return; }
    let requested = encoders.candidates.first().filter(|candidate| candidate.kind != EncoderKind::Off)
        .filter(|_| report.vision != super::super::MediaMode::Auto);
    report.encoder = Some(match requested {
        Some(candidate) => EncoderPlacement { reason: format!("requested; admission refused: {}", candidate.reason),
            ..candidate.clone() },
        None => EncoderPlacement { kind: EncoderKind::Off, weights: 0, scratch: 0, replicas: Vec::new(),
            reason: "admission refused before an encoder was chosen".into(), shortfall: 0 },
    });
}

/// A refused device solve still publishes its host owners (uploads at the
/// context extent) and prefix quota, plus a requested tower's media cache.
fn refused_host(report: &mut PlanReport, options: &LayoutOptions, host: &qwen4::QwenHost,
    encoders: &qwen4::QwenEncoders, context_tokens: u64) {
    if options.inventory { return; }
    let requested = report.vision != super::super::MediaMode::Auto && encoders.candidates.first()
        .is_some_and(|candidate| candidate.kind != super::super::encoder::EncoderKind::Off);
    let media = if requested { encoders.media_cache } else { 0 };
    let pages = crate::placement::families::qwen4::host::units_of(context_tokens);
    match crate::placement::families::qwen4::host::unsolved_ledger(host, media, pages) {
        Ok(ledger) => {
            report.host_ledger = Some(ledger);
            report.host_budget_bytes = host.prefix_quota;
        }
        Err(error) => report.hints.push(super::super::Hint { what: format!("Qwen host ledger: {error}"),
            how: "Report the host owner geometry.".into() }),
    }
}

/// Publish the admitted host owners and phase ledger on the report.
fn publish_host(report: &mut PlanReport, options: &LayoutOptions, host: &qwen4::QwenHost,
    solved: &crate::placement::planner::ResolvedPlan) -> Result<()> {
    if options.inventory { return Ok(()); }
    report.host_ledger = Some(crate::placement::families::qwen4::host::ledger(host, solved)?);
    report.host_budget_bytes = host.prefix_quota;
    report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(solved));
    Ok(())
}

/// Package scratch of Qwen's resident FP8/NVFP4 experts (`Fp8Experts::load`): one scratch per package
/// the backbone and the mixed-format MTP layer load, at the prefill rows' capacity.
fn package_scratch(input: &super::super::PlanInput, report: &PlanReport, workspace_manifest: Option<&std::path::Path>,
    prefill_rows: u64, mtp: bool) -> Result<Vec<(String, u64)>> {
    let Ok(catalog) = input.expert_catalog() else { return Ok(Vec::new()) };
    let Some(tensors) = catalog.fp8() else { return Ok(Vec::new()) };
    let lib = crate::placement::inventory::image_lib(workspace_manifest);
    let layers = catalog.routed_experts().layers;
    let all_onboarded = layers > 0 && report.onboard_layers == Some(layers);
    let local = report.placement == ExpertPlacement::Local || all_onboarded;
    let mut out = Vec::new();
    if local {
        out.extend(crate::placement::inventory::fp8moe_package_scratch(&lib, "qwen4", tensors.format(), prefill_rows)?);
    }
    if mtp {
        if let Ok(format) = tensors.layer_format(layers) {
            if !local || format != tensors.format() {
                out.extend(crate::placement::inventory::fp8moe_package_scratch(&lib, "qwen4", format, prefill_rows)?);
            }
        }
    }
    Ok(out)
}

fn exl3_arenas(input: &super::super::PlanInput, mtp: bool) -> Option<(u64, u64)> {
    use crate::V41Exl3Layer;
    let catalog = input.expert_catalog().ok()?;
    let shape = catalog.routed_experts();
    let manifest = catalog.exl3()?;
    let bytes = |layer| -> Option<u64> {
        u64::try_from(manifest.residency(layer, 1, 0).ok()?.device_arena_layout().ok()?.1).ok()
    };
    let backbone = (shape.first_layer..shape.layers).try_fold(0u64,
        |total, layer| total.checked_add(bytes(V41Exl3Layer::Backbone(layer))?))?;
    let draft = (0..if mtp { shape.draft_stages.min(1) } else { 0 }).try_fold(0u64,
        |total, stage| total.checked_add(bytes(V41Exl3Layer::Dspark(stage))?))?;
    Some((backbone, draft))
}

fn dual_layout(report: &mut PlanReport, _model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &LayoutOptions, rows: u64, concurrency: u64,
    context: u64, mtp: bool, manifest: Option<&serde_json::Value>) -> MemoryLayout {
    let pkg = if report.experts.as_ref().is_some_and(|e| e.package.contains("exl3")) { "exl3" } else { "fp8" };
    let mut devices = options.rtx_bytes.iter().enumerate().map(|(rank, &capacity)| {
        let arch = crate::placement::ArchContext::for_device("sm_120", capacity);
        let programs = manifest.map(|m| crate::placement::ProgramSet::from_manifest(m, &["qwen4"]))
            .unwrap_or_else(|| crate::placement::ProgramSet::from_names(&["qwen4"], []));
        let allowance = crate::placement::inventory::loaded_code_allowance("qwen4", pkg,
            true, rank as u8, arch, Some(&programs));
        DeviceLayout { kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: capacity, worker_budget_bytes: 0,
            items: vec![Item::new(Category::Runtime, "context+modules", "SM120",
                arch.context_bytes + allowance.bytes, allowance.basis)], kv_tokens: 0 }
    }).collect::<Vec<_>>();
    let mut notes = vec![format!("resolved context {context} tokens"),
        "Qwen whole-layer attention ranges; all routed experts native TP2; head and MTP on rtx0".into()];
    if report.requested_attention.is_some_and(|mode| mode != crate::placement::AttentionPlacement::Layers) {
        let reason = "Qwen dual supports whole-layer attention only, not head or context shards".to_string();
        report.refuse_placement(reason.clone(), "Use whole-layer attention placement.");
        notes.push(reason);
        return MemoryLayout { devices, pool_tokens: 0, waste: Vec::new(), notes };
    }
    report.attention_placement = crate::placement::AttentionPlacement::Layers;
    let cfg = match Qwen4Config::from_hf(&input.checkpoint().config) {
        Ok(cfg) => cfg,
        Err(error) => {
            report.refuse_placement(format!("Qwen dual config: {error:#}"), "Supply a valid Qwen config.json.");
            return MemoryLayout { devices, pool_tokens: 0, waste: Vec::new(), notes };
        }
    };
    let mut sparks = Vec::new();
    for rank in 0..report.placement.spark_ranks() {
        ensure_tower_spark(&mut sparks, rank, options);
    }
    let encoders = match qwen_encoders(report, options, &sparks, 2) {
        Ok(encoders) => encoders,
        Err(error) => {
            report.refuse_placement(format!("Qwen vision: {error:#}"), "Use the bundled tower or --vision=off.");
            qwen4::QwenEncoders::default()
        }
    };
    let host = match host_owners(report, options, input, &cfg, 2, rows, mtp) {
        Ok(host) => host,
        Err(error) => { report.refuse_placement(format!("Qwen host owners: {error:#}"),
            "Supply complete PLE headers and a valid host prefix geometry.");
            return MemoryLayout { devices, pool_tokens: 0, waste: Vec::new(), notes } }
    };
    let (refused, encoders_refused) = (host.clone(), encoders.clone());
    let pool_tokens = match apply(input, options, &mut devices, rows, concurrency, context, mtp, manifest, host, report.placement.spark_ranks(), encoders) {
        Ok((placement, solved, host, encoders)) => {
            if let Err(error) = publish_host(report, options, &host, &solved) {
                report.refuse_placement(format!("Qwen host ledger: {error:#}"), "Report the host owner geometry.");
            }
            if let Err(error) = adopt_encoder(report, &encoders, &solved, &mut sparks, &mut notes) {
                report.refuse_placement(format!("Qwen Spark encoder ledger: {error:#}"), "Report the solved encoder owners.");
            }
            report.solver_placement = Some(placement.clone()); notes.push(format!("placement: {}", placement.summary())); placement.pool_tokens },
        Err(error) => {
            report.refuse_placement(format!("Qwen dual admission: {error:#}"),
                "Supply complete checkpoint headers and exact unequal TP2 plus MTP package manifests.");
            notes.push(format!("Qwen dual admission: {error:#}"));
            refused_host(report, options, &refused, &encoders_refused, context);
            0
        }
    };
    refused_encoder(report, &encoders_refused);
    let media_ranks: std::collections::BTreeSet<usize> = report.encoder.as_ref()
        .map_or_else(Vec::new, super::super::encoder::EncoderPlacement::spark_ranks).into_iter().collect();
    if let Some(placement) = &report.solver_placement {
        let rtx_layers = placement.layers.iter().enumerate().filter_map(|(layer, assignment)|
            (assignment.experts != crate::placement::ExpertHome::Spark).then_some(layer)).collect::<Vec<_>>();
        let spark_layers = placement.layers.iter().enumerate().filter_map(|(layer, assignment)|
            (assignment.experts == crate::placement::ExpertHome::Spark).then_some(layer)).collect::<Vec<_>>();
        report.onboard_layers = Some(placement.onboard_layers);
        if !spark_layers.is_empty() {
            let working = crate::placement::SparkWorkingSet { rtx_layers, spark_layers,
                spark_ranks: report.placement.spark_ranks(), prefill_lanes: 1 };
            let charges = SparkCharges { host_bytes: Some(SPARK_HOST_BYTES),
                wave_bytes: qwen4::SPARK_WORKSPACE_BYTES, basis: Basis::Formula };
            sparks = spark_devices(report, _model, input, options, charges, &working,
                spark_endpoints("qwen4", 1), &mut Vec::new(), &mut notes);
            charge_spark_devices(report, _model, input, options, charges, &working,
                spark_endpoints("qwen4", 1), &mut sparks, &mut notes, |device| format!("{} I/O admission", device.name()));
        } else { sparks.retain(|device| media_ranks.contains(&(device.index as usize))); }
    }
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens, waste: Vec::new(), notes }
}

#[allow(clippy::too_many_arguments)]
fn apply(input: &super::super::PlanInput, options: &LayoutOptions, devices: &mut [DeviceLayout],
    rows: u64, concurrency: u64, context: u64, mtp: bool, manifest: Option<&serde_json::Value>, host_owners: qwen4::QwenHost, spark_ranks: usize,
    encoders: qwen4::QwenEncoders)
    -> Result<(Placement, crate::placement::planner::ResolvedPlan, qwen4::QwenHost, qwen4::QwenEncoders)> {
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
    let unit_rows = options.prefill_unit_rows.unwrap_or(rows);
    let prefill_shape = (options.prefill_waves > 1).then_some(crate::serving_capacity::PrefillShape {
        unit_rows, waves: options.prefill_waves }).map(|shape| shape.validate()).transpose()?;
    let experts = qwen4::QwenExperts::read(&catalog, &lib, cfg.layers, 2, unit_rows.max(64))?;
    let host = options.host_embedding;
    // The loader's own arenas at the fixed cut; the head, entry mixer, MTP and
    // a device embedding are views in them, not separate owner0 demands.
    let embedding = if host { 0 } else { crate::families::qwen4::weight_layout::embedding_table_bytes(checkpoint)? };
    let arenas = crate::families::qwen4::weight_layout::dual_arenas(checkpoint, &cfg,
        &qwen4::dual_owners(cfg.layers), mtp, selected, embedding)?;
    let mut fixed = Vec::new();
    if options.drafter_bytes > 0 {
        fixed.push(crate::placement::Demand::new(0, Category::Drafter, "drafter", options.drafter_bytes, Basis::Exact)
            .with_resident_weights(crate::placement::ResidentWeights(options.drafter_bytes)));
    }
    fixed.extend(qwen4::dual_projection_load_staging(selected));
    if mtp { fixed.extend(qwen4::dual_draft_demands(&catalog, &lib, unit_rows.max(64))?); }
    let extra = qwen4::dual_ple_demands(&cfg, cfg.layers, &resident)?;
    let geometry = crate::serving_capacity::qwen_cache_geometry(&cfg, cfg.layers, mtp, options.qwen_kv)?;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(concurrency,
        options.prefix_entries("qwen4"), geometry.ranks[0].retained_mark_bytes, options.mimo_prefix_mark_bytes, options.host_cache_bytes != Some(0)));
    let ple = checkpoint.tensors.iter().find(|t| t.meta.name.contains("ngram_embedding.shard_"))
        .map(|t| (resident.ple_row_bytes, t.meta.dtype == cuteafd_core::DType::F8E4M3));
    let headroom = crate::placement::reserve::gpu("qwen4");
    let inputs = qwen4::QwenDualInputs {
        experts: Some(experts), blas_head: if options.qwen_representation.fp8_head { crate::placement::planner::serving::BlasHead::Native } else { options.bf16_target_head },
        resident_weight_limit: Some(crate::placement::ResidentWeights(options.weight_budget_bytes)), context: crate::placement::planner::Context::resolve(context, (options.context_tokens > 0).then_some(options.context_tokens)),
        owner_manifests,
        admission: QwenAdmissionInputs { prefill_shape, cfg: &cfg, layers: cfg.layers, mtp, kv_format: options.qwen_kv,
            manifest, prefill_rows: rows, slots: options.resolved_state_slots("qwen4"),
            mark_bytes: marks * geometry.ranks[0].retained_mark_bytes,
            ple, future_expert_bytes: 0, headroom },
        gpus: [0, 1].map(|rank| GpuBudget { capacity_bytes: options.rtx_bytes[rank], reserve_bytes: headroom,
            baseline: Baseline::Planned { context_bytes: 0, loaded_bytes: devices[rank].used_bytes() } }),
        arenas: &arenas, layer_experts: &costs, layer_extra: &extra, fixed,
        tp2_workspace: [0; 2], pending_code_bytes: [0; 2], mark_slots: marks, max_context: context,
        requested_pool: options.pool_tokens.filter(|&n| n > 0), spark_ranks,
        onboard: options.resolve_onboard(catalog.routed_experts().routed_layers()).unwrap_or(Onboard::Auto),
        host: host_owners, encoders,
    };
    let (placement, solved) = qwen4::dual_admitted(&inputs).context("solving Qwen owner ranges")?;
    for (rank, device) in devices.iter_mut().enumerate() {
        device.items.extend(placement.items[rank].iter().cloned());
        device.kv_tokens = placement.pool_tokens;
    }
    Ok((placement, solved, inputs.host, inputs.encoders))
}
