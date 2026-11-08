//! GLM coordinator operands after GlmLoader's packing and rank ownership.
use crate::families::glm5::{GlmDsaConfig, GlmIndexer};
use crate::plan::Checkpoint;
use cuteafd_core::DType;
use anyhow::Context;
use std::collections::BTreeMap;

use super::*;

/// Spark worker allowances at 4096 capacity rows: the non-I/O wave residual
/// (exact rank I/O and rings are charged from the selected working set).
const SPARK: SparkCharges = SparkCharges { host_bytes: None, wave_bytes: gib(190), basis: Basis::Calibrated };

pub(crate) fn layout(report: &mut PlanReport, model: &dyn crate::plan::FamilyModel,
    input: &crate::plan::PlanInput, options: &mut LayoutOptions) -> MemoryLayout {
    let checkpoint = input.checkpoint();
    let mut fallback_note = None;
    let mut context_programs = None;
    if let Ok(cfg) = GlmDsaConfig::from_hf(&checkpoint.config) {
        let gpus = if options.head_split { options.rtx_bytes.len().clamp(1, 2) } else { 1 };
        let manifest = options.workspace_manifest.as_deref().or_else(|| {
            let path = std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json");
            path.is_file().then_some(path)
        }).and_then(|path| input.manifest(path).ok());
        if gpus == 2 && !matches!(report.requested_attention,
            Some(crate::placement::AttentionPlacement::Heads | crate::placement::AttentionPlacement::Layers)) {
            let selected = (|| -> anyhow::Result<_> {
                let (drafts, _) = drafter_admission(options, checkpoint)?;
                let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
                    crate::placement::families::glm5::DECODE_ROWS as usize, drafts, options.glm5_attention_plan_rows)?;
                let sms = options.physical_sms.unwrap_or_else(|| crate::placement::ArchContext::for_device(
                    "sm_120", options.rtx_bytes[0]).sms) as usize;
                manifest.as_ref().map(|manifest| crate::placement::families::glm5::manifest_context_for_cards(
                    &cfg, manifest, &[sms], plan)).transpose().map(Option::flatten)
            })();
            match selected {
                Ok(context) => context_programs = context,
                Err(error) => return unsupported_prefill(report, error),
            }
        }
        let resolved = crate::placement::families::glm5::resolve_context_attention(&cfg,
            report.requested_attention, gpus, context_programs.is_some(), true)
            .expect("standard GLM supports whole-layer attention");
        if report.requested_attention.is_none() && resolved == Some(crate::placement::AttentionPlacement::Layers) {
            fallback_note = Some(format!("auto attention fallback: layers; missing context programs: {}",
                crate::placement::families::glm5::context_program_list(&cfg)));
        }
        if resolved != report.requested_attention {
            report.requested_attention = resolved;
            report.attention_placement = resolved.unwrap_or(report.attention_placement);
        }
    }
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 };
    let layers = report.requested_attention == Some(crate::placement::AttentionPlacement::Layers);
    let limits = crate::serving_capacity::prefill_waves::PrefillLimits {
        max_waves: if spark_ranks == 0 { 1 }
            else if layers { crate::placement::LAYER_PIPELINE_MAX_LANES } else { 4 },
        ..crate::plan::Family::prefill_limits(&crate::plan::families::glm::GLM5)
    };
    let prefill = match super::resolve_prefill(input, report, options, spark_ranks, limits) {
        Ok(shape) => shape,
        Err(error) => return unsupported_prefill(report, error),
    };
    let mut shape = Shape::resolve(input, report, model, options, ShapePolicy::manifest_bounded());
    report.prefill_span_rows = Some(prefill.span_rows);
    let gpus = if options.head_split { options.rtx_bytes.len().clamp(1, 2) } else { 1 };
    let split = gpus == 2 && options.head_split;
    let ranks = if split { 2 } else { 1 };
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: bytes.saturating_sub(crate::placement::reserve::gpu("glm5")), worker_budget_bytes: 0, items: Vec::new(), kv_tokens: 0 }).collect();
    let mut waste = Vec::new();
    let mut notes = vec![format!("resolved context {} tokens (checkpoint {})", shape.context_tokens, shape.checkpoint_context),
        format!("attention placement {}", report.attention_placement)];
    notes.extend(fallback_note);
    shape.select_card_plans(report, options, &mut notes);
    mapped_table_note(report, &mut notes);
    let core = (|| -> anyhow::Result<_> {
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
        let core = crate::families::glm5::core::CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        if core.projections.is_empty() { return Ok(None); }
        let owners = layers.then(|| crate::placement::families::glm5::attention_owners(&cfg, cfg.layers)).transpose()?;
        let (_, draft_rows) = drafter_admission(options, checkpoint)?;
        Ok(Some(crate::families::glm5::core::CoreCharges::from_package(&cfg, &core, cfg.layers,
            ranks, owners.as_deref(), shape.prefill_rows as usize, prefill.waves as usize, draft_rows,
            shape.manifest.as_deref().context("GLM K6 requires PROGRAMS.json with native exports")?)?))
    })();
    let lanes = prefill.waves;
    let mut working = crate::placement::SparkWorkingSet { rtx_layers: Vec::new(),
        spark_layers: if spark_ranks > 0 { (0..model.spec().layers.len()).collect() } else { Vec::new() },
        spark_ranks, prefill_lanes: lanes };
    let mut sparks = spark_devices(report, model, input, options, SPARK, &working,
        spark_endpoints(model.spec().family, lanes), &mut waste, &mut notes);
    let mut pool_tokens = 0;
    let mut local_layers = 0;
    match core.and_then(|core| placement(report, input, options, &devices[..ranks], &sparks, shape.manifest.as_deref(),
        shape.prefill_rows, shape.context_tokens, core.as_ref(), context_programs)) {
        Ok((admitted, prefix_quota, encoder)) => {
            if let Some(note) = admitted.load_overlap_note() { notes.push(note); }
            match figure_recommendation(&admitted, options, checkpoint, shape.manifest.as_deref()) {
                Ok(Some(note)) => notes.push(note),
                Ok(None) => {},
                Err(error) => notes.push(format!("GLM census identity unavailable: {error:#}; admission unchanged")),
            }
            if encoder.kind == crate::plan::encoder::EncoderKind::Off {
                if let Some(component) = report.components.iter_mut().find(|c| c.component == Component::Vision)
                    .filter(|c| c.status != Status::Unused) {
                    report.disabled_media_bytes += component.bytes;
                    component.bytes = 0;
                    component.status = Status::Disabled;
                }
            }
            notes.push(format!("vision {:?}: {}; {} bytes admitted", encoder.kind, encoder.reason, encoder.admitted_bytes()));
            report.encoder = Some(encoder);
            if options.host_embedding {
                let cfg = GlmDsaConfig::from_hf(&checkpoint.config).expect("admitted GLM config");
                match checkpoint.require_untied_embedding(cfg.embedding()) {
                    Ok(()) => notes.push("embedding placement: host (single admitted pinned mapped copy)".into()),
                    Err(error) => notes.push(format!("host embedding ineligible: {error:#}; GPU residency retained")),
                }
            }
            report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&admitted.solved));
            report.host_budget_bytes = prefix_quota;
            let placement = admitted.placement;
            report.solver_placement = Some(placement.clone());
            // The solver resolved auto against this build's programs.
            report.attention_placement = placement.attention_placement;
            if let Some(note) = notes.iter_mut().find(|n| n.starts_with("attention placement ")) {
                *note = format!("attention placement {}", placement.attention_placement);
            }
            working = admitted.spark;
            if working.spark_layers.is_empty() { sparks.clear(); }
            if let Some(encoder) = &report.encoder {
                for rank in encoder.spark_ranks() {
                    ensure_tower_spark(&mut sparks, rank, options);
                    if let Some(device) = sparks.iter_mut().find(|device| device.index as usize == rank) {
                        device.items.push(Item::new(Category::Weights, "vision tower", "", encoder.weights, Basis::Exact));
                        device.items.push(Item::new(Category::Workspace, "vision scratch", "", encoder.scratch, Basis::Exact));
                    }
                }
            }

            notes.push(format!("GLM selected working set: {} Spark layers, {} ranks, {} prefill lanes",
                working.spark_layers.len(), working.spark_ranks, working.prefill_lanes));
            for (device, items) in devices.iter_mut().zip(&placement.items) {
                device.items = items.clone();
                device.kv_tokens = placement.pool_tokens;
            }
            local_layers = placement.onboard_layers;
            pool_tokens = placement.pool_tokens;
            notes.push(format!("placement: {}", placement.summary()));
            if ranks == 2 && placement.attention_placement == crate::placement::AttentionPlacement::Heads {
                waste.push(Waste { device: "rtx1".into(), what: "KV records replicated on both GPUs (MLA latent)".into(),
                    bytes: placement.items[1].iter().filter(|i| i.group == "records").map(|i| i.bytes).sum() });
            }
            if pool_tokens < shape.context_tokens {
                notes.push(format!("full-context admission shortfall: context {} tokens, pool {pool_tokens} tokens",
                    shape.context_tokens));
            }
        }
        Err(error) => {
            // A failure can come from a different mode than the static
            // pre-solve default: auto resolves to the family default
            // (context on two peers) against the manifest, not the
            // plan-time constant. Report the placement actually attempted
            // instead of the stale pre-solve value (issue #202).
            report.attention_placement = attempted_attention(report.requested_attention, ranks);
            if let Some(note) = notes.iter_mut().find(|n| n.starts_with("attention placement ")) {
                *note = format!("attention placement {}", report.attention_placement);
            }
            let what = format!("GLM placement: {error:#}");
            report.refuse_placement(what.clone(), failure_hint(report.attention_placement));
            notes.push(what);
        }
    }
    if local_layers > 0 { notes.push(format!("{local_layers} routed backbone layers resident on RTX")); }
    report.prefill_transports = Some(u64::from(!working.spark_layers.is_empty()));
    // Reapply after admission has selected its executable suffix and lane count.
    charge_spark_devices(report, model, input, options, SPARK, &working, spark_endpoints(model.spec().family, working.prefill_lanes),
        &mut sparks, &mut notes, |device| format!("{} I/O admission", device.name()));
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens, waste, notes }
}

/// The checkpoint's bundled tower: Mistral Large 4's Pixtral, else the generic checkpoint bytes.
fn towers(report: &PlanReport, checkpoint: &Checkpoint) -> Towers {
    if GlmDsaConfig::from_hf(&checkpoint.config).is_ok_and(|cfg| cfg.variant.ml4().is_some()) {
        mistral_large4_towers(report, checkpoint)
    } else {
        Towers::checkpoint(report)
    }
}

pub fn resident_owned(checkpoint: &Checkpoint, ranks: usize, layers: usize, owners: Option<&[u8]>)
    -> anyhow::Result<Vec<Vec<(String, String, u64)>>> {
    anyhow::ensure!((1..=2).contains(&ranks), "GLM coordinator ranks");
    let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
    let core = crate::families::glm5::core::CoreInventory::from_checkpoint(&cfg, checkpoint)?;
    if !core.native.is_empty() {
        return crate::placement::families::glm5::weights::WeightInventory::from_checkpoint(
            checkpoint, ranks, layers, owners, crate::families::glm5::native_bf16(), false,
            crate::placement::planner::EmbeddingHome::Device(0))?.resident_groups();
    }
    let trellis = !core.projections.is_empty();
    anyhow::ensure!(layers <= cfg.layers, "GLM resident layer extent exceeds checkpoint");
    anyhow::ensure!(owners.is_none_or(|owners| ranks == 2 && owners.len() == layers
        && owners.iter().all(|&owner| owner < 2)), "GLM attention owner inventory");
    let tensors: BTreeMap<_, _> = checkpoint.tensors.iter().map(|t| (t.meta.name.as_str(), &t.meta)).collect();
    let tensor = |name: &str| tensors.get(name).copied().ok_or_else(|| anyhow::anyhow!("missing GLM operand {name}"));
    let mut groups = vec![BTreeMap::<(String, String), u64>::new(); ranks];
    let mut put = |rank: usize, group: &str, format: &str, bytes: u64| {
        *groups[rank].entry((group.into(), format.into())).or_default() += bytes.max(256);
    };
    let native = crate::families::glm5::native_bf16();
    use crate::families::glm5::names::Operand as Op;
    // Operand names come from the checkpoint's variant (GLM HF names, Mistral
    // consolidated names); a variant without an operand has no such tensor.
    let w = |layer: usize, op: Op| cfg.weight(layer, op);
    for layer in 0..layers {
        // Direct-query variants project the query straight from the hidden
        // state (no q-LoRA); latent-sharing layers carry no wkv_a / kv_a norm.
        let kv_a = w(layer, Op::KvA);
        let q_in = w(layer, Op::QA).unwrap_or_else(|| w(layer, Op::QB).expect("every variant has a query projection"));
        let qkv: Vec<String> = [Some(q_in.clone()), kv_a.clone()].into_iter().flatten().collect();
        let q_b = w(layer, Op::QA).and(w(layer, Op::QB));
        let mut attention = qkv.clone();
        attention.extend(q_b.iter().cloned());
        attention.extend([w(layer, Op::KvB).unwrap(), w(layer, Op::O).unwrap()]);
        let core_attention = core.get_weight(&q_in).is_some();
        let bf16 = native && attention.iter().all(|n| tensor(n).is_ok_and(|t| t.dtype == DType::Bf16))
            || core_attention && tensor(&w(layer, Op::KvB).unwrap())?.dtype == DType::Bf16;
        let qkv_rows: u64 = if core_attention { 0 } else {
            qkv.iter().map(|n| tensor(n).map(|t| t.shape[0] as u64)).sum::<anyhow::Result<_>>()?
        };
        let h = cfg.hidden as u64;
        let projection = |name: &str| -> anyhow::Result<(u64, u64)> {
            let t = tensor(name)?;
            anyhow::ensure!(t.shape.len() == 2, "GLM matrix {name}");
            Ok((t.shape[0] as u64 * t.shape[1] as u64, (t.shape[0] as u64).div_ceil(128) * (t.shape[1] as u64).div_ceil(128) * 4))
        };
        for rank in 0..ranks {
            for name in [w(layer, Op::InputNorm).unwrap(), w(layer, Op::PostAttentionNorm).unwrap()] {
                put(rank, "norm", "bf16", tensor(&name)?.byte_length);
            }
            if owners.is_none_or(|owners| owners[layer] as usize == rank) {
                let attention_ranks = if owners.is_some() { 1 } else { ranks };
                for name in [w(layer, Op::QANorm), w(layer, Op::KvANorm), w(layer, Op::QueryScaleS), w(layer, Op::QueryScaleB)]
                    .into_iter().flatten() {
                    put(rank, "norm", "bf16", tensor(&name)?.byte_length);
                }
                if core_attention {
                    for name in qkv.iter().chain(q_b.iter()).chain([w(layer, Op::O).unwrap()].iter()) {
                        let projection = core.get_weight(name).context("complete GLM native attention group")?;
                        put(rank, "attention", "exl3-k6", projection.resident_bytes(attention_ranks)?);
                    }
                } else {
                    // The q-LoRA / latent down projections stay whole; the query up
                    // (or direct query) and output split by heads.
                    let whole = if q_b.is_some() { qkv.clone() } else { kv_a.iter().cloned().collect() };
                    let whole_rows: u64 = whole.iter().map(|n| tensor(n).map(|t| t.shape[0] as u64)).sum::<anyhow::Result<_>>()?;
                    put(rank, "attention", if bf16 { "bf16" } else { "fp8" }, whole_rows * h * if bf16 { 2 } else { 1 });
                    if !bf16 {
                        let grid = whole.iter().map(|n| projection(n).map(|(_, s)| s)).sum::<anyhow::Result<u64>>()?;
                        put(rank, "attention", "fp8-scale", grid);
                        put(rank, "attention", "fp8-scale", whole_rows * h.div_ceil(128) * 4);
                    }
                    let split = [q_b.clone().unwrap_or(q_in.clone()), w(layer, Op::O).unwrap()];
                    for name in split {
                        let (values, scales) = projection(&name)?;
                        put(rank, "attention", if bf16 { "bf16" } else { "fp8" }, values / attention_ranks as u64 * if bf16 { 2 } else { 1 });
                        if !bf16 { put(rank, "attention", "fp8-scale", scales / attention_ranks as u64); }
                    }
                    let _ = qkv_rows;
                }
                for width in [cfg.nope_dim(layer), cfg.v_head_dim] {
                    let values = cfg.heads as u64 / attention_ranks as u64 * cfg.kv_lora_rank as u64 * width as u64;
                    put(rank, "attention", if bf16 { "bf16" } else { "fp8" }, values * if bf16 { 2 } else { 1 });
                    if !bf16 { put(rank, "attention", "fp8-scale", values / 64 * 4); }
                }
                if cfg.indexers[layer] == GlmIndexer::Full {
                    let name = w(layer, Op::IndexQ).unwrap();
                    // A direct-query variant may carry wq_b in the K6 core (whole, per attention rank).
                    if let Some(core_q) = cfg.stem(layer, Op::IndexQ).and_then(|stem| core.get_weight(&stem)) {
                        put(rank, "indexer", "exl3-k6", core_q.resident_bytes(1)?);
                    } else {
                    let index_bf16 = (native || trellis) && tensor(&name)?.dtype == DType::Bf16;
                    let (values, scales) = projection(&name)?;
                    put(rank, "indexer", if index_bf16 { "bf16" } else { "fp8" }, values * if index_bf16 { 2 } else { 1 });
                    if !index_bf16 { put(rank, "indexer", "fp8-scale", scales); }
                    }
                    let ik = [Op::IndexK, Op::IndexWeights].into_iter().map(|op| tensor(&w(layer, op).unwrap())
                        .map(|t| t.shape.iter().product::<usize>() as u64 * 2)).sum::<anyhow::Result<u64>>()?;
                    put(rank, "indexer", "bf16", ik);
                    let k_norm = cfg.stem(layer, Op::IndexKNorm).unwrap();
                    let mut vectors = vec![format!("{k_norm}.weight"), format!("{k_norm}.bias")];
                    if let Some(q_norm) = cfg.stem(layer, Op::IndexQNorm) {
                        vectors.extend([format!("{q_norm}.weight"), format!("{q_norm}.bias")]);
                    }
                    vectors.extend([w(layer, Op::IndexQueryScaleS), w(layer, Op::IndexQueryScaleB)].into_iter().flatten());
                    for name in vectors {
                        put(rank, "indexer", "bf16", tensor(&name)?.byte_length);
                    }
                }
            }
            let dense = layer < cfg.first_moe_layer;
            let names = [w(layer, Op::Gate).unwrap(), w(layer, Op::Up).unwrap(), w(layer, Op::Down).unwrap()];
            let shared_bf16 = !dense && native && names.iter().all(|n| tensor(n).is_ok_and(|t| t.dtype == DType::Bf16));
            let group = if dense { "dense" } else { "shared_expert" };
            if core.get_weight(&names[0]).is_some() {
                for name in &names {
                    put(rank, group, "exl3-k6", core.get_weight(name)
                        .context("complete GLM native FFN group")?.resident_bytes(ranks)?);
                }
            } else {
                let gu = names[..2].iter().map(|n| projection(n)).collect::<anyhow::Result<Vec<_>>>()?;
                let down = projection(&names[2])?;
                for (values, scales) in [(gu[0].0 + gu[1].0, gu[0].1 + gu[1].1), down] {
                    put(rank, group, if shared_bf16 { "bf16" } else { "fp8" }, values / ranks as u64 * if shared_bf16 { 2 } else { 1 });
                    if !shared_bf16 { put(rank, group, "fp8-scale", scales / ranks as u64); }
                }
            }
            if dense && names.iter().all(|n| tensor(n).is_ok_and(|t| t.dtype == DType::F8E4M3)
                && tensors.contains_key(format!("{n}_scale").as_str())) {
                put(rank, group, "fp8-tensor-scale", 48);
                put(rank, group, "fp8-tensor-scale", 48);
            }
            if !dense && rank == 0 {
                for name in [w(layer, Op::Router), w(layer, Op::RouterBias)].into_iter().flatten() {
                    put(rank, "router", "bf16", tensor(&name)?.byte_length);
                }
            }
        }
    }
    let head = format!("{}.weight", cfg.head());
    for (name, group) in [(cfg.embedding(), "embedding"), (cfg.final_norm(), "norm"), (head.as_str(), "lm_head")] {
        if let Some(projection) = core.get_weight(name) {
            put(0, group, "exl3-k6", projection.resident_bytes(1)?);
            continue;
        }
        let t = tensor(name)?;
        let bytes = if t.dtype == DType::F8E4M3 && t.shape.len() == 2 { t.shape.iter().product::<usize>() as u64 * 2 } else { t.byte_length };
        put(0, group, "bf16", bytes);
    }
    // LeRoPE alpha stays on the host; only its generated RoPE table owns device storage.
    Ok(groups.into_iter().map(|g| g.into_iter().map(|((group, format), bytes)| (group, format, bytes)).collect()).collect())
}

/// GLM's shared admission over header-derived coordinator storage.
/// The attention placement GLM 5.3's solve attempts: the requested mode, or
/// the family default for auto (context on a two-GPU peer pair, heads
/// otherwise).
pub(crate) fn attempted_attention(requested: Option<crate::placement::AttentionPlacement>, gpus: usize)
    -> crate::placement::AttentionPlacement {
    requested.unwrap_or(if gpus == 2 { crate::placement::AttentionPlacement::Context }
        else { crate::placement::AttentionPlacement::Heads })
}

/// Why GLM 5.3's solve failed and what to change, naming the attempted mode,
/// so a failure never points at a placement that was not tried (issue #202).
fn failure_hint(mode: crate::placement::AttentionPlacement) -> String {
    use crate::placement::AttentionPlacement;
    let rebuild = "supply a PROGRAMS.json carrying the glm_* context-split kernels \
        (rebuild the GLM 5.3 coordinator programs) so placement context can run";
    match mode {
        AttentionPlacement::Heads => format!("GLM 5.3 cannot admit this request with attention \
            placement heads: heads replicates the MLA latent and DSA index on both GPUs, so a pool \
            past the device budget does not fit. Use an admissible --pool-tokens, drop local RTX \
            experts, or {rebuild}."),
        AttentionPlacement::Context => format!("GLM 5.3 cannot admit this request with attention \
            placement context: {rebuild}, or request `heads` explicitly."),
        AttentionPlacement::Layers => "GLM 5.3 layers attention needs two coordinator GPUs and whole-attention programs; use an admissible pool or context.".into(),
    }
}

fn drafter_admission(options: &LayoutOptions, checkpoint: &Checkpoint)
    -> anyhow::Result<(Option<usize>, Option<usize>)> {
    if options.glm5_drafter_disabled { return Ok((None, None)); }
    let sequences = usize::try_from(options.glm_draft_sequences())?;
    if checkpoint.model_type() == Some("mistral_large4") {
        let slots = crate::families::glm5::draft_representation::serving_draft_context_slots(
            sequences, usize::try_from(options.resolved_concurrency("mistral_large4"))?,
            options.draft_context_slots.map(usize::try_from).transpose()?);
        let context = if options.context_tokens == 0 { 1 << 20 } else { usize::try_from(options.context_tokens)? };
        let layout = crate::placement::families::glm5::ml4_draft_layout(
            &checkpoint.snapshot, slots, sequences, context, 1)?;
        return Ok((Some(layout.capacity().block_rows / sequences), Some(layout.capacity().block_rows)));
    }
    let snapshot = options.glm5_drafter_snapshot.clone().or_else(|| {
        let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("/mnt/sparknest/hf-home"));
        crate::resolve_snapshot_at_revision("incoai/GLM-5.3-DFlash2", Some(&home), None).ok()
            .and_then(|resolved| resolved.snapshot_path)
    }).context("GLM drafter checkpoint unavailable; supply its snapshot or disable speculation")?;
    let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
    let (_, block) = crate::families::glm5::draft_representation::draft_geometry(&config)?;
    let block = usize::try_from(block)?;
    Ok((Some(block.checked_sub(1).context("GLM draft block must include an anchor")?),
        Some(sequences.checked_mul(block).context("GLM draft head row extent")?)))
}

fn figure_recommendation(admitted: &crate::placement::families::glm5::admission::AdmittedGlm,
    options: &LayoutOptions, checkpoint: &Checkpoint, manifest: Option<&serde_json::Value>) -> anyhow::Result<Option<String>> {
    use sha2::{Digest, Sha256};
    let hash = match options.workspace_manifest.as_deref() {
        Some(path) => format!("{:x}", Sha256::digest(std::fs::read(path)?)),
        None => manifest.map(|value| serde_json::to_vec(value).map(|bytes| format!("{:x}", Sha256::digest(bytes))))
            .transpose()?.unwrap_or_else(|| "unavailable".into()),
    };
    let layout = format!("snapshot/{}; context/{}; concurrency/{:?}; draft/{:?}; replies/{:?}",
        checkpoint.snapshot.display(), options.context_tokens, options.concurrency,
        options.glm5_drafter_snapshot, options.spark_reply_formats);
    Ok(admitted.census_recommendation(crate::placement::inventory::FigureIdentity {
        declared_graphs: "", programs_sha256: &hash, native_sha256: "unavailable in header-only plan",
        toolchain: "unavailable in header-only plan", gpu_class: "requested physical card capacities",
        driver: "unavailable in header-only plan", layout: &layout,
    })?)
}

fn production_serving(options: &LayoutOptions, checkpoint: &Checkpoint, cfg: &GlmDsaConfig,
    context: u64, attention: crate::placement::AttentionPlacement, gpus: usize, maximum_pages: u64)
    -> anyhow::Result<(crate::placement::families::glm5::admission::DraftFacts,
        Vec<crate::placement::planner::Component>, u64)> {
    use crate::placement::{families::glm5::{self as glm, admission::DraftFacts}, planner::*};
    use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation, serving_draft_context_slots};
    use crate::placement::planner::serving::{HostPrefixOwners, SelectorOwners, ServingOwners};
    let sms = options.physical_sms.unwrap_or_else(|| crate::placement::ArchContext::for_device(
        "sm_120", options.rtx_bytes[0]).sms) as usize;
    let concurrency = options.resolved_concurrency(cfg.family_id()) as usize;
    let sequences = options.glm_draft_sequences() as usize;
    let slots = serving_draft_context_slots(sequences, concurrency, options.draft_context_slots.map(|n| n as usize));
    let host_enabled = crate::families::glm5::host_prefix_enabled(gpus, attention,
        options.host_cache_bytes != Some(0), options.mimo_prefix_entries);
    let mut mark_bytes = 0;
    let mut mark_slots = 0;
    let draft = if options.glm5_drafter_disabled { DraftFacts::default() }
        else if checkpoint.model_type() == Some("mistral_large4") {
            let layout = glm::ml4_draft_layout(&checkpoint.snapshot, slots, sequences, context as usize, sms)?;
            mark_bytes = glm::draft_window_mark_bytes(&layout)? as u64;
            if options.mimo_prefix_entries > 0 {
                mark_slots = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(
                    concurrency as u64, options.mimo_prefix_entries, mark_bytes, options.mimo_prefix_mark_bytes,
                    host_enabled));
            }
            DraftFacts::ml4(&layout)?
        } else {
            let snapshot = options.glm5_drafter_snapshot.clone().or_else(|| {
                let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
                    .unwrap_or_else(|| std::path::PathBuf::from("/mnt/sparknest/hf-home"));
                crate::resolve_snapshot_at_revision("incoai/GLM-5.3-DFlash2", Some(&home), None).ok()
                    .and_then(|resolved| resolved.snapshot_path)
            }).context("GLM shared admission needs the drafter's physical checkpoint headers; disable speculation or supply its snapshot")?;
            let draft = crate::families::glm5::draft_physical::GlmDflashConfig::read(&snapshot)?;
            DraftFacts::gqa(&snapshot, if options.glm5_draft_bf16 { GlmDraftRepresentation::Bf16Only }
                else { GlmDraftRepresentation::Fp8Only }, GlmDraftCapacity::new(slots, sequences, draft.block)?,
                sms, 0, options.draft_kv_cache)?
        };
    let host = options.host_memory.map(Ok).unwrap_or_else(crate::formats::mapped_preload::host_memory_before_process)?;
    let cache = if let Some(ml4) = cfg.variant.ml4() {
        crate::serving_capacity::mistral_large4_cache_geometry(ml4, 1)?
    } else { crate::serving_capacity::glm_cache_geometry(cfg, cfg.layers, 1)? };
    let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: cache.logical_unit_rows,
        page_bytes: cache.ranks[0].persistent_unit_bytes, mark_bytes, draft_bytes: 0 };
    let quota = if !host_enabled { 0 }
        else { geometry.resolve(options.host_cache_bytes, context, concurrency as u64, host.0, geometry.chunk_bytes())
            .context("invalid GLM host prefix geometry or quota")? };
    let serving = ServingOwners { selector: SelectorOwners { location: Location::coordinator(0),
        capacity: 64, vocab: cfg.vocab_size, device_sampling: true },
        prefix: HostPrefixOwners { quota_bytes: quota, chunk_bytes: geometry.chunk_bytes().min(quota) } };
    let owners = serving.components(maximum_pages)?;
    // Window marks snapshot the drafter's rings and live beside them.
    let draft = draft.with_window_marks(mark_slots.checked_mul(mark_bytes).context("GLM draft mark extent overflow")?);
    Ok((draft, owners, quota))
}

fn placement(report: &PlanReport, input: &crate::plan::PlanInput, options: &LayoutOptions,
    devices: &[DeviceLayout], sparks: &[DeviceLayout], manifest: Option<&serde_json::Value>, rows: u64, context: u64,
    core: Option<&crate::families::glm5::core::CoreCharges>,
    context_programs: Option<crate::placement::families::glm5::GlmContext>)
    -> anyhow::Result<(crate::placement::families::glm5::admission::AdmittedGlm, u64, crate::plan::encoder::EncoderPlacement)> {
    use crate::placement::{families::glm5 as glm, planner::*};
    use glm::{admission::*, local_physical::*};
    let checkpoint = input.checkpoint();
    let cfg = crate::families::glm5::GlmDsaConfig::from_hf(&checkpoint.config)?;
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 };
    let lanes = options.prefill_waves;
    anyhow::ensure!(report.requested_attention != Some(crate::placement::AttentionPlacement::Layers)
        || lanes <= crate::placement::LAYER_PIPELINE_MAX_LANES,
        "layer-owned prefill lacks capacity for {lanes} waves");
    let (active_drafts, _) = drafter_admission(options, checkpoint)?;
    let attention_plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
        glm::DECODE_ROWS as usize, active_drafts, options.glm5_attention_plan_rows)?;
    let selected_manifest = manifest.map(|m| -> anyhow::Result<_> {
        let mut selected = m.clone();
        let family = cuteafd_core::coordinator_programs::CoordinatorPrograms { family: cfg.program_family(),
            split_family: if devices.len() == 2 { cfg.split_program_family() } else { None } };
        cuteafd_core::coordinator_programs::select_attention_plans(&mut selected, attention_plan,
            |name| family.contains(name))?;
        Ok(selected)
    }).transpose()?;
    let scratch = selected_manifest.as_ref().map(|m| -> anyhow::Result<_> {
        let mut scratch = glm::manifest_scratch(&cfg, devices.len() == 2, m)?;
        if report.requested_attention == Some(crate::placement::AttentionPlacement::Layers) {
            let whole = glm::manifest_scratch(&cfg, false, m)?;
            for i in 0..2 { scratch[i].programs = scratch[i].programs.max(whole[i].programs); }
        }
        Ok(scratch)
    }).transpose()?;
    let gpus = devices.len();
    let attention = glm::resolve_context_attention(&cfg, report.requested_attention, gpus,
        context_programs.is_some(), true)?.unwrap_or(if gpus == 2 {
            crate::placement::AttentionPlacement::Context
        } else { crate::placement::AttentionPlacement::Heads });
    if attention == crate::placement::AttentionPlacement::Context && context_programs.is_none() {
        return Err(crate::placement::PlacementError::MissingContextPrograms {
            family: cfg.family_id(), programs: glm::context_program_list(&cfg),
        }.into());
    }
    let page_tokens = if attention == crate::placement::AttentionPlacement::Context { 128 } else { 64 };
    let maximum_pool_tokens = crate::placement::PoolPolicy::resolve(
        &options.rtx_bytes[..gpus], context, options.pool_tokens, page_tokens).maximum_tokens();
    let maximum_pages = maximum_pool_tokens.div_ceil(page_tokens);
    let prefill = PrefillShape { rows, waves: lanes };
    let operation = if options.full_prefill_logits { Operation::Scoring } else { Operation::Serving };
    let onboard = options.resolve_onboard(cfg.layers.saturating_sub(cfg.first_moe_layer)).unwrap_or_else(glm::default_onboard);
    let exchange_f32 = glm::local_exchange_f32()?;
    let manifest_path = options.workspace_manifest.as_deref().unwrap_or_else(||
        std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let catalog = input.expert_catalog()?;
    let local = glm::select_local_inventory(&cfg, &catalog, manifest_path, cfg.layers, devices.len(),
        rows, lanes, exchange_f32, onboard, false, None)?;
    let owners = (attention == crate::placement::AttentionPlacement::Layers)
        .then(|| glm::attention_owners(&cfg, cfg.layers)).transpose()?;
    let embedding = if options.host_embedding && checkpoint.require_untied_embedding(cfg.embedding()).is_ok() {
        EmbeddingHome::HostPinned
    } else { EmbeddingHome::Device(0) };
    let target = TargetFacts::checkpoint(checkpoint, cfg.layers, gpus, owners.as_deref(), crate::families::glm5::native_bf16(), &[embedding], core)?;
    let physical = LocalPhysicalInput { layers: (cfg.first_moe_layer..cfg.layers).collect(),
        max_rows: rows.max(glm::DECODE_ROWS) as usize, lanes: lanes as usize,
        storage: LocalStorage::for_snapshot(&checkpoint.snapshot)?,
        route_checks: true, peer_diagnostics: false, activations: options.exlr8_prefill_activations };
    let source = local.as_ref().map(|local| LocalPhysicalSource::read(&cfg, &catalog, local, &physical, None)).transpose()?;
    let (draft, mut additional, prefix_quota) = production_serving(options, checkpoint, &cfg,
        context, attention, gpus, maximum_pages)?;
    if let Some(envelope) = push_host_envelope(checkpoint, &cfg, cfg.layers, gpus, Some(&catalog), None, 0)? {
        additional.extend(envelope.allocations("glm.load.push")?.into_iter().map(|owner| {
            let bytes = ByteCurve::fixed(owner.bytes, maximum_pages);
            Component::allocation(owner, bytes, false, maximum_pages)
        }));
    }
    use crate::plan::encoder::{EncoderRequest, EncoderHardware, EncoderGpuBudget, EncoderSparkBudget, EncoderKind};
    let towers = towers(report, checkpoint);
    let (weights, encoder_scratch) = match towers.vision {
        Ok(bytes) => bytes,
        Err(_) if report.vision == crate::plan::MediaMode::Off => (0, 0),
        Err(error) => anyhow::bail!("bundled vision tower unavailable: {error}"),
    };
    anyhow::ensure!(weights > 0 || matches!(report.vision, crate::plan::MediaMode::Auto | crate::plan::MediaMode::Off),
        "explicit vision request requires a bundled checkpoint tower");
    let hardware = EncoderHardware { v41: false,
        gpus: devices.iter().map(|device| EncoderGpuBudget { free_bytes: device.capacity_bytes,
            kv_target_bytes: 0 }).collect(),
        sparks: sparks.iter().map(|device| EncoderSparkBudget { rank: device.index as usize,
            host: format!("spark{}", device.index), idle: false,
            expert_bytes: device.items.iter().filter(|item| item.category == Category::Experts).map(|item| item.bytes).sum(),
            free_bytes: spark_encoder_headroom(device, options.spark_allocation_budget_bytes) }).collect() };
    let mut encoders = crate::plan::encoder::candidates(&EncoderRequest { mode: report.vision,
        weights, scratch: encoder_scratch, copies: options.vision_replicas }, &hardware);
    if report.vision == crate::plan::MediaMode::Auto && weights > 0 {
        // Keep the RTX alternative even when the tower alone exceeds capacity:
        // the shared contract supplies the phase-specific refusal diagnostic.
        if !encoders.iter().any(|encoder| matches!(encoder.kind, EncoderKind::Rtx { .. })) {
            encoders.extend(crate::plan::encoder::candidates(&EncoderRequest {
                mode: crate::plan::MediaMode::Rtx(None), weights, scratch: encoder_scratch,
                copies: options.vision_replicas }, &hardware));
        }
        encoders.push(crate::plan::encoder::encoder_placement(crate::plan::MediaMode::Off, &hardware, 0, 0, 1));
    }
    anyhow::ensure!(!encoders.is_empty(), "requested bundled encoder has no executable device");
    let (decode_link, prefill_link) = production_links(&cfg, spark_ranks, prefill,
        if checkpoint.spark_input_is_bf16() { cuteafd_core::SparkIngress::Bf16 } else { cuteafd_core::SparkIngress::Fp8K32 },
        cuteafd_core::SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 },
        options.spark_reply_formats)?;
    let host = options.host_memory.map(Ok).unwrap_or_else(crate::formats::mapped_preload::host_memory_before_process)?;
    let budgets = production_budgets(&options.rtx_bytes[..gpus].iter().map(|&n| (n, n)).collect::<Vec<_>>(),
        crate::placement::reserve::gpu("glm5"), host, (8 << 30).max(host.0 / 10), &vec![options.spark_bytes; spark_ranks])?;
    let request = production_request(crate::placement::planner::Context::resolve(context, (options.context_tokens > 0).then_some(options.context_tokens)), options.pool_tokens, onboard, prefill, operation,
        crate::placement::ResidentWeights(options.weight_budget_bytes), options.spark_reply_formats);
    let mut prepared = ProductionInput { expert_usage: options.expert_usage, cfg: &cfg, layers: cfg.layers, context, maximum_pool_tokens,
        attention, gpus, prefill,
        blas_head: if core.is_some_and(|core| core.ranks[0].target_head.is_some()) { serving::BlasHead::Native }
            else { options.bf16_target_head }, blas: vec![], target, draft, scratch: scratch.unwrap_or_default(), context_scratch: context_programs,
        exchange_element_bytes: if exchange_f32 { 4 } else { 2 }, peer_diagnostics: false,
        prefix_host: crate::families::glm5::host_prefix_enabled(gpus, attention,
            options.host_cache_bytes != Some(0), options.mimo_prefix_entries),
        local: source.as_ref().map(|source| (source, physical)), spark_ranks, decode_link, prefill_link,
        skip_routed_experts: false, operation, sync_upload_bytes: 0, owners: additional, budgets,
    }.prepare()?;
    prepared.recipe.encoders = encoders.clone();
    prepared.recipe.media_cache_bytes = options.media_cache_bytes.unwrap_or((8u64 << 30).min(host.0 / 20));
    let admitted = prepared.admit(request)?;
    let mut encoder = encoders[admitted.solved.decisions.encoder.context("missing admitted encoder")?].clone();
    if report.vision == crate::plan::MediaMode::Auto && weights > 0 && encoder.kind == EncoderKind::Off {
        encoder.reason = format!("Auto disabled image input at the pool-first/onboarding solution: {}; free memory or choose a smaller layout",
            admitted.solved.encoder_diagnostics.iter().map(|(reason, _)| reason.as_str()).collect::<Vec<_>>().join("; "));
        encoder.shortfall = admitted.solved.encoder_diagnostics.iter().map(|(_, bytes)| *bytes).filter(|bytes| *bytes > 0).min().unwrap_or(0);
        tracing::warn!(reason = %encoder.reason, shortfall_bytes = encoder.shortfall, "bundled vision disabled by shared admission");
    }
    Ok((admitted, prefix_quota, encoder))
}

/// Native EXLR8 supplies physical owners to the shared production admission.
pub fn native_layout(checkpoint: &Checkpoint, options: &LayoutOptions,
    spark_ranks: usize, manifest: &serde_json::Value, selected: &str)
    -> anyhow::Result<(MemoryLayout, crate::placement::Placement)> {
    native_layout_with_metadata(checkpoint, options, spark_ranks, manifest, selected, None, None, None)
}

pub(crate) fn native_layout_with_metadata(checkpoint: &Checkpoint, options: &LayoutOptions,
    spark_ranks: usize, manifest: &serde_json::Value, selected: &str,
    metadata: Option<&crate::plan::exlr8::residency::MetadataInput>, requested: Option<crate::placement::AttentionPlacement>,
    report: Option<&mut PlanReport>)
    -> anyhow::Result<(MemoryLayout, crate::placement::Placement)> {
    use crate::placement::{families::glm5 as glm, planner::*};
    use glm::{admission::*, local_physical::*};
    use crate::plan::exlr8::{dense::{DensePlan, DenseRecord, DenseRole, DenseSelection}, residency::{ExpertSet, MetadataInput}};
    let gpu_count = options.rtx_bytes.len();
    anyhow::ensure!((1..=2).contains(&gpu_count), "EXLR8 requires 1..=2 coordinator GPUs");
    let owned_metadata;
    let metadata = if let Some(metadata) = metadata { metadata } else {
        owned_metadata = MetadataInput::open(&checkpoint.snapshot, selected)?;
        &owned_metadata
    };
    let split = gpu_count == 2;
    let attention = requested.unwrap_or(if split { crate::placement::AttentionPlacement::Context }
        else { crate::placement::AttentionPlacement::Heads });
    anyhow::ensure!(attention == crate::placement::AttentionPlacement::Heads
        || (split && matches!(attention, crate::placement::AttentionPlacement::Context | crate::placement::AttentionPlacement::Layers)),
        "GLM EXLR8 requires heads or two-GPU context/layers attention");
    let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
    let rows = if options.prefill_rows == 0 { 4096 } else { options.prefill_rows };
    let context = if options.context_tokens == 0 { 1 << 20 } else { options.context_tokens };
    let (active_drafts, _) = drafter_admission(options, checkpoint)?;
    let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
        glm::DECODE_ROWS as usize, active_drafts, options.glm5_attention_plan_rows)?;
    let mut manifest = manifest.clone();
    cuteafd_core::coordinator_programs::select_attention_plans(&mut manifest, plan,
        |name| cuteafd_core::coordinator_programs::CoordinatorPrograms {
            family: "glm", split_family: Some("glm2") }.contains(name))?;
    let owners = (attention == crate::placement::AttentionPlacement::Layers)
        .then(|| glm::attention_owners(&cfg, cfg.layers)).transpose()?;
    let whole_mla = owners.as_ref().map(|owners| owners.iter().enumerate()
        .map(|(layer, &rank)| (layer, usize::from(rank))).collect()).unwrap_or_default();
    let mut scratch = glm::manifest_scratch(&cfg, split, &manifest)?;
    if owners.is_some() {
        for (split, whole) in scratch.iter_mut().zip(glm::manifest_scratch(&cfg, false, &manifest)?) {
            split.programs = split.programs.max(whole.programs);
        }
    }
    let context_programs = if attention == crate::placement::AttentionPlacement::Context {
        let sms = options.physical_sms.unwrap_or_else(|| crate::placement::ArchContext::for_device(
            "sm_120", options.rtx_bytes[0]).sms) as usize;
        Some(glm::manifest_context_for_cards(&cfg, &manifest, &[sms], plan)?
            .ok_or_else(|| crate::placement::PlacementError::MissingContextPrograms { family: cfg.family_id(),
                programs: glm::context_program_list(&cfg) })?)
    } else { None };
    let records = DenseRecord::inventory(&checkpoint.snapshot)?;
    let selection = DenseSelection::from_hf(&checkpoint.config)?;
    let backbone = selection.backbone(&records, cfg.layers)?;
    let vocabulary = records.iter().filter_map(|record| match selection.role(&record.tensor, cfg.layers) {
        Ok(DenseRole::Vocabulary(_)) => Some(Ok(record.clone())),
        Ok(_) => None, Err(error) => Some(Err(error)),
    }).collect::<anyhow::Result<Vec<_>>>()?;
    let dense = (0..gpu_count).map(|rank| DensePlan::for_rank_owned(&backbone, gpu_count, rank, &whole_mla))
        .collect::<Result<Vec<_>, _>>()?;
    let vocabulary = (0..gpu_count).map(|rank| DensePlan::for_rank(&vocabulary, gpu_count, rank))
        .collect::<Result<Vec<_>, _>>()?;
    let shared = (0..gpu_count).map(|rank| (cfg.first_moe_layer..cfg.layers)
        .map(|layer| metadata.plan(layer, gpu_count, rank, ExpertSet::Shared))
        .collect::<Result<Vec<_>, _>>()).collect::<Result<Vec<_>, _>>()?;
    let native_scratch = dense.iter().map(|plan| crate::families::glm5::native::scratch_layouts(
        &cfg, &backbone, plan, glm::DECODE_ROWS as usize, rows as usize,
        options.exlr8_prefill_activations == crate::plan::exlr8::PrefillActivations::A8))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let workspace = crate::plan::exlr8::shared_execution_workspace(&manifest, cuteafd_ffi::exlr8::KernelFamily::Glm)?;
    let target = TargetFacts::native(&selection, cfg.layers, &dense, &vocabulary,
        &shared, &native_scratch, &workspace, LocalStorage::for_snapshot(&checkpoint.snapshot)?)?;
    let lanes = options.prefill_waves;
    let prefill = PrefillShape { rows, waves: lanes };
    let onboard = options.resolve_onboard(cfg.layers.saturating_sub(cfg.first_moe_layer)).unwrap_or_else(glm::default_onboard);
    let local = if onboard.layers(cfg.layers.saturating_sub(cfg.first_moe_layer)) == Some(0) { None }
        else {
            let manifest_path = options.workspace_manifest.as_deref()
                .unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
            Some(glm::exlr8_inventory_with_metadata(&checkpoint.snapshot, manifest_path, cfg.layers,
                gpu_count, rows, lanes, glm::local_exchange_f32()?, options.exlr8_prefill_activations, metadata)?)
        };
    let physical = LocalPhysicalInput { layers: (cfg.first_moe_layer..cfg.layers).collect(),
        max_rows: rows.max(glm::DECODE_ROWS) as usize, lanes: lanes as usize, storage: LocalStorage::for_snapshot(&checkpoint.snapshot)?,
        route_checks: true, peer_diagnostics: false, activations: options.exlr8_prefill_activations };
    let source = local.as_ref().map(|local| {
        let catalog = crate::read_expert_catalog(&checkpoint.snapshot)?;
        LocalPhysicalSource::read(&cfg, &catalog, local, &physical, Some(metadata))
    }).transpose()?;
    let page_tokens = if attention == crate::placement::AttentionPlacement::Context { 2 * glm::PAGE_ROWS } else { glm::PAGE_ROWS };
    let maximum_pool_tokens = crate::placement::PoolPolicy::resolve(
        &options.rtx_bytes, context, options.pool_tokens, page_tokens).maximum_tokens();
    let maximum_pages = maximum_pool_tokens.div_ceil(page_tokens);
    let (draft, mut additional, prefix_quota) = production_serving(options, checkpoint, &cfg,
        context, attention, gpu_count, maximum_pages)?;
    let largest_native = records.iter().flat_map(|record| &record.components)
        .map(|component| component.source.bytes as u64).max().unwrap_or(0);
    if let Some(envelope) = push_host_envelope(checkpoint, &cfg, cfg.layers, gpu_count, None,
        Some(metadata), largest_native)? {
        additional.extend(envelope.allocations("glm.load.push")?.into_iter().map(|owner| {
            let bytes = ByteCurve::fixed(owner.bytes, maximum_pages);
            Component::allocation(owner, bytes, false, maximum_pages)
        }));
    }
    let geometry = cuteafd_core::ExpertGeometry { hidden: cfg.hidden as u32, experts: cfg.experts as u32,
        topk: cfg.topk as u32, intermediate: cfg.moe_intermediate as u32, layers: cfg.layers as u32 };
    let rings = cuteafd_core::SparkRings { max_frame_bytes: geometry.native_frame_budget(
        u32::try_from(rows.max(glm::DECODE_ROWS))?).context("EXLR8 native frame geometry")?,
        depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
    let (decode_link, prefill_link) = production_links(&cfg, spark_ranks, prefill,
        cuteafd_core::SparkIngress::Fp8K32, rings, options.spark_reply_formats)?;
    let host = options.host_memory.map(Ok).unwrap_or_else(crate::formats::mapped_preload::host_memory_before_process)?;
    let budgets = production_budgets(&options.rtx_bytes.iter().map(|&n| (n, n)).collect::<Vec<_>>(),
        crate::placement::reserve::gpu("glm5"), host, (8 << 30).max(host.0 / 10), &vec![options.spark_bytes; spark_ranks])?;
    let operation = if options.full_prefill_logits { Operation::Scoring } else { Operation::Serving };
    let request = production_request(crate::placement::planner::Context::resolve(context, (options.context_tokens > 0).then_some(options.context_tokens)), options.pool_tokens, onboard, prefill, operation,
        crate::placement::ResidentWeights(options.weight_budget_bytes), options.spark_reply_formats);
    let admitted = ProductionInput { expert_usage: options.expert_usage, cfg: &cfg, layers: cfg.layers, context, maximum_pool_tokens,
        attention, gpus: gpu_count, prefill, blas_head: serving::BlasHead::Native, blas: vec![], target, draft, scratch, context_scratch: context_programs,
        exchange_element_bytes: if glm::local_exchange_f32()? { 4 } else { 2 }, peer_diagnostics: false,
        prefix_host: crate::families::glm5::host_prefix_enabled(gpu_count, attention,
            options.host_cache_bytes != Some(0), options.mimo_prefix_entries),
        local: source.as_ref().map(|source| (source, physical)), spark_ranks, decode_link, prefill_link,
        skip_routed_experts: false, operation, sync_upload_bytes: 0, owners: additional, budgets,
    }.prepare()?.admit(request)?;
    if let Some(report) = report {
        report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&admitted.solved));
        report.host_budget_bytes = prefix_quota;
    }
    let mut notes = admitted.load_overlap_note().into_iter().collect::<Vec<_>>();
    if let Some(note) = figure_recommendation(&admitted, options, checkpoint, Some(&manifest))? { notes.push(note); }
    let placement = admitted.placement;
    notes.push(format!("EXLR8 native {gpu_count} shared owners; {attention} attention; context {context}; {}", placement.summary()));
    let devices = placement.items.iter().enumerate().map(|(rank, items)| DeviceLayout {
        kind: DeviceKind::Rtx, index: rank as u32,
        capacity_bytes: options.rtx_bytes[rank].saturating_sub(crate::placement::reserve::gpu("glm5")),
        worker_budget_bytes: 0, items: items.clone(), kv_tokens: placement.pool_tokens,
    }).collect();
    Ok((MemoryLayout { devices, pool_tokens: placement.pool_tokens, waste: vec![],
        notes }, placement))
}

/// The planner's packed owners, excluding a host-mapped embedding when selected.
pub fn device_bytes(checkpoint: &Checkpoint, ranks: usize, layers: usize, host_embedding: bool, owners: Option<&[u8]>)
    -> anyhow::Result<Vec<u64>> {
    resident_owned(checkpoint, ranks, layers, owners)?.into_iter().map(|groups| groups.into_iter()
        .filter(|(group, _, _)| !host_embedding || group != "embedding")
        .try_fold(0u64, |sum, (_, _, bytes)| sum.checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("GLM retained weight bytes overflow")))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{fp8, glm5_config, glm5_tensors, write_snapshot};

    fn native_manifest() -> serde_json::Value { crate::plan::testing::glm5_native_programs() }

    fn native_fixture_options(gpus: usize) -> LayoutOptions {
        LayoutOptions { rtx_bytes: vec![96 << 30; gpus], context_tokens: 65536,
            pool_tokens: Some(131072), prefill_rows: 1024, prefill_waves: 3,
            onboard: Some(crate::placement::OnboardSetting::Layers(0)),
            glm5_drafter_disabled: true, host_memory: Some((128 << 30, 128 << 30)),
            host_cache_bytes: Some(0), ..Default::default() }
    }

    #[test]
    fn native_admission_charges_header_owned_weights_and_hook_allocations() {
        use crate::placement::{AttentionPlacement, planner::{Location, Phase}};
        use crate::plan::exlr8::{dense::{DensePlan, DenseRecord, DenseSelection}, residency::{MetadataInput, ExpertSet}};
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::exlr8::glm_admission_snapshot(dir.path());
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config).unwrap();
        let manifest = crate::plan::testing::exlr8::glm_admission_programs();
        let metadata = MetadataInput::open(dir.path(), "k4").unwrap();
        let records = DenseRecord::inventory(dir.path()).unwrap();
        let selection = DenseSelection::from_hf(&checkpoint.config).unwrap();
        for gpus in 1..=2 {
            for attention in [AttentionPlacement::Heads, AttentionPlacement::Context, AttentionPlacement::Layers] {
                if gpus == 1 && attention != AttentionPlacement::Heads { continue; }
                let options = native_fixture_options(gpus);
                let input = crate::plan::PlanInput::open(dir.path()).unwrap();
                let mut report = crate::plan::empty_report(&input, &crate::plan::PlanOptions::default()).unwrap();
                let (layout, placement) = native_layout_with_metadata(&checkpoint, &options, 4, &manifest,
                    "k4", Some(&metadata), Some(attention), Some(&mut report)).unwrap();
                // Scoring runs a prefill before Ready, so its lane owners are
                // resident at Ready. Serving allocates them on the first live
                // prefill and charges them post-Ready; the two views together
                // hold every prefill owner exactly once.
                let scoring_options = LayoutOptions { full_prefill_logits: true, ..options.clone() };
                let (scoring, _) = native_layout_with_metadata(&checkpoint, &scoring_options, 4, &manifest,
                    "k4", Some(&metadata), Some(attention), None).unwrap();
                assert_eq!(layout.pool_tokens, 131072);
                assert_eq!(placement.onboard_layers, 0);
                assert_eq!(placement.attention_placement, attention);
                assert!(placement.max_pool_tokens_no_onboard.is_none());
                let owners = if attention == AttentionPlacement::Layers {
                    crate::placement::families::glm5::attention_owners(&cfg, cfg.layers).unwrap()
                        .into_iter().enumerate().map(|(layer, rank)| (layer, usize::from(rank))).collect()
                } else { BTreeMap::new() };
                let workspace = crate::plan::exlr8::shared_execution_workspace(&manifest,
                    cuteafd_ffi::exlr8::KernelFamily::Glm).unwrap();
                for (rank, device) in layout.devices.iter().enumerate() {
                    let physical = DensePlan::for_rank_owned(&records, gpus, rank, &owners).unwrap();
                    let shared = metadata.plan(1, gpus, rank, ExpertSet::Shared).unwrap();
                    // TokenEmbedding::row_fp8 retains its 256-byte table placeholder on rank 0.
                    // The RoPE table is a weight-kind owner on every rank.
                    let expected = physical.resident_bytes as u64 + shared.resident_bytes as u64
                        + if rank == 0 { 256 } else { 0 }
                        + options.context_tokens * cfg.qk_rope_head_dim as u64 * 4;
                    assert_eq!(device.items.iter().filter(|item| matches!(item.category, Category::Weights | Category::Embedding)
                        && item.group != crate::placement::planner::ALLOCATION_GRANULARITY_GROUP)
                        .map(|item| item.bytes).sum::<u64>(), expected);
                    let backbone = selection.backbone(&records, cfg.layers).unwrap();
                    let dense = DensePlan::for_rank_owned(&backbone, gpus, rank, &owners).unwrap();
                    let scratch = crate::families::glm5::native::scratch_layouts(&cfg, &backbone, &dense,
                        64, 1024, false).unwrap();
                    let group = |layout: &MemoryLayout, group: &str| layout.devices[rank].items.iter()
                        .filter(|item| item.group == group).map(|item| item.bytes).sum::<u64>();
                    let hooks = group(&layout, "GLM target hooks");
                    let scoring_hooks = group(&scoring, "GLM target hooks");
                    let decode = u64::try_from(scratch[0].bytes().unwrap()).unwrap();
                    assert_eq!(hooks, decode);
                    assert_eq!(scoring_hooks, crate::families::glm5::native::scratch_bytes(&scratch).unwrap());
                    assert_eq!(scoring_hooks - hooks,
                        u64::try_from(scratch[1].bytes().unwrap()).unwrap());
                    let shared_arenas = group(&layout, "EXLR8 expert workspace");
                    let scoring_arenas = group(&scoring, "EXLR8 expert workspace");
                    assert_eq!(shared_arenas, workspace.bytes as u64);
                    assert_eq!(scoring_arenas, 2 * workspace.bytes as u64);
                    assert_eq!(scoring_arenas - shared_arenas, workspace.bytes as u64);
                    assert_eq!(device.items, placement.items[rank]);
                    assert!(device.used_bytes() <= device.capacity_bytes);
                }
                let ledger = report.phase_ledger.unwrap();
                let host = ledger.budgets.iter().find(|budget| budget.id == "coordinator host RAM").unwrap();
                assert_eq!(host.locations, [Location::HOST, Location::PINNED]);
                assert!(ledger.phases.iter().any(|row| row.budget == host.id && row.phase == Phase::Load && row.owned > 0));
                assert!(ledger.phases.iter().all(|row| row.owned + row.opaque <= row.available));
            }
        }
    }

    #[test]
    fn native_metadata_admission_does_not_reopen_full_forward_catalog() {
        use crate::plan::exlr8::residency::MetadataInput;
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::exlr8::glm_admission_snapshot(dir.path());
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config).unwrap();
        assert_eq!(cfg.layers, 2);
        // This fixture deliberately cannot enter the shipping full-forward catalog.
        let error = crate::read_expert_catalog(dir.path()).unwrap_err();
        assert!(error.to_string().contains("78 layers/hidden6144/first routed3"), "{error:#}");
        let metadata = MetadataInput::open(dir.path(), "k4").unwrap();
        let manifest = crate::plan::testing::exlr8::glm_admission_programs();
        for gpus in 1..=2 {
            let options = native_fixture_options(gpus);
            let (layout, _) = native_layout_with_metadata(&checkpoint, &options, 4, &manifest,
                "k4", Some(&metadata), None, None).unwrap();
            assert_eq!(layout.devices.len(), gpus);
            assert_eq!(layout.pool_tokens, options.pool_tokens.unwrap());
        }
    }

    #[test]
    fn native_admission_requires_selected_metadata_and_context_programs() {
        use crate::placement::{AttentionPlacement, PlacementError};
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::exlr8::glm_admission_snapshot(dir.path());
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let mut manifest = crate::plan::testing::exlr8::glm_admission_programs();
        let options = native_fixture_options(2);
        assert!(native_layout(&checkpoint, &options, 4, &manifest, "missing").is_err());
        manifest["programs"].as_array_mut().unwrap().retain(|p|
            !p["name"].as_str().unwrap().contains("_partial_h") && p["name"] != "glm_dsa_candidate_merge");
        native_layout_with_metadata(&checkpoint, &options, 4, &manifest, "k4", None,
            Some(AttentionPlacement::Heads), None).unwrap();
        let error = native_layout(&checkpoint, &options, 4, &manifest, "k4").unwrap_err();
        assert!(matches!(error.downcast_ref::<PlacementError>(), Some(PlacementError::MissingContextPrograms { .. })));
    }

    #[test]
    fn auto_recipe_mismatch_charges_the_same_owned_operands_as_explicit_layers() {
        use crate::plan::{PlanInput, PlanOptions, ExpertPlacement};
        use crate::placement::AttentionPlacement;
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None)), None);
        let input = PlanInput::open(dir.path()).unwrap();
        let mut manifest = native_manifest();
        let programs = manifest["programs"].as_array_mut().unwrap();
        let partials: Vec<_> = programs.iter().filter(|p| p["name"].as_str().unwrap()
            .starts_with("glm_sparse_mla_partial_")).cloned().collect();
        programs.retain(|p| !p["name"].as_str().unwrap().starts_with("glm_sparse_mla_partial_"));
        assert!(!partials.is_empty());
        for partial in partials {
            for splits in [111, 222] {
                let mut partial = partial.clone();
                let begin = if partial["name"].as_str().unwrap().contains("_h0_") { 0 } else { 32 };
                partial["name"] = serde_json::json!(format!("glm_sparse_mla_partial_h{begin}_s{splits}_m64"));
                partial["params"] = serde_json::json!({"device_sm_counts": [170], "row_caps": {"170": [64]}});
                programs.push(partial);
            }
        }
        let path = dir.path().join("PROGRAMS.json");
        std::fs::write(&path, manifest.to_string()).unwrap();
        let options = LayoutOptions { rtx_bytes: vec![96 << 30; 2], physical_sms: Some(188),
            workspace_manifest: Some(path), context_tokens: 65536, pool_tokens: Some(131072),
            onboard: Some(crate::placement::OnboardSetting::Layers(0)), glm5_drafter_disabled: true,
            host_memory: Some((128 << 30, 128 << 30)), host_cache_bytes: Some(0), ..Default::default() };
        let plan = |attention| input.plan(&PlanOptions { placement: ExpertPlacement::Sparks { ranks: 6 },
            attention_placement: attention, layout: Some(options.clone()), ..Default::default() }).unwrap();
        let auto = plan(None);
        let explicit = plan(Some(AttentionPlacement::Layers));
        assert!(auto.placement_supported, "{:?}", auto.hints);
        assert_eq!(auto.attention_placement, AttentionPlacement::Layers);
        assert_eq!(auto.solver_placement, explicit.solver_placement);
        let auto_memory = auto.memory_layout.unwrap();
        let explicit_memory = explicit.memory_layout.unwrap();
        assert_eq!(auto_memory.devices, explicit_memory.devices);
        assert!(auto_memory.notes.iter().any(|note| note.starts_with("auto attention fallback: layers")));
    }

    /// GLM 5.3's DFlash2 drafter (an unpaged ring) carries no prefix marks or draft pages: its
    /// charge is today's for any prefix retention, so the window-mark wiring is a no-op for it.
    /// Mistral Large 4's window-ring drafter adds exactly its marks.
    #[test]
    fn prefix_draft_marks_charge_only_window_ring_drafters() {
        let glm = tempfile::tempdir().unwrap();
        write_snapshot(glm.path(), &glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None)), None);
        let drafter = tempfile::tempdir().unwrap();
        std::fs::write(drafter.path().join("config.json"), serde_json::json!({"hidden_size":6144,
            "intermediate_size":12288,"num_hidden_layers":5,"num_attention_heads":32,"num_key_value_heads":8,
            "head_dim":128,"vocab_size":154880,"dflash_config":{"target_layer_ids":[0,8,16,24,32],
            "conv_group_size":16,"selector_rank":256,"block_size":16}}).to_string()).unwrap();
        let checkpoint = Checkpoint::coordinator(glm.path(), false, false).unwrap();
        let options = |entries: u64, budget: u64| LayoutOptions { rtx_bytes: vec![96 << 30],
            glm5_drafter_snapshot: Some(drafter.path().to_path_buf()), mimo_prefix_entries: entries,
            mimo_prefix_mark_bytes: budget, ..Default::default() };
        let base = drafter_admission(&options(0, 0), &checkpoint).unwrap();
        for (entries, budget) in [(20, 2 << 30), (64, 8 << 30)] {
            assert_eq!(drafter_admission(&options(entries, budget), &checkpoint).unwrap(), base);
        }
        let ml4 = tempfile::tempdir().unwrap();
        crate::plan::testing::write_mistral_large4_snapshot(ml4.path(), Some(&[1]), true);
        let checkpoint = Checkpoint::coordinator(ml4.path(), false, false).unwrap();
        let mut off = options(0, 2 << 30);
        off.glm5_drafter_snapshot = None;
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config).unwrap();
        off.host_memory = Some((128 << 30, 128 << 30));
        let mark_owner = |options: &LayoutOptions, attention, gpus| {
            let (draft, _, _) = production_serving(options, &checkpoint, &cfg, 1 << 20, attention, gpus, 32768).unwrap();
            draft.owners().iter().filter(|owner| owner.id == "glm.draft.window_marks")
                .map(|owner| owner.bytes).sum::<u64>()
        };
        let cold = mark_owner(&off, crate::placement::AttentionPlacement::Heads, 1);
        let mut on = off.clone();
        on.mimo_prefix_entries = 20;
        on.prefix_slots = Some(7);
        let warm = mark_owner(&on, crate::placement::AttentionPlacement::Heads, 1);
        let layout = crate::placement::families::glm5::ml4_draft_layout(ml4.path(),
            crate::families::glm5::draft_representation::serving_draft_context_slots(
                on.glm_draft_sequences() as usize, on.resolved_concurrency("mistral_large4") as usize, None),
            on.glm_draft_sequences() as usize, 1 << 20, 188).unwrap();
        let mark = crate::placement::families::glm5::draft_window_mark_bytes(&layout).unwrap() as u64;
        assert_eq!(mark, 16 + layout.window_mark_bytes().unwrap() as u64);
        assert_eq!(warm - cold, 7 * mark);
        on.prefix_slots = None;
        let device = mark_owner(&on, crate::placement::AttentionPlacement::Heads, 2);
        let host = mark_owner(&on, crate::placement::AttentionPlacement::Layers, 2);
        let slots = cuteafd_core::prefix::mark_slots_with_host(on.resolved_concurrency("mistral_large4"),
            on.mimo_prefix_entries, mark, on.mimo_prefix_mark_bytes, false);
        assert_eq!(device, slots * mark);
        assert!(device > host);

    }

    #[test]
    fn exlr8_explicit_stages_preserve_the_pre_namespace_fix_module_charge() {
        let mut manifest = native_manifest();
        let entries = manifest["programs"].as_array_mut().unwrap();
        // Match the sealed p0 namespace census without embedding a local build.
        for (family, count) in [("glm", 42), ("glm2", 18), ("exlr8", 84)] {
            let actual = entries.iter().filter(|p|
                cuteafd_core::coordinator_programs::program_namespace(p["name"].as_str().unwrap()) == Some(family)).count();
            assert!(actual <= count);
            entries.extend((actual..count).map(|i| serde_json::json!({"name":format!("{family}_fixture_{i}")})));
        }
        for prefix in ["glm", "glm2"] {
            for rows in [64, 4096] {
                for stage in ["rank_pack", "query_rope", "index_post", "absorb_bf16", "absorb", "values_bf16", "values"]
                    .into_iter().chain((prefix == "glm2").then_some("query_rope_context"))
                    .chain((prefix == "glm2").then_some("absorb_context")) {
                    entries.push(serde_json::json!({"name":format!("{prefix}_stage_{stage}_m{rows}")}));
                }
            }
        }
        assert_eq!(entries.len(), 176);
        let mut expanded = manifest.clone();
        let entries = expanded["programs"].as_array_mut().unwrap();
        entries.extend((0..42).map(|i| serde_json::json!({"name":format!("dense_trellis_k6_fixture_{i}")})));
        let k6_only = ["glm2_stage_absorb_bf16_context_m64", "glm2_stage_absorb_bf16_context_m4096",
            "glm_stage_swiglu_i2048", "glm_stage_swiglu_i12288", "glm2_stage_swiglu_i1024", "glm2_stage_swiglu_i6144"];
        entries.extend(k6_only.iter().map(|name| serde_json::json!({"name":name,
            "scratch_bytes_at_capacity":{"scratch":999_999}})));
        assert_eq!(entries.len(), 224);
        use crate::families::glm5::DensePrograms;
        // Runtime intentionally retains p0's split module set even on one GPU.
        let old_runtime = DensePrograms::Exlr8.program_set(&manifest, true);
        let runtime = DensePrograms::Exlr8.program_set(&expanded, true);
        assert_eq!(runtime, old_runtime);
        assert_eq!(runtime.names.len(), 176);
        assert_eq!(runtime.module_bytes(), 56_623_104);
        let broad = crate::placement::ProgramSet::from_manifest(&expanded,
            &["glm", "glm2", "exlr8", "glm_stage", "glm2_stage"]);
        assert_eq!(broad.names.len(), 182);
        assert_eq!(broad.module_bytes() - runtime.module_bytes(), 2 << 20);
        assert!(runtime.scratch.iter().all(|(_, bytes)| *bytes != 999_999));
        assert!(k6_only.iter().all(|name| DensePrograms::Trellis.stage(name, true)));
        assert!(k6_only.iter().all(|name| !DensePrograms::Exlr8.stage(name, true)));
        assert!(DensePrograms::Fp8.program_set(&expanded, true).names.iter().all(|name| !name.contains("_stage_")));
        for gpus in [1, 2] {
            let old_set = DensePrograms::Exlr8.program_set(&manifest, gpus == 2);
            let new_set = DensePrograms::Exlr8.program_set(&expanded, gpus == 2);
            assert_eq!(new_set, old_set);
            assert_eq!(new_set.names.len(), if gpus == 2 { 176 } else { 140 });

        }
    }

    #[test]
    fn native_a8_records_are_one_physical_scratch_owner_per_gpu() {
        use crate::plan::exlr8::{dense::{DenseRecord, DensePlan}, PrefillActivations};
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::exlr8::glm_admission_snapshot(dir.path());
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config).unwrap();
        let manifest = crate::plan::testing::exlr8::glm_admission_programs();
        let records = DenseRecord::inventory(dir.path()).unwrap();
        for gpus in 1..=2 {
            let options = native_fixture_options(gpus);
            let input = crate::plan::PlanInput::open(dir.path()).unwrap();
            let plan = |options: &LayoutOptions| {
                let mut report = crate::plan::empty_report(&input, &crate::plan::PlanOptions::default()).unwrap();
                let (layout, _) = native_layout_with_metadata(&checkpoint, options, 4, &manifest, "k4",
                    None, None, Some(&mut report)).unwrap();
                (layout.pool_tokens, report.phase_ledger.unwrap())
            };
            let (pool16, a16) = plan(&options);
            let (pool8, a8) = plan(&LayoutOptions {
                exlr8_prefill_activations: PrefillActivations::A8, ..options });
            // Serving allocates its prefill lanes on the first live request, so
            // the A8 records appear in the post-ready growth row, once per GPU.
            let growth = |ledger: &crate::plan::SolvedPhaseLedger, rank: usize| {
                let budget = ledger.budgets.iter().find(|budget| budget.locations.contains(
                    &crate::placement::planner::Location::coordinator(rank as u8))).unwrap();
                let row = ledger.phases.iter().find(|row| row.budget == budget.id
                    && row.phase == crate::placement::planner::Phase::Growth).unwrap();
                row.owned + row.opaque
            };
            for rank in 0..gpus {
                let plan = DensePlan::for_rank(&records, gpus, rank).unwrap();
                let scratch = crate::families::glm5::native::scratch_layouts(&cfg, &records, &plan, 64, 1024, true).unwrap();
                let records = scratch[1].linear.records.unwrap() as u64;
                assert!(records > 0);
                assert_eq!(growth(&a8, rank) - growth(&a16, rank), records);
            }
            assert_eq!(pool16, pool8);
        }
    }

    #[test]
    fn native_admission_uses_joint_host_budget_and_strict_pool_pins() {
        use crate::placement::planner::{Phase, Refusal};
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::exlr8::glm_admission_snapshot(dir.path());
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let manifest = crate::plan::testing::exlr8::glm_admission_programs();
        for gpus in 1..=2 {
            let options = native_fixture_options(gpus);
            let plan = |options: &LayoutOptions| {
                let input = crate::plan::PlanInput::open(dir.path()).unwrap();
                let mut report = crate::plan::empty_report(&input, &crate::plan::PlanOptions::default()).unwrap();
                native_layout_with_metadata(&checkpoint, options, 4, &manifest, "k4", None, None, Some(&mut report))?;
                Ok::<_, anyhow::Error>(report.phase_ledger.unwrap())
            };
            let base = plan(&options).unwrap();
            let peak = base.phases.iter().filter(|row| row.budget == "coordinator host RAM")
                .map(|row| row.owned + row.opaque).max().unwrap();
            let total = options.host_memory.unwrap().0;
            let free = peak + (8 << 30).max(total / 10);
            plan(&LayoutOptions { host_memory: Some((total, free)), ..options.clone() }).unwrap();
            let error = plan(&LayoutOptions { host_memory: Some((total, free - 1)), ..options.clone() }).unwrap_err();
            assert!(matches!(error.downcast_ref::<Refusal>(), Some(Refusal::Capacity { budget, .. })
                if budget == "coordinator host RAM"), "{error:#}");
            let quota = 1 << 30;
            let prefix = plan(&LayoutOptions { host_cache_bytes: Some(quota), ..options.clone() }).unwrap();
            for phase in [Phase::Ready, Phase::Growth] {
                let used = |ledger: &crate::plan::SolvedPhaseLedger| ledger.phases.iter().find(|row|
                    row.budget == "coordinator host RAM" && row.phase == phase).unwrap().owned;
                assert_eq!(used(&prefix) - used(&base), quota);
            }
            assert!(plan(&LayoutOptions { pool_tokens: Some(32768), ..options.clone() }).is_err());
            let error = plan(&LayoutOptions { host_memory: Some((12 << 30, 12 << 30)),
                host_cache_bytes: Some(8 << 30), ..options }).unwrap_err();
            assert!(matches!(error.downcast_ref::<Refusal>(), Some(Refusal::Capacity { .. })), "{error:#}");
        }
    }

    #[test]
    fn resident_headers_follow_rank_ownership_and_absorbed_kv_packing() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None)), None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let layers = GlmDsaConfig::from_hf(&checkpoint.config).unwrap().layers;
        let one = resident_owned(&checkpoint, 1, layers, None).unwrap();
        let two = resident_owned(&checkpoint, 2, layers, None).unwrap();
        let bytes = |rank: &Vec<(String, String, u64)>, group: &str| rank.iter()
            .filter(|(g, _, _)| g == group).map(|(_, _, b)| *b).sum::<u64>();
        assert_eq!(bytes(&one[0], "router"), bytes(&two[0], "router"));
        assert_eq!(bytes(&two[1], "router"), 0);
        for group in ["embedding", "lm_head"] {
            assert_eq!(bytes(&one[0], group), bytes(&two[0], group));
            assert_eq!(bytes(&two[1], group), 0);
        }
        assert_eq!(bytes(&two[0], "indexer"), bytes(&one[0], "indexer"));
        assert_eq!(bytes(&two[1], "indexer"), bytes(&one[0], "indexer"));
        for group in ["dense", "shared_expert"] {
            assert_eq!(bytes(&two[0], group) + bytes(&two[1], group), bytes(&one[0], group));
        }
        // uk/uv keep per-64K scales; qkv keeps both block and per-row K-major grids.
        let scales = |rank: &Vec<(String, String, u64)>| rank.iter()
            .filter(|(g, f, _)| g == "attention" && f == "fp8-scale").map(|(_, _, b)| *b).sum::<u64>();
        assert!(scales(&two[0]) > 2 * 64 * 512 * (192 + 256) / 2 / 64 * 4);
        assert_eq!(scales(&two[0]), scales(&two[1]));
    }

    #[test]
    fn retained_headers_respect_selected_extent_and_host_embedding() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None)), None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let layers = GlmDsaConfig::from_hf(&checkpoint.config).unwrap().layers;
        for ranks in [1, 2] {
            let globals = device_bytes(&checkpoint, ranks, 0, true, None).unwrap();
            assert!(globals[0] > 0);
            assert!(globals[1..].iter().all(|&bytes| bytes == 0));
            let mut previous = globals;
            for extent in 1..=layers {
                let gpu = device_bytes(&checkpoint, ranks, extent, false, None).unwrap();
                let host = device_bytes(&checkpoint, ranks, extent, true, None).unwrap();
                let embedding = checkpoint.tensors.iter().find(|t| t.meta.name == "model.embed_tokens.weight").unwrap();
                assert_eq!(gpu[0] - host[0], embedding.meta.byte_length);
                assert_eq!(gpu[1..], host[1..]);
                assert!(host.iter().zip(&previous).all(|(next, before)| next > before));
                previous = host;
            }
            assert!(resident_owned(&checkpoint, ranks, layers + 1, None).is_err());
        }
    }
    #[test]
    fn owned_attention_residency_is_not_replicated_and_ffns_remain_split() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None)), None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let layers = GlmDsaConfig::from_hf(&checkpoint.config).unwrap().layers;
        let one = resident_owned(&checkpoint, 1, layers, None).unwrap();
        let heads = resident_owned(&checkpoint, 2, layers, None).unwrap();
        let owners = crate::placement::families::glm5::attention_owners(
            &GlmDsaConfig::from_hf(&checkpoint.config).unwrap(), 2).unwrap();
        let owned = resident_owned(&checkpoint, 2, layers, Some(&owners)).unwrap();
        let bytes = |rank: &Vec<(String, String, u64)>, group: &str| rank.iter()
            .filter(|(g, _, _)| g == group).map(|(_, _, b)| *b).sum::<u64>();
        for group in ["attention", "indexer"] {
            assert_eq!(owned.iter().map(|rank| bytes(rank, group)).sum::<u64>(), bytes(&one[0], group));
        }
        for rank in 0..2 {
            for group in ["dense", "shared_expert", "router", "lm_head", "embedding"] {
                assert_eq!(bytes(&owned[rank], group), bytes(&heads[rank], group));
            }
        }
        assert!(resident_owned(&checkpoint, 2, layers, Some(&[])).is_err());
    }

}

#[cfg(test)]
mod resident_golden {
    use super::*;
    use crate::plan::testing::{fp8, glm5_config, glm5_tensors, write_snapshot};

    /// Every (rank, group, format, bytes) of the GLM fixture at one and two
    /// ranks and owned attention, pinned so a refactor of the accounting is
    /// byte-for-byte neutral for GLM 5.x.
    pub(super) fn glm_resident_fingerprint() -> Vec<String> {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None)), None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config).unwrap();
        let owners = crate::placement::families::glm5::attention_owners(&cfg, 2).unwrap();
        let mut out = Vec::new();
        for (label, ranks, owned) in [("tp1", 1, false), ("tp2", 2, false), ("owned", 2, true)] {
            let resident = resident_owned(&checkpoint, ranks, cfg.layers, owned.then_some(owners.as_slice())).unwrap();
            for (rank, groups) in resident.iter().enumerate() {
                for (group, format, bytes) in groups {
                    out.push(format!("{label} {rank} {group} {format} {bytes}"));
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "prints the fingerprint; run on a base commit to regenerate"]
    fn print_glm_resident_fingerprint() {
        for line in glm_resident_fingerprint() { println!("FP {line}"); }
    }
}
