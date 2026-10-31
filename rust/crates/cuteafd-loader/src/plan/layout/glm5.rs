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
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 };
    let layers = report.requested_attention == Some(crate::placement::AttentionPlacement::Layers);
    let limits = crate::serving_capacity::prefill_waves::PrefillLimits {
        max_waves: if spark_ranks == 0 { 1 }
            else if layers { crate::placement::LAYER_PIPELINE_MAX_LANES } else { 4 },
        ..crate::plan::Family::prefill_limits(&crate::plan::families::glm::GLM5)
    };
    let prefill = match super::resolve_prefill(report, options, spark_ranks, limits) {
        Ok(shape) => shape,
        Err(error) => return unsupported_prefill(report, error),
    };
    let mut shape = Shape::resolve(report, model, options, ShapePolicy::manifest_bounded());
    report.prefill_span_rows = Some(prefill.span_rows);
    let gpus = options.rtx_bytes.len().clamp(1, 2);
    let split = gpus == 2 && options.head_split;
    let ranks = if split { 2 } else { 1 };
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: bytes.saturating_sub(options.headroom_bytes), items: Vec::new(), kv_tokens: 0 }).collect();
    let mut waste = Vec::new();
    let mut notes = vec![format!("resolved context {} tokens (checkpoint {})", shape.context_tokens, shape.checkpoint_context),
        format!("attention placement {}", report.attention_placement)];
    shape.select_card_plans(report, options, &mut notes);
    mapped_table_note(report, &mut notes);
    let resident = (|| -> anyhow::Result<_> {
        let layers = model.spec().layers.len();
        let owners = (report.requested_attention == Some(crate::placement::AttentionPlacement::Layers))
            .then(|| {
                let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
                crate::placement::families::glm5::attention_owners(&cfg, layers).map_err(anyhow::Error::from)
            }).transpose()?;
        resident_owned(checkpoint, ranks, layers, owners.as_deref())
    })();
    match resident {
        Ok(resident) => push_resident(&mut devices, &resident, |group| match group {
            "embedding" => Category::Embedding,
            _ => Category::Weights,
        }),
        Err(error) => {
            report.placement_supported = false;
            let what = format!("GLM resident operands: {error:#}");
            report.hints.push(crate::plan::Hint { what: what.clone(), how: "Supply the checkpoint's complete coordinator tensor headers.".into() });
            notes.push(what);
        }
    }
    if options.host_embedding {
        place_host_embedding(report, checkpoint, &mut devices, &mut waste, &mut notes);
    }
    let core = (|| -> anyhow::Result<_> {
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
        let core = crate::families::glm5::core::CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        if core.projections.is_empty() { return Ok(None); }
        let owners = layers.then(|| crate::placement::families::glm5::attention_owners(&cfg, cfg.layers)).transpose()?;
        let (_, _, _, draft_rows) = drafter_admission(options)?;
        anyhow::ensure!(options.glm5_drafter_disabled || draft_rows.is_some(),
            "native K6 drafter head needs its actual DFlash config, not a bytes-only override");
        Ok(Some(crate::families::glm5::core::CoreCharges::from_package(&cfg, &core, cfg.layers,
            ranks, owners.as_deref(), shape.prefill_rows as usize, prefill.waves as usize, draft_rows,
            shape.manifest.as_ref().context("GLM K6 requires PROGRAMS.json with native exports")?)?))
    })();
    for (index, device) in devices.iter_mut().take(ranks).enumerate() {
        let item = match (core.as_ref().ok().and_then(Option::as_ref), shape.manifest.as_ref()) {
            (Some(core), Some(manifest)) => {
                let arch = crate::placement::ArchContext::coordinator(options.rtx_bytes[index], None);
                let allowance = crate::placement::inventory::loaded_code_allowance("glm-k6", local_package(report), split,
                    index as u8, &arch, Some(&core.program_set(manifest, split)));
                Item::new(Category::Runtime, "context+modules", "", arch.context_bytes + allowance.bytes, allowance.basis)
            }
            _ => runtime_item(shape.manifest.as_ref(), &program_families("glm", split),
                local_package(report), split, index, options.rtx_bytes[index]),
        };
        device.items.push(item);
    }
    let lanes = prefill.waves;
    let mut working = crate::placement::SparkWorkingSet { rtx_layers: Vec::new(),
        spark_layers: if spark_ranks > 0 { (0..model.spec().layers.len()).collect() } else { Vec::new() },
        spark_ranks, prefill_lanes: lanes };
    let mut sparks = spark_devices(report, model, input, options, SPARK, &working,
        spark_endpoints("glm5", lanes), &mut waste, &mut notes);
    let mut pool_tokens = 0;
    let mut local_layers = 0;
    match core.and_then(|core| placement(report, input, options, &devices[..ranks], shape.manifest.as_ref(),
        shape.prefill_rows, shape.context_tokens, core.as_ref())) {
        Ok((placement, selected)) => {
            report.solver_placement = Some(placement.clone());
            // The solver resolved auto against this build's programs.
            report.attention_placement = placement.attention_placement;
            if let Some(note) = notes.iter_mut().find(|n| n.starts_with("attention placement ")) {
                *note = format!("attention placement {}", placement.attention_placement);
            }
            working = selected.spark.clone();
            if working.spark_layers.is_empty() { sparks.clear(); }
            notes.push(format!("GLM selected working set: {} Spark layers, {} ranks, {} prefill lanes",
                working.spark_layers.len(), working.spark_ranks, working.prefill_lanes));
            for (device, items) in devices.iter_mut().zip(&placement.items) {
                device.items.extend(items.iter().cloned());
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
            report.placement_supported = false;
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
            report.hints.push(crate::plan::Hint { what: what.clone(), how: failure_hint(report.attention_placement) });
            notes.push(what);
        }
    }
    if local_layers > 0 { notes.push(format!("{local_layers} routed backbone layers resident on RTX")); }
    report.prefill_transports = Some(u64::from(!working.spark_layers.is_empty()));
    // Reapply after admission has selected its executable suffix and lane count.
    charge_spark_devices(report, model, input, options, SPARK, &working, spark_endpoints("glm5", working.prefill_lanes),
        &mut sparks, &mut notes, |device| format!("{} I/O admission", device.name()));
    if report.encoder.is_none() {
        resolve_encoder(report, &Towers::checkpoint(report), &mut devices, &mut sparks, &vec![0; gpus], options, &mut notes);
    }
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens, waste, notes }
}

pub fn resident_owned(checkpoint: &Checkpoint, ranks: usize, layers: usize, owners: Option<&[u8]>)
    -> anyhow::Result<Vec<Vec<(String, String, u64)>>> {
    anyhow::ensure!((1..=2).contains(&ranks), "GLM coordinator ranks");
    let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
    let core = crate::families::glm5::core::CoreInventory::from_checkpoint(&cfg, checkpoint)?;
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
    let native = std::env::var("CUTEAFD_GLM_BF16").is_ok_and(|v| v == "native");
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
                    let index_bf16 = (native || trellis) && tensor(&name)?.dtype == DType::Bf16;
                    let (values, scales) = projection(&name)?;
                    put(rank, "indexer", if index_bf16 { "bf16" } else { "fp8" }, values * if index_bf16 { 2 } else { 1 });
                    if !index_bf16 { put(rank, "indexer", "fp8-scale", scales); }
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
    if let Some(alpha) = cfg.variant.ml4().map(|_| "lerope.alpha") {
        put(0, "norm", "bf16", tensor(alpha)?.byte_length);
    }
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

fn drafter_admission(options: &LayoutOptions) -> anyhow::Result<(u64, u64, Option<usize>, Option<usize>)> {
    use crate::placement::families::glm5 as glm;
    Ok(if options.glm5_drafter_disabled { (0, 0, None, None) }
        else if options.drafter_bytes > 0 { (options.drafter_bytes, 0, None, None) }
        else {
            let snapshot = options.glm5_drafter_snapshot.clone().or_else(|| {
                let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
                    .unwrap_or_else(|| std::path::PathBuf::from("/mnt/sparknest/hf-home"));
                crate::resolve_snapshot_at_revision("incoai/GLM-5.3-DFlash2", Some(&home), None).ok().and_then(|r| r.snapshot_path)
            }).ok_or_else(|| anyhow::anyhow!("GLM drafter config unavailable; pass its snapshot, --drafter-gib, or disable speculation"))?;
            let sequences = usize::try_from(options.glm_draft_sequences())?;
            let slots = crate::families::glm5::draft_representation::serving_draft_context_slots(
                sequences, usize::try_from(options.resolved_concurrency("glm5"))?, options.draft_context_slots.map(usize::try_from).transpose()?);
            let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
            let (_, block) = crate::families::glm5::draft_representation::draft_geometry(&config)?;
            let drafts = usize::try_from(block)?.checked_sub(1)
                .ok_or_else(|| anyhow::anyhow!("GLM draft block must include an anchor"))?;
            (glm::drafter_bytes(&snapshot, slots, sequences,
                options.physical_sms.unwrap_or_else(|| crate::placement::ArchContext::for_device("sm_120", options.rtx_bytes[0]).sms) as u64,
                options.glm5_draft_bf16, options.draft_kv_cache)?, glm::drafter_staging(&snapshot, slots, sequences, options.glm5_draft_bf16)?, Some(drafts),
                Some(sequences.checked_mul(usize::try_from(block)?).context("GLM native draft head row extent")?))
        })
}

fn placement(report: &PlanReport, input: &crate::plan::PlanInput, options: &LayoutOptions,
    devices: &[DeviceLayout], manifest: Option<&serde_json::Value>, rows: u64, context: u64,
    core: Option<&crate::families::glm5::core::CoreCharges>)
    -> anyhow::Result<(crate::placement::Placement, crate::placement::families::glm5::GlmWorkingSet)> {
    use crate::placement::{families::glm5 as glm, Baseline, Onboard};
    let checkpoint = input.checkpoint();
    let cfg = crate::families::glm5::GlmDsaConfig::from_hf(&checkpoint.config)?;
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 };
    let lanes = options.prefill_waves;
    anyhow::ensure!(report.requested_attention != Some(crate::placement::AttentionPlacement::Layers)
        || lanes <= crate::placement::LAYER_PIPELINE_MAX_LANES,
        "layer-owned prefill lacks capacity for {lanes} waves");
    let (draft, draft_staging, active_drafts, _) = drafter_admission(options)?;
    let attention_plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
        glm::DECODE_ROWS as usize, active_drafts, options.glm5_attention_plan_rows)?;
    let selected_manifest = manifest.map(|m| -> anyhow::Result<_> {
        let mut selected = m.clone();
        let family = cuteafd_core::coordinator_programs::CoordinatorPrograms { family: "glm",
            split_family: (devices.len() == 2).then_some("glm2") };
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
    let gpus = devices.iter().zip(&options.rtx_bytes).map(|(d, &capacity)| (capacity,
        Baseline::Planned { context_bytes: 0, loaded_bytes: d.used_bytes() })).collect();
    let onboard = options.onboard.or(options.local_expert_layers.map(Onboard::Layers)).unwrap_or_else(glm::default_onboard);
    let exchange_f32 = glm::local_exchange_f32()?;
    let manifest_path = options.workspace_manifest.as_deref().unwrap_or_else(||
        std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let catalog = input.expert_catalog()?;
    let local = glm::select_local_inventory(&catalog, manifest_path, cfg.layers, devices.len(),
        rows, lanes, exchange_f32, onboard, false, None)?;
    let tp2_workspace = local.as_ref().map_or([0; 2], |l| l.workspace());
    let serial_workspace = local.as_ref().map_or([0; 2], |l| l.workspace_for(&cfg, rows, 1, exchange_f32));
    let inputs = glm::GlmInputs { cfg: &cfg, layers: cfg.layers, gpus,
        headroom_bytes: options.headroom_bytes, spark_ranks, native_experts: false, native_workspace: 0,
        native_dense: core.map(|c| c.bytes(cfg.hidden)).transpose()?.unwrap_or_default(),
        native_head: core.is_some_and(|c| c.ranks[0].target_head.is_some()), skip_routed_experts: false, prefill_rows: rows, prefill_lanes: lanes,
        max_context: context, automatic_context: options.context_tokens == 0, scratch, drafter_bytes: draft, drafter_staging: draft_staging, pending_code: Vec::new(),
        experts: local.as_ref().map_or_else(Vec::new, |l| l.experts.clone()), expert_workspace: 0,
        tp2_workspace, requested_pool: options.pool_tokens,
        onboard, full_prefill_logits: options.full_prefill_logits,
        context: selected_manifest.as_ref().and_then(|m| glm::manifest_context(m,
            options.physical_sms.unwrap_or_else(|| crate::placement::ArchContext::for_device(
                "sm_120", options.rtx_bytes[0]).sms) as usize, &attention_plan)) };
    // Auto resolves in the solver (context where the build carries its programs), as at runtime.
    Ok(glm::solve_working_set(&inputs, serial_workspace, report.requested_attention)?)
}

/// Native EXLR8 uses the ordinary GLM solver and coordinator shared ownership.
pub fn native_layout(checkpoint: &Checkpoint, options: &LayoutOptions, shared_bytes: &[u64],
    workspace: u64, spark_ranks: usize, manifest: &serde_json::Value, selected: Option<&str>)
    -> anyhow::Result<(MemoryLayout, crate::placement::Placement)> {
    native_layout_with_metadata(checkpoint, options, shared_bytes, workspace, spark_ranks, manifest, selected, None)
}

pub(crate) fn native_layout_with_metadata(checkpoint: &Checkpoint, options: &LayoutOptions, shared_bytes: &[u64],
    workspace: u64, spark_ranks: usize, manifest: &serde_json::Value, selected: Option<&str>,
    metadata: Option<&crate::plan::exlr8::residency::MetadataInput>)
    -> anyhow::Result<(MemoryLayout, crate::placement::Placement)> {
    use crate::placement::{families::glm5 as glm, Baseline, Onboard};
    let gpu_count = options.rtx_bytes.len();
    anyhow::ensure!((1..=2).contains(&gpu_count) && shared_bytes.len() == gpu_count,
        "EXLR8 requires 1..=2 RTX shared owners matching coordinator GPUs");
    let split = gpu_count == 2;
    let attention = if split { crate::placement::AttentionPlacement::Context }
        else { crate::placement::AttentionPlacement::Heads };
    let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
    let rows = if options.prefill_rows == 0 { 4096 } else { options.prefill_rows };
    let context = if options.context_tokens == 0 { 1 << 20 } else { options.context_tokens };
    let (draft, draft_staging, active_drafts, _) = drafter_admission(options)?;
    let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
        glm::DECODE_ROWS as usize, active_drafts, options.glm5_attention_plan_rows)?;
    let mut manifest = manifest.clone();
    cuteafd_core::coordinator_programs::select_attention_plans(&mut manifest, plan,
        |name| cuteafd_core::coordinator_programs::CoordinatorPrograms {
            family: "glm", split_family: Some("glm2") }.contains(name))?;
    let scratch = glm::manifest_scratch(&cfg, split, &manifest)?;
    let context_programs = if split { Some(glm::manifest_context(&manifest,
        options.physical_sms.unwrap_or_else(|| crate::placement::ArchContext::for_device(
            "sm_120", options.rtx_bytes[0]).sms) as usize, &plan)
        .ok_or_else(|| anyhow::anyhow!("EXLR8 requires GLM context programs in PROGRAMS.json"))?) }
        else { None };
    let native = crate::plan::exlr8::recognizes(&checkpoint.snapshot)?;
    let resident = if native {
        let records = crate::plan::exlr8::dense::DenseRecord::inventory(&checkpoint.snapshot)?;
        (0..gpu_count).map(|rank| crate::plan::exlr8::dense::resident_native(&records,
            gpu_count, rank, cfg.layers, cfg.heads, cfg.qk_nope_head_dim, cfg.v_head_dim,
            cfg.kv_lora_rank).map_err(anyhow::Error::from)).collect::<anyhow::Result<Vec<_>>>()?
    } else { resident_owned(checkpoint, gpu_count, cfg.layers, None)? };
    let lanes = options.prefill_waves;
    let dense_scratch = if native {
        let a8 = options.exlr8_prefill_activations == crate::plan::exlr8::PrefillActivations::A8;
        let bytes = crate::plan::exlr8::dense::lane_charge_bytes(&cfg, gpu_count, glm::DECODE_ROWS as usize,
            rows as usize, a8);
        vec![bytes as u64; gpu_count]
    } else { Vec::new() };
    let mut devices = Vec::new();
    for (rank, groups) in resident.into_iter().enumerate() {
        let mut items: Vec<_> = groups.into_iter().filter(|(group, _, _)| group != "shared_expert")
            .map(|(group, format, bytes)| Item::new(if group == "embedding" { Category::Embedding }
                else { Category::Weights }, group, format, bytes, Basis::Exact)).collect();
        items.push(Item::new(Category::Weights, "shared_expert", "exlr8", shared_bytes[rank], Basis::Exact));
        let set = crate::families::glm5::DensePrograms::Exlr8.program_set(&manifest, split);
        let arch = crate::placement::ArchContext::coordinator(options.rtx_bytes[rank], None);
        let allowance = crate::placement::inventory::loaded_code_allowance("glm", "glm:exlr8", split,
            rank as u8, &arch, Some(&set));
        let mut runtime = Item::new(Category::Runtime, "context+modules", "",
            arch.context_bytes + allowance.bytes, allowance.basis);
        runtime.group = "EXLR8 context+modules".into();
        items.push(runtime);
        devices.push(DeviceLayout { kind: DeviceKind::Rtx, index: rank as u32,
            capacity_bytes: options.rtx_bytes[rank].saturating_sub(options.headroom_bytes), items, kv_tokens: 0 });
    }
    let onboard = options.onboard.or(options.local_expert_layers.map(Onboard::Layers)).unwrap_or_else(glm::default_onboard);
    let local = if let Some(selected) = selected {
        let catalog = crate::read_expert_catalog(&checkpoint.snapshot)?;
        let manifest_path = options.workspace_manifest.as_deref()
            .unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
        if let Some(metadata) = metadata.filter(|_| gpu_count == 2
            && onboard.layers(cfg.layers.saturating_sub(catalog.routed_experts().first_layer)) != Some(0)
            && cfg.layers > catalog.routed_experts().first_layer) {
            Some(glm::exlr8_inventory_with_metadata(&checkpoint.snapshot, manifest_path, cfg.layers, rows, lanes,
                glm::local_exchange_f32()?, options.exlr8_prefill_activations, metadata)?)
        } else {
            glm::select_local_inventory(&catalog, manifest_path, cfg.layers, gpu_count, rows, lanes,
                glm::local_exchange_f32()?, onboard, false, Some((selected, options.exlr8_prefill_activations)))?
        }
    } else { None };
    let tp2_workspace = local.as_ref().map_or([0; 2], |local| local.workspace());
    let serial_workspace = local.as_ref().map_or([0; 2], |local| local.workspace_for(&cfg, rows, 1, false));
    let inputs = glm::GlmInputs { cfg: &cfg, layers: cfg.layers,
        gpus: devices.iter().zip(&options.rtx_bytes).map(|(device, &capacity)| (capacity,
            Baseline::Planned { context_bytes: 0, loaded_bytes: device.used_bytes() })).collect(),
        headroom_bytes: options.headroom_bytes, spark_ranks, native_experts: true,
        native_workspace: workspace, native_dense: dense_scratch, native_head: false, skip_routed_experts: false, prefill_rows: rows,
        prefill_lanes: lanes,
        max_context: context, automatic_context: options.context_tokens == 0, scratch: Some(scratch), drafter_bytes: draft,
        drafter_staging: draft_staging, pending_code: Vec::new(), experts: local.as_ref().map_or_else(Vec::new, |local| local.experts.clone()), expert_workspace: 0,
        tp2_workspace, requested_pool: options.pool_tokens, onboard,
        full_prefill_logits: options.full_prefill_logits, context: context_programs };
    let (placement, _) = glm::solve_working_set(&inputs, serial_workspace, Some(attention))?;
    for (device, items) in devices.iter_mut().zip(&placement.items) {
        device.items.extend(items.iter().cloned());
        device.kv_tokens = placement.pool_tokens;
    }
    Ok((MemoryLayout { devices, pool_tokens: placement.pool_tokens, waste: vec![],
        notes: vec![format!("EXLR8 native {gpu_count} shared owners; {attention} attention; context {context}; {}", placement.summary())] }, placement))
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

    #[test]
    fn exlr8_explicit_stages_preserve_the_pre_namespace_fix_module_charge() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k|
            crate::plan::testing::exl3(n, r, k, 4)), None);
        let checkpoint = Checkpoint::coordinator(dir.path(), false, false).unwrap();
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
            let options = LayoutOptions { rtx_bytes: vec![96 << 30; gpus], prefill_rows: 512,
                prefill_lanes: 3, prefill_waves: 3, context_tokens: 8192, pool_tokens: Some(262144),
                onboard: Some(crate::placement::Onboard::Layers(0)), glm5_drafter_disabled: true,
                ..Default::default() };
            let (layout, _) = native_layout(&checkpoint, &options, &vec![256; gpus], 256, 4,
                &expanded, None).unwrap();
            let families = if gpus == 2 { vec!["glm", "glm2", "exlr8"] } else { vec!["glm", "exlr8"] };
            // This was the exact selector before compound namespace separation.
            let names = manifest["programs"].as_array().unwrap().iter()
                .filter_map(|p| p["name"].as_str()).filter(|name|
                    name.split_once('_').is_some_and(|(prefix, _)| families.contains(&prefix)))
                .map(str::to_string).collect();
            let old = crate::placement::ProgramSet { families: families.iter().map(|f| f.to_string()).collect(),
                names, scratch: Vec::new() };
            assert!(old.names.iter().any(|name| name.starts_with("glm_stage_")));
            for (rank, device) in layout.devices.iter().enumerate().take(gpus) {
                let arch = crate::placement::ArchContext::coordinator(options.rtx_bytes[rank], None);
                let allowance = crate::placement::inventory::loaded_code_allowance("glm", "glm:exlr8",
                    gpus == 2, rank as u8, &arch, Some(&old));
                let actual = device.items.iter().find(|i| i.group == "EXLR8 context+modules").unwrap();
                assert_eq!(actual.bytes, arch.context_bytes + allowance.bytes);
            }
        }
    }

    #[test]
    fn exlr8_min_max_prefill_matches_exl3_and_charges_every_wave() {
        use crate::plan::{PlanInput, PlanOptions, ExpertPlacement, testing};
        use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress, SparkRings};
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k| testing::exl3(n, r, k, 4)), None);
        let input = PlanInput::open(dir.path()).unwrap();
        let manifest = native_manifest();
        let path = dir.path().join("PROGRAMS.json");
        std::fs::write(&path, manifest.to_string()).unwrap();
        for (gpus, sparks) in [(1, 4), (2, 6)] {
            // CLI maps PREFILL_WAVES to prefill_lanes, preserving explicit N1.
            for (unit, requested, span) in [(0, 0, None), (2048, 1, Some(2048)),
                (4096, 2, Some(8192)), (1024, 3, None)] {
                let options = LayoutOptions { rtx_bytes: vec![96 << 30; gpus], prefill_rows: unit,
                    prefill_lanes: requested, prefill_waves: requested.max(1), prefill_span_rows: span,
                    workspace_manifest: Some(path.clone()), context_tokens: 65536,
                    onboard: Some(crate::placement::Onboard::Layers(0)), glm5_drafter_disabled: true,
                    host_memory: Some((128 << 30, 128 << 30)), host_cache_bytes: Some(0), ..Default::default() };
                let plan_options = PlanOptions { placement: ExpertPlacement::Sparks { ranks: sparks },
                    layout: Some(options.clone()), ..Default::default() };
                let exl3 = input.plan(&plan_options).unwrap();
                let mut native = crate::plan::empty_report(input.checkpoint(), &plan_options).unwrap();
                let mut options = options;
                let shape = crate::plan::layout::resolve_prefill(&mut native, &mut options, sparks,
                    crate::plan::Family::prefill_limits(&crate::plan::families::glm::GLM5)).unwrap();
                native.family = exl3.family.clone();
                native.spec = exl3.spec.clone();
                native.cache_requirements = exl3.cache_requirements.clone();
                native.spark_ranks = sparks;
                native.memory_layout = Some(native_layout(input.checkpoint(), &options,
                    &vec![256; gpus], 256, sparks, &manifest, None).unwrap().0);
                charge_host(&mut native, &options, input.checkpoint(), true).unwrap();
                let geometry = |r: &PlanReport| (r.prefill_unit_rows, r.prefill_waves,
                    r.prefill_span_rows, r.prefill_transports);
                // Unpinned, the fixed depth (3 x 2752 with Sparks, one unit
                // without); an explicit PREFILL_WAVES pins k.
                let planned = crate::serving_capacity::prefill_waves::depth(sparks, Default::default());
                let waves = if requested == 0 { planned } else { requested };
                let rows = if unit != 0 { unit } else if waves <= 1 { 4096 } else {
                    crate::serving_capacity::prefill_waves::unit_rows(waves) };
                let admitted = rows * waves;
                assert_eq!(geometry(&native), (Some(rows), Some(waves), Some(span.unwrap_or(admitted)), Some(1)));
                assert_eq!(geometry(&native), geometry(&exl3), "{gpus} RTX/{sparks} Sparks");
                let spec = native.spec.as_ref().unwrap();
                let moe = spec.moe.as_ref().unwrap();
                let expert = ExpertGeometry { hidden: spec.hidden as u32, experts: moe.experts as u32,
                    topk: moe.top_k as u32, intermediate: moe.intermediate as u32, layers: spec.layers.len() as u32 };
                // The prefill transport count excludes the separate decode endpoint.
                let endpoints = 2;
                let endpoint = SparkEndpoint { ring_depth: None, capacity: rows as u32, ingress: SparkIngress::Fp8K32 };
                let (request, response) = endpoint.wire_bytes(expert).unwrap();
                let rings = SparkRings { max_frame_bytes: (64 << 20).max(request).max(response),
                    depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
                let host = native.host_ledger.as_ref().unwrap();
                let bytes = |name| host.items.iter().find(|i| i.group == name).unwrap().bytes;
                assert_eq!(bytes("spark-intake"), shape.span_rows * expert.hidden as u64 * 4 * endpoints * sparks as u64);
                assert_eq!(bytes("transport/rdma-rings"), endpoint.ring_bytes(expert, rings).unwrap() as u64 * endpoints * sparks as u64);
                assert!(!host.items.iter().any(|item| item.group == "workspace/router"));
                assert_eq!(host.shortfall_bytes, 0);
            }
        }
    }

    #[test]
    fn exlr8_a8_prefill_charges_exactly_the_lane_records_and_nothing_else() {
        use crate::placement::families::glm5 as glm;
        use crate::plan::exlr8::{dense, PrefillActivations};
        const GROUP: &str = "native dense scratch and prefill staging";
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::exlr8::glm_snapshot(dir.path());
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config).unwrap();
        let manifest = native_manifest();
        let scratch = |layout: &MemoryLayout, gpu: usize| layout.devices[gpu].items.iter()
            .filter(|item| item.group == GROUP).map(|item| item.bytes).sum::<u64>();
        let rest = |layout: &MemoryLayout| MemoryLayout { devices: layout.devices.iter().map(|device| DeviceLayout {
            items: device.items.iter().filter(|item| item.group != GROUP).cloned().collect(), ..device.clone() })
            .collect(), ..layout.clone() };
        for (gpus, sparks) in [(1, 4), (2, 6)] {
            for (rows, lanes) in [(4096u64, 1u64), (4096, 2), (2048, 3), (1024, 4)] {
                let options = LayoutOptions { rtx_bytes: vec![96 << 30; gpus], prefill_rows: rows, prefill_lanes: lanes,
                    prefill_waves: lanes, context_tokens: 65536, onboard: Some(crate::placement::Onboard::Layers(0)),
                    glm5_drafter_disabled: true, host_memory: Some((128 << 30, 128 << 30)), host_cache_bytes: Some(0),
                    ..Default::default() };
                let plan = |activations| native_layout(&checkpoint,
                    &LayoutOptions { exlr8_prefill_activations: activations, ..options.clone() },
                    &vec![256; gpus], 256, sparks, &manifest, None).unwrap();
                let (a16, a16_placement) = plan(PrefillActivations::A16);
                let (a8, a8_placement) = plan(PrefillActivations::A8);
                let (heads, inter) = (cfg.heads / gpus, cfg.dense_intermediate / gpus);
                let records = dense::a8_scratch_bytes(&cfg, heads, inter, rows as usize) as u64;
                assert!(records > 0);
                let context = format!("{gpus} RTX/{sparks} Sparks/{rows} rows/{lanes} lanes");
                for gpu in 0..gpus {
                    // Every slot shares one prefill scratch and its records (#762).
                    assert_eq!(scratch(&a16, gpu), (dense::scratch_bytes(&cfg, heads, inter, glm::DECODE_ROWS as usize)
                        + dense::scratch_bytes(&cfg, heads, inter, rows as usize)) as u64, "{context}");
                    assert_eq!(scratch(&a8, gpu), scratch(&a16, gpu) + records, "{context} gpu {gpu}");
                }
                // The automatic pool takes what is left, so it absorbs the records (never grows) ...
                assert!(a8_placement.pool_tokens <= a16_placement.pool_tokens, "{context}");
                assert!(a8_placement.pool_tokens < a16_placement.pool_tokens || records < 1 << 20, "{context}");
                // ... and with the pool pinned, nothing but the scratch item differs.
                let pinned = |activations| native_layout(&checkpoint, &LayoutOptions { pool_tokens: Some(1 << 19),
                    exlr8_prefill_activations: activations, ..options.clone() },
                    &vec![256; gpus], 256, sparks, &manifest, None).unwrap().0;
                let (pinned_a16, pinned_a8) = (pinned(PrefillActivations::A16), pinned(PrefillActivations::A8));
                assert_eq!(pinned_a8.pool_tokens, 1 << 19, "{context}");
                for gpu in 0..gpus {
                    assert_eq!(scratch(&pinned_a8, gpu), scratch(&pinned_a16, gpu) + records, "{context}");
                }
                assert_eq!(rest(&pinned_a8), rest(&pinned_a16), "{context}");
            }
        }
    }

    #[test]
    fn native_automatic_pool_matches_ordinary_and_keeps_explicit_requests_strict() {
        use crate::placement::{families::glm5 as glm, PlacementError};
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None)), None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let manifest = native_manifest();
        let default = cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS;
        for gpus in 1..=2 {
            // Whole execution must not depend on any context-split or glm2 exports.
            let mut manifest = manifest.clone();
            if gpus == 1 {
                manifest["programs"].as_array_mut().unwrap().retain(|p|
                    !p["name"].as_str().unwrap().starts_with("glm2_")
                    && !glm::CONTEXT_PROGRAMS.contains(&p["name"].as_str().unwrap()));
            }
            for context in [1 << 20, default + 128] {
                for pool_tokens in [None, Some(0)] {
                    let options = LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; gpus],
                        context_tokens: context, pool_tokens, glm5_drafter_disabled: true, ..Default::default() };
                    let (memory, placement) = native_layout(&checkpoint, &options, &vec![256; gpus], 256, 4, &manifest, None).unwrap();
                    let drafting = LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), glm5_drafter_disabled: false,
                        drafter_bytes: 1 << 30, ..options.clone() };
                    let (_, draft_placement) = native_layout(&checkpoint, &drafting,
                        &vec![256; gpus], 256, 4, &manifest, None).unwrap();
                    let draft: u64 = draft_placement.items.iter().flatten()
                        .filter(|item| item.category == Category::Drafter).map(|item| item.bytes).sum();
                    assert_eq!(draft, 1 << 30);
                    assert_eq!(memory.devices.len(), gpus);
                    assert!(memory.pool_tokens >= default.max(context));
                    assert_eq!(placement.pool_tokens, memory.pool_tokens);
                    assert_eq!(placement.attention_placement, if gpus == 2 { crate::placement::AttentionPlacement::Context }
                        else { crate::placement::AttentionPlacement::Heads });
                }
            }
            // Put the small header fixture under the same pressure as the full
            // checkpoint: automatic pools must fit, not become strict 2M requests.
            for lanes in [2, 3] {
                let options = LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; gpus],
                    prefill_rows: 4096, prefill_lanes: lanes, glm5_drafter_disabled: true, ..Default::default() };
                let sparks = if gpus == 1 { 4 } else { 6 };
                let (unpressured, _) = native_layout(&checkpoint, &options, &vec![256; gpus], 256, sparks, &manifest, None).unwrap();
                let shared: Vec<_> = unpressured.devices.iter().map(|device| {
                    let records: u64 = device.items.iter().filter(|item| item.group == "records")
                        .map(|item| item.bytes).sum();
                    let target_records = records / unpressured.pool_tokens * default;
                    256 + device.capacity_bytes - device.used_bytes() + records - target_records + target_records / 8
                }).collect();
                let input = crate::plan::PlanInput::open(dir.path()).unwrap();
                let report = crate::plan::empty_report(&checkpoint, &crate::plan::PlanOptions {
                    placement: crate::plan::ExpertPlacement::Sparks { ranks: sparks },
                    attention_placement: Some(if gpus == 2 { crate::placement::AttentionPlacement::Context }
                        else { crate::placement::AttentionPlacement::Heads }), ..Default::default()
                }).unwrap();
                for pool_tokens in [None, Some(0)] {
                    let options = LayoutOptions { pool_tokens, onboard: Some(crate::placement::Onboard::Layers(0)),
                        ..options.clone() };
                    let (memory, native) = native_layout(&checkpoint, &options, &shared, 256, sparks, &manifest, None).unwrap();
                    assert!(memory.pool_tokens >= 1 << 20);
                    assert!(memory.pool_tokens < default);
                    assert_eq!(memory.pool_tokens, native.pool_tokens);
                    assert!(memory.devices.iter().all(|device| device.used_bytes() <= device.capacity_bytes));
                    // Give the ordinary planner identical resident/working-set
                    // bytes, including the native-only shared execution arena.
                    let devices: Vec<_> = memory.devices.iter().zip(&native.items).map(|(device, solved)| {
                        let baseline = device.used_bytes() - solved.iter().map(|item| item.bytes).sum::<u64>();
                        DeviceLayout { items: vec![Item::new(Category::Weights, "identical resident bytes", "",
                            baseline + 256 * (1 + lanes), Basis::Exact)], kv_tokens: 0, ..device.clone() }
                    }).collect();
                    let (ordinary, _) = placement(&report, &input, &options, &devices, Some(&manifest),
                        4096, 1 << 20, None).unwrap();
                    assert_eq!(native.pool_tokens, ordinary.pool_tokens, "{gpus} RTX/{sparks} Sparks/{lanes} lanes");
                }
                let explicit = LayoutOptions { pool_tokens: Some(default), ..options.clone() };
                assert!(native_layout(&checkpoint, &explicit, &shared, 256, sparks, &manifest, None).is_err());
                let diagnostic = LayoutOptions { pool_tokens: Some(1 << 19), ..options };
                let (memory, _) = native_layout(&checkpoint, &diagnostic, &shared, 256, sparks, &manifest, None).unwrap();
                assert_eq!(memory.pool_tokens, 1 << 19);
            }
            let options = LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; gpus],
                context_tokens: default + 128, pool_tokens: Some(default), glm5_drafter_disabled: true, ..Default::default() };
            let error = native_layout(&checkpoint, &options, &vec![256; gpus], 256, 4, &manifest, None).unwrap_err();
            assert_eq!(error.downcast_ref::<PlacementError>(), Some(&PlacementError::RequestedBelowContext {
                requested: default, floor: default + 128,
            }));
            let options = LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![1 << 30; gpus],
                context_tokens: 1 << 20, pool_tokens: Some(1 << 20), glm5_drafter_disabled: true, ..Default::default() };
            assert!(native_layout(&checkpoint, &options, &vec![256; gpus], 256, 4, &manifest, None).is_err());
            assert!(native_layout(&checkpoint, &options, &[], 256, 4, &manifest, None).is_err());
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
