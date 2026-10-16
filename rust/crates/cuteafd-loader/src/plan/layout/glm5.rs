//! GLM coordinator operands after GlmLoader's packing and rank ownership.
use crate::families::glm5::{GlmDsaConfig, GlmIndexer};
use crate::plan::Checkpoint;
use cuteafd_core::DType;
use std::collections::BTreeMap;

use super::*;

/// Spark worker allowances at 4096 capacity rows: the non-I/O wave residual
/// (exact rank I/O and rings are charged from the selected working set).
const SPARK: SparkCharges = SparkCharges { host_bytes: None, wave_bytes: gib(190), basis: Basis::Calibrated };

pub(crate) fn layout(report: &mut PlanReport, model: &dyn crate::plan::FamilyModel,
    input: &crate::plan::PlanInput, options: &LayoutOptions) -> MemoryLayout {
    let checkpoint = input.checkpoint();
    let mut shape = Shape::resolve(report, model, options, ShapePolicy::manifest_bounded());
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
    match resident(checkpoint, ranks) {
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
    for (index, device) in devices.iter_mut().take(ranks).enumerate() {
        device.items.push(runtime_item(shape.manifest.as_ref(), &program_families("glm", split),
            local_package(report), split, index, options.rtx_bytes[index]));
    }
    let lanes = if options.prefill_lanes > 0 { options.prefill_lanes }
        else { glm_prefill_lanes(std::env::var("CUTEAFD_GLM_PREFILL_LANES").ok().as_deref()) as u64 };
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 };
    let mut working = crate::placement::SparkWorkingSet { rtx_layers: Vec::new(),
        spark_layers: if spark_ranks > 0 { (0..model.spec().layers.len()).collect() } else { Vec::new() },
        spark_ranks, prefill_lanes: lanes };
    let mut sparks = spark_devices(report, model, checkpoint, options, SPARK, &working,
        spark_endpoints("glm5", lanes), &mut waste, &mut notes);
    let mut pool_tokens = 0;
    let mut local_layers = 0;
    match placement(report, input, options, &devices[..ranks], shape.manifest.as_ref(), shape.prefill_rows,
        shape.context_tokens) {
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
    // Reapply after admission has selected its executable suffix and lane count.
    charge_spark_devices(report, model, checkpoint, options, SPARK, &working, spark_endpoints("glm5", working.prefill_lanes),
        &mut sparks, &mut notes, |device| format!("{} I/O admission", device.name()));
    if report.encoder.is_none() {
        resolve_encoder(report, &Towers::checkpoint(report), &mut devices, &mut sparks, &vec![0; gpus], options, &mut notes);
    }
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens, waste, notes }
}

pub(crate) fn resident(checkpoint: &Checkpoint, ranks: usize) -> anyhow::Result<Vec<Vec<(String, String, u64)>>> {
    anyhow::ensure!((1..=2).contains(&ranks), "GLM coordinator ranks");
    let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
    let tensors: BTreeMap<_, _> = checkpoint.tensors.iter().map(|t| (t.meta.name.as_str(), &t.meta)).collect();
    let tensor = |name: &str| tensors.get(name).copied().ok_or_else(|| anyhow::anyhow!("missing GLM operand {name}"));
    let mut groups = vec![BTreeMap::<(String, String), u64>::new(); ranks];
    let mut put = |rank: usize, group: &str, format: &str, bytes: u64| {
        *groups[rank].entry((group.into(), format.into())).or_default() += bytes.max(256);
    };
    let native = std::env::var("CUTEAFD_GLM_BF16").is_ok_and(|v| v == "native");
    for layer in 0..cfg.layers {
        let p = format!("model.layers.{layer}");
        let attn = format!("{p}.self_attn");
        let qkv = [format!("{attn}.q_a_proj.weight"), format!("{attn}.kv_a_proj_with_mqa.weight")];
        let attention = [qkv[0].clone(), qkv[1].clone(), format!("{attn}.q_b_proj.weight"),
            format!("{attn}.kv_b_proj.weight"), format!("{attn}.o_proj.weight")];
        let bf16 = native && attention.iter().all(|n| tensor(n).is_ok_and(|t| t.dtype == DType::Bf16));
        let qkv_rows: u64 = qkv.iter().map(|n| tensor(n).map(|t| t.shape[0] as u64)).sum::<anyhow::Result<_>>()?;
        let h = cfg.hidden as u64;
        let projection = |name: &str| -> anyhow::Result<(u64, u64)> {
            let t = tensor(name)?;
            anyhow::ensure!(t.shape.len() == 2, "GLM matrix {name}");
            Ok((t.shape[0] as u64 * t.shape[1] as u64, (t.shape[0] as u64).div_ceil(128) * (t.shape[1] as u64).div_ceil(128) * 4))
        };
        for rank in 0..ranks {
            for name in [format!("{p}.input_layernorm.weight"), format!("{p}.post_attention_layernorm.weight"),
                format!("{attn}.q_a_layernorm.weight"), format!("{attn}.kv_a_layernorm.weight")] {
                put(rank, "norm", "bf16", tensor(&name)?.byte_length);
            }
            put(rank, "attention", if bf16 { "bf16" } else { "fp8" }, qkv_rows * h * if bf16 { 2 } else { 1 });
            if !bf16 {
                let grid = qkv.iter().map(|n| projection(n).map(|(_, s)| s)).sum::<anyhow::Result<u64>>()?;
                put(rank, "attention", "fp8-scale", grid);
                put(rank, "attention", "fp8-scale", qkv_rows * h.div_ceil(128) * 4);
            }
            for name in [format!("{attn}.q_b_proj.weight"), format!("{attn}.o_proj.weight")] {
                let (values, scales) = projection(&name)?;
                put(rank, "attention", if bf16 { "bf16" } else { "fp8" }, values / ranks as u64 * if bf16 { 2 } else { 1 });
                if !bf16 { put(rank, "attention", "fp8-scale", scales / ranks as u64); }
            }
            for width in [cfg.qk_nope_head_dim, cfg.v_head_dim] {
                let values = cfg.heads as u64 / ranks as u64 * cfg.kv_lora_rank as u64 * width as u64;
                put(rank, "attention", if bf16 { "bf16" } else { "fp8" }, values * if bf16 { 2 } else { 1 });
                if !bf16 { put(rank, "attention", "fp8-scale", values / 64 * 4); }
            }
            if cfg.indexers[layer] == GlmIndexer::Full {
                let name = format!("{attn}.indexer.wq_b.weight");
                let index_bf16 = native && tensor(&name)?.dtype == DType::Bf16;
                let (values, scales) = projection(&name)?;
                put(rank, "indexer", if index_bf16 { "bf16" } else { "fp8" }, values * if index_bf16 { 2 } else { 1 });
                if !index_bf16 { put(rank, "indexer", "fp8-scale", scales); }
                let ik = ["wk.weight", "weights_proj.weight"].into_iter().map(|suffix| tensor(&format!("{attn}.indexer.{suffix}"))
                    .map(|t| t.shape.iter().product::<usize>() as u64 * 2)).sum::<anyhow::Result<u64>>()?;
                put(rank, "indexer", "bf16", ik);
                for suffix in ["k_norm.weight", "k_norm.bias"] {
                    put(rank, "indexer", "bf16", tensor(&format!("{attn}.indexer.{suffix}"))?.byte_length);
                }
            }
            let dense = layer < cfg.first_moe_layer;
            let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
            let names = [format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight"), format!("{mlp}.down_proj.weight")];
            let shared_bf16 = !dense && native && names.iter().all(|n| tensor(n).is_ok_and(|t| t.dtype == DType::Bf16));
            let group = if dense { "dense" } else { "shared_expert" };
            let gu = names[..2].iter().map(|n| projection(n)).collect::<anyhow::Result<Vec<_>>>()?;
            let down = projection(&names[2])?;
            for (values, scales) in [(gu[0].0 + gu[1].0, gu[0].1 + gu[1].1), down] {
                put(rank, group, if shared_bf16 { "bf16" } else { "fp8" }, values / ranks as u64 * if shared_bf16 { 2 } else { 1 });
                if !shared_bf16 { put(rank, group, "fp8-scale", scales / ranks as u64); }
            }
            if dense && names.iter().all(|n| tensor(n).is_ok_and(|t| t.dtype == DType::F8E4M3)
                && tensors.contains_key(format!("{n}_scale").as_str())) {
                put(rank, group, "fp8-tensor-scale", 48);
                put(rank, group, "fp8-tensor-scale", 48);
            }
            if !dense && rank == 0 {
                for suffix in ["weight", "e_score_correction_bias"] {
                    put(rank, "router", "bf16", tensor(&format!("{p}.mlp.gate.{suffix}"))?.byte_length);
                }
            }
        }
    }
    for (name, group) in [("model.embed_tokens.weight", "embedding"), ("model.norm.weight", "norm"), ("lm_head.weight", "lm_head")] {
        let t = tensor(name)?;
        let bytes = if t.dtype == DType::F8E4M3 && t.shape.len() == 2 { t.shape.iter().product::<usize>() as u64 * 2 } else { t.byte_length };
        put(0, group, "bf16", bytes);
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
        AttentionPlacement::Layers => "GLM 5.3 has no layers attention executor; \
            use heads or context.".into(),
    }
}

fn placement(report: &PlanReport, input: &crate::plan::PlanInput, options: &LayoutOptions,
    devices: &[DeviceLayout], manifest: Option<&serde_json::Value>, rows: u64, context: u64)
    -> anyhow::Result<(crate::placement::Placement, crate::placement::families::glm5::GlmWorkingSet)> {
    use crate::placement::{families::glm5 as glm, Baseline, Onboard};
    let checkpoint = input.checkpoint();
    let cfg = crate::families::glm5::GlmDsaConfig::from_hf(&checkpoint.config)?;
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 };
    let lanes = glm::prefill_lanes(spark_ranks, if options.prefill_lanes > 0 { options.prefill_lanes }
        else { glm_prefill_lanes(std::env::var("CUTEAFD_GLM_PREFILL_LANES").ok().as_deref()) as u64 });
    let (draft, draft_staging) = if options.glm5_drafter_disabled { (0, 0) }
        else if options.drafter_bytes > 0 { (options.drafter_bytes, 0) }
        else {
            let snapshot = options.glm5_drafter_snapshot.clone().or_else(|| {
                let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
                    .unwrap_or_else(|| std::path::PathBuf::from("/mnt/sparknest/hf-home"));
                crate::resolve_snapshot_at_revision("incoai/GLM-5.3-DFlash2", Some(&home), None).ok().and_then(|r| r.snapshot_path)
            }).ok_or_else(|| anyhow::anyhow!("GLM drafter config unavailable; pass its snapshot, --drafter-gib, or disable speculation"))?;
            let sequences = options.draft_sequences as usize;
            let slots = options.draft_context_slots.unwrap_or(20.max(options.draft_sequences)) as usize;
            (glm::drafter_bytes(&snapshot, slots, sequences,
                options.physical_sms.unwrap_or_else(|| crate::placement::ArchContext::for_device("sm_120", options.rtx_bytes[0]).sms) as u64,
                options.glm5_draft_bf16)?, glm::drafter_staging(&snapshot, slots, sequences, options.glm5_draft_bf16)?)
        };
    let scratch = manifest.map(|m| glm::manifest_scratch(&cfg, devices.len() == 2, m)).transpose()?;
    let gpus = devices.iter().zip(&options.rtx_bytes).map(|(d, &capacity)| (capacity,
        Baseline::Planned { context_bytes: 0, loaded_bytes: d.used_bytes() })).collect();
    let onboard = options.onboard.or(options.local_expert_layers.map(Onboard::Layers)).unwrap_or_else(glm::default_onboard);
    let exchange_f32 = glm::local_exchange_f32()?;
    let manifest_path = options.workspace_manifest.as_deref().unwrap_or_else(||
        std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let catalog = input.expert_catalog()?;
    let local = glm::select_local_inventory(&catalog, manifest_path, cfg.layers, devices.len(),
        rows, lanes, exchange_f32, onboard, false)?;
    let tp2_workspace = local.as_ref().map_or([0; 2], |l| l.workspace());
    let serial_workspace = local.as_ref().map_or([0; 2], |l| l.workspace_for(&cfg, rows, 1, exchange_f32));
    let inputs = glm::GlmInputs { cfg: &cfg, layers: cfg.layers, gpus,
        headroom_bytes: options.headroom_bytes, spark_ranks, skip_routed_experts: false, prefill_rows: rows, prefill_lanes: lanes,
        max_context: context, scratch, drafter_bytes: draft, drafter_staging: draft_staging, pending_code: Vec::new(),
        experts: local.as_ref().map_or_else(Vec::new, |l| l.experts.clone()), expert_workspace: 0,
        tp2_workspace, requested_pool: options.pool_tokens,
        onboard, full_prefill_logits: options.full_prefill_logits,
        context: manifest.and_then(glm::manifest_context) };
    // Auto resolves in the solver (context where the build carries its programs), as at runtime.
    Ok(glm::solve_working_set(&inputs, serial_workspace, report.requested_attention)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{fp8, glm5_config, glm5_tensors, write_snapshot};

    #[test]
    fn resident_headers_follow_rank_ownership_and_absorbed_kv_packing() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None)), None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let one = resident(&checkpoint, 1).unwrap();
        let two = resident(&checkpoint, 2).unwrap();
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
}
