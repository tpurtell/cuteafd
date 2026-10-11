//! GLM Flash plan adapter. All coordinator admission is the runtime's shared request.
use super::*;
use crate::placement::{self, families::glm5_flash as glmf, Baseline, Onboard};
use crate::families::glm5_flash::{GlmNextConfig, resident};

pub(super) fn layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    checkpoint: &super::super::Checkpoint, options: &LayoutOptions, manifest: Option<&serde_json::Value>,
    rows: u64, sequences: u64, context: u64) -> MemoryLayout {
    let mut notes = vec![format!("resolved context {context} tokens")];
    let ranks = if options.head_split && options.rtx_bytes.len() >= 2 { 2 } else { 1 };
    let mut devices: Vec<_> = options.rtx_bytes.iter().take(ranks).enumerate().map(|(rank, &bytes)| DeviceLayout {
        kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: bytes.saturating_sub(options.headroom_bytes),
        items: vec![], kv_tokens: 0 }).collect();
    let mut pool = 0;
    let admitted = (|| -> anyhow::Result<(placement::Placement, glmf::GlmfWorkingSet)> {
        let cfg = GlmNextConfig::from_hf(&checkpoint.config)?;
        let spark_ranks = match report.placement { ExpertPlacement::Local => 0, ExpertPlacement::Sparks { ranks } => ranks };
        let lanes = if options.prefill_lanes > 0 { options.prefill_lanes } else { crate::serving_capacity::GLMF_DEFAULT_PREFILL_LANES };
        let decode_rows = options.glmf_decode_rows;
        anyhow::ensure!(ranks == 1 || decode_rows == 64, "a head split takes --decode-rows 64");
        if decode_rows > 64 && manifest.is_some_and(|m| crate::serving_capacity::glmf_manifest_scratch(m)("glmf_mhc_post_pre_m128").is_none()) {
            anyhow::bail!("wide decode needs CUTEAFD_GLMF_WIDE_DECODE_ROWS=128");
        }
        let representation = options.glmf_representation;
        let mut resident = resident::resident_weights(checkpoint, &cfg, cfg.layers, ranks, representation)
            .map_err(anyhow::Error::msg)?;
        let host_embedding = options.host_embedding || (options.rtx_bytes[0] <= 32 * GIB && !options.force_gpu_embedding);
        if host_embedding {
            checkpoint.require_untied_embedding("model.language_model.embed_tokens.weight")?;
            resident[0].embedding = 0;
            notes.push("embedding placement: host (single pinned mapped copy)".into());
        }
        let router = if ranks == 2 { resident::router_replica_bytes(checkpoint, &cfg, cfg.layers).map_err(anyhow::Error::msg)? } else { 0 };
        let catalog = crate::read_expert_catalog(&checkpoint.snapshot)?;
        let shape = catalog.routed_experts();
        let onboard = options.onboard.or_else(|| options.local_expert_layers.map(|n| Onboard::Layers(n.saturating_sub(shape.first_layer))))
            .unwrap_or_else(glmf::default_onboard);
        let max_rows = crate::serving_capacity::glmf_expert_rows(rows, decode_rows);
        let tp = if ranks == 2 { 2 } else { 1 };
        let scratch = glmf::expert_workspace(&catalog, options.workspace_manifest.as_deref(), max_rows, tp);
        let needed = spark_ranks == 0 || (onboard != Onboard::Auto && onboard.layers(shape.layers - shape.first_layer) != Some(0));
        let available = match scratch { Ok(bytes) => Some(bytes), Err(error) if needed => return Err(error), Err(error) => {
            notes.push(format!("RTX expert onboarding unavailable: {error}")); None } };
        let experts = glmf::expert_costs(&catalog, ranks == 2 && available.is_some())?;
        let geometry = crate::serving_capacity::glm_flash_rank_cache_geometry_rows(&cfg, cfg.layers, ranks,
            options.glmf_index, 4, decode_rows)?;
        let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_for(
            options.glmf_mark_lanes.unwrap_or(sequences), options.mimo_prefix_entries,
            geometry.ranks.iter().map(|r| r.retained_mark_bytes).sum(), options.mimo_prefix_mark_bytes));
        let shared = if options.glmf_shared_replay { crate::serving_capacity::glm_flash_kda_replay_bytes_rows(&cfg, cfg.layers, ranks, decode_rows)? } else { 0 };
        let step_workspace = |placement: &ExpertPlacement, lanes| manifest.and_then(|m|
            glmf_step_workspace(m, checkpoint, placement, lanes, rows, context, decode_rows, shared,
                options.glmf_index == crate::serving_capacity::GlmfIndexCache::Compact,
                ranks == 2, representation, options.glmf_prefill_expanded, available.is_some() && (ranks == 2 || catalog.fp8().is_some()), options.glmf_kda_fp32_partials));
        let mut workspace = step_workspace(&report.placement, lanes);
        let mut local_workspace = step_workspace(&ExpertPlacement::Local, 1);
        if (workspace.is_none() || local_workspace.is_none()) && manifest.is_some() {
            anyhow::bail!("GLM Flash program manifest lacks the requested step scratch geometry");
        }
        if workspace.is_none() {
            notes.push("GLM Flash workspace estimate needs --workspace-manifest for qualification".into());
            let estimate = |lanes: u64| (0..ranks).map(|rank| glmf::GlmfBufferDemand::new(
                glmf::GlmfComponent::Always, rank as u8, Category::Workspace, "steps",
                (if ranks == 1 { gib(268) } else { gib(472) }) * (lanes * rows).max(8192) / 8192,
                Basis::Calibrated)).collect();
            workspace = Some(estimate(lanes));
            local_workspace = Some(estimate(1));
        }
        let mut workspace = workspace.unwrap();
        let mut local_workspace = local_workspace.unwrap();
        if checkpoint.tensors.iter().any(|t| t.meta.name.ends_with("mlp.gate_proj.weight") && t.meta.dtype == cuteafd_core::DType::U8) {
            let dense = options.glmf_dense_manifest.as_ref().and_then(|p| std::fs::read(p).ok())
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|m| placement::inventory::fp8moe_scratch_bytes(&m, "tp1", max_rows))
                .or_else(|| placement::inventory::dense_package_scratch(&placement::inventory::image_lib(options.workspace_manifest.as_deref()), "glmfdense", max_rows, glmf::nvfp4_a4()))
                .ok_or_else(|| anyhow::anyhow!("dense NVFP4 package scratch manifest missing"))?;
            for profile in [&mut workspace, &mut local_workspace] {
                profile.push(glmf::GlmfBufferDemand::new(glmf::GlmfComponent::Always, 0,
                    Category::Workspace, "steps", dense + 8 * max_rows, Basis::Formula));
            }
        }
        let draft_sms: Vec<_> = options.rtx_bytes[..ranks].iter().map(|&total|
            options.physical_sms.unwrap_or(placement::ArchContext::coordinator(total, None).sms) as u64).collect();
        // Explicit fixture totals remain opaque: do not add inferred scratch on top.
        let (draft, drafter_scratch) = if options.glmf_drafter_disabled { (0, vec![0; ranks]) }
            else if options.drafter_bytes > 0 { (options.drafter_bytes, vec![0; ranks]) } else {
            let path = options.glmf_drafter_snapshot.clone().or_else(|| {
                let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from).unwrap_or_else(|| "/mnt/sparknest/hf-home".into());
                crate::resolve_snapshot_at_revision("incoai/GLM-5.3-Flash-DFlash2", Some(&home), None).ok().and_then(|r| r.snapshot_path)
            });
            let path = path.ok_or_else(|| anyhow::anyhow!("DFlash2 config missing: provide --glmf-drafter-snapshot or disable the drafter"))?;
            let config = serde_json::from_slice::<serde_json::Value>(&std::fs::read(path.join("config.json"))?)?;
            glmf::drafter_inventory(&config,
                options.draft_context_slots.unwrap_or(sequences.max(8)) as usize,
                options.draft_sequences.max(sequences).min(options.draft_context_slots.unwrap_or(sequences.max(8))) as usize,
                &draft_sms, crate::families::glm5::draft_representation::GlmDraftRepresentation::Fp8Only, 2)?
        };
        let mut baselines = Vec::new();
        for rank in 0..ranks {
            let total = options.rtx_bytes[rank];
            let arch = placement::ArchContext::coordinator(total, None);
            let code = placement::loaded_code("glmf", if catalog.exl3().is_some() { "exl3" } else { "fp8" }, ranks == 2, rank as u8)
                .map_or_else(|| planned_context(manifest, &if ranks == 2 { vec!["glmf".into(), "glmf2".into()] } else { vec!["glmf".into()] }, total), |c| arch.context_bytes + c.bytes);
            devices[rank].items.push(Item::new(Category::Runtime, "context+modules", "", code, Basis::Calibrated));
            baselines.push((total, Baseline::Planned { context_bytes: code, loaded_bytes: 0 }));
        }
        let automatic_target = placement::PoolPolicy::resolve(&options.rtx_bytes[..ranks], context, None, 256, spark_ranks == 0).target;
        let target = options.pool_tokens.filter(|&n| n > 0).unwrap_or_else(||
            if options.target_pool_tokens == cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS {
                automatic_target
            } else { options.target_pool_tokens.max(context) });
        let graphs = options.graph_budget_bytes.map(|bytes| placement::GraphSet::budget(&vec![bytes; ranks]))
            .unwrap_or_else(|| glmf::startup_graphs(&cfg, cfg.layers, context as usize, target as usize,
                sequences as usize, draft > 0, ranks, decode_rows as usize, options.physical_sms.unwrap_or(188) as usize));
        let inputs = glmf::GlmfInputs { cfg: &cfg, layers: cfg.layers, gpus: baselines, pending_code: vec![0; ranks],
            headroom_bytes: options.headroom_bytes, spark_ranks, prefill_lanes: lanes, prefill_rows: rows,
            decode_rows, partial_bytes: if options.glmf_kda_fp32_partials { 4 } else { 2 }, max_context: context, sequences, speculation: draft > 0, state_slots: options.state_slots.unwrap_or(sequences.max(8)),
            mark_slots: marks, pool_marks: options.glmf_pool_marks && options.mimo_prefix_entries > 0,
            index: options.glmf_index, kda_state_bytes: 4, shared_replay: options.glmf_shared_replay, representation,
            resident, router_replica_bytes: router, workspace, local_workspace, graphs, experts,
            expert_workspace: if ranks == 1 { available.unwrap_or(0) } else { 0 },
            tp2_workspace: if ranks == 2 { [available.unwrap_or(0); 2] } else { [0; 2] },
            drafter_bytes: draft, drafter_scratch, requested_pool: options.pool_tokens.filter(|&n| n > 0), onboard,
            full_prefill_logits: if options.full_prefill_logits { full_prefill_logits_bytes("glm5_flash", rows, cfg.vocab_size as u64) } else { 0 } };
        glmf::solve_working_set_with_graphs(&inputs, target, options.context_tokens == 0,
            options.physical_sms.unwrap_or(placement::ArchContext::coordinator(options.rtx_bytes[0], None).sms) as usize,
            available.is_some()).map(|(placed, _, working)| {
                notes.push(format!("selected Spark ranks {}, prefill lanes {}, layers {:?}",
                    working.spark_ranks, working.prefill_lanes, working.spark_layers));
                (placed, working)
            }).map_err(Into::into)
    })();
    let mut selected = None;
    match admitted {
        Ok((placed, working)) => {
            pool = placed.pool_tokens;
            selected = Some(working);
            for (device, items) in devices.iter_mut().zip(&placed.items) {
                device.items.extend(items.iter().cloned()); device.kv_tokens = pool;
            }
            notes.push(format!("placement: {}", placed.summary()));
        }
        Err(error) => { report.placement_supported = false; notes.push(format!("GLM Flash pool-first placement: {error:#}")); }
    }
    if let Some(working) = selected.filter(|working| working.spark_ranks > 0) {
        let total = report.components.iter().filter(|c| c.component == Component::RoutedExpert).map(|c| c.bytes).sum::<u64>();
        let routed = model.spec().moe_layers().max(1);
        let remaining = total.saturating_sub(total / routed as u64 * working.rtx_layers.len() as u64);
        for rank in 0..working.spark_ranks {
            devices.push(DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: options.spark_bytes,
                kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed_expert", "", (remaining as f64 * report.spark_rank_share) as u64, Basis::Exact),
                    Item::new(Category::Workspace, "expert waves", "", gib(56) * options.spark_capacity_rows / 4096, Basis::Calibrated),
                    Item::new(Category::Transport, "rdma rings", "", gib(78), Basis::Calibrated),
                    Item::new(Category::Runtime, "context+modules", "", 512 * MIB, Basis::Calibrated)] });
        }
    }
    // Media placement is still the shared planner path, outside the coordinator solver port.
    let mut sparks = devices.split_off(ranks);
    resolve_encoder(checkpoint, report, model, &mut devices, &mut sparks, &vec![0; ranks], options, &mut notes);
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens: pool, waste: vec![], notes }
}
