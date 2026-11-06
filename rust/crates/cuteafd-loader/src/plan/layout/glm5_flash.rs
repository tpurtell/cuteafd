//! GLM Flash plan adapter. All coordinator admission is the runtime's shared request.
use super::*;
use anyhow::Context;
use crate::plan::spark_executions::SparkBackend;
use crate::placement::{self, families::glm5_flash as glmf, Baseline, Onboard};
use crate::families::glm5_flash::{GlmNextConfig, resident};

pub(crate) fn layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &mut LayoutOptions) -> MemoryLayout {
    layout_with_native(report, model, input, options, None)
}

pub(crate) fn native_layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &mut LayoutOptions,
    facts: &crate::plan::exlr8::native::DenseFacts, metadata: &crate::plan::exlr8::residency::MetadataInput)
    -> anyhow::Result<MemoryLayout> {
    let memory = layout_with_native(report, model, input, options, Some((facts, metadata)));
    anyhow::ensure!(report.solver_placement.is_some(), "{}", report.hints.iter()
        .map(|hint| hint.what.as_str()).collect::<Vec<_>>().join("; "));
    Ok(memory)
}

fn layout_with_native(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &mut LayoutOptions,
    exlr8: Option<(&crate::plan::exlr8::native::DenseFacts, &crate::plan::exlr8::residency::MetadataInput)>) -> MemoryLayout {
    let prefill = match resolve_prefill(input, report, options, report.spark_ranks,
        crate::serving_capacity::prefill_waves::PrefillLimits {
            max_waves: if report.spark_ranks == 0 { 1 }
                else if report.requested_attention == Some(placement::AttentionPlacement::Layers) {
                placement::LAYER_PIPELINE_MAX_LANES } else { 4 },
            ..crate::plan::Family::prefill_limits(&crate::plan::families::glm::GLM5_FLASH)
        }) {
        Ok(shape) => shape,
        Err(error) => return unsupported_prefill(report, error),
    };
    let shape = Shape::resolve(input, report, model, options, ShapePolicy::manifest_bounded());
    let (manifest, rows, sequences, context) = (shape.manifest.as_deref(), shape.prefill_rows,
        shape.concurrency, shape.context_tokens);
    let checkpoint = input.checkpoint();
    let mut notes = vec![format!("resolved context {context} tokens")];
    let ranks = if options.head_split && options.rtx_bytes.len() >= 2 { 2 } else { 1 };
    let mut devices: Vec<_> = options.rtx_bytes.iter().take(ranks).enumerate().map(|(rank, &bytes)| DeviceLayout {
        kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: bytes.saturating_sub(options.headroom_bytes), worker_budget_bytes: 0,
        items: vec![], kv_tokens: 0 }).collect();
    let mut pool = 0;
    let mut waste = Vec::new();
    let admitted = (|| -> anyhow::Result<(placement::Placement, glmf::GlmfWorkingSet)> {
        let cfg = GlmNextConfig::from_hf(&checkpoint.config)?;
        let spark_ranks = match report.placement { ExpertPlacement::Local => 0, ExpertPlacement::Sparks { ranks } => ranks };
        let lanes = prefill.waves;
        let decode_rows = options.glmf_decode_rows;
        anyhow::ensure!(ranks == 1 || decode_rows == 64, "a head split takes --decode-rows 64");
        let draft_config = if options.glmf_drafter_disabled || options.drafter_bytes > 0 { None } else {
            let path = options.glmf_drafter_snapshot.clone().or_else(|| {
                let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
                    .unwrap_or_else(|| "/mnt/sparknest/hf-home".into());
                crate::resolve_snapshot_at_revision("incoai/GLM-5.3-Flash-DFlash2", Some(&home), None)
                    .ok().and_then(|r| r.snapshot_path)
            }).context("DFlash2 config missing: provide --glmf-drafter-snapshot or disable the drafter")?;
            let config = serde_json::from_slice::<serde_json::Value>(&std::fs::read(path.join("config.json"))?)?;
            let (slots, batch) = crate::families::glm5::draft_representation::flash_serving_draft_limits(
                usize::try_from(sequences)?, usize::try_from(options.glm_draft_sequences())?,
                options.draft_context_slots.map(usize::try_from).transpose()?);
            let capacity = crate::families::glm5::draft_representation::flash_draft_capacity(&config, slots, batch)?;
            Some((config, capacity))
        };
        let attention_plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
            64, draft_config.as_ref().map(|(_, capacity)| capacity.block_rows / capacity.max_batch_sequences - 1),
            options.glm5_attention_plan_rows)?;
        let sms = options.physical_sms.unwrap_or(placement::ArchContext::coordinator(options.rtx_bytes[0], None).sms);
        let selected_manifest = manifest.map(|manifest| -> anyhow::Result<_> {
            let mut selected = manifest.clone();
            cuteafd_core::coordinator_programs::select_card_plans(&mut selected, sms)?;
            cuteafd_core::coordinator_programs::select_attention_plans(&mut selected, attention_plan,
                |name| (name.starts_with("glmf_") && (decode_rows > 64 || !name.ends_with("_m128")))
                    || (ranks == 2 && name.starts_with("glmf2_")))?;
            Ok(selected)
        }).transpose()?;
        let manifest = selected_manifest.as_ref();
        // Context and step admission use the same boot-time numerical recipe as serving.
        let context_programs = if ranks == 2 && !matches!(report.requested_attention,
            Some(placement::AttentionPlacement::Heads | placement::AttentionPlacement::Layers)) {
            manifest.map(|m| glmf::manifest_context_for_cards(
                m, cfg.heads, &[sms as usize], attention_plan)).transpose()?.flatten()
        } else { None };
        let requested_attention = if options.glmf_pool_marks && options.prefix_entries("glm5_flash") > 0
            && report.requested_attention.is_none() && ranks == 2 { Some(placement::AttentionPlacement::Heads) }
            else { report.requested_attention };
        let attention = glmf::resolve_attention(requested_attention, ranks, context_programs)?;
        if report.requested_attention.is_none() && ranks == 2 && context_programs.is_none() {
            notes.push(format!("auto attention fallback: layers; missing context programs: {}",
                glmf::context_program_list()));
        }
        let lanes = glmf::admitted_prefill_waves(attention.mode, lanes)?;
        report.attention_placement = attention.mode;
        let index = match attention.mode {
            placement::AttentionPlacement::Context => crate::serving_capacity::GlmfIndexCache::Compact,
            placement::AttentionPlacement::Heads if ranks == 2 => crate::serving_capacity::GlmfIndexCache::Keys,
            _ => options.glmf_index,
        };
        if decode_rows > 64 && manifest.is_some_and(|m| crate::serving_capacity::glmf_manifest_scratch(m)("glmf_mhc_post_pre_m128").is_none()) {
            anyhow::bail!("wide decode needs CUTEAFD_GLMF_WIDE_DECODE_ROWS=128");
        }
        let core = if exlr8.is_some() { crate::families::glm5_flash::core::CoreInventory::default() }
            else { crate::families::glm5_flash::core::CoreInventory::from_checkpoint(&cfg, checkpoint)? };
        let native = exlr8.is_some() || !core.projections.is_empty();
        let native_mask = if exlr8.is_some() { crate::serving_capacity::GlmfCoreScratch { kda: true, mla: true, ffn: true } }
            else { core.scratch_mask(&cfg, cfg.layers) };
        let native_kda = native_mask.kda;
        let whole_mla = |first: usize| {
            let mut next = first;
            cfg.attention.iter().enumerate().filter_map(|(layer, kind)| {
                if attention.mode != placement::AttentionPlacement::Layers
                    || *kind != crate::families::glm5_flash::GlmNextAttention::Mla { return None; }
                let owner = next; next ^= 1; Some((layer, owner))
            }).collect::<std::collections::BTreeMap<_, _>>()
        };
        let native_resident = |first| -> anyhow::Result<_> {
            let facts = exlr8.context("native EXLR8 facts")?.0;
            resident::native_resident(&facts.selection, cfg.layers, &facts.plans(&whole_mla(first))?, &facts.shared_bytes)
        };
        let mut representation = options.glmf_representation;
        if native_kda {
            representation.kda_fp8 = false;
            representation.output_shard = false;
        }
        if native { representation.fp8_head = false; }
        let state = options.glmf_kda_state;
        anyhow::ensure!(!native || state == crate::serving_capacity::GlmfKdaState::F32,
            "native Flash requires F32 KDA state programs");
        anyhow::ensure!(state == crate::serving_capacity::GlmfKdaState::F32 || ranks == 1,
            "BF16 KDA state needs one GPU; head-split BF16 state is not yet qualified");
        notes.push(format!("KDA projections: {}, state: {state:?}; shared target/draft head: {}",
            if exlr8.is_some() { "native EXLR8" } else if native { "native K6" } else if representation.kda_fp8 { "FP8" } else { "BF16" },
            if exlr8.is_some() { "native EXLR8" } else if native { "native K6" } else if representation.fp8_head { "FP8" } else { "BF16" }));
        let mut resident = if exlr8.is_some() { native_resident(attention.first_owner as usize)? }
            else { resident::resident_weights_placed(checkpoint, &cfg, cfg.layers, ranks, representation,
                attention.mode, attention.first_owner as usize).map_err(anyhow::Error::msg)? };
        let host_embedding = options.host_embedding || (options.rtx_bytes[0] <= 32 * GIB && !options.force_gpu_embedding);
        if host_embedding {
            checkpoint.require_untied_embedding("model.language_model.embed_tokens.weight")?;
            resident[0].embedding = 0;
            notes.push("embedding placement: host (single pinned mapped copy)".into());
        }
        let catalog = input.expert_catalog()?;
        let shape = catalog.routed_experts();
        let onboard = options.resolve_onboard(shape.routed_layers())
            .unwrap_or_else(glmf::default_onboard);
        let max_rows = crate::serving_capacity::glmf_expert_rows(rows, decode_rows);
        let tp = if ranks == 2 { 2 } else { 1 };
        let scratch = if let Some((_, metadata)) = exlr8 {
            crate::plan::exlr8::native::flash_routed_workspace(metadata, &cfg,
                manifest.context("native Flash requires PROGRAMS.json")?, ranks, max_rows)
        } else { glmf::expert_workspace(&catalog, options.workspace_manifest.as_deref(), max_rows, tp) };
        let needed = spark_ranks == 0 || (onboard != Onboard::Auto && onboard.layers(shape.layers - shape.first_layer) != Some(0));
        let available = match scratch { Ok(bytes) => Some(bytes), Err(error) if needed => return Err(error), Err(error) => {
            notes.push(format!("RTX expert onboarding unavailable: {error}")); None } };
        let experts = if let Some((_, metadata)) = exlr8 {
            crate::plan::exlr8::routed_expert_costs(metadata, &(shape.first_layer..cfg.layers).collect::<Vec<_>>())?
        } else { glmf::expert_costs(&catalog, ranks == 2 && available.is_some())? };
        let geometry = crate::serving_capacity::glm_flash_placed_cache_geometry_rows(&cfg, cfg.layers, ranks,
            index, state.bytes(), decode_rows, attention.mode, 0)?;
        let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(
            options.glmf_mark_lanes.unwrap_or(sequences), options.prefix_entries("glm5_flash"),
            geometry.ranks.iter().map(|r| r.retained_mark_bytes).sum(), options.mimo_prefix_mark_bytes, options.host_cache_bytes != Some(0)));
        let shared = if options.glmf_shared_replay { crate::serving_capacity::glm_flash_kda_replay_bytes_rows(&cfg, cfg.layers, ranks, decode_rows)? } else { 0 };
        let step_workspace = |placement: &ExpertPlacement, lanes| manifest.and_then(|m|
            glmf_step_workspace(m, checkpoint, placement, lanes, rows, context, decode_rows, shared,
                index == crate::serving_capacity::GlmfIndexCache::Compact,
                ranks == 2, representation, options.glmf_prefill_expanded && !native_kda,
                available.is_some() && (ranks == 2 || (exlr8.is_none() && catalog.fp8().is_some())),
                options.glmf_kda_fp32_partials && !native_kda, state, Some(native_mask)));
        let mut workspace = step_workspace(&report.placement, lanes);
        let mut local_workspace = step_workspace(&ExpertPlacement::Local, lanes);
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
            local_workspace = Some(estimate(lanes));
        }
        let mut workspace = workspace.unwrap();
        let mut local_workspace = local_workspace.unwrap();
        if checkpoint.tensors.iter().any(|t| t.meta.name.ends_with("mlp.gate_proj.weight") && t.meta.dtype == cuteafd_core::DType::U8) {
            let dense = options.glmf_dense_manifest.as_ref().and_then(|p| std::fs::read(p).ok())
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|m| placement::inventory::fp8moe_scratch_bytes(&m, "tp1", max_rows))
                .or_else(|| placement::inventory::dense_package_scratch(&placement::inventory::image_lib(options.workspace_manifest.as_deref()), "glmfdense", max_rows))
                .ok_or_else(|| anyhow::anyhow!("dense NVFP4 package scratch manifest missing"))?;
            for profile in [&mut workspace, &mut local_workspace] {
                profile.push(glmf::GlmfBufferDemand::new(glmf::GlmfComponent::Always, 0,
                    Category::Workspace, "steps", dense + 8 * max_rows, Basis::Formula));
            }
        }
        let draft_sms: Vec<_> = options.rtx_bytes[..ranks].iter().map(|&total|
            options.physical_sms.unwrap_or(placement::ArchContext::coordinator(total, None).sms) as u64).collect();
        // Explicit fixture totals remain opaque: do not add inferred scratch on top.
        let mut draft_rows = None;
        let (draft, drafter_weights, drafter_scratch, drafter_bridge) = if options.glmf_drafter_disabled { (0, placement::ResidentWeights::default(), vec![0; ranks], None) }
            else if options.drafter_bytes > 0 && options.glmf_drafter_snapshot.is_none() {
                (options.drafter_bytes, placement::ResidentWeights(options.drafter_bytes), vec![0; ranks], None)
            } else {
            let (config, capacity) = draft_config.as_ref().context("resolved Flash drafter config")?;
            let (slots, batch) = (capacity.context_slots, capacity.max_batch_sequences);
            draft_rows = Some(capacity.block_rows);
            let (bytes, scratch) = glmf::drafter_inventory(config, slots, batch,
                &draft_sms, crate::families::glm5::draft_representation::GlmDraftRepresentation::Fp8Only, 2, options.draft_kv_cache)?;
            let bridge = if ranks == 2 && glmf::remote_dflash_requested() {
                glmf::drafter_bridge_inventory(&config, slots, batch)? } else { None };
            anyhow::ensure!(options.drafter_bytes == 0 || options.drafter_bytes == bytes.resident_weights.0,
                "drafter weight override {} differs from checkpoint-resident weights {}",
                options.drafter_bytes, bytes.resident_weights.0);
            (bytes.memory.resident, bytes.resident_weights, scratch, bridge)
        };
        let core_charge = if native && exlr8.is_none() {
            let owners = (attention.mode == placement::AttentionPlacement::Layers).then(|| {
                let mut next = attention.first_owner as u8;
                cfg.attention.iter().map(|kind| {
                    let owner = next;
                    if *kind == crate::families::glm5_flash::GlmNextAttention::Mla { next ^= 1; }
                    owner
                }).collect::<Vec<_>>()
            });
            Some(crate::families::glm5_flash::core_admission::CoreCharges::from_package(&cfg, &core,
                cfg.layers, ranks, owners.as_deref(), rows as usize, lanes as usize,
                decode_rows as usize, draft_rows, ranks == 2 && representation.output_shard,
                manifest.ok_or_else(|| anyhow::anyhow!("Flash K6 requires PROGRAMS.json with native exports"))?)?)
        } else { None };
        if let Some(charge) = &core_charge {
            let bytes = charge.bytes(cfg.hidden)?;
            for profile in [&mut workspace, &mut local_workspace] {
                for (rank, &bytes) in bytes.iter().enumerate() {
                    profile.push(glmf::GlmfBufferDemand::new(glmf::GlmfComponent::Always, rank as u8,
                        Category::Workspace, "native K6 core", bytes, Basis::Exact));
                }
            }
            notes.push("native K6 body and head; exact compressed owners and projection scratch".into());
        }
        let native_charge = exlr8.map(|(facts, _)| crate::plan::exlr8::native::FlashCharges::from_manifest(
            facts, &cfg, &whole_mla(attention.first_owner as usize), rows as usize, lanes as usize,
            decode_rows as usize, index, manifest.context("native Flash requires PROGRAMS.json")?)).transpose()?;
        if let Some(charge) = &native_charge {
            for profile in [&mut workspace, &mut local_workspace] {
                for (rank, bytes) in charge.bytes()?.into_iter().enumerate() {
                    profile.push(glmf::GlmfBufferDemand::new(glmf::GlmfComponent::Always, rank as u8,
                        Category::Workspace, "native EXLR8 core/shared expert", bytes, Basis::Exact));
                }
            }
            notes.push("native EXLR8 body/head; shared format-owned residency and scratch".into());
        }
        let local_tp2_exl3 = ranks == 2 && spark_ranks == 0 && catalog.exl3().is_some();
        let mut baselines = Vec::new();
        for rank in 0..ranks {
            let total = options.rtx_bytes[rank];
            let package = if exlr8.is_some() { "glmf:exlr8" } else if local_tp2_exl3 { "exl3-tp2-local" } else if catalog.exl3().is_some() { "exl3" } else { "fp8" };
            let runtime = if let (Some(charge), Some(manifest)) = (&native_charge, manifest) {
                let mut set = placement::ProgramSet::from_manifest(manifest, &["glmf", "glmf2", "exlr8_glmf"]);
                set.names.extend(charge.programs.iter().cloned()); set.names.sort(); set.names.dedup();
                let arch = placement::ArchContext::coordinator(total, None);
                let allowance = placement::inventory::loaded_code_allowance("glmf", package,
                    ranks == 2, rank as u8, &arch, Some(&set));
                Item::new(Category::Runtime, "context+modules", "", arch.context_bytes + allowance.bytes, allowance.basis)
            } else if let (Some(charge), Some(manifest)) = (&core_charge, manifest) {
                let mut set = placement::ProgramSet::from_manifest(manifest, &["glmf", "glmf2"]);
                set.names.extend(charge.programs.iter().cloned()); set.names.sort(); set.names.dedup();
                let arch = placement::ArchContext::coordinator(total, None);
                let allowance = placement::inventory::loaded_code_allowance("glmf-k6", package,
                    ranks == 2, rank as u8, &arch, Some(&set));
                Item::new(Category::Runtime, "context+modules", "", arch.context_bytes + allowance.bytes, allowance.basis)
            } else { runtime_item(manifest, &program_families("glmf", ranks == 2), package, ranks == 2, rank, total) };
            let code = runtime.bytes;
            if local_tp2_exl3 && !native {
                devices[rank].items.extend(placement::inventory::glmf_local_tp2_runtime_items(rank));
            } else {
                devices[rank].items.push(runtime);
            }
            baselines.push((total, Baseline::Planned { context_bytes: code, loaded_bytes: 0 }));
        }
        let automatic_target = placement::PoolPolicy::resolve(&options.rtx_bytes[..ranks], context, None, 256, spark_ranks == 0).target;
        let target = options.pool_tokens.filter(|&n| n > 0).unwrap_or(automatic_target);
        let graphs = options.graph_budget_bytes.map(|bytes| placement::GraphSet::budget(&vec![bytes; ranks]))
            .unwrap_or_else(|| glmf::startup_graph_bound(&cfg, cfg.layers, context as usize, target as usize,
                sequences as usize, draft > 0, ranks, decode_rows as usize, options.physical_sms.unwrap_or(188) as usize));
        let inputs = glmf::GlmfInputs {
        resident_weight_limit: Some(crate::placement::ResidentWeights(options.weight_budget_bytes)), spark_reply_format: options.resolved_spark_reply_format(false), automatic_context: options.context_tokens == 0, attention, cfg: &cfg, layers: cfg.layers, local_tp2_exl3: local_tp2_exl3 && !native, gpus: baselines, pending_code: vec![0; ranks],
            headroom_bytes: options.headroom_bytes, spark_ranks, prefill_lanes: lanes, prefill_rows: rows,
            decode_rows, partial_bytes: if options.glmf_kda_fp32_partials && !native_kda { 4 } else { 2 }, max_context: context, sequences, speculation: draft > 0, state_slots: options.resolved_state_slots("glm5_flash"),
            mark_slots: marks, pool_marks: options.glmf_pool_marks && options.prefix_entries("glm5_flash") > 0,
            index, kda_state_bytes: state.bytes(), shared_replay: options.glmf_shared_replay, representation,
            resident, workspace, local_workspace, graphs, experts,
            expert_workspace: if ranks == 1 { available.unwrap_or(0) } else { 0 },
            tp2_workspace: if ranks == 2 { [available.unwrap_or(0); 2] } else { [0; 2] },
            drafter_bytes: draft, drafter_weights, drafter_scratch, drafter_bridge, requested_pool: options.pool_tokens.filter(|&n| n > 0), onboard,
            full_prefill_logits: if options.full_prefill_logits && exlr8.is_none() && core.get_weight("lm_head").is_none() { full_prefill_logits_bytes("glm5_flash", rows, cfg.vocab_size as u64) } else { 0 } };
        let host = options.host_memory.map(Ok)
            .unwrap_or_else(crate::formats::mapped_preload::host_memory_before_process)?;
        let mut process = glmf::admission::ProcessOwners::new(host, (8 << 30).max(host.0 / 10));
        if ranks == 2 && attention.mode == placement::AttentionPlacement::Layers && !native {
            let mut alternate = resident::resident_weights_placed(checkpoint, &cfg, cfg.layers, ranks,
                representation, attention.mode, (attention.first_owner ^ 1) as usize).map_err(anyhow::Error::msg)?;
            if host_embedding { for rank in &mut alternate { rank.embedding = 0; } }
            process.alternate_resident = Some(alternate);
        }
        let geometry = glmf::admission::host_geometry(&inputs)?;
        let quota = if options.prefix_entries("glm5_flash") == 0 || options.host_cache_bytes == Some(0) { 0 }
            else { geometry.resolve(options.host_cache_bytes, context, sequences, host.0, geometry.chunk_bytes())
                .context("invalid Flash host prefix geometry")? };
        let embedding = if host_embedding { checkpoint.tensors.iter()
            .filter(|t| t.meta.name == "model.language_model.embed_tokens.weight")
            .map(|t| t.meta.byte_length).sum() } else { 0 };
        use crate::plan::encoder::{self, EncoderGpuBudget, EncoderHardware, EncoderKind,
            EncoderPlacement, EncoderRequest, EncoderSparkBudget};
        let towers = glm_flash_towers(report, checkpoint);
        let (weights, scratch) = match towers.vision {
            Ok(bytes) => bytes,
            Err(_) if report.vision == crate::plan::MediaMode::Off => (0, 0),
            Err(reason) => anyhow::bail!("Flash vision tower unavailable: {reason}"),
        };
        let hardware = EncoderHardware { v41: false,
            gpus: options.rtx_bytes[..ranks].iter().map(|&total| EncoderGpuBudget {
                free_bytes: total.saturating_sub(options.headroom_bytes), kv_target_bytes: 0 }).collect(),
            sparks: (0..spark_ranks).map(|rank| EncoderSparkBudget { rank, host: format!("spark{rank}"),
                idle: false, expert_bytes: 0, free_bytes: options.spark_allocation_budget_bytes.unwrap_or(121 << 30) }).collect() };
        process.encoders = encoder::candidates(&EncoderRequest { mode: report.vision, weights, scratch,
            copies: options.vision_replicas }, &hardware);
        if report.vision == crate::plan::MediaMode::Auto && weights > 0 {
            process.encoders.push(EncoderPlacement { kind: EncoderKind::Off, weights: 0, scratch: 0,
                replicas: vec![], reason: "no admitted native tower without reducing the pool".into(), shortfall: 0 });
        }
        anyhow::ensure!(!process.encoders.is_empty(), "requested Flash encoder has no executable owner");
        process.media_cache_bytes = options.media_cache_bytes.unwrap_or((8 << 30).min(host.0 / 20));
        process.serving(&inputs, quota, geometry.chunk_bytes().min(quota), embedding, 0)?;
        process.spark_link(&inputs, checkpoint.spark_input_is_bf16())?;
        process.push_host = glmf::admission::push_host_envelope(checkpoint, &cfg, ranks, Some(&catalog), exlr8)?;
        if let Some((facts, metadata)) = exlr8 {
            use crate::plan::exlr8::{dense::{DensePlan, DenseRole}, residency::ExpertSet};
            let owners = whole_mla(attention.first_owner as usize);
            let records = facts.records.iter().filter(|r| matches!(facts.selection.role(&r.tensor, cfg.layers),
                Ok(DenseRole::Vocabulary(_)))).cloned().collect::<Vec<_>>();
            let vocabulary = DensePlan::for_rank(&records, ranks, 0)?;
            let records = facts.selection.backbone(&facts.records, cfg.layers)?;
            let body = (0..ranks).map(|rank| DensePlan::for_rank_owned(&records, ranks, rank, &owners))
                .collect::<Result<Vec<_>, _>>()?;
            let shared = (0..ranks).map(|rank| (0..cfg.layers).filter(|&layer| !cfg.dense[layer])
                .map(|layer| metadata.plan(layer, ranks, rank, ExpertSet::Shared))
                .collect::<Result<Vec<_>, _>>()).collect::<Result<Vec<_>, _>>()?;
            process.native_load = Some(glmf::admission::NativeLoad::from_plans(&body, &vocabulary, &shared)?);
        }
        let selected = glmf::admission::prepare(&inputs, sms as usize, available.is_some(), process)?.admit()?;
        report.encoder = selected.encoder.clone();
        if report.encoder.as_ref().is_some_and(|encoder| encoder.kind == EncoderKind::Off) {
            if let Some(component) = report.components.iter_mut().find(|component|
                component.component == Component::Vision && component.status != Status::Unused) {
                report.disabled_media_bytes += component.bytes;
                component.bytes = 0;
                component.status = Status::Disabled;
            }
        }
        if report.audio != crate::plan::MediaMode::Off {
            let reason = towers.audio.err().unwrap_or_else(|| "Flash audio encoder unavailable".into());
            report.placement_supported = false;
            report.hints.push(crate::plan::Hint { what: format!("audio tower unavailable: {reason}"),
                how: "Use --audio off or add the bundled tensors/exporter".into() });
            report.audio_encoder = Some(EncoderPlacement { kind: EncoderKind::Off, weights: 0,
                scratch: 0, replicas: vec![], shortfall: 0, reason });
        }
        report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&selected.solved));
        report.host_budget_bytes = quota;
        notes.push(format!("unused graph bound by GPU: {:?} bytes", selected.graph_slack));
        let (placed, working) = (selected.placement, selected.working);
        let replicas = glmf::replicated_mla_bytes(&cfg, inputs.index, &placed)?;
        if replicas > 0 {
            waste.push(Waste { device: "rtx1".into(), what: "KV records replicated on both GPUs (MLA latent and index)".into(), bytes: replicas });
        }
        notes.push(format!("selected Spark ranks {}, prefill lanes {}, layers {:?}",
            working.spark_ranks, working.prefill_lanes, working.spark_layers));
        Ok((placed, working))
    })();
    let mut selected = None;
    match admitted {
        Ok((placed, working)) => {
            report.solver_placement = Some(placed.clone());
            pool = placed.pool_tokens;
            selected = Some(working);
            for (device, items) in devices.iter_mut().zip(&placed.items) {
                device.items = items.clone(); device.kv_tokens = pool;
            }
            notes.push(format!("placement: {}", placed.summary()));
        }
        Err(error) => {
            let reason = format!("GLM Flash pool-first placement: {error:#}");
            // `context` is the two-GPU default, so an absent program manifest fails the solve
            // before any pool or replay constraint applies: point at the programs to rebuild
            // rather than at the pool (issue #202).
            let how: String = if matches!(error.downcast_ref::<placement::PlacementError>(),
                Some(placement::PlacementError::MissingContextPrograms { .. })) {
                "Rebuild the GLM Flash coordinator programs with the context-split kernels \
                    (CUTEAFD_CONTEXT_SPLIT_AOT), or request `heads` explicitly.".into()
            } else {
                "Change the reported pool, replay-records or placement constraint.".into()
            };
            report.refuse_placement(reason.clone(), how);
            notes.push(reason);
        }
    }
    if let Some(working) = selected.filter(|working| working.spark_ranks > 0 && exlr8.is_none()) {
        let total = report.components.iter().filter(|c| c.component == Component::RoutedExpert).map(|c| c.bytes).sum::<u64>();
        let routed = model.spec().moe_layers().max(1);
        let remaining = total.saturating_sub(total / routed as u64 * working.rtx_layers.len() as u64);
        for rank in 0..working.spark_ranks {
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: options.spark_bytes, worker_budget_bytes: options.spark_bytes,
                kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed_expert", "", (remaining as f64 * report.spark_rank_share) as u64, Basis::Exact),
                    Item::new(Category::Runtime, "context+modules", "", 512 * MIB, Basis::Calibrated)] };
            let io = model.spec().moe.as_ref()
                .ok_or(placement::PlacementError::Inventory("Spark experts need routed geometry"))
                .and_then(|moe| {
                    let geometry = cuteafd_core::ExpertGeometry { hidden: model.spec().hidden as u32,
                        experts: moe.experts as u32, topk: moe.top_k as u32,
                        intermediate: moe.intermediate as u32, layers: model.spec().layers.len() as u32 };
                    u32::try_from(options.spark_capacity_rows)
                        .map_err(|_| placement::PlacementError::Overflow("Spark capacity"))
                        .and_then(|capacity| charge_spark_io(&mut device, &working, geometry, capacity,
                            checkpoint.spark_input_is_bf16(), spark_ring_geometry(spark_endpoints("glm5_flash", working.prefill_lanes)),
                            gib(56) * options.spark_capacity_rows / 4096))
                });
            if let Err(error) = io {
                report.placement_supported = false;
                notes.push(format!("Spark rank {rank} I/O admission: {error}"));
            }
            let charged = input.expert_catalog().and_then(SparkBackend::for_catalog).and_then(|backend|
                super::planned_spark_executions("glm5_flash", backend,
                    Some(u32::try_from(options.glmf_decode_rows)?), options))
                .and_then(|executions| spark_slots::charge_executions(&mut device, input, &working, "glm5_flash", &executions));
            if let Err(error) = charged {
                report.placement_supported = false;
                notes.push(format!("Spark rank {rank} execution owners: {error:#}"));
            }
            devices.push(device);
        }
    }
    if let Some(encoder) = &report.encoder {
        for rank in encoder.spark_ranks() {
            if let Some(device) = devices.iter_mut().find(|device|
                device.kind == DeviceKind::Spark && device.index as usize == rank) {
                device.items.push(Item::new(Category::Weights, "vision weights", "bundled encoder", encoder.weights, Basis::Exact));
                device.items.push(Item::new(Category::Workspace, "vision scratch", "bundled encoder", encoder.scratch, Basis::Formula));
            }
        }
        notes.push(format!("joint encoder {:?}: {} bytes admitted", encoder.kind, encoder.admitted_bytes()));
    }
    MemoryLayout { devices, pool_tokens: pool, waste, notes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::coordinator_programs::{AttentionReductionPlan, select_attention_plans, select_card_plans};
    use crate::serving_capacity::{GlmfCoreScratch, GlmfScratchOptions, GlmfStepShape,
        glmf_core_step_scratch, glmf_manifest_scratch, glmf_step_workspaces, glmf_table_pages};

    #[test]
    fn native_tp2_step_and_context_admission_match_selected_runtime_recipes() {
        let config = crate::plan::testing::glm5_flash_config(45);
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::write_safetensors(&dir.path().join("model.safetensors"), &[]);
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        let checkpoint = crate::plan::Checkpoint::open(dir.path()).unwrap();
        let mut names = vec!["glmf_mhc_pre".to_string()];
        for (cap, mode) in [("m64", "decode"), ("m4096", "prefill")] {
            for stem in ["mhc_post_pre", "kda", "mla_producer", "o", "ffn_i2048", "ffn_i12288",
                "index_producer", "index_producer_c"] {
                names.push(format!("glmf_{stem}_{cap}"));
            }
            names.push(format!("glmf_sparse_mla_{mode}_{cap}"));
            names.push(format!("glmf_index_topk_{mode}_{cap}"));
        }
        names.extend(glmf::CONTEXT_PROGRAMS.iter().map(|name| name.to_string()));
        names.push("glmf2_kda_commit_c".into());
        let mut programs: Vec<_> = names.into_iter().map(|name| serde_json::json!({
            "name": name, "scratch_bytes_at_capacity": {"scratch": 256}})).collect();
        for begin in [0, 32] {
            for (split, cap, scratch) in [(1, 8, 4096), (9, 64, 65_536)] {
                programs.push(serde_json::json!({"name": format!("glmf_sparse_mla_partial_h{begin}_s{split}_m64"),
                    "params": {"row_caps": {"170": [cap], "188": [cap]}},
                    "scratch_bytes_at_capacity": {"scratch": scratch}}));
            }
        }
        let sparse = programs.iter_mut().find(|p| p["name"] == "glmf_sparse_mla_decode_m64").unwrap();
        sparse["geometry"] = serde_json::json!({"attention_plan_capacity_rows":64,
            "attention_plan_scratch":[{"rows_cap":8,"scratch":8192},{"rows_cap":64,"scratch":131072}]});
        sparse["scratch_bytes_at_capacity"]["scratch"] = 131072.into();
        let raw = serde_json::json!({"programs":programs});
        let native = GlmfCoreScratch { kda: true, mla: true, ffn: true };
        for sms in [170, 188] {
            for (drafts, expected_context) in [(Some(7), 4096), (None, 4096), (Some(15), 65_536)] {
                let plan = AttentionReductionPlan::new(64, drafts, None).unwrap();
                let mut selected = raw.clone();
                select_card_plans(&mut selected, sms).unwrap();
                select_attention_plans(&mut selected, plan, |name| name.starts_with("glmf")).unwrap();
                let context = glmf::manifest_context_for_plan(&selected, cfg.heads, sms as usize, &plan).unwrap().unwrap();
                assert_eq!(context.partial_scratch, expected_context);
                let planned = glmf_step_workspace(&selected, &checkpoint, &ExpertPlacement::Local, 1,
                    4096, 1 << 20, 64, 0, true, true,
                    resident::GlmfRepresentation { kda_fp8: false, fp8_head: false, output_shard: false }, false, true, false,
                    crate::serving_capacity::GlmfKdaState::F32, Some(native)).unwrap();
                // Runtime narrows individual specs, independently of the planner's manifest adapter.
                let lookup = |name: &str| {
                    let entry = raw["programs"].as_array()?.iter().find(|p| p["name"] == name)?;
                    let mut scratch = serde_json::from_value(entry["scratch_bytes_at_capacity"].clone()).unwrap();
                    cuteafd_core::coordinator_programs::select_attention_scratch(&entry["geometry"], &mut scratch, plan).unwrap();
                    scratch.get("scratch").copied()
                };
                let options = GlmfScratchOptions { split: true, index_compact: true, ..Default::default() };
                let decode = glmf_core_step_scratch(lookup, &cfg, options, 64, true, native).unwrap();
                let prefill = glmf_core_step_scratch(lookup, &cfg, options, 4096, false, native).unwrap();
                let (table_pages, table_pool_pages) = glmf_table_pages(1 << 20);
                for rank in 0..2 {
                    let shape = GlmfStepShape { lead: rank == 0, split: true, local_experts: false,
                        tp2_experts: true, spark: false, partial_bytes: 2, output_shard: false,
                        full_prefill_logits: false, table_pages, table_pool_pages };
                    let allocated = glmf_step_workspaces(&cfg, 1, 4096, 64, &shape, decode, prefill).device_bytes();
                    let charged = planned.iter().filter(|d| d.demand.gpu as usize == rank)
                        .map(|d| d.demand.bytes).sum::<u64>();
                    assert_eq!(charged, allocated, "SM{sms} rank{rank} drafts={drafts:?}");
                }
                assert_eq!(glmf_manifest_scratch(&selected)("glmf_sparse_mla_decode_m64"), Some(decode.programs));
            }
        }
    }
}
