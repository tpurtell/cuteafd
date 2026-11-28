//! `cuteafd plan`: inspect a checkpoint and report what this build can serve.
use anyhow::{Context, Result};
use cuteafd_loader::plan::{budget_bytes, render, ExpertPlacement, PlanError, PlanOptions};
#[cfg(test)]
use cuteafd_loader::plan::testing::plan;
use cuteafd_loader::{default_hf_home, resolve_snapshot_at_revision};
use std::path::PathBuf;

use crate::cli::PlanArgs;

/// The coordinator GPU cap: `--gpu-memory-cap`, else the RTX PRO 6000's
/// CUDA-reported total (`placement::inventory::PRO_TOTAL_BYTES`), what the
/// runtime's own sample sees.
fn coordinator_bytes(args: &PlanArgs) -> u64 {
    args.memory.policy().gpu_cap.unwrap_or(cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES)
}

/// The planning options `args` name, validated before any checkpoint is read.
fn options(args: &PlanArgs) -> Result<PlanOptions, PlanError> {
    if !args.layout && args.exlr8_prefill_activations.is_some() {
        return Err(PlanError::InvalidOption { option: "--exlr8-prefill-activations", reason: "requires --layout to charge the lane records".into() });
    }
    if !args.layout && args.vision_replicas != 1 {
        return Err(PlanError::InvalidOption { option: "--vision-replicas", reason: "requires --layout to describe replica inventory".into() });
    }
    let mut options = PlanOptions {
        fp8_prefill: args.fp8_prefill.clone(),
        decode_lanes: args.decode_lanes,
        decode_rows: args.decode_rows.as_option(),
        draft_length: args.draft_length,
        attention_precision: args.attention.attention_precision,
        mapped_scale_preload_bytes: cuteafd_loader::plan::mapped_scale_preload_bytes(args.mapped_scale_preload_gib)?,
        mapped_scale_host_budget_bytes: None,
        draft_ownership: None,
        qwen_auto_mtp: false,
        attention_placement: cuteafd_loader::placement::attention::parse(&args.attention_placement)
            .map_err(|reason| PlanError::InvalidOption { option: "--attention-placement", reason })?,
        vision: args.vision,
        audio: args.audio,
        placement: ExpertPlacement::from_spark_ranks(args.spark_ranks.unwrap_or(args.available_sparks)),
        spark_budget_bytes: args.memory.spark_memory_cap,
        coordinator_budget_bytes: budget_bytes("--coordinator-weight-budget-gib", cuteafd_core::launch_settings::coordinator_weight_budget(args.coordinator_weight_budget_gib, None, coordinator_bytes(args)).value)?,
        coordinator_gpu_budget_bytes: coordinator_bytes(args),
        layout: args.layout.then(|| -> Result<_, PlanError> {
            if !(1..=2).contains(&args.rtx) {
                return Err(PlanError::InvalidOption { option: "--rtx", reason: "1 or 2 coordinator GPUs".into() });
            }
            Ok(cuteafd_loader::plan::layout::LayoutOptions {
                #[cfg(test)]
                host_memory: Some(cuteafd_loader::plan::testing::SimulatedHost::raptor().host_ram),
                host_cache_bytes: match args.host_cache_bytes {
                    crate::shared::prefix::HostBudget::Auto => None,
                    crate::shared::prefix::HostBudget::Bytes(bytes) => Some(bytes),
                },
                media_cache_bytes: args.media_cache_bytes,
                rtx_bytes: vec![coordinator_bytes(args); args.rtx],
                spark_allocation_budget_bytes: args.memory.spark_memory_cap,
                pool_tokens: args.pool_tokens,
                dense_qwen_kv: args.kv_format.map(|format| match format {
                    cuteafd_core::MlaCacheFormat::Bf16 => cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,
                    cuteafd_core::MlaCacheFormat::Fp8 => cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8,
                }).unwrap_or(cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8),
                qwen_kv: args.kv_format.map(|format| match format {
                    cuteafd_core::MlaCacheFormat::Bf16 => cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,
                    cuteafd_core::MlaCacheFormat::Fp8 => cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8,
                }).unwrap_or_default(),
                step5_kv: args.kv_format.map(|format| match format {
                    cuteafd_core::MlaCacheFormat::Bf16 => cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,
                    cuteafd_core::MlaCacheFormat::Fp8 => cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8,
                }).unwrap_or_default(),
                inkling_kv: args.kv_format.map(|format| match format {
                    cuteafd_core::MlaCacheFormat::Bf16 => cuteafd_loader::families::inkling2::InklingKvCache::Bf16,
                    cuteafd_core::MlaCacheFormat::Fp8 => cuteafd_loader::families::inkling2::InklingKvCache::Fp8,
                }).unwrap_or_default(),
                muse_kv: args.kv_format.map(|format| match format {
                    cuteafd_core::MlaCacheFormat::Bf16 => cuteafd_loader::families::muse_spark::cache::MuseKvCache::Bf16,
                    cuteafd_core::MlaCacheFormat::Fp8 => cuteafd_loader::families::muse_spark::cache::MuseKvCache::Fp8,
                }).unwrap_or_default(),
                mla_kv: args.kv_format.unwrap_or_default(),
                qwen_representation: cuteafd_loader::families::qwen4::resident::Qwen4Representation {
                    fp8_projections: args.fp8_decode, fp8_head: args.mtp_fp8_head },
                qwen_attention_plan_rows: args.attention_plan_rows,
                vision_replicas: args.vision_replicas as usize,
                host_embedding: args.embedding_placement == Some(crate::shared::token_io::EmbedPlacement::Host),
                force_gpu_embedding: args.embedding_placement == Some(crate::shared::token_io::EmbedPlacement::Gpu),
                onboard: args.rtx_expert_layers,
                exchange_f32: args.exchange_f32,
                context_tokens: args.context_tokens.unwrap_or(0),
                prefill_rows: args.prefill_rows,
                bf16_target_head: args.target_head.unwrap_or_else(crate::shared::target_head::TargetHead::configured).blas_head(true),
                expert_usage: args.expert_usage.unwrap_or(true),
                // Lane families still read prefill_lanes until their unit ports land.
                prefill_lanes: args.prefill_waves,
                prefill_waves: args.prefill_waves.max(1),
                exlr8_prefill_activations: args.exlr8_prefill_activations.unwrap_or_default(),
                exlr8_prefill_pin: args.exlr8_prefill_activations,
                prefill_span_rows: (args.prefill_span_rows > 0).then_some(args.prefill_span_rows),
                glmf_representation: cuteafd_loader::families::glm5_flash::resident::GlmfRepresentation {
                    kda_fp8: args.kda_fp8 != crate::families::glm5_flash::fp8::KdaFp8::Off,
                    head: args.head_format, ..Default::default() },
                glmf_kda_state: args.kda_state.into(),
                glmf_index: args.index_cache.unwrap_or(crate::families::glm5_flash::engine::IndexCache::Keys).into(),
                glmf_pool_marks: args.prefix_marks == crate::families::glm5_flash::prefix::PrefixMarks::Pool,
                glmf_shared_replay: args.replay_records == Some(crate::cli::PlanReplayRecords::Shared),
                glmf_auto_replay: args.replay_records == Some(crate::cli::PlanReplayRecords::Auto),

                concurrency: args.concurrency,
                tp2_dspark_experts: args.tp2_dspark_experts,
                tp2_dflash: args.tp2_dflash.unwrap_or(false),
                state_slots: args.state_slots,
                glmf_mark_lanes: args.mark_lanes,
                prefix_slots: args.prefix_slots,
                mimo_prefix_entries: args.prefix_cache_entries.unwrap_or(u64::MAX),
                mimo_prefix_mark_bytes: args.prefix_cache_mark_mib.checked_mul(1 << 20)
                    .ok_or_else(|| PlanError::InvalidOption { option: "--prefix-cache-mark-mib", reason: "byte budget overflows".into() })?,
                mimo_prefix_draft: args.mimo_prefix_draft,
                mimo_rings: args.mimo_rings,
                mimo_swa_kv: args.mimo_swa_kv_cache.into(),
                mimo_full_kv: args.mimo_kv_cache.into(),
                mimo_expert_input: args.expert_input.into(),
                draft_sequences: args.draft_sequences,
                draft_kv_cache: args.draft_kv_cache,
                draft_context_window: args.draft_context_window,
                draft_context_slots: args.draft_context_slots,
                decode_graphs: args.decode_graphs == "on",
                physical_sms: args.physical_sms,
                mimo_draft_bf16: args.mimo_draft_bf16,
                mimo_expert_manifest: args.mimo_expert_manifest.clone(),
                native_mtp_layers: args.native_mtp_layers,
                workspace_manifest: args.workspace_manifest.clone(),
                glmf_drafter_disabled: args.no_speculator || args.speculator == "off",
                glmf_drafter_snapshot: args.drafter_snapshot.clone(),
                glm5_drafter_disabled: args.no_speculator || args.speculator == "off",
                glm5_drafter_snapshot: args.drafter_snapshot.clone(),
                glm5_draft_bf16: args.glm5_draft_bf16 || args.draft_fp8 == Some(false),
                kimi_drafter_snapshot: None,
                muse_drafter_snapshot: None,
                muse_partial_bf16_max_rows: args.expert_partial_bf16_max_rows,
                dense_qwen_drafter_snapshot: args.drafter_snapshot.clone().filter(|_| args.speculator != "off"),
                kimi_draft_fp8: args.draft_fp8.unwrap_or(false),
                glmf_draft_representation: cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::from_fp8_option(
                    args.draft_fp8),
                glmf_draft_fp8_mode: args.draft_linear.unwrap_or_default().code() as u8,
                glm5_attention_plan_rows: args.attention_plan_rows,
                drafter_bytes: if args.drafter_gib > 0.0 { budget_bytes("--drafter-gib", args.drafter_gib)? } else { 0 },
                ..Default::default()
            })
        }).transpose()?,
        ..Default::default()
    };
    #[cfg(test)]
    { options.inventory_host_memory = args.test_host_memory; }
    if let Some(layout) = &mut options.layout {
        if args.memory.spark_memory_cap.is_none() {
            let invalid = |error: anyhow::Error| PlanError::InvalidOption {
                option: "Spark memory inventory", reason: error.to_string() };
            use cuteafd_loader::serving_capacity::spark_memory;
            let expert_ranks = args.spark_ranks.unwrap_or(usize::MAX);
            layout.spark_rank_bytes = if let Some(samples) = &args.spark_memory_samples {
                // Ranks past the requested expert world host media only: no
                // routed-expert arena, so no display-carveout credit (#2546).
                spark_memory::sample_bytes_for(samples, expert_ranks).map_err(invalid)?
            } else {
                let cfg = launch_config(args.config.as_deref()).map_err(invalid)?;
                let hosts = configured_spark_hosts(&cfg).into_iter()
                    .take(args.spark_ranks.unwrap_or(args.available_sparks)).collect::<Vec<_>>();
                if args.spark_ranks == Some(0) {
                    Vec::new()
                } else if hosts.iter().any(|host| host.starts_with("REPLACE-ME")) {
                    spark_memory::serving_bytes_for(expert_ranks)
                } else if !hosts.is_empty() {
                    hosts.iter().enumerate().map(|(rank, host)|
                        spark_memory::measured_bytes_for(host, rank, expert_ranks))
                        .collect::<anyhow::Result<Vec<_>>>().map_err(invalid)?
                } else if !args.deployment_peers.is_empty() {
                    args.deployment_peers.iter().take(args.spark_ranks.unwrap_or(args.available_sparks))
                        .enumerate().map(|(rank, peer)| spark_memory::measured_bytes_for(peer, rank, expert_ranks))
                        .collect::<anyhow::Result<Vec<_>>>().map_err(invalid)?
                } else {
                    spark_memory::serving_bytes_for(expert_ranks)
                }
            };
        }
    }
    options.apply_spark_budget();
    options.validate()?;
    Ok(options)
}

fn configured_spark_hosts(cfg: &std::collections::BTreeMap<String, String>) -> Vec<String> {
    if let Some(hosts) = cfg.get("SPARK_HOSTS").filter(|hosts| !hosts.is_empty()) {
        return hosts.split(',').map(str::to_owned).collect();
    }
    (0..8).map_while(|rank| cfg.get(&format!("SPARK_{rank}_HOST"))
        .filter(|host| !host.is_empty()).cloned()).collect()
}

type ResolvedSettings = std::collections::BTreeMap<String, cuteafd_core::launch_settings::Resolved<serde_json::Value>>;

fn setting<T: serde::Serialize>(value: cuteafd_core::launch_settings::Resolved<T>) -> cuteafd_core::launch_settings::Resolved<serde_json::Value> {
    cuteafd_core::launch_settings::Resolved { value: serde_json::to_value(value.value).expect("setting serialization"), provenance: value.provenance }
}

fn resolve_settings(args: &mut PlanArgs, family: &str, cfg: &std::collections::BTreeMap<String, String>) -> Result<ResolvedSettings> {
    use cuteafd_core::launch_settings as defaults;
    use clap::ValueEnum;
    let mut resolved = ResolvedSettings::new();
    let budget = defaults::coordinator_weight_budget(args.coordinator_weight_budget_gib, None, coordinator_bytes(args));
    args.coordinator_weight_budget_gib = Some(budget.value);
    resolved.insert("coordinator_weight_budget_gib".into(), setting(budget));
    if family == "glm5" {
        if args.draft_fp8.is_none() {
            args.draft_fp8 = match cfg.get("SPECULATOR_FP8").map(String::as_str) {
                Some("off") => Some(false), Some("on") => Some(true), None | Some("") | Some("auto") => None,
                _ => anyhow::bail!("SPECULATOR_FP8 must be auto, on or off"),
            };
        }
        args.glm5_draft_bf16 |= args.draft_fp8 == Some(false);
    }
    if family == "glm5_flash" {
        let configured = |key: &str| cfg.get(key).map(String::as_str);
        let index = defaults::glmf_index_cache(args.index_cache.map(|v| v.to_possible_value().unwrap().get_name().to_owned()).as_deref(),
            configured("GLM5_FLASH_INDEX_CACHE"), args.rtx);
        args.index_cache = Some(crate::families::glm5_flash::engine::IndexCache::from_str(&index.value, false).map_err(|e| anyhow::anyhow!("GLM5_FLASH_INDEX_CACHE must be keys or compact: {e}"))?);
        resolved.insert("index_cache".into(), setting(index));
        let linear = defaults::glmf_draft_linear(args.draft_linear.map(|v| v.to_possible_value().unwrap().get_name().to_owned()).as_deref(),
            configured("GLM5_FLASH_DRAFT_LINEAR"), args.speculator != "off" && args.draft_fp8.unwrap_or(configured("SPECULATOR_FP8") != Some("off")));
        args.draft_linear = Some(crate::shared::fp8_linear::Fp8Rows::from_str(&linear.value, false).map_err(|e| anyhow::anyhow!("GLM5_FLASH_DRAFT_LINEAR must be w8a16, wide or w8a8: {e}"))?);
        resolved.insert("draft_linear".into(), setting(linear));
        if args.draft_fp8.is_none() {
            args.draft_fp8 = match configured("SPECULATOR_FP8") {
                Some("off") => Some(false), Some("on") => Some(true), None | Some("") | Some("auto") => None,
                _ => anyhow::bail!("SPECULATOR_FP8 must be auto, on or off"),
            };
        }
        if args.replay_records.is_none() {
            args.replay_records = Some(match configured("GLM5_FLASH_REPLAY_RECORDS") {
                None | Some("") | Some("auto") => crate::cli::PlanReplayRecords::Auto,
                Some("own") => crate::cli::PlanReplayRecords::Own,
                Some("shared") => crate::cli::PlanReplayRecords::Shared,
                _ => anyhow::bail!("GLM5_FLASH_REPLAY_RECORDS must be auto, own or shared"),
            });
        }
        anyhow::ensure!(args.rtx == 1 || args.replay_records != Some(crate::cli::PlanReplayRecords::Shared),
            "GLM5_FLASH_REPLAY_RECORDS=shared keeps the records in one GPU's prefill scratch; serve it without a head split");
    }
    if family == "kimi_k3" {
        if args.draft_fp8.is_none() {
            args.draft_fp8 = match cfg.get("SPECULATOR_FP8").map(String::as_str) {
                Some("on") => Some(true), Some("off") => Some(false), None | Some("") | Some("auto") => None,
                _ => anyhow::bail!("SPECULATOR_FP8 must be auto, on or off"),
            };
        }
        let native_default = args.speculator == "dspark" && kimi_qualification(args, cfg).qualified();
        let fp8 = defaults::selected(args.draft_fp8, None,
            args.speculator == "dflash2" || native_default,
            "FP8 for content-paired K3.1 block DSpark in every view and explicit Kimi DFlash2");
        args.draft_fp8 = Some(fp8.value);
        resolved.insert("draft_fp8".into(), setting(fp8));
    }
    if family == "qwen4" {
        let explicit = args.explicit_settings.contains("native_mtp_layers").then_some(args.native_mtp_layers);
        let configured = cfg.get("SPECULATOR_DEPTH").filter(|value| !value.is_empty())
            .map(|value| value.parse().context("SPECULATOR_DEPTH")).transpose()?;
        let enabled = matches!(args.speculator.as_str(), "auto" | "mtp");
        let depth = defaults::selected(explicit, configured, if enabled { 3 } else { 0 },
            "three recursive Qwen drafts when native MTP is selected; zero when disabled");
        args.native_mtp_layers = depth.value;
        resolved.insert("native_mtp_layers".into(), setting(depth));
    }
    if family == "mimo_v2" {
        let configured = match cfg.get("SPECULATOR_FP8").map(String::as_str) {
            Some("on" | "auto") => Some(true), Some("off") => Some(false), None | Some("") => None,
            _ => anyhow::bail!("SPECULATOR_FP8 must be auto, on or off"),
        };
        let fp8 = defaults::selected(args.draft_fp8.or(args.mimo_draft_bf16.then_some(false)),
            configured, true, "FP8 MiMo DFlash body for unset/auto/on; off keeps BF16");
        args.draft_fp8 = Some(fp8.value);
        args.mimo_draft_bf16 = !fp8.value;
        resolved.insert("draft_fp8".into(), setting(fp8));
        let depth = defaults::mimo_mtp_depth((args.explicit_settings.contains("native_mtp_layers")).then_some(args.native_mtp_layers),
            cfg.get("SPECULATOR_DEPTH").filter(|v| !v.is_empty()).map(|v| v.parse().context("SPECULATOR_DEPTH")).transpose()?, args.speculator == "mtp");
        args.native_mtp_layers = depth.value;
        resolved.insert("native_mtp_layers".into(), setting(depth));
    }
    Ok(resolved)
}

fn resolved_output(mut value: serde_json::Value, args: &PlanArgs, mut settings: ResolvedSettings,
    cfg: &std::collections::BTreeMap<String, String>, replay_explicit: Option<crate::cli::PlanReplayRecords>) -> serde_json::Value {
    value["spark_memory_measurements"] = if let Some(bytes) = args.memory.spark_memory_cap {
        serde_json::json!({"source": "explicit --spark-memory-cap", "bytes": bytes})
    } else { args.spark_memory_samples.as_deref()
        .and_then(|text| serde_json::from_str(text).ok())
        .unwrap_or_else(|| serde_json::json!({"source": if configured_spark_hosts(cfg).is_empty() || configured_spark_hosts(cfg).iter().any(|host| host.starts_with("REPLACE-ME")) { "simulated canonical inventory" } else { "checked-in per-host idle MemAvailable inventory" },
            "inventory": serde_json::from_str::<serde_json::Value>(cuteafd_loader::serving_capacity::spark_memory::INVENTORY).expect("inventory")})) };
    if value.get("family").and_then(serde_json::Value::as_str) == Some("glm5_flash") {
        let report = value.get("report").unwrap_or(&value);
        let sparks = report.get("spark_expert_world").and_then(serde_json::Value::as_u64)
            .unwrap_or_else(|| report.get("spark_ranks").and_then(serde_json::Value::as_u64).unwrap_or(0));
        let spelling = |v: crate::cli::PlanReplayRecords| match v { crate::cli::PlanReplayRecords::Auto => "auto", crate::cli::PlanReplayRecords::Own => "own", crate::cli::PlanReplayRecords::Shared => "shared" };
        let mut replay = cuteafd_core::launch_settings::glmf_replay_records(replay_explicit.map(spelling),
            cfg.get("GLM5_FLASH_REPLAY_RECORDS").map(String::as_str), args.rtx,
            args.pool_tokens.filter(|&n| n > 0).is_none(), sparks > 0);
        if let Some(actual) = report.get("replay_records").and_then(serde_json::Value::as_str) { replay.value = actual.into(); }
        settings.insert("replay_records".into(), setting(replay));
    }
    value["resolved_settings"] = serde_json::to_value(settings).expect("settings serialization");
    let report = value.get("report").unwrap_or(&value);
    let owner_items = report.pointer("/memory_layout/devices").and_then(serde_json::Value::as_array)
        .into_iter().flatten().map(|device| serde_json::json!({
            "kind": device["kind"], "index": device["index"],
            "owners": device["items"].as_array().into_iter().flatten()
                .filter(|item| item["category"] == "runtime" || item["category"] == "reserved")
                .cloned().collect::<Vec<_>>() })).collect::<Vec<_>>();
    value["planning_inputs"] = serde_json::json!({
        "device_capacity_bytes": coordinator_bytes(args),
        "device_capacity_source": if args.memory.policy().gpu_cap.is_some() { "explicit: --gpu-memory-cap" } else { "simulated: RTX PRO 6000 CUDA total" },
        "measured_owners": "planner estimates; no live CUDA/context or fabric probe",
        "runtime_owner_items": owner_items,
        "physical_sms": args.physical_sms,
    });
    value
}

fn print_settings(value: &serde_json::Value) {
    if let Some(settings) = value["resolved_settings"].as_object() {
        println!("\nresolved launch settings");
        for (name, resolved) in settings { println!("  {name}: {} ({})", resolved["value"], resolved["provenance"].as_str().unwrap_or("")); }
    }
    println!("planning inputs: {}", value["planning_inputs"]);
}

fn resolve_config(args: &mut PlanArgs, cfg: &std::collections::BTreeMap<String, String>) -> Result<()> {
    let get = |key: &str| cfg.get(key).filter(|v| !v.is_empty());
    macro_rules! configured {
        ($field:ident, $key:literal) => {
            if !args.explicit_settings.contains(stringify!($field)) {
                if let Some(value) = get($key) { args.$field = value.parse().with_context(|| $key)?; }
            }
        };
    }
    if !args.explicit_settings.contains("rtx") {
        if let Some(value) = get("RTX_GPUS").filter(|v| *v != "auto") { args.rtx = value.parse().context("RTX_GPUS")?; }
    }
    configured!(attention_placement, "ATTENTION_PLACEMENT");
    configured!(concurrency, "CONCURRENCY");
    configured!(draft_context_window, "DRAFT_CONTEXT_WINDOW");
    configured!(prefill_waves, "PREFILL_WAVES");
    if args.decode_lanes.is_none() {
        args.decode_lanes = get("DECODE_LANES").map(|v| v.parse().context("DECODE_LANES")).transpose()?;
    }
    if !args.explicit_settings.contains("decode_rows") {
        if let Some(value) = get("DECODE_ROWS") {
            args.decode_rows = value.parse()
                .map_err(|error| anyhow::anyhow!("DECODE_ROWS: {error}"))?;
        }
    }
    if args.draft_length.is_none() {
        args.draft_length = get("DRAFT_LENGTH").filter(|v| *v != "auto")
            .map(|v| v.parse().context("DRAFT_LENGTH")).transpose()?;
    }
    if !args.explicit_settings.contains("decode_graphs") {
        if let Some(value) = get("DECODE_GRAPHS") {
            anyhow::ensure!(matches!(value.as_str(), "on" | "off"), "DECODE_GRAPHS must be on or off");
            args.decode_graphs = value.clone();
        }
    }
    if !args.explicit_settings.contains("fp8_prefill") {
        if let Some(value) = get("FP8_PREFILL") { args.fp8_prefill = value.parse().map_err(anyhow::Error::msg)?; }
    }
    if args.memory.spark_memory_cap.is_none() {
        args.memory.spark_memory_cap = get("SPARK_MEMORY_CAP").map(|v| v.parse::<cuteafd_core::memory_policy::Size>().map(|v| v.0)).transpose()?;
    }
    if args.memory.gpu_memory_cap.is_none() {
        args.memory.gpu_memory_cap = get("GPU_MEMORY_CAP").map(|v| v.parse::<cuteafd_core::memory_policy::Size>().map(|v| v.0)).transpose()?;
    }
    if args.pool_tokens.is_none() { args.pool_tokens = get("POOL_TOKENS").filter(|v| *v != "auto").map(|v| v.parse().context("POOL_TOKENS")).transpose()?; }
    if args.exlr8_manifest.is_none() { args.exlr8_manifest = get("EXLR8_MANIFEST").cloned(); }
    if args.exlr8_prefill_activations.is_none() { args.exlr8_prefill_activations = get("EXLR8_PREFILL_ACTIVATIONS").map(|v| v.parse().context("EXLR8_PREFILL_ACTIVATIONS")).transpose()?; }
    if args.chat_template_from.is_none() { args.chat_template_from = get("CHAT_TEMPLATE_FROM").cloned(); }
    if !args.explicit_settings.contains("attention_precision") {
        if let Some(value) = get("ATTENTION_PRECISION") { args.attention.attention_precision = value.parse()?; }
    }
    if !args.explicit_settings.contains("core_view") {
        if let Some(value) = get("CORE_VIEW") { args.attention.core_view = value.parse()?; }
    }
    if !args.explicit_settings.contains("vision") { if let Some(value) = get("VISION") { args.vision = value.parse().map_err(anyhow::Error::msg)?; } }
    resolve_plan_audio_config(args, cfg)?;
    if !args.explicit_settings.contains("host_cache_bytes") {
        if let Some(value) = get("HOST_CACHE_BYTES") { args.host_cache_bytes = value.parse().map_err(anyhow::Error::msg)?; }
    }
    if args.rtx_expert_layers.is_none() { args.rtx_expert_layers = get("RTX_EXPERT_LAYERS").filter(|v| *v != "auto").map(|v| v.parse().map_err(anyhow::Error::msg)).transpose()?; }
    Ok(())
}

fn resolve_plan_audio_config(args: &mut PlanArgs, cfg: &std::collections::BTreeMap<String, String>) -> Result<()> {
    if !args.explicit_settings.contains("audio") {
        if let Some(value) = cfg.get("AUDIO").filter(|value| !value.is_empty()) {
            args.audio = value.parse().map_err(|error| anyhow::anyhow!("AUDIO: {error}"))?;
        }
    }
    Ok(())
}

fn resolve_family_config(args: &mut PlanArgs, family: &str, cfg: &std::collections::BTreeMap<String, String>) -> Result<()> {
    use clap::ValueEnum;
    let get = |key: &str| cfg.get(key).filter(|v| !v.is_empty());
    if family == "deepseek_v41" { resolve_v41_audio(args, cfg)?; }
    if family == "inkling2" {
        if !args.explicit_settings.contains("draft_kv_cache") {
            if let Some(value) = get("DRAFT_KV_CACHE") { args.draft_kv_cache = value.parse()?; }
        }
        anyhow::ensure!(!args.draft_fp8.unwrap_or(false)
            && matches!(get("SPECULATOR_FP8").map(String::as_str), None | Some("auto" | "off")),
            "Inkling SPECULATOR_FP8 must be auto or off (checkpoint-native drafter)");
    }
    let glm = matches!(family, "glm5" | "mistral_large4");
    if glm {
        anyhow::ensure!(get("PREFILL_BATCH_TOKENS").is_none(), "unused configuration key: PREFILL_BATCH_TOKENS");
        if let Some(value) = get("MEDIA_CACHE_BYTES").filter(|_| args.media_cache_bytes.is_none()) {
            args.media_cache_bytes = Some(crate::shared::prefix::parse_bytes(value).map_err(anyhow::Error::msg)?);
        }
    }
    if !args.explicit_settings.contains("prefill_rows") {
        if family == "qwen3_5" { args.prefill_rows = 4096; }
        else {
            let value = match family {
                "step5" | "deepseek_v41" => get("PREFILL_BATCH_TOKENS"),
                "mimo_v2" | "kimi_k3" | "muse_spark" | "minimax_m3" | "glm5" | "mistral_large4" => get("PREFILL_ROWS"),
                _ => get("PREFILL_ROWS").or_else(|| get("PREFILL_BATCH_TOKENS")),
            };
            if let Some(value) = value { args.prefill_rows = value.parse().context("family prefill rows")?; }
        }
    }
    if glm && !args.explicit_settings.contains("prefill_span_rows") {
        if let Some(value) = get("PREFILL_SPAN_ROWS") { args.prefill_span_rows = value.parse().context("PREFILL_SPAN_ROWS")?; }
    }
    if family == "mimo_v2" {
        if !args.explicit_settings.contains("mimo_kv_cache") {
            if let Some(value) = get("KV_CACHE") { args.mimo_kv_cache = crate::families::mimo_v2::KvCacheArg::from_str(value, false).map_err(anyhow::Error::msg)?; }
        }
    } else if matches!(family, "qwen4" | "qwen3_5" | "kimi_k3" | "step5" | "muse_spark" | "inkling2") && args.kv_format.is_none() {
        args.kv_format = get("KV_CACHE").map(|v| v.parse().map_err(anyhow::Error::msg)).transpose()?;
    }
    if matches!(family, "qwen4" | "step5") && !args.explicit_settings.contains("native_mtp_layers") {
        if let Some(value) = get("SPECULATOR_DEPTH") { args.native_mtp_layers = value.parse().context("SPECULATOR_DEPTH")?; }
    }
    Ok(())
}

fn resolve_v41_audio(args: &mut PlanArgs, cfg: &std::collections::BTreeMap<String, String>) -> Result<()> {
    use cuteafd_loader::plan::MediaMode;
    let configured = cfg.get("AUDIO").is_some_and(|value| !value.is_empty());
    let requested = (configured || args.explicit_settings.contains("audio") || args.audio != MediaMode::Auto)
        .then_some(args.audio);
    args.audio = cuteafd_loader::plan::v41_audio(requested)?;
    Ok(())
}

fn resolve_context_tokens(args: &mut PlanArgs, cfg: &std::collections::BTreeMap<String, String>) -> Result<()> {
    if args.context_tokens.is_none() {
        args.context_tokens = cfg.get("MAX_CONTEXT_TOKENS").filter(|value| !value.is_empty())
            .map(|value| crate::shared::context::parse_tokens::<u64>(value)
                .map_err(|error| anyhow::anyhow!("MAX_CONTEXT_TOKENS: {error}"))).transpose()?;
    }
    Ok(())
}

fn resolve_expert_usage(args: &mut PlanArgs, cfg: &std::collections::BTreeMap<String, String>) -> Result<()> {
    let configured = match cfg.get("EXPERT_USAGE").map(String::as_str) {
        None | Some("") | Some("on") => true,
        Some("off") => false,
        Some(_) => anyhow::bail!("EXPERT_USAGE must be on or off"),
    };
    args.expert_usage = Some(args.expert_usage.unwrap_or(configured));
    Ok(())
}

fn resolve_mimo_drafter(layout: &mut cuteafd_loader::plan::layout::LayoutOptions, speculator: &str) {
    layout.mimo_drafter_disabled = speculator != "dflash2";
}

fn v41_layout(layout: &mut cuteafd_loader::plan::layout::LayoutOptions, args: &PlanArgs) {
    // Absent: the loader sizes the arena with serving's `MarkRule` once dSpark is resolved.
    layout.prefix_slots = args.prefix_slots;
    // Workers must cover both prefill and the 256-row replay bucket.
    layout.spark_capacity_rows = if args.prefill_rows == 0 { 4096 }
        else if args.prefill_rows <= 256 { 256 }
        else if args.prefill_rows <= 1024 { 1024 }
        else { 4096 };
}

fn tp2_dspark_experts(args: &PlanArgs, family: &str,
    cfg: &std::collections::BTreeMap<String, String>) -> Result<bool> {
    let applicable = args.layout && matches!(family, "deepseek_v41" | "deepseek_v4") && args.rtx == 2 && args.speculator == "dspark";
    let enabled = args.tp2_dspark_experts || match cfg.get("TP2_DSPARK_EXPERTS").map(String::as_str) {
        None | Some("") => applicable,
        Some("off") => false,
        Some("on") => true,
        Some(_) => anyhow::bail!("TP2_DSPARK_EXPERTS must be on or off"),
    };
    anyhow::ensure!(!enabled || applicable,
        "--tp2-dspark-experts requires V4/V4.1, --layout, --rtx 2 and dSpark");
    Ok(enabled)
}

fn tp2_dflash(args: &PlanArgs, family: &str,
    cfg: &std::collections::BTreeMap<String, String>) -> Result<bool> {
    // Spark-range plans price both GPU counts; its one-GPU column clears TP2.
    let split = args.rtx == 2 || (matches!(family, "glm5" | "glm5_flash" | "mistral_large4") && args.spark_range);
    let applicable = if family == "mistral_large4" {
        args.layout && split && args.speculator == "dspark"
    } else {
        let draft_compatible = if matches!(family, "hy4" | "hy4_flash") {
            args.speculator == "dfly2"
        } else {
            args.speculator == "dflash2"
        };
        crate::cli::tp2_dflash_default(family, args.layout && split, draft_compatible)
    };
    let default_on = applicable
        && (!matches!(family, "glm5" | "glm5_flash" | "mistral_large4") || args.drafter_gib == 0.0);
    let configured = match cfg.get("TP2_DFLASH").map(String::as_str) {
        None | Some("") => default_on,
        Some("off") => false,
        Some("on") => true,
        Some(_) => anyhow::bail!("TP2_DFLASH must be on or off"),
    };
    let enabled = if family == "kimi_k3" && args.speculator == "dflash2" {
        let requested = args.tp2_dflash.or_else(|| cfg.get("TP2_DFLASH").filter(|v| !v.is_empty()).map(|v| v == "on"));
        match &args.drafter_snapshot {
            Some(snapshot) => cuteafd_loader::families::kimi_k3::draft::resolve_context_split(snapshot, requested, default_on)?,
            None => args.tp2_dflash.unwrap_or(configured),
        }
    } else { args.tp2_dflash.unwrap_or(configured) };
    anyhow::ensure!(!enabled || applicable,
        "TP2_DFLASH requires GLM 5.3/MiMo/Kimi/GLM Flash/Muse with DFlash2, Mistral Large 4 with dSpark, or Hy4-flash with DFly2, --layout, and --rtx 2");
    anyhow::ensure!(!enabled || !matches!(family, "glm5" | "glm5_flash" | "mistral_large4") || args.drafter_gib == 0.0,
        "TP2 DFlash requires checkpoint-derived physical owners; --drafter-gib forces whole placement");
    Ok(enabled)
}

fn resolve_exlr8_draft(args: &mut PlanArgs, snapshot: &std::path::Path, family: &str,
    cfg: &std::collections::BTreeMap<String, String>, options: &mut PlanOptions) -> Result<()> {
    resolve_speculator(args, snapshot, family, cfg)?;
    let split = tp2_dflash(args, family, cfg)?;
    if let Some(layout) = &mut options.layout {
        layout.tp2_dflash = split;
        if family == "glm5" { layout.glm5_drafter_snapshot = args.drafter_snapshot.clone(); }
        if family == "glm5_flash" { layout.glmf_drafter_snapshot = args.drafter_snapshot.clone(); }
    }
    Ok(())
}

fn validate_glmf_precision(args: &PlanArgs, family: &str) -> Result<(), PlanError> {
    use crate::families::glm5_flash::head::GlmfHeadFormat;
    if family == "glm5_flash" && args.target_head == Some(crate::shared::target_head::TargetHead::Tensor)
        && args.head_format != GlmfHeadFormat::Bf16 {
        return Err(PlanError::InvalidOption { option: "--target-head", reason: format!(
            "explicit tensor head needs --fp8-head false; the {} head has its own GEMV", args.head_format.name()) });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn run_plan(mut args: PlanArgs) -> Result<()> {
    args.test_host_memory = Some(cuteafd_loader::plan::testing::SimulatedHost::raptor().host_ram);
    run_plan_with_host(args)
}

#[cfg(not(test))]
pub(crate) fn run_plan(args: PlanArgs) -> Result<()> { run_plan_with_host(args) }

fn run_plan_with_host(mut args: PlanArgs) -> Result<()> {
    if args.spark_range { args.layout = true; }
    if args.deployment {
        args.layout = true;
        if let Some(value) = &args.kv_pool_size {
            value.parse::<crate::shared::memory::budget::ByteSize>()?;
        }
    }
    if args.config.is_some() {
        let cfg = launch_config(args.config.as_deref())?;
        resolve_config(&mut args, &cfg)?;
        resolve_expert_usage(&mut args, &cfg)?;
        resolve_context_tokens(&mut args, &cfg)?;
        args.available_sparks = (0..8).take_while(|rank|
            cfg.get(&format!("SPARK_{rank}_HOST")).is_some_and(|host| !host.is_empty())).count();
        if args.spark_ranks.is_none() {
            let count = cfg.get("SPARK_COUNT").filter(|v| !v.is_empty())
                .map(|v| v.parse::<usize>().context("SPARK_COUNT")).transpose()?;
            let hosts = spark_hosts_count(&cfg)?;
            anyhow::ensure!(count.zip(hosts).is_none_or(|(count, hosts)| count == hosts),
                "SPARK_HOSTS must name exactly SPARK_COUNT hosts");
            args.spark_ranks = count.or(hosts);
        }
    }
    {
        let cfg = launch_config(args.config.as_deref())?;
        if args.revision.is_none() && !PathBuf::from(&args.model).is_dir() {
            args.revision = cfg.get("MODEL_REVISION").filter(|v| !v.is_empty()).cloned();
        }
    }
    let snapshot = if PathBuf::from(&args.model).is_dir() {
        PathBuf::from(&args.model)
    } else {
        let hf_home = args.hf_home.clone().unwrap_or_else(default_hf_home);
        let resolved = match resolve_snapshot_at_revision(&args.model, Some(&hf_home), args.revision.as_deref()) {
            Ok(resolved) => resolved,
            // A stale main ref: describe the only snapshot present (serving still refuses it).
            Err(error) if args.revision.is_none() => {
                let snapshots = cuteafd_loader::model_cache_dir(&hf_home, &args.model).join("snapshots");
                let present: Vec<PathBuf> = std::fs::read_dir(&snapshots).map(|entries| {
                    entries.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect()
                }).unwrap_or_default();
                let [only] = present.as_slice() else { return Err(error) };
                eprintln!("note: {error:#}; describing {}", only.display());
                let revision = only.file_name().and_then(|n| n.to_str()).context("snapshot name")?.to_owned();
                resolve_snapshot_at_revision(&args.model, Some(&hf_home), Some(&revision))?
            }
            Err(error) => return Err(error),
        };
        resolved
            .snapshot_path
            .with_context(|| format!("no snapshot of {} under {}", args.model, hf_home.display()))?
    };
    // Settings-only performs one config read, never checkpoint selection or tensor I/O.
    let view = (!args.resolve_settings_only).then(|| args.attention.select(&snapshot)).transpose()?;
    let family_config = match &view {
        Some(view) => view.config()?,
        None => cuteafd_loader::read_json_file::<serde_json::Value>(&snapshot.join("config.json"))?,
    };
    let cfg = launch_config(args.config.as_deref())?;
    let family = if args.resolve_settings_only && (family_config["model_type"] == "glm5_next"
        || family_config["architectures"].as_array().is_some_and(|a| a.iter().any(|v| v == "Glm5NextForConditionalGeneration"))) {
        "glm5_flash"
    } else if args.resolve_settings_only && family_config["model_type"].as_str().is_some_and(|v| v.starts_with("mimo")) {
        "mimo_v2"
    } else if !args.resolve_settings_only && cuteafd_loader::plan::exlr8::recognizes(&snapshot)? {
        cuteafd_loader::plan::exlr8::architecture(&family_config)?
    } else {
        let checkpoint = cuteafd_loader::plan::Checkpoint { view: None, snapshot: snapshot.clone(),
            config: family_config, quantize_config: None, weight_map: Default::default(),
            tensors: vec![], missing_shards: vec![], shard_bytes: 0 };
        cuteafd_loader::plan::family::detect(&checkpoint).map_or("", |family| family.id())
    };
    resolve_family_config(&mut args, family, &cfg)?;
    let requested_speculator = args.speculator.clone();
    resolve_speculator(&mut args, &snapshot, family, &cfg)?;
    let replay_explicit = args.replay_records;
    let mut settings = resolve_settings(&mut args, family, &cfg)?;
    if family == "glm5_flash" {
        settings.insert("speculator".into(), setting(cuteafd_core::launch_settings::glmf_speculator(
            (requested_speculator != "auto" || args.explicit_settings.contains("speculator")).then_some(requested_speculator.as_str()), cfg.get("SPECULATOR").map(String::as_str))));
    }
    if family == "mimo_v2" {
        settings.insert("speculator".into(), setting(cuteafd_core::launch_settings::mimo_speculator(
            (requested_speculator != "auto" || args.explicit_settings.contains("speculator")).then_some(requested_speculator.as_str()),
            cfg.get("SPECULATOR").map(String::as_str), mimo_drafter_available(&args, &snapshot, &cfg))));
    }
    if (args.no_speculator || args.include_speculator || args.drafter_snapshot.is_some())
        && matches!(family, "glm5_flash" | "mimo_v2") {
        settings.insert("speculator".into(), setting(cuteafd_core::launch_settings::selected(
            Some(args.speculator.clone()), None, args.speculator.clone(), "explicit speculator inventory override")));
    }
    if family == "kimi_k3" {
        settings.insert("speculator".into(),setting(cuteafd_core::launch_settings::kimi_speculator(
            (requested_speculator != "auto").then_some(requested_speculator.as_str()),cfg.get("SPECULATOR").map(String::as_str), kimi_qualification(&args, &cfg))?));
    }
    if args.resolve_settings_only {
        let mut output = resolved_output(serde_json::json!({"family": family}), &args, settings, &cfg, replay_explicit);
        output["speculator"] = serde_json::json!(args.speculator);
        output["default_drafter"] = serde_json::json!(if family == "kimi_k3" { cuteafd_core::launch_settings::KIMI_K31_DSPARK } else { GLMF_DEFAULT_DRAFTER });
        if family == "kimi_k3" { output["draft_kv_cache"]=serde_json::json!(kimi_qualification(&args, &cfg).draft_kv); output["default_drafter_revision"]=serde_json::json!(cuteafd_core::launch_settings::KIMI_K31_DSPARK_REVISION); }
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }
    if family == "kimi_k3" && matches!(args.speculator.as_str(),"dflash2"|"dspark") && args.drafter_snapshot.is_none() {
        args.drafter_snapshot = file_args(args.clone(), &snapshot)?.drafter_snapshot;
    }
    let view = view.context("checkpoint view after settings-only dispatch")?;
    let mut options = options(&args)?;
    if view.mode() == cuteafd_loader::plan::checkpoint::CoreView::Original && cuteafd_loader::plan::exlr8::recognizes(&snapshot)? {
        if args.spark_range {
            return Err(PlanError::InvalidOption { option: "--spark-range", reason:
                "EXLR8 range geometry requires its native manifest planner".into() }.into());
        }
        anyhow::ensure!(!(args.files || args.fetch || args.role.is_some()),
            "EXLR8 file manifests and fetching are not supported");
        if args.layout {
            use cuteafd_loader::plan::deployment::{AdmissionArgs, DeploymentOptions, DeploymentSpec};
            let cfg = launch_config(args.config.as_deref())?;
            let family = cuteafd_loader::plan::exlr8::architecture(&view.config()?)?;
            let mut native_options = options.clone();
            resolve_exlr8_draft(&mut args, &snapshot, family, &cfg, &mut native_options)?;
            native_options.placement = ExpertPlacement::from_spark_ranks(args.spark_ranks.unwrap_or(args.available_sparks));
            let native = cuteafd_loader::plan::exlr8::native_plan(&snapshot,
                args.exlr8_manifest.as_deref().context("EXLR8 native layout requires --exlr8-manifest up1|bit325")?,
                &native_options, &args.speculator)?;
            if args.deployment {
                let deployment = DeploymentOptions { rtx_gpus: args.rtx,
                    peers: args.deployment_peers.clone(), speculator: args.speculator.clone(),
                    admission: AdmissionArgs { kv_pool_size: args.kv_pool_size.clone(),
                        pool_tokens: args.pool_tokens, memory_policy: args.memory.policy() } };
                println!("{}", serde_json::to_string_pretty(&resolved_output(serde_json::to_value(DeploymentSpec::from_exlr8(&native, &deployment)?)?, &args, settings, &cfg, replay_explicit))?);
            } else if args.json { println!("{}", serde_json::to_string_pretty(&resolved_output(serde_json::to_value(&native.report)?, &args, settings, &cfg, replay_explicit))?); }
            else { print_settings(&resolved_output(serde_json::to_value(&native.report)?, &args, settings, &cfg, replay_explicit)); print!("{}", render(&native.report));
                if let Some(layout) = &native.report.memory_layout { print!("{}", layout.render()); }
            }
            return Ok(());
        }
        let report = cuteafd_loader::plan::exlr8::inspect(&snapshot,
            args.exlr8_manifest.as_deref().unwrap_or("bit325"),
            args.spark_ranks.unwrap_or(args.available_sparks), args.rtx,
            &options.spark_capacities(args.spark_ranks.unwrap_or(args.available_sparks)), options.coordinator_budget_bytes)?;
        if args.json { println!("{}", serde_json::to_string_pretty(&report)?); }
        else { print!("{}", report.render()); }
        anyhow::ensure!(!args.require_ready, "EXLR8 is unsupported: add the kernels listed in the plan");
        return Ok(());
    }
    anyhow::ensure!(args.exlr8_manifest.is_none(),
        "--exlr8-manifest requires an EXLR8 checkpoint");
    anyhow::ensure!(args.exlr8_prefill_activations.is_none(),
        "--exlr8-prefill-activations requires an EXLR8 checkpoint");
    let role_inventory = args.files || args.fetch || args.role.is_some();
    let input = if role_inventory { None } else { Some(cuteafd_loader::plan::PlanInput::open_view(&view)?) };
    let inventory;
    let checkpoint = if let Some(input) = &input { input.checkpoint() } else {
        inventory = cuteafd_loader::plan::Checkpoint::inventory_view(&view)?;
        &inventory
    };
    let family = cuteafd_loader::plan::family::detect(checkpoint).map_or("", |family| family.id());
    args.attention.validate_view(&view)?;
    let mut precision = args.clone();
    if family == "glm5_flash" {
        let cfg = cuteafd_loader::families::glm5_flash::GlmNextConfig::from_hf(&checkpoint.config)?;
        let core = cuteafd_loader::families::glm5_flash::core::CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        precision.head_format = args.head_format.resolve(core.get_weight("lm_head").is_some(), args.rtx);
    }
    validate_glmf_precision(&precision, family)?;
    let cfg = launch_config(args.config.as_deref())?;
    resolve_speculator(&mut args, &snapshot, family, &cfg)?;
    options.qwen_auto_mtp = family == "qwen4" && args.speculator == "auto";
    let tp2_dspark_experts = tp2_dspark_experts(&args, family, &cfg)?;
    let tp2_dflash = tp2_dflash(&args, family, &cfg)?;
    options.draft_ownership = Some(cuteafd_loader::plan::spec::DraftOwnership::selected(family, &args.speculator));
    if let Some(layout) = &mut options.layout {
        layout.tp2_dspark_experts = tp2_dspark_experts;
        layout.tp2_dflash = tp2_dflash;
        if !options.qwen_auto_mtp && !options.draft_ownership.unwrap().uses_checkpoint() { layout.native_mtp_layers = 0; }
        layout.glmf_drafter_disabled = args.speculator == "off";
        layout.glm5_drafter_disabled = args.speculator == "off";
        layout.minimax_dspark = family == "minimax_m3" && args.speculator == "dspark";
        layout.inkling_dspark = matches!(family, "inkling2" | "inkling2_mini") && args.speculator == "dspark";
        if family == "inkling2_mini" && args.speculator == "dflash2" {
            layout.inkling_mini_dflash = Some(file_args(args.clone(), &snapshot)?.drafter_snapshot
                .context("Inkling 2 Mini DFlash2 planning requires its checkpoint")?);
        }
        resolve_mimo_drafter(layout, &args.speculator);
        if family == "qwen3_5" { layout.dense_qwen_mtp = args.speculator == "mtp"; }
        if family == "qwen3_5" && args.speculator == "dspark" {
            let resolved = file_args(args.clone(), &snapshot)?;
            layout.dense_qwen_drafter_snapshot = Some(resolved.drafter_snapshot
                .context("dense dSpark planning requires its checkpoint")?);
        }
        if family == "muse_spark" && args.speculator != "off" {
            let resolved = file_args(args.clone(), &snapshot)?;
            layout.muse_drafter_snapshot = Some(resolved.drafter_snapshot
                .context("Muse Spark DFlash2 planning requires its checkpoint")?);
        }
        if family == "kimi_k3" && args.speculator != "off" {
            anyhow::ensure!(matches!(args.speculator.as_str(),"dflash2"|"dspark") && args.drafter_snapshot.is_some(),
                "Kimi speculation requires --speculator dflash2|dspark --drafter-snapshot SNAPSHOT");
            layout.kimi_drafter_snapshot = args.drafter_snapshot.clone();
        }
        let config = &checkpoint.config;
        let v41 = config.get("model_type").and_then(serde_json::Value::as_str) == Some("deepseek_v41")
            || config.get("architectures").and_then(serde_json::Value::as_array)
                .is_some_and(|arches| arches.iter().any(|arch| arch == "DeepseekV41ForCausalLM"));
        if v41 { v41_layout(layout, &args); }
    }
    if args.files || args.fetch || args.role.is_some() {
        use cuteafd_loader::plan::files::{manifest_roles_view, ReadRole};
        let args = file_args(args, &snapshot)?;
        let placement = ExpertPlacement::from_spark_ranks(args.spark_ranks.unwrap_or(args.available_sparks));
        let parse = |name: &str| {
            anyhow::ensure!(!(args.no_speculator && name == "drafter"), "--no-speculator conflicts with drafter role");
            ReadRole::parse(name, placement,
                cuteafd_loader::plan::spec::DraftOwnership::selected(family, &args.speculator).uses_checkpoint())
        };
        let complete_roles = |mut roles: Vec<ReadRole>| {
            if roles.iter().any(|r| matches!(r, ReadRole::Coordinator { .. })) {
                if args.vision != cuteafd_loader::plan::MediaMode::Off { roles.push(ReadRole::Vision); }
                if args.audio != cuteafd_loader::plan::MediaMode::Off { roles.push(ReadRole::Audio); }
            }
            roles
        };
        let roles = if let Some(layout) = &args.file_layout {
            anyhow::ensure!(args.role.is_none(), "--role and --file-layout are mutually exclusive");
            let hosts: std::collections::BTreeMap<String, Vec<String>> =
                cuteafd_loader::read_json_file(layout)?;
            if let Some(host) = &args.host {
                hosts.get(host).with_context(|| format!("host {host} is absent from {}", layout.display()))?
                    .iter().map(|name| parse(name)).collect::<Result<Vec<_>>>()?
            } else {
                let manifests = hosts.iter().map(|(host, names)| {
                    let roles = complete_roles(names.iter().map(|name| parse(name)).collect::<Result<Vec<_>>>()?);
                    let mut manifest = manifest_roles_view(&view, &roles)?;
                    attach_snapshots(&mut manifest, &roles, &args)?;
                    Ok((host.clone(), manifest))
                }).collect::<Result<std::collections::BTreeMap<_, _>>>()?;
                if args.fetch {
                    let hosts: Vec<_> = manifests.into_iter().collect();
                    for chunk in hosts.chunks(args.fetch_parallel as usize) {
                        std::thread::scope(|scope| -> Result<()> {
                            let handles: Vec<_> = chunk.iter().map(|(host, manifest)| {
                                let mut host_args = args.clone();
                                host_args.host = Some(host.clone());
                                let snapshot = &snapshot;
                                scope.spawn(move || transfer(snapshot, manifest, &host_args))
                            }).collect();
                            for handle in handles { handle.join().map_err(|_| anyhow::anyhow!("host transfer thread panicked"))??; }
                            Ok(())
                        })?;
                    }
                    eprintln!("layout: {} host transfers complete", hosts.len());
                    return Ok(());
                }
                anyhow::ensure!(args.json, "all-host inventory needs --json; select --host for a plain file list");
                println!("{}", serde_json::to_string_pretty(&manifests)?);
                return Ok(());
            }
        } else {
            anyhow::ensure!(args.host.is_none() || args.fetch, "--host needs --file-layout or --fetch");
            vec![parse(args.role.as_deref().unwrap_or("coordinator"))?]
        };
        let roles = complete_roles(roles);
        let mut manifest = manifest_roles_view(&view, &roles)?;
        attach_snapshots(&mut manifest, &roles, &args)?;
        if args.fetch {
            transfer(&snapshot, &manifest, &args)?;
        } else if args.files {
            if args.json { println!("{}", serde_json::to_string_pretty(&manifest)?); }
            else {
                anyhow::ensure!(manifest.additional_snapshots.is_empty(), "multiple snapshot roots need --json; inventory each root separately for a plain rsync list");
                for file in &manifest.files { println!("{}", file.path); }
            }
        } else {
            let mut headers = 0;
            for role in &roles { headers += cuteafd_loader::plan::files::open_role_view(&view, *role)?.tensors.len(); }
            if args.json { println!("{}", serde_json::to_string_pretty(&manifest)?); }
            else { println!("role {} HEADERS READY: {headers} tensor headers, {} required files",
                manifest.role, manifest.files.len()); }
        }
        return Ok(());
    }
    options.vision = glm_flash_vision_view(&args, family, &view);
    let input = input.context("full planner input missing after role-inventory dispatch")?;
    if args.spark_range {
        let columns = input.spark_ranges(&options)?;
        if args.json { println!("{}", serde_json::to_string_pretty(&columns)?); }
        else { print!("{}", cuteafd_loader::plan::spark_range::render(&columns)); }
        return Ok(());
    }
    let report = input.plan_spark_count(&options, args.available_sparks, args.spark_ranks)?;
    if args.json || args.deployment {
        if let Some(warning) = &report.spark_count_warning { eprintln!("warning: {warning}"); }
    }
    if args.deployment {
        use cuteafd_loader::plan::deployment::{AdmissionArgs, DeploymentOptions, DeploymentSpec};
        let deployment = DeploymentOptions {
            rtx_gpus: args.rtx, peers: args.deployment_peers.clone(),
            speculator: args.speculator.clone(),
            admission: AdmissionArgs { kv_pool_size: args.kv_pool_size.clone(),
                pool_tokens: args.pool_tokens, memory_policy: args.memory.policy() },
        };
        println!("{}", serde_json::to_string_pretty(&resolved_output(serde_json::to_value(DeploymentSpec::from_report(&report, &deployment)?)?, &args, settings, &cfg, replay_explicit))?);
    } else if args.json {
        println!("{}", serde_json::to_string_pretty(&resolved_output(serde_json::to_value(&report)?, &args, settings, &cfg, replay_explicit))?);
    } else {
        print_settings(&resolved_output(serde_json::to_value(&report)?, &args, settings, &cfg, replay_explicit));
        print!("{}", render(&report));
        if let Some(layout) = &report.memory_layout {
            print!("\nmemory layout (planner)\n{}", layout.render());
        }
    }
    if args.require_ready && !report.executable() {
        anyhow::bail!("{} is not servable by this build: {}", args.model,
            report.hints.iter().map(|hint| hint.what.as_str()).collect::<Vec<_>>().join("; "));
    }
    Ok(())
}

/// The vision policy to plan: GLM 5.3 Flash's automatic vision is off when serve-glmf would serve
/// text (`serve::auto_vision`), so no encoder is placed for a checkpoint whose own chat template
/// cannot mark images; the note goes to stderr once. Chat files a plan cannot read leave the policy
/// as given: a plan needs only the checkpoint's config and tensor headers.
fn glm_flash_vision(args: &PlanArgs, family: &str, snapshot: &std::path::Path) -> cuteafd_loader::plan::MediaMode {
    args.attention.select(snapshot).map(|view| glm_flash_vision_view(args, family, &view)).unwrap_or(args.vision)
}

fn glm_flash_vision_view(args: &PlanArgs, family: &str,
    view: &cuteafd_loader::plan::checkpoint::CheckpointView) -> cuteafd_loader::plan::MediaMode {
    if family != "glm5_flash" { return args.vision; }
    match crate::families::glm5_flash::serve::auto_vision_view(args.vision, view, args.chat_template_from.as_deref()) {
        Ok((vision, note)) => {
            if let Some(note) = note { eprintln!("note: {note}"); }
            vision
        }
        Err(_) => args.vision,
    }
}

fn launch_config(path: Option<&std::path::Path>) -> Result<std::collections::BTreeMap<String, String>> {
    let Some(path) = path else { return Ok(Default::default()) };
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let configured: std::collections::BTreeMap<String, String> = text.lines().filter_map(|line| {
        let (key, value) = line.split_once('=')?;
        (!key.is_empty() && key.bytes().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_'))
            .then(|| (key.to_owned(), value.to_owned()))
    }).collect();
    Ok(configured)
}

fn spark_hosts_count(cfg: &std::collections::BTreeMap<String, String>) -> Result<Option<usize>> {
    let Some(hosts) = cfg.get("SPARK_HOSTS").filter(|value| !value.is_empty()) else { return Ok(None) };
    let configured: std::collections::BTreeSet<&str> = (0..8)
        .filter_map(|rank| cfg.get(&format!("SPARK_{rank}_HOST")).map(String::as_str)).collect();
    let pool = ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"];
    let mut seen = std::collections::BTreeSet::new();
    for host in hosts.split(',') {
        anyhow::ensure!(!host.is_empty() && host == host.trim(), "SPARK_HOSTS contains an empty or whitespace host");
        anyhow::ensure!(seen.insert(host), "SPARK_HOSTS contains duplicate host {host}");
        anyhow::ensure!(configured.contains(host) || pool.contains(&host), "SPARK_HOSTS names unknown Spark pool host {host}");
    }
    Ok(Some(seen.len()))
}

fn checkpoint_model_id<'a>(args: &'a PlanArgs, cfg: &'a std::collections::BTreeMap<String, String>) -> &'a str {
    args.checkpoint_model_id.as_deref().or_else(|| cfg.get("MODEL_ID").map(String::as_str))
        .unwrap_or(&args.model)
}

const GLMF_DEFAULT_DRAFTER: &str = "incoai/GLM-5.3-Flash-DFlash2";

fn kimi_target_matches(snapshot: &std::path::Path) -> Result<bool> {
    let config: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
    Ok(config["model_type"] == "kimi_k31" || config["architectures"].as_array()
        .is_some_and(|a| a.iter().any(|v| v == "KimiK31ForConditionalGeneration")))
}

fn kimi_content_matches(snapshot: &std::path::Path, draft: &std::path::Path) -> Result<bool> {
    let target: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
    let drafter: serde_json::Value = cuteafd_loader::read_json_file(&draft.join("config.json"))?;
    kimi_pair_metadata(&target, &drafter, snapshot)?;
    Ok(true)
}

enum DraftPairDialect<'a> {
    Speculators,
    Native { config: &'a serde_json::Value, draft: &'a serde_json::Value },
}

impl<'a> DraftPairDialect<'a> {
    fn read(draft: &'a serde_json::Value) -> Result<Self> {
        if draft.get("speculators_config").is_some() || draft["speculators_model_type"] == "dspark" {
            return Ok(Self::Speculators);
        }
        anyhow::ensure!(draft["architectures"].as_array().is_some_and(|a| a.iter().any(|v|
            matches!(v.as_str(),Some("Qwen3DSparkModel"|"DSparkDraftModel"|"DFlash2DraftModel")))),
            "drafter pairing: unsupported native draft architecture");
        let config=draft.get("dflash_config").context("drafter pairing: native dflash_config missing")?;
        Ok(Self::Native {config,draft})
    }
}

fn kimi_pair_metadata(target: &serde_json::Value, draft: &serde_json::Value, snapshot: &std::path::Path) -> Result<()> {
    if let DraftPairDialect::Native {config,draft}=DraftPairDialect::read(draft)? {
        let text=target.get("text_config").unwrap_or(target);
        anyhow::ensure!(target["architectures"].as_array().is_some_and(|a|
            a.iter().any(|v|v=="KimiK31ForConditionalGeneration")),
            "drafter pairing mismatch: native draft requires K3.1 target architectures");
        for (field,target_field) in [("num_target_layers","num_hidden_layers"),("hidden_size","hidden_size"),
            ("vocab_size","vocab_size"),("draft_vocab_size","vocab_size")] {
            anyhow::ensure!(draft[field].as_u64().is_some()&&draft[field]==text[target_field],
                "drafter pairing mismatch: {field}: drafter={}, target={}",draft[field],text[target_field]);
        }
        let layers=text["num_hidden_layers"].as_u64().context("drafter pairing: target layer count missing")?;
        let taps=config["target_layer_ids"].as_array().context("drafter pairing: native target_layer_ids missing")?;
        let mut unique=std::collections::BTreeSet::new();
        anyhow::ensure!(!taps.is_empty()&&taps.iter().all(|v|v.as_u64().is_some_and(|n|n<layers&&unique.insert(n))),
            "drafter pairing mismatch: target_layer_ids outside target layer count or duplicated");
        for field in ["bos_token_id","eos_token_id","pad_token_id"] {
            let expected=text.get(field).or_else(||target.get(field));
            anyhow::ensure!(draft.get(field).is_some_and(|v|!v.is_null()&&Some(v)==expected),
                "drafter pairing mismatch: {field} differs from target");
        }
        let vocab=text["vocab_size"].as_u64().unwrap_or(0);
        // Native block masking uses a dedicated vocabulary token, not PAD.
        anyhow::ensure!(config["mask_token_id"].as_u64().is_some_and(|n|n<vocab),
            "drafter pairing mismatch: native mask_token_id outside target vocabulary");
        return Ok(());
    }
    let verifier = &draft["speculators_config"]["verifier"];
    let architectures = verifier["architectures"].as_array().context("drafter pairing: verifier.architectures missing")?;
    anyhow::ensure!(architectures.iter().any(|v| target["architectures"].as_array().is_some_and(|a| a.contains(v))),
        "drafter pairing mismatch: verifier.architectures differs from target architectures");
    let declared = verifier["name_or_path"].as_str().context("drafter pairing: verifier.name_or_path missing")?;
    let manifest = snapshot.join("cuteafd-mc.json");
    let mc: serde_json::Value = if manifest.is_file() { cuteafd_loader::read_json_file(&manifest)? } else { serde_json::Value::Null };
    let hub_identity = snapshot.components().find_map(|part| part.as_os_str().to_str()?.strip_prefix("models--")
        .map(|name| name.replace("--", "/")));
    let identity = mc["source"]["repo"].as_str().or_else(|| target["_name_or_path"].as_str().filter(|s| !s.is_empty()))
        .or(hub_identity.as_deref()).context("drafter pairing: target model identity missing (source.repo or _name_or_path)")?;
    // Organization aliases preserve the declared model name (official and verifier mirrors).
    let model_name = |id: &str| id.trim_end_matches('/').rsplit('/').next().unwrap_or("").to_ascii_lowercase();
    anyhow::ensure!(model_name(declared) == model_name(identity),
        "drafter pairing mismatch: verifier.name_or_path={declared} differs from target source.repo={identity}");
    let text = target.get("text_config").unwrap_or(target);
    let body = draft.get("transformer_layer_config").unwrap_or(draft);
    for (field, value) in [("hidden_size",draft.get("target_hidden_size").filter(|v| !v.is_null()).unwrap_or(&body["hidden_size"])),
        ("vocab_size", &draft["draft_vocab_size"])] {
        anyhow::ensure!(value.as_u64().is_some() && value == &text[field],
            "drafter pairing mismatch: {field}: drafter={value}, target={}",text[field]);
    }
    anyhow::ensure!(body["vocab_size"] == text["vocab_size"], "drafter pairing mismatch: transformer_layer_config.vocab_size");
    let layers = text["num_hidden_layers"].as_u64().context("drafter pairing: num_hidden_layers missing")?;
    let taps = draft["aux_hidden_state_layer_ids"].as_array().context("drafter pairing: aux_hidden_state_layer_ids missing")?;
    anyhow::ensure!(!taps.is_empty() && taps.iter().all(|v| v.as_u64().is_some_and(|n| n > 0 && n <= layers)),
        "drafter pairing mismatch: aux_hidden_state_layer_ids outside target num_hidden_layers={layers}");
    let mask = draft["mask_token_id"].as_u64().context("drafter pairing: mask_token_id missing")?;
    anyhow::ensure!(mask < text["vocab_size"].as_u64().unwrap_or(0)
        && text.get("pad_token_id").or_else(|| target.get("pad_token_id")).and_then(|v|v.as_u64()) == Some(mask),
        "drafter pairing mismatch: mask_token_id differs from target pad_token_id or vocabulary");
    Ok(())
}

fn require_default_drafter(args: &PlanArgs, cfg: &std::collections::BTreeMap<String, String>,
    default_model: &str, default_revision: Option<&str>) -> Result<PathBuf> {
    let model = cfg.get("SPECULATOR_MODEL_ID").filter(|v| !v.is_empty()).map(String::as_str).unwrap_or(default_model);
    let revision = cfg.get("SPECULATOR_MODEL_REVISION").filter(|v| !v.is_empty()).map(String::as_str)
        .or_else(|| (model == default_model).then_some(default_revision).flatten());
    let home = args.hf_home.clone().unwrap_or_else(default_hf_home);
    let remedy = format!("default drafter {model}@{} is unavailable or missing config.json; run `nest hf download {model}{}` or set SPECULATOR=off",
        revision.unwrap_or("main"), revision.map(|r| format!(" --revision {r}")).unwrap_or_default());
    let path = if let Some(path) = &args.drafter_snapshot { path.clone() } else {
        resolve_snapshot_at_revision(model, Some(&home), revision).with_context(|| remedy.clone())?
            .snapshot_path.with_context(|| remedy.clone())?
    };
    anyhow::ensure!(path.join("config.json").is_file(), "{remedy}: {}", path.display());
    Ok(path)
}

fn resolve_kimi_content(args: &PlanArgs, snapshot: &std::path::Path,
    cfg: &std::collections::BTreeMap<String, String>) -> Result<bool> {
    if args.no_speculator || args.speculator == "off"
        || (args.speculator == "auto" && cfg.get("SPECULATOR").is_some_and(|v| v == "off")) { return Ok(false); }
    if !kimi_target_matches(snapshot)? {
        anyhow::ensure!(args.speculator != "dspark" && cfg.get("SPECULATOR").is_none_or(|v| v != "dspark"),
            "drafter pairing mismatch: verifier.architectures requires a K3.1 target");
        return Ok(false);
    }
    let mut qualification = kimi_qualification(args, cfg);
    qualification.content_qualified = true;
    let selection = cuteafd_core::launch_settings::kimi_speculator(
        (args.speculator != "auto").then_some(args.speculator.as_str()), cfg.get("SPECULATOR").map(String::as_str), qualification)?;
    if args.no_speculator || selection.value == "off" { return Ok(false); }
    if selection.value != "dspark" { return Ok(false); }
    let draft = require_default_drafter(args, cfg, cuteafd_core::launch_settings::KIMI_K31_DSPARK,
        Some(cuteafd_core::launch_settings::KIMI_K31_DSPARK_REVISION))?;
    kimi_content_matches(snapshot, &draft)
}

fn kimi_qualification<'a>(args: &'a PlanArgs, _cfg: &'a std::collections::BTreeMap<String, String>)
    -> cuteafd_core::launch_settings::KimiDraftQualification<'a> {
    use cuteafd_loader::{plan::checkpoint::CoreView, families::qwen4::Qwen4KvCache};
    cuteafd_core::launch_settings::KimiDraftQualification {
        content_qualified: args.kimi_content_qualified,
        core_view: match args.attention.core_view { CoreView::Original => "original", CoreView::K6Compact => "k6-compact", _ => "unqualified" },
        rtx: args.rtx,
        target_kv: if args.kv_format.unwrap_or_default() == cuteafd_core::MlaCacheFormat::Fp8 { "fp8" } else { "bf16" },
        draft_kv: if args.draft_kv_cache == Qwen4KvCache::Bf16 { "bf16" } else { "fp8" },
    }
}


// Family detection already establishes the target architecture. Drafter loading
// validates native config, taps and tensor shapes; aliases and MC views do not
// change whether the checkpoint bundles (or explicitly configures) a drafter.
fn mimo_drafter_available(args: &PlanArgs, snapshot: &std::path::Path,
    cfg: &std::collections::BTreeMap<String, String>) -> bool {
    args.drafter_snapshot.is_some() || snapshot.join("dflash").is_dir()
        || ["SPECULATOR_MODEL_ID", "DRAFT_MODEL_ID"].iter()
            .any(|key| cfg.get(*key).is_some_and(|value| !value.is_empty()))
}

fn resolve_speculator(args: &mut PlanArgs, snapshot: &std::path::Path, family: &str,
    cfg: &std::collections::BTreeMap<String, String>) -> Result<()> {
    if family == "kimi_k3" {
        if !args.explicit_settings.contains("draft_kv_cache") {
            args.draft_kv_cache = match cfg.get("DRAFT_KV_CACHE") {
                Some(value) => value.parse()?,
                None => args.draft_kv_cache,
            };
        }
        args.kimi_content_qualified = resolve_kimi_content(args, snapshot, cfg)?;
        args.speculator=cuteafd_core::launch_settings::kimi_speculator(
            (args.speculator != "auto").then_some(args.speculator.as_str()),cfg.get("SPECULATOR").map(String::as_str), kimi_qualification(&args, &cfg))?.value;
    }
    let default = match family {
        "minimax_m3" | "inkling2" => "dspark",
        "inkling2_mini" => "dflash2",
        "hy4_flash" if (args.include_speculator || (args.speculator == "auto"
            && cfg.get("SPECULATOR").is_none_or(|value| value == "auto")))
            && cuteafd_loader::families::hy4_flash::Hy4DflyConfig::bundled(
                &cuteafd_loader::read_json_file::<serde_json::Value>(&snapshot.join("config.json"))?) => "dfly2",
        "hy4" if (args.include_speculator || (args.speculator == "auto"
            && cfg.get("SPECULATOR").is_none_or(|value| value == "auto")))
            && cuteafd_loader::families::hy4::Hy4DflyConfig::bundled(
                &cuteafd_loader::read_json_file::<serde_json::Value>(&snapshot.join("config.json"))?) => "dfly2",
        "glm5" | "glm5_flash" | "muse_spark" => "dflash2",
        "qwen3_5" if serde_json::from_slice::<serde_json::Value>(&std::fs::read(snapshot.join("config.json"))?)
            .is_ok_and(|config| config.get("text_config").unwrap_or(&config)["hidden_size"] == 5120) => "dspark",
        "mimo_v2" if matches!(checkpoint_model_id(args, cfg), "XiaomiMiMo/MiMo-V2.6-Flash-MOPD" | "XiaomiMiMo/MiMo-V2.6-Pro-MOPD") => "dflash2",
        "qwen4" | "step5" if args.native_mtp_layers > 0 => "mtp",
        "deepseek_v4" | "deepseek_v41" if args.native_mtp_layers > 0 => "dspark",
        _ => "off",
    };
    if family == "glm5_flash" && !args.no_speculator {
        let resolved = cuteafd_core::launch_settings::glmf_speculator((args.speculator != "auto" || args.explicit_settings.contains("speculator")).then_some(args.speculator.as_str()),
            cfg.get("SPECULATOR").map(String::as_str));
        if resolved.provenance.starts_with("auto:") {
            require_default_drafter(args, cfg, GLMF_DEFAULT_DRAFTER, None)?;
        }
        args.speculator = resolved.value;
    }
    if family == "mimo_v2" && !args.no_speculator {
        args.speculator = cuteafd_core::launch_settings::mimo_speculator(
            (args.speculator != "auto" || args.explicit_settings.contains("speculator")).then_some(args.speculator.as_str()),
            cfg.get("SPECULATOR").map(String::as_str), mimo_drafter_available(args, snapshot, cfg)).value;
    }
    if args.speculator == "auto" {
        args.speculator = cfg.get("SPECULATOR").cloned().unwrap_or_else(|| {
            if cfg.get("DRAFT_MODEL_ID").is_some_and(|v| !v.is_empty()) { "dflash2".into() }
            else if let Some(mode) = cfg.get("DSPARK") { if mode == "on" { "dspark" } else { "off" }.into() }
            else if family == "qwen4" { "auto".into() }
            else { default.into() }
        });
    }
    if args.speculator == "auto" && family != "qwen4" { args.speculator = default.into(); }
    anyhow::ensure!(["auto", "off", "mtp", "dspark", "dflash2", "dfly2"].contains(&args.speculator.as_str()), "unknown SPECULATOR={}", args.speculator);
    anyhow::ensure!(!(args.no_speculator && args.include_speculator), "--no-speculator conflicts with --include-speculator");
    if args.no_speculator {
        anyhow::ensure!(args.drafter_snapshot.is_none() && args.role.as_deref() != Some("drafter"), "--no-speculator conflicts with drafter inputs/role");
        args.speculator = "off".into();
    } else if (args.include_speculator || args.drafter_snapshot.is_some()) && args.speculator == "off" {
        args.speculator = if matches!(family, "kimi_k3" | "muse_spark") && args.drafter_snapshot.is_some() { "dflash2".into() }
            else if family == "mimo_v2" && args.include_speculator { "mtp".into() }
            else { default.into() };
        anyhow::ensure!(args.speculator != "off", "this family has no default speculator");
    }
    let enabled = match (family, args.speculator.as_str()) {
        (_, "off") | ("hy4" | "hy4_flash", "dfly2") | ("qwen3_5", "dspark" | "mtp") | ("kimi_k3", "dflash2"|"dspark") | ("muse_spark" | "inkling2_mini", "dflash2") | ("glm5", "dflash2") | ("glm5_flash", "dflash2" | "dspark")
        | ("mimo_v2", "dflash2" | "mtp") | ("qwen4", "auto") | ("qwen4" | "step5", "mtp")
        | ("deepseek_v4" | "deepseek_v41" | "mistral_large4" | "minimax_m3" | "inkling2" | "inkling2_mini", "dspark") => true,
        _ => false,
    };
    anyhow::ensure!(enabled, "SPECULATOR={} does not apply to {family}", args.speculator);
    Ok(())
}

/// Resolve the release defaults or the launcher's plain KEY=VALUE config.
/// Never execute a shell configuration while constructing a file inventory.
fn file_args(mut args: PlanArgs, snapshot: &std::path::Path) -> Result<PlanArgs> {
    use cuteafd_loader::plan::MediaMode;
    let inventory = cuteafd_loader::plan::checkpoint::Checkpoint::inventory_view(&args.attention.select(snapshot)?)?;
    let family = cuteafd_loader::plan::family::detect(&inventory).context("unsupported checkpoint family")?.id();
    let cfg = launch_config(args.config.as_deref())?;
    if let Some(model) = cfg.get("MODEL_ID") {
        if !PathBuf::from(&args.model).is_dir() {
            anyhow::ensure!(model == &args.model, "--config MODEL_ID={model} differs from {}", args.model);
        }
    }
    if args.spark_ranks.is_none() {
        if cfg.get("EXPERT_BACKEND").is_some_and(|v| v == "local") { args.spark_ranks = Some(0); }
        else if let Some(ranks) = cfg.get("SPARK_COUNT") { args.spark_ranks = Some(ranks.parse().context("SPARK_COUNT")?); }
    }
    if family == "kimi_k3" {
        if let Some(value) = cfg.get("TP2_DFLASH") {
            let configured = match value.as_str() {
                "on" => true, "off" => false,
                _ => anyhow::bail!("TP2_DFLASH must be on or off"),
            };
            args.tp2_dflash.get_or_insert(configured);
        }
    }
    resolve_speculator(&mut args, snapshot, family, &cfg)?;
    // File inventories have no admitted layout. Retain the checkpoint-native
    // superset so fetching headers cannot omit a subsequently selected MTP owner.
    if family == "qwen4" && args.speculator == "auto" {
        args.speculator = if args.native_mtp_layers > 0 { "mtp" } else { "off" }.into();
    }
    let needs_drafter = if let Some(layout) = &args.file_layout {
        let hosts: std::collections::BTreeMap<String, Vec<String>> =
            cuteafd_loader::read_json_file(layout)?;
        let roles = if let Some(host) = &args.host {
            hosts.get(host).with_context(|| format!("host {host} is absent from {}", layout.display()))?.clone()
        } else { hosts.into_values().flatten().collect() };
        roles.iter().any(|r| matches!(r.as_str(), "coordinator" | "rtx0" | "rtx1" | "drafter"))
    } else {
        args.role.as_deref().is_none_or(|r| matches!(r, "coordinator" | "rtx0" | "rtx1" | "drafter"))
    };
    if needs_drafter && args.drafter_snapshot.is_none() && matches!(args.speculator.as_str(), "dflash2" | "dspark") {
        let model = cfg.get("SPECULATOR_MODEL_ID").or_else(|| cfg.get("DRAFT_MODEL_ID"))
            .filter(|v| !v.is_empty()).map(String::as_str).or(match (family, args.speculator.as_str()) {
                ("qwen3_5", "dspark") => Some("RadixArk/Qwen3.8-27B-DSpark"),
                ("kimi_k3", "dspark") => Some(cuteafd_core::launch_settings::KIMI_K31_DSPARK),
                ("glm5", "dflash2") => Some("incoai/GLM-5.3-DFlash2"),
                ("muse_spark", "dflash2") => Some("nvidia/Muse-Spark-Open-816B-A42B-DFlash2"),
                ("inkling2_mini", "dflash2") => Some(cuteafd_loader::families::inkling2_mini::dflash::MODEL),
                ("glm5_flash", "dflash2") => Some("incoai/GLM-5.3-Flash-DFlash2"),
                ("glm5_flash", "dspark") => Some("RedHatAI/GLM-5.3-Flash-speculator.dspark-preview"),
                _ => None,
            });
        if let Some(model) = model {
            let hf_home = args.hf_home.clone().unwrap_or_else(default_hf_home);
            let revision = cfg.get("SPECULATOR_MODEL_REVISION").or_else(|| cfg.get("DRAFT_MODEL_REVISION"))
                .filter(|v| !v.is_empty()).map(String::as_str).or_else(|| {
                    if family == "kimi_k3" && model == cuteafd_core::launch_settings::KIMI_K31_DSPARK {
                        return Some(cuteafd_core::launch_settings::KIMI_K31_DSPARK_REVISION);
                    }
                    if family == "inkling2_mini" && model == cuteafd_loader::families::inkling2_mini::dflash::MODEL {
                        return Some(cuteafd_loader::families::inkling2_mini::dflash::REVISION);
                    }
                    if family == "muse_spark" && model == "nvidia/Muse-Spark-Open-816B-A42B-DFlash2" {
                        return Some("22e2669e4c60689f9bff0749a34d9d8e59ce7cd6");
                    }
                    (family == "qwen3_5" && args.speculator == "dspark"
                        && model == "RadixArk/Qwen3.8-27B-DSpark")
                        .then_some("85ef153be924f17ce4bf62726954eeaa4a73e854")
                });
            args.drafter_snapshot = Some(resolve_snapshot_at_revision(model, Some(&hf_home), revision)?
                .snapshot_path.with_context(|| format!("required speculator {model} is not downloaded; use --no-speculator only when serving without it"))?);
        }
    }
    if let Some(mode) = cfg.get("VISION") { args.vision = mode.parse().map_err(anyhow::Error::msg)?; }
    if family == "deepseek_v41" {
        resolve_plan_audio_config(&mut args, &cfg)?;
        resolve_v41_audio(&mut args, &cfg)?;
    } else {
        if let Some(mode) = cfg.get("AUDIO") { args.audio = mode.parse().map_err(anyhow::Error::msg)?; }
        args.audio = cuteafd_loader::plan::resolve_audio_config(args.audio, snapshot, &inventory.config)?;
    }
    Ok(args)
}

fn attach_snapshots(manifest: &mut cuteafd_loader::plan::files::FileManifest,
    roles: &[cuteafd_loader::plan::files::ReadRole], args: &PlanArgs) -> Result<()> {
    use cuteafd_loader::plan::files::{manifest_standalone, ReadRole};
    let coordinator = roles.iter().any(|r| matches!(r, ReadRole::Coordinator { .. }));
    if (coordinator || roles.contains(&ReadRole::Drafter) || roles.contains(&ReadRole::ExternalDrafter)) && args.speculator == "dflash2"
        && args.drafter_snapshot.is_none() {
        let root = PathBuf::from(&manifest.snapshot);
        cuteafd_loader::plan::files::include_directory(manifest, &root, "dflash")?;
    }
    let configured = [
        ("drafter", &args.drafter_snapshot, coordinator && args.speculator != "off"
            || roles.iter().any(|r| matches!(r, ReadRole::Drafter | ReadRole::ExternalDrafter | ReadRole::Coordinator { speculator: true, .. }))),
        ("vision", &args.vision_snapshot, roles.contains(&ReadRole::Vision)),
        ("audio", &args.audio_snapshot, roles.contains(&ReadRole::Audio)),
    ];
    for (role, path, enabled) in configured {
        if let Some(path) = path.as_ref().filter(|_| enabled) { manifest.additional_snapshots.push(manifest_standalone(path, role)?); }
    }
    manifest.total_bytes = manifest.additional_snapshots.iter().fold(manifest.total_bytes, |sum, repo|
        sum.and_then(|sum| sum.checked_add(repo.total_bytes?)));
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// One SSH session for the entire file inventory, preserving input order.
fn remote_sizes(host: &str, root: &std::path::Path, files: &[cuteafd_loader::plan::files::RequiredFile]) -> Result<Vec<Option<u64>>> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("ssh").args([host,
        "while IFS= read -r path; do if test -f \"$path\"; then stat -Lc %s -- \"$path\" || exit 1; else printf '%s\\n' '-'; fi; done"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    let list = files.iter().map(|file| format!("{}\n", root.join(&file.path).display())).collect::<String>();
    child.stdin.take().context("SSH size list stdin")?.write_all(list.as_bytes())?;
    let output = child.wait_with_output()?;
    anyhow::ensure!(output.status.success(), "host {host}: inspecting file sizes failed");
    parse_remote_sizes(host, &String::from_utf8(output.stdout)?, files.len())
}

fn parse_remote_sizes(host: &str, output: &str, count: usize) -> Result<Vec<Option<u64>>> {
    let sizes = output.lines().map(|line| {
        if line == "-" { Ok(None) } else { Ok(Some(line.parse::<u64>()?)) }
    }).collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(sizes.len() == count, "host {host}: expected {count} size results, got {}", sizes.len());
    Ok(sizes)
}

fn source_location(value: Option<&str>, snapshot: &std::path::Path) -> Result<(Option<String>, PathBuf)> {
    let value = value.map(str::to_owned).unwrap_or_else(|| snapshot.display().to_string());
    let (host, path) = if let Some((host, path)) = value.split_once(':') {
        anyhow::ensure!(!host.is_empty() && !host.starts_with('-') && host.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-@".contains(&c)), "invalid source host {host}");
        (Some(host.to_owned()), PathBuf::from(path))
    } else { (None, PathBuf::from(value)) };
    anyhow::ensure!(path.is_absolute() && !path.to_string_lossy().contains(['\n', '\r']), "source needs an absolute snapshot path");
    Ok((host, path))
}

fn transfer(snapshot: &std::path::Path, manifest: &cuteafd_loader::plan::files::FileManifest,
    args: &PlanArgs) -> Result<()> {
    use anyhow::ensure;
    use std::io::Write;
    use std::process::Command;
    ensure!(manifest.additional_snapshots.is_empty(), "fetch separate snapshot roots individually; additional repos are listed in --files --json");
    let automatic = args.source.as_deref() == Some("auto");
    let (source_host, source_path) = source_location(if automatic { None } else { args.source.as_deref() }, snapshot)?;
    if automatic {
        use std::process::Command;
        let local = Command::new("hostname").arg("-s").output().ok()
            .and_then(|o| String::from_utf8(o.stdout).ok()).map(|s| s.trim().to_owned());
        let placement = manifest.repo_id.as_ref().and_then(|repo| {
            let selector = format!("hf:{repo}{}", manifest.revision.as_ref().map(|r| format!("@{r}")).unwrap_or_default());
            Command::new("nest").args(["where", &selector, "--json"]).output().ok()
                .filter(|o| o.status.success()).and_then(|o| serde_json::from_slice::<serde_json::Value>(&o.stdout).ok())
        });
        let sealed = snapshot.starts_with("/mnt/sparknest") && placement.as_ref()
            .and_then(|v| v["hosts"].as_array()).is_some_and(|hosts| hosts.iter().any(|host|
                host["ready"] == true && host["host"].as_str() == local.as_deref()));
        if sealed { eprintln!("source auto: sealed local sparknest copy at {}", snapshot.display()); }
        else if snapshot.starts_with("/mnt/sparknest") {
            eprintln!("source auto: no sealed local copy; streaming from sparknest at {}", snapshot.display());
        } else { eprintln!("source auto: using local snapshot {} (no sparknest discovery)", snapshot.display()); }
    }
    let destination = args.destination.as_ref().context("--fetch needs --destination")?;
    ensure!(destination.is_absolute(), "destination must be an absolute snapshot directory");
    let host = args.host.as_deref();
    if let Some(host) = host {
        ensure!(!host.starts_with('-') && host.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-@".contains(&c)),
            "invalid SSH host {host}");
    }
    let available = |program: &str| Command::new("sh").args(["-c", &format!("command -v {program} >/dev/null")])
        .status().is_ok_and(|s| s.success());
    let installed_remote = |host: &str| Command::new("ssh").args([host, "command -v rdmasync >/dev/null"])
        .status().is_ok_and(|s| s.success());
    let remote_rdma = host.is_none_or(installed_remote) && source_host.as_deref().is_none_or(installed_remote);
    let program = if (host.is_some() || source_host.is_some()) && available("rdmasync") && remote_rdma { "rdmasync" } else {
        eprintln!("warning: rdmasync unavailable on both ends; using rsync over SSH/local transport");
        "rsync"
    };
    ensure!(!destination.to_string_lossy().contains(['\n', '\r']), "destination contains a line break");
    for file in &manifest.files {
        ensure!(!file.path.contains(['\n', '\r']) && !std::path::Path::new(&file.path).is_absolute()
            && std::path::Path::new(&file.path).components().all(|c| matches!(c, std::path::Component::Normal(_))),
            "unsafe file-list path {}", file.path);
    }
    let source_sizes = if let Some(host) = source_host.as_deref() { remote_sizes(host, &source_path, &manifest.files)? }
        else { manifest.files.iter().map(|file| source_path.join(&file.path).metadata().ok()
            .filter(|m| m.is_file()).map(|m| m.len())).collect() };
    for (file, actual) in manifest.files.iter().zip(&source_sizes) {
        ensure!(actual.is_some(), "source host {} lacks {}", source_host.as_deref().unwrap_or("local"), file.path);
        if let Some(expected) = file.bytes { ensure!(*actual == Some(expected), "source {} size {:?}, expected {expected}", file.path, actual); }
    }
    let existing_sizes = if let Some(host) = host { remote_sizes(host, destination, &manifest.files)? }
        else { manifest.files.iter().map(|file| destination.join(&file.path).metadata().ok()
            .filter(|m| m.is_file()).map(|m| m.len())).collect() };
    let mut selected = Vec::new();
    let mut bytes = 0u64;
    for ((file, existing), source_bytes) in manifest.files.iter().zip(existing_sizes).zip(&source_sizes) {
        let source_bytes = source_bytes.context("source size missing after validation")?;
        if args.force || existing != Some(source_bytes) {
            bytes = bytes.checked_add(source_bytes).context("transfer bytes overflow")?;
            selected.push(file);
        }
    }
    let list = selected.iter().map(|file| format!("{}\n", file.path)).collect::<String>();
    let src = if host.is_none() { source_host.as_deref().map_or_else(|| format!("{}/", source_path.display()),
        |peer| format!("{peer}:{}/", source_path.display())) } else { format!("{}/", source_path.display()) };
    let dest = host.map_or_else(|| format!("{}/", destination.display()),
        |host| format!("{host}:{}/", destination.display()));
    let list_arg = "--files-from=-";
    let mut command = Command::new(program);
    // -L materializes HF blob symlinks as ordinary snapshot files accepted by
    // every loader. Never retain links pointing outside the destination cache.
    command.args(["-aL", "--info=progress2", "--protect-args", list_arg]);
    if args.force { command.arg("--ignore-times"); }
    else { command.arg("--size-only"); }
    command.args([&src, &dest]);
    // rsync cannot copy between two remote endpoints. Run it on the source
    // peer, which pushes to the selected target using its normal SSH access.
    if let Some(peer) = source_host.as_deref().filter(|_| host.is_some()) {
        let remote = std::iter::once(program.to_string()).chain(command.get_args().map(|a| shell_quote(&a.to_string_lossy())))
            .collect::<Vec<_>>().join(" ");
        command = Command::new("ssh");
        if args.forward_agent { command.arg("-A"); }
        command.args([peer, &remote]);
    }
    eprintln!("host {}: {} files, {bytes} bytes", host.unwrap_or("local"), selected.len());
    if args.dry_run {
        let rendered = std::iter::once(command.get_program().to_string_lossy().into_owned()).chain(command.get_args().map(|a| shell_quote(&a.to_string_lossy())))
            .collect::<Vec<_>>().join(" ");
        println!("{rendered}");
        for file in &selected {
            let size = manifest.files.iter().position(|entry| entry.path == file.path)
                .and_then(|index| source_sizes[index]).context("selected source size")?;
            println!("{size}\t{}", file.path);
        }
        return Ok(());
    }
    if host.is_none() { std::fs::create_dir_all(destination)?; }
    if let Some(host) = host {
        let status = Command::new("ssh").args([host, &format!("mkdir -p -- {}", shell_quote(&destination.display().to_string()))]).status()?;
        ensure!(status.success(), "host {host}: cannot create snapshot destination");
    }
    let mut child = command.stdin(std::process::Stdio::piped()).spawn()
        .with_context(|| format!("starting {program} for host {}", host.unwrap_or("local")))?;
    child.stdin.take().context("transfer list stdin")?.write_all(list.as_bytes())?;
    let status = child.wait()?;
    ensure!(status.success(), "host {} transfer failed ({status}); files: {:?}", host.unwrap_or("local"),
        selected.iter().map(|f| &f.path).collect::<Vec<_>>());
    let actual_sizes = if let Some(host) = host { remote_sizes(host, destination, &manifest.files)? }
        else { manifest.files.iter().map(|file| destination.join(&file.path).metadata().ok()
            .filter(|m| m.is_file()).map(|m| m.len())).collect() };
    for ((file, actual), expected) in manifest.files.iter().zip(actual_sizes).zip(source_sizes) {
        ensure!(actual == expected && actual.is_some(), "host {}: {} size {actual:?}, expected {expected:?}", host.unwrap_or("local"), file.path);
    }
    if destination.parent().and_then(|p| p.file_name()).is_some_and(|name| name == "snapshots") {
        let refs = destination.parent().and_then(|p| p.parent()).context("HF cache root")?.join("refs");
        let revision = destination.file_name().context("snapshot revision")?.to_string_lossy();
        if let Some(host) = host {
            let reference = shell_quote(&refs.join("main").display().to_string());
            let script = format!("mkdir -p -- {} && {{ test -e {reference} || (set -C; printf '%s\\n' {} > {reference}); }}",
                shell_quote(&refs.display().to_string()), shell_quote(&revision));
            ensure!(Command::new("ssh").args([host, &script]).status()?.success(), "host {host}: cannot create refs/main");
        } else {
            std::fs::create_dir_all(&refs)?;
            match std::fs::OpenOptions::new().write(true).create_new(true).open(refs.join("main")) {
                Ok(mut file) => writeln!(file, "{revision}")?,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {},
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_unit_options_carry_an_explicit_non_cli_host() {
        use clap::Parser;
        let crate::cli::Commands::Plan(mut request) = crate::cli::Cli::try_parse_from(
            ["cuteafd", "plan", "/unused"]).unwrap().command else { panic!("plan") };
        request.test_host_memory = Some((20 << 30, 3 << 30));
        let options = options(&request).unwrap();
        assert!(options.layout.is_none());
        assert_eq!(options.inventory_host_memory, request.test_host_memory);
        assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/unused",
            "--test-host-memory", "20GiB"]).is_err());
    }
    use cuteafd_loader::plan::testing::{mimo_flash_config, mimo_flash_tensors, mimo_pro_config, mimo_pro_tensors,
        write_snapshot};

    fn configured_context(flags: &[&str]) -> cuteafd_loader::placement::planner::Context {
        use clap::Parser;
        let config = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/configs/muse-spark-exl3-min.config");
        let argv = [vec!["cuteafd", "plan", "model", "--layout", "--config", config.to_str().unwrap()],
            flags.to_vec()].concat();
        let crate::cli::Commands::Plan(mut request) = crate::cli::Cli::try_parse_from(argv).unwrap().command
            else { panic!("plan") };
        let cfg = launch_config(request.config.as_deref()).unwrap();
        resolve_context_tokens(&mut request, &cfg).unwrap();
        let layout = options(&request).unwrap().layout.unwrap();
        let value = flags.get(1).copied().unwrap_or(&cfg["MAX_CONTEXT_TOKENS"]);
        let crate::cli::Commands::ServeMuse(serve) = crate::cli::Cli::try_parse_from([
            "cuteafd", "serve-muse", "--snapshot", "/model", "--native-lib", "/native.so",
            "--max-context-tokens", value]).unwrap().command else { panic!("serve-muse") };
        let served = cuteafd_loader::placement::planner::Context::resolve(1048576, serve.max_context);
        let planned = cuteafd_loader::placement::planner::Context::resolve(1048576,
            (layout.context_tokens > 0).then_some(layout.context_tokens));
        assert_eq!(planned, served);
        planned
    }

    #[test]
    fn installed_kimi_min_auto_context_matches_serve_and_admitted_pool() -> Result<()> {
        use clap::Parser;
        use cuteafd_loader::placement::planner::Context;
        let config = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/configs/kimi-k3-exl3-k6-min.config");
        let cfg = launch_config(Some(&config))?;
        let hf_home = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME").map(PathBuf::from)
            .expect("CUTEAFD_DEPLOYMENT_TEST_HF_HOME is required");
        let snapshot = cuteafd_loader::model_cache_dir(&hf_home, &cfg["MODEL_ID"])
            .join("snapshots").join(&cfg["MODEL_REVISION"]);
        anyhow::ensure!(snapshot.is_dir(), "required installed Kimi checkpoint missing: {}", snapshot.display());
        let manifest = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS").map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap())
                .join(".cache/cuteafd/builds/placement-inputs/current/share/PROGRAMS.json"));
        anyhow::ensure!(manifest.is_file(), "installed Kimi context check requires {}", manifest.display());
        let cli = crate::cli::Cli::try_parse_from([
            "cuteafd", "plan", snapshot.to_str().unwrap(), "--layout", "--config", config.to_str().unwrap(),
            "--rtx", "1", "--spark-ranks", "6", "--speculator", "dflash2", "--vision", "off", "--audio", "off",
            "--workspace-manifest", manifest.to_str().unwrap(), "--prefill-rows", "1024",
            "--spark-memory-cap", "118111600640"])?;
        let crate::cli::Commands::Plan(mut request) = cli.command else { panic!("plan") };
        request.memory = cli.memory;
        request.hf_home = Some(hf_home);
        resolve_context_tokens(&mut request, &cfg)?;
        let mut request = file_args(request, &snapshot)?;
        let mut selected = options(&request)?;
        resolve_speculator(&mut request, &snapshot, "kimi_k3", &cfg)?;
        selected.layout.as_mut().unwrap().kimi_drafter_snapshot = request.drafter_snapshot.clone();
        let report = plan(&snapshot, &selected)?;
        anyhow::ensure!(report.fits && report.placement_supported, "installed Kimi minimum refused: {:?}", report.hints);
        let memory = report.memory_layout.context("installed Kimi minimum memory layout")?;
        let checkpoint: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let maximum = cuteafd_loader::serving_capacity::checkpoint_context_limit(&checkpoint)?.context("Kimi model maximum")?;
        let crate::cli::Commands::ServeKimi(serve) = crate::cli::Cli::try_parse_from([
            "cuteafd", "serve-kimi", "--snapshot", snapshot.to_str().unwrap(), "--native-lib", "/native.so",
            "--max-context-tokens", cfg.get("MAX_CONTEXT_TOKENS").map(String::as_str).unwrap_or("auto")])?.command
            else { panic!("serve-kimi") };
        let served = crate::shared::context::pool_context("kimi_k3",
            Context::resolve(maximum, serve.max_context.map(|tokens| tokens as u64)), memory.pool_tokens as usize)?;
        assert_eq!(served as u64, maximum.min(memory.pool_tokens));
        assert!(served <= cuteafd_loader::families::kimi_k3::COMPILED_CONTEXT);
        assert!(memory.notes.iter().any(|note| note.starts_with(&format!("Kimi API max context: {served} tokens;"))),
            "{:?}", memory.notes);
        Ok(())
    }

    #[test]
    fn plan_config_uses_installed_muse_context_pin() {
        use cuteafd_loader::placement::planner::{Context, Pin};
        assert_eq!(configured_context(&[]), Context { maximum: 1048576, pin: Pin::Exact(634304) });
    }

    #[test]
    fn plan_context_cli_overrides_installed_muse_config() {
        use cuteafd_loader::placement::planner::{Context, Pin};
        assert_eq!(configured_context(&["--max-context-tokens", "131072"]),
            Context { maximum: 1048576, pin: Pin::Exact(131072) });
        assert_eq!(configured_context(&["--max-context-tokens", "auto"]), Context::automatic(1048576));
    }

    #[test]
    fn plan_config_context_omitted_or_empty_stays_automatic() {
        for configured in [None, Some(""), Some("auto")] {
            let mut request = args(std::path::Path::new("model"), 4, false);
            request.context_tokens = None;
            let cfg = configured.map(|value| ("MAX_CONTEXT_TOKENS".into(), value.into()))
                .into_iter().collect();
            resolve_context_tokens(&mut request, &cfg).unwrap();
            assert_eq!(options(&request).unwrap().layout.unwrap().context_tokens, 0);
        }
    }

    #[test]
    fn plan_config_context_rejects_non_launcher_values() {
        for value in ["0", "-1", "+1", "01", "1048577", "18446744073709551616"] {
            let mut request = args(std::path::Path::new("model"), 4, false);
            request.context_tokens = None;
            let cfg = [("MAX_CONTEXT_TOKENS".into(), value.into())].into_iter().collect();
            assert!(resolve_context_tokens(&mut request, &cfg).unwrap_err().to_string()
                .contains("MAX_CONTEXT_TOKENS: context must be auto or a positive integer in 1..1048576"), "{value}");
        }
    }

    #[test]
    fn plan_prefill_waves_and_span_reach_both_layout_fields() {
        use clap::Parser;
        let parse = |argv: &[&str]| match crate::cli::Cli::try_parse_from(argv).unwrap().command {
            crate::cli::Commands::Plan(args) => options(&args).unwrap().layout.unwrap(),
            _ => panic!("expected plan command"),
        };
        let layout = parse(&["cuteafd", "plan", "model", "--layout", "--prefill-waves", "2",
            "--prefill-span-rows", "8192"]);
        assert_eq!((layout.prefill_lanes, layout.prefill_waves, layout.prefill_span_rows), (2, 2, Some(8192)));
        let default = parse(&["cuteafd", "plan", "model", "--layout"]);
        assert_eq!((default.prefill_lanes, default.prefill_waves, default.prefill_span_rows), (0, 1, None));
    }

    #[test]
    fn exlr8_native_cli_resolves_default_config_and_opt_out_before_admission() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type":"glm_moe_dsa","architectures":["GlmMoeDsaForCausalLM"]}"#).unwrap();
        let experts = dir.path().join("experts");
        std::fs::create_dir(&experts).unwrap();
        let header = br#"{"__metadata__":{"format":"exlr8-v2"}}"#;
        let mut index = (header.len() as u64).to_le_bytes().to_vec();
        index.extend_from_slice(header);
        std::fs::write(experts.join("exlr8-layer-3.index.safetensors"), index).unwrap();
        let config = dir.path().join("launch.config");
        for (selected, configured, no_speculator, include_speculator, expected) in [
            ("auto", None, false, false, "EXLR8 requires GPU embedding"),
            ("off", None, false, false, "EXLR8 requires GPU embedding"),
            ("auto", Some("off"), false, false, "EXLR8 requires GPU embedding"),
            ("auto", Some("mtp"), false, false, "SPECULATOR=mtp does not apply to glm5"),
            ("dflash2", Some("mtp"), false, false, "EXLR8 requires GPU embedding"),
            ("auto", Some("mtp"), true, false, "EXLR8 requires GPU embedding"),
            ("auto", None, true, true, "--no-speculator conflicts with --include-speculator"),
        ] {
            let mut request = args(dir.path(), 4, false);
            request.exlr8_manifest = Some("up1".into());
            // Stop before checkpoint payload validation after speculator admission succeeds.
            request.embedding_placement = Some(crate::shared::token_io::EmbedPlacement::Host);
            request.speculator = selected.into();
            request.no_speculator = no_speculator;
            request.include_speculator = include_speculator;
            if let Some(value) = configured {
                std::fs::write(&config, format!("SPECULATOR={value}\n")).unwrap();
                request.config = Some(config.clone());
            }
            let error = run_plan(request).unwrap_err().to_string();
            assert!(error.contains(expected), "{selected}/{configured:?}/{no_speculator}: {error}");
        }
    }

    #[test]
    fn exlr8_tp2_dflash_resolves_default_config_and_explicit_off_before_admission() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"),
            r#"{"model_type":"glm5_next","architectures":["Glm5NextForConditionalGeneration"]}"#).unwrap();
        let experts = dir.path().join("experts");
        std::fs::create_dir(&experts).unwrap();
        let header = br#"{"__metadata__":{"format":"exlr8-v2"}}"#;
        let mut index = (header.len() as u64).to_le_bytes().to_vec();
        index.extend_from_slice(header);
        std::fs::write(experts.join("exlr8-layer-3.index.safetensors"), index).unwrap();
        for configured in [None, Some("on"), Some("off")] {
            for explicit in [None, Some(false), Some(true)] {
                let mut request = args(dir.path(), 0, false);
                request.rtx = 2;
                request.exlr8_manifest = Some("k4".into());
                request.tp2_dflash = explicit;
                let cfg = configured.map(|value| ("TP2_DFLASH".into(), value.into()))
                    .into_iter().collect();
                let mut resolved = options(&request).unwrap();
                let family = cuteafd_loader::plan::exlr8::architecture(
                    &serde_json::from_slice(&std::fs::read(dir.path().join("config.json")).unwrap()).unwrap()).unwrap();
                resolve_exlr8_draft(&mut request, dir.path(), family, &cfg, &mut resolved).unwrap();
                let layout = resolved.layout.unwrap();
                assert_eq!(layout.tp2_dflash, explicit.unwrap_or(configured != Some("off")));
                // Invalid topology must fail in the early-return path, before native admission.
                request.rtx = 1;
                request.tp2_dflash = Some(true);
                let error = run_plan(request).unwrap_err().to_string();
                assert!(error.contains("TP2_DFLASH requires"), "{error}");
            }
        }
    }

    #[test]
    fn glmf_tp2_default_requires_two_gpus_dflash2_and_checkpoint_budget() {
        let dir = tempfile::tempdir().unwrap();
        for rtx in [1, 2] {
            for drafter_gib in [0.0, 12.0] {
                for speculator in ["dflash2", "off", "dspark"] {
                    let mut request = args(dir.path(), 0, false);
                    request.rtx = rtx;
                    request.drafter_gib = drafter_gib;
                    request.speculator = speculator.into();
                    let cfg = Default::default();
                    let eligible = rtx == 2 && drafter_gib == 0.0 && speculator == "dflash2";
                    assert_eq!(tp2_dflash(&request, "glm5_flash", &cfg).unwrap(), eligible);
                    request.tp2_dflash = Some(true);
                    assert_eq!(tp2_dflash(&request, "glm5_flash", &cfg).is_ok(), eligible);
                    request.tp2_dflash = Some(false);
                    let cfg = [("TP2_DFLASH".into(), "on".into())].into_iter().collect();
                    assert!(!tp2_dflash(&request, "glm5_flash", &cfg).unwrap());
                }
            }
        }
    }

    #[test]
    fn glmf_plan_preserves_explicit_drafter_representation_and_rows() {
        use clap::Parser;
        use cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation;
        for (flags, representation, mode) in [
            (vec![], GlmDraftRepresentation::Fp8Only, 0),
            (vec!["--draft-fp8", "false", "--draft-linear", "wide"], GlmDraftRepresentation::Bf16Only, 1),
            (vec!["--draft-fp8", "true", "--draft-linear", "w8a8"], GlmDraftRepresentation::Fp8Only, 2),
        ] {
            let argv = [vec!["cuteafd", "plan", "model", "--layout"], flags].concat();
            let crate::cli::Commands::Plan(request) = crate::cli::Cli::try_parse_from(argv).unwrap().command else { panic!("plan") };
            let layout = options(&request).unwrap().layout.unwrap();
            assert_eq!(layout.glmf_draft_representation, representation);
            assert_eq!(layout.glmf_draft_fp8_mode, mode);
        }
    }

    #[test]
    fn exlr8_prefill_activations_flag_defaults_to_a16_and_takes_only_a16_or_a8() {
        use clap::Parser;
        use cuteafd_loader::plan::exlr8::PrefillActivations;
        let parse = |extra: &[&str]| {
            let mut argv = vec!["cuteafd", "plan", "model", "--layout"];
            argv.extend_from_slice(extra);
            crate::cli::Cli::try_parse_from(argv).map(|cli| match cli.command {
                crate::cli::Commands::Plan(args) => options(&args).map(|o| {
                    let layout = o.layout.unwrap();
                    assert_eq!(layout.exlr8_prefill_pin, args.exlr8_prefill_activations);
                    (layout.exlr8_prefill_pin, layout.exlr8_prefill_activations)
                }),
                _ => panic!("expected plan command"),
            })
        };
        assert_eq!(parse(&[]).unwrap().unwrap(), (None, PrefillActivations::A16));
        assert_eq!(parse(&["--exlr8-prefill-activations", "a16"]).unwrap().unwrap(),
            (Some(PrefillActivations::A16), PrefillActivations::A16));
        assert_eq!(parse(&["--exlr8-prefill-activations", "a8"]).unwrap().unwrap(),
            (Some(PrefillActivations::A8), PrefillActivations::A8));
        for invalid in ["fp8", "A8", "", "a8,a16"] {
            assert!(parse(&["--exlr8-prefill-activations", invalid]).is_err(), "{invalid:?}");
        }
        // The charge is a layout charge: without --layout the flag has nothing to apply to.
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "model", "--exlr8-prefill-activations", "a8"]).unwrap();
        let crate::cli::Commands::Plan(args) = cli.command else { panic!("plan command") };
        assert!(options(&args).unwrap_err().to_string().contains("--layout"));
    }

    #[test]
    fn exlr8_prefill_activations_requires_an_exlr8_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        for activations in [cuteafd_loader::plan::exlr8::PrefillActivations::A16,
            cuteafd_loader::plan::exlr8::PrefillActivations::A8] {
            let request = PlanArgs { exlr8_prefill_activations: Some(activations), ..args(dir.path(), 4, false) };
            let error = run_plan(request).unwrap_err().to_string();
            assert!(error.contains("--exlr8-prefill-activations requires an EXLR8 checkpoint"), "{activations}: {error}");
        }
        // The same checkpoint plans without the flag, so the refusal is the flag's.
        run_plan(args(dir.path(), 4, false)).unwrap();
    }

    #[test]
    fn exlr8_mounted_cli_reports_fit_and_refuses_serving_modes() {
        use clap::Parser;
        let Some(snapshot) = std::env::var_os("CUTEAFD_EXLR8_TEST_SNAPSHOT") else { return; };
        let root = std::path::Path::new(&snapshot);
        let cli = crate::cli::Cli::try_parse_from([
            "cuteafd", "plan", root.to_str().unwrap(), "--exlr8-manifest", "up1",
            "--spark-ranks", "4", "--rtx", "1", "--json",
        ]).unwrap();
        let crate::cli::Commands::Plan(plan) = cli.command else { panic!("plan command"); };
        run_plan(plan).unwrap();
        let refusal = run_plan(args(root, 6, true)).unwrap_err().to_string();
        assert!(refusal.contains("EXLR8 is unsupported") && refusal.contains("kernels"));
        let invalid = PlanArgs { exlr8_manifest: Some("../bad".into()), ..args(root, 6, false) };
        assert!(run_plan(invalid).unwrap_err().to_string().contains("published name"));
        let files = PlanArgs { files: true, ..args(root, 6, false) };
        assert!(run_plan(files).unwrap_err().to_string().contains("planner-only"));
    }

    #[test]
    fn kimi_pairing_accepts_new_mc_revisions_and_k8_but_names_mismatches() -> Result<()> {
        use clap::Parser;
        let root = tempfile::tempdir()?;
        let target = root.path().join("another-revision");
        let draft = root.path().join("drafter");
        std::fs::create_dir_all(&target)?;
        std::fs::create_dir_all(&draft)?;
        let config = serde_json::json!({"model_type":"kimi_k31","architectures":["KimiK31ForConditionalGeneration"],
            "text_config":{"hidden_size":7168,"vocab_size":163840,"num_hidden_layers":93,"pad_token_id":163839}});
        let mut drafter = serde_json::json!({"speculators_config":{"verifier":{"name_or_path":"Inferact/Kimi-K3.1-Pro",
            "architectures":["KimiK31ForConditionalGeneration"]}},"target_hidden_size":null,
            "transformer_layer_config":{"hidden_size":7168,"vocab_size":163840},"draft_vocab_size":163840,
            "aux_hidden_state_layer_ids":[76,80,84,88,92],"mask_token_id":163839});
        std::fs::write(target.join("config.json"),serde_json::to_vec(&config)?)?;
        std::fs::write(target.join("cuteafd-mc.json"),br#"{"source":{"repo":"moonshotai/Kimi-K3.1-Pro","revision":"another-revision"}}"#)?;
        std::fs::write(draft.join("config.json"),serde_json::to_vec(&drafter)?)?;
        assert!(kimi_content_matches(&target,&draft)?);
        let crate::cli::Commands::Plan(mut request) = crate::cli::Cli::try_parse_from([
            "cuteafd","plan",target.to_str().unwrap(),"--core-view","k8-compact","--rtx","2",
            "--drafter-snapshot",draft.to_str().unwrap()
        ])?.command else { panic!("plan") };
        resolve_speculator(&mut request,&target,"kimi_k3",&Default::default())?;
        resolve_settings(&mut request,"kimi_k3",&Default::default())?;
        assert_eq!(request.speculator,"dspark"); assert_eq!(request.draft_fp8,Some(true));
        drafter["speculators_config"]["verifier"]["name_or_path"] = serde_json::json!("other/Wrong-Model");
        std::fs::write(draft.join("config.json"),serde_json::to_vec(&drafter)?)?;
        assert!(kimi_content_matches(&target,&draft).unwrap_err().to_string().contains("verifier.name_or_path"));
        request.speculator="dspark".into();
        assert!(resolve_speculator(&mut request,&target,"kimi_k3",&Default::default()).unwrap_err().to_string().contains("verifier.name_or_path"));
        for field in ["hidden_size","vocab_size","aux_hidden_state_layer_ids","mask_token_id"] {
            let mut changed=drafter.clone();
            changed["speculators_config"]["verifier"]["name_or_path"]=serde_json::json!("Inferact/Kimi-K3.1-Pro");
            match field {
                "hidden_size"=>changed["transformer_layer_config"][field]=serde_json::json!(123),
                "vocab_size"=>changed["draft_vocab_size"]=serde_json::json!(123),
                "aux_hidden_state_layer_ids"=>changed[field]=serde_json::json!([94]),
                _=>changed[field]=serde_json::json!(123),
            }
            assert!(kimi_pair_metadata(&config,&changed,&target).unwrap_err().to_string().contains(field));
        }
        Ok(())
    }

    #[test]
    fn native_draft_pairing_checks_checkpoint_geometry_taps_and_token_ids() -> Result<()> {
        let target=serde_json::json!({"architectures":["KimiK31ForConditionalGeneration"],
            "text_config":{"hidden_size":7168,"vocab_size":163840,"num_hidden_layers":93,
                "bos_token_id":163584,"eos_token_id":163586,"pad_token_id":163839}});
        let draft=serde_json::json!({"architectures":["Qwen3DSparkModel"],"num_target_layers":93,
            "hidden_size":7168,"vocab_size":163840,"draft_vocab_size":163840,
            "bos_token_id":163584,"eos_token_id":163586,"pad_token_id":163839,
            "dflash_config":{"mask_token_id":163824,"target_layer_ids":[7,23,67,83,91]}});
        let nonexistent=std::path::Path::new("/native-pairing-needs-no-side-files");
        for architecture in ["Qwen3DSparkModel","DSparkDraftModel","DFlash2DraftModel"] {
            let mut selected=draft.clone();selected["architectures"]=serde_json::json!([architecture]);
            kimi_pair_metadata(&target,&selected,nonexistent)?;
        }
        for field in ["hidden_size","vocab_size","draft_vocab_size","num_target_layers",
            "bos_token_id","eos_token_id","pad_token_id"] {
            let mut changed=draft.clone();changed[field]=serde_json::json!(123);
            assert!(kimi_pair_metadata(&target,&changed,nonexistent).unwrap_err().to_string().contains(field));
        }
        for taps in [serde_json::json!([93]),serde_json::json!([7,7]),serde_json::json!([])] {
            let mut changed=draft.clone();changed["dflash_config"]["target_layer_ids"]=taps;
            assert!(kimi_pair_metadata(&target,&changed,nonexistent).unwrap_err().to_string().contains("target_layer_ids"));
        }
        let mut changed=draft.clone();changed["dflash_config"]["mask_token_id"]=serde_json::json!(163840);
        assert!(kimi_pair_metadata(&target,&changed,nonexistent).unwrap_err().to_string().contains("mask_token_id"));
        Ok(())
    }

    #[test]
    fn automatic_external_drafter_requires_snapshot_and_config_but_off_does_not() -> Result<()> {
        use clap::Parser;
        let root = tempfile::tempdir()?;
        let target = root.path().join("target");
        std::fs::create_dir_all(&target)?;
        std::fs::write(target.join("config.json"), "{}")?;
        for family in ["glm5_flash", "kimi_k3"] {
            let crate::cli::Commands::Plan(mut request) = crate::cli::Cli::try_parse_from([
                "cuteafd", "plan", target.to_str().unwrap(), "--rtx", "2"
            ])?.command else { panic!("plan") };
            request.hf_home = Some(root.path().join("hf"));
            let model = if family == "glm5_flash" { GLMF_DEFAULT_DRAFTER } else { cuteafd_core::launch_settings::KIMI_K31_DSPARK };
            let revision = (family == "kimi_k3").then_some(cuteafd_core::launch_settings::KIMI_K31_DSPARK_REVISION);
            let error = require_default_drafter(&request, &Default::default(), model, revision).unwrap_err();
            assert!(error.to_string().contains(model) && error.to_string().contains("nest hf download"));
            if let Some(revision) = revision { assert!(error.to_string().contains(revision)); }
            let draft = root.path().join(format!("{family}-draft"));
            std::fs::create_dir_all(&draft)?;
            request.drafter_snapshot = Some(draft.clone());
            assert!(require_default_drafter(&request, &Default::default(), model, revision).unwrap_err().to_string().contains("config.json"));
            std::fs::write(draft.join("config.json"), "{}")?;
            assert_eq!(require_default_drafter(&request, &Default::default(), model, revision)?, draft);
            request.drafter_snapshot = None;
            request.speculator = "off".into();
            resolve_speculator(&mut request, &target, family, &Default::default())?;
            assert_eq!(request.speculator, "off");
        }
        Ok(())
    }

    #[test]
    fn kimi_mc_default_weight_precision_tracks_content_and_explicit_override() {
        use clap::Parser;
        for tp in ["1", "2"] { for view in ["original", "k6-compact", "k8-compact", "fp8-compact"] {
            let crate::cli::Commands::Plan(mut request) = crate::cli::Cli::try_parse_from([
                "cuteafd", "plan", "/renamed/local", "--rtx", tp, "--core-view", view
            ]).unwrap().command else { panic!("plan") };
            request.kimi_content_qualified = true;
            request.speculator = "dspark".into();
            resolve_settings(&mut request, "kimi_k3", &Default::default()).unwrap();
            assert_eq!(request.draft_fp8, Some(true));
            request.draft_fp8 = Some(false);
            resolve_settings(&mut request, "kimi_k3", &Default::default()).unwrap();
            assert_eq!(request.draft_fp8, Some(false));
        } }
    }

    #[test]
    fn expert_usage_plan_defaults_on_and_config_off_matches_explicit_cli_off() {
        use clap::Parser;
        for configured in [None, Some("on"), Some("off")] {
            for requested in [None, Some(false), Some(true)] {
                let mut argv = vec!["cuteafd", "plan", "/model", "--layout"];
                if let Some(enabled) = requested {
                    argv.push(if enabled { "--expert-usage" } else { "--expert-usage=false" });
                }
                let crate::cli::Commands::Plan(mut request) = crate::cli::Cli::try_parse_from(argv).unwrap().command
                    else { panic!("expected plan") };
                let cfg = configured.map(|value| ("EXPERT_USAGE".into(), value.into())).into_iter().collect();
                resolve_expert_usage(&mut request, &cfg).unwrap();
                assert_eq!(options(&request).unwrap().layout.unwrap().expert_usage,
                    requested.unwrap_or(configured != Some("off")));
            }
        }
        let mut request = args(std::path::Path::new("/model"), 6, false);
        assert!(resolve_expert_usage(&mut request, &[("EXPERT_USAGE".into(), "bad".into())].into()).is_err());
    }

    #[test]
    fn step5_plan_defaults_to_fp8_and_preserves_explicit_bf16() {
        use cuteafd_loader::families::qwen4::Qwen4KvCache;
        let mut args = args(std::path::Path::new("/model"), 6, false);
        assert_eq!(options(&args).unwrap().layout.unwrap().step5_kv, Qwen4KvCache::Fp8);
        for (selected, format) in [(cuteafd_core::MlaCacheFormat::Bf16, Qwen4KvCache::Bf16),
            (cuteafd_core::MlaCacheFormat::Fp8, Qwen4KvCache::Fp8)] {
            args.kv_format = Some(selected);
            assert_eq!(options(&args).unwrap().layout.unwrap().step5_kv, format);
        }
    }

    #[test]
    fn muse_plan_preserves_bf16_until_fp8_quality_qualification() {
        use cuteafd_loader::families::muse_spark::cache::MuseKvCache;
        let mut args = args(std::path::Path::new("/model"), 6, false);
        assert_eq!(options(&args).unwrap().layout.unwrap().muse_kv, MuseKvCache::Bf16);
        for (selected, format) in [(cuteafd_core::MlaCacheFormat::Bf16, MuseKvCache::Bf16),
            (cuteafd_core::MlaCacheFormat::Fp8, MuseKvCache::Fp8)] {
            args.kv_format = Some(selected);
            assert_eq!(options(&args).unwrap().layout.unwrap().muse_kv, format);
        }
    }

    #[test]
    fn inkling_plan_and_config_preserve_selected_global_kv() {
        use cuteafd_loader::families::inkling2::InklingKvCache;
        let mut args = args(std::path::Path::new("/model"), 6, false);
        assert_eq!(options(&args).unwrap().layout.unwrap().inkling_kv, InklingKvCache::Fp8);
        let config = [("KV_CACHE".into(), "fp8".into())].into_iter().collect();
        resolve_family_config(&mut args, "inkling2", &config).unwrap();
        assert_eq!(options(&args).unwrap().layout.unwrap().inkling_kv, InklingKvCache::Fp8);
        args.kv_format = Some(cuteafd_core::MlaCacheFormat::Bf16);
        resolve_family_config(&mut args, "inkling2", &config).unwrap();
        assert_eq!(options(&args).unwrap().layout.unwrap().inkling_kv, InklingKvCache::Bf16);
    }

    #[test]
    fn inkling_bundled_speculator_is_explicit_and_preserves_draft_storage() {
        use cuteafd_loader::{families::qwen4::Qwen4KvCache, plan::spec::DraftOwnership};
        let mut args = args(std::path::Path::new("/model"), 4, false);
        args.speculator = "auto".into();
        resolve_speculator(&mut args, std::path::Path::new("/model"), "inkling2", &Default::default()).unwrap();
        assert_eq!(args.speculator, "dspark");
        assert_eq!(DraftOwnership::selected("inkling2", &args.speculator), DraftOwnership::BundledDspark);
        args.speculator = "off".into();
        resolve_speculator(&mut args, std::path::Path::new("/model"), "inkling2", &Default::default()).unwrap();
        assert_eq!(DraftOwnership::selected("inkling2", &args.speculator), DraftOwnership::None);
        args.speculator = "dspark".into();
        resolve_speculator(&mut args, std::path::Path::new("/model"), "inkling2", &Default::default()).unwrap();
        assert_eq!(DraftOwnership::selected("inkling2", &args.speculator), DraftOwnership::BundledDspark);
        let config = [("DRAFT_KV_CACHE".into(), "bf16".into()), ("SPECULATOR_FP8".into(), "off".into())].into_iter().collect();
        resolve_family_config(&mut args, "inkling2", &config).unwrap();
        assert_eq!(options(&args).unwrap().layout.unwrap().draft_kv_cache, Qwen4KvCache::Bf16);
        args.draft_fp8 = Some(true);
        assert!(resolve_family_config(&mut args, "inkling2", &config).is_err());
    }

    #[test]
    fn mini_speculator_defaults_to_dflash2_and_preserves_explicit_modes() {
        use cuteafd_loader::plan::spec::DraftOwnership;
        for (requested, expected) in [("auto", "dflash2"), ("dflash2", "dflash2"), ("dspark", "dspark"), ("off", "off")] {
            let mut request = args(std::path::Path::new("/model"), 0, false);
            request.speculator = requested.into();
            resolve_speculator(&mut request, std::path::Path::new("/model"), "inkling2_mini", &Default::default()).unwrap();
            assert_eq!(request.speculator, expected);
            assert_eq!(DraftOwnership::selected("inkling2_mini", expected).uses_checkpoint(), expected == "dspark");
        }
    }

    #[test]
    fn kimi_plan_defaults_to_fp8_and_preserves_explicit_bf16() {
        use cuteafd_core::MlaCacheFormat;
        let mut args = args(std::path::Path::new("/model"), 6, false);
        assert_eq!(options(&args).unwrap().layout.unwrap().mla_kv, MlaCacheFormat::Fp8);
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            args.kv_format = Some(format);
            assert_eq!(options(&args).unwrap().layout.unwrap().mla_kv, format);
        }
    }

    #[test]
    fn kimi_structural_draft_default_matches_direct_serve_and_explicit_refusal() -> Result<()> {
        use clap::{CommandFactory, FromArgMatches, Parser};
        let root = tempfile::tempdir()?;
        for (heads, causal, expected) in [(64, false, true), (96, true, false)] {
            let block = if causal { 5 } else { 8 };
            let mut raw = serde_json::json!({"architectures":["DFlash2DraftModel"],"hidden_size":7168,
                "intermediate_size":14336,"num_hidden_layers":5,"num_attention_heads":heads,
                "num_key_value_heads":heads,"q_lora_rank":1536,"kv_lora_rank":512,
                "qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,"rms_norm_eps":1e-5,
                "vocab_size":163840,"draft_vocab_size":163840,"max_position_embeddings":1048576,
                "num_target_layers":93,"layer_types":vec!["sliding_attention";5],
                "sliding_window":if causal {1024}else{4096},"use_sliding_window":true,
                "is_causal":causal,"mla_use_output_gate":false,"fc_norm":false,"attention_bias":false,"hidden_act":"silu",
                "rope_parameters":{"rope_type":"yarn","rope_theta":50000.,"factor":32.,
                    "original_max_position_embeddings":32768,"beta_fast":32,"beta_slow":1,"mscale":1.,"mscale_all_dim":1.},
                "dflash_config":{"attention_mode":"mla","block_size":block,"conv_kernel_size":2,"conv_group_size":16,
                    "selector_rank":256,"selector_top_k":if causal{32}else{16},"target_layer_ids":[7,23,51,67,83,91],"mask_token_id":163824}});
            if causal { raw["mla_use_nope"]=false.into();raw["rope_interleave"]=true.into();raw["dflash_config"]["aux_hidden_stream"]="attn_res".into(); }
            std::fs::write(root.path().join("config.json"), serde_json::to_vec(&raw)?)?;
            for explicit in [None,Some(false),Some(true)] {
                let mut plan = args(std::path::Path::new("/model"),6,false);
                plan.layout=true;plan.rtx=2;plan.speculator="dflash2".into();plan.drafter_snapshot=Some(root.path().into());plan.tp2_dflash=explicit;
                let mut argv=vec!["cuteafd","serve-kimi","--snapshot","/model","--native-lib","/native.so","--split-device","1","--draft",root.path().to_str().unwrap()];
                if let Some(value)=explicit {argv.extend(["--tp2-dflash",if value{"true"}else{"false"}]);}
                let matches=crate::cli::Cli::command().try_get_matches_from(argv)?;
                let mut cli=crate::cli::Cli::from_arg_matches(&matches)?;
                let direct=crate::cli::resolve_tp2_dflash(&mut cli.command,&matches);
                let planned=tp2_dflash(&plan,"kimi_k3",&Default::default());
                if causal && explicit==Some(true) {
                    for error in [direct.unwrap_err(),planned.unwrap_err()] {
                        assert!(error.to_string().contains("kimik31_mla_draft_partial_causal_w1024_h96_m64"));
                    }
                } else {
                    direct?;
                    let crate::cli::Commands::ServeKimi(served)=cli.command else {panic!("serve-kimi")};
                    assert_eq!(planned?,explicit.unwrap_or(expected));
                    assert_eq!(served.tp2_dflash,explicit.unwrap_or(expected));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn tp2_dflash_plan_and_direct_serve_defaults_agree() {
        use clap::{CommandFactory, FromArgMatches, Parser};
        for (family, command) in [("glm5", "serve-glm"), ("mimo_v2", "serve-mimo"), ("muse_spark", "serve-muse")] {
            for split in [false, true] {
                for draft in [false, true] {
                    for explicit in [None, Some(false), Some(true)] {
                        let mut plan = vec!["cuteafd", "plan", "model", "--layout", "--rtx", if split { "2" } else { "1" },
                            "--speculator", if draft { "dflash2" } else { "off" }];
                        let mut serve = vec!["cuteafd", command, "--snapshot", "/model", "--native-lib", "/native.so"];
                        if split { serve.extend(["--split-device", "1"]); }
                        if draft { serve.extend(["--draft", "/draft"]); }
                        if let Some(enabled) = explicit {
                            let flag = if enabled { "true" } else { "false" };
                            plan.extend(["--tp2-dflash", flag]);
                            serve.extend(["--tp2-dflash", flag]);
                        }
                        let crate::cli::Commands::Plan(planned) = crate::cli::Cli::try_parse_from(plan).unwrap().command
                            else { panic!("expected plan") };
                        let matches = crate::cli::Cli::command().try_get_matches_from(serve).unwrap();
                        let mut parsed = crate::cli::Cli::from_arg_matches(&matches).unwrap();
                        crate::cli::resolve_tp2_dflash(&mut parsed.command, &matches).unwrap();
                        let served = match parsed.command {
                            crate::cli::Commands::ServeGlm(args) => args.engine.tp2_dflash,
                            crate::cli::Commands::ServeMimo(args) => args.engine.tp2_dflash,
                            crate::cli::Commands::ServeKimi(args) => args.tp2_dflash,
                            crate::cli::Commands::ServeMuse(args) => args.tp2_dflash,
                            _ => panic!("expected serve"),
                        };
                        let resolved = tp2_dflash(&planned, family, &Default::default());
                        if explicit == Some(true) && !(split && draft) { assert!(resolved.is_err()); }
                        else { assert_eq!(resolved.unwrap(), served); }
                        assert_eq!(served, explicit.unwrap_or(split && draft));
                    }
                }
            }
        }
    }

    #[test]
    fn tp2_dflash_explicit_off_overrides_config_on() {
        use clap::Parser;
        let argv = ["cuteafd", "plan", "model", "--layout", "--rtx", "2", "--speculator", "dflash2"];
        let cfg = [("TP2_DFLASH".into(), "on".into())].into();
        for (explicit, expected) in [("false", false), ("true", true)] {
            let crate::cli::Commands::Plan(request) = crate::cli::Cli::try_parse_from(
                argv.into_iter().chain(["--tp2-dflash", explicit])).unwrap().command
                else { panic!("expected plan") };
            for family in ["glm5", "mimo_v2", "kimi_k3", "glm5_flash", "muse_spark"] {
                assert_eq!(tp2_dflash(&request, family, &cfg).unwrap(), expected);
            }
            assert_eq!(options(&request).unwrap().layout.unwrap().tp2_dflash, expected);
        }
        let crate::cli::Commands::Plan(mut request) = crate::cli::Cli::try_parse_from(argv).unwrap().command
            else { panic!("expected plan") };
        for family in ["glm5", "mimo_v2", "kimi_k3", "glm5_flash", "muse_spark"] {
            assert!(tp2_dflash(&request, family, &Default::default()).unwrap());
            assert!(!tp2_dflash(&request, family, &[("TP2_DFLASH".into(), "off".into())].into()).unwrap());
            assert!(tp2_dflash(&request, family, &cfg).unwrap());
        }
        request.drafter_gib = 8.0;
        assert!(!tp2_dflash(&request, "glm5", &Default::default()).unwrap());
        assert!(tp2_dflash(&request, "glm5", &cfg).is_err());
        request.drafter_gib = 0.0;
        request.rtx = 1;
        for family in ["glm5", "mimo_v2", "kimi_k3", "glm5_flash", "muse_spark"] {
            assert!(!tp2_dflash(&request, family, &Default::default()).unwrap());
            assert!(tp2_dflash(&request, family, &cfg).is_err());
        }
        // Mistral Large 4 bundled dspark defaults TP2 on with 2 RTX GPUs, honors off/--drafter-gib, refuses 1 RTX.
        let ml4_argv = ["cuteafd", "plan", "model", "--layout", "--rtx", "2", "--speculator", "dspark"];
        for (explicit, expected) in [("false", false), ("true", true)] {
            let crate::cli::Commands::Plan(ml4) = crate::cli::Cli::try_parse_from(
                ml4_argv.into_iter().chain(["--tp2-dflash", explicit])).unwrap().command
                else { panic!("expected plan") };
            assert_eq!(tp2_dflash(&ml4, "mistral_large4", &cfg).unwrap(), expected);
        }
        let crate::cli::Commands::Plan(mut ml4) = crate::cli::Cli::try_parse_from(ml4_argv).unwrap().command
            else { panic!("expected plan") };
        assert!(tp2_dflash(&ml4, "mistral_large4", &Default::default()).unwrap());
        assert!(!tp2_dflash(&ml4, "mistral_large4", &[("TP2_DFLASH".into(), "off".into())].into()).unwrap());
        ml4.drafter_gib = 8.0;
        assert!(!tp2_dflash(&ml4, "mistral_large4", &Default::default()).unwrap());
        assert!(tp2_dflash(&ml4, "mistral_large4", &cfg).is_err());
        ml4.drafter_gib = 0.0;
        ml4.rtx = 1;
        assert!(!tp2_dflash(&ml4, "mistral_large4", &Default::default()).unwrap());
        assert!(tp2_dflash(&ml4, "mistral_large4", &cfg).is_err());
    }

    #[test]
    fn v4_and_v41_tp2_dspark_cli_and_config_resolve_before_layout() {
        use clap::Parser;
        let argv = ["cuteafd", "plan", "model", "--layout", "--rtx", "2", "--speculator", "dspark"];
        let crate::cli::Commands::Plan(mut request) = crate::cli::Cli::try_parse_from(argv).unwrap().command
            else { panic!("expected plan") };
        for (configured, expected) in [(None, true), (Some(""), true), (Some("off"), false), (Some("on"), true)] {
            let cfg = configured.map(|value| [("TP2_DSPARK_EXPERTS".into(), value.into())].into()).unwrap_or_default();
            for family in ["deepseek_v4", "deepseek_v41"] {
                assert_eq!(tp2_dspark_experts(&request, family, &cfg).unwrap(), expected);
            }
        }
        let cfg = [("TP2_DSPARK_EXPERTS".into(), "on".into())].into();
        for (family, rtx, layout, speculator) in [("glm5_flash", 2, true, "dspark"),
            ("deepseek_v41", 1, true, "dspark"), ("deepseek_v41", 2, false, "dspark"),
            ("deepseek_v41", 2, true, "off"), ("deepseek_v4", 1, true, "dspark"),
            ("deepseek_v4", 2, false, "dspark"), ("deepseek_v4", 2, true, "off")] {
            request.rtx = rtx;
            request.layout = layout;
            request.speculator = speculator.into();
            assert!(!tp2_dspark_experts(&request, family, &Default::default()).unwrap());
            let off = [("TP2_DSPARK_EXPERTS".into(), "off".into())].into();
            assert!(!tp2_dspark_experts(&request, family, &off).unwrap());
            assert!(tp2_dspark_experts(&request, family, &cfg).is_err());
        }
        let crate::cli::Commands::Plan(request) = crate::cli::Cli::try_parse_from(
            argv.into_iter().chain(["--tp2-dspark-experts"])).unwrap().command else { panic!("expected plan") };
        assert!(options(&request).unwrap().layout.unwrap().tp2_dspark_experts);
        assert!(tp2_dspark_experts(&request, "deepseek_v41", &Default::default()).unwrap());
        assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "model", "--tp2-dspark-experts"]).is_err());
        let mut request = request;
        request.tp2_dspark_experts = false;
        let invalid = [("TP2_DSPARK_EXPERTS".into(), "invalid".into())].into();
        assert!(tp2_dspark_experts(&request, "deepseek_v41", &invalid).is_err());
    }

    fn args(model: &std::path::Path, spark_ranks: usize, require_ready: bool) -> PlanArgs {
        PlanArgs {
            kimi_content_qualified: false,
            explicit_settings: Default::default(),

            test_host_memory: Some(cuteafd_loader::plan::testing::SimulatedHost::raptor().host_ram),
            fp8_prefill: Default::default(),
            attention: Default::default(),
            draft_kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,
            draft_context_window: 0,
            expert_partial_bf16_max_rows: cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
            exlr8_manifest: None,
            exlr8_prefill_activations: None,
            mapped_scale_preload_gib: 0.0,
            vision_replicas: 1,
            vision: cuteafd_loader::plan::MediaMode::Auto,
            audio: cuteafd_loader::plan::MediaMode::Off,
            chat_template_from: None,
            model: model.display().to_string(),
            embedding_placement: Some(crate::shared::token_io::EmbedPlacement::Gpu),
            revision: None,
            hf_home: None,
            spark_ranks: Some(spark_ranks),
            available_sparks: 6,
            spark_range: false,
            coordinator_weight_budget_gib: Some(80.0),
            memory: crate::cli::MemoryArgs { spark_memory_cap: Some(100 << 30), ..Default::default() },
            json: true,
            resolve_settings_only: false,
            checkpoint_model_id: None,
            deployment: false, deployment_peers: vec![], spark_memory_samples: None,
            kv_pool_size: None,
            files: false, role: None, host: None, file_layout: None, fetch_parallel: 2, fetch: false, destination: None, source: None, forward_agent: false,
            dry_run: false, force: false, include_speculator: false,
            config: None, speculator: "auto".into(), no_speculator: false,
            drafter_snapshot: None, draft_fp8: None, draft_linear: Default::default(), vision_snapshot: None, audio_snapshot: None,
            require_ready,
            layout: true,
            rtx: 2,
            pool_tokens: None,
            kv_format: Default::default(),
            fp8_decode: false, mtp_fp8_head: true, attention_plan_rows: None,
            kda_fp8: crate::families::glm5_flash::fp8::KdaFp8::Row128,
            kda_state: crate::families::glm5_flash::engine::KdaState::F32,
            head_format: crate::families::glm5_flash::head::GlmfHeadFormat::Fp8, target_head: None,
            draft_head: crate::shared::speculation::dflash::DraftHead::Exact,
            drafter_gib: 0.0,
            rtx_expert_layers: None,
            deprecated_peer_expert_ranges: false,
            exchange_f32: false,
            attention_placement: "auto".into(),
            context_tokens: Some(262144),
            prefill_rows: 0, expert_usage: Some(false),
            prefill_waves: 0, prefill_span_rows: 0, decode_lanes: None,
            decode_rows: cuteafd_loader::plan::decode_rows::DecodeRowsSetting::AUTO, draft_length: None,
            index_cache: Some(crate::families::glm5_flash::engine::IndexCache::Keys),
            replay_records: Some(crate::cli::PlanReplayRecords::Own),
            concurrency: 8,
            tp2_dspark_experts: false,
            tp2_dflash: None,
            state_slots: None,
            mark_lanes: None,
            prefix_slots: None,
            host_cache_bytes: crate::shared::prefix::HostBudget::Auto,
            media_cache_bytes: None,
            prefix_cache_entries: None,
            prefix_cache_mark_mib: 2048,
            mimo_prefix_draft: false,
            mimo_rings: 0,
            mimo_swa_kv_cache: crate::families::mimo_v2::SwaKvCacheArg::Int8,
            mimo_kv_cache: crate::families::mimo_v2::KvCacheArg::Int8,
            expert_input: crate::families::mimo_v2::engine::ExpertInput::Fp8,
            draft_sequences: None,
            draft_context_slots: None,
            decode_graphs: "on".into(),
            physical_sms: None,
            mimo_draft_bf16: false,
            glm5_draft_bf16: false,
            mimo_expert_manifest: None,
            prefix_marks: crate::families::glm5_flash::prefix::PrefixMarks::Arena,
            native_mtp_layers: 3,
            workspace_manifest: None,
        }
    }

    #[test]
    fn standalone_repos_are_inventory_only_when_the_host_runs_that_role() {
        use cuteafd_loader::plan::files::{manifest_roles, ReadRole};
        let main = tempfile::tempdir().unwrap();
        write_snapshot(main.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let draft = tempfile::tempdir().unwrap();
        write_snapshot(draft.path(), &serde_json::json!({"model_type":"external_drafter"}),
            &[cuteafd_loader::plan::testing::t("draft.weight", "BF16", &[2])], None);
        let mut args = args(main.path(), 4, false);
        args.drafter_snapshot = Some(draft.path().into());
        let spark = ReadRole::Spark { rank: 0, world: 4 };
        let mut manifest = manifest_roles(main.path(), &[spark]).unwrap();
        attach_snapshots(&mut manifest, &[spark], &args).unwrap();
        assert!(manifest.additional_snapshots.is_empty());
        let coordinator = ReadRole::Coordinator { local_experts: false, speculator: true };
        let mut manifest = manifest_roles(main.path(), &[coordinator]).unwrap();
        attach_snapshots(&mut manifest, &[coordinator], &args).unwrap();
        assert_eq!(manifest.additional_snapshots.len(), 1);
        assert!(manifest.additional_snapshots[0].files.iter().any(|f| f.path == "model-00001-of-00001.safetensors"));
    }

    #[test]
    fn mimo_external_drafter_is_owned_only_by_dflash2() {
        let snapshot = tempfile::tempdir().unwrap();
        write_snapshot(snapshot.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        // A bad bundled config must be irrelevant to both off and native MTP.
        std::fs::create_dir(snapshot.path().join("dflash")).unwrap();
        std::fs::write(snapshot.path().join("dflash/config.json"), "{}").unwrap();
        std::fs::write(snapshot.path().join("dflash/dflash_draft_model.safetensors"), "").unwrap();
        for speculator in ["off", "mtp", "dflash2"] {
            let mut layout = cuteafd_loader::plan::layout::LayoutOptions {
                host_memory: Some(cuteafd_loader::plan::testing::SimulatedHost::raptor().host_ram), native_mtp_layers: 0,
                host_cache_bytes: Some(0), ..Default::default() };
            resolve_mimo_drafter(&mut layout, speculator);
            assert_eq!(layout.mimo_drafter_disabled, speculator != "dflash2");
            let report = plan(snapshot.path(), &cuteafd_loader::plan::PlanOptions {
                layout: Some(layout), placement: cuteafd_loader::plan::ExpertPlacement::Sparks { ranks: 4 },
                ..Default::default() }).unwrap();
            if speculator == "dflash2" {
                assert!(!report.placement_supported, "the selected drafter's invalid geometry must refuse");
            } else {
                assert!(report.placement_supported, "{speculator}: {:?}", report.hints);
                assert!(!report.memory_layout.unwrap().devices[0].items.iter()
                    .any(|item| item.category == cuteafd_core::memory_layout::Category::Drafter));
            }
        }
    }

    #[test]
    fn kimi_tp2_dflash_config_preserves_explicit_cli_false() {
        use cuteafd_loader::plan::testing::*;
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &kimi_k3_config(), &[t("unused.weight", "BF16", &[2])], None);
        let config = target.path().join("pair.config");
        std::fs::write(&config, "SPECULATOR=off\nTP2_DFLASH=on\nVISION=off\nAUDIO=off\n").unwrap();
        let mut input = args(target.path(), 6, false);
        input.config = Some(config.clone());
        assert_eq!(file_args(input.clone(), target.path()).unwrap().tp2_dflash, Some(true));
        input.tp2_dflash = Some(false);
        assert_eq!(file_args(input.clone(), target.path()).unwrap().tp2_dflash, Some(false));
        std::fs::write(&config, "SPECULATOR=off\nTP2_DFLASH=auto\nVISION=off\nAUDIO=off\n").unwrap();
        assert!(file_args(input, target.path()).is_err());
    }

    #[test]
    fn minimax_speculator_defaults_on_for_every_placement_and_preserves_off() {
        use cuteafd_loader::plan::spec::DraftOwnership;
        let target = tempfile::tempdir().unwrap();
        for gpus in [1, 2] {
            for placement in ["auto", "heads", "layers"] {
                for configured in [None, Some("auto"), Some("off"), Some("dspark")] {
                    for explicit in ["auto", "off", "dspark"] {
                        let mut input = args(target.path(), 4, false);
                        input.rtx = gpus;
                        input.attention_placement = placement.into();
                        input.speculator = explicit.into();
                        let cfg = configured.map(|mode| ("SPECULATOR".into(), mode.into()))
                            .into_iter().collect();
                        resolve_speculator(&mut input, target.path(), "minimax_m3", &cfg).unwrap();
                        let enabled = explicit != "off" && !(explicit == "auto" && configured == Some("off"));
                        assert_eq!(input.speculator, if enabled { "dspark" } else { "off" });
                        assert_eq!(DraftOwnership::selected("minimax_m3", &input.speculator),
                            if enabled { DraftOwnership::BundledDspark } else { DraftOwnership::None });
                        input.no_speculator = true;
                        resolve_speculator(&mut input, target.path(), "minimax_m3", &cfg).unwrap();
                        assert_eq!(input.speculator, "off");
                    }
                }
            }
        }
    }

    #[test]
    fn muse_speculator_policy_defaults_to_dflash2_and_honors_off() {
        let target = tempfile::tempdir().unwrap();
        let draft = tempfile::tempdir().unwrap();
        let cfg = std::collections::BTreeMap::new();
        let mut automatic = args(target.path(), 4, false);
        resolve_speculator(&mut automatic, target.path(), "muse_spark", &cfg).unwrap();
        assert_eq!(automatic.speculator, "dflash2");
        automatic = args(target.path(), 4, false);
        automatic.drafter_snapshot = Some(draft.path().into());
        resolve_speculator(&mut automatic, target.path(), "muse_spark", &cfg).unwrap();
        assert_eq!(automatic.speculator, "dflash2");
        for mode in ["off", "dflash2", "mtp", "dspark"] {
            let mut explicit = args(target.path(), 4, false);
            explicit.speculator = mode.into();
            let resolved = resolve_speculator(&mut explicit, target.path(), "muse_spark", &cfg);
            assert_eq!(resolved.is_ok(), matches!(mode, "off" | "dflash2"), "{mode}");
        }
        let mut disabled = automatic.clone();
        disabled.no_speculator = true;
        assert!(resolve_speculator(&mut disabled, target.path(), "muse_spark", &cfg).is_err());
        disabled.drafter_snapshot = None;
        resolve_speculator(&mut disabled, target.path(), "muse_spark", &cfg).unwrap();
        assert_eq!(disabled.speculator, "off");
    }

    #[test]
    fn muse_launch_config_resolves_both_pinned_drafters() {
        use cuteafd_loader::plan::testing::*;
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &muse_spark_config(), &[t("unused.weight", "BF16", &[2])], None);
        let home = tempfile::tempdir().unwrap();
        for (model, revision) in [
            ("RadixArk/Muse-Spark-Open-DFlash2", "46c90f412549c472941a68382d91a99e5e6372a0"),
            ("nvidia/Muse-Spark-Open-816B-A42B-DFlash2", "22e2669e4c60689f9bff0749a34d9d8e59ce7cd6"),
        ] {
            let draft = cuteafd_loader::model_cache_dir(home.path(), model).join("snapshots").join(revision);
            std::fs::create_dir_all(&draft).unwrap();
            std::fs::write(draft.join("config.json"), "{}").unwrap();
            let config = target.path().join("pair.config");
            std::fs::write(&config, format!("SPECULATOR=dflash2\nSPECULATOR_MODEL_ID={model}\nSPECULATOR_MODEL_REVISION={revision}\nVISION=off\nAUDIO=off\n")).unwrap();
            let mut input = args(target.path(), 4, false);
            input.config = Some(config);
            input.hf_home = Some(home.path().into());
            let resolved = file_args(input, target.path()).unwrap();
            assert_eq!(resolved.speculator, "dflash2");
            assert_eq!(resolved.drafter_snapshot.as_deref(), Some(draft.as_path()));
        }
    }

    #[test]
    fn muse_default_resolves_the_measured_nvidia_revision_not_main() {
        use cuteafd_loader::plan::testing::*;
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &muse_spark_config(), &[t("unused.weight", "BF16", &[2])], None);
        let home = tempfile::tempdir().unwrap();
        let model = "nvidia/Muse-Spark-Open-816B-A42B-DFlash2";
        let revision = "22e2669e4c60689f9bff0749a34d9d8e59ce7cd6";
        let cache = cuteafd_loader::model_cache_dir(home.path(), model);
        for rev in [revision, "unqualified-main"] {
            let draft = cache.join("snapshots").join(rev);
            std::fs::create_dir_all(&draft).unwrap();
            std::fs::write(draft.join("config.json"), "{}").unwrap();
        }
        std::fs::create_dir_all(cache.join("refs")).unwrap();
        std::fs::write(cache.join("refs/main"), "unqualified-main").unwrap();
        let mut input = args(target.path(), 4, false);
        input.hf_home = Some(home.path().into());
        let resolved = file_args(input.clone(), target.path()).unwrap();
        assert_eq!(resolved.speculator, "dflash2");
        assert_eq!(resolved.drafter_snapshot, Some(cache.join("snapshots").join(revision)));
        input.no_speculator = true;
        let disabled = file_args(input, target.path()).unwrap();
        assert_eq!(disabled.speculator, "off");
        assert!(disabled.drafter_snapshot.is_none());
    }

    #[test]
    #[ignore = "requires the exact hardware pair configs and downloaded snapshots"]
    fn muse_exact_pair_configs_use_the_real_snapshot_resolver() {
        let configs = std::env::var_os("CUTEAFD_MUSE_PAIR_CONFIGS")
            .expect("set CUTEAFD_MUSE_PAIR_CONFIGS to the two exact config paths separated by colon");
        let configs = std::env::split_paths(&configs).collect::<Vec<_>>();
        assert_eq!(configs.len(), 2);
        for config in configs {
            let cfg = launch_config(Some(&config)).unwrap();
            let home = default_hf_home();
            let target = resolve_snapshot_at_revision(&cfg["MODEL_ID"], Some(&home), Some(&cfg["MODEL_REVISION"]))
                .unwrap().snapshot_path.unwrap();
            let mut input = args(&target, 4, false);
            input.config = Some(config.clone());
            let resolved = file_args(input, &target).unwrap();
            assert_eq!(resolved.speculator, "dflash2", "{}", config.display());
            let draft = resolve_snapshot_at_revision(&cfg["SPECULATOR_MODEL_ID"], Some(&home),
                Some(&cfg["SPECULATOR_MODEL_REVISION"])).unwrap().snapshot_path.unwrap();
            assert_eq!(resolved.drafter_snapshot.as_deref(), Some(draft.as_path()), "{}", config.display());
        }
    }

    #[test]
    fn qwen_recursive_depth_default_and_overrides_are_resolved_once() {
        let root = tempfile::tempdir().unwrap();
        for (mode, explicit, configured, expected) in [
            ("auto", None, None, 3), ("mtp", None, None, 3), ("off", None, None, 0),
            ("auto", None, Some(2), 2), ("mtp", Some(4), Some(2), 4)] {
            let mut request = args(root.path(), 2, false);
            request.speculator = mode.into();
            if let Some(depth) = explicit {
                request.native_mtp_layers = depth;
                request.explicit_settings.insert("native_mtp_layers".into());
            }
            let cfg = configured.map(|depth: usize| [("SPECULATOR_DEPTH".into(), depth.to_string())].into()).unwrap_or_default();
            let resolved = resolve_settings(&mut request, "qwen4", &cfg).unwrap();
            assert_eq!(request.native_mtp_layers, expected);
            assert_eq!(resolved["native_mtp_layers"].value, serde_json::json!(expected));
        }
    }

    #[test]
    fn qwen_speculator_auto_is_deferred_but_overrides_and_file_owners_are_explicit() {
        use cuteafd_loader::plan::testing::*;
        let target = tempfile::tempdir().unwrap();
        let config = qwen4_config(4);
        write_snapshot(target.path(), &config, &[t("mtp.layers.0.enorm.weight", "BF16", &[2560])], None);
        for (selected, configured, disabled, expected) in [
            ("auto", None, false, "auto"),
            ("auto", Some("auto"), false, "auto"),
            ("auto", Some("mtp"), false, "mtp"),
            ("auto", Some("off"), false, "off"),
            ("mtp", Some("off"), false, "mtp"),
            ("off", Some("mtp"), false, "off"),
            ("auto", None, true, "off"),
        ] {
            let mut request = args(target.path(), 2, false);
            request.speculator = selected.into();
            request.no_speculator = disabled;
            let cfg = configured.map(|value| std::collections::BTreeMap::from([("SPECULATOR".into(), value.into())]))
                .unwrap_or_default();
            resolve_speculator(&mut request, target.path(), "qwen4", &cfg).unwrap();
            assert_eq!(request.speculator, expected);
        }
        let files = file_args(args(target.path(), 2, false), target.path()).unwrap();
        assert_eq!(files.speculator, "mtp", "file inventory retains MTP's checkpoint superset");
        let mut off = args(target.path(), 2, false);
        off.speculator = "off".into();
        assert_eq!(file_args(off, target.path()).unwrap().speculator, "off");
    }

    #[test]
    fn hy4_flash_bundled_dfly2_auto_is_independent_of_spark_layout() {
        use cuteafd_loader::plan::spec::DraftOwnership;
        let target = tempfile::tempdir().unwrap();
        for bundled in [true, false] {
            let config = serde_json::json!({"dfly_num_hidden_layers": if bundled { 5 } else { 0 }});
            std::fs::write(target.path().join("config.json"), config.to_string()).unwrap();
            for ranks in [Some(0), Some(1), Some(4), None] {
                for configured in [None, Some("auto"), Some("off")] {
                    let mut request = args(target.path(), 0, false);
                    request.spark_ranks = ranks;
                    let cfg = configured.map(|value| std::collections::BTreeMap::from([
                        ("SPECULATOR".into(), value.into())])).unwrap_or_default();
                    resolve_speculator(&mut request, target.path(), "hy4_flash", &cfg).unwrap();
                    let expected = if bundled && configured != Some("off") { "dfly2" } else { "off" };
                    assert_eq!(request.speculator, expected, "ranks={ranks:?}, config={configured:?}");
                    assert_eq!(DraftOwnership::selected("hy4_flash", &request.speculator),
                        if expected == "dfly2" { DraftOwnership::BundledDfly2 } else { DraftOwnership::None });
                }
            }
        }
    }

    #[test]
    fn hy4_bundled_dfly2_auto_is_independent_of_spark_layout() {
        use cuteafd_loader::plan::spec::DraftOwnership;
        let target = tempfile::tempdir().unwrap();
        for bundled in [true, false] {
            let config = serde_json::json!({"dfly_num_hidden_layers": if bundled { 5 } else { 0 }});
            std::fs::write(target.path().join("config.json"), config.to_string()).unwrap();
            for ranks in [Some(0), Some(4), Some(6), None] {
                for configured in [None, Some("auto"), Some("off")] {
                    let mut request = args(target.path(), 0, false);
                    request.spark_ranks = ranks;
                    let cfg = configured.map(|value| std::collections::BTreeMap::from([
                        ("SPECULATOR".into(), value.into())])).unwrap_or_default();
                    resolve_speculator(&mut request, target.path(), "hy4", &cfg).unwrap();
                    let expected = if bundled && configured != Some("off") { "dfly2" } else { "off" };
                    assert_eq!(request.speculator, expected, "ranks={ranks:?}, config={configured:?}");
                    assert_eq!(DraftOwnership::selected("hy4", &request.speculator),
                        if expected == "dfly2" { DraftOwnership::BundledDfly2 } else { DraftOwnership::None });
                }
            }
        }
    }

    #[test]
    fn auto_speculators_equal_the_sanitized_release_configs() {
        use cuteafd_loader::plan::testing::*;
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/fixtures/release-configs");
        let v41 = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"))).unwrap();
        let v4 = serde_json::json!({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"});
        for (name, config, expected) in [
            ("glm53-exl3", glm5_config(), "dflash2"),
            ("glm53f-exl3", glm5_flash_config(2), "dflash2"),
            ("mimo26-flash", mimo_flash_mopd_config(), "dflash2"),
            ("qwen38-exl3", qwen4_config(4), "mtp"),
            ("v4-flash", v4, "dspark"), ("v41-flash", v41, "dspark"),
        ] {
            let target = tempfile::tempdir().unwrap();
            write_snapshot(target.path(), &config, &[t("unused.weight", "BF16", &[2])], None);
            if name == "mimo26-flash" { std::fs::create_dir(target.path().join("dflash")).unwrap(); }
            let draft = tempfile::tempdir().unwrap();
            std::fs::write(draft.path().join("config.json"), "{}").unwrap();
            let mut auto = args(target.path(), 4, false);
            if name == "mimo26-flash" { auto.checkpoint_model_id = Some("XiaomiMiMo/MiMo-V2.6-Flash-MOPD".into()); }
            // Resolve policy independently of the machine's HF cache.
            if name.starts_with("glm") { auto.drafter_snapshot = Some(draft.path().into()); }
            let automatic = file_args(auto.clone(), target.path()).unwrap();
            auto.config = Some(root.join(format!("{name}.config")));
            let configured = file_args(auto, target.path()).unwrap();
            assert_eq!(automatic.speculator, expected, "{name}");
            assert_eq!(automatic.speculator, configured.speculator, "{name}");
            assert_eq!(automatic.drafter_snapshot, configured.drafter_snapshot);
        }
    }

    #[test]
    fn glm_flash_coordinator_files_need_only_the_primary_checkpoint() {
        use cuteafd_loader::plan::testing::*;
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &glm5_flash_config(2), &[t("lm_head.weight", "BF16", &[2])], None);
        let mut input = args(target.path(), 4, false);
        input.speculator = "off".into();
        let roles = [cuteafd_loader::plan::files::ReadRole::Coordinator { local_experts: false, speculator: false }];
        let mut manifest = cuteafd_loader::plan::files::manifest_roles(target.path(), &roles).unwrap();
        let primary_bytes = manifest.total_bytes;
        attach_snapshots(&mut manifest, &roles, &input).unwrap();
        assert!(manifest.additional_snapshots.is_empty());
        assert_eq!(manifest.total_bytes, primary_bytes);
    }

    #[test]
    fn configured_off_is_explicit_but_include_speculator_cannot_be_silenced() {
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let config = target.path().join("serving.config");
        std::fs::write(&config, "SPECULATOR=off\nVISION=off\nAUDIO=off\n").unwrap();
        let mut input = args(target.path(), 4, false);
        input.config = Some(config);
        let off = file_args(input.clone(), target.path()).unwrap();
        assert_eq!(off.speculator, "off");
        input.include_speculator = true;
        assert_eq!(file_args(input.clone(), target.path()).unwrap().speculator, "mtp");
        input.no_speculator = true;
        assert!(file_args(input, target.path()).is_err());
        std::fs::create_dir(target.path().join("dflash")).unwrap();
        let explicit_off = target.path().join("explicit-off.config");
        std::fs::write(&explicit_off, "SPECULATOR=off\n").unwrap();
        let mut input = args(target.path(), 4, false);
        input.config = Some(explicit_off);
        assert_eq!(file_args(input, target.path()).unwrap().speculator, "off");
    }

    #[test]
    fn config_local_experts_and_spark_only_layout_match_launcher_requirements() {
        use cuteafd_loader::plan::testing::*;
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let config = target.path().join("launch.config");
        std::fs::write(&config, "EXPERT_BACKEND=local\nSPARK_COUNT=4\nSPECULATOR=off\n").unwrap();
        let mut input = args(target.path(), 4, false);
        input.spark_ranks = None;
        input.config = Some(config);
        let effective = file_args(input, target.path()).unwrap();
        assert_eq!(effective.spark_ranks, Some(0));
        let role = cuteafd_loader::plan::files::ReadRole::parse("coordinator",
            ExpertPlacement::from_spark_ranks(effective.spark_ranks.unwrap()), false).unwrap();
        let manifest = cuteafd_loader::plan::files::manifest(target.path(), role).unwrap();
        assert!(manifest.files.iter().any(|f| f.path.ends_with(".safetensors")));

        let glm = tempfile::tempdir().unwrap();
        write_snapshot(glm.path(), &glm5_config(), &[t("unused.weight", "BF16", &[2])], None);
        let layout = glm.path().join("layout.json");
        std::fs::write(&layout, r#"{"spark0":["spark0"],"head":["coordinator"]}"#).unwrap();
        let mut input = args(glm.path(), 4, false);
        input.file_layout = Some(layout);
        input.host = Some("spark0".into());
        let empty_cache = tempfile::tempdir().unwrap();
        input.hf_home = Some(empty_cache.path().into());
        assert!(file_args(input, glm.path()).unwrap().drafter_snapshot.is_none());
    }

    #[test]
    fn no_speculator_rejects_drafter_in_a_file_layout() {
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let layout = target.path().join("layout.json");
        std::fs::write(&layout, r#"{"worker":["drafter"]}"#).unwrap();
        let mut input = args(target.path(), 4, false);
        input.files = true;
        input.no_speculator = true;
        input.file_layout = Some(layout);
        assert!(run_plan(input).unwrap_err().to_string().contains("conflicts with drafter role"));
    }

    #[test]
    fn bundled_inventory_contains_every_drafter_input_and_off_excludes_it() {
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let draft = target.path().join("dflash");
        std::fs::create_dir(&draft).unwrap();
        for name in ["config.json", "mask_embedding.pt", "model.safetensors.index.json", "dflash_draft_model.safetensors", "dflash.py"] {
            std::fs::write(draft.join(name), b"fixture").unwrap();
        }
        let mut official = args(target.path(), 4, false);
        official.checkpoint_model_id = Some("XiaomiMiMo/MiMo-V2.6-Flash-MOPD".into());
        let automatic = file_args(official, target.path()).unwrap();
        let roles = [cuteafd_loader::plan::files::ReadRole::Coordinator { local_experts: false, speculator: false }];
        let mut manifest = cuteafd_loader::plan::files::manifest_roles(target.path(), &roles).unwrap();
        attach_snapshots(&mut manifest, &roles, &automatic).unwrap();
        for entry in std::fs::read_dir(&draft).unwrap() {
            let relative = format!("dflash/{}", entry.unwrap().file_name().to_str().unwrap());
            assert!(manifest.files.iter().any(|f| f.path == relative), "loader input missing: {relative}");
        }
        let mut input = args(target.path(), 4, false);
        input.no_speculator = true;
        let off = file_args(input, target.path()).unwrap();
        let mut manifest = cuteafd_loader::plan::files::manifest_roles(target.path(), &roles).unwrap();
        attach_snapshots(&mut manifest, &roles, &off).unwrap();
        assert!(!manifest.files.iter().any(|f| f.path.starts_with("dflash/")));
        let external = cuteafd_loader::plan::files::ReadRole::parse("drafter", ExpertPlacement::Local, false).unwrap();
        assert_eq!(external, cuteafd_loader::plan::files::ReadRole::ExternalDrafter);
        let inventory = cuteafd_loader::plan::Checkpoint::inventory(target.path()).unwrap();
        assert!(cuteafd_loader::plan::files::required_tensors(&inventory, external).unwrap().is_empty());
        let mut manifest = cuteafd_loader::plan::files::manifest_roles(target.path(), &[external]).unwrap();
        attach_snapshots(&mut manifest, &[external], &automatic).unwrap();
        assert!(manifest.files.iter().any(|f| f.path.starts_with("dflash/")));
    }

    #[test]
    fn peer_source_paths_are_absolute_and_hosts_cannot_supply_options() {
        assert_eq!(source_location(Some("worker:/models/snapshot"), std::path::Path::new("/unused")).unwrap(),
            (Some("worker".into()), PathBuf::from("/models/snapshot")));
        assert!(source_location(Some("-oProxyCommand=bad:/snapshot"), std::path::Path::new("/unused")).is_err());
        assert!(source_location(Some("worker:relative"), std::path::Path::new("/unused")).is_err());
        assert!(source_location(Some("worker:/line\nbreak"), std::path::Path::new("/unused")).is_err());
    }

    #[test]
    fn batched_size_results_preserve_order_and_reject_partial_or_invalid_output() {
        assert_eq!(parse_remote_sizes("peer", "17\n-\n0\n", 3).unwrap(), vec![Some(17), None, Some(0)]);
        assert!(parse_remote_sizes("peer", "17\n", 2).unwrap_err().to_string().contains("peer"));
        assert!(parse_remote_sizes("peer", "banner\n17\n", 2).is_err());
    }

    #[test]
    fn files_cli_and_local_transfer_skip_matching_sizes_without_deleting() {
        use clap::Parser;
        let source = tempfile::tempdir().unwrap();
        let destination = tempfile::tempdir().unwrap();
        std::fs::write(source.path().join("config.json"), b"config").unwrap();
        std::fs::write(destination.path().join("keep.txt"), b"keep").unwrap();
        let manifest = cuteafd_loader::plan::files::FileManifest {
            role: "coordinator".into(), snapshot: source.path().display().to_string(),
            repo_id: None, revision: None, files: vec![cuteafd_loader::plan::files::RequiredFile {
                path: "config.json".into(), bytes: Some(6) }], total_bytes: Some(6), additional_snapshots: Vec::new() };
        let mut args = args(source.path(), 4, false);
        args.fetch = true;
        args.destination = Some(destination.path().into());
        if std::process::Command::new("rsync").arg("--version").output().is_ok() {
            transfer(source.path(), &manifest, &args).unwrap();
            assert_eq!(std::fs::read(destination.path().join("config.json")).unwrap(), b"config");
            std::fs::write(destination.path().join("config.json"), b"custom").unwrap();
            transfer(source.path(), &manifest, &args).unwrap();
            assert_eq!(std::fs::read(destination.path().join("config.json")).unwrap(), b"custom");
            args.force = true;
            transfer(source.path(), &manifest, &args).unwrap();
            assert_eq!(std::fs::read(destination.path().join("config.json")).unwrap(), b"config");
            assert_eq!(std::fs::read(destination.path().join("keep.txt")).unwrap(), b"keep");
        }
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--files",
            "--role", "spark0", "--spark-ranks", "4"]).unwrap();
        let crate::cli::Commands::Plan(args) = cli.command else { unreachable!() };
        assert!(args.files && args.role.as_deref() == Some("spark0"));
    }

    #[test]
    fn planner_and_serving_share_draft_head_capacity_and_charge() {
        use clap::Parser;
        use cuteafd_loader::families::{glm5::draft_representation as glm, mimo_v2::draft_representation as mimo};
        use cuteafd_loader::formats::trellis_core::ProjectionScratchLayout;
        let config = serde_json::json!({"hidden_size":4096, "intermediate_size":12288,
            "num_hidden_layers":5, "num_attention_heads":32, "num_key_value_heads":8,
            "head_dim":128, "vocab_size":154880,
            "dflash_config":{"target_layer_ids":[0,8,16,24,32], "conv_group_size":16,
                "selector_rank":256, "block_size":8}});
        for command in ["serve-glmf", "serve-glm", "serve-mimo"] {
            for concurrency in [0usize, 1, 8, 16, 32, 64] {
                for extra in [vec![], vec!["--draft-sequences", "8"],
                    vec!["--draft-sequences", "8", "--draft-context-slots", "40"]] {
                    let count = concurrency.to_string();
                    let family = match command { "serve-glmf" => "glm5_flash", "serve-glm" => "glm5", _ => "mimo_v2" };
                    let plan = crate::cli::Cli::try_parse_from(
                        ["cuteafd", "plan", "/not-read", "--layout", "--concurrency", &count]
                            .into_iter().chain(extra.iter().copied())).unwrap();
                    let crate::cli::Commands::Plan(args) = plan.command else { unreachable!() };
                    let layout = options(&args).unwrap().layout.unwrap();
                    let resolved = layout.resolved_concurrency(family) as usize;
                    let mut argv = vec!["cuteafd", command, "--snapshot", "/not-read", "--native-lib", "/not-loaded"];
                    if concurrency > 0 { argv.extend(["--max-sequences", &count]); }
                    let serving = crate::cli::Cli::try_parse_from(argv.into_iter().chain(extra.iter().copied())).unwrap();
                    let (planned, started) = match serving.command {
                        crate::cli::Commands::ServeGlmf(mut args) => {
                            let (slots, batch) = glm::flash_serving_draft_limits(resolved,
                                layout.glm_draft_sequences() as usize, layout.draft_context_slots.map(|n| n as usize));
                            let planned = glm::flash_draft_capacity(&config, slots, batch).unwrap();
                            let concurrency = crate::shared::memory::budget::resolve_serving_concurrency(
                                args.max_sequences, || Ok(layout.rtx_bytes[0])).unwrap();
                            assert_eq!(resolved, concurrency);
                            args.prefix.resolve_entries(concurrency);
                            assert_eq!(layout.resolved_state_slots(family),
                                cuteafd_core::concurrency::state_slots(concurrency as u64));
                            let marks = args.prefix.mark_rule(concurrency);
                            assert_eq!(marks.entries as u64, layout.prefix_entries(family));
                            for mark_bytes in [1024, 1 << 20, 32 << 20] {
                                assert_eq!(marks.slots(mark_bytes) as u64,
                                    cuteafd_core::prefix::mark_slots_with_host(resolved as u64,
                                        layout.prefix_entries(family), mark_bytes as u64,
                                        marks.budget_bytes, marks.host));
                            }
                            let (slots, batch) = crate::families::glm5_flash::serve::draft_capacity(
                                concurrency, args.engine.draft_sequences, args.engine.draft_context_slots);
                            let started = glm::flash_draft_capacity(&config, slots, batch).unwrap();
                            assert_eq!(planned, started);
                            (planned.block_rows, started.block_rows)
                        }
                        crate::cli::Commands::ServeGlm(args) => {
                            let concurrency = crate::shared::memory::budget::resolve_serving_concurrency(
                                args.max_sequences, || Ok(layout.rtx_bytes[0])).unwrap();
                            assert_eq!(resolved, concurrency);
                            let batch = layout.glm_draft_sequences() as usize;
                            let slots = glm::serving_draft_context_slots(batch, resolved,
                                layout.draft_context_slots.map(|n| n as usize));
                            let startup_slots = glm::serving_draft_context_slots(args.engine.draft_sequences,
                                concurrency, args.engine.draft_context_slots);
                            let planned = glm::GlmDraftCapacity::new(slots, batch, 8).unwrap();
                            let started = glm::GlmDraftCapacity::new(startup_slots, args.engine.draft_sequences, 8).unwrap();
                            assert_eq!(planned, started);
                            (planned.block_rows, started.block_rows)
                        }
                        crate::cli::Commands::ServeMimo(args) => {
                            let concurrency = crate::shared::memory::budget::resolve_serving_concurrency(
                                args.max_sequences, || Ok(layout.rtx_bytes[0])).unwrap();
                            assert_eq!(resolved, concurrency);
                            let batch = layout.mimo_draft_sequences().max(resolved as u64);
                            let slots = mimo::mimo_draft_context_slots(batch, layout.mimo_rings(),
                                layout.draft_context_slots);
                            let startup_batch = args.engine.draft_sequences.max(concurrency);
                            let startup_slots = mimo::mimo_draft_context_slots(startup_batch as u64,
                                cuteafd_core::concurrency::request_slots(concurrency as u64, args.engine.rings as u64),
                                args.engine.draft_context_slots.map(|n| n as u64));
                            let planned = mimo::MimoDraftCapacity::new(slots as usize, batch as usize, 8).unwrap();
                            let started = mimo::MimoDraftCapacity::new(startup_slots as usize, startup_batch, 8).unwrap();
                            assert_eq!(planned, started);
                            (planned.block_rows, started.block_rows)
                        }
                        _ => unreachable!(),
                    };
                    assert_eq!(planned, started, "{command} {concurrency} {extra:?}");
                    let charge = |rows| if command == "serve-mimo" {
                        // MiMo borrows a BF16/FP8 head, not K6: F32 logits plus fixed GEMM workspace.
                        cuteafd_loader::families::mimo_v2::admission::allocation_bytes("draft.workspace.logits",
                            &[rows, 154880, 4]).unwrap() as usize + (4 << 20)
                    } else { ProjectionScratchLayout::new(4096, 154880, rows).unwrap().bytes().unwrap() };
                    assert_eq!(charge(planned), charge(started), "{command} draft head charge");
                    if command == "serve-glmf" && concurrency == 16 && extra.is_empty() {
                        assert_eq!(planned, 128);
                        assert_eq!(charge(planned), 397_541_448);
                    }
                }
            }
        }
    }

    #[test]
    fn planner_threads_warm_draft_context_override() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout",
            "--mimo-prefix-draft", "--rings", "20", "--draft-sequences", "8",
            "--draft-context-slots", "25", "--attention-plan-rows", "8"]).unwrap();
        let crate::cli::Commands::Plan(args) = cli.command else { unreachable!() };
        let layout = options(&args).unwrap().layout.unwrap();
        assert!(layout.mimo_prefix_draft);
        assert_eq!(layout.mimo_rings, 20);
        assert_eq!(layout.draft_sequences, Some(8));
        assert_eq!(layout.draft_context_slots, Some(25));
        assert_eq!(layout.glm5_attention_plan_rows, Some(8));
        assert_eq!(layout.qwen_attention_plan_rows, Some(8));
    }

    #[test]
    fn ingested_config_keys_match_explicit_family_planner_argv() {
        use clap::Parser;
        let families = ["deepseek_v41", "deepseek_v4", "glm5", "glm5_flash", "kimi_k3", "mimo_v2",
            "qwen4", "qwen3_5", "step5", "muse_spark", "minimax_m3", "mistral_large4"];
        let common = [("RTX_GPUS", "2", "--rtx", "2"),
            ("ATTENTION_PLACEMENT", "heads", "--attention-placement", "heads"),
            ("CONCURRENCY", "4", "--concurrency", "4"),
            ("PREFILL_WAVES", "2", "--prefill-waves", "2"),
            ("FP8_PREFILL", "off", "--fp8-prefill", "off"),
            ("GPU_MEMORY_CAP", "32GiB", "--gpu-memory-cap", "32GiB"),
            ("SPARK_MEMORY_CAP", "110GiB", "--spark-memory-cap", "110GiB"),
            ("POOL_TOKENS", "65536", "--pool-tokens", "65536"),
            ("EXLR8_MANIFEST", "k4", "--exlr8-manifest", "k4"),
            ("EXLR8_PREFILL_ACTIVATIONS", "a8", "--exlr8-prefill-activations", "a8"),
            ("CHAT_TEMPLATE_FROM", "vendor/template", "--chat-template-from", "vendor/template"),
            ("CORE_VIEW", "fp8-compact", "--core-view", "fp8-compact"),
            ("VISION", "rtx", "--vision", "rtx"), ("AUDIO", "off", "--audio", "off"),
            ("HOST_CACHE_BYTES", "2GiB", "--host-cache-bytes", "2GiB"),
            ("RTX_EXPERT_LAYERS", "0", "--rtx-expert-layers", "0"),
            ("MAX_CONTEXT_TOKENS", "65536", "--max-context-tokens", "65536"),
            ("EXPERT_USAGE", "off", "--expert-usage", "false")];
        for family in families {
            // Unsupported family knobs must not leak into planner charging.
            let mut ignored = vec![];
            if matches!(family, "glm5" | "glm5_flash" | "deepseek_v4" | "deepseek_v41" | "minimax_m3" | "mistral_large4") {
                ignored.extend([("KV_CACHE", "int8"), ("SPECULATOR_DEPTH", "not-a-depth")]);
            }
            if !matches!(family, "glm5" | "mistral_large4") { ignored.push(("PREFILL_SPAN_ROWS", "not-rows")); }
            for (key, value) in ignored {
                let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/unused", "--layout"]).unwrap();
                let crate::cli::Commands::Plan(mut baseline) = cli.command else { panic!("plan") };
                resolve_family_config(&mut baseline, family, &Default::default()).unwrap();
                let mut configured = baseline.clone();
                let cfg = [(key.into(), value.into())].into_iter().collect();
                resolve_config(&mut configured, &cfg).unwrap();
                resolve_family_config(&mut configured, family, &cfg).unwrap();
                assert_eq!(format!("{:?}", options(&baseline).unwrap()), format!("{:?}", options(&configured).unwrap()), "{family} ignores {key}");
            }
            let mut cases = common.to_vec();
            if matches!(family, "step5" | "deepseek_v41") { cases.push(("PREFILL_BATCH_TOKENS", "1024", "--prefill-rows", "1024")); }
            else if family != "qwen3_5" { cases.push(("PREFILL_ROWS", "1024", "--prefill-rows", "1024")); }
            if matches!(family, "glm5" | "mistral_large4") {
                cases.extend([("PREFILL_SPAN_ROWS", "4096", "--prefill-span-rows", "4096"),
                    ("DECODE_GRAPHS", "off", "--decode-graphs", "off"),
                    ("MEDIA_CACHE_BYTES", "256MiB", "--media-cache-bytes", "256MiB")]);
                let mut request = args(std::path::Path::new("/model"), 4, false);
                let cfg = [("PREFILL_BATCH_TOKENS".into(), "1024".into())].into_iter().collect();
                let error = resolve_family_config(&mut request, family, &cfg).unwrap_err();
                assert!(error.to_string().contains("PREFILL_BATCH_TOKENS"));
            }
            if family == "mimo_v2" { for value in ["int8", "bf16"] { cases.push(("KV_CACHE", value, "--mimo-kv-cache", value)); } }
            else if matches!(family, "qwen4" | "qwen3_5" | "kimi_k3" | "step5" | "muse_spark") {
                cases.push(("KV_CACHE", "bf16", "--kv-cache", "bf16"));
            }
            if matches!(family, "qwen4" | "step5") { cases.push(("SPECULATOR_DEPTH", "2", "--native-mtp-layers", "2")); }
            for (key, configured, flag, value) in cases {
                let parse = |extra: &[&str]| {
                    let mut argv = vec!["cuteafd", "plan", "/unused", "--layout"];
                    if flag == "--expert-usage" && !extra.is_empty() { argv.push("--expert-usage=false"); }
                    else { argv.extend_from_slice(extra); }
                    let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
                    let crate::cli::Commands::Plan(mut args) = cli.command else { panic!("plan") };
                    args.memory = cli.memory; args.vision = cli.vision.unwrap_or(args.vision); args.audio = cli.audio.unwrap_or(args.audio);
                    args
                };
                let cfg = [(key.into(), configured.into())].into_iter().collect();
                let mut from_config = parse(&[]);
                resolve_config(&mut from_config, &cfg).unwrap();
                resolve_family_config(&mut from_config, family, &cfg).unwrap();
                resolve_context_tokens(&mut from_config, &cfg).unwrap();
                resolve_expert_usage(&mut from_config, &cfg).unwrap();
                let mut explicit = parse(&[flag, value]);
                resolve_family_config(&mut explicit, family, &Default::default()).unwrap();
                // The dense executor always uses its fixed 4096-row serving unit.
                if family == "qwen3_5" { explicit.prefill_rows = 4096; }
                assert_eq!(format!("{:?}", options(&from_config).unwrap()), format!("{:?}", options(&explicit).unwrap()), "{family} {key}");
                assert_eq!(from_config.attention.core_view, explicit.attention.core_view, "{family} {key}");
                assert_eq!(from_config.exlr8_manifest, explicit.exlr8_manifest, "{family} {key}");
                assert_eq!(from_config.chat_template_from, explicit.chat_template_from, "{family} {key}");
            }
        }
    }

    #[test]
    fn v41_planner_audio_defaults_off_and_refuses_explicit_non_off_modes() {
        use cuteafd_loader::plan::MediaMode;
        let parse = |args: &[&str]| {
            use clap::{CommandFactory, FromArgMatches};
            let matches = crate::cli::Cli::command().try_get_matches_from(
                ["cuteafd", "plan", "/unused"].into_iter().chain(args.iter().copied())).unwrap();
            let cli = crate::cli::Cli::from_arg_matches(&matches).unwrap();
            let crate::cli::Commands::Plan(mut plan) = cli.command else { unreachable!() };
            if let Some(mode) = cli.audio {
                plan.audio = mode;
                plan.explicit_settings.insert("audio".into());
            }
            plan
        };
        let mut plan = parse(&[]);
        resolve_v41_audio(&mut plan, &Default::default()).unwrap();
        assert_eq!(plan.audio, MediaMode::Off);
        for value in ["auto", "rtx", "spark:0"] {
            let cfg = [("AUDIO".into(), value.into())].into_iter().collect();
            let mut configured = parse(&[]);
            resolve_config(&mut configured, &cfg).unwrap();
            assert!(resolve_v41_audio(&mut configured, &cfg).unwrap_err().to_string().contains("AUDIO"));
            assert!(resolve_v41_audio(&mut parse(&["--audio", value]), &Default::default()).is_err());
            let mut explicit_off = parse(&["--audio", "off"]);
            resolve_config(&mut explicit_off, &cfg).unwrap();
            resolve_v41_audio(&mut explicit_off, &cfg).unwrap();
            assert_eq!(explicit_off.audio, MediaMode::Off);
        }
    }

    #[test]
    fn media_modes_parse_on_every_command_and_invalid_modes_fail() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--vision", "off", "--audio", "auto"]).unwrap();
        assert_eq!(cli.vision, Some(cuteafd_loader::plan::MediaMode::Off));
        assert_eq!(cli.audio, Some(cuteafd_loader::plan::MediaMode::Auto));
        assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--vision", "bad"]).is_err());
    }

    #[test]
    fn resolved_checkpoint_ownership_matches_config_and_legacy_choices() {
        use cuteafd_loader::plan::spec::DraftOwnership;
        let dir = tempfile::tempdir().unwrap();
        for (family, key, value, expected) in [
            ("glm5_flash", "SPECULATOR", "dspark", DraftOwnership::None),
            ("glm5", "SPECULATOR", "off", DraftOwnership::None),
            ("mimo_v2", "SPECULATOR", "dflash2", DraftOwnership::None),
            ("mimo_v2", "SPECULATOR", "mtp", DraftOwnership::NativeMtp),
            ("qwen4", "SPECULATOR", "off", DraftOwnership::None),
            ("hy4_flash", "SPECULATOR", "dfly2", DraftOwnership::BundledDfly2),
            ("deepseek_v4", "DSPARK", "off", DraftOwnership::None),
            ("deepseek_v41", "DSPARK", "on", DraftOwnership::BundledDspark),
        ] {
            let mut request = args(dir.path(), 4, false);
            let cfg = std::collections::BTreeMap::from([(key.into(), value.into())]);
            resolve_speculator(&mut request, dir.path(), family, &cfg).unwrap();
            assert_eq!(DraftOwnership::selected(family, &request.speculator), expected);
            request.speculator = "off".into();
            resolve_speculator(&mut request, dir.path(), family, &cfg).unwrap();
            assert_eq!(DraftOwnership::selected(family, &request.speculator), DraftOwnership::None);
        }
        std::fs::create_dir(dir.path().join("dflash")).unwrap();
        let mut request = args(dir.path(), 4, false);
        request.checkpoint_model_id = Some("XiaomiMiMo/MiMo-V2.6-Flash-MOPD".into());
        resolve_speculator(&mut request, dir.path(), "mimo_v2", &Default::default()).unwrap();
        assert_eq!(request.speculator, "dflash2");
        let mut request = args(dir.path(), 4, false);
        request.native_mtp_layers = 0;
        resolve_speculator(&mut request, dir.path(), "deepseek_v4", &Default::default()).unwrap();
        assert_eq!(request.speculator, "off");
    }

    #[test]
    fn mimo_draft_precision_and_plan_accounting_share_one_resolution() {
        let dir = tempfile::tempdir().unwrap();
        for alias in ["XiaomiMiMo/MiMo-V2.6-Flash-MOPD", "community/MiMo-MC", "renamed/local"] {
            for (mode, expected) in [(None, true), (Some("on"), true), (Some("auto"), true), (Some("off"), false)] {
                let mut request = args(dir.path(), 4, false);
                request.checkpoint_model_id = Some(alias.into());
                request.speculator = "dflash2".into();
                let cfg = mode.map(|mode| [("SPECULATOR_FP8".into(), mode.into())].into()).unwrap_or_default();
                let resolved = resolve_settings(&mut request, "mimo_v2", &cfg).unwrap();
                assert_eq!(resolved["draft_fp8"].value, serde_json::json!(expected));
                assert_eq!(request.draft_fp8, Some(expected));
                assert_eq!(options(&request).unwrap().layout.unwrap().mimo_draft_bf16, !expected);
                request.draft_fp8 = Some(!expected);
                resolve_settings(&mut request, "mimo_v2", &cfg).unwrap();
                assert_eq!(options(&request).unwrap().layout.unwrap().mimo_draft_bf16, expected);
            }
        }
    }

    #[test]
    fn mimo_default_drafter_tracks_checkpoint_contents_not_alias_or_core_view() {
        let dir = tempfile::tempdir().unwrap();
        for view in [cuteafd_loader::plan::checkpoint::CoreView::Original,
            cuteafd_loader::plan::checkpoint::CoreView::K8Compact,
            cuteafd_loader::plan::checkpoint::CoreView::K6Compact] {
            let mut request = args(dir.path(), 6, false);
            request.attention.core_view = view;
            request.checkpoint_model_id = Some("community/MiMo-MC".into());
            assert!(!mimo_drafter_available(&request, dir.path(), &Default::default()));
            std::fs::create_dir_all(dir.path().join("dflash")).unwrap();
            resolve_speculator(&mut request, dir.path(), "mimo_v2", &Default::default()).unwrap();
            assert_eq!(request.speculator, "dflash2");
            request.speculator = "off".into();
            resolve_speculator(&mut request, dir.path(), "mimo_v2", &Default::default()).unwrap();
            assert_eq!(request.speculator, "off");
            std::fs::remove_dir(dir.path().join("dflash")).unwrap();
        }
        let request = args(dir.path(), 4, false);
        let configured = [("SPECULATOR_MODEL_ID".into(), "community/pairable-drafter".into())].into_iter().collect();
        assert!(mimo_drafter_available(&request, dir.path(), &configured));
    }

    #[test]
    fn file_planning_preserves_auto_vision_for_bundled_towers_and_explicit_off() {
        use cuteafd_loader::plan::{testing::*, MediaMode};
        for mut config in [serde_json::json!({"model_type":"inkling2"}),
            serde_json::json!({"model_type":"minimax_m3_vl", "vision_config":{}}),
            serde_json::json!({"model_type":"deepseek_v4", "architectures":["DeepseekV4ForCausalLM"]})] {
            let dir = tempfile::tempdir().unwrap();
            config["architectures"] = match config["model_type"].as_str().unwrap() {
                "inkling2" => serde_json::json!(["Inkling2ForConditionalGeneration"]),
                "minimax_m3_vl" => serde_json::json!(["MiniMaxM3SparseForConditionalGeneration"]),
                _ => config["architectures"].clone(),
            };
            write_snapshot(dir.path(), &config, &[t("unused.weight", "BF16", &[2])], None);
            let mut request = args(dir.path(), 4, false);
            request.speculator = "off".into();
            request.vision = MediaMode::Auto;
            assert_eq!(file_args(request.clone(), dir.path()).unwrap().vision, MediaMode::Auto);
            let config_path = dir.path().join("launch.config");
            std::fs::write(&config_path, "VISION=off\nSPECULATOR=off\n").unwrap();
            request.config = Some(config_path);
            assert_eq!(file_args(request, dir.path()).unwrap().vision, MediaMode::Off);
        }
    }

    #[test]
    fn step5_mtp_defaults_on_and_preserves_admitted_depth() {
        use cuteafd_loader::plan::spec::DraftOwnership;
        let dir = tempfile::tempdir().unwrap();
        let mut automatic = args(dir.path(), 6, false);
        automatic.layout = true;
        resolve_speculator(&mut automatic, dir.path(), "step5", &Default::default()).unwrap();
        assert_eq!(automatic.speculator, "mtp");
        assert_eq!(automatic.native_mtp_layers, 3);
        assert_eq!(options(&automatic).unwrap().layout.unwrap().native_mtp_layers, 3);
        for disabled in ["flag", "config", "depth"] {
            let mut request = args(dir.path(), 6, false);
            let cfg = if disabled == "config" {
                std::collections::BTreeMap::from([("SPECULATOR".into(), "off".into())])
            } else { Default::default() };
            if disabled == "flag" { request.no_speculator = true; }
            if disabled == "depth" { request.native_mtp_layers = 0; }
            resolve_speculator(&mut request, dir.path(), "step5", &cfg).unwrap();
            assert_eq!(request.speculator, "off", "{disabled}");
            assert_eq!(DraftOwnership::selected("step5", &request.speculator), DraftOwnership::None);
        }
        for depth in 1..=3 {
            for configured in [false, true] {
                let mut request = args(dir.path(), 6, false);
                request.layout = true;
                request.rtx = 2;
                request.native_mtp_layers = depth;
                let cfg = if configured {
                    std::collections::BTreeMap::from([("SPECULATOR".into(), "mtp".into())])
                } else {
                    request.speculator = "mtp".into();
                    Default::default()
                };
                resolve_speculator(&mut request, dir.path(), "step5", &cfg).unwrap();
                assert_eq!(request.speculator, "mtp");
                assert_eq!(DraftOwnership::selected("step5", &request.speculator), DraftOwnership::NativeMtp);
                assert_eq!(options(&request).unwrap().layout.unwrap().native_mtp_layers, depth);
            }
        }
    }

    #[test]
    fn planner_and_glm_flash_share_auto_default_and_explicit_off() {
        use clap::Parser;
        use cuteafd_loader::plan::MediaMode;
        for command in ["plan", "serve-glmf"] {
            for mode in [None, Some("off")] {
                let mut argv = vec!["cuteafd", command, "--snapshot", "/not-read", "--native-lib", "/not-loaded"];
                if command == "plan" { argv = vec!["cuteafd", command, "/not-read"]; }
                if let Some(mode) = mode { argv.extend(["--vision", mode]); }
                let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
                let expected = if mode.is_some() { MediaMode::Off } else { MediaMode::Auto };
                assert_eq!(crate::resolve_vision(cli.vision, None), expected);
                match cli.command {
                    crate::cli::Commands::Plan(args) => assert_eq!(args.vision, MediaMode::Auto),
                    crate::cli::Commands::ServeGlmf(args) => assert_eq!(args.vision, MediaMode::Auto),
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn spark_range_cli_is_header_only_and_excludes_other_output_modes() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--spark-range", "--json"]).unwrap();
        let crate::cli::Commands::Plan(args) = cli.command else { panic!("plan") };
        assert!(args.spark_range && args.json);
        for mode in ["--deployment", "--files", "--fetch"] {
            assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--spark-range", mode]).is_err());
        }
    }

    #[test]
    fn replicas_require_an_explicit_layout_inventory() {
        let mut request = args(std::path::Path::new("/not-read"), 4, false);
        request.layout = false;
        request.vision_replicas = 2;
        assert!(matches!(options(&request), Err(PlanError::InvalidOption { option: "--vision-replicas", .. })));
        request.layout = true;
        assert_eq!(options(&request).unwrap().layout.unwrap().vision_replicas, 2);
    }

    #[test]
    fn gpu_memory_cap_bounds_every_layout_gpu_and_rejects_percentages() {
        use clap::Parser;
        for (size, bytes) in [("32GiB", 32u64 << 30), ("34359738368", 32 << 30), ("32768MiB", 32 << 30), ("40GB", 40_000_000_000)] {
            let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout", "--rtx", "2",
                "--gpu-memory-cap", size]).unwrap();
            let crate::cli::Commands::Plan(mut args) = cli.command else { panic!("plan") };
            args.memory = cli.memory;
            let options = options(&args).unwrap();
            assert_eq!(options.coordinator_gpu_budget_bytes, bytes);
            assert_eq!(options.layout.unwrap().rtx_bytes, vec![bytes; 2]);
            assert_eq!(options.coordinator_budget_bytes, bytes);
        }
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read"]).unwrap();
        let crate::cli::Commands::Plan(plan_args) = cli.command else { panic!("plan") };
        assert!(options(&plan_args).unwrap().layout.is_none());
        for invalid in ["50%", "-1GiB", "NaN", "32TB", "GiB"] {
            assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--gpu-memory-cap", invalid]).is_err(),
                "{invalid}");
        }
        for removed in ["--coordinator-gpu-budget-gib", "--rtx-budget-gib", "--rtx-gib"] {
            assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", removed, "32"]).is_err(), "{removed}");
        }
        let mut invalid = PlanOptions::default();
        invalid.coordinator_gpu_budget_bytes = 0;
        assert!(matches!(invalid.validate(), Err(PlanError::InvalidOption { option: "coordinator GPU budget", .. })));
    }

    #[test]
    fn simulated_5090_layout_uses_the_explicit_physical_sm_count() {
        use clap::Parser;
        for (args, sms) in [(&[][..], 170), (&["--physical-sms", "170"][..], 170),
                            (&["--physical-sms", "188"][..], 188)] {
            let cli = crate::cli::Cli::try_parse_from([
                "cuteafd", "plan", "/not-read", "--layout", "--rtx", "2",
                "--gpu-memory-cap", "32GiB",
            ].into_iter().chain(args.iter().copied())).unwrap();
            let crate::cli::Commands::Plan(mut plan) = cli.command else { panic!("plan") };
            plan.memory = cli.memory;
            let layout = options(&plan).unwrap().layout.unwrap();
            assert_eq!(layout.rtx_bytes, vec![32 << 30; 2]);
            // Kimi sizes the drafter on the head rank, a separate GPU in TP2.
            assert_eq!(layout.coordinator_sms(1), sms);
        }
    }

    #[test]
    fn v41_prefix_and_worker_layout_match_launch_arguments() {
        let mut args = args(std::path::Path::new("/not-read"), 4, false);
        let mut layout = cuteafd_loader::plan::layout::LayoutOptions { host_memory: Some(cuteafd_loader::plan::testing::SimulatedHost::raptor().host_ram), ..Default::default() };
        for entries in [0, 1, 20, 128] {
            args.prefix_cache_entries = Some(entries);
            for (rows, capacity) in [(0, 4096), (80, 256), (256, 256), (1024, 1024), (2048, 4096)] {
                args.prefill_rows = rows;
                v41_layout(&mut layout, &args);
                assert_eq!(layout.prefix_slots, None);
                assert_eq!(layout.spark_capacity_rows, capacity);
            }
        }
        args.prefix_slots = Some(8);
        v41_layout(&mut layout, &args);
        assert_eq!(layout.prefix_slots, Some(8));
    }

    #[test]
    fn explicit_spark_hosts_are_validated_before_planning() {
        let cfg = |hosts: &str| std::collections::BTreeMap::from([
            ("SPARK_HOSTS".into(), hosts.into()), ("SPARK_0_HOST".into(), "custom".into())]);
        assert_eq!(spark_hosts_count(&cfg("ostrich,custom")).unwrap(), Some(2));
        let mut eight = cfg("worker0,worker1,worker2,worker3,worker4,worker5,worker6,worker7");
        for rank in 0..8 { eight.insert(format!("SPARK_{rank}_HOST"), format!("worker{rank}")); }
        assert_eq!(spark_hosts_count(&eight).unwrap(), Some(8));
        for hosts in ["ostrich,ostrich", "ostrich,", ",ostrich", "ostrich, emu", "unknown"] {
            assert!(spark_hosts_count(&cfg(hosts)).is_err(), "{hosts}");
        }
    }

    #[test]
    fn gpu_memory_cap_is_global_on_either_side_of_the_subcommand() {
        use clap::Parser;
        let parse = |before: bool| {
            let argv = if before {
                vec!["cuteafd", "--gpu-memory-cap", "31.8GiB", "plan", "/not-read", "--layout"]
            } else {
                vec!["cuteafd", "plan", "/not-read", "--layout", "--gpu-memory-cap", "31.8GiB"]
            };
            let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
            let crate::cli::Commands::Plan(mut args) = cli.command else { panic!("plan") };
            args.memory = cli.memory;
            let mut selected = options(&args).unwrap();
            selected.layout.as_mut().unwrap().host_memory = Some(cuteafd_loader::plan::testing::SimulatedHost::raptor().host_ram);
            selected
        };
        let expected = parse(false);
        let snapshot = tempfile::tempdir().unwrap();
        write_snapshot(snapshot.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        let expected_report = serde_json::to_value(plan(snapshot.path(), &expected).unwrap()).unwrap();
        let actual = parse(true);
        assert_eq!(serde_json::to_value(plan(snapshot.path(), &actual).unwrap()).unwrap(), expected_report);
        assert_eq!(actual.layout.unwrap().rtx_bytes, expected.layout.as_ref().unwrap().rtx_bytes);
        // The same decimal size the serving cap would install.
        let serving = cuteafd_core::serving_capacity::GpuMemoryBudget::from_gib(31.8).unwrap();
        assert_eq!(expected.layout.unwrap().rtx_bytes, vec![serving.0]);
    }

    #[test]
    fn weight_budget_is_global_and_independent_of_total_memory() {
        use clap::Parser;
        for argv in [
            vec!["cuteafd", "--coordinator-weight-budget-gib", "10", "plan", "/not-read", "--layout"],
            vec!["cuteafd", "plan", "/not-read", "--layout", "--coordinator-weight-budget-gib", "10"],
        ] {
            let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
            let crate::cli::Commands::Plan(mut args) = cli.command else { panic!("plan") };
            args.coordinator_weight_budget_gib = cli.coordinator_weight_budget_gib;
            let options = options(&args).unwrap();
            assert_eq!(options.coordinator_budget_bytes, 10 << 30);
            assert_eq!(options.layout.unwrap().rtx_bytes, vec![cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES]);
        }
    }

    #[test]
    fn named_config_hosts_select_inventory_and_placeholders_are_simulated() {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("hosts.config");
        let mut request = args(std::path::Path::new("/not-read"), 2, false);
        request.memory.spark_memory_cap = None;
        request.config = Some(config.clone());
        request.deployment_peers = vec!["192.0.2.1:19441".into(), "192.0.2.2:19441".into()];
        std::fs::write(&config, "SPARK_0_HOST=rhea\nSPARK_1_HOST=ostrich\n").unwrap();
        let layout = options(&request).unwrap().layout.unwrap();
        assert_eq!(layout.spark_capacity(0), cuteafd_loader::serving_capacity::spark_memory::measured_bytes("rhea").unwrap());
        assert_eq!(layout.spark_capacity(1), cuteafd_loader::serving_capacity::spark_memory::measured_bytes("ostrich").unwrap());
        std::fs::write(&config, "SPARK_0_HOST=unknown\n").unwrap();
        request.spark_ranks = Some(0);
        assert!(options(&request).unwrap().layout.unwrap().spark_rank_bytes.is_empty());
        request.spark_ranks = Some(2);
        let error = options(&request).unwrap_err().to_string();
        assert!(error.contains("unknown") && error.contains("spark_memory.json"));
        std::fs::write(&config, "SPARK_0_HOST=REPLACE-ME-spark-0\n").unwrap();
        assert_eq!(options(&request).unwrap().layout.unwrap().spark_capacities(6),
            cuteafd_loader::serving_capacity::spark_memory::serving_bytes_for(2));
    }

    #[test]
    fn deployment_peers_select_measured_rank_capacities_and_explicit_wins() {
        let mut request = args(std::path::Path::new("/not-read"), 2, false);
        request.memory.spark_memory_cap = None;
        request.deployment_peers = vec!["10.55.0.5:19441".into(), "10.55.0.1:19441".into()];
        let layout = options(&request).unwrap().layout.unwrap();
        assert_eq!(layout.spark_capacities(2), vec![
            cuteafd_loader::serving_capacity::spark_memory::measured_bytes("rhea").unwrap(),
            cuteafd_loader::serving_capacity::spark_memory::measured_bytes("ostrich").unwrap()]);
        request.deployment_peers[0] = "unknown:19441".into();
        assert!(options(&request).is_err());
        request.memory.spark_memory_cap = Some(82 << 30);
        assert_eq!(options(&request).unwrap().layout.unwrap().spark_capacities(2), vec![82 << 30; 2]);
    }

    #[test]
    fn spark_encoder_admission_honors_runtime_budget() {
        let mut request = args(std::path::Path::new("/not-read"), 4, false);
        request.memory.spark_memory_cap = Some(82 << 30);
        let options = options(&request).unwrap();
        assert_eq!(options.spark_budget_bytes, Some(82 << 30));
        let layout = options.layout.unwrap();
        assert_eq!(layout.spark_allocation_budget_bytes, Some(82 << 30));
        assert_eq!(layout.spark_bytes, 82 << 30);
    }

    #[test]
    fn unset_spark_budget_preserves_separate_defaults() {
        let mut request = args(std::path::Path::new("/not-read"), 4, false);
        request.memory.spark_memory_cap = None;
        let options = options(&request).unwrap();
        assert_eq!(options.spark_budget_bytes, None);
        assert_eq!(options.maximum_spark_capacity(), cuteafd_loader::serving_capacity::spark_memory::serving_bytes_for(4).into_iter().max().unwrap());
        let layout = options.layout.unwrap();
        assert_eq!(layout.spark_allocation_budget_bytes, None);
        assert_eq!(layout.spark_capacities(6), cuteafd_loader::serving_capacity::spark_memory::serving_bytes_for(4));
    }

    #[test]
    fn glm5_draft_representation_reaches_the_layout() {
        use clap::Parser;
        for (extra, bf16) in [(&[][..], false), (&["--glm5-draft-bf16"][..], true)] {
            let cli = crate::cli::Cli::try_parse_from(
                ["cuteafd", "plan", "/not-read", "--layout"].into_iter().chain(extra.iter().copied())).unwrap();
            let crate::cli::Commands::Plan(args) = cli.command else { panic!("plan") };
            let layout = options(&args).unwrap().layout.unwrap();
            assert_eq!(layout.glm5_draft_bf16, bf16);
            assert!(!layout.mimo_draft_bf16, "GLM's representation must not affect MiMo");
        }
    }

    #[test]
    fn glm_flash_precision_reaches_the_layout_without_duplicate_heads() {
        use clap::Parser;
        use crate::families::glm5_flash::{fp8::KdaFp8, engine::KdaState};
        let parse = |extra: &[&str]| {
            let cli = crate::cli::Cli::try_parse_from(
                ["cuteafd", "plan", "/not-read", "--layout"].into_iter().chain(extra.iter().copied())).unwrap();
            let crate::cli::Commands::Plan(args) = cli.command else { panic!("plan") };
            args
        };
        use crate::families::glm5_flash::head::GlmfHeadFormat;
        let defaults = parse(&[]);
        assert_eq!((defaults.kda_fp8, defaults.kda_state, defaults.head_format),
            (KdaFp8::Off, KdaState::F32, GlmfHeadFormat::Auto));
        for state in ["f32", "bf16", "bf16-tile"] {
            for kda in ["off", "row128", "channel"] {
                for (head, format) in [("true", GlmfHeadFormat::Fp8), ("false", GlmfHeadFormat::Bf16),
                    ("packed", GlmfHeadFormat::Packed), ("fp8", GlmfHeadFormat::Fp8), ("bf16", GlmfHeadFormat::Bf16)] {
                    let args = parse(&["--kda-state", state, "--kda-fp8", kda, "--fp8-head", head]);
                    let layout = options(&args).unwrap().layout.unwrap();
                    assert_eq!(layout.glmf_kda_state.bytes(), if state == "f32" { 4 } else { 2 });
                    assert_eq!(layout.glmf_representation.kda_fp8, kda != "off");
                    assert_eq!(layout.glmf_representation.head, format);
                }
            }
        }
        assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout", "--fp8-head", "maybe"]).is_err());
        let tensor = parse(&["--target-head", "tensor"]);
        assert!(validate_glmf_precision(&tensor, "glm5_flash").is_err());
        assert!(validate_glmf_precision(&tensor, "glm5").is_ok());
        // The packed head runs its own exact GEMV: an explicit tensor route is refused, exact is inert.
        let packed_tensor = parse(&["--fp8-head", "packed", "--target-head", "tensor"]);
        let error = validate_glmf_precision(&packed_tensor, "glm5_flash").unwrap_err().to_string();
        assert!(error.contains("exact 12-bit"), "{error}");
        assert!(validate_glmf_precision(&parse(&["--fp8-head", "packed", "--target-head", "exact"]), "glm5_flash").is_ok());
        for target in ["exact", "tensor"] {
            for draft in ["exact", "tensor"] {
                let layout = options(&parse(&["--fp8-head", "false", "--target-head", target,
                    "--draft-head", draft])).unwrap().layout.unwrap();
                assert_eq!(layout.glmf_representation.head, GlmfHeadFormat::Bf16);
                assert_eq!(layout.drafter_bytes, 0, "head arithmetic must not add a weight copy");
            }
        }
    }

    #[test]
    fn dense_dspark_pins_qualified_revision_and_preserves_overrides() {
        use cuteafd_loader::plan::testing::*;
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &serde_json::json!({"model_type":"qwen3_5", "text_config": {
            "model_type":"qwen3_5_text", "hidden_size":5120}}), &[t("unused.weight", "BF16", &[2])], None);
        let home = tempfile::tempdir().unwrap();
        let repo = home.path().join("hub/models--RadixArk--Qwen3.8-27B-DSpark");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), "unavailable-main").unwrap();
        let config = target.path().join("launch.config");
        for selected in [None, Some("explicit-revision")] {
            let revision = selected.unwrap_or("85ef153be924f17ce4bf62726954eeaa4a73e854");
            let draft = repo.join("snapshots").join(revision);
            std::fs::create_dir_all(&draft).unwrap();
            std::fs::write(&config, selected.map_or_else(String::new,
                |revision| format!("SPECULATOR_MODEL_REVISION={revision}\n"))).unwrap();
            let mut input = args(target.path(), 0, false);
            input.hf_home = Some(home.path().into());
            input.config = Some(config.clone());
            let resolved = file_args(input, target.path()).unwrap();
            assert_eq!(resolved.speculator, "dspark");
            assert_eq!(resolved.drafter_snapshot, Some(draft));
        }
    }

    #[test]
    fn dense_kv_defaults_to_fp8_and_preserves_explicit_formats() {
        use clap::Parser;
        use cuteafd_loader::families::qwen4::Qwen4KvCache;
        for (extra, expected) in [(&[][..], Qwen4KvCache::Fp8),
            (&["--kv-cache", "bf16"][..], Qwen4KvCache::Bf16),
            (&["--kv-cache", "fp8"][..], Qwen4KvCache::Fp8)] {
            let cli = crate::cli::Cli::try_parse_from(
                ["cuteafd", "plan", "/not-read", "--layout"].into_iter().chain(extra.iter().copied())).unwrap();
            let crate::cli::Commands::Plan(args) = cli.command else { panic!("plan") };
            assert_eq!(options(&args).unwrap().layout.unwrap().dense_qwen_kv, expected);
        }
    }

    #[test]
    fn glm_flash_index_cache_reaches_the_layout() {
        use clap::Parser;
        use cuteafd_loader::serving_capacity::GlmfIndexCache;
        let parse = |extra: &[&str]| crate::cli::Cli::try_parse_from(
            ["cuteafd", "plan", "/not-read", "--layout"].into_iter().chain(extra.iter().copied()));
        for (extra, index) in [(&[][..], GlmfIndexCache::Keys),
            (&["--index-cache", "keys"][..], GlmfIndexCache::Keys),
            (&["--index-cache", "compact"][..], GlmfIndexCache::Compact)] {
            let crate::cli::Commands::Plan(args) = parse(extra).unwrap().command else { panic!("plan") };
            assert_eq!(options(&args).unwrap().layout.unwrap().glmf_index, index);
        }
        assert!(parse(&["--index-cache", "auto"]).is_err());
    }

    #[test]
    fn glm_flash_prefix_marks_reach_the_layout() {
        use clap::Parser;
        let parse = |extra: &[&str]| crate::cli::Cli::try_parse_from(
            ["cuteafd", "plan", "/not-read", "--layout"].into_iter().chain(extra.iter().copied()));
        for (extra, pool) in [(&[][..], false), (&["--prefix-marks", "arena"][..], false),
            (&["--prefix-marks", "pool"][..], true)] {
            let crate::cli::Commands::Plan(args) = parse(extra).unwrap().command else { panic!("plan") };
            assert_eq!(options(&args).unwrap().layout.unwrap().glmf_pool_marks, pool);
        }
        assert!(parse(&["--prefix-marks", "host"]).is_err());
    }

    /// The flags the launcher's encoder placement plan passes for GLM 5.3 Flash size its layout as
    /// serve-glmf allocates: its recurrent state holds max(--slots 8, --max-sequences) slots
    /// (`serve.rs`, `engine_args.slots.max(args.max_sequences)`) and its mark arena counts
    /// min(--max-sequences, DECODE_ROWS = 64) lanes (`serve.rs`, `prefix.mark_rule(lanes)`), over the
    /// entries and mark budget it gets. The launcher passes those counts as --state-slots and
    /// --mark-lanes beside --concurrency; without them the planner derives both from --concurrency.
    #[test]
    fn glm_flash_serving_knobs_reach_the_layout() {
        use clap::Parser;
        let mut config = cuteafd_loader::plan::testing::glm5_flash_config(45);
        config["text_config"]["layer_types"] = serde_json::json!((0..45)
            .map(|l| if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect::<Vec<_>>());
        let cfg = cuteafd_loader::families::glm5_flash::GlmNextConfig::from_hf(&config).unwrap();
        let rank = cuteafd_loader::serving_capacity::glm_flash_cache_geometry(&cfg, 45).unwrap().ranks[0];
        // The standard 45-layer FP32-state geometry: a mark and a sequence's state are 147,619,840 B.
        let (mark, per_sequence) = (rank.retained_mark_bytes, rank.active_state_per_sequence_bytes);
        assert_eq!((mark, per_sequence), (147_619_840, 147_619_840));
        let snapshot = tempfile::tempdir().unwrap();
        write_snapshot(snapshot.path(), &config, &cuteafd_loader::plan::testing::glm5_flash_tensors(&config), None);
        let model = snapshot.path().display().to_string();
        let layout = |extra: &[&str]| {
            let argv = ["cuteafd", "plan", model.as_str(), "--layout", "--spark-ranks", "4", "--rtx", "1",
                "--gpu-memory-cap", "96GiB", "--pool-tokens", "0", "--drafter-gib", "2"];
            let cli = crate::cli::Cli::try_parse_from(argv.into_iter().chain(extra.iter().copied())).unwrap();
            let crate::cli::Commands::Plan(mut args) = cli.command else { panic!("plan") };
            args.memory = cli.memory;
            options(&args).unwrap()
        };
        // (marks, state) bytes on the GPU.
        let planned = |extra: &[&str]| {
            let memory = plan(std::path::Path::new(&model), &layout(extra)).unwrap().memory_layout.unwrap();
            let group = |name: &str| memory.devices[0].items.iter().filter(|i| i.group == name).map(|i| i.bytes)
                .sum::<u64>();
            (group("marks"), group("KDA state and replay") + group("DSA index tails and replay") + group("commit tables"))
        };
        let state = |slots: u64| rank.fixed_state_bytes + per_sequence * slots + rank.speculative_replay_bytes;
        // serve-glmf's own counts for `sequences`, `entries` and a mark budget (MiB).
        let served = |sequences: u64, entries: u64, mib: u64| {
            let (slots, lanes) = (cuteafd_core::concurrency::state_slots(sequences), sequences);
            (cuteafd_core::prefix::mark_slots_with_host(lanes, entries, mark, mib << 20, true) * mark, state(slots))
        };
        // The launcher's arguments for CONCURRENCY, PREFIX_CACHE_ENTRIES and PREFIX_CACHE_MARK_MIB.
        let launcher = |sequences: u64, entries: u64, mib: Option<u64>| {
            let mut argv = vec!["--concurrency".to_string(), sequences.to_string(), "--state-slots".into(),
                cuteafd_core::concurrency::state_slots(sequences).to_string(), "--mark-lanes".into(), sequences.to_string(),
                "--prefix-cache-entries".into(), entries.to_string()];
            if let Some(mib) = mib { argv.extend(["--prefix-cache-mark-mib".into(), mib.to_string()]); }
            argv
        };
        for (sequences, entries, mib) in [(1, 0, None), (1, 20, None), (5, 6, Some(1971)), (8, 20, None),
            (16, 20, None), (16, 6, Some(1971)), (64, 20, None)] {
            let argv = launcher(sequences, entries, mib);
            let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
            assert_eq!(planned(&argv), served(sequences, entries, mib.unwrap_or(2048)), "{argv:?}");
        }
        // Host-on reserves a fixed staging working set even when mark lanes change.
        let staging = cuteafd_core::prefix::HOST_MARK_STAGING_SLOTS;
        assert_eq!(planned(&launcher(1, 0, None).iter().map(String::as_str).collect::<Vec<_>>()), (0, state(1)));
        assert_eq!(planned(&["--concurrency", "1", "--prefix-cache-entries", "0"]), (0, state(1)));
        for concurrency in [16, 64] {
            let args = launcher(concurrency, 20, None);
            assert_eq!(planned(&args.iter().map(String::as_str).collect::<Vec<_>>()),
                (staging * mark, state(cuteafd_core::concurrency::state_slots(concurrency))));
            assert_eq!(planned(&["--concurrency", &concurrency.to_string()]),
                (staging * mark, state(cuteafd_core::concurrency::state_slots(concurrency))));
        }
        let default_sequences = cuteafd_core::serving_capacity::LARGE_CARD_DEFAULT_CONCURRENCY;
        assert_eq!(planned(&[]), served(default_sequences, 20, 2048));
        // Host-off retention remains budgeted separately from the C active state slots.
        assert_eq!(planned(&["--host-cache-bytes", "0", "--concurrency", "16"]), (cuteafd_core::prefix::mark_slots_for(16, 20, mark, 2048 << 20) * mark, state(16)));
        // The other knobs serve-glmf gets reach the layout as given.
        let custom = layout(&["--concurrency", "16", "--state-slots", "16", "--mark-lanes", "16",
            "--prefix-cache-entries", "6", "--prefix-cache-mark-mib", "1971", "--replay-records", "shared"]);
        let knobs = custom.layout.as_ref().unwrap();
        assert_eq!((knobs.concurrency, knobs.state_slots, knobs.glmf_mark_lanes, knobs.mimo_prefix_entries,
            knobs.mimo_prefix_mark_bytes, knobs.glmf_shared_replay, knobs.glmf_auto_replay),
            (16, Some(16), Some(16), 6, 1971 << 20, true, false));
        let auto_replay = layout(&["--replay-records", "auto"]);
        let auto_knobs = auto_replay.layout.as_ref().unwrap();
        assert_eq!((auto_knobs.glmf_shared_replay, auto_knobs.glmf_auto_replay), (false, true));
        for zero in ["--state-slots", "--mark-lanes"] {
            assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout", zero, "0"]).is_err());
        }
    }

    #[test]
    fn layout_refuses_full_storage_shortfall_even_when_weights_fit() {
        let snapshot = tempfile::tempdir().unwrap();
        write_snapshot(snapshot.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        let tiny = PlanArgs { memory: crate::cli::MemoryArgs { gpu_memory_cap: Some(2 << 30), ..args(snapshot.path(), 4, true).memory },
            ..args(snapshot.path(), 4, true) };
        let mut weight_only = options(&tiny).unwrap();
        weight_only.layout = None;
        assert_eq!(weight_only.coordinator_budget_bytes, 80 << 30);
        assert_eq!(weight_only.coordinator_gpu_budget_bytes, 2 << 30);
        let inventory = plan(snapshot.path(), &weight_only).unwrap();
        assert!(!inventory.fits && !inventory.placement_supported && !inventory.executable());
        assert!(inventory.hints.iter().any(|h| h.what.contains("MiMo placement")
            && h.what.contains("memory budget 'rtx0'")), "{:?}", inventory.hints);
        let report = plan(snapshot.path(), &options(&tiny).unwrap()).unwrap();
        assert!(!report.fits && !report.executable());
        assert!(report.hints.iter().any(|h| h.what.contains("MiMo placement")
            && h.what.contains("memory budget 'rtx0'")), "{:?}", report.hints);
        let error = run_plan(tiny).unwrap_err();
        assert!(error.to_string().contains("is not servable"), "{error:#}");
    }

    #[test]
    fn resident_weight_shortfall_refuses_even_when_storage_is_sufficient() {
        let snapshot = tempfile::tempdir().unwrap();
        write_snapshot(snapshot.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        for layout in [false, true] {
            let request = PlanArgs { layout, memory: crate::cli::MemoryArgs { gpu_memory_cap: Some((95 << 30) + (1 << 29)),
                ..args(snapshot.path(), 4, true).memory }, ..args(snapshot.path(), 4, true) };
            let admitted = plan(snapshot.path(), &options(&request).unwrap()).unwrap();
            assert!(admitted.executable(), "{:?}", admitted.hints);
            let capped = PlanArgs { coordinator_weight_budget_gib: Some(0.125), ..request };
            let selected = options(&capped).unwrap();
            assert_eq!(selected.coordinator_budget_bytes, 128 << 20);
            assert_eq!(selected.coordinator_gpu_budget_bytes, (95 << 30) + (1 << 29));
            let refused = plan(snapshot.path(), &selected).unwrap();
            assert!(!refused.fits && !refused.executable());
            use cuteafd_loader::families::mimo_v2::{MimoV2Config, resident::{MimoResidentLayout, MimoResidentOptions},
                projection::MimoProjectionRepresentation as R};
            let checkpoint = cuteafd_loader::plan::Checkpoint::open(snapshot.path()).unwrap();
            let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
            let ranks = selected.layout.as_ref().map_or(1, |options| options.rtx_bytes.len());
            use cuteafd_loader::families::mimo_v2::weight_policy::{default_policy, MimoDefaultPolicy};
            let qualified_fp8 = default_policy(&checkpoint, &cfg) == MimoDefaultPolicy::Fp8;
            let representation = |name: &str| {
                let tensor = checkpoint.tensors.iter().find(|tensor| tensor.meta.name == name).unwrap();
                if tensor.meta.dtype == cuteafd_core::DType::Bf16 && !qualified_fp8 { R::Bf16 } else { R::Fp8 }
            };
            let owners = MimoResidentLayout::new(&checkpoint, &cfg, &MimoResidentOptions {
                layers: cfg.layers, coordinator_ranks: ranks, checkpoint_tp: 1, native_mtp_layers: 0,
                gpu_embedding: true, head_format: representation("lm_head.weight"),
                output_formats: (0..cfg.layers).map(|layer| {
                    let name = format!("model.layers.{layer}.self_attn.o_proj.weight");
                    let format = representation(&name);
                    (name, format)
                }).collect(),
            }).unwrap();
            let required = owners.ranks[0].iter().map(|owner| owner.bytes).sum::<u64>();
            let refusal = format!("rtx0 resident weights need {required} bytes, over the {} byte weight budget", 128 << 20);
            assert!(refused.hints.iter().any(|h| h.what.contains(&refusal)), "{:?}", refused.hints);
            let error = run_plan(capped).unwrap_err();
            assert!(error.to_string().contains("is not servable"), "{error:#}");
        }
    }

    /// `cuteafd plan` plans the vision policy serve-glmf serves: automatic vision on a GLM 5.3 Flash
    /// checkpoint whose own chat template cannot mark images is off unless the launcher passes the
    /// template source; explicit placements, other families and unreadable chat files plan as given.
    #[test]
    fn glm_flash_auto_vision_plans_text_when_the_template_cannot_mark_images() {
        use clap::Parser;
        use crate::families::glm5_flash::serve::auto_vision_tests::{chat_files, MARKS_IMAGES, TEXT_ONLY};
        use cuteafd_loader::plan::MediaMode;
        let snapshot = tempfile::tempdir().unwrap();
        let model = snapshot.path().display().to_string();
        let planned = |family: &str, extra: &[&str]| {
            let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", model.as_str()].into_iter()
                .chain(extra.iter().copied())).unwrap();
            let vision = crate::resolve_vision(cli.vision, None);
            let crate::cli::Commands::Plan(mut args) = cli.command else { panic!("plan") };
            args.vision = vision;
            glm_flash_vision(&args, family, snapshot.path())
        };
        chat_files(snapshot.path(), TEXT_ONLY, true, None);
        assert_eq!(planned("glm5_flash", &[]), MediaMode::Off);
        assert_eq!(planned("glm5_flash", &["--vision", "auto"]), MediaMode::Off);
        assert_eq!(planned("glm5_flash", &["--chat-template-from", "zai-org/GLM-5.3-Flash"]), MediaMode::Auto);
        for mode in ["spark", "spark:1", "rtx", "rtx:0", "off"] {
            assert_eq!(planned("glm5_flash", &["--vision", mode]), mode.parse().unwrap(), "{mode}");
        }
        assert_eq!(planned("mimo_v2", &[]), MediaMode::Auto);
        // A plan reads headers: without the chat files it keeps the policy it was given.
        std::fs::remove_file(snapshot.path().join("tokenizer.json")).unwrap();
        assert_eq!(planned("glm5_flash", &[]), MediaMode::Auto);
        chat_files(snapshot.path(), MARKS_IMAGES, true, None);
        assert_eq!(planned("glm5_flash", &[]), MediaMode::Auto);
        chat_files(snapshot.path(), TEXT_ONLY, true, Some("zai-org/GLM-5.3-Flash"));
        assert_eq!(planned("glm5_flash", &[]), MediaMode::Auto);
        chat_files(snapshot.path(), TEXT_ONLY, false, None);
        assert_eq!(planned("glm5_flash", &[]), MediaMode::Auto);
    }

    #[test]
    fn require_ready_exits_by_verdict_and_options_fail_typed() {
        let flash = tempfile::tempdir().unwrap();
        write_snapshot(flash.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        let pro = tempfile::tempdir().unwrap();
        write_snapshot(pro.path(), &mimo_pro_config(), &mimo_pro_tensors(), Some(8));
        // Complete inventories at packaged worlds are servable.
        run_plan(args(flash.path(), 4, true)).unwrap();
        run_plan(args(pro.path(), 6, true)).unwrap();
        // No implicit local native package is installed in this CPU fixture.
        run_plan(args(pro.path(), 0, false)).unwrap();
        let local_error = run_plan(args(pro.path(), 0, true)).unwrap_err();
        assert!(local_error.to_string().contains("is not servable"), "{local_error:#}");
        assert!(plan(pro.path(), &options(&args(pro.path(), 0, false)).unwrap()).unwrap().hints
            .iter().any(|h| h.what.contains("BF16 package manifest")));
        // V2.6 Pro has no tp4 MXFP4 layout: a verdict, an error only with --require-ready.
        run_plan(args(pro.path(), 4, false)).unwrap();
        let error = run_plan(args(pro.path(), 4, true)).unwrap_err();
        assert!(error.to_string().contains("is not servable by this build"), "{error:#}");
        // Options that describe no deployment fail before the checkpoint is read.
        for (ranks, budget) in [(9, 100u64 << 30), (4, 0)] {
            let error = run_plan(PlanArgs { memory: crate::cli::MemoryArgs { spark_memory_cap: Some(budget), ..Default::default() },
                ..args(flash.path(), ranks, false) }).unwrap_err();
            assert!(matches!(error.downcast_ref::<PlanError>(), Some(PlanError::InvalidOption { .. })),
                "{ranks} ranks, {budget} bytes: {error:#}");
        }
    }
}
