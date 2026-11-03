//! Dense Qwen3.5/3.8: recognized header contract, not yet an executable family.
use crate::families::qwen::HybridAttention;
use crate::families::qwen3_5::Qwen35Checkpoint;
use crate::plan::{checkpoint::Checkpoint, family::*, format, names::indexed, spec::*};

pub struct Qwen35;
pub static QWEN35: Qwen35 = Qwen35;

impl Family for Qwen35 {
    fn decode_lanes(&self) -> crate::plan::decode_lanes::LaneSupport {
        crate::plan::decode_lanes::LaneSupport::One {
            reason: "dense Qwen runs RTX-only: no Spark round trip for a second lane to hide (#744)" }
    }
    fn id(&self) -> &'static str { "qwen3_5" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Serving }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|a| a == "Qwen3_5ForConditionalGeneration")
            || checkpoint.model_type() == Some("qwen3_5")
    }
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        let header = Qwen35Checkpoint::open(checkpoint).map_err(ConfigError::from_anyhow)?;
        let cfg = &header.config;
        let spec = ModelSpec {
            family: self.id(), architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
            hidden: cfg.hidden, vocab: cfg.vocab,
            layers: cfg.attention.iter().map(|&kind| LayerSpec {
                attention: match kind {
                    HybridAttention::Gdn => AttentionKind::GatedDeltaNet,
                    HybridAttention::Full => AttentionKind::Gqa { heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim },
                },
                ffn: FfnKind::Dense { intermediate: cfg.intermediate },
                rope: (kind == HybridAttention::Full).then_some(RopeSpec { dims: cfg.rope_dim, theta: cfg.rope_theta }),
            }).collect(),
            moe: None, speculator: (cfg.mtp_layers > 0).then_some(SpeculatorSpec::NativeMtp { layers: cfg.mtp_layers }),
            tables: vec![], vision: checkpoint.config.get("vision_config").is_some(),
            notes: vec!["0-Spark dense target; no router, experts, mHC, PLE or QSA indexer".into(),
                "Transformers 5.13.0 reference: GDN SiLU, full-attention sigmoid; output_gate_type=swish is not a GQA gate selector".into(),
                format!("BF16 full KV {} bytes/token; exact FP32 GDN + BF16 convolution state {} bytes/sequence or retained mark; native context {} tokens",
                    cfg.full_kv_bytes_per_token(crate::families::qwen4::Qwen4KvCache::Bf16), cfg.recurrent_state_bytes(), cfg.max_context),
                "Native single-GPU BF16 allocation planning is available with matching exports; serving remains unqualified, with BF16 KV and no captured graphs".into()],
        };
        Ok(Box::new(Model { header, spec }))
    }
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        let (stem, _) = format::split_stem(name);
        let component = match stem {
            "lm_head" => Some(Component::LmHead),
            "model.language_model.embed_tokens" => Some(Component::Embedding),
            "model.language_model.norm" => Some(Component::Norm),
            _ => None,
        };
        if let Some(component) = component { return Some(TensorRole::new(component)); }
        if stem.starts_with("mtp.") { return Some(TensorRole::new(Component::Speculator)); }
        if stem.starts_with("model.visual.") { return Some(TensorRole::new(Component::Vision)); }
        let (layer, rest) = indexed(stem.strip_prefix("model.language_model.")?, "layers.")?;
        if layer >= spec.layers.len() { return None; }
        let component = if matches!(rest, "input_layernorm" | "post_attention_layernorm") { Component::Norm }
            else if rest.starts_with("self_attn.") || rest.starts_with("linear_attn.") { Component::Attention }
            else if matches!(rest, "mlp.gate_proj" | "mlp.up_proj" | "mlp.down_proj") { Component::DenseFfn }
            else { return None; };
        Some(TensorRole::layer(component, layer))
    }
    fn component_hint(&self, component: Component) -> Option<Hint> {
        let missing = match component {
            Component::Attention => "SiLU-gated GDN and unrestricted causal paged GQA (no QSA), M-RoPE, dense geometry exports",
            Component::DenseFfn => "single-stream residual + dense SiLU gate/up/down projections",
            Component::Vision => "bundled Qwen tower merger/insertion at H4096/H5120; no multimodal parity claim",
            Component::Speculator => "dense MTP feedback fc [H,2H], normalized hidden/token input order and dense block; MTP off for first golden",
            _ => "single-stream zero-centered norms, final norm/head, admission and exact prefix-state lifecycle",
        };
        Some(Hint { what: format!("qwen3_5 {} has tensors without an executable kernel", component.label()),
            how: format!("Needs {missing} using shared Qwen operators; gate against scripts/qualify/qwen3_5/reference.py.") })
    }
    fn optional(&self, component: Component) -> bool {
        matches!(component, Component::Vision | Component::Speculator)
    }
}

struct Model { header: Qwen35Checkpoint, spec: ModelSpec }
impl FamilyModel for Model {
    fn spec(&self) -> &ModelSpec { &self.spec }
    fn native_serving_admission(&self) -> bool { true }
    fn vision_mode(&self, mode: crate::plan::MediaMode) -> crate::plan::MediaMode {
        if mode == crate::plan::MediaMode::Auto { crate::plan::MediaMode::Off } else { mode }
    }
    fn placement(&self, input: &crate::plan::PlanInput, report: &mut crate::plan::PlanReport,
        options: &mut crate::plan::layout::LayoutOptions, requested: crate::plan::ExpertPlacement)
        -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        if requested != crate::plan::ExpertPlacement::Local {
            return Some(Err(anyhow::anyhow!("dense Qwen has no Spark layout or dense-FFN offboarding")));
        }
        Some(native_layout(&self.header.config, input, report, options))
    }
    fn cache_geometry(&self, options: crate::serving_capacity::CacheOptions)
        -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        use crate::serving_capacity::*;
        if options.coordinator_ranks != 1 {
            return Err(CacheGeometryError::Unsupported { family: "qwen3_5", what: "dense target currently owns one GPU" });
        }
        let cfg = &self.header.config;
        let buffers = crate::placement::families::qwen3_5::NativeBuffers::new(cfg, 0, 1, 0, false)
            .map_err(|_| CacheGeometryError::Overflow("dense native cache"))?;
        let state = buffers.mark_bytes as u64;
        let gdn = cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Gdn).count() as u64;
        Ok(Some(FamilyCacheGeometry { logical_unit_rows: 64, placement: KvPlacement::SingleDevice,
            ranks: vec![RankCacheGeometry { persistent_unit_bytes: 64 * cfg.full_kv_bytes_per_token(options.dense_qwen_kv) as u64,
                active_state_per_sequence_bytes: state, retained_mark_bytes: state,
                speculative_replay_bytes: gdn * buffers.replay_per_layer as u64, ..Default::default() }] }))
    }
    fn accepts(&self, _role: &TensorRole, stem: &str, operand: &mut format::QuantOperand) -> Result<(), String> {
        self.header.accepts(stem, operand)
    }
}

fn native_layout(cfg: &crate::families::qwen3_5::Qwen35Config, input: &crate::plan::PlanInput,
    report: &mut crate::plan::PlanReport, options: &mut crate::plan::layout::LayoutOptions)
    -> anyhow::Result<cuteafd_core::memory_layout::MemoryLayout> {
    use anyhow::{ensure, Context};
    use crate::placement::{self, families::qwen3_5::{DenseFacts, NativeBuffers, NativeWeights, PREFILL_ROWS},
        planner, ProgramSet};
    use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};
    ensure!(options.rtx_bytes.len() == 1 && report.spark_ranks == 0,
        "dense Qwen currently requires one coordinator GPU and zero Sparks");
    if options.context_tokens != 0 && options.context_tokens != cfg.max_context as u64 {
        return Err(planner::Refusal::InvalidPin("dense executor requires checkpoint maximum context").into());
    }
    ensure!(!options.host_embedding, "dense Qwen requires device token embedding");
    ensure!(report.vision == crate::plan::MediaMode::Off && report.audio == crate::plan::MediaMode::Off,
        "dense Qwen media execution is not qualified");
    ensure!(options.drafter_bytes == 0 && !(options.dense_qwen_mtp && options.dense_qwen_drafter_snapshot.is_some()),
        "dense native planner requires one checkpoint-native drafter admission");
    let rows = PREFILL_ROWS as u64;
    let prefill = crate::plan::layout::resolve_prefill(report, options, 0,
        crate::serving_capacity::prefill_waves::PrefillLimits {
            max_unit_rows: rows, max_span_rows: rows, max_waves: 1, ..Default::default()
        })?;
    ensure!(prefill.unit_rows == rows && prefill.span_rows == rows,
        "dense native prefill requires one coordinator unit of {rows} rows and one wave, with no Spark exchange");
    let prefix = match (cfg.hidden, cfg.intermediate, cfg.heads, cfg.gdn.value_heads) {
        (4096, 12288, 16, 32) => "qwen35", (5120, 17408, 24, 48) => "qwen38",
        _ => anyhow::bail!("dense Qwen geometry lacks native exports"),
    };
    let path = options.workspace_manifest.as_deref().unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(path)
        .context("dense native admission requires its PROGRAMS.json manifest")?)?;
    let set = ProgramSet::from_manifest(&manifest, &[prefix]);
    let entries = manifest["programs"].as_array().context("native manifest lacks programs")?;
    let fp8 = input.checkpoint().tensors.iter().any(|t|
        t.meta.name == "model.language_model.layers.0.mlp.gate_proj.weight" && t.meta.dtype == cuteafd_core::DType::F8E4M3);
    let mut scratch = 0;
    let mut norm = None;
    for (op, rows) in [("norm", 0), ("add", 0), ("gdn_commit", 0)].into_iter().chain(
        [64, 4096].into_iter().flat_map(|rows| ["gdn", "attn_producer", "full_gqa", "attn_o", "ffn"].map(|op| (op, rows)))) {
        let name = placement::families::qwen3_5::native_program_name(prefix, op, rows, options.dense_qwen_kv, fp8);
        let entry = entries.iter().find(|entry| entry["name"] == name)
            .with_context(|| format!("missing native dense program {name}"))?;
        let spec = placement::families::qwen3_5::NativeProgram::from_manifest(entry)?;
        placement::families::qwen3_5::validate_projection_format(op, fp8, &spec)?;
        placement::families::qwen3_5::validate(cfg, op, rows, options.dense_qwen_kv, &spec)?;
        placement::families::qwen3_5::validate_abi(op, rows, &spec)?;
        scratch = scratch.max(spec.scratch.values().copied().max().unwrap_or(0));
        if op == "gdn" {
            let dtype = placement::families::qwen3_5::norm_dtype(&spec)?;
            ensure!(norm.as_ref().is_none_or(|old| *old == dtype), "native GDN norm precision differs across shapes");
            norm = Some(dtype);
        }
    }
    ensure!(scratch > 0, "dense native prefill requires nonzero scratch");
    let slots = usize::try_from(options.resolved_state_slots("qwen3_5"))?;
    let mark = NativeBuffers::new(cfg, scratch as usize, slots, 0, options.full_prefill_logits)?.mark_bytes;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(options.resolved_concurrency("qwen3_5"),
        options.prefix_entries("qwen3_5"), mark as u64, options.mimo_prefix_mark_bytes, options.host_cache_bytes != Some(0)));
    let buffers = NativeBuffers::new(cfg, scratch as usize, slots, usize::try_from(marks)?, options.full_prefill_logits)?;
    let weights = NativeWeights::native(input.checkpoint(), cfg, norm.context("missing GDN precision")?)?;
    let capacity = options.rtx_bytes[0];
    // A logical cap does not turn a PRO into a 5090. Offline plans default
    // to the PRO reference; an explicit physical SM witness selects the class.
    let physical = *placement::ARCH_CONTEXTS.iter().find(|card| card.arch == "sm_120"
        && card.sms == options.physical_sms.unwrap_or(188)).context("unknown dense coordinator physical class")?;
    let context = physical.context_bytes + set.module_bytes();
    let draft_sequences = usize::try_from(options.resolved_concurrency("qwen3_5"))?;
    let draft_anchor_only = options.dense_qwen_drafter_snapshot.is_some()
        && !crate::formats::block_dspark::supports_sequences(draft_sequences);
    let drafter = options.dense_qwen_drafter_snapshot.as_ref().filter(|_| !draft_anchor_only).map(|snapshot| {
        let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let draft = crate::formats::block_dspark::BlockDsparkConfig::from_json(&config)?;
        ensure!(draft.hidden == cfg.hidden && draft.vocab == cfg.vocab && draft.taps.iter().all(|&layer| layer < cfg.attention.len()),
            "dense dSpark checkpoint differs from target geometry");
        ensure!(draft.window.is_none(), "dense dSpark requires full-context cache");
        Ok::<_, anyhow::Error>(crate::formats::block_dspark::BlockDraftLayout::read(snapshot, draft_sequences, draft_sequences,
            cfg.max_context, 64, physical.sms as usize,
            crate::families::glm5::draft_representation::GlmDraftRepresentation::Bf16Only, 0,
            crate::families::qwen4::Qwen4KvCache::Bf16)?)
    }).transpose()?;
    use placement::families::qwen3_5::{DenseDraft, NativeMtpLayout, NativeMtpUsage};
    let native_mtp = options.dense_qwen_mtp.then(|| NativeMtpLayout::new(input.checkpoint(), cfg,
        64, NativeMtpUsage::SERVING)).transpose()?;
    let drafter = native_mtp.as_ref().map(DenseDraft::NativeMtp).or_else(|| drafter.as_ref().map(DenseDraft::Block));
    let host = match options.host_memory {
        Some(host) => host,
        None => {
            #[cfg(test)]
            panic!("planner tests must inject LayoutOptions.host_memory");
            #[cfg(not(test))]
            crate::formats::mapped_preload::host_memory()?
        }
    };
    let budgets = placement::families::qwen3_5::forecast_budgets(&physical, capacity, set.module_bytes(), host)?;
    let onboard = match options.onboard.unwrap_or_default().resolve(0) {
        placement::Onboard::Auto => planner::Pin::Auto,
        placement::Onboard::Layers(layers) => planner::Pin::Exact(layers as u64),
    };
    let request = planner::Request { context: cfg.max_context as u64,
        pool: options.pool_tokens.map_or(planner::Pin::Auto, planner::Pin::Exact), onboard,
        prefill: planner::Pin::Exact(planner::PrefillShape { rows: prefill.unit_rows, waves: prefill.waves }),
        spark_reply_format: None,
        operation: if options.full_prefill_logits { planner::Operation::Scoring } else { planner::Operation::Serving } };
    let record_bytes = options.dense_qwen_kv.record_bytes(cfg.kv_heads, cfg.head_dim) as u64;
    let serving = placement::families::qwen3_5::ServingFacts::new(cfg, &buffers, record_bytes,
        drafter.map(DenseDraft::page_bytes).transpose()?.unwrap_or(0),
        placement::families::qwen3_5::ServingRequest {
            concurrency: options.resolved_concurrency("qwen3_5"), prefix_entries: options.prefix_entries("qwen3_5"),
            host_cache_bytes: options.host_cache_bytes, host_headroom: 8 << 30, device_sampling: options.device_sampling,
        }, host.0)?;
    let result = placement::families::qwen3_5::admit(request, budgets, DenseFacts { cfg,
        weights: &weights, buffers: &buffers, slots, drafter, serving: Some(serving), record_bytes })?;
    let mut items = result.items(planner::Location::coordinator(0))?;
    for reservation in &result.facts.deferred {
        items.push(Item::new(reservation.category, &reservation.group, "", reservation.bytes, Basis::Formula));
    }
    for allowance in &result.facts.allowances {
        if allowance.location == planner::Location::coordinator(0) && allowance.lifetime.contains(planner::Phase::Ready, result.facts.phase_order) {
            items.push(Item::new(Category::Runtime, &allowance.source, "", allowance.bytes, Basis::Estimated));
        }
    }
    for demand in buffers.fixed().into_iter().filter(|d| d.bytes == 0) {
        items.push(Item::new(demand.category, demand.group, "", 0, Basis::Exact));
    }
    items.push(Item::new(Category::Runtime, "context and native modules", "native", context, Basis::Calibrated));
    let mut notes = vec![format!("Native dense admission: {} KV, exact GDN state, no captured graphs; text-only single GPU; fixed slots and prefix marks do not grow when the pool cap frees memory", options.dense_qwen_kv)];
    if draft_anchor_only {
        notes.push(format!("dense dSpark native sequence capacity {}; C{slots} serves anchor-only with no drafter allocation",
            crate::formats::block_dspark::MAX_SEQUENCES));
    }
    let pool_tokens = result.pool_tokens;
    let capacity_bytes = result.budgets[0].physical_bytes.min(result.budgets[0].logical_bytes)
        - result.budgets[0].headroom;
    report.allocation_plan = Some(result);
    Ok(MemoryLayout { devices: vec![DeviceLayout { kind: DeviceKind::Rtx, index: 0,
        capacity_bytes, worker_budget_bytes: 0, items, kv_tokens: pool_tokens }],
        pool_tokens, waste: vec![], notes })
}
