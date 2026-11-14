//! Dense Qwen3.5/3.8 native whole-layer execution and shared admission.
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
                "Native BF16/block-FP8 whole layers on one or two RTX ranks; exact owner-local KV/GDN, prefix marks and rank-local decode graphs. Head/context sharding needs native exports (#2133)".into()],
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
    fn contract_admission(&self) -> bool { true }
    fn spec(&self) -> &ModelSpec { &self.spec }
    fn native_serving_admission(&self) -> bool { true }
    fn vision_mode(&self, mode: crate::plan::MediaMode) -> crate::plan::MediaMode {
        if mode == crate::plan::MediaMode::Auto { crate::plan::MediaMode::Off } else { mode }
    }
    fn placement(&self, input: &crate::plan::PlanInput, report: &mut crate::plan::PlanReport,
        options: &mut crate::plan::layout::LayoutOptions, requested: crate::plan::ExpertPlacement)
        -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        if requested != crate::plan::ExpertPlacement::Local
            && !matches!(report.vision, crate::plan::MediaMode::Spark(_)) {
            return Some(Err(anyhow::anyhow!("dense Qwen has no Spark layout or dense-FFN offboarding")));
        }
        Some(native_layout(&self.header.config, input, report, options))
    }
    fn cache_geometry(&self, options: crate::serving_capacity::CacheOptions)
        -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        use crate::serving_capacity::*;
        if !(1..=2).contains(&options.coordinator_ranks) {
            return Err(CacheGeometryError::Unsupported { family: "qwen3_5", what: "native whole-layer execution needs one or two RTX ranks" });
        }
        let cfg = &self.header.config;
        let buffers = crate::placement::families::qwen3_5::NativeBuffers::new(cfg, 0, 1, 0)
            .map_err(|_| CacheGeometryError::Overflow("dense native cache"))?;
        let state = buffers.mark_bytes as u64;
        let gdn = cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Gdn).count() as u64;
        if options.coordinator_ranks == 2 {
            return Err(CacheGeometryError::Unsupported { family: "qwen3_5",
                what: "whole-layer cache owners depend on the solved contiguous cut; see the per-rank phase ledger" });
        }
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
    use crate::placement::{families::qwen3_5::{self, DenseFacts, ServingFacts}, ArchContext,
        Onboard, ProgramSet, ResidentWeights};
    use crate::placement::planner::{Location, Operation, Pin, PrefillShape, Request};
    use cuteafd_core::memory_layout::{DeviceKind, DeviceLayout, MemoryLayout};
    let spark_vision = matches!(report.vision, crate::plan::MediaMode::Spark(Some(_)))
        || (matches!(report.vision, crate::plan::MediaMode::Spark(_)) && report.spark_ranks > 0);
    let ranks = options.rtx_bytes.len();
    ensure!((1..=2).contains(&ranks) && (report.spark_ranks == 0 || spark_vision),
        "dense Qwen needs one or two coordinator GPUs and zero expert Sparks");
    if ranks == 1 && report.requested_attention == Some(crate::placement::AttentionPlacement::Layers) {
        report.refuse_placement("dense Qwen layer ranges need two coordinator GPUs".into(),
            "Select two RTX ranks or automatic single-rank placement.");
        return Ok(MemoryLayout { devices: vec![], pool_tokens: 0, waste: vec![], notes: vec![] });
    }
    if ranks == 2 {
        if matches!(report.requested_attention, Some(crate::placement::AttentionPlacement::Heads | crate::placement::AttentionPlacement::Context)) {
            let mode = report.requested_attention.expect("explicit attention");
            report.refuse_placement(format!("dense Qwen {mode} attention is an unbuilt feature"),
                "Add native half-head GDN, GQA producer/output and row-parallel dense FFN exports, sliced checkpoint operands, and shared peer reductions; context additionally needs owner-partitioned KV and shared softmax merge. Select layers for the native whole-layer executor (#1991).");
            return Ok(MemoryLayout { devices: vec![], pool_tokens: 0, waste: vec![], notes: vec![] });
        }
        report.attention_placement = crate::placement::AttentionPlacement::Layers;
    }
    ensure!(!options.host_embedding, "dense Qwen requires device token embedding");
    ensure!((report.vision == crate::plan::MediaMode::Off || spark_vision)
        && report.audio == crate::plan::MediaMode::Off,
        "dense Qwen media execution is not qualified");
    ensure!(options.drafter_bytes == 0 && !(options.dense_qwen_mtp && options.dense_qwen_drafter_snapshot.is_some()),
        "dense native planner requires one checkpoint-native drafter admission");
    let rows = qwen3_5::PREFILL_ROWS as u64;
    let prefill = crate::plan::layout::resolve_prefill(input, report, options, 0,
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
        let name = qwen3_5::native_program_name(prefix, op, rows, options.dense_qwen_kv, fp8);
        let entry = entries.iter().find(|entry| entry["name"] == name)
            .with_context(|| format!("missing native dense program {name}"))?;
        let spec = qwen3_5::NativeProgram::from_manifest(entry)?;
        qwen3_5::validate_projection_format(op, fp8, &spec)?;
        qwen3_5::validate(cfg, op, rows, options.dense_qwen_kv, &spec)?;
        qwen3_5::validate_abi(op, rows, &spec)?;
        scratch = scratch.max(spec.scratch.values().copied().max().unwrap_or(0));
        if op == "gdn" {
            let dtype = qwen3_5::norm_dtype(&spec)?;
            ensure!(norm.as_ref().is_none_or(|old| *old == dtype), "native GDN norm precision differs across shapes");
            norm = Some(dtype);
        }
    }
    ensure!(scratch > 0, "dense native prefill requires nonzero scratch");
    let slots = usize::try_from(options.resolved_state_slots("qwen3_5"))?;
    let mark = qwen3_5::NativeBuffers::new(cfg, scratch as usize, slots, 0)?.mark_bytes;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(options.resolved_concurrency("qwen3_5"),
        options.prefix_entries("qwen3_5"), mark as u64, options.mimo_prefix_mark_bytes, options.host_cache_bytes != Some(0)));
    let buffers = qwen3_5::NativeBuffers::new(cfg, scratch as usize, slots, usize::try_from(marks)?)?;
    let norm = norm.context("missing GDN precision")?;
    let weights = qwen3_5::NativeWeights::native(input.checkpoint(), cfg, norm.clone())?;
    let target_arena = crate::families::qwen3_5::weight_layout::target_arena(input.checkpoint(), cfg, norm,
        crate::families::qwen3_5::weight_layout::DenseFormat::detect(input.checkpoint())?)?;
    let capacity = options.rtx_bytes[0];
    // Physical card class fixes the CUDA context/module image and SM count;
    // `--gpu-memory-cap` only lowers the logical ceiling.
    let physical = ArchContext::for_device("sm_120", capacity);
    let sms = options.physical_sms.unwrap_or(physical.sms);
    let drafter = options.dense_qwen_drafter_snapshot.as_ref().map(|snapshot| {
        ensure!(crate::formats::block_dspark::supports_sequences(slots), "dense dSpark sequence capacity exceeded");
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let draft = crate::formats::block_dspark::BlockDsparkConfig::from_json(&config)?;
        ensure!(draft.hidden == cfg.hidden && draft.vocab == cfg.vocab && draft.taps.iter().all(|&layer| layer < cfg.attention.len()),
            "dense dSpark checkpoint differs from target geometry");
        qwen3_5::DrafterBytes::block_dspark(snapshot, &config, slots, sms as u64, cfg.max_context)
    }).transpose()?;
    let drafter = if options.dense_qwen_mtp {
        Some(qwen3_5::DrafterBytes::native_mtp(input.checkpoint(), cfg)?)
    } else { drafter };
    let host = options.host_memory.map(Ok)
        .unwrap_or_else(crate::formats::mapped_preload::host_memory_before_process)?;
    let mut budgets = qwen3_5::forecast_budgets(physical, capacity, set.module_bytes(), host)?;
    if ranks == 2 {
        let mut peer = qwen3_5::forecast_budgets(ArchContext::for_device("sm_120", options.rtx_bytes[1]),
            options.rtx_bytes[1], set.module_bytes(), host)?.remove(0);
        peer.id = "rtx1".into();
        peer.locations = vec![Location::coordinator(1)];
        budgets.insert(1, peer);
    }
    let operation = Operation::Serving;
    let onboard = match options.onboard.unwrap_or_default().resolve(0) {
        Onboard::Auto => Pin::Auto,
        Onboard::Layers(layers) => Pin::Exact(layers as u64),
    };
    let request = Request {
        context: crate::placement::planner::Context::resolve(cfg.max_context as u64,
            (options.context_tokens > 0).then_some(options.context_tokens)),
        pool: options.pool_tokens.filter(|&n| n > 0).map(Pin::Exact).unwrap_or(Pin::Auto),
        onboard,
        prefill: Pin::Exact(PrefillShape { rows: prefill.unit_rows, waves: prefill.waves }),
        operation,
        resident_weight_limit: Some(ResidentWeights(options.weight_budget_bytes)),
        spark_reply_formats: options.spark_reply_formats,
    };
    let record_bytes = options.dense_qwen_kv.record_bytes(cfg.kv_heads, cfg.head_dim) as u64;
    let geometry = qwen3_5::host_prefix_geometry(cfg, record_bytes, mark as u64);
    let chunk = geometry.chunk_bytes();
    let quota = if options.prefix_entries("qwen3_5") == 0 || options.host_cache_bytes == Some(0) { 0 }
        else { geometry.resolve(options.host_cache_bytes, cfg.max_context as u64,
            options.resolved_concurrency("qwen3_5"), host.0, chunk)
            .context("dense host prefix quota")? };
    let facts = DenseFacts { graph_variant: crate::serving_capacity::qwen35_graphs::variant(cfg, fp8, options.dense_qwen_kv), graphs_enabled: crate::serving_capacity::qwen35_graphs::enabled(), cfg, weights: &weights, target_arena: &target_arena, buffers: &buffers, slots, record_bytes, drafter,
        blas_head: options.bf16_target_head,
        // The shared scheduler's one selector holds `decode_rows` rows and
        // selects on the device by default.
        serving: Some(ServingFacts { capacity: qwen3_5::VERIFY_ROWS as u32, device_sampling: true }),
        host_prefix: crate::placement::planner::serving::HostPrefixOwners {
            quota_bytes: quota, chunk_bytes: chunk.min(quota) }, verify_check: false };
    let plan = if ranks == 2 { qwen3_5::dual::admit_dual(request, budgets, facts)?.plan }
        else { qwen3_5::admit(request, budgets, facts)? };
    let pool_tokens = plan.pool_tokens;
    if ranks == 2 {
        use crate::serving_capacity::{FamilyCacheGeometry, RankCacheGeometry, KvPlacement};
        let owner_geometry = (0..2).map(|rank| {
            let kinds = cfg.attention.iter().zip(&plan.decisions.attention).filter_map(|(&kind, &mode)|
                matches!(mode, crate::placement::LayerMode::Whole { gpu, .. } if gpu == rank).then_some(kind)).collect::<Vec<_>>();
            let recurrent = kinds.iter().filter(|&&kind| kind == HybridAttention::Gdn).count() as u64;
            let full = kinds.len() as u64 - recurrent;
            RankCacheGeometry { persistent_unit_bytes: 64 * full * record_bytes,
                active_state_per_sequence_bytes: recurrent * buffers.state_per_layer as u64,
                retained_mark_bytes: recurrent * buffers.state_per_layer as u64,
                speculative_replay_bytes: recurrent * buffers.replay_per_layer as u64, ..Default::default() }
        }).collect();
        if let Some(cache) = report.cache_requirements.as_mut() {
            cache.target_only_layouts.retain(|layout| layout.ranks.len() != 2);
            cache.target_only_layouts.push(FamilyCacheGeometry { logical_unit_rows: 64,
                placement: KvPlacement::PartitionedLayers, ranks: owner_geometry });
            cache.unavailable_layouts.retain(|failure| failure.coordinator_ranks != 2);
        }
    }
    report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&plan));
    report.host_budget_bytes = quota;
    let mut notes = vec![
        format!("Native dense admission: {} KV, exact GDN state, bounded decode/verify graph reserve; text-only native whole layers; fixed slots and prefix marks do not grow when the pool cap frees memory", options.dense_qwen_kv),
        format!("shared planner: pool {pool_tokens} tokens ({} pages of {} rows), {} prefill row unit, {} lanes, {} resident weight bytes, {} physical owners",
            pool_tokens / qwen3_5::PAGE_ROWS, qwen3_5::PAGE_ROWS, prefill.unit_rows, prefill.waves,
            plan.facts.resident_weights.iter().map(|owner| owner.weights.0).sum::<u64>(),
            plan.facts.allocations.iter().filter(|owner| owner.view.is_none()).count()),
    ];
    if ranks == 2 {
        notes.push(format!("native whole-layer ranges: {}; shared residual boundary/return hops; memory capacity, not parallel attention compute; rank-local lazy graphs (untraced bootstrap)",
            plan.decisions.attention.iter().enumerate().map(|(layer, mode)| format!("{layer}:{mode:?}")).collect::<Vec<_>>().join(",")));
    }
    let mut devices = (0..ranks).map(|rank| Ok(DeviceLayout { kind: DeviceKind::Rtx, index: rank as u32,
        capacity_bytes: options.rtx_bytes[rank].saturating_sub(crate::placement::reserve::gpu("qwen3_5")),
        worker_budget_bytes: 0, items: plan.items(Location::coordinator(rank as u8))?, kv_tokens: pool_tokens }))
        .collect::<anyhow::Result<Vec<_>>>()?;
    if spark_vision {
        let mut sparks = Vec::new();
        for rank in 0..report.spark_ranks {
            crate::plan::layout::ensure_tower_spark(&mut sparks, rank, options);
        }
        let source = crate::plan::layout::vision_source_bytes(report);
        let towers = crate::plan::layout::Towers {
            name: "Qwen",
            vision: Ok(if source > 0 {
                (source, crate::plan::layout::qwen_scratch_bytes(crate::media::QWEN_MAX_IMAGE_TOKENS as u64))
            } else { (0, 512 << 20) }),
            v41: false,
            audio: crate::plan::layout::no_audio(),
        };
        crate::plan::layout::resolve_encoder(report, &towers, &mut devices, &mut sparks, &[0], options, &mut notes);
        let media_ranks: std::collections::BTreeSet<usize> = report.encoder.as_ref()
            .map_or_else(Vec::new, crate::plan::encoder::EncoderPlacement::spark_ranks).into_iter().collect();
        sparks.retain(|device| media_ranks.contains(&(device.index as usize)));
        devices.extend(sparks);
    }
    Ok(MemoryLayout { devices, pool_tokens, waste: vec![], notes })
}
