//! Dense Qwen3.5/3.8: recognized header contract, not yet an executable family.
use crate::families::qwen::HybridAttention;
use crate::families::qwen3_5::Qwen35Checkpoint;
use crate::plan::{checkpoint::Checkpoint, family::*, format, names::indexed, spec::*};

pub struct Qwen35;
pub static QWEN35: Qwen35 = Qwen35;

impl Family for Qwen35 {
    fn id(&self) -> &'static str { "qwen3_5" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Planned }
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
                    cfg.full_kv_bytes_per_token(), cfg.recurrent_state_bytes(), cfg.max_context),
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
        Some(Hint { what: format!("qwen3_5 {} is planned, not executable", component.label()),
            how: format!("Implement {missing} using shared Qwen operators; gate against scripts/qualify/qwen3_5/reference.py before enabling runtime.") })
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
        options: &crate::plan::layout::LayoutOptions, requested: crate::plan::ExpertPlacement)
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
        let buffers = crate::placement::families::qwen3_5::NativeBuffers::new(cfg, 0, 1, 0)
            .map_err(|_| CacheGeometryError::Overflow("dense native cache"))?;
        let state = buffers.mark_bytes as u64;
        let gdn = cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Gdn).count() as u64;
        Ok(Some(FamilyCacheGeometry { logical_unit_rows: 64, placement: KvPlacement::SingleDevice,
            ranks: vec![RankCacheGeometry { persistent_unit_bytes: 64 * cfg.full_kv_bytes_per_token() as u64,
                active_state_per_sequence_bytes: state, retained_mark_bytes: state,
                speculative_replay_bytes: gdn * buffers.replay_per_layer as u64, ..Default::default() }] }))
    }
    fn accepts(&self, _role: &TensorRole, stem: &str, operand: &mut format::QuantOperand) -> Result<(), String> {
        self.header.accepts(stem, operand)
    }
}

fn native_layout(cfg: &crate::families::qwen3_5::Qwen35Config, input: &crate::plan::PlanInput,
    report: &mut crate::plan::PlanReport, options: &crate::plan::layout::LayoutOptions)
    -> anyhow::Result<cuteafd_core::memory_layout::MemoryLayout> {
    use anyhow::{ensure, Context};
    use crate::placement::{self, families::qwen3_5::{DenseInputs, NativeBuffers, NativeWeights},
        ArchContext, Baseline, GpuBudget, ProgramSet};
    use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};
    ensure!(options.rtx_bytes.len() == 1 && report.spark_ranks == 0,
        "dense Qwen currently requires one coordinator GPU and zero Sparks");
    ensure!(!options.host_embedding, "dense Qwen requires device token embedding");
    ensure!(report.vision == crate::plan::MediaMode::Off && report.audio == crate::plan::MediaMode::Off,
        "dense Qwen media execution is not qualified");
    ensure!(options.native_mtp_layers == 0 && options.drafter_bytes == 0,
        "dense native planner currently requires speculation off");
    let prefix = match (cfg.hidden, cfg.intermediate, cfg.heads, cfg.gdn.value_heads) {
        (4096, 12288, 16, 32) => "qwen35", (5120, 17408, 24, 48) => "qwen38",
        _ => anyhow::bail!("dense Qwen geometry lacks native exports"),
    };
    let path = options.workspace_manifest.as_deref().unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(path)
        .context("dense native admission requires its PROGRAMS.json manifest")?)?;
    let set = ProgramSet::from_manifest(&manifest, &[prefix]);
    let entries = manifest["programs"].as_array().context("native manifest lacks programs")?;
    let mut scratch = 0;
    let mut norm = None;
    for (op, rows) in [("norm", 0), ("add", 0), ("gdn_commit", 0)].into_iter().chain(
        [64, 4096].into_iter().flat_map(|rows| ["gdn", "attn_producer", "full_gqa", "attn_o", "ffn"].map(|op| (op, rows)))) {
        let name = if rows == 0 { format!("{prefix}_{op}") } else { format!("{prefix}_{op}_m{rows}") };
        let entry = entries.iter().find(|entry| entry["name"] == name)
            .with_context(|| format!("missing native dense program {name}"))?;
        let spec = placement::families::qwen3_5::NativeProgram::from_manifest(entry)?;
        placement::families::qwen3_5::validate(cfg, op, rows, &spec)?;
        placement::families::qwen3_5::validate_abi(op, rows, &spec)?;
        scratch = scratch.max(spec.scratch.values().copied().max().unwrap_or(0));
        if op == "gdn" {
            let dtype = placement::families::qwen3_5::norm_dtype(&spec)?;
            ensure!(norm.as_ref().is_none_or(|old| *old == dtype), "native GDN norm precision differs across shapes");
            norm = Some(dtype);
        }
    }
    ensure!(scratch > 0 && matches!(options.prefill_rows, 0 | 4096), "dense native prefill requires capacity 4096");
    let slots = usize::try_from(if options.concurrency == 0 { 4 } else { options.concurrency })?;
    ensure!(options.state_slots.is_none_or(|n| n == slots as u64), "dense state slots must equal concurrency");
    let mark = NativeBuffers::new(cfg, scratch as usize, slots, 0)?.mark_bytes;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(slots as u64,
        options.mimo_prefix_entries, mark as u64, options.mimo_prefix_mark_bytes, options.host_cache_bytes != Some(0)));
    let buffers = NativeBuffers::new(cfg, scratch as usize, slots, usize::try_from(marks)?)?;
    let weights = NativeWeights::bf16(input.checkpoint(), cfg, norm.context("missing GDN precision")?)?;
    let (layers, fixed) = weights.demands(cfg, &buffers, slots);
    let capacity = options.rtx_bytes[0];
    let context = ArchContext::coordinator(capacity, None).context_bytes + set.module_bytes();
    let result = placement::families::qwen3_5::placement(DenseInputs { cfg,
        gpu: GpuBudget { capacity_bytes: capacity, headroom_bytes: 1 << 30,
            baseline: Baseline::Planned { context_bytes: context, loaded_bytes: 0 } },
        layers: &layers, fixed, drafter: None, requested_pool: options.pool_tokens.filter(|&n| n > 0),
        max_context: cfg.max_context as u64, record_bytes: (2 * cfg.kv_heads * cfg.head_dim * 2) as u64,
        pool_overhead: 0, pool_bounded: vec![] })?;
    let mut items = result.items[0].clone();
    items.push(Item::new(Category::Runtime, "context and native modules", "native", context, Basis::Calibrated));
    Ok(MemoryLayout { devices: vec![DeviceLayout { kind: DeviceKind::Rtx, index: 0,
        capacity_bytes: capacity - (1 << 30), items, kv_tokens: result.pool_tokens }],
        pool_tokens: result.pool_tokens, waste: vec![], notes: vec![
            "Native dense admission: BF16 KV, exact GDN state, no captured graphs; text-only single GPU".into()] })
}
