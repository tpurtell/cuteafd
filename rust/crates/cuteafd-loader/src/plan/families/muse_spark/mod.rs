//! Muse Spark (`MuseSparkForConditionalGeneration`, text `muse_spark_text`):
//! GQA with 3 sliding (RoPE, window 2048) : 1 full (NoPE) layers, an output
//! gate and residual gates; every layer is a 256-expert top-8 MoE whose experts
//! run at the 4096-wide `pre_expert_proj` stream and normalize each expert's
//! output before its routing weight. The planner reads only the checkpoint's
//! own config, quantization metadata and tensor headers.
//!
//! Routed experts are served from an EXL3 publication (K4 experts split per
//! expert, K6 MCG core), intermediate-sliced across Spark ranks like every
//! EXL3 family. The per-expert output norm needs each expert's complete
//! output, so ranks reply per-expert partials (BF16 or NVFP4 by the
//! per-request row cutoff) and the coordinator sums, norms and weights them.
use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use cuteafd_core::{DType, FfnActivation};

use crate::families::muse_spark::{core::CoreInventory, MuseAttention, MuseSparkConfig};
use crate::plan::{checkpoint::Checkpoint, family::*, format::{self, QuantOperand}, names::indexed, spec::*};

mod placement;
#[cfg(test)]
mod tests;

pub struct MuseSpark;
pub static MUSE_SPARK: MuseSpark = MuseSpark;

const TEXT: &str = "model.language_model.";
const HEAD: &str = "lm_head.";

/// The EXL3 expert package: whole H4096/I4096 experts (256, top-8), K4 with
/// the adjacent empty K5 tier, partitioned across ranks (`ep<world>`).
pub const PACKAGE: &str = "muse:exl3-k45";

#[derive(Clone)]
struct Tensor {
    shape: Vec<usize>,
    dtype: DType,
}

pub(super) struct Model {
    cfg: MuseSparkConfig,
    spec: ModelSpec,
    /// Non-core, non-expert text tensors and their release contract.
    tensors: BTreeMap<String, Tensor>,
    /// Header byte counts by component.
    bytes: BTreeMap<Component, u64>,
    /// Every expected text tensor has a readable header.
    complete: bool,
    /// Routed formats observed (`exl3-k4`, `bf16-fused`, ...).
    expert_formats: Vec<String>,
    core: CoreInventory,
}

fn text(name: &str) -> Option<&str> { name.strip_prefix(TEXT) }

impl Model {
    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let (h, v, e) = (cfg.hidden, cfg.vocab_size, cfg.expert_hidden);
        let (q, kv) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        let mut tensors = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>, dtype| { tensors.insert(name, Tensor { shape, dtype }); };
        add(format!("{TEXT}embed_tokens.weight"), vec![v, h], DType::Bf16);
        add(format!("{TEXT}norm.weight"), vec![h], DType::Bf16);
        add(format!("{HEAD}weight"), vec![v, h], DType::Bf16);
        for layer in 0..cfg.layers {
            let b = format!("{TEXT}layers.{layer}");
            for norm in ["input_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                add(format!("{b}.{norm}.weight"), vec![h], DType::Bf16);
            }
            for gate in ["post_attention_residual_gate", "post_feedforward_residual_gate"] {
                add(format!("{b}.{gate}.gate"), vec![h], DType::Bf16);
            }
            for (proj, n, k) in [("q_proj", q, h), ("k_proj", kv, h), ("v_proj", kv, h), ("o_proj", h, q), ("gate_proj", q, h)] {
                add(format!("{b}.self_attn.{proj}.weight"), vec![n, k], DType::Bf16);
            }
            add(format!("{b}.mlp.pre_expert_proj.weight"), vec![e, h], DType::Bf16);
            add(format!("{b}.mlp.post_expert_proj.weight"), vec![h, e], DType::Bf16);
            add(format!("{b}.mlp.pre_expert_norm.weight"), vec![e], DType::Bf16);
            add(format!("{b}.mlp.experts.post_expert_norm.weight"), vec![e], DType::Bf16);
            // The official module keeps the router and its bias in FP32.
            add(format!("{b}.mlp.gate.weight"), vec![cfg.experts, h], DType::F32);
            add(format!("{b}.mlp.gate.e_score_correction_bias"), vec![cfg.experts], DType::F32);
        }
        let spec = spec(&cfg, checkpoint);
        let core = CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        let mut headers = BTreeMap::new();
        for tensor in &checkpoint.tensors { headers.insert(tensor.meta.name.as_str(), &tensor.meta); }
        let mut complete = checkpoint.missing_shards.is_empty();
        for (name, expected) in &tensors {
            if core.get_weight(name).is_some() { continue; }
            let Some(meta) = headers.get(name.as_str()) else {
                ensure!(checkpoint.contains_tensor(name), "Muse Spark checkpoint lacks {name}");
                complete = false;
                continue;
            };
            ensure!(meta.shape == expected.shape, "{name}: header shape {:?}, the release contract is {:?}", meta.shape, expected.shape);
            ensure!(meta.dtype == expected.dtype, "{name}: header dtype {:?}, the release contract is {:?}", meta.dtype, expected.dtype);
        }
        let mut bytes = BTreeMap::new();
        let mut formats = std::collections::BTreeSet::new();
        for tensor in &checkpoint.tensors {
            if let Some(role) = MUSE_SPARK.classify(&spec, &tensor.meta.name) {
                *bytes.entry(role.component).or_default() += tensor.meta.byte_length;
            }
            let name = tensor.meta.name.as_str();
            if let Some(stem) = name.strip_suffix(".trellis").filter(|n| n.contains(".mlp.experts.")) {
                let bits = tensor.meta.shape.get(2).map_or(0, |packed| packed / 16);
                ensure!(stem.ends_with("_proj"), "{name}: unexpected routed trellis tensor");
                formats.insert(format!("exl3-k{bits}"));
            } else if name.ends_with(".mlp.experts.gate_up_proj") || name.ends_with(".mlp.experts.down_proj") {
                formats.insert("bf16-fused".to_string());
            }
        }
        Ok(Self { cfg, spec, tensors, bytes, complete, expert_formats: formats.into_iter().collect(), core })
    }

    pub(super) fn cfg(&self) -> &MuseSparkConfig { &self.cfg }
}

fn spec(cfg: &MuseSparkConfig, checkpoint: &Checkpoint) -> ModelSpec {
    let layers = (0..cfg.layers).map(|layer| match cfg.attention[layer] {
        MuseAttention::Full => LayerSpec {
            attention: AttentionKind::Gqa { heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim },
            ffn: FfnKind::Moe, rope: None },
        MuseAttention::Sliding => LayerSpec {
            attention: AttentionKind::SlidingGqa { heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim,
                window: cfg.window, sinks: false },
            ffn: FfnKind::Moe, rope: Some(RopeSpec { dims: cfg.head_dim, theta: cfg.rope_theta }) },
    }).collect();
    ModelSpec {
        family: "muse_spark",
        architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
        hidden: cfg.hidden,
        vocab: cfg.vocab_size,
        layers,
        moe: Some(MoeSpec {
            experts: cfg.experts,
            top_k: cfg.topk,
            intermediate: cfg.moe_intermediate,
            shared_experts: 0,
            shared_intermediate: 0,
            scoring: "sigmoid of FP32 logits x output_multiplier; bias selects only; renormalize (eps 1e-15)".into(),
            routed_scaling: Some(cfg.routed_scale),
            groups: None,
        }),
        speculator: None,
        tables: Vec::new(),
        vision: checkpoint.config.get("vision_config").is_some(),
        notes: vec![
            format!("{} layers: {} sliding GQA (window {}, RoPE {}) + {} full GQA (NoPE); {} Q / {} KV heads x {}; \
                q/k/o weightless per-head RMSNorm, q x {:.4}; sigmoid output gate", cfg.layers, cfg.sliding_layers(),
                cfg.window, cfg.rope_theta, cfg.full_layers(), cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.qk_scale),
            format!("MoE every layer: pre_expert_proj {}->{} + centered RMSNorm, {} experts top-{} (I {}), \
                per-expert input-scaled RMSNorm before the routing weight, post_expert_proj back; no shared expert",
                cfg.hidden, cfg.expert_hidden, cfg.experts, cfg.topk, cfg.moe_intermediate),
            format!("residual gates (temperature {}); head {} * tanh(logits * {:.4} / {})", cfg.residual_gate_temperature,
                cfg.softcap, cfg.output_multiplier, cfg.softcap),
            "Spark experts: intermediate slices (TP); per-expert partials (BF16 at decode/verify, NVFP4 for prefill \
                segments) normed and combined on the coordinator".into(),
            "no native MTP; external DFlash2 drafters (nvidia, RadixArk)".into(),
        ],
    }
}

impl Family for MuseSpark {
    fn decode_lanes(&self) -> crate::plan::decode_lanes::LaneSupport {
        crate::plan::decode_lanes::LaneSupport::One {
            reason: "new family on the shared lane structure; a second lane waits for a measured two-lane A/B (#744)" }
    }
    fn spark_decode(&self) -> Option<crate::plan::spark_executions::SparkDecode> {
        // Decode/verify steps send at most 64 rows, as Kimi and Step-5.
        Some(crate::plan::spark_executions::SparkDecode { rows: 64 })
    }
    fn id(&self) -> &'static str { "muse_spark" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Planned }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        MuseSparkConfig::is_muse_spark(&checkpoint.config)
    }
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        Ok(Box::new(Model::open(checkpoint).map_err(ConfigError::from_anyhow)?))
    }
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        if name.starts_with("model.vision_tower.") || name.starts_with("model.vision_adapter.")
            || name.starts_with("model.vision_projection.") {
            return Some(TensorRole::new(Component::Vision));
        }
        if name.starts_with("model.audio_tower.") { return Some(TensorRole::new(Component::Audio)); }
        if name.starts_with(HEAD) { return Some(TensorRole::new(Component::LmHead)); }
        let rest = text(name)?;
        if rest.starts_with("embed_tokens.") { return Some(TensorRole::new(Component::Embedding)); }
        if rest.starts_with("norm.") { return Some(TensorRole::new(Component::Norm)); }
        let (layer, rest) = indexed(rest, "layers.")?;
        if layer >= spec.layers.len() { return None; }
        if let Some(tail) = rest.strip_prefix("mlp.experts.") {
            // Split EXL3 experts carry their id; the fused BF16 stacks and the
            // per-expert output norm belong to every expert of the layer.
            return Some(match tail.split_once('.').and_then(|(e, _)| e.parse().ok()) {
                Some(expert) => TensorRole::expert(Component::RoutedExpert, layer, expert),
                None => TensorRole::layer(Component::RoutedExpert, layer),
            });
        }
        let component = if rest.starts_with("mlp.gate.") { Component::Router }
            else if rest.starts_with("mlp.") { Component::LatentProjection }
            else if rest.starts_with("self_attn.") { Component::Attention }
            else if rest.ends_with("_residual_gate.gate") { Component::ResidualMix }
            else if rest.ends_with("layernorm.weight") { Component::Norm }
            else { return None; };
        Some(TensorRole::layer(component, layer))
    }
    fn component_hint(&self, _component: Component) -> Option<Hint> { None }
    fn component_hint_for(&self, _spec: &ModelSpec, component: Component, formats: &[String]) -> Option<Hint> {
        // An EXL3 publication's routed component also carries the BF16 post-expert norm.
        if component == Component::RoutedExpert && !formats.iter().any(|f| f.starts_with("exl3")) {
            return Some(Hint {
                what: "Muse Spark routed experts need the EXL3 publication (K4 experts split per expert + K6 core)".into(),
                how: "The BF16 release fuses each layer's 256 experts into one tensor with no expert package; \
                    quantize with GPTQModel's EXL3 K4 + Core-K6 recipe and serve that snapshot".into(),
            });
        }
        self.component_hint(component)
    }
    fn optional(&self, component: Component) -> bool {
        matches!(component, Component::Vision | Component::Audio)
    }
    fn expert_catalog(&self) -> bool { true }
}

impl FamilyModel for Model {
    fn spec(&self) -> &ModelSpec { &self.spec }
    fn native_serving_admission(&self) -> bool { true }
    fn cache_geometry(&self, options: crate::serving_capacity::CacheOptions)
        -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        crate::families::muse_spark::cache::MuseCacheLayout::new(&self.cfg, options.coordinator_ranks)
            .map(|layout| Some(layout.geometry))
    }
    fn activation(&self, layer: usize, _shared: bool) -> Option<FfnActivation> {
        (layer < self.cfg.layers).then_some(FfnActivation::SiluBf16Product { limit: None })
    }
    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        if matches!(role.component, Component::Vision | Component::Audio) {
            let shapes = if role.component == Component::Vision {
                crate::media::muse_vision::tensor_shapes()
            } else { crate::media::pcm_audio::tensor_shapes().into_iter()
                .map(|(name, shape)| (format!("model.audio_tower.{name}"), shape)).collect() };
            let shape = shapes.iter().find(|(name, _)| name.strip_suffix(".weight").unwrap_or(name) == stem)
                .map(|(_, shape)| shape).ok_or_else(|| format!("{stem}: outside bundled Muse tower contract"))?;
            return super::require(operand.is_plain(&[format::Encoding::Bf16]) && operand.logical == *shape
                && operand.input_scale.is_none() && operand.companions.iter().all(|name| name == "bias"), ||
                format!("{stem}: bundled Muse tower needs plain BF16 {shape:?}, found {}", super::describe(operand)));
        }
        if let Some(projection) = self.core.projections.get(stem) {
            super::require(matches!(operand.encoding, format::Encoding::Exl3 { bits: 6 })
                && operand.logical == [projection.stored_outputs, projection.shape[1]],
                || format!("{stem}: expected header-native K6 {:?} (stored N {})", projection.shape, projection.stored_outputs))?;
            // Report the logical extent; the head's stored N128 padding stays the loader's.
            operand.logical = projection.shape.to_vec();
            return Ok(());
        }
        if role.component == Component::RoutedExpert && role.expert.is_some() {
            let expected = match super::leaf(stem) {
                "gate_proj" | "up_proj" => [self.cfg.moe_intermediate, self.cfg.expert_hidden],
                "down_proj" => [self.cfg.expert_hidden, self.cfg.moe_intermediate],
                _ => return Err(format!("{stem}: unknown Muse Spark routed projection")),
            };
            return super::require(operand.exl3_bits().is_some_and(|bits| (4..=5).contains(&bits))
                && operand.logical == expected, ||
                format!("{stem}: Muse Spark experts run EXL3 K4/K5 {expected:?}, found {}", super::describe(operand)));
        }
        if stem.ends_with(".mlp.experts.gate_up_proj") || stem.ends_with(".mlp.experts.down_proj") {
            return Err(format!("{stem}: fused BF16 expert stacks have no expert package; serve the EXL3 publication"));
        }
        let expected = self.tensors.get(stem).or_else(|| self.tensors.get(&format!("{stem}.weight")))
            .ok_or_else(|| format!("{stem}: outside the Muse Spark text tensor contract ({})", super::describe(operand)))?;
        let encoding = match expected.dtype {
            DType::Bf16 => format::Encoding::Bf16,
            DType::F32 => format::Encoding::F32,
            _ => return Err(format!("{stem}: unsupported Muse Spark text dtype {:?}", expected.dtype)),
        };
        super::require(operand.logical == expected.shape && operand.is_plain(&[encoding])
            && operand.input_scale.is_none() && operand.companions.is_empty(), ||
            format!("{stem}: expected plain {:?} {:?}, found {}", expected.dtype, expected.shape, super::describe(operand)))
    }
    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        // The per-expert output norm (BF16 [moe_hidden]) ships with the package.
        let norm = operand.is_plain(&[format::Encoding::Bf16]) && operand.logical == [self.cfg.expert_hidden];
        (operand.exl3_bits().is_some_and(|bits| (4..=5).contains(&bits)) || norm).then(|| ExpertContract {
            package: PACKAGE.into(),
            block: 128,
            spark_worlds: crate::plan::experts::spark_worlds(PACKAGE, self.cfg.moe_intermediate),
            local: Err("Muse Spark routed experts (~400 GB at K4) cannot all be resident on the coordinators".into()),
        })
    }
    fn placement(&self, input: &crate::plan::PlanInput, report: &mut crate::plan::PlanReport,
        options: &mut crate::plan::layout::LayoutOptions, requested: crate::plan::ExpertPlacement)
        -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        Some(Ok(placement::layout(self, input, report, options, requested)))
    }
}
