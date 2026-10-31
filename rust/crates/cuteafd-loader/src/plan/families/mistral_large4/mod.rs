//! Mistral Large 4 (`MistralLarge4ForConditionalGeneration`, text
//! `mistral_large4_text`): MLA without q-LoRA plus scalable softmax, a DSA
//! indexer on `full` layers, NoPE every fourth layer, cross-layer latent KV
//! sharing, softmax top-4 of 128 routed experts plus one shared expert, a
//! soft-capped untied head. Mistral consolidated tensor names
//! (`layers.N.attention.wq`, `layers.N.experts.E.w1`, `output`), read from
//! `config.json:transformers_weights` (`consolidated.safetensors.index.json`).
//!
//! Two storage forms pass the tensor contract: the official FP8 block-128
//! release (the quantization source and fidelity root, inventory only) and
//! the EXL3 K4 experts + K6 core publication the engine serves. The contract
//! is exact: every text tensor's name, shape and form comes from the
//! checkpoint's own config, so a quant that drifts from the recipe is named.
use std::collections::BTreeMap;

use anyhow::{ensure, Result};

use super::{describe, require};
use crate::families::mistral_large4::{Ml4Config, Ml4Indexer};
use crate::plan::checkpoint::Checkpoint;
use crate::plan::family::{ConfigError, ExpertContract, Family, FamilyModel, Hint, RuntimeStatus};
use crate::plan::format::{Encoding, QuantOperand, ScaleEncoding};
use crate::plan::names::indexed;
use crate::plan::spec::*;

pub struct MistralLarge4;
pub static MISTRAL_LARGE4: MistralLarge4 = MistralLarge4;

/// The Spark expert package: H6144/E128/K4/I6144, EXL3 in the K4/K5 tier pair.
pub const PACKAGE: &str = "ml4:exl3-k45";

/// Rows of one decode/verify Spark request (the shared decode lane rows).
pub const DECODE_ROWS: u32 = 64;

/// How one non-routed tensor may be stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// Plain BF16 (norm weights, SSMax vectors, router, BF16 projections).
    Bf16,
    /// BF16 LayerNorm weight with its BF16 bias (the indexer's norms).
    LayerNorm,
    /// K6 MCG core, or the official E4M3 with FP32 128x128 block scales.
    CoreOrFp8,
    /// K6 MCG core (`pad`: rows may be stored padded to 128), or official BF16.
    CoreOrBf16 { pad: bool },
}

#[derive(Debug, Clone)]
struct Expected {
    shape: Vec<usize>,
    form: Form,
}

/// Every non-routed text tensor stem the runtime reads, with its contract.
fn expected(cfg: &Ml4Config) -> BTreeMap<String, Expected> {
    let mut out = BTreeMap::new();
    let mut add = |stem: String, shape: Vec<usize>, form| { out.insert(stem, Expected { shape, form }); };
    let h = cfg.hidden;
    add("tok_embeddings".into(), vec![cfg.vocab_size, h], Form::Bf16);
    add("norm".into(), vec![h], Form::Bf16);
    add("lerope.alpha".into(), vec![cfg.qk_rope_head_dim / 2], Form::Bf16);
    for (stem, [n, k]) in cfg.projection_shapes() {
        let leaf = stem.rsplit('.').next().unwrap_or_default();
        let form = match leaf {
            "output" => Form::CoreOrBf16 { pad: false },
            "wkv_a_with_mqa" => Form::CoreOrBf16 { pad: true },
            // Absorbed MLA splits wkv_b into w_uk / w_uv from BF16 rows; BF16 in
            // the source (modules_to_not_convert), kept by the K6-core recipe.
            "wkv_b" | "wk" | "weights_proj" | "gate" => Form::Bf16,
            _ => Form::CoreOrFp8,
        };
        add(stem, vec![n, k], form);
    }
    for layer in 0..cfg.layers {
        let l = format!("layers.{layer}");
        add(format!("{l}.attention_norm"), vec![h], Form::Bf16);
        add(format!("{l}.ffn_norm"), vec![h], Form::Bf16);
        for p in ["s", "b"] {
            add(format!("{l}.attention.ssmax_scaling.{p}"), vec![cfg.heads], Form::Bf16);
        }
        if cfg.owns_kv(layer) {
            add(format!("{l}.attention.kv_a_norm"), vec![cfg.kv_lora_rank], Form::Bf16);
        }
        if cfg.indexers[layer] == Ml4Indexer::Full {
            let i = format!("{l}.attention.indexer");
            add(format!("{i}.k_norm"), vec![cfg.index_head_dim], Form::LayerNorm);
            add(format!("{i}.q_input_norm"), vec![h], Form::LayerNorm);
            for p in ["s", "b"] {
                add(format!("{i}.ssmax_scaling.{p}"), vec![cfg.index_heads], Form::Bf16);
            }
        }
    }
    out
}

/// The routed projection of `stem` (`w1`/`w3` gate/up, `w2` down) and its `[out, in]`.
fn routed_shape(cfg: &Ml4Config, stem: &str) -> Option<[usize; 2]> {
    match stem.rsplit('.').next()? {
        "w1" | "w3" => Some([cfg.moe_intermediate, cfg.hidden]),
        "w2" => Some([cfg.hidden, cfg.moe_intermediate]),
        _ => None,
    }
}

impl Family for MistralLarge4 {
    fn id(&self) -> &'static str { "mistral_large4" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Planned }
    fn decode_lanes(&self) -> crate::plan::decode_lanes::LaneSupport {
        crate::plan::decode_lanes::LaneSupport::One {
            reason: "one lane until a two-lane A/B wins: top-4 of 128 wide (6144) experts makes each Spark \
                decode wave short beside the 72-layer MLA core, little round trip to hide (predicted, #744)" }
    }
    fn spark_decode(&self) -> Option<crate::plan::spark_executions::SparkDecode> {
        Some(crate::plan::spark_executions::SparkDecode { rows: DECODE_ROWS })
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|a| a == "MistralLarge4ForConditionalGeneration")
            || checkpoint.model_type() == Some("mistral_large4")
    }
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        Ok(Box::new(Model::open(checkpoint).map_err(ConfigError::from_anyhow)?))
    }
    fn expert_catalog(&self) -> bool { true }
    fn optional(&self, component: Component) -> bool {
        // The bundled dSpark drafter and the Pixtral tower are not required for text.
        matches!(component, Component::Speculator | Component::Vision)
    }
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        use Component::*;
        if name.starts_with("mtp_dspark.") { return Some(TensorRole::new(Speculator)); }
        if ["vision_encoder.", "patch_merger.", "pre_mm_projector_norm.", "vision_language_adapter."]
            .iter().any(|p| name.starts_with(p)) {
            return Some(TensorRole::new(Vision));
        }
        if name.starts_with("tok_embeddings.") { return Some(TensorRole::new(Embedding)); }
        if name.starts_with("output.") { return Some(TensorRole::new(LmHead)); }
        if name.starts_with("norm.") { return Some(TensorRole::new(Norm)); }
        if name == "lerope.alpha" { return Some(TensorRole::new(Attention)); }
        let (layer, rest) = indexed(name, "layers.")?;
        if layer >= spec.layers.len() { return None; }
        if let Some((expert, _)) = indexed(rest, "experts.") {
            return Some(TensorRole::expert(RoutedExpert, layer, expert));
        }
        let component = if rest.starts_with("attention.indexer.") { Indexer }
            else if rest.starts_with("attention.") { Attention }
            else if rest.starts_with("gate.") { Router }
            else if rest.starts_with("shared_experts.") { SharedExpert }
            else if rest.starts_with("feed_forward.") { DenseFfn }
            else if rest.starts_with("attention_norm.") || rest.starts_with("ffn_norm.") { Norm }
            else { return None; };
        Some(TensorRole::layer(component, layer))
    }
    fn component_hint(&self, component: Component) -> Option<Hint> {
        let (what, how) = match component {
            Component::Attention | Component::Indexer => (
                "MLA (no q-LoRA) with scalable softmax, DSA indexer on full layers, NoPE every fourth layer, \
                 latent KV shared from layer 35 by layers 39..71",
                "families/mistral_large4 on the shared engine: GLM 5.3's absorbed MLA + DSA indexer programs at \
                 head 128 (NoPE: latent-only records), SSMax folded into the query, shared indexer layers reuse \
                 the previous full layer's top-k, shared-KV layers read the source layer's pages.",
            ),
            Component::RoutedExpert => (
                "softmax top-4 of 128 routed experts, H6144 I6144, SiLU",
                "EXL3 K4/K5 package ml4:exl3-k45 (python/tools/aot/package_exl3_aot.py geometry ml4); the \
                 official FP8 release is the quantization source only.",
            ),
            Component::Speculator => (
                "bundled mtp_dspark drafter: 3 dense non-causal MLA layers, RNN Markov head",
                "Wire through the shared dSpark path (target taps 1,18,36,54,71 from params.json mtp).",
            ),
            Component::Vision => (
                "Pixtral-style vision encoder + patch merger",
                "Serve text with --vision off until the bundled tower is ported.",
            ),
            _ => return None,
        };
        Some(Hint { what: what.into(), how: how.into() })
    }
}

pub(crate) struct Model {
    cfg: Ml4Config,
    spec: ModelSpec,
    expected: BTreeMap<String, Expected>,
}

impl Model {
    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = Ml4Config::from_hf(&checkpoint.config)?;
        let expected = expected(&cfg);
        // The index names every runtime tensor; its storage form is the
        // tensor contract's verdict (`accepts`), which names recipe drift.
        for (stem, e) in &expected {
            let present = if stem.contains("ssmax_scaling.") || stem == "lerope.alpha" {
                checkpoint.contains_tensor(stem)
            } else {
                checkpoint.contains_tensor(&format!("{stem}.weight")) || checkpoint.contains_tensor(&format!("{stem}.trellis"))
            };
            ensure!(present, "Mistral Large 4 checkpoint lacks {stem}");
            if e.form == Form::LayerNorm {
                ensure!(checkpoint.contains_tensor(&format!("{stem}.bias")), "Mistral Large 4 checkpoint lacks {stem}.bias");
            }
        }
        let spec = spec(&cfg, checkpoint);
        Ok(Self { cfg, spec, expected })
    }

    fn routed(&self, stem: &str, operand: &QuantOperand) -> Result<(), String> {
        let shape = routed_shape(&self.cfg, stem).ok_or("not a routed projection (w1/w3/w2)")?;
        require(operand.logical == shape, || format!("experts are {shape:?}, found {}", describe(operand)))?;
        require(operand.is_fp8_block(128, &[ScaleEncoding::F32])
            || operand.exl3_bits().is_some_and(|bits| (4..=5).contains(&bits)), || format!(
            "routed experts run EXL3 K4/K5 (served) or the official E4M3 with FP32 128x128 scales (source); found {}",
            describe(operand)))
    }
}

fn spec(cfg: &Ml4Config, checkpoint: &Checkpoint) -> ModelSpec {
    let layers = (0..cfg.layers).map(|layer| LayerSpec {
        attention: AttentionKind::MlaDsa { indexer: cfg.indexers[layer] == Ml4Indexer::Full },
        ffn: if layer < cfg.first_moe_layer { FfnKind::Dense { intermediate: cfg.dense_intermediate } } else { FfnKind::Moe },
        rope: cfg.rope[layer].then_some(RopeSpec { dims: cfg.qk_rope_head_dim, theta: cfg.rope_scaling.theta }),
    }).collect();
    let nope = (0..cfg.layers).filter(|&l| !cfg.rope[l]).count();
    let readers: Vec<String> = (0..cfg.layers).filter(|&l| !cfg.owns_kv(l))
        .map(|l| format!("{l}<-{}", cfg.kv_sources[l])).collect();
    ModelSpec {
        family: "mistral_large4",
        architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
        hidden: cfg.hidden,
        vocab: cfg.vocab_size,
        layers,
        moe: Some(MoeSpec {
            experts: cfg.experts,
            top_k: cfg.topk,
            intermediate: cfg.moe_intermediate,
            shared_experts: cfg.shared_experts,
            shared_intermediate: cfg.moe_intermediate * cfg.shared_experts,
            scoring: "softmax".into(),
            routed_scaling: Some(cfg.routed_scale),
            groups: None,
        }),
        // The bundled mtp_dspark drafter is not wired yet (optional component).
        speculator: None,
        tables: Vec::new(),
        vision: checkpoint.config.get("vision_config").is_some(),
        notes: vec![
            format!("MLA without q-LoRA: {} heads, q/k head {} (RoPE layers nope {} + rope {}, NoPE layers nope {}), \
                v {}, kv_lora {}; softmax scale {:.6}", cfg.heads, cfg.qk_head_dim(), cfg.qk_nope_head_dim,
                cfg.qk_rope_head_dim, cfg.qk_head_dim(), cfg.v_head_dim, cfg.kv_lora_rank, cfg.softmax_scale()),
            "scalable softmax: q *= s * log(pos + 1) + b per head after RoPE (attention and indexer)".into(),
            format!("NoPE on {nope} layers (every {}th): no RoPE in attention or indexer, latent-only KV records",
                cfg.layers / nope.max(1)),
            format!("latent KV owned by {} layers; readers {}", cfg.kv_owners().count(), readers.join(" ")),
            format!("DSA indexer {}x{} top-{} on {} full layers (query: wq_b(LayerNorm(h)), interleaved RoPE on \
                dims 0..{}); shared layers reuse the previous full layer's top-k",
                cfg.index_heads, cfg.index_head_dim, cfg.index_topk,
                cfg.indexers.iter().filter(|&&i| i == Ml4Indexer::Full).count(), cfg.qk_rope_head_dim),
            format!("interleaved YaRN factor {} over {} tokens, beta {}/{}, no mscale; LeRoPE inv_freq * exp(alpha)",
                cfg.rope_scaling.factor, cfg.rope_scaling.original_max_position, cfg.rope_scaling.beta_fast,
                cfg.rope_scaling.beta_slow),
            format!("softmax top-{} router (FP32 logits, renormalized, scale {}); one shared expert; logits \
                tanh soft-capped at {}", cfg.topk, cfg.routed_scale, cfg.logit_softcap),
        ],
    }
}

impl FamilyModel for Model {
    fn spec(&self) -> &ModelSpec { &self.spec }

    fn vision_mode(&self, mode: crate::plan::MediaMode) -> crate::plan::MediaMode {
        if mode == crate::plan::MediaMode::Auto { crate::plan::MediaMode::Off } else { mode }
    }

    /// Mistral Large 4 runs the GLM 5.3 engine: its adapter over the shared
    /// solver, with the operands the checkpoint's variant names.
    fn placement(&self, input: &crate::plan::PlanInput, report: &mut crate::plan::PlanReport,
        options: &mut crate::plan::layout::LayoutOptions, _requested: crate::plan::ExpertPlacement)
        -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        Some(Ok(crate::plan::layout::glm5::layout(report, self, input, options)))
    }

    fn cache_geometry(&self, options: crate::serving_capacity::CacheOptions)
        -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        crate::serving_capacity::mistral_large4_cache_geometry(&self.cfg, options.coordinator_ranks).map(Some)
    }

    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        match role.component {
            Component::RoutedExpert => return self.routed(stem, operand),
            Component::Speculator => return require(operand.is_plain(&[Encoding::Bf16]), ||
                format!("{stem}: the bundled dSpark drafter runs BF16, found {}", describe(operand))),
            Component::Vision => return require(operand.is_plain(&[Encoding::Bf16]), ||
                format!("{stem}: the Pixtral tower runs BF16, found {}", describe(operand))),
            _ => {}
        }
        let e = self.expected.get(stem)
            .ok_or_else(|| format!("{stem}: outside the Mistral Large 4 text tensor contract ({})", describe(operand)))?;
        let core = |operand: &mut QuantOperand, pad: bool| -> Option<Result<(), String>> {
            (operand.encoding == (Encoding::Exl3 { bits: 6 })).then(|| {
                let padded = pad && operand.logical == [e.shape[0].next_multiple_of(128), e.shape[1]];
                require(operand.exl3_bits() == Some(6) && (operand.logical == e.shape || padded), || format!(
                    "{stem}: native K6 MCG {:?} (rows may pad to 128: {pad}), found {}", e.shape, describe(operand)))?;
                operand.logical = e.shape.clone();
                Ok(())
            })
        };
        match e.form {
            Form::Bf16 => require(operand.is_plain(&[Encoding::Bf16]) && operand.logical == e.shape
                && operand.companions.is_empty(), || format!("{stem}: expected BF16 {:?}, found {}", e.shape, describe(operand))),
            Form::LayerNorm => require(operand.is_plain(&[Encoding::Bf16]) && operand.logical == e.shape
                && operand.companions == ["bias"], || format!("{stem}: expected a BF16 LayerNorm {:?} with bias, found {}",
                e.shape, describe(operand))),
            Form::CoreOrFp8 => core(operand, false).unwrap_or_else(|| require(
                operand.is_fp8_block(128, &[ScaleEncoding::F32]) && operand.logical == e.shape, || format!(
                "{stem}: K6 MCG core or the official E4M3 with FP32 128x128 scales {:?}, found {}", e.shape, describe(operand)))),
            Form::CoreOrBf16 { pad } => core(operand, pad).unwrap_or_else(|| require(
                operand.is_plain(&[Encoding::Bf16]) && operand.logical == e.shape, || format!(
                "{stem}: K6 MCG core or official BF16 {:?}, found {}", e.shape, describe(operand)))),
        }
    }

    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        let moe = self.spec.moe.as_ref()?;
        let geometry = cuteafd_core::ExpertGeometry { hidden: self.spec.hidden as u32, experts: moe.experts as u32,
            topk: moe.top_k as u32, intermediate: moe.intermediate as u32, layers: 0 };
        (geometry.same_shape(&cuteafd_core::ExpertGeometry::MISTRAL_LARGE4)
            && operand.exl3_bits().is_some_and(|bits| (4..=5).contains(&bits))).then(|| ExpertContract {
            package: PACKAGE.into(),
            block: 128,
            spark_worlds: crate::plan::experts::spark_worlds(PACKAGE, moe.intermediate),
            local: Err("Mistral Large 4 routed experts (about 479 GiB at K4) cannot all be resident on the coordinators".into()),
        })
    }
}

#[cfg(test)]
mod tests;
