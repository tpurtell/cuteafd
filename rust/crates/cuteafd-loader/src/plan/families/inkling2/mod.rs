//! Inkling 2 Small (`Inkling2ForConditionalGeneration`, `inkling2`): GQA with a
//! relative-position bias, per-token sinks, an output gate, L2 clips and short
//! K/V convolutions on every layer (40 sliding-window, 8 global), softmax
//! residual mixes over six-layer blocks, a sigmoid-routed SiTU MoE over fused
//! `w13`/`w2` expert tensors (MXFP8, native NVFP4 or EXL3 K4 by checkpoint and
//! layer), a bundled dSpark drafter (`mtp.safetensors`) and a bundled ViT.
//! The planner validates the checkpoint's own config and tensor headers; the
//! per-layer expert format is read from each layer's headers, never from
//! `hf_quant_config.json` (#1930).
use std::collections::BTreeMap;

use anyhow::{ensure, Result};

use crate::families::inkling2::{Inkling2Config, Inkling2MtpConfig, Inkling2VisionConfig};
use crate::plan::{checkpoint::Checkpoint, family::*, format::{self, QuantOperand, ScaleEncoding}, names::indexed, spec::*};

#[cfg(test)]
mod tests;

pub struct Inkling2;
pub static INKLING2: Inkling2 = Inkling2;

pub const TEXT: &str = "model.llm.";
pub const VISION: &str = "model.visual.";

/// How the release stores one logical weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Storage {
    Bf16,
    F32,
    /// Routed experts: fused `[E, N, K]` tensors whose format each layer's
    /// headers decide (MXFP8, native NVFP4 or EXL3 K4).
    Experts,
}

#[derive(Debug, Clone)]
pub(crate) struct Contract {
    pub role: TensorRole,
    /// The logical shape (`[E, N, K]` for fused experts).
    pub shape: Vec<usize>,
    pub storage: Storage,
    /// Member suffixes beside the weight (`bias`), in the stem-grouping sense.
    pub companions: Vec<&'static str>,
    /// The stem's lone member is `.scale`, not `.weight` (the L2-clip scales).
    pub scale_member: bool,
}

/// Per-layer routed-expert storage, read from the layer's own headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertStorage {
    /// E4M3 with UE8M0 scales per row and 32 K columns; MXFP8 activations.
    Mxfp8,
    /// Packed E2M1 with E4M3 per-16 scales, an F32 per-expert amax and the
    /// outer denominator; per-token amax activations (fouroversix).
    Nvfp4,
    /// EXL3 trellis per expert (`gate_proj`/`up_proj`/`down_proj`).
    Exl3 { bits: usize },
}

pub(super) struct Model {
    pub(crate) cfg: Inkling2Config,
    spec: ModelSpec,
    /// Logical weights by stem.
    contracts: BTreeMap<String, Contract>,
    /// Checkpoint bytes of every routed-expert tensor (headers only).
    pub(crate) routed_bytes: u64,
    /// Each routed layer's storage, from its own `w13` headers.
    pub(crate) expert_storage: Vec<ExpertStorage>,
}

mod placement;

fn text(name: &str) -> Option<&str> { name.strip_prefix(TEXT) }

/// The fused expert tensor names of one layer and projection (`w13` or `w2`).
pub fn expert_stem(layer: usize, projection: &str) -> String {
    format!("{TEXT}layers.{layer}.mlp.experts.{projection}")
}

impl Model {
    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = Inkling2Config::from_hf(&checkpoint.config)?;
        let vision = Inkling2VisionConfig::from_hf(&checkpoint.config)?;
        let mtp = Inkling2MtpConfig::from_hf(&checkpoint.config, &cfg)?;
        let (h, v, d) = (cfg.hidden, cfg.vocab_size, cfg.head_dim);
        let mut contracts = BTreeMap::new();
        let mut add = |stem: String, role: TensorRole, shape: Vec<usize>, storage| {
            let scale_member = stem.ends_with("_l2_clip");
            contracts.insert(stem, Contract { role, shape, storage, companions: Vec::new(), scale_member });
        };
        add(format!("{TEXT}embed"), TensorRole::new(Component::Embedding), vec![v, h], Storage::Bf16);
        add(format!("{TEXT}unembed"), TensorRole::new(Component::LmHead), vec![v, h], Storage::Bf16);
        for norm in ["embed_norm", "norm"] {
            add(format!("{TEXT}{norm}"), TensorRole::new(Component::Norm), vec![h], Storage::Bf16);
        }
        add(format!("{TEXT}attn_res.output_query"), TensorRole::new(Component::ResidualMix), vec![h], Storage::F32);
        for layer in 0..cfg.layers {
            let base = format!("{TEXT}layers.{layer}");
            let at = |c| TensorRole::layer(c, layer);
            let local = cfg.local[layer];
            let a = format!("{base}.attn");
            for (proj, n, k) in [("wq_du", cfg.heads * d, h), ("wr_du", cfg.heads * cfg.d_rel, h),
                ("wk_dv", cfg.kv_heads * d, h), ("wv_dv", cfg.kv_heads * d, h), ("gate_attn", cfg.heads * d, h),
                ("attn_sink_proj", cfg.heads, h), ("wo_ud", h, cfg.heads * d)] {
                add(format!("{a}.{proj}"), at(Component::Attention), vec![n, k], Storage::Bf16);
            }
            add(format!("{a}.q_l2_clip"), at(Component::Attention), vec![cfg.heads, d], Storage::Bf16);
            add(format!("{a}.rel_l2_clip"), at(Component::Attention), vec![cfg.heads, cfg.d_rel], Storage::Bf16);
            add(format!("{a}.rel_logits_proj.proj"), at(Component::Attention),
                vec![cfg.d_rel, if local { cfg.window } else { cfg.rel_extent }], Storage::Bf16);
            for name in ["k", "v"] {
                let stem = format!("{a}.{name}_sconv");
                if local {
                    add(stem, at(Component::Attention), vec![cfg.kv_heads * d, 1, cfg.sconv_kernel], Storage::Bf16);
                } else {
                    add(format!("{stem}.static_w"), at(Component::Attention), vec![cfg.sconv_kernel, cfg.kv_heads * d], Storage::Bf16);
                    add(format!("{stem}.weight_proj"), at(Component::Attention), vec![cfg.kv_heads * cfg.sconv_kernel, h], Storage::Bf16);
                }
            }
            for name in ["attn_sconv", "mlp_sconv"] {
                add(format!("{base}.{name}"), at(Component::Attention), vec![h, 1, cfg.sconv_kernel], Storage::Bf16);
            }
            let norms: &[&str] = if local { &["attn_norm"] } else { &["q_input_norm", "k_input_norm", "v_input_norm"] };
            for norm in norms.iter().chain(["mlp_norm"].iter()) {
                add(format!("{base}.{norm}"), at(Component::Norm), vec![h], Storage::Bf16);
            }
            add(format!("{base}.mlp.gate"), at(Component::Router), vec![cfg.experts, h], Storage::Bf16);
            add(expert_stem(layer, "w13"), TensorRole::layer(Component::RoutedExpert, layer),
                vec![cfg.experts, 2 * cfg.intermediate, h], Storage::Experts);
            add(expert_stem(layer, "w2"), TensorRole::layer(Component::RoutedExpert, layer),
                vec![cfg.experts, h, cfg.intermediate], Storage::Experts);
            add(format!("{TEXT}attn_res.mlp_queries.{layer}"), at(Component::ResidualMix), vec![h], Storage::F32);
            if layer > 0 {
                add(format!("{TEXT}attn_res.attn_queries.{}", layer - 1), at(Component::ResidualMix),
                    if local { vec![h] } else { vec![3, h] }, Storage::F32);
            }
        }
        if let Some(vc) = &vision {
            let role = TensorRole::new(Component::Vision);
            let (vh, vi) = (vc.hidden, vc.intermediate);
            let tower = format!("{VISION}backbone");
            add(format!("{tower}.embeddings.patch_embedding"), role.clone(), vec![vh, 3 * vc.patch * vc.patch], Storage::Bf16);
            add(format!("{tower}.embeddings.position_embedding"), role.clone(), vec![vc.posemb_grid * vc.posemb_grid, vh], Storage::Bf16);
            for name in ["class_embedding", "cls_position_embedding"] {
                add(format!("{tower}.embeddings.{name}"), role.clone(), vec![vh], Storage::Bf16);
            }
            add(format!("{tower}.ln_pre"), role.clone(), vec![vh], Storage::Bf16);
            for layer in 0..vc.layers {
                let base = format!("{tower}.layers.{layer}");
                for norm in ["ln_1", "ln_2"] { add(format!("{base}.{norm}"), role.clone(), vec![vh], Storage::Bf16); }
                for scale in ["ls_1", "ls_2"] { add(format!("{base}.{scale}.gamma"), role.clone(), vec![vh], Storage::Bf16); }
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    add(format!("{base}.self_attn.{proj}"), role.clone(), vec![vh, vh], Storage::Bf16);
                }
                add(format!("{base}.mlp.c_fc"), role.clone(), vec![vi, vh], Storage::Bf16);
                add(format!("{base}.mlp.c_proj"), role.clone(), vec![vh, vi], Storage::Bf16);
            }
            add(format!("{VISION}proj_in"), role.clone(), vec![vc.projector_hidden, vc.merge * vc.merge * vh], Storage::Bf16);
            add(format!("{VISION}proj_out"), role, vec![h, vc.projector_hidden], Storage::Bf16);
        }
        drop(add);
        for layer in 0..cfg.layers {
            // The router's F32 selection bias is a `bias` companion of the gate weight.
            contracts.get_mut(&format!("{TEXT}layers.{layer}.mlp.gate")).expect("router contract").companions.push("bias");
        }
        let routed_bytes = checkpoint.tensors.iter()
            .filter(|tensor| tensor.meta.name.contains(".mlp.experts."))
            .map(|tensor| tensor.meta.byte_length).sum();
        let expert_storage = (0..cfg.layers).map(|layer| expert_storage(checkpoint, layer)).collect::<Result<Vec<_>>>()?;
        let model = Self { spec: spec(&cfg, vision.is_some(), mtp.as_ref(), &expert_storage, checkpoint), cfg, contracts,
            routed_bytes, expert_storage };
        // Open is structural: every contract member must be present by name, so
        // a missing tensor is a typed open error that names it. Exact dtype and
        // shape agreement and unknown tensors are `accepts` verdicts, reported
        // per tensor in the census (`--files` keeps working on a drifted quant).
        let tower_present = checkpoint.tensor_names().any(|name| name.starts_with(VISION));
        for (stem, contract) in &model.contracts {
            // The bundled tower is optional: a text-only role (or a text-only
            // publication) omits every `model.visual.*` tensor together.
            if contract.role.component == Component::Vision && !tower_present { continue; }
            match contract.storage {
                Storage::Experts => {
                    let layer = contract.role.layer.expect("routed experts are per layer");
                    let present = match model.expert_storage[layer] {
                        ExpertStorage::Exl3 { .. } => {
                            let projection = if stem.ends_with(".w2") { "down_proj" } else { "gate_proj" };
                            checkpoint.contains_tensor(&format!("{TEXT}layers.{layer}.mlp.experts.0.{projection}.trellis"))
                        }
                        _ => checkpoint.contains_tensor(&format!("{stem}_weight")),
                    };
                    ensure!(present, "Inkling 2 checkpoint lacks {stem} (routed_expert)");
                }
                _ => {
                    let bare = stem.ends_with(".proj") || stem.ends_with(".static_w") || stem.ends_with(".weight_proj")
                        || stem.ends_with(".gamma") || stem.contains(".attn_res.") || stem.ends_with("_embedding");
                    let name = if contract.scale_member { format!("{stem}.scale") } else if bare { stem.clone() }
                        else { format!("{stem}.weight") };
                    ensure!(checkpoint.contains_tensor(&name), "Inkling 2 checkpoint lacks {name} ({:?})", contract.role.component);
                    for companion in &contract.companions {
                        let name = format!("{stem}.{companion}");
                        ensure!(checkpoint.contains_tensor(&name), "Inkling 2 checkpoint lacks {name} ({:?})", contract.role.component);
                    }
                }
            }
        }
        Ok(model)
    }

    fn contract(&self, stem: &str) -> Option<&Contract> {
        self.contracts.get(stem).or_else(|| stem.strip_suffix(".weight").and_then(|s| self.contracts.get(s)))
    }

    /// Routed layers in order (every Inkling 2 layer is routed).
    pub(crate) fn routed(&self) -> Vec<usize> { (0..self.cfg.layers).collect() }
}

/// A layer's routed-expert storage from its own headers: `w13_weight_amax`
/// marks native NVFP4, a per-expert EXL3 trellis marks K-bit experts, and the
/// E4M3 fused tensor is MXFP8. The quantization metadata is never consulted.
pub fn expert_storage(checkpoint: &Checkpoint, layer: usize) -> Result<ExpertStorage> {
    let stem = expert_stem(layer, "w13");
    if let Some(header) = checkpoint.header(&format!("{TEXT}layers.{layer}.mlp.experts.0.gate_proj.trellis")) {
        let bits = header.shape.get(2).map(|rows| rows / 16).filter(|bits| (1..=8).contains(bits));
        return bits.map(|bits| ExpertStorage::Exl3 { bits })
            .ok_or_else(|| anyhow::anyhow!("{}: EXL3 trellis shape {:?} encodes no width", header.name, header.shape));
    }
    let weight = checkpoint.header(&format!("{stem}_weight"))
        .ok_or_else(|| anyhow::anyhow!("Inkling 2 checkpoint lacks {stem}_weight (routed_expert)"))?;
    Ok(match weight.dtype {
        cuteafd_core::DType::F8E4M3 => ExpertStorage::Mxfp8,
        cuteafd_core::DType::U8 if checkpoint.contains_tensor(&format!("{stem}_weight_amax")) => ExpertStorage::Nvfp4,
        ref dtype => anyhow::bail!("{stem}_weight: unsupported Inkling 2 expert storage {dtype:?}"),
    })
}

fn spec(cfg: &Inkling2Config, vision: bool, mtp: Option<&Inkling2MtpConfig>, storage: &[ExpertStorage],
    checkpoint: &Checkpoint) -> ModelSpec {
    let layers = (0..cfg.layers).map(|layer| LayerSpec {
        attention: if cfg.local[layer] {
            AttentionKind::SlidingGqa { heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim,
                window: cfg.window, sinks: true }
        } else {
            AttentionKind::Gqa { heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim }
        },
        ffn: FfnKind::Moe,
        rope: None,
    }).collect();
    let mut formats = BTreeMap::<String, Vec<usize>>::new();
    for (layer, storage) in storage.iter().enumerate() {
        let name = match storage {
            ExpertStorage::Mxfp8 => "MXFP8 W8A8".to_string(),
            ExpertStorage::Nvfp4 => "native NVFP4 W4A4 (per-token amax, fouroversix)".to_string(),
            ExpertStorage::Exl3 { bits } => format!("EXL3 K{bits}"),
        };
        formats.entry(name).or_default().push(layer);
    }
    let mut notes = vec![
        format!("attention: GQA {}/{} x {} on every layer; {} sliding-window-{} layers and {} global layers {:?} with \
            relative-position bias (d_rel {}, extent {}), per-token per-head sinks, sigmoid output gate, q/k L2 clip {:.3}, \
            r L2 clip {}, short K/V convolutions (kernel {}; dynamic per-token taps on global layers)",
            cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.local_layers(), cfg.window, cfg.global_layers().len(),
            cfg.global_layers(), cfg.d_rel, cfg.rel_extent, cfg.qk_l2_clip, cfg.rel_l2_clip, cfg.sconv_kernel),
        format!("residuals: softmax mixes over {}-layer blocks (F32 queries); per-layer attn_sconv/mlp_sconv", cfg.res_block),
        format!("MoE: sigmoid + bias top-{} of {}, log-sigmoid softmax weights x {}; SiTU (gate softcap {}, up softcap {}); \
            fused interleaved w13 (even rows gate, odd rows up); route weight before the down-projection input quantization",
            cfg.topk, cfg.experts, cfg.route_scale, cfg.gate_softcap, cfg.up_softcap),
        format!("routed expert storage by layer: {}", formats.iter().map(|(name, layers)| {
            let (first, last) = (layers[0], layers[layers.len() - 1]);
            if layers.len() == last - first + 1 { format!("{name} layers {first}..={last}") } else { format!("{name} layers {layers:?}") }
        }).collect::<Vec<_>>().join("; ")),
        format!("head: logits trimmed to {} of {} stored rows, softcap {}", cfg.unpadded_vocab_size, cfg.vocab_size, cfg.final_softcap),
    ];
    if let Some((floor, alpha)) = cfg.log_scaling {
        notes.push(format!("global layers scale q and r by 1 + {alpha} ln(max(1, (pos + 1) / {floor}))"));
    }
    if let Some(mtp) = mtp {
        notes.push(format!("bundled dSpark drafter (mtp.safetensors): {} dense layers (local {:?}, window {}), block {}, \
            rnn_hidden Markov head rank {}, target feature taps {:?}", mtp.layers, mtp.local.iter().enumerate()
            .filter(|(_, l)| **l).map(|(i, _)| i).collect::<Vec<_>>(), mtp.window, mtp.block, mtp.markov_rank, mtp.target_layers));
    }
    ModelSpec {
        family: "inkling2",
        architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
        hidden: cfg.hidden,
        vocab: cfg.vocab_size,
        layers,
        moe: Some(MoeSpec {
            experts: cfg.experts,
            top_k: cfg.topk,
            intermediate: cfg.intermediate,
            shared_experts: 0,
            shared_intermediate: 0,
            scoring: "sigmoid + bias for selection; log-sigmoid softmax weights".into(),
            routed_scaling: Some(cfg.route_scale),
            groups: None,
        }),
        speculator: mtp.map(|m| SpeculatorSpec::BundledDspark { layers: m.layers }),
        tables: Vec::new(),
        vision,
        notes,
    }
}

impl Family for Inkling2 {
    fn decode_lanes(&self) -> crate::plan::decode_lanes::LaneSupport {
        crate::plan::decode_lanes::LaneSupport::One {
            reason: "decode shares the one Spark link with every prefill wave; a second lane needs its own decode \
                link, decode workspaces and drafter scratch, and no two-lane A/B exists yet (#1930)" }
    }
    fn spark_decode(&self) -> Option<crate::plan::spark_executions::SparkDecode> {
        Some(crate::plan::spark_executions::SparkDecode { rows: 64 })
    }
    fn id(&self) -> &'static str { "inkling2" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Planned }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|a| a == "Inkling2ForConditionalGeneration")
            || checkpoint.model_type() == Some("inkling2")
    }
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        Ok(Box::new(Model::open(checkpoint).map_err(ConfigError::from_anyhow)?))
    }
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        if name.starts_with(VISION) { return Some(TensorRole::new(Component::Vision)); }
        if name.starts_with("mtp.") { return Some(TensorRole::new(Component::Speculator)); }
        let rest = text(name)?;
        if rest.starts_with("embed.") || rest.starts_with("embed_norm.") {
            return Some(TensorRole::new(if rest.starts_with("embed.") { Component::Embedding } else { Component::Norm }));
        }
        if rest.starts_with("unembed.") { return Some(TensorRole::new(Component::LmHead)); }
        if rest.starts_with("norm.") { return Some(TensorRole::new(Component::Norm)); }
        if let Some(query) = rest.strip_prefix("attn_res.") {
            if query == "output_query" { return Some(TensorRole::new(Component::ResidualMix)); }
            let (kind, index) = query.split_once('.')?;
            let index: usize = index.parse().ok()?;
            // attn_queries.{i} feeds layer i + 1; mlp_queries.{i} feeds layer i.
            let layer = if kind == "attn_queries" { index + 1 } else if kind == "mlp_queries" { index } else { return None };
            return (layer < spec.layers.len()).then(|| TensorRole::layer(Component::ResidualMix, layer));
        }
        let (layer, rest) = indexed(rest, "layers.")?;
        if layer >= spec.layers.len() { return None; }
        if let Some(expert) = rest.strip_prefix("mlp.experts.") {
            // Fused `w13_weight*` / `w2_weight*` tensors, or per-expert EXL3 modules.
            if let Some((expert, _)) = expert.split_once('.').filter(|(e, _)| e.parse::<usize>().is_ok()) {
                return Some(TensorRole::expert(Component::RoutedExpert, layer, expert.parse().ok()?));
            }
            return Some(TensorRole::layer(Component::RoutedExpert, layer));
        }
        let component = if rest.starts_with("mlp.gate.") { Component::Router }
            else if rest.starts_with("attn.") || rest.starts_with("attn_sconv.") || rest.starts_with("mlp_sconv.") { Component::Attention }
            else if rest.ends_with("_norm.weight") || rest == "mlp_norm.weight" || rest == "attn_norm.weight" { Component::Norm }
            else { return None; };
        Some(TensorRole::layer(component, layer))
    }
    fn component_hint(&self, component: Component) -> Option<Hint> {
        let how = match component {
            Component::Attention => "Export the biased-sink GQA tile (per-token per-head sinks, distance-indexed additive bias, window 1024), the short-conv (static and dynamic taps), L2-clip and output-gate kernels; see #1930 measure-first notes.",
            Component::ResidualMix => "Generalize the Kimi attn_res softmax mix to hidden 4608 (#1934 S2).",
            Component::RoutedExpert => "Export routed MXFP8 W8A8 and native NVFP4 W4A4 (per-token amax, fouroversix) experts with SiTU and route-weight-before-FC2 quantization, and register the EXL3 K4 geometry (4608, 2304, 384, 8, situ); see #1930 kernel notes.",
            Component::Vision => "Wire the bundled PE-Lang ViT (2-D rotary, layer scale) and 2x2 merge projector through shared media admission with their own arena.",
            Component::Speculator => "Wire the bundled dSpark (block 8, rnn_hidden Markov head with f_rnn and head fold) through shared speculation; generalize DraftHeadOps::RnnMarkov (#1934 S4).",
            _ => "Add the Inkling 2 serving hooks over the shared scheduler, fixed-depth packed prefill and bounded graph cache; admit named owners and packed arenas through the shared recipe.",
        };
        Some(Hint { what: format!("Inkling 2 {} execution is not implemented in this build", component.label()), how: how.into() })
    }
    fn optional(&self, component: Component) -> bool {
        matches!(component, Component::Vision | Component::Speculator)
    }
    /// The shared expert catalog gains fused `w13`/`w2` readers with the Spark
    /// packages (stage 3); the stage-1 planner reads expert headers directly.
    fn expert_catalog(&self) -> bool { false }
}

impl FamilyModel for Model {
    fn activation(&self, _layer: usize, _shared: bool) -> Option<cuteafd_core::FfnActivation> {
        Some(self.cfg.activation())
    }
    fn spec(&self) -> &ModelSpec { &self.spec }
    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        // Per-expert EXL3 modules carry the expert id in their stem; the
        // contract is keyed by the fused per-layer stem.
        let (lookup, exl3_expert) = match role.expert {
            Some(_) if role.component == Component::RoutedExpert => {
                let layer = role.layer.ok_or_else(|| format!("{stem}: routed expert without a layer"))?;
                let projection = super::leaf(stem);
                let fused = match projection { "gate_proj" | "up_proj" => "w13", "down_proj" => "w2",
                    other => return Err(format!("{stem}: unknown Inkling 2 expert projection {other}")) };
                (expert_stem(layer, fused), Some(projection))
            }
            _ => (stem.to_owned(), None),
        };
        let contract = self.contract(&lookup)
            .ok_or_else(|| format!("{stem}: outside the Inkling 2 tensor contract ({})", super::describe(operand)))?;
        super::require(contract.role.component == role.component && contract.role.layer == role.layer, ||
            format!("{stem}: classified {:?}, the contract says {:?}", role.component, contract.role.component))?;
        let companions = operand.companions.iter().map(String::as_str).collect::<Vec<_>>();
        match contract.storage {
            Storage::Bf16 => super::require(operand.logical == contract.shape && operand.is_plain(&[format::Encoding::Bf16])
                && operand.input_scale.is_none() && companions == contract.companions,
                || format!("{stem}: expected BF16 {:?}{}, found {}", contract.shape,
                    if contract.companions.is_empty() { String::new() } else { format!(" with {:?}", contract.companions) },
                    super::describe(operand))),
            Storage::F32 => super::require(operand.logical == contract.shape && operand.is_plain(&[format::Encoding::F32])
                && operand.input_scale.is_none() && companions == contract.companions,
                || format!("{stem}: expected F32 {:?}, found {}", contract.shape, super::describe(operand))),
            Storage::Experts => {
                let layer = contract.role.layer.expect("routed experts are per layer");
                let [experts, n, k] = contract.shape[..] else { unreachable!("fused expert contract is [E, N, K]") };
                match (self.expert_storage[layer], exl3_expert) {
                    (ExpertStorage::Mxfp8, None) => super::require(operand.logical == contract.shape
                        && operand.is_fp8_row_groups(32, &[ScaleEncoding::Ue8m0]) && operand.global_scale.is_none()
                        && operand.input_scale.is_none(),
                        || format!("{stem}: expected MXFP8 E4M3 [{experts}, {n}, {k}] with UE8M0 scales per row and 32 columns, found {}",
                            super::describe(operand))),
                    (ExpertStorage::Nvfp4, None) => super::require(operand.logical == contract.shape
                        && operand.is_inkling_nvfp4(),
                        || format!("{stem}: expected native NVFP4 [{experts}, {n}, {k}]: packed E2M1 U8, E4M3 per-16 scales, \
                            F32 per-expert amax and outer denominator, found {}", super::describe(operand))),
                    (ExpertStorage::Exl3 { bits }, Some(projection)) => {
                        let rows = if projection == "down_proj" { n } else { n / 2 };
                        super::require(operand.exl3_bits() == Some(bits) && operand.logical == vec![rows, k],
                            || format!("{stem}: expected EXL3 K{bits} [{rows}, {k}] (interleaved w13 split by row parity), found {}",
                                super::describe(operand)))
                    }
                    (ExpertStorage::Exl3 { .. }, None) => Err(format!("{stem}: layer {layer} stores EXL3 experts per expert; a fused tensor is outside the contract")),
                    (_, Some(_)) => Err(format!("{stem}: layer {layer} stores fused experts; a per-expert module is outside the contract")),
                }
            }
        }
    }
    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        let (package, block) = if operand.is_fp8_row_groups(32, &[ScaleEncoding::Ue8m0]) {
            ("inkling2:mxfp8", 32)
        } else if operand.is_inkling_nvfp4() {
            ("inkling2:nvfp4a4", 16)
        } else if operand.exl3_bits().is_some_and(|bits| bits == 4) {
            ("inkling2:exl3-k4", 128)
        } else { return None; };
        Some(ExpertContract {
            package: package.into(), block,
            spark_worlds: crate::plan::experts::spark_worlds(package, self.cfg.intermediate),
            // Stage 1 plans whole and TP2 RTX layers in the shared recipe; the
            // executors (stage 3) and serve stay unqualified until their kernels land.
            local: Ok(format!("Inkling 2 RTX expert onboarding ({package} whole/TP2 package)")),
        })
    }
    fn contract_admission(&self) -> bool { true }
    fn placement(&self, input: &crate::plan::PlanInput, report: &mut crate::plan::PlanReport,
        options: &mut crate::plan::layout::LayoutOptions, requested: crate::plan::ExpertPlacement)
        -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        Some(Ok(placement::layout(self, input, report, options, requested)))
    }
}
