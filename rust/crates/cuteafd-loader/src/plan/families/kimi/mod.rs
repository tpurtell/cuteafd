//! Kimi K3 (`KimiK3ForConditionalGeneration`, text `kimi_linear`): KDA + NoPE
//! MLA hybrid with attention-residual blocks, latent MoE and Situ FFNs. The
//! planner validates the checkpoint's own config and tensor headers and names
//! each missing kernel; no execution path exists in this build yet.
use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use cuteafd_core::{DType, FfnActivation};

use crate::families::kimi_k3::config::{KimiAttention, KimiK3Config};
use crate::plan::{checkpoint::Checkpoint, family::*, format::QuantOperand, names::indexed, spec::*};

mod placement;
#[cfg(test)]
mod tests;

pub struct KimiK3;
pub static KIMI_K3: KimiK3 = KimiK3;

const TEXT: &str = "language_model.model.";
const HEAD: &str = "language_model.lm_head.";
const GIB: u64 = 1 << 30;

/// Routed-expert weights per Spark rank for Kimi K3 without `--spark-budget-gib`.
///
/// TP6 EXL3 K2 needs 107.54 GiB per rank (92 layers x 896 experts: each rank
/// holds 1/6 of the trellis, the 3072-wide rotation slice, and full copies of
/// the 3584-wide input/output rotations). 108 GiB is that plus 0.46 GiB margin,
/// equal to the planner's usable GB10 bytes (121.7 GiB physical minus ~13 GiB
/// host OS and storage daemon). Rank runtime and I/O stay separate charges on
/// the device layout. The shared 100 GiB default would put ~8 routed layers
/// (55 GiB) on the coordinators and leave no room for the 256K KV floor.
pub const SPARK_BUDGET_BYTES: u64 = 108 * GIB;

/// The EXL3 expert package Kimi K3 uses: H3584/E896/K16/I3072, uniform K2 in
/// the K2/K3 tier set, TP6 width 512.
pub const PACKAGE: &str = "kimik3:exl3-k23";

#[derive(Clone)]
struct Tensor {
    shape: Vec<usize>,
    dtype: DType,
}

pub(super) struct Model {
    cfg: KimiK3Config,
    spec: ModelSpec,
    tensors: BTreeMap<String, Tensor>,
    /// Header byte counts by component.
    bytes: BTreeMap<Component, u64>,
    /// Every expected text tensor has a readable header.
    complete: bool,
    /// Routed formats observed (`exl3-k2`, ...).
    expert_formats: Vec<String>,
    /// Coordinator text bytes by TP2 residency (embedding and head excluded).
    coordinator: Residency,
}

/// Coordinator text weights by how a two-GPU head split holds them.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Residency {
    /// Head/column/intermediate-sharded: half on each GPU.
    pub(super) split: u64,
    /// Whole on each GPU: low-rank A projections, routers, latent projections,
    /// norms and residual-mix vectors.
    pub(super) replicated: u64,
}

/// Whether a coordinator tensor (`layers.N.` remainder) is whole on each GPU of a head split.
pub(super) fn replicated(rest: &str) -> bool {
    !rest.starts_with("self_attn.") && !rest.starts_with("block_sparse_moe.shared_experts.") && !rest.starts_with("mlp.")
        || ["self_attn.f_a_proj.", "self_attn.o_norm.", "self_attn.q_a_proj.", "self_attn.q_a_layernorm.",
            "self_attn.kv_a_proj_with_mqa.", "self_attn.kv_a_layernorm."].iter().any(|p| rest.starts_with(p))
}

/// Strip the text prefix: `language_model.model.layers.3.x` -> `layers.3.x`.
fn text(name: &str) -> Option<&str> { name.strip_prefix(TEXT) }

impl Model {
    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let (h, v, d) = (cfg.hidden, cfg.vocab_size, cfg.kda_heads * cfg.kda_head_dim);
        let mut tensors = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>, dtype| { tensors.insert(name, Tensor { shape, dtype }); };
        add(format!("{TEXT}embed_tokens.weight"), vec![v, h], DType::Bf16);
        add(format!("{HEAD}weight"), vec![v, h], DType::Bf16);
        for name in ["norm", "output_attn_res_norm"] { add(format!("{TEXT}{name}.weight"), vec![h], DType::Bf16); }
        add(format!("{TEXT}output_attn_res_proj.weight"), vec![1, h], DType::Bf16);
        let heads = cfg.heads;
        let q_head = cfg.qk_nope_dim + cfg.qk_rope_dim;
        for layer in 0..cfg.layers {
            let base = format!("{TEXT}layers.{layer}");
            for norm in ["input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"] {
                add(format!("{base}.{norm}.weight"), vec![h], DType::Bf16);
            }
            for proj in ["self_attention_res_proj", "mlp_res_proj"] { add(format!("{base}.{proj}.weight"), vec![1, h], DType::Bf16); }
            let a = format!("{base}.self_attn");
            add(format!("{a}.g_proj.weight"), vec![heads * cfg.v_head_dim, h], DType::Bf16);
            add(format!("{a}.o_proj.weight"), vec![h, heads * cfg.v_head_dim], DType::Bf16);
            match cfg.attention[layer] {
                KimiAttention::Kda => {
                    for proj in ["q", "k", "v"] {
                        add(format!("{a}.{proj}_proj.weight"), vec![d, h], DType::Bf16);
                        add(format!("{a}.{proj}_conv1d.weight"), vec![d, 1, cfg.kda_conv], DType::F32);
                    }
                    add(format!("{a}.f_a_proj.weight"), vec![cfg.kda_head_dim, h], DType::Bf16);
                    add(format!("{a}.f_b_proj.weight"), vec![d, cfg.kda_head_dim], DType::Bf16);
                    add(format!("{a}.b_proj.weight"), vec![cfg.kda_heads, h], DType::Bf16);
                    // Zero-padded to 128 for 96 heads; the loader narrows exact +0 tails only.
                    add(format!("{a}.A_log"), vec![cfg.kda_heads.next_power_of_two()], DType::F32);
                    add(format!("{a}.dt_bias"), vec![d], DType::F32);
                    add(format!("{a}.o_norm.weight"), vec![cfg.kda_head_dim], DType::F32);
                }
                KimiAttention::Mla => {
                    add(format!("{a}.q_a_proj.weight"), vec![cfg.q_lora_rank, h], DType::Bf16);
                    add(format!("{a}.q_a_layernorm.weight"), vec![cfg.q_lora_rank], DType::Bf16);
                    add(format!("{a}.q_b_proj.weight"), vec![heads * q_head, cfg.q_lora_rank], DType::Bf16);
                    add(format!("{a}.kv_a_proj_with_mqa.weight"), vec![cfg.kv_lora_rank + cfg.qk_rope_dim, h], DType::Bf16);
                    add(format!("{a}.kv_a_layernorm.weight"), vec![cfg.kv_lora_rank], DType::Bf16);
                    add(format!("{a}.kv_b_proj.weight"), vec![heads * (cfg.qk_nope_dim + cfg.v_head_dim), cfg.kv_lora_rank], DType::Bf16);
                }
            }
            if cfg.dense[layer] {
                for (proj, n, k) in [("gate", cfg.dense_intermediate, h), ("up", cfg.dense_intermediate, h), ("down", h, cfg.dense_intermediate)] {
                    add(format!("{base}.mlp.{proj}_proj.weight"), vec![n, k], DType::Bf16);
                }
            } else {
                let m = format!("{base}.block_sparse_moe");
                add(format!("{m}.gate.weight"), vec![cfg.experts, h], DType::Bf16);
                add(format!("{m}.gate.e_score_correction_bias"), vec![cfg.experts], DType::F32);
                add(format!("{m}.routed_expert_down_proj.weight"), vec![cfg.latent, h], DType::Bf16);
                add(format!("{m}.routed_expert_up_proj.weight"), vec![h, cfg.latent], DType::Bf16);
                add(format!("{m}.routed_expert_norm.weight"), vec![cfg.latent], DType::Bf16);
                let shared = cfg.moe_intermediate * cfg.shared_experts;
                for (proj, n, k) in [("gate", shared, h), ("up", shared, h), ("down", h, shared)] {
                    add(format!("{m}.shared_experts.{proj}_proj.weight"), vec![n, k], DType::Bf16);
                }
            }
        }
        let spec = spec(&cfg, checkpoint);
        let mut model = Self { cfg, spec, tensors, bytes: BTreeMap::new(), complete: true, expert_formats: Vec::new(),
            coordinator: Residency::default() };
        let mut headers = BTreeMap::new();
        for tensor in &checkpoint.tensors { headers.insert(tensor.meta.name.as_str(), &tensor.meta); }
        for (name, expected) in &model.tensors {
            let Some(meta) = headers.get(name.as_str()) else {
                ensure!(checkpoint.contains_tensor(name), "Kimi K3 checkpoint lacks {name}");
                model.complete = false;
                continue;
            };
            ensure!(meta.shape == expected.shape, "{name}: header shape {:?}, the release contract is {:?}", meta.shape, expected.shape);
            ensure!(meta.dtype == expected.dtype, "{name}: header dtype {:?}, the release contract is {:?}", meta.dtype, expected.dtype);
        }
        for tensor in &checkpoint.tensors {
            if let Some(role) = KIMI_K3.classify(&model.spec, &tensor.meta.name) {
                *model.bytes.entry(role.component).or_default() += tensor.meta.byte_length;
                if let Some((_, rest)) = text(&tensor.meta.name).and_then(|r| indexed(r, "layers."))
                    .filter(|_| role.component != Component::RoutedExpert) {
                    let bucket = if replicated(rest) { &mut model.coordinator.replicated } else { &mut model.coordinator.split };
                    *bucket += tensor.meta.byte_length;
                } else if matches!(role.component, Component::Norm | Component::ResidualMix) {
                    model.coordinator.replicated += tensor.meta.byte_length;
                }
            }
        }
        let mut formats = std::collections::BTreeSet::new();
        for tensor in &checkpoint.tensors {
            if let Some(bits) = tensor.meta.name.strip_suffix(".trellis").filter(|n| n.contains(".block_sparse_moe.experts."))
                .and(tensor.meta.shape.get(2)).map(|packed| packed / 16) {
                formats.insert(format!("exl3-k{bits}"));
            }
        }
        model.expert_formats = formats.into_iter().collect();
        Ok(model)
    }

    pub(super) fn cfg(&self) -> &KimiK3Config { &self.cfg }
}

fn spec(cfg: &KimiK3Config, checkpoint: &Checkpoint) -> ModelSpec {
    let layers = (0..cfg.layers).map(|layer| LayerSpec {
        attention: match cfg.attention[layer] {
            KimiAttention::Kda => AttentionKind::Kda,
            KimiAttention::Mla => AttentionKind::Mla { rope: false, output_gate: true },
        },
        ffn: if cfg.dense[layer] { FfnKind::Dense { intermediate: cfg.dense_intermediate } } else { FfnKind::Moe },
        rope: None,
    }).collect();
    ModelSpec {
        family: "kimi_k3",
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
            scoring: "sigmoid".into(),
            routed_scaling: Some(cfg.routed_scale),
            groups: None,
        }),
        speculator: None,
        tables: Vec::new(),
        vision: checkpoint.config.get("vision_config").is_some(),
        notes: vec![
            format!("latent MoE: experts see routed_expert_down_proj {}->{}, combine, RMSNorm({}), up_proj back",
                cfg.hidden, cfg.latent, cfg.latent),
            format!("attention residuals: softmax mix over stored block residuals, a block every {} layers ({} max)",
                cfg.residual_block, cfg.max_residual_blocks()),
            format!("Situ FFNs (beta {}, linear beta {}); MLA without RoPE plus sigmoid output gate; KDA full-rank output gate",
                cfg.situ_beta, cfg.situ_linear_beta),
            "no native MTP; external K3 DFlash/dSpark drafters exist".into(),
        ],
    }
}

impl Family for KimiK3 {
    fn id(&self) -> &'static str { "kimi_k3" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Planned }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|a| a == "KimiK3ForConditionalGeneration")
            || checkpoint.model_type() == Some("kimi_k3")
    }
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        Ok(Box::new(Model::open(checkpoint).map_err(ConfigError::from_anyhow)?))
    }
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        if name.starts_with("vision_tower.") || name.starts_with("mm_projector.") {
            return Some(TensorRole::new(Component::Vision));
        }
        if name.starts_with(HEAD) { return Some(TensorRole::new(Component::LmHead)); }
        let rest = text(name)?;
        if rest.starts_with("embed_tokens.") { return Some(TensorRole::new(Component::Embedding)); }
        if rest.starts_with("norm.") { return Some(TensorRole::new(Component::Norm)); }
        if rest.starts_with("output_attn_res_") { return Some(TensorRole::new(Component::ResidualMix)); }
        let (layer, rest) = indexed(rest, "layers.")?;
        if layer >= spec.layers.len() { return None; }
        if let Some(expert) = rest.strip_prefix("block_sparse_moe.experts.")
            .and_then(|r| r.split_once('.')).and_then(|(e, _)| e.parse().ok()) {
            return Some(TensorRole::expert(Component::RoutedExpert, layer, expert));
        }
        let component = if rest.starts_with("block_sparse_moe.gate.") { Component::Router }
            else if rest.starts_with("block_sparse_moe.shared_experts.") { Component::SharedExpert }
            else if rest.starts_with("block_sparse_moe.routed_expert_") { Component::LatentProjection }
            else if rest.starts_with("mlp.") { Component::DenseFfn }
            else if rest.starts_with("self_attn.") { Component::Attention }
            else if rest.starts_with("self_attention_res_") || rest.starts_with("mlp_res_") { Component::ResidualMix }
            else if rest.starts_with("input_layernorm.") || rest.starts_with("post_attention_layernorm.") { Component::Norm }
            else { return None; };
        Some(TensorRole::layer(component, layer))
    }
    fn component_hint(&self, component: Component) -> Option<Hint> {
        // Work items as posted on #273.
        let (what, how) = match component {
            Component::RoutedExpert => ("EXL3 K2 latent experts (H3584 E896 K16 I3072, Situ)",
                "mixed_trellis expert ids above 512 (10-bit descriptors), Situ epilogue through compile_mixed_trellis, \
                 package_exl3_aot.py GEOMETRIES (3584,3072,896,16) TP6 width 512; reducer width 3584 (W4/W5)"),
            Component::Router => ("sigmoid noaux_tc router at E896/K16, scale 1.0",
                "expert_router.cu caps E512/K8 and glm routes fix 2.5; add an FP32 E896 top-16 variant (W5)"),
            Component::Attention => ("KDA 96x128 at hidden 7168 with a full-rank output gate; NoPE full-causal MLA 96 heads with sigmoid output gate",
                "generalize glmf KDA programs (hidden 7168, 48 heads/GPU, g_proj in w_in, F32 conv/o_norm, A_log narrowed before split) (W6); \
                 AOT-export b12x dense_mla (576/512 latent, scale 192^-0.5, unrotated rope slot), no-RoPE producer, gate before o_proj (W7)"),
            Component::ResidualMix => ("attention-residual mixer (not mHC)",
                "shared FP32 rmsnorm-score softmax mix over up to 9 BF16 candidates, fused with the following RMSNorm; block store scratch (W8)"),
            Component::LatentProjection => ("latent MoE projections and post-combine RMSNorm(3584)",
                "coordinator order: down_proj -> dispatch -> reduce at 3584 -> RMSNorm -> up_proj -> + shared (W9)"),
            Component::DenseFfn | Component::SharedExpert => ("Situ dense/shared FFN",
                "BF16 FFN programs with the Situ epilogue (FfnActivation::Situ) (W8)"),
            Component::Vision => ("MoonViT3d tower + patchmergerv2 projector", "later step; text first"),
            Component::Embedding | Component::LmHead | Component::Norm => ("Kimi K3 engine", "families/kimi_k3 engine (W9)"),
            _ => return None,
        };
        Some(Hint { what: format!("Kimi K3 {}: {what} missing", component.label()), how: how.into() })
    }
}

impl FamilyModel for Model {
    fn spec(&self) -> &ModelSpec { &self.spec }
    fn spark_budget_bytes(&self) -> u64 { SPARK_BUDGET_BYTES }
    fn activation(&self, layer: usize, _shared: bool) -> Option<FfnActivation> {
        (layer < self.cfg.layers).then(|| self.cfg.activation())
    }
    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        let known = self.tensors.contains_key(stem) || self.tensors.contains_key(&format!("{stem}.weight"))
            || (role.component == Component::RoutedExpert && operand.exl3_bits() == Some(2))
            || role.component == Component::Vision;
        if !known {
            return Err(format!("{stem}: not in the Kimi K3 release tensor contract ({})", super::describe(operand)));
        }
        let hint = KIMI_K3.component_hint(role.component)
            .unwrap_or(Hint { what: "Kimi K3 engine missing".into(), how: "families/kimi_k3 (W9)".into() });
        Err(format!("{stem}: {}: {}", hint.what, hint.how))
    }
    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        (operand.exl3_bits() == Some(2)).then(|| ExpertContract {
            package: PACKAGE.into(),
            block: 128,
            spark_worlds: crate::plan::experts::spark_worlds(PACKAGE, self.cfg.moe_intermediate),
            local: Err("Kimi K3 routed experts (637 GiB) cannot all be resident on the coordinators".into()),
        })
    }
    fn placement(&self, _input: &crate::plan::PlanInput, report: &mut crate::plan::PlanReport,
        options: &crate::plan::layout::LayoutOptions, requested: crate::plan::ExpertPlacement)
        -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        Some(Ok(placement::layout(self, report, options, requested)))
    }
}
