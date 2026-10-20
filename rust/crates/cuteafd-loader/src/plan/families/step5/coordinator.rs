//! Native source-weight contract, not an executable Step-5 coordinator.
//! Reuses the planner inventory and shared dense header/admission validator.
use super::{Config, Model};
use crate::formats::native_dense::NativeDenseLayout;
use crate::plan::{
    checkpoint::{Checkpoint, CheckpointTensor},
    spec::Component,
};
use anyhow::{ensure, Result};
use cuteafd_core::dense_attention::{
    GqaGeometry, NormWeight, RotaryArithmetic, RouterBiasOrder, SigmoidRouter, SlidingWindow,
};
use cuteafd_core::DType;

#[derive(Debug, Clone)]
pub struct CoordinatorContract {
    pub weights: NativeDenseLayout,
    pub norm_weight: NormWeight,
    pub norms: TargetNormEps,
    pub rope_tables: crate::plan::rope::Tables,
    pub router: SigmoidRouter,
    pub layers: Vec<LayerContract>,
}

/// Official target norms all receive config.rms_norm_eps, including the
/// indexer LayerNorm (whose fused kernel's default is not the model value).
/// Extra MTP enorm/hnorm/shared-head norms remain unqualified and unbound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetNormEps {
    pub input: f32,
    pub post_attention: f32,
    pub attention_q: f32,
    pub attention_k: f32,
    pub indexer_q: f32,
    pub indexer_k: f32,
    pub final_norm: f32,
}

impl TargetNormEps {
    fn from_config(cfg: &Config) -> Self {
        Self {
            input: cfg.norm_eps,
            post_attention: cfg.norm_eps,
            attention_q: cfg.norm_eps,
            attention_k: cfg.norm_eps,
            indexer_q: cfg.norm_eps,
            indexer_k: cfg.norm_eps,
            final_norm: cfg.norm_eps,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerContract {
    pub layer: usize,
    /// Dense FFN only; routed/shared branches have separate ownership.
    pub dense_activation: Option<cuteafd_core::FfnActivation>,
    pub shared_activation: Option<cuteafd_core::FfnActivation>,
    pub geometry: GqaGeometry,
    pub theta: f64,
    pub rope: crate::plan::rope::Recipe,
    pub indexer_rope: Option<crate::plan::rope::Recipe>,
    pub rotary_arithmetic: RotaryArithmetic,
    /// None means full-history sparse attention, not dense causal attention.
    pub sliding: Option<SlidingWindow>,
}

impl LayerContract {
    /// Target and predictor blocks share checkpoint geometry, not layer counts.
    pub fn from_config(cfg: &Config, layer: usize) -> Result<Self> {
        ensure!(layer < cfg.layers + cfg.mtp, "Step-5 layer contract out of range");
        Ok(Self {
            layer,
            dense_activation: (!cfg.moe[layer]).then_some(cfg.activation[layer]),
            shared_activation: cfg.moe[layer].then_some(cfg.shared_activation[layer]),
            geometry: GqaGeometry {
                query_heads: cfg.heads,
                kv_heads: cfg.groups,
                qk_dim: cfg.dim,
                value_dim: cfg.dim,
                rotary_dim: cfg.rotary[layer],
            }.validate()?,
            theta: cfg.theta[layer],
            rope: cfg.rope_recipe(layer, false)?,
            indexer_rope: cfg.full[layer].then(|| cfg.rope_recipe(layer, true)).transpose()?,
            rotary_arithmetic: RotaryArithmetic::FusedImaginary,
            sliding: (!cfg.full[layer]).then_some(SlidingWindow { window: cfg.window as u64 }),
        })
    }
}

impl CoordinatorContract {
    /// Requires every selected native target header. Routed experts, vision
    /// and independently owned MTP stages are not charged as target weights.
    /// Full tensors are conservatively replicated; no unimplemented TP packing
    /// or BF16-to-FP8 conversion is credited as a memory saving.
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        let model = Model::open(checkpoint)?;
        let cfg: &Config = &model.cfg;
        let declarations = model
            .tensors
            .iter()
            .filter(|(_, t)| {
                !matches!(
                    t.role.component,
                    Component::RoutedExpert | Component::Vision | Component::Speculator
                )
            })
            .map(|(name, t)| (name.clone(), t.shape.clone(), t.dtype.clone()));
        let weights = NativeDenseLayout::from_headers(checkpoint, declarations)?;
        let layers = (0..cfg.layers)
            .map(|layer| LayerContract::from_config(cfg, layer))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            layers.len() == 92,
            "target coordinator must exclude MTP stages"
        );
        Ok(Self {
            weights,
            norm_weight: NormWeight::ZeroCentered,
            norms: TargetNormEps::from_config(cfg),
            rope_tables: cfg.rope_tables()?,
            router: SigmoidRouter {
                topk: cfg.topk,
                factor: 3.0,
                epsilon: 1e-20,
                bias_order: RouterBiasOrder::SubtractSelected,
            },
            layers,
        })
    }
}

/// Native operand storage, derived before CUDA initialization. This excludes
/// routed experts and MTP, and is not a complete serving admission contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSlice {
    Replicated,
    Rows,
    Cols,
}

#[derive(Debug, Clone)]
pub struct WeightOperand {
    pub name: String,
    pub sources: Vec<CheckpointTensor>,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub slice: WeightSlice,
    pub allocation_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct RankWeightLayout {
    pub rank: usize,
    pub ranks: usize,
    pub operands: Vec<WeightOperand>,
    pub resident_bytes: u64,
}

impl RankWeightLayout {
    pub fn from_checkpoint(checkpoint: &Checkpoint, rank: usize, ranks: usize) -> Result<Self> {
        Self::build(checkpoint, rank, ranks, None)
    }

    /// Whole owners retain full native operands, and peers retain no private
    /// attention or FFN weights. Diagnostic head splitting stays separate.
    pub fn for_execution(checkpoint: &Checkpoint, rank: usize,
        ownership: &super::ownership::ExecutionOwnership) -> Result<Self> {
        ownership.validate_config(&Config::parse(checkpoint)?)?;
        Self::build(checkpoint, rank, ownership.ranks(), Some(ownership))
    }

    /// MTP stages stay whole on their admitted owner. Matching head shapes
    /// do not establish tied checkpoint storage or identical weights.
    pub fn for_mtp(checkpoint: &Checkpoint, depth: usize) -> Result<Self> {
        let model = Model::open(checkpoint)?;
        ensure!(depth <= model.cfg.mtp, "Step-5 MTP depth exceeds checkpoint stages");
        if depth == 0 {
            return Ok(Self { rank: 0, ranks: 1, operands: vec![], resident_bytes: 0 });
        }
        let end = model.cfg.layers + depth;
        let declarations = model.tensors.iter().filter(|(_, tensor)| {
            tensor.role.component == Component::Speculator
                && tensor.role.layer.is_some_and(|layer| layer < end)
        }).map(|(name, tensor)| (name.clone(), tensor.shape.clone(), tensor.dtype.clone()));
        let weights = NativeDenseLayout::from_headers(checkpoint, declarations)?;
        let mut layout = Self::from_native(&weights, 0, 1, None)?;
        for operand in &mut layout.operands {
            operand.slice = WeightSlice::Replicated;
        }
        Ok(layout)
    }

    fn build(checkpoint: &Checkpoint, rank: usize, ranks: usize,
        ownership: Option<&super::ownership::ExecutionOwnership>) -> Result<Self> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        Self::from_native(&contract.weights, rank, ranks, ownership)
    }

    fn from_native(weights: &NativeDenseLayout, rank: usize, ranks: usize,
        ownership: Option<&super::ownership::ExecutionOwnership>) -> Result<Self> {
        ensure!(
            matches!(ranks, 1 | 2) && rank < ranks,
            "Step-5 coordinator rank geometry"
        );
        let sources: std::collections::BTreeMap<_, _> = weights
            .tensors
            .iter()
            .map(|t| (t.meta.name.as_str(), t))
            .collect();
        let mut operands = Vec::new();
        let mut resident_bytes = 0u64;
        for source in &weights.tensors {
            let name = &source.meta.name;
            let mut split_ranks = ranks;
            if let Some(ownership) = ownership {
                if let Some(rest) = name.strip_prefix("model.layers.") {
                    let (layer, operand) = rest.split_once('.')
                        .ok_or_else(|| anyhow::anyhow!("Step-5 layer operand"))?;
                    let local = ownership.layer(layer.parse()?, rank)?;
                    let attention = operand.starts_with("self_attn.")
                        || operand.starts_with("input_layernorm.");
                    if (attention && local.query_heads == 0) || (!attention && !local.ffn) { continue; }
                    split_ranks = if local.query_heads == 32 { 2 } else { 1 };
                } else {
                    let owner = if name == "model.embed_tokens.weight" {
                        ownership.execution().spec().entry_gpu
                    } else { ownership.execution().spec().head_gpu };
                    if rank != usize::from(owner) { continue; }
                    split_ranks = 1;
                }
            }
            if name.ends_with(".up_proj.weight") {
                continue;
            }
            let mut selected = vec![source.clone()];
            let mut shape = source.meta.shape.clone();
            let mut output_name = name.clone();
            if let Some(base) = name.strip_suffix(".gate_proj.weight") {
                let up = format!("{base}.up_proj.weight");
                let up = sources
                    .get(up.as_str())
                    .ok_or_else(|| anyhow::anyhow!("missing packed up operand"))?;
                ensure!(
                    up.meta.shape == shape && up.meta.dtype == source.meta.dtype,
                    "Step-5 gate/up operands disagree"
                );
                shape[0] = shape[0]
                    .checked_mul(2)
                    .ok_or_else(|| anyhow::anyhow!("packed rows overflow"))?;
                selected.push((*up).clone());
                output_name = format!("{base}.gate_up_proj.weight");
            }
            let index_bf16 = [
                ".sparse_indexer_w.weight",
                ".sparse_indexer_q_norm.weight",
                ".sparse_indexer_k_norm.weight",
                ".sparse_indexer_k_norm.bias",
            ]
            .iter()
            .any(|suffix| name.ends_with(suffix));
            let dtype = if index_bf16 {
                DType::Bf16
            } else {
                source.meta.dtype.clone()
            };
            let slice = if ownership.is_some() && split_ranks == 1 {
                WeightSlice::Replicated
            } else if [
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
                ".self_attn.g_proj.weight",
                ".sparse_indexer_q.weight",
                ".sparse_indexer_w.weight",
            ]
            .iter()
            .any(|suffix| name.ends_with(suffix))
            {
                WeightSlice::Rows
            } else if name.ends_with(".self_attn.o_proj.weight") {
                WeightSlice::Cols
            } else {
                WeightSlice::Replicated
            };
            if slice != WeightSlice::Replicated {
                ensure!(shape.len() == 2, "Step-5 split operand must be a matrix");
                let axis = if slice == WeightSlice::Rows { 0 } else { 1 };
                ensure!(shape[axis] % split_ranks == 0, "Step-5 uneven head split");
                shape[axis] /= split_ranks;
            }
            let element_bytes = match dtype {
                DType::Bf16 => 2u64,
                DType::F32 => 4u64,
                _ => anyhow::bail!("Step-5 unsupported coordinator operand precision"),
            };
            let bytes = shape
                .iter()
                .try_fold(element_bytes, |n, dim| n.checked_mul(*dim as u64))
                .ok_or_else(|| anyhow::anyhow!("Step-5 operand size overflow"))?
                .max(256);
            resident_bytes = resident_bytes
                .checked_add(bytes)
                .ok_or_else(|| anyhow::anyhow!("Step-5 rank weight size overflow"))?;
            operands.push(WeightOperand {
                name: output_name,
                sources: selected,
                dtype,
                shape,
                slice,
                allocation_bytes: bytes,
            });
        }
        Ok(Self {
            rank,
            ranks,
            operands,
            resident_bytes,
        })
    }

    /// Caller must supply every non-weight cost before constructing an owner.
    /// This method does not infer missing workspace/module costs as zero.
    pub fn check_budget(&self, available: u64, reservations: &[u64]) -> Result<u64> {
        let total = reservations
            .iter()
            .try_fold(self.resident_bytes, |n, v| n.checked_add(*v))
            .ok_or_else(|| anyhow::anyhow!("Step-5 coordinator reservation overflow"))?;
        ensure!(
            total <= available,
            "Step-5 coordinator weights and reservations need {total} bytes, available {available}"
        );
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::families::step5::tests::snapshot;
    use cuteafd_core::DType;

    #[test]
    fn native_coordinator_preserves_source_precision_and_layer_schedule() {
        let dir = snapshot("fp8");
        let cp = Checkpoint::open(dir.path()).unwrap();
        let c = CoordinatorContract::from_checkpoint(&cp).unwrap();
        assert_eq!(c.norms.indexer_q, 1e-5);
        assert_eq!(c.norms.indexer_k, 1e-5);
        assert_eq!(c.norm_weight, NormWeight::ZeroCentered);
        assert_eq!(c.router.bias_order, RouterBiasOrder::SubtractSelected);
        assert_eq!(c.layers.len(), 92);
        assert_eq!(c.layers.iter().filter(|l| l.sliding.is_some()).count(), 69);
        for layer in &c.layers {
            assert_eq!(
                layer.dense_activation.is_some(),
                layer.layer < 3 || layer.layer == 91
            );
            if let Some(activation) = layer.dense_activation {
                assert!(matches!(
                    activation,
                    cuteafd_core::FfnActivation::SiluPostClampFp32Product { .. }
                ));
            }
            assert_eq!(layer.geometry.value_dim, 192);
            assert_eq!(layer.rotary_arithmetic, RotaryArithmetic::FusedImaginary);
            assert_eq!(
                layer.geometry.rotary_dim,
                if layer.layer % 4 == 3 { 64 } else { 192 }
            );
            assert_eq!(
                layer.sliding.map(|s| s.window),
                if layer.layer % 4 == 3 {
                    None
                } else {
                    Some(512)
                }
            );
        }
        assert!(c
            .weights
            .tensors
            .iter()
            .all(|t| !t.meta.name.contains("layers.92.")
                && !t.meta.name.contains("vision")
                && !t.meta.name.contains("moe.gate_proj")));
        let find = |name: &str| {
            c.weights
                .tensors
                .iter()
                .find(|t| t.meta.name == name)
                .unwrap()
        };
        assert_eq!(
            find("model.layers.3.moe.gate.weight").meta.dtype,
            DType::Bf16
        );
        assert_eq!(
            find("model.layers.3.moe.router_bias").meta.dtype,
            DType::F32
        );
        assert_eq!(
            find("model.layers.0.self_attn.q_norm.weight").meta.dtype,
            DType::F32
        );
        assert_eq!(
            find("model.layers.0.self_attn.v_proj.weight").meta.shape,
            [768, 4096]
        );
        assert_eq!(
            find("model.layers.0.self_attn.g_proj.weight").meta.shape,
            [64, 4096]
        );
        let exact: u64 = c
            .weights
            .tensors
            .iter()
            .map(|t| t.meta.byte_length.max(256))
            .sum();
        assert_eq!(c.weights.resident_bytes, exact);
        assert_eq!(exact, 27_651_962_624);
        assert_eq!(c.weights.tensors.len(), 1467);
        assert_eq!(
            c.weights.check_weight_budget(exact + 12, &[5, 7]).unwrap(),
            exact + 12
        );
        assert!(c.weights.check_weight_budget(exact + 11, &[5, 7]).is_err());
        assert!(c
            .weights
            .check_weight_budget(u64::MAX, &[u64::MAX])
            .is_err());
        // The contract does not depend on routed storage or sidecars.
        let nv = snapshot("nvfp4");
        assert_eq!(
            CoordinatorContract::from_checkpoint(&Checkpoint::open(nv.path()).unwrap())
                .unwrap()
                .weights
                .resident_bytes,
            exact
        );
    }

    #[test]
    fn predictor_layer_geometry_is_native_without_extending_the_target() {
        for quant in ["fp8", "nvfp4"] {
            let dir = snapshot(quant);
            let cp = Checkpoint::open(dir.path()).unwrap();
            let cfg = Config::parse(&cp).unwrap();
            assert_eq!(CoordinatorContract::from_checkpoint(&cp).unwrap().layers.len(), 92);
            for layer in 92..95 {
                let contract = LayerContract::from_config(&cfg, layer).unwrap();
                assert_eq!(contract.layer, layer);
                assert_eq!(contract.geometry.query_heads, 64);
                assert_eq!(contract.geometry.kv_heads, 4);
                assert_eq!(contract.geometry.rotary_dim, 192);
                assert_eq!(contract.sliding.unwrap().window, 512);
                assert!(contract.indexer_rope.is_none());
                assert!(contract.shared_activation.is_none());
                assert_eq!(contract.dense_activation, Some(
                    cuteafd_core::FfnActivation::SiluPostClampFp32Product { limit: 7.0 }));
            }
            assert!(LayerContract::from_config(&cfg, 95).is_err());
        }
    }

    #[test]
    fn every_target_norm_uses_the_served_config_not_kernel_defaults() {
        for quant in ["fp8", "nvfp4"] {
            let dir = snapshot(quant);
            let mut cp = Checkpoint::open(dir.path()).unwrap();
            assert_eq!(
                cp.config["text_config"]["rms_norm_eps"],
                serde_json::json!(1e-5)
            );
            for epsilon in [1e-5f32, 1e-6, 0.001] {
                cp.config["text_config"]["rms_norm_eps"] = serde_json::json!(epsilon);
                let norms = CoordinatorContract::from_checkpoint(&cp).unwrap().norms;
                for actual in [
                    norms.input,
                    norms.post_attention,
                    norms.attention_q,
                    norms.attention_k,
                    norms.indexer_q,
                    norms.indexer_k,
                    norms.final_norm,
                ] {
                    assert_eq!(actual.to_bits(), epsilon.to_bits());
                }
            }
        }
    }

    #[test]
    fn native_rank_operands_split_attention_and_replicate_ffns() {
        let dir = snapshot("fp8");
        let cp = Checkpoint::open(dir.path()).unwrap();
        let one = RankWeightLayout::from_checkpoint(&cp, 0, 1).unwrap();
        let two = RankWeightLayout::from_checkpoint(&cp, 1, 2).unwrap();
        assert_eq!(
            two.resident_bytes,
            RankWeightLayout::from_checkpoint(&cp, 0, 2)
                .unwrap()
                .resident_bytes
        );
        assert_eq!(one.resident_bytes, 27_648_912_640);
        assert_eq!(two.resident_bytes, 17_397_574_912);
        let get = |name: &str| two.operands.iter().find(|t| t.name == name).unwrap();
        for (suffix, shape, slice) in [
            (
                "self_attn.q_proj.weight",
                vec![6144, 4096],
                WeightSlice::Rows,
            ),
            (
                "self_attn.k_proj.weight",
                vec![384, 4096],
                WeightSlice::Rows,
            ),
            (
                "self_attn.v_proj.weight",
                vec![384, 4096],
                WeightSlice::Rows,
            ),
            ("self_attn.g_proj.weight", vec![32, 4096], WeightSlice::Rows),
            (
                "self_attn.o_proj.weight",
                vec![4096, 6144],
                WeightSlice::Cols,
            ),
            (
                "self_attn.sparse_indexer_q.weight",
                vec![2048, 4096],
                WeightSlice::Rows,
            ),
            (
                "self_attn.sparse_indexer_w.weight",
                vec![8, 4096],
                WeightSlice::Rows,
            ),
            (
                "share_expert.gate_up_proj.weight",
                vec![3072, 4096],
                WeightSlice::Replicated,
            ),
            (
                "share_expert.down_proj.weight",
                vec![4096, 1536],
                WeightSlice::Replicated,
            ),
        ] {
            let operand = get(&format!("model.layers.3.{suffix}"));
            assert_eq!(operand.shape, shape, "{suffix}");
            assert_eq!(operand.slice, slice, "{suffix}");
            assert_eq!(operand.dtype, DType::Bf16);
        }
        assert_eq!(
            get("model.layers.0.mlp.gate_up_proj.weight").shape,
            [27648, 4096]
        );
        let packed = get("model.layers.3.share_expert.gate_up_proj.weight");
        assert_eq!(packed.sources.len(), 2);
        assert!(packed.sources[0].meta.name.ends_with("gate_proj.weight"));
        assert!(packed.sources[1].meta.name.ends_with("up_proj.weight"));
        for suffix in [
            "self_attn.sparse_indexer_q_norm.weight",
            "self_attn.sparse_indexer_k_norm.weight",
            "self_attn.sparse_indexer_k_norm.bias",
            "self_attn.sparse_indexer_w.weight",
        ] {
            let operand = get(&format!("model.layers.3.{suffix}"));
            assert_eq!(operand.sources[0].meta.dtype, DType::F32);
            assert_eq!(operand.dtype, DType::Bf16);
        }
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.ssmax_s",
            "moe.router_bias",
        ] {
            let operand = get(&format!("model.layers.3.{suffix}"));
            assert_eq!(operand.dtype, DType::F32);
            assert_eq!(operand.slice, WeightSlice::Replicated);
        }
        for (rank, ranks) in [(0, 1), (0, 2), (1, 2)] {
            let layout = RankWeightLayout::from_checkpoint(&cp, rank, ranks).unwrap();
            for layer in CoordinatorContract::from_checkpoint(&cp).unwrap().layers.iter()
                .filter(|layer| layer.shared_activation.is_some()) {
                let name = format!("model.layers.{}.moe.gate.weight", layer.layer);
                let gate = layout.operands.iter().find(|w| w.name == name).unwrap();
                assert_eq!(gate.dtype, DType::Bf16);
                assert_eq!(gate.shape, [352, 4096]);
                assert_eq!(gate.slice, WeightSlice::Replicated);
                assert_eq!(gate.allocation_bytes, 352 * 4096 * 2);
                assert_eq!(gate.sources.len(), 1);
                assert_eq!(gate.sources[0].meta.dtype, DType::Bf16);
            }
        }
        let covered: Vec<_> = two
            .operands
            .iter()
            .flat_map(|o| o.sources.iter().map(|s| s.meta.name.clone()))
            .collect();
        let unique: std::collections::BTreeSet<_> = covered.iter().collect();
        assert_eq!(covered.len(), 1467);
        assert_eq!(unique.len(), covered.len());
        assert_eq!(two.operands.len(), 1375);
        for layout in [&one, &two] {
            let exact: u64 = layout.operands.iter().map(|o| o.allocation_bytes).sum();
            assert_eq!(layout.resident_bytes, exact);
            assert_eq!(
                layout.check_budget(exact + 12, &[5, 7]).unwrap(),
                exact + 12
            );
            assert!(layout.check_budget(exact + 11, &[5, 7]).is_err());
            assert!(layout.check_budget(u64::MAX, &[u64::MAX]).is_err());
        }
        let nv = snapshot("nvfp4");
        assert_eq!(
            two.resident_bytes,
            RankWeightLayout::from_checkpoint(&Checkpoint::open(nv.path()).unwrap(), 0, 2)
                .unwrap()
                .resident_bytes
        );
        for (rank, ranks) in [(0, 0), (0, 3), (2, 2), (1, 1)] {
            assert!(RankWeightLayout::from_checkpoint(&cp, rank, ranks).is_err());
        }
    }

    #[test]
    fn mtp_residency_charges_each_native_stage_and_head() {
        for format in ["fp8", "nvfp4"] {
            let dir = snapshot(format);
            let cp = Checkpoint::open(dir.path()).unwrap();
            let zero = RankWeightLayout::for_mtp(&cp, 0).unwrap();
            assert!(zero.operands.is_empty());
            assert_eq!(zero.resident_bytes, 0);
            for depth in 1..=3 {
                let layout = RankWeightLayout::for_mtp(&cp, depth).unwrap();
                assert_eq!(layout.operands.len(), 16 * depth);
                assert_eq!(layout.resident_bytes, 1_677_280_768 * depth as u64);
                assert!(layout.operands.iter().all(|operand| operand.slice == WeightSlice::Replicated));
                for stage in 92..92 + depth {
                    let base = format!("model.layers.{stage}");
                    for norm in ["enorm", "hnorm", "transformer.shared_head.norm"] {
                        let norm = layout.operands.iter().find(|operand|
                            operand.name == format!("{base}.{norm}.weight")).unwrap();
                        assert_eq!(norm.dtype, DType::F32);
                        assert_eq!(norm.shape, [4096]);
                    }
                    let head = layout.operands.iter().find(|operand|
                        operand.name == format!("{base}.transformer.shared_head.output.weight")).unwrap();
                    assert_eq!(head.shape, [128896, 4096]);
                    assert_eq!(head.allocation_bytes, 128896 * 4096 * 2);
                    let concat = layout.operands.iter().find(|operand|
                        operand.name == format!("{base}.eh_proj.weight")).unwrap();
                    assert_eq!(concat.shape, [4096, 8192]);
                }
                let exact = layout.resident_bytes;
                assert_eq!(layout.check_budget(exact + 12, &[5, 7]).unwrap(), exact + 12);
                assert!(layout.check_budget(exact + 11, &[5, 7]).is_err());
            }
            assert!(RankWeightLayout::for_mtp(&cp, 4).is_err());
            let mut bad = Checkpoint::open(dir.path()).unwrap();
            bad.tensors.iter_mut().find(|tensor|
                tensor.meta.name == "model.layers.94.eh_proj.weight").unwrap().meta.byte_length -= 1;
            assert!(RankWeightLayout::for_mtp(&bad, 3).is_err());
        }
    }

    #[test]
    fn headers_fail_closed_before_native_allocation() {
        let dir = snapshot("fp8");
        for mutation in 0..5 {
            let mut cp = Checkpoint::open(dir.path()).unwrap();
            let t = cp
                .tensors
                .iter_mut()
                .find(|t| t.meta.name == "model.layers.0.self_attn.q_norm.weight")
                .unwrap();
            match mutation {
                0 => t.meta.dtype = DType::Bf16,
                1 => t.meta.shape = vec![191],
                2 => t.meta.byte_length -= 1,
                3 => t.meta.byte_offset = u64::MAX,
                _ => cp
                    .tensors
                    .retain(|t| t.meta.name != "model.layers.0.self_attn.q_norm.weight"),
            }
            assert!(
                CoordinatorContract::from_checkpoint(&cp).is_err(),
                "mutation {mutation}"
            );
        }
        for bad in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!("nan"),
        ] {
            let mut cp = Checkpoint::open(dir.path()).unwrap();
            cp.config["text_config"]["rms_norm_eps"] = bad;
            assert!(CoordinatorContract::from_checkpoint(&cp).is_err());
        }
    }
}
