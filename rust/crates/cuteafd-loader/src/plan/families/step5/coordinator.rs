//! Native source-weight contract, not an executable Step-5 coordinator.
//! Reuses the planner inventory and shared dense header/admission validator.
use super::{Config, Model};
use crate::formats::native_dense::NativeDenseLayout;
use crate::plan::{
    checkpoint::{Checkpoint, CheckpointTensor},
    spec::Component,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::dense_attention::{
    GqaGeometry, NormWeight, RotaryArithmetic, RouterBiasOrder, SigmoidRouter, SlidingWindow,
};
use cuteafd_core::DType;

#[derive(Debug, Clone)]
pub struct CoordinatorContract {
    pub core_view: crate::plan::checkpoint::CoreView,
    pub fp8_tensors: Vec<String>,
    pub weights: NativeDenseLayout,
    /// Header-native EXL3 dense projections (empty for FP8/NVFP4 releases).
    pub core: super::exl3::CoreInventory,
    pub norm_weight: NormWeight,
    pub norms: TargetNormEps,
    pub rope_tables: crate::plan::rope::Tables,
    pub router: SigmoidRouter,
    pub layers: Vec<LayerContract>,
    pub(super) mtp_declarations: Vec<(String, Vec<usize>, DType)>,
}

/// Official target norms all receive config.rms_norm_eps, including the
/// indexer LayerNorm (whose fused kernel's default is not the model value).
/// MTP enorm/hnorm and each private head norm use the same checkpoint epsilon.
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
    /// FP8-compact load-only staging; final device values/scales are charged by rank layouts.
    pub fn attention_mount_load(&self, ranks: usize) -> Result<crate::shared::dense_read::MountLoad> {
        if self.core_view != crate::plan::checkpoint::CoreView::Fp8Compact {
            return Ok(Default::default());
        }
        crate::shared::dense_read::MountLoad::from_headers(self.weights.tensors.iter()
            .filter(|tensor| self.fp8_tensors.contains(&tensor.meta.name))
            .map(|tensor| &tensor.meta), ranks)
    }

    /// `copy_h2d` retains its largest pinned source until the library drops.
    /// Whole source extents conservatively cover TP shards; embedding uses the
    /// separate dense loader, not this reusable synchronous upload owner.
    pub fn retained_sync_h2d_bytes(&self) -> Result<u64> {
        let mut maximum = self.weights.tensors.iter()
            .filter(|tensor| tensor.meta.name != "model.embed_tokens.weight")
            .map(|tensor| tensor.meta.byte_length).max().unwrap_or(0);
        for (_, shape, dtype) in &self.mtp_declarations {
            let element = match dtype {
                DType::Bf16 => 2u64, DType::F32 => 4, DType::F8E4M3 => 1,
                _ => anyhow::bail!("unsupported Step-5 retained upload dtype {dtype:?}"),
            };
            let bytes = shape.iter().try_fold(element, |bytes, &dim| bytes.checked_mul(dim as u64))
                .ok_or_else(|| anyhow::anyhow!("Step-5 retained upload extent overflow"))?;
            maximum = maximum.max(bytes);
        }
        for projection in self.core.projections.values() {
            let n = projection.stored_outputs as u64;
            let k = projection.shape[1] as u64;
            let payload = n.checked_mul(k).and_then(|bytes| bytes.checked_mul(projection.bits as u64))
                .ok_or_else(|| anyhow::anyhow!("Step-5 retained core upload overflow"))? / 8;
            maximum = maximum.max(payload).max(n * 2).max(k * 2);
        }
        for recipe in &self.rope_tables.recipes {
            maximum = maximum.max(recipe.allocation_bytes(self.rope_tables.extent)? / 2);
        }
        Ok(maximum.max(256))
    }

    /// Requires every selected native target header. Routed experts, vision
    /// and independently owned MTP stages are not charged as target weights.
    /// Full tensors are conservatively replicated; no unimplemented TP packing
    /// or BF16-to-FP8 conversion is credited as a memory saving.
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        let _construction = tracing::debug_span!("step5_coordinator_contract_construction").entered();
        let model = Model::open(checkpoint)?;
        Self::from_model(checkpoint, &model)
    }

    /// Reuse the planner's validated model instead of rebuilding its tensor contract.
    pub(super) fn from_model(checkpoint: &Checkpoint, model: &Model) -> Result<Self> {
        checkpoint.validate_fp8_roles("Step-5", |name| {
            let attention = name.contains(".self_attn.");
            crate::formats::native_attention::Fp8Role::exact(if attention { "attention projection" } else { "core projection" },
                name.ends_with(".self_attn.o_proj.weight"))
        })?;
        let cfg: &Config = &model.cfg;
        let declarations = model
            .tensors
            .iter()
            .filter(|(name, t)| {
                !matches!(
                    t.role.component,
                    Component::RoutedExpert | Component::Vision | Component::Speculator
                ) && !is_exl3_member(name)
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
        let mtp_declarations = model.tensors.iter().filter(|(name, tensor)|
            !is_exl3_member(name) && tensor.role.component == Component::Speculator)
            .map(|(name, tensor)| (name.clone(), tensor.shape.clone(), tensor.dtype.clone())).collect();
        Ok(Self {
            core_view: checkpoint.core_view(),
            fp8_tensors: checkpoint.view.as_ref().map_or_else(Vec::new, |view| view.fp8_tensors().to_vec()),
            weights,
            mtp_declarations,
            core: model.core.clone(),
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

impl WeightOperand {
    pub fn native_layout(&self, rank: usize, ranks: usize) -> Result<crate::formats::native_attention::NativeLayout> {
        use crate::formats::trellis_core::CoreSplit;
        ensure!(self.shape.len() == 2, "Step-5 native output matrix");
        let split = match self.slice { WeightSlice::Replicated => CoreSplit::Whole,
            WeightSlice::Rows => CoreSplit::Output, WeightSlice::Cols => CoreSplit::Input };
        Ok(crate::formats::native_attention::NativeLayout {
            stem: self.name.strip_suffix(".weight").ok_or_else(|| anyhow::anyhow!("native weight name"))?.into(),
            shape: [self.shape[0], self.shape[1]], fp8: self.dtype == DType::F8E4M3,
            rank: if split == CoreSplit::Whole { 0 } else { rank },
            ranks: if split == CoreSplit::Whole { 1 } else { ranks }, split,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RankWeightLayout {
    pub rank: usize,
    pub ranks: usize,
    pub operands: Vec<WeightOperand>,
    /// Compressed native trellis owners by `.weight` operand name; gate and up
    /// stay separate, independently calibrated projections.
    pub core: std::collections::BTreeMap<String, crate::formats::trellis_core::CoreWeightLayout>,
    pub resident_bytes: u64,
}

fn is_exl3_member(name: &str) -> bool {
    matches!(crate::plan::format::split_stem(name).1, "trellis" | "suh" | "svh" | "mcg")
}

/// Program and allocation prefix of the target's rotary tables.
pub const ROPE_PREFIX: &str = "step5_";

impl RankWeightLayout {
    /// Final rank-local owners only; composite FP8 values/scales remain one view.
    pub fn arena_tensors(&self, select: impl Fn(&str) -> bool) -> Result<Vec<(String, usize)>> {
        let mut tensors = self.operands.iter().filter(|operand| select(&operand.name))
            .map(|operand| Ok((if operand.name == "model.embed_tokens.weight" {
                "embedding.table".to_owned()
            } else { operand.name.clone() }, usize::try_from(operand.allocation_bytes)?)))
            .collect::<Result<Vec<_>>>()?;
        for (name, core) in &self.core {
            if select(name) { tensors.extend(core.arena_tensors(&core.stem)?); }
        }
        Ok(tensors)
    }

    pub fn arena_layout(&self, id: impl Into<String>, select: impl Fn(&str) -> bool)
        -> Result<crate::serving_capacity::WeightArenaLayout> {
        crate::serving_capacity::WeightArenaLayout::new(id, self.arena_tensors(select)?)
    }

    /// The one target arena a coordinator rank allocates, planned and served
    /// alike: the selected operands and trellis cores of the first `count`
    /// layers (globals only for the whole model), the generated rotary planes,
    /// and the embedding table on the entry rank.
    pub fn target_arena(&self, rope: Option<&crate::plan::rope::Tables>, device: i32, count: usize,
        embedding: bool) -> Result<crate::serving_capacity::WeightArenaLayout> {
        let mut tensors = self.arena_tensors(|name| {
            if name == "model.embed_tokens.weight" { return embedding; }
            if let Some(rest) = name.strip_prefix("model.layers.") {
                return rest.split_once('.').and_then(|(layer, _)| layer.parse::<usize>().ok())
                    .is_some_and(|layer| layer < count);
            }
            count == 92
        })?;
        if let Some(rope) = rope {
            for (index, recipe) in rope.recipes.iter().enumerate() {
                let plane = usize::try_from(recipe.allocation_bytes(rope.extent)? / 2)?;
                for part in ["cos", "sin"] { tensors.push((format!("{ROPE_PREFIX}rotary.{index}.{part}"), plane)); }
            }
        }
        crate::serving_capacity::WeightArenaLayout::new(format!("step5/target/gpu{device}"), tensors)
    }

    /// The predictor's own arena: every stage's owners, nothing of the target.
    pub fn mtp_arena(&self, device: i32) -> Result<crate::serving_capacity::WeightArenaLayout> {
        self.arena_layout(format!("step5/mtp/gpu{device}"), |_| true)
    }

    /// The predictor's single physical backing and its separately derived SM120 tail.
    pub fn mtp_weight_charge(checkpoint: &Checkpoint, contract: &CoordinatorContract, depth: usize) -> Result<u64> {
        let layout = Self::mtp_from_contract(checkpoint, contract, depth)?;
        if layout.operands.is_empty() && layout.core.is_empty() { return Ok(0); }
        let bytes = layout.mtp_arena(0)?.bytes() as u64;
        bytes.checked_add(crate::placement::planner::allocation_granularity_forecast(&[bytes])?)
            .context("Step-5 predictor arena charge overflow")
    }

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
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        Self::mtp_from_contract(checkpoint, &contract, depth)
    }

    pub fn mtp_from_contract(checkpoint: &Checkpoint, contract: &CoordinatorContract,
        depth: usize) -> Result<Self> {
        let cfg = Config::parse(checkpoint)?;
        ensure!(depth <= cfg.mtp, "Step-5 MTP depth exceeds checkpoint stages");
        if depth == 0 {
            return Ok(Self { rank: 0, ranks: 1, operands: vec![], core: Default::default(), resident_bytes: 0 });
        }
        let declarations = contract.mtp_declarations.iter().filter(|(name, _, _)|
            (0..depth).any(|stage| name.starts_with(&format!("model.layers.{}.", cfg.layers + stage))))
            .cloned();
        let weights = NativeDenseLayout::from_headers(checkpoint, declarations)?;
        let mut layout = Self::from_native(&weights, 0, 1, None)?;
        for operand in &mut layout.operands { operand.slice = WeightSlice::Replicated; }
        layout.attach_core(&cfg, &contract.core, None, |layer| layer >= cfg.layers && layer < cfg.layers + depth)?;
        layout.select_core_view(&contract.fp8_tensors)?;
        Ok(layout)
    }

    /// One whole predictor owner, with no target or earlier-stage operands.
    pub fn for_mtp_stage(checkpoint: &Checkpoint, stage: usize) -> Result<Self> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        Self::mtp_stage_from_contract(checkpoint, &contract, stage)
    }

    pub fn mtp_stage_from_contract(checkpoint: &Checkpoint, contract: &CoordinatorContract,
        stage: usize) -> Result<Self> {
        let cfg = Config::parse(checkpoint)?;
        ensure!(stage < cfg.mtp, "Step-5 predictor stage outside checkpoint");
        let prefix = format!("model.layers.{}.", cfg.layers + stage);
        let mut layout = Self::mtp_from_contract(checkpoint, contract, stage + 1)?;
        layout.operands.retain(|operand| operand.name.starts_with(&prefix));
        layout.core.retain(|name, _| name.starts_with(&prefix));
        layout.resident_bytes = layout.operands.iter().map(|operand| operand.allocation_bytes)
            .chain(layout.core.values().map(|core| core.bytes as u64))
            .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
            .ok_or_else(|| anyhow::anyhow!("Step-5 stage weight size overflow"))?;
        // 16 operands, with BF16 gate/up packed as one; native gate and up stay separate.
        let native = layout.core.len() - usize::from(layout.core.keys().any(|n| n.ends_with(".mlp.up_proj.weight")));
        ensure!(layout.operands.len() + native == 16, "Step-5 predictor operand inventory");
        Ok(layout)
    }

    fn build(checkpoint: &Checkpoint, rank: usize, ranks: usize,
        ownership: Option<&super::ownership::ExecutionOwnership>) -> Result<Self> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let cfg = Config::parse(checkpoint)?;
        Self::from_contract(&contract, &cfg, rank, ranks, ownership)
    }

    pub fn from_contract(contract: &CoordinatorContract, cfg: &Config,
        rank: usize, ranks: usize, ownership: Option<&super::ownership::ExecutionOwnership>) -> Result<Self> {
        let _construction = tracing::debug_span!("step5_rank_layout_construction", rank).entered();
        if let Some(ownership) = ownership { ownership.validate_config(cfg)?; }
        let mut layout = Self::from_native(&contract.weights, rank, ranks, ownership)?;
        layout.attach_core(cfg, &contract.core, ownership, |layer| layer < cfg.layers)?;
        layout.select_core_view(&contract.fp8_tensors)?;
        Ok(layout)
    }

    /// Replace exactly the metadata-selected native matrices, never the EXL3 core.
    fn select_core_view(&mut self, tensors: &[String]) -> Result<()> {
        if tensors.is_empty() { return Ok(()); }
        for operand in &mut self.operands {
            if !tensors.contains(&operand.name) { continue; }
            ensure!(operand.dtype == DType::Bf16 && operand.shape.len() == 2
                && operand.shape.iter().all(|dim| dim % 128 == 0), "Step-5 FP8 output geometry");
            operand.dtype = DType::F8E4M3;
            operand.allocation_bytes = operand.native_layout(self.rank, self.ranks)?.bytes()?;
        }
        self.resident_bytes = self.operands.iter().map(|w| w.allocation_bytes)
            .chain(self.core.values().map(|w| w.bytes as u64))
            .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
            .context("Step-5 selected attention weight size overflow")?;
        Ok(())
    }

    /// Native trellis owners follow the same ownership as their BF16 operands:
    /// head-split ranks slice q/k/v by output rows and o_proj by input columns;
    /// FFNs and heads stay whole on their owner. Admission counts their bytes.
    fn attach_core(&mut self, cfg: &Config, core: &super::exl3::CoreInventory,
        ownership: Option<&super::ownership::ExecutionOwnership>, layers: impl Fn(usize) -> bool) -> Result<()> {
        use crate::formats::{dense_trellis::TrellisSide, trellis_core::CoreWeightLayout};
        for (stem, projection) in &core.projections {
            let name = format!("{stem}.weight");
            let (layer, leaf) = match stem.strip_prefix("model.layers.").and_then(|r| r.split_once('.')) {
                Some((layer, leaf)) => (Some(layer.parse::<usize>()?), leaf),
                None => (None, stem.as_str()),
            };
            let (world, side) = match (layer, ownership) {
                (Some(layer), _) if !layers(layer) => continue,
                (None, _) if layers(cfg.layers) => continue,
                (None, Some(ownership)) => {
                    if self.rank != usize::from(ownership.execution().spec().head_gpu) { continue; }
                    (1, TrellisSide::Output)
                }
                (None, None) if self.rank != 0 => continue,
                (None, None) => (1, TrellisSide::Output),
                (Some(layer), Some(ownership)) => {
                    let local = ownership.layer(layer, self.rank)?;
                    let attention = leaf.starts_with("self_attn.");
                    if (attention && local.query_heads == 0) || (!attention && !local.ffn) { continue; }
                    match (attention && local.query_heads == 32, leaf) {
                        (true, "self_attn.o_proj") => (2, TrellisSide::Input),
                        (true, _) => (2, TrellisSide::Output),
                        _ => (1, TrellisSide::Output),
                    }
                }
                (Some(_), None) if self.ranks == 1 => (1, TrellisSide::Output),
                (Some(_), None) => match leaf {
                    "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj" => (2, TrellisSide::Output),
                    "self_attn.o_proj" => (2, TrellisSide::Input),
                    _ => (1, TrellisSide::Output),
                },
            };
            let layout = CoreWeightLayout::new(projection, side, if world == 1 { 0 } else { self.rank }, world)?;
            self.resident_bytes = self.resident_bytes.checked_add(layout.bytes as u64)
                .ok_or_else(|| anyhow::anyhow!("Step-5 native core size overflow"))?;
            self.core.insert(name, layout);
        }
        Ok(())
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
            core: Default::default(),
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
            let mut stage_total = 0;
            for stage in 0..3 {
                let layout = RankWeightLayout::for_mtp_stage(&cp, stage).unwrap();
                let prefix = format!("model.layers.{}.", 92 + stage);
                assert_eq!(layout.resident_bytes, 1_677_280_768);
                assert!(layout.operands.iter().all(|operand| operand.name.starts_with(&prefix)
                    && operand.sources.iter().all(|source| source.meta.name.starts_with(&prefix))
                    && operand.slice == WeightSlice::Replicated));
                stage_total += layout.resident_bytes;
            }
            assert_eq!(stage_total, RankWeightLayout::for_mtp(&cp, 3).unwrap().resident_bytes);
            assert!(RankWeightLayout::for_mtp_stage(&cp, 3).is_err());
            assert!(RankWeightLayout::for_mtp_stage(&cp, usize::MAX).is_err());
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

    /// Before: one cudaMalloc per final owner (operand, or trellis member);
    /// after: one arena. SM120 rounds requests over 1 MiB up to 2 MiB.
    fn census(layout: &RankWeightLayout, arena: &crate::serving_capacity::WeightArenaLayout) -> String {
        let tensors = layout.arena_tensors(|_| true).unwrap();
        let bytes = tensors.iter().map(|(_, n)| *n as u64).sum::<u64>();
        let large = tensors.iter().filter(|(_, n)| *n > 1 << 20).map(|(_, n)| *n as u64).collect::<Vec<_>>();
        let rounding = large.iter().map(|n| n.div_ceil(2 << 20) * (2 << 20) - n).sum::<u64>();
        let last = arena.views().values().map(|v| v.offset() + v.bytes()).max().unwrap();
        format!("before {} allocations / {bytes} B, {} > 1 MiB / {} B, SM120 rounding {rounding} B; \
            after arena {} B, padding {} B, tail {} B", tensors.len(), large.len(), large.iter().sum::<u64>(),
            arena.bytes(), arena.padding_bytes(), arena.bytes() - last)
    }

    /// Views are exactly the final owners: unique, 256-aligned, disjoint, and
    /// summing to the admitted resident bytes; composites stay one view.
    fn assert_arena(layout: &RankWeightLayout) -> crate::serving_capacity::WeightArenaLayout {
        let arena = layout.arena_layout("step5/test", |_| true).unwrap();
        assert_eq!(arena.payload_bytes() as u64, layout.resident_bytes);
        assert_eq!(arena.views().len(), layout.operands.len() + 3 * layout.core.len());
        let mut views = arena.views().iter().collect::<Vec<_>>();
        views.sort_by_key(|(_, v)| v.offset());
        let mut end = 0;
        for (name, view) in &views {
            assert_eq!(view.offset() % 256, 0, "{name}");
            assert!(view.offset() >= end && view.offset() - end < 256, "{name}");
            assert!(!name.contains("staging") && !name.contains("marker"), "{name}");
            end = view.offset() + view.bytes();
        }
        assert_eq!(end, arena.bytes());
        for operand in &layout.operands {
            let key = if operand.name == "model.embed_tokens.weight" { "embedding.table" } else { &operand.name };
            assert_eq!(arena.view(key).unwrap().bytes() as u64, operand.allocation_bytes, "{key}");
        }
        for core in layout.core.values() {
            let parts = ["trellis", "suh", "svh"].map(|p| arena.view(&format!("{}.{p}", core.stem)).unwrap().bytes());
            assert_eq!(parts.iter().sum::<usize>(), core.bytes, "{}", core.stem);
        }
        arena
    }

    #[test]
    fn rank_arenas_hold_exactly_the_final_fp8_owners() {
        let dir = snapshot("fp8");
        let cp = Checkpoint::open(dir.path()).unwrap();
        let one = RankWeightLayout::from_checkpoint(&cp, 0, 1).unwrap();
        let arena = assert_arena(&one);
        assert!(arena.views().keys().all(|name| !name.ends_with("weight_scale_inv")));
        eprintln!("Step-5 FP8 TP1 rank 0: {}", census(&one, &arena));
        // FP8-compact attention: each converted O keeps values and scales as one
        // composite view, sized by its packed native layout.
        let mut compact = one.clone();
        compact.select_core_view(&one.operands.iter().filter(|op| op.name.ends_with(".self_attn.o_proj.weight")).map(|op| op.name.clone()).collect::<Vec<_>>()).unwrap();
        let arena = assert_arena(&compact);
        let converted = compact.operands.iter().filter(|o| o.dtype == DType::F8E4M3).collect::<Vec<_>>();
        assert!(!converted.is_empty() && converted.iter().all(|o| o.name.ends_with(".self_attn.o_proj.weight")));
        for operand in converted {
            assert_eq!(arena.view(&operand.name).unwrap().bytes() as u64,
                operand.native_layout(0, 1).unwrap().bytes().unwrap(), "{}", operand.name);
        }
        eprintln!("Step-5 FP8-compact TP1 rank 0: {}", census(&compact, &arena));
        let ranks = [0, 1].map(|rank| RankWeightLayout::from_checkpoint(&cp, rank, 2).unwrap());
        for (rank, layout) in ranks.iter().enumerate() {
            let arena = assert_arena(layout);
            eprintln!("Step-5 FP8 TP2 rank {rank}: {}", census(layout, &arena));
        }
        // TP2 holds rank-local slices: q rows halve, replicated FFNs keep whole extents.
        let q = |l: &RankWeightLayout| l.operands.iter().find(|o| o.name == "model.layers.3.self_attn.q_proj.weight")
            .unwrap().allocation_bytes;
        assert_eq!(q(&ranks[0]) * 2, q(&one));
        assert_eq!(q(&ranks[0]), q(&ranks[1]));
    }

    #[test]
    fn rank_and_predictor_arenas_hold_exactly_the_final_exl3_owners() {
        let dir = crate::plan::families::step5::tests::exl3_snapshot();
        let cp = Checkpoint::coordinator(dir.path(), false, true).unwrap();
        let one = RankWeightLayout::from_checkpoint(&cp, 0, 1).unwrap();
        assert!(!one.core.is_empty());
        let arena = assert_arena(&one);
        eprintln!("Step-5 EXL3 TP1 rank 0: {}", census(&one, &arena));
        for rank in 0..2 {
            let layout = RankWeightLayout::from_checkpoint(&cp, rank, 2).unwrap();
            let arena = assert_arena(&layout);
            eprintln!("Step-5 EXL3 TP2 rank {rank}: {}", census(&layout, &arena));
            // A K-split o_proj owner keeps only this rank's input columns.
            let o = &layout.core["model.layers.3.self_attn.o_proj.weight"];
            assert_eq!((o.rank, o.ranks), (rank, 2));
        }
        // The predictor arena is its own scope: no target layer or global appears in it.
        let mtp = RankWeightLayout::for_mtp(&cp, 3).unwrap();
        let predictor = assert_arena(&mtp);
        let cfg = crate::plan::families::step5::Config::parse(&cp).unwrap();
        assert!(predictor.views().keys().all(|name| name.strip_prefix("model.layers.")
            .and_then(|rest| rest.split('.').next()).and_then(|l| l.parse::<usize>().ok())
            .is_some_and(|layer| layer >= cfg.layers)), "predictor arena holds a target owner");
        eprintln!("Step-5 EXL3 MTP depth 3: {}", census(&mtp, &predictor));
    }
}
