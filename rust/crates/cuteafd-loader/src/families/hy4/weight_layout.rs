//! Packed coordinator weight arenas for Hy4 (`hy_v4_vl`).
//!
//! One preplanned [`WeightArenaLayout`] per physical coordinator GPU holds the
//! target core and globals; tensors are aligned views inside it. Optional
//! routed-expert onboarding, DFly2 drafter, and 40-layer ViT vision tower each
//! own separate arenas.
use std::collections::BTreeMap;

use anyhow::{ensure, Result};

use crate::families::hy4_common::{add_arena_matrix, core_split, HEAD_STEM};
use crate::formats::native_attention::{Fp8Role, NativeLayout};
use crate::formats::trellis_core::{CoreContract, CoreProjection, CoreSplit, CoreWeightLayout};
use crate::placement::LayerMode;
use crate::plan::checkpoint::Checkpoint;
use crate::serving_capacity::WeightArenaLayout;

use super::Hy4Config;

pub const ROPE_INV_FREQ: &str = "hy4.rope.inv_freq";

/// Both the original and compact target recipes use the shared windowed
/// mount reader. Expert shards have separate owners.
pub fn target_mount_load(
    checkpoint: &Checkpoint,
    gpus: usize,
) -> Result<crate::shared::dense_read::MountLoad> {
    crate::shared::dense_read::MountLoad::from_headers(
        checkpoint
            .tensors
            .iter()
            .map(|t| &t.meta)
            .filter(|h| !h.name.contains(".mlp.experts.")),
        gpus,
    )
}

/// Core projections eligible for K8/K6 EXL3 MUL1 compact views in Hy4
/// (`origin/quant/hy4-k4-mc:quantization/hy4/recipe.py::reviewed_core_modules`).
pub fn core_contract(stem: &str) -> Option<CoreContract> {
    if stem == HEAD_STEM {
        return Some(CoreContract {
            family: "Hy4",
            widths: &[6, 8],
            split: CoreSplit::Output,
            padded_outputs: false,
            view_varying: true,
        });
    }
    if stem.starts_with("model.dfly_layers.") {
        let leaf = stem.rsplit_once('.').map(|(_, l)| l)?;
        let eligible = matches!(
            leaf,
            "q_a_proj"
                | "q_b_proj"
                | "kv_a_proj_with_mqa"
                | "kv_b_proj"
                | "linear_gate"
                | "o_proj"
                | "gate_proj"
                | "up_proj"
                | "down_proj"
        );
        return eligible.then_some(CoreContract {
            family: "Hy4 DFly2 drafter",
            widths: &[6, 8],
            split: CoreSplit::Whole,
            padded_outputs: leaf == "kv_a_proj_with_mqa",
            view_varying: false,
        });
    }
    let rest = stem.strip_prefix("model.layers.")?;
    let (_, tail) = rest.split_once('.')?;
    let eligible = matches!(
        tail,
        "self_attn.q_a_proj"
            | "self_attn.q_b_proj"
            | "self_attn.kv_a_proj_with_mqa"
            | "self_attn.kv_b_proj"
            | "self_attn.linear_gate"
            | "self_attn.o_proj"
            | "mlp.gate_proj"
            | "mlp.up_proj"
            | "mlp.down_proj"
            | "mlp.shared_experts.gate_proj"
            | "mlp.shared_experts.up_proj"
            | "mlp.shared_experts.down_proj"
    );
    eligible.then(|| CoreContract {
        family: "Hy4",
        widths: &[6, 8],
        split: core_split(stem),
        padded_outputs: tail == "self_attn.kv_a_proj_with_mqa",
        view_varying: true,
    })
}

/// Block-FP8 role classification for runtime-derived `fp8-compact`.
pub fn fp8_role(name: &str) -> Fp8Role {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    if stem == HEAD_STEM {
        return Fp8Role::exact("vocabulary head", true);
    }
    if stem.starts_with("model.dfly_layers.") {
        return Fp8Role::exact("dfly2 drafter", false);
    }
    if stem.ends_with(".self_attn.kv_b_proj") {
        // Absorbed MLA keeps W_UK and W_UV in BF16 inside the packed target arena.
        return Fp8Role::exact("hy4 absorbed kv_b_proj", false);
    }
    if stem.ends_with(".self_attn.kv_a_proj_with_mqa") {
        return Fp8Role::padded("hy4 kv_a_proj_with_mqa", true, 128);
    }
    if core_contract(stem).is_some() {
        return Fp8Role::exact("hy4 core projection", true);
    }
    Fp8Role::exact("non-core projection", false)
}

/// One coordinator rank's immutable packed target arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinatorWeights {
    pub rank: usize,
    pub layers: Vec<usize>,
    pub arena: WeightArenaLayout,
    pub core: BTreeMap<String, CoreWeightLayout>,
    pub fp8: BTreeMap<String, NativeLayout>,
}

impl CoordinatorWeights {
    pub fn weight_components(
        &self,
        maximum: u64,
    ) -> Result<Vec<crate::placement::planner::Component>, crate::placement::planner::Refusal> {
        let location = crate::placement::planner::Location::coordinator(self.rank as u8);
        crate::placement::planner::weight_arena_components(
            &self.arena,
            location,
            crate::placement::planner::Lifetime::RESIDENT,
            maximum,
            cuteafd_core::memory_layout::MemoryKind::Weights,
        )
    }

    pub fn has_bf16_head(&self) -> bool {
        self.arena.views().contains_key(crate::families::hy4_common::HEAD_WEIGHT)
    }

    pub fn native_core_scratch(
        &self,
        rows: usize,
    ) -> Result<Option<crate::formats::trellis_core::ProjectionScratchLayout>> {
        let mut combined: Option<crate::formats::trellis_core::ProjectionScratchLayout> = None;
        for (stem, layout) in &self.core {
            if stem == HEAD_STEM {
                continue;
            }
            let [n, k] = layout.shape;
            let one = crate::formats::trellis_core::ProjectionScratchLayout::new(k, n, rows)?;
            match &mut combined {
                Some(acc) => acc.merge(&one)?,
                None => combined = Some(one),
            }
        }
        for stem in self.fp8.keys() {
            if stem == HEAD_STEM {
                continue;
            }
            let one = crate::formats::trellis_core::ProjectionScratchLayout {
                rows,
                extents: BTreeMap::from([("fp8", 256)]),
            };
            match &mut combined {
                Some(acc) => acc.merge(&one)?,
                None => combined = Some(one),
            }
        }
        Ok(combined)
    }

    pub fn native_head_scratch(
        &self,
        head_rows: usize,
    ) -> Result<Option<crate::formats::trellis_core::ProjectionScratchLayout>> {
        if let Some(layout) = self.core.get(HEAD_STEM) {
            let [n, k] = layout.shape;
            return Ok(Some(crate::formats::trellis_core::ProjectionScratchLayout::new(
                k, n, head_rows,
            )?));
        }
        if self.fp8.contains_key(HEAD_STEM) {
            return Ok(Some(crate::formats::trellis_core::ProjectionScratchLayout {
                rows: head_rows,
                extents: BTreeMap::from([("fp8", 256)]),
            }));
        }
        Ok(None)
    }

    pub fn trellis_scratch(
        &self,
        prefill_rows: usize,
        decode_rows: usize,
    ) -> Result<Option<crate::formats::trellis_core::ProjectionScratchLayout>> {
        let mut combined: Option<crate::formats::trellis_core::ProjectionScratchLayout> = None;
        let rows = prefill_rows.max(decode_rows);
        for layout in self.core.values() {
            let [n, k] = layout.shape;
            let one = crate::formats::trellis_core::ProjectionScratchLayout::new(k, n, rows)?;
            match &mut combined {
                Some(acc) => acc.merge(&one)?,
                None => combined = Some(one),
            }
        }
        Ok(combined)
    }
}

/// Build the exact packed target arena for every coordinator rank (`1` or `2`)
/// from the checkpoint headers and resolved per-layer modes.
pub fn target_arenas(
    checkpoint: &Checkpoint,
    cfg: &Hy4Config,
    core_projections: &BTreeMap<String, CoreProjection>,
    modes: &[LayerMode],
    gpus: usize,
) -> Result<Vec<CoordinatorWeights>> {
    ensure!(matches!(gpus, 1 | 2), "Hy4 supports 1 or 2 coordinator GPUs, not {gpus}");
    ensure!(modes.len() == cfg.layers, "Hy4 layer modes length mismatch");

    let mut out = Vec::with_capacity(gpus);
    let all_split = modes.iter().all(|m| matches!(m, LayerMode::HeadSplit | LayerMode::ContextSplit));
    let head_rank = if all_split { 0 } else { gpus - 1 };

    for rank in 0..gpus {
        let mut views: Vec<(String, usize)> = Vec::new();
        let mut core_map = BTreeMap::new();
        let mut fp8_map = BTreeMap::new();
        let mut owned_layers = Vec::new();

        // Globals: embed_tokens on rank 0, hc_head + norm on head_rank,
        // lm_head output-sharded across ranks on 2-GPU split or whole on head_rank.
        if rank == 0 {
            views.push(("model.embed_tokens.weight".into(), cfg.vocab_size * cfg.hidden * 2));
        }
        if rank == head_rank {
            views.push(("model.hc_head.hc_head_fn".into(), cfg.hc_mult * cfg.hc_mult * cfg.hidden * 4));
            views.push(("model.hc_head.hc_head_base".into(), cfg.hc_mult * 4));
            views.push(("model.hc_head.hc_head_scale".into(), 4));
            views.push(("model.norm.weight".into(), cfg.hidden * 2));
        }
        let split_head = gpus == 2 && all_split;
        if split_head || rank == head_rank {
            let head_ranks = if split_head { 2 } else { 1 };
            let head_r = if split_head { rank } else { 0 };
            add_arena_matrix(
                checkpoint,
                core_projections,
                fp8_role,
                HEAD_STEM,
                [cfg.vocab_size, cfg.hidden],
                CoreSplit::Output,
                head_r,
                head_ranks,
                &mut views,
                &mut core_map,
                &mut fp8_map,
            )?;
        }

        for (layer, &mode) in modes.iter().enumerate() {
            let (layer_ranks, layer_rank) = match mode {
                LayerMode::Whole { gpu, .. } if usize::from(gpu) == rank => (1usize, 0usize),
                LayerMode::Whole { .. } => continue,
                LayerMode::HeadSplit | LayerMode::ContextSplit => (2usize, rank),
            };
            owned_layers.push(layer);
            let p = format!("model.layers.{layer}");
            let h = cfg.hidden;
            let hc = cfg.hc_mult;
            let local_heads = cfg.heads / layer_ranks;

            // Layer norms & 4-stream iHC (F32). No sandwich norms in Hy4.
            views.push((format!("{p}.input_layernorm.weight"), h * 2));
            views.push((format!("{p}.post_attention_layernorm.weight"), h * 2));
            for tag in ["hc_attn_layer", "hc_mlp_layer"] {
                views.push((format!("{p}.{tag}.hc_pre.hc_fn"), 2 * hc * hc * h * 4));
                views.push((format!("{p}.{tag}.hc_pre.hc_base"), 2 * hc * 4));
                views.push((format!("{p}.{tag}.hc_pre.hc_scale"), 2 * 4));
            }

            // Attention norms + learnable F32 sinks (split with heads under HeadSplit/ContextSplit)
            views.push((format!("{p}.self_attn.q_a_layernorm.weight"), cfg.q_lora_rank * 2));
            views.push((format!("{p}.self_attn.kv_a_layernorm.weight"), cfg.kv_lora_rank * 2));
            views.push((format!("{p}.self_attn.learnable_sink_param"), local_heads * 4));

            // Absorbed MLA keeps `kv_b_proj` unpacked as BF16 W_UK+W_UV across all views.
            views.push((
                format!("{p}.self_attn.kv_b_proj.weight"),
                local_heads * (cfg.qk_nope_dim + cfg.v_head_dim) * cfg.kv_lora_rank * 2,
            ));
            let attn_projs: [(&str, [usize; 2], CoreSplit); 5] = [
                ("self_attn.q_a_proj", [cfg.q_lora_rank, h], CoreSplit::Whole),
                ("self_attn.q_b_proj", [cfg.heads * cfg.qk_head_dim(), cfg.q_lora_rank], CoreSplit::Output),
                ("self_attn.kv_a_proj_with_mqa", [cfg.kv_lora_rank + cfg.qk_rope_dim, h], CoreSplit::Whole),
                ("self_attn.linear_gate", [cfg.heads * cfg.v_head_dim, h], CoreSplit::Output),
                ("self_attn.o_proj", [h, cfg.heads * cfg.v_head_dim], CoreSplit::Input),
            ];
            for (sub, shape, split) in attn_projs {
                let stem = format!("{p}.{sub}");
                add_arena_matrix(
                    checkpoint,
                    core_projections,
                    fp8_role,
                    &stem,
                    shape,
                    split,
                    layer_rank,
                    layer_ranks,
                    &mut views,
                    &mut core_map,
                    &mut fp8_map,
                )?;
            }

            // 32-head per-token DSA indexer (only on the 21 full IndexCache layers;
            // the 57 shared layers reuse the preceding full layer's selection).
            if cfg.is_full_indexer(layer) {
                let d = cfg.indexer_head_dim;
                let n = cfg.indexer_heads;
                let ip = format!("{p}.self_attn.indexer");
                views.push((format!("{ip}.wq_b.weight"), n * d * cfg.q_lora_rank * 2));
                views.push((format!("{ip}.wk.weight"), d * h * 2));
                views.push((format!("{ip}.k_norm.weight"), d * 2));
                views.push((format!("{ip}.k_norm.bias"), d * 2));
                views.push((format!("{ip}.weights_proj.weight"), n * h * 2));
            }

            // FFN: dense layer 0 or direct MoE layers 1..=77
            if cfg.dense_layer[layer] {
                for (sub, shape, split) in [
                    ("mlp.gate_proj", [cfg.intermediate, h], CoreSplit::Output),
                    ("mlp.up_proj", [cfg.intermediate, h], CoreSplit::Output),
                    ("mlp.down_proj", [h, cfg.intermediate], CoreSplit::Input),
                ] {
                    add_arena_matrix(
                        checkpoint,
                        core_projections,
                        fp8_role,
                        &format!("{p}.{sub}"),
                        shape,
                        split,
                        layer_rank,
                        layer_ranks,
                        &mut views,
                        &mut core_map,
                        &mut fp8_map,
                    )?;
                }
            } else {
                let shared = cfg.moe_intermediate * cfg.shared_experts;
                for (sub, shape, split) in [
                    ("mlp.shared_experts.gate_proj", [shared, h], CoreSplit::Output),
                    ("mlp.shared_experts.up_proj", [shared, h], CoreSplit::Output),
                    ("mlp.shared_experts.down_proj", [h, shared], CoreSplit::Input),
                ] {
                    add_arena_matrix(
                        checkpoint,
                        core_projections,
                        fp8_role,
                        &format!("{p}.{sub}"),
                        shape,
                        split,
                        layer_rank,
                        layer_ranks,
                        &mut views,
                        &mut core_map,
                        &mut fp8_map,
                    )?;
                }
                views.push((format!("{p}.mlp.gate.weight"), cfg.experts * h * 2));
                views.push((format!("{p}.mlp.gate.e_score_correction_bias"), cfg.experts * 4));
            }
        }

        // RoPE inverse frequency constants live inside the target arena (never standalone allocations).
        views.push((ROPE_INV_FREQ.into(), (cfg.qk_rope_dim / 2) * 4));

        let arena = WeightArenaLayout::new(format!("hy4/target/gpu{rank}"), views)?;
        out.push(CoordinatorWeights {
            rank,
            layers: owned_layers,
            arena,
            core: core_map,
            fp8: fp8_map,
        });
    }
    Ok(out)
}
