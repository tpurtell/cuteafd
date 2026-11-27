//! Packed coordinator weight arenas for Hy4-flash (`hy_v4_flash`).
//!
//! One preplanned [`WeightArenaLayout`] per physical coordinator GPU holds the
//! target core and globals; tensors are aligned views inside it. Optional
//! routed-expert onboarding, DFly2 drafter, and ViT vision tower each own
//! separate arenas.
use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};

use crate::formats::dense_trellis::TrellisSide;
use crate::formats::native_attention::{Fp8Role, NativeLayout};
use crate::formats::trellis_core::{CoreContract, CoreProjection, CoreSplit, CoreWeightLayout};
use crate::placement::LayerMode;
use crate::plan::checkpoint::Checkpoint;
use crate::serving_capacity::WeightArenaLayout;

use super::Hy4FlashConfig;

pub const HEAD_STEM: &str = "lm_head";
pub const HEAD_WEIGHT: &str = "lm_head.weight";
pub const ROPE_INV_FREQ_DSA: &str = "hy4_flash.rope.inv_freq_dsa";
pub const ROPE_INV_FREQ_SWA: &str = "hy4_flash.rope.inv_freq_swa";

/// Classify whether a Hy4-flash core matrix is head/intermediate-splittable
/// across two coordinator GPUs under `HeadSplit` / `ContextSplit`.
///
/// - Output-split (rows divided by 2): `q_b_proj`, `kv_b_proj`, `linear_gate`,
///   dense `mlp.{gate,up}_proj`, `shared_experts.{gate,up}_proj`, `lm_head`.
/// - Input-split (cols divided by 2): `o_proj`, dense `mlp.down_proj`,
///   `shared_experts.down_proj`.
/// - Whole (replicated on each rank): `q_a_proj`, `kv_a_proj_with_mqa`,
///   `routed_expert_down_proj`, `routed_expert_up_proj`, `mlp.gate`,
///   compressed indexer projections, Engram small projections, norms, iHC.
pub fn core_split(stem: &str) -> CoreSplit {
    if stem == HEAD_STEM {
        return CoreSplit::Output;
    }
    if let Some(leaf) = stem.rsplit_once('.').map(|(_, l)| l) {
        if stem.contains(".self_attn.") {
            return match leaf {
                "q_b_proj" | "kv_b_proj" | "linear_gate" => CoreSplit::Output,
                "o_proj" => CoreSplit::Input,
                _ => CoreSplit::Whole,
            };
        }
        if stem.contains(".mlp.shared_experts.")
            || (stem.contains(".mlp.")
                && !stem.contains(".experts.")
                && !stem.contains("routed_expert_")
                && !stem.ends_with(".mlp.gate")
                && !stem.contains(".mlp.gate."))
        {
            return match leaf {
                "gate_proj" | "up_proj" => CoreSplit::Output,
                "down_proj" => CoreSplit::Input,
                _ => CoreSplit::Whole,
            };
        }
    }
    CoreSplit::Whole
}

/// Core projections eligible for K8/K6 EXL3 compact views in Hy4-flash.
pub fn core_contract(stem: &str) -> Option<CoreContract> {
    if stem == HEAD_STEM {
        return Some(CoreContract {
            family: "Hy4-flash",
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
            "q_a_proj" | "q_b_proj" | "kv_a_proj_with_mqa" | "kv_b_proj"
                | "linear_gate" | "o_proj" | "gate_proj" | "up_proj" | "down_proj"
        );
        let split = match leaf {
            "q_b_proj" | "kv_b_proj" | "linear_gate" | "gate_proj" | "up_proj" => CoreSplit::Output,
            "o_proj" | "down_proj" => CoreSplit::Input,
            _ => CoreSplit::Whole,
        };
        return eligible.then_some(CoreContract {
            family: "Hy4-flash DFly2 drafter",
            widths: &[6, 8],
            split,
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
            | "mlp.routed_expert_down_proj"
            | "mlp.routed_expert_up_proj"
    );
    eligible.then(|| CoreContract {
        family: "Hy4-flash",
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

/// Both the original and compact target recipes use the shared windowed
/// mount reader. Expert and lookup-table shards have separate owners.
pub fn target_mount_load(checkpoint: &Checkpoint, gpus: usize) -> Result<crate::shared::dense_read::MountLoad> {
    crate::shared::dense_read::MountLoad::from_headers(
        checkpoint.tensors.iter().map(|t| &t.meta).filter(|h| {
            !h.name.contains(".mlp.experts.") && !h.name.contains(".multi_head_embedding.embedding.")
        }),
        gpus,
    )
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
    pub fn weight_components(&self, maximum: u64) -> Result<Vec<crate::placement::planner::Component>, crate::placement::planner::Refusal> {
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
        self.arena.views().contains_key(HEAD_WEIGHT)
    }

    pub fn native_core_scratch(&self, rows: usize)
        -> Result<Option<crate::formats::trellis_core::ProjectionScratchLayout>> {
        let mut combined: Option<crate::formats::trellis_core::ProjectionScratchLayout> = None;
        for (stem, layout) in &self.core {
            if stem == HEAD_STEM {
                continue;
            }
            let one = crate::formats::trellis_core::ProjectionScratchLayout::for_weight(layout, rows)?;
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

    pub fn native_head_scratch(&self, head_rows: usize)
        -> Result<Option<crate::formats::trellis_core::ProjectionScratchLayout>> {
        if let Some(layout) = self.core.get(HEAD_STEM) {
            return Ok(Some(crate::formats::trellis_core::ProjectionScratchLayout::for_weight(layout, head_rows)?));
        }
        if self.fp8.contains_key(HEAD_STEM) {
            return Ok(Some(crate::formats::trellis_core::ProjectionScratchLayout {
                rows: head_rows,
                extents: BTreeMap::from([("fp8", 256)]),
            }));
        }
        Ok(None)
    }

    pub fn trellis_scratch(&self, prefill_rows: usize, decode_rows: usize)
        -> Result<Option<crate::formats::trellis_core::ProjectionScratchLayout>> {
        let mut combined: Option<crate::formats::trellis_core::ProjectionScratchLayout> = None;
        let rows = prefill_rows.max(decode_rows);
        for layout in self.core.values() {
            let one = crate::formats::trellis_core::ProjectionScratchLayout::for_weight(layout, rows)?;
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
    cfg: &Hy4FlashConfig,
    core_projections: &BTreeMap<String, CoreProjection>,
    modes: &[LayerMode],
    gpus: usize,
) -> Result<Vec<CoordinatorWeights>> {
    ensure!(matches!(gpus, 1 | 2), "Hy4-flash supports 1 or 2 coordinator GPUs, not {gpus}");
    ensure!(modes.len() == cfg.layers, "Hy4-flash layer modes length mismatch");

    let mut out = Vec::with_capacity(gpus);
    let head_rank = if modes.iter().all(|m| matches!(m, LayerMode::HeadSplit | LayerMode::ContextSplit)) {
        0
    } else {
        gpus - 1
    };

    for rank in 0..gpus {
        let mut views: Vec<(String, usize)> = Vec::new();
        let mut core_map = BTreeMap::new();
        let mut fp8_map = BTreeMap::new();
        let mut owned_layers = Vec::new();

        // Globals: embed_tokens on rank 0 (or replicated if head-split), hc_head + norm on head_rank,
        // lm_head sharded across ranks on 2-GPU head/context split or whole on head_rank.
        if rank == 0 {
            views.push(("model.embed_tokens.weight".into(), cfg.vocab_size * cfg.hidden * 2));
        }
        if rank == head_rank {
            views.push(("model.hc_head.hc_head_fn".into(), cfg.hc_mult * cfg.hc_mult * cfg.hidden * 4));
            views.push(("model.hc_head.hc_head_base".into(), cfg.hc_mult * 4));
            views.push(("model.hc_head.hc_head_scale".into(), 4));
            views.push(("model.norm.weight".into(), cfg.hidden * 2));
        }
        let split_head = gpus == 2 && modes.iter().all(|m| matches!(m, LayerMode::HeadSplit | LayerMode::ContextSplit));
        if split_head || rank == head_rank {
            let head_ranks = if split_head { 2 } else { 1 };
            let head_r = if split_head { rank } else { 0 };
            add_matrix(
                checkpoint,
                core_projections,
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
            let attn = cfg.attention(layer);

            // Norms & iHC (F32)
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
            views.push((format!("{p}.self_attn.post_proj_norm.weight"), h * 2));
            let local_heads = attn.heads / layer_ranks;
            views.push((format!("{p}.self_attn.learnable_sink_param"), local_heads * 4));

            // Attention linear projections (`kv_b_proj` lives absorbed as BF16 W_UK+W_UV in the arena
            // across all MC views; compact EXL3 checkpoints dequantize it via transient storage at load).
            views.push((
                format!("{p}.self_attn.kv_b_proj.weight"),
                local_heads * (attn.qk_nope_dim + attn.v_head_dim) * cfg.kv_lora_rank * 2,
            ));
            let attn_projs: [(&str, [usize; 2], CoreSplit); 5] = [
                ("self_attn.q_a_proj", [cfg.q_lora_rank, h], CoreSplit::Whole),
                ("self_attn.q_b_proj", [attn.heads * attn.qk_head_dim(), cfg.q_lora_rank], CoreSplit::Output),
                ("self_attn.kv_a_proj_with_mqa", [cfg.kv_lora_rank + attn.qk_rope_dim, h], CoreSplit::Whole),
                ("self_attn.linear_gate", [attn.heads * attn.v_head_dim, h], CoreSplit::Output),
                ("self_attn.o_proj", [h, attn.heads * attn.v_head_dim], CoreSplit::Input),
            ];
            for (sub, shape, split) in attn_projs {
                let stem = format!("{p}.{sub}");
                add_matrix(
                    checkpoint, core_projections, &stem, shape, split,
                    layer_rank, layer_ranks, &mut views, &mut core_map, &mut fp8_map,
                )?;
            }

            // Compressed indexer (DSA layers only, kept native & replicated on owners)
            if attn.sparse {
                let d = cfg.indexer_head_dim;
                let n = cfg.indexer_heads;
                let ip = format!("{p}.self_attn.compressed_indexer");
                views.push((format!("{ip}.query_projection.weight"), n * d * cfg.q_lora_rank * 2));
                views.push((format!("{ip}.head_weight_projection.weight"), n * h * 2));
                views.push((format!("{ip}.token_key_projection.weight"), d * h * 2));
                views.push((format!("{ip}.token_key_norm.weight"), d * 2));
                views.push((format!("{ip}.key_norm.weight"), d * 4));
                views.push((format!("{ip}.key_value_projection.weight"), d * d * 4));
                views.push((format!("{ip}.key_gate_projection.weight"), d * d * 4));
                views.push((format!("{ip}.intra_group_ape"), cfg.indexer_compress_ratio * d * 4));
            }

            // FFN: dense layer 0 or LatentMoE layers 1..41
            if cfg.dense_layer[layer] {
                for (sub, shape, split) in [
                    ("mlp.gate_proj", [cfg.intermediate, h], CoreSplit::Output),
                    ("mlp.up_proj", [cfg.intermediate, h], CoreSplit::Output),
                    ("mlp.down_proj", [h, cfg.intermediate], CoreSplit::Input),
                ] {
                    add_matrix(
                        checkpoint, core_projections, &format!("{p}.{sub}"), shape, split,
                        layer_rank, layer_ranks, &mut views, &mut core_map, &mut fp8_map,
                    )?;
                }
                views.push((format!("{p}.mlp.post_fc2_norm.weight"), h * 2));
            } else {
                let shared = cfg.moe_intermediate * cfg.shared_experts;
                for (sub, shape, split) in [
                    ("mlp.shared_experts.gate_proj", [shared, h], CoreSplit::Output),
                    ("mlp.shared_experts.up_proj", [shared, h], CoreSplit::Output),
                    ("mlp.shared_experts.down_proj", [h, shared], CoreSplit::Input),
                    ("mlp.routed_expert_down_proj", [cfg.routed_expert_hidden, h], CoreSplit::Whole),
                    ("mlp.routed_expert_up_proj", [h, cfg.routed_expert_hidden], CoreSplit::Whole),
                ] {
                    add_matrix(
                        checkpoint, core_projections, &format!("{p}.{sub}"), shape, split,
                        layer_rank, layer_ranks, &mut views, &mut core_map, &mut fp8_map,
                    )?;
                }
                views.push((format!("{p}.mlp.gate.weight"), cfg.experts * h * 2));
                views.push((format!("{p}.mlp.gate.e_score_correction_bias"), cfg.experts * 4));
                views.push((format!("{p}.mlp.combine_output_norm.weight"), h * 2));
            }

            // Engram small GPU projections/norms (packed contiguously inside target arena; table itself stays host-mapped)
            if cfg.engram_layers.contains(&layer) {
                let ep = format!("{p}.engram_layer");
                let w = cfg.engram_head_width();
                let concat = cfg.engram_concat_width();
                views.push((format!("{ep}.unified_pre_gamma"), h * 2));
                views.push((format!("{ep}.value_proj.weight"), h * concat * 2));
                views.push((format!("{ep}.value_proj.bias"), h * 2));
                views.push((format!("{ep}.lookup_per_head_norms.packed"), cfg.engram_total_heads() * w * 2));
                views.push((format!("{ep}.key_projs.packed.weight"), hc * h * concat * 2));
                views.push((format!("{ep}.key_projs.packed.bias"), hc * h * 2));
                views.push((format!("{ep}.norm_key.packed"), hc * h * 2));
                views.push((format!("{ep}.norm_query.packed"), hc * h * 2));
                views.push((format!("{ep}.unified_per_hc_scales.packed"), hc * h * 2));
            }
        }

        // RoPE inverse frequency constants live inside the target arena (never standalone allocations).
        views.push((ROPE_INV_FREQ_DSA.into(), (cfg.qk_rope_dim / 2) * 4));
        views.push((ROPE_INV_FREQ_SWA.into(), (cfg.swa_qk_rope_dim / 2) * 4));

        let arena = WeightArenaLayout::new(format!("hy4_flash/target/gpu{rank}"), views)?;
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

#[allow(clippy::too_many_arguments)]
fn add_matrix(
    checkpoint: &Checkpoint,
    core_projections: &BTreeMap<String, CoreProjection>,
    stem: &str,
    shape: [usize; 2],
    split: CoreSplit,
    rank: usize,
    ranks: usize,
    views: &mut Vec<(String, usize)>,
    core_map: &mut BTreeMap<String, CoreWeightLayout>,
    fp8_map: &mut BTreeMap<String, NativeLayout>,
) -> Result<()> {
    let effective_ranks = if split == CoreSplit::Whole { 1 } else { ranks };
    let effective_rank = if split == CoreSplit::Whole { 0 } else { rank };

    if let Some(proj) = core_projections.get(stem) {
        let side = if effective_ranks == 1 {
            TrellisSide::Output
        } else {
            match split {
                CoreSplit::Input => TrellisSide::Input,
                CoreSplit::Output | CoreSplit::Whole => TrellisSide::Output,
            }
        };
        let layout = if stem == HEAD_STEM && effective_ranks == 2 {
            CoreWeightLayout::output_shard(proj, effective_rank)?
        } else {
            CoreWeightLayout::new(proj, side, effective_rank, effective_ranks)?
        };
        views.extend(layout.arena_tensors(stem)?);
        core_map.insert(stem.to_owned(), layout);
        return Ok(());
    }

    let weight_name = format!("{stem}.weight");
    let [mut n, mut k] = shape;
    if effective_ranks == 2 {
        match split {
            CoreSplit::Output => {
                if stem == HEAD_STEM {
                    n = crate::shared::vocabulary::shard(n, 128, effective_rank)?.len();
                } else {
                    ensure!(n % 2 == 0, "{stem}: output dim {n} does not divide TP2");
                    n /= 2;
                }
            }
            CoreSplit::Input => {
                ensure!(k % 2 == 0, "{stem}: input dim {k} does not divide TP2");
                k /= 2;
            }
            CoreSplit::Whole => {}
        }
    }

    let role = fp8_role(&weight_name);
    if checkpoint.converts_to_fp8(&weight_name) && role.supported {
        let [stored_n, stored_k] = role.stored_shape([n, k])?;
        let payload = stored_n * stored_k;
        let scales = ((stored_n / 128) * (stored_k / 128) * 4).max(256);
        views.push((format!("{stem}.payload"), payload));
        views.push((format!("{stem}.scales"), scales));
        fp8_map.insert(
            stem.to_owned(),
            NativeLayout {
                stem: stem.to_owned(),
                shape: [stored_n, stored_k],
                fp8: true,
                rank: effective_rank,
                ranks: effective_ranks,
                split,
            },
        );
        return Ok(());
    }

    views.push((weight_name, n.checked_mul(k).and_then(|v| v.checked_mul(2))
        .with_context(|| format!("{stem}: BF16 weight size overflow"))?));
    Ok(())
}
