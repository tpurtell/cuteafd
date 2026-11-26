//! Bundled Hy4 5-layer DFly2 drafter checkpoint geometry and separately retired
//! weight & workspace arenas (#2550).
//!
//! Geometry (`tencent/Hy4` / `wrldsuksgo2mars/Hy4-EXL3-K4-MC`):
//! - 5 dense absorbed-MLA layers tapping `[61, 65, 69, 73, 77]` (all full IndexCache layers)
//! - `hidden=6144`, `heads=64`, `q_lora_rank=2048`, `kv_lora_rank=512`, `mlp=18432`
//! - `block_size=7` (1 anchor/bonus token + 6 speculative masks), `topk=2048`
//! - No `post_proj_norm` (matches Hy4 target); K8 MUL1 views reuse target dense/MLA trellis exports.
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;

use crate::formats::dense_trellis::TrellisSide;
use crate::formats::exl3_storage::Exl3Codebook;
use crate::formats::trellis_core::{CoreProjection, CoreSplit, CoreWeightLayout};
use crate::plan::checkpoint::Checkpoint;
use crate::serving_capacity::WeightArenaLayout;

use super::execution::KV_A_PADDED;
use super::{Hy4Config, Hy4DflyConfig, Hy4KvCache};

pub const PREFIX: &str = "model.dfly_layers.";
pub const MLP_WIDTH: usize = 18_432;
pub const HEADS: usize = 64;
/// Exact `hy4_draft_attention_{bf16,fp8}` per-slot scratch at 16 splits:
/// `7 rows/slot * 64 heads * 16 splits * (512 * 2 + 4)` = 7,368,704 bytes/slot
/// (64 slots = 471,597,056 bytes; TP2 = 235,798,528 bytes, matching PROGRAMS.json).
pub const ATTENTION_SCRATCH: usize = 7 * 64 * 16 * (512 * 2 + 4);
/// Maximum concurrent request slots in one DFly2 draft batch.
pub const MAX_BATCH: usize = 64;

pub fn arena_id(rank: usize) -> String {
    format!("hy4/dfly2/gpu{rank}")
}

pub fn workspace_id(rank: usize, tp: usize) -> String {
    if tp == 1 {
        "hy4/dfly2/workspace".to_owned()
    } else {
        format!("hy4/dfly2/workspace/gpu{rank}")
    }
}

pub fn trellis_id(rank: usize, tp: usize) -> String {
    if tp == 1 {
        "hy4/dfly2/trellis".to_owned()
    } else {
        format!("hy4/dfly2/trellis/gpu{rank}")
    }
}

pub fn attention_scratch_bytes(slots: usize, tp: usize) -> usize {
    (ATTENTION_SCRATCH / tp) * slots
}

pub fn matrix_split(suffix: &str) -> CoreSplit {
    match suffix {
        "self_attn.q_b_proj"
        | "self_attn.kv_b_proj"
        | "self_attn.linear_gate"
        | "mlp.gate_proj"
        | "mlp.up_proj" => CoreSplit::Output,
        "self_attn.o_proj" | "mlp.down_proj" => CoreSplit::Input,
        _ => CoreSplit::Whole,
    }
}

pub fn matrix_shapes(cfg: &Hy4Config) -> [(&'static str, [usize; 2]); 9] {
    let h = cfg.hidden;
    [
        ("self_attn.q_a_proj", [cfg.q_lora_rank, h]),
        (
            "self_attn.q_b_proj",
            [cfg.heads * cfg.qk_head_dim(), cfg.q_lora_rank],
        ),
        (
            "self_attn.kv_a_proj_with_mqa",
            [cfg.kv_lora_rank + cfg.qk_rope_dim, h],
        ),
        (
            "self_attn.kv_b_proj",
            [cfg.heads * (cfg.qk_nope_dim + cfg.v_head_dim), cfg.kv_lora_rank],
        ),
        ("self_attn.linear_gate", [cfg.heads * cfg.v_head_dim, h]),
        ("self_attn.o_proj", [h, cfg.heads * cfg.v_head_dim]),
        ("mlp.gate_proj", [MLP_WIDTH, h]),
        ("mlp.up_proj", [MLP_WIDTH, h]),
        ("mlp.down_proj", [h, MLP_WIDTH]),
    ]
}

pub fn validate_serving(root: &serde_json::Value, text: &Hy4Config, cfg: &Hy4DflyConfig) -> Result<()> {
    let unique: BTreeSet<usize> = cfg.target_kv_layers.iter().copied().collect();
    ensure!(
        cfg.layers == 5
            && unique.len() == cfg.layers
            && cfg.target_kv_layers == cfg.target_index_layers
            && unique.iter().all(|&layer| text.is_full_indexer(layer)),
        "Hy4 DFly2 requires five distinct paired full-IndexCache KV and indexer taps"
    );
    ensure!(
        cfg.target_index_topk == text.indexer_topk
            && cfg.enable_hidden_correction
            && !cfg.enable_ihc
            && cfg.attn_causal
            && cfg.mlp_type == "dense"
            && cfg.mtp_fusion == "slot0"
            && root
                .get("dfly_context_local_window")
                .is_none_or(|w| w.is_null() || w.as_u64() == Some(0)),
        "Hy4 DFly2 requires shared full recall, dense MLP, causal attention, slot0 fusion, and hidden correction"
    );
    ensure!(
        (1..=6).contains(&cfg.num_speculative_tokens)
            && cfg.block_size.is_none_or(|b| b == cfg.block_rows()),
        "Hy4 DFly2 supports 1..=6 speculative tokens (block_size=7), found {} (block_size={:?})",
        cfg.num_speculative_tokens,
        cfg.block_size
    );
    Ok(())
}

#[derive(Debug, Clone)]
pub struct DraftWeights {
    pub rank: usize,
    pub tp: usize,
    pub arena: WeightArenaLayout,
    /// K8 MUL1 projections retained in their native trellis storage format.
    pub native: BTreeMap<String, CoreWeightLayout>,
    /// K8 sources for load-time conversion into the arena's absorbed BF16 views.
    pub absorbed: BTreeMap<String, CoreProjection>,
}

impl DraftWeights {
    pub fn plan(
        checkpoint: &Checkpoint,
        text: &Hy4Config,
        cfg: &Hy4DflyConfig,
        core: &BTreeMap<String, CoreProjection>,
    ) -> Result<Self> {
        Self::plan_rank(checkpoint, text, cfg, core, 0, 1)
    }

    pub fn plan_rank(
        checkpoint: &Checkpoint,
        text: &Hy4Config,
        cfg: &Hy4DflyConfig,
        core: &BTreeMap<String, CoreProjection>,
        rank: usize,
        tp: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(tp, 1 | 2) && rank < tp,
            "Hy4 DFly2 supports TP1 or TP2 rank geometry, got rank {rank}/{tp}"
        );
        validate_serving(&checkpoint.config, text, cfg)?;
        let mut views = BTreeMap::new();
        let mut native = BTreeMap::new();
        let mut absorbed = BTreeMap::new();
        for stage in 0..cfg.layers {
            for (suffix, shape) in matrix_shapes(text) {
                let stem = format!("{PREFIX}{stage}.{suffix}");
                let split = matrix_split(suffix);
                let eff_ranks = if split == CoreSplit::Whole { 1 } else { tp };
                let eff_rank = if split == CoreSplit::Whole { 0 } else { rank };
                if let Some(projection) = core.get(&stem) {
                    let expected_shape = if suffix == "self_attn.kv_a_proj_with_mqa" {
                        [KV_A_PADDED, text.hidden]
                    } else {
                        shape
                    };
                    ensure!(
                        (projection.shape == shape || projection.shape == expected_shape)
                            && projection.bits == 8
                            && projection.codebook == Exl3Codebook::Mul1,
                        "{stem}: Hy4 DFly2 expects K8 MUL1 at {shape:?} (or {expected_shape:?}), found {:?}",
                        projection.shape
                    );
                    if suffix == "self_attn.kv_b_proj" {
                        ensure!(
                            shape[0] % eff_ranks == 0,
                            "{stem}: output dim {} does not divide TP{eff_ranks}",
                            shape[0]
                        );
                        views.insert(
                            format!("{stem}.weight"),
                            (shape[0] / eff_ranks) * shape[1] * 2,
                        );
                        absorbed.insert(stem, projection.clone());
                    } else {
                        let side = if eff_ranks == 2 && split == CoreSplit::Input {
                            TrellisSide::Input
                        } else {
                            TrellisSide::Output
                        };
                        let layout = CoreWeightLayout::new(projection, side, eff_rank, eff_ranks)?;
                        views.extend(layout.arena_tensors(&stem)?);
                        native.insert(stem, layout);
                    }
                } else {
                    let name = format!("{stem}.weight");
                    let full_bytes = validate_bf16(checkpoint, &name, &shape)?;
                    ensure!(
                        match split {
                            CoreSplit::Output => shape[0] % eff_ranks == 0,
                            CoreSplit::Input => shape[1] % eff_ranks == 0,
                            CoreSplit::Whole => true,
                        },
                        "{name}: shape {shape:?} does not divide TP{eff_ranks}"
                    );
                    views.insert(name, full_bytes / eff_ranks);
                }
            }
            for (suffix, width) in [
                ("input_layernorm.weight", text.hidden),
                ("post_attention_layernorm.weight", text.hidden),
                ("self_attn.q_a_layernorm.weight", text.q_lora_rank),
                ("self_attn.kv_a_layernorm.weight", text.kv_lora_rank),
            ] {
                add_bf16(
                    checkpoint,
                    &mut views,
                    format!("{PREFIX}{stage}.{suffix}"),
                    &[width],
                )?;
            }
            let name = format!("{PREFIX}{stage}.self_attn.learnable_sink_param");
            ensure!(
                text.heads % tp == 0,
                "{name}: heads {} do not divide TP{tp}",
                text.heads
            );
            validate_bf16(checkpoint, &name, &[text.heads])?;
            // Expanded BF16 -> F32 at load time to match `hy4*_mla_sink_gate_uv_dsa`.
            views.insert(name, (text.heads / tp) * 4);
        }
        if rank == 0 {
            let h = text.hidden;
            for (suffix, shape) in [
                ("mtp_fusion.enorm.weight", vec![h]),
                ("mtp_fusion.hnorm.weight", vec![h]),
                ("mtp_fusion.gate", vec![h]),
                ("mtp_fusion.eh_proj.weight", vec![h, 2 * h]),
                ("hidden_correction.hidden_norm.weight", vec![h]),
                ("hidden_correction.embed_norm.weight", vec![h]),
                ("hidden_correction.gate_proj.weight", vec![h, 2 * h]),
                ("hidden_correction.up_proj.weight", vec![h, 2 * h]),
                ("hidden_correction.down_proj.weight", vec![h, h]),
                ("final_norm.weight", vec![h]),
            ] {
                add_bf16(checkpoint, &mut views, format!("{PREFIX}{suffix}"), &shape)?;
            }
        }
        Ok(Self {
            rank,
            tp,
            arena: WeightArenaLayout::new(arena_id(rank), views)?,
            native,
            absorbed,
        })
    }

    pub fn trellis_scratch(
        &self,
        rows: usize,
    ) -> Result<Option<crate::formats::trellis_core::ProjectionScratchLayout>> {
        let mut merged: Option<crate::formats::trellis_core::ProjectionScratchLayout> = None;
        for layout in self.native.values() {
            let [n, k] = layout.shape;
            let next = crate::formats::trellis_core::ProjectionScratchLayout::new(k, n, rows)?;
            if let Some(old) = &mut merged {
                old.merge(&next)?;
            } else {
                merged = Some(next);
            }
        }
        Ok(merged)
    }
}

fn validate_bf16(checkpoint: &Checkpoint, name: &str, shape: &[usize]) -> Result<usize> {
    let header = checkpoint
        .header(name)
        .with_context(|| format!("Hy4 DFly2 lacks {name}"))?;
    ensure!(
        header.dtype == DType::Bf16 && header.shape == shape,
        "{name}: Hy4 DFly2 expects BF16 {shape:?}, found {:?} {:?}",
        header.dtype,
        header.shape
    );
    let expected = shape.iter().product::<usize>() * 2;
    ensure!(
        header.byte_length == expected as u64,
        "{name}: Hy4 DFly2 BF16 payload extent differs from {shape:?}"
    );
    Ok(expected)
}

fn add_bf16(
    checkpoint: &Checkpoint,
    views: &mut BTreeMap<String, usize>,
    name: String,
    shape: &[usize],
) -> Result<()> {
    let bytes = validate_bf16(checkpoint, &name, shape)?;
    views.insert(name, bytes);
    Ok(())
}

/// Named packed scratch arena for all admitted DFly2 request blocks on one rank.
#[allow(clippy::too_many_arguments)]
pub fn workspace_rank(
    cfg: &Hy4DflyConfig,
    text: &Hy4Config,
    kv: Hy4KvCache,
    slots: usize,
    verify_rows: usize,
    attention_scratch: usize,
    rank: usize,
    tp: usize,
    context_split: bool,
) -> Result<WeightArenaLayout> {
    ensure!(
        matches!(tp, 1 | 2) && rank < tp,
        "Hy4 DFly2 workspace rank {rank}/{tp} invalid"
    );
    ensure!(
        (1..=MAX_BATCH).contains(&slots),
        "Hy4 DFly2 draft batch {slots} exceeds installed capacity {MAX_BATCH}"
    );
    let block = cfg.block_rows();
    let r = slots * block;
    let h = text.hidden;
    let heads = text.heads / tp;
    let mlp_width = MLP_WIDTH / tp;
    let k = cfg.target_index_topk;
    let record = kv.record_bytes();
    let recall = cfg.layers * k * 4;
    let mut planes: Vec<(&str, usize)> = vec![
        ("positions", r * 24),
        ("write_slots", r * 8),
        ("page_table", r * (k + block) * 4),
        ("lengths", r * 4),
        ("cu", (r + 1) * 4),
        ("kv", slots * (k + block) * record),
        ("state", r * h * 2),
        ("residual", r * h * 2),
        ("norm", r * h * 2),
        ("delta", r * h * 2),
        ("q_a", r * text.q_lora_rank * 2),
        ("q_norm", r * text.q_lora_rank * 2),
        ("kv_a", r * record.max(KV_A_PADDED * 2)),
        ("kv_padded", r * KV_A_PADDED * 2),
        ("q_b", r * heads * text.qk_head_dim() * 2),
        ("query", r * heads * (text.kv_lora_rank + text.qk_rope_dim) * 2),
        ("gate", r * heads * text.v_head_dim * 2),
        ("uv", r * heads * text.v_head_dim * 2),
        ("attn", r * heads * text.kv_lora_rank * 2),
        ("lse", r * heads * 4),
        ("attn_scratch", attention_scratch.max(256)),
        ("gate_ffn", r * mlp_width * 2),
        ("up_ffn", r * mlp_width * 2),
        ("act", r * mlp_width * 2),
        ("tap_recall", slots * recall),
        ("verify_recall", verify_rows * recall),
    ];
    if context_split {
        planes.push(("tap_recall_peer", slots * recall));
        planes.push(("verify_recall_peer", verify_rows * recall));
        // Per-slot/per-verify local recall lengths (u32 per tap layer) for exact 2-part gather.
        planes.push(("tap_len_own", slots * cfg.layers * 4));
        planes.push(("verify_len_own", verify_rows * cfg.layers * 4));
        planes.push(("tap_len_peer", slots * cfg.layers * 4));
        planes.push(("verify_len_peer", verify_rows * cfg.layers * 4));
    }
    if tp == 2 {
        planes.extend([("delta_f32", r * h * 4), ("delta_sum_f32", r * h * 4)]);
    }
    if rank == 0 {
        planes.extend([
            ("ids", r * 4),
            ("joined", slots * h * 4),
            ("anchor_ids", slots * 4),
            ("anchor", slots * h * 2),
            ("features", slots * h * 2),
            ("join_left", slots * h * 2),
            ("join_right", slots * h * 2),
            ("greedy_ids", r * 4),
            ("greedy_scores", r * 4),
            ("tap_hidden", slots * h * 2),
            ("verify_hidden", verify_rows * h * 2),
        ]);
    }
    WeightArenaLayout::new(
        workspace_id(rank, tp),
        planes.into_iter().map(|(name, bytes)| (name.into(), bytes)),
    )
}

#[derive(Clone, Debug)]
pub struct RankAdmission {
    pub rank: usize,
    pub weights: DraftWeights,
    pub workspace: WeightArenaLayout,
    pub trellis_bytes: usize,
    pub conversion_requests: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct Admission {
    pub tp: usize,
    pub slots: usize,
    pub block_rows: usize,
    pub ranks: Vec<RankAdmission>,
}

impl Admission {
    #[allow(clippy::too_many_arguments)]
    pub fn new_tp_with_context(
        checkpoint: &Checkpoint,
        text: &Hy4Config,
        cfg: &Hy4DflyConfig,
        core: &BTreeMap<String, CoreProjection>,
        kv: Hy4KvCache,
        slots: usize,
        rows: usize,
        tp: usize,
        context_split: bool,
    ) -> Result<Self> {
        ensure!(matches!(tp, 1 | 2), "Hy4 DFly2 supports TP1 or TP2, got {tp}");
        let block_rows = cfg.block_rows();
        let extent = block_rows * slots;
        let attn_scratch = attention_scratch_bytes(slots, tp);
        let mut ranks = Vec::with_capacity(tp);
        for rank in 0..tp {
            let weights = DraftWeights::plan_rank(checkpoint, text, cfg, core, rank, tp)?;
            let workspace = workspace_rank(
                cfg,
                text,
                kv,
                slots,
                rows,
                attn_scratch,
                rank,
                tp,
                context_split,
            )?;
            let trellis_bytes = weights
                .trellis_scratch(extent)?
                .map(|s| s.packed_bytes())
                .transpose()?
                .unwrap_or(0);
            let conversion_requests = if let Some(projection) = weights.absorbed.values().next() {
                let source = CoreWeightLayout::new(projection, TrellisSide::Output, rank, tp)?;
                let [n, k] = source.shape;
                let scratch = crate::formats::trellis_core::ProjectionScratchLayout::new(k, n, k)?;
                [
                    source.allocation_extents().as_slice(),
                    &[
                        scratch.packed_bytes()? as u64,
                        (k * k * 2) as u64,
                        (k * n * 2) as u64,
                    ],
                ]
                .concat()
            } else {
                Vec::new()
            };
            ranks.push(RankAdmission {
                rank,
                weights,
                workspace,
                trellis_bytes,
                conversion_requests,
            });
        }
        Ok(Self {
            tp,
            slots,
            block_rows,
            ranks,
        })
    }

    pub fn draft_rows(&self) -> usize {
        self.slots * self.block_rows
    }

    pub fn rank(&self, gpu: usize) -> Option<&RankAdmission> {
        self.ranks.get(gpu)
    }
}
