//! Bundled DFly2 checkpoint geometry and its separately retired weight arenas.
use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;

use crate::formats::dense_trellis::TrellisSide;
use crate::formats::exl3_storage::Exl3Codebook;
use crate::formats::trellis_core::{CoreProjection, CoreSplit, CoreWeightLayout};
use crate::plan::checkpoint::Checkpoint;
use crate::serving_capacity::WeightArenaLayout;

use super::{Hy4DflyConfig, Hy4FlashConfig};

pub const PREFIX: &str = "model.dfly_layers.";
pub const ARENA_ID: &str = "hy4_flash/dfly2/gpu0";

/// The installed BF16 and MUL1 programs cover this FFN width. The real header
/// is still checked below (including the EXL3 logical geometry).
pub const MLP_WIDTH: usize = 9216;
// The draft attention export allows at most 16 resident partials per one of
// six rows across 32 heads. The eager path uses one, but reserves the manifest's bound.
pub const ATTENTION_SCRATCH: usize = 6 * 32 * 16 * (512 * 2 + 4);
/// All admitted Hy4 request slots may have a proposal in the same round.
pub const MAX_BATCH: usize = 64;

pub fn arena_id(rank: usize) -> String {
    format!("hy4_flash/dfly2/gpu{rank}")
}

pub fn workspace_id(rank: usize, tp: usize) -> String {
    if tp == 1 {
        "hy4_flash/dfly2/workspace".to_owned()
    } else {
        format!("hy4_flash/dfly2/workspace/gpu{rank}")
    }
}

pub fn trellis_id(rank: usize, tp: usize) -> String {
    if tp == 1 {
        "hy4_flash/dfly2/trellis".to_owned()
    } else {
        format!("hy4_flash/dfly2/trellis/gpu{rank}")
    }
}

pub fn attention_scratch_bytes(slots: usize, tp: usize) -> usize {
    (ATTENTION_SCRATCH / tp) * slots
}

pub fn verify_key_bytes(cfg: &Hy4FlashConfig, rows: usize) -> usize {
    cfg.sparse_layers().len() * rows * cfg.indexer_head_dim * 2
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

pub fn matrix_shapes(cfg: &Hy4FlashConfig, layer: usize) -> [(&'static str, [usize; 2]); 9] {
    let a = cfg.attention(layer);
    let h = cfg.hidden;
    [
        ("self_attn.q_a_proj", [cfg.q_lora_rank, h]),
        (
            "self_attn.q_b_proj",
            [a.heads * a.qk_head_dim(), cfg.q_lora_rank],
        ),
        (
            "self_attn.kv_a_proj_with_mqa",
            [cfg.kv_lora_rank + a.qk_rope_dim, h],
        ),
        (
            "self_attn.kv_b_proj",
            [a.heads * (a.qk_nope_dim + a.v_head_dim), cfg.kv_lora_rank],
        ),
        ("self_attn.linear_gate", [a.heads * a.v_head_dim, h]),
        ("self_attn.o_proj", [h, a.heads * a.v_head_dim]),
        ("mlp.gate_proj", [MLP_WIDTH, h]),
        ("mlp.up_proj", [MLP_WIDTH, h]),
        ("mlp.down_proj", [h, MLP_WIDTH]),
    ]
}

#[derive(Debug, Clone)]
pub struct DraftWeights {
    pub rank: usize,
    pub tp: usize,
    pub arena: WeightArenaLayout,
    /// K8 MUL1 projections retained in their native storage format.
    pub native: BTreeMap<String, CoreWeightLayout>,
    /// K8 sources for load-time conversion into the arena's absorbed BF16 views.
    pub absorbed: BTreeMap<String, CoreProjection>,
}

impl DraftWeights {
    /// `checkpoint` supplies DFly2's selected header inventory, not a side manifest.
    pub fn plan(
        checkpoint: &Checkpoint,
        text: &Hy4FlashConfig,
        cfg: &Hy4DflyConfig,
        core: &BTreeMap<String, CoreProjection>,
    ) -> Result<Self> {
        Self::plan_rank(checkpoint, text, cfg, core, 0, 1)
    }

    /// Plan one rank's DFly2 weight arena for TP1 (`tp = 1`) or TP2 (`tp = 2`).
    /// Shards attention heads (`q_b`, `kv_b`, `linear_gate`, `o_proj`, sinks) and
    /// MLP neurons (`gate`, `up`, `down`); replicates norms, `q_a`, and `kv_a`;
    /// keeps `mtp_fusion`, `hidden_correction`, and `final_norm` on lead rank 0.
    pub fn plan_rank(
        checkpoint: &Checkpoint,
        text: &Hy4FlashConfig,
        cfg: &Hy4DflyConfig,
        core: &BTreeMap<String, CoreProjection>,
        rank: usize,
        tp: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(tp, 1 | 2) && rank < tp,
            "DFly2 supports TP1 or TP2 rank geometry, got rank {rank}/{tp}"
        );
        cfg.validate_serving(&checkpoint.config, text)?;
        let mut views = BTreeMap::new();
        let mut native = BTreeMap::new();
        let mut absorbed = BTreeMap::new();
        for (stage, &layer) in cfg.target_kv_layers.iter().enumerate() {
            for (suffix, shape) in matrix_shapes(text, layer) {
                let stem = format!("{PREFIX}{stage}.{suffix}");
                let split = matrix_split(suffix);
                let eff_ranks = if split == CoreSplit::Whole { 1 } else { tp };
                let eff_rank = if split == CoreSplit::Whole { 0 } else { rank };
                if let Some(projection) = core.get(&stem) {
                    ensure!(
                        projection.shape == shape
                            && projection.bits == 8
                            && projection.codebook == Exl3Codebook::Mul1,
                        "{stem}: DFly2 expects K8 MUL1 at {shape:?}"
                    );
                    if suffix == "self_attn.kv_b_proj" {
                        ensure!(
                            shape[0] % eff_ranks == 0,
                            "{stem}: output dim {} does not divide TP{eff_ranks}",
                            shape[0]
                        );
                        // Same exception as the target: absorbed MLA needs both W_UK
                        // and W_UV. Only the packed BF16 backing remains resident.
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
                ("self_attn.post_proj_norm.weight", text.hidden),
            ] {
                add_bf16(
                    checkpoint,
                    &mut views,
                    format!("{PREFIX}{stage}.{suffix}"),
                    &[width],
                )?;
            }
            let name = format!("{PREFIX}{stage}.self_attn.learnable_sink_param");
            let heads = text.attention(layer).heads;
            ensure!(
                heads % tp == 0,
                "{name}: heads {heads} do not divide TP{tp}"
            );
            validate_bf16(checkpoint, &name, &[heads])?;
            // The shared sink/UV program takes F32; expanding BF16 is exact.
            views.insert(name, (heads / tp) * 4);
        }
        if rank == 0 {
            let h = text.hidden;
            // The pinned vendor proposal path explicitly skips confidence_head.*.
            // Shared history calibrates from verifier outcomes, as for unscored MTP.
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
}

fn validate_bf16(checkpoint: &Checkpoint, name: &str, shape: &[usize]) -> Result<usize> {
    let header = checkpoint
        .header(name)
        .with_context(|| format!("DFly2 lacks {name}"))?;
    ensure!(
        header.dtype == DType::Bf16 && header.shape == shape,
        "{name}: DFly2 expects BF16 {shape:?}, found {:?} {:?}",
        header.dtype,
        header.shape
    );
    let expected = shape.iter().product::<usize>() * 2;
    ensure!(
        header.byte_length == expected as u64,
        "{name}: DFly2 BF16 payload extent differs from {shape:?}"
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

/// Named scratch for all admitted request blocks on one rank. Target taps are
/// indexed by slot and provisional row; one latent staging slab is reused across layers.
pub fn workspace(
    cfg: &Hy4DflyConfig,
    text: &Hy4FlashConfig,
    kv: super::Hy4KvCache,
    slots: usize,
    verify_rows: usize,
    attention_scratch: usize,
) -> Result<WeightArenaLayout> {
    workspace_rank(cfg, text, kv, slots, verify_rows, attention_scratch, 0, 1, false)
}

pub fn workspace_with_context(
    cfg: &Hy4DflyConfig,
    text: &Hy4FlashConfig,
    kv: super::Hy4KvCache,
    slots: usize,
    verify_rows: usize,
    attention_scratch: usize,
    context_split: bool,
) -> Result<WeightArenaLayout> {
    workspace_rank(cfg, text, kv, slots, verify_rows, attention_scratch, 0, 1, context_split)
}

#[allow(clippy::too_many_arguments)]
pub fn workspace_rank(
    cfg: &Hy4DflyConfig,
    text: &Hy4FlashConfig,
    kv: super::Hy4KvCache,
    slots: usize,
    verify_rows: usize,
    attention_scratch: usize,
    rank: usize,
    tp: usize,
    context_split: bool,
) -> Result<WeightArenaLayout> {
    ensure!(
        matches!(tp, 1 | 2) && rank < tp,
        "DFly2 workspace rank {rank}/{tp} invalid"
    );
    ensure!(
        (1..=MAX_BATCH).contains(&slots),
        "DFly2 draft batch exceeds installed capacity"
    );
    let block = cfg.block_rows();
    let r = slots * block;
    let h = text.hidden;
    let heads = text.sparse_heads / tp;
    let mlp_width = MLP_WIDTH / tp;
    let k = cfg.target_index_topk;
    let record = kv.record_bytes();
    let recall = cfg.layers * k * 4;
    let mut planes: Vec<(&str, usize)> = vec![
        // [rows,3] M-RoPE positions (text rows repeat one position on every axis).
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
        ("kv_a", r * record.max(1280)),
        ("kv_padded", r * 1280),
        ("q_b", r * heads * 256 * 2),
        ("query", r * heads * 576 * 2),
        ("gate", r * heads * 256 * 2),
        ("uv", r * heads * 256 * 2),
        ("attn", r * heads * 512 * 2),
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

impl DraftWeights {
    pub fn trellis_scratch(
        &self,
        rows: usize,
    ) -> Result<Option<crate::formats::trellis_core::ProjectionScratchLayout>> {
        let mut merged: Option<crate::formats::trellis_core::ProjectionScratchLayout> = None;
        for layout in self.native.values() {
            let next = crate::formats::trellis_core::ProjectionScratchLayout::for_weight(layout, rows)?;
            if let Some(old) = &mut merged {
                old.merge(&next)?;
            } else {
                merged = Some(next);
            }
        }
        Ok(merged)
    }
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
    pub verify_bytes: usize,
}

impl Admission {
    pub fn new(
        checkpoint: &Checkpoint,
        text: &Hy4FlashConfig,
        cfg: &Hy4DflyConfig,
        core: &BTreeMap<String, CoreProjection>,
        kv: super::Hy4KvCache,
        slots: usize,
        rows: usize,
    ) -> Result<Self> {
        Self::new_tp_with_context(checkpoint, text, cfg, core, kv, slots, rows, 1, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_context(
        checkpoint: &Checkpoint,
        text: &Hy4FlashConfig,
        cfg: &Hy4DflyConfig,
        core: &BTreeMap<String, CoreProjection>,
        kv: super::Hy4KvCache,
        slots: usize,
        rows: usize,
        context_split: bool,
    ) -> Result<Self> {
        Self::new_tp_with_context(checkpoint, text, cfg, core, kv, slots, rows, 1, context_split)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_tp(
        checkpoint: &Checkpoint,
        text: &Hy4FlashConfig,
        cfg: &Hy4DflyConfig,
        core: &BTreeMap<String, CoreProjection>,
        kv: super::Hy4KvCache,
        slots: usize,
        rows: usize,
        tp: usize,
    ) -> Result<Self> {
        Self::new_tp_with_context(checkpoint, text, cfg, core, kv, slots, rows, tp, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_tp_with_context(
        checkpoint: &Checkpoint,
        text: &Hy4FlashConfig,
        cfg: &Hy4DflyConfig,
        core: &BTreeMap<String, CoreProjection>,
        kv: super::Hy4KvCache,
        slots: usize,
        rows: usize,
        tp: usize,
        context_split: bool,
    ) -> Result<Self> {
        ensure!(matches!(tp, 1 | 2), "DFly2 supports TP1 or TP2, got {tp}");
        let block_rows = cfg.block_rows();
        let extent = block_rows * slots;
        let attn_scratch = attention_scratch_bytes(slots, tp);
        let mut ranks = Vec::with_capacity(tp);
        for rank in 0..tp {
            let weights = DraftWeights::plan_rank(checkpoint, text, cfg, core, rank, tp)?;
            let workspace =
                workspace_rank(cfg, text, kv, slots, rows, attn_scratch, rank, tp, context_split)?;
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
            verify_bytes: verify_key_bytes(text, rows),
        })
    }

    pub fn draft_rows(&self) -> usize {
        self.slots * self.block_rows
    }

    pub fn rank(&self, gpu: usize) -> Option<&RankAdmission> {
        self.ranks.get(gpu)
    }

    pub fn backing_bytes(&self, rank: usize) -> Result<u64> {
        use crate::placement::planner::allocation_granularity_forecast;
        let mut owners = vec![self.verify_bytes as u64];
        if let Some(local) = self.rank(rank) {
            owners.extend([
                local.weights.arena.bytes() as u64,
                local.workspace.bytes() as u64,
                local.trellis_bytes as u64,
            ]);
            owners.extend(local.conversion_requests.iter().copied());
        }
        owners.retain(|&bytes| bytes != 0);
        let payload = owners.iter().sum::<u64>();
        payload
            .checked_add(allocation_granularity_forecast(&owners)?)
            .context("Hy4 DFly2 physical extent overflow")
    }
}
