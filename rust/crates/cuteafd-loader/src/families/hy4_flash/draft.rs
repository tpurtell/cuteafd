//! Bundled DFly2 checkpoint geometry and its separately retired weight arena.
use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;

use crate::formats::dense_trellis::TrellisSide;
use crate::formats::exl3_storage::Exl3Codebook;
use crate::formats::trellis_core::{CoreProjection, CoreWeightLayout};
use crate::plan::checkpoint::Checkpoint;
use crate::serving_capacity::WeightArenaLayout;

use super::{Hy4DflyConfig, Hy4FlashConfig};

pub const PREFIX: &str = "model.dfly_layers.";
pub const ARENA_ID: &str = "hy4_flash/dfly2/gpu0";

/// The installed BF16 and MUL1 programs cover this FFN width. The real header
/// is still checked below (including the EXL3 logical geometry).
pub const MLP_WIDTH: usize = 9216;
// The draft attention export allows at most 16 resident partials per one of
// six rows. The eager path uses one, but reserves the manifest's bound.
pub const ATTENTION_SCRATCH: usize = 6 * 32 * 16 * (512 * 2 + 4);
/// All admitted Hy4 request slots may have a proposal in the same round.
pub const MAX_BATCH: usize = 64;

pub fn verify_key_bytes(cfg: &Hy4FlashConfig, rows: usize) -> usize {
    cfg.sparse_layers().len() * rows * cfg.indexer_head_dim * 2
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
        cfg.validate_serving(&checkpoint.config, text)?;
        let mut views = BTreeMap::new();
        let mut native = BTreeMap::new();
        let mut absorbed = BTreeMap::new();
        for (stage, &layer) in cfg.target_kv_layers.iter().enumerate() {
            for (suffix, shape) in matrix_shapes(text, layer) {
                let stem = format!("{PREFIX}{stage}.{suffix}");
                if let Some(projection) = core.get(&stem) {
                    ensure!(
                        projection.shape == shape
                            && projection.bits == 8
                            && projection.codebook == Exl3Codebook::Mul1,
                        "{stem}: DFly2 expects K8 MUL1 at {shape:?}"
                    );
                    if suffix == "self_attn.kv_b_proj" {
                        // Same exception as the target: absorbed MLA needs both W_UK
                        // and W_UV. Only the packed BF16 backing remains resident.
                        views.insert(format!("{stem}.weight"), shape[0] * shape[1] * 2);
                        absorbed.insert(stem, projection.clone());
                    } else {
                        let layout = CoreWeightLayout::new(projection, TrellisSide::Output, 0, 1)?;
                        views.extend(layout.arena_tensors(&stem)?);
                        native.insert(stem, layout);
                    }
                } else {
                    add_bf16(checkpoint, &mut views, format!("{stem}.weight"), &shape)?;
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
            add_bf16(checkpoint, &mut views, name.clone(), &[heads])?;
            // The shared sink/UV program takes F32; expanding BF16 is exact.
            views.insert(name, heads * 4);
        }
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
        Ok(Self {
            arena: WeightArenaLayout::new(ARENA_ID, views)?,
            native,
            absorbed,
        })
    }
}

fn add_bf16(
    checkpoint: &Checkpoint,
    views: &mut BTreeMap<String, usize>,
    name: String,
    shape: &[usize],
) -> Result<()> {
    let header = checkpoint
        .header(&name)
        .with_context(|| format!("DFly2 lacks {name}"))?;
    ensure!(
        header.dtype == DType::Bf16 && header.shape == shape,
        "{name}: DFly2 expects BF16 {shape:?}, found {:?} {:?}",
        header.dtype,
        header.shape
    );
    ensure!(
        header.byte_length == (shape.iter().product::<usize>() * 2) as u64,
        "{name}: DFly2 BF16 payload extent differs from {shape:?}"
    );
    views.insert(name, usize::try_from(header.byte_length)?);
    Ok(())
}

/// Named scratch for all admitted request blocks. Target taps are indexed by
/// slot and provisional row; one latent staging slab is reused across layers.
pub fn workspace(
    cfg: &Hy4DflyConfig,
    text: &Hy4FlashConfig,
    kv: super::Hy4KvCache,
    slots: usize,
    verify_rows: usize,
    attention_scratch: usize,
) -> Result<WeightArenaLayout> {
    workspace_with_context(cfg, text, kv, slots, verify_rows, attention_scratch, false)
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
    ensure!(
        (1..=MAX_BATCH).contains(&slots),
        "DFly2 draft batch exceeds installed capacity"
    );
    let block = cfg.block_rows();
    let r = slots * block;
    let h = text.hidden;
    let heads = text.sparse_heads;
    let k = cfg.target_index_topk;
    let record = kv.record_bytes();
    let recall = cfg.layers * k * 4;
    let mut planes = vec![
        ("ids", r * 4),
        ("positions", r * 8),
        ("write_slots", r * 8),
        ("page_table", r * (k + block) * 4),
        ("lengths", r * 4),
        ("cu", (r + 1) * 4),
        ("kv", slots * (k + block) * record),
        ("state", r * h * 2),
        ("residual", r * h * 2),
        ("norm", r * h * 2),
        ("delta", r * h * 2),
        ("joined", slots * h * 4),
        ("anchor_ids", slots * 4),
        ("anchor", slots * h * 2),
        ("features", slots * h * 2),
        ("join_left", slots * h * 2),
        ("join_right", slots * h * 2),
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
        ("gate_ffn", r * MLP_WIDTH * 2),
        ("up_ffn", r * MLP_WIDTH * 2),
        ("act", r * MLP_WIDTH * 2),
        ("greedy_ids", r * 4),
        ("greedy_scores", r * 4),
        ("tap_hidden", slots * h * 2),
        ("tap_recall", slots * recall),
        ("verify_hidden", verify_rows * h * 2),
        ("verify_recall", verify_rows * recall),
    ];
    if context_split {
        planes.push(("tap_recall_peer", slots * recall));
        planes.push(("verify_recall_peer", verify_rows * recall));
    }
    WeightArenaLayout::new(
        "hy4_flash/dfly2/workspace",
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

#[derive(Clone, Debug)]
pub struct Admission {
    pub weights: DraftWeights,
    pub workspace: WeightArenaLayout,
    pub trellis_bytes: usize,
    pub verify_bytes: usize,
    pub conversion_requests: Vec<u64>,
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
        Self::new_with_context(checkpoint, text, cfg, core, kv, slots, rows, false)
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
        let weights = DraftWeights::plan(checkpoint, text, cfg, core)?;
        let workspace = workspace_with_context(cfg, text, kv, slots, rows, ATTENTION_SCRATCH * slots, context_split)?;
        let trellis_bytes = weights
            .trellis_scratch(cfg.block_rows() * slots)?
            .map(|s| s.packed_bytes())
            .transpose()?
            .unwrap_or(0);
        let conversion_requests = if let Some(projection) = weights.absorbed.values().next() {
            let [n, k] = projection.shape;
            let source = CoreWeightLayout::new(projection, TrellisSide::Output, 0, 1)?;
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
        Ok(Self {
            weights,
            workspace,
            trellis_bytes,
            verify_bytes: verify_key_bytes(text, rows),
            conversion_requests,
        })
    }

    pub fn backing_bytes(&self, rank: usize) -> Result<u64> {
        use crate::placement::planner::allocation_granularity_forecast;
        let mut owners = vec![self.verify_bytes as u64];
        if rank == 0 {
            owners.extend([
                self.weights.arena.bytes() as u64,
                self.workspace.bytes() as u64,
                self.trellis_bytes as u64,
            ]);
            owners.extend(self.conversion_requests.iter().copied());
        }
        owners.retain(|&bytes| bytes != 0);
        let payload = owners.iter().sum::<u64>();
        payload
            .checked_add(allocation_granularity_forecast(&owners)?)
            .context("Hy4 DFly2 physical extent overflow")
    }
}
