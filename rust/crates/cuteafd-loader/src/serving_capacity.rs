//! Canonical family cache storage geometry, before any CUDA allocation.
//! These costs cover persistent records/state, not the model's weights,
//! lane workspaces, graphs, transport or external drafter reservations.
use crate::families::glm5::{GlmDsaConfig, GlmIndexer};
use crate::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
use crate::families::mimo_v2::{MimoAttention, MimoKvCache, MimoV2Config};
use crate::families::qwen4::{Qwen4Attention, Qwen4Config};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

mod deepseek;
mod exl3_workspace;
pub use exl3_workspace::exl3_workspace_bytes;
mod glmf_workspace;
pub use glmf_workspace::{glmf_expert_rows, glmf_lane_bytes, glmf_manifest_scratch, glmf_selector_bytes,
    glmf_spark_intake_bytes, glmf_step_scratch,
    glmf_step_workspaces, glmf_table_pages, glmf_temporary_bytes, GlmfKdaState, GlmfLaneBytes, GlmfMissingProgram,
    GlmfScratch, GlmfScratchOptions, GlmfStepShape, GlmfStepWorkspaces, GlmfTemporaryBytes, GLMF_DECODE_ROWS,
    GLMF_DEFAULT_PREFILL_LANES, GLMF_HEAD_WORKSPACE, GLMF_SPARSE_TOPK, GLMF_WIDE_DECODE_ROWS};
mod v4_workspace;
pub use v4_workspace::{compiled_c128_width, deepseek_v4_peer_exchange_bytes, deepseek_v4_workspace_geometry, deepseek_v4_workspace_scratch, V4WorkspaceRank, V4WorkspaceScratch};
pub use deepseek::{deepseek_v41_cache_bytes, deepseek_v41_cache_geometry, deepseek_v41_pool_groups,
    deepseek_v4_cache_geometry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvPlacement {
    SingleDevice,
    Replicated,
    PartitionedHeads,
    PartitionedLayers,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankCacheGeometry {
    /// Record and index storage of one logical allocation unit on this rank.
    /// Pool-sized tables and workspaces are separate reservations.
    pub persistent_unit_bytes: u64,
    /// Shared unit-index metadata, excluding per-workspace page tables.
    pub pool_metadata_unit_bytes: u64,
    pub active_state_per_sequence_bytes: u64,
    pub retained_mark_bytes: u64,
    pub speculative_replay_bytes: u64,
    pub fixed_state_bytes: u64,
    pub context_table_bytes_per_token: u64,
}

/// GLM 5.3 Flash's DSA index cache layout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmfIndexCache {
    /// Every token's BF16 key | gate row (512 B per MLA layer) beside its latent record.
    #[default]
    Keys,
    /// The pooled keys alone, plus per sequence and MLA layer a tail of at most three BF16
    /// key | gate rows (`GLMF_INDEX_TAIL_BYTES`) and a speculative record of the decode rows.
    Compact,
}

/// One sequence's compact index tail in one MLA layer: an i32 count, 12 reserved bytes and
/// three BF16 key | gate rows of 512 B.
pub const GLMF_INDEX_TAIL_BYTES: u64 = 16 + 3 * 512;

/// Units GLM 5.3 Flash keeps out of every allocation when its prefix marks live in pool units
/// (`--prefix-marks pool`), beside the pool the admission sizes: unit 0, whose first MLA
/// record (slot 0) the decode sparse MLA reads for every masked candidate and weights by zero.
/// A mark there would put arbitrary bytes in it, and 0 x NaN is NaN; reserved, it stays zero.
pub const GLMF_POOL_MARK_RESERVED_UNITS: u64 = 1;

#[derive(Debug, Clone, Copy)]
pub struct CacheOptions {
    pub coordinator_ranks: usize,
    pub native_mtp_layers: usize,
    pub mimo_kv: MimoKvCache,
    /// DeepSeek V4's window ring holds one prefill chunk plus its window.
    pub prefill_rows: u64,
    /// GLM 5.3 Flash's DSA index cache.
    pub glmf_index: GlmfIndexCache,
    /// Bytes of one GLM Flash KDA recurrent-state element: 4 (FP32) or 2 (`--kda-state bf16`).
    pub kda_state_bytes: u64,
    /// Rows of a GLM Flash decode or verify step (`--decode-rows`, 64 or 128): its speculative
    /// replay records and commit tables hold this many rows.
    pub glmf_decode_rows: u64,
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            coordinator_ranks: 1,
            native_mtp_layers: 0,
            mimo_kv: MimoKvCache::Int8,
            prefill_rows: 4096,
            glmf_index: GlmfIndexCache::Keys,
            kda_state_bytes: 4,
            glmf_decode_rows: GLMF_DECODE_ROWS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheRequirements {
    pub checkpoint_max_context_tokens: Option<u64>,
    /// Indexed families require the concrete serving manifest, not a guessed
    /// exporter default. This storage-only report does not inspect that file.
    pub compiled_index_extent_required: bool,
    pub requested_kv_floor_tokens: Option<u64>,
    pub concurrency: u32,
    pub state_slots: u32,
    /// Target-only caches; native/external drafters are opt-in reservations.
    pub target_only_layouts: Vec<FamilyCacheGeometry>,
    pub unavailable_layouts: Vec<CacheLayoutFailure>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLayoutFailure {
    pub coordinator_ranks: usize,
    pub reason: String,
}

pub fn cache_requirements(
    model: &dyn crate::plan::FamilyModel,
    checkpoint: &Value,
) -> Result<Option<CacheRequirements>, CacheGeometryError> {
    let mut layouts = Vec::new();
    let mut failures = Vec::new();
    for coordinator_ranks in [1, 2] {
        match model.cache_geometry(CacheOptions {
            coordinator_ranks,
            ..Default::default()
        }) {
            Ok(Some(layout)) => layouts.push(layout),
            Ok(None) => return Ok(None),
            Err(error) => failures.push(CacheLayoutFailure {
                coordinator_ranks,
                reason: error.to_string(),
            }),
        }
    }
    let checkpoint_max = checkpoint_context_limit(checkpoint)?;
    let requested_floor = Some(cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS);
    Ok(Some(CacheRequirements {
        checkpoint_max_context_tokens: checkpoint_max,
        compiled_index_extent_required: matches!(
            model.spec().family,
            "glm5" | "glm5_flash" | "qwen4" | "deepseek_v4"
        ),
        requested_kv_floor_tokens: requested_floor,
        concurrency: 16,
        state_slots: 20,
        target_only_layouts: layouts,
        unavailable_layouts: failures,
    }))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyCacheGeometry {
    pub logical_unit_rows: u64,
    pub placement: KvPlacement,
    /// Logical coordinator ranks; runtime maps these to physical GPU ids.
    pub ranks: Vec<RankCacheGeometry>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CacheGeometryError {
    #[error("checkpoint max_position_embeddings must be a positive integer")]
    InvalidContext,
    #[error("cache geometry arithmetic overflows: {0}")]
    Overflow(&'static str),
    #[error("resident tensor {name}: {what}")]
    ResidentTensor { name: String, what: String },
    #[error("{family} cache geometry unsupported: {what}")]
    Unsupported {
        family: &'static str,
        what: &'static str,
    },
}

/// Read the text checkpoint's limit once. A missing limit is not replaced
/// with a small engine literal; automatic capacity planning must name it.
pub fn checkpoint_context_limit(root: &Value) -> Result<Option<u64>, CacheGeometryError> {
    let text = root.get("text_config").unwrap_or(root);
    match text.get("max_position_embeddings") {
        None => Ok(None),
        Some(value) => value
            .as_u64()
            .filter(|&v| v > 0)
            .map(Some)
            .ok_or(CacheGeometryError::InvalidContext),
    }
}

fn product(label: &'static str, values: &[u64]) -> Result<u64, CacheGeometryError> {
    values
        .iter()
        .try_fold(1u64, |n, &v| n.checked_mul(v))
        .ok_or(CacheGeometryError::Overflow(label))
}
fn sum(label: &'static str, values: &[u64]) -> Result<u64, CacheGeometryError> {
    values
        .iter()
        .try_fold(0u64, |n, &v| n.checked_add(v))
        .ok_or(CacheGeometryError::Overflow(label))
}
fn selected(
    family: &'static str,
    available: usize,
    layers: usize,
) -> Result<(), CacheGeometryError> {
    if layers == 0 || layers > available {
        return Err(CacheGeometryError::Unsupported {
            family,
            what: "selected layer count outside checkpoint",
        });
    }
    Ok(())
}

/// GLM keeps the complete latent/index pools on every head-split GPU.
pub fn glm_cache_geometry(
    cfg: &GlmDsaConfig,
    layers: usize,
    ranks: usize,
) -> Result<FamilyCacheGeometry, CacheGeometryError> {
    selected("glm5", cfg.layers, layers)?;
    if ![1, 2].contains(&ranks)
        || cfg.kv_lora_rank != 512
        || cfg.qk_rope_head_dim != 64
        || cfg.index_head_dim != 128
        || cfg.indexers.len() != cfg.layers
    {
        return Err(CacheGeometryError::Unsupported {
            family: "glm5",
            what: "record/index or coordinator rank geometry",
        });
    }
    let indexers = cfg.indexers[..layers]
        .iter()
        .filter(|&&kind| kind == GlmIndexer::Full)
        .count() as u64;
    let record_bytes = sum(
        "GLM records",
        &[
            product("GLM latent records", &[layers as u64, 656])?,
            product("GLM index records", &[indexers, 132])?,
        ],
    )?;
    let cost = RankCacheGeometry {
        persistent_unit_bytes: product("GLM pages", &[64, record_bytes])?,
        context_table_bytes_per_token: product(
            "GLM RoPE table",
            &[cfg.qk_rope_head_dim as u64, 4],
        )?,
        ..Default::default()
    };
    Ok(FamilyCacheGeometry {
        logical_unit_rows: 64,
        placement: if ranks == 1 {
            KvPlacement::SingleDevice
        } else {
            KvPlacement::Replicated
        },
        ranks: vec![cost; ranks],
    })
}

pub fn glm_flash_cache_geometry(
    cfg: &GlmNextConfig,
    layers: usize,
) -> Result<FamilyCacheGeometry, CacheGeometryError> {
    glm_flash_rank_cache_geometry(cfg, layers, 1, GlmfIndexCache::Keys, 4)
}

/// MLA records remain replicated; recurrent KDA state and replay follow each
/// coordinator's head partition, as in the GLM Flash engine's Caches. The
/// compact index cache drops the per-token index keys from the units and adds
/// each sequence's index tails to its state and marks (one GPU only for now).
/// The recurrent state takes `state_bytes` per element: 4 (FP32) or 2 (BF16).
pub fn glm_flash_rank_cache_geometry(
    cfg: &GlmNextConfig,
    layers: usize,
    ranks: usize,
    index: GlmfIndexCache,
    state_bytes: u64,
) -> Result<FamilyCacheGeometry, CacheGeometryError> {
    glm_flash_rank_cache_geometry_rows(cfg, layers, ranks, index, state_bytes, GLMF_DECODE_ROWS)
}

/// [`glm_flash_rank_cache_geometry`] for decode and verify steps of up to `decode_rows` rows
/// (`--decode-rows`: the 64-row programs' 64, or 128 with the wide `_m128` programs): every KDA
/// layer's speculative replay record, the compact index cache's key | gate records and the commit
/// tables hold that many rows. Units, state and marks do not depend on it.
pub fn glm_flash_rank_cache_geometry_rows(
    cfg: &GlmNextConfig,
    layers: usize,
    ranks: usize,
    index: GlmfIndexCache,
    state_bytes: u64,
    decode_rows: u64,
) -> Result<FamilyCacheGeometry, CacheGeometryError> {
    selected("glm5_flash", cfg.layers, layers)?;
    if ![GLMF_DECODE_ROWS, GLMF_WIDE_DECODE_ROWS].contains(&decode_rows) {
        return Err(CacheGeometryError::Unsupported {
            family: "glm5_flash",
            what: "decode rows other than the programs' 64 or 128",
        });
    }
    if ![1, 2].contains(&ranks)
        || ![2, 4].contains(&state_bytes)
        || cfg.kv_lora_rank != 512
        || cfg.kda_head_dim != 128
        || cfg.kda_heads == 0
        || cfg.kda_heads % ranks != 0
        || cfg.heads % ranks != 0
        || cfg.index_kpool != 4
        || cfg.attention.len() != cfg.layers
    {
        return Err(CacheGeometryError::Unsupported {
            family: "glm5_flash",
            what: "KDA/MLA record or pool geometry",
        });
    }
    let mla = cfg.attention[..layers]
        .iter()
        .filter(|&&a| a == GlmNextAttention::Mla)
        .count() as u64;
    let kda = layers as u64 - mla;
    let channels = product(
        "KDA channels",
        &[(cfg.kda_heads / ranks) as u64, cfg.kda_head_dim as u64],
    )?;
    let state = sum(
        "KDA state",
        &[
            product(
                "KDA recurrent state",
                &[channels, cfg.kda_head_dim as u64, state_bytes],
            )?,
            product("KDA short-conv state", &[3, 3, channels, 2])?,
        ],
    )?;
    let replay = sum(
        "KDA replay",
        &[
            product(
                "KDA recurrent replay",
                &[decode_rows, (cfg.kda_heads / ranks) as u64, 3, 128, 4],
            )?,
            product("KDA beta replay", &[decode_rows, (cfg.kda_heads / ranks) as u64, 4])?,
            product("KDA conv replay", &[decode_rows, 3, channels, 2])?,
        ],
    )?;
    let compact = index == GlmfIndexCache::Compact;
    if compact && ranks != 1 {
        return Err(CacheGeometryError::Unsupported {
            family: "glm5_flash",
            what: "compact index cache under a head split",
        });
    }
    // Per unit and MLA layer: 256 latent records, the per-token index keys unless compact,
    // and one pool-key page (64 pools).
    let per_layer_unit = if compact {
        sum("GLM Flash MLA unit", &[256 * 528, 64 * 132])?
    } else {
        sum("GLM Flash MLA unit", &[256 * (528 + 512), 64 * 132])?
    };
    // Compact: each sequence's index tails, and a key | gate record of the decode rows per MLA layer.
    let tails = if compact { product("GLM Flash index tails", &[mla, GLMF_INDEX_TAIL_BYTES])? } else { 0 };
    let index_replay = if compact { product("GLM Flash index replay", &[mla, decode_rows, 512])? } else { 0 };
    let kda_state = product("GLM Flash active KDA", &[kda, state])?;
    Ok(FamilyCacheGeometry {
        logical_unit_rows: 256,
        placement: if ranks == 1 { KvPlacement::SingleDevice } else { KvPlacement::Replicated },
        ranks: vec![RankCacheGeometry {
            persistent_unit_bytes: product("GLM Flash MLA pools", &[mla, per_layer_unit])?,
            pool_metadata_unit_bytes: 4,
            active_state_per_sequence_bytes: sum("GLM Flash active state", &[kda_state, tails])?,
            retained_mark_bytes: sum("GLM Flash mark", &[kda_state, tails])?,
            speculative_replay_bytes: sum("GLM Flash replay",
                &[product("GLM Flash KDA replay", &[kda, replay])?, index_replay])?,
            // The commit tables: slot, first row and kept rows of up to `decode_rows` sequences.
            fixed_state_bytes: product("GLM Flash commit tables", &[3, decode_rows, 4])?,
            context_table_bytes_per_token: 0,
        }; ranks],
    })
}

/// GLM 5.3 Flash's KDA speculative replay records on one GPU (`ranks` 1) or on each GPU of a
/// head split: every KDA layer's 64-row record (k | decay | v and beta FP32 per row and head, the
/// q/k/v in-projection row BF16). The KDA part of the geometry's `speculative_replay_bytes` (the
/// compact index cache adds its key | gate records); what the engine places in the prefill lanes'
/// scratch with `--replay-records shared`.
pub fn glm_flash_kda_replay_bytes(cfg: &GlmNextConfig, layers: usize, ranks: usize)
    -> Result<u64, CacheGeometryError> {
    glm_flash_kda_replay_bytes_rows(cfg, layers, ranks, GLMF_DECODE_ROWS)
}

/// [`glm_flash_kda_replay_bytes`] for decode and verify steps of up to `decode_rows` rows
/// (`--decode-rows`, 64 or 128): every KDA layer's record holds that many rows, as
/// [`glm_flash_rank_cache_geometry_rows`]'s do.
pub fn glm_flash_kda_replay_bytes_rows(cfg: &GlmNextConfig, layers: usize, ranks: usize, decode_rows: u64)
    -> Result<u64, CacheGeometryError> {
    selected("glm5_flash", cfg.layers, layers)?;
    if ![GLMF_DECODE_ROWS, GLMF_WIDE_DECODE_ROWS].contains(&decode_rows) {
        return Err(CacheGeometryError::Unsupported {
            family: "glm5_flash",
            what: "decode rows other than the programs' 64 or 128",
        });
    }
    if ![1, 2].contains(&ranks) || cfg.kda_heads % ranks != 0 || cfg.attention.len() != cfg.layers {
        return Err(CacheGeometryError::Unsupported { family: "glm5_flash", what: "KDA replay geometry" });
    }
    let kda = cfg.attention[..layers].iter().filter(|&&a| a == GlmNextAttention::Kda).count() as u64;
    let heads = (cfg.kda_heads / ranks) as u64;
    let channels = product("KDA channels", &[heads, cfg.kda_head_dim as u64])?;
    let replay = sum("KDA replay", &[
        product("KDA recurrent replay", &[decode_rows, heads, 3, 128, 4])?,
        product("KDA beta replay", &[decode_rows, heads, 4])?,
        product("KDA conv replay", &[decode_rows, 3, channels, 2])?,
    ])?;
    product("GLM Flash KDA replay", &[kda, replay])
}

pub fn qwen_cache_geometry(
    cfg: &Qwen4Config,
    layers: usize,
    mtp: bool,
) -> Result<FamilyCacheGeometry, CacheGeometryError> {
    selected("qwen4", cfg.layers, layers)?;
    if cfg.kv_heads != 2
        || cfg.head_dim != 256
        || cfg.index_head_dim != 128
        || cfg.index_block != 4
        || cfg.conv_kernel == 0
        || cfg.attention.len() != cfg.layers
        || mtp && cfg.mtp_layers == 0
    {
        return Err(CacheGeometryError::Unsupported {
            family: "qwen4",
            what: "GQA/GDN/index or optional MTP geometry",
        });
    }
    let full = cfg.attention[..layers]
        .iter()
        .filter(|&&a| a == Qwen4Attention::Full)
        .count() as u64;
    let gdn = layers as u64 - full;
    let conv_channels = product(
        "GDN conv channels",
        &[
            sum(
                "GDN heads",
                &[
                    product("GDN key heads", &[2, cfg.gdn_key_heads as u64])?,
                    cfg.gdn_value_heads as u64,
                ],
            )?,
            cfg.gdn_head_dim as u64,
        ],
    )?;
    let state = sum(
        "GDN state",
        &[
            product(
                "GDN recurrent state",
                &[
                    cfg.gdn_value_heads as u64,
                    cfg.gdn_head_dim as u64,
                    cfg.gdn_head_dim as u64,
                    4,
                ],
            )?,
            product(
                "GDN conv state",
                &[(cfg.conv_kernel - 1) as u64, conv_channels, 2],
            )?,
        ],
    )?;
    let hc_width = product(
        "Qwen hyperconnection width",
        &[cfg.hc_count as u64, cfg.hidden as u64],
    )?;
    let ple = if cfg.ple_layers.first().is_some_and(|&first| first < layers) {
        product("PLE state", &[9, hc_width, 2])?
    } else {
        0
    };
    let mark = sum(
        "Qwen mark",
        &[product("Qwen GDN mark", &[gdn, state])?, ple],
    )?;
    let mtp_pending = if mtp {
        product("MTP pending rows", &[64, hc_width, 2])?
    } else {
        0
    };
    let replay_row = sum(
        "GDN replay row",
        &[
            product(
                "GDN keys",
                &[cfg.gdn_key_heads as u64, cfg.gdn_head_dim as u64, 4],
            )?,
            product(
                "GDN values",
                &[cfg.gdn_value_heads as u64, cfg.gdn_head_dim as u64, 4],
            )?,
            product("GDN decay/beta", &[cfg.gdn_value_heads as u64, 2, 4])?,
            product("GDN conv input", &[conv_channels, 2])?,
        ],
    )?;
    let replay_layer = product("GDN replay", &[64, replay_row])?
        .div_ceil(1024)
        .checked_mul(1024)
        .ok_or(CacheGeometryError::Overflow("GDN replay padding"))?;
    let ple_replay = if ple > 0 {
        product("PLE replay", &[64, hc_width, 2])?
    } else {
        0
    };
    Ok(FamilyCacheGeometry {
        logical_unit_rows: 256,
        placement: KvPlacement::SingleDevice,
        ranks: vec![RankCacheGeometry {
            persistent_unit_bytes: product(
                "Qwen full/MTP pools",
                &[full + u64::from(mtp), 256 * (2048 + 256) + 64 * 256],
            )?,
            pool_metadata_unit_bytes: 4,
            active_state_per_sequence_bytes: sum("Qwen active state", &[mark, mtp_pending])?,
            retained_mark_bytes: mark,
            speculative_replay_bytes: sum(
                "Qwen replay storage",
                &[
                    product("Qwen GDN replay", &[gdn, replay_layer])?,
                    ple_replay,
                ],
            )?,
            // Commit tables plus the deferred MTP id buffer allocated even with MTP off.
            fixed_state_bytes: 3 * 64 * 4 + 64 * 64 * 4,
            context_table_bytes_per_token: 0,
        }],
    })
}

pub fn mimo_cache_geometry(
    cfg: &MimoV2Config,
    layers: usize,
    ranks: usize,
    kv: MimoKvCache,
    mtp_stages: usize,
) -> Result<FamilyCacheGeometry, CacheGeometryError> {
    selected("mimo_v2", cfg.layers, layers)?;
    if ![1, 2].contains(&ranks)
        || cfg.attention.len() != cfg.layers
        || cfg.window > 256 - 64
        || cfg.head_dim != 192
        || cfg.v_head_dim != 128
        || cfg.rope_dim != 64
        || cfg.program_family().is_err()
    {
        return Err(CacheGeometryError::Unsupported {
            family: "mimo_v2",
            what: "attention or coordinator rank geometry",
        });
    }
    let share = cfg
        .head_split(ranks)
        .map_err(|_| CacheGeometryError::Unsupported {
            family: "mimo_v2",
            what: "KV/query heads or dense MLP cannot be partitioned over selected ranks",
        })?;
    let full = cfg.attention[..layers]
        .iter()
        .filter(|&&a| a == MimoAttention::Full)
        .count() as u64;
    let swa = layers as u64 - full;
    let swa_record = share.record_bytes(MimoAttention::Sliding, kv) as u64;
    let base = RankCacheGeometry {
        persistent_unit_bytes: product(
            "MiMo full KV pages",
            &[64, full, share.record_bytes(MimoAttention::Full, kv) as u64],
        )?,
        active_state_per_sequence_bytes: product("MiMo SWA rings", &[swa, 256, swa_record])?,
        retained_mark_bytes: product("MiMo SWA marks", &[swa, cfg.window as u64, swa_record])?,
        context_table_bytes_per_token: product(
            "MiMo full/SWA RoPE tables",
            &[2, cfg.rope_dim as u64, 4],
        )?,
        speculative_replay_bytes: 0,
        ..Default::default()
    };
    let mut costs = vec![base; ranks];
    if mtp_stages > 0 {
        let mtp_rows = cfg
            .window
            .checked_add(mtp_stages)
            .and_then(|v| v.checked_add(1))
            .filter(|&v| v <= 256)
            .ok_or(CacheGeometryError::Unsupported {
                family: "mimo_v2",
                what: "MTP hidden history exceeds its 256-row ring",
            })?;
        let stage_ring = product(
            "MiMo MTP stage rings",
            &[
                mtp_stages as u64,
                256,
                cfg.record_bytes(MimoAttention::Sliding, kv) as u64,
            ],
        )?;
        let hidden_ring = product("MiMo MTP hidden ring", &[256, cfg.hidden as u64, 2])?;
        costs[0].active_state_per_sequence_bytes = sum(
            "MiMo MTP active state",
            &[
                costs[0].active_state_per_sequence_bytes,
                stage_ring,
                hidden_ring,
            ],
        )?;
        costs[0].retained_mark_bytes = sum(
            "MiMo MTP hidden mark",
            &[
                costs[0].retained_mark_bytes,
                product(
                    "MiMo MTP kept hidden rows",
                    &[mtp_rows as u64, cfg.hidden as u64, 2],
                )?,
            ],
        )?;
        // Four [64,H] BF16 buffers and one [64,2H], plus the ids/index allocations.
        costs[0].fixed_state_bytes = sum(
            "MiMo MTP scratch",
            &[
                product("MiMo MTP hidden scratch", &[12, 64, cfg.hidden as u64])?,
                product("MiMo MTP ids", &[mtp_stages as u64 + 1, 64, 4])?.max(256),
                256,
            ],
        )?;
    }
    Ok(FamilyCacheGeometry {
        logical_unit_rows: 64,
        placement: if ranks == 1 {
            KvPlacement::SingleDevice
        } else {
            KvPlacement::PartitionedHeads
        },
        ranks: costs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{
        glm5_config, glm5_flash_config, mimo_flash_config, mimo_pro_config, qwen4_config,
    };
    use serde_json::json;

    #[test]
    fn real_glm_indexer_schedule_and_replica_cost_are_exact() {
        let mut config = glm5_config();
        config["num_hidden_layers"] = json!(78);
        config["indexer_types"] = json!((0..78)
            .map(|i| if i < 21 { "full" } else { "shared" })
            .collect::<Vec<_>>());
        config["mlp_layer_types"] = json!((0..78)
            .map(|i| if i == 0 { "dense" } else { "sparse" })
            .collect::<Vec<_>>());
        let cfg = GlmDsaConfig::from_hf(&config).unwrap();
        let geometry = glm_cache_geometry(&cfg, 78, 2).unwrap();
        assert_eq!(geometry.placement, KvPlacement::Replicated);
        assert_eq!(geometry.ranks[0], geometry.ranks[1]);
        assert_eq!(geometry.ranks[0].persistent_unit_bytes / 64, 53940);
        assert_eq!(geometry.ranks[0].context_table_bytes_per_token, 256);
    }

    #[test]
    fn flash_kda_state_and_mla_pool_include_index_records() {
        let mut config = glm5_flash_config(45);
        config["text_config"]["layer_types"] = json!((0..45)
            .map(|i| if i % 4 == 3 {
                "deepseek_sparse_attention"
            } else {
                "linear_attention"
            })
            .collect::<Vec<_>>());
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let geometry = glm_flash_cache_geometry(&cfg, 45).unwrap();
        assert_eq!(geometry.logical_unit_rows, 256);
        assert_eq!(geometry.ranks[0].persistent_unit_bytes / 256, 11803);
        assert_eq!(geometry.ranks[0].retained_mark_bytes, 147619840);
        assert_eq!(
            geometry.ranks[0].active_state_per_sequence_bytes,
            geometry.ranks[0].retained_mark_bytes
        );
        let split = glm_flash_rank_cache_geometry(&cfg, 45, 2, GlmfIndexCache::Keys, 4).unwrap();
        assert_eq!(split.placement, KvPlacement::Replicated);
        assert_eq!(split.ranks[0], split.ranks[1]);
        assert_eq!(split.ranks[0].persistent_unit_bytes, geometry.ranks[0].persistent_unit_bytes);
        assert_eq!(split.ranks[0].pool_metadata_unit_bytes, geometry.ranks[0].pool_metadata_unit_bytes);
        assert_eq!(split.ranks[0].fixed_state_bytes, geometry.ranks[0].fixed_state_bytes);
        assert_eq!(split.ranks[0].retained_mark_bytes * 2, geometry.ranks[0].retained_mark_bytes);
        assert_eq!(split.ranks[0].speculative_replay_bytes * 2, geometry.ranks[0].speculative_replay_bytes);
        for ranks in [0, 3] {
            assert!(glm_flash_rank_cache_geometry(&cfg, 45, ranks, GlmfIndexCache::Keys, 4).is_err());
        }
        // A BF16 recurrent state: 34 KDA layers x (64 x 128 x 128 x 2 + the BF16 conv windows)
        // per sequence and per mark; the MLA pools and the FP32 replay records are unchanged.
        let bf16 = glm_flash_rank_cache_geometry(&cfg, 45, 1, GlmfIndexCache::Keys, 2).unwrap();
        assert_eq!(bf16.ranks[0].retained_mark_bytes, 76_316_672);
        assert_eq!(bf16.ranks[0].active_state_per_sequence_bytes, 76_316_672);
        assert_eq!(geometry.ranks[0].retained_mark_bytes - bf16.ranks[0].retained_mark_bytes, 34 * 64 * 128 * 128 * 2);
        assert_eq!(bf16.ranks[0].persistent_unit_bytes, geometry.ranks[0].persistent_unit_bytes);
        assert_eq!(bf16.ranks[0].speculative_replay_bytes, geometry.ranks[0].speculative_replay_bytes);
        assert!(glm_flash_rank_cache_geometry(&cfg, 45, 1, GlmfIndexCache::Keys, 8).is_err());
        // Both: the BF16 state with the compact cache's 11 index tails in every slot and mark, the
        // compact units and the index key records.
        let compact = glm_flash_rank_cache_geometry(&cfg, 45, 1, GlmfIndexCache::Compact, 4).unwrap();
        let both = glm_flash_rank_cache_geometry(&cfg, 45, 1, GlmfIndexCache::Compact, 2).unwrap();
        assert_eq!(both.ranks[0].retained_mark_bytes, 76_316_672 + 11 * GLMF_INDEX_TAIL_BYTES);
        assert_eq!(both.ranks[0].active_state_per_sequence_bytes, both.ranks[0].retained_mark_bytes);
        assert_eq!((both.ranks[0].persistent_unit_bytes, both.ranks[0].speculative_replay_bytes),
            (compact.ranks[0].persistent_unit_bytes, compact.ranks[0].speculative_replay_bytes));
    }

    /// The KDA records alone: every geometry's replay bytes less the compact index's key | gate
    /// records (34 KDA layers x 9,453,568 B on one GPU, half each under a head split).
    #[test]
    fn flash_kda_replay_bytes_are_the_geometry_s_kda_records() {
        // GLM 5.3 Flash's layout: every fourth layer MLA (11), the other 34 KDA.
        let mut config = glm5_flash_config(45);
        config["text_config"]["layer_types"] = json!((0..45)
            .map(|i| if i % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" })
            .collect::<Vec<_>>());
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        for (ranks, bytes) in [(1, 321_421_312u64), (2, 160_710_656)] {
            assert_eq!(glm_flash_kda_replay_bytes(&cfg, 45, ranks).unwrap(), bytes);
            let caches: &[(GlmfIndexCache, u64)] = if ranks == 1 {
                &[(GlmfIndexCache::Keys, 0), (GlmfIndexCache::Compact, 11 * 64 * 512)]
            } else { &[(GlmfIndexCache::Keys, 0)] };
            for &(index, records) in caches {
                for state_bytes in [4, 2] {
                    let geometry = glm_flash_rank_cache_geometry(&cfg, 45, ranks, index, state_bytes).unwrap();
                    assert_eq!(geometry.ranks[0].speculative_replay_bytes, bytes + records);
                }
            }
        }
        assert!(glm_flash_kda_replay_bytes(&cfg, 45, 3).is_err());
    }

    #[test]
    fn flash_compact_index_cache_drops_token_keys_and_carries_tails() {
        let mut config = glm5_flash_config(45);
        config["text_config"]["layer_types"] = json!((0..45)
            .map(|i| if i % 4 == 3 {
                "deepseek_sparse_attention"
            } else {
                "linear_attention"
            })
            .collect::<Vec<_>>());
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let keys = glm_flash_rank_cache_geometry(&cfg, 45, 1, GlmfIndexCache::Keys, 4).unwrap();
        assert_eq!(keys, glm_flash_cache_geometry(&cfg, 45).unwrap());
        let compact = glm_flash_rank_cache_geometry(&cfg, 45, 1, GlmfIndexCache::Compact, 4).unwrap();
        let (k, c) = (&keys.ranks[0], &compact.ranks[0]);
        // 11 MLA layers x (256 x 528 + 64 x 132) per 256-token unit.
        assert_eq!(c.persistent_unit_bytes, 1_579_776);
        assert_eq!(k.persistent_unit_bytes - c.persistent_unit_bytes, 11 * 256 * 512);
        // The admission rate: units and their 4-byte pool-page entry, rounded up per token.
        assert_eq!((c.persistent_unit_bytes + c.pool_metadata_unit_bytes).div_ceil(256), 6_172);
        assert_eq!((k.persistent_unit_bytes + k.pool_metadata_unit_bytes).div_ceil(256), 11_804);
        // Tails ride with the sequence's state and its marks; the speculative record per layer.
        assert_eq!(c.active_state_per_sequence_bytes - k.active_state_per_sequence_bytes, 11 * 1_552);
        assert_eq!(c.retained_mark_bytes - k.retained_mark_bytes, 11 * 1_552);
        assert_eq!(c.speculative_replay_bytes - k.speculative_replay_bytes, 11 * 64 * 512);
        assert_eq!(c.fixed_state_bytes, k.fixed_state_bytes);
        assert!(glm_flash_rank_cache_geometry(&cfg, 45, 2, GlmfIndexCache::Compact, 4).is_err());
    }

    /// `--decode-rows 128`: every KDA layer's replay record (9,453,568 B at 64 rows over 64 heads),
    /// the compact index cache's key | gate records and the commit tables double; units, state and
    /// marks do not move. 64 rows is the default geometry exactly.
    #[test]
    fn flash_wide_decode_rows_double_the_replay_records() {
        let mut config = glm5_flash_config(45);
        config["text_config"]["layer_types"] = json!((0..45)
            .map(|i| if i % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" })
            .collect::<Vec<_>>());
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        for (index, state) in [(GlmfIndexCache::Keys, 4), (GlmfIndexCache::Compact, 4), (GlmfIndexCache::Keys, 2)] {
            let narrow = glm_flash_rank_cache_geometry(&cfg, 45, 1, index, state).unwrap();
            assert_eq!(glm_flash_rank_cache_geometry_rows(&cfg, 45, 1, index, state, GLMF_DECODE_ROWS).unwrap(), narrow);
            let wide = glm_flash_rank_cache_geometry_rows(&cfg, 45, 1, index, state, GLMF_WIDE_DECODE_ROWS).unwrap();
            let (n, w) = (&narrow.ranks[0], &wide.ranks[0]);
            assert_eq!((w.persistent_unit_bytes, w.pool_metadata_unit_bytes, w.active_state_per_sequence_bytes,
                w.retained_mark_bytes), (n.persistent_unit_bytes, n.pool_metadata_unit_bytes,
                n.active_state_per_sequence_bytes, n.retained_mark_bytes));
            let index_records = if index == GlmfIndexCache::Compact { 11 * 64 * 512 } else { 0 };
            assert_eq!(n.speculative_replay_bytes, 34 * 9_453_568 + index_records);
            assert_eq!(w.speculative_replay_bytes, 2 * n.speculative_replay_bytes);
            assert_eq!((n.fixed_state_bytes, w.fixed_state_bytes), (3 * 64 * 4, 3 * 128 * 4));
        }
        // The 5090 arithmetic: +321,421,312 B of KDA records, +360,448 B of compact index records, +768 B
        // of commit tables.
        let keys = |rows| glm_flash_rank_cache_geometry_rows(&cfg, 45, 1, GlmfIndexCache::Keys, 4, rows).unwrap();
        let compact = |rows| glm_flash_rank_cache_geometry_rows(&cfg, 45, 1, GlmfIndexCache::Compact, 4, rows).unwrap();
        assert_eq!(keys(128).ranks[0].speculative_replay_bytes - keys(64).ranks[0].speculative_replay_bytes, 321_421_312);
        assert_eq!(compact(128).ranks[0].speculative_replay_bytes - compact(64).ranks[0].speculative_replay_bytes,
            321_421_312 + 360_448);
        assert_eq!(keys(128).ranks[0].fixed_state_bytes - keys(64).ranks[0].fixed_state_bytes, 768);
        // The KDA records alone (what `--replay-records shared` places in the prefill scratch) hold the
        // same rows: the geometry's records less the compact index's, 34 x 18,907,136 B at 128 rows.
        for (rows, bytes) in [(GLMF_DECODE_ROWS, 321_421_312u64), (GLMF_WIDE_DECODE_ROWS, 642_842_624)] {
            assert_eq!(glm_flash_kda_replay_bytes_rows(&cfg, 45, 1, rows).unwrap(), bytes);
            assert_eq!(keys(rows).ranks[0].speculative_replay_bytes, bytes);
            assert_eq!(compact(rows).ranks[0].speculative_replay_bytes, bytes + 11 * rows * 512);
        }
        assert_eq!(glm_flash_kda_replay_bytes(&cfg, 45, 1).unwrap(),
            glm_flash_kda_replay_bytes_rows(&cfg, 45, 1, GLMF_DECODE_ROWS).unwrap());
        // Only the programs' row counts.
        for rows in [0, 32, 65, 127, 256] {
            assert!(glm_flash_rank_cache_geometry_rows(&cfg, 45, 1, GlmfIndexCache::Keys, 4, rows).is_err(), "{rows}");
            assert!(glm_flash_kda_replay_bytes_rows(&cfg, 45, 1, rows).is_err(), "{rows}");
        }
        // The family model passes the option through.
        let wide = CacheOptions { glmf_decode_rows: GLMF_WIDE_DECODE_ROWS, ..Default::default() };
        assert_eq!(CacheOptions::default().glmf_decode_rows, GLMF_DECODE_ROWS);
        assert_eq!(wide.glmf_decode_rows, 128);
    }

    #[test]
    fn qwen_optional_mtp_adds_its_paged_history_and_pending_rows() {
        let mut config = qwen4_config(48);
        config["text_config"]["mtp_num_hidden_layers"] = json!(1);
        let cfg = Qwen4Config::from_hf(&config).unwrap();
        let plain = qwen_cache_geometry(&cfg, 48, false).unwrap();
        let mtp = qwen_cache_geometry(&cfg, 48, true).unwrap();
        assert_eq!(plain.ranks[0].persistent_unit_bytes / 256, 28416);
        assert_eq!(
            mtp.ranks[0].persistent_unit_bytes - plain.ranks[0].persistent_unit_bytes,
            256 * 2368
        );
        assert_eq!(
            mtp.ranks[0].active_state_per_sequence_bytes
                - plain.ranks[0].active_state_per_sequence_bytes,
            64 * cfg.hc_width() as u64 * 2
        );
        assert_eq!(
            mtp.ranks[0].retained_mark_bytes,
            plain.ranks[0].retained_mark_bytes
        );
    }

    fn full_mimo(mut config: Value, full: usize, swa: usize) -> MimoV2Config {
        config["num_hidden_layers"] = json!(full + swa);
        config["hybrid_layer_pattern"] = json!((0..full + swa)
            .map(|i| usize::from(i >= full))
            .collect::<Vec<_>>());
        config["moe_layer_freq"] = json!((0..full + swa)
            .map(|i| usize::from(i > 0))
            .collect::<Vec<_>>());
        MimoV2Config::from_hf(&config).unwrap()
    }

    #[test]
    fn mimo_pro_records_partition_heads_while_rings_and_marks_keep_the_exact_window() {
        let cfg = full_mimo(mimo_pro_config(), 10, 60);
        let one = mimo_cache_geometry(&cfg, 70, 1, MimoKvCache::Int8, 0).unwrap();
        let split = mimo_cache_geometry(&cfg, 70, 2, MimoKvCache::Int8, 0).unwrap();
        assert_eq!(one.ranks[0].persistent_unit_bytes / 64, 28800);
        assert_eq!(split.placement, KvPlacement::PartitionedHeads);
        assert_eq!(
            split
                .ranks
                .iter()
                .map(|r| r.persistent_unit_bytes)
                .sum::<u64>(),
            one.ranks[0].persistent_unit_bytes
        );
        assert_eq!(one.ranks[0].active_state_per_sequence_bytes, 75 << 20);
        assert_eq!(one.ranks[0].retained_mark_bytes, 75 << 19);
        assert_eq!(
            mimo_cache_geometry(&cfg, 70, 1, MimoKvCache::Bf16, 0)
                .unwrap()
                .ranks[0]
                .persistent_unit_bytes
                / 64,
            51200
        );
    }

    #[test]
    fn mimo_flash_and_native_mtp_keep_lead_rank_only_history_costs() {
        let cfg = full_mimo(mimo_flash_config(), 9, 39);
        let plain = mimo_cache_geometry(&cfg, 48, 2, MimoKvCache::Int8, 0).unwrap();
        let mtp = mimo_cache_geometry(&cfg, 48, 2, MimoKvCache::Int8, 3).unwrap();
        assert_eq!(
            plain
                .ranks
                .iter()
                .map(|r| r.persistent_unit_bytes)
                .sum::<u64>()
                / 64,
            12960
        );
        assert!(
            mtp.ranks[0].active_state_per_sequence_bytes
                > plain.ranks[0].active_state_per_sequence_bytes
        );
        assert_eq!(mtp.ranks[1], plain.ranks[1]);
        assert_eq!(
            mtp.ranks[0].retained_mark_bytes - plain.ranks[0].retained_mark_bytes,
            (128 + 3 + 1) * cfg.hidden as u64 * 2
        );
        assert_eq!(
            mtp.ranks[0].active_state_per_sequence_bytes
                - plain.ranks[0].active_state_per_sequence_bytes,
            6_029_312
        );
        assert_eq!(mtp.ranks[0].fixed_state_bytes, (3 << 20) + 1280);
    }

    #[test]
    fn checkpoint_context_reader_preserves_absent_limits_and_rejects_invalid_ones() {
        assert_eq!(
            checkpoint_context_limit(&json!({"text_config": {"max_position_embeddings": 1048576}})),
            Ok(Some(1048576))
        );
        assert_eq!(checkpoint_context_limit(&json!({})), Ok(None));
        for value in [json!(0), json!(-1), json!(null), json!("262144")] {
            assert_eq!(
                checkpoint_context_limit(&json!({"max_position_embeddings": value})),
                Err(CacheGeometryError::InvalidContext)
            );
        }
    }

    #[test]
    fn plan_reports_target_cache_costs_without_inventing_context_or_serving_fit() {
        use crate::plan::testing::write_snapshot;
        use crate::plan::{plan, render, PlanOptions};
        let dir = tempfile::tempdir().unwrap();
        let mut config = mimo_pro_config();
        config["max_position_embeddings"] = json!(1 << 20);
        write_snapshot(dir.path(), &config, &[], Some(8));
        let report = plan(dir.path(), &PlanOptions::default()).unwrap();
        let requirements = report.cache_requirements.as_ref().unwrap();
        assert_eq!(requirements.checkpoint_max_context_tokens, Some(1 << 20));
        assert_eq!(requirements.requested_kv_floor_tokens, Some(2 << 20));
        assert_eq!(
            (requirements.concurrency, requirements.state_slots),
            (16, 20)
        );
        assert!(!requirements.compiled_index_extent_required);
        assert_eq!(requirements.target_only_layouts.len(), 2);
        assert!(requirements.unavailable_layouts.is_empty());
        let text = render(&report);
        assert!(text.contains("partitioned KV heads"));
        assert!(text.contains("storage costs only"));
        assert!(text.contains("runtime admission must also reserve actual weights"));

        let mut flash = glm5_flash_config(4);
        flash["text_config"]["max_position_embeddings"] = json!(1 << 20);
        write_snapshot(dir.path(), &flash, &[], None);
        let indexed = plan(dir.path(), &PlanOptions::default()).unwrap();
        let requirements = indexed.cache_requirements.as_ref().unwrap();
        assert!(requirements.compiled_index_extent_required);
        assert_eq!(requirements.target_only_layouts.len(), 2);
        assert!(requirements.unavailable_layouts.is_empty());
        assert!(
            render(&indexed).contains("serving manifest must provide the compiled index extent")
        );

        config
            .as_object_mut()
            .unwrap()
            .remove("max_position_embeddings");
        write_snapshot(dir.path(), &config, &[], Some(8));
        let absent = plan(dir.path(), &PlanOptions::default())
            .unwrap()
            .cache_requirements
            .unwrap();
        assert_eq!(absent.checkpoint_max_context_tokens, None);
        assert_eq!(absent.requested_kv_floor_tokens, Some(2 << 20));
    }
}
