//! Named physical cache and packed scratch arena layouts for Hy4 (`hy_v4_vl`).
//!
//! Shared by `cuteafd plan --layout` and runtime admission (`serve-hy4`) so
//! planned and allocated bytes match owner for owner.
//!
//! Unlike `hy4_flash`, Hy4 has no SWA rings, no compressed-indexer pending
//! tokens, and no Engram tables: every layer is paged DSA MLA and IndexCache
//! stores per-token 132-byte E4M3+F32 keys on the 21 full indexer layers only
//! (`8448 B/page`, matching `glm_index_topk`). Exact prefix restore is
//! pages-only (zero per-slot device mark bytes).
use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::families::hy4_common::PAGE_ROWS;
use crate::serving_capacity::WeightArenaLayout;

use super::{Hy4Config, Hy4KvCache};

/// Per-token DSA index key record bytes: 128 E4M3 dims + 4B F32 scale
/// (matching `glm_index_topk` `index_page_bytes = 64 * 132 = 8448`).
pub const INDEX_RECORD_BYTES: usize = 132;
pub const INDEX_PAGE_BYTES: usize = PAGE_ROWS * INDEX_RECORD_BYTES;
/// Maximum rows projected through `lm_head` in one tile.
pub const HEAD_TILE_ROWS: usize = 64;
/// Padded `kv_a` width required by compact core projections (128-aligned >= 576).
pub const KV_A_PADDED: usize = 640;
/// Active-split-1 MLA scratch floor matching `PROGRAMS.json` (`hy4_mla_*`).
pub const ATTN_SCRATCH_BYTES: usize = 1024;
/// Fallback `glm_index_topk` scratch bytes (GLM53 32H/top-2048/m4096 capacity)
/// used only when no `PROGRAMS.json` manifest is supplied (e.g. synthetic unit tests).
pub const DEFAULT_INDEX_TOPK_PREFILL_SCRATCH: usize = 742_556_672;
pub const DEFAULT_INDEX_TOPK_DECODE_SCRATCH: usize = 11_537_408;
/// Fallback `scored_index_topk_decode` tile prefix bytes (GLM53 32H/top-2048/m64).
pub const DEFAULT_SCORED_TILE_PREFIX_BYTES: u64 = 1_051_648;

/// Fixed byte offset of peer routed-expert F32 partials inside `PeerExchange` FFN receive slot 1.
/// Shared-expert (or dense FFN) F32 partials sit at offset 0; routed F32 partials sit at this
/// 256-byte-aligned offset so both transfer in one `push_planes` exchange with invariant graph pointers.
pub fn hy4_peer_ffn_routed_offset(cfg: &Hy4Config, max_rows: usize) -> usize {
    (max_rows * cfg.hidden * 4).div_ceil(256) * 256
}

/// Receive-slot byte sizes for Hy4 TP2 `PeerExchange` (`[SLOT_ATTN, SLOT_FFN]`).
/// - Slot 0 (`ATTN`): attention `o_proj` F32 partials (`[max_rows, hidden]`).
/// - Slot 1 (`FFN`): dense FFN or combined `[shared_f32, routed_f32]` F32 partials
///   (Hy4 routed experts operate directly at full `H=6144`, no LatentMoE reduction).
pub fn hy4_peer_slot_sizes(cfg: &Hy4Config, max_rows: usize) -> [usize; 2] {
    let attn = (max_rows * cfg.hidden * 4).max(256);
    let routed_off = hy4_peer_ffn_routed_offset(cfg, max_rows);
    let ffn = (routed_off + max_rows * cfg.hidden * 4).max(256);
    [attn, ffn]
}

/// Physical partial O (BF16) and LSE (F32) planes for the capped MLA decode exports.
/// The export strides scratch by the compiled split capacity, not active splits.
pub fn mla_split_scratch_bytes(decode_rows: usize, head_parts: usize) -> usize {
    decode_rows * 16 * (64 / head_parts) * (512 * 2 + 4)
}

/// Scratch bytes required by `hy42_mla_dsa_partial_{bf16,fp8}_h{0,32}` (chunked at 64 rows,
/// 32 heads, 16 splits; mirrors `hy4_dsa_partial_scratch_bytes(rows, head_count=32)` in `hy4_mla.py`).
pub fn hy4_dsa_partial_scratch_bytes(rows: usize) -> usize {
    let chunk = rows.clamp(1, 64);
    let n = chunk * 32 * 16;
    let part = (n * 512 * 2).div_ceil(1024) * 1024;
    let lse = (n * 4).div_ceil(1024) * 1024;
    part + lse
}

/// Shared `ContextExchange` buffer specification for Hy4 2-GPU ContextSplit DSA layers
/// (32 heads/GPU, 576-dim absorbed query, 2048 scored per-token candidates, 2 alternating slots).
pub fn hy4_context_buffers(cfg: &Hy4Config, decode_rows: usize) -> crate::placement::ContextBuffers {
    let half_heads = (cfg.heads / 2) as u64;
    let kv_a_dim = (cfg.kv_lora_rank + cfg.qk_rope_dim) as u64;
    let topk = cfg.indexer_topk as u64;
    crate::placement::ContextBuffers {
        staging_unit_bytes: 0,
        staging_unit_rows: 1,
        query_row_bytes: half_heads * kv_a_dim * 2,
        partial_row_bytes: half_heads * (cfg.kv_lora_rank as u64 * 2 + 4),
        candidate_row_bytes: topk * 8,
        compiled_extent: 0,
        decode_rows: decode_rows as u64,
        lanes: 1,
        workspace_bytes: 0,
    }
}

/// Exact `glm_index_topk` and optional `scored_index_topk_decode` scratch bytes resolved
/// identically by `cuteafd plan --layout` and `serve-hy4` from `PROGRAMS.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexTopkScratch {
    /// Bytes for `hy4*_index_topk_{decode,prefill}` scratch (0 when ContextSplit only uses scored decode).
    pub topk_bytes: usize,
    /// Bytes for `hy42_scored_index_topk_decode` scratch (0 when not ContextSplit).
    pub scored_bytes: usize,
    /// Admitted parity pages passed as `admitted_pages` scalar to `scored_index_topk_decode`.
    pub admitted_pages: usize,
}

impl IndexTopkScratch {
    pub fn resolve(
        manifest: Option<&Path>,
        tp: usize,
        context_split: bool,
        context_tokens: usize,
    ) -> Result<Self> {
        ensure!(matches!(tp, 1 | 2) && context_tokens > 0, "invalid Hy4 IndexTopkScratch request");
        let parsed: Option<serde_json::Value> = manifest
            .filter(|p| p.is_file())
            .map(|p| -> Result<serde_json::Value> {
                let text = std::fs::read_to_string(p)
                    .with_context(|| format!("reading PROGRAMS.json at {}", p.display()))?;
                serde_json::from_str(&text)
                    .with_context(|| format!("parsing PROGRAMS.json at {}", p.display()))
            })
            .transpose()?;
        Self::from_manifest_value(parsed.as_ref(), tp, context_split, context_tokens)
    }

    pub fn from_manifest_value(
        manifest: Option<&serde_json::Value>,
        tp: usize,
        context_split: bool,
        context_tokens: usize,
    ) -> Result<Self> {
        let prefix = if tp == 1 { "hy4" } else { "hy42" };
        let programs = manifest.and_then(|v| v["programs"].as_array());
        let find = |names: &[&str]| -> Option<&serde_json::Value> {
            let list = programs?;
            for &name in names {
                if let Some(p) = list.iter().find(|p| p["name"].as_str() == Some(name)) {
                    return Some(p);
                }
            }
            None
        };
        let scratch_of = |names: &[&str], fallback: usize| -> usize {
            find(names)
                .and_then(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64())
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(fallback)
        };

        let admitted_pages = cuteafd_core::coordinator_programs::scored_decode::admitted_pages(
            context_tokens as u64,
            PAGE_ROWS as u64,
        );
        if context_split {
            let pre_name = format!("{prefix}_index_topk_prefill");
            let scored_pre_name = format!("{prefix}_scored_index_topk_prefill");
            let pre = scratch_of(&[&pre_name, "glm_index_topk_prefill_m4096"], DEFAULT_INDEX_TOPK_PREFILL_SCRATCH);
            let scored_pre = scratch_of(
                &[&scored_pre_name, "glm_scored_index_topk_prefill_m4096"],
                DEFAULT_INDEX_TOPK_PREFILL_SCRATCH,
            );
            let scored_prog = find(&["hy42_scored_index_topk_decode", "glm_scored_index_topk_decode_m64"]);
            let scored_bytes = if let Some(prog) = scored_prog {
                if let Some(tile) =
                    cuteafd_core::coordinator_programs::scored_decode::ScoredDecodeTile::from_geometry(
                        &prog["geometry"],
                    )
                    .map_err(|e| anyhow::anyhow!("Hy4 scored decode tile invalid: {e}"))?
                {
                    usize::try_from(
                        tile.bytes(admitted_pages)
                            .map_err(|e| anyhow::anyhow!("Hy4 scored decode capacity {admitted_pages} invalid: {e}"))?,
                    )?
                } else {
                    usize::try_from(
                        prog["scratch_bytes_at_capacity"]["scratch"]
                            .as_u64()
                            .unwrap_or(DEFAULT_SCORED_TILE_PREFIX_BYTES),
                    )?
                }
            } else {
                // Exact formula matching ScoredDecodeTile(prefix=1_051_648, rows=64, pages=16384).
                let tokens = admitted_pages * PAGE_ROWS as u64;
                let bytes = DEFAULT_SCORED_TILE_PREFIX_BYTES + 64 * tokens.div_ceil(512) * 512 * 4;
                usize::try_from(bytes)?
            };
            Ok(Self {
                topk_bytes: pre.max(scored_pre).max(256),
                scored_bytes: scored_bytes.max(256),
                admitted_pages: usize::try_from(admitted_pages)?,
            })
        } else {
            let dec_name = format!("{prefix}_index_topk_decode");
            let pre_name = format!("{prefix}_index_topk_prefill");
            let dec = scratch_of(&[&dec_name, "glm_index_topk_decode_m64"], DEFAULT_INDEX_TOPK_DECODE_SCRATCH);
            let pre = scratch_of(&[&pre_name, "glm_index_topk_prefill_m4096"], DEFAULT_INDEX_TOPK_PREFILL_SCRATCH);
            Ok(Self {
                topk_bytes: dec.max(pre).max(256),
                scored_bytes: 0,
                admitted_pages: usize::try_from(admitted_pages)?,
            })
        }
    }
}

/// Paged cache page strides for one Hy4 coordinator GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheLayout {
    /// Per-page bytes of one DSA MLA layer's KV records (`64 * record_bytes`).
    pub kv_page_bytes: usize,
    /// Per-page bytes of one full-indexer layer's index K cache (`64 * 132 = 8448 B`).
    pub index_page_bytes: usize,
    /// Number of MLA layers owned by this GPU.
    pub owned_kv_layers: usize,
    /// Number of full IndexCache layers owned by this GPU (shared layers own 0 index pages).
    pub owned_full_index_layers: usize,
}

impl CacheLayout {
    pub fn new(
        kv_format: Hy4KvCache,
        owned_kv_layers: usize,
        owned_full_index_layers: usize,
    ) -> Result<Self> {
        let kv_page_bytes = PAGE_ROWS
            .checked_mul(kv_format.record_bytes())
            .context("Hy4 KV page overflow")?;
        Ok(Self {
            kv_page_bytes,
            index_page_bytes: INDEX_PAGE_BYTES,
            owned_kv_layers,
            owned_full_index_layers,
        })
    }

    /// Total bytes per physical page owned by this GPU across all its MLA and full-indexer layers.
    pub fn persistent_unit_bytes(&self) -> Option<u64> {
        let kv = (self.owned_kv_layers as u64).checked_mul(self.kv_page_bytes as u64)?;
        let idx = (self.owned_full_index_layers as u64).checked_mul(self.index_page_bytes as u64)?;
        kv.checked_add(idx)
    }
}

/// Maximum parity-owned pages one prefill member of up to `max_rows` can commit.
pub fn context_commit_limit(max_rows: usize) -> usize {
    (max_rows.div_ceil(PAGE_ROWS) + 2).div_ceil(2) + 1
}

/// Number of commit planes in Hy4 gathered ContextSplit prefill: KV records, index keys, index scales.
pub const CONTEXT_COMMIT_PLANES: usize = 3;

/// One packed staging arena per coordinator GPU for 2-GPU ContextSplit gathered prefill.
/// Charged once per GPU by admission and aliased by runtime views (`kv`, `index`, `identity_pt`,
/// plus DFly2 empty-half sentinels).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextStagingLayout {
    pub arena: WeightArenaLayout,
    pub pages: usize,
    pub kv_page_bytes: usize,
    pub index_page_bytes: usize,
}

impl ContextStagingLayout {
    pub fn arena_id(gpu: usize) -> String {
        format!("hy4.context_staging.{gpu}")
    }

    pub fn new(
        cfg: &Hy4Config,
        kv_format: Hy4KvCache,
        gpu: usize,
        max_seq_pages: usize,
    ) -> Result<Self> {
        let pages = max_seq_pages.max(16);
        let kv_page_bytes = PAGE_ROWS
            .checked_mul(kv_format.record_bytes())
            .context("Hy4 ContextStaging KV page overflow")?;
        let index_page_bytes = INDEX_PAGE_BYTES;
        let kv_bytes = pages
            .checked_mul(kv_page_bytes)
            .context("Hy4 ContextStaging KV overflow")?;
        let index_bytes = pages
            .checked_mul(index_page_bytes)
            .context("Hy4 ContextStaging index overflow")?;
        let arena = WeightArenaLayout::new(
            Self::arena_id(gpu),
            vec![
                ("kv".into(), kv_bytes.max(256)),
                ("index".into(), index_bytes.max(256)),
                ("identity_pt".into(), (pages * 4).max(256)),
                ("empty_cand_scores".into(), (cfg.indexer_topk * 4).max(256)),
                ("empty_cand_indices".into(), (cfg.indexer_topk * 4).max(256)),
            ],
        )?;
        Ok(Self {
            arena,
            pages,
            kv_page_bytes,
            index_page_bytes,
        })
    }
}

/// Metadata uploaded to each rank before it consumes a wave.
pub const WAVE_METADATA: [&str; 7] = [
    "positions_i64",
    "positions_i32",
    "dsa_slots_i64",
    "dsa_page_table",
    "dsa_cu_seqlens",
    "idx_cache_lens",
    "idx_seq_page_tables",
];

/// Additional wave metadata planes uploaded when `context_split == true` (gathered prefill).
pub const WAVE_METADATA_CONTEXT: [&str; 9] = [
    "positions_i64",
    "positions_i32",
    "dsa_slots_i64",
    "dsa_page_table",
    "dsa_cu_seqlens",
    "idx_cache_lens",
    "idx_seq_page_tables",
    "ctx_gather_tables",
    "ctx_commit_tables",
];

pub fn wave_metadata_layout(slot: usize, wave: &WeightArenaLayout) -> Result<WeightArenaLayout> {
    wave_metadata_layout_for_rank(0, slot, wave)
}

pub fn wave_metadata_layout_for_rank(
    rank: usize,
    slot: usize,
    wave: &WeightArenaLayout,
) -> Result<WeightArenaLayout> {
    let names: &[&str] = if wave.view("ctx_gather_tables").is_some() {
        &WAVE_METADATA_CONTEXT
    } else {
        &WAVE_METADATA
    };
    let views = names
        .iter()
        .map(|&name| {
            let bytes = wave
                .view(name)
                .with_context(|| format!("Hy4 metadata view {name} missing"))?
                .bytes();
            Ok((name.to_owned(), bytes))
        })
        .collect::<Result<Vec<_>>>()?;
    let id = if rank == 0 {
        format!("hy4.wave.host_metadata.{slot}")
    } else {
        format!("hy4.wave.host_metadata.{rank}.{slot}")
    };
    WeightArenaLayout::new(id, views)
}

/// One packed scratch arena per coordinator GPU per in-flight prefill wave slot.
///
/// Holds every buffer that must survive across `PrefillHooks` boundaries for that slot,
/// INCLUDING `idx_cand_slots` and `idx_cand_lens` so IndexCache reuse across a full
/// indexer layer and its 3 following shared layers never collides across interleaved waves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveScratchLayout {
    pub arena: WeightArenaLayout,
}

impl WaveScratchLayout {
    pub fn arena_id(gpu: usize, slot: usize) -> String {
        format!("hy4.wave.{gpu}.{slot}")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &Hy4Config,
        gpu: usize,
        slot: usize,
        tp: usize,
        max_rows: usize,
        max_seq_pages: usize,
        max_sequences: usize,
    ) -> Result<Self> {
        Self::new_with_context(cfg, gpu, slot, tp, max_rows, max_seq_pages, max_sequences, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_context(
        cfg: &Hy4Config,
        gpu: usize,
        slot: usize,
        tp: usize,
        max_rows: usize,
        max_seq_pages: usize,
        max_sequences: usize,
        context_split: bool,
    ) -> Result<Self> {
        ensure!(
            matches!(tp, 1 | 2) && max_rows > 0 && max_sequences > 0 && (!context_split || tp == 2),
            "invalid Hy4 WaveScratchLayout shape"
        );
        let h = cfg.hidden;
        let hc = cfg.hc_mult;
        let pages = max_seq_pages.max(16);
        let tp2_f32_bytes = if tp == 2 { max_rows * h * 4 } else { 256 };

        let mut views: Vec<(String, usize)> = vec![
            ("positions_i64".into(), (max_rows * 24).max(256)),
            ("positions_i32".into(), (max_rows * 4).max(256)),
            ("dsa_slots_i64".into(), (max_rows * 8).max(256)),
            ("r_a".into(), (max_rows * hc * h * 2).max(256)),
            ("r_b".into(), (max_rows * hc * h * 2).max(256)),
            ("post_ffn".into(), (max_rows * hc * 4).max(256)),
            ("y".into(), (max_rows * h * 2).max(256)),
            ("shared_out".into(), (max_rows * h * 2).max(256)),
            ("shared_f32".into(), tp2_f32_bytes.max(256)),
            // Per-row page table for decode/scored_index_topk (`[max_rows, pages]`).
            ("dsa_page_table".into(), (max_rows * pages * 4).max(256)),
            // Per-sequence flat page tables for `index_topk_prefill` (`[max_sequences, pages]`).
            ("idx_seq_page_tables".into(), (max_sequences * pages * 4).max(256)),
            ("dsa_cu_seqlens".into(), ((max_rows + 1) * 4).max(256)),
            ("idx_cache_lens".into(), (max_rows * 4).max(256)),
            // Per-wave IndexCache selection reused across the 3 shared layers following each full layer.
            ("idx_cand_slots".into(), (max_rows * cfg.indexer_topk * 4).max(256)),
            ("idx_cand_lens".into(), (max_rows * 4).max(256)),
        ];
        if context_split {
            let gather_per_member = 2 * pages.div_ceil(2) * 4;
            let commit_limit = context_commit_limit(max_rows);
            let commit_per_member = CONTEXT_COMMIT_PLANES * 3 * commit_limit * 8;
            views.push(("ctx_gather_tables".into(), (max_sequences * gather_per_member).max(256)));
            views.push(("ctx_commit_tables".into(), (max_sequences * commit_per_member).max(256)));
        }
        let arena = WeightArenaLayout::new(Self::arena_id(gpu, slot), views)?;
        Ok(Self { arena })
    }
}

/// One packed hook-local scratch arena per coordinator GPU, shared across all wave slots
/// on that rank's ordered stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedScratchLayout {
    pub arena: WeightArenaLayout,
}

impl SharedScratchLayout {
    pub fn arena_id(gpu: usize) -> String {
        format!("hy4.scratch.{gpu}")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &Hy4Config,
        rank: usize,
        tp: usize,
        max_rows: usize,
        decode_rows: usize,
        local_vocab: usize,
        context_split: bool,
        index_topk: IndexTopkScratch,
    ) -> Result<Self> {
        ensure!(
            matches!(tp, 1 | 2)
                && rank < tp
                && max_rows > 0
                && decode_rows > 0
                && (!context_split || tp == 2),
            "invalid Hy4 SharedScratchLayout shape"
        );
        let h = cfg.hidden;
        let hc = cfg.hc_mult;
        let local_heads = cfg.heads / tp;
        let max_inter = cfg.intermediate / tp;
        let wire_row = h + h / 32;
        let tp2_f32_bytes = if tp == 2 { max_rows * h * 4 } else { 256 };
        let kv_a_dim = cfg.kv_lora_rank + cfg.qk_rope_dim;
        let qk_dim = cfg.qk_nope_dim + cfg.qk_rope_dim;
        let idx_h = cfg.indexer_heads;
        let idx_d = cfg.indexer_head_dim;

        let mut views: Vec<(String, usize)> = vec![
            ("ids".into(), (max_rows * 4).max(256)),
            ("r_head_in".into(), (HEAD_TILE_ROWS * hc * h * 2).max(256)),
            ("post".into(), (max_rows * hc * 4).max(256)),
            ("delta".into(), (max_rows * h * 2).max(256)),
            ("delta_f32".into(), tp2_f32_bytes.max(256)),
            ("delta_sum_f32".into(), tp2_f32_bytes.max(256)),
            ("q_a".into(), (max_rows * cfg.q_lora_rank * 2).max(256)),
            ("kv_a".into(), (max_rows * kv_a_dim * 2).max(256)),
            ("kv_a_padded".into(), (max_rows * KV_A_PADDED * 2).max(256)),
            ("q_resid".into(), (max_rows * cfg.q_lora_rank * 2).max(256)),
            ("q_raw".into(), (max_rows * local_heads * qk_dim * 2).max(256)),
            ("gate".into(), (max_rows * local_heads * cfg.v_head_dim * 2).max(256)),
            ("query".into(), (max_rows * local_heads * kv_a_dim * 2).max(256)),
            ("attn_out".into(), (max_rows * local_heads * cfg.kv_lora_rank * 2).max(256)),
            ("lse_out".into(), (max_rows * local_heads * 4).max(256)),
            ("uv_out".into(), (max_rows * local_heads * cfg.v_head_dim * 2).max(256)),
            (
                "attn_scratch".into(),
                mla_split_scratch_bytes(decode_rows, tp)
                    .max(if context_split { hy4_dsa_partial_scratch_bytes(decode_rows) } else { 0 })
                    .max(ATTN_SCRATCH_BYTES)
                    .max(256),
            ),
            // 32-head per-token DSA indexer scratch (full layers only)
            ("idx_q_raw".into(), (max_rows * idx_h * idx_d * 2).max(256)),
            ("idx_k_raw".into(), (max_rows * idx_d * 2).max(256)),
            ("idx_w_raw".into(), (max_rows * idx_h * 2).max(256)),
            ("idx_q_fp8".into(), (max_rows * idx_h * idx_d).max(256)),
            ("idx_w_out".into(), (max_rows * idx_h * 4).max(256)),
            ("idx_topk_scratch".into(), index_topk.topk_bytes.max(index_topk.scored_bytes).max(256)),
            // Dense / shared / direct MoE FFN scratch (no LatentMoE)
            ("ffn_gate".into(), (max_rows * max_inter * 2).max(256)),
            ("ffn_up".into(), (max_rows * max_inter * 2).max(256)),
            ("ffn_act".into(), (max_rows * max_inter * 2).max(256)),
            ("router_logits".into(), (max_rows * cfg.experts * 4).max(256)),
            ("router_select".into(), (max_rows * cfg.experts * 4).max(256)),
            ("route_ids".into(), (max_rows * cfg.topk * 4).max(256)),
            ("route_weights".into(), (max_rows * cfg.topk * 4).max(256)),
            ("wire_in".into(), (max_rows * wire_row).max(256)),
            ("routed_out".into(), (max_rows * h * 2).max(256)),
            (
                "logits_tile".into(),
                if tp == 2 { (HEAD_TILE_ROWS * local_vocab * 4).max(256) } else { 256 },
            ),
        ];
        if context_split {
            let half_heads = cfg.heads / 2;
            views.push(("ctx_cand_scores".into(), (decode_rows * cfg.indexer_topk * 4).max(256)));
            views.push(("ctx_cand_indices".into(), (decode_rows * cfg.indexer_topk * 4).max(256)));
            views.push(("ctx_merged_scores".into(), (decode_rows * cfg.indexer_topk * 4).max(256)));
            views.push(("ctx_merged_indices".into(), (decode_rows * cfg.indexer_topk * 4).max(256)));
            views.push(("ctx_partial_own".into(), (decode_rows * half_heads * cfg.kv_lora_rank * 2).max(256)));
            views.push(("ctx_lse_own".into(), (decode_rows * half_heads * 4).max(256)));
            views.push(("ctx_partial_peer".into(), (decode_rows * half_heads * cfg.kv_lora_rank * 2).max(256)));
            views.push(("ctx_lse_peer".into(), (decode_rows * half_heads * 4).max(256)));
            views.push(("ctx_draft_slots".into(), (decode_rows * cfg.indexer_topk * 4).max(256)));
            views.push(("ctx_draft_lens".into(), (decode_rows * 4).max(256)));
        }
        if rank == 0 {
            views.push(("head_logits_f32".into(), (HEAD_TILE_ROWS * cfg.vocab_size * 4).max(256)));
        }
        let arena = WeightArenaLayout::new(Self::arena_id(rank), views)?;
        Ok(Self { arena })
    }
}

/// Named scratch buffers owned by one in-flight execution unit (kept for unit-level inspection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitLayout {
    pub buffers: BTreeMap<String, usize>,
}

impl UnitLayout {
    pub fn new(cfg: &Hy4Config, rows: usize, head_parts: usize) -> Result<Self> {
        ensure!(rows > 0 && matches!(head_parts, 1 | 2), "invalid Hy4 UnitLayout shape");
        let max_seq_pages = cfg.max_position.div_ceil(PAGE_ROWS);
        let topk = IndexTopkScratch::from_manifest_value(None, head_parts, false, cfg.max_position)?;
        let shared = SharedScratchLayout::new(
            cfg,
            0,
            head_parts,
            rows,
            rows.min(64),
            cfg.vocab_size / head_parts,
            false,
            topk,
        )?;
        let wave = WaveScratchLayout::new(cfg, 0, 0, head_parts, rows, max_seq_pages, 16)?;
        let mut buffers = BTreeMap::new();
        for (k, v) in shared.arena.views() {
            buffers.insert(format!("shared.{k}"), v.bytes());
        }
        for (k, v) in wave.arena.views() {
            buffers.insert(format!("wave.{k}"), v.bytes());
        }
        Ok(Self { buffers })
    }

    pub fn total_bytes(&self) -> usize {
        self.buffers.values().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::hy4::{Hy4IndexerKind, IndexCacheSchedule};

    fn test_cfg() -> Hy4Config {
        let kinds: Vec<Hy4IndexerKind> = (0..78)
            .map(|i| {
                if i == 0 || (i >= 1 && (i - 1) % 4 == 0) {
                    Hy4IndexerKind::Full
                } else {
                    Hy4IndexerKind::Shared
                }
            })
            .collect();
        Hy4Config {
            vocab_size: 120_832,
            hidden: 6144,
            layers: 78,
            intermediate: 18_432,
            max_position: 1_048_576,
            rms_norm_eps: 1e-5,
            qa_kva_eps: 1e-6,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            heads: 64,
            q_lora_rank: 2048,
            kv_lora_rank: 512,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            rope_theta: 10_000_000.0,
            indexer_heads: 32,
            indexer_head_dim: 128,
            indexer_topk: 2048,
            index_cache: IndexCacheSchedule::from_kinds(kinds).unwrap(),
            dense_layer: (0..78).map(|i| i == 0).collect(),
            experts: 256,
            topk: 8,
            moe_intermediate: 2048,
            shared_experts: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.827,
            swiglu_limit: 10.0,
            image_token_id: 120_120,
            video_token_id: 120_124,
            eos: vec![120_025],
        }
    }

    #[test]
    fn packed_scratch_and_wave_metadata_are_aligned_and_disjoint() -> Result<()> {
        let cfg = test_cfg();
        for (tp, ctx) in [(1usize, false), (2, false), (2, true)] {
            let topk = IndexTopkScratch::from_manifest_value(None, tp, ctx, 262_144)?;
            for rank in 0..tp {
                let shared = SharedScratchLayout::new(
                    &cfg,
                    rank,
                    tp,
                    2048,
                    64,
                    cfg.vocab_size / tp,
                    ctx,
                    topk,
                )?;
                assert_eq!(shared.arena.views().contains_key("head_logits_f32"), rank == 0);
                assert_eq!(shared.arena.views().contains_key("ctx_cand_scores"), ctx);
                for (name, view) in shared.arena.views() {
                    assert_eq!(view.offset() % 256, 0, "shared {name} unaligned");
                    assert!(view.bytes() >= 256);
                }
                for slot in 0..3 {
                    let wave = WaveScratchLayout::new(&cfg, rank, slot, tp, 2048, 4096, 16)?;
                    assert_eq!(wave.arena.id(), format!("hy4.wave.{rank}.{slot}"));
                    assert!(wave.arena.view("idx_cand_slots").is_some());
                    assert!(wave.arena.view("idx_cand_lens").is_some());
                    let host = wave_metadata_layout_for_rank(rank, slot, &wave.arena)?;
                    assert_eq!(host.views().len(), WAVE_METADATA.len());
                    for name in WAVE_METADATA {
                        assert_eq!(host.view(name).unwrap().bytes(), wave.arena.view(name).unwrap().bytes());
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn mla_partial_and_peer_slots_match_hy4_geometry() {
        let cfg = test_cfg();
        assert_eq!(mla_split_scratch_bytes(64, 1), 64 * 16 * 64 * 1028);
        assert_eq!(mla_split_scratch_bytes(64, 2), 64 * 16 * 32 * 1028);
        let n = 64 * 32 * 16;
        assert_eq!(hy4_dsa_partial_scratch_bytes(64), n * 1024 + n * 4);
        let [attn, ffn] = hy4_peer_slot_sizes(&cfg, 64);
        assert_eq!(attn, 64 * 6144 * 4);
        assert_eq!(ffn, hy4_peer_ffn_routed_offset(&cfg, 64) + 64 * 6144 * 4);
    }

    #[test]
    fn context_split_staging_and_index_topk_scratch_match_gathered_prefill() -> Result<()> {
        let cfg = test_cfg();
        let manifest = serde_json::json!({
            "programs": [
                {"name": "hy42_index_topk_prefill", "scratch_bytes_at_capacity": {"scratch": 4_194_304}},
                {"name": "hy42_scored_index_topk_prefill", "scratch_bytes_at_capacity": {"scratch": 8_388_608}},
                {"name": "hy42_scored_index_topk_decode", "scratch_bytes_at_capacity": {"scratch": 1_048_576}}
            ]
        });
        let topk = IndexTopkScratch::from_manifest_value(Some(&manifest), 2, true, 131_072)?;
        assert_eq!(topk.topk_bytes, 8_388_608);
        assert_eq!(topk.scored_bytes, 1_048_576);
        assert_eq!(topk.topk_bytes.max(topk.scored_bytes), 8_388_608);

        let staging = ContextStagingLayout::new(&cfg, Hy4KvCache::Fp8, 0, 2048)?;
        assert_eq!(staging.arena.id(), "hy4.context_staging.0");
        for name in ["kv", "index", "identity_pt", "empty_cand_scores", "empty_cand_indices"] {
            let v = staging.arena.view(name).expect(name);
            assert_eq!(v.offset() % 256, 0, "staging {name} unaligned");
            assert!(v.bytes() >= 256);
        }

        let wave = WaveScratchLayout::new_with_context(&cfg, 0, 0, 2, 4096, 2048, 16, true)?;
        assert!(wave.arena.view("ctx_gather_tables").is_some());
        assert!(wave.arena.view("ctx_commit_tables").is_some());
        let host = wave_metadata_layout_for_rank(0, 0, &wave.arena)?;
        assert_eq!(host.views().len(), WAVE_METADATA_CONTEXT.len());
        for name in WAVE_METADATA_CONTEXT {
            assert_eq!(host.view(name).unwrap().bytes(), wave.arena.view(name).unwrap().bytes());
        }
        Ok(())
    }
}
