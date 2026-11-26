//! Named physical cache, prefix-mark, and packed scratch arena layouts for Hy4-flash.
//!
//! Shared by `cuteafd plan --layout` and runtime admission so planned and
//! allocated bytes match owner for owner.
use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};

use crate::serving_capacity::WeightArenaLayout;

use super::{Hy4FlashConfig, Hy4KvCache};

/// KV page quantum for the 12 DSA layers (64-token pages).
pub const PAGE_ROWS: usize = 64;
/// Minimum per-sequence SWA-512 circular ring rows: 10 pages = 640 rows
/// (`512` window + up to `64` decode/verify rows + page wrap alignment).
pub const SWA_RING_PAGES: usize = 10;
pub const SWA_RING_ROWS: usize = SWA_RING_PAGES * PAGE_ROWS;
/// Fixed byte offset of peer routed-expert F32 partials inside `PeerExchange` FFN receive slot 1.
/// Shared-expert (or dense FFN) F32 partials sit at offset 0; routed F32 partials sit at this
/// 256-byte-aligned offset so both transfer in one `push_planes` exchange with invariant graph pointers.
pub fn hy4_peer_ffn_routed_offset(cfg: &Hy4FlashConfig, max_rows: usize) -> usize {
    (max_rows * cfg.hidden * 4).div_ceil(256) * 256
}

/// Receive-slot byte sizes for Hy4-flash TP2 `PeerExchange` (`[SLOT_ATTN, SLOT_FFN]`).
/// - Slot 0 (`ATTN`): attention `o_proj` F32 partials (`[max_rows, hidden]`).
/// - Slot 1 (`FFN`): dense FFN or combined `[shared_f32, routed_f32]` F32 partials.
pub fn hy4_peer_slot_sizes(cfg: &Hy4FlashConfig, max_rows: usize) -> [usize; 2] {
    let attn = (max_rows * cfg.hidden * 4).max(256);
    let routed_off = hy4_peer_ffn_routed_offset(cfg, max_rows);
    let ffn = (routed_off + max_rows * cfg.routed_expert_hidden * 4).max(256);
    [attn, ffn]
}

/// Physical partial O (BF16) and LSE (F32) planes for the capped MLA decode exports.
/// The export strides scratch by the compiled split capacity, not active splits.
pub fn mla_split_scratch_bytes(decode_rows: usize, head_parts: usize) -> usize {
    decode_rows * 16 * (64 / head_parts) * (512 * 2 + 4)
}

/// Scratch bytes required by `hy4f2_mla_dsa_partial_{bf16,fp8}_h{0,16}` (chunked at 64 rows,
/// 16 heads, 16 splits; mirrors `hy4_dsa_partial_scratch_bytes` in `hy4_mla.py`).
pub fn hy4_dsa_partial_scratch_bytes(rows: usize) -> usize {
    let chunk = rows.clamp(1, 64);
    let n = chunk * 16 * 16;
    let part = (n * 512 * 2).div_ceil(1024) * 1024;
    let lse = (n * 4).div_ceil(1024) * 1024;
    part + lse
}

/// Maximum rows of the installed ContextSplit partial programs (internally
/// tiled at 64 rows). The index score workspace is independently sub-tiled.
pub const CONTEXT_PREFILL_ROWS: usize = 4096;

/// Physical exchange and scratch width, shared by admission and execution.
pub fn context_exchange_rows(max_rows: usize, decode_rows: usize) -> usize {
    max_rows.min(CONTEXT_PREFILL_ROWS).max(decode_rows)
}

/// Decode and verify must remain a single exchange (and one graph shape).
pub fn context_chunk_rows(max_rows: usize, decode_rows: usize, decode: bool) -> usize {
    if decode { decode_rows } else { context_exchange_rows(max_rows, decode_rows) }
}

/// Shared `ContextExchange` buffer specification for Hy4-flash 2-GPU ContextSplit DSA layers
/// (16 heads/GPU, 576-dim absorbed query, 512 scored group candidates, 2 alternating slots).
pub fn hy4_context_buffers(cfg: &Hy4FlashConfig, max_rows: usize, decode_rows: usize) -> crate::placement::ContextBuffers {
    let half_heads = (cfg.sparse_heads / 2) as u64;
    let kv_a_dim = (cfg.kv_lora_rank + cfg.qk_rope_dim) as u64;
    let group_topk = cfg.indexer_group_topk as u64;
    crate::placement::ContextBuffers {
        staging_unit_bytes: 0,
        staging_unit_rows: 1,
        query_row_bytes: half_heads * kv_a_dim * 2,
        partial_row_bytes: half_heads * (cfg.kv_lora_rank as u64 * 2 + 4),
        candidate_row_bytes: group_topk * 8,
        compiled_extent: 0,
        decode_rows: context_exchange_rows(max_rows, decode_rows) as u64,
        lanes: 1,
        workspace_bytes: 0,
    }
}

/// Compressed indexer group-cache record bytes: 128 E4M3 dims + 4B F32 scale.
pub const INDEX_GROUP_RECORD_BYTES: usize = 132;
/// Maximum rows projected through `lm_head` in one tile.
pub const HEAD_TILE_ROWS: usize = 64;
/// Cap `idx_scores` at 64 MiB (16M floats); long-context prefill tiles rows automatically.
pub const MAX_IDX_SCORE_FLOATS: usize = 16 * 1024 * 1024;
/// Padded `kv_a` width required by compact core projections (128-aligned >= 576).
pub const KV_A_PADDED: usize = 640;
/// Active-split-1 MLA scratch size matching `PROGRAMS.json` (`hy4f_mla_*`).
pub const ATTN_SCRATCH_BYTES: usize = 1024;

/// Per-sequence SWA circular ring pages required to hold the live `sliding_window`
/// plus one full prefill unit without overwriting any token still needed by queries
/// earlier in the same unit.
pub fn swa_ring_pages(sliding_window: usize, unit_rows: usize) -> usize {
    ((sliding_window + unit_rows).div_ceil(PAGE_ROWS) + 1).max(SWA_RING_PAGES)
}

/// Persistent per-sequence SWA-512 ring & prefix-mark bytes for one GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLayout {
    /// Per-sequence SWA ring pages (`>= SWA_RING_PAGES`).
    pub swa_ring_pages: usize,
    /// Per-sequence SWA ring rows (`swa_ring_pages * PAGE_ROWS`).
    pub swa_ring_rows: usize,
    /// Active SWA-512 rings across all admitted sequences and owned SWA layers.
    pub active_swa_ring_bytes: usize,
    /// Pending incomplete compressed-indexer token keys (`compress_ratio - 1 = 3`
    /// tokens × `128` F32) across admitted sequences and owned DSA layers.
    pub active_indexer_pending_bytes: usize,
    /// Exact per-snapshot prefix-mark bytes (30 SWA-512 windows + 12 DSA pending
    /// indexer tokens).
    pub mark_bytes_per_slot: usize,
    /// Per-page bytes of one DSA layer's KV records (`64 * record_bytes`).
    pub dsa_kv_page_bytes: usize,
    /// Per-page bytes of one DSA layer's compressed-indexer group keys
    /// (`(64 / compress_ratio) * INDEX_GROUP_RECORD_BYTES = 16 * 132 = 2112 B`).
    pub dsa_index_page_bytes: usize,
}

impl CacheLayout {
    pub fn new(
        cfg: &Hy4FlashConfig,
        kv_format: Hy4KvCache,
        sequences: usize,
        owned_dsa_layers: usize,
        owned_swa_layers: usize,
    ) -> Result<Self> {
        Self::for_unit(cfg, kv_format, sequences, owned_dsa_layers, owned_swa_layers, 0)
    }

    pub fn for_unit(
        cfg: &Hy4FlashConfig,
        kv_format: Hy4KvCache,
        sequences: usize,
        owned_dsa_layers: usize,
        owned_swa_layers: usize,
        unit_rows: usize,
    ) -> Result<Self> {
        ensure!(sequences > 0, "Hy4-flash requires at least one admitted sequence");
        let rec = kv_format.record_bytes();
        let dsa_kv_page_bytes = PAGE_ROWS.checked_mul(rec).context("DSA KV page overflow")?;
        let groups_per_page = PAGE_ROWS / cfg.indexer_compress_ratio;
        let dsa_index_page_bytes = groups_per_page
            .checked_mul(INDEX_GROUP_RECORD_BYTES)
            .context("DSA index page overflow")?;

        let swa_ring_pages = swa_ring_pages(cfg.sliding_window, unit_rows);
        let swa_ring_rows = swa_ring_pages
            .checked_mul(PAGE_ROWS)
            .context("SWA ring rows overflow")?;
        let swa_ring_per_seq = owned_swa_layers
            .checked_mul(swa_ring_rows)
            .and_then(|v| v.checked_mul(rec))
            .context("SWA ring overflow")?;
        let active_swa_ring_bytes = swa_ring_per_seq
            .checked_mul(sequences)
            .context("active SWA ring overflow")?;

        // Exact prefix mark keeps the last `sliding_window` (512) SWA records per SWA layer
        // plus `compress_ratio - 1` (3) pending token keys (F32[128]) per DSA layer.
        let swa_mark_per_seq = owned_swa_layers
            .checked_mul(cfg.sliding_window)
            .and_then(|v| v.checked_mul(rec))
            .context("SWA mark overflow")?;
        let pending_per_seq = owned_dsa_layers
            .checked_mul(cfg.indexer_compress_ratio - 1)
            .and_then(|v| v.checked_mul(cfg.indexer_head_dim))
            .and_then(|v| v.checked_mul(4))
            .context("indexer pending overflow")?;
        let active_indexer_pending_bytes = pending_per_seq
            .checked_mul(sequences)
            .context("active indexer pending overflow")?;
        let mark_bytes_per_slot = swa_mark_per_seq
            .checked_add(pending_per_seq)
            .context("mark bytes overflow")?;

        Ok(Self {
            swa_ring_pages,
            swa_ring_rows,
            active_swa_ring_bytes,
            active_indexer_pending_bytes,
            mark_bytes_per_slot,
            dsa_kv_page_bytes,
            dsa_index_page_bytes,
        })
    }
}

/// Metadata uploaded to each rank before it consumes a wave. One pinned
/// source backs both ranks, with the same capacities as the device views.
pub const WAVE_METADATA: [&str; 12] = [
    "positions_i64", "positions_i32", "dsa_slots_i64", "swa_slots_i64",
    "dsa_page_table", "swa_page_table", "dsa_cu_seqlens", "swa_cu_seqlens",
    "swa_seqlens", "idx_limits", "idx_group_pos", "idx_group_slots",
];

pub fn wave_metadata_layout(slot: usize, wave: &WeightArenaLayout) -> Result<WeightArenaLayout> {
    wave_metadata_layout_for_rank(0, slot, wave)
}

pub fn wave_metadata_layout_for_rank(rank: usize, slot: usize, wave: &WeightArenaLayout) -> Result<WeightArenaLayout> {
    let views = WAVE_METADATA.iter().map(|&name| {
        let bytes = wave.view(name).with_context(|| format!("Hy4 metadata view {name} missing"))?.bytes();
        Ok((name.to_owned(), bytes))
    }).collect::<Result<Vec<_>>>()?;
    let id = if rank == 0 {
        format!("hy4_flash.wave.host_metadata.{slot}")
    } else {
        format!("hy4_flash.wave.host_metadata.{rank}.{slot}")
    };
    WeightArenaLayout::new(id, views)
}

/// One packed scratch arena per coordinator GPU per in-flight prefill wave slot.
/// Holds every buffer that must survive across `PrefillHooks` boundaries for that slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveScratchLayout {
    pub arena: WeightArenaLayout,
}

impl WaveScratchLayout {
    pub fn arena_id(gpu: usize, slot: usize) -> String {
        format!("hy4_flash.wave.{gpu}.{slot}")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &Hy4FlashConfig,
        gpu: usize,
        slot: usize,
        tp: usize,
        max_rows: usize,
        max_seq_pages: usize,
        max_sequences: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(tp, 1 | 2) && max_rows > 0 && max_sequences > 0,
            "invalid Hy4-flash WaveScratchLayout shape"
        );
        let h = cfg.hidden;
        let hc = cfg.hc_mult;
        let pages = max_seq_pages.max(16);
        let max_batch_groups = max_rows / cfg.indexer_compress_ratio + max_sequences + 2;
        let tp2_f32_bytes = if tp == 2 { max_rows * h * 4 } else { 256 };
        let engram_concat = cfg.engram_concat_width();

        let mut views: Vec<(String, usize)> = vec![
            ("positions_i64".into(), (max_rows * 24).max(256)),
            ("positions_i32".into(), (max_rows * 4).max(256)),
            ("dsa_slots_i64".into(), (max_rows * 8).max(256)),
            ("swa_slots_i64".into(), (max_rows * 8).max(256)),
            ("r_a".into(), (max_rows * hc * h * 2).max(256)),
            ("r_b".into(), (max_rows * hc * h * 2).max(256)),
            ("post_ffn".into(), (max_rows * hc * 4).max(256)),
            ("y".into(), (max_rows * h * 2).max(256)),
            ("shared_out".into(), (max_rows * h * 2).max(256)),
            ("shared_f32".into(), tp2_f32_bytes.max(256)),
            ("dsa_page_table".into(), (max_rows * pages * 4).max(256)),
            ("swa_page_table".into(), (max_sequences * pages * 4).max(256)),
            ("swa_seqlens".into(), (max_sequences * 4).max(256)),
            ("dsa_cu_seqlens".into(), ((max_rows + 1) * 4).max(256)),
            ("swa_cu_seqlens".into(), ((max_sequences + 1) * 4).max(256)),
            ("idx_limits".into(), (max_rows * 4).max(256)),
            ("idx_group_pos".into(), (max_batch_groups * 24).max(256)),
            ("idx_group_slots".into(), (max_batch_groups * 8).max(256)),
        ];
        for i in 0..cfg.engram_layers.len() {
            views.push((format!("engram_raw.{i}"), (max_rows * engram_concat * 2).max(256)));
            views.push((format!("engram_masked.{i}"), (max_rows.div_ceil(16) * 16).max(256)));
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
    pub idx_score_floats: usize,
}

impl SharedScratchLayout {
    pub fn arena_id(gpu: usize) -> String {
        format!("hy4_flash.scratch.{gpu}")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &Hy4FlashConfig,
        rank: usize,
        tp: usize,
        max_rows: usize,
        decode_rows: usize,
        max_seq_pages: usize,
        max_sequences: usize,
        local_vocab: usize,
    ) -> Result<Self> {
        Self::new_with_context(
            cfg,
            rank,
            tp,
            max_rows,
            decode_rows,
            max_seq_pages,
            max_sequences,
            local_vocab,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_context(
        cfg: &Hy4FlashConfig,
        rank: usize,
        tp: usize,
        max_rows: usize,
        decode_rows: usize,
        max_seq_pages: usize,
        max_sequences: usize,
        local_vocab: usize,
        context_split: bool,
    ) -> Result<Self> {
        ensure!(
            matches!(tp, 1 | 2) && rank < tp && max_rows > 0 && decode_rows > 0 && max_sequences > 0
                && (!context_split || tp == 2),
            "invalid Hy4-flash SharedScratchLayout shape"
        );
        let h = cfg.hidden;
        let hc = cfg.hc_mult;
        let max_heads = cfg.swa_heads.max(cfg.sparse_heads) / tp;
        let max_inter = cfg.intermediate / tp;
        let pages = max_seq_pages.max(16);
        let max_seq_groups = pages * (PAGE_ROWS / cfg.indexer_compress_ratio);
        let idx_score_floats = max_rows
            .saturating_mul(max_seq_groups.max(1))
            .min(MAX_IDX_SCORE_FLOATS)
            .max(max_seq_groups.max(1));
        let wire_row = cfg.routed_expert_hidden + cfg.routed_expert_hidden / 32;
        let tp2_f32_bytes = if tp == 2 { max_rows * h * 4 } else { 256 };
        let kv_a_dim = cfg.kv_lora_rank + cfg.qk_rope_dim;
        let qk_dim = cfg.qk_nope_dim + cfg.qk_rope_dim;
        let idx_h = cfg.indexer_heads;
        let idx_d = cfg.indexer_head_dim;
        let engram_concat = cfg.engram_concat_width();

        let mut views: Vec<(String, usize)> = vec![
            ("ids".into(), (max_rows * 4).max(256)),
            ("r_head_in".into(), (HEAD_TILE_ROWS * hc * h * 2).max(256)),
            ("post".into(), (max_rows * hc * 4).max(256)),
            ("delta".into(), (max_rows * h * 2).max(256)),
            ("delta_f32".into(), tp2_f32_bytes.max(256)),
            ("delta_sum_f32".into(), tp2_f32_bytes.max(256)),
            ("delta_norm".into(), (max_rows * h * 2).max(256)),
            ("engram_normed".into(), (max_rows * engram_concat * 2).max(256)),
            ("engram_val".into(), (max_rows * h * 2).max(256)),
            ("engram_keys".into(), (max_rows * hc * h * 2).max(256)),
            ("q_a".into(), (max_rows * cfg.q_lora_rank * 2).max(256)),
            ("kv_a".into(), (max_rows * kv_a_dim * 2).max(256)),
            ("kv_a_padded".into(), (max_rows * KV_A_PADDED * 2).max(256)),
            ("q_resid".into(), (max_rows * cfg.q_lora_rank * 2).max(256)),
            ("q_raw".into(), (max_rows * max_heads * qk_dim * 2).max(256)),
            ("gate".into(), (max_rows * max_heads * cfg.v_head_dim * 2).max(256)),
            ("query".into(), (max_rows * max_heads * kv_a_dim * 2).max(256)),
            ("attn_out".into(), (max_rows * max_heads * cfg.kv_lora_rank * 2).max(256)),
            ("lse_out".into(), (max_rows * max_heads * 4).max(256)),
            ("uv_out".into(), (max_rows * max_heads * cfg.v_head_dim * 2).max(256)),
            (
                "attn_scratch".into(),
                mla_split_scratch_bytes(decode_rows, tp)
                    .max(if context_split { hy4_dsa_partial_scratch_bytes(context_exchange_rows(max_rows, decode_rows)) } else { 0 })
                    .max(ATTN_SCRATCH_BYTES)
                    .max(256),
            ),
            ("idx_q_raw".into(), (max_rows * idx_h * idx_d * 2).max(256)),
            ("idx_q_rot".into(), (max_rows * idx_h * idx_d * 2).max(256)),
            ("idx_w_raw".into(), (max_rows * idx_h * 2).max(256)),
            ("idx_w_out".into(), (max_rows * idx_h * 4).max(256)),
            ("idx_tk_raw".into(), (max_rows * idx_d * 2).max(256)),
            ("idx_tk".into(), (max_rows * idx_d * 2).max(256)),
            ("idx_group_tok".into(), ((max_rows + cfg.indexer_compress_ratio * max_sequences) * idx_d * 2).max(256)),
            ("idx_scores".into(), (idx_score_floats * 4).max(256)),
            ("idx_cand_slots".into(), (max_rows * cfg.indexer_candidate_topk * 4).max(256)),
            ("idx_cand_lens".into(), (max_rows * 4).max(256)),
            ("ffn_gate".into(), (max_rows * max_inter * 2).max(256)),
            ("ffn_up".into(), (max_rows * max_inter * 2).max(256)),
            ("ffn_act".into(), (max_rows * max_inter * 2).max(256)),
            ("router_logits".into(), (max_rows * cfg.experts * 4).max(256)),
            ("router_select".into(), (max_rows * cfg.experts * 4).max(256)),
            ("route_ids".into(), (max_rows * cfg.topk * 4).max(256)),
            ("route_weights".into(), (max_rows * cfg.topk * 4).max(256)),
            ("latent_in".into(), (max_rows * cfg.routed_expert_hidden * 2).max(256)),
            ("wire_in".into(), (max_rows * wire_row).max(256)),
            ("routed_latent".into(), (max_rows * cfg.routed_expert_hidden * 2).max(256)),
            ("routed_out".into(), (max_rows * h * 2).max(256)),
            (
                "logits_tile".into(),
                if tp == 2 { (HEAD_TILE_ROWS * local_vocab * 4).max(256) } else { 256 },
            ),
        ];
        if context_split {
            let half_dsa_heads = cfg.sparse_heads / 2;
            let ctx_rows = context_exchange_rows(max_rows, decode_rows);
            views.push(("ctx_cand_scores".into(), (ctx_rows * cfg.indexer_group_topk * 4).max(256)));
            views.push(("ctx_cand_indices".into(), (ctx_rows * cfg.indexer_group_topk * 4).max(256)));
            views.push(("ctx_partial_own".into(), (ctx_rows * half_dsa_heads * cfg.kv_lora_rank * 2).max(256)));
            views.push(("ctx_lse_own".into(), (ctx_rows * half_dsa_heads * 4).max(256)));
            views.push(("ctx_partial_peer".into(), (ctx_rows * half_dsa_heads * cfg.kv_lora_rank * 2).max(256)));
            views.push(("ctx_lse_peer".into(), (ctx_rows * half_dsa_heads * 4).max(256)));
            views.push(("ctx_draft_slots".into(), (ctx_rows * cfg.indexer_candidate_topk * 4).max(256)));
        }
        if rank == 0 {
            views.push(("head_logits_f32".into(), (HEAD_TILE_ROWS * cfg.vocab_size * 4).max(256)));
        }
        let arena = WeightArenaLayout::new(Self::arena_id(rank), views)?;
        Ok(Self { arena, idx_score_floats })
    }
}

/// Named scratch buffers owned by one in-flight execution unit (kept for unit-level inspection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitLayout {
    pub buffers: BTreeMap<String, usize>,
}

impl UnitLayout {
    pub fn new(
        cfg: &Hy4FlashConfig,
        kv_format: Hy4KvCache,
        rows: usize,
        head_parts: usize,
    ) -> Result<Self> {
        let _ = kv_format;
        let max_seq_pages = cfg.max_position.div_ceil(PAGE_ROWS);
        let shared = SharedScratchLayout::new(cfg, 0, head_parts, rows, rows.min(64), max_seq_pages, 16, cfg.vocab_size / head_parts)?;
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

    #[test]
    fn host_metadata_covers_the_same_views_at_both_ranks() -> Result<()> {
        let cfg = test_cfg();
        let wave0 = WaveScratchLayout::new(&cfg, 0, 2, 2, 2752, 16_384, 16)?;
        let wave1 = WaveScratchLayout::new(&cfg, 1, 2, 2, 2752, 16_384, 16)?;
        let host = wave_metadata_layout(2, &wave0.arena)?;
        assert_eq!(host.id(), "hy4_flash.wave.host_metadata.2");
        assert_eq!(host.views().len(), WAVE_METADATA.len());
        for name in WAVE_METADATA {
            let view = host.view(name).expect("metadata plane");
            assert_eq!(view.bytes(), wave0.arena.view(name).unwrap().bytes());
            assert_eq!(view.bytes(), wave1.arena.view(name).unwrap().bytes());
            assert!(view.offset() + view.bytes() <= host.bytes());
        }
        assert!(host.bytes() < wave0.arena.bytes());
        Ok(())
    }

    #[test]
    fn mla_split_partials_scale_with_resolved_rows_and_head_parts() {
        let full = super::mla_split_scratch_bytes(64, 1);
        assert_eq!(full, 64 * 16 * 64 * 1028);
        assert_eq!(super::mla_split_scratch_bytes(32, 1), full / 2);
        assert_eq!(super::mla_split_scratch_bytes(32, 2), full / 4);
    }

    fn test_cfg() -> Hy4FlashConfig {
        Hy4FlashConfig {
            vocab_size: 120_832,
            hidden: 3072,
            layers: 42,
            intermediate: 13_824,
            max_position: 262_144,
            rms_norm_eps: 1e-5,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            q_lora_rank: 1024,
            kv_lora_rank: 512,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            sparse_heads: 32,
            swa_heads: 64,
            swa_qk_nope_dim: 192,
            swa_qk_rope_dim: 64,
            swa_v_head_dim: 256,
            sliding_window: 512,
            sparse_rope_theta: 10_000_000.0,
            swa_rope_theta: 10_000.0,
            sparse_layer: (0..42).map(|i| i % 4 == 0).collect(),
            indexer_heads: 16,
            indexer_head_dim: 128,
            indexer_compress_ratio: 4,
            indexer_group_topk: 512,
            indexer_candidate_topk: 2048,
            dense_layer: (0..42).map(|i| i == 0).collect(),
            experts: 512,
            topk: 16,
            moe_intermediate: 1536,
            routed_expert_hidden: 1536,
            shared_experts: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 3.998,
            swiglu_limit: 10.0,
            engram_layers: vec![1, 11, 21, 31],
            engram_vocab_sizes: BTreeMap::new(),
            engram_max_ngram_size: 3,
            engram_n_embed_per_ngram: 768,
            engram_n_head_per_ngram: 8,
            engram_pad_id: 120_002,
            engram_seed: 10_007,
            engram_fp8_embedding: true,
            engram_lookup_table_size: 120_818,
            image_token_id: 120_120,
            video_token_id: 120_124,
            eos: vec![120_025],
        }
    }

    #[test]
    fn packed_scratch_arenas_are_aligned_and_disjoint() -> Result<()> {
        let cfg = test_cfg();
        for tp in [1usize, 2] {
            for rank in 0..tp {
                let shared = SharedScratchLayout::new(&cfg, rank, tp, 2752, 64, 4096, 16, cfg.vocab_size / tp)?;
                assert_eq!(shared.arena.views().contains_key("head_logits_f32"), rank == 0);
                for (name, view) in shared.arena.views() {
                    assert_eq!(view.offset() % 256, 0, "shared {name} unaligned");
                    assert!(view.bytes() >= 256);
                }
                let mut wave_ids = std::collections::BTreeSet::new();
                for slot in 0..3 {
                    let wave = WaveScratchLayout::new(&cfg, rank, slot, tp, 2752, 4096, 16)?;
                    let id = wave.arena.id().to_string();
                    assert_eq!(id, format!("hy4_flash.wave.{rank}.{slot}"));
                    assert!(wave_ids.insert(id), "duplicate wave arena id across slots");
                    let mut spans = Vec::new();
                    for (name, view) in wave.arena.views() {
                        assert_eq!(view.offset() % 256, 0, "wave {name} unaligned");
                        assert!(view.bytes() >= 256);
                        spans.push((view.offset(), view.offset() + view.bytes(), name.as_str()));
                    }
                    spans.sort_by_key(|&(s, _, _)| s);
                    for pair in spans.windows(2) {
                        assert!(
                            pair[0].1 <= pair[1].0,
                            "wave views overlap: {} [{}..{}) vs {} [{}..{})",
                            pair[0].2, pair[0].0, pair[0].1, pair[1].2, pair[1].0, pair[1].1
                        );
                    }
                    for i in 0..cfg.engram_layers.len() {
                        assert!(wave.arena.view(&format!("engram_raw.{i}")).is_some());
                        assert!(wave.arena.view(&format!("engram_masked.{i}")).is_some());
                    }
                }
            }
        }
        let cache = CacheLayout::for_unit(&cfg, Hy4KvCache::Fp8, 16, 12, 30, 2752)?;
        assert_eq!(cache.swa_ring_pages, swa_ring_pages(512, 2752));
        assert_eq!(cache.swa_ring_rows, cache.swa_ring_pages * PAGE_ROWS);
        Ok(())
    }

    #[test]
    fn attn_scratch_view_covers_runtime_split_k_cap_across_tp_and_rows() -> Result<()> {
        let cfg = test_cfg();
        for tp in [1usize, 2] {
            for decode_rows in [1usize, 32, 64] {
                let runtime_split_k_cap =
                    mla_split_scratch_bytes(decode_rows, tp).max(ATTN_SCRATCH_BYTES);
                for unit_rows in [1usize, 64, 128, 512, 2752, 4096] {
                    let capacity_rows = unit_rows.max(decode_rows);
                    for rank in 0..tp {
                        let shared = SharedScratchLayout::new(
                            &cfg,
                            rank,
                            tp,
                            capacity_rows,
                            decode_rows,
                            64,
                            16,
                            cfg.vocab_size / tp,
                        )?;
                        let view = shared
                            .arena
                            .view("attn_scratch")
                            .expect("SharedScratchLayout missing attn_scratch");
                        assert!(
                            view.bytes() >= runtime_split_k_cap,
                            "tp={tp} rank={rank} unit_rows={unit_rows} decode_rows={decode_rows}: admitted attn_scratch {} < runtime split_k_cap {runtime_split_k_cap}",
                            view.bytes()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn context_split_scratch_and_exchange_layout_match_bounds() -> Result<()> {
        let cfg = test_cfg();
        let decode_rows = 64usize;
        let partial_req = hy4_dsa_partial_scratch_bytes(decode_rows);
        assert!(partial_req > 0);
        assert_eq!(hy4_dsa_partial_scratch_bytes(4096), 16_842_752);

        for rank in 0..2 {
            let shared = SharedScratchLayout::new_with_context(
                &cfg,
                rank,
                2,
                2752,
                decode_rows,
                64,
                16,
                cfg.vocab_size / 2,
                true,
            )?;
            for view_name in [
                "ctx_cand_scores",
                "ctx_cand_indices",
                "ctx_partial_own",
                "ctx_lse_own",
                "ctx_partial_peer",
                "ctx_lse_peer",
                "ctx_draft_slots",
            ] {
                assert!(shared.arena.view(view_name).is_some(), "missing {view_name} view");
            }
            let attn_bytes = shared.arena.view("attn_scratch").unwrap().bytes();
            assert!(
                attn_bytes >= partial_req,
                "ContextSplit attn_scratch {attn_bytes} < hy4_dsa_partial_scratch_bytes({decode_rows}) = {partial_req}"
            );
        }

        let ex = hy4_context_buffers(&cfg, 2752, decode_rows);
        let ex_bytes = crate::placement::ContextExchangeLayout::new(&ex)
            .bytes()
            .expect("ContextExchangeLayout bytes");
        let old = crate::placement::ContextExchangeLayout::new(&hy4_context_buffers(&cfg, decode_rows, decode_rows));
        let wide = crate::placement::ContextExchangeLayout::new(&ex);
        assert_eq!(wide.rows, 2752);
        assert_eq!(ex_bytes - old.bytes().unwrap(), (2752 - 64) * 123_008);
        for live in [1, 64, 128, 2752, 4096, 8192] {
            assert_eq!(context_chunk_rows(live.max(decode_rows), decode_rows, true), decode_rows);
            assert_eq!(context_chunk_rows(live.max(decode_rows), decode_rows, false), live.clamp(decode_rows, 4096));
        }
        let narrow = SharedScratchLayout::new_with_context(&cfg, 0, 2, 64, 64, 64, 16, cfg.vocab_size / 2, true)?;
        let wide_scratch = SharedScratchLayout::new_with_context(&cfg, 0, 2, 2752, 64, 64, 16, cfg.vocab_size / 2, true)?;
        let ctx_bytes = |layout: &SharedScratchLayout| layout.arena.views().iter()
            .filter(|(name, _)| name.starts_with("ctx_")).map(|(_, v)| v.bytes()).sum::<usize>();
        assert_eq!(ctx_bytes(&wide_scratch) - ctx_bytes(&narrow), (2752 - 64) * 45_184);
        assert_eq!(wide_scratch.arena.view("attn_scratch").unwrap().bytes(), narrow.arena.view("attn_scratch").unwrap().bytes());
        Ok(())
    }
}
