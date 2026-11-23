//! Named physical cache, prefix-mark, and per-unit workspace layouts for Hy4-flash.
//!
//! Shared by `cuteafd plan --layout` and runtime admission so planned and
//! allocated bytes match owner for owner.
use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};

use super::{Hy4FlashConfig, Hy4KvCache};

/// KV page quantum for the 12 DSA layers (64-token pages).
pub const PAGE_ROWS: usize = 64;
/// Per-sequence SWA-512 circular ring rows: 10 pages = 640 rows
/// (`512` window + up to `64` decode/verify rows + page wrap alignment).
pub const SWA_RING_PAGES: usize = 10;
pub const SWA_RING_ROWS: usize = SWA_RING_PAGES * PAGE_ROWS;
/// Physical partial O (BF16) and LSE (F32) planes for the capped MLA decode exports.
/// The export strides scratch by the compiled split capacity, not active splits.
pub fn mla_split_scratch_bytes(decode_rows: usize, head_parts: usize) -> usize {
    decode_rows * 16 * (64 / head_parts) * (512 * 2 + 4)
}

/// Compressed indexer group-cache record bytes: 128 E4M3 dims + 4B F32 scale.
pub const INDEX_GROUP_RECORD_BYTES: usize = 132;

/// Persistent per-sequence SWA-512 ring & prefix-mark bytes for one GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLayout {
    /// Active SWA-512 rings across all admitted sequences and owned SWA layers.
    pub active_swa_ring_bytes: usize,
    /// Pending incomplete compressed-indexer token keys (`compress_ratio - 1 = 3`
    /// tokens × `128` F32) across admitted sequences and owned DSA layers.
    pub active_indexer_pending_bytes: usize,
    /// Exact per-snapshot prefix-mark bytes (30 SWA-512 windows + 12 DSA pending
    /// indexer tokens ≈ 16.9 MiB BF16 on one GPU).
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
        ensure!(sequences > 0, "Hy4-flash requires at least one admitted sequence");
        let rec = kv_format.record_bytes();
        let dsa_kv_page_bytes = PAGE_ROWS.checked_mul(rec).context("DSA KV page overflow")?;
        let groups_per_page = PAGE_ROWS / cfg.indexer_compress_ratio;
        let dsa_index_page_bytes = groups_per_page
            .checked_mul(INDEX_GROUP_RECORD_BYTES)
            .context("DSA index page overflow")?;

        // Active SWA-512 circular rings: 10 pages (640 rows) per sequence per owned SWA layer.
        let swa_ring_per_seq = owned_swa_layers
            .checked_mul(SWA_RING_ROWS)
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
            active_swa_ring_bytes,
            active_indexer_pending_bytes,
            mark_bytes_per_slot,
            dsa_kv_page_bytes,
            dsa_index_page_bytes,
        })
    }
}

/// Named scratch buffers owned by one in-flight execution unit (one decode lane
/// or one prefill wave slot) on a coordinator GPU.
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
        ensure!(rows > 0 && matches!(head_parts, 1 | 2), "invalid Hy4-flash UnitLayout shape");
        let h = cfg.hidden;
        let hc = cfg.hc_mult;
        let max_heads = cfg.swa_heads.max(cfg.sparse_heads) / head_parts;
        let mut buffers = BTreeMap::new();

        // 4-stream iHC residual [rows, 4, H] BF16 + pre/post FP32 mixing coefficients.
        buffers.insert("ihc_residual".into(), rows * hc * h * 2);
        buffers.insert("ihc_post".into(), rows * hc * 4);
        buffers.insert("layer_in".into(), rows * h * 2);
        buffers.insert("layer_out".into(), rows * h * 2);

        // MLA producer & absorbed attention scratch:
        // q_a [rows, 1024], kv_a [rows, 576], absorbed query [rows, max_heads, 576],
        // attn_latent [rows, max_heads, 512], lse [rows, max_heads] F32,
        // gate/uv [rows, max_heads, 256] BF16, SWA step KV scratch [(rows + 576), record_bytes].
        buffers.insert("q_a".into(), rows * cfg.q_lora_rank * 2);
        buffers.insert("kv_a".into(), rows * (cfg.kv_lora_rank + cfg.qk_rope_dim) * 2);
        buffers.insert("absorbed_q".into(), rows * max_heads * (cfg.kv_lora_rank + cfg.qk_rope_dim) * 2);
        buffers.insert("attn_latent".into(), rows * max_heads * cfg.kv_lora_rank * 2);
        buffers.insert("attn_lse".into(), rows * max_heads * 4);
        buffers.insert("attn_gate".into(), rows * max_heads * cfg.v_head_dim * 2);
        buffers.insert("attn_uv".into(), rows * max_heads * cfg.v_head_dim * 2);
        let swa_step_pages = (rows + cfg.sliding_window + PAGE_ROWS).div_ceil(PAGE_ROWS);
        buffers.insert("swa_step_kv".into(), swa_step_pages * PAGE_ROWS * kv_format.record_bytes());

        // Compressed indexer scratch (DSA layers): queries, token keys, candidate slots [rows, 2048] I32.
        buffers.insert("indexer_q".into(), rows * cfg.indexer_heads * cfg.indexer_head_dim * 2);
        buffers.insert("indexer_k".into(), rows * cfg.indexer_head_dim * 4);
        buffers.insert("indexer_weights".into(), rows * cfg.indexer_heads * 4);
        buffers.insert("indexer_candidates".into(), rows * cfg.indexer_candidate_topk * 4);

        // Dense layer-0 / shared expert / LatentMoE scratch.
        let dense_cols = cfg.intermediate / head_parts;
        buffers.insert("ffn_gate_up".into(), rows * dense_cols * 2 * 2);
        buffers.insert("latent_in".into(), rows * cfg.routed_expert_hidden * 2);
        buffers.insert("latent_out".into(), rows * cfg.routed_expert_hidden * 2);
        buffers.insert("router_scores".into(), rows * cfg.experts * 4);
        buffers.insert("router_ids".into(), rows * cfg.topk * 4);
        buffers.insert("router_weights".into(), rows * cfg.topk * 4);

        // Engram GPU staging & gate scratch: 16 heads × 96 BF16 + 4-stream key/value projections.
        buffers.insert("engram_embed".into(), rows * cfg.engram_concat_width() * 2);
        buffers.insert("engram_kv".into(), rows * (hc + 1) * h * 2);

        Ok(Self { buffers })
    }

    pub fn total_bytes(&self) -> usize {
        self.buffers.values().sum()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn mla_split_partials_scale_with_resolved_rows_and_head_parts() {
        let full = super::mla_split_scratch_bytes(64, 1);
        assert_eq!(full, 64 * 16 * 64 * 1028);
        assert_eq!(super::mla_split_scratch_bytes(32, 1), full / 2);
        assert_eq!(super::mla_split_scratch_bytes(32, 2), full / 4);
    }
}
