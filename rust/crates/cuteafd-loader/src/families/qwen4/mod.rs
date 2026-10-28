//! Qwen 3.8 Flash Next (qwen4_exp) family: configuration and the PLE n-gram
//! hashing for the generic engine.
pub mod config;
pub mod ngram;
pub mod resident;
pub mod rope;
pub use rope::{ImageSpan, RopeError, RopePositions};
pub use config::{Qwen4Attention, Qwen4Config};
pub use ngram::{NgramHasher, NgramHistory};

/// Whole-step local owners are serial; Spark table slots follow the shared depth.
pub fn prefill_limits(local: bool) -> crate::serving_capacity::prefill_waves::PrefillLimits {
    crate::serving_capacity::prefill_waves::PrefillLimits {
        max_waves: if local { 1 } else { 4 },
        ..crate::plan::Family::prefill_limits(&crate::plan::families::qwen::QWEN4)
    }
}

/// Destination extents shared by workspace allocation and ordered H2D admission.
#[derive(Debug)]
pub struct StepTableBytes {
    pub positions: usize, pub rope_positions: usize, pub block_rope_positions: usize,
    pub kv_slots: usize, pub slots: usize, pub seq_first: usize, pub pool_slots: usize,
    pub cache_lengths: usize, pub page_table: usize, pub pool_table: usize,
    pub ids: usize, pub ple_ids: usize,
}

impl StepTableBytes {
    pub fn new(rows: usize, decode: bool, pages: usize, pool_pages: usize, ple_rows: usize) -> anyhow::Result<Self> {
        use anyhow::Context;
        let mul = |a: usize, b: usize| a.checked_mul(b).context("Qwen host table size overflow");
        let table_rows = if decode { rows } else { 1 };
        Ok(Self { positions: mul(rows, 8)?, rope_positions: mul(rows, 12)?,
            block_rope_positions: mul(rows, 12)?, kv_slots: mul(rows, 8)?,
            slots: mul(rows, 4)?, seq_first: mul(rows, 4)?, pool_slots: mul(rows, 8)?,
            cache_lengths: mul(rows, 4)?, page_table: mul(mul(table_rows, pages)?, 4)?,
            pool_table: mul(mul(table_rows, pool_pages)?, 4)?, ids: mul(rows, 4)?,
            ple_ids: mul(mul(rows, ple_rows)?, 8)? })
    }

    pub fn ordered_bytes(&self, ple_ids: bool) -> anyhow::Result<u64> {
        use anyhow::Context;
        [self.positions, self.rope_positions, self.block_rope_positions, self.kv_slots,
            self.slots, self.seq_first, self.pool_slots, self.cache_lengths,
            self.page_table, self.pool_table, self.ids].into_iter()
            .chain(ple_ids.then_some(self.ple_ids)).try_fold(0u64, |total, bytes|
                total.checked_add(bytes.max(256) as u64).context("Qwen ordered table size overflow"))
    }
}

/// Immutable full-attention records. Index keys remain BF16 in either format.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen4KvCache {
    Bf16,
    /// E4M3 K then V, followed by per-token/per-head FP32 K and V descales.
    #[default]
    Fp8,
}

impl Qwen4KvCache {
    /// Separate K or V plane with inline head descales; align vector loads.
    pub const fn plane_record_bytes(self, kv_heads: usize, head_dim: usize) -> usize {
        match self {
            Self::Bf16 => kv_heads * head_dim * 2,
            Self::Fp8 => (kv_heads * (head_dim + 4) + 15) / 16 * 16,
        }
    }

    pub const fn record_bytes(self, kv_heads: usize, head_dim: usize) -> usize {
        match self {
            Self::Bf16 => 2 * kv_heads * head_dim * 2,
            Self::Fp8 => 2 * kv_heads * (head_dim + 4),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unsupported Qwen KV format {0:?}: expected bf16 or fp8")]
pub struct QwenKvFormatError(String);

impl std::str::FromStr for Qwen4KvCache {
    type Err = QwenKvFormatError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => Err(QwenKvFormatError(value.into())),
        }
    }
}

#[cfg(test)]
mod kv_record_tests {
    use super::Qwen4KvCache;

    #[test]
    fn planar_records_keep_inline_scales_and_vector_alignment() {
        for (heads, dim, fp8_bytes) in [(2, 192, 400), (4, 192, 784), (2, 256, 528)] {
            assert_eq!(Qwen4KvCache::Bf16.plane_record_bytes(heads, dim), heads * dim * 2);
            assert_eq!(Qwen4KvCache::Fp8.plane_record_bytes(heads, dim), fp8_bytes);
            assert_eq!(fp8_bytes % 16, 0);
            assert!(fp8_bytes >= heads * (dim + 4));
            assert!(fp8_bytes < heads * (dim + 4) + 16);
        }
    }
}

impl std::fmt::Display for Qwen4KvCache {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(match self { Self::Bf16 => "bf16", Self::Fp8 => "fp8" })
    }
}
