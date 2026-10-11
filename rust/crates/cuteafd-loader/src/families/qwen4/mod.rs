//! Qwen 3.8 Flash Next (qwen4_exp) family: configuration and the PLE n-gram
//! hashing for the generic engine.
pub mod config;
pub mod ngram;
pub mod resident;
pub mod rope;
pub use rope::{ImageSpan, RopeError, RopePositions};
pub use config::{Qwen4Attention, Qwen4Config};
pub use ngram::{NgramHasher, NgramHistory};

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

impl std::fmt::Display for Qwen4KvCache {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(match self { Self::Bf16 => "bf16", Self::Fp8 => "fp8" })
    }
}
