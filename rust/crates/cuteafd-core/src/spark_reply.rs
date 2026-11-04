//! Independent Spark prefill and decode/verify reply selections.
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt, str::FromStr};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SparkReplyFormat {
    #[default]
    Bf16,
    Fp8,
    Nvfp4,
}

/// Explicit selections remain optional so omission retains each family's defaults.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkReplyFormats {
    pub prefill: Option<SparkReplyFormat>,
    pub decode: Option<SparkReplyFormat>,
}

static REQUESTED: std::sync::OnceLock<SparkReplyFormats> = std::sync::OnceLock::new();

/// Common CLI selections, fixed before model planning or allocation.
pub fn set_requested(formats: SparkReplyFormats) -> Result<(), &'static str> {
    REQUESTED.set(formats).map_err(|_| "Spark reply formats were already initialized")
}
pub fn requested() -> SparkReplyFormats {
    REQUESTED.get().copied().unwrap_or_default()
}

impl SparkReplyFormats {
    pub fn prefill(self, partial: bool) -> SparkReplyFormat {
        SparkReplyFormat::resolve(self.prefill, partial)
    }

    pub fn decode(self) -> SparkReplyFormat {
        self.decode.unwrap_or_default()
    }

    pub fn can_compress(self) -> bool {
        self.prefill(false) != SparkReplyFormat::Bf16 || self.decode() != SparkReplyFormat::Bf16
    }

    /// `(request id, is prefill)` per row. A mixed unit uses the decode
    /// selection without changing scheduler packing. Only omitted prefill
    /// selections retain the per-request small-row BF16 fallback.
    pub fn combined(self, rows: impl IntoIterator<Item = (u64, bool)>) -> ReplyDecision {
        let mut counts = BTreeMap::<u64, usize>::new();
        let mut prefill_rows = 0;
        let mut non_prefill = false;
        for (id, prefill) in rows {
            prefill_rows += usize::from(prefill);
            non_prefill |= !prefill;
            *counts.entry(id).or_default() += 1;
        }
        let format = if counts.is_empty() {
            SparkReplyFormat::Bf16
        } else if non_prefill {
            self.decode()
        } else if self.prefill.is_none() && counts.values().any(|&n| n <= 64) {
            SparkReplyFormat::Bf16
        } else {
            self.prefill(false)
        };
        ReplyDecision { format, prefill_rows, mixed: non_prefill && prefill_rows > 0 }
    }
}

impl SparkReplyFormat {
    pub fn row_bytes(self, hidden: usize) -> Option<usize> {
        if hidden == 0 || hidden % 32 != 0 {
            return None;
        }
        match self {
            Self::Bf16 => hidden.checked_mul(2),
            Self::Fp8 => hidden.checked_add(hidden / 32),
            Self::Nvfp4 => (hidden / 2).checked_add(hidden / 16),
        }
    }

    /// The one source of omission defaults: Muse preserves its split-wire
    /// NVFP4 prefill; all combined-reply families preserve BF16.
    pub fn resolve(requested: Option<Self>, partial_replies: bool) -> Self {
        requested.unwrap_or(if partial_replies {
            Self::Nvfp4
        } else {
            Self::Bf16
        })
    }

}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplyDecision {
    pub format: SparkReplyFormat,
    pub prefill_rows: usize,
    pub mixed: bool,
}

impl fmt::Display for SparkReplyFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
        })
    }
}
impl FromStr for SparkReplyFormat {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            "nvfp4" => Ok(Self::Nvfp4),
            _ => Err("Spark reply format must be bf16, fp8 or nvfp4"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_nine_pairs_follow_the_unit_phase_at_every_width() {
        use SparkReplyFormat::*;
        for prefill in [Bf16, Fp8, Nvfp4] {
            for decode in [Bf16, Fp8, Nvfp4] {
                let policy = SparkReplyFormats { prefill: Some(prefill), decode: Some(decode) };
                for n in [1, 8, 64, 65, 2752] {
                    assert_eq!(policy.combined((0..n).map(|_| (1, true))).format, prefill);
                    assert_eq!(policy.combined((0..n).map(|_| (1, false))).format, decode);
                }
                let mixed = policy.combined([(1, true), (2, false)]);
                assert_eq!(mixed.format, decode);
                assert!(mixed.mixed);
                assert_eq!(mixed.prefill_rows, 1);
                assert_eq!(policy.combined([(1, true), (2, true)]).format, prefill);
                assert!(!policy.combined([(1, false)]).mixed);
                assert_eq!(policy.combined([]).format, Bf16);
            }
        }
    }

    #[test]
    fn omitted_phase_defaults_do_not_leak_between_selections() {
        use SparkReplyFormat::*;
        let omitted = SparkReplyFormats::default();
        assert_eq!((omitted.prefill(false), omitted.prefill(true), omitted.decode()), (Bf16, Nvfp4, Bf16));
        for n in [1, 64, 65] {
            assert_eq!(omitted.combined((0..n).map(|_| (1, true))).format, Bf16);
            assert_eq!(SparkReplyFormats { decode: Some(Fp8), ..omitted }
                .combined((0..n).map(|_| (1, true))).format, Bf16);
            assert_eq!(SparkReplyFormats { prefill: Some(Fp8), ..omitted }
                .combined((0..n).map(|_| (1, false))).format, Bf16);
        }
    }
    #[test]
    fn layouts_and_omission_defaults() {
        assert_eq!(SparkReplyFormat::Bf16.row_bytes(5120), Some(10240));
        assert_eq!(SparkReplyFormat::Fp8.row_bytes(5120), Some(5280));
        assert_eq!(SparkReplyFormat::Nvfp4.row_bytes(5120), Some(2880));
        assert_eq!(
            SparkReplyFormat::resolve(None, false),
            SparkReplyFormat::Bf16
        );
        assert_eq!(
            SparkReplyFormat::resolve(None, true),
            SparkReplyFormat::Nvfp4
        );
        assert_eq!(
            SparkReplyFormat::resolve(Some(SparkReplyFormat::Bf16), true),
            SparkReplyFormat::Bf16
        );
    }
}
