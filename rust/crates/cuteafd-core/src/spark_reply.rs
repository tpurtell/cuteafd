//! Spark reply policy. Combined units never compress a decode/verify row or
//! a request contributing at most 64 rows, including inside a wider unit.
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

static REQUESTED: std::sync::OnceLock<Option<SparkReplyFormat>> = std::sync::OnceLock::new();

/// Common CLI selection, fixed before model planning or allocation.
pub fn set_requested(format: Option<SparkReplyFormat>) -> Result<(), &'static str> {
    REQUESTED
        .set(format)
        .map_err(|_| "Spark reply format was already initialized")
}
pub fn requested() -> Option<SparkReplyFormat> {
    REQUESTED.get().copied().flatten()
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

    /// One compressed staging slice per physical prefill-capable link slot.
    /// Dedicated decode and partial-reply links do not use the combined codec.
    pub fn combined_scratch_bytes(self, hidden: usize, unit_rows: usize, slots: usize,
        prefill: bool, partial_replies: bool) -> Option<usize> {
        if !prefill || partial_replies || self == Self::Bf16 {
            return Some(0);
        }
        self.row_bytes(hidden)?.checked_mul(unit_rows)?.checked_mul(slots)
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

    /// `(request id, is prefill)` for every row. An ineligible request makes
    /// the whole combined unit BF16; the returned counts explain that fallback.
    pub fn combined(self, rows: impl IntoIterator<Item = (u64, bool)>) -> ReplyDecision {
        let mut counts = BTreeMap::<u64, (usize, bool)>::new();
        let mut prefill_rows = 0;
        let mut non_prefill = false;
        for (id, prefill) in rows {
            prefill_rows += usize::from(prefill);
            non_prefill |= !prefill;
            let entry = counts.entry(id).or_insert((0, true));
            entry.0 += 1;
            entry.1 &= prefill;
        }
        let eligible = !counts.is_empty() && counts.values().all(|&(n, p)| p && n > 64);
        ReplyDecision {
            format: if eligible { self } else { Self::Bf16 },
            prefill_rows,
            mixed_fallback_rows: if self != Self::Bf16 && non_prefill {
                prefill_rows
            } else {
                0
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplyDecision {
    pub format: SparkReplyFormat,
    pub prefill_rows: usize,
    pub mixed_fallback_rows: usize,
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
    fn combined_policy_is_per_request_and_lossless_for_speculation() {
        for format in [
            SparkReplyFormat::Bf16,
            SparkReplyFormat::Fp8,
            SparkReplyFormat::Nvfp4,
        ] {
            assert_eq!(format.combined((0..65).map(|_| (1, true))).format, format);
            assert_eq!(
                format.combined((0..64).map(|_| (1, true))).format,
                SparkReplyFormat::Bf16
            );
            assert_eq!(
                format.combined((0..128).map(|r| (r / 64, true))).format,
                SparkReplyFormat::Bf16
            );
            let mixed = format.combined((0..130).map(|r| (r / 65, r < 65)));
            assert_eq!(mixed.format, SparkReplyFormat::Bf16);
            assert_eq!(mixed.prefill_rows, 65);
            assert_eq!(
                mixed.mixed_fallback_rows,
                if format == SparkReplyFormat::Bf16 {
                    0
                } else {
                    65
                }
            );
        }
    }
    #[test]
    fn combined_scratch_tracks_physical_slots_not_ranks_or_request_role() {
        for format in [SparkReplyFormat::Bf16, SparkReplyFormat::Fp8, SparkReplyFormat::Nvfp4] {
            for slots in [1, 3, 4] {
                let expected = if format == SparkReplyFormat::Bf16 { 0 }
                    else { format.row_bytes(5120).unwrap() * 2752 * slots };
                assert_eq!(format.combined_scratch_bytes(5120, 2752, slots, true, false), Some(expected));
                assert_eq!(format.combined_scratch_bytes(5120, 80, slots, false, false), Some(0));
                assert_eq!(format.combined_scratch_bytes(5120, 2752, slots, true, true), Some(0));
            }
        }
        assert_eq!(SparkReplyFormat::Fp8.combined_scratch_bytes(5120, usize::MAX, 4, true, false), None);
        assert_eq!(SparkReplyFormat::Nvfp4.combined_scratch_bytes(17, 80, 1, true, false), None);
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
