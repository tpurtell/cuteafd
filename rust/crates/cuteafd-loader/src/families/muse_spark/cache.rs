//! Muse Spark cache storage: full (NoPE) layers page BF16 K|V records in the
//! shared pool; sliding layers keep a per-sequence append-only ring of the
//! window plus one step (exact restores: a mark retains each ring's visible
//! window). Head-split TP2 halves both per GPU. Storage only: workspaces,
//! graphs and weights are separate reservations.
use super::{MuseAttention, MuseSparkConfig};
use crate::serving_capacity::{CacheGeometryError, FamilyCacheGeometry, KvPlacement, RankCacheGeometry};

pub const PAGE_ROWS: u64 = 64;

fn invalid(what: &'static str) -> CacheGeometryError {
    CacheGeometryError::Unsupported { family: "muse_spark", what }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseCacheLayout {
    pub ranks: usize,
    /// BF16 K|V bytes of one token in one full layer on one rank.
    pub record_bytes: u64,
    /// Ring rows per sliding layer per sequence.
    pub ring: cuteafd_core::prefix::AppendRing,
    pub geometry: FamilyCacheGeometry,
}

impl MuseCacheLayout {
    /// `max_step`: the largest step (prefill unit or verify block) appended at once.
    pub fn new(cfg: &MuseSparkConfig, ranks: usize, max_step: u64) -> Result<Self, CacheGeometryError> {
        if ![1, 2].contains(&ranks) || cfg.kv_heads % ranks != 0 {
            return Err(invalid("one GPU or a head-split pair over whole KV heads"));
        }
        let record_bytes = (cfg.kv_record_bytes() / ranks) as u64;
        let ring = cuteafd_core::prefix::AppendRing::new(cfg.window as u64, max_step.max(1), PAGE_ROWS)
            .ok_or_else(|| invalid("sliding ring step"))?;
        let full = cfg.attention.iter().filter(|kind| **kind == MuseAttention::Full).count() as u64;
        let sliding = cfg.layers as u64 - full;
        let overflow = || CacheGeometryError::Overflow("Muse Spark cache storage");
        let unit = full.checked_mul(record_bytes).and_then(|b| b.checked_mul(PAGE_ROWS)).ok_or_else(overflow)?;
        let rings = sliding.checked_mul(ring.rows).and_then(|b| b.checked_mul(record_bytes)).ok_or_else(overflow)?;
        // A mark keeps each sliding layer's visible window (full layers mark through pool pages).
        let marks = sliding.checked_mul(cfg.window as u64).and_then(|b| b.checked_mul(record_bytes)).ok_or_else(overflow)?;
        let rank = RankCacheGeometry { persistent_unit_bytes: unit, active_state_per_sequence_bytes: rings + 16,
            retained_mark_bytes: marks + 16, ..Default::default() };
        Ok(Self { ranks, record_bytes, ring, geometry: FamilyCacheGeometry {
            logical_unit_rows: PAGE_ROWS,
            placement: if ranks == 1 { KvPlacement::SingleDevice } else { KvPlacement::PartitionedHeads },
            ranks: vec![rank; ranks],
        } })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_records_rings_and_marks() {
        let cfg = MuseSparkConfig::from_hf(&crate::plan::testing::muse_spark_config()).unwrap();
        let whole = MuseCacheLayout::new(&cfg, 1, 2752).unwrap();
        // 16 KV heads x 64 x (K+V) x BF16 per full layer; 16 full layers per 64-row page.
        assert_eq!(whole.record_bytes, 4096);
        assert_eq!(whole.geometry.ranks[0].persistent_unit_bytes, 16 * 4096 * 64);
        // Ring: window 2048 + one 2752-row unit, page-aligned.
        assert_eq!(whole.ring.rows, (2048u64 + 2752).next_multiple_of(64));
        assert_eq!(whole.geometry.ranks[0].retained_mark_bytes, 46 * 2048 * 4096 + 16);
        let split = MuseCacheLayout::new(&cfg, 2, 2752).unwrap();
        assert_eq!(split.record_bytes * 2, whole.record_bytes);
        assert_eq!(split.geometry.placement, KvPlacement::PartitionedHeads);
        assert!(MuseCacheLayout::new(&cfg, 3, 64).is_err());
    }
}
