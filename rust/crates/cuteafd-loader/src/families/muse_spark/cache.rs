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

/// One rank's active sliding state and its mark arena, as byte offsets: the
/// device engine allocates one plane of `active_bytes` per sequence-slot set
/// and one of `mark_bytes` per mark slot, and every copy the prefix cache
/// enqueues comes from [`MuseStateLayout::capture`] / [`MuseStateLayout::restore`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseStateLayout {
    pub ring: cuteafd_core::prefix::AppendRing,
    /// Bytes of one ring row (one token's K|V on this rank).
    pub row_bytes: u64,
    /// Sliding layers in model order (their index among all layers).
    pub sliding: Vec<usize>,
    pub sequences: u64,
}

/// One device copy: `bytes` from `source` to `destination`, offsets into the
/// rank's active-ring plane (`Ring`) or mark arena (`Mark`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateAt { Ring(u64), Mark(u64) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateCopy { pub source: StateAt, pub destination: StateAt, pub bytes: u64 }

impl MuseStateLayout {
    pub fn new(cfg: &MuseSparkConfig, cache: &MuseCacheLayout, sequences: u64) -> Result<Self, CacheGeometryError> {
        if sequences == 0 { return Err(invalid("no sequence slots")); }
        let sliding = cfg.attention.iter().enumerate()
            .filter_map(|(layer, kind)| (*kind == MuseAttention::Sliding).then_some(layer)).collect();
        Ok(Self { ring: cache.ring, row_bytes: cache.record_bytes, sliding, sequences })
    }

    fn ring_bytes(&self) -> u64 { self.ring.rows * self.row_bytes }

    /// Bytes of every sequence's rings on this rank: `[sliding layer][sequence][ring row]`.
    pub fn active_bytes(&self) -> u64 { self.sliding.len() as u64 * self.sequences * self.ring_bytes() }

    /// Bytes of one mark: each sliding layer's visible window, in position order.
    pub fn mark_bytes(&self) -> u64 { self.sliding.len() as u64 * self.ring.window * self.row_bytes }

    /// Offset of `(sliding index, sequence)`'s ring in the active plane.
    pub fn ring_offset(&self, sliding: usize, sequence: u64) -> u64 {
        (sliding as u64 * self.sequences + sequence) * self.ring_bytes()
    }

    fn check(&self, sequence: u64) -> Result<(), CacheGeometryError> {
        if sequence >= self.sequences { return Err(invalid("sequence slot outside the admitted rings")); }
        Ok(())
    }

    /// Copies that retain `sequence`'s visible window at `frontier` into mark `slot`.
    /// Position `p`'s row lands at mark row `p - (frontier - min(frontier, window))`.
    pub fn capture(&self, sequence: u64, frontier: u64, slot: u64) -> Result<Vec<StateCopy>, CacheGeometryError> {
        self.copies(sequence, frontier, slot, false)
    }

    /// Copies that put mark `slot` (taken at `frontier`) back into `sequence`'s
    /// rings at the same ring slots a prefill to `frontier` writes.
    pub fn restore(&self, sequence: u64, frontier: u64, slot: u64) -> Result<Vec<StateCopy>, CacheGeometryError> {
        self.copies(sequence, frontier, slot, true)
    }

    fn copies(&self, sequence: u64, frontier: u64, slot: u64, restore: bool) -> Result<Vec<StateCopy>, CacheGeometryError> {
        self.check(sequence)?;
        let overflow = || CacheGeometryError::Overflow("Muse Spark mark copy");
        let base = slot.checked_mul(self.mark_bytes()).ok_or_else(overflow)?;
        let mut copies = Vec::new();
        for index in 0..self.sliding.len() {
            let ring = self.ring_offset(index, sequence);
            let mark = base + index as u64 * self.ring.window * self.row_bytes;
            for run in self.ring.visible(frontier) {
                let at_ring = StateAt::Ring(ring + run.slot * self.row_bytes);
                let at_mark = StateAt::Mark(mark + run.offset * self.row_bytes);
                let bytes = run.rows * self.row_bytes;
                copies.push(if restore { StateCopy { source: at_mark, destination: at_ring, bytes } }
                    else { StateCopy { source: at_ring, destination: at_mark, bytes } });
            }
        }
        Ok(copies)
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

    /// A host model of the rings: a prefill writes position `p`'s row at ring
    /// slot `p % rows`. Capturing at any frontier and restoring into a fresh
    /// (garbage-filled) ring of another sequence reproduces every row the
    /// attention reads (the visible window) byte for byte.
    #[test]
    fn mark_restore_reproduces_the_visible_window_of_a_fresh_prefill() {
        let mut config = crate::plan::testing::muse_spark_config();
        let text = &mut config["text_config"];
        text["sliding_window"] = serde_json::json!(5);
        text["num_hidden_layers"] = serde_json::json!(4);
        text["layer_types"] = serde_json::json!(["sliding_attention", "full_attention", "sliding_attention",
            "sliding_attention"]);
        text["layer_rope_theta"] = serde_json::json!([500000.0, 0, 500000.0, 500000.0]);
        text["mlp_layer_types"] = serde_json::json!(vec!["sparse"; 4]);
        let cfg = MuseSparkConfig::from_hf(&config).unwrap();
        let cache = MuseCacheLayout::new(&cfg, 1, 3).unwrap();
        let state = MuseStateLayout::new(&cfg, &cache, 3).unwrap();
        assert_eq!(state.sliding, [0, 2, 3]);
        let row = state.row_bytes as usize;
        let token = |layer: usize, p: u64| -> Vec<u8> {
            (0..row).map(|i| (layer as u64 * 131 + p * 7 + i as u64 % 251) as u8).collect()
        };
        let prefill = |active: &mut [u8], sequence: u64, len: u64| {
            for (index, &layer) in state.sliding.iter().enumerate() {
                for p in 0..len {
                    let at = (state.ring_offset(index, sequence) + (p % state.ring.rows) * state.row_bytes) as usize;
                    active[at..at + row].copy_from_slice(&token(layer, p));
                }
            }
        };
        let apply = |copies: &[StateCopy], active: &mut [u8], marks: &mut [u8]| {
            for copy in copies {
                let n = copy.bytes as usize;
                let bytes = match copy.source {
                    StateAt::Ring(at) => active[at as usize..at as usize + n].to_vec(),
                    StateAt::Mark(at) => marks[at as usize..at as usize + n].to_vec(),
                };
                match copy.destination {
                    StateAt::Ring(at) => active[at as usize..at as usize + n].copy_from_slice(&bytes),
                    StateAt::Mark(at) => marks[at as usize..at as usize + n].copy_from_slice(&bytes),
                }
            }
        };
        for frontier in [1u64, 4, 5, 6, 63, 64, 65, 200, 1000] {
            let mut active = vec![0xA5u8; state.active_bytes() as usize];
            let mut marks = vec![0x5Au8; 2 * state.mark_bytes() as usize];
            prefill(&mut active, 0, frontier);
            apply(&state.capture(0, frontier, 1).unwrap(), &mut active, &mut marks);
            apply(&state.restore(2, frontier, 1).unwrap(), &mut active, &mut marks);
            let mut fresh = vec![0x33u8; state.active_bytes() as usize];
            prefill(&mut fresh, 2, frontier);
            for (index, _) in state.sliding.iter().enumerate() {
                for run in state.ring.visible(frontier) {
                    let at = (state.ring_offset(index, 2) + run.slot * state.row_bytes) as usize;
                    let n = (run.rows * state.row_bytes) as usize;
                    assert_eq!(active[at..at + n], fresh[at..at + n], "frontier {frontier}");
                }
            }
            // Sequence 1 is never touched by another sequence's restore.
            for index in 0..state.sliding.len() {
                assert!(active[state.ring_offset(index, 1) as usize..state.ring_offset(index, 2) as usize]
                    .iter().all(|b| *b == 0xA5));
            }
        }
        assert!(state.capture(3, 10, 0).is_err());
    }
}
