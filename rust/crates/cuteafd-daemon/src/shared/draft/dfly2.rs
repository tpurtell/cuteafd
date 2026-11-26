//! Shared DFly2 speculative drafting helpers reused by Hy4-flash and DeepSeek V4.1.
//!
//! DFly2 drafters read the target's own committed KV cache and indexer recall
//! (no private draft KV cache). Across families, they share:
//! - validated block geometry (`block_size`, `drafts = block_size - 1`, overdrive bounds);
//! - per-request slot/frontier tracking (`Dfly2Frontiers`);
//! - little-endian word packing for host-to-device metadata staging (`words`).

use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// Validated DFly2 block shape for a run (e.g. trained block 6 or overdriven block 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dfly2BlockShape {
    /// Total tokens in one block: 1 anchor + `drafts()` mask slots.
    pub block_size: usize,
    /// Checkpoint's maximum supported block slots (`max_block_slots`, e.g. 16).
    pub max_block_slots: usize,
    /// Mask token ID filling slots `1..block_size`.
    pub mask_token_id: u32,
}

impl Dfly2BlockShape {
    pub fn new(block_size: usize, max_block_slots: usize, mask_token_id: u32) -> Result<Self> {
        ensure!(
            block_size >= 2 && block_size <= max_block_slots,
            "DFly2 block_size {block_size} must be in 2..={max_block_slots}"
        );
        Ok(Self { block_size, max_block_slots, mask_token_id })
    }

    /// Number of drafted tokens produced per block (`block_size - 1`).
    pub fn drafts(self) -> usize {
        self.block_size - 1
    }
}

/// Per-request slot and target-frontier tracker for a cache-less DFly2 drafter.
///
/// Unlike self-caching drafters (dSpark), DFly2 retains only the target's
/// last-accepted final hidden state and indexer recall per active request slot.
/// Restored/evicted entries without taps report `committed_end == None` until
/// the next target commit refreshes them.
#[derive(Debug)]
pub(crate) struct Dfly2Frontiers {
    limit: usize,
    entries: BTreeMap<u64, FrontierEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrontierEntry {
    slot: usize,
    committed_end: Option<u64>,
}

impl Dfly2Frontiers {
    pub fn new(limit: usize) -> Self {
        Self { limit, entries: BTreeMap::new() }
    }

    pub fn admit(&mut self, id: u64) -> Result<usize> {
        ensure!(!self.entries.contains_key(&id), "DFly2 request {id} already admitted");
        let slot = (0..self.limit)
            .find(|slot| self.entries.values().all(|entry| entry.slot != *slot))
            .context("DFly2 request slot capacity exhausted")?;
        self.entries.insert(id, FrontierEntry { slot, committed_end: None });
        Ok(slot)
    }

    pub fn release(&mut self, id: u64) -> Option<usize> {
        self.entries.remove(&id).map(|entry| entry.slot)
    }

    pub fn slot(&self, id: u64) -> Result<usize> {
        self.entries.get(&id).map(|e| e.slot).context("DFly2 request not admitted")
    }

    pub fn contains(&self, id: u64) -> bool {
        self.entries.contains_key(&id)
    }

    pub fn set_frontier(&mut self, id: u64, committed_end: u64) -> Result<usize> {
        let entry = self.entries.get_mut(&id).context("DFly2 request not admitted")?;
        entry.committed_end = Some(committed_end);
        Ok(entry.slot)
    }

    pub fn invalidate(&mut self, id: u64) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.committed_end = None;
        }
    }

    pub fn committed_end(&self, id: u64) -> Option<u64> {
        self.entries.get(&id).and_then(|e| e.committed_end)
    }

    /// True when request `id` has fresh DFly2 taps matching `expected_end`.
    pub fn ready_at(&self, id: u64, expected_end: u64) -> bool {
        self.committed_end(id) == Some(expected_end)
    }
}

/// Pack a slice of little-endian integers for H2D metadata upload.
pub(crate) fn words<T, const N: usize>(input: &[T]) -> Vec<u8>
where
    T: LeBytes<N>,
{
    input.iter().flat_map(LeBytes::le_bytes).collect()
}

pub(crate) trait LeBytes<const N: usize> {
    fn le_bytes(&self) -> [u8; N];
}
impl LeBytes<4> for i32 {
    fn le_bytes(&self) -> [u8; 4] {
        self.to_le_bytes()
    }
}
impl LeBytes<4> for u32 {
    fn le_bytes(&self) -> [u8; 4] {
        self.to_le_bytes()
    }
}
impl LeBytes<8> for i64 {
    fn le_bytes(&self) -> [u8; 8] {
        self.to_le_bytes()
    }
}
impl LeBytes<8> for u64 {
    fn le_bytes(&self) -> [u8; 8] {
        self.to_le_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontiers_track_slots_commits_and_invalidation() -> Result<()> {
        let mut f = Dfly2Frontiers::new(2);
        assert_eq!(f.admit(10)?, 0);
        assert_eq!(f.admit(20)?, 1);
        assert!(f.admit(30).is_err());
        assert!(!f.ready_at(10, 5));
        assert_eq!(f.set_frontier(10, 5)?, 0);
        assert!(f.ready_at(10, 5));
        assert!(!f.ready_at(10, 6));
        f.invalidate(10);
        assert!(!f.ready_at(10, 5));
        assert_eq!(f.release(10), Some(0));
        assert_eq!(f.admit(30)?, 0);
        Ok(())
    }

    #[test]
    fn block_shape_validates_trained_and_overdriven_widths() -> Result<()> {
        let b6 = Dfly2BlockShape::new(6, 16, 128799)?;
        assert_eq!(b6.drafts(), 5);
        let b8 = Dfly2BlockShape::new(8, 16, 128799)?;
        assert_eq!(b8.drafts(), 7);
        assert!(Dfly2BlockShape::new(1, 16, 128799).is_err());
        assert!(Dfly2BlockShape::new(17, 16, 128799).is_err());
        Ok(())
    }
}
