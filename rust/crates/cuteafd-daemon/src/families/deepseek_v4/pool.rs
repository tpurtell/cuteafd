//! Paged DeepSeek V4 caches shared by every sequence.
//!
//! Each layer owns one window pool (FP8 584-byte records, 256 per page), one
//! compressed pool per ratio (C4: 64 rows per page with a same-numbered index
//! page; C128: 2 rows per page) and compressor state arrays indexed by a
//! sequence's state slot. Both compressed pools hold one page per 256 source
//! tokens, so one page index (a unit) names a C4 page, its index page and a
//! C128 page: a sequence owns a list of units (refcounted by the prefix cache,
//! `cuteafd_engine::prefix::RefPagePool`), and the host maps logical positions
//! and groups to physical slots for every step.
//!
//! Window attention reads only the last 128 positions, so a sequence's window
//! pages form a ring (in its state slot) large enough for one prefill chunk
//! plus the window.
use super::metadata::{compressed_page_rows, SOURCE_PAGE_TOKENS, WINDOW};
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{RefPagePool, TailCopy};

/// Page and slot geometry of the pool (identical for every layer).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoolShape {
    /// Sequences that can hold state at once.
    pub sequences: usize,
    /// Window pages per sequence (ring).
    pub ring_pages: usize,
    /// Compressed-cache units (one C4 page, its index page and one C128 page each).
    pub units: usize,
}

impl PoolShape {
    pub fn new(sequences: usize, prefill_rows: usize, units: usize) -> Self {
        Self { sequences, ring_pages: (prefill_rows + WINDOW).div_ceil(SOURCE_PAGE_TOKENS), units }
    }

    /// Units for `pool_tokens` tokens across sequences plus one partial unit per sequence.
    pub fn units_for(pool_tokens: usize, sequences: usize) -> usize {
        pool_tokens.div_ceil(SOURCE_PAGE_TOKENS) + sequences
    }

    pub fn window_pages(&self) -> usize {
        self.sequences * self.ring_pages
    }
}

/// One sequence's placement in the pool.
#[derive(Debug, Clone)]
pub(crate) struct Placement {
    pub state: usize,
    /// Compressed-cache units in token order (unit `i` holds tokens `[256 i, 256 (i + 1))`).
    pub units: Vec<u32>,
    /// The same units as the programs' page ids.
    pub pages: Vec<i32>,
    /// Tokens whose KV is in the caches.
    pub len: usize,
}

impl Placement {
    pub fn new(state: usize, units: Vec<u32>) -> Self {
        let pages = units.iter().map(|&u| u as i32).collect();
        Self { state, units, pages, len: 0 }
    }

    /// Physical window slot of `position` (ring of this sequence's pages).
    pub fn window_slot(&self, shape: &PoolShape, position: usize) -> i64 {
        let page = self.state * shape.ring_pages + (position / SOURCE_PAGE_TOKENS) % shape.ring_pages;
        (page * SOURCE_PAGE_TOKENS + position % SOURCE_PAGE_TOKENS) as i64
    }

    /// Physical compressed slot of group `group` for `ratio` (C4 also names the index slot).
    pub fn group_slot(&self, ratio: usize, group: usize) -> Result<i32> {
        let rows = compressed_page_rows(ratio);
        let page = *self.pages.get(group / rows).with_context(|| format!("group {group} has no C{ratio} page"))?;
        Ok(page * rows as i32 + (group % rows) as i32)
    }
}

/// Host bookkeeping outside the prefix cache (goldens, tests): free state
/// slots and a unit pool.
pub(crate) struct PoolAllocator {
    free_states: Vec<usize>,
    units: RefPagePool,
}

impl PoolAllocator {
    pub fn new(shape: PoolShape) -> Self {
        Self { free_states: (0..shape.sequences).rev().collect(), units: RefPagePool::new(shape.units, SOURCE_PAGE_TOKENS) }
    }

    /// Context C4 units use stable logical parity; C128 pages retain the same
    /// IDs in their replicated pools. Scratch lives outside the allocatable units.
    pub fn context(shape: PoolShape) -> Result<Self> {
        ensure!(shape.units > 0 && shape.units % 2 == 0, "C4 context pool needs two equal halves");
        Ok(Self { free_states: (0..shape.sequences).rev().collect(),
            units: RefPagePool::parity(shape.units, SOURCE_PAGE_TOKENS, 0) })
    }

    pub fn free_states(&self) -> usize {
        self.free_states.len()
    }

    /// Admits a sequence that may grow to `capacity` tokens, reserving every
    /// unit up front so decode never runs out mid-request.
    pub fn admit(&mut self, capacity: usize) -> Result<Placement> {
        ensure!(!self.free_states.is_empty(), "every sequence slot is busy");
        let units = self.units.alloc(self.units.pages_for(capacity))
            .map_err(|e| anyhow::anyhow!("compressed cache units exhausted: {e}"))?;
        Ok(Placement::new(self.free_states.pop().expect("checked"), units))
    }

    /// A second sequence that starts as `source`'s first `len` rows: full units shared, the
    /// partial tail unit to be copied (the returned copy), its own state slot.
    pub fn fork(&mut self, source: &Placement, len: usize, capacity: usize) -> Result<(Placement, Option<TailCopy>)> {
        ensure!(!self.free_states.is_empty(), "every sequence slot is busy");
        let total = self.units.pages_for(capacity.max(len));
        let fork = self.units.fork(&source.units, len, total).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok((Placement::new(self.free_states.pop().expect("checked"), fork.pages), fork.copy))
    }

    pub fn release(&mut self, placement: Placement) {
        self.free_states.push(placement.state);
        self.units.release(&placement.units);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_forks_preserve_compression_unit_owner() -> Result<()> {
        let shape = PoolShape::new(3, 4096, 16);
        let mut pool = PoolAllocator::context(shape)?;
        let source = pool.admit(1024)?;
        for (logical, &id) in source.units.iter().enumerate() {
            assert_eq!(id as usize % 2, logical % 2);
        }
        for len in [1, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1024] {
            let (fork, tail) = pool.fork(&source, len, 1280)?;
            for (logical, &id) in fork.units.iter().enumerate() {
                assert_eq!(id as usize % 2, logical % 2);
            }
            if let Some(tail) = tail {
                assert_eq!(tail.from % 2, tail.to % 2);
                assert_eq!(tail.from as usize % 2, (len / SOURCE_PAGE_TOKENS) % 2);
            }
            pool.release(fork);
        }
        pool.release(source);
        assert_eq!(pool.free_states(), shape.sequences);
        assert!(PoolAllocator::context(PoolShape::new(1, 4096, 15)).is_err());
        Ok(())
    }

    #[test]
    fn ring_and_group_slots() -> Result<()> {
        let shape = PoolShape::new(4, 4096, 64);
        assert_eq!(shape.ring_pages, 17);
        let mut pool = PoolAllocator::new(shape);
        let a = pool.admit(1000)?;
        let b = pool.admit(1000)?;
        assert_ne!(a.state, b.state);
        // Positions one ring apart share a slot; the window never spans a ring.
        assert_eq!(a.window_slot(&shape, 5), a.window_slot(&shape, 5 + 17 * 256));
        assert_ne!(a.window_slot(&shape, 5), b.window_slot(&shape, 5));
        // One unit per 256 tokens names the C4, index and C128 pages of those tokens.
        assert_eq!(a.units.len(), 4);
        assert_eq!(a.group_slot(4, 65)?, a.pages[1] * 64 + 1);
        assert_eq!(a.group_slot(128, 3)?, a.pages[1] * 2 + 1);
        assert!(a.group_slot(4, 256).is_err());
        let (c, copy) = pool.fork(&a, 300, 2000)?;
        assert_eq!((c.units.len(), c.units[0]), (8, a.units[0]));
        let copy = copy.context("a partial tail")?;
        assert_eq!((copy.from, copy.to, copy.rows), (a.units[1], c.units[1], 44));
        assert_ne!(c.state, a.state);
        pool.release(c);
        pool.release(a);
        assert_eq!(pool.free_states(), 3);
        assert_eq!(PoolShape::units_for(262_144, 8), 1032);
        Ok(())
    }
}
