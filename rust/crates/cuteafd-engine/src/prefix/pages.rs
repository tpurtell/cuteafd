//! One refcounted device page pool for every generic family (it replaces the per-family free
//! lists). A page holds `page_rows` token rows of every paged layer; a sequence's placement is a
//! list of page indices. Sharing is by reference count: a retained snapshot and every request
//! restored from it hold the same full pages, which nobody writes again because every writer
//! appends past its own committed length. The one page that can still change is a partial tail
//! (its rows past the snapshot's length belong to whoever owns it), so [`RefPagePool::fork`]
//! copies it into a fresh page instead of sharing it (copy on write, done eagerly: one page).
//!
//! Identity: a page index is reused after it is freed, so every free bumps the index's
//! generation. `(index, generation)` names one allocation for the host tier's page sharing.
//!
//! Reserved pages ([`RefPagePool::with_reserved`]): the leading page indices that no allocation
//! ever hands out, so they keep whatever the family put there at start-up.
use thiserror::Error;

/// Not enough free pages; nothing was allocated.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("device page pool exhausted: {needed} pages needed, {free} free of {capacity}")]
pub struct PoolExhausted {
    pub needed: usize,
    pub free: usize,
    pub capacity: usize,
}

/// A partial tail copied by a fork: rows `[0, rows)` of page `from` into page `to`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailCopy {
    pub from: u32,
    pub to: u32,
    pub rows: usize,
}

/// The pages of a forked sequence and the tail copy the family must run before it writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fork {
    pub pages: Vec<u32>,
    pub copy: Option<TailCopy>,
}

/// A page freed by a release: its index and the generation it had while it was allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FreedPage {
    pub index: u32,
    pub generation: u32,
}

pub struct RefPagePool {
    refs: Vec<u32>,
    generation: Vec<u32>,
    /// Free indices; popped from the end, so a fresh pool hands out its first unreserved page,
    /// then the next, ...
    free: Vec<u32>,
    /// Leading page indices never handed out.
    reserved: usize,
    page_rows: usize,
    release_epoch: u64,
}

impl RefPagePool {
    pub fn new(pages: usize, page_rows: usize) -> Self {
        Self::with_reserved(pages, page_rows, 0)
    }

    /// A pool of `pages` page indices whose first `reserved` are never handed out (they keep
    /// whatever the family wrote there at start-up).
    pub fn with_reserved(pages: usize, page_rows: usize, reserved: usize) -> Self {
        assert!(page_rows > 0, "pages hold at least one row");
        assert!(reserved <= pages, "{reserved} reserved pages of {pages}");
        let (pages, first) = (u32::try_from(pages).expect("page count fits u32"), reserved as u32);
        Self {
            refs: vec![0; pages as usize],
            generation: vec![0; pages as usize],
            free: (first..pages).rev().collect(),
            reserved,
            page_rows,
            release_epoch: 0,
        }
    }

    /// Pages the pool can hand out (reserved ones excluded).
    pub fn capacity(&self) -> usize {
        self.refs.len() - self.reserved
    }
    /// Leading page indices never handed out.
    pub fn reserved(&self) -> usize {
        self.reserved
    }
    pub fn free(&self) -> usize {
        self.free.len()
    }
    pub fn used(&self) -> usize {
        self.capacity() - self.free()
    }
    pub fn page_rows(&self) -> usize {
        self.page_rows
    }
    /// Progress relevant to admission, including a running placement releasing
    /// pages still held by an inactive snapshot (those pages become evictable
    /// without increasing the physical free-page count).
    pub fn release_epoch(&self) -> u64 {
        self.release_epoch
    }
    /// Pages a sequence of `tokens` tokens needs (at least one: a placement always has a page).
    pub fn pages_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_rows).max(1)
    }
    /// Pages whose every row is below `len` (the shareable prefix of a snapshot at `len`).
    pub fn full_pages(&self, len: usize) -> usize {
        len / self.page_rows
    }
    pub fn refs(&self, page: u32) -> u32 {
        self.refs[page as usize]
    }
    pub fn generation(&self, page: u32) -> u32 {
        self.generation[page as usize]
    }
    /// Pages referenced more than once.
    pub fn shared(&self) -> usize {
        self.refs.iter().filter(|&&r| r > 1).count()
    }

    /// `n` fresh pages, each with one reference; all or nothing.
    pub fn alloc(&mut self, n: usize) -> Result<Vec<u32>, PoolExhausted> {
        if n > self.free.len() {
            return Err(self.exhausted(n));
        }
        let pages: Vec<u32> = (0..n).map(|_| self.free.pop().expect("checked")).collect();
        for &page in &pages {
            debug_assert_eq!(self.refs[page as usize], 0);
            self.refs[page as usize] = 1;
        }
        Ok(pages)
    }

    /// One more reference to each of `pages` (which must be allocated).
    pub fn share(&mut self, pages: &[u32]) {
        for &page in pages {
            let refs = &mut self.refs[page as usize];
            assert!(*refs > 0, "shared page {page} is free");
            *refs += 1;
        }
    }

    /// Drop one reference to each of `pages`; returns the pages that became free. The caller
    /// must have drained every queued reader and writer of a page before its last release.
    pub fn release(&mut self, pages: &[u32]) -> Vec<FreedPage> {
        let mut freed = Vec::new();
        for &page in pages {
            let refs = &mut self.refs[page as usize];
            assert!(*refs > 0, "released page {page} is free");
            *refs -= 1;
            if *refs == 0 {
                let generation = self.generation[page as usize];
                self.generation[page as usize] = generation.wrapping_add(1);
                self.free.push(page);
                freed.push(FreedPage { index: page, generation });
            }
        }
        if !pages.is_empty() {
            self.release_epoch = self.release_epoch.wrapping_add(1);
        }
        freed
    }

    /// Fresh pages a fork of a snapshot at `len` into a sequence of `total` pages needs.
    pub fn fork_cost(&self, len: usize, total: usize) -> usize {
        total.saturating_sub(self.full_pages(len))
    }

    /// Pages for a sequence of `total` pages that starts as `source[..len rows]`: the full pages
    /// are shared, a partial tail is copied into a fresh page (the returned [`TailCopy`]), and
    /// fresh pages follow. All or nothing.
    pub fn fork(&mut self, source: &[u32], len: usize, total: usize) -> Result<Fork, PoolExhausted> {
        let full = self.full_pages(len);
        let partial = len % self.page_rows;
        assert!(source.len() >= full + usize::from(partial > 0), "fork source shorter than {len} rows");
        let total = total.max(full + usize::from(partial > 0)).max(1);
        let fresh = self.alloc(total - full)?;
        self.share(&source[..full]);
        let copy = (partial > 0).then(|| TailCopy { from: source[full], to: fresh[0], rows: partial });
        let mut pages = source[..full].to_vec();
        pages.extend(fresh);
        Ok(Fork { pages, copy })
    }

    fn exhausted(&self, needed: usize) -> PoolExhausted {
        PoolExhausted { needed, free: self.free.len(), capacity: self.capacity() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_is_all_or_nothing_and_release_bumps_generations() {
        let mut pool = RefPagePool::new(4, 64);
        let a = pool.alloc(3).unwrap();
        assert_eq!(a, vec![0, 1, 2]);
        assert_eq!(pool.alloc(2), Err(PoolExhausted { needed: 2, free: 1, capacity: 4 }));
        assert_eq!(pool.free(), 1);
        pool.share(&a[..1]);
        assert_eq!(pool.release(&a), vec![FreedPage { index: 1, generation: 0 }, FreedPage { index: 2, generation: 0 }]);
        assert_eq!((pool.refs(0), pool.generation(1)), (1, 1));
        assert_eq!(pool.release(&a[..1]), vec![FreedPage { index: 0, generation: 0 }]);
        assert_eq!(pool.free(), 4);
    }

    #[test]
    fn reserved_pages_are_never_handed_out() {
        let mut pool = RefPagePool::with_reserved(5, 64, 1);
        assert_eq!((pool.capacity(), pool.free(), pool.reserved()), (4, 4, 1));
        let a = pool.alloc(4).unwrap();
        assert_eq!(a, vec![1, 2, 3, 4]);
        assert_eq!(pool.alloc(1), Err(PoolExhausted { needed: 1, free: 0, capacity: 4 }));
        pool.release(&a);
        // Freed pages come back in any order; page 0 never does.
        let b = pool.alloc(4).unwrap();
        assert!(!b.contains(&0) && pool.used() == 4, "{b:?}");
        let fork = pool.fork(&b, 100, 3);
        assert!(fork.is_err(), "a fork needs two fresh pages and none is free");
        pool.release(&b);
        assert_eq!(pool.free(), 4);
        assert_eq!(RefPagePool::new(3, 64).capacity(), 3);
    }

    #[test]
    fn fork_shares_full_pages_and_copies_the_partial_tail() {
        let mut pool = RefPagePool::new(16, 64);
        let owner = pool.alloc(4).unwrap(); // 0..4, sequence of 200 rows
        let fork = pool.fork(&owner, 200, 6).unwrap();
        assert_eq!(&fork.pages[..3], &owner[..3]);
        assert_eq!(fork.pages.len(), 6);
        assert_eq!(fork.copy, Some(TailCopy { from: owner[3], to: fork.pages[3], rows: 8 }));
        assert_eq!((pool.refs(owner[0]), pool.refs(owner[3])), (2, 1));
        assert_eq!(pool.fork_cost(200, 6), 3);
        // Page-aligned: nothing to copy.
        let aligned = pool.fork(&owner, 192, 3).unwrap();
        assert_eq!((aligned.pages, aligned.copy), (owner[..3].to_vec(), None));
        // A fork never gets fewer pages than its rows need.
        assert_eq!(pool.fork(&owner, 130, 1).unwrap().pages.len(), 3);
        let before = pool.free();
        assert!(pool.fork(&owner, 64, 64).is_err());
        assert_eq!(pool.free(), before, "a failed fork holds nothing");
        assert_eq!(pool.refs(owner[0]), 4);
    }
}
