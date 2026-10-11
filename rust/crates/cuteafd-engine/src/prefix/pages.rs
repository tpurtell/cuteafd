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
//!
//! Parity half pools ([`RefPagePool::parity`], token-split `context` attention, PLAN.md "v3
//! attention placement" section 3): two half pools under one id space. Page id `i` lives on GPU
//! `i & 1` at local index `i >> 1`, and a sequence's page at logical position `j` always gets an
//! id of parity `j % 2` ([`RefPagePool::alloc_for`]). Ownership is a function of the logical
//! position alone, so a forked full page and a copied tail keep their position and therefore
//! their GPU; a sequence of `n` pages takes `ceil(n/2)` pages of GPU0 and `floor(n/2)` of GPU1
//! (at most one page of imbalance per placement or snapshot). Admission needs both halves.
use thiserror::Error;

/// Not enough free pages; nothing was allocated. In a parity pool the numbers are those of the
/// first half that is short (`half`), so a waiter compares them with
/// [`RefPagePool::admission_free`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("device page pool exhausted: {needed} pages needed, {free} free of {capacity}")]
pub struct PoolExhausted {
    pub needed: usize,
    pub free: usize,
    pub capacity: usize,
}

/// A partial tail copied by a fork: rows `[0, rows)` of page `from` into page `to`. In a parity
/// pool both pages sit at the same logical position, so they share an owner (`owner`, the GPU
/// whose stream runs the copy); `None` in a uniform pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailCopy {
    pub from: u32,
    pub to: u32,
    pub rows: usize,
    pub owner: Option<u8>,
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

/// Fresh pages an allocation needs from each half pool (a uniform pool counts in `[0]` only).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Need(pub [usize; 2]);

impl Need {
    pub fn total(self) -> usize {
        self.0[0] + self.0[1]
    }
}

impl std::ops::Add for Need {
    type Output = Need;
    fn add(self, other: Need) -> Need {
        Need([self.0[0] + other.0[0], self.0[1] + other.0[1]])
    }
}

pub struct RefPagePool {
    refs: Vec<u32>,
    generation: Vec<u32>,
    /// Free indices per half (a uniform pool uses `[0]`); popped from the end, so a fresh pool
    /// hands out its first unreserved page, then the next, ...
    free: [Vec<u32>; 2],
    /// Page ids never handed out: the leading `reserved` (uniform), or the leading `reserved`
    /// local indices of each half (parity: ids `0..2 * reserved`).
    reserved: usize,
    parity: bool,
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
        assert!(reserved <= pages, "{reserved} reserved pages of {pages}");
        Self::build(pages, page_rows, reserved, false)
    }

    /// Two half pools of `pages / 2` local pages each under one id space (`pages` even): id `i`
    /// is local page `i >> 1` of GPU `i & 1`. Each half keeps its first `reserved` local pages
    /// (GLM 5.3 Flash's zero page on every GPU).
    pub fn parity(pages: usize, page_rows: usize, reserved: usize) -> Self {
        assert!(pages % 2 == 0, "a parity pool has two equal halves, not {pages} pages");
        assert!(reserved <= pages / 2, "{reserved} reserved pages of {} per half", pages / 2);
        Self::build(pages, page_rows, reserved, true)
    }

    fn build(pages: usize, page_rows: usize, reserved: usize, parity: bool) -> Self {
        assert!(page_rows > 0, "pages hold at least one row");
        let pages = u32::try_from(pages).expect("page count fits u32");
        let first = if parity { 2 * reserved as u32 } else { reserved as u32 };
        let mut free: [Vec<u32>; 2] = Default::default();
        for page in (first..pages).rev() {
            free[if parity { (page & 1) as usize } else { 0 }].push(page);
        }
        Self {
            refs: vec![0; pages as usize],
            generation: vec![0; pages as usize],
            free,
            reserved,
            parity,
            page_rows,
            release_epoch: 0,
        }
    }

    /// Whether this is a parity pool ([`RefPagePool::parity`]).
    pub fn is_parity(&self) -> bool {
        self.parity
    }
    /// The GPU whose half holds `page` (parity pools; `None` in a uniform pool).
    pub fn owner(&self, page: u32) -> Option<u8> {
        self.parity.then_some((page & 1) as u8)
    }
    /// Pages the pool can hand out (reserved ones excluded).
    pub fn capacity(&self) -> usize {
        self.refs.len() - self.reserved_ids()
    }
    /// Pages one half can hand out (the whole capacity in a uniform pool's `[0]`).
    pub fn half_capacity(&self) -> [usize; 2] {
        if self.parity { [self.capacity() / 2; 2] } else { [self.capacity(), 0] }
    }
    /// Leading page indices never handed out (per half in a parity pool).
    pub fn reserved(&self) -> usize {
        self.reserved
    }
    fn reserved_ids(&self) -> usize {
        if self.parity { 2 * self.reserved } else { self.reserved }
    }
    pub fn free(&self) -> usize {
        self.free[0].len() + self.free[1].len()
    }
    /// Free pages per half (a uniform pool's are all in `[0]`).
    pub fn free_halves(&self) -> [usize; 2] {
        [self.free[0].len(), self.free[1].len()]
    }
    /// The free count a deferred admission compares against [`PoolExhausted::free`]: the whole
    /// pool's, or the scarcer half's in a parity pool.
    pub fn admission_free(&self) -> usize {
        if self.parity { self.free[0].len().min(self.free[1].len()) } else { self.free() }
    }
    pub fn used(&self) -> usize {
        self.capacity() - self.free()
    }
    /// Allocated pages per half.
    pub fn used_halves(&self) -> [usize; 2] {
        let capacity = self.half_capacity();
        [capacity[0] - self.free[0].len(), capacity[1] - self.free[1].len()]
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

    /// Fresh pages for logical positions `positions` of a sequence.
    pub fn need_for(&self, positions: std::ops::Range<usize>) -> Need {
        let n = positions.len();
        if !self.parity {
            return Need([n, 0]);
        }
        let even = (positions.start..positions.end).filter(|j| j % 2 == 0).count();
        Need([even, n - even])
    }

    /// Whether `need` fits the free pages of each half.
    pub fn fits(&self, need: Need) -> bool {
        need.0[0] <= self.free[0].len() && need.0[1] <= self.free[1].len()
    }

    /// `n` fresh pages for logical positions `0..n`, each with one reference; all or nothing.
    pub fn alloc(&mut self, n: usize) -> Result<Vec<u32>, PoolExhausted> {
        self.alloc_for(0..n)
    }

    /// Fresh pages for logical positions `positions` (a parity pool hands position `j` a page
    /// of parity `j % 2`), each with one reference; all or nothing.
    pub fn alloc_for(&mut self, positions: std::ops::Range<usize>) -> Result<Vec<u32>, PoolExhausted> {
        let need = self.need_for(positions.clone());
        if !self.fits(need) {
            return Err(self.exhausted(need));
        }
        let pages: Vec<u32> = positions
            .map(|j| self.free[if self.parity { j % 2 } else { 0 }].pop().expect("checked"))
            .collect();
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
                self.free[if self.parity { (page & 1) as usize } else { 0 }].push(page);
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
        self.fork_need(len, total).total()
    }

    /// [`RefPagePool::fork_cost`] per half.
    pub fn fork_need(&self, len: usize, total: usize) -> Need {
        let full = self.full_pages(len);
        let total = total.max(full + usize::from(len % self.page_rows > 0)).max(1);
        self.need_for(full..total)
    }

    /// Pages for a sequence of `total` pages that starts as `source[..len rows]`: the full pages
    /// are shared, a partial tail is copied into a fresh page at the same position (the returned
    /// [`TailCopy`], on the position's owner in a parity pool), and fresh pages follow. All or
    /// nothing.
    pub fn fork(&mut self, source: &[u32], len: usize, total: usize) -> Result<Fork, PoolExhausted> {
        let full = self.full_pages(len);
        let partial = len % self.page_rows;
        assert!(source.len() >= full + usize::from(partial > 0), "fork source shorter than {len} rows");
        let total = total.max(full + usize::from(partial > 0)).max(1);
        let fresh = self.alloc_for(full..total)?;
        self.share(&source[..full]);
        let copy = (partial > 0).then(|| TailCopy { from: source[full], to: fresh[0], rows: partial,
            owner: self.owner(source[full]) });
        debug_assert!(copy.is_none_or(|c| self.owner(c.from) == self.owner(c.to)), "a tail copy keeps its owner");
        let mut pages = source[..full].to_vec();
        pages.extend(fresh);
        Ok(Fork { pages, copy })
    }

    fn exhausted(&self, need: Need) -> PoolExhausted {
        if !self.parity {
            return PoolExhausted { needed: need.0[0], free: self.free[0].len(), capacity: self.capacity() };
        }
        let half = usize::from(need.0[0] <= self.free[0].len());
        PoolExhausted { needed: need.0[half], free: self.free[half].len(), capacity: self.capacity() / 2 }
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
        assert_eq!(fork.copy, Some(TailCopy { from: owner[3], to: fork.pages[3], rows: 8, owner: None }));
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
