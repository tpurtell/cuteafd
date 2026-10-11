//! Token-split (`context`) prefix caching over parity half pools (PLAN.md "v3 attention placement"
//! section 3): a fake two-GPU family in the stub copy engine's memory. Page `i` holds its rows on
//! GPU `i & 1` only (local page `i >> 1`); a replicated per-page row (V4's C128 records) and the
//! sliding-window ring and its marks live on both GPUs. Each GPU has its own copy queue (stream);
//! a forward first checks that every page sits on the owner its logical index names and that both
//! GPUs' replicated bytes equal a straight prefill of the same tokens, so the cache is byte-exact
//! against itself through forks, tail copies, eviction and the host tier.
use super::*;
use cuteafd_hostcache::config::Config as HostConfig;
use cuteafd_hostcache::copy::{CopyEngine, CopyModel, DeviceRange, Event, Stream, StubCopyEngine};
use cuteafd_hostcache::pool::{HostChunk, HostRange, PinnedMemory};
use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

const ROWS: usize = 4; // page rows
const RING: usize = 16; // ring slots per sequence
const WINDOW: usize = 8; // rows a forward reads back from the ring
const ROW: usize = 8; // bytes per row

#[derive(Clone)]
struct Shared(Rc<RefCell<StubCopyEngine>>);

impl PinnedMemory for Shared {
    fn allocate_chunk(&mut self, bytes: usize) -> anyhow::Result<HostChunk> {
        self.0.borrow_mut().allocate_chunk(bytes)
    }
    fn release_chunk(&mut self, chunk: HostChunk) -> anyhow::Result<()> {
        self.0.borrow_mut().release_chunk(chunk)
    }
}

impl CopyEngine for Shared {
    fn d2h(&mut self, stream: Stream, src: DeviceRange, dst: HostRange) -> anyhow::Result<()> {
        self.0.borrow_mut().d2h(stream, src, dst)
    }
    fn h2d(&mut self, stream: Stream, src: HostRange, dst: DeviceRange) -> anyhow::Result<()> {
        self.0.borrow_mut().h2d(stream, src, dst)
    }
    fn d2h_many(&mut self, stream: Stream, copies: &[(DeviceRange, HostRange)]) -> anyhow::Result<()> {
        self.0.borrow_mut().d2h_many(stream, copies)
    }
    fn h2d_many(&mut self, stream: Stream, copies: &[(HostRange, DeviceRange)]) -> anyhow::Result<()> {
        self.0.borrow_mut().h2d_many(stream, copies)
    }
    fn record(&mut self, stream: Stream) -> anyhow::Result<Event> {
        self.0.borrow_mut().record(stream)
    }
    fn completed(&mut self, event: Event) -> anyhow::Result<bool> {
        self.0.borrow_mut().completed(event)
    }
    fn wait(&mut self, event: Event, budget_ns: u64) -> anyhow::Result<bool> {
        self.0.borrow_mut().wait(event, budget_ns)
    }
    fn release_barrier(&mut self, stream: Stream) -> anyhow::Result<()> {
        self.0.borrow_mut().release_barrier(stream)
    }
    fn now_ns(&self) -> u64 {
        self.0.borrow().now_ns()
    }
}

fn value(salt: u64, tokens: &[u32]) -> [u8; ROW] {
    let mut h = DefaultHasher::new();
    salt.hash(&mut h);
    tokens.hash(&mut h);
    h.finish().to_le_bytes()
}

#[derive(Debug, Clone)]
struct Placement {
    pages: Vec<u32>,
    ring: usize,
    len: usize,
}

/// Per GPU: `pages / 2` local pages of `ROWS` rows, then one replicated row per page id, then the
/// rings, then the mark slots.
struct Split {
    mem: Rc<RefCell<StubCopyEngine>>,
    pages: usize,
    rings: usize,
    slots: usize,
    /// Copy queues of GPU0 and GPU1 (their streams).
    queues: [RefCell<Vec<(DeviceRange, DeviceRange)>>; 2],
    /// Tail copies run per GPU stream.
    tail_copies: [Cell<usize>; 2],
    replica_copies: Cell<usize>,
    /// Report a wrong owner for every page (a family/pool disagreement).
    lie_about_owners: Cell<bool>,
}

impl Split {
    fn new(pages: usize, rings: usize, slots: usize) -> Self {
        let bytes = 2 * Self::gpu_bytes(pages, rings, slots);
        let mem = Rc::new(RefCell::new(StubCopyEngine::new(CopyModel::default(), bytes, 1 << 26)));
        Self { mem, pages, rings, slots, queues: Default::default(), tail_copies: Default::default(),
            replica_copies: Cell::new(0), lie_about_owners: Cell::new(false) }
    }
    fn gpu_bytes(pages: usize, rings: usize, slots: usize) -> usize {
        (pages / 2 * ROWS + pages + rings * RING + slots * WINDOW) * ROW
    }
    fn base(&self, gpu: usize) -> usize {
        gpu * Self::gpu_bytes(self.pages, self.rings, self.slots)
    }
    fn gpu_of(&self, range: DeviceRange) -> usize {
        usize::from(range.addr as usize >= self.base(1))
    }
    fn layout(&self) -> FamilyLayout {
        FamilyLayout { page_rows: ROWS, pages: self.pages, page_bytes: ROWS * ROW + ROW, mark_bytes: WINDOW * ROW,
            draft_bytes: 0, rule: ReuseRule::EXACT, mark_store: MarkStore::Arena, page_owners: PageOwners::Parity }
    }
    /// Row `row` of page `page`, on its owner.
    fn page_row(&self, page: u32, row: usize) -> DeviceRange {
        let (gpu, local) = ((page & 1) as usize, (page >> 1) as usize);
        assert!(local < self.pages / 2 && row < ROWS);
        DeviceRange { addr: (self.base(gpu) + (local * ROWS + row) * ROW) as u64, bytes: ROW }
    }
    /// The replicated row of page `page` on `gpu`.
    fn rep_row(&self, gpu: usize, page: u32) -> DeviceRange {
        DeviceRange { addr: (self.base(gpu) + (self.pages / 2 * ROWS + page as usize) * ROW) as u64, bytes: ROW }
    }
    fn ring_row(&self, gpu: usize, ring: usize, position: usize) -> DeviceRange {
        assert!(ring < self.rings);
        DeviceRange { addr: (self.base(gpu) + (self.pages / 2 * ROWS + self.pages + ring * RING + position % RING) * ROW)
            as u64, bytes: ROW }
    }
    fn slot_range(&self, gpu: usize, slot: MarkSlot) -> DeviceRange {
        assert!((slot.0 as usize) < self.slots);
        DeviceRange { addr: (self.base(gpu) + (self.pages / 2 * ROWS + self.pages + self.rings * RING
            + slot.0 as usize * WINDOW) * ROW) as u64, bytes: WINDOW * ROW }
    }
    fn slot_row(&self, gpu: usize, slot: MarkSlot, i: usize) -> DeviceRange {
        DeviceRange { addr: self.slot_range(gpu, slot).addr + (i * ROW) as u64, bytes: ROW }
    }
    fn queue(&self, gpu: usize, from: DeviceRange, to: DeviceRange) {
        assert_eq!((self.gpu_of(from), self.gpu_of(to)), (gpu, gpu), "a local copy runs on its own GPU's stream");
        self.queues[gpu].borrow_mut().push((from, to));
    }
    fn flush(&self) {
        let mut mem = self.mem.borrow_mut();
        for queue in &self.queues {
            for (from, to) in queue.borrow_mut().drain(..) {
                let bytes = mem.read_device(from);
                mem.write_device(to, &bytes);
            }
        }
    }
    /// Prefill `tokens[p.len..]` after checking the whole context on both GPUs.
    fn forward(&self, p: &mut Placement, tokens: &[u32]) -> Result<(), String> {
        self.flush();
        let mut mem = self.mem.borrow_mut();
        for (j, &page) in p.pages.iter().enumerate() {
            if page as usize & 1 != j % 2 {
                return Err(format!("logical page {j} is page {page}, on GPU{}", page & 1));
            }
        }
        for r in 0..p.len {
            let page = p.pages[r / ROWS];
            if mem.read_device(self.page_row(page, r % ROWS)) != value(1, &tokens[..=r]) {
                return Err(format!("page row {r} differs"));
            }
            if r % ROWS == 0 {
                for gpu in 0..2 {
                    if mem.read_device(self.rep_row(gpu, page)) != value(4, &tokens[..=r]) {
                        return Err(format!("GPU{gpu} replicated row of logical page {} differs", r / ROWS));
                    }
                }
            }
        }
        for r in p.len.saturating_sub(WINDOW)..p.len {
            for gpu in 0..2 {
                if mem.read_device(self.ring_row(gpu, p.ring, r)) != value(2, &tokens[..=r]) {
                    return Err(format!("GPU{gpu} ring row {r} differs"));
                }
            }
        }
        for r in p.len..tokens.len() {
            let page = *p.pages.get(r / ROWS).ok_or("past the placement")?;
            mem.write_device(self.page_row(page, r % ROWS), &value(1, &tokens[..=r]));
            for gpu in 0..2 {
                if r % ROWS == 0 {
                    mem.write_device(self.rep_row(gpu, page), &value(4, &tokens[..=r]));
                }
                mem.write_device(self.ring_row(gpu, p.ring, r), &value(2, &tokens[..=r]));
            }
        }
        p.len = tokens.len();
        Ok(())
    }
}

impl PrefixFamily for Split {
    type Placement = Placement;
    fn layout(&self) -> FamilyLayout {
        Split::layout(self)
    }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] {
        &placement.pages
    }
    fn commit_point(&self, placement: &Placement) -> usize {
        placement.len
    }
    fn capture(&self, slot: MarkSlot, p: &Placement, len: usize) -> Result<(), BoxError> {
        assert_eq!(len, p.len, "exact frontiers only");
        for gpu in 0..2 {
            for (i, r) in (len.saturating_sub(WINDOW)..len).enumerate() {
                self.queue(gpu, self.ring_row(gpu, p.ring, r), self.slot_row(gpu, slot, i));
            }
        }
        Ok(())
    }
    fn restore(&self, mark: Option<MarkSlot>, p: &mut Placement, len: usize) -> Result<(), BoxError> {
        let slot = mark.ok_or("exact frontiers only")?;
        for gpu in 0..2 {
            for (i, r) in (len.saturating_sub(WINDOW)..len).enumerate() {
                self.queue(gpu, self.slot_row(gpu, slot, i), self.ring_row(gpu, p.ring, r));
            }
        }
        p.len = len;
        Ok(())
    }
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        let owner = copy.owner.ok_or("a parity pool names the tail's owner")? as usize;
        if copy.from as usize & 1 != owner || copy.to as usize & 1 != owner {
            return Err(format!("tail copy {copy:?} crosses GPUs").into());
        }
        for row in 0..copy.rows {
            self.queue(owner, self.page_row(copy.from, row), self.page_row(copy.to, row));
        }
        self.tail_copies[owner].set(self.tail_copies[owner].get() + 1);
        // The replicated row is on both GPUs: each copies its own.
        for gpu in 0..2 {
            self.queue(gpu, self.rep_row(gpu, copy.from), self.rep_row(gpu, copy.to));
        }
        Ok(())
    }
    fn drain(&self) -> Result<(), BoxError> {
        self.flush();
        Ok(())
    }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        vec![DeviceRange { addr: self.page_row(page, 0).addr, bytes: ROWS * ROW }, self.rep_row(0, page)]
    }
    fn page_device(&self, page: u32) -> Option<u8> {
        Some((page & 1) as u8 ^ u8::from(self.lie_about_owners.get()))
    }
    fn page_replicas(&self, page: u32) -> Vec<(DeviceRange, DeviceRange)> {
        vec![(self.rep_row(0, page), self.rep_row(1, page))]
    }
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        vec![self.slot_range(0, slot)]
    }
    fn mark_replicas(&self, slot: MarkSlot) -> Vec<(DeviceRange, DeviceRange)> {
        vec![(self.slot_range(0, slot), self.slot_range(1, slot))]
    }
    fn copy_replicas(&self, copies: &[(DeviceRange, DeviceRange)]) -> Result<(), BoxError> {
        for &(from, to) in copies {
            assert_eq!((self.gpu_of(from), self.gpu_of(to), from.bytes), (0, 1, to.bytes), "GPU0 -> GPU1 replicas");
            // A peer copy queued on the destination GPU's stream.
            self.queues[1].borrow_mut().push((from, to));
        }
        self.replica_copies.set(self.replica_copies.get() + copies.len());
        Ok(())
    }
}

fn cache(fake: &Split, entries: usize, host_bytes: u64) -> PrefixCache<Shared> {
    let host = HostConfig { bytes: host_bytes, chunk_bytes: 1 << 16, min_tokens: 1, ..HostConfig::default() };
    let config = PrefixConfig { entries, mark_slots: fake.slots, keep_logits: true, min_tokens: 1 };
    PrefixCache::new(fake.layout(), config, Some((host, Shared(fake.mem.clone())))).unwrap()
}

fn seq(base: u32, n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| base + i).collect()
}

/// Admit, prefill, capture the prompt, optionally extend and capture a turn.
fn serve(cache: &mut PrefixCache<Shared>, fake: &Split, ring: usize, prompt: &[u32], generated: &[u32],
    capacity: usize) -> (usize, Vec<u32>, Placement) {
    let admitted = cache.admit(fake, prompt, capacity, false, |pages| Placement { pages, ring, len: 0 }).unwrap();
    let mut placement = admitted.placement;
    if admitted.resume < prompt.len() {
        fake.forward(&mut placement, prompt).unwrap();
        cache.capture(fake, SnapshotKind::Prompt, prompt, &placement, After { greedy: Some(1), logits: None }).unwrap();
    }
    let mut tokens = prompt.to_vec();
    if !generated.is_empty() {
        tokens.extend_from_slice(generated);
        fake.forward(&mut placement, &tokens).unwrap();
        cache.capture(fake, SnapshotKind::Turn, &tokens, &placement, After { greedy: Some(1), logits: None }).unwrap();
    }
    (admitted.resume, tokens, placement)
}

#[test]
fn parity_pool_hands_each_position_its_owner_and_checks_both_halves() {
    let mut pool = RefPagePool::parity(8, 64, 0);
    assert_eq!((pool.capacity(), pool.half_capacity(), pool.free_halves()), (8, [4, 4], [4, 4]));
    // Seven positions: four even (GPU0), three odd (GPU1).
    let a = pool.alloc(7).unwrap();
    assert!(a.iter().enumerate().all(|(j, &page)| page as usize % 2 == j % 2), "{a:?}");
    assert_eq!((a[0], a[1], pool.owner(a[1])), (0, 1, Some(1)));
    assert_eq!((pool.free(), pool.free_halves(), pool.admission_free()), (1, [0, 1], 0));
    // One page free in all, but position 0 needs GPU0's half: refused, nothing taken.
    assert_eq!(pool.alloc(1), Err(PoolExhausted { needed: 1, free: 0, capacity: 4 }));
    assert_eq!(pool.free_halves(), [0, 1]);
    // Position 7 is GPU1's: it fits.
    let tail = pool.alloc_for(7..8).unwrap();
    assert_eq!(pool.owner(tail[0]), Some(1));
    assert_eq!(pool.free(), 0);
    pool.release(&a[..2]);
    assert_eq!(pool.free_halves(), [1, 1]);
    // GPU1's limit: positions 1 and 3 need two odd pages, one is free.
    assert_eq!(pool.alloc_for(1..4), Err(PoolExhausted { needed: 2, free: 1, capacity: 4 }));
    assert_eq!(pool.need_for(1..4), Need([1, 2]));
    // Reserved local pages on each half (GLM Flash's zero page per GPU) never hand out.
    let mut reserved = RefPagePool::parity(8, 64, 1);
    assert_eq!((reserved.capacity(), reserved.half_capacity()), (6, [3, 3]));
    let pages = reserved.alloc(6).unwrap();
    assert!(!pages.contains(&0) && !pages.contains(&1), "{pages:?}");
    // Uniform pools keep their meaning: everything in half 0.
    let uniform = RefPagePool::new(5, 64);
    assert_eq!((uniform.free_halves(), uniform.admission_free(), uniform.owner(3)), ([5, 0], 5, None));
}

#[test]
fn fork_shares_full_pages_and_copies_the_tail_on_its_owner() {
    let mut pool = RefPagePool::parity(16, 64, 0);
    let owner = pool.alloc(4).unwrap(); // 200 rows: pages 0..3, tail at position 3 (GPU1)
    let fork = pool.fork(&owner, 200, 6).unwrap();
    assert_eq!(&fork.pages[..3], &owner[..3]);
    assert!(fork.pages.iter().enumerate().all(|(j, &page)| page as usize % 2 == j % 2), "{:?}", fork.pages);
    assert_eq!(fork.copy, Some(TailCopy { from: owner[3], to: fork.pages[3], rows: 8, owner: Some(1) }));
    // A tail at an even position copies on GPU0.
    let even = pool.fork(&owner, 130, 3).unwrap();
    assert_eq!(even.copy.map(|c| (c.owner, c.from, c.to & 1)), Some((Some(0), owner[2], 0)));
    // Positions 3, 4, 5 are fresh: one even, two odd.
    assert_eq!(pool.fork_need(200, 6), Need([1, 2]));

    // Through the cache: a restore shares GPU0's and GPU1's full pages and copies the tail on its
    // owner only; both continuations stay exact on both GPUs.
    let fake = Split::new(32, 4, 8);
    let mut cache = cache(&fake, 8, 0);
    for (tail_position, prompt_len) in [(2usize, 10usize), (3, 14)] {
        let prompt = seq(100 * prompt_len as u32, prompt_len);
        let (_, _, a) = serve(&mut cache, &fake, 0, &prompt, &[], prompt_len);
        cache.release(&fake, &a.pages).unwrap();
        let before = [fake.tail_copies[0].get(), fake.tail_copies[1].get()];
        let mut next = prompt.clone();
        next.extend(seq(9000, 7));
        let admitted = cache.admit(&fake, &next, 40, false, |pages| Placement { pages, ring: 1, len: 0 }).unwrap();
        assert_eq!(admitted.resume, prompt.len());
        let after = [fake.tail_copies[0].get(), fake.tail_copies[1].get()];
        let owner = tail_position % 2;
        assert_eq!((after[owner] - before[owner], after[1 - owner] - before[1 - owner]), (1, 0),
            "tail at position {tail_position}");
        let mut b = admitted.placement;
        assert_eq!(&b.pages[..tail_position], &a.pages[..tail_position], "full pages shared by reference");
        fake.forward(&mut b, &next).unwrap();
        cache.release(&fake, &b.pages).unwrap();
    }
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free_halves(), [16, 16]);
}

#[test]
fn admission_and_growth_stop_at_either_half_and_waiters_see_it() {
    let fake = Split::new(8, 4, 4);
    let mut cache = cache(&fake, 0, 0);
    let placement = |pages| Placement { pages, ring: 0, len: 0 };
    // 7 pages: GPU0's half full, GPU1 one free.
    let mut a = cache.admit_cold(&fake, 1, 7 * ROWS, placement).unwrap().placement.pages;
    assert_eq!(cache.pool().free_halves(), [0, 1]);
    let error = match cache.admit(&fake, &seq(5, 2), 2, false, placement) {
        Err(error) => error,
        Ok(_) => panic!("position 0 needs a GPU0 page"),
    };
    assert!(matches!(error, PrefixError::Pages(PoolExhausted { needed: 1, free: 0, capacity: 4 })), "{error:?}");
    let mut waiter = DeferredAdmission::default();
    waiter.defer(1, &error, true, cache.pool().release_epoch()).unwrap();
    assert!(matches!(waiter.poll(cache.pool().admission_free(), cache.pool().release_epoch(), true, |_| false),
        AdmissionPoll::Blocked));
    // Growing to 8 pages needs GPU1's last page.
    cache.grow(&fake, &mut a, 8 * ROWS).unwrap();
    assert_eq!(cache.pool().free_halves(), [0, 0]);
    assert!(cache.grow(&fake, &mut a, 9 * ROWS).is_err(), "position 8 needs GPU0");
    assert_eq!(a.len(), 8);
    cache.release(&fake, &a).unwrap();
    assert!(matches!(waiter.poll(cache.pool().admission_free(), cache.pool().release_epoch(), true, |_| false),
        AdmissionPoll::Ready(1)));
    // GPU1's limit: a 3-page sequence takes two GPU0 and one GPU1 pages; four of them exhaust
    // GPU0 first, and a 2-page request then fails on GPU0, not GPU1.
    let held: Vec<_> = (0..2).map(|_| cache.admit_cold(&fake, 1, 3 * ROWS, placement).unwrap().placement.pages).collect();
    assert_eq!(cache.pool().free_halves(), [0, 2]);
    assert!(matches!(cache.admit_cold(&fake, 1, ROWS, placement), Err(PrefixError::Pages(_))));
    for pages in held {
        cache.release(&fake, &pages).unwrap();
    }
    let b = cache.admit_cold(&fake, 1, 2 * ROWS, placement).unwrap().placement.pages;
    let c = cache.admit_cold(&fake, 1, 2 * ROWS, placement).unwrap().placement.pages;
    let d = cache.admit_cold(&fake, 1, 2 * ROWS, placement).unwrap().placement.pages;
    let e = cache.admit_cold(&fake, 1, ROWS, placement).unwrap().placement.pages;
    assert_eq!(cache.pool().free_halves(), [0, 1]);
    for pages in [b, c, d, e] {
        cache.release(&fake, &pages).unwrap();
    }
    assert_eq!(cache.pool().free_halves(), [4, 4]);
    // A layout that cannot be split is refused before anything is allocated.
    let odd = FamilyLayout { pages: 7, ..fake.layout() };
    assert!(matches!(PrefixCache::<Shared>::new(odd, PrefixConfig { entries: 1, mark_slots: 1, keep_logits: false,
        min_tokens: 1 }, None), Err(PrefixError::Layout(_))));
    let pooled = FamilyLayout { mark_store: MarkStore::Pool { pages: 1, reserved: 0 }, ..fake.layout() };
    assert!(matches!(PrefixCache::<Shared>::new(pooled, PrefixConfig { entries: 1, mark_slots: 0, keep_logits: false,
        min_tokens: 1 }, None), Err(PrefixError::Layout(_))));
}

#[test]
fn host_tier_stores_each_page_once_and_restores_replicas_to_both_gpus() {
    let fake = Split::new(32, 4, 8);
    let mut cache = cache(&fake, 8, 1 << 20);
    let prompt = seq(100, 22); // 6 pages: tail at position 5 (GPU1), 2 rows
    let (_, tokens, p) = serve(&mut cache, &fake, 0, &prompt, &seq(500, 5), 40);
    fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    cache.release(&fake, &p.pages).unwrap();
    let host = cache.stats().host.unwrap();
    assert_eq!(host.stores_completed, 2);
    // Each logical page is one host page (6 for the prompt, 7 for the turn, copied or shared by
    // content), never one per GPU.
    assert_eq!(host.pages_copied + host.pages_shared, 6 + 7);
    // Evict the device copies and poison both GPUs' replicated rows, rings and marks.
    cache.clear(&fake).unwrap();
    {
        let mut mem = fake.mem.borrow_mut();
        let total = 2 * Split::gpu_bytes(fake.pages, fake.rings, fake.slots);
        mem.write_device(DeviceRange { addr: 0, bytes: total }, &vec![0xAB; total]);
    }
    // The turn comes back from the host: pages on their owners, replicated rows and the mark copied
    // GPU0 -> GPU1, then the continuation checks every row on both GPUs.
    let replicas = fake.replica_copies.get();
    let mut next = tokens.clone();
    next.extend(seq(3000, 3));
    let (resume, tokens, mut q) = serve(&mut cache, &fake, 2, &next, &[], 48);
    assert_eq!(resume, 27);
    let stats = cache.stats();
    assert_eq!((stats.promotions, stats.host.unwrap().restores, stats.restore_failures), (1, 1, 0));
    assert_eq!(fake.replica_copies.get() - replicas, 7 + 1, "7 pages' replicated rows and one mark");
    let mut longer = tokens;
    longer.push(9);
    fake.forward(&mut q, &longer).unwrap();
    cache.release(&fake, &q.pages).unwrap();
    // A family whose page owners disagree with the pool is a miss, never a wrong restore.
    cache.clear(&fake).unwrap();
    fake.lie_about_owners.set(true);
    let admitted = cache.admit(&fake, &next, 48, false, |pages| Placement { pages, ring: 3, len: 0 }).unwrap();
    assert_eq!((admitted.resume, cache.stats().promotions), (0, 1));
    fake.lie_about_owners.set(false);
    cache.release(&fake, &admitted.placement.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free_halves(), [16, 16]);
}

/// Conversations over a shared system prompt in a small parity pool: forks, tail copies on both
/// owners, eviction and host promotions interleave. After every request the two halves' used
/// pages differ by at most one per live placement or retained snapshot, and every forward is
/// exact on both GPUs.
#[test]
fn eviction_keeps_the_halves_balanced_and_every_restore_exact() {
    for host_bytes in [0u64, 1 << 20] {
        let fake = Split::new(48, 4, 6);
        let mut cache = cache(&fake, 3, host_bytes);
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        let mut conversations: Vec<Vec<u32>> = (0..4).map(|c| seq(100 * (c % 2 + 1), 9)).collect();
        let mut live: Vec<Placement> = Vec::new();
        for step in 0..300 {
            let c = next(4) as usize;
            let mut prompt = conversations[c].clone();
            prompt.extend(seq(10_000 + step * 7, 1 + next(6) as usize));
            if prompt.len() > 60 {
                prompt = seq(100 * (c as u32 % 2 + 1), 9);
            }
            let generated = seq(50_000 + step * 3, next(5) as usize);
            let capacity = prompt.len() + generated.len() + next(4) as usize;
            // The previous request keeps running while this one is admitted (unless it holds this
            // conversation's ring), so live placements count against the halves too.
            if let Some(i) = live.iter().position(|p| p.ring == c) {
                let p = live.remove(i);
                cache.release(&fake, &p.pages).unwrap();
            }
            let (_, tokens, p) = serve(&mut cache, &fake, c, &prompt, &generated, capacity);
            conversations[c] = tokens;
            for old in live.drain(..) {
                cache.release(&fake, &old.pages).unwrap();
            }
            live.push(p);
            fake.mem.borrow_mut().advance(1_000_000);
            cache.tick();
            let stats = cache.stats();
            let used = stats.pages_used_by_gpu.unwrap();
            let holders = live.len() + stats.entries_prompt + stats.entries_turn;
            assert!(used[0].abs_diff(used[1]) <= holders, "step {step}: used {used:?}, {holders} holders");
        }
        for p in live.drain(..) {
            cache.release(&fake, &p.pages).unwrap();
        }
        let stats = cache.stats();
        assert!(stats.hits > 50 && stats.cow_copies > 0, "{stats:?}");
        assert!(fake.tail_copies[0].get() > 0 && fake.tail_copies[1].get() > 0, "tails on both owners");
        assert_eq!(stats.pages - stats.pages_free, stats.pages_retained, "{stats:?}");
        cache.clear(&fake).unwrap();
        assert_eq!((cache.pool().free_halves(), cache.arena().in_use()), ([24, 24], 0));
    }
}
