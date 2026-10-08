//! A fake family whose device state lives in the host tier's stub memory: page row `r` holds a
//! hash of `tokens[..=r]`, the sliding-window ring holds the same for its last rows, and every
//! forward first checks that the whole context it reads is exactly what a straight prefill of
//! the same tokens would have written. Copies are queued until a drain or a forward (stream
//! order), so a snapshot published before its copies drained restores wrong bytes and fails.
//! Its mark (the ring's last `WINDOW` rows) lives in arena slots or, as `pooled`, in
//! `MARK_PAGES` pages of the same pool (mark row `i` at row `i % ROWS` of page `i / ROWS`).
//!
//! Like GLM 5.3 Flash's decode sparse MLA, every forward also reads page 0 row 0 as the stand-in
//! for masked rows and weights it by zero: mark rows carry a NaN-like byte ([`POISON`]), so a
//! mark on page 0 poisons every forward. A pooled fake therefore reserves page 0.
use super::*;
use cuteafd_hostcache::config::Config as HostConfig;
use cuteafd_hostcache::copy::{CopyEngine, CopyModel, DeviceRange, Event, Stream, StubCopyEngine};
use cuteafd_hostcache::pool::{HostChunk, HostRange, PinnedMemory};
use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use super::points::{plan as plan_points, PointPolicy};

include!("media_tests.rs");

const ROWS: usize = 4; // page rows
const RING: usize = 16; // ring slots per sequence
const WINDOW: usize = 8; // rows a forward reads back from the ring
const ROW: usize = 8; // bytes per row
const MARK_PAGES: usize = WINDOW.div_ceil(ROWS); // pool pages of a pool-page mark
/// Byte 7 of every ring (and so mark) row: as in an E4M3 record, 0x7F reads as NaN. Page rows
/// and untouched memory carry 0 there.
const POISON: u8 = 0x7F;

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

/// Page rows (salt 1) carry 0 in byte 7, ring rows (salt 2) [`POISON`].
fn value(salt: u64, tokens: &[u32]) -> [u8; ROW] {
    let mut h = DefaultHasher::new();
    salt.hash(&mut h);
    tokens.hash(&mut h);
    let mut bytes = h.finish().to_le_bytes();
    match salt {
        1 => bytes[7] = 0,
        2 => bytes[7] = POISON,
        _ => {}
    }
    bytes
}

#[derive(Debug, Clone)]
struct Placement {
    pages: Vec<u32>,
    ring: usize,
    len: usize,
    /// First position the ring holds for this sequence (a partial restore starts it empty).
    ring_from: usize,
}

enum Op {
    Copy(DeviceRange, DeviceRange),
}

struct Fake {
    mem: Rc<RefCell<StubCopyEngine>>,
    pages: usize,
    rings: usize,
    slots: usize,
    queue: RefCell<Vec<Op>>,
    fail_restore: Cell<bool>,
    drains: Cell<usize>,
    rule: ReuseRule,
    /// A pages-only family (GLM 5.3): no mark; restores resume at the snapshot's length and the
    /// host tier keeps a one-byte stand-in tail.
    markless: bool,
    /// Marks in pool pages (`MarkStore::Pool`) instead of the arena.
    pooled: bool,
    /// Leading pool pages never handed out (a pooled fake reserves page 0, the stand-in).
    reserved: usize,
}

impl Fake {
    fn new(pages: usize, rings: usize, slots: usize) -> Self {
        let bytes = (pages * ROWS + rings * RING + slots * WINDOW) * ROW;
        let mem = Rc::new(RefCell::new(StubCopyEngine::new(CopyModel::default(), bytes, 1 << 26)));
        Self { mem, pages, rings, slots, queue: RefCell::new(Vec::new()), fail_restore: Cell::new(false), drains: Cell::new(0),
            rule: ReuseRule::EXACT, markless: false, pooled: false, reserved: 0 }
    }
    /// Marks in pool pages: no arena slot exists (any use of one panics). `pages` pages hand
    /// out, past the reserved page 0.
    fn pooled(pages: usize, rings: usize) -> Self {
        Self { pooled: true, reserved: 1, ..Self::new(pages + 1, rings, 0) }
    }
    /// [`Fake::pooled`] without the reserved page: a mark can land on the stand-in.
    fn pooled_unreserved(pages: usize, rings: usize) -> Self {
        Self { pooled: true, ..Self::new(pages, rings, 0) }
    }
    fn layout(&self) -> FamilyLayout {
        FamilyLayout { page_rows: ROWS, pages: self.pages, page_bytes: ROWS * ROW,
            mark_bytes: if self.markless { 0 } else { WINDOW * ROW }, draft_bytes: 0, rule: self.rule,
            mark_store: if self.pooled { MarkStore::Pool { pages: MARK_PAGES, reserved: self.reserved } }
                else { MarkStore::Arena } }
    }
    /// Row `i` of the mark held in pool `pages`.
    fn mark_row(&self, pages: &[u32], i: usize) -> DeviceRange {
        assert!(pages.len() == MARK_PAGES && i < WINDOW, "mark of {} pages", pages.len());
        self.page_row(pages[i / ROWS], i % ROWS)
    }
    /// The ring rows a mark at `len` holds, or why `p` cannot capture there.
    fn mark_window(p: &Placement, len: usize) -> Result<std::ops::Range<usize>, BoxError> {
        if len > p.len || p.len - len > RING - WINDOW || len.saturating_sub(WINDOW) < p.ring_from.min(len) {
            return Err(format!("capture at {len} of {} is out of reach", p.len).into());
        }
        Ok(len.saturating_sub(WINDOW)..len)
    }
    fn page_row(&self, page: u32, row: usize) -> DeviceRange {
        DeviceRange { addr: ((page as usize * ROWS + row) * ROW) as u64, bytes: ROW }
    }
    fn ring_row(&self, ring: usize, position: usize) -> DeviceRange {
        assert!(ring < self.rings);
        DeviceRange { addr: ((self.pages * ROWS + ring * RING + position % RING) * ROW) as u64, bytes: ROW }
    }
    fn slot_row(&self, slot: MarkSlot, i: usize) -> DeviceRange {
        assert!((slot.0 as usize) < self.slots);
        DeviceRange { addr: ((self.pages * ROWS + self.rings * RING + slot.0 as usize * WINDOW + i) * ROW) as u64, bytes: ROW }
    }
    fn flush(&self) {
        let mut mem = self.mem.borrow_mut();
        for Op::Copy(from, to) in self.queue.borrow_mut().drain(..) {
            let bytes = mem.read_device(from);
            mem.write_device(to, &bytes);
        }
    }
    /// Prefill `tokens[p.len..]` after checking the whole context and the stand-in for masked
    /// rows (page 0 row 0); returns a logit row.
    fn forward(&self, p: &mut Placement, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.flush();
        let mut mem = self.mem.borrow_mut();
        if mem.read_device(self.page_row(0, 0))[7] == POISON {
            return Err("the stand-in for masked rows (page 0 row 0) holds mark bytes".into());
        }
        for r in 0..p.len {
            let page = p.pages[r / ROWS];
            if mem.read_device(self.page_row(page, r % ROWS)) != value(1, &tokens[..=r]) {
                return Err(format!("page row {r} differs"));
            }
        }
        for r in p.len.saturating_sub(WINDOW).max(p.ring_from)..p.len {
            if mem.read_device(self.ring_row(p.ring, r)) != value(2, &tokens[..=r]) {
                return Err(format!("ring row {r} differs"));
            }
        }
        for r in p.len..tokens.len() {
            let page = *p.pages.get(r / ROWS).ok_or("past the placement")?;
            mem.write_device(self.page_row(page, r % ROWS), &value(1, &tokens[..=r]));
            mem.write_device(self.ring_row(p.ring, r), &value(2, &tokens[..=r]));
        }
        p.len = tokens.len();
        let v = u64::from_le_bytes(value(3, tokens));
        Ok((0..8).map(|i| ((v >> (i * 8)) & 0xff) as f32).collect())
    }
}

impl PrefixFamily for Fake {
    type Placement = Placement;
    fn layout(&self) -> FamilyLayout {
        Fake::layout(self)
    }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] {
        &placement.pages
    }
    fn commit_point(&self, placement: &Placement) -> usize {
        placement.len
    }
    fn capture_reach(&self) -> usize {
        RING - WINDOW
    }
    fn capture(&self, slot: MarkSlot, p: &Placement, len: usize) -> Result<(), BoxError> {
        for (i, r) in Fake::mark_window(p, len)?.enumerate() {
            self.queue.borrow_mut().push(Op::Copy(self.ring_row(p.ring, r), self.slot_row(slot, i)));
        }
        Ok(())
    }
    fn capture_pages(&self, pages: &[u32], p: &Placement, len: usize) -> Result<(), BoxError> {
        assert!(self.pooled, "pool-page capture of an arena family");
        for (i, r) in Fake::mark_window(p, len)?.enumerate() {
            self.queue.borrow_mut().push(Op::Copy(self.ring_row(p.ring, r), self.mark_row(pages, i)));
        }
        Ok(())
    }
    fn restore_pages(&self, pages: &[u32], p: &mut Placement, len: usize) -> Result<(), BoxError> {
        assert!(self.pooled, "pool-page restore of an arena family");
        if self.fail_restore.get() {
            return Err("injected restore failure".into());
        }
        for (i, r) in (len.saturating_sub(WINDOW)..len).enumerate() {
            self.queue.borrow_mut().push(Op::Copy(self.mark_row(pages, i), self.ring_row(p.ring, r)));
        }
        p.len = len;
        Ok(())
    }
    fn mark_page_segments(&self, pages: &[u32]) -> Result<Vec<DeviceRange>, BoxError> {
        Ok((0..WINDOW).map(|i| self.mark_row(pages, i)).collect())
    }
    fn restore(&self, mark: Option<MarkSlot>, p: &mut Placement, len: usize) -> Result<(), BoxError> {
        if self.fail_restore.get() {
            return Err("injected restore failure".into());
        }
        let Some(slot) = mark else {
            // Partial hit: the ring starts empty here and the replayed rows rebuild it.
            p.len = len;
            p.ring_from = len;
            return Ok(());
        };
        let first = len.saturating_sub(WINDOW);
        for (i, r) in (first..len).enumerate() {
            self.queue.borrow_mut().push(Op::Copy(self.slot_row(slot, i), self.ring_row(p.ring, r)));
        }
        p.len = len;
        Ok(())
    }
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        for row in 0..copy.rows {
            self.queue.borrow_mut().push(Op::Copy(self.page_row(copy.from, row), self.page_row(copy.to, row)));
        }
        Ok(())
    }
    fn drain(&self) -> Result<(), BoxError> {
        self.drains.set(self.drains.get() + 1);
        self.flush();
        Ok(())
    }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        vec![DeviceRange { addr: self.page_row(page, 0).addr, bytes: ROWS * ROW }]
    }
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        vec![DeviceRange { addr: self.slot_row(slot, 0).addr, bytes: WINDOW * ROW }]
    }
    fn host_tail(&self) -> Vec<DeviceRange> {
        if self.markless { vec![DeviceRange { addr: self.slot_row(MarkSlot(0), 0).addr, bytes: 1 }] } else { Vec::new() }
    }
}

fn config(entries: usize, slots: usize) -> PrefixConfig {
    PrefixConfig { entries, mark_slots: slots, keep_logits: true, min_tokens: 1 }
}

fn cache(fake: &Fake, entries: usize, host_bytes: u64) -> PrefixCache<Shared> {
    let host = HostConfig { bytes: host_bytes, chunk_bytes: 1 << 16, min_tokens: 1, ..HostConfig::default() };
    PrefixCache::new(fake.layout(), config(entries, fake.slots), Some((host, Shared(fake.mem.clone())))).unwrap()
}

#[test]
fn growing_live_leases_is_atomic_and_never_evicts_live_pages() {
    let fake = Fake::new(8, 2, 4);
    let mut cache = cache(&fake, 0, 0);
    let placement = |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 };
    let mut a = cache.admit_cold(&fake, 1, 1, placement).unwrap().placement.pages;
    let b = cache.admit_cold(&fake, 1, 1, placement).unwrap().placement.pages;
    let rows = cache.pool().page_rows();
    cache.grow(&fake, &mut a, 7 * rows).unwrap();
    assert_eq!(a.len(), 7);
    let before = a.clone();
    assert!(cache.grow(&fake, &mut a, 8 * rows).is_err());
    assert_eq!(a, before);
    assert_eq!(cache.pool().refs(b[0]), 1);
    cache.release(&fake, &b).unwrap();
    cache.grow(&fake, &mut a, 8 * rows).unwrap();
    assert_eq!(a.len(), 8);
    cache.release(&fake, &a).unwrap();
    assert_eq!(cache.pool().free(), 8);
}

/// One request through the cache: admit, prefill the rest, capture its prompt, then optionally
/// "decode" `generated` tokens and capture a turn. Returns (resume, final tokens, placement).
fn serve(cache: &mut PrefixCache<Shared>, fake: &Fake, ring: usize, prompt: &[u32], generated: &[u32],
    capacity: usize) -> (usize, Vec<u32>, Placement) {
    let admitted = cache.admit(fake, prompt, capacity, false, |pages| Placement { pages, ring, len: 0, ring_from: 0 }).unwrap();
    let mut placement = admitted.placement;
    assert_eq!(placement.len, admitted.resume);
    let logits = if admitted.resume == prompt.len() {
        admitted.after.expect("exact-length hit brings its first token").logits.unwrap().to_vec()
    } else {
        let logits = fake.forward(&mut placement, prompt).unwrap();
        cache.capture(fake, SnapshotKind::Prompt, prompt, &placement, After::from_logits(&logits, true)).unwrap();
        logits
    };
    assert_eq!(logits.len(), 8);
    let mut tokens = prompt.to_vec();
    if !generated.is_empty() {
        tokens.extend_from_slice(generated);
        let logits = fake.forward(&mut placement, &tokens).unwrap();
        cache.capture(fake, SnapshotKind::Turn, &tokens, &placement, After::from_logits(&logits, true)).unwrap();
    }
    (admitted.resume, tokens, placement)
}

fn seq(base: u32, n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| base + i).collect()
}

#[test]
fn kv_pressure_delays_a_second_job_and_keeps_active_state_and_host_prefix_exact() {
    let fake = Fake::new(4, 2, 4);
    let mut cache = cache(&fake, 2, 1 << 20);
    let prompt = seq(100, 8);
    let (_, original, mut first) = serve(&mut cache, &fake, 0, &prompt, &[], 12);
    fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    let second_prompt = seq(1000, 8);
    let error = match cache.admit(&fake, &second_prompt, 12, false,
        |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }) {
        Err(error) => error,
        Ok(_) => panic!("both three-page requests cannot fit in four pages"),
    };
    let mut waiter = DeferredAdmission::default();
    waiter.defer(second_prompt, &error, true, cache.pool().release_epoch()).unwrap();
    assert!(matches!(waiter.poll(cache.pool().free(), cache.pool().release_epoch(), true, |_| false), AdmissionPoll::Blocked));
    // Failed admission may evict inactive snapshots to RAM, but may never
    // overwrite the running request's page rows or positional ring.
    let mut longer = original.clone();
    longer.push(900);
    fake.forward(&mut first, &longer).unwrap();
    cache.release(&fake, &first.pages).unwrap();
    let AdmissionPoll::Ready(second_prompt) = waiter.poll(cache.pool().free(), cache.pool().release_epoch(), false, |_| false) else {
        panic!("released running request must unblock admission");
    };
    let (_, _, second) = serve(&mut cache, &fake, 1, &second_prompt, &[], 12);
    cache.release(&fake, &second.pages).unwrap();
    fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    let admitted = cache.admit(&fake, &original, 12, false,
        |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 }).unwrap();
    assert_eq!(admitted.resume, original.len(), "pressure preserves a reusable host prefix");
    assert!(admitted.source.unwrap().host);
    let mut restored = admitted.placement;
    // Every prior KV and ring row is checked before extending the prefix.
    fake.forward(&mut restored, &longer).unwrap();
    cache.release(&fake, &restored.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 4);
    assert_eq!(cache.arena().in_use(), 0);
}

#[test]
fn completed_retained_pages_unblock_admission_while_another_request_is_running() {
    let fake = Fake::new(6, 3, 6);
    let mut cache = cache(&fake, 2, 0);
    let (_, first_tokens, first) = serve(&mut cache, &fake, 0, &seq(100, 12), &[], 12);
    let (_, second_tokens, mut second) = serve(&mut cache, &fake, 1, &seq(1000, 8), &[], 8);
    let third_tokens = seq(2000, 12);
    let error = match cache.admit(&fake, &third_tokens, 12, false,
        |pages| Placement { pages, ring: 2, len: 0, ring_from: 0 }) {
        Err(error) => error,
        Ok(_) => panic!("three running placements need eight pages, but the arena holds six"),
    };
    let mut waiter = DeferredAdmission::default();
    waiter.defer(third_tokens, &error, true, cache.pool().release_epoch()).unwrap();
    assert_eq!(cache.pool().free(), 1);
    // Completion retains all of the first placement's pages. Dropping the
    // active references makes that snapshot evictable, without freeing any
    // physical page; the unrelated second request still runs.
    cache.capture(&fake, SnapshotKind::Turn, &first_tokens, &first, After::default()).unwrap();
    let free_before = cache.pool().free();
    cache.release(&fake, &first.pages).unwrap();
    assert_eq!(cache.pool().free(), free_before);
    let AdmissionPoll::Ready(third_tokens) = waiter.poll(cache.pool().free(), cache.pool().release_epoch(), true, |_| false) else {
        panic!("retained completed pages must not stall the FIFO until every other request finishes");
    };
    let admitted = cache.admit(&fake, &third_tokens, 12, false,
        |pages| Placement { pages, ring: 2, len: 0, ring_from: 0 }).unwrap();
    let mut third = admitted.placement;
    fake.forward(&mut third, &third_tokens).unwrap();
    fake.forward(&mut second, &second_tokens).unwrap();
    cache.release(&fake, &second.pages).unwrap();
    cache.release(&fake, &third.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 6);
    assert_eq!(cache.arena().in_use(), 0);
}

#[test]
fn prompt_repeat_is_an_exact_hit_with_its_logits_and_no_forward() {
    let fake = Fake::new(64, 4, 8);
    let mut cache = cache(&fake, 4, 0);
    let prompt = seq(100, 50);
    let (resume, _, a) = serve(&mut cache, &fake, 0, &prompt, &[], 64);
    assert_eq!(resume, 0);
    let (resume, _, mut b) = serve(&mut cache, &fake, 1, &prompt, &[], 64);
    assert_eq!(resume, 50);
    // B continues exactly as a straight prefill would (the forward checks every context row).
    let mut longer = prompt.clone();
    longer.extend(seq(900, 9));
    fake.forward(&mut b, &longer).unwrap();
    let stats = cache.stats();
    assert_eq!((stats.hits, stats.exact_hits, stats.hit_tokens, stats.cow_copies), (1, 1, 50, 1));
    cache.release(&fake, &a.pages).unwrap();
    cache.release(&fake, &b.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 64);
    assert_eq!(cache.arena().in_use(), 0);
}

#[test]
fn a_turn_snapshot_serves_the_next_turn_and_the_writer_keeps_its_own_tail() {
    let fake = Fake::new(64, 4, 8);
    let mut cache = cache(&fake, 4, 0);
    let prompt = seq(100, 30);
    let (_, turn, mut a) = serve(&mut cache, &fake, 0, &prompt, &seq(500, 13), 80);
    // The writer keeps decoding past its turn snapshot (43 rows: page 10 is partial and copied).
    let mut more = turn.clone();
    more.extend(seq(700, 7));
    fake.forward(&mut a, &more).unwrap();
    // The next turn re-sends the whole conversation plus a new user message.
    let mut next = turn.clone();
    next.extend(seq(800, 11));
    let (resume, _, mut b) = serve(&mut cache, &fake, 1, &next, &[], 80);
    assert_eq!(resume, turn.len());
    let mut after = next.clone();
    after.push(1);
    fake.forward(&mut b, &after).unwrap();
    // The writer's rows past the snapshot are untouched by B.
    fake.forward(&mut a, &more).unwrap();
    // A prefix of the snapshot is not an exact frontier: MiMo-style families reuse no partial match.
    let mut partial = turn[..35].to_vec();
    partial.push(4242);
    let (resume, _, c) = serve(&mut cache, &fake, 2, &partial, &[], 80);
    assert_eq!(resume, prompt.len(), "the prompt snapshot is the exact ancestor");
    for p in [a, b, c] {
        cache.release(&fake, &p.pages).unwrap();
    }
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 64);
}

#[test]
fn a_parked_prefill_without_logits_gives_way_to_a_shorter_snapshot() {
    let fake = Fake::new(64, 4, 8);
    let mut cache = cache(&fake, 4, 0);
    let prompt = seq(100, 40);
    let (_, _, a) = serve(&mut cache, &fake, 0, &prompt[..20], &[], 64);
    // A prefill of the whole prompt is cancelled at a chunk boundary (32 rows): parked.
    let mut admitted = cache.admit(&fake, &prompt, 64, false, |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
    assert_eq!(admitted.resume, 20);
    fake.forward(&mut admitted.placement, &prompt[..32]).unwrap();
    assert!(cache.park(&fake, &prompt[..32], &admitted.placement).unwrap());
    cache.release(&fake, &admitted.placement.pages).unwrap();
    // The retry resumes at the parked boundary.
    let (resume, _, b) = serve(&mut cache, &fake, 1, &prompt, &[], 64);
    assert_eq!(resume, 32);
    // A request of exactly the parked tokens cannot take its first token from the parked
    // snapshot (no logits), so it resumes from the 20-token prompt snapshot instead.
    let admitted = cache.admit(&fake, &prompt[..32], 64, false, |pages| Placement { pages, ring: 2, len: 0, ring_from: 0 }).unwrap();
    assert_eq!((admitted.resume, admitted.after.is_none()), (20, true));
    assert_eq!(cache.stats().parked, 1);
    for pages in [a.pages, b.pages, admitted.placement.pages] {
        cache.release(&fake, &pages).unwrap();
    }
}

#[test]
fn eviction_is_least_recent_prompt_first_and_admission_makes_room() {
    // 20 pages: a 30-token request takes 8, its snapshot shares 7 of them and copies 1.
    let fake = Fake::new(20, 4, 8);
    let mut cache = cache(&fake, 8, 0);
    let (_, _, a) = serve(&mut cache, &fake, 0, &seq(100, 30), &[], 30);
    cache.release(&fake, &a.pages).unwrap();
    let (_, _, b) = serve(&mut cache, &fake, 1, &seq(200, 30), &[], 30);
    cache.release(&fake, &b.pages).unwrap();
    assert_eq!(cache.stats().entries_prompt, 2);
    // Touch A's snapshot so B's is the least recently used.
    let (resume, _, a2) = serve(&mut cache, &fake, 0, &seq(100, 30), &[], 30);
    assert_eq!(resume, 30);
    cache.release(&fake, &a2.pages).unwrap();
    // A third conversation needs room: B's snapshot goes, A's stays.
    let (_, _, c) = serve(&mut cache, &fake, 2, &seq(300, 40), &[], 40);
    assert!(cache.stats().evictions >= 1);
    let (resume, _, a3) = serve(&mut cache, &fake, 0, &seq(100, 30), &[], 30);
    assert_eq!(resume, 30);
    cache.release(&fake, &a3.pages).unwrap();
    let (resume, _, b2) = serve(&mut cache, &fake, 1, &seq(200, 30), &[], 30);
    assert_eq!(resume, 0);
    for pages in [c.pages, b2.pages] {
        cache.release(&fake, &pages).unwrap();
    }
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 20);
}

#[test]
fn full_arena_and_full_banks_evict_the_victim() {
    let fake = Fake::new(128, 4, 2);
    let mut cache = cache(&fake, 8, 0);
    for base in [100, 200, 300] {
        let (_, _, p) = serve(&mut cache, &fake, 0, &seq(base, 20), &[], 20);
        cache.release(&fake, &p.pages).unwrap();
    }
    assert_eq!((cache.stats().entries_prompt, cache.arena().in_use()), (2, 2));
    let (resume, _, p) = serve(&mut cache, &fake, 0, &seq(100, 20), &[], 20);
    assert_eq!(resume, 0, "the oldest snapshot was evicted for the third mark");
    cache.release(&fake, &p.pages).unwrap();
    let mut small = cache_with(&fake, 1);
    for base in [100, 200] {
        let (_, _, p) = serve(&mut small, &fake, 0, &seq(base, 20), &[], 20);
        small.release(&fake, &p.pages).unwrap();
    }
    assert_eq!(small.stats().entries_prompt, 1);
}

fn cache_with(fake: &Fake, entries: usize) -> PrefixCache<Shared> {
    cache(fake, entries, 0)
}

#[test]
fn a_failed_restore_is_a_miss_and_holds_nothing() {
    let fake = Fake::new(32, 4, 4);
    let mut cache = cache(&fake, 4, 0);
    let (_, _, a) = serve(&mut cache, &fake, 0, &seq(100, 20), &[], 20);
    cache.release(&fake, &a.pages).unwrap();
    fake.fail_restore.set(true);
    let admitted = cache.admit(&fake, &seq(100, 21), 24, false, |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
    fake.fail_restore.set(false);
    assert_eq!((admitted.resume, cache.stats().restore_failures), (0, 1));
    cache.release(&fake, &admitted.placement.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 32);
}

#[test]
fn disabled_cache_is_a_plain_allocator() {
    let fake = Fake::new(8, 2, 0);
    let mut cache = cache(&fake, 0, 1 << 20);
    let (resume, _, a) = serve(&mut cache, &fake, 0, &seq(1, 20), &[], 20);
    let (resume2, _, b) = serve(&mut cache, &fake, 1, &seq(1, 11), &[], 12);
    assert_eq!((resume, resume2), (0, 0));
    assert!(cache.admit(&fake, &seq(1, 4), 4, false, |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 }).is_err());
    cache.release(&fake, &a.pages).unwrap();
    cache.release(&fake, &b.pages).unwrap();
    assert_eq!((cache.pool().free(), cache.stats().captures_prompt), (8, 0));
}

#[test]
fn host_tier_restores_evicted_snapshots_exactly_and_shares_identical_prefixes() {
    let fake = Fake::new(24, 4, 8);
    let mut cache = cache(&fake, 8, 1 << 20);
    let system = seq(100, 16);
    let mut first = system.clone();
    first.extend(seq(1000, 10));
    let mut second = system.clone();
    second.extend(seq(2000, 10));
    for (ring, prompt) in [(0, &first), (1, &second)] {
        let (_, _, p) = serve(&mut cache, &fake, ring, prompt, &[], 26);
        // The store stream runs ahead of anything the family does next: a snapshot published
        // before its capture copies drained would be stored with stale bytes.
        fake.mem.borrow_mut().advance(10_000_000);
        cache.tick();
        cache.release(&fake, &p.pages).unwrap();
    }
    let host = cache.stats().host.unwrap();
    assert_eq!(host.stores_completed, 2);
    // The second prompt's first 16 tokens (4 full pages) were computed by a different request
    // into different device pages, yet share the first snapshot's host copies by content.
    assert_eq!((host.pages_copied, host.pages_shared), (7 + 3, 4));
    // Evict both from the device.
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 24);
    // The first conversation comes back: promoted from the host tier, then continued exactly.
    let mut next = first.clone();
    next.extend(seq(3000, 5));
    let (resume, _, mut p) = serve(&mut cache, &fake, 2, &next, &[], 40);
    assert_eq!(resume, first.len());
    let stats = cache.stats();
    assert_eq!((stats.promotions, stats.host.unwrap().restores), (1, 1));
    let mut longer = next.clone();
    longer.push(7);
    fake.forward(&mut p, &longer).unwrap();
    cache.release(&fake, &p.pages).unwrap();
}

#[test]
fn pages_only_family_restores_from_device_and_host_without_a_mark() {
    let mut fake = Fake::new(24, 4, 1);
    fake.markless = true;
    let mut cache = cache(&fake, 8, 1 << 20);
    assert_eq!(cache.arena().slots(), 0);
    let prompt = seq(100, 22);
    let (resume, _, p) = serve(&mut cache, &fake, 0, &prompt, &[], 26);
    assert_eq!(resume, 0);
    fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    cache.release(&fake, &p.pages).unwrap();
    // A device hit: shared full pages, the copied tail, no mark.
    let mut next = prompt.clone();
    next.extend(seq(3000, 5));
    let (resume, _, p) = serve(&mut cache, &fake, 1, &next, &[], 40);
    assert_eq!(resume, prompt.len());
    fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    cache.release(&fake, &p.pages).unwrap();
    assert_eq!(cache.stats().host.unwrap().stores_completed, 2);
    // Evicted from the device, the longer prompt comes back from the host tier, exactly.
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 24);
    let (resume, _, mut q) = serve(&mut cache, &fake, 2, &next, &[], 40);
    assert_eq!(resume, next.len());
    let stats = cache.stats();
    assert_eq!((stats.promotions, stats.marks_in_use, stats.host.unwrap().restores), (1, 0, 1));
    let mut longer = next.clone();
    longer.push(7);
    fake.forward(&mut q, &longer).unwrap();
    cache.release(&fake, &q.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 24);
}

/// Randomized conversations over shared system prompts, with rings reused across requests and a
/// small pool, so eviction, forks, copies and host promotions interleave. Every forward checks its
/// whole context; at the end every page and slot is back.
#[test]
fn torture_interleaved_conversations_stay_exact_and_leak_nothing() {
    for (host_bytes, pooled) in [(0u64, false), (1 << 20, false), (0, true), (1 << 20, true)] {
        // Pool-page marks compete with the rows for the same 40 pages.
        let fake = if pooled { Fake::pooled(40, 4) } else { Fake::new(40, 4, 6) };
        let mut cache = cache(&fake, 3, host_bytes);
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        let mut conversations: Vec<Vec<u32>> = (0..4).map(|c| seq(100 * (c % 2 + 1), 9)).collect();
        for step in 0..300 {
            let c = next(4) as usize;
            let mut prompt = conversations[c].clone();
            prompt.extend(seq(10_000 + step * 7, 1 + next(6) as usize));
            if prompt.len() > 60 {
                prompt = seq(100 * (c as u32 % 2 + 1), 9);
            }
            let generated = seq(50_000 + step * 3, next(5) as usize);
            let capacity = prompt.len() + generated.len() + next(4) as usize;
            let (_, tokens, p) = serve(&mut cache, &fake, c, &prompt, &generated, capacity);
            conversations[c] = tokens;
            fake.mem.borrow_mut().advance(1_000_000);
            cache.tick();
            cache.release(&fake, &p.pages).unwrap();
        }
        let stats = cache.stats();
        assert!(stats.hits > 50, "{stats:?}");
        // Idle: every used page belongs to a retained snapshot (its rows or its mark), every
        // mark to an entry.
        assert_eq!(stats.pages - stats.pages_free, stats.pages_retained, "{stats:?}");
        assert_eq!(stats.marks_in_use, stats.entries_prompt + stats.entries_turn, "{stats:?}");
        assert_eq!(stats.mark_pages, if pooled { MARK_PAGES * stats.marks_in_use } else { 0 }, "{stats:?}");
        cache.clear(&fake).unwrap();
        assert_eq!((cache.pool().free(), cache.arena().in_use(), cache.stats().mark_pages), (40, 0, 0), "{stats:?}");
        assert_eq!((cache.pool().capacity(), cache.pool().reserved()), (40, usize::from(pooled)));
    }
}

/// Prefill `prompt` from `resume` in `plan`'s chunks, capturing its intermediate points (the way
/// serve-mimo does), then the prompt-end snapshot.
fn serve_with_points(cache: &mut PrefixCache<Shared>, fake: &Fake, ring: usize, prompt: &[u32], boundaries: &[usize],
    policy: PointPolicy) -> (usize, Placement) {
    let admitted = cache.admit(fake, prompt, prompt.len() + 4, false,
        |pages| Placement { pages, ring, len: 0, ring_from: 0 }).unwrap();
    let mut placement = admitted.placement;
    let resume = admitted.resume;
    if resume == prompt.len() {
        return (resume, placement);
    }
    let plan = plan_points(resume, prompt.len(), 5, boundaries, fake.capture_reach(), 1, policy);
    for (index, &end) in plan.chunks.iter().enumerate() {
        fake.forward(&mut placement, &prompt[..end]).unwrap();
        for &(_, point) in plan.points.iter().filter(|&&(chunk, _)| chunk == index) {
            assert!(cache.capture(fake, SnapshotKind::Prompt, &prompt[..point], &placement, After::default()).unwrap());
        }
    }
    let logits = fake.forward(&mut placement, prompt).unwrap();
    cache.capture(fake, SnapshotKind::Prompt, prompt, &placement, After::from_logits(&logits, true)).unwrap();
    (resume, placement)
}

#[test]
fn divergent_suffixes_restore_exactly_from_the_deepest_shared_point() {
    let fake = Fake::new(128, 4, 16);
    let mut cache = cache(&fake, 16, 0);
    // Messages start at 0, 12, 23 and 37; the prompt is 40 tokens (chunks of 5 rows).
    let prompt = seq(100, 40);
    let policy = PointPolicy { gap: 10, boundaries: 2, per_request: 4 };
    let (resume, a) = serve_with_points(&mut cache, &fake, 0, &prompt, &[0, 12, 23, 37], policy);
    assert_eq!(resume, 0);
    // Periodic 10, 20, 30 and boundaries 23, 37: the cap keeps the four deepest (20, 23, 30,
    // 37), plus the prompt-end snapshot.
    assert_eq!(cache.stats().captures_prompt, 5);
    cache.release(&fake, &a.pages).unwrap();
    for (diverge, expected) in [(38, 37), (36, 30), (29, 23), (22, 20), (15, 0), (7, 0)] {
        let mut other = prompt[..diverge].to_vec();
        other.extend(seq(9000 + diverge as u32 * 10, 6));
        let (resume, mut b) = serve_with_points(&mut cache, &fake, 1, &other, &[], PointPolicy { gap: 0, boundaries: 0,
            per_request: 0 });
        assert_eq!(resume, expected, "diverging at {diverge}");
        // Every context row (pages and ring) of the continuation is exact.
        let mut longer = other.clone();
        longer.push(1);
        fake.forward(&mut b, &longer).unwrap();
        cache.release(&fake, &b.pages).unwrap();
    }
    // A point more than the family's reach behind the commit point is refused, not approximated.
    let admitted = cache.admit(&fake, &seq(5000, 30), 30, false,
        |pages| Placement { pages, ring: 2, len: 0, ring_from: 0 }).unwrap();
    let mut p = admitted.placement;
    fake.forward(&mut p, &seq(5000, 30)).unwrap();
    assert!(matches!(cache.capture(&fake, SnapshotKind::Prompt, &seq(5000, 21), &p, After::default()),
        Err(PrefixError::Frontier { tokens: 21, committed: 30, reach: 8 })));
    assert!(cache.capture(&fake, SnapshotKind::Prompt, &seq(5000, 22), &p, After::default()).unwrap());
    cache.release(&fake, &p.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 128);
}

/// Opt-in partial reuse (`PREFIX_PARTIAL=on`): a partial match shares pages below the replay
/// window and restarts the positional state there; the page rows it shares stay exact and the
/// snapshot it came from is untouched (its later exact restore still checks every row).
#[test]
fn opt_in_partial_reuse_replays_the_window_without_touching_shared_pages() {
    let mut fake = Fake::new(64, 4, 8);
    fake.rule = ReuseRule { align: ROWS, replay: Some(WINDOW) };
    let mut cache = cache(&fake, 4, 0);
    let prompt = seq(100, 30);
    let (_, _, a) = serve(&mut cache, &fake, 0, &prompt, &[], 30);
    cache.release(&fake, &a.pages).unwrap();
    let mut other = prompt[..27].to_vec();
    other.extend(seq(900, 5));
    let (resume, _, b) = serve(&mut cache, &fake, 1, &other, &[], 40);
    assert_eq!(resume, 24 - WINDOW);
    assert_eq!(cache.stats().partial_hits, 1);
    cache.release(&fake, &b.pages).unwrap();
    let mut longer = prompt.clone();
    longer.push(5);
    let (resume, _, c) = serve(&mut cache, &fake, 2, &longer, &[], 40);
    assert_eq!((resume, cache.stats().partial_hits), (30, 1), "an exact ancestor saves more");
    cache.release(&fake, &c.pages).unwrap();
}

/// Agentic sessions: every turn leaves a prompt-end and a turn-end snapshot, so a session that
/// forks from any earlier turn (a retry, a subagent, an edited later message) restores the
/// deepest turn it still shares, byte-exactly, and prefills only its own suffix.
#[test]
fn sessions_fork_from_the_deepest_earlier_turn() {
    let fake = Fake::new(160, 4, 16);
    let mut cache = cache(&fake, 8, 0);
    let mut conversation = seq(100, 20);
    let mut turn_ends = Vec::new();
    for turn in 0..3u32 {
        let (_, tokens, p) = serve(&mut cache, &fake, 0, &conversation, &seq(1000 * (turn + 1), 7), 64);
        cache.release(&fake, &p.pages).unwrap();
        turn_ends.push(tokens.len());
        conversation = tokens;
        conversation.extend(seq(5000 + turn * 10, 5)); // tool result / next user message
    }
    for (turn, &end) in turn_ends.iter().enumerate() {
        let mut fork = conversation[..end].to_vec();
        fork.extend(seq(7000 + turn as u32 * 10, 6));
        let (resume, _, mut p) = serve(&mut cache, &fake, 1, &fork, &[], 64);
        assert_eq!(resume, end, "fork after turn {turn}");
        let mut longer = fork.clone();
        longer.push(3);
        fake.forward(&mut p, &longer).unwrap();
        cache.release(&fake, &p.pages).unwrap();
    }
}

#[test]
fn failed_store_release_keeps_device_pages_mark_and_host_slabs() {
    use cuteafd_hostcache::copy::CopyFault;
    for pooled in [false, true] {
        let fake = if pooled { Fake::pooled(24, 2) } else { Fake::new(24, 2, 4) };
        let mut cache = cache(&fake, 2, 1 << 20);
        fake.mem.borrow_mut().inject(CopyFault::StreamStalls(Stream::Store));
        let prompt = seq(700, 13);
        let (_, _, placement) = serve(&mut cache, &fake, 0, &prompt, &[], 20);
        cache.release(&fake, &placement.pages).unwrap();
        let free = cache.pool().free();
        let marks = cache.stats().marks_in_use;
        let held = cache.stats().host.unwrap().bytes_used;
        assert!(free < 24 && marks > 0 && held > 0);
        for _ in 0..2 {
            assert!(cache.clear(&fake).is_err());
            assert_eq!(cache.pool().free(), free);
            assert_eq!(cache.stats().marks_in_use, marks);
            assert_eq!(cache.stats().host.unwrap().bytes_used, held);
        }
        // The entry was never removed: its exact positional state still restores.
        let (resume, _, mut restored) = serve(&mut cache, &fake, 1, &prompt, &[], 20);
        assert_eq!(resume, prompt.len());
        let mut continuation = prompt;
        continuation.push(99);
        fake.forward(&mut restored, &continuation).unwrap();
        cache.release(&fake, &restored.pages).unwrap();
    }
}

/// Pool-page marks: a capture takes its mark's pages from the pool beside the snapshot's rows,
/// a device hit restores the ring from them, and the host tier stores rows and mark together
/// and brings both back into fresh pages; every forward checks every context row exactly.
#[test]
fn pool_page_marks_round_trip_exactly_on_the_device_and_through_the_host_tier() {
    let fake = Fake::pooled(32, 4);
    let mut cache = cache(&fake, 8, 1 << 20);
    assert_eq!((cache.arena().slots(), cache.layout().mark_store),
        (0, MarkStore::Pool { pages: MARK_PAGES, reserved: 1 }));
    let prompt = seq(100, 22);
    let (resume, _, p) = serve(&mut cache, &fake, 0, &prompt, &[], 26);
    assert_eq!(resume, 0);
    let stats = cache.stats();
    assert_eq!((stats.marks_in_use, stats.mark_pages, stats.mark_slots), (1, MARK_PAGES, 0));
    // 7 placement pages, the snapshot's copied tail page and its mark's pages.
    assert_eq!(cache.pool().free(), 32 - 7 - 1 - MARK_PAGES);
    fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    cache.release(&fake, &p.pages).unwrap();
    // A device hit: the ring comes back from the mark's pages.
    let mut next = prompt.clone();
    next.extend(seq(3000, 5));
    let (resume, _, mut q) = serve(&mut cache, &fake, 1, &next, &[], 40);
    assert_eq!(resume, prompt.len());
    let mut longer = next.clone();
    longer.push(7);
    fake.forward(&mut q, &longer).unwrap();
    fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    cache.release(&fake, &q.pages).unwrap();
    assert_eq!(cache.stats().host.unwrap().stores_completed, 2);
    // Evicted from the device (rows and marks), the longer prompt comes back from the host tier
    // into fresh pages, rows and mark, and continues exactly.
    cache.clear(&fake).unwrap();
    assert_eq!((cache.pool().free(), cache.stats().mark_pages), (32, 0));
    let mut fork = next.clone();
    fork.extend(seq(5000, 3));
    let (resume, _, mut r) = serve(&mut cache, &fake, 2, &fork, &[], 40);
    assert_eq!(resume, next.len());
    let stats = cache.stats();
    assert_eq!((stats.promotions, stats.host.unwrap().restores), (1, 1));
    let mut longest = fork.clone();
    longest.push(9);
    fake.forward(&mut r, &longest).unwrap();
    cache.release(&fake, &r.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!((cache.pool().free(), cache.arena().in_use()), (32, 0));
}

/// Pool-page marks under pressure: a capture into a full pool evicts the least recently used
/// snapshot, rows and mark, for its own mark pages. The fake has no arena slot and no device
/// memory beyond the pool and its rings, so no mark can live anywhere but in pool pages.
#[test]
fn a_capture_into_a_full_pool_evicts_for_its_mark_pages() {
    let fake = Fake::pooled(10, 2);
    let mut cache = cache(&fake, 8, 0);
    let (_, _, a) = serve(&mut cache, &fake, 0, &seq(100, 12), &[], 12);
    cache.release(&fake, &a.pages).unwrap();
    // A's snapshot: its three page-aligned rows pages and its mark's two.
    assert_eq!((cache.pool().free(), cache.stats().mark_pages), (10 - 3 - MARK_PAGES, MARK_PAGES));
    // B runs on four pages; its prompt snapshot needs two mark pages with one free: A goes.
    let (_, _, b) = serve(&mut cache, &fake, 1, &seq(200, 16), &[], 16);
    let stats = cache.stats();
    assert_eq!((stats.evictions, stats.captures_prompt, stats.capture_skips), (1, 2, 0));
    assert_eq!((stats.entries_prompt, stats.marks_in_use, stats.mark_pages), (1, 1, MARK_PAGES));
    assert_eq!(cache.arena().slots(), 0);
    cache.release(&fake, &b.pages).unwrap();
    assert_eq!(cache.pool().free(), 10 - 4 - MARK_PAGES);
    // B's snapshot restores exactly (ring from its mark pages); A's is gone.
    let (resume, _, mut b2) = serve(&mut cache, &fake, 1, &seq(200, 16), &[], 20);
    assert_eq!(resume, 16);
    let mut longer = seq(200, 16);
    longer.push(5);
    fake.forward(&mut b2, &longer).unwrap();
    cache.release(&fake, &b2.pages).unwrap();
    let (resume, _, a2) = serve(&mut cache, &fake, 0, &seq(100, 12), &[], 12);
    assert_eq!(resume, 0);
    cache.release(&fake, &a2.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 10);
}

/// With every page held by a running request nothing can be evicted: the capture is skipped
/// and counted, holds no page, and leaves the request's own rows and ring alone.
#[test]
fn a_capture_with_nothing_to_evict_is_skipped_and_counted() {
    let fake = Fake::pooled(6, 2);
    let mut cache = cache(&fake, 8, 0);
    let prompt = seq(100, 18);
    let admitted = cache.admit(&fake, &prompt, 20, false, |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 })
        .unwrap();
    let mut a = admitted.placement;
    let logits = fake.forward(&mut a, &prompt).unwrap();
    // Five pages run A; its snapshot needs a copied tail page and two mark pages, one is free.
    assert_eq!(cache.pool().free(), 1);
    assert!(!cache.capture(&fake, SnapshotKind::Prompt, &prompt, &a, After::from_logits(&logits, true)).unwrap());
    let stats = cache.stats();
    assert_eq!((stats.capture_skips, stats.captures_prompt, stats.entries_prompt, stats.mark_pages), (1, 0, 0, 0));
    assert_eq!(cache.pool().free(), 1);
    let mut longer = prompt.clone();
    longer.push(1);
    fake.forward(&mut a, &longer).unwrap();
    cache.release(&fake, &a.pages).unwrap();
    assert_eq!(cache.pool().free(), 6);
}

/// Pool-page marks with the host tier on: under pressure a capture evicts the least recently
/// used snapshot only once its rows and mark reached pinned RAM (the eviction waits for the
/// write-behind copy still in flight), and a later request brings both back into fresh pages
/// and continues exactly.
#[test]
fn pool_mark_evictions_reach_the_host_tier_and_come_back_exactly() {
    let fake = Fake::pooled(10, 3);
    let mut cache = cache(&fake, 8, 1 << 20);
    let first = seq(100, 12);
    let (_, _, a) = serve(&mut cache, &fake, 0, &first, &[], 12);
    cache.release(&fake, &a.pages).unwrap();
    // The second prompt's snapshot needs the first one's pages; nothing has advanced the copy
    // clock, so the first snapshot's store is still in flight when it is evicted.
    let (_, _, b) = serve(&mut cache, &fake, 1, &seq(200, 16), &[], 16);
    cache.release(&fake, &b.pages).unwrap();
    let stats = cache.stats();
    let host = stats.host.clone().unwrap();
    assert_eq!((stats.evictions, stats.host_evict_waits, stats.host_evict_uncached), (1, 1, 0), "{stats:?}");
    assert_eq!((host.stores_completed, stats.entries_prompt), (1, 1), "{stats:?}");
    // The first conversation continues: rows and mark come back from RAM (evicting the second
    // snapshot, again only after its copy landed), and every context row checks out.
    let mut next = first.clone();
    next.push(5);
    let (resume, _, mut c) = serve(&mut cache, &fake, 2, &next, &[], 16);
    assert_eq!(resume, first.len());
    let stats = cache.stats();
    assert_eq!((stats.promotions, stats.host.unwrap().restores, stats.evictions, stats.host_evict_waits), (1, 1, 2, 2));
    let mut longer = next.clone();
    longer.push(6);
    fake.forward(&mut c, &longer).unwrap();
    cache.release(&fake, &c.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!((cache.pool().free(), cache.stats().mark_pages), (10, 0));
}

/// A mark on page 0 poisons the stand-in every forward reads for masked rows (GLM 5.3 Flash's
/// decode sparse MLA reads record slot 0 for every masked candidate and weights it by zero, and
/// 0 x NaN is NaN). Unreserved, the free list hands page 0 to the second request's mark and its
/// next forward fails; with page 0 reserved no allocation ever takes it.
#[test]
fn a_mark_never_lands_on_the_stand_in_page() {
    for reserved in [false, true] {
        let fake = if reserved { Fake::pooled(8, 2) } else { Fake::pooled_unreserved(8, 2) };
        let mut cache = cache(&fake, 4, 0);
        // A takes the first page handed out, its mark the next two; evicted, page 0 (unreserved)
        // goes back under the mark's pages.
        let (_, _, a) = serve(&mut cache, &fake, 0, &seq(100, 4), &[], 4);
        assert_eq!(a.pages, vec![if reserved { 1 } else { 0 }]);
        cache.release(&fake, &a.pages).unwrap();
        cache.clear(&fake).unwrap();
        // B's mark takes the two pages freed first: 0 and 1 unreserved.
        let prompt = seq(200, 4);
        let (_, _, mut b) = serve(&mut cache, &fake, 1, &prompt, &[], 8);
        let mut longer = prompt.clone();
        longer.push(9);
        match (reserved, fake.forward(&mut b, &longer)) {
            (false, Err(error)) => assert!(error.contains("stand-in"), "{error}"),
            (true, Ok(_)) => {}
            (reserved, outcome) => panic!("reserved {reserved}: {outcome:?}"),
        }
        cache.release(&fake, &b.pages).unwrap();
        cache.clear(&fake).unwrap();
    }
}

/// A pool-page layout must hold a mark of at least one page in its unreserved pages; an arena
/// family keeps its arena.
#[test]
fn pool_page_layouts_are_checked() {
    let fake = Fake::pooled(4, 1);
    for (pages, reserved) in [(0, 1), (5, 1), (2, 4), (1, 5)] {
        let layout = FamilyLayout { mark_store: MarkStore::Pool { pages, reserved }, ..fake.layout() };
        assert!(matches!(PrefixCache::<Shared>::new(layout, config(2, 0), None), Err(PrefixError::Layout(_))));
    }
    let fits = FamilyLayout { mark_store: MarkStore::Pool { pages: 4, reserved: 1 }, ..fake.layout() };
    assert_eq!(PrefixCache::<Shared>::new(fits, config(2, 0), None).unwrap().pool().capacity(), 4);
    let markless = FamilyLayout { mark_bytes: 0, ..fake.layout() };
    assert!(matches!(PrefixCache::<Shared>::new(markless, config(2, 0), None), Err(PrefixError::Layout(_))));
    let arena = Fake::new(4, 1, 3);
    assert_eq!(PrefixCache::<Shared>::new(arena.layout(), config(2, 3), None).unwrap().arena().slots(), 3);
}
