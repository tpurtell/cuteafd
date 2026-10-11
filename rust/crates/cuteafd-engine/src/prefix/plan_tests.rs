// Restore plans, snapshot metadata, queued captures and eviction by freed pages, over the same
// fake device as `tests.rs` (included there). `Lagged` wraps a `Fake` whose pages are the whole
// global state and whose ring is positional: a partial restore shares pages through an aligned
// `source_end` but rebuilds the ring from `replay_start`, as V4.1 does; its metadata is the
// frontier it was captured at, and its captures are queued until `land` runs.

/// What a `Lagged` snapshot records besides its bytes.
#[derive(Clone, Debug, Default, PartialEq)]
struct LagMeta {
    frontier: usize,
    tag: u64,
}

struct Lagged {
    fake: Fake,
    /// Source alignment and replay rows of a partial plan.
    align: usize,
    replay: usize,
    /// Queued captures: (ticket, slot, ring, len). Copies are applied by `land`.
    queued: RefCell<Vec<(CaptureTicket, MarkSlot, usize, usize)>>,
    next: Cell<u32>,
    /// Make `abort_capture` fail (copies that never drained).
    stuck: Cell<bool>,
    restored: RefCell<Vec<Option<LagMeta>>>,
}

impl Lagged {
    fn new(pages: usize, rings: usize, slots: usize) -> Self {
        let mut fake = Fake::new(pages, rings, slots);
        fake.rule = ReuseRule { align: 2, replay: Some(WINDOW) };
        Self { fake, align: 2, replay: WINDOW, queued: RefCell::new(Vec::new()), next: Cell::new(0),
            stuck: Cell::new(false), restored: RefCell::new(Vec::new()) }
    }
    /// Run every queued mark copy (their stream drained).
    fn land(&self) {
        for (_, slot, ring, len) in self.queued.borrow_mut().drain(..) {
            let placement = Placement { pages: Vec::new(), ring, len, ring_from: 0 };
            PrefixFamily::capture(&self.fake, slot, &placement, len).unwrap();
        }
        self.fake.flush();
    }
}

impl PrefixFamily<LagMeta> for Lagged {
    type Placement = Placement;
    fn layout(&self) -> FamilyLayout { self.fake.layout() }
    fn eviction(&self) -> Eviction { Eviction::FreesPages }
    fn capture_meta(&self, placement: &Placement, len: usize) -> Result<LagMeta, BoxError> {
        Ok(LagMeta { frontier: len, tag: placement.ring as u64 * 1000 + len as u64 })
    }
    fn plan_restore(&self, hit: RestoreCandidate<'_>) -> RestorePlan {
        if hit.common == hit.snapshot_end { return RestorePlan::under(self.fake.rule, hit); }
        let source_end = hit.common / self.align * self.align;
        let replay_start = source_end.saturating_sub(self.replay);
        RestorePlan { snapshot_end: hit.snapshot_end, target_end: hit.target_end,
            lag: LaggedState { source_end: if replay_start == 0 { 0 } else { source_end }, replay_start },
            fidelity: RestoreFidelity::ApproximateReplay }
    }
    fn restore_with(&self, mark: Option<&Mark>, saved: Option<&LagMeta>, p: &mut Placement, plan: &RestorePlan,
        _: &RestoreContext<'_>) -> Result<(), BoxError> {
        self.restored.borrow_mut().push(saved.cloned());
        match mark {
            Some(Mark::Slot(slot)) => PrefixFamily::restore(&self.fake, Some(*slot), p, plan.resume()),
            None => {
                // Rows below `source_end` are shared and exact; the ring restarts at the
                // replay start, whose rows the caller prefills again.
                p.len = plan.lag.replay_start;
                p.ring_from = plan.lag.replay_start;
                Ok(())
            }
            Some(Mark::Pages(_)) => Err("arena family".into()),
        }
    }
    fn queue_capture(&self, slot: MarkSlot, p: &Placement, len: usize, tail: Option<TailCopy>)
        -> Result<Option<CaptureTicket>, BoxError> {
        if let Some(tail) = tail { PrefixFamily::copy_rows(&self.fake, tail)?; }
        let ticket = CaptureTicket(self.next.get());
        self.next.set(self.next.get() + 1);
        self.queued.borrow_mut().push((ticket, slot, p.ring, len));
        Ok(Some(ticket))
    }
    fn capture_ready(&self, ticket: CaptureTicket) -> Result<bool, BoxError> {
        Ok(!self.queued.borrow().iter().any(|&(t, ..)| t == ticket))
    }
    fn abort_capture(&self, ticket: CaptureTicket) -> Result<(), BoxError> {
        if self.stuck.get() { return Err("queued copies did not drain".into()); }
        self.queued.borrow_mut().retain(|&(t, ..)| t != ticket);
        Ok(())
    }
    fn pages<'p>(&self, p: &'p Placement) -> &'p [u32] { &p.pages }
    fn commit_point(&self, p: &Placement) -> usize { p.len }
    fn capture(&self, slot: MarkSlot, p: &Placement, len: usize) -> Result<(), BoxError> {
        PrefixFamily::capture(&self.fake, slot, p, len)
    }
    fn restore(&self, mark: Option<MarkSlot>, p: &mut Placement, len: usize) -> Result<(), BoxError> {
        PrefixFamily::restore(&self.fake, mark, p, len)
    }
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> { PrefixFamily::copy_rows(&self.fake, copy) }
    fn drain(&self) -> Result<(), BoxError> { PrefixFamily::drain(&self.fake) }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> { PrefixFamily::page_segments(&self.fake, page) }
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> { PrefixFamily::mark_segments(&self.fake, slot) }
}

fn lag_cache(family: &Lagged, entries: usize, host_bytes: u64) -> PrefixCache<Shared, LagMeta> {
    let host = HostConfig { bytes: host_bytes, chunk_bytes: 1 << 16, min_tokens: 1, ..HostConfig::default() };
    PrefixCache::new(family.layout(), config(entries, family.fake.slots), Some((host, Shared(family.fake.mem.clone()))))
        .unwrap()
}

/// Admit `prompt`, prefill what the restore left, capture a prompt snapshot synchronously.
fn lag_serve(cache: &mut PrefixCache<Shared, LagMeta>, family: &Lagged, ring: usize, prompt: &[u32], capacity: usize)
    -> (Admitted<Placement>, Placement) {
    let admitted = cache.admit(family, prompt, capacity, false,
        |pages| Placement { pages, ring, len: 0, ring_from: 0 }).unwrap();
    let mut placement = admitted.placement.clone();
    if admitted.resume < prompt.len() {
        let logits = family.fake.forward(&mut placement, prompt).unwrap();
        cache.capture(family, SnapshotKind::Prompt, prompt, &placement, After::from_logits(&logits, true)).unwrap();
    }
    (Admitted { placement: admitted.placement, resume: admitted.resume, after: admitted.after, source: admitted.source },
        placement)
}

impl Clone for Admitted<Placement> {
    fn clone(&self) -> Self {
        Self { placement: self.placement.clone(), resume: self.resume, after: self.after.clone(), source: self.source.clone() }
    }
}

#[test]
fn a_lagged_partial_restore_shares_pages_through_source_end_and_replays_from_replay_start() {
    let family = Lagged::new(64, 4, 8);
    let mut cache = lag_cache(&family, 4, 0);
    let prompt = seq(100, 41);
    let (_, a) = lag_serve(&mut cache, &family, 0, &prompt, 48);
    // Shares 33 tokens: sources through 32 (aligned), ring from 24 (32 - WINDOW).
    let mut other = prompt[..33].to_vec();
    other.extend(seq(900, 6));
    let admitted = cache.admit(&family, &other, 48, false, |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 })
        .unwrap();
    let source = admitted.source.clone().unwrap();
    assert!(source.partial);
    assert_eq!(source.plan.lag, LaggedState { source_end: 32, replay_start: 24 });
    assert_eq!(source.plan.fidelity, RestoreFidelity::ApproximateReplay);
    assert_eq!(admitted.resume, 24);
    // Pages through source_end are the snapshot's own (shared, not recomputed).
    assert_eq!(&admitted.placement.pages[..32 / ROWS], &a.pages[..32 / ROWS]);
    assert_eq!(family.restored.borrow().last(), Some(&None), "a partial restore gets no metadata");
    let mut b = admitted.placement;
    family.fake.forward(&mut b, &other).unwrap();
    // The writer's rows are untouched: its exact restore still checks every row and ring entry.
    let (exact, _) = lag_serve(&mut cache, &family, 2, &prompt, 48);
    assert_eq!(exact.resume, prompt.len());
    assert_eq!(exact.source.unwrap().plan.fidelity, RestoreFidelity::Exact);
    assert_eq!(family.restored.borrow().last().cloned().flatten(), Some(LagMeta { frontier: 41, tag: 41 }));
    for p in [a, b, exact.placement] { cache.release(&family, &p.pages).unwrap(); }
    cache.clear(&family).unwrap();
    assert_eq!(cache.pool().free(), 64);
}

#[test]
fn a_plan_that_claims_exactness_for_a_partial_match_is_refused() {
    struct Liar(Lagged);
    impl PrefixFamily<LagMeta> for Liar {
        type Placement = Placement;
        fn layout(&self) -> FamilyLayout { self.0.layout() }
        fn plan_restore(&self, hit: RestoreCandidate<'_>) -> RestorePlan {
            RestorePlan { snapshot_end: hit.snapshot_end, target_end: hit.target_end,
                lag: LaggedState { source_end: hit.common, replay_start: hit.common }, fidelity: RestoreFidelity::Exact }
        }
        fn pages<'p>(&self, p: &'p Placement) -> &'p [u32] { &p.pages }
        fn commit_point(&self, p: &Placement) -> usize { p.len }
        fn capture(&self, s: MarkSlot, p: &Placement, l: usize) -> Result<(), BoxError> { self.0.capture(s, p, l) }
        fn restore(&self, m: Option<MarkSlot>, p: &mut Placement, l: usize) -> Result<(), BoxError> {
            PrefixFamily::<LagMeta>::restore(&self.0, m, p, l)
        }
        fn copy_rows(&self, c: TailCopy) -> Result<(), BoxError> { self.0.copy_rows(c) }
        fn drain(&self) -> Result<(), BoxError> { PrefixFamily::<LagMeta>::drain(&self.0) }
        fn page_segments(&self, p: u32) -> Vec<DeviceRange> { self.0.page_segments(p) }
        fn mark_segments(&self, s: MarkSlot) -> Vec<DeviceRange> { self.0.mark_segments(s) }
    }
    let family = Liar(Lagged::new(64, 4, 8));
    let mut cache = PrefixCache::<Shared, LagMeta>::new(family.layout(), config(4, 8), None).unwrap();
    let prompt = seq(100, 41);
    let mut a = cache.admit(&family, &prompt, 48, false, |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 })
        .unwrap().placement;
    let logits = family.0.fake.forward(&mut a, &prompt).unwrap();
    cache.capture(&family, SnapshotKind::Prompt, &prompt, &a, After::from_logits(&logits, true)).unwrap();
    let mut other = prompt[..33].to_vec();
    other.extend(seq(900, 6));
    let free = cache.pool().free();
    let admitted = cache.admit(&family, &other, 48, false, |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 })
        .unwrap();
    assert_eq!((admitted.resume, admitted.source.is_none()), (0, true), "refused plan falls back to cold");
    assert_eq!(cache.stats().restore_failures, 1);
    cache.release(&family, &admitted.placement.pages).unwrap();
    assert_eq!(cache.pool().free(), free);
    cache.release(&family, &a.pages).unwrap();
}

#[test]
fn snapshot_metadata_survives_the_host_tier() {
    let family = Lagged::new(24, 4, 4);
    let mut cache = lag_cache(&family, 2, 1 << 20);
    let first = seq(100, 13);
    let (_, a) = lag_serve(&mut cache, &family, 0, &first, 16);
    cache.release(&family, &a.pages).unwrap();
    family.fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    // Push the first snapshot off the device (two more prompts, bank of two).
    for (i, base) in [300u32, 500].into_iter().enumerate() {
        let (_, p) = lag_serve(&mut cache, &family, 1 + i, &seq(base, 13), 16);
        cache.release(&family, &p.pages).unwrap();
        family.fake.mem.borrow_mut().advance(10_000_000);
        cache.tick();
    }
    let promotions = cache.stats().promotions;
    let (back, _) = lag_serve(&mut cache, &family, 3, &first, 16);
    assert_eq!(back.resume, first.len());
    assert_eq!(cache.stats().promotions, promotions + 1, "the hit came from the host tier");
    assert_eq!(family.restored.borrow().last().cloned().flatten(), Some(LagMeta { frontier: 13, tag: 13 }));
    let mut p = back.placement;
    let mut longer = first.clone();
    longer.push(9);
    family.fake.forward(&mut p, &longer).unwrap();
    cache.release(&family, &p.pages).unwrap();
}

#[test]
fn queued_captures_publish_after_their_copies_land_and_own_their_storage_meanwhile() {
    let family = Lagged::new(64, 4, 8);
    let mut cache = lag_cache(&family, 4, 0);
    let prompt = seq(100, 21);
    let mut p = cache.admit(&family, &prompt, 24, false, |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 })
        .unwrap().placement;
    let logits = family.fake.forward(&mut p, &prompt).unwrap();
    let Captured::Queued(ticket) = cache.queue_capture(&family, SnapshotKind::Turn, &prompt, &[], &p,
        After::from_logits(&logits, true)).unwrap() else { panic!("the family queues") };
    // Not visible yet; its pages and mark are held.
    assert_eq!(cache.peek(&prompt, false), 0);
    let held = cache.pool().free();
    assert!(!cache.poll_capture(&family, ticket).unwrap());
    cache.release(&family, &p.pages).unwrap();
    assert!(cache.pool().free() < 64, "a pending capture keeps its pages");
    assert!(cache.pool().free() >= held);
    family.land();
    assert!(cache.poll_capture(&family, ticket).unwrap());
    assert_eq!(cache.peek(&prompt, false), prompt.len());
    assert_eq!(cache.stats().captures_turn, 1);
    let (restored, _) = lag_serve(&mut cache, &family, 1, &prompt, 24);
    assert_eq!(restored.resume, prompt.len());
    let mut q = restored.placement;
    let mut longer = prompt.clone();
    longer.push(3);
    family.fake.forward(&mut q, &longer).unwrap();
    cache.release(&family, &q.pages).unwrap();
    cache.clear(&family).unwrap();
    assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
}

#[test]
fn aborted_captures_return_their_storage_and_undrained_ones_are_quarantined() {
    let family = Lagged::new(32, 4, 8);
    let mut cache = lag_cache(&family, 4, 0);
    let prompt = seq(100, 21);
    let mut p = cache.admit(&family, &prompt, 24, false, |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 })
        .unwrap().placement;
    family.fake.forward(&mut p, &prompt).unwrap();
    let free_with_request = cache.pool().free();
    let Captured::Queued(ticket) = cache.queue_capture(&family, SnapshotKind::Turn, &prompt, &[], &p, After::default())
        .unwrap() else { panic!("queued") };
    cache.abort_capture(&family, ticket).unwrap();
    assert_eq!((cache.pool().free(), cache.arena().in_use()), (free_with_request, 0));
    assert_eq!(cache.stats().captures_aborted, 1);
    // A capture whose copies never drained keeps its storage forever.
    family.stuck.set(true);
    let Captured::Queued(ticket) = cache.queue_capture(&family, SnapshotKind::Turn, &prompt, &[], &p, After::default())
        .unwrap() else { panic!("queued") };
    let (free, marks) = (cache.pool().free(), cache.arena().in_use());
    cache.abort_capture(&family, ticket).unwrap();
    assert_eq!((cache.pool().free(), cache.arena().in_use(), cache.stats().quarantined), (free, marks, 1));
    assert!(cache.poll_capture(&family, ticket).is_err(), "a quarantined capture is gone");
    cache.release(&family, &p.pages).unwrap();
}

#[test]
fn eviction_by_freed_pages_keeps_snapshots_whose_pages_running_requests_hold() {
    // 16 pages of 4 rows. A runs a conversation whose prompt snapshot sits at a page boundary
    // (20 tokens): every page of it is A's own, so evicting it frees nothing.
    let family = Lagged::new(16, 4, 8);
    let mut cache = lag_cache(&family, 4, 0);
    let first = seq(100, 20);
    let (_, a) = lag_serve(&mut cache, &family, 0, &first, 32);
    // An unrelated finished conversation, more recent: its snapshot alone holds its pages.
    let (_, b) = lag_serve(&mut cache, &family, 1, &seq(500, 9), 9);
    cache.release(&family, &b.pages).unwrap();
    assert_eq!(cache.stats().entries_prompt, 2);
    // Least recently used would evict A's snapshot first (older) and free nothing; eviction by
    // freed pages takes B's and keeps the running conversation's prefix reusable.
    let free = cache.pool().free();
    assert!(cache.make_room(&family, free + 1, None).unwrap());
    assert_eq!(cache.peek(&first, false), first.len(), "the running conversation's snapshot stays");
    assert_eq!(cache.stats().entries_prompt, 1);
    // With nothing left that frees a page, the pool stays short and the snapshot stays.
    assert!(!cache.make_room(&family, 16, None).unwrap());
    assert_eq!(cache.stats().entries_prompt, 1);
    assert!(cache.stats().eviction_skips > 0);
    cache.release(&family, &a.pages).unwrap();
    assert!(cache.make_room(&family, 16, None).unwrap(), "released, the snapshot frees its pages");
}
