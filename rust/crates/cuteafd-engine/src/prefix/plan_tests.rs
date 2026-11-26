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
    /// Per ring: the lineage the last restore set (cold admissions reset it in `bind`).
    lineage: RefCell<Vec<Option<usize>>>,
}

impl Lagged {
    fn new(pages: usize, rings: usize, slots: usize) -> Self {
        let mut fake = Fake::new(pages, rings, slots);
        fake.rule = ReuseRule { align: 2, replay: Some(WINDOW) };
        Self { fake, align: 2, replay: WINDOW, queued: RefCell::new(Vec::new()), next: Cell::new(0),
            stuck: Cell::new(false), restored: RefCell::new(Vec::new()), lineage: RefCell::new(vec![None; rings]) }
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
    fn bind(&self, p: &mut Placement) -> Result<(), BoxError> {
        self.lineage.borrow_mut()[p.ring] = None;
        Ok(())
    }
    fn approximate_from(&self, p: &Placement) -> Result<Option<usize>, BoxError> {
        Ok(self.lineage.borrow()[p.ring])
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
        context: &RestoreContext<'_>) -> Result<(), BoxError> {
        self.restored.borrow_mut().push(saved.cloned());
        assert_eq!(context.approximate_from.is_some(), !plan.exact(), "fidelity and lineage agree");
        self.lineage.borrow_mut()[p.ring] = context.approximate_from;
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
    fn wait_capture(&self, _: CaptureTicket) -> Result<(), BoxError> {
        if self.stuck.get() { return Err("queued copies did not drain".into()); }
        self.land();
        Ok(())
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
    assert_eq!(cache.stats().approximate_replay_hits, 1);
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
fn a_third_turn_inherits_approximation_on_device_and_host() {
    for host in [false, true] {
        let family = Lagged::new(128, 6, 8);
        let mut cache = lag_cache(&family, 2, if host { 1 << 20 } else { 0 });
        let first = seq(100, 41);
        let (_, a) = lag_serve(&mut cache, &family, 0, &first, 48);
        cache.release(&family, &a.pages).unwrap();
        let mut second = first[..33].to_vec();
        second.extend(seq(900, 6));
        let (partial, b) = lag_serve(&mut cache, &family, 1, &second, 48);
        assert!(partial.source.unwrap().partial);
        cache.release(&family, &b.pages).unwrap();
        if host {
            family.fake.mem.borrow_mut().advance(10_000_000);
            cache.tick();
            for (ring, base) in [(2, 3000), (3, 4000)] {
                let (_, filler) = lag_serve(&mut cache, &family, ring, &seq(base, 41), 48);
                cache.release(&family, &filler.pages).unwrap();
                family.fake.mem.borrow_mut().advance(10_000_000);
                cache.tick();
            }
        }
        // A repeated request restores the whole second prompt and reuses its logits.
        // Mark transfer is exact to the capture; cold-prefill fidelity is not.
        let (third, _) = lag_serve(&mut cache, &family, 4, &second, 48);
        let source = third.source.unwrap();
        assert_eq!(source.host, host);
        assert!(!source.partial && source.plan.restores_mark());
        assert_eq!(source.plan.fidelity, RestoreFidelity::ApproximateReplay);
        // The second prompt's replay shared rows through 32: its snapshot is tainted from there.
        assert_eq!(source.approximate_from, Some(32));
        assert!(third.after.is_some());
        let counts = cache.stats();
        // One replay; the third turn only inherits it (counted as lineage, not a second replay).
        assert_eq!((counts.partial_hits, counts.approximate_replay_hits, counts.approximate_lineage_hits, counts.exact_hits),
            (1, 1, 1, 0));
        cache.release(&family, &third.placement.pages).unwrap();
    }
}

/// #2455 preferred the host ancestor; ranking by rows saved takes the deeper device partial.
#[test]
fn a_deeper_device_media_partial_beats_a_shallower_host_ancestor() {
    let mut fake = Fake::new(1024, 4, 8);
    fake.rule = ReuseRule::V41;
    let mut cache = cache(&fake, 1, 1 << 20);
    let original = seq(100, 1500);
    let image = media_span(ImageKey([7; 32]));
    let keys = |tokens: &[u32]| MediaKeys::new(tokens, 5000, &[image]).unwrap();
    for (ring, tokens) in [(0, &original[..512]), (1, original.as_slice())] {
        let keyed = keys(tokens);
        // Seed both snapshots cold; the deeper snapshot then evicts the ancestor.
        let mut p = cache.admit_cold(&fake, tokens.len(), tokens.len(), |pages| media_place(pages, ring)).unwrap().placement;
        let logits = fake.forward(&mut p, tokens).unwrap();
        assert!(cache.capture_media(&fake, SnapshotKind::Prompt, keyed.tokens(), keyed.spans(), &p,
            After::from_logits(&logits, true)).unwrap());
        cache.release(&fake, &p.pages).unwrap();
        fake.mem.borrow_mut().advance(10_000_000);
        cache.tick();
    }
    let mut branch = original;
    branch[1001] = 4999;
    let keyed = keys(&branch);
    // The partial diverges at 1001: sources through 1000, replay from 872 (> the 512 ancestor).
    assert_eq!(cache.peek_media(keyed.tokens(), keyed.spans(), false), 872);
    let admitted = cache.admit_native(&fake, keyed.tokens(), &branch, keyed.spans(), branch.len(), false,
        |pages| media_place(pages, 2)).unwrap();
    let source = admitted.source.unwrap();
    assert_eq!(admitted.resume, 872);
    assert!(!source.host && source.partial && !source.plan.exact());
    cache.release(&fake, &admitted.placement.pages).unwrap();
}

#[test]
fn odd_image_end_collapse_does_not_promote_or_spend_a_mark() {
    for (start, expected) in [(15, 0), (129, 0), (131, 2)] {
        let mut fake = Fake::new(512, 3, 8);
        fake.rule = ReuseRule::V41;
        let mut cache = cache(&fake, 1, 1 << 20);
        let end = start + 192;
        let first = seq(100, end + 100);
        let media = [MediaSpan { start, len: 192, key: ImageKey([7; 32]).into() }];
        let key = |tokens: &[u32]| MediaKeys::new(tokens, 5000, &media).unwrap();
        let keyed = key(&first);
        let p = media_capture(&mut cache, &fake, &keyed, &first, 0);
        cache.release(&fake, &p.pages).unwrap();
        fake.mem.borrow_mut().advance(10_000_000);
        cache.tick();
        let (_, _, filler) = serve(&mut cache, &fake, 1, &seq(3200, 32), &[], 32);
        cache.release(&fake, &filler.pages).unwrap();
        fake.mem.borrow_mut().advance(10_000_000);
        cache.tick();
        let mut branch = first;
        branch[end] = 4999;
        let keyed = key(&branch);
        let before = cache.stats();
        assert_eq!(cache.peek_media(keyed.tokens(), keyed.spans(), false), expected);
        let hit = cache.admit_native(&fake, keyed.tokens(), &branch, keyed.spans(), branch.len(), false,
            |pages| media_place(pages, 2)).unwrap();
        assert_eq!(hit.resume, expected);
        assert_eq!(hit.source.as_ref().map(|s| (s.host, s.partial)), (expected > 0).then_some((true, true)));
        let after = cache.stats();
        assert_eq!(after.host_promotions - before.host_promotions, u64::from(expected > 0));
        assert_eq!(after.fork_no_room, before.fork_no_room);
        assert_eq!(after.marks_in_use, before.marks_in_use);
        cache.release(&fake, &hit.placement.pages).unwrap();
    }
}

#[test]
fn media_ranking_uses_the_plan_that_keeps_the_whole_image() {
    let mut fake = Fake::new(1024, 3, 8);
    fake.rule = ReuseRule::V41;
    let mut cache = cache(&fake, 4, 0);
    let media = [MediaSpan { start: 501, len: 192, key: ImageKey([8; 32]).into() }];
    let key = |tokens: &[u32]| MediaKeys::new(tokens, 5000, &media).unwrap();
    let query = seq(100, 800);
    for (ring, kind, len, common) in [(0, SnapshotKind::Turn, 800, 693), (1, SnapshotKind::Prompt, 750, 694)] {
        let mut tokens = query[..len].to_vec();
        tokens[common] = 4999;
        let keyed = key(&tokens);
        let mut p = cache.admit_cold(&fake, len, len, |pages| media_place(pages, ring)).unwrap().placement;
        let logits = fake.forward(&mut p, &tokens).unwrap();
        assert!(cache.capture_media(&fake, kind, keyed.tokens(), keyed.spans(), &p, After::from_logits(&logits, true)).unwrap());
        cache.release(&fake, &p.pages).unwrap();
    }
    let keyed = key(&query);
    let hit = cache.admit_native(&fake, keyed.tokens(), &query, keyed.spans(), query.len(), false,
        |pages| media_place(pages, 2)).unwrap();
    let source = hit.source.unwrap();
    assert_eq!(source.kind, SnapshotKind::Prompt);
    assert_eq!(source.plan.lag, LaggedState { source_end: 694, replay_start: 501 });
    cache.release(&fake, &hit.placement.pages).unwrap();
}

#[test]
fn page_boundary_partial_is_exact_and_not_counted_as_replay() {
    let mut fake = Fake::new(64, 4, 8);
    fake.rule = ReuseRule::paged(ROWS);
    let mut cache = cache(&fake, 4, 0);
    let first = seq(100, 30);
    let (_, _, a) = serve(&mut cache, &fake, 0, &first, &[], 30);
    cache.release(&fake, &a.pages).unwrap();
    let mut second = first[..27].to_vec();
    second.extend(seq(900, 5));
    let b = cache.admit(&fake, &second, 40, false, |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
    let source = b.source.unwrap();
    assert!(source.partial && source.plan.exact() && !source.plan.restores_mark());
    assert_eq!((cache.stats().partial_hits, cache.stats().approximate_replay_hits), (1, 0));
    cache.release(&fake, &b.placement.pages).unwrap();
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
    let promotions = cache.stats().host_promotions;
    let (back, _) = lag_serve(&mut cache, &family, 3, &first, 16);
    assert_eq!(back.resume, first.len());
    assert_eq!(cache.stats().host_promotions, promotions + 1, "the hit came from the host tier");
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

#[test]
fn a_pending_capture_keeps_the_only_mark_and_skips_competing_capture() {
    let family = Lagged::new(64, 2, 1);
    let mut cache = lag_cache(&family, 20, 0);
    let first = seq(100, 12);
    let mut a = cache.admit(&family, &first, 16, false,
        |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 }).unwrap().placement;
    let logits = family.fake.forward(&mut a, &first).unwrap();
    let Captured::Queued(ticket) = cache.queue_capture(&family, SnapshotKind::Prompt,
        &first, &[], &a, After::from_logits(&logits, true)).unwrap() else { panic!("queued capture") };
    let second = seq(500, 12);
    let mut b = cache.admit(&family, &second, 16, false,
        |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap().placement;
    let logits = family.fake.forward(&mut b, &second).unwrap();
    assert!(matches!(cache.queue_capture(&family, SnapshotKind::Turn, &second, &[], &b,
        After::from_logits(&logits, true)).unwrap(), Captured::Skipped));
    assert_eq!(cache.stats().capture_skips, 1);
    assert_eq!(cache.arena().in_use(), 1);
    assert!(!cache.poll_capture(&family, ticket).unwrap());
    family.land();
    assert!(cache.poll_capture(&family, ticket).unwrap());
    let admitted = cache.admit(&family, &first, 16, false,
        |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
    assert_eq!(admitted.resume, first.len());
    let mut restored = admitted.placement;
    let mut longer = first;
    longer.push(900);
    family.fake.forward(&mut restored, &longer).unwrap();
    for placement in [a, b, restored] { cache.release(&family, &placement.pages).unwrap(); }
    cache.clear(&family).unwrap();
    assert_eq!(cache.pool().free(), 64);
    assert_eq!(cache.arena().in_use(), 0);
}

#[test]
fn queued_captures_publish_oldest_under_two_slot_pressure_without_losing_tickets() {
    let family = Lagged::new(96, 4, 2);
    let mut cache = lag_cache(&family, 8, 1 << 20);
    let mut tickets = Vec::new();
    let mut prompts = Vec::new();
    for i in 0..4 {
        let prompt = seq(100 + 100 * i, 13);
        let mut p = cache.admit_cold(&family, prompt.len(), 16,
            |pages| Placement { pages, ring: i as usize, len: 0, ring_from: 0 }).unwrap().placement;
        let logits = family.fake.forward(&mut p, &prompt).unwrap();
        let captured = cache.queue_capture(&family, SnapshotKind::Prompt, &prompt, &[], &p,
            After::from_logits(&logits, true)).unwrap();
        let Captured::Queued(ticket) = captured else { panic!("queued capture") };
        tickets.push(ticket);
        prompts.push((prompt, p));
        assert!(cache.arena().in_use() <= 2);
    }
    family.land();
    for ticket in tickets { assert!(cache.poll_capture(&family, ticket).unwrap()); }
    family.fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    assert_eq!(cache.arena().in_use(), 0);
    for (prompt, p) in prompts {
        cache.release(&family, &p.pages).unwrap();
        let mut next = prompt.clone(); next.push(42);
        let admitted = cache.admit(&family, &next, 16, false,
            |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 }).unwrap();
        assert_eq!(admitted.resume, prompt.len());
        let mut p = admitted.placement;
        family.fake.forward(&mut p, &next).unwrap();
        cache.release(&family, &p.pages).unwrap();
    }
    assert!(cache.stats().capture_stalls > 0);
}

// ---- Ranking by rows saved and scoped approximate lineage (#2455 follow-up, #2463) ----

/// Admit `prompt` cold-or-restored, prefill the rest, capture a prompt snapshot; drain the host.
fn lag_round(cache: &mut PrefixCache<Shared, LagMeta>, family: &Lagged, ring: usize, prompt: &[u32])
    -> (Admitted<Placement>, Placement) {
    let served = lag_serve(cache, family, ring, prompt, prompt.len() + 8);
    family.fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    cache.release(family, &served.1.pages).unwrap();
    served
}

#[test]
fn the_partial_that_leaves_fewer_rows_wins_and_an_exact_ancestor_wins_ties() {
    let (shallow, deep) = (40, 400);
    // The fixture replays WINDOW rows: a divergence at shallow + WINDOW ties the ancestor.
    for (divergence, partial) in [(shallow + WINDOW, false), (shallow + WINDOW + 2, true), (deep - 20, true)] {
        let family = Lagged::new(deep, 6, 8);
        let mut cache = lag_cache(&family, 4, 0);
        let long = seq(100, deep);
        lag_round(&mut cache, &family, 0, &long[..shallow]);
        lag_round(&mut cache, &family, 1, &long);
        let mut query = long[..divergence].to_vec();
        query.extend(seq(90_000, 7));
        let expected = if partial { divergence / 2 * 2 - WINDOW } else { shallow };
        assert_eq!(cache.peek(&query, false), expected);
        let (admitted, _) = lag_round(&mut cache, &family, 2, &query);
        let source = admitted.source.unwrap();
        assert_eq!((admitted.resume, source.partial), (expected, partial), "divergence {divergence}");
        assert_eq!(source.plan.exact(), !partial, "only the replay is approximate");
        assert_eq!(source.approximate_from, partial.then_some(divergence / 2 * 2));
    }
}

#[test]
fn the_host_tier_promotes_a_deeper_partial_over_a_shallow_device_ancestor() {
    let (shallow, deep) = (40, 400);
    let family = Lagged::new(deep * 3, 6, 8);
    let mut cache = lag_cache(&family, 1, 1 << 22);
    let long = seq(100, deep);
    // The deep prompt goes to the host tier; the shallow ancestor stays on the device.
    lag_round(&mut cache, &family, 0, &long);
    lag_round(&mut cache, &family, 1, &long[..shallow]);
    let mut query = long[..deep - 20].to_vec();
    query.extend(seq(90_000, 7));
    let promotions = cache.stats().host_promotions;
    let expected = (deep - 20) / 2 * 2 - WINDOW;
    assert_eq!(cache.peek(&query, false), expected);
    let (admitted, _) = lag_round(&mut cache, &family, 2, &query);
    let source = admitted.source.unwrap();
    assert_eq!((admitted.resume, source.host, source.partial), (expected, true, true));
    assert_eq!(cache.stats().host_promotions, promotions + 1);
}

/// A retained entry whose mark is gone (staging reaped an uncached mark) keeps its exact pages
/// under a replay rule and restores by bounded replay from just short of its frontier, instead
/// of a cold miss or a shallower ancestor.
#[test]
fn a_markless_entry_restores_by_bounded_replay_from_its_own_pages() {
    let family = Lagged::new(256, 6, 8);
    let mut cache = lag_cache(&family, 4, 0);
    let prompt = seq(100, 41);
    lag_round(&mut cache, &family, 0, &prompt);
    // Drop the entry's mark as the staging ring does for an uncached write.
    assert_eq!(cache.drop_marks_for_test(&family), 1);
    assert_eq!(cache.stats().marks_dropped, 1);
    let mut next = prompt.clone();
    next.extend(seq(2000, 9));
    assert_eq!(cache.peek(&next, false), prompt.len(), "peek is a hint; admission rechecks");
    let (admitted, _) = lag_round(&mut cache, &family, 1, &next);
    let source = admitted.source.unwrap();
    // common 40 (one short of the frontier): sources through 40, replay from 40 - WINDOW.
    assert_eq!((admitted.resume, source.partial, source.plan.exact()), (40 - WINDOW, true, false));
    assert_eq!(source.approximate_from, Some(40));
    let stats = cache.stats();
    assert_eq!((stats.markless_replays, stats.approximate_replay_hits, stats.dead_marks), (1, 1, 0));
}

/// Problem 2: a replay taints only rows from its shared frontier on. Descendants keep content
/// identities for the pages before it, so their host copies dedup with exact snapshots; a later
/// partial restore that shares only those pages is exact; an ancestor restore still inherits.
#[test]
fn approximate_lineage_is_scoped_to_its_taint_row_across_device_and_host() {
    let family = Lagged::new(256, 8, 8);
    let mut cache = lag_cache(&family, 8, 1 << 22);
    let base = seq(100, 64);
    // An exact conversation over a 64-token base (16 full pages).
    let mut first = base.clone();
    first.extend(seq(1000, 20));
    lag_round(&mut cache, &family, 0, &first);
    // Diverge at 72: the replay shares rows through 72 (aligned) and taints from there.
    let mut second = first[..72].to_vec();
    second.extend(seq(2000, 30));
    let (partial, _) = lag_round(&mut cache, &family, 1, &second);
    assert_eq!(partial.source.as_ref().unwrap().approximate_from, Some(72));
    // `partial` is the divergence question (`!restores_mark()`), not the fidelity one (`!exact()`).
    let divergent = partial.source.as_ref().unwrap();
    assert!(divergent.partial && !divergent.plan.restores_mark());
    assert_eq!(divergent.partial, !divergent.plan.restores_mark());
    // Its next turn is an ancestor restore: inherited lineage, labelled, counted as lineage.
    let mut third = second.clone();
    third.extend(seq(3000, 10));
    let before = cache.stats().host.unwrap();
    let (turn, _) = lag_round(&mut cache, &family, 2, &third);
    let source = turn.source.unwrap();
    assert!(source.plan.restores_mark() && !source.plan.exact());
    // Not exact, yet not a divergent match: a fidelity label must not make it `partial`.
    assert_eq!(source.partial, !source.plan.restores_mark());
    assert!(!source.partial);
    assert_eq!(source.approximate_from, Some(72));
    let after = cache.stats().host.unwrap();
    // The ancestor fork shares the second snapshot's device pages: its 25 full pages are host
    // shared, only the three pages the new suffix completed are copied.
    assert_eq!((after.pages_shared - before.pages_shared, after.pages_copied - before.pages_copied), (25, 3));
    // Host identities: content for the 18 pages wholly before row 72, allocation after it.
    let pages: Vec<u32> = (0..28).collect();
    let ids = cache.identities(&family, &third, &[], &pages, Some(72));
    let content = super::chain::page_chain(&third, ROWS);
    for (i, id) in ids.iter().enumerate() {
        if i < 72 / ROWS { assert_eq!(*id, super::chain::content_id(0, content[i]), "page {i}"); }
        else { assert!(id.compressor & super::chain::CONTENT_CLASS == 0, "page {i} is tainted"); }
    }
    // The exact lineage names every full page by content; 73 rounds down to 18 clean pages.
    assert!(cache.identities(&family, &third, &[], &pages, None).iter()
        .all(|id| id.compressor & super::chain::CONTENT_CLASS != 0));
    assert_eq!(cache.identities(&family, &third, &[], &pages, Some(73)).iter()
        .filter(|id| id.compressor & super::chain::CONTENT_CLASS != 0).count(), 18);
    // A request diverging inside the clean base shares only exact pages: no inherited lineage.
    let mut fourth = third[..60].to_vec();
    fourth.extend(seq(4000, 30));
    let (clean, _) = lag_round(&mut cache, &family, 3, &fourth);
    let source = clean.source.unwrap();
    assert!(source.partial);
    assert_eq!(source.approximate_from, Some(60), "its own replay, not the lineage at 72");
    let stats = cache.stats();
    assert_eq!((stats.approximate_replay_hits, stats.approximate_lineage_hits, stats.partial_hits), (2, 1, 2));
    // Diverging past the taint row inherits it (min of the inherited and its own taint row).
    let mut fifth = third[..90].to_vec();
    fifth.extend(seq(5000, 30));
    let (tainted, _) = lag_round(&mut cache, &family, 4, &fifth);
    assert_eq!(tainted.source.unwrap().approximate_from, Some(72));
    // An exact restore of the untouched first conversation stays exact.
    let (exact, _) = lag_round(&mut cache, &family, 5, &first);
    let source = exact.source.unwrap();
    assert!(source.plan.exact() && source.approximate_from.is_none());
}

/// Lineage survives the host tier: a promoted tainted snapshot keeps its taint row, and its
/// clean pages are restored under content identities that dedup with exact snapshots.
#[test]
fn scoped_lineage_survives_host_promotion() {
    let family = Lagged::new(256, 8, 8);
    let mut cache = lag_cache(&family, 1, 1 << 22);
    let first = seq(100, 84);
    lag_round(&mut cache, &family, 0, &first);
    let mut second = first[..72].to_vec();
    second.extend(seq(2000, 30));
    lag_round(&mut cache, &family, 1, &second);
    // Push the tainted snapshot off the device.
    lag_round(&mut cache, &family, 2, &seq(7000, 40));
    let promotions = cache.stats().host_promotions;
    let mut next = second.clone();
    next.extend(seq(3000, 10));
    let (back, _) = lag_round(&mut cache, &family, 3, &next);
    let source = back.source.unwrap();
    assert!(source.host && source.plan.restores_mark());
    assert_eq!(cache.stats().host_promotions, promotions + 1);
    assert_eq!(source.approximate_from, Some(72));
    assert_eq!(source.plan.fidelity, RestoreFidelity::ApproximateReplay);
}

/// A cold admission on a ring that last served a tainted request starts exact.
#[test]
fn a_cold_admission_clears_ring_lineage() {
    let family = Lagged::new(128, 2, 8);
    let mut cache = lag_cache(&family, 4, 0);
    let first = seq(100, 41);
    lag_round(&mut cache, &family, 0, &first);
    let mut second = first[..33].to_vec();
    second.extend(seq(900, 6));
    lag_round(&mut cache, &family, 1, &second);
    assert_eq!(family.lineage.borrow()[1], Some(32));
    let (cold, _) = lag_round(&mut cache, &family, 1, &seq(5000, 20));
    assert!(cold.source.is_none());
    assert_eq!(family.lineage.borrow()[1], None);
    let (exact, _) = lag_round(&mut cache, &family, 0, &seq(5000, 20));
    assert!(exact.source.unwrap().approximate_from.is_none());
}
