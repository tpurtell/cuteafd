//! Full-context draft pages share one token-budget-sized pool. GPU owners must retire
//! readers, writers and tail copies before committing, rolling back or releasing a lease.
use super::{PoolExhausted, RefPagePool, TailCopy};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

static NEXT_POOL: AtomicU64 = AtomicU64::new(1);

fn pool_id() -> Result<u64> {
    NEXT_POOL.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| DraftPageError::Geometry)
}

pub const DRAFT_PAGE_ROWS: usize = 64;

#[derive(Debug, Error)]
pub enum DraftPageError {
    #[error("invalid draft page geometry")]
    Geometry,
    #[error("draft slot is occupied or outside the admitted layout")]
    Slot,
    #[error("stale draft ownership or reservation")]
    Stale,
    #[error("draft append differs from the committed frontier or context extent")]
    Frontier,
    #[error("draft snapshot is incompatible with the requested frontier")]
    Snapshot,
    #[error(transparent)]
    Exhausted(#[from] PoolExhausted),
}

type Result<T> = std::result::Result<T, DraftPageError>;

/// A request binding cannot survive slot reuse, even when its physical pages survive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftBinding {
    pool: u64,
    slot: usize,
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftReservation {
    binding: DraftBinding,
    serial: u64,
    start: usize,
    end: usize,
}

impl DraftReservation {
    pub fn start(self) -> usize { self.start }
    pub fn end(self) -> usize { self.end }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DraftSnapshot { pool: u64, serial: u64 }

/// A page identity is meaningful only in its pool/layout namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftPageId {
    pub index: u32,
    pub generation: u32,
}

/// Plain metadata for a typed snapshot attachment. Full cold-prefill compatibility
/// additionally depends on the shared scheduler's checkpoint and stage-state predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftPageAttachment {
    pub snapshot: DraftSnapshot,
    pub layout: u64,
    pub origin: usize,
    pub frontier: usize,
    pub pages: Vec<DraftPageId>,
}

struct Sequence {
    binding: DraftBinding,
    origin: usize,
    end: usize,
    pages: Vec<u32>,
    pending: Option<DraftReservation>,
    credit: usize,
}

struct Snapshot {
    revocable: bool,
    pins: usize,
    used: u64,
    origin: usize,
    end: usize,
    pages: Vec<u32>,
}

/// Physical page IDs address the corresponding segment in every full draft layer.
/// Windowed layer rings do not allocate from this pool. Partial tails belong to one
/// writer: capture and restore eagerly copy them using the existing prefix allocator.
/// No operation allocates storage beyond ceil(target_tokens / 64) pages.
pub struct DraftPagePool {
    id: u64,
    pool: RefPagePool,
    slots: Vec<Option<Sequence>>,
    generations: Vec<u64>,
    snapshots: BTreeMap<DraftSnapshot, Snapshot>,
    serial: u64,
    context: usize,
    layout: u64,
}

impl DraftPagePool {
    pub fn new(target_tokens: usize, slots: usize, context: usize, layout: u64) -> Result<Self> {
        let pages = target_tokens.div_ceil(DRAFT_PAGE_ROWS);
        if target_tokens == 0 || slots == 0 || context == 0 || pages > i32::MAX as usize {
            return Err(DraftPageError::Geometry);
        }
        let id = pool_id()?;
        Ok(Self { id, pool: RefPagePool::new(pages, DRAFT_PAGE_ROWS),
            slots: (0..slots).map(|_| None).collect(), generations: vec![0; slots],
            snapshots: BTreeMap::new(), serial: 0, context, layout })
    }

    pub fn capacity(&self) -> usize { self.pool.capacity() }
    pub fn free(&self) -> usize { self.pool.free() }

    fn vacant(&self, slot: usize) -> Result<()> {
        if self.slots.get(slot).is_some_and(Option::is_none) { Ok(()) }
        else { Err(DraftPageError::Slot) }
    }

    fn sequence(&self, binding: DraftBinding) -> Result<&Sequence> {
        self.slots.get(binding.slot).and_then(Option::as_ref)
            .filter(|s| s.binding == binding).ok_or(DraftPageError::Stale)
    }

    fn next_serial(&mut self) -> Result<u64> {
        self.serial = self.serial.checked_add(1).ok_or(DraftPageError::Stale)?;
        Ok(self.serial)
    }

    fn install(&mut self, slot: usize, origin: usize, end: usize, pages: Vec<u32>) -> DraftBinding {
        // Callers preflight generation overflow before acquiring page references.
        self.generations[slot] += 1;
        let binding = DraftBinding { pool: self.id, slot, generation: self.generations[slot] };
        self.slots[slot] = Some(Sequence { binding, origin, end, pages, pending: None, credit: 0 });
        binding
    }

    fn outstanding_credit(&self) -> usize {
        self.slots.iter().flatten().map(|s| s.credit.saturating_sub(s.pages.len())).sum()
    }

    fn make_room(&mut self, needed: usize, keep: Option<DraftSnapshot>) -> Result<()> {
        while self.free() < needed {
            let victim = self.snapshots.iter().filter(|(id, s)| Some(**id) != keep && s.revocable && s.pins == 0)
                .min_by_key(|(_, s)| s.used).map(|(&id, _)| id);
            let Some(victim) = victim else {
                return Err(PoolExhausted { needed, free: self.free(), capacity: self.capacity() }.into());
            };
            self.release_snapshot(victim)?;
        }
        Ok(())
    }

    /// Reserve the admitted request's entire growth plus a partial page and a proposal
    /// block inside Q. Producers consume this credit; retained snapshots cannot borrow it.
    pub fn admit(&mut self, binding: DraftBinding, end: usize, block: usize) -> Result<()> {
        let s = self.sequence(binding)?;
        if end < s.end || end > self.context || block == 0 { return Err(DraftPageError::Frontier); }
        let credit = (end - s.origin).div_ceil(DRAFT_PAGE_ROWS)
            .checked_add(block.div_ceil(DRAFT_PAGE_ROWS) + 1).ok_or(DraftPageError::Geometry)?;
        let needed = self.outstanding_credit() - s.credit.saturating_sub(s.pages.len())
            + credit.saturating_sub(s.pages.len());
        if let Err(error) = self.make_room(needed, None) {
            return match error {
                DraftPageError::Exhausted(_) => Err(PoolExhausted {
                    needed: credit, free: self.free().saturating_sub(needed.saturating_sub(credit)),
                    capacity: self.capacity(),
                }.into()),
                error => Err(error),
            };
        }
        self.slots[binding.slot].as_mut().expect("checked").credit = credit;
        Ok(())
    }

    /// The caller has retired capture/host-store readers. Only these attachments may
    /// be revoked automatically when a live admission needs their pages.
    pub fn retain_snapshot(&mut self, snapshot: DraftSnapshot) -> Result<()> {
        self.snapshots.get_mut(&snapshot).ok_or(DraftPageError::Snapshot)?.revocable = true;
        Ok(())
    }

    /// Pin before enqueueing a restore or host copy; unpin only after its event retires.
    pub fn pin_snapshot(&mut self, snapshot: DraftSnapshot) -> Result<()> {
        let used = self.next_serial()?;
        let s = self.snapshots.get_mut(&snapshot).ok_or(DraftPageError::Snapshot)?;
        s.pins = s.pins.checked_add(1).ok_or(DraftPageError::Geometry)?;
        s.used = used;
        Ok(())
    }

    pub fn snapshot_revocable(&self, snapshot: DraftSnapshot) -> bool {
        self.snapshots.get(&snapshot).is_some_and(|s| s.revocable && s.pins == 0)
    }

    pub fn unpin_snapshot(&mut self, snapshot: DraftSnapshot) -> Result<()> {
        let s = self.snapshots.get_mut(&snapshot).ok_or(DraftPageError::Snapshot)?;
        s.pins = s.pins.checked_sub(1).ok_or(DraftPageError::Snapshot)?;
        Ok(())
    }

    /// What `binding`'s live reservation holds, for the KV reservation policy: its origin, the
    /// headroom `admit` adds for a proposal `block`, and the pages its credit holds.
    pub fn reservation(&self, binding: DraftBinding, block: usize) -> Result<crate::reserve::DraftDemand> {
        let s = self.sequence(binding)?;
        Ok(crate::reserve::DraftDemand { origin: s.origin, headroom: block.div_ceil(DRAFT_PAGE_ROWS) + 1,
            held: s.credit.max(s.pages.len()) })
    }

    /// Rows one live reservation can reach with proposal `block` ([`crate::reserve::draft_lifetime_rows`]).
    pub fn lifetime_rows(&self, block: usize) -> usize { crate::reserve::draft_lifetime_rows(self.capacity(), block) }

    /// Pages a live credit can still claim: free pages and the pages revoking every unpinned
    /// revocable snapshot would actually free (each of their references is such a snapshot's;
    /// a page a live sequence or a pinned snapshot shares stays), less every live credit not
    /// yet materialized.
    pub fn reservable(&self) -> usize {
        let mut held: BTreeMap<u32, u32> = BTreeMap::new();
        for s in self.snapshots.values().filter(|s| s.revocable && s.pins == 0) {
            for &page in &s.pages { *held.entry(page).or_default() += 1; }
        }
        let freed = held.into_iter().filter(|&(page, n)| self.pool.refs(page) == n).count();
        (self.free() + freed).saturating_sub(self.outstanding_credit())
    }

    /// Bind cold(P) without allocating a page or exposing the previous occupant's bytes.
    pub fn bind(&mut self, slot: usize, origin: usize) -> Result<DraftBinding> {
        self.vacant(slot)?;
        if origin > self.context { return Err(DraftPageError::Frontier); }
        self.generations[slot].checked_add(1).ok_or(DraftPageError::Stale)?;
        Ok(self.install(slot, origin, origin, Vec::new()))
    }

    pub fn frontier(&self, binding: DraftBinding) -> Result<(usize, usize)> {
        let s = self.sequence(binding)?;
        Ok((s.origin, s.end))
    }

    /// Reserve all producer destinations before enqueueing any layer's writes.
    pub fn reserve(&mut self, binding: DraftBinding, start: usize, rows: usize) -> Result<DraftReservation> {
        let s = self.sequence(binding)?;
        let end = start.checked_add(rows).ok_or(DraftPageError::Frontier)?;
        if s.pending.is_some() || start != s.end || rows == 0 || end > self.context {
            return Err(DraftPageError::Frontier);
        }
        let total = (end - s.origin).div_ceil(DRAFT_PAGE_ROWS);
        let previous = s.pages.len();
        if s.credit > 0 && total > s.credit { return Err(DraftPageError::Frontier); }
        let serial = self.next_serial()?;
        let outstanding = self.outstanding_credit();
        let own = self.sequence(binding)?.credit.saturating_sub(previous);
        self.make_room(outstanding + (total - previous).saturating_sub(own), None)?;
        let fresh = self.pool.alloc(total - previous)?;
        let reservation = DraftReservation { binding, serial, start, end };
        let s = self.slots[binding.slot].as_mut().expect("validated binding");
        s.pages.extend(fresh);
        s.pending = Some(reservation);
        Ok(reservation)
    }

    fn reserved(&self, reservation: DraftReservation) -> Result<&Sequence> {
        self.sequence(reservation.binding)?.pending.filter(|r| *r == reservation)
            .ok_or(DraftPageError::Stale)?;
        self.sequence(reservation.binding)
    }

    /// Absolute positions drive RoPE; destinations are origin-relative physical rows.
    pub fn destinations(&self, reservation: DraftReservation) -> Result<Vec<i64>> {
        let s = self.reserved(reservation)?;
        Ok((reservation.start..reservation.end).map(|position| {
            let relative = position - s.origin;
            i64::from(s.pages[relative / DRAFT_PAGE_ROWS]) * DRAFT_PAGE_ROWS as i64
                + (relative % DRAFT_PAGE_ROWS) as i64
        }).collect())
    }

    /// Read-only attention view of the committed prefix, without creating a producer lease.
    pub fn committed_page_table(&self, binding: DraftBinding) -> Result<Vec<i32>> {
        let sequence = self.sequence(binding)?;
        if sequence.pending.is_some() { return Err(DraftPageError::Frontier); }
        Ok(sequence.pages.iter().map(|&page| page as i32).collect())
    }

    /// Includes the provisional suffix. The caller bounds attention with the committed
    /// frontier and query offsets; page membership does not make proposal rows committed.
    pub fn page_table(&self, reservation: DraftReservation) -> Result<Vec<i32>> {
        Ok(self.reserved(reservation)?.pages.iter().map(|&p| p as i32).collect())
    }

    /// Publish only after accepted rows have their canonical target-derived KV and
    /// every queued user of the rejected suffix has retired. accepted=0 is rollback.
    pub fn commit(&mut self, reservation: DraftReservation, accepted: usize) -> Result<()> {
        self.reserved(reservation)?;
        if accepted > reservation.end - reservation.start { return Err(DraftPageError::Frontier); }
        let s = self.slots[reservation.binding.slot].as_mut().expect("validated binding");
        s.end = reservation.start + accepted;
        let retained = (s.end - s.origin).div_ceil(DRAFT_PAGE_ROWS);
        let released = s.pages.split_off(retained);
        s.pending = None;
        self.pool.release(&released);
        Ok(())
    }

    /// Retain a committed prefix. Run the returned tail copy on every full layer,
    /// and complete its owner events before publishing this snapshot to lookup.
    pub fn capture(&mut self, binding: DraftBinding) -> Result<(DraftSnapshot, Vec<TailCopy>)> {
        let (_, frontier) = self.frontier(binding)?;
        self.capture_writable(binding, frontier)
    }

    /// `write_frontier` is absolute, as for the target cache. Every retained page
    /// overlapping writable rows is copied, even when the page is completely filled.
    pub fn capture_writable(&mut self, binding: DraftBinding, write_frontier: usize)
        -> Result<(DraftSnapshot, Vec<TailCopy>)> {
        let s = self.sequence(binding)?;
        if s.pending.is_some() { return Err(DraftPageError::Frontier); }
        let (origin, end, source) = (s.origin, s.end, s.pages.clone());
        if !(origin..=end).contains(&write_frontier) { return Err(DraftPageError::Frontier); }
        let id = DraftSnapshot { pool: self.id, serial: self.next_serial()? };
        let len = end - origin;
        let need = if len == 0 { 0 } else { source.len() - (write_frontier - origin) / DRAFT_PAGE_ROWS };
        self.make_room(need + self.outstanding_credit(), None)?;
        let (pages, copy) = if len == 0 { (Vec::new(), Vec::new()) } else {
            self.pool.fork_writable(&source, len, source.len(), write_frontier - origin)?
        };
        self.snapshots.insert(id, Snapshot { origin, end, pages, revocable: false, pins: 0, used: id.serial });
        Ok((id, copy))
    }

    pub fn attachment(&self, snapshot: DraftSnapshot) -> Result<DraftPageAttachment> {
        let s = self.snapshots.get(&snapshot).ok_or(DraftPageError::Snapshot)?;
        Ok(DraftPageAttachment { snapshot, layout: self.layout, origin: s.origin, frontier: s.end,
            pages: s.pages.iter().map(|&index| DraftPageId {
                index, generation: self.pool.generation(index) }).collect() })
    }

    /// Allocate unpublished destinations for a host restore. Saved device identities
    /// are metadata only: the host owns its slabs after the original pool refs expire.
    pub fn promote(&mut self, saved: &DraftPageAttachment) -> Result<DraftPageAttachment> {
        let len = saved.frontier.checked_sub(saved.origin).ok_or(DraftPageError::Snapshot)?;
        if saved.layout != self.layout || saved.frontier > self.context
            || saved.pages.len() != len.div_ceil(DRAFT_PAGE_ROWS) {
            return Err(DraftPageError::Snapshot);
        }
        let snapshot = DraftSnapshot { pool: self.id, serial: self.next_serial()? };
        self.make_room(saved.pages.len() + self.outstanding_credit(), None)?;
        let pages = self.pool.alloc(saved.pages.len())?;
        self.snapshots.insert(snapshot, Snapshot { origin: saved.origin, end: saved.frontier, pages, revocable: false, pins: 0, used: snapshot.serial });
        self.attachment(snapshot)
    }

    /// Verify the entire typed attachment, not just its reusable numeric page IDs.
    pub fn check_attachment(&self, attachment: &DraftPageAttachment) -> Result<()> {
        if self.attachment(attachment.snapshot)? != *attachment { return Err(DraftPageError::Snapshot); }
        Ok(())
    }

    /// Reserve a private continuation of an exact snapshot, leaving the destination
    /// unbound on exhaustion. The returned tail copy must retire before any producer.
    pub fn restore(&mut self, slot: usize, snapshot: DraftSnapshot, frontier: usize)
        -> Result<(DraftBinding, Vec<TailCopy>)> {
        self.restore_writable(slot, snapshot, frontier, frontier)
    }

    pub fn restore_writable(&mut self, slot: usize, snapshot: DraftSnapshot, frontier: usize,
        write_frontier: usize) -> Result<(DraftBinding, Vec<TailCopy>)> {
        self.vacant(slot)?;
        self.generations[slot].checked_add(1).ok_or(DraftPageError::Stale)?;
        let s = self.snapshots.get(&snapshot).ok_or(DraftPageError::Snapshot)?;
        if frontier != s.end { return Err(DraftPageError::Snapshot); }
        let (origin, end) = (s.origin, s.end);
        if !(origin..=end).contains(&write_frontier) { return Err(DraftPageError::Frontier); }
        let len = end - origin;
        let source = s.pages.clone();
        let need = if len == 0 { 0 } else { source.len() - (write_frontier - origin) / DRAFT_PAGE_ROWS };
        self.make_room(need + self.outstanding_credit(), Some(snapshot))?;
        let (pages, copy) = if len == 0 { (Vec::new(), Vec::new()) } else {
            self.pool.fork_writable(&source, len, source.len(), write_frontier - origin)?
        };
        Ok((self.install(slot, origin, end, pages), copy))
    }

    /// Call only after queued copies/readers of the snapshot have retired.
    pub fn release_snapshot(&mut self, snapshot: DraftSnapshot) -> Result<()> {
        if self.snapshots.get(&snapshot).is_some_and(|s| s.pins != 0) { return Err(DraftPageError::Snapshot); }
        let s = self.snapshots.remove(&snapshot).ok_or(DraftPageError::Snapshot)?;
        self.pool.release(&s.pages);
        Ok(())
    }

    /// Cancellation, including a pending proposal. The owner must drain first.
    pub fn release(&mut self, binding: DraftBinding) -> Result<()> {
        self.sequence(binding)?;
        let s = self.slots[binding.slot].take().expect("validated binding");
        self.pool.release(&s.pages);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservable_counts_only_snapshot_pages_whose_release_frees_them() {
        let mut pool = DraftPagePool::new(8 * 64, 1, 1024, 3).unwrap();
        let binding = pool.bind(0, 0).unwrap();
        pool.admit(binding, 4 * 64, 1).unwrap();
        let r = pool.reserve(binding, 0, 4 * 64).unwrap();
        pool.commit(r, 4 * 64).unwrap();
        // The live credit: ceil(256/64) + ceil(1/64) + 1 = 6 pages, 4 materialized.
        assert_eq!(pool.reservable(), 8 - 4 - 2);
        // A snapshot of the live sequence shares all four of its pages (no tail copy).
        let (snapshot, copies) = pool.capture(binding).unwrap();
        assert!(copies.is_empty());
        pool.retain_snapshot(snapshot).unwrap();
        // Revoking it frees nothing: the live sequence still holds every page.
        assert_eq!(pool.reservable(), 8 - 4 - 2);
        // Once the sequence is released, the snapshot alone holds them: all four come back.
        pool.release(binding).unwrap();
        assert_eq!(pool.reservable(), 8);
    }

    #[test]
    fn committed_attention_table_rejects_pending_and_stale_bindings() {
        let mut pool = DraftPagePool::new(256, 2, 262144, 1).unwrap();
        let binding = pool.bind(0, 8191).unwrap();
        assert!(pool.committed_page_table(binding).unwrap().is_empty());
        let reservation = pool.reserve(binding, 8191, 65).unwrap();
        let expected = pool.page_table(reservation).unwrap();
        assert!(pool.committed_page_table(binding).is_err());
        pool.commit(reservation, 65).unwrap();
        assert_eq!(pool.committed_page_table(binding).unwrap(), expected);
        let (snapshot, _) = pool.capture(binding).unwrap();
        let (restored, _) = pool.restore(1, snapshot, 8256).unwrap();
        let table = pool.committed_page_table(restored).unwrap();
        assert_eq!(table.len(), 2);
        assert_eq!(table[0], expected[0]);
        assert_ne!(table[1], expected[1]);
        pool.release(binding).unwrap();
        assert!(pool.committed_page_table(binding).is_err());
    }

    #[test]
    fn aggregate_live_pressure_defers_instead_of_rejecting_a_request_that_fits_alone() {
        let mut pool = DraftPagePool::new(5 * 64, 2, 1024, 1).unwrap();
        let a = pool.bind(0, 0).unwrap();
        let b = pool.bind(1, 0).unwrap();
        pool.admit(a, 64, 8).unwrap();
        let DraftPageError::Exhausted(error) = pool.admit(b, 64, 8).unwrap_err() else { panic!("expected pressure") };
        assert_eq!(error.needed, 3);
        assert!(error.needed <= error.capacity);
        pool.release(a).unwrap();
        pool.admit(b, 64, 8).unwrap();
        let rows = pool.reserve(b, 0, 64).unwrap();
        pool.commit(rows, 64).unwrap();
        pool.release(b).unwrap();
        assert_eq!(pool.free(), pool.capacity());
    }

    #[test]
    fn live_admission_revokes_retained_snapshots_before_any_producer_can_fail() {
        let mut pool = DraftPagePool::new(8 * 64, 2, 1024, 1).unwrap();
        let mut saved = Vec::new();
        for n in 0..8 {
            let binding = pool.bind(0, n * 64).unwrap();
            let rows = pool.reserve(binding, n * 64, 64).unwrap();
            pool.commit(rows, 64).unwrap();
            let (snapshot, _) = pool.capture(binding).unwrap();
            pool.release(binding).unwrap();
            pool.retain_snapshot(snapshot).unwrap();
            saved.push(snapshot);
        }
        assert_eq!(pool.free(), 0);
        let a = pool.bind(0, 0).unwrap();
        let b = pool.bind(1, 100).unwrap();
        pool.admit(a, 65, 8).unwrap();
        pool.admit(b, 165, 8).unwrap();
        assert!(saved.into_iter().all(|id| pool.attachment(id).is_err()));
        // Each live request has two growth pages, a partial page and a block page.
        // Snapshot pressure cannot steal the unmaterialized credit.
        for (binding, start) in [(a, 0), (b, 100)] {
            let rows = pool.reserve(binding, start, 64).unwrap();
            pool.commit(rows, 64).unwrap();
            let (snapshot, _) = pool.capture(binding).unwrap();
            pool.retain_snapshot(snapshot).unwrap();
            let proposal = pool.reserve(binding, start + 64, 8).unwrap();
            pool.commit(proposal, 0).unwrap();
        }
        pool.release(a).unwrap();
        pool.release(b).unwrap();
        pool.make_room(pool.capacity(), None).unwrap();
        assert_eq!(pool.free(), pool.capacity());
    }

    #[test]
    fn revocation_never_reuses_restore_sources_before_the_copy_event_retires() {
        let mut pool = DraftPagePool::new(5 * 64, 3, 1024, 1).unwrap();
        let source = pool.bind(0, 0).unwrap();
        let rows = pool.reserve(source, 0, 63).unwrap();
        let physical = pool.destinations(rows).unwrap();
        pool.commit(rows, 63).unwrap();
        let (snapshot, copies) = pool.capture(source).unwrap();
        let mut memory = vec![0u64; 5 * 64];
        for (position, row) in physical.into_iter().enumerate() { memory[row as usize] = position as u64 + 1; }
        for copy in copies {
            for row in 0..copy.rows { memory[copy.to as usize * 64 + row] = memory[copy.from as usize * 64 + row]; }
        }
        pool.release(source).unwrap();
        pool.retain_snapshot(snapshot).unwrap();
        pool.pin_snapshot(snapshot).unwrap();
        let (restored, copies) = pool.restore(1, snapshot, 63).unwrap();
        let saved = pool.attachment(snapshot).unwrap();
        let live = pool.bind(2, 0).unwrap();
        pool.admit(live, 64, 8).unwrap();
        let proposal = pool.reserve(live, 0, 64).unwrap();
        let writes = pool.destinations(proposal).unwrap();
        assert!(writes.iter().all(|row| *row as usize / 64 != saved.pages[0].index as usize));
        assert!(pool.release_snapshot(snapshot).is_err());
        assert!(pool.make_room(pool.capacity(), None).is_err());
        // Simulate the queued event: sources must still contain their canonical bytes.
        for copy in copies {
            for row in 0..copy.rows {
                let value = memory[copy.from as usize * 64 + row];
                assert_eq!(value, row as u64 + 1);
                memory[copy.to as usize * 64 + row] = value;
            }
        }
        pool.unpin_snapshot(snapshot).unwrap();
        pool.make_room(4, None).unwrap_err();
        assert!(pool.attachment(snapshot).is_err());
        pool.commit(proposal, 0).unwrap();
        pool.release(live).unwrap();
        pool.release(restored).unwrap();
        assert_eq!(pool.free(), pool.capacity());
    }

    #[test]
    fn handles_from_another_pool_never_alias() {
        let mut a = DraftPagePool::new(128, 1, 1024, 1).unwrap();
        let mut b = DraftPagePool::new(128, 1, 1024, 1).unwrap();
        let binding_a = a.bind(0, 0).unwrap();
        let binding_b = b.bind(0, 0).unwrap();
        let (snapshot_a, _) = a.capture(binding_a).unwrap();
        let (snapshot_b, _) = b.capture(binding_b).unwrap();
        assert!(b.release(binding_a).is_err());
        assert!(b.release_snapshot(snapshot_a).is_err());
        assert!(b.attachment(snapshot_a).is_err());
        assert!(b.attachment(snapshot_b).is_ok());
        assert_eq!(b.frontier(binding_b).unwrap(), (0, 0));
    }

    #[test]
    fn aggregate_budget_is_independent_of_slots_and_context() {
        for slots in [1, 8, 20] {
            for context in [1024, 1 << 20] {
                let pool = DraftPagePool::new(1025, slots, context, 1).unwrap();
                assert_eq!((pool.capacity(), pool.free()), (17, 17));
            }
        }
    }

    #[test]
    fn rejected_suffix_and_slot_reuse_cannot_publish_stale_frontiers() {
        let mut pool = DraftPagePool::new(256, 2, 1 << 20, 7).unwrap();
        let a = pool.bind(0, 131).unwrap();
        let r = pool.reserve(a, 131, 65).unwrap();
        assert_eq!(pool.destinations(r).unwrap(), (0..65).collect::<Vec<i64>>());
        pool.commit(r, 63).unwrap();
        assert_eq!(pool.frontier(a).unwrap(), (131, 194));
        assert_eq!(pool.free(), 3);
        assert!(pool.commit(r, 1).is_err());
        let r = pool.reserve(a, 194, 2).unwrap();
        pool.commit(r, 0).unwrap();
        assert_eq!(pool.frontier(a).unwrap(), (131, 194));
        pool.release(a).unwrap();
        let b = pool.bind(0, 500).unwrap();
        assert!(pool.reserve(a, 194, 1).is_err());
        assert_eq!(pool.frontier(b).unwrap(), (500, 500));
        assert_eq!(pool.free(), 4);
    }

    #[test]
    fn capture_restore_share_full_pages_and_copy_only_the_partial_tail() {
        for rows in [63, 64, 65] {
            let mut pool = DraftPagePool::new(512, 2, 1024, 11).unwrap();
            let a = pool.bind(0, 0).unwrap();
            let r = pool.reserve(a, 0, rows).unwrap();
            pool.commit(r, rows).unwrap();
            let (snapshot, copy) = pool.capture(a).unwrap();
            assert_eq!(copy.first().map(|c| c.rows), (rows % 64 != 0).then_some(rows % 64));
            let attachment = pool.attachment(snapshot).unwrap();
            assert_eq!((attachment.origin, attachment.frontier, attachment.layout), (0, rows, 11));
            let (b, copy) = pool.restore(1, snapshot, rows).unwrap();
            assert_eq!(copy.first().map(|c| c.rows), (rows % 64 != 0).then_some(rows % 64));
            let r = pool.reserve(b, rows, 1).unwrap();
            pool.commit(r, 0).unwrap();
            assert_eq!(pool.attachment(snapshot).unwrap(), attachment);
            pool.release(a).unwrap();
            pool.release(b).unwrap();
            pool.release_snapshot(snapshot).unwrap();
            assert_eq!(pool.free(), 8);
            assert!(pool.attachment(snapshot).is_err());
        }
    }

    #[test]
    fn snapshot_bytes_equal_cold_prefill_after_proposals_and_tail_copies() {
        let mut pool = DraftPagePool::new(1024, 3, 4096, 19).unwrap();
        let mut memory = vec![0u64; 1024];
        let a = pool.bind(0, 0).unwrap();
        for end in [1, 63, 64, 65, 127, 191] {
            let (_, start) = pool.frontier(a).unwrap();
            let proposal = pool.reserve(a, start, 8).unwrap();
            for row in pool.destinations(proposal).unwrap() { memory[row as usize] = u64::MAX; }
            pool.commit(proposal, 0).unwrap();
            let append = pool.reserve(a, start, end - start).unwrap();
            for (position, row) in (start..end).zip(pool.destinations(append).unwrap()) {
                memory[row as usize] = position as u64 + 1;
            }
            pool.commit(append, end - start).unwrap();
        }
        let (snapshot, copy) = pool.capture(a).unwrap();
        for copy in copy {
            let saved = memory[copy.from as usize * 64..copy.from as usize * 64 + copy.rows].to_vec();
            memory[copy.to as usize * 64..copy.to as usize * 64 + copy.rows].copy_from_slice(&saved);
        }
        let attachment = pool.attachment(snapshot).unwrap();
        let host = attachment.pages.iter().enumerate().flat_map(|(i, page)| {
            let rows = (attachment.frontier - i * 64).min(64);
            memory[page.index as usize * 64..page.index as usize * 64 + rows].to_vec()
        }).collect::<Vec<_>>();
        assert_eq!(host, (1..=191).collect::<Vec<u64>>());
        // The writer can append rejected bytes without changing its snapshot tail.
        let proposal = pool.reserve(a, 191, 8).unwrap();
        for row in pool.destinations(proposal).unwrap() { memory[row as usize] = u64::MAX; }
        pool.commit(proposal, 0).unwrap();
        let (b, copy) = pool.restore(1, snapshot, 191).unwrap();
        for copy in copy {
            let saved = memory[copy.from as usize * 64..copy.from as usize * 64 + copy.rows].to_vec();
            memory[copy.to as usize * 64..copy.to as usize * 64 + copy.rows].copy_from_slice(&saved);
        }
        let s = pool.sequence(b).unwrap();
        let restored = (0..191).map(|position| memory[s.pages[position / 64] as usize * 64 + position % 64])
            .collect::<Vec<_>>();
        assert_eq!(restored, host);
        assert_eq!(pool.attachment(snapshot).unwrap(), attachment);
        pool.release(a).unwrap();
        pool.release(b).unwrap();
        pool.release_snapshot(snapshot).unwrap();
        assert_eq!(pool.free(), 16);
    }

    #[test]
    fn exhaustion_and_bad_restore_are_transactional() {
        let mut pool = DraftPagePool::new(128, 2, 1024, 1).unwrap();
        let a = pool.bind(0, 0).unwrap();
        let r = pool.reserve(a, 0, 65).unwrap();
        pool.commit(r, 65).unwrap();
        assert!(pool.capture(a).is_err());
        assert_eq!(pool.free(), 0);
        assert_eq!(pool.frontier(a).unwrap(), (0, 65));
        assert!(pool.reserve(a, 65, 64).is_err());
        let r = pool.reserve(a, 65, 1).unwrap();
        pool.commit(r, 0).unwrap();
        pool.release(a).unwrap();
        let a = pool.bind(0, 0).unwrap();
        let r = pool.reserve(a, 0, 64).unwrap();
        pool.commit(r, 64).unwrap();
        let (snapshot, _) = pool.capture(a).unwrap();
        assert!(pool.restore(1, snapshot, 63).is_err());
        assert!(pool.bind(1, 0).is_ok());
    }

    #[test]
    fn host_promotions_reject_stale_or_incompatible_attachments_transactionally() {
        let mut pool = DraftPagePool::new(512, 1, 1024, 17).unwrap();
        let binding = pool.bind(0, 0).unwrap();
        let reservation = pool.reserve(binding, 0, 65).unwrap();
        pool.commit(reservation, 65).unwrap();
        let (snapshot, _) = pool.capture(binding).unwrap();
        let saved = pool.attachment(snapshot).unwrap();
        pool.release_snapshot(snapshot).unwrap();
        pool.release(binding).unwrap();
        assert!(pool.check_attachment(&saved).is_err());
        let promoted = pool.promote(&saved).unwrap();
        pool.check_attachment(&promoted).unwrap();
        assert_ne!(saved.snapshot, promoted.snapshot);
        let before = pool.free();
        let mut bad = saved.clone();
        bad.layout += 1;
        assert!(pool.promote(&bad).is_err());
        bad = saved;
        bad.pages.clear();
        assert!(pool.promote(&bad).is_err());
        assert_eq!(pool.free(), before);
        pool.release_snapshot(promoted.snapshot).unwrap();
        assert_eq!(pool.free(), 8);
    }

    #[test]
    fn lagging_writer_copies_every_writable_page_at_aligned_frontiers() {
        for (origin, rows, writable) in [(0, 64, 63), (513, 128, 576), (0, 193, 63)] {
            let mut pool = DraftPagePool::new(2048, 2, 4096, 2).unwrap();
            let a = pool.bind(0, origin).unwrap();
            let r = pool.reserve(a, origin, rows).unwrap();
            pool.commit(r, rows).unwrap();
            let source = pool.sequence(a).unwrap().pages.clone();
            let (snapshot, copies) = pool.capture_writable(a, writable).unwrap();
            let saved = pool.snapshots.get(&snapshot).unwrap().pages.clone();
            let shared = (writable - origin) / 64;
            assert_eq!(&source[..shared], &saved[..shared]);
            assert_eq!(copies.len(), rows.div_ceil(64) - shared);
            for copy in copies { assert_ne!(copy.from, copy.to); }
            let (b, copies) = pool.restore_writable(1, snapshot, origin + rows, writable).unwrap();
            assert_eq!(copies.len(), rows.div_ceil(64) - shared);
            for copy in copies { assert_ne!(copy.from, copy.to); }
            assert!(pool.capture_writable(a, origin + rows + 1).is_err());
            pool.release(a).unwrap();
            pool.release(b).unwrap();
            pool.release_snapshot(snapshot).unwrap();
            assert_eq!(pool.free(), 32);
        }
    }

    #[test]
    fn empty_and_cold_origin_snapshots_keep_their_semantics() {
        let mut pool = DraftPagePool::new(64, 2, 1024, 2).unwrap();
        let a = pool.bind(0, 513).unwrap();
        let (snapshot, copy) = pool.capture(a).unwrap();
        assert!(copy.is_empty());
        assert_eq!(pool.free(), 1);
        let (b, _) = pool.restore(1, snapshot, 513).unwrap();
        assert_eq!(pool.frontier(b).unwrap(), (513, 513));
        assert_ne!(pool.attachment(snapshot).unwrap().origin, 0);
    }
}
