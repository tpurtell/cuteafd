//! A bounded host cache of table rows, filled as early as their ids are known
//! and read by the gathers that fill the pinned upload staging.
//!
//! Rows live in one fixed anonymous mapping of `capacity` slots. A short mutex
//! guards the index and slot metadata only: callers reserve under it, read or
//! copy row bytes outside it, then publish. Every operation is O(1) per row.
//!
//! - A slot is `Filling` from its reservation until its single filler
//!   publishes it (`Ready`) or drops without publishing (`Failed`, which wakes
//!   waiters so they read the row themselves).
//! - Pins keep a slot from eviction. Every pinned slot is charged to one class
//!   (prefill look-ahead and gathers, or decode); a class reserves new slots
//!   only below its share, and the shares sum to the capacity, so the prefill
//!   look-ahead can never take the decode share. Pinning a slot that already
//!   holds the row needs no new capacity and is always allowed.
//! - Unpinned `Ready` slots form one list, evicted from the head. A slot joins
//!   at the tail when it is published or its last pin drops, so eviction takes
//!   the row least recently added or re-added.
use super::{invalid, ResidentBytes, Result, TableStats};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

const NIL: u32 = u32::MAX;

/// Who a pin is charged to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowClass { Prefill = 0, Decode = 1 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State { Free, Filling, Ready, Failed }

#[derive(Clone, Copy)]
struct Slot { row: u64, state: State, pins: [u32; 2], owner: u8, prev: u32, next: u32, listed: bool }

/// Row ids are already hash values; one multiply spreads their high bits.
#[derive(Default)]
struct RowHasher(u64);
impl Hasher for RowHasher {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, bytes: &[u8]) { for &b in bytes { self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15); } }
    fn write_u64(&mut self, value: u64) { self.0 = value.wrapping_mul(0x9E37_79B9_7F4A_7C15); }
}

struct Inner {
    index: HashMap<u64, u32, BuildHasherDefault<RowHasher>>,
    slots: Vec<Slot>,
    free: Vec<u32>,
    head: u32,
    tail: u32,
    /// Pinned slots charged to each class.
    held: [usize; 2],
}

/// What a reserving caller does with one requested row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// Published: copy it now.
    Ready(u32),
    /// Another filler owns it: wait for it, then copy (or read it on failure).
    Wait(u32),
    /// Reserved for this caller: read it, then publish it.
    Fill(u32),
    /// The class's share is full: read it privately, not cached.
    Uncached,
}

pub struct RowCache {
    width: usize,
    shares: [usize; 2],
    bytes: ResidentBytes,
    inner: Mutex<Inner>,
    published: Condvar,
    stats: Arc<TableStats>,
}

// SAFETY: slot bytes are written only by the single owner of a `Filling` slot
// and read only through pins of `Ready` slots, which are never reassigned while
// pinned; the mutex orders publication before every read.
unsafe impl Sync for RowCache {}

impl RowCache {
    /// Host bytes a cache of these shares holds: rows plus index and slot metadata.
    pub fn host_bytes(width: usize, shares: [usize; 2]) -> usize {
        let capacity = shares[0] + shares[1];
        capacity * width + capacity * (std::mem::size_of::<Slot>() + 4) + capacity * 2 * 17
    }

    /// `shares` are the prefill and decode slot counts; `stats` receives the cache counters.
    pub fn new(width: usize, shares: [usize; 2], stats: Arc<TableStats>) -> Result<Arc<Self>> {
        let capacity = shares[0].checked_add(shares[1]).ok_or_else(|| invalid("row cache size overflow"))?;
        if width == 0 || shares.contains(&0) || capacity >= NIL as usize {
            return Err(invalid("invalid row cache geometry"));
        }
        let bytes = ResidentBytes::new(capacity.checked_mul(width).ok_or_else(|| invalid("row cache size overflow"))?)?;
        let empty = Slot { row: 0, state: State::Free, pins: [0; 2], owner: 0, prev: NIL, next: NIL, listed: false };
        let mut index = HashMap::with_capacity_and_hasher(capacity, Default::default());
        index.reserve(capacity);
        let inner = Inner { index, slots: vec![empty; capacity], free: (0..capacity as u32).rev().collect(),
            head: NIL, tail: NIL, held: [0; 2] };
        Ok(Arc::new(Self { width, shares, bytes, inner: Mutex::new(inner), published: Condvar::new(), stats }))
    }

    pub fn width(&self) -> usize { self.width }
    /// New slots `class` could reserve now.
    pub fn room(&self, class: RowClass) -> usize {
        let inner = self.lock();
        let c = class as usize;
        self.shares[c].saturating_sub(inner.held[c]).min(self.capacity() - inner.held[0] - inner.held[1])
    }
    pub fn capacity(&self) -> usize { self.shares[0] + self.shares[1] }

    fn lock(&self) -> MutexGuard<'_, Inner> { self.inner.lock().unwrap_or_else(|p| p.into_inner()) }

    /// Pin every row in `rows` (distinct) for `class`, reserving absent ones
    /// while the class's share allows. The returned lookups follow `rows`.
    pub fn acquire(self: &Arc<Self>, rows: &[u64], class: RowClass) -> (Pins, Vec<Lookup>) {
        let mut pins = Pins { cache: Arc::clone(self), class, slots: Vec::with_capacity(rows.len()),
            filling: Vec::new() };
        let mut lookups = Vec::with_capacity(rows.len());
        let (mut hits, mut waits, mut denied, mut evicted) = (0u64, 0u64, 0u64, 0u64);
        // Bounded critical sections: other fillers publish between chunks.
        for chunk in rows.chunks(2048) {
            let mut inner = self.lock();
            for &row in chunk {
                let lookup = match inner.index.get(&row).copied() {
                    Some(slot) => {
                        inner.pin(slot, class);
                        pins.slots.push(slot);
                        match inner.slots[slot as usize].state {
                            State::Ready => { hits += 1; Lookup::Ready(slot) },
                            _ => { waits += 1; Lookup::Wait(slot) },
                        }
                    }
                    None => match inner.reserve(row, class, self.shares, &mut evicted) {
                        Some(slot) => { pins.slots.push(slot); pins.filling.push(slot); Lookup::Fill(slot) },
                        None => { denied += 1; Lookup::Uncached },
                    },
                };
                lookups.push(lookup);
            }
        }
        let stats = &self.stats;
        stats.cache_hit_rows.fetch_add(hits, Ordering::Relaxed);
        stats.cache_wait_rows.fetch_add(waits, Ordering::Relaxed);
        stats.cache_denied_rows.fetch_add(denied, Ordering::Relaxed);
        stats.cache_evictions.fetch_add(evicted, Ordering::Relaxed);
        (pins, lookups)
    }

    /// # Safety
    /// The caller holds a pin on `slot`, which is `Ready`.
    unsafe fn row(&self, slot: u32) -> &[u8] {
        // SAFETY: slot < capacity; a pinned Ready slot is never written again.
        unsafe { std::slice::from_raw_parts(self.bytes.base.as_ptr().add(slot as usize * self.width), self.width) }
    }

    /// # Safety
    /// The caller is the single filler of the `Filling` slot `slot`.
    unsafe fn write(&self, slot: u32, parts: &[&[u8]]) {
        let mut at = slot as usize * self.width;
        for part in parts {
            // SAFETY: the parts total `width` bytes (checked by the caller) inside
            // this slot, which no reader touches before publication.
            unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), self.bytes.base.as_ptr().add(at), part.len()) };
            at += part.len();
        }
    }
}

impl Inner {
    fn unlink(&mut self, slot: u32) {
        let Slot { prev, next, listed, .. } = self.slots[slot as usize];
        if !listed { return; }
        if prev == NIL { self.head = next; } else { self.slots[prev as usize].next = next; }
        if next == NIL { self.tail = prev; } else { self.slots[next as usize].prev = prev; }
        let entry = &mut self.slots[slot as usize];
        (entry.prev, entry.next, entry.listed) = (NIL, NIL, false);
    }
    fn push_tail(&mut self, slot: u32) {
        let tail = self.tail;
        {
            let entry = &mut self.slots[slot as usize];
            (entry.prev, entry.next, entry.listed) = (tail, NIL, true);
        }
        if tail == NIL { self.head = slot; } else { self.slots[tail as usize].next = slot; }
        self.tail = slot;
    }
    fn pin(&mut self, slot: u32, class: RowClass) {
        let c = class as usize;
        let entry = &mut self.slots[slot as usize];
        if entry.pins == [0, 0] {
            entry.owner = class as u8;
            self.held[c] += 1;
            self.unlink(slot);
        }
        self.slots[slot as usize].pins[c] += 1;
    }
    /// Charge a new slot to `class` when its share and the capacity allow.
    fn reserve(&mut self, row: u64, class: RowClass, shares: [usize; 2], evicted: &mut u64) -> Option<u32> {
        let c = class as usize;
        if self.held[c] >= shares[c] || self.held[0] + self.held[1] >= shares[0] + shares[1] { return None; }
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => {
                // Every unpinned slot that is not free is Ready and listed.
                let victim = self.head;
                debug_assert_ne!(victim, NIL, "an unpinned slot exists below capacity");
                if victim == NIL { return None; }
                self.unlink(victim);
                let old = self.slots[victim as usize].row;
                self.index.remove(&old);
                *evicted += 1;
                victim
            }
        };
        self.index.insert(row, slot);
        self.slots[slot as usize] = Slot { row, state: State::Filling, pins: [0; 2], owner: class as u8,
            prev: NIL, next: NIL, listed: false };
        self.pin(slot, class);
        Some(slot)
    }
    /// Drop one pin of `class`; the slot's charge moves to the other class
    /// while it still pins, and the slot becomes evictable (or free) at zero.
    fn unpin(&mut self, slot: u32, class: RowClass) {
        let c = class as usize;
        let entry = &mut self.slots[slot as usize];
        debug_assert!(entry.pins[c] > 0);
        entry.pins[c] -= 1;
        if entry.pins[c] > 0 || entry.owner as usize != c { return; }
        let other = 1 - c;
        self.held[c] -= 1;
        if entry.pins[other] > 0 {
            entry.owner = other as u8;
            self.held[other] += 1;
            return;
        }
        match entry.state {
            State::Ready => self.push_tail(slot),
            State::Failed | State::Free => { self.slots[slot as usize].state = State::Free; self.free.push(slot); },
            State::Filling => unreachable!("a filling slot keeps its filler's pin"),
        }
    }
}

/// One caller's pins (and the slots it must fill). Dropping it fails every
/// reserved slot it did not publish, waking their waiters, then unpins.
pub struct Pins {
    cache: Arc<RowCache>,
    class: RowClass,
    slots: Vec<u32>,
    filling: Vec<u32>,
}

impl Pins {
    pub fn class(&self) -> RowClass { self.class }

    /// Write a reserved slot's row (`parts` total the cache width) and publish it.
    pub fn publish(&mut self, slot: u32, parts: &[&[u8]]) -> Result<()> {
        let at = self.filling.iter().position(|&s| s == slot).ok_or_else(|| invalid("publishing an unreserved row"))?;
        if parts.iter().map(|p| p.len()).sum::<usize>() != self.cache.width {
            return Err(invalid("published row width differs from the cache"));
        }
        // SAFETY: `slot` is reserved by these pins and not yet published.
        unsafe { self.cache.write(slot, parts) };
        self.filling.swap_remove(at);
        self.cache.lock().slots[slot as usize].state = State::Ready;
        self.cache.published.notify_all();
        Ok(())
    }

    /// Publish many reserved slots under one lock (rows written first).
    pub fn publish_all<'a>(&mut self, rows: impl Iterator<Item = (u32, [&'a [u8]; 2])>) -> Result<usize> {
        let mut written = Vec::new();
        for (slot, parts) in rows {
            if !self.filling.contains(&slot) { return Err(invalid("publishing an unreserved row")); }
            if parts[0].len() + parts[1].len() != self.cache.width {
                return Err(invalid("published row width differs from the cache"));
            }
            // SAFETY: `slot` is reserved by these pins and not yet published.
            unsafe { self.cache.write(slot, &parts) };
            written.push(slot);
        }
        if written.is_empty() { return Ok(0); }
        {
            let mut inner = self.cache.lock();
            for &slot in &written { inner.slots[slot as usize].state = State::Ready; }
        }
        self.filling.retain(|slot| !written.contains(slot));
        self.cache.published.notify_all();
        Ok(written.len())
    }

    /// Wait until `slot` (pinned here) is published; false when its filler failed.
    pub fn wait(&self, slot: u32) -> bool {
        let mut inner = self.cache.lock();
        loop {
            match inner.slots[slot as usize].state {
                State::Ready => return true,
                State::Failed | State::Free => return false,
                State::Filling => inner = self.cache.published.wait(inner).unwrap_or_else(|p| p.into_inner()),
            }
        }
    }

    /// No pinned row is still being filled (each is published, or its filler failed).
    pub fn settled(&self) -> bool {
        let inner = self.cache.lock();
        self.slots.iter().all(|&slot| inner.slots[slot as usize].state != State::Filling)
    }

    /// A published row pinned here.
    pub fn row(&self, slot: u32) -> &[u8] {
        debug_assert!(self.slots.contains(&slot));
        // SAFETY: pinned by `self`; callers read only slots reported or waited Ready.
        unsafe { self.cache.row(slot) }
    }
}

impl Drop for Pins {
    fn drop(&mut self) {
        let mut inner = self.cache.lock();
        for &slot in &self.filling {
            let entry = &mut inner.slots[slot as usize];
            entry.state = State::Failed;
            let row = entry.row;
            if inner.index.get(&row) == Some(&slot) { inner.index.remove(&row); }
        }
        for &slot in &self.slots { inner.unpin(slot, self.class); }
        drop(inner);
        if !self.filling.is_empty() { self.cache.published.notify_all(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(shares: [usize; 2]) -> Arc<RowCache> {
        RowCache::new(4, shares, Arc::new(TableStats::default())).unwrap()
    }
    fn fill(pins: &mut Pins, lookups: &[Lookup], rows: &[u64]) {
        for (lookup, &row) in lookups.iter().zip(rows) {
            if let Lookup::Fill(slot) = *lookup { pins.publish(slot, &[&(row as u32).to_le_bytes()]).unwrap(); }
        }
    }

    #[test]
    fn reserve_publish_hit_and_evict_least_recently_added() {
        let cache = cache([3, 1]);
        let (mut a, lookups) = cache.acquire(&[10, 11, 12], RowClass::Prefill);
        assert!(lookups.iter().all(|l| matches!(l, Lookup::Fill(_))));
        fill(&mut a, &lookups, &[10, 11, 12]);
        // The prefill share is full while pinned: a fourth row is uncached.
        let (b, more) = cache.acquire(&[13], RowClass::Prefill);
        assert_eq!(more, [Lookup::Uncached]);
        drop(b);
        drop(a);
        // A repeat hits; re-adding 10 moves it behind 11 and 12.
        let (again, hits) = cache.acquire(&[10], RowClass::Prefill);
        let Lookup::Ready(slot) = hits[0] else { panic!("expected a hit") };
        assert_eq!(again.row(slot), 10u32.to_le_bytes());
        drop(again);
        let (mut c, new) = cache.acquire(&[20, 21], RowClass::Prefill);
        fill(&mut c, &new, &[20, 21]);
        drop(c);
        let (d, probe) = cache.acquire(&[10, 11, 12], RowClass::Prefill);
        assert!(matches!(probe[0], Lookup::Ready(_)), "10 was re-added last and survives");
        assert_eq!(&probe[1..], [Lookup::Fill(probe_slot(&probe[1])), Lookup::Fill(probe_slot(&probe[2]))]);
        drop(d);
        // 21 evicted 11; then 11 evicted 12 and 12 evicted 20 (10 stays pinned).
        assert_eq!(cache.stats.cache_evictions.load(Ordering::Relaxed), 3);
    }
    fn probe_slot(lookup: &Lookup) -> u32 { match *lookup { Lookup::Fill(s) | Lookup::Ready(s) | Lookup::Wait(s) => s, Lookup::Uncached => NIL } }

    #[test]
    fn decode_share_survives_a_full_prefill_look_ahead() {
        let cache = cache([4, 2]);
        let (_ahead, lookups) = cache.acquire(&[1, 2, 3, 4, 5], RowClass::Prefill);
        assert_eq!(lookups.iter().filter(|l| **l == Lookup::Uncached).count(), 1);
        let (_decode, rows) = cache.acquire(&[7, 8, 9], RowClass::Decode);
        assert_eq!(rows.iter().filter(|l| matches!(l, Lookup::Fill(_))).count(), 2);
        assert_eq!(rows[2], Lookup::Uncached);
    }

    #[test]
    fn a_failed_filler_wakes_waiters_and_frees_the_row() {
        let cache = cache([2, 1]);
        let (filler, lookups) = cache.acquire(&[5], RowClass::Prefill);
        assert!(matches!(lookups[0], Lookup::Fill(_)));
        let (waiter, wait) = cache.acquire(&[5], RowClass::Decode);
        let Lookup::Wait(slot) = wait[0] else { panic!("expected a wait") };
        let handle = std::thread::spawn(move || waiter.wait(slot));
        std::thread::sleep(std::time::Duration::from_millis(20));
        drop(filler);
        assert!(!handle.join().unwrap());
        // The failed slot left the index: the next caller reserves the row again.
        let (_again, again) = cache.acquire(&[5], RowClass::Prefill);
        assert!(matches!(again[0], Lookup::Fill(_)));
        let inner = cache.lock();
        assert_eq!(inner.held, [1, 0]);
    }

    #[test]
    fn waiters_see_published_rows_and_charges_move_between_classes() {
        let cache = cache([1, 1]);
        let (mut filler, lookups) = cache.acquire(&[9], RowClass::Prefill);
        let (waiter, wait) = cache.acquire(&[9], RowClass::Decode);
        let Lookup::Wait(slot) = wait[0] else { panic!("expected a wait") };
        fill(&mut filler, &lookups, &[9]);
        assert!(waiter.wait(slot));
        assert_eq!(waiter.row(slot), 9u32.to_le_bytes());
        drop(filler);
        // The decode pin now carries the slot: prefill may reserve again.
        assert_eq!(cache.lock().held, [0, 1]);
        let (_p, more) = cache.acquire(&[3], RowClass::Prefill);
        assert!(matches!(more[0], Lookup::Fill(_)));
        drop(waiter);
        assert_eq!(cache.lock().held, [1, 0]);
    }

    #[test]
    fn publish_rejects_foreign_slots_and_wrong_widths() {
        let cache = cache([2, 1]);
        let (mut pins, lookups) = cache.acquire(&[1], RowClass::Prefill);
        let Lookup::Fill(slot) = lookups[0] else { panic!() };
        assert!(pins.publish(slot, &[&[0; 3]]).is_err());
        assert!(pins.publish(slot + 1, &[&[0; 4]]).is_err());
        assert_eq!(pins.publish_all([(slot, [&[1u8, 2][..], &[3u8, 4][..]])].into_iter()).unwrap(), 1);
        assert_eq!(pins.row(slot), [1, 2, 3, 4]);
    }
}
