//! A burst of upcoming encoder units' L1 I/O, started ahead of reservation
//! (#815). A wave is adopted only by the reservation it was prepared for;
//! every other path cancels it before dropping it.

/// A wave that must be cancelled (its queued I/O drained) before it drops.
pub(crate) trait CancelWave { fn cancel(&mut self); }

impl CancelWave for cuteafd_loader::EngramWave {
    fn cancel(&mut self) { cuteafd_loader::EngramWave::cancel(self) }
}

/// What a prefetched wave was prepared for: the request, its reserved
/// prefill frontier and the unit's tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PrefetchKey<L> { pub lease: L, pub position: u64, pub tokens: Vec<u32> }

/// The mandatory lookahead burst; each held wave owns only a layer-1 gather.
pub(crate) struct PrefetchSlot<L, W: CancelWave> { held: Vec<(PrefetchKey<L>, W)> }

impl<L: PartialEq, W: CancelWave> Default for PrefetchSlot<L, W> {
    fn default() -> Self { Self { held: Vec::new() } }
}

impl<L: PartialEq, W: CancelWave> PrefetchSlot<L, W> {
    /// Hold a distinct future unit; callers deduplicate before preparing its I/O.
    pub(crate) fn hold(&mut self, key: PrefetchKey<L>, wave: W) {
        self.held.push((key, wave));
    }
    /// Adopt exactly one matching unit, preserving later units in its burst.
    /// A stale reservation cancels every future unit for that owner only.
    pub(crate) fn take_matching(&mut self, key: &PrefetchKey<L>) -> Option<W> {
        if let Some(index) = self.held.iter().position(|(held, _)| held == key) {
            Some(self.held.remove(index).1)
        } else { self.cancel_owner(&key.lease); None }
    }
    /// Cancels a wave held for `lease` (release, decoder replay).
    pub(crate) fn cancel_owner(&mut self, lease: &L) {
        let mut index = 0;
        while index < self.held.len() {
            if self.held[index].0.lease == *lease { self.held.remove(index).1.cancel(); }
            else { index += 1; }
        }
    }
    /// Cancels every wave held for an owner other than `lease`; returns how many.
    pub(crate) fn cancel_others(&mut self, lease: &L) -> usize {
        let before = self.held.len();
        let mut index = 0;
        while index < self.held.len() {
            if self.held[index].0.lease != *lease { self.held.remove(index).1.cancel(); }
            else { index += 1; }
        }
        before - self.held.len()
    }
    /// Cancels any held wave.
    pub(crate) fn cancel(&mut self) {
        for (_, mut stale) in self.held.drain(..) { stale.cancel(); }
    }
    pub(crate) fn is_empty(&self) -> bool { self.held.is_empty() }
    pub(crate) fn matching(&self, key: &PrefetchKey<L>) -> Option<&W> {
        self.held.iter().find(|(held, _)| held == key).map(|(_, wave)| wave)
    }
}

impl<L, W: CancelWave> Drop for PrefetchSlot<L, W> {
    /// A request bank dropped with a prefetch in flight (abort) cancels it.
    fn drop(&mut self) {
        for (_, mut stale) in self.held.drain(..) { stale.cancel(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Records cancels and drops; dropping uncancelled is a leak.
    struct Wave { id: u32, log: Rc<RefCell<Vec<String>>>, cancelled: bool }
    impl CancelWave for Wave { fn cancel(&mut self) { self.cancelled = true; self.log.borrow_mut().push(format!("cancel {}", self.id)); } }
    impl Drop for Wave {
        fn drop(&mut self) { self.log.borrow_mut().push(format!("drop {} cancelled={}", self.id, self.cancelled)); }
    }
    fn key(lease: u32, position: u64, tokens: &[u32]) -> PrefetchKey<u32> { PrefetchKey { lease, position, tokens: tokens.to_vec() } }
    fn wave(id: u32, log: &Rc<RefCell<Vec<String>>>) -> Wave { Wave { id, log: log.clone(), cancelled: false } }
    fn leaked(log: &Rc<RefCell<Vec<String>>>) -> bool { log.borrow().iter().any(|line| line.ends_with("cancelled=false")) }

    #[test]
    fn another_owners_call_cancels_only_foreign_waves() {
        let log = Rc::default();
        let mut slot = PrefetchSlot::default();
        slot.hold(key(1, 0, &[1]), wave(1, &log));
        slot.hold(key(2, 0, &[2]), wave(2, &log));
        slot.hold(key(1, 1, &[3]), wave(3, &log));
        assert_eq!(slot.cancel_others(&2), 2);
        assert!(slot.matching(&key(2, 0, &[2])).is_some());
        assert_eq!(slot.cancel_others(&2), 0);
        assert!(slot.take_matching(&key(2, 0, &[2])).is_some_and(|mut w| { w.cancel(); true }));
        assert!(!leaked(&log));
    }

    #[test]
    fn an_exact_reservation_adopts_the_prefetch() {
        let log = Rc::default();
        let mut slot = PrefetchSlot::default();
        slot.hold(key(1, 2752, &[5, 6]), wave(1, &log));
        let adopted = slot.take_matching(&key(1, 2752, &[5, 6])).expect("adopted");
        assert_eq!(adopted.id, 1);
        assert!(!adopted.cancelled && slot.is_empty());
        drop(slot);
        assert!(log.borrow().is_empty(), "nothing cancelled or dropped by the slot");
        let mut adopted = adopted; adopted.cancel();
    }

    #[test]
    fn mismatched_reservations_cancel_and_never_adopt() {
        for other in [key(1, 2688, &[5, 6]), key(1, 2752, &[5, 7])] {
            let log = Rc::default();
            let mut slot = PrefetchSlot::default();
            slot.hold(key(1, 2752, &[5, 6]), wave(1, &log));
            assert!(slot.take_matching(&other).is_none(), "{other:?} adopted a wave prepared for another key");
            assert!(slot.is_empty());
            assert_eq!(*log.borrow(), ["cancel 1", "drop 1 cancelled=true"]);
        }
    }

    /// A call's lookahead holds the next call's first units at its own width;
    /// when the next call's depth (unit width) changes, its first reservation
    /// misses, cancels every held lookahead unit and frees their slots: a
    /// depth change costs a cold gather, never a pinned slot or a stall.
    #[test]
    fn a_depth_change_cancels_the_whole_lookahead_on_its_first_miss() {
        let log = Rc::default();
        let mut slot = PrefetchSlot::default();
        let tokens: Vec<u32> = (0..12).collect();
        // Lookahead at width 4 (k3): three units from position 8256.
        for (i, chunk) in tokens.chunks(4).enumerate() { slot.hold(key(1, 8256 + 4 * i as u64, chunk), wave(i as u32, &log)); }
        // The next call runs k2 at width 6: its first unit misses.
        assert!(slot.take_matching(&key(1, 8256, &tokens[..6])).is_none());
        assert!(slot.is_empty(), "lookahead units held past a depth change");
        assert!(!leaked(&log));
        assert_eq!(log.borrow().iter().filter(|line| line.starts_with("cancel")).count(), 3);
    }

    #[test]
    fn release_and_replay_cancel_only_their_own_lease() {
        let log = Rc::default();
        let mut slot = PrefetchSlot::default();
        slot.hold(key(1, 0, &[5]), wave(1, &log));
        slot.cancel_owner(&2);
        assert!(!slot.is_empty(), "another request's release keeps the prefetch");
        slot.cancel_owner(&1);
        assert!(slot.is_empty());
        assert_eq!(*log.borrow(), ["cancel 1", "drop 1 cancelled=true"]);
    }

    #[test]
    fn burst_adoption_and_owner_cancel_preserve_other_requests() {
        let log = Rc::default();
        let mut slot = PrefetchSlot::default();
        for i in 0..4 { slot.hold(key(1, i, &[i as u32]), wave(i as u32, &log)); }
        slot.hold(key(2, 0, &[9]), wave(9, &log));
        assert!(slot.take_matching(&key(3, 0, &[0])).is_none());
        for i in 0..2 {
            let mut adopted = slot.take_matching(&key(1, i, &[i as u32])).unwrap();
            assert_eq!(adopted.id, i as u32); adopted.cancel();
        }
        slot.cancel_owner(&1);
        assert!(!slot.is_empty());
        let mut other = slot.take_matching(&key(2, 0, &[9])).unwrap(); other.cancel();
        assert!(slot.is_empty());
        drop(slot); drop(other);
        assert!(!leaked(&log));
    }

    #[test]
    fn a_new_prefetch_and_an_abort_cancel_the_held_wave() {
        let log = Rc::default();
        let mut slot = PrefetchSlot::default();
        slot.hold(key(1, 0, &[5]), wave(1, &log));
        slot.hold(key(1, 1, &[6]), wave(2, &log));
        drop(slot);
        assert_eq!(*log.borrow(), ["cancel 1", "drop 1 cancelled=true", "cancel 2", "drop 2 cancelled=true"]);
        assert!(!leaked(&log));
    }
}
