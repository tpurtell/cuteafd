//! Device arena for positional marks: the state a snapshot needs besides its pages, copied at
//! capture because the owning sequence keeps overwriting it (MiMo: the last 128 rows of every
//! sliding-window ring; recurrent families: whole recurrent states of 110-141 MiB). Slots are
//! fixed-size and preallocated by the family, so capture and restore only move bytes.
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MarkSlot(pub u32);

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("mark arena exhausted: all {slots} slots hold retained marks")]
pub struct ArenaExhausted {
    pub slots: usize,
}

pub struct MarkArena {
    free: Vec<u32>,
    slots: usize,
    slot_bytes: usize,
}

impl MarkArena {
    pub fn new(slots: usize, slot_bytes: usize) -> Self {
        let count = u32::try_from(slots).expect("slot count fits u32");
        Self { free: (0..count).rev().collect(), slots, slot_bytes }
    }

    /// Host-enabled caches need only staging slots; host-off caches retain marks on device.
    pub fn slots_for(lanes: usize, entries: usize, slot_bytes: usize, budget_bytes: usize, host: bool) -> usize {
        cuteafd_core::prefix::mark_slots_with_host(
            lanes as u64, entries as u64, slot_bytes as u64, budget_bytes as u64, host,
        ) as usize
    }

    pub fn slots(&self) -> usize {
        self.slots
    }
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    pub fn in_use(&self) -> usize {
        self.slots - self.free.len()
    }
    pub fn free(&self) -> usize {
        self.free.len()
    }
    pub fn offset(&self, slot: MarkSlot) -> usize {
        slot.0 as usize * self.slot_bytes
    }

    pub fn take(&mut self) -> Result<MarkSlot, ArenaExhausted> {
        self.free.pop().map(MarkSlot).ok_or(ArenaExhausted { slots: self.slots })
    }

    /// Return a slot whose queued copies have drained.
    pub fn give_back(&mut self, slot: MarkSlot) {
        debug_assert!((slot.0 as usize) < self.slots && !self.free.contains(&slot.0), "slot {slot:?} returned twice");
        self.free.push(slot.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_sizing_keeps_the_lane_floor_and_the_byte_budget() {
        const MIB: usize = 1 << 20;
        // MiMo V2.6 Pro: 39.3 MB marks, 20 entries per bank, 2 GiB budget: all 42 fit.
        assert_eq!(MarkArena::slots_for(1, 20, 39_321_600, 2048 * MIB, false), 42);
        // GLM 5.3 Flash: 141 MiB marks, same budget: 14 slots.
        assert_eq!(MarkArena::slots_for(2, 20, 141 * MIB, 2048 * MIB, false), 14);
        // The floor wins over a tiny budget; a disabled cache takes nothing.
        assert_eq!(MarkArena::slots_for(2, 20, 141 * MIB, 0, false), 6);
        assert_eq!(MarkArena::slots_for(2, 0, 141 * MIB, 2048 * MIB, false), 0);
    }

    #[test]
    fn slots_are_handed_out_and_returned() {
        let mut arena = MarkArena::new(2, 100);
        let a = arena.take().unwrap();
        let b = arena.take().unwrap();
        assert_eq!((a, b, arena.offset(b)), (MarkSlot(0), MarkSlot(1), 100));
        assert_eq!(arena.take(), Err(ArenaExhausted { slots: 2 }));
        arena.give_back(a);
        assert_eq!((arena.in_use(), arena.take().unwrap()), (1, a));
    }
}
