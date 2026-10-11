//! Exact-key graph storage. Retirement never destroys an executable until its
//! caller proves every stream using the captured storage has drained.
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphPolicy {
    FixedStartup,
    Budgeted { bytes: Option<u64> },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphStats {
    pub captures: u64,
    pub recaptures: u64,
    pub evictions: u64,
    pub eager_runs: u64,
}

/// A bank miss is not implicit permission to capture on a live request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphDecision {
    Replay,
    Capture,
    Eager,
}

struct Entry<E> {
    exec: E,
    launched: u64,
}

/// E owns the capture's GraphOwner (including its resource pins). The bank is
/// independent of the native executable type, allowing CPU-only policy tests.
pub(crate) struct GraphBank<K, E> {
    entries: HashMap<K, Entry<E>>,
    seen: HashSet<K>,
    policy: GraphPolicy,
    startup: Option<HashSet<K>>,
    clock: u64,
    stats: GraphStats,
    measured_bytes: u64,
    measured: u64,
    frozen: Option<u64>,
    retired: Vec<E>,
}

impl<K: Hash + Eq + Clone + std::fmt::Debug, E> GraphBank<K, E> {
    pub fn new(budget: Option<u64>) -> Self {
        Self::with_policy(GraphPolicy::Budgeted { bytes: budget })
    }

    pub fn with_policy(policy: GraphPolicy) -> Self {
        Self { entries: HashMap::new(), seen: HashSet::new(), policy, startup: None,
            clock: 0, stats: GraphStats::default(), measured_bytes: 0, measured: 0,
            frozen: None, retired: Vec::new() }
    }

    /// Warm exactly the planner's key set. Duplicate keys, missing captures or
    /// unexpected captures fail startup rather than merely logging a mismatch.
    pub fn warm(&mut self, keys: &[K], mut capture: impl FnMut(&K) -> anyhow::Result<(E, Option<u64>)>) -> anyhow::Result<()> {
        anyhow::ensure!(self.entries.is_empty() && self.startup.is_none(), "graph bank already warmed");
        let expected: HashSet<K> = keys.iter().cloned().collect();
        anyhow::ensure!(expected.len() == keys.len(), "duplicate startup graph key");
        for key in keys {
            let (exec, bytes) = capture(key)?;
            self.insert(key.clone(), exec, bytes);
        }
        anyhow::ensure!(self.entries.len() == expected.len()
            && expected.iter().all(|key| self.contains(key)), "startup graph inventory differs from captured bank");
        self.startup = Some(expected);
        Ok(())
    }

    /// Seal a family-driven warm-up (some captures encompass a whole step).
    pub fn seal_startup(&mut self, keys: &[K]) -> anyhow::Result<()> {
        let expected: HashSet<K> = keys.iter().cloned().collect();
        anyhow::ensure!(expected.len() == keys.len(), "duplicate startup graph key");
        anyhow::ensure!(self.startup.is_none(), "startup graph bank already sealed");
        anyhow::ensure!(self.entries.len() == expected.len()
            && expected.iter().all(|key| self.contains(key)), "startup graph inventory differs from captured bank");
        self.startup = Some(expected);
        Ok(())
    }

    pub fn enqueue(&mut self, key: &K) -> (GraphDecision, Option<&E>) {
        let decision = if self.entries.contains_key(key) { GraphDecision::Replay }
            else if self.policy == GraphPolicy::FixedStartup { GraphDecision::Eager }
            else { GraphDecision::Capture };
        if decision == GraphDecision::Eager {
            self.stats.eager_runs += 1;
            let count = self.stats.eager_runs;
            if count <= 8 || count.is_power_of_two() {
                tracing::info!(target: "cuteafd::graph_capture", bank = self as *const Self as usize, ?key, eager_runs = count,
                    "fixed startup unknown graph key executed eagerly");
            }
        }
        (decision, self.launch(key))
    }

    pub fn set_budget(&mut self, budget: Option<u64>) {
        if matches!(self.policy, GraphPolicy::Budgeted { .. }) {
            self.policy = GraphPolicy::Budgeted { bytes: budget };
        }
    }

    pub fn budget(&self) -> Option<u64> {
        match self.policy { GraphPolicy::Budgeted { bytes } => bytes, GraphPolicy::FixedStartup => None }
    }

    pub fn calibrating(&self) -> bool { self.frozen.is_none() }
    pub fn each(&self) -> u64 {
        self.frozen.unwrap_or_else(|| self.measured_bytes.div_ceil(self.measured.max(1)))
    }

    pub fn launch(&mut self, key: &K) -> Option<&E> {
        self.clock += 1;
        let clock = self.clock;
        self.entries.get_mut(key).map(|entry| { entry.launched = clock; &entry.exec })
    }

    pub fn get(&self, key: &K) -> Option<&E> { self.entries.get(key).map(|entry| &entry.exec) }
    pub fn seen(&self, key: &K) -> bool { self.seen.contains(key) }

    /// Retired executables remain owned here until drain_retired. The mean
    /// physical-memory charge freezes at first retirement: later captures can
    /// reuse allocator chunks and no longer report their real size as a delta.
    pub fn insert(&mut self, key: K, exec: E, measured: Option<u64>) {
        self.clock += 1;
        self.stats.captures += 1;
        if !self.seen.insert(key.clone()) { self.stats.recaptures += 1; }
        if let (None, Some(bytes)) = (self.frozen, measured) {
            self.measured_bytes += bytes;
            self.measured += 1;
        }
        if let Some(old) = self.entries.insert(key.clone(), Entry { exec, launched: self.clock }) {
            self.freeze();
            self.retired.push(old.exec);
        }
        let Some(budget) = self.budget() else { return };
        if self.bytes() <= budget { return }
        self.freeze();
        let mut order: Vec<(u64, K)> = self.entries.iter().filter(|(k, _)| **k != key)
            .map(|(k, entry)| (entry.launched, k.clone())).collect();
        order.sort_unstable_by_key(|(launched, _)| *launched);
        for (_, victim) in order {
            if self.bytes() <= budget { break }
            if let Some(entry) = self.entries.remove(&victim) {
                self.stats.evictions += 1;
                self.retired.push(entry.exec);
            }
        }
    }

    fn freeze(&mut self) { self.frozen.get_or_insert(self.measured_bytes.div_ceil(self.measured.max(1))); }

    pub fn retire(&mut self, key: &K) {
        if let Some(entry) = self.entries.remove(key) {
            self.freeze();
            self.retired.push(entry.exec);
        }
    }

    pub fn retire_oldest(&mut self, matches: impl Fn(&K) -> bool) {
        if let Some(key) = self.entries.iter().filter(|(key, _)| matches(key))
            .min_by_key(|(_, entry)| entry.launched).map(|(key, _)| key.clone()) {
            self.retire(&key);
        }
    }

    pub fn retire_matching(&mut self, matches: impl Fn(&K) -> bool) {
        let keys: Vec<_> = self.entries.keys().filter(|key| matches(key)).cloned().collect();
        for key in keys { self.retire(&key); }
    }

    pub fn retire_all(&mut self) {
        self.freeze();
        self.retired.extend(self.entries.drain().map(|(_, entry)| entry.exec));
    }

    pub fn retired_len(&self) -> usize { self.retired.len() }

    /// Call only after every launch referencing a retired executable drained.
    /// A drain failure leaves all resource owners in the bank for a later retry.
    pub fn drain_retired(&mut self, drain: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<Vec<E>> {
        if self.retired.is_empty() { return Ok(Vec::new()) }
        drain()?;
        Ok(std::mem::take(&mut self.retired))
    }

    pub fn bytes(&self) -> u64 { self.entries.len() as u64 * self.each() }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn contains(&self, key: &K) -> bool { self.entries.contains_key(key) }
    pub fn count(&self, matches: impl Fn(&K) -> bool) -> usize {
        self.entries.keys().filter(|key| matches(key)).count()
    }
    pub fn stats(&self) -> GraphStats { self.stats }
}

impl<K, E> Drop for GraphBank<K, E> {
    fn drop(&mut self) {
        if self.policy == GraphPolicy::FixedStartup && self.stats.eager_runs > 0 {
            tracing::info!(target: "cuteafd::graph_capture", bank = self as *const Self as usize,
                eager_runs = self.stats.eager_runs, captures = self.stats.captures,
                "fixed startup graph bank final counters");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_startup_warms_exact_keys_and_misses_execute_eagerly() {
        let mut bank = GraphBank::with_policy(GraphPolicy::FixedStartup);
        bank.warm(&[1, 6, 16], |key| Ok((*key, Some(10)))).unwrap();
        assert_eq!(bank.enqueue(&6), (GraphDecision::Replay, Some(&6)));
        assert_eq!(bank.enqueue(&3), (GraphDecision::Eager, None));
        assert_eq!(bank.stats().captures, 3);
        assert_eq!(bank.stats().eager_runs, 1);
        bank.enqueue(&3);
        assert_eq!(bank.stats().eager_runs, 2);
        bank.set_budget(Some(100));
        assert_eq!(bank.enqueue(&4), (GraphDecision::Eager, None));
        assert_eq!(bank.stats().eager_runs, 3);
        assert!(bank.warm(&[], |_| Ok((0, None))).is_err());
        let mut duplicate = GraphBank::with_policy(GraphPolicy::FixedStartup);
        assert!(duplicate.warm(&[1, 1], |key| Ok((*key, None))).is_err());
    }
    #[test]
    fn startup_inventory_fails_closed_on_missing_or_extra_keys() {
        let mut bank = GraphBank::new(None);
        bank.insert(1, 1, None);
        assert!(bank.seal_startup(&[1, 2]).is_err());
        assert!(bank.seal_startup(&[]).is_err());
        bank.seal_startup(&[1]).unwrap();
    }
    #[test]
    fn retirement_keeps_owners_when_stream_drain_fails() {
        let mut bank = GraphBank::new(Some(20));
        for key in 0..3 { bank.insert(key, key, Some(10)); }
        assert_eq!((bank.len(), bank.retired_len()), (2, 1));
        assert!(bank.drain_retired(|| anyhow::bail!("pending")).is_err());
        assert_eq!(bank.retired_len(), 1);
        assert_eq!(bank.drain_retired(|| Ok(())).unwrap(), vec![0]);
        bank.retire_all();
        assert_eq!(bank.len(), 0);
        assert_eq!(bank.drain_retired(|| Ok(())).unwrap().len(), 2);
    }
    #[test]
    fn replacement_freezes_mean_before_allocator_reuse() {
        let mut bank = GraphBank::new(None);
        bank.insert(1, 10, Some(4));
        bank.insert(1, 11, Some(6));
        assert_eq!(bank.each(), 5);
        bank.insert(2, 12, Some(0));
        assert_eq!(bank.each(), 5);
        assert_eq!(bank.drain_retired(|| Ok(())).unwrap(), vec![10]);
    }
    #[test]
    fn budget_uses_frozen_mean_and_lru() {
        let mut bank = GraphBank::new(Some(30));
        for key in 0..3 { bank.insert(key, key, Some(10)); }
        bank.launch(&0);
        bank.insert(3, 3, Some(10));
        assert_eq!(bank.drain_retired(|| Ok(())).unwrap(), vec![1]);
        for key in 4..1000 {
            bank.insert(key, key, Some(0));
            bank.drain_retired(|| Ok(())).unwrap();
            assert_eq!((bank.len(), bank.bytes(), bank.each()), (3, 30, 10));
        }
        assert_eq!(bank.enqueue(&1), (GraphDecision::Capture, None));
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    fn insert<K: Hash + Eq + Clone + std::fmt::Debug, E>(bank: &mut GraphBank<K, E>, key: K, exec: E, bytes: Option<u64>) -> Vec<E> {
        bank.insert(key, exec, bytes);
        bank.drain_retired(|| Ok(())).unwrap()
    }

    const MIB: u64 = 1 << 20;

    #[test]
    fn unbounded_caches_keep_every_executable() {
        let mut cache = GraphBank::new(None);
        for key in 0..100u32 {
            assert!(insert(&mut cache, key, key, Some(if key % 14 == 0 { 2 * MIB } else { 0 })).is_empty());
        }
        assert_eq!(cache.len(), 100);
        assert_eq!(cache.each(), (8 * 2 * MIB).div_ceil(100));
        assert_eq!(cache.bytes(), 100 * cache.each());
        assert_eq!(cache.launch(&7), Some(&7));
        assert_eq!(cache.stats(), GraphStats { captures: 100, ..Default::default() });
        assert!(cache.calibrating());
    }

    #[test]
    fn executables_are_charged_the_mean_of_chunked_free_memory_deltas() {
        // Free memory moves in 2 MiB chunks: of 14 executables of ~146 KiB, one capture takes a
        // chunk and 13 take none. Charging each its own delta would let the 13 stay free of charge.
        let mut cache = GraphBank::new(None);
        for key in 0..1400u32 {
            insert(&mut cache, key, key, Some(if key % 14 == 13 { 2 * MIB } else { 0 }));
        }
        assert_eq!(cache.each(), (2 * MIB * 100).div_ceil(1400));
        assert_eq!(cache.bytes(), 1400 * cache.each());
        assert!(cache.bytes() >= 200 * MIB && cache.bytes() < 200 * MIB + 1400);
    }

    #[test]
    fn evictions_bound_the_executables_even_when_new_captures_reuse_freed_memory() {
        // The measured failure: once evictions free room inside the allocator's chunks, new
        // captures fill it and their free-memory deltas read 0. The cache must keep counting them.
        let (budget, each) = (64 * MIB, 147_000u64);
        let mut cache = GraphBank::new(Some(budget));
        let mut key = 0u32;
        let mut evicted = 0usize;
        while cache.calibrating() {
            evicted += insert(&mut cache, key, key, Some(each)).len();
            key += 1;
        }
        assert_eq!(cache.each(), each);
        let held = cache.len();
        assert_eq!(held as u64, budget / each);
        for _ in 0..20_000 {
            evicted += insert(&mut cache, key, key, Some(0)).len();
            key += 1;
            assert!(cache.bytes() <= budget && cache.len() == held, "held {} executables", cache.len());
        }
        assert_eq!(evicted as u64, cache.stats().evictions);
        assert_eq!(cache.stats().captures, u64::from(key));
        assert_eq!(cache.each(), each, "the size stays frozen at the first eviction");
    }

    #[test]
    fn the_least_recently_launched_leave_first_and_never_the_new_one() {
        let mut cache = GraphBank::new(Some(3 * MIB));
        for key in 0..3u32 {
            assert!(insert(&mut cache, key, key, Some(MIB)).is_empty());
        }
        // Launching 0 makes 1 the least recently launched.
        assert_eq!(cache.launch(&0), Some(&0));
        assert_eq!(insert(&mut cache, 3, 3, Some(MIB)), vec![1]);
        assert_eq!(cache.launch(&1), None);
        assert_eq!((cache.len(), cache.bytes(), cache.each()), (3, 3 * MIB, MIB));
        // Frozen: a capture that measures more is still charged one executable.
        assert_eq!(insert(&mut cache, 4, 4, Some(8 * MIB)), vec![2]);
        assert_eq!((cache.len(), cache.bytes()), (3, 3 * MIB));
        assert_eq!(cache.stats(), GraphStats { captures: 5, recaptures: 0, evictions: 2, eager_runs: 0 });
    }

    #[test]
    fn a_key_captured_again_after_eviction_is_a_recapture() {
        let mut cache = GraphBank::new(Some(2 * MIB));
        insert(&mut cache, 1u32, 1, Some(MIB));
        insert(&mut cache, 2, 2, Some(MIB));
        assert_eq!(insert(&mut cache, 3, 3, Some(MIB)), vec![1]);
        assert!(cache.launch(&1).is_none());
        assert!(cache.seen(&1) && !cache.seen(&4));
        assert_eq!(insert(&mut cache, 1, 1, None), vec![2]);
        assert_eq!(cache.stats().recaptures, 1);
        assert_eq!(cache.count(|key| *key % 2 == 1), 2);
        cache.set_budget(None);
        assert!(insert(&mut cache, 10, 10, None).is_empty());
        assert_eq!(cache.budget(), None);
    }

    #[test]
    fn replacing_a_key_returns_the_old_executable() {
        let mut cache = GraphBank::new(None);
        insert(&mut cache, 1u32, 10, Some(4));
        assert_eq!(insert(&mut cache, 1, 11, Some(6)), vec![10]);
        assert_eq!((cache.len(), cache.launch(&1)), (1, Some(&11)));
    }
}
