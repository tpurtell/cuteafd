//! Exact-key graph storage. Retirement never destroys an executable until its
//! caller proves every stream using the captured storage has drained.
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use cuteafd_loader::placement::inventory::DeclaredGraphSet;

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

/// One distinct shape a sealed startup set did not contain, with every eager
/// execution of it. Kept process-wide rather than per bank so /v1/stats reports
/// it without threading a bank handle through the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MissingVariant {
    pub family: &'static str,
    pub shape: String,
    pub runs: u64,
}

static MISSING: std::sync::Mutex<Vec<MissingVariant>> = std::sync::Mutex::new(Vec::new());

/// Counts one eager execution of `shape` and reports whether it is the first,
/// which is the only occurrence that warns. Every caller goes through
/// `note_eager`, so misses are recorded and warned about in one place; a miss is
/// a mismatch between the sealed startup inventory and real traffic, so it is
/// rare by construction.
fn note_missing(family: &'static str, shape: &str) -> bool {
    let mut missing = MISSING.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    match missing.iter_mut().find(|entry| entry.family == family && entry.shape == shape) {
        Some(entry) => { entry.runs += 1; false }
        None => { missing.push(MissingVariant { family, shape: shape.to_owned(), runs: 1 }); true }
    }
}

/// Every distinct startup-set miss, in first-seen order (its /v1/stats form).
pub(crate) fn missing_variants() -> Vec<MissingVariant> {
    MISSING.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
}

/// Counts one eager execution of `shape` outside a sealed startup set and
/// reports whether it is the first for this family, which is the only one that
/// warns. This is the single entry point for eager work: owners that hold their
/// own executable and decide eager themselves (Qwen) call it directly, and
/// `GraphBank::enqueue_with_context` calls it too, so every path lands in the
/// same /v1/stats entry. The `context` is built only on a first miss.
pub(crate) fn note_eager(family: &'static str, shape: &str, context: impl FnOnce() -> String) -> bool {
    if !note_missing(family, shape) { return false; }
    // Built outside the macro so the description is produced even when the
    // current subscriber filters this target out.
    let context = context();
    tracing::warn!(target: "cuteafd::graph_capture", family = family, key = %shape,
        context = %context,
        "graph shape outside the startup set ran eagerly; consider adding it to the startup graph set");
    true
}

struct Entry<E> {
    exec: E,
    launched: u64,
    cached: Option<(u64, bool)>,
    pinned_slot: Option<usize>,
}

struct DeviceBinding {
    cache: super::device::SharedCache,
    bank: u64,
    site: super::device::Site,
    pinned_hits: u64,
    pinned_counts: Vec<(u64, u64)>,
}

impl DeviceBinding {
    fn flush(&mut self, cache: &mut super::device::DeviceCache) {
        if std::mem::take(&mut self.pinned_hits) == 0 { return }
        cache.sample_clock();
        for (id, count) in &mut self.pinned_counts {
            cache.pinned_hits(*id, self.site.cycle, std::mem::take(count));
        }
    }
}

pub(crate) struct CaptureTicket {
    cache: super::device::SharedCache,
    id: u64,
    pinned: bool,
    committed: bool,
}

impl Drop for CaptureTicket {
    fn drop(&mut self) {
        if !self.committed {
            self.cache.lock().unwrap_or_else(|p| p.into_inner()).cancel_capture(self.id);
        }
    }
}

/// E owns the capture's GraphOwner (including its resource pins). The bank is
/// independent of the native executable type, allowing CPU-only policy tests.
pub(crate) struct GraphBank<K, E> {
    entries: HashMap<K, Entry<E>>,
    cache_keys: HashMap<K, super::device::Identity>,
    next_cache_key: u64,
    cache_key_generation: u64,
    cache_keys_dirty: bool,
    seen: HashSet<K>,
    policy: GraphPolicy,
    family: &'static str,
    startup: Option<DeclaredGraphSet<K>>,
    startup_sealed: bool,
    clock: u64,
    stats: GraphStats,
    measured_bytes: u64,
    measured: u64,
    frozen: Option<u64>,
    retired: Vec<E>,
    device: Option<DeviceBinding>,
}

impl<K: Hash + Eq + Clone + std::fmt::Debug, E> GraphBank<K, E> {
    pub fn new(budget: Option<u64>) -> Self {
        Self::with_policy(GraphPolicy::Budgeted { bytes: budget })
    }

    pub fn with_policy(policy: GraphPolicy) -> Self {
        Self::with_family(policy, "shared")
    }

    /// Names the owning family in startup-miss warnings and /v1/stats, so a
    /// shared bank's miss is attributable without looking up its construction.
    pub fn with_family(policy: GraphPolicy, family: &'static str) -> Self {
        Self { entries: HashMap::new(), cache_keys: HashMap::new(), next_cache_key: 0, cache_key_generation: 0, cache_keys_dirty: false, seen: HashSet::new(), policy, family, startup: None, startup_sealed: false,
            clock: 0, stats: GraphStats::default(), measured_bytes: 0, measured: 0,
            frozen: None, retired: Vec::new(), device: None }
    }

    pub fn attach_device(&mut self, device: i32, reserve: u64, site: super::device::Site) -> anyhow::Result<()> {
        anyhow::ensure!(self.entries.is_empty() && self.device.is_none(), "graph bank already attached or populated");
        let cache = super::device::device_cache(device, reserve, site.basis)?;
        let bank = cache.lock().unwrap_or_else(|p| p.into_inner()).bank();
        self.device = Some(DeviceBinding { cache, bank, site, pinned_hits: 0, pinned_counts: Vec::new() });
        Ok(())
    }

    /// Change attribution at a whole-cycle boundary, retaining the pinned-hit
    /// fast path and flushing its previous cycle before switching.
    pub fn cache_cycle(&mut self, cycle: super::device::Cycle) {
        if let Some(binding) = &mut self.device {
            let shared = binding.cache.clone();
            let mut cache = shared.lock().unwrap_or_else(|p| p.into_inner());
            cache.sample_clock();
            if binding.site.cycle != cycle {
                binding.flush(&mut cache);
                binding.site.cycle = cycle;
            }
        }
    }

    pub fn cached_launch(&mut self, key: &K) -> Option<&E> {
        let entry = self.entries.get_mut(key)?;
        if let (Some(binding), Some((id, pinned))) = (&mut self.device, entry.cached) {
            if pinned {
                binding.pinned_hits += 1;
                binding.pinned_counts[entry.pinned_slot.expect("pinned cache slot")].1 += 1;
            } else {
                binding.cache.lock().unwrap_or_else(|p| p.into_inner()).hit_cycle(id, binding.site.cycle);
            }
        }
        Some(&entry.exec)
    }

    pub fn cache_miss(&mut self, key: &K, pinned: bool, node_estimate: Option<super::device::Estimate>) -> super::device::Miss {
        let Some(binding) = &self.device else { return super::device::Miss::Eager };
        let identity = if let Some(identity) = self.cache_keys.get(key) { identity.clone() } else {
            self.next_cache_key = self.next_cache_key.checked_add(1).expect("graph key identity exhausted");
            let identity = super::device::Identity { bank: binding.bank, site: binding.site.name,
                key: self.next_cache_key, shape: format!("{key:?}").into() };
            self.cache_keys.insert(key.clone(), identity.clone());
            self.cache_keys_dirty = true;
            identity
        };
        binding.cache.lock().unwrap_or_else(|p| p.into_inner()).miss(identity, binding.site, node_estimate, pinned)
    }

    pub fn cache_insert(&mut self, key: K, exec: E, id: u64, pinned: bool) {
        let pinned_slot = if pinned {
            self.device.as_mut().map(|binding| {
                let slot = binding.pinned_counts.len();
                binding.pinned_counts.push((id, 0));
                slot
            })
        } else { None };
        self.entries.insert(key, Entry { exec, launched: 0, cached: Some((id, pinned)), pinned_slot });
    }

    pub fn cache_attached(&self) -> bool { self.device.is_some() }

    /// A capture reservation independent of the bank borrow. Dropping it on any
    /// failed path cancels the ticket, without synchronizing a peer segment.
    pub fn cache_ticket(&mut self, key: &K, pinned: bool, estimate: Option<super::device::Estimate>) -> Option<CaptureTicket> {
        let super::device::Miss::Capture(id) = self.cache_miss(key, pinned, estimate) else { return None };
        Some(CaptureTicket { cache: self.device.as_ref()?.cache.clone(), id, pinned, committed: false })
    }

    pub fn cache_commit(&mut self, key: K, exec: E, mut ticket: CaptureTicket,
        nodes: u64, width: usize, spans_devices: bool) {
        self.cache_topology(ticket.id, nodes, width, spans_devices);
        self.cache_insert(key, exec, ticket.id, ticket.pinned);
        ticket.committed = true;
    }

    pub fn cache_cancel(&mut self, id: u64) {
        if let Some(binding) = &self.device { binding.cache.lock().unwrap_or_else(|p| p.into_inner()).cancel_capture(id); }
    }

    pub fn cache_topology(&mut self, id: u64, nodes: u64, width: usize, spans_devices: bool) {
        if let Some(binding) = &self.device {
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).captured_topology(id, nodes, width, spans_devices);
        }
    }

    /// Explicit teardown only, after the caller drains all consumers.
    pub fn cache_clear_drained(&mut self) {
        self.entries.clear();
        self.cache_keys.clear();
        self.retired.clear();
        if let Some(binding) = &mut self.device {
            binding.pinned_hits = 0;
            binding.pinned_counts.clear();
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).remove_bank(binding.bank);
        }
    }

    pub fn cache_tick(&mut self) {
        if let Some(binding) = &mut self.device {
            let shared = binding.cache.clone();
            let mut cache = shared.lock().unwrap_or_else(|p| p.into_inner());
            binding.flush(&mut cache);
            if self.cache_keys_dirty || self.cache_key_generation != cache.identity_generation() {
                self.cache_keys.retain(|_, identity| cache.retains_identity(identity));
                self.cache_keys_dirty = false;
                self.cache_key_generation = cache.identity_generation();
            }
            cache.boundary();
        }
    }

    pub fn cache_needs_boundary(&self) -> bool {
        self.device.as_ref().is_some_and(|binding| binding.cache.lock().unwrap_or_else(|p| p.into_inner()).needs_boundary())
    }

    pub fn cache_measured(&mut self, bytes: u64) {
        if let Some(binding) = &mut self.device {
            let shared = binding.cache.clone();
            let mut cache = shared.lock().unwrap_or_else(|p| p.into_inner());
            binding.flush(&mut cache);
            cache.measured_capture(bytes);
        }
    }

    /// The caller drains ALL device consumers at a cycle boundary first. The
    /// callback samples after destruction; failed drains retain every owner.
    pub fn cache_drain(&mut self, drain: impl FnOnce() -> anyhow::Result<()>,
        measure: impl FnOnce() -> anyhow::Result<super::device::ReleaseMeasurement>) -> anyhow::Result<()> {
        let Some(binding) = &mut self.device else { return Ok(()) };
        let victims: Vec<_> = {
            let shared = binding.cache.clone();
            let mut cache = shared.lock().unwrap_or_else(|p| p.into_inner());
            binding.flush(&mut cache);
            self.entries.iter().filter_map(|(key, entry)| entry.cached
                .filter(|(id, _)| cache.pending(*id)).map(|(id, _)| (key.clone(), id))).collect()
        };
        if !victims.is_empty() {
            drain()?;
            for (key, _) in &victims { if let Some(entry) = self.entries.remove(key) { self.retired.push(entry.exec); } }
            // drain_retired's ordering, with the already completed drain proof.
            self.retired.clear();
            let measurement = measure()?;
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).destroyed(
                &victims.iter().map(|(_, id)| *id).collect::<Vec<_>>(), measurement);
        }
        let mut cache = binding.cache.lock().unwrap_or_else(|p| p.into_inner());
        if self.cache_keys_dirty || self.cache_key_generation != cache.identity_generation() {
            self.cache_keys.retain(|_, identity| cache.retains_identity(identity));
            self.cache_keys_dirty = false;
            self.cache_key_generation = cache.identity_generation();
        }
        cache.boundary();
        Ok(())
    }

    /// Declare the planner's exact inventory before owner-local capture begins.
    /// Until sealed, only declared keys may capture; after sealing every miss is
    /// eager, including a retired hot key. Capture permission is never inferred
    /// from an empty slot on a live request.
    pub fn declare_startup(&mut self, keys: &[K]) -> anyhow::Result<()> {
        anyhow::ensure!(self.entries.is_empty() && self.startup.is_none(), "graph bank already warmed");
        let expected = DeclaredGraphSet::new(keys).map_err(anyhow::Error::msg)?;
        self.policy = GraphPolicy::FixedStartup;
        self.startup = Some(expected);
        Ok(())
    }

    pub fn startup_declared(&self) -> bool { self.startup.is_some() }

    pub fn census_capture_key(&self, bank: &str, rank: i32, lane: Option<usize>, key: &K,
        exec: *mut std::ffi::c_void) {
        super::census::capture_key(self.family, bank, self as *const Self as usize, rank, lane, key, exec);
    }

    pub fn census_capture_device_key(&self, bank: &str, device: i32, key: &K,
        exec: *mut std::ffi::c_void) {
        super::census::capture_device_key(self.family, bank, self as *const Self as usize,
            device, None, key, exec);
    }

    pub fn capture_allowed(&self, key: &K) -> bool {
        !self.startup_sealed && self.startup.as_ref().map_or(
            self.policy != GraphPolicy::FixedStartup, |keys| keys.contains(key))
    }

    /// Validate the declared inventory against actual retained executables.
    pub fn finish_startup(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.startup_sealed, "startup graph bank already sealed");
        let expected = self.startup.as_ref().ok_or_else(|| anyhow::anyhow!("startup graph inventory undeclared"))?;
        expected.verify(self.entries.keys()).map_err(anyhow::Error::msg)?;
        self.startup_sealed = true;
        Ok(())
    }

    /// Warm exactly the planner's key set. Duplicate keys, missing captures or
    /// unexpected captures fail startup rather than merely logging a mismatch.
    pub fn warm(&mut self, keys: &[K], mut capture: impl FnMut(&K) -> anyhow::Result<(E, Option<u64>)>) -> anyhow::Result<()> {
        anyhow::ensure!(self.entries.is_empty() && self.startup.is_none(), "graph bank already warmed");
        let expected = DeclaredGraphSet::new(keys).map_err(anyhow::Error::msg)?;
        for key in keys {
            let (exec, bytes) = capture(key)?;
            self.insert(key.clone(), exec, bytes);
        }
        expected.verify(self.entries.keys()).map_err(anyhow::Error::msg)?;
        self.startup = Some(expected);
        self.policy = GraphPolicy::FixedStartup;
        self.startup_sealed = true;
        Ok(())
    }

    /// Seal a family-driven warm-up (some captures encompass a whole step).
    pub fn seal_startup(&mut self, keys: &[K]) -> anyhow::Result<()> {
        let expected = DeclaredGraphSet::new(keys).map_err(anyhow::Error::msg)?;
        anyhow::ensure!(self.startup.is_none(), "startup graph bank already sealed");
        expected.verify(self.entries.keys()).map_err(anyhow::Error::msg)?;
        self.startup = Some(expected);
        self.policy = GraphPolicy::FixedStartup;
        self.startup_sealed = true;
        Ok(())
    }

    pub fn enqueue(&mut self, key: &K) -> (GraphDecision, Option<&E>) {
        self.enqueue_with_context(key, String::new)
    }

    /// A sealed startup miss runs eagerly and already costs the request its
    /// graphed path, so it warns once per distinct variant with the family, the
    /// shape key and the caller's request context. `context` is evaluated only
    /// on that first miss, leaving replay and warm-startup paths untouched.
    pub fn enqueue_with_context(&mut self, key: &K, context: impl FnOnce() -> String)
        -> (GraphDecision, Option<&E>) {
        let decision = if self.entries.contains_key(key) { GraphDecision::Replay }
            else if self.capture_allowed(key) { GraphDecision::Capture }
            else { GraphDecision::Eager };
        if decision == GraphDecision::Eager {
            self.stats.eager_runs += 1;
            // Eager requires a sealed (or declared) startup set, so every miss
            // is a shape the planner's inventory lacked.
            note_eager(self.family, &format!("{key:?}"), context);
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
        if let Some(old) = self.entries.insert(key.clone(), Entry { exec, launched: self.clock, cached: None, pinned_slot: None }) {
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
    pub fn keys(&self) -> Vec<K> { self.entries.keys().cloned().collect() }
    pub fn count(&self, matches: impl Fn(&K) -> bool) -> usize {
        self.entries.keys().filter(|key| matches(key)).count()
    }
    pub fn stats(&self) -> GraphStats { self.stats }
}

impl<K, E> Drop for GraphBank<K, E> {
    fn drop(&mut self) {
        if let Some(binding) = &mut self.device {
            binding.pinned_hits = 0;
            binding.pinned_counts.clear();
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).remove_bank(binding.bank);
        }
        if self.policy == GraphPolicy::FixedStartup && self.stats.eager_runs > 0 {
            tracing::info!(target: "cuteafd::graph_capture", family = self.family, bank = self as *const Self as usize,
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
    fn sealed_miss_counts_once_per_variant_and_builds_context_only_for_the_first() {
        const FAMILY: &str = "bank-test-startup-miss";
        let mut bank = GraphBank::with_family(GraphPolicy::FixedStartup, FAMILY);
        bank.warm(&[1u32], |key| Ok((*key, Some(10)))).unwrap();
        assert_eq!(bank.enqueue_with_context(&1, || panic!("replay must not build a context")).0,
            GraphDecision::Replay);
        let contexts = std::cell::Cell::new(0);
        for _ in 0..3 {
            assert_eq!(bank.enqueue_with_context(&2, || { contexts.set(contexts.get() + 1); "batch=2".into() }),
                (GraphDecision::Eager, None));
        }
        assert_eq!(bank.enqueue(&3).0, GraphDecision::Eager);
        assert_eq!(contexts.get(), 1, "only a variant's first miss describes its request");
        let ours: Vec<_> = missing_variants().into_iter().filter(|miss| miss.family == FAMILY).collect();
        assert_eq!(ours.iter().map(|miss| (miss.shape.as_str(), miss.runs)).collect::<Vec<_>>(),
            vec![("2", 3), ("3", 1)]);
        assert_eq!(bank.stats().eager_runs, 4);
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
    fn declared_inventory_only_admits_hot_keys_until_sealed() {
        let mut bank = GraphBank::new(Some(1));
        bank.declare_startup(&[1, 2]).unwrap();
        assert_eq!(bank.enqueue(&1), (GraphDecision::Capture, None));
        assert_eq!(bank.enqueue(&3), (GraphDecision::Eager, None));
        bank.insert(1, 10, None);
        assert!(bank.finish_startup().is_err());
        bank.insert(2, 20, None);
        bank.finish_startup().unwrap();
        assert_eq!(bank.enqueue(&1), (GraphDecision::Replay, Some(&10)));
        assert_eq!(bank.enqueue(&3), (GraphDecision::Eager, None));
        bank.retire(&1);
        assert_eq!(bank.enqueue(&1), (GraphDecision::Eager, None));
        assert!(!bank.capture_allowed(&2));
        assert!(bank.finish_startup().is_err());
        assert!(bank.declare_startup(&[1, 2]).is_err());
    }

    #[test]
    fn sealed_prefill_inventory_retains_alternating_tails_and_rejects_cold_shapes() {
        let mut bank = GraphBank::new(Some(1));
        let keys: Vec<_> = (0..40).flat_map(|layer| (1..=64).map(move |rows| (layer, rows))).collect();
        bank.declare_startup(&keys).unwrap();
        for &key in &keys { bank.insert(key, key, None); }
        bank.finish_startup().unwrap();
        for rows in [54, 52, 54, 52, 56, 53, 2752, 579, 4, 128, 65] {
            for layer in 0..40 {
                let key = (layer, rows);
                if rows <= 64 {
                    assert_eq!(bank.enqueue(&key), (GraphDecision::Replay, Some(&key)));
                } else {
                    assert_eq!(bank.enqueue(&key), (GraphDecision::Eager, None));
                }
                assert!(!bank.capture_allowed(&key));
            }
        }
        assert_eq!(bank.len(), keys.len());
        assert_eq!(bank.retired_len(), 0);
        assert_eq!(bank.stats().captures, keys.len() as u64);
        assert_eq!(bank.stats().recaptures, 0);
        assert_eq!(bank.stats().evictions, 0);
    }

    #[test]
    fn sealed_empty_prefill_inventory_never_captures_arbitrary_tails() {
        let mut bank = GraphBank::<(usize, usize), usize>::new(None);
        bank.declare_startup(&[]).unwrap();
        bank.finish_startup().unwrap();
        for rows in (1..=128).chain([301, 552, 808, 810, 2723, 2752]) {
            for layer in 0..40 {
                let key = (layer, rows);
                assert_eq!(bank.enqueue(&key), (GraphDecision::Eager, None));
                assert!(!bank.capture_allowed(&key));
            }
        }
        assert_eq!(bank.len(), 0);
        assert_eq!(bank.stats().captures, 0);
        assert_eq!(bank.stats().evictions, 0);
    }

    #[test]
    fn sealed_family_warmup_disallows_live_capture() {
        let mut bank = GraphBank::new(None);
        bank.insert(1, 10, None);
        bank.seal_startup(&[1]).unwrap();
        bank.set_budget(Some(0));
        assert_eq!(bank.enqueue(&2), (GraphDecision::Eager, None));
        assert!(!bank.capture_allowed(&1));
        let mut duplicate: GraphBank<u32, u32> = GraphBank::new(None);
        assert!(duplicate.declare_startup(&[1, 1]).is_err());
        assert_eq!(duplicate.enqueue(&1), (GraphDecision::Capture, None));
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

#[cfg(test)]
mod device_tests {
    use super::*;
    use super::super::device::{Cycle, Miss, Preference, ReleaseMeasurement, Site};
    fn site() -> Site { Site { name: "bank-test", cycle: Cycle::Decode, preference: Preference::Capture, bootstrap: 10, basis: super::super::device::Basis::Bytes, spans_devices: false } }
    #[test]
    fn device_eviction_destroys_only_after_successful_drain() {
        use std::{cell::Cell, rc::Rc};
        struct Exec(Rc<Cell<usize>>);
        impl Drop for Exec { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        let drops = Rc::new(Cell::new(0));
        let mut bank = GraphBank::new(None);
        bank.attach_device(-301, 20, site()).unwrap();
        let Miss::Capture(id) = bank.cache_miss(&1, false, None) else { panic!() };
        bank.cache_insert(1, Exec(drops.clone()), id, false);
        bank.cache_measured(10);
        // Same proven kind in another bank needs this graph's slot.
        let cache = bank.device.as_ref().unwrap().cache.clone();
        let Miss::Eager = cache.lock().unwrap().miss(super::super::device::Identity {
            bank: 99, site: "bank-test", key: 1, shape: "1".into() }, site(), None, false) else { panic!() };
        assert!(bank.cache_drain(|| anyhow::bail!("not drained"), || unreachable!()).is_err());
        assert_eq!(drops.get(), 0);
        bank.cache_drain(|| { assert_eq!(drops.get(), 0); Ok(()) }, || {
            assert_eq!(drops.get(), 1); Ok(ReleaseMeasurement { released: 0 })
        }).unwrap();
    }

    #[test]
    fn cached_graph_hits_follow_the_current_cycle_including_pinned_fast_path() {
        let mut bank = GraphBank::new(None);
        bank.attach_device(-303, 100, site()).unwrap();
        for (key, pinned) in [(1, true), (2, false)] {
            let Miss::Capture(id) = bank.cache_miss(&key, pinned, None) else { panic!() };
            bank.cache_insert(key, key, id, pinned);
            bank.cache_measured(10);
        }
        bank.cache_cycle(Cycle::Verify);
        assert_eq!(bank.cached_launch(&1), Some(&1));
        assert_eq!(bank.cached_launch(&2), Some(&2));
        bank.cache_cycle(Cycle::Decode);
        let cache = bank.device.as_ref().unwrap().cache.clone();
        assert_eq!(cache.lock().unwrap().cycle_hits(Cycle::Verify), 2);
        assert_eq!(cache.lock().unwrap().cycle_hits(Cycle::Decode), 0);
        bank.cached_launch(&1);
        bank.cache_tick();
        let rows = super::super::device::snapshots();
        let snapshot = rows.iter().find(|row| row.device == -303).unwrap();
        let json = serde_json::to_value(snapshot).unwrap();
        let pinned = json["top_hits"].as_array().unwrap().iter().find(|row| row["pinned"] == true).unwrap();
        assert_eq!(pinned["hits"], 2);
        assert_eq!(pinned["cycles"]["verify"], 1);
        assert_eq!(pinned["cycles"]["decode"], 2); // Initial capture miss plus replay.
        assert!(pinned["last_hit_ms"].is_u64());
        assert_eq!(pinned["hit_histogram"]["counts"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).sum::<u64>(), 2);
    }

    #[test]
    fn miss_keys_use_typed_equality_and_do_not_reformat_repeated_misses() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static FORMATS: AtomicUsize = AtomicUsize::new(0);
        #[derive(Clone, PartialEq, Eq)]
        struct Key(u32);
        impl std::hash::Hash for Key {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) { 0u32.hash(state); }
        }
        impl std::fmt::Debug for Key {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                FORMATS.fetch_add(1, Ordering::Relaxed);
                f.write_str("identical display")
            }
        }
        let mut bank: GraphBank<Key, u32> = GraphBank::new(None);
        bank.attach_device(-304, 100, Site { preference: Preference::Eager, ..site() }).unwrap();
        assert!(matches!(bank.cache_miss(&Key(1), false, None), Miss::Eager));
        assert!(matches!(bank.cache_miss(&Key(2), false, None), Miss::Eager));
        let first = bank.cache_keys[&Key(1)].clone();
        let second = bank.cache_keys[&Key(2)].clone();
        assert_ne!(first, second, "hash collisions and identical Debug text must not alias");
        assert!(matches!(bank.cache_miss(&Key(1), false, None), Miss::Eager));
        assert_eq!(first, bank.cache_keys[&Key(1)]);
        bank.cache_tick();
        assert_eq!(bank.cache_keys.len(), 2, "retained ghosts keep exact typed identities");
        assert_eq!(FORMATS.load(Ordering::Relaxed), 2);
        assert_eq!(bank.cache_keys.len(), 2);
        bank.cache_clear_drained();
        assert!(bank.cache_keys.is_empty());
    }

    #[test]
    fn transient_interned_keys_are_pruned_at_boundary() {
        let mut bank: GraphBank<u32, u32> = GraphBank::new(None);
        bank.attach_device(-305, 1, site()).unwrap();
        assert!(matches!(bank.cache_miss(&1, false, None), Miss::Eager));
        assert_eq!(bank.cache_keys.len(), 1);
        bank.cache_tick();
        assert!(bank.cache_keys.is_empty());
    }

    #[test]
    #[ignore = "CPU microbenchmark; run explicitly with --nocapture"]
    fn cache_replay_nanoseconds() {
        use std::{hint::black_box, time::Instant};
        let mut bank = GraphBank::new(None);
        bank.attach_device(-302, 1 << 20, site()).unwrap();
        for key in [1, 2] {
            let Miss::Capture(id) = bank.cache_miss(&key, key == 1, None) else { panic!() };
            bank.cache_insert(key, key, id, key == 1); bank.cache_measured(10);
        }
        let iterations = 1_000_000;
        let started = Instant::now();
        for _ in 0..iterations { black_box(bank.get(black_box(&1))); }
        println!("plain: {:.1} ns/replay", started.elapsed().as_nanos() as f64 / iterations as f64);
        for (label, key) in [("pinned", 1), ("arc", 2), ("miss", 3)] {
            let started = Instant::now();
            for _ in 0..iterations {
                if key == 3 { black_box(bank.cache_miss(&key, false, Some(super::super::device::Estimate { bytes: u64::MAX, nodes: None, node_rate: 1, graph_rate: 0, parallel_width: 1 }))); }
                else { black_box(bank.cached_launch(black_box(&key))); }
            }
            println!("{label}: {:.1} ns/replay", started.elapsed().as_nanos() as f64 / iterations as f64);
        }
    }
}

impl<K: Hash + Eq + Clone + std::fmt::Debug, E> super::CacheAttached for GraphBank<K, E> {
    fn attach(&mut self, device: i32, reserve: u64, site: super::device::Site) -> anyhow::Result<()> {
        self.attach_device(device, reserve, site)
    }
    fn cycle(&mut self, cycle: super::device::Cycle) { self.cache_cycle(cycle); }
    fn needs_boundary(&self) -> bool { self.cache_needs_boundary() }
    fn measured(&mut self, bytes: u64) { self.cache_measured(bytes); }
    fn tick(&mut self) { self.cache_tick(); }
    fn drain(&mut self, drain: &mut dyn FnMut() -> anyhow::Result<()>,
        measure: &mut dyn FnMut() -> anyhow::Result<super::device::ReleaseMeasurement>) -> anyhow::Result<()> {
        self.cache_drain(drain, measure)
    }
}
