//! Exact-key graph storage. Persistent-buffer executables may be evicted without
//! draining; owners of per-graph storage require explicit drained retirement.
use std::collections::HashMap;
use std::hash::Hash;

/// One distinct shape that ran eagerly, with every reported execution of it.
/// Kept process-wide rather than per bank so /v1/stats reports
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
/// `note_eager`, so reported misses are recorded and warned about in one place.
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

/// Counts one eager execution of a shape the device cache could not retain.
/// The context is built only on its first miss; family owners share this
/// diagnostic entry point and /v1/stats inventory.
pub(crate) fn note_eager(family: &'static str, shape: &str, context: impl FnOnce() -> String) -> bool {
    if !note_missing(family, shape) { return false; }
    // Built outside the macro so the description is produced even when the
    // current subscriber filters this target out.
    let context = context();
    tracing::warn!(target: "cuteafd::graph_capture", family = family, key = %shape,
        context = %context,
        "graph cache selected eager; consider adding this shape to prewarm");
    true
}

struct Entry<E> {
    exec: E,
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

fn capture_failed(cache: &super::device::SharedCache, id: u64, error: &anyhow::Error) -> bool {
    let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(small) = error.downcast_ref::<cuteafd_ffi::SmallGraphCapture>() {
        cache.capture_too_small(id, small.nodes);
        false
    } else if error.downcast_ref::<cuteafd_ffi::NativeStatusError>()
        .is_some_and(|error| error.status == cuteafd_ffi::CUTEAFD_STATUS_ALLOCATION_FAILED) {
        cache.capture_oom(id)
    } else {
        cache.cancel_capture(id);
        true
    }
}

impl CaptureTicket {
    pub(super) fn require_active(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.committed, "graph capture ticket already settled");
        Ok(())
    }

    /// Keep native results and error precedence; settle failure before Drop can
    /// discard an OOM ghost. `committed` also prevents reporting twice.
    pub fn capture_result<T>(&mut self, result: anyhow::Result<T>) -> anyhow::Result<T> {
        if let Err(error) = &result {
            if !self.committed {
                if capture_failed(&self.cache, self.id, error) {
                    tracing::warn!(%error, "graph capture unavailable");
                }
                self.committed = true;
            }
        }
        result
    }
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
    family: &'static str,
    retired: Vec<E>,
    device: Option<DeviceBinding>,
}

impl<K: Hash + Eq + Clone + std::fmt::Debug, E> GraphBank<K, E> {
    pub fn new() -> Self { Self::with_family("shared") }

    pub fn with_family(family: &'static str) -> Self {
        Self { entries: HashMap::new(), cache_keys: HashMap::new(), next_cache_key: 0,
            cache_key_generation: 0, cache_keys_dirty: false, family,
            retired: Vec::new(), device: None }
    }

    pub fn attach_device(&mut self, library: &cuteafd_ffi::NativeLibrary, device: i32, reserve: u64, site: super::device::Site) -> anyhow::Result<()> {
        anyhow::ensure!(self.entries.is_empty() && self.device.is_none(), "graph bank already attached or populated");
        let cache = super::device::device_cache(library, device, reserve, site.basis)?;
        let bank = cache.lock().unwrap_or_else(|p| p.into_inner()).bank();
        self.device = Some(DeviceBinding { cache, bank, site, pinned_hits: 0, pinned_counts: Vec::new() });
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn attach_device_for_test(&mut self, device: i32, reserve: u64,
        site: super::device::Site) -> anyhow::Result<()> {
        anyhow::ensure!(device < 0, "policy fixtures require a synthetic device");
        anyhow::ensure!(self.entries.is_empty() && self.device.is_none(), "graph bank already attached or populated");
        let cache = super::device::device_cache_sampled(device, reserve, site.basis, || Ok((0, 0, 0, 0, 0)))?;
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

    fn cache_insert(&mut self, key: K, exec: E, id: u64, pinned: bool) {
        let pinned_slot = if pinned {
            self.device.as_mut().map(|binding| {
                let slot = binding.pinned_counts.len();
                binding.pinned_counts.push((id, 0));
                slot
            })
        } else { None };
        self.entries.insert(key, Entry { exec, cached: Some((id, pinned)), pinned_slot });
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
        let binding = self.device.as_ref().expect("graph capture requires an attached device cache");
        assert!(std::sync::Arc::ptr_eq(&binding.cache, &ticket.cache), "graph ticket belongs to a different device cache");
        assert!(!ticket.committed, "graph capture ticket already settled");
        self.cache_topology(ticket.id, nodes, width, spans_devices);
        self.cache_insert(key, exec, ticket.id, ticket.pinned);
        ticket.committed = true;
    }

    /// Cancel a failed capture, backing off allocation failures by exact key.
    /// Returns whether the caller should emit the failure warning.
    fn cache_capture_failed(&mut self, id: u64, error: &anyhow::Error) -> bool {
        let Some(binding) = &self.device else { return true };
        capture_failed(&binding.cache, id, error)
    }

    /// Consume a failed reservation without cancelling its retained OOM ghost.
    pub fn cache_ticket_failed(&mut self, mut ticket: CaptureTicket, error: &anyhow::Error) -> bool {
        let warn = if ticket.committed { false } else { self.cache_capture_failed(ticket.id, error) };
        ticket.committed = true;
        warn
    }

    pub fn cache_cancel(&mut self, id: u64) {
        if let Some(binding) = &self.device { binding.cache.lock().unwrap_or_else(|p| p.into_inner()).cancel_capture(id); }
    }

    fn cache_topology(&mut self, id: u64, nodes: u64, width: usize, spans_devices: bool) {
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
            let shared = binding.cache.clone();
            let mut cache = shared.lock().unwrap_or_else(|p| p.into_inner());
            binding.flush(&mut cache);
            binding.pinned_counts.clear();
            cache.remove_bank(binding.bank);
        }
    }

    /// Selective explicit invalidation after all consumers have drained.
    pub fn cache_remove_drained(&mut self, matches: impl Fn(&K) -> bool) {
        let keys: Vec<_> = self.entries.keys().filter(|key| matches(key)).cloned().collect();
        let ids: Vec<_> = keys.iter().filter_map(|key| self.entries.get(key)?.cached.map(|(id, _)| id)).collect();
        for key in keys { self.entries.remove(&key); }
        if let Some(binding) = &self.device {
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).remove_drained(&ids);
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

    /// Destroy victims without waiting for an idle lane. cudaGraphExecDestroy
    /// itself is asynchronous-safe; only captured allocation lifetime matters.
    /// # Safety
    /// Dropping E must not release captured allocations. Their persistent backing
    /// storage remains owned by the pipeline, which already orders buffer reuse.
    pub unsafe fn cache_evict_persistent(&mut self) {
        let Some(binding) = &mut self.device else { return };
        let victims: Vec<_> = {
            let cache = binding.cache.lock().unwrap_or_else(|p| p.into_inner());
            if !cache.has_pending_victims() { return }
            self.entries.iter().filter_map(|(key, entry)| entry.cached
                .filter(|(id, _)| cache.pending(*id)).map(|(id, _)| (key.clone(), id))).collect()
        };
        for (key, _) in &victims { self.entries.remove(key); }
        if !victims.is_empty() {
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).destroyed(
                &victims.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
                super::device::ReleaseMeasurement { released: 0 });
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

    pub fn census_capture_key(&self, bank: &str, rank: i32, lane: Option<usize>, key: &K,
        exec: *mut std::ffi::c_void) {
        super::census::capture_key(self.family, bank, self as *const Self as usize, rank, lane, key, exec);
    }

    pub fn census_capture_device_key(&self, bank: &str, device: i32, key: &K,
        exec: *mut std::ffi::c_void) {
        super::census::capture_device_key(self.family, bank, self as *const Self as usize,
            device, None, key, exec);
    }

    pub fn get(&self, key: &K) -> Option<&E> { self.entries.get(key).map(|entry| &entry.exec) }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn contains(&self, key: &K) -> bool { self.entries.contains_key(key) }
    pub fn keys(&self) -> Vec<K> { self.entries.keys().cloned().collect() }
    pub fn count(&self, matches: impl Fn(&K) -> bool) -> usize {
        self.entries.keys().filter(|key| matches(key)).count()
    }
}

impl<K, E> Drop for GraphBank<K, E> {
    fn drop(&mut self) {
        if let Some(binding) = &mut self.device {
            let shared = binding.cache.clone();
            let mut cache = shared.lock().unwrap_or_else(|p| p.into_inner());
            binding.flush(&mut cache);
            binding.pinned_counts.clear();
            cache.remove_bank(binding.bank);
        }

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
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(-301, 20, site()).unwrap();
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
    fn persistent_eviction_without_victims_preserves_pending_capture() {
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(-319, 1000, site()).unwrap();
        let ticket = bank.cache_ticket(&1, false, None).unwrap();
        bank.cache_commit(1, 42, ticket, 1, 1, false);
        let cache = bank.device.as_ref().unwrap().cache.clone();
        assert!(cache.lock().unwrap().needs_boundary());
        assert!(!cache.lock().unwrap().has_pending_victims());
        // SAFETY: integers own no captured allocations or queued native work.
        unsafe { bank.cache_evict_persistent(); }
        assert_eq!(bank.len(), 1);
        assert!(cache.lock().unwrap().needs_boundary());
        bank.cache_measured(10);
        // SAFETY: the same allocation-free fixture remains live.
        unsafe { bank.cache_evict_persistent(); }
        assert_eq!(bank.cached_launch(&1), Some(&42));
    }

    #[test]
    fn persistent_victims_release_capacity_without_a_drain_or_idle_lane() {
        use std::{cell::Cell, rc::Rc};
        struct Exec(Rc<Cell<usize>>);
        impl Drop for Exec { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        let drops = Rc::new(Cell::new(0));
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(-318, 20, site()).unwrap();
        let ticket = bank.cache_ticket(&1, false, None).unwrap();
        bank.cache_commit(1, Exec(drops.clone()), ticket, 1, 1, false);
        bank.cache_measured(10);
        for key in 2..20 {
            assert!(bank.cache_ticket(&key, false, None).is_none());
            // SAFETY: fixture executables own no captured buffers or native work.
            unsafe { bank.cache_evict_persistent(); }
            assert_eq!(drops.get(), key - 1);
            let ticket = bank.cache_ticket(&key, false, None).expect("victim capacity reusable without idle");
            bank.cache_commit(key, Exec(drops.clone()), ticket, 1, 1, false);
            bank.cache_measured(0);
        }
        assert_eq!(super::super::device::snapshots().into_iter().find(|row| row.device == -318).unwrap().charged, 10);
    }

    #[test]
    fn cached_graph_hits_follow_the_current_cycle_including_pinned_fast_path() {
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(-303, 100, site()).unwrap();
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
        let mut bank: GraphBank<Key, u32> = GraphBank::new();
        bank.attach_device_for_test(-304, 100, Site { preference: Preference::Eager, ..site() }).unwrap();
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
    fn allocation_failure_backoff_keeps_exact_keys_and_uses_typed_native_status() {
        #[derive(Clone, PartialEq, Eq, Hash)]
        struct Key(u32);
        impl std::fmt::Debug for Key {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("same display") }
        }
        let mut bank: GraphBank<Key, u32> = GraphBank::new();
        bank.attach_device_for_test(-320, 1000, Site { spans_devices: true, ..site() }).unwrap();
        let oom = anyhow::Error::new(cuteafd_ffi::NativeStatusError {
            context: "end capture".into(), status: cuteafd_ffi::CUTEAFD_STATUS_ALLOCATION_FAILED,
            detail: "opaque diagnostic".into(),
        }).context("peer graph capture");
        let Miss::Capture(id) = bank.cache_miss(&Key(1), false, None) else { panic!() };
        assert!(bank.cache_capture_failed(id, &oom));
        bank.cache_measured(10);
        bank.cache_tick();
        assert_eq!(bank.cache_keys.len(), 1, "OOM ghosts retain the typed identity across boundaries");
        for _ in 0..256 {
            assert!(matches!(bank.cache_miss(&Key(1), false, None), Miss::Eager));
        }
        let Miss::Capture(other) = bank.cache_miss(&Key(2), false, None) else { panic!() };
        assert_ne!(other, id, "identical display text must not share backoff");
        bank.cache_cancel(other);
        let Miss::Capture(retry) = bank.cache_miss(&Key(1), false, None) else { panic!() };
        assert_eq!(retry, id);
        assert!(!bank.cache_capture_failed(retry, &oom), "only one warning per retained key");
        bank.cache_measured(0);
        bank.cache_clear_drained();
        // Neither a different native status nor matching text is an allocation failure.
        for error in [anyhow::Error::new(cuteafd_ffi::NativeStatusError {
            context: "end capture".into(), status: cuteafd_ffi::CUTEAFD_STATUS_CUDA_UNAVAILABLE,
            detail: "cudaGraphInstantiate out of memory".into(),
        }), anyhow::anyhow!("cudaGraphInstantiate out of memory")] {
            let Miss::Capture(id) = bank.cache_miss(&Key(1), false, None) else { panic!() };
            assert!(bank.cache_capture_failed(id, &error));
            bank.cache_measured(0);
            bank.cache_tick();
            assert!(bank.cache_keys.is_empty(), "non-OOM cancellation does not retain a cooldown ghost");
        }
    }

    #[test]
    fn raii_allocation_failure_retains_ghost_but_other_errors_cancel() {
        let mut bank: GraphBank<u32, u32> = GraphBank::new();
        bank.attach_device_for_test(-321, 1000, Site { spans_devices: true, ..site() }).unwrap();
        let oom = anyhow::Error::new(cuteafd_ffi::NativeStatusError {
            context: "end capture".into(), status: cuteafd_ffi::CUTEAFD_STATUS_ALLOCATION_FAILED,
            detail: "opaque diagnostic".into(),
        }).context("peer graph capture");
        let ticket = bank.cache_ticket(&1, false, None).unwrap();
        assert!(bank.cache_ticket_failed(ticket, &oom));
        bank.cache_tick();
        assert_eq!(bank.cache_keys.len(), 1);
        for _ in 0..256 { assert!(bank.cache_ticket(&1, false, None).is_none()); }
        let ticket = bank.cache_ticket(&1, false, None).expect("cooldown expires");
        assert!(!bank.cache_ticket_failed(ticket, &oom), "retained key warns once");
        let ticket = bank.cache_ticket(&2, false, None).unwrap();
        assert!(bank.cache_ticket_failed(ticket, &anyhow::anyhow!("ordinary capture failure")));
        bank.cache_tick();
        assert!(!bank.cache_keys.contains_key(&2));
        assert!(bank.cache_ticket(&2, false, None).is_some(), "non-OOM has no cooldown");
    }

    /// #1878: once serving, a site spanning both GPUs never captures (its first
    /// instantiation would wait on the other GPU's spinning peer wait); the miss
    /// runs eagerly and stays a remembered key. Single-GPU sites still capture,
    /// and before readiness (startup prewarm) spanning sites capture as before.
    #[test]
    fn serving_never_captures_a_cross_device_graph() {
        use super::super::device::seal_cross_device_capture_on_this_thread as seal;
        seal(false);
        let mut prewarm: GraphBank<u32, u32> = GraphBank::new();
        prewarm.attach_device_for_test(-401, 1000, Site { spans_devices: true, ..site() }).unwrap();
        let ticket = prewarm.cache_ticket(&1, false, None).expect("startup prewarm captures cross-device graphs");
        drop(prewarm.cache_ticket_failed(ticket, &anyhow::anyhow!("fixture")));
        seal(true);
        let mut spanning: GraphBank<u32, u32> = GraphBank::new();
        spanning.attach_device_for_test(-402, 1000, Site { spans_devices: true, ..site() }).unwrap();
        for _ in 0..8 {
            assert!(spanning.cache_ticket(&7, true, None).is_none(), "serving must not capture across devices");
            spanning.cache_measured(0);
            spanning.cache_tick();
        }
        let traced = super::super::device::snapshots().into_iter().find(|row| row.device == -402).unwrap();
        assert_eq!(traced.charged, 0, "an eager cross-device miss reserves nothing");
        let mut local: GraphBank<u32, u32> = GraphBank::new();
        local.attach_device_for_test(-403, 1000, site()).unwrap();
        assert!(local.cache_ticket(&7, false, None).is_some(), "single-GPU sites still capture while serving");
        seal(false);
    }

    #[test]
    fn small_capture_remains_eager_across_boundaries_and_ghost_pressure() {
        for nodes in 0..=2 {
            let mut bank: GraphBank<u32, u32> = GraphBank::new();
            bank.attach_device_for_test(-330 - nodes as i32, 1000,
                Site { spans_devices: true, ..site() }).unwrap();
            let mut ticket = bank.cache_ticket(&1, true, None).unwrap();
            let small = anyhow::Error::new(cuteafd_ffi::SmallGraphCapture { nodes });
            assert!(ticket.capture_result::<()>(Err(small)).is_err());
            drop(ticket);
            bank.cache_measured(10);
            bank.cache_tick();
            let charged = super::super::device::snapshots().into_iter().find(|row| row.device == -330 - nodes as i32).unwrap().charged;
            for key in 2..100 {
                let ticket = bank.cache_ticket(&key, false, None);
                drop(ticket);
                bank.cache_measured(0);
                bank.cache_tick();
                assert!(bank.cache_ticket(&1, true, None).is_none());
            }
            assert_eq!(bank.cache_keys.len(), 1);
            assert_eq!(super::super::device::snapshots().into_iter().find(|row| row.device == -330 - nodes as i32).unwrap().charged, charged);
        }
    }

    #[test]
    fn successful_begin_then_oom_end_reports_ticket_once() {
        let mut bank: GraphBank<u32, u32> = GraphBank::new();
        bank.attach_device_for_test(-322, 1000, Site { spans_devices: true, ..site() }).unwrap();
        let mut ticket = bank.cache_ticket(&1, false, None).unwrap();
        ticket.capture_result(Ok(())).unwrap();
        assert!(!ticket.committed, "successful begin does not settle admission");
        let oom = anyhow::Error::new(cuteafd_ffi::NativeStatusError {
            context: "end capture".into(), status: cuteafd_ffi::CUTEAFD_STATUS_ALLOCATION_FAILED,
            detail: "opaque diagnostic".into(),
        });
        assert!(ticket.capture_result::<()>(Err(oom)).is_err());
        assert!(ticket.committed, "failed end settles before Drop");
        assert!(bank.cache_ticket(&1, false, None).is_none());
        // A second report must neither cancel the ghost nor reset its timer.
        assert!(ticket.capture_result::<()>(Err(anyhow::anyhow!("later cleanup error"))).is_err());
        assert!(!bank.cache_ticket_failed(ticket, &anyhow::anyhow!("duplicate report")));
        bank.cache_tick();
        for _ in 1..256 { assert!(bank.cache_ticket(&1, false, None).is_none()); }
        assert!(bank.cache_ticket(&1, false, None).is_some());
        let mut ticket = bank.cache_ticket(&2, false, None).unwrap();
        assert!(ticket.capture_result::<()>(Err(anyhow::anyhow!("not OOM"))).is_err());
        drop(ticket);
        bank.cache_tick();
        assert!(!bank.cache_keys.contains_key(&2));
    }

    #[test]
    fn transient_interned_keys_are_pruned_at_boundary() {
        let mut bank: GraphBank<u32, u32> = GraphBank::new();
        bank.attach_device_for_test(-305, 1, site()).unwrap();
        assert!(matches!(bank.cache_miss(&1, false, None), Miss::Eager));
        assert_eq!(bank.cache_keys.len(), 1);
        bank.cache_tick();
        assert!(bank.cache_keys.is_empty());
    }

    #[test]
    #[ignore = "CPU microbenchmark; run explicitly with --nocapture"]
    fn cache_replay_nanoseconds() {
        use std::{hint::black_box, time::Instant};
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(-302, 1 << 20, site()).unwrap();
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
    fn attach(&mut self, library: &cuteafd_ffi::NativeLibrary, device: i32, reserve: u64, site: super::device::Site) -> anyhow::Result<()> {
        self.attach_device(library, device, reserve, site)
    }
    fn cycle(&mut self, cycle: super::device::Cycle) { self.cache_cycle(cycle); }
    fn needs_boundary(&self) -> bool { self.cache_needs_boundary() }
    fn measured(&mut self, bytes: u64) { self.cache_measured(bytes); }
    fn tick(&mut self) { self.cache_tick(); }
    unsafe fn evict_persistent(&mut self) {
        // SAFETY: forwarded persistent-storage contract from the visitor.
        unsafe { self.cache_evict_persistent(); }
    }
    fn drain(&mut self, drain: &mut dyn FnMut() -> anyhow::Result<()>,
        measure: &mut dyn FnMut() -> anyhow::Result<super::device::ReleaseMeasurement>) -> anyhow::Result<()> {
        self.cache_drain(drain, measure)
    }
}
