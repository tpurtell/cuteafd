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
    // `GRAPH_PREWARM=off` defers rank-local startup shapes; they are not misses.
    if super::sites::deferring() { return false; }
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
    token: Option<std::sync::Arc<super::device::HitToken>>,
}

impl<E> Entry<E> {
    fn valid(&self) -> bool { self.token.as_ref().is_none_or(|token| token.valid()) }
}

struct DeviceBinding {
    cache: super::device::SharedCache,
    bank: u64,
    site: super::device::Site,
    pinned_hits: u64,
    pinned_counts: Vec<(u64, u64)>,
    touched: Vec<usize>,
    clocks: std::sync::Arc<[super::device::EpochClock; 2]>,
    lane: usize,
}

impl DeviceBinding {
    fn flush(&mut self, cache: &mut super::device::DeviceCache) {
        if std::mem::take(&mut self.pinned_hits) == 0 { return }
        cache.sample_clock();
        for slot in self.touched.drain(..) {
            let (id, count) = &mut self.pinned_counts[slot];
            cache.batched_hits(*id, self.site.cycle, std::mem::take(count));
        }
    }
}

pub(crate) struct CaptureTicket {
    cache: super::device::SharedCache,
    id: u64,
    pinned: bool,
    committed: bool,
    updated: bool,
}

fn capture_failed(cache: &super::device::SharedCache, id: u64, error: &anyhow::Error) -> bool {
    let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
    cache.source_capture_failed(id, error.downcast_ref::<cuteafd_ffi::SmallGraphCapture>().map(|small| small.nodes));
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

    /// Finish an uninstantiated capture, preferring an atomic cold-donor update.
    /// A rejected source returns the ordinary eager sentinel used by capture sites.
    pub fn complete_source(&mut self, library: &cuteafd_ffi::NativeLibrary,
        source: cuteafd_ffi::GraphSource<'_>, stream: *mut std::ffi::c_void)
        -> anyhow::Result<(*mut std::ffi::c_void, usize, usize, bool, Option<String>)> {
        let inspection_error = source.inspection_error.clone();
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if cache.updating(self.id) {
            cache.register_native(self.id, 0, source.signature.clone());
            if let Some(exec) = cache.finish_update(self.id,
                |exec| source.update(exec).map(|result| result.accepted()),
                |exec| source.transfer(exec),
                |exec| { if let Err(error) = library.cuda_graph_exec_invalidate(exec) {
                    tracing::error!(%error, "invalidating rejected donor");
                } }) {
                self.updated = true;
                // Upload outside the device cache mutex on the launch stream so
                // the first launch after refresh is a normal async enqueue,
                // matching the instantiate path below.
                drop(cache);
                // SAFETY: `exec` is exclusively owned by this ticket and `stream`
                // is the caller's launch stream.
                if let Err(error) = unsafe { library.cuda_graph_upload(exec as *mut _, stream) } {
                    // SAFETY: `exec` has never been committed or launched; destroy
                    // it so the transferred executable does not leak while
                    // capture_result/Drop removes this ticket's cache metadata.
                    let _ = unsafe { library.cuda_graph_exec_destroy(exec as *mut _) };
                    return Err(error);
                }
                return Ok((exec as *mut _, source.nodes, source.width, source.spans_devices, inspection_error));
            }
            match cache.fallback_update(self.id) {
                super::device::Miss::Capture(id) => self.id = id,
                super::device::Miss::Eager => {
                    self.committed = true;
                    return Err(cuteafd_ffi::GraphCacheEager.into());
                }
            }
        }
        // Instantiation/upload and budget sampling may block. Telemetry and
        // physical observers must not wait on CUDA while holding this mutex.
        drop(cache);
        // Time inside instantiation (instantiate and upload), counted whether or not it
        // succeeds: published top-level as `graph_instantiate_ns`. The counter is shared
        // code beside the bank, because every graph-using family captures through it.
        let instantiate_started = std::time::Instant::now();
        let exec = source.instantiate(stream);
        crate::shared::decode_graph::stats::GLOBAL
            .record_graph_instantiate(instantiate_started.elapsed().as_nanos() as u64);
        let exec = exec?;
        self.cache.lock().unwrap_or_else(|p| p.into_inner())
            .register_native(self.id, exec as usize, source.signature.clone());
        Ok((exec, source.nodes, source.width, source.spans_devices, inspection_error))
    }

    /// Keep native results and error precedence; settle failure before Drop can
    /// discard an OOM ghost. `committed` also prevents reporting twice.
    pub fn capture_result<T>(&mut self, result: anyhow::Result<T>) -> anyhow::Result<T> {
        if let Err(error) = &result {
            if !self.committed {
                if self.updated {
                    self.cache.lock().unwrap_or_else(|p| p.into_inner()).remove_drained(&[self.id]);
                } else if capture_failed(&self.cache, self.id, error) {
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
            let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            if self.updated { cache.remove_drained(&[self.id]); }
            else { cache.cancel_capture(self.id); }
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
        let (clocks, lane) = cache.lock().unwrap_or_else(|p| p.into_inner()).bank_clock(bank);
        self.device = Some(DeviceBinding { cache, bank, site, pinned_hits: 0, pinned_counts: Vec::new(), touched: Vec::new(), clocks, lane });
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn attach_device_for_test(&mut self, device: i32, reserve: u64,
        site: super::device::Site) -> anyhow::Result<()> {
        anyhow::ensure!(device < 0, "policy fixtures require a synthetic device");
        anyhow::ensure!(self.entries.is_empty() && self.device.is_none(), "graph bank already attached or populated");
        let cache = super::device::device_cache_sampled(device, reserve, site.basis, || Ok((0, 0, 0, 0, 0)))?;
        let bank = cache.lock().unwrap_or_else(|p| p.into_inner()).bank();
        let (clocks, lane) = cache.lock().unwrap_or_else(|p| p.into_inner()).bank_clock(bank);
        self.device = Some(DeviceBinding { cache, bank, site, pinned_hits: 0, pinned_counts: Vec::new(), touched: Vec::new(), clocks, lane });
        Ok(())
    }

    /// Change attribution at a whole-cycle boundary, retaining the pinned-hit
    /// fast path and flushing its previous cycle before switching.
    pub fn cache_cycle(&mut self, cycle: super::device::Cycle) {
        if let Some(binding) = &mut self.device {
            if binding.site.cycle == cycle { return }
            let shared = binding.cache.clone();
            let mut cache = shared.lock().unwrap_or_else(|p| p.into_inner());
            cache.sample_clock();
            if binding.site.cycle != cycle {
                binding.flush(&mut cache);
                cache.bank_cycle(binding.bank, cycle);
                binding.site.cycle = cycle;
            }
        }
    }

    pub fn cached_launch(&mut self, key: &K) -> Option<&E> {
        let entry = self.entries.get_mut(key)?;
        if !entry.valid() { return None }
        if let (Some(binding), Some(_)) = (&mut self.device, entry.cached) {
            if let Some(token) = &entry.token {
                token.hit(binding.clocks[binding.lane].index.load(std::sync::atomic::Ordering::Relaxed));
            }
            binding.pinned_hits += 1;
            let slot = entry.pinned_slot.expect("cache hit slot");
            let count = &mut binding.pinned_counts[slot].1;
            if *count == 0 { binding.touched.push(slot); }
            *count += 1;
        }
        Some(&entry.exec)
    }

    pub fn cache_miss(&mut self, key: &K, pinned: bool, node_estimate: Option<super::device::Estimate>) -> super::device::Miss {
        if self.entries.contains_key(key) && !self.valid_entry(key) { self.entries.remove(key); }
        let Some(binding) = &self.device else { return super::device::Miss::Eager };
        // `GRAPH_PREWARM=off`: a rank-local startup shape leaves no record and
        // captures on its first serving miss. Spanning sites still prewarm.
        if super::sites::defers(binding.site.spans_devices) { return super::device::Miss::Eager; }
        let identity = if let Some(identity) = self.cache_keys.get(key) { identity.clone() } else {
            self.next_cache_key = self.next_cache_key.checked_add(1).expect("graph key identity exhausted");
            let identity = super::device::Identity { bank: binding.bank, site: binding.site.name,
                key: self.next_cache_key, shape: format!("{key:?}").into() };
            self.cache_keys.insert(key.clone(), identity.clone());
            self.cache_keys_dirty = true;
            identity
        };
        let mut cache = binding.cache.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(id) = cache.update_ticket(identity.clone(), binding.site, node_estimate.clone(), pinned) {
            return super::device::Miss::Capture(id);
        }
        cache.miss(identity, binding.site, node_estimate, pinned)
    }

    fn cache_insert(&mut self, key: K, exec: E, id: u64, pinned: bool) {
        let pinned_slot = self.device.as_mut().map(|binding| {
            let slot = binding.pinned_counts.len();
            binding.pinned_counts.push((id, 0));
            // Reserve on insertion, not the first hit of a new cycle.
            if binding.touched.capacity() < binding.pinned_counts.len() {
                binding.touched.reserve(binding.pinned_counts.len() - binding.touched.len());
            }
            slot
        });
        let token = self.device.as_ref().map(|binding|
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).hit_token(id));
        self.entries.insert(key, Entry { exec, cached: Some((id, pinned)), pinned_slot, token });
    }

    pub fn cache_attached(&self) -> bool { self.device.is_some() }

    /// A capture reservation independent of the bank borrow. Dropping it on any
    /// failed path cancels the ticket, without synchronizing a peer segment.
    pub fn cache_ticket(&mut self, key: &K, pinned: bool, estimate: Option<super::device::Estimate>) -> Option<CaptureTicket> {
        let super::device::Miss::Capture(id) = self.cache_miss(key, pinned, estimate) else { return None };
        Some(CaptureTicket { cache: self.device.as_ref()?.cache.clone(), id, pinned, committed: false, updated: false })
    }

    pub fn cache_commit(&mut self, key: K, exec: E, mut ticket: CaptureTicket,
        nodes: u64, width: usize, spans_devices: bool) {
        let binding = self.device.as_ref().expect("graph capture requires an attached device cache");
        assert!(std::sync::Arc::ptr_eq(&binding.cache, &ticket.cache), "graph ticket belongs to a different device cache");
        assert!(!ticket.committed, "graph capture ticket already settled");
        if !ticket.updated { self.cache_topology(ticket.id, nodes, width, spans_devices); }
        self.cache_insert(key.clone(), exec, ticket.id, ticket.pinned);
        ticket.committed = true;
        let retained = ticket.updated || ticket.cache.lock().unwrap_or_else(|p| p.into_inner()).completed_capture(ticket.id);
        if !retained {
            // The caller may already hold the raw executable for its first
            // launch. Retain its owner until ordinary drained bank teardown.
            if let Some(entry) = self.entries.get_mut(&key) { entry.cached = None; entry.token = None; }
        }
    }

    /// Cancel a failed capture, backing off allocation failures by exact key.
    /// Returns whether the caller should emit the failure warning.
    fn cache_capture_failed(&mut self, id: u64, error: &anyhow::Error) -> bool {
        let Some(binding) = &self.device else { return true };
        capture_failed(&binding.cache, id, error)
    }

    /// Consume a failed reservation without cancelling its retained OOM ghost.
    pub fn cache_ticket_failed(&mut self, mut ticket: CaptureTicket, error: &anyhow::Error) -> bool {
        let warn = if ticket.committed { false } else if ticket.updated {
            ticket.cache.lock().unwrap_or_else(|p| p.into_inner()).remove_drained(&[ticket.id]);
            true
        } else { self.cache_capture_failed(ticket.id, error) };
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
        if let Some(binding) = &self.device {
            let ids: Vec<_> = self.entries.values().filter_map(|entry| entry.cached.map(|(id, _)| id)).collect();
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).retire_native_counts(&ids);
        }
        self.entries.clear();
        self.cache_keys.clear();
        self.retired.clear();
        if let Some(binding) = &mut self.device {
            let shared = binding.cache.clone();
            let mut cache = shared.lock().unwrap_or_else(|p| p.into_inner());
            binding.flush(&mut cache);
            binding.pinned_counts.clear();
            let lane = cache.bank_lane(binding.bank);
            cache.remove_bank(binding.bank);
            cache.assign_bank_lane(binding.bank, lane);
        }
    }

    /// Selective explicit invalidation after all consumers have drained.
    pub fn cache_remove_drained(&mut self, matches: impl Fn(&K) -> bool) {
        let keys: Vec<_> = self.entries.keys().filter(|key| matches(key)).cloned().collect();
        let ids: Vec<_> = keys.iter().filter_map(|key| self.entries.get(key)?.cached.map(|(id, _)| id)).collect();
        if let Some(binding) = &self.device {
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).retire_native_counts(&ids);
        }
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
            let changed = self.cache_keys_dirty || self.cache_key_generation != cache.identity_generation();
            if changed || binding.pinned_counts.len() > self.entries.len() {
                self.entries.retain(|_, entry| entry.valid());
                binding.pinned_counts.clear();
                for entry in self.entries.values_mut() {
                    if let Some((id, _)) = entry.cached {
                        entry.pinned_slot = Some(binding.pinned_counts.len());
                        binding.pinned_counts.push((id, 0));
                    }
                }
            }
            if changed {
                self.cache_keys.retain(|_, identity| cache.retains_identity(identity));
                self.cache_keys_dirty = false;
                self.cache_key_generation = cache.identity_generation();
            }
            cache.boundary();
        }
    }

    pub fn cache_lane_owner(&mut self, lane: usize) {
        if let Some(binding) = &mut self.device {
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).assign_bank_lane(binding.bank, lane);
            binding.lane = lane;
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
            let mut cache = binding.cache.lock().unwrap_or_else(|p| p.into_inner());
            if !cache.has_pending_victims() { return }
            cache.refresh_cycles();
            self.entries.iter().filter_map(|(key, entry)| entry.cached
                .filter(|(id, _)| cache.persistent_victim_ready(*id)).map(|(id, _)| (key.clone(), id))).collect()
        };
        binding.cache.lock().unwrap_or_else(|p| p.into_inner()).retire_native_counts(
            &victims.iter().map(|(_, id)| *id).collect::<Vec<_>>());
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
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).retire_native_counts(
                &victims.iter().map(|(_, id)| *id).collect::<Vec<_>>());
            for (key, _) in &victims { if let Some(entry) = self.entries.remove(key) { self.retired.push(entry.exec); } }
            // drain_retired's ordering, with the already completed drain proof.
            self.retired.clear();
            let measurement = measure()?;
            binding.cache.lock().unwrap_or_else(|p| p.into_inner()).destroyed(
                &victims.iter().map(|(_, id)| *id).collect::<Vec<_>>(), measurement);
        }
        let mut cache = binding.cache.lock().unwrap_or_else(|p| p.into_inner());
        cache.serial_boundary();
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

    fn valid_entry(&self, key: &K) -> bool {
        self.entries.get(key).is_some_and(Entry::valid)
    }
    pub fn get(&self, key: &K) -> Option<&E> {
        let entry = self.entries.get(key)?;
        entry.valid().then_some(&entry.exec)
    }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn contains(&self, key: &K) -> bool { self.valid_entry(key) }
    pub fn keys(&self) -> Vec<K> { self.entries.keys().cloned().collect() }
    pub fn count(&self, matches: impl Fn(&K) -> bool) -> usize {
        self.entries.keys().filter(|key| matches(key)).count()
    }
}

impl<K: Hash + Eq + Clone + std::fmt::Debug, P> GraphBank<K, super::GraphOwner<'_, P>> {
    /// Asynchronously pre-uploads a resident graph for `key` on `stream`.
    /// Native skips the driver call if the handle is already uploaded on `stream`.
    ///
    /// # Safety
    /// `stream` is a live CUDA stream (or null for stream 0) on this bank's device.
    #[allow(dead_code)]
    pub unsafe fn prefetch_upload(&self, library: &cuteafd_ffi::NativeLibrary,
        key: &K, stream: *mut std::ffi::c_void) -> anyhow::Result<bool> {
        let Some(owner) = self.get(key) else { return Ok(false); };
        // SAFETY: `owner.raw` is a live resident handle owned by this bank.
        unsafe { library.cuda_graph_upload(owner.raw, stream)?; }
        Ok(true)
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
    fn cross_bank_native_update_invalidates_old_key_and_survives_donor_drop() -> anyhow::Result<()> {
        use super::super::GraphOwner;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory_usage(&library, 0, 128 << 20, 0)?;
        let mut donor = GraphBank::new();
        let mut receiver = GraphBank::new();
        // Synthetic cache isolates the global device registry from fixture GPU0.
        for bank in [&mut donor, &mut receiver] {
            bank.attach_device_for_test(-164_405, 100, site())?;
        }
        donor.cache_lane_owner(1);
        receiver.cache_lane_owner(0);
        let mut ticket = donor.cache_ticket(&1, false, None).unwrap();
        // SAFETY: no fixture device work or captured allocations.
        let source = unsafe { library.cuda_graph_capture_source(std::ptr::null_mut())? };
        let (raw, nodes, width, spans, _) = ticket.complete_source(&library, source, std::ptr::null_mut())?;
        // SAFETY: fixture graph never uses device allocations.
        let owner = unsafe { GraphOwner::new(&library, 0, raw, ())? };
        donor.cache_commit(1, owner, ticket, nodes as u64, width, spans);
        donor.cache_measured(10);
        let cache = donor.device.as_ref().unwrap().cache.clone();
        cache.lock().unwrap().advance_for_update_fixture(1, 2);
        cache.lock().unwrap().set_soft_limit_for_fixture(10);
        let mut ticket = receiver.cache_ticket(&2, false, None).expect("soft-cap source update admitted");
        // SAFETY: source-only fixture records no work.
        let source = unsafe { library.cuda_graph_capture_source(std::ptr::null_mut())? };
        let (new, nodes, width, spans, _) = ticket.complete_source(&library, source, std::ptr::null_mut())?;
        assert_ne!(new, raw);
        // SAFETY: receiver owns the transferred generation.
        let owner = unsafe { GraphOwner::new(&library, 0, new, ())? };
        receiver.cache_commit(2, owner, ticket, nodes as u64, width, spans);
        assert!(donor.cached_launch(&1).is_none());
        assert!(donor.get(&1).is_none());
        assert!(!donor.contains(&1));
        assert!(receiver.cached_launch(&2).is_some());
        drop(donor);
        // SAFETY: new owner remains live after the old bank's generation retires.
        unsafe { library.cuda_graph_launch(new, std::ptr::null_mut())?; }
        let snapshot = super::super::device::snapshots().into_iter().find(|row| row.device == -164_405).unwrap();
        assert_eq!(snapshot.charged, 10);
        assert_eq!(snapshot.live_bytes, 10);
        assert_eq!(snapshot.cycles[&Cycle::Decode].updates_soft, 1);
        drop(receiver);
        Ok(())
    }

    #[test]
    fn source_capture_and_donor_update_upload_on_stream_before_launch() -> anyhow::Result<()> {
        use super::super::GraphOwner;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory_usage(&library, 0, 128 << 20, 0)?;
        let mut donor = GraphBank::new();
        let mut receiver = GraphBank::new();
        for bank in [&mut donor, &mut receiver] {
            bank.attach_device_for_test(-164_408, 100, site())?;
        }
        donor.cache_lane_owner(1);
        receiver.cache_lane_owner(0);
        let stream_a = 0x1000usize as *mut std::ffi::c_void;
        let stream_b = 0x2000usize as *mut std::ffi::c_void;

        // 1. Fresh capture/instantiate uploads on stream_a before launch.
        let mut ticket = donor.cache_ticket(&1, false, None).unwrap();
        // SAFETY: CPU fixture records no device work or captured allocations.
        let source = unsafe { library.cuda_graph_capture_source(stream_a)? };
        let (raw, nodes, width, spans, _) = ticket.complete_source(&library, source, stream_a)?;
        assert_eq!(fixture.events()?, "Gu", "fresh instantiate must upload before launch");
        // SAFETY: fixture graph has no device operands.
        let owner = unsafe { GraphOwner::new(&library, 0, raw, ())? };
        donor.cache_commit(1, owner, ticket, nodes as u64, width, spans);
        donor.cache_measured(10);
        // SAFETY: fixture submits no device work.
        unsafe { library.cuda_graph_launch(raw, stream_a)?; }
        assert_eq!(fixture.events()?, "GuL");

        // 2. Donor update uploads on stream_b inside complete_source, before launch.
        let cache = donor.device.as_ref().unwrap().cache.clone();
        cache.lock().unwrap().advance_for_update_fixture(1, 2);
        cache.lock().unwrap().set_soft_limit_for_fixture(10);
        let mut ticket = receiver.cache_ticket(&2, false, None).expect("soft-cap update admitted");
        // SAFETY: CPU fixture records no device work.
        let source = unsafe { library.cuda_graph_capture_source(stream_b)? };
        let (new, nodes, width, spans, _) = ticket.complete_source(&library, source, stream_b)?;
        assert_eq!(fixture.events()?, "GuLUu",
            "donor update must issue cudaGraphUpload on launch stream before first launch");
        // SAFETY: stale donor handle must reject upload; receiver commits and launches.
        unsafe { assert!(library.cuda_graph_upload(raw, stream_b).is_err()); }
        let owner = unsafe { GraphOwner::new(&library, 0, new, ())? };
        receiver.cache_commit(2, owner, ticket, nodes as u64, width, spans);
        // SAFETY: prefetch on same stream is idempotent; prefetch on a new stream uploads ahead of launch.
        unsafe {
            assert!(receiver.prefetch_upload(&library, &2, stream_b)?);
            assert_eq!(fixture.events()?, "GuLUu", "redundant prefetch on same stream must be a no-op");
            library.cuda_graph_launch(new, stream_b)?;
            assert_eq!(fixture.events()?, "GuLUuL");
            assert!(receiver.prefetch_upload(&library, &2, stream_a)?);
            assert_eq!(fixture.events()?, "GuLUuLu", "prefetch on another stream uploads ahead of launch");
            library.cuda_graph_launch(new, stream_a)?;
            assert_eq!(fixture.events()?, "GuLUuLuL");
        }
        drop(donor);
        drop(receiver);
        Ok(())
    }

    #[test]
    fn donor_update_upload_failure_destroys_exec_and_cleans_cache() -> anyhow::Result<()> {
        use super::super::GraphOwner;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory_usage(&library, 0, 128 << 20, 0)?;
        let mut donor = GraphBank::new();
        let mut receiver = GraphBank::new();
        for bank in [&mut donor, &mut receiver] {
            bank.attach_device_for_test(-164_409, 100, site())?;
        }
        donor.cache_lane_owner(1);
        receiver.cache_lane_owner(0);
        let mut ticket = donor.cache_ticket(&1, false, None).unwrap();
        // SAFETY: CPU fixture records no device work.
        let source = unsafe { library.cuda_graph_capture_source(std::ptr::null_mut())? };
        let (raw, nodes, width, spans, _) = ticket.complete_source(&library, source, std::ptr::null_mut())?;
        // SAFETY: fixture graph has no device operands.
        let owner = unsafe { GraphOwner::new(&library, 0, raw, ())? };
        donor.cache_commit(1, owner, ticket, nodes as u64, width, spans);
        donor.cache_measured(10);
        let cache = donor.device.as_ref().unwrap().cache.clone();
        cache.lock().unwrap().advance_for_update_fixture(1, 2);
        cache.lock().unwrap().set_soft_limit_for_fixture(10);

        let mut ticket = receiver.cache_ticket(&2, false, None).expect("soft-cap update admitted");
        // SAFETY: CPU fixture records no device work.
        let source = unsafe { library.cuda_graph_capture_source(std::ptr::null_mut())? };
        fixture.configure_capture(&library, 12)?;
        let completed = ticket.complete_source(&library, source, std::ptr::null_mut());
        assert!(ticket.capture_result(completed).is_err());
        fixture.configure_capture(&library, 0)?;
        assert_eq!(fixture.events()?, "GuUg",
            "failed upload after donor transfer must destroy the transferred executable");
        let snapshot = super::super::device::snapshots().into_iter().find(|row| row.device == -164_409).unwrap();
        assert_eq!(snapshot.pools.live_execs, 0, "failed upload must remove transferred cache entry");
        drop(donor);
        drop(receiver);
        Ok(())
    }

    #[test]
    fn failed_inspection_survives_source_completion_for_family_retention_policy() -> anyhow::Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_capture(&library, 11)?;
        let mut bank: GraphBank<u32, ()> = GraphBank::new();
        bank.attach_device_for_test(-164_406, 100, site())?;
        let mut ticket = bank.cache_ticket(&1, false, None).unwrap();
        // SAFETY: CPU fixture owns no CUDA work or operands.
        let source = unsafe { library.cuda_graph_capture_source(std::ptr::null_mut())? };
        assert_eq!(source.inspection_error.as_deref(), Some("injected CPU fixture failure"));
        fixture.configure_capture(&library, 0)?;
        let (raw, _, width, _, inspected) = ticket.complete_source(&library, source, std::ptr::null_mut())?;
        assert_eq!(width, 0);
        assert_eq!(inspected.as_deref(), Some("injected CPU fixture failure"));
        // SAFETY: fresh fixture executable has never launched.
        unsafe { library.cuda_graph_exec_destroy(raw)?; }
        Ok(())
    }

    #[test]
    fn failed_commit_measurement_retains_owner_and_charges_until_drained_teardown() -> anyhow::Result<()> {
        use std::{cell::Cell, rc::Rc};
        struct Exec(Rc<Cell<usize>>);
        impl Drop for Exec { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory_usage(&library, 4, 64 << 20, 0)?;
        let mut bank = GraphBank::new();
        bank.attach_device(&library, 4, 32 << 20, site())?;
        let ticket = bank.cache_ticket(&1, false, None).unwrap();
        let drops = Rc::new(Cell::new(0));
        fixture.configure_memory(&library, 4, 0)?;
        bank.cache_commit(1, Exec(drops.clone()), ticket, 7, 1, false);
        assert_eq!(drops.get(), 0, "raw first launch may still use the retained executable");
        assert!(bank.cached_launch(&1).is_some());
        assert!(bank.cache_needs_boundary(), "fallback credit awaits a recoverable physical sample");
        let snapshot = super::super::device::snapshots().into_iter().find(|row| row.device == 4).unwrap();
        assert_eq!(snapshot.charged, 10, "retained owner is conservatively charged");
        assert!(snapshot.hard_stopped);
        assert!(bank.cache_ticket(&2, false, None).is_none());
        bank.cache_clear_drained();
        assert_eq!(drops.get(), 1);
        Ok(())
    }

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
        let cache = bank.device.as_ref().unwrap().cache.clone();
        cache.lock().unwrap().serial_boundary();
        cache.lock().unwrap().serial_boundary();
        assert_eq!(cache.lock().unwrap().mark_victim(false), Some(id));
        assert!(bank.cache_drain(|| anyhow::bail!("not drained"), || unreachable!()).is_err());
        assert_eq!(drops.get(), 0);
        bank.cache_drain(|| { assert_eq!(drops.get(), 0); Ok(()) }, || {
            assert_eq!(drops.get(), 1); Ok(ReleaseMeasurement { released: 0 })
        }).unwrap();
    }

    #[test]
    fn persistent_eviction_without_victims_preserves_pending_capture() {
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(-319, 64 << 20, site()).unwrap();
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
    fn cycle_handle_hit_only_round_never_visits_or_locks_banks_and_survives_reuse() -> anyhow::Result<()> {
        use super::super::device;
        let mut bank = GraphBank::<usize, usize>::new();
        bank.attach_device_for_test(-395, 64 << 20, site())?;
        let mut cycles = device::LaneCycles::register(0, 1,
            |visit| visit(-395, "bank-test", false, &mut bank))?;
        let cache = bank.device.as_ref().unwrap().cache.clone();
        // Keeping the controller locked makes a mutex on the boundary deadlock;
        // the visitor must not run and must not allocate/rebuild an inventory.
        let guard = cache.lock().unwrap();
        for _ in 0..4 {
            // SAFETY: no queued work, fixture contains no captured allocations.
            unsafe { cycles.begin(|_| panic!("hit-only cycle visited banks")) }?;
            cycles.complete();
        }
        drop(guard);
        for round in 0..3 {
            let ticket = bank.cache_ticket(&round, false, None).unwrap();
            bank.cache_commit(round, round, ticket, 10, 1, false);
            bank.cache_measured(if round == 0 { 10 } else { 0 });
            bank.cache_clear_drained();
            // SAFETY: explicit drained reuse of the same registered bank.
            unsafe { cycles.begin(|_| panic!("reused bank lost its lane handle")) }?;
            cycles.complete();
        }
        assert_eq!(device::snapshots().into_iter().find(|row| row.device == -395).unwrap().charged, 10);
        Ok(())
    }

    #[test]
    fn cycle_k1_pending_owner_retires_at_its_serial_boundary() -> anyhow::Result<()> {
        use super::super::device;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        // The DSO memory table is per-library inode; the Rust graph registry
        // is process-global. Ordinal 5 is unused by other graph-cache fixtures.
        fixture.configure_memory_usage(&library, 5, 128 << 20, 0)?;
        let mut bank = GraphBank::<usize, usize>::new();
        bank.attach_device(&library, 5, 64 << 20, site())?;
        let ticket = bank.cache_ticket(&1, false, None).unwrap();
        bank.cache_commit(1, 1, ticket, 10, 1, false);
        bank.cache_measured(10);
        let cache = bank.device.as_ref().unwrap().cache.clone();
        cache.lock().unwrap().serial_boundary();
        cache.lock().unwrap().serial_boundary();
        cache.lock().unwrap().mark_victim(false).unwrap();
        // SAFETY: fixture integers own no captured allocations or native work.
        unsafe { device::observe_completed(&library,
            |visit| visit(5, "bank-test", false, &mut bank)) }?;
        assert_eq!(bank.len(), 0, "the serial boundary must not strand pending owners");
        assert!(!cache.lock().unwrap().has_pending_victims());
        Ok(())
    }

    #[test]
    fn cycle_owner_only_destroys_at_next_start_and_peer_retains_credit() -> anyhow::Result<()> {
        use super::super::{CacheAttached, device};
        use std::{cell::Cell, rc::Rc};
        struct Exec(Rc<Cell<usize>>);
        impl Drop for Exec { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        let drops = [Rc::new(Cell::new(0)), Rc::new(Cell::new(0))];
        let mut banks = [GraphBank::new(), GraphBank::new()];
        for lane in 0..2 {
            banks[lane].attach_device_for_test(-394, 64 << 20, site())?;
            banks[lane].cache_lane_owner(lane);
            let ticket = banks[lane].cache_ticket(&1, false, None).unwrap();
            banks[lane].cache_commit(1, Exec(drops[lane].clone()), ticket, 10, 1, false);
            banks[lane].cache_measured(10);
        }
        let cache = banks[0].device.as_ref().unwrap().cache.clone();
        // SAFETY: fixture owners contain counters, not captured allocations.
        for lane in 0..2 { unsafe { device::lane_cycle(lane, true,
            |visit| visit(-394, "bank-test", false, &mut banks[lane])) }?; }
        let victim = cache.lock().unwrap().mark_victim(false).unwrap();
        let owner = banks.iter().position(|bank| bank.entries.values()
            .any(|entry| entry.cached.is_some_and(|(id, _)| id == victim))).unwrap();
        let peer = 1 - owner;
        // SAFETY: completed peer, allocation-free fixture.
        unsafe {
            device::lane_cycle(peer, false, |visit| visit(-394, "bank-test", false, &mut banks[peer]))?;
            device::lane_cycle(peer, true, |visit| visit(-394, "bank-test", false, &mut banks[peer]))?;
        }
        assert_eq!(drops[owner].get(), 0, "a peer cycle must not touch the owner's bank");
        // Even a mistaken all-bank visitor cannot destroy an in-flight owner.
        unsafe { banks[owner].evict_persistent(); }
        assert_eq!(drops[owner].get(), 0);
        unsafe {
            device::lane_cycle(owner, false, |visit| visit(-394, "bank-test", false, &mut banks[owner]))?;
            device::lane_cycle(owner, true, |visit| visit(-394, "bank-test", false, &mut banks[owner]))?;
        }
        assert_eq!(drops[owner].get(), 1);
        assert_eq!(drops[peer].get(), 0);
        assert_eq!(device::snapshots().into_iter().find(|row| row.device == -394).unwrap().charged, 20);
        for lane in 0..2 {
            unsafe { device::lane_cycle(lane, false, |visit| visit(-394, "bank-test", false, &mut banks[lane]))?; }
        }
        Ok(())
    }

    #[test]
    fn persistent_victims_release_capacity_without_a_drain_or_idle_lane() {
        use std::{cell::Cell, rc::Rc};
        struct Exec(Rc<Cell<usize>>);
        impl Drop for Exec { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        let drops = Rc::new(Cell::new(0));
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(-318, 64 << 20, site()).unwrap();
        let ticket = bank.cache_ticket(&1, false, None).unwrap();
        bank.cache_commit(1, Exec(drops.clone()), ticket, 1, 1, false);
        bank.cache_measured(10);
        for key in 2..20 {
            let cache = bank.device.as_ref().unwrap().cache.clone();
            cache.lock().unwrap().serial_boundary();
            cache.lock().unwrap().serial_boundary();
            assert!(cache.lock().unwrap().mark_victim(false).is_some());
            // SAFETY: fixture executables own no captured buffers or native work.
            unsafe { bank.cache_evict_persistent(); }
            assert_eq!(drops.get(), key - 1);
            let ticket = bank.cache_ticket(&key, false, None).expect("victim capacity reusable without idle");
            bank.cache_commit(key, Exec(drops.clone()), ticket, 1, 1, false);
            bank.cache_measured(0);
            assert_eq!(bank.len(), 1, "rotating identities must not accumulate resident execs");
        }
        assert_eq!(super::super::device::snapshots().into_iter().find(|row| row.device == -318).unwrap().charged, 10);
    }

    #[test]
    fn hit_lookup_does_not_acquire_the_device_mutex() {
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(-39020, 1 << 20, site()).unwrap();
        let ticket = bank.cache_ticket(&7, false, None).unwrap();
        bank.cache_commit(7, 7, ticket, 8, 1, false);
        let shared = bank.device.as_ref().unwrap().cache.clone();
        let guard = shared.lock().unwrap();
        assert_eq!(bank.get(&7), Some(&7));
        assert!(bank.contains(&7));
        assert_eq!(bank.cached_launch(&7), Some(&7));
        drop(guard);
    }

    #[test]
    fn unflushed_hit_protects_a_graph_from_cold_donation() {
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(-39021, 1 << 20, site()).unwrap();
        let ticket = bank.cache_ticket(&7, false, None).unwrap();
        let id = ticket.id;
        bank.cache_commit(7, 7, ticket, 8, 1, false);
        let shared = bank.device.as_ref().unwrap().cache.clone();
        {
            let mut cache = shared.lock().unwrap();
            cache.register_native(id, 123, vec![3, 1]);
            cache.advance_for_update_fixture(0, 5);
            cache.set_soft_limit_for_fixture(0);
        }
        assert_eq!(bank.cached_launch(&7), Some(&7));
        // No cycle switch or counter fold occurred; the published use stamp is
        // already enough to reject retirement under pressure.
        assert!(shared.lock().unwrap().mark_victim(false).is_none());
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
        bank.attach_device_for_test(-305, 0, site()).unwrap();
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
    fn lane_owner(&mut self, lane: usize) { self.cache_lane_owner(lane); }
    unsafe fn evict_persistent(&mut self) {
        // SAFETY: forwarded persistent-storage contract from the visitor.
        unsafe { self.cache_evict_persistent(); }
    }
    fn drain(&mut self, drain: &mut dyn FnMut() -> anyhow::Result<()>,
        measure: &mut dyn FnMut() -> anyhow::Result<super::device::ReleaseMeasurement>) -> anyhow::Result<()> {
        self.cache_drain(drain, measure)
    }
}

#[cfg(test)]
mod perf_tests;

#[cfg(test)]
mod perf_mixed_tests;
