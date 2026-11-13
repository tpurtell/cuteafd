//! Exact binding adapter for waves that select one executable at a time.
use super::{GraphBank, GraphOwner};
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use std::{ffi::c_void, hash::Hash};

pub(crate) struct BindingGraphs<'a, K> {
    library: &'a NativeLibrary,
    device: i32,
    site: &'static str,
    bank: GraphBank<K, GraphOwner<'a, ()>>,
    current: Option<K>,
}

impl<'a, K: Hash + Eq + Clone + std::fmt::Debug> BindingGraphs<'a, K> {
    pub fn attach_device(&mut self, library: &NativeLibrary, reserve: u64, site: super::device::Site) -> Result<()> {
        self.bank.attach_device(library, self.device, reserve, site)
    }
    pub fn cache_cycle(&mut self, cycle: super::device::Cycle) { self.bank.cache_cycle(cycle); }
    pub fn cache_needs_boundary(&self) -> bool { self.bank.cache_needs_boundary() }
    pub fn cache_measured(&mut self, bytes: u64) { self.bank.cache_measured(bytes); }
    pub fn cache_tick(&mut self) { self.bank.cache_tick(); }
    /// # Safety
    /// Captured storage is persistent and outlives all graph launches.
    pub unsafe fn cache_evict_persistent(&mut self) {
        // SAFETY: caller guarantees persistent backing for every captured pointer.
        unsafe { self.bank.cache_evict_persistent(); }
    }

    pub fn cache_drain(&mut self, drain: impl FnOnce() -> Result<()>,
        measure: impl FnOnce() -> Result<super::device::ReleaseMeasurement>) -> Result<()> {
        self.bank.cache_drain(drain, measure)
    }
    pub fn cache_ticket(&mut self, key: &K, pinned: bool, estimate: Option<super::device::Estimate>)
        -> Option<super::bank::CaptureTicket> {
        self.bank.cache_ticket(key, pinned, estimate)
    }
    pub fn cached_launch(&mut self, key: &K) -> Option<*mut c_void> {
        let raw = self.bank.cached_launch(key)?.raw;
        self.current = Some(key.clone());
        Some(raw)
    }
    /// # Safety
    /// Capture has not launched; the containing wave pins captured storage
    /// until every consumer drains. The ticket cancels itself on any error.
    pub unsafe fn cache_insert(&mut self, key: K, raw: *mut c_void,
        ticket: super::bank::CaptureTicket, nodes: u64, width: usize, spans: bool) -> Result<()> {
        anyhow::ensure!(self.bank.get(&key).is_none(), "cached binding already present");
        // SAFETY: caller retains all captured storage through the consumer drain.
        let owner = unsafe { GraphOwner::new(self.library, self.device, raw, ())? };
        super::census::capture_device_key("deepseek_v41", self.site,
            self as *const Self as usize, self.device, None, &key, raw);
        self.bank.cache_commit(key.clone(), owner, ticket, nodes, width, spans);
        self.current = Some(key);
        Ok(())
    }

    pub fn new(library: &'a NativeLibrary, site: &'static str) -> Result<Self> {
        Ok(Self { library, site, device: library.cuda_get_device()?, bank: GraphBank::new(), current: None })
    }
    pub fn is_none(&self) -> bool { self.current.is_none() }
    pub fn get(&self) -> Option<(*mut c_void, K)> {
        let key = self.current.as_ref()?;
        Some((self.bank.get(key)?.raw, key.clone()))
    }
    /// # Safety
    /// Every stream using these captured pointers has successfully drained.
    pub unsafe fn clear(&mut self) -> Result<()> {
        self.bank.cache_clear_drained();
        self.current = None;
        Ok(())
    }
}

impl<K: Hash + Eq + Clone + std::fmt::Debug> super::CacheAttached for BindingGraphs<'_, K> {
    cache_attached_methods!(bank);
}
