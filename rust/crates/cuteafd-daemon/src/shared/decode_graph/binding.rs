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
    pub fn attach_device(&mut self, reserve: u64, site: super::device::Site) -> Result<()> {
        self.bank.attach_device(self.device, reserve, site)
    }
    pub fn cache_cycle(&mut self, cycle: super::device::Cycle) { self.bank.cache_cycle(cycle); }
    pub fn cache_needs_boundary(&self) -> bool { self.bank.cache_needs_boundary() }
    pub fn cache_measured(&mut self, bytes: u64) { self.bank.cache_measured(bytes); }
    pub fn cache_tick(&mut self) { self.bank.cache_tick(); }
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
        Ok(Self { library, site, device: library.cuda_get_device()?, bank: GraphBank::new(None), current: None })
    }
    pub fn is_none(&self) -> bool { self.current.is_none() }
    pub fn get(&self) -> Option<(*mut c_void, K)> {
        let key = self.current.as_ref()?;
        Some((self.bank.get(key)?.raw, key.clone()))
    }
    /// # Safety
    /// The containing wave pins every captured allocation and drains before
    /// clearing or replacing this binding. Capture itself enqueues no execution.
    pub unsafe fn insert(&mut self, key: K, raw: *mut c_void) -> Result<()> {
        anyhow::ensure!(!self.bank.cache_attached(), "attached binding bank requires cache_insert");
        // SAFETY: the containing wave retains weights, buffers and workspace.
        let owner = unsafe { GraphOwner::new(self.library, self.device, raw, ())? };
        super::census::capture_device_key("deepseek_v41", self.site,
            self as *const Self as usize, self.device, None, &key, raw);
        self.bank.insert(key.clone(), owner, None);
        self.current = Some(key.clone());
        tracing::debug!(target: "cuteafd::graph_capture", site="dspark_binding", ?key,
            device=self.device, bank=self as *const Self as usize, "draft graph captured");
        Ok(())
    }
    /// # Safety
    /// Every stream using these captured pointers has successfully drained.
    pub unsafe fn clear(&mut self) -> Result<()> {
        self.bank.retire_all();
        self.current = None;
        drop(self.bank.drain_retired(|| Ok(()))?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unattached_binding_keeps_selection_and_capture_stats() -> Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let mut graphs = BindingGraphs::new(&library, "fixture")?;
        assert!(graphs.is_none() && graphs.get().is_none());
        // SAFETY: fake DSO graphs have no work or device pointers and never launch.
        unsafe {
            for key in [1, 2] {
                library.cuda_graph_begin_capture(std::ptr::null_mut())?;
                graphs.insert(key, library.cuda_graph_end_capture(std::ptr::null_mut())?)?;
                assert_eq!(graphs.get().unwrap().1, key);
            }
        }
        assert_eq!(graphs.bank.len(), 2);
        assert_eq!(graphs.bank.stats(), super::super::GraphStats { captures: 2, ..Default::default() });
        assert!(!graphs.bank.cache_attached());
        // SAFETY: the fixture never submitted a graph launch.
        unsafe { graphs.clear()?; }
        assert!(graphs.is_none() && graphs.get().is_none());
        Ok(())
    }
}

impl<K: Hash + Eq + Clone + std::fmt::Debug> super::CacheAttached for BindingGraphs<'_, K> {
    cache_attached_methods!(bank);
}
