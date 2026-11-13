//! Native graph adapters. The containing wave owns stable workspace/buffers and
//! drains them before insert/clear; GraphOwner retains weight references.
use super::{GraphBank, GraphOwner};
use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;
use super::bank::CaptureTicket;
use super::device::{Cycle, Estimate, ReleaseMeasurement, Site};

macro_rules! cache_methods {
    ($key:ty) => {
        pub fn attach_device(&mut self, library: &NativeLibrary, device: i32, reserve: u64, site: Site) -> Result<()> {
            self.bank.attach_device(library, device, reserve, site)
        }
        pub fn cache_cycle(&mut self, cycle: Cycle) { self.bank.cache_cycle(cycle); }
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
            measure: impl FnOnce() -> Result<ReleaseMeasurement>) -> Result<()> {
            self.bank.cache_drain(drain, measure)
        }
        pub fn cache_ticket(&mut self, key: &$key, pinned: bool, estimate: Option<Estimate>) -> Option<CaptureTicket> {
            self.bank.cache_ticket(key, pinned, estimate)
        }
    };
}

pub(crate) struct LayerGraphs<'w, 'a, W> {
    library: &'a NativeLibrary,
    bank: GraphBank<(usize, u32), GraphOwner<'a, &'w W>>,
    current: [Option<u32>; 40],
}

impl<'w, 'a, W> LayerGraphs<'w, 'a, W> {
    cache_methods!((usize, u32));

    pub fn cached_launch(&mut self, layer: usize, weights: &W, rows: u32) -> Option<(*mut c_void, u32)> {
        if !std::ptr::eq(self.bank.get(&(layer, rows))?.pins, weights) { return None; }
        self.bank.cached_launch(&(layer, rows)).map(|owner| (owner.raw, rows))
    }

    /// # Safety
    /// The executable has not launched; weights and containing workspace remain
    /// pinned until every consumer drains. A ticket owns cancellation on error.
    pub unsafe fn cache_insert(&mut self, layer: usize, weights: &'w W, rows: u32,
        raw: *mut c_void, ticket: CaptureTicket, nodes: u64, width: usize, spans: bool) -> Result<()> {
        ensure!(layer < 40 && (1..=4096).contains(&rows) && !raw.is_null(), "invalid layer graph binding");
        ensure!(self.bank.get(&(layer, rows)).is_none(), "cached layer graph already present");
        let device = self.library.cuda_get_device()?;
        // SAFETY: caller retains weights and workspace through the consumer drain.
        let owner = unsafe { GraphOwner::new(self.library, device, raw, weights)? };
        self.bank.census_capture_device_key(std::any::type_name::<W>(), device, &(layer, rows), raw);
        self.bank.cache_commit((layer, rows), owner, ticket, nodes, width, spans);
        self.current[layer] = Some(rows);
        Ok(())
    }

    pub fn new(library: &'a NativeLibrary, _decode_rows: u32, family: &'static str) -> Self {
        Self { library, bank: GraphBank::with_family(family),
            current: [None; 40] }
    }
    pub fn get(&self, layer: usize, weights: &W) -> Option<(*mut c_void, u32)> {
        self.get_shape(layer, weights, self.current.get(layer).copied().flatten()?)
    }
    pub fn get_shape(&self, layer: usize, weights: &W, rows: u32) -> Option<(*mut c_void, u32)> {
        let owner = self.bank.get(&(layer, rows))?;
        std::ptr::eq(owner.pins, weights).then_some((owner.raw, rows))
    }
    /// # Safety
    /// Every launch of this layer's graph completed on every consumer stream.
    pub unsafe fn remove(&mut self, layer: usize) -> Result<()> {
        ensure!(layer < 40, "invalid layer graph index");
        self.bank.cache_remove_drained(|(l, _)| *l == layer);
        self.current[layer] = None;
        Ok(())
    }
    /// # Safety
    /// Every queued use of this wave's captured pointers completed.
    pub unsafe fn clear(&mut self) -> Result<()> {
        self.bank.cache_clear_drained();
        self.current = [None; 40];
        Ok(())
    }
    #[cfg(test)]
    pub fn len(&self) -> usize { self.bank.len() }
}

pub(crate) struct RowGraphs<'a> {
    library: &'a NativeLibrary,
    bank: GraphBank<usize, GraphOwner<'a, ()>>,
    site: &'static str,
}
impl<'a> RowGraphs<'a> {
    cache_methods!(usize);

    pub fn cached_launch(&mut self, rows: usize) -> Option<*mut c_void> {
        self.bank.cached_launch(&rows).map(|owner| owner.raw)
    }

    /// # Safety
    /// The executable has not launched; the containing wave pins its workspace
    /// through the consumer drain. A ticket owns cancellation on every error.
    pub unsafe fn cache_insert(&mut self, rows: usize, raw: *mut c_void,
        ticket: CaptureTicket, nodes: u64, width: usize, spans: bool) -> Result<()> {
        ensure!((1..=4096).contains(&rows) && !raw.is_null(), "invalid row graph binding");
        ensure!(self.bank.get(&rows).is_none(), "cached row graph already present");
        let device = self.library.cuda_get_device()?;
        // SAFETY: caller retains all captured storage through the consumer drain.
        let owner = unsafe { GraphOwner::new(self.library, device, raw, ())? };
        self.bank.census_capture_device_key(self.site, device, &rows, raw);
        self.bank.cache_commit(rows, owner, ticket, nodes, width, spans);
        Ok(())
    }

    pub fn new(library: &'a NativeLibrary, site: &'static str, _decode_rows: usize, family: &'static str) -> Self {
        Self { library, bank: GraphBank::with_family(family),
            site }
    }
    pub fn get(&self, rows: usize) -> Option<*mut c_void> { self.bank.get(&rows).map(|owner| owner.raw) }
    /// # Safety
    /// Every queued use completed before releasing graph executables or storage.
    pub unsafe fn clear(&mut self) -> Result<()> {
        self.bank.cache_clear_drained();
        Ok(())
    }
    #[cfg(test)]
    pub fn len(&self) -> usize { self.bank.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_ticket_cancels_and_commits_surviving_owner() -> Result<()> {
        use super::super::device::{Basis, Preference};
        let mut bank = GraphBank::<usize, ()>::new();
        bank.attach_device_for_test(-8801, 64 << 20, Site { name: "ticket-fixture", cycle: Cycle::Other,
            preference: Preference::Capture, bootstrap: 2 << 20, basis: Basis::Bytes, spans_devices: true })?;
        let ticket = bank.cache_ticket(&1, false, None).expect("capture");
        drop(ticket);
        assert!(bank.cache_needs_boundary());
        bank.cache_measured(2 << 20);
        let ticket = bank.cache_ticket(&2, false, None).expect("capture");
        bank.cache_commit(2, (), ticket, 12, 1, true);
        bank.cache_measured(2 << 20);
        assert_eq!(bank.cached_launch(&2), Some(&()));
        Ok(())
    }
}

impl<W> super::CacheAttached for LayerGraphs<'_, '_, W> {
    cache_attached_methods!(bank);
}
impl super::CacheAttached for RowGraphs<'_> {
    cache_attached_methods!(bank);
}

#[cfg(test)]
mod visitor_tests {
    use super::*;
    use super::super::{BindingGraphs, CacheAttached};
    #[test]
    fn heterogeneous_control_attaches_one_device_pool() -> Result<()> {
        use super::super::device::{Basis, Preference};
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let mut direct = GraphBank::<u8, ()>::new();
        let mut layers = LayerGraphs::<'_, '_, ()>::new(&library, 64, "test");
        let mut rows = RowGraphs::new(&library, "fixture", 64, "test");
        let mut bindings = BindingGraphs::<u16>::new(&library, "fixture")?;
        let banks: [&mut dyn CacheAttached; 4] = [&mut direct, &mut layers, &mut rows, &mut bindings];
        for bank in banks {
            bank.attach(&library, -8802, 64 << 20, Site { name: "visitor-fixture", cycle: Cycle::Other,
                preference: Preference::Capture, bootstrap: 2 << 20, basis: Basis::Bytes,
                spans_devices: false })?;
            bank.cycle(Cycle::Verify);
            assert!(!bank.needs_boundary());
            bank.tick();
            bank.drain(&mut || Ok(()), &mut || Ok(ReleaseMeasurement { released: 0 }))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod clear_tests {
    use super::*;
    use super::super::{BindingGraphs, CacheAttached};
    #[test]
    fn attached_wrapper_clear_releases_live_capacity_not_physical_charge() -> Result<()> {
        use super::super::device::{self, Basis, Preference};
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let weights = ();
        let mut rows = RowGraphs::new(&library, "clear-row", 64, "test");
        let mut layers = LayerGraphs::new(&library, 64, "test");
        let mut bindings = BindingGraphs::<usize>::new(&library, "clear-binding")?;
        let site = Site { name: "clear-test", cycle: Cycle::Other, preference: Preference::Capture,
            bootstrap: 2 << 20, basis: Basis::Bytes, spans_devices: false };
        for bank in [&mut rows as &mut dyn CacheAttached, &mut layers, &mut bindings] {
            bank.attach(&library, -8804, 64 << 20, site)?;
        }
        let capture = || -> Result<*mut c_void> {
            // SAFETY: fixture records no device work or referenced allocations.
            unsafe {
                library.cuda_graph_begin_capture(std::ptr::null_mut())?;
                library.cuda_graph_end_capture(std::ptr::null_mut())
            }
        };
        let row_ticket = rows.cache_ticket(&1, false, None).unwrap();
        let layer_ticket = layers.cache_ticket(&(0, 1), false, None).unwrap();
        let binding_ticket = bindings.cache_ticket(&1, false, None).unwrap();
        // SAFETY: fake executables never launch; all fixture storage outlives them.
        unsafe {
            rows.cache_insert(1, capture()?, row_ticket, 1, 1, false)?;
            layers.cache_insert(0, &weights, 1, capture()?, layer_ticket, 1, 1, false)?;
            bindings.cache_insert(1, capture()?, binding_ticket, 1, 1, false)?;
        }
        rows.cache_measured(6 << 20);
        let snapshot = || device::snapshots().into_iter().find(|row| row.device == -8804).unwrap();
        assert_eq!(snapshot().pools.live_execs, 3);
        let charged = snapshot().charged;
        // SAFETY: no fixture launch was submitted.
        unsafe { rows.clear()?; layers.remove(0)?; bindings.clear()?; }
        let after = snapshot();
        assert_eq!(after.pools.live_execs, 0);
        assert_eq!(after.pools.live_nodes, 0);
        assert_eq!(after.charged, charged);
        assert!(rows.cache_ticket(&1, false, None).is_some());
        assert!(layers.cache_ticket(&(0, 1), false, None).is_some());
        assert!(bindings.cache_ticket(&1, false, None).is_some());
        Ok(())
    }
}
