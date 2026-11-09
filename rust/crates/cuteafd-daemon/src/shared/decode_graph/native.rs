//! Native graph adapters. The containing wave owns stable workspace/buffers and
//! drains them before insert/clear; GraphOwner retains weight references.
use super::{GraphBank, GraphOwner, GraphPolicy};
use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;
use super::bank::CaptureTicket;
use super::device::{Cycle, Estimate, Miss, ReleaseMeasurement, Site};

// Explicit opt-in: unattached wrappers continue to use the legacy methods.
macro_rules! cache_methods {
    ($key:ty) => {
        pub fn attach_device(&mut self, device: i32, reserve: u64, site: Site) -> Result<()> {
            self.bank.attach_device(device, reserve, site)
        }
        pub fn cache_cycle(&mut self, cycle: Cycle) { self.bank.cache_cycle(cycle); }
        pub fn cache_needs_boundary(&self) -> bool { self.bank.cache_needs_boundary() }
        pub fn cache_measured(&mut self, bytes: u64) { self.bank.cache_measured(bytes); }
        pub fn cache_tick(&mut self) { self.bank.cache_tick(); }
        pub fn cache_drain(&mut self, drain: impl FnOnce() -> Result<()>,
            measure: impl FnOnce() -> Result<ReleaseMeasurement>) -> Result<()> {
            self.bank.cache_drain(drain, measure)
        }
        pub fn cache_miss(&mut self, key: &$key, pinned: bool, estimate: Option<Estimate>) -> Miss {
            self.bank.cache_miss(key, pinned, estimate)
        }
        pub fn cache_ticket(&mut self, key: &$key, pinned: bool, estimate: Option<Estimate>) -> Option<CaptureTicket> {
            self.bank.cache_ticket(key, pinned, estimate)
        }
        pub fn cache_cancel(&mut self, id: u64) { self.bank.cache_cancel(id); }
        pub fn cache_topology(&mut self, id: u64, nodes: u64, width: usize, spans: bool) {
            self.bank.cache_topology(id, nodes, width, spans);
        }
    };
}

/// Large (above-decode) shapes a bank retains per layer: the full prefill chunk
/// and the decoder replay. Other large shapes are executed eagerly upstream.
pub(crate) const LARGE_SHAPES: usize = 2;

pub(crate) struct LayerGraphs<'w, 'a, W> {
    library: &'a NativeLibrary,
    bank: GraphBank<(usize, u32), GraphOwner<'a, &'w W>>,
    current: [Option<u32>; 40],
    retain_small: bool,
    decode_rows: u32,
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

    pub fn new(library: &'a NativeLibrary, decode_rows: u32, family: &'static str) -> Self {
        Self { library, bank: GraphBank::with_family(GraphPolicy::Budgeted { bytes: None }, family),
            current: [None; 40], retain_small: false, decode_rows }
    }
    pub fn get(&self, layer: usize, weights: &W) -> Option<(*mut c_void, u32)> {
        self.get_shape(layer, weights, self.current.get(layer).copied().flatten()?)
    }
    pub fn get_shape(&self, layer: usize, weights: &W, rows: u32) -> Option<(*mut c_void, u32)> {
        let owner = self.bank.get(&(layer, rows))?;
        std::ptr::eq(owner.pins, weights).then_some((owner.raw, rows))
    }
    pub fn enable_small_shapes(&mut self) { self.retain_small = true; }
    pub fn declare_startup(&mut self, keys: &[(usize, u32)]) -> Result<()> {
        ensure!(keys.iter().all(|&(layer, rows)| layer < 40 && (1..=self.decode_rows).contains(&rows)),
            "startup layer graph key exceeds decode geometry");
        self.bank.declare_startup(keys)?;
        self.retain_small = true;
        Ok(())
    }
    pub fn finish_startup(&mut self) -> Result<()> { self.bank.finish_startup() }
    pub fn capture_allowed(&self, layer: usize, rows: u32) -> bool {
        self.bank.capture_allowed(&(layer, rows))
    }
    /// # Safety
    /// All launches using the containing wave's storage have drained. The caller
    /// retains that storage until clear. On failure caller still owns graph.
    pub unsafe fn insert(&mut self, layer: usize, weights: &'w W, rows: u32, graph: *mut c_void) -> Result<()> {
        ensure!(layer < 40 && (1..=4096).contains(&rows) && !graph.is_null(), "invalid layer graph binding");
        ensure!(!self.bank.cache_attached(), "attached layer bank requires cache_insert");
        ensure!(self.capture_allowed(layer, rows), "layer graph capture outside startup inventory or after seal");
        let different_owner = self.bank.count(|(l, _)| *l == layer) > 0
            && self.current[layer].and_then(|r| self.bank.get(&(layer, r)))
                .is_some_and(|owner| !std::ptr::eq(owner.pins, weights));
        if !self.retain_small || different_owner {
            // SAFETY: caller has drained all graph launches before replacing an owner.
            unsafe { self.remove(layer)?; }
        } else if rows > self.decode_rows {
            // Large (prefill) graphs per layer: keep the most recent LARGE_SHAPES
            // (a full chunk and the decoder replay), so neither decode rounds nor
            // alternating prefill units recapture them on every request.
            let decode_rows = self.decode_rows;
            let large = self.bank.count(|&(l, r)| l == layer && r > decode_rows && r != rows);
            if large >= LARGE_SHAPES {
                self.bank.retire_oldest(|&(l, r)| l == layer && r > decode_rows && r != rows);
            }
        }
        let device = self.library.cuda_get_device()?;
        // SAFETY: the weight borrow pins its storage and the containing wave pins workspace.
        let owner = unsafe { GraphOwner::new(self.library, device, graph, weights)? };
        self.bank.census_capture_device_key(std::any::type_name::<W>(), device, &(layer, rows), graph);
        self.bank.insert((layer, rows), owner, None);
        self.current[layer] = Some(rows);
        drop(self.bank.drain_retired(|| Ok(()))?);
        tracing::debug!(target: "cuteafd::graph_capture", layer, rows, device,
            bank = self as *const Self as usize, owner = std::any::type_name::<W>(),
            binding = weights as *const W as usize,
            retained = self.bank.count(|(l, _)| *l == layer), "native layer graph captured");
        Ok(())
    }
    /// # Safety
    /// Every launch of this layer's graph completed on every consumer stream.
    pub unsafe fn remove(&mut self, layer: usize) -> Result<()> {
        ensure!(layer < 40, "invalid layer graph index");
        self.bank.retire_matching(|(l, _)| *l == layer);
        self.current[layer] = None;
        drop(self.bank.drain_retired(|| Ok(()))?);
        Ok(())
    }
    /// # Safety
    /// Every queued use of this wave's captured pointers completed.
    pub unsafe fn clear(&mut self) -> Result<()> {
        self.bank.retire_all();
        self.current = [None; 40];
        drop(self.bank.drain_retired(|| Ok(()))?);
        Ok(())
    }
    #[cfg(test)]
    pub fn len(&self) -> usize { self.bank.len() }
}

pub(crate) struct RowGraphs<'a> {
    library: &'a NativeLibrary,
    bank: GraphBank<usize, GraphOwner<'a, ()>>,
    decode_rows: usize,
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

    pub fn new(library: &'a NativeLibrary, site: &'static str, decode_rows: usize, family: &'static str) -> Self {
        Self { library, bank: GraphBank::with_family(GraphPolicy::Budgeted { bytes: None }, family),
            decode_rows, site }
    }
    pub fn get(&self, rows: usize) -> Option<*mut c_void> { self.bank.get(&rows).map(|owner| owner.raw) }
    pub fn declare_startup(&mut self, rows: &[usize]) -> Result<()> {
        ensure!(rows.iter().all(|r| (1..=self.decode_rows).contains(r)),
            "startup row graph key exceeds decode geometry");
        self.bank.declare_startup(rows)
    }
    pub fn finish_startup(&mut self) -> Result<()> { self.bank.finish_startup() }
    pub fn capture_allowed(&self, rows: usize) -> bool { self.bank.capture_allowed(&rows) }
    /// # Safety
    /// All launches on this wave's storage drained. The wave retains weights,
    /// workspace, geometry and buffers until clear, then drops them afterward.
    /// Whether inserting `rows` retires (destroys) another graph, whose
    /// queued launches must have drained first.
    pub fn would_retire(&self, rows: usize) -> bool {
        let decode_rows = self.decode_rows;
        rows > decode_rows && self.bank.count(|&r| r > decode_rows && r != rows) >= LARGE_SHAPES
    }
    pub unsafe fn insert(&mut self, rows: usize, graph: *mut c_void) -> Result<()> {
        ensure!((1..=4096).contains(&rows) && !graph.is_null(), "invalid row graph binding");
        ensure!(!self.bank.cache_attached(), "attached row bank requires cache_insert");
        ensure!(self.capture_allowed(rows), "row graph capture outside startup inventory or after seal");
        let device = self.library.cuda_get_device()?;
        // SAFETY: the containing wave owns and pins every captured allocation.
        let owner = unsafe { GraphOwner::new(self.library, device, graph, ())? };
        if rows > self.decode_rows {
            let decode_rows = self.decode_rows;
            if self.bank.count(|&r| r > decode_rows && r != rows) >= LARGE_SHAPES {
                self.bank.retire_oldest(|&r| r > decode_rows && r != rows);
            }
        }
        self.bank.census_capture_device_key(self.site, device, &rows, graph);
        self.bank.insert(rows, owner, None);
        drop(self.bank.drain_retired(|| Ok(()))?);
        tracing::debug!(target: "cuteafd::graph_capture", site = self.site, rows, device,
            bank = self as *const Self as usize, retained = self.bank.len(), "native row graph captured");
        Ok(())
    }
    /// # Safety
    /// Every queued use completed before releasing graph executables or storage.
    pub unsafe fn clear(&mut self) -> Result<()> {
        self.bank.retire_all();
        drop(self.bank.drain_retired(|| Ok(()))?);
        Ok(())
    }
    #[cfg(test)]
    pub fn len(&self) -> usize { self.bank.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unattached_wrappers_preserve_legacy_decisions_and_stats() -> Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let mut rows = RowGraphs::new(&library, "fixture", 64, "test");
        let weights = ();
        let mut layers = LayerGraphs::new(&library, 64, "test");
        assert!(rows.capture_allowed(80) && layers.capture_allowed(0, 80));
        assert!(!rows.bank.cache_attached() && !layers.bank.cache_attached());
        // SAFETY: the fake DSO records no device work; these owners never launch.
        unsafe {
            for width in [1, 80, 96, 128] {
                library.cuda_graph_begin_capture(std::ptr::null_mut())?;
                let graph = library.cuda_graph_end_capture(std::ptr::null_mut())?;
                rows.insert(width, graph)?;
            }
        }
        assert!(rows.get(1).is_some() && rows.get(80).is_none());
        assert!(rows.get(96).is_some() && rows.get(128).is_some());
        assert_eq!(rows.bank.stats(), super::super::GraphStats { captures: 4, ..Default::default() });
        assert_eq!(rows.bank.retired_len(), 0);
        assert!(rows.would_retire(160));
        layers.declare_startup(&[(0, 1)])?;
        // SAFETY: fixture graph has no queued work and weights outlive the bank.
        unsafe {
            library.cuda_graph_begin_capture(std::ptr::null_mut())?;
            layers.insert(0, &weights, 1, library.cuda_graph_end_capture(std::ptr::null_mut())?)?;
        }
        layers.finish_startup()?;
        assert!(!layers.capture_allowed(0, 1) && !layers.capture_allowed(0, 6));
        assert_eq!(layers.bank.stats(), super::super::GraphStats { captures: 1, ..Default::default() });
        assert!(layers.get_shape(0, &weights, 1).is_some());
        Ok(())
    }

    #[test]
    fn capture_ticket_cancels_and_commits_surviving_owner() -> Result<()> {
        use super::super::device::{Basis, Preference};
        let mut bank = GraphBank::<usize, ()>::new(None);
        bank.attach_device(-8801, 64 << 20, Site { name: "ticket-fixture", cycle: Cycle::Other,
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
        let mut direct = GraphBank::<u8, ()>::new(None);
        let mut layers = LayerGraphs::<'_, '_, ()>::new(&library, 64, "test");
        let mut rows = RowGraphs::new(&library, "fixture", 64, "test");
        let mut bindings = BindingGraphs::<u16>::new(&library, "fixture")?;
        let banks: [&mut dyn CacheAttached; 4] = [&mut direct, &mut layers, &mut rows, &mut bindings];
        for bank in banks {
            bank.attach(-8802, 64 << 20, Site { name: "visitor-fixture", cycle: Cycle::Other,
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
