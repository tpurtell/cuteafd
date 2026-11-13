//! Persistent MiniMax sections on the shared per-device ARC graph cache.
//! Placement selects sites, not keys; each owner pins its stream and storage.
//! Bootstrap prewarm is unpinned until a checked-in trace exists. GRAPH_TRACE
//! emission is shared; generating a MiniMax prewarm list with
//! `graph-prewarm-from-trace.py --family minimax_m3` depends on #1938's generic
//! row-sites support. No family-local trace parser or memory model belongs here.
use crate::shared::{decode_graph::{self, device::{self, Basis, Cycle, Estimate, Preference, Site},
    GraphBank, GraphOwner}, memory::device::Stream};
use anyhow::{ensure, Result};
use std::{collections::HashSet, rc::Rc};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Op { Attention, Router, Ffn, JoinRouted }

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Key { pub rank: usize, pub layer: usize, pub rows: usize, pub op: Op }

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = match self.op { Op::Attention => "attention", Op::Router => "router",
            Op::Ffn => "ffn", Op::JoinRouted => "join_routed" };
        write!(f, "rank={} layer={} rows={} op={op}", self.rank, self.layer, self.rows)
    }
}

/// Rates/estimates come from accepted inventory. Until a MiniMax node census
/// exists, use the shared byte basis and its measured per-site bootstrap.
pub(crate) struct Config {
    pub reserve: u64,
    pub bootstrap: u64,
    pub basis: Basis,
    pub spans_devices: bool,
}
impl Config {
    fn site(&self) -> Site {
        Site { name: "minimax_m3.section", cycle: Cycle::Other,
            preference: Preference::Adaptive, bootstrap: self.bootstrap,
            basis: self.basis, spans_devices: self.spans_devices }
    }
}

/// P retains weights, cache planes, metadata and this execution owner's scratch.
/// No request, page ID, placement mode or host address is part of graph identity.
pub(crate) struct Graphs<'a, P> {
    bank: GraphBank<Key, GraphOwner<'a, Rc<P>>>,
    stream: Rc<Stream<'a>>,
    pins: Rc<P>,
    rank: usize,
    layers: usize,
    capacity: usize,
    sites: HashSet<(usize, Op)>,
    touched: HashSet<Key>,
    eager: HashSet<Key>,
    first_touch_pending: bool,
    spans_devices: bool,
}

impl<P> Drop for Graphs<'_, P> {
    fn drop(&mut self) {
        self.stream.drain_or_abort("MiniMax graph storage retirement");
        self.bank.cache_clear_drained();
    }
}

/// Project the admitted sites onto ordinary decode rows without inventing sites
/// for a particular placement. The admitted endpoint itself is always included.
pub(crate) fn bootstrap_keys(rank: usize, capacity: usize, sites: &[(usize, Op)]) -> Result<Vec<Key>> {
    ensure!(rank < 2 && (1..=64).contains(&capacity), "MiniMax graph bootstrap geometry");
    let unique = sites.iter().copied().collect::<HashSet<_>>();
    ensure!(unique.len() == sites.len(), "MiniMax duplicate graph site");
    let mut rows = decode_graph::ROW_BUCKETS.into_iter().filter(|&rows| rows <= capacity).collect::<Vec<_>>();
    if rows.last() != Some(&capacity) { rows.push(capacity); }
    Ok(sites.iter().flat_map(|&(layer, op)| rows.iter().map(move |&rows|
        Key { rank, layer, rows, op })).collect())
}

impl<'a, P> Graphs<'a, P> {
    pub(crate) fn new(stream: Rc<Stream<'a>>, pins: Rc<P>, rank: usize, layers: usize,
        capacity: usize, sites: &[(usize, Op)], config: Config) -> Result<Self> {
        ensure!(rank < 2 && layers > 0 && (1..=64).contains(&capacity)
            && sites.iter().all(|&(layer, _)| layer < layers), "MiniMax graph admitted geometry");
        let unique = sites.iter().copied().collect::<HashSet<_>>();
        ensure!(unique.len() == sites.len(), "MiniMax duplicate graph site");
        let mut bank = GraphBank::with_family("minimax_m3");
        bank.attach_device(stream.device.library, stream.device.id, config.reserve, config.site())?;
        Ok(Self { bank, stream, pins, rank, layers, capacity, sites: unique,
            touched: HashSet::new(), eager: HashSet::new(), first_touch_pending: false,
            spans_devices: config.spans_devices })
    }

    fn validate(&self, key: Key) -> Result<()> {
        ensure!(key.rank == self.rank && key.layer < self.layers
            && (1..=self.capacity).contains(&key.rows) && self.sites.contains(&(key.layer, key.op)),
            "MiniMax graph key differs from admitted execution owner");
        Ok(())
    }

    pub(crate) fn cycle(&mut self, cycle: Cycle) { self.bank.cache_cycle(cycle); }
    pub(crate) fn needs_boundary(&self) -> bool {
        self.first_touch_pending || self.bank.cache_needs_boundary()
    }
    pub(crate) fn observed(&mut self) { self.first_touch_pending = false; }
    pub(crate) fn tick(&mut self) { self.bank.cache_tick(); }
    pub(crate) fn visit(&mut self, visit: &mut device::GraphVisitor<'_>) -> Result<()> {
        visit(self.stream.device.id, "minimax_m3.section", self.spans_devices, &mut self.bank)
    }
    pub(crate) fn clear(&mut self) -> Result<()> {
        self.stream.drain()?;
        self.bank.cache_clear_drained();
        Ok(())
    }

    /// # Safety
    /// `work` launches only on this retained stream, using persistent storage
    /// retained by P. Its pointers, shapes, workspaces and launch scalars are
    /// invariant for `key` until clear/drop. Changing context-dependent values
    /// must be device metadata uploaded before the section, never captured host
    /// scalars. Host joins, uploads and allocation stay outside sections.
    pub(crate) unsafe fn run(&mut self, key: Key, estimate: Option<Estimate>,
        mut work: impl FnMut() -> Result<()>) -> Result<()> {
        self.validate(key)?;
        if self.eager.contains(&key) { return work(); }
        // Match Step-5: initialize modules/workspaces eagerly, then observe the
        // completed step before a later miss asks the shared cache to capture.
        if self.touched.insert(key) {
            self.first_touch_pending = true;
            return work();
        }
        // Prefill still has live-context launch scalars; fixed-cap decode needs
        // its exact-replay oracle before either attention route may capture.
        if key.op == Op::Attention { return work(); }
        if let Some(graph) = self.bank.cached_launch(&key) {
            let raw = graph.raw;
            return self.stream.device.run(|| {
                // SAFETY: P retains captured storage and this owner's exact stream.
                unsafe { decode_graph::launch(self.stream.device.library, raw, self.stream.raw) }
            });
        }
        let Some(ticket) = self.bank.cache_ticket(&key, false, estimate) else { return work() };
        let device = self.stream.device;
        device.run(|| {
            let library = device.library;
            // SAFETY: the shared ticket admits this persistent-stream recording.
            if let Err(error) = unsafe { decode_graph::begin_capture(library, self.stream.raw, &ticket) } {
                if self.bank.cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, ?key, "MiniMax capture unavailable; running eagerly");
                }
                return work();
            }
            let result = work();
            // SAFETY: end the matching stream capture even when work failed.
            let captured = unsafe { library.cuda_graph_end_capture_topology(self.stream.raw) };
            if let Err(error) = result {
                self.bank.cache_ticket_failed(ticket, captured.as_ref().err().unwrap_or(&error));
                if let Ok((raw, ..)) = captured {
                    // SAFETY: a failed recording has never launched.
                    unsafe { library.cuda_graph_exec_destroy(raw)?; }
                }
                return Err(error);
            }
            let (raw, nodes, width, spans, inspection_error) = match captured {
                Ok(topology) => topology,
                Err(error) => {
                    if self.bank.cache_ticket_failed(ticket, &error) {
                        tracing::warn!(%error, ?key, "MiniMax instantiate unavailable; running eagerly");
                    }
                    return work();
                }
            };
            // SAFETY: exclusively owned, not yet launched; P pins all storage.
            let graph = unsafe { GraphOwner::new(library, self.stream.device.id, raw, Rc::clone(&self.pins))? };
            if !retain(nodes, inspection_error.is_none()) {
                self.eager.insert(key);
                drop(graph);
                drop(ticket);
                return work();
            }
            self.bank.census_capture_key("section", key.rank as i32, None, &key, raw);
            self.bank.cache_commit(key, graph, ticket, nodes as u64, width, spans || self.spans_devices);
            // SAFETY: newly committed executable and storage remain retained.
            unsafe { decode_graph::launch(library, raw, self.stream.raw) }
        })
    }

    /// Unpinned startup bootstrap. Declined keys remain ordinary lazy misses.
    /// # Safety
    /// Every supplied body satisfies run's stable-storage/scalar contract.
    pub(crate) unsafe fn prewarm(&mut self, keys: &[Key],
        mut estimate: impl FnMut(Key) -> Option<Estimate>, mut work: impl FnMut(Key) -> Result<()>) -> Result<()> {
        for &key in keys { self.validate(key)?; }
        ensure!(keys.iter().collect::<HashSet<_>>().len() == keys.len(), "MiniMax duplicate prewarm key");
        self.cycle(Cycle::Other);
        for &key in keys {
            // SAFETY: forwarded persistent pointer/shape/workspace contract.
            unsafe { self.run(key, estimate(key), || work(key))?; }
            self.stream.drain()?;
            // SAFETY: the adapter, not individual graph pins, retains every
            // captured allocation; this stream has drained before eviction.
            unsafe { device::observe_completed(self.stream.device.library, |visit| self.visit(visit))?; }
            self.observed();
            // SAFETY: first touch initialized this same body's persistent storage.
            unsafe { self.run(key, estimate(key), || work(key))?; }
            self.stream.drain()?;
            // SAFETY: completed boundary; adapter pins survive graph eviction.
            unsafe { device::observe_completed(self.stream.device.library, |visit| self.visit(visit))?; }
            self.observed();
        }
        Ok(())
    }
}

fn retain(nodes: usize, inspected: bool) -> bool { inspected && nodes >= 2 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_is_bounded_placement_agnostic_and_unpinned() -> Result<()> {
        let sites = [(0, Op::Attention), (3, Op::Ffn), (3, Op::Router), (3, Op::JoinRouted)];
        for capacity in [1, 3, 16, 64] {
            for rank in [0, 1] {
                let keys = bootstrap_keys(rank, capacity, &sites)?;
                assert!(keys.iter().all(|key| key.rank == rank && key.rows <= capacity
                    && sites.contains(&(key.layer, key.op))));
                assert_eq!(keys.iter().map(|key| (key.layer, key.op)).collect::<HashSet<_>>().len(), sites.len());
                assert_eq!(keys.iter().filter(|key| key.rows == capacity).count(), sites.len());
            }
        }
        assert!(bootstrap_keys(2, 64, &sites).is_err());
        assert!(bootstrap_keys(0, 0, &sites).is_err());
        assert!(bootstrap_keys(0, 65, &sites).is_err());
        assert!(bootstrap_keys(0, 64, &[(0, Op::Ffn), (0, Op::Ffn)]).is_err());
        Ok(())
    }

    #[test]
    fn exact_keys_distinguish_every_geometry_and_trace_field() {
        let key = Key { rank: 0, layer: 1, rows: 16, op: Op::Attention };
        let variants = [key, Key { rank: 1, ..key }, Key { layer: 2, ..key },
            Key { rows: 4, ..key }, Key { op: Op::Ffn, ..key }];
        assert_eq!(variants.into_iter().collect::<HashSet<_>>().len(), variants.len());
        assert_eq!(format!("{key:?}"), "rank=0 layer=1 rows=16 op=attention");
        assert!(!retain(1, true));
        assert!(!retain(8, false));
        assert!(retain(2, true));
    }

    fn fixture_bank<'a, P>(stream: Rc<Stream<'a>>, pins: Rc<P>, synthetic: i32) -> Result<Graphs<'a, P>> {
        let config = Config { reserve: 64 << 20, bootstrap: 2 << 20,
            basis: Basis::Bytes, spans_devices: false };
        let mut bank = GraphBank::with_family("minimax_m3");
        bank.attach_device_for_test(synthetic, config.reserve, config.site())?;
        Ok(Graphs { bank, stream, pins, rank: 1, layers: 2, capacity: 16,
            sites: [(1, Op::Attention), (1, Op::Ffn)].into_iter().collect(), touched: HashSet::new(),
            eager: HashSet::new(), first_touch_pending: false, spans_devices: false })
    }

    #[test]
    fn first_touch_is_eager_small_sections_stay_eager_and_keys_fail_closed() -> Result<()> {
        use crate::shared::memory::device::Device;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let stream = Rc::new(Stream::new(Device { library: &library, id: 1 })?);
        let mut graphs = fixture_bank(Rc::clone(&stream), Rc::new(()), -1955)?;
        let key = Key { rank: 1, layer: 1, rows: 16, op: Op::Ffn };
        let mut calls = 0;
        // SAFETY: fake DSO captures no real pointers or device work.
        unsafe { graphs.run(key, None, || { calls += 1; Ok(()) })?; }
        assert_eq!(calls, 1);
        assert!(graphs.needs_boundary());
        assert!(!fixture.events()?.contains('G'));
        graphs.observed();
        // The fixture reports one node: the recording is discarded, then the
        // body runs eagerly once. Future calls cannot retry a pointless graph.
        // SAFETY: fake DSO captures no real pointers or device work.
        unsafe { graphs.run(key, None, || { calls += 1; Ok(()) })?; }
        assert_eq!(calls, 3);
        assert!(graphs.eager.contains(&key));
        let captures = fixture.events()?.matches('G').count();
        assert_eq!(captures, 1);
        for _ in 0..8 {
            // SAFETY: this permanently eager body submits no work.
            unsafe { graphs.run(key, None, || { calls += 1; Ok(()) })?; }
        }
        assert_eq!(fixture.events()?.matches('G').count(), captures);
        for invalid in [Key { rank: 0, ..key }, Key { layer: 0, ..key },
            Key { rows: 0, ..key }, Key { rows: 17, ..key }, Key { op: Op::Router, ..key }] {
            // SAFETY: invalid keys are rejected without recording this closure.
            assert!(unsafe { graphs.run(invalid, None, || panic!("invalid graph body")) }.is_err());
        }
        graphs.clear()?;
        Ok(())
    }

    #[test]
    fn attention_with_changing_context_stays_eager_in_every_cycle() -> Result<()> {
        use crate::shared::memory::device::Device;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let stream = Rc::new(Stream::new(Device { library: &library, id: 1 })?);
        let mut graphs = fixture_bank(stream, Rc::new(()), -1960)?;
        let key = Key { rank: 1, layer: 1, rows: 16, op: Op::Attention };
        let contexts = [1, 63, 64, 65, 128, 513, 4096];
        let mut seen = Vec::new();
        for cycle in [Cycle::Other, Cycle::Decode, Cycle::Prefill] {
            graphs.cycle(cycle);
            for context in contexts {
                // SAFETY: fake body submits no native launches or device pointers.
                unsafe { graphs.run(key, None, || { seen.push(context); Ok(()) })?; }
                graphs.observed();
            }
        }
        assert_eq!(seen, contexts.repeat(3));
        assert_eq!(graphs.bank.len(), 0);
        assert!(!fixture.events()?.contains('G'));
        assert!(!fixture.events()?.contains('L'));
        Ok(())
    }

    #[test]
    fn replay_uses_exact_stream_retains_storage_and_drains_before_release() -> Result<()> {
        use crate::shared::memory::device::Device;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let stream = Rc::new(Stream::new(Device { library: &library, id: 1 })?);
        let pins = Rc::new(());
        let weak = Rc::downgrade(&pins);
        let mut graphs = fixture_bank(Rc::clone(&stream), Rc::clone(&pins), -1956)?;
        let key = Key { rank: 1, layer: 1, rows: 16, op: Op::Ffn };
        graphs.touched.insert(key);
        let ticket = graphs.bank.cache_ticket(&key, false, None).expect("shared admission");
        let raw = stream.device.run(|| {
            // SAFETY: fixture recording has no real kernels or pointers.
            unsafe {
                decode_graph::begin_capture(&library, stream.raw, &ticket)?;
                library.cuda_graph_end_capture(stream.raw)
            }
        })?;
        // SAFETY: unlaunched fake graph; graph and adapter both retain storage.
        let owner = unsafe { GraphOwner::new(&library, 1, raw, Rc::clone(&pins))? };
        graphs.bank.cache_commit(key, owner, ticket, 2, 1, false);
        assert_eq!(Rc::strong_count(&pins), 3);
        drop(pins);
        for _ in 0..12 {
            // SAFETY: fixture owns the graph and stable stream; no real work.
            unsafe { graphs.run(key, None, || panic!("resident graph must replay"))?; }
        }
        assert_eq!(fixture.events()?.matches('L').count(), 12);
        assert_eq!(fixture.events()?.matches('G').count(), 1);
        assert!(weak.upgrade().is_some());
        graphs.clear()?;
        assert!(weak.upgrade().is_some(), "adapter pins outlive eviction");
        assert_eq!(fixture.events()?.matches('g').count(), 1);
        library.cuda_set_device(0)?;
        drop(graphs);
        assert!(weak.upgrade().is_none());
        assert_eq!(library.cuda_get_device()?, 0);
        Ok(())
    }

    #[test]
    fn zero_reserve_uses_shared_eager_decision_without_capture() -> Result<()> {
        let config = Config { reserve: 0, bootstrap: 2 << 20, basis: Basis::Bytes, spans_devices: false };
        let mut bank = GraphBank::<Key, ()>::with_family("minimax_m3");
        bank.attach_device_for_test(-1945, config.reserve, config.site())?;
        let keys = bootstrap_keys(0, 64, &[(0, Op::Attention), (0, Op::Ffn)])?;
        for cycle in [Cycle::Other, Cycle::Decode, Cycle::Prefill] {
            bank.cache_cycle(cycle);
            for key in &keys { assert!(bank.cache_ticket(key, false, None).is_none()); }
            bank.cache_tick();
        }
        assert_eq!(bank.len(), 0);
        assert!(!bank.cache_needs_boundary());
        Ok(())
    }
}
