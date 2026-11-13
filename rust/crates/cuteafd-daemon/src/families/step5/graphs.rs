//! Persistent layer sections; transactional KV writes and host joins stay eager.
use crate::shared::{decode_graph::{device::{self, Basis, Cycle, Preference, Site}, GraphBank,
    GraphOwner}, memory::device::Stream};
use anyhow::Result;
use std::{cell::{Cell, RefCell}, collections::HashSet};
use cuteafd_loader::serving_capacity::step5_graphs::{self, Step5GraphLayout};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Section { Projection, Attention, Dense, Frontend, Shared, Finish }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Key { pub layer: usize, pub rows: usize, pub section: Section }

pub(super) struct Graphs<'a> {
    bank: RefCell<GraphBank<Key, GraphOwner<'a, ()>>>,
    // A section found to contain fewer than two nodes always stays eager.
    eager: RefCell<HashSet<Key>>,
    touched: RefCell<HashSet<Key>>,
    device: i32,
    pins: HashSet<usize>,
    warming: Cell<bool>,
    first_touch_pending: Cell<bool>,
}

impl<'a> Graphs<'a> {
    pub(super) fn new() -> Self {
        Self { bank: RefCell::new(GraphBank::with_family("step5")),
            eager: RefCell::new(HashSet::new()), touched: RefCell::new(HashSet::new()), device: -1,
            pins: HashSet::new(), warming: Cell::new(false), first_touch_pending: Cell::new(false) }
    }

    pub(super) fn attach(&mut self, library: &'a cuteafd_ffi::NativeLibrary, device: i32,
        rank: usize, layout: Option<Step5GraphLayout>, max_rows: usize) -> Result<()> {
        let (base, tail) = step5_graphs::reserve(layout, rank);
        let policy = cuteafd_loader::placement::graph_reserve::current(rank, false, base, tail)?;
        self.pins = step5_graphs::selected_rows(layout, max_rows)?.into_iter()
            .filter_map(|(rows, pin)| pin.then_some(rows)).collect();
        self.bank.get_mut().attach_device(library, device, policy.bytes, Site {
            name: "step5.section", cycle: Cycle::Other, preference: Preference::Adaptive,
            bootstrap: 2 << 20, basis: Basis::Bytes, spans_devices: false,
        })?;
        device::set_reserve_policy(device, policy)?;
        self.device = device;
        if step5_graphs::prewarm(layout).is_none() {
            tracing::warn!(?layout, rank, reserve = policy.bytes,
                "Step-5 untraced graph layout: no pins or measured base; using byte bootstrap estimate");
        }
        Ok(())
    }

    pub(super) fn warming(&self, enabled: bool) { self.warming.set(enabled); }
    pub(super) fn cycle(&self, cycle: Cycle) {
        self.bank.borrow_mut().cache_cycle(if self.warming.get() { Cycle::Other } else { cycle });
    }
    pub(super) fn clear_drained(&self) { self.bank.borrow_mut().cache_clear_drained(); }
    pub(super) fn retained_rows(&self, rows: usize) -> usize {
        self.bank.borrow().count(|key| key.rows == rows)
    }
    pub(super) fn needs_boundary(&self) -> bool {
        self.first_touch_pending.get() || self.bank.borrow().cache_needs_boundary()
    }
    pub(super) fn observed(&self) { self.first_touch_pending.set(false); }
    pub(super) fn tick(&self) { self.bank.borrow_mut().cache_tick(); }
    pub(super) fn visit(&self, visit: &mut device::GraphVisitor<'_>) -> Result<()> {
        visit(self.device, "step5.section", false, &mut *self.bank.borrow_mut())
    }

    pub(super) fn run(&self, key: Key, stream: &Stream<'a>, work: impl Fn() -> Result<()>) -> Result<()> {
        if self.eager.borrow().contains(&key) { return work(); }
        // First use initializes modules/workspaces eagerly; the outer step drains
        // before a later use can reserve capture growth for this geometry.
        if self.touched.borrow_mut().insert(key) {
            self.first_touch_pending.set(true);
            return work();
        }
        if let Some(raw) = self.bank.borrow_mut().cached_launch(&key).map(|graph| graph.raw) {
            return stream.device.run(|| {
                // SAFETY: containing execution and staging owners retain every captured buffer;
                // all rank streams drain before retirement or scratch reuse by another step.
                unsafe { crate::shared::decode_graph::launch(&stream.device.library, raw, stream.raw) }
            });
        }
        let Some(ticket) = self.bank.borrow_mut().cache_ticket(&key, self.pins.contains(&key.rows), None) else { return work() };
        stream.device.run(|| {
            let library = stream.device.library;
            // SAFETY: sections contain only ordered launches on this stream, with persistent storage.
            if let Err(error) = unsafe { crate::shared::decode_graph::begin_capture(&library, stream.raw, &ticket) } {
                if self.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, ?key, "Step-5 capture unavailable; running eagerly");
                }
                return work();
            }
            let result = work();
            // SAFETY: always end this stream's capture, including a failed section.
            let captured = unsafe { library.cuda_graph_end_capture_topology(stream.raw) };
            if let Err(error) = result {
                self.bank.borrow_mut().cache_ticket_failed(ticket, captured.as_ref().err().unwrap_or(&error));
                if let Ok((raw, ..)) = captured {
                    // SAFETY: failed capture has never launched.
                    unsafe { library.cuda_graph_exec_destroy(raw)?; }
                }
                return Err(error);
            }
            let (raw, nodes, width, spans_devices, inspection_error) = match captured {
                Ok(topology) => topology,
                Err(error) => {
                    if self.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                        tracing::warn!(%error, ?key, "Step-5 instantiate unavailable; running eagerly");
                    }
                    return work();
                }
            };
            // SAFETY: exclusive executable ownership; all captured allocations belong to the
            // containing execution or its borrowed staging/cache owners, never to this pin.
            let graph = unsafe { GraphOwner::new(library, stream.device.id, raw, ())? };
            if !retain(nodes, inspection_error.is_none()) {
                self.eager.borrow_mut().insert(key);
                drop(graph);
                drop(ticket);
                return work();
            }
            self.bank.borrow_mut().cache_commit(key, graph, ticket, nodes as u64, width, spans_devices);
            // SAFETY: the committed executable retains the same persistent pointer contract.
            unsafe { crate::shared::decode_graph::launch(&library, raw, stream.raw) }
        })
    }
}

fn retain(nodes: usize, inspected: bool) -> bool { inspected && nodes >= 2 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sections_never_retain_single_node_or_unknown_topology() {
        assert!(!retain(0, true));
        assert!(!retain(1, true));
        assert!(!retain(10, false));
        assert!(retain(2, true));
    }
    #[test]
    fn untraced_zero_reserve_runs_every_section_eagerly_without_capture() {
        let policy = cuteafd_loader::placement::graph_reserve::resolve(None, 0, false, 0, 0).unwrap();
        assert_eq!(policy.bytes, 0);
        let mut bank: GraphBank<Key, ()> = GraphBank::with_family("step5");
        bank.attach_device_for_test(-1639, policy.bytes, Site {
            name: "step5.section", cycle: Cycle::Decode, preference: Preference::Adaptive,
            bootstrap: 2 << 20, basis: Basis::Bytes, spans_devices: false,
        }).unwrap();
        for cycle in [Cycle::Decode, Cycle::Verify, Cycle::Prefill] {
            bank.cache_cycle(cycle);
            for layer in 0..92 {
                for section in [Section::Projection, Section::Attention, Section::Dense,
                    Section::Frontend, Section::Shared, Section::Finish] {
                    for rows in [1, 4, 16, 64] {
                        assert!(bank.cache_ticket(&Key { layer, rows, section }, false, None).is_none());
                    }
                }
            }
            bank.cache_tick();
            assert!(!bank.cache_needs_boundary());
        }
        assert!(bank.cached_launch(&Key { layer: 0, rows: 1, section: Section::Projection }).is_none());
    }

    #[test]
    fn keys_are_geometry_only() {
        let key = Key { layer: 8, rows: 16, section: Section::Attention };
        assert_eq!(key, Key { layer: 8, rows: 16, section: Section::Attention });
        assert_ne!(key, Key { rows: 8, ..key });
        assert_ne!(key, Key { section: Section::Projection, ..key });
    }
}
