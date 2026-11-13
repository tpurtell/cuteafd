//! Decode-only, exact-geometry graph segments. Host metadata uploads and every
//! Spark/TP2 exchange remain outside capture; all recorded pointers belong to
//! the backend's persistent rank resources.
use crate::shared::{decode_graph::{self, device::{Basis, Cycle, Preference, Site}, GraphBank, GraphOwner},
    memory::device::Device};
use anyhow::Result;
use std::{cell::RefCell, collections::HashSet, ffi::c_void};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Section { Attention, Frontend, LocalExperts, Residual, Head }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Key {
    pub layer: usize,
    pub rows: usize,
    pub stride: usize,
    pub section: Section,
}

pub(crate) struct Graphs<'a> {
    device: Device<'a>,
    bank: RefCell<GraphBank<Key, GraphOwner<'a, ()>>>,
    singleton: RefCell<HashSet<Key>>,
    pins: RefCell<HashSet<usize>>,
}

impl<'a> Graphs<'a> {
    pub(crate) fn new(device: Device<'a>, rank: usize, ranks: usize, onboard_layers: usize) -> Result<Self> {
        let reserve = cuteafd_loader::serving_capacity::muse_graphs::policy(ranks, onboard_layers, rank)?;
        let mut bank = GraphBank::new();
        bank.attach_device(device.library, device.id, reserve.bytes, Site {
            name: "muse_spark.segment", cycle: Cycle::Decode, preference: Preference::Adaptive,
            bootstrap: 2 << 20, basis: Basis::Bytes, spans_devices: false,
        })?;
        decode_graph::device::set_reserve_policy(device.id, reserve)?;
        Ok(Self { device, bank: RefCell::new(bank), singleton: RefCell::new(HashSet::new()), pins: RefCell::new(HashSet::new()) })
    }

    pub(crate) fn pin_rows(&self, rows: usize) { self.pins.borrow_mut().insert(rows); }

    pub(crate) fn cycle(&self, cycle: Cycle) { self.bank.borrow_mut().cache_cycle(cycle); }

    pub(crate) fn run(&self, key: Key, stream: *mut c_void, mut segment: impl FnMut() -> Result<()>) -> Result<()> {
        if let Some(raw) = self.bank.borrow_mut().cached_launch(&key).map(|graph| graph.raw) {
            // SAFETY: the backend retains rank storage and drains before dropping the banks.
            return unsafe { crate::shared::decode_graph::launch(&self.device.library, raw, stream) };
        }
        if self.singleton.borrow().contains(&key) { return segment(); }
        let Some(ticket) = self.bank.borrow_mut().cache_ticket(&key, self.pins.borrow().contains(&key.rows), None) else { return segment() };
        // SAFETY: callers submit only device work on this stream, using persistent storage.
        if let Err(error) = unsafe { crate::shared::decode_graph::begin_capture(&self.device.library, stream, &ticket) } {
            if self.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                tracing::warn!(%error, "Muse graph capture unavailable; running eagerly");
            }
            return segment();
        }
        let captured = segment();
        // Always end capture, including a failed launch, before unwinding the ticket.
        // SAFETY: this is the stream whose capture began above.
        let executable = unsafe { self.device.library.cuda_graph_end_capture_topology(stream) };
        if let Err(error) = captured {
            self.bank.borrow_mut().cache_ticket_failed(ticket, executable.as_ref().err().unwrap_or(&error));
            if let Ok((raw, ..)) = executable {
                // SAFETY: this executable has never been launched.
                unsafe { self.device.library.cuda_graph_exec_destroy(raw)?; }
            }
            return Err(error);
        }
        let (raw, nodes, width, spans_devices, inspection_error) = match executable {
            Ok(info) => info,
            Err(error) => {
                if self.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, "Muse graph instantiation unavailable; running eagerly");
                }
                return segment();
            }
        };
        if nodes < 2 || inspection_error.is_some() {
            // SAFETY: capture did not execute the segment; discard and enqueue it eagerly.
            unsafe { self.device.library.cuda_graph_exec_destroy(raw)?; }
            drop(ticket);
            self.singleton.borrow_mut().insert(key);
            return segment();
        }
        decode_graph::census::capture_device_key("muse_spark", "segment", self as *const Self as usize,
            self.device.id, None, &key, raw);
        // SAFETY: pins are empty; the enclosing backend borrows every rank owner,
        // and its terminal drain precedes bank/local/drafter storage destruction.
        let graph = unsafe { GraphOwner::new(self.device.library, self.device.id, raw, ())? };
        self.bank.borrow_mut().cache_commit(key, graph, ticket, nodes as u64, width, spans_devices);
        // SAFETY: newly captured executable, with the same persistent operands.
        unsafe { crate::shared::decode_graph::launch(&self.device.library, raw, stream) }
    }

    pub(crate) fn retains_rows(&self, rows: usize) -> bool {
        self.bank.borrow().keys().iter().any(|key| key.rows == rows)
    }

    pub(crate) fn needs_boundary(&self) -> bool { self.bank.borrow().cache_needs_boundary() }
    pub(crate) fn tick(&self) { self.bank.borrow_mut().cache_tick(); }
    pub(crate) fn visit(&self, visitor: &mut decode_graph::device::GraphVisitor<'_>) -> Result<()> {
        visitor(self.device.id, "muse_spark.segment", false, &mut *self.bank.borrow_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reconciliation_uses_final_admission_and_shared_joined_publish() {
        let serve = include_str!("serve.rs");
        let provenance = serve.find("memory_reconciliation_provenance").unwrap();
        assert!(serve.find("let placement = &built.placement").unwrap() < provenance);
        assert!(provenance < serve.find("Ready::load").unwrap());
        let scheduler = include_str!("../../shared/qwen/scheduler.rs");
        assert!(scheduler.contains("static FIRST: std::sync::Once"));
        assert!(scheduler.contains("reconcile::capture(\"first_request\")"));
        assert!(!include_str!("backend.rs").contains("graph_capture_stats"));
    }

    #[test]
    fn keys_distinguish_captured_geometry_without_pointer_or_slot_phases() {
        let key = Key { layer: 3, rows: 7, stride: 65, section: Section::Attention };
        let keys = HashSet::from([key, Key { rows: 8, ..key }, Key { stride: 66, ..key },
            Key { layer: 4, ..key }, Key { section: Section::Frontend, ..key }]);
        assert_eq!(keys.len(), 5);
    }
}
