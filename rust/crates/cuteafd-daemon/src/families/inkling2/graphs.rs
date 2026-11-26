//! Exact-width, rank-local sections on the shared device graph bank.
use crate::shared::{
    decode_graph::{self, device::{self, Basis, Cycle, Preference, Site}, GraphBank, GraphOwner},
    memory::device::Stream,
};
use anyhow::Result;
use cuteafd_loader::placement::graph_reserve::Resolved;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Mode {
    Decode,
    Verify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Section {
    WholeAttention,
    RankAttention,
    AfterAttention,
    Routes,
    ExpertFinish,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    layer: usize,
    rows: usize,
    section: Section,
    mode: Mode,
}

pub(super) struct DecodeGraphs<'a> {
    bank: GraphBank<Key, GraphOwner<'a, ()>>,
    first_touched: HashSet<Key>,
    first_touch_pending: bool,
    unsupported: HashSet<Key>,
    rank: usize,
    device: i32,
}

impl<'a> DecodeGraphs<'a> {
    pub(super) fn new(stream: &Stream<'a>, rank: usize, policy: Resolved) -> Result<Option<Self>> {
        if policy.bytes == 0 {
            return Ok(None);
        }
        let mut bank = GraphBank::with_family("inkling2");
        let device = stream.device.id;
        bank.attach_device(stream.device.library, device, policy.bytes, Site {
            name: "inkling2.section", cycle: Cycle::Decode, preference: Preference::Adaptive,
            bootstrap: 2 << 20, basis: Basis::Bytes,
            spans_devices: super::graph_sites::spans("inkling2.section", false),
        })?;
        device::set_reserve_policy(device, policy)?;
        Ok(Some(Self { bank, first_touched: HashSet::new(), first_touch_pending: false,
            unsupported: HashSet::new(), rank, device }))
    }

    pub(super) fn cycle(&mut self, cycle: Cycle) { self.bank.cache_cycle(cycle); }
    pub(super) fn needs_boundary(&self) -> bool {
        self.first_touch_pending || self.bank.cache_needs_boundary()
    }
    pub(super) fn observed(&mut self) { self.first_touch_pending = false; }
    pub(super) fn tick(&mut self) { self.bank.cache_tick(); }

    pub(super) fn visit(&mut self, visit: &mut device::GraphVisitor<'_>) -> Result<()> {
        visit(self.device, "inkling2.section", false, &mut self.bank)
    }

    pub(super) fn retained(&self, rows: usize, mode: Mode) -> usize {
        self.bank.count(|key| key.rows == rows && key.mode == mode)
    }

    pub(super) fn clear_drained(&mut self) { self.bank.cache_clear_drained(); }

    fn run(&mut self, key: Key, stream: &Stream<'a>, work: impl Fn() -> Result<()>) -> Result<()> {
        if self.unsupported.contains(&key) { return work(); }
        if self.first_touched.insert(key) {
            // Initialize this exact arithmetic route eagerly before measuring a capture.
            self.first_touch_pending = true;
            return work();
        }
        let library = stream.device.library;
        if let Some(raw) = self.bank.cached_launch(&key).map(|owner| owner.raw) {
            // SAFETY: the same decode stream, geometry and rank-local owners outlive replay.
            return stream.device.run(|| unsafe { decode_graph::launch(library, raw, stream.raw, self.bank.launch_tally()) });
        }
        let Some(mut ticket) = self.bank.cache_ticket(&key, false, None) else { return work(); };
        stream.device.run(|| {
            // SAFETY: the section issues only this stream's operations on admitted persistent storage.
            if let Err(error) = unsafe { decode_graph::begin_capture(library, stream.raw, &ticket) } {
                if self.bank.cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, ?key, "Inkling capture unavailable; running eagerly");
                }
                return work();
            }
            let result = work();
            // SAFETY: always end the capture we own, even if issuing the section failed.
            let source = unsafe { library.cuda_graph_capture_source(stream.raw) };
            if let Err(error) = result {
                self.bank.cache_ticket_failed(ticket, source.as_ref().err().unwrap_or(&error));
                return Err(error);
            }
            let captured = source.and_then(|source| {
                // Reject unknown or spanning topology before a donor can be re-keyed.
                if source.spans_devices || source.inspection_error.is_some() {
                    self.unsupported.insert(key);
                    anyhow::bail!("Inkling graph must have inspected rank-local topology");
                }
                ticket.complete_source(library, source, stream.raw)
            });
            let (raw, nodes, width, spans, _) = match captured {
                Ok(topology) => topology,
                Err(error) => {
                    let warn = self.bank.cache_ticket_failed(ticket, &error);
                    if warn && !cuteafd_ffi::is_small_graph(&error) {
                        tracing::warn!(%error, ?key, "Inkling graph unavailable; running eagerly");
                    }
                    return work();
                }
            };
            // SAFETY: the execution owns all captured storage, including FIR replay
            // tapes, and drains both ranks before clearing the bank or dropping it.
            let graph = unsafe { GraphOwner::new(library, self.device, raw, ())? };
            self.bank.census_capture_key("section", self.rank as i32, None, &key, raw);
            self.bank.cache_commit(key, graph, ticket, nodes as u64, width, spans);
            // SAFETY: the bank owns the executable through this ordered launch.
            unsafe { decode_graph::launch(library, raw, stream.raw, self.bank.launch_tally()) }
        })
    }
}

pub(super) fn local<'a>(graphs: Option<&mut DecodeGraphs<'a>>, stream: &Stream<'a>,
    layer: usize, rows: usize, section: Section, mode: Mode, work: impl Fn() -> Result<()>) -> Result<()> {
    let Some(graphs) = graphs else { return work(); };
    // The router does not touch the optional FIR replay backing.
    let mode = if section == Section::Routes { Mode::Decode } else { mode };
    graphs.run(Key { layer, rows, section, mode }, stream, work)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_identity_distinguishes_exact_rows_layers_sections_and_fir_replay() {
        let base = Key { layer: 4, rows: 7, section: Section::RankAttention, mode: Mode::Decode };
        assert_eq!(HashSet::from([
            base, Key { rows: 8, ..base }, Key { layer: 5, ..base },
            Key { section: Section::ExpertFinish, ..base }, Key { mode: Mode::Verify, ..base },
        ]).len(), 5);
    }
}
