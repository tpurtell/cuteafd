//! Monolithic 48-visit exact-row decode/verify graphs on the shared #2697 soft-cap device graph bank.
//!
//! All 48 loop visits (draft taps, attention, routing, local NVFP4/MXFP8 experts via
//! `DeviceRowCopy`, and residual/conv join) capture into one kernel-only graph per
//! `(rows, mode, taps)`, keeping metadata H2D uploads and head projection outside.
use crate::shared::{
    decode_graph::{
        self,
        device::{self, Basis, Cycle, Preference, Site},
        GraphBank, GraphOwner,
    },
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
struct Key {
    rows: usize,
    mode: Mode,
    taps: bool,
}

pub(super) struct DecodeGraphs<'a> {
    bank: GraphBank<Key, GraphOwner<'a, ()>>,
    first_touched: HashSet<Key>,
    first_touch_pending: bool,
    unsupported: HashSet<Key>,
    captures: usize,
    last_launched_exec: Option<usize>,
    device: i32,
}

impl<'a> DecodeGraphs<'a> {
    pub(super) fn new(stream: &Stream<'a>, policy: Resolved) -> Result<Option<Self>> {
        if policy.bytes == 0 {
            return Ok(None);
        }
        let mut bank = GraphBank::with_family("inkling2_mini");
        let device = stream.device.id;
        bank.attach_device(
            stream.device.library,
            device,
            policy.bytes,
            Site {
                name: "inkling2_mini.step",
                cycle: Cycle::Decode,
                preference: Preference::Adaptive,
                bootstrap: 2 << 20,
                basis: Basis::Bytes,
                spans_devices: super::graph_sites::spans("inkling2_mini.step", false),
            },
        )?;
        device::set_reserve_policy(device, policy)?;
        Ok(Some(Self {
            bank,
            first_touched: HashSet::new(),
            first_touch_pending: false,
            unsupported: HashSet::new(),
            captures: 0,
            last_launched_exec: None,
            device,
        }))
    }

    #[cfg(test)]
    pub(super) fn new_for_test(stream: &Stream<'a>, synthetic_cache_device: i32, policy: Resolved) -> Result<Self> {
        let mut bank = GraphBank::with_family("inkling2_mini");
        bank.attach_device_for_test(
            synthetic_cache_device,
            policy.bytes,
            Site {
                name: "inkling2_mini.step",
                cycle: Cycle::Decode,
                preference: Preference::Adaptive,
                bootstrap: 2 << 20,
                basis: Basis::Bytes,
                spans_devices: super::graph_sites::spans("inkling2_mini.step", false),
            },
        )?;
        Ok(Self {
            bank,
            first_touched: HashSet::new(),
            first_touch_pending: false,
            unsupported: HashSet::new(),
            captures: 0,
            last_launched_exec: None,
            device: stream.device.id,
        })
    }

    pub(super) fn capture_count(&self) -> usize {
        self.captures
    }
    pub(super) fn last_launched_exec(&self) -> Option<usize> {
        self.last_launched_exec
    }

    pub(super) fn cycle(&mut self, cycle: Cycle) {
        self.bank.cache_cycle(cycle);
    }
    pub(super) fn needs_boundary(&self) -> bool {
        self.first_touch_pending || self.bank.cache_needs_boundary()
    }
    pub(super) fn observed(&mut self) {
        self.first_touch_pending = false;
    }
    pub(super) fn tick(&mut self) {
        self.bank.cache_tick();
    }
    pub(super) fn visit(&mut self, visit: &mut device::GraphVisitor<'_>) -> Result<()> {
        visit(self.device, "inkling2_mini.step", false, &mut self.bank)
    }
    pub(super) fn clear_drained(&mut self) {
        self.bank.cache_clear_drained();
    }

    pub(super) fn prewarm_capture(
        &mut self,
        key: Key,
        stream: &Stream<'a>,
        mut work: impl FnMut() -> Result<()>,
    ) -> Result<bool> {
        if self.unsupported.contains(&key) {
            return Ok(false);
        }
        self.first_touched.insert(key);
        if self.bank.cached_launch(&key).is_some() {
            return Ok(false);
        }
        let library = stream.device.library;
        let Some(mut ticket) = self.bank.cache_ticket(&key, false, None) else {
            return Ok(false);
        };
        stream.device.run(|| {
            // SAFETY: rank-local stream operations on persistent admitted owners.
            if let Err(error) = unsafe { decode_graph::begin_capture(library, stream.raw, &ticket) } {
                if self.bank.cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, ?key, "Inkling 2 Mini prewarm capture unavailable");
                }
                return Ok(false);
            }
            let result = work();
            // SAFETY: always close capture we opened.
            let source = unsafe { library.cuda_graph_capture_source(stream.raw) };
            if let Err(error) = result {
                self.bank.cache_ticket_failed(ticket, source.as_ref().err().unwrap_or(&error));
                return Err(error);
            }
            let captured = source.and_then(|source| {
                if source.spans_devices || source.inspection_error.is_some() {
                    self.unsupported.insert(key);
                    anyhow::bail!("Inkling 2 Mini graph must have inspected rank-local topology");
                }
                ticket.complete_source(library, source, stream.raw)
            });
            let (raw, nodes, width, spans, _) = match captured {
                Ok(topology) => topology,
                Err(error) => {
                    let warn = self.bank.cache_ticket_failed(ticket, &error);
                    if warn && !cuteafd_ffi::is_small_graph(&error) {
                        tracing::warn!(%error, ?key, "Inkling 2 Mini prewarm graph unavailable");
                    }
                    return Ok(false);
                }
            };
            // SAFETY: execution owns all captured persistent storage and drains before drop.
            let graph = unsafe { GraphOwner::new(library, self.device, raw, ())? };
            self.bank.census_capture_key("step", 0, None, &key, raw);
            self.bank.cache_commit(key, graph, ticket, nodes as u64, width, spans);
            self.captures += 1;
            Ok(true)
        })
    }

    fn run(
        &mut self,
        key: Key,
        slots: usize,
        stream: &Stream<'a>,
        mut work: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let verify_rows = if key.mode == Mode::Verify { key.rows } else { 0 };
        if self.unsupported.contains(&key) {
            return work();
        }
        if self.first_touched.insert(key) {
            if decode_graph::census::enabled() {
                tracing::info!(
                    target: "cuteafd::graph_capture",
                    site = "inkling2_mini.step",
                    outcome = "first_touch",
                    rows = key.rows,
                    verify_rows,
                    slots,
                    ?key,
                    "graph census miss"
                );
            }
            self.first_touch_pending = true;
            return work();
        }
        self.last_launched_exec = None;
        let library = stream.device.library;
        if let Some(raw) = self.bank.cached_launch(&key).map(|owner| owner.raw) {
            self.last_launched_exec = Some(raw as usize);
            // SAFETY: retained TP1 stream, geometry, and persistent owners outlive replay.
            return stream
                .device
                .run(|| unsafe { decode_graph::launch(library, raw, stream.raw, self.bank.launch_tally()) });
        }
        let Some(mut ticket) = self.bank.cache_ticket(&key, false, None) else {
            if decode_graph::census::enabled() {
                tracing::info!(
                    target: "cuteafd::graph_capture",
                    site = "inkling2_mini.step",
                    outcome = "eager_miss",
                    rows = key.rows,
                    verify_rows,
                    slots,
                    ?key,
                    "graph census miss"
                );
            }
            return work();
        };
        if decode_graph::census::enabled() {
            tracing::info!(
                target: "cuteafd::graph_capture",
                site = "inkling2_mini.step",
                outcome = "capture_miss",
                rows = key.rows,
                verify_rows,
                slots,
                ?key,
                "graph census miss"
            );
        }
        stream.device.run(|| {
            // SAFETY: rank-local stream operations on persistent admitted owners.
            if let Err(error) = unsafe { decode_graph::begin_capture(library, stream.raw, &ticket) } {
                if self.bank.cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, ?key, "Inkling 2 Mini graph capture unavailable; running eagerly");
                }
                return work();
            }
            let result = work();
            // SAFETY: always close capture we opened.
            let source = unsafe { library.cuda_graph_capture_source(stream.raw) };
            if let Err(error) = result {
                self.bank.cache_ticket_failed(ticket, source.as_ref().err().unwrap_or(&error));
                return Err(error);
            }
            let captured = source.and_then(|source| {
                if source.spans_devices || source.inspection_error.is_some() {
                    self.unsupported.insert(key);
                    anyhow::bail!("Inkling 2 Mini graph must have inspected rank-local topology");
                }
                ticket.complete_source(library, source, stream.raw)
            });
            let (raw, nodes, width, spans, _) = match captured {
                Ok(topology) => topology,
                Err(error) => {
                    let warn = self.bank.cache_ticket_failed(ticket, &error);
                    if warn && !cuteafd_ffi::is_small_graph(&error) {
                        tracing::warn!(%error, ?key, "Inkling 2 Mini graph unavailable; running eagerly");
                    }
                    return work();
                }
            };
            // SAFETY: execution owns all captured persistent storage and drains before drop.
            let graph = unsafe { GraphOwner::new(library, self.device, raw, ())? };
            self.bank.census_capture_key("step", 0, None, &key, raw);
            self.bank.cache_commit(key, graph, ticket, nodes as u64, width, spans);
            self.captures += 1;
            self.last_launched_exec = Some(raw as usize);
            // SAFETY: bank owns executable through this ordered launch.
            unsafe { decode_graph::launch(library, raw, stream.raw, self.bank.launch_tally()) }
        })
    }
}

pub(super) fn step<'a>(
    graphs: Option<&mut DecodeGraphs<'a>>,
    stream: &Stream<'a>,
    rows: usize,
    slots: usize,
    mode: Mode,
    taps: bool,
    mut work: impl FnMut() -> Result<()>,
) -> Result<()> {
    let Some(graphs) = graphs else {
        return work();
    };
    graphs.run(Key { rows, mode, taps }, slots, stream, work)
}

/// Shipping startup pre-capture key list (`(rows, mode)`), covering the full
/// producible key space (`Decode 1..=max_sequences` + `Verify 1..=DECODE_ROWS`
/// when `has_draft` is true), matching `inkling2_mini_graphs::KeySpace`.
pub(super) fn shipping_prewarm_keys(max_sequences: usize, has_draft: bool) -> Vec<(usize, Mode)> {
    let space = cuteafd_loader::serving_capacity::inkling2_mini_graphs::KeySpace::for_config(
        max_sequences as u64,
        cuteafd_loader::families::inkling2_mini::execution::DECODE_ROWS,
        has_draft,
    );
    let mut keys = Vec::with_capacity(space.total_keys());
    for rows in 1..=space.decode_keys {
        keys.push((rows, Mode::Decode));
    }
    for rows in 1..=space.verify_keys {
        keys.push((rows, Mode::Verify));
    }
    keys
}

pub(super) fn prewarm_step<'a>(
    graphs: &mut DecodeGraphs<'a>,
    stream: &Stream<'a>,
    rows: usize,
    mode: Mode,
    taps: bool,
    work: impl FnMut() -> Result<()>,
) -> Result<bool> {
    graphs.prewarm_capture(Key { rows, mode, taps }, stream, work)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_monolithic_graph_key_distinguishes_rows_verify_and_taps() {
        let base = Key {
            rows: 4,
            mode: Mode::Decode,
            taps: false,
        };
        assert_eq!(
            HashSet::from([
                base,
                Key { rows: 8, ..base },
                Key { mode: Mode::Verify, ..base },
                Key { taps: true, ..base },
                Key {
                    mode: Mode::Verify,
                    taps: true,
                    ..base
                },
            ])
            .len(),
            5
        );
    }

    #[test]
    fn shipping_prewarm_covers_all_dspark_verify_rows_and_decode_bands() {
        use cuteafd_loader::{
            families::inkling2_mini::execution::DECODE_ROWS,
            serving_capacity::inkling2_mini_graphs::KeySpace,
        };
        for max_seq in [1, 16, 32, 48, 64] {
            for has_draft in [false, true] {
                let keys = shipping_prewarm_keys(max_seq, has_draft);
                let space = KeySpace::for_config(max_seq as u64, DECODE_ROWS, has_draft);
                assert_eq!(keys.len(), space.total_keys());
                for rows in 1..=max_seq {
                    assert!(keys.contains(&(rows, Mode::Decode)));
                }
                if has_draft {
                    for rows in 1..=DECODE_ROWS {
                        assert!(keys.contains(&(rows, Mode::Verify)));
                    }
                } else {
                    assert!(keys.iter().all(|&(_, mode)| mode == Mode::Decode));
                }
            }
        }
    }
}
