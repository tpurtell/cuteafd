//! Captures borrow only engine-owned weights, state and planes. Uploads and tap
//! scatters remain ordered eager work outside each complete layer segment.
use super::*;
use crate::shared::decode_graph::{GraphBank, GraphOwner, device::{self, Basis, Cycle, Preference, Site}};
use cuteafd_loader::serving_capacity::qwen35_graphs::{self, Geometry};
use std::{cell::{Cell, RefCell}, rc::Rc};

pub(crate) struct GraphBypass {
    ready: Rc<Cell<bool>>,
    previous: bool,
}

impl Drop for GraphBypass {
    fn drop(&mut self) { self.ready.set(self.previous); }
}

impl GraphBypass {
    fn new(ready: &Rc<Cell<bool>>) -> Self {
        Self { previous: ready.replace(false), ready: ready.clone() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Key {
    segment: usize,
    rows: usize,
    table_stride: usize,
    spec: bool,
}

type Executable<'a> = GraphOwner<'a, ()>;

pub(super) struct Graphs<'a> {
    bank: RefCell<GraphBank<Key, Executable<'a>>>,
    variant: String,
    pub(super) enabled: bool,
    capture_ready: Rc<Cell<bool>>,
    warming: std::cell::Cell<bool>,
}

impl<'a> Graphs<'a> {
    pub(super) fn new(device: Device<'a>, variant: String) -> Result<Self> {
        let enabled = qwen35_graphs::enabled();
        let reserve = qwen35_graphs::configured_reserve(&variant, enabled)?;
        let mut bank = GraphBank::new();
        bank.attach_device(device.library, device.id, reserve.bytes, Site {
            name: "qwen3_5.segment", cycle: Cycle::Other, preference: Preference::Adaptive,
            bootstrap: qwen35_graphs::BOOTSTRAP_BYTES, basis: Basis::Bytes, spans_devices: false,
        })?;
        device::set_reserve_policy(device.id, reserve)?;
        tracing::warn!(%variant, reserve = reserve.bytes,
            "Dense Qwen graph cache uses uncalibrated byte basis");
        if qwen35_graphs::layout(&variant).is_none() {
            tracing::warn!(%variant, "Dense Qwen layout has no trace: bootstrap is unpinned");
        }
        Ok(Self { bank: RefCell::new(bank), variant,
            enabled,
            capture_ready: Rc::new(Cell::new(false)), warming: std::cell::Cell::new(false) })
    }

    pub(super) fn cycle(&self, speculative: bool) {
        self.bank.borrow_mut().cache_cycle(if self.warming.get() { Cycle::Other }
            else if speculative { Cycle::Verify } else { Cycle::Decode });
    }

    pub(super) fn prefill(&self) { self.bank.borrow_mut().cache_cycle(Cycle::Prefill); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_bypass_restores_on_error_nested_calls_and_unwind() {
        let ready = Rc::new(Cell::new(true));
        let fail = || -> Result<()> {
            let _outer = GraphBypass::new(&ready);
            assert!(!ready.get());
            {
                let _inner = GraphBypass::new(&ready);
                assert!(!ready.get());
            }
            assert!(!ready.get());
            anyhow::bail!("diagnostic failure")
        };
        assert!(fail().is_err());
        assert!(ready.get());
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _bypass = GraphBypass::new(&ready);
            assert!(!ready.get());
            panic!("diagnostic unwind");
        })).is_err());
        assert!(ready.get());
        ready.set(false);
        drop(GraphBypass::new(&ready));
        assert!(!ready.get());
    }
}

impl DenseEngine<'_> {
    /// Only used across the synchronous single-lane diagnostic submission, with
    /// no await or callback that can start another serving round on this engine.
    pub(crate) fn bypass_graphs(&self) -> GraphBypass {
        GraphBypass::new(&self.graphs.capture_ready)
    }

    pub(super) fn graph_layer(&self, index: usize, step: &Step) -> Result<()> {
        self.enter_layer(index, step.rows)?;
        let rank = self.homes.layers[index].owner;
        let segment = || self.owner_stream(rank).device.run(|| self.layer_blocks(index)
            .run(index, &self.weights.layers[index], self.owner_planes(rank),
                self.kv[index].as_ref().map(|allocation| allocation.buffer), step));
        if !self.graphs.enabled || !self.graphs.capture_ready.get() { return segment(); }
        let stream = self.owner_stream(rank);
        let _device = stream.device.enter()?;
        let graphs = if rank == 0 { &self.graphs } else { &self.peer.as_ref().expect("admitted graph owner").graphs };
        let key = Key { segment: index, rows: step.rows, table_stride: step.table_stride, spec: step.speculative };
        let library = stream.device.library;
        if let Some(raw) = graphs.bank.borrow_mut().cached_launch(&key).map(|exec| exec.raw) {
            // SAFETY: all captured pointers retain the same engine-owned backing.
            return unsafe { crate::shared::decode_graph::launch(&library, raw, stream.raw) };
        }
        let pinned = qwen35_graphs::layout(&graphs.variant).is_some_and(|layout|
            layout.traced && layout.entries.iter().any(|entry|
                entry.pin && entry.rows == key.rows && entry.table_stride == key.table_stride && entry.spec == key.spec));
        let Some(ticket) = graphs.bank.borrow_mut().cache_ticket(&key, pinned, None) else {
            return segment();
        };
        // SAFETY: blocks only enqueue launches over persistent engine storage.
        if let Err(error) = unsafe { crate::shared::decode_graph::begin_capture(&library, stream.raw, &ticket) } {
            if graphs.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                tracing::warn!(%error, "Dense Qwen capture unavailable; eager fallback");
            }
            return segment();
        }
        let captured = segment();
        // SAFETY: ends capture on the same stream even when enqueue failed.
        let ended = unsafe { library.cuda_graph_end_capture_topology(stream.raw) };
        if let Err(error) = captured {
            graphs.bank.borrow_mut().cache_ticket_failed(ticket, ended.as_ref().err().unwrap_or(&error));
            if let Ok((raw, ..)) = ended {
                // SAFETY: this executable has never been launched.
                unsafe { let _ = library.cuda_graph_exec_destroy(raw); }
            }
            return Err(error);
        }
        let (raw, nodes, width, spans, inspection) = match ended {
            Ok(info) => info,
            Err(error) => {
                if graphs.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, "Dense Qwen instantiate unavailable; eager fallback");
                }
                return segment();
            }
        };
        // SAFETY: engine field order drains the stream, then drops graphs before backing.
        let exec = unsafe { GraphOwner::new(library, stream.device.id, raw, ())? };
        // Unknown topology or a singleton is not useful enough to retain.
        if inspection.is_some() || nodes < 2 {
            drop(exec);
            drop(ticket);
            return segment();
        }
        graphs.bank.borrow_mut().cache_commit(key, exec, ticket, nodes as u64, width, spans);
        // SAFETY: successful insertion retains the executable until a drained boundary.
        unsafe { crate::shared::decode_graph::launch(&library, raw, stream.raw) }
    }

    pub(super) fn graph_completed(&self) -> Result<()> {
        let mut banks = vec![(self.stream.device.id, &self.graphs)];
        banks.extend(self.peer.iter().map(|peer| (peer.stream.device.id, &peer.graphs)));
        if !banks.iter().any(|(_, graphs)| graphs.bank.borrow().cache_needs_boundary()) {
            for (_, graphs) in banks { graphs.bank.borrow_mut().cache_tick(); }
            return Ok(());
        }
        self.drain()?;
        // SAFETY: both streams are drained; all banks retain rank-local resources.
        unsafe { device::observe_completed(self.stream.device.library, |visit| {
            for (device, graphs) in &banks {
                visit(*device, "qwen3_5.segment", false, &mut *graphs.bank.borrow_mut())?;
            }
            Ok(())
        }) }
    }

    /// Qualification only: compare identical decode shapes without changing the
    /// eager evaluate path or rebuilding the engine's captured backing.
    pub(crate) fn golden_decode_graphs(&mut self, tokens: &[u32], positions: &[usize], chunk: usize)
        -> Result<Option<(f64, f64)>> {
        use crate::shared::speculation::target::{VerifyRequest, VerifyRow};
        use sha2::{Digest, Sha256};
        if !self.graphs.enabled { return Ok(None); }
        ensure!(chunk > 0, "dense graph golden requires a nonzero chunk");
        let ready = self.graphs.capture_ready.get();
        let id = self.stream.device.id;
        let replay_hits = || device::snapshots().iter().filter(|row| row.device == id)
            .flat_map(|row| row.cycles.values()).map(|cycle| cycle.hits).sum::<u64>();
        let mut score = |capturing: bool| -> Result<(f64, Vec<u8>)> {
            self.reset_slot(0)?;
            self.graphs.capture_ready.set(capturing);
            let mut nll = 0.0;
            let mut scored = 0;
            let mut digest = Sha256::new();
            let vocab = self.cfg.vocab;
            let consumer = crate::shared::probe::CallbackRows::new(|position, values: &[f32]| {
                for value in values { digest.update(value.to_le_bytes()); }
                if positions.binary_search(&(position + 1)).is_ok() {
                    nll += crate::shared::golden_scores::scored_nll(values, &[tokens[position + 1]], vocab)?;
                    scored += 1;
                }
                Ok(())
            });
            let mut first = 0;
            for chunk in tokens.chunks(chunk.min(64)) {
                let rows = chunk.iter().enumerate().map(|(offset, &token)| VerifyRow {
                    slot: 0, position: first + offset, token: Some(token) }).collect::<Vec<_>>();
                let logits = self.decode_device(&[0], VerifyRequest { rows: &rows, tap_layers: &[] })?;
                crate::shared::probe::ScoringRows { consumer: &consumer, position: first }
                    .record(self.stream.device.library, 0, &logits)?;
                first += chunk.len();
            }
            drop(consumer);
            ensure!(scored == positions.len() && scored > 0, "dense graph scoring rows incomplete");
            Ok((nll / scored as f64, digest.finalize().to_vec()))
        };
        let result: Result<(f64, f64, u64)> = (|| {
            let (off, off_digest) = score(false)?;
            // A short window may touch each stride only once. Warm every shape
            // before measuring so capture-only execution cannot pass this oracle.
            let (_, warm_digest) = score(true)?;
            ensure!(off_digest == warm_digest, "dense graph warm/off decode logits are not byte-identical");
            let before = replay_hits();
            let (on, on_digest) = score(true)?;
            let after = replay_hits();
            ensure!(off_digest == on_digest, "dense graph-on/off decode logits are not byte-identical");
            ensure!(after > before, "dense graph golden measured pass did not replay any executable");
            Ok((off, on, after - before))
        })();
        self.graphs.capture_ready.set(ready);
        let (off, on, replays) = result?;
        println!("dense graph decode: byte-identical off/warm-on/measured-on logits over {} rows; {replays} measured replay hits",
            tokens.len());
        self.reset_slot(0)?;
        Ok(Some((off, on)))
    }

    pub(crate) fn warm_decode_graphs(&mut self) -> Result<()> {
        if !self.graphs.enabled { return Ok(()); }
        let entries = qwen35_graphs::layout(&self.graphs.variant).map_or_else(Vec::new, |layout| layout.entries.clone());
        self.graphs.warming.set(true);
        let result = (|| {
            let request = [crate::shared::speculation::target::VerifyRow { slot: 0, position: 0, token: Some(0) }];
            let plan = rows::Rows::new(&crate::shared::speculation::target::VerifyRequest {
                rows: &request, tap_layers: &[] }, self.cfg.attention.len(), self.cfg.vocab,
                &[(0, self.slots[0].clone())], self.cfg.max_context, self.pool_tokens)?;
            self.verify_rows(plan, 1, &[], true)?;
            self.verification.pending = None;
            self.reset_slot(0)?;
            for id in self.devices() { device::observe_device(self.stream.device.library, id)?; }
            self.graphs.capture_ready.set(true);
            if let Some(peer) = &self.peer { peer.graphs.capture_ready.set(true); }
            for Geometry { rows, table_stride, spec, .. } in entries {
                if table_stride * 64 > self.cfg.max_context || rows > 64 { continue; }
                let position = (table_stride - 1) * 64;
                if position + rows > self.cfg.max_context || position + rows > self.pool_tokens { continue; }
                // Prewarm through real device lengths; restore all recurrence
                // afterwards so startup leaves exactly the fresh-engine state.
                self.slots[0].position = position;
                let request = (0..rows).map(|offset| crate::shared::speculation::target::VerifyRow {
                    slot: 0, position: position + offset, token: Some(0) }).collect::<Vec<_>>();
                let plan = rows::Rows::new(&crate::shared::speculation::target::VerifyRequest {
                    rows: &request, tap_layers: &[] }, self.cfg.attention.len(), self.cfg.vocab,
                    &[(0, self.slots[0].clone())], self.cfg.max_context, self.pool_tokens)?;
                self.verify_rows(plan, rows, &[], spec)?;
                self.verification.pending = None;
                self.reset_slot(0)?;
            }
            Ok(())
        })();
        self.graphs.warming.set(false);
        result
    }
}
