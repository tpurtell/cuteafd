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
        let segment = || Blocks { programs: self.programs, admitted: &self.admitted,
            cfg: &self.cfg, homes: &self.homes, bank: &self.bank, stream: self.stream.raw }
            .run(index, &self.weights.layers[index], self.planes,
                self.kv[index].as_ref().map(|allocation| allocation.buffer), step);
        if !self.graphs.enabled || !self.graphs.capture_ready.get() { return segment(); }
        let key = Key { segment: index, rows: step.rows, table_stride: step.table_stride, spec: step.speculative };
        let library = self.stream.device.library;
        if let Some(raw) = self.graphs.bank.borrow_mut().cached_launch(&key).map(|exec| exec.raw) {
            // SAFETY: all captured pointers retain the same engine-owned backing.
            return unsafe { crate::shared::decode_graph::launch(&library, raw, self.stream.raw) };
        }
        let pinned = qwen35_graphs::layout(&self.graphs.variant).is_some_and(|layout|
            layout.traced && layout.entries.iter().any(|entry|
                entry.pin && entry.rows == key.rows && entry.table_stride == key.table_stride && entry.spec == key.spec));
        let Some(ticket) = self.graphs.bank.borrow_mut().cache_ticket(&key, pinned, None) else {
            return segment();
        };
        // SAFETY: blocks only enqueue launches over persistent engine storage.
        if let Err(error) = unsafe { crate::shared::decode_graph::begin_capture(&library, self.stream.raw, &ticket) } {
            if self.graphs.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                tracing::warn!(%error, "Dense Qwen capture unavailable; eager fallback");
            }
            return segment();
        }
        let captured = segment();
        // SAFETY: ends capture on the same stream even when enqueue failed.
        let ended = unsafe { library.cuda_graph_end_capture_topology(self.stream.raw) };
        if let Err(error) = captured {
            self.graphs.bank.borrow_mut().cache_ticket_failed(ticket, ended.as_ref().err().unwrap_or(&error));
            if let Ok((raw, ..)) = ended {
                // SAFETY: this executable has never been launched.
                unsafe { let _ = library.cuda_graph_exec_destroy(raw); }
            }
            return Err(error);
        }
        let (raw, nodes, width, spans, inspection) = match ended {
            Ok(info) => info,
            Err(error) => {
                if self.graphs.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, "Dense Qwen instantiate unavailable; eager fallback");
                }
                return segment();
            }
        };
        // SAFETY: engine field order drains the stream, then drops graphs before backing.
        let exec = unsafe { GraphOwner::new(library, self.stream.device.id, raw, ())? };
        // Unknown topology or a singleton is not useful enough to retain.
        if inspection.is_some() || nodes < 2 {
            drop(exec);
            drop(ticket);
            return segment();
        }
        self.graphs.bank.borrow_mut().cache_commit(key, exec, ticket, nodes as u64, width, spans);
        // SAFETY: successful insertion retains the executable until a drained boundary.
        unsafe { crate::shared::decode_graph::launch(&library, raw, self.stream.raw) }
    }

    pub(super) fn graph_completed(&self) -> Result<()> {
        if !self.graphs.bank.borrow().cache_needs_boundary() {
            self.graphs.bank.borrow_mut().cache_tick();
            return Ok(());
        }
        self.stream.drain()?;
        // SAFETY: the single target stream is drained; the bank owns no captured
        // backing, and persistent weights/state/planes outlive every executable.
        unsafe { device::observe_completed(self.stream.device.library, |visit|
            visit(self.stream.device.id, "qwen3_5.segment", false, &mut *self.graphs.bank.borrow_mut())) }
    }

    /// Qualification only: compare identical decode shapes without changing the
    /// eager evaluate path or rebuilding the engine's captured backing.
    pub(crate) fn golden_decode_graphs(&mut self, tokens: &[u32], positions: &[usize], chunk: usize)
        -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        use crate::shared::speculation::target::{VerifyRequest, VerifyRow};
        use sha2::{Digest, Sha256};
        if !self.graphs.enabled { return Ok(None); }
        ensure!(chunk > 0, "dense graph golden requires a nonzero chunk");
        let ready = self.graphs.capture_ready.get();
        let id = self.stream.device.id;
        let replay_hits = || device::snapshots().iter().filter(|row| row.device == id)
            .flat_map(|row| row.cycles.values()).map(|cycle| cycle.hits).sum::<u64>();
        let mut score = |capturing: bool| -> Result<(Vec<f32>, Vec<u8>)> {
            self.reset_slot(0)?;
            self.graphs.capture_ready.set(capturing);
            let mut selected = Vec::with_capacity(positions.len() * self.cfg.vocab);
            let mut digest = Sha256::new();
            let mut first = 0;
            for tokens in tokens.chunks(chunk.min(64)) {
                let rows = tokens.iter().enumerate().map(|(offset, &token)| VerifyRow {
                    slot: 0, position: first + offset, token: Some(token) }).collect::<Vec<_>>();
                let logits = self.decode_device(&[0], VerifyRequest { rows: &rows, tap_layers: &[] })?
                    .to_host(self.stream.device.library)?;
                for value in &logits { digest.update(value.to_le_bytes()); }
                for &position in positions.iter().filter(|&&p| p > first && p <= first + tokens.len()) {
                    let row = position - first - 1;
                    selected.extend_from_slice(&logits[row * self.cfg.vocab..(row + 1) * self.cfg.vocab]);
                }
                first += tokens.len();
            }
            Ok((selected, digest.finalize().to_vec()))
        };
        let result: Result<(Vec<f32>, Vec<f32>, u64)> = (|| {
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
            device::observe_device(self.stream.device.library, self.stream.device.id)?;
            self.graphs.capture_ready.set(true);
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
