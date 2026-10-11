//! Per-round draft policy signals from the GLM 5.3 Flash engine (v3 D2): each
//! verify step's per-layer device time and routed expert ids, collected where
//! the engine already has them and read once after the step's own sync.
//!
//! - Layer time: a [`LayerClock`] event at every `console::layer_mark` site,
//!   between the per-layer graph segments (or eager layers).
//! - Spark layers: the ids the step already stages to pinned host memory for
//!   the expert request, copied before the staging is reused.
//! - Local layers (FP8 or EXL3 on this GPU): one async D2H of the layer's ids
//!   into a pinned ring right after its router, decoded after the sync; no
//!   per-layer host wait.
//! - Dense layers carry no routes.
//!
//! The probe is armed only by the shared draft policy and is active only
//! between [`RoundProbe::begin`] and [`RoundProbe::finish`] around a serving
//! verify step: prefill, goldens and fixed/chain draft policies never touch it.
//! `CUTEAFD_GLMF_ROUTE_RING_CHECK=1` (diagnostic, off by default) also rings
//! the Spark layers and compares the ring with the staged ids, which checks
//! the ring on hardware wherever Spark experts serve.
use crate::shared::draft::clock::LayerClock;
use crate::shared::draft::routes::{RoundRoutes, RouteError};
use crate::shared::memory::HostAllocation;
use anyhow::Result;
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use std::ffi::c_void;

/// GLM Flash router ids are plain expert indices (no flag bits).
const ROUTE_MASK: u32 = u32::MAX;

/// Ring and staged-id agreement under `CUTEAFD_GLMF_ROUTE_RING_CHECK`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RingCheck {
    pub layers: u64,
    pub mismatched: u64,
}

pub(crate) struct RoundProbe<'a> {
    clock: LayerClock<'a>,
    routes: RoundRoutes,
    /// Pinned `[layer][row][slot]` u32 ids, `capacity` rows per layer.
    ring: HostAllocation<'a>,
    capacity: usize,
    topk: usize,
    dense: Vec<bool>,
    /// Layers whose ids were queued into the ring this round.
    ringed: Vec<bool>,
    /// Layers whose ids came from the Spark staging this round.
    staged: Vec<bool>,
    check: bool,
    pub ring_check: RingCheck,
    /// Recording this step's marks, staged ids and ring copies.
    active: bool,
    /// A step ended whose signals `finish` has not read yet.
    pending: bool,
    /// The first route error of the round; the round is then not observed.
    error: Option<RouteError>,
    scratch: Vec<u32>,
}

impl<'a> RoundProbe<'a> {
    /// A probe for `dense.len()` layers of `topk` routes and steps of up to
    /// `capacity` rows, on the current device.
    pub fn new(library: &'a NativeLibrary, dense: Vec<bool>, topk: usize, capacity: usize) -> Result<Self> {
        let layers = dense.len();
        let ring = HostAllocation::new(library, layers * capacity * topk * 4)?;
        Ok(Self { clock: LayerClock::new(library, layers)?, routes: RoundRoutes::new(layers, topk), ring, capacity,
            topk, ringed: vec![false; layers], staged: vec![false; layers], dense,
            check: std::env::var("CUTEAFD_GLMF_ROUTE_RING_CHECK").is_ok_and(|v| v == "1"),
            ring_check: RingCheck::default(), active: false, pending: false, error: None, scratch: Vec::new() })
    }

    /// Start a verify step of `rows` rows.
    pub fn begin(&mut self, rows: usize) {
        self.routes.begin(rows);
        for (layer, &dense) in self.dense.iter().enumerate() {
            if dense { let _ = self.routes.dense(layer); }
        }
        self.clock.reset();
        self.ringed.fill(false);
        self.staged.fill(false);
        self.error = (rows > self.capacity).then_some(RouteError::Extent { layer: 0, found: rows, expected: self.capacity });
        self.active = true;
        self.pending = true;
    }

    /// Stop recording: the step's work is queued (its signals are read by
    /// `finish` once its stream drained). Later engine work leaves them alone.
    pub fn end(&mut self) { self.active = false; }

    pub fn active(&self) -> bool { self.active }

    /// Whether a layer's ids go through the ring: every local layer, and
    /// Spark layers too under the ring check.
    pub fn wants_ring(&self, spark: bool) -> bool { self.active && (!spark || self.check) }

    /// Mark the end of `layer` on `stream`.
    ///
    /// # Safety
    /// `stream` is the engine's live stream on this probe's device.
    pub unsafe fn mark(&mut self, layer: usize, stream: *mut c_void) {
        if self.active {
            // SAFETY: forwarded from the caller; no read of this round's marks is in flight.
            unsafe { self.clock.mark(layer, stream, false) };
        }
    }

    /// A Spark layer's staged ids: `rows * topk` little-endian u32 in row order.
    pub fn staged(&mut self, layer: usize, bytes: &[u8]) {
        if !self.active || self.error.is_some() { return; }
        self.scratch.clear();
        self.scratch.extend(bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])));
        match self.routes.push(layer, &self.scratch, ROUTE_MASK) {
            Ok(()) => self.staged[layer] = true,
            Err(error) => self.error = Some(error),
        }
    }

    /// Queue a D2H of `layer`'s `rows * topk` router ids (`ids`, u32) into
    /// the ring on `stream`, behind the router that wrote them.
    ///
    /// # Safety
    /// `ids` holds this step's ids of `layer` in stream order on `stream`, the
    /// engine's live stream; the ring is read only after the step's sync.
    pub unsafe fn ring(&mut self, library: &NativeLibrary, layer: usize, ids: CuteafdDeviceBuffer, rows: usize,
        stream: *mut c_void) -> Result<()> {
        if !self.active || layer >= self.ringed.len() || rows > self.capacity { return Ok(()); }
        let bytes = rows * self.topk * 4;
        let offset = layer * self.capacity * self.topk * 4;
        let host = self.ring.buffer;
        let slot = CuteafdHostBuffer {
            // SAFETY: `layer < layers` and `rows <= capacity` keep the slot inside the ring.
            ptr: unsafe { host.ptr.cast::<u8>().add(offset) }.cast(),
            bytes: host.bytes - offset,
            ..host
        };
        // SAFETY: the caller guarantees the source and stream; the slot is pinned and sized above.
        unsafe { library.copy_d2h_host_buffer_async(slot, ids, bytes, stream)? };
        self.ringed[layer] = true;
        Ok(())
    }

    /// End the step once its stream drained: decode the ring into the round's
    /// routes (checking it against the staging where both exist) and read the
    /// layer clock. `None` when the round's routes are incomplete or invalid.
    pub fn finish(&mut self) -> Option<(&RoundRoutes, Vec<Option<f64>>)> {
        self.active = false;
        if !std::mem::take(&mut self.pending) { return None; }
        if self.error.is_none() {
            if let Err(error) = settle(&mut self.routes, self.ring.bytes(), self.capacity, self.topk, &self.ringed, &self.staged,
                &mut self.ring_check) {
                self.error = Some(error);
            }
        }
        if let Some(error) = self.error.take() {
            tracing::debug!(%error, "draft policy round routes not observed");
            return None;
        }
        self.routes.complete().ok()?;
        Some((&self.routes, self.clock.read()))
    }
}

/// Decode the ringed layers of a round (`ring`: layer-major slots of
/// `capacity` rows of `routes`' top-k u32 ids) into `routes`. A layer that was
/// also staged keeps its staged ids and is compared with the ring instead.
fn settle(routes: &mut RoundRoutes, ring: &[u8], capacity: usize, topk: usize, ringed: &[bool], staged: &[bool],
    check: &mut RingCheck) -> Result<(), RouteError> {
    let words = routes.rows() * topk;
    for layer in (0..ringed.len()).filter(|&layer| ringed[layer]) {
        let offset = layer * capacity * topk * 4;
        let ids: Vec<u32> = ring[offset..offset + words * 4].chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        if staged[layer] {
            check.layers += 1;
            let start = layer * words;
            if routes.flat()[start..start + words].iter().zip(&ids).any(|(&s, &r)| u32::from(s) != r) {
                check.mismatched += 1;
                tracing::warn!(layer, rows = routes.rows(), "GLM Flash route ring differs from the staged Spark ids");
            }
        } else {
            routes.push(layer, &ids, ROUTE_MASK)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(layers: usize, capacity: usize, topk: usize, fill: &[(usize, Vec<u32>)]) -> Vec<u8> {
        let mut ring = vec![0u8; layers * capacity * topk * 4];
        for (layer, ids) in fill {
            let offset = layer * capacity * topk * 4;
            for (i, id) in ids.iter().enumerate() { ring[offset + i * 4..][..4].copy_from_slice(&id.to_le_bytes()); }
        }
        ring
    }

    /// Local layers decode from their ring slots (layer-major, a step's rows
    /// at the front), dense layers stay zero, staged Spark layers keep their
    /// staged ids, and the round completes.
    #[test]
    fn ring_settles_local_layers_beside_staged_and_dense_ones() {
        let (layers, capacity, topk, rows) = (4usize, 8usize, 3usize, 5usize);
        let local: Vec<u32> = (0..15).map(|i| 100 + i).collect();
        let spark: Vec<u32> = (0..15).map(|i| 200 + i).collect();
        let bytes = ring(layers, capacity, topk, &[(1, local.clone()), (3, vec![7; 15])]);
        let mut routes = RoundRoutes::new(layers, topk);
        routes.begin(rows);
        routes.dense(0).unwrap();
        routes.push(2, &spark, ROUTE_MASK).unwrap();
        // Layer 3 is staged and ringed (the ring check): the staging stands.
        routes.push(3, &spark, ROUTE_MASK).unwrap();
        let mut check = RingCheck::default();
        settle(&mut routes, &bytes, capacity, topk, &[false, true, false, true], &[false, false, true, true], &mut check)
            .unwrap();
        routes.complete().unwrap();
        let flat = routes.flat();
        let at = |layer: usize| &flat[layer * rows * topk..][..rows * topk];
        assert!(at(0).iter().all(|&id| id == 0));
        assert_eq!(at(1), local.iter().map(|&i| i as u16).collect::<Vec<_>>());
        assert_eq!(at(3), spark.iter().map(|&i| i as u16).collect::<Vec<_>>());
        assert_eq!(check, RingCheck { layers: 1, mismatched: 1 });
        // A ring that agrees with the staging counts as checked and matched.
        let bytes = ring(layers, capacity, topk, &[(1, local.clone()), (3, spark.clone())]);
        let mut check = RingCheck::default();
        let mut again = RoundRoutes::new(layers, topk);
        again.begin(rows);
        again.dense(0).unwrap();
        again.push(2, &spark, ROUTE_MASK).unwrap();
        again.push(3, &spark, ROUTE_MASK).unwrap();
        settle(&mut again, &bytes, capacity, topk, &[false, true, false, true], &[false, false, true, true], &mut check)
            .unwrap();
        assert_eq!((check, again.flat()), (RingCheck { layers: 1, mismatched: 0 }, flat));
    }

    /// A ring id past u16 is a route error, never a silent truncation.
    #[test]
    fn ring_rejects_ids_past_u16() {
        let mut routes = RoundRoutes::new(1, 2);
        routes.begin(1);
        let bytes = ring(1, 4, 2, &[(0, vec![70_000, 1])]);
        assert_eq!(settle(&mut routes, &bytes, 4, 2, &[true], &[false], &mut RingCheck::default()),
            Err(RouteError::Id(70_000)));
    }
}
