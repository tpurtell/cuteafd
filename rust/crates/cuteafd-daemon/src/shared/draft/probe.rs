//! Per-round draft policy signals from the GLM 5.3 Flash engine (v3 D2): each
//! verify step's per-layer device time and routed expert ids, collected where
//! the engine already has them and read once after the step's own sync.
//!
//! - Layer time: owner-local [`LayerClock`] events every observed round,
//!   between the per-layer graph segments (or eager layers).
//! - Spark layers: the ids the step already stages to pinned host memory for
//!   the expert request, copied before the staging is reused.
//! - Local layers: D2D each router's ids into a device layer-major arena, then
//!   one D2H at the last owned routed layer, before the step's existing sync.
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
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::Result;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
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
    collect_routes: bool,
    /// Per-layer CUDA-event timing (the priced policy or the replay trace).
    /// Expert usage captures routes without it.
    timing: bool,
    clock: LayerClock<'a>,
    routes: RoundRoutes,
    /// Pinned `[layer][row][slot]` u32 ids, packed to this round's row stride.
    ring: HostAllocation<'a>,
    device_ring: DeviceAllocation<'a>,
    last_owned: Option<usize>,
    downloaded: bool,
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
}

impl<'a> RoundProbe<'a> {
    /// A probe for `dense.len()` layers of `topk` routes and steps of up to
    /// `capacity` rows, on the current device.
    pub fn new(library: &'a NativeLibrary, dense: Vec<bool>, topk: usize, capacity: usize) -> Result<Self> {
        let layers = dense.len();
        let bytes = layers * capacity * topk * 4;
        let ring = HostAllocation::new(library, bytes)?;
        let device_ring = DeviceAllocation::new(library, bytes)?;
        // Initialize padding once: the single contiguous download includes unused row slots.
        library.copy_h2d(device_ring.buffer, ring.bytes())?;
        let last_owned = dense.iter().rposition(|&dense| !dense);
        Ok(Self { collect_routes: super::binding::configured_needs_routes(),
            timing: super::binding::configured_needs_timing(),
            clock: LayerClock::new(library, layers)?, routes: RoundRoutes::new(layers, topk), ring,
            device_ring, last_owned, downloaded: false, capacity,
            topk, ringed: vec![false; layers], staged: vec![false; layers], dense,
            check: std::env::var("CUTEAFD_GLMF_ROUTE_RING_CHECK").is_ok_and(|v| v == "1"),
            ring_check: RingCheck::default(), active: false, pending: false, error: None })
    }

    /// Start a verify step of `rows` rows.
    pub fn begin(&mut self, rows: usize) {
        if !self.collect_routes { self.pending = true; return; }
        self.routes.begin(rows);
        for (layer, &dense) in self.dense.iter().enumerate() {
            if dense { let _ = self.routes.dense(layer); }
        }
        self.clock.reset();
        self.downloaded = false;
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
    pub fn collects_routes(&self) -> bool { self.collect_routes }

    /// Whether a layer's ids go through the ring: every local layer, and
    /// Spark layers too under the ring check.
    pub fn wants_ring(&self, spark: bool) -> bool { self.active && (!spark || self.check) }

    /// Mark the end of `layer` on `stream`.
    ///
    /// # Safety
    /// `stream` is the engine's live stream on this probe's device.
    pub unsafe fn mark(&mut self, layer: usize, stream: *mut c_void) -> Result<()> {
        if self.active {
            // SAFETY: forwarded from the caller; no read of this round's marks is in flight.
            if self.timing { unsafe { self.clock.mark(layer, stream, false) }; }
            if Some(layer) == self.last_owned && self.ringed.iter().any(|&queued| queued) {
                // SAFETY: all owner-local D2D writes precede this download on the same stream;
                // the pinned ring remains owned until the existing verifier sync and finish.
                unsafe { self.device_ring.library.copy_d2h_host_buffer_async(
                    self.ring.buffer, self.device_ring.buffer,
                    self.ringed.len() * self.routes.rows() * self.topk * 4, stream)?; }
                self.downloaded = true;
            }
        }
        Ok(())
    }

    /// Mark the first layer after an owner cut using this device's own event.
    ///
    /// # Safety
    /// `stream` is live on this probe's device and ordered after the input transfer.
    pub unsafe fn mark_entry(&mut self, layer: usize, stream: *mut c_void) {
        if self.active && self.timing {
            // SAFETY: the caller guarantees the event and stream ownership/order.
            unsafe { self.clock.mark_entry(layer, stream, false); }
        }
    }

    /// A Spark layer's staged ids: `rows * topk` little-endian u32 in row order.
    pub fn staged(&mut self, layer: usize, bytes: &[u8]) {
        if !self.active || self.error.is_some() { return; }
        match self.routes.push_bytes(layer, bytes, ROUTE_MASK) {
            Ok(()) => self.staged[layer] = true,
            Err(error) => self.error = Some(error),
        }
    }

    /// Preserve `layer`'s `rows * topk` router ids with a D2D on `stream`,
    /// behind the router and before its reusable output is overwritten.
    ///
    /// # Safety
    /// `ids` holds this step's ids of `layer` in stream order on `stream`, the
    /// engine's live stream; the ring is read only after the step's sync.
    pub unsafe fn ring(&mut self, library: &NativeLibrary, layer: usize, ids: CuteafdDeviceBuffer, rows: usize,
        stream: *mut c_void) -> Result<()> {
        if !self.active || layer >= self.ringed.len() || rows > self.capacity { return Ok(()); }
        let bytes = rows * self.topk * 4;
        let offset = layer * rows * self.topk * 4;
        let arena = self.device_ring.buffer;
        let slot = CuteafdDeviceBuffer {
            // SAFETY: `layer < layers` and `rows <= capacity` keep the slot inside the arena.
            ptr: unsafe { arena.ptr.cast::<u8>().add(offset) }.cast(),
            bytes: arena.bytes - offset,
            ..arena
        };
        // SAFETY: the source is live on this stream; the arena slot is owned and sized above.
        unsafe { library.copy_d2d_async(slot, ids, bytes, stream)? };
        self.ringed[layer] = true;
        Ok(())
    }

    /// End the step once its stream drained: decode the ring into the round's
    /// routes (checking it against the staging where both exist) and read the
    /// layer clock. `None` when the round's routes are incomplete or invalid.
    pub fn finish(&mut self) -> Option<(&RoundRoutes, Vec<Option<f64>>)> {
        let _timing = super::clock::HostSpan::new(super::clock::HostStage::Routes);
        self.active = false;
        if !std::mem::take(&mut self.pending) { return None; }
        if !self.collect_routes { return Some((&self.routes, Vec::new())); }
        if self.ringed.iter().any(|&queued| queued) && !self.downloaded {
            tracing::debug!("draft policy route arena was not downloaded before verifier sync");
            return None;
        }
        if self.error.is_none() {
            let rows = self.routes.rows();
            if let Err(error) = complete_routes(&mut self.routes, self.ring.bytes(), rows, self.topk, &self.ringed,
                &self.staged, &mut self.ring_check) {
                self.error = Some(error);
            }
        }
        if let Some(error) = self.error.take() {
            tracing::debug!(%error, "draft policy round routes not observed");
            return None;
        }
        let times = if self.timing { self.clock.read() } else { Vec::new() };
        Some((&self.routes, times))
    }
}

/// A policy observation is valid only after both local rings and remote staging settle.
fn complete_routes(routes: &mut RoundRoutes, ring: &[u8], rows: usize, topk: usize,
    ringed: &[bool], staged: &[bool], check: &mut RingCheck) -> Result<(), RouteError> {
    settle(routes, ring, rows, topk, ringed, staged, check)?;
    routes.complete()
}

/// Decode the ringed layers of a round (`ring`: layer-major slots of
/// `capacity` rows of `routes`' top-k u32 ids) into `routes`. A layer that was
/// also staged keeps its staged ids and is compared with the ring instead.
fn settle(routes: &mut RoundRoutes, ring: &[u8], capacity: usize, topk: usize, ringed: &[bool], staged: &[bool],
    check: &mut RingCheck) -> Result<(), RouteError> {
    let words = routes.rows() * topk;
    for layer in (0..ringed.len()).filter(|&layer| ringed[layer]) {
        let offset = layer * capacity * topk * 4;
        let bytes = &ring[offset..offset + words * 4];
        if staged[layer] {
            check.layers += 1;
            let start = layer * words;
            if routes.flat()[start..start + words].iter().zip(bytes.chunks_exact(4))
                .any(|(&s, w)| u32::from(s) != u32::from_le_bytes([w[0], w[1], w[2], w[3]])) {
                check.mismatched += 1;
                tracing::warn!(layer, rows = routes.rows(), "GLM Flash route ring differs from the staged Spark ids");
            }
        } else {
            routes.push_bytes(layer, bytes, ROUTE_MASK)?;
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

    #[test]
    fn glm_mixed_tp2_and_spark_round_yields_policy_observation() {
        use crate::shared::draft::{clock::RoundTimes, evidence::DraftSource, resource};
        let (layers, rows, topk) = (78, 6, 8);
        let local = 3..7;
        let homes: Vec<_> = (0..layers).map(|layer| if layer < 3 { resource::LayerHome::Dense }
            else if local.contains(&layer) { resource::LayerHome::Local { slice_bytes: 9_000_000. } }
            else { resource::LayerHome::Remote { slice_bytes: 3_000_000. } }).collect();
        let geometry = resource::geometry(&homes, 256, topk, 64, 7, 0).unwrap();
        assert!(geometry.layers[local.clone()].iter().all(|layer| layer.class == Some(0)));
        assert!(geometry.layers[7..].iter().all(|layer| layer.class == Some(1)));
        let mut policy = cuteafd_core::DraftPolicy::new(geometry, cuteafd_core::PolicySeed::weak(7), false).unwrap();
        let ids: Vec<u32> = (0..rows * topk).map(|i| i as u32 % 256).collect();
        let mut routes = RoundRoutes::new(layers, topk);
        routes.begin(rows);
        let mut staged = vec![false; layers];
        for layer in 0..layers {
            if layer < 3 { routes.dense(layer).unwrap(); }
            else if !local.contains(&layer) { routes.push(layer, &ids, ROUTE_MASK).unwrap(); staged[layer] = true; }
        }
        let mut ringed = vec![false; layers];
        let bytes = ring(layers, rows, topk, &local.clone().map(|layer| (layer, ids.clone())).collect::<Vec<_>>());
        assert_eq!(complete_routes(&mut routes, &bytes, rows, topk, &ringed, &staged,
            &mut RingCheck::default()), Err(RouteError::Missing(3)));
        ringed[local].fill(true);
        let observation = complete_routes(&mut routes, &bytes, rows, topk, &ringed, &staged,
            &mut RingCheck::default()).ok().map(|()| {
                let prior = [0.8; 5];
                let request = [resource::Verified { id: 1, rows, accepted: 4, source: DraftSource::Neural,
                    prior: Some(&prior), features: None, censor: None }];
                let times = (0..layers).map(|layer| (layer > 0).then_some(800.)).collect::<Vec<_>>();
                resource::observe_drafted(&mut policy, &request, &routes, &times,
                    RoundTimes { total_us: 70_000, draft_us: Some(3_000) }, 7, None, Some(1))
            });
        observation.expect("mixed GLM routes must produce an observation").unwrap();
        assert_eq!(policy.stats().rounds, 1);
    }

    #[test]
    fn v4_mixed_tp2_ring_and_host_or_device_spark_routes_update_policy() {
        use crate::shared::draft::{binding::{self, DraftPass, RoundRecord, RoundRequest},
            clock::RoundTimes, evidence::{DraftSource, Evidence}};
        use cuteafd_core::{DraftPolicy, LayerResource, PolicyGeometry, PolicySeed, ResourceClass};
        let (layers, topk, rows) = (43, 6, 5);
        for device_spark in [false, true] {
            let local = 0..4;
            let mut routes = RoundRoutes::new(layers, topk);
            routes.begin(rows);
            let ids: Vec<u32> = (0..rows * topk).map(|i| i as u32 % 256).collect();
            let mut staged = vec![false; layers];
            let mut ringed = vec![false; layers];
            let mut slots = Vec::new();
            for layer in 0..layers {
                if local.contains(&layer) || device_spark {
                    slots.push((layer, ids.clone()));
                    ringed[layer] = true;
                } else {
                    routes.push(layer, &ids, ROUTE_MASK).unwrap();
                    staged[layer] = true;
                }
            }
            let bytes = ring(layers, rows, topk, &slots);
            let mut missing = ringed.clone();
            missing[2] = false;
            assert_eq!(complete_routes(&mut routes, &bytes, rows, topk, &missing, &staged,
                &mut RingCheck::default()), Err(RouteError::Missing(2)));
            complete_routes(&mut routes, &bytes, rows, topk, &ringed, &staged, &mut RingCheck::default()).unwrap();
            let geometry = PolicyGeometry {
                layers: (0..layers).map(|layer| LayerResource { class: Some(if local.contains(&layer) { 0 } else { 1 }),
                    slice_bytes: if local.contains(&layer) { 9e6 } else { 3e6 }, group_rows: 16, timed: layer > 0 }).collect(),
                group_rows_by_rows: (0..=64).map(|rows| vec![if rows > 0 && rows <= 4 { 1 } else { 16 }; layers]).collect(),
                experts: 256, topk: topk as u8, max_requests: 64, max_positions: 37, widths: vec![5],
                classes: vec![ResourceClass::rtx(), ResourceClass::spark()], regimes: 1,
            };
            let mut policy = DraftPolicy::seeded(geometry, PolicySeed::weak(5)).unwrap();
            let prior = [0.8; 3];
            let requests = [
                RoundRequest { id: 1, rows: 4, accepted: 3, source: DraftSource::Neural,
                    evidence: Some(Evidence::History(&prior)), censor: None },
                RoundRequest { id: 2, rows: 1, accepted: 1, source: DraftSource::Undrafted, evidence: None, censor: None },
            ];
            let layer_us: Vec<_> = (0..layers).map(|layer| (layer > 0).then_some(800.)).collect();
            binding::observe(&mut policy, &RoundRecord { shared: false, requests: &requests, routes: &routes,
                layer_us: &layer_us, times: RoundTimes { total_us: 50_000, draft_us: Some(3_000) },
                draft: Some(DraftPass { width: 5, requests: Some(1) }), predicted: None }).unwrap();
            assert_eq!(policy.stats().rounds, 1);
            assert_eq!(policy.stats().verified_rows, 5);
            assert_eq!(policy.stats().accepted_drafts, 2);
            assert_eq!(&policy.stats().position_reached[..3], &[1, 1, 1]);
            let cost = policy.cost_snapshot(false, 2);
            assert!(cost.layers.iter().all(|class| class[3] == 1.));
            assert_eq!(cost.draft[4], 1.);
        }
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
