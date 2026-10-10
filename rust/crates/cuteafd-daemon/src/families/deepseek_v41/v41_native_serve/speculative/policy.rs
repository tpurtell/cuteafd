//! Binding for the online verification-length policy: the installed expert
//! placement, round observations from captured routes/timings, and the
//! `/v1/stats` export. The policy itself lives in `cuteafd_core` and performs
//! no device work.
use anyhow::{ensure, Result};
use crate::families::deepseek_v41::v41_experts::coordinator::NativeTp4Wave;
use cuteafd_core::{DraftPolicy, DraftRoundObservation, LayerResource, ObservedDraftRequest, PolicyGeometry,
    ResourceClass};
use std::collections::BTreeMap;
use std::sync::Mutex;

const HIDDEN: f64 = 5120.;
const INTERMEDIATE: f64 = 2304.;
/// Backbone layers.
pub(super) const LAYERS: usize = 40;
const EXPERTS: u16 = 384;
const TOPK: usize = 6;
/// Rows per weight-read group of the grouped slice kernels.
const GROUP_ROWS: u8 = 16;
/// Requests per lane round: the draft runtime's request limit.
const MAX_REQUESTS: usize = 16;
/// The dSpark drafter's trained block; a width-7 load can also draft 5.
const NATIVE_WIDTH: usize = 5;
/// Resource classes: coordinator RTX (TP1 or TP2) and Spark expert ranks.
const LOCAL: u8 = 0;
const REMOTE: u8 = 1;
/// Router ids carry flag bits above the expert index.
const ROUTE_EXPERT_MASK: u32 = 511;

/// Bytes one device reads for one routed expert: gate, up and down slices of
/// `hidden x intermediate / tp` 4-bit values plus their UE8M0 (MXFP4, per 32)
/// or E4M3 (NVFP4, per 16) block scales. This is the standardized base for the
/// official checkpoint; other formats' differences land in the fitted bandwidth.
pub(super) fn expert_slice_bytes(tp: usize, nvfp4: bool) -> f64 {
    let scale_block = if nvfp4 { 16. } else { 32. };
    3. * HIDDEN * (INTERMEDIATE / tp as f64) * (0.5 + 1. / scale_block)
}

/// V4.1's policy geometry: per layer, whether its experts are local and the
/// per-device slice bytes; `draft_width` is the loaded drafter's width (5 or 7).
pub(super) fn geometry(local: impl Fn(usize) -> bool, slice_bytes: impl Fn(usize) -> f64, draft_width: usize)
    -> PolicyGeometry {
    PolicyGeometry {
        // Layer 0 has no preceding timing boundary: its time is in the residual.
        layers: (0..LAYERS).map(|layer| LayerResource {
            class: Some(if local(layer) { LOCAL } else { REMOTE }), slice_bytes: slice_bytes(layer),
            group_rows: GROUP_ROWS, timed: layer > 0 }).collect(),
        experts: EXPERTS,
        topk: TOPK as u8,
        max_requests: MAX_REQUESTS,
        max_positions: cuteafd_core::MAX_DSPARK_PROPOSALS,
        widths: if draft_width > NATIVE_WIDTH { vec![NATIVE_WIDTH, draft_width] } else { vec![draft_width] },
        classes: vec![ResourceClass::rtx(), ResourceClass::spark()],
        // Solo and shared: a lane's round beside an active peer lane.
        regimes: 2,
    }
}

/// Geometry of the installed placement: resource class and per-device expert
/// bytes of every layer.
pub(super) fn placement(transport: &NativeTp4Wave<'_>, nvfp4: bool, draft_width: usize) -> Result<PolicyGeometry> {
    let remote_tp = transport.native_topology()
        .map_or(transport.spark_world(), |topology| topology.tp() as usize);
    ensure!(remote_tp > 0, "Spark tensor-parallel width is zero");
    Ok(geometry(|layer| transport.has_local_layer(layer), |layer| {
        let tp = if transport.has_tp2_layer(layer) { 2 }
            else if transport.has_local_layer(layer) { 1 } else { remote_tp };
        expert_slice_bytes(tp, nvfp4)
    }, draft_width))
}

/// All-Spark geometry at the official TP4 slice (fixtures).
#[cfg(test)]
pub(super) fn remote_geometry(draft_width: usize) -> PolicyGeometry {
    geometry(|_| false, |_| expert_slice_bytes(4, false), draft_width)
}

/// Captured `[layer][row]` routes as the policy's layer-major expert ids.
fn flat_routes(routes: &[Vec<[u32; 6]>]) -> Vec<u16> {
    routes.iter().flatten().flatten().map(|&expert| (expert & ROUTE_EXPERT_MASK) as u16).collect()
}

/// Layers whose experts run on the coordinator.
pub(super) fn local_layers(geometry: &PolicyGeometry) -> usize {
    geometry.layers.iter().filter(|layer| layer.class == Some(LOCAL)).count()
}

/// Per-device expert bytes of the last (Spark-side unless all local) layer.
pub(super) fn remote_expert_bytes(geometry: &PolicyGeometry) -> f64 {
    geometry.layers[LAYERS - 1].slice_bytes
}

pub(super) fn sigmoid(x: f32) -> f64 {
    let x = f64::from(x);
    if x >= 0. { 1. / (1. + (-x).exp()) } else { x.exp() / (1. + x.exp()) }
}

/// Feed one completed lane round to the policy. `requests` are (identity,
/// verifier rows, accepted inputs, copied) in lane order, and `width` is the
/// round's dSpark draft width (zero when no draft pass ran).
///
/// A copied request's drafts came from its own history, not the drafter, so it
/// is observed without confidence: its rows still teach the cost fits and its
/// route history, never the drafter's calibration or acceptance evidence. A
/// round with a copy is not a shape the policy selected, so it keeps no
/// prediction. A draft pass beside a copy covered only the other requests, so
/// it gives no draft-cost sample, and its time is taken out of the round total
/// so the round fit's residual stays exact.
#[allow(clippy::too_many_arguments)]
pub(super) fn observe(policy: &mut DraftPolicy, confidence_trace: &BTreeMap<u64, Vec<f32>>, shared: bool,
    routes: &[Vec<[u32; 6]>], layer_us: &[Option<f64>], requests: &[(u64, usize, u32, bool)],
    total_us: u64, draft_us: u64, width: usize, predicted: Option<f64>) -> Result<(), cuteafd_core::DraftPolicyError> {
    // Every drafted position's confidence, verified or not: the policy
    // bounds reliability evidence to reached positions itself.
    let confidence: Vec<Option<Vec<f64>>> = requests.iter().map(|&(id, _, _, copied)| if copied { None }
        else { confidence_trace.get(&id).map(|logits| logits.iter().map(|&x| sigmoid(x)).collect()) })
        .collect();
    let observed: Vec<_> = requests.iter().zip(&confidence).map(|(&(id, rows, accepted, _), confidence)|
        ObservedDraftRequest { id, rows, accepted: accepted as usize,
            confidence: confidence.as_deref() }).collect();
    let layer_us: Vec<Option<f64>> = (0..LAYERS).map(|layer| layer_us.get(layer).copied().flatten()).collect();
    let rows: usize = requests.iter().map(|request| request.1).sum();
    if routes.len() != LAYERS || routes.iter().any(|layer| layer.len() != rows) {
        return Err(cuteafd_core::DraftPolicyError::RouteShape);
    }
    let routes = flat_routes(routes);
    let copied = requests.iter().any(|request| request.3);
    let (total_us, draft_us) = match (width > 0, copied) {
        (true, false) => (total_us as f64, draft_us as f64),
        (true, true) => (total_us.saturating_sub(draft_us) as f64, f64::NAN),
        (false, _) => (total_us as f64, f64::NAN),
    };
    policy.observe(DraftRoundObservation { shared, requests: &observed,
        routes: &routes, layer_us: &layer_us, total_us, predicted_us: if copied { None } else { predicted },
        draft_us, wide: width > policy.widths()[0] })
}

static SNAPSHOT: Mutex<Option<serde_json::Value>> = Mutex::new(None);

/// The latest published policy state for the serving `stats` payload.
pub(crate) fn snapshot() -> serde_json::Value {
    SNAPSHOT.lock().ok().and_then(|slot| slot.clone()).unwrap_or(serde_json::Value::Null)
}

pub(super) fn publish(policy: &DraftPolicy, draft_limit: usize) {
    let stats = policy.stats();
    let geometry = policy.geometry();
    let (narrow, wide) = (policy.widths()[0], policy.widths()[policy.widths().len() - 1]);
    let fit = |shared: bool| {
        let snapshot = policy.cost_snapshot(shared);
        let classes: Vec<_> = geometry.classes.iter()
            .zip(snapshot.layers).map(|(class, c)| serde_json::json!({
                "class": class.label,
                "intercept_us": c[0], "us_per_row": c[1], "us_per_mb": c[2],
                // µs per MB → GB/s: 1 MB / (c µs) = 1000 / c GB/s.
                "effective_gb_per_s": if c[2] > 0. { 1000. / c[2] } else { f64::INFINITY },
                "samples": c[3], "residual_scale_us": c[4],
            })).collect();
        serde_json::json!({
            "warm": policy.warm(shared),
            "layers": classes,
            "round": { "intercept_us": snapshot.round[0], "us_per_row": snapshot.round[1],
                "us_per_request": snapshot.round[2], "samples": snapshot.round[3],
                "residual_scale_us": snapshot.round[4] },
            "draft": { "intercept_us": snapshot.draft[0], "us_per_request": snapshot.draft[1],
                "wide_extra_us": snapshot.draft[2], "wide_extra_us_per_request": snapshot.draft[3],
                "samples": snapshot.draft[4], "residual_scale_us": snapshot.draft[5] },
        })
    };
    let calibration = policy.calibration();
    let reliability: Vec<_> = (0..stats.position_reached.len()).map(|p| serde_json::json!({
        "position": p + 1,
        "reached": stats.position_reached[p],
        "mean_confidence": if stats.position_reached[p] > 0 {
            stats.position_confidence[p] / stats.position_reached[p] as f64 } else { 0. },
        "mean_raw_confidence": if stats.position_reached[p] > 0 {
            stats.position_raw_confidence[p] / stats.position_reached[p] as f64 } else { 0. },
        "logit_slope": calibration[p].0,
        "logit_offset": calibration[p].1,
        "accept_rate": if stats.position_reached[p] > 0 {
            stats.position_accepted[p] as f64 / stats.position_reached[p] as f64 } else { 0. },
    })).collect();
    let value = serde_json::json!({
        "mode": if policy.fixed() { "fixed" } else { "bandwidth" },
        "draft_limit": draft_limit,
        "local_layers": local_layers(geometry),
        "remote_expert_bytes": remote_expert_bytes(geometry),
        "rounds": stats.rounds,
        "selected_rounds": stats.selected_rounds,
        "verified_rows": stats.verified_rows,
        "verified_drafts": stats.verified_drafts,
        "accepted_drafts": stats.accepted_drafts,
        "request_rounds": stats.emitted_requests,
        "draft_rows_histogram": stats.draft_rows,
        "draft_widths": { "narrow": narrow, "wide": wide },
        "width_rounds": { "narrow": stats.width_rounds[0], "wide": stats.width_rounds[1] },
        "confidence_reliability": reliability,
        "time_bias_us": { "solo": policy.time_bias()[0], "shared": policy.time_bias()[1] },
        "prediction": {
            "rounds": stats.predicted_rounds,
            "mean_error_us": if stats.predicted_rounds > 0 {
                stats.prediction_error_us / stats.predicted_rounds as f64 } else { 0. },
            "mean_abs_relative_error": if stats.observed_us > 0. {
                stats.prediction_abs_error_us / stats.observed_us } else { 0. },
        },
        "solo": fit(false),
        "shared": fit(true),
    });
    if let Ok(mut slot) = SNAPSHOT.lock() { *slot = Some(value); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement() -> PolicyGeometry {
        remote_geometry(7)
    }
    fn routes(rows: usize) -> Vec<Vec<[u32; 6]>> {
        (0..LAYERS).map(|layer| (0..rows).map(|row|
            std::array::from_fn(|slot| ((layer * 7 + row * 6 + slot) % 384) as u32)).collect()).collect()
    }
    /// Request 1 drafted this round; request 2's trace is left from an earlier
    /// dSpark round.
    fn trace() -> BTreeMap<u64, Vec<f32>> {
        [(1, vec![2.0; 5]), (2, vec![-4.0; 5])].into()
    }

    /// Without copies the observation is exactly the one built before copy
    /// windows existed: the policy ends in the same state.
    #[test]
    fn rounds_without_copies_observe_as_before() {
        let layer_us = vec![Some(150.0); LAYERS];
        let requests = [(1u64, 5usize, 3u32, false), (2, 8, 1, false)];
        for (width, predicted) in [(5, Some(18_000.)), (7, None), (0, Some(9_000.))] {
            let confidence: Vec<Option<Vec<f64>>> = requests.iter().map(|&(id, ..)|
                trace().get(&id).map(|logits| logits.iter().map(|&x| sigmoid(x)).collect())).collect();
            let observed: Vec<_> = requests.iter().zip(&confidence).map(|(&(id, rows, accepted, _), confidence)|
                ObservedDraftRequest { id, rows, accepted: accepted as usize, confidence: confidence.as_deref() })
                .collect();
            let mut before = DraftPolicy::new(placement(), false).unwrap();
            before.observe(DraftRoundObservation { shared: false, requests: &observed, routes: &flat_routes(&routes(13)),
                layer_us: &layer_us, total_us: 20_000., predicted_us: predicted,
                draft_us: if width > 0 { 1_500. } else { f64::NAN }, wide: width > 5 }).unwrap();
            let mut after = DraftPolicy::new(placement(), false).unwrap();
            observe(&mut after, &trace(), false, &routes(13), &layer_us, &requests, 20_000, 1_500, width, predicted)
                .unwrap();
            assert_eq!(format!("{before:?}"), format!("{after:?}"), "width {width}");
        }
    }

    /// A copied request is verified like any other but drafted by nobody: its
    /// rows teach the cost fits, never the drafter's calibration or acceptance
    /// evidence, the draft-cost fit or the time bias, even with a stale trace.
    #[test]
    fn copy_rounds_teach_costs_but_never_the_drafters_evidence() {
        let layer_us = vec![Some(150.0); LAYERS];
        // Request 1 verified four drafts and accepted two; request 2 verified
        // seven and accepted none.
        let round = |copied| [(1u64, 5usize, 3u32, false), (2, 8, 1, copied)];
        let observe_round = |requests: &[(u64, usize, u32, bool)], width, predicted| {
            let mut policy = DraftPolicy::new(placement(), false).unwrap();
            let rows = requests.iter().map(|r| r.1).sum();
            observe(&mut policy, &trace(), false, &routes(rows), &layer_us, requests, 20_000, 1_500, width, predicted)
                .unwrap();
            policy
        };
        let control = observe_round(&round(false), 5, Some(18_000.));
        let copy = observe_round(&round(true), 5, Some(18_000.));
        let solo = observe_round(&round(false)[..1], 5, Some(18_000.));
        // Sensitivity: the drafted request 2 adds evidence at position one.
        assert_eq!(control.stats().position_reached[..3], [2, 1, 1]);
        // The copied request adds none: exactly request 1's evidence alone.
        assert_eq!(copy.stats().position_reached, solo.stats().position_reached);
        assert_eq!(copy.stats().position_accepted, solo.stats().position_accepted);
        assert_eq!(copy.calibration(), solo.calibration());
        assert_ne!(control.calibration(), solo.calibration());
        // No prediction and no time bias from a shape the policy did not select.
        assert_eq!((control.stats().predicted_rounds, copy.stats().predicted_rounds), (1, 0));
        assert_ne!(control.time_bias(), [0.; 2]);
        assert_eq!(copy.time_bias(), [0.; 2]);
        // The draft pass beside a copy covered one request of two: no draft sample.
        assert_eq!((control.cost_snapshot(false).draft[4], copy.cost_snapshot(false).draft[4]), (1., 0.));
        assert_eq!(copy.stats().width_rounds, [0, 0]);
        // The copy's rows still teach the layer and round fits, and taking the
        // draft time out of the total leaves the round residual exact.
        assert_eq!(copy.cost_snapshot(false).layers, control.cost_snapshot(false).layers);
        assert_eq!(copy.cost_snapshot(false).round, control.cost_snapshot(false).round);
        assert_eq!((copy.stats().rounds, copy.stats().verified_rows, copy.stats().verified_drafts), (1, 13, 11));

        // A round in which every request copied ran no draft pass at all.
        let copies = observe_round(&[(2, 8, 4, true)], 0, None);
        let undrafted = observe_round(&[(3, 8, 4, false)], 0, None);
        assert_eq!(copies.stats().position_reached, [0; 7]);
        assert_eq!(copies.cost_snapshot(false).draft[4], 0.);
        assert_eq!(copies.cost_snapshot(false).round, undrafted.cost_snapshot(false).round);
    }

    /// The V4.1 geometry is the constants the policy hard-coded before it took
    /// a geometry: 40 layers (layer 0 untimed), 384 experts, top-6, 16-row
    /// groups, 16 requests, 7 positions, widths (5, 7) or 5 alone, two
    /// regimes, local RTX and remote Spark classes with their old priors.
    #[test]
    fn v41_geometry_reproduces_the_former_constants() {
        let geometry = geometry(|layer| layer < 20, |layer| if layer < 3 { expert_slice_bytes(2, false) }
            else if layer < 20 { expert_slice_bytes(1, false) } else { expert_slice_bytes(4, true) }, 7);
        assert_eq!((geometry.layers.len(), geometry.experts, geometry.topk), (40, 384, 6));
        assert_eq!((geometry.max_requests, geometry.max_positions, geometry.regimes), (16, 7, 2));
        assert_eq!(geometry.widths, [5, 7]);
        assert_eq!(super::geometry(|_| false, |_| 1., 5).widths, [5]);
        assert!(geometry.layers.iter().all(|layer| layer.group_rows == 16));
        assert_eq!(geometry.layers.iter().map(|layer| layer.timed).collect::<Vec<_>>(),
            (0..40).map(|layer| layer > 0).collect::<Vec<_>>());
        let class = |layer: usize| geometry.layers[layer].class;
        assert_eq!((class(0), class(19), class(20), class(39)), (Some(0), Some(0), Some(1), Some(1)));
        assert_eq!((geometry.layers[2].slice_bytes, geometry.layers[19].slice_bytes, geometry.layers[39].slice_bytes),
            (9_400_320., 18_800_640., 4_976_640.));
        assert_eq!((local_layers(&geometry), remote_expert_bytes(&geometry)), (20, 4_976_640.));
        // Labels and priors of the former Local/Remote classes.
        let classes: Vec<_> = geometry.classes.iter().map(|c| (c.label, c.prior, c.prior_precision)).collect();
        assert_eq!(classes, [("local", [700., 15., 0.8], [0.01, 0.01 * 36., 0.01 * 300f64.powi(2)]),
            ("remote", [900., 15., 5.7], [0.01, 0.01 * 36., 0.01 * 100f64.powi(2)])]);
        let policy = DraftPolicy::new(geometry, false).unwrap();
        assert_eq!((policy.stats().draft_rows.len(), policy.stats().position_reached.len()), (8, 7));
        assert_eq!(policy.time_bias(), [0.; 2]);
        // Router flag bits above the expert index are dropped by the binding.
        assert_eq!(flat_routes(&[vec![[512 | 383, 1, 2, 3, 4, 5]]])[0], 383);
    }

    #[test]
    fn official_spark_tp4_slice_matches_the_packed_geometry() {
        // 3 x (1,474,560 FP4 bytes + 92,160 UE8M0 scale bytes).
        assert_eq!(super::expert_slice_bytes(4, false), 4_700_160.);
        assert_eq!(super::expert_slice_bytes(1, false), 18_800_640.);
        assert_eq!(super::expert_slice_bytes(2, false), 9_400_320.);
        assert_eq!(super::expert_slice_bytes(4, true), 4_976_640.);
    }
}
