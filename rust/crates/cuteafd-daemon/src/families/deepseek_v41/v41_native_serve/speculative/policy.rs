//! Binding for the online verification-length policy: the installed expert
//! placement, round observations from captured routes/timings, and the
//! `/v1/stats` export. The policy itself lives in `cuteafd_core` and performs
//! no device work.
use anyhow::{ensure, Result};
use crate::families::deepseek_v41::v41_experts::coordinator::NativeTp4Wave;
use crate::shared::draft::binding::{self, DraftPass, LaneRound, ObserveError, RoundRecord, RoundRequest};
use crate::shared::draft::clock::RoundTimes;
use crate::shared::draft::evidence::{sigmoid, DraftSource, Evidence};
use cuteafd_core::{DraftPolicy, LayerResource, PolicyGeometry, ResourceClass};
use std::collections::BTreeMap;
use std::sync::Mutex;

const HIDDEN: f64 = 5120.;
const INTERMEDIATE: f64 = 2304.;
/// Backbone layers.
pub(super) const LAYERS: usize = 40;
const EXPERTS: u16 = 384;
pub(super) const TOPK: usize = 6;
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

/// Layers whose experts run on the coordinator.
pub(super) fn local_layers(geometry: &PolicyGeometry) -> usize {
    geometry.layers.iter().filter(|layer| layer.class == Some(LOCAL)).count()
}

/// Per-device expert bytes of the last (Spark-side unless all local) layer.
pub(super) fn remote_expert_bytes(geometry: &PolicyGeometry) -> f64 {
    geometry.layers[LAYERS - 1].slice_bytes
}

/// One verified request of a lane round: identity, verifier rows, accepted
/// inputs, and whether its drafts were copied from its own history.
pub(super) type LaneRequest = (u64, usize, u32, bool);

/// Feed one completed lane round to the policy through the shared observation
/// builder. `requests` are in lane order; `lane.width` is the round's dSpark
/// draft width (zero when no draft pass ran) and `lane.predicted` the
/// selection's prediction, taken here.
///
/// A copied request's drafts came from its own history, not the drafter, so it
/// is observed without confidence: its rows still teach the cost fits and its
/// route history, never the drafter's calibration or acceptance evidence. A
/// round with a copy is not a shape the policy selected, so it keeps no
/// prediction. A draft pass beside a copy covered only the other requests, so
/// it gives no draft-cost sample, and its time is taken out of the round total
/// so the round fit's residual stays exact.
///
/// V4.1 truncates proposals by grammar and output budget before verification,
/// so the executed rows already exclude them; a stop token on an accepted draft
/// is still booked as today (no [`Censor`](crate::shared::draft::evidence::Censor)),
/// which keeps V4.1's decisions unchanged.
#[allow(clippy::too_many_arguments)]
pub(super) fn observe(policy: &mut DraftPolicy, confidence_trace: &BTreeMap<u64, Vec<f32>>, lane: &mut LaneRound,
    shared: bool, capture: &[Vec<[u32; 6]>], layer_us: &[Option<f64>], requests: &[LaneRequest], times: RoundTimes)
    -> Result<(), ObserveError> {
    let predicted = lane.predicted.take();
    // Every drafted position's confidence, verified or not: the policy
    // bounds reliability evidence to reached positions itself.
    let heads: Vec<Option<Vec<f64>>> = requests.iter().map(|&(id, _, _, copied)| if copied { None }
        else { confidence_trace.get(&id).map(|logits| logits.iter().map(|&x| sigmoid(x)).collect()) })
        .collect();
    let round: Vec<_> = requests.iter().zip(&heads).map(|(&(id, rows, accepted, copied), head)| RoundRequest {
        id, rows, accepted: accepted as usize,
        source: if copied { DraftSource::Copy } else if head.is_some() { DraftSource::Neural }
            else { DraftSource::Undrafted },
        evidence: head.as_deref().map(Evidence::Head),
        censor: None,
    }).collect();
    let rows = requests.iter().map(|request| request.1).sum();
    lane.routes.fill(capture, rows, ROUTE_EXPERT_MASK)?;
    binding::observe(policy, &RoundRecord { shared, requests: &round, routes: &lane.routes,
        layer_us, times, draft: (lane.width > 0).then_some(DraftPass { width: lane.width }), predicted })
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

    use cuteafd_core::{DraftRoundObservation, ObservedDraftRequest};

    fn placement() -> PolicyGeometry {
        remote_geometry(7)
    }

    /// The V4.1 observation exactly as built before the shared binding (work/p0
    /// ad1e529b `policy::observe`), kept as the reference the binding must match.
    #[allow(clippy::too_many_arguments)]
    fn observe_before(policy: &mut DraftPolicy, confidence_trace: &BTreeMap<u64, Vec<f32>>, shared: bool,
        routes: &[Vec<[u32; 6]>], layer_us: &[Option<f64>], requests: &[(u64, usize, u32, bool)],
        total_us: u64, draft_us: u64, width: usize, predicted: Option<f64>) -> Result<(), cuteafd_core::DraftPolicyError> {
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
    fn flat_routes(routes: &[Vec<[u32; 6]>]) -> Vec<u16> {
        routes.iter().flatten().flatten().map(|&expert| (expert & ROUTE_EXPERT_MASK) as u16).collect()
    }
    /// The shared-binding path with the pre-binding argument list.
    #[allow(clippy::too_many_arguments)]
    fn observe_now(policy: &mut DraftPolicy, confidence_trace: &BTreeMap<u64, Vec<f32>>, shared: bool,
        routes: &[Vec<[u32; 6]>], layer_us: &[Option<f64>], requests: &[(u64, usize, u32, bool)],
        total_us: u64, draft_us: u64, width: usize, predicted: Option<f64>) -> Result<(), ObserveError> {
        let mut lane = LaneRound::new(LAYERS, TOPK);
        lane.width = width;
        lane.predicted = predicted;
        observe(policy, confidence_trace, &mut lane, shared, routes, layer_us, requests,
            RoundTimes { total_us, draft_us: Some(draft_us) })
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
            observe_now(&mut after, &trace(), false, &routes(13), &layer_us, &requests, 20_000, 1_500, width, predicted)
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
            observe_now(&mut policy, &trace(), false, &routes(rows), &layer_us, requests, 20_000, 1_500, width, predicted)
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

    /// The pre-binding observation's fields, for comparison with the shared
    /// builder's (Debug prints NaN, so equal strings mean equal observations).
    #[allow(clippy::too_many_arguments)]
    fn observation_before(policy: &DraftPolicy, confidence_trace: &BTreeMap<u64, Vec<f32>>, shared: bool,
        routes: &[Vec<[u32; 6]>], layer_us: &[Option<f64>], requests: &[(u64, usize, u32, bool)],
        total_us: u64, draft_us: u64, width: usize, predicted: Option<f64>) -> String {
        let confidence: Vec<Option<Vec<f64>>> = requests.iter().map(|&(id, _, _, copied)| if copied { None }
            else { confidence_trace.get(&id).map(|logits| logits.iter().map(|&x| sigmoid(x)).collect()) })
            .collect();
        let observed: Vec<_> = requests.iter().zip(&confidence).map(|(&(id, rows, accepted, _), confidence)|
            ObservedDraftRequest { id, rows, accepted: accepted as usize, confidence: confidence.as_deref() })
            .collect();
        let layer_us: Vec<Option<f64>> = (0..LAYERS).map(|layer| layer_us.get(layer).copied().flatten()).collect();
        let routes = flat_routes(routes);
        let copied = requests.iter().any(|request| request.3);
        let (total_us, draft_us) = match (width > 0, copied) {
            (true, false) => (total_us as f64, draft_us as f64),
            (true, true) => (total_us.saturating_sub(draft_us) as f64, f64::NAN),
            (false, _) => (total_us as f64, f64::NAN),
        };
        format!("{:?}", DraftRoundObservation { shared, requests: &observed, routes: &routes, layer_us: &layer_us,
            total_us, predicted_us: if copied { None } else { predicted }, draft_us,
            wide: width > policy.widths()[0] })
    }

    /// Recorded V4.1 lane rounds (every shape the serve loop produces: both
    /// regimes, one to eight requests, widths 0/5/7, copied and undrafted
    /// requests, stale traces, router flag bits, missing layer times, stalls
    /// and malformed captures) through the pre-binding observation and the
    /// shared builder in lockstep: identical observations, identical
    /// decisions, identical policy state after every round.
    #[test]
    fn shared_builder_reproduces_recorded_v41_observations() {
        let mut seed = 0x0d1u64;
        let mut next = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize };
        for (scenario, geometry) in [remote_geometry(7), remote_geometry(5),
            geometry(|layer| layer < 20, |layer| if layer < 20 { expert_slice_bytes(1, false) }
                else { expert_slice_bytes(4, true) }, 7)].into_iter().enumerate() {
            let wide = *geometry.widths.last().unwrap();
            let mut before = DraftPolicy::new(geometry.clone(), false).unwrap();
            let mut after = DraftPolicy::new(geometry, false).unwrap();
            let mut lane = LaneRound::new(LAYERS, TOPK);
            let mut trace: BTreeMap<u64, Vec<f32>> = BTreeMap::new();
            let mut live: Vec<u64> = vec![1];
            let mut rejected = 0;
            for round in 0..400usize {
                if live.len() < 8 && next() % 7 == 0 { live.push(100 * scenario as u64 + round as u64 + 2); }
                if live.len() > 1 && next() % 13 == 0 {
                    let gone = live.remove(next() % live.len());
                    before.release(gone);
                    after.release(gone);
                    trace.remove(&gone);
                }
                let shared = next() % 3 == 0;
                // Width and lengths through both policies' own decisions.
                let budgets: Vec<_> = live.iter().map(|&id| (id, 1 + next() % wide)).collect();
                let width = before.choose_width(shared, &budgets);
                assert_eq!(width, after.choose_width(shared, &budgets), "round {round}");
                let width = if next() % 11 == 0 { 0 } else { width };
                let copied: Vec<bool> = live.iter().map(|_| width > 0 && next() % 9 == 0).collect();
                for (&id, &copied) in live.iter().zip(&copied) {
                    if copied || width == 0 { if next() % 2 == 0 { trace.remove(&id); } continue; }
                    trace.insert(id, (0..width).map(|_| (next() % 900) as f32 / 100. - 3.).collect());
                }
                let available: Vec<_> = live.iter().zip(&copied).filter(|(_, c)| !**c)
                    .map(|(&id, _)| (id, if width == 0 { 0 } else { next() % (width + 1) })).collect();
                let probabilities: Vec<Vec<f64>> = available.iter().map(|&(id, n)| trace.get(&id)
                    .map_or(Vec::new(), |t| t[..n.min(t.len())].iter().map(|&x| sigmoid(x)).collect())).collect();
                let candidates: Vec<_> = available.iter().zip(&probabilities)
                    .map(|(&(id, _), c)| cuteafd_core::DraftCandidate { id, confidence: c }).collect();
                let (mut predicted, mut lengths) = (None, probabilities.iter().map(Vec::len).collect::<Vec<_>>());
                if !candidates.is_empty() {
                    let selected = before.select(shared, &candidates, width.max(5)).unwrap();
                    assert_eq!(selected, after.select(shared, &candidates, width.max(5)).unwrap(), "round {round}");
                    if let Some(selection) = selected { predicted = Some(selection.predicted_us);
                        lengths = selection.lengths; }
                }
                let mut drafted = lengths.into_iter();
                let requests: Vec<(u64, usize, u32, bool)> = live.iter().zip(&copied).map(|(&id, &copied)| {
                    let drafts = if copied { 2 + next() % 6 } else { drafted.next().unwrap_or(0) };
                    (id, drafts + 1, (1 + next() % (drafts + 1)) as u32, copied)
                }).collect();
                let rows: usize = requests.iter().map(|r| r.1).sum();
                let mut routes: Vec<Vec<[u32; 6]>> = (0..LAYERS).map(|_| (0..rows)
                    .map(|_| std::array::from_fn(|_| (next() % 384) as u32)).collect()).collect();
                if round % 5 == 0 { routes[next() % LAYERS][next() % rows][next() % 6] |= 512; }
                let mut layer_us: Vec<Option<f64>> = (0..LAYERS).map(|layer|
                    (layer > 0).then(|| 300. + (next() % 900) as f64)).collect();
                if round % 17 == 3 { layer_us[1 + next() % (LAYERS - 1)] = None; }
                if round % 29 == 7 { layer_us.truncate(LAYERS - 3); }
                // A malformed capture is rejected by both, leaving no trace.
                if round % 61 == 30 { routes[next() % LAYERS].pop(); rejected += 1; }
                let total_us = 8_000 + 40 * rows as u64 + next() as u64 % 3_000 + if round % 37 == 1 { 30_000 } else { 0 };
                let draft_us = 1_500 + next() as u64 % 800;
                let expected = observation_before(&before, &trace, shared, &routes, &layer_us, &requests,
                    total_us, draft_us, width, predicted);
                let old = observe_before(&mut before, &trace, shared, &routes, &layer_us, &requests, total_us,
                    draft_us, width, predicted);
                lane.width = width;
                lane.predicted = predicted;
                let well_formed = lane.routes.fill(&routes, rows, ROUTE_EXPERT_MASK).is_ok();
                assert_eq!(well_formed, old.is_ok(), "round {round}");
                if well_formed {
                    let heads: Vec<Option<Vec<f64>>> = requests.iter().map(|&(id, _, _, copied)| if copied { None }
                        else { trace.get(&id).map(|t| t.iter().map(|&x| sigmoid(x)).collect()) }).collect();
                    let round_requests: Vec<_> = requests.iter().zip(&heads).map(|(&(id, rows, accepted, copied), head)|
                        RoundRequest { id, rows, accepted: accepted as usize,
                            source: if copied { DraftSource::Copy } else if head.is_some() { DraftSource::Neural }
                                else { DraftSource::Undrafted },
                            evidence: head.as_deref().map(Evidence::Head), censor: None }).collect();
                    let record = RoundRecord { shared, requests: &round_requests, routes: &lane.routes,
                        layer_us: &layer_us, times: RoundTimes { total_us, draft_us: Some(draft_us) },
                        draft: (width > 0).then_some(DraftPass { width }), predicted };
                    assert_eq!(format!("{:?}", binding::build(&record, after.widths()[0]).unwrap().view()), expected,
                        "round {round}");
                }
                let new = observe(&mut after, &trace, &mut lane, shared, &routes, &layer_us, &requests,
                    RoundTimes { total_us, draft_us: Some(draft_us) });
                assert_eq!((old.is_ok(), new.is_ok()), (well_formed, well_formed), "round {round}");
                assert_eq!(lane.predicted, None);
                assert_eq!(format!("{before:?}"), format!("{after:?}"), "scenario {scenario} round {round}");
            }
            assert!(rejected > 0 && before.stats().rounds > 300 && before.stats().selected_rounds > 0);
            assert!(before.stats().width_rounds.iter().all(|&n| n > 0) || wide == 5);
        }
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
        let mut routes = crate::shared::draft::routes::RoundRoutes::new(1, TOPK);
        routes.fill(&[vec![[512 | 383, 1, 2, 3, 4, 5]]], 1, ROUTE_EXPERT_MASK).unwrap();
        assert_eq!(routes.flat()[0], 383);
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
