//! One completed round as the draft policy's observation, and the per-lane
//! state a serve loop carries from length selection to that observation.
//!
//! A family's serve loop records, per lane round: the requests in verifier-row
//! order with their executed rows, accepted inputs, [`DraftSource`],
//! [`Evidence`] and [`Censor`] reason; the round's [`RoundRoutes`]; the
//! [`LayerClock`](super::clock::LayerClock) layer times; the
//! [`RoundClock`](super::clock::RoundClock) times; and whether a draft pass ran
//! and at which width. [`build`] turns that into `DraftRoundObservation`:
//! - neural evidence enters only for [`DraftSource::Neural`] requests, through
//!   the evidence's prior; copied rows teach costs and route history only;
//! - a censored request's evidence stops at its accepted prefix, so the
//!   position the verifier never judged is not booked as a miss;
//! - a round with any copy keeps no prediction (the policy did not select its
//!   shape); a mixed draft pass with an explicit batch extent still teaches
//!   its actual drafter cost. Unmigrated adapters without an extent omit that
//!   sample and remove its time from the round total.
use super::clock::RoundTimes;
use super::evidence::{Censor, DraftSource, Evidence};
use super::routes::{RoundRoutes, RouteError};
use cuteafd_core::{DraftPolicy, DraftPolicyError, DraftRoundObservation, ObservedDraftRequest};
use thiserror::Error;

/// Why a round was not observed. Never fatal to serving.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub(crate) enum ObserveError {
    #[error(transparent)]
    Routes(#[from] RouteError),
    #[error(transparent)]
    Policy(#[from] DraftPolicyError),
    #[error("round routes cover {routes} rows, requests {requests}")]
    Rows { routes: usize, requests: usize },
}

/// One verified request of a lane round.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoundRequest<'a> {
    pub id: u64,
    /// Verifier rows executed, anchor included (after grammar truncation).
    pub rows: usize,
    /// Accepted inputs, anchor included.
    pub accepted: usize,
    pub source: DraftSource,
    /// The drafter's evidence for this request's drafted positions, if any.
    pub evidence: Option<Evidence<'a>>,
    /// Set when verification stopped for a reason other than a miss.
    pub censor: Option<Censor>,
}

/// A draft pass that ran this round, at `width` positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DraftPass {
    pub width: usize,
    /// Actual drafter batch extent; None preserves an unmigrated adapter's pricing.
    pub requests: Option<usize>,
}

/// Everything one completed lane round reports.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoundRecord<'a> {
    /// Another lane had active requests during this round.
    pub shared: bool,
    pub requests: &'a [RoundRequest<'a>],
    pub routes: &'a RoundRoutes,
    /// Device µs per layer (missing entries `None`), by layer index.
    pub layer_us: &'a [Option<f64>],
    pub times: RoundTimes,
    pub draft: Option<DraftPass>,
    /// The policy's prediction for this round's shape, if it made one.
    pub predicted: Option<f64>,
}

/// A built observation; [`view`](Self::view) lends it to the policy.
#[derive(Clone, Debug)]
pub(crate) struct Observation<'a> {
    shared: bool,
    requests: Vec<ObservedDraftRequest<'a>>,
    routes: &'a [u16],
    layer_us: Vec<Option<f64>>,
    total_us: f64,
    draft_us: f64,
    drafted_requests: Option<usize>,
    wide: bool,
    predicted_us: Option<f64>,
}

impl Observation<'_> {
    pub fn view(&self) -> DraftRoundObservation<'_> {
        DraftRoundObservation { shared: self.shared, requests: &self.requests, routes: self.routes,
            layer_us: &self.layer_us, total_us: self.total_us, predicted_us: self.predicted_us,
            draft_us: self.draft_us, drafted_requests: self.drafted_requests, wide: self.wide }
    }
}

/// Build one round's observation. `native_width` is the narrow draft-cost
/// width (`policy.widths()[0]`), independent of the neural-confidence boundary.
pub(crate) fn build<'a>(record: &RoundRecord<'a>, native_width: usize) -> Result<Observation<'a>, ObserveError> {
    let rows: usize = record.requests.iter().map(|request| request.rows).sum();
    record.routes.complete()?;
    if record.routes.rows() != rows {
        return Err(ObserveError::Rows { routes: record.routes.rows(), requests: rows });
    }
    let requests = record.requests.iter().map(|request| ObservedDraftRequest {
        id: request.id, rows: request.rows, accepted: request.accepted,
        confidence: confidence(request),
    }).collect();
    let layer_us = (0..record.routes.layers())
        .map(|layer| record.layer_us.get(layer).copied().flatten()).collect();
    let copied = record.requests.iter().any(|request| request.source == DraftSource::Copy);
    let total = record.times.total_us;
    let draft = record.times.draft_us.unwrap_or(0);
    let extent_known = record.draft.is_some_and(|pass| pass.requests.is_some());
    let (total_us, draft_us) = match (record.draft.is_some(), copied && !extent_known) {
        (true, false) => (total as f64, draft as f64),
        (true, true) => (total.saturating_sub(draft) as f64, f64::NAN),
        (false, _) => (total as f64, f64::NAN),
    };
    Ok(Observation { shared: record.shared, requests, routes: record.routes.flat(), layer_us, total_us, draft_us,
        drafted_requests: record.draft.and_then(|pass| pass.requests),
        wide: record.draft.is_some_and(|pass| pass.width > native_width),
        predicted_us: if copied { None } else { record.predicted } })
}

/// The confidence one request contributes: its evidence's prior for a neural
/// draft, cut to the accepted prefix when censored.
fn confidence<'a>(request: &RoundRequest<'a>) -> Option<&'a [f64]> {
    if request.source != DraftSource::Neural { return None; }
    let prior = request.evidence?.prior()?;
    Some(match request.censor {
        Some(_) => &prior[..prior.len().min(request.accepted.saturating_sub(1))],
        None => prior,
    })
}

/// Feed one completed round to the policy.
pub(crate) fn observe(policy: &mut DraftPolicy, record: &RoundRecord<'_>) -> Result<(), ObserveError> {
    let observation = build(record, policy.widths()[0])?;
    trace_round(policy, record);
    let _timing = super::clock::HostSpan::new(super::clock::HostStage::Observe);
    policy.observe(observation.view())?;
    trace_simple(policy);
    Ok(())
}

/// The simple rule's round: each request's cost sample, windowed cost per
/// token, threshold and chosen width, with the row price behind them.
fn trace_simple(policy: &DraftPolicy) {
    let Some(round) = policy.simple_round() else { return };
    if tracing::enabled!(target: "cuteafd::draft_policy", tracing::Level::DEBUG) {
        tracing::debug!(target: "cuteafd::draft_policy", mode = policy.draft_mode().label(), total_us = round.total_us,
            rows = round.rows, global_cost_us = round.global_cost_us, row_price_us = round.row_price.used,
            chosen = ?round.requests.iter().map(|r| r.decision.as_ref().map(|d| d.chosen)).collect::<Vec<_>>(),
            accepted = ?round.requests.iter().map(|r| r.accepted).collect::<Vec<_>>(),
            cost_us = ?round.requests.iter().map(|r| r.cost_us).collect::<Vec<_>>(),
            waste = ?round.requests.iter().map(|r| r.decision.as_ref().map(|d| d.waste)).collect::<Vec<_>>(),
            threshold = ?round.requests.iter().map(|r| r.decision.as_ref().map(|d| d.threshold)).collect::<Vec<_>>(),
            "simple draft round");
    }
    let Some(trace) = replay_trace() else { return };
    if let Err(error) = trace.write(&simple_trace_value(policy, round)) { tracing::warn!(%error, "simple draft trace failed"); }
}

fn simple_trace_value(policy: &DraftPolicy, round: &cuteafd_core::SimpleRound) -> serde_json::Value {
    let price = |p: &cuteafd_core::RowPrice| serde_json::json!({"short_us": p.short, "pooled_us": p.pooled, "used_us": p.used});
    serde_json::json!({"type": "simple_draft_round", "round": policy.stats().rounds, "mode": policy.draft_mode().label(),
        "total_us": round.total_us, "rows": round.rows, "global_cost_us": round.global_cost_us,
        "row_price": price(&round.row_price),
        "threshold_bounds": [cuteafd_core::THRESHOLD_FLOOR, cuteafd_core::THRESHOLD_CEILING],
        "waste_bounds": [cuteafd_core::WASTE_FLOOR, cuteafd_core::WASTE_CEILING],
        "requests": round.requests.iter().map(|r| serde_json::json!({"id": r.id, "rows": r.rows,
            "accepted": r.accepted, "share_us": r.share_us, "sample_us": r.sample_us, "cost_us": r.cost_us,
            "decision": r.decision.as_ref().map(|d| serde_json::json!({"cost_us": d.cost_us,
                "batch_cost_us": d.batch_cost_us, "waste": d.waste, "base": d.base, "threshold": d.threshold,
                "joint_rule": d.joint_rule, "conditional": d.conditional,
                "joint": d.joint, "passing": d.passing, "chosen": d.chosen, "explore": d.explore,
                "scarce": d.scarce, "budget": d.budget, "row_price": price(&d.row_price)}))
        })).collect::<Vec<_>>()})
}

/// Opt-in complete observation capture, including dSpark head evidence. The
/// selector-only confidence trace does not cover that serving path.
fn replay_trace() -> Option<&'static crate::shared::speculation_trace::Trace> {
    use std::sync::OnceLock;
    static TRACE: OnceLock<Option<crate::shared::speculation_trace::Trace>> = OnceLock::new();
    TRACE.get_or_init(|| {
        match crate::shared::speculation_trace::Trace::open() {
            Ok(trace) => trace,
            Err(error) => { tracing::warn!(%error, "draft replay trace open failed"); None }
        }
    }).as_ref()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn trace_action(policy: &DraftPolicy, shared: bool, requests: &[(u64, usize)],
    actions: &[cuteafd_core::DraftAction], action_rooms: Option<&[&[usize]]>, max_rows: usize,
    chosen: Option<usize>, executed: usize) {
    let Some(trace) = replay_trace() else { return };
    let value = action_trace_value(policy, shared, requests, actions, action_rooms, max_rows, chosen, executed);
    if let Err(error) = trace.write(&value) { tracing::warn!(%error, "draft action trace failed"); }
}

#[allow(clippy::too_many_arguments)]
fn action_trace_value(policy: &DraftPolicy, shared: bool, requests: &[(u64, usize)],
    actions: &[cuteafd_core::DraftAction], action_rooms: Option<&[&[usize]]>, max_rows: usize,
    chosen: Option<usize>, executed: usize) -> serde_json::Value {
    serde_json::json!({"type": "draft_action", "round": policy.stats().rounds,
        "shared": shared, "requests": requests, "max_rows": max_rows,
        "actions": actions.iter().enumerate().map(|(index, action)| serde_json::json!({
            "width": action.width, "draft_us": action.draft_us,
            "rooms": action_rooms.map(|rooms| rooms[index])
        })).collect::<Vec<_>>(), "chosen": chosen, "executed": executed,
        "cold": chosen.is_none(), "engaged": policy.engaged(shared, requests.len())})
}

pub(crate) fn trace_selection(policy: &DraftPolicy, candidates: &[cuteafd_core::DraftCandidate<'_>],
    width: usize, drafted_requests: usize, max_rows: usize, lengths: &[usize]) {
    let Some(trace) = replay_trace() else { return };
    let value = selection_trace_value(policy, candidates, width, drafted_requests, max_rows, lengths);
    if let Err(error) = trace.write(&value) { tracing::warn!(%error, "draft selection trace failed"); }
}

fn selection_trace_value(policy: &DraftPolicy, candidates: &[cuteafd_core::DraftCandidate<'_>],
    width: usize, drafted_requests: usize, max_rows: usize, lengths: &[usize]) -> serde_json::Value {
    serde_json::json!({"type": "draft_selection", "round": policy.stats().rounds,
        "requests": candidates.iter().zip(lengths).map(|(&candidate, &length)| serde_json::json!({
            "id": candidate.id, "raw_confidence": candidate.confidence,
            "calibrated_confidence": policy.probabilities(candidates.len(), candidate), "selected_drafts": length
        })).collect::<Vec<_>>(), "width": width, "drafted_requests": drafted_requests, "max_rows": max_rows})
}

fn trace_round(policy: &DraftPolicy, record: &RoundRecord<'_>) {
    let Some(trace) = replay_trace() else { return };
    let geometry = policy.geometry();
    if record.routes.layers() != geometry.layers.len()
        || record.routes.flat().len() != geometry.layers.len() * record.routes.rows() * usize::from(geometry.topk) {
        return;
    }
    let value = round_trace_value(policy, record);
    if let Err(error) = trace.write(&value) {
        tracing::warn!(%error, "draft round trace failed");
    }
}

fn round_trace_value(policy: &DraftPolicy, record: &RoundRecord<'_>) -> serde_json::Value {
    let geometry = policy.geometry();
    let rows = record.routes.rows();
    let topk = usize::from(geometry.topk);
    let routes = record.routes.flat();
    let megabytes: Vec<_> = geometry.layers.iter().enumerate().map(|(layer, resource)| {
        if resource.class.is_none() { return 0.; }
        let mut counts = vec![0usize; usize::from(geometry.experts)];
        for &expert in &routes[layer * rows * topk..(layer + 1) * rows * topk] {
            if let Some(count) = counts.get_mut(usize::from(expert)) { *count += 1; }
        }
        counts.into_iter().map(|n| n.div_ceil(usize::from(geometry.group_rows(layer, rows))) as f64).sum::<f64>()
            * resource.slice_bytes / 1e6
    }).collect();
    let requests: Vec<_> = record.requests.iter().map(|request| {
        let raw = request.evidence.and_then(|e| e.prior());
        let calibrated = raw.map(|confidence| policy.probabilities(record.requests.len(), cuteafd_core::DraftCandidate { id: request.id, confidence }));
        serde_json::json!({"id": request.id, "chosen_rows": policy.selected_rows(request.id), "verified_rows": request.rows,
            "accepted_rows": request.accepted, "source": format!("{:?}", request.source),
            "censor": request.censor.map(|reason| format!("{reason:?}")),
            "raw_confidence": raw, "calibrated_confidence": calibrated})
    }).collect();
    let fit = policy.cost_snapshot(record.shared, record.requests.len());
    let labels: Vec<_> = geometry.classes.iter().map(|class| class.label).collect();
    // The (bucket, regime) the policy used when it selected this round's
    // shape, which differs from the round's own when grammar or admission
    // changed the request set in between.
    let selected = policy.selected_cell().map(|(bucket, shared)| serde_json::json!({"bucket": bucket, "shared": shared}));
    serde_json::json!({"type": "draft_round", "round": policy.stats().rounds,
        "shared": record.shared, "requests": requests, "rows": rows, "routes": routes,
        "route_layout": [geometry.layers.len(), rows, topk],
        "geometry": geometry.layers.iter().map(|l| serde_json::json!({"class": l.class,
            "slice_bytes": l.slice_bytes, "group_rows": l.group_rows, "timed": l.timed})).collect::<Vec<_>>(),
        "group_rows_by_rows": geometry.group_rows_by_rows,
        "executed_group_rows": (0..geometry.layers.len()).map(|layer| geometry.group_rows(layer, rows)).collect::<Vec<_>>(),
        "megabytes": megabytes, "layer_us": record.layer_us,
        "total_us": record.times.total_us, "draft_us": record.times.draft_us,
        "draft_width": record.draft.map(|d| d.width), "drafted_requests": record.draft.and_then(|d| d.requests),
        "predicted_us": record.predicted, "bucket": cuteafd_core::draft_concurrency_bucket(record.requests.len()),
        "selected": selected, "length_rule": if policy.draft_mode() != cuteafd_core::DraftMode::Shared {
            policy.draft_mode().label() } else if policy.survival_lengths() { "survival" } else { "priced" },
        "cost": fit.to_json(&labels)})
}

/// What a lane carries from its draft pass and length selection to the
/// round's observation.
#[derive(Clone, Debug)]
pub(crate) struct LaneRound {
    /// Width drafted this round (0: no draft pass ran).
    pub width: usize,
    /// Actual sequences passed to the drafter, before grammar truncation.
    pub drafted_requests: usize,
    /// The last selection saw an active peer lane.
    pub shared: bool,
    /// The policy's prediction for the selected shape, awaiting observation.
    pub predicted: Option<f64>,
    /// Route storage reused round to round.
    pub routes: RoundRoutes,
}

impl LaneRound {
    pub fn new(layers: usize, topk: usize) -> Self {
        Self { width: 0, drafted_requests: 0, shared: false, predicted: None, routes: RoundRoutes::new(layers, topk) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routes(rows: usize) -> RoundRoutes {
        let mut routes = RoundRoutes::new(2, 2);
        routes.fill(&[vec![[1u32, 2]; rows], vec![[3u32, 4]; rows]], rows, u32::MAX).unwrap();
        routes
    }
    fn request(id: u64, rows: usize, accepted: usize, source: DraftSource, evidence: Option<Evidence<'_>>,
        censor: Option<Censor>) -> RoundRequest<'_> {
        RoundRequest { id, rows, accepted, source, evidence, censor }
    }
    fn record<'a>(requests: &'a [RoundRequest<'a>], routes: &'a RoundRoutes, draft: Option<DraftPass>)
        -> RoundRecord<'a> {
        RoundRecord { shared: true, requests, routes, layer_us: &[None, Some(10.), Some(99.)],
            times: RoundTimes { total_us: 5_000, draft_us: Some(1_200) }, draft, predicted: Some(4_000.) }
    }

    #[test]
    fn neural_rounds_keep_their_draft_time_and_prediction() {
        let p = [0.9, 0.8, 0.7];
        let requests = [request(1, 4, 2, DraftSource::Neural, Some(Evidence::Head(&p)), None),
            request(2, 1, 1, DraftSource::Undrafted, None, None)];
        let routes = routes(5);
        let built = build(&record(&requests, &routes, Some(DraftPass { width: 7, requests: Some(1) })), 5).unwrap();
        let view = built.view();
        assert_eq!((view.total_us, view.draft_us, view.wide, view.predicted_us), (5_000., 1_200., true, Some(4_000.)));
        assert_eq!(view.drafted_requests, Some(1));
        assert_eq!(view.requests.len(), 2);
        assert_eq!(view.layer_us, [None, Some(10.)]);
        assert_eq!((view.requests[0].confidence, view.requests[1].confidence), (Some(&p[..]), None));
        assert_eq!(view.routes.len(), 2 * 5 * 2);
        // No draft pass: no draft sample, whatever the clock says.
        let built = build(&record(&requests, &routes, None), 5).unwrap();
        assert!(built.view().draft_us.is_nan() && !built.view().wide && built.view().total_us == 5_000.);
    }

    #[test]
    fn copies_teach_costs_only_and_take_the_draft_out_of_the_round() {
        let p = [0.9, 0.8];
        let requests = [request(1, 3, 2, DraftSource::Neural, Some(Evidence::Head(&p)), None),
            request(2, 4, 4, DraftSource::Copy, Some(Evidence::Head(&p)), None)];
        let routes = routes(7);
        let built = build(&record(&requests, &routes, Some(DraftPass { width: 5, requests: None })), 5).unwrap();
        let view = built.view();
        assert_eq!((view.total_us, view.predicted_us, view.wide), (3_800., None, false));
        assert!(view.draft_us.is_nan());
        assert_eq!((view.requests[0].confidence.is_some(), view.requests[1].confidence), (true, None));
    }

    #[test]
    fn mixed_passes_with_explicit_extent_fit_the_actual_drafter_batch() {
        let p = [0.9, 0.8];
        let requests = [request(1, 3, 2, DraftSource::Neural, Some(Evidence::Head(&p)), None),
            request(2, 4, 4, DraftSource::Copy, Some(Evidence::Head(&p)), None),
            request(3, 1, 1, DraftSource::Undrafted, None, None)];
        let routes = routes(8);
        let built = build(&record(&requests, &routes, Some(DraftPass { width: 5, requests: Some(1) })), 5).unwrap();
        let view = built.view();
        assert_eq!((view.total_us, view.draft_us, view.drafted_requests), (5_000., 1_200., Some(1)));
        assert_eq!(view.requests.len(), 3);
        assert_eq!(view.predicted_us, None);
        assert_eq!(view.requests[1].confidence, None);
    }

    #[test]
    fn censored_requests_offer_only_their_accepted_prefix() {
        let p = [0.9, 0.8, 0.7, 0.6];
        let requests = [request(1, 5, 3, DraftSource::Neural, Some(Evidence::Head(&p)), Some(Censor::Eos)),
            request(2, 5, 1, DraftSource::Neural, Some(Evidence::Head(&p)), Some(Censor::OutputLimit)),
            request(3, 5, 3, DraftSource::Neural, Some(Evidence::Head(&p)), None),
            request(4, 2, 1, DraftSource::Neural, Some(Evidence::History(&p[..1])), None)];
        let routes = routes(17);
        let built = build(&record(&requests, &routes, Some(DraftPass { width: 5, requests: None })), 5).unwrap();
        let confidence: Vec<_> = built.view().requests.iter().map(|r| r.confidence.map(<[f64]>::len)).collect();
        assert_eq!(confidence, [Some(2), Some(0), Some(4), Some(1)]);
    }

    #[test]
    fn pooled_history_trains_platt_only_on_reached_uncensored_positions() {
        use cuteafd_core::{LayerResource, PolicyGeometry, PolicySeed, ResourceClass};
        let geometry = PolicyGeometry { layers: vec![LayerResource {
            class: Some(0), slice_bytes: 1e6, group_rows: 16, timed: true }; 2],
            experts: 8, topk: 2, max_requests: 16, max_positions: 5, widths: vec![5],
            classes: vec![ResourceClass::rtx()], group_rows_by_rows: Vec::new(), regimes: 1 };
        let mut policy = DraftPolicy::seeded(geometry, PolicySeed::weak(5)).unwrap();
        let prior = [0.75; 4];
        let requests = [
            request(1, 5, 3, DraftSource::Neural, Some(Evidence::History(&prior)), Some(Censor::Eos)),
            request(2, 5, 1, DraftSource::Neural, Some(Evidence::History(&prior)), Some(Censor::Cancelled)),
            request(3, 5, 3, DraftSource::Neural, Some(Evidence::History(&prior)), None),
            request(4, 3, 3, DraftSource::Copy, Some(Evidence::History(&prior)), None),
        ];
        let routes = routes(18);
        let before = policy.calibration(requests.len());
        let c1_before = policy.calibration(1);
        let mut round = record(&requests, &routes, Some(DraftPass { width: 5, requests: Some(3) }));
        round.predicted = None;
        observe(&mut policy, &round).unwrap();
        assert_eq!(policy.stats().position_reached[..4], [2, 2, 1, 0]);
        assert_eq!(policy.stats().position_accepted[..4], [2, 2, 0, 0]);
        assert_ne!(policy.calibration(requests.len())[0], before[0]);
        assert_eq!(policy.calibration(requests.len())[3], before[3]);
        assert_eq!(policy.calibration(1), c1_before);
    }

    #[test]
    fn generic_binding_transcript_keeps_undrafted_anchors_and_complete_resources() {
        use cuteafd_core::{DraftCandidate, LayerResource, PolicyGeometry, PolicySeed, ResourceClass};
        let geometry = PolicyGeometry { layers: vec![LayerResource {
            class: Some(0), slice_bytes: 1e6, group_rows: 16, timed: true }; 2],
            experts: 8, topk: 2, max_requests: 16, max_positions: 5, widths: vec![5],
            classes: vec![ResourceClass::rtx()], group_rows_by_rows: Vec::new(), regimes: 1 };
        let mut policy = DraftPolicy::new(geometry, PolicySeed::weak(5), false).unwrap();
        let p = [0.9; 5];
        for _ in 0..160 {
            let candidates = [DraftCandidate { id: 1, confidence: &p }, DraftCandidate { id: 2, confidence: &[] }];
            let selection = policy.select_drafted_within(false, &candidates, 5, 8, 1).unwrap();
            let lengths = selection.as_ref().map_or(vec![5, 0], |s| s.lengths.clone());
            assert_eq!(lengths[1], 0);
            let requests = [request(1, lengths[0] + 1, lengths[0] + 1, DraftSource::Neural,
                Some(Evidence::Head(&p)), None), request(2, 1, 1, DraftSource::Undrafted, None, None)];
            let routes = routes(lengths[0] + 2);
            observe(&mut policy, &RoundRecord { shared: false, requests: &requests, routes: &routes,
                layer_us: &[Some(100.), Some(100.)], times: RoundTimes { total_us: 800, draft_us: Some(200) },
                draft: Some(DraftPass { width: 5, requests: Some(1) }),
                predicted: selection.as_ref().map(|s| s.predicted_us) }).unwrap();
        }
        assert!(policy.engaged(false, 2));
        let fit = policy.cost_snapshot(false, 2);
        // Each round teaches the class (one per round, summed over its
        // layers), the residual and the draft pass once.
        assert_eq!(fit.layers[0][3], 160.);
        assert!(fit.round[3] > 0.);
        assert_eq!(fit.draft[4], 160.);
        assert_eq!(policy.stats().emitted_requests, 320);
    }

    #[test]
    fn sampled_layer_times_preserve_routes_and_round_observations() {
        use cuteafd_core::{LayerResource, PolicyGeometry, PolicySeed, ResourceClass};
        let geometry = PolicyGeometry { layers: vec![LayerResource {
            class: Some(0), slice_bytes: 1e6, group_rows: 16, timed: true }; 2],
            experts: 8, topk: 2, max_requests: 16, max_positions: 5, widths: vec![5],
            classes: vec![ResourceClass::rtx()], group_rows_by_rows: Vec::new(), regimes: 1 };
        let mut policy = DraftPolicy::new(geometry, PolicySeed::weak(5), false).unwrap();
        let p = [0.9; 5];
        let requests = [request(1, 6, 4, DraftSource::Neural, Some(Evidence::Head(&p)), None)];
        let routes = routes(6);
        for round in 0..160 {
            let layer_us = if round % 8 == 7 {
                [Some(100.), Some(100.)]
            } else { [None, None] };
            let record = RoundRecord { shared: false, requests: &requests, routes: &routes,
                layer_us: &layer_us, times: RoundTimes { total_us: 800, draft_us: Some(200) },
                draft: Some(DraftPass { width: 5, requests: Some(1) }), predicted: None };
            let observation = build(&record, 5).unwrap();
            assert_eq!(observation.view().routes, routes.flat());
            assert_eq!(observation.view().total_us, 800.);
            assert_eq!(observation.view().layer_us, layer_us);
            observe(&mut policy, &record).unwrap();
        }
        let fit = policy.cost_snapshot(false, 1);
        assert_eq!(fit.layers[0][3], 20.);
        // Class sums are sampled, but untimed rounds still teach the residual.
        assert!(fit.round[3] > 0.);
        assert_eq!(policy.stats().rounds, 160);
    }

    #[test]
    fn sampled_class_fit_replay_reports_warmth_and_decisions() {
        use cuteafd_core::{DraftAction, DraftCandidate, LayerResource, PolicyGeometry, PolicySeed, ResourceClass};
        let geometry = PolicyGeometry { layers: (0..2).map(|class| LayerResource {
            class: Some(class), slice_bytes: 1e6, group_rows: 16, timed: true }).collect(),
            experts: 8, topk: 2, max_requests: 16, max_positions: 5, widths: vec![5],
            classes: vec![ResourceClass::rtx(), ResourceClass::spark()], group_rows_by_rows: Vec::new(), regimes: 1 };
        let every = DraftPolicy::new(geometry, PolicySeed::weak(5), false).unwrap();
        let mut policies = [every.clone(), every];
        let mut first_class = [[None; 2]; 2];
        let mut first_warm = [None; 2];
        let mut first_engaged = [None; 2];
        let mut decisions = [Vec::new(), Vec::new()];
        let p = [0.95, 0.9, 0.8, 0.65, 0.5];
        for round in 0..160 {
            let rows = 2 + round % 5;
            let requests: Vec<_> = (1..=16).map(|id| request(id, rows, rows.min(4), DraftSource::Neural,
                Some(Evidence::Head(&p)), None)).collect();
            let candidates: Vec<_> = (1..=16).map(|id| DraftCandidate { id, confidence: &p }).collect();
            let routes = routes(rows * 16);
            // Identical exogenous C16 transcript; a midpoint cost change tests stale fits.
            let class_us = if round < 80 { [400., 800.] } else { [1200., 2400.] };
            for (arm, policy) in policies.iter_mut().enumerate() {
                let selected = policy.select_drafted_within(false, &candidates, 5, 96, 16).unwrap();
                let lengths = selected.as_ref().map_or(vec![5; 16], |s| s.lengths.clone());
                let action_requests: Vec<_> = (1..=16).map(|id| (id, 5)).collect();
                let actions = [DraftAction { width: 0, draft_us: 0. }, DraftAction { width: 5, draft_us: 3000. }];
                let width = policy.choose_action(false, &action_requests, &actions, 96).unwrap();
                decisions[arm].push((width, lengths));
                let layer_us = if arm == 0 || round % 8 == 7 {
                    class_us.map(Some)
                } else { [None, None] };
                observe(policy, &RoundRecord { shared: false, requests: &requests, routes: &routes,
                    layer_us: &layer_us, times: RoundTimes { total_us: (class_us.iter().sum::<f64>() + 5000.) as u64,
                        draft_us: Some(3000) }, draft: Some(DraftPass { width: 5, requests: Some(16) }),
                    predicted: None }).unwrap();
                let fit = policy.cost_snapshot(false, 16);
                for class in 0..2 {
                    if fit.layers[class][3] >= 4. { first_class[arm][class].get_or_insert(round + 1); }
                }
                if policy.warm(false, 16) { first_warm[arm].get_or_insert(round + 1); }
                if policy.engaged(false, 16) { first_engaged[arm].get_or_insert(round + 1); }
                if [31, 79, 87, 159].contains(&round) {
                    println!("cadence_replay arm={arm} round={} class_fits={:?} warm={} engaged={}",
                        round + 1, fit.layers, policy.warm(false, 16), policy.engaged(false, 16));
                }
            }
        }
        let differences = decisions[0].iter().zip(&decisions[1]).filter(|(a, b)| a != b).count();
        println!("cadence_replay first_class={first_class:?} first_warm={first_warm:?} first_engaged={first_engaged:?} differing_decisions={differences}");
        assert_eq!(first_class, [[Some(4); 2], [Some(32); 2]]);
        assert_eq!(first_warm, [Some(12), Some(32)]);
        assert_eq!(policies[0].cost_snapshot(false, 16).layers[0][3], 160.);
        assert_eq!(policies[1].cost_snapshot(false, 16).layers[0][3], 20.);
    }

    #[test]
    fn round_trace_captures_head_routes_grouped_bytes_and_calibrated_probabilities() {
        use cuteafd_core::{DraftCandidate, LayerResource, PolicyGeometry, PolicySeed, ResourceClass};
        let geometry = PolicyGeometry { layers: vec![LayerResource {
            class: Some(0), slice_bytes: 1e6, group_rows: 2, timed: true }; 2],
            experts: 8, topk: 2, max_requests: 4, max_positions: 5, widths: vec![5],
            classes: vec![ResourceClass::rtx()], group_rows_by_rows: Vec::new(), regimes: 1 };
        let mut policy = DraftPolicy::new(geometry, PolicySeed::weak(5), false).unwrap();
        let p = [0.9; 5];
        let candidates = [DraftCandidate { id: 1, confidence: &p }];
        for _ in 0..160 {
            let requests = [request(1, 6, 4, DraftSource::Neural, Some(Evidence::Head(&p)), None)];
            let routes = routes(6);
            observe(&mut policy, &RoundRecord { shared: false, requests: &requests, routes: &routes,
                layer_us: &[Some(100.), Some(100.)], times: RoundTimes { total_us: 800, draft_us: Some(200) },
                draft: Some(DraftPass { width: 5, requests: Some(1) }), predicted: None }).unwrap();
        }
        let selection = policy.select_drafted_within(false, &candidates, 5, 6, 1).unwrap().unwrap();
        let requests = [request(1, 3, 2, DraftSource::Neural, Some(Evidence::Head(&p)), Some(Censor::Eos))];
        let routes = routes(3);
        let value = round_trace_value(&policy, &RoundRecord { shared: false, ..record(&requests, &routes,
            Some(DraftPass { width: 5, requests: Some(1) })) });
        assert_eq!(value["type"], "draft_round");
        assert_eq!(value["route_layout"], serde_json::json!([2, 3, 2]));
        assert_eq!(value["routes"], serde_json::json!(routes.flat()));
        assert_eq!(value["megabytes"], serde_json::json!([4., 4.]));
        let request = &value["requests"][0];
        assert_eq!(request["raw_confidence"], serde_json::json!(p));
        assert_eq!(request["calibrated_confidence"], serde_json::json!(policy.probabilities(candidates.len(), candidates[0])));
        assert_eq!(request["chosen_rows"], selection.lengths[0] + 1);
        assert_eq!(request["verified_rows"], 3);
        assert_eq!(request["accepted_rows"], 2);
        assert_eq!(request["censor"], "Eos");
        assert_eq!(value["cost"]["classes"].as_array().unwrap().len(), 1);
        assert_eq!(value["cost"]["classes"][0]["class"], "local");
        assert!(value["cost"]["residual"]["us_per_row"].is_f64());
        assert_eq!(value["bucket"], 0);
        assert_eq!(value["selected"], serde_json::json!({"bucket": 0, "shared": false}));
        assert_eq!(value["length_rule"], "priced");
    }

    #[test]
    fn round_trace_prices_executed_installed_grouping_not_static_capacity() {
        use cuteafd_core::{LayerResource, PolicyGeometry, PolicySeed, ResourceClass};
        let geometry = PolicyGeometry { layers: vec![LayerResource {
            class: Some(0), slice_bytes: 1e6, group_rows: 8, timed: true }; 2],
            experts: 8, topk: 2, max_requests: 16, max_positions: 5, widths: vec![5],
            classes: vec![ResourceClass::rtx()],
            group_rows_by_rows: (0..=16).map(|rows| vec![if rows == 1 { 1 } else { 8 }; 2]).collect(), regimes: 1 };
        let policy = DraftPolicy::seeded(geometry, PolicySeed::weak(5)).unwrap();
        let requests = [request(1, 16, 1, DraftSource::Undrafted, None, None)];
        let wide_routes = routes(16);
        let value = round_trace_value(&policy, &record(&requests, &wide_routes, None));
        assert_eq!(value["geometry"][0]["group_rows"], 8);
        assert_eq!(value["executed_group_rows"], serde_json::json!([8, 8]));
        // Two experts each appear sixteen times: two M8 reads each, not one M16 read.
        assert_eq!(value["megabytes"], serde_json::json!([4., 4.]));
        let requests = [request(1, 1, 1, DraftSource::Undrafted, None, None)];
        let routes = routes(1);
        let value = round_trace_value(&policy, &record(&requests, &routes, None));
        assert_eq!(value["executed_group_rows"], serde_json::json!([1, 1]));
        assert_eq!(value["megabytes"], serde_json::json!([2., 2.]));
    }

    #[test]
    fn replay_decisions_keep_unverified_priors_and_action_participants() {
        use cuteafd_core::{DraftAction, DraftCandidate, LayerResource, PolicyGeometry, PolicySeed, ResourceClass};
        let geometry = PolicyGeometry { layers: vec![LayerResource {
            class: Some(0), slice_bytes: 1e6, group_rows: 2, timed: true }; 2],
            experts: 8, topk: 2, max_requests: 4, max_positions: 5, widths: vec![3, 5],
            classes: vec![ResourceClass::rtx()], group_rows_by_rows: Vec::new(), regimes: 1 };
        let policy = DraftPolicy::new(geometry, PolicySeed::weak(5), false).unwrap();
        let actions = [DraftAction { width: 0, draft_us: 0. }, DraftAction { width: 3, draft_us: 2500. },
            DraftAction { width: 5, draft_us: 3000. }];
        let masks: [&[usize]; 3] = [&[0, 0], &[3, 3], &[5, 0]];
        let action = action_trace_value(&policy, false, &[(1, 5), (2, 5)], &actions, Some(&masks),
            8, None, 1);
        assert_eq!(action["type"], "draft_action");
        assert_eq!(action["actions"][1]["rooms"], serde_json::json!([3, 3]));
        assert_eq!(action["actions"][2]["rooms"], serde_json::json!([5, 0]));
        assert_eq!(action["actions"][2]["draft_us"], 3000.);
        assert!(action["chosen"].is_null());
        assert_eq!(action["executed"], 1);
        assert_eq!(action["cold"], true);
        let p = [0.9, 0.8, 0.7, 0.6, 0.5];
        let candidates = [DraftCandidate { id: 1, confidence: &p }, DraftCandidate { id: 2, confidence: &[] }];
        let selection = selection_trace_value(&policy, &candidates, 5, 1, 8, &[1, 0]);
        assert_eq!(selection["requests"][0]["raw_confidence"], serde_json::json!(p));
        assert_eq!(selection["requests"][0]["selected_drafts"], 1);
        assert_eq!(selection["requests"][0]["calibrated_confidence"].as_array().unwrap().len(), 5);
        assert_eq!(selection["requests"][1]["raw_confidence"], serde_json::json!([]));
        assert_eq!(selection["drafted_requests"], 1);
    }

    #[test]
    fn rejects_routes_that_do_not_cover_the_rows() {
        let requests = [request(1, 3, 1, DraftSource::Neural, None, None)];
        let short = routes(2);
        assert_eq!(build(&record(&requests, &short, None), 5).unwrap_err(),
            ObserveError::Rows { routes: 2, requests: 3 });
        let mut partial = RoundRoutes::new(2, 2);
        partial.begin(3);
        partial.push(0, &[0; 6], u32::MAX).unwrap();
        assert_eq!(build(&record(&requests, &partial, None), 5).unwrap_err(),
            ObserveError::Routes(RouteError::Missing(1)));
    }
}
