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
//!   shape), and a draft pass beside a copy covered only some requests, so it
//!   gives no draft-cost sample and its time leaves the round total.
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
    wide: bool,
    predicted_us: Option<f64>,
}

impl Observation<'_> {
    pub fn view(&self) -> DraftRoundObservation<'_> {
        DraftRoundObservation { shared: self.shared, requests: &self.requests, routes: self.routes,
            layer_us: &self.layer_us, total_us: self.total_us, predicted_us: self.predicted_us,
            draft_us: self.draft_us, wide: self.wide }
    }
}

/// Build one round's observation. `native_width` is the drafter's native
/// block (`policy.widths()[0]`); a pass wider than it is the wide width.
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
    let (total_us, draft_us) = match (record.draft.is_some(), copied) {
        (true, false) => (total as f64, draft as f64),
        (true, true) => (total.saturating_sub(draft) as f64, f64::NAN),
        (false, _) => (total as f64, f64::NAN),
    };
    Ok(Observation { shared: record.shared, requests, routes: record.routes.flat(), layer_us, total_us, draft_us,
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
    Ok(policy.observe(observation.view())?)
}

/// What a lane carries from its draft pass and length selection to the
/// round's observation.
#[derive(Clone, Debug)]
pub(crate) struct LaneRound {
    /// Width drafted this round (0: no draft pass ran).
    pub width: usize,
    /// The last selection saw an active peer lane.
    pub shared: bool,
    /// The policy's prediction for the selected shape, awaiting observation.
    pub predicted: Option<f64>,
    /// Route storage reused round to round.
    pub routes: RoundRoutes,
}

impl LaneRound {
    pub fn new(layers: usize, topk: usize) -> Self {
        Self { width: 0, shared: false, predicted: None, routes: RoundRoutes::new(layers, topk) }
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
        let built = build(&record(&requests, &routes, Some(DraftPass { width: 7 })), 5).unwrap();
        let view = built.view();
        assert_eq!((view.total_us, view.draft_us, view.wide, view.predicted_us), (5_000., 1_200., true, Some(4_000.)));
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
        let built = build(&record(&requests, &routes, Some(DraftPass { width: 5 })), 5).unwrap();
        let view = built.view();
        assert_eq!((view.total_us, view.predicted_us, view.wide), (3_800., None, false));
        assert!(view.draft_us.is_nan());
        assert_eq!((view.requests[0].confidence.is_some(), view.requests[1].confidence), (true, None));
    }

    #[test]
    fn censored_requests_offer_only_their_accepted_prefix() {
        let p = [0.9, 0.8, 0.7, 0.6];
        let requests = [request(1, 5, 3, DraftSource::Neural, Some(Evidence::Head(&p)), Some(Censor::Eos)),
            request(2, 5, 1, DraftSource::Neural, Some(Evidence::Head(&p)), Some(Censor::OutputLimit)),
            request(3, 5, 3, DraftSource::Neural, Some(Evidence::Head(&p)), None),
            request(4, 2, 1, DraftSource::Neural, Some(Evidence::History), None)];
        let routes = routes(17);
        let built = build(&record(&requests, &routes, Some(DraftPass { width: 5 })), 5).unwrap();
        let confidence: Vec<_> = built.view().requests.iter().map(|r| r.confidence.map(<[f64]>::len)).collect();
        assert_eq!(confidence, [Some(2), Some(0), Some(4), None]);
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
