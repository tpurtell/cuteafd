//! Opt-in simple draft policy (`DRAFT_POLICY=simple|simple-derived`), an
//! experiment beside the shared resource-priced policy. No cost model, no
//! concurrency buckets: only measured cost per committed token and one global
//! confidence calibration per draft position.
//!
//! **Samples.** After each round of wall time `T` (draft, verify and
//! everything else) over `V` verified rows, request `r` with `v_r` rows that
//! committed `a_r >= 1` tokens (bonus included) took `t_r = T * v_r / V` and
//! achieved `t_r / a_r` per token.
//!
//! **Cost per token.** Per request, a recent-weighted ratio of sums
//! (`REQUEST_DECAY` per round): `c_r = (sum t + PRIOR_TOKENS * c_g) / (sum a +
//! PRIOR_TOKENS)`. The global `c_g` is the same accounting pooled over every
//! request (`GLOBAL_DECAY`), and seeds a new request.
//!
//! **Confidence.** One online Platt fit per draft position, trained on the
//! positions the verifier reached.
//!
//! **Relative waste.** `rho_r = c_r / c_batch`, clamped to [0.5, 2], where
//! `c_batch` pools the windows of the round's requests (same prior). Within
//! a round it is the request's verified rows per committed token relative to
//! the batch's; at C1 it is exactly one.
//!
//! **Lengths.** Each request has a threshold `theta_r = theta_base * rho_r`
//! (bounded to [`THRESHOLD_FLOOR`], [`THRESHOLD_CEILING`]): a request that
//! has been wasting rows needs more confidence to get them.
//! `simple`: `theta_base = 0.5` against the calibrated conditional acceptance
//! of each position, stopping at the first failure. `simple-derived`:
//! `theta_base = r / c_batch` against the joint probability that `k` is
//! reached and accepted, where `r` is the marginal round time of one more
//! verified row (the slope of `T` against `V`): adding a row raises batch
//! tokens per second iff `p_joint * T > r * E`, i.e. `p_joint > r / c`.
//! Without enough row variation to fit `r`, it uses the `simple` rule.
//!
//! **Scarce rows.** When the passing positions exceed the row budget, both
//! arms keep the positions with the largest `p_joint / theta_r`. For
//! `simple-derived` that is a uniform clearing price on `r`. `p_joint` never
//! rises along a request, so the kept set stays a prefix.
use super::{DraftCandidate, DraftRoundObservation, DraftSelection, Platt};
use std::collections::BTreeMap;

/// Which length rule a [`DraftPolicy`](super::DraftPolicy) follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftMode {
    /// The resource-priced shared policy (default).
    Shared,
    /// Conditional calibrated acceptance above one half.
    Simple,
    /// Joint acceptance above the measured row price over cost per token.
    SimpleDerived,
}

impl DraftMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "" | "shared" => Some(Self::Shared),
            "simple" => Some(Self::Simple),
            "simple-derived" => Some(Self::SimpleDerived),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self { Self::Shared => "shared", Self::Simple => "simple", Self::SimpleDerived => "simple-derived" }
    }
}

/// Per-round decay of a request's window: weights 1, 0.6, 0.36, ... (about
/// 2.5 rounds).
const REQUEST_DECAY: f64 = 0.6;
/// Per-round decay of the global accounting and the pooled row slope (about
/// 20 rounds).
const GLOBAL_DECAY: f64 = 0.95;
/// Tokens of global prior in each request's cost per token.
const PRIOR_TOKENS: f64 = 2.;
/// Ridge strength, in rows², pulling the short-window row slope to the pooled one.
const SLOPE_PRIOR: f64 = 4.;
/// Least pooled row variance (rows², weighted) for a usable slope.
const MIN_ROW_VARIANCE: f64 = 4.;
/// The `simple` arm's conditional threshold.
pub const SIMPLE_THRESHOLD: f64 = 0.5;
/// Bounds of every request's threshold (and of the derived base).
pub const THRESHOLD_FLOOR: f64 = 0.1;
pub const THRESHOLD_CEILING: f64 = 0.9;
/// Bounds of a request's relative waste.
pub const WASTE_FLOOR: f64 = 0.5;
pub const WASTE_CEILING: f64 = 2.;
/// Consecutive selections without a draft before a request verifies one,
/// keeping its calibration evidence alive.
const EXPLORE_SKIPS: u64 = 8;

/// Decayed weighted sums for a least-squares line `T = a + r * V`.
#[derive(Clone, Copy, Debug, Default)]
struct Line { w: f64, x: f64, y: f64, xx: f64, xy: f64 }

impl Line {
    fn add(&mut self, decay: f64, x: f64, y: f64) {
        for sum in [&mut self.w, &mut self.x, &mut self.y, &mut self.xx, &mut self.xy] { *sum *= decay; }
        self.w += 1.;
        self.x += x;
        self.y += y;
        self.xx += x * x;
        self.xy += x * y;
    }
    /// Centered (Sxx, Sxy).
    fn centered(&self) -> (f64, f64) {
        if self.w <= 0. { return (0., 0.); }
        (self.xx - self.x * self.x / self.w, self.xy - self.x * self.y / self.w)
    }
    fn slope(&self) -> Option<f64> {
        let (sxx, sxy) = self.centered();
        (sxx >= MIN_ROW_VARIANCE).then(|| sxy / sxx)
    }
    fn predict(&self, x: f64, slope: f64) -> Option<f64> {
        (self.w > 0.).then(|| self.y / self.w + slope * (x - self.x / self.w))
    }
}

/// The row price of one selection.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RowPrice {
    /// Unregularized short-window slope, if the window varied enough.
    pub short: Option<f64>,
    /// Pooled slope, if the pooled window varied enough.
    pub pooled: Option<f64>,
    /// Short window shrunk to the pooled slope, nonnegative; `None` falls back
    /// to the one-half conditional rule.
    pub used: Option<f64>,
}

/// One request's decision, kept for the round's trace.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleDecision {
    pub cost_us: Option<f64>,
    /// The round's pooled cost per token.
    pub batch_cost_us: Option<f64>,
    /// `c_r / c_batch`, clamped.
    pub waste: f64,
    /// Before the waste scaling: 0.5, or `r / c_batch` for the derived rule.
    pub base: f64,
    /// `base * waste`, bounded.
    pub threshold: f64,
    /// The threshold applies to the joint probability (derived) instead of
    /// the conditional one.
    pub joint_rule: bool,
    pub conditional: Vec<f64>,
    pub joint: Vec<f64>,
    /// Lengths the threshold passed, then what the budget kept.
    pub passing: usize,
    pub chosen: usize,
    pub explore: bool,
    pub scarce: bool,
    pub budget: usize,
    pub row_price: RowPrice,
}

/// One request's sample from an observed round.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleSample {
    pub id: u64,
    pub rows: usize,
    pub accepted: usize,
    pub share_us: f64,
    pub sample_us: f64,
    /// The request's windowed cost per token after this sample.
    pub cost_us: f64,
    pub decision: Option<SimpleDecision>,
}

/// The latest observed round, for traces.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleRound {
    pub total_us: f64,
    pub rows: usize,
    pub global_cost_us: Option<f64>,
    pub row_price: RowPrice,
    pub requests: Vec<SimpleSample>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Window { time: f64, tokens: f64 }

#[derive(Clone, Debug)]
pub(crate) struct SimpleDraft {
    mode: DraftMode,
    calibration: Vec<Platt>,
    global: Window,
    short: Line,
    pooled: Line,
    requests: BTreeMap<u64, Window>,
    skipped: BTreeMap<u64, u64>,
    decisions: BTreeMap<u64, SimpleDecision>,
    last: Option<SimpleRound>,
}

impl SimpleDraft {
    pub fn new(mode: DraftMode, positions: usize) -> Self {
        Self { mode, calibration: vec![Platt::new(); positions], global: Window::default(), short: Line::default(),
            pooled: Line::default(), requests: BTreeMap::new(), skipped: BTreeMap::new(), decisions: BTreeMap::new(),
            last: None }
    }
    pub fn mode(&self) -> DraftMode { self.mode }
    pub fn calibration(&self) -> Vec<(f64, f64)> {
        self.calibration.iter().map(|platt| (platt.theta[0], platt.theta[1])).collect()
    }
    pub fn last_round(&self) -> Option<&SimpleRound> { self.last.as_ref() }
    pub fn release(&mut self, id: u64) {
        self.requests.remove(&id);
        self.skipped.remove(&id);
        self.decisions.remove(&id);
    }
    pub fn global_cost(&self) -> Option<f64> {
        (self.global.tokens > 0.).then(|| self.global.time / self.global.tokens)
    }
    /// A request's windowed (time, tokens), seeded by the global cost.
    fn window(&self, id: u64) -> Option<(f64, f64)> {
        let global = self.global_cost()?;
        let own = self.requests.get(&id).copied().unwrap_or_default();
        Some((own.time + PRIOR_TOKENS * global, own.tokens + PRIOR_TOKENS))
    }
    /// A request's windowed cost per token, seeded by the global one.
    pub fn request_cost(&self, id: u64) -> Option<f64> {
        self.window(id).map(|(time, tokens)| time / tokens)
    }
    /// The pooled windowed cost per token of a set of requests.
    pub fn batch_cost(&self, ids: impl IntoIterator<Item = u64>) -> Option<f64> {
        let (time, tokens) = ids.into_iter().filter_map(|id| self.window(id))
            .fold((0., 0.), |(t, a), (time, tokens)| (t + time, a + tokens));
        (tokens > 0.).then(|| time / tokens)
    }
    pub fn row_price(&self) -> RowPrice {
        let pooled = self.pooled.slope();
        let (sxx, sxy) = self.short.centered();
        RowPrice { short: self.short.slope(), pooled,
            used: pooled.map(|pooled| ((sxy + SLOPE_PRIOR * pooled) / (sxx + SLOPE_PRIOR)).max(0.)) }
    }
    /// Calibrated conditional acceptance per position.
    pub fn probabilities(&self, confidence: &[f64]) -> Vec<f64> {
        confidence.iter().zip(&self.calibration).map(|(&p, platt)| platt.apply(p)).collect()
    }

    /// Lengths under `max_rows` verifier rows (anchors included).
    pub fn select(&mut self, candidates: &[DraftCandidate<'_>], max_rows: usize) -> DraftSelection {
        let price = self.row_price();
        let batch_cost_us = self.batch_cost(candidates.iter().map(|c| c.id));
        let derived = match (self.mode, price.used, batch_cost_us) {
            (DraftMode::SimpleDerived, Some(r), Some(c)) if c > 0. => Some((r / c).clamp(THRESHOLD_FLOOR, THRESHOLD_CEILING)),
            _ => None,
        };
        let mut decisions: Vec<SimpleDecision> = candidates.iter().map(|candidate| {
            let conditional = self.probabilities(candidate.confidence);
            let mut product = 1.;
            let joint: Vec<f64> = conditional.iter().map(|p| { product *= p; product }).collect();
            let cost_us = self.request_cost(candidate.id);
            let waste = match (cost_us, batch_cost_us) {
                (Some(c), Some(batch)) if batch > 0. => (c / batch).clamp(WASTE_FLOOR, WASTE_CEILING),
                _ => 1.,
            };
            let base = derived.unwrap_or(SIMPLE_THRESHOLD);
            let threshold = (base * waste).clamp(THRESHOLD_FLOOR, THRESHOLD_CEILING);
            let tested = if derived.is_some() { &joint } else { &conditional };
            let mut passing = tested.iter().take_while(|&&p| p > threshold).count();
            let explore = passing == 0 && !conditional.is_empty()
                && self.skipped.get(&candidate.id).copied().unwrap_or(0) >= EXPLORE_SKIPS;
            if explore { passing = 1; }
            SimpleDecision { cost_us, batch_cost_us, waste, base, threshold, joint_rule: derived.is_some(),
                conditional, joint, passing, chosen: passing, explore, scarce: false, budget: 0, row_price: price }
        }).collect();
        let budget = max_rows.saturating_sub(candidates.len());
        let wanted: usize = decisions.iter().map(|d| d.passing).sum();
        if wanted > budget {
            // Uniform clearing price: keep the positions most likely accepted
            // relative to their request's threshold.
            let mut positions: Vec<(f64, usize, usize)> = decisions.iter().enumerate().flat_map(|(index, d)| {
                (0..d.passing).map(move |k| (d.joint[k] / d.threshold, k, index))
            }).collect();
            positions.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            for decision in &mut decisions { decision.chosen = 0; decision.scarce = true; }
            for &(_, _, index) in positions.iter().take(budget) { decisions[index].chosen += 1; }
        }
        let lengths: Vec<usize> = decisions.iter().map(|d| d.chosen).collect();
        let rows = candidates.len() + lengths.iter().sum::<usize>();
        let expected = candidates.len() as f64
            + decisions.iter().map(|d| d.joint[..d.chosen].iter().sum::<f64>()).sum::<f64>();
        let predicted_us = self.pooled.predict(rows as f64, price.used.unwrap_or(0.)).unwrap_or(0.).max(0.);
        for (candidate, mut decision) in candidates.iter().zip(decisions) {
            if !decision.conditional.is_empty() {
                if decision.chosen == 0 {
                    *self.skipped.entry(candidate.id).or_default() += 1;
                } else {
                    self.skipped.remove(&candidate.id);
                }
            }
            decision.budget = budget;
            self.decisions.insert(candidate.id, decision);
        }
        DraftSelection { lengths, expected_tokens: expected, predicted_us, evaluated: 1 }
    }

    /// Learn costs and calibration from one completed round.
    pub fn observe(&mut self, round: &DraftRoundObservation<'_>) {
        let rows: usize = round.requests.iter().map(|r| r.rows).sum();
        let total = round.total_us;
        let timed = total.is_finite() && total > 0. && rows > 0;
        let mut samples = Vec::with_capacity(round.requests.len());
        for request in round.requests {
            let decision = self.decisions.remove(&request.id);
            if let Some(confidence) = request.confidence.filter(|_| request.rows > 1) {
                let drafts = request.rows - 1;
                let accepted = request.accepted - 1;
                // Only positions whose predecessors were all accepted were judged.
                let reached = drafts.min(confidence.len()).min(accepted + 1).min(self.calibration.len());
                for (position, platt) in self.calibration.iter_mut().enumerate().take(reached) {
                    platt.observe(confidence[position], f64::from(u8::from(position < accepted)));
                }
            }
            if !timed { continue; }
            let share_us = total * request.rows as f64 / rows as f64;
            let window = self.requests.entry(request.id).or_default();
            window.time = REQUEST_DECAY * window.time + share_us;
            window.tokens = REQUEST_DECAY * window.tokens + request.accepted as f64;
            samples.push((request.id, request.rows, request.accepted, share_us, decision));
        }
        if !timed { return; }
        let tokens: usize = round.requests.iter().map(|r| r.accepted).sum();
        self.global.time = GLOBAL_DECAY * self.global.time + total;
        self.global.tokens = GLOBAL_DECAY * self.global.tokens + tokens as f64;
        self.short.add(REQUEST_DECAY, rows as f64, total);
        self.pooled.add(GLOBAL_DECAY, rows as f64, total);
        let requests = samples.into_iter().map(|(id, rows, accepted, share_us, decision)| SimpleSample {
            id, rows, accepted, share_us, sample_us: share_us / accepted as f64,
            cost_us: self.request_cost(id).unwrap_or(f64::NAN), decision,
        }).collect();
        self.last = Some(SimpleRound { total_us: total, rows, global_cost_us: self.global_cost(),
            row_price: self.row_price(), requests });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObservedDraftRequest;

    fn observe(policy: &mut SimpleDraft, requests: &[ObservedDraftRequest<'_>], total_us: f64) {
        policy.observe(&DraftRoundObservation { shared: false, requests, routes: &[], layer_us: &[],
            total_us, draft_us: f64::NAN, drafted_requests: None, wide: false, predicted_us: None });
    }

    #[test]
    fn modes_parse_and_default_to_shared() {
        assert_eq!(DraftMode::parse(""), Some(DraftMode::Shared));
        assert_eq!(DraftMode::parse("simple-derived"), Some(DraftMode::SimpleDerived));
        assert_eq!(DraftMode::parse("fancy"), None);
        assert_eq!(DraftMode::Simple.label(), "simple");
    }

    #[test]
    fn samples_split_round_time_by_rows_over_committed_tokens() {
        let mut policy = SimpleDraft::new(DraftMode::Simple, 4);
        let requests = [ObservedDraftRequest { id: 1, rows: 3, accepted: 3, confidence: None },
            ObservedDraftRequest { id: 2, rows: 1, accepted: 1, confidence: None }];
        observe(&mut policy, &requests, 8_000.);
        let round = policy.last_round().unwrap();
        // 6000 µs of share over 3 tokens, 2000 µs over 1.
        assert_eq!(round.requests[0].share_us, 6_000.);
        assert_eq!(round.requests[0].sample_us, 2_000.);
        assert_eq!(round.requests[1].sample_us, 2_000.);
        assert_eq!(policy.global_cost(), Some(2_000.));
        // A new request starts at the global cost per token.
        assert_eq!(policy.request_cost(9), Some(2_000.));
    }

    #[test]
    fn simple_verifies_while_conditional_exceeds_half_and_stops_at_first_failure() {
        let mut policy = SimpleDraft::new(DraftMode::Simple, 6);
        let p = [0.9, 0.8, 0.4, 0.9, 0.9];
        let selection = policy.select(&[DraftCandidate { id: 1, confidence: &p }], 64);
        assert_eq!(selection.lengths, [2]);
    }

    #[test]
    fn wasteful_requests_face_a_higher_threshold_and_lose_scarce_rows() {
        let mut policy = SimpleDraft::new(DraftMode::Simple, 6);
        // Request 1 costs 2x per token: its rows save twice the time.
        let one = [ObservedDraftRequest { id: 1, rows: 1, accepted: 1, confidence: None },
            ObservedDraftRequest { id: 2, rows: 1, accepted: 1, confidence: None }];
        for _ in 0..20 { observe(&mut policy, &one, 2_000.); }
        let skewed = [ObservedDraftRequest { id: 1, rows: 2, accepted: 1, confidence: None },
            ObservedDraftRequest { id: 2, rows: 2, accepted: 2, confidence: None }];
        for _ in 0..20 { observe(&mut policy, &skewed, 4_000.); }
        assert!(policy.request_cost(1).unwrap() > 1.5 * policy.request_cost(2).unwrap());
        let p = [0.9, 0.8, 0.62];
        let candidates = [DraftCandidate { id: 1, confidence: &p }, DraftCandidate { id: 2, confidence: &p }];
        let spare = policy.select(&candidates, 64);
        let (one, two) = (policy.decisions[&1].clone(), policy.decisions[&2].clone());
        assert!(one.waste > 1. && two.waste < 1., "{one:?} {two:?}");
        assert!((one.threshold - 0.5 * one.waste).abs() < 1e-12);
        // The wasteful request needs more confidence: 0.62 fails its threshold.
        assert!(one.threshold > 0.62 && two.threshold < 0.62, "{one:?} {two:?}");
        assert_eq!(spare.lengths, [2, 3]);
        // Three rows: ranked by joint over threshold, each request in prefix order.
        let scarce = policy.select(&candidates, 2 + 3);
        assert_eq!(scarce.lengths, [1, 2]);
        let decision = &policy.decisions[&1];
        assert!(decision.scarce && decision.passing == 2 && decision.budget == 3);
    }

    #[test]
    fn derived_threshold_is_row_price_over_cost_per_token() {
        let mut policy = SimpleDraft::new(DraftMode::SimpleDerived, 6);
        // T = 10 ms + 2 ms per row; every row accepted.
        for round in 0..60 {
            let rows = 1 + round % 5;
            let requests = [ObservedDraftRequest { id: 1, rows, accepted: rows, confidence: None }];
            observe(&mut policy, &requests, 10_000. + 2_000. * rows as f64);
        }
        let price = policy.row_price();
        assert!((price.pooled.unwrap() - 2_000.).abs() < 1., "{price:?}");
        assert!((price.used.unwrap() - 2_000.).abs() < 1., "{price:?}");
        let cost = policy.request_cost(1).unwrap();
        let p = [0.95, 0.9, 0.9, 0.9, 0.9];
        policy.select(&[DraftCandidate { id: 1, confidence: &p }], 64);
        let decision = &policy.decisions[&1];
        // At C1 the waste is exactly one and the batch cost is the request's.
        assert_eq!(decision.waste, 1.);
        assert!(decision.joint_rule);
        assert!((decision.threshold - (2_000. / cost).clamp(THRESHOLD_FLOOR, THRESHOLD_CEILING)).abs() < 1e-12);
        assert_eq!(decision.chosen, decision.joint.iter().take_while(|&&j| j > decision.threshold).count());
    }

    #[test]
    fn derived_without_row_variation_falls_back_to_the_half_rule() {
        let mut policy = SimpleDraft::new(DraftMode::SimpleDerived, 6);
        let requests = [ObservedDraftRequest { id: 1, rows: 3, accepted: 2, confidence: None }];
        for _ in 0..30 { observe(&mut policy, &requests, 9_000.); }
        assert_eq!(policy.row_price().used, None);
        let p = [0.9, 0.6, 0.45];
        let selection = policy.select(&[DraftCandidate { id: 1, confidence: &p }], 64);
        assert_eq!(selection.lengths, [2]);
        assert!(!policy.decisions[&1].joint_rule);
        assert_eq!(policy.decisions[&1].threshold, SIMPLE_THRESHOLD);
    }

    #[test]
    fn skipped_requests_explore_one_draft() {
        let mut policy = SimpleDraft::new(DraftMode::Simple, 4);
        let p = [0.2, 0.2];
        for _ in 0..EXPLORE_SKIPS {
            assert_eq!(policy.select(&[DraftCandidate { id: 1, confidence: &p }], 64).lengths, [0]);
        }
        assert_eq!(policy.select(&[DraftCandidate { id: 1, confidence: &p }], 64).lengths, [1]);
        assert_eq!(policy.select(&[DraftCandidate { id: 1, confidence: &p }], 64).lengths, [0]);
        // Empty proposals never explore.
        assert_eq!(policy.select(&[DraftCandidate { id: 2, confidence: &[] }], 64).lengths, [0]);
    }

    #[test]
    fn calibration_is_global_and_trained_only_on_reached_positions() {
        let mut policy = SimpleDraft::new(DraftMode::Simple, 4);
        let raw = [0.9, 0.9, 0.9, 0.9];
        for _ in 0..200 {
            // Position 1 accepted, position 2 rejected, positions 3-4 unreached.
            let requests = [ObservedDraftRequest { id: 1, rows: 5, accepted: 2, confidence: Some(&raw) }];
            observe(&mut policy, &requests, 5_000.);
        }
        let p = policy.probabilities(&raw);
        assert!(p[0] > 0.95 && p[1] < 0.3, "{p:?}");
        assert_eq!(policy.calibration()[2], (1., 0.));
        // Every request, every concurrency reads the same calibration.
        let wide: Vec<_> = (0..16).map(|id| ObservedDraftRequest { id, rows: 1, accepted: 1, confidence: None }).collect();
        observe(&mut policy, &wide, 9_000.);
        assert_eq!(policy.probabilities(&raw), p);
    }
}
