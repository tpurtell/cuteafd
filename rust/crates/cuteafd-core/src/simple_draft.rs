//! Opt-in simple draft policy (`DRAFT_POLICY=simple|simple-derived`), an
//! experiment beside the shared resource-priced policy: one global cutoff on
//! the calibrated joint acceptance, no cost model, no concurrency buckets.
//!
//! **Confidence.** One online Platt fit per draft position, trained on the
//! positions the verifier reached. `simple-derived` adds a per-request logit
//! offset: the request's realized acceptance at reached positions against
//! what Platt predicted there, as a Beta posterior shrunk to zero offset by
//! [`OFFSET_PRIOR`] tokens. Request `i`'s joint probability of position `k`
//! is `q_ik = prod_{j<=k} p_ij`, the chance that `k` is reached and accepted.
//!
//! **Selection.** Anchors are always verified, and the drafter always runs.
//! Every `(i, k)` with `q_ik > cutoff` is a candidate; when the candidates
//! exceed the row budget the largest `q` win (`q` never rises along a
//! request, so each request keeps a prefix). Otherwise one random request
//! with room verifies one position past its cutoff, so the calibration keeps
//! seeing outcomes there: without it a position whose raw confidence starts
//! below the cutoff is never verified, so never calibrated (V4.1's dSpark
//! head stayed at depth one).
//!
//! **Arms.** `simple`: cutoff 0.5. `simple-rc`: row `n + 1` is verified while
//! `q > (T(n + 1) - T(n)) / c`, `T` the pooled row-cost surface
//! ([`RowCost`]); no offset. `simple-derived`: the same, plus the
//! per-request offset.
//!
//! **Cutoff.** `simple`: 0.5. `simple-rc` and `simple-derived`: `r / c`, clamped to
//! [[`CUTOFF_FLOOR`], [`CUTOFF_CEILING`]], where `r` is the marginal round
//! time of the next verified row on the surface at the rows already taken
//! and `c` is the batch cost per committed token (`T / sum accepted` over a
//! short window): adding a row raises tokens per second iff `q * T > r * E`,
//! i.e. `q > r / c`. Until the rounds have varied their rows enough to fit
//! the surface, the cutoff is 0.5.
use super::{DraftCandidate, DraftRoundObservation, DraftSelection, Platt};

/// Which length rule a [`DraftPolicy`](super::DraftPolicy) follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftMode {
    /// The resource-priced shared policy (default).
    Shared,
    /// Joint acceptance above one half.
    Simple,
    /// Joint acceptance above the measured row price over cost per token.
    SimpleRc,
    /// [`SimpleRc`](Self::SimpleRc) plus per-request offsets and exploration.
    SimpleDerived,
}

impl DraftMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "" | "shared" => Some(Self::Shared),
            "simple" => Some(Self::Simple),
            "simple-rc" => Some(Self::SimpleRc),
            "simple-derived" => Some(Self::SimpleDerived),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Shared => "shared", Self::Simple => "simple", Self::SimpleRc => "simple-rc",
            Self::SimpleDerived => "simple-derived",
        }
    }
}

/// How `simple-rc` and `simple-derived` price one more verified row: one
/// pooled surface over every round, `T = a + k * C + b * f(n)` with `C` the
/// request count and `n` the round's verified rows (unique experts touched
/// follow the rows, concave and shared across requests; attention and
/// per-request work follow `C`). `f(n)` is `n` (linear), `n^alpha` (power,
/// alpha fitted on a grid in (0, 1]) or `ln(1 + n)` (log). Row `n + 1` costs
/// `b * (f(n + 1) - f(n))`, so a concave fit's cutoff falls with depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowCost {
    Linear,
    Power,
    Log,
}

impl RowCost {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "" | "linear" => Some(Self::Linear),
            "power" => Some(Self::Power),
            "log" => Some(Self::Log),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self { Self::Linear => "linear", Self::Power => "power", Self::Log => "log" }
    }
}

/// Exponents the power model chooses among.
const ALPHAS: [f64; 20] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
    0.8, 0.85, 0.9, 0.95, 1.];

/// Decayed least squares of `T = a + k * C + b * f(n)` for one transform `f`.
#[derive(Clone, Copy, Debug, Default)]
struct Surface { xx: [[f64; 3]; 3], xy: [f64; 3], tt: f64 }

impl Surface {
    fn add(&mut self, decay: f64, requests: f64, f: f64, t: f64) {
        let x = [1., requests, f];
        for i in 0..3 {
            for j in 0..3 { self.xx[i][j] = decay * self.xx[i][j] + x[i] * x[j]; }
            self.xy[i] = decay * self.xy[i] + x[i] * t;
        }
        self.tt = decay * self.tt + t * t;
    }
    /// `f` variance left after the intercept and concurrency explain what they can.
    fn row_variance(&self) -> f64 {
        let [a, b, c] = [self.xx[0][0], self.xx[0][1], self.xx[1][1] + CONCURRENCY_RIDGE];
        let det = a * c - b * b;
        if det.partial_cmp(&0.) != Some(std::cmp::Ordering::Greater) { return 0.; }
        let (u, v) = (self.xx[0][2], self.xx[1][2]);
        self.xx[2][2] - (c * u * u - 2. * b * u * v + a * v * v) / det
    }
    /// `(b, residual sum of squares)`.
    fn fit(&self) -> Option<(f64, f64)> {
        let mut m = self.xx;
        m[1][1] += CONCURRENCY_RIDGE;
        let theta = solve3(m, self.xy)?;
        let explained: f64 = (0..3).map(|i| theta[i] * self.xy[i]).sum();
        Some((theta[2], (self.tt - explained).max(0.)))
    }
}

/// One pooled surface over every round: round time against the request
/// count and a transform of the verified rows, for each row-cost model.
#[derive(Clone, Debug)]
struct CurveFit { rows: Surface, log: Surface, power: [Surface; ALPHAS.len()] }

impl Default for CurveFit {
    fn default() -> Self { Self { rows: Surface::default(), log: Surface::default(), power: [Surface::default(); ALPHAS.len()] } }
}

impl CurveFit {
    fn add(&mut self, decay: f64, requests: f64, rows: f64, total: f64) {
        self.rows.add(decay, requests, rows, total);
        self.log.add(decay, requests, rows.ln_1p(), total);
        for (surface, alpha) in self.power.iter_mut().zip(ALPHAS) { surface.add(decay, requests, rows.powf(alpha), total); }
    }
    fn usable(&self) -> bool { self.rows.row_variance() >= MIN_ROW_VARIANCE }
    /// `(exponent, b)` of the chosen model; exponent 0 means log, 1 linear.
    fn model(&self, cost: RowCost) -> Option<(f64, f64)> {
        if !self.usable() { return None; }
        match cost {
            RowCost::Linear => self.rows.fit().map(|(b, _)| (1., b)),
            RowCost::Log => self.log.fit().map(|(b, _)| (0., b)),
            RowCost::Power => self.power.iter().zip(ALPHAS)
                .filter_map(|(surface, alpha)| surface.fit().map(|(b, sse)| (alpha, b, sse)))
                .min_by(|a, b| a.2.total_cmp(&b.2)).map(|(alpha, b, _)| (alpha, b)),
        }
    }
}

/// `T(n + 1) - T(n)` of a fitted model, nonnegative.
fn marginal((exponent, slope): (f64, f64), n: f64) -> f64 {
    let f = |x: f64| if exponent == 0. { x.ln_1p() } else { x.powf(exponent) };
    (slope * (f(n + 1.) - f(n))).max(0.)
}

/// The `simple` cutoff, and the derived one's fallback.
pub const SIMPLE_CUTOFF: f64 = 0.5;
/// Bounds of the derived cutoff.
pub const CUTOFF_FLOOR: f64 = 0.1;
pub const CUTOFF_CEILING: f64 = 0.9;
/// Per-round decay of the short window (about ten rounds) for `c`.
const SHORT_DECAY: f64 = 0.9;
/// Per-round decay of the pooled window (about a hundred rounds) that the
/// short `r` shrinks toward.
const POOLED_DECAY: f64 = 0.99;
/// Least pooled row variance, net of concurrency (rows², weighted), for a usable `r`.
const MIN_ROW_VARIANCE: f64 = 4.;
/// Ridge on the concurrency coefficient, so a window at one concurrency is solvable.
const CONCURRENCY_RIDGE: f64 = 1e-3;
/// Tokens of prior pulling a request's logit offset to zero.
pub const OFFSET_PRIOR: f64 = 3.;
/// Per-round decay of a request's offset evidence, so it follows the content.
const OFFSET_DECAY: f64 = 0.9;
/// Bound on a request's logit offset.
const OFFSET_LIMIT: f64 = 3.;
/// Reporting-only concurrency groups: C1, C2-4, C5-8, C9+.
pub const REPORT_GROUPS: [&str; 4] = ["c1", "c2-4", "c5-8", "c9+"];
fn report_group(requests: usize) -> usize {
    match requests { 0..=1 => 0, 2..=4 => 1, 5..=8 => 2, _ => 3 }
}

fn solve3(mut m: [[f64; 3]; 3], mut y: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() < 1e-12 { return None; }
        m.swap(col, pivot);
        y.swap(col, pivot);
        for row in 0..3 {
            if row == col { continue; }
            let f = m[row][col] / m[col][col];
            for k in col..3 { m[row][k] -= f * m[col][k]; }
            y[row] -= f * y[col];
        }
    }
    let theta = [y[0] / m[0][0], y[1] / m[1][1], y[2] / m[2][2]];
    theta.iter().all(|v| v.is_finite()).then_some(theta)
}

/// The cutoff of one selection and what it came from.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cutoff {
    /// What the rule compares `q` against: one half, or `r / c` clamped for
    /// the first draft row.
    pub value: f64,
    /// `r / c` before the clamp, when both were known.
    pub raw: Option<f64>,
    /// `r / c` hit [`CUTOFF_FLOOR`] or [`CUTOFF_CEILING`].
    pub clamped: bool,
    /// Marginal µs of the first draft row, `T(C + 1) - T(C)` on the surface.
    pub row_us: Option<f64>,
    /// Batch µs per committed token over the short window.
    pub token_us: Option<f64>,
    /// The derived rule wanted `r / c` but held one half.
    pub fallback: bool,
    /// The surface's `(exponent, b)` (exponent 0 is log, 1 linear) whose
    /// marginal `T(n + 1) - T(n)` prices each row.
    pub curve: Option<(f64, f64)>,
}

/// One request's decision, kept for the round's trace.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleDecision {
    pub id: u64,
    /// The request's logit offset (zero for `simple`).
    pub offset: f64,
    pub conditional: Vec<f64>,
    pub joint: Vec<f64>,
    /// Positions above the cutoff, then what the budget kept (plus exploration).
    pub passing: usize,
    pub chosen: usize,
    pub explore: bool,
    /// Concave row cost: the cutoff each chosen position cleared (or failed).
    pub cutoffs: Vec<f64>,
}

/// One selection: the cutoff, the draft-row budget and each request's decision.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleSelection {
    pub cutoff: Cutoff,
    pub budget: usize,
    pub scarce: bool,
    pub requests: Vec<SimpleDecision>,
}

/// The latest observed round with the selection behind it, for traces.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleRound {
    pub total_us: f64,
    pub rows: usize,
    pub accepted: usize,
    pub requests: usize,
    pub selection: Option<SimpleSelection>,
    /// The cutoff state after learning from this round.
    pub after: Cutoff,
}

/// Predicted against realized acceptance at one position.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reliability {
    /// Times the position was reached (every earlier draft accepted), the
    /// summed calibrated conditional there, and acceptances.
    pub reached: u64,
    pub predicted_conditional: f64,
    pub accepted: u64,
    /// Times the position was verified, the summed calibrated joint `q`, and
    /// how often every draft through it was accepted.
    pub verified: u64,
    pub predicted_joint: f64,
    pub realized_joint: u64,
}

/// A request's decayed outcomes at reached positions: acceptances and the
/// summed Platt prediction there.
#[derive(Clone, Copy, Debug, Default)]
struct Outcomes { hits: f64, predicted: f64, reached: f64 }

impl Outcomes {
    /// Beta posterior centred on the request's mean prediction, as a logit offset.
    fn offset(&self) -> f64 {
        if self.reached <= 0. { return 0.; }
        let mean = (self.predicted / self.reached).clamp(1e-4, 1. - 1e-4);
        let posterior = ((self.hits + OFFSET_PRIOR * mean) / (self.reached + OFFSET_PRIOR)).clamp(1e-4, 1. - 1e-4);
        (logit(posterior) - logit(mean)).clamp(-OFFSET_LIMIT, OFFSET_LIMIT)
    }
}

fn logit(p: f64) -> f64 { (p / (1. - p)).ln() }
fn sigmoid(z: f64) -> f64 { 1. / (1. + (-z).exp()) }

#[derive(Clone, Debug)]
pub(crate) struct SimpleDraft {
    mode: DraftMode,
    outcomes: std::collections::BTreeMap<u64, Outcomes>,
    calibration: Vec<Platt>,
    /// First copy-tail position: the drafter's widest native block.
    native: usize,
    /// Short-window committed tokens and round time, for `c`.
    tokens: f64,
    time: f64,
    /// The pooled row-cost surface.
    curve: CurveFit,
    row_cost: RowCost,
    rng: u64,
    pending: Option<SimpleSelection>,
    last: Option<SimpleRound>,
    /// `[group][native, tail][position]`.
    reliability: [[Vec<Reliability>; 2]; 4],
}

impl SimpleDraft {
    pub fn new(mode: DraftMode, positions: usize, native: usize) -> Self {
        Self { mode, outcomes: Default::default(), calibration: vec![Platt::new(); positions], native, tokens: 0., time: 0.,
            curve: CurveFit::default(), row_cost: RowCost::Linear, rng: 0x9e37_79b9_7f4a_7c15, pending: None, last: None,
            reliability: std::array::from_fn(|_| std::array::from_fn(|_| vec![Reliability::default(); positions])) }
    }
    pub fn mode(&self) -> DraftMode { self.mode }
    pub fn row_cost(&self) -> RowCost { self.row_cost }
    pub fn set_row_cost(&mut self, cost: RowCost) { self.row_cost = cost; }
    pub fn set_native(&mut self, native: usize) { self.native = native; }
    pub fn calibration(&self) -> Vec<(f64, f64)> {
        self.calibration.iter().map(|platt| (platt.theta[0], platt.theta[1])).collect()
    }
    pub fn last_round(&self) -> Option<&SimpleRound> { self.last.as_ref() }
    /// The selection awaiting its round's observation.
    pub fn pending(&self) -> Option<&SimpleSelection> { self.pending.as_ref() }
    /// `[group][native, tail][position]`, groups as [`REPORT_GROUPS`].
    pub fn reliability(&self) -> &[[Vec<Reliability>; 2]; 4] { &self.reliability }
    /// Platt-calibrated conditional acceptance per position.
    fn platt(&self, confidence: &[f64]) -> Vec<f64> {
        confidence.iter().zip(&self.calibration).map(|(&p, platt)| platt.apply(p)).collect()
    }
    /// The request's logit offset (zero for `simple`).
    pub fn offset(&self, id: u64) -> f64 {
        if self.mode != DraftMode::SimpleDerived { return 0.; }
        self.outcomes.get(&id).map_or(0., Outcomes::offset)
    }
    /// Conditional acceptance per position as the rule uses it.
    pub fn probabilities(&self, id: u64, confidence: &[f64]) -> Vec<f64> {
        let offset = self.offset(id);
        self.platt(confidence).into_iter()
            .map(|p| if offset == 0. { p } else { sigmoid(logit(p.clamp(1e-6, 1. - 1e-6)) + offset) }).collect()
    }
    pub fn release(&mut self, id: u64) { self.outcomes.remove(&id); }
    /// The cutoff of the first draft row of a selection over `requests`
    /// requests (`n` = the anchors). Deeper rows are priced in the selection.
    pub fn cutoff(&self, requests: usize) -> Cutoff {
        let token_us = self.cost_per_token();
        let curve = matches!(self.mode, DraftMode::SimpleRc | DraftMode::SimpleDerived)
            .then(|| self.curve.model(self.row_cost)).flatten();
        let row_us = curve.map(|model| marginal(model, requests as f64));
        let mut cutoff = Cutoff { value: SIMPLE_CUTOFF, raw: None, clamped: false, row_us, token_us, fallback: false,
            curve };
        if matches!(self.mode, DraftMode::SimpleRc | DraftMode::SimpleDerived) {
            match (row_us, token_us) {
                (Some(r), Some(c)) if c > 0. => {
                    let raw = r / c;
                    cutoff.raw = Some(raw);
                    cutoff.value = raw.clamp(CUTOFF_FLOOR, CUTOFF_CEILING);
                    cutoff.clamped = cutoff.value != raw;
                }
                _ => { cutoff.fallback = true; cutoff.curve = None; }
            }
        }
        cutoff
    }
    fn cost_per_token(&self) -> Option<f64> {
        (self.tokens > 0.).then(|| self.time / self.tokens)
    }
    fn random(&mut self, below: usize) -> usize {
        // splitmix64: deterministic, so replays and tests repeat.
        self.rng = self.rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) % below as u64) as usize
    }

    /// Lengths under `max_rows` verifier rows (anchors included).
    pub fn select(&mut self, candidates: &[DraftCandidate<'_>], max_rows: usize) -> DraftSelection {
        let cutoff = self.cutoff(candidates.len());
        let mut decisions: Vec<SimpleDecision> = candidates.iter().map(|candidate| {
            let conditional = self.probabilities(candidate.id, candidate.confidence);
            let mut product = 1.;
            let joint: Vec<f64> = conditional.iter().map(|p| { product *= p; product }).collect();
            let passing = joint.iter().take_while(|&&q| q > cutoff.value).count();
            SimpleDecision { id: candidate.id, offset: self.offset(candidate.id), conditional, joint, passing,
                chosen: passing, explore: false, cutoffs: Vec::new() }
        }).collect();
        let budget = max_rows.saturating_sub(candidates.len());
        let wanted: usize = decisions.iter().map(|d| d.passing).sum();
        let scarce = wanted > budget;
        if scarce {
            let mut positions: Vec<(f64, usize, usize)> = decisions.iter().enumerate()
                .flat_map(|(index, d)| (0..d.passing).map(move |k| (d.joint[k], k, index))).collect();
            positions.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            for decision in &mut decisions { decision.chosen = 0; }
            for &(_, _, index) in positions.iter().take(budget) { decisions[index].chosen += 1; }
        } else if wanted < budget {
            let open: Vec<usize> = (0..decisions.len()).filter(|&i| decisions[i].chosen < decisions[i].joint.len()).collect();
            if !open.is_empty() {
                let index = open[self.random(open.len())];
                decisions[index].chosen += 1;
                decisions[index].explore = true;
            }
        }
        if let (Some(model), Some(token)) = (cutoff.curve, cutoff.token_us.filter(|&c| c > 0.)) {
            self.price_by_depth(&mut decisions, model, token, candidates.len(), budget);
        }
        let scarce = scarce || decisions.iter().any(|d| d.chosen < d.passing);
        let lengths: Vec<usize> = decisions.iter().map(|d| d.chosen).collect();
        let expected = candidates.len() as f64
            + decisions.iter().map(|d| d.joint[..d.chosen].iter().sum::<f64>()).sum::<f64>();
        self.pending = Some(SimpleSelection { cutoff, budget, scarce, requests: decisions });
        DraftSelection { lengths, expected_tokens: expected, predicted_us: 0., evaluated: 1 }
    }

    /// Concave row cost: take positions in descending `q` (each request's in
    /// prefix order), each while its `q` clears `T(n + 1) - T(n)` over the
    /// cost per token at the rows already taken, then explore one past.
    fn price_by_depth(&mut self, decisions: &mut [SimpleDecision], model: (f64, f64), token: f64, anchors: usize,
        budget: usize) {
        let mut positions: Vec<(f64, usize, usize)> = decisions.iter().enumerate()
            .flat_map(|(index, d)| (0..d.joint.len()).map(move |k| (d.joint[k], k, index))).collect();
        positions.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for decision in decisions.iter_mut() { decision.chosen = 0; decision.passing = 0; decision.explore = false; }
        let mut blocked = vec![false; decisions.len()];
        let mut rows = anchors;
        for (q, k, index) in positions {
            if blocked[index] || k != decisions[index].chosen { continue; }
            let cutoff = (marginal(model, rows as f64) / token).clamp(CUTOFF_FLOOR, CUTOFF_CEILING);
            decisions[index].cutoffs.push(cutoff);
            if q <= cutoff { blocked[index] = true; continue; }
            decisions[index].passing += 1;
            if rows - anchors >= budget { continue; }
            decisions[index].chosen += 1;
            rows += 1;
        }
        let open: Vec<usize> = (0..decisions.len()).filter(|&i| decisions[i].chosen < decisions[i].joint.len()
            && decisions[i].chosen == decisions[i].passing).collect();
        if rows - anchors < budget && !open.is_empty() {
            let index = open[self.random(open.len())];
            decisions[index].chosen += 1;
            decisions[index].explore = true;
        }
    }

    /// Learn the calibration, `r` and `c` from one completed round.
    pub fn observe(&mut self, round: &DraftRoundObservation<'_>) {
        let rows: usize = round.requests.iter().map(|r| r.rows).sum();
        let accepted: usize = round.requests.iter().map(|r| r.accepted).sum();
        let group = report_group(round.requests.len());
        for request in round.requests {
            let Some(confidence) = request.confidence.filter(|_| request.rows > 1) else { continue };
            let hits = request.accepted - 1;
            let positions = (request.rows - 1).min(confidence.len()).min(self.calibration.len());
            let predicted = self.probabilities(request.id, &confidence[..positions]);
            let platt = self.platt(&confidence[..positions]);
            let outcomes = self.outcomes.entry(request.id).or_default();
            outcomes.hits *= OFFSET_DECAY;
            outcomes.predicted *= OFFSET_DECAY;
            outcomes.reached *= OFFSET_DECAY;
            for (position, &p) in platt.iter().enumerate().take(hits + 1) {
                outcomes.hits += f64::from(u8::from(position < hits));
                outcomes.predicted += p;
                outcomes.reached += 1.;
            }
            let mut joint = 1.;
            for (position, &p) in predicted.iter().enumerate() {
                joint *= p;
                let cell = &mut self.reliability[group][usize::from(position >= self.native)][position];
                cell.verified += 1;
                cell.predicted_joint += joint;
                cell.realized_joint += u64::from(position < hits);
                // Only positions whose predecessors were all accepted were judged.
                if position <= hits {
                    cell.reached += 1;
                    cell.predicted_conditional += p;
                    cell.accepted += u64::from(position < hits);
                    self.calibration[position].observe(confidence[position], f64::from(u8::from(position < hits)));
                }
            }
        }
        if round.total_us.is_finite() && round.total_us > 0. && rows > 0 {
            self.tokens = SHORT_DECAY * self.tokens + accepted as f64;
            self.time = SHORT_DECAY * self.time + round.total_us;
            self.curve.add(POOLED_DECAY, round.requests.len() as f64, rows as f64, round.total_us);
        }
        self.last = Some(SimpleRound { total_us: round.total_us, rows, accepted, requests: round.requests.len(),
            selection: self.pending.take(), after: self.cutoff(round.requests.len()) });
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
    fn simple_verifies_while_joint_acceptance_exceeds_half_then_explores_one_row() {
        for mode in [DraftMode::Simple, DraftMode::SimpleRc, DraftMode::SimpleDerived] {
            let mut policy = SimpleDraft::new(mode, 6, 5);
            // Joint 0.9, 0.72, 0.58, 0.46: every conditional passes one half, the fourth joint does not.
            // No row price yet, so every arm holds one half; one exploration row follows.
            let p = [0.9, 0.8, 0.8, 0.8, 0.99];
            let selection = policy.select(&[DraftCandidate { id: 1, confidence: &p }], 64);
            assert_eq!(selection.lengths, [4], "{mode:?}");
            let pending = policy.pending.as_ref().unwrap();
            assert_eq!((pending.requests[0].passing, pending.requests[0].explore, pending.requests[0].offset), (3, true, 0.));
            assert_eq!(pending.cutoff.value, SIMPLE_CUTOFF);
            // A request that verifies everything leaves nothing to explore.
            let all = [0.99; 3];
            assert_eq!(policy.select(&[DraftCandidate { id: 1, confidence: &all }], 64).lengths, [3]);
        }
    }

    #[test]
    fn exploration_escapes_an_uncalibrated_low_second_position() {
        // V4.1-like: raw position 2 confidence 0.16, truly accepted 80% of the time.
        let mut policy = SimpleDraft::new(DraftMode::Simple, 5, 5);
        let raw = [0.9, 0.16, 0.16, 0.16, 0.16];
        let mut state = 7u64;
        for _ in 0..400 {
            let lengths = policy.select(&[DraftCandidate { id: 1, confidence: &raw }], 64).lengths;
            let mut accepted = 1;
            for _ in 0..lengths[0] {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                if (state >> 33) % 10 < 8 { accepted += 1 } else { break }
            }
            observe(&mut policy, &[ObservedDraftRequest { id: 1, rows: lengths[0] + 1, accepted, confidence: Some(&raw) }], 5_000.);
        }
        assert!(policy.probabilities(1, &raw)[1] > 0.6, "{:?}", policy.probabilities(1, &raw));
        assert!(policy.select(&[DraftCandidate { id: 1, confidence: &raw }], 64).lengths[0] >= 3);
    }

    #[test]
    fn derived_offset_shrinks_a_request_toward_its_own_outcomes() {
        let mut policy = SimpleDraft::new(DraftMode::SimpleDerived, 4, 4);
        let raw = [0.7, 0.7, 0.7, 0.7];
        // Request 1 accepts everything, request 2 nothing; Platt sees both.
        for _ in 0..20 {
            let requests = [ObservedDraftRequest { id: 1, rows: 5, accepted: 5, confidence: Some(&raw) },
                ObservedDraftRequest { id: 2, rows: 5, accepted: 1, confidence: Some(&raw) }];
            observe(&mut policy, &requests, 5_000.);
        }
        let (one, two) = (policy.offset(1), policy.offset(2));
        assert!(one > 0.5 && two < -0.5, "{one} {two}");
        assert!(policy.probabilities(1, &raw)[0] > policy.probabilities(9, &raw)[0]);
        assert!(policy.probabilities(2, &raw)[0] < policy.probabilities(9, &raw)[0]);
        // One outcome moves a fresh request less than the prior.
        let mut fresh = Outcomes::default();
        fresh.hits = 1.;
        fresh.predicted = 0.5;
        fresh.reached = 1.;
        assert!((fresh.offset() - (logit((1. + 1.5) / 4.) - logit(0.5))).abs() < 1e-12);
        policy.release(1);
        assert_eq!(policy.offset(1), 0.);
        // Simple ignores outcomes per request.
        let mut simple = SimpleDraft::new(DraftMode::Simple, 4, 4);
        let requests = [ObservedDraftRequest { id: 1, rows: 5, accepted: 5, confidence: Some(&raw) }];
        observe(&mut simple, &requests, 5_000.);
        assert_eq!(simple.offset(1), 0.);
    }

    #[test]
    fn scarce_rows_go_to_the_largest_joint_in_prefix_order_without_exploring() {
        let mut policy = SimpleDraft::new(DraftMode::Simple, 6, 5);
        let a = [0.95, 0.9, 0.9];
        let b = [0.8, 0.9, 0.9];
        let candidates = [DraftCandidate { id: 1, confidence: &a }, DraftCandidate { id: 2, confidence: &b }];
        // Joint a: 0.95 0.855 0.77; b: 0.8 0.72 0.65. Three rows: a1 a2 b1.
        let selection = policy.select(&candidates, 2 + 3);
        assert_eq!(selection.lengths, [2, 1]);
        let pending = policy.pending.as_ref().unwrap();
        assert!(pending.scarce && pending.requests.iter().all(|d| !d.explore));
    }

    #[test]
    fn derived_cutoff_is_row_price_over_batch_cost_per_token() {
        let mut policy = SimpleDraft::new(DraftMode::SimpleDerived, 8, 5);
        // T = 5 ms + 1 ms per request + 2 ms per row; rows vary at two concurrencies.
        for round in 0..200 {
            let requests = 1 + 3 * (round % 2);
            let per = 1 + round % 5;
            let observed: Vec<_> = (0..requests as u64).map(|id| ObservedDraftRequest { id, rows: per,
                accepted: per, confidence: None }).collect();
            let rows = (requests * per) as f64;
            observe(&mut policy, &observed, 5_000. + 1_000. * requests as f64 + 2_000. * rows);
        }
        // One surface over both concurrencies recovers the row price and the per-request term.
        let cutoff = policy.cutoff(1);
        assert_eq!(cutoff.curve.map(|c| c.0), Some(1.));
        assert!((cutoff.row_us.unwrap() - 2_000.).abs() < 5., "{cutoff:?}");
        let token = cutoff.token_us.unwrap();
        assert!(!cutoff.fallback);
        assert!((cutoff.value - (cutoff.row_us.unwrap() / token).clamp(CUTOFF_FLOOR, CUTOFF_CEILING)).abs() < 1e-9);
        let p = [0.95, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9];
        let selection = policy.select(&[DraftCandidate { id: 9, confidence: &p }], 64);
        let pending = policy.pending.as_ref().unwrap();
        let passing = pending.requests[0].joint.iter().take_while(|&&q| q > cutoff.value).count();
        assert_eq!(pending.requests[0].passing, passing);
        assert_eq!(selection.lengths, [(passing + 1).min(7)]);
    }

    #[test]
    fn rc_uses_the_derived_cutoff_without_offsets() {
        let mut rc = SimpleDraft::new(DraftMode::SimpleRc, 8, 5);
        let mut derived = SimpleDraft::new(DraftMode::SimpleDerived, 8, 5);
        let raw = [0.7; 7];
        for round in 0..200 {
            let per = 1 + round % 5;
            let observed = [ObservedDraftRequest { id: 1, rows: per, accepted: per, confidence: Some(&raw[..per - 1]) }];
            for policy in [&mut rc, &mut derived] { observe(policy, &observed, 5_000. + 2_000. * per as f64); }
        }
        // Both arms price rows the same way.
        let (a, b) = (rc.cutoff(1), derived.cutoff(1));
        assert_eq!(a, b);
        assert!(!a.fallback);
        assert_eq!(a.raw.map(|raw| raw.clamp(CUTOFF_FLOOR, CUTOFF_CEILING)), Some(a.value));
        // A linear surface prices every row the same at any concurrency.
        assert!((rc.cutoff(16).row_us.unwrap() - a.row_us.unwrap()).abs() < 1e-6);
        assert_eq!(rc.offset(1), 0.);
        assert!(derived.offset(1) > 0.);
        let p = [0.9, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5];
        let selection = rc.select(&[DraftCandidate { id: 9, confidence: &p }], 64);
        let pending = rc.pending.as_ref().unwrap();
        assert!(pending.requests[0].explore);
        assert_eq!(selection.lengths, [pending.requests[0].passing + 1]);
        assert_eq!(DraftMode::parse("simple-rc"), Some(DraftMode::SimpleRc));
    }

    #[test]
    fn concave_row_cost_raises_the_cutoff_for_shallow_rows() {
        for cost in [RowCost::Power, RowCost::Log] {
            let mut policy = SimpleDraft::new(DraftMode::SimpleDerived, 8, 8);
            policy.set_row_cost(cost);
            // T = 10 ms + 8 ms * sqrt(rows): marginal row cost falls with depth.
            for round in 0..300 {
                let rows = 1 + round % 8;
                let observed = [ObservedDraftRequest { id: 1, rows, accepted: rows.min(3), confidence: None }];
                observe(&mut policy, &observed, 10_000. + 8_000. * (rows as f64).sqrt());
            }
            let cutoff = policy.cutoff(1);
            let model = cutoff.curve.unwrap();
            if cost == RowCost::Power { assert!((model.0 - 0.5).abs() < 0.051, "{model:?}"); }
            assert!(marginal(model, 1.) > marginal(model, 6.));
            let p = [0.95; 7];
            policy.select(&[DraftCandidate { id: 9, confidence: &p }], 64);
            let decision = &policy.pending.as_ref().unwrap().requests[0];
            // Each successive position faces a lower (or equal, once clamped) cutoff.
            assert!(decision.cutoffs.windows(2).all(|w| w[1] <= w[0] + 1e-12), "{:?}", decision.cutoffs);
            assert_eq!(decision.cutoffs[0], cutoff.value);
        }
        assert_eq!(RowCost::parse("power"), Some(RowCost::Power));
        assert_eq!(RowCost::parse(""), Some(RowCost::Linear));
    }

    #[test]
    fn derived_without_row_variation_falls_back_to_one_half() {
        let mut policy = SimpleDraft::new(DraftMode::SimpleDerived, 6, 5);
        let requests = [ObservedDraftRequest { id: 1, rows: 3, accepted: 2, confidence: None }];
        for _ in 0..30 { observe(&mut policy, &requests, 9_000.); }
        let cutoff = policy.cutoff(1);
        assert!(cutoff.fallback && cutoff.row_us.is_none());
        assert_eq!(cutoff.value, SIMPLE_CUTOFF);
        assert!((cutoff.token_us.unwrap() - 4_500.).abs() < 1e-6);
    }

    #[test]
    fn calibration_is_global_trained_on_reached_positions_and_reported_by_segment() {
        let mut policy = SimpleDraft::new(DraftMode::Simple, 4, 2);
        let raw = [0.9, 0.9, 0.9, 0.9];
        for _ in 0..200 {
            // Position 1 accepted, position 2 rejected, positions 3-4 verified but unreached.
            let requests = [ObservedDraftRequest { id: 1, rows: 5, accepted: 2, confidence: Some(&raw) }];
            observe(&mut policy, &requests, 5_000.);
        }
        let p = policy.probabilities(1, &raw);
        assert!(p[0] > 0.95 && p[1] < 0.3, "{p:?}");
        assert_eq!(policy.calibration()[2], (1., 0.));
        let c1 = &policy.reliability()[0];
        assert_eq!((c1[0][0].reached, c1[0][0].accepted, c1[0][0].verified, c1[0][0].realized_joint), (200, 200, 200, 200));
        assert_eq!((c1[0][1].reached, c1[0][1].accepted), (200, 0));
        // Positions 3-4 are copy tail (native = 2): verified, never reached.
        assert_eq!((c1[1][2].verified, c1[1][2].reached, c1[1][2].realized_joint), (200, 0, 0));
        assert_eq!(c1[0][2], Reliability::default());
        // Every concurrency trains the same calibration; reports split by group.
        let wide: Vec<_> = (0..16).map(|id| ObservedDraftRequest { id, rows: 2, accepted: 2, confidence: Some(&raw) }).collect();
        observe(&mut policy, &wide, 9_000.);
        assert_eq!(policy.reliability()[3][0][0].verified, 16);
        assert_ne!(policy.probabilities(1, &raw), p);
    }
}
