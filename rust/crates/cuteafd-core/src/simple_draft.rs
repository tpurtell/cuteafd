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
//! request, so each request keeps a prefix). `simple-derived` then lets one
//! random request with room verify one position past its cutoff, so the
//! calibration keeps seeing outcomes there.
//!
//! **Arms.** `simple`: cutoff 0.5. `simple-rc`: `r / c` with `r` the pooled
//! slope of round time on verified rows among rounds at the selection's exact
//! request count (the nearest count with enough row variation when that one
//! has too little), so a concurrency ramp cannot leak into the slope; no
//! offset and no exploration. `simple-derived`: `r / c` with the
//! short-window `r` shrunk to the pooled one, plus the offset and exploration.
//!
//! **Cutoff.** `simple`: 0.5. `simple-rc` and `simple-derived`: `r / c`, clamped to
//! [[`CUTOFF_FLOOR`], [`CUTOFF_CEILING`]], where `r` is the marginal round
//! time of one more verified row (from `T = a0 + a1 * C + r * rows` over a
//! short window) and `c` is the batch cost per committed token (`T / sum
//! accepted`, same window): adding a row raises tokens per second iff
//! `q * T > r * E`, i.e. `q > r / c`. Until the pooled window has varied its
//! rows enough to fit `r`, the cutoff is 0.5.
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

/// The `simple` cutoff, and the derived one's fallback.
pub const SIMPLE_CUTOFF: f64 = 0.5;
/// Bounds of the derived cutoff.
pub const CUTOFF_FLOOR: f64 = 0.1;
pub const CUTOFF_CEILING: f64 = 0.9;
/// Per-round decay of the short window (about ten rounds) for `r` and `c`.
const SHORT_DECAY: f64 = 0.9;
/// Per-round decay of the pooled window (about a hundred rounds) that the
/// short `r` shrinks toward.
const POOLED_DECAY: f64 = 0.99;
/// Ridge strength, in rows², pulling the short `r` to the pooled one.
const SLOPE_PRIOR: f64 = 4.;
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

/// Decayed weighted least squares of `T = a0 + a1 * C + r * rows`, with the
/// window's committed tokens and time.
#[derive(Clone, Copy, Debug, Default)]
struct RoundFit { xx: [[f64; 3]; 3], xy: [f64; 3], tokens: f64, time: f64 }

impl RoundFit {
    fn add(&mut self, decay: f64, requests: f64, rows: f64, total: f64, tokens: f64) {
        let x = [1., requests, rows];
        for i in 0..3 {
            for j in 0..3 { self.xx[i][j] = decay * self.xx[i][j] + x[i] * x[j]; }
            self.xy[i] = decay * self.xy[i] + x[i] * total;
        }
        self.tokens = decay * self.tokens + tokens;
        self.time = decay * self.time + total;
    }
    /// Row variance left after the intercept and concurrency explain what they can.
    fn row_variance(&self) -> f64 {
        let [a, b, c] = [self.xx[0][0], self.xx[0][1], self.xx[1][1] + CONCURRENCY_RIDGE];
        let det = a * c - b * b;
        if det.partial_cmp(&0.) != Some(std::cmp::Ordering::Greater) { return 0.; }
        let (u, v) = (self.xx[0][2], self.xx[1][2]);
        self.xx[2][2] - (c * u * u - 2. * b * u * v + a * v * v) / det
    }
    /// The row coefficient with a ridge of `strength` toward `prior`, if the
    /// window varied its rows enough (or the ridge carries it).
    fn slope(&self, prior: f64, strength: f64) -> Option<f64> {
        let mut m = self.xx;
        let mut y = self.xy;
        m[1][1] += CONCURRENCY_RIDGE;
        m[2][2] += strength;
        y[2] += strength * prior;
        solve3(m, y).map(|theta| theta[2])
    }
    fn cost_per_token(&self) -> Option<f64> {
        (self.tokens > 0.).then(|| self.time / self.tokens)
    }
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
    /// What the rule compares `q` against (one half, or `r / c` clamped).
    pub value: f64,
    /// `r / c` before the clamp, when both were known.
    pub raw: Option<f64>,
    /// `r / c` hit [`CUTOFF_FLOOR`] or [`CUTOFF_CEILING`].
    pub clamped: bool,
    /// Request count whose pooled `r` `simple-rc` used.
    pub group: Option<usize>,
    /// Marginal µs per verified row: short window alone, pooled, and the
    /// short window shrunk to the pooled value (what the rule uses).
    pub row_short_us: Option<f64>,
    pub row_pooled_us: Option<f64>,
    pub row_us: Option<f64>,
    /// Batch µs per committed token over the short window.
    pub token_us: Option<f64>,
    /// The derived rule wanted `r / c` but held one half.
    pub fallback: bool,
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
    short: RoundFit,
    pooled: RoundFit,
    /// Pooled fits per exact request count (`simple-rc`'s `r`), index `C - 1`.
    by_requests: Vec<RoundFit>,
    rng: u64,
    pending: Option<SimpleSelection>,
    last: Option<SimpleRound>,
    /// `[group][native, tail][position]`.
    reliability: [[Vec<Reliability>; 2]; 4],
}

impl SimpleDraft {
    pub fn new(mode: DraftMode, positions: usize, native: usize) -> Self {
        Self { mode, outcomes: Default::default(), calibration: vec![Platt::new(); positions], native, short: RoundFit::default(),
            pooled: RoundFit::default(), by_requests: Vec::new(), rng: 0x9e37_79b9_7f4a_7c15, pending: None, last: None,
            reliability: std::array::from_fn(|_| std::array::from_fn(|_| vec![Reliability::default(); positions])) }
    }
    pub fn mode(&self) -> DraftMode { self.mode }
    pub fn set_native(&mut self, native: usize) { self.native = native; }
    pub fn calibration(&self) -> Vec<(f64, f64)> {
        self.calibration.iter().map(|platt| (platt.theta[0], platt.theta[1])).collect()
    }
    pub fn last_round(&self) -> Option<&SimpleRound> { self.last.as_ref() }
    /// `[group][native, tail][position]`, groups as [`REPORT_GROUPS`].
    pub fn reliability(&self) -> &[[Vec<Reliability>; 2]; 4] { &self.reliability }
    pub fn native_positions(&self) -> usize { self.native }
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
    /// The cutoff for a selection over `requests` requests.
    pub fn cutoff(&self, requests: usize) -> Cutoff {
        let usable = |fit: &RoundFit| fit.row_variance() >= MIN_ROW_VARIANCE;
        let short = usable(&self.short).then(|| self.short.slope(0., 0.)).flatten();
        let (pooled_fit, group) = match self.mode {
            DraftMode::SimpleRc => {
                // The nearest request count with enough row variation, ties to the larger.
                let nearest = (0..self.by_requests.len()).filter(|&i| usable(&self.by_requests[i]))
                    .min_by_key(|&i| ((i + 1).abs_diff(requests), usize::MAX - i));
                match nearest { Some(i) => (&self.by_requests[i], Some(i + 1)), None => (&self.pooled, None) }
            }
            _ => (&self.pooled, None),
        };
        let pooled = match self.mode {
            DraftMode::SimpleRc => group.and_then(|_| pooled_fit.slope(0., 0.)),
            _ => usable(pooled_fit).then(|| pooled_fit.slope(0., 0.)).flatten(),
        };
        let row_us = match self.mode {
            DraftMode::SimpleRc => pooled,
            _ => pooled.and_then(|pooled| self.short.slope(pooled, SLOPE_PRIOR)),
        }.map(|r| r.max(0.));
        let token_us = self.short.cost_per_token();
        let mut cutoff = Cutoff { value: SIMPLE_CUTOFF, raw: None, clamped: false, group, row_short_us: short,
            row_pooled_us: pooled, row_us, token_us, fallback: false };
        if matches!(self.mode, DraftMode::SimpleRc | DraftMode::SimpleDerived) {
            match (row_us, token_us) {
                (Some(r), Some(c)) if c > 0. => {
                    let raw = r / c;
                    cutoff.raw = Some(raw);
                    cutoff.value = raw.clamp(CUTOFF_FLOOR, CUTOFF_CEILING);
                    cutoff.clamped = cutoff.value != raw;
                }
                _ => cutoff.fallback = true,
            }
        }
        cutoff
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
                chosen: passing, explore: false }
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
        } else if wanted < budget && self.mode == DraftMode::SimpleDerived {
            let open: Vec<usize> = (0..decisions.len()).filter(|&i| decisions[i].chosen < decisions[i].joint.len()).collect();
            if !open.is_empty() {
                let index = open[self.random(open.len())];
                decisions[index].chosen += 1;
                decisions[index].explore = true;
            }
        }
        let lengths: Vec<usize> = decisions.iter().map(|d| d.chosen).collect();
        let expected = candidates.len() as f64
            + decisions.iter().map(|d| d.joint[..d.chosen].iter().sum::<f64>()).sum::<f64>();
        self.pending = Some(SimpleSelection { cutoff, budget, scarce, requests: decisions });
        DraftSelection { lengths, expected_tokens: expected, predicted_us: 0., evaluated: 1 }
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
            let requests = round.requests.len() as f64;
            self.short.add(SHORT_DECAY, requests, rows as f64, round.total_us, accepted as f64);
            self.pooled.add(POOLED_DECAY, requests, rows as f64, round.total_us, accepted as f64);
            let count = round.requests.len();
            if self.by_requests.len() < count { self.by_requests.resize(count, RoundFit::default()); }
            self.by_requests[count - 1].add(POOLED_DECAY, requests, rows as f64, round.total_us, accepted as f64);
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
    fn simple_verifies_while_joint_acceptance_exceeds_half_and_never_explores() {
        let mut policy = SimpleDraft::new(DraftMode::Simple, 6, 5);
        // Joint 0.9, 0.72, 0.58, 0.46: every conditional passes one half, the fourth joint does not.
        let p = [0.9, 0.8, 0.8, 0.8, 0.99];
        let selection = policy.select(&[DraftCandidate { id: 1, confidence: &p }], 64);
        assert_eq!(selection.lengths, [3]);
        let pending = policy.pending.as_ref().unwrap();
        assert_eq!((pending.requests[0].passing, pending.requests[0].explore, pending.requests[0].offset), (3, false, 0.));
        assert_eq!(pending.cutoff.value, SIMPLE_CUTOFF);
    }

    #[test]
    fn derived_explores_one_row_past_the_cutoff() {
        let mut policy = SimpleDraft::new(DraftMode::SimpleDerived, 6, 5);
        let p = [0.9, 0.8, 0.8, 0.8, 0.99];
        // No row price yet: the derived cutoff holds one half, plus one exploration row.
        assert_eq!(policy.select(&[DraftCandidate { id: 1, confidence: &p }], 64).lengths, [4]);
        assert!(policy.pending.as_ref().unwrap().requests[0].explore);
        // A request that verifies everything leaves nothing to explore.
        let all = [0.99; 3];
        assert_eq!(policy.select(&[DraftCandidate { id: 1, confidence: &all }], 64).lengths, [3]);
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
        let cutoff = policy.cutoff(1);
        assert!((cutoff.row_pooled_us.unwrap() - 2_000.).abs() < 5., "{cutoff:?}");
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
    fn rc_uses_the_derived_cutoff_without_offsets_or_exploration() {
        let mut rc = SimpleDraft::new(DraftMode::SimpleRc, 8, 5);
        let mut derived = SimpleDraft::new(DraftMode::SimpleDerived, 8, 5);
        let raw = [0.7; 7];
        for round in 0..200 {
            let per = 1 + round % 5;
            let observed = [ObservedDraftRequest { id: 1, rows: per, accepted: per, confidence: Some(&raw[..per - 1]) }];
            for policy in [&mut rc, &mut derived] { observe(policy, &observed, 5_000. + 2_000. * per as f64); }
        }
        // One concurrency group: the group's pooled r equals the pooled r.
        let (a, b) = (rc.cutoff(1), derived.cutoff(1));
        assert_eq!((a.row_pooled_us, a.token_us), (b.row_pooled_us, b.token_us));
        assert_eq!(a.row_us, a.row_pooled_us);
        assert_eq!(a.group, Some(1));
        assert!(!a.fallback);
        assert_eq!(a.raw.map(|raw| raw.clamp(CUTOFF_FLOOR, CUTOFF_CEILING)), Some(a.value));
        // C16 borrows the nearest request count with row variation (C1 here).
        assert_eq!(rc.cutoff(16).group, Some(1));
        assert_eq!(rc.cutoff(16).row_us, a.row_us);
        assert_eq!(rc.offset(1), 0.);
        assert!(derived.offset(1) > 0.);
        let p = [0.9, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5];
        let selection = rc.select(&[DraftCandidate { id: 9, confidence: &p }], 64);
        let pending = rc.pending.as_ref().unwrap();
        assert!(!pending.requests[0].explore);
        assert_eq!(selection.lengths, [pending.requests[0].passing]);
        assert_eq!(DraftMode::parse("simple-rc"), Some(DraftMode::SimpleRc));
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
