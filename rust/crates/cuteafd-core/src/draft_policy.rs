//! Online draft verification-length policy, shared by every speculating
//! family. A family binds it with a [`PolicyGeometry`]: its layers and their
//! expert resources, the router's expert count and top-k, the lane's row
//! budget, the drafter's widths and the lane regimes.
//!
//! Each round, a lane verifies an anchor row plus a chosen number of draft
//! rows per request. The policy chooses those lengths to maximize expected
//! committed tokens per unit of predicted lane time. Work that each committed
//! token needs anyway (its own rows) is paid once whichever round verifies the
//! token, so the ratio's argmax equals minimizing fixed plus wasted resource
//! per committed token.
//!
//! Time is priced from the resources that are actually consumed:
//!
//! * per timed layer, `alpha + beta * rows + (1 / bandwidth) * megabytes`,
//!   where megabytes is the routed-expert weight traffic of that layer. The
//!   traffic is `slice_bytes * sum_e ceil(routes_e / group_rows)`: grouped
//!   slice kernels read each expert's slice once per row group. Each layer
//!   belongs to one resource class (for example coordinator RTX or Spark),
//!   fitted separately;
//! * per round, a residual `A + B * rows + C * requests` for everything else
//!   (embedding and untimed layers, vocabulary head, sampling, commit);
//! * per draft pass, `D + E * requests + wide * (F + G * requests)`.
//!
//! Every coefficient is fitted online from the lane's own timings, with
//! exponential forgetting and Huber-weighted residuals, around weak physical
//! priors. Changing the quantization or tensor-parallel width changes the
//! known byte counts; the fitted bandwidths absorb clocks, thermal state and
//! kernel efficiency. No offline calibration is used.
//!
//! Expert traffic for unexecuted draft rows is forecast from each request's
//! own recent committed routes: draft row `k` stands in for the `k`-th most
//! recent committed token, averaged over several shifted windows. Cross-request
//! sharing within a lane falls out of the exact multiset union.
//!
//! This module performs no device work.
use std::collections::{BTreeMap, VecDeque};

/// Committed tokens retained per request.
const HISTORY: usize = 24;
/// Shifted windows averaged by the forecast.
const WINDOWS: usize = 4;
/// Layer samples each resource class needs before the policy engages.
const WARM_LAYER_SAMPLES: u64 = 120;
/// Round samples needed before the policy engages.
const WARM_ROUND_SAMPLES: u64 = 12;
/// Per-sample forgetting of the calibration curvature: memory of roughly 500
/// reached samples per position, so the fit tracks drift.
const CALIBRATION_DECAY: f64 = 0.998;
/// Initial curvature (prior strength) of each position's calibration.
const CALIBRATION_PRIOR: f64 = 2.;
/// Per-round weight of the mean prediction-residual correction.
const BIAS_RATE: f64 = 0.02;
/// Weight of the newest round in each request's recent calibrated confidence.
const RECENT_RATE: f64 = 0.5;
/// Rounds without one draft width before it is tried once again, keeping its
/// draft-cost fit and (for the wide width) its position calibration fresh.
const WIDTH_EXPLORE_ROUNDS: u64 = 64;
/// Draft rounds of each width needed before width choice uses the fits.
const WARM_WIDTH_SAMPLES: u64 = 6;
/// Per-round decay of each request's outcome counts past the native block.
const OUTCOME_DECAY: f64 = 0.7;
/// Prior weight, in reached samples, of the global calibrated rate when
/// blending a request's own outcomes past the native block.
const OUTCOME_PRIOR: f64 = 2.;
/// Weight of one sample that the physical priors are worth: they only resolve
/// directions the data cannot, such as the intercept/row split while every
/// round has the same shape. Rows and expert traffic are strongly correlated,
/// so any stronger prior biases the fitted bandwidth along that near-null
/// direction.
const PRIOR_SAMPLES: f64 = 0.01;
/// Resource classes a geometry may name.
pub const MAX_RESOURCE_CLASSES: usize = 4;

/// Policy failures. Each is a caller or binding error; none is fatal to
/// serving, which falls back to verifying every draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DraftPolicyError {
    #[error("invalid draft policy geometry: {0}")]
    Geometry(&'static str),
    #[error("policy requires one to {0} requests")]
    Requests(usize),
    #[error("invalid conditional draft confidence")]
    Confidence,
    #[error("route capture does not match the verified rows")]
    RouteShape,
    #[error("invalid verified request extent")]
    RequestExtent,
    #[error("route expert exceeds the model")]
    RouteExpert,
}

/// What one family's lane looks like to the policy. The policy holds no
/// family constants; a binding passes its own geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyGeometry {
    /// Every backbone layer, in order.
    pub layers: Vec<LayerResource>,
    /// Routed experts per layer; route ids are `u16`.
    pub experts: u16,
    /// Routes per token and layer.
    pub topk: u8,
    /// The lane's row budget in requests.
    pub max_requests: usize,
    /// Draft positions per request.
    pub max_positions: usize,
    /// Draft widths the drafter can switch between, ascending: one entry is a
    /// fixed width, two let the policy choose per round. The first is the
    /// drafter's native block, past which a request's own outcomes are blended
    /// into its confidence.
    pub widths: Vec<usize>,
    /// Resource classes, fitted separately; [`LayerResource::class`] indexes them.
    pub classes: Vec<ResourceClass>,
    /// Lane regimes, fitted separately: 1 for a single-lane loop, 2 when a
    /// lane may run beside an active peer lane (`shared`).
    pub regimes: usize,
}

/// One backbone layer's routed-expert resource.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerResource {
    /// Resource class of the layer's routed experts; `None` for a dense layer.
    pub class: Option<u8>,
    /// Bytes one device reads for one expert's gate, up and down slices in
    /// this format and tensor-parallel width.
    pub slice_bytes: f64,
    /// Rows per weight-read group of the installed expert kernel.
    pub group_rows: u8,
    /// The layer has its own timing sample. An untimed layer's time and
    /// traffic belong to the round residual.
    pub timed: bool,
}

/// One resource class: a monitoring label and its weak physical prior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResourceClass {
    pub label: &'static str,
    /// Prior per-layer intercept µs, µs per row and µs per MB.
    pub prior: [f64; 3],
    /// Precision of each prior coefficient, in samples per squared unit.
    pub prior_precision: [f64; 3],
}

impl ResourceClass {
    /// Coordinator RTX PRO 6000 slices at ~1.2 TB/s effective.
    pub fn rtx() -> Self {
        Self { label: "local", prior: [700., 15., 0.8],
            prior_precision: [PRIOR_SAMPLES, PRIOR_SAMPLES * 36., PRIOR_SAMPLES * 300f64.powi(2)] }
    }
    /// GB10 Spark expert ranks behind the network at ~175 GB/s effective.
    pub fn spark() -> Self {
        Self { label: "remote", prior: [900., 15., 5.7],
            prior_precision: [PRIOR_SAMPLES, PRIOR_SAMPLES * 36., PRIOR_SAMPLES * 100f64.powi(2)] }
    }
}

/// A timed layer with routed experts: the unit the layer fits price.
#[derive(Clone, Copy, Debug)]
struct Priced {
    layer: usize,
    class: usize,
    slice_bytes: f64,
    group_rows: u8,
}

impl PolicyGeometry {
    fn validate(&self) -> Result<(), DraftPolicyError> {
        let invalid = |reason| Err(DraftPolicyError::Geometry(reason));
        if self.layers.is_empty() { return invalid("no layers"); }
        if self.experts == 0 || self.topk == 0 { return invalid("no routed experts"); }
        if self.max_requests == 0 { return invalid("zero request budget"); }
        if self.max_positions == 0 { return invalid("zero draft positions"); }
        if self.classes.is_empty() || self.classes.len() > MAX_RESOURCE_CLASSES {
            return invalid("one to four resource classes");
        }
        if !(1..=2).contains(&self.regimes) { return invalid("one or two lane regimes"); }
        // The draft-cost fit prices one wide width beside the native one.
        if self.widths.is_empty() || self.widths.len() > 2 { return invalid("one or two draft widths"); }
        if self.widths[0] == 0 || self.widths.windows(2).any(|w| w[0] >= w[1])
            || self.widths.iter().any(|&w| w > self.max_positions) {
            return invalid("draft widths must ascend within the draft positions");
        }
        for layer in &self.layers {
            let Some(class) = layer.class else { continue };
            if usize::from(class) >= self.classes.len() { return invalid("layer names an unknown resource class"); }
            if !layer.slice_bytes.is_finite() || layer.slice_bytes <= 0. {
                return invalid("expert slice bytes must be finite and positive");
            }
            if layer.group_rows == 0 { return invalid("zero rows per weight-read group"); }
        }
        Ok(())
    }
    fn priced(&self) -> Vec<Priced> {
        self.layers.iter().enumerate().filter(|(_, l)| l.timed).filter_map(|(layer, l)| l.class.map(|class|
            Priced { layer, class: usize::from(class), slice_bytes: l.slice_bytes, group_rows: l.group_rows }))
            .collect()
    }
    /// Narrow (native) and wide draft widths; equal for a fixed width.
    fn narrow_wide(&self) -> (usize, usize) {
        (self.widths[0], self.widths[self.widths.len() - 1])
    }
}

/// Exponentially forgotten, prior-regularized, Huber-weighted least squares
/// with nonnegative coefficients.
#[derive(Clone, Debug)]
struct Estimator<const D: usize> {
    decay: f64,
    sxx: [[f64; D]; D],
    sxy: [f64; D],
    prior_mean: [f64; D],
    prior_precision: [f64; D],
    theta: [f64; D],
    scale: f64,
    samples: u64,
}

impl<const D: usize> Estimator<D> {
    fn new(decay: f64, prior_mean: [f64; D], prior_precision: [f64; D]) -> Self {
        Self {
            decay,
            sxx: [[0.; D]; D],
            sxy: [0.; D],
            prior_mean,
            prior_precision,
            theta: prior_mean,
            scale: 0.,
            samples: 0,
        }
    }
    fn predict(&self, x: &[f64; D]) -> f64 {
        x.iter().zip(&self.theta).map(|(x, t)| x * t).sum()
    }
    fn observe(&mut self, x: [f64; D], y: f64) {
        if !y.is_finite() || y < 0. || x.iter().any(|v| !v.is_finite()) {
            return;
        }
        let residual = y - self.predict(&x);
        let magnitude = residual.abs();
        let settled = self.samples >= 8 && self.scale > 0.;
        // Graph capture, prefill interleave and scheduler stalls produce rare
        // large positive residuals; they must not drag the fit.
        let weight = if settled && magnitude > 3. * self.scale {
            3. * self.scale / magnitude
        } else {
            1.
        };
        self.scale = if self.samples == 0 {
            magnitude.max(1.)
        } else {
            let clipped = if settled { magnitude.min(6. * self.scale) } else { magnitude };
            (0.97 * self.scale + 0.03 * clipped).max(1e-3)
        };
        for i in 0..D {
            for j in 0..D {
                self.sxx[i][j] = self.decay * self.sxx[i][j] + weight * x[i] * x[j];
            }
            self.sxy[i] = self.decay * self.sxy[i] + weight * x[i] * y;
        }
        self.samples += 1;
        self.solve();
    }
    /// Solve the regularized normal equations, holding coefficients that would
    /// turn negative at zero (active set, at most D passes).
    fn solve(&mut self) {
        let mut fixed = [false; D];
        for _ in 0..D {
            let mut a = [[0.; D]; D];
            let mut b = [0.; D];
            for i in 0..D {
                if fixed[i] {
                    a[i][i] = 1.;
                    continue;
                }
                for j in 0..D {
                    if !fixed[j] {
                        a[i][j] = self.sxx[i][j];
                    }
                }
                a[i][i] += self.prior_precision[i];
                b[i] = self.sxy[i] + self.prior_precision[i] * self.prior_mean[i];
            }
            let Some(solution) = solve_linear(a, b) else { return };
            match (0..D).find(|&i| !fixed[i] && solution[i] < 0.) {
                Some(i) => fixed[i] = true,
                None => {
                    self.theta = solution;
                    return;
                }
            }
        }
    }
}

fn solve_linear<const D: usize>(mut a: [[f64; D]; D], mut b: [f64; D]) -> Option<[f64; D]> {
    for column in 0..D {
        let pivot = (column..D).max_by(|&x, &y| a[x][column].abs().total_cmp(&a[y][column].abs()))?;
        if a[pivot][column].abs() < 1e-300 {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        for row in column + 1..D {
            let factor = a[row][column] / a[column][column];
            for k in column..D {
                a[row][k] -= factor * a[column][k];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = [0.; D];
    for row in (0..D).rev() {
        let tail: f64 = (row + 1..D).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Online Platt scaling `logit' = a * logit + b` by an online Newton step on
/// the log-loss, with a decayed Fisher-information matrix: effectively a
/// running maximum-likelihood fit over the last few hundred samples.
#[derive(Clone, Copy, Debug)]
struct Platt {
    theta: [f64; 2],
    information: [[f64; 2]; 2],
}

impl Platt {
    fn new() -> Self {
        Self { theta: [1., 0.], information: [[CALIBRATION_PRIOR, 0.], [0., CALIBRATION_PRIOR]] }
    }
    fn logit(probability: f64) -> f64 {
        let p = probability.clamp(1e-6, 1. - 1e-6);
        (p / (1. - p)).ln()
    }
    fn apply(&self, probability: f64) -> f64 {
        let z = self.theta[0] * Self::logit(probability) + self.theta[1];
        1. / (1. + (-z).exp())
    }
    fn observe(&mut self, probability: f64, outcome: f64) {
        let x = [Self::logit(probability), 1.];
        let p = self.apply(probability);
        let curvature = (p * (1. - p)).max(0.02);
        for i in 0..2 {
            for j in 0..2 {
                self.information[i][j] = CALIBRATION_DECAY * self.information[i][j] + curvature * x[i] * x[j];
            }
            // Keep the matrix well conditioned when one direction is unexcited
            // (constant raw confidence).
            self.information[i][i] = self.information[i][i].max(1e-3);
        }
        let [[a, b], [c, d]] = self.information;
        let determinant = a * d - b * c;
        if !(determinant.is_finite() && determinant > 1e-12) { return; }
        let gradient = [(outcome - p) * x[0], outcome - p];
        let step = [(d * gradient[0] - b * gradient[1]) / determinant,
            (a * gradient[1] - c * gradient[0]) / determinant];
        // Slope in [0, 3]: calibration may flatten but never invert confidence.
        self.theta[0] = (self.theta[0] + step[0].clamp(-1., 1.)).clamp(0., 3.);
        self.theta[1] = (self.theta[1] + step[1].clamp(-2., 2.)).clamp(-8., 8.);
    }
}

/// Fitted coefficients for one regime, exported for monitoring.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftCostSnapshot {
    /// Per class: intercept µs, µs per row, µs per MB, samples, residual scale µs.
    pub layers: Vec<[f64; 5]>,
    /// Round residual: intercept µs, µs per row, µs per request, samples, scale.
    pub round: [f64; 5],
    /// Draft pass: intercept µs, µs per request, wide extra µs, wide extra µs
    /// per request, samples, scale.
    pub draft: [f64; 6],
}

/// Megabytes of expert traffic per resource class.
type ClassBytes = [f64; MAX_RESOURCE_CLASSES];

#[derive(Clone, Debug)]
struct CostModel {
    /// [regime][class], x = [1, rows, MB].
    layer: Vec<Vec<Estimator<3>>>,
    /// [regime], x = [1, rows, requests]; excludes the draft pass.
    round: Vec<Estimator<3>>,
    /// [regime] draft pass, x = [1, requests, wide, wide * requests].
    draft: Vec<Estimator<4>>,
    /// [regime][narrow, wide] draft passes observed.
    draft_samples: Vec<[u64; 2]>,
    /// Priced layers per class.
    layers_in: Vec<usize>,
}

impl CostModel {
    fn new(geometry: &PolicyGeometry, priced: &[Priced]) -> Self {
        let n0 = PRIOR_SAMPLES;
        let layer: Vec<_> = geometry.classes.iter()
            .map(|class| Estimator::new(0.998, class.prior, class.prior_precision)).collect();
        let round = Estimator::new(0.98, [5000., 300., 200.], [n0, n0 * 36., n0]);
        let draft = Estimator::new(0.98, [2500., 100., 400., 50.], [n0, n0 * 16., n0, n0 * 16.]);
        Self {
            layer: vec![layer; geometry.regimes],
            round: vec![round; geometry.regimes],
            draft: vec![draft; geometry.regimes],
            draft_samples: vec![[0; 2]; geometry.regimes],
            layers_in: (0..geometry.classes.len()).map(|c| priced.iter().filter(|p| p.class == c).count()).collect(),
        }
    }
    fn warm(&self, regime: usize) -> bool {
        self.round[regime].samples >= WARM_ROUND_SAMPLES
            && (0..self.layers_in.len()).all(|c| self.layers_in[c] == 0 || self.layer[regime][c].samples >= WARM_LAYER_SAMPLES)
    }
    /// The regime's own fit once warm, else another regime's warm fit.
    fn usable_regime(&self, regime: usize) -> Option<usize> {
        std::iter::once(regime).chain((0..self.round.len()).filter(|&r| r != regime)).find(|&r| self.warm(r))
    }
    fn predict(&self, regime: usize, rows: f64, requests: f64, megabytes: ClassBytes) -> f64 {
        let mut total = self.round[regime].predict(&[1., rows, requests]);
        for class in 0..self.layers_in.len() {
            let theta = &self.layer[regime][class].theta;
            let layers = self.layers_in[class] as f64;
            total += layers * (theta[0] + theta[1] * rows) + theta[2] * megabytes[class];
        }
        total
    }
    /// Predicted µs of the draft pass.
    fn draft_us(&self, regime: usize, requests: f64, wide: bool) -> f64 {
        let wide = f64::from(u8::from(wide));
        self.draft[regime].predict(&[1., requests, wide, wide * requests])
    }
    fn widths_warm(&self, regime: usize) -> bool {
        self.draft_samples[regime].iter().all(|&n| n >= WARM_WIDTH_SAMPLES)
    }
    /// Marginal µs of one more row plus `megabytes` of new weight traffic.
    fn marginal(&self, regime: usize, megabytes: ClassBytes) -> f64 {
        let mut total = self.round[regime].theta[1];
        for class in 0..self.layers_in.len() {
            let theta = &self.layer[regime][class].theta;
            total += self.layers_in[class] as f64 * theta[1] + theta[2] * megabytes[class];
        }
        total
    }
    fn snapshot(&self, regime: usize) -> DraftCostSnapshot {
        let pack = |e: &Estimator<3>| [e.theta[0], e.theta[1], e.theta[2], e.samples as f64, e.scale];
        let draft = &self.draft[regime];
        DraftCostSnapshot {
            layers: self.layer[regime].iter().map(pack).collect(),
            round: pack(&self.round[regime]),
            draft: [draft.theta[0], draft.theta[1], draft.theta[2], draft.theta[3], draft.samples as f64, draft.scale],
        }
    }
}

/// One request's candidate: conditional acceptance probability of each
/// available draft position, in order (sigmoid of the draft confidence).
#[derive(Clone, Copy, Debug)]
pub struct DraftCandidate<'a> {
    pub id: u64,
    pub confidence: &'a [f64],
}

#[derive(Clone, Debug, PartialEq)]
pub struct DraftSelection {
    /// Draft rows retained per request, excluding the anchor.
    pub lengths: Vec<usize>,
    pub expected_tokens: f64,
    pub predicted_us: f64,
    /// Shapes evaluated along the forward trajectory.
    pub evaluated: usize,
}

/// One verified request of a completed round.
#[derive(Clone, Copy, Debug)]
pub struct ObservedDraftRequest<'a> {
    pub id: u64,
    /// Verifier rows of this request, including its anchor.
    pub rows: usize,
    /// Accepted inputs, including the anchor (at least one).
    pub accepted: usize,
    /// Conditional confidence of the verified draft positions, if known.
    pub confidence: Option<&'a [f64]>,
}

/// Timing and routes of one completed lane round.
#[derive(Clone, Copy, Debug)]
pub struct DraftRoundObservation<'a> {
    /// Another lane had active requests during this round.
    pub shared: bool,
    pub requests: &'a [ObservedDraftRequest<'a>],
    /// Expert ids, layer-major `[layer][row][slot]` over every layer of the
    /// geometry and every lane row in request order.
    pub routes: &'a [u16],
    /// Wall µs of each timed layer, by layer index (missing entries are
    /// `None`), each from the previous layer's FFN completion to its own.
    pub layer_us: &'a [Option<f64>],
    /// Round wall µs from draft start to commit completion.
    pub total_us: f64,
    /// Wall µs of the draft pass (NaN if none ran), and whether it used the
    /// wide width.
    pub draft_us: f64,
    pub wide: bool,
    /// The policy's prediction for this round's shape, if it made one.
    pub predicted_us: Option<f64>,
}

/// Cumulative serving statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftPolicyStats {
    pub rounds: u64,
    pub selected_rounds: u64,
    pub verified_rows: u64,
    pub verified_drafts: u64,
    pub accepted_drafts: u64,
    pub emitted_requests: u64,
    /// Requests verified with k draft rows, k = 0..=max_positions.
    pub draft_rows: Vec<u64>,
    /// Per draft position: times reached with all earlier drafts accepted,
    /// summed conditional confidence there, and acceptances there.
    pub position_reached: Vec<u64>,
    /// Summed calibrated confidence where reached (what the policy used).
    pub position_confidence: Vec<f64>,
    /// Summed raw drafter confidence where reached.
    pub position_raw_confidence: Vec<f64>,
    pub position_accepted: Vec<u64>,
    /// Draft passes at the narrow and wide width.
    pub width_rounds: [u64; 2],
    pub predicted_rounds: u64,
    pub prediction_error_us: f64,
    pub prediction_abs_error_us: f64,
    pub observed_us: f64,
}

impl DraftPolicyStats {
    fn new(positions: usize) -> Self {
        Self { rounds: 0, selected_rounds: 0, verified_rows: 0, verified_drafts: 0, accepted_drafts: 0,
            emitted_requests: 0, draft_rows: vec![0; positions + 1], position_reached: vec![0; positions],
            position_confidence: vec![0.; positions], position_raw_confidence: vec![0.; positions],
            position_accepted: vec![0; positions], width_rounds: [0; 2], predicted_rounds: 0,
            prediction_error_us: 0., prediction_abs_error_us: 0., observed_us: 0. }
    }
}

/// Whole-policy state: route history, cost fit, and statistics.
#[derive(Clone, Debug)]
pub struct DraftPolicy {
    geometry: PolicyGeometry,
    priced: Vec<Priced>,
    fixed: bool,
    cost: CostModel,
    /// Per request, the routes of recent committed tokens, newest first, each
    /// `layers * topk` expert ids.
    history: BTreeMap<u64, VecDeque<Box<[u16]>>>,
    /// Expected new expert groups per layer contributed by a token whose routes
    /// are not yet known (short histories), from observed novelty.
    novelty: Vec<f64>,
    /// Per-position Platt calibration of the drafter's confidence. Confidence
    /// is conditional on earlier positions being accepted, and its reliability
    /// varies by position: past the drafter's native block it saturates near
    /// 0.99 whatever the content, so a slope as well as an offset is needed.
    /// Each position is refitted online from the outcomes of the rounds that
    /// reached it.
    calibration: Vec<Platt>,
    /// Mean residual of the predicted round time per regime. The robust fit
    /// tracks typical rounds, while throughput depends on mean time including
    /// stalls; this constant restores the mean without moving the marginals.
    bias: Vec<f64>,
    /// Per request, short-memory calibrated confidence per drafted position.
    recent: BTreeMap<u64, Vec<Option<f64>>>,
    /// Per request, decayed (accepted, reached) counts per position. Past the
    /// drafter's native block its confidence carries no content signal, so a
    /// request's own recent outcomes there (long accepted runs in code, short
    /// ones in prose) are the only request-specific evidence.
    outcomes: BTreeMap<u64, Vec<(f64, f64)>>,
    /// Long-memory mean raw confidence per position, for requests without
    /// recent evidence at that position.
    raw_mean: Vec<f64>,
    /// Draft passes since each width (narrow, wide) last ran.
    since_width: [u64; 2],
    stats: DraftPolicyStats,
    /// Routes per (window, layer, expert) in the shape being evaluated.
    counts: Vec<u8>,
    /// Routes per expert of one observed layer.
    scratch: Vec<u16>,
}

impl DraftPolicy {
    /// `fixed` verifies every available draft but still fits and reports.
    pub fn new(geometry: PolicyGeometry, fixed: bool) -> Result<Self, DraftPolicyError> {
        geometry.validate()?;
        Ok(Self::build(geometry, fixed))
    }
    fn build(geometry: PolicyGeometry, fixed: bool) -> Self {
        let priced = geometry.priced();
        let positions = geometry.max_positions;
        Self {
            cost: CostModel::new(&geometry, &priced),
            priced,
            fixed,
            history: BTreeMap::new(),
            novelty: vec![2.5; geometry.layers.len()],
            calibration: vec![Platt::new(); positions],
            bias: vec![0.; geometry.regimes],
            recent: BTreeMap::new(),
            outcomes: BTreeMap::new(),
            raw_mean: vec![0.9; positions],
            since_width: [0; 2],
            stats: DraftPolicyStats::new(positions),
            counts: vec![0; WINDOWS * geometry.layers.len() * usize::from(geometry.experts)],
            scratch: vec![0; usize::from(geometry.experts)],
            geometry,
        }
    }
    /// A fresh policy over the same geometry.
    pub fn restarted(&self, fixed: bool) -> Self {
        Self::build(self.geometry.clone(), fixed)
    }
    pub fn fixed(&self) -> bool {
        self.fixed
    }
    pub fn geometry(&self) -> &PolicyGeometry {
        &self.geometry
    }
    pub fn stats(&self) -> &DraftPolicyStats {
        &self.stats
    }
    fn regime(&self, shared: bool) -> usize {
        usize::from(shared).min(self.geometry.regimes - 1)
    }
    pub fn cost_snapshot(&self, shared: bool) -> DraftCostSnapshot {
        self.cost.snapshot(self.regime(shared))
    }
    pub fn warm(&self, shared: bool) -> bool {
        self.cost.warm(self.regime(shared))
    }
    pub fn release(&mut self, id: u64) {
        self.history.remove(&id);
        self.recent.remove(&id);
        self.outcomes.remove(&id);
    }
    /// Draft widths the drafter can switch between.
    pub fn widths(&self) -> &[usize] {
        &self.geometry.widths
    }
    /// Learned (slope, offset) on the raw logit, per draft position.
    pub fn calibration(&self) -> Vec<(f64, f64)> {
        self.calibration.iter().map(|platt| (platt.theta[0], platt.theta[1])).collect()
    }
    /// Mean prediction-residual correction per regime (solo, shared), µs.
    pub fn time_bias(&self) -> &[f64] {
        &self.bias
    }
    fn calibrated(&self, position: usize, probability: f64) -> f64 {
        self.calibration[position].apply(probability)
    }
    /// Acceptance probability of `position` for request `id`: the calibrated
    /// confidence inside the native block, and past it the request's own
    /// recent outcomes shrunk toward that calibrated rate.
    fn request_probability(&self, id: u64, position: usize, calibrated: f64) -> f64 {
        if position < self.geometry.widths[0] { return calibrated; }
        let (hits, trials) = self.outcomes.get(&id).map_or((0., 0.), |o| o[position]);
        (hits + OUTCOME_PRIOR * calibrated) / (trials + OUTCOME_PRIOR)
    }

    /// Predicted lane µs for explicit lengths after a draft of `width`, if the
    /// fit is usable.
    pub fn predict(&mut self, shared: bool, ids: &[u64], lengths: &[usize], width: usize) -> Option<f64> {
        let regime = self.cost.usable_regime(self.regime(shared))?;
        self.clear_counts();
        let mut megabytes = [0.; MAX_RESOURCE_CLASSES];
        for (index, (&id, &length)) in ids.iter().zip(lengths).enumerate() {
            for row in 0..=length {
                add_bytes(&mut megabytes, self.add_row(id, row, index));
            }
        }
        let rows = ids.len() + lengths.iter().sum::<usize>();
        let draft = self.cost.draft_us(regime, ids.len() as f64, width > self.geometry.widths[0]);
        Some(self.cost.predict(regime, rows as f64, ids.len() as f64, megabytes) + draft + self.bias[regime])
    }

    /// Choose draft lengths for one lane after a draft of `width`. Returns
    /// `None` while the fit is still warming up or in fixed mode; the caller
    /// then verifies every draft.
    pub fn select(&mut self, shared: bool, candidates: &[DraftCandidate<'_>], width: usize)
        -> Result<Option<DraftSelection>, DraftPolicyError> {
        if candidates.is_empty() || candidates.len() > self.geometry.max_requests {
            return Err(DraftPolicyError::Requests(self.geometry.max_requests));
        }
        for candidate in candidates {
            if candidate.confidence.len() > self.geometry.max_positions
                || candidate.confidence.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
                return Err(DraftPolicyError::Confidence);
            }
        }
        if self.fixed {
            return Ok(None);
        }
        let Some(regime) = self.cost.usable_regime(self.regime(shared)) else { return Ok(None) };
        // Survival of row k: the product of calibrated conditional acceptance
        // probabilities of positions 1..=k.
        let cumulative: Vec<Vec<f64>> = candidates.iter().map(|c| {
            let mut product = 1.;
            c.confidence.iter().enumerate().map(|(position, &p)| {
                product *= self.request_probability(c.id, position, self.calibrated(position, p));
                product
            }).collect()
        }).collect();
        let ids: Vec<_> = candidates.iter().map(|c| c.id).collect();
        let draft = self.cost.draft_us(regime, ids.len() as f64, width > self.geometry.widths[0]);
        Ok(Some(self.select_core(regime, &ids, &cumulative, draft)))
    }

    /// Choose the draft width for the lane's next round from each request's
    /// recent calibrated confidence. `requests` are (identity, most drafts the
    /// request can use). Both widths are scored by the same length selector,
    /// each charged its own predicted draft pass; the better expected ratio
    /// wins, ties going to the narrow width.
    pub fn choose_width(&mut self, shared: bool, requests: &[(u64, usize)]) -> usize {
        let (narrow, wide) = self.geometry.narrow_wide();
        if narrow == wide || requests.is_empty() { return narrow; }
        if self.fixed { return wide; }
        let regime = self.cost.usable_regime(self.regime(shared));
        let Some(regime) = regime.filter(|&r| self.cost.widths_warm(r)) else {
            // Warm up both draft-cost fits by alternating widths.
            return if self.stats.width_rounds[0] <= self.stats.width_rounds[1] { narrow } else { wide };
        };
        if self.since_width[1] >= WIDTH_EXPLORE_ROUNDS { return wide; }
        if self.since_width[0] >= WIDTH_EXPLORE_ROUNDS { return narrow; }
        let ids: Vec<_> = requests.iter().map(|r| r.0).collect();
        let mut best: Option<(usize, f64)> = None;
        for width in [narrow, wide] {
            let cumulative: Vec<Vec<f64>> = requests.iter().map(|&(id, available)| {
                let recent = self.recent.get(&id);
                let mut product = 1.;
                (0..width.min(available)).map(|position| {
                    let p = recent.and_then(|r| r[position])
                        .unwrap_or_else(|| self.calibrated(position, self.raw_mean[position]));
                    product *= self.request_probability(id, position, p);
                    product
                }).collect()
            }).collect();
            let draft = self.cost.draft_us(regime, ids.len() as f64, width > narrow);
            let selection = self.select_core(regime, &ids, &cumulative, draft);
            let ratio = selection.expected_tokens / selection.predicted_us;
            if best.is_none_or(|(_, b)| ratio > b) { best = Some((width, ratio)); }
        }
        best.map_or(narrow, |(width, _)| width)
    }

    /// Forward-growth length selection over survival products, with the round's
    /// draft pass priced in.
    fn select_core(&mut self, regime: usize, ids: &[u64], cumulative: &[Vec<f64>], draft_us: f64)
        -> DraftSelection {
        self.clear_counts();
        let mut megabytes = [0.; MAX_RESOURCE_CLASSES];
        for (index, &id) in ids.iter().enumerate() {
            add_bytes(&mut megabytes, self.add_row(id, 0, index));
        }
        let requests = ids.len() as f64;
        let mut rows = ids.len();
        let mut lengths = vec![0usize; ids.len()];
        let mut expected = requests;
        let mut time = self.cost.predict(regime, rows as f64, requests, megabytes) + draft_us + self.bias[regime];
        let mut best = DraftSelection {
            lengths: lengths.clone(), expected_tokens: expected, predicted_us: time, evaluated: 1,
        };
        let mut evaluated = 1;
        loop {
            // Forward growth: evaluate every request's next draft row against
            // the current joint shape and take the best resulting ratio. The
            // trajectory continues through temporary losses to the full shape,
            // and the best visited shape is returned.
            let mut choice: Option<(usize, f64, f64)> = None;
            for (index, &id) in ids.iter().enumerate() {
                let next = lengths[index] + 1;
                if next > cumulative[index].len() { continue; }
                let delta = self.peek_row(id, next);
                let candidate_expected = expected + cumulative[index][next - 1];
                let candidate_time = time + self.cost.marginal(regime, delta);
                evaluated += 1;
                if choice.is_none_or(|(_, e, t)| candidate_expected / candidate_time > e / t) {
                    choice = Some((index, candidate_expected, candidate_time));
                }
            }
            let Some((index, candidate_expected, candidate_time)) = choice else { break };
            lengths[index] += 1;
            rows += 1;
            self.add_row(ids[index], lengths[index], index);
            expected = candidate_expected;
            time = candidate_time;
            // Prefer the longer shape on an exact tie.
            if expected / time >= best.expected_tokens / best.predicted_us {
                best.lengths.clone_from(&lengths);
                best.expected_tokens = expected;
                best.predicted_us = time;
            }
        }
        debug_assert_eq!(rows, ids.len() + lengths.iter().sum::<usize>());
        best.evaluated = evaluated;
        best
    }

    fn clear_counts(&mut self) {
        self.counts.fill(0);
    }

    fn count_index(&self, window: usize, layer: usize, expert: u16) -> usize {
        (window * self.geometry.layers.len() + layer) * usize::from(self.geometry.experts) + usize::from(expert)
    }

    /// Routes of `layer` in the token standing in for row `row` of request
    /// `id` in window `window`.
    fn stand_in(&self, id: u64, row: usize, window: usize) -> Option<&[u16]> {
        self.history.get(&id)?.get(window + row).map(|token| &token[..])
    }

    /// Megabytes of new weight traffic per class if row `row` of `id` joined
    /// the current shape, averaged over the forecast windows.
    fn peek_row(&self, id: u64, row: usize) -> ClassBytes {
        let topk = usize::from(self.geometry.topk);
        let mut total = [0.; MAX_RESOURCE_CLASSES];
        for window in 0..WINDOWS {
            let routes = self.stand_in(id, row, window);
            for priced in &self.priced {
                let groups = match routes {
                    Some(routes) => routes[priced.layer * topk..][..topk].iter()
                        .filter(|&&e| self.counts[self.count_index(window, priced.layer, e)] % priced.group_rows == 0)
                        .count() as f64,
                    None if row == 0 => topk as f64,
                    None => self.novelty[priced.layer],
                };
                total[priced.class] += groups * priced.slice_bytes;
            }
        }
        total.map(|bytes| bytes / WINDOWS as f64 / 1e6)
    }

    fn add_row(&mut self, id: u64, row: usize, _request: usize) -> ClassBytes {
        let delta = self.peek_row(id, row);
        let topk = usize::from(self.geometry.topk);
        for window in 0..WINDOWS {
            let Some(token) = self.history.get(&id).and_then(|h| h.get(window + row)) else { continue };
            for priced in &self.priced {
                for &expert in &token[priced.layer * topk..][..topk] {
                    let index = (window * self.geometry.layers.len() + priced.layer)
                        * usize::from(self.geometry.experts) + usize::from(expert);
                    self.counts[index] = self.counts[index].saturating_add(1);
                }
            }
        }
        delta
    }

    /// Weight-read groups of one layer of an observed round: each expert's
    /// slice is read once per `group_rows` routed rows.
    fn route_groups(&mut self, routes: &[u16], group_rows: u8) -> f64 {
        self.scratch.fill(0);
        for &expert in routes {
            if let Some(count) = self.scratch.get_mut(usize::from(expert)) { *count += 1; }
        }
        self.scratch.iter().map(|&c| u32::from(c.div_ceil(u16::from(group_rows)))).sum::<u32>() as f64
    }

    /// Record a completed round: timings update the cost fit, accepted rows'
    /// routes extend each request's history, acceptance updates statistics.
    pub fn observe(&mut self, round: DraftRoundObservation<'_>) -> Result<(), DraftPolicyError> {
        let layers = self.geometry.layers.len();
        let topk = usize::from(self.geometry.topk);
        let positions = self.geometry.max_positions;
        let rows: usize = round.requests.iter().map(|r| r.rows).sum();
        if round.routes.len() != layers * rows * topk {
            return Err(DraftPolicyError::RouteShape);
        }
        if round.requests.iter().any(|r| r.rows == 0 || r.accepted == 0 || r.accepted > r.rows) {
            return Err(DraftPolicyError::RequestExtent);
        }
        let regime = self.regime(round.shared);
        let requests = round.requests.len() as f64;
        let layer_routes = |layer: usize| &round.routes[layer * rows * topk..][..rows * topk];
        let mut layer_sum = 0.;
        let mut complete = true;
        for index in 0..self.priced.len() {
            let priced = self.priced[index];
            let Some(elapsed) = round.layer_us.get(priced.layer).copied().flatten() else { complete = false; continue };
            layer_sum += elapsed;
            let megabytes = self.route_groups(layer_routes(priced.layer), priced.group_rows) * priced.slice_bytes / 1e6;
            self.cost.layer[regime][priced.class].observe([1., rows as f64, megabytes], elapsed);
        }
        let drafted = round.draft_us.is_finite();
        let draft_us = if drafted { round.draft_us } else { 0. };
        if complete && round.total_us.is_finite() {
            self.cost.round[regime].observe([1., rows as f64, requests],
                (round.total_us - layer_sum - draft_us).max(0.));
        }
        if drafted {
            let wide = usize::from(round.wide);
            let flag = wide as f64;
            self.cost.draft[regime].observe([1., requests, flag, flag * requests], round.draft_us);
            self.cost.draft_samples[regime][wide] += 1;
            self.stats.width_rounds[wide] += 1;
            self.since_width[wide] = 0;
            self.since_width[1 - wide] += 1;
        }
        let mut offset = 0;
        for request in round.requests {
            let history = self.history.entry(request.id).or_default();
            for row in offset..offset + request.accepted {
                let mut token = vec![0u16; layers * topk].into_boxed_slice();
                for layer in 0..layers {
                    let routes = &layer_routes(layer)[row * topk..][..topk];
                    if routes.iter().any(|&expert| expert >= self.geometry.experts) {
                        return Err(DraftPolicyError::RouteExpert);
                    }
                    token[layer * topk..][..topk].copy_from_slice(routes);
                }
                if history.len() >= 3 {
                    for priced in &self.priced {
                        let layer = priced.layer;
                        let novel = token[layer * topk..][..topk].iter()
                            .filter(|e| !history.iter().take(3).any(|t| t[layer * topk..][..topk].contains(e)))
                            .count();
                        self.novelty[layer] = 0.995 * self.novelty[layer] + 0.005 * novel as f64;
                    }
                }
                history.push_front(token);
                history.truncate(HISTORY);
            }
            offset += request.rows;
            let drafts = request.rows - 1;
            let accepted = request.accepted - 1;
            self.stats.verified_drafts += drafts as u64;
            self.stats.accepted_drafts += accepted as u64;
            self.stats.emitted_requests += 1;
            self.stats.draft_rows[drafts.min(positions)] += 1;
            if let Some(confidence) = request.confidence {
                // Every drafted position's calibrated confidence feeds the
                // request's short-memory estimate used to choose the next
                // round's width, whether or not the position was verified.
                let recent = self.recent.entry(request.id).or_insert_with(|| vec![None; positions]);
                for (position, &raw) in confidence.iter().enumerate().take(positions) {
                    let calibrated = self.calibration[position].apply(raw);
                    recent[position] = Some(match recent[position] {
                        Some(previous) => previous + RECENT_RATE * (calibrated - previous),
                        None => calibrated,
                    });
                    self.raw_mean[position] += 0.02 * (raw - self.raw_mean[position]);
                }
                // Only positions whose predecessors were all accepted carry
                // evidence about the conditional acceptance probability.
                let reached = drafts.min(confidence.len()).min(accepted + 1).min(positions);
                for position in 0..reached {
                    let outcome = f64::from(u8::from(position < accepted));
                    let calibrated = self.calibrated(position, confidence[position]);
                    self.stats.position_reached[position] += 1;
                    self.stats.position_confidence[position] += calibrated;
                    self.stats.position_raw_confidence[position] += confidence[position];
                    self.stats.position_accepted[position] += u64::from(position < accepted);
                    self.calibration[position].observe(confidence[position], outcome);
                }
                let counts = self.outcomes.entry(request.id).or_insert_with(|| vec![(0., 0.); positions]);
                for count in counts.iter_mut() { count.0 *= OUTCOME_DECAY; count.1 *= OUTCOME_DECAY; }
                for position in 0..reached {
                    counts[position].0 += f64::from(u8::from(position < accepted));
                    counts[position].1 += 1.;
                }
            }
        }
        self.stats.rounds += 1;
        self.stats.verified_rows += rows as u64;
        if let Some(predicted) = round.predicted_us {
            self.stats.selected_rounds += 1;
            if round.total_us.is_finite() {
                // Bounded step: one stalled round cannot swing the correction.
                let residual = (round.total_us - predicted).clamp(-0.5 * predicted.abs(), 0.5 * predicted.abs());
                self.bias[regime] += BIAS_RATE * residual;
                self.stats.predicted_rounds += 1;
                self.stats.prediction_error_us += predicted - round.total_us;
                self.stats.prediction_abs_error_us += (predicted - round.total_us).abs();
                self.stats.observed_us += round.total_us;
            }
        }
        Ok(())
    }
}

fn add_bytes(total: &mut ClassBytes, delta: ClassBytes) {
    for (total, delta) in total.iter_mut().zip(delta) { *total += delta; }
}

/// Expected committed tokens for conditional confidences: the mandatory
/// anchor/bonus plus the cumulative acceptance probability of each draft.
pub fn draft_expected_tokens(confidence: &[f64]) -> f64 {
    let mut product = 1.;
    1. + confidence.iter().map(|p| { product *= p; product }).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The V4.1 geometry as its binding builds it: 40 layers of 384 experts,
    /// top-6, 16-row slice kernels, layer 0 untimed, 16 requests, 7 draft
    /// positions, two regimes, RTX (0) and Spark (1) classes.
    fn v41(local: impl Fn(usize) -> bool, bytes: impl Fn(usize) -> f64, widths: Vec<usize>) -> PolicyGeometry {
        PolicyGeometry {
            layers: (0..40).map(|l| LayerResource { class: Some(u8::from(!local(l))), slice_bytes: bytes(l),
                group_rows: 16, timed: l > 0 }).collect(),
            experts: 384, topk: 6, max_requests: 16, max_positions: 7, widths,
            classes: vec![ResourceClass::rtx(), ResourceClass::spark()], regimes: 2,
        }
    }

    fn placement(local: usize) -> PolicyGeometry {
        v41(|l| l < local, |l| if l < local { 18_800_640. } else { 4_700_160. }, vec![5, 7])
    }

    /// Routes as V4.1 captures them (`[layer][row]`, router flag bits above
    /// the expert index), flattened as its binding passes them.
    fn flat(routes: &[Vec<[u32; 6]>]) -> Vec<u16> {
        routes.iter().flatten().flatten().map(|&e| (e & 511) as u16).collect()
    }

    /// Weight-read groups of one layer by sixteen-row tiles, as the old core
    /// counted them.
    fn groups(rows: &[[u32; 6]]) -> f64 {
        let mut counts = [0u16; 384];
        for route in rows {
            for &expert in route {
                if let Some(count) = counts.get_mut((expert & 511) as usize) { *count += 1; }
            }
        }
        counts.iter().map(|&c| c.div_ceil(16)).sum::<u16>() as f64
    }

    /// Deterministic synthetic router: each token draws six experts from a
    /// per-layer pool so that neighboring tokens share some experts.
    fn token_routes(token: u64, pool: u32) -> Vec<[u32; 6]> {
        (0..40u64).map(|layer| {
            let mut out = [0u32; 6];
            let mut seed = token.wrapping_mul(0x9e3779b97f4a7c15) ^ layer.wrapping_mul(0xbf58476d1ce4e5b9);
            let mut used = 0;
            while used < 6 {
                seed = seed.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = seed;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                let expert = (layer as u32 * 7 + (z % pool as u64) as u32) % 384;
                if !out[..used].contains(&expert) { out[used] = expert; used += 1; }
            }
            out
        }).collect()
    }

    /// Synthetic ground truth used to train the estimator.
    struct Truth { alpha: [f64; 2], beta: [f64; 2], us_per_mb: [f64; 2], round: [f64; 3] }

    fn run_round(policy: &mut DraftPolicy, truth: &Truth, token: &mut u64, rows: usize, accepted: usize,
        predicted: Option<f64>) -> f64 {
        let token_rows: Vec<_> = (0..rows).map(|r| token_routes(*token + r as u64, 24)).collect();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|l| token_rows.iter().map(|t| t[l]).collect()).collect();
        let mut layer_us = [None; 40];
        let mut total = truth.round[0] + truth.round[1] * rows as f64 + truth.round[2];
        for layer in 1..40 {
            let class = policy.geometry.layers[layer].class.unwrap() as usize;
            let mb = groups(&routes[layer]) * policy.geometry.layers[layer].slice_bytes / 1e6;
            let t = truth.alpha[class] + truth.beta[class] * rows as f64 + truth.us_per_mb[class] * mb;
            layer_us[layer] = Some(t);
            total += t;
        }
        let requests = [ObservedDraftRequest { id: 1, rows, accepted, confidence: None }];
        policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
            layer_us: &layer_us, total_us: total, predicted_us: predicted, draft_us: f64::NAN, wide: false }).unwrap();
        *token += accepted as u64;
        total
    }

    fn truth() -> Truth {
        Truth { alpha: [600., 700.], beta: [8., 12.], us_per_mb: [0.9, 7.5], round: [9000., 250., 150.] }
    }

    #[test]
    fn estimator_recovers_coefficients_and_predicts_rounds() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        let truth = truth();
        let mut token = 0;
        for round in 0..400 {
            let rows = 1 + round % 8;
            run_round(&mut policy, &truth, &mut token, rows, rows, None);
        }
        assert!(policy.warm(false));
        let snapshot = policy.cost_snapshot(false);
        let remote = snapshot.layers[1];
        assert!((remote[2] - 7.5).abs() < 0.3, "remote µs/MB {remote:?}");
        assert!((remote[1] - 12.).abs() < 3., "remote µs/row {remote:?}");
        // Prediction of an explicit shape against the same truth.
        let lengths = [4];
        let predicted = policy.predict(false, &[1], &lengths, 5).unwrap();
        let actual = run_round(&mut policy, &truth, &mut token, 5, 5, Some(predicted));
        assert!((predicted - actual).abs() / actual < 0.05, "{predicted} vs {actual}");
    }

    #[test]
    fn noisy_layers_and_capture_outliers_do_not_bias_the_bandwidth() {
        let mut estimator: Estimator<3> = Estimator::new(0.998, [900., 15., 5.7], [0.01, 0.36, 100.]);
        let mut state = 12345u64;
        let mut uniform = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 };
        for sample in 0..20_000u64 {
            let rows = 1 + sample % 8;
            let groups = 6. + 3. * (rows - 1) as f64 + (uniform() * 6.).floor() - 3.;
            let megabytes = groups * 4.70016;
            let mut y = 700. + 12. * rows as f64 + 7.5 * megabytes;
            y *= 1. + 0.1 * (uniform() - 0.5);
            // One layer in fifty stalls on graph capture or a peer prefill.
            if uniform() < 0.02 { y += 20_000. * uniform(); }
            estimator.observe([1., rows as f64, megabytes], y);
        }
        let [alpha, beta, us_per_mb] = estimator.theta;
        assert!((us_per_mb - 7.5).abs() / 7.5 < 0.08, "{:?}", estimator.theta);
        let predicted = alpha + 6. * beta + 7.5 * 0. + us_per_mb * 21. * 4.70016;
        let truth = 700. + 72. + 7.5 * 21. * 4.70016;
        assert!((predicted - truth).abs() / truth < 0.05, "{predicted} vs {truth}");
    }

    #[test]
    fn certain_drafts_are_always_verified_and_hopeless_ones_trimmed() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        let truth = truth();
        let mut token = 0;
        for round in 0..300 { run_round(&mut policy, &truth, &mut token, 1 + round % 6, 1 + round % 6, None); }
        let certain = [1.; 5];
        let selection = policy.select(false, &[DraftCandidate { id: 1, confidence: &certain }], 5).unwrap().unwrap();
        assert_eq!(selection.lengths, [5]);
        assert!((selection.expected_tokens - 6.).abs() < 1e-4);
        let hopeless = [0.02; 5];
        let selection = policy.select(false, &[DraftCandidate { id: 1, confidence: &hopeless }], 5).unwrap().unwrap();
        assert_eq!(selection.lengths, [0]);
        // A falling confidence curve is cut where cumulative acceptance no longer
        // pays for the next row's new expert traffic.
        let falling = [0.95, 0.9, 0.5, 0.3, 0.2];
        let selection = policy.select(false, &[DraftCandidate { id: 1, confidence: &falling }], 5).unwrap().unwrap();
        assert!((1..5).contains(&selection.lengths[0]), "{selection:?}");
    }

    #[test]
    fn selection_matches_exhaustive_search_for_one_request() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        let truth = truth();
        let mut token = 0;
        for round in 0..300 { run_round(&mut policy, &truth, &mut token, 1 + round % 6, 1 + round % 6, None); }
        for confidence in [[0.9, 0.8, 0.7, 0.6, 0.5], [0.6, 0.9, 0.9, 0.9, 0.9], [0.99, 0.2, 0.9, 0.9, 0.9]] {
            let selection = policy.select(false, &[DraftCandidate { id: 1, confidence: &confidence }], 5).unwrap().unwrap();
            let best = (0..=5).max_by(|&a, &b| {
                let ratio = |k: usize| draft_expected_tokens(&confidence[..k]) / policy.clone().predict(false, &[1], &[k], 5).unwrap();
                ratio(a).total_cmp(&ratio(b))
            }).unwrap();
            assert_eq!(selection.lengths, [best], "{confidence:?}");
        }
    }

    #[test]
    fn warmup_and_fixed_mode_verify_everything() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        assert_eq!(policy.select(false, &[DraftCandidate { id: 1, confidence: &[0.1; 5] }], 5), Ok(None));
        let mut fixed = DraftPolicy::new(placement(5), true).unwrap();
        let truth = truth();
        let mut token = 0;
        for _ in 0..300 { run_round(&mut fixed, &truth, &mut token, 6, 3, None); }
        assert!(fixed.warm(false));
        assert_eq!(fixed.select(false, &[DraftCandidate { id: 1, confidence: &[0.1; 5] }], 5), Ok(None));
        assert_eq!(fixed.stats().draft_rows[5], 300);
        assert_eq!(fixed.stats().accepted_drafts, 600);
    }

    #[test]
    fn shared_experts_across_requests_are_charged_once() {
        let mut policy = DraftPolicy::new(placement(0), false).unwrap();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]]).collect();
        let layer_us = [None; 40];
        for id in [1, 2] {
            for _ in 0..4 {
                let requests = [ObservedDraftRequest { id, rows: 1, accepted: 1, confidence: None }];
                policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
                    layer_us: &layer_us, total_us: f64::NAN, predicted_us: None, draft_us: f64::NAN, wide: false }).unwrap();
            }
        }
        policy.clear_counts();
        let first = policy.add_row(1, 0, 0);
        let second = policy.add_row(2, 0, 1);
        assert!(first[1] > 0.);
        assert_eq!(second[1], 0.);
    }

    #[test]
    fn groups_follow_the_kernel_row_tiles() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        let rows: Vec<u16> = [1, 2, 3, 4, 5, 6].repeat(17);
        assert_eq!(policy.route_groups(&rows, 16), 12.);
        assert_eq!(policy.route_groups(&rows[..96], 16), 6.);
        assert_eq!(policy.route_groups(&rows, 8), 18.);
    }

    #[test]
    fn confidence_reliability_counts_only_reached_positions() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]; 5]).collect();
        let confidence = [0.9, 0.8, 0.7, 0.6];
        let requests = [ObservedDraftRequest { id: 1, rows: 5, accepted: 2, confidence: Some(&confidence) }];
        policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
            layer_us: &[None; 40], total_us: f64::NAN, predicted_us: None, draft_us: f64::NAN, wide: false }).unwrap();
        let stats = policy.stats();
        assert_eq!(stats.position_reached, [1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(stats.position_accepted, [1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(stats.draft_rows[4], 1);
    }

    #[test]
    fn overconfident_positions_are_recalibrated_from_reached_outcomes() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]; 8]).collect();
        // The drafter reports 0.95 at every position; position 1 is truly 0.95,
        // position 6 only 0.8.
        let confidence = [0.95; 7];
        let mut state = 99u64;
        let mut uniform = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 };
        for round in 0..20_000 {
            if round == 600 {
                // Position 6 has been reached ~460 times: already close.
                assert!(policy.stats().position_reached[5] < 520);
                assert!((policy.calibrated(5, 0.95) - 0.8).abs() < 0.05, "{:?}", policy.calibration());
            }
            let truth = [0.95, 0.95, 0.95, 0.95, 0.95, 0.8, 0.8];
            let accepted = truth.iter().take_while(|&&p| uniform() < p).count();
            let requests = [ObservedDraftRequest { id: 1, rows: 8, accepted: accepted + 1, confidence: Some(&confidence) }];
            policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
                layer_us: &[None; 40], total_us: f64::NAN, predicted_us: None, draft_us: f64::NAN, wide: false }).unwrap();
        }
        assert!((policy.calibrated(0, 0.95) - 0.95).abs() < 0.02, "{:?}", policy.calibration());
        assert!((policy.calibrated(5, 0.95) - 0.8).abs() < 0.03, "{:?}", policy.calibration());
        // Reliability stats report the calibrated confidence the policy used.
        let stats = policy.stats();
        let mean = stats.position_confidence[5] / stats.position_reached[5] as f64;
        let observed = stats.position_accepted[5] as f64 / stats.position_reached[5] as f64;
        assert!((mean - observed).abs() < 0.03, "{mean} vs {observed}");
    }

    #[test]
    fn saturated_confidence_falls_back_to_its_base_rate() {
        let mut state = 7u64;
        let mut uniform = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 };
        // Uninformative: raw 0.95..0.999 while acceptance is always 0.82.
        let mut flat = Platt::new();
        // Informative: acceptance equals the raw confidence.
        let mut honest = Platt::new();
        for _ in 0..4000 {
            let raw = 0.95 + 0.049 * uniform();
            flat.observe(raw, f64::from(u8::from(uniform() < 0.82)));
            let raw = 0.3 + 0.69 * uniform();
            honest.observe(raw, f64::from(u8::from(uniform() < raw)));
        }
        for raw in [0.95, 0.98, 0.995] {
            assert!((flat.apply(raw) - 0.82).abs() < 0.04, "{raw} -> {} {:?}", flat.apply(raw), flat.theta);
        }
        for raw in [0.4, 0.7, 0.9] {
            assert!((honest.apply(raw) - raw).abs() < 0.05, "{raw} -> {} {:?}", honest.apply(raw), honest.theta);
        }
    }

    #[test]
    fn mean_time_bias_absorbs_skewed_stalls() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        let truth = truth();
        let mut token = 0;
        for round in 0..300 { run_round(&mut policy, &truth, &mut token, 1 + round % 6, 1 + round % 6, None); }
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]]).collect();
        for round in 0..2000 {
            let predicted = policy.predict(false, &[1], &[0], 5).unwrap();
            // Every tenth round stalls by 10 ms: mean excess 1 ms.
            let total = predicted - policy.time_bias()[0] + if round % 10 == 0 { 10_000. } else { 0. };
            let requests = [ObservedDraftRequest { id: 1, rows: 1, accepted: 1, confidence: None }];
            policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
                layer_us: &[None; 40], total_us: total, predicted_us: Some(predicted), draft_us: f64::NAN, wide: false }).unwrap();
        }
        assert!((policy.time_bias()[0] - 1000.).abs() < 250., "{:?}", policy.time_bias());
    }

    /// One solo round of request `id` with `rows` verified rows after a draft
    /// of `width`, timed by the synthetic truth plus the draft pass.
    fn width_round(policy: &mut DraftPolicy, token: &mut u64, id: u64, width: usize, rows: usize,
        accepted: usize, confidence: &[f64]) {
        let truth = truth();
        let token_rows: Vec<_> = (0..rows).map(|r| token_routes(*token + r as u64, 24)).collect();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|l| token_rows.iter().map(|t| t[l]).collect()).collect();
        let mut layer_us = [None; 40];
        let draft_us = if width > 5 { 3300. } else { 2500. };
        let mut total = truth.round[0] + truth.round[1] * rows as f64 + truth.round[2] + draft_us;
        for layer in 1..40 {
            let class = policy.geometry.layers[layer].class.unwrap() as usize;
            let mb = groups(&routes[layer]) * policy.geometry.layers[layer].slice_bytes / 1e6;
            let t = truth.alpha[class] + truth.beta[class] * rows as f64 + truth.us_per_mb[class] * mb;
            layer_us[layer] = Some(t);
            total += t;
        }
        let requests = [ObservedDraftRequest { id, rows, accepted, confidence: Some(confidence) }];
        policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
            layer_us: &layer_us, total_us: total, predicted_us: None, draft_us, wide: width > 5 }).unwrap();
        *token += accepted as u64;
    }

    #[test]
    fn width_follows_each_requests_recent_runs() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        let mut token = 0;
        // Warmup alternates widths so both draft-cost fits are identified.
        let mut chosen = Vec::new();
        for round in 0..400 {
            let width = policy.choose_width(false, &[(1, 7)]);
            if round < 8 { chosen.push(width); }
            // Request 1 is code-like: long runs fully accepted at every width.
            width_round(&mut policy, &mut token, 1, width, width + 1, width + 1, &[0.97; 7][..width]);
            // Request 2 is prose-like: short accepted runs.
            width_round(&mut policy, &mut token, 2, 5, 6, 2, &[0.6; 5]);
        }
        assert!(chosen.contains(&5) && chosen.contains(&7), "{chosen:?}");
        let fit = policy.cost_snapshot(false).draft;
        assert!((fit[2] + fit[3] - 800.).abs() < 100., "wide draft extra {fit:?}");
        assert_eq!(policy.choose_width(false, &[(1, 7)]), 7);
        assert_eq!(policy.choose_width(false, &[(2, 7)]), 5);
        // A short remaining budget cannot use the wide positions.
        assert_eq!(policy.choose_width(false, &[(1, 4)]), 5);
    }

    #[test]
    fn unused_width_is_revisited() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        let mut token = 0;
        for _ in 0..400 {
            let width = policy.choose_width(false, &[(2, 7)]);
            width_round(&mut policy, &mut token, 2, width, 3, 2, &[0.6; 7][..width]);
        }
        // Prose stays narrow except for periodic exploration of the wide width.
        let wide = policy.stats().width_rounds[1];
        assert!(wide >= 400 / WIDTH_EXPLORE_ROUNDS - 1 && wide < 40, "{:?}", policy.stats().width_rounds);
    }

    /// FNV-1a over a replay transcript: every decision, prediction and fit
    /// the policy produced, bit for bit.
    struct Transcript { hash: u64, records: u64, dump: Option<String> }
    impl Transcript {
        fn new() -> Self {
            Self { hash: 0xcbf2_9ce4_8422_2325, records: 0,
                dump: std::env::var_os("CUTEAFD_POLICY_REPLAY_DUMP").map(|_| String::new()) }
        }
        fn word(&mut self, tag: &str, value: u64) {
            for byte in tag.bytes().chain(value.to_le_bytes()) {
                self.hash = (self.hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
            }
            self.records += 1;
            if let Some(dump) = &mut self.dump { dump.push_str(&format!("{tag} {value:#x}\n")); }
        }
        fn real(&mut self, tag: &str, value: f64) { self.word(tag, value.to_bits()); }
    }

    struct Mix(u64);
    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
        fn below(&mut self, n: u64) -> u64 { self.next() % n }
    }

    /// Replay of V4.1-shaped rounds through every public entry point: mixed
    /// local/TP2/Spark placements, MXFP4 and NVFP4 slices, both regimes, one
    /// to four requests with admission and release, both widths, warm-up,
    /// selection, prediction, stalls, missing layer timings, undrafted and
    /// unconfident (copied) requests, router flag bits and a rejected round.
    fn replay(transcript: &mut Transcript) {
        let mut mix = Mix(0x5eed);
        // (local layers, TP2 layers, remote slice bytes, wide width, fixed)
        let scenarios = [(0, 0, 4_700_160., 7, false), (20, 0, 4_700_160., 7, false),
            (40, 40, 9_400_320., 5, false), (8, 3, 4_976_640., 7, false), (0, 0, 4_700_160., 7, true)];
        for (scenario, &(local, tp2, remote, wide, fixed)) in scenarios.iter().enumerate() {
            let geometry = v41(|l| l < local, |l| if l < tp2 { 9_400_320. } else if l < local { 18_800_640. } else { remote },
                if wide > 5 { vec![5, wide] } else { vec![5] });
            let mut policy = DraftPolicy::new(geometry, fixed).unwrap();
            let mut live: Vec<(u64, u64, u32, f64)> = Vec::new(); // (id, next token, pool, quality)
            let mut next_id = 1 + 1000 * scenario as u64;
            for round in 0..320u64 {
                let shared = round % 5 >= 3;
                if live.is_empty() || (live.len() < 4 && mix.unit() < 0.12) {
                    let code = mix.unit() < 0.5;
                    live.push((next_id, next_id * 100_000, if code { 12 } else { 40 }, if code { 0.93 } else { 0.62 }));
                    next_id += 1;
                }
                if live.len() > 1 && mix.unit() < 0.06 {
                    let gone = live.remove(mix.below(live.len() as u64) as usize);
                    policy.release(gone.0);
                }
                let requests: Vec<(u64, usize)> = live.iter()
                    .map(|r| (r.0, 1 + mix.below(wide as u64) as usize)).collect();
                let width = policy.choose_width(shared, &requests);
                transcript.word("width", width as u64);
                let confidence: Vec<Vec<f64>> = requests.iter().zip(&live).map(|(&(_, available), r)|
                    (0..width.min(available)).map(|p| (r.3 + 0.06 * mix.unit() - 0.02 * p as f64).clamp(0., 1.))
                        .collect()).collect();
                let candidates: Vec<_> = live.iter().zip(&confidence)
                    .map(|(r, c)| DraftCandidate { id: r.0, confidence: c }).collect();
                let ids: Vec<u64> = live.iter().map(|r| r.0).collect();
                let lengths = match policy.select(shared, &candidates, width).unwrap() {
                    Some(selection) => {
                        for &length in &selection.lengths { transcript.word("length", length as u64); }
                        transcript.real("expected", selection.expected_tokens);
                        transcript.real("selected_us", selection.predicted_us);
                        transcript.word("evaluated", selection.evaluated as u64);
                        selection.lengths
                    }
                    None => {
                        transcript.word("unselected", 0);
                        confidence.iter().map(Vec::len).collect()
                    }
                };
                let predicted = policy.predict(shared, &ids, &lengths, width);
                transcript.real("predicted", predicted.unwrap_or(-1.));
                // Execute the round against a synthetic truth.
                let mut rows_routes: Vec<Vec<[u32; 6]>> = Vec::new();
                let mut accepted = Vec::new();
                for ((r, &length), c) in live.iter().zip(&lengths).zip(&confidence) {
                    let hits = c[..length].iter().take_while(|&&p| mix.unit() < p * 0.97).count();
                    accepted.push(1 + hits);
                    for row in 0..=length { rows_routes.push(token_routes(r.1 + row as u64, r.2)); }
                }
                let rows = rows_routes.len();
                let mut routes: Vec<Vec<[u32; 6]>> = (0..40).map(|l| rows_routes.iter().map(|t| t[l]).collect()).collect();
                // Router flag bits above the expert index are ignored.
                if round % 7 == 0 { routes[3][0][2] |= 512; }
                let regime = if shared { 1.3 } else { 1. };
                let mut layer_us = [None; 40];
                let mut total = (9000. + 250. * rows as f64 + 150. * ids.len() as f64) * regime;
                for layer in 1..40 {
                    let c = policy.geometry().layers[layer].class.unwrap() as usize;
                    let mb = groups(&routes[layer]) * policy.geometry().layers[layer].slice_bytes / 1e6;
                    let t = ([600., 700.][c] + [8., 12.][c] * rows as f64 + [0.9, 7.5][c] * mb)
                        * regime * (0.95 + 0.1 * mix.unit());
                    layer_us[layer] = Some(t);
                    total += t;
                }
                if round % 23 == 11 { layer_us[1 + mix.below(39) as usize] = None; }
                if round % 31 == 5 { total += 25_000.; }
                let drafted = round % 13 != 6;
                let draft_us = if drafted { (2500. + 120. * ids.len() as f64) * if width > 5 { 1.3 } else { 1. } } else { f64::NAN };
                if drafted { total += draft_us; }
                let copied: Vec<bool> = live.iter().map(|_| mix.unit() < 0.1).collect();
                let observed: Vec<_> = live.iter().zip(&lengths).zip(&accepted).zip(&confidence).zip(&copied)
                    .map(|((((r, &length), &accepted), c), &copied)| ObservedDraftRequest {
                        id: r.0, rows: length + 1, accepted, confidence: if copied { None } else { Some(&c[..]) } })
                    .collect();
                let result = policy.observe(DraftRoundObservation { shared, requests: &observed, routes: &flat(&routes),
                    layer_us: &layer_us, total_us: total, draft_us, wide: drafted && width > policy.widths()[0],
                    predicted_us: predicted });
                transcript.word("observed", u64::from(result.is_ok()));
                for (r, &a) in live.iter_mut().zip(&accepted) { r.1 += a as u64; }
                if round % 97 == 50 {
                    // A route outside the model is rejected after earlier
                    // requests' histories were extended.
                    let mut bad = routes.clone();
                    bad[5][rows - 1][0] = 400;
                    let result = policy.observe(DraftRoundObservation { shared, requests: &observed, routes: &flat(&bad),
                        layer_us: &layer_us, total_us: total, draft_us, wide: false, predicted_us: None });
                    transcript.word("rejected", u64::from(result.is_err()));
                }
                if round % 16 == 15 { record_state(transcript, &policy); }
            }
            record_state(transcript, &policy);
        }
    }

    fn record_state(transcript: &mut Transcript, policy: &DraftPolicy) {
        for shared in [false, true] {
            transcript.word("warm", u64::from(policy.warm(shared)));
            let snapshot = policy.cost_snapshot(shared);
            for value in snapshot.layers.iter().flatten().chain(&snapshot.round).chain(&snapshot.draft) {
                transcript.real("fit", *value);
            }
        }
        for (slope, offset) in policy.calibration() { transcript.real("slope", slope); transcript.real("offset", offset); }
        for &bias in policy.time_bias() { transcript.real("bias", bias); }
        let stats = policy.stats();
        for value in [stats.rounds, stats.selected_rounds, stats.verified_rows, stats.verified_drafts,
            stats.accepted_drafts, stats.emitted_requests, stats.predicted_rounds] { transcript.word("stat", value); }
        for &value in stats.draft_rows.iter().chain(&stats.position_reached).chain(&stats.position_accepted)
            .chain(&stats.width_rounds) { transcript.word("count", value); }
        for &value in stats.position_confidence.iter().chain(&stats.position_raw_confidence)
            .chain([&stats.prediction_error_us, &stats.prediction_abs_error_us, &stats.observed_us]) {
            transcript.real("sum", value);
        }
    }

    /// Decisions and fits recorded from the pre-geometry `dspark_policy` (work/p0
    /// e55eeaeb) on the replay above. A change here is a decision change.
    #[test]
    fn replay_reproduces_recorded_decisions_and_fits() {
        let mut transcript = Transcript::new();
        replay(&mut transcript);
        if let (Some(dump), Some(path)) = (&transcript.dump, std::env::var_os("CUTEAFD_POLICY_REPLAY_DUMP")) {
            std::fs::write(path, dump).unwrap();
        }
        assert_eq!((transcript.records, transcript.hash), (23_994, 2_848_516_388_929_143_687));
    }

    #[test]
    fn rejects_invalid_inputs() {
        let mut policy = DraftPolicy::new(placement(5), false).unwrap();
        assert!(policy.select(false, &[], 5).is_err());
        assert!(policy.select(false, &[DraftCandidate { id: 1, confidence: &[f64::NAN] }], 5).is_err());
        assert!(policy.select(false, &[DraftCandidate { id: 1, confidence: &[1.1] }], 5).is_err());
        assert!(policy.select(false, &[DraftCandidate { id: 1, confidence: &[0.5; 8] }], 5).is_err());
        let requests = [ObservedDraftRequest { id: 1, rows: 2, accepted: 1, confidence: None }];
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]]).collect();
        assert!(policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
            layer_us: &[None; 40], total_us: 1., predicted_us: None, draft_us: f64::NAN, wide: false }).is_err());
        assert!(policy.observe(DraftRoundObservation { shared: false, requests: &requests[..0], routes: &[1],
            layer_us: &[], total_us: 1., predicted_us: None, draft_us: f64::NAN, wide: false }).is_err());
        assert!(DraftPolicy::new(v41(|_| false, |_| 0., vec![5, 7]), false).is_err());
    }
}
