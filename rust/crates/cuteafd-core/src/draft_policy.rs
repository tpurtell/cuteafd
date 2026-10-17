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
//! Time is priced from the resources that are actually consumed, with one
//! physical coefficient pair per resource class and a handful of offsets:
//!
//! * per resource class (for example coordinator RTX or Spark), `layers * a +
//!   c * megabytes`: a fixed per-layer cost (launch, exchange round trip,
//!   packing) plus the routed-expert weight traffic at the class's effective
//!   bandwidth. The traffic is `slice_bytes * sum_e ceil(routes_e /
//!   group_rows)`: grouped slice kernels read each expert's slice once per
//!   row group. Fitted on the class's summed layer times, which separates the
//!   classes even when their traffic moves together;
//! * per round, a residual for everything else (embedding and untimed layers,
//!   attention, vocabulary head, sampling, commit): an intercept and row price per concurrency bucket;
//! * per draft pass, `D + E * requests + wide * (F + G * requests)`.
//!
//! Every coefficient is fitted online from the lane's own timings, with
//! exponential forgetting and Huber-weighted residuals, around physical
//! priors. Changing the quantization or tensor-parallel width changes the
//! known byte counts; the fitted coefficients absorb clocks, thermal state and
//! kernel efficiency. No offline calibration is used.
//!
//! Expert traffic for unexecuted draft rows is forecast from each request's
//! own recent committed routes: draft row `k` stands in for the `k`-th most
//! recent committed token, averaged over several shifted windows. Cross-request
//! sharing within a lane falls out of the exact multiset union.
//!
//! This module performs no device work.
use std::collections::{BTreeMap, VecDeque};
#[path = "simple_draft.rs"]
mod simple_draft;
pub use simple_draft::{Cutoff, DraftMode, Reliability, SimpleDecision, SimpleRound, SimpleSelection, CUTOFF_CEILING,
    CUTOFF_FLOOR, OFFSET_PRIOR, REPORT_GROUPS, SIMPLE_CUTOFF};
use simple_draft::SimpleDraft;

/// Committed tokens retained per request.
const HISTORY: usize = 24;
/// Shifted windows averaged by the forecast.
const WINDOWS: usize = 4;
/// Round samples needed before the policy engages.
const WARM_ROUND_SAMPLES: u64 = 12;
/// Per-sample forgetting of the calibration curvature: memory of roughly 500
/// reached samples per position, so the fit tracks drift.
const CALIBRATION_DECAY: f64 = 0.998;
/// Initial curvature (prior strength) of each position's calibration.
const CALIBRATION_PRIOR: f64 = 2.;
/// Weight of the newest round in each request's recent calibrated confidence.
const RECENT_RATE: f64 = 0.5;
const SKIP_FORGET_RATE: f64 = 0.125;
const REQUEST_EXPLORE_SKIPS: u64 = 8;
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
/// Weight of one sample that an uninformed prior is worth.
const PRIOR_SAMPLES: f64 = 0.01;
/// Resource classes a geometry may name.
pub const MAX_RESOURCE_CLASSES: usize = 4;
/// Smoothing of a seeded policy's relative prediction error.
const ERROR_RATE: f64 = 0.1;
/// Independent evidence regimes for resource costs and confidence calibration.
pub const CONCURRENCY_BUCKETS: usize = 4;
/// C1, C2-4, C5-8 and C9+.
pub fn draft_concurrency_bucket(requests: usize) -> usize {
    match requests { 0..=1 => 0, 2..=4 => 1, 5..=8 => 2, _ => 3 }
}
/// Reached samples before a position uses its own calibration.
const POOLED_BELOW: u64 = 32;
/// Least survival (cumulative acceptance probability) a draft row needs under
/// the survival length rule: the row is more likely verified than wasted.
pub const SURVIVAL_FLOOR: f64 = 0.5;

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
    /// Installed grouping by total live verifier rows, then backbone layer.
    /// Empty keeps the static fast path; row zero records the compiled tile.
    /// A nonempty table covers every row through the verifier capacity.
    pub group_rows_by_rows: Vec<Vec<u8>>,
    /// Routed experts per layer; route ids are `u16`. Zero with `topk == 0`
    /// describes a target whose layers are all dense (`class == None`).
    pub experts: u16,
    /// Routes per token and layer; zero for a fully dense target.
    pub topk: u8,
    /// The lane's row budget in requests.
    pub max_requests: usize,
    /// Draft positions per request.
    pub max_positions: usize,
    /// Draft widths the drafter can switch between, ascending: one entry is a
    /// fixed width, two let the policy choose per round. The first is the narrow
    /// draft-cost width and the default neural-confidence boundary. Bindings
    /// may set a wider native boundary with `set_native_confidence_positions`.
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

/// One resource class: a monitoring label and its physical prior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResourceClass {
    pub label: &'static str,
    /// Prior per-layer fixed µs, (unused, kept for bindings), µs per MB.
    pub prior: [f64; 3],
    /// Unused: the prior's weight is [`CLASS_PRIOR_ROUNDS`] rounds of one-row
    /// traffic for every class.
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

/// What a binding knows about its lane before the first round: typical warm
/// fits as priors, how many drafts to verify until the fits have earned
/// trust. Every binding supplies a seed; weak priors use the same behavior
/// path as measured warm priors.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicySeed {
    /// Round residual prior: intercept µs, µs per verified row (the third
    /// entry is unused).
    pub round: [f64; 3],
    /// Narrow-width draft pass prior: intercept µs, µs per request.
    pub draft: [f64; 2],
    /// Weight of the round and draft priors, in samples (the class priors
    /// carry [`CLASS_PRIOR_ROUNDS`]).
    pub samples: f64,
    /// Drafts each request verifies, within the row budget, until the policy
    /// engages: conservative, not minimal.
    pub cold_drafts: usize,
    /// Fully timed rounds the policy needs before it engages.
    pub warm_rounds: u64,
    /// Most smoothed relative error of its out-of-sample round prediction
    /// with which it engages.
    pub warm_error: f64,
}

impl PolicySeed {
    /// Conservative weak priors for bindings without measured warm fits.
    pub fn weak(max_positions: usize) -> Self {
        Self { round: [5000., 300., 200.], draft: [2500., 100.], samples: PRIOR_SAMPLES,
            cold_drafts: max_positions, warm_rounds: WARM_ROUND_SAMPLES, warm_error: 1. }
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
        let routed = self.layers.iter().any(|layer| layer.class.is_some());
        if self.experts == 0 || self.topk == 0 {
            if routed || self.experts != 0 || self.topk != 0 { return invalid("no routed experts"); }
        }
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
        if !self.group_rows_by_rows.is_empty() {
            if self.group_rows_by_rows.len() <= self.max_requests {
                return invalid("grouping schedule must cover every verifier anchor");
            }
            if self.group_rows_by_rows.iter().any(|row| row.len() != self.layers.len()
                || row.iter().any(|&group| group == 0)) {
                return invalid("grouping schedule needs one positive group per layer and row");
            }
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
    /// Effective installed grouping at an executed verifier shape.
    pub fn group_rows(&self, layer: usize, rows: usize) -> u8 {
        if self.group_rows_by_rows.is_empty() { self.layers[layer].group_rows }
        else { self.group_rows_by_rows[rows][layer] }
    }
    fn verify_capacity(&self) -> usize {
        if self.group_rows_by_rows.is_empty() { usize::MAX } else { self.group_rows_by_rows.len() - 1 }
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

/// Rounds of one-row traffic a class's physical prior is worth: it holds the
/// fixed/bandwidth split of a short or collinear phase (C1, or local and remote
/// traffic that move together) near physics, and yields to a few wide rounds.
pub const CLASS_PRIOR_ROUNDS: f64 = 2.;
/// Rounds of one-row traffic a class's traffic price prior is worth: a
/// measurable bandwidth, so it moves only on strong evidence (V4.1 hardware:
/// remote 6.3 us/MB fitted against 5.7 physical, where a weak prior drifted
/// to 9.4).
pub const CLASS_TRAFFIC_PRIOR_ROUNDS: f64 = 200.;
/// Per-round forgetting of the class fits: physical costs drift slowly
/// (clocks, thermals), so they keep a memory of about five hundred rounds.
const CLASS_DECAY: f64 = 0.998;
/// Per-round forgetting of the round residual and draft-pass fits.
const ROUND_DECAY: f64 = 0.98;
/// Layer-timed rounds each class fit needs before the policy engages; layer
/// times may be sampled on a cadence, the physical priors carry the rest.
const WARM_CLASS_SAMPLES: u64 = 4;
/// Prior rounds shrinking a regime's mean residual correction toward its
/// seeded line. Only observations in that concurrency bucket update it.
const BUCKET_PRIOR_ROUNDS: f64 = 4.;
/// Within one round shape, wall time varies by about a quarter (V4.1 traces:
/// CV 0.26-0.28). The relative Huber guard starts from that scale, so the
/// first wide batch (about six times the median) cannot poison the fits.
const INITIAL_RELATIVE_SCALE: f64 = 0.25;
/// Floor of the relative residual scale: below the within-shape noise of any
/// measured lane, so a systematic shift (a new bucket, a clock change) is
/// learned rather than rejected as an outlier forever.
const MIN_RELATIVE_SCALE: f64 = 0.05;

/// Exponentially forgotten, prior-regularized least squares with
/// nonnegative coefficients and a relative Huber guard.
#[derive(Clone, Debug)]
struct Fit<const D: usize> {
    decay: f64,
    sxx: [[f64; D]; D],
    sxy: [f64; D],
    prior_mean: [f64; D],
    prior_precision: [f64; D],
    theta: [f64; D],
    scale: f64,
    samples: u64,
}

impl<const D: usize> Fit<D> {
    fn new(decay: f64, prior_mean: [f64; D], prior_precision: [f64; D]) -> Self {
        Self { decay, sxx: [[0.; D]; D], sxy: [0.; D], prior_mean, prior_precision, theta: prior_mean,
            scale: INITIAL_RELATIVE_SCALE, samples: 0 }
    }
    fn predict(&self, x: &[f64; D]) -> f64 {
        x.iter().zip(&self.theta).map(|(x, t)| x * t).sum()
    }
    /// Huber weight of a residual relative to `predicted`, updating the
    /// running relative scale. Graph capture, prefill interleave, scheduler
    /// stalls and the first wide batch are rare large residuals: they must
    /// not drag the fit, and past six scales they are dropped. The scale still
    /// grows with them, so a persistent change (or a poor seed) is accepted
    /// within a few rounds.
    fn weigh(scale: &mut f64, predicted: f64, y: f64) -> f64 {
        let relative = (y - predicted).abs() / predicted.max(1.);
        let weight = if relative > 6. * *scale { 0. } else if relative > 3. * *scale { 3. * *scale / relative } else { 1. };
        *scale = (0.97 * *scale + 0.03 * relative.min(6. * *scale)).max(MIN_RELATIVE_SCALE);
        weight
    }
    fn observe(&mut self, x: [f64; D], y: f64) {
        if !y.is_finite() || y < 0. || x.iter().any(|v| !v.is_finite()) { return; }
        let predicted = self.predict(&x);
        let weight = Self::weigh(&mut self.scale, predicted, y);
        for i in 0..D {
            for j in 0..D { self.sxx[i][j] = self.decay * self.sxx[i][j] + weight * x[i] * x[j]; }
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
                if fixed[i] { a[i][i] = 1.; continue; }
                for j in 0..D {
                    if !fixed[j] { a[i][j] = self.sxx[i][j]; }
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

/// Fitted coefficients for one (regime, concurrency bucket), exported for
/// monitoring and traces.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftCostSnapshot {
    /// Per class: µs per layer, µs per MB, µs per layer and verified row,
    /// samples, relative residual scale.
    pub layers: Vec<[f64; 5]>,
    /// Bucket-local round residual: intercept µs, µs per verified row,
    /// shrunk correction µs, effective rounds, relative scale.
    pub round: [f64; 5],
    /// Draft pass: intercept µs, µs per request, wide extra µs, wide extra µs
    /// per request, samples, relative residual scale.
    pub draft: [f64; 6],
    /// Concurrency bucket of these coefficients.
    pub bucket: usize,
}

impl DraftCostSnapshot {
    /// Named coefficients for `/v1/stats` and traces.
    pub fn to_json(&self, labels: &[&str]) -> serde_json::Value {
        serde_json::json!({
            "bucket": self.bucket,
            "classes": self.layers.iter().zip(labels).map(|(c, label)| serde_json::json!({
                "class": label, "us_per_layer": c[0], "us_per_mb": c[1], "us_per_layer_row": c[2],
                // µs per MB -> GB/s: 1 MB / (c µs) = 1000 / c GB/s.
                "effective_gb_per_s": if c[1] > 0. { 1000. / c[1] } else { f64::INFINITY },
                "samples": c[3], "relative_scale": c[4]})).collect::<Vec<_>>(),
            "residual": {"intercept_us": self.round[0], "us_per_row": self.round[1], "bucket_offset_us": self.round[2],
                "bucket_rounds": self.round[3], "relative_scale": self.round[4]},
            "draft": {"intercept_us": self.draft[0], "us_per_request": self.draft[1], "wide_extra_us": self.draft[2],
                "wide_extra_us_per_request": self.draft[3], "samples": self.draft[4], "relative_scale": self.draft[5]},
        })
    }
}

/// Megabytes of expert traffic per resource class.
type ClassBytes = [f64; MAX_RESOURCE_CLASSES];

/// Independent concurrency-bucket costs: resource-class layer/traffic/row
/// prices, a round residual line with a shrunk correction, and the draft pass.
#[derive(Clone, Debug)]
struct CostModel {
    /// Per class, x = [layers, MB, layers * rows], fitted on the class's
    /// summed layer time.
    class: [Vec<Fit<3>>; CONCURRENCY_BUCKETS],
    /// Per bucket: forgotten weighted moments of the round residual against
    /// [1, rows]: (Σw, Σw·rows, Σw·rows², Σw·y, Σw·rows·y).
    residuals: [[f64; 5]; CONCURRENCY_BUCKETS],
    /// Seed residual intercept and µs per row, and the prior weight in rounds.
    residual_prior: ([f64; 2], f64),
    /// Per bucket: residual intercept and µs per row, plus its shrunk
    /// mean correction. No other bucket contributes evidence.
    residual: [([f64; 2], f64); CONCURRENCY_BUCKETS],
    /// Relative residual scale of the whole round, for the Huber guard.
    round_scale: [f64; CONCURRENCY_BUCKETS],
    /// Per bucket, x = [1, requests, wide, wide * requests].
    draft: Vec<Fit<4>>,
    /// Per bucket [narrow, wide] draft passes observed.
    draft_samples: [[u64; 2]; CONCURRENCY_BUCKETS],
    /// Priced layers per class.
    layers_in: Vec<usize>,
    rounds: [u64; CONCURRENCY_BUCKETS],
}

impl CostModel {
    fn new(geometry: &PolicyGeometry, priced: &[Priced], seed: &PolicySeed) -> Self {
        let layers_in: Vec<usize> = (0..geometry.classes.len())
            .map(|c| priced.iter().filter(|p| p.class == c).count()).collect();
        let class: Vec<_> = geometry.classes.iter().enumerate().map(|(c, class)| {
            let layers = layers_in[c] as f64;
            // One row's traffic: every timed layer reads top-k slices.
            let one_row: f64 = priced.iter().filter(|p| p.class == c)
                .map(|p| f64::from(geometry.topk) * p.slice_bytes / 1e6).sum();
            // Traffic price held at its physical bandwidth: rows and MB move
            // together, and without a strong prior the fit loads per-row
            // layer work (exchange, packing, attention rows) onto MB.
            Fit::new(CLASS_DECAY, [class.prior[0], class.prior[2], class.prior[1]],
                [CLASS_PRIOR_ROUNDS * layers.powi(2), CLASS_TRAFFIC_PRIOR_ROUNDS * one_row.powi(2),
                    CLASS_PRIOR_ROUNDS * layers.powi(2)].map(|p| p.max(1e-9)))
        }).collect();
        let n = seed.samples;
        let draft = Fit::new(ROUND_DECAY, [seed.draft[0], seed.draft[1], 400., 50.],
            [n / seed.draft[0].max(1.).powi(2), n / seed.draft[1].max(1.).powi(2), PRIOR_SAMPLES / 400f64.powi(2),
                PRIOR_SAMPLES / 50f64.powi(2)]);
        Self {
            class: std::array::from_fn(|_| class.clone()),
            residuals: [[0.; 5]; CONCURRENCY_BUCKETS],
            residual_prior: ([seed.round[0], seed.round[1]], n),
            residual: [([seed.round[0], seed.round[1]], 0.); CONCURRENCY_BUCKETS],
            round_scale: [INITIAL_RELATIVE_SCALE; CONCURRENCY_BUCKETS],
            draft: vec![draft; CONCURRENCY_BUCKETS],
            draft_samples: [[0; 2]; CONCURRENCY_BUCKETS],
            layers_in, rounds: [0; CONCURRENCY_BUCKETS],
        }
    }
    /// Fit only this concurrency regime. A cold regime retains its seed;
    /// wide-round timings must not change a single-request row price.
    fn solve_residual(&mut self, bucket: usize) {
        let ([r0, s0], prior_weight) = self.residual_prior;
        let [w, x, xx, y, xy] = self.residuals[bucket];
        // Ridge toward the seed: intercept weight `prior_weight` rounds, row
        // price weight `prior_weight` rounds at the mean row count.
        let mean_rows = if w > 0. { x / w } else { 1. };
        let (pi, ps) = (prior_weight, prior_weight * mean_rows.max(1.).powi(2));
        let a = [[w + pi, x], [x, xx + ps]];
        let b = [y + pi * r0, xy + ps * s0];
        let line = match solve_linear(a, b) {
            Some([intercept, slope]) if slope >= 0. => [intercept, slope],
            _ => [(y + pi * r0) / (w + pi), 0.],
        };
        let offset = if w > 0. { (y - line[0] * w - line[1] * x) / (w + BUCKET_PRIOR_ROUNDS) } else { 0. };
        self.residual[bucket] = (line, offset);
    }
    fn layers_us(&self, bucket: usize, rows: f64, megabytes: ClassBytes) -> f64 {
        self.class[bucket].iter().enumerate().filter(|&(c, _)| self.layers_in[c] > 0)
            .map(|(c, fit)| fit.predict(&[self.layers_in[c] as f64, megabytes[c], self.layers_in[c] as f64 * rows])).sum()
    }
    fn residual_us(&self, bucket: usize, rows: f64) -> f64 {
        (self.residual[bucket].0[0] + self.residual[bucket].0[1] * rows + self.residual[bucket].1).max(0.)
    }
    /// Predicted µs of the verify round, without the draft pass.
    fn predict(&self, bucket: usize, rows: f64, megabytes: ClassBytes) -> f64 {
        self.layers_us(bucket, rows, megabytes) + self.residual_us(bucket, rows)
    }
    /// Marginal µs of one more row that brings `megabytes` of new traffic.
    fn marginal(&self, bucket: usize, rows: f64, megabytes: ClassBytes) -> f64 {
        self.residual_us(bucket, rows + 1.) - self.residual_us(bucket, rows)
            + self.class[bucket].iter().enumerate()
                .map(|(c, fit)| fit.theta[1] * megabytes[c] + fit.theta[2] * self.layers_in[c] as f64).sum::<f64>()
    }
    /// Predicted µs of the draft pass.
    fn draft_us(&self, bucket: usize, requests: f64, wide: bool) -> f64 {
        if requests == 0. { return 0.; }
        let wide = f64::from(u8::from(wide));
        self.draft[bucket].predict(&[1., requests, wide, wide * requests])
    }
    /// The class and residual fits have seen enough fully timed rounds.
    fn warm(&self, bucket: usize) -> bool {
        self.rounds[bucket] >= WARM_ROUND_SAMPLES
            && self.class[bucket].iter().zip(&self.layers_in).all(|(fit, &n)| n == 0 || fit.samples >= WARM_CLASS_SAMPLES)
    }
    fn widths_warm(&self, bucket: usize) -> bool {
        self.draft_samples[bucket].iter().all(|&n| n >= WARM_WIDTH_SAMPLES)
    }
    /// Learn from one fully timed round: each class from its summed layer
    /// time, the residual from the rest of the verify time.
    fn observe_round(&mut self, bucket: usize, rows: f64, megabytes: ClassBytes, class_us: Option<&[f64]>,
        verify_us: f64) {
        if !verify_us.is_finite() { return; }
        // The guard is relative to the whole round: a residual is small next
        // to the layer time it sits beside.
        let predicted = self.predict(bucket, rows, megabytes);
        let weight = Fit::<1>::weigh(&mut self.round_scale[bucket], predicted, verify_us);
        if let Some(class_us) = class_us {
            for (c, fit) in self.class[bucket].iter_mut().enumerate() {
                let layers = self.layers_in[c] as f64;
                if layers > 0. { fit.observe([layers, megabytes[c], layers * rows], class_us[c]); }
            }
        }
        let residual = verify_us - self.layers_us(bucket, rows, megabytes);
        let m = &mut self.residuals[bucket];
        for (moment, value) in m.iter_mut().zip([1., rows, rows * rows, residual, rows * residual]) {
            *moment = ROUND_DECAY * *moment + weight * value;
        }
        self.solve_residual(bucket);
        self.rounds[bucket] += 1;
    }
    fn observe_draft(&mut self, bucket: usize, requests: f64, wide: bool, elapsed: f64) {
        let flag = f64::from(u8::from(wide));
        self.draft[bucket].observe([1., requests, flag, flag * requests], elapsed);
        self.draft_samples[bucket][usize::from(wide)] += 1;
    }
    fn snapshot(&self, bucket: usize) -> DraftCostSnapshot {
        let draft = &self.draft[bucket];
        DraftCostSnapshot {
            layers: self.class[bucket].iter().map(|f| [f.theta[0], f.theta[1], f.theta[2], f.samples as f64, f.scale]).collect(),
            round: [self.residual[bucket].0[0], self.residual[bucket].0[1], self.residual[bucket].1, self.residuals[bucket][0],
                self.round_scale[bucket]],
            draft: [draft.theta[0], draft.theta[1], draft.theta[2], draft.theta[3], draft.samples as f64, draft.scale],
            bucket,
        }
    }
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

/// One request's candidate: conditional acceptance probability of each
/// available draft position, in order (sigmoid of the draft confidence).
#[derive(Clone, Copy, Debug)]
pub struct DraftCandidate<'a> {
    pub id: u64,
    pub confidence: &'a [f64],
}

/// One pre-draft action proposed by a family adapter. Width zero means skip.
/// A block drafter supplies skip/block; a chain supplies its depth caps.
#[derive(Clone, Copy, Debug)]
pub struct DraftAction {
    pub width: usize,
    /// This action's predicted draft time, from the adapter's draft-cost fit.
    pub draft_us: f64,
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
    /// Sequences passed to the drafter; None retains all-request legacy pricing.
    pub drafted_requests: Option<usize>,
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

/// Confidence evidence observed at one verifier concurrency, including each
/// request's short memory so a live request changing regimes cannot leak it.
#[derive(Clone, Debug)]
struct Confidence {
    /// Per-position Platt calibration of the drafter's confidence. Confidence
    /// is conditional on earlier positions being accepted, and its reliability
    /// varies by position: past the drafter's native block it saturates near
    /// 0.99 whatever the content, so a slope as well as an offset is needed.
    /// Each position is refitted online from the outcomes of the rounds that
    /// reached it.
    calibration: Vec<Platt>,
    reached: Vec<u64>,
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
    /// Native-head and copy-tail priors remain separate within the regime.
    pooled: [Platt; 2],
}

impl Confidence {
    fn new(positions: usize) -> Self {
        Self { calibration: vec![Platt::new(); positions], reached: vec![0; positions],
            recent: BTreeMap::new(), outcomes: BTreeMap::new(), raw_mean: vec![0.9; positions],
            pooled: [Platt::new(); 2] }
    }
}

/// Whole-policy state: route history, concurrency-local evidence, and statistics.
#[derive(Clone, Debug)]
pub struct DraftPolicy {
    geometry: PolicyGeometry,
    /// Native neural positions need not equal the narrow execution width.
    native_confidence_positions: usize,
    priced: Vec<Priced>,
    fixed: bool,
    cost: CostModel,
    /// Per request, the routes of recent committed tokens, newest first, each
    /// `layers * topk` expert ids.
    history: BTreeMap<u64, VecDeque<Box<[u16]>>>,
    /// Expected new expert groups per layer contributed by a token whose routes
    /// are not yet known (short histories), from observed novelty.
    novelty: [Vec<f64>; CONCURRENCY_BUCKETS],
    confidence: [Confidence; CONCURRENCY_BUCKETS],
    /// Draft passes since each width (narrow, wide) last ran, per bucket.
    since_width: [[u64; 2]; CONCURRENCY_BUCKETS],
    stats: DraftPolicyStats,
    /// Routes per (window, layer, expert) in the shape being evaluated.
    counts: Vec<u16>,
    /// Unknown stand-ins have fractional novelty, retained across grouping changes.
    unknown_groups: Vec<f64>,
    dynamic_grouping: bool,
    /// Routes per expert of one observed layer.
    scratch: Vec<u16>,
    /// Experts already in the shape being evaluated, one bit per (window,
    /// layer, expert). While no count has reached its layer's `group_rows`,
    /// `count % group_rows == 0` is exactly "bit clear", so a peek tests bits
    /// in a few kilobytes instead of counts.
    occupancy: Vec<u64>,
    /// Some count reached its layer's `group_rows`: peeks use the counts.
    saturated: bool,
    seed: PolicySeed,
    /// Per concurrency bucket: timed rounds and smoothed relative error of
    /// the out-of-sample round prediction.
    timed: [(u64, f64); CONCURRENCY_BUCKETS],
    engaged: [bool; CONCURRENCY_BUCKETS],
    /// Verify drafts while their calibrated survival is at least one half,
    /// instead of pricing each row (opt-in comparison rule).
    survival_lengths: bool,
    /// Bucket and regime of the latest selection, for traces.
    selected_cell: Option<(usize, bool)>,
    action_age: BTreeMap<(usize, usize), u64>,
    /// Outstanding extent per request; independent lanes may share a regime.
    selected_requests: BTreeMap<u64, usize>,
    selected_rows: BTreeMap<u64, usize>,
    /// Eligibility is recorded by selection, but skips age only on observation.
    selected_draft_room: BTreeMap<u64, bool>,
    skipped: BTreeMap<u64, u64>,
    /// The opt-in simple length rule; `None` is the shared priced policy.
    simple: Option<SimpleDraft>,
}

impl DraftPolicy {
    /// Every binding supplies priors; fixed mode still fits and reports.
    pub fn new(geometry: PolicyGeometry, seed: PolicySeed, fixed: bool) -> Result<Self, DraftPolicyError> {
        geometry.validate()?;
        let finite = seed.round.iter().chain(&seed.draft).chain([&seed.samples, &seed.warm_error]).all(|v| v.is_finite());
        if !finite || seed.samples <= 0. || seed.warm_error <= 0. || seed.round.iter().chain(&seed.draft).any(|&v| v < 0.) {
            return Err(DraftPolicyError::Geometry("seed priors must be finite and nonnegative"));
        }
        Ok(Self::build(geometry, fixed, seed))
    }
    pub fn seeded(geometry: PolicyGeometry, seed: PolicySeed) -> Result<Self, DraftPolicyError> {
        Self::new(geometry, seed, false)
    }
    fn build(geometry: PolicyGeometry, fixed: bool, seed: PolicySeed) -> Self {
        // Constant installed schedules retain their capacity but use the original selector.
        let dynamic_grouping = geometry.group_rows_by_rows.iter().skip(1)
            .any(|row| row.iter().zip(&geometry.layers).any(|(&group, layer)| group != layer.group_rows));
        let priced = geometry.priced();
        let positions = geometry.max_positions;
        Self {
            cost: CostModel::new(&geometry, &priced, &seed),
            timed: [(0, 1.); CONCURRENCY_BUCKETS],
            engaged: [false; CONCURRENCY_BUCKETS],
            survival_lengths: false, selected_cell: None,
            since_width: [[0; 2]; CONCURRENCY_BUCKETS],
            selected_requests: BTreeMap::new(),
            selected_rows: BTreeMap::new(),
            selected_draft_room: BTreeMap::new(), skipped: BTreeMap::new(),
            action_age: BTreeMap::new(), simple: None,
            seed, priced, fixed,
            history: BTreeMap::new(),
            novelty: std::array::from_fn(|_| vec![2.5; geometry.layers.len()]),
            confidence: std::array::from_fn(|_| Confidence::new(positions)),
            stats: DraftPolicyStats::new(positions),
            counts: vec![0; WINDOWS * geometry.layers.len() * usize::from(geometry.experts)],
            unknown_groups: vec![0.; WINDOWS * geometry.layers.len()], dynamic_grouping,
            scratch: vec![0; usize::from(geometry.experts)],
            occupancy: vec![0; WINDOWS * geometry.layers.len() * usize::from(geometry.experts).div_ceil(64)],
            saturated: false, native_confidence_positions: geometry.widths[0], geometry,
        }
    }
    /// A fresh policy over the same geometry and priors.
    pub fn restarted(&self, fixed: bool) -> Self {
        let mut policy = Self::build(self.geometry.clone(), fixed, self.seed.clone());
        policy.survival_lengths = self.survival_lengths;
        policy.set_draft_mode(self.draft_mode());
        policy.native_confidence_positions = self.native_confidence_positions;
        policy
    }
    /// Configure the checkpoint's neural extent before observations. Legacy
    /// bindings retain their narrow-width boundary unless explicitly changed.
    pub fn set_native_confidence_positions(&mut self, positions: usize) -> Result<(), DraftPolicyError> {
        if positions == 0 || positions > self.geometry.max_positions || self.stats.rounds != 0 {
            return Err(DraftPolicyError::Geometry("invalid native confidence extent or policy already observed"));
        }
        self.native_confidence_positions = positions;
        if let Some(simple) = &mut self.simple { simple.set_native(positions.max(*self.geometry.widths.last().unwrap())); }
        Ok(())
    }
    /// Choose lengths by calibrated survival (draft while at least one half)
    /// instead of by priced rows; width, skip and copy actions stay priced.
    pub fn set_survival_lengths(&mut self, enabled: bool) {
        self.survival_lengths = enabled;
    }
    pub fn survival_lengths(&self) -> bool {
        self.survival_lengths
    }
    /// Switch the length rule. The simple modes (`simple_draft`) replace
    /// selection, width and action choice and calibration; the priced fits
    /// keep running for reports only.
    pub fn set_draft_mode(&mut self, mode: DraftMode) {
        let native = self.native_confidence_positions.max(*self.geometry.widths.last().unwrap());
        self.simple = (mode != DraftMode::Shared)
            .then(|| SimpleDraft::new(mode, self.geometry.max_positions, native));
    }
    pub fn draft_mode(&self) -> DraftMode {
        self.simple.as_ref().map_or(DraftMode::Shared, SimpleDraft::mode)
    }
    /// The simple rule's per-position predicted against realized acceptance,
    /// `[group][native, tail][position]` with groups [`REPORT_GROUPS`].
    pub fn simple_reliability(&self) -> Option<&[[Vec<Reliability>; 2]; 4]> {
        self.simple.as_ref().map(SimpleDraft::reliability)
    }
    /// The simple rule's cutoff now for a selection over `requests`, for reports.
    pub fn simple_cutoff(&self, requests: usize) -> Option<Cutoff> {
        self.simple.as_ref().map(|simple| simple.cutoff(requests))
    }
    /// The simple rule's latest observed round (samples, costs, row price and
    /// the decisions behind it), for traces.
    pub fn simple_round(&self) -> Option<&SimpleRound> {
        self.simple.as_ref().and_then(SimpleDraft::last_round)
    }
    /// The simple rule's per-position (slope, offset) calibration.
    pub fn simple_calibration(&self) -> Option<Vec<(f64, f64)>> {
        self.simple.as_ref().map(SimpleDraft::calibration)
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
    /// The lane regime no longer splits the fits; `shared` is kept in the
    /// interface for traces.
    pub fn cost_snapshot(&self, _shared: bool, requests: usize) -> DraftCostSnapshot {
        self.cost.snapshot(draft_concurrency_bucket(requests))
    }
    pub fn warm(&self, _shared: bool, requests: usize) -> bool {
        self.cost.warm(draft_concurrency_bucket(requests))
    }
    pub fn engaged(&self, _shared: bool, requests: usize) -> bool {
        self.engaged[draft_concurrency_bucket(requests)]
    }
    pub fn cold_drafts(&self) -> usize { self.seed.cold_drafts }
    pub fn prediction_health(&self, _shared: bool, requests: usize) -> (u64, f64) {
        self.timed[draft_concurrency_bucket(requests)]
    }
    /// Concurrency bucket and regime of the latest selection, for traces.
    pub fn selected_cell(&self) -> Option<(usize, bool)> { self.selected_cell }
    fn usable(&self, requests: usize) -> Option<usize> {
        let bucket = draft_concurrency_bucket(requests);
        self.engaged[bucket].then_some(bucket)
    }
    /// Predicted drafter time when every verifier request drafts.
    pub fn draft_cost(&self, shared: bool, requests: usize, width: usize) -> f64 {
        self.draft_cost_drafted(shared, requests, requests, width)
    }
    /// A mixed batch keeps its verifier concurrency bucket, but prices only
    /// the sequences actually passed to the drafter.
    pub fn draft_cost_drafted(&self, shared: bool, requests: usize, drafted_requests: usize, width: usize) -> f64 {
        if drafted_requests == 0 || width == 0 { return 0.; }
        let _ = shared;
        self.cost.draft_us(draft_concurrency_bucket(requests), drafted_requests as f64, width > self.geometry.widths[0])
    }
    pub fn release(&mut self, id: u64) {
        self.history.remove(&id);
        self.selected_requests.remove(&id);
        self.selected_rows.remove(&id);
        self.selected_draft_room.remove(&id);
        self.skipped.remove(&id);
        if let Some(simple) = &mut self.simple { simple.release(id); }
        for confidence in &mut self.confidence {
            confidence.recent.remove(&id);
            confidence.outcomes.remove(&id);
        }
    }
    /// Draft widths the drafter can switch between.
    pub fn widths(&self) -> &[usize] {
        &self.geometry.widths
    }
    /// Learned (slope, offset) on the raw logit, per draft position.
    pub fn calibration(&self, requests: usize) -> Vec<(f64, f64)> {
        self.confidence[draft_concurrency_bucket(requests)].calibration.iter().map(|platt| (platt.theta[0], platt.theta[1])).collect()
    }
    fn calibrated(&self, bucket: usize, position: usize, probability: f64) -> f64 {
        let confidence = &self.confidence[bucket];
        if confidence.reached[position] < POOLED_BELOW {
            return confidence.pooled[usize::from(position >= *self.geometry.widths.last().unwrap())].apply(probability);
        }
        confidence.calibration[position].apply(probability)
    }
    /// Acceptance probability of `position` for request `id`: the calibrated
    /// confidence inside the native block, and past it the request's own
    /// recent outcomes shrunk toward that calibrated rate.
    fn request_probability(&self, bucket: usize, id: u64, position: usize, calibrated: f64) -> f64 {
        if position < self.native_confidence_positions { return calibrated; }
        let (hits, trials) = self.confidence[bucket].outcomes.get(&id).map_or((0., 0.), |o| o[position]);
        (hits + OUTCOME_PRIOR * calibrated) / (trials + OUTCOME_PRIOR)
    }

    /// Conditional acceptance used for one request, exposed for opt-in decision
    /// diagnostics. This does not advance calibration or route history.
    pub fn probabilities(&self, requests: usize, candidate: DraftCandidate<'_>) -> Vec<f64> {
        if let Some(simple) = &self.simple {
            let positions = candidate.confidence.len().min(self.geometry.max_positions);
            return simple.probabilities(candidate.id, &candidate.confidence[..positions]);
        }
        let bucket = draft_concurrency_bucket(requests);
        candidate.confidence.iter().enumerate().take(self.geometry.max_positions)
            .map(|(position, &p)| self.request_probability(bucket, candidate.id, position, self.calibrated(bucket, position, p)))
            .collect()
    }

    /// Drafts a request verifies before its concurrency bucket engages: the
    /// calibrated survival rule capped at the seed's cold drafts. Before the
    /// cost fit can price rows, the drafter's own confidence still says which
    /// rows are more likely verified than wasted. A drafted request verifies
    /// at least one draft, so calibration keeps learning while cold.
    pub fn cold_length(&self, requests: usize, candidate: DraftCandidate<'_>) -> usize {
        let cap = self.seed.cold_drafts.min(candidate.confidence.len());
        let mut survival = 1.;
        let surviving = self.probabilities(requests, candidate).iter().take(cap)
            .take_while(|&&p| { survival *= p; survival >= SURVIVAL_FLOOR }).count();
        surviving.max(cap.min(1))
    }

    /// Last chosen verifier extent before grammar/output censoring, for traces.
    pub fn selected_rows(&self, id: u64) -> Option<usize> { self.selected_rows.get(&id).copied() }

    /// Predicted lane µs for explicit lengths after a draft of `width`, if the
    /// fit is usable.
    pub fn predict(&mut self, shared: bool, ids: &[u64], lengths: &[usize], width: usize) -> Option<f64> {
        self.predict_drafted(shared, ids, lengths, width, ids.len())
    }
    /// Price all verifier anchors with the actual drafter batch extent.
    pub fn predict_drafted(&mut self, shared: bool, ids: &[u64], lengths: &[usize], width: usize,
        drafted_requests: usize) -> Option<f64> {
        if ids.is_empty() || ids.len() != lengths.len() || drafted_requests > ids.len()
            || ids.len().saturating_add(lengths.iter().sum::<usize>()) > self.geometry.verify_capacity() { return None; }
        let bucket = self.usable(ids.len())?;
        self.selected_cell = Some((bucket, shared));
        for &id in ids { self.selected_requests.insert(id, ids.len()); }
        Some(self.predict_bucket(bucket, ids, lengths, width, drafted_requests))
    }

    /// Forecast expert megabytes per resource class of explicit lengths,
    /// for diagnostics and traces. Does not touch the fits.
    pub fn forecast_megabytes(&mut self, ids: &[u64], lengths: &[usize]) -> Vec<f64> {
        let bucket = draft_concurrency_bucket(ids.len());
        if ids.len() != lengths.len()
            || ids.len().saturating_add(lengths.iter().sum::<usize>()) > self.geometry.verify_capacity() {
            return vec![f64::NAN; self.geometry.classes.len()];
        }
        self.clear_counts();
        self.use_grouping(ids.len() + lengths.iter().sum::<usize>());
        let mut megabytes = [0.; MAX_RESOURCE_CLASSES];
        for (index, (&id, &length)) in ids.iter().zip(lengths).enumerate() {
            for row in 0..=length { add_bytes(&mut megabytes, self.add_row(bucket, id, row, index)); }
        }
        megabytes[..self.geometry.classes.len()].to_vec()
    }

    fn predict_bucket(&mut self, bucket: usize, ids: &[u64], lengths: &[usize], width: usize, drafted_requests: usize) -> f64 {
        self.clear_counts();
        self.use_grouping(ids.len() + lengths.iter().sum::<usize>());
        let mut megabytes = [0.; MAX_RESOURCE_CLASSES];
        for (index, (&id, &length)) in ids.iter().zip(lengths).enumerate() {
            for row in 0..=length {
                add_bytes(&mut megabytes, self.add_row(bucket, id, row, index));
            }
        }
        let draft = if width == 0 { 0. } else {
            self.cost.draft_us(bucket, drafted_requests as f64, width > self.geometry.widths[0])
        };
        let rows = ids.len() + lengths.iter().sum::<usize>();
        self.cost.predict(bucket, rows as f64, megabytes) + draft
    }

    /// Expected committed tokens and predicted lane µs of explicit `lengths`
    /// for `candidates` after a draft of `width`, with the same calibrated
    /// survival products and pricing [`select`](Self::select) uses; `None`
    /// while the fit is warming up. Lets a binding compare its own reference
    /// shape against the selection on equal terms.
    pub fn evaluate(&mut self, shared: bool, candidates: &[DraftCandidate<'_>], lengths: &[usize], width: usize)
        -> Option<(f64, f64)> {
        self.evaluate_drafted(shared, candidates, lengths, width, candidates.len())
    }

    /// Evaluate a mixed verifier batch with the actual drafter batch extent.
    pub fn evaluate_drafted(&mut self, shared: bool, candidates: &[DraftCandidate<'_>], lengths: &[usize],
        width: usize, drafted_requests: usize) -> Option<(f64, f64)> {
        if drafted_requests > candidates.len() || candidates.len() != lengths.len()
            || candidates.iter().zip(lengths).any(|(c, &n)| n > c.confidence.len()) {
            return None;
        }
        let _ = shared;
        let bucket = self.usable(candidates.len())?;
        self.evaluate_bucket(bucket, candidates, lengths, width, drafted_requests)
    }

    /// Diagnostic evaluation including seed/physical priors before engagement.
    /// It does not make those warming predictions usable for selection.
    pub fn diagnostic_evaluate_drafted(&mut self, shared: bool, candidates: &[DraftCandidate<'_>],
        lengths: &[usize], width: usize, drafted_requests: usize) -> Option<(f64, f64)> {
        let _ = shared;
        if drafted_requests > candidates.len() { return None; }
        self.evaluate_bucket(draft_concurrency_bucket(candidates.len()), candidates, lengths, width, drafted_requests)
    }

    fn evaluate_bucket(&mut self, bucket: usize, candidates: &[DraftCandidate<'_>], lengths: &[usize], width: usize, drafted_requests: usize)
        -> Option<(f64, f64)> {
        if candidates.len() != lengths.len()
            || candidates.len().saturating_add(lengths.iter().sum::<usize>()) > self.geometry.verify_capacity()
            || candidates.iter().zip(lengths).any(|(c, &n)| n > c.confidence.len()
                || n > self.geometry.max_positions) {
            return None;
        }
        let mut expected = candidates.len() as f64;
        for (c, &n) in candidates.iter().zip(lengths) {
            let mut product = 1.;
            for (position, &p) in c.confidence.iter().enumerate().take(n) {
                product *= self.request_probability(bucket, c.id, position, self.calibrated(bucket, position, p));
                expected += product;
            }
        }
        let ids: Vec<_> = candidates.iter().map(|c| c.id).collect();
        Some((expected, self.predict_bucket(bucket, &ids, lengths, width, drafted_requests)))
    }

    /// Choose draft lengths for one lane after a draft of `width`. Returns
    /// `None` while the fit is still warming up or in fixed mode; the caller
    /// then verifies every draft.
    pub fn select(&mut self, shared: bool, candidates: &[DraftCandidate<'_>], width: usize)
        -> Result<Option<DraftSelection>, DraftPolicyError> {
        self.select_within(shared, candidates, width, usize::MAX)
    }

    /// [`select`](Self::select) under one row budget across the round's
    /// requests (anchors included): forward growth gives each row to the
    /// request whose next draft pays most, so a confident request may use
    /// rows a hopeless one leaves. A budget below the anchors keeps every
    /// request at its anchor.
    pub fn select_within(&mut self, shared: bool, candidates: &[DraftCandidate<'_>], width: usize, max_rows: usize)
        -> Result<Option<DraftSelection>, DraftPolicyError> {
        self.select_drafted_within(shared, candidates, width, max_rows, candidates.len())
    }

    /// Select over all verifier requests while pricing only the sequences drafted.
    pub fn select_drafted_within(&mut self, shared: bool, candidates: &[DraftCandidate<'_>], width: usize,
        max_rows: usize, drafted_requests: usize) -> Result<Option<DraftSelection>, DraftPolicyError> {
        if candidates.is_empty() || candidates.len() > self.geometry.max_requests {
            return Err(DraftPolicyError::Requests(self.geometry.max_requests));
        }
        if drafted_requests > candidates.len() { return Err(DraftPolicyError::RequestExtent); }
        for candidate in candidates {
            if candidate.confidence.len() > self.geometry.max_positions
                || candidate.confidence.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
                return Err(DraftPolicyError::Confidence);
            }
        }
        if self.fixed {
            return Ok(None);
        }
        if let Some(simple) = &mut self.simple {
            let selection = simple.select(candidates, max_rows.min(self.geometry.verify_capacity()));
            for (c, &length) in candidates.iter().zip(&selection.lengths) {
                self.selected_requests.insert(c.id, candidates.len());
                self.selected_rows.insert(c.id, length + 1);
            }
            return Ok(Some(selection));
        }
        let Some(bucket) = self.usable(candidates.len()) else { return Ok(None) };
        self.selected_cell = Some((bucket, shared));
        // Survival of row k: the product of calibrated conditional acceptance
        // probabilities of positions 1..=k.
        let cumulative: Vec<Vec<f64>> = candidates.iter().map(|c| {
            let mut product = 1.;
            c.confidence.iter().enumerate().map(|(position, &p)| {
                product *= self.request_probability(bucket, c.id, position, self.calibrated(bucket, position, p));
                product
            }).collect()
        }).collect();
        let ids: Vec<_> = candidates.iter().map(|c| c.id).collect();
        let draft = if width == 0 { 0. } else {
            self.cost.draft_us(bucket, drafted_requests as f64, width > self.geometry.widths[0])
        };
        let selection = if self.survival_lengths {
            self.select_survival(bucket, &ids, &cumulative, draft, max_rows)
        } else {
            self.select_core(bucket, &ids, &cumulative, draft, max_rows, true)
        };
        for (c, &length) in candidates.iter().zip(&selection.lengths) {
            self.selected_requests.insert(c.id, candidates.len());
            self.selected_rows.insert(c.id, length + 1);
            // Empty proposals preserve an intentional pre-draft skip. A
            // selected row later censored by grammar must not age as a skip.
            if !c.confidence.is_empty() {
                self.selected_draft_room.insert(c.id, max_rows > candidates.len() && length == 0);
            }
        }
        Ok(Some(selection))
    }

    /// Choose the draft width for the lane's next round from each request's
    /// recent calibrated confidence. `requests` are (identity, most drafts the
    /// request can use), including every verifier anchor with zero room for
    /// nonparticipants. Both widths use that verifier concurrency bucket and
    /// price only participating requests; the better expected ratio
    /// wins, ties going to the narrow width.
    pub fn choose_width(&mut self, shared: bool, requests: &[(u64, usize)]) -> usize {
        let (narrow, wide) = self.geometry.narrow_wide();
        if narrow == wide || requests.is_empty() { return narrow; }
        if self.fixed || self.simple.is_some() { return wide; }
        let _ = shared;
        let bucket = draft_concurrency_bucket(requests.len());
        if !self.engaged[bucket] || !self.cost.widths_warm(bucket) {
            // Warm up both draft-cost fits by alternating widths.
            let [narrow_samples, wide_samples] = self.cost.draft_samples[bucket];
            return if narrow_samples <= wide_samples { narrow } else { wide };
        }
        if self.since_width[bucket][1] >= WIDTH_EXPLORE_ROUNDS { return wide; }
        if self.since_width[bucket][0] >= WIDTH_EXPLORE_ROUNDS { return narrow; }
        let ids: Vec<_> = requests.iter().map(|r| r.0).collect();
        let mut best: Option<(usize, f64)> = None;
        for width in [narrow, wide] {
            let cumulative: Vec<Vec<f64>> = requests.iter().map(|&(id, available)| {
                let recent = self.confidence[bucket].recent.get(&id);
                let mut product = 1.;
                (0..width.min(available)).map(|position| {
                    let p = recent.and_then(|r| r[position])
                        .unwrap_or_else(|| self.calibrated(bucket, position, self.confidence[bucket].raw_mean[position]));
                    product *= self.request_probability(bucket, id, position, p);
                    product
                }).collect()
            }).collect();
            let drafted = requests.iter().filter(|r| r.1 > 0).count();
            let draft = self.cost.draft_us(bucket, drafted as f64, width > narrow);
            if let Some((_, narrow_ratio)) = best.filter(|_| width > narrow) {
                // Exact pruning: no wide shape beats every expected token over
                // the anchors-only time with the wide pass.
                if self.wide_cannot_win(bucket, &ids, &cumulative, draft, narrow_ratio) { continue; }
            }
            let selection = self.select_core(bucket, &ids, &cumulative, draft, usize::MAX, false);
            let ratio = selection.expected_tokens / selection.predicted_us;
            if best.is_none_or(|(_, b)| ratio > b) { best = Some((width, ratio)); }
        }
        best.map_or(narrow, |(width, _)| width)
    }

    /// Score skip/block/chain-depth actions with the same resource-priced row
    /// selector as post-draft length selection. The adapter retains its cold
    /// start action until the resource fit is usable. Ties retain the first action.
    pub fn choose_action(&mut self, shared: bool, requests: &[(u64, usize)], actions: &[DraftAction],
        max_rows: usize) -> Result<Option<usize>, DraftPolicyError> {
        self.choose_action_impl(shared, requests, actions, None, max_rows)
    }

    /// Action-specific proposal room, aligned with all verifier requests. Zero
    /// excludes a request from that drafter pass, never from the anchor budget.
    /// The adapter supplies each action's cost for its actual participant count.
    pub fn choose_action_with_rooms(&mut self, shared: bool, requests: &[(u64, usize)],
        actions: &[DraftAction], action_rooms: &[&[usize]], max_rows: usize)
        -> Result<Option<usize>, DraftPolicyError> {
        if action_rooms.len() != actions.len() || action_rooms.iter().zip(actions).any(|(rooms, action)|
            rooms.len() != requests.len() || rooms.iter().zip(requests).any(|(&room, &(_, available))|
                room > available || action.width == 0 && room != 0)) {
            return Err(DraftPolicyError::RequestExtent);
        }
        self.choose_action_impl(shared, requests, actions, Some(action_rooms), max_rows)
    }

    fn choose_action_impl(&mut self, shared: bool, requests: &[(u64, usize)], actions: &[DraftAction],
        action_rooms: Option<&[&[usize]]>, max_rows: usize) -> Result<Option<usize>, DraftPolicyError> {
        if requests.is_empty() || requests.len() > self.geometry.max_requests {
            return Err(DraftPolicyError::Requests(self.geometry.max_requests));
        }
        if actions.is_empty() || actions.iter().any(|action| action.width > self.geometry.max_positions
            || !action.draft_us.is_finite() || action.draft_us < 0.
            || action.width == 0 && action.draft_us != 0.) {
            return Err(DraftPolicyError::Geometry("invalid pre-draft action"));
        }
        let room = |action: usize, request: usize| action_rooms
            .map_or(requests[request].1, |rooms| rooms[action][request]);
        let eligible: Vec<_> = actions.iter().enumerate().map(|(index, action)|
            max_rows > requests.len() && action.width > 0
                && (0..requests.len()).any(|request| room(index, request) > 0)).collect();
        // The simple rule always drafts at the widest eligible width.
        if self.fixed || self.simple.is_some() { return Ok(actions.iter().enumerate()
            .filter(|&(index, action)| action_rooms.is_none() || action.width == 0 || eligible[index])
            .max_by_key(|(_, action)| action.width).map(|(index, _)| index)); }
        let _ = shared;
        let Some(bucket) = self.usable(requests.len()) else { return Ok(None) };
        let ids: Vec<_> = requests.iter().map(|r| r.0).collect();
        let mut best: Option<(usize, f64)> = None;
        for (index, action) in actions.iter().enumerate() {
            let cumulative: Vec<Vec<f64>> = requests.iter().enumerate().map(|(request, &(id, _))| {
                let recent = self.confidence[bucket].recent.get(&id);
                let mut product = 1.;
                (0..action.width.min(room(index, request))).map(|position| {
                    let p = recent.and_then(|r| r[position])
                        .unwrap_or_else(|| self.calibrated(bucket, position, self.confidence[bucket].raw_mean[position]));
                    product *= self.request_probability(bucket, id, position, p);
                    product
                }).collect()
            }).collect();
            let selection = self.select_core(bucket, &ids, &cumulative, action.draft_us, max_rows, false);
            let ratio = selection.expected_tokens / selection.predicted_us;
            if best.is_none_or(|(_, score)| ratio > score) { best = Some((index, ratio)); }
        }
        for (_, action) in actions.iter().enumerate().filter(|&(index, _)| eligible[index]) {
            let age = self.action_age.entry((bucket, action.width)).or_default();
            *age = age.saturating_add(1);
        }
        let explore = actions.iter().enumerate().filter(|&(index, _)| eligible[index])
            .filter_map(|(index, action)| self.action_age.get(&(bucket, action.width))
                .filter(|&&age| age >= WIDTH_EXPLORE_ROUNDS).map(|&age| (index, age)))
            .max_by_key(|&(index, age)| (age, usize::MAX - index)).map(|(index, _)| index);
        // Recovery must pick a pass that actually includes an overdue request.
        let recover = actions.iter().enumerate().filter(|&(index, _)| eligible[index]
            && requests.iter().enumerate().any(|(request, &(id, _))| room(index, request) > 0
                && self.skipped.get(&id).copied().unwrap_or(0) >= REQUEST_EXPLORE_SKIPS))
            .min_by_key(|&(index, action)| (action.width, index)).map(|(index, _)| index);
        let chosen = recover.or(explore).or_else(|| best.map(|(index, _)| index));
        if let Some(index) = chosen.filter(|&index| actions[index].width > 0) {
            self.action_age.insert((bucket, actions[index].width), 0);
        }
        for (request, &(id, _)) in requests.iter().enumerate() {
            self.selected_requests.insert(id, requests.len());
            self.selected_draft_room.insert(id, eligible.iter().enumerate()
                .any(|(index, &can_draft)| can_draft && room(index, request) > 0));
        }
        Ok(chosen)
    }

    /// Exact pruning of a width: any shape of `k` drafts has at most the `k`
    /// largest survivals as extra expected tokens, and costs at least the
    /// anchors-only time plus `k` rows that read no new expert weights. If no
    /// `k` beats `ratio` under those bounds, no shape does.
    fn wide_cannot_win(&mut self, bucket: usize, ids: &[u64], cumulative: &[Vec<f64>], draft_us: f64, ratio: f64)
        -> bool {
        // More rows can switch from compact to grouped and reduce existing traffic.
        if self.dynamic_grouping { return false; }
        let mut survivals: Vec<f64> = cumulative.iter().flatten().copied().collect();
        survivals.sort_by(|a, b| b.total_cmp(a));
        let anchors = vec![0; ids.len()];
        let mut time = self.predict_bucket_draft(bucket, ids, &anchors, draft_us);
        let mut expected = ids.len() as f64;
        if expected / time > ratio { return false; }
        let mut rows = ids.len() as f64;
        for survival in survivals {
            expected += survival;
            time += self.cost.marginal(bucket, rows, [0.; MAX_RESOURCE_CLASSES]).max(0.);
            rows += 1.;
            if expected / time > ratio { return false; }
        }
        true
    }

    /// Forward-growth length selection over survival products, with the round's
    /// draft pass priced in.
    fn select_core(&mut self, bucket: usize, ids: &[u64], cumulative: &[Vec<f64>], draft_us: f64, max_rows: usize,
        recover: bool) -> DraftSelection {
        self.clear_counts();
        let max_rows = max_rows.min(self.geometry.verify_capacity());
        self.use_grouping(ids.len());
        let mut megabytes = [0.; MAX_RESOURCE_CLASSES];
        for (index, &id) in ids.iter().enumerate() {
            add_bytes(&mut megabytes, self.add_row(bucket, id, 0, index));
        }
        let mut rows = ids.len();
        let mut lengths = vec![0usize; ids.len()];
        let mut expected = ids.len() as f64;
        let mut time = self.cost.predict(bucket, rows as f64, megabytes) + draft_us;
        if recover {
            let mut overdue: Vec<_> = ids.iter().enumerate().filter_map(|(index, &id)| {
                let age = self.skipped.get(&id).copied().unwrap_or(0);
                (age >= REQUEST_EXPLORE_SKIPS && !cumulative[index].is_empty()).then_some((index, age, id))
            }).collect();
            // Oldest first prevents starvation under a tight global row budget.
            overdue.sort_by_key(|&(_, age, id)| (std::cmp::Reverse(age), id));
            for (index, _, id) in overdue {
                if rows >= max_rows { break; }
                if self.use_grouping(rows + 1) { megabytes = self.counted_megabytes(); }
                add_bytes(&mut megabytes, self.add_row(bucket, id, 1, index));
                lengths[index] = 1;
                rows += 1;
                expected += cumulative[index][0];
            }
            time = self.cost.predict(bucket, rows as f64, megabytes) + draft_us;
        }
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
            if rows >= max_rows { break; }
            if self.use_grouping(rows + 1) {
                megabytes = self.counted_megabytes();
                time = self.cost.predict(bucket, rows as f64, megabytes) + draft_us;
            }
            for (index, &id) in ids.iter().enumerate() {
                let next = lengths[index] + 1;
                if next > cumulative[index].len() { continue; }
                let delta = self.peek_row(bucket, id, next);
                let candidate_expected = expected + cumulative[index][next - 1];
                // Exact: the shape's price grows by the new traffic's price.
                let candidate_time = time + self.cost.marginal(bucket, rows as f64, delta);
                evaluated += 1;
                if choice.is_none_or(|(_, e, t)| candidate_expected / candidate_time > e / t) {
                    choice = Some((index, candidate_expected, candidate_time));
                }
            }
            let Some((index, candidate_expected, candidate_time)) = choice else { break };
            lengths[index] += 1;
            rows += 1;
            add_bytes(&mut megabytes, self.add_row(bucket, ids[index], lengths[index], index));
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

    /// The opt-in survival rule: each request verifies its drafts while their
    /// calibrated survival is at least one half, the most confident request's
    /// rows first under the row budget. Overdue requests still verify one
    /// probe row. The shape is priced with the same cost model.
    fn select_survival(&mut self, bucket: usize, ids: &[u64], cumulative: &[Vec<f64>], draft_us: f64,
        max_rows: usize) -> DraftSelection {
        let max_rows = max_rows.min(self.geometry.verify_capacity());
        let mut lengths: Vec<usize> = cumulative.iter().map(|c| c.iter().take_while(|&&s| s >= SURVIVAL_FLOOR).count()).collect();
        for (index, &id) in ids.iter().enumerate() {
            if lengths[index] == 0 && !cumulative[index].is_empty()
                && self.skipped.get(&id).copied().unwrap_or(0) >= REQUEST_EXPLORE_SKIPS {
                lengths[index] = 1;
            }
        }
        // Under a row budget, drop the least likely rows first.
        let mut rows = ids.len() + lengths.iter().sum::<usize>();
        while rows > max_rows {
            let Some(index) = (0..ids.len()).filter(|&i| lengths[i] > 0)
                .min_by(|&a, &b| cumulative[a][lengths[a] - 1].total_cmp(&cumulative[b][lengths[b] - 1])) else { break };
            lengths[index] -= 1;
            rows -= 1;
        }
        let expected = ids.len() as f64 + cumulative.iter().zip(&lengths).map(|(c, &n)| c[..n].iter().sum::<f64>()).sum::<f64>();
        let predicted_us = self.predict_bucket_draft(bucket, ids, &lengths, draft_us);
        DraftSelection { lengths, expected_tokens: expected, predicted_us, evaluated: 1 }
    }

    fn predict_bucket_draft(&mut self, bucket: usize, ids: &[u64], lengths: &[usize], draft_us: f64) -> f64 {
        self.clear_counts();
        self.use_grouping(ids.len() + lengths.iter().sum::<usize>());
        let mut megabytes = [0.; MAX_RESOURCE_CLASSES];
        for (index, (&id, &length)) in ids.iter().zip(lengths).enumerate() {
            for row in 0..=length { add_bytes(&mut megabytes, self.add_row(bucket, id, row, index)); }
        }
        let rows = ids.len() + lengths.iter().sum::<usize>();
        self.cost.predict(bucket, rows as f64, megabytes) + draft_us
    }

    fn clear_counts(&mut self) {
        self.counts.fill(0);
        if self.dynamic_grouping { self.unknown_groups.fill(0.); }
        self.occupancy.fill(0);
        self.saturated = false;
    }

    /// Switch the precomputed grouping without querying a package on the hot path.
    fn use_grouping(&mut self, rows: usize) -> bool {
        if !self.dynamic_grouping { return false; }
        let groups = &self.geometry.group_rows_by_rows[rows];
        let mut changed = false;
        for priced in &mut self.priced {
            changed |= priced.group_rows != groups[priced.layer];
            priced.group_rows = groups[priced.layer];
        }
        if changed {
            // Counts, not occupancy, remain valid when a compact path changes tile.
            self.saturated = true;
        }
        changed
    }

    /// Reprice the already-added shape only at an installed grouping boundary.
    fn counted_megabytes(&self) -> ClassBytes {
        let mut total = [0.; MAX_RESOURCE_CLASSES];
        let experts = usize::from(self.geometry.experts);
        for window in 0..WINDOWS {
            for priced in &self.priced {
                let at = (window * self.geometry.layers.len() + priced.layer) * experts;
                let groups = self.counts[at..at + experts].iter()
                    .map(|&count| f64::from(count.div_ceil(u16::from(priced.group_rows)))).sum::<f64>()
                    + self.unknown_groups[window * self.geometry.layers.len() + priced.layer];
                total[priced.class] += groups * priced.slice_bytes;
            }
        }
        total.map(|bytes| bytes / WINDOWS as f64 / 1e6)
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
    fn peek_row(&self, bucket: usize, id: u64, row: usize) -> ClassBytes {
        if self.saturated { return self.peek_row_counts(bucket, id, row); }
        let topk = usize::from(self.geometry.topk);
        let words = usize::from(self.geometry.experts).div_ceil(64);
        let layers = self.geometry.layers.len();
        let history = self.history.get(&id);
        let mut total = [0.; MAX_RESOURCE_CLASSES];
        for window in 0..WINDOWS {
            let routes = history.and_then(|h| h.get(window + row));
            let base = window * layers;
            for priced in &self.priced {
                let groups = match routes {
                    Some(routes) => {
                        let occupied = &self.occupancy[(base + priced.layer) * words..][..words];
                        routes[priced.layer * topk..][..topk].iter()
                            .filter(|&&e| occupied[usize::from(e) >> 6] >> (e & 63) & 1 == 0).count() as f64
                    }
                    None if row == 0 => topk as f64,
                    None => self.novelty[bucket][priced.layer],
                };
                total[priced.class] += groups * priced.slice_bytes;
            }
        }
        let total = total.map(|bytes| bytes / WINDOWS as f64 / 1e6);
        debug_assert_eq!(total, self.peek_row_counts(bucket, id, row));
        total
    }

    /// The count-table forecast `peek_row` must equal.
    fn peek_row_counts(&self, bucket: usize, id: u64, row: usize) -> ClassBytes {
        let topk = usize::from(self.geometry.topk);
        let mut total = [0.; MAX_RESOURCE_CLASSES];
        for window in 0..WINDOWS {
            let routes = self.stand_in(id, row, window);
            for priced in &self.priced {
                let groups = match routes {
                    Some(routes) => routes[priced.layer * topk..][..topk].iter()
                        .filter(|&&e| self.counts[self.count_index(window, priced.layer, e)] % u16::from(priced.group_rows) == 0)
                        .count() as f64,
                    None if row == 0 => topk as f64,
                    None => self.novelty[bucket][priced.layer],
                };
                total[priced.class] += groups * priced.slice_bytes;
            }
        }
        total.map(|bytes| bytes / WINDOWS as f64 / 1e6)
    }

    fn add_row(&mut self, bucket: usize, id: u64, row: usize, _request: usize) -> ClassBytes {
        let delta = self.peek_row(bucket, id, row);
        let topk = usize::from(self.geometry.topk);
        for window in 0..WINDOWS {
            let Some(token) = self.history.get(&id).and_then(|h| h.get(window + row)) else {
                if self.dynamic_grouping {
                    for priced in &self.priced {
                        self.unknown_groups[window * self.geometry.layers.len() + priced.layer] +=
                            if row == 0 { topk as f64 } else { self.novelty[bucket][priced.layer] };
                    }
                }
                continue;
            };
            for priced in &self.priced {
                let words = usize::from(self.geometry.experts).div_ceil(64);
                let slot = window * self.geometry.layers.len() + priced.layer;
                for &expert in &token[priced.layer * topk..][..topk] {
                    let index = slot * usize::from(self.geometry.experts) + usize::from(expert);
                    self.counts[index] = self.counts[index].saturating_add(1);
                    self.occupancy[slot * words + (usize::from(expert) >> 6)] |= 1 << (expert & 63);
                    self.saturated |= self.counts[index] >= u16::from(priced.group_rows);
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
        if rows > self.geometry.verify_capacity() { return Err(DraftPolicyError::RequestExtent); }
        if round.requests.is_empty() || round.requests.len() > self.geometry.max_requests
            || round.requests.iter().any(|r| r.rows == 0 || r.accepted == 0 || r.accepted > r.rows)
            || round.drafted_requests.is_some_and(|n| n > round.requests.len()) {
            return Err(DraftPolicyError::RequestExtent);
        }
        if round.predicted_us.is_some() && round.requests.iter().any(|r| self.selected_requests.get(&r.id) != Some(&round.requests.len())) {
            return Err(DraftPolicyError::RequestExtent);
        }
        for r in round.requests {
            self.selected_requests.remove(&r.id);
            self.selected_rows.remove(&r.id);
        }
        let bucket = draft_concurrency_bucket(round.requests.len());
        let drafted_requests = round.drafted_requests.unwrap_or(round.requests.len()) as f64;
        let layer_routes = |layer: usize| &round.routes[layer * rows * topk..][..rows * topk];
        let drafted = round.draft_us.is_finite();
        let draft_us = if drafted { round.draft_us } else { 0. };
        // Layer times may be sampled on a cadence: the class fits learn only
        // from rounds that carry every priced layer's time, while prediction
        // health and the round residual (the total less the class fits'
        // prediction) learn from every timed round.
        let complete = self.priced.iter().all(|p| round.layer_us.get(p.layer).copied().flatten().is_some());
        if round.total_us.is_finite() {
            let mut megabytes = [0.; MAX_RESOURCE_CLASSES];
            let mut class_us = [0.; MAX_RESOURCE_CLASSES];
            for index in 0..self.priced.len() {
                let priced = self.priced[index];
                megabytes[priced.class] += self.route_groups(layer_routes(priced.layer), self.geometry.group_rows(priced.layer, rows))
                    * priced.slice_bytes / 1e6;
                class_us[priced.class] += round.layer_us[priced.layer].unwrap_or(0.);
            }
            // Out of sample: the fits' price of the executed shape before
            // they learn from it, against the round's wall time.
            let predicted = self.cost.predict(bucket, rows as f64, megabytes)
                + if drafted { self.cost.draft_us(bucket, drafted_requests, round.wide) } else { 0. };
            let relative = ((round.total_us - predicted) / round.total_us.max(1.)).abs().min(1.);
            // Evidence, forgetting, and engagement are local to this regime.
            let (count, error) = &mut self.timed[bucket];
            *count += 1;
            *error += (relative - *error) * (1. / *count as f64).max(ERROR_RATE);
            self.cost.observe_round(bucket, rows as f64, megabytes, complete.then_some(&class_us[..]),
                round.total_us - draft_us);
            self.engaged[bucket] |= self.cost.warm(bucket) && *count >= self.seed.warm_rounds
                && *error <= self.seed.warm_error;
        }
        if drafted {
            let wide = usize::from(round.wide);
            self.cost.observe_draft(bucket, drafted_requests, round.wide, round.draft_us);
            self.stats.width_rounds[wide] += 1;
            self.since_width[bucket][wide] = 0;
            self.since_width[bucket][1 - wide] += 1;
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
                        self.novelty[bucket][layer] = 0.995 * self.novelty[bucket][layer] + 0.005 * novel as f64;
                    }
                }
                history.push_front(token);
                history.truncate(HISTORY);
            }
            offset += request.rows;
            let drafts = request.rows - 1;
            let accepted = request.accepted - 1;
            let eligible = self.selected_draft_room.remove(&request.id).unwrap_or(false);
            if drafts == 0 {
                if eligible {
                    let age = self.skipped.entry(request.id).or_default();
                    *age = age.saturating_add(1);
                }
                if let Some(recent) = self.confidence[bucket].recent.get_mut(&request.id) {
                    for (position, value) in recent.iter_mut().enumerate() {
                        let pooled = self.confidence[bucket].pooled[usize::from(position >= *self.geometry.widths.last().unwrap())]
                            .apply(self.confidence[bucket].raw_mean[position]);
                        if let Some(value) = value { *value += SKIP_FORGET_RATE * (pooled - *value); }
                    }
                }
                if let Some(counts) = self.confidence[bucket].outcomes.get_mut(&request.id) {
                    for count in counts { count.0 *= OUTCOME_DECAY; count.1 *= OUTCOME_DECAY; }
                }
            } else {
                self.skipped.remove(&request.id);
            }
            self.stats.verified_drafts += drafts as u64;
            self.stats.accepted_drafts += accepted as u64;
            self.stats.emitted_requests += 1;
            self.stats.draft_rows[drafts.min(positions)] += 1;
            if let Some(confidence) = request.confidence.filter(|_| drafts > 0) {
                // Every drafted position's calibrated confidence feeds the
                // request's short-memory estimate used to choose the next
                // round's width, whether or not the position was verified.
                let calibrated: Vec<f64> = confidence.iter().enumerate().take(positions)
                    .map(|(position, &raw)| self.calibrated(bucket, position, raw)).collect();
                let recent = self.confidence[bucket].recent.entry(request.id).or_insert_with(|| vec![None; positions]);
                for ((position, &raw), calibrated) in confidence.iter().enumerate().take(positions).zip(calibrated) {
                    recent[position] = Some(match recent[position] {
                        Some(previous) => previous + RECENT_RATE * (calibrated - previous),
                        None => calibrated,
                    });
                    self.confidence[bucket].raw_mean[position] += 0.02 * (raw - self.confidence[bucket].raw_mean[position]);
                }
                // Only positions whose predecessors were all accepted carry
                // evidence about the conditional acceptance probability.
                let reached = drafts.min(confidence.len()).min(accepted + 1).min(positions);
                for position in 0..reached {
                    let outcome = f64::from(u8::from(position < accepted));
                    let calibrated = self.calibrated(bucket, position, confidence[position]);
                    self.confidence[bucket].reached[position] += 1;
                    self.stats.position_reached[position] += 1;
                    self.stats.position_confidence[position] += calibrated;
                    self.stats.position_raw_confidence[position] += confidence[position];
                    self.stats.position_accepted[position] += u64::from(position < accepted);
                    self.confidence[bucket].calibration[position].observe(confidence[position], outcome);
                    self.confidence[bucket].pooled[usize::from(position >= *self.geometry.widths.last().unwrap())].observe(confidence[position], outcome);
                }
                let counts = self.confidence[bucket].outcomes.entry(request.id).or_insert_with(|| vec![(0., 0.); positions]);
                for count in counts.iter_mut() { count.0 *= OUTCOME_DECAY; count.1 *= OUTCOME_DECAY; }
                for position in 0..reached {
                    counts[position].0 += f64::from(u8::from(position < accepted));
                    counts[position].1 += 1.;
                }
            }
        }
        if let Some(simple) = &mut self.simple { simple.observe(&round); }
        self.stats.rounds += 1;
        self.stats.verified_rows += rows as u64;
        if let Some(predicted) = round.predicted_us {
            self.stats.selected_rounds += 1;
            if round.total_us.is_finite() {
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
            group_rows_by_rows: Vec::new(),
            experts: 384, topk: 6, max_requests: 16, max_positions: 7, widths,
            classes: vec![ResourceClass::rtx(), ResourceClass::spark()], regimes: 2,
        }
    }

    fn hybrid_geometry() -> PolicyGeometry {
        PolicyGeometry {
            layers: vec![
                LayerResource { class: Some(0), slice_bytes: 1e6, group_rows: 4, timed: false },
                LayerResource { class: Some(0), slice_bytes: 1e6, group_rows: 4, timed: true },
            ],
            group_rows_by_rows: (0..=16).map(|rows| vec![if (1..=4).contains(&rows) { 1 } else { 4 }; 2]).collect(),
            experts: 4, topk: 1, max_requests: 4, max_positions: 7, widths: vec![5, 7],
            classes: vec![ResourceClass::rtx()], regimes: 1,
        }
    }

    #[test]
    fn dense_only_geometry_selects_and_learns_without_synthetic_routes() {
        let mut geometry = hybrid_geometry();
        for layer in &mut geometry.layers { layer.class = None; layer.slice_bytes = 0.; }
        geometry.experts = 0;
        geometry.topk = 0;
        let mut policy = DraftPolicy::seeded(geometry, PolicySeed::weak(5)).unwrap();
        assert!(policy.priced.is_empty());
        assert!(policy.counts.is_empty() && policy.scratch.is_empty() && policy.occupancy.is_empty());
        let confidence = [0.9; 7];
        let candidates = [DraftCandidate { id: 1, confidence: &confidence }];
        policy.engaged[draft_concurrency_bucket(1)] = true;
        let selection = policy.select(false, &candidates, 5).unwrap().unwrap();
        assert!(selection.predicted_us.is_finite() && selection.predicted_us > 0.);
        assert_eq!(policy.forecast_megabytes(&[1], &selection.lengths), vec![0.]);
        let requests = [ObservedDraftRequest { id: 1, rows: 1 + selection.lengths[0],
            accepted: 1, confidence: Some(&confidence) }];
        policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &[],
            layer_us: &[None, None], total_us: 8000., draft_us: 2600., drafted_requests: Some(1),
            wide: false, predicted_us: Some(selection.predicted_us) }).unwrap();
        assert_eq!(policy.stats.rounds, 1);
        assert!(policy.history[&1].iter().all(|routes| routes.is_empty()));
        assert!(policy.select(false, &candidates, 7).unwrap().unwrap().predicted_us.is_finite());
        policy.release(1);
        assert!(!policy.history.contains_key(&1));
    }

    #[test]
    fn dense_zero_routes_reject_inconsistent_routed_geometry() {
        for (experts, topk, routed) in [(0, 0, true), (0, 1, false), (4, 0, false),
            (0, 1, true), (4, 0, true)] {
            let mut geometry = hybrid_geometry();
            geometry.experts = experts;
            geometry.topk = topk;
            if !routed { for layer in &mut geometry.layers { layer.class = None; } }
            assert!(DraftPolicy::seeded(geometry, PolicySeed::weak(5)).is_err(),
                "experts={experts}, topk={topk}, routed={routed}");
        }
    }

    #[test]
    fn installed_grouping_schedule_validates_full_rows_and_layers() {
        let mut geometry = hybrid_geometry();
        geometry.group_rows_by_rows.truncate(4);
        assert!(DraftPolicy::seeded(geometry, PolicySeed::weak(5)).is_err());
        for malformed in [vec![0, 4], vec![4]] {
            let mut geometry = hybrid_geometry();
            geometry.group_rows_by_rows[5] = malformed;
            assert!(DraftPolicy::seeded(geometry, PolicySeed::weak(5)).is_err());
        }
    }

    #[test]
    fn live_grouping_reprices_existing_routes_across_compact_boundary() {
        let mut policy = DraftPolicy::seeded(hybrid_geometry(), PolicySeed::weak(5)).unwrap();
        policy.history.insert(1, (0..24).map(|_| vec![0u16; 2].into_boxed_slice()).collect());
        assert_eq!(policy.forecast_megabytes(&[1], &[1]), vec![2.]);
        assert_eq!(policy.forecast_megabytes(&[1], &[3]), vec![4.]);
        assert_eq!(policy.forecast_megabytes(&[1], &[4]), vec![2.]);
        assert_eq!(policy.forecast_megabytes(&[1], &[7]), vec![2.]);
        // The forward-growth trajectory crosses the cheaper grouped boundary.
        let selection = policy.select_core(0, &[1], &[vec![0.9; 7]], 100., 16, false);
        let exact = policy.predict_bucket_draft(0, &[1], &selection.lengths, 100.);
        assert!((selection.predicted_us - exact).abs() < 1e-9);
        assert!(selection.lengths[0] >= 4);
        assert!(!policy.wide_cannot_win(0, &[1], &[vec![0.9; 7]], 100., 100.));
    }

    #[test]
    fn grouping_boundaries_preserve_unknown_novelty_and_observe_actual_rows() {
        let mut policy = DraftPolicy::seeded(hybrid_geometry(), PolicySeed::weak(5)).unwrap();
        let selection = policy.select_core(0, &[1, 2], &[vec![0.9; 7], vec![0.8; 7]], 100., 16, false);
        let exact = policy.predict_bucket_draft(0, &[1, 2], &selection.lengths, 100.);
        assert!((selection.predicted_us - exact).abs() < 1e-9);
        let mut constant = policy.geometry.clone();
        constant.group_rows_by_rows.clear();
        let mut constant = DraftPolicy::seeded(constant, PolicySeed::weak(5)).unwrap();
        let requests = [ObservedDraftRequest { id: 1, rows: 5, accepted: 1, confidence: None }];
        let routes = vec![0; 10];
        let round = DraftRoundObservation { shared: false, requests: &requests, routes: &routes,
            layer_us: &[None, Some(1000.)], total_us: 5000., draft_us: f64::NAN,
            drafted_requests: Some(0), wide: false, predicted_us: None };
        // Use fresh policies so the same pre-observation fits see the same true traffic.
        policy = DraftPolicy::seeded(hybrid_geometry(), PolicySeed::weak(5)).unwrap();
        policy.observe(round).unwrap();
        constant.observe(round).unwrap();
        assert_eq!(policy.cost_snapshot(false, 1), constant.cost_snapshot(false, 1));
        let overflow = [ObservedDraftRequest { rows: 17, ..requests[0] }];
        assert_eq!(policy.observe(DraftRoundObservation { requests: &overflow, routes: &[0; 34], ..round }),
            Err(DraftPolicyError::RequestExtent));
    }

    #[test]
    fn constant_installed_schedule_retains_static_decisions_and_capacity() {
        let geometry = placement(8);
        let mut scheduled = geometry.clone();
        scheduled.group_rows_by_rows = vec![vec![16; 40]; 129];
        let mut left = DraftPolicy::seeded(geometry, PolicySeed::weak(5)).unwrap();
        let mut right = DraftPolicy::seeded(scheduled, PolicySeed::weak(5)).unwrap();
        assert!(!right.dynamic_grouping);
        for policy in [&mut left, &mut right] {
            policy.history.insert(1, (0..24).map(|_| vec![0; 40 * 6].into_boxed_slice()).collect());
        }
        for n in 0..8 {
            assert_eq!(left.forecast_megabytes(&[1], &[n]), right.forecast_megabytes(&[1], &[n]));
        }
        let l = left.select_core(0, &[1], &[vec![0.9; 7]], 1000., 128, false);
        let r = right.select_core(0, &[1], &[vec![0.9; 7]], 1000., 128, false);
        assert_eq!(l, r);
        assert!(right.forecast_megabytes(&[1], &[128])[0].is_nan());
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
            layer_us: &layer_us, total_us: total, predicted_us: predicted, draft_us: f64::NAN, drafted_requests: None, wide: false }).unwrap();
        *token += accepted as u64;
        total
    }

    /// One untimed C1 round of request `id`, extending its route history.
    fn truth_round_id(policy: &mut DraftPolicy, _truth: &Truth, token: &mut u64, id: u64) {
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|l| vec![token_routes(*token, 24)[l]]).collect();
        let requests = [ObservedDraftRequest { id, rows: 1, accepted: 1, confidence: None }];
        policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
            layer_us: &[None; 40], total_us: f64::NAN, predicted_us: None, draft_us: f64::NAN, drafted_requests: None,
            wide: false }).unwrap();
        *token += 1;
    }

    /// A round in which every layer and the residual ran `factor` times slow.
    fn stalled_round(policy: &mut DraftPolicy, truth: &Truth, token: &mut u64, rows: usize, factor: f64) {
        let token_rows: Vec<_> = (0..rows).map(|r| token_routes(*token + r as u64, 24)).collect();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|l| token_rows.iter().map(|t| t[l]).collect()).collect();
        let mut layer_us = [None; 40];
        let mut total = factor * (truth.round[0] + truth.round[1] * rows as f64 + truth.round[2]);
        for layer in 1..40 {
            let class = policy.geometry.layers[layer].class.unwrap() as usize;
            let mb = groups(&routes[layer]) * policy.geometry.layers[layer].slice_bytes / 1e6;
            let t = factor * (truth.alpha[class] + truth.beta[class] * rows as f64 + truth.us_per_mb[class] * mb);
            layer_us[layer] = Some(t);
            total += t;
        }
        let requests = [ObservedDraftRequest { id: 1, rows, accepted: rows, confidence: None }];
        policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
            layer_us: &layer_us, total_us: total, predicted_us: None, draft_us: f64::NAN, drafted_requests: None, wide: false }).unwrap();
        *token += rows as u64;
    }

    fn truth() -> Truth {
        Truth { alpha: [600., 700.], beta: [8., 12.], us_per_mb: [0.9, 7.5], round: [9000., 250., 150.] }
    }

    #[test]
    fn evaluate_prices_explicit_shapes_as_select_does() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let candidates = [DraftCandidate { id: 1, confidence: &[0.9, 0.8, 0.7, 0.6, 0.5] }];
        assert_eq!(policy.evaluate(false, &candidates, &[3], 5), None, "cold fit");
        let (truth, mut token) = (truth(), 0);
        for round in 0..400 { let rows = 1 + round % 8; run_round(&mut policy, &truth, &mut token, rows, rows, None); }
        let best = policy.select(false, &candidates, 5).unwrap().unwrap();
        let (expected, us) = policy.evaluate(false, &candidates, &best.lengths, 5).unwrap();
        assert!((expected - best.expected_tokens).abs() < 1e-9, "{expected} vs {best:?}");
        assert!((us - best.predicted_us).abs() < 1e-6 * us, "{us} vs {best:?}");
        // Any other shape is no better than the selection.
        for n in 0..=5 {
            let (e, t) = policy.evaluate(false, &candidates, &[n], 5).unwrap();
            assert!(e / t <= best.expected_tokens / best.predicted_us + 1e-12, "{n}");
        }
        assert_eq!(policy.evaluate(false, &candidates, &[6], 5), None, "past the confidence");
    }

    #[test]
    fn estimator_recovers_coefficients_and_predicts_rounds() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let truth = truth();
        let mut token = 0;
        for round in 0..400 {
            let rows = 1 + round % 8;
            run_round(&mut policy, &truth, &mut token, rows, rows, None);
        }
        assert!(policy.warm(false, 1));
        let snapshot = policy.cost_snapshot(false, 1);
        let remote = snapshot.layers[1];
        // The traffic price stays near its physical prior (5.7) against a
        // 7.5 truth: the per-layer row price (truth 12 µs) absorbs the rest.
        assert!((remote[1] - 5.7).abs() < 1., "remote µs/MB {remote:?}");
        assert!(remote[2] > 12., "remote µs per layer row {remote:?}");
        // Prediction of an explicit shape against the same truth.
        let lengths = [4];
        let predicted = policy.predict(false, &[1], &lengths, 5).unwrap();
        let actual = run_round(&mut policy, &truth, &mut token, 5, 5, Some(predicted));
        assert!((predicted - actual).abs() / actual < 0.05, "{predicted} vs {actual}");
    }

    #[test]
    fn certain_drafts_are_always_verified_and_hopeless_ones_trimmed() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
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
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
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
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        assert_eq!(policy.select(false, &[DraftCandidate { id: 1, confidence: &[0.1; 5] }], 5), Ok(None));
        let mut fixed = DraftPolicy::new(placement(5), PolicySeed::weak(7), true).unwrap();
        let truth = truth();
        let mut token = 0;
        for _ in 0..300 { run_round(&mut fixed, &truth, &mut token, 6, 3, None); }
        assert!(fixed.warm(false, 1));
        assert_eq!(fixed.select(false, &[DraftCandidate { id: 1, confidence: &[0.1; 5] }], 5), Ok(None));
        assert_eq!(fixed.stats().draft_rows[5], 300);
        assert_eq!(fixed.stats().accepted_drafts, 600);
    }

    #[test]
    fn shared_experts_across_requests_are_charged_once() {
        let mut policy = DraftPolicy::new(placement(0), PolicySeed::weak(7), false).unwrap();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]]).collect();
        let layer_us = [None; 40];
        for id in [1, 2] {
            for _ in 0..4 {
                let requests = [ObservedDraftRequest { id, rows: 1, accepted: 1, confidence: None }];
                policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
                    layer_us: &layer_us, total_us: f64::NAN, predicted_us: None, draft_us: f64::NAN, drafted_requests: None, wide: false }).unwrap();
            }
        }
        policy.clear_counts();
        let first = policy.add_row(0, 1, 0, 0);
        let second = policy.add_row(0, 2, 0, 1);
        assert!(first[1] > 0.);
        assert_eq!(second[1], 0.);
    }

    #[test]
    fn groups_follow_the_kernel_row_tiles() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let rows: Vec<u16> = [1, 2, 3, 4, 5, 6].repeat(17);
        assert_eq!(policy.route_groups(&rows, 16), 12.);
        assert_eq!(policy.route_groups(&rows[..96], 16), 6.);
        assert_eq!(policy.route_groups(&rows, 8), 18.);
    }

    #[test]
    fn confidence_reliability_counts_only_reached_positions() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]; 5]).collect();
        let confidence = [0.9, 0.8, 0.7, 0.6];
        let requests = [ObservedDraftRequest { id: 1, rows: 5, accepted: 2, confidence: Some(&confidence) }];
        policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
            layer_us: &[None; 40], total_us: f64::NAN, predicted_us: None, draft_us: f64::NAN, drafted_requests: None, wide: false }).unwrap();
        let stats = policy.stats();
        assert_eq!(stats.position_reached, [1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(stats.position_accepted, [1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(stats.draft_rows[4], 1);
    }

    #[test]
    fn overconfident_positions_are_recalibrated_from_reached_outcomes() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
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
                assert!((policy.calibrated(0, 5, 0.95) - 0.8).abs() < 0.05, "{:?}", policy.calibration(1));
            }
            let truth = [0.95, 0.95, 0.95, 0.95, 0.95, 0.8, 0.8];
            let accepted = truth.iter().take_while(|&&p| uniform() < p).count();
            let requests = [ObservedDraftRequest { id: 1, rows: 8, accepted: accepted + 1, confidence: Some(&confidence) }];
            policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
                layer_us: &[None; 40], total_us: f64::NAN, predicted_us: None, draft_us: f64::NAN, drafted_requests: None, wide: false }).unwrap();
        }
        assert!((policy.calibrated(0, 0, 0.95) - 0.95).abs() < 0.02, "{:?}", policy.calibration(1));
        assert!((policy.calibrated(0, 5, 0.95) - 0.8).abs() < 0.03, "{:?}", policy.calibration(1));
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
            layer_us: &layer_us, total_us: total, predicted_us: None, draft_us, drafted_requests: None, wide: width > 5 }).unwrap();
        *token += accepted as u64;
    }

    #[test]
    fn skipped_confidence_forgets_toward_its_source_pool() {
        let mut geometry = placement(5);
        geometry.max_positions = 9;
        let mut policy = DraftPolicy::new(geometry, PolicySeed::weak(7), false).unwrap();
        let mut token = 0;
        for _ in 0..400 { width_round(&mut policy, &mut token, 99, 5, 6, 6, &[0.9; 5]); }
        policy.confidence[0].recent.insert(1, vec![Some(0.01); 9]);
        policy.confidence[0].pooled[1].theta = [0., -4.];
        let head = policy.confidence[0].pooled[0].apply(policy.confidence[0].raw_mean[0]);
        let tail = policy.confidence[0].pooled[1].apply(policy.confidence[0].raw_mean[7]);
        let before = policy.confidence[0].recent[&1].clone();
        width_round(&mut policy, &mut token, 1, 5, 1, 1, &[]);
        assert!((policy.confidence[0].recent[&1][0].unwrap() - (0.01 + SKIP_FORGET_RATE * (head - 0.01))).abs() < 1e-12);
        assert!((policy.confidence[0].recent[&1][7].unwrap() - (0.01 + SKIP_FORGET_RATE * (tail - 0.01))).abs() < 1e-12);
        assert_ne!(before, policy.confidence[0].recent[&1]);
        assert!(!policy.skipped.contains_key(&1), "unselected rounds do not create exploration obligations");
    }

    #[test]
    fn request_recovers_from_early_low_confidence_with_minimal_evidence() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let mut token = 0;
        for _ in 0..400 { width_round(&mut policy, &mut token, 99, 5, 6, 6, &[0.9; 5]); }
        for _ in 0..8 { width_round(&mut policy, &mut token, 1, 5, 2, 1, &[0.01; 5]); }
        let actions = [DraftAction { width: 0, draft_us: 0. }, DraftAction { width: 5, draft_us: 1500. }];
        let mut first_probe = None;
        let mut recovered = false;
        for round in 0..32 {
            let action = policy.choose_action(false, &[(1, 5)], &actions, 64).unwrap().unwrap();
            if action == 0 {
                width_round(&mut policy, &mut token, 1, 0, 1, 1, &[]);
            } else {
                first_probe.get_or_insert(round);
                let selected = policy.select(false, &[DraftCandidate { id: 1, confidence: &[0.99; 5] }], 5)
                    .unwrap().unwrap();
                assert!(selected.lengths[0] > 0, "recovery must verify evidence, not just generate it");
                width_round(&mut policy, &mut token, 1, 5, selected.lengths[0] + 1,
                    selected.lengths[0] + 1, &[0.99; 5]);
                recovered |= policy.choose_action(false, &[(1, 5)], &actions, 64).unwrap() == Some(1);
                if recovered { break; }
            }
        }
        assert!(first_probe.is_some_and(|round| round <= REQUEST_EXPLORE_SKIPS as usize));
        assert!(recovered, "later acceptance must restore useful drafting within 32 rounds");
        policy.release(1);
        assert!(!policy.skipped.contains_key(&1) && !policy.selected_draft_room.contains_key(&1));
    }

    #[test]
    fn request_probes_share_a_budget_without_starving_other_members() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let mut token = 0;
        for _ in 0..400 { width_round(&mut policy, &mut token, 99, 5, 6, 6, &[0.9; 5]); }
        for id in 1..=3 { policy.skipped.insert(id, REQUEST_EXPLORE_SKIPS + id); }
        policy.engaged[1] = true;
        let candidates = [DraftCandidate { id: 1, confidence: &[0.001; 5] },
            DraftCandidate { id: 2, confidence: &[0.001; 5] }, DraftCandidate { id: 3, confidence: &[] }];
        let selected = policy.select_within(false, &candidates, 5, 4).unwrap().unwrap();
        assert_eq!(selected.lengths, [0, 1, 0], "oldest eligible member gets the single spare row");
        let (expected, time) = policy.evaluate(false, &candidates, &selected.lengths, 5).unwrap();
        assert!((selected.expected_tokens - expected).abs() < 1e-12);
        assert!((selected.predicted_us - time).abs() < 1e-6);
        let anchor_only = policy.select_within(false, &candidates, 5, 3).unwrap().unwrap();
        assert_eq!(anchor_only.lengths, [0, 0, 0]);
        // Another lane's successful drafting cannot reset this request's age.
        width_round(&mut policy, &mut token, 99, 5, 6, 6, &[0.9; 5]);
        assert_eq!(policy.skipped[&1], REQUEST_EXPLORE_SKIPS + 1);
    }

    #[test]
    fn request_exploration_uses_the_smallest_supported_pass_when_eligible() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let mut token = 0;
        for _ in 0..400 { width_round(&mut policy, &mut token, 99, 5, 6, 6, &[0.9; 5]); }
        let actions = [DraftAction { width: 0, draft_us: 0. }, DraftAction { width: 5, draft_us: 1e12 },
            DraftAction { width: 1, draft_us: 1e12 }];
        for round in 0..REQUEST_EXPLORE_SKIPS {
            assert_eq!(policy.choose_action(false, &[(1, 5)], &actions, 64).unwrap(), Some(0));
            width_round(&mut policy, &mut token, 1, 0, 1, 1, &[]);
            assert_eq!(policy.skipped[&1], round + 1);
        }
        assert_eq!(policy.choose_action(false, &[(1, 5)], &actions, 1).unwrap(), Some(0));
        assert_eq!(policy.choose_action(false, &[(1, 0)], &actions, 64).unwrap(), Some(0));
        assert_eq!(policy.choose_action(false, &[(1, 5)], &actions[..1], 64).unwrap(), Some(0),
            "an overdue request cannot invent an unsupported positive action");
        assert_eq!(policy.choose_action(false, &[(1, 5)], &actions, 64).unwrap(), Some(2));
        assert_eq!(policy.skipped[&1], REQUEST_EXPLORE_SKIPS, "planning alone never resets the request");
        width_round(&mut policy, &mut token, 1, 1, 2, 2, &[0.99]);
        assert!(!policy.skipped.contains_key(&1));
    }

    #[test]
    fn zero_selected_members_age_on_observation_not_selection() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let mut token = 0;
        for _ in 0..400 { width_round(&mut policy, &mut token, 99, 5, 6, 6, &[0.9; 5]); }
        let candidate = [DraftCandidate { id: 1, confidence: &[0.001; 5] }];
        for round in 0..REQUEST_EXPLORE_SKIPS {
            for _ in 0..3 {
                let selected = policy.select_within(false, &candidate, 5, 2).unwrap().unwrap();
                assert_eq!(selected.lengths, [0]);
                assert_eq!(policy.skipped.get(&1).copied().unwrap_or(0), round);
            }
            width_round(&mut policy, &mut token, 1, 5, 1, 1, &[0.001; 5]);
        }
        assert_eq!(policy.skipped[&1], REQUEST_EXPLORE_SKIPS);
        let selected = policy.select_within(false, &candidate, 5, 2).unwrap().unwrap();
        assert_eq!(selected.lengths, [1], "post-draft zero members receive one verified probe");
        // Grammar can censor that chosen row. It is not a policy skip and
        // retains the existing age so the request can retry when eligible.
        width_round(&mut policy, &mut token, 1, 5, 1, 1, &[]);
        assert_eq!(policy.skipped[&1], REQUEST_EXPLORE_SKIPS);
        policy.select_within(false, &candidate, 5, 2).unwrap();
        width_round(&mut policy, &mut token, 1, 5, 2, 2, &[0.99]);
        assert!(!policy.skipped.contains_key(&1));
    }

    #[test]
    fn pre_draft_actions_share_row_prices_and_explore_unused_depths() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let actions = [DraftAction { width: 0, draft_us: 0. }, DraftAction { width: 3, draft_us: 100. },
            DraftAction { width: 5, draft_us: 1e12 }];
        assert_eq!(policy.choose_action(false, &[(1, 5)], &actions, 64).unwrap(), None);
        let mut token = 0;
        for _ in 0..400 { width_round(&mut policy, &mut token, 1, 5, 6, 6, &[0.99; 5]); }
        assert_eq!(policy.choose_action(false, &[(1, 5)], &actions, 64).unwrap(), Some(1));
        // With only anchor room, paying for a draft cannot help.
        assert_eq!(policy.choose_action(false, &[(1, 5)], &actions, 1).unwrap(), Some(0));
        let mut explored = false;
        for _ in 0..WIDTH_EXPLORE_ROUNDS {
            explored |= policy.choose_action(false, &[(1, 5)], &actions, 64).unwrap() == Some(2);
        }
        assert!(explored);
        for _ in 0..2 * WIDTH_EXPLORE_ROUNDS {
            assert_eq!(policy.choose_action(false, &[(1, 5)], &actions, 1).unwrap(), Some(0));
            assert_eq!(policy.choose_action(false, &[(1, 0)], &actions, 64).unwrap(), Some(0));
        }
        assert!(policy.choose_action(false, &[], &actions, 64).is_err());
        assert!(policy.choose_action(false, &[(1, 5)], &[], 64).is_err());
        assert!(policy.choose_action(false, &[(1, 5)], &[DraftAction { width: 0, draft_us: 1. }], 64).is_err());
        assert!(policy.choose_action(false, &[(1, 5)], &[DraftAction { width: 3, draft_us: f64::NAN }], 64).is_err());
        assert!(policy.choose_action(false, &[(1, 5)], &[DraftAction { width: 8, draft_us: 1. }], 64).is_err());
        let mut fixed = DraftPolicy::new(placement(5), PolicySeed::weak(7), true).unwrap();
        assert_eq!(fixed.choose_action(false, &[(1, 5)], &[
            DraftAction { width: 5, draft_us: 100. }, DraftAction { width: 0, draft_us: 0. }], 64).unwrap(), Some(0));
    }

    #[test]
    fn action_rooms_reserve_all_anchors_and_recover_only_participants() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let mut token = 0;
        for _ in 0..400 { width_round(&mut policy, &mut token, 99, 5, 6, 6, &[0.99; 5]); }
        policy.engaged[1] = true;
        let requests = [(1, 5), (2, 5)];
        let actions = [DraftAction { width: 0, draft_us: 0. },
            DraftAction { width: 3, draft_us: 1e12 }, DraftAction { width: 5, draft_us: 1e12 }];
        let masks: [&[usize]; 3] = [&[0, 0], &[3, 0], &[0, 5]];
        policy.skipped.insert(2, REQUEST_EXPLORE_SKIPS);
        assert_eq!(policy.choose_action_with_rooms(false, &requests, &actions, &masks, 3).unwrap(), Some(2));
        assert_eq!(policy.selected_requests[&1], 2);
        assert_eq!(policy.selected_requests[&2], 2);
        assert!(policy.selected_draft_room[&1] && policy.selected_draft_room[&2]);
        assert_eq!(policy.choose_action_with_rooms(false, &requests, &actions, &masks, 2).unwrap(), Some(0));
        assert!(!policy.selected_draft_room[&1] && !policy.selected_draft_room[&2]);
        let none: [&[usize]; 3] = [&[0, 0], &[0, 0], &[0, 0]];
        for _ in 0..2 * WIDTH_EXPLORE_ROUNDS {
            assert_eq!(policy.choose_action_with_rooms(false, &requests, &actions, &none, 64).unwrap(), Some(0));
        }
        assert!(policy.choose_action_with_rooms(false, &requests, &actions, &masks[..2], 64).is_err());
        assert!(policy.choose_action_with_rooms(false, &requests, &actions,
            &[&[0, 0], &[6, 0], &[0, 5]], 64).is_err());
        assert!(policy.choose_action_with_rooms(false, &requests, &actions,
            &[&[1, 0], &[3, 0], &[0, 5]], 64).is_err());
    }

    #[test]
    fn action_rooms_match_legacy_and_price_distinct_participant_shapes() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let mut token = 0;
        for _ in 0..400 { width_round(&mut policy, &mut token, 99, 5, 6, 6, &[0.99; 5]); }
        policy.engaged[1] = true;
        let requests = [(1, 5), (2, 5), (3, 5), (4, 5)];
        let actions = [DraftAction { width: 0, draft_us: 0. },
            DraftAction { width: 3, draft_us: policy.draft_cost_drafted(false, 4, 4, 3) },
            DraftAction { width: 5, draft_us: policy.draft_cost_drafted(false, 4, 1, 5) }];
        let mut legacy = policy.clone();
        let full: [&[usize]; 3] = [&[0; 4], &[5; 4], &[5; 4]];
        assert_eq!(legacy.choose_action(false, &requests, &actions, 16).unwrap(),
            policy.choose_action_with_rooms(false, &requests, &actions, &full, 16).unwrap());
        let masks: [&[usize]; 3] = [&[0; 4], &[3; 4], &[0, 0, 0, 5]];
        let chosen = policy.choose_action_with_rooms(false, &requests, &actions, &masks, 16).unwrap().unwrap();
        let ids = [1, 2, 3, 4];
        let mut best = (0, 0.);
        for (index, action) in actions.iter().enumerate() {
            let cumulative: Vec<Vec<_>> = masks[index].iter().map(|&room| {
                let mut product = 1.;
                (0..room.min(action.width)).map(|position| {
                    product *= policy.calibrated(1, position, policy.confidence[1].raw_mean[position]);
                    product
                }).collect()
            }).collect();
            let selection = policy.select_core(draft_concurrency_bucket(4), &ids, &cumulative,
                action.draft_us, 16, false);
            assert!(selection.lengths.iter().sum::<usize>() + 4 <= 16);
            assert!(selection.lengths.iter().zip(masks[index]).all(|(&length, &room)| length <= room));
            let ratio = selection.expected_tokens / selection.predicted_us;
            if ratio > best.1 { best = (index, ratio); }
        }
        assert_eq!(chosen, best.0);
    }

    #[test]
    fn native_confidence_extent_is_independent_of_execution_width() {
        let mut geometry = placement(5);
        geometry.widths = vec![3, 5];
        let mut policy = DraftPolicy::new(geometry, PolicySeed::weak(5), false).unwrap();
        policy.confidence[0].outcomes.insert(1, vec![(0., 10.); 7]);
        assert!(policy.request_probability(0, 1, 3, 0.9) < 0.9);
        policy.set_native_confidence_positions(5).unwrap();
        assert_eq!(policy.request_probability(0, 1, 3, 0.9), 0.9);
        assert_eq!(policy.request_probability(0, 1, 4, 0.9), 0.9);
        assert!(policy.request_probability(0, 1, 5, 0.9) < 0.9);
        assert_eq!(policy.restarted(false).native_confidence_positions, 5);
        assert!(policy.set_native_confidence_positions(0).is_err());
        assert!(policy.set_native_confidence_positions(8).is_err());
        policy.stats.rounds = 1;
        assert!(policy.set_native_confidence_positions(5).is_err());
    }

    #[test]
    fn mixed_batches_price_and_fit_only_the_drafted_sequences() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let mut token = 0;
        for _ in 0..400 { width_round(&mut policy, &mut token, 1, 5, 6, 6, &[0.99; 5]); }
        let candidates: Vec<_> = (0..16).map(|id| DraftCandidate { id,
            confidence: if id < 4 { &[0.99; 5] } else { &[] } }).collect();
        for round in 0..400 { surcharged_round(&mut policy, &mut token, 16 * (1 + round % 4), 16, 0., 0.); }
        let selection = policy.select_drafted_within(false, &candidates, 5, 64, 4).unwrap().unwrap();
        assert!(selection.lengths[4..].iter().all(|&n| n == 0));
        let (tokens, us) = policy.evaluate_drafted(false, &candidates, &selection.lengths, 5, 4).unwrap();
        assert!((tokens - selection.expected_tokens).abs() < 1e-10);
        assert!((us - selection.predicted_us).abs() < 1e-8);
        let (_, all_drafted_us) = policy.evaluate(false, &candidates, &selection.lengths, 5).unwrap();
        let extra = policy.cost.draft_us(3, 16., false) - policy.cost.draft_us(3, 4., false);
        assert!((all_drafted_us - us - extra).abs() < 1e-8);
        assert!(policy.select_drafted_within(false, &candidates, 5, 64, 17).is_err());

        let requests: Vec<_> = (0..16).map(|id| ObservedDraftRequest { id,
            rows: if id < 4 { 6 } else { 1 }, accepted: 1, confidence: None }).collect();
        let routes = vec![1u16; 40 * 36 * 6];
        let mut expected = policy.cost.draft[3].clone();
        expected.observe([1., 4., 0., 0.], 3_000.);
        policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &routes,
            layer_us: &[None; 40], total_us: f64::NAN, predicted_us: None, draft_us: 3_000.,
            drafted_requests: Some(4), wide: false }).unwrap();
        assert_eq!(format!("{:?}", policy.cost.draft[3]), format!("{expected:?}"));
        // Undrafted members still retain their verifier route history.
        assert!(policy.history.contains_key(&15));
    }

    #[test]
    fn width_follows_each_requests_recent_runs() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
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
        let fit = policy.cost_snapshot(false, 1).draft;
        assert!((fit[2] + fit[3] - 800.).abs() < 100., "wide draft extra {fit:?}");
        assert_eq!(policy.choose_width(false, &[(1, 7)]), 7);
        assert_eq!(policy.choose_width(false, &[(2, 7)]), 5);
        // A short remaining budget cannot use the wide positions.
        assert_eq!(policy.choose_width(false, &[(1, 4)]), 5);
    }

    #[test]
    fn unused_width_is_revisited() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
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
            let mut policy = DraftPolicy::new(geometry, PolicySeed::weak(7), fixed).unwrap();
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
                    layer_us: &layer_us, total_us: total, draft_us, drafted_requests: None, wide: drafted && width > policy.widths()[0],
                    predicted_us: predicted });
                transcript.word("observed", u64::from(result.is_ok()));
                for (r, &a) in live.iter_mut().zip(&accepted) { r.1 += a as u64; }
                if round % 97 == 50 {
                    // A route outside the model is rejected after earlier
                    // requests' histories were extended.
                    let mut bad = routes.clone();
                    bad[5][rows - 1][0] = 400;
                    let result = policy.observe(DraftRoundObservation { shared, requests: &observed, routes: &flat(&bad),
                        layer_us: &layer_us, total_us: total, draft_us, drafted_requests: None, wide: false, predicted_us: None });
                    transcript.word("rejected", u64::from(result.is_err()));
                }
                if round % 16 == 15 { record_state(transcript, &policy); }
            }
            record_state(transcript, &policy);
        }
    }

    fn record_state(transcript: &mut Transcript, policy: &DraftPolicy) {
        for shared in [false, true] {
            for requests in [1, 4, 8, 16] {
                transcript.word("warm", u64::from(policy.warm(shared, requests)));
                let snapshot = policy.cost_snapshot(shared, requests);
                for value in snapshot.layers.iter().flatten().chain(&snapshot.round).chain(&snapshot.draft) {
                    transcript.real("fit", *value);
                }
                let (timed, error) = policy.prediction_health(shared, requests);
                transcript.word("timed", timed);
                transcript.real("relative_error", error);
            }
        }
        for (slope, offset) in policy.calibration(1) { transcript.real("slope", slope); transcript.real("offset", offset); }
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


    /// Decisions and fits: every selection, prediction, cost coefficient,
    /// calibration and statistic of the replay. Re-pinned for the
    /// resource-priced cost rewrite (per-class a/c, residual r0 + s * rows with
    /// shrunk bucket offsets, per-bucket draft pass): a rewrite, so no
    /// attribution audit against the segmented fingerprint (98,217 records);
    /// `decision_rule_is_unchanged_given_identical_costs` checks the rule.
    /// A change here is a decision or fitted-state change.
    #[test]
    fn replay_reproduces_recorded_decisions_and_fits() {
        let mut transcript = Transcript::new();
        replay(&mut transcript);
        if let (Some(dump), Some(path)) = (&transcript.dump, std::env::var_os("CUTEAFD_POLICY_REPLAY_DUMP")) {
            std::fs::write(path, dump).unwrap();
        }
        assert_eq!((transcript.records, transcript.hash), (39_389, 4_805_264_383_091_156_835));
    }

    fn seed() -> PolicySeed {
        PolicySeed { round: [9000., 250., 150.], draft: [2500., 120.], samples: 4., cold_drafts: 3, warm_rounds: 24,
            warm_error: 0.08 }
    }

    /// One solo round of `rows` rows against the truth, with a per-row
    /// surcharge at one request that the linear fits cannot express (the
    /// head split's C1 attention and exchange), returning the wall µs.
    fn surcharged_round(policy: &mut DraftPolicy, token: &mut u64, rows: usize, requests: usize, surcharge: f64,
        c1_fixed: f64) -> f64 {
        truth_round(policy, &truth(), token, rows, requests, surcharge, c1_fixed)
    }

    /// Physics-shaped truth: per-layer fixed cost plus traffic, a fixed residual.
    fn physics() -> Truth {
        Truth { beta: [0., 0.], round: [9000., 0., 150.], ..truth() }
    }

    #[allow(clippy::too_many_arguments)]
    fn truth_round(policy: &mut DraftPolicy, truth: &Truth, token: &mut u64, rows: usize, requests: usize,
        surcharge: f64, c1_fixed: f64) -> f64 {
        let per = rows / requests;
        let token_rows: Vec<_> = (0..rows).map(|r| token_routes(*token + (r % per) as u64 + 1000 * (r / per) as u64, 24))
            .collect();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|l| token_rows.iter().map(|t| t[l]).collect()).collect();
        let mut layer_us = [None; 40];
        let mut total = truth.round[0] + truth.round[1] * rows as f64 + truth.round[2] * requests as f64
            + if requests == 1 { surcharge * (rows - 1) as f64 + c1_fixed } else { 0. };
        for layer in 1..40 {
            let class = policy.geometry.layers[layer].class.unwrap() as usize;
            let mb = groups(&routes[layer]) * policy.geometry.layers[layer].slice_bytes / 1e6;
            let t = truth.alpha[class] + truth.beta[class] * rows as f64 + truth.us_per_mb[class] * mb;
            layer_us[layer] = Some(t);
            total += t;
        }
        let observed: Vec<_> = (0..requests).map(|r| ObservedDraftRequest { id: 1 + r as u64, rows: per,
            accepted: per, confidence: None }).collect();
        policy.observe(DraftRoundObservation { shared: false, requests: &observed, routes: &flat(&routes),
            layer_us: &layer_us, total_us: total, predicted_us: None, draft_us: f64::NAN, drafted_requests: None, wide: false }).unwrap();
        *token += per as u64;
        total
    }

    #[test]
    fn a_seeded_policy_engages_on_prediction_health_not_sample_counts_alone() {
        let mut policy = DraftPolicy::seeded(placement(5), seed()).unwrap();
        assert_eq!(policy.cold_drafts(), 3);
        assert!(!policy.engaged(false, 1));
        assert_eq!(policy.select(false, &[DraftCandidate { id: 1, confidence: &[0.9; 5] }], 5), Ok(None));
        let mut token = 0;
        for round in 0..200 {
            surcharged_round(&mut policy, &mut token, 1 + round % 6, 1, 0., 0.);
            if round == 20 { assert!(!policy.engaged(false, 1), "{:?}", policy.prediction_health(false, 1)); }
        }
        let (rounds, error) = policy.prediction_health(false, 1);
        assert!(policy.engaged(false, 1), "{rounds} {error}");
        assert!(error < 0.03, "{error}");
        assert!(policy.select(false, &[DraftCandidate { id: 1, confidence: &[0.9; 5] }], 5).unwrap().is_some());
        assert!(DraftPolicy::seeded(placement(5), PolicySeed { samples: 0., ..seed() }).is_err());
        assert!(DraftPolicy::seeded(placement(5), PolicySeed { round: [f64::NAN, 0., 0.], ..seed() }).is_err());
    }

    #[test]
    fn mixed_c16_width_uses_verifier_bucket_and_prices_one_drafter() {
        let geometry = v41(|l| l < 5, |_| 4e6, vec![3, 7]);
        let mut policy = DraftPolicy::new(geometry, PolicySeed::weak(7), false).unwrap();
        let requests: Vec<_> = (1..=16).map(|id| (id, if id == 1 { 7 } else { 0 })).collect();
        let observed: Vec<_> = requests.iter().map(|&(id, room)| ObservedDraftRequest {
            id, rows: if room > 0 { 4 } else { 1 }, accepted: 1,
            confidence: if room > 0 { Some(&[0.9; 3]) } else { None } }).collect();
        for _ in 0..8 {
            policy.observe(DraftRoundObservation { shared: false, requests: &observed,
                routes: &vec![1; 40 * 19 * 6], layer_us: &[Some(2000.); 40], total_us: 90_000.,
                draft_us: 1200., drafted_requests: Some(1), wide: false, predicted_us: None }).unwrap();
        }
        assert_eq!(policy.cost.draft_samples[3], [8, 0]);
        assert_eq!(policy.cost.draft_samples[0], [0, 0]);
        assert_eq!(policy.prediction_health(false, 16).0, 8);
        assert_eq!(policy.prediction_health(false, 1), (0, 1.));
        // Only C9+ needs a wide warm-up pass; a borrowed C1 cell picks narrow.
        assert_eq!(policy.choose_width(false, &requests), 7);
        assert_eq!(policy.choose_width(false, &[(1, 7)]), 3);

        // With identical confidence, the wide pass pays for one participant
        // but not sixteen. Nonparticipants still contribute verifier anchors.
        policy.engaged[3] = true;
        policy.cost.draft_samples[3] = [8, 8];
        policy.since_width[3] = [0, 0];
        policy.confidence[3] = Confidence::new(7);
        for class in &mut policy.cost.class[3] { class.theta = [0.; 3]; }
        policy.cost.residual[3] = ([10_000., 0.], 0.);
        policy.cost.draft[3].theta = [100., 10., 0., 1000.];
        assert_eq!(policy.choose_width(false, &requests), 7);
        assert_eq!(policy.choose_width(false, &[(1, 7)]), 3);
    }

    #[test]
    fn c16_then_c1_replay_matches_a_c1_only_policy() {
        let mut cold = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let mut swept = cold.clone();
        let confidence = [0.85; 7];
        let mut token = 0;
        for _ in 0..100 {
            let requests: Vec<_> = (0..16).map(|id| ObservedDraftRequest {
                id: 100 + id, rows: 8, accepted: 8, confidence: Some(&confidence) }).collect();
            swept.observe(DraftRoundObservation { shared: false, requests: &requests,
                routes: &vec![1; 40 * 128 * 6], layer_us: &[Some(4000.); 40], total_us: 180_000.,
                draft_us: 6000., drafted_requests: Some(16), wide: true, predicted_us: None }).unwrap();
        }
        let wide_before = swept.cost_snapshot(false, 16);
        for policy in [&mut cold, &mut swept] {
            for id in 100..116 { policy.release(id); }
        }
        let mut priced = 0;
        for round in 0..120 {
            let candidates = [DraftCandidate { id: 1, confidence: &confidence }];
            assert_eq!(cold.calibration(1), swept.calibration(1));
            assert_eq!(cold.probabilities(1, candidates[0]), swept.probabilities(1, candidates[0]));
            assert_eq!(cold.choose_width(false, &[(1, 7)]), swept.choose_width(false, &[(1, 7)]));
            let left = cold.select(false, &candidates, 7).unwrap();
            let right = swept.select(false, &candidates, 7).unwrap();
            assert_eq!(left, right, "C1 round {round}");
            priced += usize::from(left.is_some());
            let accepted = 1 + round % 6;
            let requests = [ObservedDraftRequest { id: 1, rows: 8, accepted, confidence: Some(&confidence) }];
            let routes = flat(&(0..40).map(|l| (0..8).map(|r| token_routes(token + r, 24)[l]).collect())
                .collect::<Vec<_>>());
            let observation = DraftRoundObservation { shared: false, requests: &requests, routes: &routes,
                layer_us: &[Some(1000.); 40], total_us: 50_000., draft_us: 3000.,
                drafted_requests: Some(1), wide: true, predicted_us: None };
            cold.observe(observation).unwrap();
            swept.observe(observation).unwrap();
            token += accepted as u64;
            assert_eq!(cold.cost_snapshot(false, 1), swept.cost_snapshot(false, 1));
            assert_eq!(cold.forecast_megabytes(&[999], &[5]), swept.forecast_megabytes(&[999], &[5]));
        }
        assert!(priced > 0, "replay must reach priced C1 depth decisions");
        assert_eq!(swept.cost_snapshot(false, 16), wide_before, "C1 cannot overwrite wide evidence either");
    }

    #[test]
    fn c16_evidence_cannot_warm_or_reprice_a_cold_c1() {
        let mut swept = DraftPolicy::seeded(placement(5), seed()).unwrap();
        let cold = swept.clone();
        let mut token = 0;
        for round in 0..100 {
            truth_round(&mut swept, &physics(), &mut token, 16 * (1 + round % 4), 16, 0., 0.);
        }
        assert_eq!(swept.cost_snapshot(false, 1), cold.cost_snapshot(false, 1));
        assert_eq!(swept.prediction_health(false, 1), cold.prediction_health(false, 1));
        assert!(!swept.warm(false, 1));
        assert!(!swept.engaged(false, 1));
        let candidates = [DraftCandidate { id: 100, confidence: &[0.9; 5] }];
        assert_eq!(swept.select(false, &candidates, 5), cold.clone().select(false, &candidates, 5));
    }

    #[test]
    fn c1_costs_survive_a_c16_phase_while_request_routes_follow_the_request() {
        let mut policy = DraftPolicy::seeded(placement(5), seed()).unwrap();
        let mut token = 0;
        // C1 has its own fixed surcharge (attention split, exchange latency)
        // that the C16 phase does not.
        for round in 0..400 { truth_round(&mut policy, &physics(), &mut token, 1 + round % 7, 1, 0., 6000.); }
        let fresh = policy.clone();
        let before = policy.cost_snapshot(false, 1);
        let history = policy.history[&1].clone();
        for round in 0..1200 {
            truth_round(&mut policy, &physics(), &mut token, 16 * (1 + round % 4), 16, 0., 0.);
        }
        let after = policy.cost_snapshot(false, 1);
        // The C1 evidence is untouched by C16 rounds: it forgets only when C1 is observed.
        assert_eq!(before.round[3], after.round[3]);
        assert_eq!(before, after);
        assert_ne!(history, policy.history[&1]);
        assert_eq!(policy.prediction_health(false, 1), fresh.prediction_health(false, 1));
        // A fresh request has identical C1 evidence on both arms, including
        // its empty route history. Compare before any new C1 training.
        let candidate = [DraftCandidate { id: 999, confidence: &[0.9; 5] }];
        let old = fresh.clone().select(false, &candidate, 5).unwrap().unwrap();
        let swept = policy.select(false, &candidate, 5).unwrap().unwrap();
        assert_eq!(old, swept);
        // C1 total predictions stay accurate after the wide phase.
        for rows in [2usize, 6] {
            let predicted = policy.predict(false, &[1], &[rows - 1], 5).unwrap() - policy.draft_cost(false, 1, 5);
            let actual = truth_round(&mut policy, &physics(), &mut token, rows, 1, 0., 6000.);
            assert!((predicted - actual).abs() / actual < 0.06, "{rows}: {predicted} vs {actual}");
        }
    }

    #[test]
    fn the_first_wide_batch_cannot_poison_the_fit() {
        let truth = physics();
        let mut clean = DraftPolicy::seeded(placement(5), seed()).unwrap();
        let mut poisoned = DraftPolicy::seeded(placement(5), seed()).unwrap();
        let (mut a, mut b) = (0, 0);
        // Rounds one and two after launch run six times slow (first-use
        // workspaces and graph capture) in every layer and the residual.
        for rows in [4, 6] { stalled_round(&mut poisoned, &truth, &mut b, rows, 6.); }
        for rows in [4, 6] { run_round(&mut clean, &truth, &mut a, rows, rows, None); }
        for round in 0..60 {
            let rows = 1 + round % 8;
            run_round(&mut clean, &truth, &mut a, rows, rows, None);
            run_round(&mut poisoned, &truth, &mut b, rows, rows, None);
        }
        for rows in [2usize, 8] {
            let p = poisoned.clone().predict(false, &[1], &[rows - 1], 5);
            let c = clean.clone().predict(false, &[1], &[rows - 1], 5);
            let (Some(p), Some(c)) = (p, c) else { panic!("both engaged") };
            assert!((p - c).abs() / c < 0.02, "{rows}: poisoned {p} vs clean {c}");
        }
        // A persistent change is learned, not rejected forever.
        for round in 0..80 { stalled_round(&mut poisoned, &truth, &mut b, 1 + round % 8, 3.); }
        let p = poisoned.clone().predict(false, &[1], &[3], 5).unwrap();
        let c = clean.clone().predict(false, &[1], &[3], 5).unwrap();
        assert!(p > 2. * c, "persistent 3x: {p} vs {c}");
    }

    #[test]
    fn rows_are_priced_by_their_new_expert_traffic() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let (truth, mut token) = (truth(), 0);
        for round in 0..400 { let rows = 1 + round % 8; run_round(&mut policy, &truth, &mut token, rows, rows, None); }
        // Request 1 repeats one token: extra rows read no new experts and cost
        // only their share of nothing; request 2 reads fresh experts per row.
        let repeat: Vec<Vec<[u32; 6]>> = (0..40).map(|l| vec![token_routes(7, 24)[l]]).collect();
        let requests = [ObservedDraftRequest { id: 1, rows: 1, accepted: 1, confidence: None }];
        for _ in 0..8 {
            policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&repeat),
                layer_us: &[None; 40], total_us: f64::NAN, predicted_us: None, draft_us: f64::NAN,
                drafted_requests: None, wide: false }).unwrap();
        }
        let fresh_rows: Vec<_> = (0..8).map(|r| token_routes(10_000 + 97 * r, 384)).collect();
        for t in &fresh_rows {
            let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|l| vec![t[l]]).collect();
            let requests = [ObservedDraftRequest { id: 2, rows: 1, accepted: 1, confidence: None }];
            policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
                layer_us: &[None; 40], total_us: f64::NAN, predicted_us: None, draft_us: f64::NAN,
                drafted_requests: None, wide: false }).unwrap();
        }
        let price = |policy: &mut DraftPolicy, id| policy.predict(false, &[id], &[4], 5).unwrap()
            - policy.predict(false, &[id], &[0], 5).unwrap();
        let repeated = price(&mut policy, 1);
        let novel = price(&mut policy, 2);
        // A repeated row reads no new experts but still costs its per-row
        // layer work; a novel row also pays its traffic.
        assert!(repeated < 0.3 * novel, "repeated {repeated} vs novel {novel}");
        // A Spark MB costs more than an RTX MB: the classes keep their own prices.
        let fit = policy.cost_snapshot(false, 1);
        assert!(fit.layers[1][1] > 4. * fit.layers[0][1], "{:?}", fit.layers);
    }

    #[test]
    fn collinear_classes_hold_their_physical_split() {
        // Local and remote traffic move together; the data cannot tell which
        // class is expensive per MB, so the classes stay near their priors.
        let mut policy = DraftPolicy::new(placement(8), PolicySeed::weak(7), false).unwrap();
        let (truth, mut token) = (truth(), 0);
        for round in 0..400 { let rows = 1 + round % 8; run_round(&mut policy, &truth, &mut token, rows, rows, None); }
        let fit = policy.cost_snapshot(false, 1);
        let [local, remote] = [fit.layers[0][1], fit.layers[1][1]];
        assert!(local < 2.5 && remote > 4., "local {local} remote {remote} µs/MB");
    }

    #[test]
    fn sparse_buckets_keep_their_seed_and_only_learn_their_own_residual() {
        let mut policy = DraftPolicy::seeded(placement(5), seed()).unwrap();
        let (truth, mut token) = (physics(), 0);
        for round in 0..400 { truth_round(&mut policy, &truth, &mut token, 8 * (1 + round % 6), 8, 0., 0.); }
        // C1 and C9+ have no evidence: retain their seed, not C8's fit.
        // Residual at the bucket's own shape: two rows at C1, 24 at C8.
        let residual = |p: &DraftPolicy, requests: usize| { let f = p.cost_snapshot(false, requests);
            f.round[0] + if requests == 1 { 2. } else { 24. } * f.round[1] + f.round[2] };
        let c8 = residual(&policy, 8);
        for requests in [1, 16] {
            let fit = policy.cost_snapshot(false, requests);
            assert_eq!(fit.round, [seed().round[0], seed().round[1], 0., 0., INITIAL_RELATIVE_SCALE], "{requests}");
            assert!(!policy.engaged(false, requests));
        }
        let c8_before = policy.cost_snapshot(false, 8);
        assert!((residual(&policy, 1) - (truth.round[0] + truth.round[2])).abs() < 0.1 * c8);
        // C1 rounds carry 8 ms more: two rounds move C1 part of the way
        // (shrinkage), sixty move it all the way, and C8 stays unchanged.
        let c1 = residual(&policy, 1);
        for _ in 0..2 { truth_round(&mut policy, &truth, &mut token, 2, 1, 0., 8000.); }
        let early = residual(&policy, 1) - c1;
        for _ in 0..60 { truth_round(&mut policy, &truth, &mut token, 2, 1, 0., 8000.); }
        let c1_truth = truth.round[0] + truth.round[2] + 8000.;
        assert!(early > 1000. && early < 5000., "early C1 shift {early}");
        assert!((residual(&policy, 1) - c1_truth).abs() < 0.15 * c1_truth, "C1 residual {}", residual(&policy, 1));
        assert_eq!(policy.cost_snapshot(false, 8), c8_before);
    }

    /// The decision rule is the old one: given the same prices, forward
    /// growth over calibrated survival chooses the same lengths as an
    /// independent reference implementation of #74's rule.
    #[test]
    fn decision_rule_is_unchanged_given_identical_costs() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let (truth, mut token) = (physics(), 0);
        for round in 0..300 { run_round(&mut policy, &truth, &mut token, 1 + round % 8, 1 + round % 8, None); }
        // Route histories for requests 2-4 (request 1 has the warm-up's).
        for id in 2..=4 { for _ in 0..8 { truth_round_id(&mut policy, &truth, &mut token, id); } }
        policy.engaged[1] = true;
        let mut mix = Mix(0xdec1de);
        for case in 0..200 {
            let requests = 1 + mix.below(4) as usize;
            let confidence: Vec<Vec<f64>> = (0..requests).map(|_| (0..5).map(|_| mix.unit()).collect()).collect();
            let candidates: Vec<_> = (0..requests).map(|i| DraftCandidate { id: 1 + i as u64, confidence: &confidence[i] })
                .collect();
            let candidates = &candidates[..];
            let selected = policy.select(false, candidates, 5).unwrap().unwrap();
            // Reference: every shape priced by `evaluate`, forward growth with
            // the best visited ratio and ties to the longer shape.
            let mut lengths = vec![0usize; candidates.len()];
            let (e, t) = policy.evaluate(false, candidates, &lengths, 5).unwrap();
            let mut best = (e / t, lengths.clone());
            loop {
                let mut choice: Option<(usize, f64)> = None;
                for i in 0..candidates.len() {
                    if lengths[i] >= candidates[i].confidence.len() { continue; }
                    let mut next = lengths.clone();
                    next[i] += 1;
                    let (e, t) = policy.evaluate(false, candidates, &next, 5).unwrap();
                    if choice.is_none_or(|(_, r)| e / t > r) { choice = Some((i, e / t)); }
                }
                let Some((i, ratio)) = choice else { break };
                lengths[i] += 1;
                if ratio >= best.0 { best = (ratio, lengths.clone()); }
            }
            assert_eq!(selected.lengths, best.1, "case {case}");
        }
    }

    /// Layer times on one round in eight: the class fits learn from those,
    /// the residual and prediction health from every round.
    #[test]
    fn sampled_layer_times_still_fit_every_round() {
        let truth = physics();
        let mut sampled = DraftPolicy::seeded(placement(5), seed()).unwrap();
        let mut every = DraftPolicy::seeded(placement(5), seed()).unwrap();
        let (mut a, mut b) = (0, 0);
        for round in 0..400 {
            let rows = 1 + round % 8;
            run_round(&mut every, &truth, &mut a, rows, rows, None);
            cadence_round(&mut sampled, &truth, &mut b, rows, round % 8 == 0);
        }
        assert_eq!(sampled.prediction_health(false, 1).0, every.prediction_health(false, 1).0);
        assert!(sampled.engaged(false, 1));
        for rows in [2usize, 8] {
            let s = sampled.clone().predict(false, &[1], &[rows - 1], 5).unwrap();
            let e = every.clone().predict(false, &[1], &[rows - 1], 5).unwrap();
            assert!((s - e).abs() / e < 0.03, "{rows}: sampled {s} vs every {e}");
        }
    }

    fn cadence_round(policy: &mut DraftPolicy, truth: &Truth, token: &mut u64, rows: usize, timed: bool) {
        let token_rows: Vec<_> = (0..rows).map(|r| token_routes(*token + r as u64, 24)).collect();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|l| token_rows.iter().map(|t| t[l]).collect()).collect();
        let mut layer_us = [None; 40];
        let mut total = truth.round[0] + truth.round[1] * rows as f64 + truth.round[2];
        for layer in 1..40 {
            let class = policy.geometry.layers[layer].class.unwrap() as usize;
            let mb = groups(&routes[layer]) * policy.geometry.layers[layer].slice_bytes / 1e6;
            let t = truth.alpha[class] + truth.beta[class] * rows as f64 + truth.us_per_mb[class] * mb;
            if timed { layer_us[layer] = Some(t); }
            total += t;
        }
        let requests = [ObservedDraftRequest { id: 1, rows, accepted: rows, confidence: None }];
        policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
            layer_us: &layer_us, total_us: total, predicted_us: None, draft_us: f64::NAN, drafted_requests: None, wide: false }).unwrap();
        *token += rows as u64;
    }

    #[test]
    fn survival_lengths_draft_while_survival_is_at_least_half() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        policy.set_survival_lengths(true);
        let (truth, mut token) = (truth(), 0);
        for round in 0..300 { run_round(&mut policy, &truth, &mut token, 1 + round % 6, 1 + round % 6, None); }
        policy.engaged[1] = true;
        let candidates = [DraftCandidate { id: 1, confidence: &[0.95, 0.9, 0.7, 0.9, 0.9] },
            DraftCandidate { id: 2, confidence: &[0.4, 0.99, 0.99, 0.99, 0.99] }];
        let survival = |p: &DraftPolicy, c: &DraftCandidate<'_>| {
            let mut product = 1.;
            p.probabilities(candidates.len(), *c).iter().take_while(|&&q| { product *= q; product >= 0.5 }).count()
        };
        let expected: Vec<_> = candidates.iter().map(|c| survival(&policy, c)).collect();
        let selection = policy.select(false, &candidates, 5).unwrap().unwrap();
        assert_eq!(selection.lengths, expected);
        let (tokens, us) = policy.evaluate(false, &candidates, &selection.lengths, 5).unwrap();
        assert!((tokens - selection.expected_tokens).abs() < 1e-9 && (us - selection.predicted_us).abs() < 1e-6 * us);
        // A row budget drops the least likely rows first.
        let tight = policy.select_within(false, &candidates, 5, 3).unwrap().unwrap();
        assert_eq!(tight.lengths.iter().sum::<usize>(), 1);
        assert!(policy.restarted(false).survival_lengths());
    }

    #[test]
    fn cold_lengths_follow_calibrated_survival_not_a_fixed_count() {
        let seed = PolicySeed { cold_drafts: 5, ..PolicySeed::weak(7) };
        let policy = DraftPolicy::new(placement(7), seed, false).unwrap();
        assert!(!policy.engaged(false, 1));
        let length = |confidence: &[f64]| policy.cold_length(1, DraftCandidate { id: 1, confidence });
        // Seeded calibration is the identity: survival 0.9, 0.81, 0.73, 0.66, 0.59 ...
        assert_eq!(length(&[0.9; 7]), 5, "capped at the seed's cold drafts");
        // 0.95, 0.76, 0.27: only two rows are more likely verified than wasted.
        assert_eq!(length(&[0.95, 0.8, 0.35, 0.99, 0.99, 0.99, 0.99]), 2);
        // A hopeless block still verifies one draft so calibration keeps learning.
        assert_eq!(length(&[0.05; 7]), 1);
        assert_eq!(length(&[0.9, 0.9]), 2, "within the request's room");
        assert_eq!(length(&[]), 0, "an undrafted request verifies only its anchor");
        // Cold lengths read the bucket's own calibration: wide rounds whose
        // confident drafts all fail recalibrate C16 only, never C1.
        let mut swept = policy.clone();
        let confidence = [0.9; 7];
        let p = [0.9, 0.9, 0.9, 0.6, 0.9, 0.9, 0.9];
        for round in 0..64u64 {
            let requests: Vec<_> = (0..16).map(|id| ObservedDraftRequest { id: 100 + id, rows: 3, accepted: 1,
                confidence: Some(&confidence) }).collect();
            let routes = flat(&(0..40).map(|l| (0..48).map(|r| token_routes(round * 48 + r, 24)[l]).collect())
                .collect::<Vec<_>>());
            swept.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &routes,
                layer_us: &[Some(1000.); 40], total_us: 90_000., draft_us: 6000., drafted_requests: Some(16),
                wide: true, predicted_us: None }).unwrap();
        }
        assert!(swept.cold_length(16, DraftCandidate { id: 2, confidence: &p }) < 3, "C16 calibration learned");
        assert_eq!(swept.calibration(1), policy.calibration(1));
        assert_eq!(swept.cold_length(1, DraftCandidate { id: 2, confidence: &p }),
            policy.cold_length(1, DraftCandidate { id: 2, confidence: &p }));
    }

    #[test]
    fn concurrency_boundaries_and_bucket_exploration_are_independent() {
        assert_eq!([0, 1, 2, 4, 5, 8, 9, 16, 128].map(draft_concurrency_bucket), [0, 0, 1, 1, 2, 2, 3, 3, 3]);
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        policy.engaged[0] = true;
        policy.engaged[3] = true;
        let actions = [DraftAction { width: 0, draft_us: 0. }, DraftAction { width: 5, draft_us: 1e12 }];
        for _ in 0..WIDTH_EXPLORE_ROUNDS - 1 {
            assert_eq!(policy.choose_action(false, &[(1, 5)], &actions, 64).unwrap(), Some(0));
        }
        let wide: Vec<_> = (0..16).map(|id| (100 + id, 5)).collect();
        assert_eq!(policy.choose_action(false, &wide, &actions, 64).unwrap(), Some(0));
        assert_eq!(policy.action_age[&(3, 5)], 1);
        assert_eq!(policy.choose_action(false, &[(1, 5)], &actions, 64).unwrap(), Some(1));
        assert_eq!(policy.action_age[&(3, 5)], 1);
        assert_eq!(policy.prediction_health(true, 1), (0, 1.));
        assert_eq!(policy.draft_cost(false, 1, 0), 0.);
        assert_eq!(policy.draft_cost_drafted(false, 16, 0, 5), 0.);
    }

    #[test]
    fn calibration_is_concurrency_local_and_native_and_tail_pools_are_separate() {
        let mut geometry = placement(5);
        geometry.max_positions = 8;
        let mut policy = DraftPolicy::new(geometry, PolicySeed::weak(8), false).unwrap();
        let initial = policy.calibration(1);
        let native_pool = policy.confidence[3].pooled[0].theta;
        let confidence = [0.9; 8];
        let cold_request = policy.probabilities(1, DraftCandidate { id: 15, confidence: &confidence });
        // Only tail position seven is reached with a miss; native positions accept.
        let requests: Vec<_> = (0..16).map(|id| ObservedDraftRequest {
            id, rows: 9, accepted: 8, confidence: Some(&confidence) }).collect();
        policy.observe(DraftRoundObservation { shared: false, requests: &requests,
            routes: &vec![1; 40 * 144 * 6], layer_us: &[None; 40], total_us: f64::NAN,
            draft_us: f64::NAN, drafted_requests: None, wide: false, predicted_us: None }).unwrap();
        assert_eq!(initial, policy.calibration(1));
        assert_eq!(cold_request, policy.probabilities(1, DraftCandidate { id: 15, confidence: &confidence }));
        assert!(!policy.confidence[0].recent.contains_key(&15));
        assert!(!policy.confidence[0].outcomes.contains_key(&15));
        assert_ne!(initial, policy.calibration(16));
        assert_ne!(native_pool, policy.confidence[3].pooled[0].theta);
        assert_ne!(policy.confidence[3].pooled[0].theta, policy.confidence[3].pooled[1].theta);
        let native_after = policy.confidence[3].pooled[0].theta;
        let probability = policy.probabilities(16, DraftCandidate { id: 99, confidence: &[0.9] })[0];
        // Training only the tail cannot move a cold native position's prior.
        for _ in 0..16 { policy.confidence[3].pooled[1].observe(0.9, 0.); }
        assert_eq!(native_after, policy.confidence[3].pooled[0].theta);
        assert_eq!(probability, policy.probabilities(16, DraftCandidate { id: 99, confidence: &[0.9] })[0]);
        assert!(policy.history.contains_key(&15));
        assert!(policy.confidence[3].recent.contains_key(&15));
        policy.release(15);
        assert!(!policy.history.contains_key(&15) && !policy.confidence[3].recent.contains_key(&15));
    }

    #[test]
    fn observation_requires_the_selection_request_extent() {
        let mut policy = DraftPolicy::seeded(placement(5), seed()).unwrap();
        let mut token = 0;
        for round in 0..400 { surcharged_round(&mut policy, &mut token, 1 + round % 7, 1, 0., 0.); }
        let selected = policy.select(false, &[DraftCandidate { id: 1, confidence: &[0.9; 5] }], 5).unwrap().unwrap();
        let requests = [ObservedDraftRequest { id: 1, rows: 1, accepted: 1, confidence: None },
            ObservedDraftRequest { id: 2, rows: 1, accepted: 1, confidence: None }];
        let before = format!("{policy:?}");
        assert_eq!(policy.observe(DraftRoundObservation { shared: false, requests: &requests,
            routes: &[1; 40 * 2 * 6], layer_us: &[None; 40], total_us: 10000., draft_us: f64::NAN,
            drafted_requests: None, wide: false, predicted_us: Some(selected.predicted_us) }),
            Err(DraftPolicyError::RequestExtent));
        assert_eq!(before, format!("{policy:?}"));
    }

    #[test]
    fn a_row_budget_goes_to_the_requests_whose_drafts_pay() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        let (truth, mut token) = (truth(), 0);
        for round in 0..300 { run_round(&mut policy, &truth, &mut token, 1 + round % 8, 1 + round % 8, None); }
        let sure = [0.99; 5];
        // A row costs about 0.3% of a C3 round here; 0.01 cannot pay for it.
        let hopeless = [0.01; 5];
        let candidates = [DraftCandidate { id: 1, confidence: &sure }, DraftCandidate { id: 2, confidence: &hopeless },
            DraftCandidate { id: 3, confidence: &sure }];
        for round in 0..400 { surcharged_round(&mut policy, &mut token, 3 * (1 + round % 4), 3, 0., 0.); }
        let free = policy.select(false, &candidates, 5).unwrap().unwrap();
        assert_eq!(free.lengths, [5, 0, 5]);
        // Nine rows: three anchors and six drafts, all to the confident requests.
        let bounded = policy.select_within(false, &candidates, 5, 9).unwrap().unwrap();
        assert_eq!(bounded.lengths.iter().sum::<usize>(), 6, "{bounded:?}");
        assert_eq!(bounded.lengths[1], 0);
        // Fewer rows than anchors: every request verifies its anchor alone.
        assert_eq!(policy.select_within(false, &candidates, 5, 2).unwrap().unwrap().lengths, [0, 0, 0]);
        // An unbounded budget is select itself.
        assert_eq!(policy.select_within(false, &candidates, 5, usize::MAX).unwrap().unwrap(), free);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let mut policy = DraftPolicy::new(placement(5), PolicySeed::weak(7), false).unwrap();
        assert!(policy.select(false, &[], 5).is_err());
        assert!(policy.select(false, &[DraftCandidate { id: 1, confidence: &[f64::NAN] }], 5).is_err());
        assert!(policy.select(false, &[DraftCandidate { id: 1, confidence: &[1.1] }], 5).is_err());
        assert!(policy.select(false, &[DraftCandidate { id: 1, confidence: &[0.5; 8] }], 5).is_err());
        let requests = [ObservedDraftRequest { id: 1, rows: 2, accepted: 1, confidence: None }];
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]]).collect();
        assert!(policy.observe(DraftRoundObservation { shared: false, requests: &requests, routes: &flat(&routes),
            layer_us: &[None; 40], total_us: 1., predicted_us: None, draft_us: f64::NAN, drafted_requests: None, wide: false }).is_err());
        assert!(policy.observe(DraftRoundObservation { shared: false, requests: &requests[..0], routes: &[1],
            layer_us: &[], total_us: 1., predicted_us: None, draft_us: f64::NAN, drafted_requests: None, wide: false }).is_err());
        assert!(DraftPolicy::new(v41(|_| false, |_| 0., vec![5, 7]), PolicySeed::weak(7), false).is_err());
    }
}
