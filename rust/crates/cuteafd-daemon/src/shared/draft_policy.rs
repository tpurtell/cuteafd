//! The speculative draft-length policy every family shares (GLM and GLM
//! Flash DFlash2, MiMo DFlash and MTP, Qwen 3.8 MTP).
//!
//! Acceptance: each sequence keeps its last 16 (proposed, accepted) outcomes,
//! which give conditional acceptance rates per draft position (a position is
//! observed only when every earlier draft was accepted) under a prior: 3
//! successes in 4 trials, plus (`DraftHistory::pooled`) the other
//! sequences' outcomes at that position, up to 8 trials. Families may
//! refine the rates (GLM's frozen selector calibration) before planning.
//!
//! Cost: a cycle is one verify step over every sequence's next token and
//! drafts, plus the draft work and host time. The verify step's cost is
//! driven by the weights and experts it reads, i.e. by the union of routes
//! over all its rows: sub-additive when sequences share experts (identical
//! prompts, repetitive text), near-additive otherwise, so no fixed table of
//! one sequence's step times prices several sequences correctly.
//! [`CycleCost`] keeps a measured single-sequence table as the shape (its
//! steps included: tile changes make some row counts much dearer) and fits
//! `ms = a + b * table_increment(rows) + c * (sequences - 1)` online
//! (exponentially forgotten least squares, regularized toward the table), so
//! it learns both the intercept and how steeply rows cost in the running mix.
//! Independent fits for 1, 2-4, 5-8 and 9+ sequences prevent a concurrency
//! sweep from changing a short single-sequence request's cost estimates.
//! Draft work fits `d * drafting + e * chained steps` the same way.
//!
//! Allocation: [`allocate`] admits draft positions across sequences one at a
//! time, each time the one with the most expected emitted tokens per
//! marginal millisecond, and keeps the plan with the best expected emitted
//! tokens per cycle millisecond. Chained drafters (MTP) pay one draft step
//! per position of the deepest sequence; the allocator tries every depth cap.
use std::collections::VecDeque;

const HISTORY: usize = 16;
const COLD_START_CYCLES: usize = 4;
const PRIOR_SUCCESSES: usize = 3;
const PRIOR_TRIALS: usize = 4;

/// A sequence's recent draft outcomes.
#[derive(Debug, Clone, Default)]
pub(crate) struct DraftHistory {
    outcomes: VecDeque<(usize, usize)>,
}

impl DraftHistory {
    /// Fewer than four outcomes observed.
    pub fn cold(&self) -> bool {
        self.outcomes.len() < COLD_START_CYCLES
    }

    pub fn observe(&mut self, proposed: usize, accepted: usize) {
        if proposed == 0 {
            return;
        }
        self.outcomes.push_back((proposed, accepted.min(proposed)));
        while self.outcomes.len() > HISTORY {
            self.outcomes.pop_front();
        }
    }

    /// (successes, trials) of `position` (censored after a miss).
    fn counts(&self, position: usize) -> (usize, usize) {
        let (mut successes, mut trials) = (0, 0);
        for &(proposed, accepted) in &self.outcomes {
            if proposed >= position && accepted + 1 >= position {
                trials += 1;
                successes += usize::from(accepted >= position);
            }
        }
        (successes, trials)
    }

    /// Conditional acceptance of positions 1..=max with a prior of 3
    /// successes in 4 trials per position.
    pub fn conditional(&self, max: usize) -> Vec<f64> {
        (1..=max).map(|position| {
            let (successes, trials) = self.counts(position);
            (PRIOR_SUCCESSES + successes) as f64 / (PRIOR_TRIALS + trials) as f64
        }).collect()
    }

    /// Conditional acceptance of positions 1..=max under a prior pooled
    /// from `others` (the other sequences in the step): their outcomes at
    /// each position on top of 3 in 4, weighing at most `POOL_TRIALS`
    /// trials. Alone, this is [`Self::conditional`].
    pub fn pooled<'h>(&self, max: usize, others: impl Iterator<Item = &'h DraftHistory> + Clone) -> Vec<f64> {
        (1..=max).map(|position| {
            let (pooled, trials) = others.clone().map(|h| h.counts(position))
                .fold((PRIOR_SUCCESSES, PRIOR_TRIALS), |(s, t), (hs, ht)| (s + hs, t + ht));
            let weight = (trials as f64).min(POOL_TRIALS);
            let mean = pooled as f64 / trials as f64;
            let (successes, own) = self.counts(position);
            (weight * mean + successes as f64) / (weight + own as f64)
        }).collect()
    }
}

/// Most prior weight, in trials, the other sequences' outcomes get.
///
/// Served traces (Qwen 3.8 MTP, four sequences) show one sequence's
/// 16-outcome history is a noisy estimate: positions it rated 0.4-0.5 were
/// accepted 60-70% of the time and 0.8-0.9 ones 75-80%, so plans cut drafts
/// that would have been kept. Borrowing the other sequences' outcomes shrinks
/// both tails; a lone sequence keeps its own history (pooling a sequence
/// with its own past made single-sequence plans slower to follow content).
/// Half the history's weight keeps a sequence unlike the others its own.
const POOL_TRIALS: f64 = 8.0;

/// Observation weight decay per verified draft position (about 100
/// outcomes of memory).
const CALIBRATION_FORGET: f64 = 0.99;
/// Range of the calibration's slope.
const CALIBRATION_SLOPE: (f64, f64) = (0.25, 1.0);

/// Online linear recalibration of conditional acceptance rates:
/// `rate' = a + b * rate`, fit by forgotten least squares on the outcomes
/// of the positions actually verified, regularized toward the identity.
///
/// Served traces (Qwen 3.8 MTP, 16-outcome histories) show the history
/// rates are over-dispersed: positions rated 0.4-0.5 were accepted 60-70% of
/// the time and 0.8-0.9 ones 75-80%, so plans cut drafts that would have
/// been kept. Replayed causally on those traces, the fit (`b` mostly 0.3-0.6)
/// cut the log loss of the verified positions by 1.5% (one sequence, four
/// distinct) to 10% (four identical sequences).
#[derive(Debug, Clone)]
pub(crate) struct Calibration {
    fit: Ridge<2>,
}

impl Default for Calibration {
    fn default() -> Self {
        // The slope's prior is weak: history rates spread over a few tenths.
        let mut fit = Ridge::new([0.0, 1.0], [1.0, 0.1]);
        fit.forget = CALIBRATION_FORGET;
        Self { fit }
    }
}

impl Calibration {
    /// The calibrated rate of a history rate.
    pub fn apply(&self, rate: f64) -> f64 {
        self.fit.predict(&[1.0, rate]).clamp(0.02, 0.98)
    }

    /// Fitted (a, b).
    pub fn fitted(&self) -> [f64; 2] {
        self.fit.theta
    }

    /// Folds one verified position in: the rate it was planned with and
    /// whether it was accepted.
    pub fn observe(&mut self, rate: f64, accepted: bool) {
        if !rate.is_finite() {
            return;
        }
        let fit = &mut self.fit;
        fit.observe(&[1.0, rate], if accepted { 1.0 } else { 0.0 });
        // Rates concentrated in a narrow band leave the slope loose (served
        // fits wandered below zero, inverting the plan's order): hold it in
        // CALIBRATION_SLOPE and refit the intercept alone.
        let slope = fit.theta[1].clamp(CALIBRATION_SLOPE.0, CALIBRATION_SLOPE.1);
        if slope != fit.theta[1] {
            let intercept = (fit.moment[0] - slope * fit.gram[0][1] + fit.penalty[0] * fit.prior[0])
                / (fit.gram[0][0] + fit.penalty[0]);
            fit.theta = [intercept, slope];
        }
    }

    /// Folds a sequence's verify outcome in: `rates` it was planned with
    /// (conditional, per draft position), `proposed` drafts, `accepted`.
    pub fn observe_outcome(&mut self, rates: &[f64], proposed: usize, accepted: usize) {
        for (index, &rate) in rates.iter().enumerate().take(proposed.min(accepted + 1)) {
            self.observe(rate, accepted > index);
        }
    }
}

/// The rows of a verify step: `rows` in all, `distinct` of them not
/// duplicates of an identical sequence's rows, over `sequences` sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    pub rows: usize,
    pub distinct: usize,
    pub sequences: usize,
}

impl Shape {
    /// Rows of distinct sequences.
    pub fn plain(rows: usize, sequences: usize) -> Self {
        Self { rows, distinct: rows, sequences }
    }
}

/// Observation weight decay per step (about 50 steps of memory).
const FORGET: f64 = 0.98;
/// Weight of a prior, in observations at its reference point.
const PRIOR_WEIGHT: f64 = 0.25;
/// Rows, sequences and chained draft steps that scale the priors.
const REFERENCE_ROWS: usize = 8;
const REFERENCE_SEQUENCES: f64 = 3.0;
const REFERENCE_CHAIN: f64 = 3.0;

/// Exponentially forgotten least squares `y ~ theta . phi`, regularized
/// toward a fixed prior (which never decays, so directions the recent steps
/// do not excite stay near it instead of winding up).
#[derive(Debug, Clone)]
struct Ridge<const N: usize> {
    forget: f64,
    prior: [f64; N],
    penalty: [f64; N],
    gram: [[f64; N]; N],
    moment: [f64; N],
    theta: [f64; N],
}

impl<const N: usize> Ridge<N> {
    fn new(prior: [f64; N], penalty: [f64; N]) -> Self {
        Self { forget: FORGET, prior, penalty, gram: [[0.0; N]; N], moment: [0.0; N], theta: prior }
    }

    fn predict(&self, phi: &[f64; N]) -> f64 {
        phi.iter().zip(&self.theta).map(|(p, t)| p * t).sum()
    }

    fn observe(&mut self, phi: &[f64; N], y: f64) {
        for i in 0..N {
            self.moment[i] = self.forget * self.moment[i] + phi[i] * y;
            for j in 0..N {
                self.gram[i][j] = self.forget * self.gram[i][j] + phi[i] * phi[j];
            }
        }
        let mut matrix = self.gram;
        let mut rhs = self.moment;
        for i in 0..N {
            matrix[i][i] += self.penalty[i];
            rhs[i] += self.penalty[i] * self.prior[i];
        }
        if let Some(theta) = solve(matrix, rhs) {
            self.theta = theta;
        }
    }
}

/// Cycle milliseconds: the verify step, the draft work and host time.
///
/// Verify: `a + b * x + c * (sequences - 1)`, where `x` is the table's
/// increment from one row to the step's distinct rows plus a duplicate-row
/// price per row repeating an identical sequence's. The prior is the table
/// itself (`a` = one row, `b` = 1, `c` = 0).
/// Draft: `d * drafting + e * chained steps` (a drafting cycle's fixed cost,
/// e.g. a DFlash block or an MTP cycle's first step over the pending rows,
/// and each chained MTP step).
#[derive(Debug, Clone)]
pub(crate) struct CycleCost {
    /// Single-sequence verify ms by rows (index = rows, 0 priced as 1).
    table: Vec<f64>,
    /// Table-scale ms of a row whose routes an identical row already reads.
    duplicate_row_ms: f64,
    fits: [CostFit; 4],
}

#[derive(Debug, Clone)]
struct CostFit {
    verify: Ridge<3>,
    draft: Ridge<2>,
    host_ms: f64,
}

fn concurrency_bucket(sequences: usize) -> usize {
    match sequences {
        0..=1 => 0,
        2..=4 => 1,
        5..=8 => 2,
        _ => 3,
    }
}

impl CycleCost {
    /// `points` are (rows, ms) of one sequence's verify step with increasing
    /// rows, the first at one row; linear between points and past the last.
    pub fn new(points: &[(usize, f64)], max_rows: usize) -> Self {
        assert!(points.len() >= 2 && points[0].0 == 1, "a verify table starts at one row and has two points");
        let table: Vec<f64> = (0..=max_rows.max(REFERENCE_ROWS)).map(|rows| {
            let rows = rows.max(1);
            let upper = points.iter().position(|&(r, _)| r >= rows).unwrap_or(points.len() - 1).max(1);
            let ((r0, m0), (r1, m1)) = (points[upper - 1], points[upper]);
            m0 + (m1 - m0) * (rows as f64 - r0 as f64) / (r1 as f64 - r0 as f64)
        }).collect();
        let reach = (table[REFERENCE_ROWS] - table[1]).max(1e-3);
        let verify = Ridge::new([table[1], 1.0, 0.0], [PRIOR_WEIGHT, PRIOR_WEIGHT * reach * reach,
            PRIOR_WEIGHT * REFERENCE_SEQUENCES * REFERENCE_SEQUENCES]);
        let draft = Ridge::new([0.0, 0.0], [PRIOR_WEIGHT, PRIOR_WEIGHT * REFERENCE_CHAIN * REFERENCE_CHAIN]);
        Self { table, duplicate_row_ms: 0.0, fits: std::array::from_fn(|_| CostFit {
            verify: verify.clone(), draft: draft.clone(), host_ms: 0.0,
        }) }
    }

    /// Prices duplicate rows (identical sequences' rows beyond the first).
    pub fn duplicate_rows(mut self, ms: f64) -> Self {
        self.duplicate_row_ms = ms;
        self
    }

    /// Starting draft ms per drafting cycle, per chained draft step, and host ms per cycle.
    pub fn drafts(mut self, draft_ms: f64, chain_ms: f64, host_ms: f64) -> Self {
        for fit in &mut self.fits {
            fit.draft = Ridge::new([draft_ms, chain_ms], fit.draft.penalty);
            fit.host_ms = host_ms;
        }
        self
    }

    fn features(&self, shape: Shape) -> [f64; 3] {
        let rows = shape.rows.max(1);
        let distinct = shape.distinct.clamp(1, rows);
        let last = self.table.len() - 1;
        let beyond = distinct.saturating_sub(last) as f64 * (self.table[last] - self.table[last - 1]);
        let x = self.table[distinct.min(last)] + beyond - self.table[1]
            + self.duplicate_row_ms * (rows - distinct) as f64;
        [1.0, x, shape.sequences.saturating_sub(1) as f64]
    }

    /// Verify-step ms of `shape`.
    pub fn verify_ms(&self, shape: Shape) -> f64 {
        self.fits[concurrency_bucket(shape.sequences)].verify.predict(&self.features(shape)).max(0.25 * self.table[1])
    }

    /// Draft ms of a cycle: the drafting cost when `drafting`, and `chain` chained steps.
    pub fn draft_ms(&self, sequences: usize, drafting: bool, chain: usize) -> f64 {
        if !drafting && chain == 0 {
            return 0.0;
        }
        self.fits[concurrency_bucket(sequences)].draft.predict(&[1.0, chain as f64]).max(0.0)
    }

    /// Cycle ms: verify `shape` after the draft work, plus host time.
    pub fn cycle_ms(&self, shape: Shape, chain: usize, drafting: bool) -> f64 {
        self.verify_ms(shape) + self.draft_ms(shape.sequences, drafting, chain)
            + self.fits[concurrency_bucket(shape.sequences)].host_ms
    }

    /// Fitted verify (intercept ms, slope relative to the table, ms per extra
    /// sequence) and draft (ms per drafting cycle, per chained step).
    pub fn fitted(&self, sequences: usize) -> ([f64; 3], [f64; 2]) {
        let fit = &self.fits[concurrency_bucket(sequences)];
        (fit.verify.theta, fit.draft.theta)
    }

    /// Folds one observed verify step in and refits.
    pub fn observe_verify(&mut self, shape: Shape, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            let predicted = self.verify_ms(shape);
            let features = self.features(shape);
            let fit = &mut self.fits[concurrency_bucket(shape.sequences)].verify;
            fit.observe(&features, ms.clamp(0.5 * predicted, 2.0 * predicted));
            let [a, b, c] = fit.theta;
            // Rows never get cheaper with more of them; the intercept stays physical.
            fit.theta = [a.max(0.25 * self.table[1]), b.max(0.05), c];
        }
    }

    /// Folds one drafting cycle's draft ms in (drafted before planning).
    pub fn observe_draft(&mut self, sequences: usize, ms: f64) {
        self.observe_chain(sequences, 0, ms);
    }

    /// Folds a drafting cycle of `steps` chained draft steps that took `ms` in.
    pub fn observe_chain(&mut self, sequences: usize, steps: usize, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            let predicted = self.draft_ms(sequences, true, steps);
            let y = if predicted > 0.0 { ms.clamp(0.25 * predicted, 4.0 * predicted) } else { ms };
            let fit = &mut self.fits[concurrency_bucket(sequences)].draft;
            fit.observe(&[1.0, steps as f64], y);
            let [d, e] = fit.theta;
            fit.theta = [d.max(0.0), e.max(0.0)];
        }
    }

    /// Folds one cycle's host time outside the draft and verify steps in.
    pub fn observe_host(&mut self, sequences: usize, ms: f64) {
        if ms.is_finite() && ms >= 0.0 {
            let host_ms = &mut self.fits[concurrency_bucket(sequences)].host_ms;
            *host_ms += 0.1 * (ms.min(20.0) - *host_ms);
        }
    }
}

/// Solves a symmetric positive definite system (Gaussian elimination with
/// partial pivoting); None when it is singular.
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let pivot = (col..N).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..N {
            let f = a[row][col] / a[col][col];
            for k in col..N {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; N];
    for row in (0..N).rev() {
        let tail: f64 = (row + 1..N).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Sequences that plan together: identical ones (same tokens at the same
/// position) form one group of `members` sharing one draft.
#[derive(Debug, Clone)]
pub(crate) struct Group {
    /// Conditional acceptance per draft position, at most as long as the
    /// group may verify.
    pub confidence: Vec<f64>,
    pub members: usize,
    /// Drafts the group verifies regardless of price.
    pub minimum: usize,
}

/// How a plan's drafts are paid for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Drafter {
    /// Drafted before planning (DFlash blocks, fixed MTP stages): every
    /// plan pays the draft cost.
    Block,
    /// Chained draft steps (MTP): a cycle pays one step per position of its
    /// deepest sequence.
    Chain,
}

/// Sequences in the step that do not draft (each verifies its next token).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Base {
    pub rows: usize,
    pub distinct: usize,
}

/// A plan's step shape and expected emitted tokens.
fn tally(groups: &[Group], lengths: &[usize], base: Base) -> (Shape, f64) {
    let mut shape = Shape { rows: base.rows, distinct: base.distinct, sequences: base.rows };
    let mut expected = base.rows as f64;
    for (group, &length) in groups.iter().zip(lengths) {
        shape.rows += group.members * (1 + length);
        shape.distinct += 1 + length;
        shape.sequences += group.members;
        let mut survival = 1.0;
        expected += group.members as f64;
        for &rate in &group.confidence[..length] {
            survival *= rate.clamp(0.0, 1.0);
            expected += group.members as f64 * survival;
        }
    }
    (shape, expected)
}

/// Expected emitted tokens and cycle ms of the plan `lengths`.
fn evaluate(groups: &[Group], lengths: &[usize], base: Base, drafter: Drafter, cost: &CycleCost) -> (f64, f64) {
    let (shape, expected) = tally(groups, lengths, base);
    let ms = match drafter {
        Drafter::Block => cost.cycle_ms(shape, 0, true),
        Drafter::Chain => cost.cycle_ms(shape, lengths.iter().copied().max().unwrap_or(0), false),
    };
    (expected, ms)
}

/// Expected emitted tokens per cycle millisecond of the plan `lengths`.
pub(crate) fn rate(groups: &[Group], lengths: &[usize], base: Base, drafter: Drafter, cost: &CycleCost) -> f64 {
    let (expected, ms) = evaluate(groups, lengths, base, drafter, cost);
    expected / ms
}

/// Draft counts per group maximizing expected emitted tokens per cycle
/// millisecond, and that rate. Starting from each group's minimum,
/// draft positions are admitted greedily by expected emitted tokens per
/// marginal verify millisecond; the best plan seen wins. Chained drafters
/// repeat this under every depth cap (a deeper chain costs a draft step for
/// all).
pub(crate) fn allocate(groups: &[Group], base: Base, drafter: Drafter, cost: &CycleCost) -> (Vec<usize>, f64) {
    let minimum: Vec<usize> = groups.iter().map(|g| g.minimum.min(g.confidence.len())).collect();
    let deepest = groups.iter().map(|g| g.confidence.len()).max().unwrap_or(0);
    let floor = minimum.iter().copied().max().unwrap_or(0);
    let caps: Vec<usize> = match drafter {
        Drafter::Block => vec![deepest],
        Drafter::Chain => (floor..=deepest).collect(),
    };
    let plan = |lengths: &[usize]| {
        let (expected, ms) = evaluate(groups, lengths, base, drafter, cost);
        expected / ms
    };
    let mut best = (minimum.clone(), plan(&minimum));
    for cap in caps {
        let mut lengths = minimum.clone();
        let (mut shape, _) = tally(groups, &lengths, base);
        let mut survival: Vec<f64> = groups.iter().zip(&lengths)
            .map(|(g, &n)| g.confidence[..n].iter().map(|r| r.clamp(0.0, 1.0)).product()).collect();
        loop {
            let verify = cost.verify_ms(shape);
            let mut pick: Option<(f64, usize, f64)> = None;
            for (g, group) in groups.iter().enumerate() {
                let n = lengths[g];
                if n >= cap || n >= group.confidence.len() {
                    continue;
                }
                let next = survival[g] * group.confidence[n].clamp(0.0, 1.0);
                if next <= 0.0 {
                    continue;
                }
                let grown = Shape { rows: shape.rows + group.members, distinct: shape.distinct + 1, ..shape };
                let marginal = (cost.verify_ms(grown) - verify).max(1e-6);
                let value = group.members as f64 * next / marginal;
                if pick.is_none_or(|(v, _, _)| value > v) {
                    pick = Some((value, g, next));
                }
            }
            let Some((_, g, next)) = pick else { break };
            lengths[g] += 1;
            survival[g] = next;
            shape.rows += groups[g].members;
            shape.distinct += 1;
            let rate = plan(&lengths);
            if rate > best.1 {
                best = (lengths.clone(), rate);
            }
        }
    }
    best
}

/// Longest run of steps without a draft step after plans that verified none.
const MAX_DRAFT_SKIP: usize = 8;

/// After plans that verify no drafts, skip drafting for a while (doubling up
/// to `MAX_DRAFT_SKIP` steps): a draft step costs about a verified row.
#[derive(Debug, Clone)]
pub(crate) struct DraftSkip {
    skip: usize,
    next: usize,
}

impl Default for DraftSkip {
    fn default() -> Self {
        Self { skip: 0, next: 1 }
    }
}

impl DraftSkip {
    /// Whether this step drafts.
    pub fn drafts(&self) -> bool {
        self.skip == 0
    }

    /// After planning a step: `drafted` when it drafted, `none_planned` when
    /// the plan verifies no drafts (adaptive plans only).
    pub fn after(&mut self, drafted: bool, none_planned: bool) {
        if self.skip > 0 {
            self.skip -= 1;
        } else if drafted {
            if none_planned {
                self.skip = self.next;
                self.next = (self.next * 2).min(MAX_DRAFT_SKIP);
            } else {
                self.next = 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: [(usize, f64); 4] = [(1, 10.0), (4, 13.0), (16, 25.0), (64, 73.0)];

    #[test]
    fn history_censors_positions_after_a_miss() {
        let mut history = DraftHistory::default();
        history.observe(5, 2);
        // Positions 1, 2 accepted, 3 rejected, 4 and 5 unobserved.
        assert_eq!(history.conditional(5), vec![4.0 / 5.0, 4.0 / 5.0, 3.0 / 5.0, 0.75, 0.75]);
    }

    #[test]
    fn pooled_rates_borrow_other_sequences() {
        let mut me = DraftHistory::default();
        for _ in 0..4 {
            me.observe(3, 0);
        }
        // Alone: the 3-in-4 prior.
        assert_eq!(me.pooled(2, std::iter::empty()), me.conditional(2));
        let mut other = DraftHistory::default();
        for _ in 0..16 {
            other.observe(3, 3);
        }
        // Two others at 16/16 on position 1: prior (3 + 32) / (4 + 32) at 8 trials, then 0 of 4 own.
        let rates = me.pooled(2, [&other, &other].into_iter());
        let prior = 35.0 / 36.0;
        assert!((rates[0] - 8.0 * prior / 12.0).abs() < 1e-12, "{rates:?}");
        // Position 2 is unobserved for me: the prior itself.
        assert!((rates[1] - prior).abs() < 1e-12, "{rates:?}");
    }

    #[test]
    fn calibration_learns_over_dispersed_rates() {
        let mut calibration = Calibration::default();
        assert!((calibration.apply(0.4) - 0.4).abs() < 1e-12);
        // History rates of 0.4 and 0.9 whose positions are accepted 65% and 85% of the time.
        for step in 0..2000 {
            // Spread over each 20 steps so the forgetting window sees the rates.
            let low = (step * 7) % 20 < 13;
            let high = (step * 3) % 20 < 17;
            calibration.observe(0.4, low);
            calibration.observe(0.9, high);
        }
        assert!((calibration.apply(0.4) - 0.65).abs() < 0.03, "{}", calibration.apply(0.4));
        assert!((calibration.apply(0.9) - 0.85).abs() < 0.03, "{}", calibration.apply(0.9));
        // A band of rates whose outcomes run against them keeps a positive slope.
        let mut inverted = Calibration::default();
        for step in 0..500 {
            inverted.observe(0.7, step % 2 == 0);
            inverted.observe(0.8, step % 4 == 0);
        }
        let [_, slope] = inverted.fitted();
        assert_eq!(slope, CALIBRATION_SLOPE.0);
        assert!(inverted.apply(0.8) > inverted.apply(0.7));
        // Censoring: after 2 of 3 accepted, positions 1-3 were verified, not 4.
        let mut outcome = Calibration::default();
        outcome.observe_outcome(&[0.5, 0.5, 0.5, 0.5], 4, 2);
        let (gram, _) = (outcome.fit.gram, ());
        assert!((gram[0][0] - (1.0 + 0.99 + 0.99 * 0.99)).abs() < 1e-12, "{gram:?}");
    }

    #[test]
    fn cost_starts_at_the_table() {
        let cost = CycleCost::new(&TABLE, 64);
        assert!((cost.verify_ms(Shape::plain(1, 1)) - 10.0).abs() < 1e-9);
        assert!((cost.verify_ms(Shape::plain(8, 1)) - 17.0).abs() < 1e-9);
        assert!((cost.verify_ms(Shape::plain(64, 1)) - 73.0).abs() < 1e-9);
        // Past the table, linear at its last slope.
        let wide = CycleCost::new(&TABLE, 80);
        assert!((wide.verify_ms(Shape::plain(80, 1)) - 89.0).abs() < 1e-9);
        let dup = CycleCost::new(&TABLE, 64).duplicate_rows(0.5);
        assert!((dup.verify_ms(Shape { rows: 8, distinct: 2, sequences: 4 }) - 14.0).abs() < 1e-9);
    }

    /// Noiseless steps of a cost `a + b * rows` over varied row counts.
    fn feed(cost: &mut CycleCost, a: f64, b: f64, sequences: usize, steps: usize) {
        for step in 0..steps {
            let rows = sequences * (1 + step % 5);
            cost.observe_verify(Shape::plain(rows, sequences), a + b * rows as f64);
        }
    }

    #[test]
    fn fit_learns_intercept_and_slope() {
        // The table's slope is 1 ms per row; serving sees 12 ms + 0.4 ms per row.
        let mut cost = CycleCost::new(&TABLE, 64);
        feed(&mut cost, 11.6, 0.4, 4, 200);
        for rows in [4, 8, 12, 20] {
            let expected = 11.6 + 0.4 * rows as f64;
            let got = cost.verify_ms(Shape::plain(rows, 4));
            assert!((got - expected).abs() < 0.3, "{rows} rows: {got} vs {expected}");
        }
        // A single ratio could not do this: 20 rows at the table's shape would cost 3x 4 rows.
        let ([_, slope, _], _) = cost.fitted(4);
        assert!(slope < 0.6, "{slope}");
    }

    #[test]
    fn c16_sweep_does_not_poison_c1_plans() {
        // GLM Flash table; a wide sweep learns sub-additive expert reads.
        let table = [(1, 19.1), (2, 26.0), (4, 35.2), (6, 43.5), (8, 51.3), (64, 200.0)];
        let fresh = CycleCost::new(&table, 128).drafts(5.0, 0.0, 0.3);
        let mut swept = fresh.clone();
        for step in 0..200 {
            let rows = 16 * (4 + step % 5);
            swept.observe_verify(Shape::plain(rows, 16), 30.0 + 0.6 * rows as f64);
            swept.observe_draft(16, 12.0);
            swept.observe_host(16, 3.0);
        }
        let groups = [group(0.8, 7, 1)];
        assert_eq!(allocate(&groups, Base::default(), Drafter::Block, &swept),
            allocate(&groups, Base::default(), Drafter::Block, &fresh));
        for rows in 1..=8 {
            assert_eq!(swept.cycle_ms(Shape::plain(rows, 1), 0, true),
                fresh.cycle_ms(Shape::plain(rows, 1), 0, true));
        }
        // Returning to a previously learned regime preserves its fit too.
        feed(&mut swept, 20.0, 2.0, 1, 50);
        let learned = swept.fitted(1);
        feed(&mut swept, 30.0, 0.5, 16, 100);
        assert_eq!(swept.fitted(1), learned);
    }

    #[test]
    fn concurrency_bucket_boundaries_share_only_their_own_fit() {
        assert_eq!((0..=17).map(concurrency_bucket).collect::<Vec<_>>(),
            vec![0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3]);
        let mut cost = CycleCost::new(&TABLE, 128).drafts(5.0, 0.0, 0.3);
        let prior = cost.fitted(1);
        feed(&mut cost, 12.0, 0.4, 4, 100);
        assert_eq!(cost.fitted(2), cost.fitted(4));
        assert_ne!(cost.fitted(4), prior);
        for sequences in [1, 5, 8, 9, 16, 32] {
            assert_eq!(cost.fitted(sequences), prior);
        }
    }

    #[test]
    fn fit_forgets_an_old_regime() {
        let mut cost = CycleCost::new(&TABLE, 64);
        feed(&mut cost, 9.0, 1.0, 1, 200);
        feed(&mut cost, 9.0, 3.0, 1, 300);
        let got = cost.verify_ms(Shape::plain(5, 1));
        assert!((got - 24.0).abs() < 0.5, "{got}");
    }

    #[test]
    fn fit_without_spread_stays_near_the_table_shape() {
        // Every step at 8 rows, 20% dearer than the table: the fit matches the
        // point and splits the difference between intercept and slope.
        let mut cost = CycleCost::new(&TABLE, 64);
        for _ in 0..200 {
            cost.observe_verify(Shape::plain(8, 1), 1.2 * 17.0);
        }
        assert!((cost.verify_ms(Shape::plain(8, 1)) - 20.4).abs() < 0.2);
        let ([a, b, _], _) = cost.fitted(1);
        assert!(a > 10.0 && b > 1.0, "{a} {b}");
        // Outliers are clipped at twice the prediction.
        let before = cost.verify_ms(Shape::plain(8, 1));
        cost.observe_verify(Shape::plain(8, 1), 1e6);
        assert!(cost.verify_ms(Shape::plain(8, 1)) < 1.1 * before);
    }

    #[test]
    fn draft_fit_separates_the_first_step() {
        // An MTP cycle: 0.6 ms over the pending rows plus 0.8 ms per chained step.
        let mut cost = CycleCost::new(&TABLE, 64).drafts(0.0, 0.9, 0.0);
        for step in 0..200 {
            let steps = 1 + step % 4;
            cost.observe_chain(1, steps, 0.6 + 0.8 * steps as f64);
            cost.observe_host(1, 0.5);
        }
        for steps in 1..=4 {
            let got = cost.draft_ms(1, true, steps);
            assert!((got - 0.6 - 0.8 * steps as f64).abs() < 0.05, "{steps}: {got}");
        }
        assert_eq!(cost.draft_ms(1, false, 0), 0.0);
        let shape = Shape::plain(4, 1);
        assert!((cost.cycle_ms(shape, 0, false) - 13.5).abs() < 0.01);
        // A block drafter pays its draft whatever the plan.
        let mut block = CycleCost::new(&TABLE, 64).drafts(5.0, 0.0, 0.0);
        for _ in 0..200 {
            block.observe_draft(1, 3.0);
        }
        assert!((block.cycle_ms(shape, 0, true) - 16.0).abs() < 0.05);
    }

    fn group(rate: f64, room: usize, members: usize) -> Group {
        Group { confidence: vec![rate; room], members, minimum: 0 }
    }

    #[test]
    fn allocation_follows_acceptance() {
        let cost = CycleCost::new(&TABLE, 64).drafts(0.0, 0.5, 0.0);
        let (deep, _) = allocate(&[group(0.95, 7, 1)], Base::default(), Drafter::Chain, &cost);
        assert!(deep[0] >= 5, "{deep:?}");
        let (none, _) = allocate(&[group(0.05, 7, 1)], Base::default(), Drafter::Chain, &cost);
        assert_eq!(none, vec![0]);
        let (both, _) = allocate(&[group(0.9, 7, 1), group(0.3, 7, 1)], Base::default(), Drafter::Chain, &cost);
        assert!(both[0] > both[1], "{both:?}");
        // Minimums are verified regardless of price.
        let forced = Group { minimum: 3, ..group(0.05, 7, 1) };
        assert_eq!(allocate(&[forced], Base::default(), Drafter::Block, &cost).0, vec![3]);
    }

    #[test]
    fn allocation_is_optimal_on_small_cases() {
        // Brute force over every plan of three sequences.
        let mut cost = CycleCost::new(&TABLE, 64).drafts(0.0, 0.8, 0.3);
        feed(&mut cost, 12.0, 0.6, 3, 100);
        let groups = [Group { confidence: vec![0.9, 0.8, 0.7, 0.6], members: 1, minimum: 0 },
            Group { confidence: vec![0.6, 0.5, 0.5, 0.4], members: 1, minimum: 0 },
            Group { confidence: vec![0.95, 0.9, 0.2, 0.9], members: 2, minimum: 0 }];
        for drafter in [Drafter::Chain, Drafter::Block] {
            let (plan, got) = allocate(&groups, Base { rows: 1, distinct: 1 }, drafter, &cost);
            let mut best = 0.0f64;
            for a in 0..=4 {
                for b in 0..=4 {
                    for c in 0..=4 {
                        best = best.max(rate(&groups, &[a, b, c], Base { rows: 1, distinct: 1 }, drafter, &cost));
                    }
                }
            }
            assert!(got >= 0.995 * best, "{drafter:?} {plan:?}: {got} vs {best}");
        }
    }

    #[test]
    fn cheaper_rows_buy_deeper_drafts() {
        // Four sequences at 60% acceptance: the table's steep rows stop them
        // early; the flat slope serving observed buys more depth.
        let table = CycleCost::new(&TABLE, 64).drafts(0.0, 0.5, 0.0);
        let mut flat = table.clone();
        feed(&mut flat, 12.0, 0.2, 4, 200);
        let groups = vec![group(0.7, 7, 1); 4];
        let steep: usize = allocate(&groups, Base::default(), Drafter::Chain, &table).0.iter().sum();
        let cheap: usize = allocate(&groups, Base::default(), Drafter::Chain, &flat).0.iter().sum();
        assert!(cheap > steep, "{cheap} vs {steep}");
    }

    #[test]
    fn identical_groups_pay_duplicate_rows() {
        let cost = CycleCost::new(&TABLE, 64).duplicate_rows(0.1);
        let shared = Shape { rows: 16, distinct: 4, sequences: 4 };
        assert!(cost.verify_ms(shared) < cost.verify_ms(Shape::plain(16, 4)) - 8.0);
        let (lengths, _) = allocate(&[group(0.6, 7, 4)], Base::default(), Drafter::Block, &cost);
        let (alone, _) = allocate(&[group(0.6, 7, 1)], Base::default(), Drafter::Block, &cost);
        assert!(lengths[0] >= alone[0], "{lengths:?} {alone:?}");
    }

    #[test]
    fn draft_skip_backs_off_and_resets() {
        let mut skip = DraftSkip::default();
        skip.after(true, true);
        assert!(!skip.drafts());
        skip.after(false, false);
        assert!(skip.drafts());
        skip.after(true, true);
        skip.after(false, false);
        assert!(!skip.drafts());
        skip.after(false, false);
        skip.after(true, false);
        assert!(skip.drafts());
    }

    #[test]
    fn solve_matches_known_system() {
        let x = solve([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]], [5.0, 5.0, 3.0]).unwrap();
        assert!(solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]).is_none());
        for (got, want) in x.iter().zip([1.0, 1.0, 1.0]) {
            assert!((got - want).abs() < 1e-12);
        }
    }
}

/// Where a serve path appends its per-cycle speculation trace (JSON lines):
/// `CUTEAFD_SPECULATION_TRACE`, else the family's pre-rename variable (`legacy`,
/// e.g. `CUTEAFD_GLM_TRACE`), kept for one release. Returns the variable read
/// and the path.
pub(crate) fn speculation_trace_path(legacy: &'static str) -> Option<(&'static str, String)> {
    ["CUTEAFD_SPECULATION_TRACE", legacy]
        .into_iter()
        .find_map(|var| std::env::var(var).ok().filter(|path| !path.is_empty()).map(|path| (var, path)))
}
