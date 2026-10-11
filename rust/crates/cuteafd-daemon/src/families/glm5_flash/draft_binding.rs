//! GLM 5.3 Flash on the shared resource-priced draft policy
//! (`cuteafd_core::draft_policy`, v3 D2).
//!
//! **Geometry.** Every backbone layer, in order. Dense layers (the first three)
//! have no class. Each MoE layer's experts are priced in the class of where
//! they run: `local` (FP8 or EXL3 on the coordinator RTX) or `remote` (Spark
//! ranks). `slice_bytes` is what one device reads for one expert at its tensor
//! parallel width: from `Fp8Layer::bytes_for` for an FP8/NVFP4 checkpoint
//! (the busiest rank, divided by the experts), and from the EXL3 manifest's
//! per-projection tiers (`H * S * bits / 8` trellis payload plus the
//! intermediate rotations) for EXL3. Layer 0 has no preceding boundary, so its
//! time is in the round residual. One regime (one decode lane). The width is
//! the drafter's single block (DFlash2 7 drafts, dSpark 8); the request budget
//! is the step's row budget, since every active sequence verifies at least its
//! anchor row.
//!
//! **Evidence.** DFlash2: the selector features through the keyed
//! `SelectorFit` prior (the request's own history rate is its first feature),
//! then the policy's online per-position Platt. dSpark: the confidence head's
//! sigmoid blended with the history rate at the deployment's head weight (GLM
//! Flash's existing dSpark prior), then Platt.
//!
//! **Seed and cold start** (D3). The policy starts from GLM Flash's typical
//! warm fits (D2's cards) and verifies `COLD_DRAFTS` per request until its
//! fits are warm and have predicted enough rounds within `WARM_ERROR`; then
//! it engages for good. No table from the old per-family policy is used.
//!
//! **One row budget** (D3): the step's verify rows across every request, no
//! equal quota; the selection gives each row to the draft that pays most.
//!
//! **Copy extension** (D3, PLAN "Copies extend the neural draft"): when a copy
//! window from the request's own history agrees with the drafter over its
//! whole window, the copy's continuation past the window becomes further
//! draft positions, `TAIL_PRIOR` raw, calibrated per position by the policy
//! and blended with the request's own outcomes there; the policy prices them
//! with the resource model like any other row.
//!
//! **Rules kept as adapter data** (PLAN "v3 draft policy" section 3): a cold
//! DFlash2 sequence (fewer than four outcomes) verifies five drafts for its
//! first four cycles, and a lone warm DFlash2 sequence keeps five drafts
//! unless the selection beats them by 2% in predicted tokens per µs.
use crate::families::glm5::dflash_policy::{DraftHistory, START_DRAFTS};
use crate::shared::draft::binding::{self, DraftPass, ObserveError, RoundRecord, RoundRequest};
use crate::shared::draft::clock::RoundTimes;
use crate::shared::draft::evidence::{Censor, DraftSource, Evidence, SelectorFeatures};
use crate::shared::draft::routes::RoundRoutes;
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DraftCandidate, DraftPolicy, LayerResource, PolicyGeometry, PolicySeed, ResourceClass};
use std::sync::Mutex;

/// Resource classes: coordinator RTX and Spark expert ranks.
const LOCAL: u8 = 0;
const REMOTE: u8 = 1;
/// Rows per weight-read group of the installed expert kernels. The CuTe
/// grouped slice kernels use 16; the fp8moe/EXL3 manifests do not record a
/// tile yet, and a wrong constant biases the fitted bandwidth, not the order.
const GROUP_ROWS: u8 = 16;
/// Most drafter positions the binding prices (GLM drafters propose 7 or 8).
const MAX_POSITIONS: usize = 16;
/// Most copy positions appended past the drafter's window.
pub(crate) const COPY_EXTENSION: usize = 32;
/// Raw acceptance of an agreed copy's continuation before calibration.
pub(crate) const TAIL_PRIOR: f64 = 0.9;
/// Drafts per request until the policy engages: about the C1 optimum of the
/// D2 cards (3-5 at the measured acceptance), never the minimal one.
const COLD_DRAFTS: usize = 4;
/// Timed rounds and smoothed relative prediction error before engaging.
const WARM_ROUNDS: u64 = 24;
const WARM_ERROR: f64 = 0.08;

/// GLM Flash's typical warm fits (D2, rc3 cards, EXL3 and FP8, 1 and 2 RTX,
/// TP2/TP4 Sparks: round residual 870-1050 µs + 0-36 µs/row + 60-160
/// µs/request; DFlash2 pass 900-1730 µs + 90-470 µs/request; Spark layers
/// 90-250 µs + 9-12 µs/row + 4.3-5.0 µs/MB).
pub(crate) fn seed() -> PolicySeed {
    PolicySeed { round: [950., 20., 100.], draft: [1300., 300.], samples: 4., cold_drafts: COLD_DRAFTS,
        warm_rounds: WARM_ROUNDS, warm_error: WARM_ERROR }
}

/// Spark expert layers of GLM Flash as D2 fitted them (~215 GB/s effective).
fn spark_class() -> ResourceClass {
    ResourceClass { prior: [150., 10., 4.7], ..ResourceClass::spark() }
}
/// A lone warm DFlash2 sequence keeps `START_DRAFTS` unless the selection
/// beats them by this ratio.
const REFERENCE_MARGIN: f64 = 1.02;

/// Which verify-length policy GLM Flash runs (`CUTEAFD_GLMF_DRAFT_POLICY`):
/// `cycle` keeps the `CycleCost` table fit; `shared` is the resource-priced
/// core with this binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PolicyKind { Cycle, Shared }

impl PolicyKind {
    pub fn from_env() -> Result<Self> {
        Self::parse(std::env::var("CUTEAFD_GLMF_DRAFT_POLICY").ok().as_deref())
    }
    fn parse(value: Option<&str>) -> Result<Self> {
        match value.map(str::trim) {
            None | Some("" | "cycle") => Ok(Self::Cycle),
            Some("shared") => Ok(Self::Shared),
            Some(other) => anyhow::bail!("CUTEAFD_GLMF_DRAFT_POLICY must be cycle or shared, got {other:?}"),
        }
    }
}

/// Where a layer's routed experts run and what one device reads per expert.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LayerHome {
    Dense,
    Local { slice_bytes: f64 },
    Remote { slice_bytes: f64 },
}

/// GLM Flash's policy geometry from each layer's home, the router's expert
/// count and top-k, the step's row budget, the drafter's block and the copy
/// positions that may extend it.
pub(crate) fn geometry(homes: &[LayerHome], experts: usize, topk: usize, rows: usize, drafts: usize,
    extension: usize) -> Result<PolicyGeometry> {
    ensure!(drafts > 0, "a speculating drafter proposes at least one token");
    let width = drafts.min(MAX_POSITIONS);
    Ok(PolicyGeometry {
        layers: homes.iter().enumerate().map(|(layer, home)| {
            let (class, slice_bytes) = match *home {
                LayerHome::Dense => (None, 0.),
                LayerHome::Local { slice_bytes } => (Some(LOCAL), slice_bytes),
                LayerHome::Remote { slice_bytes } => (Some(REMOTE), slice_bytes),
            };
            LayerResource { class, slice_bytes, group_rows: GROUP_ROWS, timed: layer > 0 }
        }).collect(),
        experts: u16::try_from(experts).context("GLM Flash routes more than 65535 experts")?,
        topk: u8::try_from(topk).context("GLM Flash top-k exceeds 255")?,
        max_requests: rows,
        max_positions: width + extension,
        widths: vec![width],
        classes: vec![ResourceClass::rtx(), spark_class()],
        regimes: 1,
    })
}

/// Per-device bytes of one expert of an FP8/NVFP4 checkpoint at `tp` ranks:
/// the busiest rank's layer slice over the experts (`Fp8Layer::bytes_for`).
pub(crate) fn fp8_slice_bytes(tensors: &cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors, tp: usize)
    -> Result<f64> {
    use cuteafd_loader::formats::fp8_experts::Slicing;
    let experts = tensors.shape().experts.max(1) as f64;
    let widest = (0..tp).map(|rank| crate::shared::experts::fp8::Fp8Layer::bytes_for(tensors, tp, rank, Slicing::Padded))
        .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
    Ok(widest as f64 / experts)
}

/// Per-device bytes of one expert of `layer` from an EXL3 manifest at `tp`
/// ranks (see [`exl3_bytes`]).
pub(crate) fn exl3_slice_bytes(manifest: &cuteafd_loader::V41Exl3Manifest, layer: usize, tp: usize) -> Result<f64> {
    use cuteafd_loader::V41Exl3ProjectionKind as Kind;
    let shape = manifest.experts;
    exl3_bytes(shape.hidden, shape.intermediate, shape.experts, tp, |expert, projection| {
        let kind = [Kind::Gate, Kind::Up, Kind::Down][projection];
        manifest.projections.get(&manifest.naming.projection(false, layer, expert, kind)).map(|p| p.bits)
            .with_context(|| format!("EXL3 projection of layer {layer} expert {expert} missing"))
    })
}

/// One expert's per-device EXL3 bytes at `tp` ranks, averaged over the
/// layer's `experts`: each projection's trellis payload at its own tier
/// (`bits(expert, projection)`, projections gate/up/down) over the widest
/// rank's intermediate slice, plus its hidden- and intermediate-side
/// rotations (BF16).
fn exl3_bytes(hidden: usize, intermediate: usize, experts: usize, tp: usize,
    bits: impl Fn(usize, usize) -> Result<usize>) -> Result<f64> {
    let width = (intermediate / 128).div_ceil(tp.max(1)) * 128;
    let mut total = 0usize;
    for expert in 0..experts {
        for projection in 0..3 {
            total += hidden * width * bits(expert, projection)? / 8 + hidden * 2 + width * 2;
        }
    }
    Ok(total as f64 / experts.max(1) as f64)
}

/// Layer homes of a deployment: `local` experts on this GPU (FP8 or EXL3
/// catalog), else `spark_ranks` Spark ranks serving `snapshot`'s experts.
pub(crate) fn homes(dense: &[bool], catalog: Option<&cuteafd_loader::OfficialV41Catalog>,
    snapshot: &std::path::Path, spark_ranks: Option<usize>) -> Result<Vec<LayerHome>> {
    let owned;
    let (catalog, tp, local) = match (catalog, spark_ranks) {
        (Some(catalog), _) => (catalog, 1, true),
        (None, Some(ranks)) => {
            owned = cuteafd_loader::read_expert_catalog(snapshot)?;
            (&owned, ranks, false)
        }
        (None, None) => anyhow::bail!("no routed expert placement to price"),
    };
    let fp8 = catalog.fp8().map(|tensors| fp8_slice_bytes(tensors, tp)).transpose()?;
    dense.iter().enumerate().map(|(layer, &dense)| {
        if dense { return Ok(LayerHome::Dense); }
        let slice_bytes = match (fp8, catalog.exl3()) {
            (Some(bytes), _) => bytes,
            (None, Some(manifest)) => exl3_slice_bytes(manifest, layer, tp)?,
            (None, None) => anyhow::bail!("routed experts are neither FP8/NVFP4 nor EXL3"),
        };
        Ok(if local { LayerHome::Local { slice_bytes } } else { LayerHome::Remote { slice_bytes } })
    }).collect()
}

/// One sequence's draft this round, as the binding prices it.
pub(crate) enum DraftEvidence<'a> {
    /// DFlash2's selector features of each drafted token.
    Selector(&'a [[f32; 4]]),
    /// A dSpark confidence head's per-token acceptance.
    Head(&'a [f32]),
}

/// The keyed selector prior and dSpark head blend the binding evaluates.
pub(crate) enum Prior {
    Selector(crate::shared::draft_confidence::SelectorFit),
    /// dSpark: the head's weight against the history rate, in logit space.
    Head,
}

/// Per-position `p0` of one draft: the prior evaluated on its evidence and the
/// request's history. `None` when the evidence does not fit the prior (an
/// invalid selector row): the request then verifies on history alone.
pub(crate) fn prior(prior: &Prior, history: &DraftHistory, evidence: &DraftEvidence<'_>) -> Vec<f64> {
    match (prior, evidence) {
        (Prior::Selector(fit), DraftEvidence::Selector(features)) => {
            let width = features.len().min(MAX_POSITIONS);
            let rates = history.conditional(width);
            fit.confidence(&rates, &features[..width]).unwrap_or(rates)
        }
        (_, DraftEvidence::Head(head)) =>
            crate::families::glm5::dflash_policy::head_confidence(history, &head[..head.len().min(MAX_POSITIONS)]),
        (Prior::Head, DraftEvidence::Selector(features)) => history.conditional(features.len().min(MAX_POSITIONS)),
    }
}

/// One active sequence's inputs to the round's selection.
pub(crate) struct Candidate<'a> {
    pub id: u64,
    /// `p0` per drafted position, copy extension included (empty: no draft).
    pub prior: &'a [f64],
    /// Most drafts the sequence may verify (its output and context room).
    pub limit: usize,
    /// DFlash2 without a trained head: the cold and lone five-draft rules apply.
    pub selector: bool,
    pub cold: bool,
}

/// What the selection decided and predicted.
pub(crate) struct Selection {
    pub lengths: Vec<usize>,
    pub predicted: Option<f64>,
}

/// Draft counts for every active sequence through the shared policy under
/// one budget of `max_rows` verify rows (anchors included), with GLM's kept
/// adapter rules. Until the policy engages every request verifies up to its
/// cold drafts.
pub(crate) fn select(policy: &mut DraftPolicy, candidates: &[Candidate<'_>], width: usize, max_rows: usize)
    -> Selection {
    let positions = policy.geometry().max_positions;
    let confidence: Vec<&[f64]> = candidates.iter()
        .map(|c| &c.prior[..c.prior.len().min(c.limit).min(positions)]).collect();
    let cores: Vec<DraftCandidate<'_>> = candidates.iter().zip(&confidence)
        .map(|(c, &confidence)| DraftCandidate { id: c.id, confidence }).collect();
    let minimum = |c: &Candidate<'_>, available: usize| if c.selector && c.cold { START_DRAFTS.min(available) } else { 0 };
    let cold = policy.cold_drafts().unwrap_or(usize::MAX);
    let mut lengths: Vec<usize> = confidence.iter().map(|c| c.len().min(cold)).collect();
    let mut predicted = None;
    if let Ok(Some(selection)) = policy.select_within(false, &cores, width, max_rows) {
        lengths = selection.lengths;
        predicted = Some(selection.predicted_us);
        let ratio = selection.expected_tokens / selection.predicted_us;
        if let [only] = candidates {
            let n = START_DRAFTS.min(confidence[0].len());
            if only.selector && !only.cold && lengths[0] != n {
                if let Some((tokens, us)) = policy.evaluate(false, &cores, &[n], width) {
                    if ratio < tokens / us * REFERENCE_MARGIN {
                        lengths[0] = n;
                        predicted = Some(us);
                    }
                }
            }
        }
    }
    for ((length, c), confidence) in lengths.iter_mut().zip(candidates).zip(&confidence) {
        let floor = minimum(c, confidence.len());
        if *length < floor {
            *length = floor;
            predicted = None;
        }
    }
    if fit_budget(&mut lengths, max_rows) { predicted = None; }
    Selection { lengths, predicted }
}

/// Inputs retained inside a round; expensive diagnostics run only after its
/// clock closes, before the policy learns from the observed outcome.
pub(crate) struct DecisionTrace {
    requests: Vec<(u64, Vec<f64>, usize, bool, bool)>,
    width: usize,
    max_rows: usize,
    selection: Selection,
}

impl DecisionTrace {
    pub fn capture(candidates: &[Candidate<'_>], width: usize, max_rows: usize, selection: &Selection) -> Self {
        Self { requests: candidates.iter().map(|c| (c.id, c.prior.to_vec(), c.limit, c.selector, c.cold)).collect(),
            width, max_rows, selection: Selection { lengths: selection.lengths.clone(), predicted: selection.predicted } }
    }
    pub fn evaluate(self, policy: &mut DraftPolicy) -> serde_json::Value {
        let candidates: Vec<_> = self.requests.iter().map(|&(id, ref prior, limit, selector, cold)|
            Candidate { id, prior, limit, selector, cold }).collect();
        decision_trace(policy, &candidates, self.width, self.max_rows, &self.selection)
    }
}

/// Opt-in decision diagnostic: vary each request's length with the other
/// requests held at their chosen lengths. At C1 this enumerates every shape.
/// Call only when tracing is enabled; evaluations do not train the policy.
pub(crate) fn decision_trace(policy: &mut DraftPolicy, candidates: &[Candidate<'_>], width: usize,
    max_rows: usize, selection: &Selection) -> serde_json::Value {
    let cores: Vec<_> = candidates.iter().map(|c| DraftCandidate { id: c.id,
        confidence: &c.prior[..c.prior.len().min(c.limit).min(policy.geometry().max_positions)] }).collect();
    let requests: Vec<_> = cores.iter().enumerate().map(|(i, c)| {
        let probabilities = policy.probabilities(*c);
        let cold_request = candidates[i].cold;
        let others = candidates.len() + selection.lengths.iter().sum::<usize>() - selection.lengths[i];
        let available = c.confidence.len().min(max_rows.saturating_sub(others));
        let candidates: Vec<_> = (0..=available).map(|length| {
            let mut lengths = selection.lengths.clone();
            lengths[i] = length;
            let prediction = policy.diagnostic_evaluate(false, &cores, &lengths, width);
            serde_json::json!({"length": length, "expected_tokens": prediction.map(|p| p.0),
                "predicted_us": prediction.map(|p| p.1)})
        }).collect();
        serde_json::json!({"id": c.id, "prior": c.confidence, "calibrated": probabilities,
            "chosen": selection.lengths[i], "cold_request": cold_request, "candidates": candidates})
    }).collect();
    let cost = policy.cost_snapshot(false);
    serde_json::json!({"engaged": policy.engaged(false), "requests": requests,
        "chosen": selection.lengths, "predicted_us": selection.predicted,
        "corrections": policy.corrections(false), "platt": policy.calibration(),
        "position_reached": policy.stats().position_reached,
        "cost": {"layers": cost.layers, "round": cost.round, "draft": cost.draft}})
}

/// Trims the longest drafts one row at a time until every request's anchor
/// and drafts fit `max_rows`; true when it trimmed.
fn fit_budget(lengths: &mut [usize], max_rows: usize) -> bool {
    let mut rows: usize = lengths.len() + lengths.iter().sum::<usize>();
    let trimmed = rows > max_rows;
    while rows > max_rows {
        let Some(longest) = (0..lengths.len()).filter(|&i| lengths[i] > 0).max_by_key(|&i| (lengths[i], usize::MAX - i))
            else { break };
        lengths[longest] -= 1;
        rows -= 1;
    }
    trimmed
}

/// An agreed copy's continuation: the tokens a copy window `copy` proposes
/// past the drafter's whole window `drafted`, when it repeats that window
/// exactly; empty otherwise (a partial or coincidental match drops out).
pub(crate) fn copy_extension<'c>(drafted: &[u32], copy: &'c [u32]) -> &'c [u32] {
    if drafted.is_empty() || copy.len() <= drafted.len() || copy[..drafted.len()] != *drafted { return &[]; }
    &copy[drafted.len()..]
}

/// One verified request of a GLM Flash round.
pub(crate) struct Verified<'a> {
    pub id: u64,
    /// Rows executed (anchor included, after grammar truncation).
    pub rows: usize,
    /// Accepted inputs (anchor included).
    pub accepted: usize,
    pub source: DraftSource,
    /// The prior's `p0` of the request's drafts (neural drafts only).
    pub prior: Option<&'a [f64]>,
    pub features: Option<&'a [[f32; 4]]>,
    /// Verification stopped for a reason other than a miss: the request
    /// finished on a token its next draft also proposed, or was cancelled.
    pub censor: Option<Censor>,
}

/// Feed one completed round to the policy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn observe(policy: &mut DraftPolicy, requests: &[Verified<'_>], routes: &RoundRoutes,
    layer_us: &[Option<f64>], times: RoundTimes, width: usize, predicted: Option<f64>) -> Result<(), ObserveError> {
    let features: Vec<Vec<SelectorFeatures>> = requests.iter().map(|r| r.features.unwrap_or_default().iter()
        .map(|&[margin, p_top, entropy, rank]| SelectorFeatures { margin, p_top, entropy, rank }).collect()).collect();
    let round: Vec<RoundRequest<'_>> = requests.iter().zip(&features).map(|(r, features)| RoundRequest {
        id: r.id, rows: r.rows, accepted: r.accepted, source: r.source,
        evidence: r.prior.map(|prior| if r.features.is_some() {
            Evidence::Selector { features, prior }
        } else {
            Evidence::Head(prior)
        }),
        censor: r.censor,
    }).collect();
    binding::observe(policy, &RoundRecord { shared: false, requests: &round, routes, layer_us, times,
        draft: (width > 0).then_some(DraftPass { width }), predicted })
}

static SNAPSHOT: Mutex<Option<serde_json::Value>> = Mutex::new(None);

/// The latest published policy state for the serving `stats` payload.
pub(crate) fn snapshot() -> serde_json::Value {
    SNAPSHOT.lock().ok().and_then(|slot| slot.clone()).unwrap_or(serde_json::Value::Null)
}

/// Publish the policy's fits and reliability (`/v1/stats` `draft_policy`).
pub(crate) fn publish(policy: &DraftPolicy, observed: u64, skipped: u64, ring: Option<(u64, u64)>) {
    let stats = policy.stats();
    let geometry = policy.geometry();
    let snapshot = policy.cost_snapshot(false);
    let classes: Vec<_> = geometry.classes.iter().zip(&snapshot.layers).map(|(class, c)| serde_json::json!({
        "class": class.label, "layers": geometry.layers.iter().filter(|l| l.class.is_some_and(|k|
            geometry.classes.get(usize::from(k)).is_some_and(|g| g.label == class.label))).count(),
        "intercept_us": c[0], "us_per_row": c[1], "us_per_mb": c[2],
        "effective_gb_per_s": if c[2] > 0. { 1000. / c[2] } else { f64::INFINITY },
        "samples": c[3], "residual_scale_us": c[4],
    })).collect();
    let calibration = policy.calibration();
    let reliability: Vec<_> = (0..stats.position_reached.len()).map(|p| serde_json::json!({
        "position": p + 1, "reached": stats.position_reached[p],
        "mean_confidence": if stats.position_reached[p] > 0 {
            stats.position_confidence[p] / stats.position_reached[p] as f64 } else { 0. },
        "logit_slope": calibration[p].0, "logit_offset": calibration[p].1,
        "accept_rate": if stats.position_reached[p] > 0 {
            stats.position_accepted[p] as f64 / stats.position_reached[p] as f64 } else { 0. },
    })).collect();
    let value = serde_json::json!({
        "mode": "shared",
        "warm": policy.warm(false),
        "engaged": policy.engaged(false),
        "prediction_health": { "timed_rounds": policy.prediction_health(false).0,
            "relative_error": policy.prediction_health(false).1 },
        "corrections": policy.corrections(false).iter().enumerate().filter(|(_, c)| c[2] > 0.)
            .map(|(bucket, c)| serde_json::json!({"requests_up_to": 1usize << bucket, "intercept_us": c[0],
                "us_per_row": c[1], "samples": c[2]})).collect::<Vec<_>>(),
        "rounds": stats.rounds, "observed_rounds": observed, "unobserved_rounds": skipped,
        "selected_rounds": stats.selected_rounds,
        "verified_rows": stats.verified_rows, "verified_drafts": stats.verified_drafts,
        "accepted_drafts": stats.accepted_drafts, "draft_rows_histogram": stats.draft_rows,
        "confidence_reliability": reliability,

        "prediction": {
            "rounds": stats.predicted_rounds,
            "mean_error_us": if stats.predicted_rounds > 0 {
                stats.prediction_error_us / stats.predicted_rounds as f64 } else { 0. },
            "mean_abs_relative_error": if stats.observed_us > 0. {
                stats.prediction_abs_error_us / stats.observed_us } else { 0. },
        },
        "layers": classes,
        "round": { "intercept_us": snapshot.round[0], "us_per_row": snapshot.round[1],
            "us_per_request": snapshot.round[2], "samples": snapshot.round[3] },
        "draft": { "intercept_us": snapshot.draft[0], "us_per_request": snapshot.draft[1],
            "samples": snapshot.draft[4] },
        "route_ring_check": ring.map(|(layers, mismatched)| serde_json::json!({"layers": layers, "mismatched": mismatched})),
    });
    if let Ok(mut slot) = SNAPSHOT.lock() { *slot = Some(value); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn homes(spark: bool) -> Vec<LayerHome> {
        (0..45).map(|layer| match layer {
            0..3 => LayerHome::Dense,
            _ if spark => LayerHome::Remote { slice_bytes: 4_000_000. },
            _ => LayerHome::Local { slice_bytes: 9_000_000. },
        }).collect()
    }

    #[test]
    fn policy_kind_defaults_to_cycle_and_rejects_unknown_values() {
        assert_eq!(PolicyKind::parse(None).unwrap(), PolicyKind::Cycle);
        assert_eq!(PolicyKind::parse(Some("cycle")).unwrap(), PolicyKind::Cycle);
        assert_eq!(PolicyKind::parse(Some("shared")).unwrap(), PolicyKind::Shared);
        assert!(PolicyKind::parse(Some("buckets")).is_err());
    }

    #[test]
    fn geometry_has_dense_layers_without_class_and_one_class_per_home() {
        let g = geometry(&homes(true), 288, 8, 64, 7, 0).unwrap();
        DraftPolicy::new(g.clone(), false).unwrap();
        assert_eq!((g.layers.len(), g.experts, g.topk, g.max_requests, g.widths.clone(), g.regimes),
            (45, 288, 8, 64, vec![7], 1));
        assert!(g.layers[..3].iter().all(|l| l.class.is_none()));
        assert!(g.layers[3..].iter().all(|l| l.class == Some(REMOTE) && l.slice_bytes == 4e6));
        assert!(!g.layers[0].timed && g.layers[1].timed);
        let local = geometry(&homes(false), 288, 8, 128, 8, 0).unwrap();
        assert!(local.layers[3..].iter().all(|l| l.class == Some(LOCAL)));
        // dSpark's eight-token block keeps all eight positions.
        assert_eq!((local.widths.clone(), local.max_positions, local.max_requests), (vec![8], 8, 128));
        assert!(geometry(&homes(true), 288, 8, 64, 0, 0).is_err());
    }

    #[test]
    fn exl3_bytes_follow_each_projections_tier() {
        // Two experts, TP2 (1024 intermediate columns per rank); expert 0's down projection at K4.
        let bits = |expert: usize, projection: usize| Ok(if expert == 0 && projection == 2 { 4 } else { 3 });
        let per = |bits: usize| 4096 * 1024 * bits / 8 + 4096 * 2 + 1024 * 2;
        let expected = (2 * per(3) + per(4) + 3 * per(3)) as f64 / 2.;
        assert_eq!(exl3_bytes(4096, 2048, 2, 2, bits).unwrap(), expected);
        // TP6 of 2048: the widest rank holds three 128-column blocks.
        let tp6 = exl3_bytes(4096, 2048, 1, 6, |_, _| Ok(3)).unwrap();
        assert_eq!(tp6, (3 * (4096 * 384 * 3 / 8 + 4096 * 2 + 384 * 2)) as f64);
        assert!(exl3_bytes(4096, 2048, 1, 1, |_, _| anyhow::bail!("missing")).is_err());
    }

    fn warm_policy(spark: bool) -> DraftPolicy {
        let mut policy = DraftPolicy::new(geometry(&homes(spark), 288, 8, 64, 7, 0).unwrap(), false).unwrap();
        let mut routes = RoundRoutes::new(45, 8);
        for round in 0..200usize {
            let rows = 1 + round % 8;
            routes.begin(rows);
            for layer in 0..45 {
                if layer < 3 { routes.dense(layer).unwrap(); continue; }
                let ids: Vec<u32> = (0..rows * 8).map(|i| ((layer * 31 + round * 7 + i * 13) % 288) as u32).collect();
                routes.push(layer, &ids, u32::MAX).unwrap();
            }
            let layer_us: Vec<Option<f64>> = (0..45).map(|l| (l > 0).then_some(300. + 40. * rows as f64)).collect();
            let total = 4_000. + layer_us.iter().flatten().sum::<f64>() + 3_000.;
            let p = [0.8; 7];
            let request = [Verified { id: 1, rows, accepted: rows.min(3), source: DraftSource::Neural,
                prior: Some(&p[..rows - 1]), features: None, censor: None }];
            observe(&mut policy, &request, &routes, &layer_us, RoundTimes { total_us: total as u64,
                draft_us: Some(3_000) }, 7, None).unwrap();
        }
        assert!(policy.warm(false));
        policy
    }

    #[test]
    fn cold_policy_verifies_every_draft_within_limits_and_cold_dflash_floors_at_five() {
        let mut policy = DraftPolicy::new(geometry(&homes(true), 288, 8, 64, 7, 0).unwrap(), false).unwrap();
        let p = [0.1; 7];
        let candidates = [Candidate { id: 1, prior: &p, limit: 7, selector: true, cold: true },
            Candidate { id: 2, prior: &p, limit: 3, selector: false, cold: false },
            Candidate { id: 3, prior: &[], limit: 7, selector: true, cold: true }];
        let selection = select(&mut policy, &candidates, 7, 64);
        assert_eq!((selection.lengths, selection.predicted), (vec![7, 3, 0], None));
        // Warm fits: a hopeless cold DFlash2 request still verifies five.
        let mut policy = warm_policy(true);
        let cold = [Candidate { id: 9, prior: &[0.01; 7], limit: 7, selector: true, cold: true }];
        assert_eq!(select(&mut policy, &cold, 7, 64).lengths, vec![5]);
        let head = [Candidate { id: 9, prior: &[0.01; 7], limit: 7, selector: false, cold: true }];
        assert_eq!(select(&mut policy, &head, 7, 64).lengths, vec![0]);
    }

    #[test]
    fn decision_trace_enumerates_c1_lengths_without_training_or_changing_selection() {
        let mut policy = warm_policy(true);
        let candidate = [Candidate { id: 4, prior: &[0.6; 7], limit: 6, selector: true, cold: false }];
        let chosen = select(&mut policy, &candidate, 7, 64);
        let before = policy.stats().clone();
        let trace = DecisionTrace::capture(&candidate, 7, 64, &chosen).evaluate(&mut policy);
        assert_eq!(trace["requests"][0]["candidates"].as_array().unwrap().len(), 7);
        assert_eq!(trace["requests"][0]["calibrated"].as_array().unwrap().len(), 6);
        assert_eq!(trace["requests"][0]["chosen"], chosen.lengths[0]);
        let cost = policy.cost_snapshot(false);
        assert_eq!(trace["cost"]["round"], serde_json::json!(cost.round));
        assert_eq!(trace["cost"]["layers"], serde_json::json!(cost.layers));
        assert_eq!(trace["cost"]["draft"], serde_json::json!(cost.draft));
        assert_eq!(*policy.stats(), before);
        let after = select(&mut policy, &candidate, 7, 64);
        assert_eq!((after.lengths, after.predicted), (chosen.lengths, chosen.predicted));
    }

    #[test]
    fn cold_decision_trace_exposes_seed_prices_without_engaging_policy() {
        let mut policy = DraftPolicy::seeded(geometry(&homes(true), 288, 8, 64, 7, 0).unwrap(), seed()).unwrap();
        let candidate = [Candidate { id: 4, prior: &[0.6; 7], limit: 7, selector: true, cold: true }];
        let chosen = select(&mut policy, &candidate, 7, 64);
        assert!(chosen.predicted.is_none());
        let trace = decision_trace(&mut policy, &candidate, 7, 64, &chosen);
        assert_eq!(trace["engaged"], false);
        assert_eq!(trace["requests"][0]["cold_request"], true);
        let prices = trace["requests"][0]["candidates"].as_array().unwrap();
        assert_eq!(prices.len(), 8);
        assert!(prices.iter().all(|p| p["predicted_us"].as_f64().is_some_and(|us| us > 0.)));
        assert!(!policy.engaged(false));
        assert_eq!(policy.stats().rounds, 0);
        assert_eq!(select(&mut policy, &candidate, 7, 64).lengths, chosen.lengths);
    }

    #[test]
    fn lone_warm_dflash_keeps_five_unless_the_selection_is_two_percent_better() {
        let mut policy = warm_policy(true);
        // Confident: the selection runs past five and beats it.
        let sure = [Candidate { id: 4, prior: &[0.99; 7], limit: 7, selector: true, cold: false }];
        let chosen = select(&mut policy, &sure, 7, 64);
        let (tokens, us) = policy.evaluate(false, &[DraftCandidate { id: 4, confidence: &[0.99; 7] }], &[5], 7).unwrap();
        let best = policy.evaluate(false, &[DraftCandidate { id: 4, confidence: &[0.99; 7] }], &chosen.lengths, 7)
            .unwrap();
        if chosen.lengths[0] != 5 { assert!(best.0 / best.1 >= tokens / us * REFERENCE_MARGIN); }
        // Two requests: no five-draft reference.
        let pair = [Candidate { id: 5, prior: &[0.2; 7], limit: 7, selector: true, cold: false },
            Candidate { id: 6, prior: &[0.2; 7], limit: 7, selector: true, cold: false }];
        assert!(select(&mut policy, &pair, 7, 64).lengths.iter().all(|&n| n < 5));
    }

    #[test]
    fn censored_stops_on_accepted_drafts_offer_only_their_accepted_prefix() {
        let mut policy = DraftPolicy::new(geometry(&homes(true), 288, 8, 64, 7, 0).unwrap(), false).unwrap();
        let mut routes = RoundRoutes::new(45, 8);
        routes.begin(9);
        for layer in 0..45 {
            if layer < 3 { routes.dense(layer).unwrap(); } else { routes.push(layer, &[1; 72], u32::MAX).unwrap(); }
        }
        let p = [0.9; 4];
        let features = [[1.0, 0.8, 0.3, 0.0]; 4];
        let requests = [
            // EOS on the third accepted draft: positions 1-3 reached, 4 censored.
            Verified { id: 1, rows: 5, accepted: 4, source: DraftSource::Neural, prior: Some(&p),
                features: Some(&features), censor: Some(Censor::Eos) },
            // Ran to the output limit with every draft accepted.
            Verified { id: 2, rows: 3, accepted: 3, source: DraftSource::Neural, prior: Some(&p[..2]),
                features: None, censor: Some(Censor::OutputLimit) },
            Verified { id: 3, rows: 1, accepted: 1, source: DraftSource::Undrafted, prior: None, features: None,
                censor: None },
        ];
        let layer_us = vec![Some(100.); 45];
        observe(&mut policy, &requests, &routes, &layer_us, RoundTimes { total_us: 9_000, draft_us: Some(2_000) },
            7, None).unwrap();
        // Request 1 reaches three positions (all accepted), request 2 reaches two.
        assert_eq!(policy.stats().position_reached[..4], [2, 2, 1, 0]);
        assert_eq!(policy.stats().position_accepted[..4], [2, 2, 1, 0]);
    }

    #[test]
    fn a_seeded_cold_policy_verifies_its_cold_drafts_within_the_row_budget() {
        let mut policy = DraftPolicy::seeded(geometry(&homes(true), 288, 8, 64, 7, COPY_EXTENSION).unwrap(), seed())
            .unwrap();
        assert!(!policy.engaged(false));
        let p = [0.1; 7];
        let long = [0.9; 20];
        let candidates = [Candidate { id: 1, prior: &p, limit: 7, selector: false, cold: false },
            Candidate { id: 2, prior: &long, limit: 63, selector: false, cold: false },
            // A cold DFlash2 request keeps its five.
            Candidate { id: 3, prior: &p, limit: 7, selector: true, cold: true }];
        let selection = select(&mut policy, &candidates, 7, 64);
        assert_eq!((selection.lengths, selection.predicted), (vec![COLD_DRAFTS, COLD_DRAFTS, 5], None));
        // Eight rows: three anchors and five drafts, trimmed from the longest.
        let tight = select(&mut policy, &candidates, 7, 8);
        assert_eq!(tight.lengths.iter().sum::<usize>() + 3, 8, "{:?}", tight.lengths);
        assert!(tight.lengths.iter().max().unwrap() - tight.lengths.iter().min().unwrap() <= 1);
    }

    #[test]
    fn warm_selection_spends_one_row_budget_where_drafts_pay() {
        let mut policy = warm_policy(true);
        let sure = [0.98; 7];
        let poor = [0.05; 7];
        let candidates = [Candidate { id: 1, prior: &sure, limit: 63, selector: false, cold: false },
            Candidate { id: 2, prior: &poor, limit: 63, selector: false, cold: false },
            Candidate { id: 3, prior: &sure, limit: 63, selector: false, cold: false }];
        let free = select(&mut policy, &candidates, 7, 64);
        assert!(free.lengths[0] > 1 && free.lengths[1] == 0 && free.predicted.is_some(), "{:?}", free.lengths);
        // Five rows: three anchors and two drafts, both to the confident requests.
        let tight = select(&mut policy, &candidates, 7, 5);
        assert_eq!((tight.lengths.iter().sum::<usize>(), tight.lengths[1]), (2, 0), "{:?}", tight.lengths);
        assert!(tight.predicted.is_some());
    }

    #[test]
    fn copies_extend_only_a_draft_they_agree_with_over_its_whole_window() {
        assert_eq!(copy_extension(&[1, 2, 3], &[1, 2, 3, 4, 5]), [4, 5]);
        assert!(copy_extension(&[1, 2, 3], &[1, 2, 9, 4, 5]).is_empty());
        assert!(copy_extension(&[1, 2, 3], &[1, 2, 3]).is_empty());
        assert!(copy_extension(&[1, 2, 3], &[1, 2]).is_empty());
        assert!(copy_extension(&[], &[1, 2]).is_empty());
        // Extension positions sit past the drafter's width in the geometry.
        let g = geometry(&homes(true), 288, 8, 64, 7, COPY_EXTENSION).unwrap();
        assert_eq!((g.widths.clone(), g.max_positions), (vec![7], 7 + COPY_EXTENSION));
    }
}
