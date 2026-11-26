//! Serving glue for the `history` draft policy (`cuteafd_core::DraftHistory`):
//! the launch configuration, the per-round trace record and the `/v1/stats`
//! snapshot. Every speculating family drives the same five calls; no family
//! holds policy code.
//!
//! A serve loop's round:
//! 1. [`Policy::plan`] before the draft pass: drafts to ask for per request;
//! 2. [`Policy::shape`] after it: each proposal cut to its keep prefix and one
//!    explore, extended by an agreeing history copy, and
//!    fitted to the verify-row budget;
//! 3. [`Policy::observe`] after verification, with what the target accepted.
use super::copy;
use super::length::DraftLength;
use crate::shared::speculation_trace::Trace;
use cuteafd_core::{DraftControl, DraftHistory, DraftOutcome, DrafterSpec, HeadScore};
use std::sync::{Arc, Mutex, OnceLock};

/// One active request after the draft pass, in verifier order.
pub(crate) struct Proposal<'a> {
    pub id: u64,
    /// The drafter's tokens (empty: not drafted). Cut and extended in place.
    pub tokens: &'a mut Vec<u32>,
    /// Per-token score, in the drafter's declared score format.
    pub head: Option<&'a [f32]>,
    /// Prompt and generated tokens with the round's anchor last, for a history copy.
    pub history: &'a [u32],
    /// Copies continue greedy requests only.
    pub greedy: bool,
}

pub(crate) struct Policy {
    history: DraftHistory,
    trace: Option<&'static Trace>,
    rounds: u64,
    /// Each live request's committed context after its last observed round,
    /// so a round's band is known when it is planned (`clamped`).
    contexts: std::collections::HashMap<u64, usize>,
}

impl Policy {
    /// `block` is the drafter's native block; `head` its score source;
    /// `copies` whether the family verifies history copies.
    pub fn new(length: DraftLength, block: usize, head: HeadScore, copies: bool) -> Self {
        let spec = DrafterSpec { block, head, copy: if copies { copy::COPY_EXTENSION } else { 0 } };
        let history = DraftHistory::new(spec, length.control()).with_cut_rule(length.cut_rule());
        SPECULATION_ENABLED.store(block > 0 && history.control() != DraftControl::Off,
            std::sync::atomic::Ordering::Relaxed);
        cuteafd_bench::context::set_resolved("draft-mode", &length.label(block));
        tracing::info!(policy = %history.control().label(), block, head = ?head, copy_positions = spec.copy,
            requested_length = ?length.draft_length, "draft policy");
        let policy = Self { history, trace: replay_trace(), rounds: 0, contexts: Default::default() };
        policy.publish();
        policy
    }

    pub fn block(&self) -> usize { self.history.spec().block }
    /// History estimate before a neural confidence score exists.
    pub fn probability(&self, id: u64, position: usize) -> f64 { self.history.probability(id, position) }
    /// The most drafted positions one request verifies in a round.
    pub fn max_positions(&self) -> usize { self.history.max_positions() }
    pub fn fixed(&self) -> bool { matches!(self.history.control(), DraftControl::Fixed(_)) }

    /// Drafts to ask the drafter for, per request. `requests` are (identity,
    /// room) in verifier order; `rows` is the round's verify-row budget.
    pub fn plan(&mut self, requests: &[(u64, usize)], rows: usize) -> Vec<usize> {
        let known: Vec<usize> = requests.iter().filter_map(|(id, _)| self.contexts.get(id).copied()).collect();
        let context = if known.is_empty() { 0 } else { known.iter().sum::<usize>() / known.len() };
        self.history.begin_round(context, requests.len());
        self.history.plan(requests, rows)
    }

    /// Fit every proposal to its plan under `rows` verify rows. `proposals`
    /// cover every active request in verifier order. Returns how many of each
    /// request's tokens are the drafter's own; the rest are a history copy.
    pub fn shape(&mut self, proposals: &mut [Proposal<'_>], rows: usize) -> Vec<usize> {
        self.shape_with_padding(proposals, rows, |_| None)
    }

    pub fn shape_with_padding(&mut self, proposals: &mut [Proposal<'_>], rows: usize,
        bucket: impl Fn(usize) -> Option<usize>) -> Vec<usize> {
        let mut neural = Vec::with_capacity(proposals.len());
        for proposal in proposals.iter_mut() {
            let kept = self.history.refine(proposal.id, proposal.tokens.len(), proposal.head);
            proposal.tokens.truncate(kept);
            if kept > 0 && self.history.spec().copy > 0 {
                if let Some(tail) = copy::continuation(proposal.history, proposal.tokens, proposal.greedy) {
                    let extension = self.history.extend_copy(proposal.id, tail.len());
                    proposal.tokens.extend_from_slice(&tail[..extension]);
                }
            }
            neural.push(kept);
        }
        let ids: Vec<u64> = proposals.iter().map(|proposal| proposal.id).collect();
        for ((proposal, length), neural) in proposals.iter_mut().zip(self.history.settle_with_padding(&ids, rows, bucket)).zip(&mut neural) {
            proposal.tokens.truncate(length);
            *neural = (*neural).min(length);
        }
        neural
    }

    /// Learn from one verified round and record it.
    pub fn observe(&mut self, outcomes: &[DraftOutcome]) { self.observe_timed(outcomes, None); }

    /// [`Self::observe`] with the round's measured host timing, which the
    /// trace record carries alongside the policy's view (no extra timers).
    pub fn observe_timed(&mut self, outcomes: &[DraftOutcome], timing: Option<RoundTiming>) {
        if let Some(timing) = &timing {
            let rows: usize = outcomes.iter().map(|o| o.proposed + 1).sum();
            let context = if timing.context.is_empty() { 0 } else { timing.context.iter().sum::<usize>() / timing.context.len() };
            self.history.observe_cost(context, rows, timing.round_us as f64 / 1e6);
            for (outcome, &start) in outcomes.iter().zip(&timing.context) {
                self.contexts.insert(outcome.id, start + outcome.accepted + 1);
            }
        }
        let record = self.history.observe(outcomes, self.trace.is_some());
        let timing = timing.filter(|_| self.trace.is_some());
        if let (Some(trace), Some(mut record)) = (self.trace, record) {
            attach_request_ids(&mut record);
            if let Err(error) = trace.write(RoundRecord { r#type: "draft_round", round: self.rounds, timing,
                draft: record }) {
                tracing::warn!(%error, "draft round trace failed");
            }
        }
        self.rounds += 1;
        if self.rounds % PUBLISH_ROUNDS == 0 { self.publish(); }
    }

    pub fn release(&mut self, id: u64) { self.history.release(id); self.contexts.remove(&id); }

    fn publish(&self) {
        if let Ok(mut slot) = SNAPSHOT.lock() { *slot = Some(Arc::new(self.history.snapshot())); }
    }
}

/// Which drafter score readbacks a serve loop turns on: per-token scores whenever
/// the policy keys on them (`history`) or a speculation trace records them (a
/// fixed `DRAFT_LENGTH` trace carries the same scores), and the selector feature
/// vector only for the trace. Returns `(traced, scored)` for `set_selector_readback`.
pub(crate) fn score_readback(control: DraftControl) -> (bool, bool) {
    readback_for(control, replay_trace().is_some())
}

fn readback_for(control: DraftControl, traced: bool) -> (bool, bool) {
    (traced, control == DraftControl::History || traced)
}

static SPECULATION_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn speculation_enabled() -> bool {
    SPECULATION_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

// Numeric tickets stay engine-local; HTTP ids join response usage to this trace.
static REQUEST_IDS: Mutex<Option<std::collections::HashMap<u64, String>>> = Mutex::new(None);

pub(crate) fn register_request(id: u64, request_id: &str) {
    if replay_trace().is_some() {
        if let Ok(mut ids) = REQUEST_IDS.lock() {
            ids.get_or_insert_with(Default::default).insert(id, request_id.to_owned());
        }
    }
}

pub(crate) fn unregister_request(id: u64) {
    if let Ok(mut ids) = REQUEST_IDS.lock() {
        if let Some(ids) = ids.as_mut() { ids.remove(&id); }
    }
}

fn attach_request_ids(record: &mut cuteafd_core::RoundTrace) {
    if let Ok(ids) = REQUEST_IDS.lock() {
        if let Some(ids) = ids.as_ref() {
            for request in &mut record.requests { request.request_id = ids.get(&request.id).cloned(); }
        }
    }
}

/// One `SPECULATION_TRACE` line per observed round: the policy, the running
/// short-round share and explore counts, and per request every planned position
/// with its per-token score, history counts, exploration and target outcome.
#[derive(serde::Serialize)]
struct RoundRecord {
    r#type: &'static str,
    round: u64,
    #[serde(flatten)]
    timing: Option<RoundTiming>,
    #[serde(flatten)]
    draft: cuteafd_core::RoundTrace,
}

/// A verify round's host-clock times, as the serve loop already measures
/// them for its console round: `round_us` from the round's start to the
/// policy observe, `draft_us` the draft pass and `verify_us` the verify step.
/// `context` is each request's committed length before the round, in the
/// same order as the observed outcomes.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub(crate) struct RoundTiming {
    pub lane: usize,
    pub round_us: u64,
    pub draft_us: u64,
    pub verify_us: u64,
    pub context: Vec<usize>,
}

impl RoundTiming {
    /// Times from seconds and a round start, as the serve loops hold them.
    pub fn since(lane: usize, started: std::time::Instant, draft_s: f64, verify_s: f64,
        context: Vec<usize>) -> Self {
        let us = |s: f64| (s.max(0.) * 1e6) as u64;
        Self { lane, round_us: started.elapsed().as_micros() as u64, draft_us: us(draft_s), verify_us: us(verify_s),
            context }
    }
}

/// The stats view may lag the policy by this many rounds.
const PUBLISH_ROUNDS: u64 = 32;
static SNAPSHOT: Mutex<Option<Arc<serde_json::Value>>> = Mutex::new(None);

/// The latest published policy state for the serving `stats` payload.
pub(crate) fn snapshot() -> serde_json::Value {
    trace_snapshot().0.as_deref().cloned().unwrap_or(serde_json::Value::Null)
}

/// Cheap owned view for asynchronous trace serialization.
pub(crate) struct TraceSnapshot(Option<Arc<serde_json::Value>>);
impl serde::Serialize for TraceSnapshot {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.as_deref().serialize(serializer)
    }
}
pub(crate) fn trace_snapshot() -> TraceSnapshot {
    TraceSnapshot(SNAPSHOT.lock().ok().and_then(|slot| slot.clone()))
}

/// The `SPECULATION_TRACE` file, shared with each family's own round records.
pub(crate) fn replay_trace() -> Option<&'static Trace> {
    static TRACE: OnceLock<Option<Trace>> = OnceLock::new();
    TRACE.get_or_init(|| match Trace::open() {
        Ok(trace) => trace,
        Err(error) => { tracing::warn!(%error, "draft round trace open failed"); None }
    }).as_ref()
}

/// One request's verified round. `rows` is the verifier rows executed and
/// `accepted` the accepted inputs, both with the anchor; `censored` when
/// verification stopped for a reason other than a miss.
pub(crate) fn outcome(id: u64, rows: usize, accepted: usize, censored: bool) -> DraftOutcome {
    DraftOutcome { id, proposed: rows.saturating_sub(1), accepted: accepted.saturating_sub(1), censored }
}

/// Whether a finished request stopped without the target judging its next
/// draft: it ended on its last verified row, or on a token the next draft
/// also proposed. A terminal token that disagrees with the draft is a miss.
pub(crate) fn finish_censored(finished: bool, committed: usize, rows: &[u32], last: Option<u32>) -> bool {
    finished && !rows.get(committed).is_some_and(|next| Some(*next) != last)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn length(fixed: Option<usize>) -> DraftLength { DraftLength { draft_length: fixed, ..Default::default() } }

    /// History whose last eight tokens repeat an earlier stretch followed by 100, 101, ...
    fn repeating(suffix: usize) -> Vec<u32> {
        let mut history: Vec<u32> = (0..8).collect();
        history.extend(100..140);
        history.extend(50..60);
        history.extend((8 - suffix) as u32..8);
        history
    }

    #[test]
    fn shape_cuts_to_the_plan_extends_agreeing_copies_and_fits_the_budget() {
        let mut policy = Policy::new(length(None), 4, HeadScore::None, true);
        let history = repeating(8);
        assert_eq!(policy.plan(&[(1, 40), (2, 40)], 64), [4, 4]);
        let (mut agreeing, mut differing) = (vec![100, 101, 102, 103], vec![100, 101, 7, 103]);
        let neural = policy.shape(&mut [
            Proposal { id: 1, tokens: &mut agreeing, head: None, history: &history, greedy: true },
            Proposal { id: 2, tokens: &mut differing, head: None, history: &history, greedy: true }], 64);
        // A cold copy track verifies one explored position.
        assert_eq!((neural, agreeing.as_slice(), differing.len()), (vec![4, 4], &[100, 101, 102, 103, 104][..], 4));
        policy.observe(&[outcome(1, 6, 6, false), outcome(2, 5, 3, false)]);
        // The .3 stop keeps both native blocks after one round. Sampled requests
        // never copy; the later budget still prioritizes the agreeing request.
        assert_eq!(policy.plan(&[(1, 40), (2, 40)], 64), [4, 4]);
        let (mut first, mut second) = (vec![100, 101, 102, 103], vec![100, 101, 7]);
        let neural = policy.shape(&mut [
            Proposal { id: 1, tokens: &mut first, head: None, history: &history, greedy: false },
            Proposal { id: 2, tokens: &mut second, head: None, history: &history, greedy: true }], 7);
        assert_eq!((neural, first.len(), second.len()), (vec![3, 2], 3, 2));
    }

    #[test]
    fn fixed_length_and_off_come_from_the_launch_configuration() {
        let (mut fixed, settings) = cuteafd_bench::context::capture_resolved(||
            Policy::new(length(Some(9)), 7, HeadScore::Logit, true));
        assert!(fixed.fixed() && fixed.max_positions() == 7);
        assert!(settings.0.iter().any(|(name, value)| name == "draft-mode" && value == "fixed 7"));
        assert_eq!(fixed.plan(&[(1, 40)], 64), [7]);
        let mut off = Policy::new(DraftLength { draft_policy: super::super::length::DraftPolicyChoice::Off,
            ..Default::default() }, 7, HeadScore::None, true);
        assert_eq!((off.plan(&[(1, 40)], 64), off.max_positions()), (vec![0], 0));
    }

    #[test]
    fn the_round_record_is_flat_and_carries_positions_and_the_short_share() {
        let mut core = DraftHistory::new(DrafterSpec { block: 3, head: HeadScore::Probability, copy: 0 },
            cuteafd_core::DraftControl::History);
        core.plan(&[(7, 3)], 64);
        core.refine(7, 3, Some(&[0.9, 0.8, 0.1]));
        let draft = core.observe(&[outcome(7, 4, 2, false)], true).unwrap();
        let value = serde_json::to_value(RoundRecord { r#type: "draft_round", round: 0, timing: None, draft }).unwrap();
        assert_eq!((value["type"].as_str(), value["policy"].as_str(), value["block"].as_u64()),
            (Some("draft_round"), Some("history"), Some(3)));
        assert_eq!((value["rounds"].as_u64(), value["short_rounds"].as_u64(), value["short_share"].as_f64()),
            (Some(1), Some(0), Some(0.)));
        let request = &value["requests"][0];
        assert_eq!((request["id"].as_u64(), request["proposed"].as_u64(), request["accepted_drafts"].as_u64()),
            (Some(7), Some(3), Some(1)));
        let position = &request["positions"][1];
        for field in ["p", "score", "accepted", "reached", "server", "server_reached", "explore", "head", "calibrated", "outcome"] {
            assert!(position.get(field).is_some(), "{field}");
        }
        assert_eq!((position["head"].as_f64().map(|x| (x * 10.).round()), position["outcome"].as_bool()), (Some(8.), Some(false)));
    }

    #[test]
    fn round_requests_join_response_usage_by_http_id_and_retirement_removes_it() {
        let id = u64::MAX - 2443;
        REQUEST_IDS.lock().unwrap().get_or_insert_with(Default::default)
            .insert(id, "http-history-request".to_owned());
        let mut core = DraftHistory::new(DrafterSpec { block: 3, head: HeadScore::None, copy: 0 },
            DraftControl::History);
        core.plan(&[(id, 3)], 64);
        let mut record = core.observe(&[outcome(id, 4, 2, false)], true).unwrap();
        attach_request_ids(&mut record);
        let value = serde_json::to_value(RoundRecord { r#type: "draft_round", round: 0, timing: None,
            draft: record.clone() }).unwrap();
        assert_eq!(value["requests"][0]["id"], id);
        assert_eq!(value["requests"][0]["request_id"], "http-history-request");
        unregister_request(id);
        attach_request_ids(&mut record);
        assert!(record.requests[0].request_id.is_none());
    }

    #[test]
    fn a_timed_round_record_carries_lane_times_and_context_and_an_untimed_one_none() {
        let mut core = DraftHistory::new(DrafterSpec { block: 3, head: HeadScore::None, copy: 0 },
            cuteafd_core::DraftControl::Fixed(3));
        core.plan(&[(7, 3), (8, 3)], 64);
        let draft = core.observe(&[outcome(7, 4, 2, false), outcome(8, 4, 4, false)], true).unwrap();
        let timing = RoundTiming { lane: 1, round_us: 9000, draft_us: 1500, verify_us: 7000, context: vec![4096, 131072] };
        let value = serde_json::to_value(RoundRecord { r#type: "draft_round", round: 3, timing: Some(timing),
            draft: draft.clone() }).unwrap();
        assert_eq!((value["lane"].as_u64(), value["round_us"].as_u64(), value["draft_us"].as_u64(),
            value["verify_us"].as_u64()), (Some(1), Some(9000), Some(1500), Some(7000)));
        assert_eq!(value["context"], serde_json::json!([4096, 131072]));
        assert_eq!(value["requests"].as_array().map(Vec::len), Some(2));
        let untimed = serde_json::to_value(RoundRecord { r#type: "draft_round", round: 3, timing: None, draft }).unwrap();
        for field in ["lane", "round_us", "draft_us", "verify_us", "context"] { assert!(untimed.get(field).is_none(), "{field}"); }
        let since = RoundTiming::since(0, std::time::Instant::now(), 0.0015, -1., vec![7]);
        assert_eq!((since.draft_us, since.verify_us, since.context), (1500, 0, vec![7]));
    }

    #[test]
    fn clamped_policy_learns_round_cost_and_plans_each_round_at_its_band() {
        use super::super::length::DraftPolicyChoice;
        let length = DraftLength { draft_policy: DraftPolicyChoice::Clamped, ..Default::default() };
        let mut policy = Policy::new(length, 4, HeadScore::Probability, false);
        assert_eq!(policy.history.cut_rule(), cuteafd_core::CutRule::Clamped);
        // Rounds at 40K context: 10 ms + 2 ms/row, eight requests at 1 or 4 drafts each.
        for round in 0..64 {
            let drafts = if round % 2 == 0 { 0 } else { 3 };
            let ids: Vec<(u64, usize)> = (0..8).map(|id| (id, 4)).collect();
            policy.plan(&ids, 64);
            let outcomes: Vec<_> = (0..8).map(|id| outcome(id, drafts + 1, 1, false)).collect();
            let rows = 8 * (drafts + 1);
            let timing = RoundTiming { lane: 0, round_us: (10_000 + 2_000 * rows) as u64, draft_us: 0,
                verify_us: 0, context: vec![40_000 + round; 8] };
            policy.observe_timed(&outcomes, Some(timing));
        }
        // Each request's next context is its committed length plus what it accepted.
        assert_eq!(policy.contexts[&0], 40_000 + 63 + 1);
        // N = 10 ms, e = 2 ms/row: at C8 x = 0.625, 3.4 / 1.625 clamps to HI.
        policy.plan(&(0..8).map(|id| (id, 4)).collect::<Vec<_>>(), 64);
        assert_eq!(policy.history.round_cut(), cuteafd_core::clamped::HI);
        // A short-context round is in another, still-cold band: the history cut.
        policy.contexts.clear();
        policy.plan(&[(99, 4)], 64);
        assert_eq!(policy.history.round_cut(), cuteafd_core::KEEP_CUT);
        policy.release(0);
        assert!(!policy.contexts.contains_key(&0));
    }

    #[test]
    fn every_family_feeds_its_round_time_to_the_policy() {
        for (family, source, call) in [
            ("shared scheduler", include_str!("../qwen/scheduler.rs"), "round_us: cycle.elapsed().as_micros() as u64"),
            ("GLM 5.3", include_str!("../../families/glm5/serve.rs"), "policy.observe_timed(&outcomes, Some(history::RoundTiming::since("),
            ("GLM Flash", include_str!("../../families/glm5_flash/serve.rs"), "policy.observe_timed(&outcomes, Some(history::RoundTiming::since("),
            ("MiMo", include_str!("../../families/mimo_v2/serve.rs"), "policy.observe_timed(&outcomes, Some(history::RoundTiming::since("),
            ("DeepSeek V4", include_str!("../../families/deepseek_v4/serve.rs"), "round_us: round_started.elapsed().as_micros() as u64"),
            ("V4.1", include_str!("../../families/deepseek_v41/v41_native_serve/scheduler.rs"), "draft.observe_round(&requests, Some(timing()));"),
        ] {
            assert!(source.contains(call), "{family}");
            assert!(!source.contains("policy.observe(&outcomes)") && !source.contains("shared.observe(&outcomes)"), "{family} untimed observe");
        }
    }

    #[test]
    fn scores_are_read_back_for_the_history_policy_or_any_trace() {
        // Untraced: exactly as before, scores only when `history` keys on them.
        assert_eq!(readback_for(DraftControl::History, false), (false, true));
        assert_eq!(readback_for(DraftControl::Fixed(7), false), (false, false));
        assert_eq!(readback_for(DraftControl::Off, false), (false, false));
        // A trace records the scores under every control, fixed-length runs included.
        assert_eq!(readback_for(DraftControl::History, true), (true, true));
        assert_eq!(readback_for(DraftControl::Fixed(7), true), (true, true));
        // Every family configures readback through this one helper.
        for source in [include_str!("../qwen/scheduler.rs"), include_str!("../../families/glm5/serve.rs"),
            include_str!("../../families/glm5_flash/serve.rs"), include_str!("../../families/mimo_v2/serve.rs")] {
            assert!(source.contains("history::score_readback("));
            assert!(!source.contains("set_selector_readback(trace.is_some(), policy.length.control() == "));
        }
        assert!(!include_str!("../qwen/scheduler.rs").contains("configure_draft_scores(length.control() =="));
    }

    #[test]
    fn outcomes_drop_the_anchor_and_terminal_agreement_is_censored() {
        assert_eq!(outcome(3, 6, 3, false), DraftOutcome { id: 3, proposed: 5, accepted: 2, censored: false });
        assert_eq!(outcome(3, 1, 1, true), DraftOutcome { id: 3, proposed: 0, accepted: 0, censored: true });
        let rows = [10, 11, 12, 13];
        assert!(!finish_censored(false, 2, &rows, Some(12)));
        assert!(!finish_censored(true, 2, &rows, Some(99)));
        assert!(finish_censored(true, 2, &rows, Some(12)));
        assert!(finish_censored(true, 4, &rows, Some(99)));
    }
}
