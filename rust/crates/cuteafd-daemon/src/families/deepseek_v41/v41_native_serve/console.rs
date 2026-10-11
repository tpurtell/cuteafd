//! V4.1's feed into the shared live console (`crate::shared::console`).
//!
//! V4.1 keeps its own hook sites (two decode lanes, dSpark rounds, CUDA-event
//! layer timings, encoder prefill chunks); this module turns its round shapes
//! into the shared schema: the [`Tally`] of each lane round becomes a generic
//! round whose stages the V4.1 [`layout`] declares.
use crate::families::deepseek_v41::v41_backbone_lane::FfnSplit;
use crate::shared::console::{Color, Layers, Layout, Speculator, StepGroup};
pub(crate) use crate::shared::console::{install, live, totals, Event, Gauges, Kv, Live, Prefill,
    PrefillKind, Round, RoundRequest, Ticket};
use serde_json::{json, Value};
use std::time::Instant;

/// Layers whose FFN finish times the policy's CUDA events report.
const LAYERS: usize = 40;

/// The per-round counts every path needs, captured before emission consumes
/// the token vectors. Updates the lifetime totals.
pub(crate) struct Tally {
    drafted: [u8; 8],
    verified: [u8; 8],
    accepted: [u8; 8],
    emitted: [u8; 8],
    proposal: Vec<Vec<u32>>,
    emissions: Vec<Vec<u32>>,
}

/// The proposal of each member before truncation, captured right after drafting.
pub(crate) struct Proposal { drafted: [u8; 8], tokens: Vec<Vec<u32>> }
impl Proposal {
    #[inline]
    pub fn capture(inputs: &[Vec<u32>], live: Option<Live>) -> Self {
        let mut drafted = [0u8; 8];
        for (slot, input) in drafted.iter_mut().zip(inputs) { *slot = input.len().saturating_sub(1) as u8; }
        let tokens = if live.is_some_and(Live::text) {
            inputs.iter().map(|input| input.get(1..).unwrap_or_default().to_vec()).collect()
        } else { Vec::new() };
        Self { drafted, tokens }
    }
}

impl Tally {
    pub fn new(proposal: Proposal, inputs: &[Vec<u32>], accepted_inputs: &[u32],
        emissions: &[Vec<u32>], live: Option<Live>) -> Self {
        let mut tally = Self { drafted: proposal.drafted, verified: [0; 8], accepted: [0; 8], emitted: [0; 8],
            proposal: proposal.tokens, emissions: Vec::new() };
        for (index, input) in inputs.iter().enumerate().take(8) {
            tally.verified[index] = input.len().saturating_sub(1) as u8;
            tally.accepted[index] = accepted_inputs.get(index).map_or(0, |&a| a.saturating_sub(1)) as u8;
            tally.emitted[index] = emissions.get(index).map_or(0, Vec::len) as u8;
        }
        let sum = |values: &[u8; 8]| values.iter().map(|&v| u64::from(v)).sum::<u64>();
        totals::round(sum(&tally.drafted), sum(&tally.verified), sum(&tally.accepted), sum(&tally.emitted));
        if live.is_some_and(Live::text) { tally.emissions = emissions.to_vec(); }
        tally
    }
    pub fn usage<'a>(&self, handles: impl Iterator<Item = Option<&'a cuteafd_api::usage::UsageHandle>>) {
        for (index, handle) in handles.enumerate().take(8) {
            if let Some(handle) = handle { handle.round(self.drafted[index].into(), self.accepted[index].into()); }
        }
    }
    /// Build the round event; `members` gives each member's id, grammar mask
    /// state and whether it finished this round.
    #[allow(clippy::too_many_arguments)]
    pub fn round(mut self, lane: usize, shared: bool, started: Instant, draft_us: u64, prepare_us: u64,
        verify_us: u64, layer_us: &[Option<f64>], ffn: FfnSplit,
        members: impl Iterator<Item = (u64, bool, bool)>) -> Event {
        let requests = members.enumerate().map(|(index, (id, masked, finished))| RoundRequest {
            id, masked, finished,
            drafted: self.drafted[index], verified: self.verified[index],
            accepted: self.accepted[index], emitted: self.emitted[index],
            proposal: self.proposal.get_mut(index).map(std::mem::take).unwrap_or_default(),
            emissions: self.emissions.get_mut(index).map(std::mem::take).unwrap_or_default(),
        }).collect();
        let finished = Instant::now();
        let cycle_us = finished.saturating_duration_since(started).as_micros() as f64;
        let (draft, prepare, verify) = (draft_us as f64, prepare_us as f64, verify_us as f64);
        Event::Round(Round {
            lane: lane as u8, shared, started, finished,
            stages: stages(draft, prepare, verify, (cycle_us - draft - prepare - verify).max(0.0), &ffn),
            layer_us: layer_us.iter().map(|v| v.map_or(f32::NAN, |v| v as f32)).collect(),
            requests,
        })
    }
}

/// A lane round's stage times by the keys [`layout`] declares; FFN stages are
/// means per layer of the resource class that ran any.
fn stages(draft: f64, prepare: f64, verify: f64, commit: f64, ffn: &FfnSplit) -> Vec<(&'static str, f32)> {
    let mut stages = vec![("draft", draft as f32), ("prepare", prepare as f32), ("verify", verify as f32),
        ("commit", commit as f32)];
    if ffn.local_layers > 0 {
        let per = |us: u64| (us as f64 / f64::from(ffn.local_layers)) as f32;
        stages.extend([("lroute", per(ffn.local_routed_us)), ("ltotal", per(ffn.local_total_us))]);
    }
    if ffn.remote_layers > 0 {
        let per = |us: u64| (us as f64 / f64::from(ffn.remote_layers)) as f32;
        stages.extend([("rroute", per(ffn.remote_routed_us)), ("rdispatch", per(ffn.remote_dispatch_us)),
            ("rshared", per(ffn.remote_shared_us)), ("rcollect", per(ffn.remote_collect_us))]);
    }
    stages
}

/// What the V4.1 console shows. The revision comes from the release image's
/// environment (or the binary's build); `CUTEAFD_CONSOLE_REVISION` overrides it.
pub(crate) fn layout(args: &crate::cli::NativeServeArgs) -> Layout {
    let mut layout = Layout::new("deepseek_v41", cuteafd_api::openai::MODEL.into(), args.snapshot.clone());
    layout.hardware = crate::shared::console::hardware(args.rtx_gpus as usize, args.peers.len(), false);
    layout.split = (args.rtx_gpus > 1).then(|| "layer range · 2 lanes".to_string());
    layout.lanes = 2;
    layout.concurrency = args.concurrency as usize;
    layout.eos = vec![1];
    layout.speculator = args.dspark.then(|| Speculator {
        name: "dSpark".into(),
        positions: usize::from(args.dspark_draft_limit).clamp(5, 7),
        policy: format!("{} · limit {}", if args.dspark_fixed { "fixed" } else { "bandwidth" }, args.dspark_draft_limit),
    });
    layout.steps = vec![
        StepGroup::new("Lane round", "host clock", &[
            ("round.cycle0", "lane 0 cycle", Color::Target), ("round.cycle1", "lane 1 cycle", Color::Target),
            ("draft", "dSpark draft + length select", Color::Accepted), ("prepare", "prepare rows", Color::Ink),
            ("verify", "verification pass + head", Color::Target), ("commit", "commit + emit", Color::Ink)]),
        StepGroup::new("Verification pass", "CUDA events, FFN finish to FFN finish", &[
            ("layers.sum", "layers 1–39 on device", Color::Target),
            ("layers.mean.0", "RTX-expert layer (mean)", Color::Rtx),
            ("layers.mean.1", "Spark-expert layer (mean)", Color::Spark),
            ("layers.max", "slowest layer", Color::Warn)]),
        StepGroup::new("Expert stages per layer", "host clock, mean per layer", &[
            ("lroute", "RTX: router + route ids", Color::Rtx), ("ltotal", "RTX: routed + shared experts", Color::Rtx),
            ("rroute", "Spark: router + route ids", Color::Spark), ("rdispatch", "Spark: dispatch to ranks", Color::Spark),
            ("rshared", "Spark: shared experts on RTX", Color::Rtx),
            ("rcollect", "Spark: wait replies + reduce", Color::Spark)]),
        StepGroup::admission(true),
    ];
    let local = if args.rtx_gpus > 1 { LAYERS / 2 } else { 0 };
    layout.layers = Some(Layers {
        title: "Layer profile · last round · CUDA events".into(),
        first: 1,
        classes: vec![("RTX-resident experts".into(), Color::Rtx), ("Spark-routed experts".into(), Color::Spark)],
        class: layer_classes(local),
    });
    layout.extra = json!({
        "prefill_chunk_rows": args.prefill_batch_tokens,
        "max_context_tokens": args.max_context_tokens,
        "prefix_cache_entries": args.prefix_cache_entries,
        "lane_capacity": 8,
    });
    let dspark = args.dspark;
    layout.dynamic = Some(Box::new(move || dynamic(&super::speculative::policy_snapshot(), dspark)));
    layout
}

fn layer_classes(local: usize) -> Vec<u8> {
    (0..LAYERS).map(|layer| u8::from(layer >= local)).collect()
}

/// The dSpark policy's facts and the RTX/Spark layer split it runs on.
fn dynamic(policy: &Value, dspark: bool) -> Value {
    let mode = policy.get("mode").and_then(Value::as_str).unwrap_or(if dspark { "…" } else { "off" });
    let warm = |key: &str| policy.get(key).and_then(|fit| fit.get("warm")).and_then(Value::as_bool);
    let fit = match (warm("solo"), warm("shared")) {
        (Some(solo), shared) => format!("fit: solo {} · shared {}", if solo { "warm" } else { "cold" },
            if shared == Some(true) { "warm" } else { "cold" }),
        _ => "fit: –".into(),
    };
    let error = policy.get("prediction").and_then(|p| p.get("mean_abs_relative_error")).and_then(Value::as_f64)
        .map_or_else(|| "–".into(), |e| format!("{:.1}%", 100.0 * e));
    let mut value = json!({"facts": [["Policy", mode, fit], ["Cost model error", error, "mean abs, per round"]]});
    if let Some(local) = policy.get("local_layers").and_then(Value::as_u64) {
        value["layer_class"] = json!(layer_classes(local as usize));
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_derives_counts_from_round_shapes() {
        // Two requests: 5 drafted, 4 verified, 2 accepted; and an anchor-only row.
        let inputs = vec![vec![10, 11, 12, 13, 14], vec![20]];
        let proposal = Proposal { drafted: [5, 0, 0, 0, 0, 0, 0, 0], tokens: Vec::new() };
        let tally = Tally::new(proposal, &inputs, &[3, 1], &[vec![11, 12, 99], vec![21]], None);
        assert_eq!(&tally.verified[..2], &[4, 0]);
        assert_eq!(&tally.accepted[..2], &[2, 0]);
        assert_eq!(&tally.emitted[..2], &[3, 1]);
        let ffn = FfnSplit { remote_layers: 2, remote_collect_us: 10, ..FfnSplit::default() };
        let Event::Round(round) = tally.round(1, true, Instant::now(), 1, 2, 3, &[None, Some(5.0)],
            ffn, [(7, false, false), (8, true, true)].into_iter()) else { panic!() };
        assert_eq!(round.requests.len(), 2);
        assert_eq!((round.requests[0].id, round.requests[0].drafted, round.requests[0].accepted), (7, 5, 2));
        assert!(round.requests[1].masked && round.requests[1].finished);
        assert!(round.layer_us[0].is_nan() && round.layer_us[1] == 5.0);
        let stage = |key: &str| round.stages.iter().find(|(k, _)| *k == key).map(|&(_, v)| v);
        assert_eq!((stage("draft"), stage("rcollect"), stage("lroute")), (Some(1.0), Some(5.0), None));
    }

    #[test]
    fn policy_facts_and_layer_split() {
        let value = dynamic(&json!({"mode": "bandwidth", "local_layers": 20, "solo": {"warm": true},
            "prediction": {"mean_abs_relative_error": 0.14}}), true);
        assert_eq!(value["facts"][0], json!(["Policy", "bandwidth", "fit: solo warm · shared cold"]));
        assert_eq!(value["facts"][1][1], "14.0%");
        assert_eq!(value["layer_class"][19], 0);
        assert_eq!(value["layer_class"][20], 1);
        assert_eq!(dynamic(&Value::Null, false)["facts"][0][1], "off");
    }
}
