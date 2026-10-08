//! The mandatory baseline: the basic card (C1 decode on code, prose and JSON
//! with thinking off; C8 code aggregate tok/s; 8K prefill rate and TTFT) and quick quality (logit
//! fidelity against eight sealed published windows, prefix-cache restore exactness,
//! lossless speculation, chat-template round trip, C1 vs C4 divergence).
//! Budgeted to about three minutes so Release smoke fits five with the load.
//! The warmed C8 batch adds about 20-30 seconds, clamped to server admission.
use crate::client::{Chat, Client};
use crate::panels::{common, speed, Progress, Rates};
use crate::report::{
    now_rfc3339, BasicCard, Baseline, Check, CheckStatus, ConcurrentRate, ConcurrentTiming, ContentRate,
    PrefillRate, Quality, ServerInfo, StreamTiming,
};
use crate::text::{filler, nonce};
use anyhow::{Context, Result};
use cuteafd_api::openai::probe::{ProbeRecord, ProbeSpec};
use serde_json::{json, Value};
use std::time::Instant;

/// Decode tokens per content type.
const DECODE_TOKENS: u64 = 320;
/// The prefill case's prompt length.
const PREFILL_TOKENS: u64 = 8192;

pub const CONTENT: [(&str, &str); 3] = [
    ("code", "Write a complete Python module that implements a thread-safe LRU cache with per-entry TTL expiry, \
        a background janitor thread, hit/miss statistics and a full pytest test suite. Output only the code."),
    ("prose", "Write a long, richly detailed short story about a lighthouse keeper on a remote northern island \
        who finds a message in a bottle that seems to predict the weather. Use vivid description and dialogue."),
    ("json", "Output a JSON array of 40 fictional customer records. Each record has: id (integer), name, email, \
        city, country, signup_date (YYYY-MM-DD), plan (free, pro or team), monthly_spend (number) and tags (an \
        array of two to four strings). Output only the JSON array."),
];

fn messages(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

fn plain(text: &str, max_tokens: u64) -> Value {
    json!({"messages": messages(text), "max_tokens": max_tokens, "temperature": 0, "thinking": {"type": "disabled"}})
}

/// Expected seconds of the whole baseline at `rates`.
pub fn estimate_s(rates: &Rates, info: &ServerInfo) -> f64 {
    let decode = 3.0 * rates.seconds(60.0, DECODE_TOKENS as f64);
    let prefill = rates.seconds(PREFILL_TOKENS as f64 + 1600.0, 4.0);
    let fidelity = 4096.0 / rates.decode_tok_s + 24000.0 / rates.prefill_tok_s + 8.0 * 0.15;
    let cache = rates.seconds(6.0 * 1500.0, 30.0);
    let spec = rates.seconds(200.0, 2.0 * 128.0 * 1.6);
    let template = rates.seconds(400.0, 400.0);
    let c4 = rates.seconds(500.0, 2.0 * 64.0);
    decode + concurrent_estimate_s(rates, info) + prefill + fidelity + cache + spec + template + c4 + 4.0
}

fn concurrent_width(info: &ServerInfo) -> usize {
    common::concurrency(info).min(8)
}

fn concurrent_estimate_s(rates: &Rates, info: &ServerInfo) -> f64 {
    let width = concurrent_width(info) as f64;
    if width == 0.0 { return 0.0; }
    // Warm-up and timed wave; the sweep's estimate models the same batching cost.
    2.0 * rates.seconds(120.0 * width, DECODE_TOKENS as f64) * (1.0 + 0.08 * width)
}

fn concurrent_rate(results: Vec<common::Timed>, warmup_s: f64) -> ConcurrentRate {
    let mut per: Vec<_> = results.iter().map(|r| r.chat.timing.decode_tok_s()).collect();
    let first = results.iter().map(common::Timed::first).fold(f64::INFINITY, f64::min);
    let last = results.iter().map(common::Timed::last).fold(0.0, f64::max);
    ConcurrentRate { width: results.len(), aggregate_tok_s: common::aggregate(&results),
        per_stream_median_tok_s: common::median(&mut per), decode_s: (last - first).max(0.0), warmup_s,
        runs: results.into_iter().map(|r| ConcurrentTiming { sent_s: r.sent, timing: r.chat.timing }).collect() }
}

struct Run<'a> {
    client: &'a Client,
    info: &'a ServerInfo,
    progress: &'a Progress,
    max_context: u64,
    baseline: Baseline,
}

impl Run<'_> {
    fn step(&self, fraction: f64, label: &str) {
        self.progress.step(fraction, label);
    }

    fn publish(&self) {
        if let Ok(value) = serde_json::to_value(&self.baseline) {
            self.progress.partial(value);
        }
    }

    fn setting(&self, name: &str) -> Option<&str> {
        self.info.configuration.settings.iter().find(|s| s.name == name).and_then(|s| s.value.as_deref())
    }

    fn check(&mut self, id: &str, title: &str, run: impl FnOnce(&mut Self, &mut Check) -> Result<()>) {
        let started = Instant::now();
        let mut check = Check::new(id, title);
        if let Err(error) = run(self, &mut check) {
            if error.downcast_ref::<crate::client::Cancelled>().is_some() {
                check.status = CheckStatus::Pending;
                check.summary = "cancelled".into();
            } else {
                check.status = CheckStatus::Fail;
                check.summary = format!("{error:#}");
            }
        }
        check.seconds = started.elapsed().as_secs_f64();
        self.baseline.quality.checks.push(check);
        self.baseline.quality.settle();
        self.publish();
    }
}

/// Probe record of a chat, or an error naming what is missing.
fn probe_of(chat: &Chat) -> Result<&ProbeRecord> {
    chat.probe.as_ref().context("no probe record")
}

/// Whether the engine honoured the probe at all.
fn honoured(record: &ProbeRecord) -> bool {
    record.engine.is_some()
}

fn unsupported(check: &mut Check) {
    check.status = CheckStatus::Unsupported;
    check.summary = "this engine does not run benchmark probes yet".into();
}

pub fn run(client: &Client, info: &ServerInfo, progress: &Progress, run_id: &str, fingerprint: &str,
    max_context: u64, max_output: u64) -> Result<Baseline> {
    let started = Instant::now();
    let kv = crate::context::kv().unwrap_or_default();
    let from_settings = crate::report::Capacity::from_settings(info);
    let capacity = crate::report::Capacity { kv_tokens: kv.tokens.or(from_settings.kv_tokens), kv_pages: kv.pages,
        kv_format: kv.format, max_requests: from_settings.max_requests, max_context: Some(max_context),
        max_output: Some(max_output), host_cache_bytes: kv.host_bytes.or(from_settings.host_cache_bytes) };
    let mut run = Run {
        client, info, progress, max_context,
        baseline: Baseline { fingerprint: fingerprint.into(), run_id: run_id.into(), created: now_rfc3339(),
            card: BasicCard { capacity: Some(capacity), ..BasicCard::default() }, quality: Quality::default(), seconds: 0.0 },
    };
    // Warm-up: first-use workspaces, graphs and tables (untimed), and a two-point
    // fit of tokens per filler word for the prefill case.
    run.step(0.01, "warm-up");
    let warmup = Instant::now();
    let mut fit = Vec::new();
    for (i, words) in [300usize, 900].into_iter().enumerate() {
        let text = format!("[{}] Summarize in one word.\n\n{}", nonce(), filler(17 + i as u64, words));
        let chat = client.chat(plain(&text, 8), None).context("warm-up request")?;
        fit.push((words as f64, chat.timing.prompt_tokens as f64));
    }
    client.chat(plain(&format!("[{}] {}", nonce(), CONTENT[0].1), 32), None).context("warm-up decode")?;
    run.baseline.card.warmup_s = Some(warmup.elapsed().as_secs_f64());
    // C1 decode per content type.
    for (i, (content, prompt)) in CONTENT.iter().enumerate() {
        run.step(0.04 + 0.08 * i as f64, &format!("C1 decode · {content}"));
        let chat = client.chat(plain(&format!("[{}] {prompt}", nonce()), DECODE_TOKENS), None)
            .with_context(|| format!("{content} decode"))?;
        run.baseline.card.decode.push(ContentRate { content: content.to_string(), tok_s: chat.timing.decode_tok_s(),
            runs: vec![chat.timing], acceptance: None });
        run.publish();
    }
    // C8 code: never enqueue more than the server admits, and keep warm-up untimed.
    let width = concurrent_width(info);
    anyhow::ensure!(width > 0, "server admits no concurrent sequences");
    run.step(0.28, &format!("C{width} code warm-up"));
    let warmup = Instant::now();
    speed::code_batch(client, width, DECODE_TOKENS).into_iter().collect::<Result<Vec<_>>>()
        .with_context(|| format!("C{width} code warm-up"))?;
    let warmup_s = warmup.elapsed().as_secs_f64();
    client.check()?;
    run.step(0.34, &format!("C{width} code aggregate"));
    let batch = speed::code_batch(client, width, DECODE_TOKENS).into_iter().collect::<Result<Vec<_>>>()
        .with_context(|| format!("C{width} code decode"))?;
    run.baseline.card.concurrent = Some(concurrent_rate(batch, warmup_s));
    run.publish();
    // 8K prefill: a cold prompt sized from the fit.
    run.step(0.40, "8K prefill");
    let (slope, intercept) = match fit.as_slice() {
        [(w0, t0), (w1, t1)] if w1 > w0 && t1 > t0 => ((t1 - t0) / (w1 - w0), t0 - (t1 - t0) / (w1 - w0) * w0),
        _ => (1.3, 20.0),
    };
    let target = PREFILL_TOKENS.min(max_context.saturating_sub(64)) as f64;
    let words = ((target - intercept) / slope).max(64.0) as usize;
    // Warm beyond the full 8K chunk when context permits. Near-equal filler
    // lengths can move a lane boundary and leave part of the measured lane
    // cold. Keep normal prefix-cache retention work in both requests.
    let warm_target = (target + 512.0).min(max_context.saturating_sub(64) as f64);
    let warm_words = ((warm_target - intercept) / slope).max(64.0) as usize;
    let text = format!("[{}] Reply with the single word OK.\n\n{}", nonce(), filler(98, warm_words));
    client.chat(plain(&text, 1), None).context("8K prefill warm-up")?;
    let text = format!("[{}] Reply with the single word OK.\n\n{}", nonce(), filler(99, words));
    // Prime this exact lane shape and its routed expert accesses too. A cold
    // probe neither reads nor retains a prefix, so the timed ordinary request
    // still performs its full prefill and normal snapshot work.
    client.chat(plain(&text, 1), Some(ProbeSpec { cold: true, ..ProbeSpec::default() }))
        .context("8K prefill route warm-up")?;
    let chat = client.chat(plain(&text, 1), None).context("8K prefill")?;
    let t = chat.timing.clone();
    anyhow::ensure!(t.cached_tokens == 0, "8K cold prefill reused {} cached tokens", t.cached_tokens);
    run.baseline.card.prefill = Some(PrefillRate { prompt_tokens: t.prompt_tokens, tok_s: t.prefill_tok_s(),
        ttft_s: t.ttft_s, runs: vec![t] });
    run.publish();
    // Quick quality.
    run.step(0.50, "logit fidelity");
    run.check("fidelity", "Logit fidelity", fidelity);
    run.step(0.62, "prefix-cache restore");
    run.check("cache_exact", "Prefix-cache restore", cache_exact);
    run.step(0.74, "lossless speculation");
    run.check("spec_lossless", "Speculation lossless", spec_lossless);
    run.step(0.82, "chat-template round trip");
    run.check("template", "Template round trip", template);
    run.step(0.90, "C1 vs C4");
    run.check("c1_c4", "C1 vs C4 divergence", c1_c4);
    batch_variant_speculation(&mut run.baseline.quality);
    run.baseline.seconds = started.elapsed().as_secs_f64();
    run.baseline.quality.settle();
    run.step(1.0, "baseline done");
    run.publish();
    Ok(run.baseline)
}

fn fidelity(run: &mut Run<'_>, check: &mut Check) -> Result<()> {
    if let Some(reason) = crate::fidelity_dataset::unavailable(&run.info.checkpoint()) {
        check.status = CheckStatus::Unsupported;
        check.summary = reason;
        return Ok(());
    }
    let scored = crate::panels::fidelity::score(run.client, run.info, "quick", "decode", run.progress,
        0.50, 0.11, run.max_context)?;
    let verdict = crate::fidelity::verdict(&scored);
    let f = &verdict.generated;
    check.set("kl", f.kl); check.set("top1", f.top1); check.set("nll", f.nll);
    check.set("ref_nll", f.ref_nll); check.set("positions", f.positions as u64);
    check.set("missing", scored.score.missing as u64);
    check.set("kl_max", verdict.kl_max); check.set("top1_min", verdict.top1_min);
    check.set("confident_top1", json!(f.confident_top1)); check.set("top3_contained", f.top3_contained);
    check.set("dataset", json!(scored.dataset));
    check.set("quick_subset", "bench-v1:legacy,a00,a04,a08,a20,c00,d00,e00");
    check.set("per_window", json!(crate::panels::fidelity::window_summaries(&scored)));
    check.set("verdict", json!(verdict));
    check.set("run", json!(scored));
    check.status = if verdict.pass { CheckStatus::Pass } else { CheckStatus::Fail };
    check.summary = format!("KL {:.3} · top-1 {:.1}% · {} generated / 4096 total rows · 8 windows",
        f.kl, 100.0 * f.top1, f.positions);
    Ok(())
}

/// The probe record of one request.
fn probed(client: &Client, body: Value, spec: ProbeSpec) -> Result<ProbeRecord> {
    let chat = client.chat(body, Some(spec))?;
    probe_of(&chat).cloned()
}

/// Whether `a` and `b` recorded byte-identical rows at every position both hold:
/// (positions compared, positions that differ).
fn compare_rows(a: &ProbeRecord, b: &ProbeRecord) -> (usize, Vec<usize>) {
    let mut compared = 0;
    let mut differ = Vec::new();
    for row in &b.rows {
        if let Some(other) = a.rows.iter().find(|r| r.position == row.position) {
            compared += 1;
            if other.hash != row.hash {
                differ.push(row.position);
            }
        }
    }
    (compared, differ)
}

/// Restore exactness, against the state each snapshot was taken from (drafts off, same chunking).
///
/// Prompt end: a prompt computed and retained, then asked again (a whole hit): its rows must equal
/// the first request's.
///
/// Turn end: a 24-token turn records its own rows; the same prompt asked for one token more is a
/// whole hit on that prompt snapshot, so its 25 rows come from the restored prompt state and plain
/// decode steps and must equal the turn's; then the turn's tokens again are a whole hit on the turn
/// snapshot, whose rows (the retained logits, then a decode step on the restored state) must equal
/// the reference's at those positions. Neither side recomputes a prefill: a cold recompute is
/// reported, not gated, because prefill kernels with unordered reductions (Spark FP32 atomics) are
/// not bit-reproducible run to run, which is a property of the kernels, not of the cache.
fn cache_exact(run: &mut Run<'_>, check: &mut Check) -> Result<()> {
    if run.setting("prefix-cache-entries").is_some_and(|v| v == "0") {
        check.status = CheckStatus::Skipped;
        check.summary = "prefix cache off (--prefix-cache-entries 0)".into();
        return Ok(());
    }
    let client = run.client;
    let rows = |n: usize| ProbeSpec { no_speculation: true, record_rows: n, top_k: 1, ..ProbeSpec::default() };
    // Prompt end: the first request computes and retains, the second restores it whole.
    let text = format!("[{}] Read the notes and answer in one word.\n\n{}", nonce(), filler(31, 1100));
    let first = probed(client, plain(&text, 2), rows(2))?;
    if !honoured(&first) {
        unsupported(check);
        return Ok(());
    }
    let restored = probed(client, plain(&text, 2), rows(2))?;
    // Turn end. Long enough to clear the cache's minimum snapshot size and several units; the
    // answer runs past 25 tokens, so the turn ends at its length limit and the turn snapshot is
    // followed by a decode step.
    let turn_text = format!("[{}] Here are some notes.\n\n{}\n\nCount from one to forty in words, separated by commas.",
        nonce(), filler(57, 900));
    let turn = probed(client, plain(&turn_text, TURN_TOKENS as u64), rows(TURN_TOKENS))?;
    let reference = probed(client, plain(&turn_text, TURN_TOKENS as u64 + 1), rows(TURN_TOKENS + 1))?;
    let mut ids = turn.prompt_ids.clone();
    ids.extend(&turn.generated[..turn.generated.len().saturating_sub(1)]);
    let again = probed(client, plain("cache probe", 2), ProbeSpec { prompt_ids: Some(ids), ..rows(2) })?;
    let cold = probed(client, plain(&turn_text, 1), ProbeSpec { cold: true, ..rows(1) })?;
    let verdict = Restores::judge(&first, &restored, &turn, &reference, &again, &cold);
    check.set("prompt_tokens", restored.prompt_ids.len() as u64);
    check.set("prompt_restored", restored.cached_tokens as u64);
    check.set("prompt_rows_compared", verdict.prompt.compared as u64);
    check.set("turn_tokens", again.prompt_ids.len() as u64);
    check.set("turn_restored", again.cached_tokens as u64);
    check.set("turn_rows_compared", verdict.turn.compared as u64);
    check.set("turn_prompt_rows_compared", verdict.turn_prompt.compared as u64);
    check.set("cold_rows_identical", u64::from(verdict.cold_identical == Some(true)));
    check.status = verdict.status();
    check.summary = verdict.summary();
    Ok(())
}

/// Generated tokens of the turn-end case's turn.
const TURN_TOKENS: usize = 24;

/// One restore compared with the rows it must reproduce.
#[derive(Debug, Default, Clone, PartialEq)]
struct Restore {
    /// The request restored its whole prompt from a snapshot.
    hit: bool,
    restored: usize,
    tokens: usize,
    compared: usize,
    differ: Vec<usize>,
}

impl Restore {
    /// `restored`'s rows against `source`'s at every position both hold.
    fn of(source: &ProbeRecord, restored: &ProbeRecord) -> Self {
        let (compared, differ) = compare_rows(source, restored);
        Self { hit: !restored.prompt_ids.is_empty() && restored.cached_tokens == restored.prompt_ids.len(),
            restored: restored.cached_tokens, tokens: restored.prompt_ids.len(), compared, differ }
    }

    fn failed(&self) -> bool {
        self.hit && !self.differ.is_empty()
    }

    fn exact(&self, rows: usize) -> bool {
        self.hit && self.differ.is_empty() && self.compared >= rows
    }

    fn describe(&self) -> String {
        if !self.hit {
            format!("no whole restore ({}/{})", self.restored, self.tokens)
        } else if !self.differ.is_empty() {
            format!("{} restored, rows DIFFER at {:?}", self.tokens, self.differ)
        } else {
            format!("{} restored, {} rows byte-identical", self.tokens, self.compared)
        }
    }
}

/// The prefix-cache check's verdict.
#[derive(Debug)]
struct Restores {
    prompt: Restore,
    /// The turn's prompt asked for one token more, from its prompt snapshot, against the turn.
    turn_prompt: Restore,
    /// The turn's tokens from the turn snapshot, against that reference.
    turn: Restore,
    /// The reference emitted the turn's tokens first.
    reproduced: bool,
    /// Whether a cold recompute of the turn's prompt gave the turn's first row bit for bit.
    cold_identical: Option<bool>,
}

impl Restores {
    fn judge(first: &ProbeRecord, restored: &ProbeRecord, turn: &ProbeRecord, reference: &ProbeRecord,
        again: &ProbeRecord, cold: &ProbeRecord) -> Self {
        let (compared, differ) = compare_rows(turn, cold);
        let turn_prompt = Restore::of(turn, reference);
        // Without a prompt restore the reference recomputed the prefill: only the turn's own
        // (retained) row is a fair comparison then.
        let turn_source = if turn_prompt.hit { reference } else { turn };
        Self {
            prompt: Restore::of(first, restored),
            turn: Restore::of(turn_source, again),
            turn_prompt,
            reproduced: !turn.generated.is_empty() && reference.generated.starts_with(&turn.generated),
            cold_identical: (compared > 0).then_some(differ.is_empty()),
        }
    }

    fn status(&self) -> CheckStatus {
        if self.prompt.failed() || self.turn_prompt.failed() || self.turn.failed()
            || (self.turn_prompt.hit && !self.reproduced) {
            CheckStatus::Fail
        } else if self.prompt.exact(2) && self.turn_prompt.exact(2) && self.turn.exact(2) {
            CheckStatus::Pass
        } else {
            CheckStatus::Info
        }
    }

    fn summary(&self) -> String {
        let mut summary = format!("prompt end: {} · turn end: {}", self.prompt.describe(), self.turn.describe());
        if !self.turn_prompt.hit || self.turn_prompt.failed() {
            summary.push_str(&format!(" · turn's prompt: {}", self.turn_prompt.describe()));
        }
        if self.turn_prompt.hit && !self.reproduced {
            summary.push_str(" · the turn's greedy text did not reproduce from its prompt snapshot");
        }
        if self.prompt.compared < 2 || self.turn.compared < 2 {
            summary.push_str(" (no decode rows recorded after a restore)");
        }
        match self.cold_identical {
            Some(true) => summary.push_str(" · cold recompute identical"),
            Some(false) => summary.push_str(" · cold recompute differs (prefill not bit-reproducible; not a cache defect)"),
            None => {}
        }
        summary
    }
}

fn spec_lossless(run: &mut Run<'_>, check: &mut Check) -> Result<()> {
    let Some(speculator) = run.info.configuration.speculator.clone() else {
        check.status = CheckStatus::Skipped;
        check.summary = "no speculator".into();
        return Ok(());
    };
    const TOKENS: u64 = 128;
    let text = format!("[{}] {}", nonce(), CONTENT[0].1);
    let spec = |off: bool| ProbeSpec { cold: true, no_speculation: off, record_rows: TOKENS as usize, top_k: 2,
        ..ProbeSpec::default() };
    let on = run.client.chat(plain(&text, TOKENS), Some(spec(false)))?;
    let on_record = probe_of(&on)?;
    if !honoured(on_record) {
        unsupported(check);
        return Ok(());
    }
    let off = run.client.chat(plain(&text, TOKENS), Some(spec(true)))?;
    let off_record = probe_of(&off)?;
    let (a, b) = (&on_record.generated, &off_record.generated);
    let same = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    check.set("tokens", a.len() as u64);
    check.set("identical_prefix", same as u64);
    check.set("speculator", speculator.clone());
    check.set("decode_tok_s_on", on.timing.decode_tok_s());
    check.set("decode_tok_s_off", off.timing.decode_tok_s());
    let rates = format!("({} vs {} tok/s)", crate::render::rate(on.timing.decode_tok_s()),
        crate::render::rate(off.timing.decode_tok_s()));
    if a == b {
        check.status = CheckStatus::Pass;
        check.summary = format!("{speculator}: {} greedy tokens identical with drafts on and off {rates}", a.len());
        return Ok(());
    }
    // Evidence for numerics vs state: how far the drafted run's rows already
    // were from the one-row run's on the identical prefix. A flip within that
    // noise is rounding; noise that is zero until a point and then grows
    // points at state (KV or recurrent rows a verify left behind).
    let noise = RowNoise::between(on_record, off_record, off_record.prompt_ids.len() + same);
    noise.record(check);
    // And whether one-row decoding is even reproducible: the same request
    // again without drafts. Rows that differ here are run-to-run
    // nondeterminism (reduction order), not anything a verify does.
    let again = run.client.chat(plain(&text, TOKENS), Some(spec(true)))?;
    let again_record = probe_of(&again)?;
    let same_again = b.iter().zip(&again_record.generated).take_while(|(x, y)| x == y).count();
    let repeat = RowNoise::between(again_record, off_record, off_record.prompt_ids.len() + same_again);
    check.set("repeat_identical_prefix", same_again as u64);
    check.set("repeat_rows_identical", repeat.identical as u64);
    check.set("repeat_noise_max", repeat.max);
    let repeat_note = format!("; drafts off twice: {same_again} of {} tokens identical, {} of {} rows byte-identical, \
        up to {:.3} nats", b.len(), repeat.identical, repeat.compared, repeat.max);
    // Verify rows and single-row steps may round differently: a flip where the
    // top two candidates are within rounding of each other is a tie, not a loss.
    let position = off_record.prompt_ids.len() + same;
    let margin = |record: &ProbeRecord| record.rows.iter().find(|r| r.position == position)
        .and_then(|r| (r.top.len() >= 2).then(|| f64::from(r.top[0].1 - r.top[1].1)));
    let margins = (margin(off_record), margin(on_record));
    let tie = match margins {
        (Some(x), Some(y)) => x.min(y) < TIE_NATS,
        (Some(x), None) | (None, Some(x)) => x < TIE_NATS,
        _ => false,
    };
    if let Some(m) = margins.0.or(margins.1) {
        check.set("divergence_margin", m);
    }
    let decode_repeats = same_again == b.len() && again_record.generated.len() == b.len()
        && repeat.identical == repeat.compared;
    check.status = flip_verdict(tie, decode_repeats, &noise);
    check.summary = if tie {
        format!("{speculator}: identical up to token {same} of {}, then a near-tie flips \
            (top-two margin {:.3} nats) {rates}{}{repeat_note}", a.len().max(b.len()), margins.0.or(margins.1).unwrap_or(0.0),
            noise.describe())
    } else {
        format!("{speculator}: greedy output diverges at token {same} of {}{} {rates}{}{}{}",
            a.len().max(b.len()), margins.0.or(margins.1).map(|m| format!(" (top-two margin {m:.3} nats)"))
                .unwrap_or_default(), noise.describe(), repeat_note,
            if check.status == CheckStatus::Info { " · not gated: verify rounding (plain decode repeats exactly, \
                drafted rows already differed before the flip)" } else { "" })
    };
    Ok(())
}

/// The lossless verdict for a drafts-on vs drafts-off flip.
///
/// A near tie passes. Otherwise the flip is verify rounding, reported but not
/// gated, when plain decoding is deterministic (`decode_repeats`: the
/// drafts-off rerun reproduced every token and row byte for byte) and the
/// drafted rows already differed from one-row decode before the flip (verify
/// steps round differently from single rows). It fails when plain decoding
/// does not repeat (the comparison proves nothing) or when the rows were
/// byte-identical up to the flip (a sudden change points at state, not
/// rounding).
fn flip_verdict(tie: bool, decode_repeats: bool, noise: &RowNoise) -> CheckStatus {
    if tie {
        CheckStatus::Pass
    } else if decode_repeats && noise.compared > 0 && noise.identical < noise.compared {
        CheckStatus::Info
    } else {
        CheckStatus::Fail
    }
}

/// Per-row difference between two greedy runs' recorded rows over their
/// identical prefix: the log-probability of the (shared) top token.
#[derive(Debug, Default, PartialEq)]
struct RowNoise {
    compared: usize,
    identical: usize,
    max: f64,
    median: f64,
    /// Generated-token index of the first row more than 0.01 nats apart.
    first_over: Option<usize>,
}

impl RowNoise {
    /// Rows predicting positions before `end` that both records hold.
    fn between(a: &ProbeRecord, b: &ProbeRecord, end: usize) -> Self {
        let start = b.prompt_ids.len();
        let mut deltas = Vec::new();
        let mut noise = RowNoise::default();
        for row in b.rows.iter().filter(|r| r.position < end) {
            let Some(other) = a.rows.iter().find(|r| r.position == row.position) else { continue };
            let (Some(x), Some(y)) = (row.top.first(), other.top.first()) else { continue };
            if x.0 != y.0 {
                continue;
            }
            noise.compared += 1;
            noise.identical += usize::from(row.hash == other.hash);
            let delta = f64::from((x.1 - y.1).abs());
            if delta > 0.01 && noise.first_over.is_none() {
                noise.first_over = Some(row.position.saturating_sub(start));
            }
            deltas.push(delta);
        }
        deltas.sort_by(f64::total_cmp);
        noise.max = deltas.last().copied().unwrap_or(0.0);
        noise.median = deltas.get(deltas.len() / 2).copied().unwrap_or(0.0);
        noise
    }

    fn record(&self, check: &mut Check) {
        if self.compared == 0 {
            return;
        }
        check.set("prefix_rows_compared", self.compared as u64);
        check.set("prefix_rows_identical", self.identical as u64);
        check.set("prefix_noise_max", self.max);
        check.set("prefix_noise_median", self.median);
        if let Some(at) = self.first_over {
            check.set("prefix_noise_first_over_0_01", at as u64);
        }
    }

    fn describe(&self) -> String {
        if self.compared == 0 {
            return String::new();
        }
        format!("; before it {} of {} rows byte-identical, top-token log-prob differs by up to {:.3} nats \
            (median {:.4}){}", self.identical, self.compared, self.max, self.median,
            self.first_over.map(|at| format!(", first over 0.01 at token {at}")).unwrap_or_default())
    }
}

/// Top-two log-probability margin under which a greedy flip counts as a tie.
const TIE_NATS: f64 = 0.05;

/// Top-two margin under which a speculation flip is kernel noise: drafts are verified
/// as several rows of one request, a different kernel shape than one-row decode.
const VERIFY_NOISE_NATS: f64 = 0.5;

/// A flip past a near tie but under [`VERIFY_NOISE_NATS`] says the verify rows round
/// differently, not that speculation changes the model: reported, not gated. Larger
/// flips still fail. The summary says whether plain C4 also differs from C1.
fn batch_variant_speculation(quality: &mut Quality) {
    let batch_variant = quality.checks.iter().any(|c| c.id == "c1_c4" && c.status == CheckStatus::Info
        && c.metrics.get("identical").and_then(|v| v.as_u64()).is_some_and(|n| n < 4));
    let Some(check) = quality.checks.iter_mut().find(|c| c.id == "spec_lossless" && c.status == CheckStatus::Fail)
    else { return };
    let margin = check.metrics.get("divergence_margin").and_then(|v| v.as_f64());
    if margin.is_some_and(|m| m < VERIFY_NOISE_NATS) {
        check.status = CheckStatus::Info;
        check.summary = format!("{} · not gated: under {VERIFY_NOISE_NATS} nats{}", check.summary,
            if batch_variant { "; C4 differs from C1 too" } else { "" });
    }
}

fn template(run: &mut Run<'_>, check: &mut Check) -> Result<()> {
    let tools = json!([{"type": "function", "function": {"name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]);
    let question = format!("[{}] What is the weather in Paris right now? Use the get_weather tool.", nonce());
    let mut body = json!({"messages": messages(&question), "tools": tools, "max_tokens": 600, "temperature": 0,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "thinking": {"type": "enabled"}, "reasoning_effort": "low"});
    let first = run.client.chat(body.clone(), Some(ProbeSpec::default()))?;
    let Some(call) = first.tool_calls.first().cloned() else {
        check.status = CheckStatus::Fail;
        check.summary = format!("no tool call in {} tokens (finish {})", first.timing.completion_tokens,
            first.timing.finish_reason.as_deref().unwrap_or("?"));
        return Ok(());
    };
    let arguments: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap_or(""))
        .context("tool arguments are not JSON")?;
    anyhow::ensure!(call["function"]["name"] == "get_weather", "called {}", call["function"]["name"]);
    anyhow::ensure!(arguments["city"].is_string(), "arguments {arguments} lack the city");
    check.set("reasoning_chars", first.reasoning.chars().count() as u64);
    // Second turn: the assistant turn (reasoning and call) and the tool result rendered back.
    let id = call["id"].as_str().filter(|s| !s.is_empty()).unwrap_or("call_0").to_string();
    let mut call = call;
    call["id"] = json!(id);
    let mut history = messages(&question);
    let history_list = history.as_array_mut().expect("array");
    history_list.push(json!({"role": "assistant", "content": first.content, "reasoning_content": first.reasoning,
        "tool_calls": [call]}));
    history_list.push(json!({"role": "tool", "tool_call_id": id, "content": "{\"temperature_c\": 18, \"conditions\": \"cloudy\"}"}));
    body["messages"] = history;
    body["max_tokens"] = json!(1);
    body.as_object_mut().expect("object").remove("tool_choice");
    let second = run.client.chat(body, Some(ProbeSpec::default()))?;
    let (Some(a), Some(b)) = (first.probe.as_ref(), second.probe.as_ref()) else {
        anyhow::bail!("no probe records");
    };
    let thought = !first.reasoning.trim().is_empty();
    if !honoured(a) || !honoured(b) {
        check.status = if thought { CheckStatus::Pass } else { CheckStatus::Info };
        check.summary = format!("tool call parsed ({}), {}; re-render not checked (no probes)", arguments,
            if thought { "reasoning returned" } else { "no reasoning returned" });
        return Ok(());
    }
    let mut expected = a.prompt_ids.clone();
    expected.extend(&a.generated);
    let common = expected.iter().zip(&b.prompt_ids).take_while(|(x, y)| x == y).count();
    // The final stop token may be re-rendered differently (or not at all).
    let exact = common + 1 >= expected.len();
    check.set("rendered_prefix", common as u64);
    check.set("turn_tokens", expected.len() as u64);
    check.set("cached_tokens", b.cached_tokens as u64);
    check.status = match (thought, exact) {
        (true, true) => CheckStatus::Pass,
        _ => CheckStatus::Info,
    };
    check.summary = format!("tool call parsed · {} · re-render {}", if thought { "reasoning kept" } else { "no reasoning" },
        if exact { format!("identical ({} tokens, {} cached)", expected.len(), b.cached_tokens) }
        else { format!("differs at token {common} of {}", expected.len()) });
    Ok(())
}

fn c1_c4(run: &mut Run<'_>, check: &mut Check) -> Result<()> {
    let text = format!("[{}] {}", nonce(), CONTENT[1].1);
    let probe = || Some(ProbeSpec { cold: true, record_rows: 64, top_k: 2, ..ProbeSpec::default() });
    let one = run.client.chat(plain(&text, 64), probe())?;
    let client = run.client.clone();
    let four: Vec<Result<Chat>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4).map(|_| {
            let client = client.clone();
            let text = text.clone();
            scope.spawn(move || client.chat(plain(&text, 64), probe()))
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("thread panicked")))).collect()
    });
    let reference = output_of(&one);
    let mut identical = 0;
    let mut first_divergence: Option<usize> = None;
    // Row noise against C1 over the earliest-diverging output's identical prefix.
    let mut noise = RowNoise::default();
    for chat in &four {
        let chat = chat.as_ref().map_err(|e| anyhow::anyhow!("{e:#}"))?;
        let other = output_of(chat);
        if other == reference {
            identical += 1;
        } else {
            let at = reference.iter().zip(&other).take_while(|(a, b)| a == b).count();
            if first_divergence.is_none_or(|d| at < d) {
                if let (Some(a), Some(b)) = (chat.probe.as_ref().filter(|r| honoured(r)),
                    one.probe.as_ref().filter(|r| honoured(r))) {
                    noise = RowNoise::between(a, b, b.prompt_ids.len() + at);
                }
            }
            first_divergence = Some(first_divergence.map_or(at, |d: usize| d.min(at)));
        }
    }
    noise.record(check);
    check.status = CheckStatus::Info;
    check.set("identical", identical as u64);
    if let Some(at) = first_divergence {
        check.set("first_divergence", at as u64);
    }
    let unit = if one.probe.as_ref().is_some_and(honoured) { "tokens" } else { "characters" };
    check.summary = match first_divergence {
        None => format!("4 of 4 concurrent greedy outputs identical to C1 ({} {unit})", reference.len()),
        Some(at) => format!("{identical} of 4 identical to C1; first divergence at {unit} {at}{}", noise.describe()),
    };
    Ok(())
}

/// Qualification comparison at the first generated-token divergence. Only rows
/// with identical preceding context are comparable; later rows are not noise.
pub(crate) fn compare_probe_outputs(reference: &ProbeRecord, other: &ProbeRecord) -> Result<Value> {
    anyhow::ensure!(honoured(reference) && honoured(other) && reference.error.is_none() && other.error.is_none(),
        "comparison requires honoured, successful probes");
    anyhow::ensure!(reference.prompt_ids == other.prompt_ids, "comparison prompt ids differ");
    let at = reference.generated.iter().zip(&other.generated).take_while(|(a, b)| a == b).count();
    let identical = reference.generated == other.generated;
    let position = reference.prompt_ids.len() + at;
    let noise = RowNoise::between(other, reference, position);
    let mut check = Check::new("prefill_share_noise", "Common-prefix row noise");
    noise.record(&mut check);
    let rows = reference.rows.iter().find(|r| r.position == position)
        .zip(other.rows.iter().find(|r| r.position == position));
    let divergence = (!identical).then(|| rows.map(|(a, b)| {
        // Coarsen to ids held by both summaries plus one tail bucket. This is
        // a KL lower bound, not a claim of full-vocabulary KL from top-k rows.
        let common: Vec<_> = a.top.iter().filter_map(|&(id, lp)| b.top.iter()
            .find(|(other, _)| *other == id).map(|&(_, other_lp)| (f64::from(lp).exp(), f64::from(other_lp).exp()))).collect();
        let p_tail = (1.0 - common.iter().map(|x| x.0).sum::<f64>()).max(0.0);
        let q_tail = (1.0 - common.iter().map(|x| x.1).sum::<f64>()).max(0.0);
        let kl: f64 = common.iter().copied().chain(std::iter::once((p_tail, q_tail)))
            .filter(|&(p, _)| p > 0.0).map(|(p, q)| p * (p / q.max(f64::MIN_POSITIVE)).ln()).sum();
        let margin = |row: &cuteafd_api::openai::probe::ProbeRow| row.top.first().zip(row.top.get(1))
            .map(|(a, b)| (a.1 - b.1).abs());
        json!({"reference_top1": a.argmax, "other_top1": b.argmax, "row_byte_exact": a.hash == b.hash,
            "reference_top2_margin": margin(a), "other_top2_margin": margin(b),
            "coarsened_kl": kl.max(0.0), "kl_kind": "common-top-k-plus-tail lower bound", "common_ids": common.len()})
    })).flatten();
    Ok(json!({"identical": identical, "first_divergence": (!identical).then_some(at),
        "reference_tokens": reference.generated.len(), "other_tokens": other.generated.len(),
        "prefix_noise": check.metrics, "divergence_row": divergence}))
}

/// Generated token ids when the probe recorded them, else the text's characters.
fn output_of(chat: &Chat) -> Vec<u32> {
    match chat.probe.as_ref().filter(|r| honoured(r)) {
        Some(record) => record.generated.clone(),
        None => chat.content.chars().map(|c| c as u32).collect(),
    }
}

/// A timing summary for logs.
pub fn describe(timing: &StreamTiming) -> String {
    format!("{} prompt, {} out, ttft {:.3}s, decode {:.1} tok/s", timing.prompt_tokens, timing.completion_tokens,
        timing.ttft_s, timing.decode_tok_s())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::Check;

    #[test]
    fn prefill_share_comparison_reports_common_prefix_and_divergence_only() {
        use cuteafd_api::openai::probe::{summarize, ProbeRecord};
        let record = |generated: Vec<u32>, logits: &[&[f32]]| ProbeRecord {
            engine: Some("fixture".into()), prompt_ids: vec![9, 8], generated,
            rows: logits.iter().enumerate().map(|(i, row)| summarize(2 + i, row, 2, &[])).collect(),
            ..ProbeRecord::default()
        };
        let a = record(vec![0, 0, 1], &[&[1.0, 0.0, -1.0], &[1.0, 0.9, -1.0], &[0.0, 1.0, -1.0]]);
        let exact = compare_probe_outputs(&a, &a).unwrap();
        assert_eq!(exact["identical"], true);
        assert!(exact["first_divergence"].is_null());
        assert!(exact["divergence_row"].is_null());
        let b = record(vec![0, 1, 0], &[&[1.0, 0.0, -1.0], &[0.9, 1.0, -1.0], &[100.0, -100.0, 0.0]]);
        let drift = compare_probe_outputs(&a, &b).unwrap();
        assert_eq!(drift["first_divergence"], 1);
        assert_eq!(drift["prefix_noise"]["prefix_rows_compared"], 1);
        assert_eq!(drift["prefix_noise"]["prefix_rows_identical"], 1);
        assert_eq!(drift["divergence_row"]["reference_top1"], 0);
        assert_eq!(drift["divergence_row"]["other_top1"], 1);
        assert!(drift["divergence_row"]["coarsened_kl"].as_f64().unwrap() > 0.0);
        let mut wrong = b.clone(); wrong.prompt_ids.push(7);
        assert!(compare_probe_outputs(&a, &wrong).is_err());
        wrong = b; wrong.engine = None;
        assert!(compare_probe_outputs(&a, &wrong).is_err());
    }

    #[test]
    fn prefill_share_panel_is_opt_in_only() {
        assert!(crate::panels::find("prefill_share").is_some());
        assert!(crate::profiles::builtin().iter().all(|profile| profile.panels.iter()
            .all(|panel| panel.id != "prefill_share")));
    }

    #[test]
    fn concurrent_width_never_exceeds_server_admission() {
        use crate::report::Setting;
        let mut info = ServerInfo::default();
        assert_eq!(concurrent_width(&info), 8);
        for name in ["concurrency", "max-sequences"] {
            for (limit, width) in [(0, 0), (1, 1), (4, 4), (6, 6), (8, 8), (16, 8)] {
                info.configuration.settings = vec![Setting { name: name.into(), value: Some(limit.to_string()),
                    ..Setting::default() }];
                assert_eq!(concurrent_width(&info), width);
            }
        }
    }

    #[test]
    fn estimate_includes_warm_and_timed_concurrent_batches() {
        use crate::report::Setting;
        let rates = Rates { decode_tok_s: 50.0, prefill_tok_s: 1500.0, measured: true };
        let mut info = ServerInfo::default();
        let c8 = estimate_s(&rates, &info);
        let expected = 2.0 * rates.seconds(8.0 * 120.0, DECODE_TOKENS as f64) * 1.64;
        assert!((20.0..30.0).contains(&expected));
        assert!((concurrent_estimate_s(&rates, &info) - expected).abs() < 1e-9);
        info.configuration.settings = vec![Setting { name: "concurrency".into(), value: Some("0".into()),
            ..Setting::default() }];
        assert!((c8 - estimate_s(&rates, &info) - expected).abs() < 1e-9);
        info.configuration.settings[0].value = Some("4".into());
        assert!(concurrent_estimate_s(&rates, &info) < expected);
    }

    #[test]
    fn concurrent_rate_matches_sweep_and_preserves_batch_clock() {
        let timed = |sent, ttft_s, decode_s, completion_tokens| common::Timed { sent,
            chat: Chat { timing: StreamTiming { ttft_s, decode_s, completion_tokens,
                ..StreamTiming::default() }, ..Chat::default() } };
        let results = vec![timed(0.0, 1.0, 4.0, 321), timed(0.5, 1.5, 2.0, 161)];
        let expected = common::aggregate(&results);
        let rate = concurrent_rate(results, 9.0);
        assert_eq!(rate.width, 2);
        assert_eq!(rate.aggregate_tok_s, expected);
        // 320 + 160 emitted decode tokens over [1, 5], not a sum of stream rates.
        assert_eq!(rate.aggregate_tok_s, 120.0);
        assert_eq!(rate.per_stream_median_tok_s, 80.0);
        assert_eq!(rate.decode_s, 4.0);
        assert_eq!(rate.warmup_s, 9.0);
        assert_eq!(rate.runs[1].sent_s, 0.5);
        assert_eq!(rate.runs[1].timing.completion_tokens, 161);
    }

    #[test]
    fn row_noise_compares_the_shared_top_token_over_the_identical_prefix() {
        use cuteafd_api::openai::probe::ProbeRow;
        let row = |position, hash: &str, top: Vec<(u32, f32)>| ProbeRow { position, hash: hash.into(), top,
            ..ProbeRow::default() };
        let off = ProbeRecord { prompt_ids: vec![0; 10], rows: vec![row(10, "a", vec![(5, -0.1), (6, -2.0)]),
            row(11, "b", vec![(7, -0.5), (8, -1.0)]), row(12, "c", vec![(9, -0.2), (1, -3.0)]),
            row(13, "d", vec![(2, -0.3), (3, -0.4)])], ..ProbeRecord::default() };
        let on = ProbeRecord { prompt_ids: vec![0; 10], rows: vec![row(10, "a", vec![(5, -0.1), (6, -2.0)]),
            row(11, "x", vec![(7, -0.505), (8, -1.0)]), row(12, "y", vec![(9, -0.3), (1, -3.0)]),
            row(13, "z", vec![(3, -0.3), (2, -0.4)])], ..ProbeRecord::default() };
        // Position 13 is the flip: it is outside the identical prefix.
        let noise = RowNoise::between(&on, &off, 13);
        assert_eq!((noise.compared, noise.identical, noise.first_over), (3, 1, Some(2)));
        assert!((noise.max - 0.1).abs() < 1e-6 && (noise.median - 0.005).abs() < 1e-6, "{noise:?}");
        assert!(noise.describe().contains("1 of 3 rows byte-identical"));
        assert_eq!(RowNoise::between(&on, &ProbeRecord::default(), 13).describe(), "");
    }

    /// A record of `prompt` prompt ids with `cached` restored, rows `(position, hash)` and `generated`.
    fn record(prompt: usize, cached: usize, rows: &[(usize, &str)], generated: &[u32]) -> ProbeRecord {
        use cuteafd_api::openai::probe::ProbeRow;
        ProbeRecord { engine: Some("test".into()), prompt_ids: vec![1; prompt], cached_tokens: cached,
            rows: rows.iter().map(|&(position, hash)| ProbeRow { position, hash: hash.into(), ..ProbeRow::default() })
                .collect(),
            generated: generated.to_vec(), ..ProbeRecord::default() }
    }

    /// Prompt end at 100 (rows 100, 101); a 3-token turn at 50 (rows 50..53), its reference (one
    /// token more, rows 50..54), the turn's tokens again from the turn snapshot (rows 52, 53) and a
    /// cold recompute (row 50).
    fn restores(reference_cached: usize, reference_rows: &[(usize, &str)], again_rows: &[(usize, &str)],
        cold_row: &str) -> Restores {
        let first = record(100, 0, &[(100, "p0"), (101, "p1")], &[7, 8]);
        let restored = record(100, 100, &[(100, "p0"), (101, "p1")], &[7, 8]);
        let turn = record(50, 0, &[(50, "t0"), (51, "t1"), (52, "t2")], &[4, 5, 6]);
        let reference = record(50, reference_cached, reference_rows, &[4, 5, 6, 9]);
        let again = record(52, 52, again_rows, &[6, 9]);
        let cold = record(50, 0, &[(50, cold_row)], &[4]);
        Restores::judge(&first, &restored, &turn, &reference, &again, &cold)
    }

    const REFERENCE: [(usize, &str); 4] = [(50, "t0"), (51, "t1"), (52, "t2"), (53, "t3")];

    #[test]
    fn turn_end_restores_compare_with_their_snapshot_not_a_cold_recompute() {
        // A cold prefill that rounds differently (Spark FP32 atomics) is reported, not a failure.
        let verdict = restores(50, &REFERENCE, &[(52, "t2"), (53, "t3")], "cold");
        assert_eq!(verdict.status(), CheckStatus::Pass, "{verdict:?}");
        assert_eq!((verdict.turn.compared, verdict.turn_prompt.compared), (2, 3));
        assert!(verdict.summary().contains("turn end: 52 restored, 2 rows byte-identical"), "{}", verdict.summary());
        assert!(verdict.summary().contains("cold recompute differs"), "{}", verdict.summary());
        assert!(restores(50, &REFERENCE, &[(52, "t2"), (53, "t3")], "t0").summary().contains("cold recompute identical"));
    }

    #[test]
    fn turn_end_state_that_restores_inexactly_fails() {
        // The retained row is right but the decode step on the restored state is not.
        let verdict = restores(50, &REFERENCE, &[(52, "t2"), (53, "bad")], "t0");
        assert_eq!(verdict.status(), CheckStatus::Fail);
        assert!(verdict.summary().contains("rows DIFFER at [53]"), "{}", verdict.summary());
        // The turn's prompt restored inexactly: its decode rows differ from the turn's own.
        let verdict = restores(50, &[(50, "t0"), (51, "x"), (52, "y"), (53, "z")], &[(52, "y"), (53, "z")], "t0");
        assert_eq!(verdict.status(), CheckStatus::Fail);
        assert!(verdict.summary().contains("turn's prompt: 50 restored, rows DIFFER at [51, 52]"), "{}", verdict.summary());
    }

    #[test]
    fn without_a_prompt_restore_the_turn_compares_with_its_own_row() {
        // The reference recomputed its prefill (no hit) and rounded differently: the turn snapshot
        // is judged on the row the turn itself produced, and the check stays informational.
        let verdict = restores(0, &[(50, "c0"), (51, "c1"), (52, "c2"), (53, "c3")], &[(52, "t2"), (53, "t3")], "c0");
        assert_eq!(verdict.status(), CheckStatus::Info, "{verdict:?}");
        assert_eq!((verdict.turn.compared, verdict.turn.differ.len()), (1, 0));
        let verdict = restores(0, &[(50, "c0")], &[(52, "bad"), (53, "t3")], "c0");
        assert_eq!(verdict.status(), CheckStatus::Fail);
    }

    fn check(id: &str, status: CheckStatus, metrics: Value) -> Check {
        Check { id: id.into(), status, metrics: metrics.as_object().cloned().unwrap_or_default(), ..Check::default() }
    }

    #[test]
    fn a_flip_after_verify_rounding_is_informational_only_when_decode_repeats() {
        let noise = |compared, identical| RowNoise { compared, identical, max: 0.1, median: 0.01, first_over: None };
        // A near tie passes whatever else holds.
        assert_eq!(flip_verdict(true, false, &noise(0, 0)), CheckStatus::Pass);
        // Deterministic decode, and drafted rows differed before the flip: rounding.
        assert_eq!(flip_verdict(false, true, &noise(30, 2)), CheckStatus::Info);
        // Plain decode does not repeat: the comparison proves nothing.
        assert_eq!(flip_verdict(false, false, &noise(30, 2)), CheckStatus::Fail);
        // Byte-identical rows up to a large flip: state, not rounding.
        assert_eq!(flip_verdict(false, true, &noise(30, 30)), CheckStatus::Fail);
        // A flip at the first token leaves no pre-flip evidence.
        assert_eq!(flip_verdict(false, true, &noise(0, 0)), CheckStatus::Fail);
    }

    #[test]
    fn small_speculation_flips_are_reported_not_gated() {
        let quality = |margin: f64, identical: u64| {
            let mut q = Quality::default();
            q.checks.push(check("spec_lossless", CheckStatus::Fail, json!({"divergence_margin": margin})));
            q.checks.push(check("c1_c4", CheckStatus::Info, json!({"identical": identical})));
            batch_variant_speculation(&mut q);
            q.checks[0].status
        };
        assert_eq!(quality(0.28, 0), CheckStatus::Info);
        assert_eq!(quality(0.12, 4), CheckStatus::Info);
        assert_eq!(quality(1.2, 0), CheckStatus::Fail);
    }
}
