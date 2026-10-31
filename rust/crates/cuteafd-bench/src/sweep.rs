//! A client-side code-decode sweep at chosen concurrency levels, run by
//! `cuteafd bench smoke` against a launched server after (or instead of) its
//! profile: one untimed warm wave and one measured wave per level, in the
//! order given, then optionally C1 again ("post-load C1": single-stream decode
//! after the widest level has filled workspaces, graphs and pools, which the
//! draft-policy gates read). It runs in the driver's own binary, so both arms
//! of an A/B measure the same workload whatever their server builds contain.
use crate::client::Client;
use crate::panels::common::{aggregate, mapped_counters, mapped_interval, median, plain, wave, Timed};
use crate::panels::speed::CONTENT;
use crate::text::{corpus, nonce};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Output tokens per request (the concurrency panel's width).
pub const TOKENS: u64 = 192;
/// Upper bound for a sweep's per-request output length.
pub const MAX_TOKENS: u64 = 8192;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Point {
    pub c: usize,
    /// `levels`, `post_load` (C1 after the widest level) or `post_load_repeat` (that C1 re-sending the levels C1's prompts).
    pub phase: String,
    pub aggregate_tok_s: f64,
    pub per_request_tok_s: f64,
    pub ttft_s: f64,
    pub errors: usize,
    /// GPU throttle report over the timed wave (`/v1/thermal`; absent without one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fabric: Option<Value>,
    /// Mapped-table counters over the timed wave alone.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mapped_tables: Vec<Value>,
}

/// One cold prefill at a chosen length (after one untimed warm request there).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PrefillPoint {
    /// Requested prompt tokens.
    pub length: u64,
    pub prompt_tokens: u64,
    pub tok_s: f64,
    pub ttft_s: f64,
    /// GPU throttle report over the timed request (`/v1/thermal`; absent without one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fabric: Option<Value>,
    /// Mapped-table counters over the timed request alone (`/v1/stats` before and after).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mapped_tables: Vec<Value>,
}

/// How the prefill points run beyond their lengths.
#[derive(Default)]
pub struct PrefillOptions<'a> {
    /// The timed request re-sends its warm-up's corpus behind a fresh leading
    /// nonce (no prefix-cache reuse): table rows the warm-up read are the same.
    pub repeat: bool,
    /// A cold purge (e.g. `nest drop-caches`), run before each point's warm-up
    /// and again before its timed request.
    pub purge: Option<&'a dyn Fn() -> Result<()>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Sweep {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_content: Option<String>,
    pub points: Vec<Point>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prefill: Vec<PrefillPoint>,
    /// GPU throttle report per phase (`prefill`, `decode`), warm-ups included: the
    /// sustained heat each phase built up, next to the timed points' own windows.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub thermal: serde_json::Map<String, Value>,
}

impl Sweep {
    /// Aggregate decode tok/s at `c` in `phase`.
    pub fn rate(&self, c: usize, phase: &str) -> Option<f64> {
        self.points.iter().find(|p| p.c == c && p.phase == phase).map(|p| p.aggregate_tok_s)
    }
}

/// Parses `1,16` (levels 1..=256, no repeats).
pub fn parse_levels(text: &str) -> Result<Vec<usize>, String> {
    let mut levels = Vec::new();
    for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let level: usize = part.parse().map_err(|_| format!("concurrency level {part:?} is not a number"))?;
        if !(1..=256).contains(&level) {
            return Err(format!("concurrency level {level} is outside 1..=256"));
        }
        if levels.contains(&level) {
            return Err(format!("concurrency level {level} repeats"));
        }
        levels.push(level);
    }
    if levels.is_empty() {
        return Err("no concurrency levels".into());
    }
    Ok(levels)
}

/// Called once, just before the sweep's first timed request (after its first
/// untimed warm-up): `cuteafd bench smoke` opens an entry's measurement window
/// (e.g. a profiler's capture range) there.
pub type OnTimed<'a> = Option<Box<dyn FnOnce() -> Result<()> + 'a>>;

fn timed(on_timed: &mut OnTimed) -> Result<()> {
    on_timed.take().map_or(Ok(()), |open| open())
}

/// The exact request prompts of one C1 measurement, kept so a later point can re-send them.
#[derive(Debug, Clone, PartialEq)]
struct Prompts {
    warm: Vec<String>,
    timed: Vec<String>,
}

impl Prompts {
    /// Fresh nonces: one distinct prompt per request of the warm-up and timed waves.
    fn fresh(c: usize) -> Prompts {
        let wave = || (0..c).map(|_| format!("[{}] {}", nonce(), CONTENT[0].1)).collect();
        Prompts { warm: wave(), timed: wave() }
    }
}

/// Measures concurrency `c`; `repeat` re-sends those prompts instead of drawing fresh ones.
/// Returns the point and the prompts used.
fn measure(client: &Client, c: usize, phase: &str, repeat: Option<&Prompts>, tokens: u64, on_timed: &mut OnTimed)
    -> Result<(Point, Prompts)> {
    let prompts = repeat.cloned().unwrap_or_else(|| Prompts::fresh(c));
    let batch = |texts: &[String]| -> Vec<Result<Timed>> {
        wave(client, texts.iter().map(|text| plain(client, text, tokens)).collect())
    };
    // A repeat only means something if nothing came from the prefix cache.
    let uncached = |results: Vec<Result<Timed>>| -> Result<Vec<Result<Timed>>> {
        if repeat.is_some() {
            if let Some(hit) = results.iter().flatten().map(|r| r.chat.timing.cached_tokens).find(|&n| n > 0) {
                bail!("C{c} {phase}: a repeated request reported {hit} cached tokens; post_load_repeat only means \
                    something with the prefix cache disabled (PREFIX_CACHE_ENTRIES=0)");
            }
        }
        Ok(results)
    };
    // Prime each width first: graphs, workspaces and mapped tables.
    uncached(batch(&prompts.warm))?.into_iter().collect::<Result<Vec<_>>>().with_context(|| format!("C{c} warm-up"))?;
    timed(on_timed)?;
    let mark = client.thermal_mark();
    let fabric_mark = client.fabric_mark();
    let tables = mapped_counters(client);
    let results = uncached(batch(&prompts.timed))?;
    let mapped_tables = mapped_interval(&tables, &mapped_counters(client));
    let fabric = client.fabric_since(fabric_mark, &format!("{phase}/c{c}"), true).map(|p| serde_json::json!(p));
    let thermal = client.thermal_since(mark);
    let errors = results.iter().filter(|r| r.is_err()).count();
    let ok: Vec<Timed> = results.into_iter().filter_map(Result::ok).collect();
    if ok.is_empty() {
        bail!("C{c} {phase}: every request failed");
    }
    let mut per: Vec<f64> = ok.iter().map(|r| r.chat.timing.decode_tok_s()).collect();
    let mut ttft: Vec<f64> = ok.iter().map(|r| r.chat.timing.ttft_s).collect();
    let point = Point { c, phase: phase.into(), aggregate_tok_s: aggregate(&ok), per_request_tok_s: median(&mut per),
        ttft_s: median(&mut ttft), errors, thermal, fabric, mapped_tables };
    Ok((point, prompts))
}

/// Cold prefill at each length: fit tokens per reference token from two short
/// requests, then per length one untimed warm request and one measured request
/// with a fresh prefix (it must reuse no cached tokens). With `options.repeat`
/// the measured request re-sends the warm-up's corpus behind a fresh nonce;
/// with `options.purge` the purge runs before the warm-up and again before the
/// measured request, so both start from the same cold state.
fn prefill(client: &Client, lengths: &[u64], max_context: u64, options: &PrefillOptions, on_timed: &mut OnTimed)
    -> Result<Vec<PrefillPoint>> {
    let ask = |words: usize, seed: u64| -> Result<crate::client::Chat> {
        client.chat(plain(client, &format!("[{}] Summarize in one word.\n\n{}", nonce(), corpus(seed, words)), 1), None)
    };
    let (a, b) = (ask(300, 101)?, ask(900, 101)?);
    let slope = (b.timing.prompt_tokens as f64 - a.timing.prompt_tokens as f64) / 600.0;
    let intercept = a.timing.prompt_tokens as f64 - slope * 300.0;
    let (slope, intercept) = if slope.is_finite() && slope > 0.3 { (slope, intercept) } else { (1.0, 20.0) };
    let purge = || options.purge.map_or(Ok(()), |purge| purge());
    let mut points = Vec::new();
    for &length in lengths {
        let target = length.min(max_context.saturating_sub(64)) as f64;
        let words = ((target - intercept) / slope).max(64.0) as usize;
        let (warm_seed, timed_seed) = prefill_seeds(length, options.repeat);
        purge().with_context(|| format!("{length}-token prefill purge before warm-up"))?;
        ask(words, warm_seed).with_context(|| format!("{length}-token prefill warm-up"))?;
        timed(on_timed)?;
        purge().with_context(|| format!("{length}-token prefill purge before the timed request"))?;
        let mark = client.thermal_mark();
        let fabric_mark = client.fabric_mark();
        let tables = mapped_counters(client);
        let chat = ask(words, timed_seed).with_context(|| format!("{length}-token prefill"))?;
        let mapped_tables = mapped_interval(&tables, &mapped_counters(client));
        let thermal = client.thermal_since(mark);
        let fabric = client.fabric_since(fabric_mark, &format!("prefill/{length}"), true).map(|p| serde_json::json!(p));
        let t = chat.timing;
        anyhow::ensure!(t.cached_tokens == 0, "{length}-token cold prefill reused {} cached tokens", t.cached_tokens);
        points.push(PrefillPoint { length, prompt_tokens: t.prompt_tokens, tok_s: t.prefill_tok_s(), ttft_s: t.ttft_s,
            thermal, fabric, mapped_tables });
    }
    Ok(points)
}

/// Corpus seeds of one length's warm-up and timed request: the same text when
/// `repeat` (only the leading nonce differs), different texts otherwise.
fn prefill_seeds(length: u64, repeat: bool) -> (u64, u64) {
    (200 + length, if repeat { 200 + length } else { 300 + length })
}

/// Parses `8192,32768` (prompt lengths 256..=1M tokens).
pub fn parse_lengths(text: &str) -> Result<Vec<u64>, String> {
    let lengths: Vec<u64> = text.split(',').map(str::trim).filter(|p| !p.is_empty())
        .map(|p| p.parse::<u64>().map_err(|_| format!("prefill length {p:?} is not a number")))
        .collect::<Result<_, _>>()?;
    if lengths.is_empty() || lengths.iter().any(|l| !(256..=1 << 20).contains(l)) {
        return Err("prefill lengths must be 256..=1048576 tokens".into());
    }
    Ok(lengths)
}

/// Runs the sweep against `url`; `api_key` is the server's bearer key.
pub fn run(url: &str, api_key: Option<&str>, levels: &[usize], post_load_c1: bool, post_load_repeat: bool,
    prefill_lengths: &[u64]) -> Result<Sweep> {
    run_windowed(url, api_key, levels, post_load_c1, post_load_repeat, prefill_lengths, &PrefillOptions::default(), false,
        TOKENS, None)
}

/// [`run`], calling `on_timed` once just before the first timed request. `tokens`
/// is each decode request's output length ([`TOKENS`] by default; longer runs give
/// a profiled wave more decode steps at every width).
#[allow(clippy::too_many_arguments)]
pub fn run_windowed(url: &str, api_key: Option<&str>, levels: &[usize], post_load_c1: bool,
    post_load_repeat: bool, prefill_lengths: &[u64], prefill_options: &PrefillOptions, fabric: bool, tokens: u64,
    mut on_timed: OnTimed) -> Result<Sweep> {
    if !(1..=MAX_TOKENS).contains(&tokens) {
        bail!("sweep tokens must be 1..={MAX_TOKENS}, got {tokens}");
    }
    if post_load_repeat && !(post_load_c1 && levels.contains(&1)) {
        bail!("post_load_repeat needs post_load_c1 and a levels list that includes 1 (it replays the levels C1)");
    }
    let key = api_key.map(cuteafd_api::openai::auth::ApiKey::new).transpose().map_err(anyhow::Error::msg)?;
    let mut client = Client::new(url, None, Arc::new(AtomicBool::new(false))).with_api_key(key).with_fabric(fabric);
    let _subscription = client.fabric_subscription();
    let record = client.discover().context("sweep: /v1/models")?;
    let mut sweep = Sweep { prompt_content: Some(crate::text::PROMPT_CONTENT.into()), ..Sweep::default() };
    if !prefill_lengths.is_empty() {
        let max_context = record["max_context_tokens"].as_u64().unwrap_or(8192);
        let mark = client.thermal_mark();
        sweep.prefill = prefill(&client, prefill_lengths, max_context, prefill_options, &mut on_timed)?;
        if let Some(window) = client.thermal_since(mark) { sweep.thermal.insert("prefill".into(), window); }
    }
    let mark = client.thermal_mark();
    let mut c1_prompts = None;
    for &c in levels {
        let (point, prompts) = measure(&client, c, "levels", None, tokens, &mut on_timed)?;
        if c == 1 && post_load_repeat { c1_prompts = Some(prompts); }
        sweep.points.push(point);
    }
    if post_load_c1 {
        let (phase, repeat) = if post_load_repeat { ("post_load_repeat", c1_prompts.as_ref()) } else { ("post_load", None) };
        sweep.points.push(measure(&client, 1, phase, repeat, tokens, &mut on_timed)?.0);
    }
    if !levels.is_empty() || post_load_c1 {
        if let Some(window) = client.thermal_since(mark) { sweep.thermal.insert("decode".into(), window); }
    }
    for (phase, window) in &sweep.thermal {
        eprintln!("sweep {phase} GPU throttle: {}", cuteafd_core::thermal::describe(window));
    }
    Ok(sweep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_parse_in_order_and_refuse_nonsense() {
        assert_eq!(parse_levels("1,16").unwrap(), vec![1, 16]);
        assert_eq!(parse_levels(" 16 , 8,1").unwrap(), vec![16, 8, 1]);
        for bad in ["", "0", "1,1", "x", "300"] {
            assert!(parse_levels(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn prefill_lengths_parse_and_bound() {
        assert_eq!(parse_lengths("8192,32768").unwrap(), vec![8192, 32768]);
        for bad in ["", "100", "x", "2000000"] {
            assert!(parse_lengths(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn repeat_reuses_prompts_and_wide_levels_stay_fresh() {
        let c1 = Prompts::fresh(1);
        assert_ne!(c1.warm, c1.timed);
        // The repeat takes the stored prompts verbatim.
        let again = Some(&c1).cloned().unwrap();
        assert_eq!(again, c1);
        assert_eq!(again.warm[0].as_bytes(), c1.warm[0].as_bytes());
        assert_eq!(again.timed[0].as_bytes(), c1.timed[0].as_bytes());
        // Wider levels draw distinct nonces per request and per wave.
        let wide = Prompts::fresh(16);
        let all: std::collections::HashSet<&String> = wide.warm.iter().chain(&wide.timed).collect();
        assert_eq!(all.len(), 32);
        assert_ne!(Prompts::fresh(1), c1);
    }

    #[test]
    fn repeat_needs_post_load_c1_and_level_one() {
        for (levels, c1) in [(&[1usize, 16][..], false), (&[16][..], true)] {
            let err = run("http://127.0.0.1:1", None, levels, c1, true, &[]).unwrap_err().to_string();
            assert!(err.contains("post_load_repeat needs post_load_c1"), "{err}");
        }
    }

    #[test]
    fn prefill_repeat_resends_the_warm_up_corpus() {
        let (warm, timed) = prefill_seeds(8192, true);
        assert_eq!(warm, timed);
        assert_eq!(corpus(warm, 512), corpus(timed, 512));
        let (warm, timed) = prefill_seeds(8192, false);
        assert_ne!(corpus(warm, 512), corpus(timed, 512));
        assert_ne!(prefill_seeds(8192, true), prefill_seeds(65536, true));
    }

    #[test]
    fn on_timed_fires_once() {
        let count = std::cell::Cell::new(0);
        let mut hook: OnTimed = Some(Box::new(|| { count.set(count.get() + 1); Ok(()) }));
        timed(&mut hook).unwrap();
        timed(&mut hook).unwrap();
        assert_eq!(count.get(), 1);
        let mut failing: OnTimed = Some(Box::new(|| anyhow::bail!("refused")));
        assert!(timed(&mut failing).is_err());
        timed(&mut None).unwrap();
    }

    #[test]
    fn rate_looks_up_level_and_phase() {
        let sweep = Sweep { points: vec![
            Point { c: 1, phase: "levels".into(), aggregate_tok_s: 50.0, ..Point::default() },
            Point { c: 1, phase: "post_load".into(), aggregate_tok_s: 48.0, ..Point::default() }], ..Sweep::default() };
        assert_eq!(sweep.rate(1, "levels"), Some(50.0));
        assert_eq!(sweep.rate(1, "post_load"), Some(48.0));
        assert_eq!(sweep.rate(16, "levels"), None);
    }
}
