//! Decode concurrency over one retained long prefix, not repeated cold prefills.
use super::{
    common::{self, aggregate, median, table},
    Ctx, Panel, Rates,
};
use crate::{
    client::Client,
    report::{Capacity, ServerInfo},
    text::{corpus, nonce},
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub static CONTEXT_LOAD: ContextLoad = ContextLoad;
pub struct ContextLoad;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Options {
    pub lengths: Vec<u64>,
    pub levels: Vec<usize>,
    /// Additional fresh suffix tokens, targeted using the panel's reference-token calibration.
    #[serde(default)]
    pub suffix_tokens: u64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            lengths: vec![32768, 131072],
            levels: vec![1, 8, 16],
            suffix_tokens: 0,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<(), String> {
        crate::sweep::parse_lengths(
            &self
                .lengths
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )?;
        crate::sweep::parse_levels(
            &self
                .levels
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )?;
        if self.suffix_tokens > 1 << 20 {
            return Err("context suffix must be 0..=1048576 tokens".into());
        }
        Ok(())
    }
}

/// Clap keeps the CSV as one argument; the parser validates the whole list.
pub fn parse_lengths_arg(text: &str) -> Result<String, String> {
    crate::sweep::parse_lengths(text)?;
    Ok(text.to_string())
}
pub fn parse_levels_arg(text: &str) -> Result<String, String> {
    crate::sweep::parse_levels(text)?;
    Ok(text.to_string())
}

fn base(run: &str, length: u64, words: usize) -> Value {
    json!([{"role": "system", "content": format!("[{run}-{length}] You are a helpful assistant. Reference notes:\n{}", corpus(4242, words))},
        {"role": "user", "content": "Read the reference notes. Reply with the single word OK."}])
}
pub(crate) fn body(configuration: &crate::report::Configuration, messages: Value, output: u64) -> Value {
    crate::request::thinking_off(configuration, json!({"messages": messages, "max_tokens": output, "temperature": 0, "thinking": {"type": "disabled"}}))
}
pub(crate) fn request(configuration: &crate::report::Configuration, base: &Value, run: &str, phase: &str, c: usize, i: usize, output: u64, suffix_tokens: usize) -> Value {
    let mut messages = base.as_array().unwrap().clone();
    let mut content = format!("[{run}-{phase}-C{c}-{i}] Ignore the notes. Write a complete Python LRU cache module with TTL, a background janitor and pytest tests. Output only code.");
    if suffix_tokens > 0 {
        // The unique tag precedes the corpus, so neither another request nor the warm wave can cache it.
        let seed = (c as u64 * 257 + i as u64).wrapping_add(if phase == "warm" { 0 } else { 1 << 32 });
        content.push_str("\n\nFresh reference notes:\n");
        content.push_str(&corpus(seed, suffix_tokens));
    }
    messages.push(json!({"role": "user", "content": content}));
    body(configuration, json!(messages), output)
}

/// A client must check the returned evidence too: an older server may ignore the request option.
pub(crate) fn validate_suffix(record: &Value, requested: u64) -> Result<(), String> {
    if requested == 0 { return Ok(()); }
    let error = || format!("server does not support context suffix: expected every measured request to report uncached_rows >= {requested}");
    let points = record["points"].as_array().filter(|p| !p.is_empty()).ok_or_else(error)?;
    for point in points {
        let requests = point["requests"].as_array().filter(|r| !r.is_empty()).ok_or_else(error)?;
        if point["status"] != "ok" || point["suffix_tokens"].as_u64() != Some(requested)
            || point["c"].as_u64() != Some(requests.len() as u64)
            || requests.iter().any(|r| r["uncached_rows"].as_u64().is_none_or(|rows| rows < requested)) {
            return Err(error());
        }
    }
    Ok(())
}

fn fit(
    pool: Option<u64>,
    slots: Option<u64>,
    c: usize,
    base: u64,
    extra: u64,
    page: u64,
) -> (u64, u64, Option<String>) {
    let round = |n: u64| n.saturating_add(page - 1) / page * page;
    let copied = round(base.saturating_add(extra)).saturating_mul(c as u64);
    let shared =
        round(base).saturating_add(round(extra.saturating_add(page)).saturating_mul(c as u64));
    let why = if slots.is_some_and(|n| (c as u64) > n) {
        Some(format!(
            "C{c} exceeds admitted request slots {}",
            slots.unwrap()
        ))
    } else {
        match pool {
            None => Some("admitted KV pool capacity unavailable".into()),
            Some(n) if shared > n => Some(format!(
                "shared estimate {shared} tokens exceeds {n}-token KV pool"
            )),
            _ => None,
        }
    };
    (copied, shared, why)
}

fn occupancy(client: &Client) -> Value {
    client.pool_observation()
}

/// Observe pool use without adding stats reads to the request hot path. The
/// condition variable ends the sampler immediately when the wave completes.
/// The wave runs a retained base, which re-renders the model's own answer.
fn measured_wave(client: &Client, bodies: Vec<Value>) -> (Vec<Result<common::Timed>>, Vec<Value>) {
    let done = Arc::new((Mutex::new(false), Condvar::new()));
    std::thread::scope(|scope| {
        let stop = done.clone();
        let observer = scope.spawn(move || {
            let mut samples = Vec::new();
            loop {
                samples.push(occupancy(client));
                let (lock, wake) = &*stop;
                let guard = lock.lock().unwrap();
                if *guard {
                    break;
                }
                let (guard, _) = wake.wait_timeout(guard, Duration::from_secs(1)).unwrap();
                if *guard {
                    break;
                }
            }
            samples
        });
        let results = common::wave_output_dependent(client, bodies);
        *done.0.lock().unwrap() = true;
        done.1.notify_all();
        (results, observer.join().unwrap_or_default())
    })
}

fn sharing(before: &Value, samples: &[Value], c: usize, base_pages: u64) -> &'static str {
    let used = |v: &Value| {
        Some(
            v["prefix_cache"]["pages"]
                .as_u64()?
                .saturating_sub(v["prefix_cache"]["pages_free"].as_u64()?),
        )
    };
    let Some(initial) = used(before) else {
        return "unknown";
    };
    for sample in samples {
        if sample["active"].as_u64() != Some(c as u64) {
            continue;
        }
        let Some(now) = used(sample) else {
            continue;
        };
        let growth = now.saturating_sub(initial);
        if c > 1
            && growth < base_pages.saturating_mul((c - 1) as u64) / 2
            && sample["prefix_cache"]["pages_shared"]
                .as_u64()
                .is_some_and(|n| n >= base_pages)
        {
            return "shared_device_pages";
        }
        if growth >= base_pages.saturating_mul(c as u64) {
            return "copied_pages";
        }
    }
    "unknown"
}

impl Panel for ContextLoad {
    fn id(&self) -> &'static str {
        "context_load"
    }
    fn title(&self) -> &'static str {
        "Long-context concurrency"
    }
    fn description(&self) -> &'static str {
        "Decode at C1/C8/C16 over one retained 32K/128K base. One cold prefill per length; one untimed warm batch per concurrency. Pool estimates and observed cache reuse accompany every point."
    }
    fn estimate_s(&self, rates: &Rates, info: &ServerInfo) -> f64 {
        let limit = Capacity::from_settings(info).max_context.unwrap_or(131072);
        [32768u64, 131072]
            .into_iter()
            .map(|l| rates.seconds(l.min(limit) as f64, 1.0) + 6.0 * rates.seconds(96.0, 256.0))
            .sum::<f64>()
            + rates.seconds(1200.0, 2.0)
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let options = ctx.context_load;
        options.validate().map_err(anyhow::Error::msg)?;
        let output = 256.min(ctx.max_output);
        let suffix = 128 + options.suffix_tokens;
        let room = output + suffix + 256;
        let capacity = ctx
            .baseline
            .and_then(|b| b.card.capacity.clone())
            .unwrap_or_else(|| Capacity::from_settings(ctx.info));
        let initial = occupancy(ctx.client);
        let pages = initial["prefix_cache"]["pages"].as_u64();
        let resolved = crate::context::kv();
        let pool = resolved
            .as_ref()
            .and_then(|k| k.tokens)
            .or(capacity.kv_tokens);
        let page = initial["console"]["kv"]["tokens_per_page"]
            .as_u64()
            .filter(|p| *p > 0)
            .or_else(|| {
                pool.zip(pages.filter(|p| *p > 0))
                    .map(|(t, p)| (t / p).max(1))
            })
            .unwrap_or(256);
        let slots = capacity.max_requests.or_else(|| {
            common::setting(ctx.info, &["concurrency", "max-sequences"])
                .and_then(|s| s.parse().ok())
        });
        let run = nonce();
        let mut points = Vec::new();
        let mut bases = Vec::new();
        // As in the cold-prefill sweep, two short requests calibrate corpus
        // reference-token density before the one long base is built.
        let a = ctx.client.chat(body(&ctx.info.configuration, base(&run, 300, 300), 1), None)?;
        let b = ctx.client.chat(body(&ctx.info.configuration, base(&run, 900, 900), 1), None)?;
        let slope = (b.timing.prompt_tokens as f64 - a.timing.prompt_tokens as f64) / 600.0;
        let slope = if slope.is_finite() && slope > 0.3 {
            slope
        } else {
            1.0
        };
        let intercept = a.timing.prompt_tokens as f64 - slope * 300.0;
        let suffix_words = (options.suffix_tokens as f64 / slope).ceil() as usize;
        let mut lengths = Vec::new();
        for &requested in &options.lengths {
            let length = requested.min(ctx.max_context.saturating_sub(room));
            if length < 256 {
                points.push(json!({"context": requested, "status": "skipped", "reason": "max_context leaves no room for suffix and output"}));
                continue;
            }
            if lengths.contains(&length) {
                continue;
            }
            lengths.push(length);
            let (_, _, why) = fit(pool, None, 1, length, output + suffix, page);
            if let Some(reason) = why {
                for &c in &options.levels {
                    points.push(json!({"context": length, "requested_context": requested, "c": c, "status": "skipped", "reason": reason}));
                }
                continue;
            }
            ctx.progress.step(
                bases.len() as f64 / options.lengths.len() as f64,
                format!("{length}-token cold base"),
            );
            let words = ((length as f64 - intercept) / slope).max(8.0) as usize;
            let messages = base(&run, length, words);
            let cold = match ctx.client.chat(body(&ctx.info.configuration, messages.clone(), 1), None) {
                Ok(chat) => chat,
                Err(error) => {
                    for &c in &options.levels {
                        points.push(json!({"context": length, "c": c, "status": "failed", "reason": format!("base prefill: {error:#}")}));
                    }
                    continue;
                }
            };
            let actual = cold.timing.prompt_tokens;
            bases.push(json!({"context": length, "requested_context": requested, "prompt_tokens": actual,
                "cold_ttft_s": cold.timing.ttft_s, "cached_tokens": cold.timing.cached_tokens, "pool_after": occupancy(ctx.client)}));
            let mut retained = messages.as_array().unwrap().clone();
            retained.push(json!({"role": "assistant", "content": cold.content}));
            let retained = json!(retained);
            let mut cache_failed = false;
            for &c in &options.levels {
                ctx.client.check()?;
                let (copied, shared, why) = fit(pool, slots, c, actual, output + suffix, page);
                let mut point = json!({"context": length, "requested_context": requested, "base_tokens": actual, "c": c,
                    "cold_ttft_s": cold.timing.ttft_s, "pool_tokens": pool, "tokens_per_page": page, "max_tokens": output, "suffix_allowance_tokens": suffix, "suffix_tokens": options.suffix_tokens,
                    "copied_estimate_tokens": copied, "shared_estimate_tokens": shared,
                    "admission_estimate": "shared full pages; one base plus per-request suffix, output and partial tail; state marks observed separately"});
                let why = why.or_else(|| {
                    cache_failed.then(|| {
                        "previous batch did not reuse the base; no more cold long batches".into()
                    })
                });
                if let Some(reason) = why {
                    point["status"] = json!("skipped");
                    point["reason"] = json!(reason);
                    points.push(point);
                    continue;
                }
                ctx.progress.step(
                    bases.len().saturating_sub(1) as f64 / options.lengths.len() as f64,
                    format!("{length} tokens C{c}: warm then timed"),
                );
                let bodies = |phase| {
                    (0..c)
                        .map(|i| request(&ctx.info.configuration, &retained, &run, phase, c, i, output, suffix_words))
                        .collect()
                };
                // Every request of a retained-base batch re-renders the
                // model's own answer: model-determined, so not comparable
                // between arms.
                let warm = common::wave_output_dependent(ctx.client, bodies("warm"));
                let warm_errors: Vec<_> = warm
                    .iter()
                    .filter_map(|r| r.as_ref().err().map(|e| format!("{e:#}")))
                    .collect();
                point["warm_errors"] = json!(warm_errors);
                let warm_reused = warm.iter().all(|r| {
                    r.as_ref().is_ok_and(|r| {
                        r.chat.timing.cached_tokens >= actual.saturating_sub(page.max(actual / 50))
                    })
                });
                point["warm_cache_reuse"] = json!(warm_reused);
                point["warm_requests"] = json!(warm
                    .iter()
                    .filter_map(|r| r.as_ref().ok())
                    .map(|r| &r.chat.timing)
                    .collect::<Vec<_>>());
                if !warm_errors.is_empty() || !warm_reused {
                    cache_failed = !warm_reused;
                    point["status"] = json!("failed");
                    point["reason"] =
                        json!("warm batch rejected requests or did not reuse the base");
                    points.push(point);
                    continue;
                }
                // Never retry failures sequentially: that would pretend a queued
                // or rejected wide batch achieved the requested concurrency.
                let before = occupancy(ctx.client);
                let (results, samples) = measured_wave(ctx.client, bodies("timed"));
                let errors: Vec<_> = results
                    .iter()
                    .filter_map(|r| r.as_ref().err().map(|e| format!("{e:#}")))
                    .collect();
                let ok: Vec<_> = results.into_iter().filter_map(Result::ok).collect();
                let reused = ok.iter().all(|r| {
                    r.chat.timing.cached_tokens >= actual.saturating_sub(page.max(actual / 50))
                });
                let mut per: Vec<_> = ok.iter().map(|r| r.chat.timing.decode_tok_s()).collect();
                let mut ttft: Vec<_> = ok.iter().map(|r| r.chat.timing.ttft_s).collect();
                let mut cached: Vec<_> = ok
                    .iter()
                    .map(|r| r.chat.timing.cached_tokens as f64)
                    .collect();
                let mut prompts: Vec<_> = ok
                    .iter()
                    .map(|r| r.chat.timing.prompt_tokens as f64)
                    .collect();
                point["status"] = json!(if ok.len() == c && reused {
                    "ok"
                } else {
                    "failed"
                });
                point["cache_reuse"] = json!(reused && !ok.is_empty());
                cache_failed = !reused;
                point["errors"] = json!(errors);
                point["aggregate_tok_s"] = json!(aggregate(&ok));
                point["per_request_tok_s"] = json!(median(&mut per));
                point["ttft_s"] = json!(median(&mut ttft));
                point["cached_tokens"] = json!(median(&mut cached));
                point["prompt_tokens"] = json!(median(&mut prompts));
                let mut uncached: Vec<_> = ok.iter().map(|r|
                    r.chat.timing.prompt_tokens.saturating_sub(r.chat.timing.cached_tokens) as f64).collect();
                point["uncached_rows"] = json!(median(&mut uncached));
                point["requests"] = json!(ok.iter().map(|r| json!({"timing": r.chat.timing, "decode_tok_s": r.chat.timing.decode_tok_s(), "sent_s": r.sent,
                    "uncached_rows": r.chat.timing.prompt_tokens.saturating_sub(r.chat.timing.cached_tokens)})).collect::<Vec<_>>());
                point["restore_pages"] = json!(sharing(&before, &samples, c, actual / page));
                point["observed_peak_active"] =
                    json!(samples.iter().filter_map(|s| s["active"].as_u64()).max());
                point["full_concurrency_observed"] = json!(samples
                    .iter()
                    .any(|s| s["active"].as_u64() == Some(c as u64)));
                point["observed_queued"] = json!(samples
                    .iter()
                    .filter_map(|s| s["queued"]
                        .as_u64()
                        .or_else(|| s["console"]["queued"].as_u64()))
                    .max());
                point["pool_before"] = before;
                point["pool_samples"] = json!(samples);
                point["pool_after"] = occupancy(ctx.client);
                points.push(point);
                ctx.progress.partial(record(&bases, &points));
            }
        }
        let result = record(&bases, &points);
        validate_suffix(&result, options.suffix_tokens).map_err(anyhow::Error::msg)?;
        Ok(result)
    }
}

fn record(bases: &[Value], points: &[Value]) -> Value {
    let rows = points
        .iter()
        .map(|p| {
            vec![
                p["context"].clone(),
                p["c"].clone(),
                p["status"].clone(),
                p["aggregate_tok_s"].clone(),
                p["per_request_tok_s"].clone(),
                json!(p["ttft_s"].as_f64().map(|v| v * 1e3)),
                p["cached_tokens"].clone(),
                p["prompt_tokens"].clone(),
                p["suffix_tokens"].clone(),
                p["uncached_rows"].clone(),
                json!(p["cold_ttft_s"].as_f64().map(|v| v * 1e3)),
                p["restore_pages"].clone(),
                p["reason"].clone(),
            ]
        })
        .collect();
    json!({"bases": bases, "points": points, "table": table(&["context", "C", "status", "aggregate tok/s", "median tok/s", "TTFT ms", "cached tokens", "prompt tokens", "suffix tokens", "uncached rows", "cold TTFT ms", "restore pages", "reason"], rows)})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn common_base_and_distinct_suffixes() {
        let b = base("test", 32768, 100);
        assert_eq!(b, base("test", 32768, 100));
        let a = request(&crate::report::Configuration::default(), &b, "test", "timed", 8, 0, 256, 0);
        let c = request(&crate::report::Configuration::default(), &b, "test", "timed", 8, 1, 256, 0);
        assert_eq!(a["messages"][0], c["messages"][0]);
        assert_eq!(a["messages"][1], c["messages"][1]);
        assert_ne!(a["messages"][2], c["messages"][2]);
        assert_ne!(a, request(&crate::report::Configuration::default(), &b, "test", "warm", 8, 0, 256, 0));
    }
    #[test]
    fn zero_suffix_preserves_the_original_request_and_retained_base() {
        let configuration = crate::report::Configuration::default();
        let mut retained = base("test", 8192, 100).as_array().unwrap().clone();
        retained.push(json!({"role": "assistant", "content": "OK"}));
        let retained = json!(retained);
        let request = request(&configuration, &retained, "test", "timed", 16, 0, 256, 0);
        let messages = request["messages"].as_array().unwrap();
        assert_eq!(&messages[..3], retained.as_array().unwrap());
        assert_eq!(messages[3], json!({"role": "user", "content":
            "[test-timed-C16-0] Ignore the notes. Write a complete Python LRU cache module with TTL, a background janitor and pytest tests. Output only code."}));
        assert_eq!(Options::default().suffix_tokens, 0);
    }
    #[test]
    fn fresh_suffixes_are_unique_per_request_and_wave_after_the_same_base() {
        let configuration = crate::report::Configuration::default();
        let retained = json!([{"role": "system", "content": "retained notes"},
            {"role": "user", "content": "Read them"}, {"role": "assistant", "content": "OK"}]);
        let mut suffixes = std::collections::HashSet::new();
        for phase in ["warm", "timed"] {
            for i in 0..16 {
                let request = request(&configuration, &retained, "test", phase, 16, i, 256, 512);
                let messages = request["messages"].as_array().unwrap();
                assert_eq!(&messages[..3], retained.as_array().unwrap());
                let content = messages[3]["content"].as_str().unwrap();
                assert!(content.starts_with(&format!("[test-{phase}-C16-{i}]")));
                let notes = content.split_once("Fresh reference notes:\n").unwrap().1;
                assert!(notes.len() > corpus(0, 32).len());
                assert!(suffixes.insert(content.to_string()));
            }
        }
        assert_eq!(suffixes.len(), 32);
    }
    #[test]
    fn suffix_evidence_fails_closed_for_old_or_underfilled_servers() {
        let good = json!({"points": [{"status": "ok", "c": 2, "suffix_tokens": 512,
            "requests": [{"uncached_rows": 570}, {"uncached_rows": 512}]}]});
        assert!(validate_suffix(&good, 512).is_ok());
        assert!(validate_suffix(&json!({}), 0).is_ok());
        for bad in [json!({}), json!({"points": []}),
            json!({"points": [{"status": "ok", "c": 1, "requests": [{}]}]}),
            json!({"points": [{"status": "ok", "c": 1, "suffix_tokens": 512, "requests": [{"uncached_rows": 57}]}]}),
            json!({"points": [{"status": "ok", "c": 2, "suffix_tokens": 512, "requests": [{"uncached_rows": 570}, {}]}]})] {
            assert!(validate_suffix(&bad, 512).unwrap_err().contains("server does not support context suffix"));
        }
    }
    #[test]
    fn pool_fit_uses_shared_not_copied_base() {
        let (copied, shared, skip) = fit(Some(2 << 20), Some(16), 16, 131072, 384, 256);
        assert!(copied > 2 << 20);
        assert!(shared < 2 << 20);
        assert!(skip.is_none());
        assert!(fit(Some(32768), Some(16), 16, 32768, 384, 256).2.is_some());
        assert!(fit(Some(2 << 20), Some(8), 16, 32768, 384, 256).2.is_some());
        assert!(fit(None, Some(16), 1, 32768, 384, 256).2.is_some());
    }
    #[test]
    fn options_validate() {
        assert!(Options::default().validate().is_ok());
        assert!(Options { suffix_tokens: 512, ..Options::default() }.validate().is_ok());
        assert!(Options { suffix_tokens: 1048577, ..Options::default() }.validate().is_err());
        assert_eq!(parse_lengths_arg("32768,131072").unwrap(), "32768,131072");
        assert_eq!(parse_levels_arg("1,8,16").unwrap(), "1,8,16");
        assert!(parse_levels_arg("8,8").is_err());
        assert!(parse_lengths_arg("255").is_err());
        for lengths in [vec![], vec![255], vec![1048577]] {
            assert!(Options {
                lengths,
                ..Options::default()
            }
            .validate()
            .is_err());
        }
        for levels in [vec![], vec![0], vec![257], vec![1, 1]] {
            assert!(Options {
                levels,
                ..Options::default()
            }
            .validate()
            .is_err());
        }
    }
    #[test]
    fn occupancy_requires_full_concurrency_to_claim_sharing() {
        let before = json!({"prefix_cache": {"pages": 10000, "pages_free": 9000}});
        let samples = vec![
            json!({"active": 16, "prefix_cache": {"pages": 10000, "pages_free": 8970, "pages_shared": 500}}),
        ];
        assert_eq!(sharing(&before, &samples, 16, 500), "shared_device_pages");
        assert_eq!(sharing(&before, &samples, 8, 500), "unknown");
        assert_eq!(sharing(&before, &[], 16, 500), "unknown");
        let copies = vec![
            json!({"active": 16, "prefix_cache": {"pages": 10000, "pages_free": 1000, "pages_shared": 500}}),
        ];
        assert_eq!(sharing(&before, &copies, 16, 500), "copied_pages");
    }
}
