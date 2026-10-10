//! Helpers the panels share: request bodies, prompt sizing, settings,
//! concurrent requests with absolute timing, server counters.
use super::Ctx;
use crate::client::{Chat, Client};
use crate::report::ServerInfo;
use crate::text::{filler, nonce};
use anyhow::Result;
use serde_json::{json, Value};
use std::time::Instant;

pub fn messages(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

/// Greedy, thinking off.
pub fn plain(text: &str, max_tokens: u64) -> Value {
    json!({"messages": messages(text), "max_tokens": max_tokens, "temperature": 0, "thinking": {"type": "disabled"}})
}

pub fn setting<'a>(info: &'a ServerInfo, names: &[&str]) -> Option<&'a str> {
    info.configuration.settings.iter().find(|s| names.contains(&s.name.as_str())).and_then(|s| s.value.as_deref())
}

/// Sequences the server decodes at once.
pub fn concurrency(info: &ServerInfo) -> usize {
    setting(info, &["concurrency", "max-sequences"]).and_then(|v| v.parse().ok()).unwrap_or(8)
}

/// Tokens per filler word, from the baseline's 8K prompt (about 1.3 otherwise).
pub fn tokens_per_word(ctx: &Ctx<'_>) -> f64 {
    ctx.baseline.and_then(|b| b.card.prefill.as_ref()).map(|p| p.prompt_tokens as f64 / 6200.0)
        .filter(|r| r.is_finite() && *r > 0.5 && *r < 4.0).unwrap_or(1.33)
}

/// A unique prompt of about `tokens` tokens: nonce, filler, then `question`.
pub fn sized_prompt(ctx: &Ctx<'_>, seed: u64, tokens: u64, question: &str) -> String {
    let words = ((tokens.saturating_sub(40)) as f64 / tokens_per_word(ctx)).max(8.0) as usize;
    format!("[{}] Read the notes below.\n\n{}\n\n{question}", nonce(), filler(seed, words))
}

/// One request of a concurrent wave, on the wave's clock.
#[derive(Debug, Clone)]
pub struct Timed {
    pub chat: Chat,
    /// Seconds from the wave start to the send.
    pub sent: f64,
}

impl Timed {
    pub fn first(&self) -> f64 {
        self.sent + self.chat.timing.ttft_s
    }
    pub fn last(&self) -> f64 {
        self.first() + self.chat.timing.decode_s
    }
}

/// Sends every body at once (one thread each) and waits for all.
pub fn wave(client: &Client, bodies: Vec<Value>) -> Vec<Result<Timed>> {
    let start = Instant::now();
    std::thread::scope(|scope| {
        let handles: Vec<_> = bodies.into_iter().map(|body| {
            let client = client.clone();
            scope.spawn(move || {
                let sent = start.elapsed().as_secs_f64();
                client.chat(body, None).map(|chat| Timed { chat, sent })
            })
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("request thread panicked"))))
            .collect()
    })
}

/// Requests refused because their batch crowded the server (KV pool or queue).
pub fn crowded(result: &Result<Timed>) -> bool {
    result.as_ref().err().is_some_and(|e| {
        let text = format!("{e:#}");
        text.contains("pool exhausted") || text.contains("HTTP 429") || text.contains("queue is full")
    })
}

/// [`wave`], then each crowded request again on its own.
pub fn wave_retrying(client: &Client, bodies: Vec<Value>) -> Vec<Result<Timed>> {
    let mut results = wave(client, bodies.clone());
    for (k, result) in results.iter_mut().enumerate() {
        if crowded(result) {
            *result = wave(client, vec![bodies[k].clone()]).pop().unwrap_or_else(|| Err(anyhow::anyhow!("no result")));
        }
    }
    results
}

/// Aggregate output tokens per second of a wave (first token to last token).
pub fn aggregate(results: &[Timed]) -> f64 {
    let tokens: u64 = results.iter().map(|r| r.chat.timing.completion_tokens.saturating_sub(1)).sum();
    let start = results.iter().map(Timed::first).fold(f64::INFINITY, f64::min);
    let end = results.iter().map(Timed::last).fold(0.0, f64::max);
    if end > start { tokens as f64 / (end - start) } else { 0.0 }
}

/// Counters live under V4.1's `totals` or a generic family's `verify`.
fn counter_pair(stats: &Value, first: &str, second: &str) -> Option<(f64, f64)> {
    ["totals", "verify"].into_iter().find_map(|path| {
        let counters = &stats[path];
        Some((counters[first].as_f64()?, counters[second].as_f64()?))
    })
}

/// Draft counters the server publishes, for acceptance.
pub fn draft_counters(client: &Client) -> Option<(f64, f64)> {
    counter_pair(&client.stats().ok()?, "drafted_tokens", "accepted_drafts")
}

/// Verification rounds and tokens emitted by those rounds (not prefill).
pub fn round_counters(client: &Client) -> Option<(f64, f64)> {
    counter_pair(&client.stats().ok()?, "verification_rounds", "output_tokens")
}

/// Cumulative mapped-table counters; absence means this family has no mapped tables.
pub fn mapped_counters(client: &Client) -> Vec<Value> {
    client.stats().ok().and_then(|v| v["mapped_tables"].as_array().cloned()).unwrap_or_default()
}

/// Differences belong to this measurement; maxima remain explicitly lifetime values.
pub fn mapped_interval(before: &[Value], after: &[Value]) -> Vec<Value> {
    after.iter().map(|table| {
        let previous = before.iter().find(|old| old["name"] == table["name"]);
        let mut counters = serde_json::Map::new();
        let mut maxima = serde_json::Map::new();
        if let Some(fields) = table["cumulative"].as_object() {
            for (key, value) in fields {
                let old = previous.map(|p| &p["cumulative"][key]);
                if key.ends_with("max_ns") {
                    maxima.insert(key.clone(), value.clone());
                } else if let Some(n) = value.as_u64() {
                    counters.insert(key.clone(), json!(n.saturating_sub(old.and_then(Value::as_u64).unwrap_or(0))));
                } else if let Some(bins) = value.as_array() {
                    counters.insert(key.clone(), json!(bins.iter().enumerate().map(|(i, bin)| {
                        bin.as_u64().unwrap_or(0).saturating_sub(old.and_then(|v| v.get(i)).and_then(Value::as_u64).unwrap_or(0))
                    }).collect::<Vec<_>>()));
                }
            }
        }
        let devices: Vec<_> = table["host_wide_device_reads"].as_array().into_iter().flatten().map(|device| {
            let old = previous.and_then(|p| p["host_wide_device_reads"].as_array())
                .and_then(|ds| ds.iter().find(|d| d["device"] == device["device"]))
                .and_then(|d| d["bytes"].as_u64()).unwrap_or(0);
            json!({"device": device["device"], "bytes": device["bytes"].as_u64().unwrap_or(0).saturating_sub(old)})
        }).collect();
        json!({"name": table["name"], "backend": table["backend"], "accounting": table["accounting"],
            "interval": counters, "lifetime_maxima": maxima, "host_wide_device_reads": devices})
    }).collect()
}

pub fn acceptance(before: Option<(f64, f64)>, after: Option<(f64, f64)>) -> Option<f64> {
    let ((d0, a0), (d1, a1)) = (before?, after?);
    (d1 > d0).then(|| (a1 - a0) / (d1 - d0))
}

pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// A table for the dashboard's table view.
pub fn table(columns: &[&str], rows: Vec<Vec<Value>>) -> Value {
    json!({"columns": columns, "rows": rows})
}

/// Powers of two from `from` up to `max` (inclusive).
pub fn doublings(from: u64, max: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut v = from;
    while v <= max {
        out.push(v);
        v *= 2;
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn counters_support_v41_totals_and_generic_verify_without_mixing_scopes() {
        use serde_json::json;
        for path in ["totals", "verify"] {
            let mut stats = json!({});
            stats[path] = json!({"verification_rounds": 10, "output_tokens": 25,
                "drafted_tokens": 30, "accepted_drafts": 15});
            assert_eq!(super::counter_pair(&stats, "verification_rounds", "output_tokens"), Some((10., 25.)));
            assert_eq!(super::counter_pair(&stats, "drafted_tokens", "accepted_drafts"), Some((30., 15.)));
        }
        let incomplete = json!({"totals": {"verification_rounds": 10}, "verify": {"output_tokens": 25}});
        assert_eq!(super::counter_pair(&incomplete, "verification_rounds", "output_tokens"), None);
        assert_eq!(super::counter_pair(&json!({}), "drafted_tokens", "accepted_drafts"), None);
    }

    #[test]
    fn mapped_intervals_preserve_histograms_and_label_lifetime_maxima() {
        use serde_json::json;
        let before = json!({"name":"ple", "cumulative":{"page_hits":10,"miss_histogram":[1,2],"miss_max_ns":50},
            "host_wide_device_reads":[{"device":"259:2","bytes":1024}]});
        let after = json!({"name":"ple", "backend":"uring", "accounting":"nowait",
            "cumulative":{"page_hits":14,"miss_histogram":[2,5],"miss_max_ns":50},
            "host_wide_device_reads":[{"device":"259:2","bytes":4096}]});
        let delta = super::mapped_interval(&[before], &[after]);
        assert_eq!(delta[0]["interval"]["page_hits"], 4);
        assert_eq!(delta[0]["interval"]["miss_histogram"], json!([1,3]));
        assert!(delta[0]["interval"].get("miss_max_ns").is_none());
        assert_eq!(delta[0]["lifetime_maxima"]["miss_max_ns"], 50);
        assert_eq!(delta[0]["host_wide_device_reads"][0]["bytes"], 3072);
        assert!(super::mapped_interval(&[], &[]).is_empty());
    }

    #[test]
    fn mapped_intervals_keep_unclassified_wave_units_separate() {
        use serde_json::json;
        let before = json!({"name":"ple", "cumulative":{"unclassified_rows":320,
            "unclassified_batches":2,"unclassified_batch_ns":12000,
            "unclassified_batch_histogram":[0,2],"unclassified_batch_max_ns":7000,
            "prefetch_unclassified_batches":1,"prefetch_unclassified_batch_histogram":[0,1]}});
        let after = json!({"name":"ple", "cumulative":{"unclassified_rows":640,
            "unclassified_batches":4,"unclassified_batch_ns":24000,
            "unclassified_batch_histogram":[0,4],"unclassified_batch_max_ns":7000,
            "prefetch_unclassified_batches":2,"prefetch_unclassified_batch_histogram":[0,2]}});
        let delta = super::mapped_interval(&[before], &[after]);
        assert_eq!(delta[0]["interval"]["unclassified_rows"], 320);
        assert_eq!(delta[0]["interval"]["unclassified_batches"], 2);
        assert_eq!(delta[0]["interval"]["unclassified_batch_ns"], 12000);
        assert_eq!(delta[0]["interval"]["unclassified_batch_histogram"], json!([0,2]));
        assert_eq!(delta[0]["interval"]["prefetch_unclassified_batches"], 1);
        assert_eq!(delta[0]["interval"]["prefetch_unclassified_batch_histogram"], json!([0,1]));
        assert!(delta[0]["interval"].get("unclassified_batch_max_ns").is_none());
        assert_eq!(delta[0]["lifetime_maxima"]["unclassified_batch_max_ns"], 7000);
    }

    #[test]
    fn doublings_stop_at_the_limit() {
        assert_eq!(super::doublings(1024, 8192), vec![1024, 2048, 4096, 8192]);
        assert!(super::doublings(4096, 1000).is_empty());
    }
}
