//! Helpers the panels share: request bodies, prompt sizing, settings,
//! concurrent requests with absolute timing, server counters.
use super::Ctx;
use crate::client::{Chat, Client};
use crate::report::ServerInfo;
use crate::text::{corpus, nonce};
use anyhow::Result;
use cuteafd_api::openai::probe::ProbeSpec;
use serde_json::{json, Value};
use std::time::Instant;

pub fn messages(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

/// Greedy, thinking off when the checkpoint renderer supports it.
///
/// `max_tokens: 0` sends no output budget at all, so the server applies its own
/// output default: the request reserves the whole default lifetime, which is
/// what a KV-reservation admission probe has to exercise.
pub fn plain(client: &crate::client::Client, text: &str, max_tokens: u64) -> Value {
    let mut body = json!({"messages": messages(text), "max_tokens": max_tokens, "temperature": 0, "thinking": {"type": "disabled"}});
    if max_tokens == 0 {
        body.as_object_mut().expect("benchmark request object").remove("max_tokens");
    }
    crate::request::thinking_off(&client.configuration, body)
}

/// The probe a request whose prompt re-renders model output carries: a
/// follow-up turn, a rendered assistant/tool message, or ids taken from a
/// response. Its token ids are model-determined, so a matched pair compares
/// only the input-determined prompts and the engine logs this one without a
/// prompt token hash.
pub fn output_dependent() -> ProbeSpec {
    ProbeSpec { output_dependent: true, ..ProbeSpec::default() }
}

pub fn setting<'a>(info: &'a ServerInfo, names: &[&str]) -> Option<&'a str> {
    info.configuration.settings.iter().find(|s| names.contains(&s.name.as_str())).and_then(|s| s.value.as_deref())
}

/// Sequences the server decodes at once.
pub fn concurrency(info: &ServerInfo) -> usize {
    setting(info, &["concurrency", "max-sequences"]).and_then(|v| v.parse().ok()).unwrap_or(8)
}

/// Server tokens per corpus reference token, measured by the baseline fit.
pub fn prompt_token_ratio(ctx: &Ctx<'_>) -> f64 {
    ctx.baseline.and_then(|b| b.card.prompt_token_ratio)
        .filter(|r| r.is_finite() && *r > 0.3 && *r < 4.0).unwrap_or(1.0)
}

/// A unique prompt of about `tokens` tokens: nonce, corpus, then `question`.
pub fn sized_prompt(ctx: &Ctx<'_>, seed: u64, tokens: u64, question: &str) -> String {
    let words = ((tokens.saturating_sub(40)) as f64 / prompt_token_ratio(ctx)).max(8.0) as usize;
    format!("[{}] Read the notes below.\n\n{}\n\n{question}", nonce(), corpus(seed, words))
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
    send(client, bodies.into_iter().map(|body| (body, None)).collect())
}

/// [`wave`] of prompts that re-render model output (a retained base or a
/// follow-up turn), so every request carries [`output_dependent`].
pub fn wave_output_dependent(client: &Client, bodies: Vec<Value>) -> Vec<Result<Timed>> {
    send(client, bodies.into_iter().map(|body| (body, Some(output_dependent()))).collect())
}

fn send(client: &Client, bodies: Vec<(Value, Option<ProbeSpec>)>) -> Vec<Result<Timed>> {
    let start = Instant::now();
    std::thread::scope(|scope| {
        let handles: Vec<_> = bodies.into_iter().map(|(body, probe)| {
            let client = client.clone();
            scope.spawn(move || {
                let sent = start.elapsed().as_secs_f64();
                client.chat(body, probe).map(|chat| Timed { chat, sent })
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

#[derive(Clone, Copy, Debug)]
pub struct DraftCounters {
    pub drafted: f64,
    pub accepted: f64,
    pub passes: Option<f64>,
}

impl DraftCounters {
    pub fn from_stats(stats: &Value) -> Option<Self> {
        ["totals", "verify"].into_iter().find_map(|path| {
            let counters = &stats[path];
            Some(Self { drafted: counters["drafted_tokens"].as_f64()?,
                accepted: counters["accepted_drafts"].as_f64()?,
                passes: counters["draft_verify_passes"].as_f64() })
        })
    }
}

/// Draft counters from one scope/snapshot; old servers still supply the rate.
pub fn draft_counters(client: &Client) -> Option<DraftCounters> {
    DraftCounters::from_stats(&client.stats().ok()?)
}

/// Verification rounds and tokens emitted by those rounds (not prefill).
pub fn round_counters(client: &Client) -> Option<(f64, f64)> {
    counter_pair(&client.stats().ok()?, "verification_rounds", "output_tokens")
}

/// Prefix-cache tier counters (`/v1/stats` `prefix_cache`); absent fields read as zero.
const PREFIX_TIER_COUNTERS: [&str; 6] =
    ["device_hits", "host_hits", "mark_promotions", "promotions", "capture_stalls", "capture_stall_ns"];

/// The server's cumulative prefix-cache tier counters, `None` when it publishes none.
pub fn prefix_tier_counters(client: &Client) -> Option<Value> {
    let stats = client.stats().ok()?;
    let cache = stats["prefix_cache"].as_object()?;
    PREFIX_TIER_COUNTERS.iter().any(|k| cache.contains_key(*k))
        .then(|| json!(PREFIX_TIER_COUNTERS.iter().map(|k| (k.to_string(), json!(cache.get(*k).and_then(Value::as_u64).unwrap_or(0))))
            .collect::<serde_json::Map<_, _>>()))
}

/// Tier counter differences across one measurement.
pub fn prefix_tier_interval(before: Option<&Value>, after: Option<&Value>) -> Option<Value> {
    let (before, after) = (before?, after?);
    Some(json!(PREFIX_TIER_COUNTERS.iter().map(|k| (k.to_string(),
        json!(after[*k].as_u64().unwrap_or(0).saturating_sub(before[*k].as_u64().unwrap_or(0)))))
        .collect::<serde_json::Map<_, _>>()))
}

/// Which tier served a request's cached prefix (`usage.prefix_cache_source`: device, host or miss).
pub fn prefix_source(chat: &Chat) -> Option<String> {
    chat.usage["prefix_cache_source"].as_str().map(str::to_string)
}

/// Cumulative mapped-table counters; absence means this family has no mapped tables.
pub fn mapped_counters(client: &Client) -> Vec<Value> {
    client.stats().ok().and_then(|v| v["mapped_tables"].as_array().cloned()).unwrap_or_default()
}

/// One stats read at a phase boundary, shared by mapped-table, route and graph telemetry.
pub struct PhaseCounters {
    pub mapped_tables: Vec<Value>,
    pub expert_usage: Option<Value>,
    pub graph: Option<Value>,
}
pub fn phase_counters(client: &Client) -> PhaseCounters {
    phase_counters_from(&client.stats().unwrap_or(Value::Null))
}
pub fn phase_counters_from(stats: &Value) -> PhaseCounters {
    PhaseCounters {
        mapped_tables: stats["mapped_tables"].as_array().cloned().unwrap_or_default(),
        expert_usage: stats.get("expert_usage").cloned(),
        graph: graph_sample(stats),
    }
}

/// The cumulative graph counters of one `/v1/stats` read, kept compact: the
/// process capture total and its per-site totals, eager runs of shapes a sealed
/// graph set lacked, and per device the graph cache's per-cycle counters
/// (decode, verify, prefill, wave, other) and completed lane cycles. None when
/// the server publishes neither total (an older build, or no graph cache).
pub fn graph_sample(stats: &Value) -> Option<Value> {
    let captures = stats["graph_captures"].as_u64();
    let cache = stats["graph_cache"].as_array();
    if captures.is_none() && cache.is_none() { return None; }
    let devices: Vec<Value> = cache.into_iter().flatten().map(|device| {
        // Spanning is declared per identity; only the ranked rows publish it.
        let mut spans = serde_json::Map::new();
        for row in ["top_hits", "top_churn"].iter().flat_map(|key| device[*key].as_array().into_iter().flatten()) {
            let (Some(site), Some(span)) = (row["site"].as_str(), row["spans_devices"].as_bool()) else { continue };
            let known = spans.get(site).and_then(Value::as_bool).unwrap_or(false);
            spans.insert(site.to_string(), json!(known || span));
        }
        json!({"device": device["device"], "cycles": device["cycles"], "update_sites": device["update_sites"],
            "site_spans": spans, "lane_cycles_completed": device["epochs"]["completed"],
            "boundary": {"boundary_drains": device["epochs"]["boundary_drains"],
                "boundary_wait_us": device["epochs"]["boundary_wait_us"]},
            "event_pending_cycles": device["epochs"]["event_pending_cycles"], "charged": device["charged"]})
    }).collect();
    Some(json!({"captures": captures, "sites": stats["graph_capture_sites"], "devices": devices,
        "missing_variants": stats["graph_missing_variants"]}))
}

/// Per-device totals every graph interval reports, zero included; `updates`
/// is soft plus hard executable updates.
pub const GRAPH_TOTALS: [&str; 7] = ["captures", "misses", "eager_runs", "evictions", "updates", "update_failures", "hits"];

/// Adds one counter delta into [`GRAPH_TOTALS`] (soft and hard updates as `updates`).
fn add_totals(totals: &mut serde_json::Map<String, Value>, delta: &serde_json::Map<String, Value>) {
    for (key, value) in delta {
        let total = if key == "updates_soft" || key == "updates_hard" { "updates" } else { key.as_str() };
        if !GRAPH_TOTALS.contains(&total) { continue; }
        let slot = totals.entry(total.to_string()).or_insert(json!(0u64));
        *slot = json!(slot.as_u64().unwrap_or(0) + value.as_u64().unwrap_or(0));
    }
}

/// Per-site differences of a device's `update_sites` (site -> cycle -> counters):
/// sites whose counters moved, with nonzero [`GRAPH_TOTALS`] and per-cycle deltas,
/// and `spans_devices` when a ranked row of either sample named it. None when one
/// went backwards.
fn site_deltas(before: &Value, after: &Value) -> Option<serde_json::Map<String, Value>> {
    let mut sites = serde_json::Map::new();
    for (site, cycles) in after["update_sites"].as_object().into_iter().flatten() {
        let (mut totals, mut by_cycle) = (serde_json::Map::new(), serde_json::Map::new());
        for (cycle, counters) in cycles.as_object().into_iter().flatten() {
            let delta = counter_delta(&before["update_sites"][site.as_str()][cycle.as_str()], counters)?;
            add_totals(&mut totals, &delta);
            if !delta.is_empty() { by_cycle.insert(cycle.clone(), Value::Object(delta)); }
        }
        if by_cycle.is_empty() { continue; }
        let mut entry = json!({"totals": totals, "by_cycle": by_cycle});
        if let Some(span) = after["site_spans"][site.as_str()].as_bool().or(before["site_spans"][site.as_str()].as_bool()) {
            entry["spans_devices"] = json!(span);
        }
        sites.insert(site.clone(), entry);
    }
    Some(sites)
}

/// Nonzero differences of one counter object's integer fields; None when one went backwards.
fn counter_delta(before: &Value, after: &Value) -> Option<serde_json::Map<String, Value>> {
    let mut delta = serde_json::Map::new();
    for (key, value) in after.as_object().into_iter().flatten() {
        let Some(new) = value.as_u64() else { continue };
        let old = before[key.as_str()].as_u64().unwrap_or(0);
        let change = new.checked_sub(old)?;
        if change > 0 { delta.insert(key.clone(), json!(change)); }
    }
    Some(delta)
}

/// Differences of `[name, count]` pairs (capture sites); nonzero only, None when one went backwards.
fn pair_delta(before: &Value, after: &Value) -> Option<Vec<Value>> {
    let old = |name: &Value| before.as_array().into_iter().flatten()
        .find(|pair| &pair[0] == name).and_then(|pair| pair[1].as_u64()).unwrap_or(0);
    let mut delta = Vec::new();
    for pair in after.as_array().into_iter().flatten() {
        let Some(new) = pair[1].as_u64() else { continue };
        let change = new.checked_sub(old(&pair[0]))?;
        if change > 0 { delta.push(json!([pair[0], change])); }
    }
    Some(delta)
}

/// Graph counters over one timed band: differences of two cumulative
/// [`graph_sample`]s read just before and just after it. `zero_captures` is the
/// steady-state check (AGENTS.md): no process capture and no device-cache
/// capture in the band. Device rows report [`GRAPH_TOTALS`] and the nonzero
/// per-cycle counters; a device cache first seen after the start reads as new.
pub fn graph_interval(before: Option<&Value>, after: Option<&Value>) -> Option<Value> {
    let after = after?;
    let unavailable = |reason: &str| Some(json!({"available": false, "reason": reason}));
    let Some(before) = before else { return unavailable("no start sample") };
    let captures = match (before["captures"].as_u64(), after["captures"].as_u64()) {
        (Some(old), Some(new)) => match new.checked_sub(old) {
            Some(change) => Some(change),
            None => return unavailable("capture total went backwards (server restarted?)"),
        },
        _ => None,
    };
    let Some(sites) = pair_delta(&before["sites"], &after["sites"]) else {
        return unavailable("a capture site total went backwards");
    };
    let empty = json!({});
    let old_devices = before["devices"].as_array().cloned().unwrap_or_default();
    let new_devices = after["devices"].as_array().cloned().unwrap_or_default();
    if old_devices.iter().any(|old| !new_devices.iter().any(|new| new["device"] == old["device"])) {
        return unavailable("a device graph cache disappeared");
    }
    let mut devices = Vec::new();
    for device in &new_devices {
        let old = old_devices.iter().find(|old| old["device"] == device["device"]);
        let start = old.unwrap_or(&empty);
        let mut by_cycle = serde_json::Map::new();
        let mut totals: serde_json::Map<String, Value> = GRAPH_TOTALS.iter().map(|k| (k.to_string(), json!(0u64))).collect();
        for (cycle, counters) in device["cycles"].as_object().into_iter().flatten() {
            let Some(delta) = counter_delta(&start["cycles"][cycle.as_str()], counters) else {
                return unavailable("a device graph counter went backwards");
            };
            add_totals(&mut totals, &delta);
            if !delta.is_empty() { by_cycle.insert(cycle.clone(), Value::Object(delta)); }
        }
        let Some(sites) = site_deltas(start, device) else {
            return unavailable("a per-site graph counter went backwards");
        };
        let lanes = device["lane_cycles_completed"].as_array().map(|lanes| lanes.iter().enumerate().map(|(i, n)| {
            n.as_u64().and_then(|n| n.checked_sub(start["lane_cycles_completed"][i].as_u64().unwrap_or(0)))
        }).collect::<Option<Vec<_>>>());
        let lanes = match lanes {
            Some(None) => return unavailable("a lane cycle count went backwards"),
            Some(Some(lanes)) => json!(lanes),
            None => Value::Null,
        };
        let Some(mut boundary) = counter_delta(&start["boundary"], &device["boundary"]) else {
            return unavailable("a boundary counter went backwards");
        };
        for key in ["boundary_drains", "boundary_wait_us"] {
            if device["boundary"][key].is_u64() { boundary.entry(key.to_string()).or_insert(json!(0)); }
        }
        devices.push(json!({"device": device["device"], "new_device": old.is_none(), "totals": totals,
            "boundary": boundary, "event_pending_cycles": device["event_pending_cycles"],
            "charged_before": start["charged"], "charged_after": device["charged"],
            "by_cycle": by_cycle, "sites": sites, "lane_cycles_completed": lanes}));
    }
    let mut eager = Vec::new();
    for variant in after["missing_variants"].as_array().into_iter().flatten() {
        let old = before["missing_variants"].as_array().into_iter().flatten()
            .find(|old| old["family"] == variant["family"] && old["shape"] == variant["shape"])
            .and_then(|old| old["runs"].as_u64()).unwrap_or(0);
        let Some(runs) = variant["runs"].as_u64().and_then(|runs| runs.checked_sub(old)) else {
            return unavailable("a missing-variant run count went backwards");
        };
        if runs > 0 { eager.push(json!({"family": variant["family"], "shape": variant["shape"], "runs": runs})); }
    }
    let zero = captures.unwrap_or(0) == 0 && sites.is_empty()
        && devices.iter().all(|device| device["totals"]["captures"] == 0);
    Some(json!({"available": true, "boundaries": "/v1/stats reads just before and after the timed band",
        "captures": captures, "zero_captures": zero, "capture_sites": sites, "devices": devices,
        "missing_variant_runs": eager}))
}

/// Worst-GPU throttle summary of one `/v1/thermal` window (`thermal::window`):
/// software power cap plus hardware power brake seconds, software plus hardware
/// thermal slowdown seconds, the lowest busy SM clock mean, the highest
/// temperature and the lowest T.Limit headroom; per GPU under `gpus`. None
/// without GPU rows. A field is null when no GPU reported it.
pub fn thermal_summary(window: &Value) -> Option<Value> {
    let gpus = window["gpus"].as_array().filter(|gpus| !gpus.is_empty())?;
    let sum = |gpu: &Value, reasons: &[&str]| gpu["throttle_s"].is_object().then(||
        reasons.iter().filter_map(|reason| gpu["throttle_s"][*reason].as_f64()).sum::<f64>());
    let rows: Vec<Value> = gpus.iter().map(|gpu| json!({"index": gpu["index"],
        "power_cap_s": sum(gpu, &["sw_power_cap", "hw_power_brake"]),
        "thermal_slowdown_s": sum(gpu, &["sw_thermal", "hw_thermal"]),
        "busy_sm_clock_mhz": gpu["busy"]["sm_clock_mean_mhz"], "max_temperature_c": gpu["temperature_c"]["max"],
        "min_tlimit_headroom_c": gpu["tlimit_margin_c"]["min"]})).collect();
    let pick = |key: &str, max: bool| rows.iter().filter_map(|row| row[key].as_f64())
        .reduce(|a, b| if max { a.max(b) } else { a.min(b) });
    Some(json!({"seconds": window["seconds"], "power_cap_s": pick("power_cap_s", true),
        "thermal_slowdown_s": pick("thermal_slowdown_s", true), "busy_sm_clock_mhz": pick("busy_sm_clock_mhz", false),
        "max_temperature_c": pick("max_temperature_c", true), "min_tlimit_headroom_c": pick("min_tlimit_headroom_c", false),
        "gpus": rows}))
}

/// One summary row per timed band with graph counters (sweep bands and
/// context_load points): process and device-cache captures, misses, eager
/// runs, `zero_captures`, the sites that missed, ran eagerly, updated or
/// captured (summed over devices, most eager runs first) and the band's
/// [`thermal_summary`].
pub fn graph_band(band: String, graph: &Value, thermal: Option<&Value>) -> Value {
    let devices = graph["devices"].as_array();
    let device = |key: &str| devices.map(|devices| devices.iter().filter_map(|d| d["totals"][key].as_u64()).sum::<u64>());
    let mut sites: std::collections::BTreeMap<String, Value> = Default::default();
    for (site, entry) in devices.into_iter().flatten().flat_map(|d| d["sites"].as_object().into_iter().flatten()) {
        let row = sites.entry(site.clone()).or_insert_with(|| json!({"site": site, "eager_runs": 0u64, "misses": 0u64,
            "updates": 0u64, "update_failures": 0u64, "captures": 0u64}));
        for key in ["eager_runs", "misses", "updates", "update_failures", "captures"] {
            row[key] = json!(row[key].as_u64().unwrap_or(0) + entry["totals"][key].as_u64().unwrap_or(0));
        }
        if let Some(span) = entry["spans_devices"].as_bool() {
            row["spans_devices"] = json!(span || row["spans_devices"].as_bool().unwrap_or(false));
        }
    }
    let mut sites: Vec<Value> = sites.into_values().filter(|row|
        ["eager_runs", "misses", "updates", "update_failures", "captures"].iter().any(|k| row[*k].as_u64().unwrap_or(0) > 0)).collect();
    sites.sort_by(|a, b| b["eager_runs"].as_u64().cmp(&a["eager_runs"].as_u64())
        .then_with(|| b["misses"].as_u64().cmp(&a["misses"].as_u64())));
    json!({"band": band, "available": graph["available"], "captures": graph["captures"],
        "device_captures": device("captures"), "misses": device("misses"),
        "eager_runs": device("eager_runs"), "updates": device("updates"), "zero_captures": graph["zero_captures"],
        "reason": graph["reason"], "sites": sites, "thermal": thermal.and_then(thermal_summary),
        "boundary_devices": devices.map(|devices| devices.iter().map(|d| json!({
            "device": d["device"], "boundary": d["boundary"], "event_pending_cycles": d["event_pending_cycles"],
            "charged_before": d["charged_before"], "charged_after": d["charged_after"]})).collect::<Vec<_>>())})
}

/// Differences of asynchronous total snapshots, not exact phase-isolated counts.
pub fn expert_usage_interval(before: Option<&Value>, after: Option<&Value>) -> Option<Value> {
    let after = after?;
    if after["enabled"] != true { return None; }
    let start = before.and_then(|v| v["sampled_s"].as_f64());
    let end = after["sampled_s"].as_f64();
    let unavailable = |reason: &str| Some(json!({"available": false, "reason": reason,
        "boundaries": "approximate_async_snapshots", "start_sampled_s": start, "end_sampled_s": end}));
    let Some(before) = before.filter(|v| v["enabled"] == true) else {
        return unavailable("no enabled start sample");
    };
    if !before["error"].is_null() || !after["error"].is_null() {
        return unavailable("expert usage snapshot worker failed");
    }
    let Some((start, end)) = start.zip(end).filter(|(a, b)| a.is_finite() && *a >= 0.0 && b.is_finite() && b >= a) else {
        return unavailable("invalid or restarted sample clock");
    };
    if before["geometry"] != after["geometry"] || before["semantics"] != after["semantics"]
        || after["semantics"] != "executed_routed_rows" {
        return unavailable("geometry or counting semantics changed");
    }
    let Ok(geometry) = serde_json::from_value::<cuteafd_core::expert_usage::Geometry>(after["geometry"].clone()) else {
        return unavailable("invalid geometry");
    };
    if !geometry.validate() { return unavailable("invalid geometry"); }
    let cells = geometry.cells().unwrap();
    let delta = |name: &str| -> Option<Vec<u64>> {
        let old = before["windows"]["total"][name].as_array()?;
        let new = after["windows"]["total"][name].as_array()?;
        if old.len() != cells || new.len() != cells { return None; }
        old.iter().zip(new).map(|(a, b)| b.as_u64()?.checked_sub(a.as_u64()?)).collect()
    };
    let Some(prefill) = delta("prefill_routed_rows") else { return unavailable("invalid prefill totals"); };
    let Some(decode) = delta("decode_routed_rows") else { return unavailable("invalid decode totals"); };
    let Some(combined) = prefill.iter().zip(&decode).map(|(a, b)| a.checked_add(*b)).collect::<Option<Vec<_>>>() else {
        return unavailable("combined counter overflow");
    };
    if combined.iter().try_fold(0u64, |sum, n| sum.checked_add(*n)).is_none() {
        return unavailable("summary counter overflow");
    }
    use cuteafd_core::expert_usage::summarize;
    Some(json!({"available": true, "semantics": "executed_routed_rows", "geometry": geometry,
        "boundaries": "approximate_async_snapshots", "start_sampled_s": start, "end_sampled_s": end,
        "sample_elapsed_s": end - start, "same_sample": start == end,
        "prefill_routed_rows": prefill, "decode_routed_rows": decode,
        "prefill": summarize(&geometry, &prefill), "decode": summarize(&geometry, &decode),
        "combined": summarize(&geometry, &combined)}))
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

pub fn acceptance(before: Option<DraftCounters>, after: Option<DraftCounters>) -> Option<f64> {
    let (before, after) = (before?, after?);
    (after.drafted > before.drafted && after.accepted >= before.accepted)
        .then(|| (after.accepted - before.accepted) / (after.drafted - before.drafted))
}

/// Vendor-style acceptance length includes one target/bonus token per draft pass.
pub fn acceptance_length(before: Option<DraftCounters>, after: Option<DraftCounters>) -> Option<f64> {
    let (before, after) = (before?, after?);
    let (p0, p1) = (before.passes?, after.passes?);
    (p1 > p0 && after.accepted >= before.accepted)
        .then(|| 1.0 + (after.accepted - before.accepted) / (p1 - p0))
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
    use serde_json::{json, Value};
    #[test]
    fn acceptance_length_counts_bonus_and_only_draft_passes() {
        use super::{DraftCounters, acceptance, acceptance_length};
        let before = DraftCounters::from_stats(&json!({"totals": {
            "drafted_tokens":20,"accepted_drafts":10,"draft_verify_passes":4}}));
        let after = DraftCounters::from_stats(&json!({"totals": {
            "drafted_tokens":40,"accepted_drafts":22,"draft_verify_passes":8}}));
        assert_eq!(acceptance(before, after), Some(0.6));
        assert_eq!(acceptance_length(before, after), Some(4.0));
        assert_eq!(acceptance_length(before, before), None);
        assert_eq!(acceptance_length(after, before), None);
        let rejected = Some(DraftCounters { drafted:40., accepted:10., passes:Some(8.) });
        assert_eq!(acceptance_length(before, rejected), Some(1.0));
        let old = DraftCounters::from_stats(&json!({"totals": {"drafted_tokens":40,"accepted_drafts":22},
            "verify": {"draft_verify_passes":8}}));
        assert_eq!(acceptance(before, old), Some(0.6));
        assert_eq!(acceptance_length(before, old), None, "never mix counter scopes");
    }

    #[test]
    fn zero_tokens_omits_the_output_budget() {
        use serde_json::json;
        let client = crate::client::Client::new("http://127.0.0.1:1", None, std::sync::Arc::default());
        let omitted = super::plain(&client, "test", 0);
        assert_eq!(omitted.get("max_tokens"), None, "{omitted}");
        assert_eq!(omitted, json!({"messages":[{"role":"user","content":"test"}], "temperature":0,
            "thinking":{"type":"disabled"}}));
        let explicit = super::plain(&client, "test", 192);
        assert_eq!(explicit["max_tokens"], 192);
    }

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

    fn usage_sample(t: f64, n: u64) -> Value {
        let reader = cuteafd_core::expert_usage::Reader::default();
        reader.initialize(cuteafd_core::expert_usage::Geometry {
            layers: 2, experts: 4, topk: 2, routed_layers: vec![1],
        }).unwrap();
        reader.publish(cuteafd_core::expert_usage::Snapshot { elapsed_s: t,
            prefill: vec![0, 0, 0, 0, n, n, 0, 0],
            decode: vec![0, 0, 0, 0, 0, n, n, 0] }).unwrap();
        reader.snapshot()
    }

    #[test]
    fn expert_usage_intervals_keep_phases_and_approximate_boundaries() {
        let a = usage_sample(1.0, 2);
        let b = usage_sample(3.0, 5);
        let delta = super::expert_usage_interval(Some(&a), Some(&b)).unwrap();
        assert_eq!(delta["boundaries"], "approximate_async_snapshots");
        assert_eq!(delta["start_sampled_s"], 1.0);
        assert_eq!(delta["end_sampled_s"], 3.0);
        assert_eq!(delta["prefill_routed_rows"], json!([0,0,0,0,3,3,0,0]));
        assert_eq!(delta["decode_routed_rows"], json!([0,0,0,0,0,3,3,0]));
        assert_eq!(delta["prefill"]["routed_rows"], 3);
        assert_eq!(delta["combined"]["touched"], 3);
        assert_eq!(delta["combined"]["possible"], 4);
        assert_eq!(delta["combined"]["concentration"], 0.375);
        let same = super::expert_usage_interval(Some(&b), Some(&b)).unwrap();
        assert_eq!(same["same_sample"], true);
        assert_eq!(same["combined"]["touched"], 0);
    }

    #[test]
    fn expert_usage_intervals_reject_missing_stale_or_changed_inputs() {
        let a = usage_sample(1.0, 2);
        assert!(super::expert_usage_interval(None, None).is_none());
        assert!(super::expert_usage_interval(None, Some(&json!({"enabled": false}))).is_none());
        assert_eq!(super::expert_usage_interval(None, Some(&a)).unwrap()["available"], false);
        for (path, value) in [
            (vec!["geometry", "experts"], json!(8)),
            (vec!["sampled_s"], json!(0.5)),
            (vec!["error"], json!("worker failed")),
            (vec!["windows", "total", "prefill_routed_rows"], json!([0,0,0,0,1,1,0,0])),
            (vec!["windows", "total", "decode_routed_rows"], json!([0])),
        ] {
            let mut b = usage_sample(2.0, 3);
            let mut field = &mut b;
            for key in path { field = &mut field[key]; }
            *field = value;
            assert_eq!(super::expert_usage_interval(Some(&a), Some(&b)).unwrap()["available"], false);
        }
    }

    /// A `/v1/stats` payload as the daemon serves it (probe::graph_capture_stats):
    /// totals, per-site totals, missing variants and per-device graph_cache rows.
    fn graph_stats(captures: u64, decode_misses: u64, verify_captures: u64, completed: u64) -> Value {
        json!({"active": 0, "graph_captures": captures,
            "graph_capture_sites": [["engine.rs:1937", captures], ["draft.rs:20", 1]],
            "graph_capture_site_deltas": [],
            "graph_missing_variants": [{"family": "glm5", "shape": "c=3", "runs": decode_misses}],
            "graph_cache": [
                {"device": 0, "charged": 335544320, "reserve": 1 << 30,
                 "top_hits": [{"hits": 9, "site": "glm5.decode", "spans_devices": false},
                              {"hits": 3, "site": "glm5.exchange", "spans_devices": true}],
                 "top_churn": [{"site": "glm5.decode", "spans_devices": false}],
                 "update_sites": {
                     "glm5.decode": {"decode": {"hits": 90 + completed, "misses": decode_misses, "eager_runs": decode_misses,
                                                "captures": 4, "updates_soft": 2, "updates_hard": 0}},
                     "glm5.exchange": {"verify": {"hits": 10, "misses": verify_captures, "eager_runs": verify_captures,
                                                  "updates_hard": 1}},
                     "glm5.idle": {"decode": {"hits": 0}}},
                 "cycles": {"decode": {"hits": 100 + completed, "misses": decode_misses, "captures": 4,
                                       "segments_launched": 3 * completed, "eager_runs": decode_misses,
                                       "evictions": 0, "updates_soft": 2, "updates_hard": 0, "update_failures": 0},
                            "verify": {"hits": 10, "misses": 0, "captures": verify_captures, "eager_runs": 0,
                                       "evictions": verify_captures / 2, "updates_soft": 0, "updates_hard": 1}},
                 "epochs": {"cycles": [completed + 1, 0], "completed": [completed, 0], "pending": [0, 0]}},
                {"device": 1, "cycles": {"decode": {"hits": 7, "captures": 2}},
                 "epochs": {"completed": [5, 0]}}]})
    }

    #[test]
    fn graph_intervals_difference_cumulative_stats_and_certify_zero_captures() {
        let before = super::phase_counters_from(&graph_stats(6, 1, 3, 40)).graph;
        let sample = before.as_ref().unwrap();
        assert_eq!(sample["captures"], 6);
        assert_eq!(sample["devices"][0]["charged"], 335544320);
        assert!(sample["devices"][1]["charged"].is_null(), "missing legacy charge remains unknown");
        // Steady state: hits and lane cycles move, nothing captures.
        let steady = super::phase_counters_from(&graph_stats(6, 1, 3, 72)).graph;
        let band = super::graph_interval(before.as_ref(), steady.as_ref()).unwrap();
        assert_eq!(band["available"], true);
        assert_eq!(band["captures"], 0);
        assert_eq!(band["zero_captures"], true);
        assert_eq!(band["capture_sites"], json!([]));
        let device = &band["devices"][0];
        assert_eq!(device["totals"], json!({"captures": 0, "misses": 0, "eager_runs": 0, "evictions": 0,
            "updates": 0, "update_failures": 0, "hits": 32}));
        assert_eq!(device["by_cycle"], json!({"decode": {"hits": 32, "segments_launched": 96}}));
        assert_eq!(device["lane_cycles_completed"], json!([32, 0]));
        assert_eq!(band["devices"][1]["totals"]["hits"], 0);
        assert_eq!(band["missing_variant_runs"], json!([]));
        // A band that captured: the process total, the site, the verify cycle and an eager miss.
        let after = super::phase_counters_from(&graph_stats(8, 3, 5, 80)).graph;
        let band = super::graph_interval(before.as_ref(), after.as_ref()).unwrap();
        assert_eq!(band["captures"], 2);
        assert_eq!(band["zero_captures"], false);
        assert_eq!(band["capture_sites"], json!([["engine.rs:1937", 2]]));
        let device = &band["devices"][0];
        assert_eq!(device["totals"]["captures"], 2);
        assert_eq!(device["totals"]["misses"], 2);
        assert_eq!(device["totals"]["eager_runs"], 2);
        assert_eq!(device["totals"]["evictions"], 1);
        assert_eq!(device["by_cycle"]["verify"], json!({"captures": 2, "evictions": 1}));
        assert_eq!(band["missing_variant_runs"], json!([{"family": "glm5", "shape": "c=3", "runs": 2}]));
        // Per site: the rank-local decode site and the spanning exchange site each missed eagerly.
        assert_eq!(device["sites"], json!({
            "glm5.decode": {"totals": {"hits": 40, "misses": 2, "eager_runs": 2}, "spans_devices": false,
                            "by_cycle": {"decode": {"hits": 40, "misses": 2, "eager_runs": 2}}},
            "glm5.exchange": {"totals": {"misses": 2, "eager_runs": 2}, "spans_devices": true,
                              "by_cycle": {"verify": {"misses": 2, "eager_runs": 2}}}}));
        assert_eq!(band["devices"][1]["sites"], json!({}), "a device without update_sites reports none");
        let row = super::graph_band("levels C16".into(), &band, None);
        assert_eq!(row["sites"], json!([
            {"site": "glm5.decode", "eager_runs": 2, "misses": 2, "updates": 0, "update_failures": 0, "captures": 0, "spans_devices": false},
            {"site": "glm5.exchange", "eager_runs": 2, "misses": 2, "updates": 0, "update_failures": 0, "captures": 0, "spans_devices": true}]));
        assert_eq!((row["device_captures"].as_u64(), row["misses"].as_u64(), row["eager_runs"].as_u64()), (Some(2), Some(2), Some(2)));
        assert!(row["thermal"].is_null());
        // Steady band: hits only, so no site row survives the summary.
        let steady = super::graph_interval(before.as_ref(), steady.as_ref()).unwrap();
        assert_eq!(steady["devices"][0]["sites"]["glm5.decode"]["totals"], json!({"hits": 32}));
        assert_eq!(super::graph_band("levels C1".into(), &steady, None)["sites"], json!([]));
    }

    #[test]
    fn graph_boundaries_report_wait_deltas_and_pending_gauge() {
        let mut before = graph_stats(6, 1, 3, 40);
        before["graph_cache"][0]["epochs"]["boundary_drains"] = json!(2);
        before["graph_cache"][0]["epochs"]["boundary_wait_us"] = json!(10);
        before["graph_cache"][0]["charged"] = json!(40);
        let mut after = before.clone();
        after["graph_cache"][0]["epochs"]["boundary_drains"] = json!(4);
        after["graph_cache"][0]["epochs"]["boundary_wait_us"] = json!(60);
        after["graph_cache"][0]["epochs"]["event_pending_cycles"] = json!(3);
        after["graph_cache"][0]["charged"] = json!(42);
        let interval = super::graph_interval(super::graph_sample(&before).as_ref(), super::graph_sample(&after).as_ref()).unwrap();
        let band = super::graph_band("C1".into(), &interval, None);
        assert_eq!(band["boundary_devices"][0], json!({"device": 0,
            "boundary": {"boundary_drains": 2, "boundary_wait_us": 50},
            "event_pending_cycles": 3, "charged_before": 40, "charged_after": 42}));
    }

    #[test]
    fn per_site_deltas_count_updates_and_refuse_backwards_sites() {
        let before = super::graph_sample(&graph_stats(6, 1, 3, 40));
        let mut stats = graph_stats(6, 1, 3, 40);
        stats["graph_cache"][0]["update_sites"]["glm5.decode"]["decode"]["updates_soft"] = json!(5);
        stats["graph_cache"][0]["update_sites"]["glm5.decode"]["decode"]["updates_hard"] = json!(1);
        stats["graph_cache"][0]["update_sites"]["glm5.new"] = json!({"wave": {"captures": 1}});
        let band = super::graph_interval(before.as_ref(), super::graph_sample(&stats).as_ref()).unwrap();
        let sites = &band["devices"][0]["sites"];
        assert_eq!(sites["glm5.decode"]["totals"], json!({"updates": 4}));
        assert_eq!(sites["glm5.decode"]["by_cycle"]["decode"], json!({"updates_soft": 3, "updates_hard": 1}));
        assert_eq!(sites["glm5.new"]["totals"], json!({"captures": 1}));
        assert!(sites["glm5.new"].get("spans_devices").is_none(), "no ranked row: spanning unknown");
        let row = super::graph_band("context 32768 C8".into(), &band, None);
        assert_eq!(row["updates"], 0, "device totals come from cycles, not sites");
        assert_eq!(row["sites"].as_array().unwrap().iter().map(|s| s["site"].as_str().unwrap()).collect::<Vec<_>>(),
            ["glm5.decode", "glm5.new"]);
        let mut backwards = graph_stats(6, 1, 3, 40);
        backwards["graph_cache"][0]["update_sites"]["glm5.decode"]["decode"]["misses"] = json!(0);
        assert_eq!(super::graph_interval(before.as_ref(), super::graph_sample(&backwards).as_ref()).unwrap()["reason"],
            "a per-site graph counter went backwards");
    }

    /// A `/v1/thermal?from=` window as `thermal::window` serves it: one capped,
    /// slowed GPU and one cool one.
    fn thermal_window() -> Value {
        json!({"from_s": 10.0, "to_s": 42.5, "seconds": 32.5, "governor_name": "gpu-powerd", "gpus": [
            {"index": 0, "samples": 130,
             "throttle_s": {"sw_power_cap": 12.0, "sw_thermal": 0.5, "hw_thermal": 1.25, "hw_power_brake": 0.25},
             "busy": {"fraction": 0.98, "sm_clock_min_mhz": 1500.0, "sm_clock_mean_mhz": 2210.5},
             "temperature_c": {"max": 83.0, "mean": 79.0}, "tlimit_margin_c": {"min": 4.0, "mean": 7.5}},
            {"index": 1, "samples": 130,
             "throttle_s": {"sw_power_cap": 0.0, "sw_thermal": 0.0, "hw_thermal": 0.0, "hw_power_brake": 0.0},
             "busy": {"fraction": 0.97, "sm_clock_min_mhz": 2600.0, "sm_clock_mean_mhz": 2700.0},
             "temperature_c": {"max": 70.0, "mean": 66.0}, "tlimit_margin_c": {"min": null, "mean": null}}]})
    }

    #[test]
    fn thermal_summary_reports_the_worst_gpu_and_each_gpu() {
        let summary = super::thermal_summary(&thermal_window()).unwrap();
        assert_eq!(summary["seconds"], 32.5);
        assert_eq!(summary["power_cap_s"], 12.25);
        assert_eq!(summary["thermal_slowdown_s"], 1.75);
        assert_eq!(summary["busy_sm_clock_mhz"], 2210.5);
        assert_eq!(summary["max_temperature_c"], 83.0);
        assert_eq!(summary["min_tlimit_headroom_c"], 4.0);
        assert_eq!(summary["gpus"][1], json!({"index": 1, "power_cap_s": 0.0, "thermal_slowdown_s": 0.0,
            "busy_sm_clock_mhz": 2700.0, "max_temperature_c": 70.0, "min_tlimit_headroom_c": null}));
        // An empty window (no samples, so no reason bits) has nulls, not zeros.
        let empty = json!({"seconds": 0.1, "gpus": [{"index": 0, "samples": 0, "throttle_s": null,
            "busy": {"sm_clock_mean_mhz": null}, "temperature_c": {"max": null}, "tlimit_margin_c": {"min": null}}]});
        let summary = super::thermal_summary(&empty).unwrap();
        assert!(summary["power_cap_s"].is_null() && summary["max_temperature_c"].is_null());
        assert!(super::thermal_summary(&json!({"seconds": 1.0, "gpus": []})).is_none());
        assert!(super::thermal_summary(&Value::Null).is_none());
        let band = super::graph_band("context 131072 C16".into(), &json!({"available": true, "zero_captures": true,
            "captures": 0, "devices": []}), Some(&thermal_window()));
        assert_eq!(band["thermal"]["power_cap_s"], 12.25);
    }

    #[test]
    fn graph_intervals_refuse_restarts_and_absent_counters() {
        assert!(super::graph_sample(&json!({"active": 0})).is_none());
        assert!(super::graph_interval(None, None).is_none());
        let sample = super::graph_sample(&graph_stats(6, 1, 3, 40));
        assert_eq!(super::graph_interval(None, sample.as_ref()).unwrap()["available"], false);
        let restarted = super::graph_sample(&graph_stats(2, 0, 0, 1));
        assert_eq!(super::graph_interval(sample.as_ref(), restarted.as_ref()).unwrap()["available"], false);
        let mut gone = graph_stats(6, 1, 3, 40);
        gone["graph_cache"].as_array_mut().unwrap().pop();
        assert_eq!(super::graph_interval(sample.as_ref(), super::graph_sample(&gone).as_ref()).unwrap()["available"], false);
        // A device cache published only after the start reads as new, from zero.
        let mut late = graph_stats(6, 1, 3, 40);
        late["graph_cache"].as_array_mut().unwrap().pop();
        let band = super::graph_interval(super::graph_sample(&late).as_ref(), sample.as_ref()).unwrap();
        assert_eq!(band["devices"][1]["new_device"], true);
        assert_eq!(band["devices"][1]["totals"]["captures"], 2);
        assert_eq!(band["zero_captures"], false);
    }

    #[test]
    fn doublings_stop_at_the_limit() {
        assert_eq!(super::doublings(1024, 8192), vec![1024, 2048, 4096, 8192]);
        assert!(super::doublings(4096, 1000).is_empty());
    }
}
