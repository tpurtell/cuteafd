//! A blocking OpenAI client for the server's own API: streamed chat requests
//! timed the way a user sees them (SSE arrival times), optional benchmark
//! probes, and the bench token that passes the lockout.
use crate::report::StreamTiming;
use anyhow::{bail, Context, Result};
use cuteafd_api::openai::probe::{self, ProbeRecord, ProbeSpec};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Header carrying the active run's token past the lockout.
pub const BENCH_HEADER: &str = "x-cuteafd-bench";

#[derive(Clone)]
pub struct Client {
    pub base: String,
    pub model: String,
    pub configuration: crate::report::Configuration,
    agent: ureq::Agent,
    fabric_enabled: bool,
    token: Option<String>,
    api_key: Option<cuteafd_api::openai::auth::ApiKey>,
    cancel: Arc<AtomicBool>,
}

/// Renew an off-path observation lease for a long standalone sweep.
pub(crate) struct FabricSubscription {
    stop: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Drop for FabricSubscription {
    fn drop(&mut self) {
        let (stop, wake) = &*self.stop;
        *stop.lock().unwrap() = true;
        wake.notify_one();
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}

/// One completed chat request.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Chat {
    pub timing: StreamTiming,
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<Value>,
    pub usage: Value,
    pub probe: Option<ProbeRecord>,
}

#[derive(Debug, thiserror::Error)]
#[error("benchmark cancelled")]
pub struct Cancelled;

/// An upstream rejection retains its status through the benchmark API.
#[derive(Debug, thiserror::Error)]
#[error("HTTP {code}: {body}")]
pub struct UpstreamHttpError {
    pub code: u16,
    pub body: String,
}

impl Client {
    pub fn new(base: &str, token: Option<String>, cancel: Arc<AtomicBool>) -> Self {
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(900)).build();
        Self { configuration: crate::report::Configuration::default(), base: base.trim_end_matches('/').to_string(), model: String::new(), agent, token, cancel, api_key: None, fabric_enabled: false }
    }

    pub fn with_api_key(mut self, key: Option<cuteafd_api::openai::auth::ApiKey>) -> Self {
        self.api_key = key;
        self
    }
    pub fn with_fabric(mut self, enabled: bool) -> Self {
        self.fabric_enabled = enabled;
        self
    }

    fn authorize(&self, mut request: ureq::Request) -> ureq::Request {
        if let Some(key) = &self.api_key { request = request.set("Authorization", &key.authorization()); }
        else if let Some(token) = &self.token { request = request.set("Authorization", &format!("Bearer {token}")); }
        request
    }
    /// The run's lockout token (subprocesses pass it as their API key).
    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn check(&self) -> Result<()> {
        if self.cancelled() { Err(Cancelled.into()) } else { Ok(()) }
    }

    fn get(&self, path: &str) -> Result<Value> {
        let response = self.authorize(self.agent.get(&format!("{}{path}", self.base))).call()
            .with_context(|| format!("GET {path}"))?;
        Ok(response.into_json()?)
    }

    /// The served model id and its `/v1/models` record; sets `self.model`.
    pub fn discover(&mut self) -> Result<Value> {
        let models = self.get("/v1/models")?;
        let record = models["data"].get(0).cloned().context("/v1/models lists no model")?;
        self.configuration.supports_thinking_off = record["supports_thinking_off"].as_bool();
        self.model = record["id"].as_str().context("model id")?.to_string();
        Ok(record)
    }

    pub fn model_record(&self) -> Result<Value> {
        let models = self.get("/v1/models")?;
        models["data"].as_array().and_then(|records| records.iter()
            .find(|record| record["id"].as_str() == Some(self.model.as_str())))
            .cloned().context("served model is no longer advertised")
    }

    /// HTTP media probes bypass the older in-process ProbeSpec registration shape.
    pub fn media_probe(&self, mut body: Value, spec: Value) -> Result<Value> {
        self.check()?;
        crate::request::validate(&self.configuration, &body)?;
        body["speculative_acceptance"] = json!(self.configuration.speculative_acceptance);
        body["model"] = json!(self.model);
        body["stream"] = json!(false);
        let mut request = self.agent.post(&format!("{}/v1/bench/probe", self.base))
            .set("content-type", "application/json");
        if let Some(token) = &self.token {
            request = request.set(BENCH_HEADER, token).set("Authorization", &format!("Bearer {token}"));
        }
        match self.authorize(request).send_json(serde_json::json!({"body": body, "spec": spec})) {
            Ok(response) => Ok(response.into_json()?),
            Err(ureq::Error::Status(code, response)) => {
                let detail = response.into_string().unwrap_or_default();
                bail!("media probe HTTP {code}: {}", detail.chars().take(400).collect::<String>());
            }
            Err(error) => Err(error).context("media probe request"),
        }
    }

    /// Short, bounded reads for the in-wave pool observer.
    pub fn pool_observation(&self) -> Value {
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(2))
            .timeout_read(Duration::from_secs(2)).build();
        let get = |path: &str| self.authorize(agent.get(&format!("{}{path}", self.base)))
            .call().ok().and_then(|r| r.into_json::<Value>().ok()).unwrap_or(Value::Null);
        let stats = get("/v1/stats");
        let console = get("/v1/console/snapshot");
        json!({"active": stats["active"], "prefilling": stats["prefilling"], "queued": stats["queued"],
            "prefix_cache": stats["prefix_cache"], "console": console["g"]})
    }

    pub(crate) fn fabric_subscription(&self) -> Option<FabricSubscription> {
        if !self.fabric_enabled { return None; }
        let stop = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let (client, signal) = (self.clone(), stop.clone());
        let thread = std::thread::Builder::new().name("fabric-lease".into()).spawn(move || {
            cuteafd_transport::traffic::background_priority();
            loop {
                let _ = client.fabric();
                let (stop, wake) = &*signal;
                let stopped = wake.wait_timeout_while(stop.lock().unwrap(), Duration::from_secs(1), |stop| !*stop).unwrap().0;
                if *stopped || client.cancelled() { break; }
            }
        }).ok()?;
        Some(FabricSubscription { stop, thread: Some(thread) })
    }

    /// Cached 1 Hz observations; no sysfs or peer I/O in this HTTP handler.
    pub fn fabric(&self) -> Option<cuteafd_transport::traffic::Snapshot> {
        if !self.fabric_enabled { return None; }
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_millis(200))
            .timeout_read(Duration::from_millis(300)).build();
        self.authorize(agent.get(&format!("{}/v1/fabric", self.base))).call().ok()?.into_json().ok()
    }

    /// Decode counters are sampled outside the timed request.
    pub fn decode_counters(&self) -> Option<Value> {
        self.get("/v1/console/snapshot").ok()?.get("g")?.get("totals").cloned()
    }

    /// Wait outside request timing for a new 1 Hz observation on each known
    /// host. This excludes the preceding warm-up and includes the final work;
    /// rates describe the sampled envelope, including idle boundary padding.
    pub fn fabric_mark(&self) -> Option<cuteafd_transport::traffic::Snapshot> {
        let mut before = self.fabric()?;
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if self.cancel.load(Ordering::Relaxed) || Instant::now() >= deadline { return None; }
            let after = self.fabric()?;
            if after.hosts.keys().any(|name| !before.hosts.contains_key(name)) {
                before = after;
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            let fresh = !after.hosts.is_empty() && after.hosts.values().all(|h| !h.incarnation.is_empty() && h.sampled_s > 0.0)
                && after.hosts.values().any(|h| !h.ports.is_empty()) && before.hosts.iter().all(|(name, b)| after.hosts.get(name)
                .is_none_or(|a| a.incarnation != b.incarnation || a.sampled_s > b.sampled_s));
            let storage_ready = after.hosts.values().all(|h| h.sources.get("sparknest")
                .is_none_or(|s| !matches!(s.status.as_str(), "waiting" | "stale")));
            if fresh && storage_ready { return Some(after); }
            if Instant::now() >= deadline { return None; }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    pub fn fabric_since(&self, before: Option<cuteafd_transport::traffic::Snapshot>, name: &str, timed: bool)
        -> Option<crate::fabric::Phase> {
        let phase = crate::fabric::interval(name, timed, &before?, &self.fabric_mark()?);
        if phase.ports.is_empty() { return None; }
        // Logging runs after timing and remains best-effort for older servers.
        if let Ok(value) = serde_json::to_value(&phase) {
            let request = self.authorize(self.agent.post(&format!("{}/v1/fabric/phase", self.base)))
                .timeout(Duration::from_secs(2));
            let _ = request.send_json(value);
        }
        Some(phase)
    }

    pub fn stats(&self) -> Result<Value> {
        self.get("/v1/stats")
    }

    /// The server's report clock (`/v1/thermal` `now_s`): the query time a window is
    /// measured on. A GPU's sample time is `current.gpus[].sampled_s` and is not this
    /// mark. `None` when the server has no throttle report (older build, or no NVML).
    pub fn thermal_mark(&self) -> Option<f64> {
        let value = self.get("/v1/thermal").ok()?;
        value["available"].as_bool().filter(|a| *a)?;
        value["now_s"].as_f64()
    }

    /// The GPU throttle summary since `from` (a [`Self::thermal_mark`]) until now.
    pub fn thermal_since(&self, from: Option<f64>) -> Option<Value> {
        let value = self.get(&format!("/v1/thermal?from={}", from?)).ok()?;
        value.get("window").cloned().filter(|w| !w.is_null())
    }

    /// A streamed chat completion. `body` needs no `model` or `stream`;
    /// `probe` registers benchmark diagnostics for this request.
    pub fn chat(&self, mut body: Value, probe: Option<ProbeSpec>) -> Result<Chat> {
        self.check()?;
        crate::request::validate(&self.configuration, &body)?;
        body["speculative_acceptance"] = json!(self.configuration.speculative_acceptance);
        body["model"] = json!(self.model);
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
        let mut request = self.agent.post(&format!("{}/v1/chat/completions", self.base))
            .set("content-type", "application/json");
        if let Some(token) = &self.token {
            request = request.set(BENCH_HEADER, token);
        }
        let registered = probe.map(|spec| probe::registry().register(spec));
        if let Some((id, _)) = &registered {
            request = request.set(probe::HEADER, id);
        }
        let started = Instant::now();
        let response = match self.authorize(request).send_string(&body.to_string()) {
            Ok(response) => response,
            Err(ureq::Error::Status(code, response)) => {
                let text = response.into_string().unwrap_or_default();
                return Err(UpstreamHttpError { code, body: text }.into());
            }
            Err(error) => return Err(error).context("chat request"),
        };
        let mut chat = Chat::default();
        let (mut first, mut last, mut reasoning_end) = (None::<f64>, None::<f64>, None::<f64>);
        let mut finish_reason = None;
        let mut tool_calls: Vec<Value> = Vec::new();
        for line in BufReader::new(response.into_reader()).lines() {
            if self.cancelled() {
                // Dropping the reader closes the connection: the server stops this request.
                return Err(Cancelled.into());
            }
            let line = line.context("reading the event stream")?;
            let Some(data) = line.strip_prefix("data: ") else { continue };
            if data == "[DONE]" {
                break;
            }
            let at = started.elapsed().as_secs_f64();
            let event: Value = serde_json::from_str(data).with_context(|| format!("event {data}"))?;
            if let Some(error) = event.get("error") {
                bail!("stream error: {error}");
            }
            if let Some(usage) = event.get("usage").filter(|u| !u.is_null()) {
                chat.usage = usage.clone();
            }
            let Some(choice) = event["choices"].get(0) else { continue };
            let delta = &choice["delta"];
            let mut produced = false;
            if let Some(text) = delta["reasoning_content"].as_str().filter(|t| !t.is_empty()) {
                chat.reasoning.push_str(text);
                produced = true;
            }
            let mut answer = false;
            if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
                chat.content.push_str(text);
                produced = true;
                answer = true;
            }
            if let Some(calls) = delta["tool_calls"].as_array().filter(|c| !c.is_empty()) {
                merge_tool_calls(&mut tool_calls, calls);
                produced = true;
                answer = true;
            }
            if produced {
                first.get_or_insert(at);
                last = Some(at);
            }
            if answer && !chat.reasoning.is_empty() {
                reasoning_end.get_or_insert(at);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_string());
                last.get_or_insert(at);
            }
        }
        let total = started.elapsed().as_secs_f64();
        let usage = &chat.usage;
        let number = |v: &Value| v.as_u64().unwrap_or(0);
        chat.timing = StreamTiming {
            prompt_tokens: number(&usage["prompt_tokens"]),
            completion_tokens: number(&usage["completion_tokens"]),
            cached_tokens: usage.get("prompt_cache_hit_tokens").map(number)
                .unwrap_or_else(|| number(&usage["prompt_tokens_details"]["cached_tokens"])),
            reasoning_tokens: number(&usage["completion_tokens_details"]["reasoning_tokens"]),
            ttft_s: first.unwrap_or(total),
            total_s: total,
            decode_s: match (first, last) { (Some(a), Some(b)) => (b - a).max(0.0), _ => 0.0 },
            reasoning_end_s: reasoning_end,
            finish_reason,
        };
        chat.tool_calls = tool_calls;
        if let Some((_, probe)) = &registered {
            if probe.spec.record_state && probe.record().engine.as_deref() == Some("deepseek_v41") {
                probe.wait_state()?;
            }
        }
        chat.probe = registered.map(|(_, probe)| probe.record());
        Ok(chat)
    }
}

/// Streamed tool-call deltas assembled by index.
fn merge_tool_calls(calls: &mut Vec<Value>, deltas: &[Value]) {
    for delta in deltas {
        let index = delta["index"].as_u64().unwrap_or(calls.len() as u64) as usize;
        while calls.len() <= index {
            calls.push(json!({"id": "", "type": "function", "function": {"name": "", "arguments": ""}}));
        }
        let call = &mut calls[index];
        if let Some(id) = delta["id"].as_str() {
            call["id"] = json!(id);
        }
        if let Some(name) = delta["function"]["name"].as_str() {
            let joined = format!("{}{}", call["function"]["name"].as_str().unwrap_or(""), name);
            call["function"]["name"] = json!(joined);
        }
        if let Some(arguments) = delta["function"]["arguments"].as_str() {
            let joined = format!("{}{}", call["function"]["arguments"].as_str().unwrap_or(""), arguments);
            call["function"]["arguments"] = json!(joined);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fabric_mark_waits_for_new_peers_to_advance() {
        use std::io::Write;
        use cuteafd_transport::traffic::{HostSample, PortSample, Snapshot};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            for step in 0..4 {
                let mut sample = Snapshot::default();
                for (name, time) in [("coordinator", step as f64), ("peer", (step as f64 - 1.0).max(0.0))] {
                    if time > 0.0 {
                        sample.hosts.insert(name.into(), HostSample { incarnation: name.into(), sampled_s: time,
                            ports: vec![PortSample { device: "nic".into(), port: 1, ..Default::default() }], ..Default::default() });
                    }
                }
                let body = serde_json::to_vec(&sample).unwrap();
                let (mut socket, _) = listener.accept().unwrap();
                socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                let mut reader = BufReader::new(socket.try_clone().unwrap());
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" { break; }
                }
                write!(socket, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
                socket.write_all(&body).unwrap();
            }
        });
        let client = Client::new(&base, None, Arc::new(AtomicBool::new(false))).with_fabric(true);
        let mark = client.fabric_mark().unwrap();
        assert_eq!(mark.hosts["peer"].sampled_s, 2.0);
        worker.join().unwrap();
    }

    #[test]
    fn fabric_is_opt_in_without_http_or_background_work() {
        let client = Client::new("not-a-url", None, Arc::new(AtomicBool::new(false)));
        assert!(client.fabric().is_none());
        assert!(client.fabric_mark().is_none());
        assert!(client.fabric_subscription().is_none());
    }

    #[test]
    fn chat_rejection_has_a_typed_upstream_status() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut request = Vec::new();
            let mut byte = [0];
            while !request.ends_with(b"\r\n\r\n") {
                socket.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            let headers = String::from_utf8(request).unwrap();
            let length: usize = headers.lines().find_map(|line| line.split_once(':')
                .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .map(|(_, value)| value.trim().parse().unwrap())).unwrap();
            let mut body = vec![0; length];
            socket.read_exact(&mut body).unwrap();
            socket.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 26\r\nConnection: close\r\n\r\nvision encoder unavailable").unwrap();
        });
        let client = Client::new(&base, None, Arc::new(AtomicBool::new(false)));
        let error = client.chat(json!({"messages": []}), None).unwrap_err();
        server.join().unwrap();
        let upstream = error.downcast_ref::<UpstreamHttpError>().unwrap();
        assert_eq!(upstream.code, 503);
        assert_eq!(upstream.body, "vision encoder unavailable");
    }

    #[test]
    fn media_probe_uses_http_token_and_model() {
        use std::io::Read;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut reader = BufReader::new(&mut socket);
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            assert!(line.starts_with("POST /v1/bench/probe "));
            let (mut length, mut auth) = (0usize, false);
            loop {
                line.clear(); reader.read_line(&mut line).unwrap();
                if line == "\r\n" { break; }
                if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
                if line.to_lowercase().starts_with("x-cuteafd-bench: secret") { auth = true; }
            }
            assert!(auth);
            let mut bytes = vec![0; length]; reader.read_exact(&mut bytes).unwrap();
            let request: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(request["body"]["model"], "model");
            assert_eq!(request["body"]["stream"], false);
            assert_eq!(request["body"]["speculative_acceptance"], json!({"mode":"typical", "epsilon":0.1_f32, "delta":2.0_f32}));
            assert_eq!(request["spec"]["media"][0]["key"], "key");
            drop(reader);
            use std::io::Write;
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
        });
        let mut client = Client::new(&format!("http://{address}"), Some("secret".into()), Arc::new(AtomicBool::new(false)));
        client.model = "model".into();
        client.configuration.speculative_acceptance = cuteafd_core::SpeculativeAcceptance::Typical { epsilon: 0.1, delta: 2.0 };
        client.media_probe(json!({"messages": [], "speculative_acceptance":{"mode":"exact"}}), json!({"media": [{"key":"key"}]})).unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn tool_call_deltas_merge_by_index() {
        let mut calls = Vec::new();
        merge_tool_calls(&mut calls, &[json!({"index": 0, "id": "c1", "function": {"name": "get_", "arguments": "{\"ci"}})]);
        merge_tool_calls(&mut calls, &[json!({"index": 0, "function": {"name": "weather", "arguments": "ty\": 1}"}})]);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], "{\"city\": 1}");
        assert_eq!(calls[0]["id"], "c1");
    }
}
