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
    agent: ureq::Agent,
    token: Option<String>,
    api_key: Option<cuteafd_api::openai::auth::ApiKey>,
    cancel: Arc<AtomicBool>,
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
        Self { base: base.trim_end_matches('/').to_string(), model: String::new(), agent, token, cancel, api_key: None }
    }

    pub fn with_api_key(mut self, key: Option<cuteafd_api::openai::auth::ApiKey>) -> Self {
        self.api_key = key;
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

    pub fn stats(&self) -> Result<Value> {
        self.get("/v1/stats")
    }

    /// A streamed chat completion. `body` needs no `model` or `stream`;
    /// `probe` registers benchmark diagnostics for this request.
    pub fn chat(&self, body: Value, probe: Option<ProbeSpec>) -> Result<Chat> {
        self.chat_with_output(body, probe, || {})
    }

    /// A cancellable server-local stream for qualification panels. An absolute
    /// deadline covers connect, headers and body, including non-output keepalives.
    pub(crate) async fn chat_bounded(&self, mut body: Value, probe: Option<ProbeSpec>,
        deadline: Instant, mut abort: tokio::sync::watch::Receiver<bool>,
        mut output: impl FnMut()) -> Result<Chat> {
        use http_body_util::{BodyExt, Full};
        use hyper_util::rt::TokioIo;
        self.check()?;
        body["model"] = json!(self.model);
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
        let registered = probe.map(|spec| probe::registry().register(spec));
        let started = Instant::now();
        let work = async {
            let uri: hyper::Uri = format!("{}/v1/chat/completions", self.base).parse()?;
            let address = bounded_address(&uri, crate::context::loopback().as_deref())?;
            let socket = tokio::net::TcpStream::connect((address, uri.port_u16().unwrap_or(80))).await?;
            let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(socket)).await?;
            // Aborting this task drops the socket, even while body reads are silent.
            let _connection = AbortConnection(tokio::spawn(async move { let _ = connection.await; }));
            let mut request = hyper::Request::builder().method("POST")
                .uri(uri.path_and_query().context("probe path missing")?.as_str())
                .header("host", uri.authority().context("probe authority missing")?.as_str())
                .header("content-type", "application/json");
            if let Some(key) = &self.api_key { request = request.header("authorization", key.authorization()); }
            else if let Some(token) = &self.token { request = request.header("authorization", format!("Bearer {token}")); }
            if let Some(token) = &self.token { request = request.header(BENCH_HEADER, token); }
            if let Some((id, _)) = &registered { request = request.header(probe::HEADER, id); }
            let response = sender.send_request(request.body(Full::new(bytes::Bytes::from(body.to_string())))?).await?;
            let code = response.status().as_u16();
            let mut reader = response.into_body();
            if code != 200 {
                let bytes = reader.collect().await?.to_bytes();
                return Err(UpstreamHttpError { code, body: String::from_utf8_lossy(&bytes).into_owned() }.into());
            }
            let mut state = ChatStream::default();
            let mut pending = Vec::new();
            let mut done = false;
            while let Some(frame) = reader.frame().await {
                let frame = frame?;
                let Ok(data) = frame.into_data() else { continue; };
                pending.extend_from_slice(&data);
                while let Some(end) = pending.iter().position(|&b| b == b'\n') {
                    let line = std::str::from_utf8(&pending[..end])?.trim_end_matches('\r');
                    done = state.line(line, started.elapsed().as_secs_f64(), &mut output)?;
                    pending.drain(..=end);
                    if done { break; }
                }
                anyhow::ensure!(pending.len() <= 1024 * 1024, "probe event line too large");
                if done { break; }
            }
            anyhow::ensure!(done, "probe stream ended without DONE");
            let mut chat = state.finish(started.elapsed().as_secs_f64());
            chat.probe = registered.as_ref().map(|(_, probe)| probe.record());
            Ok(chat)
        };
        let cancelled = async {
            loop {
                if self.cancelled() || *abort.borrow() { break; }
                tokio::select! {
                    _ = abort.changed() => {},
                    _ = tokio::time::sleep(Duration::from_millis(20)) => {},
                }
            }
        };
        tokio::select! {
            biased;
            _ = cancelled => Err(Cancelled.into()),
            result = tokio::time::timeout_at(deadline.into(), work) =>
                result.context("prefill-share panel deadline exceeded")?,
        }
    }

    /// Qualification panels may wait for live SSE output before injecting work.
    pub(crate) fn chat_with_output(&self, mut body: Value, probe: Option<ProbeSpec>,
        mut output: impl FnMut()) -> Result<Chat> {
        self.check()?;
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
        let mut state = ChatStream::default();
        for line in BufReader::new(response.into_reader()).lines() {
            self.check()?;
            if state.line(&line.context("reading the event stream")?, started.elapsed().as_secs_f64(), &mut output)? {
                break;
            }
        }
        let mut chat = state.finish(started.elapsed().as_secs_f64());
        chat.probe = registered.map(|(_, probe)| probe.record());
        Ok(chat)
    }
}

fn bounded_address(uri: &hyper::Uri, server: Option<&str>) -> Result<std::net::IpAddr> {
    anyhow::ensure!(uri.scheme_str() == Some("http"), "bounded probes require local HTTP");
    let address: std::net::IpAddr = uri.host().context("probe host missing")?
        .trim_matches(['[', ']']).parse().context("bounded probes require a local IP")?;
    let own_endpoint = server.and_then(|s| s.parse::<hyper::Uri>().ok())
        .is_some_and(|s| s.scheme_str() == uri.scheme_str() && s.authority() == uri.authority());
    anyhow::ensure!(address.is_loopback() || own_endpoint,
        "bounded probes require loopback or the server's own listen address");
    Ok(address)
}

struct AbortConnection(tokio::task::JoinHandle<()>);
impl Drop for AbortConnection {
    fn drop(&mut self) { self.0.abort(); }
}

#[derive(Default)]
struct ChatStream {
    chat: Chat,
    first: Option<f64>,
    last: Option<f64>,
    reasoning_end: Option<f64>,
    finish_reason: Option<String>,
}
impl ChatStream {
    fn line(&mut self, line: &str, at: f64, output: &mut impl FnMut()) -> Result<bool> {
        let Some(data) = line.strip_prefix("data: ") else { return Ok(false); };
        if data == "[DONE]" {
            return Ok(true);
        }
        let event: Value = serde_json::from_str(data).with_context(|| format!("event {data}"))?;
        if let Some(error) = event.get("error") {
            bail!("stream error: {error}");
        }
        if let Some(usage) = event.get("usage").filter(|u| !u.is_null()) {
            self.chat.usage = usage.clone();
        }
        let Some(choice) = event["choices"].get(0) else { return Ok(false); };
        let delta = &choice["delta"];
        let mut produced = false;
        if let Some(text) = delta["reasoning_content"].as_str().filter(|t| !t.is_empty()) {
            self.chat.reasoning.push_str(text);
            produced = true;
        }
        let mut answer = false;
        if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            self.chat.content.push_str(text);
            produced = true;
            answer = true;
        }
        if let Some(calls) = delta["tool_calls"].as_array().filter(|c| !c.is_empty()) {
            merge_tool_calls(&mut self.chat.tool_calls, calls);
            produced = true;
            answer = true;
        }
        if produced {
            self.first.get_or_insert(at);
            self.last = Some(at);
            output();
        }
        if answer && !self.chat.reasoning.is_empty() {
            self.reasoning_end.get_or_insert(at);
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish_reason = Some(reason.to_string());
            self.last.get_or_insert(at);
        }
        Ok(false)
    }

    fn finish(mut self, total: f64) -> Chat {
        let usage = &self.chat.usage;
        let number = |v: &Value| v.as_u64().unwrap_or(0);
        self.chat.timing = StreamTiming {
            prompt_tokens: number(&usage["prompt_tokens"]),
            completion_tokens: number(&usage["completion_tokens"]),
            cached_tokens: usage.get("prompt_cache_hit_tokens").map(number)
                .unwrap_or_else(|| number(&usage["prompt_tokens_details"]["cached_tokens"])),
            reasoning_tokens: number(&usage["completion_tokens_details"]["reasoning_tokens"]),
            ttft_s: self.first.unwrap_or(total),
            total_s: total,
            decode_s: match (self.first, self.last) { (Some(a), Some(b)) => (b - a).max(0.0), _ => 0.0 },
            reasoning_end_s: self.reasoning_end,
            finish_reason: self.finish_reason,
        };

        self.chat
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
    fn bounded_probes_allow_only_loopback_or_own_bound_endpoint() {
        for base in ["http://10.55.0.22:8000", "http://[fd00::22]:8000"] {
            let uri = format!("{base}/v1/chat/completions").parse().unwrap();
            assert!(bounded_address(&uri, Some(base)).is_ok());
            assert!(bounded_address(&uri, Some("http://10.55.0.23:8000")).is_err());
            assert!(bounded_address(&uri, None).is_err());
        }
        let uri = "http://127.0.0.1:8000/v1/chat/completions".parse().unwrap();
        assert!(bounded_address(&uri, None).unwrap().is_loopback());
        let uri = "https://127.0.0.1:8000/v1/chat/completions".parse().unwrap();
        assert!(bounded_address(&uri, None).is_err());
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
            assert_eq!(request["spec"]["media"][0]["key"], "key");
            drop(reader);
            use std::io::Write;
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
        });
        let mut client = Client::new(&format!("http://{address}"), Some("secret".into()), Arc::new(AtomicBool::new(false)));
        client.model = "model".into();
        client.media_probe(json!({"messages": []}), json!({"media": [{"key":"key"}]})).unwrap();
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
