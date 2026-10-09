//! Payload-free request accounting. The final handle emits without blocking.
use axum::{body::{Body, Bytes}, extract::{Request, State}, http::{HeaderMap, HeaderValue}, middleware::Next, response::Response};
use http_body::{Body as _, Frame, SizeHint};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{pin::Pin, sync::{atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed}, Arc, OnceLock}, task::{Context, Poll}, time::{Instant, SystemTime, UNIX_EPOCH}};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Record {
    pub rid: String, pub ts_ms: i64, pub protocol: String, pub route: String, pub method: String,
    pub client_kind: String, pub client_ua: Option<String>, pub key_label: Option<String>, pub client_ip: Option<String>,
    pub model_requested: Option<String>, pub model_served: Option<String>, pub session_id: Option<String>,
    pub session_source: Option<String>, pub turn_index: Option<u64>, pub stream: bool,
    pub n_items: Option<u64>, pub n_tools: Option<u64>, pub n_images: Option<u64>, pub n_audio: Option<u64>,
    pub tokens_in: Option<u64>, pub tokens_cached: Option<u64>, pub tokens_out: Option<u64>, pub tokens_reasoning: Option<u64>,
    pub draft_proposed: Option<u64>, pub draft_accepted: Option<u64>, pub rounds: Option<u64>,
    pub t_queue_ms: Option<f64>, pub t_admit_ms: Option<f64>, pub t_ttft_ms: Option<f64>, pub t_total_ms: Option<f64>,
    pub prefill_tps: Option<f64>, pub decode_tps: Option<f64>, pub concurrency_http: Option<u64>, pub concurrency_engine: Option<u64>,
    pub status: u16, pub outcome: String, pub stop_reason: Option<String>, pub error_class: Option<String>,
    pub bytes_in: Option<u64>, pub bytes_out: Option<u64>, pub bench: bool,
}
#[derive(Default, Serialize)]
pub struct Counters {
    pub recorded: AtomicU64, pub dropped: AtomicU64, pub log_recorded: AtomicU64,
    pub log_dropped: AtomicU64, pub db_bytes: AtomicU64, pub log_bytes: AtomicU64,
}
impl Counters {
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({"recorded": self.recorded.load(Relaxed), "dropped": self.dropped.load(Relaxed),
            "log_recorded": self.log_recorded.load(Relaxed), "log_dropped": self.log_dropped.load(Relaxed),
            "db_bytes": self.db_bytes.load(Relaxed), "log_bytes": self.log_bytes.load(Relaxed)})
    }
}
pub trait UsageSink: Send + Sync + 'static {
    fn record(&self, record: Record);
    fn counters(&self) -> &Counters;
    fn client_ip(&self) -> bool { false }
}
/// Handler-owned metadata only. No payload or headers can be attached here.
#[derive(Clone, Default)]
pub struct Details {
    pub model_requested: Option<String>, pub model_served: Option<String>, pub session_id: Option<String>,
    pub session_source: Option<String>, pub stream: bool, pub n_items: Option<u64>, pub n_tools: Option<u64>,
    pub n_images: Option<u64>, pub n_audio: Option<u64>, pub stop_reason: Option<String>, pub error_class: Option<String>,
}
struct Inner {
    record: Record, sink: Arc<dyn UsageSink>, start: Instant, details: OnceLock<Details>,
    status: AtomicU64, complete: AtomicU64, bytes_out: AtomicU64, tokens_in: AtomicU64,
    tokens_cached: AtomicU64, tokens_out: AtomicU64, tokens_reasoning: AtomicU64,
    proposed: AtomicU64, accepted: AtomicU64, rounds: AtomicU64,
    admit: OnceLock<(f64, u64)>, first: OnceLock<f64>, queue: OnceLock<f64>,
}
/// Opaque, cloneable handle usable by gateway and engine admission tickets.
#[derive(Clone)]
pub struct UsageHandle(Arc<Inner>);
pub type UsageScope = UsageHandle;
impl UsageHandle {
    pub fn new(record: Record, sink: Arc<dyn UsageSink>) -> Self {
        Self(Arc::new(Inner { record, sink, start: Instant::now(), details: OnceLock::new(),
            status: AtomicU64::new(0), complete: AtomicU64::new(0), bytes_out: AtomicU64::new(0),
            tokens_in: AtomicU64::new(u64::MAX), tokens_cached: AtomicU64::new(u64::MAX),
            tokens_out: AtomicU64::new(u64::MAX), tokens_reasoning: AtomicU64::new(u64::MAX),
            proposed: AtomicU64::new(0), accepted: AtomicU64::new(0), rounds: AtomicU64::new(0),
            admit: OnceLock::new(), first: OnceLock::new(), queue: OnceLock::new() }))
    }
    pub fn rid(&self) -> &str { &self.0.record.rid }
    pub fn details(&self, details: Details) { let _ = self.0.details.set(details); }
    pub fn admitted(&self, active: u64) { let _ = self.0.admit.set((self.0.start.elapsed().as_secs_f64() * 1000., active)); }
    pub fn queued(&self, ms: f64) { let _ = self.0.queue.set(ms); }
    pub fn first_token(&self) { let _ = self.0.first.set(self.0.start.elapsed().as_secs_f64() * 1000.); }
    pub fn tokens(&self, input: u64, cached: u64, output: u64, reasoning: u64) {
        for (field, value) in [(&self.0.tokens_in, input), (&self.0.tokens_cached, cached), (&self.0.tokens_out, output), (&self.0.tokens_reasoning, reasoning)] { field.store(value, Relaxed); }
    }
    pub fn round(&self, proposed: u64, accepted: u64) {
        self.0.proposed.fetch_add(proposed, Relaxed); self.0.accepted.fetch_add(accepted, Relaxed); self.0.rounds.fetch_add(1, Relaxed);
    }
    pub fn finished(&self, status: u16) { self.0.status.store(status.into(), Relaxed); self.0.complete.store(1, Relaxed); }
}
impl Drop for Inner {
    fn drop(&mut self) {
        let mut r = self.record.clone();
        if let Some(d) = self.details.get() {
            r.model_requested = d.model_requested.clone(); r.model_served = d.model_served.clone();
            r.session_id = d.session_id.clone(); r.session_source = d.session_source.clone(); r.stream = d.stream;
            r.n_items = d.n_items; r.n_tools = d.n_tools; r.n_images = d.n_images; r.n_audio = d.n_audio;
            r.stop_reason = d.stop_reason.clone(); r.error_class = d.error_class.clone();
        }
        let optional = |v: &AtomicU64| { let n = v.load(Relaxed); (n != u64::MAX).then_some(n) };
        r.tokens_in = optional(&self.tokens_in); r.tokens_cached = optional(&self.tokens_cached);
        r.tokens_out = optional(&self.tokens_out); r.tokens_reasoning = optional(&self.tokens_reasoning);
        if self.rounds.load(Relaxed) > 0 { r.draft_proposed = Some(self.proposed.load(Relaxed)); r.draft_accepted = Some(self.accepted.load(Relaxed)); r.rounds = Some(self.rounds.load(Relaxed)); }
        r.t_total_ms = Some(self.start.elapsed().as_secs_f64() * 1000.); r.t_ttft_ms = self.first.get().copied(); r.t_queue_ms = self.queue.get().copied();
        if let Some(&(ms, active)) = self.admit.get() { r.t_admit_ms = Some(ms); r.concurrency_engine = Some(active); }
        if let (Some(first), Some(admit), Some(input)) = (r.t_ttft_ms, r.t_admit_ms, r.tokens_in) {
            if first > admit { r.prefill_tps = Some(input.saturating_sub(r.tokens_cached.unwrap_or(0)) as f64 * 1000. / (first - admit)); }
        }
        if let (Some(first), Some(total), Some(output)) = (r.t_ttft_ms, r.t_total_ms, r.tokens_out) {
            if total > first { r.decode_tps = Some(output.saturating_sub(1) as f64 * 1000. / (total - first)); }
        }
        r.status = self.status.load(Relaxed) as u16; r.bytes_out = Some(self.bytes_out.load(Relaxed));
        r.outcome = if self.complete.load(Relaxed) == 0 { "cancelled" } else { match r.status { 401 | 403 => "auth", 429 => "overloaded", 400..=499 => "client_error", 500.. => "engine_error", _ => "ok" } }.into();
        self.sink.record(r);
    }
}
pub fn key_label(key: &str) -> String { format!("k:{:x}", Sha256::digest(key.as_bytes()))[..10].into() }
pub fn client_kind(headers: &HeaderMap) -> &'static str {
    let ua = headers.get("user-agent").and_then(|v| v.to_str().ok()).unwrap_or("").to_ascii_lowercase();
    if headers.contains_key("x-cuteafd-bench") { "bench" }
    else if ua.starts_with("claude-cli/") || headers.get("x-app").is_some_and(|v| v == "cli") { "claude_code" }
    else if ua.starts_with("codex_cli_rs/") || headers.contains_key("originator") { "codex" }
    else if ua.contains("openai") { "openai_sdk" } else if ua.contains("anthropic") { "anthropic_sdk" }
    else if ua.starts_with("curl/") { "curl" } else if ua.contains("mozilla/") { "browser" } else { "other" }
}
#[derive(Clone)]
pub struct Middleware { pub sink: Arc<dyn UsageSink>, pub inflight: Arc<AtomicUsize> }
impl Middleware { pub fn new(sink: Arc<dyn UsageSink>) -> Self { Self { sink, inflight: Arc::new(AtomicUsize::new(0)) } } }
pub async fn track(State(state): State<Middleware>, mut request: Request, next: Next) -> Response {
    let path = request.uri().path().to_owned();
    // Queries (including the unlock token), auth headers and payloads never enter Record.
    let protocol = match path.as_str() { "/v1/chat/completions" => "chat", "/v1/completions" => "completions", "/v1/messages" => "messages", "/v1/messages/count_tokens" => "count_tokens", "/v1/responses" => "responses", "/v1/realtime" => "realtime", "/v1/models" => "models", _ => "other" };
    let headers = request.headers();
    let scope = UsageHandle::new(Record { rid: uuid::Uuid::new_v4().to_string(),
        ts_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as i64,
        protocol: protocol.into(), route: path.clone(), method: request.method().to_string(),
        client_kind: client_kind(headers).into(), client_ua: headers.get("user-agent").and_then(|v| v.to_str().ok()).map(|s| s.chars().take(120).collect()),
        key_label: crate::gateway::auth::presented(headers).map(key_label),
        session_id: headers.get("x-session-id").and_then(|v| v.to_str().ok()).map(str::to_owned),
        session_source: headers.contains_key("x-session-id").then(|| "explicit".into()),
        client_ip: if state.sink.client_ip() { request.extensions().get::<axum::extract::ConnectInfo<std::net::SocketAddr>>().map(|a| a.0.ip().to_string()) } else { None },
        concurrency_http: Some(state.inflight.fetch_add(1, Relaxed) as u64 + 1),
        bytes_in: headers.get("content-length").and_then(|v| v.to_str().ok()).and_then(|s| s.parse().ok()),
        bench: client_kind(headers) == "bench", ..Record::default() }, state.sink.clone());
    request.extensions_mut().insert(scope.clone());
    let mut response = next.run(request).await;
    response.headers_mut().insert("x-request-id", HeaderValue::from_str(scope.rid()).expect("UUID header"));
    if path == "/v1/stats" && response.status().is_success() {
        let (parts, body) = response.into_parts();
        match axum::body::to_bytes(body, 16 << 20).await {
            Ok(bytes) => { let body = match serde_json::from_slice::<serde_json::Value>(&bytes) {
                Ok(mut v) if v.is_object() => { v["usage"] = state.sink.counters().snapshot(); Body::from(serde_json::to_vec(&v).expect("stats JSON")) }, _ => Body::from(bytes) };
                response = Response::from_parts(parts, body); response.headers_mut().remove("content-length"); }
            Err(_) => response = Response::from_parts(parts, Body::empty()),
        }
    }
    scope.0.status.store(response.status().as_u16().into(), Relaxed);
    let (parts, body) = response.into_parts();
    Response::from_parts(parts, Body::new(TrackedBody { body: Box::pin(body), scope, inflight: state.inflight }))
}
struct TrackedBody { body: Pin<Box<Body>>, scope: UsageHandle, inflight: Arc<AtomicUsize> }
impl http_body::Body for TrackedBody {
    type Data = Bytes; type Error = axum::Error;
    fn poll_frame(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let result = self.body.as_mut().poll_frame(cx);
        match &result { Poll::Ready(None) => { self.scope.0.complete.store(1, Relaxed); },
            Poll::Ready(Some(Ok(frame))) => { if let Some(bytes) = frame.data_ref() { self.scope.0.bytes_out.fetch_add(bytes.len() as u64, Relaxed); } }, _ => {} }
        result
    }
    fn is_end_stream(&self) -> bool { self.body.is_end_stream() }
    fn size_hint(&self) -> SizeHint { self.body.size_hint() }
}
impl Drop for TrackedBody {
    fn drop(&mut self) { if self.body.is_end_stream() { self.scope.0.complete.store(1, Relaxed); } self.inflight.fetch_sub(1, Relaxed); }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Sink(std::sync::Mutex<Vec<Record>>, Counters);
    impl UsageSink for Sink { fn record(&self, r: Record) { self.0.lock().unwrap().push(r); } fn counters(&self) -> &Counters { &self.1 } }
    #[test]
    fn last_reference_and_cancelled() {
        let sink = Arc::new(Sink(std::sync::Mutex::new(vec![]), Counters::default()));
        let scope = UsageScope::new(Record::default(), sink.clone()); let other = scope.clone(); drop(scope);
        assert!(sink.0.lock().unwrap().is_empty()); drop(other); assert_eq!(sink.0.lock().unwrap()[0].outcome, "cancelled");
    }
    #[test]
    fn labels_and_classifier() { assert_eq!(key_label("secret").len(), 10); let mut h = HeaderMap::new(); h.insert("user-agent", "codex_cli_rs/1".parse().unwrap()); assert_eq!(client_kind(&h), "codex"); }
}
