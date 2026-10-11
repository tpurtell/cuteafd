//! Payload-free request accounting. The final handle emits without blocking.
use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::HeaderMap,
    middleware::Next,
    response::Response,
};
use crate::usage_log::{loggable, FrameTee, LogRecord, LogSink, ResponsePayload, REQUEST_CAP, RESPONSE_CAP};
use http_body::{Body as _, Frame, SizeHint};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed},
        Arc, OnceLock,
    },
    task::{Context, Poll},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Record {
    pub rid: String,
    pub ts_ms: i64,
    pub protocol: String,
    pub route: String,
    pub method: String,
    pub client_kind: String,
    pub client_ua: Option<String>,
    pub key_label: Option<String>,
    pub client_ip: Option<String>,
    pub model_requested: Option<String>,
    pub model_served: Option<String>,
    pub session_id: Option<String>,
    pub session_source: Option<String>,
    pub turn_index: Option<u64>,
    pub stream: bool,
    pub n_items: Option<u64>,
    pub n_tools: Option<u64>,
    pub n_images: Option<u64>,
    pub n_audio: Option<u64>,
    pub tokens_in: Option<u64>,
    pub tokens_cached: Option<u64>,
    pub tokens_out: Option<u64>,
    pub tokens_reasoning: Option<u64>,
    pub draft_proposed: Option<u64>,
    pub draft_accepted: Option<u64>,
    pub rounds: Option<u64>,
    pub t_queue_ms: Option<f64>,
    pub t_admit_ms: Option<f64>,
    pub t_ttft_ms: Option<f64>,
    pub t_total_ms: Option<f64>,
    /// Engine retirement (last token), from arrival; decode time is retire − first token.
    pub t_retire_ms: Option<f64>,
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    pub concurrency_http: Option<u64>,
    pub concurrency_engine: Option<u64>,
    pub status: u16,
    pub outcome: String,
    pub stop_reason: Option<String>,
    pub error_class: Option<String>,
    pub bytes_in: Option<u64>,
    pub bytes_out: Option<u64>,
    pub bench: bool,
}
#[derive(Default, Serialize)]
pub struct Counters {
    pub recorded: AtomicU64,
    pub dropped: AtomicU64,
    pub log_recorded: AtomicU64,
    pub log_dropped: AtomicU64,
    pub db_bytes: AtomicU64,
    pub log_bytes: AtomicU64,
    pub media_bytes: AtomicU64,
    pub media_files: AtomicU64,
}
impl Counters {
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({"recorded": self.recorded.load(Relaxed), "dropped": self.dropped.load(Relaxed),
            "log_recorded": self.log_recorded.load(Relaxed), "log_dropped": self.log_dropped.load(Relaxed),
            "db_bytes": self.db_bytes.load(Relaxed), "log_bytes": self.log_bytes.load(Relaxed),
            "media_bytes": self.media_bytes.load(Relaxed), "media_files": self.media_files.load(Relaxed)})
    }
}
pub trait UsageSink: Send + Sync + 'static {
    fn record(&self, record: Record);
    fn counters(&self) -> &Counters;
    fn client_ip(&self) -> bool {
        false
    }
    /// The full-log tier, when this sink has one.
    fn log_sink(&self) -> Option<Arc<dyn LogSink>> {
        None
    }
}
/// Handler-owned metadata only. No payload or headers can be attached here.
#[derive(Clone, Default)]
pub struct Details {
    pub model_requested: Option<String>,
    pub model_served: Option<String>,
    pub session_id: Option<String>,
    pub session_source: Option<String>,
    pub stream: bool,
    pub n_items: Option<u64>,
    pub n_tools: Option<u64>,
    pub n_images: Option<u64>,
    pub n_audio: Option<u64>,
    pub stop_reason: Option<String>,
    pub error_class: Option<String>,
}
struct Inner {
    record: Record,
    key_name: OnceLock<String>,
    sink: Arc<dyn UsageSink>,
    /// Present only while the full log is on and this protocol is loggable.
    log: Option<Arc<dyn LogSink>>,
    /// The full-log store, passed to WebSocket turn children.
    log_sink: Option<Arc<dyn LogSink>>,
    log_request: OnceLock<(Vec<Bytes>, bool)>,
    log_request_value: OnceLock<serde_json::Value>,
    log_response: OnceLock<ResponsePayload>,
    start: Instant,
    details: OnceLock<Details>,
    status: AtomicU64,
    complete: AtomicU64,
    bytes_out: AtomicU64,
    tokens_in: AtomicU64,
    tokens_cached: AtomicU64,
    tokens_out: AtomicU64,
    tokens_reasoning: AtomicU64,
    proposed: AtomicU64,
    accepted: AtomicU64,
    rounds: AtomicU64,
    admit: OnceLock<(f64, u64)>,
    first: OnceLock<f64>,
    queue: OnceLock<f64>,
    served: OnceLock<String>,
    session: OnceLock<(String, String)>,
    stop: OnceLock<String>,
    error: OnceLock<String>,
    terminal: AtomicU64,
    retired: OnceLock<f64>,
}
/// Opaque, cloneable handle usable by gateway and engine admission tickets.
#[derive(Clone)]
pub struct UsageHandle(Arc<Inner>);
pub type UsageScope = UsageHandle;
impl UsageHandle {
    pub fn key_name(&self, name: String) { let _ = self.0.key_name.set(name); }
    pub fn new(record: Record, sink: Arc<dyn UsageSink>) -> Self {
        let log_sink = sink.log_sink();
        Self::with_log(record, sink, log_sink)
    }
    /// A scope that also captures payloads when `log` is on for its protocol.
    fn with_log(record: Record, sink: Arc<dyn UsageSink>, log_sink: Option<Arc<dyn LogSink>>) -> Self {
        let log = log_sink
            .clone()
            .filter(|l| loggable(&record.protocol) && l.enabled() && (!record.bench || l.bench()));
        Self(Arc::new(Inner {
            record,
            key_name: OnceLock::new(),
            sink,
            log,
            log_sink,
            log_request: OnceLock::new(),
            log_request_value: OnceLock::new(),
            log_response: OnceLock::new(),
            start: Instant::now(),
            details: OnceLock::new(),
            status: AtomicU64::new(0),
            complete: AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
            tokens_in: AtomicU64::new(u64::MAX),
            tokens_cached: AtomicU64::new(u64::MAX),
            tokens_out: AtomicU64::new(u64::MAX),
            tokens_reasoning: AtomicU64::new(u64::MAX),
            proposed: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            rounds: AtomicU64::new(0),
            admit: OnceLock::new(),
            first: OnceLock::new(),
            queue: OnceLock::new(),
            served: OnceLock::new(),
            session: OnceLock::new(),
            stop: OnceLock::new(),
            error: OnceLock::new(),
            terminal: AtomicU64::new(0),
            retired: OnceLock::new(),
        }))
    }
    pub fn rid(&self) -> &str {
        &self.0.record.rid
    }
    /// True when this scope captures payloads for the full log.
    pub fn logging(&self) -> bool {
        self.0.log.is_some()
    }
    /// The turn's request as an owned structure (WebSocket front ends); the log
    /// writer serializes it. The closure runs only while the log is on.
    pub fn log_request(&self, body: impl FnOnce() -> serde_json::Value) {
        if self.0.log.is_some() {
            let _ = self.0.log_request_value.set(body());
        }
    }
    /// The turn's request exactly as the client sent it (one refcounted frame).
    pub fn log_request_bytes(&self, body: impl FnOnce() -> Bytes) {
        if self.0.log.is_some() {
            let _ = self.0.log_request.set((vec![body()], false));
        }
    }
    /// The turn's final response object (WebSocket front ends), serialized on the writer.
    pub fn log_response(&self, body: impl FnOnce() -> serde_json::Value) {
        if self.0.log.is_some() {
            let _ = self.0.log_response.set(ResponsePayload::Value(body()));
        }
    }
    pub fn details(&self, details: Details) {
        let _ = self.0.details.set(details);
    }
    pub fn served_model(&self, model: &str) {
        let _ = self.0.served.set(model.to_owned());
    }
    /// Higher-confidence handler sessions win over the prefix fallback.
    pub fn session(&self, id: String, source: &str) {
        if self.0.record.session_id.is_none() {
            let _ = self.0.session.set((id, source.to_owned()));
        }
    }
    pub fn session_id(&self) -> &str {
        self.0.record.session_id.as_deref()
            .or_else(|| self.0.session.get().map(|s| s.0.as_str()))
            .or_else(|| self.0.details.get().and_then(|d| d.session_id.as_deref()))
            .unwrap_or_else(|| self.rid())
    }
    pub fn cache_session(&self, key: &str) {
        self.session(session_hash(key), "cache_key");
    }
    pub fn stop(&self, reason: &str) {
        let _ = self.0.stop.set(reason.to_owned());
        if reason == "cancelled" { self.0.terminal.store(1, Relaxed); }
    }
    pub fn engine_error(&self, class: &str) {
        let _ = self.0.error.set(class.to_owned());
        self.0.terminal.store(2, Relaxed);
    }
    pub fn prompt_tokens(&self, input: u64, cached: u64) {
        self.0.tokens_in.store(input, Relaxed);
        self.0.tokens_cached.store(cached, Relaxed);
    }
    pub fn retired(&self, output: u64, reason: &str) {
        // API usage can include reasoning that the engine cannot classify.
        let _ = self.0.tokens_out.compare_exchange(u64::MAX, output, Relaxed, Relaxed);
        let _ = self.0.retired.set(self.0.start.elapsed().as_secs_f64() * 1000.);
        if reason == "cancelled" { self.stop(reason); }
        if reason == "failed" { self.engine_error("worker"); }
    }
    /// A websocket turn owns a distinct row and never retains the connection scope.
    pub fn child(&self, protocol: &str) -> Self {
        let mut record = self.0.record.clone();
        if let Some(name) = self.0.key_name.get() { record.key_label = Some(name.clone()); }
        record.rid = uuid::Uuid::new_v4().to_string();
        record.ts_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as i64;
        record.protocol = protocol.into();
        record.bytes_in = None;
        let child = Self::with_log(record, self.0.sink.clone(), self.0.log_sink.clone());
        child.0.status.store(200, Relaxed);
        child
    }
    pub fn admitted(&self, active: u64) {
        let _ = self
            .0
            .admit
            .set((self.0.start.elapsed().as_secs_f64() * 1000., active));
    }
    pub fn queued(&self, ms: f64) {
        let _ = self.0.queue.set(ms);
    }
    pub fn first_token(&self) {
        let _ = self
            .0
            .first
            .set(self.0.start.elapsed().as_secs_f64() * 1000.);
    }
    pub fn tokens(&self, input: u64, cached: u64, output: u64, reasoning: u64) {
        for (field, value) in [
            (&self.0.tokens_in, input),
            (&self.0.tokens_cached, cached),
            (&self.0.tokens_out, output),
            (&self.0.tokens_reasoning, reasoning),
        ] {
            field.store(value, Relaxed);
        }
    }
    pub fn round(&self, proposed: u64, accepted: u64) {
        self.0.proposed.fetch_add(proposed, Relaxed);
        self.0.accepted.fetch_add(accepted, Relaxed);
        self.0.rounds.fetch_add(1, Relaxed);
    }
    pub fn finished(&self, status: u16) {
        self.0.status.store(status.into(), Relaxed);
        self.0.complete.store(1, Relaxed);
    }
}
impl Drop for Inner {
    fn drop(&mut self) {
        let mut r = self.record.clone();
        if let Some(name) = self.key_name.get() { r.key_label = Some(name.clone()); }
        if let Some(d) = self.details.get() {
            r.model_requested = d.model_requested.clone();
            r.model_served = d.model_served.clone();
            if r.session_id.is_none() {
                r.session_id = d.session_id.clone();
                r.session_source = d.session_source.clone();
            }
            r.stream = d.stream;
            r.n_items = d.n_items;
            r.n_tools = d.n_tools;
            r.n_images = d.n_images;
            r.n_audio = d.n_audio;
            r.stop_reason = d.stop_reason.clone();
            r.error_class = d.error_class.clone();
        }
        if let Some(model) = self.served.get() { r.model_served = Some(model.clone()); }
        if r.session_id.is_none() {
            if let Some((id, source)) = self.session.get() {
                r.session_id = Some(id.clone());
                r.session_source = Some(source.clone());
            }
        }
        if let Some(stop) = self.stop.get() { r.stop_reason = Some(stop.clone()); }
        if let Some(error) = self.error.get() { r.error_class = Some(error.clone()); }
        let optional = |v: &AtomicU64| {
            let n = v.load(Relaxed);
            (n != u64::MAX).then_some(n)
        };
        r.tokens_in = optional(&self.tokens_in);
        r.tokens_cached = optional(&self.tokens_cached);
        r.tokens_out = optional(&self.tokens_out);
        r.tokens_reasoning = optional(&self.tokens_reasoning);
        if self.rounds.load(Relaxed) > 0 {
            r.draft_proposed = Some(self.proposed.load(Relaxed));
            r.draft_accepted = Some(self.accepted.load(Relaxed));
            r.rounds = Some(self.rounds.load(Relaxed));
        }
        r.t_total_ms = Some(self.start.elapsed().as_secs_f64() * 1000.);
        r.t_retire_ms = self.retired.get().copied();
        r.t_ttft_ms = self.first.get().copied();
        r.t_queue_ms = self.queue.get().copied();
        if let Some(&(ms, active)) = self.admit.get() {
            r.t_admit_ms = Some(ms);
            r.concurrency_engine = Some(active);
        }
        if let (Some(first), Some(admit), Some(input)) = (r.t_ttft_ms, r.t_admit_ms, r.tokens_in) {
            if first > admit {
                r.prefill_tps = Some(
                    input.saturating_sub(r.tokens_cached.unwrap_or(0)) as f64 * 1000.
                        / (first - admit),
                );
            }
        }
        if let (Some(first), Some(total), Some(output)) = (r.t_ttft_ms, r.t_total_ms, r.tokens_out)
        {
            let end = self.retired.get().copied().unwrap_or(total);
            if end > first {
                r.decode_tps = Some(output.saturating_sub(1) as f64 * 1000. / (end - first));
            }
        }
        r.status = self.status.load(Relaxed) as u16;
        r.bytes_out = Some(self.bytes_out.load(Relaxed));
        r.outcome = if self.terminal.load(Relaxed) == 2 {
            "engine_error"
        } else if self.terminal.load(Relaxed) == 1 || self.complete.load(Relaxed) == 0 {
            "cancelled"
        } else {
            match r.status {
                401 | 403 => "auth",
                429 => "overloaded",
                400..=499 => "client_error",
                500.. => "engine_error",
                _ => "ok",
            }
        }
        .into();
        if let Some(log) = self.log.take() {
            let (request, request_truncated) = self.log_request.take().unwrap_or_default();
            let request_value = self.log_request_value.take();
            let unauthenticated = matches!(r.status, 401 | 403);
            if (!request.is_empty() || request_value.is_some()) && !unauthenticated {
                log.record_log(LogRecord {
                    meta: r.clone(),
                    request,
                    request_value,
                    request_truncated,
                    response: self.log_response.take().unwrap_or_default(),
                });
            }
        }
        self.sink.record(r);
    }
}
impl std::fmt::Debug for UsageHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("UsageHandle") }
}
impl PartialEq for UsageHandle {
    fn eq(&self, _: &Self) -> bool { true }
}
#[axum::async_trait]
impl<S: Send + Sync> axum::extract::FromRequestParts<S> for UsageHandle {
    type Rejection = axum::http::StatusCode;
    async fn from_request_parts(parts: &mut axum::http::request::Parts, _: &S) -> Result<Self, Self::Rejection> {
        parts.extensions.get::<Self>().cloned().ok_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
    }
}
pub fn session_hash(key: &str) -> String {
    format!("{:x}", Sha256::digest(key.as_bytes()))[..16].into()
}
pub fn key_label(key: &str) -> String {
    format!("k:{:x}", Sha256::digest(key.as_bytes()))[..10].into()
}
pub fn client_kind(headers: &HeaderMap) -> &'static str {
    let ua = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    if headers.contains_key("x-cuteafd-bench") {
        // A label only; `Record::bench` (the recording exemption) needs a verified token.
        "bench"
    } else if ua.starts_with("claude-cli/") || headers.get("x-app").is_some_and(|v| v == "cli") {
        "claude_code"
    } else if ua.starts_with("codex_cli_rs/") || headers.contains_key("originator") {
        "codex"
    } else if ua.contains("openai") {
        "openai_sdk"
    } else if ua.contains("anthropic") {
        "anthropic_sdk"
    } else if ua.starts_with("curl/") {
        "curl"
    } else if ua.contains("mozilla/") {
        "browser"
    } else {
        "other"
    }
}
#[derive(Clone)]
pub struct Middleware {
    pub sink: Arc<dyn UsageSink>,
    pub inflight: Arc<AtomicUsize>,
    /// Verifies a benchmark run's token. Only a verified token marks a request
    /// as bench (and so exempt from recording by default); the header alone does not.
    pub bench_token: Option<Arc<dyn Fn(&str) -> bool + Send + Sync>>,
}
impl Middleware {
    pub fn new(sink: Arc<dyn UsageSink>) -> Self {
        Self {
            sink,
            inflight: Arc::new(AtomicUsize::new(0)),
            bench_token: None,
        }
    }
    pub fn with_bench(mut self, verify: Arc<dyn Fn(&str) -> bool + Send + Sync>) -> Self {
        self.bench_token = Some(verify);
        self
    }
    fn bench(&self, headers: &HeaderMap) -> bool {
        let Some(verify) = &self.bench_token else { return false };
        headers.get("x-cuteafd-bench").and_then(|v| v.to_str().ok()).is_some_and(|t| verify(t))
    }
}
/// The dashboards' own polling and static assets: recording them would make
/// the history mostly the viewer watching itself.
fn self_traffic(path: &str) -> bool {
    path == "/agent" || path.starts_with("/agent/") || path == "/" || path == "/usage" || path == "/bench" || path.starts_with("/assets/")
        || path.starts_with("/console/usage/") || path.starts_with("/v1/console") || path.starts_with("/v1/bench/")
        || path == "/health" || path == "/bench/banner.js"
}
pub async fn track(State(state): State<Middleware>, mut request: Request, next: Next) -> Response {
    let path = request.uri().path().to_owned();
    if self_traffic(&path) {
        return next.run(request).await;
    }
    // Queries (including the unlock token), auth headers and payloads never enter Record.
    let protocol = match path.as_str() {
        "/v1/chat/completions" => "chat",
        "/v1/completions" => "completions",
        "/v1/messages" => "messages",
        "/v1/messages/count_tokens" => "count_tokens",
        "/v1/responses" => "responses",
        "/v1/realtime" => "realtime",
        "/v1/models" => "models",
        _ => "other",
    };
    let headers = request.headers();
    let scope = UsageHandle::new(
        Record {
            rid: uuid::Uuid::new_v4().to_string(),
            ts_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64,
            protocol: if protocol == "realtime" { "realtime_session" } else { protocol }.into(),
            route: path.clone(),
            method: request.method().to_string(),
            client_kind: client_kind(headers).into(),
            client_ua: headers
                .get("user-agent")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.chars().take(120).collect()),
            key_label: crate::gateway::auth::presented(headers).map(key_label),
            session_id: headers
                .get("x-session-id")
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned),
            session_source: headers
                .contains_key("x-session-id")
                .then(|| "explicit".into()),
            client_ip: if state.sink.client_ip() {
                request
                    .extensions()
                    .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
                    .map(|a| a.0.ip().to_string())
            } else {
                None
            },
            concurrency_http: Some(state.inflight.fetch_add(1, Relaxed) as u64 + 1),
            bytes_in: headers
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse().ok()),
            bench: state.bench(headers),
            ..Record::default()
        },
        state.sink.clone(),
    );
    request.extensions_mut().insert(scope.clone());
    if scope.logging() {
        // One refcount per body frame; parsing and redaction run on the log writer.
        let (parts, body) = request.into_parts();
        request = Request::from_parts(
            parts,
            Body::new(TeeBody { body: Box::pin(body), tee: FrameTee::default(), scope: scope.clone() }),
        );
    }
    let mut response = next.run(request).await;
    if path == "/v1/stats" && response.status().is_success() {
        let (parts, body) = response.into_parts();
        match axum::body::to_bytes(body, 16 << 20).await {
            Ok(bytes) => {
                let body = match serde_json::from_slice::<serde_json::Value>(&bytes) {
                    Ok(mut v) if v.is_object() => {
                        v["usage"] = state.sink.counters().snapshot();
                        Body::from(serde_json::to_vec(&v).expect("stats JSON"))
                    }
                    _ => Body::from(bytes),
                };
                response = Response::from_parts(parts, body);
                response.headers_mut().remove("content-length");
            }
            Err(_) => response = Response::from_parts(parts, Body::empty()),
        }
    }
    scope
        .0
        .status
        .store(response.status().as_u16().into(), Relaxed);
    let (parts, body) = response.into_parts();
    let tee = scope.logging().then(|| {
        let sse = parts
            .headers
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"));
        (FrameTee::default(), sse)
    });
    Response::from_parts(
        parts,
        Body::new(TrackedBody {
            body: Box::pin(body),
            scope,
            inflight: state.inflight,
            tee,
        }),
    )
}
/// Clones request frames for the full log as the handler reads them.
struct TeeBody {
    body: Pin<Box<Body>>,
    tee: FrameTee,
    scope: UsageHandle,
}
impl http_body::Body for TeeBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let result = self.body.as_mut().poll_frame(cx);
        if let Poll::Ready(Some(Ok(frame))) = &result {
            if let Some(bytes) = frame.data_ref() {
                self.tee.push(bytes, REQUEST_CAP);
            }
        }
        result
    }
    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}
impl Drop for TeeBody {
    fn drop(&mut self) {
        let tee = std::mem::take(&mut self.tee);
        let _ = self.scope.0.log_request.set((tee.frames, tee.truncated));
    }
}
struct TrackedBody {
    body: Pin<Box<Body>>,
    scope: UsageHandle,
    inflight: Arc<AtomicUsize>,
    /// Response frames for the full log (refcounts) and whether they are SSE.
    tee: Option<(FrameTee, bool)>,
}
impl http_body::Body for TrackedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let result = self.body.as_mut().poll_frame(cx);
        match &result {
            Poll::Ready(None) => {
                self.scope.0.complete.store(1, Relaxed);
            }
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(bytes) = frame.data_ref() {
                    self.scope
                        .0
                        .bytes_out
                        .fetch_add(bytes.len() as u64, Relaxed);
                    let bytes = bytes.clone();
                    if let Some((tee, _)) = &mut self.tee {
                        tee.push(&bytes, RESPONSE_CAP);
                    }
                }
            }
            _ => {}
        }
        result
    }
    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}
impl Drop for TrackedBody {
    fn drop(&mut self) {
        if self.body.is_end_stream() {
            self.scope.0.complete.store(1, Relaxed);
        }
        if let Some((tee, sse)) = self.tee.take() {
            let _ = self.scope.0.log_response.set(ResponsePayload::Frames {
                frames: tee.frames,
                sse,
                truncated: tee.truncated,
            });
        }
        self.inflight.fetch_sub(1, Relaxed);
    }
}
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[derive(Default)]
    pub(crate) struct Sink(pub std::sync::Mutex<Vec<Record>>, Counters, pub Option<Arc<LogTape>>);
    impl UsageSink for Sink {
        fn record(&self, r: Record) {
            self.0.lock().unwrap_or_else(|e| e.into_inner()).push(r);
        }
        fn counters(&self) -> &Counters {
            &self.1
        }
        fn log_sink(&self) -> Option<Arc<dyn LogSink>> {
            self.2.clone().map(|l| l as Arc<dyn LogSink>)
        }
    }
    /// A full-log sink that keeps records in memory.
    #[derive(Default)]
    pub(crate) struct LogTape(pub std::sync::Mutex<Vec<LogRecord>>, pub std::sync::atomic::AtomicBool);
    impl LogSink for LogTape {
        fn enabled(&self) -> bool {
            !self.1.load(Relaxed)
        }
        fn record_log(&self, r: LogRecord) {
            self.0.lock().unwrap_or_else(|e| e.into_inner()).push(r);
        }
    }
    impl LogTape {
        pub fn request(&self, i: usize) -> serde_json::Value {
            let r = &self.0.lock().unwrap()[i];
            if let Some(v) = &r.request_value { return v.clone(); }
            let bytes: Vec<u8> = r.request.iter().flat_map(|b| b.iter().copied()).collect();
            serde_json::from_slice(&bytes).unwrap()
        }
        pub fn response_text(&self, i: usize) -> String {
            match &self.0.lock().unwrap()[i].response {
                ResponsePayload::Object(b) => String::from_utf8_lossy(b).into_owned(),
                ResponsePayload::Frames { frames, .. } => frames.iter().map(|f| String::from_utf8_lossy(f).into_owned()).collect(),
                ResponsePayload::Value(v) => v.to_string(),
                ResponsePayload::None => String::new(),
            }
        }
    }
    pub(crate) fn logging_sink() -> (Arc<Sink>, Arc<LogTape>) {
        let tape = Arc::new(LogTape::default());
        (Arc::new(Sink(Default::default(), Counters::default(), Some(tape.clone()))), tape)
    }

    /// The chat path: one refcount clone per frame; the stream's frames, the
    /// session and the metadata reach the log, and a failed auth stores nothing.
    #[tokio::test]
    async fn chat_capture_tees_request_and_stream_frames() {
        use tower::ServiceExt;
        let (sink, tape) = logging_sink();
        let app = axum::Router::new()
            .route("/v1/chat/completions", axum::routing::post(|body: Bytes| async move {
                let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
                if v["fail"] == true { return axum::response::IntoResponse::into_response(axum::http::StatusCode::UNAUTHORIZED); }
                let frames = ["data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"he\"}}]}\n\n", "data: [DONE]\n\n"];
                axum::response::IntoResponse::into_response(([("content-type", "text/event-stream")], Body::from_stream(futures::stream::iter(
                    frames.into_iter().map(|f| Ok::<_, std::convert::Infallible>(Bytes::from_static(f.as_bytes())))))))
            }))
            .route("/v1/models", axum::routing::get(|| async { "{}" }))
            .layer(axum::middleware::from_fn_with_state(Middleware::new(sink.clone()), track));
        let send = |path: &str, body: &str| {
            app.clone().oneshot(axum::http::Request::post(path).header("user-agent", "claude-cli/1.0")
                .header("Authorization", "Bearer SECRET").header("x-session-id", "s1").body(Body::from(body.to_owned())).unwrap())
        };
        let r = send("/v1/chat/completions", r#"{"messages":[{"role":"user","content":"hi"}]}"#).await.unwrap();
        axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
        let r = send("/v1/chat/completions", r#"{"fail":true}"#).await.unwrap();
        axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
        let r = app.clone().oneshot(axum::http::Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
        axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
        let logs = tape.0.lock().unwrap().len();
        assert_eq!(logs, 1, "auth failures and non-inference routes store no payload");
        assert_eq!(tape.request(0)["messages"][0]["content"], "hi");
        assert!(tape.response_text(0).contains("\"he\""));
        let record = tape.0.lock().unwrap()[0].clone();
        assert!(matches!(record.response, ResponsePayload::Frames { sse: true, truncated: false, .. }));
        assert_eq!(record.meta.session_id.as_deref(), Some("s1"));
        assert_eq!(record.meta.client_kind, "claude_code");
        assert_eq!(record.meta.status, 200);
        assert_eq!(sink.0.lock().unwrap().len(), 3);
        // Off means no tee at all.
        tape.1.store(true, Relaxed);
        let r = send("/v1/chat/completions", r#"{"messages":[]}"#).await.unwrap();
        axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
        assert_eq!(tape.0.lock().unwrap().len(), 1);
    }
    /// Benchmark requests skip the log unless the store records them.
    #[tokio::test]
    async fn bench_requests_skip_the_full_log_by_default() {
        use tower::ServiceExt;
        let (sink, tape) = logging_sink();
        let app = axum::Router::new().route("/v1/chat/completions", axum::routing::post(|_: Bytes| async { "{}" }))
            .layer(axum::middleware::from_fn_with_state(Middleware::new(sink.clone()), track));
        let r = app.oneshot(axum::http::Request::post("/v1/chat/completions").header("x-cuteafd-bench", "t")
            .body(Body::from("{}")).unwrap()).await.unwrap();
        axum::body::to_bytes(r.into_body(), 1024).await.unwrap();
        assert_eq!(tape.0.lock().unwrap().len(), 1, "an unverified bench header is ordinary traffic");
        assert!(!sink.0.lock().unwrap()[0].bench);
        let (sink, tape) = logging_sink();
        let verified = Middleware::new(sink.clone()).with_bench(Arc::new(|t: &str| t == "run-token"));
        let app = axum::Router::new().route("/v1/chat/completions", axum::routing::post(|_: Bytes| async { "{}" }))
            .layer(axum::middleware::from_fn_with_state(verified, track));
        for token in ["run-token", "guess"] {
            let r = app.clone().oneshot(axum::http::Request::post("/v1/chat/completions").header("x-cuteafd-bench", token)
                .body(Body::from("{}")).unwrap()).await.unwrap();
            axum::body::to_bytes(r.into_body(), 1024).await.unwrap();
        }
        let rows = sink.0.lock().unwrap();
        assert_eq!((rows[0].bench, rows[1].bench), (true, false));
        assert_eq!(tape.0.lock().unwrap().len(), 1, "only the verified bench request skips the log");
    }
    #[test]
    fn last_reference_and_cancelled() {
        let sink = Arc::new(Sink::default());
        let scope = UsageScope::new(Record::default(), sink.clone());
        let other = scope.clone();
        drop(scope);
        assert!(sink.0.lock().unwrap().is_empty());
        drop(other);
        assert_eq!(sink.0.lock().unwrap()[0].outcome, "cancelled");
    }
    #[tokio::test]
    async fn stats_preserves_key_order_and_chat_response() {
        use tower::ServiceExt;
        let sink = Arc::new(Sink::default());
        let bare = axum::Router::new()
            .route(
                "/v1/stats",
                axum::routing::get(|| async { "{\"z\":1,\"a\":2}" }),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(|| async { "chat bytes" }),
            );
        let app = bare.clone().layer(axum::middleware::from_fn_with_state(
            Middleware::new(sink),
            track,
        ));
        let make = || {
            axum::http::Request::post("/v1/chat/completions")
                .body(Body::empty())
                .unwrap()
        };
        let baseline = bare.oneshot(make()).await.unwrap();
        let result = app.clone().oneshot(make()).await.unwrap();
        assert_eq!(baseline.headers(), result.headers());
        assert_eq!(
            axum::body::to_bytes(baseline.into_body(), 4096)
                .await
                .unwrap(),
            axum::body::to_bytes(result.into_body(), 4096)
                .await
                .unwrap()
        );
        let response = app
            .oneshot(
                axum::http::Request::get("/v1/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert!(std::str::from_utf8(&bytes)
            .unwrap()
            .starts_with("{\"z\":1,\"a\":2,\"usage\":"));
    }
    #[test]
    fn sessions_preserve_confidence_and_hash_payload_free_keys() {
        for (record, choices, expected, source) in [
            (Record {session_id: Some("header".into()), session_source: Some("explicit".into()), ..Default::default()}, vec![("root", "explicit"),("prefix", "prefix")], "header".to_string(), "explicit"),
            (Record::default(), vec![("root", "explicit"),("prefix", "prefix")], "root".to_string(), "explicit"),
            (Record::default(), vec![("prefix", "prefix")], "prefix".to_string(), "prefix"),
        ] {
            let sink = Arc::new(Sink::default());
            let scope = UsageHandle::new(record, sink.clone());
            for (id, source) in choices { scope.session(id.into(), source); }
            scope.details(Details::default());
            scope.finished(200);
            drop(scope);
            let rows = sink.0.lock().unwrap();
            assert_eq!(rows[0].session_id.as_deref(), Some(expected.as_str()));
            assert_eq!(rows[0].session_source.as_deref(), Some(source));
        }
        assert_eq!(session_hash("private-user").len(), 16);
        assert_ne!(session_hash("private-user"), "private-user");
    }
    #[test]
    fn engine_retirement_rounds_and_child_lifetimes() {
        let sink = Arc::new(Sink::default());
        let scope = UsageHandle::new(Record {rid:"root-rid".into(), ..Default::default()}, sink.clone());
        let child = scope.child("realtime");
        child.admitted(2);
        child.prompt_tokens(12, 4);
        child.first_token();
        child.round(4, 3);
        child.retired(5, "cancelled");
        child.finished(200);
        drop(child);
        let rows = sink.0.lock().unwrap();
        assert_eq!(rows.len(), 1);
        assert_ne!(rows[0].rid, scope.rid());
        assert_eq!(rows[0].outcome, "cancelled");
        assert_eq!(rows[0].tokens_out, Some(5));
        assert_eq!(rows[0].tokens_cached, Some(4));
        assert_eq!(rows[0].concurrency_engine, Some(2));
        assert_eq!(rows[0].rounds, Some(1));
        assert_eq!(rows[0].draft_accepted, Some(3));
        drop(rows);
        scope.engine_error("worker");
        drop(scope);
        assert_eq!(sink.0.lock().unwrap()[1].outcome, "engine_error");
    }
    #[tokio::test]
    async fn disconnected_http_stream_emits_cancelled_row() {
        use tower::ServiceExt;
        let sink = Arc::new(Sink::default());
        let app = axum::Router::new().route("/v1/chat/completions", axum::routing::post(|| async {
            Body::from_stream(futures::stream::pending::<Result<Bytes, std::convert::Infallible>>())
        })).layer(axum::middleware::from_fn_with_state(Middleware::new(sink.clone()), track));
        let response = app.oneshot(axum::http::Request::post("/v1/chat/completions").body(Body::empty()).unwrap()).await.unwrap();
        drop(response);
        assert_eq!(sink.0.lock().unwrap()[0].outcome, "cancelled");
    }
    #[tokio::test]
    async fn optional_extractor_keeps_usage_off_valid() {
        use tower::ServiceExt;
        let app = axum::Router::new().route("/", axum::routing::get(|scope: Option<UsageHandle>| async move {
            assert!(scope.is_none());
            "ok"
        }));
        let response = app.oneshot(axum::http::Request::get("/").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), 200);
    }
    #[test]
    fn dashboard_polling_is_not_recorded() {
        for p in ["/usage", "/console/usage/summary", "/assets/cuteafd-ui.js", "/v1/console/events", "/v1/bench/status", "/health"] {
            assert!(self_traffic(p), "{p}");
        }
        for p in ["/v1/chat/completions", "/v1/messages", "/v1/stats", "/v1/models", "/console/unlock"] {
            assert!(!self_traffic(p), "{p}");
        }
    }
    #[test]
    fn labels_and_classifier() {
        assert_eq!(key_label("secret").len(), 10);
        let mut h = HeaderMap::new();
        h.insert("user-agent", "codex_cli_rs/1".parse().unwrap());
        assert_eq!(client_kind(&h), "codex");
    }
}
