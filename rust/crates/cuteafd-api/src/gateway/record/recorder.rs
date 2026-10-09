use std::{path::PathBuf, sync::{Arc, Mutex, OnceLock}};
use axum::{body::{Body, to_bytes}, extract::{Request, State}, http::HeaderMap, middleware::Next, response::Response};
use futures::StreamExt;
use regex::Regex;
use serde_json::{json, Value};
use super::{Tape, TapeSink};

const MAX_BODY: usize = 16 * 1024 * 1024;
fn key_regex() -> &'static Regex { static R: OnceLock<Regex> = OnceLock::new(); R.get_or_init(|| Regex::new(r#"sk-[A-Za-z0-9_-]{20,}|(?i:Bearer\s+)[^\s\"\\,;]+|openai-insecure-api-key\.[^\s\"\\,;]+"#).unwrap()) }
fn path_regex() -> &'static Regex { static R: OnceLock<Regex> = OnceLock::new(); R.get_or_init(|| Regex::new(r#"/home/[^/\s\"\\]+"#).unwrap()) }
fn email_regex() -> &'static Regex { static R: OnceLock<Regex> = OnceLock::new(); R.get_or_init(|| Regex::new(r#"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"#).unwrap()) }
pub(super) fn secret_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase().replace('-',"_");
    matches!(name.as_str(),"authorization"|"cookie"|"set_cookie"|"sec_websocket_protocol") || name.contains("api_key") || name == "key" || name.ends_with("_key")
}
pub(super) fn query_secret_name(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase().replace('-',"_");
    secret_name(name) || matches!(normalized.as_str(),"apikey"|"token"|"access_token"|"secret"|"password"|"sig"|"signature")
}
#[derive(Clone, Default)]
pub struct Sanitizer { secrets: Vec<String> }
impl Sanitizer {
    pub fn new(secrets: impl IntoIterator<Item = String>) -> Self {
        let mut secrets: Vec<_> = secrets.into_iter().filter(|s| !s.is_empty()).collect();
        secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
        Self { secrets }
    }
    pub fn from_env(extra: impl IntoIterator<Item = String>) -> Self {
        let mut secrets:Vec<String> = extra.into_iter().chain(["DEEPSEEK_API_KEY","EXA_API_KEY","OPENROUTER_API_KEY","LITELLM_API_KEY","LITELLM_BASE_URL"].into_iter().filter_map(|name| std::env::var(name).ok())).collect();
        if let Ok(url) = std::env::var("LITELLM_BASE_URL") {
            if let Ok(url) = reqwest::Url::parse(&url) { if let Some(host) = url.host_str() { secrets.push(host.into()); } }
        }
        Self::new(secrets)
    }
    pub fn text(&self, text: &str) -> String {
        let mut text = text.to_string();
        for secret in &self.secrets { text = text.replace(secret,"[REDACTED]"); }
        text = key_regex().replace_all(&text,"[REDACTED]").into_owned();
        text = path_regex().replace_all(&text,"~").into_owned();
        text = email_regex().replace_all(&text,"[REDACTED_EMAIL]").into_owned();
        // SSE embeds JSON on data lines; apply key-name redaction there as well.
        if text.lines().any(|line| line.starts_with("data:")) {
            let mut lines = Vec::new();
            for line in text.split_inclusive('\n') {
                if let Some(data) = line.strip_prefix("data:") {
                    if let Ok(mut value) = serde_json::from_str::<Value>(data.trim()) {
                        let original = value.clone();
                        self.value(&mut value);
                        if value != original {
                            lines.push(format!("data: {}{}",value,if line.ends_with('\n') { "\n" } else { "" }));
                            continue;
                        }
                    }
                }
                lines.push(line.to_string());
            }
            return lines.concat();
        }
        text
    }
    pub fn value(&self, value: &mut Value) {
        match value {
            Value::String(text) => {
                // Bodies and WebSocket frames often contain JSON encoded inside strings.
                if let Ok(mut inner) = serde_json::from_str::<Value>(text) {
                    if inner.is_object() || inner.is_array() {
                        let original = inner.clone();
                        self.value(&mut inner);
                        if inner != original { *text = inner.to_string(); }
                        return;
                    }
                }
                *text = self.text(text);
            }
            Value::Array(values) => for value in values { self.value(value); },
            Value::Object(object) => {
                let original = std::mem::take(object);
                for (name,mut value) in original {
                    if secret_name(&name) { value = json!("[REDACTED]"); } else { self.value(&mut value); }
                    let clean = self.text(&name);
                    let mut key = clean.clone();
                    let mut suffix = 1;
                    while object.contains_key(&key) {
                        key = format!("{clean}_{suffix}");
                        suffix += 1;
                    }
                    object.insert(key,value);
                }
            },
            _ => {}
        }
    }
    /// Decode before redacting, then re-encode so percent-escaped credentials cannot leak.
    pub fn query(&self, query: &str) -> String {
        let mut out = url::form_urlencoded::Serializer::new(String::new());
        for (name,value) in url::form_urlencoded::parse(query.as_bytes()) {
            let sensitive = query_secret_name(&name);
            let value = if sensitive { "[REDACTED]".into() } else { self.text(&value) };
            out.append_pair(&self.text(&name),&value);
        }
        out.finish()
    }
    pub fn headers(&self, headers: &HeaderMap) -> Value {
        let mut out = serde_json::Map::new();
        for (name,value) in headers {
            let name = name.as_str();
            if secret_name(name) { out.insert(name.into(),json!("[REDACTED]")); }
            else if matches!(name,"content-type"|"accept"|"anthropic-version"|"anthropic-beta"|"openai-beta"|"user-agent"|"x-request-id"|"request-id") {
                out.insert(name.into(),json!(self.text(value.to_str().unwrap_or("[NON_TEXT]"))));
            }
        }
        Value::Object(out)
    }
}
#[derive(Clone)]
pub struct Recorder { directory: PathBuf, sanitizer: Sanitizer }
impl Recorder {
    pub fn new(directory: PathBuf, sanitizer: Sanitizer) -> std::io::Result<Self> {
        std::fs::create_dir_all(&directory)?;
        Ok(Self { directory, sanitizer })
    }
}
struct Sink { path: PathBuf, sanitizer: Sanitizer, fixture: Mutex<Value> }
impl Sink {
    fn update(&self, update: impl FnOnce(&mut Value)) { update(&mut self.fixture.lock().unwrap_or_else(|p| p.into_inner())); }
}
impl TapeSink for Sink {
    fn record(&self, kind: &str, mut entry: Value) {
        self.sanitizer.value(&mut entry);
        self.update(|fixture| { fixture["entries"].as_array_mut().unwrap().push(json!({"kind":kind,"entry":entry})); });
    }
}
impl Drop for Sink {
    fn drop(&mut self) {
        let fixture = self.fixture.get_mut().unwrap_or_else(|p| p.into_inner());
        self.sanitizer.value(fixture);
        let result = serde_json::to_vec_pretty(fixture).map_err(std::io::Error::other)
            .and_then(|bytes| std::fs::write(&self.path,bytes));
        if result.is_err() { tracing::warn!("could not persist gateway fixture"); }
    }
}
struct BodyCapture { sink: Arc<Sink>, body: Vec<u8>, truncated: bool, complete: bool, websocket: bool }
impl Drop for BodyCapture {
    fn drop(&mut self) {
        self.sink.update(|fixture| {
            fixture["response_body"] = json!(String::from_utf8_lossy(&self.body));
            fixture["truncated"] = json!(self.truncated);
            fixture["complete"] = json!(self.complete && !self.truncated);
            fixture["websocket"] = json!(self.websocket);
        });
    }
}
pub async fn middleware(State(recorder): State<Recorder>, request: Request, next: Next) -> Response {
    let (mut parts,body) = request.into_parts();
    let body = match to_bytes(body,MAX_BODY).await {
        Ok(body) => body,
        Err(_) => return crate::gateway::GatewayError::new(crate::gateway::ErrorKind::RequestTooLarge,"recorded request exceeds body limit").openai_response(),
    };
    let path = recorder.directory.join(format!("{}.json",uuid::Uuid::new_v4()));
    let mut fixture = json!({"version":1,"route":parts.uri.path(),"query":parts.uri.query().map(|query| recorder.sanitizer.query(query)),"method":parts.method.as_str(),
        "request_headers":recorder.sanitizer.headers(&parts.headers),"request_body":String::from_utf8_lossy(&body),
        "entries":[],"response_status":null,"response_headers":{},"response_body":"","complete":false});
    recorder.sanitizer.value(&mut fixture);
    let sink = Arc::new(Sink { path,sanitizer:recorder.sanitizer.clone(),fixture:Mutex::new(fixture) });
    parts.extensions.insert(Tape(Some(sink.clone())));
    let response = next.run(Request::from_parts(parts,Body::from(body))).await;
    let (parts,body) = response.into_parts();
    sink.update(|fixture| { fixture["response_status"] = json!(parts.status.as_u16()); fixture["response_headers"] = sink.sanitizer.headers(&parts.headers); });
    // The upgraded connection owns its Tape; finalization happens when its last clone drops.
    let websocket = parts.status == axum::http::StatusCode::SWITCHING_PROTOCOLS;
    let capture = BodyCapture { sink,body:Vec::new(),truncated:false,complete:false,websocket };
    let stream = async_stream::stream! {
        let mut capture = capture;
        let mut stream = body.into_data_stream();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let available = MAX_BODY.saturating_sub(capture.body.len());
                    capture.body.extend_from_slice(&bytes[..bytes.len().min(available)]);
                    capture.truncated |= bytes.len() > available;
                    // Sanitize at finalization, not per chunk: keys can span chunks.
                    yield Ok::<_,axum::Error>(bytes);
                }
                Err(error) => { yield Err(error); return; }
            }
        }
        capture.complete = true;
    };
    Response::from_parts(parts,Body::from_stream(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sanitizer_scrubs_http_and_websocket_secrets_and_private_paths() {
        let sanitizer = Sanitizer::new(["literal-secret".into()]);
        let mut value = json!({"authorization":"Bearer test-secret", "nested":{"x-api-key":"anything"},
            "frame":"{\"type\":\"session.update\",\"api_key\":\"other-secret\",\"text\":\"literal-secret /home/person/file person@example.org\"}",
            "body":"Bearer token123 sk-abcdefghijklmnopqrstuvwxyz openai-insecure-api-key.another"});
        sanitizer.value(&mut value);
        let text = value.to_string();
        for bad in ["literal-secret","anything","other-secret","/home/","person@example","token123","sk-abc","key.another"] { assert!(!text.contains(bad),"failed scrub {bad}"); }
    }
    #[test]
    fn sanitizer_scrubs_router_credentials_and_private_endpoint_literals() {
        let key = format!("sk-or-v1-{}","a".repeat(64));
        let sanitizer = Sanitizer::new(["https://private.invalid/v1".into(),"private.invalid".into()]);
        let text = sanitizer.text(&format!("{key} https://private.invalid/v1 private.invalid"));
        assert!(!text.contains("sk-or-v1-") && !text.contains("private.invalid"));
    }
    #[test]
    fn sanitizer_redacts_object_keys_without_losing_collisions() {
        let sanitizer = Sanitizer::new(["secret-a".into(),"secret-b".into()]);
        let mut value = json!({"secret-a":1,"secret-b":2,"[REDACTED]":3,"[REDACTED]_1":4,
            "nested":"{\"secret-a\":\"kept\",\"secret-b\":\"also kept\"}"});
        sanitizer.value(&mut value);
        assert!(!value.to_string().contains("secret-a") && !value.to_string().contains("secret-b"));
        let object = value.as_object().unwrap();
        assert_eq!(object.len(),5);
        for expected in [1,2,3,4] { assert!(object.values().any(|v| v == &json!(expected))); }
        let nested:Value = serde_json::from_str(value["nested"].as_str().unwrap()).unwrap();
        assert_eq!(nested.as_object().unwrap().len(),2);
    }
    #[test]
    fn query_redacts_named_tokens_and_encoded_registered_values() {
        let sanitizer = Sanitizer::new(["literal/value".into()]);
        let query = sanitizer.query("key=a&api_key=b&apikey=c&token=d&access_token=e&secret=f&password=g&sig=h&signature=i&API-KEY=j&safe=literal%2Fvalue&literal%2Fvalue=hidden&model=test&model=second");
        let pairs:Vec<_> = url::form_urlencoded::parse(query.as_bytes()).collect();
        assert!(pairs[..11].iter().all(|(_,value)| value == "[REDACTED]"));
        assert!(!query.contains("literal") && pairs.iter().any(|(name,_)| name == "[REDACTED]"));
        assert!(pairs.iter().any(|(name,value)| name == "model" && value == "test"));
        assert!(pairs.iter().any(|(name,value)| name == "model" && value == "second"));
    }
    #[test]
    fn sanitizer_preserves_clean_sse_bytes_and_scrubs_key_fields() {
        let sanitizer = Sanitizer::default();
        let clean = "event: test\r\ndata: { \"text\": \"hi\" }\r\n\r\n";
        assert_eq!(sanitizer.text(clean),clean);
        assert!(!sanitizer.text("data: {\"api_key\":\"unknown\"}\n\n").contains("unknown"));
    }
    #[tokio::test]
    async fn websocket_fixture_lifetime_headers_and_frame_order() {
        use axum::extract::ws::{Message,WebSocketUpgrade};
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;
        use futures::SinkExt;
        let directory = tempfile::tempdir().unwrap();
        let recorder = Recorder::new(directory.path().into(),Sanitizer::new(["literal-key".into()])).unwrap();
        let app = axum::Router::new().route("/ws",axum::routing::get(|upgrade:WebSocketUpgrade,tape:Tape| async {
            upgrade.protocols(["realtime"]).on_upgrade(move |mut socket| async move {
                while let Some(Ok(Message::Text(text))) = socket.next().await {
                    tape.frame("client",&text);
                    tape.frame("server",&text);
                    if socket.send(Message::Text(text)).await.is_err() { break; }
                }
            })
        })).layer(axum::middleware::from_fn_with_state(recorder,middleware));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move { axum::serve(listener,app).await.unwrap(); });
        let mut request = format!("ws://{address}/ws?access_token=handshake-only&safe=literal%2Dkey&model=test").into_client_request().unwrap();
        request.headers_mut().insert("sec-websocket-protocol","realtime, openai-insecure-api-key.literal-key".parse().unwrap());
        request.headers_mut().insert("authorization","Bearer literal-key".parse().unwrap());
        let (mut socket,_) = tokio_tungstenite::connect_async(request).await.unwrap();
        socket.send(tokio_tungstenite::tungstenite::Message::Text("{\"api_key\":\"literal-key\",\"text\":\"/home/person/path\"}".into())).await.unwrap();
        socket.next().await.unwrap().unwrap();
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(),0,"live websocket must keep tape open");
        socket.close(None).await.unwrap();
        drop(socket);
        tokio::time::timeout(std::time::Duration::from_secs(3),async {
            while std::fs::read_dir(directory.path()).unwrap().count() == 0 { tokio::time::sleep(std::time::Duration::from_millis(10)).await; }
        }).await.unwrap();
        let path = std::fs::read_dir(directory.path()).unwrap().next().unwrap().unwrap().path();
        let text = std::fs::read_to_string(path).unwrap();
        assert!(!text.contains("literal-key") && !text.contains("/home/") && !text.contains("handshake-only"));
        let fixture:Value = serde_json::from_str(&text).unwrap();
        let query:Vec<_> = url::form_urlencoded::parse(fixture["query"].as_str().unwrap().as_bytes()).collect();
        assert!(query.iter().any(|(name,value)| name == "access_token" && value == "[REDACTED]"));
        assert!(query.iter().any(|(name,value)| name == "model" && value == "test"));
        assert_eq!(fixture["entries"][0]["entry"]["direction"],"client");
        assert_eq!(fixture["entries"][1]["entry"]["direction"],"server");
        assert_eq!(fixture["request_headers"]["sec-websocket-protocol"],"[REDACTED]");
        task.abort();
    }
    #[tokio::test]
    async fn recorder_tees_body_and_waits_for_tape_drop() {
        use tower::ServiceExt;
        let directory = tempfile::tempdir().unwrap();
        let recorder = Recorder::new(directory.path().into(),Sanitizer::new(["secret".into()])).unwrap();
        let app = axum::Router::new().route("/test",axum::routing::post(|tape:Tape| async move {
            tape.frame("client","{\"api_key\":\"secret\"}");
            tape.record("upstream",|| json!({"request":{},"status":200,"body":"data: hello\n\n"}));
            "data: secret\n\ndata: /home/person/file\n\n"
        })).layer(axum::middleware::from_fn_with_state(recorder,middleware));
        let response = app.oneshot(Request::builder().uri("/test?password=http-only&safe=%73ecret&model=test").method("POST").header("authorization","Bearer secret").body(Body::from("hello")).unwrap()).await.unwrap();
        let body = to_bytes(response.into_body(),1024).await.unwrap();
        assert!(String::from_utf8_lossy(&body).contains("secret"));
        let path = std::fs::read_dir(directory.path()).unwrap().next().unwrap().unwrap().path();
        let fixture:Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(!fixture.to_string().contains("http-only"));
        let query:Vec<_> = url::form_urlencoded::parse(fixture["query"].as_str().unwrap().as_bytes()).collect();
        assert!(query.iter().any(|(name,value)| name == "password" && value == "[REDACTED]"));
        assert!(query.iter().any(|(name,value)| name == "model" && value == "test"));
        assert_eq!(fixture["complete"],true);
        assert_eq!(fixture["entries"].as_array().unwrap().len(),2);
        assert!(!fixture.to_string().contains("/home/"));
        assert!(!fixture.to_string().contains("secret"));
    }
}
