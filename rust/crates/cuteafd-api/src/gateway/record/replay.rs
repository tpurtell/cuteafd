//! Route-independent offline fixture harness (JSON/SSE), plus Backend-level replay.
use std::{collections::VecDeque, sync::{Arc, Mutex}};
use axum::{body::Body, http::StatusCode, response::Response, routing::any, Router};
use futures::{future::BoxFuture, StreamExt};
use serde_json::{json, Value};
use crate::gateway::{search::{SearchHit, SearchProvider, SearchQuery}, turn::{TurnEvent, TurnRequest}, upstream::{Flavor, Upstream, UpstreamConfig}, Backend, GatewayError};

pub struct ReplayUpstream { pub url: String, task: tokio::task::JoinHandle<()>, remaining: Arc<Mutex<VecDeque<Value>>> }
impl Drop for ReplayUpstream { fn drop(&mut self) { self.task.abort(); } }
impl ReplayUpstream {
    /// Every recorded upstream exchange is served in order, independent of its path.
    pub async fn start(entries: &[Value]) -> std::io::Result<Self> {
        let remaining = Arc::new(Mutex::new(entries.iter().filter(|e| e["kind"] == "upstream").map(|e| e["entry"].clone()).collect::<VecDeque<_>>()));
        let queue = remaining.clone();
        let app = Router::new().fallback(any(move |body: axum::body::Bytes| {
            let queue = queue.clone();
            async move {
                let Some(entry) = queue.lock().unwrap().pop_front() else { return Response::builder().status(500).body(Body::from("fixture exhausted")).unwrap(); };
                if entry.get("request").is_some() {
                    let received: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                    if normalize(received) != normalize(entry["request"].clone()) {
                        return Response::builder().status(400).body(Body::from("fixture request mismatch")).unwrap();
                    }
                }
                let body = entry["body"].as_str().map(str::to_string).unwrap_or_else(|| entry["body"].to_string());
                let content_type = if body.trim_start().starts_with("data:") || body.trim_start().starts_with("event:") || body.trim_start().starts_with(':') { "text/event-stream" } else { "application/json" };
                Response::builder().status(entry["status"].as_u64().unwrap_or(200) as u16)
                    .header("content-type",content_type).body(Body::from(body)).unwrap()
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let url = format!("http://{}",listener.local_addr()?);
        let task = tokio::spawn(async move { let _ = axum::serve(listener,app).await; });
        Ok(Self { url,task,remaining })
    }
    pub fn assert_consumed(&self) -> Result<(),GatewayError> {
        if self.remaining.lock().unwrap().is_empty() { Ok(()) } else { Err(GatewayError::internal("unconsumed upstream fixture entries")) }
    }
}
#[derive(Clone)]
pub struct ReplaySearch { entries: Arc<Mutex<VecDeque<Value>>> }
impl ReplaySearch {
    pub fn new(entries: &[Value]) -> Self { Self { entries: Arc::new(Mutex::new(entries.iter().filter(|e| e["kind"] == "search").map(|e| e["entry"].clone()).collect())) } }
}
impl SearchProvider for ReplaySearch {
    fn name(&self) -> &str { "replay" }
    fn search(&self, query: SearchQuery) -> BoxFuture<'static, Result<Vec<SearchHit>,GatewayError>> {
        let result = (|| {
            let entry = self.entries.lock().unwrap().pop_front().ok_or_else(|| GatewayError::internal("search fixture exhausted"))?;
            if entry["query"] != serde_json::to_value(query).unwrap() { return Err(GatewayError::internal("search fixture query mismatch")); }
            serde_json::from_value(entry["hits"].clone()).map_err(|_| GatewayError::upstream("recorded search failed"))
        })();
        Box::pin(async { result })
    }
}
/// Ignore generated identity/time fields, but preserve all semantic event content.
pub fn normalize(mut value: Value) -> Value {
    fn visit(value: &mut Value) {
        match value {
            Value::Object(object) => {
                for (key,value) in object.iter_mut() {
                    if matches!(key.as_str(),"id"|"call_id"|"tool_call_id"|"tool_use_id"|"item_id"|"response_id"|"session_id"|"signature") && value.is_string() { *value = json!("<generated>"); }
                    else if matches!(key.as_str(),"created"|"created_at"|"timestamp"|"sequence_number") { *value = json!(0); }
                    else { visit(value); }
                }
            }
            Value::Array(values) => for value in values { visit(value); },
            _ => {}
        }
    }
    visit(&mut value); value
}
pub fn normalize_body(body: &str) -> Value {
    if let Ok(value) = serde_json::from_str(body) { return normalize(value); }
    let mut events = Vec::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            events.push(serde_json::from_str(data).map(normalize).unwrap_or_else(|_| json!(data)));
        }
    }
    if events.is_empty() { json!(body) } else { json!(events) }
}
pub async fn replay_backend(fixture: &Value) -> Result<Vec<TurnEvent>,GatewayError> {
    let entries = fixture["entries"].as_array().ok_or_else(|| GatewayError::invalid("fixture entries missing"))?;
    let server = ReplayUpstream::start(entries).await.map_err(|_| GatewayError::internal("could not bind replay upstream"))?;
    let flavor: Flavor = fixture["flavor"].as_str().unwrap_or("openai-chat").parse()?;
    let turn: TurnRequest = serde_json::from_value(fixture["turn"].clone()).map_err(|_| GatewayError::invalid("fixture turn missing"))?;
    let mut config = UpstreamConfig::new(server.url.clone(),flavor,turn.model.clone());
    config.thinking_toggle = fixture["deepseek_thinking"].as_bool().unwrap_or(false);
    config.capabilities.strict_tools = fixture["strict_tools"].as_bool().unwrap_or(false);
    config.capabilities.json_schema = fixture["json_schema"].as_bool().unwrap_or(false);
    let backend = Upstream::new(config)?;
    let events: Result<Vec<_>,_> = backend.start(turn).await?.collect::<Vec<_>>().await.into_iter().collect();
    server.assert_consumed()?;
    events
}
/// Call after the frontend router has been wired to ReplayUpstream/ReplaySearch.
pub async fn replay_client(app: Router, fixture: &Value) -> Result<(),GatewayError> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.map_err(|_| GatewayError::internal("could not bind replay gateway"))?;
    let address = listener.local_addr().map_err(|_| GatewayError::internal("missing replay address"))?;
    let task = tokio::spawn(async move { let _ = axum::serve(listener,app).await; });
    struct Guard(tokio::task::JoinHandle<()>);
    impl Drop for Guard { fn drop(&mut self) { self.0.abort(); } }
    let _guard = Guard(task);
    let method = fixture["method"].as_str().unwrap_or("POST").parse().map_err(|_| GatewayError::invalid("invalid fixture method"))?;
    let mut request = reqwest::Client::new().request(method,format!("http://{address}{}",fixture["route"].as_str().unwrap_or("/")))
        .header("content-type","application/json").body(fixture["request_body"].as_str().unwrap_or_default().to_string());
    if let Some(headers) = fixture["request_headers"].as_object() {
        for (name,value) in headers { if let Some(value) = value.as_str().filter(|v| !v.contains("REDACTED")) { request = request.header(name,value); } }
    }
    let response = request.send().await.map_err(|_| GatewayError::internal("replay client request failed"))?;
    let status = response.status();
    let body = response.text().await.map_err(|_| GatewayError::internal("replay client body failed"))?;
    if status != StatusCode::from_u16(fixture["response_status"].as_u64().unwrap_or(200) as u16).unwrap_or(StatusCode::OK)
        || normalize_body(&body) != normalize_body(fixture["response_body"].as_str().unwrap_or_default()) {
        return Err(GatewayError::internal("replayed client response differs from fixture"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn replay_json_and_sse_client_responses() {
        for body in ["{\"id\":\"generated\",\"output\":\"hello\"}","event: response.output_text.delta\ndata: {\"id\":\"generated\",\"text\":\"hello\",\"sequence_number\":4}\n\n"] {
            let owned = body.to_string();
            let app = Router::new().route("/any",axum::routing::post(move || { let body = owned.clone(); async { body } }));
            replay_client(app,&json!({"method":"POST","route":"/any","request_body":"{}","response_status":200,"response_body":body})).await.unwrap();
        }
    }
    fn fixture_paths(directory: &std::path::Path, paths: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() { fixture_paths(&path,paths); } else if path.extension().is_some_and(|s| s == "json") { paths.push(path); }
        }
    }
    #[test]
    fn every_gateway_fixture_is_free_of_secrets_paths_and_emails() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gateway");
        let mut paths = Vec::new(); fixture_paths(&directory,&mut paths);
        assert!(!paths.is_empty(),"fixture suite must not be empty");
        let key = regex::Regex::new(r#"sk-or-v1-[0-9a-f]{20,}|sk-[A-Za-z0-9_-]{20,}"#).unwrap();
        let bearer = regex::Regex::new(r#"(?i:Bearer\s+)([^\s\"\\,;]+)"#).unwrap();
        let email = regex::Regex::new(r#"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"#).unwrap();
        fn fields(value:&Value) {
            match value {
                Value::Object(object) => for (name,value) in object {
                    let name = name.to_ascii_lowercase().replace('-',"_");
                    if name.contains("api_key") || name == "key" || name.ends_with("_key") {
                        assert!(value.is_null() || value.as_str().is_some_and(|s| s.contains("REDACTED")),"unredacted key-like field");
                    }
                    fields(value);
                },
                Value::Array(values) => for value in values { fields(value); },
                Value::String(text) => if let Ok(inner) = serde_json::from_str::<Value>(text) { if inner.is_object() || inner.is_array() { fields(&inner); } },
                _ => {}
            }
        }
        for path in paths {
            let text = std::fs::read_to_string(&path).unwrap();
            assert!(!key.is_match(&text),"key in {}",path.display());
            assert!(!text.contains("/home/"),"home path in {}",path.display());
            assert!(!email.is_match(&text),"email in {}",path.display());
            for capture in bearer.captures_iter(&text) { assert!(capture[1].contains("REDACTED"),"Bearer token in {}",path.display()); }
            for name in ["DEEPSEEK_API_KEY","EXA_API_KEY","OPENROUTER_API_KEY","LITELLM_API_KEY","LITELLM_BASE_URL"] {
                if let Ok(secret) = std::env::var(name) { assert!(secret.is_empty() || !text.contains(&secret),"literal credential in {}",path.display()); }
            }
            if let Ok(url) = std::env::var("LITELLM_BASE_URL") {
                if let Ok(url) = reqwest::Url::parse(&url) {
                    if let Some(host) = url.host_str() { assert!(!text.contains(host),"private host in {}",path.display()); }
                }
            }
            fields(&serde_json::from_str::<Value>(&text).unwrap());
        }
    }
    #[tokio::test]
    async fn replay_every_backend_fixture() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gateway/upstream");
        let mut paths = Vec::new();fixture_paths(&directory,&mut paths);
        for path in paths {
            let fixture:Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let events = replay_backend(&fixture).await;
            if let Some(status) = fixture["expected_error_status"].as_u64() {
                assert_eq!(events.unwrap_err().status(),status as u16,"{}",path.display());
            } else {
                assert_eq!(normalize(serde_json::to_value(events.unwrap()).unwrap()),normalize(fixture["expected_events"].clone()),"{}",path.display());
            }
        }
    }
}
