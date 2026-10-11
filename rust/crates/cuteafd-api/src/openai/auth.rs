//! Optional bearer authentication, adapted from Hugh Madden's glm53f-afd auth.rs.
/*
MIT License

Copyright (c) 2026 Turquoise Bay AI Pty Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
use axum::{extract::Request, http::{header, HeaderMap, StatusCode}, middleware::Next,
    response::{IntoResponse, Response}, Json};
use serde_json::json;
use std::{path::Path, sync::Arc};

#[derive(Clone)]
pub struct ApiKey(Arc<KeySet>);
struct KeySet {
    default: Vec<u8>,
    keys: std::sync::Mutex<Vec<(String, Vec<u8>)>>,
    source: Option<std::path::PathBuf>,
    checked: std::sync::Mutex<Option<std::time::Instant>>,
    on_reload: std::sync::Mutex<Option<Arc<dyn Fn(Vec<String>) + Send + Sync>>>,
}

/// Verified credential identity, never the secret itself.
#[derive(Clone, Debug)]
pub struct KeyIdentity(pub String);
impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}
impl ApiKey {
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut key = Self::new(text.trim_end_matches(['\r', '\n']))
            .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidData, message))?;
        let named = std::env::var_os("CUTEAFD_API_KEYS_FILE").map(std::path::PathBuf::from).unwrap_or_else(|| path.with_file_name("api-keys"));
        if named.exists() { key = key.with_named_file(&named)?; }
        else { Arc::get_mut(&mut key.0).expect("fresh key").source = Some(named); }
        Ok(key)
    }
    pub fn new(key: &str) -> Result<Self, &'static str> {
        if key.is_empty() || key.bytes().any(|b| !b.is_ascii_graphic()) {
            return Err("API key must be nonempty and contain no whitespace or control bytes");
        }
        Ok(Self(Arc::new(KeySet { default: key.as_bytes().to_vec(), keys: std::sync::Mutex::new(vec![("default".into(), key.as_bytes().to_vec())]),
            source: None, checked: std::sync::Mutex::new(None), on_reload: std::sync::Mutex::new(None) })))
    }
    /// JSON object of stable names to keys; only the owner may read/write it.
    pub fn with_named_file(&self, path: &Path) -> std::io::Result<Self> {
        use std::os::unix::fs::PermissionsExt;
        if std::fs::metadata(path)?.permissions().mode() & 0o777 != 0o600 {
            return Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "named API key file must be mode 0600"));
        }
        let entries: std::collections::BTreeMap<String, String> = serde_json::from_str(&std::fs::read_to_string(path)?)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid named API key file"))?;
        let mut keys = vec![("default".into(), self.0.default.clone())];
        for (name, value) in entries {
            if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') || Self::new(&value).is_err() {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid named API key entry"));
            }
            if name == "default" && !constant_time_eq(&keys[0].1, value.as_bytes()) {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "named default key disagrees with legacy key"));
            }
            if name != "default" {
                if keys.iter().any(|(_, k)| constant_time_eq(k, value.as_bytes())) {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "duplicate API key value"));
                }
                keys.push((name, value.into_bytes()));
            }
        }
        Ok(Self(Arc::new(KeySet { default: self.0.default.clone(), keys: std::sync::Mutex::new(keys), source: Some(path.into()),
            checked: std::sync::Mutex::new(None), on_reload: std::sync::Mutex::new(None) })))
    }
    pub fn on_reload(&self, callback: Arc<dyn Fn(Vec<String>) + Send + Sync>) {
        *self.0.on_reload.lock().unwrap_or_else(|e| e.into_inner()) = Some(callback);
    }
    fn refresh(&self) {
        let Some(path) = &self.0.source else { return; };
        let mut checked = self.0.checked.lock().unwrap_or_else(|e| e.into_inner());
        if checked.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(1)) { return; }
        *checked = Some(std::time::Instant::now());
        // Atomic rename on the writer side; a malformed rotation fails closed for named keys.
        let keys = self.with_named_file(path).map(|key| key.0.keys.lock().unwrap_or_else(|e| e.into_inner()).clone())
            .unwrap_or_else(|_| vec![("default".into(), self.0.default.clone())]);
        let secrets = keys.iter().map(|(_, key)| String::from_utf8(key.clone()).expect("validated ASCII")).collect();
        if let Some(callback) = self.0.on_reload.lock().unwrap_or_else(|e| e.into_inner()).as_ref() { callback(secrets); }
        *self.0.keys.lock().unwrap_or_else(|e| e.into_inner()) = keys;
    }
    pub fn identity(&self, token: &str) -> Option<KeyIdentity> {
        // Legacy/default traffic stays on the memory-only fast path.
        if !constant_time_eq(&self.0.default, token.as_bytes()) { self.refresh(); }
        let mut found = None;
        for (name, key) in self.0.keys.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            if constant_time_eq(key, token.as_bytes()) { found = Some(KeyIdentity(name.clone())); }
        }
        found
    }
    pub fn secrets(&self) -> Vec<String> {
        self.0.keys.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(_, value)| String::from_utf8(value.clone()).expect("validated ASCII key")).collect()
    }
    pub fn authorization(&self) -> String {
        format!("Bearer {}", std::str::from_utf8(&self.0.default).expect("key originated as UTF-8"))
    }
    pub fn accepts(&self, headers: &HeaderMap) -> bool {
        let token = bearer(headers).unwrap_or("");
        self.identity(token).is_some()
    }
}
pub fn bearer(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim_start())
}
/// Work depends on the secret's length, not the differing byte or sent length.
pub fn constant_time_eq(secret: &[u8], sent: &[u8]) -> bool {
    let mut difference = secret.len() ^ sent.len();
    for (i, byte) in secret.iter().enumerate() {
        difference |= usize::from(byte ^ sent.get(i).copied().unwrap_or(0));
    }
    std::hint::black_box(difference) == 0
}
/// A private in-process benchmark token may authorize only its inference/readback.
#[derive(Clone, Default)]
pub struct Auth {
    pub key: Option<ApiKey>,
    pub internal: Option<Arc<dyn Fn(&str, &HeaderMap) -> bool + Send + Sync>>,
}
pub async fn require_key(axum::extract::State(auth): axum::extract::State<Auth>,
    mut request: Request, next: Next) -> Response {
    let path = request.uri().path();
    // Every credential form the official clients send (Bearer, Claude Code's
    // `x-api-key`, the Realtime websocket subprotocol) carries the same key.
    if (path == "/v1" || path.starts_with("/v1/")) && auth.key.as_ref().is_some_and(|key|
        !crate::gateway::auth::accepts(key, request.headers()) && !auth.internal.as_ref().is_some_and(|check| check(path, request.headers()))) {
        if path.starts_with("/v1/messages") {
            return crate::gateway::GatewayError::new(crate::gateway::ErrorKind::Authentication,
                "invalid x-api-key or Authorization bearer key").anthropic_response();
        }
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": {
            "message": "send a valid Authorization: Bearer API key", "type": "invalid_request_error",
            "code": "invalid_api_key"}}))).into_response();
    }
    if let Some(identity) = auth.key.as_ref().and_then(|key| crate::gateway::auth::presented(request.headers()).and_then(|token| key.identity(token))) {
        if let Some(scope) = request.extensions().get::<crate::usage::UsageHandle>() { scope.key_name(identity.0.clone()); }
        request.extensions_mut().insert(identity);
    }
    next.run(request).await
}
#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, routing::{get, post}, Router};
    use tower::ServiceExt;
    #[test]
    fn whole_key_and_redacted_debug() {
        let key = ApiKey::new("test-key").unwrap();
        assert_eq!(format!("{key:?}"), "ApiKey(<redacted>)");
        for sent in ["", "test-ke", "test-key-long", "test-kez", "test-key\0"] {
            assert!(!constant_time_eq(b"test-key", sent.as_bytes()));
        }
        assert!(constant_time_eq(b"test-key", b"test-key"));
        assert!(ApiKey::new("").is_err());
    }
    #[test]
    fn key_file_trims_only_line_endings_and_never_debugs_secret() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "file-secret\r\n").unwrap();
        let key = ApiKey::from_file(file.path()).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer file-secret".parse().unwrap());
        assert!(key.accepts(&headers));
        assert!(!format!("{key:?}").contains("file-secret"));
        for invalid in ["", "\n", "bad key", "key\nother"] {
            std::fs::write(file.path(), invalid).unwrap();
            assert!(ApiKey::from_file(file.path()).is_err());
        }
    }
    #[test]
    fn named_key_live_reload_without_restart() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let legacy = directory.path().join("api-key");
        let named = directory.path().join("api-keys");
        std::fs::write(&legacy, "legacy").unwrap();
        let key = ApiKey::from_file(&legacy).unwrap();
        std::fs::write(&named, r#"{"default":"legacy","agent":"new-agent"}"#).unwrap();
        std::fs::set_permissions(&named, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(key.identity("new-agent").unwrap().0, "agent");
    }
    #[tokio::test]
    async fn named_keys_identity_and_usage_attribution() {
        use std::os::unix::fs::PermissionsExt;
        #[derive(Default)]
        struct Sink(std::sync::Mutex<Vec<crate::usage::Record>>, crate::usage::Counters);
        impl crate::usage::UsageSink for Sink {
            fn record(&self, row: crate::usage::Record) { self.0.lock().unwrap().push(row); }
            fn counters(&self) -> &crate::usage::Counters { &self.1 }
        }
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("api-keys");
        std::fs::write(&file, r#"{"default":"legacy","agent":"agent-secret"}"#).unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
        let key = ApiKey::new("legacy").unwrap().with_named_file(&file).unwrap();
        let sink = Arc::new(Sink::default());
        let app = Router::new().route("/v1/models", get(|axum::Extension(identity): axum::Extension<KeyIdentity>| async move { identity.0 }))
            .layer(axum::middleware::from_fn_with_state(Auth {key:Some(key), internal:None}, require_key))
            .layer(axum::middleware::from_fn_with_state(crate::usage::Middleware::new(sink.clone()), crate::usage::track));
        for (token, name) in [("legacy", "default"), ("agent-secret", "agent")] {
            let response = app.clone().oneshot(axum::http::Request::get("/v1/models").header("authorization", format!("Bearer {token}")).body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(axum::body::to_bytes(response.into_body(), 1024).await.unwrap(), name);
            assert_eq!(sink.0.lock().unwrap().last().unwrap().key_label.as_deref(), Some(name));
        }
        let response = app.oneshot(axum::http::Request::get("/v1/models").header("authorization", "Bearer unknown").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(ApiKey::new("legacy").unwrap().with_named_file(&file).is_err());
    }
    #[tokio::test]
    async fn auth_runs_before_json_and_leaves_health_open() {
        let app = Router::new().route("/v1/chat/completions", post(|_: Json<serde_json::Value>| async { StatusCode::OK }))
            .route("/health", get(|| async { StatusCode::OK }))
            .layer(axum::middleware::from_fn_with_state(Auth { key: Some(ApiKey::new("secret").unwrap()), internal: None }, require_key));
        for authorization in [None, Some("Bearer wrong"), Some("Bearer secret-long")] {
            let mut req = axum::http::Request::post("/v1/chat/completions").header("content-type", "application/json");
            if let Some(value) = authorization { req = req.header("Authorization", value); }
            let response = app.clone().oneshot(req.body(Body::from("malformed")).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
            assert!(String::from_utf8_lossy(&body).contains("invalid_api_key"));
            assert!(!String::from_utf8_lossy(&body).contains("secret"));
        }
        assert_eq!(app.clone().oneshot(axum::http::Request::post("/v1/chat/completions")
            .header("Authorization", "bearer  secret").header("content-type", "application/json")
            .body(Body::from("{}")).unwrap()).await.unwrap().status(), StatusCode::OK);
        assert_eq!(app.oneshot(axum::http::Request::get("/health").body(Body::empty()).unwrap()).await.unwrap().status(), StatusCode::OK);
    }
}
