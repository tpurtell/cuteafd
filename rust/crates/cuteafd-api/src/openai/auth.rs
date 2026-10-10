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
pub struct ApiKey(Arc<[u8]>);
impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}
impl ApiKey {
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::new(text.trim_end_matches(['\r', '\n']))
            .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidData, message))
    }
    pub fn new(key: &str) -> Result<Self, &'static str> {
        if key.is_empty() || key.bytes().any(|b| !b.is_ascii_graphic()) {
            return Err("API key must be nonempty and contain no whitespace or control bytes");
        }
        Ok(Self(Arc::from(key.as_bytes())))
    }
    pub fn authorization(&self) -> String {
        format!("Bearer {}", std::str::from_utf8(&self.0).expect("key originated as UTF-8"))
    }
    pub fn accepts(&self, headers: &HeaderMap) -> bool {
        let token = bearer(headers).unwrap_or("");
        constant_time_eq(&self.0, token.as_bytes())
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
    request: Request, next: Next) -> Response {
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
