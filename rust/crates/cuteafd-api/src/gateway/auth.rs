//! Gateway API-key check, accepting the credential forms the official
//! clients send so users configure a key exactly as for the real service:
//! - `Authorization: Bearer KEY` (OpenAI SDKs, Codex, `ANTHROPIC_AUTH_TOKEN`);
//! - `x-api-key: KEY` (Anthropic SDKs, Claude Code `ANTHROPIC_API_KEY`);
//! - `Sec-WebSocket-Protocol: openai-insecure-api-key.KEY` (browser Realtime).
//!
//! Failures answer in the protocol's own error shape: Anthropic's for
//! `/v1/messages*`, OpenAI's elsewhere (Realtime rejects before upgrade).
use axum::{extract::{Request, State}, http::HeaderMap, middleware::Next, response::Response};

use super::error::{ErrorKind, GatewayError};
use crate::openai::auth::{bearer, ApiKey};

#[derive(Clone, Default)]
pub struct GatewayAuth {
    pub key: Option<ApiKey>,
}

/// The credential a request presented, in any accepted form.
pub fn presented(headers: &HeaderMap) -> Option<&str> {
    if let Some(token) = bearer(headers) { return Some(token); }
    if let Some(key) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) { return Some(key.trim()); }
    headers.get("sec-websocket-protocol").and_then(|v| v.to_str().ok()).and_then(|value| {
        value.split(',').map(str::trim).find_map(|proto| proto.strip_prefix("openai-insecure-api-key."))
    })
}

pub fn accepts(key: &ApiKey, headers: &HeaderMap) -> bool {
    let mut authorization = HeaderMap::new();
    let sent = presented(headers).unwrap_or("");
    // Reuse the constant-time bearer comparison on the normalized credential.
    let Ok(value) = format!("Bearer {sent}").parse() else { return false };
    authorization.insert(axum::http::header::AUTHORIZATION, value);
    key.accepts(&authorization)
}

pub async fn require_key(State(auth): State<GatewayAuth>, request: Request, next: Next) -> Response {
    let path = request.uri().path();
    if let Some(key) = &auth.key {
        if (path == "/v1" || path.starts_with("/v1/")) && !accepts(key, request.headers()) {
            let error = GatewayError::new(ErrorKind::Authentication, "invalid x-api-key or Authorization bearer key");
            return if path.starts_with("/v1/messages") { error.anthropic_response() } else { error.openai_response() };
        }
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_client_credential_form() {
        let key = ApiKey::new("sk-local").unwrap();
        for (name, value) in [("authorization", "Bearer sk-local"), ("x-api-key", "sk-local"),
            ("sec-websocket-protocol", "realtime, openai-insecure-api-key.sk-local")] {
            let mut headers = HeaderMap::new();
            headers.insert(name, value.parse().unwrap());
            assert!(accepts(&key, &headers), "{name}");
        }
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-other".parse().unwrap());
        assert!(!accepts(&key, &headers));
        assert!(!accepts(&key, &HeaderMap::new()));
    }
}
