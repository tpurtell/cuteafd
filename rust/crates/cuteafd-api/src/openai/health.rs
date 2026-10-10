//! Immediate engine readiness, adapted from Hugh Madden's glm53f-afd health.rs.
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
use axum::{extract::{Request, State}, http::StatusCode, middleware::Next,
    response::{IntoResponse, Response}, Json};
use serde_json::json;
use std::sync::Arc;

#[derive(Clone)]
pub struct HealthWitness(pub Arc<dyn Fn() -> Option<String> + Send + Sync>);
impl std::fmt::Debug for HealthWitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("HealthWitness") }
}
impl HealthWitness {
    pub fn reason(&self) -> Option<String> { (self.0)() }
}
/// A failed expert wire is terminal: reject new inference without queueing it.
pub async fn require_ready(State(health): State<HealthWitness>, request: Request, next: Next) -> Response {
    let path = request.uri().path();
    // Gateway turns (Messages, Responses) run on the same engine.
    if request.method() == axum::http::Method::POST && matches!(path, "/v1/chat/completions" | "/v1/completions"
        | "/v1/messages" | "/v1/responses") {
        if let Some(reason) = health.reason() {
            return unavailable(reason);
        }
    }
    next.run(request).await
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{ModelProfile, NativeLimits, NativeRequest, ConsoleHub, router_for_model};
    use std::sync::{Mutex, atomic::{AtomicBool, Ordering}};
    use axum::body::Body;
    use tower::ServiceExt;

    #[tokio::test]
    async fn terminal_wire_failure_rejects_inference_before_parsing_and_health_is_immediate() {
        let failed = Arc::new(AtomicBool::new(false));
        let signal = failed.clone();
        let mut profile = ModelProfile::default();
        profile.engine_health = Some(HealthWitness(Arc::new(move || signal.load(Ordering::Acquire)
            .then(|| "expert rank 2 disconnected".into()))));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<NativeRequest>(1);
        let app = router_for_model(tx, NativeLimits::default(), Arc::new(Mutex::new(serde_json::Value::Null)),
            std::time::Duration::from_secs(25), ConsoleHub::disabled(), profile);
        let get = || axum::http::Request::get("/health").body(Body::empty()).unwrap();
        assert_eq!(app.clone().oneshot(get()).await.unwrap().status(), StatusCode::OK);
        failed.store(true, Ordering::Release);
        let response = app.clone().oneshot(get()).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = serde_json::from_slice(&axum::body::to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body, json!({"status":"unavailable", "reason":"expert rank 2 disconnected"}));
        let response = app.oneshot(axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json").body(Body::from("malformed")).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(rx.try_recv().is_err());
    }
    #[tokio::test]
    async fn stopped_scheduler_has_json_reason() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let app = crate::openai::router(tx);
        drop(rx);
        let response = app.oneshot(axum::http::Request::get("/health").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = serde_json::from_slice(&axum::body::to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body, json!({"status":"unavailable", "reason":"scheduler stopped"}));
    }
}
pub fn unavailable(reason: String) -> Response {
    (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status":"unavailable","reason":reason}))).into_response()
}
