//! Typed gateway errors. Each front end renders these in its own wire shape
//! (Anthropic `{"type":"error","error":{...}}`, OpenAI `{"error":{...}}`,
//! Realtime `error` events).
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// 400: malformed or unsupported request content.
    InvalidRequest,
    /// 401: missing or wrong API key.
    Authentication,
    /// 403
    PermissionDenied,
    /// 404: unknown model, response id, session or route.
    NotFound,
    /// 413
    RequestTooLarge,
    /// 429
    RateLimited,
    /// 503/529: backend busy or not ready.
    Overloaded,
    /// 502: the upstream API failed or sent something unparsable.
    Upstream,
    /// 400 with a clear "not supported" message (e.g. audio output without TTS).
    Unsupported,
    /// 500
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayError {
    pub kind: ErrorKind,
    pub message: String,
    /// The request field at fault, when known (OpenAI `param`).
    pub param: Option<String>,
    /// HTTP status an upstream returned, preserved for pass-through.
    pub upstream_status: Option<u16>,
}

impl GatewayError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: bounded(message.into()), param: None, upstream_status: None }
    }
    pub fn invalid(message: impl Into<String>) -> Self { Self::new(ErrorKind::InvalidRequest, message) }
    pub fn unsupported(message: impl Into<String>) -> Self { Self::new(ErrorKind::Unsupported, message) }
    pub fn not_found(message: impl Into<String>) -> Self { Self::new(ErrorKind::NotFound, message) }
    pub fn upstream(message: impl Into<String>) -> Self { Self::new(ErrorKind::Upstream, message) }
    pub fn internal(message: impl Into<String>) -> Self { Self::new(ErrorKind::Internal, message) }
    pub fn with_param(mut self, param: impl Into<String>) -> Self { self.param = Some(param.into()); self }

    pub fn status(&self) -> u16 {
        if let (ErrorKind::Upstream, Some(status)) = (self.kind, self.upstream_status) {
            if (400..600).contains(&status) { return status; }
        }
        match self.kind {
            ErrorKind::InvalidRequest | ErrorKind::Unsupported => 400,
            ErrorKind::Authentication => 401,
            ErrorKind::PermissionDenied => 403,
            ErrorKind::NotFound => 404,
            ErrorKind::RequestTooLarge => 413,
            ErrorKind::RateLimited => 429,
            ErrorKind::Overloaded => 503,
            ErrorKind::Upstream => 502,
            ErrorKind::Internal => 500,
        }
    }
}

impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.message) }
}
impl std::error::Error for GatewayError {}

/// Request-derived strings end up in messages; keep responses bounded.
fn bounded(mut message: String) -> String {
    const MAX: usize = 2048;
    if message.len() > MAX {
        let mut end = MAX;
        while !message.is_char_boundary(end) { end -= 1; }
        message.truncate(end);
        message.push_str("...");
    }
    message
}

impl GatewayError {
    /// Anthropic `error.type` for this error.
    pub fn anthropic_type(&self) -> &'static str {
        match self.status() {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            503 | 529 => "overloaded_error",
            400..=499 => "invalid_request_error",
            _ => "api_error",
        }
    }
    /// OpenAI `error.type` for this error.
    pub fn openai_type(&self) -> &'static str {
        match self.status() {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            429 => "rate_limit_error",
            400..=499 => "invalid_request_error",
            _ => "server_error",
        }
    }
    /// Anthropic error body: `{"type":"error","error":{"type","message"}}`.
    pub fn anthropic_body(&self) -> serde_json::Value {
        serde_json::json!({"type": "error", "error": {"type": self.anthropic_type(), "message": self.message}})
    }
    /// OpenAI error body: `{"error":{"message","type","param","code"}}`.
    pub fn openai_body(&self) -> serde_json::Value {
        let code = match self.kind {
            ErrorKind::NotFound if self.param.as_deref() == Some("model") => Some("model_not_found"),
            ErrorKind::Authentication => Some("invalid_api_key"),
            ErrorKind::Unsupported => Some("unsupported_value"),
            _ => None,
        };
        serde_json::json!({"error": {"message": self.message, "type": self.openai_type(), "param": self.param, "code": code}})
    }
    pub fn anthropic_response(&self) -> axum::response::Response {
        use axum::response::IntoResponse;
        let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
        let mut body = self.anthropic_body();
        body["request_id"] = serde_json::Value::String(request_id.clone());
        let status = if self.kind == ErrorKind::Overloaded { 529 } else { self.status() };
        let mut response = (status_code(status), axum::Json(body)).into_response();
        response.headers_mut().insert("request-id", request_id.parse().unwrap());
        response
    }
    pub fn openai_response(&self) -> axum::response::Response {
        use axum::response::IntoResponse;
        (status_code(self.status()), axum::Json(self.openai_body())).into_response()
    }
}

fn status_code(status: u16) -> axum::http::StatusCode {
    axum::http::StatusCode::from_u16(status).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
}
