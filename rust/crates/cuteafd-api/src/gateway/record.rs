//! Traffic recording seam. A recording middleware puts a [`Tape`] into the
//! request extensions; front ends take it as an extractor and copy it into
//! [`TurnRequest::tape`](super::turn::TurnRequest), so the backend and the
//! hosted-tool driver can log their upstream exchanges against the client
//! request that caused them. An empty tape records nothing.
use std::sync::Arc;

use axum::{extract::FromRequestParts, http::request::Parts};
use serde_json::Value;

/// Receives sanitized entries for one client exchange.
pub trait TapeSink: Send + Sync + 'static {
    /// `kind` names the entry (`upstream`, `search`, ...); `entry` must
    /// already be free of credentials.
    fn record(&self, kind: &str, entry: Value);
}

#[derive(Clone, Default)]
pub struct Tape(pub Option<Arc<dyn TapeSink>>);

impl Tape {
    pub fn is_recording(&self) -> bool { self.0.is_some() }
    pub fn record(&self, kind: &str, entry: impl FnOnce() -> Value) {
        if let Some(sink) = &self.0 { sink.record(kind, entry()); }
    }
}

impl std::fmt::Debug for Tape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "Tape(recording)" } else { "Tape(off)" })
    }
}

/// Tapes never affect equality of the turns that carry them.
// Replaced by the upstream recorder component when integrated.
impl Tape {
    pub fn frame(&self, _direction: &str, _text: &str) {}
}

impl PartialEq for Tape {
    fn eq(&self, _: &Self) -> bool { true }
}

#[axum::async_trait]
impl<S: Send + Sync> FromRequestParts<S> for Tape {
    type Rejection = std::convert::Infallible;
    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Self::Rejection> {
        Ok(parts.extensions.get::<Tape>().cloned().unwrap_or_default())
    }
}
