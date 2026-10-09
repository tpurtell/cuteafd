//! Multi-protocol API gateway: protocol front ends over one session/turn
//! model and a pluggable backend (PLAN.md "v3 API gateway and sessions").
//!
//! ```text
//!  Anthropic Messages ─┐                         ┌─ Upstream (OpenAI/Anthropic-compatible HTTP)
//!  OpenAI Responses ───┼─ TurnRequest/TurnEvent ─┤
//!  OpenAI Realtime ────┘   + SessionStore        └─ Engine (phase B)
//! ```
//!
//! The existing `/v1/chat/completions` path (`crate::openai`) is untouched;
//! the gateway router is mounted beside it.
use std::sync::Arc;

use axum::Router;

pub mod anthropic;
pub mod auth;
pub mod backend;
mod driver;
pub mod error;
pub mod models;
pub mod realtime;
pub mod record;
pub mod responses;
pub mod search;
pub mod session;
pub mod turn;
pub mod upstream;

#[cfg(test)]
pub(crate) mod testing;

pub use backend::{Backend, BackendCapabilities, ModelInfo, TurnStream};
pub use error::{ErrorKind, GatewayError};
pub use models::ModelMap;
pub use search::SearchProvider;
pub use session::SessionStore;
pub use turn::{TurnEvent, TurnRequest};

/// Everything a front end needs.
pub struct Gateway {
    pub backend: Arc<dyn Backend>,
    pub models: ModelMap,
    pub sessions: SessionStore,
    /// Hosted web search; `None` makes hosted search tools a clear 400.
    pub search: Option<Arc<dyn SearchProvider>>,
}

impl Gateway {
    pub fn new(backend: Arc<dyn Backend>, models: ModelMap) -> Self {
        Self { backend, models, sessions: SessionStore::default(), search: None }
    }

    pub fn with_search(mut self, provider: Arc<dyn SearchProvider>) -> Self {
        self.search = Some(provider);
        self
    }

    /// Resolve the requested model and run one turn, executing hosted tools.
    /// `turn.requested_model` must be set; `turn.model` is filled here.
    pub async fn run(self: &Arc<Self>, mut turn: TurnRequest) -> Result<TurnStream, GatewayError> {
        turn.model = self.models.resolve(&turn.requested_model)?;
        driver::run(self.clone(), turn).await
    }

    /// Prompt tokens for `turn` (exact when the backend can count).
    pub async fn count_tokens(self: &Arc<Self>, mut turn: TurnRequest) -> Result<u32, GatewayError> {
        turn.model = self.models.resolve(&turn.requested_model)?;
        if let Some(spec) = turn.hosted.web_search.take() { turn.tools.push(search::tool_spec(&spec)); }
        self.backend.count_tokens(turn).await
    }
}

/// Every gateway route: Anthropic Messages, OpenAI Responses, Realtime and
/// the shared model listing. Merge it beside the chat-completions router, or
/// serve it alone with an upstream backend (`cuteafd gateway`).
pub fn router(gateway: Arc<Gateway>) -> Router {
    Router::new()
        .merge(anthropic::routes(gateway.clone()))
        .merge(responses::routes(gateway.clone()))
        .merge(realtime::routes(gateway.clone()))
        .merge(models_routes(gateway))
}

/// `GET /v1/models` and `GET /v1/models/{id}` in a shape both protocols
/// accept: OpenAI fields (`object`, `owned_by`, `created`) and Anthropic
/// fields (`type`, `display_name`, `created_at`, paging) side by side.
fn models_routes(gateway: Arc<Gateway>) -> Router {
    use axum::{extract::{Path, Query, State}, http::HeaderMap, response::{IntoResponse, Response}, routing::get, Json};
    #[derive(serde::Deserialize, Default)]
    struct Page { limit: Option<usize>, before_id: Option<String>, after_id: Option<String> }
    fn error_response(error: GatewayError, headers: &HeaderMap) -> Response {
        anthropic::with_request_id(if headers.contains_key("anthropic-version") || headers.contains_key("x-api-key") {
            error.anthropic_response()
        } else { error.openai_response() })
    }
    use serde_json::{json, Value};
    fn entry(model: &ModelInfo) -> Value {
        json!({"id": model.id, "object": "model", "type": "model", "display_name": model.id,
            "created": 0, "created_at": "1970-01-01T00:00:00Z", "owned_by": model.owned_by,
            "context_window": model.context_tokens, "max_output_tokens": model.max_output_tokens,
            "max_input_tokens": model.context_tokens, "max_tokens": model.max_output_tokens,
            "capabilities": null, "lifecycle": "active", "deprecated_at": null, "retires_at": null, "line": null})
    }
    async fn list(State(gateway): State<Arc<Gateway>>, headers: HeaderMap, page: Result<Query<Page>, axum::extract::rejection::QueryRejection>) -> Response {
        let page = match page { Ok(Query(page)) => page, Err(_) => return error_response(GatewayError::invalid("invalid model pagination parameters"), &headers) };
        let limit = page.limit.unwrap_or(20);
        if !(1..=1000).contains(&limit) || (page.before_id.is_some() && page.after_id.is_some()) {
            return error_response(GatewayError::invalid("limit must be 1..1000; use before_id or after_id, not both"), &headers);
        }
        let models = gateway.models.listing(&gateway.backend.models());
        let (mut start, mut end) = (0, models.len());
        if let Some(cursor) = page.after_id {
            let Some(i) = models.iter().position(|m| m.id == cursor) else { return error_response(GatewayError::invalid("unknown after_id cursor"), &headers) };
            start = i + 1;
        }
        let before = page.before_id.is_some();
        if let Some(cursor) = page.before_id {
            let Some(i) = models.iter().position(|m| m.id == cursor) else { return error_response(GatewayError::invalid("unknown before_id cursor"), &headers) };
            end = i;
        }
        let has_more = end - start > limit;
        if before { start = end.saturating_sub(limit).max(start); } else { end = (start + limit).min(end); }
        let page = &models[start..end];
        let data: Vec<Value> = page.iter().map(entry).collect();
        anthropic::with_request_id(Json(json!({"object": "list", "data": data, "has_more": has_more,
            "first_id": page.first().map(|m| m.id.clone()), "last_id": page.last().map(|m| m.id.clone())})).into_response())
    }
    async fn one(State(gateway): State<Arc<Gateway>>, Path(id): Path<String>, headers: HeaderMap) -> Response {
        match gateway.models.resolve(&id) {
            Ok(served) => {
                let listing = gateway.models.listing(&gateway.backend.models());
                let base = listing.iter().find(|m| m.id == served).cloned().unwrap_or_else(|| listing[0].clone());
                anthropic::with_request_id(Json(entry(&ModelInfo { id, ..base })).into_response())
            }
            Err(error) => error_response(error, &headers),
        }
    }
    Router::new()
        .route("/v1/models", get(list))
        .route("/v1/models/:id", get(one))
        .with_state(gateway)
}
