//! Codex 0.161 provider catalog (ModelsResponse, not OpenAI's /models list).
use std::sync::Arc;

use axum::{
    extract::State,
    http::header,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

use crate::gateway::{Gateway, GatewayError};

// Public rust-v0.161.0 gpt-6.1-sol metadata, without prompts, account/plan
// marketing or paid provider tiers. Tool/protocol flags remain unchanged.
const TEMPLATE: &str = include_str!("codex-model-template.json");

pub(super) fn document(gateway: &Gateway) -> Result<Value, GatewayError> {
    let backend = gateway.backend.models();
    let mut models = Vec::new();
    for advertised in gateway.models.listing(&backend) {
        // This catalog is for Codex, not the separate Claude model picker.
        if advertised.id.starts_with("claude-") {
            continue;
        }
        let target = gateway.models.resolve(&advertised.id)?;
        let served = backend.iter().find(|m| m.id == target).ok_or_else(|| {
            GatewayError::unsupported(format!("backend metadata unavailable for '{target}'"))
        })?;
        let mut entry: Value = serde_json::from_str(TEMPLATE).expect("checked-in Codex template");
        entry["slug"] = json!(advertised.id);
        entry["display_name"] = json!(format!("{} (cuteafd)", advertised.id));
        entry["description"] = json!(format!("Served by {}", served.id));
        entry["context_window"] = json!(served.context_tokens);
        entry["max_context_window"] = json!(served.context_tokens);
        // Codex has no max-output metadata field. Keep an extension for other
        // clients and account for the actual output reserve in its input budget.
        entry["max_output_tokens"] = json!(served.max_output_tokens);
        let limit = served.context_tokens.map(|context| {
            let reserve = served.max_output_tokens.unwrap_or(0);
            (u64::from(context) * 9 / 10).min(u64::from(context.saturating_sub(reserve)))
        });
        entry["auto_compact_token_limit"] = json!(limit);
        entry["effective_context_window_percent"] = json!(match served.context_tokens {
            Some(context) if context > 0 =>
                (u64::from(context.saturating_sub(served.max_output_tokens.unwrap_or(0))) * 100
                    / u64::from(context))
                .min(95),
            _ => 95,
        });
        models.push(entry);
    }
    Ok(json!({"models":models}))
}

pub(super) async fn get(State(gateway): State<Arc<Gateway>>) -> Response {
    match document(&gateway) {
        Ok(value) => {
            // Stable across requests, changes whenever limits or the catalog do.
            let bytes = serde_json::to_vec(&value).unwrap();
            let hash = bytes.iter().fold(0xcbf29ce484222325_u64, |h, b| {
                (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
            });
            ([(header::ETAG, format!("\"{hash:016x}\""))], Json(value)).into_response()
        }
        Err(e) => e.openai_response(),
    }
}
