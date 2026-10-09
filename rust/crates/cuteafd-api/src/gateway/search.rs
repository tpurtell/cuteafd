//! Hosted web search: the gateway runs the search when the model calls the
//! `web_search` tool, so CLIs that expect a server-side search tool
//! (Anthropic `web_search_*`, Responses `web_search`) get one.
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use super::error::GatewayError;
use super::turn::{ToolSpec, WebSearchSpec};

/// The function-tool name the backend model sees for hosted search.
pub const TOOL_NAME: &str = "web_search";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    #[serde(default)]
    pub blocked_domains: Vec<String>,
    #[serde(default = "default_results")]
    pub max_results: usize,
}

fn default_results() -> usize { 5 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub url: String,
    pub title: String,
    /// Extracted page text or snippet, bounded by the provider.
    pub content: String,
    /// ISO date or provider's age string, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published: Option<String>,
}

mod providers;
pub use providers::{Exa, Searxng};

pub trait SearchProvider: Send + Sync + 'static {
    /// `exa`, `searxng`, ...
    fn name(&self) -> &str;
    fn search(&self, query: SearchQuery) -> BoxFuture<'static, Result<Vec<SearchHit>, GatewayError>>;
    /// Optional recency-aware search. Providers without date filtering return a clear error.
    fn search_recent_with_tape(&self, query: SearchQuery, recency: Option<u64>, tape: super::record::Tape) -> BoxFuture<'static, Result<Vec<SearchHit>, GatewayError>> {
        if recency.is_some() {
            return Box::pin(async { Err(GatewayError::unsupported("this search provider does not support recency")) });
        }
        self.search_with_tape(query, tape)
    }
    /// Page contents by URL, when the provider supports fetching (no direct gateway fetch).
    fn fetch_with_tape(&self, _url: String, _tape: super::record::Tape) -> BoxFuture<'static, Result<SearchHit, GatewayError>> {
        Box::pin(async { Err(GatewayError::unsupported("open not supported by this search provider")) })
    }
    /// Default keeps existing providers compatible; concrete HTTP providers record raw exchanges.
    fn search_with_tape(&self, query: SearchQuery, tape: super::record::Tape) -> BoxFuture<'static, Result<Vec<SearchHit>, GatewayError>> {
        let provider = self.name().to_string();
        let request = serde_json::to_value(&query).unwrap_or_default();
        let future = self.search(query);
        Box::pin(async move {
            let result = future.await;
            tape.record("search", || serde_json::json!({"provider":provider,"query":request,
                "hits":result.as_ref().ok(),"error":result.as_ref().err().map(|_| "unavailable")}));
            result
        })
    }
}

/// The function tool the backend sees in place of the client's hosted tool.
pub fn tool_spec(spec: &WebSearchSpec) -> ToolSpec {
    let mut description = String::from(
        "Search the web for current information. Returns result titles, URLs and page text. \
         Use it for facts that may be newer than your training data, and cite the URLs you use.");
    if !spec.allowed_domains.is_empty() {
        description.push_str(&format!(" Results are limited to: {}.", spec.allowed_domains.join(", ")));
    }
    ToolSpec {
        name: TOOL_NAME.into(),
        description: Some(description),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
            "additionalProperties": false,
        }),
        strict: false,
    }
}
