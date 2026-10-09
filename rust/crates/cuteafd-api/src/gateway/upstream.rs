//! HTTP backends for OpenAI Chat Completions and Anthropic Messages.
mod mapping;
mod sse;
#[cfg(test)]
mod tests;

use std::{sync::Arc, time::Duration};
use futures::{future::BoxFuture, StreamExt};
use serde_json::{json, Value};
use super::{backend::{estimate_tokens, BackendCapabilities, ModelInfo, TurnStream}, record::Tape,
    turn::TurnRequest, Backend, ErrorKind, GatewayError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor { OpenaiChat, Anthropic }
impl Flavor {
    pub fn name(self) -> &'static str { match self { Self::OpenaiChat => "openai-chat", Self::Anthropic => "anthropic" } }
}
impl std::str::FromStr for Flavor {
    type Err = GatewayError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value { "openai-chat" => Ok(Self::OpenaiChat), "anthropic" => Ok(Self::Anthropic),
            _ => Err(GatewayError::invalid("upstream flavor must be openai-chat or anthropic")) }
    }
}

/// Provider extensions stay opt-in; the generic Chat flavor uses only standard fields.
#[derive(Clone)]
pub struct UpstreamConfig {
    pub url: String,
    pub flavor: Flavor,
    pub key: Option<String>,
    pub model: String,
    pub capabilities: BackendCapabilities,
    pub context_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub thinking_toggle: bool,
    /// Optional absolute path for strict Chat requests; never downgrade on rejection.
    pub strict_tools_path: Option<String>,
    /// Anthropic-compatible providers do not necessarily implement this endpoint.
    pub anthropic_count_tokens: bool,
}
impl UpstreamConfig {
    pub fn new(url: impl Into<String>, flavor: Flavor, model: impl Into<String>) -> Self {
        Self { url: url.into(), flavor, model: model.into(), key: None,
            capabilities: BackendCapabilities { reasoning: true, ..Default::default() },
            context_tokens: None, max_output_tokens: None, thinking_toggle: false, strict_tools_path: None,
            anthropic_count_tokens: false }
    }
}

#[derive(Clone)]
pub struct Upstream {
    config: Arc<UpstreamConfig>,
    client: reqwest::Client,
    model: ModelInfo,
    capabilities: BackendCapabilities,
}
impl Upstream {
    pub fn new(config: UpstreamConfig) -> Result<Self, GatewayError> {
        let url = reqwest::Url::parse(&config.url).map_err(|_| GatewayError::invalid("invalid upstream URL"))?;
        if !matches!(url.scheme(), "http" | "https") || !url.username().is_empty() || url.password().is_some()
            || url.query().is_some() || url.fragment().is_some() {
            return Err(GatewayError::invalid("upstream URL must be HTTP(S), without credentials, query or fragment"));
        }
        if config.strict_tools_path.as_ref().is_some_and(|path| !path.starts_with('/') || path.starts_with("//") || path.contains(['?', '#'])) {
            return Err(GatewayError::invalid("strict tools path must be an absolute URL path"));
        }
        let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(20)).read_timeout(Duration::from_secs(120))
            .build().map_err(|_| GatewayError::internal("could not create upstream HTTP client"))?;
        let model = ModelInfo { id: config.model.clone(), context_tokens: config.context_tokens,
            max_output_tokens: config.max_output_tokens, owned_by: "upstream".into() };
        let capabilities = BackendCapabilities { exact_token_count: config.anthropic_count_tokens,
            ..config.capabilities };
        Ok(Self { config: Arc::new(config), client, model, capabilities })
    }

    pub fn endpoint(&self, suffix: &str) -> String {
        format!("{}/{}", self.config.url.trim_end_matches('/'), suffix.trim_start_matches('/'))
    }
    fn request(&self, method: reqwest::Method, suffix: &str) -> reqwest::RequestBuilder {
        let request = self.client.request(method, self.endpoint(suffix));
        match self.config.flavor {
            Flavor::OpenaiChat => if let Some(key) = &self.config.key { request.bearer_auth(key) } else { request },
            Flavor::Anthropic => {
                let request = request.header("anthropic-version", "2023-06-01");
                if let Some(key) = &self.config.key { request.header("x-api-key", key) } else { request }
            }
        }
    }

    /// Probe once during startup. A missing listing never prevents serving.
    pub async fn discover(mut self) -> Self {
        let result = async {
            let response = self.request(reqwest::Method::GET, "models").send().await
                .map_err(|_| GatewayError::upstream("upstream model discovery transport failure"))?;
            if !response.status().is_success() { return Err(http_error(response.status().as_u16())); }
            let value = bounded_json(response).await?;
            if let Some(model) = value["data"].as_array().and_then(|models| models.iter().find(|m| m["id"] == self.model.id)) {
                self.model.context_tokens = number(model, "context_window").or_else(|| number(model,"context_length")).or(self.model.context_tokens);
                self.model.max_output_tokens = number(model, "max_output_tokens").or_else(|| number(&model["top_provider"],"max_completion_tokens")).or(self.model.max_output_tokens);
                if let Some(owner) = model["owned_by"].as_str() { self.model.owned_by = owner.chars().take(128).collect(); }
                if let Some(modalities) = model.get("input_modalities").or_else(|| model["architecture"].get("input_modalities")).and_then(Value::as_array) {
                    self.capabilities.vision = modalities.iter().any(|m| m == "image");
                    self.capabilities.audio_in = modalities.iter().any(|m| m == "audio");
                }
            }
            if let Some(model) = value["data"].as_array().and_then(|models| models.iter().find(|m| m["id"] == self.model.id)) {
                if let Some(parameters) = model["supported_parameters"].as_array() {
                    self.capabilities.json_schema |= parameters.iter().any(|p| p == "structured_outputs");
                    self.capabilities.strict_tools |= parameters.iter().any(|p| p == "strict_tools");
                    self.capabilities.reasoning |= parameters.iter().any(|p| p == "reasoning");
                }
            }
            Arc::make_mut(&mut self.config).capabilities = self.capabilities;
            Ok::<_, GatewayError>(())
        }.await;
        if let Err(error) = result { tracing::warn!(status = error.upstream_status, "upstream model discovery failed; using configured metadata"); }
        self
    }

    fn completion_request(&self, turn: &TurnRequest, suffix: &str) -> reqwest::RequestBuilder {
        if self.config.flavor == Flavor::OpenaiChat && turn.tools.iter().any(|tool| tool.strict) {
            if let Some(path) = &self.config.strict_tools_path {
                let mut url = reqwest::Url::parse(&self.config.url).expect("validated upstream URL");
                url.set_path(path);
                let request = self.client.post(url);
                return if let Some(key) = &self.config.key { request.bearer_auth(key) } else { request };
            }
        }
        self.request(reqwest::Method::POST,suffix)
    }
    pub fn map_request(&self, turn: &TurnRequest) -> Result<Value, GatewayError> {
        mapping::request(turn, &self.config)
    }
}
impl Backend for Upstream {
    fn name(&self) -> &str { "upstream" }
    fn capabilities(&self) -> BackendCapabilities { self.capabilities }
    fn models(&self) -> Vec<ModelInfo> { vec![self.model.clone()] }
    fn start(&self, turn: TurnRequest) -> BoxFuture<'static, Result<TurnStream, GatewayError>> {
        let this = self.clone();
        Box::pin(async move {
            let request = this.map_request(&turn)?;
            let names = mapping::WireNames::new(&turn);
            let suffix = match this.config.flavor { Flavor::OpenaiChat => "chat/completions", Flavor::Anthropic => "v1/messages" };
            let response = this.completion_request(&turn, suffix).json(&request).send().await
                .map_err(|_| GatewayError::upstream("upstream request transport failure"))?;
            let status = response.status().as_u16();
            let mut recording = Exchange::new(turn.tape, this.config.flavor, request, status);
            if !response.status().is_success() {
                let body = bounded_body(response, 64 * 1024).await.unwrap_or_default();
                recording.push(&body);
                return Err(http_error(status));
            }
            let mut stream = sse::events(response.bytes_stream(), this.config.flavor, recording).map(move |event| {
                event.map(|event| match event {
                    super::turn::TurnEvent::ToolCallStart { index,id,name } => super::turn::TurnEvent::ToolCallStart {
                        index,id,name:names.original(&name) },
                    other => other,
                })
            });
            // Await a meaningful event before committing the client HTTP response.
            let first = stream.next().await.transpose()?.ok_or_else(|| GatewayError::upstream("empty upstream stream"))?;
            Ok(Box::pin(futures::stream::once(async { Ok(first) }).chain(stream)) as TurnStream)
        })
    }
    fn count_tokens(&self, turn: TurnRequest) -> BoxFuture<'static, Result<u32, GatewayError>> {
        let this = self.clone();
        Box::pin(async move {
            if !this.config.anthropic_count_tokens { return Ok(estimate_tokens(&turn)); }
            let mut request = this.map_request(&turn)?;
            for key in ["stream", "max_tokens", "temperature", "top_p", "top_k", "stop_sequences"] {
                request.as_object_mut().expect("mapped request object").remove(key);
            }
            let response = this.request(reqwest::Method::POST, "v1/messages/count_tokens").json(&request).send().await
                .map_err(|_| GatewayError::upstream("upstream token count transport failure"))?;
            let status = response.status().as_u16();
            let body = bounded_body(response, 64 * 1024).await?;
            turn.tape.record("upstream", || json!({"request": request, "status": status, "body": String::from_utf8_lossy(&body), "path": "v1/messages/count_tokens"}));
            if !(200..300).contains(&status) { return Err(http_error(status)); }
            let value: Value = serde_json::from_slice(&body).map_err(|_| GatewayError::upstream("invalid upstream token count JSON"))?;
            number(&value, "input_tokens").ok_or_else(|| GatewayError::upstream("missing upstream input_tokens"))
        })
    }
}

pub(crate) fn http_error(status: u16) -> GatewayError {
    let kind = match status { 429 => ErrorKind::RateLimited, 503 | 529 => ErrorKind::Overloaded, _ => ErrorKind::Upstream };
    let mut error = GatewayError::new(kind, format!("upstream returned HTTP {status}"));
    error.upstream_status = Some(status);
    error
}
pub(crate) fn number(value: &Value, key: &str) -> Option<u32> {
    value[key].as_u64().and_then(|n| u32::try_from(n).ok())
}
pub(crate) async fn bounded_body(response: reqwest::Response, limit: usize) -> Result<Vec<u8>, GatewayError> {
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| GatewayError::upstream("upstream response transport failure"))?;
        if bytes.len().saturating_add(chunk.len()) > limit { return Err(GatewayError::upstream("upstream response exceeds size limit")); }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}
pub(crate) async fn bounded_json(response: reqwest::Response) -> Result<Value, GatewayError> {
    serde_json::from_slice(&bounded_body(response, 2 * 1024 * 1024).await?)
        .map_err(|_| GatewayError::upstream("invalid upstream JSON"))
}

/// Owned by the response stream, so drop/cancellation also closes and records the exchange.
pub(super) struct Exchange { tape: Tape, flavor: Flavor, request: Value, status: u16, body: Vec<u8>, truncated: bool }
impl Exchange {
    fn new(tape: Tape, flavor: Flavor, request: Value, status: u16) -> Self {
        Self { tape, flavor, request, status, body: Vec::new(), truncated: false }
    }
    fn push(&mut self, bytes: &[u8]) {
        if !self.tape.is_recording() { return; }
        let available = (16 * 1024 * 1024usize).saturating_sub(self.body.len());
        self.body.extend_from_slice(&bytes[..bytes.len().min(available)]);
        self.truncated |= bytes.len() > available;
    }
}
impl Drop for Exchange {
    fn drop(&mut self) {
        self.tape.record("upstream", || json!({"flavor": self.flavor.name(), "request": self.request,
            "status": self.status, "body": String::from_utf8_lossy(&self.body), "truncated": self.truncated}));
    }
}
