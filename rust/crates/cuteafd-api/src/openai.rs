//! Official V4.1 protocol conversion and bounded handoff to a CUDA owner.
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
pub use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage};
use deepseek_recipe::{
    openai::ChatCompletionRequest,
    request::{ConversionOptions, ProtocolRequest},
    response::ProtocolResponse,
    stream::StreamProcessor,
    util::append_delta::AppendDelta,
};
use deepseek_recipe_encoding::{v4::dsv4::DeepseekV4Encoding, v4::dsv41::DeepseekV41Encoding, PromptEncoding};
use deepseek_recipe_core::conversation::ReasoningEffort;
use cuteafd_core::TargetSamplingParams;
use futures::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

pub const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";

/// DeepSeek V4/V4.1 `<｜end▁of▁sentence｜>`.
pub const DEEPSEEK_EOS_TOKEN_ID: u32 = 1;
/// DeepSeek V4/V4.1 `</think>`.
const DEEPSEEK_THINK_CLOSE_TOKEN_ID: u32 = 128_822;

/// Prompt rendering and output parsing a served model uses.
#[derive(Debug, Clone)]
pub enum ModelEncoding {
    DeepseekV4,
    DeepseekV41,
    /// GLM 5.x: the checkpoint's chat template and GLM XML tool calls.
    Glm(Arc<glm5::GlmEncoding>),
    /// Qwen 3.8 Flash Next: the checkpoint's chat template and Qwen3-Coder XML tool calls.
    Qwen(Arc<qwen4::QwenEncoding>),
}

impl ModelEncoding {
    /// The checkpoint's end-of-sequence token ids.
    pub fn eos_token_ids(&self) -> Vec<u32> {
        match self {
            Self::DeepseekV4 | Self::DeepseekV41 => vec![DEEPSEEK_EOS_TOKEN_ID],
            Self::Glm(encoding) => encoding.tokens().eos.clone(),
            Self::Qwen(encoding) => encoding.tokens().eos.clone(),
        }
    }

    fn think_close_token_id(&self) -> u32 {
        match self {
            Self::DeepseekV4 | Self::DeepseekV41 => DEEPSEEK_THINK_CLOSE_TOKEN_ID,
            Self::Glm(encoding) => encoding.tokens().think_close,
            Self::Qwen(encoding) => encoding.tokens().think_close,
        }
    }
}

/// Availability is explicit so a template can never stand in for an encoder.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct MediaCapabilities {
    pub vision: bool,
    pub audio: bool,
}

static MEDIA_INPUT_POLICY: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
/// Restrict inputs according to launcher policy; enabling never creates a capability.
pub fn set_media_input_policy(vision: bool, audio: bool) {
    let _ = MEDIA_INPUT_POLICY.set((vision, audio));
}

/// The served model's public id, prompt encoding and EOS token ids.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Live readiness for remote vision; absent on existing local serving paths.
    pub vision_health: Option<Arc<std::sync::atomic::AtomicBool>>,
    pub media_preparer: Option<Arc<media::MediaPreparer>>,
    /// Loaded encoder capabilities, not checkpoint metadata or requested placement.
    pub capabilities: MediaCapabilities,
    pub id: String,
    pub encoding: ModelEncoding,
    /// Token ids that end generation. Requests carry these (plus any
    /// family turn markers) in `NativeRequest::stop_token_ids`.
    pub eos_token_ids: Vec<u32>,
}

impl ModelProfile {
    /// A profile whose EOS ids come from `encoding`.
    pub fn new(id: impl Into<String>, encoding: ModelEncoding) -> Self {
        let eos_token_ids = encoding.eos_token_ids();
        let capabilities = MediaCapabilities { vision: matches!(encoding, ModelEncoding::DeepseekV41), audio: false };
        Self { id: id.into(), encoding, eos_token_ids, capabilities, media_preparer: None, vision_health: None }
    }

    /// Install only after the matching encoder is loaded and ready. A processor
    /// alone must never be advertised as a vision deployment.
    pub fn with_loaded_vision(mut self, preparer: Arc<media::MediaPreparer>) -> Self {
        self.media_preparer = Some(preparer);
        self.capabilities.vision = true;
        self
    }

    /// Token ids that end generation for one request.
    fn stop_token_ids(&self, tools_declared: bool) -> Vec<u32> {
        match &self.encoding {
            ModelEncoding::Glm(_) | ModelEncoding::Qwen(_) => {
                let mut ids = match &self.encoding {
                    ModelEncoding::Glm(encoding) => encoding.stop_token_ids(tools_declared),
                    ModelEncoding::Qwen(encoding) => encoding.stop_token_ids(tools_declared),
                    _ => unreachable!(),
                };
                ids.extend(&self.eos_token_ids);
                ids.sort_unstable();
                ids.dedup();
                ids
            }
            ModelEncoding::DeepseekV4 | ModelEncoding::DeepseekV41 => self.eos_token_ids.clone(),
        }
    }
}

impl Default for ModelProfile {
    fn default() -> Self {
        Self::new(MODEL, ModelEncoding::DeepseekV41)
    }
}
mod limits;
mod admission;
mod constraints;
mod tools;
pub mod chat;
use chat::{glm5, qwen4};
pub use constraints::NativeConstraint;
mod images;
pub mod media;
pub mod console;
pub use console::ConsoleHub;
pub mod probe;
#[cfg(test)]
mod unicode_tests;
pub use limits::{NativeLimits, MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS};
#[derive(Debug, Clone)]
pub enum NativeFailure {
    BadRequest(String),
    Unavailable(String),
    Worker(String),
}
impl std::fmt::Display for NativeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::BadRequest(message) | Self::Unavailable(message) | Self::Worker(message) => f.write_str(message) }
    }
}
impl std::error::Error for NativeFailure {}
impl From<String> for NativeFailure {
    fn from(message: String) -> Self { Self::Worker(message) }
}
impl From<&str> for NativeFailure {
    fn from(message: &str) -> Self { Self::Worker(message.into()) }
}
/// Silence after which a streamed response sends an SSE comment line.
const SSE_KEEPALIVE: std::time::Duration =
    std::time::Duration::from_millis(if cfg!(test) { 50 } else { 15_000 });

pub struct NativeRequest {
    pub prompt: String,
    pub constraint: Option<NativeConstraint>,
    pub images: Vec<cuteafd_loader::V41Image>,
    /// Generic-family media, in template order. V4.1 only uses `images`.
    pub media: Vec<Arc<cuteafd_loader::media::PreparedImage>>,
    pub max_tokens: usize,
    /// Resolved target-sampling parameters. `TargetSamplingParams::greedy()`
    /// keeps the legacy device-argmax route; anything else selects from the
    /// full vocabulary through the shared exact sampler.
    pub sampling: TargetSamplingParams,
    /// Token ids that end generation: the profile's EOS ids plus, for GLM,
    /// its turn markers (and `<tool_call>` when no tools are declared).
    /// DeepSeek engines keep their built-in EOS handling.
    pub stop_token_ids: Vec<u32>,
    pub events: mpsc::UnboundedSender<Result<InferenceChunk, NativeFailure>>,
    /// Benchmark diagnostics for this request (`probe::HEADER`); `None` for every ordinary client.
    pub probe: Option<Arc<probe::Probe>>,
}
/// Serving statistics the CUDA owner publishes (a JSON object; `null` until the first publish).
pub type SharedStats = Arc<Mutex<Value>>;
#[derive(Clone)]
struct NativeState {
    queue: mpsc::Sender<NativeRequest>,
    limits: NativeLimits,
    images: images::ImageDecoder,
    stats: SharedStats,
    tables: cuteafd_loader::MappedTableStatsReader,
    admission: admission::Admission,
    profile: Arc<ModelProfile>,
}
pub fn router(queue: mpsc::Sender<NativeRequest>) -> Router {
    router_with_limits(queue, NativeLimits::default())
}
pub fn router_with_limits(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits) -> Router {
    router_with_limits_and_stats(queue, limits, Arc::new(Mutex::new(Value::Null)))
}
pub fn router_with_limits_and_stats(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits, stats: SharedStats) -> Router {
    router_with_admission(queue, limits, stats, std::time::Duration::from_secs(25))
}
pub fn router_with_admission(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits,
    stats: SharedStats, wait: std::time::Duration) -> Router {
    router_with_console(queue, limits, stats, wait, ConsoleHub::disabled())
}
/// The serving router plus the live console at `/` fed by `console`.
pub fn router_with_console(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits,
    stats: SharedStats, wait: std::time::Duration, console: Arc<ConsoleHub>) -> Router {
    router_for_model(queue, limits, stats, wait, console, ModelProfile::default())
}
/// The serving router for `profile` (its model id and prompt encoding).
pub fn router_for_model(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits,
    stats: SharedStats, wait: std::time::Duration, console: Arc<ConsoleHub>, profile: ModelProfile) -> Router {
    let mut profile = profile;
    if let Some(&(vision, audio)) = MEDIA_INPUT_POLICY.get() {
        profile.capabilities.vision &= vision;
        profile.capabilities.audio &= audio;
    }
    let tables = cuteafd_loader::MappedTableStatsReader::registered();
    let admission = admission::Admission::new(queue.max_capacity(), wait);
    let images = images::ImageDecoder::new(queue.max_capacity());
    let console_routes = Router::new()
        .route("/", get(console::page))
        .route("/v1/console", get(console::socket))
        .route("/v1/console/snapshot", get(console::snapshot))
        .route("/assets/cuteafd-ui.css", get(console::ui_css))
        .route("/assets/cuteafd-ui.js", get(console::ui_js))
        .route("/assets/cuteafd-logo.svg", get(console::logo))
        .route("/assets/cuteafd-mark.svg", get(console::mark))
        .with_state(console);
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/stats", get(stats_route))
        .route("/v1/chat/completions", post(chat))
        .layer(axum::extract::DefaultBodyLimit::max(if profile.media_preparer.is_some() { 256 << 20 } else { images::BODY_BYTES }))
        .with_state(NativeState { queue, limits, images, stats, tables, admission, profile: Arc::new(profile) })
        .merge(console_routes)
}
async fn stats_route(State(state): State<NativeState>) -> Json<Value> {
    let mut value = state.stats.lock().map(|stats| stats.clone()).unwrap_or(Value::Null);
    if !value.is_object() { value = json!({}); }
    refresh_mapped_tables(&mut value, &state.tables);
    let object = value.as_object_mut().unwrap();
    object.extend(state.admission.metrics().as_object().unwrap().clone());
    object.insert("http_queue_len".into(), json!(state.queue.max_capacity() - state.queue.capacity()));
    Json(value)
}
// Preserve scheduler-published interval/device diagnostics; refresh only the
// cumulative counters and backend metadata, including before the first publish.
fn refresh_mapped_tables(value: &mut Value, reader: &cuteafd_loader::MappedTableStatsReader) {
    let tables = reader.snapshot();
    if tables.is_empty() { return; }
    let fresh = tables.into_iter().map(|table| {
        let mut entry = value["mapped_tables"].as_array().and_then(|cached| cached.iter()
            .find(|entry| entry["name"].as_str() == Some(&table.name)))
            .cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
        entry["name"] = json!(table.name);
        entry["backend"] = json!(table.backend);
        entry["accounting"] = json!(table.accounting);
        entry["cumulative"] = json!(table.cumulative);
        entry
    }).collect::<Vec<_>>();
    value["mapped_tables"] = json!(fresh);
}

async fn models(State(state): State<NativeState>) -> Json<Value> {
    let owner = state.profile.id.split_once('/').map_or("cuteafd", |(owner, _)| owner);
    let mut model = json!({"id":state.profile.id,"object":"model","owned_by":owner,
        "capabilities":state.profile.capabilities,"max_context_tokens":state.limits.context(),"max_output_tokens":state.limits.output()});
    if let ModelEncoding::Glm(encoding) = &state.profile.encoding {
        if let Some(provenance) = encoding.template_provenance() {
            model["chat_template"] = json!(provenance);
        }
    }
    Json(json!({"object":"list","data":[model]}))
}
async fn health(State(state): State<NativeState>) -> Response {
    let vision = state.profile.vision_health.as_ref().map(|h| h.load(Ordering::Acquire));
    let status = if state.queue.is_closed() || vision == Some(false) {
        StatusCode::SERVICE_UNAVAILABLE
    } else { StatusCode::OK };
    match vision {
        Some(healthy) => (status, Json(json!({"vision": if healthy { "ready" } else { "failed" }}))).into_response(),
        None => status.into_response(),
    }
}
fn error_body(message: impl ToString) -> Value {
    // Bound upstream parse/validation details before they reach the response
    // body: serde invalid-type errors echo the full offending string (e.g. a
    // 100 KB string in a wrongly-typed field). Same class as the JsonRejection
    // echo fixed in lib.rs (upstream vLLM #49239). This router is the
    // production serving path (mounted by the daemon) — verified live on the
    // fleet 2026-09-15.
    let message = crate::error::bounded_error_detail(&message.to_string());
    json!({"error":{"message":message,"type":"native_v41_error"}})
}
fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(error_body(message))).into_response()
}
fn sse_error(message: impl ToString) -> String {
    // Keep the HTTP error envelope and bound, with JSON escaping so a worker
    // message containing newlines cannot inject SSE frames.
    format!("data: {}\n\n", error_body(message))
}

static NEXT_TARGET_SEED: AtomicU64 = AtomicU64::new(0);

fn generated_target_seed() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    now ^ NEXT_TARGET_SEED.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
}

/// Resolve the served target-sampling parameters from the raw request body.
///
/// `top_k`, `min_p` and the signed `seed` are not part of the pinned
/// `deepseek-recipe` adapter, so they are read here. `top_p`/`temperature` are
/// also re-read so one validator owns the whole sampling contract.
///
/// Greedy is the served default: an absent or zero `temperature` keeps the
/// legacy argmax route and ignores the other filters. Once sampling is opted
/// into, unspecified filters are disabled (`top_p = 1`, no `top_k`,
/// `min_p = 0`) rather than silently inheriting a truncation.
fn request_target_sampling(body: &Value) -> Result<TargetSamplingParams, String> {
    let number = |name: &str| -> Result<Option<f64>, String> {
        match body.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_f64()
                .map(Some)
                .ok_or_else(|| format!("{name} must be a finite number")),
        }
    };
    let temperature = number("temperature")?.unwrap_or(0.0) as f32;
    let top_p = number("top_p")?.unwrap_or(1.0) as f32;
    let min_p = number("min_p")?.unwrap_or(0.0) as f32;
    let top_k = match body.get("top_k") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let k = value
                .as_i64()
                .ok_or_else(|| "top_k must be an integer".to_owned())?;
            if k == 0 || k == -1 {
                None
            } else if k < 0 {
                return Err("top_k must be -1, 0, or a positive integer".to_owned());
            } else {
                Some(k as usize)
            }
        }
    };
    let seed = match body.get("seed") {
        None | Some(Value::Null) => generated_target_seed(),
        Some(value) => {
            let seed = value
                .as_i64()
                .ok_or_else(|| "seed must be an integer".to_owned())?;
            TargetSamplingParams::seed_from_i64(seed)
        }
    };
    TargetSamplingParams::new(temperature, top_p, top_k, min_p, seed)
        .map_err(|error| error.to_string())
}
type ChatGenerator = <<ChatCompletionRequest as ProtocolRequest>::Response as ProtocolResponse>::ChunkGenerator;
type ChatChunk = <ChatGenerator as deepseek_recipe::stream::ChunkGenerator>::Chunk;
type ChatChunks = std::pin::Pin<Box<dyn futures::Stream<Item = Result<ChatChunk, deepseek_recipe::stream::StreamError>> + Send>>;

/// A family whose prompt is the checkpoint's own chat template.
enum Templated {
    Glm(Arc<glm5::GlmEncoding>),
    Qwen(Arc<qwen4::QwenEncoding>),
}

impl Templated {
    /// The request's thinking switch and, for GLM, the effort its template
    /// renders under the server's off form. Qwen's template reads
    /// `reasoning_effort` from the request itself.
    fn thinking(&self, body: &Value) -> Result<glm5::GlmThinking, String> {
        match self {
            Self::Glm(encoding) => glm5::resolve_glm_thinking(body, encoding.thinking_off()),
            Self::Qwen(_) => Ok(glm5::GlmThinking { enabled: glm5::resolve_thinking(body)?, effort: None }),
        }
    }
}

/// The family's generated-text parser in front of the OpenAI chunk generator.
enum OutputProcessor {
    Deepseek(StreamProcessor<ChatGenerator>),
    Glm(glm5::GlmStreamProcessor<ChatGenerator>),
    Qwen(glm5::GlmStreamProcessor<ChatGenerator, qwen4::QwenOutputParser>),
}

impl OutputProcessor {
    fn process(self, input: impl futures::Stream<Item = InferenceChunk> + Send + 'static) -> ChatChunks {
        match self {
            Self::Deepseek(processor) => Box::pin(processor.process(input)),
            Self::Glm(processor) => Box::pin(processor.process(input)),
            Self::Qwen(processor) => Box::pin(processor.process(input)),
        }
    }
}

async fn chat(State(state): State<NativeState>, headers: axum::http::HeaderMap, Json(mut body): Json<Value>) -> Response {
    if let Err(message) = images::guard_content(&body, state.profile.capabilities) {
        return error(StatusCode::BAD_REQUEST, message);
    }
    let probe = headers.get(probe::HEADER).and_then(|v| v.to_str().ok()).and_then(|id| probe::registry().claim(id));
    if let Some(p) = &probe {
        if let Err(message) = p.spec.validate_media() {
            p.fail(format!("{message:#}"));
            return error(StatusCode::BAD_REQUEST, message);
        }
        if !p.spec.media.is_empty() && (!state.profile.capabilities.vision || state.profile.media_preparer.is_none()
            || matches!(state.profile.encoding, ModelEncoding::DeepseekV41)) {
            p.fail("probe media requires a loaded encoder");
            return error(StatusCode::BAD_REQUEST, "probe media requires a loaded encoder");
        }
    }
    let media_sources = match &state.profile.media_preparer {
        Some(preparer) => match media::extract_image_sources(&body, preparer.limits.images) {
            Ok(sources) => sources,
            Err(message) => return error(StatusCode::BAD_REQUEST, message),
        },
        None => Vec::new(),
    };
    if (!media_sources.is_empty() || probe.as_ref().is_some_and(|p| !p.spec.media.is_empty()))
        && state.profile.vision_health.as_ref().is_some_and(|h| !h.load(Ordering::Acquire)) {
        return error(StatusCode::SERVICE_UNAVAILABLE, "vision encoder unavailable");
    }
    if !matches!(state.profile.encoding, ModelEncoding::DeepseekV41) && state.profile.capabilities.vision
        && state.profile.media_preparer.is_none() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "vision image processor unavailable");
    }
    let media_sources = if let Some(p) = probe.as_ref().filter(|p| !p.spec.media.is_empty()) {
        if !media_sources.is_empty() {
            return error(StatusCode::BAD_REQUEST, "probe media sources must not also appear in chat content");
        }
        p.spec.media.iter().map(|span| {
            let source = span.image_url.as_ref().expect("validated probe source");
            media::MediaSource { url: source.url.clone(), low: source.detail.as_deref() == Some("low") }
        }).collect()
    } else { media_sources };
    // Families rendered from the checkpoint's own chat template.
    let glm = match &state.profile.encoding {
        ModelEncoding::Glm(encoding) => Some(Templated::Glm(encoding.clone())),
        ModelEncoding::Qwen(encoding) => Some(Templated::Qwen(encoding.clone())),
        ModelEncoding::DeepseekV4 | ModelEncoding::DeepseekV41 => None,
    };
    let assistance = match body.get("tool_decoding_assistance") {
        // glmrt constrained GLM tool calls only when strict or required.
        None | Some(Value::Null) => glm.is_none(),
        Some(Value::Bool(value)) => *value,
        _ => return error(StatusCode::BAD_REQUEST, "tool_decoding_assistance must be boolean"),
    };
    let response_format = body.get("response_format").cloned().filter(|v| !v.is_null());
    // The adapter crate deserializes `response_format.json_schema` into a
    // fieldless "accepted and ignored" variant, so the schema is only available
    // in the raw body. Enforce the OpenAI strict-mode subset here, before any
    // backend admission and independently of the thinking mode, reusing the
    // same validator as the OpenAI-compat `validate_request` path.
    if let Some(format) = response_format.as_ref() {
        if format.get("type").and_then(Value::as_str) == Some("json_schema") {
            if let Some(definition) = format.get("json_schema") {
                let strict = match definition.get("strict") {
                    None | Some(Value::Null) => false,
                    Some(Value::Bool(strict)) => *strict,
                    _ => return error(
                        StatusCode::BAD_REQUEST,
                        "response_format.json_schema.strict must be boolean",
                    ),
                };
                if strict {
                    if let Some(schema) = definition.get("schema") {
                        if let Err(rejection) = crate::schema::validate_strict_json_schema(
                            schema,
                            "response_format.json_schema.schema",
                        ) {
                            return rejection.into_response();
                        }
                    }
                }
            }
        }
    }
    // The recipe rejects its regex variant, while native XGrammar supports it.
    // Keep the original format for enforcement and render it as ordinary text.
    if response_format.as_ref().and_then(|v| v.get("type")).and_then(Value::as_str) == Some("regex") {
        body["response_format"] = json!({"type":"text"});
    }
    let sampling = match request_target_sampling(&body) {
        Ok(sampling) => sampling,
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    // V4.1's checkpoint encoder also takes a numeric budget (1-100), which the
    // adapter's enum cannot parse: take it out here and apply it after rendering.
    let v41_budget = match v41_numeric_effort(&mut body, &state.profile.encoding) {
        Ok(budget) => budget,
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    // The adapter's `seed` field is `u64`; cuteafd keeps the signed convention,
    // so drop it after resolution to keep a negative seed from failing serde.
    if let Some(object) = body.as_object_mut() {
        object.remove("seed");
    }
    // GLM renders the checkpoint template from the request itself; the
    // adapter below still validates it and owns tools and sampling options.
    let glm_request = match glm.map(|encoding| (encoding.thinking(&body), encoding)) {
        None => None,
        Some((Ok(thinking), encoding)) => Some((encoding, body.clone(), thinking)),
        Some((Err(message), _)) => return error(StatusCode::BAD_REQUEST, message),
    };
    // Clients written for vLLM/SGLang send `enable_thinking` (top level or in
    // `chat_template_kwargs`) instead of `thinking.type`; honour it for every
    // template when `thinking` itself is absent.
    let enable_thinking = match (body.get("thinking").filter(|v| !v.is_null()), glm_request.is_none()) {
        (None, true) => match glm5::resolve_thinking(&body) {
            Ok(thinking) if body.get("enable_thinking").filter(|v| !v.is_null()).is_some()
                || body.get("chat_template_kwargs").and_then(|v| v.get("enable_thinking")).filter(|v| !v.is_null()).is_some()
                => Some(thinking),
            Ok(_) => None,
            Err(message) => return error(StatusCode::BAD_REQUEST, message),
        },
        _ => None,
    };
    let mut parsed: ChatCompletionRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let include_usage = parsed.include_usage();
    let parallel = parsed.parallel_tool_calls.unwrap_or(true);
    let selection = tools::Selection::extract(&mut parsed);
    // Native serving defaults to thinking at the adapter's high effort. Explicit
    // thinking/effort settings retain the official conversion precedence.
    let mut converted = match parsed.convert(ConversionOptions::default().with_default_thinking_mode(true)) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    if let Err(e) = selection.apply(&mut converted.conversation.tools) {
        return error(StatusCode::BAD_REQUEST, e);
    }
    if let Some((_, _, thinking)) = &glm_request {
        converted.conversation.thinking_mode = thinking.enabled;
    }
    if let Some(thinking) = enable_thinking {
        converted.conversation.thinking_mode = thinking;
    }
    let model = state.profile.id.clone();
    if converted.model.as_deref() != Some(model.as_str()) {
        return error(StatusCode::BAD_REQUEST, format!("model must be {model}"));
    }
    let max_tokens = match state.limits.requested_output(converted.inference_options.max_tokens) {
        Ok(limit) => limit,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let response_validator = match constraints::response_validator(response_format.as_ref()) {
        Ok(validator) => validator,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let syntax = match &glm_request {
        Some((Templated::Glm(_), _, _)) => tools::ToolSyntax::GlmXml,
        Some((Templated::Qwen(_), _, _)) => tools::ToolSyntax::QwenXml,
        None => tools::ToolSyntax::Dsml,
    };
    let tool_constraints = match tools::ToolConstraints::new(&converted.conversation.tools,
        converted.conversation.tool_choice, selection.required, parallel,
        assistance || response_format.as_ref().is_some_and(|v| v["type"] != "text"), syntax) {
        Ok(tools) => tools,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let constraint = match constraints::response_constraint(response_format.clone(), converted.conversation.thinking_mode,
        tool_constraints.as_ref(), state.profile.encoding.think_close_token_id()) {
        Ok(constraint) => constraint,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let mut validator = tools::CompletionValidator::new(response_validator, tool_constraints);
    let tools_declared = !converted.conversation.tools.is_empty();
    let stop_token_ids = state.profile.stop_token_ids(tools_declared);
    let streaming = converted.stream;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let generator = ChatCompletionRequest::chunk_generator(&converted, id.clone(), model.clone())
        .with_include_usage(!streaming || include_usage);
    let expanded_media_probe = probe.as_ref().is_some_and(|p| !p.spec.media.is_empty());
    let (prompt, image_sources, processor) = match glm_request {
        Some((Templated::Glm(encoding), raw, thinking)) => {
            let tool_choice = match (selection.name(), selection.required) {
                (Some(name), _) => glm5::GlmToolChoice::Named(name.to_owned()),
                (None, true) => glm5::GlmToolChoice::Required,
                (None, false) => glm5::GlmToolChoice::Auto,
            };
            let options = glm5::GlmPromptOptions { thinking: thinking.enabled, reasoning_effort: thinking.effort,
                tool_names: converted.conversation.tools.iter().map(|tool| tool.name.clone()).collect(),
                tool_choice, response_format };
            // Native probe ids already contain image rows, irrespective of chat template.
            let prompt = if expanded_media_probe { String::new() } else {
                match encoding.render(&raw, &options) {
                    Ok(prompt) => prompt,
                    Err(message) => return error(StatusCode::BAD_REQUEST, message),
                }
            };
            let parser = glm5::GlmOutputParser::new(glm5::GlmParserOptions { thinking: thinking.enabled,
                tools: tools_declared.then(|| converted.conversation.tools.clone()),
                stop_sequences: converted.parsing_options.stop_sequences.clone(), id: id.clone() });
            (prompt, Vec::new(), OutputProcessor::Glm(glm5::GlmStreamProcessor::new(generator, parser)))
        }
        Some((Templated::Qwen(encoding), raw, thinking)) => {
            let tool_choice = match (selection.name(), selection.required) {
                (Some(name), _) => qwen4::prompt::QwenToolChoice::Named(name.to_owned()),
                (None, true) => qwen4::prompt::QwenToolChoice::Required,
                (None, false) => qwen4::prompt::QwenToolChoice::Auto,
            };
            let options = qwen4::QwenPromptOptions { thinking: thinking.enabled,
                tool_names: converted.conversation.tools.iter().map(|tool| tool.name.clone()).collect(),
                tool_choice, response_format };
            // Native probe ids already contain image rows, irrespective of chat template.
            let prompt = if expanded_media_probe { String::new() } else {
                match encoding.render(&raw, &options) {
                    Ok(prompt) => prompt,
                    Err(message) => return error(StatusCode::BAD_REQUEST, message),
                }
            };
            let parser = qwen4::QwenOutputParser::new(qwen4::QwenParserOptions { thinking: thinking.enabled,
                tools: tools_declared.then(|| converted.conversation.tools.clone()),
                stop_sequences: converted.parsing_options.stop_sequences.clone() });
            (prompt, Vec::new(), OutputProcessor::Qwen(glm5::GlmStreamProcessor::new(generator, parser)))
        }
        None => {
            if expanded_media_probe {
                // Supplied native ids already contain image rows; do not render another prompt.
                (String::new(), Vec::new(), OutputProcessor::Deepseek(StreamProcessor::new(generator, converted.parsing_options)))
            } else {
            let rendered = if matches!(state.profile.encoding, ModelEncoding::DeepseekV4) {
                DeepseekV4Encoding::new().render_conversation(&converted.conversation)
            } else {
                let mut rendered = DeepseekV41Encoding::new().render_conversation(&converted.conversation);
                if converted.conversation.thinking_mode {
                    let budget = v41_budget.unwrap_or_else(|| v41_effort_budget(converted.conversation.reasoning_effort));
                    rendered.prompt = v41_set_effort_budget(rendered.prompt, budget);
                }
                rendered
            };
            (rendered.prompt, rendered.image_sources,
                OutputProcessor::Deepseek(StreamProcessor::new(generator, converted.parsing_options)))
            }
        }
    };
    // Rendered sources own the image payloads needed by preprocessing. Do not
    // retain another copy of their data URLs throughout the generated response.
    drop(converted.conversation);
    if image_sources.len() > cuteafd_loader::V41_MAX_IMAGES {
        return error(StatusCode::BAD_REQUEST, "at most 16 images are supported");
    }
    let permit = match state.admission.reserve(state.queue.clone()).await {
        Ok(permit) => permit,
        Err(admission::Rejected::Closed) => return error(StatusCode::SERVICE_UNAVAILABLE, "worker queue is closed"),
        Err(admission::Rejected::Overloaded) => {
            let mut response = error(StatusCode::TOO_MANY_REQUESTS, "request queue is full or its wait budget expired");
            response.headers_mut().insert(axum::http::header::RETRY_AFTER, axum::http::HeaderValue::from_static("1"));
            return response;
        }
    };
    let prepared = if image_sources.is_empty() { Vec::new() } else {
        // The queue permit bounds waiters while up to four decoders run. A C16
        // burst should wait here instead of imposing a hidden C4 image limit.
        let slot = match state.images.slots.clone().acquire_owned().await {
            Ok(slot) => slot,
            Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "image preparation is closed"),
        };
        let decoder = state.images.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _slot = slot;
            decoder.decode(image_sources)
        }).await;
        match result {
            Ok(Ok(images)) => images,
            Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, format!("{e:#}")),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    };
    let media = if media_sources.is_empty() { Vec::new() } else {
        let preparer = state.profile.media_preparer.as_ref().expect("sources require preparer").clone();
        let slot = match preparer.slots.clone().acquire_owned().await {
            Ok(slot) => slot,
            Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "image preparation is closed"),
        };
        let hashes = probe.as_ref().map(|p| p.spec.media.iter()
            .map(|span| span.fixture.as_ref().map(|f| f.sha256.clone())).collect::<Vec<_>>()).unwrap_or_default();
        match tokio::task::spawn_blocking(move || {
            let _slot = slot;
            preparer.prepare_verified(&media_sources, &hashes)
        }).await {
            Ok(Ok(prepared)) => prepared.images,
            Ok(Err(message)) => return error(StatusCode::BAD_REQUEST, message),
            Err(message) => return error(StatusCode::INTERNAL_SERVER_ERROR, message),
        }
    };
    let image_tokens: usize = media.iter().map(|image| image.tokens).sum();
    // Unbounded on purpose: inference threads send without ever blocking, so a
    // client that stops reading cannot stall the shared scheduler. A request's
    // backlog is bounded by its own max_tokens.
    let (events, mut receive) = mpsc::unbounded_channel();
    // Recipe 0.1.0 uses a protocol placeholder; the pinned model tokenizer
    // spells token 129264 differently. Preserve the text-only prompt verbatim.
    let prompt = if prepared.is_empty() { prompt }
        else { prompt.replace("<｜image｜>", "<｜deepseek_image｜>") };
    let job = NativeRequest {
        media,
        prompt,
        constraint,
        images: prepared,
        max_tokens,
        sampling,
        stop_token_ids,
        events,
        probe,
    };
    permit.send(job);
    // Admission errors must retain their cause and HTTP status, including for
    // SSE, before a protocol processor can turn early EOF into a finish chunk.
    let first = match receive.recv().await {
        Some(Ok(chunk)) => chunk,
        Some(Err(NativeFailure::BadRequest(message))) => return error(StatusCode::BAD_REQUEST, message),
        Some(Err(NativeFailure::Unavailable(message))) => {
            let mut response = error(StatusCode::SERVICE_UNAVAILABLE, message);
            response.headers_mut().insert(axum::http::header::RETRY_AFTER, axum::http::HeaderValue::from_static("1"));
            return response;
        }
        Some(Err(message)) => return error(StatusCode::INTERNAL_SERVER_ERROR, message),
        None => return error(StatusCode::INTERNAL_SERVER_ERROR, "native worker ended without completion"),
    };
    // Never let a failed/disconnected backend be converted to a successful EOF.
    let failure = Arc::new(Mutex::new(None::<String>));
    let input_failure = failure.clone();
    let input = async_stream::stream! {
        let mut finished = matches!(first, InferenceChunk::Finish { .. });
        yield first;
        while !finished {
            let Some(event) = receive.recv().await else { break; };
            match event {
                Ok(chunk) => {
                    finished = matches!(chunk,InferenceChunk::Finish { .. });
                    yield chunk;
                    if finished { break; }
                }
                Err(message) => { *input_failure.lock().unwrap() = Some(message.to_string()); break; }
            }
        }
        if !finished {
            input_failure.lock().unwrap().get_or_insert_with(|| "native worker ended without completion".into());
        }
    };
    let chunks = processor.process(input);
    let chunks = async_stream::stream! {
        futures::pin_mut!(chunks);
        while let Some(chunk) = chunks.next().await {
            match chunk {
                Ok(chunk) => {
                    if validator.enabled() {
                        if let Err(e) = validator.observe(&serde_json::to_value(&chunk).unwrap()) {
                            yield Err(e); return;
                        }
                    }
                    yield Ok(chunk);
                }
                Err(e) => { yield Err(anyhow::anyhow!(e.to_string())); return; }
            }
        }
    };
    if streaming {
        let stream = async_stream::stream! {
            futures::pin_mut!(chunks);
            let mut usage_chunk = None;
            loop {
                // A comment line keeps proxies and clients from timing out while
                // nothing is emitted: a long prefill, or a tool call the parser
                // holds back until it is complete.
                let chunk = match tokio::time::timeout(SSE_KEEPALIVE, chunks.next()).await {
                    Ok(Some(chunk)) => chunk,
                    Ok(None) => break,
                    Err(_) => { yield Ok::<String, std::convert::Infallible>(": keepalive\n\n".to_owned()); continue; }
                };
                let failed = failure.lock().unwrap().clone();
                if let Some(message) = failed {
                    yield Ok(sse_error(message)); return;
                }
                match chunk {
                    Ok(chunk) => {
                        let mut value = serde_json::to_value(&chunk).unwrap();
                        add_image_usage(&mut value, image_tokens);
                        if include_usage {
                            let taken = value.get_mut("usage").map(Value::take);
                            if let Some(usage) = taken.filter(Value::is_object) {
                                // Spec-shaped clients discard usage on chunks with choices.
                                usage_chunk = Some(json!({
                                    "id": value["id"], "object": value["object"],
                                    "created": value["created"], "model": value["model"],
                                    "system_fingerprint": value["system_fingerprint"],
                                    "choices": [], "usage": usage,
                                }));
                            }
                        }
                        yield Ok(format!("data: {}\n\n", serde_json::to_string(&value).unwrap()));
                    },
                    Err(e) => { yield Ok(sse_error(e)); return; }
                }
            }
            let failed = failure.lock().unwrap().clone();
            if let Some(message) = failed { yield Ok(sse_error(message)); return; }
            if let Some(usage) = usage_chunk {
                yield Ok(format!("data: {}\n\n", serde_json::to_string(&usage).unwrap()));
            }
            yield Ok("data: [DONE]\n\n".to_owned());
        };
        return (
            [
                ("content-type", "text/event-stream"),
                ("cache-control", "no-cache"),
            ],
            Body::from_stream(stream),
        )
            .into_response();
    }
    type ChatResponse = <ChatCompletionRequest as ProtocolRequest>::Response;
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut response = ChatResponse::new(id, model, created, 0, 0);
    futures::pin_mut!(chunks);
    while let Some(chunk) = chunks.next().await {
        match chunk {
            Ok(chunk) => response.append(chunk),
            Err(e) => {
                let message = failure.lock().unwrap().clone().unwrap_or_else(|| e.to_string());
                return error(StatusCode::INTERNAL_SERVER_ERROR, message);
            }
        }
    }
    if let Some(message) = failure.lock().unwrap().clone() {
        return error(StatusCode::INTERNAL_SERVER_ERROR, message);
    }
    if image_tokens == 0 { return Json(response).into_response(); }
    let mut value = serde_json::to_value(response).expect("chat response serializes");
    add_image_usage(&mut value, image_tokens);
    Json(value).into_response()
}

fn add_image_usage(response: &mut Value, image_tokens: usize) {
    if image_tokens == 0 { return; }
    if let Some(usage) = response.get_mut("usage").and_then(Value::as_object_mut) {
        let details = usage.entry("prompt_tokens_details").or_insert_with(|| json!({}));
        if let Some(details) = details.as_object_mut() { details.insert("image_tokens".into(), json!(image_tokens)); }
    }
}

/// V4.1 reasoning-effort budgets from the checkpoint's own encoder
/// (`encoding/encoding.py`: low 25, high 50, xhigh 75, max 100, default high).
/// The recipe crate renders 50/75/75/100 with 75 by default.
fn v41_effort_budget(effort: Option<ReasoningEffort>) -> u8 {
    match effort {
        Some(ReasoningEffort::Low) => 25,
        None | Some(ReasoningEffort::High) => 50,
        Some(ReasoningEffort::Xhigh) => 75,
        Some(ReasoningEffort::Max) => 100,
    }
}

/// Removes an integer `reasoning_effort` (top level or `chat_template_kwargs`)
/// for V4.1 and returns it; a named level stays for the adapter.
fn v41_numeric_effort(body: &mut Value, encoding: &ModelEncoding) -> Result<Option<u8>, String> {
    if !matches!(encoding, ModelEncoding::DeepseekV41) {
        return Ok(None);
    }
    let mut budget = None;
    let top = body.get("reasoning_effort").cloned();
    let kwargs = body.get("chat_template_kwargs").and_then(|v| v.get("reasoning_effort")).cloned();
    for (value, nested) in [(top, false), (kwargs, true)] {
        let Some(Value::Number(number)) = value else { continue };
        let Some(number) = number.as_u64().filter(|n| (1..=100).contains(n)) else {
            return Err("reasoning_effort must be low, high, xhigh, max, none or an integer 1-100".into());
        };
        budget.get_or_insert(number as u8);
        if nested {
            if let Some(kwargs) = body.get_mut("chat_template_kwargs").and_then(Value::as_object_mut) {
                kwargs.remove("reasoning_effort");
            }
        } else if let Some(object) = body.as_object_mut() {
            object.remove("reasoning_effort");
            // A numeric budget asks for thinking, like a named level does.
            if object.get("thinking").map_or(true, Value::is_null) {
                object.insert("thinking".into(), json!({"type": "enabled"}));
            }
        }
    }
    Ok(budget)
}

/// Rewrites the budget in the rendered `Reasoning Effort: N (range 1-100` prefix.
fn v41_set_effort_budget(prompt: String, budget: u8) -> String {
    const PREFIX: &str = "Reasoning Effort: ";
    let Some(start) = prompt.find(PREFIX).map(|i| i + PREFIX.len()) else { return prompt };
    let digits = prompt[start..].bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 || !prompt[start + digits..].starts_with(" (range 1-100") {
        return prompt;
    }
    format!("{}{budget}{}", &prompt[..start], &prompt[start + digits..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;
    fn request(stream: bool) -> axum::http::Request<Body> {
        axum::http::Request::post("/v1/chat/completions").header("content-type","application/json").body(Body::from(json!({"model":MODEL,"messages":[{"role":"user","content":"What is 2 + 2? Answer with just the number."}],"thinking":{"type":"disabled"},"temperature":0,"max_tokens":16,"stream":stream}).to_string())).unwrap()
    }
    #[tokio::test]
    async fn stats_read_live_tables_before_first_publish_and_after_idle_batch() {
        use cuteafd_loader::{MappedTable, RowFormat, TableBackend, TablePart};
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(4096).unwrap();
        // SAFETY: the test owns the file and keeps its length fixed while mapped.
        let table = unsafe { MappedTable::open(&[TablePart { path: file.path().into(),
            offset: 0, rows: 4 }], RowFormat::of(cuteafd_core::DType::U8, 16).unwrap()).unwrap() };
        table.select_backend(TableBackend::Mmap);
        let name = format!("stats-live-{}", file.path().display());
        table.name_stats(name.clone());
        for profile in [ModelProfile::default(),
            ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())))] {
            let (tx, _rx) = mpsc::channel(4);
            let cached = Arc::new(Mutex::new(Value::Null));
            let app = router_for_model(tx, NativeLimits::default(), cached.clone(),
                std::time::Duration::from_secs(25), ConsoleHub::disabled(), profile);
            for expected in [table.stats().snapshot().gathers, table.stats().snapshot().gathers + 1] {
                let response = app.clone().oneshot(axum::http::Request::get("/v1/stats")
                    .body(Body::empty()).unwrap()).await.unwrap();
                let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
                let value: Value = serde_json::from_slice(&body).unwrap();
                let entry = value["mapped_tables"].as_array().unwrap().iter()
                    .find(|entry| entry["name"] == name).unwrap();
                assert_eq!(entry["backend"], "mmap");
                assert_eq!(entry["cumulative"]["gathers"], expected);
                assert_eq!(value["http_queue_len"], 0);
                if cached.lock().unwrap().is_null() {
                    assert_eq!(entry["cumulative"]["rows"], expected * 2);
                } else {
                    assert_eq!(value["family_marker"], "unchanged");
                    assert_eq!(entry["interval"]["gathers"], 77);
                    assert_eq!(entry["host_wide_device_reads"][0]["bytes"], 123);
                }
                // No scheduler publication is necessary after recording a batch.
                table.stats().record_gather(2, 32, std::time::Duration::from_micros(3), [0; 3]);
                *cached.lock().unwrap() = json!({"family_marker":"unchanged",
                    "mapped_tables":[{"name":name,"backend":"stale","cumulative":{"gathers":0},
                        "interval":{"gathers":77},"host_wide_device_reads":[{"device":"test","bytes":123}]}]});
            }
        }
    }

    #[tokio::test]
    async fn stats_without_tables_preserve_published_family_fields() {
        let published = json!({"family":"qwen4", "active":0, "nested":{"tokens":12}});
        let (queue, _receive) = mpsc::channel(4);
        let state = NativeState { queue, limits: NativeLimits::default(),
            images: images::ImageDecoder::new(4), stats: Arc::new(Mutex::new(published.clone())),
            tables: cuteafd_loader::MappedTableStatsReader::default(),
            admission: admission::Admission::new(4, std::time::Duration::from_secs(25)),
            profile: Arc::new(ModelProfile::default()) };
        let Json(value) = stats_route(State(state)).await;
        for (key, expected) in published.as_object().unwrap() { assert_eq!(&value[key], expected); }
        assert_eq!(value["http_queue_len"], 0);
        assert!(value.get("mapped_tables").is_none());
    }

    #[tokio::test]
    async fn streaming_include_usage_emits_spec_shaped_usage_chunk() {
        let profiles = [ModelProfile::default(),
            ModelProfile::new("test-glm", ModelEncoding::Glm(Arc::new(glm5::fixtures::encoding()))),
            ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())))];
        for profile in profiles {
            for include_usage in [true, false] {
                let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
                let worker = tokio::spawn(async move {
                    let job = rx.recv().await.unwrap();
                    for event in [
                        InferenceChunk::Ready { system_fingerprint: Some("fp-test".into()),
                            prompt_usage: PromptUsage { prompt_tokens: 18, prompt_cache_hit_tokens: 7 } },
                        InferenceChunk::Text { content: "4".into(), content_tokens: 1 },
                        InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop },
                    ] { job.events.send(Ok(event)).unwrap(); }
                });
                let body = json!({"model": profile.id, "messages": [{"role": "user", "content": "2+2?"}],
                    "thinking": {"type": "disabled"}, "max_tokens": 16, "stream": true,
                    "stream_options": {"include_usage": include_usage}});
                let app = router_for_model(tx, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
                    std::time::Duration::from_secs(25), ConsoleHub::disabled(), profile.clone());
                let response = app.oneshot(axum::http::Request::post("/v1/chat/completions")
                    .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
                assert_eq!(response.status(), StatusCode::OK, "{}", profile.id);
                let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
                let text = std::str::from_utf8(&bytes).unwrap();
                assert!(text.ends_with("data: [DONE]\n\n"));
                let events: Vec<Value> = text.split("data: ")
                    .filter_map(|s| s.trim_end().parse::<Value>().ok()).collect();
                let finish = events.iter().find(|e| e["choices"][0]["finish_reason"] == "stop").unwrap();
                let usage_chunks: Vec<_> = events.iter().filter(|e| e["choices"] == json!([])).collect();
                if include_usage {
                    assert!(finish["usage"].is_null());
                    assert_eq!(usage_chunks.len(), 1);
                    assert_eq!(events.last().unwrap(), usage_chunks[0]);
                    let usage = &usage_chunks[0]["usage"];
                    assert_eq!(usage["prompt_tokens"], 18);
                    assert_eq!(usage["completion_tokens"], 1);
                    assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 7);
                    assert_eq!(usage["prompt_cache_hit_tokens"], 7);
                } else {
                    assert!(usage_chunks.is_empty());
                    assert_eq!(finish["usage"]["prompt_tokens_details"]["cached_tokens"], 7);
                }
                worker.await.unwrap();
            }
        }
    }

    pub(super) fn terminal_sse_error(bytes: &[u8]) -> Value {
        let text = std::str::from_utf8(bytes).unwrap();
        assert!(!text.split("\n\n").any(|frame| frame == "data: [DONE]"),
            "failed streams must not finish successfully: {text}");
        let frames: Vec<Value> = text.split("\n\n")
            .filter_map(|frame| frame.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).unwrap()).collect();
        assert!(!frames.is_empty(), "failed streams must include an error frame: {text}");
        assert_eq!(frames.iter().filter(|frame| frame.get("error").is_some()).count(), 1);
        assert!(!frames.iter().any(|frame| frame["choices"].as_array().is_some_and(|choices|
            choices.iter().any(|choice| !choice["finish_reason"].is_null()))),
            "failed streams must not emit a success finish chunk: {text}");
        let last = frames.last().unwrap();
        assert_eq!(last["error"]["type"], "native_v41_error");
        last.clone()
    }
    fn strict_schema_body(thinking_disabled: bool, schema: Value) -> Body {
        let mut body = json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Return x."}],
            "temperature": 0,
            "max_tokens": 16,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "strict_probe", "strict": true, "schema": schema}}
        });
        if thinking_disabled {
            body["thinking"] = json!({"type": "disabled"});
        }
        Body::from(body.to_string())
    }

    #[tokio::test]
    async fn strict_schema_subset_is_rejected_before_admission_in_every_thinking_mode() {
        // `strict: true` without `required` is not a valid OpenAI strict schema.
        let invalid = json!({"type": "object", "properties": {"x": {"type": "string"}},
            "additionalProperties": false});
        for thinking_disabled in [false, true] {
            let (tx, _rx) = mpsc::channel::<NativeRequest>(1);
            let request = axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(strict_schema_body(thinking_disabled, invalid.clone()))
                .unwrap();
            let response = router(tx).oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST,
                "thinking_disabled={thinking_disabled} must still reject");
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["error"]["type"], "invalid_request_error");
            assert_eq!(value["error"]["param"], "response_format.json_schema.schema");
        }
    }

    /// The rendered DeepSeek prompt for one request body (the worker side
    /// answers with an immediate stop).
    async fn rendered_prompt(extra: Value) -> String {
        let mut body = json!({"model": MODEL, "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0, "max_tokens": 4});
        for (key, value) in extra.as_object().unwrap() {
            body[key] = value.clone();
        }
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let worker = tokio::spawn(async move {
            let job = rx.recv().await.unwrap();
            job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
            job.events.send(Ok(InferenceChunk::Finish {
                finish_reason: InferenceFinishReason::Stop })).unwrap();
            job.prompt
        });
        let request = axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
        let response = router(tx).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        worker.await.unwrap()
    }

    #[tokio::test]
    async fn remote_vision_failure_rejects_images_but_keeps_text_and_live_health() {
        use cuteafd_loader::media::{EncoderId, ImageFamily, ProcessorConfig};
        use std::sync::atomic::AtomicBool;
        let preparer = Arc::new(media::MediaPreparer::new(ProcessorConfig::for_family(ImageFamily::Mimo),
            EncoderId([1; 32]), media::ImageUrlFetch::Off, 1).unwrap());
        let healthy = Arc::new(AtomicBool::new(true));
        let mut profile = ModelProfile::new(MODEL, ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())))
            .with_loaded_vision(preparer);
        profile.vision_health = Some(healthy.clone());
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let app = router_for_model(tx, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
            std::time::Duration::from_secs(1), ConsoleHub::disabled(), profile);
        let response = app.clone().oneshot(axum::http::Request::get("/health").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        healthy.store(false, Ordering::Release);
        let response = app.clone().oneshot(axum::http::Request::get("/health").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap()["vision"], "failed");
        for stream in [false, true] {
            let body = json!({"model": MODEL, "messages":[{"role":"user","content":[{"type":"image_url",
                "image_url":{"url":"data:image/png;base64,not-decoded"}}]}],"stream":stream});
            let response = app.clone().oneshot(axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            let bytes = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
            assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap()["error"]["message"], "vision encoder unavailable");
            assert!(rx.try_recv().is_err(), "failed vision never reaches admission");
        }
        let worker = tokio::spawn(async move {
            let job = rx.recv().await.unwrap();
            assert!(job.media.is_empty());
            job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
            job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop })).unwrap();
        });
        let body = json!({"model": MODEL, "messages":[{"role":"user","content":"hello"}],"max_tokens":1});
        let response = app.oneshot(axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn encoder_failure_during_admission_keeps_503_for_streams() {
        for stream in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                job.events.send(Err(NativeFailure::Unavailable("vision encoder unavailable".into()))).unwrap();
            });
            let body = json!({"model": MODEL, "messages":[{"role":"user","content":"hello"}],"stream":stream});
            let response = router(tx).oneshot(axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            worker.await.unwrap();
        }
    }

    #[tokio::test]
    async fn media_admission_permanent_400_and_pressure_503_retry_after() {
        for stream in [false, true] {
            for permanent in [false, true] {
                let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
                let message = if permanent { "image needs 5 bytes > media cache capacity 4" }
                    else { "embedding cache budget exhausted: need 4 bytes, 0 free of 8" };
                let worker = tokio::spawn(async move {
                    let job = rx.recv().await.unwrap();
                    let failure = if permanent { NativeFailure::BadRequest(message.into()) }
                        else { NativeFailure::Unavailable(message.into()) };
                    job.events.send(Err(failure)).unwrap();
                });
                let body = json!({"model": MODEL, "messages":[{"role":"user","content":"hello"}],"stream":stream});
                let response = router(tx).oneshot(axum::http::Request::post("/v1/chat/completions")
                    .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
                assert_eq!(response.status(), if permanent { StatusCode::BAD_REQUEST } else { StatusCode::SERVICE_UNAVAILABLE });
                if permanent { assert!(!response.headers().contains_key(axum::http::header::RETRY_AFTER)); }
                else { assert_eq!(response.headers()[axum::http::header::RETRY_AFTER], "1"); }
                let bytes = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
                assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap()["error"]["message"], message);
                worker.await.unwrap();
            }
        }
    }

    #[tokio::test]
    async fn expanded_media_probe_prepares_sources_without_chat_rendering() {
        use base64::Engine;
        use cuteafd_loader::media::{EncoderId, ImageFamily, ProcessorConfig};
        let preparer = Arc::new(media::MediaPreparer::new(ProcessorConfig::for_family(ImageFamily::Mimo),
            EncoderId([1; 32]), media::ImageUrlFetch::Off, 1).unwrap());
        let bytes = include_bytes!("openai/fixtures/black.png");
        let url = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes));
        let image = preparer.prepare(&[media::MediaSource { url: url.clone(), low: false }]).unwrap().images.remove(0);
        let key: String = image.key.0.iter().map(|v| format!("{v:02x}")).collect();
        let spec = probe::ProbeSpec { prompt_ids: Some(vec![1; image.tokens + 2]),
            media: vec![probe::ProbeMedia { start: 1, len: image.tokens, kind: "image".into(), key,
                grid: [image.grid.t, image.grid.h, image.grid.w], fixture: None,
                image_url: Some(probe::ProbeImageUrl { url, detail: None }) }], ..Default::default() };
        for encoding in [ModelEncoding::DeepseekV4,
            ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())),
            ModelEncoding::Glm(Arc::new(glm5::fixtures::encoding()))] {
            let image = image.clone();
            let (id, _) = probe::registry().register(spec.clone());
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                assert!(job.prompt.is_empty() && job.images.is_empty());
                assert_eq!(job.media.len(), 1); assert_eq!(job.media[0].key, image.key);
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
                job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length })).unwrap();
            });
            let profile = ModelProfile::new(MODEL, encoding).with_loaded_vision(preparer.clone());
            let app = router_for_model(tx, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
                std::time::Duration::from_secs(1), ConsoleHub::disabled(), profile);
            let body = json!({"model":MODEL,"messages":[{"role":"user","content":"probe"}],"max_tokens":1});
            let response = app.oneshot(axum::http::Request::post("/v1/chat/completions").header(probe::HEADER, id)
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
            let status = response.status();
            if status != StatusCode::OK {
                let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
                worker.abort();
                panic!("expanded media rejected: {status}: {}", String::from_utf8_lossy(&bytes));
            }
            worker.await.unwrap();
        }
    }

    #[tokio::test]
    #[ignore = "requires CUTEAFD_TEST_MEDIA_REQUEST and CUTEAFD_TEST_MEDIA_SNAPSHOT"]
    async fn replay_native_media_probe_request_without_gpu() {
        use cuteafd_loader::media::{EncoderId, ImageFamily, ProcessorConfig};
        let snapshot = std::path::PathBuf::from(std::env::var_os("CUTEAFD_TEST_MEDIA_SNAPSHOT").unwrap());
        let request: Value = serde_json::from_slice(&std::fs::read(
            std::env::var_os("CUTEAFD_TEST_MEDIA_REQUEST").unwrap()).unwrap()).unwrap();
        let spec: probe::ProbeSpec = serde_json::from_value(request["spec"].clone()).unwrap();
        spec.validate_media().unwrap();
        let expected = spec.clone();
        let (id, _) = probe::registry().register(spec);
        let preparer = Arc::new(media::MediaPreparer::new(ProcessorConfig::from_snapshot(&snapshot,
            ImageFamily::Mimo).unwrap(), EncoderId([1; 32]), media::ImageUrlFetch::Off, 1).unwrap());
        let profile = ModelProfile::new(request["body"]["model"].as_str().unwrap(),
            ModelEncoding::Qwen(Arc::new(qwen4::QwenEncoding::from_snapshot(&snapshot).unwrap())))
            .with_loaded_vision(preparer);
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let worker = tokio::spawn(async move {
            let job = rx.recv().await.unwrap();
            assert!(job.prompt.is_empty() && job.images.is_empty());
            assert_eq!(job.probe.as_ref().unwrap().spec.prompt_ids, expected.prompt_ids);
            assert_eq!(job.media.len(), expected.media.len());
            for (image, span) in job.media.iter().zip(&expected.media) {
                assert_eq!([image.grid.t, image.grid.h, image.grid.w], span.grid);
                assert_eq!(image.tokens, span.len);
            }
            job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
            job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length })).unwrap();
        });
        let app = router_for_model(tx, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
            std::time::Duration::from_secs(1), ConsoleHub::disabled(), profile);
        let response = app.oneshot(axum::http::Request::post("/v1/chat/completions").header(probe::HEADER, id)
            .header("content-type", "application/json").body(Body::from(request["body"].to_string())).unwrap()).await.unwrap();
        let status = response.status();
        if status != StatusCode::OK {
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            worker.abort();
            panic!("replayed media rejected: {status}: {}", String::from_utf8_lossy(&bytes));
        }
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn enable_thinking_selects_the_deepseek_thinking_mode() {
        let official_off = rendered_prompt(json!({"thinking": {"type": "disabled"}})).await;
        let official_on = rendered_prompt(json!({"thinking": {"type": "enabled"}})).await;
        assert_ne!(official_off, official_on);
        assert_eq!(rendered_prompt(json!({"enable_thinking": false})).await, official_off);
        assert_eq!(rendered_prompt(json!({"chat_template_kwargs": {"enable_thinking": false}})).await, official_off);
        assert_eq!(rendered_prompt(json!({"enable_thinking": true})).await, official_on);
        // `thinking` keeps precedence over the template kwarg.
        assert_eq!(rendered_prompt(json!({"thinking": {"type": "enabled"}, "enable_thinking": false})).await, official_on);
    }

    #[tokio::test]
    async fn a_client_that_stops_reading_never_blocks_the_inference_thread() {
        // The worker emits far more events than any bounded channel holds while
        // nobody reads the response body; its sends must all complete.
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let worker = tokio::spawn(async move {
            let job = rx.recv().await.unwrap();
            job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
            for _ in 0..10_000 {
                job.events.send(Ok(InferenceChunk::Text { content: "x".into(), content_tokens: 1 })).unwrap();
            }
            job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length })).unwrap();
        });
        let response = router(tx).oneshot(request(true)).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), worker).await
            .expect("inference sends blocked on an unread stream").unwrap();
        drop(response);
    }

    #[tokio::test]
    async fn silent_streams_send_keepalive_comments() {
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        tokio::spawn(async move {
            let job = rx.recv().await.unwrap();
            job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
            tokio::time::sleep(SSE_KEEPALIVE * 4).await;
            job.events.send(Ok(InferenceChunk::Text { content: "4".into(), content_tokens: 1 })).unwrap();
            job.events.send(Ok(InferenceChunk::Finish {
                finish_reason: InferenceFinishReason::Stop })).unwrap();
        });
        let response = router(tx).oneshot(request(true)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains(": keepalive\n\n"), "{text}");
        assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    }

    #[tokio::test]
    async fn valid_strict_schema_is_not_newly_rejected() {
        let valid = json!({"type": "object", "properties": {"x": {"type": "string"}},
            "required": ["x"], "additionalProperties": false});
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let worker = tokio::spawn(async move {
            let job = rx.recv().await.unwrap();
            assert!(job.constraint.is_some());
            job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
            job.events.send(Ok(InferenceChunk::Text {
                content: "{\"x\":\"a\"}".into(), content_tokens: 5 })).unwrap();
            job.events.send(Ok(InferenceChunk::Finish {
                finish_reason: InferenceFinishReason::Stop })).unwrap();
        });
        let request = axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(strict_schema_body(false, valid)).unwrap();
        let response = router(tx).oneshot(request).await.unwrap();
        assert_ne!(response.status(), StatusCode::BAD_REQUEST);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn output_limits_reach_worker_and_model_metadata() {
        for (limits, requested, expected) in [
            (NativeLimits::default(), None, 393_216),
            (NativeLimits::default(), Some(8192), 8192),
            (NativeLimits::default(), Some(u32::MAX), 393_216),
            (NativeLimits::new(256, 128).unwrap(), None, 128),
            (NativeLimits::new(256, 128).unwrap(), Some(8192), 128),
            (NativeLimits::new(256, 128).unwrap(), Some(8), 8),
        ] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let app = router_with_limits(tx, limits);
            let response = app.clone().oneshot(axum::http::Request::get("/v1/models")
                .body(Body::empty()).unwrap()).await.unwrap();
            let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["data"][0]["max_context_tokens"], limits.context());
            assert_eq!(value["data"][0]["max_output_tokens"], limits.output());
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                assert_eq!(job.max_tokens, expected);
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
                job.events.send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Length,
                })).unwrap();
            });
            let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Count."}],
                "max_tokens":requested});
            let response = app.oneshot(axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            worker.await.unwrap();
        }
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Count."}],"max_tokens":0});
        let response = router(tx).oneshot(axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(rx.try_recv().is_err());
    }
    #[tokio::test]
    async fn thinking_defaults_high_and_honors_explicit_overrides() {
        for (options, score) in [
            (json!({}), Some(50)),
            (json!({"thinking":{"type":"enabled"}}), Some(50)),
            (json!({"reasoning_effort":"low"}), Some(25)),
            (json!({"reasoning_effort":"high"}), Some(50)),
            (json!({"reasoning_effort":"xhigh"}), Some(75)),
            (json!({"reasoning_effort":"max"}), Some(100)),
            (json!({"reasoning_effort":37}), Some(37)),
            (json!({"chat_template_kwargs":{"reasoning_effort":90}}), Some(90)),
            (json!({"reasoning_effort":"none"}), None),
            (json!({"thinking":{"type":"disabled"},"reasoning_effort":"max"}), None),
        ] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                if let Some(score) = score {
                    assert!(job.prompt.contains(&format!("Reasoning Effort: {score} (range 1-100")));
                    assert!(job.prompt.ends_with("<think>"));
                } else {
                    assert!(!job.prompt.contains("Reasoning Effort:"));
                    assert!(job.prompt.ends_with("</think>"));
                }
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
                job.events.send(Ok(InferenceChunk::Text {
                    content: if score.is_some() { "Compute. </think>4" } else { "4" }.into(),
                    content_tokens: 1,
                })).unwrap();
                job.events.send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Stop,
                })).unwrap();
            });
            let mut body = json!({"model":MODEL,"messages":[{"role":"user","content":"2+2?"}],
                "max_tokens":16,"stream":false});
            body.as_object_mut().unwrap().extend(options.as_object().unwrap().clone());
            let request = axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
            let response = router(tx).oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["choices"][0]["message"]["content"], "4");
            if score.is_some() {
                assert_eq!(value["choices"][0]["message"]["reasoning_content"], "Compute. ");
            }
            worker.await.unwrap();
        }
    }
    #[tokio::test]
    async fn official_prompt_and_both_response_modes() {
        for streaming in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                assert_eq!(job.prompt,"<｜begin▁of▁sentence｜><｜User｜>What is 2 + 2? Answer with just the number.<｜Assistant｜></think>");
                assert_eq!(job.max_tokens, 16);
                for event in [
                    InferenceChunk::Ready {
                        system_fingerprint: None,
                        prompt_usage: PromptUsage {
                            prompt_tokens: 18,
                            prompt_cache_hit_tokens: 0,
                        },
                    },
                    InferenceChunk::Text {
                        content: "4".into(),
                        content_tokens: 1,
                    },
                    InferenceChunk::Finish {
                        finish_reason: InferenceFinishReason::Stop,
                    },
                ] {
                    job.events.send(Ok(event)).unwrap();
                }
            });
            let response = router(tx).oneshot(request(streaming)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
            if streaming {
                let text = std::str::from_utf8(&body).unwrap();
                assert!(text.contains("\"content\":\"4\""));
                assert!(text.ends_with("data: [DONE]\n\n"));
            } else {
                let value: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(value["choices"][0]["message"]["content"], "4");
                assert_eq!(value["usage"]["prompt_tokens"], 18);
            }
            worker.await.unwrap();
        }
    }
    #[tokio::test]
    async fn admission_errors_preserve_status_and_cause_before_json_or_sse() {
        for streaming in [false, true] {
            for bad_request in [false, true] {
                let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
                let worker = tokio::spawn(async move {
                    let job = rx.recv().await.unwrap();
                    let message = "required parameter has incompatible value constraints: nx".to_string();
                    job.events.send(Err(if bad_request { NativeFailure::BadRequest(message) }
                        else { NativeFailure::Worker(message) })).unwrap();
                });
                let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Call lookup."}],
                    "tools":[{"type":"function","function":{"name":"lookup","strict":true,
                        "parameters":{"type":"object","properties":{"nx":{"const":1}},"required":["nx"]}}}],
                    "tool_choice":"required","stream":streaming});
                let request = axum::http::Request::post("/v1/chat/completions")
                    .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
                let response = router(tx).oneshot(request).await.unwrap();
                assert_eq!(response.status(), if bad_request { StatusCode::BAD_REQUEST }
                    else { StatusCode::INTERNAL_SERVER_ERROR });
                let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["error"]["message"], "required parameter has incompatible value constraints: nx");
                worker.await.unwrap();
            }
        }
    }
    #[tokio::test]
    async fn late_worker_failure_is_not_replaced_by_required_tool_validation() {
        for streaming in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
                job.events.send(Err(NativeFailure::Worker("late execution failure".into()))).unwrap();
            });
            let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Call lookup."}],
                "tools":[{"type":"function","function":{"name":"lookup","strict":true,
                    "parameters":{"type":"object","properties":{"n":{"const":1}},"required":["n"]}}}],
                "tool_choice":"required","stream":streaming});
            let request = axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
            let response = router(tx).oneshot(request).await.unwrap();
            if streaming {
                assert_eq!(response.status(), StatusCode::OK);
                let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
                assert_eq!(terminal_sse_error(&bytes)["error"]["message"], "late execution failure");
            } else {
                assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["error"]["message"], "late execution failure");
            }
        }
    }
    #[tokio::test]
    async fn late_worker_errors_and_disconnects_end_with_an_error_data_frame() {
        let escaped = "worker said \"failed\" \\\r\n\ndata: [DONE]\n\nevent: injected\0".to_owned();
        let oversized = format!("backend detail: {}END_OF_UNBOUNDED_DETAIL", "界\n\"\\".repeat(5000));
        for message in [None, Some(escaped), Some(oversized)] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let worker_message = message.clone();
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
                job.events.send(Ok(InferenceChunk::Text { content: "Partial output.".into(), content_tokens: 2 })).unwrap();
                if let Some(message) = worker_message {
                    job.events.send(Err(NativeFailure::Worker(message))).unwrap();
                }
                // Dropping the sender without Finish is also a worker failure.
            });
            let response = router(tx).oneshot(request(true)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()["content-type"], "text/event-stream");
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let error = terminal_sse_error(&bytes);
            let detail = error["error"]["message"].as_str().unwrap();
            match message {
                None => assert_eq!(detail, "native worker ended without completion"),
                Some(message) if message.chars().count() <= 512 => assert_eq!(detail, message),
                Some(_) => {
                    assert!(detail.starts_with("backend detail: "));
                    assert!(detail.ends_with("... (truncated)"));
                    assert!(detail.chars().count() < 530);
                    assert!(!detail.contains("END_OF_UNBOUNDED_DETAIL"));
                    assert!(bytes.len() < 8192, "error detail escaped beyond the response bound");
                }
            }
            worker.await.unwrap();
        }
    }
    #[tokio::test]
    async fn a_failed_stream_does_not_abort_a_sibling_stream() {
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(2);
        let worker = tokio::spawn(async move {
            let failed = rx.recv().await.unwrap();
            let healthy = rx.recv().await.unwrap();
            for job in [&failed, &healthy] {
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).unwrap();
            }
            failed.events.send(Err(NativeFailure::Worker("one request failed".into()))).unwrap();
            healthy.events.send(Ok(InferenceChunk::Text { content: "4".into(), content_tokens: 1 })).unwrap();
            healthy.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop })).unwrap();
        });
        let app = router(tx);
        let (failed, healthy) = tokio::join!(app.clone().oneshot(request(true)), app.oneshot(request(true)));
        let failed = axum::body::to_bytes(failed.unwrap().into_body(), 1 << 20);
        let healthy = axum::body::to_bytes(healthy.unwrap().into_body(), 1 << 20);
        let (failed, healthy) = tokio::join!(failed, healthy);
        assert_eq!(terminal_sse_error(&failed.unwrap())["error"]["message"], "one request failed");
        let healthy = String::from_utf8(healthy.unwrap().to_vec()).unwrap();
        assert!(healthy.contains("\"content\":\"4\""));
        assert!(healthy.ends_with("data: [DONE]\n\n"));
        assert!(!healthy.contains("\"error\""));
        worker.await.unwrap();
    }
    #[tokio::test]
    async fn worker_error_and_missing_finish_are_not_success() {
        for explicit_error in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                if explicit_error {
                    job.events
                        .send(Err("execution failed".into()))
                        
                        .unwrap();
                }
            });
            let response = router(tx).oneshot(request(false)).await.unwrap();
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        }
    }
}
