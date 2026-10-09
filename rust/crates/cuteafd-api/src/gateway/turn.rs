//! The protocol-neutral turn model every front end maps onto.
//!
//! A front end (Anthropic Messages, OpenAI Responses, Realtime) parses its
//! wire request into a [`TurnRequest`], hands it to the gateway driver, and
//! renders the [`TurnEvent`] stream back in its own wire format. Backends
//! (the in-process engine, an upstream HTTP API) only ever see this model.
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Who authored a conversation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    /// OpenAI `developer`/`system` messages that appear inside the item list.
    /// The request-level system prompt lives in [`TurnRequest::system`].
    System,
}

/// Image bytes or a reference the backend may fetch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

/// One piece of message content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text { text: String },
    Image { source: ImageSource, #[serde(default, skip_serializing_if = "Option::is_none")] detail: Option<String> },
    /// Base64 audio (`format`: `wav`, `mp3`, `pcm16` at 24 kHz mono, ...).
    Audio { format: String, data: String },
    /// A document/file the backend cannot read natively; front ends may
    /// flatten it to text first.
    File { name: Option<String>, media_type: Option<String>, data: String },
}

impl Part {
    pub fn text(text: impl Into<String>) -> Self { Self::Text { text: text.into() } }
}

/// One entry of the conversation history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "item", rename_all = "snake_case")]
pub enum Item {
    Message { role: Role, content: Vec<Part> },
    /// Model reasoning ("thinking"). `signature` carries the protocol's opaque
    /// round-trip token (Anthropic `signature`, Responses `encrypted_content`).
    Reasoning { text: String, #[serde(default, skip_serializing_if = "Option::is_none")] signature: Option<String> },
    /// A client-executed function call the model made.
    ToolCall { id: String, name: String, arguments: String },
    /// The client's result for a [`Item::ToolCall`].
    ToolResult { call_id: String, content: Vec<Part>, #[serde(default)] is_error: bool },
    /// A gateway-executed (hosted) tool call, e.g. web search, kept in
    /// history so a follow-up turn can see what was searched.
    ServerToolCall { id: String, name: String, input: Value },
    ServerToolResult { call_id: String, name: String, output: Value },
}

/// A client function tool the model may call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema of the arguments object.
    pub parameters: Value,
    #[serde(default)]
    pub strict: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    /// At least one tool call.
    Required,
    Named { name: String },
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
}

/// Reasoning controls, normalized across protocols.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Reasoning {
    /// `None` leaves the backend default.
    pub enabled: Option<bool>,
    /// OpenAI effort (`minimal`/`low`/`medium`/`high`/`xhigh`, ...).
    pub effort: Option<String>,
    /// Anthropic `thinking.budget_tokens`.
    pub budget_tokens: Option<u32>,
    /// Whether the client wants reasoning text back (Responses `summary`,
    /// Anthropic `thinking`); hidden reasoning still counts in usage.
    pub return_text: bool,
}

/// Hosted (gateway-executed) web search requested by the client.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct WebSearchSpec {
    /// The tool name the client knows it by (`web_search`).
    pub name: String,
    pub max_uses: Option<u32>,
    pub allowed_domains: Vec<String>,
    pub blocked_domains: Vec<String>,
    /// Free-form user location hint passed to providers that take one.
    pub user_location: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HostedTools {
    pub web_search: Option<WebSearchSpec>,
}

/// Requested output modalities (Realtime `modalities`/`output_modalities`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Modalities {
    pub audio_out: bool,
}

/// One model turn: everything a backend needs to produce the next assistant
/// message, independent of the wire protocol.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TurnRequest {
    /// The model id the client asked for; echoed back in responses.
    pub requested_model: String,
    /// The served model id after aliasing; what the backend runs.
    pub model: String,
    pub system: Option<String>,
    pub items: Vec<Item>,
    pub tools: Vec<ToolSpec>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    pub max_output_tokens: Option<u32>,
    pub sampling: Sampling,
    pub reasoning: Reasoning,
    /// JSON-schema structured output (`{"name","schema","strict"}`), or
    /// `{"type":"json_object"}`.
    pub response_format: Option<Value>,
    pub hosted: HostedTools,
    pub modalities: Modalities,
    /// The session this turn belongs to, when the front end has one.
    pub session: Option<super::session::SessionId>,
    /// Traffic recording for this turn (front ends copy the request's tape).
    #[serde(skip)]
    pub tape: super::record::Tape,
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
    StopSequence { sequence: Option<String> },
    /// The model refused (Anthropic `refusal`).
    Refusal,
    /// Output was stopped by a content filter (chat `finish_reason:
    /// content_filter`, Responses `incomplete_details.reason: content_filter`).
    ContentFilter,
    /// The client or session cancelled the turn.
    Cancelled,
    /// A long turn paused (Anthropic `pause_turn`, e.g. hosted-tool rounds
    /// exhausted); the client may continue it by resending. Protocols without
    /// an equivalent render it as a normal end of turn.
    PauseTurn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Prompt tokens served from a prefix/prompt cache.
    pub cached_input_tokens: u32,
    /// Output tokens spent on reasoning (included in `output_tokens`).
    pub reasoning_tokens: u32,
    /// Hosted web searches executed by the gateway for this turn.
    pub web_search_requests: u32,
    /// Prompt tokens written to a prompt cache (Anthropic
    /// `cache_creation_input_tokens`); 0 when the backend doesn't report it.
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
}

impl Usage {
    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.web_search_requests += other.web_search_requests;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
    }
}

/// One step of a streamed turn. Tool-call events carry a per-turn `index`
/// that is stable from `ToolCallStart` to `ToolCallEnd`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TurnEvent {
    ReasoningDelta { text: String },
    /// Opaque reasoning round-trip token, if the backend has one.
    ReasoningSignature { signature: String },
    TextDelta { text: String },
    ToolCallStart { index: usize, id: String, name: String },
    ToolCallDelta { index: usize, arguments: String },
    ToolCallEnd { index: usize },
    /// The gateway is executing a hosted tool (web search).
    ServerToolCall { id: String, name: String, input: Value },
    ServerToolResult { id: String, name: String, output: Value },
    /// Cumulative usage for the turn so far; the last one wins.
    Usage { usage: Usage },
    /// Terminal event of a successful turn.
    Done { stop: StopReason },
}
