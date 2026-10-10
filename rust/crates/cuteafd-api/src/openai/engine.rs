//! The gateway's in-process engine backend (PLAN "Phase B: engine backend").
//!
//! A [`TurnRequest`] becomes exactly the Chat Completions body the serving
//! chat route consumes, then runs through the same [`super::build`] /
//! [`super::Submitter::submit`] pipeline: the family's chat template, tool
//! syntax and grammar, thinking and effort, images and audio, stop ids and
//! sampling are the chat path's, not a copy. The chat chunks it streams back
//! become [`TurnEvent`]s.
//!
//! # Item mapping (every family)
//!
//! | `TurnRequest` | chat message the family template renders |
//! |---|---|
//! | `system` | leading `system` message |
//! | `Message{System}` | `system` message in place (templates merge or relocate it as they do for chat) |
//! | `Message{User}` text/image/audio | `user` content parts `text` / `image_url` / `input_audio` |
//! | `Message{Assistant}` | `assistant` `content` |
//! | `Reasoning` | `reasoning_content` of the next assistant message |
//! | `ToolCall` | `tool_calls[]` entry of that assistant message |
//! | `ToolResult` | `tool` message with its `tool_call_id` |
//! | `ServerToolCall` (web search) | `tool_calls[]` entry named after the hosted tool (`web_search`) |
//! | `ServerToolResult` | `tool` message whose content is the result JSON |
//!
//! Consecutive reasoning, text and calls fold into one assistant message, in
//! the order a model emits them (reasoning, text, calls).
//!
//! # Per-family template behaviour (the checkpoint's own template decides)
//!
//! | Family | tool-call syntax | prior-turn reasoning | thinking off |
//! |---|---|---|---|
//! | DeepSeek V4 / V4.1 | DSML `<｜DSML｜ invoke>` | kept while thinking is on; dropped (`</think>` only) when off | `</think>` generation prompt |
//! | GLM 5.x | `<tool_call>name<arg_key>…` | kept (`clear_thinking` false); empty `<think></think>` when absent | server `--thinking-off` form (`low` or empty block) |
//! | Qwen 3.8 | `<tool_call><function=…><parameter=…>` | dropped before the last user query (`preserve_thinking` unset) | `<think>\n\n</think>` |
//! | MiMo V2 (Qwen encoding) | `<tool_call><function=…><parameter=…>` | always kept | `<think></think>` |
//!
//! Hosted web search is a plain `web_search` function to the model; its
//! results return as a `tool` message, so every family renders search rounds
//! with the tool-result syntax it already has.
use std::sync::Arc;

use axum::http::{HeaderMap, StatusCode};
use futures::{future::BoxFuture, StreamExt};
use serde_json::{json, Map, Value};

use super::{ChatChunk, ModelEncoding, ModelProfile, NativeLimits, NativeState, Rejection, Running, Submitter};
use crate::gateway::{
    backend::{Backend, BackendCapabilities, ModelInfo, TurnStream},
    error::{ErrorKind, GatewayError},
    turn::{ImageSource, Item, Part, Role, StopReason, ToolChoice, TurnEvent, TurnRequest, Usage},
};

/// Serving options the gateway engine backend reads at startup.
#[derive(Debug, Clone, Copy, Default)]
pub struct EngineOptions {
    /// Advertise `json_schema` / strict tools. Off until a per-model probe
    /// passed (PLAN phase B decision); set by [`probe_json_schema`] callers.
    pub json_schema: bool,
}

/// Probe hook for the per-model `json_schema` capability: returns whether the
/// served model followed a strict schema. No probe campaign runs yet; serving
/// keeps the capability off until one passes for the checkpoint.
pub fn probe_json_schema(_profile: &ModelProfile) -> bool { false }

/// The serving engine as a gateway [`Backend`].
#[derive(Clone)]
pub struct Engine {
    state: NativeState,
    snapshot: Option<std::path::PathBuf>,
    options: EngineOptions,
    systems: SystemPlacement,
}

/// Where a family's template accepts system messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SystemPlacement {
    /// Anywhere (GLM, MiMo); the DeepSeek recipe normalizes on its own.
    Anywhere,
    /// Only one, first (Qwen 3.8 raises otherwise). Leading system items
    /// merge into it with a blank line and later ones become user turns:
    /// the DeepSeek recipe's own normalization (`normalize_messages`).
    FirstOnly,
}

/// Probe the checkpoint template once: does it render a second system message?
pub(crate) fn system_placement(profile: &ModelProfile) -> SystemPlacement {
    let body = json!({"messages":[{"role":"system","content":"a"},{"role":"user","content":"b"},
        {"role":"system","content":"c"},{"role":"user","content":"d"}]});
    let rejects = match &profile.encoding {
        ModelEncoding::Qwen(encoding) => encoding.render(&body, &super::qwen4::QwenPromptOptions { thinking: false,
            tool_names: Vec::new(), tool_choice: super::qwen4::prompt::QwenToolChoice::Auto, response_format: None }).is_err(),
        ModelEncoding::Glm(encoding) => encoding.render(&body, &super::glm5::GlmPromptOptions { thinking: false,
            reasoning_effort: None, tool_names: Vec::new(), tool_choice: super::glm5::GlmToolChoice::Auto, response_format: None }).is_err(),
        ModelEncoding::DeepseekV4 | ModelEncoding::DeepseekV41 => false,
    };
    if rejects { SystemPlacement::FirstOnly } else { SystemPlacement::Anywhere }
}

impl Engine {
    /// `snapshot` holds the tokenizer used for exact `count_tokens`.
    pub(crate) fn new(state: NativeState, snapshot: Option<std::path::PathBuf>, options: EngineOptions) -> Self {
        let systems = system_placement(&state.profile);
        Self { state, snapshot, options, systems }
    }
    fn profile(&self) -> &ModelProfile { &self.state.profile }
    fn submitter(&self) -> Submitter { self.state.submitter() }
    fn limits(&self) -> NativeLimits { self.state.limits }

    /// The Chat Completions body for `turn`, as `/v1/chat/completions` would receive it.
    pub fn chat_body(&self, turn: &TurnRequest) -> Result<Value, GatewayError> {
        chat_body_with(turn, self.profile(), &ToolNames::new(turn), self.systems)
    }
}

fn rejection(rejection: Rejection) -> GatewayError {
    let kind = match rejection.status {
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        StatusCode::SERVICE_UNAVAILABLE => ErrorKind::Overloaded,
        StatusCode::PAYLOAD_TOO_LARGE => ErrorKind::RequestTooLarge,
        status if status.is_client_error() => ErrorKind::InvalidRequest,
        _ => ErrorKind::Internal,
    };
    GatewayError::new(kind, rejection.message)
}

fn image_url(source: &ImageSource) -> String {
    match source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
    }
}

/// Chat content for `content`: one text part is a plain string, anything
/// else a part list, so each family's chat conversion joins parts by its own
/// rule (DeepSeek's recipe joins text parts with a blank line) exactly as
/// for a chat client sending the same parts.
fn content(content: &[Part], role: &str) -> Result<Value, GatewayError> {
    match content {
        [] => return Ok(Value::String(String::new())),
        [Part::Text { text }] => return Ok(Value::String(text.clone())),
        _ => {}
    }
    let mut parts = Vec::with_capacity(content.len());
    for part in content {
        parts.push(match part {
            Part::Text { text } => json!({"type":"text","text":text}),
            Part::Image { source, detail } => {
                let mut image = json!({"url": image_url(source)});
                if let Some(detail) = detail { image["detail"] = json!(detail); }
                json!({"type":"image_url","image_url":image})
            }
            Part::Audio { format, data } => {
                let (format, data) = if format == "pcm16" { ("wav".to_owned(), pcm16_wav(data)?) } else { (format.clone(), data.clone()) };
                json!({"type":"input_audio","input_audio":{"format":format,"data":data}})
            }
            Part::File { name, .. } => return Err(GatewayError::unsupported(format!(
                "file input{} is not supported in {role} content; send its text", name.as_deref().map(|n| format!(" '{n}'")).unwrap_or_default()))),
        });
    }
    Ok(Value::Array(parts))
}

/// Realtime commits base64 PCM16 mono at 24 kHz; the audio preparer takes WAV.
fn pcm16_wav(data: &str) -> Result<String, GatewayError> {
    use base64::{engine::general_purpose::STANDARD, Engine};
    let pcm = STANDARD.decode(data).map_err(|_| GatewayError::invalid("invalid PCM16 audio base64").with_param("audio"))?;
    if pcm.len() % 2 != 0 { return Err(GatewayError::invalid("PCM16 audio must contain whole samples").with_param("audio")); }
    let len = u32::try_from(pcm.len()).ok().filter(|n| *n <= u32::MAX - 36)
        .ok_or_else(|| GatewayError::invalid("PCM16 audio exceeds WAV size limit"))?;
    let mut wav = Vec::with_capacity(pcm.len() + 44);
    wav.extend_from_slice(b"RIFF"); wav.extend_from_slice(&(len + 36).to_le_bytes()); wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); wav.extend_from_slice(&1u16.to_le_bytes()); wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&24000u32.to_le_bytes()); wav.extend_from_slice(&48000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes()); wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data"); wav.extend_from_slice(&len.to_le_bytes()); wav.extend_from_slice(&pcm);
    Ok(STANDARD.encode(wav))
}

/// For a template that takes one leading system message: leading system
/// messages merge with a blank line, later ones become user turns (Claude
/// Code's mid-conversation `<system-reminder>` messages, Codex's developer
/// items), as the DeepSeek recipe normalizes for its own template.
fn first_system_only(messages: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for mut message in messages {
        if message["role"] != "system" { out.push(message); continue; }
        let leading = out.iter().all(|m| m["role"] == "system");
        match out.last_mut().filter(|_| leading) {
            Some(head) => {
                let mut parts = parts_of(head["content"].take());
                parts.push(json!({"type":"text","text":"\n\n"}));
                parts.extend(parts_of(message["content"].take()));
                head["content"] = joined(parts);
            }
            None if leading => out.push(message),
            None => { message["role"] = json!("user"); out.push(message); }
        }
    }
    out
}

/// Content parts of a chat content value (a string is one text part).
fn parts_of(value: Value) -> Vec<Value> {
    match value {
        Value::String(text) => vec![json!({"type":"text","text":text})],
        Value::Array(parts) => parts,
        _ => Vec::new(),
    }
}

/// Text-only parts as one string (system content is text in every template).
fn joined(parts: Vec<Value>) -> Value {
    if parts.iter().all(|p| p["type"] == "text") {
        Value::String(parts.iter().map(|p| p["text"].as_str().unwrap_or_default()).collect())
    } else { Value::Array(parts) }
}

/// Tool names the chat path accepts: 1-128 ASCII letters, digits, `_`, `-`.
/// Namespaced Responses tools (`functions.exec`) and other names are mapped
/// to a legal, unique wire name per turn and back.
#[derive(Debug, Default)]
pub(crate) struct ToolNames { to_wire: std::collections::HashMap<String, String>, to_client: std::collections::HashMap<String, String> }

fn legal(name: &str) -> bool {
    !name.is_empty() && name.len() <= 128 && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

impl ToolNames {
    fn new(turn: &TurnRequest) -> Self {
        let mut names: Vec<&str> = turn.tools.iter().map(|t| t.name.as_str()).collect();
        for item in &turn.items {
            if let Item::ToolCall { name, .. } | Item::ServerToolCall { name, .. } = item { names.push(name); }
        }
        if let ToolChoice::Named { name } = &turn.tool_choice { names.push(name); }
        let mut out = Self::default();
        let mut taken: std::collections::HashSet<String> = names.iter().filter(|n| legal(n)).map(|n| n.to_string()).collect();
        for name in names {
            if out.to_wire.contains_key(name) { continue; }
            let wire = if legal(name) { name.to_owned() } else {
                // `functions.exec` reads as `exec` to the model when that is free;
                // otherwise `functions__exec`, then a numbered variant.
                let local = name.rsplit('.').next().unwrap_or(name);
                let clean = |s: &str| s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect::<String>();
                let mut candidates = vec![clean(local), clean(&name.replace('.', "__"))];
                candidates.retain(|c| legal(c));
                let mut chosen = candidates.into_iter().find(|c| !taken.contains(c));
                let mut n = 2;
                while chosen.is_none() {
                    let c = format!("tool_{n}");
                    if !taken.contains(&c) { chosen = Some(c); }
                    n += 1;
                }
                chosen.unwrap()
            };
            taken.insert(wire.clone());
            out.to_client.insert(wire.clone(), name.to_owned());
            out.to_wire.insert(name.to_owned(), wire);
        }
        out
    }
    fn wire(&self, name: &str) -> String { self.to_wire.get(name).cloned().unwrap_or_else(|| name.to_owned()) }
    fn client(&self, name: &str) -> String { self.to_client.get(name).cloned().unwrap_or_else(|| name.to_owned()) }
}

/// Map `turn` onto a Chat Completions request body for `profile`.
pub(crate) fn chat_body(turn: &TurnRequest, profile: &ModelProfile) -> Result<Value, GatewayError> {
    chat_body_with(turn, profile, &ToolNames::new(turn), system_placement(profile))
}

fn chat_body_with(turn: &TurnRequest, profile: &ModelProfile, names: &ToolNames, systems: SystemPlacement) -> Result<Value, GatewayError> {
    if turn.modalities.audio_out {
        return Err(GatewayError::unsupported("audio output needs a speech synthesizer; this server has none").with_param("modalities"));
    }
    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = turn.system.as_ref().filter(|s| !s.is_empty()) {
        messages.push(json!({"role":"system","content":system}));
    }
    // One assistant turn is every assistant-side item (reasoning, text, tool
    // calls, hosted tool calls) between two user, system or tool-result
    // items: one Anthropic assistant message, or one run of Responses output
    // items. It becomes one chat assistant message, which every template
    // renders as reasoning, then content, then calls. Reasoning blocks
    // concatenate; content keeps every part in order; text after a call stays
    // in that message (chat requires a call's results to follow it directly).
    fn current(messages: &mut Vec<Value>) -> &mut Map<String, Value> {
        if messages.last().is_none_or(|m| m["role"] != "assistant") {
            messages.push(json!({"role":"assistant","content":null}));
        }
        messages.last_mut().and_then(Value::as_object_mut).expect("assistant message")
    }
    fn push_call(message: &mut Map<String, Value>, id: &str, name: String, arguments: String) {
        let calls = message.entry("tool_calls").or_insert_with(|| json!([]));
        if !calls.is_array() { *calls = json!([]); }
        calls.as_array_mut().unwrap().push(json!({"id":id,"type":"function","function":{"name":name,"arguments":arguments}}));
    }
    for item in &turn.items {
        match item {
            Item::Message { role: Role::System, content: parts } => {
                messages.push(json!({"role":"system","content":content(parts, "system")?}));
            }
            Item::Message { role: Role::User, content: parts } => {
                messages.push(json!({"role":"user","content":content(parts, "user")?}));
            }
            Item::Message { role: Role::Assistant, content: parts } => {
                let added = content(parts, "assistant")?;
                let message = current(&mut messages);
                // The chat path joins parts by the family's rule and decides
                // whether the family can carry assistant media.
                let merged = match message.remove("content").filter(|c| !c.is_null()) {
                    None => added,
                    Some(existing) => { let mut all = parts_of(existing); all.extend(parts_of(added)); Value::Array(all) }
                };
                message.insert("content".into(), merged);
            }
            Item::Reasoning { text, .. } => {
                let message = current(&mut messages);
                match message.get_mut("reasoning_content") {
                    Some(Value::String(existing)) => existing.push_str(text),
                    _ => { message.insert("reasoning_content".into(), json!(text)); }
                }
            }
            Item::ToolCall { id, name, arguments } => {
                let arguments = if arguments.trim().is_empty() { "{}".to_owned() } else { arguments.clone() };
                push_call(current(&mut messages), id, names.wire(name), arguments);
            }
            Item::ServerToolCall { id, name, input } =>
                push_call(current(&mut messages), id, names.wire(name), input.to_string()),
            Item::ToolResult { call_id, content: parts, is_error } => {
                let mut body = content(parts, "tool")?;
                // Chat has no error flag on tool results; the text says so.
                if *is_error {
                    match &mut body {
                        Value::String(text) => text.insert_str(0, "Error: "),
                        Value::Array(parts) => match parts.iter_mut().find(|p| p["type"] == "text") {
                            Some(part) => part["text"] = json!(format!("Error: {}", part["text"].as_str().unwrap_or_default())),
                            None => parts.insert(0, json!({"type":"text","text":"Error:"})),
                        },
                        _ => {}
                    }
                }
                messages.push(json!({"role":"tool","tool_call_id":call_id,"content":body}));
            }
            Item::ServerToolResult { call_id, output, .. } => {
                messages.push(json!({"role":"tool","tool_call_id":call_id,"content":output.to_string()}));
            }
        }
    }
    // An assistant message is never left with neither content nor calls.
    for message in &mut messages {
        if message["role"] == "assistant" && message["content"].is_null() && message.get("tool_calls").is_none() {
            message["content"] = json!("");
        }
    }
    if systems == SystemPlacement::FirstOnly { messages = first_system_only(messages); }
    // A forced choice needs a tool to force, as on /v1/chat/completions.
    match &turn.tool_choice {
        ToolChoice::Required if turn.tools.is_empty() =>
            return Err(GatewayError::invalid("required tool_choice needs at least one tool").with_param("tool_choice")),
        ToolChoice::Named { name } if !turn.tools.iter().any(|t| &t.name == name) =>
            return Err(GatewayError::invalid(format!("tool_choice: no tool named '{name}' was specified")).with_param("tool_choice")),
        _ => {}
    }
    let mut body = json!({"model": profile.id, "messages": messages, "stream": true,
        "stream_options": {"include_usage": true}});
    if !turn.tools.is_empty() && turn.tool_choice != ToolChoice::None {
        body["tools"] = Value::Array(turn.tools.iter().map(|tool| {
            // Templates render tool JSON in key order (Jinja preserve_order):
            // the conventional OpenAI order name, description, parameters.
            let mut function = Map::new();
            function.insert("name".into(), json!(names.wire(&tool.name)));
            if let Some(description) = &tool.description { function.insert("description".into(), json!(description)); }
            function.insert("parameters".into(), tool.parameters.clone());
            if tool.strict { function.insert("strict".into(), json!(true)); }
            json!({"type":"function","function":function})
        }).collect());
        body["tool_choice"] = match &turn.tool_choice {
            ToolChoice::Auto | ToolChoice::None => json!("auto"),
            ToolChoice::Required => json!("required"),
            ToolChoice::Named { name } => json!({"type":"function","function":{"name":names.wire(name)}}),
        };
        if let Some(parallel) = turn.parallel_tool_calls { body["parallel_tool_calls"] = json!(parallel); }
    }
    // Thinking: an explicit switch wins, then an effort; the chat default
    // (thinking on) applies otherwise.
    let effort = turn.reasoning.effort.as_deref().map(|e| match (e, &profile.encoding) {
        // The V4.1 adapter knows low/high/xhigh/max; medium is its high.
        ("medium", ModelEncoding::DeepseekV4 | ModelEncoding::DeepseekV41) => "high",
        (e, _) => e,
    });
    match turn.reasoning.enabled {
        Some(enabled) => {
            body["thinking"] = json!({"type": if enabled { "enabled" } else { "disabled" }});
            if enabled { if let Some(e) = effort.filter(|e| !matches!(*e, "none")) { body["reasoning_effort"] = json!(e); } }
        }
        None => if let Some(e) = effort { body["reasoning_effort"] = json!(e); },
    }
    if let Some(max) = turn.max_output_tokens { body["max_tokens"] = json!(max); }
    if let Some(t) = turn.sampling.temperature { body["temperature"] = json!(t); }
    if let Some(p) = turn.sampling.top_p { body["top_p"] = json!(p); }
    if let Some(k) = turn.sampling.top_k { body["top_k"] = json!(k); }
    if let Some(seed) = turn.sampling.seed { body["seed"] = json!(seed as i64); }
    if !turn.sampling.stop.is_empty() { body["stop"] = json!(turn.sampling.stop); }
    if let Some(format) = &turn.response_format {
        body["response_format"] = match format["type"].as_str() {
            Some("json_object") => json!({"type":"json_object"}),
            Some("json_schema") | None if format.get("schema").is_some() || format.get("json_schema").is_some() => {
                let definition = format.get("json_schema").cloned().unwrap_or_else(|| {
                    let mut d = format.clone();
                    if let Some(o) = d.as_object_mut() { o.remove("type"); }
                    if d.get("name").is_none() { d["name"] = json!("response"); }
                    d
                });
                json!({"type":"json_schema","json_schema":definition})
            }
            Some("text") => json!({"type":"text"}),
            _ => return Err(GatewayError::unsupported("response_format must be json_object or a JSON schema").with_param("response_format")),
        };
    }
    if let Some(key) = &turn.prompt_cache_key { body["prompt_cache_key"] = json!(key); }
    Ok(body)
}

impl Engine {
    fn check(&self, turn: &TurnRequest) -> Result<(), GatewayError> {
        let caps = self.capabilities();
        if !caps.json_schema {
            if turn.response_format.as_ref().is_some_and(|f| f["type"] == "json_schema" || f.get("schema").is_some()) {
                return Err(GatewayError::unsupported("response_format json_schema is not enabled for this model (no structured-output probe has passed); json_object is supported")
                    .with_param("response_format"));
            }
            if turn.tools.iter().any(|t| t.strict) {
                return Err(GatewayError::unsupported("strict function tools are not enabled for this model (no structured-output probe has passed)")
                    .with_param("tools"));
            }
        }
        Ok(())
    }

    fn built(&self, turn: &TurnRequest, names: &ToolNames) -> Result<super::Built, GatewayError> {
        self.check(turn)?;
        let body = chat_body_with(turn, self.profile(), names, self.systems)?;
        super::build(self.profile(), self.limits(), &HeaderMap::new(), body).map_err(rejection)
    }
}

impl Backend for Engine {
    fn name(&self) -> &str { "engine" }

    fn capabilities(&self) -> BackendCapabilities {
        let media = self.profile().capabilities;
        BackendCapabilities {
            vision: media.vision,
            audio_in: media.audio,
            audio_out: false,
            reasoning: true,
            json_schema: self.options.json_schema,
            strict_tools: self.options.json_schema,
            exact_token_count: self.snapshot.is_some(),
            kv_hooks: false,
        }
    }

    fn models(&self) -> Vec<ModelInfo> {
        let owner = self.profile().id.split_once('/').map_or("cuteafd", |(owner, _)| owner);
        vec![ModelInfo { id: self.profile().id.clone(), context_tokens: Some(self.limits().context()),
            max_output_tokens: Some(self.limits().output()), owned_by: owner.to_owned() }]
    }

    fn model_metadata(&self) -> Option<Value> {
        let mut record = super::model_record(&self.state);
        record["capabilities"]["json_schema"] = json!(self.options.json_schema);
        Some(record)
    }

    fn start(&self, turn: TurnRequest) -> BoxFuture<'static, Result<TurnStream, GatewayError>> {
        let this = self.clone();
        Box::pin(async move {
            // Per-turn engine admission: every front end, socket and compact
            // turn passes here, not only the routes the health layer sees.
            if let Some(reason) = this.profile().engine_health.as_ref().and_then(super::health::HealthWitness::reason) {
                return Err(GatewayError::new(ErrorKind::Overloaded, format!("engine unavailable: {reason}")));
            }
            let names = ToolNames::new(&turn);
            let built = this.built(&turn, &names)?;
            let stops = built.stop_sequences.clone();
            // The gateway accounts the turn from its events; the engine fills
            // admission and retirement through the handle in the job.
            let running = this.submitter().submit(this.profile(), built, turn.usage.clone(), false).await.map_err(rejection)?;
            Ok(events(running, names, stops))
        })
    }

    fn count_tokens(&self, turn: TurnRequest) -> BoxFuture<'static, Result<u32, GatewayError>> {
        let this = self.clone();
        Box::pin(async move {
            let names = ToolNames::new(&turn);
            let mut turn = turn;
            // Generation-only fields never fail a count; everything that
            // shapes the prompt (tools, formats, thinking) stays as generated.
            turn.max_output_tokens = None;
            turn.sampling = Default::default();
            let built = this.built(&turn, &names)?;
            let Some(snapshot) = this.snapshot.clone() else {
                return Ok(crate::gateway::backend::estimate_tokens(&turn));
            };
            super::count_prompt(&this.state, built, snapshot).await.map_err(rejection)
        })
    }
}

/// Chat chunks -> turn events. Tool-call indexes are the chat path's; ends
/// are emitted at the finish and sealed by the driver (`seal_tool_calls`).
fn events(running: Running, names: ToolNames, stops: Vec<String>) -> TurnStream {
    let Running { mut chunks, failure, tap, .. } = running;
    Box::pin(async_stream::stream! {
        // Dropping this stream drops `chunks`, which owns the engine event
        // receiver: the scheduler sees the closed channel and cancels the request.
        let mut started: Vec<usize> = Vec::new();
        let mut usage = Usage::default();
        while let Some(chunk) = chunks.next().await {
            let failed = failure.lock().unwrap().clone();
            if let Some(message) = failed {
                yield Err(GatewayError::internal(message)); return;
            }
            let chunk: ChatChunk = match chunk {
                Ok(chunk) => chunk,
                Err(e) => {
                    let failed = failure.lock().unwrap().clone();
                    yield Err(GatewayError::internal(failed.unwrap_or_else(|| e.to_string()))); return;
                }
            };
            if let Some(Some(u)) = &chunk.usage {
                usage.input_tokens = u32::try_from(u.prompt_tokens).unwrap_or(u32::MAX);
                usage.output_tokens = u32::try_from(u.completion_tokens).unwrap_or(u32::MAX);
                usage.cached_input_tokens = u32::try_from(u.prompt_tokens_details.cached_tokens).unwrap_or(u32::MAX);
                usage.reasoning_tokens = u.completion_tokens_details.as_ref().map_or(0, |d| d.reasoning_tokens as u32);
            }
            let value = serde_json::to_value(&chunk).expect("chat chunk serializes");
            let mut finish = None;
            for choice in value["choices"].as_array().into_iter().flatten() {
                let delta = &choice["delta"];
                if let Some(text) = delta["reasoning_content"].as_str().filter(|t| !t.is_empty()) {
                    yield Ok(TurnEvent::ReasoningDelta { text: text.to_owned() });
                }
                if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
                    yield Ok(TurnEvent::TextDelta { text: text.to_owned() });
                }
                for call in delta["tool_calls"].as_array().into_iter().flatten() {
                    let Some(index) = call["index"].as_u64().and_then(|i| usize::try_from(i).ok()) else { continue };
                    if let Some(id) = call["id"].as_str() {
                        let name = names.client(call["function"]["name"].as_str().unwrap_or_default());
                        started.push(index);
                        yield Ok(TurnEvent::ToolCallStart { index, id: id.to_owned(), name });
                    }
                    if let Some(arguments) = call["function"]["arguments"].as_str().filter(|a| !a.is_empty()) {
                        yield Ok(TurnEvent::ToolCallDelta { index, arguments: arguments.to_owned() });
                    }
                }
                if let Some(reason) = choice["finish_reason"].as_str() { finish = Some(reason.to_owned()); }
            }
            let Some(reason) = finish else { continue };
            let tapped = { tap.lock().unwrap().clone() };
            if usage.input_tokens == 0 { usage.input_tokens = u32::try_from(tapped.prompt.prompt_tokens).unwrap_or(u32::MAX); }
            let stop = match reason.as_str() {
                "tool_calls" => StopReason::ToolUse,
                "length" => StopReason::MaxTokens,
                "content_filter" => StopReason::ContentFilter,
                "aborted" => StopReason::Cancelled,
                "insufficient_system_resource" => {
                    yield Err(GatewayError::new(ErrorKind::Overloaded, "the engine ran out of resources for this request")); return;
                }
                _ => match tapped.stop_sequence {
                    Some(sequence) if stops.contains(&sequence) => StopReason::StopSequence { sequence: Some(sequence) },
                    _ => StopReason::EndTurn,
                },
            };
            for index in started.drain(..) { yield Ok(TurnEvent::ToolCallEnd { index }); }
            yield Ok(TurnEvent::Usage { usage });
            yield Ok(TurnEvent::Done { stop });
            return;
        }
        let failed = failure.lock().unwrap().clone();
        yield Err(GatewayError::internal(failed.unwrap_or_else(|| "engine stream ended without a finish reason".into())));
    })
}

/// The gateway engine backend for a serving router's state.
pub(crate) fn backend(state: NativeState, snapshot: Option<std::path::PathBuf>, options: EngineOptions) -> Arc<Engine> {
    Arc::new(Engine::new(state, snapshot, options))
}

#[cfg(test)]
mod tests;
