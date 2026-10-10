use std::collections::HashMap;

use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::gateway::{
    turn::{StopReason, TurnEvent, Usage},
    GatewayError,
};

enum Block {
    Thinking {
        text: String,
        signature: String,
    },
    Text(String),
    Tool {
        id: String,
        name: String,
        arguments: String,
    },
    Fixed(Value),
}
struct Slot {
    block: Block,
    closed: bool,
}

/// One accumulator drives both JSON and SSE, so they cannot disagree on
/// block boundaries, tool inputs, stop reasons or cumulative usage.
pub(super) struct Renderer {
    id: String,
    model: String,
    slots: Vec<Slot>,
    active: Option<usize>,
    tools: HashMap<usize, usize>,
    usage: Usage,
    stop: Option<StopReason>,
}
impl Renderer {
    pub fn new(model: String) -> Self {
        Self {
            id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            model,
            slots: Vec::new(),
            active: None,
            tools: HashMap::new(),
            usage: Usage::default(),
            stop: None,
        }
    }
    pub fn done(&self) -> bool {
        self.stop.is_some()
    }
    pub fn start(&self) -> Value {
        json!({"type":"message_start","message":self.message()})
    }
    pub fn message(&self) -> Value {
        let content: Vec<Value> = self.slots.iter().map(|slot| match &slot.block {
            Block::Thinking { text, signature } => json!({"type":"thinking","thinking":text,"signature":signature}),
            Block::Text(text) => json!({"type":"text","text":text}),
            Block::Tool { id, name, arguments } => json!({"type":"tool_use","id":id,"name":name,"input":tool_input(arguments, self.stop.as_ref().is_some_and(truncated)).unwrap_or_else(|_| json!({}))}),
            Block::Fixed(value) => value.clone(),
        }).collect();
        let (stop_reason, stop_sequence) = stop(self.stop.as_ref());
        json!({"id":self.id,"type":"message","role":"assistant","model":self.model,"content":content,"stop_reason":stop_reason,"stop_sequence":stop_sequence,"usage":usage(self.usage)})
    }
    fn add(&mut self, block: Block, start: Value, frames: &mut Vec<Value>) -> usize {
        let index = self.slots.len();
        self.slots.push(Slot {
            block,
            closed: false,
        });
        frames.push(json!({"type":"content_block_start","index":index,"content_block":start}));
        index
    }
    fn close(&mut self, index: usize, frames: &mut Vec<Value>) -> Result<(), GatewayError> {
        let slot = &mut self.slots[index];
        if slot.closed {
            return Ok(());
        }
        if let Block::Thinking { text, signature } = &mut slot.block {
            if signature.is_empty() {
                *signature = thinking_signature(text);
                frames.push(delta(
                    index,
                    json!({"type":"signature_delta","signature":signature}),
                ));
            }
        }
        if let Block::Tool { arguments, .. } = &mut slot.block {
            if arguments.is_empty() {
                arguments.push_str("{}");
                frames.push(delta(
                    index,
                    json!({"type":"input_json_delta","partial_json":"{}"}),
                ));
            }
        }
        slot.closed = true;
        frames.push(json!({"type":"content_block_stop","index":index}));
        Ok(())
    }
    fn close_active(&mut self, frames: &mut Vec<Value>) -> Result<(), GatewayError> {
        if let Some(index) = self.active.take() {
            self.close(index, frames)?;
        }
        Ok(())
    }
    pub fn push(&mut self, event: TurnEvent) -> Result<Vec<Value>, GatewayError> {
        if self.done() {
            return Err(GatewayError::upstream("backend emitted events after Done"));
        }
        let mut frames = Vec::new();
        match event {
            TurnEvent::ReasoningDelta { text } => {
                if !self
                    .active
                    .is_some_and(|i| matches!(self.slots[i].block, Block::Thinking { .. }))
                {
                    self.close_active(&mut frames)?;
                    self.active = Some(self.add(
                        Block::Thinking {
                            text: String::new(),
                            signature: String::new(),
                        },
                        json!({"type":"thinking","thinking":"","signature":""}),
                        &mut frames,
                    ));
                }
                let i = self.active.unwrap();
                if let Block::Thinking {
                    text: accumulated, ..
                } = &mut self.slots[i].block
                {
                    accumulated.push_str(&text);
                }
                frames.push(delta(i, json!({"type":"thinking_delta","thinking":text})));
            }
            TurnEvent::ReasoningSignature { signature } => {
                if !self
                    .active
                    .is_some_and(|i| matches!(self.slots[i].block, Block::Thinking { .. }))
                {
                    self.close_active(&mut frames)?;
                    self.active = Some(self.add(
                        Block::Thinking {
                            text: String::new(),
                            signature: String::new(),
                        },
                        json!({"type":"thinking","thinking":"","signature":""}),
                        &mut frames,
                    ));
                }
                let i = self.active.unwrap();
                if let Block::Thinking {
                    signature: accumulated,
                    ..
                } = &mut self.slots[i].block
                {
                    accumulated.push_str(&signature);
                }
                frames.push(delta(
                    i,
                    json!({"type":"signature_delta","signature":signature}),
                ));
            }
            TurnEvent::TextDelta { text } => {
                if !self
                    .active
                    .is_some_and(|i| matches!(self.slots[i].block, Block::Text(_)))
                {
                    self.close_active(&mut frames)?;
                    self.active = Some(self.add(
                        Block::Text(String::new()),
                        json!({"type":"text","text":""}),
                        &mut frames,
                    ));
                }
                let i = self.active.unwrap();
                if let Block::Text(accumulated) = &mut self.slots[i].block {
                    accumulated.push_str(&text);
                }
                frames.push(delta(i, json!({"type":"text_delta","text":text})));
            }
            TurnEvent::ToolCallStart { index, id, name } => {
                self.close_active(&mut frames)?;
                if self.tools.contains_key(&index) {
                    return Err(GatewayError::upstream("duplicate backend tool index"));
                }
                let id = wire_id("toolu_", &id);
                let i = self.add(
                    Block::Tool {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: String::new(),
                    },
                    json!({"type":"tool_use","id":id,"name":name,"input":{}}),
                    &mut frames,
                );
                self.tools.insert(index, i);
            }
            TurnEvent::ToolCallDelta { index, arguments } => {
                let i = *self
                    .tools
                    .get(&index)
                    .ok_or_else(|| GatewayError::upstream("tool input without tool start"))?;
                if self.slots[i].closed {
                    return Err(GatewayError::upstream("tool input after tool stop"));
                }
                if let Block::Tool {
                    arguments: accumulated,
                    ..
                } = &mut self.slots[i].block
                {
                    accumulated.push_str(&arguments);
                }
                frames.push(delta(
                    i,
                    json!({"type":"input_json_delta","partial_json":arguments}),
                ));
            }
            TurnEvent::ToolCallEnd { index } => {
                let i = *self
                    .tools
                    .get(&index)
                    .ok_or_else(|| GatewayError::upstream("tool stop without tool start"))?;
                self.close(i, &mut frames)?;
            }
            TurnEvent::ServerToolCall { id, name, input } => {
                self.close_active(&mut frames)?;
                let id = wire_id("srvtoolu_", &id);
                let value = json!({"type":"server_tool_use","id":id,"name":name,"input":input});
                let i = self.add(
                    Block::Fixed(value),
                    json!({"type":"server_tool_use","id":id,"name":name,"input":{}}),
                    &mut frames,
                );
                frames.push(delta(
                    i,
                    json!({"type":"input_json_delta","partial_json":input.to_string()}),
                ));
                self.close(i, &mut frames)?;
            }
            TurnEvent::ServerToolResult { id, name, output } => {
                self.close_active(&mut frames)?;
                if name != "web_search" {
                    return Err(GatewayError::unsupported(format!(
                        "unsupported hosted result {name}"
                    )));
                }
                let value = json!({"type":"web_search_tool_result","tool_use_id":wire_id("srvtoolu_", &id),"content":search_content(output)});
                let i = self.add(Block::Fixed(value.clone()), value, &mut frames);
                self.close(i, &mut frames)?;
            }
            TurnEvent::Usage { usage } => self.usage = usage,
            TurnEvent::Done { stop: mut reason } => {
                if matches!(reason, StopReason::Cancelled)
                    && self
                        .slots
                        .iter()
                        .any(|slot| !slot.closed && matches!(slot.block, Block::Tool { .. }))
                {
                    reason = StopReason::MaxTokens;
                }
                self.close_active(&mut frames)?;
                for index in 0..self.slots.len() {
                    self.close(index, &mut frames)?;
                }
                // ToolCallEnd does not imply complete JSON: a token cap may
                // truncate the call. Match the official SDK's partial parser.
                for slot in &self.slots {
                    if let Block::Tool { arguments, .. } = &slot.block {
                        tool_input(arguments, truncated(&reason))?;
                    }
                }
                let (stop_reason, stop_sequence) = stop(Some(&reason));
                frames.push(json!({"type":"message_delta","delta":{"stop_reason":stop_reason,"stop_sequence":stop_sequence},"usage":usage(self.usage)}));
                frames.push(json!({"type":"message_stop"}));
                self.stop = Some(reason);
            }
        }
        Ok(frames)
    }
}
fn truncated(reason: &StopReason) -> bool {
    matches!(
        reason,
        StopReason::MaxTokens
            | StopReason::Refusal
            | StopReason::ContentFilter
            | StopReason::Cancelled
    )
}
pub(super) fn tool_input(arguments: &str, partial: bool) -> Result<Value, GatewayError> {
    let value = if partial {
        jiter::JsonValue::parse_with_config(arguments.as_bytes(), false, jiter::PartialMode::On)
            .map(|v| partial_value(&v))
            .map_err(|_| ())
    } else {
        serde_json::from_str(arguments).map_err(|_| ())
    };
    value
        .ok()
        .filter(Value::is_object)
        .ok_or_else(|| GatewayError::upstream("backend tool input must be a valid JSON object"))
}
fn partial_value(value: &jiter::JsonValue<'_>) -> Value {
    use jiter::JsonValue as J;
    match value {
        J::Null => Value::Null,
        J::Bool(v) => json!(v),
        J::Int(v) => json!(v),
        J::Float(v) => json!(v),
        J::Str(v) => json!(v),
        J::Array(v) => Value::Array(v.iter().map(partial_value).collect()),
        J::Object(v) => Value::Object(
            v.iter()
                .map(|(key, v)| (key.to_string(), partial_value(v)))
                .collect(),
        ),
    }
}
fn delta(index: usize, delta: Value) -> Value {
    json!({"type":"content_block_delta","index":index,"delta":delta})
}
/// Signature synthesized for a thinking block this gateway produced: a digest
/// of the exact thinking text, so an edited block can be recognized on input.
pub(super) fn thinking_signature(text: &str) -> String {
    STANDARD.encode(format!(
        "{THINKING_SIGNATURE_PREFIX}{}",
        hex_digest(text.as_bytes())
    ))
}
pub(super) const THINKING_SIGNATURE_PREFIX: &str = "cuteafd-thinking-v1:";
fn hex_digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn wire_id(prefix: &str, id: &str) -> String {
    if id.starts_with(prefix) {
        id.into()
    } else {
        format!("{prefix}{}", hex_digest(id.as_bytes()))
    }
}
fn stop(reason: Option<&StopReason>) -> (Option<&'static str>, Option<&str>) {
    match reason {
        None => (None, None),
        Some(StopReason::EndTurn | StopReason::Cancelled) => (Some("end_turn"), None),
        Some(StopReason::MaxTokens) => (Some("max_tokens"), None),
        Some(StopReason::ToolUse) => (Some("tool_use"), None),
        Some(StopReason::PauseTurn) => (Some("pause_turn"), None),
        Some(StopReason::Refusal | StopReason::ContentFilter) => (Some("refusal"), None),
        Some(StopReason::StopSequence { sequence }) => (Some("stop_sequence"), sequence.as_deref()),
    }
}
fn usage(usage: Usage) -> Value {
    let uncached = usage
        .input_tokens
        .saturating_sub(usage.cached_input_tokens)
        .saturating_sub(usage.cache_creation_input_tokens);
    json!({"input_tokens":uncached,"output_tokens":usage.output_tokens,
        "cache_creation_input_tokens":usage.cache_creation_input_tokens,"cache_read_input_tokens":usage.cached_input_tokens,
        "server_tool_use":{"web_search_requests":usage.web_search_requests},"service_tier":"standard"})
}
fn search_content(output: Value) -> Value {
    if let Some(error) = output.get("error").and_then(Value::as_str) {
        let error = match error {
            "invalid_input" => "invalid_tool_input",
            "too_many_requests" | "invalid_tool_input" | "max_uses_exceeded" | "query_too_long"
            | "request_too_large" => error,
            _ => "unavailable",
        };
        return json!({"type":"web_search_tool_result_error","error_code":error});
    }
    json!(output.get("results").and_then(Value::as_array).into_iter().flatten().map(|hit| {
        json!({"type":"web_search_result","url":hit["url"],"title":hit["title"],
            "encrypted_content":STANDARD.encode(hit["content"].as_str().unwrap_or_default()),"page_age":hit.get("published")})
    }).collect::<Vec<_>>())
}
