use std::collections::BTreeMap;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::Value;
use super::{number, Exchange, Flavor};
use crate::gateway::{turn::{StopReason, TurnEvent, Usage}, GatewayError, TurnStream};

/// SSE decoding is byte-based: a UTF-8 code point or CRLF can span HTTP chunks.
#[derive(Default)]
pub(super) struct Decoder { line: Vec<u8>, data: Vec<String>, after_cr: bool }
impl Decoder {
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<String>, GatewayError> {
        let mut frames = Vec::new();
        for &byte in bytes {
            if self.after_cr { self.after_cr = false; if byte == b'\n' { continue; } }
            if byte == b'\r' || byte == b'\n' {
                self.after_cr = byte == b'\r';
                let line = String::from_utf8(std::mem::take(&mut self.line)).map_err(|_| GatewayError::upstream("invalid UTF-8 in upstream SSE"))?;
                if line.is_empty() {
                    if !self.data.is_empty() { frames.push(self.data.join("\n")); self.data.clear(); }
                } else if let Some(value) = line.strip_prefix("data:") {
                    self.data.push(value.strip_prefix(' ').unwrap_or(value).to_string());
                }
            } else { self.line.push(byte); }
            if self.line.len() + self.data.iter().map(String::len).sum::<usize>() > 2 * 1024 * 1024 {
                return Err(GatewayError::upstream("upstream SSE frame exceeds size limit"));
            }
        }
        Ok(frames)
    }
}
#[derive(Default)]
struct Call { id: String, name: String, arguments: String, started: bool, ended: bool }
#[derive(Default)]
struct State { calls: BTreeMap<usize, Call>, usage: Usage, stop: Option<StopReason>, terminal: bool }
fn text(value: &Value, key: &str) -> String { value[key].as_str().unwrap_or_default().to_string() }
fn stop(value: &str, sequence: Option<String>) -> Result<StopReason, GatewayError> {
    match value {
        "stop" | "end_turn" => Ok(StopReason::EndTurn),
        "length" | "max_tokens" => Ok(StopReason::MaxTokens),
        "tool_calls" | "tool_use" => Ok(StopReason::ToolUse),
        "stop_sequence" => Ok(StopReason::StopSequence { sequence }),
        "content_filter" => Ok(StopReason::ContentFilter),
        "refusal" => Ok(StopReason::Refusal),
        "aborted" => Ok(StopReason::Cancelled),
        "pause_turn" => Ok(StopReason::PauseTurn),
        "insufficient_system_resource" => Err(GatewayError::new(crate::gateway::ErrorKind::Overloaded, "upstream generation ran out of resources")),
        _ => Err(GatewayError::upstream("unknown upstream finish reason")),
    }
}
impl State {
    fn call_events(&mut self, index: usize, force: bool) -> Result<Vec<TurnEvent>, GatewayError> {
        let call = self.calls.get_mut(&index).expect("call exists");
        let mut events = Vec::new();
        if !call.started && (force || !call.arguments.is_empty()) {
            if call.id.is_empty() || call.name.is_empty() { return Err(GatewayError::upstream("upstream tool call lacks id or name")); }
            call.started = true;
            events.push(TurnEvent::ToolCallStart { index, id: call.id.clone(), name: call.name.clone() });
        }
        if call.started && !call.arguments.is_empty() {
            events.push(TurnEvent::ToolCallDelta { index, arguments: std::mem::take(&mut call.arguments) });
        }
        Ok(events)
    }
    fn end_call(&mut self, index: usize) -> Result<Vec<TurnEvent>, GatewayError> {
        let mut events = self.call_events(index, true)?;
        let call = self.calls.get_mut(&index).unwrap();
        if !call.ended { call.ended = true; events.push(TurnEvent::ToolCallEnd { index }); }
        Ok(events)
    }
    fn finish(&mut self) -> Result<Vec<TurnEvent>, GatewayError> {
        let stop = self.stop.clone().ok_or_else(|| GatewayError::upstream("upstream stream ended without a finish reason"))?;
        let mut events = Vec::new();
        for index in self.calls.keys().copied().collect::<Vec<_>>() { events.extend(self.end_call(index)?); }
        events.push(TurnEvent::Usage { usage: self.usage });
        events.push(TurnEvent::Done { stop });
        self.terminal = true;
        Ok(events)
    }
    fn parse(&mut self, frame: &str, flavor: Flavor) -> Result<Vec<TurnEvent>, GatewayError> {
        if frame == "[DONE]" { return self.finish(); }
        let value: Value = serde_json::from_str(frame).map_err(|_| GatewayError::upstream("invalid upstream SSE JSON"))?;
        if value.get("error").is_some() || value["type"] == "error" { return Err(GatewayError::upstream("upstream emitted a stream error")); }
        let mut events = Vec::new();
        match flavor {
            Flavor::OpenaiChat => {
                if let Some(usage) = value.get("usage").filter(|u| u.is_object()) {
                    self.usage.input_tokens = number(usage, "prompt_tokens").unwrap_or(0);
                    self.usage.output_tokens = number(usage, "completion_tokens").unwrap_or(0);
                    self.usage.cached_input_tokens = number(&usage["prompt_tokens_details"], "cached_tokens").or_else(|| number(usage, "prompt_cache_hit_tokens")).unwrap_or(0);
                    self.usage.reasoning_tokens = number(&usage["completion_tokens_details"], "reasoning_tokens").unwrap_or(0);
                    events.push(TurnEvent::Usage { usage: self.usage });
                }
                if let Some(choices) = value["choices"].as_array() {
                    for choice in choices {
                        if choice["index"].as_u64().unwrap_or(0) != 0 { continue; }
                        let delta = &choice["delta"];
                        if let Some(t) = delta["reasoning_content"].as_str().filter(|s| !s.is_empty()) { events.push(TurnEvent::ReasoningDelta { text: t.into() }); }
                        if let Some(t) = delta["content"].as_str().filter(|s| !s.is_empty()) { events.push(TurnEvent::TextDelta { text: t.into() }); }
                        if let Some(calls) = delta["tool_calls"].as_array() {
                            for update in calls {
                                let index = update["index"].as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| GatewayError::upstream("missing upstream tool index"))?;
                                if index >= 1024 { return Err(GatewayError::upstream("upstream tool index exceeds limit")); }
                                let call = self.calls.entry(index).or_default();
                                call.id.push_str(&text(update, "id"));
                                call.name.push_str(&text(&update["function"], "name"));
                                call.arguments.push_str(&text(&update["function"], "arguments"));
                                events.extend(self.call_events(index, false)?);
                            }
                        }
                        if let Some(reason) = choice["finish_reason"].as_str() { self.stop = Some(stop(reason, None)?); }
                    }
                }
            }
            Flavor::Anthropic => match value["type"].as_str().unwrap_or_default() {
                "message_start" => {
                    let usage = &value["message"]["usage"];
                    self.usage.input_tokens = number(usage, "input_tokens").unwrap_or(0);
                    self.usage.cached_input_tokens = number(usage, "cache_read_input_tokens").unwrap_or(0);
                    self.usage.cache_creation_input_tokens = number(usage, "cache_creation_input_tokens").unwrap_or(0);
                    self.usage.input_tokens = self.usage.input_tokens.saturating_add(self.usage.cached_input_tokens).saturating_add(self.usage.cache_creation_input_tokens);
                    self.usage.output_tokens = number(usage, "output_tokens").unwrap_or(0);
                    events.push(TurnEvent::Usage { usage: self.usage });
                }
                "content_block_start" => {
                    let index = block_index(&value)?;
                    let block = &value["content_block"];
                    match block["type"].as_str() {
                        Some("tool_use") => {
                            let mut call = Call { id: text(block,"id"), name: text(block,"name"), ..Default::default() };
                            if block["input"].as_object().is_some_and(|m| !m.is_empty()) { call.arguments = block["input"].to_string(); }
                            self.calls.insert(index, call);
                            events.extend(self.call_events(index, true)?);
                        }
                        Some("text") => if let Some(t) = block["text"].as_str().filter(|t| !t.is_empty()) { events.push(TurnEvent::TextDelta { text: t.into() }); },
                        Some("thinking") => if let Some(t) = block["thinking"].as_str().filter(|t| !t.is_empty()) { events.push(TurnEvent::ReasoningDelta { text: t.into() }); },
                        Some("redacted_thinking") => events.push(TurnEvent::ReasoningSignature { signature: format!("redacted:{}", text(block,"data")) }),
                        _ => {}
                    }
                }
                "content_block_delta" => {
                    let delta = &value["delta"];
                    match delta["type"].as_str().unwrap_or_default() {
                        "text_delta" => events.push(TurnEvent::TextDelta { text: text(delta,"text") }),
                        "thinking_delta" => events.push(TurnEvent::ReasoningDelta { text: text(delta,"thinking") }),
                        "signature_delta" => events.push(TurnEvent::ReasoningSignature { signature: text(delta,"signature") }),
                        "input_json_delta" => {
                            let index = block_index(&value)?;
                            let call = self.calls.get_mut(&index).ok_or_else(|| GatewayError::upstream("tool delta without a start"))?;
                            call.arguments.push_str(&text(delta,"partial_json"));
                            events.extend(self.call_events(index, false)?);
                        }
                        _ => {}
                    }
                }
                "content_block_stop" => { let index = block_index(&value)?; if self.calls.contains_key(&index) { events.extend(self.end_call(index)?); } }
                "message_delta" => {
                    if let Some(n) = number(&value["usage"],"output_tokens") { self.usage.output_tokens = n; }
                    events.push(TurnEvent::Usage { usage: self.usage });
                    if let Some(reason) = value["delta"]["stop_reason"].as_str() {
                        self.stop = Some(stop(reason, value["delta"]["stop_sequence"].as_str().map(str::to_string))?);
                    }
                }
                "message_stop" => events.extend(self.finish()?),
                "ping" => {},
                _ => return Err(GatewayError::upstream("unknown upstream Anthropic stream event")),
            }
        }
        Ok(events)
    }
}
fn block_index(value: &Value) -> Result<usize, GatewayError> {
    value["index"].as_u64().filter(|n| *n < 1024).map(|n| n as usize).ok_or_else(|| GatewayError::upstream("invalid upstream block index"))
}
pub(super) fn events<S, E>(stream: S, flavor: Flavor, mut recording: Exchange) -> TurnStream
where S: Stream<Item = Result<Bytes, E>> + Send + 'static, E: Send + 'static {
    Box::pin(async_stream::stream! {
        futures::pin_mut!(stream);
        let mut decoder = Decoder::default();
        let mut state = State::default();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk { Ok(chunk) => chunk, Err(_) => { yield Err(GatewayError::upstream("upstream disconnected during streaming")); return; } };
            recording.push(&chunk);
            let frames = match decoder.feed(&chunk) { Ok(frames) => frames, Err(error) => { yield Err(error); return; } };
            for frame in frames {
                let events = match state.parse(&frame, flavor) { Ok(events) => events, Err(error) => { yield Err(error); return; } };
                for event in events { yield Ok(event); }
                if state.terminal { return; }
            }
        }
        yield Err(GatewayError::upstream("upstream disconnected before stream terminator"));
    })
}
