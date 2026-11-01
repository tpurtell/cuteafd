//! Incremental GLM output parser: `<think>` reasoning, visible content and
//! `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
//! calls, emitted as protocol-neutral [`OutputChunk`]s.
//!
//! Ported from glmrt's `GlmAssistantOutputFilter` and
//! `GlmToolCallStreamParser`: schema-declared string arguments are
//! JSON-escaped as they arrive, other values wait for `</arg_value>` to keep
//! their JSON type, text after the first parsed call is dropped, and GLM turn
//! markers end the output. A call is held until its closing tag (the SSE
//! keepalive covers the wait), so a client never sees part of a call that
//! turns out malformed:
//!
//! - a call that cannot be read (closing tag missing, markup in the name,
//!   arguments without a name, or empty) is returned as content from its
//!   opening tag;
//! - a name followed by stray closing tags (`bash</arg_key>`) is recovered
//!   when what is left is a declared tool;
//! - an argument that cannot be read (markup in its key, or no `<arg_value>`)
//!   loses its whole call: the argument cannot be named or has no value, and
//!   the call that would be released with it silently missing is not the one
//!   the model meant. Stray text between arguments is still dropped, and a
//!   repeated key keeps its last value, in the place of its first.
//!
//! Each case is logged. Results are independent of how the text is chunked.
//!
//! FR-G.14's hold-until-close contract is adopted from glm53f-api's
//! `dialect/glm.rs` (Hugh Madden, MIT); the code here follows glmrt's filter.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::Value;

use crate::openai::chat::stream_args::{ArgStream, name_is_readable};

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const TOOL_CALL: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
const ARG_KEY: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";
/// What ends a call's name or stray text in a call: an argument, the
/// closing tag, or a new call (the closing tag was missing).
const CALL_TAGS: [&str; 3] = [ARG_KEY, TOOL_CALL_END, TOOL_CALL];
/// Turn markers that end the assistant message when decoded as text.
pub const STOP_MARKERS: [&str; 8] = ["<|endoftext|>", "<eop>", "<|system|>", "<|user|>", "<|assistant|>",
    "<|observation|>", "<tool_response>", "</tool_response>"];

/// Parser settings for one request.
#[derive(Debug, Clone, Default)]
pub struct GlmParserOptions {
    /// The prompt ends inside `<think>`: output starts as reasoning.
    pub thinking: bool,
    /// Declared tools; `None` disables tool-call parsing and makes
    /// `<tool_call>` end the output (as glmrt's stop set did).
    pub tools: Option<Vec<ToolDefinition>>,
    /// Client stop sequences, matched in visible content only.
    pub stop_sequences: Vec<String>,
    /// The completion id, naming the request in log lines.
    pub id: String,
}

/// Why the parser stopped consuming text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlmStop {
    /// A GLM turn marker (or a disabled `<tool_call>`) appeared.
    Marker(String),
    /// A client stop sequence matched.
    Sequence(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode { Reasoning, Content, Name, Key, ValueStart, Value, KeyOrEnd, Skip, Lost(&'static str), Done }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker { ThinkOpen, ThinkClose, Call, Stop, Sequence }

/// A call being read; nothing of it goes out before its closing tag.
#[derive(Debug, Default)]
struct Call {
    name: String,
    key: Option<String>,
    /// Each argument's `"key":value` JSON in pieces (a string value is
    /// escaped piece by piece as it arrives), in the order keys first appear.
    arguments: Vec<(String, Vec<String>)>,
    /// The pieces of the argument being read.
    value: Vec<String>,
    string_open: bool,
    /// The call's text from its opening tag, content again if the call is lost.
    text: String,
    /// Whitespace held before the opening tag, kept with a lost call's text.
    before: String,
}

#[derive(Debug)]
pub struct GlmOutputParser {
    options: GlmParserOptions,
    mode: Mode,
    pending: String,
    call: Option<Call>,
    calls: usize,
    content_started: bool,
    held_whitespace: String,
    stop: Option<GlmStop>,
    /// Emit tool-call argument deltas as they arrive instead of holding the
    /// whole call until it closes (`CUTEAFD_STREAM_TOOLCALL_ARGS=1`).
    stream_args: ArgStream,
    /// The open call's `ToolCall` was emitted to the client: a call that turns
    /// out unreadable has already been streamed and cannot be retracted.
    streamed: bool,
}

impl GlmOutputParser {
    pub fn new(options: GlmParserOptions) -> Self {
        Self::with_stream_args(options, false)
    }

    /// [`Self::new`] with the streaming knob set explicitly, so tests and the
    /// router can enable it without touching process env.
    pub fn with_stream_args(mut options: GlmParserOptions, stream_args: bool) -> Self {
        options.stop_sequences.retain(|sequence| !sequence.is_empty());
        let mode = if options.thinking { Mode::Reasoning } else { Mode::Content };
        Self { options, mode, pending: String::new(), call: None, calls: 0, content_started: false,
            held_whitespace: String::new(), stop: None, stream_args: ArgStream::new(stream_args),
            streamed: false }
    }

    /// Consume generated text; returns the chunks it completes.
    pub fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        if self.mode == Mode::Done { return Vec::new(); }
        self.pending.push_str(text);
        let mut out = Vec::new();
        while self.step(false, &mut out) {}
        out
    }

    /// Flush held text at end of output. A call still open never closed: its
    /// text is returned as content.
    pub fn finish(&mut self) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        if self.mode != Mode::Done {
            while self.step(true, &mut out) {}
            if self.call.is_some() {
                let reason = if let Mode::Lost(reason) = self.mode { reason } else { "closing tag missing" };
                self.take(self.pending.len());
                self.lose(reason, &mut out);
            }
            self.mode = Mode::Done;
        }
        self.held_whitespace.clear();
        out
    }

    /// The reason the parser stopped early, if it did.
    pub fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }

    /// Tool calls parsed so far (each sent its chunks).
    pub fn tool_calls(&self) -> usize { self.calls }

    fn tools_enabled(&self) -> bool { self.options.tools.is_some() }

    fn step(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        match self.mode {
            Mode::Reasoning | Mode::Content => self.step_text(finishing, out),
            Mode::Name => self.step_name(out),
            Mode::Key => self.step_key(),
            Mode::ValueStart => self.step_value_start(out),
            Mode::Value => self.step_value(out),
            Mode::KeyOrEnd => self.step_key_or_end(out),
            Mode::Skip => self.step_skip(),
            Mode::Lost(reason) => self.step_lost(reason, out),
            Mode::Done => false,
        }
    }

    fn markers(&self) -> Vec<(&str, Marker)> {
        let mut markers = vec![(THINK_OPEN, Marker::ThinkOpen), (THINK_CLOSE, Marker::ThinkClose)];
        markers.extend(STOP_MARKERS.iter().map(|marker| (*marker, Marker::Stop)));
        if self.tools_enabled() {
            if self.mode == Mode::Content { markers.push((TOOL_CALL, Marker::Call)); }
        } else {
            markers.push((TOOL_CALL, Marker::Stop));
            markers.push((TOOL_CALL_END, Marker::Stop));
        }
        if self.mode == Mode::Content {
            markers.extend(self.options.stop_sequences.iter().map(|s| (s.as_str(), Marker::Sequence)));
        }
        markers
    }

    fn step_text(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        let markers = self.markers();
        let found = markers.iter()
            .filter_map(|(marker, kind)| self.pending.find(marker).map(|index| (index, marker.len(), *kind, *marker)))
            .min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let Some((index, length, kind, marker)) = found else {
            let retained = if finishing { 0 } else {
                markers.iter().map(|(marker, _)| held_prefix(&self.pending, marker)).max().unwrap_or(0)
            };
            let emit = self.pending.len() - retained;
            if emit == 0 { return false; }
            let text: String = self.pending.drain(..emit).collect();
            self.emit_text(text, out);
            return true;
        };
        let marker = marker.to_owned();
        let text: String = self.pending.drain(..index).collect();
        self.pending.drain(..length);
        self.emit_text(text, out);
        match kind {
            Marker::ThinkOpen => self.mode = Mode::Reasoning,
            Marker::ThinkClose => self.mode = Mode::Content,
            Marker::Call => {
                let before = std::mem::take(&mut self.held_whitespace);
                self.call = Some(Call { text: TOOL_CALL.into(), before, ..Call::default() });
                self.mode = Mode::Name;
            }
            Marker::Stop => self.halt(GlmStop::Marker(marker)),
            Marker::Sequence => self.halt(GlmStop::Sequence(marker)),
        }
        self.mode != Mode::Done
    }

    fn halt(&mut self, reason: GlmStop) {
        self.stop = Some(reason);
        self.pending.clear();
        self.held_whitespace.clear();
        self.mode = Mode::Done;
    }

    fn emit_text(&mut self, text: String, out: &mut Vec<OutputChunk>) {
        if text.is_empty() { return; }
        if self.mode == Mode::Reasoning {
            out.push(OutputChunk::Reasoning { content: text });
            return;
        }
        if self.calls == 0 { self.emit_content(&text, out); }
    }

    /// Visible content: leading whitespace is dropped, trailing whitespace
    /// waits for more content.
    fn emit_content(&mut self, text: &str, out: &mut Vec<OutputChunk>) {
        let text = if self.content_started { text } else { text.trim_start() };
        if text.is_empty() { return; }
        self.content_started = true;
        let body = text.trim_end();
        let trailing = &text[body.len()..];
        if !body.is_empty() {
            let content = std::mem::take(&mut self.held_whitespace) + body;
            out.push(OutputChunk::Raw { content });
        }
        self.held_whitespace.push_str(trailing);
    }

    /// Consume `length` bytes of pending text; an open call keeps them.
    fn take(&mut self, length: usize) -> String {
        let text: String = self.pending.drain(..length).collect();
        if let Some(call) = self.call.as_mut() { call.text.push_str(&text); }
        text
    }

    fn take_whitespace(&mut self) {
        self.take(self.pending.len() - self.pending.trim_start().len());
    }

    /// Whether the request declared a tool named `name`.
    fn offered(&self, name: &str) -> bool {
        !name.is_empty() && self.options.tools.iter().flatten().any(|tool| tool.name == name)
    }

    fn step_name(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        let Some((offset, tag)) = first_of(&self.pending, &CALL_TAGS) else { return false; };
        if tag == TOOL_CALL {
            self.take(offset);
            self.lose("closing tag missing before the next call", out);
            return true;
        }
        let written = self.pending[..offset].trim();
        let name = without_closing_tags(written);
        let lost = if written.is_empty() {
            Some(if tag == ARG_KEY { "arguments without a name" } else { "empty call" })
        } else if written.contains(['<', '>']) && !self.offered(name) {
            Some("markup in the name")
        } else {
            None
        };
        if let Some(reason) = lost {
            self.mode = Mode::Lost(reason);
            return true;
        }
        if name.len() != written.len() {
            tracing::warn!(id = %self.options.id, tool = name, text = %excerpt(written), "GLM tool call name recovered");
        }
        let name = name.to_owned();
        self.take(offset + tag.len());
        if tag == TOOL_CALL_END {
            self.call = None;
            self.calls += 1;
            if self.stream_args.enabled() && name_is_readable(&name, None) {
                self.stream_args.begin(out);
                self.stream_args.call(&name, out);
                self.stream_args.delta("{}", out);
            } else {
                out.push(OutputChunk::ToolCall { tool_name: name, arguments: "{}".into() });
            }
            self.mode = Mode::Content;
            return true;
        }
        self.call.as_mut().expect("a call is open").name = name.clone();
        // Name check before emitting: a call with an unreadable name is never
        // streamed as a tool call (it is returned as content on loss).
        if self.stream_args.enabled() && name_is_readable(&name, None) {
            self.stream_args.begin(out);
            self.stream_args.call(&name, out);
            self.stream_args.delta("{", out);
            self.streamed = true;
        }
        self.mode = Mode::Key;
        true
    }

    fn step_key(&mut self) -> bool {
        let Some(end) = self.pending.find(ARG_KEY_END) else { return false; };
        let key = self.pending[..end].trim().to_owned();
        self.take(end + ARG_KEY_END.len());
        if key.is_empty() || key.contains(['<', '>']) {
            self.mode = Mode::Lost("unreadable argument key");
            return true;
        }
        self.call.as_mut().expect("a call is open").key = Some(key);
        self.mode = Mode::ValueStart;
        true
    }

    fn step_value_start(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        self.take_whitespace();
        if self.pending.starts_with(ARG_VALUE) {
            self.take(ARG_VALUE.len());
            let label = {
                let call = self.call.as_mut().expect("a call is open");
                let key = call.key.as_deref().expect("a key precedes its value");
                if accepts_string(&call.name, key, self.options.tools.as_deref().unwrap_or_default()) {
                    let label = format!("{}:\"", serde_json::to_string(key).expect("string key"));
                    call.value.push(label.clone());
                    call.string_open = true;
                    Some(label)
                } else {
                    None
                }
            };
            if let Some(label) = label {
                self.stream_label(&label, out);
            }
            self.mode = Mode::Value;
            return true;
        }
        if ARG_VALUE.starts_with(self.pending.as_str()) { return false; }
        // The key was readable but its value never opens: the argument cannot
        // be delivered, and the call is not the one the model meant.
        self.mode = Mode::Lost("argument without a value");
        true
    }

    fn step_value(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        let string_open = self.call.as_ref().expect("a call is open").string_open;
        let Some(end) = self.pending.find(ARG_VALUE_END) else {
            if !string_open { return false; }
            let emit = self.pending.len() - held_prefix(&self.pending, ARG_VALUE_END);
            if emit == 0 { return false; }
            let fragment = self.take(emit);
            let escaped = json_string_contents(&fragment);
            self.call.as_mut().expect("a call is open").value.push(escaped.clone());
            if self.streamed { self.stream_args.delta(&escaped, out); }
            return true;
        };
        let raw = self.take(end);
        self.take(ARG_VALUE_END.len());
        let tools = self.options.tools.as_deref().unwrap_or_default();
        let call = self.call.as_mut().expect("a call is open");
        let key = call.key.take().expect("a key precedes its value");
        let string_open = call.string_open;
        let last = if string_open {
            call.string_open = false;
            format!("{}\"", json_string_contents(&raw))
        } else {
            let value = parse_value(&call.name, &key, &raw, tools);
            format!("{}:{}", serde_json::to_string(&key).expect("string key"),
                serde_json::to_string(&value).expect("JSON value"))
        };
        call.value.push(last.clone());
        if self.streamed {
            if string_open {
                // The label and earlier fragments were emitted as they arrived;
                // this is the last fragment plus the closing quote.
                self.stream_args.delta(&last, out);
            } else {
                // A non-string value is typed at its close, so it is emitted
                // whole, with the comma that separates it from the previous
                // argument.
                let index = call.arguments.len();
                if index > 0 { self.stream_args.delta(",", out); }
                self.stream_args.delta(&last, out);
            }
        }
        let value = std::mem::take(&mut call.value);
        match call.arguments.iter_mut().find(|argument| argument.0 == key) {
            Some(argument) => {
                tracing::warn!(id = %self.options.id, tool = %call.name, key = %key,
                    "GLM tool call argument repeated; the last value is kept");
                argument.1 = value;
            }
            None => call.arguments.push((key, value)),
        }
        self.mode = Mode::KeyOrEnd;
        true
    }

    /// Emit an argument's `"key":` label now, with the comma that separates it
    /// from the previous argument, when streaming.
    fn stream_label(&mut self, label: &str, out: &mut Vec<OutputChunk>) {
        if !self.streamed { return; }
        let index = self.call.as_ref().map_or(0, |call| call.arguments.len());
        if index > 0 { self.stream_args.delta(",", out); }
        self.stream_args.delta(label, out);
    }

    fn step_key_or_end(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        self.take_whitespace();
        if self.pending.starts_with(ARG_KEY) {
            self.take(ARG_KEY.len());
            self.mode = Mode::Key;
            return true;
        }
        if self.pending.starts_with(TOOL_CALL_END) {
            self.take(TOOL_CALL_END.len());
            self.release(out);
            return true;
        }
        if self.pending.starts_with(TOOL_CALL) {
            self.lose("closing tag missing before the next call", out);
            return true;
        }
        if CALL_TAGS.iter().any(|tag| tag.starts_with(self.pending.as_str())) { return false; }
        self.skip("stray text", excerpt(&self.pending));
        true
    }

    /// The call closed: send it, a comma before each argument but the first.
    fn release(&mut self, out: &mut Vec<OutputChunk>) {
        let call = self.call.take().expect("a call is open");
        if self.streamed {
            // The deltas were emitted as they arrived; only the close is owed.
            self.stream_args.delta("}", out);
            self.streamed = false;
        } else {
            out.push(OutputChunk::ToolCall { tool_name: call.name, arguments: "{".into() });
            for (index, (_, mut pieces)) in call.arguments.into_iter().enumerate() {
                if index > 0 { pieces[0].insert(0, ','); }
                out.extend(pieces.into_iter().map(arguments));
            }
            out.push(arguments("}".into()));
        }
        self.calls += 1;
        self.mode = Mode::Content;
    }

    /// Drop stray text between arguments up to the call's next tag.
    fn skip(&mut self, reason: &'static str, text: String) {
        let call = self.call.as_mut().expect("a call is open");
        call.key = None;
        tracing::warn!(id = %self.options.id, tool = %call.name, reason, %text, "GLM tool call text dropped");
        self.mode = Mode::Skip;
    }

    fn step_skip(&mut self) -> bool {
        if let Some((index, _)) = first_of(&self.pending, &CALL_TAGS) {
            self.take(index);
            self.mode = Mode::KeyOrEnd;
            return true;
        }
        let held = CALL_TAGS.iter().map(|tag| held_prefix(&self.pending, tag)).max().unwrap_or(0);
        self.take(self.pending.len() - held);
        false
    }

    /// A lost call runs through its closing tag, or up to a new call.
    fn step_lost(&mut self, reason: &'static str, out: &mut Vec<OutputChunk>) -> bool {
        let Some((index, tag)) = first_of(&self.pending, &[TOOL_CALL_END, TOOL_CALL]) else { return false; };
        self.take(if tag == TOOL_CALL_END { index + tag.len() } else { index });
        self.lose(reason, out);
        true
    }

    /// Return a call that cannot be read as content, from its opening tag
    /// (after the whitespace that preceded it), and log it.
    ///
    /// A call that was already streamed cannot be retracted: it is counted so
    /// the finish is `tool_calls`, and the validator rejects its assembled
    /// arguments with a clean 400/SSE error rather than a 500.
    fn lose(&mut self, reason: &'static str, out: &mut Vec<OutputChunk>) {
        let call = self.call.take().expect("a call is open");
        if std::mem::take(&mut self.streamed) {
            self.calls += 1;
            tracing::warn!(id = %self.options.id, reason, text = %excerpt(&call.text),
                "GLM tool call streamed, then found unreadable");
        } else {
            tracing::warn!(id = %self.options.id, reason, text = %excerpt(&call.text),
                "GLM tool call returned as content");
            self.held_whitespace = call.before;
            self.emit_content(&call.text, out);
        }
        self.mode = Mode::Content;
    }
}

fn arguments(content: String) -> OutputChunk { OutputChunk::ToolArgumentsDelta { content } }

/// Length of the longest suffix of `text` that is a proper prefix of `marker`.
fn held_prefix(text: &str, marker: &str) -> usize {
    let longest = text.len().min(marker.len().saturating_sub(1));
    (1..=longest).rev()
        .find(|length| {
            let start = text.len() - length;
            text.is_char_boundary(start) && marker.starts_with(&text[start..])
        })
        .unwrap_or(0)
}

/// The earliest of `tags` in `text`: (index, tag).
fn first_of(text: &str, tags: &[&'static str]) -> Option<(usize, &'static str)> {
    tags.iter().filter_map(|&tag| text.find(tag).map(|index| (index, tag))).min_by_key(|&(index, _)| index)
}

/// `name` without the closing tags (`</...>`) and whitespace at its end.
fn without_closing_tags(name: &str) -> &str {
    let mut name = name.trim_end();
    while let Some(head) = name.strip_suffix('>') {
        match head.rfind("</") {
            Some(start) if !head[start + 2..].contains(['<', '>']) => name = head[..start].trim_end(),
            _ => break,
        }
    }
    name
}

/// A short excerpt of model text for a log line.
fn excerpt(text: &str) -> String {
    match text.char_indices().nth(80) {
        Some((cut, _)) => format!("{:?}...", &text[..cut]),
        None => format!("{text:?}"),
    }
}

fn json_string_contents(value: &str) -> String {
    let quoted = serde_json::to_string(value).expect("string serializes");
    quoted[1..quoted.len() - 1].to_owned()
}

fn parse_value(tool: &str, key: &str, raw: &str, tools: &[ToolDefinition]) -> Value {
    if accepts_string(tool, key, tools) { return Value::String(raw.to_owned()); }
    let value = raw.trim();
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()))
}

/// Whether the declared schema for `tool.key` admits a string, so the raw
/// `<arg_value>` text is the value itself rather than JSON.
pub fn accepts_string(tool: &str, key: &str, tools: &[ToolDefinition]) -> bool {
    tools.iter().find(|definition| definition.name == tool)
        .and_then(|definition| definition.parameters.get("properties"))
        .and_then(|properties| properties.get(key))
        .is_some_and(schema_accepts_string)
}

fn schema_accepts_string(schema: &Value) -> bool {
    let direct = match schema.get("type") {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some("string")),
        _ => false,
    };
    direct
        || ["anyOf", "oneOf"].into_iter()
            .filter_map(|key| schema.get(key).and_then(Value::as_array))
            .flatten()
            .any(schema_accepts_string)
        || schema.get("enum").and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty() && values.iter().all(Value::is_string))
}
