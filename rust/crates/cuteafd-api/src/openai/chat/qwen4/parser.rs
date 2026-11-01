//! Incremental Qwen output parser: `<think>` reasoning, visible content and
//! Qwen3-Coder XML tool calls, emitted as protocol-neutral [`OutputChunk`]s.
//! MiMo V2 is served through this same dialect (`ModelEncoding::Qwen`), so its
//! malformed-call fixtures live here too.
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Values follow vLLM's `qwen3_coder` parser: one leading and one trailing
//! newline are stripped; a parameter whose declared schema admits a string
//! (or has no type, or is not declared) is the raw text; other values parse as
//! JSON, then `true`/`false` in any case, else stay text. Several calls may
//! follow each other; text after the first parsed call is dropped and turn
//! markers end the output.
//!
//! A call is held until its closing `</tool_call>` (the SSE keepalive covers
//! the wait), so a client never sees part of a call that turns out malformed.
//! FR-G.14's hold-until-close contract is adopted from glm53f-api's
//! `dialect/glm.rs` (Hugh Madden, MIT); the code here reads the tag format
//! above (vLLM's `qwen3_coder` dialect):
//!
//! - a call that cannot be read (closing tag missing, `<function=...>` absent,
//!   markup in the name, arguments without a name, or an empty/markup
//!   parameter key) is returned as content from its opening tag;
//! - `finish_reason: tool_calls` counts only calls that parsed, so an
//!   unreadable call never fails the request and never claims a tool call;
//! - an unreadable parameter key loses the whole call: the argument cannot be
//!   named, and the call that would be released (with that argument silently
//!   missing, or with a `""` key) is not the one the model meant. Stray text
//!   between arguments is still dropped from its call (its bytes still return
//!   as content if the call is later lost);
//! - each lost call is logged.
//!
//! Results are independent of how the text is chunked.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::Value;

use crate::openai::chat::glm5::parser::GlmStop;
use crate::openai::chat::glm5::TextParser;
use crate::openai::chat::stream_args::{ArgStream, name_is_readable};

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const TOOL_CALL: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";
/// Turn markers that end the assistant message when decoded as text.
pub const STOP_MARKERS: [&str; 5] = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "<tool_response>", "</tool_response>"];
/// What ends a call's name or stray text in a call: a parameter, the function
/// close, or the call close.
const CALL_TAGS: [&str; 3] = [PARAMETER, FUNCTION_END, TOOL_CALL_END];

/// Parser settings for one request.
#[derive(Debug, Clone, Default)]
pub struct QwenParserOptions {
    /// The prompt ends inside `<think>`: output starts as reasoning.
    pub thinking: bool,
    /// Declared tools; `None` disables tool-call parsing and makes
    /// `<tool_call>` end the output.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Client stop sequences, matched in visible content only.
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Reasoning,
    Content,
    Function,
    Name,
    Arguments,
    Key,
    Value,
    CallEnd,
    Skip,
    Lost(&'static str),
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker { ThinkOpen, ThinkClose, Call, Stop, Sequence }

/// A call being read; nothing of it goes out before its closing tag.
#[derive(Debug, Default)]
struct Call {
    name: String,
    key: Option<String>,
    /// Each argument's `"key":value` JSON, in the order keys first appear.
    arguments: Vec<(String, String)>,
    /// The call's text from its opening tag, content again if the call is lost.
    text: String,
    /// Whitespace held before the opening tag, kept with a lost call's text.
    before: String,
    /// The value's leading newline has been consumed (streaming only).
    value_leading: bool,
}

#[derive(Debug)]
pub struct QwenOutputParser {
    options: QwenParserOptions,
    turn_markers: &'static [&'static str],
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

impl QwenOutputParser {
    pub fn new(options: QwenParserOptions) -> Self {
        Self::with_turn_markers(options, &STOP_MARKERS)
    }

    /// [`Self::new`] with the streaming knob set explicitly, so tests and the
    /// router can enable it without touching process env.
    pub fn with_stream_args(options: QwenParserOptions, stream_args: bool) -> Self {
        Self::with_turn_markers_and_stream_args(options, &STOP_MARKERS, stream_args)
    }

    /// The XML call grammar is shared; each checkpoint owns its turn stops.
    pub fn with_turn_markers(options: QwenParserOptions, turn_markers: &'static [&'static str]) -> Self {
        Self::with_turn_markers_and_stream_args(options, turn_markers, false)
    }

    pub(crate) fn with_turn_markers_and_stream_args(mut options: QwenParserOptions,
        turn_markers: &'static [&'static str], stream_args: bool) -> Self {
        options.stop_sequences.retain(|sequence| !sequence.is_empty());
        let mode = if options.thinking { Mode::Reasoning } else { Mode::Content };
        Self { options, turn_markers, mode, pending: String::new(), call: None, calls: 0,
            content_started: false, held_whitespace: String::new(), stop: None,
            stream_args: ArgStream::new(stream_args), streamed: false }
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
                let pending = std::mem::take(&mut self.pending);
                if let Some(call) = self.call.as_mut() { call.text.push_str(&pending); }
                self.lose(reason, &mut out);
            }
            self.mode = Mode::Done;
        }
        self.held_whitespace.clear();
        out
    }

    /// The reason the parser stopped early, if it did.
    pub fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }

    /// Parsed (closed) tool calls so far; a lost call never counts.
    pub fn tool_calls(&self) -> usize { self.calls }

    fn tools_enabled(&self) -> bool { self.options.tools.is_some() }

    fn step(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        match self.mode {
            Mode::Reasoning | Mode::Content => self.step_text(finishing, out),
            Mode::Function => self.step_function(),
            Mode::Name => self.step_name(out),
            Mode::Arguments => self.step_arguments(),
            Mode::Key => self.step_key(),
            Mode::Value => self.step_value(out),
            Mode::CallEnd => self.step_call_end(out),
            Mode::Skip => self.step_skip(),
            Mode::Lost(reason) => self.step_lost(reason, out),
            Mode::Done => false,
        }
    }

    fn markers(&self) -> Vec<(&str, Marker)> {
        let mut markers = vec![(THINK_OPEN, Marker::ThinkOpen), (THINK_CLOSE, Marker::ThinkClose)];
        markers.extend(self.turn_markers.iter().map(|marker| (*marker, Marker::Stop)));
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
                self.mode = Mode::Function;
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
        self.call = None;
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

    /// After `<tool_call>`: expect `<function=NAME>`; anything else is a lost
    /// call (empty, arguments without a name, or no function tag at all).
    fn step_function(&mut self) -> bool {
        self.take_whitespace();
        if self.pending.starts_with(FUNCTION) {
            self.take(FUNCTION.len());
            self.mode = Mode::Name;
            return true;
        }
        if [FUNCTION, TOOL_CALL_END].iter().any(|tag| tag.starts_with(self.pending.as_str())) { return false; }
        let text = self.pending.trim_start();
        let reason = if text.starts_with(TOOL_CALL_END) { "empty call" }
            else if text.starts_with(PARAMETER) { "arguments without a name" }
            else { "missing <function=...>" };
        self.mode = Mode::Lost(reason);
        true
    }

    /// Read the function name up to `>`.
    fn step_name(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        let Some(end) = self.pending.find('>') else { return false; };
        let name = self.pending[..end].trim().to_owned();
        self.take(end + 1);
        if name.is_empty() || name.contains(['<', '\n']) {
            self.mode = Mode::Lost(if name.is_empty() { "empty call" } else { "markup in the name" });
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
        self.mode = Mode::Arguments;
        true
    }

    /// After the name: expect `<parameter=...>` or `</function>`; stray text is
    /// dropped from the call.
    fn step_arguments(&mut self) -> bool {
        self.take_whitespace();
        if self.pending.starts_with(PARAMETER) {
            self.take(PARAMETER.len());
            self.mode = Mode::Key;
            return true;
        }
        if self.pending.starts_with(FUNCTION_END) {
            self.take(FUNCTION_END.len());
            self.mode = Mode::CallEnd;
            return true;
        }
        if self.pending.starts_with(TOOL_CALL_END) {
            self.mode = Mode::Lost("closing tag missing before </function>");
            return true;
        }
        if CALL_TAGS.iter().any(|tag| tag.starts_with(self.pending.as_str())) { return false; }
        self.skip("stray text between arguments");
        true
    }

    /// Read the parameter key up to `>`; an unreadable key loses the whole
    /// call — the argument it introduces cannot be named, and the call that
    /// would be released is not the one the model meant.
    fn step_key(&mut self) -> bool {
        let Some(end) = self.pending.find('>') else { return false; };
        let key = self.pending[..end].trim().to_owned();
        self.take(end + 1);
        if key.is_empty() || key.contains(['<', '\n']) {
            self.mode = Mode::Lost("unreadable argument key");
            return true;
        }
        self.call.as_mut().expect("a call is open").key = Some(key);
        self.mode = Mode::Value;
        true
    }

    /// Read a value up to `</parameter>`; one leading and one trailing newline
    /// are stripped, then the value is typed by the tool schema.
    ///
    /// When streaming, a string value is emitted as it arrives (minus the
    /// stripped newlines); a non-string value is typed at its close, so it is
    /// emitted whole.
    fn step_value(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        let (name, key) = {
            let call = self.call.as_ref().expect("a call is open");
            (call.name.clone(), call.key.clone().expect("a key precedes its value"))
        };
        let string = value_is_string(&self.options, &name, &key);
        if self.streamed && string {
            if self.pending.is_empty() { return false; }
            if !self.call.as_ref().expect("a call is open").value_leading {
                let index = self.call.as_ref().expect("a call is open").arguments.len();
                if index > 0 { self.stream_args.delta(",", out); }
                let label = format!("{}:\"", serde_json::to_string(&key).expect("string key"));
                self.stream_args.delta(&label, out);
                if self.pending.starts_with('\n') { self.take(1); }
                self.call.as_mut().expect("a call is open").value_leading = true;
            }
            if let Some(end) = self.pending.find(PARAMETER_END) {
                let raw: String = self.pending[..end].to_owned();
                self.take(end + PARAMETER_END.len());
                let raw = raw.strip_suffix('\n').unwrap_or(&raw);
                let escaped = json_string_contents(raw);
                self.stream_args.delta(&escaped, out);
                self.stream_args.delta("\"", out);
                let text = format!("{}:{}", serde_json::to_string(&key).expect("string key"),
                    serde_json::to_string(&Value::String(raw.to_owned())).expect("JSON value"));
                let call = self.call.as_mut().expect("a call is open");
                match call.arguments.iter_mut().find(|(existing, _)| *existing == key) {
                    Some(argument) => {
                        tracing::warn!(tool = %call.name, key = %key,
                            "Qwen tool call argument repeated; the last value is kept");
                        argument.1 = text;
                    }
                    None => call.arguments.push((key, text)),
                }
                self.mode = Mode::Arguments;
                return true;
            }
            // Emit all but the last character: it may be the stripped trailing
            // newline, which is only known at the close.
            let hold = self.pending.chars().last().map_or(0, |character| character.len_utf8());
            let emit = self.pending.len() - hold;
            if emit == 0 { return false; }
            let fragment = self.take(emit);
            let escaped = json_string_contents(&fragment);
            self.stream_args.delta(&escaped, out);
            return true;
        }
        let Some(end) = self.pending.find(PARAMETER_END) else { return false; };
        let raw: String = self.pending[..end].to_owned();
        self.take(end + PARAMETER_END.len());
        let raw = raw.strip_prefix('\n').unwrap_or(&raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let value = if string { Value::String(raw.to_owned()) } else { typed_value(raw) };
        let text = format!("{}:{}", serde_json::to_string(&key).expect("string key"),
            serde_json::to_string(&value).expect("JSON value"));
        if self.streamed {
            let index = self.call.as_ref().expect("a call is open").arguments.len();
            if index > 0 { self.stream_args.delta(",", out); }
            self.stream_args.delta(&text, out);
        }
        let call = self.call.as_mut().expect("a call is open");
        match call.arguments.iter_mut().find(|(existing, _)| *existing == key) {
            Some(argument) => {
                tracing::warn!(tool = %call.name, key = %key,
                    "Qwen tool call argument repeated; the last value is kept");
                argument.1 = text;
            }
            None => call.arguments.push((key, text)),
        }
        self.mode = Mode::Arguments;
        true
    }

    /// After `</function>`: expect `</tool_call>`.
    fn step_call_end(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        self.take_whitespace();
        if self.pending.starts_with(TOOL_CALL_END) {
            self.take(TOOL_CALL_END.len());
            self.release(out);
            return true;
        }
        if TOOL_CALL_END.starts_with(self.pending.as_str()) { return false; }
        self.mode = Mode::Lost("closing tag missing before </tool_call>");
        true
    }

    /// Drop stray text between arguments up to the call's next tag.
    fn skip(&mut self, reason: &'static str) {
        let call = self.call.as_mut().expect("a call is open");
        call.key = None;
        tracing::warn!(tool = %call.name, reason, "Qwen tool call text dropped");
        self.mode = Mode::Skip;
    }

    fn step_skip(&mut self) -> bool {
        if let Some((index, _)) = first_of(&self.pending, &CALL_TAGS) {
            self.take(index);
            self.mode = Mode::Arguments;
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
            tracing::warn!(reason, text = %excerpt(&call.text),
                "Qwen tool call streamed, then found unreadable");
        } else {
            tracing::warn!(reason, text = %excerpt(&call.text), "Qwen tool call returned as content");
            self.held_whitespace = call.before;
            self.emit_content(&call.text, out);
        }
        self.mode = Mode::Content;
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
            for (index, (_, text)) in call.arguments.into_iter().enumerate() {
                let text = if index > 0 { format!(",{text}") } else { text };
                out.push(arguments(text));
            }
            out.push(arguments("}".into()));
        }
        self.calls += 1;
        self.mode = Mode::Content;
    }
}

impl TextParser for QwenOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> { QwenOutputParser::push(self, text) }
    fn finish(&mut self) -> Vec<OutputChunk> { QwenOutputParser::finish(self) }
    fn stop(&self) -> Option<&GlmStop> { QwenOutputParser::stop(self) }
    fn tool_calls(&self) -> usize { QwenOutputParser::tool_calls(self) }
}

fn arguments(content: String) -> OutputChunk { OutputChunk::ToolArgumentsDelta { content } }

/// `content` as a JSON string body, for a schema-typed (string) parameter.
fn json_string_contents(value: &str) -> String {
    let quoted = serde_json::to_string(value).expect("string serializes");
    quoted[1..quoted.len() - 1].to_owned()
}

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

/// A short excerpt of model text for a log line.
fn excerpt(text: &str) -> String {
    match text.char_indices().nth(80) {
        Some((cut, _)) => format!("{:?}...", &text[..cut]),
        None => format!("{text:?}"),
    }
}

/// A non-string parameter: JSON, else `true`/`false` in any case, else the text.
fn typed_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) { return value; }
    match trimmed.to_ascii_lowercase().as_str() {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ => Value::String(raw.to_owned()),
    }
}

/// Whether `tool.key`'s value is raw text: an undeclared tool or parameter, a
/// schema without type information, or one that admits a string.
pub fn value_is_string(options: &QwenParserOptions, tool: &str, key: &str) -> bool {
    let Some(parameters) = options.tools.as_deref().unwrap_or_default().iter()
        .find(|definition| definition.name == tool).map(|definition| &definition.parameters) else { return true };
    let Some(schema) = parameters.get("properties").and_then(|properties| properties.get(key)) else { return true };
    let schema = resolve(schema, parameters, 0);
    let typed = ["type", "anyOf", "oneOf", "allOf", "enum", "const"].iter().any(|k| schema.get(*k).is_some());
    !typed || accepts_string(schema, parameters, 0)
}

fn resolve<'a>(schema: &'a Value, root: &'a Value, depth: usize) -> &'a Value {
    match schema.get("$ref").and_then(Value::as_str).and_then(|r| r.strip_prefix('#')) {
        Some(pointer) if depth < 32 => root.pointer(pointer).map_or(schema, |target| resolve(target, root, depth + 1)),
        _ => schema,
    }
}

fn accepts_string(schema: &Value, root: &Value, depth: usize) -> bool {
    if depth >= 32 { return false; }
    let schema = resolve(schema, root, depth);
    let direct = match schema.get("type") {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some("string")),
        _ => false,
    };
    direct
        || ["anyOf", "oneOf", "allOf"].into_iter()
            .filter_map(|key| schema.get(key).and_then(Value::as_array))
            .flatten()
            .any(|option| accepts_string(option, root, depth + 1))
        || schema.get("enum").and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty() && values.iter().all(Value::is_string))
        || schema.get("const").is_some_and(Value::is_string)
}
