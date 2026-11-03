//! Incremental Muse output parser: recipient-addressed messages emitted as
//! protocol-neutral [`OutputChunk`]s.
//!
//! ```text
//!  to=self<|message|>REASONING<|eom|>
//! <|start|>assistant to=commentary<|message|>VISIBLE<|eom|>
//! <|start|>assistant to=NAME<|message|><atem:function_calls>
//! <atem:invoke name="NAME">
//! <atem:parameter name="KEY">VALUE</atem:parameter>
//! </atem:invoke>
//! </atem:function_calls><|eot|>
//! ```
//!
//! - `self` streams as reasoning; `commentary` and `user` stream as visible
//!   content (consecutive visible messages are joined by a blank line); any
//!   other recipient is a tool message, held until it ends and released only
//!   when its `<atem:function_calls>` block reads completely (the qwen4
//!   parser's hold-until-close contract).
//! - A tool message that cannot be read (block not closed, markup in a name,
//!   a parameter without a name, a duplicated parameter, an undeclared tool)
//!   returns its text as content and never counts as a call; each is logged.
//! - With no tools declared, a tool-addressed message ends the output.
//! - Values follow the template's renderer: a parameter whose declared schema
//!   admits a string is the raw text; other values parse as JSON (the
//!   template prints booleans, null, numbers and JSON objects/arrays that way)
//!   and stay text when they do not. Attribute names are XML-unescaped as the
//!   template escaped them; values are left as written.
//! - `<|eot|>`/`<|end_of_text|>` end the output. They are also stop tokens,
//!   so the engine usually ends first: `finish` then closes the open message
//!   (a tool message whose block is complete still releases).
//! - Visible content after the first parsed call is dropped, as for Qwen.
//!
//! Results are independent of how the text is chunked.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::Value;

use super::{END_OF_MESSAGE, END_OF_TEXT, END_OF_TURN, MESSAGE, START};
use crate::openai::chat::glm5::parser::{accepts_string, GlmStop};
use crate::openai::chat::glm5::TextParser;

const CALLS: &str = "<atem:function_calls>";
const CALLS_END: &str = "</atem:function_calls>";
const INVOKE: &str = "<atem:invoke name=\"";
const INVOKE_END: &str = "</atem:invoke>";
const PARAMETER: &str = "<atem:parameter name=\"";
const PARAMETER_END: &str = "</atem:parameter>";
/// A header longer than this without `<|message|>` is not a header.
const MAX_HEADER: usize = 1024;

#[derive(Debug, Clone, Default)]
pub struct MuseParserOptions {
    /// Declared tools; `None` makes a tool-addressed message end the output.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Client stop sequences, matched in visible content only.
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Recipient { Reasoning, Visible, Tool(String) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode { Header, Body, Done }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker { EndOfMessage, Start, EndOfTurn, Sequence }

#[derive(Debug)]
pub struct MuseOutputParser {
    options: MuseParserOptions,
    mode: Mode,
    pending: String,
    recipient: Recipient,
    /// The open tool message's text.
    body: String,
    calls: usize,
    content_started: bool,
    held_whitespace: String,
    stop: Option<GlmStop>,
}

impl MuseOutputParser {
    pub fn new(mut options: MuseParserOptions) -> Self {
        options.stop_sequences.retain(|sequence| !sequence.is_empty());
        Self { options, mode: Mode::Header, pending: String::new(), recipient: Recipient::Visible, body: String::new(),
            calls: 0, content_started: false, held_whitespace: String::new(), stop: None }
    }

    pub fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        if self.mode == Mode::Done { return Vec::new(); }
        self.pending.push_str(text);
        let mut out = Vec::new();
        while self.step(false, &mut out) {}
        out
    }

    /// Flush at end of output: the open message closes (a complete tool block
    /// releases), a dangling header is dropped.
    pub fn finish(&mut self) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        if self.mode != Mode::Done {
            while self.step(true, &mut out) {}
            match self.mode {
                Mode::Body => self.end_message(&mut out),
                Mode::Header if !self.pending.trim().is_empty() =>
                    tracing::warn!(text = %excerpt(&self.pending), "Muse output ended inside a message header"),
                _ => {}
            }
            self.pending.clear();
            self.mode = Mode::Done;
        }
        self.held_whitespace.clear();
        out
    }

    pub fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }
    pub fn tool_calls(&self) -> usize { self.calls }

    fn step(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        match self.mode {
            Mode::Header => self.step_header(),
            Mode::Body => self.step_body(finishing, out),
            Mode::Done => false,
        }
    }

    /// Up to `<|message|>`: the recipient (` to=NAME`, after an optional
    /// `<|start|>assistant`). A turn end before it ends the output.
    fn step_header(&mut self) -> bool {
        let message = self.pending.find(MESSAGE);
        let end = [END_OF_TURN, END_OF_TEXT].iter().filter_map(|marker| self.pending.find(marker)
            .map(|index| (index, *marker))).min();
        if let Some((index, marker)) = end.filter(|(index, _)| message.is_none_or(|at| *index < at)) {
            let dropped = self.pending[..index].trim();
            if !dropped.is_empty() && dropped != START {
                tracing::warn!(text = %excerpt(dropped), "Muse turn ended inside a message header");
            }
            self.halt(GlmStop::Marker(marker.to_owned()));
            return false;
        }
        let Some(index) = message else {
            if self.pending.len() > MAX_HEADER {
                // Not a header: read it as a visible message so no text is lost.
                tracing::warn!(text = %excerpt(&self.pending), "Muse message without a header");
                self.open(Recipient::Visible);
                return true;
            }
            return false;
        };
        let header: String = self.pending.drain(..index + MESSAGE.len()).collect();
        let recipient = recipient(&header[..index]);
        if let Recipient::Tool(name) = &recipient {
            if self.options.tools.is_none() {
                self.halt(GlmStop::Marker(format!("to={name}")));
                return false;
            }
        }
        self.open(recipient);
        true
    }

    fn open(&mut self, recipient: Recipient) {
        if recipient == Recipient::Visible && self.content_started { self.held_whitespace.push_str("\n\n"); }
        self.recipient = recipient;
        self.body.clear();
        self.mode = Mode::Body;
    }

    fn markers(&self) -> Vec<(&str, Marker)> {
        let mut markers = vec![(END_OF_MESSAGE, Marker::EndOfMessage), (START, Marker::Start),
            (END_OF_TURN, Marker::EndOfTurn), (END_OF_TEXT, Marker::EndOfTurn)];
        if self.recipient == Recipient::Visible {
            markers.extend(self.options.stop_sequences.iter().map(|s| (s.as_str(), Marker::Sequence)));
        }
        markers
    }

    fn step_body(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
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
            self.route(text, out);
            return true;
        };
        let marker = marker.to_owned();
        let text: String = self.pending.drain(..index).collect();
        self.pending.drain(..length);
        self.route(text, out);
        match kind {
            Marker::EndOfMessage => { self.end_message(out); self.mode = Mode::Header; }
            Marker::Start => {
                // A message that never ended: close it and read the next header.
                self.end_message(out);
                self.mode = Mode::Header;
            }
            Marker::EndOfTurn => { self.end_message(out); self.halt(GlmStop::Marker(marker)); }
            Marker::Sequence => self.halt(GlmStop::Sequence(marker)),
        }
        self.mode != Mode::Done
    }

    fn route(&mut self, text: String, out: &mut Vec<OutputChunk>) {
        if text.is_empty() { return; }
        match self.recipient {
            Recipient::Reasoning => out.push(OutputChunk::Reasoning { content: text }),
            Recipient::Visible => if self.calls == 0 { self.emit_content(&text, out) },
            Recipient::Tool(_) => self.body.push_str(&text),
        }
    }

    /// Close the open message: a tool message releases or returns as content.
    fn end_message(&mut self, out: &mut Vec<OutputChunk>) {
        let Recipient::Tool(recipient) = std::mem::replace(&mut self.recipient, Recipient::Visible) else { return };
        let body = std::mem::take(&mut self.body);
        let tools = self.options.tools.as_deref().unwrap_or_default();
        match read_calls(&body).and_then(|calls| {
            for (name, _) in &calls {
                if !tools.iter().any(|tool| &tool.name == name) { return Err("undeclared tool"); }
            }
            Ok(calls)
        }) {
            Ok(calls) => {
                for (name, arguments) in calls {
                    if name != recipient && recipient != "tool" {
                        tracing::debug!(recipient, name, "Muse tool message recipient differs from its invoke name");
                    }
                    let mut fields = Vec::with_capacity(arguments.len());
                    for (key, raw) in arguments {
                        let value = value(&name, &key, &raw, tools);
                        fields.push(format!("{}:{}", Value::String(key), value));
                    }
                    out.push(OutputChunk::ToolCall { tool_name: name, arguments: "{".into() });
                    for (index, field) in fields.into_iter().enumerate() {
                        let field = if index > 0 { format!(",{field}") } else { field };
                        out.push(OutputChunk::ToolArgumentsDelta { content: field });
                    }
                    out.push(OutputChunk::ToolArgumentsDelta { content: "}".into() });
                    self.calls += 1;
                }
            }
            Err(reason) => {
                tracing::warn!(reason, recipient, text = %excerpt(&body), "Muse tool message returned as content");
                if self.calls == 0 {
                    if self.content_started { self.held_whitespace.push_str("\n\n"); }
                    self.emit_content(&body, out);
                }
            }
        }
    }

    fn halt(&mut self, reason: GlmStop) {
        self.stop = Some(reason);
        self.pending.clear();
        self.held_whitespace.clear();
        self.body.clear();
        self.mode = Mode::Done;
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
}

impl TextParser for MuseOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> { MuseOutputParser::push(self, text) }
    fn finish(&mut self) -> Vec<OutputChunk> { MuseOutputParser::finish(self) }
    fn stop(&self) -> Option<&GlmStop> { MuseOutputParser::stop(self) }
    fn tool_calls(&self) -> usize { MuseOutputParser::tool_calls(self) }
}

/// The recipient of a header (`<|start|>assistant to=NAME`, or ` to=NAME`
/// for the turn's first message); no `to=` reads as the user.
fn recipient(header: &str) -> Recipient {
    let header = header.trim();
    let header = header.strip_prefix(START).unwrap_or(header).trim_start();
    let header = header.strip_prefix("assistant").unwrap_or(header).trim();
    match header.strip_prefix("to=").map(str::trim) {
        Some("self") => Recipient::Reasoning,
        Some("commentary" | "user") | None | Some("") => Recipient::Visible,
        Some(name) => Recipient::Tool(name.to_owned()),
    }
}

type Calls = Vec<(String, Vec<(String, String)>)>;

/// Every invoke of a complete `<atem:function_calls>` block, in order.
fn read_calls(body: &str) -> Result<Calls, &'static str> {
    let mut rest = body.trim().strip_prefix(CALLS).ok_or("no <atem:function_calls> block")?;
    let mut calls = Vec::new();
    loop {
        rest = rest.trim_start();
        if let Some(tail) = rest.strip_prefix(CALLS_END) {
            if !tail.trim().is_empty() { return Err("text after the call block"); }
            if calls.is_empty() { return Err("empty call block"); }
            return Ok(calls);
        }
        let tail = rest.strip_prefix(INVOKE).ok_or("closing tag missing")?;
        let (name, tail) = attribute(tail)?;
        let mut arguments: Vec<(String, String)> = Vec::new();
        rest = tail;
        loop {
            rest = rest.trim_start();
            if let Some(tail) = rest.strip_prefix(INVOKE_END) { rest = tail; break; }
            let tail = rest.strip_prefix(PARAMETER).ok_or("closing tag missing")?;
            let (key, tail) = attribute(tail)?;
            let end = tail.find(PARAMETER_END).ok_or("closing tag missing")?;
            if arguments.iter().any(|(seen, _)| *seen == key) { return Err("duplicated parameter"); }
            arguments.push((key, tail[..end].to_owned()));
            rest = &tail[end + PARAMETER_END.len()..];
        }
        calls.push((name, arguments));
    }
}

/// A quoted attribute value up to `">`, XML-unescaped; markup refuses.
fn attribute(text: &str) -> Result<(String, &str), &'static str> {
    let end = text.find("\">").ok_or("unterminated name")?;
    let raw = &text[..end];
    if raw.is_empty() { return Err("empty name"); }
    if raw.contains(['<', '>', '"', '\n']) { return Err("markup in a name"); }
    let name = raw.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'")
        .replace("&amp;", "&");
    Ok((name, &text[end + 2..]))
}

fn value(tool: &str, key: &str, raw: &str, tools: &[ToolDefinition]) -> Value {
    if accepts_string(tool, key, tools) { return Value::String(raw.to_owned()); }
    let trimmed = raw.trim();
    serde_json::from_str(trimmed).unwrap_or_else(|_| Value::String(raw.to_owned()))
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

fn excerpt(text: &str) -> String {
    match text.char_indices().nth(80) {
        Some((cut, _)) => format!("{:?}...", &text[..cut]),
        None => format!("{text:?}"),
    }
}
