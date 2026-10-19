//! Incremental XTML channels. Calls remain buffered until their closing tag,
//! so malformed or truncated calls cannot become partial OpenAI tool calls.
use super::{close, open, END, OPEN, SEP};
use crate::openai::chat::glm5::{parser::GlmStop, TextParser};
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{Map, Value};

#[derive(Debug, Clone, Default)]
pub struct KimiParserOptions {
    pub thinking: bool,
    pub tools: Option<Vec<ToolDefinition>>,
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Reasoning,
    ResponseOpen,
    Content,
    AfterResponse,
    Calls,
    Call,
    AfterTools,
    End,
    Done,
}

#[derive(Debug)]
pub struct KimiOutputParser {
    options: KimiParserOptions,
    mode: Mode,
    pending: String,
    calls: usize,
    stop: Option<GlmStop>,
}

impl KimiOutputParser {
    pub fn new(mut options: KimiParserOptions) -> Self {
        options.stop_sequences.retain(|s| !s.is_empty());
        let mode = if options.thinking {
            Mode::Reasoning
        } else {
            Mode::Content
        };
        Self {
            options,
            mode,
            pending: String::new(),
            calls: 0,
            stop: None,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        if self.mode == Mode::Done {
            return Vec::new();
        }
        self.pending.push_str(text);
        let mut out = Vec::new();
        while self.step(false, &mut out) {}
        out
    }

    pub fn finish(&mut self) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        while self.step(true, &mut out) {}
        if self.mode != Mode::Done && !self.pending.is_empty() {
            // Preserve malformed/truncated wire text rather than claiming a call.
            out.push(OutputChunk::Raw {
                content: std::mem::take(&mut self.pending),
            });
        }
        self.mode = Mode::Done;
        out
    }

    pub fn stop(&self) -> Option<&GlmStop> {
        self.stop.as_ref()
    }
    pub fn tool_calls(&self) -> usize {
        self.calls
    }

    fn consume(&mut self, prefix: &str, mode: Mode) -> bool {
        if !self.pending.starts_with(prefix) {
            return false;
        }
        self.pending.drain(..prefix.len());
        self.mode = mode;
        true
    }

    fn halt(&mut self, marker: String, sequence: bool) {
        self.stop = Some(if sequence {
            GlmStop::Sequence(marker)
        } else {
            GlmStop::Marker(marker)
        });
        self.mode = Mode::Done;
        self.pending.clear();
    }

    fn step(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        match self.mode {
            Mode::Reasoning | Mode::Content => {
                let reasoning = self.mode == Mode::Reasoning;
                let boundary = close(if reasoning { "think" } else { "response" });
                let mut markers = vec![(boundary.as_str(), false), (END, false)];
                if !reasoning {
                    markers.extend(self.options.stop_sequences.iter().map(|s| (s.as_str(), true)));
                }
                if let Some((at, marker, sequence)) = markers
                    .iter()
                    .filter_map(|(marker, sequence)| self.pending.find(marker).map(|at| (at, *marker, *sequence)))
                    .min_by(|a, b| a.0.cmp(&b.0).then(b.1.len().cmp(&a.1.len())))
                {
                    let marker = marker.to_owned();
                    let text = self.pending.drain(..at).collect::<String>();
                    self.pending.drain(..marker.len());
                    if !text.is_empty() {
                        out.push(if reasoning {
                            OutputChunk::Reasoning { content: text }
                        } else {
                            OutputChunk::Raw { content: text }
                        });
                    }
                    if marker == boundary {
                        self.mode = if reasoning {
                            Mode::ResponseOpen
                        } else {
                            Mode::AfterResponse
                        };
                    } else {
                        self.halt(marker, sequence);
                    }
                    return true;
                }
                let held = if finishing {
                    0
                } else {
                    markers
                        .iter()
                        .map(|(marker, _)| held_prefix(&self.pending, marker))
                        .max()
                        .unwrap_or(0)
                };
                let emit = self.pending.len() - held;
                if emit == 0 {
                    return false;
                }
                let text = self.pending.drain(..emit).collect();
                out.push(if reasoning {
                    OutputChunk::Reasoning { content: text }
                } else {
                    OutputChunk::Raw { content: text }
                });
                true
            }
            Mode::ResponseOpen => self.consume(&open("response", &[]), Mode::Content),
            Mode::AfterResponse => {
                if self.consume(&close("message"), Mode::End) {
                    return true;
                }
                if self.pending.starts_with(&open("tools", &[])) && self.options.tools.is_none() {
                    self.halt(open("tools", &[]), false);
                    return true;
                }
                self.consume(&open("tools", &[]), Mode::Calls)
            }
            Mode::Calls => {
                if self.consume(&close("tools"), Mode::AfterTools) {
                    return true;
                }
                if self.pending.starts_with(&format!("{OPEN}call ")) {
                    self.mode = Mode::Call;
                    return true;
                }
                false
            }
            Mode::Call => {
                let end = close("call");
                let Some(at) = self.pending.find(&end) else {
                    return false;
                };
                let wire: String = self.pending.drain(..at + end.len()).collect();
                if let Some((name, arguments)) = parse_call(&wire) {
                    out.push(OutputChunk::ToolCall {
                        tool_name: name,
                        arguments,
                    });
                    self.calls += 1;
                } else {
                    tracing::warn!("malformed Kimi XTML call returned as content");
                    out.push(OutputChunk::Raw { content: wire });
                }
                self.mode = Mode::Calls;
                true
            }
            Mode::AfterTools => self.consume(&close("message"), Mode::End),
            Mode::End => {
                if self.pending.starts_with(END) {
                    self.halt(END.into(), false);
                    return true;
                }
                false
            }
            Mode::Done => false,
        }
    }
}

fn held_prefix(text: &str, marker: &str) -> usize {
    (1..=text.len().min(marker.len()))
        .rev()
        .find(|&len| text.is_char_boundary(text.len() - len) && marker.starts_with(&text[text.len() - len..]))
        .unwrap_or(0)
}

fn unescape(text: &str) -> String {
    text.replace("&quot;", "\"").replace("&amp;", "&")
}

/// Read only quoted XTML attributes, not arbitrary XML or HTML entities.
fn attributes<'a>(text: &'a str, tag: &str) -> Option<(Vec<(String, String)>, &'a str)> {
    let mut rest = text.strip_prefix(OPEN)?.strip_prefix(tag)?;
    let mut attrs = Vec::new();
    loop {
        if let Some(rest) = rest.strip_prefix(SEP) {
            return Some((attrs, rest));
        }
        rest = rest.strip_prefix(' ')?;
        let equal = rest.find("=\"")?;
        let key = &rest[..equal];
        if key.is_empty() || !key.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
            return None;
        }
        rest = &rest[equal + 2..];
        let end = rest.find('"')?;
        if attrs.iter().any(|(name, _)| name == key) {
            return None;
        }
        attrs.push((key.into(), unescape(&rest[..end])));
        rest = &rest[end + 1..];
    }
}

fn parse_call(wire: &str) -> Option<(String, String)> {
    let (attrs, mut rest) = attributes(wire, "call")?;
    let name = attrs.iter().find(|(key, _)| key == "tool")?.1.clone();
    let index = attrs.iter().find(|(key, _)| key == "index")?.1.parse::<usize>().ok()?;
    if attrs.len() != 2 || name.is_empty() || index == 0 {
        return None;
    }
    let mut values = Map::new();
    if rest.starts_with(&format!("{OPEN}json")) {
        let (attrs, value) = attributes(rest, "json")?;
        if attrs != [("type".into(), "object".into())] {
            return None;
        }
        let end = value.find(&close("json"))?;
        let raw = crate::openai::tools::unique_json(&value[..end]).ok()?;
        values = raw.as_object()?.clone();
        rest = &value[end + close("json").len()..];
    } else {
        while rest.starts_with(&format!("{OPEN}argument")) {
            let (attrs, text) = attributes(rest, "argument")?;
            if attrs.len() != 2 {
                return None;
            }
            let key = attrs.iter().find(|(key, _)| key == "key")?.1.clone();
            let kind = &attrs.iter().find(|(key, _)| key == "type")?.1;
            let end = text.find(&close("argument"))?;
            let text_value = &text[..end];
            let value = if kind == "string" {
                Value::String(text_value.into())
            } else {
                let value = crate::openai::tools::unique_json(text_value).ok()?;
                if super::kind(&value) != kind {
                    return None;
                }
                value
            };
            if values.insert(key, value).is_some() {
                return None;
            }
            rest = &text[end + close("argument").len()..];
        }
    }
    if rest != close("call") {
        return None;
    }
    Some((name, Value::Object(values).to_string()))
}

impl TextParser for KimiOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        Self::push(self, text)
    }
    fn finish(&mut self) -> Vec<OutputChunk> {
        Self::finish(self)
    }
    fn stop(&self) -> Option<&GlmStop> {
        Self::stop(self)
    }
    fn tool_calls(&self) -> usize {
        Self::tool_calls(self)
    }
}
