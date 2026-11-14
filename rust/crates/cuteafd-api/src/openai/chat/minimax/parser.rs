//! Incremental adaptive reasoning and MiniMax's namespaced, recursive XML.
//! Calls are held until the entire block closes, so malformed or length-cut
//! XML never becomes a partial API tool call. Chunk boundaries are immaterial.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{Map, Value};
use super::{CALL_CLOSE, CALL_OPEN, NAMESPACE, STOP_MARKERS, THINK_CLOSE, THINK_OPEN};
use crate::openai::chat::glm5::{parser::GlmStop, TextParser};
use crate::openai::chat::stream_args::held_prefix;

#[derive(Debug, Clone, Default)]
pub struct MiniMaxParserOptions {
    pub tools: Option<Vec<ToolDefinition>>,
    pub stop_sequences: Vec<String>,
}

#[derive(Debug)]
pub struct MiniMaxOutputParser {
    options: MiniMaxParserOptions,
    pending: String,
    reasoning: bool,
    in_call: bool,
    done: bool,
    calls: usize,
    stop: Option<GlmStop>,
}

impl MiniMaxOutputParser {
    pub fn new(mut options: MiniMaxParserOptions) -> Self {
        options.stop_sequences.retain(|s| !s.is_empty());
        Self { options, pending: String::new(), reasoning: false, in_call: false,
            done: false, calls: 0, stop: None }
    }

    fn emit(&self, content: String, out: &mut Vec<OutputChunk>) {
        if content.is_empty() { return; }
        out.push(if self.reasoning { OutputChunk::Reasoning { content } } else { OutputChunk::Raw { content } });
    }

    fn step(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        if self.done { return false; }
        if self.in_call {
            if let Some(at) = self.pending.find(CALL_CLOSE) {
                let end = at + CALL_CLOSE.len();
                let wire: String = self.pending.drain(..end).collect();
                match parse_calls(&wire[CALL_OPEN.len()..at], self.options.tools.as_deref().unwrap_or_default()) {
                    Ok(calls) => {
                        self.calls += calls.len();
                        out.extend(calls.into_iter().map(|(tool_name, arguments)| OutputChunk::ToolCall { tool_name, arguments }));
                    }
                    Err(error) => {
                        tracing::warn!(error, "malformed MiniMax tool block returned as content");
                        self.emit(wire, out);
                    }
                }
                self.in_call = false;
                return true;
            }
            // A token stop may terminate a truncated call before finish().
            if finishing || STOP_MARKERS.iter().any(|marker| self.pending.contains(marker)) {
                self.in_call = false;
                let stop = STOP_MARKERS.iter().filter_map(|marker| self.pending.find(marker)).min().unwrap_or(self.pending.len());
                let text: String = self.pending.drain(..stop).collect();
                self.emit(text, out);
                return true;
            }
            return false;
        }
        let mut markers: Vec<(&str, u8)> = vec![(THINK_OPEN, 0), (THINK_CLOSE, 1)];
        markers.extend(STOP_MARKERS.iter().map(|marker| (*marker, 2)));
        if self.options.tools.is_none() {
            markers.extend([(NAMESPACE, 2), ("<tool_call>", 2), ("</tool_call>", 2)]);
        } else if !self.reasoning { markers.push((CALL_OPEN, 3)); }
        if !self.reasoning { markers.extend(self.options.stop_sequences.iter().map(|s| (s.as_str(), 4))); }
        let found = markers.iter().filter_map(|(marker, kind)| self.pending.find(marker)
            .map(|at| (at, marker.len(), *kind, (*marker).to_owned())))
            .min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let Some((at, length, kind, marker)) = found else {
            let keep = if finishing { 0 } else { markers.iter().map(|(m, _)| held_prefix(&self.pending, m)).max().unwrap_or(0) };
            let length = self.pending.len() - keep;
            if length == 0 { return false; }
            let text = self.pending.drain(..length).collect();
            self.emit(text, out);
            return true;
        };
        let text = self.pending.drain(..at).collect();
        self.emit(text, out);
        match kind {
            3 => { self.in_call = true; }
            _ => {
                self.pending.drain(..length);
                match kind {
                    0 => self.reasoning = true,
                    1 => self.reasoning = false,
                    _ => {
                        self.stop = Some(if kind == 4 { GlmStop::Sequence(marker) } else { GlmStop::Marker(marker) });
                        self.done = true;
                        self.pending.clear();
                    }
                }
            }
        }
        true
    }
}

impl TextParser for MiniMaxOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        if self.done { return Vec::new(); }
        self.pending.push_str(text);
        let mut out = Vec::new();
        while self.step(false, &mut out) {}
        out
    }
    fn finish(&mut self) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        while self.step(true, &mut out) {}
        self.done = true;
        out
    }
    fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }
    fn tool_calls(&self) -> usize { self.calls }
}

fn parse_calls(mut text: &str, tools: &[ToolDefinition]) -> Result<Vec<(String, String)>, &'static str> {
    let mut calls = Vec::new();
    while !text.trim().is_empty() {
        text = text.trim_start();
        text = text.strip_prefix(NAMESPACE).and_then(|t| t.strip_prefix("<invoke name=\""))
            .ok_or("expected namespaced invoke")?;
        let at = text.find("\">").ok_or("invoke name is incomplete")?;
        let name = &text[..at];
        if name.is_empty() || name.contains(['<', '>', '"', '\n', '\r']) { return Err("invalid invoke name"); }
        let tool = tools.iter().find(|tool| tool.name == name).ok_or("undeclared tool name")?;
        text = &text[at + 2..];
        let close = format!("{NAMESPACE}</invoke>");
        let end = text.find(&close).ok_or("invoke is incomplete")?;
        let nodes = nodes(&text[..end], 0)?;
        let value = object(nodes, &tool.parameters, &tool.parameters, 0)?;
        calls.push((name.to_owned(), value.to_string()));
        text = &text[end + close.len()..];
    }
    if calls.is_empty() { return Err("tool block is empty"); }
    Ok(calls)
}

#[derive(Debug)]
struct Node<'a> { name: &'a str, raw: &'a str, children: Vec<Node<'a>> }

fn nodes(mut text: &str, depth: usize) -> Result<Vec<Node<'_>>, &'static str> {
    if depth > 64 { return Err("XML nesting exceeds 64"); }
    let mut result = Vec::new();
    while !text.is_empty() {
        text = text.strip_prefix(NAMESPACE).and_then(|t| t.strip_prefix('<')).ok_or("expected namespaced argument")?;
        let end = text.find('>').ok_or("argument name is incomplete")?;
        let name = &text[..end];
        if name.is_empty() || name.contains(['<', '/', '"', ' ', '\n', '\r']) { return Err("invalid argument name"); }
        text = &text[end + 1..];
        // Balance same-name nested elements instead of stopping at the first close.
        let open = format!("{NAMESPACE}<{name}>");
        let close = format!("{NAMESPACE}</{name}>");
        let mut cursor = 0;
        let mut level = 1;
        let finish = loop {
            let next_close = text[cursor..].find(&close).map(|at| cursor + at).ok_or("argument is incomplete")?;
            if let Some(next_open) = text[cursor..].find(&open).map(|at| cursor + at).filter(|at| *at < next_close) {
                level += 1;
                cursor = next_open + open.len();
            } else {
                level -= 1;
                if level == 0 { break next_close; }
                cursor = next_close + close.len();
            }
        };
        let raw = &text[..finish];
        let children = if raw.starts_with(NAMESPACE) { nodes(raw, depth + 1)? } else {
            if raw.contains(NAMESPACE) { return Err("mixed XML text and elements"); }
            Vec::new()
        };
        result.push(Node { name, raw, children });
        text = &text[finish + close.len()..];
    }
    Ok(result)
}

fn resolve<'a>(schema: &'a Value, root: &'a Value, depth: usize) -> &'a Value {
    if depth >= 64 { return schema; }
    match schema["$ref"].as_str().and_then(|r| r.strip_prefix('#')).and_then(|r| root.pointer(r)) {
        Some(target) => resolve(target, root, depth + 1),
        None => schema,
    }
}

fn object(nodes: Vec<Node<'_>>, schema: &Value, root: &Value, depth: usize) -> Result<Value, &'static str> {
    let schema = resolve(schema, root, 0);
    let mut result = Map::new();
    for node in nodes {
        if result.contains_key(node.name) { return Err("duplicate argument key"); }
        let key = node.name.to_owned();
        let property = schema["properties"].get(&key).unwrap_or(&Value::Null);
        result.insert(key, value(node, property, root, depth + 1)?);
    }
    Ok(Value::Object(result))
}

fn accepts_string(schema: &Value, root: &Value, depth: usize) -> bool {
    if depth > 64 { return true; }
    let schema = resolve(schema, root, 0);
    let direct = match &schema["type"] {
        Value::String(kind) => kind == "string",
        Value::Array(types) => types.iter().any(|kind| kind == "string"),
        _ => !["anyOf", "oneOf", "allOf", "enum", "const"].iter().any(|key| schema.get(*key).is_some()),
    };
    direct || ["anyOf", "oneOf"].iter().filter_map(|key| schema[*key].as_array())
        .flatten().any(|branch| accepts_string(branch, root, depth + 1))
        || schema["allOf"].as_array().is_some_and(|branches| branches.iter().all(|branch| accepts_string(branch, root, depth + 1)))
        || schema["enum"].as_array().is_some_and(|values| values.iter().any(Value::is_string))
        || schema["const"].is_string()
}

fn value(node: Node<'_>, schema: &Value, root: &Value, depth: usize) -> Result<Value, &'static str> {
    if depth > 64 { return Err("argument nesting exceeds 64"); }
    let schema = resolve(schema, root, 0);
    let kind = schema["type"].as_str();
    if kind == Some("array") || (!node.children.is_empty() && node.children.iter().all(|n| n.name == "item") && kind != Some("object")) {
        if node.children.iter().any(|node| node.name != "item") || (node.children.is_empty() && !node.raw.is_empty()) {
            return Err("array must contain item elements");
        }
        return node.children.into_iter().map(|node| value(node, &schema["items"], root, depth + 1))
            .collect::<Result<Vec<_>, _>>().map(Value::Array);
    }
    if kind == Some("object") || !node.children.is_empty() {
        if node.children.is_empty() && !node.raw.is_empty() { return Err("object must contain elements"); }
        return object(node.children, schema, root, depth + 1);
    }
    // XML carries no scalar type: preserve strings unless the declared schema
    // excludes them. Never turn a numeric-looking string into a number.
    if accepts_string(schema, root, 0) { return Ok(Value::String(node.raw.to_owned())); }
    let parsed: Value = serde_json::from_str(node.raw).map_err(|_| "invalid typed XML scalar")?;
    if parsed.is_array() || parsed.is_object() { return Err("structured arguments require XML elements"); }
    Ok(parsed)
}
