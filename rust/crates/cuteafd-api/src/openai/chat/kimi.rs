//! Native Kimi K3 XTML wire format from the checkpoint's encoding_k3.py.
//! Checkpoint Python is a CPU test oracle only, never part of request serving.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

pub mod parser;
pub use parser::{KimiOutputParser, KimiParserOptions};

pub const OPEN: &str = "<|open|>";
pub const CLOSE: &str = "<|close|>";
pub const SEP: &str = "<|sep|>";
pub const END: &str = "<|end_of_msg|>";

#[derive(Debug)]
pub struct KimiEncoding {
    eos: Vec<u32>,
}

impl KimiEncoding {
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let config: Value = serde_json::from_slice(&std::fs::read(snapshot.join("tokenizer_config.json"))?)?;
        let tokens = config["added_tokens_decoder"]
            .as_object()
            .context("Kimi added tokens missing")?;
        for (name, expected) in [(OPEN, 163587), (CLOSE, 163588), (SEP, 163589), (END, 163586)] {
            ensure!(
                tokens
                    .get(&expected.to_string())
                    .is_some_and(|token| token["content"] == name),
                "Kimi XTML token contract differs: {name}"
            );
        }
        let generation: Value = serde_json::from_slice(&std::fs::read(snapshot.join("generation_config.json"))?)?;
        let eos = match &generation["eos_token_id"] {
            Value::Number(id) => vec![u32::try_from(id.as_u64().context("Kimi EOS")?)?],
            Value::Array(ids) => ids
                .iter()
                .map(|id| Ok(u32::try_from(id.as_u64().context("Kimi EOS")?)?))
                .collect::<Result<Vec<_>>>()?,
            _ => bail!("Kimi EOS missing"),
        };
        ensure!(eos.contains(&163586), "Kimi end-of-message EOS missing");
        Ok(Self { eos })
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        self.eos.clone()
    }

    pub fn render(&self, body: &Value, thinking: bool) -> Result<String, String> {
        render(body, thinking).map_err(|error| format!("Kimi XTML: {error:#}"))
    }

    pub(crate) fn render_literals(
        &self,
        body: &Value,
        thinking: bool,
        literal: &mut dyn FnMut(&str) -> String,
    ) -> Result<String, String> {
        render_literals(body, thinking, literal).map_err(|error| format!("Kimi XTML: {error:#}"))
    }
}

fn attr(text: &str) -> String {
    text.replace('&', "&amp;").replace('"', "&quot;")
}
fn open(tag: &str, attrs: &[(&str, String)]) -> String {
    let mut out = format!("{OPEN}{tag}");
    for (key, value) in attrs {
        out.push_str(&format!(" {key}=\"{}\"", attr(value)));
    }
    out.push_str(SEP);
    out
}
fn close(tag: &str) -> String {
    format!("{CLOSE}{tag}{SEP}")
}
fn open_literals(tag: &str, attrs: &[(&str, String)], literal: &mut dyn FnMut(&str) -> String) -> String {
    let mut out = format!("{OPEN}{tag}");
    for (key, value) in attrs {
        out.push_str(&format!(" {key}=\"{}\"", literal(&attr(value))));
    }
    out.push_str(SEP);
    out
}
fn internal(kind: &str, body: &str, literal: &mut dyn FnMut(&str) -> String) -> String {
    format!(
        "{}{}{}{END}",
        open("message", &[("role", "system".into()), ("type", kind.into())]),
        literal(body.trim()),
        close("message")
    )
}
fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            Value::Object(keys.into_iter().map(|key| (key.clone(), sorted(&map[key]))).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        _ => value.clone(),
    }
}
fn python_json(value: &Value) -> String {
    super::glm5::template::python_json(value, false)
}
fn compact_json(value: &Value) -> String {
    super::glm5::template::python_json(value, true)
}
fn kind(value: &Value) -> &'static str {
    match value {
        Value::String(_) => "string",
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Number(_) => "number",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    }
}
fn content(value: &Value) -> Result<String> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(|part| {
                ensure!(
                    matches!(part["type"].as_str(), Some("text")),
                    "Kimi media encoder unavailable"
                );
                Ok(part["text"].as_str().context("Kimi content text")?.to_owned())
            })
            .collect::<Result<Vec<_>>>()
            .map(|parts| parts.concat()),
        _ => bail!("Kimi message content must be text or text parts"),
    }
}
fn function(call: &Value) -> &Value {
    call.get("function").filter(|value| value.is_object()).unwrap_or(call)
}

/// Preserve raw non-string JSON literals, duplicate keys and trailing material,
/// just like the official one-level JSONDecoder.raw_decode loop.
fn raw_arguments(text: &str) -> Result<Vec<(String, &'static str, String)>> {
    fn whitespace(text: &str, at: &mut usize) {
        while text
            .as_bytes()
            .get(*at)
            .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
        {
            *at += 1;
        }
    }
    fn value(text: &str, at: &mut usize) -> Result<Value> {
        if text.as_bytes().get(*at) == Some(&b'"') {
            let first = *at;
            *at += 1;
            let mut escaped = false;
            let mut json = String::from("\"");
            for ch in text[*at..].chars() {
                *at += ch.len_utf8();
                if !escaped && ch == '"' {
                    json.push(ch);
                    return Ok(serde_json::from_str(&json)?);
                }
                // The official decoder uses strict=False for unescaped controls.
                if ch < ' ' {
                    json.push_str(&format!("\\u{:04x}", ch as u32));
                } else {
                    json.push(ch);
                }
                escaped = !escaped && ch == '\\';
            }
            bail!("unterminated JSON string at {first}");
        }
        let mut decoder = serde_json::Deserializer::from_str(&text[*at..]).into_iter::<Value>();
        let value = decoder.next().context("missing JSON argument")??;
        *at += decoder.byte_offset();
        Ok(value)
    }
    let mut at = 0;
    whitespace(text, &mut at);
    ensure!(text.as_bytes().get(at) == Some(&b'{'), "arguments not an object");
    at += 1;
    whitespace(text, &mut at);
    if text.as_bytes().get(at) == Some(&b'}') {
        return Ok(Vec::new());
    }
    let mut arguments = Vec::new();
    loop {
        let key = value(text, &mut at)?
            .as_str()
            .context("argument key not string")?
            .to_owned();
        whitespace(text, &mut at);
        ensure!(text.as_bytes().get(at) == Some(&b':'), "argument colon missing");
        at += 1;
        whitespace(text, &mut at);
        let first = at;
        let decoded = value(text, &mut at)?;
        arguments.push((
            key,
            kind(&decoded),
            decoded.as_str().unwrap_or(&text[first..at]).to_owned(),
        ));
        whitespace(text, &mut at);
        match text.as_bytes().get(at) {
            Some(b'}') => return Ok(arguments),
            Some(b',') => {
                at += 1;
                whitespace(text, &mut at);
            }
            _ => bail!("argument delimiter missing"),
        }
    }
}
#[cfg(test)]
fn arguments(value: &Value) -> Result<String> {
    arguments_literals(value, &mut str::to_owned)
}
fn arguments_literals(value: &Value, literal: &mut dyn FnMut(&str) -> String) -> Result<String> {
    let (arguments, raw) = match value {
        Value::Null => (Vec::new(), None),
        Value::String(text) if text.is_empty() => (Vec::new(), None),
        Value::String(text) => match raw_arguments(text) {
            Ok(arguments) => (arguments, None),
            Err(_) => (Vec::new(), Some(text.as_str())),
        },
        Value::Object(map) => (
            map.iter()
                .map(|(key, value)| {
                    (
                        key.clone(),
                        kind(value),
                        value.as_str().map(str::to_owned).unwrap_or_else(|| python_json(value)),
                    )
                })
                .collect(),
            None,
        ),
        _ => bail!("Kimi tool arguments must be an object or JSON object string"),
    };
    if let Some(raw) = raw {
        return Ok(format!(
            "{}{}{}",
            open("json", &[("type", "object".into())]),
            literal(raw),
            close("json")
        ));
    }
    Ok(arguments
        .into_iter()
        .map(|(key, kind, text)| {
            format!(
                "{}{}{}",
                open_literals("argument", &[("key", key), ("type", kind.into())], literal),
                literal(&text),
                close("argument")
            )
        })
        .collect())
}
fn declaration(tools: &Value, dynamic: bool, literal: &mut dyn FnMut(&str) -> String) -> String {
    let prefix = if dynamic {
        "## New Tools Available\nThe system dynamically extends the toolset via lazy-loading.\nYou have access to all existing and extended tools.\nHere are the specs for the extended tools.\n\n"
    } else {
        "# Tools\nHere are the available tools, described in JSONSchema.\n\n"
    };
    internal(
        "tool-declare",
        &format!("{prefix}```json\n{}\n```", compact_json(&sorted(tools))),
        literal,
    )
}
fn call_id(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        Value::Bool(value) => Some(if *value { "True" } else { "False" }.into()),
        _ => Some(value.to_string()),
    }
}
fn normalize_results(messages: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    let mut calls: Vec<Value> = Vec::new();
    let mut at = 0;
    while at < messages.len() {
        let message = &messages[at];
        if message["role"] == "assistant" {
            calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
        }
        if message["role"] != "tool" {
            out.push(message.clone());
            at += 1;
            continue;
        }
        let first = at;
        while at < messages.len() && messages[at]["role"] == "tool" {
            at += 1;
        }
        let mut run = messages[first..at]
            .iter()
            .map(|message| {
                let id = message
                    .get("tool_call_id")
                    .or_else(|| message.get("id"))
                    .and_then(call_id);
                (
                    calls
                        .iter()
                        .position(|call| id.is_some() && call.get("id").and_then(call_id) == id),
                    message.clone(),
                )
            })
            .collect::<Vec<_>>();
        if run.iter().all(|(position, _)| position.is_some()) {
            run.sort_by_key(|(position, _)| *position);
            for (position, message) in &mut run {
                if let Some(name) = function(&calls[position.unwrap()]).get("name") {
                    message["tool"] = name.clone();
                    if message.get("name").is_some() {
                        message["name"] = name.clone();
                    }
                }
            }
        }
        out.extend(run.into_iter().map(|(_, message)| message));
    }
    out
}
fn render(body: &Value, thinking: bool) -> Result<String> {
    render_literals(body, thinking, &mut str::to_owned)
}
fn render_literals(body: &Value, thinking: bool, literal: &mut dyn FnMut(&str) -> String) -> Result<String> {
    let messages = normalize_results(body["messages"].as_array().context("Kimi messages required")?);
    let mut out = String::new();
    if body["tools"].as_array().is_some_and(|tools| !tools.is_empty()) {
        out.push_str(&declaration(&body["tools"], false, literal));
    }
    let effort = body
        .get("chat_template_kwargs")
        .and_then(|kwargs| kwargs.get("thinking_effort"))
        .or_else(|| body.get("thinking_effort"))
        .or_else(|| body.get("reasoning_effort"))
        .or_else(|| body.get("reasoning").and_then(|reasoning| reasoning.get("effort")));
    if thinking {
        if let Some(effort) = effort.filter(|value| !value.is_null()) {
            let effort = effort.as_str().context("Kimi thinking effort must be a string")?;
            ensure!(
                matches!(effort, "low" | "high" | "max"),
                "Kimi thinking effort must be low, high or max"
            );
            out.push_str(&internal("thinking-effort", &format!("`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort={effort}`."), literal));
        }
    }
    let mut calls = Vec::new();
    let mut tool_index = 0;
    for message in &messages {
        let role = message["role"].as_str().context("Kimi message role missing")?;
        ensure!(
            matches!(role, "user" | "system" | "assistant" | "tool"),
            "unknown Kimi message role"
        );
        if role == "system" && message["tools"].as_array().is_some_and(|tools| !tools.is_empty()) {
            out.push_str(&declaration(&message["tools"], true, literal));
            continue;
        }
        let mut attrs = vec![("role", role.into())];
        if role == "tool" {
            tool_index += 1;
            let name = message
                .get("tool")
                .or_else(|| message.get("name"))
                .and_then(Value::as_str)
                .or_else(|| {
                    calls
                        .get(tool_index - 1)
                        .and_then(|call| function(call)["name"].as_str())
                })
                .context("Kimi tool result needs a resolvable tool name")?;
            attrs.extend([("tool", name.into()), ("index", tool_index.to_string())]);
        } else if let Some(name) = message["name"].as_str().filter(|name| !name.is_empty()) {
            attrs.push(("name", name.into()));
        }
        out.push_str(&open_literals("message", &attrs, literal));
        if role == "assistant" {
            calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
            tool_index = 0;
            if thinking {
                out.push_str(&open("think", &[]));
                let reasoning = message
                    .get("reasoning_content")
                    .filter(|value| value.as_str().is_some_and(|s| !s.is_empty()))
                    .or_else(|| message.get("reasoning"))
                    .unwrap_or(&Value::Null);
                out.push_str(&literal(&content(reasoning)?));
                out.push_str(&close("think"));
            }
            out.push_str(&open("response", &[]));
            out.push_str(&literal(&content(&message["content"])?));
            out.push_str(&close("response"));
            if !calls.is_empty() {
                out.push_str(&open("tools", &[]));
                for (index, call) in calls.iter().enumerate() {
                    let function = function(call);
                    out.push_str(&open_literals(
                        "call",
                        &[
                            ("tool", function["name"].as_str().context("Kimi tool name")?.into()),
                            ("index", (index + 1).to_string()),
                        ],
                        literal,
                    ));
                    out.push_str(&arguments_literals(&function["arguments"], literal)?);
                    out.push_str(&close("call"));
                }
                out.push_str(&close("tools"));
            }
        } else {
            out.push_str(&literal(&content(&message["content"])?));
        }
        out.push_str(&close("message"));
        out.push_str(END);
    }
    match body["tool_choice"].as_str() {
        Some("required") => out.push_str(&internal(
            "tool-choice",
            "The system is invoked with `tool_choice=required`.\nYou MUST call tools in the next message.",
            literal,
        )),
        Some("none") => out.push_str(&internal(
            "tool-choice",
            "The system is invoked with `tool_choice=none`.\nYou MUST NOT call any tools in the next message.",
            literal,
        )),
        _ => {}
    }
    match body["response_format"].get("type").unwrap_or(&body["response_format"]).as_str() {
        Some("json_object") => out.push_str(&internal("response-format", "The system is invoked with `response_format=json_object`.\nYour response must be raw JSON data without markdown code blocks (```json) or any additional formatting.", literal)),
        Some("json_schema") => {
            let schema = body.get("response_schema").unwrap_or_else(|| {
                let schema = &body["response_format"]["json_schema"];
                schema.get("schema").or_else(|| schema.get("json_schema")).unwrap_or(schema)
            });
            out.push_str(&internal("response-format", &format!("The system is invoked with `response_format=json_schema`.\nYour response must be raw JSON data without markdown code blocks (```json) or any additional formatting.\nThe JSON data must match the following schema:\n```json\n{}\n```", compact_json(&sorted(schema))), literal));
        }
        _ => {}
    }
    out.push_str(&open("message", &[("role", "assistant".into())]));
    out.push_str(&open(if thinking { "think" } else { "response" }, &[]));
    Ok(out)
}

#[cfg(test)]
mod tests;
