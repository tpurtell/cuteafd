use std::{collections::HashMap, sync::Arc};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::gateway::{session::Snapshot, turn::*, Gateway, GatewayError};

#[derive(Debug, Clone, PartialEq)]
pub(super) enum ToolKind {
    Function,
    Custom,
    LocalShell,
}

pub(super) struct Parsed {
    pub response_id: String,
    pub wire: Value,
    pub turn: TurnRequest,
    pub kinds: HashMap<String, ToolKind>,
    pub names: HashMap<String, (String, String)>,
    pub parent: Option<Arc<Snapshot>>,
    #[cfg(test)]
    pub new_items: Vec<Item>,
    pub new_stored: Vec<crate::gateway::session::StoredItem>,
    pub input: Vec<Value>,
    pub store: bool,
    pub encrypted: bool,
    pub summary: bool,
}

pub(super) fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, GatewayError> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| GatewayError::invalid(format!("'{key}' must be a string")).with_param(key))
}

fn optional_string(v: &Value, key: &str) -> Result<Option<String>, GatewayError> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        _ => Err(GatewayError::invalid(format!("'{key}' must be a string")).with_param(key)),
    }
}

fn boolean(v: &Value, key: &str, default: bool) -> Result<bool, GatewayError> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        _ => Err(GatewayError::invalid(format!("'{key}' must be a boolean")).with_param(key)),
    }
}

pub(super) fn parse(
    gateway: &Gateway,
    wire: Value,
    ephemeral: Option<(&str, Arc<Snapshot>)>,
) -> Result<Parsed, GatewayError> {
    if !wire.is_object() {
        return Err(GatewayError::invalid("request must be an object"));
    }
    let model = string(&wire, "model")?.to_owned();
    if boolean(&wire, "background", false)? {
        return Err(
            GatewayError::unsupported("background responses are not supported")
                .with_param("background"),
        );
    }
    boolean(&wire, "stream", false)?;
    let store = boolean(&wire, "store", true)?;
    let parent = if let Some(id) = optional_string(&wire, "previous_response_id")? {
        Some(
            gateway
                .sessions
                .response(&id)
                .or_else(|| {
                    ephemeral
                        .as_ref()
                        .filter(|(last, _)| *last == id)
                        .map(|(_, s)| s.clone())
                })
                .ok_or_else(|| {
                    GatewayError::not_found(format!("response '{id}' not found"))
                        .with_param("previous_response_id")
                })?,
        )
    } else {
        None
    };
    let history = parent.as_ref().map(|s| s.history()).unwrap_or_default();
    let mut input = match wire.get("input") {
        Some(Value::String(text)) => vec![
            json!({"type":"message","role":"user","content":[{"type":"input_text","text":text}]}),
        ],
        Some(Value::Array(items)) => items.clone(),
        None => Vec::new(),
        _ => {
            return Err(
                GatewayError::invalid("input must be a string or an array").with_param("input")
            )
        }
    };
    for entry in &mut input {
        if !entry.is_object() {
            return Err(GatewayError::invalid("input items must be objects").with_param("input"));
        }
        if entry.get("id").is_none() {
            entry["id"] = json!(super::render::id("item"));
        }
        if entry.get("type").is_none() {
            entry["type"] = json!("message");
        }
    }
    let mut new_items = Vec::new();
    let mut new_stored = Vec::new();
    let mut additional_tools = Vec::new();
    let mut effort_update = None;
    for entry in &input {
        if entry["type"] == "additional_tools" {
            additional_tools.extend(
                entry["tools"]
                    .as_array()
                    .ok_or_else(|| {
                        GatewayError::invalid("additional_tools.tools must be an array")
                            .with_param("input")
                    })?
                    .clone(),
            );
            continue;
        }
        if entry["type"] == "configuration_update" {
            effort_update = optional_string(&entry["reasoning"], "effort")?;
            continue;
        }
        let id = string(entry, "id")?;
        let parsed = if entry["type"] == "item_reference" {
            let matches: Vec<Item> = history
                .iter()
                .filter(|item| item.id == id)
                .map(|item| item.item.clone())
                .collect();
            if matches.is_empty() {
                return Err(GatewayError::not_found(format!(
                    "referenced item '{id}' not found in previous response"
                ))
                .with_param("input"));
            }
            matches
        } else {
            parse_item(entry)?
        };
        for item in parsed {
            new_stored.push(crate::gateway::session::StoredItem {
                id: id.into(),
                item: item.clone(),
            });
            new_items.push(item);
        }
    }
    let mut turn = TurnRequest {
        requested_model: model,
        system: optional_string(&wire, "instructions")?,
        items: history
            .into_iter()
            .map(|s| s.item)
            .chain(new_items.clone())
            .collect(),
        ..Default::default()
    };
    turn.parallel_tool_calls = wire
        .get("parallel_tool_calls")
        .filter(|v| !v.is_null())
        .map(|_| boolean(&wire, "parallel_tool_calls", true))
        .transpose()?;
    if let Some(v) = wire.get("max_output_tokens").filter(|v| !v.is_null()) {
        turn.max_output_tokens = Some(
            v.as_u64()
                .filter(|n| *n > 0 && *n <= u32::MAX as u64)
                .ok_or_else(|| {
                    GatewayError::invalid("max_output_tokens must be a positive integer")
                        .with_param("max_output_tokens")
                })? as u32,
        );
    }
    for (key, dest, limit) in [
        ("temperature", &mut turn.sampling.temperature, 2.0),
        ("top_p", &mut turn.sampling.top_p, 1.0),
    ] {
        if let Some(v) = wire.get(key).filter(|v| !v.is_null()) {
            let n = v
                .as_f64()
                .filter(|n| n.is_finite() && *n >= 0.0 && *n <= limit)
                .ok_or_else(|| {
                    GatewayError::invalid(format!("{key} must be between 0 and {limit}"))
                        .with_param(key)
                })?;
            *dest = Some(n as f32);
        }
    }
    turn.prompt_cache_key = optional_string(&wire, "prompt_cache_key")?;
    turn.service_tier = optional_string(&wire, "service_tier")?;
    if let Some(tier) = &turn.service_tier {
        if !["auto", "default", "flex", "priority", "scale"].contains(&tier.as_str()) {
            return Err(GatewayError::invalid("invalid service_tier").with_param("service_tier"));
        }
    }
    if let Some(t) = optional_string(&wire, "truncation")? {
        if t != "disabled" && t != "auto" {
            return Err(GatewayError::invalid("truncation must be auto or disabled")
                .with_param("truncation"));
        }
        if t == "auto" {
            return Err(GatewayError::unsupported(
                "automatic context truncation requires backend context admission",
            )
            .with_param("truncation"));
        }
    }
    if let Some(metadata) = wire.get("metadata").filter(|v| !v.is_null()) {
        let obj = metadata.as_object().ok_or_else(|| {
            GatewayError::invalid("metadata must be an object").with_param("metadata")
        })?;
        if obj.len() > 16
            || obj.iter().any(|(k, v)| {
                k.chars().count() > 64 || v.as_str().is_none_or(|s| s.chars().count() > 512)
            })
        {
            return Err(GatewayError::invalid("metadata accepts at most 16 string entries (64-character keys, 512-character values)").with_param("metadata"));
        }
    }
    let mut summary = false;
    if let Some(reasoning) = wire.get("reasoning").filter(|v| !v.is_null()) {
        if !reasoning.is_object() {
            return Err(
                GatewayError::invalid("reasoning must be an object").with_param("reasoning")
            );
        }
        turn.reasoning.effort = optional_string(reasoning, "effort")?;
        if let Some(s) = optional_string(reasoning, "summary")? {
            if !["auto", "concise", "detailed"].contains(&s.as_str()) {
                return Err(GatewayError::invalid("invalid reasoning summary")
                    .with_param("reasoning.summary"));
            }
            summary = true;
        }
    }
    turn.reasoning.effort = effort_update.or(turn.reasoning.effort);
    if let Some(effort) = &turn.reasoning.effort {
        if !["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(&effort.as_str())
        {
            return Err(
                GatewayError::invalid("invalid reasoning effort").with_param("reasoning.effort")
            );
        }
        turn.reasoning.enabled = Some(effort != "none");
    }
    if let Some(text) = wire.get("text").filter(|v| !v.is_null()) {
        if !text.is_object() {
            return Err(GatewayError::invalid("text must be an object").with_param("text"));
        }
        turn.text_verbosity = optional_string(text, "verbosity")?;
        if turn
            .text_verbosity
            .as_ref()
            .is_some_and(|v| !["low", "medium", "high"].contains(&v.as_str()))
        {
            return Err(
                GatewayError::invalid("invalid text verbosity").with_param("text.verbosity")
            );
        }
        if let Some(format) = text.get("format") {
            match string(format, "type")? {
                "text" => {}
                "json_object" => turn.response_format = Some(format.clone()),
                "json_schema" => {
                    string(format, "name")?;
                    if !format["schema"].is_object() {
                        return Err(
                            GatewayError::invalid("text.format.schema must be an object")
                                .with_param("text.format.schema"),
                        );
                    }
                    boolean(format, "strict", false)?;
                    turn.response_format = Some(format.clone());
                }
                kind => {
                    return Err(GatewayError::unsupported(format!(
                        "text format '{kind}' is not supported"
                    ))
                    .with_param("text.format"))
                }
            }
        }
    }
    let include = match wire.get("include") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => a.clone(),
        _ => {
            return Err(
                GatewayError::invalid("include must be a string array").with_param("include")
            )
        }
    };
    let encrypted = include.iter().any(|v| v == "reasoning.encrypted_content");
    // Keep reasoning available for opaque round trips even when summaries are hidden.
    turn.reasoning.return_text = summary || encrypted;
    let mut kinds = HashMap::new();
    let mut names = HashMap::new();
    let mut tools = match wire.get("tools") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.clone(),
        _ => return Err(GatewayError::invalid("tools must be an array").with_param("tools")),
    };
    tools.extend(additional_tools);
    let mut flattened = Vec::new();
    for tool in tools {
        if tool["type"] == "namespace" {
            let namespace = string(&tool, "name")?;
            let children = tool["tools"].as_array().ok_or_else(|| {
                GatewayError::invalid("namespace tools must be an array").with_param("tools")
            })?;
            for child in children {
                if !["function", "custom"].contains(&string(child, "type")?) {
                    return Err(GatewayError::invalid(
                        "namespaces accept function/custom tools only",
                    )
                    .with_param("tools"));
                }
                let name = string(child, "name")?;
                let mapped = format!("{namespace}.{name}");
                names.insert(mapped.clone(), (namespace.to_owned(), name.to_owned()));
                let mut child = child.clone();
                child["name"] = json!(mapped);
                flattened.push(child);
            }
        } else {
            flattened.push(tool);
        }
    }
    for tool in flattened {
        let kind = string(&tool, "type")?;
        if kind == "web_search" || kind == "web_search_preview" {
            if turn.hosted.web_search.is_some() {
                return Err(
                    GatewayError::invalid("only one hosted web search tool is allowed")
                        .with_param("tools"),
                );
            }
            let domains = |key| -> Result<Vec<String>, GatewayError> {
                match tool["filters"].get(key) {
                    None => Ok(Vec::new()),
                    Some(Value::Array(a)) if a.iter().all(Value::is_string) => {
                        Ok(a.iter().map(|v| v.as_str().unwrap().to_owned()).collect())
                    }
                    _ => Err(GatewayError::invalid(
                        "web search domain filters must be string arrays",
                    )
                    .with_param("tools")),
                }
            };
            turn.hosted.web_search = Some(WebSearchSpec {
                name: "web_search".into(),
                allowed_domains: domains("allowed_domains")?,
                blocked_domains: domains("blocked_domains")?,
                user_location: tool.get("user_location").cloned(),
                ..Default::default()
            });
            continue;
        }
        let (name, parameters, description, mapped) = match kind {
            "function" => (
                string(&tool, "name")?.to_owned(),
                tool.get("parameters")
                    .cloned()
                    .unwrap_or(json!({"type":"object","properties":{}})),
                optional_string(&tool, "description")?,
                ToolKind::Function,
            ),
            "custom" => {
                let mut description = optional_string(&tool, "description")?.unwrap_or_default();
                if let Some(format) = tool.get("format") {
                    match string(format, "type")? {
                        "text" => {}
                        "grammar" => {
                            let syntax = string(format, "syntax")?;
                            if !["lark", "regex"].contains(&syntax) {
                                return Err(GatewayError::invalid(
                                    "custom grammar syntax must be lark or regex",
                                )
                                .with_param("tools"));
                            }
                            description.push_str(&format!(
                                "\nReturn input matching this {syntax} grammar:\n{}",
                                string(format, "definition")?
                            ));
                        }
                        _ => {
                            return Err(GatewayError::invalid(
                                "custom format must be text or grammar",
                            )
                            .with_param("tools"))
                        }
                    }
                }
                (
                    string(&tool, "name")?.to_owned(),
                    json!({"type":"object","properties":{"input":{"type":"string"}},"required":["input"],"additionalProperties":false}),
                    Some(description),
                    ToolKind::Custom,
                )
            }
            "local_shell" => (
                "local_shell".into(),
                json!({"type":"object","properties":{"type":{"const":"exec"},"command":{"type":"array","items":{"type":"string"}},"timeout_ms":{"type":"integer"},"working_directory":{"type":"string"},"env":{"type":"object","additionalProperties":{"type":"string"}}},"required":["type","command"]}),
                Some("Execute a command on the client's local shell.".into()),
                ToolKind::LocalShell,
            ),
            _ => {
                return Err(GatewayError::unsupported(format!(
                    "hosted tool '{kind}' is not supported; use client function/custom tools"
                ))
                .with_param("tools"))
            }
        };
        if name.is_empty() || kinds.insert(name.clone(), mapped).is_some() {
            return Err(
                GatewayError::invalid("tool names must be nonempty and unique").with_param("tools"),
            );
        }
        if !parameters.is_object() {
            return Err(
                GatewayError::invalid("function parameters must be a JSON Schema object")
                    .with_param("tools"),
            );
        }
        turn.tools.push(ToolSpec {
            name,
            parameters,
            description,
            strict: boolean(&tool, "strict", false)?,
        });
    }
    turn.tool_choice = match wire.get("tool_choice").filter(|v| !v.is_null()) {
        None => ToolChoice::Auto,
        Some(Value::String(s)) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => return Err(GatewayError::invalid("invalid tool_choice").with_param("tool_choice")),
        },
        Some(v) => match string(v, "type")? {
            "function" | "custom" => ToolChoice::Named {
                name: qualified_name(v)?,
            },
            "local_shell" => ToolChoice::Named {
                name: "local_shell".into(),
            },
            "web_search" | "web_search_preview" => ToolChoice::Named {
                name: "web_search".into(),
            },
            _ => {
                return Err(
                    GatewayError::unsupported("tool_choice type is not supported")
                        .with_param("tool_choice"),
                )
            }
        },
    };
    if let ToolChoice::Named { name } = &turn.tool_choice {
        if !kinds.contains_key(name) && !(name == "web_search" && turn.hosted.web_search.is_some())
        {
            return Err(GatewayError::invalid(format!(
                "tool_choice names undefined tool '{name}'"
            ))
            .with_param("tool_choice"));
        }
    }
    Ok(Parsed {
        response_id: super::render::id("resp"),
        wire,
        turn,
        kinds,
        names,
        parent,
        #[cfg(test)]
        new_items,
        new_stored,
        input,
        store,
        encrypted,
        summary,
    })
}

fn qualified_name(v: &Value) -> Result<String, GatewayError> {
    let name = string(v, "name")?;
    Ok(match optional_string(v, "namespace")? {
        Some(ns) => format!("{ns}.{name}"),
        None => name.to_owned(),
    })
}

pub(super) fn parse_item(v: &Value) -> Result<Vec<Item>, GatewayError> {
    let kind = v["type"].as_str().unwrap_or("message");
    let call_id = || string(v, "call_id").map(str::to_owned);
    Ok(match kind {
        "message" => {
            let role = match string(v, "role")? {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "system" | "developer" => Role::System,
                role => {
                    return Err(
                        GatewayError::invalid(format!("unsupported message role '{role}'"))
                            .with_param("input"),
                    )
                }
            };
            vec![Item::Message {
                role,
                content: parts(
                    v.get("content")
                        .ok_or_else(|| GatewayError::invalid("message content is required"))?,
                )?,
            }]
        }
        "function_call" => vec![Item::ToolCall {
            id: call_id()?,
            name: qualified_name(v)?,
            arguments: string(v, "arguments")?.into(),
        }],
        "custom_tool_call" => vec![Item::ToolCall {
            id: call_id()?,
            name: qualified_name(v)?,
            arguments: json!({"input":string(v, "input")?}).to_string(),
        }],
        "local_shell_call" => vec![Item::ToolCall {
            id: call_id()?,
            name: "local_shell".into(),
            arguments: v
                .get("action")
                .filter(|v| v.is_object())
                .ok_or_else(|| GatewayError::invalid("local_shell_call action is required"))?
                .to_string(),
        }],
        "function_call_output" | "custom_tool_call_output" | "local_shell_call_output" => {
            vec![Item::ToolResult {
                call_id: call_id()?,
                content: parts(
                    v.get("output")
                        .ok_or_else(|| GatewayError::invalid("tool output is required"))?,
                )?,
                is_error: false,
            }]
        }
        "reasoning" => {
            let visible = reasoning_text(v)?;
            let text = match optional_string(v, "encrypted_content")? {
                Some(token) => {
                    let token = decode(&token, "reasoning")?;
                    // The token carries the full trace plus a digest of what was
                    // shown beside it. Visible text that no longer matches that
                    // digest was edited by the client, and the edit wins. Tokens
                    // without the digest predate it and keep their own text.
                    let edited = !visible.is_empty()
                        && token["shown_sha256"]
                            .as_str()
                            .is_some_and(|shown| shown != sha256_hex(&visible));
                    if edited {
                        visible
                    } else {
                        token["text"].as_str().unwrap().to_owned()
                    }
                }
                None => visible,
            };
            vec![Item::Reasoning {
                text,
                signature: None,
            }]
        }
        "compaction" => {
            let token = decode(string(v, "encrypted_content")?, "compaction")?;
            vec![Item::Message {
                role: Role::System,
                content: vec![Part::text(string(&token, "text")?)],
            }]
        }
        "web_search_call" => {
            let id = string(v, "id")?.to_owned();
            let action = v.get("action").cloned().unwrap_or(json!({}));
            vec![Item::ServerToolCall {
                id,
                name: "web_search".into(),
                input: action,
            }]
        }
        "item_reference" => {
            return Err(GatewayError::not_found(
                "item_reference needs a stored previous_response_id",
            )
            .with_param("input"))
        }
        _ => {
            return Err(GatewayError::unsupported(format!(
                "input item type '{kind}' is not supported"
            ))
            .with_param("input"))
        }
    })
}

fn reasoning_text(v: &Value) -> Result<String, GatewayError> {
    let mut texts = Vec::new();
    // Content is the complete trace; summary is a fallback, not a duplicate.
    let a = v
        .get("content")
        .filter(|v| v.as_array().is_some_and(|a| !a.is_empty()))
        .or_else(|| v.get("summary"));
    if let Some(a) = a {
        for part in a
            .as_array()
            .ok_or_else(|| GatewayError::invalid("reasoning content/summary must be arrays"))?
        {
            texts.push(string(part, "text")?);
        }
    }
    Ok(texts.join("\n"))
}

fn parts(v: &Value) -> Result<Vec<Part>, GatewayError> {
    if let Some(s) = v.as_str() {
        return Ok(vec![Part::text(s)]);
    }
    let mut result = Vec::new();
    for part in v.as_array().ok_or_else(|| {
        GatewayError::invalid("content must be a string or array").with_param("input")
    })? {
        result.push(match string(part, "type")? {
            "input_text" | "output_text" | "summary_text" => Part::text(string(part, "text")?),
            "refusal" => Part::text(string(part, "refusal")?),
            "input_image" => {
                if part.get("file_id").is_some_and(|v| !v.is_null()) {
                    return Err(GatewayError::unsupported(
                        "input_image file_id is not supported; supply image_url",
                    )
                    .with_param("input"));
                }
                let url = string(part, "image_url")?;
                let source = if let Some(data) = url.strip_prefix("data:") {
                    let (mime, data) = data
                        .split_once(";base64,")
                        .ok_or_else(|| GatewayError::invalid("image data URL must be base64"))?;
                    base64::engine::general_purpose::STANDARD
                        .decode(data)
                        .map_err(|_| GatewayError::invalid("invalid image base64"))?;
                    ImageSource::Base64 {
                        media_type: mime.into(),
                        data: data.into(),
                    }
                } else {
                    ImageSource::Url { url: url.into() }
                };
                let detail = optional_string(part, "detail")?;
                if detail
                    .as_ref()
                    .is_some_and(|s| !["auto", "low", "high", "original"].contains(&s.as_str()))
                {
                    return Err(GatewayError::invalid("invalid image detail"));
                }
                Part::Image { source, detail }
            }
            "input_file" => {
                if part.get("file_id").is_some_and(|v| !v.is_null())
                    || part.get("file_url").is_some_and(|v| !v.is_null())
                {
                    return Err(GatewayError::unsupported(
                        "input_file file_id/file_url is not supported; supply file_data",
                    )
                    .with_param("input"));
                }
                Part::File {
                    name: optional_string(part, "filename")?,
                    media_type: None,
                    data: string(part, "file_data")?.into(),
                }
            }
            kind => {
                return Err(GatewayError::unsupported(format!(
                    "content part '{kind}' is not supported"
                ))
                .with_param("input"))
            }
        });
    }
    Ok(result)
}

pub(super) fn encode(kind: &str, text: &str) -> String {
    token(json!({"kind":kind,"text":text}))
}

/// A reasoning token: the full trace, and a digest of the visible text the
/// client received beside it (as [`reasoning_text`] reads it back), so an
/// edited summary can be told apart from an echoed one.
pub(super) fn encode_reasoning(text: &str, shown: &str) -> String {
    token(json!({"kind":"reasoning","text":text,"shown_sha256":sha256_hex(shown)}))
}

fn token(v: Value) -> String {
    format!("cuteafd.v1.{}", URL_SAFE_NO_PAD.encode(v.to_string()))
}

fn sha256_hex(s: &str) -> String {
    format!("{:x}", Sha256::digest(s.as_bytes()))
}

fn decode(token: &str, kind: &str) -> Result<Value, GatewayError> {
    let invalid = || {
        GatewayError::invalid(format!(
            "invalid or foreign {kind} encrypted_content; only cuteafd.v1 tokens are supported"
        ))
        .with_param("input")
    };
    let bytes = URL_SAFE_NO_PAD
        .decode(token.strip_prefix("cuteafd.v1.").ok_or_else(invalid)?)
        .map_err(|_| invalid())?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
    if v["kind"] != kind || !v["text"].is_string() {
        return Err(invalid());
    }
    Ok(v)
}
