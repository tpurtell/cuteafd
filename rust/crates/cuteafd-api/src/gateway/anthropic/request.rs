use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::{json, Value};

use crate::gateway::{
    turn::{ImageSource, Item, Part, Role, ToolChoice, ToolSpec, TurnRequest, WebSearchSpec},
    GatewayError,
};

pub(super) fn parse(value: &Value, require_max: bool) -> Result<(TurnRequest, bool), GatewayError> {
    if !value.is_object() {
        return Err(GatewayError::invalid("request must be a JSON object"));
    }
    let mut turn = TurnRequest {
        requested_model: string(value, "model")?,
        ..Default::default()
    };
    turn.max_output_tokens = number(value, "max_tokens")?;
    if require_max && turn.max_output_tokens.is_none() {
        return Err(GatewayError::invalid("max_tokens is required"));
    }
    if let Some(system) = value.get("system") {
        turn.system = Some(
            parts(system)?
                .into_iter()
                .filter_map(|p| match p {
                    Part::Text { text } => Some(text),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    let messages = value
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| GatewayError::invalid("messages must be an array"))?;
    for message in messages {
        let role = match string(message, "role")?.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => Role::System,
            _ => {
                return Err(GatewayError::invalid(
                    "message role must be user, assistant or system",
                ))
            }
        };
        let content = message
            .get("content")
            .ok_or_else(|| GatewayError::invalid("message content is required"))?;
        if let Some(text) = content.as_str() {
            turn.items.push(Item::Message {
                role,
                content: vec![Part::text(text)],
            });
            continue;
        }
        let blocks = content.as_array().ok_or_else(|| {
            GatewayError::invalid("message content must be a string or block array")
        })?;
        let mut pending = Vec::new();
        for block in blocks {
            let item = match string(block, "type")?.as_str() {
                "tool_use" => Some(Item::ToolCall {
                    id: string(block, "id")?,
                    name: string(block, "name")?,
                    arguments: object(block, "input")?.to_string(),
                }),
                "tool_result" => Some(Item::ToolResult {
                    call_id: string(block, "tool_use_id")?,
                    content: block
                        .get("content")
                        .map(parts)
                        .transpose()?
                        .unwrap_or_default(),
                    is_error: boolean(block, "is_error")?.unwrap_or(false),
                }),
                "thinking" => Some(Item::Reasoning {
                    text: string(block, "thinking")?,
                    signature: optional_string(block, "signature")?,
                }),
                // Preserve opaque replay material without treating it as readable reasoning.
                "redacted_thinking" => Some(Item::Reasoning {
                    text: String::new(),
                    signature: Some(format!("redacted:{}", string(block, "data")?)),
                }),
                "server_tool_use" => Some(Item::ServerToolCall {
                    id: string(block, "id")?,
                    name: string(block, "name")?,
                    input: object(block, "input")?,
                }),
                "web_search_tool_result" => Some(Item::ServerToolResult {
                    call_id: string(block, "tool_use_id")?,
                    name: "web_search".into(),
                    output: search_output(block.get("content").unwrap_or(&Value::Null)),
                }),
                _ => {
                    pending.extend(part(block)?);
                    None
                }
            };
            if let Some(item) = item {
                if !pending.is_empty() {
                    turn.items.push(Item::Message {
                        role,
                        content: std::mem::take(&mut pending),
                    });
                }
                turn.items.push(item);
            }
        }
        if !pending.is_empty() {
            turn.items.push(Item::Message {
                role,
                content: pending,
            });
        }
    }
    if let Some(tools) = value.get("tools") {
        for tool in tools
            .as_array()
            .ok_or_else(|| GatewayError::invalid("tools must be an array"))?
        {
            let kind = tool.get("type").and_then(Value::as_str).unwrap_or("custom");
            if kind.starts_with("web_search_") {
                if turn.hosted.web_search.is_some() {
                    return Err(GatewayError::invalid(
                        "only one hosted web search tool is supported",
                    ));
                }
                let allowed_domains = strings(tool, "allowed_domains")?;
                let blocked_domains = strings(tool, "blocked_domains")?;
                if !allowed_domains.is_empty() && !blocked_domains.is_empty() {
                    return Err(GatewayError::invalid(
                        "web search accepts allowed_domains or blocked_domains, not both",
                    ));
                }
                turn.hosted.web_search = Some(WebSearchSpec {
                    name: string(tool, "name")?,
                    max_uses: number(tool, "max_uses")?,
                    allowed_domains,
                    blocked_domains,
                    user_location: tool.get("user_location").cloned(),
                });
                continue;
            }
            let parameters = if kind == "custom" {
                object(tool, "input_schema")?
            } else {
                client_schema(kind).ok_or_else(|| GatewayError::unsupported(format!("Input tag '{kind}': unsupported Anthropic tool type; supply a custom tool with input_schema")))?
            };
            turn.tools.push(ToolSpec { name: string(tool, "name")?, description: optional_string(tool, "description")?.or_else(|| (kind != "custom").then(|| format!("Anthropic client tool {kind}; execute the requested command on the client."))), parameters, strict: boolean(tool, "strict")?.unwrap_or(false) });
        }
    }
    if let Some(choice) = value.get("tool_choice") {
        turn.tool_choice = match string(choice, "type")?.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "any" => ToolChoice::Required,
            "tool" => ToolChoice::Named {
                name: string(choice, "name")?,
            },
            _ => return Err(GatewayError::invalid("unknown tool_choice.type")),
        };
        turn.parallel_tool_calls = boolean(choice, "disable_parallel_tool_use")?.map(|v| !v);
    }
    turn.sampling.stop = strings(value, "stop_sequences")?;
    turn.sampling.temperature = float(value, "temperature")?;
    turn.sampling.top_p = float(value, "top_p")?;
    turn.sampling.top_k = number(value, "top_k")?;
    if let Some(thinking) = value.get("thinking") {
        turn.reasoning.enabled = Some(match string(thinking, "type")?.as_str() {
            "enabled" | "adaptive" => true,
            "disabled" | "between_tools" => false,
            _ => return Err(GatewayError::invalid("unknown thinking.type")),
        });
        turn.reasoning.budget_tokens = number(thinking, "budget_tokens")?;
        turn.reasoning.return_text =
            thinking.get("display").and_then(Value::as_str) != Some("omitted");
    }
    let output = value.get("output_config");
    turn.reasoning.effort = output
        .map(|v| optional_string(v, "effort"))
        .transpose()?
        .flatten()
        .or(optional_string(value, "effort")?);
    turn.response_format = output
        .and_then(|v| v.get("format"))
        .cloned()
        .or_else(|| value.get("output_format").cloned());
    Ok((turn, boolean(value, "stream")?.unwrap_or(false)))
}

fn string(v: &Value, key: &str) -> Result<String, GatewayError> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| GatewayError::invalid(format!("{key} must be a string")))
}
fn optional_string(v: &Value, key: &str) -> Result<Option<String>, GatewayError> {
    v.get(key).map(|_| string(v, key)).transpose()
}
fn object(v: &Value, key: &str) -> Result<Value, GatewayError> {
    v.get(key)
        .filter(|v| v.is_object())
        .cloned()
        .ok_or_else(|| GatewayError::invalid(format!("{key} must be an object")))
}
fn number(v: &Value, key: &str) -> Result<Option<u32>, GatewayError> {
    v.get(key)
        .map(|n| {
            n.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    GatewayError::invalid(format!("{key} must be an unsigned 32-bit integer"))
                })
        })
        .transpose()
}
fn boolean(v: &Value, key: &str) -> Result<Option<bool>, GatewayError> {
    v.get(key)
        .map(|n| {
            n.as_bool()
                .ok_or_else(|| GatewayError::invalid(format!("{key} must be boolean")))
        })
        .transpose()
}
fn float(v: &Value, key: &str) -> Result<Option<f32>, GatewayError> {
    v.get(key)
        .map(|n| {
            n.as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| GatewayError::invalid(format!("{key} must be a number")))
        })
        .transpose()
}
fn strings(v: &Value, key: &str) -> Result<Vec<String>, GatewayError> {
    match v.get(key) {
        None => Ok(Vec::new()),
        Some(v) => v
            .as_array()
            .ok_or_else(|| GatewayError::invalid(format!("{key} must be a string array")))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| GatewayError::invalid(format!("{key} must contain strings")))
            })
            .collect(),
    }
}
fn parts(v: &Value) -> Result<Vec<Part>, GatewayError> {
    if let Some(text) = v.as_str() {
        return Ok(vec![Part::text(text)]);
    }
    v.as_array()
        .ok_or_else(|| GatewayError::invalid("content must be a string or block array"))?
        .iter()
        .try_fold(Vec::new(), |mut out, b| {
            out.extend(part(b)?);
            Ok(out)
        })
}
fn part(block: &Value) -> Result<Vec<Part>, GatewayError> {
    Ok(match string(block, "type")?.as_str() {
        "text" => vec![Part::text(string(block, "text")?)],
        "image" => {
            let source = block
                .get("source")
                .ok_or_else(|| GatewayError::invalid("image source is required"))?;
            let source = match string(source, "type")?.as_str() {
                "base64" => ImageSource::Base64 {
                    media_type: string(source, "media_type")?,
                    data: string(source, "data")?,
                },
                "url" => ImageSource::Url {
                    url: string(source, "url")?,
                },
                kind => {
                    return Err(GatewayError::unsupported(format!(
                        "image source {kind} is not supported"
                    )))
                }
            };
            vec![Part::Image {
                source,
                detail: None,
            }]
        }
        "document" => {
            let source = block
                .get("source")
                .ok_or_else(|| GatewayError::invalid("document source is required"))?;
            let mut out = Vec::new();
            if let Some(context) = optional_string(block, "context")? {
                out.push(Part::text(context));
            }
            match string(source, "type")?.as_str() {
                "text" => out.push(Part::text(string(source, "data")?)),
                "content" => out.extend(parts(source.get("content").ok_or_else(|| {
                    GatewayError::invalid("document source.content is required")
                })?)?),
                "base64" if source["media_type"] == "text/plain" => {
                    let bytes = STANDARD.decode(string(source, "data")?).map_err(|_| {
                        GatewayError::invalid("text document data is invalid base64")
                    })?;
                    out.push(Part::text(String::from_utf8(bytes).map_err(|_| {
                        GatewayError::invalid("text document must be UTF-8")
                    })?));
                }
                "base64" => out.push(Part::File {
                    name: optional_string(block, "title")?,
                    media_type: optional_string(source, "media_type")?,
                    data: string(source, "data")?,
                }),
                "url" => out.push(Part::File {
                    name: optional_string(block, "title")?,
                    media_type: None,
                    data: string(source, "url")?,
                }),
                "file" => {
                    return Err(GatewayError::unsupported(
                        "document file_id requires a Files API store; use text or base64",
                    ))
                }
                kind => {
                    return Err(GatewayError::unsupported(format!(
                        "document source {kind} is not supported"
                    )))
                }
            }
            out
        }
        "search_result" => {
            let mut out = Vec::new();
            if let Some(title) = block.get("title").and_then(Value::as_str) {
                out.push(Part::text(title));
            }
            if let Some(source) = block.get("source").and_then(Value::as_str) {
                out.push(Part::text(source));
            }
            if let Some(content) = block.get("content") {
                out.extend(parts(content)?);
            }
            out
        }
        // Forward-compatible content such as container_upload/tool_reference is
        // not executable input. Preserve readable text if provided, otherwise skip.
        _ => block
            .get("text")
            .and_then(Value::as_str)
            .map(|s| vec![Part::text(s)])
            .unwrap_or_default(),
    })
}

fn search_output(content: &Value) -> Value {
    if let Some(hits) = content.as_array() {
        json!({"results": hits.iter().map(|hit| {
            let text = hit.get("encrypted_content").and_then(Value::as_str).and_then(|v| STANDARD.decode(v).ok()).and_then(|v| String::from_utf8(v).ok()).unwrap_or_default();
            json!({"url": hit["url"], "title": hit["title"], "content": text, "published": hit.get("page_age")})
        }).collect::<Vec<_>>()})
    } else {
        json!({"error": content.get("error_code").and_then(Value::as_str).unwrap_or("unavailable")})
    }
}

// Schemas derive from the documented commands for these schema-less client
// tools. Unknown versions are rejected rather than silently inventing a schema.
fn client_schema(kind: &str) -> Option<Value> {
    if matches!(kind, "bash_20241022" | "bash_20250124") {
        return Some(
            json!({"type":"object","properties":{"command":{"type":"string"},"restart":{"type":"boolean"}},"additionalProperties":false}),
        );
    }
    if matches!(
        kind,
        "text_editor_20241022"
            | "text_editor_20250124"
            | "text_editor_20250429"
            | "text_editor_20250728"
            | "memory_20250818"
    ) {
        let mut commands = vec!["view", "create", "str_replace", "insert"];
        if kind == "memory_20250818" {
            commands.extend(["delete", "rename"]);
        } else if matches!(kind, "text_editor_20241022" | "text_editor_20250124") {
            commands.push("undo_edit");
        }
        return Some(json!({"type":"object","properties":{
            "command":{"type":"string","enum":commands},"path":{"type":"string"},
            "file_text":{"type":"string"},"old_str":{"type":"string"},"new_str":{"type":"string"},
            "insert_line":{"type":"integer"},"insert_text":{"type":"string"},
            "view_range":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2},
            "old_path":{"type":"string"},"new_path":{"type":"string"}},"required":["command"],"additionalProperties":false}));
    }
    if matches!(
        kind,
        "computer_20241022" | "computer_20250124" | "computer_20251124"
    ) {
        // Official anthropic-quickstarts computer.py defines these versioned
        // action vocabularies and handler input fields.
        let mut actions = vec![
            "key",
            "type",
            "mouse_move",
            "left_click",
            "left_click_drag",
            "right_click",
            "middle_click",
            "double_click",
            "screenshot",
            "cursor_position",
        ];
        if kind != "computer_20241022" {
            actions.extend([
                "left_mouse_down",
                "left_mouse_up",
                "scroll",
                "hold_key",
                "wait",
                "triple_click",
            ]);
        }
        if kind == "computer_20251124" {
            actions.push("zoom");
        }
        let coordinate = json!({"type":"array","items":{"type":"integer","minimum":0},"minItems":2,"maxItems":2});
        return Some(json!({"type":"object","properties":{
            "action":{"type":"string","enum":actions},"text":{"type":"string"},
            "coordinate":coordinate,"start_coordinate":coordinate,
            "scroll_direction":{"type":"string","enum":["up","down","left","right"]},
            "scroll_amount":{"type":"integer","minimum":0},"duration":{"type":"number","minimum":0},
            "region":{"type":"array","items":{"type":"integer","minimum":0},"minItems":4,"maxItems":4}
        },"required":["action"],"additionalProperties":false}));
    }
    None
}
