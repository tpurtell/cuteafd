use serde_json::{json, Value};
use super::{Flavor, UpstreamConfig};
use crate::gateway::{turn::*, GatewayError};

pub(super) fn request(turn: &TurnRequest, config: &UpstreamConfig) -> Result<Value, GatewayError> {
    if turn.modalities.audio_out { return Err(GatewayError::unsupported("upstream audio output is not implemented")); }
    match config.flavor { Flavor::OpenaiChat => chat(turn, config), Flavor::Anthropic => anthropic(turn, config) }
}
/// Preserve standard tool names; encode namespace/custom names deterministically without collisions with plain names.
pub(super) fn wire_name(name: &str) -> String {
    if !name.is_empty() && name.len() <= 64 && !name.starts_with("cf_")
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') { return name.to_string(); }
    use sha2::{Digest, Sha256};
    let hash = format!("{:x}",Sha256::digest(name.as_bytes()));
    format!("cf_{}",&hash[..60])
}
fn role(role: Role) -> &'static str { match role { Role::User => "user", Role::Assistant => "assistant", Role::System => "system" } }
fn parts(content: &[Part], flavor: Flavor) -> Result<Value, GatewayError> {
    let mut out = Vec::new();
    for part in content {
        out.push(match part {
            Part::Text { text } => json!({"type":"text", "text":text}),
            Part::Image { source, detail } => match flavor {
                Flavor::OpenaiChat => {
                    let url = match source { ImageSource::Url { url } => url.clone(), ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}") };
                    let mut image = json!({"url":url});
                    if let Some(detail) = detail { image["detail"] = json!(detail); }
                    json!({"type":"image_url", "image_url":image})
                }
                Flavor::Anthropic => json!({"type":"image", "source":match source {
                    ImageSource::Url { url } => json!({"type":"url", "url":url}),
                    ImageSource::Base64 { media_type, data } => json!({"type":"base64", "media_type":media_type,"data":data}) }}),
            },
            Part::Audio { format, data } if flavor == Flavor::OpenaiChat => json!({"type":"input_audio", "input_audio":{"format":format,"data":data}}),
            Part::Audio { .. } => return Err(GatewayError::unsupported("Anthropic upstream audio input is not supported")),
            Part::File { .. } => return Err(GatewayError::unsupported("upstream file input requires a frontend text conversion")),
        });
    }
    Ok(Value::Array(out))
}
fn chat_content(content: &[Part]) -> Result<Value, GatewayError> {
    if content.iter().all(|p| matches!(p, Part::Text { .. })) {
        return Ok(json!(content.iter().filter_map(|p| if let Part::Text { text } = p { Some(text.as_str()) } else { None }).collect::<Vec<_>>().join("")));
    }
    parts(content, Flavor::OpenaiChat)
}
fn assistant(messages: &mut Vec<Value>) -> &mut Value {
    if messages.last().is_none_or(|m| m["role"] != "assistant") {
        messages.push(json!({"role":"assistant","content":null}));
    }
    messages.last_mut().expect("assistant exists")
}
fn add_chat_call(messages: &mut Vec<Value>, id: &str, name: &str, arguments: &str) {
    let message = assistant(messages);
    if !message["tool_calls"].is_array() { message["tool_calls"] = json!([]); }
    message["tool_calls"].as_array_mut().unwrap().push(json!({"id":id,"type":"function","function":{"name":wire_name(name),"arguments":arguments}}));
}
fn chat(turn: &TurnRequest, config: &UpstreamConfig) -> Result<Value, GatewayError> {
    let mut messages = Vec::new();
    if let Some(system) = &turn.system { messages.push(json!({"role":"system","content":system})); }
    for item in &turn.items {
        match item {
            Item::Message { role: r, content } => {
                let content = chat_content(content)?;
                if *r == Role::Assistant && messages.last().is_some_and(|m| m["role"] == "assistant" && m["content"].is_null()) {
                    assistant(&mut messages)["content"] = content;
                } else { messages.push(json!({"role":role(*r),"content":content})); }
            }
            Item::Reasoning { text, .. } => { assistant(&mut messages)["reasoning_content"] = json!(text); }
            Item::ToolCall { id, name, arguments } => add_chat_call(&mut messages, id, name, arguments),
            Item::ToolResult { call_id, content, .. } => messages.push(json!({"role":"tool","tool_call_id":call_id,"content":chat_content(content)?})),
            Item::ServerToolCall { id, name, input } => add_chat_call(&mut messages, id, name, &input.to_string()),
            Item::ServerToolResult { call_id, output, .. } => messages.push(json!({"role":"tool","tool_call_id":call_id,"content":output.to_string()})),
        }
    }
    let mut request = json!({"model":turn.model,"messages":messages,"stream":true,"stream_options":{"include_usage":true}});
    if !turn.tools.is_empty() {
        request["tools"] = json!(turn.tools.iter().map(|tool| {
            let mut function = json!({"name":wire_name(&tool.name),"parameters":tool.parameters});
            if let Some(description) = &tool.description { function["description"] = json!(description); }
            if tool.strict { function["strict"] = json!(true); }
            json!({"type":"function","function":function})
        }).collect::<Vec<_>>());
        request["tool_choice"] = match &turn.tool_choice { ToolChoice::Auto => json!("auto"), ToolChoice::None => json!("none"),
            ToolChoice::Required => json!("required"), ToolChoice::Named { name } => json!({"type":"function","function":{"name":wire_name(name)}}) };
        if let Some(parallel) = turn.parallel_tool_calls { request["parallel_tool_calls"] = json!(parallel); }
    }
    let thinking = turn.reasoning.enabled.unwrap_or(config.deepseek_thinking);
    if config.deepseek_thinking {
        request["thinking"] = json!({"type":if thinking { "enabled" } else { "disabled" }});
        if thinking && matches!(turn.tool_choice, ToolChoice::Required | ToolChoice::Named { .. }) {
            return Err(GatewayError::unsupported("DeepSeek thinking mode does not support required or named tool choice; disable thinking"));
        }
    }
    if let Some(effort) = &turn.reasoning.effort { request["reasoning_effort"] = json!(effort); }
    else if !config.deepseek_thinking && turn.reasoning.enabled == Some(false) { request["reasoning_effort"] = json!("none"); }
    if let Some(max) = turn.max_output_tokens { request["max_tokens"] = json!(max); }
    if !(config.deepseek_thinking && thinking) { if let Some(t) = turn.sampling.temperature { request["temperature"] = json!(t); } }
    if let Some(p) = turn.sampling.top_p { request["top_p"] = json!(p); }
    if let Some(seed) = turn.sampling.seed { request["seed"] = json!(seed); }
    if !turn.sampling.stop.is_empty() { request["stop"] = json!(turn.sampling.stop); }
    if let Some(format) = &turn.response_format {
        request["response_format"] = if format.get("schema").is_some() && format.get("type").is_none() { json!({"type":"json_schema","json_schema":format}) } else { format.clone() };
    }
    Ok(request)
}
fn append_block(messages: &mut Vec<Value>, role: &str, block: Value) {
    if messages.last().is_none_or(|m| m["role"] != role) { messages.push(json!({"role":role,"content":[]})); }
    messages.last_mut().unwrap()["content"].as_array_mut().unwrap().push(block);
}
fn anthropic(turn: &TurnRequest, config: &UpstreamConfig) -> Result<Value, GatewayError> {
    let mut messages = Vec::new();
    let mut system = turn.system.clone().unwrap_or_default();
    for item in &turn.items {
        match item {
            Item::Message { role: Role::System, content } => {
                for part in content { if let Part::Text { text } = part { if !system.is_empty() { system.push('\n'); } system.push_str(text); }
                    else { return Err(GatewayError::unsupported("non-text system content is not supported")); } }
            }
            Item::Message { role, content } => for block in parts(content, Flavor::Anthropic)?.as_array().unwrap() { append_block(&mut messages, self::role(*role), block.clone()); },
            Item::Reasoning { text, signature } => {
                if let Some(data) = signature.as_ref().and_then(|s| s.strip_prefix("redacted:")) {
                    append_block(&mut messages,"assistant",json!({"type":"redacted_thinking","data":data}));
                    continue;
                }
                let mut block = json!({"type":"thinking","thinking":text});
                if let Some(signature) = signature { block["signature"] = json!(signature); }
                append_block(&mut messages, "assistant", block);
            }
            Item::ToolCall { id, name, arguments } => {
                let input: Value = serde_json::from_str(arguments).map_err(|_| GatewayError::invalid("tool call arguments must be JSON"))?;
                append_block(&mut messages, "assistant", json!({"type":"tool_use","id":id,"name":wire_name(name),"input":input}));
            }
            Item::ToolResult { call_id, content, is_error } => append_block(&mut messages, "user", json!({"type":"tool_result","tool_use_id":call_id,"content":parts(content, Flavor::Anthropic)?,"is_error":is_error})),
            Item::ServerToolCall { id, name, input } => append_block(&mut messages, "assistant", json!({"type":"tool_use","id":id,"name":wire_name(name),"input":input})),
            Item::ServerToolResult { call_id, output, .. } => append_block(&mut messages, "user", json!({"type":"tool_result","tool_use_id":call_id,"content":output.to_string()})),
        }
    }
    let mut request = json!({"model":turn.model,"messages":messages,"stream":true,"max_tokens":turn.max_output_tokens.unwrap_or(4096)});
    if !system.is_empty() { request["system"] = json!(system); }
    if !turn.tools.is_empty() {
        request["tools"] = json!(turn.tools.iter().map(|t| json!({"name":wire_name(&t.name),"description":t.description.clone().unwrap_or_default(),"input_schema":t.parameters})).collect::<Vec<_>>());
        request["tool_choice"] = match &turn.tool_choice { ToolChoice::Auto => json!({"type":"auto"}), ToolChoice::None => json!({"type":"none"}),
            ToolChoice::Required => json!({"type":"any"}), ToolChoice::Named { name } => json!({"type":"tool","name":wire_name(name)}) };
        if let Some(parallel) = turn.parallel_tool_calls { request["tool_choice"]["disable_parallel_tool_use"] = json!(!parallel); }
    }
    if let Some(enabled) = turn.reasoning.enabled {
        request["thinking"] = if enabled { json!({"type":"enabled","budget_tokens":turn.reasoning.budget_tokens.unwrap_or(1024)}) } else { json!({"type":"disabled"}) };
    }
    if let Some(effort) = &turn.reasoning.effort { request["output_config"]["effort"] = json!(effort); }
    if !(config.deepseek_thinking && turn.reasoning.enabled != Some(false)) { if let Some(t) = turn.sampling.temperature { request["temperature"] = json!(t); } }
    if let Some(p) = turn.sampling.top_p { request["top_p"] = json!(p); }
    if let Some(k) = turn.sampling.top_k { request["top_k"] = json!(k); }
    if !turn.sampling.stop.is_empty() { request["stop_sequences"] = json!(turn.sampling.stop); }
    if let Some(format) = &turn.response_format {
        if let Some(schema) = format.get("schema") { request["output_config"]["format"] = json!({"type":"json_schema","schema":schema}); }
        else { return Err(GatewayError::unsupported("Anthropic upstream requires a JSON schema for structured output")); }
    }
    Ok(request)
}
