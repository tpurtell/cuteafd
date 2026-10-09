use serde_json::{json, Value};
use super::{Flavor, UpstreamConfig};
use crate::gateway::{turn::*, GatewayError};

pub(super) fn request(turn: &TurnRequest, config: &UpstreamConfig) -> Result<Value, GatewayError> {
    if turn.modalities.audio_out { return Err(GatewayError::unsupported("upstream audio output is not implemented")); }
    if !config.capabilities.json_schema && turn.response_format.as_ref().is_some_and(|format| format["type"] == "json_schema" || format.get("schema").is_some()) {
        return Err(GatewayError::unsupported("response_format: json_schema is not supported by this upstream backend; json_object is supported")
            .with_param("response_format"));
    }
    if !config.capabilities.strict_tools && turn.tools.iter().any(|tool| tool.strict) {
        return Err(GatewayError::unsupported("strict function tools are not supported by this upstream backend").with_param("tools"));
    }
    let names = WireNames::new(turn);
    match config.flavor { Flavor::OpenaiChat => chat(turn, config, &names), Flavor::Anthropic => anthropic(turn, config, &names) }
}
/// Per-turn tool names on the wire. Upstream models see readable names: a
/// namespaced tool (`functions.exec`) goes out under its local name (`exec`)
/// when that is unique and doesn't clash with another tool, because client
/// prompts (Codex) tell the model to call the local name. Otherwise it is
/// sanitized to `functions__exec`, and only names that still aren't legal
/// (or collide) are hashed. `original` maps every name the model might use
/// back: the wire name, the full name, or a unique local name.
#[derive(Debug, Default)]
pub(super) struct WireNames { to_wire: std::collections::HashMap<String, String>, to_original: std::collections::HashMap<String, String> }

fn legal(name: &str) -> bool {
    !name.is_empty() && name.len() <= 64 && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}
fn hashed(name: &str) -> String {
    use sha2::{Digest, Sha256};
    format!("cf_{}", &format!("{:x}", Sha256::digest(name.as_bytes()))[..60])
}

impl WireNames {
    pub(super) fn new(turn: &TurnRequest) -> Self {
        let mut names: Vec<&str> = turn.tools.iter().map(|t| t.name.as_str()).collect();
        for item in &turn.items {
            if let Item::ToolCall { name, .. } | Item::ServerToolCall { name, .. } = item { names.push(name); }
        }
        if let ToolChoice::Named { name } = &turn.tool_choice { names.push(name); }
        names.sort_unstable(); names.dedup();
        let local = |n: &str| n.rsplit('.').next().unwrap_or(n).to_string();
        let mut local_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for n in &names { *local_count.entry(local(n)).or_default() += 1; }
        let mut out = Self::default();
        let mut taken: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Plain legal names keep themselves.
        for n in names.iter().filter(|n| legal(n) && !n.starts_with("cf_")) {
            out.to_wire.insert(n.to_string(), n.to_string());
            taken.insert(n.to_string());
        }
        let pending: Vec<&str> = names.iter().copied().filter(|n| !out.to_wire.contains_key(*n)).collect();
        for n in pending {
            let short = local(n);
            let sanitized: String = n.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect::<String>();
            let dotted = n.replace('.', "__");
            let wire = [ (local_count[&short] == 1).then(|| short.clone()), Some(dotted), Some(sanitized) ]
                .into_iter().flatten().find(|w| legal(w) && !w.starts_with("cf_") && !taken.contains(w))
                .unwrap_or_else(|| hashed(n));
            taken.insert(wire.clone());
            out.to_wire.insert(n.to_string(), wire);
        }
        for (original, wire) in &out.to_wire {
            out.to_original.insert(wire.clone(), original.clone());
            out.to_original.entry(original.clone()).or_insert_with(|| original.clone());
            let short = local(original);
            if local_count[&short] == 1 { out.to_original.entry(short).or_insert_with(|| original.clone()); }
        }
        out
    }
    pub(super) fn wire(&self, name: &str) -> String {
        self.to_wire.get(name).cloned().unwrap_or_else(|| if legal(name) { name.to_string() } else { hashed(name) })
    }
    pub(super) fn original(&self, name: &str) -> String {
        self.to_original.get(name).cloned().unwrap_or_else(|| name.to_string())
    }
}

/// Stateless wire name for a lone tool name (tests, single-name callers).
pub(super) fn wire_name(name: &str) -> String {
    if legal(name) && !name.starts_with("cf_") { name.to_string() } else { hashed(name) }
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
            Part::Audio { format, data } if flavor == Flavor::OpenaiChat => chat_audio(format,data)?,
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
    // `name` is already the wire name.
    let message = assistant(messages);
    if !message["tool_calls"].is_array() { message["tool_calls"] = json!([]); }
    message["tool_calls"].as_array_mut().unwrap().push(json!({"id":id,"type":"function","function":{"name":name,"arguments":arguments}}));
}
fn chat(turn: &TurnRequest, config: &UpstreamConfig, names: &WireNames) -> Result<Value, GatewayError> {
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
            Item::ToolCall { id, name, arguments } => add_chat_call(&mut messages, id, &names.wire(name), arguments),
            Item::ToolResult { call_id, content, .. } => messages.push(json!({"role":"tool","tool_call_id":call_id,"content":chat_content(content)?})),
            Item::ServerToolCall { id, name, input } => add_chat_call(&mut messages, id, &names.wire(name), &input.to_string()),
            Item::ServerToolResult { call_id, output, .. } => messages.push(json!({"role":"tool","tool_call_id":call_id,"content":output.to_string()})),
        }
    }
    let thinking_on = config.thinking_toggle && turn.reasoning.enabled.unwrap_or(true);
    if thinking_on {
        // Thinking-mode upstreams reject a tool-call loop whose assistant
        // messages lack `reasoning_content`, even when the model produced no
        // reasoning for that step (an empty string is accepted).
        for message in &mut messages {
            if message["role"] == "assistant" && message["tool_calls"].is_array() && message.get("reasoning_content").is_none() {
                message["reasoning_content"] = json!("");
            }
        }
    }
    let mut request = json!({"model":turn.model,"messages":messages,"stream":true,"stream_options":{"include_usage":true}});
    if !turn.tools.is_empty() {
        request["tools"] = json!(turn.tools.iter().map(|tool| {
            let mut function = json!({"name":names.wire(&tool.name),"parameters":tool.parameters});
            if let Some(description) = &tool.description { function["description"] = json!(description); }
            if tool.strict { function["strict"] = json!(true); }
            json!({"type":"function","function":function})
        }).collect::<Vec<_>>());
        request["tool_choice"] = match &turn.tool_choice { ToolChoice::Auto => json!("auto"), ToolChoice::None => json!("none"),
            ToolChoice::Required => json!("required"), ToolChoice::Named { name } => json!({"type":"function","function":{"name":names.wire(name)}}) };
        if let Some(parallel) = turn.parallel_tool_calls { request["parallel_tool_calls"] = json!(parallel); }
    }
    let thinking = turn.reasoning.enabled.unwrap_or(config.thinking_toggle);
    if config.thinking_toggle {
        request["thinking"] = json!({"type":if thinking { "enabled" } else { "disabled" }});
        if thinking && matches!(turn.tool_choice, ToolChoice::Required | ToolChoice::Named { .. }) {
            return Err(GatewayError::unsupported("configured thinking mode does not support required or named tool choice; disable thinking"));
        }
    }
    if let Some(effort) = &turn.reasoning.effort { request["reasoning_effort"] = json!(effort); }
    else if !config.thinking_toggle && turn.reasoning.enabled == Some(false) { request["reasoning_effort"] = json!("none"); }
    if let Some(max) = turn.max_output_tokens { request["max_tokens"] = json!(max); }
    if !(config.thinking_toggle && thinking) { if let Some(t) = turn.sampling.temperature { request["temperature"] = json!(t); } }
    if let Some(p) = turn.sampling.top_p { request["top_p"] = json!(p); }
    if let Some(seed) = turn.sampling.seed { request["seed"] = json!(seed); }
    if !turn.sampling.stop.is_empty() { request["stop"] = json!(turn.sampling.stop); }
    if let Some(format) = &turn.response_format {
        request["response_format"] = if format.get("schema").is_some() && (format.get("type").is_none() || format["type"] == "json_schema") {
            let mut schema = format.clone();
            schema.as_object_mut().expect("schema format object").remove("type");
            json!({"type":"json_schema","json_schema":schema})
        } else { format.clone() };
    }
    Ok(request)
}
fn append_block(messages: &mut Vec<Value>, role: &str, block: Value) {
    if messages.last().is_none_or(|m| m["role"] != role) { messages.push(json!({"role":role,"content":[]})); }
    messages.last_mut().unwrap()["content"].as_array_mut().unwrap().push(block);
}
fn anthropic(turn: &TurnRequest, config: &UpstreamConfig, names: &WireNames) -> Result<Value, GatewayError> {
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
                append_block(&mut messages, "assistant", json!({"type":"tool_use","id":id,"name":names.wire(name),"input":input}));
            }
            Item::ToolResult { call_id, content, is_error } => append_block(&mut messages, "user", json!({"type":"tool_result","tool_use_id":call_id,"content":parts(content, Flavor::Anthropic)?,"is_error":is_error})),
            Item::ServerToolCall { id, name, input } => append_block(&mut messages, "assistant", json!({"type":"tool_use","id":id,"name":names.wire(name),"input":input})),
            Item::ServerToolResult { call_id, output, .. } => append_block(&mut messages, "user", json!({"type":"tool_result","tool_use_id":call_id,"content":output.to_string()})),
        }
    }
    let mut request = json!({"model":turn.model,"messages":messages,"stream":true,"max_tokens":turn.max_output_tokens.unwrap_or(4096)});
    if !system.is_empty() { request["system"] = json!(system); }
    if !turn.tools.is_empty() {
        request["tools"] = json!(turn.tools.iter().map(|t| {
            let mut tool = json!({"name":names.wire(&t.name),"description":t.description.clone().unwrap_or_default(),"input_schema":t.parameters});
            if t.strict { tool["strict"] = json!(true); }
            tool
        }).collect::<Vec<_>>());
        request["tool_choice"] = match &turn.tool_choice { ToolChoice::Auto => json!({"type":"auto"}), ToolChoice::None => json!({"type":"none"}),
            ToolChoice::Required => json!({"type":"any"}), ToolChoice::Named { name } => json!({"type":"tool","name":names.wire(name)}) };
        if let Some(parallel) = turn.parallel_tool_calls { request["tool_choice"]["disable_parallel_tool_use"] = json!(!parallel); }
    }
    if let Some(enabled) = turn.reasoning.enabled {
        request["thinking"] = if enabled { json!({"type":"enabled","budget_tokens":turn.reasoning.budget_tokens.unwrap_or(1024)}) } else { json!({"type":"disabled"}) };
    }
    if let Some(effort) = &turn.reasoning.effort { request["output_config"]["effort"] = json!(effort); }
    if !(config.thinking_toggle && turn.reasoning.enabled != Some(false)) { if let Some(t) = turn.sampling.temperature { request["temperature"] = json!(t); } }
    if let Some(p) = turn.sampling.top_p { request["top_p"] = json!(p); }
    if let Some(k) = turn.sampling.top_k { request["top_k"] = json!(k); }
    if !turn.sampling.stop.is_empty() { request["stop_sequences"] = json!(turn.sampling.stop); }
    if let Some(format) = &turn.response_format {
        if let Some(schema) = format.get("schema").or_else(|| format["json_schema"].get("schema")) { request["output_config"]["format"] = json!({"type":"json_schema","schema":schema}); }
        else { return Err(GatewayError::unsupported("Anthropic upstream requires a JSON schema for structured output")); }
    }
    Ok(request)
}

/// Realtime commits base64 PCM16 mono at 24 kHz. Chat audio inputs use a WAV container.
fn chat_audio(format: &str, data: &str) -> Result<Value, GatewayError> {
    if format != "pcm16" { return Ok(json!({"type":"input_audio","input_audio":{"format":format,"data":data}})); }
    use base64::{Engine, engine::general_purpose::STANDARD};
    let pcm = STANDARD.decode(data).map_err(|_| GatewayError::invalid("invalid PCM16 audio base64").with_param("audio"))?;
    if pcm.len() % 2 != 0 { return Err(GatewayError::invalid("PCM16 audio must contain whole samples").with_param("audio")); }
    let len = u32::try_from(pcm.len()).ok().filter(|n| *n <= u32::MAX - 36).ok_or_else(|| GatewayError::invalid("PCM16 audio exceeds WAV size limit"))?;
    let mut wav = Vec::with_capacity(pcm.len()+44);
    wav.extend_from_slice(b"RIFF"); wav.extend_from_slice(&(len+36).to_le_bytes()); wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); wav.extend_from_slice(&1u16.to_le_bytes()); wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&24000u32.to_le_bytes()); wav.extend_from_slice(&48000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes()); wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data"); wav.extend_from_slice(&len.to_le_bytes()); wav.extend_from_slice(&pcm);
    Ok(json!({"type":"input_audio","input_audio":{"format":"wav","data":STANDARD.encode(wav)}}))
}
