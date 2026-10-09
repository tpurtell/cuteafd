//! Wire validation and protocol-neutral item/configuration translation.
use super::super::{
    error::GatewayError,
    turn::{ImageSource, Item, Part, Role, ToolChoice, ToolSpec, TurnRequest},
};
use serde_json::{json, Value};

pub fn id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}
pub fn invalid(param: &str, message: &str) -> GatewayError {
    GatewayError::invalid(message).with_param(param)
}
pub fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, GatewayError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(key, "required string field"))
}
pub fn error(error: GatewayError, event_id: Option<&str>, code: Option<&str>) -> Value {
    let mut body = error.openai_body()["error"].clone();
    body["event_id"] = json!(event_id);
    if let Some(code) = code {
        body["code"] = json!(code);
    }
    json!({"type":"error", "error":body})
}

pub fn parse_item(value: &Value, format: &str) -> Result<Item, GatewayError> {
    match string(value, "type")? {
        "message" => {
            let role = match string(value, "role")? {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "system" => Role::System,
                _ => {
                    return Err(invalid(
                        "item.role",
                        "role must be user, assistant or system",
                    ))
                }
            };
            let parts = value["content"]
                .as_array()
                .ok_or_else(|| invalid("item.content", "content must be an array"))?;
            let mut content = Vec::new();
            for part in parts {
                content.push(match string(part, "type")? {
                    "input_text" if role != Role::Assistant => Part::text(string(part, "text")?),
                    "text" | "output_text" if role == Role::Assistant => {
                        Part::text(string(part, "text")?)
                    }
                    "input_audio" if role == Role::User => {
                        let data = string(part, "audio")?;
                        super::audio::decode(data, format)?;
                        Part::Audio {
                            format: format.into(),
                            data: data.into(),
                        }
                    }
                    "input_image" if role == Role::User => Part::Image {
                        source: ImageSource::Url {
                            url: string(part, "image_url")?.into(),
                        },
                        detail: part
                            .get("detail")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                    },
                    _ => return Err(invalid(
                        "item.content.type",
                        "unsupported content type for role; assistant audio cannot be populated",
                    )),
                });
            }
            if content.is_empty() {
                return Err(invalid("item.content", "content cannot be empty"));
            }
            Ok(Item::Message { role, content })
        }
        "function_call" => Ok(Item::ToolCall {
            id: string(value, "call_id")?.into(),
            name: string(value, "name")?.into(),
            arguments: string(value, "arguments")?.into(),
        }),
        "function_call_output" => Ok(Item::ToolResult {
            call_id: string(value, "call_id")?.into(),
            content: vec![Part::text(string(value, "output")?)],
            is_error: false,
        }),
        _ => Err(invalid("item.type", "unsupported conversation item type")),
    }
}

pub fn wire_item(id: &str, item: &Item, status: &str, beta: bool) -> Value {
    match item {
        Item::Message { role, content } => {
            let content: Vec<Value> = content.iter().map(|part| match part {
                Part::Text { text } => json!({"type": if *role == Role::Assistant {if beta {"text"} else {"output_text"}} else {"input_text"}, "text":text}),
                Part::Audio { data, .. } => json!({"type":"input_audio", "audio":data, "transcript":null}),
                Part::Image { source: ImageSource::Url { url }, detail } => json!({"type":"input_image", "image_url":url, "detail":detail}),
                _ => json!({"type":"input_text", "text":""}),
            }).collect();
            json!({"id":id, "object":"realtime.item", "type":"message", "status":status, "role":role, "content":content})
        }
        Item::ToolCall {
            id: call_id,
            name,
            arguments,
        } => {
            json!({"id":id,"object":"realtime.item","type":"function_call","status":status,"call_id":call_id,"name":name,"arguments":arguments})
        }
        Item::ToolResult {
            call_id, content, ..
        } => {
            json!({"id":id,"object":"realtime.item","type":"function_call_output","status":status,"call_id":call_id,"output": content.iter().filter_map(|p| if let Part::Text {text} = p {Some(text.as_str())} else {None}).collect::<String>()})
        }
        _ => json!({"id":id,"type":"message","role":"assistant","status":status,"content":[]}),
    }
}

pub fn defaults(model: &str, beta: bool, transcription: bool) -> Value {
    if beta {
        json!({"object":if transcription {"realtime.transcription_session"} else {"realtime.session"},"model":model,"instructions":"","modalities":["text"],"temperature":0.8,"max_response_output_tokens":"inf","tools":[],"tool_choice":"auto","input_audio_format":"pcm16","output_audio_format":"pcm16","input_audio_transcription":null,"turn_detection":{"type":"server_vad","threshold":0.5,"prefix_padding_ms":300,"silence_duration_ms":500,"create_response":true,"interrupt_response":true}})
    } else {
        json!({"object":"realtime.session","type":if transcription {"transcription"} else {"realtime"},"model":model,"instructions":"","output_modalities":["text"],"max_output_tokens":"inf","tools":[],"tool_choice":"auto","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":null,"turn_detection":{"type":"server_vad","threshold":0.5,"prefix_padding_ms":300,"silence_duration_ms":500,"create_response":true,"interrupt_response":true}},"output":{"format":{"type":"audio/pcm","rate":24000}}}})
    }
}

// Merge nested audio configuration while preserving explicit null (VAD disabled).
fn merge(base: &mut Value, patch: &Value) {
    if let (Some(base), Some(patch)) = (base.as_object_mut(), patch.as_object()) {
        for (key, value) in patch {
            if value.is_object() && base.get(key).is_some_and(Value::is_object) {
                merge(base.get_mut(key).unwrap(), value);
            } else {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}
pub fn updated(
    config: &Value,
    patch: &Value,
    beta: bool,
    response: bool,
) -> Result<Value, GatewayError> {
    let fields = if response {
        vec![
            "instructions",
            "output_modalities",
            "modalities",
            "temperature",
            "max_output_tokens",
            "max_response_output_tokens",
            "tools",
            "tool_choice",
            "audio",
            "conversation",
            "input",
            "metadata",
            "prompt",
            "voice",
            "output_audio_format",
            "parallel_tool_calls",
            "reasoning",
        ]
    } else {
        vec![
            "type",
            "model",
            "instructions",
            "output_modalities",
            "modalities",
            "temperature",
            "max_output_tokens",
            "max_response_output_tokens",
            "tools",
            "tool_choice",
            "audio",
            "input_audio_format",
            "output_audio_format",
            "input_audio_transcription",
            "turn_detection",
            "input_audio_noise_reduction",
            "voice",
            "speed",
            "tracing",
            "truncation",
            "prompt",
            "include",
            "parallel_tool_calls",
            "reasoning",
        ]
    };
    let obj = patch
        .as_object()
        .ok_or_else(|| invalid("session", "configuration must be an object"))?;
    for key in obj.keys() {
        if !fields.contains(&key.as_str()) {
            return Err(invalid(key, "unknown configuration parameter"));
        }
    }
    for key in [
        "prompt",
        "tracing",
        "input_audio_noise_reduction",
        "include",
    ] {
        if obj.get(key).is_some_and(|v| !v.is_null()) {
            return Err(GatewayError::unsupported(format!(
                "{key} is not configured on this gateway"
            ))
            .with_param(key));
        }
    }
    if obj.get("truncation").is_some_and(|v| v != "disabled") {
        return Err(GatewayError::unsupported(
            "automatic context truncation is not available; use disabled",
        )
        .with_param("truncation"));
    }
    if !response && obj.get("model").is_some_and(|m| m != &config["model"]) {
        return Err(invalid(
            "model",
            "model cannot be changed in an existing session",
        ));
    }
    if obj.get("type").is_some_and(|v| v != &config["type"]) {
        return Err(invalid("type", "session type cannot be changed"));
    }
    if obj
        .get("parallel_tool_calls")
        .is_some_and(|v| !v.is_boolean())
    {
        return Err(invalid("parallel_tool_calls", "must be a boolean"));
    }
    if let Some(reasoning) = obj.get("reasoning") {
        if !reasoning.is_object() || reasoning["effort"].as_str().is_none() {
            return Err(invalid(
                "reasoning.effort",
                "reasoning requires an effort string",
            ));
        }
    }
    let wrong_max = if beta {
        "max_output_tokens"
    } else {
        "max_response_output_tokens"
    };
    if obj.contains_key(wrong_max) {
        return Err(invalid(
            wrong_max,
            "parameter belongs to the other Realtime protocol version",
        ));
    }
    if !beta {
        for key in [
            "turn_detection",
            "input_audio_format",
            "output_audio_format",
            "input_audio_transcription",
            "voice",
            "speed",
        ] {
            if obj.contains_key(key) {
                return Err(invalid(
                    key,
                    "GA audio settings belong under audio.input or audio.output",
                ));
            }
        }
    } else if obj.contains_key("audio") {
        return Err(invalid(
            "audio",
            "beta audio configuration uses top-level fields",
        ));
    }
    if let Some(audio) = obj.get("audio") {
        let a = audio
            .as_object()
            .ok_or_else(|| invalid("audio", "audio must be an object"))?;
        for (direction, value) in a {
            if !["input", "output"].contains(&direction.as_str()) {
                return Err(invalid("audio", "unknown audio configuration field"));
            }
            let fields = value
                .as_object()
                .ok_or_else(|| invalid("audio", "audio input/output must be objects"))?;
            for key in fields.keys() {
                let allowed = if direction == "input" {
                    vec![
                        "format",
                        "transcription",
                        "turn_detection",
                        "noise_reduction",
                    ]
                } else {
                    vec!["format", "voice", "speed"]
                };
                if !allowed.contains(&key.as_str()) {
                    return Err(invalid("audio", "unknown audio input/output field"));
                }
            }
        }
    }
    let mut result = config.clone();
    merge(&mut result, patch);
    let modality_key = if beta {
        "modalities"
    } else {
        "output_modalities"
    };
    if let Some(mods) = result.get(modality_key) {
        if mods != &json!(["text"]) {
            return Err(GatewayError::unsupported(
                "audio output is unavailable: no Synthesizer configured; request [\"text\"]",
            )
            .with_param(modality_key));
        }
    }
    let wrong = if beta {
        "output_modalities"
    } else {
        "modalities"
    };
    if obj.contains_key(wrong) {
        return Err(invalid(
            wrong,
            "parameter belongs to the other Realtime protocol version",
        ));
    }
    if let Some(t) = result.get("temperature") {
        if !beta {
            return Err(invalid(
                "temperature",
                "temperature is not supported by the GA interface",
            ));
        }
        let t = t
            .as_f64()
            .ok_or_else(|| invalid("temperature", "temperature must be a number"))?;
        if !(0.6..=1.2).contains(&t) {
            return Err(invalid(
                "temperature",
                "temperature must be between 0.6 and 1.2",
            ));
        }
    }
    let max_key = if beta {
        "max_response_output_tokens"
    } else {
        "max_output_tokens"
    };
    let max = &result[max_key];
    if max != "inf" && !max.as_u64().is_some_and(|n| (1..=4096).contains(&n)) {
        return Err(invalid(max_key, "token budget must be 1..4096 or inf"));
    }
    if !result["instructions"].is_string() {
        return Err(invalid("instructions", "instructions must be a string"));
    }
    tools(&result["tools"])?;
    choice(&result["tool_choice"])?;
    super::audio::settings(&result, beta)?;
    if let Some(audio) = obj.get("audio") {
        if audio
            .pointer("/input/noise_reduction")
            .is_some_and(|v| !v.is_null())
        {
            return Err(
                GatewayError::unsupported("audio noise reduction is not configured")
                    .with_param("audio.input.noise_reduction"),
            );
        }
    }
    if let Some(conversation) = obj.get("conversation") {
        if conversation != "auto" && conversation != "none" {
            return Err(invalid("conversation", "conversation must be auto or none"));
        }
    }
    if let Some(metadata) = obj.get("metadata") {
        if !metadata.is_null() {
            let m = metadata
                .as_object()
                .ok_or_else(|| invalid("metadata", "metadata must be an object"))?;
            if m.len() > 16
                || m.iter()
                    .any(|(k, v)| k.len() > 64 || v.as_str().is_none_or(|s| s.len() > 512))
            {
                return Err(invalid(
                    "metadata",
                    "metadata accepts at most 16 string pairs, keys <=64 and values <=512",
                ));
            }
        }
    }
    Ok(result)
}
fn tools(value: &Value) -> Result<Vec<ToolSpec>, GatewayError> {
    value
        .as_array()
        .ok_or_else(|| invalid("tools", "tools must be an array"))?
        .iter()
        .map(|tool| {
            if tool["type"] != "function" {
                return Err(
                    GatewayError::unsupported("only function tools are supported")
                        .with_param("tools.type"),
                );
            }
            let parameters = tool
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type":"object","properties":{}}));
            if !parameters.is_object() {
                return Err(invalid(
                    "tools.parameters",
                    "parameters must be a JSON schema object",
                ));
            }
            Ok(ToolSpec {
                name: string(tool, "name")?.into(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                parameters,
                strict: false,
            })
        })
        .collect()
}
fn choice(value: &Value) -> Result<ToolChoice, GatewayError> {
    match value.as_str() {
        Some("auto") => Ok(ToolChoice::Auto),
        Some("none") => Ok(ToolChoice::None),
        Some("required") => Ok(ToolChoice::Required),
        _ if value["type"] == "function" => Ok(ToolChoice::Named {
            name: string(value, "name")?.into(),
        }),
        _ => Err(invalid("tool_choice", "invalid tool choice")),
    }
}
pub fn turn(config: &Value, beta: bool, items: Vec<Item>) -> Result<TurnRequest, GatewayError> {
    let key = if beta {
        "max_response_output_tokens"
    } else {
        "max_output_tokens"
    };
    Ok(TurnRequest {
        requested_model: string(config, "model")?.into(),
        system: Some(string(config, "instructions")?.into()),
        items,
        tools: tools(&config["tools"])?,
        tool_choice: choice(&config["tool_choice"])?,
        max_output_tokens: config[key].as_u64().map(|n| n as u32),
        sampling: super::super::turn::Sampling {
            temperature: config
                .get("temperature")
                .and_then(Value::as_f64)
                .map(|t| t as f32),
            ..Default::default()
        },
        parallel_tool_calls: config.get("parallel_tool_calls").and_then(Value::as_bool),
        reasoning: super::super::turn::Reasoning {
            effort: config
                .pointer("/reasoning/effort")
                .and_then(Value::as_str)
                .map(str::to_owned),
            ..Default::default()
        },
        ..Default::default()
    })
}
