//! OpenAI requests mapped to the Mistral checkpoint template's own context.
//!
//! The template reads `tools`, `response_format` (rendered into
//! `[MODEL_SETTINGS]` as the provider does) and `reasoning_effort`
//! (low/medium/high, default high). It has no thinking-off form and no
//! tool-choice text: `mistral_common` renders `tool_choice` identically for
//! every value, so selection is enforced by constrained decoding only.
use serde_json::{json, Value};

use crate::openai::chat::glm5::prompt::reasoning_effort;

#[derive(Debug, Clone)]
pub struct MistralPromptOptions {
    pub tool_names: Vec<String>,
    pub response_format: Option<Value>,
}

/// The template's effort: an explicit low/medium/high, or the template default.
/// Thinking-off spellings are rejected rather than rendered as another effort.
pub fn effort(body: &Value) -> Result<Option<String>, String> {
    let switch = crate::openai::chat::glm5::prompt::resolve_thinking(body)?;
    let alias = body.get("chat_template_kwargs").and_then(|v| v.get("thinking")).and_then(Value::as_bool);
    if !switch || alias == Some(false) {
        return Err("Mistral checkpoint template does not support disabling thinking; \
            use reasoning_effort low, medium or high".into());
    }
    match reasoning_effort(body)? {
        None => Ok(None),
        Some(effort) if matches!(effort.as_str(), "low" | "medium" | "high") => Ok(Some(effort)),
        Some(effort) => Err(format!("reasoning_effort must be one of low, medium, high for this checkpoint, got {effort}")),
    }
}

pub fn template_context(body: &Value, options: &MistralPromptOptions, bos: &str, eos: &str) -> Result<Value, String> {
    let effort = effort(body)?;
    let raw = body.get("messages").and_then(Value::as_array).ok_or("messages must be an array")?;
    if raw.is_empty() { return Err("messages must not be empty".into()); }
    let messages = raw.iter().enumerate().map(|(index, message)| template_message(message, index))
        .collect::<Result<Vec<_>, _>>()?;
    let mut context = json!({"messages": messages, "bos_token": bos, "eos_token": eos, "add_generation_prompt": true});
    if !options.tool_names.is_empty() {
        let tools = body.get("tools").and_then(Value::as_array).ok_or("tools must be an array")?;
        context["tools"] = Value::Array(tools.iter().filter(|tool| tool["function"]["name"].as_str()
            .is_some_and(|name| options.tool_names.iter().any(|wanted| wanted == name))).cloned().collect());
    }
    if let Some(format) = options.response_format.as_ref().filter(|format| format["type"] != "regex") {
        context["response_format"] = format.clone();
    }
    if let Some(effort) = effort { context["reasoning_effort"] = Value::String(effort); }
    Ok(context)
}

/// The fields the template reads. Tool-call arguments that are a JSON string
/// become the value it holds, so the template's `tojson` spells them as the
/// provider does (`mistral_common` re-serializes them); text that is not JSON
/// (a length-cut call) renders verbatim.
fn template_message(message: &Value, index: usize) -> Result<Value, String> {
    let object = message.as_object().ok_or_else(|| format!("messages[{index}] must be an object"))?;
    let mut out = serde_json::Map::new();
    for key in ["role", "content", "reasoning_content", "reasoning", "tool_call_id", "name"] {
        if let Some(value) = object.get(key).filter(|value| !value.is_null()) {
            out.insert(key.into(), value.clone());
        }
    }
    if !out.contains_key("content") { out.insert("content".into(), Value::Null); }
    if let Some(calls) = object.get("tool_calls").filter(|v| !v.is_null()) {
        let calls = calls.as_array().ok_or_else(|| format!("messages[{index}].tool_calls must be an array"))?;
        let converted = calls.iter().enumerate().map(|(call_index, call)| {
            let path = format!("messages[{index}].tool_calls[{call_index}]");
            let function = call.get("function").and_then(Value::as_object)
                .ok_or_else(|| format!("{path}.function must be an object"))?;
            let name = function.get("name").and_then(Value::as_str)
                .ok_or_else(|| format!("{path}.function.name must be a string"))?;
            let arguments = match function.get("arguments") {
                None | Some(Value::Null) => Value::String(String::new()),
                Some(Value::String(text)) => serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone())),
                Some(value) => value.clone(),
            };
            let mut out = json!({"type": "function", "function": {"name": name, "arguments": arguments}});
            if let Some(id) = call.get("id") { out["id"] = id.clone(); }
            Ok(out)
        }).collect::<Result<Vec<_>, String>>()?;
        out.insert("tool_calls".into(), Value::Array(converted));
    }
    Ok(Value::Object(out))
}
