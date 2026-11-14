//! Preserve the checkpoint's role mapping and recursive XML argument rendering.
use serde_json::{json, Value};
use crate::openai::chat::glm5::prompt::{reasoning_effort, resolve_thinking};

#[derive(Debug, Clone, Default)]
pub struct MiniMaxPromptOptions {
    pub tool_names: Vec<String>,
}

pub fn effort(body: &Value) -> Result<Option<String>, String> {
    let kwargs = body.get("chat_template_kwargs");
    if !resolve_thinking(body)? || kwargs.and_then(|v| v.get("thinking")).and_then(Value::as_bool) == Some(false) {
        return Err("MiniMax checkpoint template does not support disabling thinking; thinking is adaptive".into());
    }
    reasoning_effort(body)
}

pub fn template_context(body: &Value, options: &MiniMaxPromptOptions) -> Result<Value, String> {
    let effort = effort(body)?;
    let raw = body.get("messages").and_then(Value::as_array).ok_or("messages must be an array")?;
    if raw.is_empty() { return Err("messages must not be empty".into()); }
    let mut messages = raw.clone();
    for (index, message) in messages.iter_mut().enumerate() {
        if !message.is_object() { return Err(format!("messages[{index}] must be an object")); }
        if message["content"].is_null() { message["content"] = Value::Null; }
        if let Some(calls) = message.get_mut("tool_calls").filter(|v| !v.is_null()) {
            for call in calls.as_array_mut().ok_or("tool_calls must be an array")? {
                let args = &mut call["function"]["arguments"];
                if let Some(text) = args.as_str() {
                    *args = serde_json::from_str(text).map_err(|error| format!("invalid MiniMax tool-call arguments: {error}"))?;
                }
                if !args.is_object() { return Err("MiniMax tool-call arguments must be an object".into()); }
            }
        }
    }
    let tools: Vec<_> = body["tools"].as_array().into_iter().flatten().filter(|tool| {
        tool["function"]["name"].as_str().is_some_and(|name| options.tool_names.iter().any(|want| want == name))
    }).cloned().collect();
    let mut context = json!({"messages": messages, "tools": tools, "add_generation_prompt": true});
    if let Some(effort) = effort { context["reasoning_effort"] = Value::String(effort); }
    Ok(context)
}
