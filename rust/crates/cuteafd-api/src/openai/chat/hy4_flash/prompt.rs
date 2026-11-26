//! Normalize OpenAI chat requests before rendering Hy4-flash's unmodified `chat_template.jinja`.
use crate::openai::chat::glm5::prompt::resolve_thinking;
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct Hy4PromptOptions {
    pub tool_names: Vec<String>,
}

/// Resolve Hy4 / Hy4-flash `reasoning_effort` (`no_think`, `high`, or `max` when supported by template).
pub fn effort(body: &Value) -> Result<String, String> {
    effort_with_max(body, true)
}

pub fn effort_with_max(body: &Value, allow_max: bool) -> Result<String, String> {
    let raw = body
        .get("reasoning_effort")
        .filter(|v| !v.is_null())
        .or_else(|| {
            body.get("chat_template_kwargs")
                .and_then(|v| v.get("reasoning_effort"))
                .filter(|v| !v.is_null())
        });
    let resolved = match raw {
        None => "high".to_owned(),
        Some(Value::String(text)) => match text.trim() {
            "no_think" | "none" => "no_think".to_owned(),
            "high" | "medium" | "low" => "high".to_owned(),
            "max" | "xhigh" => if allow_max { "max" } else { "high" }.to_owned(),
            other => {
                return Err(format!(
                    "unknown Hy4 reasoning_effort: {other} (expected no_think/high{})",
                    if allow_max { "/max" } else { "" }
                ))
            }
        },
        Some(_) => return Err("Hy4 reasoning_effort must be a string (no_think/high/max)".into()),
    };
    let mut switches = body.clone();
    if let Some(obj) = switches.as_object_mut() {
        obj.remove("reasoning_effort");
    }
    if let Some(obj) = switches.get_mut("chat_template_kwargs").and_then(Value::as_object_mut) {
        obj.remove("reasoning_effort");
    }
    if !resolve_thinking(&switches)? {
        return Ok("no_think".to_owned());
    }
    Ok(resolved)
}

pub fn template_context(body: &Value, options: &Hy4PromptOptions) -> Result<Value, String> {
    template_context_with_max(body, options, true)
}

pub fn template_context_with_max(body: &Value, options: &Hy4PromptOptions, allow_max: bool) -> Result<Value, String> {
    let reasoning_effort = effort_with_max(body, allow_max)?;
    let raw = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("messages must be an array")?;
    if raw.is_empty() {
        return Err("messages must not be empty".into());
    }
    let mut messages = raw.clone();
    for (index, message) in messages.iter_mut().enumerate() {
        if !message.is_object() {
            return Err(format!("messages[{index}] must be an object"));
        }
        if let Some(calls) = message.get_mut("tool_calls").filter(|v| !v.is_null()) {
            for call in calls.as_array_mut().ok_or("tool_calls must be an array")? {
                let function = call
                    .get_mut("function")
                    .and_then(Value::as_object_mut)
                    .ok_or("tool call function must be an object")?;
                let args = function.entry("arguments").or_insert(json!({}));
                if let Some(text) = args.as_str() {
                    *args = serde_json::from_str(text)
                        .map_err(|e| format!("invalid Hy4-flash tool-call arguments: {e}"))?;
                }
                if !args.is_object() {
                    return Err("Hy4-flash tool-call arguments must be an object".into());
                }
            }
        }
    }
    let tools: Vec<_> = body["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|tool| {
            tool["function"]["name"]
                .as_str()
                .is_some_and(|name| options.tool_names.iter().any(|want| want == name))
        })
        .cloned()
        .collect();
    Ok(json!({
        "messages": messages,
        "tools": tools,
        "reasoning_effort": reasoning_effort,
        "add_generation_prompt": true,
    }))
}
