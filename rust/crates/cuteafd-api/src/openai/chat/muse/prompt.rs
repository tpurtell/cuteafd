//! OpenAI requests mapped to the Muse checkpoint's own template context.
//!
//! The template takes `reasoning_effort` in {low, medium, high, xhigh}
//! (default xhigh) and always reasons privately (`to=self`); there is no
//! thinking-off form, so such requests are refused rather than rendered under
//! another family's convention. Tool-call arguments become mappings (the
//! template's sandbox has no JSON parser), `current_date` is the server's.
use serde_json::{json, Value};
use crate::openai::chat::glm5::prompt::{reasoning_effort, response_format_instruction, template_message};
pub use crate::openai::chat::glm5::GlmToolChoice as MuseToolChoice;

/// The template's effort names; anything else raises in the template itself.
pub const EFFORTS: [&str; 4] = ["low", "medium", "high", "xhigh"];

#[derive(Debug, Clone)]
pub struct MusePromptOptions {
    pub tool_names: Vec<String>,
    pub tool_choice: MuseToolChoice,
    pub response_format: Option<Value>,
}

/// The template always opens private reasoning; refuse an off request.
pub fn validate_thinking(body: &Value) -> Result<(), String> {
    if !crate::openai::chat::glm5::prompt::resolve_thinking(body)? {
        return Err("Muse checkpoint template does not support disabling thinking; \
            use reasoning_effort low for the shortest reasoning".into());
    }
    Ok(())
}

/// The request's effort mapped onto the template's names: `minimal` reads as
/// `low`, `max` as `xhigh`; absent keeps the template default.
pub fn template_effort(body: &Value) -> Result<Option<String>, String> {
    let Some(effort) = reasoning_effort(body)? else { return Ok(None) };
    let effort = match effort.as_str() {
        "minimal" => "low".to_owned(),
        "max" => "xhigh".to_owned(),
        other => other.to_owned(),
    };
    if !EFFORTS.contains(&effort.as_str()) {
        return Err(format!("reasoning_effort {effort:?} is not one of the Muse template's {EFFORTS:?}"));
    }
    Ok(Some(effort))
}

pub fn template_context(body: &Value, options: &MusePromptOptions, bos: &str) -> Result<Value, String> {
    validate_thinking(body)?;
    let raw = body.get("messages").and_then(Value::as_array).ok_or("messages must be an array")?;
    if raw.is_empty() { return Err("messages must not be empty".into()); }
    let mut messages = raw.iter().enumerate().map(|(index, message)| {
        let mut converted = template_message(message, index)?;
        // The template reads either reasoning field; keep the OpenAI-compatible spelling too.
        if let Some(reasoning) = message.get("reasoning").filter(|v| v.is_string()) {
            if converted.get("reasoning_content").is_none() { converted["reasoning_content"] = reasoning.clone(); }
        }
        Ok(converted)
    }).collect::<Result<Vec<_>, String>>()?;
    let mut instructions = Vec::new();
    if let Some(text) = options.response_format.as_ref().and_then(response_format_instruction) {
        instructions.push(text);
    }
    if !options.tool_names.is_empty() {
        match &options.tool_choice {
            MuseToolChoice::Auto => {}
            MuseToolChoice::Required => instructions.push("You must call at least one provided function.".into()),
            MuseToolChoice::Named(name) => instructions.push(format!("You must call the function {name}.")),
        }
    }
    if !instructions.is_empty() {
        // The template folds only the first system/developer message into its
        // system block; merge ours into it rather than adding a second one.
        let extra = instructions.join("\n\n");
        match messages.first_mut() {
            Some(first) if matches!(first["role"].as_str(), Some("system" | "developer"))
                && first["content"].is_string() => {
                let text = first["content"].as_str().unwrap_or_default();
                first["content"] = Value::String(if text.is_empty() { extra } else { format!("{text}\n\n{extra}") });
            }
            _ => messages.insert(0, json!({"role": "system", "content": extra})),
        }
    }
    let mut context = json!({"messages": messages, "bos_token": bos, "add_generation_prompt": true});
    if !options.tool_names.is_empty() {
        let tools = body.get("tools").and_then(Value::as_array).ok_or("tools must be an array")?;
        context["tools"] = Value::Array(tools.iter().filter(|tool| tool["function"]["name"].as_str()
            .is_some_and(|name| options.tool_names.iter().any(|wanted| wanted == name))).cloned().collect());
    }
    if let Some(effort) = template_effort(body)? { context["reasoning_effort"] = Value::String(effort); }
    Ok(context)
}
