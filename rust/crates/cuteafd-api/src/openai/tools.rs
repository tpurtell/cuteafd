use anyhow::{ensure, Context, Result};
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub(super) struct Selection { pub required: bool, name: Option<String> }
impl Selection {
    pub fn extract(request: &mut deepseek_recipe::openai::ChatCompletionRequest) -> Self {
        use deepseek_recipe::openai::chat_completion::request::{ChatCompletionToolChoiceOption as Choice, ChatCompletionToolChoiceMode as Mode};
        let (required, name) = match request.tool_choice.as_ref() {
            Some(Choice::Named(choice)) => (true, Some(choice.function.name.clone())),
            Some(Choice::Mode(Mode::Required)) => (true, None),
            _ => (false, None),
        };
        // Leave the assistant outside the calls block so it can think first.
        // Native constraints enforce selection after the reasoning terminator.
        if required { request.tool_choice = Some(Choice::Mode(Mode::Auto)); }
        Self { required, name }
    }
    /// The tool a named `tool_choice` selected.
    pub fn name(&self) -> Option<&str> { self.name.as_deref() }
    pub fn apply(&self, tools: &mut Vec<ToolDefinition>) -> Result<()> {
        if let Some(name) = &self.name {
            ensure!(tools.iter().any(|tool| &tool.name == name), "tool_choice: no tool named '{name}' was specified");
            tools.retain(|tool| &tool.name == name);
        }
        ensure!(!self.required || !tools.is_empty(), "required tool_choice needs at least one tool");
        Ok(())
    }
}

/// Tool-call markup a model family emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolSyntax {
    /// DeepSeek V4/V4.1 `<｜DSML｜ calls>` blocks (engine-side schema grammar).
    Dsml,
    /// GLM `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
    /// (XGrammar's native `glm_xml` JSON-schema style, as glmrt served it).
    GlmXml,
    /// Qwen3-Coder `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`
    /// (XGrammar's `qwen_xml` style, as its `qwen_3_5` structural tag).
    QwenXml,
    /// Kimi XTML; assisted decoding uses its native JSON-object call form.
    KimiXtml,
}

pub(super) struct ToolConstraints {
    pub required: bool,
    /// At least one tool call: the required branch, or the tool alternative
    /// to a response format.
    pub format: Option<Value>,
    /// Free text until a call trigger, then tool calls only.
    pub triggered: Option<Value>,
    parallel: bool,
    validators: BTreeMap<String, Option<jsonschema::JSONSchema>>,
}

impl ToolConstraints {
    pub fn new(tools: &[ToolDefinition], choice: ToolChoice, required: bool, parallel: bool, assistance: bool,
        syntax: ToolSyntax) -> Result<Option<Self>> {
        if tools.is_empty() || choice == ToolChoice::None { return Ok(None); }
        let mut validators = BTreeMap::new();
        let mut tags = Vec::new();
        let mut strict_requested = false;
        for tool in tools {
            let strict = tool.strict.unwrap_or(false);
            strict_requested |= strict;
            let schema = if strict { tool.parameters.clone() }
                else { json!({"type":"object", "additionalProperties":true}) };
            let validator = if strict { Some(super::constraints::compile_schema(&schema)
                .with_context(|| format!("invalid parameters for tool {}", tool.name))?) } else { None };
            validators.insert(tool.name.clone(), validator);
            tags.push(match syntax {
                ToolSyntax::KimiXtml => json!({"type":"tag",
                    "begin":format!("<|open|>call tool=\"{}\" index=\"", tool.name.replace('&', "&amp;").replace('"', "&quot;")),
                    "content":{"type":"sequence", "elements":[
                        {"type":"regex", "pattern":"[1-9][0-9]*"},
                        {"type":"const_string", "value":"\"<|sep|><|open|>json type=\"object\"<|sep|>"},
                        {"type":"ds41_json_schema", "strict":strict, "json_schema":schema},
                        {"type":"const_string", "value":"<|close|>json<|sep|>"}]},
                    "end":"<|close|>call<|sep|>"}),
                ToolSyntax::Dsml => json!({"type":"tag", "begin":format!("<｜DSML｜ invoke name=\"{}\">", tool.name),
                    "content":{"type":"ds41_tool_schema", "strict":strict, "json_schema":schema},
                    "end":"</｜DSML｜ invoke>"}),
                ToolSyntax::GlmXml => {
                    // An absent parameter schema converts to `{}`; glmrt
                    // constrained such tools to zero arguments.
                    let schema = if strict && schema.as_object().is_some_and(|object| object.is_empty()) {
                        json!({"type":"object", "properties":{}, "required":[], "additionalProperties":false})
                    } else { schema };
                    json!({"type":"tag", "begin":format!("<tool_call>{}", tool.name),
                        "content":{"type":"json_schema", "json_schema":schema, "style":"glm_xml", "any_order":false},
                        "end":"</tool_call>"})
                }
                ToolSyntax::QwenXml => json!({"type":"tag", "begin":format!("<tool_call>\n<function={}>\n", tool.name),
                    "content":{"type":"json_schema", "json_schema":schema, "style":"qwen_xml", "any_order":false},
                    "end":"\n</function>\n</tool_call>"}),
            });
        }
        if !(assistance || strict_requested || required) {
            return Ok(Some(Self { required, format: None, triggered: None, parallel, validators }));
        }
        let (format, triggered) = match syntax {
            ToolSyntax::KimiXtml => {
                let calls = json!({"type":"tag", "begin":"<|open|>tools<|sep|>",
                    "content":{"type":"tags_with_separator", "tags":tags, "separator":"",
                        "at_least_one":true, "stop_after_first":!parallel},
                    "end":"<|close|>tools<|sep|>"});
                // XTML response composition owns the optional/required branch.
                (calls.clone(), calls)
            }
            ToolSyntax::Dsml => {
                let calls = json!({"type":"tag",
                    "begin":"<｜DSML｜ calls>",
                    "content":{"type":"sequence", "elements":[
                        {"type":"regex", "pattern":"[ \\n\\t]{0,16}"},
                        {"type":"tags_with_separator", "tags":tags, "separator":"\n", "at_least_one":true, "stop_after_first":!parallel},
                        {"type":"regex", "pattern":"[ \\n\\t]{0,16}"}]},
                    "end":"</｜DSML｜ calls>"});
                let triggered = json!({"type":"triggered_tags", "triggers":["<｜DSML｜ calls>"],
                    "tags":[calls.clone()], "at_least_one":false, "stop_after_first":true});
                (calls, triggered)
            }
            ToolSyntax::GlmXml => (
                json!({"type":"tags_with_separator", "tags":tags, "separator":"", "at_least_one":true,
                    "stop_after_first":!parallel}),
                json!({"type":"triggered_tags", "triggers":["<tool_call>"], "tags":tags, "at_least_one":false,
                    "stop_after_first":!parallel}),
            ),
            ToolSyntax::QwenXml => (
                json!({"type":"tags_with_separator", "tags":tags, "separator":"\n", "at_least_one":true,
                    "stop_after_first":!parallel}),
                json!({"type":"triggered_tags", "triggers":["<tool_call>\n<function="], "tags":tags,
                    "at_least_one":false, "stop_after_first":!parallel}),
            ),
        };
        Ok(Some(Self { required, format: Some(format), triggered: Some(triggered), parallel, validators }))
    }

    fn validate(&self, calls: &BTreeMap<u64, Call>, reason: &str) -> Result<()> {
        if !matches!(reason, "stop" | "tool_calls") { return Ok(()); }
        ensure!(!self.required || !calls.is_empty(), "native response omitted a required tool call");
        ensure!(self.parallel || calls.len() <= 1, "native response violated parallel_tool_calls=false");
        ensure!(calls.is_empty() || reason == "tool_calls", "native tool response ended without tool_calls finish");
        ensure!(reason != "tool_calls" || !calls.is_empty(), "native tool response has no calls");
        for (expected, (index, call)) in calls.iter().enumerate() {
            ensure!(*index == expected as u64, "native tool call indices are not contiguous");
            let validator = self.validators.get(&call.name)
                .with_context(|| format!("native response selected an unavailable tool: {}", call.name))?;
            let value = unique_json(&call.arguments).context("native tool arguments are not a complete JSON object")?;
            ensure!(value.is_object(), "native tool arguments must be an object");
            if let Some(validator) = validator {
                ensure!(validator.is_valid(&value), "native arguments do not satisfy the schema for tool {}", call.name);
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct Call { name: String, arguments: String }

/// Accumulate only constrained answers and tool arguments, never reasoning.
/// Validation precedes successful finish chunks in both buffered and SSE paths.
pub(super) struct CompletionValidator {
    response: Option<jsonschema::JSONSchema>,
    tools: Option<ToolConstraints>,
    content: String,
    calls: BTreeMap<u64, Call>,
}
impl CompletionValidator {
    pub fn new(response: Option<jsonschema::JSONSchema>, tools: Option<ToolConstraints>) -> Self {
        Self { response, tools, content: String::new(), calls: BTreeMap::new() }
    }
    pub fn enabled(&self) -> bool { self.response.is_some() || self.tools.is_some() }
    pub fn observe(&mut self, chunk: &Value) -> Result<()> {
        for choice in chunk["choices"].as_array().into_iter().flatten() {
            if self.response.is_some() {
                if let Some(text) = choice["delta"]["content"].as_str() { self.content.push_str(text); }
            }
            if self.tools.is_some() {
                for delta in choice["delta"]["tool_calls"].as_array().into_iter().flatten() {
                    let index = delta["index"].as_u64().context("native tool delta has no index")?;
                    let call = self.calls.entry(index).or_default();
                    if let Some(name) = delta["function"]["name"].as_str() { call.name.push_str(name); }
                    if let Some(arguments) = delta["function"]["arguments"].as_str() { call.arguments.push_str(arguments); }
                }
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                if let Some(tools) = &self.tools { tools.validate(&self.calls, reason)?; }
                if reason == "stop" {
                    if let Some(validator) = &self.response {
                        super::constraints::validate_complete(validator, &self.content)?;
                    }
                }
            }
        }
        Ok(())
    }
}

// serde_json::Value normally keeps the last duplicate property. Reject duplicates
// recursively, including escaped aliases, before validating argument schemas.
pub(super) fn unique_json(text: &str) -> Result<Value> {
    use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
    struct Unique(Value);
    struct UniqueVisitor;
    impl<'de> Deserialize<'de> for Unique {
        fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            d.deserialize_any(UniqueVisitor)
        }
    }
    impl<'de> Visitor<'de> for UniqueVisitor {
        type Value = Unique;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("JSON with unique object keys") }
        fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Unique,E> { Ok(Unique(v.into())) }
        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Unique,E> { Ok(Unique(v.into())) }
        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Unique,E> { Ok(Unique(v.into())) }
        fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Unique,E> {
            serde_json::Number::from_f64(v).map(|v| Unique(v.into())).ok_or_else(|| E::custom("non-finite JSON number"))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Unique,E> { Ok(Unique(v.into())) }
        fn visit_unit<E: de::Error>(self) -> std::result::Result<Unique,E> { Ok(Unique(Value::Null)) }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> std::result::Result<Unique,A::Error> {
            let mut values = Vec::new();
            while let Some(Unique(value)) = a.next_element()? { values.push(value); }
            Ok(Unique(values.into()))
        }
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> std::result::Result<Unique,A::Error> {
            let mut values = serde_json::Map::new();
            while let Some(key) = a.next_key::<String>()? {
                if values.contains_key(&key) { return Err(de::Error::custom("duplicate JSON property")); }
                let Unique(value) = a.next_value()?;
                values.insert(key, value);
            }
            Ok(Unique(Value::Object(values)))
        }
    }
    Ok(serde_json::from_str::<Unique>(text)?.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::*;
    use axum::{body::{Body, to_bytes}, http::{Request, StatusCode}};
    use tower::ServiceExt;

    #[test]
    fn arguments_reject_duplicate_and_escaped_alias_keys_recursively() {
        for invalid in [r#"{"x":1,"x":2}"#, r#"{"x":1,"\u0078":2}"#,
            r#"{"a":[{"x":1,"x":2}]}"#, "{} garbage", "[NaN]"] {
            assert!(unique_json(invalid).is_err(), "{invalid}");
        }
        let valid = r#"{"a":[true,null,-1,1.25,"台北",{"x":2}],"x":1}"#;
        assert_eq!(unique_json(valid).unwrap(), serde_json::from_str::<Value>(valid).unwrap());
    }

    #[test]
    fn assistance_opt_out_cannot_disable_explicit_constraints() {
        let mut definitions = vec![ToolDefinition { name:"lookup".into(), description:None,
            parameters:json!({"type":"object","properties":{"s":{"type":"string","pattern":"^x+$","maxLength":1}}}), strict:Some(false) }];
        assert!(ToolConstraints::new(&definitions,ToolChoice::Auto,false,true,false,ToolSyntax::Dsml).unwrap().unwrap().format.is_none());
        assert!(ToolConstraints::new(&definitions,ToolChoice::Auto,true,true,false,ToolSyntax::Dsml).unwrap().unwrap().format.is_some());
        definitions[0].strict=Some(true);
        let tools=ToolConstraints::new(&definitions,ToolChoice::Auto,false,true,false,ToolSyntax::Dsml).unwrap().unwrap();
        assert!(tools.format.is_some());
        let mut calls=BTreeMap::new();
        calls.insert(0,Call { name:"lookup".into(),arguments:r#"{"s":"xx"}"#.into() });
        assert!(tools.validate(&calls,"tool_calls").is_err());
        calls.get_mut(&0).unwrap().arguments=r#"{"s":"x"}"#.into();
        assert!(tools.validate(&calls,"tool_calls").is_ok());
    }

    fn invoke(name: &str, parameters: &str) -> String {
        format!("<｜DSML｜ invoke name=\"{name}\">\n{parameters}\n</｜DSML｜ invoke>")
    }
    fn parameter(name: &str, value: &str, string: bool) -> String {
        format!("<｜DSML｜ parameter name=\"{name}\" string=\"{string}\">{value}</｜DSML｜ parameter>")
    }
    fn block(calls: &[String]) -> String {
        format!("<｜DSML｜ calls>\n{}\n</｜DSML｜ calls>", calls.join("\n"))
    }
    #[tokio::test]
    async fn tool_policy_and_completion_validation_cover_json_and_character_sse() {
        let schema = json!({"type":"object","properties":{"n":{"const":42}},"required":["n"],"additionalProperties":false});
        let good_call = invoke("lookup", &parameter("n", "42", false));
        let bad_type = invoke("lookup", &parameter("n", "42", true));
        let duplicate = invoke("lookup", &format!("{}\n{}", parameter("n", "42", false), parameter("n", "42", false)));
        let scenarios = vec![
            ("required-valid", json!("required"), false, block(&[good_call.clone()]), true, false),
            ("named-valid", json!({"type":"function","function":{"name":"lookup"}}), false, block(&[good_call.clone()]), true, false),
            ("auto-valid", json!("auto"), true, block(&[good_call.clone()]), true, false),
            ("auto-answer", json!("auto"), false, "Done.".into(), true, false),
            ("required-missing", json!("required"), false, "Done.".into(), false, false),
            ("strict-type", json!("auto"), false, block(&[bad_type]), false, false),
            ("strict-duplicate", json!("auto"), false, block(&[duplicate]), false, false),
            ("unknown-name", json!("auto"), false, block(&[invoke("unknown", &parameter("n","42",false))]), false, false),
            ("parallel-disabled", json!("auto"), false, block(&[good_call.clone(),good_call.clone()]), false, false),
            ("parallel-enabled", json!("auto"), true, block(&[good_call.clone(),good_call.clone()]), true, false),
            ("truncated", json!("required"), false, "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">".into(), true, true),
        ];
        for (name, choice, parallel, text, valid, truncated) in scenarios {
            for thinking in [false, true] {
                for streaming in [false, true] {
                    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(1);
                    let text = if thinking { format!("I will choose carefully.\n</think>{text}") } else { text.clone() };
                    let audit_text = text.clone();
                    let worker = tokio::spawn(async move {
                        let job = receive.recv().await.unwrap();
                        let grammar = job.constraint.unwrap();
                        assert!(grammar.0.contains("ds41_tool_schema"));
                        if let Ok(directory) = std::env::var("CUTEAFD_TOOL_GRAMMAR_AUDIT") {
                            let file = std::path::Path::new(&directory).join(format!("{name}-{thinking}-{streaming}.json"));
                            std::fs::write(file, serde_json::to_vec_pretty(&json!({"name":format!("{name}-{thinking}-{streaming}"),
                                "grammar":serde_json::from_str::<Value>(&grammar.0).unwrap(), "text":audit_text,
                                "expected":valid && !truncated})).unwrap()).unwrap();
                        }
                        if job.events.send(Ok(InferenceChunk::Ready { system_fingerprint:None,
                            prompt_usage:PromptUsage { prompt_tokens:1,prompt_cache_hit_tokens:0 } })).is_err() { return; }
                        for character in text.chars() {
                            if job.events.send(Ok(InferenceChunk::Text { content:character.to_string(),content_tokens:1 })).is_err() { return; }
                        }
                        let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason:if truncated { InferenceFinishReason::Length } else { InferenceFinishReason::Stop } }));
                    });
                    let mut body = json!({"model":MODEL,"messages":[{"role":"user","content":"Use lookup."}],"stream":streaming,
                        "tools":[{"type":"function","function":{"name":"lookup","parameters":schema,"strict":true}}],
                        "tool_choice":choice,"parallel_tool_calls":parallel,"tool_decoding_assistance":false});
                    if !thinking { body["thinking"] = json!({"type":"disabled"}); }
                    let request = Request::post("/v1/chat/completions").header("content-type","application/json").body(Body::from(body.to_string())).unwrap();
                    let response = router(queue).oneshot(request).await.unwrap();
                    if !valid && !streaming { assert_eq!(response.status(),StatusCode::INTERNAL_SERVER_ERROR,"{name}/{thinking}"); }
                    else { assert_eq!(response.status(),StatusCode::OK,"{name}/{thinking}"); }
                    let bytes = to_bytes(response.into_body(),1<<20).await;
                    if !valid && streaming {
                        crate::openai::tests::terminal_sse_error(&bytes.unwrap());
                    }
                    else { assert!(bytes.is_ok(),"{name}/{thinking}: {bytes:?}"); }
                    worker.await.unwrap();
                }
            }
        }
    }
}
