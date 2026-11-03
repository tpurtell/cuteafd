use super::fixtures::{encoding, GOLDENS, TEMPLATE};
use super::*;
use crate::openai::chat::glm5::parser::GlmStop;
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{json, Value};

fn goldens() -> Vec<Value> { serde_json::from_str(GOLDENS).unwrap() }

/// The pinned transformers renderer's output (`muse-template-goldens.py`).
#[test]
fn checkpoint_template_matches_transformers_goldens() {
    let template = ChatTemplate::new(TEMPLATE).unwrap();
    for case in goldens() {
        let rendered = template.render(&case["context"]).unwrap_or_else(|error| panic!("{}: {error:#}", case["name"]));
        assert_eq!(rendered, case["expected"].as_str().unwrap(), "{}", case["name"]);
    }
}

/// The OpenAI request a client would send for a golden context.
fn request_for(context: &Value) -> Value {
    let mut messages = context["messages"].clone();
    for message in messages.as_array_mut().unwrap() {
        for call in message.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
            call["function"]["arguments"] = Value::String(serde_json::to_string(&call["function"]["arguments"]).unwrap());
        }
    }
    let mut body = json!({"model": "meta-models/Muse-Spark-Open-816B-A42B", "messages": messages});
    for key in ["tools", "reasoning_effort"] {
        if let Some(value) = context.get(key) { body[key] = value.clone(); }
    }
    body
}

fn options(body: &Value) -> MusePromptOptions {
    MusePromptOptions { tool_names: body["tools"].as_array().into_iter().flatten()
        .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned()).collect(),
        tool_choice: MuseToolChoice::Auto, response_format: None }
}

/// Requests convert to the golden contexts: rendered at the golden's date,
/// the prompt is byte-identical.
#[test]
fn openai_requests_render_the_golden_prompts() {
    for case in goldens() {
        let body = request_for(&case["context"]);
        let mut context = prompt::template_context(&body, &options(&body), "<|begin_of_text|>").unwrap();
        context["current_date"] = case["context"]["current_date"].clone();
        context["add_generation_prompt"] = case["context"]["add_generation_prompt"].clone();
        let rendered = ChatTemplate::new(TEMPLATE).unwrap().render(&context).unwrap();
        assert_eq!(rendered, case["expected"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[test]
fn efforts_map_onto_the_template_and_thinking_off_refuses() {
    let encoding = encoding();
    let render = |extra: Value| {
        let mut body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        for (key, value) in extra.as_object().unwrap() { body[key] = value.clone(); }
        encoding.render(&body, &options(&body))
    };
    assert!(render(json!({})).unwrap().contains("Reasoning strength: 512."));
    assert!(render(json!({"reasoning_effort": "minimal"})).unwrap().contains("Reasoning strength: 8."));
    assert!(render(json!({"reasoning_effort": "MEDIUM"})).unwrap().contains("Reasoning strength: 32."));
    assert!(render(json!({"reasoning_effort": "max"})).unwrap().contains("Reasoning strength: 512."));
    assert!(render(json!({"reasoning_effort": "bogus"})).unwrap_err().contains("not one of"));
    assert!(render(json!({"reasoning_effort": "none"})).unwrap_err().contains("disabling thinking"));
    assert!(render(json!({"enable_thinking": false})).unwrap_err().contains("disabling thinking"));
    let prompt = render(json!({})).unwrap();
    assert!(prompt.starts_with("<|begin_of_text|><|start|>system<|message|>") && prompt.ends_with("<|start|>assistant"));
}

#[test]
fn tool_choice_joins_the_leading_system_message() {
    let body = json!({"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Go"}],
        "tools": [{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]});
    let options = MusePromptOptions { tool_names: vec!["b".into()], tool_choice: MuseToolChoice::Named("b".into()),
        response_format: Some(json!({"type": "json_object"})) };
    let context = prompt::template_context(&body, &options, "").unwrap();
    assert_eq!(context["messages"].as_array().unwrap().len(), 2);
    assert_eq!(context["messages"][0]["content"],
        "Be terse.\n\nReturn only one valid JSON object with no surrounding prose or markdown.\n\nYou must call the function b.");
    assert_eq!(context["tools"], json!([{"type": "function", "function": {"name": "b"}}]));
}

#[test]
fn stop_tokens_are_the_generation_config_eos() {
    assert_eq!(encoding().stop_token_ids(), vec![200_001, 200_008]);
}

#[test]
fn served_snapshot_matches_the_fixture_when_present() {
    let snapshots = std::path::Path::new(
        "/mnt/sparknest/hf-home/hub/models--meta-models--Muse-Spark-Open-816B-A42B/snapshots");
    let Ok(mut entries) = std::fs::read_dir(snapshots) else { return };
    let Some(Ok(snapshot)) = entries.next() else { return };
    let snapshot = snapshot.path();
    assert_eq!(std::fs::read_to_string(snapshot.join("chat_template.jinja")).unwrap(), TEMPLATE);
    assert_eq!(MuseEncoding::from_snapshot(&snapshot).unwrap().tokens(), encoding().tokens());
}

// ---------------------------------------------------------------- parser

fn tool(name: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition { name: name.into(), description: None, parameters, strict: None }
}

fn tools() -> Vec<ToolDefinition> {
    vec![tool("get_weather", json!({"type": "object", "properties": {"city": {"type": "string"},
        "days": {"type": "integer"}, "detail": {"type": "object"}, "exact": {"type": "boolean"}}})),
        tool("write_file", json!({"type": "object", "properties": {"path": {"type": "string"},
            "text": {"type": "string"}, "tags": {"type": "array"}}}))]
}

#[derive(Debug, Default, PartialEq)]
struct Projection { reasoning: String, content: String, calls: Vec<(String, Value)>, stop: Option<GlmStop>, count: usize }

fn project(chunks: Vec<OutputChunk>, parser: &MuseOutputParser) -> Projection {
    let mut out = Projection::default();
    let mut arguments: Option<(String, String)> = None;
    let flush = |arguments: &mut Option<(String, String)>, out: &mut Projection| {
        if let Some((name, text)) = arguments.take() { out.calls.push((name, serde_json::from_str(&text).unwrap())); }
    };
    for chunk in chunks {
        match chunk {
            OutputChunk::Reasoning { content } => out.reasoning.push_str(&content),
            OutputChunk::Raw { content } => out.content.push_str(&content),
            OutputChunk::ToolCall { tool_name, arguments: first } => {
                flush(&mut arguments, &mut out);
                arguments = Some((tool_name, first));
            }
            OutputChunk::ToolArgumentsDelta { content } => arguments.as_mut().unwrap().1.push_str(&content),
            other => panic!("unexpected {other:?}"),
        }
    }
    flush(&mut arguments, &mut out);
    out.stop = parser.stop().cloned();
    out.count = parser.tool_calls();
    out
}

/// Every split of the text (whole, per char, and in pairs) parses the same.
fn parse(text: &str, tools: Option<Vec<ToolDefinition>>, stops: &[&str]) -> Projection {
    let make = || MuseOutputParser::new(MuseParserOptions { tools: tools.clone(),
        stop_sequences: stops.iter().map(|s| s.to_string()).collect() });
    let mut whole = make();
    let mut chunks = whole.push(text);
    chunks.extend(whole.finish());
    let expected = project(chunks, &whole);
    for size in [1usize, 2, 7] {
        let mut parser = make();
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();
        for piece in chars.chunks(size) { chunks.extend(parser.push(&piece.iter().collect::<String>())); }
        chunks.extend(parser.finish());
        assert_eq!(project(chunks, &parser), expected, "split {size}");
    }
    expected
}

#[test]
fn reasoning_commentary_and_answer_route_by_recipient() {
    let text = " to=self<|message|>think hard<|eom|><|start|>assistant to=commentary<|message|>Working on it.<|eom|>\
        <|start|>assistant to=user<|message|>The answer is 4.<|eot|>";
    let parsed = parse(text, None, &[]);
    assert_eq!(parsed.reasoning, "think hard");
    assert_eq!(parsed.content, "Working on it.\n\nThe answer is 4.");
    assert_eq!(parsed.stop, Some(GlmStop::Marker("<|eot|>".into())));
    // The engine usually stops at <|eot|> itself (a stop token): finish closes the message.
    let parsed = parse(" to=self<|message|>x<|eom|><|start|>assistant to=user<|message|>Done.", None, &[]);
    assert_eq!((parsed.reasoning.as_str(), parsed.content.as_str(), parsed.stop), ("x", "Done.", None));
}

#[test]
fn tool_messages_release_typed_calls() {
    let text = " to=self<|message|>need weather<|eom|><|start|>assistant to=commentary<|message|>Checking.<|eom|>\
        <|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n\
        <atem:parameter name=\"city\">台北</atem:parameter>\n<atem:parameter name=\"days\">3</atem:parameter>\n\
        <atem:parameter name=\"detail\">{\"wind\": true}</atem:parameter>\n<atem:parameter name=\"exact\">false</atem:parameter>\n\
        </atem:invoke>\n</atem:function_calls><|eom|><|start|>assistant to=write_file<|message|><atem:function_calls>\n\
        <atem:invoke name=\"write_file\">\n<atem:parameter name=\"path\">/tmp/a&lt;b&gt;.txt</atem:parameter>\n\
        <atem:parameter name=\"text\">line 1\n  \"2\"</atem:parameter>\n<atem:parameter name=\"tags\">[\"x\"]</atem:parameter>\n\
        </atem:invoke>\n</atem:function_calls>";
    let parsed = parse(text, Some(tools()), &[]);
    assert_eq!(parsed.reasoning, "need weather");
    assert_eq!(parsed.content, "Checking.");
    assert_eq!(parsed.calls, vec![
        ("get_weather".into(), json!({"city": "台北", "days": 3, "detail": {"wind": true}, "exact": false})),
        // Values are the raw text (only attribute names are escaped by the template).
        ("write_file".into(), json!({"path": "/tmp/a&lt;b&gt;.txt", "text": "line 1\n  \"2\"", "tags": ["x"]})),
    ]);
    assert_eq!(parsed.count, 2);
}

#[test]
fn unreadable_tool_messages_return_as_content() {
    for (body, reason) in [
        ("<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Oslo", "unclosed"),
        ("<atem:function_calls>\n<atem:invoke name=\"nope\">\n</atem:invoke>\n</atem:function_calls>", "undeclared"),
        ("<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">a</atem:parameter>\n\
          <atem:parameter name=\"city\">b</atem:parameter>\n</atem:invoke>\n</atem:function_calls>", "duplicate"),
        ("just text", "no block"),
    ] {
        let text = format!(" to=get_weather<|message|>{body}");
        let parsed = parse(&text, Some(tools()), &[]);
        assert_eq!((parsed.count, parsed.calls.len()), (0, 0), "{reason}");
        assert_eq!(parsed.content, body.trim(), "{reason}");
    }
}

#[test]
fn tool_messages_without_tools_and_stop_sequences_end_the_output() {
    let parsed = parse(" to=user<|message|>Hi<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>",
        None, &[]);
    assert_eq!((parsed.content.as_str(), parsed.stop), ("Hi", Some(GlmStop::Marker("to=get_weather".into()))));
    let parsed = parse(" to=self<|message|>STOP here<|eom|><|start|>assistant to=user<|message|>a STOP b", None, &["STOP"]);
    assert_eq!((parsed.reasoning.as_str(), parsed.content.as_str()), ("STOP here", "a"));
    assert_eq!(parsed.stop, Some(GlmStop::Sequence("STOP".into())));
}

#[test]
fn content_after_a_call_is_dropped() {
    let text = " to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n\
        <atem:parameter name=\"city\">Oslo</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>\
        <|start|>assistant to=user<|message|>after<|eot|>";
    let parsed = parse(text, Some(tools()), &[]);
    assert_eq!((parsed.count, parsed.content.as_str()), (1, ""));
}
