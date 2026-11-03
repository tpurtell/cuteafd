use super::{fixtures::{encoding, GOLDENS}, *};
use crate::openai::chat::glm5::processor::TextParser;
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{json, Value};

fn options(body: &Value) -> MistralPromptOptions {
    MistralPromptOptions { tool_names: body["tools"].as_array().into_iter().flatten()
        .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned()).collect(),
        response_format: body.get("response_format").cloned() }
}

#[test]
fn own_template_matches_transformers_and_mistral_common_goldens() {
    let goldens: Vec<Value> = serde_json::from_str(GOLDENS).unwrap();
    assert!(goldens.len() >= 12 && goldens.iter().filter(|c| c["name"].as_str().unwrap().contains("image")).count() == 3);
    for case in goldens {
        let body = &case["request"];
        assert_eq!(encoding().render(body, &options(body)).unwrap(), case["expected"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[test]
fn served_snapshot_is_recognized_and_matches_the_fixture_when_present() {
    let snapshot = std::path::Path::new("/mnt/sparknest/hf-home/hub/models--mistralai--Mistral-Large-4-1T-A52B/snapshots/f12dabdc0c1e99e7f7924461a6f308859b05af4d");
    if !snapshot.exists() { return; }
    assert!(MistralEncoding::recognizes(snapshot).unwrap());
    assert_eq!(std::fs::read_to_string(snapshot.join("chat_template.jinja")).unwrap(), fixtures::TEMPLATE);
    assert_eq!(MistralEncoding::from_snapshot(snapshot).unwrap().tokens(), encoding().tokens());
}

#[test]
fn recognition_reads_the_checkpoint_not_a_family_switch() {
    let dir = tempfile::tempdir().unwrap();
    let added = |names: &[&str]| json!({"added_tokens": names.iter().enumerate()
        .map(|(id, name)| json!({"id": id, "content": name, "special": true})).collect::<Vec<_>>()});
    std::fs::write(dir.path().join("chat_template.jinja"), fixtures::TEMPLATE).unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), added(&["<s>", "</s>", "[INST]", "[/INST]", "[TOOL_CALLS]",
        "[ARGS]", "[THINK]", "[/THINK]"]).to_string()).unwrap();
    assert!(MistralEncoding::recognizes(dir.path()).unwrap());
    // A GLM-style tokenizer under the same template is not this contract.
    std::fs::write(dir.path().join("tokenizer.json"), added(&["<think>", "</think>", "<tool_call>"]).to_string()).unwrap();
    assert!(!MistralEncoding::recognizes(dir.path()).unwrap());
    // Nor is a Mistral tokenizer whose template never emits the call grammar.
    std::fs::write(dir.path().join("tokenizer.json"), added(&["[INST]", "[/INST]", "[TOOL_CALLS]", "[ARGS]",
        "[THINK]", "[/THINK]"]).to_string()).unwrap();
    std::fs::write(dir.path().join("chat_template.jinja"), "{{ bos_token }}[INST]{{ messages[0].content }}[/INST]").unwrap();
    assert!(!MistralEncoding::recognizes(dir.path()).unwrap());
}

#[test]
fn effort_and_stops_follow_the_checkpoint_contract() {
    let tokens = encoding();
    assert!(!tokens.supports_thinking_off());
    let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
    assert_eq!(tokens.render(&body, &options(&body)).unwrap(),
        "<s>[MODEL_SETTINGS]{\"reasoning_effort\": \"high\"}[/MODEL_SETTINGS][INST]Hi[/INST]");
    let mut low = body.clone();
    low["chat_template_kwargs"] = json!({"reasoning_effort": "LOW"});
    assert!(tokens.render(&low, &options(&low)).unwrap().contains("\"reasoning_effort\": \"low\""));
    for extra in [json!({"enable_thinking": false}), json!({"thinking": {"type": "disabled"}}),
        json!({"reasoning_effort": "none"}), json!({"reasoning_effort": "minimal"}), json!({"reasoning_effort": "max"}),
        json!({"chat_template_kwargs": {"thinking": false}})] {
        let mut request = body.clone();
        request.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        assert!(tokens.render(&request, &options(&request)).is_err(), "{extra}");
    }
    let without = tokens.stop_token_ids(false);
    let with = tokens.stop_token_ids(true);
    for id in [2, 3, 4, 7, 17, 36] { assert!(without.contains(&id) && with.contains(&id), "{id}"); }
    assert!(without.contains(&9) && without.contains(&32));
    assert!(!with.contains(&9) && !with.contains(&32) && !with.contains(&34) && !with.contains(&35));
}

fn weather() -> ToolDefinition {
    ToolDefinition { name: "get_weather".into(), description: None, strict: None,
        parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}) }
}

fn run(text: &str, tools: bool, split: usize) -> (String, String, Vec<(String, String)>, usize) {
    let mut parser = MistralOutputParser::new(MistralParserOptions {
        tools: tools.then(|| vec![weather()]), stop_sequences: Vec::new() });
    let mut chunks = parser.push(&text[..split]);
    chunks.extend(parser.push(&text[split..]));
    chunks.extend(parser.finish());
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
    for chunk in chunks {
        match chunk {
            OutputChunk::Reasoning { content: text } => reasoning.push_str(&text),
            OutputChunk::Raw { content: text } => content.push_str(&text),
            OutputChunk::ToolCall { tool_name, arguments } => calls.push((tool_name, arguments)),
            OutputChunk::ToolArgumentsDelta { content: text } => calls.last_mut().unwrap().1.push_str(&text),
            other => panic!("unexpected {other:?}"),
        }
    }
    (reasoning, content, calls, parser.tool_calls())
}

fn splits(text: &str) -> impl Iterator<Item = usize> + '_ {
    text.char_indices().map(|(index, _)| index).chain([text.len()])
}

#[test]
fn reasoning_content_and_calls_are_independent_of_chunking() {
    let text = "[THINK]Need é weather.[/THINK]Checking.[TOOL_CALLS]get_weather[ARGS]{\"city\": \"台北\", \"n\": [1, {\"a\": \"}\"}]}[TOOL_CALLS]get_weather[ARGS]{}";
    for split in splits(text) {
        let (reasoning, content, calls, count) = run(text, true, split);
        assert_eq!(reasoning, "Need é weather.");
        assert_eq!(content, "Checking.");
        assert_eq!(calls, [("get_weather".into(), "{\"city\": \"台北\", \"n\": [1, {\"a\": \"}\"}]}".into()),
            ("get_weather".into(), "{}".into())], "split {split}");
        assert_eq!(count, 2);
    }
}

#[test]
fn answers_without_thinking_start_as_content() {
    for split in splits("Just 4.") {
        let (reasoning, content, calls, _) = run("Just 4.", true, split);
        assert_eq!((reasoning.as_str(), content.as_str(), calls.len()), ("", "Just 4.", 0));
    }
}

#[test]
fn unreadable_calls_return_as_content_and_never_count() {
    for (text, expected) in [
        ("A [TOOL_CALLS]get_weather{\"city\": 1}", "A [TOOL_CALLS]get_weather{\"city\": 1}"),
        ("A [TOOL_CALLS][ARGS]{}", "A [TOOL_CALLS][ARGS]{}"),
        ("A [TOOL_CALLS]get_weather[ARGS]not json", "A [TOOL_CALLS]get_weather[ARGS]not json"),
        ("A [TOOL_CALLS]get_weather[ARGS]{\"city\": \"x\"", "A [TOOL_CALLS]get_weather[ARGS]{\"city\": \"x\""),
        ("A [TOOL_CALLS]get_weather[ARGS][1]", "A [TOOL_CALLS]get_weather[ARGS][1]"),
    ] {
        for split in splits(text) {
            let (_, content, calls, count) = run(text, true, split);
            assert_eq!((content.as_str(), calls.len(), count), (expected, 0, 0), "{text} @ {split}");
        }
    }
}

#[test]
fn undeclared_tools_and_turn_markers_end_the_output() {
    let mut parser = MistralOutputParser::new(MistralParserOptions { tools: None, stop_sequences: vec!["STOP".into()] });
    let chunks = parser.push("Answer[TOOL_CALLS]get_weather[ARGS]{}");
    assert!(matches!(parser.stop(), Some(crate::openai::chat::glm5::parser::GlmStop::Marker(m)) if m == TOOL_CALLS));
    assert!(matches!(&chunks[..], [OutputChunk::Raw { content }] if content == "Answer"));
    let mut parser = MistralOutputParser::new(MistralParserOptions { tools: Some(vec![weather()]), stop_sequences: vec!["STOP".into()] });
    let chunks = parser.push("[THINK]no STOP here[/THINK]ok STOP more");
    assert!(matches!(parser.stop(), Some(crate::openai::chat::glm5::parser::GlmStop::Sequence(s)) if s == "STOP"));
    assert_eq!(chunks.iter().filter(|c| matches!(c, OutputChunk::Raw { .. })).count(), 1);
}

#[test]
fn text_after_a_parsed_call_is_dropped() {
    let text = "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"x\"} trailing words";
    for split in splits(text) {
        let (_, content, calls, count) = run(text, true, split);
        assert_eq!((content.as_str(), calls.len(), count), ("", 1, 1), "{split}");
    }
}
