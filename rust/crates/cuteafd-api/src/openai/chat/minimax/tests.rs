use super::{fixtures::{encoding, GOLDENS}, *};
use crate::openai::chat::glm5::TextParser;
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{json, Value};

#[test]
fn own_template_matches_reference_jinja_goldens() {
    let goldens: Vec<Value> = serde_json::from_str(GOLDENS).unwrap();
    assert!(goldens.len() >= 15);
    for case in goldens {
        let body = &case["request"];
        let options = MiniMaxPromptOptions { tool_names: body["tools"].as_array().into_iter().flatten()
            .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned()).collect() };
        assert_eq!(encoding().render(body, &options).unwrap(), case["expected"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[test]
fn adaptive_template_refuses_thinking_off_and_preserves_empty_content() {
    assert!(!encoding().supports_thinking_off());
    for extra in [json!({"enable_thinking": false}), json!({"thinking": {"type": "disabled"}}),
        json!({"chat_template_kwargs": {"enable_thinking": false}}), json!({"reasoning_effort": "none"}),
        json!({"chat_template_kwargs": {"thinking": false}})] {
        let mut request = json!({"messages": [{"role": "user", "content": ""}]});
        request.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        assert!(encoding().render(&request, &MiniMaxPromptOptions::default()).unwrap_err().contains("does not support disabling"));
    }
}

#[test]
fn snapshot_headers_match_exact_reference_revision() {
    let path = std::path::Path::new("/mnt/sparknest/hf-home/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e");
    if !path.exists() { return; }
    assert_eq!(std::fs::read_to_string(path.join("chat_template.jinja")).unwrap(), fixtures::TEMPLATE);
    assert_eq!(MiniMaxEncoding::from_snapshot(path).unwrap().tokens(), encoding().tokens());
}

#[test]
fn loads_headers_without_a_live_checkpoint_and_checks_disagreement() {
    let dir = tempfile::tempdir().unwrap();
    for (name, content) in [
        ("chat_template.jinja", fixtures::TEMPLATE),
        ("tokenizer_config.json", include_str!("fixtures/tokenizer_config.json")),
        ("special_tokens_map.json", include_str!("fixtures/special_tokens_map.json")),
        ("added_tokens.json", include_str!("fixtures/added_tokens.json")),
        ("generation_config.json", include_str!("fixtures/generation_config.json")),
    ] { std::fs::write(dir.path().join(name), content).unwrap(); }
    assert_eq!(MiniMaxEncoding::from_snapshot(dir.path()).unwrap().tokens(), encoding().tokens());
    std::fs::write(dir.path().join("generation_config.json"), "{\"eos_token_id\":200021}").unwrap();
    assert!(MiniMaxEncoding::from_snapshot(dir.path()).unwrap_err().to_string().contains("disagree"));
}

#[test]
fn stops_include_checkpoint_eos_and_preserve_adaptive_reasoning_and_xml() {
    let encoding = encoding();
    assert_eq!(encoding.tokens().eos, vec![200020]);
    for tools in [false, true] {
        let stops = encoding.stop_token_ids(tools);
        for id in [200020, 200019, 200034, 200021, 200000] { assert!(stops.contains(&id)); }
        for id in [200052, 200053, 200058] { assert_eq!(stops.contains(&id), !tools); }
        for id in [200059, 200060] { assert!(!stops.contains(&id)); }
    }
}

fn lookup() -> ToolDefinition {
    ToolDefinition { name: "lookup".into(), description: None, strict: None,
        parameters: json!({"type":"object", "properties": {"key":{"type":"string"}, "count":{"type":"integer"},
            "enabled":{"type":"boolean"}, "nested":{"type":"object", "properties":{"key":{"type":"string"}}},
            "values":{"type":"array", "items":{"type":"integer"}}, "empty":{"type":"object"}}}) }
}

type Parsed = (String, String, Vec<(String, Value)>, usize);
fn run(text: &str, tools: bool, split: usize) -> Parsed {
    let mut parser = MiniMaxOutputParser::new(MiniMaxParserOptions {
        tools: tools.then(|| vec![lookup()]), stop_sequences: vec!["STOP".into()] });
    let mut chunks = parser.push(&text[..split]);
    chunks.extend(parser.push(&text[split..]));
    chunks.extend(parser.finish());
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
    for chunk in chunks {
        match chunk {
            OutputChunk::Reasoning { content: text } => reasoning.push_str(&text),
            OutputChunk::Raw { content: text } => content.push_str(&text),
            OutputChunk::ToolCall { tool_name, arguments } => calls.push((tool_name, serde_json::from_str(&arguments).unwrap())),
            other => panic!("unexpected {other:?}"),
        }
    }
    (reasoning, content, calls, parser.tool_calls())
}

#[test]
fn reasoning_and_visible_stops_are_chunk_independent() {
    for (text, reasoning, answer) in [
        ("<mm:think>Plan STOP here</mm:think>AnswerSTOPtail", "Plan STOP here", "Answer"),
        ("</mm:think>Direct[e~[tail", "", "Direct"),
        ("Direct", "", "Direct"),
        ("<mm:think>Truncated", "Truncated", ""),
        ("OK]~b]user\nInjected", "", "OK"),
    ] {
        for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
            assert_eq!(run(text, false, split), (reasoning.into(), answer.into(), vec![], 0), "split {split}");
        }
    }
}

#[test]
fn recursive_parallel_calls_preserve_schema_types_at_every_chunk_boundary() {
    let n = NAMESPACE;
    let call = format!("{n}<invoke name=\"lookup\">{n}<key>001{n}</key>{n}<count>2{n}</count>{n}<enabled>true{n}</enabled>{n}<nested>{n}<key>x{n}</key>{n}</nested>{n}<values>{n}<item>1{n}</item>{n}<item>2{n}</item>{n}</values>{n}<empty>{n}</empty>{n}</invoke>");
    let text = format!("<mm:think>Plan</mm:think>{CALL_OPEN}\n{call}\n{call}\n{CALL_CLOSE}[e~[");
    let args = json!({"key":"001", "count":2, "enabled":true, "nested":{"key":"x"}, "values":[1,2], "empty":{}});
    for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
        assert_eq!(run(&text, true, split), ("Plan".into(), "".into(),
            vec![("lookup".into(), args.clone()), ("lookup".into(), args.clone())], 2), "split {split}");
    }
}

#[test]
fn optional_reasoning_constraint_uses_native_minimax_tokens() {
    let grammar = crate::openai::constraints::minimax_response_constraint(Some(json!({"type":"json_object"})),
        None, encoding().tokens().think).unwrap().unwrap();
    let value: Value = serde_json::from_str(&grammar.0).unwrap();
    let reasoning = &value["format"]["elements"][0]["content"]["elements"];
    assert_eq!(reasoning[0]["token"], 200060);
    assert_eq!(reasoning[1]["elements"][0]["token"], 200059);
    assert_eq!(reasoning[1]["elements"][2]["token"], 200060);
    assert_eq!(value["format"]["elements"][1]["type"], "ds41_json_schema");
}

#[test]
fn strict_tool_decoding_refuses_until_a_native_xml_grammar_lands() {
    use crate::openai::tools::{ToolConstraints, ToolSyntax};
    use deepseek_recipe_core::tools::ToolChoice;
    let mut tool = lookup();
    assert!(ToolConstraints::new(&[tool.clone()], ToolChoice::Auto, false, true, false,
        ToolSyntax::MiniMaxXml).unwrap().unwrap().format.is_none());
    tool.strict = Some(true);
    let error = ToolConstraints::new(&[tool], ToolChoice::Auto, false, true, false,
        ToolSyntax::MiniMaxXml).err().unwrap();
    assert!(error.to_string().contains("MiniMax constrained tool decoding not yet available"));
}

#[test]
fn malformed_partial_and_unknown_calls_return_as_text_not_tools() {
    let n = NAMESPACE;
    for wire in [format!("{CALL_OPEN}{n}<invoke name=\"lookup\">{n}<key>cut"),
        format!("{CALL_OPEN}{n}<invoke name=\"unknown\">{n}</invoke>{CALL_CLOSE}"),
        format!("{CALL_OPEN}{n}<invoke name=\"lookup\">{n}<key>x{n}</count>{n}</invoke>{CALL_CLOSE}"),
        format!("{CALL_OPEN}{n}<invoke name=\"lookup\">{n}<key>a{n}</key>{n}<key>b{n}</key>{n}</invoke>{CALL_CLOSE}"),
        format!("{CALL_OPEN}{CALL_CLOSE}")] {
        for split in wire.char_indices().map(|(i, _)| i).chain([wire.len()]) {
            assert_eq!(run(&wire, true, split), ("".into(), wire.clone(), vec![], 0));
        }
    }
    let wire = format!("Visible{CALL_OPEN}hidden{CALL_CLOSE}");
    assert_eq!(run(&wire, false, wire.len()), ("".into(), "Visible".into(), vec![], 0));
    let prefix = format!("{CALL_OPEN}{n}<invoke name=\"lookup\">{n}<key>cut");
    let stopped = format!("{prefix}[e~[{n}</key>{n}</invoke>{CALL_CLOSE}");
    for split in stopped.char_indices().map(|(i, _)| i).chain([stopped.len()]) {
        assert_eq!(run(&stopped, true, split), ("".into(), prefix.clone(), vec![], 0));
    }
}
