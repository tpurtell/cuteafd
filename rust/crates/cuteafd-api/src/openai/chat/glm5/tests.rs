use super::fixtures::{encoding, EXL3_K4, GOLDENS, ZAI};
use super::*;
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{json, Value};

fn goldens() -> Vec<Value> {
    serde_json::from_str(GOLDENS).unwrap()
}

#[test]
fn checkpoint_templates_match_transformers_goldens() {
    for (name, source) in [("exl3_k4", EXL3_K4), ("zai", ZAI)] {
        let template = ChatTemplate::new(source).unwrap();
        for case in goldens() {
            let rendered = template.render(&case["context"])
                .unwrap_or_else(|error| panic!("{name}/{}: {error:#}", case["name"]));
            assert_eq!(rendered, case["expected"][name].as_str().unwrap(), "{name}/{}", case["name"]);
        }
    }
}

#[test]
fn generation_annotations_preserve_checkpoint_template_goldens() {
    for (name, source) in [("exl3_k4", EXL3_K4), ("zai", ZAI)] {
        let annotated = format!("{{% generation %}}{source}{{% endgeneration %}}");
        let template = ChatTemplate::new(annotated).unwrap();
        for case in goldens() {
            assert_eq!(template.render(&case["context"]).unwrap(),
                case["expected"][name].as_str().unwrap(), "{name}/{}", case["name"]);
        }
    }
}

/// The OpenAI request a client would send for a golden context: tool-call
/// arguments as JSON strings, `clear_thinking` under `thinking`.
fn request_for(context: &Value) -> Value {
    let mut messages = context["messages"].clone();
    for message in messages.as_array_mut().unwrap() {
        for call in message.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
            let arguments = serde_json::to_string(&call["function"]["arguments"]).unwrap();
            call["function"]["arguments"] = Value::String(arguments);
        }
    }
    let mut body = json!({"model": "zai-org/GLM-5.3", "messages": messages});
    if let Some(tools) = context.get("tools") { body["tools"] = tools.clone(); }
    if let Some(effort) = context.get("reasoning_effort") { body["reasoning_effort"] = effort.clone(); }
    if let Some(clear) = context.get("clear_thinking") {
        body["thinking"] = json!({"type": "enabled", "clear_thinking": clear});
    }
    body
}

/// Prompt options as the router resolves them for `body`.
fn prompt_options(body: &Value, off: GlmThinkingOff) -> GlmPromptOptions {
    let tool_names = body["tools"].as_array().into_iter().flatten()
        .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned()).collect();
    let thinking = resolve_glm_thinking(body, off).unwrap();
    GlmPromptOptions { thinking: thinking.enabled, reasoning_effort: thinking.effort, tool_names,
        tool_choice: GlmToolChoice::Auto, response_format: None }
}

#[test]
fn openai_requests_render_the_golden_prompts() {
    let encoding = encoding();
    for case in goldens() {
        let body = request_for(&case["context"]);
        assert_eq!(encoding.render(&body, &prompt_options(&body, encoding.thinking_off())).unwrap(),
            case["expected"]["exl3_k4"].as_str().unwrap(), "{}", case["name"]);
    }
}

/// Every way of turning thinking off: `thinking.type`, `enable_thinking`
/// (top level or a template kwarg), the `thinking` kwarg, and the lowest
/// effort's names.
fn thinking_off_forms() -> [Value; 7] {
    [json!({"thinking": {"type": "disabled"}}), json!({"enable_thinking": false}),
        json!({"chat_template_kwargs": {"enable_thinking": false}}), json!({"chat_template_kwargs": {"thinking": false}}),
        json!({"reasoning_effort": "none"}), json!({"reasoning_effort": "minimal"}),
        json!({"chat_template_kwargs": {"reasoning_effort": "minimal"}})]
}

/// With the opt-in `low` setting every off form renders the template's Low
/// effort with the think block open: byte for byte the Transformers golden
/// for `reasoning_effort` "low".
#[test]
fn thinking_off_renders_the_low_effort_golden() {
    assert_eq!(encoding().thinking_off(), GlmThinkingOff::Empty);
    let encoding = encoding().with_thinking_off(GlmThinkingOff::Low);
    let case = goldens().into_iter().find(|case| case["name"] == "low_effort_multi_turn_reasoning").unwrap();
    let mut request = request_for(&case["context"]);
    request.as_object_mut().unwrap().remove("reasoning_effort");
    for off in thinking_off_forms() {
        let mut body = request.clone();
        body.as_object_mut().unwrap().extend(off.as_object().unwrap().clone());
        assert_eq!(encoding.render(&body, &prompt_options(&body, encoding.thinking_off())).unwrap(),
            case["expected"]["exl3_k4"].as_str().unwrap(), "{off}");
    }
}

/// By default (`empty`) every off form, "minimal" included, renders glmrt's
/// form: an empty think block after the default (Max) effort.
#[test]
fn the_empty_setting_closes_an_empty_think_block() {
    let encoding = encoding();
    for off in thinking_off_forms() {
        let mut body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        body.as_object_mut().unwrap().extend(off.as_object().unwrap().clone());
        assert_eq!(encoding.render(&body, &prompt_options(&body, encoding.thinking_off())).unwrap(),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think></think>", "{off}");
    }
}

#[test]
fn stop_tokens_add_turn_markers_and_disabled_tool_markers() {
    let encoding = encoding();
    assert_eq!(encoding.stop_token_ids(true), vec![154_820, 154_825, 154_826, 154_827, 154_828, 154_829, 154_845, 154_846]);
    assert_eq!(encoding.stop_token_ids(false),
        vec![154_820, 154_825, 154_826, 154_827, 154_828, 154_829, 154_843, 154_844, 154_845, 154_846]);
}

#[test]
fn snapshot_loading_reads_template_eos_and_special_tokens() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path();
    std::fs::write(path.join("tokenizer_config.json"), json!({"chat_template": EXL3_K4}).to_string()).unwrap();
    std::fs::write(path.join("generation_config.json"), json!({"eos_token_id": [154_820, 154_827, 154_829]}).to_string()).unwrap();
    let added: Vec<Value> = [(154_825, "<eop>"), (154_826, "<|system|>"), (154_828, "<|assistant|>"),
        (154_842, "</think>"), (154_843, "<tool_call>"), (154_844, "</tool_call>"),
        (154_845, "<tool_response>"), (154_846, "</tool_response>")]
        .into_iter().map(|(id, content)| json!({"id": id, "content": content, "special": true})).collect();
    std::fs::write(path.join("tokenizer.json"), json!({"added_tokens": added}).to_string()).unwrap();
    let loaded = GlmEncoding::from_snapshot(path).unwrap();
    assert_eq!(loaded.tokens(), &GlmTokenIds::glm5());
    std::fs::write(path.join("chat_template.jinja"), "[gMASK]<sop>{{ messages | length }}").unwrap();
    let loaded = GlmEncoding::from_snapshot(path).unwrap();
    let options = GlmPromptOptions { thinking: true, reasoning_effort: None, tool_names: vec![],
        tool_choice: GlmToolChoice::Auto, response_format: None };
    assert_eq!(loaded.render(&json!({"messages": [{"role": "user", "content": "x"}]}), &options).unwrap(), "[gMASK]<sop>1");
}

#[test]
fn served_snapshot_template_matches_the_fixture_when_present() {
    let snapshot = std::path::Path::new("/mnt/sparknest/hf-home/hub/models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-v1/snapshots/47af23347db743b4666d952e2eb48f2b01c3fede");
    if !snapshot.exists() { return; }
    assert_eq!(std::fs::read_to_string(snapshot.join("chat_template.jinja")).unwrap(), EXL3_K4);
    assert_eq!(GlmEncoding::from_snapshot(snapshot).unwrap().tokens(), &GlmTokenIds::glm5());
}

mod vision_template {
    use super::*;
    use std::path::Path;

    const VISION: &str = "{% for m in messages %}{% for p in m.content %}{% if p.type in ['image','image_url'] %}<boi><img><eoi>{% elif p.type == 'text' %}{{ p.text }}{% endif %}{% endfor %}{% endfor %}<think>";
    const TEXT_ONLY: &str = "text-only quant reminder<think>";

    fn snapshot(path: &Path, template: &str, base: Option<&str>) {
        std::fs::create_dir_all(path).unwrap();
        let markers = ["<unk>", "<boi>", "<img>", "<eoi>", "</think>", "<tool_call>", "</tool_call>", "<think>"];
        let vocab: serde_json::Map<String, Value> = markers.iter().enumerate()
            .map(|(id, token)| (token.to_string(), json!(id))).collect();
        let added: Vec<_> = markers.iter().enumerate().map(|(id, content)| json!({"id":id,
            "content":content,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true})).collect();
        let tokenizer = json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":added,
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":vocab,"unk_token":"<unk>"}});
        std::fs::write(path.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        let mut config = json!({"model_type":"glm5_next","vocab_size":8,"image_start_token_id":1,
            "image_token_id":2,"image_end_token_id":3,"eos_token_id":[0]});
        if let Some(base) = base { config["base_model"] = json!(base); }
        std::fs::write(path.join("config.json"), config.to_string()).unwrap();
        std::fs::write(path.join("chat_template.jinja"), template).unwrap();
    }

    fn body() -> Value {
        json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"unused"}},
            {"type":"text","text":"Read it."}]}]})
    }

    fn options() -> GlmPromptOptions {
        GlmPromptOptions { thinking:true, reasoning_effort:None, tool_names:vec![], tool_choice:GlmToolChoice::Auto, response_format:None }
    }

    #[test]
    fn vision_selected_template_preserves_low_effort_and_text_goldens() {
        let temp = tempfile::tempdir().unwrap();
        let served = temp.path().join("served");
        let vendor = temp.path().join("vendor");
        let reminder = "{{- \"<reminder>You are unable to process this \" ~ media_type ~ \" because you don't have multi-modal input ability. Try different methods.</reminder>\" }}";
        assert!(EXL3_K4.contains(reminder));
        let vision_source = EXL3_K4.replace(reminder, "{{- '<boi><img><eoi>' }}");
        snapshot(&served, EXL3_K4, None);
        snapshot(&vendor, &vision_source, None);
        let selected = GlmEncoding::from_snapshot_for_vision(&served, Some(vendor.to_str().unwrap()), None)
            .unwrap().with_thinking_off(GlmThinkingOff::Low);
        assert_eq!(selected.template_provenance().unwrap().selection, "explicit");
        let old = GlmEncoding::from_snapshot(&served).unwrap().with_thinking_off(GlmThinkingOff::Low);
        let low_image = json!({"messages":body()["messages"], "reasoning_effort":"low"});
        let expected_image = selected.render(&low_image, &prompt_options(&low_image, selected.thinking_off())).unwrap();
        assert!(expected_image.contains("Reasoning Effort: Low"));
        assert!(expected_image.contains("<boi><img><eoi>"));
        assert!(!expected_image.contains("unable to process"));
        let case = goldens().into_iter().find(|case| case["name"] == "low_effort_multi_turn_reasoning").unwrap();
        let mut text = request_for(&case["context"]);
        text.as_object_mut().unwrap().remove("reasoning_effort");
        for off in thinking_off_forms() {
            let mut image = body();
            image.as_object_mut().unwrap().extend(off.as_object().unwrap().clone());
            assert_eq!(selected.render(&image, &prompt_options(&image, selected.thinking_off())).unwrap(), expected_image);
            let mut request = text.clone();
            request.as_object_mut().unwrap().extend(off.as_object().unwrap().clone());
            for encoding in [&selected, &old] {
                assert_eq!(encoding.render(&request, &prompt_options(&request, encoding.thinking_off())).unwrap(),
                    case["expected"]["exl3_k4"].as_str().unwrap());
            }
        }
        assert!(old.render(&low_image, &prompt_options(&low_image, old.thinking_off())).unwrap().contains("unable to process"));
    }

    #[test]
    fn compatible_checkpoint_records_its_own_template() {
        let temp = tempfile::tempdir().unwrap();
        snapshot(temp.path(), VISION, None);
        let loaded = GlmEncoding::from_snapshot_for_vision(temp.path(), None, None).unwrap();
        assert_eq!(loaded.render(&body(), &options()).unwrap(), "<boi><img><eoi>Read it.<think>");
        let p = loaded.template_provenance().unwrap();
        assert_eq!(p.selection, "checkpoint");
        assert_eq!(p.template_sha256, format!("{:x}", Sha256::digest(VISION.as_bytes())));
        assert_eq!(p.requested_source, None);
    }

    #[test]
    fn explicit_snapshot_overrides_only_template_and_keeps_served_stops() {
        let temp = tempfile::tempdir().unwrap();
        let served = temp.path().join("served");
        let vendor = temp.path().join("vendor");
        snapshot(&served, TEXT_ONLY, None);
        snapshot(&vendor, VISION, None);
        let mut config = read_json(&vendor.join("config.json")).unwrap();
        config["eos_token_id"] = json!([7]);
        std::fs::write(vendor.join("config.json"), config.to_string()).unwrap();
        let old = GlmEncoding::from_snapshot(&served).unwrap();
        let loaded = GlmEncoding::from_snapshot_for_vision(&served, Some(vendor.to_str().unwrap()), None).unwrap();
        assert_eq!(loaded.tokens(), old.tokens());
        assert_eq!(loaded.stop_token_ids(false), old.stop_token_ids(false));
        assert_eq!(loaded.render(&body(), &options()).unwrap(), "<boi><img><eoi>Read it.<think>");
        let p = loaded.template_provenance().unwrap();
        assert_eq!(p.selection, "explicit");
        assert_eq!(p.requested_source.as_deref(), vendor.to_str());
        // Off uses the existing constructor even when an override was configured.
        assert_eq!(old.render(&body(), &options()).unwrap(), TEXT_ONLY);
    }

    #[test]
    fn config_named_hf_base_resolves_cached_ref_without_inference() {
        let temp = tempfile::tempdir().unwrap();
        let served = temp.path().join("served");
        let hf = temp.path().join("hf");
        let root = hf.join("hub/models--vendor--flash");
        let vendor = root.join("snapshots/revision-a");
        snapshot(&served, TEXT_ONLY, Some("vendor/flash"));
        snapshot(&vendor, VISION, None);
        std::fs::create_dir_all(root.join("refs")).unwrap();
        std::fs::write(root.join("refs/main"), "revision-a\n").unwrap();
        let loaded = GlmEncoding::from_snapshot_for_vision(&served, None, Some(&hf)).unwrap();
        let p = loaded.template_provenance().unwrap();
        assert_eq!(p.selection, "config_base_model");
        assert_eq!(p.requested_source.as_deref(), Some("vendor/flash"));
        assert_eq!(p.snapshot, std::fs::canonicalize(vendor).unwrap());
        let explicit = GlmEncoding::from_snapshot_for_vision(&served, Some("vendor/flash"), Some(&hf)).unwrap();
        assert_eq!(explicit.template_provenance().unwrap().selection, "explicit");
    }

    #[test]
    fn missing_metadata_refuses_vision_but_text_only_is_unchanged() {
        let temp = tempfile::tempdir().unwrap();
        snapshot(temp.path(), TEXT_ONLY, None);
        let error = GlmEncoding::from_snapshot_for_vision(temp.path(), None, None).unwrap_err().to_string();
        assert!(error.contains("VISION enabled") && error.contains("CHAT_TEMPLATE_FROM") && error.contains("no base_model"), "{error}");
        assert_eq!(GlmEncoding::from_snapshot(temp.path()).unwrap().render(&body(), &options()).unwrap(), TEXT_ONLY);
    }

    #[test]
    fn template_gap_is_a_text_only_template_without_a_declared_base() {
        let temp = tempfile::tempdir().unwrap();
        snapshot(temp.path(), TEXT_ONLY, None);
        let gap = GlmEncoding::vision_template_gap(temp.path()).unwrap().unwrap();
        // The refusal names the same gap.
        let refusal = GlmEncoding::from_snapshot_for_vision(temp.path(), None, None).unwrap_err().to_string();
        assert_eq!(refusal, format!("VISION enabled but {gap}; set CHAT_TEMPLATE_FROM=<HF id or snapshot> to the vendor template, or VISION=off"));
        assert!(gap.contains("does not render image markers") && gap.contains("no base_model")
            && gap.contains(&temp.path().display().to_string()), "{gap}");
        // An unused image macro or a refusing template is a gap; a declared base (even one that is
        // not downloaded) or a template that marks images is not.
        snapshot(temp.path(), "{% macro image() %}<boi><img><eoi>{% endmacro %}no images<think>", None);
        assert!(GlmEncoding::vision_template_gap(temp.path()).unwrap().is_some());
        snapshot(temp.path(), "{{ raise_exception('text only') }}", None);
        assert!(GlmEncoding::vision_template_gap(temp.path()).unwrap().is_some());
        // The EXL3 quant's own template answers an image with its text-only reminder.
        snapshot(temp.path(), EXL3_K4, None);
        assert!(GlmEncoding::vision_template_gap(temp.path()).unwrap().is_some());
        snapshot(temp.path(), TEXT_ONLY, Some("missing/base"));
        assert_eq!(GlmEncoding::vision_template_gap(temp.path()).unwrap(), None);
        snapshot(temp.path(), VISION, None);
        assert_eq!(GlmEncoding::vision_template_gap(temp.path()).unwrap(), None);
        std::fs::remove_file(temp.path().join("tokenizer.json")).unwrap();
        assert!(GlmEncoding::vision_template_gap(temp.path()).is_err());
    }

    #[test]
    fn invalid_missing_or_text_only_override_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        let served = temp.path().join("served");
        let vendor = temp.path().join("vendor");
        snapshot(&served, TEXT_ONLY, None);
        snapshot(&vendor, TEXT_ONLY, None);
        for source in ["../vendor/flash", "vendor/../../flash", "missing/flash", vendor.to_str().unwrap()] {
            assert!(GlmEncoding::from_snapshot_for_vision(&served, Some(source), Some(temp.path())).is_err(), "{source}");
        }
        std::fs::write(vendor.join("chat_template.jinja"), "{{ broken").unwrap();
        assert!(GlmEncoding::from_snapshot_for_vision(&served, Some(vendor.to_str().unwrap()), None).is_err());
        std::fs::remove_file(vendor.join("chat_template.jinja")).unwrap();
        assert!(GlmEncoding::from_snapshot_for_vision(&served, Some(vendor.to_str().unwrap()), None).is_err());
    }

    #[test]
    fn vocabulary_and_configured_image_marker_mismatch_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        let served = temp.path().join("served");
        let vendor = temp.path().join("vendor");
        snapshot(&served, TEXT_ONLY, None);
        snapshot(&vendor, VISION, None);
        let mut config = read_json(&vendor.join("config.json")).unwrap();
        config["image_token_id"] = json!(1);
        std::fs::write(vendor.join("config.json"), config.to_string()).unwrap();
        assert!(GlmEncoding::from_snapshot_for_vision(&served, Some(vendor.to_str().unwrap()), None).is_err());
        snapshot(&vendor, VISION, None);
        let mut tokenizer = read_json(&vendor.join("tokenizer.json")).unwrap();
        tokenizer["model"]["vocab"]["<unk>"] = json!(7);
        std::fs::write(vendor.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        let error = GlmEncoding::from_snapshot_for_vision(&served, Some(vendor.to_str().unwrap()), None).unwrap_err().to_string();
        assert!(error.contains("tokenizer differs"), "{error}");
    }

    #[test]
    fn render_refusal_can_use_a_declared_vendor_template() {
        let temp = tempfile::tempdir().unwrap();
        let served = temp.path().join("served");
        let vendor = temp.path().join("vendor");
        snapshot(&vendor, VISION, None);
        snapshot(&served, "{{ raise_exception('text only') }}", vendor.to_str());
        assert!(GlmEncoding::from_snapshot(&served).unwrap().render(&body(), &options()).is_err());
        let loaded = GlmEncoding::from_snapshot_for_vision(&served, None, None).unwrap();
        assert_eq!(loaded.template_provenance().unwrap().selection, "config_base_model");
        assert_eq!(loaded.render(&body(), &options()).unwrap(), "<boi><img><eoi>Read it.<think>");
    }

    #[test]
    fn unused_image_macro_does_not_qualify_a_template() {
        let temp = tempfile::tempdir().unwrap();
        snapshot(temp.path(), "{% macro image() %}<boi><img><eoi>{% endmacro %}no images<think>", None);
        assert!(GlmEncoding::from_snapshot_for_vision(temp.path(), None, None).is_err());
    }

    #[test]
    fn explicit_override_wins_over_incompatible_config_base() {
        let temp = tempfile::tempdir().unwrap();
        let served = temp.path().join("served");
        let vendor = temp.path().join("vendor");
        snapshot(&served, VISION, Some("missing/base"));
        snapshot(&vendor, VISION, None);
        let encoding = GlmEncoding::from_snapshot_for_vision(&served, Some(vendor.to_str().unwrap()), None).unwrap();
        assert_eq!(encoding.template_provenance().unwrap().selection, "explicit");
        let mut config = read_json(&vendor.join("config.json")).unwrap();
        config["model_type"] = json!("other_family");
        std::fs::write(vendor.join("config.json"), config.to_string()).unwrap();
        assert!(GlmEncoding::from_snapshot_for_vision(&served, Some(vendor.to_str().unwrap()), None).is_err());
    }

    #[tokio::test]
    async fn selected_template_identity_is_public_model_provenance() {
        use crate::openai::{router_for_model, ConsoleHub, ModelEncoding, ModelProfile, NativeLimits};
        use axum::{body::{to_bytes, Body}, http::Request};
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;
        let temp = tempfile::tempdir().unwrap();
        snapshot(temp.path(), VISION, None);
        let encoding = GlmEncoding::from_snapshot_for_vision(temp.path(), None, None).unwrap();
        let expected = serde_json::to_value(encoding.template_provenance().unwrap()).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let app = router_for_model(tx, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
            std::time::Duration::from_secs(5), ConsoleHub::disabled(),
            ModelProfile::new("test/glm", ModelEncoding::Glm(Arc::new(encoding))));
        let response = app.oneshot(Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
        let body: Value = serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap();
        assert_eq!(body["data"][0]["chat_template"], expected);
        assert_eq!(body["data"][0]["capabilities"]["vision"], false);
    }

    #[test]
    fn installed_official_and_tr3_templates_follow_vision_policy_when_present() {
        let root = Path::new("/mnt/sparknest/hf-home/hub");
        let official = root.join("models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a");
        let tr3 = root.join("models--brandonmusic--GLM-5.3-Flash-tr3-4bpw/snapshots/a5fee929cf4888b1824323e33e8a19b60129e025");
        if !official.is_dir() || !tr3.is_dir() { return; }
        assert_eq!(GlmEncoding::from_snapshot_for_vision(&official, None, None).unwrap()
            .template_provenance().unwrap().selection, "checkpoint");
        let error = GlmEncoding::from_snapshot_for_vision(&tr3, None, None).unwrap_err().to_string();
        assert!(error.contains("CHAT_TEMPLATE_FROM"), "{error}");
        assert!(GlmEncoding::vision_template_gap(&tr3).unwrap().is_some());
        assert_eq!(GlmEncoding::vision_template_gap(&official).unwrap(), None);
        let old = GlmEncoding::from_snapshot(&tr3).unwrap();
        let fallback = GlmEncoding::from_snapshot_for_vision(&tr3, Some(official.to_str().unwrap()), None).unwrap();
        assert_eq!(old.tokens(), fallback.tokens());
        assert!(fallback.render(&body(), &options()).unwrap().contains("<|begin_of_image|><|image|><|end_of_image|>"));
        assert!(old.render(&body(), &options()).unwrap().contains("unable to process"));
    }
}

// ---------------------------------------------------------------- parser

fn tool(name: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition { name: name.into(), description: None, parameters, strict: None }
}

fn tools() -> Vec<ToolDefinition> {
    vec![
        tool("lookup", json!({"type": "object", "properties": {
            "city": {"type": "string"}, "units": {"type": "string", "enum": ["metric", "imperial"]}}})),
        tool("search", json!({"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}, "exact": {"type": "boolean"},
            "filters": {"type": "object"}, "tags": {"anyOf": [{"type": "null"}, {"type": "array"}]}}})),
        tool("write_file", json!({"type": "object", "properties": {
            "path": {"type": "string"}, "text": {"type": ["string", "null"]}}})),
    ]
}

#[derive(Debug, Default, PartialEq)]
struct Projection {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
    stop: Option<GlmStop>,
}

fn project(chunks: &[OutputChunk], stop: Option<GlmStop>) -> Projection {
    let mut projection = Projection { stop, ..Default::default() };
    let mut arguments = Vec::<String>::new();
    for chunk in chunks {
        match chunk {
            OutputChunk::Reasoning { content } => projection.reasoning.push_str(content),
            OutputChunk::Raw { content } => projection.content.push_str(content),
            OutputChunk::ToolCall { tool_name, arguments: first } => {
                projection.calls.push((tool_name.clone(), Value::Null));
                arguments.push(first.clone());
            }
            OutputChunk::ToolArgumentsDelta { content } => arguments.last_mut().expect("delta follows a call").push_str(content),
            other => panic!("unexpected chunk {other:?}"),
        }
    }
    for (call, text) in projection.calls.iter_mut().zip(arguments) {
        call.1 = serde_json::from_str(&text).unwrap_or_else(|error| panic!("arguments {text:?}: {error}"));
    }
    projection
}

fn parse(options: &GlmParserOptions, pieces: &[&str]) -> Projection {
    let mut parser = GlmOutputParser::new(options.clone());
    let mut chunks = Vec::new();
    for piece in pieces { chunks.extend(parser.push(piece)); }
    chunks.extend(parser.finish());
    project(&chunks, parser.stop().cloned())
}

/// Parse whole, at every two-way split, and one character at a time; all
/// must agree.
fn parse_everywhere(options: &GlmParserOptions, text: &str) -> Projection {
    let whole = parse(options, &[text]);
    for (index, _) in text.char_indices() {
        assert_eq!(parse(options, &[&text[..index], &text[index..]]), whole, "split at {index}");
    }
    let characters: Vec<String> = text.chars().map(String::from).collect();
    let pieces: Vec<&str> = characters.iter().map(String::as_str).collect();
    assert_eq!(parse(options, &pieces), whole, "character stream");
    whole
}

fn options(thinking: bool, tools: Option<Vec<ToolDefinition>>) -> GlmParserOptions {
    GlmParserOptions { thinking, tools, ..Default::default() }
}

/// Two calls after reasoning, from glmrt's fixtures.
const GLMRT_CALLS: &str = concat!("I need the weather and a search.</think>\n",
    "<tool_call>lookup<arg_key>city</arg_key><arg_value>Taipei</arg_value><arg_key>units</arg_key><arg_value>metric</arg_value></tool_call>",
    "\n<tool_call>search<arg_key>query</arg_key><arg_value>RDMA ring buffers</arg_value><arg_key>limit</arg_key><arg_value>4</arg_value></tool_call>");

/// Typed values, a call without arguments, escapes and an undeclared tool.
const TYPED_CALLS: &str = concat!("Checking.<tool_call>search<arg_key>query</arg_key><arg_value>42</arg_value>",
    "<arg_key>exact</arg_key><arg_value>true</arg_value><arg_key>filters</arg_key><arg_value>{\"lang\": \"zh\", \"n\": [1, 2.5]}</arg_value>",
    "<arg_key>tags</arg_key><arg_value>null</arg_value></tool_call>",
    "<tool_call>refresh</tool_call>",
    "<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/a \"b\".txt</arg_value>",
    "<arg_key>text</arg_key><arg_value>line 1\n\tline 2 \\ 台北 💡\n</arg_value></tool_call>",
    "<tool_call>unknown<arg_key>n</arg_key><arg_value> 7 </arg_value><arg_key>s</arg_key><arg_value>plain</arg_value></tool_call>",
    "trailing text is dropped");

#[test]
fn reasoning_then_answer() {
    let projection = parse_everywhere(&options(true, None),
        "The user asks for 2 + 2, which is 4.</think>\n\n2 + 2 = **4**.\n");
    assert_eq!(projection, Projection { reasoning: "The user asks for 2 + 2, which is 4.".into(),
        content: "2 + 2 = **4**.".into(), ..Default::default() });
}

#[test]
fn non_thinking_answer_and_stray_think_markers() {
    let projection = parse_everywhere(&options(false, None), "Hello <think>aside</think>world");
    assert_eq!(projection.content, "Hello world");
    assert_eq!(projection.reasoning, "aside");
}

#[test]
fn glmrt_fixture_calls_parse_with_schema_types() {
    let projection = parse_everywhere(&options(true, Some(tools())), GLMRT_CALLS);
    assert_eq!(projection, Projection { reasoning: "I need the weather and a search.".into(), content: String::new(),
        calls: vec![("lookup".into(), json!({"city": "Taipei", "units": "metric"})),
            ("search".into(), json!({"query": "RDMA ring buffers", "limit": 4}))], stop: None });
}

#[test]
fn typed_values_zero_argument_calls_and_template_json() {
    // Values rendered by the template's `tojson` (spaced separators) and
    // multi-line strings round-trip; an undeclared tool keeps JSON typing.
    let projection = parse_everywhere(&options(false, Some(tools())), TYPED_CALLS);
    assert_eq!(projection.content, "Checking.");
    assert_eq!(projection.calls, vec![
        ("search".into(), json!({"query": "42", "exact": true, "filters": {"lang": "zh", "n": [1, 2.5]}, "tags": null})),
        ("refresh".into(), json!({})),
        ("write_file".into(), json!({"path": "/tmp/a \"b\".txt", "text": "line 1\n\tline 2 \\ 台北 💡\n"})),
        ("unknown".into(), json!({"n": 7, "s": "plain"})),
    ]);
}

/// Each call's name and argument text, as a client joins its deltas.
fn call_texts(options: &GlmParserOptions, pieces: &[&str]) -> Vec<(String, String)> {
    let mut parser = GlmOutputParser::new(options.clone());
    let mut chunks = Vec::new();
    for piece in pieces { chunks.extend(parser.push(piece)); }
    chunks.extend(parser.finish());
    let mut calls: Vec<(String, String)> = Vec::new();
    for chunk in chunks {
        match chunk {
            OutputChunk::ToolCall { tool_name, arguments } => calls.push((tool_name, arguments)),
            OutputChunk::ToolArgumentsDelta { content } => calls.last_mut().expect("delta follows a call").1.push_str(&content),
            _ => {}
        }
    }
    calls
}

/// Valid calls keep the argument bytes the streaming parser sent before calls
/// were held (recorded from it), whole or a character at a time.
#[test]
fn valid_calls_keep_their_argument_bytes() {
    for (thinking, text, expected) in [
        (true, GLMRT_CALLS, vec![("lookup", r#"{"city":"Taipei","units":"metric"}"#),
            ("search", r#"{"query":"RDMA ring buffers","limit":4}"#)]),
        (false, TYPED_CALLS, vec![("search", r#"{"query":"42","exact":true,"filters":{"lang":"zh","n":[1,2.5]},"tags":null}"#),
            ("refresh", "{}"), ("write_file", r#"{"path":"/tmp/a \"b\".txt","text":"line 1\n\tline 2 \\ 台北 💡\n"}"#),
            ("unknown", r#"{"n":7,"s":"plain"}"#)]),
    ] {
        let expected: Vec<(String, String)> = expected.into_iter().map(|(name, arguments)| (name.into(), arguments.into())).collect();
        let parser = options(thinking, Some(tools()));
        let characters: Vec<String> = text.chars().map(String::from).collect();
        assert_eq!(call_texts(&parser, &[text]), expected);
        assert_eq!(call_texts(&parser, &characters.iter().map(String::as_str).collect::<Vec<_>>()), expected);
    }
}

#[test]
fn tool_calls_are_held_until_they_close() {
    // Nothing of a call goes out before its closing tag; then its chunks are
    // the ones a string argument always streamed as.
    let mut parser = GlmOutputParser::new(options(false, Some(tools())));
    assert_eq!(parser.push("<tool_call>write_file<arg_key>text</arg_key><arg_value>first \"quoted\" line</arg_"), vec![]);
    assert_eq!(parser.push("value>"), vec![]);
    assert_eq!(parser.tool_calls(), 0);
    let delta = |content: &str| OutputChunk::ToolArgumentsDelta { content: content.into() };
    assert_eq!(parser.push("</tool_call>"), vec![
        OutputChunk::ToolCall { tool_name: "write_file".into(), arguments: "{".into() },
        delta(r#""text":""#), delta(r#"first \"quoted\" line"#), delta("\""), delta("}")]);
    assert_eq!(parser.tool_calls(), 1);
}

#[test]
fn partial_markers_are_withheld_until_resolved() {
    let mut parser = GlmOutputParser::new(options(false, Some(tools())));
    let chunks = parser.push("visible <tool_");
    assert_eq!(chunks, vec![OutputChunk::Raw { content: "visible".into() }]);
    let chunks = parser.push("x> literal");
    assert_eq!(chunks, vec![OutputChunk::Raw { content: " <tool_x> literal".into() }]);
}

/// One fixture per shape of unreadable call: its text, from the opening tag,
/// comes back as content (with the whitespace before it), beside any call
/// that parsed. Text after the first parsed call stays dropped.
#[test]
fn unreadable_calls_return_as_content() {
    let oslo = "<tool_call>lookup<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>";
    let parsed = || vec![("lookup".to_owned(), json!({"city": "Oslo"}))];
    for (shape, text, content, calls) in [
        // Closing tag missing: the output ends inside a value (max_tokens) ...
        ("truncated value", "Writing.\n<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/x</arg_value><arg_key>text</arg_key><arg_value>partial".to_owned(),
            "Writing.\n<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/x</arg_value><arg_key>text</arg_key><arg_value>partial".to_owned(), vec![]),
        // ... or inside the name ...
        ("truncated name", "Sure.<tool_call>look".to_owned(), "Sure.<tool_call>look".to_owned(), vec![]),
        // ... or a new call opens first, which still parses.
        ("next call", format!("<tool_call>lookup<arg_key>city</arg_key><arg_value>Rome</arg_value>\n{oslo}"),
            "<tool_call>lookup<arg_key>city</arg_key><arg_value>Rome</arg_value>".to_owned(), parsed()),
        // Markup in the name: either angle bracket.
        ("markup in the name", "a <tool_call>look<up</tool_call> b".to_owned(), "a <tool_call>look<up</tool_call> b".to_owned(), vec![]),
        ("markup in the name, >", "<tool_call>lookup><arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>".to_owned(),
            "<tool_call>lookup><arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>".to_owned(), vec![]),
        // Arguments without a name.
        ("no name", "<tool_call><arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>".to_owned(),
            "<tool_call><arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>".to_owned(), vec![]),
        // An empty call.
        ("empty", "<tool_call>\n</tool_call>".to_owned(), "<tool_call>\n</tool_call>".to_owned(), vec![]),
        // After a parsed call a lost call is still returned; the text around it is not.
        ("after a call", format!("A{oslo}B<tool_call><x></tool_call>C"), "A<tool_call><x></tool_call>".to_owned(), parsed()),
    ] {
        let projection = parse_everywhere(&options(false, Some(tools())), &text);
        assert_eq!((projection.content, projection.calls, projection.stop), (content, calls, None), "{shape}");
    }
}

/// A name closed by stray closing tags is recovered when what is left is a
/// declared tool; otherwise the call is returned as content.
#[test]
fn a_stray_closing_tag_after_a_declared_name_is_recovered() {
    let text = "<tool_call>lookup</arg_key>\n<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>";
    let projection = parse_everywhere(&options(false, Some(tools())), text);
    assert_eq!((projection.content.as_str(), projection.calls), ("", vec![("lookup".into(), json!({"city": "Oslo"}))]));
    let undeclared = text.replace("lookup", "locate");
    let projection = parse_everywhere(&options(false, Some(tools())), &undeclared);
    assert_eq!((projection.content.as_str(), projection.calls), (undeclared.as_str(), vec![]));
}

/// A repeated key keeps its last value, in the place of its first, so the
/// arguments stay a JSON object without duplicate keys.
#[test]
fn a_repeated_argument_keeps_its_last_value() {
    let text = concat!("<tool_call>search<arg_key>query</arg_key><arg_value>first</arg_value>",
        "<arg_key>limit</arg_key><arg_value>1</arg_value><arg_key>query</arg_key><arg_value>second</arg_value>",
        "<arg_key>limit</arg_key><arg_value>2</arg_value></tool_call>");
    let expected = vec![("search".to_owned(), r#"{"query":"second","limit":2}"#.to_owned())];
    let parser = options(false, Some(tools()));
    let characters: Vec<String> = text.chars().map(String::from).collect();
    assert_eq!(call_texts(&parser, &[text]), expected);
    assert_eq!(call_texts(&parser, &characters.iter().map(String::as_str).collect::<Vec<_>>()), expected);
}

/// Stray text between arguments is dropped; the call and its other arguments
/// survive.
#[test]
fn stray_text_is_dropped_from_its_call() {
    let limit = "<arg_key>limit</arg_key><arg_value>4</arg_value>";
    let text = format!("<tool_call>search{limit} stray </tool_call>");
    let projection = parse_everywhere(&options(false, Some(tools())), &text);
    assert_eq!((projection.content.as_str(), projection.calls), ("", vec![("search".into(), json!({"limit": 4}))]));
}

/// An argument that cannot be read — a markup or empty key, or a key with no
/// `<arg_value>` — loses its whole call: the call comes back as content,
/// nothing of it is a tool call, and it is not counted. The call would
/// otherwise be released with that argument silently missing, which is not
/// the call the model meant.
#[test]
fn an_unreadable_argument_returns_its_whole_call_as_content() {
    let limit = "<arg_key>limit</arg_key><arg_value>4</arg_value>";
    for (shape, text) in [
        ("no value", format!("<tool_call>search<arg_key>query</arg_key>no value{limit}</tool_call>")),
        ("markup in the key", format!("<tool_call>search<arg_key><bad></arg_key><arg_value>1</arg_value>{limit}</tool_call>")),
        ("markup in the key, >", format!("<tool_call>search<arg_key>a>b</arg_key><arg_value>1</arg_value>{limit}</tool_call>")),
        ("empty key", format!("<tool_call>search<arg_key></arg_key><arg_value>1</arg_value>{limit}</tool_call>")),
        ("the first one", format!(concat!("<tool_call>search<arg_key>query</arg_key>no value",
            "<arg_key><bad></arg_key><arg_value>1</arg_value>{} stray </tool_call>"), limit)),
    ] {
        let projection = parse_everywhere(&options(false, Some(tools())), &text);
        assert_eq!((projection.content.as_str(), projection.calls), (text.as_str(), vec![]), "{shape}");
    }
}

#[test]
fn turn_markers_and_client_stop_sequences_end_output() {
    let projection = parse_everywhere(&options(true, Some(tools())),
        "Call it.</think><tool_call>lookup<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call><|observation|>ignored");
    assert_eq!(projection.calls, vec![("lookup".into(), json!({"city": "Oslo"}))]);
    assert_eq!(projection.stop, Some(GlmStop::Marker("<|observation|>".into())));

    let projection = parse_everywhere(&options(false, None), "Answer.<|user|>next turn");
    assert_eq!((projection.content.as_str(), projection.stop), ("Answer.", Some(GlmStop::Marker("<|user|>".into()))));

    // Without declared tools a call marker ends the turn (glmrt's stop set).
    let projection = parse_everywhere(&options(true, None), "Maybe a tool.</think>I can't.<tool_call>lookup</tool_call>");
    assert_eq!((projection.content.as_str(), projection.stop), ("I can't.", Some(GlmStop::Marker("<tool_call>".into()))));

    // Client stop sequences apply to content, not reasoning or arguments.
    let mut stop = options(true, Some(tools()));
    stop.stop_sequences = vec!["END".into(), "Oslo".into()];
    let projection = parse_everywhere(&stop, "END of thought.</think>Visible END hidden");
    assert_eq!((projection.reasoning.as_str(), projection.content.as_str()), ("END of thought.", "Visible"));
    assert_eq!(projection.stop, Some(GlmStop::Sequence("END".into())));
    let projection = parse_everywhere(&stop, "</think><tool_call>lookup<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>");
    assert_eq!((projection.calls.len(), projection.stop), (1, None));
}

// ---------------------------------------------------------------- router

mod router {
    use super::*;
    use crate::openai::{router_for_model, ConsoleHub, InferenceChunk, InferenceFinishReason, ModelEncoding,
        ModelProfile, NativeLimits, NativeRequest, PromptUsage};
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc;
    use tower::ServiceExt;

    const MODEL: &str = "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1";

    fn app(queue: mpsc::Sender<NativeRequest>, encoding: GlmEncoding) -> axum::Router {
        let profile = ModelProfile::new(MODEL, ModelEncoding::Glm(Arc::new(encoding)));
        router_for_model(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
            std::time::Duration::from_secs(5), ConsoleHub::disabled(), profile)
    }

    /// Serve one request whose worker streams `text` a character at a time.
    async fn serve(body: Value, text: impl Into<String>, check: impl FnOnce(&NativeRequest) + Send + 'static)
        -> (StatusCode, Vec<u8>) {
        serve_with(encoding(), body, text, check).await
    }

    /// [`serve`] for a server with `encoding`.
    async fn serve_with(encoding: GlmEncoding, body: Value, text: impl Into<String>,
        check: impl FnOnce(&NativeRequest) + Send + 'static) -> (StatusCode, Vec<u8>) {
        let text = text.into();
        let (queue, mut receive) = mpsc::channel::<NativeRequest>(1);
        let worker = tokio::spawn(async move {
            let job = receive.recv().await.unwrap();
            check(&job);
            let _ = job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 9, prompt_cache_hit_tokens: 0 } }));
            for character in text.chars() {
                if job.events.send(Ok(InferenceChunk::Text { content: character.to_string(), content_tokens: 1 })).is_err() {
                    return;
                }
            }
            let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }));
        });
        let request = Request::post("/v1/chat/completions").header("content-type", "application/json")
            .body(Body::from(body.to_string())).unwrap();
        let response = app(queue, encoding).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap().to_vec();
        worker.await.unwrap();
        (status, bytes)
    }

    fn sse_events(bytes: &[u8]) -> Vec<Value> {
        let text = std::str::from_utf8(bytes).unwrap();
        assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
        text.split("\n\n").filter_map(|event| event.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]").map(|data| serde_json::from_str(data).unwrap()).collect()
    }

    #[tokio::test]
    async fn answer_with_reasoning_in_both_response_modes() {
        for streaming in [false, true] {
            let body = json!({"model": MODEL, "messages": [{"role": "user", "content": "2+2?"}], "stream": streaming});
            let (status, bytes) = serve(body, "Add them.</think>\n4", |job| {
                assert_eq!(job.prompt, "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>2+2?<|assistant|><think>");
                assert!(job.constraint.is_none());
                assert!(job.stop_token_ids.contains(&154_843) && job.stop_token_ids.contains(&154_829));
            }).await;
            assert_eq!(status, StatusCode::OK);
            if streaming {
                let events = sse_events(&bytes);
                let field = |name: &str| events.iter()
                    .filter_map(|event| event["choices"][0]["delta"][name].as_str()).collect::<String>();
                assert_eq!((field("reasoning_content").as_str(), field("content").as_str()), ("Add them.", "4"));
                assert_eq!(events.last().unwrap()["choices"][0]["finish_reason"], "stop");
            } else {
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                let message = &value["choices"][0]["message"];
                assert_eq!((message["reasoning_content"].as_str(), message["content"].as_str()), (Some("Add them."), Some("4")));
                assert_eq!(value["choices"][0]["finish_reason"], "stop");
                assert_eq!(value["usage"]["completion_tokens"], 19);
            }
        }
    }

    #[tokio::test]
    async fn tool_calls_stream_and_aggregate_with_tool_calls_finish() {
        let tools = json!([{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]);
        for streaming in [false, true] {
            let body = json!({"model": MODEL, "stream": streaming, "tools": tools,
                "messages": [{"role": "user", "content": "Weather in Taipei for 2 days?"}]});
            let (status, bytes) = serve(body,
                "Use lookup.</think>\n<tool_call>lookup<arg_key>city</arg_key><arg_value>Taipei</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>",
                |job| {
                    assert!(job.prompt.contains("<tools>\n{\"name\": \"lookup\", \"parameters\": {\"type\": \"object\""));
                    // Auto choice without strict tools is unconstrained (glmrt default).
                    assert!(job.constraint.is_none());
                    assert!(!job.stop_token_ids.contains(&154_843));
                }).await;
            assert_eq!(status, StatusCode::OK);
            let (name, arguments, finish) = if streaming {
                let events = sse_events(&bytes);
                let deltas: Vec<&Value> = events.iter()
                    .flat_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array().into_iter().flatten()).collect();
                assert!(deltas.len() > 2, "arguments stream incrementally");
                let name = deltas[0]["function"]["name"].as_str().unwrap().to_owned();
                let arguments: String = deltas.iter().filter_map(|d| d["function"]["arguments"].as_str()).collect();
                (name, arguments, events.last().unwrap()["choices"][0]["finish_reason"].clone())
            } else {
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                let call = &value["choices"][0]["message"]["tool_calls"][0];
                assert_eq!(call["type"], "function");
                (call["function"]["name"].as_str().unwrap().to_owned(),
                    call["function"]["arguments"].as_str().unwrap().to_owned(), value["choices"][0]["finish_reason"].clone())
            };
            assert_eq!(name, "lookup");
            assert_eq!(serde_json::from_str::<Value>(&arguments).unwrap(), json!({"city": "Taipei", "days": 2}));
            assert_eq!(finish, "tool_calls");
        }
    }

    #[tokio::test]
    async fn required_strict_tool_uses_the_glm_xml_grammar_after_reasoning() {
        let tools = json!([{"type": "function", "function": {"name": "lookup", "strict": true, "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": false}}},
            {"type": "function", "function": {"name": "other"}}]);
        for (choice, valid) in [(json!("required"), true), (json!({"type": "function", "function": {"name": "lookup"}}), true),
            (json!("required"), false)] {
            let body = json!({"model": MODEL, "tools": tools, "tool_choice": choice, "parallel_tool_calls": false,
                "messages": [{"role": "user", "content": "Weather?"}]});
            let named = body["tool_choice"].is_object();
            let text = if valid { "Must call.</think><tool_call>lookup<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>" }
                else { "Must call.</think>No." };
            let (status, _) = serve(body, text, move |job| {
                let grammar: Value = serde_json::from_str(&job.constraint.as_ref().unwrap().0).unwrap();
                let format = &grammar["format"];
                assert_eq!((format["type"].as_str(), format["elements"][1]["token"].as_u64()), (Some("sequence"), Some(154_842)));
                assert_eq!(format["elements"][0]["exclude_tokens"], json!([154_842]));
                let calls = &format["elements"][2];
                assert_eq!((calls["type"].as_str(), calls["at_least_one"].as_bool(), calls["stop_after_first"].as_bool()),
                    (Some("tags_with_separator"), Some(true), Some(true)));
                assert_eq!(calls["tags"][0]["begin"], "<tool_call>lookup");
                assert_eq!(calls["tags"][0]["content"]["style"], "glm_xml");
                assert_eq!(calls["tags"].as_array().unwrap().len(), if named { 1 } else { 2 });
                let instruction = if named { "<|system|>You must call the function lookup." }
                    else { "<|system|>You must call at least one provided function." };
                assert!(job.prompt.contains(instruction), "{}", job.prompt);
            }).await;
            assert_eq!(status, if valid { StatusCode::OK } else { StatusCode::INTERNAL_SERVER_ERROR });
        }
    }

    #[tokio::test]
    async fn multi_turn_tool_round_trip_prompt_matches_transformers() {
        let case = goldens().into_iter().find(|case| case["name"] == "tool_round_trip").unwrap();
        let mut body = request_for(&case["context"]);
        body["model"] = json!(MODEL);
        let expected = case["expected"]["exl3_k4"].as_str().unwrap().to_owned();
        let (status, _) = serve(body, "Paris.</think>Checking Paris next.", move |job| {
            assert_eq!(job.prompt, expected);
        }).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn thinking_toggles_and_model_metadata() {
        // Under the opt-in `low` setting every off form renders the Low effort
        // with the think block open, so the model's short plan comes back as
        // reasoning.
        let efforts = thinking_off_forms().into_iter().map(|off| (off, "Low"))
            .chain([(json!({"reasoning_effort": "low"}), "Low"), (json!({"reasoning_effort": "high"}), "High"),
                (json!({"reasoning_effort": "max"}), "Max"), (json!({}), "Max")]);
        for (options, effort) in efforts {
            let mut body = json!({"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]});
            body.as_object_mut().unwrap().extend(options.as_object().unwrap().clone());
            let (status, bytes) = serve_with(encoding().with_thinking_off(GlmThinkingOff::Low), body, "Greet.</think>Hello.", move |job| {
                assert_eq!(job.prompt, format!("[gMASK]<sop><|system|>Reasoning Effort: {effort}<|user|>Hi<|assistant|><think>"));
            }).await;
            assert_eq!(status, StatusCode::OK);
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            let message = &value["choices"][0]["message"];
            assert_eq!((message["content"].as_str(), message["reasoning_content"].as_str()), (Some("Hello."), Some("Greet.")),
                "{options}");
        }
        let (queue, _receive) = mpsc::channel::<NativeRequest>(1);
        let response = app(queue, encoding()).oneshot(Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
        let value: Value = serde_json::from_slice(&to_bytes(response.into_body(), 1 << 16).await.unwrap()).unwrap();
        assert_eq!((value["data"][0]["id"].as_str(), value["data"][0]["owned_by"].as_str()), (Some(MODEL), Some("wrldsuksgo2mars")));
    }

    /// By default (`empty`) an off request keeps glmrt's empty think block, and
    /// the reply is content only.
    #[tokio::test]
    async fn the_empty_thinking_off_setting_answers_without_reasoning() {
        for options in [json!({"thinking": {"type": "disabled"}}), json!({"reasoning_effort": "minimal"})] {
            let mut body = json!({"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]});
            body.as_object_mut().unwrap().extend(options.as_object().unwrap().clone());
            let (status, bytes) = serve_with(encoding(), body, "Hello.", |job| {
                assert_eq!(job.prompt, "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think></think>");
            }).await;
            assert_eq!(status, StatusCode::OK);
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            let message = &value["choices"][0]["message"];
            assert_eq!((message["content"].as_str(), message["reasoning_content"].as_str()), (Some("Hello."), None),
                "{options}");
        }
    }

    /// A call that cannot be read comes back as content, whole or streamed,
    /// beside any call that parsed; the finish is `tool_calls` only when one did.
    #[tokio::test]
    async fn unreadable_calls_come_back_as_content_in_both_response_modes() {
        let tools = json!([{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}}}}}]);
        let nameless = "<tool_call><arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>";
        let oslo = "<tool_call>lookup<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>";
        let truncated = "<tool_call>lookup<arg_key>city</arg_key><arg_value>Os";
        for (text, content, calls, finish) in [
            (format!("Plan.</think>Checking.\n{nameless}"), format!("Checking.\n{nameless}"), 0, "stop"),
            (format!("Plan.</think>Checking.\n{nameless}{oslo}"), format!("Checking.\n{nameless}"), 1, "tool_calls"),
            (format!("Plan.</think>{truncated}"), truncated.to_owned(), 0, "stop"),
            // An argument whose value never opens loses the whole call:
            // content, stop, never a tool call.
            (format!("Plan.</think><tool_call>lookup<arg_key>city</arg_key>Oslo</tool_call>"),
                "<tool_call>lookup<arg_key>city</arg_key>Oslo</tool_call>".to_owned(), 0, "stop"),
            // Markup in the key does the same.
            (format!("Plan.</think><tool_call>lookup<arg_key><bad></arg_key><arg_value>Oslo</arg_value></tool_call>"),
                "<tool_call>lookup<arg_key><bad></arg_key><arg_value>Oslo</arg_value></tool_call>".to_owned(), 0, "stop"),
        ] {
            for streaming in [false, true] {
                let body = json!({"model": MODEL, "stream": streaming, "tools": tools,
                    "messages": [{"role": "user", "content": "Weather in Oslo?"}]});
                let (status, bytes) = serve(body, text.clone(), |_| {}).await;
                assert_eq!(status, StatusCode::OK);
                let (reply, names, reason) = if streaming {
                    let events = sse_events(&bytes);
                    let reply: String = events.iter().filter_map(|event| event["choices"][0]["delta"]["content"].as_str()).collect();
                    let names: Vec<String> = events.iter()
                        .flat_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array().cloned().unwrap_or_default())
                        .filter_map(|delta| delta["function"]["name"].as_str().map(str::to_owned)).collect();
                    (reply, names, events.last().unwrap()["choices"][0]["finish_reason"].clone())
                } else {
                    let value: Value = serde_json::from_slice(&bytes).unwrap();
                    let message = &value["choices"][0]["message"];
                    let names = message["tool_calls"].as_array().into_iter().flatten()
                        .map(|call| call["function"]["name"].as_str().unwrap().to_owned()).collect();
                    (message["content"].as_str().unwrap_or_default().to_owned(), names, value["choices"][0]["finish_reason"].clone())
                };
                assert_eq!((reply.as_str(), names.len(), reason.as_str()), (content.as_str(), calls, Some(finish)),
                    "{text} streaming={streaming}");
            }
        }
    }

    /// A call is held until it closes: nothing of it streams early, and the
    /// stream's keepalive comment covers the wait.
    #[tokio::test]
    async fn a_held_call_keeps_the_stream_alive() {
        let (queue, mut receive) = mpsc::channel::<NativeRequest>(1);
        let worker = tokio::spawn(async move {
            let job = receive.recv().await.unwrap();
            let text = |content: &str| Ok(InferenceChunk::Text { content: content.into(), content_tokens: 1 });
            job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 9, prompt_cache_hit_tokens: 0 } })).unwrap();
            job.events.send(text("Look it up.</think><tool_call>lookup<arg_key>city</arg_key><arg_value>Tai")).unwrap();
            tokio::time::sleep(crate::openai::SSE_KEEPALIVE * 4).await;
            job.events.send(text("pei</arg_value></tool_call>")).unwrap();
            job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop })).unwrap();
        });
        let body = json!({"model": MODEL, "stream": true, "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object",
                "properties": {"city": {"type": "string"}}}}}]});
        let request = Request::post("/v1/chat/completions").header("content-type", "application/json")
            .body(Body::from(body.to_string())).unwrap();
        let response = app(queue, encoding()).oneshot(request).await.unwrap();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        worker.await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        let keepalive = text.find(": keepalive\n\n").expect("a keepalive while the call is held");
        assert!(keepalive < text.find("\"tool_calls\"").unwrap(), "{text}");
        let arguments: String = sse_events(&bytes).iter()
            .flat_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array().cloned().unwrap_or_default())
            .filter_map(|delta| delta["function"]["arguments"].as_str().map(str::to_owned)).collect();
        assert_eq!(arguments, r#"{"city":"Taipei"}"#);
    }
}
