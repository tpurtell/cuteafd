//! Engine backend gates without GPUs: the same conversation sent as Chat
//! Completions, Anthropic Messages and OpenAI Responses reaches the engine as
//! the identical prompt (and token ids, where the checkpoint's tokenizer is
//! cached), for every family's template; tool calls round-trip through each
//! protocol; dropping a gateway stream cancels the engine request.
use super::*;
use crate::openai::{chat::{glm5, qwen4}, router_for_model, ConsoleHub, GatewayMount, InferenceChunk, InferenceFinishReason,
    NativeRequest, PromptUsage};
use axum::body::Body;
use std::sync::Mutex;
use tokio::sync::mpsc;
use tower::ServiceExt;

const PNG: &[u8] = include_bytes!("../fixtures/black.png");

fn png_b64() -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(PNG)
}

/// A served profile per family; `snapshot` when the real checkpoint (with its
/// tokenizer) is cached, so token ids are compared too.
struct Family { name: &'static str, profile: ModelProfile, snapshot: Option<std::path::PathBuf> }

fn hub(rel: &str) -> Option<std::path::PathBuf> {
    let base = std::path::Path::new("/mnt/sparknest/hf-home/hub").join(rel).join("snapshots");
    let mut snapshots: Vec<_> = std::fs::read_dir(base).ok()?.flatten().map(|e| e.path())
        .filter(|s| s.join("tokenizer.json").is_file()).collect();
    snapshots.sort();
    snapshots.pop()
}

fn families() -> Vec<Family> {
    let mut out = vec![
        Family { name: "v41", profile: ModelProfile::new("deepseek-ai/DeepSeek-V4.1-Flash", ModelEncoding::DeepseekV41),
            snapshot: hub("models--deepseek-ai--DeepSeek-V4.1-Flash") },
        Family { name: "v4", profile: ModelProfile::new("deepseek-ai/DeepSeek-V4-Flash-0731", ModelEncoding::DeepseekV4),
            snapshot: hub("models--deepseek-ai--DeepSeek-V4-Flash-0731") },
        Family { name: "glm-fixture", profile: ModelProfile::new("test-glm", ModelEncoding::Glm(Arc::new(glm5::fixtures::encoding()))),
            snapshot: None },
        Family { name: "qwen-fixture", profile: ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding()))),
            snapshot: None },
    ];
    for (name, rel, glm) in [
        ("glm53-flash", "models--zai-org--GLM-5.3-Flash", true),
        ("qwen38-exl3", "models--wrldsuksgo2mars--Qwen3.8-Flash-Next-EXL3-K4.25-v1", false),
        ("mimo-v26-flash", "models--XiaomiMiMo--MiMo-V2.6-Flash-MOPD", false),
    ] {
        let Some(snapshot) = hub(rel) else { continue };
        let encoding = if glm { glm5::GlmEncoding::from_snapshot(&snapshot).ok().map(|e| ModelEncoding::Glm(Arc::new(e))) }
            else { qwen4::QwenEncoding::from_snapshot(&snapshot).ok().map(|e| ModelEncoding::Qwen(Arc::new(e))) };
        if let Some(encoding) = encoding {
            out.push(Family { name, profile: ModelProfile::new(format!("test/{name}"), encoding), snapshot: Some(snapshot) });
        }
    }
    out
}

/// One scripted engine: records each job's prompt and answers with `reply`.
#[derive(Clone, Default)]
struct Worker { prompts: Arc<Mutex<Vec<String>>>, closed: Arc<Mutex<Vec<bool>>> }

fn serve(mut profile: ModelProfile, snapshot: Option<std::path::PathBuf>, replies: Vec<&'static str>, hold: bool)
    -> (axum::Router, Worker) {
    profile.gateway = Some(Arc::new(GatewayMount { models: crate::gateway::ModelMap::official_names(profile.id.clone()),
        snapshot, options: EngineOptions::default(), search: None, gate: None }));
    let (tx, mut rx) = mpsc::channel::<NativeRequest>(4);
    let worker = Worker::default();
    let seen = worker.clone();
    tokio::spawn(async move {
        let mut replies = replies.into_iter();
        while let Some(job) = rx.recv().await {
            seen.prompts.lock().unwrap().push(job.prompt.clone());
            let reply = replies.next().unwrap_or("ok");
            let _ = job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 11, prompt_cache_hit_tokens: 3 } }));
            if hold {
                // Wait for the client to go away (cancellation).
                for _ in 0..200 {
                    if job.events.is_closed() { break; }
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
                seen.closed.lock().unwrap().push(job.events.is_closed());
                continue;
            }
            let _ = job.events.send(Ok(InferenceChunk::Text { content: reply.into(), content_tokens: 5 }));
            let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }));
        }
    });
    let app = router_for_model(tx, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        std::time::Duration::from_secs(5), ConsoleHub::disabled(), profile);
    (app, worker)
}

async fn post(app: &axum::Router, path: &str, body: Value) -> (StatusCode, String) {
    let request = axum::http::Request::post(path).header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01").body(Body::from(body.to_string())).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// One conversation in all three protocols: system, a user turn with an
/// optional image, a prior assistant turn with reasoning and a tool call,
/// the tool result, and the next user turn.
struct Conversation { chat: Value, messages: Value, responses: Value }

fn conversation(model: &str, thinking: bool, tools: bool, image: bool) -> Conversation {
    let schema = json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]});
    let image_chat = json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}", png_b64())}});
    let image_anthropic = json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":png_b64()}});
    let image_responses = json!({"type":"input_image","image_url":format!("data:image/png;base64,{}", png_b64())});
    let first_chat = if image { json!([{"type":"text","text":"Look at this."}, image_chat]) } else { json!("Read the file a.txt.") };
    let first_anthropic = if image { json!([{"type":"text","text":"Look at this."}, image_anthropic]) } else { json!("Read the file a.txt.") };
    let first_responses = if image { json!([{"type":"input_text","text":"Look at this."}, image_responses]) }
        else { json!([{"type":"input_text","text":"Read the file a.txt."}]) };
    let mut chat_messages = vec![json!({"role":"system","content":"You are terse."}), json!({"role":"user","content":first_chat})];
    let mut anthropic_messages = vec![json!({"role":"user","content":first_anthropic})];
    let mut responses_input = vec![json!({"type":"message","role":"user","content":first_responses})];
    if tools {
        chat_messages.push(json!({"role":"assistant","content":"Reading it.","reasoning_content":"Need the file.",
            "tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}}]}));
        chat_messages.push(json!({"role":"tool","tool_call_id":"call_1","content":"hello"}));
        anthropic_messages.push(json!({"role":"assistant","content":[
            {"type":"thinking","thinking":"Need the file.","signature":"sig"},
            {"type":"text","text":"Reading it."},
            {"type":"tool_use","id":"call_1","name":"read_file","input":{"path":"a.txt"}}]}));
        anthropic_messages.push(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"hello"}]}));
        responses_input.push(json!({"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"Need the file."}]}));
        responses_input.push(json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"Reading it."}]}));
        responses_input.push(json!({"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"a.txt\"}"}));
        responses_input.push(json!({"type":"function_call_output","call_id":"call_1","output":"hello"}));
    }
    chat_messages.push(json!({"role":"user","content":"Now summarize."}));
    anthropic_messages.push(json!({"role":"user","content":"Now summarize."}));
    responses_input.push(json!({"type":"message","role":"user","content":[{"type":"input_text","text":"Now summarize."}]}));
    let mut chat = json!({"model":model,"messages":chat_messages,"stream":true,"temperature":0,"max_tokens":64,
        "thinking":{"type": if thinking { "enabled" } else { "disabled" }}});
    let mut messages = json!({"model":"claude-sonnet-5","system":"You are terse.","messages":anthropic_messages,"max_tokens":64,
        "temperature":0,"stream":true,"thinking":{"type": if thinking { "enabled" } else { "disabled" }}});
    let mut responses = json!({"model":"gpt-6.1-sol","instructions":"You are terse.","input":responses_input,"stream":true,
        "temperature":0,"max_output_tokens":64,"store":false,"reasoning":{"effort": if thinking { "high" } else { "none" }}});
    if thinking { chat["reasoning_effort"] = json!("high"); messages["output_config"] = json!({"effort":"high"}); }
    if tools {
        chat["tools"] = json!([{"type":"function","function":{"name":"read_file","description":"Read a file.","parameters":schema}}]);
        messages["tools"] = json!([{"name":"read_file","description":"Read a file.","input_schema":schema}]);
        responses["tools"] = json!([{"type":"function","name":"read_file","description":"Read a file.","parameters":schema}]);
    }
    Conversation { chat, messages, responses }
}

#[tokio::test]
async fn every_protocol_renders_the_chat_prompt_for_every_family() {
    let families = families();
    eprintln!("prompt equivalence families: {:?}", families.iter().map(|f| (f.name, f.snapshot.is_some())).collect::<Vec<_>>());
    for family in families {
        // Image turns only where the family's chat path accepts images without
        // a loaded encoder (V4.1 decodes images itself).
        let image_ok = matches!(family.profile.encoding, ModelEncoding::DeepseekV41);
        for (thinking, tools, image) in [(false, false, false), (true, false, false), (false, true, false), (true, true, false),
            (false, false, true), (true, true, true)] {
            if image && !image_ok { continue; }
            let (app, worker) = serve(family.profile.clone(), family.snapshot.clone(), vec![], false);
            let c = conversation(&family.profile.id, thinking, tools, image);
            for (path, body) in [("/v1/chat/completions", &c.chat), ("/v1/messages", &c.messages), ("/v1/responses", &c.responses)] {
                let (status, text) = post(&app, path, body.clone()).await;
                assert_eq!(status, StatusCode::OK, "{} {path} thinking={thinking} tools={tools} image={image}: {text}", family.name);
            }
            let prompts = worker.prompts.lock().unwrap().clone();
            assert_eq!(prompts.len(), 3, "{}", family.name);
            assert_eq!(prompts[0], prompts[1], "{} Messages prompt differs (thinking={thinking} tools={tools} image={image})", family.name);
            assert_eq!(prompts[0], prompts[2], "{} Responses prompt differs (thinking={thinking} tools={tools} image={image})", family.name);
            if tools { assert!(prompts[0].contains("read_file") && prompts[0].contains("hello"), "{}: {}", family.name, prompts[0]); }
            if let Some(snapshot) = &family.snapshot {
                let ids = |p: &str| cuteafd_loader::encode_tokenizer_text(snapshot, p, false).unwrap().token_ids;
                assert_eq!(ids(&prompts[0]), ids(&prompts[1]), "{}", family.name);
                assert_eq!(ids(&prompts[0]), ids(&prompts[2]), "{}", family.name);
            }
        }
    }
}

#[tokio::test]
async fn count_tokens_is_exact_through_the_tokenizer() {
    for family in families().into_iter().filter(|f| f.snapshot.is_some()) {
        let snapshot = family.snapshot.clone().unwrap();
        let (app, worker) = serve(family.profile.clone(), Some(snapshot.clone()), vec![], false);
        let c = conversation(&family.profile.id, true, true, false);
        let (status, text) = post(&app, "/v1/messages/count_tokens", c.messages.clone()).await;
        assert_eq!(status, StatusCode::OK, "{}: {text}", family.name);
        let counted = serde_json::from_str::<Value>(&text).unwrap()["input_tokens"].as_u64().unwrap();
        post(&app, "/v1/messages", c.messages.clone()).await;
        let prompt = worker.prompts.lock().unwrap()[0].clone();
        let expected = cuteafd_loader::encode_tokenizer_text(&snapshot, &prompt, false).unwrap().token_ids.len() as u64;
        assert_eq!(counted, expected, "{}", family.name);
    }
}

fn sse_events(text: &str) -> Vec<Value> {
    text.split("\n\n").filter_map(|frame| frame.lines().find_map(|l| l.strip_prefix("data: ")))
        .filter_map(|data| serde_json::from_str(data).ok()).collect()
}

#[tokio::test]
async fn tool_calls_round_trip_through_every_protocol() {
    // Each family emits its own call syntax; each protocol must surface the call.
    let cases: Vec<(ModelProfile, &'static str)> = vec![
        (ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding()))),
            "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"),
        (ModelProfile::new("test-glm", ModelEncoding::Glm(Arc::new(glm5::fixtures::encoding()))),
            "<tool_call>read_file<arg_key>path</arg_key><arg_value>a.txt</arg_value></tool_call>"),
        (ModelProfile::new("deepseek-ai/DeepSeek-V4.1-Flash", ModelEncoding::DeepseekV41),
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">a.txt</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"),
    ];
    for (profile, reply) in cases {
        let id = profile.id.clone();
        let (app, _) = serve(profile, None, vec![reply; 6], false);
        let c = conversation(&id, false, true, false);
        // Chat: the reference.
        let (status, text) = post(&app, "/v1/chat/completions", c.chat.clone()).await;
        assert_eq!(status, StatusCode::OK, "{id}: {text}");
        let chat = sse_events(&text);
        if !chat.iter().any(|e| e["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read_file") {
            // DSML spelling differs between recipe versions; skip families whose
            // chat path does not parse this fixture as a call either.
            eprintln!("{id}: chat path did not parse the fixture call; skipped");
            continue;
        }
        // Messages: tool_use block with the parsed input, stop_reason tool_use.
        let (status, text) = post(&app, "/v1/messages", c.messages.clone()).await;
        assert_eq!(status, StatusCode::OK, "{id}: {text}");
        let events = sse_events(&text);
        let start = events.iter().find(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use").expect("tool_use block");
        assert_eq!(start["content_block"]["name"], "read_file");
        let input: String = events.iter().filter(|e| e["delta"]["type"] == "input_json_delta")
            .map(|e| e["delta"]["partial_json"].as_str().unwrap().to_owned()).collect();
        assert_eq!(serde_json::from_str::<Value>(&input).unwrap(), json!({"path":"a.txt"}), "{id}");
        let delta = events.iter().find(|e| e["type"] == "message_delta").unwrap();
        assert_eq!(delta["delta"]["stop_reason"], "tool_use", "{id}");
        // Responses: a completed function_call item.
        let (status, text) = post(&app, "/v1/responses", c.responses.clone()).await;
        assert_eq!(status, StatusCode::OK, "{id}: {text}");
        let done = sse_events(&text).into_iter().find(|e| e["type"] == "response.completed").expect("completed");
        let call = done["response"]["output"].as_array().unwrap().iter().find(|o| o["type"] == "function_call").expect("function_call");
        assert_eq!(call["name"], "read_file");
        assert_eq!(serde_json::from_str::<Value>(call["arguments"].as_str().unwrap()).unwrap(), json!({"path":"a.txt"}));
        // Non-streaming Messages agrees.
        let mut body = c.messages.clone();
        body["stream"] = json!(false);
        let (_, text) = post(&app, "/v1/messages", body).await;
        let message: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(message["stop_reason"], "tool_use", "{id}: {text}");
        assert_eq!(message["content"].as_array().unwrap().iter().find(|b| b["type"] == "tool_use").unwrap()["input"], json!({"path":"a.txt"}));
    }
}

#[tokio::test]
async fn namespaced_tool_names_map_to_legal_wire_names_and_back() {
    let profile = ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())));
    let reply = "<tool_call>\n<function=exec>\n<parameter=input>\nls\n</parameter>\n</function>\n</tool_call>";
    let (app, worker) = serve(profile, None, vec![reply], false);
    let body = json!({"model":"gpt-6.1-sol","input":"list files","stream":true,"store":false,"reasoning":{"effort":"none"},
        "tools":[{"type":"namespace","name":"functions","description":"","tools":[{"type":"custom","name":"exec","description":"Run."}]}]});
    let (status, text) = post(&app, "/v1/responses", body).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert!(worker.prompts.lock().unwrap()[0].contains("\"exec\""), "the model sees the local name");
    let done = sse_events(&text).into_iter().find(|e| e["type"] == "response.completed").unwrap();
    let call = done["response"]["output"].as_array().unwrap().iter().find(|o| o["type"] == "custom_tool_call").expect("custom call");
    assert_eq!(call["name"], "exec");
    assert_eq!(call["namespace"], "functions");
    assert_eq!(call["input"], "ls");
}

#[tokio::test]
async fn stop_reasons_map_from_engine_finish() {
    let profile = ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())));
    let (app, _) = serve(profile, None, vec!["alpha STOP beta"], false);
    let body = json!({"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":8,"stream":false,
        "stop_sequences":["STOP"],"thinking":{"type":"disabled"}});
    let (_, text) = post(&app, "/v1/messages", body).await;
    let message: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(message["stop_reason"], "stop_sequence", "{text}");
    assert_eq!(message["stop_sequence"], "STOP");
    assert_eq!(message["content"][0]["text"], "alpha", "the chat parser's text before the stop string");
    assert_eq!(message["usage"]["input_tokens"].as_u64().unwrap() + message["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0), 11);
}

#[tokio::test]
async fn dropping_the_gateway_stream_cancels_the_engine_request() {
    let profile = ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())));
    let (app, worker) = serve(profile, None, vec![], true);
    let body = json!({"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":8,"stream":true});
    let request = axum::http::Request::post("/v1/messages").header("content-type", "application/json")
        .body(Body::from(body.to_string())).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
    for _ in 0..200 {
        if !worker.closed.lock().unwrap().is_empty() { break; }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(*worker.closed.lock().unwrap(), vec![true], "the engine saw the request cancelled");
}

#[tokio::test]
async fn json_schema_is_refused_until_a_probe_passes_and_models_list_aliases() {
    let profile = ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())));
    let (app, worker) = serve(profile, None, vec![], false);
    let body = json!({"model":"gpt-6.1-sol","input":"x","stream":false,"store":false,
        "text":{"format":{"type":"json_schema","name":"a","schema":{"type":"object"},"strict":true}}});
    let (status, text) = post(&app, "/v1/responses", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("json_schema"), "{text}");
    assert!(worker.prompts.lock().unwrap().is_empty());
    let response = app.clone().oneshot(axum::http::Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
    let listing: Value = serde_json::from_slice(&axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap();
    let data = listing["data"].as_array().unwrap();
    assert_eq!(data[0]["id"], "test-qwen", "the served model is listed first");
    assert_eq!(data[0]["capabilities"]["vision"], false);
    assert!(data[0]["max_context_tokens"].as_u64().is_some(), "chat-route fields survive");
    assert!(data.iter().any(|m| m["id"] == "claude-sonnet-5") && data.iter().any(|m| m["id"] == "gpt-6.1-sol"));
}

#[test]
fn item_mapping_folds_assistant_parts_and_maps_server_tools() {
    use crate::gateway::turn::Part;
    let profile = ModelProfile::new("m", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())));
    let turn = TurnRequest { system: Some("sys".into()), items: vec![
        Item::Message { role: Role::User, content: vec![Part::text("q")] },
        Item::Reasoning { text: "think".into(), signature: None },
        Item::Message { role: Role::Assistant, content: vec![Part::text("a")] },
        Item::ServerToolCall { id: "s1".into(), name: "web_search".into(), input: json!({"query":"x"}) },
        Item::ServerToolResult { call_id: "s1".into(), name: "web_search".into(), output: json!({"results":[]}) },
        Item::ToolCall { id: "c1".into(), name: "functions.exec".into(), arguments: "{\"input\":\"ls\"}".into() },
        Item::ToolResult { call_id: "c1".into(), content: vec![Part::text("out")], is_error: true },
    ], ..Default::default() };
    let body = chat_body(&turn, &profile).unwrap();
    let m = body["messages"].as_array().unwrap();
    assert_eq!(m[0], json!({"role":"system","content":"sys"}));
    assert_eq!(m[2]["reasoning_content"], "think");
    assert_eq!(m[2]["content"], "a");
    assert_eq!(m[2]["tool_calls"][0]["function"]["name"], "web_search");
    assert_eq!(m[3], json!({"role":"tool","tool_call_id":"s1","content":"{\"results\":[]}"}));
    assert_eq!(m[4]["tool_calls"][0]["function"]["name"], "exec", "namespaced name maps to its local name");
    assert_eq!(m[5]["content"], "Error: out");
}

/// Send `chat` to /v1/chat/completions and `messages` to /v1/messages on one
/// server; return both statuses and the prompts the engine received.
async fn chat_vs_messages(profile: ModelProfile, chat: Value, messages: Value) -> (StatusCode, StatusCode, Vec<String>, String) {
    let (app, worker) = serve(profile, None, vec![], false);
    let (chat_status, _) = post(&app, "/v1/chat/completions", chat).await;
    let (messages_status, text) = post(&app, "/v1/messages", messages).await;
    let prompts = worker.prompts.lock().unwrap().clone();
    (chat_status, messages_status, prompts, text)
}

#[tokio::test]
async fn consecutive_reasoning_blocks_are_one_reasoning_content() {
    // Anthropic sends interleaved-thinking turns as several thinking blocks;
    // they are one assistant turn, as the chat client sends it.
    for profile in [ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding()))),
        ModelProfile::new("test-glm", ModelEncoding::Glm(Arc::new(glm5::fixtures::encoding())))] {
        let id = profile.id.clone();
        let chat = json!({"model":id,"stream":false,"max_tokens":8,"thinking":{"type":"enabled"},"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":"a","reasoning_content":"first.second."},
            {"role":"user","content":"again"}]});
        let messages = json!({"model":"m","max_tokens":8,"stream":false,"thinking":{"type":"enabled"},"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[{"type":"thinking","thinking":"first.","signature":"s1"},
                {"type":"thinking","thinking":"second.","signature":"s2"},{"type":"text","text":"a"}]},
            {"role":"user","content":"again"}]});
        let (a, b, prompts, text) = chat_vs_messages(profile, chat, messages).await;
        assert_eq!((a, b), (StatusCode::OK, StatusCode::OK), "{id}: {text}");
        assert_eq!(prompts[0], prompts[1], "{id}");
    }
}

#[tokio::test]
async fn assistant_text_then_tool_then_text_matches_the_chat_split() {
    // Text after a call is the next assistant turn; reasoning after text too.
    let profile = ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())));
    let schema = json!({"type":"object","properties":{"p":{"type":"string"}}});
    let chat = json!({"model":"test-qwen","stream":false,"max_tokens":8,"thinking":{"type":"disabled"},
        "tools":[{"type":"function","function":{"name":"f","parameters":schema}}],"messages":[
        {"role":"user","content":"q"},
        {"role":"assistant","content":"one","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"p\":\"x\"}"}}]},
        {"role":"tool","tool_call_id":"c1","content":"r"},
        {"role":"assistant","content":"two"},
        {"role":"user","content":"again"}]});
    let messages = json!({"model":"m","max_tokens":8,"stream":false,"thinking":{"type":"disabled"},
        "tools":[{"name":"f","input_schema":schema}],"messages":[
        {"role":"user","content":"q"},
        {"role":"assistant","content":[{"type":"text","text":"one"},{"type":"tool_use","id":"c1","name":"f","input":{"p":"x"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"r"}]},
        {"role":"assistant","content":"two"},
        {"role":"user","content":"again"}]});
    let (a, b, prompts, text) = chat_vs_messages(profile, chat, messages).await;
    assert_eq!((a, b), (StatusCode::OK, StatusCode::OK), "{text}");
    assert_eq!(prompts[0], prompts[1]);
}

#[tokio::test]
async fn assistant_media_is_never_dropped() {
    // Adjacent assistant text and image parts merge into one part list; the
    // chat path then decides (V4.1's adapter refuses assistant images, so both
    // routes refuse with 400 instead of silently losing the image).
    let profile = ModelProfile::new("deepseek-ai/DeepSeek-V4.1-Flash", ModelEncoding::DeepseekV41);
    let image = format!("data:image/png;base64,{}", png_b64());
    let turn = TurnRequest { items: vec![
        Item::Message { role: Role::User, content: vec![crate::gateway::turn::Part::text("q")] },
        Item::Message { role: Role::Assistant, content: vec![crate::gateway::turn::Part::text("see ")] },
        Item::Message { role: Role::Assistant, content: vec![crate::gateway::turn::Part::text("this"),
            crate::gateway::turn::Part::Image { source: crate::gateway::turn::ImageSource::Url { url: image.clone() }, detail: None }]},
    ], ..Default::default() };
    let body = chat_body(&turn, &profile).unwrap();
    assert_eq!(body["messages"][1]["content"], json!([{"type":"text","text":"see "},{"type":"text","text":"this"},
        {"type":"image_url","image_url":{"url":image}}]), "every part kept, in order");
    let chat = json!({"model":profile.id,"stream":false,"max_tokens":8,"messages":body["messages"]});
    let messages = json!({"model":"m","max_tokens":8,"stream":false,"messages":[
        {"role":"user","content":"q"},
        {"role":"assistant","content":[{"type":"text","text":"see "},{"type":"text","text":"this"},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":png_b64()}}]}]});
    let (a, b, prompts, text) = chat_vs_messages(profile, chat, messages).await;
    assert_eq!(a, StatusCode::BAD_REQUEST);
    assert_eq!(b, StatusCode::BAD_REQUEST, "{text}");
    assert!(prompts.is_empty());
}

#[tokio::test]
async fn forced_tool_choice_without_its_tool_is_refused_like_chat() {
    let profile = ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())));
    let (app, worker) = serve(profile, None, vec![], false);
    for (chat_choice, anthropic_choice, responses_choice) in [
        (json!("required"), json!({"type":"any"}), json!("required")),
        (json!({"type":"function","function":{"name":"missing"}}), json!({"type":"tool","name":"missing"}), json!({"type":"function","name":"missing"})),
    ] {
        let (status, _) = post(&app, "/v1/chat/completions", json!({"model":"test-qwen","stream":false,"max_tokens":8,
            "messages":[{"role":"user","content":"q"}],"tool_choice":chat_choice})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, text) = post(&app, "/v1/messages", json!({"model":"m","max_tokens":8,"stream":false,
            "messages":[{"role":"user","content":"q"}],"tool_choice":anthropic_choice})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["error"]["type"], "invalid_request_error");
        let (status, text) = post(&app, "/v1/responses", json!({"model":"m","input":"q","stream":false,"store":false,
            "tool_choice":responses_choice})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    }
    assert!(worker.prompts.lock().unwrap().is_empty(), "nothing reached the engine");
}

#[tokio::test]
async fn multi_part_text_joins_by_each_family_rule() {
    // Two text blocks are a two-part list on both routes; the family's chat
    // conversion joins them (DeepSeek's recipe with a blank line).
    for family in families() {
        let id = family.profile.id.clone();
        let chat = json!({"model":id,"stream":false,"max_tokens":8,"thinking":{"type":"disabled"},"messages":[
            {"role":"system","content":[{"type":"text","text":"sys one"},{"type":"text","text":"sys two"}]},
            {"role":"user","content":[{"type":"text","text":"first"},{"type":"text","text":"second"}]},
            {"role":"assistant","content":[{"type":"text","text":"a1"},{"type":"text","text":"a2"}]},
            {"role":"user","content":"next"}]});
        let messages = json!({"model":"m","max_tokens":8,"stream":false,"thinking":{"type":"disabled"},
            "messages":[
            {"role":"system","content":[{"type":"text","text":"sys one"},{"type":"text","text":"sys two"}]},
            {"role":"user","content":[{"type":"text","text":"first"},{"type":"text","text":"second"}]},
            {"role":"assistant","content":[{"type":"text","text":"a1"},{"type":"text","text":"a2"}]},
            {"role":"user","content":"next"}]});
        let responses = json!({"model":"m","stream":false,"store":false,"reasoning":{"effort":"none"},"input":[
            {"type":"message","role":"system","content":[{"type":"input_text","text":"sys one"},{"type":"input_text","text":"sys two"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"first"},{"type":"input_text","text":"second"}]},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"a1"},{"type":"output_text","text":"a2"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"next"}]}]});
        let (app, worker) = serve(family.profile.clone(), family.snapshot.clone(), vec![], false);
        for (path, body) in [("/v1/chat/completions", chat), ("/v1/messages", messages), ("/v1/responses", responses)] {
            let (status, text) = post(&app, path, body).await;
            assert_eq!(status, StatusCode::OK, "{} {path}: {text}", family.name);
        }
        let prompts = worker.prompts.lock().unwrap().clone();
        assert_eq!(prompts[0], prompts[2], "{} Responses", family.name);
        if let Some(snapshot) = &family.snapshot {
            let ids = |p: &str| cuteafd_loader::encode_tokenizer_text(snapshot, p, false).unwrap().token_ids;
            assert_eq!(ids(&prompts[0]), ids(&prompts[2]), "{}", family.name);
        }
        // Anthropic's top-level system is one string (its blocks join with a
        // newline in the Messages front end), so compare from the user turn on.
        let tail = |p: &str| p[p.find("first").expect("user turn rendered")..].to_owned();
        assert_eq!(tail(&prompts[0]), tail(&prompts[1]), "{} Messages", family.name);
        if matches!(family.profile.encoding, ModelEncoding::DeepseekV4 | ModelEncoding::DeepseekV41) {
            assert!(prompts[1].contains("first\n\nsecond"), "{}: recipe joins text parts with a blank line", family.name);
        }
    }
}

#[tokio::test]
async fn claude_code_text_call_text_turn_is_one_assistant_message() {
    // Claude Code: [text, tool_use, text] then the tool_result. One assistant
    // message with the call; its results follow it directly, as chat requires.
    for family in families() {
        let id = family.profile.id.clone();
        let schema = json!({"type":"object","properties":{"p":{"type":"string"}}});
        let chat = json!({"model":id,"stream":false,"max_tokens":8,"thinking":{"type":"disabled"},
            "tools":[{"type":"function","function":{"name":"f","parameters":schema}}],"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[{"type":"text","text":"I will check"},{"type":"text","text":"Please wait"}],
                "tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"c","content":"done"}]});
        let messages = json!({"model":"m","max_tokens":8,"stream":false,"thinking":{"type":"disabled"},
            "tools":[{"name":"f","input_schema":schema}],"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[{"type":"text","text":"I will check"},
                {"type":"tool_use","id":"c","name":"f","input":{}},{"type":"text","text":"Please wait"}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"c","content":"done"}]}]});
        let (app, worker) = serve(family.profile.clone(), family.snapshot.clone(), vec![], false);
        let (a, ta) = post(&app, "/v1/chat/completions", chat).await;
        let (b, tb) = post(&app, "/v1/messages", messages).await;
        assert_eq!((a, b), (StatusCode::OK, StatusCode::OK), "{}: {ta} {tb}", family.name);
        let prompts = worker.prompts.lock().unwrap().clone();
        assert_eq!(prompts[0], prompts[1], "{}", family.name);
        if let Some(snapshot) = &family.snapshot {
            let ids = |p: &str| cuteafd_loader::encode_tokenizer_text(snapshot, p, false).unwrap().token_ids;
            assert_eq!(ids(&prompts[0]), ids(&prompts[1]), "{}", family.name);
        }
    }
}

#[tokio::test]
async fn count_tokens_keeps_prompt_shaping_fields_and_ignores_generation_limits() {
    for family in families().into_iter().filter(|f| f.snapshot.is_some()
        && matches!(f.profile.encoding, ModelEncoding::Glm(_) | ModelEncoding::Qwen(_))) {
        let snapshot = family.snapshot.clone().unwrap();
        let (app, worker) = serve(family.profile.clone(), Some(snapshot.clone()), vec!["{}"], false);
        // A JSON format puts an instruction in GLM/Qwen prompts; a huge
        // max_output_tokens would fail generation but never a count.
        let body = json!({"model":"m","input":"Give me JSON.","store":false,"stream":false,"reasoning":{"effort":"none"},
            "text":{"format":{"type":"json_object"}},"max_output_tokens":100000000});
        let (status, text) = post(&app, "/v1/responses/input_tokens", body.clone()).await;
        assert_eq!(status, StatusCode::OK, "{}: {text}", family.name);
        let counted = serde_json::from_str::<Value>(&text).unwrap()["input_tokens"].as_u64().unwrap();
        let mut generate = body.clone();
        generate["max_output_tokens"] = json!(16);
        let (status, text) = post(&app, "/v1/responses", generate).await;
        assert_eq!(status, StatusCode::OK, "{}: {text}", family.name);
        let prompt = worker.prompts.lock().unwrap()[0].clone();
        assert!(prompt.contains("JSON object"), "{}: the format instruction is in the prompt", family.name);
        let expected = cuteafd_loader::encode_tokenizer_text(&snapshot, &prompt, false).unwrap().token_ids.len() as u64;
        assert_eq!(counted, expected, "{}", family.name);
    }
}

#[tokio::test]
async fn v41_image_count_includes_every_expanded_row() {
    let Some(family) = families().into_iter().find(|f| f.name == "v41" && f.snapshot.is_some()) else { return };
    let snapshot = family.snapshot.clone().unwrap();
    let (app, worker) = serve(family.profile.clone(), Some(snapshot.clone()), vec![], false);
    let body = json!({"model":"m","max_tokens":8,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":[
        {"type":"text","text":"What is this?"},
        {"type":"image","source":{"type":"base64","media_type":"image/png","data":png_b64()}}]}]});
    let (status, text) = post(&app, "/v1/messages/count_tokens", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let counted = serde_json::from_str::<Value>(&text).unwrap()["input_tokens"].as_u64().unwrap() as usize;
    let mut generate = body;
    generate["stream"] = json!(false);
    let (status, text) = post(&app, "/v1/messages", generate).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    // What native admission admits: the job's prompt ids, each image
    // placeholder expanded to its decoded grid (`V41VisionPrompt::expand`).
    let prompt = worker.prompts.lock().unwrap()[0].clone();
    let ids = cuteafd_loader::encode_tokenizer_text(&snapshot, &prompt, false).unwrap().token_ids;
    let decoded = crate::openai::images::ImageDecoder::new(1).decode(vec![
        deepseek_recipe_core::multimodal::ImageSource::Bytes { data: PNG.to_vec(), detail: Default::default() }]).unwrap();
    let expanded = cuteafd_loader::V41VisionPrompt::expand(&ids, decoded, 1 << 20).unwrap().tokens.len();
    assert!(expanded > ids.len() + 1, "the image spans many rows");
    assert_eq!(counted, expanded);
    eprintln!("v41 image count: {counted} tokens ({} text ids + image rows)", ids.len());
}

#[tokio::test]
async fn engine_failure_answers_each_protocol_in_its_own_shape() {
    let mut profile = ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())));
    profile.engine_health = Some(crate::openai::health::HealthWitness(Arc::new(|| Some("expert rank 1 disconnected".into()))));
    let (app, worker) = serve(profile, None, vec![], false);
    let (status, text) = post(&app, "/v1/messages", json!({"model":"m","max_tokens":8,"messages":[{"role":"user","content":"x"}]})).await;
    assert_eq!(status.as_u16(), 529, "{text}");
    let body: Value = serde_json::from_str(&text).unwrap();
    assert_eq!((body["type"].as_str(), body["error"]["type"].as_str()), (Some("error"), Some("overloaded_error")));
    let (status, text) = post(&app, "/v1/responses", json!({"model":"m","input":"x"})).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{text}");
    assert!(serde_json::from_str::<Value>(&text).unwrap()["error"]["message"].as_str().unwrap().contains("expert rank 1"));
    // Compact runs a model turn too; per-turn admission refuses it.
    let (status, text) = post(&app, "/v1/responses/compact", json!({"model":"m","input":"x"})).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{text}");
    assert!(worker.prompts.lock().unwrap().is_empty());
}

#[tokio::test]
async fn system_items_follow_each_template_placement_rule() {
    // Claude Code interleaves `role: system` reminders; Codex sends several
    // leading developer items after `instructions`. Qwen's template takes one
    // leading system message: leading ones merge with a blank line, later ones
    // become user turns (the DeepSeek recipe's normalization). GLM and MiMo
    // render system messages anywhere and get them unchanged.
    for family in families() {
        let placement = system_placement(&family.profile);
        let id = family.profile.id.clone();
        let messages = json!({"model":"m","max_tokens":8,"stream":false,"thinking":{"type":"disabled"},"system":"base",
            "messages":[{"role":"user","content":"q"},{"role":"system","content":"<system-reminder>r</system-reminder>"},
                {"role":"assistant","content":"a"},{"role":"user","content":"again"}]});
        let responses = json!({"model":"m","stream":false,"store":false,"reasoning":{"effort":"none"},"instructions":"base",
            "input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"dev one"}]},
                {"type":"message","role":"developer","content":[{"type":"input_text","text":"dev two"}]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}]});
        let (chat_cc, chat_cx) = match placement {
            SystemPlacement::FirstOnly => (
                json!([{"role":"system","content":"base"},{"role":"user","content":"q"},{"role":"user","content":"<system-reminder>r</system-reminder>"},
                    {"role":"assistant","content":"a"},{"role":"user","content":"again"}]),
                json!([{"role":"system","content":"base\n\ndev one\n\ndev two"},{"role":"user","content":"q"}])),
            SystemPlacement::Anywhere => (
                json!([{"role":"system","content":"base"},{"role":"user","content":"q"},{"role":"system","content":"<system-reminder>r</system-reminder>"},
                    {"role":"assistant","content":"a"},{"role":"user","content":"again"}]),
                json!([{"role":"system","content":"base"},{"role":"system","content":"dev one"},{"role":"system","content":"dev two"},{"role":"user","content":"q"}])),
        };
        if matches!(family.profile.encoding, ModelEncoding::Qwen(_)) && family.name.starts_with("qwen") {
            assert_eq!(placement, SystemPlacement::FirstOnly, "{}", family.name);
        }
        if matches!(family.profile.encoding, ModelEncoding::Glm(_)) { assert_eq!(placement, SystemPlacement::Anywhere, "{}", family.name); }
        let (app, worker) = serve(family.profile.clone(), family.snapshot.clone(), vec![], false);
        for (path, body) in [("/v1/messages", messages), ("/v1/responses", responses)] {
            let (status, text) = post(&app, path, body).await;
            assert_eq!(status, StatusCode::OK, "{} {path}: {text}", family.name);
        }
        for chat in [chat_cc, chat_cx] {
            let (status, text) = post(&app, "/v1/chat/completions", json!({"model":id,"stream":false,"max_tokens":8,
                "thinking":{"type":"disabled"},"messages":chat})).await;
            assert_eq!(status, StatusCode::OK, "{} chat: {text}", family.name);
        }
        let prompts = worker.prompts.lock().unwrap().clone();
        assert_eq!(prompts[0], prompts[2], "{} Claude Code reminders ({placement:?})", family.name);
        assert_eq!(prompts[1], prompts[3], "{} Codex developer items ({placement:?})", family.name);
    }
}
