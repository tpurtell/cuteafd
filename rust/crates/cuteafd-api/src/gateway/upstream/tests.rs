use super::*;
use crate::gateway::turn::*;
use axum::{routing::{get, post}, Router, response::IntoResponse};
use futures::StreamExt;

fn backend(flavor: Flavor) -> Upstream {
    let mut config = UpstreamConfig::new("http://localhost/v1",flavor,"test-model");
    config.capabilities.json_schema = true;
    config.capabilities.strict_tools = true;
    Upstream::new(config).unwrap()
}
fn turn() -> TurnRequest { TurnRequest { model: "test-model".into(), items: vec![Item::Message { role: Role::User, content: vec![Part::text("hello")] }], ..Default::default() } }
#[test]
fn chat_mapping_preserves_reasoning_tools_images_and_controls() {
    let mut turn = turn();
    turn.system = Some("system".into());
    turn.items.extend([
        Item::Reasoning { text: "plan".into(), signature: None },
        Item::Message { role: Role::Assistant, content: vec![Part::text("checking")] },
        Item::ToolCall { id: "call_1".into(), name: "check".into(), arguments: "{}".into() },
        Item::ToolCall { id: "call_2".into(), name: "check".into(), arguments: "{}".into() },
        Item::ToolResult { call_id: "call_1".into(), content: vec![Part::text("yes")], is_error: false },
        Item::Message { role: Role::User, content: vec![Part::Image { source: ImageSource::Base64 { media_type:"image/png".into(), data:"abc".into() }, detail:Some("low".into()) }] },
    ]);
    turn.tools.push(ToolSpec { name:"check".into(), description:None, parameters:json!({"type":"object"}), strict:true });
    turn.parallel_tool_calls = Some(false);
    turn.sampling.temperature = Some(0.2);
    turn.max_output_tokens = Some(256);
    turn.response_format = Some(json!({"name":"test","schema":{"type":"object"},"strict":true}));
    let request = backend(Flavor::OpenaiChat).map_request(&turn).unwrap();
    assert_eq!(request["messages"][2]["reasoning_content"], "plan");
    assert_eq!(request["messages"][2]["tool_calls"].as_array().unwrap().len(), 2);
    assert_eq!(request["messages"][2]["content"], "checking");
    assert_eq!(request["messages"][4]["content"][0]["image_url"]["url"], "data:image/png;base64,abc");
    assert_eq!(request["stream_options"]["include_usage"], true);
    assert_eq!(request["response_format"]["type"], "json_schema");
    assert!(request.get("thinking").is_none());
}
#[tokio::test]
async fn strict_tools_use_configured_endpoint_without_silent_downgrade() {
    let (url,task) = server(Router::new().route("/beta/chat/completions",post(|| async {
        ([("content-type","text/event-stream")],"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
    }))).await;
    let mut config = UpstreamConfig::new(url,Flavor::OpenaiChat,"test");
    config.thinking_toggle = true;
    config.capabilities.strict_tools = true;
    config.strict_tools_path = Some("/beta/chat/completions".into());
    let upstream = Upstream::new(config).unwrap();
    let mut turn = turn();
    turn.tools.push(ToolSpec { name:"check".into(),description:None,parameters:json!({"type":"object"}),strict:true });
    assert!(upstream.start(turn.clone()).await.is_ok());
    turn.response_format = Some(json!({"type":"json_schema","json_schema":{"schema":{"type":"object"}}}));
    let error = upstream.map_request(&turn).unwrap_err();
    assert_eq!(error.kind,ErrorKind::Unsupported);
    assert_eq!(error.param.as_deref(),Some("response_format"));
    turn.response_format = Some(json!({"type":"json_object"}));
    assert_eq!(upstream.map_request(&turn).unwrap()["response_format"]["type"],"json_object");
    task.abort();
}
#[test]
fn namespaced_tools_use_legal_wire_names_consistently() {
    let mut turn = turn();
    turn.tools.push(ToolSpec { name:"functions.apply_patch".into(),description:None,parameters:json!({"type":"object"}),strict:false });
    turn.tool_choice = ToolChoice::Named { name:"functions.apply_patch".into() };
    turn.items.push(Item::ToolCall { id:"call".into(),name:"functions.apply_patch".into(),arguments:"{}".into() });
    for flavor in [Flavor::OpenaiChat,Flavor::Anthropic] {
        let request = backend(flavor).map_request(&turn).unwrap();
        let name = mapping::wire_name("functions.apply_patch");
        assert!(name.len() <= 64 && !name.contains('.'));
        let wire = request.to_string();
        assert!(!wire.contains("functions.apply_patch"));
        assert!(wire.contains(&name));
    }
}
#[test]
fn thinking_quirks_are_opt_in_and_reject_forced_thinking_tools() {
    let mut config = UpstreamConfig::new("http://localhost/v1", Flavor::OpenaiChat,"test-model");
    config.thinking_toggle = true;
    let upstream = Upstream::new(config).unwrap();
    let mut turn = turn();
    turn.tool_choice = ToolChoice::Required;
    assert!(upstream.map_request(&turn).is_err());
    turn.reasoning.enabled = Some(false);
    assert_eq!(upstream.map_request(&turn).unwrap()["thinking"]["type"], "disabled");
}
#[test]
fn anthropic_mapping_groups_blocks_and_round_trips_thinking() {
    let mut turn = turn();
    turn.items.extend([
        Item::Reasoning { text:"plan".into(), signature:Some("signature".into()) },
        Item::ToolCall { id:"a".into(),name:"check".into(),arguments:"{}".into() },
        Item::ToolResult { call_id:"a".into(),content:vec![Part::text("ok")],is_error:true },
    ]);
    turn.reasoning.enabled = Some(true);
    turn.reasoning.effort = Some("high".into());
    let request = backend(Flavor::Anthropic).map_request(&turn).unwrap();
    assert_eq!(request["messages"][1]["content"][0]["signature"], "signature");
    assert_eq!(request["messages"][1]["content"][1]["type"], "tool_use");
    assert_eq!(request["messages"][2]["content"][0]["is_error"], true);
    assert_eq!(request["output_config"]["effort"], "high");
}
#[test]
fn sse_decoder_handles_split_utf8_crlf_multiline_comments_and_done() {
    let bytes = ": keepalive\r\ndata: {\r\ndata: \"text\":\"caf\u{e9}\"}\r\n\r\ndata: [DONE]\n\n".as_bytes();
    let mut decoder = sse::Decoder::default();
    let mut frames = Vec::new();
    for byte in bytes { frames.extend(decoder.feed(&[*byte]).unwrap()); }
    assert_eq!(frames, ["{\n\"text\":\"caf\u{e9}\"}", "[DONE]"]);
}
async fn parse(body: &str, flavor: Flavor) -> Result<Vec<TurnEvent>, GatewayError> {
    let stream = futures::stream::iter(body.as_bytes().chunks(3).map(|c| Ok::<_, ()>(bytes::Bytes::copy_from_slice(c))).collect::<Vec<_>>());
    sse::events(stream, flavor, Exchange::new(Tape::default(),flavor,json!({}),200)).collect::<Vec<_>>().await.into_iter().collect()
}
#[tokio::test]
async fn chat_sse_orders_sparse_tool_indices_and_usage_before_done() {
    let body = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"plan\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":3,\"id\":\"a\",\"function\":{\"name\":\"check\",\"arguments\":\"{\"}},{\"index\":0,\"id\":\"b\",\"function\":{\"name\":\"run\",\"arguments\":\"{}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":3,\"function\":{\"arguments\":\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":20,\"prompt_cache_hit_tokens\":80,\"completion_tokens_details\":{\"reasoning_tokens\":10}}}\n\n",
        "data: [DONE]\n\n");
    let events = parse(body, Flavor::OpenaiChat).await.unwrap();
    assert_eq!(events[0], TurnEvent::ReasoningDelta { text:"plan".into() });
    assert!(events.contains(&TurnEvent::ToolCallStart { index:3,id:"a".into(),name:"check".into() }));
    assert_eq!(events.last(),Some(&TurnEvent::Done { stop:StopReason::ToolUse }));
    let usage = events.iter().filter_map(|e| if let TurnEvent::Usage { usage } = e { Some(*usage) } else { None }).last().unwrap();
    assert_eq!((usage.input_tokens,usage.cached_input_tokens,usage.reasoning_tokens),(100,80,10));
}
#[tokio::test]
async fn disconnect_and_stream_error_are_errors() {
    assert!(parse("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",Flavor::OpenaiChat).await.is_err());
    assert!(parse("data: {\"error\":{\"message\":\"secret\"}}\n\n",Flavor::OpenaiChat).await.is_err());
    assert!(parse("data: [DONE]\n\n",Flavor::OpenaiChat).await.is_err());
}
#[tokio::test]
async fn anthropic_sse_thinking_signature_cached_usage_and_pause() {
    let body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"cache_read_input_tokens\":10,\"cache_creation_input_tokens\":7}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"pause_turn\"},\"usage\":{\"output_tokens\":12}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n");
    let events = parse(body,Flavor::Anthropic).await.unwrap();
    assert!(events.contains(&TurnEvent::ReasoningSignature { signature:"sig".into() }));
    assert_eq!(events.last(),Some(&TurnEvent::Done { stop:StopReason::PauseTurn }));
    let TurnEvent::Usage { usage } = events[events.len()-2] else { panic!("usage missing") };
    assert_eq!((usage.input_tokens,usage.cached_input_tokens,usage.cache_creation_input_tokens),(22,10,7));
}
#[test]
fn errors_bound_and_do_not_echo_upstream_bodies() {
    assert_eq!(http_error(429).kind,ErrorKind::RateLimited);
    assert_eq!(http_error(503).kind,ErrorKind::Overloaded);
    assert_eq!(http_error(401).upstream_status,Some(401));
    assert!(Upstream::new(UpstreamConfig::new("https://user:secret@example.com",Flavor::OpenaiChat,"test")).is_err());
}
async fn server(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener,app).await.unwrap(); });
    (format!("http://{addr}"),task)
}
#[tokio::test]
async fn http_status_is_reported_before_start_returns() {
    let (url, task) = server(Router::new().route("/chat/completions",post(|| async { (axum::http::StatusCode::TOO_MANY_REQUESTS,"never expose this upstream body") }))).await;
    let upstream = Upstream::new(UpstreamConfig::new(url,Flavor::OpenaiChat,"test")).unwrap();
    let error = match upstream.start(turn()).await { Err(e) => e, Ok(_) => panic!("expected error") };
    assert_eq!(error.status(),429);
    assert!(!error.message.contains("expose"));
    task.abort();
}
#[tokio::test]
async fn models_metadata_is_discovered_and_missing_listing_is_nonfatal() {
    let (url,task) = server(Router::new().route("/models",get(|| async { axum::Json(json!({"data":[{"id":"test","context_window":1000,"max_output_tokens":100,"input_modalities":["text","image"]}]})) }))).await;
    let upstream = Upstream::new(UpstreamConfig::new(url,Flavor::OpenaiChat,"test")).unwrap().discover().await;
    assert!(upstream.capabilities().vision);
    assert_eq!(upstream.models()[0].context_tokens,Some(1000));
    task.abort();
}
/// Explicit opt-in test capture: allowlisted endpoints/models, invented scratch prompts only.
#[tokio::test]
#[ignore = "paid provider capture; requires keys and scratch cases"]
async fn capture_scratch_fixtures() {
    use crate::gateway::record::{Sanitizer, TapeSink};
    #[derive(Default)]
    struct Collector(std::sync::Mutex<Vec<Value>>);
    impl TapeSink for Collector {
        fn record(&self, kind:&str, entry:Value) { self.0.lock().unwrap().push(json!({"kind":kind,"entry":entry})); }
    }
    fn merge(target:&mut Value, patch:&Value) {
        if let (Some(target),Some(patch)) = (target.as_object_mut(),patch.as_object()) {
            for (key,value) in patch { if value.is_object() && target.get(key).is_some_and(Value::is_object) { merge(target.get_mut(key).unwrap(),value); } else { target.insert(key.clone(),value.clone()); } }
        }
    }
    let cases:Vec<Value> = serde_json::from_slice(&std::fs::read("/home/tj/.cache/cuteafd/builds/api-gateway/scratch/upstream-cases.json").unwrap()).unwrap();
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gateway/upstream");
    std::fs::create_dir_all(&directory).unwrap();
    let sanitizer = Sanitizer::from_env([]);
    for case in cases {
        let flavor:Flavor = case["flavor"].as_str().unwrap().parse().unwrap();
        let mut value = serde_json::to_value(TurnRequest::default()).unwrap();
        merge(&mut value,&case["turn"]);
        let mut turn:TurnRequest = serde_json::from_value(value).unwrap();
        let collector = Arc::new(Collector::default());
        turn.tape = Tape(Some(collector.clone()));
        let provider = case["provider"].as_str().unwrap_or("deepseek");
        let (base,key_env) = match provider {
            "deepseek" => (match flavor { Flavor::OpenaiChat => "https://api.deepseek.com",Flavor::Anthropic => "https://api.deepseek.com/anthropic" }.to_string(),"DEEPSEEK_API_KEY"),
            "openrouter" => {
                assert!(turn.model.starts_with("xiaomi/") || turn.model.starts_with("qwen/"));
                ("https://openrouter.ai/api/v1".into(),"OPENROUTER_API_KEY")
            }
            "litellm" => {
                assert!(["claude/xiaomi/","claude/qwen/","claude/z-ai/","claude/moonshotai/","claude/deepseek/"].iter().any(|prefix| turn.model.starts_with(prefix)));
                (format!("{}/v1",std::env::var("LITELLM_BASE_URL").expect("private endpoint env required").trim_end_matches('/')),"LITELLM_API_KEY")
            }
            _ => panic!("provider not allowed by capture test"),
        };
        let mut config = UpstreamConfig::new(base,flavor,turn.model.clone());
        config.key = Some(std::env::var(key_env).expect("provider key must be set"));
        config.thinking_toggle = case["thinking"].as_bool().unwrap_or(false);
        config.capabilities.strict_tools = case["strict_tools"].as_bool().unwrap_or(false);
        config.capabilities.json_schema = case["json_schema"].as_bool().unwrap_or(false);
        config.strict_tools_path = case["strict_path"].as_str().map(str::to_string);
        let upstream = Upstream::new(config).unwrap();
        let result = match upstream.start(turn.clone()).await {
            Ok(stream) => stream.collect::<Vec<_>>().await.into_iter().collect::<Result<Vec<_>,_>>(),
            Err(error) => Err(error),
        };
        let mut fixture = json!({"version":1,"flavor":flavor.name(),"deepseek_thinking":case["thinking"],"strict_tools":case["strict_tools"],"json_schema":case["json_schema"],"turn":turn,
            "entries":collector.0.lock().unwrap().clone()});
        match result { Ok(events) => fixture["expected_events"] = json!(events), Err(error) => { fixture["expected_error_status"] = json!(error.status()); } }
        sanitizer.value(&mut fixture);
        std::fs::write(directory.join(format!("{}.json",case["name"].as_str().unwrap())),serde_json::to_vec_pretty(&fixture).unwrap()).unwrap();
        println!("captured {}",case["name"]);
        if case["expect_success"].as_bool() == Some(true) { assert!(fixture["expected_events"].is_array(),"provider probe failed; sanitized error fixture recorded"); }
        if let Some(expected) = case.get("expected_json") {
            let text = fixture["expected_events"].as_array().unwrap().iter().filter_map(|event| if event["event"] == "text_delta" { event["text"].as_str() } else { None }).collect::<String>();
            let actual = serde_json::from_str::<Value>(&text).unwrap();
            if case["expect_schema_violation"].as_bool() == Some(true) { assert_ne!(actual,*expected,"expected provider schema violation changed"); }
            else { assert_eq!(actual,*expected,"schema was not honored"); }
        }
        if let Some(expected) = case.get("expected_tool_arguments") {
            let arguments = fixture["expected_events"].as_array().unwrap().iter().filter_map(|event| if event["event"] == "tool_call_delta" { event["arguments"].as_str() } else { None }).collect::<String>();
            assert_eq!(serde_json::from_str::<Value>(&arguments).unwrap(),*expected,"strict tool schema was not honored");
        }
        if case["followup"].as_bool() == Some(true) {
            let mut reasoning = String::new();
            let mut text = String::new();
            let mut signature = None;
            let mut calls:std::collections::BTreeMap<usize,(String,String,String)> = Default::default();
            for event in fixture["expected_events"].as_array().expect("thinking tool turn must succeed") {
                let event:TurnEvent = serde_json::from_value(event.clone()).unwrap();
                match event {
                    TurnEvent::ReasoningDelta { text:t } => reasoning.push_str(&t),
                    TurnEvent::ReasoningSignature { signature:s } => signature=Some(s),
                    TurnEvent::TextDelta { text:t } => text.push_str(&t),
                    TurnEvent::ToolCallStart { index,id,name } => { calls.insert(index,(id,name,String::new())); },
                    TurnEvent::ToolCallDelta { index,arguments } => calls.get_mut(&index).unwrap().2.push_str(&arguments),
                    _ => {}
                }
            }
            assert!(!calls.is_empty(),"thinking tool turn must call a tool");
            if !reasoning.is_empty() { turn.items.push(Item::Reasoning { text:reasoning,signature }); }
            if !text.is_empty() { turn.items.push(Item::Message { role:Role::Assistant,content:vec![Part::text(text)] }); }
            for (id,name,arguments) in calls.values() { turn.items.push(Item::ToolCall { id:id.clone(),name:name.clone(),arguments:arguments.clone() }); }
            for (id,_,_) in calls.values() { turn.items.push(Item::ToolResult { call_id:id.clone(),content:vec![Part::text("Cloudvale: sunny, 20 C")],is_error:false }); }
            collector.0.lock().unwrap().clear();
            let events:Result<Vec<_>,_> = upstream.start(turn.clone()).await.unwrap().collect::<Vec<_>>().await.into_iter().collect();
            let mut followup = json!({"version":1,"flavor":flavor.name(),"deepseek_thinking":case["thinking"],"strict_tools":case["strict_tools"],"json_schema":case["json_schema"],"turn":turn,
                "entries":collector.0.lock().unwrap().clone(),"expected_events":events.unwrap()});
            sanitizer.value(&mut followup);
            std::fs::write(directory.join(format!("{}-followup.json",case["name"].as_str().unwrap())),serde_json::to_vec_pretty(&followup).unwrap()).unwrap();
            println!("captured {} followup",case["name"]);
        }
    }
}
#[tokio::test]
async fn dropping_turn_stream_closes_http_body() {
    use std::sync::atomic::{AtomicBool,Ordering};
    struct Guard(std::sync::Arc<AtomicBool>);
    impl Drop for Guard { fn drop(&mut self) { self.0.store(true,Ordering::SeqCst); } }
    let dropped = Arc::new(AtomicBool::new(false));
    let marker = dropped.clone();
    let (url,task) = server(Router::new().route("/chat/completions",post(move || {
        let guard = Guard(marker.clone());
        async move {
            let stream = async_stream::stream! {
                let _guard = guard;
                yield Ok::<_,std::io::Error>(bytes::Bytes::from_static(b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"));
                futures::future::pending::<()>().await;
            };
            ([("content-type","text/event-stream")],axum::body::Body::from_stream(stream)).into_response()
        }
    }))).await;
    let upstream = Upstream::new(UpstreamConfig::new(url,Flavor::OpenaiChat,"test")).unwrap();
    let stream = upstream.start(turn()).await.unwrap();
    drop(stream);
    tokio::time::timeout(Duration::from_secs(3),async { while !dropped.load(Ordering::SeqCst) { tokio::time::sleep(Duration::from_millis(10)).await; } }).await.unwrap();
    task.abort();
}

#[test]
fn structured_capabilities_are_opt_in_and_pcm16_is_wrapped_as_wav() {
    use base64::{Engine, engine::general_purpose::STANDARD};
    let upstream = Upstream::new(UpstreamConfig::new("http://localhost",Flavor::OpenaiChat,"test")).unwrap();
    let mut turn = turn();
    turn.tools.push(ToolSpec { name:"f".into(),description:None,parameters:json!({"type":"object"}),strict:true });
    assert_eq!(upstream.map_request(&turn).unwrap_err().param.as_deref(),Some("tools"));
    turn.tools.clear();
    turn.items.push(Item::Message { role:Role::User,content:vec![Part::Audio { format:"pcm16".into(),data:STANDARD.encode([1,0,2,0]) }] });
    let request = upstream.map_request(&turn).unwrap();
    let audio = &request["messages"][1]["content"][0]["input_audio"];
    assert_eq!(audio["format"],"wav");
    let wav = STANDARD.decode(audio["data"].as_str().unwrap()).unwrap();
    assert_eq!(&wav[..4],b"RIFF");
    assert_eq!(&wav[8..16],b"WAVEfmt ");
    assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()),24000);
    assert_eq!(&wav[44..],&[1,0,2,0]);
    turn.items.push(Item::Message { role:Role::User,content:vec![Part::Audio { format:"pcm16".into(),data:STANDARD.encode([1]) }] });
    assert_eq!(upstream.map_request(&turn).unwrap_err().param.as_deref(),Some("audio"));
}
#[tokio::test]
async fn nested_model_metadata_resolves_served_model_capabilities() {
    let (url,task) = server(Router::new().route("/models",get(|| async { axum::Json(json!({"data":[
        {"id":"other","architecture":{"input_modalities":["text"]}},
        {"id":"test","context_length":1050000,"top_provider":{"max_completion_tokens":131072},
         "architecture":{"input_modalities":["text","image","audio"]},"supported_parameters":["structured_outputs","tools","reasoning"]}
    ]})) }))).await;
    let upstream = Upstream::new(UpstreamConfig::new(url,Flavor::OpenaiChat,"test")).unwrap().discover().await;
    let caps = upstream.capabilities();
    assert!(caps.vision && caps.audio_in && caps.json_schema);
    assert!(!caps.strict_tools,"tools metadata alone is not a strict schema guarantee");
    assert_eq!(upstream.models()[0].context_tokens,Some(1050000));
    let mut turn = turn();turn.response_format=Some(json!({"type":"json_schema","json_schema":{"name":"test","schema":{"type":"object"},"strict":true}}));
    assert!(upstream.map_request(&turn).is_ok());
    task.abort();
}

#[test]
fn thinking_tool_loops_echo_reasoning_even_when_empty() {
    // Live Claude Code run (2026-10-09): the second tool round had no
    // reasoning, and the thinking-mode upstream rejected the follow-up.
    let mut turn = turn();
    turn.items.extend([
        Item::Reasoning { text: "look first".into(), signature: None },
        Item::ToolCall { id: "a".into(), name: "Read".into(), arguments: "{}".into() },
        Item::ToolResult { call_id: "a".into(), content: vec![Part::text("x")], is_error: false },
        Item::Message { role: Role::Assistant, content: vec![Part::text("editing")] },
        Item::ToolCall { id: "b".into(), name: "Edit".into(), arguments: "{}".into() },
        Item::ToolResult { call_id: "b".into(), content: vec![Part::text("ok")], is_error: false },
    ]);
    let mut config = UpstreamConfig::new("http://localhost/v1", Flavor::OpenaiChat, "test-model");
    config.thinking_toggle = true;
    let upstream = Upstream::new(config).unwrap();
    let request = upstream.map_request(&turn).unwrap();
    let assistants: Vec<&serde_json::Value> = request["messages"].as_array().unwrap().iter().filter(|m| m["role"] == "assistant").collect();
    assert_eq!(assistants[0]["reasoning_content"], "look first");
    assert_eq!(assistants[1]["reasoning_content"], "");
    turn.reasoning.enabled = Some(false);
    let request = upstream.map_request(&turn).unwrap();
    assert!(request["messages"].as_array().unwrap().iter().all(|m| m.get("reasoning_content").is_none() || m["reasoning_content"] != ""));
    let plain = backend(Flavor::OpenaiChat).map_request(&turn).unwrap();
    assert!(plain["messages"].as_array().unwrap().iter().all(|m| m["reasoning_content"] != ""), "off without the switch");
}

#[tokio::test]
async fn chat_reasoning_alias_streams_without_double_counting() {
    let events = parse("data: {\"choices\":[{\"delta\":{\"reasoning\":\"plan\",\"reasoning_content\":\"plan\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",Flavor::OpenaiChat).await.unwrap();
    assert_eq!(events.iter().filter(|e| matches!(e,TurnEvent::ReasoningDelta { .. })).count(),1);
}
