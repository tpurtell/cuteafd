use super::*;
use crate::gateway::turn::*;
use axum::{routing::{get, post}, Router, response::IntoResponse};
use futures::StreamExt;

fn backend(flavor: Flavor) -> Upstream {
    Upstream::new(UpstreamConfig::new("http://localhost/v1", flavor, "test-model")).unwrap()
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
#[test]
fn deepseek_quirks_are_opt_in_and_reject_forced_thinking_tools() {
    let mut config = UpstreamConfig::new("http://localhost/v1", Flavor::OpenaiChat,"test-model");
    config.deepseek_thinking = true;
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
