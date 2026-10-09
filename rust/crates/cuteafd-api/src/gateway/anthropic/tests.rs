use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use futures::future::BoxFuture;
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::gateway::{self, testing::Scripted, turn::*, Gateway, GatewayError, ModelMap};

fn router(scripts: Vec<Vec<TurnEvent>>) -> (axum::Router, Scripted) {
    let backend = Scripted::new(scripts);
    let mut models = ModelMap::single("served-model");
    models.accept_any = true;
    (
        gateway::router(Arc::new(Gateway::new(Arc::new(backend.clone()), models))),
        backend,
    )
}
fn request(value: Value) -> Request<Body> {
    Request::post("/v1/messages?beta=true")
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .header("anthropic-beta", "anything,new-beta")
        .header("x-stainless-lang", "python")
        .header("anthropic-dangerous-direct-browser-access", "true")
        .body(Body::from(value.to_string()))
        .unwrap()
}
fn prompt(stream: bool) -> Value {
    json!({"model":"claude-test","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"stream":stream})
}
async fn wire(
    router: axum::Router,
    request: Request<Body>,
) -> (StatusCode, axum::http::HeaderMap, String) {
    let response = router.oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, headers, String::from_utf8(bytes.to_vec()).unwrap())
}
fn frames(body: &str) -> Vec<Value> {
    body.split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            let data = frame
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap();
            let value: Value = serde_json::from_str(data).unwrap();
            assert!(frame
                .lines()
                .any(|l| l == format!("event: {}", value["type"].as_str().unwrap())));
            value
        })
        .collect()
}
fn kinds(frames: &[Value]) -> Vec<&str> {
    frames.iter().map(|v| v["type"].as_str().unwrap()).collect()
}
fn text(text: &str) -> TurnEvent {
    TurnEvent::TextDelta { text: text.into() }
}
fn done(stop: StopReason) -> TurnEvent {
    TurnEvent::Done { stop }
}

#[tokio::test]
async fn text_sse_golden_and_json_equivalence() {
    let script = vec![
        text("hello"),
        text(" world"),
        TurnEvent::Usage {
            usage: Usage {
                input_tokens: 17,
                output_tokens: 2,
                cached_input_tokens: 4,
                cache_creation_input_tokens: 3,
                ..Default::default()
            },
        },
        done(StopReason::EndTurn),
    ];
    let (app, _) = router(vec![script.clone(), script]);
    let (status, headers, body) = wire(app.clone(), request(prompt(true))).await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers["request-id"].to_str().unwrap().starts_with("req_"));
    assert_eq!(headers["content-type"], "text/event-stream");
    let events = frames(&body);
    assert_eq!(
        kinds(&events),
        vec![
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
    );
    assert_eq!(events[0]["message"]["content"], json!([]));
    assert_eq!(events[0]["message"]["stop_reason"], Value::Null);
    assert_eq!(
        events[2],
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})
    );
    assert_eq!(
        events[3],
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}})
    );
    let (_, _, body) = wire(app, request(prompt(false))).await;
    let message: Value = serde_json::from_str(&body).unwrap();
    let mut folded = fold(&events);
    folded["id"] = message["id"].clone();
    assert_eq!(message, folded);
    assert_eq!(message["usage"]["input_tokens"], 10);
    assert_eq!(message["usage"]["cache_creation_input_tokens"], 3);
    assert_eq!(message["usage"]["cache_read_input_tokens"], 4);
}

// Mirrors the SDK's fold, independent of the renderer's accumulator.
fn fold(events: &[Value]) -> Value {
    let mut message = events[0]["message"].clone();
    for event in &events[1..] {
        let index = event["index"].as_u64().unwrap_or(0) as usize;
        match event["type"].as_str().unwrap() {
            "content_block_start" => message["content"]
                .as_array_mut()
                .unwrap()
                .push(event["content_block"].clone()),
            "content_block_delta" => {
                let delta = &event["delta"];
                let key = match delta["type"].as_str().unwrap() {
                    "text_delta" => "text",
                    "thinking_delta" => "thinking",
                    "signature_delta" => "signature",
                    "input_json_delta" => "partial_json",
                    _ => panic!("unknown delta"),
                };
                let block = &mut message["content"][index];
                let old = block[key].as_str().unwrap_or_default();
                block[key] = Value::String(format!("{old}{}", delta[key].as_str().unwrap()));
            }
            "content_block_stop" => {
                let block = &mut message["content"][index];
                if let Some(args) = block.get("partial_json") {
                    block["input"] = serde_json::from_str(args.as_str().unwrap()).unwrap();
                    block.as_object_mut().unwrap().remove("partial_json");
                }
            }
            "message_delta" => {
                message["stop_reason"] = event["delta"]["stop_reason"].clone();
                message["stop_sequence"] = event["delta"]["stop_sequence"].clone();
                message["usage"]
                    .as_object_mut()
                    .unwrap()
                    .extend(event["usage"].as_object().unwrap().clone());
            }
            _ => (),
        }
    }
    message
}

#[tokio::test]
async fn thinking_signature_and_parallel_tools_round_trip() {
    let script = vec![
        TurnEvent::ReasoningDelta {
            text: "reason".into(),
        },
        text("before"),
        TurnEvent::ToolCallStart {
            index: 0,
            id: "call-a".into(),
            name: "a".into(),
        },
        TurnEvent::ToolCallStart {
            index: 1,
            id: "call-b".into(),
            name: "b".into(),
        },
        TurnEvent::ToolCallDelta {
            index: 1,
            arguments: "{\"b\":".into(),
        },
        TurnEvent::ToolCallDelta {
            index: 0,
            arguments: "{\"a\":1}".into(),
        },
        TurnEvent::ToolCallDelta {
            index: 1,
            arguments: "2}".into(),
        },
        TurnEvent::ToolCallEnd { index: 0 },
        TurnEvent::ToolCallEnd { index: 1 },
        text("after"),
        done(StopReason::ToolUse),
    ];
    let (app, _) = router(vec![script.clone(), script]);
    let (_, _, body) = wire(app.clone(), request(prompt(true))).await;
    let events = frames(&body);
    assert_eq!(
        events
            .iter()
            .filter(|v| v["type"] == "content_block_start")
            .map(|v| v["index"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    let signature = events
        .iter()
        .find(|v| v["delta"]["type"] == "signature_delta")
        .unwrap();
    assert!(!signature["delta"]["signature"].as_str().unwrap().is_empty());
    let (_, _, body) = wire(app, request(prompt(false))).await;
    let message: Value = serde_json::from_str(&body).unwrap();
    let mut folded = fold(&events);
    folded["id"] = message["id"].clone();
    assert_eq!(message, folded);
    assert_eq!(message["content"][2]["input"], json!({"a":1}));
    assert_eq!(message["content"][3]["input"], json!({"b":2}));
    let replay = json!({"model":"m","max_tokens":1,"messages":[{"role":"assistant","content":message["content"]}]});
    let (turn, _) = super::request::parse(&replay, true).unwrap();
    assert!(
        matches!(&turn.items[0], Item::Reasoning { text, signature: Some(_) } if text == "reason")
    );
}

#[tokio::test]
async fn stop_reasons_and_last_usage_win() {
    for (reason, name, sequence) in [
        (StopReason::EndTurn, "end_turn", None),
        (StopReason::MaxTokens, "max_tokens", None),
        (StopReason::ToolUse, "tool_use", None),
        (StopReason::PauseTurn, "pause_turn", None),
        (StopReason::Refusal, "refusal", None),
        (StopReason::ContentFilter, "refusal", None),
        (
            StopReason::StopSequence {
                sequence: Some("STOP".into()),
            },
            "stop_sequence",
            Some("STOP"),
        ),
    ] {
        let script = vec![
            TurnEvent::Usage {
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 99,
                    ..Default::default()
                },
            },
            TurnEvent::Usage {
                usage: Usage {
                    input_tokens: 12,
                    output_tokens: 5,
                    ..Default::default()
                },
            },
            done(reason),
        ];
        let (app, _) = router(vec![script.clone(), script]);
        let (_, _, body) = wire(app.clone(), request(prompt(false))).await;
        let result: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(result["stop_reason"], name);
        assert_eq!(result["stop_sequence"], json!(sequence));
        assert_eq!(result["usage"]["output_tokens"], 5);
        let (_, _, body) = wire(app, request(prompt(true))).await;
        let events = frames(&body);
        assert_eq!(events[events.len() - 2]["delta"]["stop_reason"], name);
    }
}

#[test]
fn all_request_blocks_and_controls() {
    let value = json!({"model":"m","max_tokens":100,"unknown":{"a":1},"context_management":{},"metadata":{"user_id":"scratch"},"service_tier":"auto",
        "system":[{"type":"text","text":"one","cache_control":{"type":"ephemeral"}},{"type":"text","text":"two"}],
        "messages":[{"role":"user","content":[
            {"type":"text","text":"text","citations":[],"cache_control":{}},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":"aQ=="}},
            {"type":"image","source":{"type":"url","url":"https://example.com/i"}},
            {"type":"document","source":{"type":"text","media_type":"text/plain","data":"plain"}},
            {"type":"document","source":{"type":"base64","media_type":"text/plain","data":"Ynl0ZXM="}},
            {"type":"document","title":"pdf","source":{"type":"base64","media_type":"application/pdf","data":"cGRm"}},
            {"type":"document","source":{"type":"content","content":[{"type":"text","text":"nested"}]}},
            {"type":"search_result","title":"hit","source":"https://example.com","content":[{"type":"text","text":"snippet"}]},
            {"type":"container_upload","file_id":"ignored"}]},
            {"role":"assistant","content":[{"type":"thinking","thinking":"reason","signature":"any"},{"type":"redacted_thinking","data":"opaque"},
                {"type":"tool_use","id":"toolu_a","name":"fn","input":{"x":1}},
                {"type":"server_tool_use","id":"srvtoolu_s","name":"web_search","input":{"query":"q"}},
                {"type":"web_search_tool_result","tool_use_id":"srvtoolu_s","content":[{"type":"web_search_result","url":"https://example.com","title":"hit","encrypted_content":"c25pcHBldA==","page_age":"2026"}]}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_a","is_error":true,"content":[{"type":"text","text":"oops"},{"type":"image","source":{"type":"url","url":"https://example.com/i"}}]}]}],
        "tools":[{"type":"custom","name":"fn","description":"desc","input_schema":{"type":"object"},"strict":true,"cache_control":{}}],
        "tool_choice":{"type":"tool","name":"fn","disable_parallel_tool_use":true},"stop_sequences":["STOP"],"temperature":0.2,"top_p":0.9,"top_k":20,
        "thinking":{"type":"enabled","budget_tokens":50},"output_config":{"effort":"high","format":{"type":"json_schema","schema":{"type":"object"}}}});
    let (turn, _) = super::request::parse(&value, true).unwrap();
    assert_eq!(turn.system.as_deref(), Some("one\ntwo"));
    assert_eq!(turn.max_output_tokens, Some(100));
    assert_eq!(turn.parallel_tool_calls, Some(false));
    assert_eq!(turn.reasoning.budget_tokens, Some(50));
    assert_eq!(turn.reasoning.effort.as_deref(), Some("high"));
    assert_eq!(turn.sampling.top_k, Some(20));
    assert_eq!(turn.tool_choice, ToolChoice::Named { name: "fn".into() });
    assert!(turn.tools[0].strict);
    assert!(
        matches!(&turn.items[0], Item::Message { content, .. } if content.iter().any(|p| matches!(p, Part::File { .. })))
    );
    assert!(
        matches!(&turn.items[5], Item::ServerToolResult { output, .. } if output["results"][0]["content"] == "snippet")
    );
    assert!(
        matches!(&turn.items[6], Item::ToolResult { content, is_error:true, .. } if content.len() == 2)
    );
}

#[test]
fn tool_definitions_and_unknown_server_tools() {
    for kind in [
        "bash_20250124",
        "text_editor_20250124",
        "text_editor_20250429",
        "text_editor_20250728",
        "memory_20250818",
        "computer_20241022",
        "computer_20250124",
        "computer_20251124",
    ] {
        let mut value = prompt(false);
        value["tools"] = json!([{"type":kind,"name":"client","display_width_px":1920,"display_height_px":1080,"display_number":1}]);
        assert!(super::request::parse(&value, true).unwrap().0.tools[0]
            .parameters
            .is_object());
    }
    for kind in [
        "web_search_20250305",
        "web_search_20260209",
        "web_search_20260318",
        "web_search_future",
    ] {
        let mut value = prompt(false);
        value["tools"] = json!([{"type":kind,"name":"web_search","max_uses":2,"allowed_domains":["example.com"],"user_location":{"type":"approximate","country":"US"}}]);
        let (turn, _) = super::request::parse(&value, true).unwrap();
        assert_eq!(turn.hosted.web_search.unwrap().max_uses, Some(2));
    }
    for kind in [
        "web_fetch_20260209",
        "code_execution_20260521",
        "computer_future",
        "unknown_server",
    ] {
        let mut value = prompt(false);
        value["tools"] = json!([{"type":kind,"name":"unsupported"}]);
        assert!(super::request::parse(&value, true)
            .unwrap_err()
            .message
            .contains(kind));
    }
    for mode in ["auto", "none", "any"] {
        let mut value = prompt(false);
        value["tool_choice"] = json!({"type":mode});
        assert!(super::request::parse(&value, true).is_ok());
    }
    for mode in ["adaptive", "disabled", "between_tools"] {
        let mut value = prompt(false);
        value["thinking"] = json!({"type":mode});
        assert!(super::request::parse(&value, true).is_ok());
    }
}

#[test]
fn request_bounds_computer_metadata_and_unsupported_content() {
    for (key, invalid) in [
        ("temperature", json!(-0.1)),
        ("temperature", json!(1e100)),
        ("top_p", json!(1.1)),
        ("top_k", json!(-1)),
    ] {
        let mut value = prompt(false);
        value[key] = invalid;
        assert!(super::request::parse(&value, true)
            .unwrap_err()
            .message
            .contains(key));
    }
    let mut value = prompt(false);
    value["max_tokens"] = json!(0);
    assert!(super::request::parse(&value, true).is_ok());
    value["tools"] = json!([{"type":"computer_20251124","name":"computer","display_width_px":1920,"display_height_px":1080,"display_number":1}]);
    for enabled in [false, true] {
        value["tools"][0]["enable_zoom"] = json!(enabled);
        let tool = super::request::parse(&value, true)
            .unwrap()
            .0
            .tools
            .remove(0);
        assert!(tool.description.unwrap().contains("1920x1080"));
        assert_eq!(
            tool.parameters["properties"]["action"]["enum"]
                .as_array()
                .unwrap()
                .contains(&json!("zoom")),
            enabled
        );
    }
    for kind in ["input_audio", "audio", "code_execution_tool_result"] {
        let mut value = prompt(false);
        value["messages"][0]["content"] = json!([{"type":kind,"data":"opaque"}]);
        assert!(super::request::parse(&value, true)
            .unwrap_err()
            .message
            .contains(kind));
    }
}

#[tokio::test]
async fn truncated_tools_preserve_max_tokens_and_partial_object() {
    for arguments in [
        "{\"x\":",
        "{\"x\":1,\"unfinished\":\"abc",
        "{\"nested\":{\"x\":1",
        "{\"items\":[1,2,",
    ] {
        let expected = super::render::tool_input(arguments, true).unwrap();
        let script = vec![
            TurnEvent::ToolCallStart {
                index: 0,
                id: "a".into(),
                name: "f".into(),
            },
            TurnEvent::ToolCallDelta {
                index: 0,
                arguments: arguments.into(),
            },
            TurnEvent::Usage {
                usage: Usage {
                    input_tokens: 12,
                    output_tokens: 2,
                    ..Default::default()
                },
            },
            done(StopReason::MaxTokens),
        ];
        let (app, _) = router(vec![script.clone(), script]);
        let (status, _, body) = wire(app.clone(), request(prompt(false))).await;
        assert_eq!(status, StatusCode::OK);
        let message: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(message["stop_reason"], "max_tokens");
        assert_eq!(message["content"][0]["input"], expected);
        let (_, _, body) = wire(app, request(prompt(true))).await;
        let events = frames(&body);
        assert_eq!(
            events[events.len() - 2]["delta"]["stop_reason"],
            "max_tokens"
        );
        assert_eq!(events.last().unwrap()["type"], "message_stop");
    }
}

#[tokio::test]
async fn refused_or_cancelled_open_tools_are_not_runnable() {
    for (stop, expected) in [
        (StopReason::Refusal, "refusal"),
        (StopReason::ContentFilter, "refusal"),
        (StopReason::Cancelled, "max_tokens"),
    ] {
        let script = vec![
            TurnEvent::ToolCallStart {
                index: 0,
                id: "a".into(),
                name: "f".into(),
            },
            TurnEvent::ToolCallDelta {
                index: 0,
                arguments: "{\"x\":1,\"cut\":".into(),
            },
            done(stop),
        ];
        let (app, _) = router(vec![script.clone(), script]);
        let (status, _, body) = wire(app.clone(), request(prompt(false))).await;
        assert_eq!(status, StatusCode::OK);
        let message: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(message["stop_reason"], expected);
        assert_eq!(message["content"][0]["input"], json!({"x":1}));
        let (_, _, body) = wire(app, request(prompt(true))).await;
        let events = frames(&body);
        assert!(kinds(&events).contains(&"content_block_stop"));
        assert_eq!(events[events.len() - 2]["delta"]["stop_reason"], expected);
    }
}

#[tokio::test]
async fn zero_output_limit_counts_without_generation() {
    let (app, backend) = router(vec![]);
    let mut value = prompt(false);
    value["max_tokens"] = json!(0);
    let (status, _, body) = wire(app.clone(), request(value.clone())).await;
    assert_eq!(status, StatusCode::OK);
    let message: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(message["content"], json!([]));
    assert_eq!(message["stop_reason"], "max_tokens");
    assert_eq!(message["usage"]["output_tokens"], 0);
    assert!(message["usage"]["input_tokens"].as_u64().unwrap() > 0);
    value["stream"] = json!(true);
    let (status, _, body) = wire(app, request(value)).await;
    assert_eq!(status, StatusCode::OK);
    let mut folded = fold(&frames(&body));
    let mut message = message;
    folded.as_object_mut().unwrap().remove("id");
    message.as_object_mut().unwrap().remove("id");
    assert_eq!(folded, message);
    assert!(backend.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn request_rejections_and_count_tokens() {
    let (app, _) = router(vec![]);
    let mut value = prompt(false);
    value.as_object_mut().unwrap().remove("max_tokens");
    let (status, headers, body) = wire(app.clone(), request(value.clone())).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["request_id"], headers["request-id"].to_str().unwrap());
    let req = Request::post("/v1/messages/count_tokens?beta=true")
        .header("content-type", "application/json")
        .body(Body::from(value.to_string()))
        .unwrap();
    let (status, _, body) = wire(app.clone(), req).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        serde_json::from_str::<Value>(&body).unwrap()["input_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    let req = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from("{"))
        .unwrap();
    let (status, _, body) = wire(app, req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["type"],
        "error"
    );
}

#[tokio::test]
async fn errors_before_first_event_and_mid_stream() {
    for prefix in [vec![], vec![Ok(text("partial"))]] {
        let (app, backend) = router(vec![]);
        let mut script = prefix.clone();
        script.push(Err(GatewayError::new(
            gateway::ErrorKind::Overloaded,
            "busy",
        )));
        backend.scripts.lock().unwrap().push(script);
        let (status, _, body) = wire(app, request(prompt(true))).await;
        if prefix.is_empty() {
            assert_eq!(status.as_u16(), 529);
            assert_eq!(
                serde_json::from_str::<Value>(&body).unwrap()["error"]["type"],
                "overloaded_error"
            );
        } else {
            assert_eq!(status, StatusCode::OK);
            let events = frames(&body);
            assert_eq!(events.last().unwrap()["type"], "error");
            assert!(!kinds(&events).contains(&"message_stop"));
        }
    }
    let (app, _) = router(vec![]);
    let (status, _, _) = wire(app, request(prompt(true))).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn initial_silence_keeps_stream_live_and_drop_cancels() {
    use futures::StreamExt;
    use gateway::backend::{Backend, BackendCapabilities, ModelInfo, TurnStream};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    };
    use std::time::Duration;

    struct Cancel(Arc<AtomicBool>);
    impl Drop for Cancel {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    struct Delayed {
        release: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
        cancelled: Arc<AtomicBool>,
    }
    impl Backend for Delayed {
        fn name(&self) -> &str {
            "delayed"
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities::default()
        }
        fn models(&self) -> Vec<ModelInfo> {
            Vec::new()
        }
        fn start(&self, _: TurnRequest) -> BoxFuture<'static, Result<TurnStream, GatewayError>> {
            let release = self.release.lock().unwrap().take().unwrap();
            let cancel = Cancel(self.cancelled.clone());
            Box::pin(async move {
                let stream: TurnStream = Box::pin(async_stream::stream! {
                    let _cancel = cancel;
                    let _ = release.await;
                    yield Ok(text("after silence"));
                    yield Ok(done(StopReason::EndTurn));
                });
                Ok(stream)
            })
        }
        fn count_tokens(&self, _: TurnRequest) -> BoxFuture<'static, Result<u32, GatewayError>> {
            Box::pin(async { Ok(1) })
        }
    }
    for complete in [true, false] {
        let (release, receiver) = tokio::sync::oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let backend = Delayed {
            release: Mutex::new(Some(receiver)),
            cancelled: cancelled.clone(),
        };
        let mut models = ModelMap::single("served-model");
        models.accept_any = true;
        let app = gateway::router(Arc::new(Gateway::new(Arc::new(backend), models)));
        let response =
            tokio::time::timeout(Duration::from_secs(1), app.oneshot(request(prompt(true))))
                .await
                .expect("headers must not wait for a model token")
                .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body().into_data_stream();
        for kind in ["message_start", "ping"] {
            let bytes = tokio::time::timeout(Duration::from_secs(1), body.next())
                .await
                .expect("initial frames must not wait for a model token")
                .unwrap()
                .unwrap();
            assert_eq!(
                frames(std::str::from_utf8(&bytes).unwrap())[0]["type"],
                kind
            );
        }
        if complete {
            let bytes = tokio::time::timeout(Duration::from_secs(17), body.next())
                .await
                .expect("silent backend must receive a keepalive")
                .unwrap()
                .unwrap();
            assert_eq!(
                frames(std::str::from_utf8(&bytes).unwrap())[0]["type"],
                "ping"
            );
            release.send(()).unwrap();
            let mut remaining = String::new();
            while let Some(bytes) = body.next().await {
                remaining.push_str(std::str::from_utf8(&bytes.unwrap()).unwrap());
            }
            let events = frames(&remaining);
            assert_eq!(events[1]["delta"]["text"], "after silence");
            assert_eq!(events.last().unwrap()["type"], "message_stop");
        }
        drop(body);
        assert!(
            cancelled.load(Ordering::SeqCst),
            "dropping the response must cancel the backend"
        );
    }
}

#[tokio::test]
async fn malformed_backend_and_missing_done_are_errors() {
    for script in [
        vec![text("partial")],
        vec![
            TurnEvent::ToolCallStart {
                index: 0,
                id: "a".into(),
                name: "f".into(),
            },
            TurnEvent::ToolCallDelta {
                index: 0,
                arguments: "{".into(),
            },
            done(StopReason::ToolUse),
        ],
    ] {
        let (app, _) = router(vec![script.clone(), script]);
        let (status, _, _) = wire(app.clone(), request(prompt(false))).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        let (_, _, body) = wire(app, request(prompt(true))).await;
        assert_eq!(frames(&body).last().unwrap()["type"], "error");
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_SDK_PYTHON pointing to an installed official SDK"]
async fn scripted_sdk_smoke() {
    let script = vec![
        TurnEvent::ReasoningDelta {
            text: "offline reasoning".into(),
        },
        text("before"),
        TurnEvent::ToolCallStart {
            index: 0,
            id: "a".into(),
            name: "a".into(),
        },
        TurnEvent::ToolCallStart {
            index: 1,
            id: "b".into(),
            name: "b".into(),
        },
        TurnEvent::ToolCallDelta {
            index: 1,
            arguments: "{\"b\":".into(),
        },
        TurnEvent::ToolCallDelta {
            index: 0,
            arguments: "{\"a\":1}".into(),
        },
        TurnEvent::ToolCallEnd { index: 0 },
        TurnEvent::ToolCallDelta {
            index: 1,
            arguments: "2}".into(),
        },
        TurnEvent::ToolCallEnd { index: 1 },
        text("after"),
        TurnEvent::Usage {
            usage: Usage {
                input_tokens: 17,
                output_tokens: 9,
                ..Default::default()
            },
        },
        done(StopReason::ToolUse),
    ];
    let truncated = vec![
        TurnEvent::ToolCallStart {
            index: 0,
            id: "partial".into(),
            name: "f".into(),
        },
        TurnEvent::ToolCallDelta {
            index: 0,
            arguments: "{\"nested\":{\"x\":1},\"unfinished\":\"abc".into(),
        },
        TurnEvent::ToolCallEnd { index: 0 },
        done(StopReason::MaxTokens),
    ];
    let (app, _) = router(vec![script.clone(), script, truncated.clone(), truncated]);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async { axum::serve(listener, app).await.unwrap() });
    let python = std::env::var("ANTHROPIC_SDK_PYTHON").expect("set ANTHROPIC_SDK_PYTHON");
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../scripts/gateway/anthropic-sdk-smoke.py");
    let result = tokio::process::Command::new(python)
        .arg(script)
        .arg("--base-url")
        .arg(format!("http://{address}"))
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("ANTHROPIC_API_KEY")
        .output()
        .await
        .unwrap();
    server.abort();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

struct Search;
impl gateway::SearchProvider for Search {
    fn name(&self) -> &str {
        "offline"
    }
    fn search(
        &self,
        _: gateway::search::SearchQuery,
    ) -> BoxFuture<'static, Result<Vec<gateway::search::SearchHit>, GatewayError>> {
        Box::pin(async {
            Ok(vec![gateway::search::SearchHit {
                url: "https://example.com".into(),
                title: "hit".into(),
                content: "restored page text".into(),
                published: Some("2026".into()),
            }])
        })
    }
}
#[tokio::test]
async fn hosted_search_loop_and_round_trip() {
    let first = vec![
        text("searching"),
        TurnEvent::ToolCallStart {
            index: 0,
            id: "s".into(),
            name: "web_search".into(),
        },
        TurnEvent::ToolCallDelta {
            index: 0,
            arguments: "{\"query\":\"scratch query\"}".into(),
        },
        TurnEvent::ToolCallEnd { index: 0 },
        done(StopReason::ToolUse),
    ];
    let second = vec![text("answer"), done(StopReason::EndTurn)];
    let backend = Scripted::new(vec![first, second]);
    let mut models = ModelMap::single("served-model");
    models.accept_any = true;
    let app = gateway::router(Arc::new(
        Gateway::new(Arc::new(backend.clone()), models).with_search(Arc::new(Search)),
    ));
    let mut value = prompt(true);
    value["tools"] = json!([{"type":"web_search_20250305","name":"web_search"}]);
    let (status, _, body) = wire(app, request(value)).await;
    assert_eq!(status, StatusCode::OK);
    let result = fold(&frames(&body));
    assert_eq!(result["content"][1]["type"], "server_tool_use");
    assert_eq!(result["content"][2]["type"], "web_search_tool_result");
    assert_eq!(
        result["content"][1]["id"],
        result["content"][2]["tool_use_id"]
    );
    assert_eq!(result["usage"]["server_tool_use"]["web_search_requests"], 1);
    let value = json!({"model":"m","messages":[{"role":"assistant","content":result["content"]}]});
    let turn = super::request::parse(&value, false).unwrap().0;
    assert!(turn.items.iter().any(|i| matches!(i, Item::ServerToolResult { output, .. } if output["results"][0]["content"] == "restored page text")));
    assert_eq!(backend.turns().len(), 2);
}

#[test]
fn hosted_error_and_backend_signature() {
    let mut r = super::render::Renderer::new("m".into());
    r.push(TurnEvent::ReasoningDelta { text: "r".into() })
        .unwrap();
    r.push(TurnEvent::ReasoningSignature {
        signature: "original".into(),
    })
    .unwrap();
    r.push(TurnEvent::ServerToolCall {
        id: "s".into(),
        name: "web_search".into(),
        input: json!({"query":"q"}),
    })
    .unwrap();
    r.push(TurnEvent::ServerToolResult {
        id: "s".into(),
        name: "web_search".into(),
        output: json!({"error":"invalid_input"}),
    })
    .unwrap();
    r.push(done(StopReason::EndTurn)).unwrap();
    assert_eq!(r.message()["content"][0]["signature"], "original");
    assert_eq!(
        r.message()["content"][2]["content"],
        json!({"type":"web_search_tool_result_error","error_code":"invalid_tool_input"})
    );
}

#[tokio::test]
async fn models_pagination_retrieve_and_errors() {
    let backend = Scripted::default();
    let mut models = ModelMap::single("served-model");
    models.listed = vec!["claude-a".into(), "claude-b".into(), "claude-c".into()];
    models
        .aliases
        .push(("claude-*".into(), "served-model".into()));
    let app = gateway::router(Arc::new(Gateway::new(Arc::new(backend), models)));
    for (uri, ids, more) in [
        ("/v1/models?limit=2", vec!["served-model", "claude-a"], true),
        (
            "/v1/models?after_id=claude-a&limit=2",
            vec!["claude-b", "claude-c"],
            false,
        ),
        (
            "/v1/models?before_id=claude-c&limit=2",
            vec!["claude-a", "claude-b"],
            true,
        ),
    ] {
        let req = Request::get(uri).body(Body::empty()).unwrap();
        let (status, headers, body) = wire(app.clone(), req).await;
        assert_eq!(status, StatusCode::OK);
        assert!(headers.contains_key("request-id"));
        let result: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(result["has_more"], more);
        assert_eq!(
            result["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ids
        );
        assert_eq!(result["data"][0]["type"], "model");
        assert!(result["data"][0].get("max_input_tokens").is_some());
    }
    let req = Request::get("/v1/models/claude-a")
        .body(Body::empty())
        .unwrap();
    let (_, _, body) = wire(app.clone(), req).await;
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["id"],
        "claude-a"
    );
    for uri in [
        "/v1/models?limit=0",
        "/v1/models?limit=nope",
        "/v1/models?after_id=unknown",
        "/v1/models/unknown",
    ] {
        let req = Request::get(uri)
            .header("anthropic-version", "2023-06-01")
            .body(Body::empty())
            .unwrap();
        let (status, _, body) = wire(app.clone(), req).await;
        assert!(status.is_client_error());
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["type"],
            "error"
        );
    }
}
