use crate::gateway::search::{SearchHit, SearchProvider, SearchQuery};
use futures::{future::BoxFuture, SinkExt, StreamExt};
use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    Router,
};
use serde_json::{json, Value};
use tower::ServiceExt;

use super::*;
use crate::gateway::{testing::Scripted, turn::*, ModelMap};

fn app(scripts: Vec<Vec<TurnEvent>>) -> (Router, Scripted, Arc<Gateway>) {
    let backend = Scripted::new(scripts);
    let gateway = Arc::new(Gateway::new(
        Arc::new(backend.clone()),
        ModelMap {
            accept_any: true,
            ..ModelMap::single("served-model")
        },
    ));
    (crate::gateway::router(gateway.clone()), backend, gateway)
}

async fn request(app: &Router, method: &str, path: &str, body: Value) -> (StatusCode, String) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    (status, body)
}

fn events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|s| s.strip_prefix("data: "))
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
fn done() -> TurnEvent {
    TurnEvent::Done {
        stop: StopReason::EndTurn,
    }
}
fn text(s: &str) -> TurnEvent {
    TurnEvent::TextDelta { text: s.into() }
}
fn call(index: usize, id: &str, name: &str) -> TurnEvent {
    TurnEvent::ToolCallStart {
        index,
        id: id.into(),
        name: name.into(),
    }
}
fn delta(index: usize, s: &str) -> TurnEvent {
    TurnEvent::ToolCallDelta {
        index,
        arguments: s.into(),
    }
}
fn end(index: usize) -> TurnEvent {
    TurnEvent::ToolCallEnd { index }
}

#[tokio::test]
async fn text_golden_sequence_and_json_fold() {
    let script = vec![
        text("hello"),
        text(" world"),
        TurnEvent::Usage {
            usage: Usage {
                input_tokens: 7,
                output_tokens: 2,
                cached_input_tokens: 4,
                ..Default::default()
            },
        },
        done(),
    ];
    let (app, _, _) = app(vec![script.clone(), script]);
    let (status, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-5.4","input":"hi","stream":true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let ev = events(&body);
    assert_eq!(
        ev.iter()
            .map(|v| v["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    for (i, e) in ev.iter().enumerate() {
        assert_eq!(e["sequence_number"], i);
    }
    assert!(body.contains("event: response.completed\n"));
    assert!(!body.contains("[DONE]"));
    let final_response = &ev.last().unwrap()["response"];
    assert_eq!(
        final_response["output"][0]["content"][0]["text"],
        "hello world"
    );
    assert_eq!(final_response["usage"]["total_tokens"], 9);
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-5.4","input":"hi"}),
    )
    .await;
    let mut nonstream: Value = serde_json::from_str(&body).unwrap();
    nonstream["id"] = final_response["id"].clone();
    nonstream["created_at"] = final_response["created_at"].clone();
    nonstream["completed_at"] = final_response["completed_at"].clone();
    nonstream["output"][0]["id"] = final_response["output"][0]["id"].clone();
    assert_eq!(&nonstream, final_response);
}

#[tokio::test]
async fn reasoning_round_trip_without_storage() {
    let (app, backend, _) = app(vec![
        vec![
            TurnEvent::ReasoningDelta {
                text: "private trace".into(),
            },
            text("answer"),
            done(),
        ],
        vec![text("next"), done()],
    ]);
    let (_,body) = request(&app,"POST","/v1/responses",json!({"model":"gpt-5","input":"hi","stream":true,"store":false,"reasoning":{"summary":"auto","effort":"high"},"include":["reasoning.encrypted_content"]})).await;
    let ev = events(&body);
    let types: Vec<_> = ev.iter().map(|v| v["type"].as_str().unwrap()).collect();
    assert!(types.contains(&"response.reasoning_summary_part.added"));
    assert!(types.contains(&"response.reasoning_summary_text.delta"));
    assert!(types.contains(&"response.reasoning_summary_text.done"));
    assert!(types.contains(&"response.reasoning_summary_part.done"));
    let output = ev.last().unwrap()["response"]["output"].as_array().unwrap();
    assert_eq!(output[0]["summary"][0]["text"], "private trace");
    assert!(output[0]["encrypted_content"]
        .as_str()
        .unwrap()
        .starts_with("cuteafd.v1."));
    let id = ev.last().unwrap()["response"]["id"].as_str().unwrap();
    assert_eq!(
        request(&app, "GET", &format!("/v1/responses/{id}"), Value::Null)
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    let (status, _) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-5","input":output,"store":false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        matches!(&backend.turns()[1].items[0],Item::Reasoning { text,.. } if text == "private trace")
    );
}

#[tokio::test]
async fn function_parallel_custom_and_local_shell_events() {
    let script = vec![
        call(0, "a", "f"),
        call(1, "b", "g"),
        delta(1, "{\"x\":2}"),
        delta(0, "{}"),
        end(1),
        end(0),
        done(),
    ];
    let (app, _, _) = app(vec![
        script,
        vec![
            call(0, "c", "apply_patch"),
            delta(0, "{\"input\":\"line"),
            delta(0, "\\n\\uD83D"),
            delta(0, "\\uDE00\"}"),
            end(0),
            done(),
        ],
        vec![
            call(0, "s", "local_shell"),
            delta(0, "{\"type\":\"exec\",\"command\":[\"ls\"]}"),
            end(0),
            done(),
        ],
    ]);
    let (_,body) = request(&app,"POST","/v1/responses",json!({"model":"gpt-5","input":"go","stream":true,"tools":[{"type":"function","name":"f"},{"type":"function","name":"g"}]})).await;
    let ev = events(&body);
    let output = &ev.last().unwrap()["response"]["output"];
    assert_eq!(output[0]["call_id"], "a");
    assert_eq!(output[1]["call_id"], "b");
    let delta = ev
        .iter()
        .find(|v| v["type"] == "response.function_call_arguments.delta")
        .unwrap();
    assert_eq!(delta["output_index"], 1);
    assert_eq!(delta["item_id"], output[1]["id"]);
    let (_,body) = request(&app,"POST","/v1/responses",json!({"model":"gpt-5","input":"patch","stream":true,"tools":[{"type":"custom","name":"apply_patch","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}}]})).await;
    let ev = events(&body);
    assert_eq!(ev.last().unwrap()["type"], "response.completed");
    let input = ev
        .iter()
        .filter(|e| e["type"] == "response.custom_tool_call_input.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect::<String>();
    assert_eq!(input, "line\n\u{1f600}");
    assert_eq!(ev.last().unwrap()["response"]["output"][0]["input"], input);
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-5","tools":[{"type":"local_shell"}]}),
    )
    .await;
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["output"][0]["action"]["command"],
        json!(["ls"])
    );
}

#[tokio::test]
async fn stored_chain_routes_and_input_pagination() {
    let (app, backend, _) = app(vec![
        vec![text("first"), done()],
        vec![text("second"), done()],
    ]);
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-5","input":"one","instructions":"old"}),
    )
    .await;
    let v: Value = serde_json::from_str(&body).unwrap();
    let id = v["id"].as_str().unwrap();
    let (status, got) = request(&app, "GET", &format!("/v1/responses/{id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(serde_json::from_str::<Value>(&got).unwrap(), v);
    let (_, items) = request(
        &app,
        "GET",
        &format!("/v1/responses/{id}/input_items?order=asc&limit=1"),
        Value::Null,
    )
    .await;
    let items: Value = serde_json::from_str(&items).unwrap();
    assert_eq!(items["data"][0]["content"][0]["text"], "one");
    assert_eq!(items["has_more"], false);
    let (status, _) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-5","input":"two","previous_response_id":id}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(backend.turns()[1].items.len(), 3);
    assert_eq!(
        backend.turns()[1].system,
        None,
        "previous response instructions must not carry over"
    );
    assert_eq!(
        request(
            &app,
            "POST",
            &format!("/v1/responses/{id}/cancel"),
            json!({})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        request(&app, "DELETE", &format!("/v1/responses/{id}"), Value::Null)
            .await
            .0,
        StatusCode::OK
    );
    assert_eq!(
        request(&app, "GET", &format!("/v1/responses/{id}"), Value::Null)
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        request(
            &app,
            "POST",
            "/v1/responses",
            json!({"model":"gpt-5","previous_response_id":"unknown"})
        )
        .await
        .0,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn incomplete_and_failures() {
    let (app, backend, _) = app(vec![
        vec![
            text("partial"),
            TurnEvent::Done {
                stop: StopReason::MaxTokens,
            },
        ],
        vec![],
    ]);
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-5","stream":true,"max_output_tokens":1}),
    )
    .await;
    let ev = events(&body);
    assert_eq!(ev.last().unwrap()["type"], "response.incomplete");
    assert_eq!(
        ev.last().unwrap()["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    backend.scripts.lock().unwrap()[0] =
        vec![Ok(text("hello")), Err(GatewayError::upstream("broken"))];
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-5","stream":true}),
    )
    .await;
    let ev = events(&body);
    assert_eq!(ev[ev.len() - 2]["type"], "error");
    assert_eq!(ev.last().unwrap()["type"], "response.failed");
    assert_eq!(
        request(
            &app,
            "POST",
            "/v1/responses",
            json!({"model":"gpt-5","stream":true})
        )
        .await
        .0,
        StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(
        request(
            &app,
            "POST",
            "/v1/responses",
            json!({"model":"gpt-5","background":true})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn parsing_every_input_item_and_controls() {
    let (app, backend, gateway) = app(vec![vec![done()]]);
    let input = json!([
        {"role":"developer","content":"rules"},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"a"},{"type":"input_image","image_url":"data:image/png;base64,aGk=","detail":"high"},{"type":"input_file","filename":"a.txt","file_data":"hi"}]},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"},{"type":"refusal","refusal":"no"}]},
        {"type":"function_call","call_id":"f","name":"f","arguments":"{}"},
        {"type":"function_call_output","call_id":"f","output":"done"},
        {"type":"custom_tool_call","call_id":"c","name":"patch","input":"patch"},
        {"type":"custom_tool_call_output","call_id":"c","output":[{"type":"input_text","text":"ok"}]},
        {"type":"local_shell_call","call_id":"s","action":{"type":"exec","command":["ls"]}},
        {"type":"local_shell_call_output","call_id":"s","output":"files"},
        {"type":"reasoning","summary":[{"type":"summary_text","text":"thought"}]},
        {"type":"web_search_call","id":"w","action":{"type":"search","query":"q"}},
        {"type":"compaction","encrypted_content":parse::encode("compaction","summary")}
    ]);
    let (status,body) = request(&app,"POST","/v1/responses",json!({"model":"gpt-5","input":input,"temperature":0.2,"top_p":0.8,"parallel_tool_calls":false,"reasoning":{"effort":"low"},"text":{"format":{"type":"json_schema","name":"answer","schema":{"type":"object"},"strict":true},"verbosity":"low"},"prompt_cache_key":"cache","service_tier":"default","unknown":"ignored"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let turn = &backend.turns()[0];
    assert_eq!(turn.items.len(), 12);
    assert_eq!(turn.prompt_cache_key.as_deref(), Some("cache"));
    assert_eq!(turn.text_verbosity.as_deref(), Some("low"));
    assert_eq!(turn.reasoning.effort.as_deref(), Some("low"));
    assert_eq!(turn.response_format.as_ref().unwrap()["name"], "answer");
    let v: Value = serde_json::from_str(&body).unwrap();
    let p = parse::parse(&gateway,json!({"model":"gpt-5","previous_response_id":v["id"],"input":[{"type":"item_reference","id":input[10]["id"]}]}),None).unwrap();
    assert!(matches!(p.new_items[0], Item::ServerToolCall { .. }));
    for bad in [
        json!({"type":"input_image","file_id":"f"}),
        json!({"type":"input_file","file_url":"https://x"}),
    ] {
        assert!(parse::parse_item(&json!({"role":"user","content":[bad]})).is_err());
    }
    assert!(parse::parse_item(&json!({"type":"reasoning","encrypted_content":"foreign"})).is_err());
    assert!(parse::parse(
        &gateway,
        json!({"model":"gpt-5","tools":[{"type":"file_search"}]}),
        None
    )
    .is_err());
}

#[tokio::test]
async fn count_and_compaction_roundtrip() {
    let (app, backend, _) = app(vec![
        vec![text("Keep the outstanding task"), done()],
        vec![done()],
    ]);
    let (status, body) = request(
        &app,
        "POST",
        "/v1/responses/input_tokens",
        json!({"model":"gpt-5","input":"one two"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        serde_json::from_str::<Value>(&body).unwrap()["input_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    let (status, body) = request(
        &app,
        "POST",
        "/v1/responses/compact",
        json!({"model":"gpt-5","input":"task","tools":[{"type":"function","name":"f","parameters":{}}],"tool_choice":{"type":"function","name":"f"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "response.compaction");
    assert_eq!(
        backend.turns()[0].tool_choice,
        crate::gateway::turn::ToolChoice::None
    );
    let (status, _) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-5","input":v["output"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        matches!(&backend.turns()[1].items[0],Item::Message { role:Role::System,content } if content == &vec![Part::text("Keep the outstanding task")])
    );
}

struct Search;
impl SearchProvider for Search {
    fn name(&self) -> &str {
        "offline"
    }
    fn search(
        &self,
        query: SearchQuery,
    ) -> BoxFuture<'static, Result<Vec<SearchHit>, GatewayError>> {
        assert_eq!(query.allowed_domains, vec!["example.org"]);
        Box::pin(async {
            Ok(vec![SearchHit {
                url: "https://example.org/source".into(),
                title: "Source".into(),
                content: "fact".into(),
                published: None,
            }])
        })
    }
}

#[tokio::test]
async fn hosted_search_events_and_stored_result() {
    let backend = Scripted::new(vec![
        vec![
            call(0, "search", "web_search"),
            delta(0, r#"{"query":"question"}"#),
            end(0),
            TurnEvent::Done {
                stop: StopReason::ToolUse,
            },
        ],
        vec![text("fact"), done()],
        vec![done()],
    ]);
    let gateway = Arc::new(
        Gateway::new(
            Arc::new(backend.clone()),
            ModelMap {
                accept_any: true,
                ..ModelMap::single("served-model")
            },
        )
        .with_search(Arc::new(Search)),
    );
    let app = crate::gateway::router(gateway);
    let (_,body) = request(&app,"POST","/v1/responses",json!({"model":"gpt-6.1-sol","input":"q","stream":true,"tools":[{"type":"web_search","filters":{"allowed_domains":["example.org"]}}]})).await;
    let ev = events(&body);
    let types: Vec<_> = ev.iter().map(|v| v["type"].as_str().unwrap()).collect();
    for t in [
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
    ] {
        assert!(types.contains(&t), "{types:?}");
    }
    let response = &ev.last().unwrap()["response"];
    assert_eq!(
        response["output"][0]["action"]["sources"][0]["url"],
        "https://example.org/source"
    );
    request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-6.1-sol","previous_response_id":response["id"],"input":[{"type":"item_reference","id":response["output"][0]["id"]}]}),
    )
    .await;
    assert_eq!(backend.turns()[2].items.iter().filter(|i| matches!(i,Item::ServerToolResult { output,.. } if output["results"][0]["content"] == "fact")).count(),2);
}

#[tokio::test]
async fn namespace_tools_and_pause_turn() {
    let (app, backend, _) = app(vec![vec![
        call(0, "p", "functions.apply_patch"),
        delta(0, r#"{"input":"patch"}"#),
        end(0),
        TurnEvent::Done {
            stop: StopReason::PauseTurn,
        },
    ]]);
    let (status,body) = request(&app,"POST","/v1/responses",json!({"model":"gpt-6.1-sol","tools":[{"type":"namespace","name":"functions","description":"","tools":[{"type":"custom","name":"apply_patch","format":{"type":"text"}}]}],"tool_choice":{"type":"custom","namespace":"functions","name":"apply_patch"}})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "completed");
    assert_eq!(v["output"][0]["namespace"], "functions");
    assert_eq!(v["output"][0]["name"], "apply_patch");
    assert_eq!(backend.turns()[0].tools[0].name, "functions.apply_patch");
    let items = parse::parse_item(&v["output"][0]).unwrap();
    assert!(matches!(&items[0],Item::ToolCall { name,.. } if name == "functions.apply_patch"));
}

#[tokio::test]
async fn websocket_warmup_and_ephemeral_previous_response() {
    let (app, backend, _) = app(vec![
        vec![text("first"), done()],
        vec![text("second"), done()],
    ]);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/responses"))
        .await
        .unwrap();
    for (i, input) in ["warm", "one", "two"].iter().enumerate() {
        let mut req =
            json!({"type":"response.create","model":"gpt-6.1-sol","input":input,"store":false});
        if i == 0 {
            req["generate"] = json!(false);
        }
        if i == 2 {
            req["previous_response_id"] = json!(LAST_ID.with(|s| s.borrow().clone()));
        }
        ws.send(tokio_tungstenite::tungstenite::Message::Text(
            req.to_string(),
        ))
        .await
        .unwrap();
        loop {
            let message = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let v: Value = serde_json::from_str(message.to_text().unwrap()).unwrap();
            assert_ne!(v["type"], "error", "{v}");
            if v["type"] == "response.completed" {
                if i == 0 {
                    assert_eq!(v["response"]["output"], json!([]));
                    assert!(backend.turns().is_empty());
                }
                LAST_ID.with(|s| *s.borrow_mut() = v["response"]["id"].as_str().unwrap().into());
                break;
            }
        }
    }
    assert_eq!(backend.turns().len(), 2);
    assert_eq!(backend.turns()[1].items.len(), 3);
    ws.close(None).await.unwrap();
    server.abort();
}
thread_local! { static LAST_ID:std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) }; }

#[test]
fn response_document_eviction_is_atomic() {
    let sessions = crate::gateway::SessionStore::new(1);
    let snap = || {
        Arc::new(Snapshot {
            root_response_id: "root".into(),
            parent: None,
            system: None,
            items: vec![],
            additional_tools: vec![],
        })
    };
    sessions.put_response_document("a".into(), snap(), json!({"id":"a"}));
    sessions.put_response_document("b".into(), snap(), json!({"id":"b"}));
    assert!(sessions.response("a").is_none());
    assert!(sessions.response_document("a").is_none());
    assert!(sessions.response("b").is_some());
    assert!(sessions.response_document("b").is_some());
    assert!(sessions.delete_response("b"));
    assert!(sessions.response_document("b").is_none());
}

#[tokio::test]
async fn truncated_calls_content_filter_and_typed_error() {
    let (app, backend, _) = app(vec![
        vec![
            call(0, "c", "patch"),
            delta(0, r#"{"input":"partial"#),
            end(0),
            TurnEvent::Done {
                stop: StopReason::MaxTokens,
            },
        ],
        vec![
            call(0, "s", "local_shell"),
            delta(0, r#"{"type":"exec","command":["#),
            end(0),
            TurnEvent::Done {
                stop: StopReason::MaxTokens,
            },
        ],
        vec![
            text("filtered"),
            TurnEvent::Done {
                stop: StopReason::ContentFilter,
            },
        ],
        vec![],
        vec![],
    ]);
    for tool in [
        json!({"type":"custom","name":"patch"}),
        json!({"type":"local_shell"}),
    ] {
        let (_, body) = request(
            &app,
            "POST",
            "/v1/responses",
            json!({"model":"gpt-6.1-sol","stream":true,"tools":[tool]}),
        )
        .await;
        let ev = events(&body);
        assert_eq!(ev.last().unwrap()["type"], "response.incomplete", "{body}");
        assert_eq!(
            ev.last().unwrap()["response"]["output"][0]["status"],
            "incomplete"
        );
    }
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-6.1-sol"}),
    )
    .await;
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["incomplete_details"]["reason"],
        "content_filter"
    );
    backend.scripts.lock().unwrap()[1] = vec![Err(GatewayError::new(
        crate::gateway::ErrorKind::RateLimited,
        "retry later",
    )
    .with_param("model"))];
    let (status, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-6.1-sol"}),
    )
    .await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["param"],
        "model"
    );
    backend.scripts.lock().unwrap()[0] = vec![Err(GatewayError::upstream("failed"))];
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-6.1-sol","stream":true}),
    )
    .await;
    let ev = events(&body);
    let id = ev.last().unwrap()["response"]["id"].as_str().unwrap();
    let (status, body) = request(&app, "GET", &format!("/v1/responses/{id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["status"],
        "failed"
    );
}

#[tokio::test]
#[ignore = "manual localhost fixture for scripts/gateway/check-responses-sdk.py"]
async fn serve_sdk_fixture() {
    let (app, _, _) = app(vec![
        vec![text("hello"), text(" world"), done()],
        vec![
            TurnEvent::ReasoningDelta {
                text: "trace".into(),
            },
            text("answer"),
            done(),
        ],
        vec![
            call(0, "call_function", "f"),
            delta(0, r#"{"x":1}"#),
            end(0),
            done(),
        ],
        vec![
            call(0, "call_custom", "patch"),
            delta(0, r#"{"input":"patch\ntext"}"#),
            end(0),
            done(),
        ],
        vec![
            text("partial"),
            TurnEvent::Done {
                stop: StopReason::MaxTokens,
            },
        ],
    ]);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:18491")
        .await
        .unwrap();
    println!("SDK fixture ready on http://127.0.0.1:18491/v1");
    axum::serve(listener, app).await.unwrap();
}

#[derive(Clone)]
struct PendingStart {
    pending_count: bool,
    entered: Arc<tokio::sync::Notify>,
    dropped: Arc<tokio::sync::Notify>,
}
struct SignalOnDrop(Arc<tokio::sync::Notify>);
impl Drop for SignalOnDrop {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}
impl crate::gateway::Backend for PendingStart {
    fn name(&self) -> &str {
        "pending"
    }
    fn capabilities(&self) -> crate::gateway::BackendCapabilities {
        Default::default()
    }
    fn models(&self) -> Vec<crate::gateway::ModelInfo> {
        vec![]
    }
    fn start(
        &self,
        _: crate::gateway::TurnRequest,
    ) -> BoxFuture<'static, Result<crate::gateway::TurnStream, GatewayError>> {
        let me = self.clone();
        Box::pin(async move {
            let _guard = SignalOnDrop(me.dropped);
            me.entered.notify_one();
            futures::future::pending().await
        })
    }
    fn count_tokens(
        &self,
        _: crate::gateway::TurnRequest,
    ) -> BoxFuture<'static, Result<u32, GatewayError>> {
        let me = self.clone();
        Box::pin(async move {
            if me.pending_count {
                let _guard = SignalOnDrop(me.dropped);
                me.entered.notify_one();
                futures::future::pending().await
            } else {
                Ok(0)
            }
        })
    }
}
#[tokio::test]
async fn websocket_disconnect_cancels_pending_backend_start() {
    pending_websocket_disconnect(false).await;
}

#[tokio::test]
async fn websocket_disconnect_cancels_pending_token_count() {
    pending_websocket_disconnect(true).await;
}

async fn pending_websocket_disconnect(pending_count: bool) {
    let backend = PendingStart {
        pending_count,
        entered: Arc::new(tokio::sync::Notify::new()),
        dropped: Arc::new(tokio::sync::Notify::new()),
    };
    let gateway = Arc::new(Gateway::new(
        Arc::new(backend.clone()),
        ModelMap::single("test"),
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, crate::gateway::router(gateway))
            .await
            .unwrap();
    });
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/responses"))
        .await
        .unwrap();
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        json!({"type":"response.create","model":"test","input":"hi","generate":!pending_count})
            .to_string(),
    ))
    .await
    .unwrap();
    tokio::time::timeout(
        std::time::Duration::from_secs(3),
        backend.entered.notified(),
    )
    .await
    .unwrap();
    ws.close(None).await.unwrap();
    tokio::time::timeout(
        std::time::Duration::from_secs(3),
        backend.dropped.notified(),
    )
    .await
    .unwrap();
    server.abort();
}

#[tokio::test]
async fn open_function_cutoff_and_ordinary_refusal() {
    let (app, _, _) = app(vec![
        vec![
            call(0, "c", "f"),
            delta(0, r#"{"x":1}"#),
            TurnEvent::Usage {
                usage: Default::default(),
            },
            TurnEvent::Done {
                stop: StopReason::MaxTokens,
            },
        ],
        vec![
            text("I cannot do that"),
            TurnEvent::Done {
                stop: StopReason::Refusal,
            },
        ],
    ]);
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-6.1-sol","stream":true}),
    )
    .await;
    let ev = events(&body);
    assert_eq!(ev.last().unwrap()["type"], "response.incomplete");
    assert_eq!(
        ev.last().unwrap()["response"]["output"][0]["status"],
        "incomplete"
    );
    assert!(!ev
        .iter()
        .any(|v| v["type"] == "response.function_call_arguments.done"));
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-6.1-sol"}),
    )
    .await;
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["status"],
        "completed"
    );
}

#[tokio::test]
async fn codex_catalog_and_lite_additional_tools() {
    let backend = Scripted::new(vec![vec![done()]]);
    let gateway = Arc::new(Gateway::new(
        Arc::new(backend.clone()),
        ModelMap::official_names("served-model"),
    ));
    let app = crate::gateway::router(gateway.clone());
    let (status, body) = request(
        &app,
        "GET",
        "/v1/codex/models.json?client_version=0.161.0&deployment=ignored",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.len() < 800 * 1024, "default catalog exceeds 800 KiB");
    let catalog: Value = serde_json::from_str(&body).unwrap();
    let entry = catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["slug"] == "gpt-6.1-sol")
        .unwrap();
    let sample: Value = serde_json::from_str(include_str!("codex-model-template.json")).unwrap();
    for key in [
        "shell_type",
        "apply_patch_tool_type",
        "supported_reasoning_levels",
        "default_reasoning_level",
        "supports_reasoning_summary_parameter",
        "use_responses_lite",
        "tool_mode",
        "experimental_supported_tools",
    ] {
        assert_eq!(entry[key], sample[key], "{key}");
    }
    assert_eq!(entry["context_window"], 131072);
    assert_eq!(entry["max_output_tokens"], 32768);
    assert_eq!(entry["auto_compact_token_limit"], 98304);
    assert_eq!(entry["effective_context_window_percent"], 75);
    assert!(!catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v["slug"].as_str().unwrap().starts_with("claude-")));
    let (status,body)=request(&app,"POST","/v1/responses",json!({"model":"gpt-6.1-sol","reasoning":{"effort":"low"},"input":[{"type":"additional_tools","id":"at_tools","role":"developer","tools":[{"type":"namespace","name":"functions","description":"","tools":[{"type":"custom","name":"apply_patch","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}}]}]},{"type":"configuration_update","reasoning":{"effort":"high"}},{"role":"developer","content":"rules"},{"role":"user","content":"task"}]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let turn = &backend.turns()[0];
    assert_eq!(turn.tools[0].name, "functions.apply_patch");
    assert_eq!(turn.reasoning.effort.as_deref(), Some("high"));
    assert_eq!(turn.items.len(), 2);
}

#[tokio::test]
async fn configuration_update_effort_without_top_level_reasoning() {
    let (app, backend, _) = app(vec![vec![done()], vec![done()]]);
    for (effort, enabled) in [("high", true), ("none", false)] {
        let (status, body) = request(
            &app,
            "POST",
            "/v1/responses",
            json!({"model":"gpt-6.1-sol","input":[{"type":"configuration_update","reasoning":{"effort":effort}},{"role":"user","content":"hi"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let turns = backend.turns();
        let turn = turns.last().unwrap();
        assert_eq!(turn.reasoning.effort.as_deref(), Some(effort));
        assert_eq!(turn.reasoning.enabled, Some(enabled));
        assert_eq!(turn.items.len(), 1);
    }
}

#[tokio::test]
async fn configuration_update_valid_effort_overrides_top_level_reasoning() {
    let (app, backend, _) = app(vec![vec![done()]]);
    let (status, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-6.1-sol","reasoning":{"effort":"none"},"input":[{"type":"configuration_update","reasoning":{"effort":"high"}}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(backend.turns()[0].reasoning.effort.as_deref(), Some("high"));
    assert_eq!(backend.turns()[0].reasoning.enabled, Some(true));
}

#[tokio::test]
async fn configuration_update_invalid_effort_without_top_level_reasoning() {
    let (app, backend, _) = app(vec![]);
    let (status, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-6.1-sol","input":[{"type":"configuration_update","reasoning":{"effort":"invalid"}}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let error: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(error["error"]["param"], "reasoning.effort");
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert!(backend.turns().is_empty());
}

#[tokio::test]
async fn failed_response_marks_open_calls_incomplete() {
    let (app, backend, _) = app(vec![vec![]]);
    backend.scripts.lock().unwrap()[0] = vec![
        Ok(call(0, "c", "f")),
        Ok(delta(0, "{")),
        Err(GatewayError::upstream("interrupted")),
    ];
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-6.1-sol","stream":true}),
    )
    .await;
    let ev = events(&body);
    assert_eq!(
        ev.last().unwrap()["response"]["output"][0]["status"],
        "incomplete"
    );
    assert!(!ev
        .iter()
        .any(|v| v["type"] == "response.function_call_arguments.done"));
}

#[tokio::test]
async fn generated_input_id_reference_and_root_lineage() {
    let (app, backend, gateway) = app(vec![vec![text("first"), done()], vec![done()]]);
    let (_, body) = request(
        &app,
        "POST",
        "/v1/responses",
        json!({"model":"gpt-6.1-sol","input":"hello"}),
    )
    .await;
    let response: Value = serde_json::from_str(&body).unwrap();
    let id = response["id"].as_str().unwrap();
    let (_, body) = request(
        &app,
        "GET",
        &format!("/v1/responses/{id}/input_items"),
        Value::Null,
    )
    .await;
    let listing: Value = serde_json::from_str(&body).unwrap();
    let (status,body)=request(&app,"POST","/v1/responses",json!({"model":"gpt-6.1-sol","previous_response_id":id,"input":[{"type":"item_reference","id":listing["data"][0]["id"]}]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(backend.turns()[1].items.len(), 3);
    let next: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        gateway
            .sessions
            .response(next["id"].as_str().unwrap())
            .unwrap()
            .root_response_id,
        id
    );
}

fn reasoning_of(item: Value) -> String {
    match parse::parse_item(&item).unwrap().as_slice() {
        [Item::Reasoning { text, signature: None }] => text.clone(),
        other => panic!("expected one reasoning item, got {other:?}"),
    }
}

fn summary(text: &str) -> Value {
    json!([{"type":"summary_text","text":text}])
}

#[tokio::test]
async fn reasoning_unedited_round_trip_keeps_token_text() {
    for summary_mode in [json!("auto"), Value::Null] {
        let (app, _, _) = app(vec![vec![
            TurnEvent::ReasoningDelta { text: "full trace".into() },
            text("answer"),
            done(),
        ]]);
        let (_, body) = request(&app,"POST","/v1/responses",json!({"model":"gpt-5","input":"hi","store":false,"reasoning":{"summary":summary_mode},"include":["reasoning.encrypted_content"]})).await;
        let item = serde_json::from_str::<Value>(&body).unwrap()["output"][0].clone();
        assert_eq!(item["type"], "reasoning");
        assert_eq!(reasoning_of(item), "full trace", "summary={summary_mode}");
    }
    // A summary shorter than the trace still recovers the full trace.
    let token = parse::encode_reasoning("full trace", "short");
    assert_eq!(
        reasoning_of(json!({"type":"reasoning","summary":summary("short"),"encrypted_content":token})),
        "full trace"
    );
}

#[test]
fn reasoning_edited_summary_wins_over_token() {
    let token = parse::encode_reasoning("full trace", "full trace");
    assert_eq!(
        reasoning_of(json!({"type":"reasoning","summary":summary("edited"),"encrypted_content":token})),
        "edited"
    );
    // Visible text added where none was shown is an edit too.
    let hidden = parse::encode_reasoning("full trace", "");
    assert_eq!(
        reasoning_of(json!({"type":"reasoning","content":[{"type":"reasoning_text","text":"added"}],"encrypted_content":hidden})),
        "added"
    );
}

#[test]
fn reasoning_without_visible_text_uses_token() {
    let token = parse::encode_reasoning("full trace", "full trace");
    for item in [
        json!({"type":"reasoning","encrypted_content":token}),
        json!({"type":"reasoning","summary":[],"encrypted_content":token}),
    ] {
        assert_eq!(reasoning_of(item), "full trace");
    }
}

#[test]
fn reasoning_old_token_keeps_token_text() {
    let old = parse::encode("reasoning", "old trace");
    assert_eq!(
        reasoning_of(json!({"type":"reasoning","summary":summary("edited"),"encrypted_content":old})),
        "old trace"
    );
    assert_eq!(
        reasoning_of(json!({"type":"reasoning","encrypted_content":old})),
        "old trace"
    );
}
#[tokio::test]
async fn usage_rows_messages_responses_and_root_chain() {
    use crate::usage::{tests::Sink, Middleware, track, session_hash};
    let u = Usage { input_tokens: 12, cached_input_tokens: 4, output_tokens: 5, reasoning_tokens: 2, ..Default::default() };
    let script = || vec![text("answer"), TurnEvent::Usage {usage:u}, done()];
    let (router, backend, _) = app(vec![script(),script(),script(),script(),script()]);
    let sink = Arc::new(Sink::default());
    let router = router.layer(axum::middleware::from_fn_with_state(Middleware::new(sink.clone()), track));
    let (status, body) = request(&router,"POST","/v1/responses",json!({"model":"alias","input":"hello"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let root: Value = serde_json::from_str(&body).unwrap();
    let (status, _) = request(&router,"POST","/v1/responses",json!({"model":"alias","input":"next","previous_response_id":root["id"]})).await;
    assert_eq!(status,StatusCode::OK);
    let (status, _) = request(&router,"POST","/v1/responses",json!({"model":"alias","input":"hello","store":false,"stream":true,"prompt_cache_key":"codex-secret"})).await;
    assert_eq!(status,StatusCode::OK);
    for stream in [false,true] {
        let (status, body) = request(&router,"POST","/v1/messages",json!({"model":"alias","max_tokens":10,"stream":stream,"metadata":{"user_id":"claude-secret"},"messages":[{"role":"user","content":"hello"}]})).await;
        assert_eq!(status,StatusCode::OK,"{body}");
    }
    // Scripted retains requests for assertions; release those engine references.
    backend.seen.lock().unwrap().clear();
    let rows = sink.0.lock().unwrap();
    assert_eq!(rows.len(),5);
    assert_eq!(rows[0].session_id.as_deref(),root["id"].as_str());
    assert_eq!(rows[1].session_id,rows[0].session_id);
    assert_eq!(rows[1].session_source.as_deref(),Some("explicit"));
    assert_eq!(rows[2].session_id,Some(session_hash("codex-secret")));
    assert_eq!(rows[2].session_source.as_deref(),Some("cache_key"));
    for row in &rows[3..] {
        assert_eq!(row.protocol,"messages");
        assert_eq!(row.session_id,Some(session_hash("claude-secret")));
    }
    for row in rows.iter() {
        assert_eq!(row.model_requested.as_deref(),Some("alias"));
        assert_eq!(row.model_served.as_deref(),Some("served-model"));
        assert_eq!(row.tokens_in,Some(12));
        assert_eq!(row.tokens_out,Some(5));
        assert_eq!(row.tokens_reasoning,Some(2));
        assert_eq!(row.outcome,"ok");
        assert_eq!(row.stop_reason.as_deref(),Some("end_turn"));
    }
}

/// A benchmark that starts while a Responses socket is open refuses that
/// socket's next turn with a busy error, and admits it again afterwards.
#[tokio::test]
async fn turn_gate_refuses_turns_on_an_open_socket() {
    let backend = Scripted::new(vec![vec![text("after"), done()]]);
    let locked = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = locked.clone();
    let mut gateway = Gateway::new(Arc::new(backend.clone()), ModelMap { accept_any: true, ..ModelMap::single("served-model") });
    gateway.gate = Some(Arc::new(move || flag.load(std::sync::atomic::Ordering::SeqCst).then_some(30)));
    let app = crate::gateway::router(Arc::new(gateway));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/responses")).await.unwrap();
    for (lock, expect) in [(true, "error"), (false, "response.completed")] {
        locked.store(lock, std::sync::atomic::Ordering::SeqCst);
        ws.send(tokio_tungstenite::tungstenite::Message::Text(
            json!({"type":"response.create","model":"gpt-6.1-sol","input":"hi","store":false}).to_string())).await.unwrap();
        loop {
            let message = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next()).await.unwrap().unwrap().unwrap();
            let v: Value = serde_json::from_str(message.to_text().unwrap()).unwrap();
            if v["type"] == "error" || v["type"] == "response.completed" || v["type"] == "response.failed" {
                assert_eq!(v["type"], expect, "{v}");
                if lock { assert!(v.to_string().contains("benchmark"), "{v}"); }
                break;
            }
        }
    }
    assert_eq!(backend.turns().len(), 1, "the refused turn never reached the backend");
    ws.close(None).await.unwrap();
    server.abort();
}

#[tokio::test]
async fn turn_gate_refuses_compact() {
    let backend = Scripted::new(vec![]);
    let mut gateway = Gateway::new(Arc::new(backend.clone()), ModelMap { accept_any: true, ..ModelMap::single("served-model") });
    gateway.gate = Some(Arc::new(|| Some(30)));
    let app = crate::gateway::router(Arc::new(gateway));
    let (status, text) = request(&app, "POST", "/v1/responses/compact", json!({"model":"gpt-6.1-sol","input":"hi"})).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{text}");
    assert!(text.contains("benchmark"), "{text}");
    assert!(backend.turns().is_empty());
}

/// Codex over WebSocket declares its tools in an `additional_tools` input item
/// on the first turn only; continuations by `previous_response_id` keep them
/// (live Codex 0.161 run: the second turn had no tools, so the model could
/// only describe the edit it meant to make).
#[tokio::test]
async fn websocket_continuation_keeps_additional_tools() {
    let (app, backend, _) = app(vec![vec![text("first"), done()], vec![text("second"), done()]]);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/responses")).await.unwrap();
    let tools = json!({"type":"additional_tools","role":"developer","tools":[{"type":"namespace","name":"functions","description":"",
        "tools":[{"type":"custom","name":"exec","description":"Run a command."}]}]});
    let mut last = String::new();
    for i in 0..2 {
        let mut req = json!({"type":"response.create","model":"gpt-6.1-sol","store":false,
            "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":format!("turn {i}")}]}]});
        if i == 0 { req["input"].as_array_mut().unwrap().insert(0, tools.clone()); }
        else { req["previous_response_id"] = json!(last); }
        ws.send(tokio_tungstenite::tungstenite::Message::Text(req.to_string())).await.unwrap();
        loop {
            let message = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next()).await.unwrap().unwrap().unwrap();
            let v: Value = serde_json::from_str(message.to_text().unwrap()).unwrap();
            assert_ne!(v["type"], "error", "{v}");
            if v["type"] == "response.completed" { last = v["response"]["id"].as_str().unwrap().into(); break; }
        }
    }
    let turns = backend.turns();
    assert_eq!(turns.len(), 2);
    for turn in &turns {
        assert_eq!(turn.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), vec!["functions.exec"], "tools on every turn");
    }
    ws.close(None).await.unwrap();

/// Full-log capture through the Messages and Responses front ends (HTTP and WebSocket).
#[tokio::test]
async fn full_log_captures_messages_responses_and_websocket_turns() {
    use crate::usage::{tests::logging_sink, Middleware, track};
    use futures::SinkExt;
    let script = || vec![text("answer"), done()];
    let (router, backend, _) = app(vec![script(), script(), script(), script()]);
    let (sink, tape) = logging_sink();
    let router = router.layer(axum::middleware::from_fn_with_state(Middleware::new(sink.clone()), track));
    let (status, _) = request(&router, "POST", "/v1/messages", json!({"model":"alias","max_tokens":10,"stream":true,
        "messages":[{"role":"user","content":"hello"}]})).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = request(&router, "POST", "/v1/responses", json!({"model":"alias","input":"hi","stream":true})).await;
    assert_eq!(status, StatusCode::OK);
    // Scripted retains turns (and their usage handles); the record emits when they drop.
    backend.seen.lock().unwrap().clear();
    let n = tape.0.lock().unwrap().len();
    assert_eq!(n, 2);
    assert_eq!(tape.request(0)["messages"][0]["content"], "hello");
    assert!(tape.response_text(0).contains("message_start") && tape.response_text(0).contains("answer"));
    assert!(tape.response_text(1).contains("response.completed"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap(); });
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/responses")).await.unwrap();
    for input in ["one", "two"] {
        ws.send(tokio_tungstenite::tungstenite::Message::Text(
            json!({"type":"response.create","model":"alias","input":input,"store":false}).to_string())).await.unwrap();
        loop {
            let m = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next()).await.unwrap().unwrap().unwrap();
            if m.to_text().unwrap().contains("\"response.completed\"") { break; }
        }
    }
    ws.close(None).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            backend.seen.lock().unwrap().clear();
            if tape.0.lock().unwrap().len() >= 4 { break; }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }).await.unwrap();
    let logs = tape.0.lock().unwrap().clone();
    let turns = logs.iter().filter(|l| l.meta.protocol == "responses" && l.meta.route == "/v1/responses" && matches!(l.response, crate::usage_log::ResponsePayload::Object(_))).count();
    assert_eq!(turns, 2, "one log record per WebSocket turn");
    for (i, l) in logs.iter().enumerate().skip(2) {
        assert_eq!(tape.request(i)["type"], "response.create");
        let crate::usage_log::ResponsePayload::Object(b) = &l.response else { panic!("turn response") };
        assert_eq!(serde_json::from_slice::<Value>(b).unwrap()["status"], "completed");
    }
    server.abort();
}
