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
        json!({"model":"gpt-5","input":"task"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "response.compaction");
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
