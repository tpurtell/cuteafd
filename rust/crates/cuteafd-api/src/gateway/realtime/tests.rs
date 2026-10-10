use super::super::{
    backend::{Backend, BackendCapabilities, ModelInfo, TurnStream},
    models::ModelMap,
    testing::Scripted,
    turn::Usage,
};
use super::*;
use futures::{future::BoxFuture, SinkExt, StreamExt};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, Message as ClientMessage},
    MaybeTlsStream, WebSocketStream,
};
type Client = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;
struct Server {
    url: String,
    task: tokio::task::JoinHandle<()>,
    gateway: Arc<Gateway>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn server(backend: Arc<dyn Backend>) -> Server {
    server_with_usage(backend, None).await
}
async fn server_with_usage(backend: Arc<dyn Backend>, sink: Option<Arc<crate::usage::tests::Sink>>) -> Server {
    let gateway = Arc::new(Gateway::new(backend, ModelMap::single("served-model")));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "ws://{}/v1/realtime?model=served-model",
        listener.local_addr().unwrap()
    );
    let mut router = super::super::router(gateway.clone());
    if let Some(sink) = sink { router = router.layer(axum::middleware::from_fn_with_state(crate::usage::Middleware::new(sink), crate::usage::track)); }
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Server { url, task, gateway }
}
async fn connect(s: &Server, beta: bool) -> Client {
    let mut request = s.url.clone().into_client_request().unwrap();
    request.headers_mut().insert(
        "sec-websocket-protocol",
        "realtime, openai-insecure-api-key.sk-test".parse().unwrap(),
    );
    if beta {
        request
            .headers_mut()
            .insert("openai-beta", "realtime=v1".parse().unwrap());
    }
    let (mut client, response) = connect_async(request).await.unwrap();
    assert_eq!(response.headers()["sec-websocket-protocol"], "realtime");
    assert_eq!(recv(&mut client).await["type"], "session.created");
    if beta {
        assert_eq!(recv(&mut client).await["type"], "conversation.created");
    }
    client
}
async fn recv(client: &mut Client) -> Value {
    loop {
        let frame = tokio::time::timeout(std::time::Duration::from_secs(5), client.next())
            .await
            .expect("event timeout")
            .unwrap()
            .unwrap();
        if let ClientMessage::Text(text) = frame {
            let v: Value = serde_json::from_str(&text).unwrap();
            assert!(v["event_id"].is_string());
            return v;
        }
    }
}
async fn emit(client: &mut Client, event: Value) {
    client
        .send(ClientMessage::Text(event.to_string()))
        .await
        .unwrap();
}
async fn until(client: &mut Client, kind: &str) -> Vec<Value> {
    let mut events = vec![];
    loop {
        let event = recv(client).await;
        let done = event["type"] == kind;
        events.push(event);
        if done {
            return events;
        }
        assert!(events.len() < 100);
    }
}
fn user(id: &str, text: &str) -> Value {
    json!({"type":"message","id":id,"role":"user","content":[{"type":"input_text","text":text}]})
}
async fn create(client: &mut Client, item: Value, previous: Value, beta: bool) {
    emit(
        client,
        json!({"type":"conversation.item.create","item":item,"previous_item_id":previous}),
    )
    .await;
    assert_eq!(
        recv(client).await["type"],
        if beta {
            "conversation.item.created"
        } else {
            "conversation.item.added"
        }
    );
    if !beta {
        assert_eq!(recv(client).await["type"], "conversation.item.done");
    }
}

#[tokio::test]
async fn ga_history_text_tools_and_out_of_band() {
    let backend = Scripted::new(vec![
        vec![
            TurnEvent::TextDelta {
                text: "Hello ".into(),
            },
            TurnEvent::TextDelta {
                text: "world".into(),
            },
            TurnEvent::Usage {
                usage: Usage {
                    input_tokens: 11,
                    output_tokens: 2,
                    cached_input_tokens: 3,
                    ..Default::default()
                },
            },
            TurnEvent::Done {
                stop: StopReason::EndTurn,
            },
        ],
        vec![
            TurnEvent::ToolCallStart {
                index: 0,
                id: "call_1".into(),
                name: "weather".into(),
            },
            TurnEvent::ToolCallDelta {
                index: 0,
                arguments: "{\"city\":\"SF\"}".into(),
            },
            TurnEvent::ToolCallEnd { index: 0 },
            TurnEvent::Done {
                stop: StopReason::ToolUse,
            },
        ],
        vec![
            TurnEvent::TextDelta {
                text: "sunny".into(),
            },
            TurnEvent::Done {
                stop: StopReason::EndTurn,
            },
        ],
        vec![
            TurnEvent::TextDelta {
                text: "private".into(),
            },
            TurnEvent::Done {
                stop: StopReason::MaxTokens,
            },
        ],
    ]);
    let s = server(Arc::new(backend.clone())).await;
    let mut c = connect(&s, false).await;
    emit(&mut c,json!({"type":"session.update","session":{"instructions":"be brief","output_modalities":["text"],"tools":[{"type":"function","name":"weather","parameters":{"type":"object"}}],"tool_choice":"auto","audio":{"input":{"turn_detection":null}}}})).await;
    assert_eq!(recv(&mut c).await["type"], "session.updated");
    create(&mut c, user("a", "first"), Value::Null, false).await;
    create(&mut c, user("b", "second"), Value::Null, false).await;
    create(&mut c, user("middle", "middle"), json!("a"), false).await;
    create(&mut c, user("root_item", "root"), json!("root"), false).await;
    for item_id in ["a", "middle", "root_item"] {
        emit(
            &mut c,
            json!({"type":"conversation.item.retrieve","item_id":item_id}),
        )
        .await;
        let event = recv(&mut c).await;
        assert_eq!(event["type"], "conversation.item.retrieved");
        assert_eq!(event["item"]["id"], item_id);
    }
    emit(
        &mut c,
        json!({"type":"conversation.item.delete","item_id":"b"}),
    )
    .await;
    assert_eq!(recv(&mut c).await["type"], "conversation.item.deleted");
    emit(&mut c, json!({"type":"response.create"})).await;
    let events = until(&mut c, "response.done").await;
    let kinds: Vec<_> = events.iter().map(|v| v["type"].as_str().unwrap()).collect();
    assert_eq!(
        kinds,
        vec![
            "response.created",
            "conversation.item.added",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "conversation.item.done",
            "response.done"
        ]
    );
    // GA content-part events use "text"/"audio" (openai-python
    // RealtimeServerEvent types); item content uses "output_text". Agents
    // Python rejected "output_text" here in a live run (2026-10-10).
    for event in events.iter().filter(|v| v["type"].as_str().is_some_and(|t| t.starts_with("response.content_part."))) {
        assert_eq!(event["part"]["type"], "text", "{event}");
    }
    let done = &events.last().unwrap()["response"];
    assert_eq!(done["status"], "completed");
    assert_eq!(done["usage"]["total_tokens"], 13);
    assert_eq!(done["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(done["output"][0]["content"][0]["text"], "Hello world");
    let turn = &backend.turns()[0];
    assert_eq!(turn.system.as_deref(), Some("be brief"));
    assert_eq!(turn.items.len(), 3);
    assert!(
        matches!(&turn.items[0],Item::Message {content,..} if content==&vec![Part::text("root")])
    );
    emit(&mut c, json!({"type":"response.create"})).await;
    let events = until(&mut c, "response.done").await;
    assert!(events
        .iter()
        .any(|v| v["type"] == "response.function_call_arguments.delta"));
    assert_eq!(
        events.last().unwrap()["response"]["output"][0]["call_id"],
        "call_1"
    );
    create(
        &mut c,
        json!({"type":"function_call_output","id":"result","call_id":"call_1","output":"sunny"}),
        Value::Null,
        false,
    )
    .await;
    emit(&mut c, json!({"type":"response.create"})).await;
    until(&mut c, "response.done").await;
    emit(&mut c,json!({"type":"response.create","response":{"conversation":"none","input":[{"type":"item_reference","id":"a"}],"metadata":{"tag":"oob"}}})).await;
    let events = until(&mut c, "response.done").await;
    assert!(!events
        .iter()
        .any(|e| e["type"].as_str().unwrap().starts_with("conversation.")));
    let done = &events.last().unwrap()["response"];
    assert_eq!(done["status"], "incomplete");
    assert_eq!(done["metadata"]["tag"], "oob");
    assert!(done["conversation_id"].is_null());
    assert_eq!(backend.turns()[3].items.len(), 1);
    emit(
        &mut c,
        json!({"type":"conversation.item.retrieve","item_id":done["output"][0]["id"]}),
    )
    .await;
    assert_eq!(recv(&mut c).await["type"], "error");
}

#[tokio::test]
async fn every_client_event_error_is_recoverable() {
    let s = server(Arc::new(Scripted::default())).await;
    let mut c = connect(&s, false).await;
    c.send(ClientMessage::Text("{".into())).await.unwrap();
    assert_eq!(recv(&mut c).await["error"]["code"], "invalid_json");
    let cases = vec![
        json!({"type":"unknown"}),
        json!({"type":"conversation.item.retrieve","item_id":"missing"}),
        json!({"type":"conversation.item.delete","item_id":"missing"}),
        json!({"type":"conversation.item.create","previous_item_id":"missing","item":user("bad","x")}),
        json!({"type":"conversation.item.truncate","item_id":"missing","content_index":0,"audio_end_ms":100}),
        json!({"type":"response.cancel"}),
        json!({"type":"output_audio_buffer.clear"}),
        json!({"type":"transcription_session.update","session":{}}),
        json!({"type":"input_audio_buffer.append","audio":"invalid!"}),
        json!({"type":"input_audio_buffer.commit"}),
        json!({"type":"session.update","session":{"output_modalities":["audio"]}}),
        json!({"type":"session.update","session":{"audio":{"input":{"turn_detection":{"type":"semantic_vad"}}}}}),
        json!({"type":"session.update","session":{"prompt":{"id":"p"}}}),
    ];
    for (i, mut event) in cases.into_iter().enumerate() {
        let id = format!("client_{i}");
        event["event_id"] = json!(id);
        emit(&mut c, event).await;
        let e = recv(&mut c).await;
        assert_eq!(e["type"], "error", "{e}");
        assert_eq!(e["error"]["event_id"], id);
    }
    create(&mut c, user("ok", "still open"), Value::Null, false).await;
    emit(
        &mut c,
        json!({"type":"conversation.item.create","item":user("ok","duplicate")}),
    )
    .await;
    assert_eq!(recv(&mut c).await["type"], "error");
    emit(&mut c, json!({"type":"input_audio_buffer.clear"})).await;
    assert_eq!(recv(&mut c).await["type"], "input_audio_buffer.cleared");
}

#[tokio::test]
async fn manual_audio_vad_and_transcription_fail_honestly() {
    let s = server(Arc::new(Scripted::default())).await;
    let mut c = connect(&s, false).await;
    emit(&mut c,json!({"type":"session.update","session":{"audio":{"input":{"turn_detection":null,"transcription":{"model":"whisper-1"}}}}})).await;
    recv(&mut c).await;
    emit(
        &mut c,
        json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0u8;4800])}),
    )
    .await;
    emit(&mut c, json!({"type":"input_audio_buffer.commit"})).await;
    let events = until(&mut c, "conversation.item.input_audio_transcription.failed").await;
    assert_eq!(
        events
            .iter()
            .map(|v| v["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec![
            "input_audio_buffer.committed",
            "conversation.item.added",
            "conversation.item.done",
            "conversation.item.input_audio_transcription.failed"
        ]
    );
    emit(&mut c, json!({"type":"response.create"})).await;
    let events = until(&mut c, "response.done").await;
    assert_eq!(events.last().unwrap()["response"]["status"], "failed");
    assert!(
        events.last().unwrap()["response"]["status_details"]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("audio_in")
    );
    create(&mut c,json!({"type":"message","id":"direct_audio","role":"user","content":[{"type":"input_audio","audio":STANDARD.encode(vec![0u8;4800])}]}),Value::Null,false).await;
    let failure = recv(&mut c).await;
    assert_eq!(
        failure["type"],
        "conversation.item.input_audio_transcription.failed"
    );
    assert_eq!(failure["item_id"], "direct_audio");
    emit(&mut c,json!({"type":"session.update","session":{"audio":{"input":{"turn_detection":{"type":"server_vad","threshold":0.1,"prefix_padding_ms":20,"silence_duration_ms":30}}}}})).await;
    recv(&mut c).await;
    let bytes = [vec![0xff, 0x7f].repeat(2400), vec![0u8; 1440]].concat();
    emit(
        &mut c,
        json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(bytes)}),
    )
    .await;
    let events = until(&mut c, "response.done").await;
    assert_eq!(events[0]["type"], "input_audio_buffer.speech_started");
    assert_eq!(events[1]["type"], "input_audio_buffer.speech_stopped");
    assert_eq!(events[0]["item_id"], events[2]["item_id"]);
    assert_eq!(events.last().unwrap()["response"]["status"], "failed");
}

#[derive(Clone)]
struct Pending {
    dropped: Arc<std::sync::atomic::AtomicBool>,
}
struct DropStream(Arc<std::sync::atomic::AtomicBool>, bool);
impl futures::Stream for DropStream {
    type Item = Result<TurnEvent, GatewayError>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if !self.1 {
            self.1 = true;
            return std::task::Poll::Ready(Some(Ok(TurnEvent::TextDelta {
                text: "partial".into(),
            })));
        }
        std::task::Poll::Pending
    }
}
impl Drop for DropStream {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}
impl Backend for Pending {
    fn name(&self) -> &str {
        "pending"
    }
    fn capabilities(&self) -> BackendCapabilities {
        Default::default()
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    fn count_tokens(
        &self,
        _: super::super::TurnRequest,
    ) -> BoxFuture<'static, Result<u32, GatewayError>> {
        Box::pin(async { Ok(0) })
    }
    fn start(
        &self,
        _: super::super::TurnRequest,
    ) -> BoxFuture<'static, Result<TurnStream, GatewayError>> {
        let dropped = self.dropped.clone();
        Box::pin(async move { Ok(Box::pin(DropStream(dropped, false)) as TurnStream) })
    }
}
#[tokio::test]
async fn busy_cancel_and_disconnect_drop_backend() {
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let s = server(Arc::new(Pending {
        dropped: dropped.clone(),
    }))
    .await;
    let (mut c, _) = connect_async(&s.url).await.unwrap();
    let created = recv(&mut c).await;
    let session_id =
        super::super::session::SessionId(created["session"]["id"].as_str().unwrap().into());
    let session = s.gateway.sessions.get(&session_id).unwrap();
    emit(&mut c, json!({"type":"response.create"})).await;
    let first = recv(&mut c).await;
    assert_eq!(first["type"], "response.created");
    let delta = until(&mut c, "response.output_text.delta").await;
    let item_id = delta.last().unwrap()["item_id"].clone();
    emit(
        &mut c,
        json!({"type":"conversation.item.retrieve","item_id":item_id}),
    )
    .await;
    assert_eq!(recv(&mut c).await["item"]["content"][0]["text"], "partial");
    emit(&mut c, json!({"type":"response.create","event_id":"busy"})).await;
    assert_eq!(
        recv(&mut c).await["error"]["code"],
        "conversation_already_has_active_response"
    );
    emit(
        &mut c,
        json!({"type":"response.cancel","response_id":first["response"]["id"]}),
    )
    .await;
    let events = until(&mut c, "response.done").await;
    assert_eq!(events.last().unwrap()["response"]["status"], "cancelled");
    assert_eq!(
        events.last().unwrap()["response"]["output"][0]["status"],
        "incomplete"
    );
    assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
    dropped.store(false, std::sync::atomic::Ordering::SeqCst);
    emit(&mut c, json!({"type":"response.create"})).await;
    recv(&mut c).await;
    until(&mut c, "response.output_text.delta").await;
    c.close(None).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !dropped.load(std::sync::atomic::Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        while s.gateway.sessions.get(&session_id).is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let session = session.lock().await;
    assert!(session.running.is_none());
    assert_eq!(session.revision, 5); // Two append/final edits and the explicit cancel operation.
    assert!(
        matches!(&session.items[1].item,Item::Message {content,..} if content==&vec![Part::text("partial")])
    );
}

#[tokio::test]
async fn audio_truncation_changes_shared_storage_and_http_gaps_are_explicit() {
    let s = server(Arc::new(Scripted::default())).await;
    let (mut c, _) = connect_async(&s.url).await.unwrap();
    let created = recv(&mut c).await;
    let session_id =
        super::super::session::SessionId(created["session"]["id"].as_str().unwrap().into());
    let session = s.gateway.sessions.get(&session_id).unwrap();
    session
        .lock()
        .await
        .apply_with_item_id(
            SessionOp::Append {
                items: vec![Item::Message {
                    role: Role::Assistant,
                    content: vec![Part::Audio {
                        format: "pcm16".into(),
                        data: STANDARD.encode(vec![0u8; 9600]),
                    }],
                }],
            },
            "assistant_audio".into(),
        )
        .unwrap();
    emit(&mut c,json!({"type":"conversation.item.truncate","item_id":"assistant_audio","content_index":0,"audio_end_ms":100})).await;
    assert_eq!(recv(&mut c).await["type"], "conversation.item.truncated");
    emit(
        &mut c,
        json!({"type":"conversation.item.retrieve","item_id":"assistant_audio"}),
    )
    .await;
    let retrieved = recv(&mut c).await;
    assert_eq!(
        STANDARD
            .decode(retrieved["item"]["content"][0]["audio"].as_str().unwrap())
            .unwrap()
            .len(),
        4800
    );
    emit(&mut c,json!({"type":"conversation.item.truncate","item_id":"assistant_audio","content_index":0,"audio_end_ms":101})).await;
    assert_eq!(recv(&mut c).await["type"], "error");
    for path in [
        "client_secrets",
        "sessions",
        "transcription_sessions",
        "calls",
    ] {
        use tower::ServiceExt;
        let response = super::super::router(s.gateway.clone())
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(format!("/v1/realtime/{path}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
            "unsupported_value"
        );
    }
    c.close(None).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while s.gateway.sessions.get(&session_id).is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn beta_mechanical_event_names_and_transcription_intent() {
    let backend = Scripted::new(vec![vec![
        TurnEvent::TextDelta {
            text: "beta".into(),
        },
        TurnEvent::Done {
            stop: StopReason::EndTurn,
        },
    ]]);
    let mut s = server(Arc::new(backend)).await;
    let mut c = connect(&s, true).await;
    emit(&mut c,json!({"type":"session.update","session":{"modalities":["text"],"turn_detection":null,"temperature":0.7}})).await;
    assert_eq!(recv(&mut c).await["type"], "session.updated");
    create(&mut c, user("u", "hello"), Value::Null, true).await;
    emit(&mut c, json!({"type":"response.create"})).await;
    let events = until(&mut c, "response.done").await;
    assert!(events.iter().any(|v| v["type"] == "response.text.delta"));
    assert_eq!(
        events.last().unwrap()["response"]["output"][0]["content"][0]["type"],
        "text"
    );
    s.url.push_str("&intent=transcription");
    let mut request = s.url.clone().into_client_request().unwrap();
    request
        .headers_mut()
        .insert("openai-beta", "realtime=v1".parse().unwrap());
    let (mut t, _) = connect_async(request).await.unwrap();
    assert_eq!(recv(&mut t).await["type"], "transcription_session.created");
    assert_eq!(recv(&mut t).await["type"], "conversation.created");
    emit(&mut t,json!({"type":"transcription_session.update","session":{"turn_detection":null,"input_audio_transcription":{"model":"whisper-1"}}})).await;
    assert_eq!(recv(&mut t).await["type"], "transcription_session.updated");
    emit(
        &mut t,
        json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0u8;4800])}),
    )
    .await;
    emit(&mut t, json!({"type":"input_audio_buffer.commit"})).await;
    until(&mut t, "conversation.item.input_audio_transcription.failed").await;
}

#[tokio::test]
async fn official_sdk_captured_first_frames_replay() {
    let captures = vec![
        (
            "openai-python",
            false,
            r###"[{"type":"session.update","session":{"type":"realtime","output_modalities":["text"],"audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"turn_detection":null}},"tools":[{"type":"function","name":"get_time","description":"Return a fixed test time.","parameters":{"type":"object","properties":{},"additionalProperties":false}}],"tool_choice":"auto"}},{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hello in one short sentence."}]}},{"type":"response.create","response":{"tool_choice":"none"}},{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Call get_time now."}]}},{"type":"response.create","response":{"tool_choice":{"type":"function","name":"get_time"}}},{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_mock","output":"{\"time\":\"2000-01-01T00:00:00Z\"}"}}]"###,
        ),
        (
            "openai-python-beta",
            true,
            r###"[{"type":"session.update","session":{"modalities":["text"],"input_audio_format":"pcm16","turn_detection":null,"tools":[{"type":"function","name":"get_time","description":"Return a fixed test time.","parameters":{"type":"object","properties":{},"additionalProperties":false}}],"tool_choice":"auto"}},{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hello in one short sentence."}]}},{"type":"response.create","response":{"tool_choice":"none"}},{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Call get_time now."}]}},{"type":"response.create","response":{"tool_choice":{"type":"function","name":"get_time"}}},{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_mock","output":"{\"time\":\"2000-01-01T00:00:00Z\"}"}}]"###,
        ),
        (
            "openai-node",
            false,
            r###"[{"type":"session.update","session":{"type":"realtime","output_modalities":["text"],"audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"turn_detection":null}},"tools":[{"type":"function","name":"get_time","description":"Return a fixed test time.","parameters":{"type":"object","properties":{},"additionalProperties":false}}],"tool_choice":"auto"}},{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hello in one short sentence."}]}},{"type":"response.create","response":{"tool_choice":"none"}},{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Call get_time now."}]}},{"type":"response.create","response":{"tool_choice":{"type":"function","name":"get_time"}}},{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_mock","output":"{\"time\":\"2000-01-01T00:00:00Z\"}"}}]"###,
        ),
        (
            "openai-node-native",
            false,
            r###"[{"type":"session.update","session":{"type":"realtime","output_modalities":["text"],"audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"turn_detection":null}},"tools":[{"type":"function","name":"get_time","description":"Return a fixed test time.","parameters":{"type":"object","properties":{},"additionalProperties":false}}],"tool_choice":"auto"}},{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hello in one short sentence."}]}},{"type":"response.create","response":{"tool_choice":"none"}},{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Call get_time now."}]}},{"type":"response.create","response":{"tool_choice":{"type":"function","name":"get_time"}}},{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_mock","output":"{\"time\":\"2000-01-01T00:00:00Z\"}"}}]"###,
        ),
        (
            "agents-js",
            false,
            r###"[{"type":"session.update","session":{"type":"realtime","instructions":"Answer briefly. Call get_time when requested.","model":"default","output_modalities":["text"],"audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"noise_reduction":null,"transcription":null,"turn_detection":null},"output":{"format":{"type":"audio/pcm","rate":24000},"speed":1}},"tools":[{"type":"function","name":"get_time","description":"Return a fixed test time.","parameters":{"type":"object","properties":{},"additionalProperties":false,"required":[]}}]}},{"type":"session.update","session":{"type":"realtime","tracing":null}},{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hello in one short sentence."}]}},{"type":"response.create","response":{"tool_choice":"none"},"event_id":"agents_js_response_create_1"},{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Call get_time now."}]}},{"type":"response.create","response":{"tool_choice":{"type":"function","name":"get_time"}},"event_id":"agents_js_response_create_2"}]"###,
        ),
    ];
    for (client_name, beta, frames) in captures {
        let backend = Scripted::new(vec![
            vec![
                TurnEvent::TextDelta {
                    text: "hello".into(),
                },
                TurnEvent::Done {
                    stop: StopReason::EndTurn,
                },
            ],
            vec![
                TurnEvent::ToolCallStart {
                    index: 0,
                    id: "call_mock".into(),
                    name: "get_time".into(),
                },
                TurnEvent::ToolCallDelta {
                    index: 0,
                    arguments: "{}".into(),
                },
                TurnEvent::ToolCallEnd { index: 0 },
                TurnEvent::Done {
                    stop: StopReason::ToolUse,
                },
            ],
        ]);
        let mut s = server(Arc::new(backend.clone())).await;
        s.url = s.url.replace("served-model", "default");
        // The deployment aliases default through ModelMap; preserve echo of the alias.
        // Replace the router with an accept-any one for this fixture replay.
        s.task.abort();
        let mut map = ModelMap::single("served-model");
        map.accept_any = true;
        let gateway = Arc::new(Gateway::new(Arc::new(backend.clone()), map));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        s.url = format!(
            "ws://{}/v1/realtime?model=default",
            listener.local_addr().unwrap()
        );
        let router = super::super::router(gateway);
        s.task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let mut c = connect(&s, beta).await;
        let frames: Vec<Value> = serde_json::from_str(frames).unwrap();
        let mut had_tools = false;
        for frame in frames {
            let kind = frame["type"].as_str().unwrap().to_owned();
            emit(&mut c, frame).await;
            match kind.as_str() {
                "session.update" => {
                    let event = recv(&mut c).await;
                    assert_eq!(event["type"], "session.updated", "{client_name}: {event}");
                    if had_tools {
                        assert_eq!(event["session"]["tools"][0]["name"], "get_time");
                    }
                    had_tools = event["session"]["tools"]
                        .as_array()
                        .is_some_and(|t| !t.is_empty());
                }
                "conversation.item.create" => {
                    assert_eq!(
                        recv(&mut c).await["type"],
                        if beta {
                            "conversation.item.created"
                        } else {
                            "conversation.item.added"
                        }
                    );
                    if !beta {
                        assert_eq!(recv(&mut c).await["type"], "conversation.item.done");
                    }
                }
                "response.create" => {
                    let events = until(&mut c, "response.done").await;
                    assert_eq!(
                        events.last().unwrap()["response"]["status"],
                        "completed",
                        "{client_name}"
                    );
                }
                _ => panic!("unexpected captured client event"),
            }
        }
        assert_eq!(
            backend.turns()[1].tool_choice,
            super::super::turn::ToolChoice::Named {
                name: "get_time".into()
            }
        );
        emit(
            &mut c,
            json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0u8;12000])}),
        )
        .await;
        emit(&mut c, json!({"type":"input_audio_buffer.commit"})).await;
        until(
            &mut c,
            if beta {
                "conversation.item.created"
            } else {
                "conversation.item.done"
            },
        )
        .await;
        emit(&mut c, json!({"type":"response.create"})).await;
        let events = until(&mut c, "response.done").await;
        assert_eq!(events.last().unwrap()["response"]["status"], "failed");
    }
}

#[tokio::test]
async fn incomplete_tools_never_publish_completed_calls() {
    let stops = vec![
        StopReason::MaxTokens,
        StopReason::ContentFilter,
        StopReason::Refusal,
        StopReason::Cancelled,
    ];
    for stop in stops {
        let expected = if stop == StopReason::Cancelled {
            "cancelled"
        } else {
            "incomplete"
        };
        let backend = Scripted::new(vec![vec![
            TurnEvent::ToolCallStart {
                index: 0,
                id: "cut".into(),
                name: "get_time".into(),
            },
            TurnEvent::ToolCallDelta {
                index: 0,
                arguments: "{\"partial\":".into(),
            },
            TurnEvent::ToolCallEnd { index: 0 },
            TurnEvent::Done { stop: stop.clone() },
        ]]);
        let s = server(Arc::new(backend)).await;
        let mut c = connect(&s, false).await;
        emit(&mut c, json!({"type":"response.create"})).await;
        let events = until(&mut c, "response.done").await;
        assert_eq!(events.last().unwrap()["response"]["status"], expected);
        assert_eq!(
            events.last().unwrap()["response"]["output"][0]["status"],
            "incomplete"
        );
        assert!(!events.iter().any(
            |e| e["type"] == "response.output_item.done" && e["item"]["status"] == "completed"
        ));
        if stop == StopReason::MaxTokens {
            assert_eq!(
                events.last().unwrap()["response"]["status_details"]["reason"],
                "max_output_tokens"
            );
        }
    }
    for eof in [false, true] {
        let backend = Scripted::default();
        let mut script = vec![
            Ok(TurnEvent::ToolCallStart {
                index: 0,
                id: "cut".into(),
                name: "get_time".into(),
            }),
            Ok(TurnEvent::ToolCallDelta {
                index: 0,
                arguments: "{".into(),
            }),
            Ok(TurnEvent::ToolCallEnd { index: 0 }),
        ];
        if !eof {
            script.push(Err(GatewayError::upstream("test upstream failure")));
        }
        backend.scripts.lock().unwrap().push(script);
        let s = server(Arc::new(backend)).await;
        let mut c = connect(&s, false).await;
        emit(&mut c, json!({"type":"response.create"})).await;
        let events = until(&mut c, "response.done").await;
        assert_eq!(events.last().unwrap()["response"]["status"], "failed");
        assert_eq!(
            events.last().unwrap()["response"]["output"][0]["status"],
            "incomplete"
        );
    }
}

#[tokio::test]
async fn text_tool_text_order_and_agents_pipecat_wire_fields() {
    let backend = Scripted::new(vec![vec![
        TurnEvent::TextDelta {
            text: "before".into(),
        },
        TurnEvent::ToolCallStart {
            index: 0,
            id: "call_time".into(),
            name: "get_time".into(),
        },
        TurnEvent::ToolCallDelta {
            index: 0,
            arguments: "{}".into(),
        },
        TurnEvent::ToolCallEnd { index: 0 },
        TurnEvent::TextDelta {
            text: "after".into(),
        },
        TurnEvent::Done {
            stop: StopReason::EndTurn,
        },
    ]]);
    let s = server(Arc::new(backend.clone())).await;
    let (mut c, _) = connect_async(&s.url).await.unwrap();
    let initial = recv(&mut c).await;
    let sid = super::super::session::SessionId(initial["session"]["id"].as_str().unwrap().into());
    emit(&mut c,json!({"type":"session.update","session":{"tools":[{"type":"function","name":"get_time","parameters":{"type":"object","required":[]}}]}})).await;
    recv(&mut c).await;
    emit(
        &mut c,
        json!({"type":"response.create","response":{"tool_choice":"required"}}),
    )
    .await;
    let events = until(&mut c, "response.done").await;
    assert_eq!(
        backend.turns()[0].tool_choice,
        super::super::turn::ToolChoice::Named {
            name: "get_time".into()
        }
    );
    assert!(!events
        .iter()
        .any(|e| e["type"] == "conversation.item.created" || e["type"] == "response.text.delta"));
    let added = events
        .iter()
        .position(|e| {
            e["type"] == "conversation.item.added" && e["item"]["type"] == "function_call"
        })
        .unwrap();
    let args_done = events
        .iter()
        .position(|e| e["type"] == "response.function_call_arguments.done")
        .unwrap();
    assert!(added < args_done);
    assert_eq!(events[args_done]["name"], "get_time");
    let done = &events.last().unwrap()["response"];
    assert!(done.get("status_details").is_some());
    assert!(done["usage"]["input_token_details"].is_object());
    assert!(done["usage"]["output_token_details"].is_object());
    let output = done["output"].as_array().unwrap();
    assert_eq!(output.len(), 3);
    assert_eq!(output[0]["content"][0]["text"], "before");
    assert_eq!(output[0]["content"][0]["type"], "output_text");
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[2]["content"][0]["text"], "after");
    let session = s.gateway.sessions.get(&sid).unwrap();
    let session = session.lock().await;
    assert!(
        matches!(&session.items[0].item,Item::Message {content,..} if content==&vec![Part::text("before")])
    );
    assert!(matches!(&session.items[1].item, Item::ToolCall { .. }));
    assert!(
        matches!(&session.items[2].item,Item::Message {content,..} if content==&vec![Part::text("after")])
    );
}

#[tokio::test]
async fn streaming_history_revision_is_constant_not_per_delta() {
    let backend = Scripted::new(vec![[
        (0..2048)
            .map(|_| TurnEvent::TextDelta { text: "x".into() })
            .collect::<Vec<_>>(),
        vec![TurnEvent::Done {
            stop: StopReason::EndTurn,
        }],
    ]
    .concat()]);
    let s = server(Arc::new(backend)).await;
    let (mut c, _) = connect_async(&s.url).await.unwrap();
    let initial = recv(&mut c).await;
    let sid = super::super::session::SessionId(initial["session"]["id"].as_str().unwrap().into());
    emit(&mut c, json!({"type":"response.create"})).await;
    loop {
        if recv(&mut c).await["type"] == "response.done" {
            break;
        }
    }
    let session = s.gateway.sessions.get(&sid).unwrap();
    let session = session.lock().await;
    assert_eq!(
        session.revision, 2,
        "one append and one final edit, independent of delta count"
    );
    assert!(
        matches!(&session.items[0].item,Item::Message {content,..} if content==&vec![Part::text("x".repeat(2048))])
    );
}

#[tokio::test]
async fn idle_preserves_deadline_for_silence_and_drains_buffer() {
    let gateway = Arc::new(Gateway::new(
        Arc::new(Scripted::default()),
        ModelMap::single("served-model"),
    ));
    let session = gateway.sessions.create("sess");
    let mut config = protocol::defaults("served-model", false, false);
    config["audio"]["input"]["turn_detection"]["idle_timeout_ms"] = json!(6000);
    let mut c = Connection {
        gateway,
        session,
        config,
        beta: false,
        transcription: false,
        conversation_id: "conv_test".into(),
        tape: Default::default(),
        audio: Default::default(),
        active: None,
        cancel: None,
        idle: None,
        usage: None, turn_usage: None,
    };
    c.arm_idle();
    let deadline = c.idle.unwrap();
    let silence = vec![0u8; 4800];
    for _ in 0..10 {
        c.client(&json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(&silence)}))
            .await
            .unwrap();
        assert_eq!(c.idle, Some(deadline));
    }
    let settings = audio::settings(&c.config, false).unwrap();
    let (item_id, audio, start, end) = c.audio.drain_idle(&settings);
    assert!(!audio.is_empty());
    assert_eq!((start, end), (700, 1000));
    let expected = audio.len();
    c.commit(item_id, audio).await.unwrap();
    assert!(c.audio.is_empty());
    let guard = c.session.lock().await;
    let Item::Message { content, .. } = &guard.items[0].item else {
        panic!()
    };
    let Part::Audio { data, .. } = &content[0] else {
        panic!()
    };
    assert_eq!(STANDARD.decode(data).unwrap().len(), expected);
    drop(guard);
    c.client(&json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0xff,0x7f].repeat(240))})).await.unwrap();
    assert!(c.idle.is_none(), "speech clears the idle deadline");
    c.arm_idle();
    assert!(
        c.idle.is_none(),
        "response completion cannot arm idle during speech"
    );
}

#[tokio::test]
async fn idle_timeout_survives_silent_websocket_frames_and_commits_audio() {
    let s = server(Arc::new(Scripted::default())).await;
    let (mut c, _) = connect_async(&s.url).await.unwrap();
    let initial = recv(&mut c).await;
    let sid = super::super::session::SessionId(initial["session"]["id"].as_str().unwrap().into());
    emit(&mut c,json!({"type":"session.update","session":{"audio":{"input":{"turn_detection":{"idle_timeout_ms":6000,"create_response":false}}}}})).await;
    assert_eq!(recv(&mut c).await["type"], "session.updated");
    for _ in 0..64 {
        emit(
            &mut c,
            json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0u8;4800])}),
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let timeout = recv(&mut c).await;
    assert_eq!(timeout["type"], "input_audio_buffer.timeout_triggered");
    assert!(
        timeout["audio_end_ms"].as_u64().unwrap() > timeout["audio_start_ms"].as_u64().unwrap()
    );
    assert_eq!(recv(&mut c).await["type"], "input_audio_buffer.committed");
    assert_eq!(recv(&mut c).await["type"], "conversation.item.added");
    assert_eq!(recv(&mut c).await["type"], "conversation.item.done");
    let session = s.gateway.sessions.get(&sid).unwrap();
    let session = session.lock().await;
    let Item::Message { content, .. } = &session.items[0].item else {
        panic!()
    };
    let Part::Audio { data, .. } = &content[0] else {
        panic!()
    };
    assert_eq!(STANDARD.decode(data).unwrap(), vec![0u8; 14400]);
}

#[tokio::test]
async fn pipecat_default_vad_silence_manual_commit_and_empty_warmup() {
    let mut backend = Scripted::new(vec![
        vec![TurnEvent::Done {
            stop: StopReason::EndTurn,
        }],
        vec![
            TurnEvent::TextDelta {
                text: "hello".into(),
            },
            TurnEvent::Done {
                stop: StopReason::EndTurn,
            },
        ],
    ]);
    backend.capabilities.audio_in = true;
    let s = server(Arc::new(backend.clone())).await;
    let mut c = connect(&s, false).await;
    emit(&mut c,json!({"type":"session.update","session":{"output_modalities":["text"],"instructions":"Pipecat synthetic test"}})).await;
    let updated = recv(&mut c).await;
    assert_eq!(
        updated["session"]["audio"]["input"]["turn_detection"]["type"],
        "server_vad"
    );
    assert!(updated["session"]["audio"]["input"]["transcription"].is_null());
    emit(&mut c, json!({"type":"response.create"})).await;
    let warmup = until(&mut c, "response.done").await;
    assert_eq!(warmup.last().unwrap()["response"]["status"], "completed");
    assert!(backend.turns()[0].items.is_empty());
    emit(
        &mut c,
        json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0u8;4800])}),
    )
    .await;
    emit(&mut c, json!({"type":"input_audio_buffer.commit"})).await;
    // Any auto speech/response events would appear ahead of this explicit commit.
    assert_eq!(recv(&mut c).await["type"], "input_audio_buffer.committed");
    let added = recv(&mut c).await;
    assert_eq!(added["type"], "conversation.item.added");
    assert_eq!(added["item"]["content"][0]["type"], "input_audio");
    assert_eq!(recv(&mut c).await["type"], "conversation.item.done");
    create(&mut c, user("pipecat_text", "hello"), Value::Null, false).await;
    emit(&mut c, json!({"type":"response.create"})).await;
    let response = until(&mut c, "response.done").await;
    assert_eq!(response.last().unwrap()["response"]["status"], "completed");
    assert_eq!(backend.turns().len(), 2);
}

#[tokio::test]
async fn session_update_cancels_old_idle_deadline_without_committing() {
    for turn_detection in [json!({"idle_timeout_ms":null}), Value::Null] {
        let s = server(Arc::new(Scripted::default())).await;
        let mut c = connect(&s, false).await;
        emit(&mut c,json!({"type":"session.update","session":{"audio":{"input":{"turn_detection":{"idle_timeout_ms":6000,"create_response":false}}}}})).await;
        assert_eq!(recv(&mut c).await["type"], "session.updated");
        emit(
            &mut c,
            json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(vec![0u8;4800])}),
        )
        .await;
        emit(&mut c,json!({"type":"session.update","session":{"audio":{"input":{"turn_detection":turn_detection}}}})).await;
        assert_eq!(recv(&mut c).await["type"], "session.updated");
        tokio::time::sleep(std::time::Duration::from_millis(6200)).await;
        // A stale timeout would precede this explicit commit and consume the buffer.
        emit(&mut c, json!({"type":"input_audio_buffer.commit"})).await;
        assert_eq!(recv(&mut c).await["type"], "input_audio_buffer.committed");
        let added = recv(&mut c).await;
        assert_eq!(added["type"], "conversation.item.added");
        assert_eq!(
            STANDARD
                .decode(added["item"]["content"][0]["audio"].as_str().unwrap())
                .unwrap()
                .len(),
            4800
        );
        assert_eq!(recv(&mut c).await["type"], "conversation.item.done");
    }
}

#[tokio::test]
async fn session_update_rearms_changed_idle_timeout() {
    let gateway = Arc::new(Gateway::new(
        Arc::new(Scripted::default()),
        ModelMap::single("served-model"),
    ));
    let session = gateway.sessions.create("sess");
    let mut c = Connection {
        gateway,
        session,
        config: protocol::defaults("served-model", false, false),
        beta: false,
        transcription: false,
        conversation_id: "conv_test".into(),
        tape: Default::default(),
        audio: Default::default(),
        active: None,
        cancel: None,
        idle: None,
        usage: None, turn_usage: None,
    };
    for ms in [6000, 30000, 6000] {
        let before = tokio::time::Instant::now();
        c.client(&json!({"type":"session.update","session":{"audio":{"input":{"turn_detection":{"idle_timeout_ms":ms}}}}})).await.unwrap();
        let after = tokio::time::Instant::now();
        let deadline = c.idle.unwrap();
        assert!(deadline >= before + std::time::Duration::from_millis(ms));
        assert!(deadline <= after + std::time::Duration::from_millis(ms));
    }
}

#[tokio::test]
async fn usage_connection_and_turn_rows_are_independent() {
    let backend = Scripted::new(vec![vec![TurnEvent::TextDelta {text:"hello".into()},
        TurnEvent::Usage {usage: Usage {input_tokens:7,output_tokens:2,..Default::default()}},
        TurnEvent::Done {stop:StopReason::EndTurn}]]);
    let sink = Arc::new(crate::usage::tests::Sink::default());
    let s = server_with_usage(Arc::new(backend.clone()),Some(sink.clone())).await;
    let mut c = connect(&s,false).await;
    create(&mut c,user("u","hello"),Value::Null,false).await;
    emit(&mut c,json!({"type":"response.create"})).await;
    until(&mut c,"response.done").await;
    backend.seen.lock().unwrap().clear();
    tokio::time::timeout(std::time::Duration::from_secs(2),async {
        loop {
            if sink.0.lock().unwrap().iter().any(|r| r.protocol == "realtime") {break;}
            tokio::task::yield_now().await;
        }
    }).await.unwrap();
    let rows = sink.0.lock().unwrap();
    let row = rows.iter().find(|r| r.protocol == "realtime").unwrap();
    assert_eq!(row.tokens_in,Some(7));
    assert_eq!(row.tokens_out,Some(2));
    assert_eq!(row.outcome,"ok");
    assert_eq!(row.session_source.as_deref(),Some("explicit"));
    assert_eq!(row.stop_reason.as_deref(),Some("end_turn"));
    drop(rows);
    c.close(None).await.unwrap();
}
