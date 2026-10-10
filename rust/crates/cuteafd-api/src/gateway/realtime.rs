//! GA and beta Realtime WebSocket front end. No WebRTC or synthetic audio output.
use super::{
    record::Tape,
    session::{Position, Session, SessionOp},
    turn::{Item, Part, Role, StopReason, TurnEvent},
    Gateway, GatewayError,
};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::HeaderMap,
    response::{IntoResponse, Response as HttpResponse},
    routing::{get, post},
    Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;
pub mod audio;
mod protocol;
mod response;
#[cfg(test)]
mod tests;
use protocol::{id, invalid, string};

pub fn routes(gateway: Arc<Gateway>) -> Router {
    Router::new()
        .route("/v1/realtime", get(upgrade))
        .route("/v1/realtime/client_secrets", post(no_secrets))
        .route("/v1/realtime/sessions", post(no_secrets))
        .route("/v1/realtime/transcription_sessions", post(no_secrets))
        .route("/v1/realtime/calls", post(no_webrtc))
        .with_state(gateway)
}
async fn no_secrets() -> HttpResponse {
    GatewayError::unsupported("ephemeral credential minting is not configured; connect with the configured gateway API key").openai_response()
}
async fn no_webrtc() -> HttpResponse {
    GatewayError::unsupported("WebRTC SDP calls are not supported; use GET /v1/realtime WebSocket")
        .openai_response()
}
#[derive(Deserialize)]
struct Connect {
    model: Option<String>,
    intent: Option<String>,
}
async fn upgrade(
    State(gateway): State<Arc<Gateway>>,
    Query(query): Query<Connect>,
    headers: HeaderMap,
    tape: Tape,
    usage: Option<crate::usage::UsageHandle>,
    ws: WebSocketUpgrade,
) -> HttpResponse {
    let transcription = match query.intent.as_deref() {
        None => false,
        Some("transcription") => true,
        _ => return invalid("intent", "only transcription intent is supported").openai_response(),
    };
    let model = query.model.unwrap_or_else(|| {
        gateway
            .models
            .listing(&gateway.backend.models())
            .first()
            .map(|m| m.id.clone())
            .unwrap_or_default()
    });
    if let Err(e) = gateway.models.resolve(&model) {
        return e.openai_response();
    }
    let protocols = headers
        .get("sec-websocket-protocol")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let beta = headers
        .get("openai-beta")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|v| v.trim() == "realtime=v1"))
        || protocols
            .split(',')
            .any(|v| v.trim() == "openai-beta.realtime-v1");
    ws.protocols(["realtime"])
        .max_message_size(32 * 1024 * 1024)
        .on_upgrade(move |socket| serve(socket, gateway, model, beta, transcription, tape, usage))
        .into_response()
}
struct Connection {
    gateway: Arc<Gateway>,
    session: Arc<Mutex<Session>>,
    config: Value,
    beta: bool,
    transcription: bool,
    conversation_id: String,
    tape: Tape,
    audio: audio::Buffer,
    active: Option<response::Response>,
    cancel: Option<tokio::sync::watch::Receiver<bool>>,
    idle: Option<tokio::time::Instant>,
    usage: Option<crate::usage::UsageHandle>,
    turn_usage: Option<crate::usage::UsageHandle>,
}
async fn send(socket: &mut WebSocket, tape: &Tape, mut event: Value) -> bool {
    event["event_id"] = json!(id("event"));
    let text = event.to_string();
    tape.frame("server", &text);
    socket.send(Message::Text(text)).await.is_ok()
}
async fn serve(
    mut socket: WebSocket,
    gateway: Arc<Gateway>,
    model: String,
    beta: bool,
    transcription: bool,
    tape: Tape,
    usage: Option<crate::usage::UsageHandle>,
) {
    let session = gateway.sessions.create("sess");
    let session_id = session.lock().await.id.clone();
    let mut config = protocol::defaults(&model, beta, transcription);
    config["id"] = json!(session_id.0);
    let mut c = Connection {
        gateway,
        session,
        config,
        beta,
        transcription,
        conversation_id: id("conv"),
        tape,
        audio: audio::Buffer::default(),
        active: None,
        cancel: None,
        idle: None,
        usage,
        turn_usage: None,
    };
    if let Some(usage) = &c.usage {
        usage.session(session_id.0.clone(), "explicit");
        usage.details(crate::usage::Details { model_requested: Some(model.clone()), stream: true, ..Default::default() });
        usage.served_model(&c.gateway.models.resolve(&model).unwrap_or(model));
    }
    let initial = if beta && transcription {
        "transcription_session.created"
    } else {
        "session.created"
    };
    if !send(
        &mut socket,
        &c.tape,
        json!({"type":initial,"session":c.config}),
    )
    .await
    {
        c.gateway.sessions.close(&session_id);
        return;
    }
    if beta && !send(&mut socket,&c.tape,json!({"type":"conversation.created","conversation":{"id":c.conversation_id,"object":"realtime.conversation"}})).await {c.gateway.sessions.close(&session_id);return;}
    enum Next {
        Socket(Option<Result<Message, axum::Error>>),
        Turn(Option<Result<TurnEvent, GatewayError>>),
        Cancel,
        Idle,
        Expire,
    }
    let expires = tokio::time::Instant::now() + std::time::Duration::from_secs(3600);
    loop {
        let next = tokio::select! {
            biased;
            message=socket.next()=>Next::Socket(message),
            _=async { if let Some(cancel)=c.cancel.as_mut(){let _=cancel.changed().await;} else {std::future::pending::<()>().await;} }=>Next::Cancel,
            event=async {if let Some(active)=c.active.as_mut(){active.stream.next().await} else {std::future::pending().await}}=>Next::Turn(event),
            _=async {if let Some(deadline)=c.idle{tokio::time::sleep_until(deadline).await} else {std::future::pending().await}}=>Next::Idle,
            _=tokio::time::sleep_until(expires)=>Next::Expire,
        };
        let events = match next {
            Next::Socket(Some(Ok(Message::Text(text)))) => {
                c.tape.frame("client", &text);
                match serde_json::from_str::<Value>(&text) {
                    Ok(v) => {
                        let event_id = v.get("event_id").and_then(Value::as_str);
                        match c.client(&v).await {
                            Ok(events) => events,
                            Err(e) => {
                                let code = if e.message.contains("active response") {
                                    Some("conversation_already_has_active_response")
                                } else if e.message.contains("no response") {
                                    Some("response_cancel_not_active")
                                } else {
                                    None
                                };
                                vec![protocol::error(e, event_id, code)]
                            }
                        }
                    }
                    Err(_) => vec![protocol::error(
                        invalid("event", "malformed JSON"),
                        None,
                        Some("invalid_json"),
                    )],
                }
            }
            Next::Socket(Some(Ok(Message::Binary(bytes)))) => {
                c.tape
                    .frame("client", &format!("[binary:{}]", STANDARD.encode(bytes)));
                vec![protocol::error(
                    invalid("event", "Realtime client events must be JSON text frames"),
                    None,
                    None,
                )]
            }
            Next::Socket(Some(Ok(Message::Ping(bytes)))) => {
                c.tape
                    .frame("client", &format!("[ping:{}]", STANDARD.encode(&bytes)));
                c.tape
                    .frame("server", &format!("[pong:{}]", STANDARD.encode(&bytes)));
                if socket.send(Message::Pong(bytes)).await.is_err() {
                    break;
                }
                continue;
            }
            Next::Socket(Some(Ok(Message::Pong(bytes)))) => {
                c.tape
                    .frame("client", &format!("[pong:{}]", STANDARD.encode(bytes)));
                continue;
            }
            Next::Socket(Some(Ok(Message::Close(_)))) => {
                c.tape.frame("client", "[close]");
                break;
            }
            Next::Socket(_) | Next::Expire => break,
            Next::Turn(Some(Ok(TurnEvent::Done { stop }))) => c.finish(Some(stop), None).await,
            Next::Turn(Some(Ok(event))) => {
                let events = c.active.as_mut().unwrap().event_turn(event);
                c.sync_output().await;
                events
            }
            Next::Turn(Some(Err(e))) => c.finish(None, Some(e)).await,
            Next::Turn(None) => {
                c.finish(
                    None,
                    Some(GatewayError::upstream("backend stream ended without Done")),
                )
                .await
            }
            Next::Cancel => c.finish(Some(StopReason::Cancelled), None).await,
            Next::Idle => {
                c.idle = None;
                let settings =
                    audio::settings(&c.config, c.beta).expect("validated session audio config");
                let (item_id, bytes, start, end) = c.audio.drain_idle(&settings);
                let mut events = vec![
                    json!({"type":"input_audio_buffer.timeout_triggered","item_id":item_id,"audio_start_ms":start,"audio_end_ms":end}),
                ];
                match c.commit(item_id, bytes).await {
                    Ok(e) => events.extend(e),
                    Err(e) => events.push(protocol::error(e, None, None)),
                };
                if !c.transcription
                    && audio::settings(&c.config, c.beta)
                        .ok()
                        .and_then(|s| s.vad)
                        .is_some_and(|v| v.create_response)
                {
                    match c.start(&json!({})).await {
                        Ok(e) => events.extend(e),
                        Err(e) => events.push(protocol::error(e, None, None)),
                    };
                }
                events
            }
        };
        let mut open = true;
        for event in events {
            if !send(&mut socket, &c.tape, event).await {
                open = false;
                break;
            }
        }
        if !open {
            break;
        }
    }
    // Retain the final partial contents for any shared holders before closing.
    c.finish(Some(StopReason::Cancelled), None).await;
    c.cancel.take();
    c.session.lock().await.running = None;
    c.gateway.sessions.close(&session_id);
}
impl Connection {
    fn item_events(&self, item_id: &str, item: &Item, previous: Option<String>) -> Vec<Value> {
        let item = protocol::wire_item(item_id, item, "completed", self.beta);
        let mut events = vec![
            json!({"type":if self.beta {"conversation.item.created"} else {"conversation.item.added"},"previous_item_id":previous,"item":item}),
        ];
        if !self.beta {
            events.push(json!({"type":"conversation.item.done","item":item}));
        }
        events
    }
    async fn client(&mut self, event: &Value) -> Result<Vec<Value>, GatewayError> {
        match string(event, "type")? {
            "session.update" | "transcription_session.update" => {
                if event["type"] == "transcription_session.update"
                    && !(self.beta && self.transcription)
                {
                    return Err(invalid(
                        "type",
                        "transcription_session.update requires a beta transcription session",
                    ));
                }
                let config = protocol::updated(&self.config, &event["session"], self.beta, false)?;
                if !self.audio.is_empty()
                    && audio::settings(&config, self.beta)?.format
                        != audio::settings(&self.config, self.beta)?.format
                {
                    return Err(invalid(
                        "audio.input.format",
                        "clear or commit audio before changing format",
                    ));
                }
                self.config = config;
                self.idle = None;
                self.arm_idle();
                self.session.lock().await.system =
                    self.config["instructions"].as_str().map(str::to_owned);
                Ok(vec![
                    json!({"type":if self.beta && self.transcription {"transcription_session.updated"} else {"session.updated"},"session":self.config}),
                ])
            }
            "conversation.item.create" => {
                let item = protocol::parse_item(
                    &event["item"],
                    &audio::settings(&self.config, self.beta)?.format,
                )?;
                let item_id = event["item"].get("id").map_or_else(
                    || Ok(id("item")),
                    |v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| invalid("item.id", "id must be a string"))
                    },
                )?;
                let position = match event.get("previous_item_id") {
                    None | Some(Value::Null) => Position::End,
                    Some(v) if v == "root" => Position::Start,
                    Some(v) => Position::After {
                        item_id: v
                            .as_str()
                            .ok_or_else(|| invalid("previous_item_id", "must be a string"))?
                            .into(),
                    },
                };
                let mut session = self.session.lock().await;
                session.apply_with_item_id(
                    SessionOp::Insert {
                        position,
                        item: item.clone(),
                    },
                    item_id.clone(),
                )?;
                let index = session.index_of(&item_id)?;
                let previous = index.checked_sub(1).map(|i| session.items[i].id.clone());
                drop(session);
                let mut events = self.item_events(&item_id, &item, previous);
                if matches!(&item,Item::Message {content,..} if content.iter().any(|p|matches!(p,Part::Audio {..})))
                    && (self.transcription
                        || audio::settings(&self.config, self.beta)?.transcription)
                {
                    events.push(json!({"type":"conversation.item.input_audio_transcription.failed","item_id":item_id,"content_index":0,"error":{"type":"server_error","code":"transcription_not_configured","message":"No Transcriber configured for audio input"}}));
                }
                Ok(events)
            }
            "conversation.item.retrieve" => {
                let item_id = string(event, "item_id")?;
                let session = self.session.lock().await;
                let index = session.index_of(item_id)?;
                let live = self
                    .active
                    .as_ref()
                    .and_then(|r| r.output.iter().find(|o| o.id == item_id));
                let (item, status) = live.map_or((&session.items[index].item, "completed"), |o| {
                    (&o.item, o.status)
                });
                Ok(vec![
                    json!({"type":"conversation.item.retrieved","item":protocol::wire_item(item_id,item,status,self.beta)}),
                ])
            }
            "conversation.item.delete" => {
                let item_id = string(event, "item_id")?;
                self.session.lock().await.apply(SessionOp::Delete {
                    item_id: item_id.into(),
                })?;
                Ok(vec![
                    json!({"type":"conversation.item.deleted","item_id":item_id}),
                ])
            }
            "conversation.item.truncate" => {
                let item_id = string(event, "item_id")?;
                let content_index = event["content_index"]
                    .as_u64()
                    .ok_or_else(|| invalid("content_index", "must be a non-negative integer"))?
                    as usize;
                let end = event["audio_end_ms"]
                    .as_u64()
                    .filter(|n| *n <= u32::MAX as u64)
                    .ok_or_else(|| {
                        invalid("audio_end_ms", "must be a non-negative 32-bit integer")
                    })? as u32;
                let mut session = self.session.lock().await;
                let index = session.index_of(item_id)?;
                let Item::Message {
                    role: Role::Assistant,
                    content,
                } = &session.items[index].item
                else {
                    return Err(invalid(
                        "item_id",
                        "only assistant audio messages can be truncated",
                    ));
                };
                let Some(Part::Audio { format, data }) = content.get(content_index) else {
                    return Err(invalid(
                        "content_index",
                        "only audio content can be truncated, not text",
                    ));
                };
                let mut bytes = audio::decode(data, format)?;
                let bytes_per_ms = if format == "pcm16" { 48 } else { 8 };
                let keep = end as usize * bytes_per_ms;
                if keep > bytes.len() {
                    return Err(invalid(
                        "audio_end_ms",
                        "cannot truncate beyond audio duration",
                    ));
                }
                bytes.truncate(keep);
                session.apply(SessionOp::Truncate {
                    item_id: item_id.into(),
                    content_index,
                    keep_chars: None,
                    audio_end_ms: Some(end),
                })?;
                let Item::Message { content, .. } = &mut session.items[index].item else {
                    unreachable!()
                };
                let Part::Audio { data, .. } = &mut content[content_index] else {
                    unreachable!()
                };
                *data = STANDARD.encode(bytes);
                Ok(vec![
                    json!({"type":"conversation.item.truncated","item_id":item_id,"content_index":content_index,"audio_end_ms":end}),
                ])
            }
            "input_audio_buffer.append" => {
                let settings = audio::settings(&self.config, self.beta)?;
                let activities = self.audio.append(string(event, "audio")?, &settings)?;
                if self.audio.speaking() || !activities.is_empty() || self.active.is_some() {
                    self.idle = None;
                } else if self.idle.is_none() {
                    self.arm_idle();
                }
                let mut events = vec![];
                for activity in activities {
                    match activity {
                        audio::Activity::Started { item_id, start_ms } => {
                            events.push(json!({"type":"input_audio_buffer.speech_started","item_id":item_id,"audio_start_ms":start_ms}));
                            if settings.vad.as_ref().is_some_and(|v| v.interrupt_response)
                                && self.active.is_some()
                            {
                                events.extend(self.finish(Some(StopReason::Cancelled), None).await);
                            }
                        }
                        audio::Activity::Stopped {
                            item_id,
                            end_ms,
                            audio,
                        } => {
                            events.push(json!({"type":"input_audio_buffer.speech_stopped","item_id":item_id,"audio_end_ms":end_ms}));
                            events.extend(self.commit(item_id, audio).await?);
                            if settings.vad.as_ref().is_some_and(|v| v.create_response)
                                && !self.transcription
                            {
                                match self.start(&json!({})).await {
                                    Ok(e) => events.extend(e),
                                    Err(e) => events.push(protocol::error(
                                        e,
                                        event.get("event_id").and_then(Value::as_str),
                                        None,
                                    )),
                                };
                            }
                        }
                    }
                }
                if !self.audio.speaking() && self.active.is_none() && self.idle.is_none() {
                    self.arm_idle();
                }
                Ok(events)
            }
            "input_audio_buffer.commit" => {
                let settings = audio::settings(&self.config, self.beta)?;
                let (id, bytes) = self.audio.commit(&settings)?;
                self.commit(id, bytes).await
            }
            "input_audio_buffer.clear" => {
                self.audio.clear();
                Ok(vec![json!({"type":"input_audio_buffer.cleared"})])
            }
            "response.create" => {
                if self.transcription {
                    return Err(invalid(
                        "type",
                        "transcription sessions cannot generate responses",
                    ));
                }
                self.start(event.get("response").unwrap_or(&json!({})))
                    .await
            }
            "response.cancel" => {
                let active = self
                    .active
                    .as_ref()
                    .ok_or_else(|| invalid("response_id", "no response is in progress"))?;
                if let Some(requested) = event.get("response_id") {
                    if requested != &active.id {
                        return Err(invalid(
                            "response_id",
                            "response_id does not match active response",
                        ));
                    }
                }
                self.session.lock().await.apply(SessionOp::Cancel)?;
                Ok(self.finish(Some(StopReason::Cancelled), None).await)
            }
            "output_audio_buffer.clear" => Err(GatewayError::unsupported(
                "output_audio_buffer.clear is only available over WebRTC and SIP, not WebSocket",
            )
            .with_param("type")),
            _ => Err(invalid("type", "unknown Realtime client event type")),
        }
    }
    async fn commit(
        &mut self,
        item_id: String,
        bytes: Vec<u8>,
    ) -> Result<Vec<Value>, GatewayError> {
        let settings = audio::settings(&self.config, self.beta)?;
        let item = Item::Message {
            role: Role::User,
            content: vec![Part::Audio {
                format: settings.format,
                data: STANDARD.encode(bytes),
            }],
        };
        let mut session = self.session.lock().await;
        let previous = session.items.last().map(|i| i.id.clone());
        session.apply_with_item_id(
            SessionOp::Insert {
                position: Position::End,
                item: item.clone(),
            },
            item_id.clone(),
        )?;
        drop(session);
        let mut events = vec![
            json!({"type":"input_audio_buffer.committed","item_id":item_id,"previous_item_id":previous}),
        ];
        events.extend(self.item_events(&item_id, &item, previous));
        if settings.transcription || self.transcription {
            events.push(json!({"type":"conversation.item.input_audio_transcription.failed","item_id":item_id,"content_index":0,"error":{"type":"server_error","code":"transcription_not_configured","message":"No Transcriber configured for audio input"}}));
        }
        Ok(events)
    }
    async fn start(&mut self, overrides: &Value) -> Result<Vec<Value>, GatewayError> {
        let usage = self.usage.as_ref().map(|u| u.child("realtime"));
        if let Some(scope) = &usage {
            scope.session(self.session.lock().await.id.0.clone(), "explicit");
        }
        let result = self.start_accounted(overrides, usage.clone()).await;
        if let (Err(error), Some(scope)) = (&result, usage) { scope.finished(error.status()); }
        result
    }
    async fn start_accounted(&mut self, overrides: &Value, usage: Option<crate::usage::UsageHandle>) -> Result<Vec<Value>, GatewayError> {
        if self.active.is_some() {
            return Err(invalid(
                "response",
                "conversation already has an active response",
            ));
        }
        let config = protocol::updated(&self.config, overrides, self.beta, true)?;
        let session = self.session.lock().await;
        let items = if let Some(input) = overrides.get("input") {
            let input = input
                .as_array()
                .ok_or_else(|| invalid("response.input", "input must be an array"))?;
            let mut items = vec![];
            for item in input {
                if item["type"] == "item_reference" {
                    items.push(
                        session.items[session.index_of(string(item, "id")?)?]
                            .item
                            .clone(),
                    );
                } else {
                    items.push(protocol::parse_item(
                        item,
                        &audio::settings(&config, self.beta)?.format,
                    )?);
                }
            }
            items
        } else {
            session.items()
        };
        let mut turn = protocol::turn(&config, self.beta, items)?;
        turn.session = Some(session.id.clone());
        turn.tape = self.tape.clone();
        super::driver::account_request(&mut turn, usage.clone(), overrides, true);
        if let Some(scope) = &usage {
            // The turn as the backend sees it: the realtime log has no single wire request.
            scope.log_request(|| json!({"type":"realtime.turn","conversation_id":self.conversation_id,
                "model":turn.requested_model,"instructions":turn.system,"tools":turn.tools,"items":turn.items}));
        }
        self.turn_usage = usage;
        drop(session);
        let has_audio=turn.items.iter().any(|i|matches!(i,Item::Message {content,..} if content.iter().any(|p|matches!(p,Part::Audio {..}))));
        let has_image=turn.items.iter().any(|i|matches!(i,Item::Message {content,..} if content.iter().any(|p|matches!(p,Part::Image {..}))));
        let caps = self.gateway.backend.capabilities();
        let stream = if has_audio && !caps.audio_in {
            Err(GatewayError::unsupported(
                "backend lacks audio_in capability; configure an audio-capable backend",
            ))
        } else if has_image && !caps.vision {
            Err(GatewayError::unsupported("backend lacks vision capability"))
        } else {
            let gateway = self.gateway.clone();
            let stream = async_stream::try_stream! {
                tracing::info!(protocol = "realtime", transport = "websocket", items = turn.items.len(), "gateway turn");
                let mut upstream=gateway.run(turn).await?;
                while let Some(event)=upstream.next().await {yield event?;}
            };
            Ok(Box::pin(stream) as super::TurnStream)
        };
        let (stream, error) = match stream {
            Ok(s) => (s, None),
            Err(e) => (
                Box::pin(futures::stream::empty()) as super::TurnStream,
                Some(e),
            ),
        };
        let previous_item_id = self
            .session
            .lock()
            .await
            .items
            .last()
            .map(|item| item.id.clone());
        let active = response::Response {
            id: id("resp"),
            conversation_id: if overrides["conversation"] == "none" {
                None
            } else {
                Some(self.conversation_id.clone())
            },
            metadata: overrides.get("metadata").cloned().unwrap_or(Value::Null),
            output: vec![],
            usage: Default::default(),
            stream,
            beta: self.beta,
            previous_item_id,
            config,
        };
        let events = vec![
            json!({"type":"response.created","response":active.object("in_progress",Value::Null)}),
        ];
        let (tx, rx) = tokio::sync::watch::channel(false);
        self.session.lock().await.running = Some(tx);
        self.cancel = Some(rx);
        self.active = Some(active);
        self.idle = None;
        if let Some(e) = error {
            let mut events = events;
            events.extend(self.finish(None, Some(e)).await);
            Ok(events)
        } else {
            Ok(events)
        }
    }
    async fn sync_output(&mut self) {
        let Some(active) = &mut self.active else {
            return;
        };
        if active.conversation_id.is_none() {
            return;
        }
        let mut session = self.session.lock().await;
        for output in &mut active.output {
            if output.synced_final || (output.published && !output.finished) {
                continue;
            }
            let exists = session.items.iter().any(|i| i.id == output.id);
            // A client may delete an announced output item during generation.
            if output.published && !exists {
                continue;
            }
            let op = if exists {
                SessionOp::Edit {
                    item_id: output.id.clone(),
                    item: output.item.clone(),
                }
            } else {
                SessionOp::Append {
                    items: vec![output.item.clone()],
                }
            };
            let outcome = if matches!(op, SessionOp::Append { .. }) {
                session.apply_with_item_id(op, output.id.clone())
            } else {
                session.apply(op)
            };
            if let Err(e) = outcome {
                tracing::warn!(error=%e,"Realtime history update failed");
            } else {
                output.published = true;
                output.synced_final = output.finished;
            }
        }
    }
    async fn finish(
        &mut self,
        stop: Option<StopReason>,
        error: Option<GatewayError>,
    ) -> Vec<Value> {
        self.cancel = None;
        let Some(active) = self.active.as_mut() else {
            return vec![];
        };
        let usage = self.turn_usage.take();
        if let Some(usage) = &usage {
            usage.tokens(active.usage.input_tokens.into(), active.usage.cached_input_tokens.into(),
                active.usage.output_tokens.into(), active.usage.reasoning_tokens.into());
            if matches!(stop, Some(StopReason::Cancelled)) { usage.stop("cancelled"); }
            if let Some(error) = &error { usage.engine_error(error.anthropic_type()); }
            usage.finished(if error.is_some() { 500 } else { 200 });
        }
        let events = active.finish_turn(stop, error);
        if let Some(usage) = usage {
            if let Some(done) = events.iter().rev().find(|e| e["type"] == "response.done") {
                usage.log_response(|| done["response"].clone());
            }
        }
        self.sync_output().await;
        self.active.take();
        let mut session = self.session.lock().await;
        session.running = None;
        drop(session);
        self.arm_idle();
        events
    }
    fn arm_idle(&mut self) {
        if self.audio.speaking() || self.active.is_some() {
            self.idle = None;
            return;
        }
        self.idle = audio::settings(&self.config, self.beta)
            .ok()
            .and_then(|s| s.vad)
            .and_then(|v| v.idle_timeout_ms)
            .map(|ms| tokio::time::Instant::now() + std::time::Duration::from_millis(ms));
    }
}
