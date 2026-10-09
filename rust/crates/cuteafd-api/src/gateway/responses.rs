//! OpenAI Responses HTTP/SSE and websocket front end over the shared turn driver.
use std::{convert::Infallible, sync::Arc};

use axum::{
    extract::{
        rejection::JsonRejection,
        ws::{Message, WebSocket},
        Path, Query, State, WebSocketUpgrade,
    },
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};

use super::{
    record::Tape,
    session::{Snapshot, StoredItem},
    turn::{Item, Part, Role, TurnEvent},
    Gateway, GatewayError,
};

mod parse;
mod render;
#[cfg(test)]
mod tests;

pub fn routes(gateway: Arc<Gateway>) -> Router {
    Router::new()
        .route("/v1/responses", post(create).get(websocket))
        .route("/v1/responses/compact", post(compact))
        .route("/v1/responses/input_tokens", post(input_tokens))
        .route("/v1/responses/:id", get(retrieve).delete(delete))
        .route("/v1/responses/:id/cancel", post(cancel))
        .route("/v1/responses/:id/input_items", get(input_items))
        .with_state(gateway)
}

fn body(body: Result<Json<Value>, JsonRejection>) -> Result<Value, GatewayError> {
    body.map(|Json(v)| v)
        .map_err(|e| GatewayError::invalid(e.body_text()))
}

async fn create(
    State(gateway): State<Arc<Gateway>>,
    tape: Tape,
    request: Result<Json<Value>, JsonRejection>,
) -> Response {
    let mut p = match body(request).and_then(|v| parse::parse(&gateway, v, None)) {
        Ok(p) => p,
        Err(e) => return e.openai_response(),
    };
    p.turn.tape = tape;
    let stream = match gateway.run(p.turn.clone()).await {
        Ok(s) => s,
        Err(e) => return e.openai_response(),
    };
    let streaming = p.wire["stream"].as_bool().unwrap_or(false);
    let events = run(gateway, p, stream);
    if streaming {
        let sse = events.map(|v| {
            Ok::<_, Infallible>(
                Event::default()
                    .event(v["type"].as_str().unwrap_or("error"))
                    .json_data(v)
                    .unwrap(),
            )
        });
        Sse::new(sse)
            .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
            .into_response()
    } else {
        futures::pin_mut!(events);
        let mut response = None;
        while let Some(event) = events.next().await {
            if event["type"] == "response.failed" {
                return GatewayError::upstream(
                    event["response"]["error"]["message"]
                        .as_str()
                        .unwrap_or("generation failed"),
                )
                .openai_response();
            }
            if event["type"] == "response.completed" || event["type"] == "response.incomplete" {
                response = Some(event["response"].clone());
            }
        }
        response
            .map(|v| Json(v).into_response())
            .unwrap_or_else(|| {
                GatewayError::internal("missing terminal response").openai_response()
            })
    }
}

fn snapshot(p: &parse::Parsed, fold: &render::Fold) -> Arc<Snapshot> {
    let mut items: Vec<StoredItem> = p
        .new_items
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, item)| StoredItem {
            id: p
                .input
                .get(i)
                .and_then(|v| v["id"].as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| render::id("item")),
            item,
        })
        .collect();
    for (i, v) in fold.response["output"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        if let Ok(parsed) = fold.history_item(i) {
            for item in parsed {
                items.push(StoredItem {
                    id: v["id"].as_str().unwrap().into(),
                    item,
                });
            }
        }
    }
    Arc::new(Snapshot {
        parent: p.parent.clone(),
        system: p.turn.system.clone(),
        items,
    })
}

fn store(gateway: &Gateway, p: &parse::Parsed, fold: &render::Fold) -> Arc<Snapshot> {
    let snap = snapshot(p, fold);
    if p.store {
        let mut document = fold.response.clone();
        // Input listing is stored alongside the response, not exposed on GET.
        document["_input_items"] = json!(p
            .input
            .iter()
            .map(|v| {
                let mut v = v.clone();
                if v.get("id").is_none() {
                    v["id"] = json!(render::id("item"));
                }
                if v.get("type").is_none() {
                    v["type"] = json!("message");
                }
                v
            })
            .collect::<Vec<_>>());
        gateway.sessions.put_response_document(
            fold.response["id"].as_str().unwrap().into(),
            snap.clone(),
            document,
        );
    }
    snap
}

fn run(
    gateway: Arc<Gateway>,
    p: parse::Parsed,
    mut stream: super::TurnStream,
) -> impl Stream<Item = Value> + Send {
    async_stream::stream! {
        let mut fold = render::Fold::new(&p);
        for event in std::mem::take(&mut fold.events) { yield event; }
        while let Some(event) = stream.next().await {
            if let Err(error) = event.and_then(|e| fold.accept(e)) { fold.fail(&error); }
            if fold.terminal {
                if fold.response["status"] != "failed" { store(&gateway, &p, &fold); }
                for event in std::mem::take(&mut fold.events) { yield event; }
                return;
            }
            for event in std::mem::take(&mut fold.events) { yield event; }
        }
        fold.fail(&GatewayError::upstream("backend stream ended before Done"));
        for event in fold.events { yield event; }
    }
}

async fn retrieve(
    State(gateway): State<Arc<Gateway>>,
    Path(id): Path<String>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Response {
    if q.get("stream").is_some_and(|v| v == "true") {
        return GatewayError::unsupported("stored response stream replay is not supported")
            .with_param("stream")
            .openai_response();
    }
    match gateway.sessions.response_document(&id) {
        Some(mut v) => {
            v.as_object_mut().unwrap().remove("_input_items");
            Json(v).into_response()
        }
        None => GatewayError::not_found(format!("response '{id}' not found")).openai_response(),
    }
}

async fn delete(State(gateway): State<Arc<Gateway>>, Path(id): Path<String>) -> Response {
    if gateway.sessions.delete_response(&id) {
        Json(json!({"id":id,"object":"response.deleted","deleted":true})).into_response()
    } else {
        GatewayError::not_found(format!("response '{id}' not found")).openai_response()
    }
}

async fn cancel(State(gateway): State<Arc<Gateway>>, Path(id): Path<String>) -> Response {
    if gateway.sessions.response(&id).is_none() {
        return GatewayError::not_found(format!("response '{id}' not found")).openai_response();
    }
    GatewayError::invalid("only background responses can be cancelled")
        .with_param("response_id")
        .openai_response()
}

async fn input_items(
    State(gateway): State<Arc<Gateway>>,
    Path(id): Path<String>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Response {
    let Some(v) = gateway.sessions.response_document(&id) else {
        return GatewayError::not_found(format!("response '{id}' not found")).openai_response();
    };
    let mut items = v["_input_items"].as_array().cloned().unwrap_or_default();
    let order = q.get("order").map(String::as_str).unwrap_or("desc");
    if order != "asc" && order != "desc" {
        return GatewayError::invalid("order must be asc or desc")
            .with_param("order")
            .openai_response();
    }
    if order == "desc" {
        items.reverse();
    }
    if let Some(after) = q.get("after") {
        let Some(i) = items.iter().position(|v| v["id"] == after.as_str()) else {
            return GatewayError::invalid("after item not found")
                .with_param("after")
                .openai_response();
        };
        items.drain(..=i);
    }
    let limit = match q.get("limit") {
        None => 20,
        Some(s) => match s.parse::<usize>() {
            Ok(n) if (1..=100).contains(&n) => n,
            _ => {
                return GatewayError::invalid("limit must be between 1 and 100")
                    .with_param("limit")
                    .openai_response()
            }
        },
    };
    let more = items.len() > limit;
    items.truncate(limit);
    Json(json!({"object":"list","first_id":items.first().map(|v| &v["id"]),"last_id":items.last().map(|v| &v["id"]),"has_more":more,"data":items})).into_response()
}

async fn input_tokens(
    State(gateway): State<Arc<Gateway>>,
    tape: Tape,
    request: Result<Json<Value>, JsonRejection>,
) -> Response {
    let mut p = match body(request).and_then(|v| parse::parse(&gateway, v, None)) {
        Ok(p) => p,
        Err(e) => return e.openai_response(),
    };
    p.turn.tape = tape;
    match gateway.count_tokens(p.turn).await {
        Ok(n) => Json(json!({"object":"response.input_tokens","input_tokens":n})).into_response(),
        Err(e) => e.openai_response(),
    }
}

async fn compact(
    State(gateway): State<Arc<Gateway>>,
    tape: Tape,
    request: Result<Json<Value>, JsonRejection>,
) -> Response {
    let mut p = match body(request).and_then(|v| parse::parse(&gateway, v, None)) {
        Ok(p) => p,
        Err(e) => return e.openai_response(),
    };
    p.turn.tape = tape;
    p.turn.tools.clear();
    p.turn.hosted = Default::default();
    p.turn.items.push(Item::Message { role:Role::User, content:vec![Part::text("Summarize this conversation for continuation. Preserve the user's requirements, decisions, tool results and pending work. Return only the compact context, not an answer to the task.")] });
    let mut stream = match gateway.run(p.turn).await {
        Ok(s) => s,
        Err(e) => return e.openai_response(),
    };
    let mut text = String::new();
    let mut usage = Value::Null;
    let mut done = false;
    while let Some(event) = stream.next().await {
        match event {
            Ok(TurnEvent::TextDelta { text: delta }) => text.push_str(&delta),
            Ok(TurnEvent::Usage { usage: u }) => {
                usage = json!({"input_tokens":u.input_tokens,"input_tokens_details":{"cached_tokens":u.cached_input_tokens},"output_tokens":u.output_tokens,"output_tokens_details":{"reasoning_tokens":u.reasoning_tokens},"total_tokens":u64::from(u.input_tokens)+u64::from(u.output_tokens)})
            }
            Ok(TurnEvent::Done { stop }) => {
                if stop != super::turn::StopReason::EndTurn {
                    return GatewayError::upstream("compaction did not complete successfully")
                        .openai_response();
                }
                done = true;
                break;
            }
            Err(e) => return e.openai_response(),
            _ => {}
        }
    }
    if !done || text.is_empty() {
        return GatewayError::upstream("compaction returned no summary").openai_response();
    }
    Json(json!({"id":render::id("resp"),"object":"response.compaction","created_at":render::now(),
        "output":[{"id":render::id("cmp"),"type":"compaction","encrypted_content":parse::encode("compaction",&text)}],"usage":usage})).into_response()
}

async fn websocket(
    State(gateway): State<Arc<Gateway>>,
    tape: Tape,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| websocket_loop(socket, gateway, tape))
        .into_response()
}

async fn websocket_loop(mut socket: WebSocket, gateway: Arc<Gateway>, tape: Tape) {
    let mut last: Option<(String, Arc<Snapshot>)> = None;
    while let Some(Ok(message)) = socket.next().await {
        let Message::Text(text) = message else {
            if matches!(message, Message::Close(_)) {
                break;
            }
            continue;
        };
        let request: Result<Value, GatewayError> = serde_json::from_str(&text)
            .map_err(|_| GatewayError::invalid("invalid websocket JSON"));
        let parsed = request.and_then(|mut v| {
            if v["type"] != "response.create" {
                return Err(GatewayError::unsupported(
                    "only response.create websocket events are supported",
                )
                .with_param("type"));
            }
            v.as_object_mut().unwrap().remove("type");
            parse::parse(
                &gateway,
                v,
                last.as_ref().map(|(id, s)| (id.as_str(), s.clone())),
            )
        });
        let mut p = match parsed {
            Ok(p) => p,
            Err(e) => {
                if socket.send(Message::Text(json!({"type":"error","status":e.status(),"error":e.openai_body()["error"]}).to_string())).await.is_err() { break; }
                continue;
            }
        };
        p.turn.tape = tape.clone();
        let mut stream = match gateway.run(p.turn.clone()).await {
            Ok(s) => s,
            Err(e) => {
                if socket.send(Message::Text(json!({"type":"error","status":e.status(),"error":e.openai_body()["error"]}).to_string())).await.is_err() { break; }
                continue;
            }
        };
        let mut fold = render::Fold::new(&p);
        loop {
            for event in std::mem::take(&mut fold.events) {
                if socket.send(Message::Text(event.to_string())).await.is_err() {
                    return;
                }
            }
            if fold.terminal {
                break;
            }
            tokio::select! {
                event = stream.next() => {
                    match event {
                        Some(event) => if let Err(e) = event.and_then(|e| fold.accept(e)) { fold.fail(&e); },
                        None => fold.fail(&GatewayError::upstream("backend stream ended before Done")),
                    }
                    if fold.terminal && fold.response["status"] != "failed" { last = Some((fold.response["id"].as_str().unwrap().into(), store(&gateway,&p,&fold))); }
                },
                message = socket.next() => match message {
                    Some(Ok(Message::Ping(bytes))) => { if socket.send(Message::Pong(bytes)).await.is_err() { return; } },
                    Some(Ok(Message::Text(_))) => { if socket.send(Message::Text(json!({"type":"error","status":400,"error":{"type":"invalid_request_error","code":"response_in_progress","message":"a response is already in progress","param":null}}).to_string())).await.is_err() { return; } },
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return,
                    _ => {},
                }
            }
        }
    }
}
