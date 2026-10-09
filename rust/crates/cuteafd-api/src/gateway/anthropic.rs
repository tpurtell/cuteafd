//! Anthropic Messages wire protocol over the protocol-neutral gateway.
use std::{convert::Infallible, sync::Arc, time::Duration};

use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::post,
    Json, Router,
};
use futures::StreamExt;
use serde_json::{json, Value};

use super::{record::Tape, Gateway, GatewayError};

mod render;
mod request;
#[cfg(test)]
mod tests;

pub fn routes(gateway: Arc<Gateway>) -> Router {
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(gateway)
}

fn body(value: Result<Json<Value>, JsonRejection>) -> Result<Value, GatewayError> {
    value.map(|Json(v)| v).map_err(|error| {
        let kind = if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
            super::ErrorKind::RequestTooLarge
        } else {
            super::ErrorKind::InvalidRequest
        };
        GatewayError::new(kind, error.body_text())
    })
}

async fn count_tokens(
    State(gateway): State<Arc<Gateway>>,
    tape: Tape,
    value: Result<Json<Value>, JsonRejection>,
) -> Response {
    let result = async {
        let (mut turn, _) = request::parse(&body(value)?, false)?;
        turn.tape = tape;
        gateway.count_tokens(turn).await
    }
    .await;
    with_request_id(match result {
        Ok(n) => Json(json!({"input_tokens": n})).into_response(),
        Err(error) => error.anthropic_response(),
    })
}

async fn messages(
    State(gateway): State<Arc<Gateway>>,
    tape: Tape,
    value: Result<Json<Value>, JsonRejection>,
) -> Response {
    let result = async {
        let (mut turn, streaming) = request::parse(&body(value)?, true)?;
        turn.tape = tape;
        let model = turn.requested_model.clone();
        let stream = gateway.run(turn).await?;
        Ok::<_, GatewayError>((model, streaming, stream))
    }
    .await;
    let (model, streaming, mut stream) = match result {
        Ok(result) => result,
        Err(error) => return with_request_id(error.anthropic_response()),
    };
    let mut renderer = render::Renderer::new(model);
    if !streaming {
        while let Some(event) = stream.next().await {
            match event.and_then(|event| renderer.push(event)) {
                Ok(_) => (),
                Err(error) => return with_request_id(error.anthropic_response()),
            }
            if renderer.done() {
                return with_request_id(Json(renderer.message()).into_response());
            }
        }
        return with_request_id(
            GatewayError::upstream("backend stream ended without a stop reason")
                .anthropic_response(),
        );
    }
    // Starting errors remain HTTP errors. The first actual event is not buffered
    // beyond this check; subsequent errors are Anthropic SSE error frames.
    let first = match stream.next().await {
        Some(Ok(event)) => event,
        Some(Err(error)) => return with_request_id(error.anthropic_response()),
        None => {
            return with_request_id(
                GatewayError::upstream("backend stream ended without a stop reason")
                    .anthropic_response(),
            )
        }
    };
    let frames = async_stream::stream! {
        yield Ok::<_, Infallible>(sse(renderer.start()));
        yield Ok(sse(json!({"type":"ping"})));
        let mut pending = Some(first);
        loop {
            let next = if let Some(event) = pending.take() { Some(Ok(event)) } else {
                // Keep the same pending read alive when a ping is due, so dropping
                // a polled future never loses a backend event.
                let next = stream.next();
                tokio::pin!(next);
                loop {
                    tokio::select! {
                        event = &mut next => break event,
                        _ = tokio::time::sleep(Duration::from_secs(15)) => yield Ok(sse(json!({"type":"ping"}))),
                    }
                }
            };
            let result = match next {
                Some(event) => event.and_then(|event| renderer.push(event)),
                None => Err(GatewayError::upstream("backend stream ended without a stop reason")),
            };
            match result {
                Ok(events) => for event in events { yield Ok(sse(event)); },
                Err(error) => { yield Ok(sse(error.anthropic_body())); break; }
            }
            if renderer.done() { break; }
        }
    };
    let mut response = Sse::new(frames).into_response();
    response
        .headers_mut()
        .insert("x-accel-buffering", "no".parse().unwrap());
    with_request_id(response)
}

fn sse(value: Value) -> Event {
    Event::default()
        .event(value["type"].as_str().unwrap_or("error"))
        .data(value.to_string())
}

pub(super) fn with_request_id(mut response: Response) -> Response {
    if !response.headers().contains_key("request-id") {
        let id = format!("req_{}", uuid::Uuid::new_v4().simple());
        response
            .headers_mut()
            .insert("request-id", id.parse().unwrap());
    }
    response
}
