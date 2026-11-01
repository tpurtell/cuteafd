//! CX-01: 64 simultaneous client disconnects through the HTTP/API layer.
//!
//! The W64 c64 wedge (2026-10-31 06:54:48Z) cut 64 in-flight streams at once
//! and the coordinator froze. This test drives the same mass cancel through
//! the API/admission layer with a fake engine and asserts every slot, permit
//! and ticket comes back. It cannot reproduce the engine-side transport reset
//! (see `v2/cuteafd/W64-F3-CANCEL-ANALYSIS-2026-10-31.md` §5 W1); a green run
//! only clears the API layer.
use super::*;
use std::time::Duration;
use tower::ServiceExt;

fn stream_request() -> axum::http::Request<Body> {
    axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "disabled"},
            "max_tokens": 16,
            "stream": true,
        }).to_string()))
        .unwrap()
}

fn ready() -> InferenceChunk {
    InferenceChunk::Ready { system_fingerprint: None,
        prompt_usage: PromptUsage { prompt_tokens: 4, prompt_cache_hit_tokens: 0 } }
}

fn text(content: &str) -> InferenceChunk {
    InferenceChunk::Text { content: content.into(), content_tokens: 1 }
}

fn finish() -> InferenceChunk {
    InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }
}

/// Drive `n` streaming requests, drop every client body in the same tick, then
/// assert the worker sees all `n` event senders closed, the admission queue is
/// empty, and a follow-up request is admitted and completes.
async fn drop_storm(app: &Router, tx: &mpsc::Sender<NativeRequest>,
    rx: &mut mpsc::Receiver<NativeRequest>, n: usize) {
    let worker = async {
        let mut senders = Vec::with_capacity(n);
        for _ in 0..n {
            let job = rx.recv().await.expect("engine job");
            job.events.send(Ok(ready())).unwrap();
            job.events.send(Ok(text("a"))).unwrap();
            job.events.send(Ok(text("b"))).unwrap();
            senders.push(job.events);
        }
        tokio::time::timeout(Duration::from_secs(5), async {
            while !senders.iter().all(|sender| sender.is_closed()) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }).await.expect("worker did not observe every client disconnect");
        let job = rx.recv().await.expect("follow-up job");
        job.events.send(Ok(ready())).unwrap();
        job.events.send(Ok(finish())).unwrap();
    };

    let clients = async {
        let mut tasks = Vec::with_capacity(n);
        for _ in 0..n {
            let app = app.clone();
            tasks.push(tokio::spawn(async move {
                let response = app.oneshot(stream_request()).await.unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                response
            }));
        }
        let mut responses = Vec::with_capacity(n);
        for task in tasks { responses.push(task.await.unwrap()); }
        assert_eq!(tx.capacity(), tx.max_capacity(),
            "admission queue not empty after {n} admissions");
        // Every client body is dropped in the same tick.
        drop(responses);
        let response = app.clone().oneshot(stream_request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let body = std::str::from_utf8(&bytes).unwrap();
        assert!(body.ends_with("data: [DONE]\n\n"), "follow-up did not complete: {body}");
    };

    tokio::join!(worker, clients);
}

#[tokio::test]
async fn sixty_four_streams_dropped_at_once_release_everything() {
    let (tx, mut rx) = mpsc::channel::<NativeRequest>(128);
    let app = router_for_model(tx.clone(), NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(25), ConsoleHub::disabled(), ModelProfile::default());
    drop_storm(&app, &tx, &mut rx, 64).await;
}

#[tokio::test]
async fn forty_eight_then_sixty_four_drops() {
    let (tx, mut rx) = mpsc::channel::<NativeRequest>(128);
    let app = router_for_model(tx.clone(), NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(25), ConsoleHub::disabled(), ModelProfile::default());
    drop_storm(&app, &tx, &mut rx, 48).await;
    drop_storm(&app, &tx, &mut rx, 64).await;
}
