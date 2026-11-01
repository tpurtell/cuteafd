//! CX-01: client cancellations through the HTTP/API admission layer.
//!
//! The W64 c64 wedge cut 64 in-flight streams at once. These tests drive the
//! same mass cancel through the real router with a fake engine. The queue is
//! deliberately small: a 128-slot queue never queues 64 sends, so
//! `try_reserve_owned()` always takes the fast path and the waiter/timeout
//! paths are never entered. With a one-slot queue the second request waits and
//! the third is rejected, which is the path a cancel storm exercises.
//!
//! They cover cancels before the engine sends `Ready` and cancels while the
//! request is queued, and assert every slot, permit and ticket comes back.
//! They cannot reproduce the engine-side transport reset; a green run only
//! clears the API layer.
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

fn finish() -> InferenceChunk {
    InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }
}

/// Wait until `condition` holds, yielding to the runtime. Bounded so a broken
/// admission path fails the test instead of hanging it.
async fn until(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !condition() { tokio::task::yield_now().await; }
    }).await.expect("condition did not hold");
}

fn app(queue: mpsc::Sender<NativeRequest>, wait: Duration) -> Router {
    router_for_model(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        wait, ConsoleHub::disabled(), ModelProfile::default())
}

async fn body_json(response: Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// A request that is admitted but whose engine never sends `Ready` is cancelled
/// before `Ready`; a request that is queued behind it is cancelled while it
/// waits. Both must release their slot, permit and ticket, and a follow-up
/// request must be admitted and complete.
#[tokio::test]
async fn pre_ready_and_queued_cancels_release_every_slot() {
    let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
    let app = app(tx.clone(), Duration::from_secs(60));
    let (release, release_rx) = tokio::sync::oneshot::channel::<()>();

    let worker = tokio::spawn(async move {
        // Hold the queue full until the clients have been dropped, so the
        // second request really waits instead of taking the fast path.
        let _ = release_rx.await;
        let first = rx.recv().await.expect("first job");
        until(|| first.events.is_closed()).await;
        let follow_up = rx.recv().await.expect("follow-up job");
        follow_up.events.send(Ok(ready())).unwrap();
        follow_up.events.send(Ok(finish())).unwrap();
    });

    // Client 1 fills the one-slot queue and waits for `Ready`.
    let first = tokio::spawn({ let app = app.clone(); async move { app.oneshot(stream_request()).await } });
    until(|| tx.capacity() == 0).await;

    // Client 2 cannot reserve a slot, so it takes the waiter path.
    let second = tokio::spawn({ let app = app.clone(); async move { app.oneshot(stream_request()).await } });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!second.is_finished(), "the second request must be queued, not rejected");

    // Client 3 finds the waiter set full and is rejected with the reason.
    let third = app.clone().oneshot(stream_request()).await.unwrap();
    assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(third).await;
    assert_eq!(body["error"]["message"], "request queue is full");

    // Cancel the queued request, then the pre-`Ready` one.
    second.abort(); let _ = second.await;
    first.abort(); let _ = first.await;
    let _ = release.send(());

    let response = app.clone().oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
    assert!(std::str::from_utf8(&bytes).unwrap().ends_with("data: [DONE]\n\n"));
    assert_eq!(tx.capacity(), tx.max_capacity(), "admission queue not empty");
    worker.await.unwrap();
}

/// A waiter whose budget expires is rejected with a reason distinct from a full
/// queue, so a client can tell "retry later" from "the queue is saturated".
#[tokio::test]
async fn expired_wait_rejects_with_a_timeout_reason() {
    let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
    let app = app(tx.clone(), Duration::from_millis(50));
    let (release, release_rx) = tokio::sync::oneshot::channel::<()>();

    let worker = tokio::spawn(async move {
        let _ = release_rx.await;
        let first = rx.recv().await.expect("first job");
        until(|| first.events.is_closed()).await;
    });

    let first = tokio::spawn({ let app = app.clone(); async move { app.oneshot(stream_request()).await } });
    until(|| tx.capacity() == 0).await;

    let second = app.clone().oneshot(stream_request()).await.unwrap();
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(second).await;
    assert_eq!(body["error"]["message"], "request queue wait budget expired");

    first.abort(); let _ = first.await;
    let _ = release.send(());
    worker.await.unwrap();
}

/// 64 simultaneous disconnects through a queue small enough that most requests
/// are rejected and the rest queue. Every admitted job must observe its client
/// gone and a follow-up request must be admitted and complete.
#[tokio::test]
async fn sixty_four_drops_with_a_small_queue_release_everything() {
    let (tx, mut rx) = mpsc::channel::<NativeRequest>(8);
    let app = app(tx.clone(), Duration::from_secs(60));
    let (dropped, dropped_rx) = tokio::sync::oneshot::channel::<()>();

    let worker = tokio::spawn(async move {
        // Wait until every client body has been dropped, then drain the queue.
        let _ = dropped_rx.await;
        let mut senders = Vec::new();
        while let Ok(job) = rx.try_recv() { senders.push(job.events); }
        assert_eq!(senders.len(), 8, "every admitted job must be in the queue");
        tokio::time::timeout(Duration::from_secs(5), async {
            while !senders.iter().all(|sender| sender.is_closed()) {
                tokio::task::yield_now().await;
            }
        }).await.expect("worker did not observe every client disconnect");
        let follow_up = rx.recv().await.expect("follow-up job");
        follow_up.events.send(Ok(ready())).unwrap();
        follow_up.events.send(Ok(finish())).unwrap();
    });

    let clients = async {
        let mut tasks = Vec::with_capacity(64);
        for _ in 0..64 {
            let app = app.clone();
            tasks.push(tokio::spawn(async move { app.oneshot(stream_request()).await }));
        }
        // The queue and the waiter set are both full: 8 admitted, 8 queued,
        // 48 rejected. Every client body is dropped in the same tick.
        until(|| tx.capacity() == 0).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        for task in &tasks { task.abort(); }
        for task in tasks { let _ = task.await; }
        let _ = dropped.send(());
        let response = app.clone().oneshot(stream_request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        assert!(std::str::from_utf8(&bytes).unwrap().ends_with("data: [DONE]\n\n"));
    };

    let _ = tokio::join!(clients, worker);
}
