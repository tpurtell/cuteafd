//! The sweep's measurement window opens exactly once, after the first warm-up
//! wave and before the first timed request (smoke's `window.open` hook).
mod common;
use common::fake_engine;
use cuteafd_api::openai::{router_with_console, ConsoleHub, NativeLimits, NativeRequest};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn window_opens_after_warm_up_before_timed_requests() {
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    // Count requests on their way to the fake engine.
    let (counted, forward) = mpsc::channel::<NativeRequest>(16);
    let seen = Arc::new(AtomicUsize::new(0));
    let tally = seen.clone();
    tokio::spawn(async move {
        let mut receive = receive;
        while let Some(job) = receive.recv().await {
            tally.fetch_add(1, Ordering::SeqCst);
            if counted.send(job).await.is_err() {
                return;
            }
        }
    });
    fake_engine(forward);
    let router = router_with_console(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(5), ConsoleHub::disabled());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, router.into_make_service_with_connect_info::<std::net::SocketAddr>()).await.unwrap();
    });
    let (opened_at, sweep, refused) = tokio::task::spawn_blocking(move || {
        let opened_at = Arc::new(Mutex::new(Vec::new()));
        let record = opened_at.clone();
        let counter = seen.clone();
        let sweep = cuteafd_bench::sweep::run_windowed(&base, None, &[2, 1], true, false, &[], &Default::default(), false, cuteafd_bench::sweep::TOKENS, Some(Box::new(move || {
            record.lock().unwrap().push(counter.load(Ordering::SeqCst));
            Ok(())
        }))).unwrap();
        let opened_at = opened_at.lock().unwrap().clone();
        // A refused open fails the sweep before any timed request: only the
        // discovery and the one C1 warm-up reach the engine.
        let before = seen.load(Ordering::SeqCst);
        let refused = cuteafd_bench::sweep::run_windowed(&base, None, &[1], false, false, &[], &Default::default(), false, cuteafd_bench::sweep::TOKENS,
            Some(Box::new(|| anyhow::bail!("refused"))));
        assert!(format!("{:#}", refused.unwrap_err()).contains("refused"));
        (opened_at, sweep, seen.load(Ordering::SeqCst) - before)
    }).await.unwrap();
    // One call, after exactly the C2 warm-up wave (two requests).
    assert_eq!(opened_at, vec![2]);
    assert_eq!(sweep.points.len(), 3);
    assert_eq!(refused, 1);
}

/// A cold purge runs before every prefill point's warm-up and again before its
/// timed request; the measurement window still opens exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prefill_purge_precedes_each_warm_up_and_timed_request() {
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let (counted, forward) = mpsc::channel::<NativeRequest>(16);
    let seen = Arc::new(AtomicUsize::new(0));
    let tally = seen.clone();
    tokio::spawn(async move {
        let mut receive = receive;
        while let Some(job) = receive.recv().await {
            tally.fetch_add(1, Ordering::SeqCst);
            if counted.send(job).await.is_err() {
                return;
            }
        }
    });
    fake_engine(forward);
    let router = router_with_console(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(5), ConsoleHub::disabled());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, router.into_make_service_with_connect_info::<std::net::SocketAddr>()).await.unwrap();
    });
    let (purged_at, opened_at, sweep) = tokio::task::spawn_blocking(move || {
        let purged_at = Mutex::new(Vec::new());
        let opened_at = Arc::new(Mutex::new(Vec::new()));
        let (record, counter) = (opened_at.clone(), seen.clone());
        let purge = || { purged_at.lock().unwrap().push(seen.load(Ordering::SeqCst)); Ok(()) };
        let options = cuteafd_bench::sweep::PrefillOptions { repeat: true, purge: Some(&purge) };
        let sweep = cuteafd_bench::sweep::run_windowed(&base, None, &[], false, false, &[256, 512], &options, false,
            cuteafd_bench::sweep::TOKENS, Some(Box::new(move || {
                record.lock().unwrap().push(counter.load(Ordering::SeqCst));
                Ok(())
            }))).unwrap();
        let purged = purged_at.lock().unwrap().clone();
        let opened = opened_at.lock().unwrap().clone();
        (purged, opened, sweep)
    }).await.unwrap();
    // Two calibration requests, then per length: purge, warm-up, purge, timed.
    assert_eq!(purged_at, vec![2, 3, 4, 5]);
    assert_eq!(opened_at, vec![3]);
    assert_eq!(sweep.prefill.len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn phase_logging_is_bounded_and_separate_from_inference() {
    let (queue, receive) = mpsc::channel::<NativeRequest>(1);
    let router = router_with_console(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(5), ConsoleHub::disabled());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, router.into_make_service_with_connect_info::<std::net::SocketAddr>()).await.unwrap();
    });
    tokio::task::spawn_blocking(move || {
        let agent = ureq::agent();
        assert_eq!(agent.post(&format!("{base}/v1/fabric/phase"))
            .send_json(serde_json::json!({"name":"decode","timed":true,"ports":[]})).unwrap().status(),204);
        let oversized = agent.post(&format!("{base}/v1/fabric/phase"))
            .send_json(serde_json::json!({"name":"x".repeat(129),"timed":true,"ports":[]}));
        assert!(matches!(oversized,Err(ureq::Error::Status(400,_))));
    }).await.unwrap();
    assert!(receive.is_empty());
}
