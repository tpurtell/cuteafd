//! The in-server benchmark: `/bench`, `/v1/bench/*` and `cuteafd bench`.
//!
//! An explicitly enabled, keyed server runs benchmarks on request. The runner drives
//! the server through its own OpenAI API over loopback (so SSE and the
//! scheduler are measured), holds every other inference request off with
//! 503 + Retry-After while it runs, and stores each report in SQLite. The
//! mandatory baseline (basic card + quick quality) runs once per server
//! lifetime and configuration fingerprint and is shared by later runs.
pub mod baseline;
pub mod cli;
pub mod client;
pub mod context;
pub mod fabric;
pub mod fidelity;
pub mod fidelity_cli;
pub mod fidelity_dataset;
pub mod fidelity_match;
pub mod fidelity_precision;
pub mod fidelity_rows;
pub mod http;
pub mod panels;
pub mod profiles;
pub mod publish;
pub mod quality;
pub mod reference;
pub mod request;
pub mod render;
pub mod report;
pub mod runner;
pub mod sample;
pub mod server;
pub mod smoke;
pub mod store;
pub mod sweep;
pub mod text;

pub use runner::{Bench, UsageToggle};

/// Compatibility entry point: no benchmark routes or lockout by default.
/// Serving explicitly mounts `http::mount` only after validating its key.
pub fn app(router: axum::Router, console: std::sync::Arc<cuteafd_api::openai::ConsoleHub>) -> axum::Router {
    let _ = console;
    // Serving opts in explicitly with a key via the daemon's API policy.
    router
}

/// The serving process is shutting down (`reason`, e.g. "SIGTERM"): an active
/// benchmark stops at once and is saved as interrupted with that reason, not as
/// a panel failure when the listener closes under it. Serve loops call this
/// from their graceful-shutdown future before the listener stops.
pub fn stopping(reason: &str) {
    runner::Bench::global().stopping(reason);
}

/// Set once a serve loop owns shutdown (see [`shutdown_signal`]).
static GRACEFUL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

async fn terminate() -> &'static str {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! { _ = tokio::signal::ctrl_c() => "SIGINT", _ = term.recv() => "SIGTERM" }
}

/// Waits until the benchmark has saved its interrupted run (bounded).
async fn settle() {
    let bench = runner::Bench::global();
    for _ in 0..50 {
        if bench.active().is_none() { return; }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

/// A graceful-shutdown future for serve loops that drain on SIGTERM/SIGINT:
/// it marks the benchmark interrupted and lets it save before the listener closes.
pub async fn shutdown_signal() {
    GRACEFUL.store(true, std::sync::atomic::Ordering::Relaxed);
    let reason = terminate().await;
    stopping(reason);
    settle().await;
}

/// Records that the API now accepts requests on `listener` (readiness time
/// and the runner's loopback address). A server without its own graceful
/// shutdown still exits at once on SIGTERM/SIGINT, as before, but a running
/// benchmark is saved as interrupted first instead of left "running".
pub fn ready(listener: &tokio::net::TcpListener) {
    if let Ok(addr) = listener.local_addr() {
        context::mark_ready(addr);
    }
    // The opt-in expert route log records serving waves only (start-up waves carry synthetic routes).
    cuteafd_transport::route_log::mark_serving();
    tokio::spawn(async {
        let reason = terminate().await;
        if GRACEFUL.load(std::sync::atomic::Ordering::Relaxed) { return; }
        stopping(reason);
        settle().await;
        std::process::exit(if reason == "SIGINT" { 130 } else { 143 });
    });
}
