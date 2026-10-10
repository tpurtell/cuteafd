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
pub mod fidelity;
pub mod fidelity_cli;
pub mod fidelity_dataset;
pub mod fidelity_match;
pub mod fidelity_rows;
pub mod http;
pub mod panels;
pub mod profiles;
pub mod publish;
pub mod reference;
pub mod render;
pub mod report;
pub mod runner;
pub mod sample;
pub mod server;
pub mod smoke;
pub mod store;
pub mod text;

pub use runner::{Bench, UsageToggle};

/// Compatibility entry point: no benchmark routes or lockout by default.
/// Serving explicitly mounts `http::mount` only after validating its key.
pub fn app(router: axum::Router, console: std::sync::Arc<cuteafd_api::openai::ConsoleHub>) -> axum::Router {
    let _ = console;
    // Serving opts in explicitly with a key via the daemon's API policy.
    router
}

/// Records that the API now accepts requests on `listener` (readiness time
/// and the runner's loopback address).
pub fn ready(listener: &tokio::net::TcpListener) {
    if let Ok(addr) = listener.local_addr() {
        context::mark_ready(addr);
    }
}
