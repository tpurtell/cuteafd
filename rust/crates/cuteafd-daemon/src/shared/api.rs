//! Common API policy for every serving family.
use cuteafd_api::openai::{auth::{ApiKey, Auth}, health::HealthWitness, ModelProfile};
use std::{path::PathBuf, sync::Arc};

#[derive(Debug, Clone, Default, clap::Args)]
pub(crate) struct ApiArgs {
    /// Require a bearer key from this file on /v1/* (health stays open).
    #[arg(long, env = "API_KEY_FILE")]
    pub api_key_file: Option<PathBuf>,
    /// Mount benchmark controls and lockout; requires --api-key-file.
    #[arg(long, env = "CUTEAFD_ENABLE_BENCH")]
    pub enable_bench: bool,
    /// Usage history directory; absent uses an in-memory store.
    #[arg(long, env = "CUTEAFD_USAGE_DIR")]
    pub usage_dir: Option<PathBuf>,
    /// Set to off to disable all request accounting.
    #[arg(long, default_value = "on", value_parser = ["on", "off"])]
    pub usage: Option<String>,
    /// File containing the host console secret (never an API credential).
    #[arg(long)]
    pub console_secret_file: Option<PathBuf>,
    /// Always mark the console cookie Secure.
    #[arg(long)]
    pub console_cookie_secure: bool,
}
pub(crate) struct ApiPolicy {
    key: Option<ApiKey>,
    bench: bool,
    usage: Option<Arc<cuteafd_usage::Store>>,
    gate: cuteafd_api::console_gate::ConsoleGate,
}
impl ApiArgs {
    pub fn load(&self) -> anyhow::Result<ApiPolicy> {
        let key = self.api_key_file.as_deref().map(ApiKey::from_file).transpose()?;
        anyhow::ensure!(!self.enable_bench || key.is_some(), "--enable-bench requires --api-key-file");
        let usage = if self.usage.as_deref() == Some("off") { None } else { Some(cuteafd_usage::Store::open(self.usage_dir.as_deref())?) };
        let gate = self.console_secret_file.as_deref().map(|p| cuteafd_api::console_gate::ConsoleGate::from_file(p, self.console_cookie_secure)).transpose()?.unwrap_or_else(cuteafd_api::console_gate::ConsoleGate::locked);
        Ok(ApiPolicy { key, bench: self.enable_bench, usage, gate })
    }
}
impl ApiPolicy {
    pub(crate) fn gateway_auth(&self) -> cuteafd_api::gateway::auth::GatewayAuth {
        cuteafd_api::gateway::auth::GatewayAuth { key: self.key.clone() }
    }
    pub fn app(self, router: axum::Router, hub: Arc<cuteafd_api::openai::ConsoleHub>) -> axum::Router {
        let (router, internal) = if self.bench {
            let bench = cuteafd_bench::Bench::global();
            bench.set_console(hub);
            bench.set_api_key(self.key.clone().expect("validated benchmark key"));
            let witness = bench.clone();
            let internal: Arc<dyn Fn(&str, &axum::http::HeaderMap) -> bool + Send + Sync> = Arc::new(move |path, headers| {
                matches!(path, "/v1/chat/completions" | "/v1/completions" | "/v1/models" | "/v1/stats")
                    && cuteafd_api::openai::auth::bearer(headers).is_some_and(|token| witness.accepts_internal(token))
            });
            (cuteafd_bench::http::mount(router, bench), Some(internal))
        } else { (router, None) };
        let router = if let Some(store) = &self.usage { cuteafd_usage::http::mount(router, store.clone(), self.gate.clone()) } else { router };
        let router = self.gate.mount(router);
        tracing::info!("console: protected views unlock through the launcher's link");
        let router = router.layer(axum::middleware::from_fn_with_state(Auth { key: self.key, internal },
            cuteafd_api::openai::auth::require_key));
        if let Some(store) = self.usage {
            router.layer(axum::middleware::from_fn_with_state(cuteafd_api::usage::Middleware::new(store), cuteafd_api::usage::track))
        } else { router }
    }
}
pub(crate) fn catch_scheduler_panic(work: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<()> {
    // The worker's state is discarded after unwind; no partially mutated engine is reused.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
        .unwrap_or_else(|_| Err(anyhow::anyhow!("scheduler panicked")))
}
pub(crate) async fn watch_scheduler(worker: tokio::task::JoinHandle<anyhow::Result<()>>) {
    let reason = match worker.await {
        Ok(Ok(())) => "scheduler stopped".into(),
        Ok(Err(error)) => format!("scheduler stopped: {error:#}"),
        Err(error) => format!("scheduler stopped: {error}"),
    };
    tracing::error!(%reason);
    cuteafd_transport::health::record_failure(reason);
    // Stay pending so select keeps polling the HTTP server after worker death.
    std::future::pending::<()>().await;
}
pub(crate) fn profile(mut profile: ModelProfile) -> ModelProfile {
    profile.engine_health = Some(HealthWitness(Arc::new(cuteafd_transport::health::failure_reason)));
    profile
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn default_has_no_benchmark_routes_and_keyed_auth_is_outermost() {
        use axum::{body::Body, http::StatusCode};
        use tower::ServiceExt;
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let app = ApiArgs::default().load().unwrap().app(cuteafd_api::openai::router(tx),
            cuteafd_api::openai::ConsoleHub::disabled());
        assert_eq!(app.oneshot(axum::http::Request::get("/v1/bench/status").body(Body::empty()).unwrap())
            .await.unwrap().status(), StatusCode::NOT_FOUND);
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let app = ApiPolicy { key: Some(ApiKey::new("secret").unwrap()), bench: false, usage: None, gate: cuteafd_api::console_gate::ConsoleGate::locked() }.app(
            cuteafd_api::openai::router(tx), cuteafd_api::openai::ConsoleHub::disabled());
        assert_eq!(app.clone().oneshot(axum::http::Request::post("/v1/chat/completions")
            .body(Body::from("malformed")).unwrap()).await.unwrap().status(), StatusCode::UNAUTHORIZED);
        assert_eq!(app.oneshot(axum::http::Request::get("/health").body(Body::empty()).unwrap())
            .await.unwrap().status(), StatusCode::SERVICE_UNAVAILABLE);
    }
    #[tokio::test]
    async fn usage_off_preserves_chat_headers_and_body() {
        use axum::{body::Body, http::Request};
        use tower::ServiceExt;
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let original = cuteafd_api::openai::router(tx);
        let policy = ApiArgs { usage: Some("off".into()), ..Default::default() }.load().unwrap();
        assert!(policy.usage.is_none());
        let tracked = policy.app(original.clone(), cuteafd_api::openai::ConsoleHub::disabled());
        let make = || Request::post("/v1/chat/completions").header("content-type", "application/json").body(Body::from("{}" )).unwrap();
        let baseline = original.oneshot(make()).await.unwrap();
        let result = tracked.oneshot(make()).await.unwrap();
        assert_eq!(baseline.status(), result.status());
        assert_eq!(baseline.headers(), result.headers());
        assert_eq!(axum::body::to_bytes(baseline.into_body(), 4096).await.unwrap(), axum::body::to_bytes(result.into_body(), 4096).await.unwrap());
    }
    #[test]
    fn every_family_accepts_api_policy_options() {
        use clap::Parser;
        use crate::cli::{Cli, Commands};
        for serve in ["serve-glm", "serve-glmf", "serve-mimo", "serve-qwen4", "serve-dsv4", "serve-native"] {
            let mut argv = vec!["cuteafd", serve, "--snapshot", "/model", "--native-lib", "/native.so",
                "--api-key-file", "/key", "--enable-bench"];
            if serve == "serve-native" { argv.extend(["--peers", "127.0.0.1:9000"]); }
            let cli = Cli::try_parse_from(argv).unwrap();
            let api = match cli.command {
                Commands::ServeGlm(args) => args.api,
                Commands::ServeGlmf(args) => args.api,
                Commands::ServeMimo(args) => args.api,
                Commands::ServeQwen4(args) => args.api,
                Commands::ServeDsv4(args) => args.api,
                Commands::ServeNative(args) => args.api,
                _ => panic!("expected serving"),
            };
            assert_eq!(api.api_key_file, Some(PathBuf::from("/key")));
            assert!(api.enable_bench);
        }
    }
    #[test]
    fn scheduler_panic_becomes_terminal_error() {
        let stopped = catch_scheduler_panic(|| panic!("private scheduler payload"));
        assert_eq!(stopped.unwrap_err().to_string(), "scheduler panicked");
        assert!(catch_scheduler_panic(|| Ok(())).is_ok());
        assert_eq!(catch_scheduler_panic(|| anyhow::bail!("worker failed")).unwrap_err().to_string(), "worker failed");
    }
    #[test]
    fn benchmark_controls_require_a_key() {
        assert!(ApiArgs { enable_bench: true, api_key_file: None, ..Default::default() }.load().is_err());
        assert!(ApiArgs::default().load().is_ok());
    }
}
