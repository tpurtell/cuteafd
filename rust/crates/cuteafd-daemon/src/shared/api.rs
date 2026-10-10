//! Common API policy for every serving family.
use cuteafd_api::openai::{auth::{ApiKey, Auth}, health::HealthWitness, ModelProfile};
use anyhow::Context;
use std::{path::PathBuf, sync::Arc};

#[derive(Debug, Clone, clap::Args)]
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
    /// Request accounting (metadata only; measured at no C1 cost).
    #[arg(long, default_value = "on", value_parser = ["on", "off"])]
    pub usage: Option<String>,
    /// File containing the host console secret (never an API credential).
    #[arg(long)]
    pub console_secret_file: Option<PathBuf>,
    /// Always mark the console cookie Secure.
    #[arg(long)]
    pub console_cookie_secure: bool,
    /// Serve Anthropic Messages, OpenAI Responses (HTTP and WebSocket) and
    /// Realtime beside /v1/chat/completions, over the same engine.
    #[arg(long, default_value = "on", value_parser = ["on", "off"])]
    pub gateway: String,
    /// Answer the official Claude Code and Codex model ids (and any other id)
    /// with the served model, and list them in /v1/models.
    #[arg(long, default_value = "on", value_parser = ["on", "off"])]
    pub official_model_names: String,
    /// Replace or extend the advertised official ids (text or JSON list).
    #[arg(long)]
    pub official_model_names_file: Option<PathBuf>,
    /// Hosted web search for the gateway: none, exa (EXA_API_KEY), or searxng=URL.
    #[arg(long, default_value = "none")]
    pub gateway_search: String,
}
impl Default for ApiArgs {
    fn default() -> Self {
        Self { api_key_file: None, enable_bench: false, usage_dir: None, usage: Some("on".into()), console_secret_file: None,
            console_cookie_secure: false, gateway: "on".into(), official_model_names: "on".into(),
            official_model_names_file: None, gateway_search: "none".into() }
    }
}
pub(crate) struct ApiPolicy {
    key: Option<ApiKey>,
    bench: bool,
    usage: Option<Arc<cuteafd_usage::Store>>,
    gate: cuteafd_api::console_gate::ConsoleGate,
    gateway: Option<GatewayPolicy>,
}
/// The serving gateway's model names and hosted search.
#[derive(Clone)]
struct GatewayPolicy {
    official: bool,
    names_file: Option<String>,
    search: Option<Arc<dyn cuteafd_api::gateway::SearchProvider>>,
}
impl ApiArgs {
    pub fn load(&self) -> anyhow::Result<ApiPolicy> {
        let key = self.api_key_file.as_deref().map(ApiKey::from_file).transpose()?;
        anyhow::ensure!(!self.enable_bench || key.is_some(), "--enable-bench requires --api-key-file");
        let usage = if self.usage.as_deref() == Some("off") { None } else { Some(cuteafd_usage::Store::open(self.usage_dir.as_deref())?) };
        let gate = self.console_secret_file.as_deref().map(|p| cuteafd_api::console_gate::ConsoleGate::from_file(p, self.console_cookie_secure)).transpose()?.unwrap_or_else(cuteafd_api::console_gate::ConsoleGate::locked);
        let gateway = if self.gateway == "on" {
            let names_file = self.official_model_names_file.as_deref()
                .map(|p| std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))).transpose()?;
            let search: Option<Arc<dyn cuteafd_api::gateway::SearchProvider>> = match self.gateway_search.as_str() {
                "none" => None,
                "exa" => Some(Arc::new(cuteafd_api::gateway::search::Exa::new(
                    std::env::var("EXA_API_KEY").context("--gateway-search exa needs EXA_API_KEY")?)?)),
                other => match other.strip_prefix("searxng=") {
                    Some(url) => Some(Arc::new(cuteafd_api::gateway::search::Searxng::new(url)?)),
                    None => anyhow::bail!("--gateway-search must be none, exa, or searxng=URL"),
                },
            };
            Some(GatewayPolicy { official: self.official_model_names == "on" || names_file.is_some(), names_file, search })
        } else { None };
        Ok(ApiPolicy { key, bench: self.enable_bench, usage, gate, gateway })
    }
}
impl ApiPolicy {
    /// Install the gateway mount on a family's profile (`--gateway on`).
    /// `snapshot` holds the tokenizer `count_tokens` uses.
    pub(crate) fn serve(&self, profile: ModelProfile, snapshot: &std::path::Path) -> anyhow::Result<ModelProfile> {
        let mut profile = profile_with_health(profile);
        if let Some(policy) = &self.gateway {
            use cuteafd_api::gateway::ModelMap;
            let mut models = if policy.official { ModelMap::official_names(profile.id.clone()) } else { ModelMap::single(profile.id.clone()) };
            if let Some(text) = &policy.names_file { models.apply_names_file(text).map_err(anyhow::Error::msg)?; }
            let snapshot = snapshot.join("tokenizer.json").is_file().then(|| snapshot.to_path_buf());
            profile.gateway = Some(Arc::new(cuteafd_api::openai::GatewayMount { models, snapshot,
                options: cuteafd_api::openai::engine::EngineOptions {
                    json_schema: cuteafd_api::openai::engine::probe_json_schema(&profile) },
                search: policy.search.clone(),
                gate: self.bench.then(|| cuteafd_bench::http::gate(cuteafd_bench::Bench::global())) }));
        }
        Ok(profile)
    }
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
fn profile_with_health(mut profile: ModelProfile) -> ModelProfile {
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
        let app = ApiPolicy { key: Some(ApiKey::new("secret").unwrap()), bench: false, usage: None, gate: cuteafd_api::console_gate::ConsoleGate::locked(), gateway: None }.app(
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
    fn usage_is_on_unless_turned_off() {
        assert!(ApiArgs::default().load().unwrap().usage.is_some());
        assert!(ApiArgs { usage: Some("off".into()), ..Default::default() }.load().unwrap().usage.is_none());
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
    #[tokio::test]
    async fn gateway_routes_mount_by_default_with_every_key_form_and_turn_off() {
        use axum::{body::Body, http::{Request, StatusCode}};
        use clap::Parser;
        use tower::ServiceExt;
        let snapshot = tempfile::tempdir().unwrap();
        for (gateway, mounted) in [(None, true), (Some("off"), false)] {
            let mut argv = vec!["cuteafd", "serve-qwen4", "--snapshot", "/model", "--native-lib", "/native.so"];
            if let Some(value) = gateway { argv.extend(["--gateway", value]); }
            let crate::cli::Commands::ServeQwen4(args) = crate::cli::Cli::try_parse_from(argv).unwrap().command else { panic!() };
            assert_eq!(args.api.gateway, gateway.unwrap_or("on"));
            assert_eq!(args.api.official_model_names, "on", "official names default on in serving");
            let mut policy = args.api.load().unwrap();
            policy.key = Some(ApiKey::new("secret").unwrap());
            let profile = cuteafd_api::openai::ModelProfile::new("org/served", cuteafd_api::openai::ModelEncoding::DeepseekV41);
            let profile = policy.serve(profile, snapshot.path()).unwrap();
            assert_eq!(profile.gateway.is_some(), mounted);
            let (tx, _rx) = tokio::sync::mpsc::channel(1);
            let router = cuteafd_api::openai::router_for_model(tx, cuteafd_api::openai::NativeLimits::default(),
                Arc::new(std::sync::Mutex::new(serde_json::Value::Null)), std::time::Duration::from_secs(1),
                cuteafd_api::openai::ConsoleHub::disabled(), profile);
            let app = policy.app(router, cuteafd_api::openai::ConsoleHub::disabled());
            // Claude Code's x-api-key is accepted; a wrong key answers in Anthropic shape.
            let count = |key: &'static str| Request::post("/v1/messages/count_tokens").header("x-api-key", key)
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"claude-sonnet-5","messages":[{"role":"user","content":"hi"}]}"#)).unwrap();
            let response = app.clone().oneshot(count("wrong")).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
            assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap()["type"], "error");
            let status = app.clone().oneshot(count("secret")).await.unwrap().status();
            assert_eq!(status, if mounted { StatusCode::OK } else { StatusCode::NOT_FOUND });
            let models = app.clone().oneshot(Request::get("/v1/models").header("authorization", "Bearer secret")
                .body(Body::empty()).unwrap()).await.unwrap();
            let listing: serde_json::Value = serde_json::from_slice(&axum::body::to_bytes(models.into_body(), 1 << 20).await.unwrap()).unwrap();
            assert_eq!(listing["data"][0]["id"], "org/served", "the served model stays first");
            assert!(listing["data"][0]["max_context_tokens"].is_u64());
            assert_eq!(listing["data"].as_array().unwrap().iter().any(|m| m["id"] == "claude-opus-5"), mounted);
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
