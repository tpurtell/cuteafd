//! Common API policy for every serving family.
use cuteafd_api::openai::{auth::{ApiKey, Auth}, health::HealthWitness, ModelProfile};
use anyhow::Context;
use std::{future::IntoFuture, path::PathBuf, sync::Arc};

/// Admission errors terminate only this request, never the scheduler's receive loop.
pub(crate) fn admission_result<T>(events: &tokio::sync::mpsc::UnboundedSender<
    Result<cuteafd_api::openai::InferenceChunk, cuteafd_api::openai::NativeFailure>>,
    result: anyhow::Result<T>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            let _ = events.send(Err(cuteafd_api::openai::NativeFailure::from_admission_error(error)));
            None
        }
    }
}

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ApiArgs {
    /// Require a bearer key from this file on /v1/* (health stays open).
    #[arg(long, env = "API_KEY_FILE")]
    pub api_key_file: Option<PathBuf>,
    /// Mount benchmark controls and lockout; the console cookie (or --api-key-file's key) runs them.
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
    /// Browser origin (scheme://host[:port]) allowed to open Responses and
    /// Realtime sockets without the API key; repeatable. Same-origin pages,
    /// keyed requests and clients that send no Origin are always allowed.
    #[arg(long = "gateway-allow-origin")]
    pub gateway_allow_origins: Vec<String>,
}
impl Default for ApiArgs {
    fn default() -> Self {
        Self { api_key_file: None, enable_bench: false, usage_dir: None, usage: Some("on".into()), console_secret_file: None,
            console_cookie_secure: false, gateway: "on".into(), official_model_names: "on".into(),
            official_model_names_file: None, gateway_search: "none".into(), gateway_allow_origins: Vec::new() }
    }
}
pub(crate) struct ApiPolicy {
    key: Option<ApiKey>,
    bench: bool,
    usage: Option<Arc<cuteafd_usage::Store>>,
    gate: cuteafd_api::console_gate::ConsoleGate,
    gateway: Option<GatewayPolicy>,
    /// Verifies a running benchmark's token for usage's bench exemption.
    bench_token: Option<Arc<dyn Fn(&str) -> bool + Send + Sync>>,
    /// The engine's active count for /usage (read from the console snapshot).
    live: Option<cuteafd_usage::http::LiveCount>,
}
/// The serving gateway's model names and hosted search.
#[derive(Clone)]
struct GatewayPolicy {
    official: bool,
    allowed_origins: Vec<String>,
    names_file: Option<String>,
    search: Option<Arc<dyn cuteafd_api::gateway::SearchProvider>>,
}
impl ApiArgs {
    pub fn load(&self) -> anyhow::Result<ApiPolicy> {
        let key = self.api_key_file.as_deref().map(ApiKey::from_file).transpose()?;
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
            Some(GatewayPolicy { official: self.official_model_names == "on" || names_file.is_some(), names_file, search,
                allowed_origins: self.gateway_allow_origins.clone() })
        } else { None };
        if let Some(store) = &usage {
            // The server's own secrets never reach the full log, wherever a client pastes them.
            let mut secrets = vec![];
            for file in [&self.api_key_file, &self.console_secret_file].into_iter().flatten() {
                if let Ok(text) = std::fs::read_to_string(file) { secrets.push(text.trim().to_owned()); }
            }
            store.log.set_secrets(secrets);
        }
        Ok(ApiPolicy { key, bench: self.enable_bench, usage, gate, gateway, bench_token: None, live: None })
    }
}
impl ApiPolicy {
    /// Install the gateway mount on a family's profile (`--gateway on`).
    /// `snapshot` holds the tokenizer `count_tokens` uses.
    pub(crate) fn serve(&self, profile: ModelProfile, snapshot: &std::path::Path) -> anyhow::Result<ModelProfile> {
        cuteafd_bench::context::set_thinking_off(profile.encoding.supports_thinking_off());
        let profile = profile.with_tokenizer(snapshot)?;
        let mut profile = profile_with_health(profile);
        let mut drafter = cuteafd_bench::context::drafter().unwrap_or(cuteafd_bench::report::DrafterLayout::NONE);
        if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") { drafter.bind_visible_gpus(&visible); }
        tracing::info!(layout = %drafter.label(), gpu_uuids = ?drafter.gpu_uuids, "resolved serving drafter");
        cuteafd_bench::context::set_drafter(drafter);
        profile.stats_diagnostics = Some(collect_stats);
        if let Some(policy) = &self.gateway {
            use cuteafd_api::gateway::ModelMap;
            let mut models = if policy.official { ModelMap::official_names(profile.id.clone()) } else { ModelMap::single(profile.id.clone()) };
            if let Some(text) = &policy.names_file { models.apply_names_file(text).map_err(anyhow::Error::msg)?; }
            let snapshot = Some(snapshot.to_path_buf());
            profile.gateway = Some(Arc::new(cuteafd_api::openai::GatewayMount { models, snapshot,
                options: cuteafd_api::openai::engine::EngineOptions {
                    json_schema: cuteafd_api::openai::engine::probe_json_schema(&profile) },
                search: policy.search.clone(),
                gate: self.bench.then(|| cuteafd_bench::http::gate(cuteafd_bench::Bench::global())),
                origins: cuteafd_api::gateway::OriginPolicy { key: self.key.clone(), allowed: policy.allowed_origins.clone() } }));
        }
        Ok(profile)
    }
    pub(crate) fn gateway_auth(&self) -> cuteafd_api::gateway::auth::GatewayAuth {
        cuteafd_api::gateway::auth::GatewayAuth { key: self.key.clone() }
    }
    pub fn app(self, router: axum::Router, hub: Arc<cuteafd_api::openai::ConsoleHub>) -> axum::Router {
        // Token text reaches only viewers holding the console cookie.
        hub.set_gate(self.gate.clone());
        let snapshot_hub = hub.clone();
        let mut this = self;
        this.live = Some(Arc::new(move || {
            serde_json::from_str::<serde_json::Value>(&snapshot_hub.snapshot()).ok()?["g"]["active"].as_u64()
        }));
        let self_ = this;
        let (router, internal) = if self_.bench {
            let bench = cuteafd_bench::Bench::global();
            bench.set_console(hub);
            if let Some(store) = &self_.usage { bench.set_usage(usage_toggle(store.clone())); }
            if let Some(key) = &self_.key { bench.set_api_key(key.clone()); }
            bench.set_console_gate(self_.gate.clone());
            let witness = bench.clone();
            let internal: Arc<dyn Fn(&str, &axum::http::HeaderMap) -> bool + Send + Sync> = Arc::new(move |path, headers| {
                matches!(path, "/v1/chat/completions" | "/v1/completions" | "/v1/models" | "/v1/stats")
                    && cuteafd_api::openai::auth::bearer(headers).is_some_and(|token| witness.accepts_internal(token))
            });
            let verify = bench.clone();
            // Learn the served configuration in the background once the API answers.
            let discovering = bench.clone();
            let _ = std::thread::Builder::new().name("bench-discover".into()).spawn(move || discovering.discover());
            (cuteafd_bench::http::mount(router, bench), Some((internal, verify)))
        } else {
            (router.route("/bench", axum::routing::get(cuteafd_api::openai::console::bench_off)), None)
        };
        let mut policy = self_;
        let internal = internal.map(|(internal, verify)| {
            policy.bench_token = Some(Arc::new(move |token: &str| verify.accepts_internal(token)));
            internal
        });
        let router = policy.mount_console(router);
        let router = router.layer(axum::middleware::from_fn_with_state(Auth { key: policy.key.clone(), internal },
            cuteafd_api::openai::auth::require_key));
        policy.track(router)
    }
    /// The cookie-gated usage routes, `/usage` and `/console/unlock`.
    pub(crate) fn mount_console(&self, router: axum::Router) -> axum::Router {
        let router = if let Some(store) = &self.usage {
            cuteafd_usage::http::mount_with_live(router, store.clone(), self.gate.clone(), self.live.clone())
        } else { router };
        self.gate.log_location();
        self.gate.mount(router)
    }
    /// The outermost layer: request accounting and full-log capture.
    pub(crate) fn track(self, router: axum::Router) -> axum::Router {
        if let Some(store) = self.usage {
            let mut middleware = cuteafd_api::usage::Middleware::new(store);
            if let Some(verify) = self.bench_token { middleware = middleware.with_bench(verify); }
            router.layer(axum::middleware::from_fn_with_state(middleware, cuteafd_api::usage::track))
        } else { router }
    }
}
fn collect_stats(stats: &mut serde_json::Value) {
    crate::shared::probe::graph_capture_stats(stats);
    stats["drafter"] = serde_json::json!(cuteafd_bench::context::drafter());
}

/// The bench page's include/skip switch for benchmark requests in usage history.
fn usage_toggle(store: Arc<cuteafd_usage::Store>) -> cuteafd_bench::UsageToggle {
    let read = store.clone();
    cuteafd_bench::UsageToggle {
        get: Arc::new(move || read.settings().record_bench),
        set: Arc::new(move |record| {
            let mut settings = store.settings();
            settings.record_bench = record;
            store.update_settings(settings).map_err(|e| e.to_string())
        }),
    }
}
pub(crate) fn catch_scheduler_panic(work: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<()> {
    // The worker's state is discarded after unwind; no partially mutated engine is reused.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
        .unwrap_or_else(|_| Err(anyhow::anyhow!("scheduler panicked")))
}
/// Borrow scheduler request owners from outside `work`, so fatal diagnostics precede channel closure.
/// Publish while the caller still owns the in-flight reply channels.
pub(crate) fn publish_scheduler_failure(error: &anyhow::Error) {
    cuteafd_transport::health::record_failure(format!("scheduler stopped: {error:#}"));
}

pub(crate) fn run_scheduler(work: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<()> {
    run_scheduler_with_health(work, |reason| cuteafd_transport::health::record_failure(reason))
}

fn run_scheduler_with_health(work: impl FnOnce() -> anyhow::Result<()>, publish: impl FnOnce(String)) -> anyhow::Result<()> {
    let result = catch_scheduler_panic(work);
    if let Err(error) = &result {
        let reason = format!("scheduler stopped: {error:#}");
        publish(reason.clone());
        tracing::error!(%reason);
    }
    result
}
pub(crate) async fn watch_scheduler(worker: &mut Option<tokio::task::JoinHandle<anyhow::Result<()>>>) {
    let Some(handle) = worker.as_mut() else { std::future::pending::<()>().await; return };
    let result = handle.await;
    worker.take();
    let reason = match result {
        Ok(Ok(())) => "scheduler stopped".into(),
        Ok(Err(error)) => format!("scheduler stopped: {error:#}"),
        Err(error) => format!("scheduler stopped: {error}"),
    };
    tracing::error!(%reason);
    cuteafd_transport::health::record_failure(reason);
    // Stay pending so select keeps polling the HTTP server after worker death.
    std::future::pending::<()>().await;
}
/// Stop monitoring streams immediately, while letting in-flight completions finish.
pub(crate) async fn serve_http(listener: tokio::net::TcpListener, router: axum::Router,
    family: &str,
) -> std::io::Result<()> {
    serve_http_until(listener, router, family, cuteafd_bench::shutdown_signal(),
        std::time::Duration::from_secs(10)).await
}

async fn serve_http_until(listener: tokio::net::TcpListener, router: axum::Router,
    family: &str, shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    grace: std::time::Duration,
) -> std::io::Result<()> {
    let drain = cuteafd_api::shutdown::Drain::default();
    let signal_drain = drain.clone();
    let (stopping, stopped) = tokio::sync::oneshot::channel();
    let signal = async move {
        shutdown.await;
        signal_drain.stop_monitors();
        let _ = stopping.send(());
    };
    let http = axum::serve(listener, drain.layer(router)
        .into_make_service_with_connect_info::<std::net::SocketAddr>())
        .with_graceful_shutdown(signal).into_future();
    let server = async {
        http.await?;
        drain.sessions_finished().await;
        Ok(())
    };
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result,
        _ = stopped => match tokio::time::timeout(grace, &mut server).await {
            Ok(result) => result,
            Err(_) => {
                drain.stop_requests();
                // Dropping Axum's listener future does not cancel spawned connection tasks.
                // The body middleware cancels those tasks' work before producer shutdown.
                let result = tokio::time::timeout(std::time::Duration::from_secs(1), &mut server).await;
                tracing::warn!(family, ?grace, "HTTP completion grace elapsed; cancelled remaining requests");
                result.unwrap_or(Ok(()))
            }
        }
    }
}

/// Only called after the listener stops: a broken peer must not prevent trace drain.
pub(crate) async fn finish_scheduler(worker: &mut Option<tokio::task::JoinHandle<anyhow::Result<()>>>, family: &str) -> anyhow::Result<()> {
    let Some(worker) = worker.take() else { return Ok(()) };
    finish_scheduler_with_timeout(worker, family, std::time::Duration::from_secs(10)).await
}

async fn finish_scheduler_with_timeout(worker: tokio::task::JoinHandle<anyhow::Result<()>>, family: &str, timeout: std::time::Duration) -> anyhow::Result<()> {
    match tokio::time::timeout(timeout, worker).await {
        Ok(result) => result.context("serving worker join")?,
        Err(_) => { tracing::warn!(family, ?timeout, "serving worker did not stop; draining available traces"); Ok(()) }
    }
}

fn profile_with_health(mut profile: ModelProfile) -> ModelProfile {
    profile.engine_health = Some(HealthWitness(Arc::new(cuteafd_transport::health::failure_reason)));
    profile
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn every_serving_profile_publishes_draft_totals_and_startup_layout() {
        use axum::{body::Body, http::Request};
        use tower::ServiceExt;
        let snapshot = tempfile::tempdir().unwrap();
        cuteafd_loader::plan::testing::write_kimi_tokenizer(snapshot.path());
        let policy = ApiArgs::default().load().unwrap();
        let profile = cuteafd_api::openai::ModelProfile::new("test", cuteafd_api::openai::ModelEncoding::DeepseekV41);
        let profile = policy.serve(profile, snapshot.path()).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let router = cuteafd_api::openai::router_for_model(tx, Default::default(),
            Arc::new(std::sync::Mutex::new(serde_json::json!({"verify":{"drafted_tokens":99}}))),
            std::time::Duration::from_secs(1), cuteafd_api::openai::ConsoleHub::disabled(), profile);
        let response = router.oneshot(Request::get("/v1/stats").body(Body::empty()).unwrap()).await.unwrap();
        let stats: serde_json::Value = serde_json::from_slice(&axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap();
        assert_eq!(stats["drafter"], serde_json::json!({"kind":"none", "placement":null,
            "ranks":[], "lead":null, "arena_bytes":[]}));
        assert!(stats["totals"]["draft_verify_passes"].is_u64());
        assert!(stats["totals"]["accepted_drafts"].is_u64());
        assert_eq!(stats["verify"]["drafted_tokens"], 99, "preserve private diagnostics");
    }

    #[tokio::test]
    async fn every_family_profile_publishes_shared_graph_phase_and_exchange_stats() {
        use axum::{body::Body, http::Request};
        use cuteafd_api::openai::{chat, ModelEncoding, ModelProfile};
        use tower::ServiceExt;

        let snapshot = tempfile::tempdir().unwrap();
        cuteafd_loader::plan::testing::write_kimi_tokenizer(snapshot.path());
        let kimi_dir = tempfile::tempdir().unwrap();
        let mut added = serde_json::Map::new();
        for (id, content) in [(163587, "<|open|>"), (163588, "<|close|>"), (163589, "<|sep|>"), (163586, "<|end_of_msg|>")] {
            added.insert(id.to_string(), serde_json::json!({"content": content}));
        }
        std::fs::write(kimi_dir.path().join("tokenizer_config.json"),
            serde_json::json!({"added_tokens_decoder": added}).to_string()).unwrap();
        std::fs::write(kimi_dir.path().join("generation_config.json"),
            serde_json::json!({"eos_token_id": 163586}).to_string()).unwrap();

        let template = "{{ messages[0].content }}";
        let glm = Arc::new(chat::glm5::GlmEncoding::new(template, chat::glm5::GlmTokenIds {
            eos: vec![1], think_close: 2, tool_call: [3, 4], turn_markers: vec![],
        }).unwrap());
        let qwen = Arc::new(chat::qwen4::QwenEncoding::new(template, chat::qwen4::QwenTokenIds {
            eos: vec![1], think_close: 2, tool_call: [3, 4], turn_markers: vec![],
        }).unwrap());
        let kimi = Arc::new(chat::kimi::KimiEncoding::from_snapshot(kimi_dir.path()).unwrap());
        let minimax = Arc::new(chat::minimax::MiniMaxEncoding::new(template, chat::minimax::MiniMaxTokenIds {
            eos: vec![1], think: [2, 3], namespace: 4, tool_call: [5, 6], turn_markers: vec![],
        }).unwrap());
        let step5 = Arc::new(chat::step5::Step5Encoding::new(template, chat::step5::Step5TokenIds {
            bos: 0, eos: vec![1], think_close: 2, tool_call: [3, 4], turn_markers: vec![],
        }, "<s>".into()).unwrap());
        let hy4 = Arc::new(chat::hy4_flash::Hy4Encoding::new(template, chat::hy4_flash::Hy4TokenIds {
            eos: vec![1], think_open: 2, think_close: 3, tool_calls: [4, 5], tool_call: [6, 7], turn_markers: vec![],
        }).unwrap());
        let inkling = Arc::new(chat::inkling2::InklingEncoding::new(template, chat::inkling2::InklingTokenIds {
            eos: vec![1], message_model: 2, thinking: 3, text: 4, invoke_tool_json: 5, end_message: 6,
        }).unwrap());
        let muse = Arc::new(chat::muse::MuseEncoding::new(template, chat::muse::MuseTokenIds {
            bos: 0, eos: vec![1], start: 2, message: 3, end_of_message: 4,
        }, "<s>".into()).unwrap());
        let mistral = Arc::new(chat::mistral::MistralEncoding::new(template, chat::mistral::MistralTokenIds {
            eos: vec![1], think: [2, 3], tool_calls: 4, args: 5, turn_markers: vec![],
        }, "<s>".into(), "</s>".into()).unwrap());

        let families = [
            ("deepseek_v4", ModelEncoding::DeepseekV4),
            ("deepseek_v41", ModelEncoding::DeepseekV41),
            ("glm5", ModelEncoding::Glm(glm.clone())),
            ("glm5_flash", ModelEncoding::Glm(glm)),
            ("kimi_k3", ModelEncoding::Kimi(kimi)),
            ("qwen4", ModelEncoding::Qwen(qwen.clone())),
            ("qwen3_5", ModelEncoding::Qwen(qwen.clone())),
            ("mimo_v2", ModelEncoding::Qwen(qwen)),
            ("minimax_m3", ModelEncoding::MiniMax(minimax)),
            ("step5", ModelEncoding::Step5(step5)),
            ("hy4_flash", ModelEncoding::Hy4Flash(hy4)),
            ("inkling2", ModelEncoding::Inkling(inkling.clone())),
            ("inkling2_mini", ModelEncoding::Inkling(inkling)),
            ("muse_spark", ModelEncoding::Muse(muse)),
            ("mistral_large4", ModelEncoding::Mistral(mistral)),
        ];
        let policy = ApiArgs::default().load().unwrap();
        for (family, encoding) in families {
            let profile = policy.serve(ModelProfile::new(family, encoding), snapshot.path()).unwrap();
            let (tx, _rx) = tokio::sync::mpsc::channel(1);
            let router = cuteafd_api::openai::router_for_model(
                tx,
                Default::default(),
                Arc::new(std::sync::Mutex::new(serde_json::json!({"active": 2}))),
                std::time::Duration::from_secs(1),
                cuteafd_api::openai::ConsoleHub::disabled(),
                profile,
            );
            let response = router.oneshot(Request::get("/v1/stats").body(Body::empty()).unwrap()).await.unwrap();
            let stats: serde_json::Value = serde_json::from_slice(
                &axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap(),
            ).unwrap();
            // Shared graph group.
            assert!(stats["graph_captures"].is_u64(), "{family}: missing graph_captures");
            assert!(stats["graph_capture_sites"].is_array(), "{family}: missing graph_capture_sites");
            assert!(stats["graph_capture_site_deltas"].is_array(), "{family}: missing graph_capture_site_deltas");
            assert!(stats["graph_cache"].is_array(), "{family}: missing graph_cache");
            assert!(stats["graph_missing_variants"].is_array(), "{family}: missing graph_missing_variants");
            assert!(stats["graph_instantiate_ns"].is_u64(), "{family}: missing graph_instantiate_ns");
            // Shared phase group.
            let phases = &stats["phases"];
            for phase in ["prefill_round", "decode_call", "decode_call_with_prefill_waiting", "idle_wait"] {
                assert!(phases["phase_ns_total"][phase].is_u64(), "{family}: missing phases.phase_ns_total.{phase}");
            }
            assert!(phases["phase_ns_total"].get("graph_instantiate").is_none(),
                "{family}: graph_instantiate must not be duplicated under phases");
            assert!(phases["rounds_total"]["prefill"].is_u64(), "{family}: missing rounds_total.prefill");
            assert!(phases["rounds_total"]["decode_calls"].is_u64(), "{family}: missing rounds_total.decode_calls");
            assert!(phases["decode_call_members_total"].is_u64(), "{family}: missing decode_call_members_total");
            assert!(phases["prefill_prompt_rounds_total"].is_u64(), "{family}: missing prefill_prompt_rounds_total");
            assert!(phases["prefill_queue_wait_ns_total"].is_u64(), "{family}: missing prefill_queue_wait_ns_total");
            assert_eq!(phases["active_decode"], 2, "{family}: active_decode must mirror published active count");
            // Shared exchange group.
            let exchange = &stats["exchange"];
            for key in ["staging", "post", "wait"] {
                assert!(exchange["wave_ns_total"][key].is_u64(), "{family}: missing exchange.wave_ns_total.{key}");
            }
            assert!(exchange["waves_total"].is_u64(), "{family}: missing exchange.waves_total");
            assert!(exchange["wave_rows_total"].is_u64(), "{family}: missing exchange.wave_rows_total");
            assert!(exchange["reply_plane_bytes_total"].is_u64(), "{family}: missing exchange.reply_plane_bytes_total");
            for key in ["busy_ns_total", "idle_ns_total", "units_total"] {
                assert_eq!(exchange["expert_prefill_units"][key].as_array().unwrap().len(),
                    cuteafd_core::MAX_SPARK_WORLD, "{family}: bad expert_prefill_units.{key}");
            }
        }
    }

    #[test]
    fn sigterm_closes_monitor_but_preserves_completion_grace() {
        const CHILD: &str = "CUTEAFD_HTTP_DRAIN_CHILD";
        if std::env::var_os(CHILD).is_some() {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                let (queue, _requests) = tokio::sync::mpsc::channel(1);
                let router = cuteafd_api::openai::router(queue).route("/completion-fixture", axum::routing::get(|| async {
                    let stream = async_stream::stream! {
                        yield Ok::<_, std::convert::Infallible>("data: started\n\n");
                        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                        yield Ok("data: [DONE]\n\n");
                    };
                    axum::body::Body::from_stream(stream)
                }));
                let clients = tokio::spawn(async move {
                    let client = reqwest::Client::new();
                    let monitor = client.get(format!("http://{address}/v1/console/events")).send().await.unwrap();
                    let mut completion = client.get(format!("http://{address}/completion-fixture")).send().await.unwrap();
                    assert!(completion.chunk().await.unwrap().is_some());
                    println!("TRACE_SERVER_READY {}", std::process::id());
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();
                    let start = std::time::Instant::now();
                    let text = monitor.text().await.unwrap();
                    assert!(text.contains("data:"));
                    assert!(start.elapsed() < std::time::Duration::from_millis(1000), "monitor did not stop promptly");
                    assert!(completion.text().await.unwrap().contains("[DONE]"), "completion cut off during grace");
                });
                crate::shared::api::serve_http(listener, router, "trace fixture").await.unwrap();
                clients.await.unwrap();
            });
            return;
        }
        let mut child = std::process::Command::new("/usr/bin/timeout")
            .args(["--signal=KILL", "20"]).arg(std::env::current_exe().unwrap())
            .args(["--exact", "shared::api::tests::sigterm_closes_monitor_but_preserves_completion_grace", "--nocapture"])
            .env(CHILD, "1").stdout(std::process::Stdio::piped()).spawn().unwrap();
        use std::io::BufRead;
        let reader = std::io::BufReader::new(child.stdout.take().unwrap());
        let mut signalled = false;
        for line in reader.lines() {
            let line = line.unwrap();
            if let Some((_, pid)) = line.split_once("TRACE_SERVER_READY ") {
                assert!(std::process::Command::new("/bin/kill").args(["-TERM", pid.trim()]).status().unwrap().success());
                signalled = true;
            }
        }
        assert!(child.wait().unwrap().success());
        assert!(signalled);
    }

    #[tokio::test]
    async fn stalled_completion_is_cancelled_after_its_grace() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (signal, shutdown) = tokio::sync::oneshot::channel();
        let router = axum::Router::new().route("/stalled", axum::routing::get(|| async {
            let stream = async_stream::stream! {
                yield Ok::<_, std::convert::Infallible>("started");
                std::future::pending::<()>().await;
            };
            axum::body::Body::from_stream(stream)
        }));
        let server = tokio::spawn(serve_http_until(listener, router, "stalled fixture",
            async { let _ = shutdown.await; }, std::time::Duration::from_millis(30)));
        let mut response = reqwest::get(format!("http://{address}/stalled")).await.unwrap();
        assert!(response.chunk().await.unwrap().is_some());
        let start = std::time::Instant::now();
        signal.send(()).unwrap();
        assert!(tokio::time::timeout(std::time::Duration::from_secs(2), response.text()).await.unwrap().unwrap().is_empty());
        assert!(start.elapsed() >= std::time::Duration::from_millis(25));
        tokio::time::timeout(std::time::Duration::from_secs(2), server).await.unwrap().unwrap().unwrap();
    }

    #[tokio::test]
    async fn scheduler_join_is_only_after_http_ends_and_is_bounded() {
        let (finish, wait) = tokio::sync::oneshot::channel();
        let mut worker = Some(tokio::spawn(async { let _ = wait.await; Ok(()) }));
        let mut served = false;
        tokio::select! {
            _ = async { served = true; } => (),
            _ = watch_scheduler(&mut worker) => panic!("watch must stay pending"),
        }
        assert!(served && worker.is_some());
        finish.send(()).unwrap();
        finish_scheduler(&mut worker, "fixture").await.unwrap();
        assert!(worker.is_none());
        let worker = tokio::spawn(std::future::pending::<anyhow::Result<()>>());
        let abort = worker.abort_handle();
        finish_scheduler_with_timeout(worker, "stuck fixture", std::time::Duration::from_millis(1)).await.unwrap();
        abort.abort();
    }

    #[test]
    fn scheduler_error_is_published_before_request_owners_close() {
        use tracing_subscriber::{layer::SubscriberExt, Layer};
        use std::sync::Mutex;
        struct Observe {
            client: Arc<Mutex<tokio::sync::mpsc::UnboundedReceiver<()>>>,
            messages: Arc<Mutex<Vec<String>>>,
        }
        impl<S: tracing::Subscriber> Layer<S> for Observe {
            fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
                assert_eq!(*event.metadata().level(), tracing::Level::ERROR);
                assert!(!self.client.lock().unwrap().is_closed());
                struct Message(String);
                impl tracing::field::Visit for Message {
                    fn record_debug(&mut self, _: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                        self.0.push_str(&format!("{value:?}"));
                    }
                }
                let mut message = Message(String::new());
                event.record(&mut message);
                self.messages.lock().unwrap().push(message.0);
            }
        }
        let (owner, client) = tokio::sync::mpsc::unbounded_channel();
        let client = Arc::new(Mutex::new(client));
        let messages = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(Observe {
            client: client.clone(), messages: messages.clone(),
        });
        let published = std::cell::RefCell::new(None);
        let result = tracing::subscriber::with_default(subscriber, || run_scheduler_with_health(|| {
            assert!(!owner.is_closed());
            Err(anyhow::anyhow!("underlying CUDA failure")).context("decode scheduler")
        }, |reason| {
            assert!(!client.lock().unwrap().is_closed());
            published.replace(Some(reason));
        }));
        assert_eq!(published.into_inner().as_deref(), Some("scheduler stopped: decode scheduler: underlying CUDA failure"));
        assert_eq!(format!("{:#}", result.unwrap_err()), "decode scheduler: underlying CUDA failure");
        let messages = messages.lock().unwrap();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].contains("decode scheduler: underlying CUDA failure"));
        assert!(!client.lock().unwrap().is_closed());
        drop(owner);
        assert!(client.lock().unwrap().is_closed());
    }

    #[test]
    fn scheduler_panic_publishes_health_without_touching_the_process_latch() {
        let mut published = None;
        let result = run_scheduler_with_health(|| panic!("scheduler fixture"), |reason| published = Some(reason));
        assert!(result.is_err());
        assert_eq!(published.as_deref(), Some("scheduler stopped: scheduler panicked"));
    }

    #[tokio::test]
    async fn default_has_no_benchmark_routes_and_keyed_auth_is_outermost() {
        use axum::{body::Body, http::StatusCode};
        use tower::ServiceExt;
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let app = ApiArgs::default().load().unwrap().app(cuteafd_api::openai::router(tx),
            cuteafd_api::openai::ConsoleHub::disabled());
        assert_eq!(app.clone().oneshot(axum::http::Request::get("/v1/bench/status").body(Body::empty()).unwrap())
            .await.unwrap().status(), StatusCode::NOT_FOUND);
        // The page itself says the runner is off instead of 404.
        let page = app.oneshot(axum::http::Request::get("/bench").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(page.status(), StatusCode::OK);
        let body = axum::body::to_bytes(page.into_body(), 1 << 20).await.unwrap();
        assert!(String::from_utf8_lossy(&body).contains("ENABLE_BENCH=on"));
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let app = ApiPolicy { key: Some(ApiKey::new("secret").unwrap()), bench: false, usage: None, gate: cuteafd_api::console_gate::ConsoleGate::locked(), gateway: None, bench_token: None, live: None }.app(
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
    /// The server's API key and console secret are scrubbed from logged payloads.
    #[tokio::test]
    async fn server_secrets_never_reach_the_full_log() {
        use axum::{body::Body, http::Request};
        use tower::ServiceExt;
        let dir = tempfile::tempdir().unwrap();
        let key = dir.path().join("key");
        std::fs::write(&key, "sk-local-SERVER-KEY-777\n").unwrap();
        let secret = dir.path().join("secret");
        std::fs::write(&secret, "e".repeat(64)).unwrap();
        let usage = dir.path().join("usage");
        let policy = ApiArgs { api_key_file: Some(key), console_secret_file: Some(secret), usage_dir: Some(usage.clone()), ..Default::default() }.load().unwrap();
        let store = policy.usage.clone().unwrap();
        let router = axum::Router::new().route("/v1/chat/completions", axum::routing::post(|_: axum::body::Bytes| async { "{}" }));
        let app = policy.app(router, cuteafd_api::openai::ConsoleHub::disabled());
        let body = format!(r#"{{"messages":[{{"role":"user","content":"my key is sk-local-SERVER-KEY-777 and {}"}}]}}"#, "e".repeat(64));
        let r = app.oneshot(Request::post("/v1/chat/completions").header("Authorization", "Bearer sk-local-SERVER-KEY-777")
            .body(Body::from(body)).unwrap()).await.unwrap();
        axum::body::to_bytes(r.into_body(), 4096).await.unwrap();
        store.log.flush().unwrap();
        store.flush().unwrap();
        let full = store.log.get(&store.rows().unwrap()[0].rid).unwrap().unwrap();
        assert!(full["request"]["messages"][0]["content"].as_str().unwrap().contains("[REDACTED]"));
        for entry in std::fs::read_dir(&usage).unwrap() {
            let text = String::from_utf8_lossy(&std::fs::read(entry.unwrap().path()).unwrap()).into_owned();
            assert!(!text.contains("SERVER-KEY-777") && !text.contains(&"e".repeat(64)));
        }
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
    #[test]
    fn serving_requires_a_real_snapshot_tokenizer() {
        let snapshot = tempfile::tempdir().unwrap();
        let policy = ApiArgs::default().load().unwrap();
        let profile = cuteafd_api::openai::ModelProfile::new("test", cuteafd_api::openai::ModelEncoding::DeepseekV41);
        assert!(policy.serve(profile, snapshot.path()).is_err());
    }

    #[tokio::test]
    async fn gateway_routes_mount_by_default_with_every_key_form_and_turn_off() {
        use axum::{body::Body, http::{Request, StatusCode}};
        use clap::Parser;
        use tower::ServiceExt;
        let snapshot = tempfile::tempdir().unwrap();
        cuteafd_loader::plan::testing::write_kimi_tokenizer(snapshot.path());
        assert!(!snapshot.path().join("tokenizer.json").exists());
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
    fn benchmark_controls_need_no_api_key() {
        assert!(ApiArgs { enable_bench: true, api_key_file: None, ..Default::default() }.load().is_ok());
    }
}
