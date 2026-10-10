//! CPU-only development/test harness over a generic HTTP upstream.
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use anyhow::{Context, Result};
use clap::Args;
use cuteafd_api::gateway::{self, record::{Recorder, Sanitizer}, search::{Exa, Searxng}, upstream::{Flavor, Upstream, UpstreamConfig}, Gateway, ModelMap};

#[derive(Debug, Args)]
pub(crate) struct GatewayArgs {
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    /// Base URL containing any protocol prefix (e.g. /v1 for Chat, /anthropic for DeepSeek Messages).
    #[arg(long, required_unless_present = "upstream_url_env")]
    upstream_url: Option<String>,
    #[arg(long, default_value = "openai-chat", value_parser = ["openai-chat", "anthropic"])]
    upstream_flavor: String,
    /// Comma-separated served-model capabilities, supplemented by generic /models metadata.
    #[arg(long, value_delimiter = ',', value_parser = ["vision","audio_in","reasoning","json_schema","strict_tools"])]
    upstream_capabilities: Vec<String>,
    /// Emit a thinking enabled/disabled object; reject forced tools and omit temperature while thinking.
    #[arg(long)]
    upstream_thinking_toggle: bool,
    /// The upstream rejects forced tool choice; send a named choice as that one tool with auto.
    #[arg(long)]
    upstream_no_forced_tool_choice: bool,
    /// Optional absolute endpoint path for strict Chat tools, with no fallback on errors.
    #[arg(long)]
    upstream_strict_tools_path: Option<String>,
    /// Environment variable holding the base URL (for private test endpoints).
    #[arg(long, conflicts_with = "upstream_url")]
    upstream_url_env: Option<String>,
    /// Environment variable name; the credential itself must never be a CLI argument.
    #[arg(long)]
    upstream_key_env: Option<String>,
    #[arg(long)]
    model: String,
    #[arg(long)]
    accept_any_model: bool,
    /// Advertise official client model ids and accept any requested model alias.
    #[arg(long)]
    official_model_names: bool,
    /// Refresh/replace advertised names from a text or JSON file; implies --official-model-names.
    #[arg(long)]
    official_model_names_file: Option<PathBuf>,
    #[arg(long = "alias", value_parser = alias)]
    aliases: Vec<(String,String)>,
    #[arg(long = "list-model")]
    listed: Vec<String>,
    #[arg(long)]
    api_key_file: Option<PathBuf>,
    #[arg(long, default_value = "none")]
    search: String,
    #[arg(long)]
    record: Option<PathBuf>,
    /// Usage history directory (metadata and the full log); enables usage recording.
    #[arg(long)]
    usage_dir: Option<PathBuf>,
    /// Request accounting; defaults to on when --usage-dir is given (in memory otherwise).
    #[arg(long, value_parser = ["on", "off"])]
    usage: Option<String>,
    /// File containing the host console secret; unlocks /usage and its data routes.
    #[arg(long)]
    console_secret_file: Option<PathBuf>,
}
fn alias(value: &str) -> Result<(String,String),String> {
    let (pattern,target) = value.split_once('=').ok_or("alias must be PATTERN=MODEL")?;
    if pattern.is_empty() || target.is_empty() { return Err("alias must have a nonempty pattern and model".into()); }
    Ok((pattern.into(),target.into()))
}
fn resolve_base(args: &GatewayArgs, lookup: impl FnOnce(&str) -> Result<String,std::env::VarError>) -> Result<String> {
    match (&args.upstream_url,&args.upstream_url_env) {
        (Some(url),_) => Ok(url.clone()),
        (_,Some(name)) => lookup(name).with_context(|| format!("upstream URL environment variable {name} is not set")),
        _ => anyhow::bail!("upstream URL is required"),
    }
}
fn endpoint_secrets(base: &str) -> Result<Vec<String>> {
    let url = url::Url::parse(base).context("invalid upstream URL")?;
    let mut secrets = vec![base.to_string()];
    if let Some(host) = url.host_str() { secrets.push(host.into()); }
    Ok(secrets)
}
fn log_ready(flavor: Flavor, model: &str, provider: &str) {
    tracing::info!(flavor=flavor.name(),upstream="configured",model,search=provider,"gateway ready");
}
pub(crate) async fn run(args: GatewayArgs) -> Result<()> {
    let flavor: Flavor = args.upstream_flavor.parse()?;
    let base = resolve_base(&args,|name| std::env::var(name))?;
    let mut secrets = endpoint_secrets(&base)?;
    let mut config = UpstreamConfig::new(base,flavor,args.model.clone());
    config.key = args.upstream_key_env.as_ref().map(|name| std::env::var(name).with_context(|| format!("upstream key environment variable {name} is not set"))).transpose()?;
    config.thinking_toggle = args.upstream_thinking_toggle;
    config.no_forced_tool_choice = args.upstream_no_forced_tool_choice;
    config.strict_tools_path = args.upstream_strict_tools_path;
    config.capabilities = Default::default();
    for capability in &args.upstream_capabilities {
        match capability.as_str() {
            "vision" => config.capabilities.vision = true,
            "audio_in" => config.capabilities.audio_in = true,
            "reasoning" => config.capabilities.reasoning = true,
            "json_schema" => config.capabilities.json_schema = true,
            "strict_tools" => config.capabilities.strict_tools = true,
            _ => unreachable!("validated capability"),
        }
    }
    secrets.extend(config.key.iter().cloned());
    if let Some(file) = &args.api_key_file { secrets.push(std::fs::read_to_string(file).context("read gateway API key file")?.trim().to_string()); }
    let usage = args.usage.clone().unwrap_or_else(|| if args.usage_dir.is_some() { "on".into() } else { "off".into() });
    let api = crate::shared::api::ApiArgs { api_key_file:args.api_key_file,enable_bench:false,usage:Some(usage),gateway:"off".into(),
        usage_dir:args.usage_dir,console_secret_file:args.console_secret_file,..Default::default() }.load()?;
    let backend = Arc::new(Upstream::new(config)?.discover().await);
    let mut models = if args.official_model_names || args.official_model_names_file.is_some() {
        ModelMap::official_names(args.model.clone())
    } else { ModelMap::single(args.model.clone()) };
    if let Some(file) = args.official_model_names_file {
        models.apply_names_file(&std::fs::read_to_string(file).context("read official model names file")?).map_err(anyhow::Error::msg)?;
    }
    models.accept_any |= args.accept_any_model;
    models.aliases.extend(args.aliases);
    models.listed.extend(args.listed);
    let mut gateway = Gateway::new(backend,models);
    gateway.origins = gateway::OriginPolicy { key: api.gateway_auth().key, allowed: Vec::new() };
    let provider = if args.search == "none" { "none" }
        else if args.search == "exa" {
            let key = std::env::var("EXA_API_KEY").context("EXA_API_KEY is not set")?;
            secrets.push(key.clone());
            gateway = gateway.with_search(Arc::new(Exa::new(key)?)); "exa"
        } else if let Some(url) = args.search.strip_prefix("searxng=") {
            gateway = gateway.with_search(Arc::new(Searxng::new(url)?)); "searxng"
        } else { anyhow::bail!("--search must be none, exa, or searxng=URL"); };
    let app = gateway::router(Arc::new(gateway))
        .route("/health",axum::routing::get(|| async { axum::Json(serde_json::json!({"status":"ok"})) }))
        .merge(cuteafd_api::openai::console::asset_routes());
    let app = api.mount_console(app).layer(axum::middleware::from_fn_with_state(api.gateway_auth(),gateway::auth::require_key));
    let mut app = api.track(app);
    if let Some(directory) = args.record {
        let recorder = Recorder::new(directory,Sanitizer::from_env(secrets))?;
        app = app.layer(axum::middleware::from_fn_with_state(recorder,gateway::record::middleware));
    }
    log_ready(flavor,&args.model,provider);
    let listener = tokio::net::TcpListener::bind(args.listen).await.context("bind gateway listener")?;
    axum::serve(listener,app).with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; }).await?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[tokio::test]
    async fn custom_url_env_echo_is_redacted_from_recordings() {
        use tower::ServiceExt;
        let cli = crate::cli::Cli::try_parse_from(["cuteafd","gateway","--upstream-url-env","CUSTOM_TEST_ENDPOINT","--model","local"]).unwrap();
        let crate::cli::Commands::Gateway(args) = cli.command else { panic!("gateway expected") };
        let base = resolve_base(&args,|name| {
            assert_eq!(name,"CUSTOM_TEST_ENDPOINT");
            Ok("https://private-example.invalid/v1".into())
        }).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let recorder = Recorder::new(directory.path().into(),Sanitizer::from_env(endpoint_secrets(&base).unwrap())).unwrap();
        let echo = format!("data: {{\"text\":\"{base} private-example.invalid\"}}\n\n");
        let original = echo.clone();
        let app = axum::Router::new().route("/test",axum::routing::get(move |tape:gateway::record::Tape| {
            let echo = echo.clone();
            async move {
                tape.record("upstream",|| serde_json::json!({"body":echo}));
                echo
            }
        })).layer(axum::middleware::from_fn_with_state(recorder,gateway::record::middleware));
        let response = app.oneshot(axum::http::Request::builder().uri("/test").body(axum::body::Body::empty()).unwrap()).await.unwrap();
        let body = axum::body::to_bytes(response.into_body(),1024).await.unwrap();
        assert_eq!(&body[..],original.as_bytes(),"client delivery must stay unmodified");
        let fixture = std::fs::read_dir(directory.path()).unwrap().next().unwrap().unwrap().path();
        let fixture = std::fs::read_to_string(fixture).unwrap();
        assert!(!fixture.contains(&base) && !fixture.contains("private-example.invalid"));
        assert!(fixture.contains("REDACTED"));
    }
    #[test]
    fn direct_upstream_url_is_never_in_startup_log() {
        use std::io::Write;
        #[derive(Clone)]
        struct Capture(Arc<std::sync::Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self,bytes:&[u8]) -> std::io::Result<usize> { self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len()) }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let cli = crate::cli::Cli::try_parse_from(["cuteafd","gateway","--upstream-url","https://private-example.invalid/private-path","--model","local"]).unwrap();
        let crate::cli::Commands::Gateway(args) = cli.command else { panic!("gateway expected") };
        let bytes = Arc::new(std::sync::Mutex::new(Vec::new()));
        let capture = Capture(bytes.clone());
        let subscriber = tracing_subscriber::fmt().with_ansi(false).without_time().with_writer(move || capture.clone()).finish();
        tracing::subscriber::with_default(subscriber,|| log_ready(Flavor::OpenaiChat,&args.model,"none"));
        let log = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
        assert!(log.contains("configured") && log.contains("gateway ready"));
        assert!(!log.contains("private-example") && !log.contains("private-path") && !log.contains("https://"));
    }
    #[test]
    fn gateway_cli_requires_env_name_not_secret_and_parses_aliases() {
        let cli = crate::cli::Cli::try_parse_from(["cuteafd","gateway","--upstream-url","http://127.0.0.1:8000/v1","--model","local", "--alias","gpt-*=local","--list-model","gpt-test","--upstream-key-env","TEST_KEY"]).unwrap();
        let crate::cli::Commands::Gateway(args) = cli.command else { panic!("gateway expected") };
        assert_eq!(args.aliases,vec![("gpt-*".into(),"local".into())]);
        assert_eq!(args.upstream_key_env.as_deref(),Some("TEST_KEY"));
        assert!(alias("=model").is_err());
    }
}
