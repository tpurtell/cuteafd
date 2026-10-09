//! CPU-only protocol gateway over an HTTP upstream.
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use anyhow::{Context, Result};
use clap::Args;
use cuteafd_api::gateway::{self, record::{Recorder, Sanitizer}, search::{Exa, Searxng}, upstream::{Flavor, Upstream, UpstreamConfig}, Gateway, ModelMap};

#[derive(Debug, Args)]
pub(crate) struct GatewayArgs {
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    /// Base URL containing any protocol prefix (e.g. /v1 for Chat, /anthropic for DeepSeek Messages).
    #[arg(long)]
    upstream_url: String,
    #[arg(long, default_value = "openai-chat", value_parser = ["openai-chat", "anthropic"])]
    upstream_flavor: String,
    /// Environment variable name; the credential itself must never be a CLI argument.
    #[arg(long)]
    upstream_key_env: Option<String>,
    #[arg(long)]
    model: String,
    #[arg(long)]
    accept_any_model: bool,
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
}
fn alias(value: &str) -> Result<(String,String),String> {
    let (pattern,target) = value.split_once('=').ok_or("alias must be PATTERN=MODEL")?;
    if pattern.is_empty() || target.is_empty() { return Err("alias must have a nonempty pattern and model".into()); }
    Ok((pattern.into(),target.into()))
}
pub(crate) async fn run(args: GatewayArgs) -> Result<()> {
    let flavor: Flavor = args.upstream_flavor.parse()?;
    let url = url::Url::parse(&args.upstream_url).context("invalid upstream URL")?;
    let deepseek = url.host_str() == Some("api.deepseek.com");
    let mut config = UpstreamConfig::new(args.upstream_url.clone(),flavor,args.model.clone());
    config.key = args.upstream_key_env.as_ref().map(|name| std::env::var(name).with_context(|| format!("upstream key environment variable {name} is not set"))).transpose()?;
    config.deepseek_thinking = deepseek;
    config.capabilities.vision = deepseek && args.model == "deepseek-flash";
    let mut secrets:Vec<String> = config.key.iter().cloned().collect();
    if let Some(file) = &args.api_key_file { secrets.push(std::fs::read_to_string(file).context("read gateway API key file")?.trim().to_string()); }
    let api = crate::shared::api::ApiArgs { api_key_file:args.api_key_file,enable_bench:false }.load()?;
    let backend = Arc::new(Upstream::new(config)?.discover().await);
    let models = ModelMap { served:args.model.clone(),accept_any:args.accept_any_model,aliases:args.aliases,listed:args.listed };
    let mut gateway = Gateway::new(backend,models);
    let provider = if args.search == "none" { "none" }
        else if args.search == "exa" {
            let key = std::env::var("EXA_API_KEY").context("EXA_API_KEY is not set")?;
            secrets.push(key.clone());
            gateway = gateway.with_search(Arc::new(Exa::new(key)?)); "exa"
        } else if let Some(url) = args.search.strip_prefix("searxng=") {
            gateway = gateway.with_search(Arc::new(Searxng::new(url)?)); "searxng"
        } else { anyhow::bail!("--search must be none, exa, or searxng=URL"); };
    let mut app = gateway::router(Arc::new(gateway))
        .route("/health",axum::routing::get(|| async { axum::Json(serde_json::json!({"status":"ok"})) }))
        .layer(axum::middleware::from_fn_with_state(api.gateway_auth(),gateway::auth::require_key));
    if let Some(directory) = args.record {
        let recorder = Recorder::new(directory,Sanitizer::from_env(secrets))?;
        app = app.layer(axum::middleware::from_fn_with_state(recorder,gateway::record::middleware));
    }
    tracing::info!(flavor=flavor.name(),url=%args.upstream_url,model=%args.model,search=provider,"gateway ready");
    let listener = tokio::net::TcpListener::bind(args.listen).await.context("bind gateway listener")?;
    axum::serve(listener,app).with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; }).await?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[test]
    fn gateway_cli_requires_env_name_not_secret_and_parses_aliases() {
        let cli = crate::cli::Cli::try_parse_from(["cuteafd","gateway","--upstream-url","http://127.0.0.1:8000/v1","--model","local", "--alias","gpt-*=local","--list-model","gpt-test","--upstream-key-env","TEST_KEY"]).unwrap();
        let crate::cli::Commands::Gateway(args) = cli.command else { panic!("gateway expected") };
        assert_eq!(args.aliases,vec![("gpt-*".into(),"local".into())]);
        assert_eq!(args.upstream_key_env.as_deref(),Some("TEST_KEY"));
        assert!(alias("=model").is_err());
    }
}
