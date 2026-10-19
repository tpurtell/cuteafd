//! Native text serving through the shared scheduler, sampler and exact prefix cache.
use super::{backend::KimiBackend, startup};
use crate::shared::{prefix::PrefixArgs, qwen::scheduler::{self, Backend}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{chat::kimi::KimiEncoding, ModelEncoding, ModelProfile, NativeLimits, NativeRequest};
use std::{net::SocketAddr, path::PathBuf, sync::{Arc, Mutex}};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)] pub device: i32,
    #[arg(long, default_value_t = 1)] pub split_device: i32,
    #[arg(long, value_delimiter = ',')] pub peers: Vec<SocketAddr>,
    /// Auto admission targets 2M tokens on the reference coordinator pair.
    #[arg(long)] pub pool_tokens: Option<u64>,
    #[arg(long, default_value_t = 1_048_576)] pub max_context: usize,
    #[arg(long, default_value_t = 5)] pub rtx_expert_layers: usize,
    #[arg(long, default_value = "0.0.0.0:8000")] pub listen: String,
    #[arg(long, default_value_t = 32768)] pub max_output: u32,
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u8).range(1..=64))]
    pub max_sequences: u8,
    #[arg(long)] pub model_id: Option<String>,
    #[command(flatten)] pub prefix: PrefixArgs,
    #[command(flatten)] pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)] pub token_io: crate::shared::token_io::TokenIoArgs,
    #[command(flatten)] pub api: crate::shared::api::ApiArgs,
}

impl ServeArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.device >= 0 && self.split_device >= 0 && self.device != self.split_device,
            "Kimi serving requires two distinct coordinator devices");
        ensure!(self.peers.len() == 6 && self.peers.iter().collect::<std::collections::HashSet<_>>().len() == 6,
            "Kimi serving requires six distinct real Spark endpoints");
        ensure!((1..=1_048_576).contains(&self.max_context), "Kimi context exceeds the compiled 1M extent");
        ensure!(self.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Kimi serving requires exact prefix frontiers");
        ensure!(matches!(self.token_io.embed_placement, crate::shared::token_io::EmbedPlacement::Gpu),
            "Kimi serving admits GPU token embeddings only");
        Ok(())
    }
}

pub(crate) async fn run(args: ServeArgs) -> Result<()> {
    args.validate()?;
    let api = args.api.load()?;
    let encoding = Arc::new(KimiEncoding::from_snapshot(&args.snapshot)?);
    let eos = encoding.eos_token_ids();
    let model = args.model_id.clone().or_else(|| args.snapshot.ancestors().find_map(|path| {
        let name = path.file_name()?.to_str()?.strip_prefix("models--")?;
        let (organization, model) = name.split_once("--")?;
        Some(format!("{organization}/{model}"))
    })).context("Kimi serving requires --model-id for a snapshot without HF identity")?;
    let profile = ModelProfile::new(model, ModelEncoding::Kimi(encoding));
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<usize>>();
    let worker_args = args.clone();
    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| -> Result<()> {
            cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
                .map_err(|_| anyhow::anyhow!("Kimi serving needs an isolated process"))?;
            let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&worker_args.snapshot, false, false)?;
            let cfg = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(&checkpoint.config)?;
            cfg.release_geometry()?;
            let slots = usize::from(worker_args.max_sequences);
            let mark_bytes = super::state::KdaLayout::new(&cfg, slots, 1, 1)?.mark_bytes()
                .checked_mul(2).context("Kimi TP2 prefix mark extent overflow")?;
            let marks = worker_args.prefix.mark_rule(slots).slots(mark_bytes);
            // SAFETY: every native owner drops on this worker before the library.
            let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&worker_args.native_lib)? });
            let programs = library.programs()?.with_manifest(&worker_args.manifest)?;
            let loaded = startup::load(&library, &programs, &checkpoint, &cfg, &cfg, startup::Options {
                devices: [worker_args.device, worker_args.split_device], native_lib: &worker_args.native_lib,
                peers: worker_args.peers.clone(), rtx_expert_layers: worker_args.rtx_expert_layers,
                rows: 4096, slots, marks, max_context: worker_args.max_context,
                pool_tokens: worker_args.pool_tokens.filter(|&tokens| tokens > 0), graphs: true,
                ffn_mode: cuteafd_ffi::KimiFfnMode::default(), hosted_layers: None,
            })?;
            crate::shared::memory_report::release_load_staging(&library);
            let backend = KimiBackend::new(&loaded.engine)?;
            // The shared selector allocates on the calling thread's device and
            // consumes logits on the terminal rank's exact producing stream.
            library.cuda_set_device(worker_args.split_device)?;
            scheduler::schedule(&backend, &worker_args.snapshot, &mut receive, &worker_stats, slots,
                scheduler::Drafts::None, eos, worker_args.decode_share, &worker_args.prefix,
                worker_args.token_io.token_select, &mut super::super::qwen3_5::serve::text_media(), None, || {
                    loaded.engine.require_live()?;
                    crate::shared::inventory::RuntimeInventory::measure(&library,
                        &[worker_args.device, worker_args.split_device],
                        crate::shared::inventory::InventoryPhase::PostLoad, |_| Ok(()))?
                        .record("kimi_k3-serving-loaded", &loaded.admission_free);
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Ok(backend.geometry().max_context));
                    }
                    Ok(())
                })
        })();
        if let Some(ready) = ready.take() {
            let error = result.as_ref().err().map(|error| format!("{error:#}"))
                .unwrap_or_else(|| "scheduler exited before readiness".into());
            let _ = ready.send(Err(anyhow::anyhow!("Kimi initialization failed: {error}")));
        }
        result
    });
    let max_context = ready_rx.await.context("Kimi worker exited before readiness")??;
    let limits = NativeLimits::new(u32::try_from(max_context)?, args.max_output)?;
    let snapshot = args.snapshot.clone();
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new(
        "kimi_k3", profile.id.clone(), snapshot.clone())));
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, std::time::Duration::from_secs(25),
        hub.clone(), api.serve(profile, &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(worker) => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args(extra: &[&str]) -> Result<ServeArgs, clap::Error> {
        let mut argv = vec!["cuteafd", "serve-kimi", "--snapshot", "/checkpoint", "--native-lib", "/native.so"];
        argv.extend_from_slice(extra);
        let cli = crate::cli::Cli::try_parse_from(argv)?;
        let crate::cli::Commands::ServeKimi(args) = cli.command else { panic!("Kimi command") };
        Ok(args)
    }

    #[test]
    fn production_defaults_and_preflight_reject_unsupported_geometry() -> Result<()> {
        let peers = "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4,127.0.0.1:5,127.0.0.1:6";
        let defaults = args(&["--peers", peers])?;
        defaults.validate()?;
        assert_eq!(defaults.max_context, 1_048_576);
        assert_eq!(defaults.rtx_expert_layers, 5);
        assert_eq!(defaults.max_sequences, 16);
        assert_eq!(defaults.pool_tokens, None);
        assert!(args(&[])?.validate().is_err());
        for extra in [["--split-device", "0"], ["--max-context", "1048577"],
            ["--embedding-placement", "host"], ["--prefix-partial", "on"]] {
            assert!(args(&["--peers", peers, extra[0], extra[1]])?.validate().is_err());
        }
        assert!(args(&["--max-sequences", "65"]).is_err());
        assert!(args(&["--speculator", "mtp"]).is_err());
        Ok(())
    }
}
