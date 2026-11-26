//! Inkling 2 Mini HTTP/request lifecycle (`serve-inkling2-mini`) over the shared scheduler.
use crate::shared::{prefix::PrefixArgs, token_io::TokenIoArgs};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{
    families::inkling2_mini::{Inkling2MiniConfig, InklingKvCache},
    placement::planner::PrefillShape,
    plan::Checkpoint,
};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub vision: crate::shared::vision::still::Options,
    #[command(flatten)]
    pub attention: crate::cli::CoreViewArgs,
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long)]
    pub pool_tokens: Option<u64>,
    #[arg(long = "kv-cache", default_value = "fp8")]
    pub kv_format: InklingKvCache,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pub dspark: bool,
    #[arg(long)]
    pub draft: Option<PathBuf>,
    #[arg(long, default_value = "fp8")]
    pub draft_kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
    #[arg(long = "max-context-tokens", value_parser = crate::shared::context::parse_tokens::<usize>)]
    pub max_context: Option<usize>,
    #[arg(long, default_value_t = 16)]
    pub max_sequences: usize,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long)]
    pub model_id: Option<String>,
    #[arg(long)]
    pub prefill_rows: Option<usize>,
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=3))]
    pub prefill_waves: Option<u8>,
    #[arg(long, value_enum, default_value = "unrestricted")]
    pub prefill_packing: crate::shared::prefill_packing::PackingMode,
    #[command(flatten)]
    pub prefix: PrefixArgs,
    #[command(flatten)]
    pub draft_length: crate::shared::draft::length::DraftLength,
    #[command(flatten)]
    pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)]
    pub loop_abort: crate::shared::loop_abort::LoopAbortArgs,
    #[command(flatten)]
    pub token_io: TokenIoArgs,
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,
}

impl ServeArgs {
    fn prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        Ok(prefill_shape(
            0,
            cuteafd_loader::plan::Family::prefill_limits(
                &cuteafd_loader::plan::families::inkling2_mini::INKLING2_MINI,
            ),
            PrefillOverrides {
                unit_rows: self.prefill_rows.map(|rows| rows as u64),
                waves: self.prefill_waves.map(u64::from),
                span_rows: None,
                reserve_rows: None,
            },
        )?)
    }
}

pub(crate) async fn run(mut args: ServeArgs) -> Result<()> {
    use crate::shared::{
        inventory::{InventoryPhase, RuntimeInventory},
        qwen::scheduler::{self, Backend},
    };
    use cuteafd_api::openai::{chat::inkling2::InklingEncoding, ModelEncoding, ModelProfile, NativeRequest};

    cuteafd_core::concurrency::validate("inkling2_mini", args.max_sequences as u64)?;
    args.prefill_packing.require_unrestricted("Inkling 2 Mini")?;
    ensure!(
        args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu
            && args.token_io.token_select == crate::shared::token_io::SelectPlacement::Device,
        "Inkling 2 Mini serving requires admitted GPU embedding and device selection"
    );
    args.prefix.resolve_entries(args.max_sequences);
    let prefill = args.prefill()?;
    cuteafd_bench::context::ResolvedSettings::prefill(prefill).publish();
    let encoding = ModelEncoding::Inkling(Arc::new(InklingEncoding::from_snapshot(&args.snapshot)?));
    let eos = encoding.eos_token_ids();
    ensure!(eos == [200_006], "Inkling 2 Mini native EOS differs");
    let model = args
        .model_id
        .clone()
        .or_else(|| cuteafd_api::openai::snapshot_model_id(&args.snapshot))
        .context("Inkling 2 Mini snapshot without HF identity requires --model-id")?;
    let profile = ModelProfile::new(model, encoding);
    let api = args.api.load()?;
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(
        cuteafd_core::concurrency::queue_depth(args.max_sequences as u64) as usize,
    );
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let (ready_tx, ready_rx) =
        tokio::sync::oneshot::channel::<Result<(usize, Option<Arc<cuteafd_api::openai::media::MediaPreparer>>)>>();
    let worker_args = args.clone();
    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| {
            let view = worker_args
                .attention
                .selected_view
                .clone()
                .map(Ok)
                .unwrap_or_else(|| worker_args.attention.select(&worker_args.snapshot))?;
            worker_args.attention.validate_view(&view)?;
            let checkpoint = Checkpoint::open_view(&view)?;
            let cfg = Inkling2MiniConfig::from_hf(&checkpoint.config)?;
            cuteafd_bench::context::set_family("inkling2_mini");
            cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::INKLING2_MINI)
                .map_err(|prev| anyhow::anyhow!("Mini process geometry already fixed to {prev:?}"))?;
            // SAFETY: every GPU/graph owner retires before library drops.
            let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&worker_args.native_lib)? });
            let programs = library.programs()?.with_manifest(&worker_args.manifest)?;
            let (execution, inputs, admitted, _, media) = super::startup::whole(
                &library,
                &programs,
                &view,
                &checkpoint,
                &cfg,
                super::startup::Options {
                    device: worker_args.device,
                    manifest: &worker_args.manifest,
                    native_lib: &worker_args.native_lib,
                    prefill: PrefillShape {
                        rows: prefill.unit_rows,
                        waves: prefill.waves,
                    },
                    sequences: worker_args.max_sequences as u64,
                    context_tokens: worker_args.max_context.unwrap_or(0) as u64,
                    pool_tokens: worker_args.pool_tokens,
                    kv_format: worker_args.kv_format,
                    drafter: (worker_args.dspark && worker_args.draft.is_none()).then_some(worker_args.draft_kv_cache),
                    dflash: worker_args.draft.as_deref().map(|p| (p, worker_args.draft_kv_cache)),
                    prefix: &worker_args.prefix,
                    resident_weights: crate::shared::memory::budget::resident_weight_limit(),
                    expert_usage: false,
                    prewarm_graphs: true,
                    vision: &worker_args.vision,
                },
            )?;
            let context = usize::try_from(admitted.solved.served_context())?;
            let family = super::cache::MiniPrefix::for_execution(&execution);
            let host = (inputs.host_prefix.quota_bytes > 0)
                .then(|| {
                    use cuteafd_engine::prefix::PrefixFamily;
                    worker_args
                        .prefix
                        .host_config_for(family.layout(), context, inputs.host_prefix.quota_bytes)
                })
                .transpose()?;
            let drafts = execution.draft.as_ref().map_or(scheduler::Drafts::None, |draft| {
                scheduler::Drafts::Neural {
                    depth: draft.block - 1,
                }
            });
            let binding = crate::shared::vision::perception::Binding::new(
                &worker_args.snapshot,
                cuteafd_loader::media::ImageFamily::Inkling2,
                media.preparer.is_some(),
                cfg.text.unpadded_vocab_size as u32,
            )?;
            let backend =
                super::backend::MiniBackend::new(execution, context, host, &admitted.solved)?.with_vision(binding);
            let mut media_admission = cuteafd_engine::media::MediaAdmission::new(
                cuteafd_engine::media::EmbeddingCache::new(media.cache_bytes),
                media.encoder,
                worker_args.max_sequences,
            );
            library.cuda_set_device(worker_args.device)?;
            scheduler::schedule(
                &backend,
                &worker_args.snapshot,
                &mut receive,
                &worker_stats,
                worker_args.max_sequences,
                drafts,
                worker_args.draft_length,
                eos,
                worker_args.decode_share,
                &worker_args.prefix,
                worker_args.loop_abort.config(),
                worker_args.token_io.token_select,
                &mut media_admission,
                None,
                || {
                    backend.require_live()?;
                    RuntimeInventory::measure(&library, &[worker_args.device], InventoryPhase::PostLoad, |_| Ok(()))?
                        .record("inkling2_mini-serving-loaded", &[]);
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Ok((context, media.preparer.clone())));
                    }
                    Ok(())
                },
            )
        })();
        crate::shared::serve::forward_initialization(result, ready.is_some(), "Inkling 2 Mini initialization failed", |error| {
            ready
                .take()
                .expect("pending readiness")
                .send(Err(error))
                .map_err(|result| match result {
                    Err(error) => error,
                    Ok(_) => unreachable!("sent error"),
                })
        })
    });
    let (context, preparer) = ready_rx
        .await
        .context("Inkling 2 Mini worker exited before readiness")??;
    let profile = if let Some(preparer) = preparer {
        profile.with_grouped_vision(preparer)
    } else {
        profile
    };
    let limits = args.prefix.reserve.limits(context)?;
    let snapshot = args.snapshot.clone();
    let hub = crate::shared::console::hub(false, || {
        Ok(crate::shared::console::Layout::new(
            "inkling2_mini",
            profile.id.clone(),
            snapshot.clone(),
        ))
    });
    let router = cuteafd_api::openai::router_for_model(
        queue,
        limits,
        stats,
        std::time::Duration::from_secs(25),
        hub.clone(),
        api.serve(profile, &snapshot)?,
    );
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    let mut worker = Some(worker);
    tokio::select! {
        served = crate::shared::api::serve_http(listener, api.app(router, hub), "inkling2_mini") => served?,
        _ = crate::shared::api::watch_scheduler(&mut worker) => unreachable!(),
    }
    crate::shared::api::finish_scheduler(&mut worker, "inkling2_mini").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn inkling2_mini_serve_startup_seam_resolves_settings_and_rejects_exact_packing() -> Result<()> {
        let argv = [
            "cuteafd",
            "serve-inkling2-mini",
            "--snapshot",
            "/model",
            "--native-lib",
            "/native.so",
            "--listen",
            "0.0.0.0:8823",
            "--max-sequences",
            "16",
            "--max-context-tokens",
            "auto",
            "--embedding-placement",
            "gpu",
            "--token-select",
            "device",
            "--kv-cache",
            "fp8",
            "--dspark",
            "true",
            "--draft-kv-cache",
            "fp8",
            "--prefix-cache-entries",
            "20",
            "--host-cache-bytes",
            "auto",
            "--prefill-rows",
            "2048",
            "--prefill-waves",
            "2",
        ];
        let crate::cli::Commands::ServeInkling2Mini(mut args) = crate::cli::Cli::try_parse_from(argv)?.command else {
            anyhow::bail!("Inkling 2 Mini command not parsed")
        };
        assert_eq!((args.max_context, args.max_sequences), (Some(0), 16));
        assert_eq!(args.prefill_packing, crate::shared::prefill_packing::PackingMode::Unrestricted);
        let mut exact = argv.to_vec();
        exact.extend(["--prefill-packing", "exact"]);
        let crate::cli::Commands::ServeInkling2Mini(exact) = crate::cli::Cli::try_parse_from(exact)?.command else {
            anyhow::bail!("Inkling 2 Mini command not parsed")
        };
        assert!(exact.prefill_packing.require_unrestricted("Inkling 2 Mini").is_err());
        crate::commands::bench::assert_resolved_family_seam("serve-inkling2-mini", &[], || {
            args.prefix.resolve_entries(args.max_sequences);
            let prefill = args.prefill()?;
            assert_eq!((prefill.unit_rows, prefill.waves), (2048, 2));
            cuteafd_bench::context::ResolvedSettings::prefill(prefill)
                .value("max-context-tokens", 262_144)
                .value("pool-tokens", 2_097_152)
                .publish();
            Ok(())
        })?;
        for invalid in ["0", "01", "1048577"] {
            let mut bad = argv.to_vec();
            let at = bad.iter().position(|value| *value == "--max-context-tokens").unwrap();
            bad[at + 1] = invalid;
            assert!(crate::cli::Cli::try_parse_from(bad).is_err());
        }
        Ok(())
    }
}
