//! Inkling HTTP/request lifecycle over the shared scheduler.
use crate::shared::{prefix::PrefixArgs, token_io::TokenIoArgs};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{families::inkling2::Inkling2Config, placement::planner::PrefillShape, plan::Checkpoint};
use std::{net::SocketAddr, path::PathBuf, sync::{Arc, Mutex}};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    /// Bounded rank-local decode/verify graph replay; prefill and peer exchanges stay eager.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub decode_graphs: bool,
    #[command(flatten)] pub vision: crate::shared::vision::still::Options,
    #[command(flatten)] pub attention: crate::cli::CoreViewArgs,
    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long, env="CUTEAFD_NATIVE_LIB")] pub native_lib: PathBuf,
    #[arg(long, default_value="/opt/cuteafd/share/PROGRAMS.json")] pub manifest: PathBuf,
    #[arg(long, default_value_t=0)] pub device: i32,
    #[arg(long)] pub split_device: Option<i32>,
    #[arg(long, value_delimiter=',')] pub peers: Vec<SocketAddr>,
    #[arg(long)] pub pool_tokens: Option<u64>,
    #[arg(long="kv-cache", default_value="fp8")] pub kv_format: cuteafd_loader::families::inkling2::InklingKvCache,
    #[arg(long,default_value_t=true,action=clap::ArgAction::Set,num_args=0..=1,default_missing_value="true")] pub dspark:bool,
    #[arg(long,default_value="fp8")] pub draft_kv_cache:cuteafd_loader::families::qwen4::Qwen4KvCache,
    #[arg(long="max-context-tokens", value_parser=crate::shared::context::parse_tokens::<usize>)]
    pub max_context: Option<usize>,
    #[arg(long, default_value="auto")] pub rtx_expert_layers: cuteafd_loader::placement::OnboardSetting,
    #[arg(long, default_value_t=16)] pub max_sequences: usize,
    #[arg(long, default_value_t = cuteafd_loader::plan::decode_rows::DecodeRowsSetting::AUTO)]
    pub decode_rows: cuteafd_loader::plan::decode_rows::DecodeRowsSetting,
    #[arg(long, default_value="0.0.0.0:8000")] pub listen: String,
    #[arg(long)] pub model_id: Option<String>,
    #[arg(long)] pub prefill_rows: Option<usize>,
    #[arg(long, value_parser=clap::value_parser!(u8).range(1..=3))] pub prefill_waves: Option<u8>,
    /// `exact` needs an attested coordinator route cap, which Inkling does not report yet.
    #[arg(long, value_enum, default_value="unrestricted")] pub prefill_packing: crate::shared::prefill_packing::PackingMode,
    #[command(flatten)] pub prefix: PrefixArgs,
    #[command(flatten)] pub draft_length: crate::shared::draft::length::DraftLength,
    #[command(flatten)] pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)] pub loop_abort: crate::shared::loop_abort::LoopAbortArgs,
    #[command(flatten)] pub token_io: TokenIoArgs,
    #[command(flatten)] pub api: crate::shared::api::ApiArgs,
}
impl ServeArgs {
    fn prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        Ok(prefill_shape(self.peers.len(), cuteafd_loader::plan::Family::prefill_limits(
            &cuteafd_loader::plan::families::inkling2::INKLING2), PrefillOverrides {
            unit_rows: self.prefill_rows.map(|rows| rows as u64), waves: self.prefill_waves.map(u64::from),
            span_rows: None, reserve_rows: None })?)
    }
}

pub(crate) async fn run(mut args: ServeArgs) -> Result<()> {
    use crate::shared::{qwen::scheduler::{self, Backend}, inventory::{InventoryPhase, RuntimeInventory}};
    use cuteafd_api::openai::{ModelEncoding, ModelProfile, NativeRequest, chat::inkling2::InklingEncoding};
    cuteafd_core::concurrency::validate("inkling2", args.max_sequences as u64)?;
    let gpus = if args.split_device.is_some() { 2 } else { 1 };
    let decode_rows = cuteafd_loader::plan::decode_rows::resolve(
        &cuteafd_loader::plan::families::inkling2::INKLING2,
        cuteafd_loader::plan::decode_rows::DecodeRowsRequest {
            requested: args.decode_rows.as_option(),
            concurrency: args.max_sequences as u64,
            decode_lanes: 1,
            rtx_gpus: gpus,
            draft_length: args.draft_length.draft_length.map(|n| n as u32),
        },
    )?;
    args.prefill_packing.require_unrestricted("Inkling")?;
    ensure!(args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu
        && args.token_io.token_select == crate::shared::token_io::SelectPlacement::Device,
        "Inkling initial serving requires admitted GPU embedding and selection");
    args.prefix.resolve_entries(args.max_sequences);
    let prefill = args.prefill()?;
    cuteafd_bench::context::ResolvedSettings::prefill(prefill).publish();
    let encoding = ModelEncoding::Inkling(Arc::new(InklingEncoding::from_snapshot(&args.snapshot)?));
    let eos = encoding.eos_token_ids();
    ensure!(eos == [200006], "Inkling native EOS differs");
    let model = args.model_id.clone().or_else(|| cuteafd_api::openai::snapshot_model_id(&args.snapshot))
        .context("Inkling snapshot without HF identity requires --model-id")?;
    let profile = ModelProfile::new(model, encoding);
    let api = args.api.load()?;
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(
        cuteafd_core::concurrency::queue_depth(args.max_sequences as u64) as usize);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(usize, Option<Arc<cuteafd_api::openai::media::MediaPreparer>>)>>();
    let worker_args = args.clone();
    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| {
            let view = worker_args.attention.selected_view.clone().map(Ok)
                .unwrap_or_else(|| worker_args.attention.select(&worker_args.snapshot))?;
            worker_args.attention.validate_view(&view)?;
            // Admission inventories remote expert scales as well as target tensors;
            // opening headers does not load or allocate the remote weights.
            let checkpoint = Checkpoint::open_view(&view)?;
            let cfg = Inkling2Config::from_hf(&checkpoint.config)?;
            cuteafd_bench::context::set_family("inkling2");
            cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::INKLING2)
                .map_err(|previous| anyhow::anyhow!("Inkling process geometry already fixed to {previous:?}"))?;
            // SAFETY: every device, transport and graph owner retires inside
            // this worker before its native library and program table drop.
            let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&worker_args.native_lib)? });
            let programs = library.programs()?.with_manifest(&worker_args.manifest)?;
            let (execution, inputs, admitted, _, media) = super::startup::whole(&library, &programs, &view, &checkpoint, &cfg,
                super::startup::Options { device: worker_args.device, split_device: worker_args.split_device, peers: &worker_args.peers,
                    native_lib: &worker_args.native_lib, manifest: &worker_args.manifest,
                    prefill: PrefillShape { rows: prefill.unit_rows, waves: prefill.waves },
                    decode_rows,
                    sequences: worker_args.max_sequences as u64, context_tokens: worker_args.max_context.unwrap_or(0) as u64,
                    pool_tokens: worker_args.pool_tokens, kv_format: worker_args.kv_format, drafter:worker_args.dspark.then_some(worker_args.draft_kv_cache), onboard: worker_args.rtx_expert_layers.resolve(cfg.layers),
                    prefix: &worker_args.prefix, resident_weights: crate::shared::memory::budget::resident_weight_limit(),
                    expert_usage: false, vision: &worker_args.vision, decode_graphs: worker_args.decode_graphs })?;
            let context = usize::try_from(admitted.solved.served_context())?;
            let family = super::cache::InklingPrefix::for_execution(&execution);
            let host = (inputs.host_prefix.quota_bytes > 0).then(|| {
                use cuteafd_engine::prefix::PrefixFamily;
                worker_args.prefix.host_config_for(family.layout(), context, inputs.host_prefix.quota_bytes)
            }).transpose()?;
            let drafts = execution.draft.as_ref().map_or(scheduler::Drafts::None, |draft|
                scheduler::Drafts::Neural { depth: draft.layout.config.transformer.block - 1 });
            let binding = crate::shared::vision::perception::Binding::new(&worker_args.snapshot,
                cuteafd_loader::media::ImageFamily::Inkling2, media.preparer.is_some(), cfg.unpadded_vocab_size as u32)?;
            let backend = super::backend::InklingBackend::new(execution, context, host, &admitted.solved)?.with_vision(binding);
            let mut media_admission = cuteafd_engine::media::MediaAdmission::new(
                cuteafd_engine::media::EmbeddingCache::new(media.cache_bytes), media.encoder, worker_args.max_sequences);
            library.cuda_set_device(worker_args.device)?;
            scheduler::schedule(&backend, &worker_args.snapshot, &mut receive, &worker_stats,
                worker_args.max_sequences, drafts, worker_args.draft_length, eos,
                worker_args.decode_share, &worker_args.prefix, worker_args.loop_abort.config(),
                worker_args.token_io.token_select, &mut media_admission, None, || {
                    backend.require_live()?;
                    let devices = std::iter::once(worker_args.device).chain(worker_args.split_device).collect::<Vec<_>>();
                    RuntimeInventory::measure(&library, &devices, InventoryPhase::PostLoad,
                        |_| Ok(()))?.record("inkling2-serving-loaded", &[]);
                    if let Some(ready) = ready.take() { let _ = ready.send(Ok((context, media.preparer.clone()))); }
                    Ok(())
                })
        })();
        crate::shared::serve::forward_initialization(result, ready.is_some(), "Inkling initialization failed", |error|
            ready.take().expect("pending readiness").send(Err(error)).map_err(|result| match result {
                Err(error) => error, Ok(_) => unreachable!("sent error") }))
    });
    let (context, preparer) = ready_rx.await.context("Inkling worker exited before readiness")??;
    let profile = if let Some(preparer) = preparer { profile.with_grouped_vision(preparer) } else { profile };
    let limits = args.prefix.reserve.limits(context)?;
    let snapshot = args.snapshot.clone();
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new("inkling2", profile.id.clone(), snapshot.clone())));
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, std::time::Duration::from_secs(25),
        hub.clone(), api.serve(profile, &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    let mut worker = Some(worker);
    tokio::select! {
        served = crate::shared::api::serve_http(listener, api.app(router, hub), "inkling2") => served?,
        _ = crate::shared::api::watch_scheduler(&mut worker) => unreachable!(),
    }
    crate::shared::api::finish_scheduler(&mut worker, "inkling2").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn launcher_argv_and_resolved_startup_seam_match() -> anyhow::Result<()> {
        let argv = ["cuteafd", "--gpu-memory-cap", "97250MiB", "serve-inkling2",
            "--snapshot", "/model", "--coordinator-weight-budget-gib", "94.97",
            "--native-lib", "/opt/cuteafd/lib/libcuteafd_native.so",
            "--peers", "10.55.0.1:19923,10.55.0.2:19923,10.55.0.3:19923,10.55.0.4:19923,10.55.0.5:19923,10.55.0.6:19923",
            "--listen", "0.0.0.0:8823", "--max-sequences", "16", "--max-context-tokens", "auto",
            "--max-output-tokens", "131072", "--embedding-placement", "gpu", "--core-view", "original",
            "--prefix-cache-entries", "20", "--host-cache-bytes", "auto", "--rtx-expert-layers", "auto",
            "--prefill-rows", "2752", "--prefill-waves", "3",
            "--api-key-file", "/run/cuteafd-api-key", "--enable-bench", "--usage", "on", "--gateway", "on",
            "--console-secret-file", "/run/cuteafd-console-secret", "--usage-dir", "/run/cuteafd/usage"];
        let crate::cli::Commands::ServeInkling2(mut args) = crate::cli::Cli::try_parse_from(argv)?.command
            else { anyhow::bail!("Inkling command not parsed") };
        assert_eq!((args.max_context, args.max_sequences, args.peers.len()), (Some(0), 16, 6));
        assert_eq!(args.prefill_packing, crate::shared::prefill_packing::PackingMode::Unrestricted);
        let mut exact = argv.to_vec();
        exact.extend(["--prefill-packing", "exact"]);
        let crate::cli::Commands::ServeInkling2(exact) = crate::cli::Cli::try_parse_from(exact)?.command
            else { anyhow::bail!("Inkling command not parsed") };
        assert!(exact.prefill_packing.require_unrestricted("Inkling").is_err());
        crate::commands::bench::assert_resolved_family_seam("serve-inkling2", &[], || {
            args.prefix.resolve_entries(args.max_sequences);
            let prefill = args.prefill()?;
            assert_eq!((prefill.unit_rows, prefill.waves), (2752, 3));
            cuteafd_bench::context::ResolvedSettings::default()
                .value("max-context-tokens", 262144).value("pool-tokens", 2097152)
                .value("rtx-expert-layers", 0).value("prefill-rows", prefill.unit_rows)
                .value("prefill-waves", prefill.waves).value("decode-rows", 64).publish();
            Ok(())
        })?;
        let mut wide = argv.to_vec();
        wide.extend(["--decode-rows", "128"]);
        let crate::cli::Commands::ServeInkling2(wide) = crate::cli::Cli::try_parse_from(wide)?.command
            else { anyhow::bail!("Inkling command not parsed") };
        assert_eq!(wide.decode_rows.as_option(), Some(128));
        for invalid in ["0", "01", "1048577"] {
            let mut bad = argv.to_vec();
            let at = bad.iter().position(|value| *value == "--max-context-tokens").unwrap();
            bad[at+1] = invalid;
            assert!(crate::cli::Cli::try_parse_from(bad).is_err());
        }
        Ok(())
    }

    #[test]
    fn shared_scheduler_owns_lanes_commit_and_last_fallible_readiness() {
        let source = include_str!("serve.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("scheduler::schedule(&backend"));
        assert!(source.contains("Checkpoint::open_view(&view)?"));
        assert!(source.contains("context::set_family(\"inkling2\")"));
        assert!(!source.contains("Checkpoint::coordinator_view("));
        assert!(source.contains("forward_initialization("));
        assert!(source.contains("finish_scheduler(&mut worker"));
        assert!(source.find("InklingEncoding::from_snapshot").unwrap() < source.find("NativeLibrary::load").unwrap());
        let ready = source.find("ready.send(Ok((context, media.preparer.clone())))").unwrap();
        assert!(source.find("startup::whole(").unwrap() < ready);
        assert!(source.find("InklingBackend::new(").unwrap() < ready);
        assert!(source.find("backend.require_live()?").unwrap() < ready);
    }
}
