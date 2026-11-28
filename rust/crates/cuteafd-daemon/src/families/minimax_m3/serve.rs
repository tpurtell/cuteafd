//! MiniMax whole-rank API on the shared request scheduler.
use super::startup;
use crate::shared::{prefix::PrefixArgs, inventory::{RuntimeInventory, InventoryPhase}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::expert_usage::Reader;
use cuteafd_loader::{families::minimax_m3::MiniMaxM3Config, placement::planner::PrefillShape, plan::Checkpoint};
use std::{net::SocketAddr, sync::Arc};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    /// Bounded decode/verify CUDA graph replay; false runs decode eagerly and charges no reserve.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub decode_graphs: bool,
    #[command(flatten)] pub media: crate::shared::vision::still::Options,
    #[command(flatten)] pub(crate) attention: crate::cli::CoreViewArgs,
    #[arg(long)] pub snapshot: std::path::PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] pub native_lib: std::path::PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] pub manifest: std::path::PathBuf,
    #[arg(long, default_value_t = 0)] pub device: i32,
    /// Distinct peer for contiguous attention with qualified FP32 split FFNs.
    #[arg(long)] pub split_device: Option<i32>,
    #[arg(long, env = "CUTEAFD_ATTENTION_PLACEMENT", default_value = "auto",
        value_parser = ["auto", "layers", "heads"])] pub attention_placement: String,
    #[arg(long, value_delimiter = ',')] pub peers: Vec<SocketAddr>,
    #[arg(long)] pub pool_tokens: Option<u64>,
    #[arg(long = "max-context-tokens", value_parser = crate::shared::context::parse_tokens::<usize>)]
    pub max_context: Option<usize>,
    #[arg(long, default_value = "auto")] pub rtx_expert_layers: cuteafd_loader::placement::OnboardSetting,
    /// Qualified split FFN wire. One-RTX execution does not transfer partials.
    #[arg(long, default_value = "fp32", value_parser = ["bf16", "fp32"])] pub split_wire: String,
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u8).range(1..=64))]
    pub max_sequences: u8,
    /// Per-lane decode/verify row budget (`DECODE_ROWS`: auto=64, up to 128).
    #[arg(long, default_value = "auto")]
    pub decode_rows: cuteafd_loader::plan::decode_rows::DecodeRowsSetting,
    #[arg(long, default_value = "0.0.0.0:8000")] pub listen: String,
    #[arg(long)] pub model_id: Option<String>,
    /// Bundled native DSpark; --dspark=false disables it.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set,
        num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    pub dspark: bool,
    #[command(flatten)] pub draft_length: crate::shared::draft::length::DraftLength,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set,
        num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    pub expert_usage: bool,
    #[arg(skip)] pub expert_usage_reader: Reader,
    #[arg(long)] pub prefill_rows: Option<usize>,
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=4))] pub prefill_waves: Option<u8>,
    #[arg(long, value_enum, default_value = "unrestricted")]
    pub prefill_packing: crate::shared::prefill_packing::PackingMode,
    #[command(flatten)] pub prefix: PrefixArgs,
    #[command(flatten)] pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)] pub loop_abort: crate::shared::loop_abort::LoopAbortArgs,
    #[command(flatten)] pub token_io: crate::shared::token_io::TokenIoArgs,
    #[command(flatten)] pub api: crate::shared::api::ApiArgs,
}

fn attention_placement(peer: Option<i32>, requested: &str) -> Result<cuteafd_loader::placement::AttentionPlacement> {
    use cuteafd_loader::placement::AttentionPlacement;
    match (peer, requested) {
        (_, "auto" | "heads") => Ok(AttentionPlacement::Heads),
        (Some(_), "layers") => Ok(AttentionPlacement::Layers),
        _ => anyhow::bail!("MiniMax layers attention requires two coordinator GPUs"),
    }
}

impl ServeArgs {
    fn prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        let plan = prefill_shape(self.peers.len(), cuteafd_loader::plan::Family::prefill_limits(
            &cuteafd_loader::plan::families::minimax::MINIMAX_M3), PrefillOverrides {
                unit_rows: self.prefill_rows.map(|rows| rows as u64), waves: self.prefill_waves.map(u64::from),
                span_rows: None, reserve_rows: None })?;
        cuteafd_bench::context::ResolvedSettings::prefill(plan).publish();
        Ok(plan)
    }
}

pub(crate) async fn run(mut args: ServeArgs) -> Result<()> {
    use crate::shared::qwen::scheduler::{self, Backend};
    use cuteafd_api::openai::{ModelProfile, NativeRequest};
    use std::sync::Mutex;
    cuteafd_core::concurrency::validate("minimax_m3", u64::from(args.max_sequences))?;
    let decode_rows = cuteafd_loader::plan::resolve_decode_rows(
        "minimax_m3",
        cuteafd_loader::plan::decode_rows::DecodeRowsRequest {
            requested: args.decode_rows.as_option(),
            concurrency: u64::from(args.max_sequences),
            decode_lanes: 1,
            rtx_gpus: 1 + usize::from(args.split_device.is_some()),
            draft_length: args.draft_length.draft_length.map(|n| n as u32),
        },
    ).map_err(|error| anyhow::anyhow!("{error}"))?;
    ensure!(matches!(args.token_io.embed_placement, crate::shared::token_io::EmbedPlacement::Gpu)
        && args.token_io.token_select == crate::shared::token_io::SelectPlacement::Device,
        "MiniMax serving requires admitted GPU embedding and device selection");
    cuteafd_bench::context::set(cuteafd_bench::context::ServerContext {
        command: "serve-minimax".into(), family: Some("minimax_m3".into()), snapshot: Some(args.snapshot.clone()),
        supports_thinking_off: Some(false), ..Default::default()
    });
    cuteafd_bench::context::ResolvedSettings::default().value("max-sequences", args.max_sequences)
        .value("decode-rows", decode_rows)
        .value("prefill-packing", format!("{:?}", args.prefill_packing).to_lowercase())
        .value("draft-mode", if args.dspark { "adaptive" } else { "no speculation" })
        .value("dspark",args.dspark).value("expert-usage", args.expert_usage).publish();
    args.prefix.resolve_entries(usize::from(args.max_sequences));
    let prefill = args.prefill()?;
    let api = args.api.load()?;
    // Native encoding is mandatory: no other family's template or output parser
    // may stand in for this checkpoint. Resolve it before native/device access.
    let encoding = super::encoding(&args.snapshot)
        .context("MiniMax API encoding not yet available")?;
    let eos = encoding.eos_token_ids();
    let model = args.model_id.clone().or_else(|| cuteafd_api::openai::snapshot_model_id(&args.snapshot))
        .context("MiniMax serving requires --model-id for a snapshot without HF identity")?;
    let mut profile = ModelProfile::new(model, encoding);
    profile.expert_usage = args.expert_usage_reader.clone();

    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(
        cuteafd_core::concurrency::queue_depth(u64::from(args.max_sequences)) as usize);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = Arc::clone(&stats);
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(usize, Option<Arc<cuteafd_api::openai::media::MediaPreparer>>, Option<Arc<std::sync::atomic::AtomicBool>>)>>();
    let worker_args = args.clone();
    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| -> Result<()> {
            let view = worker_args.attention.selected_view.clone().map(Ok)
                .unwrap_or_else(|| worker_args.attention.select(&worker_args.snapshot))?;
            worker_args.attention.validate_view(&view)?;
            crate::cli::publish_core_view(&view);
            let checkpoint = Checkpoint::coordinator_view(&view, false, false)?;
            let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
            cuteafd_bench::context::set_family("minimax_m3");
            cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::MINIMAX_M3)
                .map_err(|previous| anyhow::anyhow!("MiniMax serving needs an isolated process: expert geometry already fixed to {previous:?}"))?;
            // SAFETY: all native owners and the scheduler retire inside this worker
            // before the library and installed program manifest are dropped.
            let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&worker_args.native_lib)? });
            let programs = library.programs()?.with_manifest(&worker_args.manifest)?;
            let concurrency = usize::from(worker_args.max_sequences);
            let loaded = super::startup::load(&library, &programs, &view, &checkpoint, &cfg, super::startup::Options {
                media: &worker_args.media,
                device: worker_args.device, split_device: worker_args.split_device,
                peers: &worker_args.peers, native_lib: &worker_args.native_lib,
                manifest_path: &worker_args.manifest,
                attention_placement: attention_placement(worker_args.split_device, &worker_args.attention_placement)?,
                split_wire: match worker_args.split_wire.as_str() {
                    "bf16" => super::dense::PartialPrecision::Bf16,
                    _ => super::dense::PartialPrecision::DEFAULT,
                },
                prefill: PrefillShape { rows: prefill.unit_rows, waves: prefill.waves },
                decode_rows,
                sequences: concurrency as u64, context_tokens: worker_args.max_context.unwrap_or(0) as u64,
                pool_tokens: worker_args.pool_tokens.filter(|&tokens| tokens > 0),
                onboard: worker_args.rtx_expert_layers.resolve(cfg.moe.iter().filter(|&&moe| moe).count()),
                prefix: &worker_args.prefix, decode_graphs: worker_args.decode_graphs, expert_usage: worker_args.expert_usage,
                usage_reader: worker_args.expert_usage_reader.clone(),
                resident_weights: crate::shared::memory::budget::resident_weight_limit(),draft:worker_args.dspark })?;
            if loaded.owners.storage.draft_prefix.is_some() {
                let physical = std::iter::once(worker_args.device).chain(worker_args.split_device).collect::<Vec<_>>();
                crate::shared::draft::report::publish(drafter_report(&physical, &loaded.admitted.solved)?);
            }
            let served_context = usize::try_from(loaded.admitted.solved.served_context())?;
            let backend = super::backend::MiniMaxBackend::new(&loaded.owners, &cfg,
                loaded.execution, served_context, loaded.host_prefix, worker_args.prefill_packing)?.with_probe_snapshot(&worker_args.snapshot)
                .with_media_enabled(loaded.media.preparer.is_some());
            let image = loaded.media.preparer.clone();
            let health = loaded.media.encoder.health_handle();
            let mut media = cuteafd_engine::media::MediaAdmission::new(
                cuteafd_engine::media::EmbeddingCache::new(loaded.media.cache_bytes), loaded.media.encoder, 16);
            library.cuda_set_device(worker_args.split_device.unwrap_or(worker_args.device))?;
            let physical = std::iter::once(worker_args.device).chain(worker_args.split_device).collect::<Vec<_>>();
            scheduler::schedule(&backend, &worker_args.snapshot, &mut receive, &worker_stats,
                concurrency, if worker_args.dspark { scheduler::Drafts::Neural { depth:7 } } else { scheduler::Drafts::None },
                worker_args.draft_length, eos, worker_args.decode_share,
                &worker_args.prefix, worker_args.loop_abort.config(), worker_args.token_io.token_select,
                &mut media, None, || {
                    backend.require_live()?;
                    RuntimeInventory::measure(&library, &physical, InventoryPhase::PostLoad,
                        |_| Ok(()))?.record("minimax_m3-serving-loaded", &loaded.admission_free);
                    if let Some(ready) = ready.take() { let _ = ready.send(Ok((served_context, image, health))); }
                    Ok(())
                })
        })();
        crate::shared::serve::forward_initialization(result, ready.is_some(), "MiniMax initialization failed", |error| {
            ready.take().expect("pending readiness").send(Err(error)).map_err(|result| match result {
                Err(error) => error, Ok(_) => unreachable!("sent an error") })
        })
    });
    let (context, image, health) = ready_rx.await.context("MiniMax worker exited before readiness")??;
    if let Some(image) = image { profile = profile.with_grouped_vision(image); profile.vision_health=health; }
    let limits = args.prefix.reserve.limits(context)?;
    let snapshot = args.snapshot.clone();
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new(
        "minimax_m3", profile.id.clone(), snapshot.clone())));
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats,
        std::time::Duration::from_secs(25), hub.clone(), api.serve(profile, &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    let mut worker = Some(worker);
    tokio::select! {
        served = crate::shared::api::serve_http(listener, api.app(router, hub), "minimax_m3") => served?,
        _ = crate::shared::api::watch_scheduler(&mut worker) => unreachable!(),
    }
    crate::shared::api::finish_scheduler(&mut worker, "minimax_m3").await?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundled_dspark_defaults_on_for_all_placements_and_has_explicit_off() -> Result<()> {
        use clap::Parser;
        let base = ["cuteafd", "serve-minimax", "--snapshot", "/fixture", "--native-lib", "/native"];
        for placement in ["auto", "heads", "layers"] {
            for peer in [None, Some("1")] {
                for (flag, enabled) in [(None, true), (Some("--dspark"), true),
                    (Some("--dspark=false"), false), (Some("--dspark=true"), true)] {
                    let mut argv = base.to_vec();
                    argv.extend(["--attention-placement", placement]);
                    if let Some(peer) = peer { argv.extend(["--split-device", peer]); }
                    if let Some(flag) = flag { argv.push(flag); }
                    let crate::cli::Commands::ServeMinimax(args) = crate::cli::Cli::try_parse_from(argv)?.command
                        else { unreachable!() };
                    assert_eq!(args.dspark, enabled);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn serve_fixes_expert_geometry_before_any_native_library_load() {
        let source = include_str!("serve.rs").split_once("pub(crate) async fn run(").unwrap().1
            .split_once("#[cfg(test)]").unwrap().0;
        let fixed = source.find("cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::MINIMAX_M3)").unwrap();
        let native = source.find("cuteafd_ffi::NativeLibrary::load(").unwrap();
        assert!(fixed < native, "native loading syncs expert hidden size and fixes the process default");
        assert_eq!(source.matches("set_expert_geometry(").count(), 1);
        assert!(source.contains("super::startup::load("), "constructor still validates the same fixed geometry");
    }

    #[test]
    fn auto_attention_matches_native_planner_on_one_and_two_rtx() -> Result<()> {
        use cuteafd_loader::placement::AttentionPlacement;
        for peer in [None, Some(1)] {
            assert_eq!(attention_placement(peer, "auto")?, AttentionPlacement::Heads);
            assert_eq!(attention_placement(peer, "heads")?, AttentionPlacement::Heads);
        }
        assert_eq!(attention_placement(Some(1), "layers")?, AttentionPlacement::Layers);
        assert!(attention_placement(None, "layers").is_err());
        Ok(())
    }

    #[test]
    fn hidden_serve_cli_and_runtime_settings_have_a_cpu_seam() -> Result<()> {
        use clap::Parser;
        let argv = ["cuteafd", "serve-minimax", "--snapshot", "/fixture", "--native-lib", "/native",
            "--max-sequences", "16", "--prefill-rows", "2752", "--prefill-waves", "3"];
        let crate::cli::Commands::ServeMinimax(mut args) = crate::cli::Cli::try_parse_from(argv)?.command else {
            panic!("MiniMax serve command");
        };
        assert_eq!(args.split_wire, "fp32");
        crate::commands::bench::assert_resolved_family_seam("serve-minimax", &[], || {
            args.prefix.resolve_entries(16);
            args.prefill()?;
            cuteafd_bench::context::ResolvedSettings::default().value("max-context-tokens", 131072)
                .value("pool-tokens", 2097152).value("rtx-expert-layers", 0)
                .value("decode-rows", 64)
                .value("attention-placement", "heads").publish();
            Ok(())
        })
    }

}

fn drafter_report(devices: &[i32], plan: &cuteafd_loader::placement::planner::ResolvedPlan) -> Result<crate::shared::draft::report::DrafterLayout> {
    use crate::shared::draft::report::{self, Kind, Placement};
    let lead = plan.decisions.draft_home.context("MiniMax admitted draft head absent")?;
    report::admitted(Kind::Dspark, Placement::Whole, lead, devices, plan)
}

#[cfg(test)]
#[test]
fn minimax_reports_whole_dspark_on_the_admitted_head() -> Result<()> {
    use crate::shared::draft::report::{Kind, Placement, Rank, tests::{owner, plan}};
    let mut admitted = plan(vec![owner(1, 4096)]);
    admitted.decisions.draft_home = Some(1);
    let layout = drafter_report(&[4, 7], &admitted)?;
    assert_eq!((layout.kind, layout.placement, layout.lead), (Kind::Dspark, Some(Placement::Whole), Some(Rank::CudaRank(7))));
    assert_eq!(layout.arena_bytes, [4096]);
    Ok(())
}
