//! Muse Spark serving entry: validated arguments, program preflight, the
//! measured pre-weight inventory and shared admission, then the bring-up:
//! Spark worker admission (the sealed endpoint), every coordinator rank's
//! resident state (and onboarded experts) loaded while the workers admit, the
//! pair's exchange and the Spark link. The shared scheduler then serves the
//! text path over `MuseBackend` (with an optional DFlash2 drafter) behind the OpenAI router with
//! the checkpoint's own chat encoding.
use super::startup;
use crate::shared::inventory::{InventoryPhase, RuntimeInventory};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::muse_spark::{core::CoreInventory, MuseSparkConfig};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] pub manifest: PathBuf,
    /// The entry coordinator GPU; `--split-device` adds the second of a head-split
    /// pair (attention heads halve), as every family's launcher passes them.
    #[arg(long, default_value_t = 0)] pub device: i32,
    #[arg(long)] pub split_device: Option<i32>,
    /// Spark expert ranks: every expert's intermediate is sliced across them (TP),
    /// each replying per-expert partials the coordinator combines.
    #[arg(long, value_delimiter = ',')] pub peers: Vec<SocketAddr>,
    /// Auto admission grows beyond the selected context floor while memory allows.
    #[arg(long)] pub pool_tokens: Option<u64>,
    /// API ceiling and compiled extent; unset takes the checkpoint's 1M with the
    /// pool deciding the admitted context.
    #[arg(long)] pub max_context: Option<u64>,
    #[arg(long, default_value = "auto")] pub rtx_expert_layers: cuteafd_loader::placement::Onboard,
    /// Complete per-rank Spark allocation ceiling, estimated until the first worker census.
    #[arg(long)] pub spark_budget_gib: Option<f64>,
    #[arg(long, default_value = "0.0.0.0:8000")] pub listen: String,
    #[arg(long, default_value_t = 32768)] pub max_output: u32,
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u8).range(1..=64))]
    pub max_sequences: u8,
    #[arg(long)] pub model_id: Option<String>,
    #[arg(long, default_value = "auto")] pub vision: cuteafd_loader::plan::MediaMode,
    #[arg(long, default_value = "auto")] pub audio: cuteafd_loader::plan::MediaMode,
    #[arg(long, value_delimiter = ',')] pub vision_peers: Vec<SocketAddr>,
    #[arg(long, requires = "vision_peers")] pub encoder_plan_hash: Option<String>,
    #[arg(long, requires = "vision_peers")] pub encoder_revision: Option<String>,
    #[arg(long, value_delimiter = ',')] pub audio_peers: Vec<SocketAddr>,
    #[arg(long, requires = "audio_peers")] pub audio_encoder_plan_hash: Option<String>,
    #[arg(long, requires = "audio_peers")] pub audio_encoder_revision: Option<String>,
    #[arg(long)] pub media_cache_bytes: Option<u64>,
    /// Opt in to a DFlash2 drafter snapshot (RadixArk/Muse-Spark-Open-DFlash2); it runs
    /// beside the target head and reads the target's exact head (softcap and multiplier).
    #[arg(long)] pub draft: Option<PathBuf>,
    /// Fixed draft depth; absent lets the shared policy choose up to the drafter's block.
    #[arg(long, requires = "draft", value_parser = clap::value_parser!(u8).range(1..=15))]
    pub draft_fixed: Option<u8>,
    /// The drafter's ring storage (independent of weight precision).
    #[arg(long, default_value = "fp8")]
    pub draft_kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
    /// Prefill unit rows, at most 4096; unset follows the shared wave policy.
    #[arg(long)] pub prefill_rows: Option<usize>,
    /// Prefill units in flight on the one Spark transport; unset follows the shared policy.
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=4))]
    pub prefill_waves: Option<u8>,
    /// Spark expert partial replies: a request contributing at most this many rows to a
    /// step replies BF16, wider segments NVFP4 (the shared EXL3 worker's reply cutoff).
    /// Must cover the widest verify step; the default keeps decode and verify lossless.
    #[arg(long, default_value_t = cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS)]
    pub expert_partial_bf16_max_rows: usize,
    /// Partial format of segments above the cutoff: nvfp4 (E2M1, E4M3 per 16) or fp8 (E4M3, UE8M0 per 32).
    #[arg(long, default_value = "nvfp4")]
    pub expert_partial_wide_format: cuteafd_loader::families::muse_spark::replies::WideFormat,
    #[command(flatten)] pub prefix: crate::shared::prefix::PrefixArgs,
    #[command(flatten)] pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)] pub token_io: crate::shared::token_io::TokenIoArgs,
    #[command(flatten)] pub api: crate::shared::api::ApiArgs,
}

impl ServeArgs {
    /// The coordinator GPUs in rank order.
    pub(super) fn devices(&self) -> Vec<i32> { std::iter::once(self.device).chain(self.split_device).collect() }

    pub(super) fn prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        Ok(prefill_shape(self.peers.len(), cuteafd_loader::plan::Family::prefill_limits(
            &cuteafd_loader::plan::families::muse_spark::MUSE_SPARK), PrefillOverrides {
            unit_rows: self.prefill_rows.map(|rows| rows as u64),
            waves: self.prefill_waves.map(u64::from), span_rows: None, reserve_rows: None,
        })?)
    }

    fn spark_budget_bytes(&self) -> u64 {
        self.spark_budget_gib.map_or(cuteafd_loader::plan::DEFAULT_SPARK_BUDGET_BYTES,
            |gib| (gib * (1u64 << 30) as f64) as u64)
    }

    pub(super) fn validate(&self) -> Result<()> {
        let devices = self.devices();
        cuteafd_loader::placement::families::muse_spark::executor(devices.len())?;
        ensure!(devices.iter().all(|&device| device >= 0)
            && devices.iter().collect::<std::collections::HashSet<_>>().len() == devices.len(),
            "Muse Spark coordinator devices must be distinct");
        let distinct = self.peers.iter().collect::<std::collections::HashSet<_>>().len();
        // Every Spark world the shared package range data gives the checkpoint's width.
        let worlds = cuteafd_loader::plan::experts::spark_worlds(cuteafd_loader::plan::families::muse_spark::PACKAGE, 4096);
        ensure!(worlds.contains(&self.peers.len()) && distinct == self.peers.len(),
            "Muse Spark serves its EXL3 TP experts on {worlds:?} distinct Spark endpoints, not {}", self.peers.len());
        ensure!(self.max_context.is_none_or(|context| (1..=1 << 20).contains(&context)),
            "Muse Spark context exceeds the checkpoint's 1M extent");
        ensure!(self.spark_budget_gib.is_none_or(|gib| gib.is_finite() && gib > 0.0
            && gib < u64::MAX as f64 / (1u64 << 30) as f64),
            "Muse Spark requires a finite positive --spark-budget-gib");
        self.prefill()?;
        for (name, mode, peers, hash, revision) in [
            ("vision", self.vision, &self.vision_peers, &self.encoder_plan_hash, &self.encoder_revision),
            ("audio", self.audio, &self.audio_peers, &self.audio_encoder_plan_hash, &self.audio_encoder_revision),
        ] {
            use cuteafd_loader::plan::MediaMode;
            let remote = !peers.is_empty();
            ensure!(!remote || matches!(mode, MediaMode::Auto | MediaMode::Spark(_)),
                "Muse {name} encoder peers require auto or Spark placement");
            ensure!(!matches!(mode, MediaMode::Spark(_)) || remote,
                "Muse Spark {name} requires planner-resolved encoder peers");
            ensure!(remote == hash.is_some() && remote == revision.is_some(),
                "Muse {name} encoder peers, plan hash and revision must be supplied together");
            ensure!(peers.len() <= 8 && peers.iter().collect::<std::collections::HashSet<_>>().len() == peers.len(),
                "Muse {name} encoder peers must be distinct and at most eight");
            if let Some(hash) = hash { crate::shared::vision::worker::parse_plan_hash(hash)?; }
            ensure!(revision.as_ref().is_none_or(|revision| !revision.trim().is_empty()),
                "Muse {name} encoder revision cannot be empty");
        }
        ensure!(self.audio_peers.len() <= 1, "Muse audio admits one encoder owner");
        // A request's widest step is a verify (anchor plus drafts, at most the 64-row
        // rollback step); the cutoff must keep it BF16, so decode and verify stay lossless.
        cuteafd_loader::families::muse_spark::replies::Cutoff::new(self.expert_partial_bf16_max_rows,
            cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS)?;
        ensure!(self.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Muse Spark serving requires exact prefix frontiers");
        Ok(())
    }
}

/// Bring the engine up and hand `run` the backend and its admitted sequences.
fn with_backend<T>(args: &ServeArgs, run: impl FnOnce(&super::backend::MuseBackend<'_, '_>, usize, super::media::Ready) -> Result<T>)
    -> Result<T> {
    ensure!(args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu,
        "Muse Spark keeps its embedding on the entry GPU");
    cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::MUSE_SPARK)
        .map_err(|_| anyhow::anyhow!("Muse Spark serving needs an isolated process"))?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&args.snapshot, false, false)?;
    let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
    cfg.release_geometry()?;
    let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
    let device_ids = args.devices();
    let gpus = device_ids.len();
    let manifest = startup::preflight(&args.manifest, gpus)?;
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    // SAFETY: every native owner drops on this worker before the library.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&args.native_lib)? });
    let inventory = RuntimeInventory::measure(&library, &device_ids, InventoryPhase::PreWeight, |_| Ok(()))?;
    let (built, placement) = startup::admit(startup::Options { cfg: &cfg, core: &core, catalog: &catalog,
        manifest: &manifest,
        measured: inventory.gpus.iter().map(|sample| startup::Measured { total_bytes: sample.total_bytes,
            free_bytes: sample.free_bytes, untracked_bytes: sample.context_bytes }).collect(),
        headroom_bytes: 2 << 30, spark_ranks: args.peers.len(), prefill: args.prefill()?,
        slots: u64::from(args.max_sequences),
        mark_slots: args.prefix.mark_rule(usize::from(args.max_sequences))
            .slots(cuteafd_loader::placement::families::muse_spark::mark_bytes(&cfg, 1) as usize) as u64,
        max_context: args.max_context,
        pool_tokens: args.pool_tokens.filter(|&tokens| tokens > 0), onboard: args.rtx_expert_layers,
        spark_budget_bytes: args.spark_budget_bytes(),
        partial_bf16_max_rows: args.expert_partial_bf16_max_rows,
        partial_wide_format: args.expert_partial_wide_format,
        local: cuteafd_loader::placement::families::muse_spark::local_workspace(&catalog,
            Some(&args.manifest), gpus, args.prefill()?.unit_rows)?,
        drafter: args.draft.as_deref().map(|snapshot| -> Result<_> {
            let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))
                .with_context(|| format!("Muse drafter config {}", snapshot.display()))?)?;
            let head = core.get_weight(cuteafd_loader::families::muse_spark::core::HEAD)
                .map(|head| head.stored_outputs).context("Muse drafter needs the K6 target head")?;
            let head_device = crate::shared::memory::device::Device { library: &library,
                id: *device_ids.last().context("Muse devices")? };
            let sms = head_device.run(|| library.sm_count())?;
            cuteafd_loader::placement::families::muse_spark::drafter_inventory(&cfg, &config,
                usize::from(args.max_sequences), usize::from(args.max_sequences), head, sms as u64,
                args.draft_kv_cache)
        }).transpose()?, media: Some(super::media::inventory(&args.snapshot,
            args.vision, args.audio, args.vision_peers.len().max(1))?) })?;
    tracing::info!(placement = %placement.summary(), rtx_expert_layers = placement.onboard_layers,
        pool_tokens = placement.pool_tokens, k6_programs = built.projections.iter().map(|r| r.programs.len()).sum::<usize>(),
        "Muse Spark admitted TP experts with coordinator-combined partials");
    let media = super::media::Ready::load(args, &library, built.media.as_ref().context("Muse media admission missing")?)?;
    let prefill = args.prefill()?;
    let cutoff = cuteafd_loader::families::muse_spark::replies::Cutoff::with_wide(args.expert_partial_bf16_max_rows,
        cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS, args.expert_partial_wide_format)?;
    let sequences = usize::from(args.max_sequences);
    let spark = super::admission::spark_plan(&placement, prefill, cutoff, sequences, args.peers.len())?;
    let admission = crate::shared::placement::SparkAdmission::plan(&args.snapshot,
        cuteafd_core::ExpertGeometry::MUSE_SPARK, Some(&catalog), &spark.working, args.peers.len(), None,
        spark.model.clone())?.start(&args.peers)?;
    let devices = device_ids.iter().map(|&id| crate::shared::memory::device::Device { library: &library, id })
        .collect::<Vec<_>>();
    let programs = devices.iter().map(|device| device.run(|| library.programs()?.with_manifest(&args.manifest)))
        .collect::<Result<Vec<_>>>()?;
    let context = args.max_context.unwrap_or(cfg.max_position as u64);
    // The BF16 token embedding on the entry GPU (single residency; rank 1 gathers over
    // peer access), filled in the background while the layers load.
    let name = "model.language_model.embed_tokens.weight";
    let tensor = checkpoint.tensors.iter().find(|tensor| tensor.meta.name == name)
        .with_context(|| format!("Muse checkpoint lacks {name}"))?;
    let source = crate::shared::token_io::EmbedSource::new(&checkpoint.snapshot, &tensor.shard, &tensor.meta,
        cfg.hidden)?;
    // Onboarded layers (experts on the RTXs): the installed coordinator package, each GPU's
    // half (a pair) or whole layer, loaded after the dense weights inside the same admission window.
    let onboarded = placement.layers.iter().enumerate().filter_map(|(layer, assignment)| matches!(assignment.experts,
        cuteafd_loader::placement::ExpertHome::RtxTp2 | cuteafd_loader::placement::ExpertHome::RtxWhole { .. })
        .then_some(layer)).collect::<Vec<_>>();
    let package = if onboarded.is_empty() { None } else {
        Some(cuteafd_loader::placement::families::muse_spark::local_package(&catalog, &args.manifest, gpus,
            prefill.unit_rows)?.0)
    };
    let program_refs = programs.iter().collect::<Vec<_>>();
    let (embedding, (ranks, local)) = crate::shared::placement::load_while_admitting(Some(admission), || {
        devices[0].run(|| crate::shared::token_io::TokenEmbedding::load(&library, source.clone(),
            crate::shared::memory::EmbedPlacement::Gpu, || {
            devices.iter().zip(&programs).enumerate().map(|(rank, (&device, programs))|
                super::engine::RankResources::load(device, programs, super::engine::RankLoad { cfg: &cfg,
                    checkpoint: &checkpoint, built: &built, placement: &placement, rank, slots: sequences as u64,
                    unit_rows: usize::try_from(prefill.unit_rows)?, waves: usize::try_from(prefill.waves)?, context }))
                .collect::<Result<Vec<_>>>()
                .and_then(|ranks| Ok((ranks, match &package {
                    None => Vec::new(),
                    Some(package) => super::onboard::load(&devices, &program_refs, &cfg, &catalog, package, &placement,
                        onboarded.clone(), usize::try_from(prefill.unit_rows)?)?,
                })))
        }))
    })?;
    tracing::info!(layers = ?onboarded, gpus = local.len(), "Muse onboarded expert layers loaded");
    let streams = devices.iter().map(|&device| crate::shared::memory::device::Stream::new(device).map(std::rc::Rc::new))
        .collect::<Result<Vec<_>>>()?;
    let (intake_row_bytes, _) = cuteafd_loader::placement::families::muse_spark::intake(&cfg, prefill, cutoff,
        sequences, args.peers.len())?;
    let mut engine = super::engine::MuseEngine::connect(&library, &cfg, &ranks, streams, args.peers.clone(),
        super::engine::LinkPlan { spark, cutoff, unit_rows: usize::try_from(prefill.unit_rows)?,
            waves: usize::try_from(prefill.waves)?, intake_row_bytes }, !local.is_empty())?;
    tracing::info!(ranks = engine.ranks.len(), peer = engine.peer.is_some(), vocab = embedding.vocab(),
        placement = %placement.summary(), pool_tokens = placement.pool_tokens, "Muse Spark brought up");
    // The admitted context: the pool clamps an automatic one.
    let served_context = usize::try_from(context.min(placement.pool_tokens))?;
    let pages = usize::try_from(placement.pool_tokens)? / super::step::PAGE_ROWS;
    let streams = engine.streams.iter().map(|stream| stream.raw).collect::<Vec<_>>();
    // The drafter beside the target head, on the head rank's stream (admission charged it there).
    let head_stream = *streams.last().context("Muse head stream")?;
    let draft = match (args.draft.as_deref(), &built.drafter) {
        (Some(snapshot), Some(admitted)) => Some(super::draft::MuseDraft::load(&cfg, ranks.last().context("Muse head rank")?,
            head_stream, snapshot, &embedding, admitted, args.draft_kv_cache)?),
        (None, None) => None,
        _ => anyhow::bail!("Muse drafter admission differs from --draft"),
    };
    let pricing = draft.as_ref().map(|_| super::backend::DraftPricing::new(&cfg,
        catalog.exl3().context("Muse draft pricing needs EXL3 headers")?, &onboarded, gpus, engine.link.world_size()?,
        engine.link.installed_grouping()?)).transpose()?;
    tracing::info!(drafter = ?args.draft, onboarded = onboarded.len(), "Muse Spark drafter");
    let backend = super::backend::MuseBackend::new(&cfg, &ranks, streams, &embedding, &mut engine.link,
        engine.peer.as_ref(), usize::try_from(prefill.unit_rows)?, usize::try_from(prefill.waves)?, sequences,
        served_context, pages, local, draft, pricing, super::media::Binding::from_snapshot(&args.snapshot, cfg.vocab_size)?)?;
    run(&backend, sequences, media)
}

fn worker(args: &ServeArgs, receive: &mut tokio::sync::mpsc::Receiver<cuteafd_api::openai::NativeRequest>,
    stats: &Arc<std::sync::Mutex<serde_json::Value>>, eos: Vec<u32>,
    initialized: impl FnOnce(super::media::ApiReady) -> Result<()>) -> Result<()> {
    use crate::shared::qwen::scheduler::{self, Backend as _};
    with_backend(args, |backend, sequences, ready| {
        let super::media::Ready { encoder, image, audio, cache_bytes, prefix } = ready;
        let api = super::media::ApiReady { context: backend.geometry().max_context, image, audio,
            image_health: encoder.health_handle_for(false), audio_health: encoder.health_handle_for(true) };
        let mut media = cuteafd_engine::media::MediaAdmission::new(cuteafd_engine::media::EmbeddingCache::new(cache_bytes),
            encoder, 16);
        let drafts = match (&args.draft, args.draft_fixed) {
            (None, _) => scheduler::Drafts::None,
            (Some(_), Some(depth)) => scheduler::Drafts::Neural { depth: usize::from(depth), fixed: true },
            (Some(snapshot), None) => scheduler::Drafts::Neural {
                depth: crate::shared::speculation::dflash::DflashConfig::read(snapshot)?.drafts(), fixed: false },
        };
        scheduler::schedule(backend, &args.snapshot, receive, stats, sequences, drafts, eos,
            args.decode_share, &prefix, args.token_io.token_select, &mut media, None,
            || initialized(api))
    })
}

pub(crate) async fn run(args: ServeArgs) -> Result<()> {
    use cuteafd_api::openai::{chat::muse::MuseEncoding, ModelEncoding, ModelProfile, NativeLimits};
    args.validate()?;
    let api = args.api.load()?;
    let snapshot = args.snapshot.clone();
    let encoding = Arc::new(MuseEncoding::from_snapshot(&snapshot)?);
    let eos = encoding.tokens().eos.clone();
    let model = args.model_id.clone().or_else(|| snapshot.ancestors().find_map(|path| {
        let name = path.file_name()?.to_str()?.strip_prefix("models--")?;
        let (organization, model) = name.split_once("--")?;
        Some(format!("{organization}/{model}"))
    })).context("Muse Spark requires --model-id for a snapshot without HF identity")?;
    let profile = ModelProfile::new(model, ModelEncoding::Muse(encoding));
    let (queue, mut receive) = tokio::sync::mpsc::channel(16);
    let stats = Arc::new(std::sync::Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let engine_args = args.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<super::media::ApiReady>>();
    let handle = tokio::task::spawn_blocking(move || {
        let mut ready = Some(ready_tx);
        let result = worker(&engine_args, &mut receive, &worker_stats, eos, |context| {
            if let Some(ready) = ready.take() { let _ = ready.send(Ok(context)); }
            Ok(())
        });
        if let Some(ready) = ready.take() {
            let _ = ready.send(Err(anyhow::anyhow!("Muse Spark initialization failed: {:#}",
                result.as_ref().err().context("worker exited before readiness")?)));
        }
        result
    });
    let ready = ready_rx.await.context("Muse Spark worker exited before readiness")??;
    let limits = NativeLimits::new(u32::try_from(ready.context)?, args.max_output)?;
    let profile = ready.install(profile)?;
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new(
        "muse_spark", profile.id.clone(), snapshot.clone())));
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, std::time::Duration::from_secs(25),
        hub.clone(), api.serve(profile, &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(handle) => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    const PEERS: &str = "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4,127.0.0.1:5,127.0.0.1:6";

    fn args(extra: &[&str]) -> Result<ServeArgs, clap::Error> {
        let mut argv = vec!["cuteafd", "serve-muse", "--snapshot", "/checkpoint", "--native-lib", "/native.so"];
        argv.extend_from_slice(extra);
        let cli = crate::cli::Cli::try_parse_from(argv)?;
        let crate::cli::Commands::ServeMuse(args) = cli.command else { panic!("Muse command") };
        Ok(args)
    }

    #[test]
    fn defaults_are_one_entry_gpu_with_automatic_context_and_onboarding() -> Result<()> {
        let defaults = args(&["--peers", PEERS])?;
        defaults.validate()?;
        assert_eq!(defaults.devices(), [0]);
        assert_eq!(args(&["--peers", PEERS, "--split-device", "1"])?.devices(), [0, 1]);
        assert_eq!(defaults.max_context, None);
        assert_eq!(defaults.pool_tokens, None);
        assert_eq!(defaults.rtx_expert_layers, cuteafd_loader::placement::Onboard::Auto);
        assert_eq!(defaults.max_sequences, 16);
        assert_eq!(defaults.vision, cuteafd_loader::plan::MediaMode::Auto);
        assert_eq!(defaults.audio, cuteafd_loader::plan::MediaMode::Auto);
        assert!(defaults.vision_peers.is_empty() && defaults.audio_peers.is_empty());
        assert_eq!(defaults.spark_budget_bytes(), cuteafd_loader::plan::DEFAULT_SPARK_BUDGET_BYTES);
        assert_eq!(defaults.expert_partial_bf16_max_rows, 64);
        assert_eq!(defaults.expert_partial_wide_format, cuteafd_loader::families::muse_spark::replies::WideFormat::Nvfp4);
        assert!(args(&["--peers", PEERS, "--expert-partial-wide-format", "fp8"]).is_ok());
        assert!(args(&["--peers", PEERS, "--expert-partial-wide-format", "bf16"]).is_err());
        args(&["--peers", PEERS, "--expert-partial-bf16-max-rows", "4096"])?.validate()?;
        let single = args(&["--peers", PEERS, "--device", "1"])?;
        single.validate()?;
        assert_eq!(single.devices(), [1]);
        for peers in ["127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4", PEERS] {
            args(&["--peers", peers])?.validate()?;
        }
        Ok(())
    }

    /// Bring-up order: admission seals the Spark endpoint before the coordinator
    /// loads (onboarded experts inside the same window), the link connects
    /// after the load, and the backend serves the shared scheduler with no
    /// drafter, admitted media owners, and the readiness callback last.
    #[test]
    fn serve_brings_up_then_schedules_the_backend() {
        let source = include_str!("serve.rs").split("#[cfg(test)]").next().unwrap();
        let body = source.split("fn with_backend<T>").nth(1).unwrap().split("\nfn worker(").next().unwrap();
        let order = ["startup::admit(", "super::media::Ready::load(", "SparkAdmission::plan(", "load_while_admitting(Some(admission)",
            "super::onboard::load(", "MuseEngine::connect(", "MuseBackend::new(", "run(&backend, sequences, media)"];
        let at: Vec<usize> = order.iter().map(|needle| body.find(needle).unwrap_or_else(|| panic!("{needle}"))).collect();
        assert!(at.windows(2).all(|pair| pair[0] < pair[1]), "{order:?}");
        assert!(!body.contains("bail!(\"Muse Spark step executor"), "serving no longer stops after bring-up");
        let worker = source.split("\nfn worker(").nth(1).unwrap().split("\npub(crate) async fn run(").next().unwrap();
        assert!(worker.contains("scheduler::Drafts::None") && worker.contains("EmbeddingCache::new(cache_bytes)"));
        assert!(worker.contains("encoder.health_handle_for(false)") && worker.contains("encoder.health_handle_for(true)"));
        assert!(worker.contains("args.decode_share, &prefix"));
        // `--draft` drives neural drafts (the policy's depth, or the fixed one); without it, none.
        assert!(worker.contains("(Some(_), Some(depth)) => scheduler::Drafts::Neural { depth: usize::from(depth), fixed: true }"));
        assert!(worker.contains("DflashConfig::read(snapshot)?.drafts(), fixed: false"));
        let backend = body.split("MuseBackend::new(").nth(1).unwrap();
        assert!(backend.contains("served_context, pages, local, draft, pricing"));
        let run = source.split("\npub(crate) async fn run(").nth(1).unwrap();
        assert!(run.contains("MuseEncoding::from_snapshot(") && run.contains("ModelEncoding::Muse("));
        assert!(run.find("ready_rx.await").unwrap() < run.find("TcpListener::bind").unwrap());
    }

    #[test]
    fn media_peer_identity_is_validated_before_device_access() -> Result<()> {
        let hash = "ab".repeat(32);
        for tower in ["vision", "audio"] {
            let peers = if tower == "vision" { "--vision-peers" } else { "--audio-peers" };
            let hash_arg = if tower == "vision" { "--encoder-plan-hash" } else { "--audio-encoder-plan-hash" };
            let revision = if tower == "vision" { "--encoder-revision" } else { "--audio-encoder-revision" };
            let mode = if tower == "vision" { "--vision" } else { "--audio" };
            let base = ["--peers", PEERS, peers, "127.0.0.1:19442"];
            let complete = [base.as_slice(), &[hash_arg, hash.as_str(), revision, "checkpoint-revision"]].concat();
            args(&complete)?.validate()?;
            args(&[complete.as_slice(), &[mode, "spark:0"]].concat())?.validate()?;
            for invalid in [base.to_vec(),
                [base.as_slice(), &[hash_arg, hash.as_str()]].concat(),
                [base.as_slice(), &[hash_arg, "bad-hash", revision, "revision"]].concat(),
                [base.as_slice(), &[hash_arg, hash.as_str(), revision, ""]].concat(),
                [complete.as_slice(), &[mode, "off"]].concat(),
                [complete.as_slice(), &[mode, "rtx:0"]].concat(),
                vec!["--peers", PEERS, mode, "spark:0"]] {
                assert!(args(&invalid)?.validate().is_err(), "{tower}: {invalid:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn prefill_shape_is_the_planners() -> Result<()> {
        let shape = args(&["--peers", PEERS])?.prefill()?;
        let planned = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(6,
            cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::muse_spark::MUSE_SPARK),
            Default::default())?;
        assert_eq!(shape, planned);
        assert_eq!(args(&["--peers", PEERS, "--prefill-waves", "2"])?.prefill()?.waves, 2);
        Ok(())
    }

    #[test]
    fn unsupported_layouts_refuse_before_any_device_access() -> Result<()> {
        assert!(args(&[])?.validate().is_err(), "experts need Spark ranks");
        let nine = (1..=9).map(|n| format!("127.0.0.1:{n}")).collect::<Vec<_>>().join(",");
        // Worlds 5, 7 and 8 come from the shared range data (#941).
        for world in [5, 7, 8] {
            let peers = (1..=world).map(|n| format!("127.0.0.1:{n}")).collect::<Vec<_>>().join(",");
            args(&["--peers", peers.as_str()])?.validate()?;
        }
        for extra in [vec!["--peers", nine.as_str()], vec!["--peers", "127.0.0.1:1,127.0.0.1:1"],
            vec!["--peers", PEERS, "--device", "1", "--split-device", "1"], vec!["--peers", PEERS, "--split-device", "0"],
            vec!["--peers", PEERS, "--max-context", "1048577"], vec!["--peers", PEERS, "--max-context", "0"],
            vec!["--peers", PEERS, "--spark-budget-gib", "0"], vec!["--peers", PEERS, "--spark-budget-gib", "NaN"],
            vec!["--peers", PEERS, "--prefill-rows", "4097"], vec!["--peers", PEERS, "--prefix-partial", "on"],
            vec!["--peers", PEERS, "--expert-partial-bf16-max-rows", "8"]] {
            assert!(args(&extra)?.validate().is_err(), "{extra:?}");
        }
        assert!(args(&["--peers", PEERS, "--max-sequences", "65"]).is_err());
        Ok(())
    }
}
