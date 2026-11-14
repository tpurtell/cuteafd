//! Whole-rank startup on the shared admission, push, cache and expert owners.
//! The API/readiness edge is attached only after serving qualification.
use super::{admission::serve_admission::{self, Sample}, cache::RequestStreams,
    engine::{RoutedBinding, ServingPrefill}, experts::Dispatcher, layer::Kernels,
    owners::{LoadedRank, Owners}, weights::push};
use crate::shared::{experts::{fp8, rtx::{push::PushedExperts, whole::Whole}},
    inventory::{InventoryPhase, RuntimeInventory}, memory::device::{Device, Stream},
    placement::{spark_rings, SparkAdmission}, prefill_pipeline::FfnRoute,
    prefix::{budget::HostBudget, PrefixArgs}, spark_intake::SparkLink};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{expert_usage::Reader, ExpertGeometry, SparkEndpoint, SparkIngress};
use cuteafd_engine::prefix::PrefixFamily;
use cuteafd_ffi::programs::Programs;
use cuteafd_loader::{families::minimax_m3::{admission::{Coordinator, FactsRequest},
    execution::{ForwardPrograms, MsaPrograms, ProducerPrograms}, weight_layout::CoordinatorWeights,
    MiniMaxM3Config}, placement::{families::minimax_m3::{AdmittedMiniMax, LocalPackage, MiniMaxInputs,
    RequestCapacity}, planner::PrefillShape, Onboard, ResidentWeights, SparkWorkingSet},
    plan::{checkpoint::CheckpointView, Checkpoint}};
use cuteafd_transport::{expert::{SparkWorld, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16},
    worker_selection::ModelAdmission, ExpertV2Dtype, SparkSessionShape, TcpTransportConfig};
use std::{collections::BTreeSet, net::SocketAddr, path::Path, rc::Rc, sync::Arc};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)] pub(crate) attention: crate::cli::CoreViewArgs,
    #[arg(long)] pub snapshot: std::path::PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] pub native_lib: std::path::PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] pub manifest: std::path::PathBuf,
    #[arg(long, default_value_t = 0)] pub device: i32,
    #[arg(long, value_delimiter = ',')] pub peers: Vec<SocketAddr>,
    #[arg(long)] pub pool_tokens: Option<u64>,
    #[arg(long = "max-context-tokens", value_parser = crate::shared::context::parse_tokens::<usize>)]
    pub max_context: Option<usize>,
    #[arg(long, default_value = "auto")] pub rtx_expert_layers: cuteafd_loader::placement::OnboardSetting,
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u8).range(1..=64))]
    pub max_sequences: u8,
    #[arg(long, default_value = "0.0.0.0:8000")] pub listen: String,
    #[arg(long)] pub model_id: Option<String>,
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
    ensure!(matches!(args.token_io.embed_placement, crate::shared::token_io::EmbedPlacement::Gpu)
        && args.token_io.token_select == crate::shared::token_io::SelectPlacement::Device,
        "MiniMax serving requires admitted GPU embedding and device selection");
    cuteafd_bench::context::set(cuteafd_bench::context::ServerContext {
        command: "serve-minimax".into(), family: Some("minimax_m3".into()), snapshot: Some(args.snapshot.clone()),
        supports_thinking_off: Some(false), ..Default::default()
    });
    cuteafd_bench::context::ResolvedSettings::default().value("max-sequences", args.max_sequences)
        .value("prefill-packing", format!("{:?}", args.prefill_packing).to_lowercase())
        .value("draft-mode", "no speculation").value("expert-usage", args.expert_usage).publish();
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
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<usize>>();
    let worker_args = args.clone();
    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| -> Result<()> {
            let view = worker_args.attention.selected_view.clone().map(Ok)
                .unwrap_or_else(|| worker_args.attention.select(&worker_args.snapshot))?;
            worker_args.attention.validate_view(&view)?;
            let checkpoint = Checkpoint::coordinator_view(&view, false, false)?;
            let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
            cuteafd_bench::context::set_family("minimax_m3");
            // SAFETY: all native owners and the scheduler retire inside this worker
            // before the library and installed program manifest are dropped.
            let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&worker_args.native_lib)? });
            let programs = library.programs()?.with_manifest(&worker_args.manifest)?;
            let concurrency = usize::from(worker_args.max_sequences);
            let loaded = load(&library, &programs, &view, &checkpoint, &cfg, Options {
                device: worker_args.device, peers: &worker_args.peers, native_lib: &worker_args.native_lib,
                manifest_path: &worker_args.manifest,
                prefill: PrefillShape { rows: prefill.unit_rows, waves: prefill.waves },
                sequences: concurrency as u64, context_tokens: worker_args.max_context.unwrap_or(0) as u64,
                pool_tokens: worker_args.pool_tokens.filter(|&tokens| tokens > 0),
                onboard: worker_args.rtx_expert_layers.resolve(cfg.moe.iter().filter(|&&moe| moe).count()),
                prefix: &worker_args.prefix, expert_usage: worker_args.expert_usage,
                usage_reader: worker_args.expert_usage_reader.clone(),
                resident_weights: crate::shared::memory::budget::resident_weight_limit() })?;
            let served_context = usize::try_from(loaded.admitted.solved.served_context())?;
            let backend = super::backend::MiniMaxBackend::new(&loaded.owners, &cfg,
                loaded.execution, served_context, loaded.host_prefix)?;
            library.cuda_set_device(worker_args.device)?;
            scheduler::schedule(&backend, &worker_args.snapshot, &mut receive, &worker_stats,
                concurrency, scheduler::Drafts::None, Default::default(), eos, worker_args.decode_share,
                &worker_args.prefix, worker_args.loop_abort.config(), worker_args.token_io.token_select,
                &mut super::super::qwen3_5::serve::text_media(), None, || {
                    backend.require_live()?;
                    RuntimeInventory::measure(&library, &[worker_args.device], InventoryPhase::PostLoad,
                        |_| Ok(()))?.record("minimax_m3-serving-loaded", &loaded.admission_free);
                    if let Some(ready) = ready.take() { let _ = ready.send(Ok(served_context)); }
                    Ok(())
                })
        })();
        crate::shared::serve::forward_initialization(result, ready.is_some(), "MiniMax initialization failed", |error| {
            ready.take().expect("pending readiness").send(Err(error)).map_err(|result| match result {
                Err(error) => error, Ok(_) => unreachable!("sent an error") })
        })
    });
    let context = ready_rx.await.context("MiniMax worker exited before readiness")??;
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

pub(crate) struct Options<'a> {
    pub device: i32,
    pub peers: &'a [SocketAddr],
    pub native_lib: &'a Path,
    pub manifest_path: &'a Path,
    pub prefill: PrefillShape,
    pub sequences: u64,
    pub context_tokens: u64,
    pub pool_tokens: Option<u64>,
    pub onboard: Onboard,
    pub prefix: &'a PrefixArgs,
    pub expert_usage: bool,
    pub usage_reader: Reader,
    pub resident_weights: ResidentWeights,
}

/// The executor drops first: it terminally drains links before storage retires.
/// Readiness remains the shared scheduler's responsibility, after its selector
/// and prefix cache have been constructed and reconciled.
pub(crate) struct Loaded<'a> {
    pub execution: ServingPrefill<'a>,
    pub owners: Owners<'a>,
    pub inputs: MiniMaxInputs,
    pub admitted: AdmittedMiniMax,
    pub admission_free: Vec<u64>,
    pub host_prefix: Option<cuteafd_hostcache::config::Config>,
}

fn fix_geometry() -> Result<()> {
    cuteafd_core::set_expert_geometry(ExpertGeometry::MINIMAX_M3)
        .map_err(|previous| anyhow::anyhow!("MiniMax serving needs an isolated process: expert geometry already fixed to {previous:?}"))
}

fn validate(options: &Options<'_>, cfg: &MiniMaxM3Config) -> Result<()> {
    ensure!(options.device >= 0, "MiniMax serving requires a nonnegative CUDA device");
    cuteafd_core::concurrency::validate("minimax_m3", options.sequences)?;
    serve_admission::validate_prefill(options.prefill)?;
    ensure!(options.context_tokens == 0 || options.context_tokens <= cfg.max_position as u64,
        "MiniMax served context exceeds checkpoint extent");
    ensure!(options.peers.iter().copied().collect::<BTreeSet<_>>().len() == options.peers.len(),
        "MiniMax Spark endpoints must be distinct");
    validate_spark_world(options.peers.len(), cfg.moe_intermediate)?;
    ensure!(options.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
        "MiniMax serving requires exact prefix restores");
    Ok(())
}

/// Do not replace the planner's minimum with a private four-Spark golden pin.
/// Every installed native recipe in N_min..N_max uses the same startup path.
fn validate_spark_world(world: usize, intermediate: usize) -> Result<()> {
    ensure!(world <= cuteafd_core::MAX_SPARK_WORLD, "MiniMax Spark world exceeds eight ranks");
    if world > 0 {
        cuteafd_loader::plan::experts::spark_recipe("minimax:nvfp4a4", intermediate, world)
            .context("MiniMax serving Spark count has no native NVFP4 package recipe")?;
    }
    Ok(())
}

fn program_names(cfg: &MiniMaxM3Config, target: &CoordinatorWeights,
    manifest: &serde_json::Value, capacity: RequestCapacity, context: u32) -> Result<BTreeSet<String>> {
    let (_, msa) = MsaPrograms::resolve(cfg, manifest, capacity.decode.rows, capacity.prefill.rows, context)?;
    let (_, producers) = ProducerPrograms::resolve(manifest, capacity.execution_rows())?;
    let (_, forward) = ForwardPrograms::resolve(manifest, capacity.execution_rows())?;
    let dense = target.resolve_dense_programs(manifest, capacity.execution_rows())?;
    Ok(msa.iter().chain(&producers).chain(&forward).chain(&dense)
        .map(|program| program.name.clone()).collect())
}

pub(crate) fn load<'a>(library: &'a Arc<cuteafd_ffi::NativeLibrary>, programs: &'a Programs<'a>,
    view: &CheckpointView, checkpoint: &Checkpoint, cfg: &MiniMaxM3Config,
    options: Options<'_>) -> Result<Loaded<'a>> {
    validate(&options, cfg)?;
    cfg.release_geometry()?;
    fix_geometry()?;
    ensure!(checkpoint.snapshot == view.snapshot(), "MiniMax startup checkpoint view differs");
    let manifest = Arc::new(serde_json::from_slice(&std::fs::read(options.manifest_path)?)?);
    let catalog = cuteafd_loader::read_expert_catalog_view(view)?;
    let targets = CoordinatorWeights::plan(checkpoint, cfg, 1, &vec![0; cfg.layers])?;
    let capacity = RequestCapacity::new(options.sequences, options.prefill.rows)?;
    let context = if options.context_tokens == 0 { cfg.max_position as u64 } else { options.context_tokens };
    let names = program_names(cfg, &targets[0], &manifest, capacity, u32::try_from(context)?)?;
    let device = Device { library, id: options.device };
    let package = fp8::package_directory(options.native_lib, 1,
        catalog.fp8().context("MiniMax requires native NVFP4 experts")?.format());
    let routed = cfg.moe.iter().enumerate().filter_map(|(layer, moe)| moe.then_some(layer)).collect::<Vec<_>>();
    let may_onboard = options.onboard.layers(routed.len()) != Some(0);
    let local = cuteafd_loader::families::minimax_m3::admission::local_package(
        &catalog, options.manifest_path, capacity, 1);
    let mut native = None;
    let mut expert_modules = None;
    let runtime = RuntimeInventory::measure(library, &[options.device], InventoryPhase::PreWeight, |_| {
        programs.load_matching(|name| names.contains(name))?;
        native = Some(Kernels::bind(device, programs, &manifest, cfg, &targets[0],
            capacity.execution_rows(), u32::try_from(context)?)?);
        if may_onboard && matches!(local, LocalPackage::Installed { .. }) {
            expert_modules = Some(fp8::Fp8Modules::load(&package, None)?);
        }
        if !options.peers.is_empty() {
            library.v41_compact_reducer()?.require_plane_reduction(options.peers.len() as u32)?;
        }
        Ok(())
    })?;
    let policy = cuteafd_core::memory_policy::MemoryPolicy::current();
    let samples = runtime.gpus.iter().map(|sample| Sample { total: sample.total_bytes,
        cap: policy.gpu_cap_bytes(sample.total_bytes), free: sample.free_bytes, pending_code: 0 }).collect::<Vec<_>>();
    let usage_enabled = options.usage_reader.configure(options.expert_usage, false).map_err(anyhow::Error::msg)?;
    let physical = [options.device];
    let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
    let host_cache_bytes = match options.prefix.host_cache_bytes {
        HostBudget::Auto => None, HostBudget::Bytes(bytes) => Some(bytes),
    };
    let replies = cuteafd_core::spark_reply::requested();
    fp8::validate_reply_formats(catalog.fp8().context("MiniMax expert catalog")?.format(), replies)?;
    let (inputs, admitted) = serve_admission::prepare(FactsRequest { checkpoint, cfg,
        catalog: &catalog, manifest_path: options.manifest_path, manifest: Arc::clone(&manifest),
        coordinator: Coordinator::Measured { samples: vec![], loaded_code: vec![] },
        attention: None, split_wire: None, physical_devices: &physical,
        spark_ranks: options.peers.len(), spark_bytes: policy.spark_cap_bytes(100 << 30),
        prefill: options.prefill, context_tokens: options.context_tokens, pool_tokens: options.pool_tokens,
        sequences: options.sequences, host, host_cache_bytes,
        prefix_entries: options.prefix.prefix_cache_entries as u64, replies,
        head: crate::shared::target_head::TargetHead::configured().blas_head(true),
        expert_usage: usage_enabled, resident_weight_limit: options.resident_weights }, &samples, options.onboard)?;
    cuteafd_bench::context::ResolvedSettings::placement(&admitted.placement).publish();
    cuteafd_bench::context::set_resolved("max-context-tokens", &admitted.solved.served_context().to_string());
    let config = TcpTransportConfig::default();
    let endpoint = SparkEndpoint { ring_depth: None, capacity: capacity.prefill.rows,
        ingress: SparkIngress::Bf16, partial_reply_bytes: None };
    let working = SparkWorkingSet { rtx_layers: inputs.routed[..admitted.onboard].to_vec(),
        spark_layers: inputs.routed[admitted.onboard..].to_vec(),
        spark_ranks: options.peers.len(), prefill_lanes: 1 };
    let spark = if options.peers.is_empty() { None } else {
        let model = ModelAdmission::for_clients(ExpertGeometry::MINIMAX_M3, endpoint,
            spark_rings(config.max_frame_bytes)?, 1,
            cuteafd_loader::placement::families::minimax_m3::spark_executions(options.prefill)?)?;
        Some(SparkAdmission::plan_view(view, ExpertGeometry::MINIMAX_M3, Some(&catalog),
            &working, options.peers.len(), model, None)?)
    };
    let (pending, membership) = push::prepare_serving(view, &catalog, checkpoint, &[device], &inputs, &admitted)?;
    let spark = spark.map(|spark| spark.start_with(options.peers, membership)).transpose()?;
    let push::ServingWeights { targets, embedding, mut expert_sets } = pending.finish(&catalog, spark)?;
    let stream = Rc::new(Stream::new(device)?);
    let loaded = targets.into_iter().map(|weights| LoadedRank { weights,
        streams: RequestStreams { decode: Rc::clone(&stream),
            prefill: (0..options.prefill.waves).map(|_| Rc::clone(&stream)).collect() } }).collect();
    let owners = Owners::with_usage_reader(&inputs, &admitted, programs, loaded, embedding,
        options.usage_reader.clone())?;
    let mut dispatchers = Vec::new();
    let mut bindings = vec![None; cfg.layers];
    if admitted.onboard > 0 {
        ensure!(expert_sets.len() == 1, "MiniMax selected expert set missing");
        let mut set = expert_sets.pop().context("MiniMax expert set missing")?;
        ensure!(set.ranks.len() == 1 && set.backings == admitted.expert_backings,
            "MiniMax whole expert set differs from admission");
        let PushedExperts::Fp8(pushed) = set.ranks.pop().context("MiniMax pushed expert rank missing")? else {
            anyhow::bail!("MiniMax expert set is not native NVFP4");
        };
        let LocalPackage::Installed { scratch, output } = inputs.local else {
            anyhow::bail!("MiniMax admitted local experts without installed workspace");
        };
        let backing = &admitted.expert_backings[0];
        let budget = backing.arena.bytes().checked_add(usize::try_from(scratch)?)
            .and_then(|bytes| bytes.checked_add(usize::try_from(output).ok()?))
            .context("MiniMax expert owner budget overflow")?;
        let experts = Whole::from_pushed(device, catalog.fp8().context("MiniMax native experts")?,
            &package, working.rtx_layers.clone(), capacity.execution_rows() as usize, budget, pushed)?;
        for &layer in &working.rtx_layers {
            bindings[layer] = Some(RoutedBinding { dispatcher: dispatchers.len(), route: FfnRoute::Local });
        }
        dispatchers.push(Dispatcher::whole(Box::new(experts))?);
    } else { ensure!(expert_sets.is_empty(), "MiniMax received unselected expert sets"); }
    if !working.spark_layers.is_empty() {
        let layout = inputs.links.first().context("MiniMax admitted Spark link missing")?.clone();
        let mut link = device.run(|| SparkLink::prepared(library, SparkWorld::canonical(options.peers.len())?,
            options.peers, config, layout))?;
        link.enable_terminal_ownership(Arc::clone(library))?;
        let mut shape = SparkSessionShape::new(capacity.prefill.rows, 6144, 4, ExpertV2Dtype::Bf16)?;
        shape.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        device.run(|| link.bootstrap(&shape))?;
        for &layer in &working.spark_layers {
            bindings[layer] = Some(RoutedBinding { dispatcher: dispatchers.len(), route: FfnRoute::Spark });
        }
        dispatchers.push(Dispatcher::spark(link)?);
    }
    let execution = ServingPrefill::new(cfg, dispatchers, bindings, 1)?;
    owners.prewarm(cfg)?;
    let quota = inputs.host_prefix.quota_bytes;
    let host_prefix = (quota > 0).then(|| options.prefix.host_config_for(owners.storage.prefix.layout(),
        admitted.solved.served_context() as usize, quota)).transpose()?;
    // These module pins cover the measured baseline until final owner binding.
    drop(expert_modules);
    drop(native);
    crate::shared::memory_report::release_load_staging(library)?;
    publish_provenance(options.native_lib, options.manifest_path, &runtime, &admitted,
        options.peers.len())?;
    Ok(Loaded { execution, owners, inputs, admitted,
        admission_free: samples.iter().map(|sample| sample.free).collect(), host_prefix })
}

fn publish_provenance(native_lib: &Path, manifest: &Path, runtime: &RuntimeInventory,
    admitted: &AdmittedMiniMax, spark_world: usize) -> Result<()> {
    use sha2::{Digest, Sha256};
    let hash = |path: &Path| -> Result<String> {
        let mut file = std::fs::File::open(path)?;
        let mut digest = Sha256::new();
        std::io::copy(&mut file, &mut digest)?;
        Ok(format!("{:x}", digest.finalize()))
    };
    cuteafd_ffi::memory_ledger::fault::set_inputs(serde_json::json!({
        "memory_reconciliation_provenance": {
            "schema_version": 1, "family": "minimax_m3",
            "source_commit": env!("CUTEAFD_BUILD_COMMIT"),
            "source_dirty": env!("CUTEAFD_BUILD_DIRTY"), "toolchain": env!("CUTEAFD_BUILD_TOOLCHAIN"),
            "native_sha256": hash(native_lib)?, "programs_sha256": hash(manifest)?,
            "layout": format!("1 RTX/{spark_world} Sparks; whole; context/{}; slots/{}",
                admitted.solved.served_context(), admitted.solved.decisions.prefill.rows),
            "devices": runtime.gpus, "spark_world": spark_world,
            "onboard_layers": admitted.onboard, "graph_basis": "uncalibrated MiniMax byte fallback"
        }
    }));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_serve_cli_and_runtime_settings_have_a_cpu_seam() -> Result<()> {
        use clap::Parser;
        let argv = ["cuteafd", "serve-minimax", "--snapshot", "/fixture", "--native-lib", "/native",
            "--max-sequences", "16", "--prefill-rows", "2752", "--prefill-waves", "3"];
        let crate::cli::Commands::ServeMinimax(mut args) = crate::cli::Cli::try_parse_from(argv)?.command else {
            panic!("MiniMax serve command");
        };
        crate::commands::bench::assert_resolved_family_seam("serve-minimax", &[], || {
            args.prefix.resolve_entries(16);
            args.prefill()?;
            cuteafd_bench::context::ResolvedSettings::default().value("max-context-tokens", 131072)
                .value("pool-tokens", 2097152).value("rtx-expert-layers", 0).publish();
            Ok(())
        })
    }

    #[test]
    fn process_geometry_refuses_a_conflicting_family_without_replacement() -> Result<()> {
        const CHILD: &str = "CUTEAFD_MINIMAX_GEOMETRY_CHILD";
        if let Ok(order) = std::env::var(CHILD) {
            if order == "prior" {
                cuteafd_core::set_expert_geometry(ExpertGeometry::KIMI_K3).unwrap();
                let error = fix_geometry().unwrap_err();
                assert!(error.to_string().contains("isolated process: expert geometry already fixed"));
                assert_eq!(cuteafd_core::expert_geometry(), ExpertGeometry::KIMI_K3);
            } else {
                fix_geometry()?;
                assert_eq!(cuteafd_core::set_expert_geometry(ExpertGeometry::KIMI_K3), Err(ExpertGeometry::MINIMAX_M3));
                assert_eq!(cuteafd_core::expert_geometry(), ExpertGeometry::MINIMAX_M3);
            }
            return Ok(());
        }
        for order in ["prior", "after"] {
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "families::minimax_m3::startup::tests::process_geometry_refuses_a_conflicting_family_without_replacement", "--nocapture"])
                .env(CHILD, order).status()?;
            ensure!(status.success(), "MiniMax isolated geometry child failed: {order}");
        }
        let source = include_str!("startup.rs");
        let constructor = source.split("pub(crate) fn load<'a>").nth(1).unwrap().split("#[cfg(test)]").next().unwrap();
        assert_eq!(constructor.matches("fix_geometry()?").count(), 1);
        assert!(constructor.find("fix_geometry()?").unwrap() < constructor.find("fp8::package_directory(").unwrap());
        Ok(())
    }

    #[test]
    fn native_spark_range_is_not_the_golden_four_rank_pin() -> Result<()> {
        validate_spark_world(0, 3072)?;
        for world in 2..=8 { validate_spark_world(world, 3072)?; }
        for world in [1, 9] {
            let error = validate_spark_world(world, 3072).unwrap_err();
            assert!(error.to_string().contains(if world == 1 { "native NVFP4 package recipe" } else { "eight ranks" }));
        }
        Ok(())
    }

    #[test]
    fn constructor_orders_admission_before_push_and_releases_staging() {
        let source = include_str!("startup.rs");
        let constructor = source.split("pub(crate) fn load<'a>").nth(1).unwrap().split("#[cfg(test)]").next().unwrap();
        let admission = constructor.find("serve_admission::prepare(").unwrap();
        let push = constructor.find("push::prepare_serving(").unwrap();
        let owners = constructor.find("Owners::with_usage_reader(").unwrap();
        let release = constructor.find("release_load_staging(library)?").unwrap();
        assert!(admission < push && push < owners && owners < release);
        assert!(!constructor.contains("ready()"));
    }
}
