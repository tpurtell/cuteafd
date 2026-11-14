//! Cold full-model qualification only. No serving entrypoint or reference bridge.
use super::{admission, cache::Cache, head::Head, layer::{Kernels, UnitOwner}, prefill::{LayerTaps, Prefill},
    weights::TargetWeights};
use crate::shared::{inventory::{InventoryPhase, RuntimeInventory},
    memory::{device::{Allocation, Device, Stream}, receipts::Inventory},
    placement::{spark_rings, SparkAdmission}, prefill_pipeline::sequence_units,
    spark_intake::SparkLink};
use anyhow::{ensure, Context, Result};
use cuteafd_bench::{fidelity_dataset, fidelity_rows, reference::{Fidelity, Window}};
use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress, SparkReplyFormat,
    memory_layout::{Category, Tag}};
use cuteafd_loader::{families::minimax_m3::{MiniMaxM3Config, execution::*, weight_layout::CoordinatorWeights},
    placement::{PreparedSparkLink, SparkEndpointOwners, SparkIntakeCandidates, SparkWorkingSet,
        planner::{CoordinatorDevices, Lifetime}}, plan::{Checkpoint, checkpoint::CheckpointView}};
use cuteafd_transport::{ExpertV2Dtype, SparkSessionShape, TcpTransportConfig,
    expert::{SparkWorld, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16}, worker_selection::ModelAdmission};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, net::SocketAddr, path::{Path, PathBuf}, rc::Rc, sync::Arc};

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)] pub(crate) attention: crate::cli::CoreViewArgs,
    #[arg(long)] pub(crate) snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] manifest: PathBuf,
    /// Immutable MiniMax top1024-plus-tail reference publication.
    #[arg(long)] golden: PathBuf,
    /// New directory for shared scoring probes and fidelity.json.
    #[arg(long)] save_logits: PathBuf,
    /// Also write scored raw FP32 rows to WINDOW.raw-logits.bin (not log-probs).
    #[arg(long)] raw_logits: bool,
    #[arg(long)] window: Option<String>,
    /// Diagnostic residual/branch taps beneath save-logits; requires selected windows.
    #[arg(long, value_delimiter = ',', num_args = 1..)] tap_layers: Vec<usize>,
    /// Keep only layer residuals, plus final normalized hidden and scored logits.
    #[arg(long, requires = "tap_layers")] tap_residual_only: bool,
    #[arg(long, value_delimiter = ',', num_args = 1..)] peers: Vec<SocketAddr>,
    #[arg(long, default_value_t = 0)] device: i32,
    /// Optional distinct physical peer; absent preserves the one-GPU cold golden.
    #[arg(long, requires = "wire")] split_device: Option<i32>,
    #[arg(long, env = "CUTEAFD_ATTENTION_PLACEMENT", default_value = "auto",
        value_parser = ["auto", "layers", "heads"])] attention_placement: String,
    /// Diagnostic choice only; does not qualify or select a shipping default.
    #[arg(long, requires = "split_device", value_parser = ["bf16", "fp32"])] wire: Option<String>,
    /// Teacher-forced one-row serving decode after the unscored prompt.
    #[arg(long, requires = "split_device")] decode: bool,
    #[arg(long, default_value_t = 1024)] prefill_rows: u32,
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..=4))] prefill_waves: u8,
    /// Validate checkpoint/reference/export headers without starting CUDA or peers.
    #[arg(long)] check_headers: bool,
    /// Full-model scoring is the only supported MiniMax golden operation.
    #[arg(long, required = true)] nll: bool,
}

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

const EMBEDDING_UPLOAD: &str = "minimax.unit.0.0.route_ids.upload";

struct RawLogits(std::io::BufWriter<std::fs::File>, Vec<usize>);
impl RawLogits {
    fn create(args: &GoldenArgs, window: &Window) -> Result<Option<Self>> {
        args.raw_logits.then(|| Ok(Self(std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true)
            .open(args.save_logits.join(format!("{}.raw-logits.bin", window.id)))?),
            window.positions.iter().map(|row| row.pos).collect()))).transpose()
    }
    fn row(&mut self, position: usize, logits: &[f32]) -> Result<()> {
        use std::io::Write;
        if self.1.contains(&position) {
            for value in logits { self.0.write_all(&value.to_le_bytes())?; }
        }
        Ok(())
    }
}

fn reference(root: &Path, id: Option<&str>, vocab: usize) -> Result<Vec<Window>> {
    let bytes = std::fs::read(root.join("manifest.json"))?;
    ensure!(digest(&bytes) == fidelity_dataset::MINIMAX_M3_MANIFEST_SHA256,
        "MiniMax reference manifest differs from immutable publication");
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
    ensure!(manifest["config"] == fidelity_dataset::MINIMAX_M3_CONFIG
        && manifest["vocab"].as_u64() == Some(vocab as u64), "MiniMax reference config/vocabulary differs");
    let panel = fidelity_dataset::load(root, &manifest)?;
    let mut windows = fidelity_dataset::quick_windows(&panel)?;
    if let Some(id) = id {
        let ids = id.split(',').collect::<BTreeSet<_>>();
        ensure!(ids.len() == id.split(',').count(), "MiniMax window IDs must be unique");
        windows.retain(|window| ids.contains(window.id.as_str()));
        ensure!(windows.len() == ids.len(), "MiniMax windows must belong to the sealed quick subset");
    }
    ensure!(!windows.is_empty() && windows.iter().all(|window| window.media.is_empty()
        && (2..=16384).contains(&window.tokens.len()) && window.score_from > 0
        && window.score_from < window.tokens.len()), "MiniMax requires bounded text-only causal windows");
    Ok(windows)
}

fn selected_programs(cfg: &MiniMaxM3Config, weights: &CoordinatorWeights,
    manifest: &serde_json::Value, rows: u32, tokens: u32) -> Result<BTreeSet<String>> {
    let (_, msa) = MsaPrograms::resolve(cfg, manifest, 1, rows, tokens)?;
    let (_, producers) = ProducerPrograms::resolve(manifest, rows)?;
    let (_, forward) = ForwardPrograms::resolve(manifest, rows)?;
    let dense = weights.resolve_dense_programs(manifest, rows)?;
    Ok(msa.iter().chain(&producers).chain(&forward).chain(&dense)
        .map(|spec| spec.name.clone()).collect())
}

fn split_selection(args: &GoldenArgs) -> Result<Option<([i32; 2], WirePrecision)>> {
    let Some(peer) = args.split_device else { return Ok(None); };
    ensure!(args.device >= 0 && peer >= 0 && peer != args.device,
        "MiniMax golden peer must be a distinct nonnegative physical device");
    ensure!(matches!(args.attention_placement.as_str(), "auto" | "layers"),
        "MiniMax serving split golden requires layers attention placement");
    let wire = match args.wire.as_deref() {
        Some("bf16") => WirePrecision::Bf16,
        Some("fp32") => WirePrecision::F32,
        _ => return Err(cuteafd_loader::placement::families::minimax_m3::ServingRefusal::SplitWireNotQualified.into()),
    };
    Ok(Some(([args.device, peer], wire)))
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || run(args)).await?
}

fn run(args: GoldenArgs) -> Result<()> {
    ensure!(args.nll && args.prefill_rows >= 256 && args.prefill_rows <= 4096
        && args.prefill_rows % 64 == 0, "MiniMax needs --nll and page-aligned prefill rows 256..=4096");
    ensure!(args.tap_layers.is_empty() || (args.window.is_some()
        && args.tap_layers.iter().all(|&layer| layer < 60)
        && args.tap_layers.iter().collect::<BTreeSet<_>>().len() == args.tap_layers.len()),
        "MiniMax taps require --window and unique layers 0..59");
    let view = args.attention.selected_view.clone().map(Ok)
        .unwrap_or_else(|| args.attention.select(&args.snapshot))?;
    args.attention.validate_view(&view)?;
    crate::cli::publish_core_view(&view);
    let checkpoint = Checkpoint::coordinator_view(&view, false, false)?;
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
    cfg.release_geometry()?;
    let windows = reference(&args.golden, args.window.as_deref(), cfg.vocab_size)?;
    let tokens = u32::try_from(windows.iter().map(|window| window.tokens.len() - 1).max().unwrap())?;
    if args.split_device.is_some() { return run_split(args, view, checkpoint, cfg, windows, tokens); }
    let weights = CoordinatorWeights::plan(&checkpoint, &cfg, 1, &vec![0; cfg.layers])?
        .into_iter().next().context("MiniMax target layout missing")?;
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
    let names = selected_programs(&cfg, &weights, &manifest, args.prefill_rows, tokens)?;
    println!("MiniMax native full golden: reference={} windows={} scored_rows={} programs={}",
        fidelity_dataset::MINIMAX_M3_MANIFEST_SHA256, windows.len(),
        windows.iter().map(|window| window.positions.len()).sum::<usize>(), names.len());
    if args.check_headers { return Ok(()); }
    cuteafd_core::set_expert_geometry(ExpertGeometry::MINIMAX_M3)
        .map_err(|_| anyhow::anyhow!("MiniMax golden needs an isolated process"))?;
    ensure!(args.peers.len() == 4, "MiniMax golden currently qualifies four real Spark ranks");
    ensure!(!args.save_logits.try_exists()? && args.save_logits.parent().is_some_and(Path::is_dir),
        "MiniMax logits output needs an existing parent and a new directory");
    let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
    // SAFETY: every program, allocation and Spark terminal owner remains within
    // the library lifetime; terminal ownership retains an Arc during failures.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&args.native_lib)? });
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let runtime = RuntimeInventory::measure(&library, &[args.device], InventoryPhase::PreWeight,
        |_| { programs.load_matching(|name| names.contains(name))?; Ok(()) })?;
    let sample = &runtime.gpus[0];
    let device = Device { library: &library, id: args.device };
    let config = TcpTransportConfig::default();
    let endpoint = SparkEndpoint { ring_depth: None, capacity: args.prefill_rows,
        ingress: SparkIngress::Bf16, partial_reply_bytes: None };
    let rings = spark_rings(config.max_frame_bytes)?;
    let waves = usize::from(args.prefill_waves);
    let link_layout = PreparedSparkLink::endpoint("spark", 0,
        SparkEndpointOwners { ranks: args.peers.len(), geometry: ExpertGeometry::MINIMAX_M3, endpoint, rings },
        waves, true, true, SparkIntakeCandidates::Probed, SparkReplyFormat::Bf16, Lifetime::TARGET)?;
    let mut contract = admission::contract(&checkpoint, &cfg, &weights, &manifest,
        args.prefill_rows, tokens, waves, &link_layout, &sample.arch)?;
    if !args.tap_layers.is_empty() {
        use cuteafd_loader::placement::planner::{Allocation as PlannedAllocation, ByteCurve, Component, Location};
        let bytes = LayerTaps::host_bytes(args.prefill_rows);
        contract.components.push(Component::allocation(PlannedAllocation::buffer("golden.layer-taps",
            Location::HOST, bytes, Lifetime::TARGET, Category::Workspace, "diagnostic layer download"),
            ByteCurve::fixed(bytes, contract.maximum_pages), false, contract.maximum_pages));
    }
    use crate::shared::{experts::rtx::{dense::HostStaging, push::{push_host_envelope, PushPolicy}},
        token_io::{EmbedPlacement, TokenEmbedding}};
    use cuteafd_loader::placement::planner::{ByteCurve, Component, ComponentKind};
    let policy = PushPolicy::for_snapshot(view.snapshot())?;
    let envelope = if policy == PushPolicy::Required {
        // Every target operand is a direct native upload; no transform pool or
        // retained index is needed, only this coordinator's receive ring.
        let envelope = push_host_envelope(&[None], &[HostStaging::default()])?;
        contract.components.retain(|component| !matches!(&component.kind,
            ComponentKind::Allocation { template, .. } if template.group == "target mount"));
        for allocation in envelope.allocations("minimax.push")? {
            let bytes = allocation.bytes;
            contract.components.push(Component::allocation(allocation,
                ByteCurve::fixed(bytes, contract.maximum_pages), false, contract.maximum_pages));
        }
        let bytes = u64::from(args.prefill_rows) * 4;
        let allocation = cuteafd_loader::placement::planner::Allocation::buffer(EMBEDDING_UPLOAD,
            cuteafd_loader::placement::planner::Location::PINNED, bytes, Lifetime::TARGET,
            Category::Staging, "ordered embedding ids");
        contract.components.push(Component::allocation(allocation,
            ByteCurve::fixed(bytes, contract.maximum_pages), false, contract.maximum_pages));
        contract.add_driver_page_tables();
        Some(envelope)
    } else { None };
    let facts = admission::admit(&contract, sample.free_bytes, sample.total_bytes, host)?;
    let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
    let exports = cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(&catalog)?.exports();
    let executions = cuteafd_loader::plan::spark_executions::admit_family("minimax_m3", &exports,
        None, u64::from(args.prefill_rows), waves as u64)?;
    let model = ModelAdmission::for_clients(ExpertGeometry::MINIMAX_M3, endpoint, rings, 1, executions)?;
    let admission = SparkAdmission::plan_view(&view, ExpertGeometry::MINIMAX_M3, Some(&catalog),
        &SparkWorkingSet { rtx_layers: vec![], spark_layers: (3..cfg.layers).collect(),
            spark_ranks: args.peers.len(), prefill_lanes: 1 }, args.peers.len(), model, None)?;
    let weight_load = crate::shared::memory::load_timing::Split::start();
    let target_load = crate::shared::memory::load_timing::Split::start();
    tracing::info!("MiniMax weight transaction start");
    let (target, embedding): (TargetWeights<'_>, Option<TokenEmbedding<'_>>) = if let Some(envelope) = envelope {
        ensure!(admission.coordinator_pushes(), "MiniMax target and Spark experts must join one push group");
        let (prepared, membership) = super::weights::push::prepare(&view, &catalog, &[device],
            args.peers.len(), vec![], envelope, ||
                TargetWeights::load_with_embedding(device, &checkpoint, &weights, EmbedPlacement::Gpu))?;
        let pending = admission.start_with(&args.peers, membership)?;
        let ((target, embedding), experts) = prepared.receive(&catalog)?;
        ensure!(experts.iter().all(Vec::is_empty), "MiniMax cold golden has no coordinator expert sets");
        target_load.log("minimax.target-resident");
        tracing::info!(resident_bytes = weights.arena.bytes(), "MiniMax target weights resident");
        let spark_wait = crate::shared::memory::load_timing::Split::start();
        tracing::info!("MiniMax pending Spark wait start");
        pending.wait()?;
        spark_wait.log("minimax.experts-resident.wait");
        (target, Some(embedding))
    } else {
        let pending = admission.start(&args.peers)?;
        let target = TargetWeights::mount(device, &checkpoint, &weights)?;
        target_load.log("minimax.target-resident");
        tracing::info!(resident_bytes = weights.arena.bytes(), "MiniMax target weights resident");
        let spark_wait = crate::shared::memory::load_timing::Split::start();
        tracing::info!("MiniMax pending Spark wait start");
        pending.wait()?;
        spark_wait.log("minimax.experts-resident.wait");
        (target, None)
    };
    weight_load.log("minimax.experts-resident.total");
    let kernels = Kernels::bind(device, &programs, &manifest, &cfg, &weights, args.prefill_rows, tokens)?;
    let stream = Rc::new(Stream::new(device)?);
    let mut owners = (0..waves).map(|_| UnitOwner::new(&kernels, Rc::clone(&stream), &cfg, &weights,
        args.prefill_rows)).collect::<Result<Vec<_>>>()?;
    let cache = Cache::new(Rc::clone(&stream), &cfg, tokens as usize)?;
    let mut metadata = (0..waves).map(|_| cache.metadata(args.prefill_rows)).collect::<Result<Vec<_>>>()?;
    let head = Head::new(Rc::clone(&stream), &programs, &cfg, args.prefill_rows as usize)?;
    let residual = Allocation::new(device, tokens as usize * 6144 * 4, Tag::workspace("minimax/residual"))?;
    let embedding_ids = crate::shared::prefix::view(owners[0].plane("route_ids")?, 0,
        args.prefill_rows as usize * 4)?;
    let embedding_upload = if embedding.is_some() {
        device.run(|| {
            // SAFETY: this stable full-capacity U32 view and the issuing stream
            // live through all golden work; every consumer is stream ordered.
            unsafe { library.reserve_h2d_ordered(embedding_ids, stream.raw) }
        })?;
        Some(library.ordered_h2d_staging_buffer(embedding_ids, stream.raw)?
            .context("MiniMax ordered embedding staging was not reserved")?)
    } else { None };
    let devices = [args.device];
    let mut inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "minimax",
        Lifetime::TARGET, Category::Workspace, "cold golden");
    target.inventory(&mut inventory, "target")?;
    if let Some(embedding) = &embedding {
        embedding.inventory(&mut inventory, "embedding", Lifetime::TARGET)?;
    }
    cache.inventory(&mut inventory, "cache")?;
    head.inventory(&mut inventory, "head")?;
    residual.inventory(&mut inventory, "residual")?;
    for (slot, owner) in owners.iter().enumerate() { owner.inventory(&mut inventory, &format!("unit.{slot}"))?; }
    for (slot, owner) in metadata.iter().enumerate() { owner.inventory(&mut inventory, &format!("metadata.{slot}"))?; }
    if let Some(buffer) = embedding_upload {
        let planned = facts.allocations.iter().find(|allocation| allocation.id == EMBEDDING_UPLOAD)
            .context("MiniMax ordered embedding staging was not admitted")?;
        ensure!(buffer.bytes as u64 == planned.bytes && buffer.bytes == args.prefill_rows as usize * 4,
            "MiniMax ordered embedding staging differs from exact admitted extent");
        inventory.receipts.push(cuteafd_loader::placement::planner::Receipt {
            allocation: planned.clone(), address: buffer.ptr as u64 });
    }
    admission::reconcile_sync_h2d(&library, &facts, &mut inventory)?;
    let receipts = inventory.receipts.iter().map(|receipt| serde_json::json!({
        "allocation":receipt.allocation, "address":receipt.address })).collect::<Vec<_>>();
    tracing::info!(receipts = %serde_json::to_string(&receipts)?, "MiniMax retained owner inventory");
    let mut link = device.run(|| SparkLink::prepared(&library, SparkWorld::canonical(args.peers.len())?,
        &args.peers, config, link_layout))?;
    link.enable_terminal_ownership(Arc::clone(&library))?;
    // The link is declared after all compute owners. Every error, including
    // bootstrap, goes through this terminal block before their destructors run.
    let result = (|| {
        let mut shape = SparkSessionShape::new(args.prefill_rows, 6144, 4, ExpertV2Dtype::Bf16)?;
        shape.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        device.run(|| link.bootstrap(&shape))?;
        score(&args, &view, &cfg, &weights, &target, embedding.as_ref(), &kernels, &cache, &head, &residual,
            &mut owners, &mut metadata, &mut link)
    })();
    let terminal = link.terminal_quiesce().and_then(|_| link.terminal_drain_copies())
        .and_then(|_| stream.drain()).and_then(|_| link.terminal_release());
    if let Err(error) = terminal {
        tracing::error!(%error, "MiniMax terminal drainage failed; refusing to retire queued compute storage");
        std::process::abort();
    }
    let mut final_inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "minimax",
        Lifetime::RESIDENT, Category::Staging, "cold golden completion");
    admission::reconcile_sync_h2d(&library, &facts, &mut final_inventory)?;
    if let Some(initial) = embedding_upload {
        let actual = library.ordered_h2d_staging_buffer(embedding_ids, stream.raw)?
            .context("MiniMax ordered embedding staging retired before completion")?;
        ensure!(actual.ptr == initial.ptr && actual.bytes == initial.bytes,
            "MiniMax ordered embedding staging grew after first reserve");
    }
    result
}

fn run_split(args: GoldenArgs, view: CheckpointView, checkpoint: Checkpoint,
    cfg: MiniMaxM3Config, windows: Vec<Window>, tokens: u32) -> Result<()> {
    use super::{cache::{Placement, RequestStreams}, engine::{Requests, RoutedBinding, ServingPrefill, ServingRequest},
        experts::Dispatcher, owners::{LoadedRank, Owners}};
    use cuteafd_loader::{families::minimax_m3::admission::{gather, Coordinator, FactsRequest},
        placement::{AttentionPlacement, Onboard, ResidentWeights, planner::{Context as ModelContext, Operation, PrefillShape}}};
    use cuteafd_api::openai::probe::{Probe, ProbeSpec};
    let (devices, wire) = split_selection(&args)?.context("MiniMax split golden peer missing")?;
    ensure!(!args.decode || args.tap_layers.is_empty(), "MiniMax split layer taps require prefill scoring");
    let manifest: Arc<serde_json::Value> = Arc::new(serde_json::from_slice(&std::fs::read(&args.manifest)?)?);
    let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
    let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
    let fact_request = |coordinator| FactsRequest { checkpoint: &checkpoint, cfg: &cfg, catalog: &catalog,
        manifest_path: &args.manifest, manifest: Arc::clone(&manifest), coordinator,
        attention: Some(AttentionPlacement::Layers), split_wire: Some(wire), physical_devices: &devices,
        spark_ranks: args.peers.len(), spark_bytes: 100 << 30,
        prefill: PrefillShape { rows: u64::from(args.prefill_rows), waves: u64::from(args.prefill_waves) },
        context_tokens: u64::from(tokens), pool_tokens: Some(u64::from(tokens).div_ceil(64) * 64), sequences: 1,
        host, host_cache_bytes: Some(0), prefix_entries: 1, replies: Default::default(),
        head: crate::shared::target_head::TargetHead::configured().blas_head(true), expert_usage: false,
        resident_weight_limit: ResidentWeights::DEFAULT_LIMIT };
    let planned = gather(&fact_request(Coordinator::Planned { cards: devices.iter().map(|_| {
        cuteafd_loader::placement::CoordinatorCard { physical_bytes: 96 << 30, logical_bytes: 96 << 30 }
    }).collect() })).map_err(|error| anyhow::anyhow!("{}; {}", error.what, error.how))?;
    if args.check_headers {
        planned.admit(cuteafd_loader::placement::families::minimax_m3::request(planned.context,
            Some(u64::from(tokens).div_ceil(64) * 64), Onboard::Layers(0), planned.prefill,
            Operation::Serving, None, Default::default()))?;
        return Ok(());
    }
    ensure!(args.peers.len() == 4, "MiniMax split golden qualifies four real Spark ranks");
    ensure!(!args.save_logits.try_exists()? && args.save_logits.parent().is_some_and(Path::is_dir),
        "MiniMax split logits output needs an existing parent and new directory");
    cuteafd_core::set_expert_geometry(ExpertGeometry::MINIMAX_M3)
        .map_err(|_| anyhow::anyhow!("MiniMax golden needs an isolated process"))?;
    // SAFETY: all owners and terminal dispatchers retire inside this Arc lifetime.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&args.native_lib)? });
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let runtime = RuntimeInventory::measure(&library, &devices, InventoryPhase::PreWeight, |_| {
        programs.load_matching(|name| name.starts_with("minimax_") || name.starts_with("exlr8_dense_"))?;
        Ok(())
    })?;
    let policy = cuteafd_core::memory_policy::MemoryPolicy::current();
    let inputs = gather(&fact_request(Coordinator::Measured { samples: runtime.gpus.iter().map(|sample|
        (sample.total_bytes, policy.gpu_cap_bytes(sample.total_bytes), sample.free_bytes)).collect(),
        loaded_code: vec![0; 2] })).map_err(|error| anyhow::anyhow!("{}; {}", error.what, error.how))?;
    let admitted = inputs.admit(cuteafd_loader::placement::families::minimax_m3::request(inputs.context,
        Some(u64::from(tokens).div_ceil(64) * 64), Onboard::Layers(0), inputs.prefill,
        Operation::Serving, None, Default::default()))?;
    if !args.tap_layers.is_empty() {
        use cuteafd_loader::placement::planner::{Allocation as PlannedAllocation, Location};
        let mut diagnostic = admitted.solved.facts.clone();
        diagnostic.allocations.push(PlannedAllocation::buffer("golden.layer-taps", Location::HOST,
            LayerTaps::host_bytes(args.prefill_rows), Lifetime::TARGET, Category::Workspace,
            "diagnostic layer download"));
        cuteafd_loader::placement::planner::evaluate(&diagnostic, &inputs.budgets)?;
    }
    let endpoint = SparkEndpoint { ring_depth: None, capacity: args.prefill_rows,
        ingress: SparkIngress::Bf16, partial_reply_bytes: None };
    let config = TcpTransportConfig::default();
    let rings = spark_rings(config.max_frame_bytes)?;
    let waves = usize::from(args.prefill_waves);
    let exports = cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(&catalog)?.exports();
    let executions = cuteafd_loader::plan::spark_executions::admit_family("minimax_m3", &exports,
        None, u64::from(args.prefill_rows), waves as u64)?;
    let model = ModelAdmission::for_clients(ExpertGeometry::MINIMAX_M3, endpoint, rings, 1, executions)?;
    let spark_admission = SparkAdmission::plan_view(&view, ExpertGeometry::MINIMAX_M3, Some(&catalog),
        &SparkWorkingSet { rtx_layers: vec![], spark_layers: inputs.routed.clone(), spark_ranks: args.peers.len(),
            prefill_lanes: 1 }, args.peers.len(), model, None)?;
    let native_devices = devices.map(|id| Device { library: &library, id });
    let (pending, membership) = super::weights::push::prepare_serving(&view, &catalog, &checkpoint,
        &native_devices, &inputs, &admitted)?;
    let spark_pending = spark_admission.start_with(&args.peers, membership)?;
    let bundle = pending.finish(&catalog, Some(spark_pending))?;
    ensure!(bundle.expert_sets.is_empty(), "MiniMax diagnostic count0 must not retain local expert sets");
    let loaded = bundle.targets.into_iter().zip(native_devices).map(|(weights, device)| {
        let decode = Rc::new(Stream::new(device)?);
        Ok(LoadedRank { weights, streams: RequestStreams { decode: Rc::clone(&decode),
            prefill: vec![decode; waves] } })
    }).collect::<Result<Vec<_>>>()?;
    let mut owners = Owners::new(&inputs, &admitted, &programs, loaded, bundle.embedding)?;
    let link_layout = PreparedSparkLink::endpoint("spark", 0,
        SparkEndpointOwners { ranks: args.peers.len(), geometry: ExpertGeometry::MINIMAX_M3, endpoint, rings },
        waves, true, true, SparkIntakeCandidates::Probed, SparkReplyFormat::Bf16, Lifetime::TARGET)?;
    let mut link = native_devices[0].run(|| SparkLink::prepared(&library,
        SparkWorld::canonical(args.peers.len())?, &args.peers, config, link_layout))?;
    link.enable_terminal_ownership(Arc::clone(&library))?;
    let mut shape = SparkSessionShape::new(args.prefill_rows, 6144, 4, ExpertV2Dtype::Bf16)?;
    shape.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    native_devices[0].run(|| link.bootstrap(&shape))?;
    let bindings = cfg.moe.iter().map(|&moe| moe.then_some(RoutedBinding {
        dispatcher: 0, route: crate::shared::prefill_pipeline::FfnRoute::Spark })).collect();
    let mut execution = ServingPrefill::new(&cfg, vec![Dispatcher::spark(link)?], bindings, 17)?;
    let schedule = Requests::new(inputs.request_capacity()?, tokens as usize, cfg.layers, waves)?;
    let provenance = serde_json::json!({"family":"minimax_m3", "snapshot":view.snapshot(),
        "reference_manifest_sha256":fidelity_dataset::MINIMAX_M3_MANIFEST_SHA256,
        "native_lib":args.native_lib,"manifest_sha256":digest(&std::fs::read(&args.manifest)?),
        "split_devices":devices,"attention_placement":"layers","wire":args.wire,
        "serving_execution":true,"decode":args.decode,"prefill_rows":args.prefill_rows,
        "prefill_waves":args.prefill_waves,"diagnostic_wire_not_qualified":true});
    std::fs::create_dir(&args.save_logits)?;
    let mut records = Vec::new();
    for (identity, window) in windows.into_iter().enumerate() {
        let count = window.tokens.len() - 1;
        let dump = args.save_logits.join(&window.id);
        let mut raw = RawLogits::create(&args, &window)?;
        let probe = Probe::new(ProbeSpec { cold: true, no_speculation: true,
            prompt_ids: Some(window.tokens.clone()), score_from: Some(window.score_from),
            score_positions: Some(window.positions.iter().map(|position| position.pos).collect()),
            top_k: 32, want: window.want(), dump_rows: Some(dump.clone()), ..Default::default() });
        probe.admitted("minimax_m3", &window.tokens, 0);
        probe.selected_score_path(if args.decode { "decode" } else { "prefill" });
        probe.provenance(provenance.clone());
        let mut placement = Placement { units: (0..count.div_ceil(64)).map(|page| page as u32).collect(), slot: 0, len: 0 };
        for rank in &owners.storage.requests.borrow().ranks {
            let stream = &rank.bank.readers()[0];
            rank.bank.install(&placement, stream.raw)?;
        }
        let first_decode = if args.decode { window.score_from.saturating_sub(1).max(1).min(count) } else { count };
        let mut request = [ServingRequest { identity: identity as u64 + 1, kind: cuteafd_transport::ExpertV2SourceKind::Prefill,
            placement: &mut placement, start: 0, tokens: &window.tokens[..first_decode], heads: 0..first_decode }];
        let mut taps = if args.tap_layers.is_empty() { None } else {
            let parent = args.save_logits.join("layer-taps");
            std::fs::create_dir_all(&parent)?;
            let mut taps = LayerTaps::new(parent.join(&window.id), &args.tap_layers, args.prefill_rows)?;
            taps.residual_only = args.tap_residual_only;
            Some(taps)
        };
        execution.execute_with_taps(&owners.storage, &cfg, &schedule, &mut request, false, |_, positions, logits| {
            let values = logits.to_host(&library)?;
            for (row, logits) in values.chunks_exact(cfg.vocab_size).enumerate() {
                let position = positions.start + row + 1;
                if position >= window.score_from {
                    if let Some(raw) = &mut raw { raw.row(position, logits)?; }
                    probe.row(position, logits);
                }
            }
            Ok(())
        }, taps.as_mut())?;
        for position in first_decode..count {
            let request = [ServingRequest { identity: identity as u64 + 1, kind: cuteafd_transport::ExpertV2SourceKind::Decode,
                placement: &mut placement, start: position, tokens: &window.tokens[position..position + 1], heads: position..position + 1 }];
            let logits = execution.decode(&owners.storage, &cfg, &schedule, &request,
                &mut owners.graphs.borrow_mut(), false)?;
            let values = logits.to_host(&library)?;
            if let Some(raw) = &mut raw { raw.row(position + 1, &values)?; }
            probe.row(position + 1, &values);
            placement.len = position + 1;
        }
        let record = probe.record();
        ensure!(record.error.is_none(), "MiniMax split row dump failed: {:?}", record.error);
        let mut scored = window.score(&record.rows);
        fidelity_rows::score_compact(cfg.vocab_size, &window, &dump, &mut scored)?;
        ensure!(scored.missing == 0 && scored.non_finite == 0, "MiniMax split scored rows missing/nonfinite");
        println!("MiniMax split {} wire={:?}: rows={} top1={:.6} NLL={:.8} KL={:.8}",
            window.id, args.wire, scored.positions, scored.top1, scored.nll, scored.kl);
        serde_json::to_writer(std::fs::OpenOptions::new().write(true).create_new(true)
            .open(args.save_logits.join(format!("{}.json", window.id)))?,
            &serde_json::json!({"probe":record,"score":scored}))?;
        records.extend(scored.records);
    }
    execution.drain(&owners.storage)?;
    let report = cuteafd_bench::fidelity::minimax_golden_report(Fidelity::from_records(records), provenance)?;
    serde_json::to_writer(std::fs::OpenOptions::new().write(true).create_new(true)
        .open(args.save_logits.join("fidelity.json"))?, &report)?;
    ensure!(report["qualified"] == true, "MiniMax split qualification failed: {}", report["verdict"]);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn score<'a>(args: &GoldenArgs, view: &CheckpointView, cfg: &MiniMaxM3Config,
    admitted: &CoordinatorWeights, weights: &TargetWeights<'a>,
    embedding: Option<&crate::shared::token_io::TokenEmbedding<'a>>, kernels: &Kernels<'a>,
    cache: &Cache<'a>, head: &Head<'a>, residual: &Allocation<'a>, owners: &mut [UnitOwner<'a>],
    metadata: &mut [super::cache::Metadata<'a>], link: &mut SparkLink<'a>) -> Result<()> {
    use cuteafd_api::openai::probe::{Probe, ProbeSpec};
    let windows = reference(&args.golden, args.window.as_deref(), cfg.vocab_size)?;
    std::fs::create_dir(&args.save_logits)?;
    let provenance = serde_json::json!({"family":"minimax_m3", "snapshot":view.snapshot(),
        "reference_manifest_sha256":fidelity_dataset::MINIMAX_M3_MANIFEST_SHA256,
        "native_lib":args.native_lib,"manifest_sha256":digest(&std::fs::read(&args.manifest)?),
        "prefill_rows":args.prefill_rows,"prefill_waves":args.prefill_waves,
        "expert_ingress":"BF16; native calibrated NVFP4 W4A4 experts", "shared_prefill":true});
    let mut records = Vec::new();
    let mut next_identity = 1;
    for window in windows {
        let count = window.tokens.len() - 1;
        let dump = args.save_logits.join(&window.id);
        let mut raw = RawLogits::create(&args, &window)?;
        let probe = Probe::new(ProbeSpec { cold: true, no_speculation: true,
            prompt_ids: Some(window.tokens.clone()), score_from: Some(window.score_from),
            score_positions: Some(window.positions.iter().map(|position| position.pos).collect()),
            top_k: 32, want: window.want(), dump_rows: Some(dump.clone()), ..Default::default() });
        probe.admitted("minimax_m3", &window.tokens, 0);
        probe.selected_score_path("prefill");
        probe.provenance(provenance.clone());
        for first in (0..count).step_by(args.prefill_rows as usize) {
            let end = (first + args.prefill_rows as usize).min(count);
            let span = crate::shared::prefix::view(residual.buffer, first * 6144 * 4, (end - first) * 6144 * 4)?;
            if let Some(embedding) = embedding {
                let owner = &owners[0];
                let ids = crate::shared::prefix::view(owner.plane("route_ids")?, 0,
                    args.prefill_rows as usize * 4)?;
                Head::embed_tokens(cfg, args.prefill_rows as usize, embedding,
                    &window.tokens[first..end], &owner.stream, ids, owner.dense.producer_destination(), span)?;
            } else { head.embed(weights, &window.tokens[first..end], span)?; }
        }
        let mut taps = if args.tap_layers.is_empty() { None } else {
            let parent = args.save_logits.join("layer-taps");
            std::fs::create_dir_all(&parent)?;
            let root = parent.join(&window.id);
            let mut taps = LayerTaps::new(root.clone(), &args.tap_layers, args.prefill_rows)?;
            taps.residual_only = args.tap_residual_only;
            std::fs::write(root.join("meta.json"), serde_json::to_vec(&serde_json::json!({
                "diagnostic_only":true,"window":window.id,"tokens":window.tokens,
                "input_rows":count,"hidden":6144,"layers":args.tap_layers,
                "residual_dtype":"F32","branch_dtype":"BF16","byte_order":"little",
                "residual_only":args.tap_residual_only,"final_norm_dtype":"BF16",
                "row_index":"zero-based input token","provenance":provenance}))?)?;
            Some(taps)
        };
        let units = sequence_units(count, args.prefill_rows as usize, owners.len(), cfg.layers)?;
        Prefill { cfg, admitted, weights, kernels, cache, owners, metadata, waves: link.exchange()?,
            residual: residual.buffer, placement_version: 17, next_identity: &mut next_identity,
            taps: taps.as_mut() }.run(&units)?;
        for chunk in units {
            let first = chunk.rows.start;
            let rows = chunk.rows.len();
            let span = crate::shared::prefix::view(residual.buffer, first * 6144 * 4, rows * 6144 * 4)?;
            let logits = head.project(weights, span, rows)?;
            if let Some(taps) = &mut taps { taps.final_norm(head, first, rows)?; }
            for (row, logits) in logits.chunks_exact(cfg.vocab_size).enumerate() {
                let position = first + row + 1;
                if position >= window.score_from {
                    if let Some(raw) = &mut raw { raw.row(position, logits)?; }
                    probe.row(position, logits);
                }
            }
            probe.scored_prefill_chunk(first + 1, rows);
        }
        let record = probe.record();
        ensure!(record.error.is_none(), "MiniMax row dump failed: {:?}", record.error);
        let mut scored = window.score(&record.rows);
        ensure!(scored.missing == 0 && scored.non_finite == 0 && scored.records.len() == window.positions.len(),
            "MiniMax full forward did not cover all finite scored positions");
        fidelity_rows::score_compact(cfg.vocab_size, &window, &dump, &mut scored)?;
        println!("MiniMax {}: rows={} top1={:.6} NLL={:.8} KL(top1024+tail)={:.8}",
            window.id, scored.positions, scored.top1, scored.nll, scored.kl);
        serde_json::to_writer(std::fs::OpenOptions::new().write(true).create_new(true)
            .open(args.save_logits.join(format!("{}.json", window.id)))?,
            &serde_json::json!({"probe":record,"score":scored}))?;
        records.extend(scored.records);
    }
    let report = cuteafd_bench::fidelity::minimax_golden_report(Fidelity::from_records(records), provenance)?;
    serde_json::to_writer(std::fs::OpenOptions::new().write(true).create_new(true)
        .open(args.save_logits.join("fidelity.json"))?, &report)?;
    println!("MiniMax generated-aggregate verdict: {}", report["verdict"]);
    ensure!(report["qualified"] == true,
        "MiniMax qualification failed: {}", report["verdict"]["reasons"]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(clap::Parser)]
    struct ParseGolden { #[command(flatten)] args: GoldenArgs }

    fn parse_args(extra: &[&str]) -> std::result::Result<GoldenArgs, clap::Error> {
        use clap::Parser;
        let mut args = vec!["golden", "--snapshot", "/snapshot", "--native-lib", "/native.so",
            "--golden", "/reference", "--save-logits", "/output", "--nll"];
        args.extend_from_slice(extra);
        ParseGolden::try_parse_from(args).map(|parsed| parsed.args)
    }

    #[test]
    fn peer_args_require_explicit_wire_and_keep_no_peer_whole_selection() -> Result<()> {
        let single = parse_args(&[])?;
        assert!(split_selection(&single)?.is_none());
        assert!(parse_args(&["--split-device", "1"]).is_err());
        assert!(parse_args(&["--wire", "bf16"]).is_err());
        for (name, expected) in [("bf16", WirePrecision::Bf16), ("fp32", WirePrecision::F32)] {
            let args = parse_args(&["--device", "3", "--split-device", "1", "--wire", name,
                "--attention-placement", "layers"])?;
            assert_eq!(split_selection(&args)?, Some(([3, 1], expected)));
        }
        assert!(split_selection(&parse_args(&["--split-device", "0", "--wire", "bf16"])?).is_err());
        assert!(split_selection(&parse_args(&["--split-device", "1", "--wire", "fp32",
            "--attention-placement", "heads"])?).is_err());
        Ok(())
    }

    #[test]
    fn peer_checkpoint_preserves_the_canonical_cold_spark_operand_headers() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let (whole, cfg) = super::super::owners::tests::snapshot(directory.path())?;
        let view = CheckpointView::select(directory.path(), cuteafd_loader::plan::checkpoint::CoreView::Original)?;
        let peer = Checkpoint::coordinator_view(&view, false, false)?;
        let layers = (0..cfg.layers).filter(|&layer| cfg.moe[layer]).collect::<Vec<_>>();
        let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
        let charges = |checkpoint: &Checkpoint| cuteafd_loader::families::minimax_m3::admission::spark_charges(
            checkpoint, &catalog, &layers, 4, cuteafd_loader::placement::planner::PrefillShape { rows: 1024, waves: 3 });
        assert_eq!(charges(&peer)?, charges(&whole)?);
        let scalar = "language_model.model.layers.2.block_sparse_moe.experts.0.w1.input_global_scale";
        assert!(peer.header(scalar).is_none(), "target load remains coordinator-scoped");
        assert!(whole.header(scalar).is_some(), "shared Spark charge reader supplies full native headers");
        Ok(())
    }

    #[test]
    #[ignore = "requires installed MiniMax native headers and Spark package inventories"]
    fn installed_peer_and_cold_whole_have_identical_spark_charges() -> Result<()> {
        let home = std::env::var_os("HF_HOME").map(PathBuf::from).unwrap_or_else(||
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface"));
        let snapshot = home.join("hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e");
        let view = CheckpointView::select(&snapshot, cuteafd_loader::plan::checkpoint::CoreView::Original)?;
        let peer = Checkpoint::coordinator_view(&view, false, false)?;
        let whole = Checkpoint::open_view(&view)?;
        let cfg = MiniMaxM3Config::from_hf(&whole.config)?;
        let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
        let routed = (0..cfg.layers).filter(|&layer| cfg.moe[layer]).collect::<Vec<_>>();
        let prefill = cuteafd_loader::placement::planner::PrefillShape { rows: 1024, waves: 3 };
        let charges = |checkpoint: &Checkpoint| cuteafd_loader::families::minimax_m3::admission::spark_charges(
            checkpoint, &catalog, &routed, 4, prefill);
        assert_eq!(charges(&peer)?, charges(&whole)?);
        Ok(())
    }

    #[test]
    fn reference_rejects_an_unpinned_manifest_before_loading_any_rows() -> Result<()> {
        let root = tempfile::tempdir()?;
        std::fs::write(root.path().join("manifest.json"), b"{}")?;
        assert!(reference(root.path(), None, 200064).unwrap_err().to_string()
            .contains("immutable publication"));
        Ok(())
    }
    #[test]
    fn golden_uses_shared_estimator_and_terminal_release_before_retirement() {
        let source = include_str!("golden.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("set_expert_geometry(ExpertGeometry::MINIMAX_M3)"));
        assert!(source.find("set_expert_geometry(").unwrap() < source.find("NativeLibrary::load(").unwrap());
        assert!(source.contains("fidelity_rows::score_compact("));
        assert!(source.contains("sequence_units("));
        assert!(source.contains("pending.wait()?"));
        assert!(source.contains("terminal_quiesce().and_then(|_| link.terminal_drain_copies())"));
        assert!(source.contains("stream.drain()).and_then(|_| link.terminal_release())"));
    }
}
