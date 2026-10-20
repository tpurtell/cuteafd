//! Eager native target and opt-in MTP over the shared scheduler.
use super::{admission::{self, RuntimeInputs}, backend::{Inputs, Step5Backend}, cache::CacheRank,
    execution::Execution, transaction::Staging};
use crate::shared::{inventory::{InventoryPhase, RuntimeInventory}, memory::device::Device,
    placement::{spark_rings, SparkAdmission}, prefix::PrefixArgs, qwen::scheduler::{self, Backend},
    spark_intake::SparkLane, token_io::{EmbedPlacement, EmbedSource, TokenEmbedding, TokenIoArgs}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::{placement::{Demand, GpuBudget, Inventory, ProgramSet, SparkWorkingSet},
    plan::{checkpoint::Checkpoint, families::step5::Config, spec::DraftOwnership}};
use cuteafd_transport::{expert::SparkWorld, worker_selection::ModelAdmission, TcpTransportConfig};
use std::{cell::RefCell, net::SocketAddr, path::PathBuf, sync::{Arc, Mutex}};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long)]
    split_device: Option<i32>,
    #[arg(long, value_delimiter = ',', required = true)]
    peers: Vec<SocketAddr>,
    #[arg(long)]
    pool_tokens: Option<u64>,
    #[arg(long, default_value_t = 1_048_576)]
    max_context: u64,
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u32).range(1..=4096))]
    prefill_rows: u32,
    /// Native execution lanes, independent of scheduler sequence slots.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..=2))]
    execution_lanes: u8,
    /// Native predictor depth. Opt-in until live lossless and speed qualification.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=3))]
    mtp: u8,
    #[arg(long, default_value = "0.0.0.0:8000")]
    listen: String,
    #[arg(long, default_value_t = 32768)]
    max_output: u32,
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u8).range(1..=64))]
    max_sequences: u8,
    #[arg(long)]
    model_id: Option<String>,
    /// All prefill rows are always scored by this eager target.
    #[arg(long)]
    full_prefill_logits: bool,
    #[command(flatten)]
    prefix: PrefixArgs,
    #[command(flatten)]
    decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)]
    token_io: TokenIoArgs,
    #[command(flatten)]
    api: crate::shared::api::ApiArgs,
}

impl ServeArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.device >= 0 && self.split_device.is_none_or(|d| d >= 0 && d != self.device),
            "Step-5 needs one GPU or two distinct actual GPUs");
        ensure!(matches!(self.peers.len(), 4 | 6), "Step-5 requires TP4 or TP6 Spark experts");
        ensure!(self.peers.iter().enumerate().all(|(i, peer)| !self.peers[..i].contains(peer)),
            "Step-5 Spark peers must be distinct");
        ensure!((262_144..=1_048_576).contains(&self.max_context), "Step-5 request context must be 256K..=1M");
        ensure!(usize::from(self.max_sequences) <= self.prefill_rows as usize,
            "Step-5 sequence capacity exceeds row capacity");
        ensure!(usize::from(self.mtp) < self.prefill_rows as usize,
            "Step-5 predictor depth exceeds row capacity");
        ensure!(self.token_io.embed_placement == EmbedPlacement::Gpu,
            "Step-5 requires admitted GPU token embeddings");
        ensure!(self.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Step-5 supports exact prefix frontiers only");
        Ok(())
    }
}

/// The sealed package's actual compiled extent, not the live row count. All
/// layouts must agree; worker admission independently validates loaded exports.
fn compiled_capacity(manifest: &serde_json::Value, rows: u32) -> Result<u32> {
    let layouts = manifest["layouts"].as_object().context("Step-5 package layouts missing")?;
    let mut compiled = None;
    for layout in layouts.values() {
        ensure!(layout["hidden"] == 4096 && layout["experts"] == 352
            && layout["top_k"] == 8 && layout["intermediate"] == 1536,
            "Step-5 package geometry differs from checkpoint");
        let capacity = layout["capacities"].as_array().context("Step-5 package capacities missing")?
            .iter().map(|v| v["capacity"].as_u64().context("Step-5 package capacity must be integer"))
            .collect::<Result<Vec<_>>>()?.into_iter().max().context("Step-5 package capacities empty")?;
        let capacity = u32::try_from(capacity)?;
        ensure!((rows..=4096).contains(&capacity) && compiled.is_none_or(|n| n == capacity),
            "Step-5 installed package extents disagree or cannot cover live rows");
        compiled = Some(capacity);
    }
    compiled.context("Step-5 installed package has no layouts")
}

fn text_media() -> cuteafd_engine::media::MediaAdmission<scheduler::Prompt, crate::shared::qwen::Encoder> {
    cuteafd_engine::media::MediaAdmission::new(cuteafd_engine::media::EmbeddingCache::new(0),
        crate::shared::qwen::Encoder::Off, 16)
}

fn with_backend<T>(args: &ServeArgs,
    run: impl FnOnce(&Step5Backend<'_, '_>, usize) -> Result<T>) -> Result<T> {
    args.validate()?;
    let checkpoint = Checkpoint::coordinator(&args.snapshot, false, false)?;
    let cfg = Config::parse(&checkpoint)?;
    let mut catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    catalog.select_draft(DraftOwnership::None)?;
    ensure!(catalog.routed_experts().geometry()? == ExpertGeometry::STEP5, "Step-5 routed geometry mismatch");
    let format = catalog.fp8().context("Step-5 requires native FP8 or NVFP4 expert headers")?.format();
    ensure!(matches!(format, cuteafd_loader::formats::fp8_experts::ExpertFormat::Fp8Block128
        | cuteafd_loader::formats::fp8_experts::ExpertFormat::Nvfp4),
        "Step-5 BF16 is a fidelity reference, not a serving fallback");
    cuteafd_core::set_expert_geometry(ExpertGeometry::STEP5)
        .map_err(|_| anyhow::anyhow!("Step-5 serving requires an isolated process"))?;
    let package = args.native_lib.parent().context("native library directory")?.join("fp8")
        .join(format.package_name("step5")).join("manifest.json");
    let metadata = cuteafd_loader::plan::checkpoint::read_json(&package)?;
    let compiled = compiled_capacity(&metadata, args.prefill_rows)?;
    // SAFETY: this owner outlives all programs, native lanes, transports and allocations.
    let library = Arc::new(unsafe { NativeLibrary::load(&args.native_lib)? });
    library.cuda_set_device(args.device)?;
    let ids = std::iter::once(args.device).chain(args.split_device).collect::<Vec<_>>();
    let devices = ids.iter().map(|&id| Device { library: &library, id }).collect::<Vec<_>>();
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms {
        family: "step5", split_family: args.split_device.map(|_| "step52") };
    let measured = RuntimeInventory::measure(&library, &ids, InventoryPhase::PreWeight, |id| {
        programs.load_matching(|name| selected.contains(name))?;
        if let Some(other) = ids.iter().copied().find(|&other| other != id) {
            library.cuda_enable_peer(other)?;
            library.tp2_native_initialize()?;
            library.peer_exchange_initialize()?;
            library.peer_abort_initialize()?;
            if std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").as_deref() == Ok("1") {
                library.peer_diagnostic_initialize()?;
            }
        }
        Ok(())
    })?;
    let selected_programs = ProgramSet::from_names(&["step5", "step52"], programs.names());
    let pending_code = ids.iter().enumerate().map(|(rank, &id)| crate::shared::inventory::pending_code(
        &library, id, rank, ids.len() == 2, "step5", &format.package_name("step5"), &selected_programs))
        .collect::<Result<Vec<_>>>()?;
    let headroom = cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes;
    let mut inputs = RuntimeInputs { inventory: Inventory { gpus: measured.baselines(&[]).into_iter()
        .map(|(capacity_bytes, baseline)| GpuBudget { capacity_bytes, baseline, headroom_bytes: headroom }).collect(),
        spark_ranks: args.peers.len(), peer_access: ids.len() == 2 }, rows: args.prefill_rows as usize,
        sequences: usize::from(args.max_sequences), lanes: usize::from(args.execution_lanes), context: args.max_context,
        requested_pool: args.pool_tokens,
        marks: admission::prefix_marks(&args.prefix, usize::from(args.max_sequences)),
        peer_diagnostics: std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").as_deref() == Ok("1"), pending_code, startup: vec![] };
    let admission = admission::admit_mtp(&programs, &checkpoint, &inputs, usize::from(args.mtp))?;
    // Uploads convert on the host and copy synchronously. No transient device
    // conversion arena exists: each loading phase is bounded by retained owners.
    inputs.startup = admission.placement.items.iter().enumerate().map(|(rank, items)| {
        let bytes = items.iter().try_fold(0u64, |sum, item| sum.checked_add(item.bytes))
            .context("Step-5 loading phase overflow")?;
        Ok(Demand::new(rank as u8, cuteafd_core::memory_layout::Category::Workspace,
            "Step-5 retained loading peak", bytes, cuteafd_core::memory_layout::Basis::Exact))
    }).collect::<Result<Vec<_>>>()?;
    let admission = admission::admit_mtp(&programs, &checkpoint, &inputs, usize::from(args.mtp))?;
    measured.record("step5", &measured.gpus.iter().map(|g| g.free_bytes).collect::<Vec<_>>());
    let endpoint = SparkEndpoint { capacity: args.prefill_rows, ingress: SparkIngress::Bf16 };
    let clients = admission.ownership.routed_endpoints()?;
    let config = TcpTransportConfig::default();
    SparkAdmission::plan(&args.snapshot, ExpertGeometry::STEP5, Some(&catalog), &SparkWorkingSet {
        rtx_layers: vec![], spark_layers: (3..91).collect(), spark_ranks: args.peers.len(),
        prefill_lanes: inputs.lanes as u64 }, args.peers.len(), None,
        ModelAdmission::for_clients(ExpertGeometry::STEP5, endpoint, compiled,
            spark_rings(config.max_frame_bytes)?, clients)?)?.select(&args.peers)?;
    cuteafd_transport::worker_selection::require_peer_clients(&args.peers, endpoint, clients)?;
    let available = inputs.inventory.gpus.iter().map(GpuBudget::available).collect::<Vec<_>>();
    let reserves = (0..ids.len()).map(|rank| {
        let native = Execution::serving_admission(&programs, &checkpoint, &admission.ownership, rank, inputs.rows)?;
        let native = native.check(inputs.lanes, u64::MAX, &[])?;
        let total = admission.placement.items[rank].iter().try_fold(0u64, |sum, item| sum.checked_add(item.bytes))
            .context("Step-5 solved allocation overflow")?;
        total.checked_sub(native).context("Step-5 native allocation missing from solved ledger")
    }).collect::<Result<Vec<_>>>()?;
    let head_rank = usize::from(admission.ownership.execution().spec().head_gpu);
    let predictor_budget = if args.mtp > 0 {
        let own = cuteafd_loader::plan::families::step5::coordinator::RankWeightLayout::for_mtp(
            &checkpoint, usize::from(args.mtp))?.resident_bytes.checked_add(
            cuteafd_loader::plan::families::step5::native::mtp_forecast_workspace(
                inputs.rows, usize::from(args.mtp))?).context("Step-5 predictor budget overflow")?;
        let total = admission.placement.items[head_rank].iter().try_fold(0u64, |sum, item| sum.checked_add(item.bytes))
            .context("Step-5 solved predictor allocation overflow")?;
        Some((available[head_rank], total.checked_sub(own).context("Step-5 predictor absent from ledger")?))
    } else { None };
    let caches = devices.iter().enumerate().map(|(rank, &device)| Ok(RefCell::new(CacheRank::new(device,
        admission.cache.clone(), rank, inputs.sequences as u64, admission.marks as u64,
        admission.placement.pool_tokens / 64, admission.reservations[rank].total)?)))
        .collect::<Result<Vec<_>>>()?;
    let stages = (0..inputs.lanes).map(|_| devices.iter().enumerate().map(|(rank, &device)| {
        let cache = caches[rank].borrow();
        Staging::new_for_sequences(device, &programs, &cache,
            Staging::bytes_for_sequences(&admission.cache, rank, inputs.sequences)?, inputs.sequences)
    }).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
    let lane_inputs = (0..inputs.lanes).map(|_| Inputs::allocate(devices[0], inputs.rows, Inputs::bytes(inputs.rows)?))
        .collect::<Result<Vec<_>>>()?;
    let world = SparkWorld::canonical(args.peers.len())?;
    let wave_ranks = admission.ownership.routed_wave_ranks()?;
    let transports = (0..inputs.lanes).map(|_| devices.iter().enumerate().map(|(rank, device)| {
        if wave_ranks.contains(&rank) { device.own(|| SparkLane::new(&library, world,
            args.peers.clone(), args.prefill_rows, config.clone(), 4096 * 2)).map(Some) } else { Ok(None) }
    }).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
    let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == "model.embed_tokens.weight")
        .context("Step-5 embedding header missing")?;
    let source = EmbedSource::new(&args.snapshot, &tensor.shard, &tensor.meta, cfg.hidden)?;
    library.cuda_set_device(args.device)?;
    let (embedding, execution) = TokenEmbedding::load(&library, source, EmbedPlacement::Gpu, ||
        Execution::load_lanes_count(&programs, &checkpoint, admission.ownership,
            caches.iter().collect(), stages.iter().map(|rank| rank.iter().collect()).collect(), inputs.rows,
            available, reserves, transports, 92, 1))?;
    let mtp = predictor_budget.map(|(available, reserved)| super::mtp::Runtime::load(
        &execution, &programs, &checkpoint, &caches[head_rank].borrow(),
        usize::from(args.mtp), inputs.rows, available, &[reserved])).transpose()?;
    let backend = Step5Backend::new(&execution, caches.iter().collect(), &embedding, lane_inputs.iter().collect())?;
    let backend = if let Some(mtp) = &mtp { backend.with_mtp(mtp) } else { backend };
    library.cuda_set_device(args.device)?;
    tracing::info!(rows=inputs.rows, sequences=inputs.sequences, lanes=inputs.lanes, mtp=args.mtp,
        context=backend.geometry().max_context, pool_tokens=admission.placement.pool_tokens,
        "Step-5 admitted eager target and native predictor depth");
    run(&backend, inputs.sequences)
}

fn worker(args: &ServeArgs, receive: &mut tokio::sync::mpsc::Receiver<cuteafd_api::openai::NativeRequest>,
    stats: &Arc<Mutex<serde_json::Value>>, eos: Vec<u32>, initialized: impl FnOnce(usize) -> Result<()>) -> Result<()> {
    with_backend(args, |backend, sequences| {
        let drafts = if args.mtp == 0 { scheduler::Drafts::None }
            else { scheduler::Drafts::Neural { depth: usize::from(args.mtp), fixed: false } };
        scheduler::schedule(backend, &args.snapshot, receive, stats, sequences, drafts,
            eos, args.decode_share, &args.prefix, args.token_io.token_select, &mut text_media(), None,
            || initialized(backend.geometry().max_context))
    })
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    use cuteafd_api::openai::{chat::step5::Step5Encoding, ModelEncoding, ModelProfile, NativeLimits};
    args.validate()?;
    let api = args.api.load()?;
    let snapshot = args.snapshot.clone();
    let encoding = Arc::new(Step5Encoding::from_snapshot(&snapshot)?);
    let eos = encoding.tokens().eos.clone();
    let model = args.model_id.clone().or_else(|| snapshot.ancestors().find_map(|path| {
        let name = path.file_name()?.to_str()?.strip_prefix("models--")?;
        let (organization, model) = name.split_once("--")?;
        Some(format!("{organization}/{model}"))
    })).context("Step-5 requires --model-id for a snapshot without HF identity")?;
    let profile = ModelProfile::new(model, ModelEncoding::Step5(encoding));
    let (queue, mut receive) = tokio::sync::mpsc::channel(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let engine_args = args.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<usize>>();
    let handle = tokio::task::spawn_blocking(move || {
        let mut ready = Some(ready_tx);
        let result = worker(&engine_args, &mut receive, &worker_stats, eos, |context| {
            if let Some(ready) = ready.take() { let _ = ready.send(Ok(context)); }
            Ok(())
        });
        if let Some(ready) = ready.take() {
            let _ = ready.send(Err(anyhow::anyhow!("Step-5 initialization failed: {:#}",
                result.as_ref().err().context("worker exited before readiness")?)));
        }
        result
    });
    let context = ready_rx.await.context("Step-5 worker exited before readiness")??;
    let limits = NativeLimits::new(u32::try_from(context)?, args.max_output)?;
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new(
        "step5", profile.id.clone(), snapshot.clone())));
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

    fn args() -> ServeArgs {
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-step5", "--snapshot", "/m",
            "--native-lib", "/lib.so", "--peers", "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4"]).unwrap();
        let crate::cli::Commands::ServeStep5(args) = cli.command else { panic!("Step-5 dispatch") };
        args
    }

    #[test]
    fn target_serving_defaults_and_invalid_owners() -> Result<()> {
        let mut args = args();
        args.validate()?;
        assert_eq!((args.execution_lanes, args.max_context, args.max_sequences), (2, 1_048_576, 16));
        args.split_device = Some(args.device);
        assert!(args.validate().is_err());
        args.split_device = Some(1);
        args.validate()?;
        args.max_context = 262_143;
        assert!(args.validate().is_err());
        args.max_context = 262_144;
        args.peers[1] = args.peers[0];
        assert!(args.validate().is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires sealed native exports, full target weights and real Spark workers"]
    fn live_mtp_target_abort_restore_and_fresh_marks_are_exact() -> Result<()> {
        use crate::shared::qwen::scheduler::DraftState as _;
        use cuteafd_engine::prefix::{After, PrefixFamily, SnapshotKind};
        use super::super::prefix::Placement;
        let mut args = args();
        args.snapshot = std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?.into();
        args.native_lib = std::env::var("CUTEAFD_NATIVE_LIB")?.into();
        args.manifest = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?.into();
        args.peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?.split(',')
            .map(str::parse).collect::<std::result::Result<Vec<_>, _>>()?;
        args.split_device = Some(1);
        args.prefill_rows = 64;
        args.max_sequences = 3;
        args.mtp = 3;
        args.pool_tokens = Some(262_144);
        args.prefix.prefix_cache_entries = 1;
        args.prefix.prefix_cache_min_tokens = 1;
        let windows: serde_json::Value = serde_json::from_slice(&std::fs::read(
            std::path::Path::new(&std::env::var("CUTEAFD_GOLDEN_FIXTURES")?).join("windows.json"))?)?;
        let tokens = windows["windows"][0]["tokens"].as_array().context("live exactness tokens")?
            .iter().map(|v| v.as_u64().context("live token id").and_then(|n| Ok(u32::try_from(n)?)))
            .collect::<Result<Vec<_>>>()?;
        ensure!(tokens.len() > 577, "live exactness needs at least 578 real tokens");
        with_backend(&args, |backend, _| {
            backend.draft_geometry(&args.snapshot, 3)?;
            let (family, mut pool) = backend.prefix_cache(&args.prefix, 3)?;
            let prefill = |placement: &mut Placement, state: &mut super::super::mtp::DraftState,
                start: usize, end: usize, pool: &cuteafd_engine::prefix::RefPagePool| -> Result<()> {
                for at in (start..end).step_by(64) {
                    let last = (at + 64).min(end);
                    backend.prefill_device(placement, &tokens[at..last], 0, pool)?;
                    backend.prefill_draft(placement, at, &tokens[at..last], tokens.get(last).copied(), state)?;
                }
                Ok(())
            };
            for frontier in [63usize, 64, 65, 575, 576] {
                let mut source = pool.admit_cold(&family, frontier, frontier + 4,
                    |pages| Placement { pages, sequence: 0, committed: 0 })?.placement;
                let mut source_draft = super::super::mtp::DraftState::default();
                backend.bind_request(&mut source, (), cuteafd_engine::media::RequestMedia::new(Vec::new(), 4096, frontier)?,
                    &tokens, 0);
                prefill(&mut source, &mut source_draft, 0, frontier, pool.pool())?;
                source_draft.close(tokens[frontier]);
                let expected = backend.diagnostic_state(&source)?;
                ensure!(pool.capture(&family, SnapshotKind::Prompt, &tokens[..frontier], &source, After::default())?,
                    "live exactness snapshot not retained");
                let mut cold = pool.admit_cold(&family, frontier, frontier + 4,
                    |pages| Placement { pages, sequence: 1, committed: 0 })?.placement;
                let mut cold_draft = super::super::mtp::DraftState::default();
                backend.bind_request(&mut cold, (), cuteafd_engine::media::RequestMedia::new(Vec::new(), 4096, frontier)?,
                    &tokens, 0);
                prefill(&mut cold, &mut cold_draft, 0, frontier, pool.pool())?;
                ensure!(expected == backend.diagnostic_state(&cold)?, "live fresh mark differs at {frontier}");
                let restored = pool.admit(&family, &tokens[..frontier + 1], frontier + 4, false,
                    |pages| Placement { pages, sequence: 2, committed: 0 })?;
                ensure!(restored.resume == frontier, "live exact prefix miss at {frontier}");
                let mut restored = restored.placement;
                let mut restored_draft = super::super::mtp::DraftState::default();
                backend.bind_request(&mut restored, (), cuteafd_engine::media::RequestMedia::new(Vec::new(), 4096, frontier)?,
                    &tokens, frontier);
                ensure!(expected == backend.diagnostic_state(&restored)?, "live restore mark differs at {frontier}");
                // Each independent continuation gets a private append lease,
                // exactly as the scheduler's page ownership requires.
                for placement in [&mut source, &mut cold, &mut restored] {
                    let old = std::mem::take(&mut placement.pages);
                    placement.pages = pool.fork_step(&family, &old, frontier, frontier + 4)?;
                    pool.release(&family, &old)?;
                }
                let drafts = backend.draft(&mut [scheduler::DraftSeq { placement: &source,
                    seq: &mut source_draft, depth: 3 }], &mut scheduler::DraftTiming::default())?.remove(0);
                ensure!(drafts.len() == 3, "live MTP did not propose every admitted stage");
                ensure!(expected == backend.diagnostic_state(&source)?, "draft mutated committed mark at {frontier}");
                let rows = std::iter::once(tokens[frontier]).chain(drafts).collect::<Vec<_>>();
                backend.diagnostic_abort_once();
                ensure!(backend.verify_device(&mut [(&mut source, &rows)], true, pool.pool()).is_err(),
                    "live abort injection did not fire");
                ensure!(expected == backend.diagnostic_state(&source)?, "aborted target changed mark at {frontier}");
                backend.verify_device(&mut [(&mut source, &rows)], true, pool.pool())?;
                backend.accept(&mut [scheduler::Verified { placement: &mut source, seq: &mut source_draft,
                    start: frontier, history: (), rows: &rows, first_row: 0,
                    kept: Some((1, tokens[frontier + 1])), finished: true }], true, true)?;
                for (placement, draft) in [(&mut cold, &mut cold_draft), (&mut restored, &mut restored_draft)] {
                    backend.verify_device(&mut [(placement, &rows)], true, pool.pool())?;
                    backend.accept(&mut [scheduler::Verified { placement, seq: draft, start: frontier,
                        history: (), rows: &rows, first_row: 0, kept: Some((1, tokens[frontier + 1])),
                        finished: true }], true, true)?;
                }
                let continued = backend.diagnostic_state(&source)?;
                ensure!(continued == backend.diagnostic_state(&cold)?, "live replay/fresh continuation differs at {frontier}");
                ensure!(continued == backend.diagnostic_state(&restored)?, "live restored continuation differs at {frontier}");
                backend.require_live()?;
                println!("Step-5 live exactness frontier={frontier} target_kv_csa=real predictor_stages=3 abort_replay=true restore=true fresh_marks=true byte_exact=true");
                for placement in [&source, &cold, &restored] { pool.release(&family, &placement.pages)?; }
                pool.clear(&family)?;
            }
            family.drain().map_err(|e| anyhow::anyhow!("live exactness drain: {e}"))?;
            Ok(())
        })
    }

    #[test]
    fn installed_capacity_is_not_live_rows_or_assumed_fp8_format() -> Result<()> {
        let mut metadata = serde_json::json!({"layouts":{"tp4":{"hidden":4096,"experts":352,
            "top_k":8,"intermediate":1536,"capacities":[{"capacity":1},{"capacity":1024}]}}});
        assert_eq!(compiled_capacity(&metadata, 256)?, 1024);
        assert!(compiled_capacity(&metadata, 2048).is_err());
        metadata["layouts"]["tp6"] = metadata["layouts"]["tp4"].clone();
        metadata["layouts"]["tp6"]["capacities"][1]["capacity"] = 4096.into();
        assert!(compiled_capacity(&metadata, 256).is_err());
        assert!(compiled_capacity(&serde_json::json!({"layouts":{}}), 1).is_err());
        Ok(())
    }
}
