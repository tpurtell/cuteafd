pub(crate) mod prefill_target;
use prefill_target::PrefillTarget;
use speculative::DraftChain;
pub(crate) mod speculative;
pub(crate) mod scheduler;
pub(crate) mod console;
mod distributed;
mod placement;
pub(crate) mod scores;
use scores::TokenScores;
mod copy_drafts;
pub(crate) mod prefix;
pub(crate) mod memory;
use crate::families::deepseek_v41::v41_backbone_cache::BackboneCache;
use crate::families::deepseek_v41::v41_backbone_execution::BackboneExecution;
use crate::families::deepseek_v41::v41_backbone_execution::CacheProducerWeights;
use crate::families::deepseek_v41::v41_backbone_lane::BackboneLane;
use crate::families::deepseek_v41::v41_backbone_lane::BackboneLaneWeights;
use crate::families::deepseek_v41::v41_engram::{
    layer::{EngramGate, EngramLayerWeights},
    EngramDeviceRows,
};
use crate::families::deepseek_v41::v41_experts::coordinator::NativeTp4Wave;
use crate::families::deepseek_v41::v41_index_lane::IndexLane;
use crate::families::deepseek_v41::v41_index_lane::IndexLaneWeights;
use crate::families::deepseek_v41::v41_requests::{RequestTokens, Requests};
use crate::families::deepseek_v41::v41_target_embedding::TargetEmbeddingWave;
use crate::families::deepseek_v41::v41_target_head::{TargetHeadWave, TargetHeadWeights};
use crate::families::deepseek_v41::v41_target_pass::TargetPass;
use crate::families::deepseek_v41::v41_tensors::{NativeRtxTensors, VocabularyHead};
use anyhow::Context;
use anyhow::{ensure, Result};
use cuteafd_api::openai::{InferenceChunk, InferenceFinishReason, NativeRequest, PromptUsage};
use cuteafd_ffi::NativeLibrary;
use cuteafd_transport::expert::SparkExperts;
use cuteafd_transport::{ExpertV2SourceKind, TcpTransportConfig};
use speculative::DraftRuntime;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

pub(crate) async fn run(args: crate::cli::NativeServeArgs) -> Result<()> {
    let Started { send, readiness, worker_thread, stats, http } = start(args, true)?;
    let Http { api, listen, snapshot, limits, http_queue_wait, console_hub } = http.expect("serving start loads the API");
    let vision_health = readiness
        .await
        .context("native target startup stopped")?
        .map_err(anyhow::Error::msg)?;
    cuteafd_bench::context::phase("engine loaded");
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    cuteafd_bench::ready(&listener);
    tracing::info!(%listen,"native V4.1 target API ready");
    let mut profile = cuteafd_api::openai::ModelProfile::default();
    profile.vision_health = vision_health;
    let router = cuteafd_api::openai::router_for_model(send, limits, stats, http_queue_wait, console_hub.clone(),
        api.serve(profile, &snapshot)?);
    axum::serve(listener, api.app(router, console_hub).into_make_service_with_connect_info::<std::net::SocketAddr>())
        .with_graceful_shutdown(async {
            let mut term =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("install SIGTERM handler");
            tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
        })
        .await?;
    tokio::task::spawn_blocking(move || worker_thread.join())
        .await?
        .map_err(|_| anyhow::anyhow!("native CUDA worker panicked during shutdown"))?;
    Ok(())
}

type Readiness = std::result::Result<Option<std::sync::Arc<std::sync::atomic::AtomicBool>>, String>;
type Stats = std::sync::Arc<std::sync::Mutex<serde_json::Value>>;

/// What the HTTP API needs, loaded in the launch's original order.
struct Http {
    api: crate::shared::api::ApiPolicy,
    listen: String,
    /// Tokenizer snapshot for the gateway's exact `count_tokens`.
    snapshot: std::path::PathBuf,
    limits: cuteafd_api::openai::NativeLimits,
    http_queue_wait: Duration,
    console_hub: std::sync::Arc<cuteafd_api::openai::ConsoleHub>,
}

/// The CUDA worker thread, its request queue and its startup readiness.
struct Started {
    send: mpsc::Sender<NativeRequest>,
    readiness: oneshot::Receiver<Readiness>,
    worker_thread: std::thread::JoinHandle<()>,
    stats: Stats,
    http: Option<Http>,
}

/// A worker serving requests from a queue, without the HTTP API (the golden).
/// Dropping the sender drains the scheduler; `join` waits for it to stop.
pub(crate) struct WorkerHandle {
    worker_thread: std::thread::JoinHandle<()>,
}
impl WorkerHandle {
    pub(crate) fn join(self) -> Result<()> {
        self.worker_thread.join().map_err(|_| anyhow::anyhow!("native CUDA worker panicked"))
    }
}

/// Start the worker without the API and wait until it takes requests.
pub(crate) fn start_worker(args: crate::cli::NativeServeArgs)
    -> Result<(mpsc::Sender<NativeRequest>, WorkerHandle)> {
    let Started { send, readiness, worker_thread, .. } = start(args, false)?;
    readiness.blocking_recv().context("native target startup stopped")?.map_err(anyhow::Error::msg)?;
    Ok((send, WorkerHandle { worker_thread }))
}

/// Validate and normalize the launch, then spawn the CUDA worker thread.
/// With `http`, the API and live console load where `serve-native` always loaded them.
fn start(mut args: crate::cli::NativeServeArgs, http: bool) -> Result<Started> {
    match cuteafd_transport::fabric::discover() {
        Ok(report) => tracing::info!(target: "cuteafd::fabric", rails = report.rails.use_rails, "{}", report.summary()),
        Err(error) => tracing::warn!(target: "cuteafd::fabric", "fabric discovery failed: {error:#}"),
    }
    let topology = crate::shared::spark_topology::resolve(
        args.spark_tp,
        args.spark_ep,
        args.peers.len(),
        "serve-native",
    )?;
    // The legacy compact profile is the non-topology, EXL3 path with either two
    // or three Spark peers: one RTX, a hard 32 GiB device ceiling and no
    // paired TP4 artifacts. An explicit topology carries its own rank count and
    // is never compact.
    let compact = legacy_compact(topology, args.peers.len());
    ensure!(
        topology.is_some() || matches!(args.peers.len(), 2 | 3 | 4),
        "two, three or four Spark peers required for the legacy non-topology layout; \
         an explicit --spark-tp/--spark-ep topology carries its own rank count"
    );
    ensure!(
        args.peers.len() == 4 || topology.is_some() || (args.rtx_gpus == 1 && !args.exl3_paired_tp4),
        "two or three Spark peers require the single-RTX, non-paired EXL3 profile"
    );
    if compact {
        ensure!(args.pool_tokens.is_none(), "planner pool admission is not qualified for the compact 32 GiB profile");
        compact_budget(&mut args.memory_reservation, &mut args.kv_pool_size)?;
        if args.prefill_batch_tokens > 256 {
            tracing::info!(requested=args.prefill_batch_tokens, effective=256,
                "compact 32 GiB profile limits prefill workspace capacity");
            args.prefill_batch_tokens = 256;
        }
    }
    if args.pool_tokens == Some(0) && args.memory_reservation.is_none() {
        args.memory_reservation = Some("97%".parse()?);
    }
    args.host_cache_config()?;
    // The golden runs the worker without the API, so it loads no keyed policy.
    let api = if http { Some(args.api.load()?) } else { None };
    let listen = args.listen.clone();
    let snapshot = args.snapshot.clone();
    let limits = cuteafd_api::openai::NativeLimits::new(args.max_context_tokens, args.max_output_tokens)?;
    let (send, receive) = mpsc::channel(args.http_queue_depth.unwrap_or(args.concurrency) as usize);
    let http_queue_wait = Duration::from_millis(args.http_queue_wait_ms);
    let (ready, readiness) = oneshot::channel();
    let stats = std::sync::Arc::new(std::sync::Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let http = match api {
        Some(api) => {
            let console_hub = cuteafd_api::openai::ConsoleHub::new(args.console_text);
            console::install(console_hub.clone(), console::layout(&args))?;
            Some(Http { api, listen, snapshot, limits, http_queue_wait, console_hub })
        }
        None => None,
    };
    let worker_thread = std::thread::Builder::new()
        .name("v41-target-cuda".into())
        .spawn(move || {
            let mut ready = Some(ready);
            let result = crate::shared::api::catch_scheduler_panic(|| worker(args, receive, &mut ready, worker_stats));
            if let Some(ready) = ready.take() {
                let _ = ready.send(Err(result
                    .as_ref()
                    .err()
                    .map(|e| format!("{e:#}"))
                    .unwrap_or_else(|| "worker stopped during startup".into())));
            }
            let reason = result.err().map(|error| format!("scheduler stopped: {error:#}"))
                .unwrap_or_else(|| "scheduler stopped".into());
            tracing::error!(%reason, "native target worker stopped");
            cuteafd_transport::health::record_failure(reason);
        })?;
    Ok(Started { send, readiness, worker_thread, stats, http })
}
// Reserve a supported AOT capacity once; live prefill chunks retain the user's
// requested size. All backbone/draft workspaces and transport share this bound.
fn prefill_capacity(batch_tokens: u32) -> Result<u32> {
    cuteafd_core::coordinator_programs::v41_aot_rows(batch_tokens)
        .context("prefill batch must be in 80..=4096")
}

fn admit_small_card_capacity(total: usize, batch: u32) -> Result<()> {
    ensure!(total > 32usize << 30 || prefill_capacity(batch)? <= 1024,
        "32 GB V4.1 capacity 4096 cannot fit with vision/dSpark; set --prefill-batch-tokens 1024 (256 for pool-first)");
    Ok(())
}

#[cfg(test)]
mod prefill_capacity_tests {
    use super::{legacy_compact, prefill_capacity, compact_budget, memory};

    /// One predicate drives the whole compact profile: the 32 GiB absolute
    /// ceiling, the prefill workspace clamp, the reservation headroom cap and the
    /// final residency check. It must cover the implicit two- and three-rank EXL3
    /// groups and nothing else — an explicit TP3EP1 native launch is not compact.
    #[test]
    fn compact_profile_covers_implicit_two_and_three_peers_only() {
        let tp3ep1 = cuteafd_transport::expert::SparkTopology::new(3, 1).unwrap();
        for (topology, peers, expected) in [
            (None, 2, true),
            (None, 3, true),
            (None, 4, false),
            (None, 6, false),
            (Some(tp3ep1), 3, false),
            (Some(tp3ep1), 2, false),
        ] {
            assert_eq!(
                legacy_compact(topology, peers),
                expected,
                "topology {topology:?} with {peers} Spark peers"
            );
        }
    }

    #[test]
    fn compact_budget_is_absolute_including_headroom() {
        let (mut reservation, mut kv) = (None, None);
        compact_budget(&mut reservation, &mut kv).unwrap();
        assert!(matches!(reservation, Some(memory::Reservation::Bytes(memory::ByteSize(n))) if n == 32usize << 30));
        assert_eq!(kv.unwrap().0, 2usize << 30);
        for invalid in ["33GiB", "100%", "32%"] {
            assert!(compact_budget(&mut Some(invalid.parse().unwrap()), &mut None).is_err());
        }
        let mut kv = Some(memory::ByteSize(1usize << 30));
        compact_budget(&mut Some("31GiB".parse().unwrap()), &mut kv).unwrap();
        assert_eq!(kv.unwrap().0, 1usize << 30);
    }

    #[test]
    fn small_card_refuses_large_capacity_before_allocation() {
        for batch in [80, 256, 1024] { assert!(super::admit_small_card_capacity(32usize << 30, batch).is_ok()); }
        for total in [32usize << 30, (31.8 * (1u64 << 30) as f64) as usize] {
            for batch in [1025, 2048, 4096] {
                let reason = super::admit_small_card_capacity(total, batch).unwrap_err().to_string();
                assert!(reason.contains("capacity 4096"), "{reason}");
                assert!(reason.contains("--prefill-batch-tokens 1024 (256 for pool-first)"), "{reason}");
            }
        }
        assert!(super::admit_small_card_capacity(96usize << 30, 4096).is_ok());
    }

    #[test]
    fn intermediate_batches_use_covering_preallocated_capacity() {
        for (batch, expected) in [
            (80, 256),
            (81, 256),
            (256, 256),
            (257, 1024),
            (1024, 1024),
            (1025, 4096),
            (2048, 4096),
            (4096, 4096),
        ] {
            assert_eq!(prefill_capacity(batch).unwrap(), expected);
        }
        for invalid in [0, 79, 4097, u32::MAX] {
            assert!(prefill_capacity(invalid).is_err());
        }
    }
}

// Compact is a real total-device ceiling, including the planner's 2 GiB
// runtime headroom. Bound KV explicitly so it cannot consume the entire
// ceiling before bottom-up expert placement runs.
fn compact_budget(reservation: &mut Option<memory::Reservation>, kv: &mut Option<memory::ByteSize>) -> Result<()> {
    let ceiling = 32usize << 30;
    match reservation {
        None => *reservation = Some(memory::Reservation::Bytes(memory::ByteSize(ceiling))),
        Some(memory::Reservation::Bytes(bytes)) => ensure!(bytes.0 <= ceiling, "compact EXL3 requires a total device budget at most 32 GiB including headroom"),
        Some(memory::Reservation::Percent(_)) => anyhow::bail!("compact EXL3 requires an absolute device budget at most 32 GiB, not a percentage"),
    }
    if kv.is_none() { *kv = Some(memory::ByteSize(2usize << 30)); }
    Ok(())
}

/// Protocol-v2 per-chunk transport diagnostics. Resolved once here so the RDMA
/// progress/poll path never reaches the process environment; the value is only
/// meaningful for a launch, so a start-time read is sufficient.
pub(crate) fn protocol_v2_timing() -> bool {
    cuteafd_transport::protocol_v2_timing_from_env()
}

/// True for the legacy compact profile: the non-topology launch with an
/// implicit two- or three-rank EXL3 Spark group. Kept in one place because the
/// 32 GiB device ceiling, the prefill workspace clamp and the reservation
/// headroom cap all belong to exactly this profile; an explicit topology (any
/// native layout, including `TP3EP1`) is never compact.
fn legacy_compact(topology: Option<cuteafd_transport::expert::SparkTopology>, peers: usize) -> bool {
    topology.is_none() && matches!(peers, 2 | 3)
}

fn spark_transport(
    peers: &[std::net::SocketAddr],
    capacity: u32,
    timing: bool,
    topology: Option<cuteafd_transport::expert::SparkTopology>,
) -> Result<SparkExperts> {
    let config = TcpTransportConfig { timing, timeout: Duration::from_secs(120), max_frame_bytes: 64 * 1024 * 1024 };
    if let Some(topology) = topology {
        // Topology-bound transport: canonical executor ids come from the shared
        // topology and requests must carry the native ownership contract.
        return SparkExperts::new_topology(topology, peers, capacity, config);
    }
    match peers.len() {
        2 => SparkExperts::new_tp2(peers.try_into().expect("two peers"), [
            cuteafd_transport::expert::v41_spark_executor_id(2, 0)?,
            cuteafd_transport::expert::v41_spark_executor_id(2, 1)?,
        ], capacity, config),
        // The implicit three-rank EXL3 group answers in the TP3EP1 namespace
        // (7..=9) but keeps the canonical non-ownership frame contract, so it
        // must not be built through `new_topology`.
        3 => SparkExperts::new_ranks(peers, &[
            cuteafd_transport::expert::v41_spark_executor_id(3, 0)?,
            cuteafd_transport::expert::v41_spark_executor_id(3, 1)?,
            cuteafd_transport::expert::v41_spark_executor_id(3, 2)?,
        ], capacity, config),
        4 => SparkExperts::new(peers.try_into().expect("four peers"), [1, 2, 3, 4], capacity, config),
        _ => anyhow::bail!(
            "the legacy non-topology transport takes two, three or four Spark peers; \
             a six-rank layout (pure TP6EP1 or replicated) must pass --spark-tp/--spark-ep"
        ),
    }
}

fn worker(
    mut args: crate::cli::NativeServeArgs,
    mut receive: mpsc::Receiver<NativeRequest>,
    ready: &mut Option<oneshot::Sender<std::result::Result<Option<std::sync::Arc<std::sync::atomic::AtomicBool>>, String>>>,
    stats: std::sync::Arc<std::sync::Mutex<serde_json::Value>>,
) -> Result<()> {
    // Auto placement must publish its live boundary even when no local layer
    // fits. Explicit all-remote single-RTX launches do not need a handoff.
    if let Some(directory) = args.placement_directory.as_deref() {
        ensure!(
            args.rtx_gpus == 2 || args.rtx_expert_layers != memory::LocalLayers::Count(0),
            "placement handoff requires --rtx-gpus 2 or single-RTX auto/local expert placement"
        );
        // The handoff directory must name a real path, not an empty string.
        ensure!(
            !directory.as_os_str().is_empty(),
            "placement handoff requires a non-empty directory"
        );
    }
    ensure!(!args.tp2_dspark_experts || (args.rtx_gpus==2 && args.dspark),
        "--tp2-dspark-experts requires --rtx-gpus 2 and --dspark");
    ensure!(!args.tp2_output_projection || args.rtx_gpus==2,"--tp2-output-projection requires --rtx-gpus 2");
    ensure!(!args.tp2_query_projection || args.rtx_gpus==2,"--tp2-query-projection requires --rtx-gpus 2");
    ensure!(!args.tp2_attention || args.rtx_gpus==2,"--tp2-attention requires --rtx-gpus 2");
    if args.rtx_gpus == 2 {
        // Probe before the distributed loader reserves either device's weights.
        // SAFETY: the configured native library is trusted and remains live for the probe.
        let library = unsafe { NativeLibrary::load(&args.native_lib) }?;
        if crate::shared::peer_split::probed_device(&library, 0, Some(1))?.is_none() {
            args.rtx_gpus = 1;
            args.tp2_dspark_experts = false;
            args.tp2_output_projection = false;
            args.tp2_query_projection = false;
            args.tp2_attention = false;
        } else {
            return distributed::worker(args, receive, ready, stats);
        }
    }
    // SAFETY: the configured library remains owned by this worker until all CUDA work drains.
    let lib = unsafe { NativeLibrary::load(&args.native_lib)? };
    let (_, device_total) = lib.cuda_memory_info()?;
    let small_card = device_total <= 32usize << 30;
    if small_card {
        let explicit = std::env::var("CUTEAFD_V41_FIXED_GRAPH_ROWS").ok();
        if let Some(shapes) = super::graph_policy::profile_fixed_shapes(device_total, explicit.as_deref())? {
            tracing::info!(?shapes, "V4.1 small-card fixed exact graph set; other rows execute eagerly");
            super::graph_policy::set_fixed_shapes(shapes)?;
        }
    }
    if small_card && !legacy_compact(None, args.peers.len()) {
        admit_small_card_capacity(device_total, args.prefill_batch_tokens)?;
        if args.rtx_expert_layers == memory::LocalLayers::Auto { args.rtx_expert_layers = memory::LocalLayers::Count(0); }
        tracing::info!(device_total, prefill=args.prefill_batch_tokens, "selected V4.1 32 GB all-remote profile; explicit pool/reservation overrides retained");
    }
    // Local expert waves need an exported AOT capacity; every other row
    // buffer follows the live prefill chunk (as the dual-RTX path does: the
    // FP8 plans keep their full scratch). 2048-row chunks: ~9 GiB less.
    let aot_capacity = prefill_capacity(args.prefill_batch_tokens)?;
    let capacity = cuteafd_core::coordinator_programs::v41_live_rows(args.prefill_batch_tokens)
        .context("invalid V4.1 prefill batch")?;
    let rows = capacity as usize;
    let catalog = cuteafd_loader::read_official_v41_catalog(
        cuteafd_loader::OFFICIAL_V41_MODEL_ID,
        &args.snapshot,
    )?;
    let topology = crate::shared::spark_topology::resolve(
        args.spark_tp,
        args.spark_ep,
        args.peers.len(),
        "serve-native",
    )?;
    // Explicit replicated groups are native-only and are rejected here, before
    // any expert weight is allocated or readiness published.
    crate::shared::spark_topology::require_native(topology, &catalog)?;
    if let Some(topology) = topology {
        // Fail before any CUDA allocation or readiness publication when the
        // library cannot reduce this physical-rank count.
        lib.v41_compact_reducer()?
            .require_rank_count(topology.world_size() as u32)?;
    } else if args.peers.len() == 3 {
        // The implicit three-rank EXL3 group reduces through the N-plane entry,
        // so it gets the same fail-fast rule. The two-rank group keeps using its
        // historical pairwise reducer path untouched.
        lib.v41_compact_reducer()?.require_rank_count(3)?;
    }
    ensure!(
        args.peers.len() == 4 || topology.is_some() || catalog.exl3().is_some(),
        "an implicit two- or three-peer Spark group requires an EXL3 checkpoint; \
         a native three-rank group must pass --spark-tp 3 --spark-ep 1"
    );
    let paired_profile = crate::families::deepseek_v41::v41_experts::paired::PairedProfile::for_serving(&catalog, args.exl3_paired_tp4)?;
    if small_card && !legacy_compact(topology, args.peers.len()) {
        memory::admit_small_card_startup(&lib, &catalog, &args)?;
    }
    let start = Instant::now();
    cuteafd_ffi::memory_ledger::relabel_other("v41/startup");
    if small_card { memory::startup_phase(&lib, "v41/startup", device_total)?; }
    let weights = BackboneLaneWeights::load(
        &lib,
        &catalog,
        BackboneLaneWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let producers = CacheProducerWeights::load(
        &lib,
        &catalog,
        CacheProducerWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let index_weights = IndexLaneWeights::load(
        &lib,
        &catalog,
        IndexLaneWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let table = NativeRtxTensors::load_embedding(&lib, &catalog,
        memory::embedding_placement(args.embedding_placement, device_total))?;
    eprintln!(
        "native target backbone/index/embedding weights loaded in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let embedding =
        TargetEmbeddingWave::new(&lib, &table, rows, TargetEmbeddingWave::device_bytes(rows)?)?;
    cuteafd_ffi::memory_ledger::relabel_other("v41/weights:backbone,cache-producers,index,embedding");
    if small_card { memory::startup_phase(&lib, "v41/weights:backbone,cache-producers,index,embedding", device_total)?; }
    let lane = BackboneLane::new(
        &weights,
        capacity,
        BackboneLane::workspace_bytes(&lib, capacity)?.into_iter().sum(),
    )?;
    let index = IndexLane::new(
        &index_weights,
        capacity,
        IndexLane::workspace_bytes(&lib, capacity)?.into_iter().sum(),
    )?;
    let execution = BackboneExecution::new(
        &producers,
        capacity,
        BackboneExecution::workspace_bytes(&lib, capacity)?,
    )?;
    cuteafd_ffi::memory_ledger::relabel_other("v41/workspace:backbone-lanes");
    if small_card { memory::startup_phase(&lib, "v41/workspace:backbone-lanes", device_total)?; }
    let map = cuteafd_loader::EngramTokenMap::from_file(&&args.snapshot.join("tokenizer.json"))?;
    // Both Engram layers retain their staging leases until consumed. Reserve
    // both layers for both active lanes so the second lane can gather early.
    let gather_slots = 2 * cuteafd_core::ENGRAM_LAYERS.len();
    let pipeline =
        unsafe { cuteafd_loader::EngramPipeline::new(&catalog, map, rows, gather_slots, rows * 64 * 1024)? };
    let engram_weights = [0, 1]
        .map(|i| EngramLayerWeights::load(&lib, &catalog, i, 256 * 1024 * 1024, 16 * 1024 * 1024))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let gates = [
        EngramGate::new(&engram_weights[0], rows, 1024 * 1024 * 1024)?,
        EngramGate::new(&engram_weights[1], rows, 1024 * 1024 * 1024)?,
    ];
    let upload = EngramDeviceRows::new(&lib, rows, EngramDeviceRows::device_bytes(rows)?)?;
    cuteafd_ffi::memory_ledger::relabel_other("v41/engram");
    if small_card { memory::startup_phase(&lib, "v41/engram", device_total)?; }
    let vocabulary = VocabularyHead::load(
        &lib,
        &catalog,
        VocabularyHead::plan(&catalog)?,
        16 * 1024 * 1024,
    )?;
    let head_weights = TargetHeadWeights::load(
        &lib,
        &catalog,
        TargetHeadWeights::device_bytes(&catalog)?,
        16 * 1024 * 1024,
    )?;
    let head = head_weights.wave(&vocabulary, if args.dspark_draft_limit > 5 { 64 } else { 48 }, TargetHeadWave::device_bytes(if args.dspark_draft_limit > 5 { 64 } else { 48 })?)?;
    cuteafd_ffi::memory_ledger::relabel_other("v41/weights:head");
    if small_card { memory::startup_phase(&lib, "v41/weights:head", device_total)?; }
    let mut pass = TargetPass::new(
        embedding,
        lane,
        index,
        execution,
        upload,
        gates,
        head,
        crate::families::deepseek_v41::v41_target_pass::TargetTapWave::new(
            &lib,
            rows,
            crate::families::deepseek_v41::v41_target_pass::TargetTapWave::device_bytes(rows)?,
        )?,
        Duration::from_secs(120),
    )?;
    cuteafd_ffi::memory_ledger::relabel_other("v41/workspace:target-pass");
    if small_card { memory::startup_phase(&lib, "v41/workspace:target-pass", device_total)?; }
    let protocol_v2_timing = protocol_v2_timing();
    // Every replicated layout reserves its exact physical rank plane count; the
    // legacy 4-rank forecast is no longer reused for six ranks.
    let wave_bytes = NativeTp4Wave::device_bytes_for(capacity, args.peers.len())?;
    let roce = spark_transport(&args.peers, capacity, protocol_v2_timing, topology)?;
    let mut transport = NativeTp4Wave::new(&lib, roce, wave_bytes)?;
    if let Some(profile) = &paired_profile { transport.install_paired(profile.clone())?; }
    cuteafd_ffi::memory_ledger::relabel_other("v41/transport");
    if small_card { memory::startup_phase(&lib, "v41/transport", device_total)?; }
    let mut prefill_pass = TargetPass::new(
        TargetEmbeddingWave::new(&lib, &table, rows, TargetEmbeddingWave::device_bytes(rows)?)?,
        BackboneLane::new(&weights, capacity, BackboneLane::workspace_bytes(&lib, capacity)?.into_iter().sum())?,
        IndexLane::new(&index_weights, capacity, IndexLane::workspace_bytes(&lib, capacity)?.into_iter().sum())?,
        BackboneExecution::new(&producers, capacity, BackboneExecution::workspace_bytes(&lib, capacity)?)?,
        EngramDeviceRows::new(&lib, rows, EngramDeviceRows::device_bytes(rows)?)?,
        [EngramGate::new(&engram_weights[0], rows, 1024 * 1024 * 1024)?,
         EngramGate::new(&engram_weights[1], rows, 1024 * 1024 * 1024)?],
        head_weights.wave(&vocabulary, if args.dspark_draft_limit > 5 { 64 } else { 48 }, TargetHeadWave::device_bytes(if args.dspark_draft_limit > 5 { 64 } else { 48 })?)?,
        crate::families::deepseek_v41::v41_target_pass::TargetTapWave::new(&lib, rows, crate::families::deepseek_v41::v41_target_pass::TargetTapWave::device_bytes(rows)?)?,
        Duration::from_secs(120),
    )?;
    if args.dspark && args.dspark_draft_limit > 5 {
        pass.reserve_sparse_decode_rows(64)?;
        prefill_pass.reserve_sparse_decode_rows(64)?;
    }
    cuteafd_ffi::memory_ledger::relabel_other("v41/workspace:prefill-pass");
    if small_card { memory::startup_phase(&lib, "v41/workspace:prefill-pass", device_total)?; }
    let prefill_roce = spark_transport(&args.peers, capacity, protocol_v2_timing, topology)?;
    let mut prefill_transport = NativeTp4Wave::new(&lib, prefill_roce, wave_bytes)?;
    if let Some(profile) = &paired_profile { prefill_transport.install_paired(profile.clone())?; }
    cuteafd_ffi::memory_ledger::relabel_other("v41/transport");
    if small_card { memory::startup_phase(&lib, "v41/transport", device_total)?; }
    let exl3_tiers: &[usize] = catalog.exl3().map(|m| m.decoder_tiers()).unwrap_or(&[]);
    let draft_free_before = if small_card { lib.cuda_memory_info()?.0 } else { 0 };
    let draft_weights = if args.dspark {
        Some(crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::load_serving_with_width(
            &lib,
            &catalog,
            capacity,
            args.concurrency,
            32 * 1024 * 1024 * 1024,
            16 * 1024 * 1024,
            if args.dspark_draft_limit > 5 { 7 } else { 5 },
            Some(&crate::families::deepseek_v41::v41_experts::exl3::aot_layout_directory(&args.native_lib, exl3_tiers, "dspark")),
        )?)
    } else {
        None
    };
    let mut draft = draft_weights
        .as_ref()
        .map(|weights| DraftRuntime::with_requests(&lib, weights, &table, &vocabulary, capacity, args.concurrency))
        .transpose()?;
    if let Some(draft) = &mut draft {
        draft.set_draft_limit(args.dspark_draft_limit)?;
        draft.set_fixed(args.dspark_fixed);
    }
    cuteafd_ffi::memory_ledger::relabel_other("v41/drafter");
    if small_card { memory::startup_phase(&lib, "v41/drafter", device_total)?; }
    let vision_free_before = if small_card { lib.cuda_memory_info()?.0 } else { 0 };
    let dspark_bytes = draft_free_before.saturating_sub(vision_free_before);
    let mut vision = crate::families::deepseek_v41::v41_vision_encoder::Encoder::load(&args, &catalog)?;
    let vision_bytes = if small_card { vision_free_before.saturating_sub(lib.cuda_memory_info()?.0) } else { 0 };
    // Reserve both retention banks plus one in-flight snapshot per lane. These
    // allocations are counted before choosing KV capacity and local expert layers.
    ensure!(args.prefix_cache_entries <= 128, "invalid retained-turn limit");
    cuteafd_ffi::memory_ledger::relabel_other("v41/vision");
    if small_card { memory::startup_phase(&lib, "v41/vision", device_total)?; }
    let snapshot_slots = if args.prefix_cache_entries == 0 { 0 } else {
        (args.prefix_cache_entries as usize).checked_mul(2).and_then(|n| n.checked_add(2))
            .context("snapshot slot count overflow")?
    };
    let target_prefix_pool = (snapshot_slots > 0).then(|| crate::shared::memory::SnapshotPool::new(
        &lib, crate::families::deepseek_v41::v41_backbone_cache::BackbonePrefix::device_bytes(), snapshot_slots)).transpose()?;
    let draft_snapshot_bytes = draft.as_mut().map(|d| d.reserve_prefixes(snapshot_slots)).transpose()?.unwrap_or(0);
    let snapshot_bytes = target_prefix_pool.as_ref().map_or(0, crate::shared::memory::SnapshotPool::device_bytes) + draft_snapshot_bytes;
    tracing::info!(snapshot_slots, snapshot_bytes, "snapshot arenas reserved before serving");
    cuteafd_ffi::memory_ledger::relabel_other("v41/prefix-snapshots");
    if small_card { memory::startup_phase(&lib, "v41/prefix-snapshots", device_total)?; }
    // Size after vision, both lanes, transports and optional draft allocations are live.
    let (free, total) = if small_card { memory::measured_pool_memory(&lib)? } else { lib.cuda_memory_info()? };
    // A nominal 32 GiB card can expose slightly less memory to CUDA. The
    // compact ceiling may become smaller, never larger, on that hardware.
    // This cap belongs to the legacy two- or three-peer EXL3 compact profile
    // only; an explicit TP3EP1 native topology is not compact.
    let compact = legacy_compact(topology, args.peers.len());
    let reservation = if compact {
        match args.memory_reservation {
            Some(memory::Reservation::Bytes(memory::ByteSize(bytes))) =>
                Some(memory::Reservation::Bytes(memory::ByteSize(bytes.min(total)))),
            other => other,
        }
    } else { args.memory_reservation };
    args.kv_pool_size = memory::planned_pool_size(&args, &[(free, total)])?;
    let pool = memory::PoolPlan::new(args.concurrency as usize, args.max_context_tokens as usize,
        args.prefix_cache_entries as usize, snapshot_bytes, args.kv_pool_size, reservation, free, total)?;
    tracing::info!(retained_turn_limit=args.prefix_cache_entries, prompt_snapshot_limit=args.prefix_cache_entries, source_pages=?pool.pages, global_bytes=pool.global_bytes,
        cache_bytes=pool.cache_bytes, device_occupied_bytes=pool.occupied_before,
        dspark_bytes, vision_bytes, snapshot_bytes,
        pool_tokens=pool.pages[0].saturating_sub(args.concurrency as usize + 2 * args.prefix_cache_entries as usize) * 512,
        projected_free_bytes=free.saturating_sub(pool.cache_bytes),
        reservation_bytes=pool.reservation_bytes, runtime_headroom_bytes=pool.runtime_headroom_bytes,
        "native KV pool reservation");
    let mut requests = Requests::new(&lib, pipeline, args.concurrency as usize, pool.pages, pool.cache_bytes)?;
    cuteafd_ffi::memory_ledger::relabel_other("v41/kv");
    if small_card { memory::startup_phase(&lib, "v41/kv", device_total)?; }
    if let Some(pool) = target_prefix_pool { requests.install_prefix_pool(pool)?; }
    let mut local_layers = 0usize;
    // Published only on the single-RTX path; the 2-RTX distributed worker owns
    // its own handshake. Dropping it without a ready acknowledgement leaves the
    // launch unpublished, which the launcher treats as a failure.
    let mut placement_handoff: Option<placement::StartupPlacement> = None;
    cuteafd_ffi::memory_ledger::relabel_other("v41/kv-prefix-install");
    if small_card { memory::startup_phase(&lib, "v41/kv-prefix-install", device_total)?; }
    if args.rtx_expert_layers != memory::LocalLayers::Count(0) {
        use crate::families::deepseek_v41::v41_experts::{ExpertLayer, ExpertWeights, local::LocalExpertWave};
        let local_started = Instant::now();
        use crate::families::deepseek_v41::v41_experts::exl3::Exl3Weights;
        let exl3_directory = crate::families::deepseek_v41::v41_experts::exl3::aot_layout_directory(&args.native_lib, exl3_tiers, "rtx-tp1");
        let compressed = catalog.exl3().is_some();
        let per_lane = if compressed { LocalExpertWave::exl3_device_bytes(&exl3_directory, aot_capacity)? }
            else {
                LocalExpertWave::device_bytes_for(&lib, aot_capacity, catalog.nvfp4().is_some())?
            };
        let budgets = (0..40).map(|layer| {
            let selection = ExpertLayer::BackboneFull { layer };
            if compressed { Exl3Weights::plan(&catalog, selection) }
            else { ExpertWeights::plan(&lib, &catalog, selection) }
        }).collect::<Result<Vec<_>>>()?;
        let (free, total) = lib.cuda_memory_info()?;
        let free = if args.pool_tokens == Some(0) { free.saturating_sub((3usize << 30) - memory::RUNTIME_HEADROOM) }
            else { free };
        let plan = memory::LocalLayerPlan::new(args.rtx_expert_layers, &budgets,
            per_lane.checked_mul(2).context("local lane budget overflow")?, free, total, pool.reservation_bytes)?;
        local_layers = plan.layers;
        tracing::info!(layers=plan.layers, resident_bytes=plan.resident_bytes,
            workspace_bytes=plan.workspace_bytes, peak_bytes=plan.peak_bytes,
            graph_reserve_bytes=plan.graph_reserve_bytes,
            "bottom-up RTX expert placement");
        // Publish the resolved boundary before anything waits on it. On 1 RTX the
        // 2-RTX distributed path never ran, so without this the launcher cannot
        // know the real first remote layer and must start every worker at 0,
        // over-reserving remote layers that the coordinator already owns.
        // Computing the plan only needs the local device: it does not read the
        // remote transport, connect to a peer, or wait on readiness.
        if let Some(directory) = args.placement_directory.as_deref() {
            placement_handoff = Some(placement::StartupPlacement::publish(
                std::path::Path::new(directory),
                args.rtx_gpus,
                local_layers,
            )?);
        }        if plan.layers > 0 && compressed {
            let mut loaded = Vec::with_capacity(plan.layers);
            for layer in 0..plan.layers {
                loaded.push(Exl3Weights::load(&lib, &catalog, ExpertLayer::BackboneFull { layer },
                    budgets[layer].peak_device_bytes()?)?);
            }
            let weights = std::rc::Rc::new(loaded);
            transport.install_local(unsafe { LocalExpertWave::new_exl3(&lib, weights.clone(),
                &exl3_directory, aot_capacity, per_lane)? })?;
            prefill_transport.install_local(unsafe { LocalExpertWave::new_exl3(&lib, weights,
                &exl3_directory, aot_capacity, per_lane)? })?;
        } else if plan.layers > 0 {
            let mut loaded = Vec::with_capacity(plan.layers);
            for layer in 0..plan.layers {
                loaded.push(ExpertWeights::load(&lib, &catalog, ExpertLayer::BackboneFull { layer },
                    budgets[layer].peak_device_bytes()?)?);
            }
            let weights = std::rc::Rc::new(loaded);
            transport.install_local(LocalExpertWave::new(&lib, weights.clone(), aot_capacity, per_lane)?)?;
            prefill_transport.install_local(LocalExpertWave::new(&lib, weights, aot_capacity, per_lane)?)?;
        }
        tracing::info!(layers=plan.layers, elapsed_ms=local_started.elapsed().as_millis(),
            "local RTX experts ready");
    }
    // The launcher has already started the workers at the published boundary and
    // now acknowledges it; only then may remote transport creation and the first
    // request proceed. The local experts above are installed either way.
    if let Some(handoff) = placement_handoff {
        handoff.wait_ready(Duration::from_secs(900))?;
    }
    if crate::shared::memory::chain::device_exchange_enabled() && paired_profile.is_none() && topology.is_none()
        && catalog.nvfp4().is_none() && catalog.exl3().is_none() && local_layers < 40 {
        // Verification waves (up to 80 rows) on the device-driven exchange; it
        // connects (and warms) the workers, so only once they are serving.
        transport.install_device_link(&args.peers, 80.min(capacity as usize), 39,
            cuteafd_transport::TcpTransportConfig { timing: protocol_v2_timing, timeout: Duration::from_secs(120),
                max_frame_bytes: 64 << 20 })?;
    }
    let (free, total) = lib.cuda_memory_info()?;
    let occupied = total - free;
    if compact {
        ensure!(occupied.checked_add(memory::RUNTIME_HEADROOM).is_some_and(|n| n <= pool.reservation_bytes),
            "compact residency {occupied} bytes plus runtime headroom exceeds {} byte device ceiling", pool.reservation_bytes);
    }
    cuteafd_ffi::memory_ledger::relabel_other("v41/local-experts");
    if small_card { memory::startup_phase(&lib, "v41/local-experts", device_total)?; }
    crate::shared::memory_report::release_load_staging(&lib);
    tracing::info!(rtx_layers=local_layers, first_remote_dispatch_layer=local_layers,
        remote_dispatch_layers=40-local_layers, spark_world=args.peers.len(),
        spark_topology=?topology.map(|t| (t.tp(), t.ep())),
        device_occupied_bytes=occupied, device_budget_bytes=pool.reservation_bytes,
        runtime_headroom_bytes=pool.runtime_headroom_bytes, "native serving residency ready");
    if let Some(draft) = &mut draft { draft.configure_policy(&transport, catalog.nvfp4().is_some())?; }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let prefixes = scheduler::prepare_prefix_cache(&lib, &args, &requests)?;
    scheduler::publish_capacity(&requests, &prefixes);
    vision.connect()?;
    ready
        .take()
        .context("startup readiness missing")?
        .send(Ok(vision.health_handle()))
        .map_err(|_| anyhow::anyhow!("API startup cancelled"))?;
    scheduler::serve(&lib, &args, &runtime, &mut receive, &mut pass, &mut prefill_pass,
        &mut requests, &mut transport, &mut prefill_transport, draft.as_mut(), &mut vision, stats, prefixes)
}

/// The HC-9 prefill pacing hold at a chunk boundary: bounded by the host cache's store
/// pace, counted in its metrics, and never a reason to fail the request.
fn prefill_hold(hold: &mut dyn FnMut() -> Result<()>) {
    if let Err(error) = hold() {
        tracing::warn!(target: "cuteafd::host_cache", %error, "prefill hold failed; continuing");
    }
}

fn prefill<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(
    lib: &'a NativeLibrary,
    runtime: &tokio::runtime::Runtime,
    pass: &mut P,
    other: &mut P,
    requests: &mut Requests<'a>,
    transport: &mut P::Transport,
    other_transport: &mut P::Transport,
    lease: crate::families::deepseek_v41::v41_backbone_cache::CacheLease,
    tokens: &[u32],
    chunk_rows: usize,
    job: &NativeRequest,
    draft: Option<&mut DraftRuntime<'_, 'a, C>>,
    hold: &mut dyn FnMut() -> Result<()>,
) -> Result<TokenScores> {
    use crate::families::deepseek_v41::v41_backbone_cache::{CacheStage, CacheWork};
    let end = tokens.len() as u64;
    let cached = requests.cache().committed_end(lease)? as usize;
    let stage = requests.cache().stage(lease)?;
    let replay = stage == CacheStage::EncoderReplay;
    if cached > 0 && stage == CacheStage::Full {
        return prefill_continuation(
            lib,
            runtime,
            pass,
            requests,
            transport,
            lease,
            &tokens[cached..],
            chunk_rows,
            job,
            draft,
            hold,
        );
    }
    let mut suffix = pass.new_suffix(lib, end)?;
    if replay {
        let start = requests.cache().history_end(lease)? as usize;
        ensure!(
            cached - start <= 128,
            "encoder prefix replay exceeds one window"
        );
        for chunk in tokens[start..cached].chunks(chunk_rows) {
            ensure!(!job.events.is_closed(), "client disconnected");
            prefill_hold(hold);
            let mut batch = requests.prepare(&[RequestTokens {
                lease,
                tokens: chunk,
                image_mask: None,
                kind: ExpertV2SourceKind::Prefill,
            }])?;
            let result = (|| -> Result<()> {
                runtime.block_on(unsafe {
                    pass.encoder_part(requests, &mut batch, transport, &mut suffix)
                })?;
                ensure!(!job.events.is_closed(), "client disconnected");
                runtime.block_on(pass.commit_prefill::<C>(requests, &mut batch, None, chunk.len() as u32))
            })();
            if result.is_err() {
                pass.discard(&mut batch)?;
            }
            result?;
        }
    } else if stage == CacheStage::Full {
        requests.begin_encoder(lease, end)?;
    }
    let mut chunks = tokens[cached..].chunks(chunk_rows);
    // Keep the ordinary path for short prompts; pair full chunks first otherwise.
    if chunks.len() == 1 {
        let chunk = chunks.next().expect("one chunk");
        ensure!(!job.events.is_closed(), "client disconnected");
        prefill_hold(hold);
        let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
            image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
        let started = Instant::now();
        let result = (|| -> Result<()> {
            runtime.block_on(unsafe { pass.encoder_part(requests, &mut batch, transport, &mut suffix) })?;
            ensure!(!job.events.is_closed(), "client disconnected");
            runtime.block_on(pass.commit_prefill::<C>(requests, &mut batch, None, chunk.len() as u32))
        })();
        if result.is_err() { pass.discard(&mut batch)?; }
        result?;
        console::totals::prefill(chunk.len());
        console::Prefill::done(console::PrefillKind::Single, 0, 0, 1, chunk.len(), started);
        tracing::debug!(target: "cuteafd::timing", rows=chunk.len(), total_us=started.elapsed().as_micros() as u64, "target encoder step");
    }
    if chunks.len() != 0 {
        let chunks: Vec<_> = chunks.collect();
        let started = Instant::now();
        // The hold must reach every chunk the stream dispatches; both lanes share one
        // `&dyn Fn` hook, so the mutable callback goes through a RefCell (calls are
        // synchronous, single-threaded, never nested).
        let hold = std::cell::RefCell::new(&mut *hold);
        let before_chunk = || -> Result<()> {
            prefill_hold(&mut *hold.borrow_mut());
            Ok(())
        };
        runtime.block_on(unsafe { pass.execute_encoder_stream_held(other, requests, lease, &chunks,
            [transport, other_transport], &mut suffix, &|| !job.events.is_closed(), &before_chunk) })?;
        tracing::debug!(target: "cuteafd::timing", rows=tokens.len(),
            total_us=started.elapsed().as_micros() as u64, "target encoder stream");
    }
    ensure!(!job.events.is_closed(), "client disconnected");
    let start = requests.begin_decoder_replay(lease)?;
    let rows = (end - start) as u32;
    let mut batch = requests.prepare_replay(&[CacheWork { lease, tokens: rows, kind: ExpertV2SourceKind::Prefill }])?;
    let started = Instant::now();
    let result = (|| -> Result<TokenScores> {
        let bytes = runtime.block_on(unsafe { pass.prefill_logits(lib, requests, &mut batch,
            transport, &[rows as usize - 1], Some(&suffix)) })?;
        let scores = TokenScores::new(scores::VOCAB, bytes)?;
        ensure!(!job.events.is_closed(), "client disconnected");
        runtime.block_on(pass.commit_prefill(requests, &mut batch, draft, rows))?;
        Ok(scores)
    })();
    if result.is_err() { pass.discard(&mut batch)?; }
    console::Prefill::done(console::PrefillKind::Replay, 0, 0, 1, rows as usize, started);
    tracing::debug!(target: "cuteafd::timing", rows, total_us=started.elapsed().as_micros() as u64, "target decoder replay");
    result
}

/// A benchmark probe's teacher-forced scoring pass finished: the request
/// already sent its Finish and is released without a failure.
#[derive(Debug, thiserror::Error)]
#[error("teacher-forced scoring finished")]
pub(crate) struct ScoringDone;

/// Rows per teacher-forced scoring chunk: the smallest target-head wave capacity.
const SCORING_ROWS: usize = 48;

#[derive(Debug)]
struct ScoringPlan {
    path: crate::shared::probe::ScorePath,
    rows: usize,
}

impl ScoringPlan {
    fn new(spec: &cuteafd_api::openai::probe::ProbeSpec) -> Result<Self> {
        use crate::shared::probe::ScorePath;
        let path = ScorePath::parse(spec.score_path.as_deref(), ScorePath::Prefill)?;
        let rows = match path {
            ScorePath::Prefill => {
                ensure!(spec.verify_rows.is_none(), "probe verify_rows requires score_path=decode for deepseek_v41");
                SCORING_ROWS
            }
            ScorePath::Decode => {
                let rows = spec.verify_rows.unwrap_or(8);
                ensure!((1..=SCORING_ROWS).contains(&rows),
                    "unsupported probe verify_rows={rows} for deepseek_v41; supports 1..={SCORING_ROWS}");
                rows
            }
        };
        Ok(Self { path, rows })
    }

    fn steps(&self, from: usize, len: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
        let rows = self.rows;
        (from..len - 1).step_by(rows).map(move |start| start..(start + rows).min(len - 1))
    }

    fn source_kind(&self, rows: usize) -> ExpertV2SourceKind {
        use crate::shared::probe::ScorePath;
        match self.path {
            ScorePath::Prefill => ExpertV2SourceKind::Prefill,
            ScorePath::Decode if rows == 1 => ExpertV2SourceKind::Decode,
            ScorePath::Decode => ExpertV2SourceKind::MtpVerify,
        }
    }
}

/// Teacher-forced scoring: prefill the prefix, then record rows from the
/// explicitly selected decode-shaped verification or legacy prefill-shaped
/// continuation. The first row comes from prefix prefill in either path.
/// Returns [`ScoringDone`] when done.
#[allow(clippy::too_many_arguments)]
fn score<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    pass: &mut P, other: &mut P, requests: &mut Requests<'a>, transport: &mut P::Transport,
    other_transport: &mut P::Transport, lease: crate::families::deepseek_v41::v41_backbone_cache::CacheLease,
    tokens: &[u32], from: usize, chunk_rows: usize, job: &NativeRequest,
    mut draft: Option<&mut DraftRuntime<'_, 'a, C>>, hold: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    let probe = job.probe.as_ref().context("scoring without a probe")?;
    let plan = ScoringPlan::new(&probe.spec)?;
    ensure!(tokens.len() >= 2, "scoring needs at least two tokens");
    probe.selected_score_path(plan.path.name());
    let from = from.clamp(1, tokens.len() - 1);
    let first = prefill(lib, runtime, pass, other, requests, transport, other_transport, lease, &tokens[..from],
        chunk_rows, job, draft.as_deref_mut(), hold)?;
    probe.row(from, &first.logits()?);
    for step in plan.steps(from, tokens.len()) {
        ensure!(!job.events.is_closed(), "client disconnected");
        let chunk = &tokens[step.clone()];
        let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
            image_mask: None, kind: plan.source_kind(chunk.len()) }])?;
        let selected: Vec<usize> = (0..chunk.len()).collect();
        let result = (|| -> Result<Vec<u8>> {
            let bytes = match plan.path {
                crate::shared::probe::ScorePath::Prefill => {
                    // SAFETY: this request owns the batch; both passes and its
                    // cache/storage remain live through synchronous completion.
                    runtime.block_on(unsafe { pass.prefill_logits(lib, requests, &mut batch, transport,
                        &selected, None) })?
                }
                crate::shared::probe::ScorePath::Decode => runtime.block_on(async {
                    // SAFETY: the teacher-forced batch is a full-phase verify
                    // on this lease. No other lane accesses its state; the
                    // future and logits download complete before commit/release.
                    unsafe { pass.execute_shared(&std::cell::RefCell::new(&mut *requests), &mut batch,
                        transport, 0, &selected).await?; }
                    pass.download_logits(&batch, &selected).await
                })?,
            };
            ensure!(bytes.len() == chunk.len() * scores::ROW_BYTES, "scoring logits extent differs");
            // Scoring never proposes drafts. Retain the legacy draft/cache
            // commit for calibration, but verify commits only teacher-forced rows.
            let scoring_draft = if plan.path == crate::shared::probe::ScorePath::Prefill { draft.as_deref_mut() } else { None };
            runtime.block_on(pass.commit_prefill::<C>(requests, &mut batch, scoring_draft, chunk.len() as u32))?;
            Ok(bytes)
        })();
        if result.is_err() { pass.discard(&mut batch)?; }
        let bytes = result?;
        for (j, row) in bytes.chunks_exact(scores::ROW_BYTES).enumerate() {
            probe.row(step.start + j + 1, &TokenScores::new(scores::VOCAB, row.to_vec())?.logits()?);
        }
    }
    let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length }));
    Err(ScoringDone.into())
}

#[cfg(test)]
mod scoring_tests {
    use super::*;
    use crate::shared::probe::ScorePath;
    use cuteafd_api::openai::probe::{Probe, ProbeSpec};

    #[test]
    fn scoring_path_is_explicit_and_legacy_default_is_unchanged() {
        let legacy = ScoringPlan::new(&ProbeSpec::default()).unwrap();
        assert_eq!(legacy.path, ScorePath::Prefill);
        assert_eq!(legacy.rows, 48);
        assert_eq!(legacy.source_kind(1), ExpertV2SourceKind::Prefill);
        assert_eq!(legacy.steps(3, 101).collect::<Vec<_>>(), vec![3..51, 51..99, 99..100]);
        let decode = ScoringPlan::new(&ProbeSpec { score_path: Some("decode".into()), ..ProbeSpec::default() }).unwrap();
        assert_eq!(decode.rows, 8);
        assert_eq!(decode.source_kind(1), ExpertV2SourceKind::Decode);
        assert_eq!(decode.source_kind(8), ExpertV2SourceKind::MtpVerify);
        let prefill = ScoringPlan::new(&ProbeSpec { score_path: Some("prefill".into()), ..ProbeSpec::default() }).unwrap();
        assert_eq!(prefill.rows, legacy.rows);
    }

    #[test]
    fn scoring_plan_rejects_unknown_paths_and_unsupported_widths() {
        for path in [None, Some("prefill"), Some("other"), Some("")] {
            assert!(ScoringPlan::new(&ProbeSpec { score_path: path.map(str::to_owned), verify_rows: Some(1),
                ..ProbeSpec::default() }).is_err());
        }
        for rows in [0, SCORING_ROWS + 1, usize::MAX] {
            assert!(ScoringPlan::new(&ProbeSpec { score_path: Some("decode".into()), verify_rows: Some(rows),
                ..ProbeSpec::default() }).is_err());
        }
    }

    #[test]
    fn teacher_forced_verify_windows_predict_each_position_once() {
        for width in [1, 3, 8, SCORING_ROWS] {
            for (from, len) in [(1, 2), (3, 14), (7, 111)] {
                let spec = ProbeSpec { score_path: Some("decode".into()), verify_rows: Some(width),
                    score_from: Some(from), ..ProbeSpec::default() };
                let plan = ScoringPlan::new(&spec).unwrap();
                let probe = Probe::new(spec);
                probe.selected_score_path(plan.path.name());
                probe.row(from, &[1.0, 0.0]); // Prefix prefill's final row.
                let steps: Vec<_> = plan.steps(from, len).collect();
                for step in steps {
                    assert!(!step.is_empty() && step.len() <= width);
                    assert_eq!(plan.source_kind(step.len()), if step.len() == 1 {
                        ExpertV2SourceKind::Decode
                    } else { ExpertV2SourceKind::MtpVerify });
                    for input_position in step {
                        probe.row(input_position + 1, &[1.0, 0.0]);
                    }
                }
                let record = probe.record();
                assert_eq!(record.score_path.as_deref(), Some("decode"));
                assert_eq!(record.scored, len - from);
                assert_eq!(record.rows.iter().map(|row| row.position).collect::<Vec<_>>(), (from..len).collect::<Vec<_>>());
            }
        }
    }
}

fn prefill_continuation<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    pass: &mut P, requests: &mut Requests<'a>, transport: &mut P::Transport,
    lease: crate::families::deepseek_v41::v41_backbone_cache::CacheLease, tokens: &[u32], chunk_rows: usize,
    job: &NativeRequest, mut draft: Option<&mut DraftRuntime<'_, 'a, C>>, hold: &mut dyn FnMut() -> Result<()>) -> Result<TokenScores> {
    ensure!(!tokens.is_empty(), "prefix continuation has no uncached rows");
    let mut anchor = None;
    let count = tokens.len().div_ceil(chunk_rows);
    for (index, chunk) in tokens.chunks(chunk_rows).enumerate() {
        ensure!(!job.events.is_closed(), "client disconnected");
        prefill_hold(hold);
        let started = Instant::now();
        let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
            image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
        let result = (|| -> Result<TokenScores> {
            let bytes = runtime.block_on(unsafe { pass.prefill_logits(lib, requests, &mut batch, transport,
                &[chunk.len() - 1], None) })?;
            let scores = TokenScores::new(scores::VOCAB, bytes)?;
            ensure!(!job.events.is_closed(), "client disconnected");
            runtime.block_on(pass.commit_prefill(requests, &mut batch, draft.as_deref_mut(), chunk.len() as u32))?;
            Ok(scores)
        })();
        if result.is_err() { pass.discard(&mut batch)?; }
        anchor = Some(result?);
        console::totals::prefill(chunk.len());
        console::Prefill::done(console::PrefillKind::Continuation, 0, index, count, chunk.len(), started);
    }
    anchor.context("prefix continuation produced no logits")
}
