pub(crate) mod prefill_target;
use prefill_target::PrefillTarget;
use speculative::DraftChain;
pub(crate) mod speculative;
pub(crate) mod scheduler;
pub(crate) mod console;
mod distributed;
mod coordinator_push;
pub(crate) mod scores;
use scores::RetainedScores;
pub(crate) mod prefix;
pub(crate) mod admission;
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
    // Identify the checkpoint before loading it: an unidentifiable directory fails in seconds.
    let mut profile = served_profile(args.model_id.clone(), &args.snapshot)?;
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
    profile.vision_health = vision_health;
    let router = cuteafd_api::openai::router_for_model(send, limits, stats, http_queue_wait, console_hub.clone(),
        api.serve(profile, &snapshot)?);
    axum::serve(listener, api.app(router, console_hub).into_make_service_with_connect_info::<std::net::SocketAddr>())
        .with_graceful_shutdown(cuteafd_bench::shutdown_signal())
        .await?;
    tokio::task::spawn_blocking(move || worker_thread.join())
        .await?
        .map_err(|_| anyhow::anyhow!("native CUDA worker panicked during shutdown"))?;
    Ok(())
}

/// The served profile: the public id is `--model-id`, else the checkpoint's
/// own id. The checkpoint is identified from its own files (hub path id,
/// model-card base_model ancestry, or its encoding/encoding.py against the
/// known V4.1 releases), never assumed; an unidentifiable plain directory
/// without --model-id fails closed. The history rule follows the lineage.
fn served_profile(model_id: Option<String>, snapshot: &std::path::Path) -> Result<cuteafd_api::openai::ModelProfile> {
    use cuteafd_api::openai::{chat::deepseek::v41_turns, snapshot_model_id, ModelEncoding, ModelProfile, V41Turns};
    let hub_id = snapshot_model_id(snapshot);
    // A plain directory's own card: its base_model ancestors (the walk lists the directory itself first).
    let card_ancestors = if hub_id.is_none() {
        cuteafd_bench::fidelity_match::lineage("local/snapshot", Some(snapshot)).split_off(1)
    } else { Vec::new() };
    let identity = v41_turns::identify(snapshot, hub_id, &card_ancestors, model_id.as_deref())?;
    let lineage: Vec<String> = identity.iter()
        .flat_map(|id| cuteafd_bench::fidelity_match::lineage(id, (identity.len() == 1).then_some(snapshot)))
        .collect();
    let turns = V41Turns::for_lineage(lineage.iter().map(String::as_str));
    let checkpoint = identity[0].clone();
    tracing::info!(checkpoint, ?lineage, ?turns, "V4.1 prompt history rule from the checkpoint lineage");
    Ok(ModelProfile::new(model_id.unwrap_or(checkpoint), ModelEncoding::DeepseekV41).with_v41_turns(turns))
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
    cuteafd_loader::plan::resolve_decode_lanes("deepseek_v41", Some(args.decode_lanes), u64::from(args.concurrency))?;
    match cuteafd_transport::fabric::discover() {
        Ok(report) => tracing::info!(target: "cuteafd::fabric", rails = report.rails.use_rails, "{}", report.summary()),
        Err(error) => tracing::warn!(target: "cuteafd::fabric", "fabric discovery failed: {error:#}"),
    }
    crate::shared::spark_topology::resolve(
        args.spark_tp,
        args.spark_ep,
        args.peers.len(),
        "serve-native",
    )?;
    ensure!((1..=8).contains(&args.peers.len()),
        "V4.1 Spark transport requires 1..=8 peers, got {}", args.peers.len());
    if args.pool_tokens == Some(0) && args.memory_reservation.is_none() {
        args.memory_reservation = Some("97%".parse()?);
    }
    args.host_cache_config()?;
    // The golden runs the worker without the API, so it loads no keyed policy.
    let api = if http { Some(args.api.load()?) } else { None };
    let listen = args.listen.clone();
    let snapshot = args.snapshot.clone();
    let limits = args.reserve.limits(args.max_context_tokens as usize)?;
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
            // `--host-wait block` sleeps briefly in this thread's per-layer
            // waits: wake within microseconds, not the default 50 us slack.
            if let Err(error) = cuteafd_ffi::set_thread_timer_slack_ns(1_000) {
                tracing::warn!("V4.1 engine thread keeps the default timer slack: {error:#}");
            }
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

#[cfg(test)]
mod served_profile_tests {
    use super::served_profile;
    use cuteafd_api::openai::V41Turns;

    #[test]
    fn plain_directory_is_identified_or_refused_never_assumed_original() {
        let dir = tempfile::tempdir().unwrap();
        let error = served_profile(None, dir.path()).unwrap_err().to_string();
        assert!(error.contains("--model-id"), "{error}");
        let aliased = served_profile(Some("org/served".into()), dir.path()).unwrap();
        assert_eq!((aliased.id.as_str(), aliased.v41_turns), ("org/served", V41Turns::KeepReasoning));
        // A hub path names its checkpoint; the rule follows it.
        let hub = dir.path().join("models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/abc");
        std::fs::create_dir_all(&hub).unwrap();
        let profile = served_profile(None, &hub).unwrap();
        assert_eq!((profile.id.as_str(), profile.v41_turns), ("deepseek-ai/DeepSeek-V4.1-Flash-1017", V41Turns::DropCompleted));
        // A plain directory whose own card names 1017 as base_model.
        let quant = dir.path().join("my-quant");
        std::fs::create_dir(&quant).unwrap();
        std::fs::write(quant.join("README.md"), "---\nbase_model: deepseek-ai/DeepSeek-V4.1-Flash-1017\nbase_model_relation: quantized\n---\n").unwrap();
        let profile = served_profile(Some("served".into()), &quant).unwrap();
        assert_eq!((profile.id.as_str(), profile.v41_turns), ("served", V41Turns::DropCompleted));
    }
}

#[cfg(test)]
mod prefill_capacity_tests {
    use super::prefill_capacity;

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

/// Protocol-v2 per-chunk transport diagnostics. Resolved once here so the RDMA
/// progress/poll path never reaches the process environment; the value is only
/// meaningful for a launch, so a start-time read is sufficient.
pub(crate) fn protocol_v2_timing() -> bool {
    cuteafd_transport::protocol_v2_timing_from_env()
}

/// The Spark link configuration every V4.1 transport uses (serve and its
/// GPU fixtures): the admitted endpoints' sealed frames fit it.
pub(crate) fn transport_config(timing: bool) -> TcpTransportConfig {
    TcpTransportConfig { timing, timeout: Duration::from_secs(120), max_frame_bytes: 64 * 1024 * 1024 }
}

fn spark_transport(
    peers: &[std::net::SocketAddr],
    capacity: u32,
    timing: bool,
    world: cuteafd_transport::expert::SparkWorld,
    lane: u32,
) -> Result<SparkExperts> {
    SparkExperts::new_world(world, peers, capacity, transport_config(timing), lane)
}

/// Background page-cache warm readers for `--engram-warm`.
const ENGRAM_WARM_READERS: usize = 8;

/// The Engram pipeline both V4.1 layouts serve: host staging for `slots`
/// prefill units of `rows` and the decode lanes (admitted against host
/// memory), the admitted scale or whole-table preload and, with
/// `--engram-warm`, a background page-cache warm that starts once the API
/// reports ready (and skips preloaded tables).
fn engram_pipeline(lib: &NativeLibrary, catalog: &cuteafd_loader::OfficialV41Catalog,
    args: &crate::cli::NativeServeArgs, rows: usize, slots: usize) -> Result<cuteafd_loader::EngramPipeline> {
    let map = cuteafd_loader::EngramTokenMap::from_file(&args.snapshot.join("tokenizer.json"))?;
    let shape = cuteafd_loader::EngramStagingShape::new(rows, slots)?;
    crate::shared::memory::budget::admit_host_memory(0,
        &[("v41/engram-prefill", shape.host_bytes() as u64),
          ("v41/engram-decode", shape.decode_host_bytes() as u64),
          ("v41/engram-row-cache", shape.cache_bytes(catalog.engram_row_bytes()?) as u64)], 0, 0)?;
    // SAFETY: the checkpoint remains immutable for all mapping lifetimes.
    let mut pipeline = unsafe { cuteafd_loader::EngramPipeline::new(lib, catalog, map, shape,
        cuteafd_loader::plan::mapped_scale_preload_bytes(args.mapped_scale_preload_gib)?)? };
    // Preload placement is final now: shared admission adds the reactor once
    // only if at least one table remained mapped, including with cache disabled.
    crate::shared::memory::budget::admit_host_memory_with_resident_formulas(0,
        &[("v41/engram-prefill-worker", shape.host_workspace_bytes() as u64),
          ("v41/engram-decode-worker", shape.decode_host_workspace_bytes() as u64),
          ("v41/engram-row-cache", pipeline.cache_bytes() as u64)],
        &[("v41/engram-prefill", shape.host_resident_bytes() as u64),
          ("v41/engram-decode", shape.decode_host_resident_bytes() as u64)], 0, 0)?;
    cuteafd_bench::context::set_mapped_placement(pipeline.placement().clone());
    if args.engram_warm {
        pipeline.start_warm(ENGRAM_WARM_READERS, || cuteafd_bench::context::readiness_s().is_some())?;
    }
    Ok(pipeline)
}

fn worker(
    mut args: crate::cli::NativeServeArgs,
    mut receive: mpsc::Receiver<NativeRequest>,
    ready: &mut Option<oneshot::Sender<std::result::Result<Option<std::sync::Arc<std::sync::atomic::AtomicBool>>, String>>>,
    stats: std::sync::Arc<std::sync::Mutex<serde_json::Value>>,
) -> Result<()> {
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
    admission::startup_phase(&lib, "v41/context", device_total)?;
    if small_card {
        let explicit = std::env::var("CUTEAFD_V41_FIXED_GRAPH_ROWS").ok();
        if let Some(shapes) = super::graph_policy::profile_fixed_shapes(device_total, explicit.as_deref())? {
            tracing::info!(?shapes, "V4.1 small-card fixed exact graph set; other rows execute eagerly");
            super::graph_policy::set_fixed_shapes(shapes)?;
        }
    }
    // Local expert waves need an exported AOT capacity; every other row
    // buffer follows the live prefill chunk (as the dual-RTX path does: the
    // FP8 plans keep their full scratch). 2048-row chunks: ~9 GiB less.
    let aot_capacity = prefill_capacity(args.prefill_batch_tokens)?;
    let capacity = cuteafd_core::coordinator_programs::v41_live_rows(args.prefill_batch_tokens)
        .context("invalid V4.1 prefill batch")?;
    let rows = capacity as usize;
    // Live requests capture decode shapes and the full prefill chunk only.
    super::graph_policy::set_prefill_rows(args.prefill_batch_tokens)?;
    let mut catalog = cuteafd_loader::read_official_v41_catalog(&args.snapshot,
    )?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::selected("deepseek_v41",
        if args.dspark { "dspark" } else { "off" }))?;
    let topology = crate::shared::spark_topology::resolve(
        args.spark_tp,
        args.spark_ep,
        args.peers.len(),
        "serve-native",
    )?;
    let spark_world = crate::shared::spark_topology::world(topology, args.peers.len())?;
    crate::shared::spark_topology::require_world(spark_world, &catalog,
        &(catalog.routed_experts().first_layer..catalog.routed_experts().layers).collect::<Vec<_>>())?;
    // Check reduction support before CUDA allocation or readiness publication.
    lib.v41_compact_reducer()?
        .require_rank_count(spark_world.topology().world_size() as u32)?;
    // Two private decode lanes at decode width, and `slots` prefill slots on
    // one shared storage set (PLAN "Prefill contract").
    let admitted = AdmittedPrefill::resolve(&args)?;
    ensure!(admitted.unit_cap == capacity as usize, "V4.1 prefill unit cap {} differs from the live rows {capacity}",
        admitted.unit_cap);
    let slots = admitted.waves;
    let decode_rows = DECODE_LANE_ROWS;
    admission::startup_phase(&lib, "v41/before native preload", device_total)?;
    let inventory = crate::shared::inventory::RuntimeInventory::measure(&lib, &[lib.cuda_get_device()?], crate::shared::inventory::InventoryPhase::PreWeight, |_| {
        admission::preload(&lib, capacity, 1, catalog.nvfp4().is_some())?;
        Ok(())
    })?;
    admission::startup_phase(&lib, "v41/after native preload", device_total)?;
    admission::admit_startup(&lib, &catalog, &args)?;
    let start = Instant::now();
    cuteafd_ffi::memory_ledger::relabel_other("v41/startup");
    admission::startup_phase(&lib, "v41/startup", device_total)?;
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
        admission::embedding_placement(args.embedding_placement, device_total))?;
    eprintln!(
        "native target backbone/index/embedding weights loaded in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    cuteafd_ffi::memory_ledger::relabel_other("v41/weights:backbone,cache-producers,index,embedding");
    admission::startup_phase(&lib, "v41/weights:backbone,cache-producers,index,embedding", device_total)?;
    let attention_plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
        cuteafd_core::V41_VERIFY_ROWS, args.dspark.then_some(usize::from(args.dspark_draft_limit)),
        args.attention_plan_rows)?;
    let pipeline = engram_pipeline(&lib, &catalog, &args, rows, slots)?;
    let engram_weights = [0, 1]
        .map(|i| EngramLayerWeights::load(&lib, &catalog, i, 256 * 1024 * 1024, 16 * 1024 * 1024))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
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
    cuteafd_ffi::memory_ledger::relabel_other("v41/weights:head");
    admission::startup_phase(&lib, "v41/weights:head", device_total)?;
    use crate::families::deepseek_v41::v41_target_pass::TargetTapWave;
    let head = || head_weights.wave(&vocabulary, cuteafd_core::V41_VERIFY_ROWS,
        TargetHeadWave::device_bytes(cuteafd_core::V41_VERIFY_ROWS)?);
    // A decode lane: every owner private at decode width.
    let decode_rows_usize = decode_rows as usize;
    // Each component under its own ledger scope (the plan == runtime check
    // names the one that differs).
    let make_decode_pass = || TargetPass::new(
        admission::scoped(DECODE_LANES_SCOPE, "embedding", || TargetEmbeddingWave::new(&lib, &table, decode_rows_usize,
            TargetEmbeddingWave::device_bytes(decode_rows_usize)?))?,
        admission::scoped(DECODE_LANES_SCOPE, "backbone", || BackboneLane::new(&weights, decode_rows,
            BackboneLane::workspace_bytes(&lib, decode_rows)?.into_iter().sum()))?,
        admission::scoped(DECODE_LANES_SCOPE, "index", || IndexLane::new(&index_weights, decode_rows,
            IndexLane::workspace_bytes(&lib, decode_rows)?.into_iter().sum()))?,
        admission::scoped(DECODE_LANES_SCOPE, "cache producers", || BackboneExecution::new(&producers, decode_rows,
            BackboneExecution::workspace_bytes(&lib, decode_rows)?))?,
        admission::scoped(DECODE_LANES_SCOPE, "engram rows", || EngramDeviceRows::new(&lib, decode_rows_usize,
            EngramDeviceRows::device_bytes(decode_rows_usize)?))?,
        admission::scoped(DECODE_LANES_SCOPE, "engram gates", || Ok([
            EngramGate::new(&engram_weights[0], decode_rows_usize, 1024 * 1024 * 1024)?,
            EngramGate::new(&engram_weights[1], decode_rows_usize, 1024 * 1024 * 1024)?]))?,
        admission::scoped(DECODE_LANES_SCOPE, "target head", head)?,
        admission::scoped(DECODE_LANES_SCOPE, "target taps", || TargetTapWave::new(&lib, decode_rows_usize,
            TargetTapWave::device_bytes(decode_rows_usize)?))?,
        Duration::from_secs(120),
    );
    let (mut pass, mut second) = (make_decode_pass()?, make_decode_pass()?);
    admission::startup_phase(&lib, "v41/workspace:decode-lanes", device_total)?;
    // The prefill slots: one storage set (row planes at the reserve, scratch
    // at the unit), each slot's host objects on it (`shared::memory::row_window`).
    let reserve = admitted.reserve_rows();
    let prefill_reserve = reserve as u64;
    let readers = [lib.cuda_get_device()?];
    let p = PREFILL_SLOTS_SCOPE;
    let slot_storage = SlotStorage {
        lane: admission::scoped(p, "lane", || crate::families::deepseek_v41::v41_backbone_lane::LaneStorage::new(
            &weights, prefill_reserve as u32, capacity, &readers))?,
        index: admission::scoped(p, "index", || IndexLane::storage(&index_weights, prefill_reserve as u32, capacity, &readers))?,
        producers: admission::scoped(p, "cache producers", || crate::families::deepseek_v41::v41_backbone_execution::ProducerStorage::new(
            &lib, reserve, rows))?,
        embedding: admission::scoped(p, "embedding", || TargetEmbeddingWave::storage(&lib, reserve, rows))?,
        upload: admission::scoped(p, "engram rows", || EngramDeviceRows::storage(&lib, reserve, rows))?,
        gates: admission::scoped(p, "engram gates", || Ok([EngramGate::storage(&lib, reserve, rows)?, EngramGate::storage(&lib, reserve, rows)?]))?,
        taps: admission::scoped(p, "target taps", || TargetTapWave::storage(&lib, reserve, rows))?,
    };
    let mut slot_passes = (0..slots).map(|_| TargetPass::new(
        TargetEmbeddingWave::on_storage(&lib, &table, slot_storage.embedding.clone())?,
        BackboneLane::on_storage(&weights, slot_storage.lane.clone())?,
        IndexLane::on_storage(&index_weights, slot_storage.index.clone())?,
        // Each slot's window batch table at the unit (charged with the producers).
        admission::scoped(p, "cache producers", || BackboneExecution::on_storage(&producers, slot_storage.producers.clone()))?,
        EngramDeviceRows::on_storage(slot_storage.upload.clone())?,
        [EngramGate::on_storage(&engram_weights[0], slot_storage.gates[0].clone())?,
         EngramGate::on_storage(&engram_weights[1], slot_storage.gates[1].clone())?],
        admission::scoped(p, "target heads", head)?,
        TargetTapWave::on_storage(slot_storage.taps.clone())?,
        Duration::from_secs(120),
    )).collect::<Result<Vec<_>>>()?;
    // The slots hold their components' storage; the set itself is no owner.
    drop(slot_storage);
    let packed_suffix = admission::scoped(p, "encoder suffix", ||
        slot_passes[0].new_packed_storage(&lib, capacity as usize))?;
    // Plan == runtime (WP6): the slot set and decode lanes are exactly the
    // planner's charges, nothing retained beyond them.
    admission::verify_target_charges(&lib, &args.native_lib, &[lib.cuda_get_device()?], decode_rows, reserve as u64, capacity, slots)?;
    admission::startup_phase(&lib, "v41/workspace:prefill-slots", device_total)?;
    let protocol_v2_timing = protocol_v2_timing();
    // Every replicated layout reserves its exact physical rank plane count; the
    // legacy 4-rank forecast is no longer reused for six ranks.
    // Decode lanes' links at their own width (P3); the prefill link below at the unit.
    let wave_bytes = NativeTp4Wave::device_bytes_for(decode_rows, args.peers.len())?;
    let lanes = crate::shared::serve::lanes::LaneSet::default();
    let [roce, second_roce] = lanes.try_map(|lane| spark_transport(&args.peers, decode_rows,
        protocol_v2_timing, spark_world, u32::try_from(lane.index())?))?;
    let mut transport = NativeTp4Wave::new(&lib, roce, wave_bytes)?;
    let mut second_transport = NativeTp4Wave::new(&lib, second_roce, wave_bytes)?;
    cuteafd_ffi::memory_ledger::relabel_other("v41/transport");
    admission::startup_phase(&lib, "v41/transport", device_total)?;
    // The one prefill link (endpoint after the lanes'), one return slot per
    // prefill slot, held by slot 0's FFN owner; the other slots' units send
    // on it too.
    let prefill_link = crate::families::deepseek_v41::v41_experts::coordinator::V41Link::prefill(&lib,
        spark_transport(&args.peers, capacity, protocol_v2_timing, spark_world, 2)?, slots, capacity)?;
    let mut prefill_link = Some(prefill_link);
    let slot_transports = (0..slots).map(|slot| match slot {
        0 => NativeTp4Wave::with_link(&lib, prefill_link.take().context("prefill link")?),
        _ => NativeTp4Wave::unlinked(&lib, capacity),
    }).collect::<Result<Vec<_>>>()?;
    let mut prefill_slots = prefill_target::PrefillSlots { packed_suffix, slots: slot_passes.drain(..).zip(slot_transports).collect() };
    cuteafd_ffi::memory_ledger::relabel_other("v41/transport");
    admission::startup_phase(&lib, "v41/transport", device_total)?;
    pass.install_attention_plan(attention_plan)?;
    second.install_attention_plan(attention_plan)?;
    for (slot, _) in &mut prefill_slots.slots { slot.install_attention_plan(attention_plan)?; }
    // L2 prefetch serves decode/verify steps only: the decode lanes.
    if let Some(budget) = args.l2.budget(&lib, crate::shared::l2_prefetch::OTHER_DEFAULT)? {
        pass.enable_l2_prefetch(budget)?;
        second.enable_l2_prefetch(budget)?;
    }
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
    attention_plan.validate_loaded(draft.as_ref().map(|draft| draft.attention_drafts()))?;
    tracing::info!(plan_rows=attention_plan.rows(), capacity=attention_plan.capacity_rows(),
        "V4.1 fixed BF16 AOT compressed/native FP32 window attention plan");
    cuteafd_ffi::memory_ledger::relabel_other("v41/drafter");
    admission::startup_phase(&lib, "v41/drafter", device_total)?;
    let vision_free_before = if small_card { lib.cuda_memory_info()?.0 } else { 0 };
    let dspark_bytes = draft_free_before.saturating_sub(vision_free_before);
    let mut vision = crate::families::deepseek_v41::v41_vision_encoder::Encoder::load(&args, &catalog)?;
    let vision_bytes = if small_card { vision_free_before.saturating_sub(lib.cuda_memory_info()?.0) } else { 0 };
    // Reserve both retention banks plus one in-flight snapshot per lane. These
    // allocations are counted before choosing KV capacity and local expert layers.
    ensure!(args.prefix_cache_entries <= 128, "invalid retained-turn limit");
    cuteafd_ffi::memory_ledger::relabel_other("v41/vision");
    admission::startup_phase(&lib, "v41/vision", device_total)?;
    let snapshot_slots = cuteafd_core::prefix::mark_slots_with_host(0, u64::from(args.prefix_cache_entries), 1, u64::MAX, args.host_cache_bytes.enabled()) as usize;
    let prefix_arenas = prefix::Arenas::new(
        crate::shared::memory::device::Device { library: &lib, id: lib.cuda_get_device()? },
        draft.as_ref().map(|d| d.mark_device()), snapshot_slots)?;
    let snapshot_bytes = prefix::Arenas::device_bytes(snapshot_slots, draft.is_some());
    tracing::info!(snapshot_slots, snapshot_bytes, "snapshot arenas reserved before serving");
    cuteafd_ffi::memory_ledger::relabel_other("v41/prefix-snapshots");
    admission::startup_phase(&lib, "v41/prefix-snapshots", device_total)?;
    // Size after vision, both lanes, transports and optional draft allocations are live.
    let experts = if args.rtx_expert_layers == admission::LocalLayers::Count(0) { Vec::new() }
        else { admission::expert_demands(&lib, &catalog, aot_capacity, 1)? };
    let local_workspace = admission::local_workspace(&lib, &catalog, &args, aot_capacity)?;
    if local_workspace > 0 {
        admission::verify_expert_charges(&catalog, &args.native_lib, u32::try_from(lib.sm_count()?)?, 1, capacity,
            slots, reserve as u64, local_workspace)?;
    }
    let memory = admission::admission_memory(&lib, &inventory)?;
    let (free, _) = memory[0];
    let pool = admission::resolve(&args, &memory, experts, local_workspace, &[0], 0, None)?;
    pass.bind_execution_plan(&pool.placement)?;
    second.bind_execution_plan(&pool.placement)?;
    for (slot, _) in &mut prefill_slots.slots { slot.bind_execution_plan(&pool.placement)?; }
    tracing::info!(source_pages=?pool.pages,global_bytes=pool.global_bytes,cache_bytes=?pool.cache_bytes,
        pool_tokens=pool.placement.pool_tokens,local_layers=pool.placement.onboard_layers,
        projected_free_bytes=free.saturating_sub(pool.cache_bytes[0]),reservation_bytes=?pool.reservation_bytes,
        "shared V4.1 cache and expert admission");
    let local_layers = pool.placement.onboard_layers;
    cuteafd_bench::context::set_rtx_expert_layers(pool.placement.onboard_layers);
    let spark_admission = admission::start_sparks(&args, &catalog, local_layers, capacity,
        topology)?;
    let mut requests = Requests::new(&lib, pipeline, args.concurrency as usize, pool.pages, pool.cache_bytes[0])?;
    cuteafd_ffi::memory_ledger::relabel_other("v41/kv");
    admission::startup_phase(&lib, "v41/kv", device_total)?;
    requests.use_engine_pages()?;
    cuteafd_ffi::memory_ledger::relabel_other("v41/kv-prefix-install");
    admission::startup_phase(&lib, "v41/kv-prefix-install", device_total)?;
    if args.rtx_expert_layers != admission::LocalLayers::Count(0) {
        use crate::families::deepseek_v41::v41_experts::{ExpertLayer, ExpertWeights, local::LocalExpertWave};
        let local_started = Instant::now();
        let lane_bytes = |rows: u32| LocalExpertWave::device_bytes_for(&lib, rows, catalog.nvfp4().is_some());
        let (per_slot, per_decode) = (lane_bytes(aot_capacity)?, lane_bytes(decode_rows)?);
        let budgets = (0..40).map(|layer| {
            let selection = ExpertLayer::BackboneFull { layer };
            ExpertWeights::plan(&lib, &catalog, selection)
        }).collect::<Result<Vec<_>>>()?;
        let admitted = pool.placement.expert_ranges[0];
        let expected = budgets.iter().take(local_layers).map(|b| b.resident_bytes).sum::<usize>()
            + if local_layers>0 {per_slot*slots + per_decode*2
                + budgets.iter().take(local_layers).map(|b|b.device_staging_bytes).max().unwrap_or(0)} else {0};
        ensure!(expected as u64 <= admitted.peak_bytes,"runtime local owners exceed shared admission");
        tracing::info!(layers=local_layers,peak_bytes=admitted.peak_bytes,"shared RTX expert placement");
        if local_layers > 0 {
            let mut loaded = Vec::with_capacity(local_layers);
            for layer in 0..local_layers {
                loaded.push(ExpertWeights::load(&lib, &catalog, ExpertLayer::BackboneFull { layer },
                    budgets[layer].peak_device_bytes()?)?);
            }
            let weights = std::rc::Rc::new(loaded);
            for (_, transport) in &mut prefill_slots.slots {
                transport.install_local(LocalExpertWave::new(&lib, weights.clone(), aot_capacity, per_slot)?)?;
            }
            for transport in [&mut transport, &mut second_transport] {
                transport.install_local(LocalExpertWave::new(&lib, weights.clone(), decode_rows, per_decode)?)?;
            }
        }
        tracing::info!(layers=local_layers, elapsed_ms=local_started.elapsed().as_millis(),
            "local RTX experts ready");
    }
    if let Some(pending) = spark_admission { pending.wait()?; }
    if local_layers < 40 {
        // The workers serve their layers now: open the sessions and start each
        // link's exchange worker (the first remote layer sizes the rings).
        transport.bootstrap(local_layers as u32, catalog.nvfp4().is_some())?;
        second_transport.bootstrap(local_layers as u32, catalog.nvfp4().is_some())?;
        prefill_slots.slots[0].1.bootstrap(local_layers as u32, catalog.nvfp4().is_some())?;
    }
    let (free, total) = lib.cuda_memory_info()?;
    let occupied = total - free;
    cuteafd_ffi::memory_ledger::relabel_other("v41/local-experts");
    admission::startup_phase(&lib, "v41/local-experts", device_total)?;
    crate::shared::memory_report::release_load_staging(&lib);
    tracing::info!(rtx_layers=local_layers, first_remote_dispatch_layer=local_layers,
        remote_dispatch_layers=40-local_layers, spark_world=args.peers.len(),
        spark_topology=?topology.map(|t| (t.tp(), t.ep())),
        device_occupied_bytes=occupied, device_budget_bytes=?pool.reservation_bytes,
        "native serving residency ready");
    if let Some(draft) = &mut draft { draft.configure_policy(&transport, catalog.nvfp4().is_some())?; }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let prefixes = scheduler::prepare_prefix_cache(&lib, &args, &requests, prefix_arenas, draft.is_some())?;
    scheduler::publish_capacity(&requests, &prefixes, pool.placement.max_pool_tokens_no_onboard);
    vision.connect()?;
    admission::startup_phase(&lib, "v41/ready", device_total)?;
    ready
        .take()
        .context("startup readiness missing")?
        .send(Ok(vision.health_handle()))
        .map_err(|_| anyhow::anyhow!("API startup cancelled"))?;
    scheduler::serve(&lib, &args, &runtime, &mut receive, &mut pass, &mut second,
        &mut requests, &mut transport, &mut second_transport, &mut prefill_slots, draft.as_mut(), &mut vision, stats, prefixes)
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
    requests: &mut Requests<'a>,
    prefill_slots: &mut prefill_target::PrefillSlots<'a, P>,
    // A decode lane for a cold-replay probe's decode-shaped steps: only a
    // decode lane's transport reaches a Spark decode execution.
    mut decode_lane: Option<(&mut P, &mut P::Transport)>,
    lease: crate::families::deepseek_v41::v41_backbone_cache::CacheLease,
    tokens: &[u32],
    shape: PrefillShape,
    job: &NativeRequest,
    draft: Option<&mut DraftRuntime<'_, 'a, C>>,
    hold: &mut dyn FnMut() -> Result<()>,
) -> Result<RetainedScores> {
    use prefill_target::{V41Prefill, V41PrefillStep, V41PrefillTimes};
    let PrefillShape { rows: chunk_rows, waves, .. } = shape;
    if let Some(probe) = job.probe.as_ref().filter(|probe| !probe.spec.cold_steps.is_empty()) {
        probe.spec.validate_cold_steps(tokens.len(), tokens.len(), 1)?;
        ensure!(!probe.spec.cold_steps[0].decode, "V4.1 cold replay must begin with encoder prefill");
        let mut previous = 0;
        let mut scores = None;
        let mut draft = draft;
        for (index, cold_step) in probe.spec.cold_steps.iter().enumerate() {
            let chunk = &tokens[previous..cold_step.end];
            if index == 0 {
                let mut times = V41PrefillTimes::default();
                let mut step = V41PrefillStep { lib, runtime, requests, prefill: &mut *prefill_slots, lease,
                    tokens: chunk, chunk_rows, waves, job,
                    draft: draft.as_deref_mut(), hold, times: &mut times };
                scores = Some(crate::shared::prefill_share::run_to_end(&mut step, &mut V41Prefill::new())?);
            } else if cold_step.decode {
                let (pass, transport) = decode_lane.as_mut()
                    .context("V4.1 cold replay decode steps need a decode lane")?;
                let (pass, transport) = (&mut **pass, &mut **transport);
                let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
                    image_mask: None, kind: ExpertV2SourceKind::Decode }])?;
                let result = (|| -> Result<RetainedScores> {
                    // SAFETY: the probe's request and batch stay owned until execution and commit drain.
                    runtime.block_on(unsafe { pass.execute_shared(&std::cell::RefCell::new(&mut *requests),
                        &mut batch, transport, 0, &[chunk.len() - 1]) })?;
                    let bytes = runtime.block_on(pass.download_logits(&batch, &[0]))?;
                    runtime.block_on(pass.commit_prefill(requests, &mut batch, draft.as_deref_mut(), chunk.len() as u32))?;
                    RetainedScores::new(scores::VOCAB, bytes)
                })();
                if result.is_err() { pass.discard(&mut batch)?; }
                scores = Some(result?);
            } else {
                if index + 1 == probe.spec.cold_steps.len() {
                    requests.record_state(lease, &job.probe, "checkpoint")?;
                }
                let (pass, transport) = prefill_slots.slot0();
                scores = Some(prefill_continuation(lib, runtime, pass, requests, transport, lease,
                    chunk, chunk_rows, job, draft.as_deref_mut(), hold)?);
            }
            previous = cold_step.end;
        }
        return scores.context("V4.1 cold replay produced no logits");
    }
    let mut times = V41PrefillTimes::default();
    let mut step = V41PrefillStep { lib, runtime, requests, prefill: prefill_slots, lease, tokens, chunk_rows, waves,
        job, draft, hold, times: &mut times };
    crate::shared::prefill_share::run_to_end(&mut step, &mut V41Prefill::new())
}

/// Rows of a decode lane's private owners: the compiled cover of the widest
/// decode/verify step (`V41_VERIFY_ROWS`, which every verify selection and
/// draft trim already bounds).
pub(crate) const DECODE_LANE_ROWS: u32 = cuteafd_loader::placement::families::deepseek_v41::owners::DECODE_LANE_ROWS as u32;

/// Ledger scopes of the target owners the planner charges (WP6 plan == runtime).
pub(crate) const DECODE_LANES_SCOPE: &str = "v41/workspace:decode-lanes";
pub(crate) const PREFILL_SLOTS_SCOPE: &str = "v41/workspace:prefill-slots";

/// One RTX's prefill-slot storage set (`shared::memory::row_window`): row
/// planes at the reserve, scratch at the unit cap. Every slot's components
/// take their storage from it; it owns nothing once the slots are built.
struct SlotStorage<'a> {
    lane: std::rc::Rc<crate::families::deepseek_v41::v41_backbone_lane::LaneStorage<'a>>,
    index: std::rc::Rc<crate::families::deepseek_v41::v41_index_lane::IndexLaneStorage<'a>>,
    producers: crate::families::deepseek_v41::v41_backbone_execution::ProducerStorage<'a>,
    embedding: std::rc::Rc<crate::families::deepseek_v41::v41_target_embedding::TargetEmbeddingStorage<'a>>,
    upload: std::rc::Rc<crate::families::deepseek_v41::v41_engram::EngramRowsStorage<'a>>,
    gates: [std::rc::Rc<crate::families::deepseek_v41::v41_engram::layer::EngramGateStorage<'a>>; 2],
    taps: std::rc::Rc<crate::families::deepseek_v41::v41_target_pass::TargetTapStorage<'a>>,
}

/// The admitted prefill shape: `waves` slot passes on one storage set,
/// units of `unit_cap` rows (three units of the reserve by default; the
/// reserve is the standard one or the planner's fit cap).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmittedPrefill { pub unit_cap: usize, pub waves: usize }

impl AdmittedPrefill {
    /// The serve's resolution of the deployment's shape through the shared
    /// prefill policy (the planner's own resolution). The unit is
    /// `--prefill-batch-tokens`.
    pub(crate) fn resolve(args: &crate::cli::NativeServeArgs) -> Result<Self> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides, RESERVE_ROWS};
        let limits = cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::deepseek::DEEPSEEK_V41);
        let reserve = u64::from(args.prefill_reserve_rows);
        let plan = prefill_shape(args.peers.len(), limits, PrefillOverrides {
            unit_rows: Some(u64::from(args.prefill_batch_tokens)), waves: Some(u64::from(args.prefill_waves)),
            span_rows: None, reserve_rows: (reserve != RESERVE_ROWS).then_some(reserve),
        })?;
        let shape = Self { unit_cap: usize::try_from(plan.unit_rows)?, waves: usize::try_from(plan.waves)? };
        ensure!(shape.reserve_rows() <= crate::shared::memory::row_window::MAX_RESERVE_ROWS,
            "V4.1 prefill reserve {} exceeds the row planes", shape.reserve_rows());
        Ok(shape)
    }
    /// Rows every shared row plane spans: the slots' units.
    pub(crate) fn reserve_rows(&self) -> usize { self.waves * self.unit_cap }
}

/// The serving prefill shape: unit rows (`--prefill-batch-tokens`) and units
/// in flight on the shared executor (`--prefill-waves`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillShape {
    pub rows: usize,
    pub waves: usize,
}

impl PrefillShape {
    pub(crate) fn from_args(args: &crate::cli::NativeServeArgs) -> Self {
        Self { rows: args.prefill_batch_tokens as usize, waves: usize::from(args.prefill_waves) }
    }
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

    fn for_probe(spec: &cuteafd_api::openai::probe::ProbeSpec, chunk_rows: usize) -> Result<Self> {
        let mut plan = Self::new(spec)?;
        if plan.path == crate::shared::probe::ScorePath::Prefill && spec.score_positions.is_some() {
            // Sparse scoring keeps the head bounded while exercising the actual prefill geometry.
            ensure!(chunk_rows > 0, "prefill scoring capacity is zero");
            plan.rows = chunk_rows;
        }
        Ok(plan)
    }

    fn head_rows(&self, probe: &cuteafd_api::openai::probe::Probe,
        step: &std::ops::Range<usize>) -> Result<Vec<usize>> {
        let sparse = self.path == crate::shared::probe::ScorePath::Prefill
            && probe.spec.score_positions.is_some();
        let mut rows: Vec<usize> = (0..step.len())
            .filter(|&row| !sparse || probe.wants(step.start + row + 1)).collect();
        ensure!(rows.len() <= SCORING_ROWS, "scored prefill selection exceeds target head capacity");
        if rows.is_empty() { rows.push(step.len() - 1); }
        Ok(rows)
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
/// Decode-shaped steps (Decode/MtpVerify rows) run on `decode_lane`, whose
/// transport is a Spark decode execution; prefill-shaped steps stay on prefill
/// slot 0, whose link the Spark admits only for prefill rows.
/// Returns [`ScoringDone`] when done.
#[allow(clippy::too_many_arguments)]
fn score<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    requests: &mut Requests<'a>, decode_lane: (&mut P, &mut P::Transport),
    prefill_slots: &mut prefill_target::PrefillSlots<'a, P>,
    lease: crate::families::deepseek_v41::v41_backbone_cache::CacheLease,
    tokens: &[u32], from: usize, shape: PrefillShape, job: &NativeRequest,
    mut draft: Option<&mut DraftRuntime<'_, 'a, C>>, hold: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    let probe = job.probe.as_ref().context("scoring without a probe")?;
    let plan = ScoringPlan::for_probe(&probe.spec, shape.rows)?;
    ensure!(tokens.len() >= 2, "scoring needs at least two tokens");
    probe.selected_score_path(plan.path.name());
    let from = from.clamp(1, tokens.len() - 1);
    let (decode_pass, decode_transport) = decode_lane;
    let first = prefill(lib, runtime, requests, prefill_slots, Some((&mut *decode_pass, &mut *decode_transport)), lease, &tokens[..from],
        shape, job, draft.as_deref_mut(), hold)?;
    let (slot_pass, slot_transport) = prefill_slots.slot0();
    let (pass, transport) = match plan.path {
        crate::shared::probe::ScorePath::Decode => (decode_pass, decode_transport),
        crate::shared::probe::ScorePath::Prefill => (slot_pass, slot_transport),
    };
    probe.row(from, &first.logits()?);
    for step in plan.steps(from, tokens.len()) {
        scheduler::stall::tick();
        ensure!(!job.events.is_closed(), "client disconnected");
        let chunk = &tokens[step.clone()];
        let selected = plan.head_rows(probe, &step)?;
        let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
            image_mask: None, kind: plan.source_kind(chunk.len()) }])?;
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
            ensure!(bytes.len() == selected.len() * scores::ROW_BYTES, "scoring logits extent differs");
            // Scoring never proposes drafts. Retain the legacy draft/cache
            // commit for calibration, but verify commits only teacher-forced rows.
            let scoring_draft = if plan.path == crate::shared::probe::ScorePath::Prefill { draft.as_deref_mut() } else { None };
            runtime.block_on(pass.commit_prefill::<C>(requests, &mut batch, scoring_draft, chunk.len() as u32))?;
            Ok(bytes)
        })();
        if result.is_err() { pass.discard(&mut batch)?; }
        let bytes = result?;
        for (&selected_row, row) in selected.iter().zip(bytes.chunks_exact(scores::ROW_BYTES)) {
            probe.row(step.start + selected_row + 1, &RetainedScores::new(scores::VOCAB, row.to_vec())?.logits()?);
        }
        if plan.path == crate::shared::probe::ScorePath::Prefill {
            probe.scored_prefill_chunk(step.start + 1, chunk.len());
        }
        scheduler::stall::tick();
    }
    let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length }));
    Err(ScoringDone.into())
}

#[cfg(test)]
mod admitted_prefill_tests {
    use super::*;
    fn resolve(extra: &[&str]) -> AdmittedPrefill {
        use clap::Parser;
        let mut argv = vec!["cuteafd", "serve-native", "--snapshot", "/model", "--native-lib", "/native.so"];
        argv.extend_from_slice(extra);
        let crate::cli::Commands::ServeNative(args) = crate::cli::Cli::try_parse_from(argv).unwrap().command else {
            panic!("expected serve-native");
        };
        AdmittedPrefill::resolve(&args).unwrap()
    }
    /// The deployment's shape (TJ 2026-10-28: fixed 3) resolves exactly; a
    /// pinned depth keeps its own shape.
    #[test]
    fn the_deployment_shape_resolves_exactly_and_pins_keep_theirs() {
        let three = resolve(&["--prefill-waves", "3", "--prefill-batch-tokens", "2752"]);
        assert_eq!((three.unit_cap, three.waves, three.reserve_rows()), (2752, 3, 8256));
        let one = resolve(&["--prefill-waves", "1", "--prefill-batch-tokens", "4096"]);
        assert_eq!((one.unit_cap, one.waves, one.reserve_rows()), (4096, 1, 4096));
        let capped = resolve(&["--prefill-waves", "3", "--prefill-batch-tokens", "1792", "--prefill-reserve-rows", "5376"]);
        assert_eq!((capped.unit_cap, capped.waves, capped.reserve_rows()), (1792, 3, 5376));
        assert!(cuteafd_loader::placement::families::deepseek_v41::owners::transports_bytes(80, 4096, 4, 1).is_ok());
    }
}

#[cfg(test)]
mod spark_lane_tests {
    use super::*;

    /// The Spark admits the two decode lanes' endpoints as decode executions
    /// and the prefill link (endpoint 2) only for prefill rows, at every k.
    /// So decode-shaped rows (Decode/MtpVerify: scoring verify steps, cold
    /// replay decode steps) must run on a decode lane, never on prefill slot 0
    /// (its transport is the prefill link): cards30 min failed on exactly that.
    #[test]
    fn decode_shaped_rows_never_ride_the_prefill_link() {
        use cuteafd_transport::{worker_selection::{execution_owner, ExecutionOwner}, ExpertV2SourceKind::*};
        for (slots, unit) in [(2u64, 4096u64), (3, 2752), (4, 2048)] {
            let executions = cuteafd_loader::plan::spark_executions::admit_family("deepseek_v41",
                &cuteafd_core::EXL3_CAPACITIES, None, unit, slots).unwrap();
            for lane in 0..2 {
                for (kind, rows) in [(Decode, 1), (MtpVerify, 8)] {
                    assert_eq!(execution_owner(&executions, kind, lane, rows).unwrap(), ExecutionOwner::Decode(lane as usize));
                }
            }
            for kind in [Decode, MtpVerify] {
                assert!(execution_owner(&executions, kind, 2, 8).is_err(), "k={slots}: decode rows on the prefill link");
            }
            assert_eq!(execution_owner(&executions, Prefill, 2, unit as u32).unwrap(), ExecutionOwner::Prefill);
        }
        // Every decode-shaped call site takes its pass from a decode lane.
        let source = include_str!("v41_native_serve.rs");
        let body = |name: &str| { let start = source.find(name).unwrap(); &source[start..start + source[start..].find("\n}\n").unwrap()] };
        let cold = body("fn prefill<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(");
        let decode_step = &cold[cold.find("} else if cold_step.decode {").unwrap()..cold.find("} else {\n                if index + 1").unwrap()];
        assert!(decode_step.contains("decode_lane") && !decode_step.contains("slot0()"), "cold replay decode step must use a decode lane");
        let score = body("fn score<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(");
        assert!(score.contains("ScorePath::Decode => (decode_pass, decode_transport)"), "decode-path scoring must use the decode lane");
    }
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
    fn sparse_prefill_uses_backbone_capacity_without_changing_decode_head_indices() {
        let spec = ProbeSpec { score_path: Some("prefill".into()),
            score_positions: Some(vec![101, 1100, 2048]), ..ProbeSpec::default() };
        let plan = ScoringPlan::for_probe(&spec, 2048).unwrap();
        assert_eq!(plan.rows, 2048);
        let probe = Probe::new(spec.clone());
        assert_eq!(plan.head_rows(&probe, &(100..2148)).unwrap(), vec![0, 999, 1947]);
        assert_eq!(plan.head_rows(&probe, &(2148..4196)).unwrap(), vec![2047]);
        assert!(ScoringPlan::for_probe(&spec, 0).is_err());
        let too_many = Probe::new(ProbeSpec { score_positions: Some((101..150).collect()),
            ..spec.clone() });
        assert!(plan.head_rows(&too_many, &(100..2148)).is_err());

        let decode_spec = ProbeSpec { score_path: Some("decode".into()),
            verify_rows: Some(8), ..spec };
        let decode = ScoringPlan::for_probe(&decode_spec, 2048).unwrap();
        assert_eq!(decode.rows, 8);
        assert_eq!(decode.head_rows(&Probe::new(decode_spec), &(100..108)).unwrap(),
            (0..8).collect::<Vec<_>>());
        assert_eq!(ScoringPlan::for_probe(&ProbeSpec::default(), 2048).unwrap().rows, 48);
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
    job: &NativeRequest, mut draft: Option<&mut DraftRuntime<'_, 'a, C>>, hold: &mut dyn FnMut() -> Result<()>) -> Result<RetainedScores> {
    ensure!(!tokens.is_empty(), "prefix continuation has no uncached rows");
    let mut anchor = None;
    let count = tokens.len().div_ceil(chunk_rows);
    for (index, chunk) in tokens.chunks(chunk_rows).enumerate() {
        ensure!(!job.events.is_closed(), "client disconnected");
        prefill_hold(hold);
        let started = Instant::now();
        let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
            image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
        let result = (|| -> Result<RetainedScores> {
            let bytes = runtime.block_on(unsafe { pass.prefill_logits(lib, requests, &mut batch, transport,
                &[chunk.len() - 1], None) })?;
            let scores = RetainedScores::new(scores::VOCAB, bytes)?;
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
