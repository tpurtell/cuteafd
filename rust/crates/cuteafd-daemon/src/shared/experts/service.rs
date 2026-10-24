//! Native GPU resources and inference QPs share one owning thread.
mod local;
mod backend;
mod census;
mod modules;
mod routes;
mod exlr8;
mod push;
mod push_fp8;
mod push_native;
mod push_exlr8;

use crate::shared::experts::execution::HostExpertExchange;
use crate::shared::experts::layer::{ExpertLayer, ExpertWeights};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::OfficialV41Catalog;
use cuteafd_transport::expert::{BackboneRequest, SparkTopology, SparkWorld};
use std::{path::PathBuf, sync::{mpsc, Arc}, thread};

pub(crate) async fn run(args: crate::cli::NativeExpertDaemonArgs) -> Result<()> {
    tracing::info!(target: "cuteafd::fabric", source = "sparknest", status = cuteafd_transport::traffic::storage_status(), "local storage traffic source");
    match cuteafd_transport::fabric::discover() {
        Ok(report) => tracing::info!(target: "cuteafd::fabric", rails = report.rails.use_rails, "{}", report.summary()),
        Err(error) => tracing::warn!(target: "cuteafd::fabric", "fabric discovery failed: {error:#}"),
    }
    let encoder = if args.encoder || args.encoder_only {
        Some(crate::shared::vision::worker::EncoderWorkerConfig {
            listen: args.encoder_listen.clone(),
            plan_hash: crate::shared::vision::worker::parse_plan_hash(args.encoder_plan_hash.as_deref().context("--encoder-plan-hash required")?)?,
            revision: args.encoder_revision.clone().or_else(|| args.snapshot.file_name().and_then(|s| s.to_str()).map(str::to_owned)).context("encoder revision required")?,
            max_tokens: args.encoder_max_tokens as usize,
        })
    } else { None };
    let audio_encoder = if args.audio_encoder || args.audio_encoder_only {
        Some(crate::shared::vision::worker::AudioWorkerConfig {
            listen: args.audio_encoder_listen.clone(),
            plan_hash: crate::shared::vision::worker::parse_plan_hash(args.audio_encoder_plan_hash.as_deref().context("--audio-encoder-plan-hash required")?)?,
            revision: args.audio_encoder_revision.clone().or_else(|| args.snapshot.file_name().and_then(|s| s.to_str()).map(str::to_owned)).context("audio encoder revision required")?,
        })
    } else { None };
    if args.encoder_only || args.audio_encoder_only {
        return tokio::task::spawn_blocking(move || -> Result<()> {
            let (_vision, _audio, _) = crate::shared::vision::worker::start_encoders(encoder.as_ref(), audio_encoder.as_ref(),
                &args.snapshot, args.native_lib, args.device_budget_bytes as u64)?;
            loop { thread::park(); }
        }).await.context("encoder-only owner failed")?;
    }
    let topology = crate::shared::spark_topology::resolve(
        args.spark_tp,
        args.spark_ep,
        args.world as usize,
        "expertd-native",
    )?;
    if let Some(topology) = topology {
        ensure!(
            (args.rank as usize) < topology.world_size(),
            "expertd-native rank {} is outside the {}x{} topology",
            args.rank,
            topology.tp(),
            topology.ep()
        );
    }
    let config = NativeExpertServiceConfig {
        library: args.native_lib,
        exl3_aot_dir: args.exl3_aot_dir,
        exl3_schedule: args.exl3_schedule,
        fp8_package: args.fp8_package,
        snapshot: args.snapshot,
        exlr8_manifest: args.exlr8_manifest,
        rank: args.rank as usize,
        world: args.world as usize,
        model_admission: None,
        selected_layers: None,
        capacity: 0,
        device_budget: args.device_budget_bytes,
        max_frame_bytes: 0,
        rdma_endpoints: 0,
        declared_world: crate::shared::spark_topology::world(topology, args.world as usize)?,
        native_spark_tp2: false,
        bf16_ingress: false,
        encoder,
        audio_encoder,
    };
    tokio::task::spawn_blocking(move || local::run(config, &args.listen))
        .await
        .context("native local RoCE owner failed")?
}

#[derive(Clone)]
pub(crate) struct NativeExpertServiceConfig {
    pub encoder: Option<crate::shared::vision::worker::EncoderWorkerConfig>,
    pub audio_encoder: Option<crate::shared::vision::worker::AudioWorkerConfig>,
    pub library: PathBuf,
    pub exl3_aot_dir: Option<PathBuf>,
    /// Decode schedule of the EXL3 exports (`m<capacity>` or, for GB10,
    /// the GLM 5.3 Flash `m<capacity>-gb10` siblings; the same bits).
    pub exl3_schedule: crate::shared::experts::exl3::execution::Exl3Schedule,
    /// FP8 package layout directory (default `<libdir>/fp8/fp8-<family>/tp<world>`).
    pub fp8_package: Option<PathBuf>,
    pub snapshot: PathBuf,
    pub exlr8_manifest: Option<String>,
    pub rank: usize,
    pub world: usize,
    model_admission: Option<cuteafd_transport::worker_selection::ModelAdmission>,
    selected_layers: Option<Vec<usize>>,
    pub capacity: u32,
    pub device_budget: usize,
    pub max_frame_bytes: usize,
    /// Explicit selected-family connection geometry, never inferred from a different family.
    pub rdma_endpoints: usize,
    /// Physical topology and request encoding resolved before worker startup.
    pub declared_world: SparkWorld,
    /// Native Flash's TP2 kernel uses the Spark shard role with legacy wire
    /// requests. Set only after catalog validation, never from a CLI override.
    native_spark_tp2: bool,
    bf16_ingress: bool,
}

fn admit_worker_peak(library: &NativeLibrary, load_peak: usize, serve_peak: usize) -> Result<()> {
    let available = library.cuda_allocatable_memory_info()
        .context("query worker allocatable memory for admission")?.0;
    ensure!(load_peak <= available && serve_peak <= available,
        "Spark worker needs {load_peak} bytes at load / {serve_peak} at serve but only {available} bytes are available");
    Ok(())
}

fn load_weights<'a>(
    library: &'a NativeLibrary,
    catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig,
    modules: &mut modules::WarmModules<'a>,
    push: Option<&mut std::net::TcpStream>,
) -> Result<(backend::Weights<'a>, usize)> {
    let layers = config.resident_layers(catalog.routed_experts().layers)?;
    let first = catalog.routed_experts().first_layer;
    ensure!(layers.iter().all(|&layer| layer >= first),
        "coordinator selected experts before this checkpoint's routed boundary {first}");
    validate_topology(config, catalog)?;
    // Release metadata temporaries after the final topology recheck, before admission.
    cuteafd_ffi::trim_startup_metadata();
    if catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8 {
        return exlr8::load(library, catalog, config, modules.exlr8()?, push);
    }
    if catalog.exl3().is_some() { return backend::load_exl3(library, catalog, config, push); }
    if catalog.fp8().is_some() { return backend::load_fp8(library, catalog, config, modules.take_fp8()?, push); }
    log_spark_memory_if_enabled(library, config, "worker startup", None, None);
    // NVFP4 backbone experts load through the format-aware ExpertWeights path.
    let nvfp4 = catalog.nvfp4().is_some();
    let mut resident = 0usize;
    let mut staging = 0usize;
    let mut pinned_host = 0usize;
    let mut read_scratch = 0usize;
    for &layer in &layers {
        let plan = ExpertWeights::plan(library, catalog, config.selection(layer)?)?;
        resident = resident
            .checked_add(plan.resident_bytes)
            .context("resident budget overflow")?;
        staging = staging.max(plan.device_staging_bytes);
        pinned_host = pinned_host.max(plan.pinned_host_bytes);
        read_scratch = read_scratch.max(plan.read_scratch_bytes);
    }
    ensure!(
        resident
            .checked_add(staging)
            .context("loading budget overflow")?
            <= config.device_budget,
        "native TP weights and staging exceed device budget"
    );
    // The workspace is planned from the same layer selection as the load, so a
    // TP2/TP3 topology cannot be budgeted with TP4 scratch.
    let workspace = ExpertWeights::plan_execution(
        config.selection(0)?,
        library,
        config.compiled_capacity()?,
        config.capacity,
        nvfp4,
    )?
    .total()?;
    ensure!(
        resident
            .checked_add(workspace)
            .context("execution budget overflow")?
            <= config.device_budget,
        "native TP weights and execution workspace exceed device budget"
    );
    // Every layout admits the unified-pool peak before allocation. Pinned host
    // staging, read scratch and exchange storage compete with device weights on
    // GB10; explicit topologies also enforce the configured reservation below.
    let budget = spark_admission_budget(config, resident, staging, pinned_host,
        read_scratch, workspace)?;
    admit_worker_peak(library, budget.load_peak, budget.serve_peak)?;
    if let Some(topology) = config.declared_world.ownership_topology() {
        ensure!(
            budget.load_peak <= config.device_budget
                && budget.serve_peak <= config.device_budget,
            "explicit {}x{} Spark admission exceeds the {}-byte device budget: \
             resident {} + load peak {} / serve peak {}",
            topology.tp(),
            topology.ep(),
            config.device_budget,
            resident,
            budget.load_peak,
            budget.serve_peak
        );
        // A configured budget above the real pool must not admit more than the
        // startup measurement can hold. The OS reserve stays outside this budget
        // by construction; page-cache bytes are reclaimable and are not added.
        // This is a safety gate, so a failed query fails closed instead of
        // silently skipping the real-availability check.
        let free = library.cuda_allocatable_memory_info().context("query Spark memory for admission")?.0;
        ensure!(
            budget.load_peak <= free && budget.serve_peak <= free,
            "explicit replicated Spark admission needs {} bytes at load / {} at serve \
             but only {free} bytes are actually available",
            budget.load_peak,
            budget.serve_peak
        );
        tracing::info!(
            rank = config.rank,
            resident_bytes = resident,
            staging_bytes = staging,
            pinned_host_bytes = pinned_host,
            read_scratch_bytes = read_scratch,
            workspace_bytes = workspace,
            exchange_bytes = budget.exchange,
            row_indices_bytes = budget.row_indices,
            endpoint_staging_bytes = budget.endpoint_staging,
            ring_bytes = budget.rings,
            runtime_headroom_bytes = budget.headroom,
            load_peak_bytes = budget.load_peak,
            serve_peak_bytes = budget.serve_peak,
            device_budget_bytes = config.device_budget,
            "explicit Spark replicas admission"
        );
    }
    if let Some(stream) = push {
        if let Some(weights) = push_native::load(library, catalog, config, resident, stream)? {
            return Ok((backend::Weights::Full(weights), config.device_budget - resident));
        }
    }
    let mut remaining = config.device_budget;
    let weights = super::load_in_rank_order(&layers, config.rank, |_, layer| {
        let started = std::time::Instant::now();
        let weight = ExpertWeights::load(
            library,
            catalog,
            config.selection(layer)?,
            remaining,
        )?;
        remaining = remaining
            .checked_sub(weight.budget().resident_bytes)
            .context("resident budget exhausted")?;
        tracing::info!(
            rank = config.rank,
            layer,
            resident_bytes = weight.budget().resident_bytes,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "native expert layer loaded"
        );
        Ok(weight)
    })?;
    log_spark_memory_if_enabled(library, config, "weights resident", None, None);
    Ok((backend::Weights::Full(weights), remaining))
}

/// Known explicit-topology peak bytes for the two live phases.
struct SparkAdmissionBudget {
    exchange: usize,
    row_indices: usize,
    endpoint_staging: usize,
    rings: usize,
    headroom: usize,
    load_peak: usize,
    serve_peak: usize,
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var_os(name) {
        None => Ok(default),
        Some(value) => value
            .to_str()
            .with_context(|| format!("{name} is not valid UTF-8"))?
            .trim()
            .parse::<usize>()
            .with_context(|| format!("{name} must be an unsigned integer")),
    }
}

#[cfg(test)]
fn align_up(value: usize, alignment: usize) -> Result<usize> {
    ensure!(alignment > 0, "ring alignment must be non-zero");
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .context("ring slot alignment overflow")
}

/// Registered/mapped RDMA ring bytes the Spark worker pins for `endpoints`
/// persistent ProtocolV2 endpoints. Mirrors the transport's
/// `verbs_host_persistent_rings` rule exactly for the native compact-BF16
/// request shape: one request ring and one response ring per endpoint, each
/// `depth` slots of `align_up(capacity, verbs-host alignment)`, with
/// `capacity = min(max(slot, wire_bytes), max_frame_bytes)`.
///
/// A request whose wire size exceeds the frame budget would be rejected by the
/// transport at connect time, so it fails admission here instead.
#[cfg(test)]
fn spark_ring_bytes(
    capacity: u32,
    max_frame_bytes: usize,
    depth: usize,
    slot_bytes: usize,
    alignment: usize,
    endpoints: usize,
) -> Result<usize> {
    spark_ring_bytes_for_ingress(capacity, max_frame_bytes, depth, slot_bytes, alignment, endpoints, false)
}

#[cfg(test)]
fn spark_ring_bytes_for_ingress(capacity: u32, max_frame_bytes: usize, depth: usize,
    slot_bytes: usize, alignment: usize, endpoints: usize, bf16: bool) -> Result<usize> {
    ensure!((1..=8).contains(&depth), "verbs-host ring depth must be in 1..=8");
    ensure!(endpoints > 0, "at least one RDMA endpoint is required");
    ensure!(slot_bytes > 0, "RDMA ring slot bytes must be non-zero");
    ensure!(max_frame_bytes > 0, "native frame budget must be non-zero");
    let (request_wire, response_wire) = cuteafd_transport::protocol_v2::compact_expert_wire_bytes(
        cuteafd_core::expert_geometry(), capacity, bf16)?;
    ensure!(
        request_wire <= max_frame_bytes && response_wire <= max_frame_bytes,
        "native compact-BF16 wire frames ({request_wire} request / {response_wire} response) \
         exceed the {max_frame_bytes}-byte native frame budget"
    );
    cuteafd_core::expert_geometry().compact_ring_bytes(capacity, max_frame_bytes, depth,
        slot_bytes, alignment, endpoints, bf16).context("registered ring span overflow")
}

fn selected_spark_io(config: &NativeExpertServiceConfig, geometry: cuteafd_core::ExpertGeometry,
    _rings: cuteafd_loader::placement::SparkRingGeometry) -> Result<cuteafd_loader::placement::SparkIoBytes> {
    let working = cuteafd_loader::placement::SparkWorkingSet {
        spark_layers: vec![0], spark_ranks: config.world,
        prefill_lanes: 1, ..Default::default()
    };
    ensure!(working.selected_ranks().contains(&config.rank), "Spark worker rank is not selected");
    let model = config.model_admission.as_ref().context("Spark I/O requires coordinator model admission")?;
    Ok(working.rank_endpoints_io(config.rank, geometry, &model.endpoints, model.rings)?)
}

fn worker_ring_geometry(config: &NativeExpertServiceConfig)
    -> Result<cuteafd_loader::placement::SparkRingGeometry> {
    let model = config.model_admission.as_ref().context("Spark rings require coordinator model admission")?;
    Ok(cuteafd_loader::placement::SparkRingGeometry {
        max_frame_bytes: model.rings.max_frame_bytes, depth: model.rings.depth,
        slot_bytes: model.rings.slot_bytes, endpoints: model.endpoints.len(),
        alignment: model.rings.alignment,
    })
}

/// Ring bytes for this worker's explicit selected topology and frame budget.
fn spark_transport_bytes(config: &NativeExpertServiceConfig) -> Result<usize> {
    let rings = worker_ring_geometry(config)?;
    let bytes = usize::try_from(selected_spark_io(config, cuteafd_core::expert_geometry(),
        rings)?.mapped_rings)?;
    Ok(bytes)
}

fn worker_ring_budget(config: &NativeExpertServiceConfig, geometry: cuteafd_core::ExpertGeometry)
    -> Result<Arc<cuteafd_transport::RingBudget>> {
    let rings = worker_ring_geometry(config)?;
    let limit = usize::try_from(selected_spark_io(config, geometry, rings)?.mapped_rings)?;
    Ok(cuteafd_transport::RingBudget::new(limit))
}

/// Shared ready-resident residual estimate from the seven-rank Spark census.
/// It is separate from known context/modules and applies to every family.
fn spark_runtime_headroom() -> Result<usize> {
    env_usize("CUTEAFD_SPARK_RUNTIME_HEADROOM_BYTES",
        usize::try_from(cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES)?)
}

fn spark_admission_budget(
    config: &NativeExpertServiceConfig,
    resident: usize,
    staging: usize,
    pinned_host: usize,
    read_scratch: usize,
    workspace: usize,
) -> Result<SparkAdmissionBudget> {
    spark_admission_budget_for_geometry(config, cuteafd_core::expert_geometry(),
        resident, staging, pinned_host, read_scratch, workspace)
}

fn spark_admission_budget_for_geometry(config: &NativeExpertServiceConfig,
    geometry: cuteafd_core::ExpertGeometry, resident: usize, staging: usize,
    pinned_host: usize, read_scratch: usize, workspace: usize) -> Result<SparkAdmissionBudget> {
    let io = selected_spark_io(config, geometry, worker_ring_geometry(config)?)?;
    let (exchange, row_indices) = if config.bf16_ingress || config.exlr8_manifest.is_some() {
        (usize::try_from(io.host_exchange)?, usize::try_from(io.row_indices)?)
    } else { (HostExpertExchange::bytes_for(config.capacity)?, config.capacity as usize * 4) };
    let rings = usize::try_from(io.mapped_rings)?;
    let endpoint_staging = usize::try_from(io.endpoint_staging)?;
    let headroom = spark_runtime_headroom()?;
    // Every execution slot owns a full execution workspace (stream, scratch,
    // inputs, staging) sized for the largest admitted unit, over the shared
    // resident weights.
    let slots = config.model_admission.as_ref().map_or(1, |model| usize::from(model.execution_slots));
    let workspace = workspace.checked_mul(slots).context("Spark execution slot workspace overflow")?;
    // The loader's pinned buffers and read scratch live only for one layer load;
    // the exchange, row-index scratch and registered rings live while serving.
    let load_peak = resident
        .checked_add(staging)
        .and_then(|bytes| bytes.checked_add(pinned_host))
        .and_then(|bytes| bytes.checked_add(read_scratch))
        .and_then(|bytes| bytes.checked_add(headroom))
        .context("Spark load peak overflow")?;
    let serve_peak = resident
        .checked_add(workspace)
        .and_then(|bytes| bytes.checked_add(exchange))
        .and_then(|bytes| bytes.checked_add(row_indices))
        .and_then(|bytes| bytes.checked_add(rings))
        .and_then(|bytes| bytes.checked_add(endpoint_staging))
        .and_then(|bytes| bytes.checked_add(headroom))
        .context("Spark serve peak overflow")?;
    Ok(SparkAdmissionBudget { exchange, row_indices, endpoint_staging, rings, headroom, load_peak, serve_peak })
}

/// Startup/load/serve-milestone memory diagnostics: the actual CUDA free/total
/// and one `/proc/meminfo` snapshot (`MemTotal`, `MemFree`, `MemAvailable`,
/// `Cached`, `Mlocked`, `Unevictable`, `SReclaimable`, all in KiB) on the
/// unified GB10 pool. Reclaimable page cache is reported, never added to the
/// permanent footprint; the extra fields separate host-available from UMA free.
///
/// `owned_connections` and `rings` are optional point-in-time observations; the
/// ring peak is an atomic high-water mark that concurrent admissions can raise,
/// so it is reported as a sampled value, not a reservation or a claim.
fn log_spark_memory(
    library: &NativeLibrary,
    config: &NativeExpertServiceConfig,
    stage: &str,
    owned_connections: Option<usize>,
    rings: Option<(usize, usize)>,
) {
    let cuda = library.cuda_physical_memory_info().ok();
    let allocatable = library.cuda_memory_info().ok();
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok();
    let field = |name: &str| -> Option<u64> {
        meminfo.as_deref()?.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == name).then(|| value.split_whitespace().next()?.parse::<u64>().ok())?
        })
    };
    tracing::info!(
        target: "cuteafd::spark_memory",
        rank = config.rank,
        stage,
        topology = ?config.declared_world.topology(),
        wire = ?config.declared_world.wire(),
        owned_connections = ?owned_connections,
        cuda_free_bytes = cuda.map(|(free, _)| free),
        cuda_free_scope = "raw cudaMemGetInfo; GB10 under-reports reclaimable page cache, use host MemAvailable for admission",
        cuda_total_bytes = cuda.map(|(_, total)| total),
        allocatable_free_bytes = allocatable.map(|(free, _)| free),
        host_mem_total_kib = field("MemTotal"),
        host_mem_free_kib = field("MemFree"),
        host_mem_available_kib = field("MemAvailable"),
        host_mem_cached_kib = field("Cached"),
        host_mem_mlocked_kib = field("Mlocked"),
        host_mem_unevictable_kib = field("Unevictable"),
        host_mem_reclaimable_kib = field("SReclaimable"),
        ring_budget_used_bytes_point_in_time = rings.map(|(used, _)| used),
        ring_budget_peak_bytes_point_in_time = rings.map(|(_, peak)| peak),
        "Spark unified memory state"
    );
}

/// Query and log only when INFO is enabled, so the new serve-time milestones do
/// not pay a `cudaMemGetInfo` plus `/proc/meminfo` read on every connection when
/// diagnostics are off. The guard uses the same explicit target as the emitted
/// event, so a per-target filter cannot let the guard fire while the event is
/// suppressed (or vice versa).
fn log_spark_memory_if_enabled(
    library: &NativeLibrary,
    config: &NativeExpertServiceConfig,
    stage: &str,
    owned_connections: Option<usize>,
    rings: Option<(usize, usize)>,
) {
    if tracing::enabled!(target: "cuteafd::spark_memory", tracing::Level::INFO) {
        log_spark_memory(library, config, stage, owned_connections, rings);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The planner's Spark I/O (`plan --layout`) against the worker's own
    /// sample: the expert service's selected rank I/O (per-endpoint rings,
    /// shared intake/exchange/indices) for the typed admission the
    /// coordinator sends it: capacity, one endpoint per MiMo prefill lane,
    /// and the wire format. MiMo Flash/Pro on their natural minimum
    /// Spark counts, FP8 and BF16 wires.
    #[test]
    fn mimo_planner_spark_io_equals_worker_sample() {
        use cuteafd_core::memory_layout::DeviceKind;
        use cuteafd_loader::plan::{layout::LayoutOptions, testing as fixture, ExpertPlacement, PlanOptions};
        for (pro, ranks) in [(false, 2usize), (true, 4)] {
            let dir = tempfile::tempdir().unwrap();
            let mut cfg = if pro { fixture::mimo_pro_config() } else { fixture::mimo_flash_mopd_config() };
            cfg["max_position_embeddings"] = serde_json::json!(1 << 20);
            fixture::write_snapshot(dir.path(), &cfg, &if pro { fixture::mimo_pro_tensors() }
                else { fixture::mimo_flash_mopd_tensors() }, Some(if pro { 8 } else { 4 }));
            let geometry = if pro { cuteafd_core::ExpertGeometry::MIMO_V26_PRO } else { cuteafd_core::ExpertGeometry::MIMO_V2_FLASH };
            for input in [cuteafd_loader::families::mimo_v2::MimoExpertInput::Fp8,
                cuteafd_loader::families::mimo_v2::MimoExpertInput::Bf16] {
                let report = cuteafd_loader::plan::plan(dir.path(), &PlanOptions {
                    placement: ExpertPlacement::Sparks { ranks },
                    layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), pool_tokens: Some(32768), concurrency: 4, mimo_expert_input: input,
                        onboard: Some(cuteafd_loader::placement::Onboard::Layers(0)), ..Default::default() }),
                    ..Default::default() }).unwrap();
                let layout = report.memory_layout.unwrap();
                let lanes = cuteafd_loader::families::mimo_v2::admission::resolve_transport_lanes(true,
                    if pro { "mimop" } else { "mimo" }, None).unwrap();
                let sparks = layout.devices.iter().filter(|d| d.kind == DeviceKind::Spark).collect::<Vec<_>>();
                assert_eq!(sparks.len(), ranks, "pro={pro} {input:?}: {}", layout.render());
                for device in sparks {
                    let planned = |group: &str| device.items.iter().find(|i| i.group == group)
                        .unwrap_or_else(|| panic!("pro={pro} {input:?} {}: no {group}", device.name())).bytes;
                    // The worker's typed admission as the coordinator selects it: one endpoint per
                    // MiMo prefill lane (the runtime's own lane resolution), at its wire format.
                    let mut worker = config(ranks, device.index as usize, None);
                    worker.capacity = 4096;
                    worker.rdma_endpoints = lanes;
                    worker.bf16_ingress = input.bf16(false);
                    set_model(&mut worker, 4096, input.bf16(false), lanes);
                    let rings = worker_ring_geometry(&worker).unwrap();
                    let sample = selected_spark_io(&worker, geometry, rings).unwrap();
                    assert_eq!(planned("rdma rings"), sample.mapped_rings, "pro={pro} {input:?} {}", device.name());
                    assert_eq!(planned("expert intake"), sample.intake, "pro={pro} {input:?}");
                    assert_eq!(planned("host expert exchange"), sample.host_exchange, "pro={pro} {input:?}");
                    assert_eq!(planned("expert row indices"), sample.row_indices, "pro={pro} {input:?}");
                }
            }
        }
    }

    fn config(world: usize, rank: usize, topology: Option<SparkTopology>) -> NativeExpertServiceConfig {
        NativeExpertServiceConfig {
            library: PathBuf::from("/native.so"),
            exl3_aot_dir: None,
            exl3_schedule: Default::default(),
            fp8_package: None,
            snapshot: PathBuf::from("/model"),
            exlr8_manifest: None,
            rank,
            world,
            model_admission: Some(cuteafd_transport::worker_selection::ModelAdmission {
                capacity: 16, compiled_capacity: 16,
                endpoints: vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity: 16,
                    ingress: cuteafd_core::SparkIngress::Fp8K32 }; 2],
                rings: cuteafd_core::SparkRings { max_frame_bytes: 64 << 20,
                    depth: 8, slot_bytes: 8 << 20, alignment: 4096 }, execution_slots: 1
            }),
            selected_layers: None,
            capacity: 16,
            device_budget: 1 << 40,
            max_frame_bytes: 64 << 20,
            rdma_endpoints: 2,
            declared_world: crate::shared::spark_topology::world(topology, world).unwrap(),
            native_spark_tp2: false,
            bf16_ingress: false,
            encoder: None,
            audio_encoder: None,
        }
    }

    #[test]
    #[ignore = "CPU-only isolated metadata startup RSS; requires CUTEAFD_KIMI_EXL3_SNAPSHOT"]
    fn mounted_worker_metadata_host_rss() -> Result<()> {
        let snapshot = PathBuf::from(std::env::var_os("CUTEAFD_KIMI_EXL3_SNAPSHOT").context("checkpoint")?);
        let sample = |phase: &str| -> Result<(usize, usize)> {
            let status = std::fs::read_to_string("/proc/self/status")?;
            let value = |key| -> Result<usize> {
                Ok(status.lines().find_map(|line| line.strip_prefix(key)).context("missing RSS")?
                    .split_whitespace().next().context("empty RSS")?.parse::<usize>()? * 1024)
            };
            let (rss, peak) = (value("VmRSS:")?, value("VmHWM:")?);
            eprintln!("worker metadata CPU RSS: phase={phase} rss={rss} peak={peak}");
            Ok((rss, peak))
        };
        sample("start")?;
        let inventory = cuteafd_loader::plan::Checkpoint::inventory(&snapshot)?;
        sample("inventory")?;
        let mut catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
        catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
        let geometry = catalog.routed_experts().geometry()?;
        cuteafd_core::set_expert_geometry(geometry).map_err(|_| anyhow::anyhow!("run alone"))?;
        drop(inventory);
        sample("catalog_before_native")?;
        let _identity = cuteafd_transport::worker_selection::WorkerIdentity::from_snapshot(&snapshot,
            geometry, SparkWorld::canonical(6)?)?;
        sample("identity")?;
        let mut worker = config(6, 0, None);
        worker.snapshot = snapshot;
        worker.selected_layers = Some((6..93).collect());
        validate_topology(&worker, &catalog)?;
        sample("topology")?;
        // Worker selection validates once; load_weights validates again immediately before trimming.
        validate_topology(&worker, &catalog)?;
        cuteafd_ffi::trim_startup_metadata();
        let (rss, peak) = sample("startup_complete_trimmed")?;
        ensure!(peak < 2usize << 30, "worker metadata peak exceeds 2 GiB: {peak}");
        ensure!(rss < 512usize << 20, "retained worker metadata exceeds 512 MiB: {rss}");
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only mounted Kimi checkpoint/package planning; run alone with CUTEAFD_KIMI_EXL3_SNAPSHOT and CUTEAFD_KIMI_SPARK_PACKAGE"]
    fn mounted_kimi_all_spark_golden_demand() -> Result<()> {
        use crate::shared::experts::exl3::{Exl3Weights, worker::Exl3Worker};
        cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
            .map_err(|_| anyhow::anyhow!("run Kimi demand test alone"))?;
        let snapshot = PathBuf::from(std::env::var_os("CUTEAFD_KIMI_EXL3_SNAPSHOT").context("Kimi snapshot")?);
        let package = PathBuf::from(std::env::var_os("CUTEAFD_KIMI_SPARK_PACKAGE").context("Kimi Spark package")?);
        let mut catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
        catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
        let available = std::env::var("CUTEAFD_KIMI_MIN_AVAILABLE").ok().map(|s| s.parse::<usize>()).transpose()?;
        let measured_host = std::env::var("CUTEAFD_KIMI_MEASURED_HOST_BYTES").ok()
            .map(|s| s.parse::<usize>()).transpose()?;
        let mut minimum_onboard = None;
        for onboard in 0..=6 {
            let layers = (1 + onboard..93).collect::<Vec<_>>();
            let mut worst_peak = 0;
            for rank in 0..6 {
                for capacity in [512, 4096] {
                    let mut worker = config(6, rank, None);
                    worker.capacity = capacity;
                    worker.rdma_endpoints = 1;
                    worker.selected_layers = Some(layers.clone());
                    let model = worker.model_admission.as_mut().unwrap();
                    model.capacity = capacity;
                    model.compiled_capacity = 4096;
                    model.endpoints = vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity,
                        ingress: cuteafd_core::SparkIngress::Fp8K32 }];
                    let workspace = Exl3Worker::plan(&package.join(format!("tp6-rank{rank}")), capacity,
                        Default::default())?;
                    let mut resident = 0;
                    let (mut staging, mut pinned, mut scratch) = (0, 0, 0);
                    for &layer in &layers {
                        let plan = Exl3Weights::plan(&catalog, worker.exl3_selection(layer)?)?;
                        resident += plan.resident_bytes;
                        staging = staging.max(plan.device_staging_bytes);
                        pinned = pinned.max(plan.pinned_host_bytes);
                        scratch = scratch.max(plan.read_scratch_bytes);
                    }
                    let peak = spark_admission_budget(&worker, resident, staging, pinned, scratch, workspace)?;
                    let planned_peak = peak.serve_peak.max(peak.load_peak);
                    let measured_host_peak = if let Some(host) = measured_host {
                        planned_peak.checked_sub(peak.headroom).and_then(|bytes| bytes.checked_add(host))
                            .context("measured host forecast overflow")?
                    } else { planned_peak };
                    worst_peak = worst_peak.max(measured_host_peak);
                    eprintln!("Kimi CPU golden demand: onboard={onboard} rank={rank} rows={capacity} layers={} resident={resident} workspace={workspace} exchange={} indices={} rings={} estimated_runtime={} load_peak={} serve_peak={} staging={staging} pinned={pinned} read_scratch={scratch}",
                        layers.len(), peak.exchange, peak.row_indices, peak.rings, peak.headroom, peak.load_peak, peak.serve_peak);
                    assert!(peak.serve_peak >= resident + workspace && peak.load_peak >= resident);
                }
            }
            if let Some(host) = measured_host {
                eprintln!("Kimi CPU measured-host forecast: onboard={onboard} host_bytes={host} worst_peak={worst_peak}");
            }
            if let Some(available) = available {
                let fits = worst_peak.checked_add(4usize << 30).is_some_and(|required| required <= available);
                eprintln!("Kimi CPU fit: onboard={onboard} worst_peak={worst_peak} available={available} margin={} fits_4gib={fits}",
                    available.saturating_sub(worst_peak));
                if fits && minimum_onboard.is_none() { minimum_onboard = Some(onboard); }
            }
        }
        if available.is_some() {
            assert!(minimum_onboard.is_some(), "no proposed onboard count preserves 4 GiB");
            eprintln!("Kimi CPU minimum onboard at sampled availability: {}", minimum_onboard.unwrap());
        }
        Ok(())
    }

    #[test]
    fn worker_refuses_unqualified_recipe_after_selection_before_weight_load() -> Result<()> {
        use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        let root = tempfile::tempdir()?;
        let (tensors, _) = qwen4_exl3(3, 4);
        write_snapshot(root.path(), &qwen4_config(3), &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        use cuteafd_transport::worker_selection::{WorkerIdentity, WorkerSelection};
        let admit = |world| -> Result<()> {
            let mut config = config(world, 0, None);
            // Sparse header fixture and no native library: this is the production
            // admission boundary before package warming or weight allocation.
            assert!(!config.library.exists());
            let identity = WorkerIdentity::from_snapshot(root.path(), catalog.routed_experts().geometry()?,
                config.spark_world()?)?;
            let selection = WorkerSelection::new(identity, &[0, 1, 2])?
                .with_model(config.model_admission.clone().unwrap())?;
            local::admit_worker_selection(&mut config, &selection, &catalog)
        };
        admit(3)?;
        let error = admit(5).unwrap_err();
        assert!(error.downcast_ref::<cuteafd_loader::plan::experts::SparkRecipeRefusal>().is_some(),
            "{error:#}");
        Ok(())
    }

    #[test]
    fn exlr8_worker_admits_only_selected_routed_layers_when_supplied() -> Result<()> {
        use cuteafd_transport::worker_selection::{WorkerIdentity, WorkerSelection};
        let Some(snapshot) = std::env::var_os("CUTEAFD_EXLR8_TEST_SNAPSHOT") else { return Ok(()); };
        let snapshot = PathBuf::from(snapshot);
        let mut catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
        catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
        let shape = catalog.routed_experts();
        let mut worker = config(4, 0, None);
        worker.snapshot = snapshot.clone();
        worker.exlr8_manifest = Some("up1".into());
        for endpoint in &mut worker.model_admission.as_mut().unwrap().endpoints {
            endpoint.ingress = cuteafd_core::SparkIngress::Fp8K32;
        }
        let identity = WorkerIdentity::from_snapshot_manifest(&snapshot, shape.geometry()?,
            worker.spark_world()?, Some("up1"))?;
        let selection = |layers: &[usize]| -> Result<WorkerSelection> {
            Ok(WorkerSelection::new(identity.clone(), layers)?
                .with_model(worker.model_admission.clone().unwrap())?)
        };
        let valid = selection(&[shape.first_layer, shape.first_layer + 2])?;
        let idle = selection(&[])?;
        let dense = selection(&[0])?;
        assert!(selection(&[shape.layers]).is_err());
        local::admit_worker_selection(&mut worker, &valid, &catalog)?;
        assert_eq!(worker.resident_layers(shape.layers)?, valid.layers());
        assert!(local::admit_worker_selection(&mut worker, &dense, &catalog).is_err());
        let mut owned = config(4, 0, Some(SparkTopology::NATIVE_TP4_EP1));
        assert!(local::admit_worker_selection(&mut owned, &valid, &catalog).is_err());
        // Idle identity acknowledgement intentionally needs no storage recipe.
        local::admit_worker_selection(&mut worker, &idle, &catalog)?;
        assert!(worker.resident_layers(shape.layers)?.is_empty());
        Ok(())
    }

    fn set_model(config: &mut NativeExpertServiceConfig, capacity: u32, bf16: bool, endpoints: usize) {
        config.model_admission = Some(cuteafd_transport::worker_selection::ModelAdmission {
            capacity, compiled_capacity: capacity,
            endpoints: vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity, ingress: if bf16 {
                cuteafd_core::SparkIngress::Bf16
            } else { cuteafd_core::SparkIngress::Fp8K32 } }; endpoints],
            rings: cuteafd_core::SparkRings { max_frame_bytes: 64 << 20,
                depth: 8, slot_bytes: 8 << 20, alignment: 4096 }, execution_slots: 1
        });
    }

    #[test]
    fn explicit_topology_maps_every_physical_rank_to_its_local_shard() {
        for (tp, ep) in [(2u8, 1u8), (3, 1), (4, 1), (2, 2), (3, 2), (2, 3), (6, 1)] {
            let topology = SparkTopology::new(tp, ep).unwrap();
            for rank in 0..topology.world_size() {
                let selection = config(topology.world_size(), rank, Some(topology))
                    .selection(7)
                    .unwrap();
                assert_eq!(selection.layer(), 7);
                if tp == 4 {
                    assert_eq!(selection, ExpertLayer::Backbone { layer: 7, rank: rank });
                } else {
                    assert_eq!(
                        selection,
                        ExpertLayer::BackboneReplicatedTp {
                            layer: 7,
                            rank: rank % tp as usize,
                            world: tp as usize,
                        }
                    );
                    assert_eq!(selection.role(), match tp {
                        2 => crate::shared::spark_topology::SPARK_TP2_ROLE,
                        3 => crate::shared::spark_topology::SPARK_TP3_ROLE,
                        // Pure TP6EP1 is its own native shard family: six
                        // disjoint intermediate slices of every expert.
                        6 => crate::shared::spark_topology::SPARK_TP6_ROLE,
                        other => panic!("unexpected explicit TP degree {other}"),
                    });
                    assert_eq!(
                        selection.expert(3),
                        cuteafd_loader::V41ExpertSelection::BackboneTp {
                            layer: 7,
                            expert: 3,
                            rank: rank % tp as usize,
                            world: tp as usize,
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn spark_exl3_uses_compressed_shards_for_every_registered_world() {
        for world in [1, 2, 3, 4, 6] {
            for rank in 0..world {
                assert_eq!(config(world, rank, None).exl3_selection(7).unwrap(),
                    ExpertLayer::BackboneExl3Tp { layer: 7, rank, world });
            }
        }
    }

    #[test]
    fn canonical_tp1_selects_whole_exl3_experts_and_rank_package() {
        let one = config(1, 0, None);
        assert_eq!(one.selection(7).unwrap(), ExpertLayer::BackboneExl3Tp { layer: 7, rank: 0, world: 1 });
        assert!(one.selection(7).unwrap().role() > 7);
        assert_eq!(one.native_group().unwrap(), None);
        assert!(one.exl3_directory_for(&[4, 5], "tp1-rank0").ends_with("tp1-rank0"));
        assert!(one.exl3_directory_for(&[4, 5], "tp6-compact-rank0").ends_with("tp6-compact-rank0"));
        let mut explicit = one;
        explicit.exl3_aot_dir = Some("/custom-export".into());
        assert_eq!(explicit.exl3_directory_for(&[4, 5], "tp6-compact-rank0"), PathBuf::from("/custom-export"));
    }

    #[test]
    fn unqualified_transport_worlds_are_rejected_before_shard_selection() {
        for world in [0, cuteafd_core::MAX_SPARK_WORLD + 1] {
            assert!(crate::shared::spark_topology::world(None, world).is_err(), "TP{world}");
        }
        for world in [5, 7, 8] {
            for rank in 0..world {
                assert_eq!(config(world, rank, None).selection(7).unwrap(),
                    ExpertLayer::BackboneExl3Tp { layer: 7, rank, world });
            }
        }
    }

    #[test]
    fn canonical_selection_is_unchanged_and_topology_must_agree() {
        assert_eq!(config(4, 3, None).selection(2).unwrap(), ExpertLayer::Backbone { layer: 2, rank: 3 });
        assert_eq!(config(2, 1, None).selection(2).unwrap(), ExpertLayer::BackboneTp2 { layer: 2, rank: 1 });
        // Canonical TP3 selects generic compressed shards independently of
        // the ownership-wire native TP3 shard family.
        assert_eq!(
            config(3, 2, None).selection(2).unwrap(),
            ExpertLayer::BackboneExl3Tp { layer: 2, rank: 2, world: 3 }
        );
        // The explicit topology must match the launched rank count.
        let topology = SparkTopology::new(2, 2).unwrap();
        let mut mismatch = config(4, 0, Some(topology));
        mismatch.world = 6;
        assert!(mismatch.spark_world().is_err());
        // A rank outside the topology is rejected before selection is used.
        assert!(config(4, 4, Some(topology)).selection(0).is_err());
        // A native three-rank launch therefore has to carry its topology: the
        // explicit path stays on the native shard family, never the EXL3 layer.
        let tp3ep1 = SparkTopology::new(3, 1).unwrap();
        assert_eq!(
            config(3, 2, Some(tp3ep1)).selection(2).unwrap(),
            ExpertLayer::BackboneReplicatedTp { layer: 2, rank: 2, world: 3 }
        );
    }

    #[test]
    fn compact_exl3_topology_keeps_shards_packages_and_group_zero() {
        for rank in 0..3 {
            let config = config(3, rank, Some(SparkTopology::NATIVE_TP3_EP1));
            assert_eq!(config.exl3_selection(7).unwrap(),
                ExpertLayer::BackboneExl3Tp { layer: 7, rank, world: 3 });
            assert!(config.exl3_directory_for(&[2, 3], &format!("tp3-rank{rank}")).ends_with(format!("tp3-rank{rank}")));
            assert_eq!(config.native_group().unwrap(), Some(0));
        }
        let config = config(6, 0, Some(SparkTopology::NATIVE_TP3_EP2));
        assert!(config.exl3_selection(7).is_err());
    }

    #[test]
    fn flash_native_tp2_selects_spark_role_without_changing_wire_protocol() {
        for rank in 0..2 {
            let mut flash = config(2, rank, None);
            flash.native_spark_tp2 = true;
            let layer = flash.selection(7).unwrap();
            assert_eq!(layer, ExpertLayer::BackboneReplicatedTp { layer: 7, rank, world: 2 });
            assert_eq!(layer.role(), crate::shared::spark_topology::SPARK_TP2_ROLE);
            assert!(flash.declared_world.ownership_topology().is_none());
            assert_eq!(flash.native_group().unwrap(), None);
        }
    }

    /// A compressed shard must never be routable to the native expert family:
    /// that would silently substitute the native weight format for EXL3.
    #[test]
    fn exl3_compact_shard_reports_no_native_role() {
        for rank in 0..3 {
            let shard = config(3, rank, None).selection(7).unwrap();
            assert_eq!(shard.layer(), 7);
            // Native roles occupy 0..=7; the sentinel is outside that range, so
            // every `info.role == layer.role()` guard fails loudly instead of
            // matching the spark_tp3 native role by accident.
            assert!(shard.role() > 7, "EXL3 shard reported native role {}", shard.role());
            assert_ne!(shard.role(), crate::shared::spark_topology::SPARK_TP3_ROLE);
            assert_eq!(shard, ExpertLayer::BackboneExl3Tp { layer: 7, rank, world: 3 });
        }
    }

    #[test]
    #[should_panic(expected = "never stage native expert weights")]
    fn exl3_compact_shard_cannot_select_native_staging() {
        let _ = ExpertLayer::BackboneExl3Tp { layer: 0, rank: 0, world: 3 }.expert(0);
    }

    /// Both phase peaks include the shared measured-residual estimate, alongside
    /// their own disjoint source-known transient and retained owners.
    #[test]
    fn explicit_admission_counts_every_known_transient_exactly() {
        let topology = SparkTopology::new(2, 2).unwrap();
        let mut config = config(4, 0, Some(topology));
        config.capacity = 4096;
        set_model(&mut config, 4096, false, 2);
        let headroom = spark_runtime_headroom().unwrap();
        let rings = spark_transport_bytes(&config).unwrap();
        let budget = spark_admission_budget(&config, 10, 1, 2, 3, 4).unwrap();
        let exchange = HostExpertExchange::bytes_for(4096).unwrap();
        assert_eq!(exchange, 4096 * (2 * 6 * 4 + 5120 * 2));
        assert_eq!(budget.exchange, exchange);
        assert_eq!(budget.row_indices, 4096 * 4);
        assert_eq!(budget.rings, rings);
        assert_eq!(budget.load_peak, 10 + 1 + 2 + 3 + headroom);
        assert_eq!(budget.endpoint_staging, 2 * (128 + 4096 * 4));
        assert_eq!(budget.serve_peak, 10 + 4 + exchange + 4096 * 4 + rings + budget.endpoint_staging + headroom);
        // The registered rings and the 40 MiB host exchange dominate the serve
        // peak at full capacity.
        assert!(budget.serve_peak > budget.load_peak);
        assert!(rings > exchange);
        assert_eq!(HostExpertExchange::bytes_for(16).unwrap(), 16 * (48 + 10240));
    }

    /// Every execution slot owns one execution workspace; every other term
    /// (rings, exchange, staging) is unchanged by the slot count.
    #[test]
    fn each_execution_slot_charges_one_execution_workspace() {
        let topology = SparkTopology::new(2, 2).unwrap();
        let mut config = config(4, 0, Some(topology));
        config.capacity = 4096;
        set_model(&mut config, 4096, false, 3);
        let one = spark_admission_budget(&config, 10, 1, 2, 3, 4).unwrap();
        for slots in 2..=4u8 {
            config.model_admission.as_mut().unwrap().execution_slots = slots;
            let pool = spark_admission_budget(&config, 10, 1, 2, 3, 4).unwrap();
            assert_eq!(pool.serve_peak - one.serve_peak, 4 * (u64::from(slots) as usize - 1));
            assert_eq!((pool.load_peak, pool.rings, pool.exchange), (one.load_peak, one.rings, one.exchange));
        }
    }

    #[test]
    fn registered_ring_bytes_match_the_transport_sizing_rule() {
        let alignment = 4096;
        let depth = 8;
        let slot = 8 << 20;
        let max_frame = 64 << 20;
        // At capacity 4096 the wire frames exceed the 8 MiB minimum slot.
        let request_wire = 96 + 4096 * (40 + 6 * 12 + 5280);
        let response_wire = 96 + 4096 * (4 + 5120 * 2);
        let expected = 2 * depth * (align_up(request_wire, alignment).unwrap()
            + align_up(response_wire, alignment).unwrap());
        assert_eq!(
            spark_ring_bytes(4096, max_frame, depth, slot, alignment, 2).unwrap(),
            expected
        );
        assert!(expected > 900 << 20 && expected < 1100 << 20, "{expected}");
        // At small capacity the 8 MiB minimum slot dominates: 2 endpoints x
        // (64 MiB request + 64 MiB response) = 256 MiB.
        assert_eq!(
            spark_ring_bytes(1, max_frame, depth, slot, alignment, 2).unwrap(),
            256 << 20
        );
        assert_eq!(
            spark_ring_bytes(1, max_frame, depth, slot, alignment, 1).unwrap(),
            128 << 20
        );
        // Depth, endpoints, slot and frame budget are validated.
        assert!(spark_ring_bytes(1, max_frame, 0, slot, alignment, 2).is_err());
        assert!(spark_ring_bytes(1, max_frame, 9, slot, alignment, 2).is_err());
        assert!(spark_ring_bytes(1, max_frame, depth, 0, alignment, 2).is_err());
        assert!(spark_ring_bytes(1, max_frame, depth, slot, alignment, 0).is_err());
        assert!(spark_ring_bytes(1, max_frame, depth, slot, 0, 2).is_err());
        // A frame that cannot fit the configured budget fails before connect.
        assert!(spark_ring_bytes(4096, 1 << 20, depth, slot, alignment, 2).is_err());
    }

    #[test]
    fn explicit_admission_peak_overflow_is_rejected() {
        let topology = SparkTopology::new(3, 2).unwrap();
        let mut config = config(6, 0, Some(topology));
        config.capacity = 16;
        assert!(spark_admission_budget(&config, usize::MAX, 1, 0, 0, 0).is_err());
        assert!(spark_admission_budget(&config, 1, usize::MAX, 1, 1, 1).is_err());
        assert!(spark_admission_budget(&config, 1, 0, 0, 0, usize::MAX).is_err());
        assert!(HostExpertExchange::bytes_for(0).is_err());
        assert!(HostExpertExchange::bytes_for(4097).is_err());
    }

    #[test]
    fn host_exchange_allocation_matches_its_declared_extents() {
        for capacity in [1u32, 16, 80, 256, 1024, 4096] {
            let exchange = HostExpertExchange::new(capacity).unwrap();
            assert_eq!(exchange.ids.len(), capacity as usize * 6);
            assert_eq!(exchange.routing.len(), capacity as usize * 6);
            assert_eq!(exchange.partials.len(), capacity as usize * 5120 * 2);
            assert_eq!(
                HostExpertExchange::bytes_for(capacity).unwrap(),
                capacity as usize * 6 * 8 + capacity as usize * 10240
            );
        }
    }

    #[test]
    fn step5_two_rank_clients_admit_all_rings_and_shared_staging_before_allocation() -> Result<()> {
        use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress, SparkRings};
        use cuteafd_loader::placement::SparkWorkingSet;
        use cuteafd_transport::worker_selection::ModelAdmission;
        let geometry = ExpertGeometry::STEP5;
        let endpoint = SparkEndpoint { ring_depth: None, capacity: 640, ingress: SparkIngress::Bf16 };
        let rings = SparkRings { max_frame_bytes: 64 << 20, depth: 8,
            slot_bytes: 8 << 20, alignment: 4096 };
        let working = SparkWorkingSet { spark_layers: (3..91).collect(), spark_ranks: 6,
            prefill_lanes: 1, ..Default::default() };
        for rank in 0..6 {
            let mut config = config(6, rank, None);
            config.capacity = 640;
            config.bf16_ingress = true;
            config.device_budget = 104 << 30;
            let mut previous = None;
            for clients in [1, 2] {
                let model = ModelAdmission::for_clients(geometry, endpoint, 4096, rings, clients)?;
                let forecast = working.rank_endpoints_io(rank, geometry, &model.endpoints, rings)?;
                config.rdma_endpoints = clients;
                config.model_admission = Some(model);
                assert_eq!(selected_spark_io(&config, geometry, worker_ring_geometry(&config)?)?, forecast);
                assert_eq!(forecast.intake, 640 * (4096 * 4 + 8 * 8));
                assert_eq!(forecast.host_exchange, 640 * (4096 * 2 + 8 * 8));
                assert_eq!(forecast.row_indices, 640 * 4);
                assert_eq!(forecast.mapped_rings, clients as u64 * 134217728);
                assert_eq!(forecast.endpoint_staging, clients as u64 * 2688);
                // Retained bytes and maximum load transient from the first
                // diagnostic; runtime residual remains an estimate, not a census.
                let budget = spark_admission_budget_for_geometry(&config, geometry,
                    100965469184, 0, 5033164800, 0, 199584768)?;
                assert!(199584768 >= forecast.intake);
                assert_eq!(budget.rings as u64, forecast.mapped_rings);
                assert_eq!(budget.exchange as u64, forecast.host_exchange);
                assert_eq!(budget.row_indices as u64, forecast.row_indices);
                assert_eq!(budget.endpoint_staging as u64, forecast.endpoint_staging);
                assert_eq!(budget.load_peak, 100965469184 + 5033164800 + budget.headroom);
                assert!(budget.load_peak <= config.device_budget);
                assert!(budget.serve_peak <= config.device_budget);
                assert!(budget.load_peak < 108 << 30 && budget.serve_peak < 108 << 30);
                if let Some(old) = previous {
                    assert_eq!(budget.serve_peak - old, 134217728 + 2688);
                }
                previous = Some(budget.serve_peak);
                let ring_budget = worker_ring_budget(&config, geometry)?;
                let reservations = (0..clients).map(|_| ring_budget.reserve(134217728))
                    .collect::<Result<Vec<_>>>()?;
                assert_eq!(ring_budget.used(), budget.rings);
                assert!(ring_budget.reserve(1).is_err());
                drop(reservations);
                assert_eq!(ring_budget.used(), 0);
            }
        }
        Ok(())
    }

    #[test]
    fn typed_v41_ring_budget_covers_three_or_four_endpoints() {
        let geometry = cuteafd_core::ExpertGeometry::DEEPSEEK_V41;
        for device in [1, 2] {
            let mut config = config(4, 0, None);
            let mut endpoints = vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity: 1024,
                ingress: cuteafd_core::SparkIngress::Fp8K32 }; 2];
            endpoints.extend(std::iter::repeat_n(cuteafd_core::SparkEndpoint { ring_depth: None, capacity: 80,
                ingress: cuteafd_core::SparkIngress::Fp8K32 }, device));
            let model = cuteafd_transport::worker_selection::ModelAdmission {
                capacity: 1024, compiled_capacity: 1024, endpoints,
                rings: cuteafd_core::SparkRings { max_frame_bytes: 64 << 20,
                    depth: 8, slot_bytes: 8 << 20, alignment: 4096 }, execution_slots: 1
            };
            model.validate(geometry).unwrap();
            let exact = model.ring_bytes(geometry).unwrap();
            config.model_admission = Some(model);
            let budget = worker_ring_budget(&config, geometry).unwrap();
            assert_eq!(budget.limit(), exact);
            let all = budget.reserve(exact).unwrap();
            assert!(budget.reserve(1).is_err());
            drop(all);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn v41_replacement_sequence_enforces_budget_on_both_topology_paths() {
        for topology in [None, Some(SparkTopology::new(3, 1).unwrap())] {
            for bf16 in [false, true] {
                let mut config = config(3, 0, topology);
                config.capacity = 1024;
                config.bf16_ingress = bf16;
                set_model(&mut config, 1024, bf16, 2);
                let budget = worker_ring_budget(&config, cuteafd_core::ExpertGeometry::DEEPSEEK_V41).unwrap();
                let bytes = spark_ring_bytes_for_ingress(1024, 64 << 20, 8, 8 << 20, 4096, 1, bf16).unwrap();
                let decode = budget.reserve(bytes).unwrap();
                let prefill = budget.reserve(bytes).unwrap();
                assert_eq!(budget.used(), budget.limit());
                assert!(budget.reserve(bytes).is_err());
                drop(decode);
                let replacement = budget.reserve(bytes).unwrap();
                assert_eq!(budget.peak(), 2 * bytes);
                drop((replacement, prefill));
                assert_eq!(budget.used(), 0);
            }
        }
    }

    #[test]
    fn spark_io_equals_actual_transport_allocation_for_every_family() {
        use cuteafd_core::ExpertGeometry as G;
        use cuteafd_loader::placement::{SparkRingGeometry, SparkWorkingSet};
        let cases = [(G::DEEPSEEK_V41, 2), (G::DEEPSEEK_V4_FLASH, 2),
            (G::DEEPSEEK_V4_PRO, 2), (G::GLM5, 4), (G::GLM5_FLASH, 2),
            (G::QWEN4, 1), (G::MIMO_V2_FLASH, 2), (G::MIMO_V26_PRO, 3)];
        let working = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
            prefill_lanes: 3, ..SparkWorkingSet::default() };
        for (geometry, endpoints) in cases {
            for bf16 in [false, true] {
                for capacity in [1, 1024, 2048, 4096] {
                    for depth in [1, 4, 8] {
                        let rings = SparkRingGeometry { max_frame_bytes: 64 << 20, depth,
                            slot_bytes: 8 << 20, alignment: 4096, endpoints };
                        let (request, response) = cuteafd_transport::protocol_v2::compact_expert_wire_bytes(
                            geometry, capacity, bf16).unwrap();
                        let actual = cuteafd_transport::verbs_host_persistent_ring_bytes(
                            rings.max_frame_bytes, request, response, rings.slot_bytes,
                            rings.alignment, rings.depth).unwrap() * endpoints;
                        let planned = working.rank_io(0, geometry, capacity, bf16, rings)
                            .unwrap().mapped_rings;
                        assert_eq!(planned, actual as u64,
                            "geometry={geometry:?} bf16={bf16} capacity={capacity} depth={depth} endpoints={endpoints}");
                        if geometry == G::GLM5 && bf16 && capacity == 4096 && depth == 8 {
                            assert_eq!(actual, 3_239_837_696);
                        }
                    }
                }
            }
        }
    }

    /// CPU geometry/reservation coverage, not a substitute for measured live
    /// registered-memory peaks on each min/max deployment.
    #[test]
    fn selected_endpoint_planner_runtime_and_peak_agree_across_family_layouts() {
        use cuteafd_core::{ExpertGeometry as G, SparkEndpoint, SparkIngress};
        use cuteafd_loader::placement::SparkWorkingSet;
        for geometry in [G::DEEPSEEK_V41, G::DEEPSEEK_V4_FLASH, G::DEEPSEEK_V4_PRO,
            G::GLM5, G::GLM5_FLASH, G::QWEN4, G::MIMO_V2_FLASH, G::MIMO_V26_PRO] {
            for world in [3, 4, 6] {
                for rtx in [1, 2] {
                    let mut config = config(world, 0, None);
                    config.capacity = 4096;
                    let mut endpoints = vec![SparkEndpoint { ring_depth: None, capacity: 4096,
                        ingress: SparkIngress::Bf16Decode { rows: 80 } }];
                    endpoints.extend(std::iter::repeat_n(SparkEndpoint { ring_depth: None, capacity: 80,
                        ingress: SparkIngress::Fp8K32 }, rtx));
                    endpoints.push(SparkEndpoint { ring_depth: Some(2), capacity: 1024, ingress: SparkIngress::Fp8K32 });
                    config.model_admission.as_mut().unwrap().endpoints = endpoints.clone();
                    config.model_admission.as_mut().unwrap().capacity = 4096;
                    config.model_admission.as_mut().unwrap().compiled_capacity = 4096;
                    let model = config.model_admission.as_ref().unwrap();
                    model.validate(geometry).unwrap();
                    let working = SparkWorkingSet { spark_layers: vec![3], spark_ranks: world,
                        prefill_lanes: 1, ..Default::default() };
                    let planned = working.rank_endpoints_io(0, geometry, &endpoints, model.rings).unwrap();
                    let budget = worker_ring_budget(&config, geometry).unwrap();
                    let mut reservations = Vec::new();
                    for endpoint in endpoints {
                        let (request, response) = endpoint.wire_bytes(geometry).unwrap();
                        let rings = endpoint.resolved_rings(model.rings).unwrap();
                        let actual = cuteafd_transport::verbs_host_persistent_ring_bytes(
                            rings.max_frame_bytes, request, response, rings.slot_bytes,
                            rings.alignment, rings.depth).unwrap();
                        reservations.push(budget.reserve(actual).unwrap());
                    }
                    assert_eq!(budget.limit() as u64, planned.mapped_rings);
                    assert_eq!(budget.used(), budget.limit());
                    assert_eq!(budget.peak(), budget.limit());
                    assert!(budget.reserve(1).is_err());
                    drop(reservations);
                    assert_eq!(budget.used(), 0);
                    assert_eq!(budget.peak() as u64, planned.mapped_rings);
                }
            }
        }
    }

    #[test]
    fn generic_ring_budget_uses_explicit_endpoints_for_every_format() {
        let mut config = config(4, 0, None);
        config.capacity = 4096;
        set_model(&mut config, 4096, false, 2);
        for geometry in [cuteafd_core::ExpertGeometry::GLM5,
            cuteafd_core::ExpertGeometry::GLM5_FLASH, cuteafd_core::ExpertGeometry::QWEN4] {
            for bf16 in [false, true] {
                config.bf16_ingress = bf16;
                let mut previous = 0;
                for endpoints in [1, 2, 4] {
                    config.rdma_endpoints = endpoints;
                    set_model(&mut config, 4096, bf16, endpoints);
                    let rings = worker_ring_geometry(&config).unwrap();
                    let expected = selected_spark_io(&config, geometry, rings).unwrap().mapped_rings as usize;
                    let budget = worker_ring_budget(&config, geometry).unwrap();
                    assert_eq!(budget.limit(), expected);
                    assert!(expected < usize::MAX);
                    if previous > 0 { assert_eq!(expected, 2 * previous); }
                    previous = expected;
                    assert!(budget.reserve(expected + 1).is_err());
                }
            }
        }
        config.model_admission = None;
        assert!(worker_ring_geometry(&config).is_err());
    }

    /// The runtime ring budget equals the two-endpoint allowance that admission
    /// already charged, and a peer advertising larger negotiated slots (or a
    /// second oversized endpoint) is rejected before any allocation.
    #[test]
    fn runtime_ring_budget_bounds_aggregate_endpoint_advertisements() {
        use cuteafd_transport::RingBudget;
        let topology = SparkTopology::new(2, 2).unwrap();
        let mut config = config(4, 0, Some(topology));
        config.capacity = 4096;
        set_model(&mut config, 4096, false, 2);
        let limit = spark_transport_bytes(&config).unwrap();
        // One capacity-sized endpoint is half the allowance; two fit exactly.
        let planned_endpoint = spark_ring_bytes(
            config.capacity,
            config.max_frame_bytes,
            env_usize("CUTEAFD_VERBS_HOST_RING_DEPTH", 8).unwrap(),
            env_usize("CUTEAFD_VERBS_HOST_RING_SLOT_BYTES", 8 << 20).unwrap(),
            cuteafd_transport::verbs_host_capabilities().preferred_alignment,
            1,
        )
        .unwrap();
        assert_eq!(limit, 2 * planned_endpoint);
        let budget = RingBudget::new(limit);
        // A peer whose advertised rings are larger than the planned allowance
        // cannot even get one endpoint admitted.
        assert!(budget.reserve(limit + 1).is_err());
        assert_eq!(budget.used(), 0);
        // Two endpoints of a larger negotiated size are bounded in aggregate:
        // the first fits, the second is rejected, and dropping the first
        // releases the credit for a compliant peer.
        let oversize = planned_endpoint + (1 << 20);
        assert!(oversize > planned_endpoint && 2 * oversize > limit);
        let first = budget.reserve(oversize).unwrap();
        assert!(budget.reserve(oversize).is_err());
        assert_eq!(budget.used(), oversize);
        drop(first);
        assert_eq!(budget.used(), 0);
        assert!(budget.reserve(planned_endpoint).is_ok());
    }
}

/// Validate format support and exact physical rank identity before allocation.
/// Compressed compact TP3 shares ownership topology, not native weight storage.
/// Native V4 Flash TP2 retains the generic coordinator's canonical protocol.
fn validate_topology(config: &NativeExpertServiceConfig, catalog: &OfficialV41Catalog) -> Result<()> {
    let world = config.spark_world()?;
    let layers = config.resident_layers(catalog.routed_experts().layers)?;
    crate::shared::spark_topology::require_world(world, catalog, &layers)?;
    ensure!(config.world == world.topology().world_size() && config.rank < config.world,
        "Spark worker rank/world differs from its declared topology");
    Ok(())
}

impl NativeExpertServiceConfig {
    fn compiled_capacity(&self) -> Result<u32> {
        Ok(self.model_admission.as_ref().context("expert extent requires coordinator model admission")?.compiled_capacity)
    }

    /// Only exact coordinator-selected absolute IDs may become resident.
    pub(super) fn resident_layers(&self, layers: usize) -> Result<Vec<usize>> {
        let selected = self.selected_layers.as_ref().context("expert ownership requires coordinator selection")?;
        ensure!(selected.windows(2).all(|pair| pair[0] < pair[1])
            && selected.iter().all(|&layer| layer < layers),
            "selected worker layers are outside checkpoint or not ordered and unique");
        Ok(selected.clone())
    }

    /// Declare both physical ownership and the selected request wire contract.
    fn spark_world(&self) -> Result<SparkWorld> {
        ensure!(self.declared_world.topology().world_size() == self.world,
            "Spark worker world differs from its declared topology");
        Ok(self.declared_world)
    }

    fn selection(&self, layer: usize) -> Result<ExpertLayer> {
        let world = self.spark_world()?;
        let topology = world.topology();
        let rank = topology.tp_rank(self.rank)? as usize;
        Ok(match (topology.tp(), world.wire()) {
            (4, _) => ExpertLayer::Backbone { layer, rank },
            (2, cuteafd_transport::expert::SparkWire::Canonical) if !self.native_spark_tp2 =>
                ExpertLayer::BackboneTp2 { layer, rank },
            (tp, cuteafd_transport::expert::SparkWire::NativeOwnership) =>
                ExpertLayer::BackboneReplicatedTp { layer, rank, world: tp as usize },
            (2, _) if self.native_spark_tp2 =>
                ExpertLayer::BackboneReplicatedTp { layer, rank, world: 2 },
            (tp, _) => ExpertLayer::BackboneExl3Tp { layer, rank, world: tp as usize },
        })
    }

    fn exl3_selection(&self, layer: usize) -> Result<ExpertLayer> {
        let world = self.spark_world()?;
        if world.ownership_topology().is_some() {
            ensure!(world.topology() == SparkTopology::NATIVE_TP3_EP1,
                "compact EXL3 ownership wire requires TP3EP1");
        }
        let topology = world.topology();
        // Spark EXL3 always uses the compressed-shard variant. Coordinator
        // BackboneTp2 keeps its separate legacy layout and arithmetic.
        Ok(ExpertLayer::BackboneExl3Tp { layer,
            rank: topology.tp_rank(self.rank)? as usize, world: topology.tp() as usize })
    }

    fn native_group(&self) -> Result<Option<u8>> {
        self.spark_world()?.ownership_topology().map(|topology| topology.group(self.rank)).transpose()
    }
    /// Resolve this rank's EXL3 AOT package for the running checkpoint's
    /// decoder tiers (multi-family images) with the legacy single-family
    /// location as fallback. An explicit --exl3-aot-dir is used verbatim.
    fn exl3_directory_for(&self, tiers: &[usize], layout: &str) -> PathBuf {
        self.exl3_aot_dir.clone().unwrap_or_else(|| {
            crate::shared::experts::exl3::aot_layout_directory(&self.library, tiers, layout)
        })
    }

    fn exl3_directory(&self, catalog: &OfficialV41Catalog) -> Result<PathBuf> {
        let layer = *self.resident_layers(catalog.routed_experts().layers)?.first()
            .context("EXL3 package selection requires a resident layer")?;
        let plan = crate::shared::experts::exl3::planned_residency(catalog,
            self.exl3_selection(layer)?)?;
        Ok(self.exl3_directory_for(&plan.tiers, &plan.aot_layout()))
    }

    fn validate_exl3_residency(&self, catalog: &OfficialV41Catalog, directory: &std::path::Path) -> Result<()> {
        let geometry = catalog.routed_experts().geometry()?;
        for layer in self.resident_layers(catalog.routed_experts().layers)? {
            let plan = crate::shared::experts::exl3::planned_residency(catalog,
                self.exl3_selection(layer)?)?;
            crate::shared::experts::exl3::worker::Exl3Worker::validate_planned_residency(
                directory, self.capacity, self.exl3_schedule, &plan, geometry)?;
        }
        Ok(())
    }
}
