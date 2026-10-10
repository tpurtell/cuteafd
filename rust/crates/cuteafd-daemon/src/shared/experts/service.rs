//! Native GPU resources and inference QPs share one owning thread.
mod local;
mod backend;

use crate::shared::experts::execution::HostExpertExchange;
use crate::shared::experts::layer::{ExpertLayer, ExpertWeights};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::OfficialV41Catalog;
use cuteafd_transport::expert::{BackboneRequest, SparkTopology};
use std::{path::PathBuf, sync::{mpsc, Arc}, thread};

pub(crate) async fn run(args: crate::cli::NativeExpertDaemonArgs) -> Result<()> {
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
        rank: args.rank as usize,
        world: args.world as usize,
        first_layer: args.first_layer as usize,
        last_layer: args.last_layer.map(|layer| layer as usize),
        capacity: args.capacity,
        device_budget: args.device_budget_bytes,
        max_frame_bytes: args.max_frame_bytes,
        topology,
        native_spark_tp2: false,
        bf16_ingress: false,
        encoder,
        audio_encoder,
    };
    tokio::task::spawn_blocking(move || local::run(config, &args.listen))
        .await
        .context("native local RoCE owner failed")?
}

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
    pub rank: usize,
    pub world: usize,
    pub first_layer: usize,
    /// Inclusive last resident layer; `None` serves through the model's last.
    pub last_layer: Option<usize>,
    pub capacity: u32,
    pub device_budget: usize,
    pub max_frame_bytes: usize,
    /// Explicit replicated `TP×EP` topology; `None` keeps the legacy world 2/4
    /// behavior (EXL3 compact may still select a TP2 RTX pair).
    pub topology: Option<SparkTopology>,
    /// Native Flash's TP2 kernel uses the Spark shard role with legacy wire
    /// requests. Set only after catalog validation, never from a CLI override.
    native_spark_tp2: bool,
    bf16_ingress: bool,
}

fn mem_available(text: &str) -> Result<usize> {
    let value = text.lines().find_map(|line| line.strip_prefix("MemAvailable:"))
        .context("/proc/meminfo has no MemAvailable")?;
    let fields: Vec<_> = value.split_whitespace().collect();
    ensure!(fields.len() == 2 && fields[1] == "kB", "invalid MemAvailable units");
    fields[0].parse::<usize>()?.checked_mul(1024).context("MemAvailable overflow")
}

/// GB10 device and host allocations consume the same pool. CUDA's free-byte
/// counter alone omits reclaimable cache; use the OS's unified availability.
fn worker_available(library: &NativeLibrary) -> Result<usize> {
    let info = library.cuda_device_info(library.cuda_get_device()?)?;
    if info.compute_capability_major == 12 && info.compute_capability_minor == 1 {
        mem_available(&std::fs::read_to_string("/proc/meminfo")?)
    } else {
        Ok(library.cuda_memory_info()?.0)
    }
}

fn admit_worker_peak(library: &NativeLibrary, load_peak: usize, serve_peak: usize) -> Result<()> {
    let available = worker_available(library)?;
    ensure!(load_peak <= available && serve_peak <= available,
        "Spark worker needs {load_peak} bytes at load / {serve_peak} at serve but only {available} bytes are available");
    Ok(())
}

fn load_weights<'a>(
    library: &'a NativeLibrary,
    catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig,
) -> Result<(backend::Weights<'a>, usize)> {
    let layers = config.resident_layers(catalog.routed_experts().layers)?.end;
    let first = catalog.routed_experts().first_layer;
    ensure!(config.first_layer >= first,
        "this checkpoint's routed experts start at layer {first}; pass --first-layer {first} or later");
    validate_topology(config, catalog)?;
    if catalog.exl3().is_some() { return backend::load_exl3(library, catalog, config); }
    if catalog.fp8().is_some() { return backend::load_fp8(library, catalog, config); }
    log_spark_memory_if_enabled(library, config, "worker startup", None, None);
    // NVFP4 backbone experts load through the format-aware ExpertWeights path.
    let nvfp4 = catalog.nvfp4().is_some();
    let mut resident = 0usize;
    let mut staging = 0usize;
    let mut pinned_host = 0usize;
    let mut read_scratch = 0usize;
    for layer in config.first_layer..layers {
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
    if config.topology.is_some() {
        ensure!(
            budget.load_peak <= config.device_budget
                && budget.serve_peak <= config.device_budget,
            "explicit {}x{} Spark admission exceeds the {}-byte device budget: \
             resident {} + load peak {} / serve peak {}",
            config.topology.map_or(0, |t| t.tp()),
            config.topology.map_or(0, |t| t.ep()),
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
        let free = worker_available(library).context("query Spark memory for admission")?;
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
            ring_bytes = budget.rings,
            runtime_headroom_bytes = budget.headroom,
            load_peak_bytes = budget.load_peak,
            serve_peak_bytes = budget.serve_peak,
            device_budget_bytes = config.device_budget,
            "explicit Spark replicas admission"
        );
    }
    let mut weights = Vec::with_capacity(layers - config.first_layer);
    let mut remaining = config.device_budget;
    for layer in config.first_layer..layers {
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
        weights.push(weight);
    }
    log_spark_memory_if_enabled(library, config, "weights resident", None, None);
    Ok((backend::Weights::Full(weights), remaining))
}

/// Known explicit-topology peak bytes for the two live phases.
struct SparkAdmissionBudget {
    exchange: usize,
    row_indices: usize,
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

/// The Spark worker always opens exactly two persistent endpoints (decode and
/// prefill); that is the count the admission model reserves. The obsolete
/// `CUTEAFD_SPARK_RDMA_ENDPOINTS` knob is rejected unless it names that value, so
/// a stale launch cannot silently under-reserve the mapped rings.
fn parse_rdma_endpoints(value: Option<&str>) -> Result<usize> {
    match value.map(str::trim) {
        None => Ok(2),
        Some("2") => {
            tracing::warn!(
                "CUTEAFD_SPARK_RDMA_ENDPOINTS is obsolete; the Spark worker always opens two persistent endpoints"
            );
            Ok(2)
        }
        Some(other) => anyhow::bail!(
            "CUTEAFD_SPARK_RDMA_ENDPOINTS={other} is obsolete and unsupported; the Spark worker \
             requires exactly 2 persistent endpoints (decode and prefill)"
        ),
    }
}

/// Ring bytes for this worker's configured topology and frame budget. The
/// decode and prefill lanes each open one persistent endpoint.
fn spark_transport_bytes(config: &NativeExpertServiceConfig) -> Result<usize> {
    let depth = env_usize("CUTEAFD_VERBS_HOST_RING_DEPTH", 8)?;
    let slot_bytes = env_usize("CUTEAFD_VERBS_HOST_RING_SLOT_BYTES", 8 << 20)?;
    let requested = std::env::var("CUTEAFD_SPARK_RDMA_ENDPOINTS").ok();
    let endpoints = parse_rdma_endpoints(requested.as_deref())?;
    let alignment = cuteafd_transport::verbs_host_capabilities().preferred_alignment;
    let bytes = spark_ring_bytes_for_ingress(
        config.capacity,
        config.max_frame_bytes,
        depth,
        slot_bytes,
        alignment,
        endpoints,
        config.bf16_ingress,
    )?;
    validate_v41_ring_charge(config, cuteafd_core::expert_geometry(), depth, slot_bytes, bytes)?;
    Ok(bytes)
}

fn validate_v41_ring_charge(config: &NativeExpertServiceConfig, geometry: cuteafd_core::ExpertGeometry,
    depth: usize, slot_bytes: usize, bytes: usize) -> Result<()> {
    if geometry.family() == Some("v41") {
        let charge = cuteafd_loader::plan::layout::v41_spark_ring_allowance(
            "deepseek_v41", config.capacity as u64, 0, config.bf16_ingress);
        ensure!(bytes as u64 <= charge,
            "V4.1 CUTEAFD_VERBS_HOST_RING_DEPTH={depth} CUTEAFD_VERBS_HOST_RING_SLOT_BYTES={slot_bytes} \
             geometry uses {bytes} bytes, exceeding planner charge {charge} bytes");
    }
    Ok(())
}

fn worker_ring_budget(config: &NativeExpertServiceConfig, geometry: cuteafd_core::ExpertGeometry)
    -> Result<Arc<cuteafd_transport::RingBudget>> {
    // Topology changes expert ownership, not the two V4.1 transport owners.
    let limit = if geometry.family() == Some("v41") { spark_transport_bytes(config)? }
        else { usize::MAX };
    Ok(cuteafd_transport::RingBudget::new(limit))
}

/// Optional explicit reserve for allocation granularity, the CUDA context and
/// library state that the source cannot see. It defaults to zero: no arbitrary
/// headroom is invented, and a real reserve is supplied from an actual
/// measurement. `CUTEAFD_SPARK_RUNTIME_HEADROOM_BYTES` sets it.
fn spark_runtime_headroom() -> Result<usize> {
    env_usize("CUTEAFD_SPARK_RUNTIME_HEADROOM_BYTES", 0)
}

fn spark_admission_budget(
    config: &NativeExpertServiceConfig,
    resident: usize,
    staging: usize,
    pinned_host: usize,
    read_scratch: usize,
    workspace: usize,
) -> Result<SparkAdmissionBudget> {
    let exchange = HostExpertExchange::bytes_for(config.capacity)?;
    let row_indices = config.capacity as usize * 4;
    let rings = spark_transport_bytes(config)?;
    let headroom = spark_runtime_headroom()?;
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
        .and_then(|bytes| bytes.checked_add(headroom))
        .context("Spark serve peak overflow")?;
    Ok(SparkAdmissionBudget { exchange, row_indices, rings, headroom, load_peak, serve_peak })
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
    let cuda = library.cuda_memory_info().ok();
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
        topology = ?config.topology.map(|topology| (topology.tp(), topology.ep())),
        owned_connections = ?owned_connections,
        cuda_free_bytes = cuda.map(|(free, _)| free),
        cuda_total_bytes = cuda.map(|(_, total)| total),
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

    fn config(world: usize, rank: usize, topology: Option<SparkTopology>) -> NativeExpertServiceConfig {
        NativeExpertServiceConfig {
            library: PathBuf::from("/native.so"),
            exl3_aot_dir: None,
            exl3_schedule: Default::default(),
            fp8_package: None,
            snapshot: PathBuf::from("/model"),
            rank,
            world,
            first_layer: 0,
            last_layer: None,
            capacity: 16,
            device_budget: 1 << 40,
            max_frame_bytes: 64 << 20,
            topology,
            native_spark_tp2: false,
            bf16_ingress: false,
            encoder: None,
            audio_encoder: None,
        }
    }

    #[test]
    fn gb10_available_uses_reclaimable_memory_and_rejects_bad_snapshots() {
        assert_eq!(mem_available("MemFree: 1 kB\nMemAvailable: 1024 kB\n").unwrap(), 1 << 20);
        for text in ["MemFree: 1 kB", "MemAvailable: 8 MB", "MemAvailable: nope kB"] {
            assert!(mem_available(text).is_err());
        }
        assert!(mem_available(&format!("MemAvailable: {} kB", usize::MAX)).is_err());
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
    fn implicit_tp1_selects_whole_exl3_experts_and_rank_package() {
        let one = config(1, 0, None);
        assert_eq!(one.selection(7).unwrap(), ExpertLayer::BackboneExl3Tp { layer: 7, rank: 0, world: 1 });
        assert!(one.selection(7).unwrap().role() > 7);
        assert_eq!(one.native_group().unwrap(), None);
        assert!(one.exl3_directory_for(&[4, 5]).ends_with("tp1-rank0"));
    }

    #[test]
    fn legacy_selection_is_unchanged_and_topology_must_agree() {
        assert_eq!(config(4, 3, None).selection(2).unwrap(), ExpertLayer::Backbone { layer: 2, rank: 3 });
        assert_eq!(config(2, 1, None).selection(2).unwrap(), ExpertLayer::BackboneTp2 { layer: 2, rank: 1 });
        // The implicit three-rank group is the EXL3 compact shard, keyed on the
        // launched world rather than an explicit topology key.
        assert_eq!(
            config(3, 2, None).selection(2).unwrap(),
            ExpertLayer::BackboneExl3Tp { layer: 2, rank: 2, world: 3 }
        );
        // The explicit topology must match the launched rank count.
        let topology = SparkTopology::new(2, 2).unwrap();
        assert!(resolve_topology_mismatch(&config(6, 0, Some(topology))));
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
    fn flash_native_tp2_selects_spark_role_without_changing_wire_protocol() {
        for rank in 0..2 {
            let mut flash = config(2, rank, None);
            flash.native_spark_tp2 = true;
            let layer = flash.selection(7).unwrap();
            assert_eq!(layer, ExpertLayer::BackboneReplicatedTp { layer: 7, rank, world: 2 });
            assert_eq!(layer.role(), crate::shared::spark_topology::SPARK_TP2_ROLE);
            assert!(flash.topology.is_none());
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

    fn resolve_topology_mismatch(config: &NativeExpertServiceConfig) -> bool {
        crate::shared::spark_topology::resolve(
            Some(config.topology.unwrap().tp()),
            Some(config.topology.unwrap().ep()),
            config.world,
            "expertd-native",
        )
        .is_err()
    }

    /// The explicit-topology admission is exactly the source-known transient sum,
    /// with no invented headroom when the override is unset.
    #[test]
    fn explicit_admission_counts_every_known_transient_exactly() {
        let topology = SparkTopology::new(2, 2).unwrap();
        let mut config = config(4, 0, Some(topology));
        config.capacity = 4096;
        let headroom = spark_runtime_headroom().unwrap();
        let rings = spark_transport_bytes(&config).unwrap();
        let budget = spark_admission_budget(&config, 10, 1, 2, 3, 4).unwrap();
        let exchange = HostExpertExchange::bytes_for(4096).unwrap();
        assert_eq!(exchange, 4096 * (2 * 6 * 4 + 5120 * 2));
        assert_eq!(budget.exchange, exchange);
        assert_eq!(budget.row_indices, 4096 * 4);
        assert_eq!(budget.rings, rings);
        assert_eq!(budget.load_peak, 10 + 1 + 2 + 3 + headroom);
        assert_eq!(budget.serve_peak, 10 + 4 + exchange + 4096 * 4 + rings + headroom);
        // The registered rings and the 40 MiB host exchange dominate the serve
        // peak at full capacity.
        assert!(budget.serve_peak > budget.load_peak);
        assert!(rings > exchange);
        assert_eq!(HostExpertExchange::bytes_for(16).unwrap(), 16 * (48 + 10240));
    }

    /// The ring term mirrors the transport's own sizing rule: minimum 8 MiB per
    /// slot, `depth` slots per ring, one request and one response ring per
    /// persistent endpoint, rounded up to the verbs-host alignment.
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
    fn standard_v41_planner_allowance_covers_exact_worker_rings() {
        assert_eq!(cuteafd_transport::protocol_v2::EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN, 96);
        assert_eq!(cuteafd_transport::protocol_v2::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN, 96);
        assert_eq!(cuteafd_transport::protocol_v2::EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN, 40);
        assert_eq!(cuteafd_transport::protocol_v2::EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN, 12);
        let allowance = cuteafd_loader::plan::layout::family_costs("deepseek_v41").spark_ring_bytes;
        for bf16 in [false, true] {
            for capacity in [1, 16, 80, 256, 1024, 2048, 4096] {
                let exact = spark_ring_bytes_for_ingress(capacity, 64 << 20, 8, 8 << 20,
                    cuteafd_transport::verbs_host_capabilities().preferred_alignment, 2, bf16).unwrap();
                let charged = cuteafd_loader::plan::layout::v41_spark_ring_allowance(
                    "deepseek_v41", capacity as u64, allowance, bf16);
                assert!(exact as u64 <= charged, "capacity={capacity} bf16={bf16} rings={exact} charged={charged}");
                assert_eq!(charged, exact as u64);
            }
        }
    }

    #[test]
    fn v41_ring_overrides_cannot_exceed_planner_charge() {
        let geometry = cuteafd_core::ExpertGeometry::DEEPSEEK_V41;
        let config = config(3, 0, None);
        let charge = cuteafd_loader::plan::layout::v41_spark_ring_allowance(
            "deepseek_v41", config.capacity as u64, 0, false) as usize;
        let smaller = geometry.compact_ring_bytes(config.capacity, 64 << 20, 4, 8 << 20, 4096, 2, false).unwrap();
        let larger = geometry.compact_ring_bytes(config.capacity, 64 << 20, 8, 64 << 20, 4096, 2, false).unwrap();
        assert!(validate_v41_ring_charge(&config, geometry, 4, 8 << 20, smaller).is_ok());
        let error = validate_v41_ring_charge(&config, geometry, 8, 64 << 20, larger)
            .unwrap_err().to_string();
        for detail in ["CUTEAFD_VERBS_HOST_RING_DEPTH=8", "CUTEAFD_VERBS_HOST_RING_SLOT_BYTES=67108864",
            &format!("{larger} bytes"), &format!("planner charge {charge}")] {
            assert!(error.contains(detail), "{error}");
        }
    }

    #[test]
    fn v41_replacement_sequence_enforces_budget_on_both_topology_paths() {
        for topology in [None, Some(SparkTopology::new(3, 1).unwrap())] {
            for bf16 in [false, true] {
                let mut config = config(3, 0, topology);
                config.capacity = 1024;
                config.bf16_ingress = bf16;
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

    /// The obsolete endpoint override is gone: exactly two endpoints (decode and
    /// prefill) are always reserved, and a stale non-2 value fails closed.
    #[test]
    fn endpoint_count_is_fixed_at_two_and_stale_overrides_are_rejected() {
        assert_eq!(parse_rdma_endpoints(None).unwrap(), 2);
        assert_eq!(parse_rdma_endpoints(Some("2")).unwrap(), 2);
        assert_eq!(parse_rdma_endpoints(Some(" 2 ")).unwrap(), 2);
        for obsolete in ["1", "0", "3", "16", "x", ""] {
            assert!(
                parse_rdma_endpoints(Some(obsolete)).is_err(),
                "{obsolete:?} must be rejected"
            );
        }
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

/// Reject an explicit topology before any weight allocation: it is defined only
/// for the official native checkpoint, its rank count must match `--world`, and
/// every rank must be inside the topology. Legacy V4.1 native TP4/EXL3
/// selection is preserved. Native V4 Flash TP2 selects its Spark shard while
/// keeping the generic coordinator's legacy request protocol.
fn validate_topology(config: &NativeExpertServiceConfig, catalog: &OfficialV41Catalog) -> Result<()> {
    if let Some(topology) = config.topology {
        crate::shared::spark_topology::require_native(Some(topology), catalog)?;
        ensure!(
            config.world == topology.world_size() && config.rank < config.world,
            "explicit Spark topology {}x{} needs --world {} and --rank below it, got world {} rank {}",
            topology.tp(),
            topology.ep(),
            topology.world_size(),
            config.world,
            config.rank
        );
        return Ok(());
    }
    ensure!(
        matches!(config.world, 1 | 2 | 3 | 4 | 6) && config.rank < config.world,
        "implicit Spark world must be 1, 2, 3, 4 or 6 with rank below world; \
         an explicit TP x EP topology must pass --spark-tp/--spark-ep"
    );
    ensure!(
        config.world != 1 || (catalog.exl3().is_some()
            && catalog.routed_experts().geometry()?.family() == Some("qwen4")),
        "implicit Spark TP1 requires Qwen EXL3 experts (qwen4:exl3-k45)"
    );
    // Native FP4 Flash has a distinct TP2 shard. Other native FP4 layouts
    // continue to require the explicit ownership-aware topology outside TP4.
    ensure!(
        config.world == 4 || catalog.exl3().is_some() || config.native_spark_tp2
            || (matches!(config.world, 2 | 6) && catalog.fp8().is_some()),
        "this implicit Spark group requires EXL3/FP8 experts or native V4 Flash TP2; \
         other native groups must pass --spark-tp/--spark-ep"
    );
    Ok(())
}

impl NativeExpertServiceConfig {
    /// Resident backbone layers: `first_layer..=last_layer`, bounded by the model.
    pub(super) fn resident_layers(&self, layers: usize) -> Result<std::ops::Range<usize>> {
        let end = match self.last_layer {
            Some(last) => {
                ensure!(last < layers, "native last layer must be below {layers}");
                last + 1
            }
            None => layers,
        };
        ensure!(
            self.first_layer < end,
            "native first layer {} must be below {layers} and not after the last layer",
            self.first_layer
        );
        Ok(self.first_layer..end)
    }
    /// Resident layer selection for the running checkpoint format. Explicit
    /// replicated topologies always select the native generic shard; the legacy
    /// path keeps the fixed TP4/EXL3-TP2 behavior byte-for-byte, and an
    /// admitted implicit three-rank group selects the EXL3 compact shard layer.
    fn selection(&self, layer: usize) -> Result<ExpertLayer> {
        let Some(topology) = self.topology else {
            return Ok(match self.world {
                2 if self.native_spark_tp2 => ExpertLayer::BackboneReplicatedTp { layer, rank: self.rank, world: 2 },
                2 => ExpertLayer::BackboneTp2 { layer, rank: self.rank },
                // Admission above admits world 3 only for an EXL3 checkpoint, so
                // this can never resolve to the native FP8 shard family.
                1 | 3 | 6 => ExpertLayer::BackboneExl3Tp { layer, rank: self.rank, world: self.world },
                _ => ExpertLayer::Backbone { layer, rank: self.rank },
            });
        };
        let shard = topology.tp_rank(self.rank)? as usize;
        Ok(match topology.tp() {
            4 => ExpertLayer::Backbone { layer, rank: shard },
            world => ExpertLayer::BackboneReplicatedTp { layer, rank: shard, world: world as usize },
        })
    }
    /// Replicated group this worker unpacks routes for, or `None` for legacy.
    fn native_group(&self) -> Result<Option<u8>> {
        crate::shared::spark_topology::group_of(self.topology, self.rank)
    }
    /// Resolve this rank's EXL3 AOT package for the running checkpoint's
    /// decoder tiers (multi-family images) with the legacy single-family
    /// location as fallback. An explicit --exl3-aot-dir is used verbatim.
    fn exl3_directory_for(&self, tiers: &[usize]) -> PathBuf {
        self.exl3_aot_dir.clone().unwrap_or_else(|| {
            crate::shared::experts::exl3::aot_layout_directory(
                &self.library,
                tiers,
                &format!("tp{}-rank{}", self.world, self.rank),
            )
        })
    }
}
