//! The GPU owner polls admitted QPs and sends results without work queues.
use super::*;
use cuteafd_transport::{LocalVerbsExpertConnection, ProtocolV2ExecutorResponseRef};
use super::backend::{Execution, Weights};
use std::{
    net::TcpListener,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

fn apply_worker_selection(config: &mut NativeExpertServiceConfig,
    selection: &cuteafd_transport::worker_selection::WorkerSelection,
    first_routed: usize, layers: usize) -> Result<()> {
    let selected = selection.layers();
    ensure!(selected.iter().all(|&layer| layer >= first_routed && layer < layers),
        "solved worker selection is outside checkpoint routed layers");
    {
        let model = selection.model().context("worker selection requires typed model admission")?;
        model.validate(selection.geometry())?;
        config.capacity = model.capacity();
        config.max_frame_bytes = model.rings.max_frame_bytes;
        config.rdma_endpoints = model.endpoints.len();
        config.bf16_ingress = model.endpoints.iter().any(|endpoint|
            matches!(endpoint.ingress, cuteafd_core::SparkIngress::Bf16
                | cuteafd_core::SparkIngress::Native
                | cuteafd_core::SparkIngress::Bf16Decode { .. }));
        config.model_admission = Some(model.clone());
    }
    config.selected_layers = Some(selected);
    Ok(())
}

pub(super) fn admit_worker_selection(config: &mut NativeExpertServiceConfig,
    selection: &cuteafd_transport::worker_selection::WorkerSelection, catalog: &OfficialV41Catalog) -> Result<()> {
    apply_worker_selection(config, selection, catalog.routed_experts().first_layer,
        catalog.routed_experts().layers)?;
    if !selection.ranges().is_empty() { validate_topology(config, catalog)?; }
    Ok(())
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    #[test]
    fn received_only_intake_flushes_a_lone_unit_without_a_wait_path() {
        assert!(continue_intake(true, true, 0, 4));
        // The next nonblocking poll finds no peer: launch now, no deadline.
        assert!(!continue_intake(false, true, 1, 4));
        assert!(!continue_intake(true, false, 0, 4));
        assert!(!continue_intake(true, true, 3, 4));
        let source = include_str!("local.rs").rsplit_once("pub(super) fn run(").unwrap().1;
        let receive = source.find("let launched = connections[index].begin_deferred").unwrap();
        let direct = source[receive..].find("serve.launch(&mut executions[owner]").unwrap() + receive;
        let sweep = source[direct..].find("index += 1;").unwrap() + direct;
        let repeat = source[sweep..].find("continue_intake(received_this_sweep").unwrap() + sweep;
        assert!(receive < direct && direct < sweep && sweep < repeat);
        assert!(source.contains("let wait = (intake == 0 && !executions.iter().any(Execution::has_staged)"));
        // Negative control: repeating the first peer before advancing the normal
        // sweep delays the decode on peer 1. Production advances first.
        let normal = ["prefill peer 0", "decode peer 1", "extra intake"];
        let delayed = ["prefill peer 0", "extra intake", "decode peer 1"];
        assert!(normal.iter().position(|&e| e == "decode peer 1")
            < delayed.iter().position(|&e| e == "decode peer 1"));
    }

    #[test]
    fn full_response_ring_stops_staged_intake_before_any_send_wait() -> Result<()> {
        assert!(intake_has_capacity(0, false, || panic!("ordinary decode must not preflight"))?);
        for (sweep, staged) in [(0, true), (1, false), (1, true)] {
            let mut checked = 0;
            let mut launched = 0;
            let mut send_waits = 0;
            if intake_has_capacity(sweep, staged, || { checked += 1; Ok(false) })? {
                send_waits += 1; // Negative control: begin_deferred on a full ring waits.
            } else {
                launched += usize::from(staged);
            }
            assert_eq!((checked, send_waits, launched), (1, 0, usize::from(staged)));
        }
        let source = include_str!("local.rs").rsplit_once("pub(super) fn run(").unwrap().1;
        assert!(source.find("try_deferred_capacity").unwrap() < source.find(".begin_deferred(").unwrap());
        let stop = source.find("if intake_blocked {\n                        sweep_blocked = true;\n                        break;").unwrap();
        assert!(source[stop..].contains("flush_after_sweep(execution.has_staged(), sweep_blocked,"));
        Ok(())
    }

    #[test]
    fn staged_sweep_waits_for_idle_unless_intake_is_blocked() -> Result<()> {
        assert!(!flush_after_sweep(true, false, || Ok(false))?);
        assert!(flush_after_sweep(true, false, || Ok(true))?);
        assert!(flush_after_sweep(true, true, || panic!("blocked intake must flush even while busy"))?);
        assert!(!flush_after_sweep(false, false, || panic!("no staged work needs no stream query"))?);
        assert!(flush_after_sweep(true, false, || anyhow::bail!("query failed")).is_err());
        // A lone staged unit stays nonblocking until the next sweep sees idle.
        for idle in [false, false, true] {
            assert_eq!(flush_after_sweep(true, false, || Ok(idle))?, idle);
        }
        Ok(())
    }

    #[test]
    fn staged_cancellation_neither_publishes_nor_strands_other_credits() -> Result<()> {
        let owners = cuteafd_core::SparkExecutions::admit(&[64], 4096, 2752, 3, &cuteafd_core::EXL3_CAPACITIES)?;
        let mut queues = OwnerQueues::new(&owners);
        let first = queues.record_request(1, 0x1000)?;
        let second = queues.record_request(1, 0x8000)?;
        assert!(!queues.ready(first) && !queues.ready(second));
        assert!(queues.release(first).is_err());
        queues.cancel_staged(first)?;
        assert_eq!(queues.retained[1], 1);
        assert_eq!(queues.pending[1].front(), Some(&second));
        assert!(queues.cancel_staged(first).is_err());
        queues.publish_oldest(1, None)?;
        assert!(queues.ready(second));
        assert!(queues.cancel_staged(second).is_err());
        queues.release(second)?;
        assert_eq!(queues.retained[1], 0);
        Ok(())
    }

    #[test]
    fn solved_worker_ids_control_loaded_and_admitted_set() {
        use cuteafd_transport::worker_selection::{WorkerIdentity, WorkerSelection};
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors.index.json"), "{}").unwrap();
        let geometry = cuteafd_core::ExpertGeometry::MIMO_V26_PRO;
        let identity = WorkerIdentity::from_snapshot(dir.path(), geometry, cuteafd_transport::expert::SparkWorld::canonical(6).unwrap()).unwrap();
        // Every backend consumes the same exact selected IDs within checkpoint bounds.
        let mut config = NativeExpertServiceConfig { encoder: None, audio_encoder: None,
            library: "/native.so".into(), exl3_aot_dir: None, exl3_schedule: Default::default(), fp8_package: None,
            snapshot: dir.path().into(), exlr8_manifest: None, exlr8_prefill_activations: Default::default(), rank: 0, world: 6, model_admission: None, selected_layers: None, capacity: 4096, device_budget: 121 << 30, max_frame_bytes: 64 << 20,
            rdma_endpoints: 3, declared_world: cuteafd_transport::expert::SparkWorld::canonical(6).unwrap(), native_spark_roles: false, bf16_ingress: false };
        let model = cuteafd_transport::worker_selection::ModelAdmission {
            executions: cuteafd_core::SparkExecutions::admit(&[80], 80, 80, 1, &[80]).unwrap(),
            endpoints: vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity: 80,
                ingress: cuteafd_core::SparkIngress::Bf16Decode { rows: 16 } }],
            rings: cuteafd_core::SparkRings { max_frame_bytes: 32 << 20, depth: 2,
                slot_bytes: 1 << 20, alignment: 4096 } };
        let selection = WorkerSelection::new(identity.clone(), &(13..70).collect::<Vec<_>>()).unwrap()
            .with_model(model.clone()).unwrap();
        apply_worker_selection(&mut config, &selection, 1, 70).unwrap();
        assert_eq!(config.resident_layers(70).unwrap(), (13..70).collect::<Vec<_>>());
        assert_eq!(config.selection(13).unwrap(), ExpertLayer::BackboneExl3Tp { layer: 13, rank: 0, world: 6 });
        let disjoint = WorkerSelection::new(identity.clone(), &[13, 15]).unwrap().with_model(model.clone()).unwrap();
        apply_worker_selection(&mut config, &disjoint, 1, 70).unwrap();
        assert_eq!(config.resident_layers(70).unwrap(), vec![13, 15]);
        let before = WorkerSelection::new(identity.clone(), &[0]).unwrap().with_model(model.clone()).unwrap();
        assert!(apply_worker_selection(&mut config, &before, 1, 70).is_err());
        let untyped = WorkerSelection::new(identity.clone(), &[13]).unwrap();
        assert!(apply_worker_selection(&mut config, &untyped, 1, 70).is_err());
        let sealed = selection;
        apply_worker_selection(&mut config, &sealed, 1, 70).unwrap();
        assert_eq!(config.capacity, 80);
        assert_eq!(config.max_frame_bytes, 32 << 20);
        assert_eq!(config.rdma_endpoints, 1);
        assert!(config.bf16_ingress);
        assert_eq!(config.model_admission, Some(model.clone()));
        let mut covered = model.clone();
        covered.executions.prefill = cuteafd_core::PrefillExecution { capacity: 2048,
            compiled_capacity: 4096, unit_rows: 2048, buffered_units: 1 };
        covered.endpoints[0].capacity = 2048;
        let covered = sealed.clone().with_model(covered).unwrap();
        apply_worker_selection(&mut config, &covered, 1, 70).unwrap();
        assert_eq!(config.capacity, 2048);
        assert_eq!(config.compiled_capacity().unwrap(), 4096);
        let empty = WorkerSelection::new(identity, &[]).unwrap().with_model(model).unwrap();
        apply_worker_selection(&mut config, &empty, 1, 70).unwrap();
        assert!(config.resident_layers(70).unwrap().is_empty());
    }

    #[test]
    fn retained_workspace_subtracts_owners_not_weight_scope() {
        // Scratch may share the weights scope; only actual owners are resident.
        assert_eq!(retained_workspace(100, 190, 70, 20, 25).unwrap(), (20, 5));
        assert_eq!(retained_workspace(100, 100, 0, 20, 20).unwrap(), (0, 0));
        assert!(retained_workspace(100, 99, 0, 20, 20).is_err());
        assert!(retained_workspace(100, 190, 91, 20, 20).is_err());
        assert!(retained_workspace(100, 190, 70, 20, 19).is_err());
    }
}

fn run_empty_selection(listener: TcpListener, rank: usize,
    installed: &cuteafd_transport::worker_selection::InstalledModelAdmission) -> Result<()> {
    listener.set_nonblocking(false)?;
    for accepted in listener.incoming() {
        let mut stream = accepted?;
        let result = (|| -> Result<()> {
            let repeated = cuteafd_transport::worker_selection::receive_worker_selection(&stream, rank, Duration::from_secs(30))?;
            ensure!(&repeated == installed.selection(), "empty worker selection is immutable until restart");
            installed.acknowledge(&mut stream)
        })();
        if let Err(error) = result {
            if error.is::<cuteafd_transport::traffic::TrafficProbeOnly>() { continue; }
            let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
            tracing::warn!(%error, "empty worker rejects expert traffic or changed selection");
        }
    }
    Ok(())
}

fn retained_workspace(before_device: usize, after_device: usize, resident: u64,
    before_pinned: usize, after_pinned: usize) -> Result<(u64, u64)> {
    let retained = after_device.checked_sub(before_device)
        .context("worker device ledger decreased across expert loading")?;
    let workspace = u64::try_from(retained)?.checked_sub(resident)
        .context("resident owners exceed retained worker device ledger delta")?;
    let pinned = after_pinned.checked_sub(before_pinned)
        .context("worker pinned ledger decreased across expert loading")?;
    Ok((workspace, u64::try_from(pinned)?))
}

fn measured_worker_runtime(library: &NativeLibrary,
    baseline: &cuteafd_ffi::memory_ledger::Snapshot, resident: u64, census: &super::census::Census)
    -> Result<cuteafd_core::SparkRuntimeInventory> {
    use cuteafd_ffi::memory_ledger::{self, Space};
    let device = library.cuda_get_device()?;
    let inventory = crate::shared::inventory::RuntimeInventory::measure(library, &[device],
        crate::shared::inventory::InventoryPhase::PostLoad, |_| Ok(()))?;
    let retained = memory_ledger::snapshot();
    let (workspace_bytes, pinned_workspace_bytes) = retained_workspace(
        baseline.total(Space::Device, device), retained.total(Space::Device, device), resident,
        baseline.total(Space::Pinned, -1), retained.total(Space::Pinned, -1))?;
    Ok(cuteafd_core::SparkRuntimeInventory { detail: census.detail(baseline, &retained, device)?, sample: inventory.gpus.into_iter().next()
        .context("worker runtime sample missing")?, workspace_bytes, pinned_workspace_bytes })
}

struct Admission {
    stop: Arc<AtomicBool>,
}
impl Drop for Admission {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

pub(super) fn run(mut config: NativeExpertServiceConfig, listen: &str) -> Result<()> {
    ensure!(
        config.rank < config.world && (1..=8).contains(&config.world),
        "native rank must be below the launched Spark world"
    );
    // The checkpoint fixes the process expert geometry before the native
    // library loads, so its helpers and every wire size agree with it.
    let inventory = cuteafd_loader::plan::Checkpoint::inventory(&config.snapshot)?;
    let v41 = cuteafd_loader::plan::family::detect(&inventory).is_some_and(|family| family.id() == "deepseek_v41");
    let mut catalog = if v41 {
        cuteafd_loader::read_official_v41_spark_catalog(&config.snapshot, config.rank, config.world)?
    } else {
        cuteafd_loader::read_expert_catalog(&config.snapshot)?
    };
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    crate::shared::spark_topology::require_wire(config.spark_world()?, &catalog)?;
    let geometry = catalog.routed_experts().geometry()?;
    config.native_spark_roles = catalog.exl3().is_none() && catalog.nvfp4().is_none() && catalog.fp8().is_none();
    cuteafd_core::set_expert_geometry(geometry).map_err(|fixed| {
        anyhow::anyhow!("expert geometry is already {fixed:?}; the checkpoint needs {geometry:?}")
    })?;
    drop(inventory);
    let mut census = super::census::Census::new();
    census.sample("before_native_library", None)?;
    // SAFETY: this owner outlives media, weights, retained modules and executions.
    let library = unsafe { NativeLibrary::load(&config.library) }?;
    census.identify_device(&library)?;
    census.sample("before_context", None)?;
    // cudaGetDevice can report the default ordinal with no context current.
    // Bind its runtime primary context on this execution owner before warming
    // any backend's modules, even when the optional census is disabled.
    library.cuda_set_device(library.cuda_get_device()?)?;
    if census.enabled() {
        library.cuda_device_synchronize()?;
        census.sample("after_context_before_media", Some(&library))?;
    }
    // The vision owner loads in this process/device's primary CUDA context,
    // never beside expertd in a second process. Charge it before expert admission.
    let (_encoder, _audio_encoder, reserved) = crate::shared::vision::worker::start_encoders(
        config.encoder.as_ref(), config.audio_encoder.as_ref(), &config.snapshot,
        config.library.clone(), config.device_budget as u64)?;
    census.sample("after_media_owners_before_expert_bootstrap", None)?;
    config.device_budget = config.device_budget.checked_sub(usize::try_from(reserved)?)
        .context("media reservation exceeds Spark budget")?;
    // Media may be admitted first, but expert allocation waits for the one
    // coordinator-authoritative selection. Bootstrap-ready is not weights-ready.
    let (listener, mut stream, selection) = {
        let listener = TcpListener::bind(listen)?;
        listener.set_nonblocking(true)?;
        tracing::info!(rank = config.rank, world = config.world, "native worker bootstrap ready");
        let deadline = Instant::now() + Duration::from_secs(1200);
        let (mut stream, selection) = loop {
            ensure!(Instant::now() < deadline, "coordinator worker selection timed out");
            match listener.accept() {
                Ok((stream, _)) => match cuteafd_transport::worker_selection::receive_worker_selection(
                    &stream, config.rank, deadline.saturating_duration_since(Instant::now()).min(Duration::from_secs(30))) {
                    Ok(selection) => break (stream, selection),
                    Err(error) if error.is::<cuteafd_transport::traffic::TrafficProbeOnly>() => continue,
                    Err(error) => return Err(error),
                },
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(100)),
                Err(error) => return Err(error.into()),
            }
        };
        let selected = (|| -> Result<_> {
            let identity = cuteafd_transport::worker_selection::WorkerIdentity::from_snapshot_manifest(
                &config.snapshot, geometry, config.spark_world()?, config.exlr8_manifest.as_deref())?;
            selection.validate_for(&identity)?;
            admit_worker_selection(&mut config, &selection, &catalog)?;
            Ok(selection)
        })();
        let selection = match selected {
            Ok(selection) => selection,
            Err(error) => {
                let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
                return Err(error);
            }
        };
        if selection.ranges().is_empty() {
            let installed = (|| -> Result<_> {
                let report = {
                    library.release_sync_h2d_staging()?;
                    let baseline = cuteafd_ffi::memory_ledger::snapshot();
                    Some(cuteafd_core::ModelAdmissionResponse { rank: u32::try_from(config.rank)?,
                        resident_bytes: 0, slices: vec![], runtime: measured_worker_runtime(&library, &baseline, 0, &census)?,
                        mapped_ring_bytes: 0, host_exchange_bytes: 0, row_indices_bytes: 0 })
                };
                cuteafd_transport::worker_selection::InstalledModelAdmission::new(config.rank, selection, report)
            })();
            let installed = match installed {
                Ok(installed) => installed,
                Err(error) => {
                    let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
                    return Err(error);
                }
            };
            installed.acknowledge(&mut stream)?;
            tracing::info!(rank = config.rank, world = config.world, layer_ids = ?installed.selection().layers(),
                layers = 0, resident_bytes = 0, "native empty expert worker ready");
            return run_empty_selection(listener, config.rank, &installed);
        }
        (listener, stream, selection)
    };
    // Every endpoint shares the sealed finite allowance, fixed before loading.
    let ring_budget = worker_ring_budget(&config, geometry)?;
    tracing::info!(rdma_endpoints = config.rdma_endpoints, bf16_ingress = config.bf16_ingress,
        ring_budget_bytes = ring_budget.limit(), "Spark ring admission policy");
    // Remove upload transients left by media before taking the retained baseline.
    library.release_sync_h2d_staging()?;
    // Existing media owners belong to the baseline, not expert residency/workspace.
    let baseline = cuteafd_ffi::memory_ledger::snapshot();
    census.sample("before_expert_modules", Some(&library))?;
    let programs = (|| -> Result<_> {
        if catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8 {
            let path = config.library.parent().context("native library directory")?
                .join("../share/PROGRAMS.json");
            Ok(Some(library.programs()?.with_manifest(&path)?))
        } else { Ok(None) }
    })();
    let programs = match programs {
        Ok(programs) => programs,
        Err(error) => {
            let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
            return Err(error);
        }
    };
    let mut modules = match super::modules::WarmModules::load(&library, &catalog, &config, programs.as_ref()) {
        Ok(modules) => modules,
        Err(error) => {
            let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
            return Err(error);
        }
    };
    census.sample("after_expert_modules_before_weights", Some(&library))?;
    let loaded = {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        load_weights(&library, &catalog, &config, &mut modules,
            selection.push_weights().then_some(&mut stream))
    };
    catalog.finish_expert_reads();
    let (weights, remaining) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
            return Err(error);
        }
    };
    // The loaders' synchronous uploads grew the library's pinned staging buffer
    // to the largest projection (hundreds of MB of unified memory on GB10);
    // requests stage far smaller rows, so drop it and let them regrow it.
    census.sample("after_expert_load_includes_weights_and_loader_code", Some(&library))?;
    let released = library.release_sync_h2d_staging()?;
    tracing::info!(released_bytes = released, "released load-time pinned upload staging");
    census.sample("before_retained_arenas", Some(&library))?;
    let owners = config.model_admission.as_ref().context("Spark execution owners require model admission")?
        .executions.clone();
    owners.validate()?;
    let mut queues = OwnerQueues::new(&owners);
    let allocated = (|| -> Result<Vec<Execution<'_, '_>>> {
        let device = library.cuda_get_device()?;
        let mut executions: Vec<_> = std::iter::repeat_with(|| None).take(owners.decode.len() + 1).collect();
        let mut budget = remaining;
        // Consume the maximum-unit warmed exports before constructing narrower
        // decode owners, whose module sets need not have the same capacities.
        let mut extents: Vec<_> = owners.owners().collect();
        extents.rotate_right(1);
        for extent in extents {
            let index = match extent.owner {
                cuteafd_core::ExecutionOwner::Decode(lane) => lane,
                cuteafd_core::ExecutionOwner::Prefill => owners.decode.len(),
            };
            let before = cuteafd_ffi::memory_ledger::snapshot().total(cuteafd_ffi::memory_ledger::Space::Device, device);
            let mut owner_config = config.clone();
            owner_config.capacity = extent.capacity;
            let mut execution = {
                let _memory_scope = cuteafd_ffi::memory_ledger::scope(
                    if index < owners.decode.len() { "experts/workspace/decode" } else { "experts/workspace/prefill" });
                weights.execution(&library, &owner_config, extent.compiled, extent, budget, &mut modules)?
            };
            execution.defer(usize::from(extent.buffered_units), extent.staging_rows as usize, extent.owner)?;
            let used = cuteafd_ffi::memory_ledger::snapshot().total(cuteafd_ffi::memory_ledger::Space::Device, device)
                .saturating_sub(before);
            budget = budget.checked_sub(used).context("Spark execution owners exceed admitted workspace")?;
            executions[index] = Some(execution);
        }
        executions.into_iter().map(|owner| owner.context("Spark execution owner was not constructed")).collect()
    })();
    let mut executions = match allocated {
        Ok(executions) => executions,
        Err(error) => {
            let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
            return Err(error);
        }
    };
    census.sample("after_retained_arenas", Some(&library))?;
    let (mut exchange, mut row_indices) = worker_host_exchange(&config)?;
    // Transport benchmarks only: answer each request with its response slot
    // as it is, without running the experts.
    let skip_compute = std::env::var("CUTEAFD_EXPERTD_SKIP_COMPUTE").is_ok_and(|v| v == "1");
    if skip_compute {
        tracing::warn!("CUTEAFD_EXPERTD_SKIP_COMPUTE=1: answering without running the experts");
    }
    // Point-in-time ring counters for the main-thread memory milestones. The
    // bootstrap thread keeps its own clone; the atomic peak can be raised by a
    // concurrent admission, so these are sampled observations, not reservations.
    let ring_budget_log = ring_budget.clone();
    log_spark_memory_if_enabled(
        &library,
        &config,
        "execution and exchange allocated",
        None,
        Some((ring_budget_log.used(), ring_budget_log.peak())),
    );
    census.sample("serve_owners_before_endpoint_rings", Some(&library))?;
    let worker_selection = {
        if weights.layer_ids()? != selection.layers() {
            let error = anyhow::anyhow!("loaded worker layers do not match the solved selection");
            let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
            return Err(error);
        }
        let reported = (|| -> Result<_> {
            let report = {
                let model = selection.model().context("worker selection requires typed model admission")?;
                weights.validate_ingress(&library, model)?;
                let (resident_bytes, slices) = weights.loaded_residency(&library, config.compiled_capacity()?)?;
                let mut runtime = measured_worker_runtime(&library, &baseline, resident_bytes, &census)?;
                runtime.detail.load_transient = weights.load_inventory();
                runtime.detail.allocations.extend(weights.retained_host_inventory()?);
                tracing::info!(target: "cuteafd::spark_memory", rank = config.rank,
                    record = %serde_json::to_string(&runtime)?, "Spark retained runtime inventory");
                let host_exchange_bytes = exchange.ids.capacity().checked_mul(std::mem::size_of::<i32>())
                    .and_then(|bytes| bytes.checked_add(exchange.routing.capacity().checked_mul(std::mem::size_of::<f32>())?))
                    .and_then(|bytes| bytes.checked_add(exchange.partials.capacity()))
                    .context("worker host exchange extent overflow")?;
                Some(cuteafd_core::ModelAdmissionResponse { rank: u32::try_from(config.rank)?,
                    resident_bytes, slices, runtime, mapped_ring_bytes: u64::try_from(model.ring_bytes(geometry)?)?,
                    host_exchange_bytes: u64::try_from(host_exchange_bytes)?,
                    row_indices_bytes: u64::try_from(row_indices.capacity().checked_mul(std::mem::size_of::<u32>())
                        .context("worker row index extent overflow")?)? })
            };
            cuteafd_transport::worker_selection::InstalledModelAdmission::new(config.rank, selection, report)
        })();
        let installed = match reported {
            Ok(installed) => installed,
            Err(error) => {
                let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
                return Err(error);
            }
        };
        installed.acknowledge(&mut stream)?;
        tracing::info!(rank = config.rank, world = config.world, layer_ids = ?weights.layer_ids()?, layers = weights.len(),
            selection_digest = %installed.selection().digest()?, "coordinator-selected worker weights ready");
        installed
    };
    listener.set_nonblocking(true)?;
    let (admit, incoming) = mpsc::sync_channel(2);
    let stop = Arc::new(AtomicBool::new(false));
    let _guard = Admission { stop: stop.clone() };
    let max_frame_bytes = config.max_frame_bytes;
    // Resolved once here: the admission thread and the poll loop must not read
    // the process environment per connection or per poll.
    let protocol_v2_timing = cuteafd_transport::protocol_v2_timing_from_env();
    let worker_rank = config.rank;
    let resident = weights.layer_ids()?;
    ensure!(resident == config.resident_layers(geometry.layers as usize)?, "loaded worker IDs differ from selected IDs");
    ensure!(weights.len() == resident.len(), "worker loaded range differs from grouping report");
    let loaded = worker_selection.selection().clone();
    let expert_grouping = executions[owners.decode.len()].installed_grouping(loaded, worker_rank)?;
    thread::Builder::new()
        .name("v41-roce-bootstrap".into())
        .spawn(move || {
            while !stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let admitted = LocalVerbsExpertConnection::accept_with_grouping(
                            stream,
                            max_frame_bytes,
                            protocol_v2_timing,
                            Arc::clone(&ring_budget),
                            Some(&worker_selection),
                            expert_grouping.as_ref(),
                        );
                        match admitted {
                            Ok(connection) => {
                                if admit.try_send(connection).is_err() {
                                    tracing::warn!("native RoCE admission queue full or stopped");
                                }
                            }
                            // A coordinator placing its flows on a bonded port
                            // probes, then connects its sessions separately.
                            Err(error) if error.is::<cuteafd_transport::worker_selection::WorkerSelectionOnly>() => {
                                tracing::debug!("acknowledged unchanged worker selection")
                            }
                            Err(error) if error.is::<cuteafd_transport::traffic::TrafficProbeOnly>() => {}
                            Err(error) if error.is::<cuteafd_transport::FlowProbesOnly>() => {
                                tracing::debug!("served RDMA flow probes")
                            }
                            Err(error) => tracing::warn!(%error, "native RoCE bootstrap failed"),
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => {
                        tracing::error!(%error, "native RoCE listener failed");
                        break;
                    }
                }
            }
        })?;
    let spark_world = config.spark_world()?;
    let executor_id = spark_world.topology().executor_id(config.rank)?;
    let mut connections = Vec::<LocalVerbsExpertConnection>::with_capacity(16);
    // Unique unit IDs survive swap-removal and preserve owner submission order
    // independently of each connection's ordered request/reply ring storage.
    let mut holding = Vec::<std::collections::VecDeque<HeldUnit>>::with_capacity(16);
    let deferred_limit = usize::from(owners.prefill.buffered_units) + owners.decode.len();
    // Structured startup evidence: one line per rank naming the native role and
    // logical intermediate this worker actually loaded. A captured log can then
    // be hashed and checked against the declared topology, instead of trusting a
    // hand-written value; the role comes from the same selection the loader used.
    let role = if config.exlr8_manifest.is_some() {
        None
    } else if catalog.exl3().is_some() {
        Some(config.exl3_selection(resident[0])?.role())
    } else {
        Some(config.selection(resident[0])?.role())
    };
    tracing::info!(
        rank = config.rank,
        world = config.world,
        role,
        format = ?catalog.format(),
        intermediate = weights.intermediate(),
        capacity = config.capacity,
        layer_ids = ?resident,
        layers = weights.len(),
        "native local RoCE expert worker ready"
    );
    // Requests arrive back-to-back while serving, so the loop spins: it is the
    // wakeup path and this is the GPU owner thread. A quiet connection switches
    // to the endpoint's QP completion event wait (`ibv_req_notify_cq` +
    // completion-channel poll, which still busy-polls briefly) with an
    // `idle_wait` upper bound. One wait covers one endpoint, so while idle the
    // loop blocks on one connection per pass and rotates; the pass itself still
    // sweeps every connection with a non-blocking poll, so a request on a
    // connection other than the waited one is picked up when that wait ends:
    // within one window, or two when the rotation wraps past it (three or more
    // connections). The wait must therefore end after `idle_wait`
    // (`LocalVerbsExpertConnection::poll`), not after an active exchange's
    // timeout. Any request or admission resets the idle timer.
    // Busy serving spins; after the shared idle-park time without a request
    // (`--idle-park-seconds`, 30 s by default) the loop parks on completion
    // events, so an idle worker uses no CPU.
    let idle_spin = cuteafd_transport::wait_policy::wait_policy().idle_park;
    let idle_wait = Duration::from_millis(100);
    let mut last_activity = Instant::now();
    let mut idle_cursor = 0usize;
    // One steady-state memory observation per admission event: after the owned
    // connection set changes, the next successful request triggers a single
    // sample. This is "first success after the admission event", not a proof
    // that the request arrived on the newly added connection.
    let mut pending_steady_log = false;
    let profile_queue = std::env::var("CUTEAFD_EXPERT_TIMING").is_ok_and(|value| value == "1");
    let mut prefill_overlap_high_water = 1usize;
    let result = (|| -> Result<()> { loop {
        let mut progressed = false;
        for (owner, execution) in executions.iter_mut().enumerate() {
            progressed |= queues.complete_ready(owner, execution)?;
        }
        if connections.is_empty() {
            connections.push(incoming.recv().context("native RoCE admission stopped")?);
            holding.push(std::collections::VecDeque::new());
            progressed = true;
            pending_steady_log = true;
            log_spark_memory_if_enabled(
                &library,
                &config,
                "connection owned after accept",
                Some(connections.len()),
                Some((ring_budget_log.used(), ring_budget_log.peak())),
            );
        } else if let Ok(connection) = incoming.try_recv() {
            progressed = true;
            if connections.len() < 16 {
                connections.push(connection);
                holding.push(std::collections::VecDeque::new());
                pending_steady_log = true;
                log_spark_memory_if_enabled(
                    &library,
                    &config,
                    "connection owned after accept",
                    Some(connections.len()),
                    Some((ring_budget_log.used(), ring_budget_log.peak())),
                );
            } else {
                tracing::warn!("native RoCE active connection limit reached");
            }
        }
        let waiting = !progressed
            && !connections.is_empty()
            && last_activity.elapsed() >= idle_spin;
        let wait_index = waiting.then(|| idle_cursor % connections.len());
        let mut index = 0;
        let mut intake = 0usize;
        let mut intake_stats = IntakeStats::default();
        let mut received_this_sweep = false;
        let mut sweep_blocked = false;
        while index < connections.len() {
            let mut execution_failed = false;
            let mut executed = false;
            let mut intake_blocked = false;
            let mut received = false;
            let wait = (intake == 0 && !executions.iter().any(Execution::has_staged)
                && wait_index == Some(index)).then_some(idle_wait);
            let mut serve = Serve { config: &config, executor_id, spark_world: &spark_world, resident: &resident,
                weights: &weights, exchange: &mut exchange, row_indices: &mut row_indices, skip_compute,
                log_ingress: pending_steady_log };
            let result = {
                // Completion bookkeeping follows each owner's submission FIFO,
                // independently of each connection's required reply order.
                let finished = match holding[index].front().copied() {
                    Some(unit) if queues.ready(unit) => connections[index].finish_deferred(|view, mapped, emit| {
                        let response = Serve::device_response(view, mapped, executor_id, &config, &spark_world)?
                            .with_pace(queues.pace(unit));
                        emit(ProtocolV2ExecutorResponseRef::Device(response))
                    }).and_then(|()| {
                        holding[index].pop_front();
                        queues.release(unit)?;
                        executed = true;
                        Ok(true)
                    }),
                    _ => Ok(false),
                };
                finished.and_then(|finished| {
                    if !intake_has_capacity(intake, executions.iter().any(Execution::has_staged),
                        || connections[index].try_deferred_capacity(deferred_limit))? {
                        intake_blocked = true;
                        intake_stats.send_ring_full += 1;
                        tracing::debug!(target: "cuteafd::spark_coalesce", peer = index, intake,
                            send_ring_full = intake_stats.send_ring_full, reason = "send_or_ring_full",
                            "Spark intake preflight refused");
                        return Ok(finished);
                    }
                    let mut launch_failed = false;
                    let mut input_address = 0;
                    let launched = connections[index].begin_deferred(wait, deferred_limit, |view, mapped| {
                        input_address = mapped.hidden_payload.ptr as usize;
                        let owner = execution_owner(&owners, view, mapped.execution_lane)?;
                        queues.admit(owner)?;
                        serve.launch(&mut executions[owner], view, mapped).map(|launched| {
                            (owner, launched)
                        }).map_err(|failure| {
                            launch_failed = failure.execution;
                            failure.error
                        })
                    });
                    match launched {
                        Ok(Some((owner, true))) => {
                            received = true;
                            intake_stats.received += 1;
                            holding[index].push_back(queues.record_request(owner, input_address)?);
                            let queued = queues.pending[owner].len();
                            // Count overlap only while the oldest GPU fence is
                            // incomplete, not merely while replies retain storage.
                            if profile_queue && owner == owners.decode.len()
                                && queued > prefill_overlap_high_water
                                && !executions[owner].unit_ready()? {
                                prefill_overlap_high_water = queued;
                                tracing::info!(target: "cuteafd::expert_timing", rank = config.rank, owner,
                                    queued_units = queued, oldest_complete = false,
                                    "Spark prefill owner overlap");
                            }
                            Ok(true)
                        }
                        // The host-only path drains its owner before using blocking
                        // uploads. Earlier replies still leave in connection order.
                        Ok(Some((owner, false))) => (|| -> Result<bool> {
                            received = true;
                            intake_stats.received += 1;
                            executions[owner].flush_staged()?;
                            executions[owner].synchronize()?;
                            queues.complete_ready(owner, &mut executions[owner])?;
                            while let Some(earlier) = holding[index].front().copied() {
                                executions[earlier.owner].flush_staged()?;
                                executions[earlier.owner].synchronize()?;
                                queues.complete_ready(earlier.owner, &mut executions[earlier.owner])?;
                                connections[index].finish_deferred(|view, mapped, emit| {
                                    let response = Serve::device_response(view, mapped, executor_id, &config, &spark_world)?;
                                    emit(ProtocolV2ExecutorResponseRef::Device(response))
                                })?;
                                holding[index].pop_front();
                                queues.release(earlier)?;
                            }
                            connections[index].finish_deferred(|view, mapped, emit| {
                                let served = serve.request(&mut executions[owner], view, mapped, emit);
                                execution_failed = served.as_ref().is_err_and(|failure| failure.execution);
                                executed = served.as_ref().is_ok_and(|served| served.executed);
                                served.map(|_| ()).map_err(|failure| failure.error)
                            })?;
                            Ok(true)
                        })(),
                        Ok(None) => {
                            if executions.iter().any(Execution::has_staged) { intake_stats.empty_second_receive += 1; }
                            Ok(finished)
                        },
                        Err(error) => { execution_failed = launch_failed; Err(error) }
                    }
                })
            };
            match result {
                Ok(processed) => {
                    progressed |= processed;
                    if processed && executed && pending_steady_log {
                        if census.enabled() {
                            library.cuda_device_synchronize()?;
                            census.sample("steady_after_first_expert_request", Some(&library))?;
                            let retained = cuteafd_ffi::memory_ledger::snapshot();
                            let detail = census.detail(&baseline, &retained, library.cuda_get_device()?)?;
                            tracing::info!(target: "cuteafd::spark_memory", record = %serde_json::to_string(&detail)?,
                                "Spark steady runtime census");
                        }
                        log_spark_memory_if_enabled(
                            &library,
                            &config,
                            "steady state sample after first post-admission request",
                            Some(connections.len()),
                            Some((ring_budget_log.used(), ring_budget_log.peak())),
                        );
                        pending_steady_log = false;
                    }
                    // A full reply ring cannot delay the staged wave behind a
                    // SEND wait. Launch it now, then resume the ordinary sweep.
                    if intake_blocked {
                        sweep_blocked = true;
                        break;
                    }
                    received_this_sweep |= received;
                    index += 1;
                    // Finish the original connection sweep before any extra
                    // polling: a decode on another peer is never postponed.
                    if index == connections.len() && continue_intake(received_this_sweep,
                        executions.iter().any(Execution::has_staged), intake, deferred_limit) {
                        intake += 1;
                        index = 0;
                        received_this_sweep = false;
                    }
                }
                Err(error) => {
                    // A failed launch can have queued copies before its error,
                    // including a unit not yet recorded in the host FIFO. Drain
                    // before any connection releases the borrowed mapped frames.
                    if execution_failed {
                        for execution in &executions {
                            crate::shared::decode_graph::fatal_drain(execution.synchronize(), "Spark failed launch");
                        }
                        return Err(error).context("native GPU execution failed");
                    }
                    // Staged frames have only queued copies: retire them before
                    // draining any launched members or releasing their peer rings.
                    let mut retained = std::collections::VecDeque::new();
                    for unit in holding[index].drain(..) {
                        if executions[unit.owner].cancel_staged(unit.input_address)? {
                            queues.cancel_staged(unit)?;
                        } else { retained.push_back(unit); }
                    }
                    holding[index] = retained;
                    for owner in 0..executions.len() {
                        if holding[index].iter().any(|unit| unit.owner == owner) {
                            executions[owner].flush_staged()?;
                            crate::shared::decode_graph::fatal_drain(executions[owner].synchronize(), "Spark removed peer");
                            queues.complete_ready(owner, &mut executions[owner])?;
                        }
                    }
                    for unit in holding[index].drain(..) { queues.release(unit)?; }
                    tracing::warn!(%error, "native RoCE peer removed");
                    connections.swap_remove(index);
                    holding.swap_remove(index);
                    intake = 0;
                }
            }
        }
        if executions.iter().any(Execution::has_staged) {
            tracing::debug!(target: "cuteafd::spark_coalesce", received = intake_stats.received,
                send_ring_full = intake_stats.send_ring_full, empty_second_receive = intake_stats.empty_second_receive,
                sweeps = intake + 1, "Spark intake flush decision");
        }
        for execution in &mut executions {
            // Keep arrivals staged behind active compute, not in its CUDA queue.
            // Staged work disables receive waits, so an empty intake keeps polling.
            let flush = (|| -> Result<()> {
                if flush_after_sweep(execution.has_staged(), sweep_blocked,
                    || execution.stream_idle())? {
                    execution.flush_staged()?;
                }
                Ok(())
            })();
            if let Err(error) = flush {
                for execution in &executions {
                    crate::shared::decode_graph::fatal_drain(execution.synchronize(), "Spark grouped launch");
                }
                return Err(error).context("native grouped execution failed");
            }
        }
        if progressed {
            last_activity = Instant::now();
        }
        if waiting {
            idle_cursor = idle_cursor.wrapping_add(1);
        } else {
            std::hint::spin_loop();
        }
    } })();
    for execution in &executions {
        crate::shared::decode_graph::fatal_drain(execution.synchronize(), "Spark owner teardown");
    }
    result
}

#[derive(Default)]
struct IntakeStats { received: u64, send_ring_full: u64, empty_second_receive: u64 }

fn flush_after_sweep(staged: bool, blocked: bool, idle: impl FnOnce() -> Result<bool>) -> Result<bool> {
    Ok(staged && (blocked || idle()?))
}

fn intake_has_capacity(sweep: usize, staged: bool, check: impl FnOnce() -> Result<bool>) -> Result<bool> {
    if sweep == 0 && !staged { return Ok(true); }
    check()
}

/// Extra intake follows a full ordinary sweep and never waits for an arrival.
pub(crate) fn continue_intake(received: bool, staged: bool, sweep: usize, bound: usize) -> bool {
    received && staged && sweep + 1 < bound
}

#[derive(Debug, thiserror::Error)]
enum ExecutionProtocolError {
    #[error("expert request has no source rows")]
    Empty,
    #[error("expert request mixes decode and prefill owners")]
    MixedOwners,
    #[error("Spark execution owner {owner} already retains its admitted {limit} units")]
    Full { owner: usize, limit: usize },
}

fn execution_owner(owners: &cuteafd_transport::worker_selection::SparkExecutions,
    view: &cuteafd_transport::ExpertProtocolV2RequestView<'_>, lane: u32) -> Result<usize> {
    let source = view.row(0).map_err(|_| ExecutionProtocolError::Empty)?.source_kind;
    let owner = cuteafd_transport::worker_selection::execution_owner(owners, source, lane, view.header.row_count)?;
    for row in 1..view.header.row_count as usize {
        if cuteafd_transport::worker_selection::execution_owner(owners, view.row(row)?.source_kind, lane, view.header.row_count)? != owner {
            return Err(ExecutionProtocolError::MixedOwners.into());
        }
    }
    Ok(match owner {
        cuteafd_transport::worker_selection::ExecutionOwner::Decode(lane) => lane,
        cuteafd_transport::worker_selection::ExecutionOwner::Prefill => owners.decode.len(),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct HeldUnit { owner: usize, id: u64, input_address: usize }

/// GPU publication follows owner FIFO; transport emission follows connection
/// FIFO. A completed unit still retains its own I/O and admission credit, and
/// its measured pace until its reply carries it.
struct OwnerQueues {
    limits: Vec<usize>,
    retained: Vec<usize>,
    pending: Vec<std::collections::VecDeque<HeldUnit>>,
    completed: std::collections::HashMap<HeldUnit, Option<crate::shared::experts::execution::timing::UnitPace>>,
    next_id: u64,
    /// The prefill owner's index: only its units report a pace (#762).
    prefill: usize,
}
impl OwnerQueues {
    fn new(owners: &cuteafd_transport::worker_selection::SparkExecutions) -> Self {
        let limits: Vec<_> = owners.owners().map(|extent| usize::from(extent.buffered_units)).collect();
        let capacity = limits.iter().sum();
        Self { retained: vec![0; limits.len()],
            pending: limits.iter().map(|&limit| std::collections::VecDeque::with_capacity(limit)).collect(),
            limits, completed: std::collections::HashMap::with_capacity(capacity), next_id: 0,
            prefill: owners.decode.len() }
    }
    fn admit(&self, owner: usize) -> Result<()> {
        if self.retained[owner] >= self.limits[owner] {
            return Err(ExecutionProtocolError::Full { owner, limit: self.limits[owner] }.into());
        }
        Ok(())
    }
    #[cfg(test)]
    fn record(&mut self, owner: usize) -> Result<HeldUnit> { self.record_request(owner, self.next_id as usize) }
    fn record_request(&mut self, owner: usize, input_address: usize) -> Result<HeldUnit> {
        self.admit(owner)?;
        let unit = HeldUnit { owner, id: self.next_id, input_address };
        self.next_id = self.next_id.checked_add(1).context("expert unit identity exhausted")?;
        self.retained[owner] += 1;
        self.pending[owner].push_back(unit);
        Ok(unit)
    }
    fn cancel_staged(&mut self, unit: HeldUnit) -> Result<()> {
        ensure!(!self.ready(unit), "cannot cancel a published unit as staged");
        let position = self.pending[unit.owner].iter().position(|&pending| pending == unit)
            .context("staged unit missing from owner FIFO")?;
        self.pending[unit.owner].remove(position);
        self.retained[unit.owner] = self.retained[unit.owner].checked_sub(1).context("staged credit underflow")?;
        Ok(())
    }
    fn ready(&self, unit: HeldUnit) -> bool { self.completed.contains_key(&unit) }
    /// A completed unit's pace for its reply header: prefill owner units only.
    fn pace(&self, unit: HeldUnit) -> Option<(u32, u32)> {
        self.completed.get(&unit).copied().flatten().filter(|_| unit.owner == self.prefill)
            .map(|pace| (pace.busy_us, pace.idle_us))
    }
    fn complete_ready(&mut self, owner: usize, execution: &mut Execution<'_, '_>) -> Result<bool> {
        let mut progressed = false;
        while !self.pending[owner].is_empty() && execution.unit_ready()? {
            let pace = execution.complete()?;
            self.publish_oldest(owner, pace)?;
            progressed = true;
        }
        Ok(progressed)
    }
    fn publish_oldest(&mut self, owner: usize,
        pace: Option<crate::shared::experts::execution::timing::UnitPace>) -> Result<()> {
        let unit = self.pending[owner].pop_front().context("expert owner has no queued unit")?;
        ensure!(self.completed.insert(unit, pace).is_none(), "expert unit published twice");
        Ok(())
    }
    fn release(&mut self, unit: HeldUnit) -> Result<()> {
        ensure!(self.completed.remove(&unit).is_some(), "expert I/O released before unit completion");
        self.retained[unit.owner] = self.retained[unit.owner].checked_sub(1)
            .context("expert unit credit underflow")?;
        Ok(())
    }
}

/// A request served synchronously: whether it ran the experts.
struct Served { executed: bool }

/// A failed request: whether the GPU execution itself failed (fatal) rather
/// than the peer or the wire.
struct ServeFailure { error: anyhow::Error, execution: bool }

impl From<anyhow::Error> for ServeFailure {
    fn from(error: anyhow::Error) -> Self { Self { error, execution: false } }
}

/// Everything one request needs from the GPU owner besides its execution slot.
struct Serve<'s, 'w, 'a> {
    config: &'s NativeExpertServiceConfig,
    executor_id: u64,
    spark_world: &'s SparkWorld,
    resident: &'s [usize],
    weights: &'w Weights<'a>,
    exchange: &'s mut HostExpertExchange,
    row_indices: &'s mut Vec<u32>,
    skip_compute: bool,
    log_ingress: bool,
}

impl<'w, 'a> Serve<'_, 'w, 'a> {
    fn parse<'v>(view: &'v cuteafd_transport::ExpertProtocolV2RequestView<'_>, config: &NativeExpertServiceConfig,
        spark_world: &SparkWorld) -> Result<BackboneRequest<'v>> {
        // A topology-bound worker admits only the ownership-aware request
        // contract; every other family is a protocol mismatch, not a fallback.
        match spark_world.ownership_topology() {
            Some(topology) => BackboneRequest::parse_native_group(view.frame_bytes(), config.capacity, topology),
            None => BackboneRequest::parse(view.frame_bytes(), config.capacity),
        }
    }

    /// The device response a launched request writes into its send slot.
    fn device_response(view: &cuteafd_transport::ExpertProtocolV2RequestView<'_>,
        mapped: cuteafd_transport::ProtocolV2RequestDevicePayload, executor_id: u64,
        config: &NativeExpertServiceConfig, spark_world: &SparkWorld)
        -> Result<cuteafd_transport::ExpertProtocolV2DeviceResponseRef<'static>> {
        let request = Self::parse(view, config, spark_world)?;
        let slot = mapped.response_slot.context("deferred request lost its response slot")?;
        let prefix = cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let bytes = request.plane_bytes()?;
        ensure!(request.permits_device_response() && slot.bytes >= prefix + bytes,
            "deferred request does not fit its response slot");
        // SAFETY: the payload follows the header inside the mapped send slot.
        let output = cuteafd_ffi::CuteafdDeviceBuffer { ptr: unsafe { slot.ptr.cast::<u8>().add(prefix) }.cast(),
            bytes, ..slot };
        request.response_device(executor_id, output)
    }

    /// Queue a device-answerable request on `execution` without waiting
    /// (`true`); `false` when it must be served synchronously instead.
    fn launch(&mut self, execution: &mut Execution<'w, 'a>,
        view: &cuteafd_transport::ExpertProtocolV2RequestView<'_>,
        mapped: cuteafd_transport::ProtocolV2RequestDevicePayload) -> std::result::Result<bool, ServeFailure> {
        if view.header.row_count > self.config.capacity || self.skip_compute || mapped.response_slot.is_none() {
            return Ok(false);
        }
        let request = Self::parse(view, self.config, self.spark_world)?;
        if !request.permits_device_response() { return Ok(false); }
        let layer = self.resident.binary_search(&(request.layer() as usize))
            .map_err(|_| anyhow::anyhow!("requested expert layer is not resident on this Spark"))?;
        execution.bind_layer(self.weights, layer)?;
        // SAFETY: the transport holds both slots through finish_deferred,
        // which runs only after this unit's completion fence.
        let response = unsafe { execution.execute_mapped(&request, self.executor_id, self.exchange, mapped) }
            .map_err(|error| ServeFailure { error, execution: true })?;
        Ok(response.is_some())
    }

    /// Serve one request to completion, emitting its response.
    fn request(&mut self, execution: &mut Execution<'w, 'a>,
        view: &cuteafd_transport::ExpertProtocolV2RequestView<'_>,
        mapped: cuteafd_transport::ProtocolV2RequestDevicePayload,
        emit: &mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>)
        -> std::result::Result<Served, ServeFailure> {
        let (config, executor_id) = (self.config, self.executor_id);
        // The wire parser has validated framing and identities. Reject
        // oversize work with a final response, not an unanswered peer drop.
        if view.header.row_count > config.capacity {
            let response = cuteafd_transport::protocol_v2::ExpertProtocolV2ResponseRef::row_capacity_error(
                &view.header, config.capacity, executor_id)?;
            tracing::warn!(rows = view.header.row_count, capacity = config.capacity,
                request_id = view.header.request_id, "native expert request rejected");
            emit(ProtocolV2ExecutorResponseRef::Host(response))?;
            return Ok(Served { executed: false });
        }
        let request = Self::parse(view, config, self.spark_world)?;
        if self.log_ingress {
            tracing::info!(rank = config.rank, ingress_dtype = ?view.header.hidden_dtype,
                capacity = config.capacity, "Spark endpoint ingress geometry");
        }
        if self.skip_compute {
            if let Some(slot) = mapped.response_slot {
                let prefix = cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
                let bytes = request.plane_bytes()?;
                if request.permits_device_response() && slot.bytes >= prefix + bytes {
                    let output = cuteafd_ffi::CuteafdDeviceBuffer {
                        // SAFETY: the payload follows the header inside the mapped slot.
                        ptr: unsafe { slot.ptr.cast::<u8>().add(prefix) }.cast(), bytes, ..slot
                    };
                    emit(ProtocolV2ExecutorResponseRef::Device(request.response_device(executor_id, output)?))?;
                    return Ok(Served { executed: false });
                }
            }
        }
        let layer = self.resident.binary_search(&(request.layer() as usize))
            .map_err(|_| anyhow::anyhow!("requested expert layer is not resident on this Spark"))?;
        execution.bind_layer(self.weights, layer)?;
        if mapped.response_slot.is_some() {
            // The mapped frame's hidden rows are device-visible, so the
            // worker copies them on its stream instead of uploading.
            // SAFETY: the transport retains the mapped slots through execution and emission.
            let response = unsafe { execution.execute_mapped(&request, executor_id, self.exchange, mapped) }
                .map_err(|error| ServeFailure { error, execution: true })?;
            if let Some(response) = response {
                execution.synchronize().and_then(|()| execution.complete().map(|_| ()))
                    .map_err(|error| ServeFailure { error, execution: true })?;
                emit(ProtocolV2ExecutorResponseRef::Device(response))?;
                return Ok(Served { executed: request.rows() > 0 });
            }
        }
        let mut emit_failed = false;
        let result = execution.execute_host_chunks(&request, executor_id, self.exchange, self.row_indices,
            config.max_frame_bytes, |response| {
                let result = emit(ProtocolV2ExecutorResponseRef::Host(response));
                emit_failed |= result.is_err();
                result
            });
        match result {
            Ok(()) => Ok(Served { executed: request.rows() > 0 }),
            Err(error) => Err(ServeFailure { error, execution: !emit_failed }),
        }
    }
}

#[cfg(test)]
mod owner_tests {
    use super::*;
    use cuteafd_transport::worker_selection::{DecodeExecution, PrefillExecution, SparkExecutions};

    fn owners() -> SparkExecutions {
        SparkExecutions { decode: vec![DecodeExecution { capacity: 80, compiled_capacity: 80 }; 2],
            prefill: PrefillExecution { capacity: 4096, compiled_capacity: 4096, unit_rows: 2752, buffered_units: 3 } }
    }

    #[test]
    fn request_sources_route_to_typed_owners_and_reject_mixed_or_oversize_units() -> Result<()> {
        use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RequestView,
            ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry, ExpertV2Dtype, ExpertV2SourceKind};
        use ExpertV2SourceKind::*;
        let geometry = cuteafd_core::expert_geometry();
        let frame = |sources: &[ExpertV2SourceKind]| -> Result<Vec<u8>> {
            let rows = u32::try_from(sources.len())?;
            let request = ExpertProtocolV2Request::new(91, 17, 3, geometry.hidden,
                ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                sources.iter().enumerate().map(|(row, &source_kind)| ExpertProtocolV2RowDescriptor {
                    row_id: row as u64, source_kind, source_request_id: 1, token_position: row as u64,
                    route_offset: row as u32 * geometry.topk, route_count: geometry.topk }).collect(),
                (0..rows * geometry.topk).map(|route| ExpertProtocolV2RouteEntry {
                    row_index: route / geometry.topk, expert_id: route % geometry.experts,
                    gate_weight: 1.0 }).collect(),
                vec![0; rows as usize * (geometry.hidden as usize + geometry.hidden as usize / 32)])?;
            request.encode()
        };
        for (sources, lane, expected) in [(vec![Prefill; 2], 1, 2),
            (vec![Benchmark; 2], 9, 2), (vec![Decode, MtpVerify], 1, 1)] {
            let frame = frame(&sources)?;
            let view = ExpertProtocolV2RequestView::parse(&frame)?;
            assert_eq!(execution_owner(&owners(), &view, lane)?, expected);
        }
        let mixed = frame(&[Decode, Prefill])?;
        let error = execution_owner(&owners(), &ExpertProtocolV2RequestView::parse(&mixed)?, 0).unwrap_err();
        assert!(matches!(error.downcast_ref::<ExecutionProtocolError>(), Some(ExecutionProtocolError::MixedOwners)));
        for (sources, lane) in [(vec![Decode; 81], 0), (vec![Decode], 2), (vec![Prefill; 2753], 0)] {
            let frame = frame(&sources)?;
            assert!(execution_owner(&owners(), &ExpertProtocolV2RequestView::parse(&frame)?, lane).is_err());
        }
        Ok(())
    }

    #[test]
    fn owner_credit_remains_held_until_each_reply_is_released() -> Result<()> {
        let mut queues = OwnerQueues::new(&owners());
        let units = (0..3).map(|_| queues.record(2)).collect::<Result<Vec<_>>>()?;
        let error = queues.admit(2).unwrap_err();
        assert!(matches!(error.downcast_ref::<ExecutionProtocolError>(),
            Some(ExecutionProtocolError::Full { owner: 2, limit: 3 })));
        assert!(queues.release(units[0]).is_err());
        queues.publish_oldest(2, None)?;
        assert!(queues.ready(units[0]) && !queues.ready(units[1]));
        assert!(queues.admit(2).is_err(), "completion alone cannot recycle retained I/O");
        queues.release(units[0])?;
        let replacement = queues.record(2)?;
        assert_ne!(replacement, units[0]);
        for unit in [units[1], units[2], replacement] {
            queues.publish_oldest(2, None)?;
            assert!(queues.ready(unit));
            queues.release(unit)?;
        }
        assert_eq!(queues.retained, vec![0; 3]);
        Ok(())
    }

    #[test]
    fn lane_set_transports_use_distinct_decode_credits_and_one_prefill_owner() -> Result<()> {
        use cuteafd_transport::expert::{SparkExperts, SparkWorld};
        use cuteafd_transport::worker_selection::{execution_owner, ExecutionOwner};
        use cuteafd_transport::{ExpertV2SourceKind, TcpTransportConfig};
        let mut admitted = owners();
        admitted.prefill.unit_rows = 2048;
        admitted.prefill.buffered_units = 2;
        let mut queues = OwnerQueues::new(&admitted);
        let set = crate::shared::serve::lanes::LaneSet::default();
        let peers = ["127.0.0.1:19455".parse()?];
        let mut transports = set.try_map(|lane| SparkExperts::new_world(SparkWorld::canonical(1)?,
            &peers, 2048, TcpTransportConfig::default(), u32::try_from(lane.index())?))?;
        let mut units = Vec::new();
        for (lane, transport) in transports.iter_mut().enumerate() {
            assert_eq!(transport.execution_lane(), lane as u32);
            for source in [ExpertV2SourceKind::Decode, ExpertV2SourceKind::MtpVerify] {
                assert_eq!(execution_owner(&admitted, source, transport.execution_lane(), 62)?,
                    ExecutionOwner::Decode(lane));
            }
            units.push(queues.record(lane)?);
            assert!(queues.admit(lane).is_err());
            assert_eq!(execution_owner(&admitted, ExpertV2SourceKind::Prefill,
                transport.execution_lane(), 2048)?, ExecutionOwner::Prefill);
            units.push(queues.record(admitted.decode.len())?);
            transport.reset_connections();
            assert_eq!(transport.execution_lane(), lane as u32);
        }
        assert_eq!(queues.retained, [1, 1, 2]);
        assert!(queues.admit(admitted.decode.len()).is_err());
        for unit in units {
            queues.publish_oldest(unit.owner, None)?;
            queues.release(unit)?;
        }
        assert_eq!(queues.retained, [0; 3]);
        Ok(())
    }

    #[test]
    fn only_prefill_owner_units_carry_their_pace_until_release() -> Result<()> {
        use crate::shared::experts::execution::timing::UnitPace;
        let mut queues = OwnerQueues::new(&owners());
        let (decode, prefill) = (queues.record(0)?, queues.record(2)?);
        queues.publish_oldest(0, Some(UnitPace { busy_us: 5, idle_us: 6 }))?;
        queues.publish_oldest(2, Some(UnitPace { busy_us: 700, idle_us: 30 }))?;
        assert_eq!(queues.pace(decode), None, "decode owners never report a pace");
        assert_eq!(queues.pace(prefill), Some((700, 30)));
        queues.release(prefill)?;
        assert_eq!(queues.pace(prefill), None);
        queues.release(decode)?;
        Ok(())
    }

    #[test]
    fn decode_lanes_never_borrow_prefill_or_other_decode_credit() -> Result<()> {
        let mut queues = OwnerQueues::new(&owners());
        let first = queues.record(0)?;
        assert!(queues.admit(0).is_err());
        assert!(queues.admit(1).is_ok() && queues.admit(2).is_ok());
        let second = queues.record(1)?;
        queues.publish_oldest(1, None)?;
        assert!(!queues.ready(first) && queues.ready(second));
        queues.release(second)?;
        queues.publish_oldest(0, None)?;
        queues.release(first)?;
        Ok(())
    }

    #[test]
    fn the_owner_loop_parks_after_the_shared_idle_time() {
        let source = include_str!("local.rs");
        assert!(source.contains("let idle_spin = cuteafd_transport::wait_policy::wait_policy().idle_park;"));
    }
}
