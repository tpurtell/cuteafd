//! The GPU owner polls admitted QPs and sends results without work queues.
use super::*;
use cuteafd_transport::{LocalVerbsExpertConnection, ProtocolV2ExecutorResponseRef};
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
        config.capacity = model.capacity;
        config.max_frame_bytes = model.rings.max_frame_bytes;
        config.rdma_endpoints = model.endpoints.len();
        config.bf16_ingress = model.endpoints.iter().any(|endpoint|
            matches!(endpoint.ingress, cuteafd_core::SparkIngress::Bf16
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
            snapshot: dir.path().into(), exlr8_manifest: None, rank: 0, world: 6, model_admission: None, selected_layers: None, capacity: 4096, device_budget: 121 << 30, max_frame_bytes: 64 << 20,
            rdma_endpoints: 3, declared_world: cuteafd_transport::expert::SparkWorld::canonical(6).unwrap(), native_spark_tp2: false, bf16_ingress: false };
        let model = cuteafd_transport::worker_selection::ModelAdmission { capacity: 80, compiled_capacity: 80,
            endpoints: vec![cuteafd_core::SparkEndpoint { capacity: 80,
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
        covered.capacity = 2048;
        covered.compiled_capacity = 4096;
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
        config.rank < config.world && (1..=6).contains(&config.world),
        "native rank must be below the launched Spark world"
    );
    // The checkpoint fixes the process expert geometry before the native
    // library loads, so its helpers and every wire size agree with it.
    let inventory = cuteafd_loader::plan::Checkpoint::inventory(&config.snapshot)?;
    let v41 = cuteafd_loader::plan::family::detect(&inventory).is_some_and(|family| family.id() == "deepseek_v41");
    let mut catalog = if v41 && inventory.quantization().is_none_or(|quant| quant["quant_method"] != "exl3") {
        cuteafd_loader::read_official_v41_spark_catalog(&config.snapshot, config.rank, config.world)?
    } else {
        cuteafd_loader::read_expert_catalog(&config.snapshot)?
    };
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    let topology = crate::shared::spark_topology::for_checkpoint(
        config.declared_world.ownership_topology(), config.world, v41 && catalog.exl3().is_some());
    config.declared_world = crate::shared::spark_topology::world(topology, config.world)?;
    crate::shared::spark_topology::require_wire(config.spark_world()?, &catalog)?;
    let geometry = catalog.routed_experts().geometry()?;
    config.native_spark_tp2 = config.world == 2 && geometry.family() == Some("dsv4f")
        && catalog.exl3().is_none() && catalog.nvfp4().is_none() && catalog.fp8().is_none();
    cuteafd_core::set_expert_geometry(geometry).map_err(|fixed| {
        anyhow::anyhow!("expert geometry is already {fixed:?}; the checkpoint needs {geometry:?}")
    })?;
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
        load_weights(&library, &catalog, &config, &mut modules)
    };
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
    let allocated = {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/workspace");
        weights.execution(&library, &config, remaining, &mut modules)
    };
    let mut execution = match allocated {
        Ok(execution) => execution,
        Err(error) => {
            let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
            return Err(error);
        }
    };
    census.sample("after_retained_arenas", Some(&library))?;
    let mut exchange = HostExpertExchange::with_output(config.capacity, config.exlr8_manifest.is_some())?;
    let mut row_indices = vec![0; config.capacity as usize];
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
    let expert_grouping = execution.installed_grouping(loaded, worker_rank)?;
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
    let idle_spin = Duration::from_secs(30);
    let idle_wait = Duration::from_millis(100);
    let mut last_activity = Instant::now();
    let mut idle_cursor = 0usize;
    // One steady-state memory observation per admission event: after the owned
    // connection set changes, the next successful request triggers a single
    // sample. This is "first success after the admission event", not a proof
    // that the request arrived on the newly added connection.
    let mut pending_steady_log = false;
    loop {
        let mut progressed = false;
        if connections.is_empty() {
            connections.push(incoming.recv().context("native RoCE admission stopped")?);
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
        while index < connections.len() {
            let mut execution_failed = false;
            let mut executed = false;
            let wait = (wait_index == Some(index)).then_some(idle_wait);
            let result = connections[index].poll(wait, |view, mapped, emit| {
                // The wire parser has validated framing and identities. Reject
                // oversize work with a final response, not an unanswered peer drop.
                if view.header.row_count > config.capacity {
                    let response = cuteafd_transport::protocol_v2::ExpertProtocolV2ResponseRef::row_capacity_error(
                        &view.header, config.capacity, executor_id)?;
                    tracing::warn!(rows = view.header.row_count, capacity = config.capacity,
                        request_id = view.header.request_id, "native expert request rejected");
                    return emit(ProtocolV2ExecutorResponseRef::Host(response));
                }
                // A topology-bound worker admits only the ownership-aware
                // request contract; every other family is a protocol mismatch,
                // not a silent fallback.
                let request = match spark_world.ownership_topology() {
                    Some(topology) => BackboneRequest::parse_native_group(
                        view.frame_bytes(),
                        config.capacity,
                        topology,
                    )?,
                    None => BackboneRequest::parse(view.frame_bytes(), config.capacity)?,
                };
                if pending_steady_log {
                    tracing::info!(rank = config.rank, ingress_dtype = ?view.header.hidden_dtype,
                        capacity = config.capacity, "Spark endpoint ingress geometry");
                }
                if skip_compute {
                    if let Some(slot) = mapped.response_slot {
                        let prefix = cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
                        let bytes = request.plane_bytes()?;
                        if request.permits_device_response() && slot.bytes >= prefix + bytes {
                            let output = cuteafd_ffi::CuteafdDeviceBuffer {
                                // SAFETY: the payload follows the header inside the mapped slot.
                                ptr: unsafe { slot.ptr.cast::<u8>().add(prefix) }.cast(), bytes, ..slot
                            };
                            return emit(ProtocolV2ExecutorResponseRef::Device(
                                request.response_device(executor_id, output)?));
                        }
                    }
                }
                let layer = resident.binary_search(&(request.layer() as usize))
                    .map_err(|_| anyhow::anyhow!("requested expert layer is not resident on this Spark"))?;
                execution.bind_layer(&weights, layer)?;
                if mapped.response_slot.is_some() {
                    // The mapped frame's hidden rows are device-visible, so the
                    // worker copies them on its stream instead of uploading.
                    // SAFETY: the transport retains the mapped slots through execution and emission.
                    let response = unsafe { execution.execute_mapped(&request,
                        executor_id, &mut exchange, mapped) };
                    let response = match response {
                        Ok(response) => response,
                        Err(error) => { execution_failed = true; return Err(error); }
                    };
                    if let Some(response) = response {
                        emit(ProtocolV2ExecutorResponseRef::Device(response))?;
                        executed = request.rows() > 0;
                        return Ok(());
                    }
                }
                let mut emit_failed = false;
                let result = execution.execute_host_chunks(
                    &request,
                    executor_id,
                    &mut exchange,
                    &mut row_indices,
                    config.max_frame_bytes,
                    |response| {
                        let result = emit(ProtocolV2ExecutorResponseRef::Host(response));
                        emit_failed |= result.is_err();
                        result
                    },
                );
                execution_failed = result.is_err() && !emit_failed;
                executed = result.is_ok() && request.rows() > 0;
                result
            });
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
                    index += 1;
                }
                Err(error) => {
                    if execution_failed {
                        return Err(error).context("native GPU execution failed");
                    }
                    tracing::warn!(%error, "native RoCE peer removed");
                    connections.swap_remove(index);
                }
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
    }
}
