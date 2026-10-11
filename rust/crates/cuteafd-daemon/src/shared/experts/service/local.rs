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
    ensure!(selection.ranges().len() <= 1,
        "disjoint worker layer selection needs a sparse native loader");
    if let Some(range) = selection.ranges().first() {
        let allowed = config.resident_layers(layers)?;
        ensure!(range.first as usize >= first_routed && range.first as usize >= allowed.start
            && range.end as usize <= allowed.end, "solved worker selection is outside launched resident range");
        config.first_layer = range.first as usize;
        config.last_layer = Some(range.end as usize - 1);
    }
    Ok(())
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    #[test]
    fn solved_worker_suffix_controls_loaded_and_admitted_range() {
        use cuteafd_transport::worker_selection::{WorkerIdentity, WorkerSelection};
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors.index.json"), "{}").unwrap();
        let geometry = cuteafd_core::ExpertGeometry::MIMO_V26_PRO;
        let identity = WorkerIdentity::from_snapshot(dir.path(), geometry, 6, None).unwrap();
        // Native config uses the same range consumed by every backend planner.
        let mut config = NativeExpertServiceConfig { encoder: None, audio_encoder: None,
            library: "/native.so".into(), exl3_aot_dir: None, exl3_schedule: Default::default(), fp8_package: None,
            snapshot: dir.path().into(), rank: 0, world: 6, first_layer: 1, last_layer: None,
            placement_handshake: true, capacity: 4096, device_budget: 121 << 30, max_frame_bytes: 64 << 20,
            topology: None, native_spark_tp2: false, bf16_ingress: false };
        let selection = WorkerSelection::new(identity.clone(), &(13..70).collect::<Vec<_>>()).unwrap();
        apply_worker_selection(&mut config, &selection, 1, 70).unwrap();
        assert_eq!(config.resident_layers(70).unwrap(), 13..70);
        assert_eq!(config.selection(13).unwrap(), ExpertLayer::BackboneExl3Tp { layer: 13, rank: 0, world: 6 });
        let disjoint = WorkerSelection::new(identity.clone(), &[13, 15]).unwrap();
        assert!(apply_worker_selection(&mut config, &disjoint, 1, 70).is_err());
        let before = WorkerSelection::new(identity.clone(), &[0]).unwrap();
        assert!(apply_worker_selection(&mut config, &before, 1, 70).is_err());
        let empty = WorkerSelection::new(identity, &[]).unwrap();
        apply_worker_selection(&mut config, &empty, 1, 70).unwrap();
        assert!(empty.ranges().is_empty());
    }
}

fn run_empty_selection(listener: TcpListener, rank: usize,
    selection: &cuteafd_transport::worker_selection::WorkerSelection) -> Result<()> {
    listener.set_nonblocking(false)?;
    for accepted in listener.incoming() {
        let mut stream = accepted?;
        let result = (|| -> Result<()> {
            let repeated = cuteafd_transport::worker_selection::receive_worker_selection(&stream, rank, Duration::from_secs(30))?;
            ensure!(&repeated == selection, "empty worker selection is immutable until restart");
            cuteafd_transport::worker_selection::acknowledge_worker_selection(&mut stream, rank, selection)
        })();
        if let Err(error) = result {
            let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
            tracing::warn!(%error, "empty worker rejects expert traffic or changed selection");
        }
    }
    Ok(())
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
        config.rank < config.world && matches!(config.world, 1 | 2 | 3 | 4 | 6),
        "native rank must be below the launched Spark world"
    );
    ensure!(
        matches!(config.capacity, 1 | 16 | 80 | 256 | 1024 | 4096),
        "unsupported native capacity"
    );
    // The checkpoint fixes the process expert geometry before the native
    // library loads, so its helpers and every wire size agree with it.
    let inventory = cuteafd_loader::plan::Checkpoint::inventory(&config.snapshot)?;
    let v41 = cuteafd_loader::plan::family::detect(&inventory).is_some_and(|family| family.id() == "deepseek_v41");
    let catalog = if v41 && inventory.quantization().is_none_or(|quant| quant["quant_method"] != "exl3") {
        cuteafd_loader::read_official_v41_spark_catalog(&config.snapshot, config.rank, config.world)?
    } else {
        cuteafd_loader::read_expert_catalog(&config.snapshot)?
    };
    let geometry = catalog.routed_experts().geometry()?;
    config.bf16_ingress = geometry.family() == Some("v41") && catalog.nvfp4().is_some();
    config.native_spark_tp2 = config.world == 2 && geometry.family() == Some("dsv4f")
        && catalog.exl3().is_none() && catalog.nvfp4().is_none() && catalog.fp8().is_none();
    cuteafd_core::set_expert_geometry(geometry).map_err(|fixed| {
        anyhow::anyhow!("expert geometry is already {fixed:?}; the checkpoint needs {geometry:?}")
    })?;
    let minimum_frame = 128 + geometry.row_bytes() as usize + 40 + geometry.topk as usize * 12;
    ensure!(
        (minimum_frame..=64 * 1024 * 1024).contains(&config.max_frame_bytes),
        "invalid native frame budget"
    );
    // The mapped rings each accepted endpoint pins are bounded by the
    // capacity-sized two-endpoint allowance that admission already reserved and
    // proved against the device budget and actual free memory. A stale or larger
    // peer advertisement is rejected at accept instead of overcommitting.
    let enforced_ring_budget = geometry.family() == Some("v41");
    let ring_budget = Some(worker_ring_budget(&config, geometry)?);
    tracing::info!(enforced_ring_budget, bf16_ingress = config.bf16_ingress,
        ring_budget_bytes = ring_budget.as_ref().unwrap().limit(), "Spark ring admission policy");
    // The vision owner loads in this process/device's primary CUDA context,
    // never beside expertd in a second process. Charge it before expert admission.
    let (_encoder, _audio_encoder, reserved) = crate::shared::vision::worker::start_encoders(
        config.encoder.as_ref(), config.audio_encoder.as_ref(), &config.snapshot,
        config.library.clone(), config.device_budget as u64)?;
    config.device_budget = config.device_budget.checked_sub(usize::try_from(reserved)?)
        .context("media reservation exceeds Spark budget")?;
    // Media may be admitted first, but expert allocation waits for the one
    // coordinator-authoritative selection. Bootstrap-ready is not weights-ready.
    let mut bootstrap = if config.placement_handshake {
        let listener = TcpListener::bind(listen)?;
        listener.set_nonblocking(true)?;
        tracing::info!(rank = config.rank, world = config.world, "native worker bootstrap ready");
        let deadline = Instant::now() + Duration::from_secs(1200);
        let mut stream = loop {
            ensure!(Instant::now() < deadline, "coordinator worker selection timed out");
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(100)),
                Err(error) => return Err(error.into()),
            }
        };
        let selected = (|| -> Result<_> {
            let selection = cuteafd_transport::worker_selection::receive_worker_selection(&stream, config.rank, Duration::from_secs(1200))?;
            let identity = cuteafd_transport::worker_selection::WorkerIdentity::from_snapshot(
                &config.snapshot, geometry, config.world, config.topology)?;
            selection.validate_for(&identity)?;
            apply_worker_selection(&mut config, &selection, catalog.routed_experts().first_layer,
                catalog.routed_experts().layers)?;
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
            cuteafd_transport::worker_selection::acknowledge_worker_selection(&mut stream, config.rank, &selection)?;
            tracing::info!(rank = config.rank, layers = 0, resident_bytes = 0, "native empty expert worker ready");
            return run_empty_selection(listener, config.rank, &selection);
        }
        Some((listener, stream, selection))
    } else { None };
    // SAFETY: this owner keeps the library loaded through weights/execution Drop.
    let library = unsafe { NativeLibrary::load(&config.library) }?;
    let loaded = {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        load_weights(&library, &catalog, &config)
    };
    let (weights, remaining) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            if let Some((_, stream, _)) = bootstrap.as_mut() {
                let _ = cuteafd_transport::worker_selection::reject_worker_selection(stream, &error);
            }
            return Err(error);
        }
    };
    // The loaders' synchronous uploads grew the library's pinned staging buffer
    // to the largest projection (hundreds of MB of unified memory on GB10);
    // requests stage far smaller rows, so drop it and let them regrow it.
    let released = library.release_sync_h2d_staging()?;
    // CUTEAFD_SPARK_DROP_PAGE_CACHE=0 keeps the checkpoint pages cached.
    let cached_before = crate::shared::memory_report::cached_bytes();
    let advised = if std::env::var("CUTEAFD_SPARK_DROP_PAGE_CACHE").is_ok_and(|v| v == "0") { 0 }
        else { cuteafd_loader::page_cache::drop_snapshot_pages(&config.snapshot) };
    tracing::info!(released_bytes = released, page_cache_advised_bytes = advised, cached_before,
        cached_after = crate::shared::memory_report::cached_bytes(),
        "released load-time pinned upload staging and checkpoint page cache");
    let allocated = {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/workspace");
        weights.execution(&library, &config, remaining)
    };
    let mut execution = match allocated {
        Ok(execution) => execution,
        Err(error) => {
            if let Some((_, stream, _)) = bootstrap.as_mut() {
                let _ = cuteafd_transport::worker_selection::reject_worker_selection(stream, &error);
            }
            return Err(error);
        }
    };
    let mut exchange = HostExpertExchange::new(config.capacity)?;
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
        ring_budget_log.as_ref().map(|budget| (budget.used(), budget.peak())),
    );
    let (listener, worker_selection) = if let Some((listener, mut stream, selection)) = bootstrap.take() {
        let range = &selection.ranges()[0];
        let exact = range.first as usize == config.first_layer
            && range.end as usize == config.last_layer.context("selected worker end")? + 1
            && weights.len() == (range.end - range.first) as usize;
        if !exact {
            let error = anyhow::anyhow!("loaded worker layers do not match the solved selection");
            let _ = cuteafd_transport::worker_selection::reject_worker_selection(&mut stream, &error);
            return Err(error);
        }
        cuteafd_transport::worker_selection::acknowledge_worker_selection(&mut stream, config.rank, &selection)?;
        tracing::info!(rank = config.rank, first_layer = config.first_layer, layers = weights.len(),
            selection_digest = %selection.digest()?, "coordinator-selected worker weights ready");
        (listener, Some(selection))
    } else { (TcpListener::bind(listen)?, None) };
    listener.set_nonblocking(true)?;
    let (admit, incoming) = mpsc::sync_channel(2);
    let stop = Arc::new(AtomicBool::new(false));
    let _guard = Admission { stop: stop.clone() };
    let max_frame_bytes = config.max_frame_bytes;
    // Resolved once here: the admission thread and the poll loop must not read
    // the process environment per connection or per poll.
    let protocol_v2_timing = cuteafd_transport::protocol_v2_timing_from_env();
    let worker_rank = config.rank;
    thread::Builder::new()
        .name("v41-roce-bootstrap".into())
        .spawn(move || {
            while !stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let admitted = match &ring_budget {
                            Some(budget) => LocalVerbsExpertConnection::accept_with_selection(
                                stream,
                                max_frame_bytes,
                                protocol_v2_timing,
                                Arc::clone(budget),
                                worker_selection.as_ref().map(|selection| (worker_rank, selection)),
                            ),
                            None => LocalVerbsExpertConnection::accept(
                                stream,
                                max_frame_bytes,
                                protocol_v2_timing,
                            ),
                        };
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
    let executor_id = match config.topology {
        Some(topology) => topology.executor_id(config.rank)?,
        None => cuteafd_transport::expert::v41_spark_executor_id(config.world, config.rank)?,
    };
    let mut connections = Vec::<LocalVerbsExpertConnection>::with_capacity(16);
    // Structured startup evidence: one line per rank naming the native role and
    // logical intermediate this worker actually loaded. A captured log can then
    // be hashed and checked against the declared topology, instead of trusting a
    // hand-written value; the role comes from the same selection the loader used.
    let loaded = config.selection(config.first_layer)?;
    tracing::info!(
        rank = config.rank,
        world = config.world,
        role = loaded.role(),
        intermediate = weights.intermediate(),
        capacity = config.capacity,
        first_layer = config.first_layer,
        layers = weights.len(),
        "native local RoCE expert worker ready"
    );
    // Requests arrive back-to-back while serving, so the loop spins: it is the
    // wakeup path and this is the GPU owner thread. A quiet connection switches
    // to the endpoint's QP completion event wait (`ibv_req_notify_cq` +
    // completion-channel poll, which still busy-polls briefly) with an
    // `idle_wait` upper bound. One wait covers one endpoint, so while idle the
    // loop blocks on one connection per pass and rotates; the pass itself still
    // sweeps every connection with a non-blocking poll, which bounds pickup to
    // one wait window. Any request or admission resets the idle timer.
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
                ring_budget_log.as_ref().map(|budget| (budget.used(), budget.peak())),
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
                    ring_budget_log.as_ref().map(|budget| (budget.used(), budget.peak())),
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
            let wait = (wait_index == Some(index)).then_some(idle_wait);
            let result = connections[index].poll(wait, |view, mapped, emit| {
                // A topology-bound worker admits only the ownership-aware
                // request contract; every other family is a protocol mismatch,
                // not a silent fallback.
                let request = match config.topology {
                    Some(topology) => BackboneRequest::parse_native_group(
                        view.frame_bytes(),
                        config.capacity,
                        topology,
                    )?,
                    None if execution.is_paired() =>
                        BackboneRequest::parse_paired(view.frame_bytes(), config.capacity)?,
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
                let layer = (request.layer() as usize).checked_sub(config.first_layer)
                    .context("requested expert layer is not resident on this Spark")?;
                execution.bind_layer(&weights, layer)?;
                if let Some(slot) = mapped.response_slot {
                    // The mapped frame's hidden rows are device-visible, so the
                    // worker copies them on its stream instead of uploading.
                    let response = unsafe { execution.execute_mapped_request(&request,
                        executor_id, &mut exchange, slot, Some(mapped.hidden_payload)) };
                    let response = match response {
                        Ok(response) => response,
                        Err(error) => { execution_failed = true; return Err(error); }
                    };
                    if let Some(response) = response {
                        return emit(ProtocolV2ExecutorResponseRef::Device(response));
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
                result
            });
            match result {
                Ok(processed) => {
                    progressed |= processed;
                    if processed && pending_steady_log {
                        log_spark_memory_if_enabled(
                            &library,
                            &config,
                            "steady state sample after first post-admission request",
                            Some(connections.len()),
                            ring_budget_log.as_ref().map(|budget| (budget.used(), budget.peak())),
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
