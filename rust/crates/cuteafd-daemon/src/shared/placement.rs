//! Shared startup admission protocols.
pub(crate) mod layers;

use anyhow::{ensure, Context, Result};
use cuteafd_core::{ExpertGeometry, ModelAdmissionResponse, SparkRings};
use cuteafd_loader::placement::{SparkAssignment, SparkResidentLayout, SparkWorkingSet};
use cuteafd_transport::{expert::SparkTopology, worker_selection::{ModelAdmission, WorkerIdentity, WorkerSelection}};
use std::{net::SocketAddr, path::Path, time::Duration};

const ADMISSION_TIMEOUT_ENV: &str = "CUTEAFD_SPARK_ADMISSION_TIMEOUT_SECONDS";

fn admission_timeout(value: Option<&str>) -> Result<Duration> {
    let seconds = value.unwrap_or("1200").parse::<u64>()
        .with_context(|| format!("{ADMISSION_TIMEOUT_ENV} must be positive integer seconds"))?;
    ensure!(seconds > 0 && std::time::Instant::now().checked_add(Duration::from_secs(seconds)).is_some(),
        "{ADMISSION_TIMEOUT_ENV} must be positive, representable seconds");
    Ok(Duration::from_secs(seconds))
}

/// Header-derived expectations exist before workers load or coordinator owners allocate.
pub(crate) struct SparkAdmission {
    selection: WorkerSelection,
    assignments: Vec<SparkAssignment>,
    push_snapshot: Option<std::path::PathBuf>,
    push_manifest: Option<String>,
}

impl SparkAdmission {
    pub(crate) fn plan(snapshot: &Path, geometry: ExpertGeometry,
        catalog: Option<&cuteafd_loader::OfficialV41Catalog>, working: &SparkWorkingSet,
        world: usize, topology: Option<SparkTopology>,
        model: ModelAdmission) -> Result<Self> {
        Self::plan_manifest(snapshot, geometry, catalog, working, world, topology, model, None)
    }

    pub(crate) fn plan_manifest(snapshot: &Path, geometry: ExpertGeometry,
        catalog: Option<&cuteafd_loader::OfficialV41Catalog>, working: &SparkWorkingSet,
        world: usize, topology: Option<SparkTopology>, model: ModelAdmission,
        manifest: Option<&str>) -> Result<Self> {
        ensure!(working.spark_layers.is_empty() || working.spark_ranks == world,
            "Spark admission working-set world differs from launched peers");
        let identity = WorkerIdentity::from_snapshot_manifest(snapshot, geometry,
            crate::shared::spark_topology::world(topology, world)?, manifest)?;
        let selection = WorkerSelection::new(identity, &working.spark_layers)?.with_model(model.clone())?;
        // Idle workers still acknowledge the identity, but need no storage package.
        let assignments = if working.spark_layers.is_empty() {
            (0..world).map(|rank| Ok(SparkAssignment { rank: u32::try_from(rank)?, resident_bytes: 0,
                slices: vec![], io: Default::default() })).collect::<Result<Vec<_>>>()?
        } else if let Some(manifest) = manifest {
            use cuteafd_loader::plan::exlr8::residency::{ExpertSet, Residency};
            let catalog = catalog.context("EXLR8 Spark admission needs native catalog")?;
            ensure!(catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8
                && catalog.routed_experts().geometry()? == geometry,
                "EXLR8 Spark admission catalog differs from coordinator");
            crate::shared::spark_topology::require_world(
                crate::shared::spark_topology::world(topology, world)?, catalog, &working.spark_layers)?;
            ensure!(model.endpoints.iter().all(|endpoint|
                endpoint.ingress == cuteafd_core::SparkIngress::Fp8K32),
                "EXLR8 Spark admission requires FP8 K32 endpoints");
            (0..world).map(|rank| {
                let mut assignment = SparkAssignment { rank: u32::try_from(rank)?, resident_bytes: 0,
                    slices: vec![], io: working.rank_endpoints_io(rank, geometry, &model.endpoints, model.rings)? };
                for &layer in &working.spark_layers {
                    ensure!(layer >= catalog.routed_experts().first_layer && layer < geometry.layers as usize,
                        "EXLR8 selected layer outside native backbone");
                    let layout = Residency::plan(snapshot, manifest, layer, world, rank, ExpertSet::Routed)?;
                    assignment.resident_bytes = assignment.resident_bytes.checked_add(layout.resident_bytes as u64)
                        .context("EXLR8 Spark assignment resident overflow")?;
                    assignment.slices.push(layout.spark_slice()?);
                }
                Ok(assignment)
            }).collect::<Result<Vec<_>>>()?
        } else {
            let catalog = catalog.context("Spark admission needs routed checkpoint headers")?;
            ensure!(catalog.routed_experts().geometry()? == geometry,
                "Spark admission catalog geometry differs from coordinator");
            crate::shared::spark_topology::require_supported(topology, catalog)?;
            let tp = topology.map_or(world, |topology| topology.tp() as usize);
            let headers = cuteafd_loader::plan::files::open_role(snapshot,
                cuteafd_loader::plan::files::ReadRole::Spark { rank: 0, world: tp })?;
            let recipe = cuteafd_loader::plan::experts::checkpoint_spark_recipe(&headers, tp, &working.spark_layers)?;
            (0..world).map(|rank| {
                let shard = topology.map(|topology| topology.tp_rank(rank)).transpose()?
                    .map_or(rank, usize::from);
                let layout = if catalog.exl3().is_some() {
                    SparkResidentLayout::Exl3 { world: tp, rank: shard }
                } else if catalog.fp8().is_some() {
                    SparkResidentLayout::Fp8 { world: tp, rank: shard, slicing: recipe.slicing }
                } else {
                    SparkResidentLayout::Native { world: tp, rank: shard,
                        stored_width: *recipe.widths.get(shard).context("Spark recipe has no rank width")? }
                };
                SparkAssignment::from_catalog(catalog, working, rank, layout, &model.endpoints, model.rings)
                    .with_context(|| format!("Spark admission rank {rank} checkpoint assignment"))
            }).collect::<Result<Vec<_>>>()?
        };
        let push_snapshot = if catalog.is_some_and(|c|
            c.format() != cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8 || manifest.is_some())
            && !working.spark_layers.is_empty() && assignments.iter().all(|a| a.resident_bytes > 0)
            && topology.is_none_or(|t| t.tp() as usize == world)
            && cuteafd_loader::push_load::namespace(snapshot).is_ok() {
            Some(snapshot.to_path_buf())
        } else { None };
        let selection = if push_snapshot.is_some() { selection.with_push_weights() } else { selection };
        Ok(Self { selection, assignments, push_snapshot, push_manifest: manifest.map(str::to_owned) })
    }

    fn reconcile(&self, reports: &[ModelAdmissionResponse]) -> Result<()> {
        ensure!(reports.len() == self.assignments.len(), "Spark admission response count differs from launched peers");
        for (assignment, report) in self.assignments.iter().zip(reports) {
            assignment.validate(report)?;
        }
        Ok(())
    }

    pub(crate) fn start(self, peers: &[SocketAddr]) -> Result<PendingSparkAdmission> {
        self.start_with(peers, None)
    }

    /// Whether this admission opens a push group (Spark experts on SparkNest).
    pub(crate) fn pushes(&self) -> bool { self.push_snapshot.is_some() }

    /// Whether coordinator GPUs may join this admission's push group as ranks:
    /// every supported Spark format accepts coordinator entries after its own.
    pub(crate) fn coordinator_pushes(&self) -> bool { self.pushes() }

    /// `coordinator`: in-process coordinator-GPU ranks joining the Spark push
    /// group (loopback streams in rank order, and their plan entries).
    pub(crate) fn start_with(self, peers: &[SocketAddr],
        coordinator: Option<(Vec<std::net::TcpStream>, super::memory::push_exl3::CoordinatorRanks)>)
        -> Result<PendingSparkAdmission> {
        ensure!(coordinator.is_none() || self.pushes(),
            "coordinator push ranks without a Spark push group");
        let value = std::env::var(ADMISSION_TIMEOUT_ENV).ok();
        let timeout = admission_timeout(value.as_deref())?;
        let pending = cuteafd_transport::worker_selection::start_model_admission(
            peers, &self.selection, timeout)
            .with_context(|| format!("Spark worker admission start failed (timeout {} s)", timeout.as_secs()))?;
        let pending = if let Some(snapshot) = self.push_snapshot.clone() {
            let assignments = self.assignments.clone();
            let layers = self.selection.layers();
            let manifest = self.push_manifest.clone();
            let cancel = pending.cancellation_streams()?;
            let (streams, ranks) = match coordinator {
                Some((streams, ranks)) => (streams, Some(ranks)),
                None => (Vec::new(), None),
            };
            AdmissionWait::Push(PushAdmissionDriver { cancel, thread: Some(std::thread::spawn(move || {
                let mut pending = pending;
                pending.push_weights(streams, |offers| {
                    let started = std::time::Instant::now();
                    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
                    tracing::info!(elapsed_ms = started.elapsed().as_millis(),
                        "Spark push admission catalog opened");
                    if let Some(manifest) = manifest.as_deref() {
                        super::memory::push_exlr8::prepare(&snapshot, &catalog, manifest, &layers, &assignments, offers, ranks.as_ref())
                    } else if catalog.exl3().is_some() {
                        super::memory::push_exl3::prepare(&snapshot, &catalog, &layers, &assignments, offers, ranks.as_ref())
                    } else if catalog.fp8().is_some() {
                        super::memory::push_fp8::prepare(&snapshot, &catalog, &assignments, offers, ranks.as_ref())
                    } else {
                        super::memory::push_native::prepare(&snapshot, &catalog, &assignments, offers, ranks.as_ref())
                    }
                })?;
                pending.wait()
            })) })
        } else {
            // Callers add coordinator ranks only when this admission pushes.
            ensure!(coordinator.is_none(), "coordinator push ranks without a Spark push group");
            AdmissionWait::Mount(pending)
        };
        Ok(PendingSparkAdmission { plan: self, pending, timeout })
    }

    pub(crate) fn select(self, peers: &[SocketAddr]) -> Result<()> {
        self.start(peers)?.wait()
    }
}

struct PushAdmissionDriver {
    cancel: Vec<std::net::TcpStream>,
    thread: Option<std::thread::JoinHandle<Result<Vec<ModelAdmissionResponse>>>>,
}
impl PushAdmissionDriver {
    fn wait(mut self) -> Result<Vec<ModelAdmissionResponse>> {
        self.thread.take().expect("owned admission driver").join()
            .map_err(|_| anyhow::anyhow!("Spark push admission driver panicked"))?
    }
}
impl Drop for PushAdmissionDriver {
    fn drop(&mut self) {
        for stream in &self.cancel { let _ = stream.shutdown(std::net::Shutdown::Both); }
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}

enum AdmissionWait {
    Mount(cuteafd_transport::worker_selection::PendingModelAdmission),
    Push(PushAdmissionDriver),
}

pub(crate) struct PendingSparkAdmission {
    plan: SparkAdmission,
    pending: AdmissionWait,
    timeout: Duration,
}

impl PendingSparkAdmission {
    pub(crate) fn wait(self) -> Result<()> {
        let reports = match self.pending {
            AdmissionWait::Mount(pending) => pending.wait(),
            AdmissionWait::Push(driver) => driver.wait(),
        }
            .with_context(|| format!("Spark worker admission failed (timeout {} s)", self.timeout.as_secs()))?;
        self.plan.reconcile(&reports)?;
        for report in reports {
            tracing::info!(rank = report.rank, resident_bytes = report.resident_bytes,
                workspace_bytes = report.runtime.workspace_bytes, ring_bytes = report.mapped_ring_bytes,
                slices = ?report.slices, "Spark checkpoint assignment reconciled before transport connection");
        }
        Ok(())
    }
}

/// Placement and reservations are final; only worker validation is pending.
/// Keep loaded owners local until validation succeeds, before any serving pool or link.
pub(crate) fn load_while_admitting<T>(pending: Option<PendingSparkAdmission>,
    load: impl FnOnce() -> Result<T>) -> Result<T> {
    let started = std::time::Instant::now();
    tracing::info!(spark_pending = pending.is_some(), "loading admitted coordinator target weights");
    let loaded = load()?;
    tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64,
        "coordinator target weights loaded; awaiting Spark admission");
    if let Some(pending) = pending { pending.wait()?; }
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::{RuntimeSample, SparkEndpoint, SparkIngress, SparkRuntimeInventory};

    struct LoadedOwners(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for LoadedOwners {
        fn drop(&mut self) { self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst); }
    }

    fn pending_fixture() -> Result<(PendingSparkAdmission, Vec<std::net::TcpListener>)> {
        let plan = idle_plan()?;
        let listeners = (0..2).map(|_| std::net::TcpListener::bind("127.0.0.1:0"))
            .collect::<std::io::Result<Vec<_>>>()?;
        let peers = listeners.iter().map(|listener| listener.local_addr())
            .collect::<std::io::Result<Vec<_>>>()?;
        let timeout = Duration::from_secs(5);
        let pending = cuteafd_transport::worker_selection::start_model_admission(
            &peers, &plan.selection, timeout)?;
        Ok((PendingSparkAdmission { plan, pending: AdmissionWait::Mount(pending), timeout }, listeners))
    }

    #[test]
    fn coordinator_load_overlaps_slow_admission_and_returns_only_after_reconciliation() -> Result<()> {
        let (pending, listeners) = pending_fixture()?;
        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (loaded_tx, loaded_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let mut releases = Vec::new();
        let mut workers = Vec::new();
        for (rank, listener) in listeners.into_iter().enumerate() {
            let (release, hold) = std::sync::mpsc::channel();
            releases.push(release);
            workers.push(std::thread::spawn(move || -> Result<()> {
                let (mut stream, _) = listener.accept()?;
                let selection = cuteafd_transport::worker_selection::receive_worker_selection(
                    &stream, rank, Duration::from_secs(5))?;
                hold.recv_timeout(Duration::from_secs(5))?;
                cuteafd_transport::worker_selection::acknowledge_model_admission(
                    &mut stream, rank, &selection, Some(&idle_report(rank as u32)))
            }));
        }
        let owned_drops = drops.clone();
        let coordinator = std::thread::spawn(move || {
            let result = load_while_admitting(Some(pending), || {
                loaded_tx.send(())?;
                Ok(LoadedOwners(owned_drops))
            });
            done_tx.send(result).unwrap();
        });
        loaded_rx.recv_timeout(Duration::from_secs(5))?;
        assert!(matches!(done_rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)));
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 0);
        for release in releases { release.send(())?; }
        let owners = done_rx.recv_timeout(Duration::from_secs(5))??;
        for worker in workers { worker.join().unwrap()?; }
        coordinator.join().unwrap();
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 0);
        drop(owners);
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn admission_failure_after_load_drops_coordinator_owners_without_publishing() -> Result<()> {
        for reject in [false, true] {
            let (pending, listeners) = pending_fixture()?;
            let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let mut releases = Vec::new();
            let workers = listeners.into_iter().enumerate().map(|(rank, listener)| {
                let (release, hold) = std::sync::mpsc::channel();
                releases.push(release);
                std::thread::spawn(move || -> Result<()> {
                    let (mut stream, _) = listener.accept()?;
                    let selection = cuteafd_transport::worker_selection::receive_worker_selection(
                        &stream, rank, Duration::from_secs(5))?;
                    hold.recv_timeout(Duration::from_secs(5))?;
                    if reject {
                        cuteafd_transport::worker_selection::reject_worker_selection(
                            &mut stream, &anyhow::anyhow!("fixture Spark load failed"))
                    } else {
                        let mut report = idle_report(rank as u32);
                        report.resident_bytes = 1;
                        cuteafd_transport::worker_selection::acknowledge_model_admission(
                            &mut stream, rank, &selection, Some(&report))
                    }
                })
            }).collect::<Vec<_>>();
            let result = load_while_admitting(Some(pending), || {
                let owners = LoadedOwners(drops.clone());
                for release in releases { release.send(())?; }
                Ok(owners)
            });
            assert!(result.is_err());
            assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
            for worker in workers { worker.join().unwrap()?; }
        }
        Ok(())
    }

    #[test]
    fn coordinator_load_failure_closes_pending_worker_connections() -> Result<()> {
        let (pending, listeners) = pending_fixture()?;
        assert!(load_while_admitting::<()>(Some(pending), || anyhow::bail!("load failed")).is_err());
        for (rank, listener) in listeners.into_iter().enumerate() {
            let (mut stream, _) = listener.accept()?;
            cuteafd_transport::worker_selection::receive_worker_selection(
                &stream, rank, Duration::from_secs(5))?;
            let mut byte = [0];
            assert_eq!(std::io::Read::read(&mut stream, &mut byte)?, 0);
        }
        Ok(())
    }

    #[test]
    fn coordinator_only_load_needs_no_worker_admission() -> Result<()> {
        assert_eq!(load_while_admitting(None, || Ok(42))?, 42);
        assert!(load_while_admitting::<()>(None, || anyhow::bail!("load failed")).is_err());
        Ok(())
    }

    #[test]
    fn admission_timeout_uses_launch_bound_and_rejects_invalid_values() -> Result<()> {
        assert_eq!(admission_timeout(Some("2400"))?, Duration::from_secs(2400));
        assert_eq!(admission_timeout(None)?, Duration::from_secs(1200));
        for value in ["", "0", "-1", "1.5", "invalid", "18446744073709551615"] {
            let error = format!("{:#}", admission_timeout(Some(value)).unwrap_err());
            assert!(error.contains(ADMISSION_TIMEOUT_ENV), "{error}");
        }
        Ok(())
    }

    #[test]
    fn admission_timeout_failure_names_the_configured_bound() -> Result<()> {
        let plan = idle_plan()?;
        let listeners = (0..2).map(|_| std::net::TcpListener::bind("127.0.0.1:0"))
            .collect::<std::io::Result<Vec<_>>>()?;
        let peers = listeners.iter().map(|listener| listener.local_addr())
            .collect::<std::io::Result<Vec<_>>>()?;
        let (release, hold) = std::sync::mpsc::channel::<()>();
        let worker = std::thread::spawn(move || {
            let _streams = listeners.into_iter().map(|listener| listener.accept().unwrap())
                .collect::<Vec<_>>();
            let _ = hold.recv_timeout(Duration::from_secs(5));
        });
        let timeout = Duration::from_secs(1);
        let pending = cuteafd_transport::worker_selection::start_model_admission(
            &peers, &plan.selection, timeout)?;
        let result = PendingSparkAdmission { plan, pending: AdmissionWait::Mount(pending), timeout }.wait();
        let _ = release.send(());
        worker.join().unwrap();
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("Spark worker admission failed (timeout 1 s)"), "{error}");
        Ok(())
    }

    fn idle_plan() -> Result<SparkAdmission> {
        let root = tempfile::tempdir()?;
        // Identity only: idle admission must not try to infer a storage recipe.
        std::fs::write(root.path().join("config.json"), b"{}")?;
        std::fs::write(root.path().join("model.safetensors.index.json"), b"{}")?;
        SparkAdmission::plan(root.path(), ExpertGeometry::GLM5_FLASH, None,
            &SparkWorkingSet::default(), 2, None,
            ModelAdmission { capacity: 2048, compiled_capacity: 4096,
                endpoints: vec![SparkEndpoint { ring_depth: None, capacity: 2048, ingress: SparkIngress::Fp8K32 }],
                rings: SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 }, execution_slots: 1 })
    }

    fn idle_report(rank: u32) -> ModelAdmissionResponse {
        ModelAdmissionResponse { rank, resident_bytes: 0, slices: vec![],
            runtime: SparkRuntimeInventory { detail: Default::default(), sample: RuntimeSample { device: 0,
                total_bytes: 120 << 30, free_bytes: 100 << 30, context_bytes: 1 << 30,
                module_bytes: 0, sms: 48, arch: "sm_121".into(), driver: "test".into() },
                workspace_bytes: 0, pinned_workspace_bytes: 0 },
            mapped_ring_bytes: 0, host_exchange_bytes: 0, row_indices_bytes: 0 }
    }

    fn fixture_model() -> ModelAdmission {
        ModelAdmission { capacity: 80, compiled_capacity: 80,
            endpoints: vec![SparkEndpoint { ring_depth: None, capacity: 80, ingress: SparkIngress::Fp8K32 },
                SparkEndpoint { ring_depth: None, capacity: 16, ingress: SparkIngress::Fp8K32 }],
            rings: SparkRings { max_frame_bytes: 64 << 20, depth: 2, slot_bytes: 1 << 20, alignment: 4096 }, execution_slots: 1 }
    }

    fn check_reports(plan: &SparkAdmission) -> Result<()> {
        let reports = plan.assignments.iter().map(|assignment| {
            let mut report = idle_report(assignment.rank);
            report.resident_bytes = assignment.resident_bytes;
            report.slices = assignment.slices.clone();
            report.runtime.workspace_bytes = assignment.io.intake;
            report.mapped_ring_bytes = assignment.io.mapped_rings;
            report.host_exchange_bytes = assignment.io.host_exchange;
            report.row_indices_bytes = assignment.io.row_indices;
            report
        }).collect::<Vec<_>>();
        plan.reconcile(&reports)?;
        for field in 0..6 {
            let mut changed = reports.clone();
            let report = &mut changed[0];
            match field {
                0 => report.resident_bytes += 1,
                1 => report.slices[0].stored_width += 128,
                2 => report.mapped_ring_bytes += 1,
                3 => report.host_exchange_bytes += 1,
                4 => report.row_indices_bytes += 1,
                _ => report.runtime.workspace_bytes = plan.assignments[0].io.intake - 1,
            }
            assert!(plan.reconcile(&changed).is_err());
        }
        Ok(())
    }

    #[test]
    fn exlr8_spark_plan_reconciles_native_manifest_when_supplied() -> Result<()> {
        let Some(snapshot) = std::env::var_os("CUTEAFD_EXLR8_TEST_SNAPSHOT") else { return Ok(()); };
        let snapshot = Path::new(&snapshot);
        let mut catalog = cuteafd_loader::read_expert_catalog(snapshot)?;
        catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
        let shape = catalog.routed_experts();
        let selected = [shape.first_layer];
        for ranks in 1..=8 {
            crate::shared::spark_topology::require_world(
                cuteafd_transport::expert::SparkWorld::canonical(ranks)?, &catalog, &selected)?;
        }
        assert!(crate::shared::spark_topology::require_world(
            cuteafd_transport::expert::SparkWorld::owned(SparkTopology::NATIVE_TP4_EP1)?, &catalog, &selected).is_err());
        let canonical = cuteafd_transport::expert::SparkWorld::canonical(4)?;
        crate::shared::spark_topology::require_wire(canonical, &catalog)?;
        for layers in [vec![], vec![shape.first_layer, shape.first_layer],
            vec![shape.first_layer + 1, shape.first_layer], vec![0], vec![shape.layers]] {
            let error = crate::shared::spark_topology::require_world(canonical, &catalog, &layers).unwrap_err();
            assert!(error.to_string().contains("hosted layers"), "{layers:?}: {error:#}");
        }
        let geometry = catalog.routed_experts().geometry()?;
        let working = SparkWorkingSet { spark_layers: vec![catalog.routed_experts().first_layer],
            spark_ranks: 4, prefill_lanes: 1, ..Default::default() };
        let mut model = fixture_model();
        model.endpoints.iter_mut().for_each(|endpoint| endpoint.ingress = SparkIngress::Fp8K32);
        let plan = SparkAdmission::plan_manifest(snapshot, geometry, Some(&catalog), &working,
            4, None, model.clone(), Some("up1"))?;
        assert_eq!(plan.selection.layers(), working.spark_layers);
        for assignment in &plan.assignments {
            assert!(assignment.resident_bytes > 0);
            assert_eq!(assignment.slices[0].first, 0);
            assert_eq!(assignment.slices[0].width, 2048);
            assert_eq!(assignment.slices[0].tiers, [2, 3, 4]);
            assert_eq!(assignment.slices[0].projection_counts.iter().flatten().sum::<u32>(), 256 * 16 * 3 / 4);
        }
        check_reports(&plan)?;
        model.endpoints[0].ingress = SparkIngress::Bf16;
        assert!(SparkAdmission::plan_manifest(snapshot, geometry, Some(&catalog), &working,
            4, None, model, Some("up1")).is_err());
        Ok(())
    }

    #[test]
    fn fp8_spark_plan_uses_checkpoint_exact_widths_and_all_endpoints() -> Result<()> {
        use cuteafd_loader::plan::testing::{fp8, mimo_flash_config, write_snapshot};
        let root = tempfile::tempdir()?;
        let config = mimo_flash_config();
        let mut tensors = Vec::new();
        for expert in 0..256 {
            for (name, rows, cols) in [("gate", 2048, 4096), ("up", 2048, 4096), ("down", 4096, 2048)] {
                tensors.extend(fp8(&format!("model.layers.1.mlp.experts.{expert}.{name}_proj"), rows, cols, None));
            }
        }
        write_snapshot(root.path(), &config, &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let geometry = catalog.routed_experts().geometry()?;
        let working = SparkWorkingSet { spark_layers: vec![1], spark_ranks: 6, prefill_lanes: 1,
            ..Default::default() };
        let plan = SparkAdmission::plan(root.path(), geometry, Some(&catalog), &working, 6, None,
            fixture_model())?;
        assert_eq!(plan.selection.layers(), [1]);
        assert_eq!(plan.assignments.iter().map(|rank| rank.slices[0].stored_width).collect::<Vec<_>>(),
            [256, 384, 384, 384, 384, 256]);
        for assignment in &plan.assignments {
            assert!(assignment.resident_bytes > 0 && assignment.io.mapped_rings > 0);
            assert_eq!(assignment.io.mapped_rings, fixture_model().ring_bytes(geometry)? as u64);
        }
        check_reports(&plan)
    }

    /// Marks `root` a SparkNest namespace with every file a hosted shard, so a
    /// populated Spark admission opens a push group.
    fn mark_namespace(root: &Path) -> Result<()> {
        let mark = |path: &Path, name: &std::ffi::CStr, value: &[u8]| -> Result<()> {
            let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
            // SAFETY: both strings are NUL-terminated and live across the call; the
            // value pointer and length describe one live buffer.
            let status = unsafe { libc::setxattr(path.as_ptr(), name.as_ptr(), value.as_ptr().cast(), value.len(), 0) };
            ensure!(status == 0, "setxattr: {}", std::io::Error::last_os_error());
            Ok(())
        };
        mark(root, c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
        for entry in std::fs::read_dir(root)? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "safetensors") { mark(&path, c"user.sparknest.hosts", b"coordinator")?; }
        }
        Ok(())
    }

    /// Every Spark push format accepts coordinator ranks after its Spark prefix.
    #[test]
    fn coordinator_ranks_join_every_spark_push_format() -> Result<()> {
        use cuteafd_loader::plan::testing::{fp8, mimo_flash_config, qwen4_config, qwen4_exl3, write_snapshot,
            write_v41_snapshot};
        let check = |root: &Path, working: SparkWorkingSet, world: usize, geometry: Option<ExpertGeometry>,
            manifest: Option<&str>| -> Result<()> {
            let mut catalog = cuteafd_loader::read_expert_catalog(root)?;
            if manifest.is_some() { catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?; }
            let geometry = match geometry { Some(g) => g, None => catalog.routed_experts().geometry()? };
            let mut model = fixture_model();
            if manifest.is_some() { model.endpoints.iter_mut().for_each(|e| e.ingress = SparkIngress::Fp8K32); }
            let plan = SparkAdmission::plan_manifest(root, geometry, Some(&catalog), &working, world, None, model,
                manifest)?;
            assert!(plan.pushes(), "{root:?}: the Spark ranks push");
            assert!(plan.coordinator_pushes(), "{root:?}: coordinator ranks join the push group");
            Ok(())
        };
        // EXL3: coordinator ranks join.
        let root = tempfile::tempdir()?;
        let (tensors, _) = qwen4_exl3(3, 4);
        write_snapshot(root.path(), &qwen4_config(3), &tensors, None);
        mark_namespace(root.path())?;
        check(root.path(), SparkWorkingSet { spark_layers: vec![0, 2], spark_ranks: 3, prefill_lanes: 1,
            ..Default::default() }, 3, None, None)?;
        // FP8.
        let root = tempfile::tempdir()?;
        let mut tensors = Vec::new();
        for expert in 0..256 {
            for (name, rows, cols) in [("gate", 2048, 4096), ("up", 2048, 4096), ("down", 4096, 2048)] {
                tensors.extend(fp8(&format!("model.layers.1.mlp.experts.{expert}.{name}_proj"), rows, cols, None));
            }
        }
        write_snapshot(root.path(), &mimo_flash_config(), &tensors, None);
        mark_namespace(root.path())?;
        check(root.path(), SparkWorkingSet { spark_layers: vec![1], spark_ranks: 6, prefill_lanes: 1,
            ..Default::default() }, 6, None, None)?;
        // Native (V4.1 MXFP4).
        let root = tempfile::tempdir()?;
        write_v41_snapshot(root.path());
        mark_namespace(root.path())?;
        check(root.path(), SparkWorkingSet { spark_layers: vec![17, 39], spark_ranks: 4, prefill_lanes: 2,
            ..Default::default() }, 4, Some(ExpertGeometry::DEEPSEEK_V41), None)?;
        // EXLR8 (installed snapshot, read-only: its own namespace marks).
        if let Some(snapshot) = std::env::var_os("CUTEAFD_EXLR8_TEST_SNAPSHOT") {
            let snapshot = Path::new(&snapshot);
            let catalog = cuteafd_loader::read_expert_catalog(snapshot)?;
            let first = catalog.routed_experts().first_layer;
            if cuteafd_loader::push_load::namespace(snapshot).is_ok() {
                check(snapshot, SparkWorkingSet { spark_layers: vec![first], spark_ranks: 4, prefill_lanes: 1,
                    ..Default::default() }, 4, None, Some("up1"))?;
            }
        }
        Ok(())
    }

    #[test]
    fn exl3_spark_plan_preserves_sparse_absolute_ids_and_tier_counts() -> Result<()> {
        use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        let root = tempfile::tempdir()?;
        let (tensors, _) = qwen4_exl3(3, 4);
        write_snapshot(root.path(), &qwen4_config(3), &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let working = SparkWorkingSet { spark_layers: vec![0, 2], rtx_layers: vec![1], spark_ranks: 3,
            prefill_lanes: 1 };
        let plan = SparkAdmission::plan(root.path(), catalog.routed_experts().geometry()?, Some(&catalog),
            &working, 3, None, fixture_model())?;
        assert_eq!(plan.selection.layers(), [0, 2]);
        for (rank, assignment) in plan.assignments.iter().enumerate() {
            assert_eq!(assignment.slices.iter().map(|slice| slice.layer).collect::<Vec<_>>(), [0, 2]);
            let expected = catalog.exl3().unwrap().balanced_residency(cuteafd_loader::V41Exl3Layer::Backbone(0), 3, rank)?;
            let other = catalog.exl3().unwrap().balanced_residency(cuteafd_loader::V41Exl3Layer::Backbone(2), 3, rank)?;
            assert_eq!(assignment.resident_bytes,
                (expected.device_arena_layout()?.1 + other.device_arena_layout()?.1) as u64);
            assert_eq!(assignment.slices[0].tiers, expected.tiers.iter().map(|&tier| tier as u32).collect::<Vec<_>>());
            assert!(!assignment.slices[0].projection_counts.is_empty());
        }
        check_reports(&plan)
    }

    #[test]
    fn kimi_tp6_push_plans_match_admitted_exl3_residency() -> Result<()> {
        use crate::shared::{experts::{exl3::Exl3Weights, layer::ExpertLayer}, memory::push_exl3};
        use cuteafd_transport::worker_selection::push::Offer;
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_kimi_snapshot(root.path(), true);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let working = SparkWorkingSet { spark_layers: vec![1], spark_ranks: 6, prefill_lanes: 1,
            ..Default::default() };
        let admission = SparkAdmission::plan(root.path(), catalog.routed_experts().geometry()?,
            Some(&catalog), &working, 6, None, fixture_model())?;
        assert_eq!(admission.selection.layers(), [1]);
        for (rank, assignment) in admission.assignments.iter().enumerate() {
            assert_eq!(assignment.slices[0].width, 512);
            let budget = Exl3Weights::plan(&catalog, ExpertLayer::BackboneExl3Tp { layer: 1, rank, world: 6 })?;
            assert_eq!(assignment.resident_bytes, budget.resident_bytes as u64);
            let plan = push_exl3::RankPlan::new(&catalog, &[1], 6, rank, budget.resident_bytes)?;
            let offer = Offer { rank, host: format!("rank-{rank}"),
                arena_bytes: plan.layout.logical_bytes() as u64, placement: plan.placement.clone() };
            push_exl3::validate_offer(&offer, rank, assignment, &plan.placement, offer.arena_bytes)?;
            let checked = plan.checked(&catalog, root.path(), 6, rank)?;
            assert!(!checked.plan.entries.is_empty());
            assert_eq!(checked.arenas[rank], offer.arena_bytes);
            // Mount staging is not a push limit; service::push tests exercise
            // independent ring admission against the actual allocatable pool.
            let ring = cuteafd_ffi::push_ring::DEFAULT_SLOTS * cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES;
            assert_eq!(ring, 134_217_728);
            assert_eq!(budget.pinned_host_bytes, 44_826_624);
            assert_eq!(budget.read_scratch_bytes, 268_435_456);
            assert!(ring > budget.pinned_host_bytes, "Kimi push must admit its ring independently");
            assert!(plan.validation.bytes() <= budget.read_scratch_bytes);
        }
        check_reports(&admission)?;
        use cuteafd_loader::plan::testing::{exl3, glm5_config, write_snapshot};
        let glm = tempfile::tempdir()?;
        let tensors = (0..256).flat_map(|expert| [
            ("gate_proj", 2048, 6144), ("up_proj", 2048, 6144), ("down_proj", 6144, 2048),
        ].into_iter().flat_map(move |(projection, n, k)|
            exl3(&format!("model.layers.1.mlp.experts.{expert}.{projection}"), n, k, 4))).collect::<Vec<_>>();
        let mut config = glm5_config();
        config["quantization_config"] = serde_json::json!({"quant_method": "exl3", "bits": 4.0,
            "checkpoint_format": "exl3", "format": "exl3"});
        write_snapshot(glm.path(), &config, &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(glm.path())?;
        let budget = Exl3Weights::plan(&catalog, ExpertLayer::BackboneExl3Tp { layer: 1, rank: 0, world: 6 })?;
        assert_eq!(budget.pinned_host_bytes, 114_499_584);
        assert_eq!(budget.read_scratch_bytes, 268_435_456);
        assert!(cuteafd_ffi::push_ring::DEFAULT_SLOTS * cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES
            > budget.pinned_host_bytes, "GLM K4 push must admit its ring independently");
        Ok(())
    }

    #[test]
    fn native_spark_plan_maps_physical_ranks_to_replicated_tp_shards() -> Result<()> {
        use cuteafd_loader::plan::testing::write_v4_snapshot;
        let root = tempfile::tempdir()?;
        write_v4_snapshot(root.path());
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let geometry = catalog.routed_experts().geometry()?;
        let working = SparkWorkingSet { spark_layers: vec![1, 3], spark_ranks: 6,
            prefill_lanes: 1, ..Default::default() };
        let plan = SparkAdmission::plan(root.path(), geometry, Some(&catalog), &working, 6,
            Some(SparkTopology::new(2, 3)?), fixture_model())?;
        for rank in 0..6 {
            let assignment = &plan.assignments[rank];
            assert_eq!(assignment.rank, rank as u32);
            assert_eq!(assignment.slices[0].first, (rank % 2) as u32 * 1024);
            assert_eq!(assignment.slices[0].stored_width, 1024);
            assert_eq!(assignment.slices, plan.assignments[rank % 2].slices);
        }
        check_reports(&plan)
    }

    #[test]
    fn v41_native_and_modelopt_headers_reconcile_all_worker_owners() -> Result<()> {
        use cuteafd_loader::plan::testing::{write_v41_snapshot, write_v41_nvfp4_snapshot};
        for nvfp4 in [false, true] {
            let root = tempfile::tempdir()?;
            if nvfp4 { write_v41_nvfp4_snapshot(root.path()); } else { write_v41_snapshot(root.path()); }
            let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
            let working = SparkWorkingSet { spark_layers: vec![17, 39], spark_ranks: 4,
                prefill_lanes: 2, ..Default::default() };
            let mut model = fixture_model();
            if nvfp4 { model.endpoints.iter_mut().for_each(|e| e.ingress = SparkIngress::Bf16); }
            let plan = SparkAdmission::plan(root.path(), ExpertGeometry::DEEPSEEK_V41, Some(&catalog),
                &working, 4, None, model)?;
            assert_eq!(plan.selection.layers(), [17, 39]);
            assert!(plan.assignments.iter().all(|a| a.resident_bytes > 0 && a.slices.len() == 2));
            check_reports(&plan)?;
        }
        Ok(())
    }

    #[test]
    fn v41_exl3_tp4_reconciles_checkpoint_layout() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_v41_exl3_snapshot(root.path(), 4);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let working = SparkWorkingSet { spark_layers: vec![17, 39], spark_ranks: 4,
            prefill_lanes: 2, ..Default::default() };
        let plan = SparkAdmission::plan(root.path(), ExpertGeometry::DEEPSEEK_V41, Some(&catalog),
            &working, 4, None, fixture_model())?;
        for (rank, assignment) in plan.assignments.iter().enumerate() {
            let expected = [17, 39].into_iter().map(|layer| -> Result<u64> {
                catalog.exl3().unwrap().balanced_residency(
                    cuteafd_loader::V41Exl3Layer::Backbone(layer), 4, rank).and_then(|plan| Ok(plan.device_arena_layout()?.1 as u64))
            }).collect::<Result<Vec<_>>>()?;
            assert_eq!(assignment.resident_bytes, expected.iter().sum::<u64>());
        }
        check_reports(&plan)
    }

    #[test]
    fn empty_spark_admission_needs_no_storage_recipe_and_reconciles_every_rank() -> Result<()> {
        let plan = idle_plan()?;
        assert!(plan.selection.layers().is_empty());
        assert_eq!(plan.assignments.len(), 2);
        let reports = [idle_report(0), idle_report(1)];
        plan.reconcile(&reports)?;
        assert!(plan.reconcile(&reports[..1]).is_err());
        let mut reports = reports;
        reports.swap(0, 1);
        assert!(plan.reconcile(&reports).is_err());
        Ok(())
    }

    #[test]
    fn spark_reconciliation_refuses_each_mismatched_owner_field() -> Result<()> {
        let plan = idle_plan()?;
        for field in 0..6 {
            let mut reports = [idle_report(0), idle_report(1)];
            match field {
                0 => reports[1].resident_bytes = 1,
                1 => reports[1].mapped_ring_bytes = 1,
                2 => reports[1].host_exchange_bytes = 1,
                3 => reports[1].row_indices_bytes = 1,
                4 => reports[1].runtime.sample.arch.clear(),
                _ => reports[1].slices.push(cuteafd_core::SparkSlice { unit_map: None, layer: 13,
                    first: 0, width: 128, stored_width: 128, tiers: vec![], projection_counts: vec![] }),
            }
            let error = format!("{:#}", plan.reconcile(&reports).unwrap_err());
            assert!(error.contains("Spark rank 1"), "{error}");
        }
        Ok(())
    }
}

pub(crate) fn spark_rings(max_frame_bytes: usize) -> Result<SparkRings> {
    let configured = |name: &str, default| -> Result<usize> {
        match std::env::var(name) {
            Ok(value) => value.parse().with_context(|| format!("invalid {name}")),
            Err(std::env::VarError::NotPresent) => Ok(default),
            Err(error) => Err(error).with_context(|| format!("invalid {name}")),
        }
    };
    Ok(SparkRings { max_frame_bytes,
        depth: configured("CUTEAFD_VERBS_HOST_RING_DEPTH", 8)?,
        slot_bytes: configured("CUTEAFD_VERBS_HOST_RING_SLOT_BYTES", 8 << 20)?,
        alignment: cuteafd_transport::verbs_host_capabilities().preferred_alignment })
}
