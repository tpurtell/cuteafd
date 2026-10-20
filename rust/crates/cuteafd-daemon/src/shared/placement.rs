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
        Ok(Self { selection, assignments })
    }

    fn reconcile(&self, reports: &[ModelAdmissionResponse]) -> Result<()> {
        ensure!(reports.len() == self.assignments.len(), "Spark admission response count differs from launched peers");
        for (assignment, report) in self.assignments.iter().zip(reports) {
            assignment.validate(report)?;
        }
        Ok(())
    }

    pub(crate) fn start(self, peers: &[SocketAddr]) -> Result<PendingSparkAdmission> {
        let value = std::env::var(ADMISSION_TIMEOUT_ENV).ok();
        let timeout = admission_timeout(value.as_deref())?;
        let pending = cuteafd_transport::worker_selection::start_model_admission(
            peers, &self.selection, timeout)
            .with_context(|| format!("Spark worker admission start failed (timeout {} s)", timeout.as_secs()))?;
        Ok(PendingSparkAdmission { plan: self, pending, timeout })
    }

    pub(crate) fn select(self, peers: &[SocketAddr]) -> Result<()> {
        self.start(peers)?.wait()
    }
}

pub(crate) struct PendingSparkAdmission {
    plan: SparkAdmission,
    pending: cuteafd_transport::worker_selection::PendingModelAdmission,
    timeout: Duration,
}

impl PendingSparkAdmission {
    pub(crate) fn wait(self) -> Result<()> {
        let reports = self.pending.wait()
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

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::{RuntimeSample, SparkEndpoint, SparkIngress, SparkRuntimeInventory};

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
        let result = PendingSparkAdmission { plan, pending, timeout }.wait();
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
                endpoints: vec![SparkEndpoint { capacity: 2048, ingress: SparkIngress::Fp8K32 }],
                rings: SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 } })
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
            endpoints: vec![SparkEndpoint { capacity: 80, ingress: SparkIngress::Fp8K32 },
                SparkEndpoint { capacity: 16, ingress: SparkIngress::Fp8K32 }],
            rings: SparkRings { max_frame_bytes: 64 << 20, depth: 2, slot_bytes: 1 << 20, alignment: 4096 } }
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
        for ranks in 1..=6 {
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
