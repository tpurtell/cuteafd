//! Planner-owned launch decisions. Live artifact proof and measured placement
//! remain launcher/runtime responsibilities, not offline identity claims.
use super::{encoder::EncoderPlacement, PlanReport};
use crate::placement::Placement;
use anyhow::{ensure, Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Default)]
pub struct DeploymentOptions {
    pub rtx_gpus: usize,
    pub spark_tp: Option<usize>,
    pub spark_ep: Option<usize>,
    pub peers: Vec<String>,
    pub speculator: String,
    pub admission: AdmissionArgs,
}

/// Preserve the requested spelling for the resolved-config fingerprint. Parsed
/// byte values are supplementary; percentages still need the live device total.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AdmissionArgs {
    pub kv_pool_size: Option<String>,
    pub memory_reservation: Option<String>,
    pub pool_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Topology {
    pub tp: usize,
    pub ep: usize,
    pub wire: SparkWire,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SparkWire {
    Canonical,
    NativeOwnership,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SparkRole {
    pub rank: usize,
    pub group: usize,
    pub tp_rank: usize,
    pub peer: Option<String>,
}

/// All Spark owners are selected over the startup TCP protocol after the live solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SparkAdmission {
    CoordinatorAuthoritative,
}

#[derive(Debug, Serialize)]
pub struct ArtifactRequirements {
    pub verify_live_identity: bool,
    pub verify_resolved_config_fingerprint: bool,
    pub spark_tp_role: Option<String>,
    pub spark_tp_manifest: Option<&'static str>,
    pub require_symbols_verified: bool,
    pub verify_native_library_sha256: bool,
}

#[derive(Debug, Serialize)]
pub struct DeploymentSpec<'a> {
    pub version: u32,
    pub family: &'a str,
    pub snapshot: &'a str,
    pub rtx_gpus: usize,
    pub spark_ranks: usize,
    pub prefill_rows: u64,
    pub topology: Topology,
    pub roles: Vec<SparkRole>,
    pub peers: Vec<String>,
    pub speculator: &'a str,
    pub admission: &'a AdmissionArgs,
    pub spark_admission: SparkAdmission,
    pub launch: Option<&'a super::launch::LaunchDescription>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exlr8: Option<&'a super::exlr8::NativeSelection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spark_requirements: Option<&'a super::exlr8::SparkRequirements>,
    pub estimated_placement: Option<&'a Placement>,
    pub memory_layout: &'a cuteafd_core::memory_layout::MemoryLayout,
    pub encoder: Option<&'a EncoderPlacement>,
    pub audio_encoder: Option<&'a EncoderPlacement>,
    pub encoder_plan_hash: &'a str,
    pub artifact_requirements: ArtifactRequirements,
    pub executable: bool,
    /// Includes release-ready deployments and native-admitted, unqualified WIP runtimes.
    pub wip_launchable: bool,
    /// Keep the full unsupported/tensor diagnostics available to consumers.
    pub report: &'a PlanReport,
}

impl<'a> DeploymentSpec<'a> {
    pub fn from_report(report: &'a PlanReport, options: &'a DeploymentOptions) -> Result<Self> {
        Self::build(report, options, None)
    }

    pub fn from_exlr8(native: &'a super::exlr8::NativePlan, options: &'a DeploymentOptions) -> Result<Self> {
        ensure!(options.rtx_gpus == 2 && options.spark_tp.is_none() && options.spark_ep.is_none()
            && options.speculator == "off", "EXLR8 deployment requires two RTX, canonical TP4 and no speculator");
        Self::build(&native.report, options, Some(native))
    }

    fn build(report: &'a PlanReport, options: &'a DeploymentOptions,
        native: Option<&'a super::exlr8::NativePlan>) -> Result<Self> {
        let family = report.family.as_deref().context("deployment requires a recognized family")?;
        let memory_layout = report.memory_layout.as_ref().context("deployment requires a memory layout")?;
        ensure!((1..=2).contains(&options.rtx_gpus), "deployment requires one or two RTX GPUs");
        let ranks = report.spark_ranks;
        ensure!(options.spark_tp.is_some() == options.spark_ep.is_some(),
            "--spark-tp and --spark-ep must be set together");
        let exl3 = report.experts.as_ref().is_some_and(|e| e.package.contains("exl3"));
        let explicit = options.spark_tp.is_some()
            || (family == "deepseek_v41" && if exl3 { ranks == 3 } else { ranks != 0 && ranks != 4 });
        let topology = Topology { tp: options.spark_tp.unwrap_or(ranks),
            ep: options.spark_ep.unwrap_or(1),
            wire: if explicit { SparkWire::NativeOwnership } else { SparkWire::Canonical } };
        if explicit {
            ensure!(ranks > 0 && topology.tp > 0 && topology.tp <= 6 && topology.ep > 0
                && topology.ep <= 3 && topology.tp * topology.ep == ranks,
                "Spark topology {}x{} must match selected physical rank count {ranks} (TP 1..6, EP 1..3)",
                topology.tp, topology.ep);
            ensure!(topology.ep == 1 || matches!((topology.tp, topology.ep), (2, 2) | (3, 2) | (2, 3)),
                "unsupported replicated Spark topology {}x{}; supported: TP2EP2, TP3EP2, TP2EP3",
                topology.tp, topology.ep);
            ensure!(!exl3 || (family == "deepseek_v41" && topology.tp == 3 && topology.ep == 1),
                "explicit Spark topology requires native experts or V4.1 compact EXL3 TP3EP1");
        }
        if ranks > 0 {
            ensure!(!explicit || family == "deepseek_v41",
                "native ownership wire requires V4.1 experts");
            ensure!(!explicit || matches!((topology.tp, topology.ep),
                (2, 1) | (3, 1) | (4, 1) | (6, 1) | (2, 2) | (3, 2) | (2, 3)),
                "unqualified Spark ownership topology {}x{}; supported: TP2EP1, TP3EP1, TP4EP1, TP6EP1, TP2EP2, TP3EP2, TP2EP3",
                topology.tp, topology.ep);
            ensure!(ranks <= 6, "Spark worlds above six ranks have no qualified package");
            let experts = report.experts.as_ref().context("Spark deployment requires an expert package")?;
            let intermediate = report.spec.as_ref().and_then(|spec| spec.moe.as_ref())
                .context("Spark deployment requires routed expert geometry")?.intermediate;
            if native.is_none() { super::experts::spark_recipe(&experts.package, intermediate, topology.tp)?; }
            else { ensure!(family == "glm5" && ranks == 4 && experts.package == "glm:exlr8",
                "EXLR8 deployment requires GLM native TP4"); }
        }
        ensure!(options.peers.is_empty() || options.peers.len() >= ranks,
            "deployment peer inventory does not cover selected Spark ranks");
        for peer in &options.peers {
            let (host, port) = peer.rsplit_once(':').context("deployment peer must be HOST:PORT")?;
            ensure!(!host.is_empty() && !host.chars().any(char::is_whitespace)
                && port.parse::<u16>().is_ok_and(|port| port != 0), "invalid deployment peer {peer:?}");
        }
        let peers = options.peers.iter().take(ranks).cloned().collect::<Vec<_>>();
        let roles = (0..ranks).map(|rank| SparkRole { rank,
            group: rank / topology.tp, tp_rank: rank % topology.tp,
            peer: peers.get(rank).cloned() }).collect();
        let spark_tp_role = (family == "deepseek_v41" && !exl3 && explicit && topology.tp != 4)
            .then(|| format!("tp{}", topology.tp));
        Ok(Self { version: 1, family, snapshot: &report.snapshot,
            rtx_gpus: options.rtx_gpus, spark_ranks: ranks,
            prefill_rows: report.prefill_rows.unwrap_or(0), topology, roles, peers,
            speculator: &options.speculator, admission: &options.admission,
            spark_admission: SparkAdmission::CoordinatorAuthoritative,
            launch: report.launch.as_ref(), exlr8: native.map(|n| &n.selection),
            spark_requirements: native.map(|n| &n.sparks),
            estimated_placement: report.solver_placement.as_ref(), memory_layout,
            encoder: report.encoder.as_ref(), audio_encoder: report.audio_encoder.as_ref(),
            encoder_plan_hash: &report.encoder_plan_hash,
            artifact_requirements: ArtifactRequirements { verify_live_identity: true,
                verify_resolved_config_fingerprint: family == "deepseek_v41",
                require_symbols_verified: spark_tp_role.is_some(),
                verify_native_library_sha256: spark_tp_role.is_some(), spark_tp_role,
                spark_tp_manifest: (family == "deepseek_v41").then_some("V41_EXPERT_TP_AOT.json") },
            executable: report.executable(), wip_launchable: report.wip_launchable(), report })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{layout::LayoutOptions, plan_spark_count, testing, ExpertPlacement, PlanOptions};

    fn fixture(gpus: usize, ranks: usize) -> (tempfile::TempDir, PlanReport) {
        let dir = tempfile::tempdir().unwrap();
        testing::write_v41_snapshot(dir.path());
        let manifest = testing::write_v41_workspaces(dir.path());
        let report = plan_spark_count(dir.path(), &PlanOptions {
            placement: ExpertPlacement::from_spark_ranks(ranks),
            vision: crate::plan::MediaMode::Off, audio: crate::plan::MediaMode::Off,
            layout: Some(LayoutOptions { rtx_bytes: vec![96 << 30; gpus],
                workspace_manifest: Some(manifest), pool_tokens: Some(0),
                prefill_rows: 256, context_tokens: 262_144, ..Default::default() }),
            ..Default::default()
        }, ranks, Some(ranks)).unwrap();
        (dir, report)
    }

    #[test]
    fn deployment_preserves_solver_and_marks_live_boundary_unresolved() {
        for (gpus, ranks) in [(1, 3), (2, 4)] {
            let (_dir, report) = fixture(gpus, ranks);
            assert!(report.executable(), "{}", crate::plan::render(&report));
            let options = DeploymentOptions { rtx_gpus: gpus, speculator: "dspark".into(),
                peers: (0..ranks).map(|rank| format!("worker-{rank}:9200")).collect(),
                admission: AdmissionArgs { pool_tokens: Some(0), ..Default::default() },
                ..Default::default() };
            let deployment = DeploymentSpec::from_report(&report, &options).unwrap();
            assert_eq!(deployment.roles.len(), ranks);
            assert_eq!(deployment.topology, Topology { tp: ranks, ep: 1, wire: if ranks != 4 { SparkWire::NativeOwnership } else { SparkWire::Canonical } });
            assert_eq!(deployment.spark_admission, SparkAdmission::CoordinatorAuthoritative);
            assert!(deployment.estimated_placement.is_some());
            assert_eq!(serde_json::to_value(deployment.estimated_placement).unwrap(),
                serde_json::to_value(&report.solver_placement).unwrap());
            let value = serde_json::to_value(deployment).unwrap();
            assert!(value["topology"].get("explicit").is_none());
            assert_eq!(value["topology"]["wire"], if ranks != 4 { "native_ownership" } else { "canonical" });
            assert_eq!(value["spark_admission"], "coordinator_authoritative");
            assert!(value.get("placement_handoff").is_none());
            assert!(value.get("placement_handshake").is_none());
            assert_eq!(value["admission"]["pool_tokens"], 0);
            assert_eq!(value["artifact_requirements"]["verify_live_identity"], true);
            assert!(value.get("native_library_sha256").is_none());
            assert!(value.get("resolved_config_fingerprint").is_none());
        }
    }

    #[test]
    fn native_deployment_preserves_selection_and_live_worker_boundary() {
        use crate::plan::{exlr8::{NativePlan, NativeSelection, SparkRequirements}, RuntimeStatus, Status};
        let (_dir, mut report) = fixture(2, 4);
        report.family = Some("glm5".into());
        report.experts.as_mut().unwrap().package = "glm:exlr8".into();
        report.runtime = Some(RuntimeStatus::Planned);
        report.native_serving_admitted = true;
        for component in &mut report.components {
            if component.status == Status::Ready { component.status = Status::Planned; }
        }
        let native = NativePlan { report, selection: NativeSelection {
            manifest: "up1".into(), set_id: "checkpoint-set".into(), dense_from: "official-fp8".into() },
            sparks: SparkRequirements { resident_bytes_per_rank: vec![100; 4],
                k1_workspace: "checked_by_worker_live_admission" } };
        let mut options = DeploymentOptions { rtx_gpus: 2, speculator: "off".into(), ..Default::default() };
        let deployment = DeploymentSpec::from_exlr8(&native, &options).unwrap();
        assert!(!deployment.executable && deployment.wip_launchable);
        let value = serde_json::to_value(deployment).unwrap();
        assert_eq!(value["version"], 1);
        assert_eq!(value["exlr8"]["manifest"], "up1");
        assert_eq!(value["exlr8"]["set_id"], "checkpoint-set");
        assert_eq!(value["exlr8"]["dense_from"], "official-fp8");
        assert_eq!(value["spark_requirements"]["resident_bytes_per_rank"], serde_json::json!([100, 100, 100, 100]));
        assert_eq!(value["spark_requirements"]["k1_workspace"], "checked_by_worker_live_admission");
        options.spark_tp = Some(4);
        options.spark_ep = Some(1);
        assert!(DeploymentSpec::from_exlr8(&native, &options).is_err());
        options.spark_tp = None;
        options.spark_ep = None;
        options.speculator = "auto".into();
        assert!(DeploymentSpec::from_exlr8(&native, &options).is_err());
    }

    #[test]
    fn wip_readiness_preserves_release_and_hard_admission_guards() {
        use crate::plan::{ComponentPlan, Rejection, RoleReadiness, RuntimeStatus, Status, spec::Component};
        let (_dir, mut report) = fixture(1, 3);
        assert!(report.executable() && report.wip_launchable());
        report.runtime = Some(RuntimeStatus::Planned);
        for component in &mut report.components {
            if component.status == Status::Ready { component.status = Status::Planned; }
        }
        assert!(!report.executable() && !report.wip_launchable());
        report.native_serving_admitted = true;
        assert!(!report.executable() && report.wip_launchable());
        let planned = report.clone();
        let index = report.components.iter().position(|c| c.status == Status::Planned).unwrap();
        report.components[index].rejected = 1;
        assert!(!report.wip_launchable());
        report = planned.clone();
        report.components[index].status = Status::MissingKernel;
        assert!(!report.wip_launchable());
        for change in 0..8 {
            report = planned.clone();
            match change {
                0 => report.family = None,
                1 => report.config_error = Some("invalid".into()),
                2 => report.missing_shards.push("missing".into()),
                3 => report.unclassified.push("unknown".into()),
                4 => report.role_readiness.push(RoleReadiness { role: "target".into(), required_shards: vec![],
                    unavailable: vec![Rejection { tensor: "missing".into(), reason: "unavailable".into() }] }),
                5 => report.fits = false,
                6 => report.placement_supported = false,
                _ => report.runtime = Some(RuntimeStatus::Serving),
            }
            assert!(!report.wip_launchable(), "hard guard {change}");
        }
        for status in [Status::Unused, Status::Disabled] {
            report = planned.clone();
            report.components[index].status = status;
            report.components[index].rejected = 1;
            assert!(report.wip_launchable());
        }
        report = planned;
        report.components.clear();
        assert!(!report.executable() && report.wip_launchable());
        report.components.push(ComponentPlan { component: Component::Vision, owner: crate::plan::Owner::Rtx,
            tensors: 0, bytes: 0, formats: Default::default(), status: Status::Disabled, rejected: 0, rejections: vec![] });
        assert!(!report.executable() && report.wip_launchable());
        let value = serde_json::to_value(&report).unwrap();
        assert!(value.get("native_serving_admitted").is_none());
    }

    #[test]
    fn compact_exl3_tp3_normalizes_topology_without_native_role_proof() {
        let (_dir, mut report) = fixture(1, 3);
        report.experts.as_mut().unwrap().package = "v41:exl3-k23".into();
        for explicit in [false, true] {
            let options = DeploymentOptions { rtx_gpus: 1,
                spark_tp: explicit.then_some(3), spark_ep: explicit.then_some(1),
                ..Default::default() };
            let deployment = DeploymentSpec::from_report(&report, &options).unwrap();
            assert_eq!(deployment.topology, Topology { tp: 3, ep: 1, wire: SparkWire::NativeOwnership });
            assert_eq!(deployment.roles.iter().map(|r| (r.group, r.tp_rank)).collect::<Vec<_>>(),
                [(0, 0), (0, 1), (0, 2)]);
            assert!(deployment.artifact_requirements.verify_live_identity);
            assert!(deployment.artifact_requirements.spark_tp_role.is_none());
            assert!(!deployment.artifact_requirements.require_symbols_verified);
            assert!(!deployment.artifact_requirements.verify_native_library_sha256);
        }
        for (tp, ep) in [(2, 1), (4, 1), (3, 2)] {
            report.spark_ranks = tp * ep;
            let options = DeploymentOptions { rtx_gpus: 1,
                spark_tp: Some(tp), spark_ep: Some(ep), ..Default::default() };
            assert!(DeploymentSpec::from_report(&report, &options).is_err());
        }
    }

    #[test]
    fn generic_deployment_retains_family_ranges_and_runtime_proof_limits() {
        let (_dir, mut report) = fixture(1, 4);
        for family in ["deepseek_v4", "glm5", "glm5_flash", "mimo_v2", "qwen4"] {
            report.family = Some(family.into());
            report.launch = Some(super::super::launch::LaunchDescription {
                family, model_type: family.into(), first_expert_layer: 3,
                last_expert_layer: Some(45),
            });
            let options = DeploymentOptions { rtx_gpus: 1,
                speculator: "off".into(),
                admission: AdmissionArgs { kv_pool_size: Some("1.50GiB".into()),
                    memory_reservation: Some("7.5%".into()), pool_tokens: Some(0) },
                ..Default::default() };
            let deployment = DeploymentSpec::from_report(&report, &options).unwrap();
            assert_eq!(deployment.launch.unwrap(), report.launch.as_ref().unwrap());
            assert_eq!(deployment.spark_admission, SparkAdmission::CoordinatorAuthoritative);
            assert!(deployment.artifact_requirements.verify_live_identity);
            assert!(!deployment.artifact_requirements.verify_resolved_config_fingerprint);
            assert!(deployment.artifact_requirements.spark_tp_manifest.is_none());
            assert!(deployment.estimated_placement.is_some());
            assert_eq!(deployment.admission.kv_pool_size.as_deref(), Some("1.50GiB"));
            assert_eq!(deployment.admission.memory_reservation.as_deref(), Some("7.5%"));
        }
    }

    #[test]
    fn canonical_deployment_retains_only_qualified_package_worlds() {
        let (_dir, mut report) = fixture(1, 4);
        let options = DeploymentOptions { rtx_gpus: 1, ..Default::default() };
        for (family, package, intermediate, ranks) in [
            ("deepseek_v41", "v41:mxfp4", 2304, 4),
            ("deepseek_v41", "v41:nvfp4", 2304, 4),
            ("qwen4", "qwen4:exl3-k45", 640, 1),
            ("qwen4", "qwen4:exl3-k45", 640, 3),
            ("glm5_flash", "glmf:nvfp4", 2048, 5),
            ("glm5", "glm:nvfp4", 2048, 3),
        ] {
            report.family = Some(family.into());
            report.experts.as_mut().unwrap().package = package.into();
            report.spec.as_mut().unwrap().moe.as_mut().unwrap().intermediate = intermediate;
            report.spark_ranks = ranks;
            let deployment = DeploymentSpec::from_report(&report, &options).unwrap();
            assert_eq!(deployment.topology.wire, SparkWire::Canonical);
            assert_eq!(deployment.roles.len(), ranks);
            assert_eq!(deployment.topology.tp, ranks);
            for unqualified in [7, 8] {
                report.spark_ranks = unqualified;
                assert!(DeploymentSpec::from_report(&report, &options).is_err());
            }
        }
        for package in ["qwen4:nvfp4", "qwen4:fp8", "qwen4:exl3-k45"] {
            report.family = Some("qwen4".into());
            report.experts.as_mut().unwrap().package = package.into();
            report.spark_ranks = 5;
            assert!(DeploymentSpec::from_report(&report, &options).is_err());
        }
        report.family = Some("deepseek_v41".into());
        report.experts.as_mut().unwrap().package = "v41:exl3-k23".into();
        report.spark_ranks = 6;
        assert!(DeploymentSpec::from_report(&report, &options).is_err());
    }

    #[test]
    fn topology_refusals_name_requested_layout_and_physical_count() {
        let (_dir, mut report) = fixture(1, 4);
        let options = DeploymentOptions { rtx_gpus: 1,
            spark_tp: Some(3), spark_ep: Some(1), ..Default::default() };
        let error = DeploymentSpec::from_report(&report, &options).unwrap_err().to_string();
        assert!(error.contains("topology 3x1"), "{error}");
        assert!(error.contains("physical rank count 4"), "{error}");
        report.spark_ranks = 1;
        let options = DeploymentOptions { spark_tp: Some(1), ..options };
        let error = DeploymentSpec::from_report(&report, &options).unwrap_err().to_string();
        assert!(error.contains("ownership topology 1x1"), "{error}");
        assert!(error.contains("supported: TP2EP1, TP3EP1, TP4EP1, TP6EP1"), "{error}");
    }

    #[test]
    fn deployment_validates_replicated_roles_endpoints_and_admission() {
        let (_dir, mut report) = fixture(1, 4);
        let mut options = DeploymentOptions { rtx_gpus: 1,
            spark_tp: Some(2), spark_ep: Some(2),
            peers: (0..4).map(|rank| format!("worker-{rank}:9200")).collect(),
            ..Default::default() };
        let deployment = DeploymentSpec::from_report(&report, &options).unwrap();
        assert_eq!(deployment.roles.iter().map(|r| (r.group, r.tp_rank)).collect::<Vec<_>>(),
            [(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(deployment.spark_admission, SparkAdmission::CoordinatorAuthoritative);
        assert_eq!(deployment.artifact_requirements.spark_tp_role.as_deref(), Some("tp2"));
        options.spark_ep = Some(3);
        assert!(DeploymentSpec::from_report(&report, &options).is_err());
        options.spark_ep = Some(2);
        options.peers.pop();
        assert!(DeploymentSpec::from_report(&report, &options).is_err());
        options.peers.clear();
        options.peers.push("worker:0".into());
        assert!(DeploymentSpec::from_report(&report, &options).is_err());
        options.peers.clear();
        report.experts.as_mut().unwrap().package = "v41:exl3-k23".into();
        assert!(DeploymentSpec::from_report(&report, &options).is_err());
        options.spark_tp = None; options.spark_ep = None;
        assert_eq!(DeploymentSpec::from_report(&report, &options).unwrap().topology.wire,
            SparkWire::Canonical);
        report.memory_layout = None;
        assert!(DeploymentSpec::from_report(&report, &options).is_err());
    }
}
