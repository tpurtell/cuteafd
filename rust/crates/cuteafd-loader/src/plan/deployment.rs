//! Planner-owned launch decisions. Live artifact proof and measured placement
//! remain launcher/runtime responsibilities, not offline identity claims.
use super::{encoder::EncoderPlacement, PlanReport};
use crate::placement::Placement;
use anyhow::{ensure, Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Default)]
pub struct DeploymentOptions {
    pub rtx_gpus: usize,
    pub peers: Vec<String>,
    pub speculator: String,
    pub admission: AdmissionArgs,
}

/// Preserve the requested spelling for the resolved-config fingerprint, plus
/// the one memory policy (caps, reserves, KV target/min/max) the plan used.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AdmissionArgs {
    pub kv_pool_size: Option<String>,
    pub pool_tokens: Option<u64>,
    pub memory_policy: cuteafd_core::memory_policy::MemoryPolicy,
}

/// Every Spark rank holds a TP slice of every expert of its layers and
/// receives the same canonical request: TP is the expert rank count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Topology {
    pub tp: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SparkRole {
    pub rank: usize,
    pub peer: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SparkRankRoles {
    pub rank: usize,
    pub experts: bool,
    pub layers: Vec<u32>,
    pub encoder: Option<EncoderPlacement>,
    pub audio_encoder: Option<EncoderPlacement>,
    pub drafter: Option<EncoderPlacement>,
}

/// All Spark owners are selected over the startup TCP protocol after the live solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SparkAdmission {
    CoordinatorAuthoritative,
}

#[derive(Debug, Serialize)]
pub struct ArtifactRequirements {
    /// Exact package-relative directories required on every expert host.
    pub spark_packages: std::collections::BTreeMap<String, Vec<String>>,
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
    pub spark_expert_world: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_records: Option<&'static str>,
    pub spark_roles: Vec<SparkRankRoles>,
    pub prefill_unit_rows: u64,
    pub prefill_waves: u64,
    pub decode_lanes: u8,
    /// Graphs with kernel nodes on more than one coordinator GPU (see `PlanReport`).
    pub cross_device_graph_kernels: bool,
    pub prefill_span_rows: u64,
    /// The reserve the depth cuts (the standard 8192, or the planner's fit
    /// cap below it); launches pass it back.
    pub prefill_reserve_rows: u64,
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
        ensure!((1..=2).contains(&options.rtx_gpus) && matches!(options.speculator.as_str(), "off" | "dflash2"),
            "EXLR8 deployment requires one or two RTX GPUs and off or dflash2 speculation");
        Self::build(&native.report, options, Some(native))
    }

    fn build(report: &'a PlanReport, options: &'a DeploymentOptions,
        native: Option<&'a super::exlr8::NativePlan>) -> Result<Self> {
        let family = report.family.as_deref().context("deployment requires a recognized family")?;
        let memory_layout = report.memory_layout.as_ref().context("deployment requires a memory layout")?;
        ensure!((1..=2).contains(&options.rtx_gpus), "deployment requires one or two RTX GPUs");
        let layers = report.spark_expert_layers();
        let ranks = report.spark_expert_world();
        let encoder_ranks = report.encoder.as_ref().map_or_else(Vec::new, EncoderPlacement::spark_ranks);
        let audio_ranks = report.audio_encoder.as_ref().map_or_else(Vec::new, EncoderPlacement::spark_ranks);
        let physical = encoder_ranks.iter().chain(&audio_ranks).copied().max().map_or(ranks, |r| ranks.max(r + 1));
        let spark_roles = (0..physical).map(|rank| SparkRankRoles { rank,
            experts: rank < ranks, layers: if rank < ranks { layers.clone() } else { Vec::new() },
            encoder: encoder_ranks.contains(&rank).then(|| report.encoder.clone()).flatten(),
            audio_encoder: audio_ranks.contains(&rank).then(|| report.audio_encoder.clone()).flatten(),
            // Drafter placement is coordinator-only until a solved Spark owner exists.
            drafter: None,
        }).filter(|role| role.experts || role.encoder.is_some() || role.audio_encoder.is_some()
            || role.drafter.is_some()).collect::<Vec<_>>();
        let physical = spark_roles.len();
        let exl3 = report.experts.as_ref().is_some_and(|e| e.package.contains("exl3"));
        let topology = Topology { tp: ranks };
        let mut spark_packages = std::collections::BTreeMap::new();
        if ranks > 0 {
            ensure!(ranks <= 8, "Spark transport supports at most eight ranks");
            let experts = report.experts.as_ref().context("Spark deployment requires an expert package")?;
            let intermediate = report.spec.as_ref().and_then(|spec| spec.moe.as_ref())
                .context("Spark deployment requires routed expert geometry")?.intermediate;
            if native.is_none() {
                for package in experts.package.split(" + ") {
                    let package = package.split_whitespace().next().context("expert package key")?;
                    let recipe = super::experts::spark_recipe(package, intermediate, topology.tp)?;
                    if !recipe.native_artifacts.is_empty() { continue; }
                    let (family, format) = package.split_once(':').context("expert package format")?;
                    let (root, layouts) = if let Some(tiers) = format.strip_prefix("exl3-") {
                        (format!("exl3/exl3-{family}-{tiers}"), recipe.required_layouts.iter()
                            .map(|layout| format!("{layout}/m1")).collect())
                    } else {
                        let suffix = match format { "fp8" | "mxfp4" => String::new(), "bf16" => "-bf16-weights".into(), "nvfp4" => "-nvfp4a4".into(),
                            other => format!("-{other}") };
                        (format!("fp8/fp8-{family}{suffix}"), recipe.required_layouts)
                    };
                    spark_packages.insert(root, layouts);
                }
            }
            else { ensure!((1..=8).contains(&ranks) && matches!((family, experts.package.as_str()),
                ("glm5", "glm:exlr8") | ("glm5_flash", "glmf:exlr8")),
                "EXLR8 deployment requires a matching native family/package and canonical Sparks"); }
        }
        ensure!(options.peers.is_empty() || options.peers.len() >= ranks,
            "deployment peer inventory does not cover selected Spark ranks");
        for peer in &options.peers {
            let (host, port) = peer.rsplit_once(':').context("deployment peer must be HOST:PORT")?;
            ensure!(!host.is_empty() && !host.chars().any(char::is_whitespace)
                && port.parse::<u16>().is_ok_and(|port| port != 0), "invalid deployment peer {peer:?}");
        }
        let peers = options.peers.iter().take(ranks).cloned().collect::<Vec<_>>();
        let roles = (0..ranks).map(|rank| SparkRole { rank, peer: peers.get(rank).cloned() }).collect();
        // Canonical TP4 has no namespaced role; every other native V4.1 TP does.
        let spark_tp_role = (family == "deepseek_v41" && !exl3 && ranks != 0 && ranks != 4)
            .then(|| format!("tp{ranks}"));
        Ok(Self { version: 1, family, snapshot: &report.snapshot,
            rtx_gpus: options.rtx_gpus, spark_ranks: physical, spark_expert_world: ranks,
            replay_records: report.replay_records, spark_roles,
            prefill_unit_rows: report.prefill_unit_rows.unwrap_or(0),
            prefill_waves: report.prefill_waves.unwrap_or(0),
            decode_lanes: report.decode_lanes,
            cross_device_graph_kernels: report.cross_device_graph_kernels,
            prefill_span_rows: report.prefill_span_rows.unwrap_or(0),
            prefill_reserve_rows: report.prefill_reserve_rows
                .unwrap_or(crate::serving_capacity::prefill_waves::RESERVE_ROWS), topology, roles, peers,
            speculator: report.speculator.unwrap_or(&options.speculator), admission: &options.admission,
            spark_admission: SparkAdmission::CoordinatorAuthoritative,
            launch: report.launch.as_ref(), exlr8: native.map(|n| &n.selection),
            spark_requirements: native.map(|n| &n.sparks),
            estimated_placement: report.solver_placement.as_ref(), memory_layout,
            encoder: report.encoder.as_ref(), audio_encoder: report.audio_encoder.as_ref(),
            encoder_plan_hash: &report.encoder_plan_hash,
            artifact_requirements: ArtifactRequirements { spark_packages, verify_live_identity: true,
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

    fn fixture(gpus: usize, ranks: usize) -> (tempfile::TempDir, PlanReport) { fixture_with(gpus, ranks, false) }

    fn fixture_with(gpus: usize, ranks: usize, tp2_dspark_experts: bool) -> (tempfile::TempDir, PlanReport) {
        let dir = tempfile::tempdir().unwrap();
        testing::write_v41_snapshot(dir.path());
        let manifest = testing::write_v41_workspaces(dir.path());
        let report = plan_spark_count(dir.path(), &PlanOptions {
            placement: ExpertPlacement::from_spark_ranks(ranks),
            vision: crate::plan::MediaMode::Off, audio: crate::plan::MediaMode::Off,
            layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; gpus],
                workspace_manifest: Some(manifest), pool_tokens: Some(0), tp2_dspark_experts,
                prefill_rows: 256, context_tokens: 262_144, ..Default::default() }),
            ..Default::default()
        }, ranks, Some(ranks)).unwrap();
        (dir, report)
    }

    /// #2094: nsys node-level graph tracing faults on a graph whose kernels run
    /// on a second GPU. The launcher keys on this declared fact, not a family name.
    #[test]
    fn deployment_declares_graphs_with_kernels_on_both_coordinator_gpus() {
        for (gpus, tp2, expected) in [(1, false, false), (2, false, false), (2, true, true)] {
            let (_dir, report) = fixture_with(gpus, 4, tp2);
            assert_eq!(report.cross_device_graph_kernels, expected, "gpus={gpus} tp2={tp2}: {}", crate::plan::render(&report));
            let options = DeploymentOptions { rtx_gpus: gpus, speculator: "dspark".into(),
                peers: (0..4).map(|rank| format!("worker-{rank}:9200")).collect(),
                admission: AdmissionArgs { pool_tokens: Some(0), ..Default::default() }, ..Default::default() };
            let spec = DeploymentSpec::from_report(&report, &options).unwrap();
            assert_eq!(serde_json::to_value(&spec).unwrap()["cross_device_graph_kernels"], expected);
        }
    }

    #[test]
    fn all_onboarded_experts_keep_only_sparse_media_roles() {
        use super::super::encoder::{EncoderKind, EncoderPlacement};
        let (_dir, mut report) = fixture(2, 4);
        for layer in &mut report.solver_placement.as_mut().unwrap().layers {
            if layer.experts == crate::placement::ExpertHome::Spark {
                layer.experts = crate::placement::ExpertHome::RtxTp2;
            }
        }
        let options = DeploymentOptions { rtx_gpus: 2, peers: vec!["worker:9200".into(); 4], ..Default::default() };
        let local = DeploymentSpec::from_report(&report, &options).unwrap();
        assert_eq!((local.spark_ranks, local.spark_expert_world), (0, 0));
        assert!(local.spark_roles.is_empty() && local.peers.is_empty() && local.roles.is_empty());
        report.encoder = Some(EncoderPlacement { kind: EncoderKind::Spark { rank: 2 },
            weights: 128, scratch: 256, replicas: vec![], reason: "tower only".into(), shortfall: 0 });
        report.audio_encoder = Some(EncoderPlacement { kind: EncoderKind::Spark { rank: 2 },
            weights: 64, scratch: 512, replicas: vec![], reason: "audio only".into(), shortfall: 0 });
        let media = DeploymentSpec::from_report(&report, &options).unwrap();
        assert_eq!((media.spark_ranks, media.spark_expert_world), (1, 0));
        assert_eq!(media.spark_roles.len(), 1);
        assert_eq!(media.spark_roles[0].rank, 2);
        assert!(!media.spark_roles[0].experts && media.spark_roles[0].layers.is_empty());
        assert_eq!(media.spark_roles[0].encoder, report.encoder);
        assert_eq!(media.spark_roles[0].audio_encoder, report.audio_encoder);
        assert!(media.peers.is_empty() && media.roles.is_empty());
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
            for device in deployment.memory_layout.devices.iter()
                .filter(|device| device.kind == cuteafd_core::memory_layout::DeviceKind::Spark) {
                assert_eq!(device.worker_budget_bytes, 108 << 30);
                assert_eq!(device.capacity_bytes, 121 << 30);
            }
            assert_eq!(deployment.topology, Topology { tp: ranks });
            assert_eq!(deployment.artifact_requirements.spark_tp_role.as_deref(),
                (ranks != 4).then(|| format!("tp{ranks}")).as_deref());
            assert_eq!(deployment.spark_admission, SparkAdmission::CoordinatorAuthoritative);
            assert!(deployment.estimated_placement.is_some());
            assert_eq!(serde_json::to_value(deployment.estimated_placement).unwrap(),
                serde_json::to_value(&report.solver_placement).unwrap());
            let value = serde_json::to_value(deployment).unwrap();
            assert_eq!(value["topology"], serde_json::json!({"tp": ranks}));
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
    fn deployment_preserves_independent_unit_wave_span_and_transport_geometry() {
        for (gpus, ranks) in [(1, 4), (2, 6)] {
            let (_dir, mut report) = fixture(gpus, ranks);
            for family in ["deepseek_v41", "deepseek_v4", "glm5", "glm5_flash", "mimo_v2", "qwen4", "kimi_k3", "step5"] {
                report.family = Some(family.into());
                for (unit, waves, span, transports) in [(2048, 2, 16384, 1), (4096, 2, 8192, 3), (4096, 1, 16384, 2)] {
                    report.prefill_unit_rows = Some(unit);
                    report.prefill_waves = Some(waves);
                    report.prefill_span_rows = Some(span);
                    report.prefill_transports = Some(transports);
                    let options = DeploymentOptions { rtx_gpus: gpus, ..Default::default() };
                    let deployment = DeploymentSpec::from_report(&report, &options).unwrap();
                    assert_eq!((deployment.prefill_unit_rows, deployment.prefill_waves, deployment.prefill_span_rows),
                        (unit, waves, span), "{family} {gpus}+{ranks}");
                    let value = serde_json::to_value(deployment).unwrap();
                    assert_eq!(value["report"]["prefill_transports"], transports);
                    for legacy in ["prefill_rows", "prefill_lanes", "prefill_slots"] {
                        assert!(value.get(legacy).is_none());
                        assert!(value["report"].get(legacy).is_none());
                    }
                }
            }
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
        report.exlr8_prefill_activations = Some(crate::plan::exlr8::PrefillActivations::A8);
        let native = NativePlan { report, selection: NativeSelection {
            manifest: "up1".into(), set_id: "checkpoint-set".into() },
            sparks: SparkRequirements { resident_bytes_per_rank: vec![100; 4],
                k1_workspace: "checked_by_worker_live_admission" } };
        let mut options = DeploymentOptions { rtx_gpus: 2, speculator: "off".into(), ..Default::default() };
        for gpus in 0..=3 {
            options.rtx_gpus = gpus;
            if !(1..=2).contains(&gpus) {
                assert!(DeploymentSpec::from_exlr8(&native, &options).is_err());
                continue;
            }
            let deployment = DeploymentSpec::from_exlr8(&native, &options).unwrap();
            assert!(!deployment.executable && deployment.wip_launchable);
            let value = serde_json::to_value(deployment).unwrap();
            assert_eq!(value["version"], 1);
            assert_eq!(value["exlr8"]["manifest"], "up1");
            assert_eq!(value["exlr8"]["set_id"], "checkpoint-set");
            assert_eq!(value["report"]["exlr8_prefill_activations"], "a8");
            assert_eq!(value["spark_requirements"]["resident_bytes_per_rank"], serde_json::json!([100, 100, 100, 100]));
            assert_eq!(value["spark_requirements"]["k1_workspace"], "checked_by_worker_live_admission");
            assert_eq!(value["rtx_gpus"], gpus);
        }
        options.rtx_gpus = 1;
        options.speculator = "dflash2".into();
        assert!(DeploymentSpec::from_exlr8(&native, &options).is_ok());
        options.speculator = "auto".into();
        assert!(DeploymentSpec::from_exlr8(&native, &options).is_err());
    }

    #[test]
    fn native_flash_deployment_admits_only_declared_coordinator_worlds() {
        use crate::plan::exlr8::{NativePlan, NativeSelection, SparkRequirements};
        for gpus in [1, 2] {
            let (_dir, mut report) = fixture(gpus, 4);
            report.family = Some("glm5_flash".into());
            report.experts.as_mut().unwrap().package = "glmf:exlr8".into();
            use crate::plan::family::Family;
            let execution = crate::plan::families::glm::GLM5_FLASH.exlr8_execution().check(gpus, true);
            let admitted = execution.is_ok();
            report.runtime = Some(if admitted { crate::plan::RuntimeStatus::Serving }
                else { crate::plan::RuntimeStatus::Planned });
            report.native_serving_admitted = admitted;
            report.placement_supported = admitted;
            if let Err(error) = execution { report.hints.push(error.hint()); }
            let mut native = NativePlan { report,
                selection: NativeSelection { manifest: "k4".into(), set_id: "flash-set".into() },
                sparks: SparkRequirements { resident_bytes_per_rank: vec![100; 4],
                    k1_workspace: "checked_by_worker_live_admission" } };
            let options = DeploymentOptions { rtx_gpus: gpus, speculator: "off".into(), ..Default::default() };
            let deployment = DeploymentSpec::from_exlr8(&native, &options).unwrap();
            assert!(deployment.wip_launchable);
            assert_eq!(deployment.exlr8.unwrap().manifest, "k4");
            assert_eq!(deployment.memory_layout, native.report.memory_layout.as_ref().unwrap());
            native.report.experts.as_mut().unwrap().package = "glm:exlr8".into();
            assert!(DeploymentSpec::from_exlr8(&native, &options).is_err());
        }
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
                admission: AdmissionArgs { kv_pool_size: Some("1.50GiB".into()), pool_tokens: Some(0),
                    memory_policy: cuteafd_core::memory_policy::MemoryPolicy { gpu_reserve: Some(1 << 30), ..Default::default() } },
                ..Default::default() };
            let deployment = DeploymentSpec::from_report(&report, &options).unwrap();
            assert_eq!(deployment.launch.unwrap(), report.launch.as_ref().unwrap());
            assert_eq!(deployment.spark_admission, SparkAdmission::CoordinatorAuthoritative);
            assert!(deployment.artifact_requirements.verify_live_identity);
            assert!(!deployment.artifact_requirements.verify_resolved_config_fingerprint);
            assert!(deployment.artifact_requirements.spark_tp_manifest.is_none());
            assert!(deployment.estimated_placement.is_some());
            assert_eq!(deployment.admission.kv_pool_size.as_deref(), Some("1.50GiB"));
            assert_eq!(deployment.admission.memory_policy.gpu_reserve, Some(1 << 30));
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
            assert_eq!(deployment.roles.len(), ranks);
            assert_eq!(deployment.topology.tp, ranks);
            for unqualified in [7, 8] {
                report.spark_ranks = unqualified;
                let result = DeploymentSpec::from_report(&report, &options);
                if package == "v41:mxfp4" {
                    assert!(result.is_ok());
                } else {
                    assert!(result.is_err());
                    if package.ends_with(":nvfp4") {
                        assert!(format!("{:#}", result.unwrap_err()).contains("NVFP4 is deprecated"));
                    }
                }
            }
        }
        for package in ["qwen4:nvfp4", "qwen4:fp8", "qwen4:exl3-k45"] {
            report.family = Some("qwen4".into());
            report.experts.as_mut().unwrap().package = package.into();
            report.spec.as_mut().unwrap().moe.as_mut().unwrap().intermediate = 640;
            report.spark_ranks = 5;
            let result = DeploymentSpec::from_report(&report, &options);
            if package == "qwen4:fp8" {
                assert!(result.is_ok());
            } else {
                assert!(result.is_err());
            }
        }
        report.family = Some("deepseek_v41".into());
        report.experts.as_mut().unwrap().package = "v41:exl3-k23".into();
        report.spark_ranks = 6;
        assert!(DeploymentSpec::from_report(&report, &options).is_err());
    }

    /// V4.1 plans and deploys on the canonical wire at every world its MXFP4
    /// package range names (2..=8, 7 and 8 CPU-gated only) from its planner
    /// minimum up, and refuses below it. The full checkpoint's routed experts
    /// need three Sparks, so two is refused on fit, not on wire. TP4 keeps the
    /// legacy shard; every other world names its native Spark TP role.
    #[test]
    fn v41_plans_and_deploys_every_registry_world_on_the_canonical_wire() {
        let worlds = cuteafd_core::tp_units::spark_package_worlds("v41", "mxfp4");
        assert_eq!(worlds, [2, 3, 4, 5, 6, 7, 8]);
        let (_dir, below) = fixture(2, 2);
        assert!(!below.executable());
        assert!(below.hints.iter().any(|hint| hint.what.contains("Spark count 2 refused")
            || hint.how.contains("Spark count 2 refused")), "{:?}", below.hints);
        for ranks in 3..=cuteafd_core::MAX_SPARK_WORLD {
            let (_dir, report) = fixture(2, ranks);
            assert!(report.executable(), "TP{ranks}: {}", crate::plan::render(&report));
            assert_eq!(report.spark_expert_world(), ranks, "TP{ranks}");
            let options = DeploymentOptions { rtx_gpus: 2, speculator: "off".into(),
                peers: (0..ranks).map(|rank| format!("worker-{rank}:9200")).collect(),
                ..Default::default() };
            let deployment = DeploymentSpec::from_report(&report, &options).unwrap();
            assert!(deployment.executable, "TP{ranks}");
            assert_eq!(deployment.topology, Topology { tp: ranks });
            assert_eq!(deployment.roles.iter().map(|role| role.rank).collect::<Vec<_>>(),
                (0..ranks).collect::<Vec<_>>());
            assert_eq!(deployment.artifact_requirements.spark_tp_role.as_deref(),
                (ranks != 4).then(|| format!("tp{ranks}")).as_deref(), "TP{ranks}");
            // Native V4.1 slices are attested by the TP role manifest, not a
            // generic package root, so the package preflight names none.
            assert!(deployment.artifact_requirements.spark_packages.is_empty(), "TP{ranks}");
            assert_eq!(serde_json::to_value(&deployment).unwrap()["artifact_requirements"]["spark_tp_role"],
                serde_json::json!((ranks != 4).then(|| format!("tp{ranks}"))), "TP{ranks}");
        }
        let (_dir, mut report) = fixture(1, 4);
        report.spark_ranks = 1;
        let options = DeploymentOptions { rtx_gpus: 1, ..Default::default() };
        let error = DeploymentSpec::from_report(&report, &options).unwrap_err().to_string();
        assert!(error.contains("TP1"), "{error}");
    }

    #[test]
    fn ml4_deployment_names_every_compact_or_balanced_rank_package() {
        let (_dir, mut report) = fixture(1, 4);
        report.family = Some("mistral_large4".into());
        report.experts.as_mut().unwrap().package = "ml4:exl3-k45".into();
        report.spec.as_mut().unwrap().moe.as_mut().unwrap().intermediate = 6144;
        let options = DeploymentOptions { rtx_gpus: 1, ..Default::default() };
        for ranks in [5, 6] {
            report.spark_ranks = ranks;
            let spec = DeploymentSpec::from_report(&report, &options).unwrap();
            let compact = if ranks == 5 { "-compact" } else { "" };
            assert_eq!(spec.artifact_requirements.spark_packages["exl3/exl3-ml4-k45"],
                (0..ranks).map(|rank| format!("tp{ranks}{compact}-rank{rank}/m1")).collect::<Vec<_>>());
        }
    }

    #[test]
    fn decorated_packages_keep_canonical_artifact_roots() {
        let (_dir, mut report) = fixture(1, 4);
        report.family = Some("mimo_v2".into());
        report.spark_ranks = 6;
        report.experts.as_mut().unwrap().package = "mimop:fp8 (MXFP4)".into();
        report.spec.as_mut().unwrap().moe.as_mut().unwrap().intermediate = 2048;
        let options = DeploymentOptions { rtx_gpus: 1, ..Default::default() };
        let spec = DeploymentSpec::from_report(&report, &options).unwrap();
        assert_eq!(spec.artifact_requirements.spark_packages.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["fp8/fp8-mimop"]);
    }

    #[test]
    fn deployment_refuses_an_unpackaged_world() {
        let (_dir, mut report) = fixture(1, 4);
        report.spark_ranks = 1;
        let options = DeploymentOptions { rtx_gpus: 1, ..Default::default() };
        let error = DeploymentSpec::from_report(&report, &options).unwrap_err().to_string();
        assert!(error.contains("missing Spark package recipe"), "{error}");
        assert!(error.contains("TP1"), "{error}");
    }

    #[test]
    fn deployment_validates_roles_endpoints_and_admission() {
        let (_dir, mut report) = fixture(1, 4);
        let mut options = DeploymentOptions { rtx_gpus: 1,
            peers: (0..4).map(|rank| format!("worker-{rank}:9200")).collect(),
            ..Default::default() };
        let deployment = DeploymentSpec::from_report(&report, &options).unwrap();
        assert_eq!(deployment.roles.iter().map(|r| r.rank).collect::<Vec<_>>(), [0, 1, 2, 3]);
        assert_eq!(deployment.spark_admission, SparkAdmission::CoordinatorAuthoritative);
        assert!(deployment.artifact_requirements.spark_tp_role.is_none());
        options.peers.pop();
        assert!(DeploymentSpec::from_report(&report, &options).is_err());
        options.peers.clear();
        options.peers.push("worker:0".into());
        assert!(DeploymentSpec::from_report(&report, &options).is_err());
        options.peers.clear();
        report.experts.as_mut().unwrap().package = "dsv4p:exl3-k23 (expertd-native)".into();
        report.family = Some("deepseek_v4".into());
        report.spec.as_mut().unwrap().moe.as_mut().unwrap().intermediate = 3072;
        let pro = DeploymentSpec::from_report(&report, &options).unwrap();
        assert_eq!(pro.topology.tp, 4);
        // The shipped k23 tier-pair package a WIP or release image carries.
        assert_eq!(pro.artifact_requirements.spark_packages.keys().collect::<Vec<_>>(), ["exl3/exl3-dsv4p-k23"]);
        report.memory_layout = None;
        assert!(DeploymentSpec::from_report(&report, &options).is_err());
    }
}
