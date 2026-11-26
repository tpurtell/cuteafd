//! The launched Spark world shared by both daemons, and its checkpoint
//! qualification.
//!
//! The only source of rank identity is
//! [`cuteafd_transport::expert::SparkTopology`]: this module never
//! re-derives executor ids or rank maps, so transport, worker shard selection
//! and coordinator assembly cannot drift apart.
use anyhow::{ensure, Result};
use cuteafd_loader::OfficialV41Catalog;
use cuteafd_transport::expert::SparkWorld;

/// The canonical world for the launched physical rank count (coordinator
/// peers / worker `--world`).
pub(crate) fn world(ranks: usize) -> Result<SparkWorld> {
    SparkWorld::canonical(ranks)
}

/// Check the checkpoint's reply formats before admitting any world.
pub(crate) fn require_supported(catalog: &OfficialV41Catalog) -> Result<()> {
    if let Some(tensors) = catalog.fp8() {
        super::experts::fp8::validate_reply_formats(tensors.format(), cuteafd_core::spark_reply::requested())?;
    }
    Ok(())
}

/// Every rank count in the checkpoint's Spark package range is served on the
/// canonical wire; the registry, not a per-family list, gates native worlds.
pub(crate) fn require_wire(world: SparkWorld, catalog: &OfficialV41Catalog) -> Result<()> {
    if catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8 {
        ensure!((1..=8).contains(&world.topology().world_size()), "EXLR8 requires 1..=8 ranks");
        return Ok(());
    }
    require_supported(catalog)?;
    if catalog.exl3().is_none() && catalog.fp8().is_none() {
        let geometry = catalog.routed_experts().geometry()?;
        let family = geometry.family().unwrap_or("v41");
        let format = if catalog.nvfp4().is_some() { "nvfp4" } else { "mxfp4" };
        ensure!(cuteafd_core::tp_units::spark_package_worlds(family, format)
                .contains(&(world.topology().tp() as usize)),
            "checkpoint has no native Spark width-class recipe for TP{}", world.topology().tp());
    }
    Ok(())
}

pub(crate) fn require_world(world: SparkWorld, catalog: &OfficialV41Catalog, layers: &[usize]) -> Result<()> {
    require_wire(world, catalog)?;
    if catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8 {
        let shape = catalog.routed_experts();
        ensure!(!layers.is_empty() && layers.windows(2).all(|pair| pair[0] < pair[1])
            && layers.iter().all(|&layer| layer >= shape.first_layer && layer < shape.layers),
            "EXLR8 hosted layers must be nonempty, sorted, unique routed target layers");
        return Ok(());
    }
    let tp = world.topology().tp() as usize;
    let headers = cuteafd_loader::plan::files::open_role(catalog.snapshot(),
        cuteafd_loader::plan::files::ReadRole::Spark { rank: 0, world: tp })?;
    cuteafd_loader::plan::experts::checkpoint_spark_recipe(&headers, tp, layers)?;
    Ok(())
}

/// Roles published by the native Spark TP shard families, for callers that
/// need the expected role without loading the library: each rank holds one
/// disjoint intermediate slice of every expert. Multi-rank layouts reduce
/// through the native generic N-plane entry point; admission uses
/// `V41CompactReducer::require_rank_count` on the loaded library before any
/// allocation or readiness publication, so a missing artifact fails startup
/// rather than the first request.
pub(crate) const SPARK_TP2_ROLE: u32 = 5;
pub(crate) const SPARK_TP3_ROLE: u32 = 6;
pub(crate) const SPARK_TP6_ROLE: u32 = 7;

#[cfg(test)]
mod tests {
    use super::*;

    fn require_world(world: SparkWorld, catalog: &OfficialV41Catalog) -> Result<()> {
        let shape = catalog.routed_experts();
        super::require_world(world, catalog, &(shape.first_layer..shape.layers).collect::<Vec<_>>())
    }

    #[test]
    fn eight_rank_user_launch_chain_has_no_hidden_count_caps() -> Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::{testing, ExpertPlacement, PlanOptions,
            deployment::{DeploymentOptions, DeploymentSpec}, layout::LayoutOptions};
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let data: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../scripts/build/spark-package-ranges.json"))?;
        let python = std::env::var_os("CUTEAFD_TEST_PYTHON").map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/cuteafd/gate-venv/bin/python"));
        ensure!(python.is_file(), "CUTEAFD_TEST_PYTHON must name the gate Python");
        let mut spark_rank_bytes = cuteafd_loader::serving_capacity::spark_memory::serving_bytes();
        // The two simulated ranks repeat the first measured serving host.
        spark_rank_bytes.extend([spark_rank_bytes[0]; 2]);
        let mut tested = Vec::new();
        for (key, _) in data["entries"].as_object().unwrap() {
            let (family, format) = key.split_once(':').unwrap();
            let worlds = cuteafd_core::tp_units::spark_package_worlds(family, format);
            if !worlds.iter().any(|n| *n >= 7) { continue; }
            assert!(worlds.contains(&8), "{key}: add N7 contract if N8 is genuinely unavailable");
            eprintln!("N8 launch-chain: {key}");
            let root = tempfile::tempdir()?;
            let snapshot = root.path().join("snapshot");
            std::fs::create_dir(&snapshot)?;
            match (family, format) {
                ("v41", _) => testing::write_v41_snapshot(&snapshot),
                ("dsv4f", _) => testing::write_v4_snapshot(&snapshot),
                ("dsv4p", "mxfp4") => testing::write_v4_pro_mxfp4_snapshot(&snapshot),
                ("dsv4p", "exl3") => testing::write_v4_pro_exl3_snapshot(&snapshot),
                ("kimik3", _) => testing::write_kimi_snapshot(&snapshot, true),
                ("minimax", "nvfp4") => write_minimax_release_text_snapshot(&snapshot, &source)?,
                ("inkling2", _) => write_inkling_release_text_snapshot(&snapshot, format, &source)?,
                ("muse", _) => testing::write_muse_spark_snapshot(&snapshot, &(0..62).collect::<Vec<_>>()),
                ("mimo", _) => testing::write_snapshot(&snapshot, &testing::mimo_flash_config(), &testing::mimo_flash_tensors(), None),
                ("mimof", _) => testing::write_snapshot(&snapshot, &testing::mimo_flash_mopd_config(), &testing::mimo_flash_mopd_tensors(), Some(4)),
                ("mimop", _) => testing::write_snapshot(&snapshot, &testing::mimo_pro_config(), &testing::mimo_pro_tensors(), Some(8)),
                ("glm", _) => {
                    let mut config = testing::glm5_config();
                    if format == "exl3" { config["quantization_config"] = testing::exl3_compact(4); }
                    testing::write_snapshot(&snapshot, &config,
                        &testing::glm5_tensors(|name, n, k| if format == "exl3" {
                            testing::exl3(name, n, k, 4)
                        } else { testing::fp8(name, n, k, None) }), None);
                }
                ("glmf", _) => {
                    let mut config = testing::glm5_flash_config(3);
                    let mut tensors = testing::glm5_flash_tensors(&config);
                    if format == "exl3" {
                        tensors.retain(|(name, ..)| !name.contains(".mlp.experts."));
                        for layer in 1..3 { for expert in 0..288 {
                            for (proj, n, k) in [("gate_proj", 2048, 4096), ("up_proj", 2048, 4096), ("down_proj", 4096, 2048)] {
                                tensors.extend(testing::exl3(&format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"), n, k, 3 + expert % 2));
                            }
                        } }
                        config["quantization_config"] = testing::exl3_compact(3);
                    }
                    testing::write_snapshot(&snapshot, &config, &tensors, None);
                }
                // The full release header inventory (every expert layer), EXL3 K4 + Core-K6.
                ("ml4", _) => testing::write_mistral_large4_snapshot(&snapshot, None, true),
                ("hy4", _) => write_hy4_exl3_snapshot(&snapshot)?,
                _ => anyhow::bail!("missing full launch fixture for covered {key}"),
            }
            if !matches!(family, "v41" | "kimik3" | "muse" | "ml4" | "minimax" | "inkling2" | "hy4") {
                expand_eight_rank_fixture(&snapshot)?;
            }
            let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
            let gpus = if family == "kimik3" { 2 } else { 1 };
            let manifest = if family == "v41" { Some(testing::write_v41_workspaces(&snapshot)) }
                else if family == "kimik3" { Some(testing::write_kimi_installed_inventory(root.path(), false)) }
                else if family == "ml4" {
                    // Its K6 core plans against the native programs of its own geometry.
                    let checkpoint = cuteafd_loader::plan::Checkpoint::open(&snapshot)?;
                    let cfg = cuteafd_loader::families::glm5::GlmDsaConfig::from_hf(&checkpoint.config)?;
                    let core = cuteafd_loader::families::glm5::core::CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
                    let path = root.path().join("PROGRAMS.json");
                    std::fs::write(&path, testing::glm_k6_programs(&cfg, &core)?.to_string())?;
                    Some(path)
                }
                else { None };
            let options = PlanOptions {
                // This topology fixture publishes owned-MLA programs, not Context kernels.
                attention_placement: (family == "kimik3").then_some(cuteafd_loader::placement::AttentionPlacement::Layers),
                placement: ExpertPlacement::from_spark_ranks(8),
                draft_ownership: Some(cuteafd_loader::plan::spec::DraftOwnership::None),
                vision: cuteafd_loader::plan::MediaMode::Off, audio: cuteafd_loader::plan::MediaMode::Off,
                mapped_scale_host_budget_bytes: Some(64 << 30),
                layout: Some(LayoutOptions { host_memory: Some(cuteafd_loader::plan::testing::SimulatedHost::raptor().host_ram),
                    rtx_bytes: vec![cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES; gpus], host_cache_bytes: Some(0), pool_tokens: Some(262_144),
                    spark_rank_bytes: spark_rank_bytes.clone(),
                    context_tokens: 262_144, prefill_rows: 4096, prefill_lanes: 1,
                    workspace_manifest: manifest, glm5_drafter_disabled: true, glmf_drafter_disabled: true,
                    ..Default::default() }), ..Default::default()
            };
            let report = testing::BudgetedPlanInput::open(&snapshot)?.plan(&options)?;
            if key == "inkling2:nvfp4" {
                // The mixed checkpoint's MXFP8 and NVFP4 rows use incompatible TP8 ownership.
                assert!(!report.placement_supported && !report.executable(), "{key}: TP8 must be refused");
                assert!(report.hints.iter().any(|hint| hint.what ==
                    "mixed Spark packages inkling2:mxfp8 + inkling2:nvfp4a4 require incompatible row ownership at TP8"),
                    "{key}: wrong refusal: {:?}", report.hints);
                tested.push(key.clone());
                continue;
            }
            if key == "minimax:nvfp4" {
                // When MiniMax becomes servable, replace this refusal witness with the
                // complete N8 launch/count-cap chain below rather than silently accepting it.
                assert_eq!(report.runtime, Some(cuteafd_loader::plan::RuntimeStatus::Planned), "{key}: qualify the real launch chain");
                assert!(!report.wip_launchable(), "{key}: qualify the real launch chain");
                assert!(report.hints.iter().any(|hint| hint.what
                    == "planned family has no executable serving admission in this build"),
                    "{key}: wrong refusal: {:?}", report.hints);
            }
            // Coverage follows each family's own runtime status: a Planned family must
            // be refused at deployment until it serves; flipping it to Serving puts it
            // through the whole chain below (and requires its launch dispatch).
            if report.runtime == Some(cuteafd_loader::plan::RuntimeStatus::Planned) {
                let refused = DeploymentOptions { rtx_gpus: gpus,
                    peers: (0..8).map(|rank| format!("192.0.2.{}:9100", rank + 1)).collect(),
                    speculator: "off".into(), ..Default::default() };
                let deployment = DeploymentSpec::from_report(&report, &refused)?;
                assert!(!report.executable() && !deployment.executable, "{key}: a planned family must be refused at deployment");
                eprintln!("N8 launch-chain: {key} planned; refused at deployment");
                tested.push(key.clone());
                continue;
            }
            assert!(report.fits && report.placement_supported, "{key}: {:?}", report.hints);
            let deployment_options = DeploymentOptions { rtx_gpus: gpus,
                peers: (0..8).map(|rank| format!("192.0.2.{}:9100", rank + 1)).collect(),
                speculator: "off".into(), ..Default::default() };
            let deployment = DeploymentSpec::from_report(&report, &deployment_options)?;
            assert!(deployment.executable, "{key}: {:?}", report.hints);
            let spark_devices = deployment.memory_layout.devices.iter()
                .filter(|device| device.kind == cuteafd_core::memory_layout::DeviceKind::Spark).collect::<Vec<_>>();
            assert_eq!(spark_devices.len(), 8, "{key}");
            for device in spark_devices {
                assert_eq!(device.worker_budget_bytes, spark_rank_bytes[device.index as usize], "{key} rank {}", device.index);
                assert!(device.worker_budget_bytes <= device.capacity_bytes && device.worker_budget_bytes <= 121 << 30,
                    "{key}: worker ceiling exceeds device capacity");
            }
            let deployment_path = root.path().join("deployment.json");
            std::fs::write(&deployment_path, serde_json::to_vec(&deployment)?)?;
            let temporary = root.path().join("launch");
            std::fs::create_dir(&temporary)?;
            let output = std::process::Command::new(&python).args(["-I", "-c", r#"
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts/tests'))
from test_run_family_dispatch import capture_real_eight_rank_launch
print(json.dumps(capture_real_eight_rank_launch(*map(Path, sys.argv[2:]))))
"#]).arg(&source).arg(&snapshot).arg(&deployment_path).arg(&temporary).output()?;
            assert!(output.status.success(), "{key}: {}", String::from_utf8_lossy(&output.stderr));
            let captured: serde_json::Value = serde_json::from_slice(&output.stdout)?;
            let argv = captured["argv"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect::<Vec<_>>();
            let cli = crate::cli::Cli::try_parse_from(&argv)?;
            use crate::cli::Commands;
            let peers = match cli.command {
                Commands::ServeNative(a) => a.peers.len(),
                Commands::ServeDsv4(a) => a.engine.peers.split(',').count(),
                Commands::ServeGlm(a) => a.engine.peers.as_deref().unwrap().split(',').count(),
                Commands::ServeGlmf(a) => a.engine.peers.as_deref().unwrap().split(',').count(),
                Commands::ServeMimo(a) => a.engine.peers.as_deref().unwrap().split(',').count(),
                Commands::ServeKimi(a) => a.peers.len(),
                Commands::ServeInkling2(a) => {
                    assert_eq!(family, "inkling2", "{key}: wrong Inkling dispatch");
                    assert_eq!(a.prefill_waves, Some(deployment.prefill_waves as _),
                        "{key}: launch must preserve the admitted prefill wave count");
                    a.peers.len()
                }
                Commands::ServeMuse(a) => {
                    assert_eq!(family, "muse", "{key}: wrong Muse dispatch");
                    assert_eq!(a.prefill_waves, Some(deployment.prefill_waves as _),
                        "{key}: launch must preserve the admitted prefill wave count");
                    assert_eq!(a.prefill_waves, Some(1), "{key}: fixture requests one prefill lane");
                    a.peers.len()
                }
                other => anyhow::bail!("{key}: wrong dispatch {other:?}"),
            };
            assert_eq!(peers, 8, "{key}: lost peers in launch/parse");
            require_world(world(peers)?, &catalog)?;
            if family == "v41" {
                // TP8 retains logical/stored S288: payloads plus coded K32 scales.
                let (hidden, width, experts) = (5120u64, 288u64, 384u64);
                let payloads = hidden * width * 3 / 2;
                let scales = 2 * width * (1 + 2 * (hidden / 128))
                    + hidden * (1 + 2 * (width / 32).div_ceil(4));
                assert_eq!(captured["layer_bytes"], experts * (payloads + scales));
            }
            let hidden = catalog.routed_experts().geometry()?.hidden as u64;
            let rows = deployment.prefill_unit_rows;
            let waves = deployment.prefill_waves;
            let intake = cuteafd_loader::serving_capacity::prefill_return_bytes(waves, rows, peers as u64, hidden);
            assert_eq!(intake, 8 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(waves * rows * hidden * 2)?, "{key}");
            if family == "v41" {
                use cuteafd_loader::placement::families::deepseek_v41::owners;
                let charged = owners::transports_bytes(64, 2, rows, peers, waves as usize, cuteafd_core::SparkReplyFormat::Bf16)?;
                assert_eq!(charged, 2 * (8 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(64 * 5120 * 2)?
                    + 2 * 64 * 5120 * 2) + rows * waves * 5120 * 2
                    + 8 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(rows * waves * 5120 * 2)?);
            }
            if family == "v41" {
                // Intake is an explicitly estimated PINNED allocation in the
                // joint contract, not an exact item in the old host ledger.
                let ledger = report.phase_ledger.as_ref().expect("joint V4.1 physical ledger");
                let host = ledger.budgets.iter().find(|budget|
                    budget.locations.contains(&cuteafd_loader::placement::planner::Location::HOST))
                    .expect("shared host budget");
                let intake = rows * waves * hidden * 4 * peers as u64 * 3;
                let router = (rows + 64) * (hidden * 2 + 8 * 8);
                let staging = (rows + 64) * hidden * 8 + (2 << 20);
                let ready = ledger.phases.iter().find(|phase| phase.budget == host.id
                    && phase.phase == cuteafd_loader::placement::planner::Phase::Ready)
                    .expect("ready host phase");
                assert_eq!(ready.opaque, intake + router + staging,
                    "{key}: shared PINNED intake estimate omits a rank or wave");
                assert!(ledger.phases.iter().all(|phase| phase.owned + phase.opaque <= phase.available));
            } else if matches!(family, "glm" | "ml4" | "glmf") {
                // GLM's unified physical ledger admits GPU return reserves;
                // the deleted ordinary-host ledger is not an allocation owner.
                // Candidate-invariant coordinator-device intake planes are
                // named Transport owners; only probe-selected pinned/host
                // storage stays an allowance envelope.
                let placement = report.solver_placement.as_ref().expect("shared GLM placement");
                let planes: Vec<_> = placement.items[0].iter().filter(|item|
                    item.group == "Spark intake"
                        && item.category == cuteafd_core::memory_layout::Category::Transport).collect();
                assert_eq!(planes.len(), 1, "{key}: one admitted prefill endpoint");
                assert!(planes[0].bytes >= intake, "{key}: device intake omits ranks");
                let ledger = report.phase_ledger.as_ref().expect("joint physical ledger");
                assert!(ledger.phases.iter().all(|phase| phase.owned + phase.opaque <= phase.available));
            } else if matches!(family, "dsv4f" | "dsv4p") {
                // V4's contract owns its host tier: the eight ranks' pinned link
                // storage is a host-budget owner in the one solved ledger.
                let ledger = report.phase_ledger.as_ref().expect("V4 joint physical ledger");
                assert!(report.host_ledger.is_none(), "{key}: no generic host ledger beside the contract");
                assert!(ledger.budgets.iter().any(|budget| budget.id == "coordinator host RAM"), "{key}");
                assert!(ledger.phases.iter().any(|row| row.budget == "coordinator host RAM" && row.owned > 0), "{key}");
                assert!(ledger.phases.iter().all(|phase| phase.owned + phase.opaque <= phase.available));
            } else if family == "inkling2" {
                // The joint ledger owns coordinator device planes, not a generic host intake.
                assert!(report.host_ledger.is_none(), "{key}: unexpected generic host ledger");
                let planes = deployment.memory_layout.devices.iter()
                    .filter(|device| device.kind == cuteafd_core::memory_layout::DeviceKind::Rtx)
                    .flat_map(|device| &device.items)
                    .filter(|item| item.group == "Spark intake"
                        && item.category == cuteafd_core::memory_layout::Category::Transport)
                    .map(|item| item.bytes).sum::<u64>();
                assert_eq!(planes, intake, "{key}: one admitted intake plane per rank");
                let ledger = report.phase_ledger.as_ref().expect("Inkling joint physical ledger");
                assert!(ledger.phases.iter().all(|phase| phase.owned + phase.opaque <= phase.available));
            } else if family != "kimik3" {
                let host = report.host_ledger.as_ref().expect("full host intake ledger");
                let (group, expected) = if matches!(family, "mimo" | "mimof" | "mimop") {
                    ("mimo/spark-intake-upper-bound", (rows * waves + 64) * hidden * 2 * 8)
                } else if family == "muse" {
                    use cuteafd_loader::families::muse_spark::{MuseSparkConfig, replies::Cutoff};
                    let checkpoint = cuteafd_loader::plan::Checkpoint::open(&snapshot)?;
                    let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
                    let layout = options.layout.as_ref().unwrap();
                    let prefill = cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan {
                        unit_rows: rows, waves, span_rows: rows * waves,
                    };
                    let cutoff = Cutoff::with_formats(layout.muse_partial_bf16_max_rows,
                        cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
                        layout.spark_reply_formats)?;
                    let transport = cuteafd_loader::placement::families::muse_spark::host_transport(
                        &cfg, prefill, cutoff, layout.resolved_state_slots("muse_spark") as usize, peers)?;
                    ("spark-intake", transport.intake)
                } else {
                    let endpoints = cuteafd_loader::plan::layout::spark_endpoints(
                        report.family.as_deref().unwrap(), report.prefill_transports.unwrap()) as u64;
                    ("spark-intake", rows * waves * hidden * 2 * 8 * endpoints)
                };
                let charge = host.items.iter().filter(|item| item.group == group).collect::<Vec<_>>();
                assert_eq!(charge.len(), 1, "{key}: missing/duplicate {group}");
                assert_eq!(charge[0].bytes, expected, "{key}: host intake omits a rank");
            }
            if family == "kimik3" {
                let demand = crate::families::kimi_k3::spark::admitted_demand(waves as usize, rows as usize, 0, peers, cuteafd_core::SparkReplyFormat::Bf16)?;
                assert_eq!(demand.bytes, intake);
                let charged = report.memory_layout.as_ref().unwrap().devices.iter().flat_map(|d| &d.items)
                    .filter(|item| item.group == demand.group).collect::<Vec<_>>();
                assert_eq!(charged.len(), 1);
                assert_eq!(charged[0].bytes, demand.bytes);
            }
            tested.push(key.clone());
        }
        assert!(!tested.is_empty());
        eprintln!("N8 real config -> launch -> clap -> topology -> intake: {}", tested.join(", "));
        Ok(())
    }

    fn write_inkling_release_text_snapshot(snapshot: &std::path::Path, format: &str,
        source: &std::path::Path) -> Result<()> {
        use cuteafd_loader::{families::inkling2::Inkling2Config, plan::testing::{self, t, write_snapshot}};
        let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(source.join(
            "rust/crates/cuteafd-loader/src/families/inkling2/fixtures/config.json"))?)?;
        config.as_object_mut().unwrap().remove("vision_config");
        config.as_object_mut().unwrap().remove("mtp_config");
        // Keep a subset of the real layer pattern with enough routed bytes
        // to exercise the eight-rank size policy in every expert format.
        let layers = if format == "mxfp8" { 12 } else { 24 };
        let text = &mut config["text_config"];
        text["num_hidden_layers"] = layers.into();
        text["local_layer_ids"].as_array_mut().unwrap()
            .retain(|id| id.as_u64().unwrap() < layers);
        if format == "exl3" { config["quantization_config"] = testing::exl3_compact(4); }
        let cfg = Inkling2Config::from_hf(&config)?;
        let mut tensors = vec![t("model.llm.embed.weight", "BF16", &[cfg.vocab_size, cfg.hidden]),
            t("model.llm.unembed.weight", "BF16", &[cfg.vocab_size, cfg.hidden]),
            t("model.llm.embed_norm.weight", "BF16", &[cfg.hidden]), t("model.llm.norm.weight", "BF16", &[cfg.hidden]),
            t("model.llm.attn_res.output_query", "F32", &[cfg.hidden])];
        for layer in 0..cfg.layers {
            let p = format!("model.llm.layers.{layer}");
            for (name, n) in [("wq_du", 6144), ("wr_du", 768), ("wk_dv", 1024), ("wv_dv", 1024),
                ("gate_attn", 6144), ("attn_sink_proj", 48)] {
                tensors.push(t(format!("{p}.attn.{name}.weight"), "BF16", &[n, 4608]));
            }
            tensors.push(t(format!("{p}.attn.wo_ud.weight"), "BF16", &[4608, 6144]));
            tensors.push(t(format!("{p}.attn.q_l2_clip.scale"), "BF16", &[48, 128]));
            tensors.push(t(format!("{p}.attn.rel_l2_clip.scale"), "BF16", &[48, 16]));
            tensors.push(t(format!("{p}.attn.rel_logits_proj.proj"), "BF16", &[16, 1024]));
            for branch in ["k", "v"] {
                if cfg.local[layer] { tensors.push(t(format!("{p}.attn.{branch}_sconv.weight"), "BF16", &[1024, 1, 4])); }
                else {
                    tensors.push(t(format!("{p}.attn.{branch}_sconv.static_w"), "BF16", &[4, 1024]));
                    tensors.push(t(format!("{p}.attn.{branch}_sconv.weight_proj"), "BF16", &[32, 4608]));
                }
            }
            for name in ["attn_sconv", "mlp_sconv"] { tensors.push(t(format!("{p}.{name}.weight"), "BF16", &[4608, 1, 4])); }
            for name in if cfg.local[layer] { vec!["attn_norm", "mlp_norm"] }
                else { vec!["q_input_norm", "k_input_norm", "v_input_norm", "mlp_norm"] } {
                tensors.push(t(format!("{p}.{name}.weight"), "BF16", &[4608]));
            }
            tensors.push(t(format!("{p}.mlp.gate.weight"), "BF16", &[384, 4608]));
            tensors.push(t(format!("{p}.mlp.gate.bias"), "F32", &[384]));
            tensors.push(t(format!("model.llm.attn_res.mlp_queries.{layer}"), "F32", &[4608]));
            if layer > 0 { tensors.push(t(format!("model.llm.attn_res.attn_queries.{}", layer-1), "F32",
                if cfg.local[layer] { &[4608] } else { &[3, 4608] })); }
            if format == "exl3" {
                for expert in 0..384 {
                    for (name, n, k) in [("gate_proj", 2304, 4608), ("up_proj", 2304, 4608), ("down_proj", 4608, 2304)] {
                        tensors.extend(testing::exl3(&format!("{p}.mlp.experts.{expert}.{name}"), n, k, 4));
                    }
                }
            } else {
                let nvfp4 = format == "nvfp4" && layer + 8 < cfg.layers;
                for (name, n, k) in [("w13", 4608, 4608), ("w2", 4608, 2304)] {
                    tensors.push(t(format!("{p}.mlp.experts.{name}_weight"), if nvfp4 { "U8" } else { "F8_E4M3" },
                        &[384, n, k / if nvfp4 { 2 } else { 1 }]));
                    tensors.push(t(format!("{p}.mlp.experts.{name}_weight_scale"), if nvfp4 { "F8_E4M3" } else { "U8" },
                        &[384, n, k / if nvfp4 { 16 } else { 32 }]));
                    if nvfp4 {
                        tensors.push(t(format!("{p}.mlp.experts.{name}_weight_amax"), "F32", &[384]));
                        tensors.push(t(format!("{p}.mlp.experts.{name}_weight_outer_scale_denominator"), "F32", &[1]));
                    }
                }
            }
        }
        write_snapshot(snapshot, &config, &tensors, None);
        Ok(())
    }

    fn write_minimax_release_text_snapshot(snapshot: &std::path::Path, source: &std::path::Path) -> Result<()> {
        use cuteafd_loader::plan::testing::{t, write_snapshot};
        let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(source.join(
            "rust/crates/cuteafd-loader/src/families/minimax_m3/fixtures/config.json"))?)?;
        config.as_object_mut().unwrap().remove("vision_config");
        let mut tensors = vec![
            t("language_model.model.embed_tokens.weight", "BF16", &[200064, 6144]),
            t("language_model.lm_head.weight", "BF16", &[200064, 6144]),
            t("language_model.model.norm.weight", "BF16", &[6144]),
        ];
        // Independent release-header recipe: full text depth and all 128 native experts.
        for layer in 0..60 {
            let p = format!("language_model.model.layers.{layer}.");
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                tensors.push(t(format!("{p}{norm}.weight"), "BF16", &[6144]));
            }
            for (name, n, k) in [("q_proj", 8192, 6144), ("k_proj", 512, 6144),
                ("v_proj", 512, 6144), ("o_proj", 6144, 8192),
                ("index_q_proj", 512, 6144), ("index_k_proj", 128, 6144)] {
                let stem = format!("{p}self_attn.{name}");
                tensors.push(t(format!("{stem}.weight"), "F8_E4M3", &[n, k]));
                tensors.push(t(format!("{stem}.weight_scale_inv"), "U8", &[n, k / 32]));
            }
            for norm in ["q_norm", "k_norm", "index_q_norm", "index_k_norm"] {
                tensors.push(t(format!("{p}self_attn.{norm}.weight"), "BF16", &[128]));
            }
            let (stem, intermediate) = if layer < 3 { ("mlp", 12288) }
                else { ("block_sparse_moe.shared_experts", 3072) };
            for (proj, n, k) in [("gate", intermediate, 6144), ("up", intermediate, 6144),
                ("down", 6144, intermediate)] {
                let stem = format!("{p}{stem}.{proj}_proj");
                tensors.push(t(format!("{stem}.weight"), "F8_E4M3", &[n, k]));
                tensors.push(t(format!("{stem}.weight_scale_inv"), "U8", &[n, k / 32]));
            }
            if layer < 3 { continue; }
            tensors.push(t(format!("{p}block_sparse_moe.gate.weight"), "F32", &[128, 6144]));
            tensors.push(t(format!("{p}block_sparse_moe.e_score_correction_bias"), "F32", &[128]));
            for expert in 0..128 {
                for (proj, n, k) in [("w1", 3072, 6144), ("w3", 3072, 6144), ("w2", 6144, 3072)] {
                    let stem = format!("{p}block_sparse_moe.experts.{expert}.{proj}");
                    tensors.push(t(format!("{stem}.weight_packed"), "U8", &[n, k / 2]));
                    tensors.push(t(format!("{stem}.weight_scale"), "F8_E4M3", &[n, k / 16]));
                    for scale in ["weight_global_scale", "input_global_scale"] {
                        tensors.push(t(format!("{stem}.{scale}"), "F32", &[]));
                    }
                }
            }
        }
        write_snapshot(snapshot, &config, &tensors, None);
        Ok(())
    }

    fn write_hy4_exl3_snapshot(snapshot: &std::path::Path) -> Result<()> {
        use cuteafd_loader::plan::testing::{t, write_snapshot, Tensor};
        let indexer_types: Vec<&str> = (0..78)
            .map(|i| if i == 0 || (i >= 1 && (i - 1) % 4 == 0) { "full" } else { "shared" })
            .collect();
        let layer_types: Vec<&str> = vec!["deepseek_sparse_attention"; 78];
        let mlp_layer_types: Vec<&str> = (0..78).map(|i| if i == 0 { "dense" } else { "sparse" }).collect();
        let mut text_config: serde_json::Value = serde_json::from_str(r#"{
            "model_type": "hy_v4", "vocab_size": 120832, "hidden_size": 6144, "num_hidden_layers": 78,
            "intermediate_size": 18432, "first_k_dense_replace": 1, "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-5, "hidden_act": "silu", "attention_bias": false,
            "enable_ihc": true, "gated_mla": true, "gating_type": "elementwise",
            "enable_lm_head_fp32": true, "learnable_sink": true, "tie_word_embeddings": false,
            "hc_mult": 4, "hc_eps": 1e-6, "hc_magnitude": 2.0,
            "num_attention_heads": 64, "q_lora_rank": 2048, "kv_lora_rank": 512,
            "qk_head_dim": 256, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
            "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048,
            "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            "n_shared_experts": 1, "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
            "routed_scaling_factor": 2.827, "swiglu_limit": 10.0,
            "rope_parameters": { "mrope_interleaved": true, "mrope_section": [12, 10, 10], "rope_theta": 10000000, "rope_type": "default" }
        }"#)?;
        text_config["indexer_types"] = serde_json::json!(indexer_types);
        text_config["layer_types"] = serde_json::json!(layer_types);
        text_config["mlp_layer_types"] = serde_json::json!(mlp_layer_types);
        let config = serde_json::json!({
            "architectures": ["HYV4VLForConditionalGeneration"],
            "model_type": "hy_v4_vl",
            "vocab_size": 120832,
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "tie_word_embeddings": false,
            "image_token_id": 120120,
            "video_token_id": 120124,
            "eos_token_id": [120025],
            "quantization_config": {
                "quant_method": "exl3", "format": "exl3", "checkpoint_format": "exl3",
                "bits": 4.0, "codebook": "mul1", "mul1_multiplier": 2212286765u64, "expert_bits": 4.0
            },
            "text_config": text_config
        });
        let (h, v, hc) = (6144usize, 120832usize, 4usize);
        let mut tensors: Vec<Tensor> = vec![
            t("model.embed_tokens.weight", "BF16", &[v, h]),
            t("model.norm.weight", "BF16", &[h]),
            t("model.hc_head.hc_head_fn", "F32", &[hc, hc * h]),
            t("model.hc_head.hc_head_base", "F32", &[hc]),
            t("model.hc_head.hc_head_scale", "F32", &[1]),
            t("lm_head.weight", "BF16", &[v, h]),
        ];
        for layer in 0..78usize {
            let base = format!("model.layers.{layer}");
            tensors.push(t(format!("{base}.input_layernorm.weight"), "BF16", &[h]));
            tensors.push(t(format!("{base}.post_attention_layernorm.weight"), "BF16", &[h]));
            for tag in ["hc_attn_layer", "hc_mlp_layer"] {
                let p = format!("{base}.{tag}.hc_pre");
                tensors.push(t(format!("{p}.hc_fn"), "F32", &[2 * hc, hc * h]));
                tensors.push(t(format!("{p}.hc_base"), "F32", &[2 * hc]));
                tensors.push(t(format!("{p}.hc_scale"), "F32", &[2]));
            }
            let a = format!("{base}.self_attn");
            tensors.push(t(format!("{a}.q_a_proj.weight"), "BF16", &[2048, h]));
            tensors.push(t(format!("{a}.q_a_layernorm.weight"), "BF16", &[2048]));
            tensors.push(t(format!("{a}.q_b_proj.weight"), "BF16", &[64 * 256, 2048]));
            tensors.push(t(format!("{a}.kv_a_proj_with_mqa.weight"), "BF16", &[576, h]));
            tensors.push(t(format!("{a}.kv_a_layernorm.weight"), "BF16", &[512]));
            tensors.push(t(format!("{a}.kv_b_proj.weight"), "BF16", &[64 * 448, 512]));
            tensors.push(t(format!("{a}.linear_gate.weight"), "BF16", &[64 * 256, h]));
            tensors.push(t(format!("{a}.o_proj.weight"), "BF16", &[h, 64 * 256]));
            tensors.push(t(format!("{a}.learnable_sink_param"), "F32", &[64]));
            if layer == 0 || (layer >= 1 && (layer - 1) % 4 == 0) {
                let ip = format!("{a}.indexer");
                tensors.push(t(format!("{ip}.wq_b.weight"), "BF16", &[32 * 128, 2048]));
                tensors.push(t(format!("{ip}.wk.weight"), "BF16", &[128, h]));
                tensors.push(t(format!("{ip}.k_norm.weight"), "BF16", &[128]));
                tensors.push(t(format!("{ip}.k_norm.bias"), "BF16", &[128]));
                tensors.push(t(format!("{ip}.weights_proj.weight"), "BF16", &[32, h]));
            }
            if layer == 0 {
                tensors.push(t(format!("{base}.mlp.gate_proj.weight"), "BF16", &[18432, h]));
                tensors.push(t(format!("{base}.mlp.up_proj.weight"), "BF16", &[18432, h]));
                tensors.push(t(format!("{base}.mlp.down_proj.weight"), "BF16", &[h, 18432]));
            } else {
                tensors.push(t(format!("{base}.mlp.gate.weight"), "BF16", &[256, h]));
                tensors.push(t(format!("{base}.mlp.gate.e_score_correction_bias"), "F32", &[256]));
                tensors.push(t(format!("{base}.mlp.shared_experts.gate_proj.weight"), "BF16", &[2048, h]));
                tensors.push(t(format!("{base}.mlp.shared_experts.up_proj.weight"), "BF16", &[2048, h]));
                tensors.push(t(format!("{base}.mlp.shared_experts.down_proj.weight"), "BF16", &[h, 2048]));
                for expert in 0..256usize {
                    for (proj, n, k) in [("gate_proj", 2048usize, h), ("up_proj", 2048, h), ("down_proj", h, 2048)] {
                        let stem = format!("{base}.mlp.experts.{expert}.{proj}");
                        tensors.push(t(format!("{stem}.trellis"), "I16", &[k / 16, n / 16, 64]));
                        tensors.push(t(format!("{stem}.suh"), "F16", &[k]));
                        tensors.push(t(format!("{stem}.svh"), "F16", &[n]));
                        tensors.push(t(format!("{stem}.mul1"), "I32", &[]));
                    }
                }
            }
        }
        write_snapshot(snapshot, &config, &tensors, None);
        Ok(())
    }

    // Small family fixtures intentionally understate routed bytes. Complete all
    // experts and extend real layer templates so the N8 size policy applies.
    fn expand_eight_rank_fixture(snapshot: &std::path::Path) -> Result<()> {
        use cuteafd_loader::plan::testing;
        let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let inference_path = snapshot.join("inference/config.json");
        let native = inference_path.is_file();
        let text = if config.get("text_config").is_some() { &mut config["text_config"] } else { &mut config };
        let old_layers = if native { 4 } else { text["num_hidden_layers"].as_u64().unwrap() as usize };
        let layers = 48;
        let experts = if native {
            let mut args: serde_json::Value = serde_json::from_slice(&std::fs::read(&inference_path)?)?;
            args["n_layers"] = layers.into();
            let ratios = args["compress_ratios"].as_array().unwrap().clone();
            args["compress_ratios"] = (0..layers).map(|i| ratios[i.min(old_layers - 1)].clone()).collect();
            let experts = args["n_routed_experts"].as_u64().unwrap() as usize;
            std::fs::write(&inference_path, serde_json::to_vec(&args)?)?;
            experts
        } else {
            text["num_hidden_layers"] = layers.into();
            for key in ["layer_types", "mlp_layer_types", "hybrid_layer_pattern", "moe_layer_freq", "indexer_types"] {
                if let Some(original) = text.get(key).and_then(|v| v.as_array()).cloned() {
                    text[key] = (0..layers).map(|i| original[i.min(old_layers - 1)].clone()).collect();
                }
            }
            text["n_routed_experts"].as_u64().unwrap() as usize
        };
        let mut index: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("model.safetensors.index.json"))?)?;
        let shard = index["weight_map"].as_object().unwrap().values().next().unwrap().as_str().unwrap().to_owned();
        use std::io::Read;
        let mut file = std::fs::File::open(snapshot.join(&shard))?;
        let mut length = [0u8; 8]; file.read_exact(&mut length)?;
        let mut bytes = vec![0; u64::from_le_bytes(length) as usize]; file.read_exact(&mut bytes)?;
        let header: serde_json::Value = serde_json::from_slice(&bytes)?;
        let mut globals = Vec::new();
        let mut templates = std::collections::BTreeMap::<usize, Vec<testing::Tensor>>::new();
        for (name, tensor) in header.as_object().unwrap() {
            if name == "__metadata__" { continue; }
            let dtype = match tensor["dtype"].as_str().unwrap() {
                "BF16" => "BF16", "F16" => "F16", "F32" => "F32", "I32" => "I32", "F8_E4M3" => "F8_E4M3",
                "F8_E8M0" => "F8_E8M0", "I8" => "I8", "U8" => "U8", "I16" => "I16", "I64" => "I64",
                other => anyhow::bail!("unhandled fixture dtype {other}"),
            };
            let shape = tensor["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect::<Vec<_>>();
            let item = testing::t(name, dtype, &shape);
            if let Some((_, tail)) = name.split_once("layers.") {
                let layer = tail.split('.').next().unwrap().parse::<usize>()?;
                templates.entry(layer).or_default().push(item);
            } else { globals.push(item); }
        }
        let mut map = serde_json::Map::new();
        for layer in 0..layers {
            let template = layer.min(old_layers - 1);
            let mut tensors = if layer == 0 { globals.clone() } else { Vec::new() };
            let complete_experts = templates[&template].iter().any(|(name, ..)| name.contains(".experts.1."));
            for (name, dtype, shape) in &templates[&template] {
                let name = name.replacen(&format!("layers.{template}."), &format!("layers.{layer}."), 1);
                if name.contains(".experts.0.") && !complete_experts {
                    for expert in 0..experts { tensors.push(testing::t(name.replace(".experts.0.", &format!(".experts.{expert}.")), dtype, shape)); }
                } else { tensors.push(testing::t(name, dtype, shape)); }
            }
            let filename = format!("eight-rank-{layer:03}.safetensors");
            testing::write_safetensors(&snapshot.join(&filename), &tensors);
            for (name, ..) in tensors { map.insert(name, filename.clone().into()); }
        }
        std::fs::write(snapshot.join("config.json"), serde_json::to_vec(&config)?)?;
        index["weight_map"] = map.into();
        std::fs::write(snapshot.join("model.safetensors.index.json"), serde_json::to_vec(&index)?)?;
        std::fs::remove_file(snapshot.join(shard))?;
        Ok(())
    }

    #[test]
    fn native_worlds_share_serve_transport_and_launch_qualification() -> Result<()> {
        use cuteafd_loader::plan::testing;
        launch_check()?;
        for (family, write) in [
            ("v41", testing::write_v41_snapshot as fn(&std::path::Path)),
            ("dsv4f", testing::write_v4_snapshot),
            ("dsv4p", testing::write_v4_pro_mxfp4_snapshot),
        ] {
            let root = tempfile::tempdir()?;
            write(root.path());
            let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
            let supported = cuteafd_core::tp_units::spark_package_worlds(family, "mxfp4");
            for ranks in 1..=cuteafd_core::MAX_SPARK_WORLD {
                let canonical = require_world(SparkWorld::canonical(ranks)?, &catalog);
                assert_eq!(canonical.is_ok(), supported.contains(&ranks), "{family} TP{ranks}: {canonical:?}");
            }
        }
        // V4.1 serves every rank count its MXFP4 package range names on the
        // canonical wire, the frozen NVFP4 range only its admitted worlds.
        assert_eq!(cuteafd_core::tp_units::spark_package_worlds("v41", "mxfp4"), [2, 3, 4, 5, 6, 7, 8]);
        let root = tempfile::tempdir()?;
        testing::write_v41_nvfp4_snapshot(root.path());
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        for ranks in 1..=cuteafd_core::MAX_SPARK_WORLD {
            let served = require_world(SparkWorld::canonical(ranks)?, &catalog);
            assert_eq!(served.is_ok(), matches!(ranks, 2 | 3 | 4 | 6), "NVFP4 TP{ranks}: {served:?}");
        }
        Ok(())
    }

    /// The launcher carries no Spark topology keys: every count passes its
    /// shell validation and checkpoint recipes are checked by serve.
    fn launch_check() -> Result<()> {
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let output = std::process::Command::new("bash")
            .args(["-c", "source \"$1/scripts/lib/release-common.sh\"; for SPARK_COUNT in {1..8}; do release_validate_spark_count || exit; done", "launch-test"])
            .arg(source).output()?;
        assert!(output.status.success(), "launcher count qualification: {}", String::from_utf8_lossy(&output.stderr));
        Ok(())
    }

    #[test]
    fn generic_worlds_share_serve_and_launch_qualification() -> Result<()> {
        use cuteafd_loader::plan::testing;
        for family in ["mimo", "mimof", "mimop", "glm", "glmf", "qwen4", "dsv4p", "kimik3"] {
            let root = tempfile::tempdir()?;
            let (format, intermediate) = match family {
                "mimo" => { testing::write_snapshot(root.path(), &testing::mimo_flash_config(), &testing::mimo_flash_tensors(), None); ("fp8", 2048) }
                "mimof" => { testing::write_snapshot(root.path(), &testing::mimo_flash_mopd_config(), &testing::mimo_flash_mopd_tensors(), None); ("fp8", 2048) }
                "mimop" => { testing::write_snapshot(root.path(), &testing::mimo_pro_config(), &testing::mimo_pro_tensors(), None); ("fp8", 2048) }
                "glm" => { testing::write_snapshot(root.path(), &testing::glm5_config(), &testing::glm5_tensors(|name, n, k| testing::fp8(name, n, k, None)), None); ("fp8", 2048) }
                "glmf" => { let cfg = testing::glm5_flash_config(3); testing::write_snapshot(root.path(), &cfg, &testing::glm5_flash_tensors(&cfg), None); ("fp8", 2048) }
                "qwen4" => { let (tensors, _) = testing::qwen4_exl3(3, 4); testing::write_snapshot(root.path(), &testing::qwen4_config(3), &tensors, None); ("exl3-k4", 640) }
                "dsv4p" => { testing::write_v4_pro_exl3_snapshot(root.path()); ("exl3-k23", 3072) }
                "kimik3" => { testing::write_kimi_snapshot(root.path(), true); ("exl3-k23", 3072) }
                _ => unreachable!(),
            };
            let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
            for ranks in 1..=cuteafd_core::MAX_SPARK_WORLD {
                let expected = cuteafd_loader::plan::experts::spark_recipe(&format!("{family}:{format}"), intermediate, ranks).is_ok();
                let served = require_world(SparkWorld::canonical(ranks)?, &catalog);
                assert_eq!(served.is_ok(), expected, "{family} TP{ranks}: {served:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn checkpoint_recipe_refuses_unqualified_generic_world() -> Result<()> {
        use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        let root = tempfile::tempdir()?;
        let (tensors, _) = qwen4_exl3(3, 4);
        write_snapshot(root.path(), &qwen4_config(3), &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        require_world(SparkWorld::canonical(1)?, &catalog)?;
        require_world(SparkWorld::canonical(3)?, &catalog)?;
        let error = require_world(SparkWorld::canonical(5)?, &catalog).unwrap_err();
        assert!(error.downcast_ref::<cuteafd_loader::plan::experts::SparkRecipeRefusal>().is_some(),
            "{error:#}");
        Ok(())
    }

    #[test]
    fn world_binds_count_and_device_exchange_is_tp4_and_tp6() {
        for count in 1..=cuteafd_core::MAX_SPARK_WORLD {
            let resolved = world(count).unwrap();
            assert_eq!(resolved.topology().world_size(), count);
            assert_eq!(resolved.supports_device_exchange(), matches!(count, 4 | 6));
        }
        for count in [0, cuteafd_core::MAX_SPARK_WORLD + 1] { assert!(world(count).is_err()); }
        assert_eq!(world(6).unwrap().topology().executor_ids(), [27, 28, 29, 30, 31, 32]);
    }
}
