//! CLI-resolved replicated Spark `TP×EP` topology shared by both daemons.
//!
//! The only source of rank identity is
//! [`cuteafd_transport::expert::SparkTopology`]: this module never
//! re-derives executor ids, group indices or physical rank maps, so transport,
//! worker shard selection and coordinator assembly cannot drift apart.
use anyhow::{ensure, Result};
use cuteafd_loader::OfficialV41Catalog;
use cuteafd_transport::expert::{SparkTopology, SparkWorld};

/// Resolve the opt-in `--spark-tp N --spark-ep M` pair for a command that knows
/// how many physical Spark ranks were launched.
///
/// Both keys are all-or-none: absent selects canonical wire for the pure-TP rank
/// count, present selects the ownership-aware replicated-group contract. `world` is the
/// physical rank count (coordinator peers / worker `--world`), and it must equal
/// `TP * EP` exactly.
pub(crate) fn resolve(
    tp: Option<u8>,
    ep: Option<u8>,
    world: usize,
    component: &str,
) -> Result<Option<SparkTopology>> {
    let topology = match (tp, ep) {
        (None, None) => return Ok(None),
        (Some(tp), Some(ep)) => SparkTopology::new(tp, ep)?,
        _ => anyhow::bail!(
            "{component} --spark-tp and --spark-ep must be given together or not at all"
        ),
    };
    ensure!(
        topology.world_size() == world,
        "{component} Spark topology {}x{} needs {} physical ranks but {world} were given",
        topology.tp(),
        topology.ep(),
        topology.world_size()
    );
    Ok(Some(topology))
}

/// Resolve the declared wire independently from the physical TP/EP layout.
/// Omitted CLI keys choose canonical pure TP. No transport derives rank
/// identities from a fallback.
pub(crate) fn world(topology: Option<SparkTopology>, ranks: usize) -> Result<SparkWorld> {
    let world = match topology {
        Some(topology) => SparkWorld::owned(topology)?,
        None => SparkWorld::canonical(ranks)?,
    };
    ensure!(world.topology().world_size() == ranks,
        "declared Spark topology {}x{} needs {} physical ranks but {ranks} were given",
        world.topology().tp(), world.topology().ep(), world.topology().world_size());
    Ok(world)
}

/// Check package qualification before admitting any retained world. The native
/// ownership wire requires native experts; generic programs keep canonical routes.
pub(crate) fn require_supported(
    topology: Option<SparkTopology>,
    catalog: &OfficialV41Catalog,
) -> Result<()> {
    if let Some(topology) = topology {
        ensure!(catalog.fp8().is_none(),
            "generic expert programs require canonical Spark wire");
        let nvfp4 = catalog.nvfp4().is_some();
        ensure!(catalog.exl3().is_none() && (!nvfp4
                || (topology.ep() == 1 && matches!(topology.tp(), 3 | 4 | 6))),
            "unsupported Spark ownership wire {}x{} for this checkpoint; NVFP4 on TP3EP1, TP4EP1, TP6EP1 only; EXL3 requires canonical wire",
            topology.tp(), topology.ep());
        let geometry = catalog.routed_experts().geometry()?;
        let family = geometry.family().unwrap_or("v41");
        ensure!(cuteafd_core::tp_units::spark_package_worlds(family,
                if nvfp4 { "nvfp4" } else { "mxfp4" }).contains(&(topology.tp() as usize)),
            "checkpoint has no native Spark width-class recipe for TP{}", topology.tp());
        SparkWorld::owned(topology)?;
    }
    Ok(())
}

pub(crate) fn require_wire(world: SparkWorld, catalog: &OfficialV41Catalog) -> Result<()> {
    if catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8 {
        ensure!(world.ownership_topology().is_none(),
            "EXLR8 unit ownership requires canonical Spark wire");
        ensure!((1..=8).contains(&world.topology().world_size()), "EXLR8 requires 1..=8 ranks");
        return Ok(());
    }
    require_supported(world.ownership_topology(), catalog)?;
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

/// Roles published by the native Spark shard families, for callers that need
/// the expected role without loading the library. `TP6` is the pure
/// unreplicated six-rank layout (one disjoint intermediate slice of every
/// expert per rank); `TP2`/`TP3` are the replicated-group shards. Three-, four-
/// and six-rank layouts reduce through the native generic N-plane entry point;
/// admission uses `V41CompactReducer::require_rank_count` on the loaded library
/// before any allocation or readiness publication, so a missing artifact fails
/// startup rather than the first request.
pub(crate) const SPARK_TP2_ROLE: u32 = 5;
pub(crate) const SPARK_TP3_ROLE: u32 = 6;
/// Native role id of the pure `TP6EP1` Spark shard family.
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
        let python = source.join(".venv/bin/python");
        let python = if python.is_file() { python } else {
            let common = std::process::Command::new("git").arg("-C").arg(&source)
                .args(["rev-parse", "--path-format=absolute", "--git-common-dir"]).output()?;
            anyhow::ensure!(common.status.success(), "cannot locate checkout Python environment");
            std::path::Path::new(std::str::from_utf8(&common.stdout)?.trim())
                .parent().unwrap().join(".venv/bin/python")
        };
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
                _ => anyhow::bail!("missing full launch fixture for covered {key}"),
            }
            if !matches!(family, "v41" | "kimik3") {
                expand_eight_rank_fixture(&snapshot)?;
            }
            let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
            let gpus = if family == "kimik3" { 2 } else { 1 };
            let manifest = if family == "v41" { Some(testing::write_v41_workspaces(&snapshot)) }
                else if family == "kimik3" { Some(testing::write_kimi_installed_inventory(root.path(), false)) }
                else { None };
            let options = PlanOptions {
                placement: ExpertPlacement::from_spark_ranks(8),
                draft_ownership: Some(cuteafd_loader::plan::spec::DraftOwnership::None),
                vision: cuteafd_loader::plan::MediaMode::Off, audio: cuteafd_loader::plan::MediaMode::Off,
                mapped_scale_host_budget_bytes: Some(64 << 30),
                layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)),
                    rtx_bytes: vec![96 << 30; gpus], host_cache_bytes: Some(0), pool_tokens: Some(262_144),
                    context_tokens: 262_144, prefill_rows: 4096, prefill_lanes: 1,
                    workspace_manifest: manifest, glm5_drafter_disabled: true, glmf_drafter_disabled: true,
                    ..Default::default() }), ..Default::default()
            };
            let report = testing::BudgetedPlanInput::open(&snapshot)?.plan(&options)?;
            assert!(report.fits && report.placement_supported, "{key}: {:?}", report.hints);
            let deployment_options = DeploymentOptions { rtx_gpus: gpus,
                peers: (0..8).map(|rank| format!("192.0.2.{}:9100", rank + 1)).collect(),
                speculator: "off".into(), ..Default::default() };
            let deployment = DeploymentSpec::from_report(&report, &deployment_options)?;
            assert!(deployment.executable, "{key}: {:?}", report.hints);
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
            let (peers, tp, ep) = match cli.command {
                Commands::ServeNative(a) => (a.peers.len(), a.spark_tp, a.spark_ep),
                Commands::ServeDsv4(a) => (a.engine.peers.split(',').count(), None, None),
                Commands::ServeGlm(a) => (a.engine.peers.as_deref().unwrap().split(',').count(), None, None),
                Commands::ServeGlmf(a) => (a.engine.peers.as_deref().unwrap().split(',').count(), None, None),
                Commands::ServeMimo(a) => (a.engine.peers.as_deref().unwrap().split(',').count(), None, None),
                Commands::ServeKimi(a) => (a.peers.len(), None, None),
                other => anyhow::bail!("{key}: wrong dispatch {other:?}"),
            };
            assert_eq!(peers, 8, "{key}: lost peers in launch/parse");
            require_world(world(resolve(tp, ep, peers, "launch-chain")?, peers)?, &catalog)?;
            if family == "v41" { assert_eq!(captured["layer_bytes"], 1_203_240_960u64); }
            let hidden = catalog.routed_experts().geometry()?.hidden as u64;
            let rows = deployment.prefill_unit_rows;
            let waves = deployment.prefill_waves;
            let intake = cuteafd_loader::serving_capacity::spark_intake_bytes(waves, peers as u64, rows, hidden);
            assert_eq!(intake, 8 * waves * rows * hidden * 2, "{key}");
            if family == "v41" {
                use cuteafd_loader::placement::families::deepseek_v41::owners;
                let charged = owners::transports_bytes(64, rows, peers, waves as usize)?;
                assert_eq!(charged, 2 * 64 * (8 + 2) * 5120 * 2 + rows * waves * (1 + 8) * 5120 * 2);
            }
            if family != "kimik3" {
                let host = report.host_ledger.as_ref().expect("full host intake ledger");
                let (group, expected) = if matches!(family, "mimo" | "mimof" | "mimop") {
                    ("mimo/spark-intake-upper-bound", (rows * waves + 64) * hidden * 2 * 8)
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
                let demand = crate::families::kimi_k3::spark::admitted_demand(waves as usize, rows as usize, 0, peers)?;
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
                if supported.contains(&ranks) {
                    require_world(SparkWorld::owned(SparkTopology::new(ranks as u8, 1)?)?, &catalog)?;
                }
                launch_check(ranks, supported.contains(&ranks), true)?;
            }
        }
        Ok(())
    }

    fn launch_check(ranks: usize, supported: bool, owned: bool) -> Result<()> {
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let output = std::process::Command::new("bash")
            .args(["-c", "source \"$1/scripts/lib/release-common.sh\"; SPARK_COUNT=$2; SPARK_TP=$3; SPARK_EP=$4; EXPERT_FORMAT=native; SPARKINFER_EXL3=disable; release_validate_spark_topology", "launch-test"])
            .arg(source).arg(ranks.to_string())
            .arg(if owned { ranks.to_string() } else { String::new() })
            .arg(if owned { "1" } else { "" }).output()?;
        // Launch checks only wire qualification; checkpoint recipes are checked
        // by serve, so another native family's valid width may pass the shell.
        let expected = if owned { cuteafd_core::tp_units::spark_ownership_topology(ranks, 1) } else { true };
        assert_eq!(output.status.success(), expected, "TP{ranks} supported={supported}: {}", String::from_utf8_lossy(&output.stderr));
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
                launch_check(ranks, expected, false)?;
            }
        }
        Ok(())
    }

    #[test]
    fn generic_programs_refuse_ownership_wire() -> Result<()> {
        use cuteafd_loader::plan::testing::{mimo_flash_config, mimo_flash_tensors, write_snapshot};
        let root = tempfile::tempdir()?;
        write_snapshot(root.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let error = require_world(SparkWorld::owned(SparkTopology::NATIVE_TP4_EP1)?, &catalog)
            .unwrap_err();
        assert_eq!(error.to_string(), "generic expert programs require canonical Spark wire");
        Ok(())
    }

    #[test]
    fn ownership_refusal_names_requested_and_supported_layouts() -> Result<()> {
        for exl3 in [false, true] {
            let root = tempfile::tempdir()?;
            if exl3 {
                cuteafd_loader::plan::testing::write_v4_pro_exl3_snapshot(root.path());
            } else {
                cuteafd_loader::plan::testing::write_v41_nvfp4_snapshot(root.path());
            }
            let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
            let error = require_supported(Some(SparkTopology::NATIVE_TP2_EP1), &catalog)
                .unwrap_err().to_string();
            assert!(error.contains("ownership wire 2x1"), "{error}");
            assert!(error.contains("NVFP4 on TP3EP1, TP4EP1, TP6EP1"), "{error}");
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

    fn topology(tp: u8, ep: u8) -> SparkTopology {
        SparkTopology::new(tp, ep).unwrap()
    }

    #[test]
    fn absent_keys_select_canonical_wire() {
        assert_eq!(resolve(None, None, 4, "serve-native").unwrap(), None);
        // Physical topology is declared when the wire selection becomes a world.
        assert_eq!(resolve(None, None, 2, "expertd-native").unwrap(), None);
    }

    #[test]
    fn world_resolution_binds_count_and_retains_canonical_device_tp4() {
        let canonical = world(None, 4).unwrap();
        assert!(canonical.supports_device_exchange());
        assert_eq!(canonical.topology(), SparkTopology::NATIVE_TP4_EP1);
        assert!(!world(Some(SparkTopology::NATIVE_TP4_EP1), 4).unwrap().supports_device_exchange());
        let error = world(Some(SparkTopology::NATIVE_TP3_EP1), 4).unwrap_err().to_string();
        assert!(error.contains("topology 3x1"), "{error}");
        assert!(error.contains("needs 3 physical ranks but 4 were given"), "{error}");
        for count in [0, cuteafd_core::MAX_SPARK_WORLD + 1] { assert!(world(None, count).is_err()); }
        for count in [7, 8] {
            let resolved = world(None, count).unwrap();
            assert_eq!(resolved.topology().world_size(), count);
            assert!(!resolved.supports_device_exchange());
        }
    }

    #[test]
    fn keys_are_all_or_none() {
        assert!(resolve(Some(2), None, 4, "serve-native").is_err());
        assert!(resolve(None, Some(2), 4, "serve-native").is_err());
    }

    #[test]
    fn approved_layouts_require_their_exact_rank_count() {
        for (tp, ep) in [(1u8, 1u8), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (2, 2), (3, 2), (2, 3)] {
            let expected = topology(tp, ep);
            let resolved = resolve(Some(tp), Some(ep), expected.world_size(), "serve-native")
                .unwrap()
                .expect("explicit topology");
            assert_eq!(resolved, expected);
            assert_eq!(resolved.world_size(), tp as usize * ep as usize);
            assert!(resolve(Some(tp), Some(ep), expected.world_size() + 1, "serve-native").is_err());
        }
    }

    #[test]
    fn unapproved_layouts_and_ranges_are_rejected() {
        for (tp, ep) in [(9u8, 1u8), (2, 4), (4, 2), (3, 3), (0, 1), (6, 2), (6, 3)] {
            let world = tp as usize * ep as usize;
            assert!(resolve(Some(tp), Some(ep), world, "serve-native").is_err(), "{tp}x{ep}");
        }
    }

    #[test]
    fn pure_tp6_is_one_unreplicated_group_of_six_disjoint_shards() {
        let resolved = resolve(Some(6), Some(1), 6, "expertd-native")
            .unwrap()
            .expect("explicit TP6EP1 topology");
        assert_eq!(resolved.tp(), 6);
        assert_eq!(resolved.ep(), 1);
        assert_eq!(resolved.group_count(), 1);
        for rank in 0..6 {
            assert_eq!(resolved.group(rank).unwrap(), 0);
            assert_eq!(resolved.tp_rank(rank).unwrap(), rank as u8);
            assert_eq!(resolved.executor_id(rank).unwrap(), 27 + rank as u64);
        }
        assert!(resolved.group(6).is_err());
        // The canonical six-rank group shares the TP6EP1 namespace.
        for rank in 0..6 {
            assert_eq!(cuteafd_transport::expert::SparkWorld::canonical(6).unwrap().topology().executor_id(rank).unwrap(),
                resolved.executor_id(rank).unwrap());
        }
    }

}
