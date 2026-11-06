//! CPU-only end-to-end budget parsing and deployment/launcher contracts.
use std::process::Command;

#[test]
fn plan_accepts_global_budget_and_warns_once_for_each_alias() {
    for (flag, replacement) in [
        ("--coordinator-gpu-budget-gib", None),
        ("--rtx-budget-gib", Some("--coordinator-gpu-budget-gib")),
        ("--rtx-gib", Some("--coordinator-gpu-budget-gib")),
        ("--coordinator-weight-budget-gib", None),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
            .args(["plan", "/budget-cli-missing-snapshot", "--layout", flag, "31.8"])
            .env_remove("COORDINATOR_GPU_BUDGET_GIB")
            .output().unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        // Missing checkpoint fails after budget resolution, not at command admission.
        assert!(!output.status.success());
        assert!(!stderr.contains("applies only"), "{stderr}");
        assert!(!stderr.contains("unexpected argument"), "{stderr}");
        let warnings: Vec<_> = stderr.lines().filter(|line| line.starts_with("warning: deprecated coordinator budget flag")).collect();
        if let Some(replacement) = replacement {
            assert_eq!(warnings.len(), 1, "{stderr}");
            assert!(warnings[0].contains(replacement), "{stderr}");
        } else {
            assert!(warnings.is_empty(), "{stderr}");
        }
    }
}

#[test]
fn direct_family_concurrency_matches_logical_capacity_and_explicit_launcher_input() {
    for family in ["serve-glmf", "serve-glm", "serve-mimo"] {
        for gib in [30u64, 32, 96] {
            let expected = cuteafd_core::serving_capacity::default_concurrency(gib << 30);
            for explicit in [None, Some(expected), Some(4)] {
                let mut command = Command::new(env!("CARGO_BIN_EXE_cuteafd"));
                command.args([family, "--snapshot", "/direct-cli-missing-snapshot", "--native-lib", "/unused",
                    "--coordinator-gpu-budget-gib", &gib.to_string(), "--vision", "off", "--audio", "off"]);
                if let Some(value) = explicit { command.args(["--max-sequences", &value.to_string()]); }
                let output = command.env("RUST_LOG", "info")
                    .env_remove("COORDINATOR_GPU_BUDGET_GIB").output().unwrap();
                assert!(!output.status.success());
                let stderr = String::from_utf8_lossy(&output.stderr);
                let resolved = stderr.lines().find(|line| line.contains("resolved boot-time serving concurrency"))
                    .unwrap_or_else(|| panic!("{family} at {gib} GiB: {stderr}"));
                assert!(resolved.contains(&format!("concurrency={}", explicit.unwrap_or(expected))), "{resolved}");
                assert!(!stderr.contains("/unused"), "explicit logical budget must not load CUDA: {stderr}");
            }
        }
    }
}

#[test]
fn global_resident_weight_limit_survives_real_binary_family_forwarding() {
    for command in ["serve", "golden"] {
        for inner in [false, true] {
            let mut process = Command::new(env!("CARGO_BIN_EXE_cuteafd"));
            process.args(["--coordinator-weight-budget-gib", if inner { "73" } else { "0" },
                "--vision", "off", "--audio", "off", command, "--family", "qwen3_5",
                "--snapshot", "/budget-cli-missing-snapshot", "--native-lib", "/budget-cli-missing-native.so"]);
            if command == "golden" { process.args(["--golden", "/budget-cli-missing-golden"]); }
            if inner { process.args(["--coordinator-weight-budget-gib", "0"]); }
            let output = process.output().unwrap();
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!output.status.success());
            assert!(stderr.contains("--coordinator-weight-budget-gib: 0 GiB is not a finite positive size"),
                "{command}, inner={inner}: {stderr}");
            assert!(!stderr.contains("unexpected argument"), "{stderr}");
        }
    }
}

#[test]
fn global_resident_weight_flag_reaches_plan_admission() {
    use cuteafd_loader::plan::testing;
    let snapshot = tempfile::tempdir().unwrap();
    testing::write_kimi_snapshot(snapshot.path(), true);
    let package = tempfile::tempdir().unwrap();
    let manifest = testing::write_kimi_installed_inventory(package.path(), false);
    for (gib, accepted) in [("80", true), ("1", false)] {
        let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
            .args(["--coordinator-weight-budget-gib", gib, "plan"])
            .arg(snapshot.path()).arg("--workspace-manifest").arg(&manifest)
            .args(["--layout", "--rtx", "2", "--spark-ranks", "6", "--spark-budget-gib", "256",
                "--speculator", "off", "--attention-placement", "heads", "--rtx-expert-layers", "0", "--context-tokens", "262144",
                "--pool-tokens", "262144", "--concurrency", "2", "--prefix-slots", "0",
                "--host-cache-bytes", "0", "--prefill-rows", "2048", "--prefill-waves", "2",
                "--vision", "off", "--audio", "off", "--json", "--require-ready"])
            .env_remove("COORDINATOR_GPU_BUDGET_GIB")
            .env("HF_HOME", snapshot.path()).output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        let report: serde_json::Value = serde_json::from_slice(&output.stdout)
            .unwrap_or_else(|error| panic!("invalid plan JSON: {error}; {stderr}"));
        assert_eq!(output.status.success(), accepted, "{gib} GiB: {stderr}");
        if accepted {
            assert_eq!(report["fits"], true, "{report}");
            assert_eq!(report["memory_layout"]["pool_tokens"], 262144);
        } else {
            assert_eq!(report["fits"], false, "{report}");
            assert!(stderr.contains("resident weights need") && stderr.contains("1073741824 byte weight budget"), "{stderr}");
        }
    }
}

#[test]
fn global_gpu_storage_flag_reaches_inventory_and_full_layout_admission() {
    use cuteafd_loader::plan::testing;
    let snapshot = tempfile::tempdir().unwrap();
    testing::write_snapshot(snapshot.path(), &testing::mimo_flash_config(),
        &testing::mimo_flash_tensors(), Some(1));
    for layout in [false, true] {
        for (gib, accepted) in [("95.5", true), ("2", false)] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_cuteafd"));
            command.args(["--coordinator-gpu-budget-gib", gib,
                "--coordinator-weight-budget-gib", "80", "plan"]).arg(snapshot.path())
                .args(["--spark-ranks", "4", "--spark-budget-gib", "100", "--speculator", "off",
                    "--context-tokens", "262144", "--concurrency", "8", "--prefill-rows", "2048",
                    "--prefill-waves", "2", "--vision", "off", "--audio", "off", "--json", "--require-ready"]);
            if layout { command.args(["--layout", "--rtx", "2"]); }
            let output = command.env_remove("COORDINATOR_GPU_BUDGET_GIB")
                .env("HF_HOME", snapshot.path()).output().unwrap();
            let stderr = String::from_utf8_lossy(&output.stderr);
            let report: serde_json::Value = serde_json::from_slice(&output.stdout)
                .unwrap_or_else(|error| panic!("invalid plan JSON: {error}; {stderr}"));
            assert_eq!(output.status.success(), accepted, "layout={layout}, {gib} GiB: {stderr}; {report}");
            assert_eq!(report["placement_supported"], accepted, "{report}");
            assert_eq!(report["memory_layout"].is_object(), layout, "{report}");
            if !accepted {
                assert!(stderr.contains("is not servable"), "{stderr}");
                assert!(report["hints"].as_array().unwrap().iter().any(|h| h["what"].as_str()
                    .is_some_and(|h| h.contains("MiMo placement") && h.contains("memory budget 'rtx0'"))), "{report}");
            }
        }
    }
}

#[test]
fn plan_json_keeps_info_tracing_on_stderr() {
    let snapshot = tempfile::tempdir().unwrap();
    std::fs::write(snapshot.path().join("config.json"), br#"{"model_type":"unknown"}"#).unwrap();
    std::fs::write(snapshot.path().join("model.safetensors.index.json"), br#"{"weight_map":{}}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
        .arg("plan").arg(snapshot.path()).arg("--json")
        .env("RUST_LOG", "info").env("HF_HOME", snapshot.path())
        .output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("invalid JSON: {error}: {}", String::from_utf8_lossy(&output.stdout)));
    assert_eq!(report["fits"], false);
    assert!(stderr.contains("planner phase durations"), "{stderr}");
}

#[test]
fn deployment_rejects_conflicting_modes_and_incomplete_topology() {
    for flags in [
        vec!["--deployment", "--files"],
        vec!["--deployment", "--role", "coordinator"],
        vec!["--deployment", "--spark-tp", "2"],
        vec!["--deployment", "--spark-ep", "2"],
        vec!["--deployment-peers", "worker:9200"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
            .args(["plan", "/deployment-cli-missing-snapshot"])
            .args(&flags).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{flags:?}: {}",
            String::from_utf8_lossy(&output.stderr));
    }
}

#[test]
fn deployment_validates_admission_before_checkpoint_access() {
    for (flag, invalid) in [("--kv-pool-size", "-1GiB"), ("--memory-reservation", "101%")] {
        let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
            .args(["plan", "/deployment-cli-missing-snapshot", "--deployment"])
            .arg(format!("{flag}={invalid}")).output().unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("unexpected argument"), "{stderr}");
        assert!(!stderr.contains("missing config.json"), "{stderr}");
    }
}

fn release_accepts_prefill(deployment: &[u8]) -> serde_json::Value {
    let value: serde_json::Value = serde_json::from_slice(deployment).unwrap();
    for field in ["prefill_unit_rows", "prefill_waves"] {
        assert!(value[field].as_u64().is_some_and(|n| n > 0), "{field}: {value}");
    }
    assert!(value.get("prefill_slots").is_none(), "{value}");
    // Exercise the launcher's actual consumer, not a copied jq predicate or a
    // fabricated plan. Sourcing the whole launcher would start containers.
    let source = include_str!("../../../../scripts/lib/family-keys.sh");
    let function = |name: &str| {
        let start = source.find(&format!("{name}() {{\n")).unwrap();
        let end = source[start..].find("\n}\n").unwrap() + start + 3;
        &source[start..end]
    };
    let script = format!("set -euo pipefail\nrelease_die() {{ echo \"$*\" >&2; exit 1; }}\n{}\n{}\nrelease_prefill_shape \"$(cat \"$1\")\" \"$2\"\nprintf '%s %s\\n' \"$PREFILL_ROWS\" \"$PREFILL_WAVES\"",
        function("family_prefill_mode"), function("release_prefill_shape"));
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), deployment).unwrap();
    let result = Command::new("bash").args(["-c", &script, "prefill-contract"])
        .arg(file.path()).arg(value["family"].as_str().unwrap()).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    assert!(String::from_utf8_lossy(&result.stdout).ends_with(&format!("{} {}\n",
        value["prefill_unit_rows"], value["prefill_waves"])));
    value
}

#[test]
fn mimo_pro_max_plan_deployment_is_accepted_by_release_prefill_shape() {
    use cuteafd_loader::plan::testing;
    let snapshot = tempfile::tempdir().unwrap();
    testing::write_snapshot(snapshot.path(), &testing::mimo_pro_config(),
        &testing::mimo_pro_tensors(), Some(8));
    let manifest = snapshot.path().join("PROGRAMS.json");
    std::fs::write(&manifest, br#"{"capacities":{"prefill_rows":4096}}"#).unwrap();
    use cuteafd_loader::plan::{deployment::{DeploymentOptions, DeploymentSpec},
        layout::LayoutOptions, MediaMode, PlanInput, PlanOptions};
    let example = include_str!("../../../../examples/configs/mimo-v26-pro-mxfp4-max.config");
    let config = example.lines().filter_map(|line| line.split_once('=')).collect::<std::collections::BTreeMap<_, _>>();
    let gpus = config["RTX_GPUS"].parse::<usize>().unwrap();
    let ranks = config["SPARK_COUNT"].parse::<usize>().unwrap();
    let options = PlanOptions {
        vision: MediaMode::Off, audio: MediaMode::Off,
        layout: Some(LayoutOptions {
            host_memory: Some((128 << 30, 128 << 30)),
            rtx_bytes: vec![cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES; gpus],
            prefill_rows: 2752, workspace_manifest: Some(manifest), ..Default::default()
        }), ..Default::default()
    };
    let report = PlanInput::open(snapshot.path()).unwrap().plan_spark_count(&options, ranks, Some(ranks)).unwrap();
    let launch = DeploymentOptions { rtx_gpus: gpus, speculator: "off".into(), ..Default::default() };
    let deployment = release_accepts_prefill(&serde_json::to_vec(
        &DeploymentSpec::from_report(&report, &launch).unwrap()).unwrap());
    assert_eq!(deployment["family"], "mimo_v2");
    assert_eq!(deployment["rtx_gpus"], 2);
    assert_eq!(deployment["spark_ranks"], 6);
    assert_eq!(deployment["prefill_unit_rows"], 2752);
    assert_eq!(deployment["prefill_waves"], 3);
    assert!(deployment.get("prefill_slots").is_none());
}

#[test]
fn glm53_max_deployment_runs_three_2752_row_units() {
    use cuteafd_loader::plan::testing;
    let snapshot = tempfile::tempdir().unwrap();
    testing::write_snapshot(snapshot.path(), &testing::glm5_config(),
        &testing::glm5_tensors(|name, n, k| testing::fp8(name, n, k, None)), None);
    let manifest = snapshot.path().join("PROGRAMS.json");
    std::fs::write(&manifest, testing::glm5_native_programs().to_string()).unwrap();
    use cuteafd_loader::plan::{deployment::{DeploymentOptions, DeploymentSpec},
        layout::LayoutOptions, MediaMode, PlanInput, PlanOptions};
    let input = PlanInput::open(snapshot.path()).unwrap();
    let mut options = PlanOptions {
        vision: MediaMode::Off, audio: MediaMode::Off,
        layout: Some(LayoutOptions {
            // Shape contracts must not depend on the test runner's live cgroup.
            host_memory: Some((128 << 30, 128 << 30)),
            rtx_bytes: vec![cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES; 2],
            head_split: true, workspace_manifest: Some(manifest), ..Default::default()
        }), ..Default::default()
    };
    let launch = DeploymentOptions { rtx_gpus: 2, speculator: "off".into(), ..Default::default() };
    let report = input.plan_spark_count(&options, 6, Some(6)).unwrap();
    assert!(report.executable(), "{}", cuteafd_loader::plan::render(&report));
    let deployment = release_accepts_prefill(&serde_json::to_vec(
        &DeploymentSpec::from_report(&report, &launch).unwrap()).unwrap());
    // The fixed depth: three units of the 8192-row reserve.
    assert_eq!(deployment["prefill_unit_rows"], 2752);
    assert_eq!(deployment["prefill_waves"], 3);
    assert_eq!(deployment["prefill_span_rows"], 8256);

    options.layout.as_mut().unwrap().host_memory = Some((20 << 30, 15 << 30));
    let capped = input.plan_spark_count(&options, 6, Some(6)).unwrap();
    // Whole-process load RSS plus physical headroom cannot fit, even with tiny
    // prefill units. The former workspace-only host ledger missed this owner.
    assert!(!capped.executable());
    assert!(capped.hints.iter().any(|hint| hint.what.contains("coordinator host RAM")
        && hint.what.contains("checkpoint loading")), "{:?}", capped.hints);
    assert!(!DeploymentSpec::from_report(&capped, &launch).unwrap().executable);

    options.layout.as_mut().unwrap().prefill_rows = 2752;
    let refused = input.plan_spark_count(&options, 6, Some(6)).unwrap();
    assert_eq!(refused.prefill_unit_rows, Some(2752));
    assert!(!refused.executable());
    assert!(refused.hints.iter().any(|hint| hint.what.contains("coordinator host RAM")));
}

fn installed_example_snapshot(model_id: &str, revision: Option<&str>, hf_home: &std::path::Path)
    -> Option<std::path::PathBuf> {
    let snapshots = cuteafd_loader::model_cache_dir(hf_home, model_id).join("snapshots");
    if !snapshots.is_dir() || revision.is_some_and(|revision| !snapshots.join(revision).is_dir()) {
        return None;
    }
    cuteafd_loader::resolve_snapshot_at_revision(model_id, Some(hf_home), revision)
        .unwrap_or_else(|error| panic!("{model_id}: {error:#}")).snapshot_path
}

#[test]
fn installed_example_inventory_detects_absent_checkpoints_and_revisions() {
    let home = tempfile::tempdir().unwrap();
    let model = "test/installed";
    assert!(installed_example_snapshot(model, None, home.path()).is_none());
    assert!(installed_example_snapshot(model, Some("missing"), home.path()).is_none());
    let cache = cuteafd_loader::model_cache_dir(home.path(), model);
    let snapshot = cache.join("snapshots/present");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(cache.join("refs")).unwrap();
    std::fs::write(cache.join("refs/main"), "present\n").unwrap();
    assert_eq!(installed_example_snapshot(model, None, home.path()), Some(snapshot.clone()));
    assert_eq!(installed_example_snapshot(model, Some("present"), home.path()), Some(snapshot));
    assert!(installed_example_snapshot(model, Some("missing"), home.path()).is_none());
    std::fs::write(cache.join("refs/main"), "missing\n").unwrap();
    assert!(std::panic::catch_unwind(|| installed_example_snapshot(model, None, home.path())).is_err(),
        "a broken installed main ref is not an absent-checkpoint skip");
}

fn installed_example_programs(override_path: Option<std::path::PathBuf>, home: &std::path::Path) -> std::path::PathBuf {
    let manifest = override_path.unwrap_or_else(|| home.join(".cache/cuteafd/builds/placement-inputs/current/PROGRAMS.json"));
    cuteafd_loader::plan::inputs::validate_placement_inputs(&manifest)
        .unwrap_or_else(|error| panic!("{error:#}"));
    manifest
}

#[test]
fn installed_example_programs_fail_on_missing_override_or_published_default() {
    let home = tempfile::tempdir().unwrap();
    let absent = home.path().join("absent/PROGRAMS.json");
    for path in [None, Some(absent.clone())] {
        let failed = std::panic::catch_unwind(|| installed_example_programs(path, home.path()));
        let error = failed.unwrap_err();
        let message = error.downcast_ref::<String>().unwrap();
        assert!(message.contains("PROGRAMS.json missing"), "{message}");
        assert!(message.contains("export-placement-inputs.py"), "{message}");
    }
    std::fs::create_dir_all(absent.parent().unwrap()).unwrap();
    std::fs::write(&absent, r#"{"schema":1,"programs":[]}"#).unwrap();
    let failed = std::panic::catch_unwind(|| installed_example_programs(Some(absent.clone()), home.path()));
    let error = failed.unwrap_err();
    assert!(error.downcast_ref::<String>().unwrap().contains("missing placement provenance"));
}

fn require_installed_hf_home(test_name: &str, hf_home: Option<std::ffi::OsString>) -> std::path::PathBuf {
    let Some(hf_home) = hf_home.filter(|value| !value.is_empty()).map(std::path::PathBuf::from) else {
        panic!("{test_name}: CUTEAFD_DEPLOYMENT_TEST_HF_HOME is required (use --skip-live-planners only for an explicit opt-out)");
    };
    assert!(hf_home.is_dir(), "{test_name}: CUTEAFD_DEPLOYMENT_TEST_HF_HOME is not a directory: {}",
        hf_home.display());
    hf_home
}

fn require_installed_snapshot(scope: &str, model_id: &str, revision: Option<&str>,
    hf_home: &std::path::Path) -> std::path::PathBuf {
    installed_example_snapshot(model_id, revision, hf_home).unwrap_or_else(|| panic!(
        "{scope}: required checkpoint not installed in {}: {model_id}@{}",
        hf_home.display(), revision.unwrap_or("main")))
}

fn run_installed_glm_exlr8_worker_budget(hf_home: Option<std::ffi::OsString>,
    programs_override: Option<std::path::PathBuf>, home: &std::path::Path,
    root: &std::path::Path) {
    let hf_home = require_installed_hf_home(
        "installed_glm_exlr8_min_max_plans_pass_launcher_worker_budget", hf_home);
    let programs = installed_example_programs(programs_override, home);
    let mut tested = 0;
    for (name, gpus, ranks) in [("min", "1", 4), ("max", "2", 6)] {
        let example = root.join(format!("examples/configs/glm53-exlr8-{name}.config"));
        let raw = std::fs::read_to_string(&example).unwrap();
        let config = raw.lines().filter_map(|line| line.split_once('='))
            .collect::<std::collections::BTreeMap<_, _>>();
        let snapshot = require_installed_snapshot(
            name, config["MODEL_ID"], Some(config["MODEL_REVISION"]), &hf_home);
        for explicit in [None, Some("104")] {
            let mut command = Command::new("timeout");
            command.args(["30", env!("CARGO_BIN_EXE_cuteafd"), "plan"])
                .arg(&snapshot).args(["--deployment", "--config"]).arg(&example)
                .args(["--rtx", gpus, "--spark-ranks", &ranks.to_string(),
                    "--vision", "off", "--audio", "off", "--speculator", "off",
                    "--exlr8-manifest", config["EXLR8_MANIFEST"]])
                .env_remove("COORDINATOR_GPU_BUDGET_GIB");
            command.arg("--workspace-manifest").arg(&programs);
            if let Some(budget) = explicit { command.args(["--spark-budget-gib", budget]); }
            let start = std::time::Instant::now();
            let output = command.output().unwrap();
            let elapsed = start.elapsed();
            eprintln!("installed plan: glm53-exlr8-{name} budget={explicit:?} wall_seconds={:.3}", elapsed.as_secs_f64());
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(output.status.success(), "{name}, budget={explicit:?}: {stderr}");
            assert!(elapsed < std::time::Duration::from_secs(30), "{name}: plan took {elapsed:?}");
            let deployment: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(deployment["spark_ranks"], ranks);
            let devices = deployment["memory_layout"]["devices"].as_array().unwrap().iter()
                .filter(|device| device["kind"] == "spark").collect::<Vec<_>>();
            assert_eq!(devices.len(), ranks as usize);
            let expected = explicit.map_or(108u64, |_| 104) << 30;
            for (rank, device) in devices.iter().enumerate() {
                assert_eq!(device["index"], rank);
                assert_eq!(device["worker_budget_bytes"], expected);
                assert_eq!(device["capacity_bytes"], expected);
                assert_eq!(device["items"][0]["bytes"],
                    deployment["spark_requirements"]["resident_bytes_per_rank"][rank]);
            }
            let temporary = tempfile::tempdir().unwrap();
            let path = temporary.path().join("deployment.json");
            std::fs::write(&path, &output.stdout).unwrap();
            let interpreter = std::env::var_os("CUTEAFD_TEST_PYTHON")
                .expect("CUTEAFD_TEST_PYTHON must name the gate's venv Python for launcher checks");
            let python = Command::new(interpreter).args(["-I", "-c", r#"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts/tests'))
from test_run_family_dispatch import capture_native_exlr8_launch
capture_native_exlr8_launch(*map(Path, sys.argv[2:]))
"#]).arg(root).arg(&snapshot).arg(&path).arg(temporary.path()).output().unwrap();
            assert!(python.status.success(), "{name}: {}", String::from_utf8_lossy(&python.stderr));
            tested += 1;
        }
    }
    assert_eq!(tested, 4);
}

#[test]
#[ignore = "requires CUTEAFD_EXLR8_FLASH_SNAPSHOT and current CUTEAFD_DEPLOYMENT_TEST_PROGRAMS"]
fn installed_flash_exlr8_bounded_spark_plans_preserve_worker_budget_and_capabilities() {
    let snapshot = std::path::PathBuf::from(std::env::var_os("CUTEAFD_EXLR8_FLASH_SNAPSHOT")
        .expect("explicit Flash snapshot required"));
    let programs = std::path::PathBuf::from(std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS")
        .expect("explicit current exporter manifest required"));
    assert!(snapshot.join("config.json").is_file() && programs.is_file());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    for (name, gpus, ranks) in [("bounded-1rtx-4spark", "1", 4), ("bounded-2rtx-6spark", "2", 6)] {
        for explicit in [None, Some("104")] {
            let mut command = Command::new("timeout");
            command.args(["30", env!("CARGO_BIN_EXE_cuteafd"), "plan"]).arg(&snapshot)
                // Keep a Spark suffix for this wire-budget proof; automatic TP2 can onboard every layer.
                .args(["--deployment", "--rtx", gpus, "--spark-ranks", &ranks.to_string(),
                    "--rtx-expert-layers", "1", "--vision", "off", "--audio", "off", "--speculator", "off", "--exlr8-manifest", "k4",
                    "--workspace-manifest"]).arg(&programs).env_remove("COORDINATOR_GPU_BUDGET_GIB");
            if let Some(budget) = explicit { command.args(["--spark-budget-gib", budget]); }
            let output = command.output().unwrap();
            assert!(output.status.success(), "{name}, budget={explicit:?}: {}", String::from_utf8_lossy(&output.stderr));
            let deployment: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(deployment["family"], "glm5_flash");
            assert_eq!(deployment["exlr8"]["manifest"], "k4");
            assert_eq!(deployment["wip_launchable"], true);
            assert_eq!(deployment["report"]["runtime"], "serving");
            assert_eq!(deployment["report"]["placement_supported"], true);
            assert_eq!(deployment["spark_ranks"], ranks);
            let devices = deployment["memory_layout"]["devices"].as_array().unwrap().iter()
                .filter(|device| device["kind"] == "spark").collect::<Vec<_>>();
            assert_eq!(devices.len(), ranks as usize);
            let expected = explicit.map_or(108u64, |_| 104) << 30;
            for (rank, device) in devices.iter().enumerate() {
                assert_eq!(device["index"], rank);
                assert_eq!(device["worker_budget_bytes"], expected);
                assert_eq!(device["capacity_bytes"], expected);
                assert_eq!(device["items"][0]["bytes"], deployment["spark_requirements"]["resident_bytes_per_rank"][rank]);
            }
            let temporary = tempfile::tempdir().unwrap();
            let path = temporary.path().join("flash-deployment.json");
            std::fs::write(&path, &output.stdout).unwrap();
            let interpreter = std::env::var_os("CUTEAFD_TEST_PYTHON")
                .expect("CUTEAFD_TEST_PYTHON must name the gate's venv Python for launcher checks");
            let python = Command::new(interpreter).args(["-I", "-c", r#"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts/tests'))
from test_run_family_dispatch import capture_native_exlr8_capability_deployment
capture_native_exlr8_capability_deployment(Path(sys.argv[2]))
"#]).arg(&root).arg(&path).output().unwrap();
            assert!(python.status.success(), "{name}: {}", String::from_utf8_lossy(&python.stderr));
        }
    }
}

fn installed_flash_exlr8_automatic_tp2_plan(ranks: &str) -> serde_json::Value {
    let snapshot = std::path::PathBuf::from(std::env::var_os("CUTEAFD_EXLR8_FLASH_SNAPSHOT")
        .expect("explicit Flash snapshot required"));
    let programs = std::path::PathBuf::from(std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS")
        .expect("explicit current exporter manifest required"));
    assert!(snapshot.join("config.json").is_file() && programs.is_file());
    let output = Command::new("timeout").args(["30", env!("CARGO_BIN_EXE_cuteafd"), "plan"])
        .arg(&snapshot).args(["--deployment", "--rtx", "2", "--spark-ranks", ranks,
            "--vision", "off", "--audio", "off", "--speculator", "off", "--exlr8-manifest", "k4",
            "--workspace-manifest"]).arg(&programs).env_remove("COORDINATOR_GPU_BUDGET_GIB")
        .output().unwrap();
    assert!(output.status.success(), "requested ranks={ranks}: {}", String::from_utf8_lossy(&output.stderr));
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
#[ignore = "requires CUTEAFD_EXLR8_FLASH_SNAPSHOT and current CUTEAFD_DEPLOYMENT_TEST_PROGRAMS"]
fn installed_flash_exlr8_automatic_zero_spark_pool_stops_at_2m() {
    let deployment = installed_flash_exlr8_automatic_tp2_plan("0");
    let devices = deployment["memory_layout"]["devices"].as_array().unwrap();
    assert_eq!(devices.len(), 2);
    for device in devices {
        assert_eq!(device["kv_tokens"], 2_097_152,
            "automatic zero-Spark pool must stop at its 2M target when capacity permits");
    }
}

#[test]
#[ignore = "requires CUTEAFD_EXLR8_FLASH_SNAPSHOT and current CUTEAFD_DEPLOYMENT_TEST_PROGRAMS"]
fn installed_flash_exlr8_automatic_tp2_max_is_zero_spark_and_launchable() {
    // A requested Spark ceiling must not attach idle workers when all routed layers fit locally.
    for ranks in ["0", "6"] {
        let deployment = installed_flash_exlr8_automatic_tp2_plan(ranks);
        assert_eq!(deployment["family"], "glm5_flash");
        assert_eq!(deployment["exlr8"]["manifest"], "k4");
        assert_eq!(deployment["spark_ranks"], 0);
        assert_eq!(deployment["spark_requirements"]["resident_bytes_per_rank"], serde_json::json!([]));
        assert_eq!(deployment["report"]["fits"], true);
        assert_eq!(deployment["report"]["runtime"], "serving");
        assert_eq!(deployment["report"]["placement_supported"], true);
        assert_eq!(deployment["executable"], true);
        assert_eq!(deployment["wip_launchable"], true);
        let devices = deployment["memory_layout"]["devices"].as_array().unwrap();
        assert_eq!(devices.len(), 2);
        for (rank, device) in devices.iter().enumerate() {
            assert_eq!(device["kind"], "rtx");
            assert_eq!(device["index"], rank);
            assert_eq!(device["kv_tokens"], 2_097_152);
            let items = device["items"].as_array().unwrap();
            let routed = items.iter().find(|item| item["group"] == "TP2 expert layer halves 3..45")
                .expect("all 42 routed layer halves must be resident on each RTX");
            assert!(routed["bytes"].as_u64().unwrap() > 70u64 << 30);
            let resident = items.iter().map(|item| item["bytes"].as_u64().unwrap()).sum::<u64>();
            assert!(resident <= device["capacity_bytes"].as_u64().unwrap());
        }
    }
}

#[test]
fn installed_glm_exlr8_min_max_plans_pass_launcher_worker_budget() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    run_installed_glm_exlr8_worker_budget(
        std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME"),
        std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS").map(Into::into),
        &std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME")),
        &root);
}

/// Drive one shipped config through the real launcher and the real planner;
/// only hardware and payload execution are stubbed.
fn installed_real_planner_launch(root: &std::path::Path, capture: &str, snapshot: &std::path::Path,
    example: &std::path::Path, programs: &std::path::Path, hf_home: &std::path::Path) {
    let temporary = tempfile::tempdir().unwrap();
    let interpreter = std::env::var_os("CUTEAFD_TEST_PYTHON")
        .expect("CUTEAFD_TEST_PYTHON must name the gate's venv Python for launcher checks");
    let output = Command::new(interpreter).args(["-I", "-c", r#"
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts/tests'))
getattr(importlib.import_module('test_run_family_dispatch'), sys.argv[2])(*map(Path, sys.argv[3:]))
"#]).arg(root).arg(capture).arg(snapshot).arg(example).arg(env!("CARGO_BIN_EXE_cuteafd"))
        .arg(programs).arg(hf_home).arg(temporary.path()).output().unwrap();
    assert!(output.status.success(), "{} real launcher: {}", example.display(),
        String::from_utf8_lossy(&output.stderr));
}

/// The successful min-card smoke's native package must keep the same local prefix.
/// Called by the full-tier installed-example sweep, never an ignored/optional test.
fn installed_mimo_min_onboarding_matches_smoke(root: &std::path::Path,
    snapshot: &std::path::Path, example: &std::path::Path, programs: &std::path::Path) {
    let manifest = root.join("rust/crates/cuteafd-loader/tests/fixtures/mimo-flash-min-native-manifest.json");
    assert!(manifest.is_file(), "required MiMo native manifest fixture missing: {}", manifest.display());
    let output = Command::new("timeout").args(["30", env!("CARGO_BIN_EXE_cuteafd"), "plan"])
        .arg(snapshot).args(["--deployment", "--config"]).arg(example)
        .args(["--rtx", "1", "--spark-ranks", "2", "--pool-tokens", "0", "--native-mtp-layers", "0",
            "--speculator", "dflash2", "--vision", "auto", "--audio", "auto", "--workspace-manifest"])
        .arg(programs).arg("--mimo-expert-manifest").arg(manifest)
        .env_remove("COORDINATOR_GPU_BUDGET_GIB").output().unwrap();
    assert!(output.status.success(), "MiMo min installed native plan: {}", String::from_utf8_lossy(&output.stderr));
    let deployment: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(deployment["report"]["fits"], true, "{deployment}");
    assert_eq!(deployment["report"]["onboard_layers"], 15, "measured min-card local prefix must not drift");
    assert_eq!(deployment["memory_layout"]["pool_tokens"], 2_097_152);
    assert_eq!(deployment["prefill_unit_rows"], 2752);
    assert_eq!(deployment["prefill_waves"], 3);
    let lead = deployment["memory_layout"]["devices"].as_array().unwrap().iter()
        .find(|device| device["kind"] == "rtx" && device["index"] == 0).unwrap();
    let admitted = lead["items"].as_array().unwrap().iter()
        .map(|item| item["bytes"].as_u64().unwrap()).sum::<u64>();
    assert!(admitted >= 95_762_907_136, "admission must cover measured corrected peak");
    assert!(admitted <= lead["capacity_bytes"].as_u64().unwrap());
}

fn run_installed_example_deployments(hf_home: Option<std::ffi::OsString>,
    programs_override: Option<std::path::PathBuf>, home: &std::path::Path,
    root: &std::path::Path) {
    let hf_home = require_installed_hf_home(
        "installed_example_deployments_publish_executable_prefill_shapes", hf_home);
    let programs = installed_example_programs(programs_override, home);
    eprintln!("installed example PROGRAMS: {}", programs.display());
    let matrix: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("benchmarks/current-matrix.json")).unwrap()).unwrap();
    let retired = retired_configs(&matrix);
    let mut examples = std::fs::read_dir(root.join("examples/configs")).unwrap()
        .map(|entry| entry.unwrap().path()).filter(|path|
            path.extension().is_some_and(|extension| extension == "config")
            && !retired.contains(path.strip_prefix(root).unwrap().to_str().unwrap()))
        .collect::<Vec<_>>();
    examples.sort();
    assert!(!examples.is_empty(), "no example configs found");
    let total = examples.len();
    let mut tested = 0;
    let mut families = std::collections::BTreeSet::new();
    for example in examples {
        let raw = std::fs::read_to_string(&example).unwrap();
        let config = raw.lines().filter_map(|line| line.split_once('='))
            .collect::<std::collections::BTreeMap<_, _>>();
        let name = example.file_name().unwrap().to_str().unwrap();
        let revision = config.get("MODEL_REVISION").copied();
        let snapshot = require_installed_snapshot(name, config["MODEL_ID"], revision, &hf_home);
        let capture = [
            (&["glm53-exl3-min.config", "glm53-exl3-max.config",
              "glm53-exlr8-min.config", "glm53-exlr8-max.config"][..], "capture_glm_real_planner_launch"),
            (&["qwen35-9b-bf16-min.config", "qwen38-27b-bf16-min.config",
              "qwen38-27b-fp8-min.config"][..], "capture_qwen_real_planner_launch"),
            (&["mimo-v26-flash-mxfp4-min.config", "mimo-v26-flash-mxfp4-max.config",
              "mimo-v26-pro-mxfp4-min.config", "mimo-v26-pro-mxfp4-max.config"][..], "capture_mimo_real_planner_launch"),
        ].into_iter().find_map(|(names, capture)| names.contains(&name).then_some(capture));
        if let Some(capture) = capture {
            installed_real_planner_launch(root, capture, &snapshot, &example, &programs, &hf_home);
        }
        if name == "mimo-v26-flash-mxfp4-min.config" {
            installed_mimo_min_onboarding_matches_smoke(root, &snapshot, &example, &programs);
        }
        if name.starts_with("kimi-k3-") {
            let temporary = tempfile::tempdir().unwrap();
            let interpreter = std::env::var_os("CUTEAFD_TEST_PYTHON")
                .expect("CUTEAFD_TEST_PYTHON must name the gate's venv Python for launcher checks");
            let output = Command::new(interpreter).args(["-I", "-c", r#"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts/tests'))
from test_run_family_dispatch import capture_kimi_real_planner_launch
capture_kimi_real_planner_launch(*map(Path, sys.argv[2:]))
"#]).arg(root).arg(&snapshot).arg(&example).arg(env!("CARGO_BIN_EXE_cuteafd"))
                .arg(&programs).arg(&hf_home).arg(temporary.path()).output().unwrap();
            assert!(output.status.success(), "{name} real launcher: {}", String::from_utf8_lossy(&output.stderr));
        }
        let mut command = Command::new("timeout");
        command.args(["30", env!("CARGO_BIN_EXE_cuteafd"), "plan", config["MODEL_ID"],
            "--deployment", "--config"]).arg(&example)
            .args(["--rtx", config["RTX_GPUS"], "--vision", "off", "--audio", "off",
                "--speculator", "off", "--hf-home"]).arg(&hf_home)
            .env("HF_HOME", &hf_home).env_remove("COORDINATOR_GPU_BUDGET_GIB");
        if let Some(revision) = config.get("MODEL_REVISION") {
            command.args(["--revision", revision]);
        }
        for (key, flag) in [("SPARK_TP", "--spark-tp"), ("SPARK_EP", "--spark-ep"),
            ("PREFILL_ROWS", "--prefill-rows"), ("PREFILL_BATCH_TOKENS", "--prefill-rows"),
            ("PREFILL_WAVES", "--prefill-waves"), ("PREFILL_SPAN_ROWS", "--prefill-span-rows"),
            ("EXLR8_MANIFEST", "--exlr8-manifest")] {
            if let Some(value) = config.get(key) { command.args([flag, value]); }
        }
        command.arg("--workspace-manifest").arg(&programs);
        let start = std::time::Instant::now();
        let output = command.output().unwrap();
        let elapsed = start.elapsed();
        eprintln!("installed plan: {name} wall_seconds={:.3}", elapsed.as_secs_f64());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{name}: {stderr}");
        assert!(elapsed < std::time::Duration::from_secs(30), "{name}: plan took {elapsed:?}");
        let deployment = release_accepts_prefill(&output.stdout);
        tested += 1;
        families.insert(deployment["family"].as_str().unwrap().to_owned());
        eprintln!("{name}: {} unit_rows={} waves={} slots={} span_rows={}", deployment["family"],
            deployment["prefill_unit_rows"], deployment["prefill_waves"], deployment["prefill_slots"],
            deployment["prefill_span_rows"]);
    }
    assert_eq!(tested, total);
    eprintln!("installed example deployments: {tested} tested, {total} configs, families={families:?}");
}

#[test]
fn installed_example_deployments_publish_executable_prefill_shapes() {
    // cpu-gate --full enables the installed inventory; sparse fixtures always run.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    run_installed_example_deployments(
        std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME"),
        std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS").map(Into::into),
        &std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME")),
        &root);
}

fn retired_configs(matrix: &serde_json::Value) -> std::collections::BTreeSet<String> {
    matrix["retired"].as_array().unwrap().iter()
        .map(|entry| entry["config"].as_str().unwrap().to_owned()).collect()
}

fn live_release_entries(root: &std::path::Path, matrix: &serde_json::Value) -> Vec<serde_json::Value> {
    let mut entries = Vec::new();
    for cell in matrix["cells"].as_array().unwrap() {
        for column in ["5090", "1rtx", "2rtx"] {
            let card = &cell[column];
            if card["status"] != "current" { continue; }
            let report_path = root.join(card["report"].as_str().unwrap()).join("report.json");
            let report: serde_json::Value = serde_json::from_slice(&std::fs::read(&report_path)
                .unwrap_or_else(|error| panic!("{}: {error}", report_path.display()))).unwrap();
            let revision = report["server"]["revision"].as_str().filter(|revision| !revision.is_empty())
                .unwrap_or_else(|| panic!("{}: required pinned revision missing", report_path.display()));
            let ranks = report["server"]["hardware"]["sparks"].as_array().map_or(0, Vec::len);
            entries.push(serde_json::json!({"name": card["report"], "model": cell["checkpoint"],
                "revision": revision, "ranks": ranks}));
        }
    }
    assert!(!entries.is_empty(), "no live current-matrix release cells found");
    entries
}

fn run_installed_release_matrix_recipes(hf_home: Option<std::ffi::OsString>,
    root: &std::path::Path, matrix: &serde_json::Value) {
    let hf_home = require_installed_hf_home(
        "installed_release_matrix_recipe_ids_parse_from_checkpoint_headers", hf_home);
    let entries = live_release_entries(root, matrix);
    let mut tested = 0;
    for entry in &entries {
        let entry_name = entry["name"].as_str().unwrap();
        let snapshot = require_installed_snapshot(entry_name, entry["model"].as_str().unwrap(),
            Some(entry["revision"].as_str().unwrap()), &hf_home);
        let ranks = entry["ranks"].as_u64().unwrap() as usize;
        use cuteafd_loader::plan::{family, format, spec::Component, Checkpoint};
        // Exercise production recipe construction from real tensor headers without
        // running the unrelated memory-placement search for every matrix entry.
        let checkpoint = Checkpoint::open(&snapshot).unwrap();
        let family = family::detect(&checkpoint).unwrap();
        let model = family.open(&checkpoint).unwrap();
        tested += 1;
        let Some(moe) = model.spec().moe.as_ref() else { continue; };
        let tensors = checkpoint.tensors.iter().filter(|tensor|
            family.classify(model.spec(), &tensor.meta.name)
                .is_some_and(|role| role.component == Component::RoutedExpert))
            .cloned().collect::<Vec<_>>();
        let mut packages = std::collections::BTreeSet::new();
        for members in format::group_by_stem(&tensors).values() {
            let operand = format::detect(members).unwrap();
            if let Some(contract) = model.experts(&operand) { packages.insert(contract.package); }
        }
        if cuteafd_loader::plan::exlr8::recognizes(&snapshot).unwrap() {
            // EXLR8 is admitted by exlr8_native_plan, not spark_recipe.
            assert!(packages.is_empty(), "{entry_name}: EXLR8 unexpectedly selected a generic package");
            continue;
        }
        assert!(!packages.is_empty(), "{entry_name}: missing routed recipe");
        for package in packages {
            // Local-only coordinators need no Spark TP1 package, but their IDs
            // must parse against a registered Spark world.
            let world = if ranks == 0 {
                *cuteafd_loader::plan::experts::spark_worlds(&package, moe.intermediate).first()
                    .unwrap_or_else(|| panic!("{entry_name}: unrecognized recipe {package}"))
            } else { ranks };
            cuteafd_loader::plan::experts::spark_recipe(&package, moe.intermediate, world)
                .unwrap_or_else(|error| panic!("{entry_name}: {package}: {error}"));
        }
    }
    assert_eq!(tested, entries.len());
}

#[test]
fn installed_release_matrix_recipe_ids_parse_from_checkpoint_headers() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let matrix: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../benchmarks/current-matrix.json")).unwrap();
    run_installed_release_matrix_recipes(
        std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME"), &root, &matrix);
}

#[test]
fn installed_inventory_filters_retired_examples_but_keeps_live_release_cells() {
    let root = tempfile::tempdir().unwrap();
    let report = root.path().join("benchmarks/live");
    std::fs::create_dir_all(&report).unwrap();
    std::fs::write(report.join("report.json"), serde_json::json!({"server": {
        "revision": "pinned", "hardware": {"sparks": [{}, {}]}}}).to_string()).unwrap();
    let matrix = serde_json::json!({"retired": [{"config": "examples/configs/old.config"}], "cells": [{
        "checkpoint": "test/current", "1rtx": {"status": "current", "report": "benchmarks/live"},
        "5090": {"status": "na"}, "2rtx": {"status": "retired", "report": "missing"}}]});
    let retired = retired_configs(&matrix);
    assert!(retired.contains("examples/configs/old.config"));
    assert!(!retired.contains("examples/configs/live.config"));
    let entries = live_release_entries(root.path(), &matrix);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["model"], "test/current");
    assert_eq!(entries[0]["revision"], "pinned");
    assert_eq!(entries[0]["ranks"], 2);
    std::fs::write(report.join("report.json"), r#"{"server":{"revision":""}}"#).unwrap();
    assert!(std::panic::catch_unwind(|| live_release_entries(root.path(), &matrix)).is_err());
}

#[test]
fn installed_test_entrypoints_fail_instead_of_skipping_absent_inputs() {
    let home = tempfile::tempdir().unwrap();
    let tests = ["installed_glm_exlr8_min_max_plans_pass_launcher_worker_budget",
        "installed_example_deployments_publish_executable_prefill_shapes",
        "installed_release_matrix_recipe_ids_parse_from_checkpoint_headers"];
    for name in tests {
        for input in ["hf-home", "programs-or-checkpoint"] {
            let mut command = Command::new(std::env::current_exe().unwrap());
            command.args(["--exact", name, "--nocapture"])
                .env_remove("CUTEAFD_DEPLOYMENT_TEST_HF_HOME")
                .env("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS", home.path().join("missing/PROGRAMS.json"));
            if input != "hf-home" { command.env("CUTEAFD_DEPLOYMENT_TEST_HF_HOME", home.path()); }
            let output = command.output().unwrap();
            assert!(!output.status.success(), "{name}: missing {input} was silently skipped");
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.contains("0 passed; 1 failed"), "{stdout}");
            let expected = if input == "hf-home" { "CUTEAFD_DEPLOYMENT_TEST_HF_HOME is required" }
                else if name.starts_with("installed_release_") { "required checkpoint not installed" }
                else { "required PROGRAMS.json missing" };
            assert!(stderr.contains(expected), "{name}: {stderr}");
        }
    }
}

#[test]
fn installed_checkpoint_runners_fail_on_missing_hf_home_checkpoints_or_programs() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let matrix = serde_json::json!({"cells": [], "retired": []});
    for missing_hf in [None, Some( std::ffi::OsString::from("") ),
        Some(temp.path().join("missing-hf").into_os_string())] {
        assert!(std::panic::catch_unwind(|| run_installed_glm_exlr8_worker_budget(
            missing_hf.clone(), None, temp.path(), &root)).is_err());
        assert!(std::panic::catch_unwind(|| run_installed_example_deployments(
            missing_hf.clone(), None, temp.path(), &root)).is_err());
        assert!(std::panic::catch_unwind(|| run_installed_release_matrix_recipes(
            missing_hf.clone(), &root, &matrix)).is_err());
    }
    let hf_home = temp.path().join("hf");
    std::fs::create_dir_all(&hf_home).unwrap();
    assert!(std::panic::catch_unwind(|| run_installed_glm_exlr8_worker_budget(
        Some(hf_home.clone().into_os_string()), None, temp.path(), &root)).is_err());
    assert!(std::panic::catch_unwind(|| run_installed_example_deployments(
        Some(hf_home.clone().into_os_string()), None, temp.path(), &root)).is_err());
    assert!(std::panic::catch_unwind(|| run_installed_release_matrix_recipes(
        Some(hf_home.clone().into_os_string()), &root, &matrix)).is_err());

}

#[test]
fn bench_panels_json_cli_matches_fixture_and_selectable_panel_ids() {
    let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
        .args(["bench", "panels", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let actual: Vec<String> = serde_json::from_slice(&output.stdout).unwrap();
    let fixture: Vec<String> = serde_json::from_str(
        include_str!("../../../../scripts/bench/fixtures/bench-panels-v1.json")).unwrap();
    assert_eq!(actual, fixture);
    assert!(actual.iter().any(|id| id == "shared_prefix"));
}

#[test]
#[ignore = "requires real CUTEAFD_KIMI_SNAPSHOT, CUTEAFD_KIMI_DRAFTER and CUTEAFD_DEPLOYMENT_TEST_PROGRAMS"]
fn installed_kimi_tp1_automatic_plan_finishes_within_ten_seconds() {
    let required = |key| std::env::var_os(key).unwrap_or_else(|| panic!("{key} required"));
    let start = std::time::Instant::now();
    let output = Command::new("timeout")
        .args(["30", env!("CARGO_BIN_EXE_cuteafd"), "plan"])
        .arg(required("CUTEAFD_KIMI_SNAPSHOT"))
        .args(["--deployment", "--rtx", "1", "--spark-ranks", "6", "--spark-budget-gib", "110",
            "--speculator", "dflash2", "--drafter-snapshot"])
        .arg(required("CUTEAFD_KIMI_DRAFTER"))
        .args(["--vision", "off", "--audio", "off", "--workspace-manifest"])
        .arg(required("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS"))
        .env_remove("COORDINATOR_GPU_BUDGET_GIB").output().unwrap();
    let elapsed = start.elapsed();
    assert!(output.status.success(), "{elapsed:?}: {}", String::from_utf8_lossy(&output.stderr));
    assert!(elapsed <= std::time::Duration::from_secs(10), "TP1 plan took {elapsed:?}");
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(plan["rtx_gpus"], 1);
    assert_eq!(plan["spark_ranks"], 6);
    assert_eq!(plan["speculator"], "dflash2");
    assert_eq!(plan["executable"], true, "{plan}");
    let placement = &plan["estimated_placement"];
    assert_eq!(placement["items"].as_array().unwrap().len(), 1);
    assert!(placement["hops"].as_array().unwrap().is_empty());
    for layer in placement["layers"].as_array().unwrap() {
        assert_eq!(layer["mode"]["mode"], "whole");
        assert_eq!(layer["mode"]["gpu"], 0);
        assert_eq!(layer["mode"]["ffn"], "owner");
    }
    eprintln!("real K6 TP1 automatic plan: {elapsed:?}");
}
