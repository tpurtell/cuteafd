"""run.sh hands every family but DeepSeek V4.1 to scripts/launch/run-family.sh."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import pytest
from launcher_sandbox import copy_input

ROOT = Path(__file__).resolve().parents[2]


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / "scripts" / "launch").mkdir(parents=True)
    copy_input(ROOT / "run.sh", repo / "run.sh")
    for name in ("release-common.sh", "checkpoint-family.py", "cdi-gpus.sh"):
        copy_input(ROOT / "scripts" / "lib" / name, repo / "scripts" / "lib" / name)
    fake = repo / "scripts" / "launch" / "run-family.sh"
    fake.write_text('#!/usr/bin/env bash\nprintf "run-family %s\\n" "$*"\n')
    fake.chmod(0o755)
    return repo


def _snapshot(hf: Path, model: str, config: dict, revision: str = "abc") -> None:
    root = hf / "hub" / f"models--{model.replace('/', '--')}"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(revision)
    (root / "snapshots" / revision).mkdir(parents=True)
    (root / "snapshots" / revision / "config.json").write_text(json.dumps(config))
    if config.get("audio_config"):
        audio = root / "snapshots" / revision / "audio_tokenizer"
        audio.mkdir()
        (audio / "config.json").write_text("{}")
        (audio / "model.safetensors").write_bytes(b"stub")


def _run(repo: Path, hf: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HF_HOME": str(hf)}
    return subprocess.run(["bash", str(repo / "run.sh"), *args], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=60)


def test_glm_flash_config_goes_to_run_family(tmp_path: Path) -> None:
    repo, hf = _repo(tmp_path), tmp_path / "hf"
    _snapshot(hf, "zai-org/GLM-5.3-Flash", {"model_type": "glm5_next", "num_hidden_layers": 45,
                                           "layer_types": ["linear_attention"] * 45,
                                           "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42})
    (repo / "glmf.config").write_text("MODEL_ID=zai-org/GLM-5.3-Flash\nSPECULATOR_MODEL_ID=incoai/x\n")
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--restart")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"run-family --config {repo / 'glmf.config'} --family glm5_flash --restart\n"
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--restart", "--all")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"run-family --config {repo / 'glmf.config'} --family glm5_flash --restart --all\n"
    # run-family.sh owns the option rules (--all needs --restart).
    real = subprocess.run(["bash", str(ROOT / "scripts/launch/run-family.sh"), "--config", str(repo / "glmf.config"), "--all"],
                          capture_output=True, text=True, env={**os.environ, "HF_HOME": str(hf)})
    assert real.returncode != 0 and "--all requires --restart" in real.stderr
    # The former DeepSeek V4.1 options are configuration keys now, for every family.
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--concurrency", "4")
    assert result.returncode != 0 and "use --set CONCURRENCY=VALUE" in result.stderr
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--set", "CONCURRENCY=4", "--restart")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"run-family --config {repo / 'glmf.config'} --family glm5_flash --set CONCURRENCY=4 --restart\n"
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--embedding-placement", "host")
    assert result.returncode == 0, result.stderr
    assert "--embedding-placement host" in result.stdout
    for backend in ("uring", "mincore-routed"):
        result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--table-backend", backend)
        assert result.returncode != 0
        assert "unknown run argument: --table-backend" in result.stderr
    # A ./wip.sh slot reaches run-family.sh, which serves it from the development images.
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--wip", "s1", "--restart")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"run-family --config {repo / 'glmf.config'} --family glm5_flash --wip s1 --restart\n"


def test_public_launcher_resolves_family_from_selected_namespace(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    namespace = tmp_path / "namespace"
    model = "local/flash"
    _snapshot(namespace / "hf-home", model, GLMF_TWO_LAYER)
    config = repo / "selected.config"
    config.write_text(f"MODEL_ID={model}\n")
    result = subprocess.run(["bash", str(repo / "run.sh"), "--config", str(config)],
        env={**os.environ, "HF_HOME": str(tmp_path / "missing"), "CUTEAFD_NAMESPACE_ROOT": str(namespace)},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--family glm5_flash" in result.stdout


def test_deepseek_v41_goes_to_run_family_like_every_family(tmp_path: Path) -> None:
    """run.sh is the same thin dispatcher for DeepSeek V4.1 (#82)."""
    repo, hf = _repo(tmp_path), tmp_path / "hf"
    _snapshot(hf, "deepseek-ai/DeepSeek-V4.1-Flash", {"model_type": "deepseek_v41"})
    (repo / "v41.config").write_text("MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash\n")
    result = _run(repo, hf, "--config", str(repo / "v41.config"), "--set", "DSPARK_DRAFT_LIMIT=7",
                  "--wip", "gate", "--restart")
    assert result.returncode == 0, result.stderr
    assert result.stdout == (f"run-family --config {repo / 'v41.config'} --family deepseek_v41 "
                             "--set DSPARK_DRAFT_LIMIT=7 --wip gate --restart\n")
    result = _run(repo, hf, "--config", str(repo / "v41.config"), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("--family deepseek_v41 --dry-run\n")
    run_sh = (ROOT / "run.sh").read_text()
    # No private launch path is left in run.sh.
    for private in ("serve-native", "expertd-native", "docker run", "release_load_config"):
        assert private not in run_sh, private


@pytest.mark.parametrize("value", [None, "on", "off"])
def test_shared_draft_policy_switches_are_explicit_and_default_off(tmp_path: Path, value) -> None:
    names = ["DRAFT_SURVIVAL_LENGTHS"]
    keys = "".join(f"{name}={value}\n" for name in names) if value else ""
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                                            "moe_layer_freq": [0, 1]}, "XiaomiMiMo/MiMo-V2-Flash",
                                  keys + "SPECULATION_TRACE=" + str(tmp_path / "trace.jsonl") + "\n")
    assert result.returncode == 0, result.stderr
    for name in names:
        assert f"CUTEAFD_{name}={int(value == 'on')}" in result.stderr
    assert "CUTEAFD_SPECULATION_TRACE=" in result.stderr


def test_glm53_bare_default_uses_release_dflash2(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3,
    }, "zai-org/GLM-5.3", "")
    assert result.returncode == 0, result.stderr
    assert "GLM 5.3 drafts with DFlash2" in result.stderr
    assert "models--incoai--GLM-5.3-DFlash2/snapshots/abc" in result.stderr


@pytest.mark.parametrize("key", ["COPY_DRAFTS", "COPY_DRAFT_POLICY", "CUTEAFD_COPY_DRAFT_POLICY",
    "DRAFT_CONFIDENCE", "DRAFT_COST_BUCKETS", "MIMO_COPY_WINDOWS", "GLM5_FLASH_DRAFT_POLICY",
    "DRAFT_MODEL_ID", "DRAFT_MODEL_REVISION", "DRAFT_FP8", "DFLASH", "MTP"])
@pytest.mark.parametrize("value", ["on", "off"])
def test_removed_draft_keys_are_unknown(tmp_path: Path, key: str, value: str) -> None:
    result = _family_launch_result(tmp_path, {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3,
    }, "zai-org/GLM-5.3", f"{key}={value}\n")
    assert result.returncode == 2, result.stderr
    assert f"unknown configuration key: {key}" in result.stderr
    assert "deprecated" not in result.stderr and "was removed" not in result.stderr
    assert "docker run -d" not in result.stderr


@pytest.mark.parametrize("key", ["COPY_DRAFTS", "COPY_DRAFT_POLICY", "DRAFT_CONFIDENCE", "DRAFT_COST_BUCKETS"])
@pytest.mark.parametrize("value", ["on", "off"])
def test_v4_rejects_removed_draft_switches_as_unknown_keys(tmp_path: Path, key: str, value: str) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v4"},
                                  "deepseek-ai/DeepSeek-V4-Flash-0731", f"{key}={value}\n")
    assert result.returncode != 0
    assert f"unknown configuration key: {key}" in result.stderr


@pytest.mark.parametrize("rows,waves", [(4096, 2), (2048, 1), (512, 3)])
def test_v4_prefill_units_reach_plan_serve_and_container(tmp_path: Path, rows: int, waves: int) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v4"}, "test/model",
        f"RTX_GPUS=1\nSPECULATOR=off\nVISION=off\nAUDIO=off\nPREFILL_ROWS={rows}\n"
        f"PREFILL_WAVES={waves}\nPREFILL_SPAN_ROWS=16384\n")
    assert result.returncode == 0, result.stderr
    plan = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    assert f"--prefill-rows {rows}" in plan
    assert f"--prefill-waves {waves}" in plan
    assert "--prefill-lanes" not in plan
    serve = next(line for line in result.stderr.splitlines() if "serve-dsv4" in line)
    assert f"--prefill-rows {rows}" in serve
    assert f"--prefill-waves {waves}" in serve
    assert "CUTEAFD_PREFILL_SPAN_ROWS=16384" in result.stderr
    assert "--prefill-lanes" not in serve


@pytest.mark.parametrize("value", [None, "auto", "1", "8", "64"])
@pytest.mark.parametrize("config,model", [
    ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}, "zai-org/GLM-5.3"),
    ({"model_type": "deepseek_v41"}, "deepseek-ai/DeepSeek-V4.1-Flash"),
])
def test_attention_plan_rows_reaches_only_the_coordinator(tmp_path: Path, value, config, model) -> None:
    keys = f"ATTENTION_PLAN_ROWS={value}\n" if value else ""
    if config["model_type"] == "deepseek_v41":
        keys += ("MODEL_REVISION=\nSPECULATOR=off\nVISION=off\nRTX_GPUS=1\n"
                 "RTX_EXPERT_LAYERS=0\nSPARK_COUNT=2\nSPARK_1_HOST=h1\n"
                 "SPARK_1_LANE_A=10.0.0.2\n")
    result = _family_launch_result(tmp_path, config, model, keys)
    assert result.returncode == 0, result.stderr
    launches = [line for line in result.stderr.splitlines() if "docker run -d" in line]
    if value in (None, "auto"):
        assert all("CUTEAFD_ATTENTION_PLAN_ROWS=" not in line for line in launches)
    else:
        assert any(f"CUTEAFD_ATTENTION_PLAN_ROWS={value}" in line for line in launches)
        assert all("CUTEAFD_ATTENTION_PLAN_ROWS=" not in line for line in launches
                   if "cuteafd-spark-expert" in line)


@pytest.mark.parametrize("value", ["0", "-1", "8.0", "maybe", "08"])
def test_attention_plan_rows_rejects_invalid_launcher_values(tmp_path: Path, value) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "glm_moe_dsa", "num_hidden_layers": 4,
                                            "first_k_dense_replace": 3}, "zai-org/GLM-5.3",
                                  f"ATTENTION_PLAN_ROWS={value}\n")
    assert result.returncode != 0
    assert "ATTENTION_PLAN_ROWS must be auto or a positive integer" in result.stderr


@pytest.mark.parametrize("value", [None, "on", "off"])
def test_thermal_report_key_defaults_on_and_reaches_the_coordinator(tmp_path: Path, value) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                                            "moe_layer_freq": [0, 1]}, "XiaomiMiMo/MiMo-V2-Flash",
                                  f"THERMAL={value}\n" if value else "")
    assert result.returncode == 0, result.stderr
    launches = [line for line in result.stderr.splitlines() if "docker run -d" in line]
    assert any("cuteafd-spark-expert" in line for line in launches)
    assert all(f"CUTEAFD_THERMAL={value or 'on'}" in line for line in launches), launches


def test_thermal_report_key_rejects_unknown_value(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                                            "moe_layer_freq": [0, 1]}, "XiaomiMiMo/MiMo-V2-Flash", "THERMAL=maybe\n")
    assert result.returncode != 0
    assert "THERMAL must be on or off" in result.stderr


def test_shared_draft_policy_rejects_unknown_switch_value(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                                            "moe_layer_freq": [0, 1]}, "XiaomiMiMo/MiMo-V2-Flash",
                                  "DRAFT_SURVIVAL_LENGTHS=maybe\n")
    assert result.returncode != 0
    assert "DRAFT_SURVIVAL_LENGTHS must be on or off" in result.stderr


@pytest.mark.parametrize("value", [None, "shared", "simple", "simple-rc", "simple-derived"])
def test_simple_draft_policy_is_forwarded_only_when_set(tmp_path: Path, value) -> None:
    keys = f"DRAFT_POLICY={value}\n" if value else ""
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                                            "moe_layer_freq": [0, 1]}, "XiaomiMiMo/MiMo-V2-Flash", keys)
    assert result.returncode == 0, result.stderr
    if value:
        assert f"CUTEAFD_DRAFT_POLICY={value}" in result.stderr
    else:
        assert "CUTEAFD_DRAFT_POLICY" not in result.stderr


@pytest.mark.parametrize("value", ["linear", "power", "log"])
def test_simple_draft_row_cost_is_forwarded(tmp_path: Path, value) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                                            "moe_layer_freq": [0, 1]}, "XiaomiMiMo/MiMo-V2-Flash",
                                  f"DRAFT_POLICY=simple-derived\nDRAFT_ROW_COST={value}\n")
    assert result.returncode == 0, result.stderr
    assert f"CUTEAFD_DRAFT_ROW_COST={value}" in result.stderr


def test_simple_draft_policy_rejects_unknown_value(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                                            "moe_layer_freq": [0, 1]}, "XiaomiMiMo/MiMo-V2-Flash",
                                  "DRAFT_POLICY=clever\n")
    assert result.returncode != 0
    assert "DRAFT_POLICY must be shared, simple, simple-rc or simple-derived" in result.stderr


def test_family_table_names_every_launchable_family() -> None:
    table = ROOT / "scripts" / "lib" / "checkpoint-family.py"
    cases = {
        "deepseek_v41": {"model_type": "deepseek_v41"},
        "deepseek_v4": {"model_type": "deepseek_v4"},
        "glm5": {"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3},
        "glm5_flash": {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
                       "layer_types": ["linear_attention", "deepseek_sparse_attention"]},
        "mimo_v2": {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
        "qwen4": {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
                                                             "layer_types": ["linear_attention", "full_attention"]}},
    }
    for family, config in cases.items():
        path = Path(os.environ.get("TMPDIR", "/tmp")) / f"family-{os.getpid()}-{family}.json"
        path.write_text(json.dumps(config))
        try:
            out = subprocess.run(["python3", str(table), str(path)], capture_output=True, text=True, check=True)
        finally:
            path.unlink()
        assert out.stdout.split()[0] == family


def test_glmf_solved_worker_startup_waits_bootstrap_by_default(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "SPECULATOR=off\nVISION=off\nAUDIO=off\n",
                                  extra_env={"STUB_WORKER_READY": "worker bootstrap ready"})
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    worker = next(line for line in lines if "cuteafd expertd-native" in line)
    coordinator = next(line for line in lines if "cuteafd" in line and "serve-glmf" in line)
    assert "--placement-handshake" not in worker
    assert "--placement-handshake" not in coordinator
    marker = "worker bootstrap ready"
    readiness = next(i for i, line in enumerate(lines) if f"grep -q '{marker}'" in line)
    assert readiness < lines.index(coordinator)
    assert not any("grep -q 'worker ready'" in line for line in lines)


DENSE_CONFIG = {"model_type": "qwen3_5", "text_config": {
    "model_type": "qwen3_5_text", "num_hidden_layers": 32,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 28,
}}


@pytest.mark.parametrize("model", ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8"])
@pytest.mark.parametrize("selection", [None, "off", "dspark"])
@pytest.mark.parametrize("revision", [None, "explicit-revision"])
def test_dense_27b_dspark_default_and_explicit_disable(tmp_path, model, selection, revision):
    config = {**DENSE_CONFIG, "text_config": {**DENSE_CONFIG["text_config"], "hidden_size": 5120}}
    drafter = "RadixArk/Qwen3.8-27B-DSpark"
    selected_revision = revision or "85ef153be924f17ce4bf62726954eeaa4a73e854"
    _snapshot(tmp_path / "hf", drafter, {"speculators_model_type": "dspark"}, selected_revision)
    # A moving main ref must not replace the qualified default or an explicit revision.
    (tmp_path / "hf" / "hub" / f"models--{drafter.replace('/', '--')}" / "refs" / "main").write_text("unavailable-main")
    keys = "SPARK_COUNT=0\n" + (f"SPECULATOR={selection}\n" if selection else "")
    if revision:
        keys += f"SPECULATOR_MODEL_REVISION={revision}\n"
    if selection == "dspark":
        keys += "SPECULATOR_DRAFTS=3\n"
    result = _family_launch_result(tmp_path, config, model, keys,
        extra_args=("--wip", "dense-test"),
        encoder_plan={"placement_supported": True, "fits": True, "spark_ranks": 0})
    assert result.returncode == 0, result.stderr
    launch = shlex.split(next(line for line in result.stderr.splitlines() if "serve-qwen35 --snapshot" in line))
    plan = shlex.split(next(line for line in result.stderr.splitlines() if "cuteafd plan " in line))
    assert ("--draft" in launch) == (selection != "off")
    assert ("--drafter-snapshot" in plan) == (selection != "off")
    if selection != "off":
        snapshot = str(tmp_path / "hf" / "hub" / f"models--{drafter.replace('/', '--')}" / "snapshots" / selected_revision)
        assert launch[launch.index("--draft") + 1] == snapshot
        assert plan[plan.index("--drafter-snapshot") + 1] == snapshot
    if selection == "dspark":
        assert launch[launch.index("--draft-fixed") + 1] == "3"


@pytest.mark.parametrize("model", ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8"])
@pytest.mark.parametrize("depth", [None, 1, 2, 3])
def test_dense_27b_native_mtp_needs_no_external_checkpoint(tmp_path, model, depth):
    config = {**DENSE_CONFIG, "text_config": {**DENSE_CONFIG["text_config"],
        "hidden_size": 5120, "mtp_num_hidden_layers": 1}}
    keys = "SPARK_COUNT=0\nSPECULATOR=mtp\n"
    if depth is not None:
        keys += f"SPECULATOR_DEPTH={depth}\n"
    result = _family_launch_result(tmp_path, config, model, keys,
        extra_args=("--wip", "dense-test"),
        encoder_plan={"placement_supported": True, "fits": True, "spark_ranks": 0})
    assert result.returncode == 0, result.stderr
    launch = shlex.split(next(line for line in result.stderr.splitlines() if "serve-qwen35 --snapshot" in line))
    plan = shlex.split(next(line for line in result.stderr.splitlines() if "cuteafd plan " in line))
    assert launch[launch.index("--mtp") + 1] == str(depth or 1)
    assert plan[plan.index("--speculator") + 1] == "mtp"
    assert "--draft" not in launch and "--drafter-snapshot" not in plan
    assert "cuteafd expertd-native" not in result.stderr


@pytest.mark.parametrize("cache_keys,entries,boundaries", [
    ("", 20, 1),
    ("PREFIX_CACHE_ENTRIES=4\nPREFIX_CACHE_MARK_MIB=1024\nPREFIX_POINT_BOUNDARIES=2\n", 4, 2),
    ("PREFIX_CACHE_ENTRIES=0\nPREFIX_POINT_BOUNDARIES=0\n", 0, 0),
])
@pytest.mark.parametrize("kv_format", ["bf16", "fp8"])
def test_dense_qwen_wip_uses_native_single_gpu_and_no_workers(tmp_path: Path, cache_keys, entries, boundaries, kv_format) -> None:
    result = _family_launch_result(tmp_path, DENSE_CONFIG, "Qwen/Qwen3.5-9B",
        "SPARK_COUNT=0\nCONCURRENCY=8\n" + f"KV_CACHE={kv_format}\n" + cache_keys,
        extra_args=("--wip", "dense-test"),
        encoder_plan={"placement_supported": True, "fits": True, "spark_ranks": 0})
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-qwen35 --snapshot" in line)
    assert f"--kv-cache {kv_format}" in launch
    assert "--max-sequences 8" in launch
    assert launch.count("--host-cache-bytes") == 1 and "--host-cache-bytes auto" in launch
    assert f"--prefix-cache-entries {entries}" in launch
    assert f"--prefix-point-boundaries {boundaries}" in launch
    if "PREFIX_CACHE_MARK_MIB" in cache_keys:
        assert "--prefix-cache-mark-mib 1024" in launch
    assert "--pool-tokens 0" in launch and "--vision off --audio off" in launch
    for unsupported in ("--max-context", "--local-experts", "--peers", "--split-device"):
        assert unsupported not in launch
    assert "cuteafd expertd-native" not in result.stderr
    assert "ssh h0" not in result.stderr
    plan = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    assert "--spark-ranks 0" in plan and "--state-slots 8" in plan
    assert f"--prefix-cache-entries {entries}" in plan
    assert "--prefill-rows 4096" in plan and "--require-ready" not in plan


@pytest.mark.parametrize("family", ["mimo_v2", "deepseek_v41"])
@pytest.mark.parametrize("wip", [False, True])
@pytest.mark.parametrize("allowed", [True, False, None, "true", "absent"])
def test_deployment_planned_permission_is_strict_and_wip_only(tmp_path, family, wip, allowed):
    config = {"model_type": "deepseek_v41"} if family == "deepseek_v41" else {
        "model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    model = "deepseek-ai/DeepSeek-V4.1-Flash" if family == "deepseek_v41" else "XiaomiMiMo/MiMo-V2-Flash"
    keys = ("MODEL_REVISION=\nRTX_EXPERT_LAYERS=0\nSPARK_COUNT=2\nSPARK_1_HOST=h1\nSPARK_1_LANE_A=10.0.0.2\n"
            if family == "deepseek_v41" else "SPARK_COUNT=0\nEXPERT_BACKEND=local\n")
    result = _family_launch_result(tmp_path, config, model, keys + "SPECULATOR=off\nVISION=off\nAUDIO=off\n",
        extra_args=("--wip", "planned-test") if wip else (),
        deployment_overrides={"executable": False, "wip_launchable": allowed})
    assert (result.returncode == 0) == (wip and allowed is True), result.stderr
    plans = [line for line in result.stderr.splitlines() if " plan " in line and "--deployment" in line]
    assert plans and ("--require-ready" not in plans[0]) == wip, result.stderr
    if result.returncode:
        assert "docker run -d" not in result.stderr


@pytest.mark.parametrize("family", ["mimo_v2", "deepseek_v41"])
@pytest.mark.parametrize("launch", [None, {"family": "wrong"}])
def test_deployment_wip_still_requires_matching_launch_family(tmp_path, family, launch):
    config = {"model_type": "deepseek_v41"} if family == "deepseek_v41" else {
        "model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    model = "deepseek-ai/DeepSeek-V4.1-Flash" if family == "deepseek_v41" else "XiaomiMiMo/MiMo-V2-Flash"
    keys = ("MODEL_REVISION=\nRTX_EXPERT_LAYERS=0\nSPARK_COUNT=2\nSPARK_1_HOST=h1\nSPARK_1_LANE_A=10.0.0.2\n"
            if family == "deepseek_v41" else "SPARK_COUNT=0\nEXPERT_BACKEND=local\n")
    result = _family_launch_result(tmp_path, config, model, keys + "SPECULATOR=off\nVISION=off\nAUDIO=off\n",
        extra_args=("--wip", "planned-test"),
        deployment_overrides={"executable": False, "wip_launchable": True, "launch": launch})
    assert result.returncode != 0
    assert "docker run -d" not in result.stderr


@pytest.mark.parametrize("keys,diagnostic", [
    ("SPARK_COUNT=1\n", "no Spark layout"),
    ("SPARK_COUNT=0\nEXPERT_BACKEND=spark\n", "no Spark layout"),
    ("SPARK_COUNT=0\nRTX_GPUS=2\n", "two-RTX layer ranges are not qualified"),
    ("SPARK_COUNT=0\nVISION=rtx\n", "media execution is not qualified"),
    ("SPARK_COUNT=0\nEMBEDDING=host\n", "requires device token embedding"),
    ("SPARK_COUNT=0\nMAX_CONTEXT_TOKENS=4096\n", "unused configuration key"),
])
def test_dense_qwen_refuses_unqualified_configuration(tmp_path: Path, keys: str, diagnostic: str) -> None:
    result = _family_launch_result(tmp_path, DENSE_CONFIG, "Qwen/Qwen3.5-9B", keys,
                                  extra_args=("--wip", "dense-test"))
    assert result.returncode != 0, result.stderr
    assert diagnostic in result.stderr
    assert "docker run -d" not in result.stderr


def test_dense_qwen_refuses_failed_native_plan(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, DENSE_CONFIG, "Qwen/Qwen3.5-9B", "SPARK_COUNT=0\n",
        extra_args=("--wip", "dense-test"),
        encoder_plan={"placement_supported": False, "fits": False, "spark_ranks": 0, "hints": ["missing export"]})
    assert result.returncode != 0
    assert "invalid deployment specification" in result.stderr
    assert "missing export" in result.stderr
    assert "docker run -d" not in result.stderr


@pytest.mark.parametrize("family", ["deepseek_v41", "deepseek_v4", "glm5", "glm5_flash", "mimo_v2", "qwen4", "qwen3_5"])
@pytest.mark.parametrize("failure", ["refused", "wrong-launch-family", "wrong-spark-admission"])
def test_every_family_failed_wip_deployment_prints_planner_hints(tmp_path, family, failure):
    config = ({"model_type": "deepseek_v41"} if family == "deepseek_v41" else
              {"model_type": "deepseek_v4"} if family == "deepseek_v4" else
              {"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3} if family == "glm5" else
              DENSE_CONFIG if family == "qwen3_5" else SPLIT_CONFIGS["mimo_flash" if family == "mimo_v2" else family])
    keys = "SPECULATOR=off\nVISION=off\nAUDIO=off\n"
    if family == "deepseek_v41":
        keys += "MODEL_REVISION=\nRTX_EXPERT_LAYERS=0\nSPARK_COUNT=1\n"
    elif family == "qwen3_5":
        keys += "SPARK_COUNT=0\n"
    overrides = {"executable": False, "wip_launchable": failure != "refused"}
    if failure == "wrong-launch-family":
        overrides["launch"] = {"family": "wrong"}
    if failure == "wrong-spark-admission":
        overrides["spark_admission"] = "worker_local"
    hints = ["missing native export", "requested allocation exceeds budget"]
    result = _family_launch_result(tmp_path, config, "test/model", keys,
        extra_args=("--wip", "refusal-test"), encoder_plan={"hints": hints},
        deployment_overrides=overrides)
    assert result.returncode != 0, result.stderr
    assert "invalid deployment specification" in result.stderr
    for hint in hints:
        assert hint in result.stderr
    assert "docker run -d" not in result.stderr


def test_glmf_zero_spark_launch_does_not_enable_worker_selection(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "SPARK_COUNT=0\nSPECULATOR=off\nVISION=off\nAUDIO=off\n",
                                  extra_env={"CUTEAFD_PLACEMENT_HANDSHAKE": "1"})
    assert result.returncode == 0, result.stderr
    assert "--placement-handshake" not in result.stderr
    assert "cuteafd expertd-native" not in result.stderr


def test_glmf_tp2_combine_consumes_shared_before_reusing_exchange_slot() -> None:
    source = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5_flash/engine.rs").read_text()
    pair = source.split("    fn tp2_pair(", 1)[1].split("    fn tp2_publish(", 1)[0]
    assert "self.tp2_rank(1, layer, lane, w1, rows, w1.delta.buffer.ptr)?;" in pair
    assert "self.tp2_rank(1, layer, lane, w1, rows, w1.shared.buffer.ptr)" not in pair
    assert pair.index("self.tp2_rank(1,") < pair.index("self.library.copy_d2d_async(")
    assert "self.library.copy_d2d_async(w1.shared.buffer, w1.delta.buffer," in pair
    assert "rows * self.cfg.hidden * 2, self.stream_of(1)" in pair
    publish = source.split("    fn tp2_publish(", 1)[1].split("    fn router_on(", 1)[0]
    assert "w1.shared.buffer.ptr" in publish
    for body in (pair, publish):
        assert "tp2_prepared_peer(prepared_peer)?" in body
        assert "self.peer_workspaces(" not in body
    header = (ROOT / "native/shared/include/cuteafd_experts.h").read_text()
    assert "No output/input overlap." in header


def test_glmf_remote_dflash_metadata_and_terminal_ownership_ordering() -> None:
    family = ROOT / "rust/crates/cuteafd-daemon/src/families/glm5_flash"
    remote = (family / "remote_dflash.rs").read_text()
    shared = (family.parent.parent / "shared/speculation/dflash.rs").read_text()
    update = shared.split("    pub(crate) fn update_with(", 1)[1].split("    fn put(", 1)[0]
    assert update.index("self.put(&self.context_positions") < update.index("ready()?")
    assert "let DraftContext::Gqa(context) = &self.context" in update
    assert update.index("self.put(&context.slots") < update.index("ready()?") < update.index("self.linear(")
    draft = shared.split("    fn draft_from(", 1)[1].split("let (s, l)", 1)[0]
    assert "if let AttentionWorkspace::Gqa(attention) = &w.attention" in draft
    for metadata in ("w.positions", "attention.tables", "w.anchors"):
        assert draft.index(f"self.put(&{metadata}") < draft.index("input(&ids, w.h.buffer")
    transport = remote.split("    pub fn update(", 1)[1].split("    pub fn put_taps(", 1)[0]
    assert "self.synchronize()?" not in transport
    assert "cuda_stream_synchronize" not in transport
    bridge = (family.parent.parent / "shared/draft/remote.rs").read_text()
    final_ack = bridge.split("    pub fn finish_request(", 1)[1].split("    pub fn finish_terminal(", 1)[0]
    assert final_ack.index("self.exchange.wait(0, CONTEXT)") < final_ack.index("cuda_stream_synchronize")
    release = (family / "serve.rs").read_text().split("fn release(", 1)[1].split("#[derive", 1)[0]
    assert release.index("family.finish_request()") < release.index("cache.release(")
    publish = bridge.split("    pub fn publish_context(", 1)[1].split("    pub fn acknowledge_context(", 1)[0]
    assert publish.index("self.exchange.wait(0, CONTEXT)") < publish.index("self.exchange.push_planes(0, CONTEXT")
    acknowledge = bridge.split("    pub fn acknowledge_context(", 1)[1].split("\n    }", 1)[0]
    assert "self.exchange.wait(1, CONTEXT)" in publish and "self.exchange.signal(1, CONTEXT)" in acknowledge
    assert bridge.index("    pub fn publish_context(") < bridge.index("    pub fn acknowledge_context(")
    assert transport.index("self.bridge.publish_context(") < transport.index("self.bridge.acknowledge_context()")
    taps = remote.split("    pub fn tap_streams_at(", 1)[1].split("    pub fn update(", 1)[0]
    assert "self.target.taps.buffer.ptr" in taps and "push_planes" not in taps
    owner = remote.split("pub(crate) struct RemoteDflash", 1)[1].split("impl<'a>", 1)[0]
    assert owner.index("pub draft:") < owner.index("bridge: RemoteBridge")
    shared_owner = bridge.split("pub(crate) struct RemoteBridge", 1)[1].split("impl<'a>", 1)[0]
    assert shared_owner.index("pub exchange:") < shared_owner.index("pub stream:")
    drain = (family / "engine.rs").read_text().split("    pub(crate) fn synchronize(", 1)[1].split("    fn start(", 1)[0]
    assert "drain_ranks(" in drain
    shutdown = (family / "engine.rs").read_text().split("    fn shutdown_submission(", 1)[1].split("    fn publish_tp2_abort(", 1)[0]
    assert shutdown.index("ranks:") < shutdown.index("links:") < shutdown.index("copies:") < shutdown.index("graphs:")
    assert "drafter.finish_terminal(drained)" in shutdown
    assert "synchronize_remote()" in drain
    serving = (family / "serve.rs").read_text()
    assert "drafter.require_live()?;" in serving
    prime = remote.split("    pub fn prime(", 1)[1].split("    fn launch_head(", 1)[0]
    assert "self.draft.prime_remote()" in prime
    assert "self.launch_head(&head, rows, true)?" in prime
    assert "self.launch_head(&head, rows, false)?" in prime
    assert prime.index("self.launch_head(&head, rows, true)?") < prime.index("self.launch_head(&head, rows, false)?") < prime.index("cuda_stream_synchronize(self.bridge.exchange.stream(0))")
    assert "cuda_stream_synchronize(self.bridge.exchange.stream(0))" in prime
    assert "self.bridge.exchange.wait(" not in prime and "self.bridge.exchange.push" not in prime
    isolated = shared.split("    pub(crate) fn prime_remote(", 1)[1].split("    fn workspace(", 1)[0]
    assert "0..TAP_ROWS" in isolated and "1..=self.max_sequences" in isolated
    assert "self.draft_device_with(" in isolated
    assert isolated.index("self.draft_device_with(") < isolated.index("for ring in")
    assert "DraftAttention::Gqa(layer)" in isolated and "self.discard_context(slot)?" in isolated
    assert "buffers.push(&layer.page_table)" in isolated
    assert "&attention.lengths, &attention.query_offsets" in isolated
    head_trip = bridge.split("    pub fn round_trip_head(", 1)[1].split("\n}\n", 1)[0]
    assert "head_route(self.head_rank," in head_trip
    route = bridge.split("fn head_route(", 1)[1].split("impl<'a> Prepared", 1)[0]
    assert route.index("if rank == 1 { return head(true); }") < route.index("send_hidden()?")
    assert "head(hidden, logits, self.stream.raw)" in head_trip
    assert "head(target_hidden.cast_const(), target_logits.cast_mut(), self.exchange.stream(0))" in head_trip
    startup = (family / "mod.rs").read_text().split("    pub fn with_engine<", 1)[1]
    assert startup.index("draft.prime(&engine.embedding, engine.draft_head())?") < startup.index("let result = with_startup_graphs(")


def test_shared_startup_load_overlaps_admission_before_serving_allocation() -> None:
    for family, engine in [("glm5", "GlmEngine"), ("glm5_flash", "GlmfEngine"), ("mimo_v2", "MimoEngine")]:
        source = (ROOT / f"rust/crates/cuteafd-daemon/src/families/{family}/mod.rs").read_text()
        start = source.index("let admission = if")
        body = source[start:]
        start_call = ".start(&peers)?" if ".start(&peers)?" in body else ".start_with(&peers, ranks)?"
        assert body.index(start_call) < body.index("load_while_admitting(admission,")
        assert body.index("load_while_admitting(admission,") < body.index("TokenEmbedding::load(")
        assert body.index("TokenEmbedding::load(") < body.index(f"engine::{engine}::new(")
        assert ".select(&peers)?" not in body
    shared = (ROOT / "rust/crates/cuteafd-daemon/src/shared/placement.rs").read_text()
    helper = shared.split("pub(crate) fn load_while_admitting", 1)[1].split("#[cfg(test)]", 1)[0]
    assert helper.index("let loaded = load()?") < helper.index("pending.wait()?") < helper.index("Ok(loaded)")


def test_glmf_selection_starts_after_solve_and_seals_before_opening_lanes() -> None:
    source = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5_flash/mod.rs").read_text()
    body = source[source.index("    pub fn with_engine<"):source.index("    /// The drafter `--draft`")]
    assert body.index("self.admit(") < body.index("admitted_spark_layers(") < body.index("SparkAdmission::plan(")
    assert body.index("SparkAdmission::plan(") < body.index("TokenEmbedding::load(") < body.index("self.experts_for_layers(")
    assert "args.experts_snapshot.as_deref().unwrap_or(&args.snapshot)" in body
    assert "&working, peers.len(), None, model)?;" in body and "admission.start_with(&peers, ranks)?" in body
    # Coordinator GPUs (dense operands, TP2 halves) join the Spark push group only when it pushes,
    # and without Spark peers form the push group alone.
    assert "if admission.coordinator_pushes() && dense_pushable(&args)" in body
    alone = body.split("// No Spark peers: the coordinator GPUs push alone", 1)[1].split("None\n        };", 1)[0]
    assert "plan_dense(0)?" in alone and "CoordinatorPush::prepare_alone(" in alone
    assert "coordinator push unavailable; mount" in alone
    assert body.index("load_while_admitting(admission,") < body.index("TokenEmbedding::load(")
    assert body.index("TokenEmbedding::load(") < body.index("engine::GlmfEngine::new(")
    assert "let spark = !spark_layers.is_empty();" in body
    assert "selected_args.prefill_waves = usize::try_from(working.prefill_lanes)?;" in body
    assert body.index("working.prefill_lanes") < body.index("TokenEmbedding::load(")
    assert "spark_layers == working.spark_layers" in body
    assert "rtx_layers == working.rtx_layers" in body
    assert "admitted_tp2_range" not in source and "spark_range" not in body
    assert "&local_layers, Some(budget)" in body and "&spark_layers, None" in body
    assert "Fp8MoeTp2::load_pair(devices, tensors, &package, tp2_layers.clone(),\n                    args.expert_rows(), budgets)" in body
    # Admitted TP2 expert weights land inside the Spark admission window;
    # execution binds after the engine exists and before step workspaces.
    window = body.split("load_while_admitting(admission,", 1)[1].split("probe(\"target-weights\")", 1)[0]
    assert "tp2_weights(pushed)?" in window and "push.receive_with(devices, catalog, sparks, Some(Ok(dense))" in window
    # Dense records are planned before membership: a planning error offers no
    # coordinator ranks (the Sparks still push); a group fallback after
    # membership fills the same records through the mount.
    planning = body.split("let ranks = if admission.coordinator_pushes()", 1)[1].split("admission.start_with(&peers, ranks)?", 1)[0]
    assert "plan_dense(peers.len())?" in planning and "coordinator push unavailable; mount" in planning
    plan_dense = body.split("let plan_dense = |sparks: usize|", 1)[1].split("let mut coordinator_push = None;", 1)[0]
    assert "planning.model(&self.cfg, &execution)?" in plan_dense and "dense::plan_ranks(" in plan_dense
    assert body.index("let plan_dense = |sparks: usize|") < body.index("admission.start_with(&peers, ranks)?")
    assert "dense::fill_by_mount(" in window
    assert body.index("engine::GlmfEngine::new(") < body.index("tp2.execution()?") < body.index("engine.install_tp2(ranks)?")
    assert body.index("engine.install_tp2(ranks)?") < body.index("engine.install_workspaces(workspaces)?")
    admission = source[source.index("    fn admit("):source.index("    pub fn with_engine<")]
    assert "solve_attention_working_set_with_graphs(" in admission
    assert "let local_workspace = workspaces(false, lanes);" in admission
    assert "configured_prefill_waves(layers == self.cfg.layers, args.prefill_waves)" in admission
    assert "admitted_prefill_waves(attention.mode, lanes as u64)" in admission
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5_flash/engine.rs").read_text()
    pipeline = engine.split("    fn pipelined(&self)", 1)[1].split("    /// The continuous span", 1)[0]
    assert "self.weights.layers.len() == self.cfg.layers" in pipeline
    units = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5_flash/engine/units.rs").read_text()
    assert "run_prefill_units(&mut hooks, &units, waves)" in units
    assert "wave_land(" in units and "with_waves(" in units and "unit_ffn_finish(" in units
    begin = units.split("    fn begin_unit(", 1)[1].split("    fn attention(", 1)[0]
    attention = units.split("    fn attention(", 1)[1].split("    fn local_ffn(", 1)[0]
    assert "exchange_direct(" not in begin
    assert attention.index("return e.unit_attention(") < attention.index("e.exchange_direct(") < attention.index("e.peer_attention(")
    layout = (ROOT / "rust/crates/cuteafd-loader/src/plan/layout/glm5_flash.rs").read_text()
    assert "step_workspace(&ExpertPlacement::Local, lanes)" in layout
    assert "local_workspace = Some(estimate(lanes))" in layout

    assert "admission::step_workspace_demands(" in admission
    normalization = body.split("let mut selected_args = args.clone();", 1)[1].split("let args = &selected_args;", 1)[0]
    assert "if " not in normalization, "zero-peer exchanges must also use admitted effective lanes"
    planner = (ROOT / "rust/crates/cuteafd-loader/src/plan/layout/glm5_flash.rs").read_text()
    assert "Ok((placed, working))" in planner and "selected = Some(working);" in planner
    assert "0..working.spark_ranks" in planner and "selected.filter(|working| working.spark_ranks > 0)" in planner
    loader = (ROOT / "rust/crates/cuteafd-loader/src/placement/families/glm5_flash.rs").read_text()
    request = loader.split("pub fn request_for_working_set(", 1)[1].split("/// Exact header-only", 1)[0]
    assert "working.demands(&fixed).cloned().collect()" in request
    assert "SparkWorkingSet as GlmfWorkingSet" in loader
    assert "GlmfComponent::RtxExperts, gpu, Category::Transport, \"route exchange\"" in request
    assert "GlmfComponent::SparkExperts, gpu, Category::Transport, \"Spark intake\"" in request
    expert = source[source.index("    fn experts_for_layers<"):source.index("impl Opened {\n    /// The checkpoint's `embed_tokens`")]
    assert expert.index("validate_expert_layers(&self.cfg, served, selected)?;") < expert.index("if selected.is_empty() { return Ok(None); }")
    assert expert.index("if selected.is_empty() { return Ok(None); }") < expert.index("Fp8Experts::load(") < expert.index("SparkLink::new_waves(")
    assert "tensors, &directory, selected, 1, 0," in expert
    assert "layers: selected.to_vec()" in expert
    assert "spark_warmup_request(&self.cfg, warm_rows, remote[0], self.checkpoint.spark_input_is_bf16())" in expert


def test_family_table_matches_the_rust_launch_fixtures(tmp_path: Path) -> None:
    """checkpoint-family.py and cuteafd-loader plan::launch read the same cases the
    same way (both spellings, exact pattern lengths, agreement)."""
    table = ROOT / "scripts" / "lib" / "checkpoint-family.py"
    fixtures = ROOT / "rust" / "crates" / "cuteafd-loader" / "tests" / "fixtures" / "launch-families.json"
    for case in json.loads(fixtures.read_text()):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(case["config"]))
        out = subprocess.run(["python3", str(table), str(path)], capture_output=True, text=True)
        got = out.stdout.strip() if out.returncode == 0 else None
        assert got == case["line"], (case["name"], out.stderr)


def _family_launch_result(tmp_path: Path, family_config: dict, model: str, keys: str,
                          physical_gpus: tuple[int, ...] = (0, 1), *, preflight_error: bool = False,
                          restart: bool = False, preferred_ranks: int | None = None,
                          gpu_free_mib: int = 97000, gpu_total_mib: int = 98304,
                          cuda_total_bytes: int | None = None,
                          container_pids: tuple[int, ...] = (),
                          gpu_allocations: tuple[tuple[int, int], ...] = (),
                          previous_peers: str | None = None, encoder_plan: dict | None = None,
                          extra_args: tuple[str, ...] = (), with_nest: bool = True,
                          extra_env: dict[str, str] | None = None,
                          program_manifest: dict | None = None,
                          launcher_source: Path | None = None,
                          deployment_overrides: dict | None = None,
                          probe_fails: bool = False) -> subprocess.CompletedProcess[str]:
    """Run the real run-family.sh up to its docker calls (docker/ssh/nest/curl are stubs
    that print their argv) and return what it would launch."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / "scripts" / "launch").mkdir(parents=True)
    copy_input(launcher_source or ROOT / "scripts" / "launch" / "run-family.sh", repo / "scripts" / "launch")
    copy_input(ROOT / "scripts" / "launch" / "preflight-fp8-bf16.py", repo / "scripts" / "launch")
    copy_input(ROOT / "scripts" / "lib" / "checkpoint-family.py", repo / "scripts" / "lib")
    copy_input(ROOT / "scripts" / "lib" / "release-common.sh", repo / "scripts" / "lib")
    from launcher_sandbox import stub_cuda_capacity
    stub_cuda_capacity(repo / "scripts/lib/release-common.sh",
                       cuda_total_bytes if cuda_total_bytes is not None else gpu_total_mib * 2**20)
    copy_input(ROOT / "scripts" / "lib" / "cdi-gpus.sh", repo / "scripts" / "lib")
    capacity = Path("rust/crates/cuteafd-core/src/serving_capacity.rs")
    (repo / capacity.parent).mkdir(parents=True)
    copy_input(ROOT / capacity, repo / capacity)
    native_v41 = family_config.get("model_type") == "deepseek_v41"
    if native_v41:
        programs = Path("rust/crates/cuteafd-core/src/coordinator_programs.rs")
        (repo / programs.parent).mkdir(parents=True, exist_ok=True)
        copy_input(ROOT / programs, repo / programs)
        copy_input(ROOT / "scripts" / "launch" / "select-release-gpus.py", repo / "scripts" / "launch")
        (repo / "scripts" / "build").mkdir(exist_ok=True)
        (repo / "scripts" / "build" / "verify-sparkinfer-source.py").write_text("print('test-pin')\n")
    wip = "--wip" in extra_args
    exact_count = 0 if "EXPERT_BACKEND=local" in keys else 1
    for line in keys.splitlines():
        if line.startswith("SPARK_COUNT="):
            exact_count = int(line.split("=", 1)[1])
    if encoder_plan is not None:
        encoder_plan = {**encoder_plan, "spark_ranks": exact_count}
    if wip:
        (repo / "scripts" / "build").mkdir(exist_ok=True)
        (repo / "scripts" / "build" / "verify-sparkinfer-source.py").write_text("print('test-pin')\n")
    hf = tmp_path / "hf"
    _snapshot(hf, model, family_config, "a" * 40 if native_v41 else "abc")
    if family_config.get("model_type") == "glm_moe_dsa":
        _snapshot(hf, "incoai/GLM-5.3-DFlash2", {"architectures": ["DFlash2DraftModel"]})
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("docker", "ssh", "nest") if with_nest else ("docker", "ssh"):
        # One line per invocation, in a single write: the per-host worker legs
        # run concurrently, and a two-write line (argv, then newline) lets a
        # sibling's argv slip between them and merge two lines into one.
        (bin_dir / tool).write_text('#!/usr/bin/env bash\nprintf "%s \\n" "$(basename "$0") $*" >&2\n'
                                    + ('case "$*" in *"docker run --rm"*"python3"*) exit 2 ;; esac\n'
                                       if preflight_error and tool == "ssh" else '') +
                                    'case "$*" in *"docker logs"*) echo "${STUB_WORKER_READY:-worker bootstrap ready}"; '
                                    'echo "audio encoder ready backend=mimo_audio_fp32_v1/cuda13000/cufft12000/cublas13.0.0/cute_aot_sm121/export' + 'cd' * 32 + '" ;; esac\n' +
                                    ('case "$*" in image\\ inspect*) echo test-pin ;; '
                                     'cp\\ *) dst="${@: -1}"; mkdir -p "$dst"; '
                                     'touch "$dst/cuteafd" "$dst/libcuteafd_native.so" "$dst/release-entrypoint.sh"; '
                                     'if [[ "$2" == *verify-sparkinfer-source.py ]]; then printf "print(42)\\n" > "$dst/verify-sparkinfer-source.py"; fi ;; esac\n'
                                     if tool == "docker" and wip else '') +
                                    (f"case \"$*\" in *\"cuteafd plan\"*) echo '{{\"spark_ranks\":{preferred_ranks}}}' ;; esac\n"
                                     if tool == "docker" and preferred_ranks is not None and encoder_plan is None else '') +
                                    ('case "$*" in *"cuteafd plan"*) n=1; prev=""; for a in "$@"; do '
                                     '[[ "$prev" != --spark-ranks ]] || n="$a"; prev="$a"; done; '
                                     'printf \'{"spark_ranks":%s}\\n\' "$n" ;; esac\n'
                                     if tool == "docker" and preferred_ranks is None and encoder_plan is None and not native_v41 else '') +
                                    (f"case \"$*\" in *\"cuteafd plan\"*) printf '%s\\n' '{json.dumps(encoder_plan)}' ;; esac\n"
                                     if tool == "docker" and encoder_plan is not None else '') +
                                    ("case \"$*\" in top*) printf '%s\\n' PID " +
                                     " ".join(map(str, container_pids)) + " ;; esac\n"
                                     if tool == "docker" and container_pids else '') +
                                    (f"case \"$*\" in inspect*) echo '[\"serve-qwen4\",\"--peers\",\"{previous_peers}\"]' ;; esac\n"
                                     if tool == "docker" and previous_peers is not None else ''))
        if tool == "ssh" and native_v41:
            with (bin_dir / tool).open("a") as stream:
                stream.write('case "$*" in *spark_tp_roles*|*"python3 -"*"tp2"*) echo tp2 ;; *"docker container inspect"*) exit 1 ;; esac\n')
        if tool == "docker":
            stub = (bin_dir / tool).read_text()
            intercept = 'case "$*" in *" plan "*"--deployment"*) exec python3 "$STUB_DEPLOYMENT" "$@" ;; esac\n'
            if probe_fails:
                # A slot binary that predates the family's serve command rejects it.
                intercept += 'case "$*" in *"/opt/cuteafd-serve-probe"*) exit 2 ;; esac\n'
            stub = stub.replace('case "$*" in', intercept + 'case "$*" in', 1)
            if native_v41:
                stub += ('case "$*" in container\\ inspect*|inspect\\ *) ' + ('if [[ "$*" == *-wip* ]]; then exit 0; else exit 1; fi' if wip else 'exit 1') + ' ;; image\\ inspect*) echo test-pin ;; esac\n'
                         'case "$*" in *"cuteafd plan"*) printf \'{"spark_ranks":%s,"prefill_unit_rows":4096,"prefill_waves":1,"prefill_span_rows":4096}\\n\' '
                         '"$STUB_SPARK_COUNT" ;; esac\n')
            if program_manifest is None:
                stub += 'case "$*" in *"PROGRAMS.json"*) exit 1 ;; esac\n'
            else:
                manifest_path = tmp_path / "PROGRAMS.json"
                manifest_path.write_text(json.dumps(program_manifest))
                stub += '''case "$*" in *"PROGRAMS.json"*)
while [[ $1 != -c ]]; do shift; done
code="${2//\\/opt\\/cuteafd\\/share\\/PROGRAMS.json/$STUB_PROGRAMS}"
shift 2
exec python3 -c "$code" "$@"
;; esac
'''
            (bin_dir / tool).write_text(stub)
        (bin_dir / tool).chmod(0o755)
    (bin_dir / "curl").write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' \'{"data":[{"id":"test/model"}]}\'\n')
    (bin_dir / "curl").chmod(0o755)
    (bin_dir / "nvidia-smi").write_text('#!/usr/bin/env bash\ncase "$*" in *memory.total*) echo ' + str(gpu_total_mib) +
                                         ' ;; *memory.free*) echo ' + str(gpu_free_mib) +
                                         " ;; *query-gpu=uuid*) echo GPU-00000000-0000-0000-0000-000000000000 ;; *) printf '%s\\n' " + " ".join(map(str, physical_gpus)) +
                                         ' ;; esac\n' if preferred_ranks is None else
                                         '#!/usr/bin/env bash\ncase "$*" in *memory.free*) echo ' + str(gpu_free_mib) +
                                         ' ;; *memory.total*) echo ' + str(gpu_total_mib) +
                                         ' ;; *query-gpu=uuid*) echo GPU-00000000-0000-0000-0000-000000000000 ;; *query-compute-apps*) printf \'%s\\n\' ' +
                                         " ".join(f"'{pid}, {mib}'" for pid, mib in gpu_allocations) +
                                         ' ;; *) echo 0; echo 1 ;; esac\n')
    (bin_dir / "nvidia-smi").chmod(0o755)
    if native_v41:
        (bin_dir / "nvidia-smi").write_text(
            '#!/usr/bin/env bash\ncase "$*" in '
            '*index,uuid,pci.bus_id,memory.total,memory.free*) '
            f'echo "0, GPU-00000000-0000-0000-0000-000000000000, 00000000:01:00.0, {gpu_total_mib}, {gpu_free_mib}" ;; '
            '*index,uuid,pci.bus_id*) echo "0, GPU-00000000-0000-0000-0000-000000000000, 00000000:01:00.0" ;; '
            f'*memory.total*) echo {gpu_total_mib} ;; *memory.free*) echo {gpu_free_mib} ;; *) echo 0 ;; esac\n')
        (bin_dir / "curl").write_text(
            '#!/usr/bin/env bash\nprintf \'%s\\n\' \'{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4.1-Flash"}]}\'\n')
    deployment_stub = tmp_path / "deployment-stub.py"
    deployment_stub.write_text("""import json, os, subprocess, sys
args = sys.argv[1:]
def value(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default
config = json.loads(os.environ["STUB_CHECKPOINT"])
described = subprocess.check_output([sys.executable, os.environ["STUB_FAMILY_READER"], os.environ["STUB_CONFIG_PATH"]], text=True).split()
family, _, first, last = described
ranks = int(os.environ["STUB_PREFERRED_RANKS"] if os.environ["STUB_FORCE_PREFERRED"] else value("--spark-ranks", os.environ["STUB_PREFERRED_RANKS"]))
tp = int(value("--spark-tp", str(ranks)))
ep = int(value("--spark-ep", "1"))
explicit = "--spark-tp" in args or family == "deepseek_v41" and ranks not in (0, 4)
peers = value("--deployment-peers", "").split(",")[:ranks] if ranks else []
report = json.loads(os.environ["STUB_ENCODER_PLAN"])
report.update(spark_ranks=ranks)
gpus = int(value("--rtx", "1"))
placement_ready = report.get("placement_supported", True) and report.get("fits", True)
shared = family in {"step5", "kimi_k3"}
glm = family == "glm5"
unit = int(value("--prefill-rows", "0")) or (2048 if shared else 4096)
waves = 1 if glm and ranks == 0 else int(value("--prefill-waves", "0")) or (2 if shared or glm or family == "deepseek_v4" else 1)
span = int(value("--prefill-span-rows", "0")) or 16384 if glm or family == "deepseek_v4" else unit * waves
report.update(prefill_unit_rows=unit, prefill_waves=waves, prefill_span_rows=span, prefill_transports=1)
deployment = dict(version=1, family=family, executable=family != "qwen3_5" and placement_ready, wip_launchable=placement_ready, rtx_gpus=gpus, spark_ranks=ranks,
    topology=dict(tp=tp, ep=ep,
                  wire="native_ownership" if explicit else "canonical"), peers=peers,
    speculator=value("--speculator", "off"), prefill_unit_rows=unit, prefill_waves=waves, prefill_span_rows=span,
    admission=dict(kv_pool_size=value("--kv-pool-size"), memory_reservation=value("--memory-reservation")),
    spark_admission="coordinator_authoritative",
    launch=dict(family=family, first_expert_layer=int(first), last_expert_layer=None if last == "-1" else int(last)),
    artifact_requirements=dict(spark_tp_role=f"tp{tp}" if family == "deepseek_v41" and explicit and tp != 4 else None), report=report)
deployment.update(json.loads(os.environ["STUB_DEPLOYMENT_OVERRIDES"]))
if deployment.get("wip_launchable") == "absent":
    deployment.pop("wip_launchable")
print(json.dumps(deployment))
""")
    config = repo / "f.config"
    initial_count = 0 if "EXPERT_BACKEND=local" in keys else 1
    config.write_text(f"MODEL_ID={model}\nSPARK_COUNT={initial_count}\nSPARK_0_HOST=h0\nSPARK_0_LANE_A=10.0.0.1\n{keys}")
    env = {**os.environ, "HF_HOME": str(hf), "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "STUB_PROGRAMS": str(tmp_path / "PROGRAMS.json"), "STUB_SPARK_COUNT": str(exact_count),
           "STUB_DEPLOYMENT": str(deployment_stub), "STUB_CHECKPOINT": json.dumps(family_config),
           "STUB_CONFIG_PATH": str(hf / "hub" / f"models--{model.replace('/', '--')}" / "snapshots" / ("a" * 40 if native_v41 else "abc") / "config.json"),
           "STUB_FAMILY_READER": str(repo / "scripts/lib/checkpoint-family.py"),
           "STUB_PREFERRED_RANKS": str(preferred_ranks if preferred_ranks is not None else 1),
           "STUB_FORCE_PREFERRED": "1" if preferred_ranks is not None and encoder_plan is None else "",
           "STUB_ENCODER_PLAN": json.dumps(encoder_plan or {"spark_ranks": exact_count}),
           "STUB_DEPLOYMENT_OVERRIDES": json.dumps(deployment_overrides or {}),
           **(extra_env or {})}
    env["HOME"] = str(tmp_path / "home")
    if not with_nest:
        # Hide any nest the host has, keeping only the stub directory and the system tools.
        env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
    # Capture through files, not pipes: the per-host worker legs print
    # concurrently, and a pipe capture also loses a writer that is a moment
    # behind the launcher when it exits.
    stdout_path, stderr_path = tmp_path / "launch.stdout", tmp_path / "launch.stderr"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(["bash", str(repo / "scripts" / "launch" / "run-family.sh"), "--config", str(config),
                                    *(["--restart"] if restart else []), *extra_args],
                                   env=env, stdout=stdout, stderr=stderr, timeout=30)

    def read(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    # V4.1 launches its experts through a heredoc (the command line is stdin, so
    # it prints no worker argv); every other family prints one per host.
    expected = 0 if native_v41 or completed.returncode else exact_count
    deadline = time.monotonic() + 10
    while True:
        text = read(stderr_path)
        if text.count("cuteafd expertd-native") >= expected or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    return subprocess.CompletedProcess(completed.args, completed.returncode, read(stdout_path), text)


@pytest.mark.parametrize("family_config,model,row_key,rtx,ranks", [
    ({"model_type": "step5_vl", "text_config": {"model_type": "step5", "num_hidden_layers": 92,
      "moe_layer_list": list(range(3, 91))}}, "test/Step-5-NVFP4", "PREFILL_BATCH_TOKENS", 1, 4),
    ({"model_type": "step5_vl", "text_config": {"model_type": "step5", "num_hidden_layers": 92,
      "moe_layer_list": list(range(3, 91))}}, "test/Step-5", "PREFILL_BATCH_TOKENS", 2, 6),
    ({"model_type": "kimi_k3", "text_config": {"model_type": "kimi_linear", "num_hidden_layers": 93,
      "first_k_dense_replace": 1, "moe_layer_freq": 1}}, "test/Kimi-K3", "PREFILL_ROWS", 2, 6),
    ({"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
      "mtp_num_hidden_layers": 1, "layer_types": ["linear_attention", "full_attention"]}},
     "test/Qwen4", "PREFILL_ROWS", 1, 4),
    ({"model_type": "deepseek_v4", "num_hidden_layers": 61}, "test/V4", "PREFILL_ROWS", 1, 4),
])
@pytest.mark.parametrize("explicit", [False, True])
def test_shared_prefill_launcher_forwards_resolved_units_without_span(
    tmp_path: Path, family_config: dict, model: str, row_key: str, rtx: int, ranks: int, explicit: bool,
) -> None:
    keys = (f"SPARK_COUNT={ranks}\nSPECULATOR=off\nVISION=off\nAUDIO=off\n"
            f"RTX_GPUS={rtx}\nCOORDINATOR_GPUS=" + ("0,1" if rtx == 2 else "0") + "\n")
    keys += "".join(f"SPARK_{rank}_HOST=h{rank}\nSPARK_{rank}_LANE_A=10.0.0.{rank + 1}\n"
                    for rank in range(1, ranks))
    if explicit:
        keys += f"{row_key}=1024\nPREFILL_WAVES=2\n"
    unit, waves = (1024, 2) if explicit else (4096 if family_config["model_type"] == "deepseek_v4" else 2048, 2)
    result = _family_launch_result(tmp_path, family_config, model, keys,
        extra_args=("--wip", "prefill-shape"),
        encoder_plan={"placement_supported": True, "fits": True, "spark_budget_bytes": 108 << 30},
        deployment_overrides={"prefill_unit_rows": unit, "prefill_waves": waves,
                              "prefill_span_rows": 32768})
    assert result.returncode == 0, result.stderr
    plan = next(shlex.split(line) for line in result.stderr.splitlines()
                if "cuteafd plan" in line and "--deployment" in line)
    if explicit:
        assert plan[plan.index("--prefill-rows") + 1] == str(unit)
        assert plan[plan.index("--prefill-waves") + 1] == str(waves)
    assert "--prefill-lanes" not in plan
    command_name = {"kimi_k3": "serve-kimi", "qwen4_exp": "serve-qwen4", "deepseek_v4": "serve-dsv4"}.get(
        family_config["model_type"], "serve-step5")
    serve = next(shlex.split(line) for line in result.stderr.splitlines()
                 if "docker run -d" in line and command_name in line)
    assert serve.count("--prefill-rows") == serve.count("--prefill-waves") == 1
    assert serve[serve.index("--prefill-rows") + 1] == str(unit)
    assert serve[serve.index("--prefill-waves") + 1] == str(waves)
    assert "--prefill-span-rows" not in serve
    assert "PREFILL_SPAN_ROWS" not in result.stdout + result.stderr
    assert f"Resolved prefill: PREFILL_ROWS={unit} PREFILL_WAVES={waves}" in result.stdout


@pytest.mark.parametrize("family_config", [
    {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
])
def test_legacy_prefill_launcher_does_not_forward_typed_shape(tmp_path: Path, family_config: dict) -> None:
    result = _family_launch_result(tmp_path, family_config, "test/model",
        "SPECULATOR=off\nVISION=off\nAUDIO=off\n",
        deployment_overrides={"prefill_unit_rows": 1024, "prefill_waves": 3, "prefill_span_rows": 32768})
    assert result.returncode == 0, result.stderr
    serve = next(shlex.split(line) for line in result.stderr.splitlines()
                 if "docker run -d" in line and "cuteafd serve-" in line)
    assert "--prefill-waves" not in serve and "--prefill-span-rows" not in serve
    assert "Resolved prefill:" not in result.stdout
    assert "PREFILL_SPAN_ROWS" not in result.stdout + result.stderr


@pytest.mark.parametrize("wip", [False, True])
def test_serving_coordinator_and_workers_use_host_identity(tmp_path, wip):
    result = _family_launch_result(
        tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
        "SPECULATOR=off\nVISION=off\nAUDIO=off\n",
        extra_args=("--wip", "slot-test") if wip else ())
    assert result.returncode == 0, result.stderr
    coordinator = next(line for line in result.stderr.splitlines() if "serve-qwen4 --snapshot" in line)
    assert f"--user {os.getuid()}:{os.getgid()}" in coordinator
    assert "HF_HOME=/hf" in coordinator
    assert "CUTEAFD_BENCH_DIR=/run/cuteafd/bench" in coordinator
    # Dashboard fidelity reuses the host's verified reference cache (no re-download).
    assert "/.cache/cuteafd/fidelity:/run/cuteafd/fidelity" in coordinator
    assert "CUTEAFD_FIDELITY_CACHE=/run/cuteafd/fidelity" in coordinator
    # Nothing may be bind-mounted under the container HOME: Docker would create
    # root-owned parents there, and the non-root coordinator's placement handoff fails.
    assert ":/tmp/cuteafd-home/" not in coordinator
    worker = result.stderr.split("ssh h0 release_serving_docker_args ()", 1)[1]
    assert 'id -u' in worker and 'id -g' in worker
    assert 'release_serving_docker_args; cuteafd_cdi_gpu_args all || exit 2; docker run -d' in worker or ('release_serving_docker_args; cuteafd_cdi_gpu_args all' in worker and 'docker run -d' in worker)
    assert '"${RELEASE_SERVING_ARGS[@]}"' in worker
    assert '/root/' not in coordinator + worker


@pytest.mark.parametrize("qwen", [False, True])
def test_wip_startup_plans_use_staged_slot_mounts_and_entrypoint(tmp_path, qwen):
    config = ({**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": "exl3"},
               "vision_config": {"depth": 24}}
              if qwen else {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                            "moe_layer_freq": [0, 1], "vision_config": {"depth": 28}})
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32,
            "encoder": {"kind": {"kind": "rtx", "gpu": 0}, "replicas": []}}
    result = _family_launch_result(tmp_path, config, "test/model",
                                  "VISION=rtx\nAUDIO=off\nRTX_GPUS=1\nSPECULATOR=off\nWIP_INSTANCE=plan-test\n",
                                  preferred_ranks=1 if qwen else None, encoder_plan=plan,
                                  extra_args=("--wip", "slot-test"))
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    plans = [line for line in lines if "cuteafd plan" in line]
    assert len(plans) == 1
    staging = next(i for i, line in enumerate(lines) if "docker cp" in line and ".cuteafd-wip" in line)
    for preflight in plans:
        assert staging < lines.index(preflight)
        assert "cuteafd-coordinator-dev cuteafd plan" in preflight
        for part in ("bin", "lib", "share"):
            assert f"wip-run/plan-test/slot-test/serve-default/{part}:/opt/cuteafd/{part}:ro" in preflight
        assert "--entrypoint /opt/cuteafd/share/release-entrypoint.sh" in preflight
        # Draft snapshots resolve under HF_HOME inside the full namespace mount.
        assert "HF_HOME=/mnt/sparknest/hf-home" in preflight and "/mnt/sparknest:/mnt/sparknest:ro" in preflight


@pytest.mark.parametrize("budget,total,embedding", [("", 32768, "host"), ("31.8", 98304, "host"), ("", 98304, "gpu")])
def test_mimo_full_context_profile_respects_physical_and_logical_memory(tmp_path, budget, total, embedding):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    keys = f"COORDINATOR_GPU_BUDGET_GIB={budget}\nEXPERT_BACKEND=spark\nSPECULATOR=off\nVISION=off\nAUDIO=off\n"
    result = _family_launch_result(tmp_path, config, "test/mimo", keys, gpu_total_mib=total)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "--max-context 0" in launch
    assert f"--max-sequences {8 if embedding == 'host' else 16}" in launch
    assert f"--embedding-placement {embedding}" in launch
    assert "--pool-tokens 0" in launch


@pytest.mark.parametrize("family", ["glm5_flash", "glm5", "mimo_v2", "deepseek_v4", "qwen4"])
@pytest.mark.parametrize("budget,total", [("", 32768), ("31.8", 98304), ("", 98304)])
def test_family_prelaunch_plan_uses_coordinator_card_memory(tmp_path, family, budget, total):
    result = _family_launch_result(
        tmp_path, ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}
                   if family == "glm5" else {"model_type": "deepseek_v4"}
                   if family == "deepseek_v4" else SPLIT_CONFIGS["mimo_flash" if family == "mimo_v2" else family]), "test/model",
        f"COORDINATOR_GPU_BUDGET_GIB={budget}\nRTX_GPUS=1\nSPECULATOR=off\nVISION=off\nAUDIO=off\n",
        gpu_total_mib=total, gpu_free_mib=total)
    assert result.returncode == 0, result.stderr
    plan = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    args = plan.split()
    expected = float(budget) if budget else total / 1024
    for flag in ("--coordinator-gpu-budget-gib", "--coordinator-weight-budget-gib"):
        assert float(args[args.index(flag) + 1]) == expected
    assert "94.97" not in plan


@pytest.mark.parametrize("family", ["glm5_flash", "glm5", "mimo_v2", "deepseek_v4", "qwen4", "deepseek_v41"])
@pytest.mark.parametrize("budget,total,override,expected", [
    ("", 98304, "", 16), ("", 32768, "", 8), ("32", 98304, "", 8),
    ("32.001", 98304, "", 16), ("96", 32768, "", 8),
    ("32", 98304, "config", 4), ("", 98304, "set", 4),
])
def test_shared_concurrency_default_reaches_plan_and_serve(tmp_path, family, budget, total, override, expected):
    config = ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}
              if family == "glm5" else {"model_type": "deepseek_v4"}
              if family == "deepseek_v4" else {"model_type": "deepseek_v41"}
              if family == "deepseek_v41" else SPLIT_CONFIGS["mimo_flash" if family == "mimo_v2" else family])
    model = "deepseek-ai/DeepSeek-V4.1-Flash" if family == "deepseek_v41" else "test/model"
    keys = (f"COORDINATOR_GPU_BUDGET_GIB={budget}\nRTX_GPUS=1\n"
            "SPECULATOR=off\nVISION=off\nAUDIO=off\n")
    if family == "deepseek_v41":
        keys += ("MODEL_REVISION=\nRTX_EXPERT_LAYERS=0\nSPARK_COUNT=2\n"
                 "SPARK_1_HOST=h1\nSPARK_1_LANE_A=10.0.0.2\n")
    if override:
        keys += f"CONCURRENCY={4 if override == 'config' else 7}\n"
    result = _family_launch_result(tmp_path, config, model, keys, gpu_total_mib=total,
                                   gpu_free_mib=total - 1024,
                                   extra_args=("--set", "CONCURRENCY=4") if override == "set" else ())
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    plans = [line for line in lines if "cuteafd plan" in line and "--deployment" in line]
    assert plans, result.stderr
    for plan in plans:
        args = plan.split()
        assert args.count("--concurrency") == 1
        assert args[args.index("--concurrency") + 1] == str(expected)
    serving = next(line for line in lines if "serve-native --snapshot" in line) if family == "deepseek_v41" else next(
        line for line in lines if "--max-sequences" in line and "--snapshot" in line)
    flag = "--concurrency" if family == "deepseek_v41" else "--max-sequences"
    assert f"{flag} {expected}" in serving


def test_mimo_worker_selection_is_default_on_both_ends(tmp_path):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(
        tmp_path, config, "test/mimo",
        "SPARK_COUNT=1\nEXPERT_BACKEND=spark\nSPECULATOR=off\nVISION=off\nAUDIO=off\n",
        extra_env={"STUB_WORKER_READY": "worker bootstrap ready"})
    assert result.returncode == 0, result.stderr
    worker = next(line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line)
    coordinator = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "--placement-handshake" not in worker
    assert "--placement-handshake" not in coordinator
    assert "worker bootstrap ready" in result.stderr


@pytest.mark.parametrize("embedding", ["host", "gpu"])
def test_mimo_small_card_profile_keeps_explicit_overrides(tmp_path, embedding):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    keys = f"EMBEDDING={embedding}\nMAX_CONTEXT_TOKENS=65536\nCONCURRENCY=2\nPOOL_TOKENS=131072\nEXPERT_BACKEND=spark\nSPECULATOR=off\nVISION=off\nAUDIO=off\n"
    result = _family_launch_result(tmp_path, config, "test/mimo", keys, gpu_total_mib=32768)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "--max-context 65536" in launch
    assert "--max-sequences 2" in launch
    assert f"--embedding-placement {embedding}" in launch
    assert "--pool-tokens 131072" in launch


@pytest.mark.parametrize("setting,expected", [("", None), ("on", "1"), ("off", "0")])
def test_qwen_startup_graph_default_is_owned_by_engine(tmp_path, setting, expected):
    keys = "" if not setting else f"QWEN_STARTUP_GRAPHS={setting}\n"
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/qwen", keys)
    assert result.returncode == 0, result.stderr
    if expected is None:
        assert "CUTEAFD_QWEN4_STARTUP_GRAPHS=" not in result.stderr
    else:
        assert f"CUTEAFD_QWEN4_STARTUP_GRAPHS={expected}" in result.stderr


@pytest.mark.parametrize("setting,expected", [("", "fp8"), ("bf16", "bf16"), ("fp8", "fp8")])
def test_qwen_kv_cache_is_explicit_and_fp8_default(tmp_path, setting, expected):
    keys = "" if not setting else f"KV_CACHE={setting}\n"
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/qwen", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-qwen4 --snapshot" in line)
    plans = [line for line in result.stderr.splitlines() if "cuteafd plan" in line]
    assert plans
    assert f"--kv-cache {expected}" in launch
    assert all(f"--kv-cache {expected}" in line for line in plans)
    assert "--kv-format" not in result.stderr


def test_qwen_invalid_kv_cache_fails_before_launch(tmp_path):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/qwen", "KV_CACHE=int8\n")
    assert result.returncode == 2, result.stderr
    assert "KV_CACHE must be bf16 or fp8" in result.stderr
    assert "serve-qwen4 --snapshot" not in result.stderr
    assert "docker run -d" not in result.stderr


def test_kv_format_is_an_unknown_key(tmp_path):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/qwen", "KV_FORMAT=fp8\n")
    assert result.returncode == 2, result.stderr
    assert "unknown configuration key: KV_FORMAT" in result.stderr
    assert "deprecated" not in result.stderr and "was removed" not in result.stderr
    assert "docker run -d" not in result.stderr


@pytest.mark.parametrize("setting,expected", [("", None), ("int8", "int8"), ("bf16", "bf16")])
def test_mimo_kv_cache_omits_the_flag_unless_set(tmp_path, setting, expected):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    keys = "SPECULATOR=off\n" + ("" if not setting else f"KV_CACHE={setting}\n")
    result = _family_launch_result(tmp_path, config, "test/mimo", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    plans = [line for line in result.stderr.splitlines() if "cuteafd plan" in line]
    assert plans and all("--kv-cache" not in line for line in plans)
    if expected is None:
        assert "--kv-cache" not in launch
    else:
        assert f"--kv-cache {expected}" in launch


def test_mimo_invalid_kv_cache_fails_before_launch(tmp_path):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(tmp_path, config, "test/mimo", "SPECULATOR=off\nKV_CACHE=fp8\n")
    assert result.returncode == 2, result.stderr
    assert "KV_CACHE must be int8 or bf16" in result.stderr
    assert "docker run -d" not in result.stderr


@pytest.mark.parametrize("family,extra,unused", [
    ("qwen4", "MIMO_FP8_HEAD=on\n", "MIMO_FP8_HEAD"),
    ("mimo_flash", "QWEN_FP8_DECODE=on\n", "QWEN_FP8_DECODE"),
    ("glm5", "KV_CACHE=fp8\n", "KV_CACHE"),
    ("glm5_flash", "KV_CACHE=fp8\n", "KV_CACHE"),
    ("deepseek_v4", "KV_CACHE=bf16\n", "KV_CACHE"),
    ("deepseek_v41", "KV_CACHE=fp8\n", "KV_CACHE"),
])
def test_a_key_the_family_never_reads_fails_before_launch(tmp_path, family, extra, unused):
    configs = {
        "qwen4": SPLIT_CONFIGS["qwen4"],
        "mimo_flash": {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
        "glm5": {"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3},
        "glm5_flash": SPLIT_CONFIGS["glm5_flash"],
        "deepseek_v4": {"model_type": "deepseek_v4"},
        "deepseek_v41": {"model_type": "deepseek_v41"},
    }
    model = {"glm5": "zai-org/GLM-5.3", "deepseek_v41": "deepseek-ai/DeepSeek-V4.1-Flash"}.get(family, "test/model")
    keys = "SPECULATOR=off\n" + extra
    if family == "deepseek_v41":
        keys += ("MODEL_REVISION=\nRTX_EXPERT_LAYERS=0\nSPARK_COUNT=2\n"
                 "SPARK_1_HOST=h1\nSPARK_1_LANE_A=10.0.0.2\n")
    result = _family_launch_result(tmp_path, configs[family], model, keys)
    assert result.returncode == 2, result.stderr
    assert f"unused configuration key: {unused}" in result.stderr
    assert "docker run -d" not in result.stderr


def test_the_wip_slot_root_a_card_config_names_launches_every_family(tmp_path):
    # The card harness launches each card from a generated config that names the
    # WIP root it staged, and the shared `--wip` staging hands that root to every
    # rank's staging script. A family whose allowed set omits it dies before any
    # container runs ("unused configuration key: WIP_ROOT"), so the key must be
    # accepted here and must reach the ranks.
    root = tmp_path / "home" / ".cache" / "cuteafd" / "builds" / "wip-test"
    result = _family_launch_result(
        tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
        f"SPECULATOR=off\nVISION=off\nAUDIO=off\nWIP_ROOT={root}\n",
        extra_args=("--wip", "slot-test"))
    assert result.returncode == 0, result.stderr
    assert "unused configuration key" not in result.stderr
    stages = [line for line in result.stderr.splitlines() if line.startswith("ssh h0 bash -s -- slot-test")]
    assert any(str(root) in line for line in stages), "\n".join(stages)


def test_mimo_transport_timing_is_resolved_once_for_all_prefill_lanes():
    source = (ROOT / "rust/crates/cuteafd-daemon/src/families/mimo_v2/mod.rs").read_text()
    timing = "let timing = cuteafd_transport::protocol_v2_timing_from_env();"
    start = source.index(timing)
    link = source.index("let link = ||", start)
    end = source.index("crate::shared::memory_report::release_load_staging", link)
    assert start < link
    assert source.count(timing) == 1
    assert "TcpTransportConfig { timing," in source[link:end]
    assert "timing: false" not in source[link:end]


@pytest.mark.parametrize("config,keys,env,endpoints", [
    ({"model_type": "deepseek_v4"}, "", {}, 2),
    ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}, "", {}, 4),
    ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}, "PREFILL_WAVES=1\n", {}, 1),
    ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}, "PREFILL_WAVES=4\n", {}, 5),
    ({"model_type": "glm5_next"}, "", {}, 1),
    ({"model_type": "glm5_next"}, "PREFILL_WAVES=4\n", {}, 1),
    ({"model_type": "qwen4_exp"}, "", {}, 1),
    ({"model_type": "mimo_v2_flash"}, "", {}, 2),
    ({"model_type": "mimo_v2", "hidden_size": 6144}, "", {}, 3),
    ({"model_type": "mimo_v2", "hidden_size": 6144}, "", {"CUTEAFD_MIMO_PREFILL_LANES": "4"}, 4),
])
def test_worker_endpoint_count_matches_family_connection_ownership(tmp_path, config, keys, env, endpoints):
    dense_layers = config.get("first_k_dense_replace", 0)
    fixture = {"num_hidden_layers": 4, "layer_types": ["linear_attention"] * 4,
        "mlp_layer_types": ["dense"] * dense_layers + ["sparse"] * (4 - dense_layers), **config}
    result = _family_launch_result(tmp_path, fixture, "test/model",
        "EXPERT_BACKEND=spark\nSPECULATOR=off\nVISION=off\nAUDIO=off\n" + keys,
        extra_env=env)
    assert result.returncode == 0, result.stderr
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert workers
    assert all("--rdma-endpoints" not in line and "--capacity" not in line for line in workers)
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert all(f"{name}={value}" in launch for name, value in env.items())


@pytest.mark.parametrize("mode,wide", [("fp8", False), ("bf16", True), ("bf16-decode", False)])
def test_only_wide_optional_bf16_changes_worker_ring_admission(tmp_path, mode, wide):
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 4}, "test/model",
        f"EXPERT_BACKEND=spark\nSPECULATOR=off\nVISION=off\nAUDIO=off\nEXPERT_INPUT={mode}\n")
    assert result.returncode == 0, result.stderr
    worker = next(line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line)
    assert "--bf16-ingress" not in worker
    coordinator = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert f"--expert-input {mode}" in coordinator


def test_qwen_tp1_explicit_pool_host_maps_physical_rails(tmp_path: Path) -> None:
    config = {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": "exl3"}}
    result = _family_launch_result(tmp_path, config, "test/model",
                                  "SPARK_HOSTS=moa\nEXPERT_BACKEND=spark\nSPECULATOR=off\n")
    assert result.returncode == 0, result.stderr
    worker = next(line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line)
    assert "moa" in worker
    assert "--rank 0 --world 1" in worker
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--peers 10.55.0.6:" in launch
    assert "--local-experts" not in launch
    assert not any("ssh" in line and "h0" in line for line in result.stderr.splitlines())


@pytest.mark.parametrize("hosts,message", [
    ("moa,moa", "exactly SPARK_COUNT"),
    ("unknown", "unknown Spark pool host"),
    ("moa,", "comma-separated Spark pool host list"),
    ("moa,moa\nSPARK_COUNT=2", "duplicate host"),
])
def test_explicit_pool_hosts_reject_invalid_selection(tmp_path: Path, hosts: str, message: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  f"SPARK_HOSTS={hosts}\nEXPERT_BACKEND=spark\nSPECULATOR=off\n")
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert "cuteafd expertd-native" not in result.stderr
    assert "cuteafd serve-qwen4" not in result.stderr


@pytest.mark.parametrize("count", [1, 5])
def test_explicit_custom_hosts_use_configured_rails_in_csv_order(count):
    hosts = [f"spark-{rank}.example" for rank in range(count)]
    script = ['set -eu', 'source "$1"']
    for rank, host in enumerate(hosts):
        script += [f'SPARK_{rank}_HOST={host}', f'SPARK_{rank}_LANE_A=192.0.2.{rank + 1}',
                   f'SPARK_{rank}_LANE_B=198.51.100.{rank + 1}']
    script.append(f'release_spark_host_rows "{",".join(reversed(hosts))}" {count}')
    result = subprocess.run(["bash", "-c", "\n".join(script), "test",
                             str(ROOT / "scripts/lib/release-common.sh")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"{rank} {host} 192.0.2.{count - rank} 198.51.100.{count - rank}"
        for rank, host in enumerate(reversed(hosts))]


@pytest.mark.parametrize("count", [7, 8])
def test_custom_hosts_reject_deleted_worlds(count):
    hosts = ",".join(f"spark-{rank}.example" for rank in range(count))
    result = subprocess.run(["bash", "-c", 'set -eu; source "$1"; release_spark_host_rows "$2" "$3"',
                             "test", str(ROOT / "scripts/lib/release-common.sh"), hosts, str(count)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert f"SPARK_HOSTS must name exactly SPARK_COUNT={count} hosts" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("host,lane_a,message", [
    ("spark-5.example", "", "configure SPARK_<rank>_HOST and LANE_A"),
    ("spark-5.example,spark-5.example", "192.0.2.6", "duplicate host"),
    ("spark-5.example,", "192.0.2.6", "comma-separated Spark pool host list"),
    ("spark;bad", "192.0.2.6", "comma-separated Spark pool host list"),
])
def test_explicit_custom_hosts_fail_closed(host, lane_a, message):
    env = {**os.environ, "SPARK_5_HOST": "spark-5.example", "SPARK_5_LANE_A": lane_a}
    result = subprocess.run(["bash", "-c", 'set -eu; source "$1"; release_spark_host_rows "$2" "$3"',
                             "test", str(ROOT / "scripts/lib/release-common.sh"), host,
                             str(len(host.split(",")))], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert message in result.stderr


def test_explicit_custom_host_secondary_rail_is_optional():
    env = {**os.environ, "SPARK_5_HOST": "spark-5.example", "SPARK_5_LANE_A": "192.0.2.6",
           "SPARK_5_LANE_B": ""}
    result = subprocess.run(["bash", "-c", 'set -eu; source "$1"; release_spark_host_rows spark-5.example 1',
                             "test", str(ROOT / "scripts/lib/release-common.sh")], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "0 spark-5.example 192.0.2.6 \n"


def _family_launch_lines(tmp_path: Path, family_config: dict, model: str, keys: str) -> str:
    return _family_launch_result(tmp_path, family_config, model, keys).stderr


@pytest.mark.parametrize("value", [None, "auto", "0", "5", "50%", "all", "max"])
def test_deepseek_v4_honors_explicit_local_expert_limit(tmp_path, value):
    keys = "" if value is None else f"RTX_EXPERT_LAYERS={value}\n"
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v4", "num_hidden_layers": 61}, "test/dsv4", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-dsv4" in line)
    if value is None:
        assert "--local-expert-layers" not in launch and "--rtx-expert-layers" not in launch
    elif value not in ("auto", "max"):
        value = {"all": "61", "50%": "31"}.get(value, value)
        # Whole layer counts keep the flag older images accept.
        assert f"--local-expert-layers {value}" in launch
    else:
        assert f"--rtx-expert-layers {value}" in launch


@pytest.mark.parametrize("value", [None, "auto", "0", "5", "50%", "all", "max"])
def test_glm53_honors_explicit_tp2_expert_limit(tmp_path, value):
    keys = "" if value is None else f"RTX_EXPERT_LAYERS={value}\n"
    result = _family_launch_result(tmp_path, {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 9, "first_k_dense_replace": 3,
    }, "test/glm", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glm " in line)
    assert "--local-expert-layers" not in launch
    if value is None:
        assert "--rtx-expert-layers" not in launch
    else:
        expected = {"all": "6", "50%": "3"}.get(value, value)
        assert f"--rtx-expert-layers {expected}" in launch


@pytest.mark.parametrize("explicit", [False, True])
def test_glm53_units_span_forwards_the_planned_units_waves_and_span(tmp_path, explicit):
    keys = "PREFILL_ROWS=2048\nPREFILL_WAVES=3\nPREFILL_SPAN_ROWS=8192\n" if explicit else ""
    result = _family_launch_result(tmp_path, {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 9, "first_k_dense_replace": 3,
    }, "test/glm", keys)
    assert result.returncode == 0, result.stderr
    plan = next(shlex.split(line) for line in result.stderr.splitlines()
                if "cuteafd plan" in line and "--deployment" in line)
    unit, waves, span = (2048, 3, 8192) if explicit else (4096, 2, 16384)
    if explicit:
        for flag, value in [("--prefill-rows", unit), ("--prefill-waves", waves), ("--prefill-span-rows", span)]:
            assert plan[plan.index(flag) + 1] == str(value)
    assert "--prefill-lanes" not in plan
    serve = next(shlex.split(line) for line in result.stderr.splitlines()
                 if "docker run -d" in line and "serve-glm" in line)
    for flag, value in [("--prefill-rows", unit), ("--prefill-waves", waves), ("--prefill-span-rows", span)]:
        assert serve.count(flag) == 1 and serve[serve.index(flag) + 1] == str(value)
    assert not any("CUTEAFD_PREFILL_SPAN_ROWS" in arg or "CUTEAFD_GLM_PREFILL_LANES" in arg for arg in serve)
    assert f"Resolved prefill: PREFILL_ROWS={unit} PREFILL_WAVES={waves} PREFILL_SPAN_ROWS={span}" in result.stdout


def test_glm53_all_local_two_rtx_launch_has_no_spark_peers(tmp_path):
    result = _family_launch_result(tmp_path, {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3,
    }, "test/glm", "SPARK_COUNT=0\nRTX_GPUS=2\nRTX_EXPERT_LAYERS=all\nSPECULATOR=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glm " in line)
    assert "--split-device 1" in launch
    assert "--rtx-expert-layers 1" in launch
    assert "--peers" not in launch
    assert "cuteafd expertd-native" not in result.stderr
    assert "--local-experts" not in launch
    assert "--prefill-waves 1" in launch and "CUTEAFD_GLM_PREFILL_LANES" not in launch
    assert "--rdma-endpoints" not in result.stderr


@pytest.mark.parametrize("value", [None, "auto", "0", "2", "50%", "all", "max"])
def test_glm_flash_forwards_explicit_rtx_expert_onboarding(tmp_path, value):
    keys = "SPECULATOR=off\n"
    if value is not None:
        keys += f"RTX_EXPERT_LAYERS={value}\n"
    result = _family_launch_result(tmp_path, {
        "model_type": "glm5_next", "num_hidden_layers": 2,
        "mlp_layer_types": ["sparse"] * 2,
        "layer_types": ["linear_attention", "deepseek_sparse_attention"],
    }, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    if value is None:
        assert "--rtx-expert-layers" not in launch
    else:
        expected = {"all": "2", "50%": "1"}.get(value, value)
        assert f"--rtx-expert-layers {expected}" in launch


@pytest.mark.parametrize("value", ["-1", "101%", "half", "5x"])
def test_glm53_rejects_invalid_tp2_expert_limit_before_launch(tmp_path, value):
    result = _family_launch_result(tmp_path, {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3,
    }, "test/glm", f"RTX_EXPERT_LAYERS={value}\n")
    assert result.returncode == 2 and "RTX_EXPERT_LAYERS must be" in result.stderr
    assert "docker run" not in result.stderr and "nest drop-caches" not in result.stderr


@pytest.mark.parametrize("value", [None, "off", "on"])
def test_glm53_rejects_retired_peer_expert_ranges(tmp_path, value):
    keys = "" if value is None else f"RTX_EXPERT_PEER={value}\n"
    result = _family_launch_result(tmp_path, {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3,
    }, "test/glm", keys)
    if value == "on":
        assert result.returncode == 2
        assert "GPU1 whole-layer expert ranges were replaced by TP2 halves" in result.stderr
        assert "docker run" not in result.stderr and "nest drop-caches" not in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glm " in line)
        assert "--peer-expert-ranges" not in launch


@pytest.mark.parametrize("value", [None, "off", "on"])
def test_deepseek_v4_rejects_retired_peer_expert_ranges(tmp_path, value):
    keys = "" if value is None else f"RTX_EXPERT_PEER={value}\n"
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v4"}, "test/dsv4", keys)
    if value == "on":
        assert result.returncode == 2
        assert "GPU1 whole-layer expert ranges were replaced by TP2 halves (v3 P4); remove RTX_EXPERT_PEER" in result.stderr
        assert "docker run" not in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-dsv4" in line)
        assert "--peer-expert-ranges" not in launch


@pytest.mark.parametrize("value", ["-1", "101%", "half", "5x"])
def test_deepseek_v4_rejects_invalid_local_limit_before_launch(tmp_path, value):
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v4"}, "test/dsv4",
                                  f"RTX_EXPERT_LAYERS={value}\n")
    assert result.returncode == 2 and "RTX_EXPERT_LAYERS must be" in result.stderr
    assert "docker run" not in result.stderr and "nest drop-caches" not in result.stderr


@pytest.mark.parametrize("layout,physical_gpus,split", [
    ("RTX_GPUS=1\n", (0, 1), False),
    ("RTX_GPUS=auto\n", (0,), False),
    ("COORDINATOR_GPUS=1\n", (0, 1), False),
    ("COORDINATOR_GPUS=0,1\nCOORDINATOR_SPLIT=off\n", (0, 1), False),
    ("RTX_GPUS=auto\n", (0, 1), True),
    ("RTX_GPUS=2\n", (0, 1), True),
    ("COORDINATOR_GPUS=1,0\n", (0, 1), True),
    ("RTX_GPUS=1\nCOORDINATOR_SPLIT=heads\n", (0, 1), True),
])
@pytest.mark.parametrize("precision", ["", "GLM5_FLASH_KDA_FP8=auto\nGLM5_FLASH_FP8_HEAD=auto\n",
                                        "GLMF_KDA_FP8=auto\nGLMF_FP8_HEAD=auto\n"])
def test_glmf_precision_defaults_follow_serving_split(tmp_path, layout, physical_gpus, split, precision):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  layout + precision,
                                  physical_gpus=physical_gpus)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert ("--split-device" in launch) == split
    # Both layouts default to row128 KDA and an FP8 head; the split adds
    # token-row KDA output ownership.
    assert "--kda-fp8 row128" in launch
    assert "--fp8-head true" in launch
    assert launch.count("--kda-fp8") == launch.count("--fp8-head") == 1
    assert launch.count("--snapshot ") == 1
    assert ("--kda-output-shard --kda-prefill-expanded" in launch) == split


@pytest.mark.parametrize("keys,shard", [("", True), ("GLM5_FLASH_KDA_SPLIT=partials\n", False),
                                        ("GLM5_FLASH_KDA_FP8=off\n", False)])
def test_glmf_split_token_rows_follow_fp8_kda(tmp_path, keys, shard):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "RTX_GPUS=2\n" + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert ("--kda-output-shard" in launch) == shard


@pytest.mark.parametrize("layout", ["RTX_GPUS=1\n", "RTX_GPUS=2\n"])
@pytest.mark.parametrize("prefix", ["GLM5_FLASH", "GLMF"])
@pytest.mark.parametrize("kda,head", [("off", "off"), ("row128", "on"), ("channel", "off")])
def test_glmf_explicit_precision_wins_on_either_layout(tmp_path, layout, prefix, kda, head):
    # Current names also take precedence over conflicting deprecated names.
    keys = "GLMF_KDA_FP8=channel\nGLMF_FP8_HEAD=on\n" if prefix == "GLM5_FLASH" else ""
    keys += f"{prefix}_KDA_FP8={kda}\n{prefix}_FP8_HEAD={head}\n"
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  layout + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert f"--kda-fp8 {kda}" in launch
    assert f"--fp8-head {'true' if head == 'on' else 'false'}" in launch
    assert launch.count("--kda-fp8") == launch.count("--fp8-head") == 1
    assert launch.count("--snapshot ") == 1


@pytest.mark.parametrize("layout,keys,kda,head", [
    ("RTX_GPUS=1\n", "GLM5_FLASH_KDA_FP8=off\n", "off", "true"),
    ("RTX_GPUS=1\n", "GLM5_FLASH_FP8_HEAD=off\n", "row128", "false"),
    ("RTX_GPUS=2\n", "GLM5_FLASH_KDA_FP8=off\n", "off", "true"),
    ("RTX_GPUS=2\n", "GLM5_FLASH_FP8_HEAD=off\n", "row128", "false"),
])
def test_glmf_precision_overrides_are_independent(tmp_path, layout, keys, kda, head):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  layout + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert f"--kda-fp8 {kda}" in launch
    assert f"--fp8-head {head}" in launch


@pytest.mark.parametrize("layout,accepted", [("RTX_GPUS=1\n", True), ("RTX_GPUS=2\n", False)])
def test_glmf_kda_prefill_validation_uses_resolved_precision(tmp_path, layout, accepted):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "GLM5_FLASH_FP8_PREFILL=all\n" + layout)
    assert result.returncode == (0 if accepted else 2), result.stderr
    if accepted:
        assert "--fp8-prefill all" in result.stderr
    else:
        # The split's default token-row KDA output excludes KDA output W8A8.
        assert "GLM5_FLASH_KDA_SPLIT=partials" in result.stderr
        assert "docker run" not in result.stderr


@pytest.mark.parametrize("keys,expected", [
    ("GLM5_FLASH_KDA_FP8=row128\n", ["--kda-fp8 row128"]),
    ("GLMF_KDA_FP8=channel\n", ["--kda-fp8 channel"]),
    ("GLM5_FLASH_FP8_HEAD=on\n", ["--fp8-head true"]),
    ("GLMF_FP8_HEAD=on\n", ["--fp8-head true"]),
    # KDA output W8A8 (kda-o/all) excludes the split's token-row output, so
    # those cases select the partial-sum split path explicitly.
    ("GLM5_FLASH_KDA_FP8=row128\nGLM5_FLASH_KDA_SPLIT=partials\nGLM5_FLASH_FP8_PREFILL=all\n",
     ["--fp8-prefill all"]),
    ("GLMF_KDA_FP8=row128\nGLMF_FP8_PREFILL=mla,kda-in\n", ["--fp8-prefill mla,kda-in"]),
    ("GLM5_FLASH_KDA_FP8=channel\nGLM5_FLASH_KDA_SPLIT=partials\nGLM5_FLASH_FP8_PREFILL=kda-o,ffn\n",
     ["--fp8-prefill kda-o,ffn"]),
])
def test_glmf_single_copy_fp8_options_are_forwarded(tmp_path, keys, expected):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    for option in expected:
        assert option in launch, launch


@pytest.mark.parametrize("key,value,message", [
    ("GLM5_FLASH_FP8_PREFILL", "all", "GLM5_FLASH_KDA_FP8=row128"),
    ("GLMF_FP8_PREFILL", "mla,kda-in", "GLM5_FLASH_KDA_FP8=row128"),
    ("GLM5_FLASH_FP8_PREFILL", "kda-o,ffn", "GLM5_FLASH_KDA_FP8=row128"),
    ("GLM5_FLASH_FP8_HEAD", "maybe", "GLM5_FLASH_FP8_HEAD must be"),
    ("GLM5_FLASH_INDEX_CACHE", "tails", "GLM5_FLASH_INDEX_CACHE must be"),
])
def test_glmf_invalid_fp8_options_fail_before_workers_launch(tmp_path, key, value, message):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf",
                                  f"GLM5_FLASH_KDA_FP8=off\n{key}={value}\n")
    assert result.returncode == 2 and message in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


GLMF_TWO_LAYER = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
                  "layer_types": ["linear_attention", "deepseek_sparse_attention"]}


@pytest.mark.parametrize("keys,forwarded", [
    # auto (unset) forwards nothing: the shared setting applies (tensor over the BF16 head; the FP8 head's
    # own program otherwise), so both default launch lines are unchanged.
    ("", None), ("GLM5_FLASH_TARGET_HEAD=auto\n", None), ("GLM5_FLASH_FP8_HEAD=off\n", None),
    ("GLM5_FLASH_FP8_HEAD=off\nGLM5_FLASH_TARGET_HEAD=auto\n", None),
    # Explicit settings are forwarded; exact stays an explicit diagnostic.
    ("GLM5_FLASH_FP8_HEAD=off\nGLM5_FLASH_TARGET_HEAD=tensor\n", "tensor"),
    ("GLM5_FLASH_FP8_HEAD=off\nGLM5_FLASH_TARGET_HEAD=exact\n", "exact"),
    ("GLM5_FLASH_TARGET_HEAD=exact\n", "exact")])
def test_glmf_target_head_defaults_to_tensor_with_the_bf16_head(tmp_path, keys, forwarded):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "RTX_GPUS=1\n" + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert ("--target-head" in launch) == (forwarded is not None), launch
    if forwarded:
        assert f"--target-head {forwarded}" in launch, launch


def test_glmf_fp8_head_launch_is_identical_with_and_without_the_target_head_key(tmp_path):
    """The FP8-head default config is byte-for-byte unaffected by the target-head default."""
    lines = []
    for index, keys in enumerate(["", "GLM5_FLASH_TARGET_HEAD=auto\n"]):
        root = tmp_path / str(index)
        root.mkdir()
        result = _family_launch_result(root, GLMF_TWO_LAYER, "test/glmf",
                                      "RTX_GPUS=1\n" + keys)
        assert result.returncode == 0, result.stderr
        launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
        assert "--fp8-head true" in launch and "--target-head" not in launch, launch
        lines.append(launch.replace(str(root), "ROOT"))
    assert lines[0] == lines[1], lines


@pytest.mark.parametrize("keys,message", [
    ("GLM5_FLASH_FP8_HEAD=off\nGLM5_FLASH_TARGET_HEAD=fp8\n", "GLM5_FLASH_TARGET_HEAD must be auto, exact or tensor"),
    # The launcher's FP8 head default (auto) is on: an explicit tensor target head needs the BF16 head.
    ("GLM5_FLASH_TARGET_HEAD=tensor\n", "set GLM5_FLASH_FP8_HEAD=off"),
    ("GLM5_FLASH_FP8_HEAD=on\nGLM5_FLASH_TARGET_HEAD=tensor\n", "set GLM5_FLASH_FP8_HEAD=off"),
])
def test_glmf_target_head_rejects_bad_values_and_explicit_tensor_with_the_fp8_head(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf", keys)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("value", [None, "exact", "tensor", "fp8"])
def test_glmf_serving_head_has_no_arithmetic_selection(tmp_path, value):
    keys = "" if value is None else f"GLM5_FLASH_DRAFT_HEAD={value}\n"
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "RTX_GPUS=1\n" + keys)
    if value is None:
        assert result.returncode == 0, result.stderr
        launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
        assert "--draft-head" not in launch, launch
        return
    assert result.returncode == 2, result.stderr
    assert "unused configuration key: GLM5_FLASH_DRAFT_HEAD" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,forwarded", [("", None), ("TARGET_HEAD=tensor\n", "tensor"),
                                           ("TARGET_HEAD=exact\n", "exact")])
def test_target_head_key_is_forwarded_to_the_coordinator_only_when_set(tmp_path, keys, forwarded):
    """TARGET_HEAD (every family's BF16 target head) reaches the coordinator as CUTEAFD_TARGET_HEAD only
    when set: unset launch lines are unchanged and the engine's default (tensor) applies."""

    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "RTX_GPUS=1\n" + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert ("CUTEAFD_TARGET_HEAD" in launch) == (forwarded is not None), launch
    if forwarded:
        assert f"CUTEAFD_TARGET_HEAD={forwarded}" in launch, launch


def test_target_head_key_rejects_bad_values_before_launch(tmp_path):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf", "TARGET_HEAD=fp8\n")
    assert result.returncode == 2 and "TARGET_HEAD must be tensor or exact" in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())



@pytest.mark.parametrize("keys,forwarded", [("", None), ("GLM5_FLASH_DRAFT_LINEAR=w8a16\n", None),
                                           ("GLM5_FLASH_DRAFT_LINEAR=wide\n", "wide"),
                                           ("GLM5_FLASH_DRAFT_LINEAR=w8a8\n", "w8a8")])
def test_glmf_draft_linear_is_forwarded_only_past_w8a16(tmp_path, keys, forwarded):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "RTX_GPUS=1\n" + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    if forwarded is None:
        assert "--draft-linear" not in launch, launch
    else:
        assert f"--draft-linear {forwarded}" in launch and launch.count("--draft-linear") == 1, launch


def test_glmf_draft_linear_rejects_unknown_values_before_launch(tmp_path):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "GLM5_FLASH_DRAFT_LINEAR=w4a16\n")
    assert result.returncode == 2 and "GLM5_FLASH_DRAFT_LINEAR must be w8a16, wide or w8a8" in result.stderr, \
        result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,env", [
    ("", []),
    ("GLM5_FLASH_EXL3_WORKER_PATH=async\n", []),
    ("GLM5_FLASH_EXL3_WORKER_PATH=blocking\n", ["-e CUTEAFD_EXL3_WORKER_PATH=blocking"]),
    ("GLM5_FLASH_EXL3_ROUTE_DUMP=/data/routes-1\n",
     ["-v /data/routes-1:/data/routes-1", "-e CUTEAFD_EXL3_ROUTE_DUMP=/data/routes-1/routes",
      "-e CUTEAFD_EXL3_ROUTE_DUMP_CALLS=200000"]),
    ("GLM5_FLASH_EXL3_ROUTE_DUMP=/data/r\nGLM5_FLASH_EXL3_ROUTE_DUMP_CALLS=5000\n",
     ["-e CUTEAFD_EXL3_ROUTE_DUMP=/data/r/routes", "-e CUTEAFD_EXL3_ROUTE_DUMP_CALLS=5000"]),
])
def test_glmf_exl3_worker_env_reaches_only_the_spark_workers(tmp_path, keys, env):
    """The Spark worker's host path and route capture are worker environment: forwarded to
    every Spark rank's container, never to the coordinator; the defaults add nothing."""
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    workers = [line for line in lines if "cuteafd expertd-native" in line]
    coordinator = next(line for line in lines if "cuteafd serve-glmf" in line)
    assert workers and "CUTEAFD_EXL3_" not in coordinator
    for worker in workers:
        for flag in env:
            assert f" {flag} " in worker, worker
        if not env:
            assert "CUTEAFD_EXL3_WORKER_PATH" not in worker and "CUTEAFD_EXL3_ROUTE_DUMP" not in worker


@pytest.mark.parametrize("keys,message", [
    ("GLM5_FLASH_EXL3_WORKER_PATH=spin\n", "GLM5_FLASH_EXL3_WORKER_PATH must be async or blocking"),
    ("GLM5_FLASH_EXL3_ROUTE_DUMP=routes\n", "GLM5_FLASH_EXL3_ROUTE_DUMP must be an absolute directory"),
    ("GLM5_FLASH_EXL3_ROUTE_DUMP=/data/r x\n", "GLM5_FLASH_EXL3_ROUTE_DUMP must be an absolute directory"),
    ("GLM5_FLASH_EXL3_ROUTE_DUMP=/data/r\nGLM5_FLASH_EXL3_ROUTE_DUMP_CALLS=0\n",
     "GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS must be a positive call count"),
    ("GLM5_FLASH_EXL3_ROUTE_DUMP=/data/r\nSPARK_COUNT=0\n", "GLM5_FLASH_EXL3_ROUTE_DUMP records Spark expert calls"),
])
def test_glmf_exl3_worker_env_rejects_bad_requests_before_launch(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf", keys)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,compact", [("", False), ("GLM5_FLASH_INDEX_CACHE=keys\n", False),
                                         ("GLM5_FLASH_INDEX_CACHE=compact\n", True)])
def test_glmf_index_cache_is_forwarded_only_when_compact(tmp_path, keys, compact):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert ("--index-cache compact" in launch) == compact, launch
    assert "--index-cache keys" not in launch


@pytest.mark.parametrize("keys,expected,absent", [
    ("", [], ["--prefill-waves", "--prefill-rows", "--headroom-gib", "--graph-budget-mib"]),
    ("PREFILL_WAVES=4\nPREFILL_ROWS=2048\n",
     ["--prefill-waves 4", "--prefill-rows 2048"], []),
    ("PREFILL_WAVES=1\n", ["--prefill-waves 1"], ["--prefill-rows"]),
    ("GLM5_FLASH_HEADROOM_GIB=1\n", ["--headroom-gib 1"], ["--prefill-waves"]),
    ("GLM5_FLASH_GRAPH_BUDGET_MIB=512\n", ["--graph-budget-mib 512"], ["--headroom-gib"]),
])
def test_glmf_units_and_headroom_are_forwarded_only_when_set(tmp_path, keys, expected, absent):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    for option in expected:
        assert option in launch, launch
    for option in absent:
        assert option not in launch, launch


@pytest.mark.parametrize("key,value,message", [
    ("PREFILL_WAVES", "0", "PREFILL_WAVES must be 1 to 4"),
    ("PREFILL_WAVES", "5", "PREFILL_WAVES must be 1 to 4"),
    ("PREFILL_ROWS", "2000", "PREFILL_ROWS must be a multiple of 64"),
    ("PREFILL_ROWS", "8192", "PREFILL_ROWS must be a multiple of 64"),
    ("PREFILL_ROWS", "0", "PREFILL_ROWS must be a multiple of 64"),
    ("GLM5_FLASH_HEADROOM_GIB", "-1", "GLM5_FLASH_HEADROOM_GIB must be a non-negative size"),
    ("GLM5_FLASH_HEADROOM_GIB", "1GiB", "GLM5_FLASH_HEADROOM_GIB must be a non-negative size"),
    ("GLM5_FLASH_GRAPH_BUDGET_MIB", "0", "GLM5_FLASH_GRAPH_BUDGET_MIB must be a positive whole number"),
])
def test_glmf_invalid_units_or_headroom_fail_before_workers_launch(tmp_path, key, value, message):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf", f"{key}={value}\n")
    assert result.returncode == 2 and message in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


GLMF_CONFIG = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
               "layer_types": ["linear_attention", "deepseek_sparse_attention"]}


def test_glmf_default_launch_passes_no_admission_option(tmp_path):
    """Packed admission prefill is opt-in; the default keeps one pass per prompt."""
    result = _family_launch_result(tmp_path, GLMF_CONFIG, "test/glmf", "")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert "--prefill-batch" not in launch, launch


@pytest.mark.parametrize("keys,expected,absent", [
    ("GLM5_FLASH_PREFILL_BATCH=off\n", (), ("--prefill-batch",)),
    ("GLM5_FLASH_PREFILL_BATCH=on\n", ("--prefill-batch",), ()),
])
def test_glmf_admission_keys_are_forwarded(tmp_path, keys, expected, absent):
    result = _family_launch_result(tmp_path, GLMF_CONFIG, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    for option in expected:
        assert option in launch, (option, launch)
    for option in absent:
        assert option not in launch, (option, launch)


@pytest.mark.parametrize("keys,message", [
    ("GLM5_FLASH_PREFILL_BATCH=yes\n", "GLM5_FLASH_PREFILL_BATCH must be"),
])
def test_glmf_admission_keys_reject_bad_values_before_launch(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, GLMF_CONFIG, "test/glmf", keys)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,expected,absent", [
    ("", (), ("CUTEAFD_GLMF_ROUTE_RING_CHECK", "CUTEAFD_GLMF_DRAFT_POLICY")),
    ("GLM5_FLASH_ROUTE_RING_CHECK=on\n", ("CUTEAFD_GLMF_ROUTE_RING_CHECK=1",), ("CUTEAFD_GLMF_DRAFT_POLICY",)),
])
def test_glmf_draft_policy_keys_reach_the_coordinator(tmp_path, keys, expected, absent):
    """The shared draft policy has no switch; the route ring check is diagnostic."""
    result = _family_launch_result(tmp_path, GLMF_CONFIG, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    for option in expected:
        assert option in launch, (option, launch)
    for option in absent:
        assert option not in launch, (option, launch)


@pytest.mark.parametrize("keys,message", [
    ("GLM5_FLASH_DRAFT_POLICY=shared\n", "unknown configuration key: GLM5_FLASH_DRAFT_POLICY"),
    ("GLM5_FLASH_DRAFT_POLICY=cycle\n", "unknown configuration key: GLM5_FLASH_DRAFT_POLICY"),
    ("GLM5_FLASH_ROUTE_RING_CHECK=yes\n", "GLM5_FLASH_ROUTE_RING_CHECK must be"),
    ("GLM5_FLASH_DRAFT_TRACE=yes\n", "unknown configuration key: GLM5_FLASH_DRAFT_TRACE"),
    ("GLM5_FLASH_REMOTE_DRAFTER=yes\n", "GLM5_FLASH_REMOTE_DRAFTER must be"),
])
def test_glmf_draft_policy_keys_reject_bad_values_before_launch(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, GLMF_CONFIG, "test/glmf", keys)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,forwarded", [("", None), ("GLM5_FLASH_KDA_STATE=f32\n", None),
                                             ("GLM5_FLASH_KDA_STATE=bf16\n", "bf16"),
                                             ("GLM5_FLASH_KDA_STATE=bf16-tile\n", "bf16-tile")])
@pytest.mark.parametrize("projection", ("off", "row128", "channel"))
def test_glmf_kda_state_is_forwarded_with_either_kda_storage(tmp_path, keys, forwarded, projection):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  f"RTX_GPUS=1\nGLM5_FLASH_KDA_FP8={projection}\n" + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    if forwarded is None:
        assert "--kda-state" not in launch
    else:
        assert f"--kda-state {forwarded}" in launch and launch.count("--kda-state") == 1


@pytest.mark.parametrize("keys,message", [
    ("RTX_GPUS=2\nGLM5_FLASH_KDA_FP8=off\nGLM5_FLASH_KDA_STATE=bf16\n", "without a head split"),
    ("RTX_GPUS=1\nGLM5_FLASH_KDA_FP8=off\nGLM5_FLASH_KDA_STATE=fp16\n", "GLM5_FLASH_KDA_STATE must be"),
])
def test_glmf_kda_state_rejects_unsupported_layouts_before_launch(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  keys)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,schedule", [("", None), ("GLM5_FLASH_EXL3_SCHEDULE=default\n", None),
                                            ("GLM5_FLASH_EXL3_SCHEDULE=gb10\n", "gb10")])
def test_glmf_exl3_schedule_reaches_only_the_spark_workers(tmp_path, keys, schedule):
    """The Spark EXL3 decode schedule is a worker option: gb10 is forwarded to every Spark
    rank as --exl3-schedule gb10, the default passes nothing, the coordinator never sees it."""
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    worker = next(line for line in lines if "cuteafd expertd-native" in line)
    coordinator = next(line for line in lines if "cuteafd serve-glmf" in line)
    assert "--exl3-schedule" not in coordinator
    if schedule is None:
        assert "--exl3-schedule" not in worker, worker
    else:
        assert f"--exl3-schedule {schedule} " in worker, worker


@pytest.mark.parametrize("keys,message", [
    ("GLM5_FLASH_EXL3_SCHEDULE=fast\n", "GLM5_FLASH_EXL3_SCHEDULE must be default or gb10"),
    ("GLM5_FLASH_EXL3_SCHEDULE=gb10\nSPARK_COUNT=0\n", "GLM5_FLASH_EXL3_SCHEDULE=gb10 is a Spark expert schedule"),
])
def test_glmf_exl3_schedule_rejects_bad_requests_before_launch(tmp_path, keys, message):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf", keys)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,expected", [("", None), ("GLM5_FLASH_PREFIX_MARKS=arena\n", "arena"),
                                           ("GLM5_FLASH_PREFIX_MARKS=pool\n", "pool")])
def test_glmf_prefix_marks_are_forwarded(tmp_path, keys, expected):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    if expected is None:
        assert "--prefix-marks" not in launch, launch
    else:
        assert f"--prefix-marks {expected}" in launch and launch.count("--prefix-marks") == 1, launch


@pytest.mark.parametrize("keys,expected", [
    ("GLM5_FLASH_PREFIX_MARKS=pool\n", "auto"),
    ("GLM5_FLASH_PREFIX_MARKS=pool\nHOST_CACHE_BYTES=64GiB\n", "64GiB"),
    ("GLM5_FLASH_PREFIX_MARKS=pool\nHOST_CACHE_BYTES=0\n", None),
    ("GLM5_FLASH_PREFIX_MARKS=arena\n", "auto"),
    ("", "auto"),
])
def test_glmf_marks_default_to_host_retention(tmp_path, keys, expected):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    if expected is None:
        assert "--host-cache-bytes" not in launch, launch
    else:
        assert launch.count("--host-cache-bytes") == 1 and f"--host-cache-bytes {expected}" in launch, launch


@pytest.mark.parametrize("keys,expected", [("", None), ("GLM5_FLASH_PREFIX_MARKS=arena\n", "arena"),
                                           ("GLM5_FLASH_PREFIX_MARKS=pool\n", "pool")])
def test_glmf_prefix_marks_reach_the_encoder_plan(tmp_path, keys, expected):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"], "vision_config": {"depth": 24}}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": {"kind": {"kind": "spark", "rank": 0}, "replicas": []}}
    result = _family_launch_result(tmp_path, config, "zai-org/GLM-5.3-Flash",
                                  f"RTX_GPUS=1\nSPECULATOR=off\nVISION=auto\n{keys}", encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    planner = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line and "--deployment" in line)
    for command in [launch, planner]:
        if expected is None:
            assert "--prefix-marks" not in command, command
        else:
            assert command.count("--prefix-marks") == 1 and f"--prefix-marks {expected}" in command, command


def test_glmf_prefix_marks_reject_unknown_stores_before_launch(tmp_path):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "GLM5_FLASH_PREFIX_MARKS=host\n")
    assert result.returncode == 2 and "GLM5_FLASH_PREFIX_MARKS must be arena or pool" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,shared", [("", True), ("GLM5_FLASH_REPLAY_RECORDS=own\n", False),
                                        ("GLM5_FLASH_REPLAY_RECORDS=shared\n", True)])
def test_glmf_replay_records_are_forwarded_only_when_shared(tmp_path, keys, shared):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "RTX_GPUS=1\n" + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert ("--replay-records shared" in launch) == shared, launch
    assert "--replay-records own" not in launch


@pytest.mark.parametrize("keys,message", [
    ("RTX_GPUS=2\nGLM5_FLASH_REPLAY_RECORDS=shared\n", "without a head split"),
    ("RTX_GPUS=1\nGLM5_FLASH_REPLAY_RECORDS=host\n", "GLM5_FLASH_REPLAY_RECORDS must be own or shared"),
])
def test_glmf_replay_records_reject_unsupported_layouts_before_launch(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  keys)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,forwarded", [("", False), ("GLM5_FLASH_DECODE_ROWS=64\n", False),
                                             ("GLM5_FLASH_DECODE_ROWS=128\n", True)])
def test_glmf_decode_rows_are_forwarded_only_at_128(tmp_path, keys, forwarded):
    """GLM5_FLASH_DECODE_ROWS=128 reaches the coordinator as --decode-rows 128 (the wide decode
    programs); 64, the default, passes nothing; no Spark worker sees it."""
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "RTX_GPUS=1\n" + keys)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    launch = next(line for line in lines if "cuteafd serve-glmf" in line)
    assert ("--decode-rows 128" in launch) == forwarded and launch.count("--decode-rows") == int(forwarded)
    assert not any("--decode-rows" in line for line in lines if "cuteafd expertd-native" in line)


def test_glmf_wide_decode_rows_take_shared_replay_records(tmp_path):
    """128-row steps keep their records in the prefill scratch too: both keys reach the coordinator."""
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "RTX_GPUS=1\nGLM5_FLASH_DECODE_ROWS=128\n"
                                  "GLM5_FLASH_REPLAY_RECORDS=shared\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert "--decode-rows 128" in launch and "--replay-records shared" in launch, launch


@pytest.mark.parametrize("keys,message", [
    ("RTX_GPUS=2\nGLM5_FLASH_DECODE_ROWS=128\n", "a head split takes 64"),
    ("RTX_GPUS=1\nGLM5_FLASH_DECODE_ROWS=127\n", "GLM5_FLASH_DECODE_ROWS must be 64 or 128"),
    ("RTX_GPUS=1\nGLM5_FLASH_DECODE_ROWS=wide\n", "GLM5_FLASH_DECODE_ROWS must be 64 or 128"),
])
def test_glmf_decode_rows_reject_unsupported_layouts_before_launch(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  keys)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


def test_probe_dump_root_is_mounted_for_remote_row_dumps(tmp_path):
    root = tmp_path / "dumps"
    root.mkdir()
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  f"RTX_GPUS=1\nPROBE_DUMP_ROOT={root}\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert f"-v {root}:{root} -e CUTEAFD_PROBE_DUMP_ROOT={root}" in launch
    result = _family_launch_result(tmp_path / "unset", SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "RTX_GPUS=1\n")
    assert result.returncode == 0 and "CUTEAFD_PROBE_DUMP_ROOT" not in result.stderr
    result = _family_launch_result(tmp_path / "missing", SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  f"PROBE_DUMP_ROOT={tmp_path}/absent\n")
    assert result.returncode == 2 and "PROBE_DUMP_ROOT must be" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


def test_glmf_kda_rejects_invalid_conversion_before_launch(tmp_path):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf",
                                  "GLM5_FLASH_KDA_FP8=invalid\n")
    assert result.returncode == 2 and "GLM5_FLASH_KDA_FP8 must be" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("family", ["glm5", "glm5_flash"])
@pytest.mark.parametrize("selection, expected", [(None, None), ("auto", None), ("on", "true"), ("off", "false")])
def test_glm_drafter_quantization_is_explicit(tmp_path, family, selection, expected):
    config = ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}
              if family == "glm5" else SPLIT_CONFIGS["glm5_flash"])
    model = "test/model" if family == "glm5" else "zai-org/GLM-5.3-Flash"
    keys = f"SPECULATOR=dflash2\nSPECULATOR_MODEL_ID={model}\n"
    if selection is not None:
        keys += f"SPECULATOR_FP8={selection}\n"
    result = _family_launch_result(tmp_path, config, model, keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    if expected is None:
        assert "--draft-fp8" not in launch
    else:
        assert f"--draft-fp8 {expected}" in launch


@pytest.mark.parametrize("family", ["glm5", "glm5_flash", "mimo_v2"])
@pytest.mark.parametrize("selection, expected", [(None, "fp8"), ("bf16", "bf16"), ("fp8", "fp8")])
def test_drafter_kv_format_is_independent_of_target_kv(tmp_path, family, selection, expected):
    configs = {
        "glm5": {"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3},
        "glm5_flash": SPLIT_CONFIGS["glm5_flash"],
        "mimo_v2": {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
    }
    keys = ""
    if selection is not None:
        keys += f"DRAFT_KV_CACHE={selection}\n"
    result = _family_launch_result(tmp_path, configs[family], "test/model", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert f"--draft-kv-cache {expected}" in launch


def test_invalid_drafter_kv_rejects_before_starting_containers(tmp_path):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/model",
                                  "DRAFT_KV_CACHE=invalid\n")
    assert result.returncode == 2, result.stderr
    assert "DRAFT_KV_CACHE must be bf16 or fp8" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("family", ["glm5", "glm5_flash"])
def test_glm_drafter_context_slots_and_batch_are_independent(tmp_path, family):
    config = ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}
              if family == "glm5" else SPLIT_CONFIGS["glm5_flash"])
    model = "test/model" if family == "glm5" else "zai-org/GLM-5.3-Flash"
    result = _family_launch_result(tmp_path, config, model,
                                  f"SPECULATOR=dflash2\nSPECULATOR_MODEL_ID={model}\n"
                                  "DRAFT_CONTEXT_SLOTS=20\nDRAFT_SEQUENCES=16\nSPECULATOR_FP8=on\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    for argument in ("--draft-context-slots 20", "--draft-sequences 16", "--draft-fp8 true"):
        assert argument in launch


def test_invalid_glm_drafter_quantization_rejects_before_starting_containers(tmp_path):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "zai-org/GLM-5.3-Flash",
                                  "SPECULATOR=dflash2\nSPECULATOR_MODEL_ID=zai-org/GLM-5.3-Flash\nSPECULATOR_FP8=bogus\n")
    assert result.returncode == 2, result.stderr
    assert "SPECULATOR_FP8 must be auto, on or off" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("mode", ["bf16", "bf16-decode"])
@pytest.mark.parametrize("store, geometry, ranks, hidden", [("fp8", "mimo", 4, 4096), ("mxfp4", "mimop", 6, 6144),
                                                        ("mxfp4", "mimof", 2, 4096), ("mxfp4", "mimof", 4, 4096)])
def test_mimo_expert_input_preflights_every_rank_before_serving(tmp_path, mode, store, geometry, ranks, hidden):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1],
              "hidden_size": hidden, "quantization_config": {"store_dtype": store}}
    keys = f"EXPERT_INPUT={mode}\nSPARK_COUNT={ranks}\nSPARK_EXPERT_DOCKER_INFERENCE=spark:test\n"
    keys += "".join(f"SPARK_{r}_HOST=h{r}\nSPARK_{r}_LANE_A=10.0.0.{r + 1}\n" for r in range(ranks))
    result = _family_launch_result(tmp_path, config, "test/mimo", keys)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    checks = [line for line in lines if "docker run --rm -i --entrypoint python3" in line]
    assert len(checks) == ranks
    for rank, check in enumerate(checks):
        assert f"ssh h{rank} " in check and f"- {geometry} tp{ranks} 4096" in check
    assert max(lines.index(check) for check in checks) < next(i for i, line in enumerate(lines) if "docker run -d" in line)
    assert f"--expert-input {mode}" in result.stderr


def test_mimo_unavailable_bf16_package_keeps_existing_containers(tmp_path):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(tmp_path, config, "test/mimo",
                                  "EXPERT_INPUT=bf16-decode\nSPARK_EXPERT_DOCKER_INFERENCE=spark:test\n",
                                  preflight_error=True, restart=True)
    assert result.returncode == 2 and "cannot serve EXPERT_INPUT=bf16-decode" in result.stderr
    assert "CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES=mimo" in result.stderr
    assert "docker rm" not in result.stderr and "docker run -d" not in result.stderr
    assert "nest drop-caches" not in result.stderr


@pytest.mark.parametrize("keys, error", [("EXPERT_INPUT=garbage\n", "must be fp8"),
                                        ("EXPERT_INPUT=bf16-decode\nSPARK_COUNT=0\n", "requires Spark experts")])
def test_mimo_expert_input_rejects_invalid_modes_before_launch(tmp_path, keys, error):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(tmp_path, config, "test/mimo", keys)
    assert result.returncode == 2 and error in result.stderr
    assert "docker run" not in result.stderr


def test_expert_input_is_opt_in_and_mimo_only(tmp_path):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    for mode in ("", "fp8"):
        text = _family_launch_lines(tmp_path / (mode or "default"), config, "test/mimo",
                                   f"EXPERT_INPUT={mode}\n")
        assert '"python3"' not in text
        serve_line = next(line for line in text.splitlines() if "serve-mimo" in line)
        assert ("--expert-input fp8" in serve_line) == bool(mode)
    other = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
             "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path / "other", other, "test/glm", "EXPERT_INPUT=bf16-decode\n")
    assert result.returncode == 2 and "applies to MiMo" in result.stderr
    assert "docker run" not in result.stderr


@pytest.mark.parametrize("key", ["SPECULATOR_FP8"])
@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("auto", None),
                                            ("on", "true"), ("off", "false")])
def test_mimo_drafter_precision_preserves_auto_and_forwards_explicit_conversion(tmp_path, key, value, expected):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    keys = "SPECULATOR=dflash2\n"
    if value is not None:
        keys += f"{key}={value}\n"
    result = _family_launch_result(tmp_path, config, "test/mimo", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    if expected is None:
        assert "--draft-fp8" not in launch
        assert ("--draft-representation checkpoint" in launch) == (value == "auto")
    else:
        assert f"--draft-fp8 {expected}" in launch


def test_mimo_drafter_context_override_reaches_serving(tmp_path):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(tmp_path, config, "test/mimo",
                                  "SPECULATOR=dflash2\nDRAFT_CONTEXT_SLOTS=25\nDRAFT_SEQUENCES=8\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "--draft-context-slots 25" in launch
    assert "--draft-sequences 8" in launch


def test_mimo_prefix_budget_matches_serving_and_encoder_plan(tmp_path):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
              "moe_layer_freq": [0, 1], "vision_config": {}}
    config["vision_config"] = {"model_type": "test"}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": {"kind": {"kind": "off"}, "replicas": []}}
    result = _family_launch_result(tmp_path, config, "test/mimo",
                                  "SPECULATOR=off\nVISION=auto\nPREFIX_CACHE_ENTRIES=24\nPREFIX_CACHE_MARK_MIB=512\n",
                                  encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    planner = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line and "--deployment" in line)
    for command in [launch, planner]:
        assert "--prefix-cache-entries 24" in command
        assert "--prefix-cache-mark-mib 512" in command


@pytest.mark.parametrize("policy, expected", [(None, None), ("auto", None), ("checkpoint", "checkpoint")])
def test_mimo_weight_policy_is_resolved_by_runtime_and_explicit_checkpoint_is_forwarded(tmp_path, policy, expected):
    config = {"model_type": "mimo_v2", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    keys = "" if policy is None else f"MIMO_WEIGHT_POLICY={policy}\n"
    result = _family_launch_result(tmp_path, config, "arbitrary/local-mimo", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "--fp8-head" not in launch and "--fp8-o-proj" not in launch
    if expected is None:
        assert "--weight-policy" not in launch
    else:
        assert f"--weight-policy {expected}" in launch


@pytest.mark.parametrize("quota", [None, "4MiB", "0"])
@pytest.mark.parametrize("vision", ["off", "auto"])
@pytest.mark.parametrize("checkpoint,serve", [("mimo_flash", "serve-mimo"), ("qwen4", "serve-qwen4"),
                                              ("glmf", "serve-glmf")])
def test_embedding_cache_quota_is_explicit_only(tmp_path, quota, vision, checkpoint, serve):
    config = (SPLIT_CONFIGS[checkpoint] if checkpoint != "glmf" else
              {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
               "layer_types": ["linear_attention", "deepseek_sparse_attention"]})
    keys = f"VISION={vision}\nSPECULATOR=off\n" + ("" if quota is None else f"MEDIA_CACHE_BYTES={quota}\n")
    model = "zai-org/GLM-5.3-Flash" if checkpoint == "glmf" else f"test/{checkpoint}"
    result = _family_launch_result(tmp_path, config, model, keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if f"cuteafd {serve}" in line)
    if quota is None or vision == "off":
        assert "--media-cache-bytes" not in launch
    else:
        assert f"--media-cache-bytes {quota}" in launch


@pytest.mark.parametrize("source_kind", ["hf", "hub_snapshot", "external_snapshot"])
def test_glm_vision_template_override_is_explicit_and_coordinator_only(tmp_path, source_kind):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2, "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    source = "test/model"
    expected = source
    if source_kind == "hub_snapshot":
        source = str(tmp_path / "hf/hub/models--test--model/snapshots/abc")
        expected = source
    elif source_kind == "external_snapshot":
        source = str(tmp_path / "vendor")
        Path(source).mkdir()
        expected = source
    result = _family_launch_result(tmp_path, config, "test/model",
                                  f"SPECULATOR=off\nVISION=auto\nCHAT_TEMPLATE_FROM={source}\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert f"--chat-template-from {expected}" in launch
    if source_kind == "external_snapshot":
        assert f"-v {source}:{source}:ro" in launch
    worker = next(line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line)
    assert "--chat-template-from" not in worker
    if source_kind != "hub_snapshot":
        assert source not in worker


def test_text_only_glm_ignores_template_override(tmp_path):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2, "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/model",
                                  "SPECULATOR=off\nVISION=off\nCHAT_TEMPLATE_FROM=missing/vendor\n")
    assert result.returncode == 0, result.stderr
    assert "--chat-template-from" not in result.stderr and "missing/vendor" not in result.stderr


@pytest.mark.parametrize("source", ["../vendor", "vendor/../flash", "vendor/..", "missing/vendor"])
def test_glm_template_override_invalid_source_refused_before_workers(tmp_path, source):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2, "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/model",
                                  f"SPECULATOR=off\nVISION=auto\nCHAT_TEMPLATE_FROM={source}\n")
    assert result.returncode == 2, result.stderr
    assert "CHAT_TEMPLATE_FROM" in result.stderr or "missing snapshot" in result.stderr
    assert "cuteafd expertd-native" not in result.stderr and "cuteafd serve-glmf" not in result.stderr


@pytest.mark.parametrize("key, option", [("MIMO_FP8_HEAD", "--fp8-head"), ("MIMO_FP8_O_PROJ", "--fp8-o-proj")])
@pytest.mark.parametrize("value, expected", [("auto", None), ("on", "true"), ("off", "false")])
def test_mimo_explicit_target_format_overrides_are_forwarded(tmp_path, key, option, value, expected):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(tmp_path, config, "test/mimo", f"{key}={value}\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    if expected is None:
        assert option not in launch
    else:
        assert f"{option} {expected}" in launch


@pytest.mark.parametrize("key", ["MIMO_WEIGHT_POLICY", "MIMO_FP8_HEAD", "MIMO_FP8_O_PROJ"])
def test_invalid_mimo_weight_policy_rejects_before_services(tmp_path, key):
    config = {"model_type": "mimo_v2", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(tmp_path, config, "test/mimo", f"{key}=bogus\n")
    assert result.returncode != 0
    assert "must be" in result.stderr and "docker run" not in result.stderr


def test_mimo_invalid_drafter_precision_rejects_before_launch(tmp_path):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(tmp_path, config, "test/mimo", "SPECULATOR=dflash2\nSPECULATOR_FP8=garbage\n")
    assert result.returncode == 2 and "must be auto, on or off" in result.stderr
    assert "docker run" not in result.stderr


SPLIT_CONFIGS = {
    "qwen4": {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
               "mtp_num_hidden_layers": 1, "layer_types": ["linear_attention", "full_attention"]}},
    "glm5_flash": {"model_type": "glm5_next", "num_hidden_layers": 2,
                   "mlp_layer_types": ["sparse"] * 2,
                   "layer_types": ["linear_attention", "deepseek_sparse_attention"]},
    "mimo_flash": {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
    "mimo_pro": {"model_type": "mimo_v2", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
}


@pytest.mark.parametrize("family_config", [
    SPLIT_CONFIGS["mimo_flash"], SPLIT_CONFIGS["mimo_pro"], SPLIT_CONFIGS["qwen4"],
    SPLIT_CONFIGS["glm5_flash"], {"model_type": "glm_moe_dsa", "num_hidden_layers": 4,
                                "first_k_dense_replace": 3}, {"model_type": "deepseek_v4"},
])
@pytest.mark.parametrize("setting", ["", "FULL_PREFILL_LOGITS=off\n", "FULL_PREFILL_LOGITS=on\n"])
def test_full_prefill_logits_is_one_shared_opt_in(tmp_path, family_config, setting):
    keys = "SPECULATOR=off\n" + setting
    result = _family_launch_result(tmp_path, family_config, "test/model", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert launch.count("--full-prefill-logits") == int("=on" in setting)


def test_full_prefill_logits_rejects_invalid_setting_before_launch(tmp_path):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["mimo_flash"], "test/model",
                                  "FULL_PREFILL_LOGITS=true\n")
    assert result.returncode == 2, result.stderr
    assert "FULL_PREFILL_LOGITS must be on or off" in result.stderr
    assert "docker run" not in result.stderr


@pytest.mark.parametrize("checkpoint", ["qwen4"])
@pytest.mark.parametrize("keys", ["RTX_GPUS=2\n", "COORDINATOR_GPUS=0,1\n", "COORDINATOR_SPLIT=heads\n"])
def test_explicit_split_without_kernels_serves_from_the_first_gpu(
        tmp_path: Path, checkpoint: str, keys: str) -> None:
    model = "zai-org/GLM-5.3-Flash" if checkpoint == "glm5_flash" else "test/model"
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS[checkpoint], model, keys)
    assert result.returncode == 0, result.stderr
    assert "has no head split yet" in result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert "nvidia.com/gpu=0" in launch and "--split-device" not in launch


@pytest.mark.parametrize("checkpoint", ["qwen4"])
def test_auto_keeps_unsupported_checkpoint_on_one_gpu(tmp_path: Path, checkpoint: str) -> None:
    model = "zai-org/GLM-5.3-Flash" if checkpoint == "glm5_flash" else "test/model"
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS[checkpoint], model, "")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert "nvidia.com/gpu=0" in launch and "--split-device" not in launch
    assert "auto selected GPU 0 alone" in result.stderr


@pytest.mark.parametrize("keys,physical_gpus,ranks,dual", [
    ("", (0, 1), 0, True),
    ("RTX_GPUS=2\n", (0, 1), 0, True),
    ("RTX_GPUS=1\n", (0, 1), 0, False),
    ("COORDINATOR_SPLIT=off\n", (0, 1), 0, False),
    ("", (0,), 0, False),
    ("", (0, 1), 1, False),
])
def test_qwen_default_matches_spark_free_coordinator_layout(tmp_path: Path, keys: str,
                                                           physical_gpus: tuple[int, ...],
                                                           ranks: int, dual: bool) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  keys + f"SPARK_COUNT={ranks}\nSPECULATOR=off\n",
                                  physical_gpus=physical_gpus)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert ("--split-device" in launch) == dual
    assert ("CUTEAFD_ATTENTION_PLACEMENT=layers" in launch) == dual
    assert ("--local-experts" in launch) == (ranks == 0)
    assert "--prefill-rows 4096" in launch
    assert "--prefill-waves 1" in launch
    if dual:
        assert "--rtx-expert-layers all" in launch
        preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
        assert "--attention-placement layers" in preflight
        assert "--rtx-expert-layers all" in preflight


@pytest.mark.parametrize("rows,waves", [(512, 1), (1024, 2), (2048, 2), (4096, 2)])
def test_qwen_prefill_shape_reaches_plan_and_server(tmp_path, rows, waves):
    keys = ("RTX_GPUS=1\nSPARK_COUNT=4\nSPECULATOR=off\n"
            f"PREFILL_ROWS={rows}\nPREFILL_WAVES={waves}\n")
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model", keys)
    assert result.returncode == 0, result.stderr
    planner = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    server = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    for command in (planner, server):
        assert f"--prefill-rows {rows}" in command
        assert f"--prefill-waves {waves}" in command
    assert "--split-device" not in server


@pytest.mark.parametrize("keys", ["RTX_GPUS=2\n", "COORDINATOR_GPUS=1,0\n", "COORDINATOR_SPLIT=heads\n"])
def test_qwen_explicit_spark_free_dual_passes_both_owners(tmp_path: Path, keys: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  keys + "SPARK_COUNT=0\nEXPERT_BACKEND=local\nSPECULATOR=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--split-device" in launch and "--local-experts" in launch
    assert "nvidia.com/gpu=0" in launch and "nvidia.com/gpu=1" in launch
    assert "has no head split yet" not in result.stderr



@pytest.mark.parametrize("onboard", ["auto", "all", "max", "100%"])
def test_qwen_dual_forwards_shared_onboard_policy(tmp_path: Path, onboard: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  f"RTX_GPUS=2\nSPARK_COUNT=0\nEXPERT_BACKEND=local\nSPECULATOR=off\nRTX_EXPERT_LAYERS={onboard}\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    expected = {"all": "2", "100%": "2"}.get(onboard, onboard)
    assert f"--rtx-expert-layers {expected}" in launch
    assert "CUTEAFD_ATTENTION_PLACEMENT=layers" in launch


@pytest.mark.parametrize("family", ["qwen4", "glm5_flash", "mimo_flash", "mimo_pro"])
@pytest.mark.parametrize("owners", ["0,1", "1,0"])
@pytest.mark.parametrize("encoder_owner", [0, 1])
def test_dual_maps_encoder_owner_to_visible_device(tmp_path: Path, family: str,
                                                 owners: str, encoder_owner: int) -> None:
    config = {**SPLIT_CONFIGS[family], "vision_config": {"depth": 24}}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 0,
            "encoder_plan_hash": "ab" * 32,
            "encoder": {"kind": {"kind": "rtx", "gpu": encoder_owner}, "replicas": []}}
    keys = (f"COORDINATOR_GPUS={owners}\nSPARK_COUNT=0\nEXPERT_BACKEND=local\n"
            "SPECULATOR=off\nVISION=auto\nAUDIO=off\n")
    result = _family_launch_result(tmp_path, config, "test/model", keys, encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    serve = {"qwen4": "serve-qwen4", "glm5_flash": "serve-glmf",
             "mimo_flash": "serve-mimo", "mimo_pro": "serve-mimo"}[family]
    launch = next(line for line in result.stderr.splitlines() if f"cuteafd {serve}" in line)
    expected = owners.split(",")[encoder_owner]
    assert f"--vision rtx:{expected}" in launch


@pytest.mark.parametrize("family", ["mimo_flash", "mimo_pro"])
@pytest.mark.parametrize("owners", ["0,1", "1,0"])
@pytest.mark.parametrize("encoder_owner", [0, 1])
def test_dual_maps_audio_encoder_owner_to_visible_device(tmp_path: Path, family: str,
                                                       owners: str, encoder_owner: int) -> None:
    config = {**SPLIT_CONFIGS[family], "model_type": "mimo_v2",
              "hidden_size": 4096 if family == "mimo_flash" else 6144,
              "audio_config": {"audio_channels": 20}}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 0,
            "encoder_plan_hash": "ab" * 32, "encoder": None,
            "audio_encoder": {"kind": {"kind": "rtx", "gpu": encoder_owner}, "replicas": []}}
    result = _family_launch_result(tmp_path, config, "test/model",
                                  f"COORDINATOR_GPUS={owners}\nSPARK_COUNT=0\nEXPERT_BACKEND=local\n"
                                  "SPECULATOR=off\nVISION=off\nAUDIO=auto\n", encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert f"--audio rtx:{owners.split(',')[encoder_owner]}" in launch


def test_qwen_dual_rejects_attention_head_sharding(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  "RTX_GPUS=2\nSPARK_COUNT=0\nEXPERT_BACKEND=local\nATTENTION_PLACEMENT=heads\n")
    assert result.returncode != 0
    assert "head-split attention is not implemented" in result.stderr


def test_split_off_explicitly_uses_the_first_gpu(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  "RTX_GPUS=2\nCOORDINATOR_GPUS=1,0\nCOORDINATOR_SPLIT=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "nvidia.com/gpu=1" in launch and "--split-device" not in launch


@pytest.mark.parametrize("keys", ["RTX_GPUS=2\n", "COORDINATOR_GPUS=1,0\n", "COORDINATOR_SPLIT=heads\n"])
def test_glm_flash_explicit_split_passes_both_gpus(tmp_path: Path, keys: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "zai-org/GLM-5.3-Flash", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert "--split-device" in launch and "nvidia.com/gpu=0" in launch and "nvidia.com/gpu=1" in launch


@pytest.mark.parametrize("checkpoint", ["mimo_flash", "mimo_pro"])
def test_both_mimo_model_types_keep_their_supported_split(tmp_path: Path, checkpoint: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS[checkpoint], "test/model",
                                  "COORDINATOR_GPUS=1,0\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "--device 1 --split-device 0" in launch and "nvidia.com/gpu=0" in launch and "nvidia.com/gpu=1" in launch


def test_auto_uses_one_gpu_when_only_one_exists(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["mimo_pro"], "test/model", "", physical_gpus=(0,))
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "nvidia.com/gpu=0" in launch and "--split-device" not in launch


def test_split_off_needs_only_the_first_physical_gpu(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  "RTX_GPUS=2\nCOORDINATOR_SPLIT=off\n", physical_gpus=(0,))
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "nvidia.com/gpu=0" in launch and "--split-device" not in launch


@pytest.mark.parametrize("keys", ["RTX_GPUS=2\n", "COORDINATOR_GPUS=0,1\n",
                                 "COORDINATOR_SPLIT=heads\n"])
def test_explicit_split_requires_two_physical_gpus(tmp_path: Path, keys: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["mimo_pro"], "test/model", keys,
                                  physical_gpus=(0,))
    assert result.returncode == 2, result.stderr
    assert "physical" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


def test_speculator_uses_canonical_depth_and_validates_family(tmp_path: Path) -> None:
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 3, "moe_layer_freq": [0, 1, 1]}
    new = _family_launch_lines(tmp_path / "a", config, "XiaomiMiMo/MiMo-V2-Flash", "SPECULATOR=mtp\nSPECULATOR_DEPTH=2\n")
    launch = [line for line in new.splitlines() if "cuteafd serve-mimo" in line]
    assert launch and "--mtp 2" in launch[0]
    assert "deprecated" not in new
    bad = _family_launch_lines(tmp_path / "b", config, "XiaomiMiMo/MiMo-V2-Flash", "SPECULATOR=dspark\n")
    assert "does not apply to mimo_v2" in bad


@pytest.mark.parametrize("host, expected", [("", "auto"), ("HOST_CACHE_BYTES=0\n", None),
                                           ("HOST_CACHE_BYTES=16GiB\n", "16GiB")])
def test_mimo_port_flags_preserve_explicit_host_budget(tmp_path: Path, host: str, expected: str | None) -> None:
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 3, "moe_layer_freq": [0, 1, 1]}
    keys = ("MIMO_HOST_CACHE=on\nMIMO_PREFIX_DRAFT=on\n"
            "MIMO_SNAPSHOT_WAIT=on\nMIMO_PREFILL_CHUNK_S=4\nHTTP_QUEUE_DEPTH=32\nHTTP_QUEUE_WAIT_MS=1234\n") + host
    result = _family_launch_result(tmp_path, config, "XiaomiMiMo/MiMo-V2-Flash", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    for flag in ("--mimo-host-cache", "--mimo-prefix-draft",
                 "--mimo-snapshot-wait", "--prefill-chunk-s 4", "--http-queue-depth 32", "--http-queue-wait-ms 1234"):
        assert flag in launch
    if expected is None:
        assert "--host-cache-bytes" not in launch
    else:
        assert "--host-cache-bytes " + expected in launch


def test_qwen_launches_with_the_prefix_cache_keys(tmp_path: Path) -> None:
    config = {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
                                                         "layer_types": ["linear_attention", "full_attention"]}}
    keys = "PREFIX_CACHE_ENTRIES=8\nHOST_CACHE_BYTES=16GiB\nPOOL_TOKENS=65536\n"
    text = _family_launch_lines(tmp_path / "a", config, "Qwen/Qwen3.8-Flash-Next", keys)
    launch = [l for l in text.splitlines() if "cuteafd serve-qwen4" in l]
    assert launch, text
    for flag in ("--prefix-cache-entries 8", "--host-cache-bytes 16GiB", "--pool-tokens 65536"):
        assert flag in launch[0], flag
    default = _family_launch_lines(tmp_path / "b", config, "Qwen/Qwen3.8-Flash-Next", "")
    launch = [l for l in default.splitlines() if "cuteafd serve-qwen4" in l]
    assert "--prefix-cache-entries 20" in launch[0]
    assert "--host-cache-bytes auto" in launch[0] and "--pool-tokens 0" in launch[0]


@pytest.mark.parametrize("setting, expected", [("", "0"), ("auto", "0"), ("73728", "73728")])
def test_qwen_pool_defaults_to_free_memory_admission(tmp_path: Path, setting: str, expected: str) -> None:
    config = {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
                                                         "layer_types": ["linear_attention", "full_attention"]}}
    result = _family_launch_result(tmp_path, config, "Qwen/Qwen3.8-Flash-Next",
                                  "POOL_TOKENS=" + setting + "\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--pool-tokens " + expected in launch


def test_deepseek_v4_launches_with_the_prefix_cache_keys(tmp_path: Path) -> None:
    config = {"model_type": "deepseek_v4"}
    keys = "PREFIX_CACHE_ENTRIES=12\nHOST_CACHE_BYTES=32GiB\nPOOL_TOKENS=524288\nSPECULATOR=dspark\n"
    text = _family_launch_lines(tmp_path / "a", config, "deepseek-ai/DeepSeek-V4-Flash-0731", keys)
    launch = [l for l in text.splitlines() if "cuteafd serve-dsv4" in l]
    assert launch, text
    for flag in ("--prefix-cache-entries 12", "--host-cache-bytes 32GiB", "--pool-tokens 524288", "--dspark"):
        assert flag in launch[0], flag
    default = _family_launch_lines(tmp_path / "b", config, "deepseek-ai/DeepSeek-V4-Flash-0731", "")
    launch = [l for l in default.splitlines() if "cuteafd serve-dsv4" in l]
    assert "--prefix-cache-entries 20" in launch[0]
    # The pool defaults to auto (planner admission), as the release cards ran.
    assert "--host-cache-bytes auto" in launch[0] and "--pool-tokens 0" in launch[0]


@pytest.mark.parametrize("draft_kv", ["bf16", "fp8"])
def test_glm_flash_drafts_with_its_default_speculator(tmp_path: Path, draft_kv: str) -> None:
    config = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    model = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-Core-FP8"
    default = "incoai/GLM-5.3-Flash-DFlash2"
    dspark = "RedHatAI/GLM-5.3-Flash-speculator.dspark-preview"
    keys = f"DRAFT_KV_CACHE={draft_kv}\n"

    def launch(sub: str, extra: str, drafters: tuple[str, ...] = (default, dspark)) -> tuple[str, list[str]]:
        hf = tmp_path / sub / "hf"
        for name in drafters:
            _snapshot(hf, name, {"speculators_model_type": "dspark"} if name == dspark else {})
        text = _family_launch_lines(tmp_path / sub, config, model, keys + extra)
        lines = [l for l in text.splitlines() if "cuteafd serve-glmf" in l]
        for line in lines:
            command = shlex.split(line)
            assert command.count("--draft-kv-cache") == 1
            assert command[command.index("--draft-kv-cache") + 1] == draft_kv
        return text, lines

    snap = lambda sub, name: f"--draft {tmp_path}/{sub}/hf/hub/models--{name.replace('/', '--')}/snapshots/abc"
    text, lines = launch("a", "")
    assert lines and snap("a", default) in lines[0], text
    assert "drafts with dflash2" in text
    text, lines = launch("b", "SPECULATOR=off\n")
    assert lines and "--draft" not in shlex.split(lines[0]), text
    text, lines = launch("c", f"SPECULATOR=dspark\nSPECULATOR_MODEL_ID={dspark}\n")
    assert lines and snap("c", dspark) in lines[0], text
    text, lines = launch("d", "", ())
    assert lines and "--draft" not in shlex.split(lines[0]) and "hf download " + default in text, text


def test_glmf_pool_defaults_to_the_planned_pool(tmp_path: Path) -> None:
    config = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    for keys, flag in (("", "--pool-tokens 0"), ("POOL_TOKENS=auto\n", "--pool-tokens 0"),
                       ("POOL_TOKENS=65536\n", "--pool-tokens 65536")):
        result = _family_launch_result(tmp_path / str(len(keys)), config, "test/glmf",
                                       keys)
        assert result.returncode == 0, result.stderr
        launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
        assert flag in launch, launch
    dsv4 = _family_launch_lines(tmp_path / "dsv4", {"model_type": "deepseek_v4"},
                               "deepseek-ai/DeepSeek-V4-Flash-0731", "POOL_TOKENS=auto\n")
    assert "cuteafd serve-dsv4" in dsv4
    assert "--pool-tokens 0" in dsv4


@pytest.mark.parametrize("family_config", [*SPLIT_CONFIGS.values(),
    {"model_type": "deepseek_v4"},
    {"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}])
def test_coordinator_gpu_budget_is_forwarded_only_to_the_coordinator(tmp_path, family_config):
    model = "zai-org/GLM-5.3-Flash" if family_config.get("model_type") == "glm5_next" else "test/model"
    result = _family_launch_result(tmp_path, family_config, model,
                                  "COORDINATOR_GPU_BUDGET_GIB=32.5\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "--coordinator-gpu-budget-gib" in line and " serve-" in line)
    assert "cuteafd --coordinator-gpu-budget-gib 32.5 serve-" in launch
    for line in result.stderr.splitlines():
        if "expertd-native" in line:
            assert "--coordinator-gpu-budget-gib" not in line


@pytest.mark.parametrize("budget", ["0", "-1", "NaN", "inf", "32GiB", "0.0000000000001", "9999999999999999999999"])
def test_coordinator_gpu_budget_rejects_invalid_values_before_side_effects(tmp_path, budget):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  f"COORDINATOR_GPU_BUDGET_GIB={budget}\n")
    assert result.returncode != 0
    assert "COORDINATOR_GPU_BUDGET_GIB must be" in result.stderr
    assert "docker " not in result.stderr and "ssh " not in result.stderr and "nest " not in result.stderr


def test_qwen_backend_preflight_charges_physical_usage_against_the_ceiling(tmp_path):
    result = _family_launch_result(tmp_path,
        {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": "exl3"}}, "test/model",
        "COORDINATOR_GPU_BUDGET_GIB=32\n", preferred_ranks=1,
        gpu_free_mib=90 * 1024, gpu_total_mib=96 * 1024)
    assert result.returncode == 0, result.stderr
    preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    assert "--coordinator-gpu-budget-gib 26.0" in preflight and "--coordinator-weight-budget-gib 26.0" in preflight
    launch = next(line for line in result.stderr.splitlines() if "serve-qwen4" in line)
    assert "--coordinator-gpu-budget-gib 32" in launch and "--peers" in launch


def test_budget_key_is_accepted_for_cleanup_and_native_launchers(tmp_path):
    config = tmp_path / "budget.config"
    config.write_text("COORDINATOR_GPU_BUDGET_GIB=32\n")
    result = subprocess.run(["bash", "-c", 'source "$1"; release_load_config "$2" stop; printf "%s" "$COORDINATOR_GPU_BUDGET_GIB"',
        "bash", str(ROOT / "scripts/lib/release-common.sh"), str(config)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout == "32", result.stderr
    for file in ("scripts/launch/run-family.sh", "scripts/launch/run-tp-ep-native-candidate.sh"):
        assert 'args+=(--coordinator-gpu-budget-gib "$COORDINATOR_GPU_BUDGET_GIB")' in (ROOT / file).read_text()


@pytest.mark.parametrize("keys,expected", [
    ("", "fp8"), ("SPECULATOR=dflash2\n", "fp8"),
    ("SPECULATOR_FP8=auto\n", "checkpoint"), ("SPECULATOR_FP8=off\n", "bf16"),
    ("SPECULATOR=off\n", "off"), ("SPECULATOR=mtp\n", "mtp"),
])
@pytest.mark.parametrize("draft_kv", ["bf16", "fp8"])
def test_flash_mopd_defaults_to_its_qualified_bundled_drafter(tmp_path: Path, keys: str, expected: str, draft_kv: str) -> None:
    config = {"model_type": "mimo_v2", "hidden_size": 4096, "num_hidden_layers": 2,
              "moe_layer_freq": [0, 1], "quantization_config": {"store_dtype": "mxfp4"}}
    model = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD"
    result = _family_launch_result(tmp_path, config, model, keys + f"DRAFT_KV_CACHE={draft_kv}\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    command = shlex.split(launch)
    assert command.count("--draft-kv-cache") == 1
    assert command[command.index("--draft-kv-cache") + 1] == draft_kv
    assert "--fp8-head" not in launch and "--fp8-o-proj" not in launch
    if expected in ("off", "mtp"):
        assert "--draft" not in command
        assert ("--mtp 1" in launch) == (expected == "mtp")
    else:
        assert f"--draft {tmp_path}/hf/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-MOPD/snapshots/abc" in launch
        if expected == "checkpoint":
            assert "--draft-representation checkpoint" in launch and "--draft-fp8" not in launch
        else:
            assert f"--draft-fp8 {'true' if expected == 'fp8' else 'false'}" in launch


def test_flash_mopd_external_drafter_does_not_inherit_bundled_precision(tmp_path: Path) -> None:
    _snapshot(tmp_path / "hf", "test/external-draft", {})
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2", "num_hidden_layers": 2,
                                  "moe_layer_freq": [0, 1]}, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
                                  "SPECULATOR=dflash2\nSPECULATOR_MODEL_ID=test/external-draft\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "models--test--external-draft/snapshots/abc" in launch
    assert "--draft-fp8" not in launch


def test_family_config_reads_share_the_stop_key_grammar() -> None:
    import re

    launcher = (ROOT / "scripts/launch/run-family.sh").read_text()
    # Include old spellings read by key(), plus the two generated projection loops.
    keys = set(re.findall(r'\bget ([A-Z][A-Z_0-9]+)', launcher))
    for new, old in re.findall(r'\bkey ([A-Z][A-Z_0-9]+) ([A-Z][A-Z_0-9]+)', launcher):
        if new != "NEW":
            keys.update((new, old))
    keys.update(("MIMO_FP8_HEAD", "MIMO_FP8_O_PROJ", "QWEN_FP8_DECODE", "QWEN_FP8_HEAD"))
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; shift; for key; do release_known_key "$key" || exit 1; done',
         "bash", str(ROOT / "scripts/lib/release-common.sh"), *sorted(keys)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert 'release_known_key "$key"' in launcher


def test_family_launcher_rejects_unknown_keys_before_launching(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                                            "moe_layer_freq": [0, 1]}, "test/mimo", "SPECULATOR_TYPO=off\n")
    assert result.returncode != 0
    assert "unknown configuration key: SPECULATOR_TYPO" in result.stderr
    assert "docker " not in result.stderr


@pytest.mark.parametrize("backend,preferred,local", [("auto", 0, True), ("auto", 4, False),
                                                     ("spark", 0, False), ("local", 4, True)])
def test_qwen_preferred_experts_use_the_planner_before_launch(tmp_path: Path, backend: str,
                                                             preferred: int, local: bool) -> None:
    result = _family_launch_result(tmp_path, {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": "exl3"}}, "test/model",
                                  f"EXPERT_BACKEND={backend}\n", preferred_ranks=preferred)
    assert result.returncode != 0
    assert "planner changed exact Spark count" in result.stderr
    assert "cuteafd serve-qwen4" not in result.stderr


def test_qwen_preferred_local_allows_native_mtp(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": "exl3"}}, "test/model",
                                  "SPARK_COUNT=0\nSPECULATOR=mtp\nSPECULATOR_DEPTH=3\n", preferred_ranks=0)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--mtp 3" in launch and "--local-experts" in launch


@pytest.mark.parametrize("keys,depth", [("", 3), ("SPECULATOR=mtp\n", 3),
                                      ("SPECULATOR_DEPTH=2\n", 2),
                                      ("SPECULATOR=off\n", None)])
def test_qwen_local_mtp_default_preserves_overrides(tmp_path: Path, keys: str, depth: int | None) -> None:
    result = _family_launch_result(tmp_path, {**SPLIT_CONFIGS["qwen4"],
                                            "quantization_config": {"quant_method": "exl3"}},
                                  "test/model", "EXPERT_BACKEND=local\n" + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--local-experts" in launch
    if depth is None:
        assert "--mtp " not in launch
    else:
        assert f"--mtp {depth}" in launch


def test_qwen_spark_mtp_keeps_the_draft_layer_on_coordinator(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, {**SPLIT_CONFIGS["qwen4"],
                                            "quantization_config": {"quant_method": "exl3"}},
                                  "test/model", "EXPERT_BACKEND=spark\nSPECULATOR=mtp\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--mtp 3" in launch and "--peers " in launch and "--local-experts" not in launch
    assert "expertd-native" in result.stderr


@pytest.mark.parametrize("method,mtp_layers", [("exl3", 0), ("exl3", 2), ("fp8", 1), ("nvfp4", 1)])
def test_qwen_local_unqualified_mtp_keeps_the_existing_default(tmp_path: Path, method: str, mtp_layers: int) -> None:
    config = {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": method},
              "text_config": {**SPLIT_CONFIGS["qwen4"]["text_config"], "mtp_num_hidden_layers": mtp_layers}}
    result = _family_launch_result(tmp_path, config, "test/model", "EXPERT_BACKEND=local\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--local-experts" in launch and "--mtp " not in launch


@pytest.mark.parametrize("method", ["fp8", "nvfp4"])
def test_qwen_unqualified_explicit_mtp_keeps_the_existing_depth(tmp_path: Path, method: str) -> None:
    config = {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": method}}
    result = _family_launch_result(tmp_path, config, "test/model", "EXPERT_BACKEND=local\nSPECULATOR=mtp\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--local-experts" in launch and "--mtp 1" in launch


@pytest.mark.parametrize("restart,admitted_gib", [(False, 12000 / 1024), (True, 96000 / 1024)])
def test_qwen_restart_admission_credits_only_its_own_gpu_memory(tmp_path: Path, restart: bool,
                                                              admitted_gib: float) -> None:
    result = _family_launch_result(tmp_path, {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": "exl3"}},
                                  "test/model", "INSTANCE=own\nSPARK_COUNT=0\n", preferred_ranks=0, restart=restart,
                                  gpu_free_mib=12000, container_pids=(123,),
                                  gpu_allocations=((123, 84000), (456, 2000)))
    assert result.returncode == 0, result.stderr
    preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    assert f"--coordinator-gpu-budget-gib {admitted_gib}" in preflight


def test_spark_admission_inherits_release_readiness_bound(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model", "",
                                  preferred_ranks=1,
                                  extra_env={"CUTEAFD_RELEASE_READY_TIMEOUT_SECONDS": "2400"})
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "CUTEAFD_SPARK_ADMISSION_TIMEOUT_SECONDS=2400" in launch


def test_qwen_other_formats_skip_the_local_qualification_preflight(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  "", preferred_ranks=1)
    assert result.returncode == 0, result.stderr
    assert "cuteafd plan" in result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--peers" in launch and "--local-experts" not in launch


@pytest.mark.parametrize("previous_peers,cleanup", [(None, False), ("10.0.0.1:19555", True),
                                                   ("10.0.0.9:19555", False)])
def test_qwen_local_restart_releases_only_its_previous_workers(tmp_path: Path, previous_peers: str | None,
                                                             cleanup: bool) -> None:
    result = _family_launch_result(tmp_path, {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": "exl3"}},
                                  "test/model", "INSTANCE=own\nSPARK_COUNT=0\n", preferred_ranks=0, restart=True,
                                  previous_peers=previous_peers)
    assert result.returncode == 0, result.stderr
    assert ("docker rm -f cuteafd-spark-expert-h0-19555" in result.stderr) == cleanup
    assert "filter name=^cuteafd-spark-expert-" not in result.stderr

@pytest.mark.parametrize("mode,kind", [("auto", "spark"), ("spark", "spark"), ("spark:0", "spark"),
                                        ("rtx", "rtx"), ("rtx:0", "rtx"), ("off", "off"), (None, "spark")])
def test_mimo_encoder_plan_hash_and_selected_rank(tmp_path, mode, kind):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1], "vision_config": {"depth": 28}}
    placement = {"kind": kind}
    if kind == "spark": placement["rank"] = 0
    if kind == "rtx": placement["gpu"] = 0
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": {"kind": placement, "replicas": []}}
    vision_key = f"VISION={mode}\n" if mode is not None else ""
    result = _family_launch_result(tmp_path, config, "test/mimo", f"{vision_key}RTX_GPUS=1\nSPECULATOR=off\n", encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    worker = next(line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line)
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert ("--encoder-listen" in worker) == (kind == "spark")
    assert ("--vision-peers 10.0.0.1:19442" in launch) == (kind == "spark")
    assert f"--vision {kind}" in launch
    if kind == "spark":
        assert f"--encoder-plan-hash {'ab' * 32}" in worker
        assert f"--encoder-plan-hash {'ab' * 32}" in launch
        assert "--encoder-revision abc" in worker and "--encoder-revision abc" in launch
    if kind == "off":
        assert "cuteafd plan" in result.stderr
    else:
        preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
        assert f"--vision {mode or 'auto'}" in preflight


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("concurrency", [None, 8, 16])
@pytest.mark.parametrize("gpu_total_mib", [32768, 98304])
def test_mimo_warm_marks_reach_startup_layout(tmp_path, warm, concurrency, gpu_total_mib):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
              "moe_layer_freq": [0, 1], "vision_config": {"depth": 28}}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32,
            "encoder": {"kind": {"kind": "rtx", "gpu": 0}, "replicas": []}}
    keys = "VISION=rtx\nAUDIO=off\nRTX_GPUS=1\nSPECULATOR=off\nMAX_CONTEXT_TOKENS=131072\n"
    if concurrency is not None:
        keys += f"CONCURRENCY={concurrency}\n"
    keys += f"MIMO_PREFIX_DRAFT={'on' if warm else 'off'}\n"
    result = _family_launch_result(tmp_path, config, "test/mimo", keys, encoder_plan=plan,
                                   gpu_total_mib=gpu_total_mib)
    assert result.returncode == 0, result.stderr
    preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert ("--mimo-prefix-draft" in preflight) == warm
    assert ("--mimo-prefix-draft" in launch) == warm
    effective = concurrency if concurrency is not None else (8 if gpu_total_mib <= 32768 else 16)
    assert f"--concurrency {effective}" in preflight
    assert f"--max-sequences {effective}" in launch
    assert "unbound variable" not in result.stderr
    assert "--context-tokens 131072" in preflight
    assert "--prefill-rows 4096" in preflight
    assert "--expert-input fp8" in preflight


@pytest.mark.parametrize("ingress", ["fp8", "bf16", "bf16-decode"])
def test_mimo_media_plan_keeps_ingress_onboarding_and_graph_contract(tmp_path, ingress):
    config = {"model_type": "mimo_v2", "hidden_size": 4096, "num_hidden_layers": 2,
              "moe_layer_freq": [0, 1], "vision_config": {"depth": 28}}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 2,
            "encoder_plan_hash": "ab" * 32,
            "encoder": {"kind": {"kind": "rtx", "gpu": 0}, "replicas": []}}
    keys = (f"EXPERT_INPUT={ingress}\nRTX_EXPERT_LAYERS=25%\nSPARK_COUNT=2\n"
            "VISION=rtx\nAUDIO=off\nRTX_GPUS=1\nSPECULATOR=off\n"
            "MAX_CONTEXT_TOKENS=262144\nDECODE_GRAPHS=on\nFULL_PREFILL_LOGITS=on\n")
    result = _family_launch_result(tmp_path, config, "test/mimo", keys, encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert f"--expert-input {ingress}" in preflight and f"--expert-input {ingress}" in launch
    assert "--rtx-expert-layers 0" in preflight and "--onboard 0" in launch
    assert "--context-tokens 262144" in preflight and "--max-context 262144" in launch
    assert "--mimo-decode-graphs" in preflight and "--decode-graphs true" in launch
    assert "--full-prefill-logits" in preflight and "--full-prefill-logits" in launch


@pytest.mark.parametrize("vision_kind,audio_kind", [("off", "spark"), ("spark", "spark"), ("rtx", "spark"), ("spark", "rtx"), ("rtx", "rtx")])
def test_mimo_audio_independent_planner_peers_backend_and_rtx_fallback(tmp_path, vision_kind, audio_kind):
    config = {"model_type": "mimo_v2", "hidden_size": 4096, "num_hidden_layers": 2, "moe_layer_freq": [0, 1],
              "vision_config": {"depth": 28}, "audio_token_id": 151669, "audio_config": {"audio_channels": 20}}
    def placement(kind):
        return {"kind": {"kind": kind, **({"rank": 0} if kind == "spark" else {"gpu": 0} if kind == "rtx" else {})}, "replicas": []}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": placement(vision_kind), "audio_encoder": placement(audio_kind)}
    result = _family_launch_result(tmp_path, config, "test/mimo", f"VISION={vision_kind}\nAUDIO=auto\nRTX_GPUS=1\nSPECULATOR=off\n", encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    worker = next(line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line)
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert f"--audio {audio_kind}:0" in launch
    assert ("--audio-encoder-listen 0.0.0.0:19443" in worker) == (audio_kind == "spark")
    assert ("--encoder-listen 0.0.0.0:19442" in worker) == (vision_kind == "spark")
    assert ("--audio-peers 10.0.0.1:19443" in launch) == (audio_kind == "spark")
    assert ("--vision-peers 10.0.0.1:19442" in launch) == (vision_kind == "spark")
    if audio_kind == "spark":
        assert f"--audio-encoder-plan-hash {'ab' * 32}" in worker
        assert f"--audio-encoder-plan-hash {'ab' * 32}" in launch
        assert "--audio-encoder-revision abc" in launch
        assert "--audio-encoder-backend mimo_audio_fp32_v1/cuda13000/cufft12000/cublas13.0.0/cute_aot_sm121/export" + "cd" * 32 in launch
    else:
        assert "--audio-encoder-backend" not in launch


@pytest.mark.parametrize("mode,kind", [(None, "spark"), ("off", "off"), ("auto", "spark"),
                                      ("spark:0", "spark"), ("rtx:0", "rtx")])
def test_glmf_encoder_defaults_auto_and_forwards_remote_identity(tmp_path, mode, kind):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"], "vision_config": {"depth": 24}}
    placement = {"kind": kind}
    if kind == "spark": placement["rank"] = 0
    if kind == "rtx": placement["gpu"] = 0
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": {"kind": placement, "replicas": []}}
    keys = "RTX_GPUS=1\nSPECULATOR=off\n" + (f"VISION={mode}\n" if mode else "")
    result = _family_launch_result(tmp_path, config, "zai-org/GLM-5.3-Flash", keys, encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert f"--vision {kind}" in launch
    assert ("--encoder-listen" in result.stderr) == (kind == "spark")
    assert ("--vision-peers 10.0.0.1:19442" in launch) == (kind == "spark")
    if kind == "spark":
        assert f"--encoder-plan-hash {'ab' * 32}" in launch and "--encoder-revision abc" in launch
    assert "cuteafd plan" in result.stderr
    if kind != "off":
        preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
        assert f"--vision {mode or 'auto'}" in preflight


@pytest.mark.parametrize("setting,value,legacy", [("on", "1", "0"), ("off", "0", "1")])
def test_glmf_remote_drafter_config_overrides_env_for_plan_and_server(tmp_path, setting, value, legacy):
    config = {**SPLIT_CONFIGS["glm5_flash"], "vision_config": {"depth": 24}}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": {"kind": {"kind": "rtx", "gpu": 0}, "replicas": []}}
    result = _family_launch_result(tmp_path, config, "zai-org/GLM-5.3-Flash",
                                  f"RTX_GPUS=2\nSPECULATOR=off\nVISION=rtx\nGLM5_FLASH_REMOTE_DRAFTER={setting}\n",
                                  encoder_plan=plan, extra_env={"CUTEAFD_GLMF_REMOTE_DFLASH": legacy})
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    planner = next(line for line in lines if "cuteafd plan" in line and "--deployment" in line)
    server = next(line for line in lines if "cuteafd serve-glmf" in line)
    for command in (planner, server):
        assert command.count("-e CUTEAFD_GLMF_REMOTE_DFLASH=") == 1, command
        assert f"-e CUTEAFD_GLMF_REMOTE_DFLASH={value}" in command, command
    assert not any("CUTEAFD_GLMF_REMOTE_DFLASH=" in line for line in lines if "cuteafd expertd-native" in line)


@pytest.mark.parametrize("value", ["", "0", "1"])
def test_glmf_remote_dflash_candidate_env_matches_plan_and_server(tmp_path, value):
    config = {**SPLIT_CONFIGS["glm5_flash"], "vision_config": {"depth": 24}}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": {"kind": {"kind": "rtx", "gpu": 0}, "replicas": []}}
    result = _family_launch_result(tmp_path, config, "zai-org/GLM-5.3-Flash",
                                  "RTX_GPUS=2\nSPECULATOR=off\nVISION=rtx\n", encoder_plan=plan,
                                  extra_env={"CUTEAFD_GLMF_REMOTE_DFLASH": value})
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    planner = next(line for line in lines if "cuteafd plan" in line and "--deployment" in line)
    server = next(line for line in lines if "cuteafd serve-glmf" in line)
    expected = f"-e CUTEAFD_GLMF_REMOTE_DFLASH={value}"
    for command in (planner, server):
        assert (expected in command) == bool(value), command
    assert not any("CUTEAFD_GLMF_REMOTE_DFLASH=" in line for line in lines if "cuteafd expertd-native" in line)


@pytest.mark.parametrize("vision", [None, "auto"])
def test_glmf_auto_vision_serves_text_when_the_plan_turns_vision_off(tmp_path, vision):
    """VISION=auto (the default) follows the encoder plan: when `cuteafd plan` resolves automatic
    vision to off (a checkpoint whose own chat template cannot mark images, with no
    CHAT_TEMPLATE_FROM), the launch is text only: serve-glmf gets --vision off and no encoder
    identity, and no worker starts an encoder. The plan is asked for auto, without a template source."""
    config = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"], "vision_config": {"depth": 24}}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1, "encoder_plan_hash": "ab" * 32,
            "vision": "off", "encoder": {"kind": {"kind": "off"}, "replicas": []}}
    keys = "RTX_GPUS=1\nSPECULATOR=off\n" + (f"VISION={vision}\n" if vision else "")
    result = _family_launch_result(tmp_path, config, "zai-org/GLM-5.3-Flash", keys, encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    planner = next(line for line in lines if "cuteafd plan" in line and "--deployment" in line) + " "
    launch = next(line for line in lines if "cuteafd serve-glmf" in line) + " "
    worker = next(line for line in lines if "cuteafd expertd-native" in line)
    assert " --vision auto " in planner and "--chat-template-from" not in planner, planner
    assert " --vision off " in launch and launch.count(" --vision ") == 1, launch
    for flag in ("--vision-peers", "--encoder-plan-hash", "--encoder-revision", "--chat-template-from"):
        assert flag not in launch, (flag, launch)
    assert "--encoder" not in worker, worker


@pytest.mark.parametrize("vision,kind", [(None, "spark"), ("auto", "spark"), ("auto", "rtx"), ("spark:0", "spark"),
                                         ("rtx:0", "rtx")])
def test_glmf_encoder_placement_plan_gets_the_template_source_serving_gets(tmp_path, vision, kind):
    """With CHAT_TEMPLATE_FROM the plan is told the template source serve-glmf gets (once each), so
    automatic vision stays on for a checkpoint whose own template cannot mark images; the launch is
    the placement the plan chose, as before."""
    config = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"], "vision_config": {"depth": 24}}
    placement = {"kind": kind, **({"rank": 0} if kind == "spark" else {"gpu": 0})}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": {"kind": placement, "replicas": []}}
    keys = ("RTX_GPUS=1\nSPECULATOR=off\nCHAT_TEMPLATE_FROM=test/model\n"
            + (f"VISION={vision}\n" if vision else ""))
    result = _family_launch_result(tmp_path, config, "test/model", keys, encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    planner = next(line for line in lines if "cuteafd plan" in line and "--deployment" in line) + " "
    launch = next(line for line in lines if "cuteafd serve-glmf" in line) + " "
    for command in [planner, launch]:
        assert " --chat-template-from test/model " in command and command.count("--chat-template-from") == 1, command
    assert f" --vision {vision or 'auto'} " in planner, planner
    assert f" --vision {kind}:0 " in launch, launch
    assert ("--vision-peers 10.0.0.1:19442" in launch) == (kind == "spark")


def test_glmf_text_launch_gives_the_plan_no_template_source(tmp_path):
    """VISION=off ignores CHAT_TEMPLATE_FROM on both commands, as before."""
    config = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"], "vision_config": {"depth": 24}}
    result = _family_launch_result(tmp_path, config, "test/model",
                                   "RTX_GPUS=1\nSPECULATOR=off\nVISION=off\n"
                                   "CHAT_TEMPLATE_FROM=test/model\n")
    assert result.returncode == 0, result.stderr
    assert "cuteafd plan" in result.stderr and "--chat-template-from" not in result.stderr


@pytest.mark.parametrize("keys,forwarded", [("", False), ("GLM5_FLASH_DECODE_ROWS=64\n", False),
                                             ("GLM5_FLASH_DECODE_ROWS=128\n", True)])
def test_glmf_encoder_placement_plans_the_decode_rows_serving_takes(tmp_path, keys, forwarded):
    """The encoder placement plan charges what serving admits: GLM5_FLASH_DECODE_ROWS=128 reaches
    `cuteafd plan --layout` as --decode-rows 128 beside the coordinator's, with a fixed near-fit pool
    and the vision tower on the RTX; 64, the default, passes it to neither."""
    config = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"], "vision_config": {"depth": 24}}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": {"kind": {"kind": "rtx", "gpu": 0}, "replicas": []}}
    result = _family_launch_result(tmp_path, config, "zai-org/GLM-5.3-Flash",
                                   "RTX_GPUS=1\nSPECULATOR=off\nVISION=rtx\nPOOL_TOKENS=262144\n" + keys,
                                   encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    planner = next(line for line in lines if "cuteafd plan" in line and "--deployment" in line)
    launch = next(line for line in lines if "cuteafd serve-glmf" in line)
    assert "--vision rtx" in planner and "--pool-tokens 262144" in planner, planner
    for command in [planner, launch]:
        assert ("--decode-rows 128" in command) == forwarded and command.count("--decode-rows") == int(forwarded), command


GLMF_VISION = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
               "layer_types": ["linear_attention", "deepseek_sparse_attention"], "vision_config": {"depth": 24}}
GLMF_RTX_ENCODER_PLAN = {"placement_supported": True, "fits": True, "spark_ranks": 1, "encoder_plan_hash": "ab" * 32,
                         "encoder": {"kind": {"kind": "rtx", "gpu": 0}, "replicas": []}}


@pytest.mark.parametrize("keys,serve,plan", [
    # The resolved logical-capacity default reaches both commands explicitly.
    ("", {"--max-sequences": "16", "--prefix-cache-entries": "20"},
     {"--concurrency": "16", "--state-slots": "16", "--mark-lanes": "16"}),
    # C1 without entries: 8 state slots (serve-glmf's --slots), one mark lane, no marks.
    ("CONCURRENCY=1\nPREFIX_CACHE_ENTRIES=0\n", {"--max-sequences": "1", "--prefix-cache-entries": "0"},
     {"--concurrency": "1", "--state-slots": "8", "--mark-lanes": "1", "--prefix-cache-entries": "0"}),
    ("CONCURRENCY=8\nPREFIX_CACHE_ENTRIES=20\nPREFIX_CACHE_MARK_MIB=2048\nGLM5_FLASH_REPLAY_RECORDS=own\n",
     {"--max-sequences": "8", "--prefix-cache-entries": "20", "--prefix-cache-mark-mib": "2048"},
     {"--concurrency": "8", "--state-slots": "8", "--mark-lanes": "8", "--prefix-cache-entries": "20",
      "--prefix-cache-mark-mib": "2048"}),
    ("CONCURRENCY=16\nPREFIX_CACHE_ENTRIES=6\nPREFIX_CACHE_MARK_MIB=1971\nGLM5_FLASH_REPLAY_RECORDS=shared\n",
     {"--max-sequences": "16", "--prefix-cache-entries": "6", "--prefix-cache-mark-mib": "1971",
      "--replay-records": "shared"},
     {"--concurrency": "16", "--state-slots": "16", "--mark-lanes": "16", "--prefix-cache-entries": "6",
      "--prefix-cache-mark-mib": "1971", "--replay-records": "shared"}),
    # C128: 128 state slots, and the mark arena's lanes capped at DECODE_ROWS (64).
    ("CONCURRENCY=128\n", {"--max-sequences": "128", "--prefix-cache-entries": "20"},
     {"--concurrency": "128", "--state-slots": "128", "--mark-lanes": "64"}),
])
def test_glmf_encoder_placement_plans_the_prefix_knobs_serving_takes(tmp_path, keys, serve, plan):
    """The encoder placement plan reserves what serve-glmf allocates for the keys a launch sets:
    the sequences (CONCURRENCY) with serve-glmf's state slots, max(--slots 8, --max-sequences), and
    its mark-arena lanes, min(--max-sequences, 64), PREFIX_CACHE_ENTRIES, PREFIX_CACHE_MARK_MIB and
    shared replay records. Effective sequences always reach the planner and server."""
    result = _family_launch_result(tmp_path, GLMF_VISION, "zai-org/GLM-5.3-Flash",
                                   "RTX_GPUS=1\nSPECULATOR=off\nVISION=rtx\nGLM5_FLASH_REPLAY_RECORDS=own\n" + keys,
                                   encoder_plan=GLMF_RTX_ENCODER_PLAN)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    commands = {"plan": next(line for line in lines if "cuteafd plan" in line and "--deployment" in line) + " ",
                "serve": next(line for line in lines if "cuteafd serve-glmf" in line) + " "}
    flags = ("--max-sequences", "--concurrency", "--state-slots", "--mark-lanes", "--prefix-cache-entries",
             "--prefix-cache-mark-mib", "--replay-records")
    for name, expected in [("plan", plan), ("serve", serve)]:
        command = commands[name]
        for flag in flags:
            if flag in expected:
                assert f" {flag} {expected[flag]} " in command and command.count(f" {flag} ") == 1, (flag, command)
            else:
                assert f" {flag} " not in command, (flag, command)


@pytest.mark.parametrize("split,fixed,local,wide", [
    (False, False, False, True), (True, False, False, True),
    (False, True, False, True), (False, False, True, True),
    (False, False, False, False),
])
def test_glmf_auto_defaults_and_encoder_plan_agree(tmp_path, split, fixed, local, wide):
    config = {**GLMF_VISION, "moe_intermediate_size": 2048, "intermediate_size": 12288}
    stems = ["mhc_post_pre", "mla_producer", "o", "sparse_mla_decode", "index_producer",
             "index_topk_decode", "ffn_i2048", "ffn_i12288", "kda", "kda_w8",
             "index_producer_c", "kda_commit_c"]
    manifest = {"programs": [{"name": "glmf_" + n + "_m128"} for n in stems]} if wide else None
    keys = (f"RTX_GPUS={2 if split else 1}\nSPECULATOR=off\nVISION=rtx\n"
            "GLM5_FLASH_DECODE_ROWS=auto\nGLM5_FLASH_INDEX_CACHE=auto\nGLM5_FLASH_REPLAY_RECORDS=auto\n")
    if fixed: keys += "POOL_TOKENS=262144\n"
    if local: keys += "SPARK_COUNT=0\nEXPERT_BACKEND=local\n"
    else: keys += f"SPARK_COUNT={4 if split else 2}\n"
    result = _family_launch_result(tmp_path, config, "zai-org/GLM-5.3-Flash", keys,
                                  program_manifest=manifest, encoder_plan=GLMF_RTX_ENCODER_PLAN)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    planner = next(line for line in lines if "cuteafd plan" in line)
    serve = next(line for line in lines if "cuteafd serve-glmf" in line)
    for command in (planner, serve):
        assert "--decode-rows 128" not in command, command
        assert ("--index-cache compact" in command) == (not split), command
        assert ("--replay-records shared" in command) == (not split and not fixed and not local), command
        assert "--prefix-marks pool" not in command
    assert result.stderr.count("GLM5_FLASH_DECODE_ROWS=auto ->") == 1
    assert result.stderr.count("GLM5_FLASH_INDEX_CACHE=auto ->") == 1
    assert result.stderr.count("GLM5_FLASH_REPLAY_RECORDS=auto ->") == 1
    assert serve.count("--host-cache-bytes") == 1 and "--host-cache-bytes auto" in serve


@pytest.mark.parametrize("index,state,missing", [
    ("keys", "f32", None), ("keys", "bf16", None),
    ("compact", "bf16", None), ("compact", "bf16-tile", None),
    ("compact", "f32", "kda_commit_c"),
])
def test_glmf_auto_rows_keep_64_even_with_wide_programs(tmp_path, index, state, missing):
    config = {**GLMF_VISION, "moe_intermediate_size": 2048, "intermediate_size": 12288}
    stems = ["mhc_post_pre", "mla_producer", "o", "sparse_mla_decode", "index_producer",
             "index_topk_decode", "ffn_i2048", "ffn_i12288"]
    stems += ["kda" if state == "f32" else "kda_s16"]
    if index == "compact": stems += ["index_producer_c"]
    stems += ["kda_commit" + ("_c" if index == "compact" else "") + ("_s16" if state != "f32" else "")]
    manifest = {"programs": [{"name": "glmf_" + n + "_m128"} for n in stems if n != missing]}
    keys = (f"RTX_GPUS=1\nSPECULATOR=off\nVISION=rtx\nSPARK_COUNT=2\n"
            f"GLM5_FLASH_INDEX_CACHE={index}\nGLM5_FLASH_KDA_STATE={state}\n"
            "GLM5_FLASH_KDA_FP8=off\nGLM5_FLASH_DECODE_ROWS=auto\n")
    result = _family_launch_result(tmp_path, config, "zai-org/GLM-5.3-Flash", keys,
                                  program_manifest=manifest, encoder_plan=GLMF_RTX_ENCODER_PLAN)
    assert result.returncode == 0, result.stderr
    for line in result.stderr.splitlines():
        if "cuteafd plan" in line or "cuteafd serve-glmf" in line:
            assert "--decode-rows 128" not in line, line
    assert "GLM5_FLASH_DECODE_ROWS=auto -> 64 (" in result.stderr
    assert "128 rows remain opt-in; C16 and post-C16 C1 draft-cost gate" in result.stderr


@pytest.mark.parametrize("fp8", ["auto", "off"])
def test_glmf_draft_defaults_apply_only_to_the_fp8_drafter(tmp_path, fp8):
    _snapshot(tmp_path / "hf", "incoai/GLM-5.3-Flash-DFlash2", {})
    result = _family_launch_result(tmp_path, _GLMF, "zai-org/GLM-5.3-Flash",
                                  f"RTX_GPUS=1\nSPECULATOR_FP8={fp8}\n")
    assert result.returncode == 0, result.stderr
    serve = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert "--draft-head" not in serve
    assert ("--draft-linear w8a8" in serve) == (fp8 != "off")


def test_glmf_plan_counts_follow_serve_glmf_formulas():
    """The launcher's --state-slots and --mark-lanes restate serve-glmf's own rules: the engine keeps
    max(--slots, --max-sequences) KDA state slots, --slots defaults to 8, and the prefix mark arena
    counts min(--max-sequences, DECODE_ROWS) lanes with DECODE_ROWS = 64. Change one, change both."""
    glmf = ROOT / "rust/crates/cuteafd-daemon/src/families/glm5_flash"
    serving, engine_args, engine = ((glmf / name).read_text() for name in ("serve.rs", "mod.rs", "engine.rs"))
    assert "engine_args.slots = engine_args.slots.max(args.max_sequences);" in serving
    assert "    let lanes = max_sequences.min(DECODE_ROWS);" in serving
    assert "PrefixMarks::Arena => super::prefix::ArenaMarks::Rule(prefix.mark_rule(lanes))," in serving
    assert "    #[arg(long, default_value_t = 8)]\n    pub slots: usize,\n" in engine_args
    assert "pub(crate) const DECODE_ROWS: usize = 64;" in engine
    launcher = (ROOT / "scripts/launch/run-family.sh").read_text()
    assert '--state-slots "$((glmf_sequences > 8 ? glmf_sequences : 8))"' in launcher
    assert '--mark-lanes "$((glmf_sequences < 64 ? glmf_sequences : 64))"' in launcher


@pytest.mark.parametrize("value", ["0", "eight", "-4"])
def test_glmf_encoder_placement_refuses_a_bad_sequence_count(tmp_path, value):
    result = _family_launch_result(tmp_path, GLMF_VISION, "zai-org/GLM-5.3-Flash",
                                   f"RTX_GPUS=1\nSPECULATOR=off\nVISION=rtx\nCONCURRENCY={value}\n",
                                   encoder_plan=GLMF_RTX_ENCODER_PLAN)
    assert result.returncode == 2 and "CONCURRENCY must be a positive sequence count" in result.stderr, result.stderr
    assert not any("cuteafd serve-glmf" in line for line in result.stderr.splitlines())


@pytest.mark.parametrize("family_config,serve", [
    ({"model_type": "deepseek_v4"}, "serve-dsv4"),
    ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}, "serve-glm"),
])
def test_towerless_families_default_to_auto_and_start_no_tower(tmp_path, family_config, serve):
    # VISION defaults to auto for every family, as the release configs set it. DeepSeek V4
    # and GLM 5.3 have no tower: no encoder plan runs and nothing is placed.
    result = _family_launch_result(tmp_path, family_config, "test/model", "SPECULATOR=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if f"cuteafd {serve}" in line)
    assert "--vision auto" in launch
    assert "cuteafd plan" in result.stderr
    assert "--encoder-listen" not in result.stderr and "--vision-peers" not in launch


@pytest.mark.parametrize("config,model,serve", [
    ({"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
     "test/mimo", "serve-mimo"),
    ({"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
      "layer_types": ["linear_attention", "deepseek_sparse_attention"]},
     "zai-org/GLM-5.3-Flash", "serve-glmf"),
    ({"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
      "layer_types": ["linear_attention", "full_attention"]}}, "test/qwen", "serve-qwen4"),
])
def test_text_only_qualified_family_auto_default_does_not_start_a_tower(tmp_path, config, model, serve):
    result = _family_launch_result(tmp_path, config, model, "SPECULATOR=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if f"cuteafd {serve}" in line)
    assert "--vision auto" in launch
    assert "cuteafd plan" in result.stderr and "--encoder-listen" not in result.stderr


_GLMF = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
         "layer_types": ["linear_attention", "deepseek_sparse_attention"]}


@pytest.mark.parametrize("slot", ["../x", "-s", "a b", ""])
def test_invalid_wip_slot_fails_before_any_container(tmp_path, slot):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "",
                                   extra_args=("--wip", slot))
    assert result.returncode != 0
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


def test_restart_removes_only_own_worker_port(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf",
                                   "EXPERT_PORT=19555\nINSTANCE=own\n", restart=True)
    assert result.returncode == 0, result.stderr
    assert "docker rm -f cuteafd-spark-expert-h0-19555" in result.stderr
    assert "docker ps -aq --filter" not in result.stderr
    assert "docker rm -f cuteafd-spark-expert-h0-19441" not in result.stderr
    assert "docker rm -f cuteafd-spark-expert-wip" not in result.stderr


def test_restart_all_sweeps_workers_and_keeps_the_wip_container(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "",
                                   restart=True, extra_args=("--all",))
    assert result.returncode == 0, result.stderr
    cleanup = next(line for line in result.stderr.splitlines()
                   if line.startswith("ssh h0 ") and 'docker ps -a --format' in line)
    pattern = re.search(r'--filter "?name=([^")\s]+)', cleanup).group(1)
    assert re.search(pattern, "cuteafd-spark-expert-h0-19441")
    assert not re.search(pattern, "cuteafd-spark-expert-wip")
    # Numeric WIP_INSTANCE suffixes can match the worker regex: exclude the
    # entire persistent WIP namespace explicitly before deleting names.
    assert 'grep -vE "^cuteafd-spark-expert-wip($|-)"' in cleanup


def test_rdma_device_map_reaches_the_workers_and_the_coordinator(tmp_path):
    device_map = "10.0.0.9=mlx5_bond_0,10.0.0.1=rocep1s0f1"
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "",
                                   extra_env={"CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP": device_map})
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    worker = next(line for line in lines if "docker run -d --name cuteafd-spark-expert-" in line)
    coordinator = next(line for line in lines if "cuteafd serve-glmf" in line)
    assert f"-e CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP={device_map}" in worker, worker
    assert f"-e CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP={device_map}" in coordinator, coordinator


def test_rdma_device_map_is_not_set_unless_given(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "",
                                   extra_env={"CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP": ""})
    assert result.returncode == 0, result.stderr
    assert "CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP" not in result.stderr


def test_invalid_rdma_device_map_fails_before_workers_launch(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "",
                                   extra_env={"CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP": "10.0.0.1=dev x"})
    assert result.returncode == 2
    assert "local-ip=device" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("with_nest", [False, True])
def test_spark_launch_preserves_page_cache(tmp_path, with_nest):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "", with_nest=with_nest)
    assert result.returncode == 0, result.stderr
    assert "drop_caches" not in result.stderr
    assert "drop-caches" not in result.stderr


def test_serving_source_has_no_page_cache_drop():
    source = (ROOT / "scripts/launch/run-family.sh").read_text()
    assert "drop-caches" not in source
    assert "drop_caches" not in source


@pytest.mark.parametrize("keys,mode", [("", None), ("RDMA_BOND_BALANCE=off\n", None),
                                      ("RDMA_BOND_BALANCE=labels\n", "labels"),
                                      ("RDMA_BOND_BALANCE=probe\n", "probe")])
def test_rdma_bond_balance_reaches_only_the_coordinator(tmp_path, keys, mode):
    """The flow-label switch is the coordinator's: forwarded only when not off. Workers get
    nothing; they connect with whatever label each coordinator connection asks for."""
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", keys)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    launch = next(line for line in lines if "cuteafd serve-glmf" in line)
    assert ("CUTEAFD_RDMA_BOND_BALANCE" in launch) == (mode is not None), launch
    if mode:
        assert f"-e CUTEAFD_RDMA_BOND_BALANCE={mode}" in launch, launch
    workers = [line for line in lines if "docker run -d --name cuteafd-spark-expert-" in line]
    assert workers and not any("CUTEAFD_RDMA_BOND_BALANCE" in line for line in workers)


def test_rdma_bond_balance_rejects_unknown_modes_before_launch(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf",
                                   "RDMA_BOND_BALANCE=yes\n")
    assert result.returncode == 2 and "RDMA_BOND_BALANCE must be off, labels or probe" in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())
@pytest.mark.parametrize("mode,kind", [("auto", "spark"), ("spark:0", "spark"),
                                        ("rtx:0", "rtx"), ("off", "off"), (None, "spark")])
def test_qwen_encoder_explicit_placement_and_default_auto(tmp_path, mode, kind):
    config = {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
              "layer_types": ["linear_attention", "full_attention"]}, "vision_config": {"depth": 27}}
    placement = {"kind": kind}
    if kind == "spark": placement["rank"] = 0
    if kind == "rtx": placement["gpu"] = 0
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": {"kind": placement, "replicas": []}}
    vision_key = f"VISION={mode}\n" if mode is not None else ""
    result = _family_launch_result(tmp_path, config, "test/qwen", f"{vision_key}RTX_GPUS=1\nSPECULATOR=off\n", encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    worker = next(line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line)
    assert f"--vision {kind}" in launch
    assert ("--encoder-listen" in worker) == (kind == "spark")
    assert ("--vision-peers 10.0.0.1:19442" in launch) == (kind == "spark")
    if kind == "spark":
        assert f"--encoder-plan-hash {'ab' * 32}" in worker
        assert f"--encoder-plan-hash {'ab' * 32}" in launch
        assert "--encoder-max-tokens 1024" in worker
        assert "--encoder-revision abc" in worker and "--encoder-revision abc" in launch
    if kind == "off":
        assert "cuteafd plan" in result.stderr


@pytest.mark.parametrize("width", [4096, 6144])
@pytest.mark.parametrize("mode,kind", [(None, "spark"), ("off", "off"), ("auto", "spark")])
def test_qualified_mimo_audio_default_and_explicit_off(tmp_path, width, mode, kind):
    config = {"model_type": "mimo_v2", "hidden_size": width, "num_hidden_layers": 2,
              "moe_layer_freq": [0, 1], "audio_config": {"audio_channels": 20}}
    placement = {"kind": {"kind": "spark", "rank": 0}, "replicas": []}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": None, "audio_encoder": placement}
    keys = "VISION=off\nRTX_GPUS=1\nSPECULATOR=off\n" + (f"AUDIO={mode}\n" if mode else "")
    result = _family_launch_result(tmp_path, config, "test/mimo", keys, encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    expected = "spark:0" if kind == "spark" else "off"
    assert f"--audio {expected}" in launch
    assert ("--audio-peers" in launch) == (kind == "spark")


@pytest.mark.parametrize("tower,reason", [(False, "no tower"), (True, "no device headroom")])
def test_mimo_audio_auto_without_admitted_owner_serves_text(tmp_path, tower, reason):
    config = {"model_type": "mimo_v2", "hidden_size": 4096, "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    if tower:
        config["audio_config"] = {"audio_channels": 20}
    plan = {"placement_supported": True, "fits": True, "spark_ranks": 1,
            "encoder_plan_hash": "ab" * 32, "encoder": None,
            "audio_encoder": {"kind": {"kind": "off"}, "replicas": [], "reason": reason, "shortfall": 1}}
    result = _family_launch_result(tmp_path, config, "test/mimo", "VISION=off\nAUDIO=auto\nRTX_GPUS=1\nSPECULATOR=off\n", encoder_plan=plan)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "--audio off" in launch and "--audio-peers" not in launch
    assert ("audio auto disabled" in result.stderr) == tower


@pytest.mark.parametrize("model_type,width,tower,want", [
    ("mimo_v2", 4096, True, "auto"), ("mimo_v2", 6144, True, "auto"),
    ("mimo_v2", 4096, False, "off"), ("mimo_v2", 128, True, "off"),
    ("deepseek_v41", 4096, True, "off"), ("qwen4_exp", 4096, True, "off"),
    ("glm5_next", 4096, True, "off"), ("deepseek_v4", 4096, True, "off"),
])
def test_release_audio_default_resolves_per_snapshot(tmp_path, model_type, width, tower, want):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": model_type, "hidden_size": width, "audio_config": {"audio_channels": 20}}))
    if tower:
        (snapshot / "audio_tokenizer").mkdir()
        (snapshot / "audio_tokenizer/config.json").write_text("{}")
        (snapshot / "audio_tokenizer/model.safetensors").write_bytes(b"stub")
    config = tmp_path / "release.config"
    config.write_text("SPARK_COUNT=4\n" + "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i+1}\nSPARK_{i}_LANE_B=10.0.1.{i+1}\n" for i in range(4)))
    script = f'source "{ROOT}/scripts/lib/release-common.sh"; release_load_config "$1"; test "$AUDIO" = auto; release_resolve_audio_mode "$AUDIO" "$2"'
    result = subprocess.run(["bash", "-c", script, "bash", str(config), str(snapshot)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == want
    result = subprocess.run(["bash", "-c", f'source "{ROOT}/scripts/lib/release-common.sh"; release_resolve_audio_mode off "$1"', "bash", str(snapshot)], capture_output=True, text=True)
    assert result.stdout.strip() == "off"


@pytest.mark.parametrize("mode", ["spark:0", "rtx"])
def test_explicit_audio_without_tower_refuses(tmp_path, mode):
    config = {"model_type": "mimo_v2", "hidden_size": 4096, "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(tmp_path, config, "test/mimo", f"VISION=off\nAUDIO={mode}\nRTX_GPUS=1\nSPECULATOR=off\n")
    assert result.returncode != 0
    assert "cuteafd serve-mimo" not in result.stderr


@pytest.mark.parametrize("family", ["glm5", "glm5_flash", "qwen4", "mimo_flash", "mimo_pro", "deepseek_v4"])
@pytest.mark.parametrize("context", [None, 65536])
def test_generic_family_checkpoint_context_default_and_override(tmp_path, family, context):
    config = SPLIT_CONFIGS.get(family, {"model_type": "glm_moe_dsa" if family == "glm5" else "deepseek_v4", "num_hidden_layers": 2, "first_k_dense_replace": 1})
    keys = "SPECULATOR=off\nVISION=off\nAUDIO=off\n"
    if context is not None:
        keys += f"MAX_CONTEXT_TOKENS={context}\n"
    result = _family_launch_result(tmp_path, config, "test/model", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert f"--max-context {context or 0}" in launch


def test_pre_108_wip_seal_serves_with_hub_under_hf_home(tmp_path):
    """#111: binaries sealed before #108 resolve drafter snapshots under HF_HOME (falling
    back to a root-only default otherwise). Every planner and serving container the
    launcher starts sets HF_HOME to the namespace HF home, so such a seal plans and
    serves instead of being reported unsupported. Measured on wip-glmf65final:final1
    (9968af44, no #108 ancestry): its `cuteafd plan` refused Spark count 2 without
    HF_HOME and fit with HF_HOME=/hf."""
    _snapshot(tmp_path / "hf", "incoai/GLM-5.3-Flash-DFlash2", {"architectures": ["DFlash2DraftModel"]})
    family_config, model = GLMF_TWO_LAYER, "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-Core-FP8"
    result = _family_launch_result(tmp_path, family_config, model, "VISION=off\nAUDIO=off\n",
                                   extra_args=("--wip", "pre108"))
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    plans = [line for line in lines if "cuteafd plan" in line]
    coordinator = [line for line in lines if "serve-glmf --snapshot" in line]
    assert plans and coordinator
    for line in plans + coordinator:
        assert "HF_HOME=/mnt/sparknest/hf-home" in line and "/mnt/sparknest:/mnt/sparknest:ro" in line, line
        assert "/root/" not in line, line
    # The default drafter is passed as the same resolved host snapshot path.
    assert f"--draft {tmp_path}/hf/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/" in coordinator[0]
    worker = result.stderr.split("release_serving_docker_args ()", 1)[1]
    assert "HF_HOME=/mnt/sparknest/hf-home" in worker and "/mnt/sparknest:/mnt/sparknest:ro" in worker


@pytest.mark.parametrize("wip", [False, True])
@pytest.mark.parametrize("family", ["glm5", "glm5_flash"])
@pytest.mark.parametrize("mode", [None, "layers"])
def test_glm_qualified_layers_release_and_wip_preserve_auto_default(tmp_path, wip, family, mode):
    config = ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}
              if family == "glm5" else SPLIT_CONFIGS["glm5_flash"])
    keys = "RTX_GPUS=2\nVISION=off\nAUDIO=off\nSPECULATOR=off\n"
    if mode is not None:
        keys += f"ATTENTION_PLACEMENT={mode}\n"
    result = _family_launch_result(tmp_path, config, "test/model", keys,
                                  extra_args=("--wip", "slot-test") if wip else ())
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert f"CUTEAFD_ATTENTION_PLACEMENT={mode or 'auto'}" in launch
    plan = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    assert f"--attention-placement {mode or 'auto'}" in plan
    assert "under qualification" not in result.stderr
    assert ("layers explicitly selected; auto remains context on two GPUs" in result.stderr) == (mode == "layers")


@pytest.mark.parametrize("wip", [False, True])
def test_deepseek_v4_layers_is_unsupported_in_release_and_wip(tmp_path, wip):
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v4"}, "test/dsv4",
                                  "ATTENTION_PLACEMENT=layers\n",
                                  extra_args=("--wip", "slot-test") if wip else ())
    assert result.returncode != 0
    assert "deepseek_v4 cannot run attention placement layers: executor not implemented" in result.stderr
    assert "docker run" not in result.stderr


@pytest.mark.parametrize("keys,expected", [
    ("SPECULATOR=off\n", "off"),
    ("SPECULATOR=dspark\n", "dspark"),
])
def test_family_plan_charges_the_speculator_this_launch_serves(tmp_path, keys, expected):
    # A launch with SPECULATOR=off must not be planned as if it carried the family's
    # default drafter (V4 would otherwise charge an unused MTP stage and refuse layouts).
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v4"}, "test/dsv4", keys)
    assert result.returncode == 0, result.stderr
    plan = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    assert f"--speculator {expected}" in plan


@pytest.mark.parametrize("value,enabled", [("dspark", True), ("off", False)])
def test_v41_uses_canonical_speculator(tmp_path: Path, value: str, enabled: bool) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v41"},
        "deepseek-ai/DeepSeek-V4.1-Flash", f"MODEL_REVISION=\nSPECULATOR={value}\nVISION=off\nRTX_GPUS=1\nRTX_EXPERT_LAYERS=0\nSPARK_COUNT=2\nSPARK_1_HOST=h1\nSPARK_1_LANE_A=10.0.0.2\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-native" in line and "docker run -d" in line)
    assert ("--dspark" in launch) == enabled


@pytest.mark.parametrize("keys,error", [("DSPARK=on\n", "unknown configuration key: DSPARK"),
    ("SPECULATOR=mtp\n", "SPECULATOR must be dspark or off for DeepSeek V4.1")])
def test_v41_rejects_noncanonical_speculator(tmp_path: Path, keys: str, error: str) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v41"},
        "deepseek-ai/DeepSeek-V4.1-Flash", keys)
    assert result.returncode == 2, result.stderr
    assert error in result.stderr
    assert "docker run -d" not in result.stderr


@pytest.mark.parametrize("value,expected", [("48", "48"), ("all", "60"), ("50%", "30")])
def test_expert_onboarding_uses_checkpoint_count_above_forty(tmp_path, value, expected):
    result = _family_launch_result(tmp_path, {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 63, "first_k_dense_replace": 3,
    }, "test/glm", f"RTX_EXPERT_LAYERS={value}\nSPECULATOR=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glm " in line)
    assert f"--rtx-expert-layers {expected}" in launch


def test_expert_onboarding_rejects_more_than_checkpoint_routed_layers(tmp_path):
    result = _family_launch_result(tmp_path, {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 63, "first_k_dense_replace": 3,
    }, "test/glm", "RTX_EXPERT_LAYERS=61\nSPECULATOR=off\n")
    assert result.returncode == 2
    assert "checkpoint has 60 routed layers" in result.stderr
    assert "docker run" not in result.stderr


@pytest.mark.parametrize("value", [None, "off", "on"])
def test_console_assets_serve_checkout_pages_only_for_wip(tmp_path: Path, value) -> None:
    keys = "SPARK_COUNT=0\n" + (f"CONSOLE_ASSETS={value}\n" if value else "")
    result = _family_launch_result(tmp_path, DENSE_CONFIG, "Qwen/Qwen3.5-9B", keys,
        extra_args=("--wip", "dense-test"),
        encoder_plan={"placement_supported": True, "fits": True, "spark_ranks": 0})
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-qwen35 --snapshot" in line)
    pages = ("CUTEAFD_CONSOLE_PAGE=/run/cuteafd-assets/cuteafd-api/console.html",
             "CUTEAFD_CONSOLE_ASSETS=/run/cuteafd-assets/cuteafd-api",
             "CUTEAFD_BENCH_PAGE=/run/cuteafd-assets/cuteafd-bench/bench.html",
             "CUTEAFD_USAGE_PAGE=/run/cuteafd-assets/cuteafd-usage/usage.html")
    if value == "on":
        for crate in ("cuteafd-api", "cuteafd-bench", "cuteafd-usage"):
            assert f"rust/crates/{crate}/assets,dst=/run/cuteafd-assets/{crate},readonly" in launch
        assert all(page in launch for page in pages)
    else:
        assert "/run/cuteafd-assets" not in launch and "CUTEAFD_CONSOLE_PAGE" not in launch


def test_console_assets_refuse_release_images(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                                            "moe_layer_freq": [0, 1]}, "XiaomiMiMo/MiMo-V2-Flash", "CONSOLE_ASSETS=on\n")
    assert result.returncode != 0
    assert "CONSOLE_ASSETS=on serves pages from a checkout and needs --wip SLOT" in result.stderr
    assert "docker run -d" not in result.stderr


def test_console_assets_rejects_unknown_value(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, DENSE_CONFIG, "Qwen/Qwen3.5-9B",
        "SPARK_COUNT=0\nCONSOLE_ASSETS=maybe\n", extra_args=("--wip", "dense-test"),
        encoder_plan={"placement_supported": True, "fits": True, "spark_ranks": 0})
    assert result.returncode != 0
    assert "CONSOLE_ASSETS must be on or off" in result.stderr


def test_worker_env_survives_the_remote_worker_shell(tmp_path: Path) -> None:
    # Shared worker env reaches the ssh remote command as shell-quoted words.
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 3, "moe_layer_freq": [0, 1, 1]}
    result = _family_launch_result(tmp_path, config, "XiaomiMiMo/MiMo-V2-Flash", "SPECULATOR=off\n")
    assert result.returncode == 0, result.stderr
    workers = [line for line in result.stderr.splitlines() if "expertd-native" in line]
    assert workers and all("CUTEAFD_THERMAL=" in worker for worker in workers)


@pytest.mark.parametrize("family", ["glm5_flash", "glm5", "mimo_v2", "deepseek_v4", "qwen4", "deepseek_v41"])
@pytest.mark.parametrize("budget", ["", "96", "90"])
def test_launcher_plans_against_cuda_total_not_nvml_total(tmp_path, family, budget):
    config = ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}
              if family == "glm5" else {"model_type": "deepseek_v4"} if family == "deepseek_v4"
              else {"model_type": "deepseek_v41"} if family == "deepseek_v41"
              else SPLIT_CONFIGS["mimo_flash" if family == "mimo_v2" else family])
    total = 101_973_491_712
    result = _family_launch_result(tmp_path, config,
        "deepseek-ai/DeepSeek-V4.1-Flash" if family == "deepseek_v41" else "test/model",
        f"COORDINATOR_GPU_BUDGET_GIB={budget}\nRTX_GPUS=1\nSPECULATOR=off\nVISION=off\nAUDIO=off\n"
        + ("SPARK_COUNT=4\n" + "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\n" for i in range(4))
           if family == "deepseek_v41" else ""),
        gpu_total_mib=97887, gpu_free_mib=97887, cuda_total_bytes=total,
        extra_args=("--wip", "capacity-test") if family == "deepseek_v41" else ())
    assert result.returncode == 0, result.stderr
    plan = next(line for line in result.stderr.splitlines() if " plan " in line and "--deployment" in line)
    args = plan.split()
    expected = min(total / 2**30, float(budget) if budget else float("inf"))
    assert float(args[args.index("--coordinator-gpu-budget-gib") + 1]) == expected
    assert expected < 97887 / 1024


@pytest.mark.parametrize("restart", [False, True])
def test_qwen_external_occupancy_is_nvml_used_subtracted_from_cuda_capacity(tmp_path, restart):
    total = 101_973_491_712
    result = _family_launch_result(tmp_path,
        {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": "exl3"}},
        "test/model", "INSTANCE=own\nSPARK_COUNT=0\n", preferred_ranks=0, restart=restart,
        gpu_total_mib=97887, gpu_free_mib=12000, cuda_total_bytes=total,
        container_pids=(123,), gpu_allocations=((123, 84000), (456, 2000)))
    assert result.returncode == 0, result.stderr
    plan = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line).split()
    used = (97887 - 12000 - (84000 if restart else 0)) * 2**20
    assert float(plan[plan.index("--coordinator-gpu-budget-gib") + 1]) == (total - used) / 2**30


def test_namespace_root_and_push_endpoint_reach_both_serving_roles(tmp_path):
    root = tmp_path / "namespace"
    hf = root / "hf-home"
    model = "zai-org/GLM-5.3-Flash"
    _snapshot(hf, model, GLMF_TWO_LAYER)
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, model,
        "SPECULATOR=off\nVISION=off\nAUDIO=off\n",
        extra_env={"CUTEAFD_NAMESPACE_ROOT": str(root), "CUTEAFD_PUSH_SOCKET": "@isolated-push"})
    assert result.returncode == 0, result.stderr
    for line in result.stderr.splitlines():
        if "docker run" in line and ("cuteafd plan" in line or "serve-glmf --snapshot" in line):
            assert f"{root}:{root}:ro" in line
            assert f"HF_HOME={hf}" in line
            assert f"{hf}/hub/models--zai-org--GLM-5.3-Flash/snapshots/abc" in line
    assert result.stderr.count("CUTEAFD_PUSH_SOCKET=@isolated-push") >= 2
    assert f"--snapshot '{hf}/hub/models--zai-org--GLM-5.3-Flash/snapshots/abc'" in result.stderr


@pytest.mark.parametrize("format", ["native", "fp8", "exlr8"])
def test_push_group_appends_coordinator_suffix_before_one_finish_and_upload(format: str) -> None:
    source = (ROOT / f"rust/crates/cuteafd-daemon/src/shared/memory/push_{format}.rs").read_text()
    plan = source.split("fn checked_plan(", 1)[1].split("pub(crate) fn prepare(", 1)[0]
    prepare = source.split("pub(crate) fn prepare(", 1)[1]
    assert "let sparks = assignments.len();" in plan
    assert "offers[sparks..]" in plan
    assert plan.count("builder.finish()?") == 1
    assert plan.index("plan.append(") < plan.index("(coordinator.append)") < plan.index("builder.finish()?")
    assert "offers.len()" not in plan.split("let mut plans", 1)[1]
    assert prepare.count("checked_plan(") == 1
    assert prepare.count(".create(") == 1
    assert prepare.count("checked.upload(&group)?") == 1
    assert "builder.finish(" not in prepare
    assert prepare.index("checked_plan(") < prepare.index(".create(") < prepare.index("checked.upload(")


@pytest.mark.parametrize("model_type,model", [
    ("mimo_v2_flash", "XiaomiMiMo/MiMo-V2-Flash"),
    ("mimo_v2", "XiaomiMiMo/MiMo-V2.6-Pro-MOPD"),
])
@pytest.mark.parametrize("value", [None, "bf16", "int8", "fp8"])
def test_mimo_swa_record_format_reaches_serve_and_plan(tmp_path, value, model_type, model):
    config = {"model_type": model_type, "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    keys = "SPECULATOR=off\n" + (f"SWA_KV_CACHE={value}\n" if value else "")
    result = _family_launch_result(tmp_path, config, model, keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    expected = value or "int8"
    assert f"--swa-kv-cache {expected}" in launch
    plans = [line for line in result.stderr.splitlines() if "cuteafd plan" in line]
    assert plans and all(f"--mimo-swa-kv-cache {expected}" in line for line in plans)


@pytest.mark.parametrize("full", ["off", "on"])
@pytest.mark.parametrize("kv", [None, "bf16", "fp8"])
def test_dense_full_prefill_logits_reaches_serve_and_plan(tmp_path, full, kv):
    result = _family_launch_result(tmp_path, DENSE_CONFIG, "Qwen/Qwen3.5-9B",
        f"SPARK_COUNT=0\nSPECULATOR=off\nFULL_PREFILL_LOGITS={full}\n" + (f"KV_CACHE={kv}\n" if kv else ""),
        extra_args=("--wip", "dense-test"),
        encoder_plan={"placement_supported": True, "fits": True, "spark_ranks": 0})
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-qwen35 --snapshot" in line)
    plan = next(line for line in result.stderr.splitlines() if "cuteafd plan " in line)
    assert ("--full-prefill-logits" in launch) == (full == "on")
    assert ("--full-prefill-logits" in plan) == (full == "on")
    expected_kv = kv or "fp8"
    assert f"--kv-cache {expected_kv}" in launch
    assert f"--kv-cache {expected_kv}" in plan


def test_dense_qwen_forwards_speculation_trace(tmp_path: Path) -> None:
    # Dense Qwen writes the trace through the shared Qwen scheduler; the card's
    # --trace sets SPECULATION_TRACE, which the launcher must mount and pass on.
    trace = tmp_path / "trace" / "speculation-trace.jsonl"
    result = _family_launch_result(tmp_path, DENSE_CONFIG, "Qwen/Qwen3.5-9B",
        f"SPARK_COUNT=0\nSPECULATION_TRACE={trace}\n",
        extra_args=("--wip", "dense-test"),
        encoder_plan={"placement_supported": True, "fits": True, "spark_ranks": 0})
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-qwen35 --snapshot" in line)
    assert f"CUTEAFD_SPECULATION_TRACE={trace}" in launch
    assert f"{trace.parent}:{trace.parent}" in launch
