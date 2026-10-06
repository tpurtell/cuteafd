"""run.sh hands every family but DeepSeek V4.1 to scripts/launch/run-family.sh."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / "scripts" / "launch").mkdir(parents=True)
    shutil.copy(ROOT / "run.sh", repo / "run.sh")
    for name in ("release-common.sh", "checkpoint-family.py"):
        shutil.copy(ROOT / "scripts" / "lib" / name, repo / "scripts" / "lib" / name)
    fake = repo / "scripts" / "launch" / "run-family.sh"
    fake.write_text('#!/usr/bin/env bash\nprintf "run-family %s\\n" "$*"\n')
    fake.chmod(0o755)
    return repo


def _snapshot(hf: Path, model: str, config: dict) -> None:
    root = hf / "hub" / f"models--{model.replace('/', '--')}"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("abc")
    (root / "snapshots" / "abc").mkdir(parents=True)
    (root / "snapshots" / "abc" / "config.json").write_text(json.dumps(config))


def _run(repo: Path, hf: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HF_HOME": str(hf)}
    return subprocess.run(["bash", str(repo / "run.sh"), *args], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=60)


def test_glm_flash_config_goes_to_run_family(tmp_path: Path) -> None:
    repo, hf = _repo(tmp_path), tmp_path / "hf"
    _snapshot(hf, "zai-org/GLM-5.3-Flash", {"model_type": "glm5_next", "num_hidden_layers": 45,
                                           "layer_types": ["linear_attention"] * 45,
                                           "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42})
    (repo / "glmf.config").write_text("MODEL_ID=zai-org/GLM-5.3-Flash\nDRAFT_MODEL_ID=incoai/x\n")
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--restart")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"run-family --config {repo / 'glmf.config'} --family glm5_flash --restart\n"
    # DeepSeek V4.1 options do not apply to other families.
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--concurrency", "4")
    assert result.returncode != 0 and "take --config, --restart and --embedding-placement" in result.stderr
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--embedding-placement", "host")
    assert result.returncode == 0, result.stderr
    assert "--embedding-placement host" in result.stdout


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
                          container_pids: tuple[int, ...] = (),
                          gpu_allocations: tuple[tuple[int, int], ...] = (),
                          previous_peers: str | None = None, encoder_plan: dict | None = None) -> subprocess.CompletedProcess[str]:
    """Run the real run-family.sh up to its docker calls (docker/ssh/nest/curl are stubs
    that print their argv) and return what it would launch."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / "scripts" / "launch").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "launch" / "run-family.sh", repo / "scripts" / "launch")
    shutil.copy(ROOT / "scripts" / "launch" / "preflight-fp8-bf16.py", repo / "scripts" / "launch")
    shutil.copy(ROOT / "scripts" / "lib" / "checkpoint-family.py", repo / "scripts" / "lib")
    shutil.copy(ROOT / "scripts" / "lib" / "release-common.sh", repo / "scripts" / "lib")
    hf = tmp_path / "hf"
    _snapshot(hf, model, family_config)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("docker", "ssh", "nest"):
        (bin_dir / tool).write_text('#!/usr/bin/env bash\nprintf "%s " "$(basename "$0")" "$@" >&2; echo >&2\n'
                                    + ('case "$*" in *"docker run --rm"*"python3"*) exit 2 ;; esac\n'
                                       if preflight_error and tool == "ssh" else '') +
                                    'case "$*" in *"docker logs"*) echo "worker ready" ;; esac\n' +
                                    (f"case \"$*\" in *\"cuteafd plan\"*) echo '{{\"spark_ranks\":{preferred_ranks}}}' ;; esac\n"
                                     if tool == "docker" and preferred_ranks is not None else '') +
                                    (f"case \"$*\" in *\"cuteafd plan\"*) printf '%s\\n' '{json.dumps(encoder_plan)}' ;; esac\n"
                                     if tool == "docker" and encoder_plan is not None else '') +
                                    ("case \"$*\" in top*) printf '%s\\n' PID " +
                                     " ".join(map(str, container_pids)) + " ;; esac\n"
                                     if tool == "docker" and container_pids else '') +
                                    (f"case \"$*\" in inspect*) echo '[\"serve-qwen4\",\"--peers\",\"{previous_peers}\"]' ;; esac\n"
                                     if tool == "docker" and previous_peers is not None else ''))
        (bin_dir / tool).chmod(0o755)
    (bin_dir / "curl").write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' \'{"data":[{"id":"test/model"}]}\'\n')
    (bin_dir / "curl").chmod(0o755)
    (bin_dir / "nvidia-smi").write_text("#!/usr/bin/env bash\nprintf '%s\\n' " +
                                         " ".join(map(str, physical_gpus)) + "\n" if preferred_ranks is None else
                                         '#!/usr/bin/env bash\ncase "$*" in *memory.free*) echo ' + str(gpu_free_mib) +
                                         ' ;; *memory.total*) echo ' + str(gpu_total_mib) +
                                         ' ;; *query-compute-apps*) printf \'%s\\n\' ' +
                                         " ".join(f"'{pid}, {mib}'" for pid, mib in gpu_allocations) +
                                         ' ;; *) echo 0; echo 1 ;; esac\n')
    (bin_dir / "nvidia-smi").chmod(0o755)
    config = repo / "f.config"
    config.write_text(f"MODEL_ID={model}\nSPARK_COUNT=1\nSPARK_0_HOST=h0\nSPARK_0_LANE_A=10.0.0.1\n{keys}")
    env = {**os.environ, "HF_HOME": str(hf), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(repo / "scripts" / "launch" / "run-family.sh"), "--config", str(config),
                           *(["--restart"] if restart else [])],
                          env=env, capture_output=True, text=True, timeout=30)


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


def _family_launch_lines(tmp_path: Path, family_config: dict, model: str, keys: str) -> str:
    return _family_launch_result(tmp_path, family_config, model, keys).stderr


@pytest.mark.parametrize("value", [None, "auto", "0", "5"])
def test_deepseek_v4_honors_explicit_local_expert_limit(tmp_path, value):
    keys = "" if value is None else f"RTX_EXPERT_LAYERS={value}\n"
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v4"}, "test/dsv4", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-dsv4" in line)
    if value in (None, "auto"):
        assert "--local-expert-layers" not in launch
    else:
        assert f"--local-expert-layers {value}" in launch


def test_deepseek_v4_rejects_invalid_local_limit_before_launch(tmp_path):
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v4"}, "test/dsv4",
                                  "RTX_EXPERT_LAYERS=-1\n")
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
                                  "GLM5_FLASH_FP8_MODEL_ID=off\n" + layout + precision,
                                  physical_gpus=physical_gpus)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert ("--split-device" in launch) == split
    # Both layouts default to row128 KDA and an FP8 head; the split adds
    # token-row KDA output ownership.
    assert "--kda-fp8 row128" in launch
    assert "--fp8-head true" in launch
    assert launch.count("--kda-fp8") == launch.count("--fp8-head") == 1
    assert ("--kda-output-shard --kda-prefill-expanded" in launch) == split


@pytest.mark.parametrize("keys,shard", [("", True), ("GLM5_FLASH_KDA_SPLIT=partials\n", False),
                                        ("GLM5_FLASH_KDA_FP8=off\n", False)])
def test_glmf_split_token_rows_follow_fp8_kda(tmp_path, keys, shard):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nRTX_GPUS=2\n" + keys)
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
                                  "GLM5_FLASH_FP8_MODEL_ID=off\n" + layout + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert f"--kda-fp8 {kda}" in launch
    assert f"--fp8-head {'true' if head == 'on' else 'false'}" in launch
    assert launch.count("--kda-fp8") == launch.count("--fp8-head") == 1


@pytest.mark.parametrize("layout,keys,kda,head", [
    ("RTX_GPUS=1\n", "GLM5_FLASH_KDA_FP8=off\n", "off", "true"),
    ("RTX_GPUS=1\n", "GLM5_FLASH_FP8_HEAD=off\n", "row128", "false"),
    ("RTX_GPUS=2\n", "GLM5_FLASH_KDA_FP8=off\n", "off", "true"),
    ("RTX_GPUS=2\n", "GLM5_FLASH_FP8_HEAD=off\n", "row128", "false"),
])
def test_glmf_precision_overrides_are_independent(tmp_path, layout, keys, kda, head):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\n" + layout + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert f"--kda-fp8 {kda}" in launch
    assert f"--fp8-head {head}" in launch


@pytest.mark.parametrize("layout,accepted", [("RTX_GPUS=1\n", True), ("RTX_GPUS=2\n", False)])
def test_glmf_kda_prefill_validation_uses_resolved_precision(tmp_path, layout, accepted):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nGLM5_FLASH_FP8_PREFILL=all\n" + layout)
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
    result = _family_launch_result(tmp_path, config, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    for option in expected:
        assert option in launch, launch


@pytest.mark.parametrize("key,value,message", [
    ("GLM5_FLASH_FP8_PREFILL", "all", "GLM5_FLASH_KDA_FP8=row128"),
    ("GLMF_FP8_PREFILL", "mla,kda-in", "GLM5_FLASH_KDA_FP8=row128"),
    ("GLM5_FLASH_FP8_PREFILL", "kda-o,ffn", "GLM5_FLASH_KDA_FP8=row128"),
    ("GLM5_FLASH_FP8_HEAD", "maybe", "GLM5_FLASH_FP8_HEAD must be"),
])
def test_glmf_invalid_fp8_options_fail_before_workers_launch(tmp_path, key, value, message):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf",
                                  f"GLM5_FLASH_FP8_MODEL_ID=off\nGLM5_FLASH_KDA_FP8=off\n{key}={value}\n")
    assert result.returncode == 2 and message in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


def test_glmf_kda_rejects_invalid_conversion_before_launch(tmp_path):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nGLM5_FLASH_KDA_FP8=invalid\n")
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


@pytest.mark.parametrize("family", ["glm5", "glm5_flash"])
def test_glm_drafter_context_slots_and_batch_are_independent(tmp_path, family):
    config = ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}
              if family == "glm5" else SPLIT_CONFIGS["glm5_flash"])
    model = "test/model" if family == "glm5" else "zai-org/GLM-5.3-Flash"
    result = _family_launch_result(tmp_path, config, model,
                                  f"SPECULATOR=dflash2\nSPECULATOR_MODEL_ID={model}\n"
                                  "DRAFT_CONTEXT_SLOTS=20\nDRAFT_SEQUENCES=16\nDRAFT_FP8=on\n")
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
        assert "docker run --rm" not in text
        assert ("--expert-input fp8" in text) == bool(mode)
    other = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
             "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path / "other", other, "test/glm", "EXPERT_INPUT=bf16-decode\nGLM5_FLASH_FP8_MODEL_ID=off\n")
    assert result.returncode == 2 and "applies to MiMo" in result.stderr
    assert "docker run" not in result.stderr


@pytest.mark.parametrize("key", ["SPECULATOR_FP8", "DRAFT_FP8"])
@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("auto", None),
                                            ("on", "true"), ("off", "false")])
def test_mimo_drafter_precision_preserves_auto_and_forwards_explicit_conversion(tmp_path, key, value, expected):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    keys = "SPECULATOR=dflash2\n"
    if value is not None:
        keys += f"{key}={value}\n"
    result = _family_launch_result(tmp_path, config, "test/mimo", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    if expected is None:
        assert "--draft-fp8" not in launch
        assert ("--draft-representation checkpoint" in launch) == (value == "auto")
    else:
        assert f"--draft-fp8 {expected}" in launch


@pytest.mark.parametrize("policy, expected", [(None, None), ("auto", None), ("checkpoint", "checkpoint")])
def test_mimo_weight_policy_is_resolved_by_runtime_and_explicit_checkpoint_is_forwarded(tmp_path, policy, expected):
    config = {"model_type": "mimo_v2", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    keys = "" if policy is None else f"MIMO_WEIGHT_POLICY={policy}\n"
    result = _family_launch_result(tmp_path, config, "arbitrary/local-mimo", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert "--fp8-head" not in launch and "--fp8-o-proj" not in launch
    if expected is None:
        assert "--weight-policy" not in launch
    else:
        assert f"--weight-policy {expected}" in launch


@pytest.mark.parametrize("quota", [None, "4MiB", "0"])
def test_mimo_embedding_cache_quota_is_explicit_only(tmp_path, quota):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    keys = "" if quota is None else f"MEDIA_CACHE_BYTES={quota}\n"
    result = _family_launch_result(tmp_path, config, "test/mimo", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    if quota is None:
        assert "--media-cache-bytes" not in launch
    else:
        assert f"--media-cache-bytes {quota}" in launch


@pytest.mark.parametrize("key, option", [("MIMO_FP8_HEAD", "--fp8-head"), ("MIMO_FP8_O_PROJ", "--fp8-o-proj")])
@pytest.mark.parametrize("value, expected", [("auto", None), ("on", "true"), ("off", "false")])
def test_mimo_explicit_target_format_overrides_are_forwarded(tmp_path, key, option, value, expected):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(tmp_path, config, "test/mimo", f"{key}={value}\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
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


@pytest.mark.parametrize("checkpoint", ["qwen4"])
@pytest.mark.parametrize("keys", ["RTX_GPUS=2\n", "COORDINATOR_GPUS=0,1\n", "COORDINATOR_SPLIT=heads\n"])
def test_explicit_split_without_kernels_serves_from_the_first_gpu(
        tmp_path: Path, checkpoint: str, keys: str) -> None:
    model = "zai-org/GLM-5.3-Flash" if checkpoint == "glm5_flash" else "test/model"
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS[checkpoint], model, keys)
    assert result.returncode == 0, result.stderr
    assert "has no head split yet" in result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert "device=0" in launch and "--split-device" not in launch


@pytest.mark.parametrize("checkpoint", ["qwen4"])
def test_auto_keeps_unsupported_checkpoint_on_one_gpu(tmp_path: Path, checkpoint: str) -> None:
    model = "zai-org/GLM-5.3-Flash" if checkpoint == "glm5_flash" else "test/model"
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS[checkpoint], model, "")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert "device=0" in launch and "--split-device" not in launch
    assert "auto selected GPU 0 alone" in result.stderr


def test_split_off_explicitly_uses_the_first_gpu(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  "RTX_GPUS=2\nCOORDINATOR_GPUS=1,0\nCOORDINATOR_SPLIT=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "device=1" in launch and "--split-device" not in launch


@pytest.mark.parametrize("keys", ["RTX_GPUS=2\n", "COORDINATOR_GPUS=1,0\n", "COORDINATOR_SPLIT=heads\n"])
def test_glm_flash_explicit_split_passes_both_gpus(tmp_path: Path, keys: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "zai-org/GLM-5.3-Flash", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert "--split-device" in launch and "device=0,1" in launch


@pytest.mark.parametrize("checkpoint", ["mimo_flash", "mimo_pro"])
def test_both_mimo_model_types_keep_their_supported_split(tmp_path: Path, checkpoint: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS[checkpoint], "test/model",
                                  "COORDINATOR_GPUS=1,0\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert "--device 1 --split-device 0" in launch and "device=0,1" in launch


def test_auto_uses_one_gpu_when_only_one_exists(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["mimo_pro"], "test/model", "", physical_gpus=(0,))
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert "device=0" in launch and "--split-device" not in launch


def test_split_off_needs_only_the_first_physical_gpu(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  "RTX_GPUS=2\nCOORDINATOR_SPLIT=off\n", physical_gpus=(0,))
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "device=0" in launch and "--split-device" not in launch


@pytest.mark.parametrize("keys", ["RTX_GPUS=2\n", "COORDINATOR_GPUS=0,1\n",
                                 "COORDINATOR_SPLIT=heads\n"])
def test_explicit_split_requires_two_physical_gpus(tmp_path: Path, keys: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["mimo_pro"], "test/model", keys,
                                  physical_gpus=(0,))
    assert result.returncode == 2, result.stderr
    assert "physical" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


def test_speculator_and_its_pre_rename_keys_launch_the_same(tmp_path: Path) -> None:
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 3, "moe_layer_freq": [0, 1, 1]}
    new = _family_launch_lines(tmp_path / "a", config, "XiaomiMiMo/MiMo-V2-Flash", "SPECULATOR=mtp\nSPECULATOR_DEPTH=2\n")
    old = _family_launch_lines(tmp_path / "b", config, "XiaomiMiMo/MiMo-V2-Flash", "MTP=2\n")
    launch = lambda text: [l for l in text.splitlines() if "cuteafd serve-mimo" in l]
    assert launch(new) and "--mtp 2" in launch(new)[0]
    assert [l.replace(str(tmp_path / "b"), "X") for l in launch(old)] == \
        [l.replace(str(tmp_path / "a"), "X") for l in launch(new)]
    assert "deprecated" in old and "deprecated" not in new
    bad = _family_launch_lines(tmp_path / "c", config, "XiaomiMiMo/MiMo-V2-Flash", "SPECULATOR=dspark\n")
    assert "does not apply to mimo_v2" in bad


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
    assert "--host-cache-bytes" not in launch[0] and "--pool-tokens" not in launch[0]


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
    assert "--host-cache-bytes" not in launch[0] and "--pool-tokens" not in launch[0]


def test_glm_flash_drafts_with_its_default_speculator(tmp_path: Path) -> None:
    config = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    model = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1"
    default = "incoai/GLM-5.3-Flash-DFlash2"
    dspark = "RedHatAI/GLM-5.3-Flash-speculator.dspark-preview"
    keys = "GLM5_FLASH_FP8_MODEL_ID=off\n"

    def launch(sub: str, extra: str, drafters: tuple[str, ...] = (default, dspark)) -> tuple[str, list[str]]:
        hf = tmp_path / sub / "hf"
        for name in drafters:
            _snapshot(hf, name, {"speculators_model_type": "dspark"} if name == dspark else {})
        text = _family_launch_lines(tmp_path / sub, config, model, keys + extra)
        return text, [l for l in text.splitlines() if "cuteafd serve-glmf" in l]

    snap = lambda name: f"--draft /root/.cache/huggingface/hub/models--{name.replace('/', '--')}/snapshots/abc"
    text, lines = launch("a", "")
    assert lines and snap(default) in lines[0], text
    assert "drafts with dflash2" in text
    text, lines = launch("b", "SPECULATOR=off\n")
    assert lines and "--draft" not in lines[0], text
    text, lines = launch("c", f"SPECULATOR=dspark\nSPECULATOR_MODEL_ID={dspark}\n")
    assert lines and snap(dspark) in lines[0], text
    text, lines = launch("d", "", ())
    assert lines and "--draft" not in lines[0] and "hf download " + default in text, text


def test_glmf_pool_defaults_to_the_planned_pool(tmp_path: Path) -> None:
    config = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    for keys, flag in (("", "--pool-tokens 0"), ("POOL_TOKENS=auto\n", "--pool-tokens 0"),
                       ("POOL_TOKENS=65536\n", "--pool-tokens 65536")):
        result = _family_launch_result(tmp_path / str(len(keys)), config, "test/glmf",
                                       "GLM5_FLASH_FP8_MODEL_ID=off\n" + keys)
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
    launch = next(line for line in result.stderr.splitlines() if "--coordinator-gpu-budget-gib" in line)
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
        "COORDINATOR_GPU_BUDGET_GIB=32\n", preferred_ranks=4,
        gpu_free_mib=90 * 1024, gpu_total_mib=96 * 1024)
    assert result.returncode == 0, result.stderr
    preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    assert "--rtx-gib 26.0" in preflight and "--coordinator-budget-gib 26.0" in preflight
    launch = next(line for line in result.stderr.splitlines() if "serve-qwen4" in line)
    assert "--coordinator-gpu-budget-gib 32" in launch and "--peers" in launch


def test_budget_key_is_accepted_for_cleanup_and_native_launchers(tmp_path):
    config = tmp_path / "budget.config"
    config.write_text("COORDINATOR_GPU_BUDGET_GIB=32\n")
    result = subprocess.run(["bash", "-c", 'source "$1"; release_load_config "$2" stop; printf "%s" "$COORDINATOR_GPU_BUDGET_GIB"',
        "bash", str(ROOT / "scripts/lib/release-common.sh"), str(config)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout == "32", result.stderr
    for file in ("run.sh", "scripts/launch/run-tp-ep-native-candidate.sh"):
        assert 'args+=(--coordinator-gpu-budget-gib "$COORDINATOR_GPU_BUDGET_GIB")' in (ROOT / file).read_text()


@pytest.mark.parametrize("keys,expected", [
    ("", "fp8"), ("SPECULATOR=dflash2\n", "fp8"),
    ("SPECULATOR_FP8=auto\n", "checkpoint"), ("SPECULATOR_FP8=off\n", "bf16"),
    ("DRAFT_FP8=off\n", "bf16"), ("SPECULATOR=off\n", "off"),
    ("MTP=0\n", "off"), ("DFLASH=off\n", "off"), ("SPECULATOR=mtp\n", "mtp"),
])
def test_flash_mopd_defaults_to_its_qualified_bundled_drafter(tmp_path: Path, keys: str, expected: str) -> None:
    config = {"model_type": "mimo_v2", "hidden_size": 4096, "num_hidden_layers": 2,
              "moe_layer_freq": [0, 1], "quantization_config": {"store_dtype": "mxfp4"}}
    model = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD"
    result = _family_launch_result(tmp_path, config, model, keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert "--fp8-head" not in launch and "--fp8-o-proj" not in launch
    if expected in ("off", "mtp"):
        assert "--draft" not in launch
        assert ("--mtp 1" in launch) == (expected == "mtp")
    else:
        assert "--draft /root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-MOPD/snapshots/abc" in launch
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
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
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
    keys.update(("MIMO_FP8_HEAD", "MIMO_FP8_O_PROJ", "QWEN_FP8_DECODE", "QWEN_FP8_HEAD", "V41_COPY_DRAFTS"))
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
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert ("--local-experts" in launch) == local
    assert ("--peers" in launch) != local
    assert ("--mtp 3" in launch) == local
    if local:
        assert "expertd-native" not in result.stderr


def test_qwen_preferred_local_allows_native_mtp(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": "exl3"}}, "test/model",
                                  "SPECULATOR=mtp\nSPECULATOR_DEPTH=3\n", preferred_ranks=0)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--mtp 3" in launch and "--local-experts" in launch


@pytest.mark.parametrize("keys,depth", [("", 3), ("SPECULATOR=mtp\n", 3),
                                      ("SPECULATOR_DEPTH=2\n", 2), ("MTP=2\n", 2),
                                      ("SPECULATOR=off\n", None), ("MTP=0\n", None)])
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
                                  "test/model", "INSTANCE=own\n", preferred_ranks=0, restart=restart,
                                  gpu_free_mib=12000, container_pids=(123,),
                                  gpu_allocations=((123, 84000), (456, 2000)))
    assert result.returncode == 0, result.stderr
    preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    assert f"--rtx-gib {admitted_gib}" in preflight


def test_qwen_other_formats_skip_the_local_qualification_preflight(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  "", preferred_ranks=0)
    assert result.returncode == 0, result.stderr
    assert "cuteafd plan" not in result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "--peers" in launch and "--local-experts" not in launch


@pytest.mark.parametrize("previous_peers,cleanup", [(None, False), ("10.0.0.1:19555", True),
                                                   ("10.0.0.9:19555", False)])
def test_qwen_local_restart_releases_only_its_previous_workers(tmp_path: Path, previous_peers: str | None,
                                                             cleanup: bool) -> None:
    result = _family_launch_result(tmp_path, {**SPLIT_CONFIGS["qwen4"], "quantization_config": {"quant_method": "exl3"}},
                                  "test/model", "INSTANCE=own\n", preferred_ranks=0, restart=True,
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
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert ("--encoder-listen" in worker) == (kind == "spark")
    assert ("--vision-peers 10.0.0.1:19442" in launch) == (kind == "spark")
    assert f"--vision {kind}" in launch
    if kind == "spark":
        assert f"--encoder-plan-hash {'ab' * 32}" in worker
        assert f"--encoder-plan-hash {'ab' * 32}" in launch
        assert "--encoder-revision abc" in worker and "--encoder-revision abc" in launch
    if kind == "off":
        assert "cuteafd plan" not in result.stderr
    else:
        preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
        assert f"--vision {mode or 'auto'}" in preflight


@pytest.mark.parametrize("family_config,serve", [
    ({"model_type": "deepseek_v4"}, "serve-dsv4"),
    ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}, "serve-glm"),
    ({"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
      "layer_types": ["linear_attention", "deepseek_sparse_attention"]}, "serve-glmf"),
    ({"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
      "layer_types": ["linear_attention", "full_attention"]}}, "serve-qwen4"),
])
def test_other_generic_families_keep_vision_off_by_default(tmp_path, family_config, serve):
    model = "zai-org/GLM-5.3-Flash" if serve == "serve-glmf" else "test/model"
    result = _family_launch_result(tmp_path, family_config, model, "SPECULATOR=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if f"cuteafd {serve}" in line)
    assert "--vision off" in launch
    assert "--encoder-listen" not in result.stderr and "--vision-peers" not in launch


def test_text_only_mimo_auto_default_does_not_start_a_tower(tmp_path):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    result = _family_launch_result(tmp_path, config, "test/mimo", "SPECULATOR=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert "--vision auto" in launch
    assert "cuteafd plan" not in result.stderr and "--encoder-listen" not in result.stderr


@pytest.mark.parametrize("mode,kind", [("auto", "spark"), ("spark:0", "spark"),
                                        ("rtx:0", "rtx"), ("off", "off"), (None, "off")])
def test_qwen_encoder_explicit_placement_and_default_off(tmp_path, mode, kind):
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
        assert "--encoder-revision abc" in worker and "--encoder-revision abc" in launch
    if kind == "off":
        assert "cuteafd plan" not in result.stderr
