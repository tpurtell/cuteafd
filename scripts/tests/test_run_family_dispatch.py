"""run.sh hands every family but DeepSeek V4.1 to scripts/launch/run-family.sh."""
from __future__ import annotations

import json
import os
import re
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
    if config.get("audio_config"):
        audio = root / "snapshots" / "abc" / "audio_tokenizer"
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
    (repo / "glmf.config").write_text("MODEL_ID=zai-org/GLM-5.3-Flash\nDRAFT_MODEL_ID=incoai/x\n")
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--restart")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"run-family --config {repo / 'glmf.config'} --family glm5_flash --restart\n"
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--restart", "--all")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"run-family --config {repo / 'glmf.config'} --family glm5_flash --restart --all\n"
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--all")
    assert result.returncode != 0 and "--all requires --restart" in result.stderr
    # DeepSeek V4.1 options do not apply to other families.
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--concurrency", "4")
    assert result.returncode != 0 and "take --config, --restart, --wip and --embedding-placement" in result.stderr
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--embedding-placement", "host")
    assert result.returncode == 0, result.stderr
    assert "--embedding-placement host" in result.stdout
    for backend in ("uring", "mincore-routed"):
        result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--table-backend", backend)
        assert result.returncode == 0, result.stderr
        assert f"--table-backend {backend}" in result.stdout
    # A ./wip.sh slot reaches run-family.sh, which serves it from the development images.
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--wip", "s1", "--restart")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"run-family --config {repo / 'glmf.config'} --family glm5_flash --restart --wip s1\n"


@pytest.mark.parametrize("value", [None, "on", "off"])
def test_shared_draft_policy_switches_are_explicit_and_default_off(tmp_path: Path, value) -> None:
    names = ["DRAFT_COST_BUCKETS", "DRAFT_CONFIDENCE", "COPY_DRAFT_POLICY"]
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


def test_shared_draft_policy_rejects_unknown_switch_value(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, {"model_type": "mimo_v2_flash", "num_hidden_layers": 2,
                                            "moe_layer_freq": [0, 1]}, "XiaomiMiMo/MiMo-V2-Flash",
                                  "DRAFT_CONFIDENCE=maybe\n")
    assert result.returncode != 0
    assert "DRAFT_CONFIDENCE must be on or off" in result.stderr


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


@pytest.mark.parametrize("enabled", [False, True])
def test_glmf_solved_worker_startup_waits_bootstrap_only_when_opted_in(tmp_path: Path, enabled: bool) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "SPECULATOR=off\nVISION=off\nAUDIO=off\nGLM5_FLASH_FP8_MODEL_ID=off\n",
                                  extra_env={"CUTEAFD_PLACEMENT_HANDSHAKE": str(int(enabled)),
                                             "STUB_WORKER_READY": "worker bootstrap ready" if enabled else "worker ready"})
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    worker = next(line for line in lines if "cuteafd expertd-native" in line)
    coordinator = next(line for line in lines if "cuteafd" in line and "serve-glmf" in line)
    assert ("--placement-handshake" in worker) == enabled
    assert ("--placement-handshake" in coordinator) == enabled
    marker = "worker bootstrap ready" if enabled else "worker ready"
    readiness = next(i for i, line in enumerate(lines) if f"grep -q '{marker}'" in line)
    assert readiness < lines.index(coordinator)
    if enabled:
        assert not any("grep -q 'worker ready'" in line for line in lines)


def test_glmf_zero_spark_launch_does_not_enable_worker_selection(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "SPARK_COUNT=0\nSPECULATOR=off\nVISION=off\nAUDIO=off\nGLM5_FLASH_FP8_MODEL_ID=off\n",
                                  extra_env={"CUTEAFD_PLACEMENT_HANDSHAKE": "1"})
    assert result.returncode == 0, result.stderr
    assert "--placement-handshake" not in result.stderr
    assert "cuteafd expertd-native" not in result.stderr


def test_glmf_selection_is_sealed_after_solve_before_loading_or_opening_lanes() -> None:
    source = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5_flash/mod.rs").read_text()
    body = source[source.index("    pub fn with_engine<"):source.index("    /// The drafter `--draft`")]
    assert body.index("self.admit(") < body.index("admitted_spark_layers(") < body.index("select_worker_layers(")
    assert body.index("select_worker_layers(") < body.index("TokenEmbedding::load(") < body.index("self.experts_range(")
    assert "args.experts_snapshot.as_deref().unwrap_or(&args.snapshot)" in body
    assert "WorkerSelection::new(identity, &spark_layers)" in body
    assert "let spark = !spark_layers.is_empty();" in body
    assert "selected_args.prefill_lanes = usize::try_from(working.prefill_lanes)?;" in body
    assert body.index("working.prefill_lanes") < body.index("TokenEmbedding::load(")
    assert "spark_layers == working.spark_layers" in body
    admission = source[source.index("    fn admit("):source.index("    pub fn with_engine<")]
    assert "solve_working_set_with_graphs(" in admission
    assert "let local_workspace = workspaces(false, 1);" in admission
    expert = source[source.index("    fn experts_range<"):source.index("impl Opened {\n    /// The checkpoint's `embed_tokens`")]
    assert expert.index("if remote.is_empty() { return Ok(None); }") < expert.index("SparkLink::new(")
    assert "spark_warmup_request(&self.cfg, warm_rows, remote.start)" in expert


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
                          previous_peers: str | None = None, encoder_plan: dict | None = None,
                          extra_args: tuple[str, ...] = (), with_nest: bool = True,
                          extra_env: dict[str, str] | None = None,
                          program_manifest: dict | None = None) -> subprocess.CompletedProcess[str]:
    """Run the real run-family.sh up to its docker calls (docker/ssh/nest/curl are stubs
    that print their argv) and return what it would launch."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / "scripts" / "launch").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "launch" / "run-family.sh", repo / "scripts" / "launch")
    shutil.copy(ROOT / "scripts" / "launch" / "preflight-fp8-bf16.py", repo / "scripts" / "launch")
    shutil.copy(ROOT / "scripts" / "lib" / "checkpoint-family.py", repo / "scripts" / "lib")
    shutil.copy(ROOT / "scripts" / "lib" / "release-common.sh", repo / "scripts" / "lib")
    wip = "--wip" in extra_args
    if wip:
        (repo / "scripts" / "build").mkdir()
        (repo / "scripts" / "build" / "verify-sparkinfer-source.py").write_text("print('test-pin')\n")
    hf = tmp_path / "hf"
    _snapshot(hf, model, family_config)
    if family_config.get("model_type") == "glm_moe_dsa":
        _snapshot(hf, "incoai/GLM-5.3-DFlash2", {"architectures": ["DFlash2DraftModel"]})
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("docker", "ssh", "nest") if with_nest else ("docker", "ssh"):
        (bin_dir / tool).write_text('#!/usr/bin/env bash\nprintf "%s " "$(basename "$0")" "$@" >&2; echo >&2\n'
                                    + ('case "$*" in *"docker run --rm"*"python3"*) exit 2 ;; esac\n'
                                       if preflight_error and tool == "ssh" else '') +
                                    'case "$*" in *"docker logs"*) echo "${STUB_WORKER_READY:-worker ready}"; '
                                    'echo "audio encoder ready backend=mimo_audio_fp32_v1/cuda13000/cufft12000/cublas13.0.0/cute_aot_sm121/export' + 'cd' * 32 + '" ;; esac\n' +
                                    ('case "$*" in image\\ inspect*) echo test-pin ;; '
                                     'cp\\ *) dst="${@: -1}"; mkdir -p "$dst"; '
                                     'touch "$dst/cuteafd" "$dst/libcuteafd_native.so" "$dst/release-entrypoint.sh"; '
                                     'if [[ "$2" == *verify-sparkinfer-source.py ]]; then printf "print(42)\\n" > "$dst/verify-sparkinfer-source.py"; fi ;; esac\n'
                                     if tool == "docker" and wip else '') +
                                    (f"case \"$*\" in *\"cuteafd plan\"*) echo '{{\"spark_ranks\":{preferred_ranks}}}' ;; esac\n"
                                     if tool == "docker" and preferred_ranks is not None and encoder_plan is None else '') +
                                    (f"case \"$*\" in *\"cuteafd plan\"*) printf '%s\\n' '{json.dumps(encoder_plan)}' ;; esac\n"
                                     if tool == "docker" and encoder_plan is not None else '') +
                                    ("case \"$*\" in top*) printf '%s\\n' PID " +
                                     " ".join(map(str, container_pids)) + " ;; esac\n"
                                     if tool == "docker" and container_pids else '') +
                                    (f"case \"$*\" in inspect*) echo '[\"serve-qwen4\",\"--peers\",\"{previous_peers}\"]' ;; esac\n"
                                     if tool == "docker" and previous_peers is not None else ''))
        if tool == "docker":
            stub = (bin_dir / tool).read_text()
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
                                         " ;; *) printf '%s\\n' " + " ".join(map(str, physical_gpus)) +
                                         ' ;; esac\n' if preferred_ranks is None else
                                         '#!/usr/bin/env bash\ncase "$*" in *memory.free*) echo ' + str(gpu_free_mib) +
                                         ' ;; *memory.total*) echo ' + str(gpu_total_mib) +
                                         ' ;; *query-compute-apps*) printf \'%s\\n\' ' +
                                         " ".join(f"'{pid}, {mib}'" for pid, mib in gpu_allocations) +
                                         ' ;; *) echo 0; echo 1 ;; esac\n')
    (bin_dir / "nvidia-smi").chmod(0o755)
    config = repo / "f.config"
    config.write_text(f"MODEL_ID={model}\nSPARK_COUNT=1\nSPARK_0_HOST=h0\nSPARK_0_LANE_A=10.0.0.1\n{keys}")
    env = {**os.environ, "HF_HOME": str(hf), "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "STUB_PROGRAMS": str(tmp_path / "PROGRAMS.json"), **(extra_env or {})}
    env["HOME"] = str(tmp_path / "home")
    if not with_nest:
        # Hide any nest the host has, keeping only the stub directory and the system tools.
        env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
    return subprocess.run(["bash", str(repo / "scripts" / "launch" / "run-family.sh"), "--config", str(config),
                           *(["--restart"] if restart else []), *extra_args],
                          env=env, capture_output=True, text=True, timeout=30)


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
    assert len(plans) == (2 if qwen else 1)
    staging = next(i for i, line in enumerate(lines) if "docker cp" in line and ".cuteafd-wip" in line)
    for preflight in plans:
        assert staging < lines.index(preflight)
        assert "cuteafd-coordinator-dev cuteafd plan" in preflight
        for part in ("bin", "lib", "share"):
            assert f"wip-run/plan-test/slot-test/{part}:/opt/cuteafd/{part}:ro" in preflight
        assert "--entrypoint /opt/cuteafd/share/release-entrypoint.sh" in preflight


@pytest.mark.parametrize("budget,total,embedding", [("", 32768, "host"), ("31.8", 98304, "host"), ("", 98304, "gpu")])
def test_mimo_full_context_profile_respects_physical_and_logical_memory(tmp_path, budget, total, embedding):
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]}
    keys = f"COORDINATOR_GPU_BUDGET_GIB={budget}\nEXPERT_BACKEND=spark\nSPECULATOR=off\nVISION=off\nAUDIO=off\n"
    result = _family_launch_result(tmp_path, config, "test/mimo", keys, gpu_total_mib=total)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "--max-context 0" in launch
    assert f"--max-sequences {16 if embedding == 'host' else 8}" in launch
    assert f"--embedding-placement {embedding}" in launch
    assert "--pool-tokens 0" in launch


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


@pytest.mark.parametrize("value", [None, "auto", "0", "5", "50%", "all", "max"])
def test_deepseek_v4_honors_explicit_local_expert_limit(tmp_path, value):
    keys = "" if value is None else f"RTX_EXPERT_LAYERS={value}\n"
    result = _family_launch_result(tmp_path, {"model_type": "deepseek_v4"}, "test/dsv4", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-dsv4" in line)
    if value is None:
        assert "--local-expert-layers" not in launch and "--rtx-expert-layers" not in launch
    elif value.isdigit():
        # Whole layer counts keep the flag older images accept.
        assert f"--local-expert-layers {value}" in launch
    else:
        assert f"--rtx-expert-layers {value}" in launch


@pytest.mark.parametrize("value", [None, "auto", "0", "5", "50%", "all", "max"])
def test_glm_flash_forwards_explicit_rtx_expert_onboarding(tmp_path, value):
    keys = "SPECULATOR=off\nGLM5_FLASH_FP8_MODEL_ID=off\n"
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
        assert f"--rtx-expert-layers {value}" in launch


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
    ("GLM5_FLASH_INDEX_CACHE", "tails", "GLM5_FLASH_INDEX_CACHE must be"),
])
def test_glmf_invalid_fp8_options_fail_before_workers_launch(tmp_path, key, value, message):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf",
                                  f"GLM5_FLASH_FP8_MODEL_ID=off\nGLM5_FLASH_KDA_FP8=off\n{key}={value}\n")
    assert result.returncode == 2 and message in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


GLMF_TWO_LAYER = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
                  "layer_types": ["linear_attention", "deepseek_sparse_attention"]}


@pytest.mark.parametrize("keys,tensor", [("", False), ("GLM5_FLASH_DRAFT_HEAD=exact\n", False),
                                        ("GLM5_FLASH_DRAFT_HEAD=tensor\n", True)])
def test_glmf_draft_head_is_forwarded_only_when_tensor(tmp_path, keys, tensor):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nRTX_GPUS=1\n" + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert ("--draft-head tensor" in launch) == tensor, launch
    assert ("--draft-head" in launch) == tensor, launch


def test_glmf_draft_head_rejects_unknown_values_before_launch(tmp_path):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nGLM5_FLASH_DRAFT_HEAD=fp8\n")
    assert result.returncode == 2 and "GLM5_FLASH_DRAFT_HEAD must be exact or tensor" in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,forwarded", [("", None), ("GLM5_FLASH_DRAFT_LINEAR=w8a16\n", None),
                                           ("GLM5_FLASH_DRAFT_LINEAR=wide\n", "wide"),
                                           ("GLM5_FLASH_DRAFT_LINEAR=w8a8\n", "w8a8")])
def test_glmf_draft_linear_is_forwarded_only_past_w8a16(tmp_path, keys, forwarded):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nRTX_GPUS=1\n" + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    if forwarded is None:
        assert "--draft-linear" not in launch, launch
    else:
        assert f"--draft-linear {forwarded}" in launch and launch.count("--draft-linear") == 1, launch


def test_glmf_draft_linear_rejects_unknown_values_before_launch(tmp_path):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nGLM5_FLASH_DRAFT_LINEAR=w4a16\n")
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
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
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
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,compact", [("", False), ("GLM5_FLASH_INDEX_CACHE=keys\n", False),
                                         ("GLM5_FLASH_INDEX_CACHE=compact\n", True)])
def test_glmf_index_cache_is_forwarded_only_when_compact(tmp_path, keys, compact):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert ("--index-cache compact" in launch) == compact, launch
    assert "--index-cache keys" not in launch


@pytest.mark.parametrize("keys,expected,absent", [
    ("", [], ["--prefill-lanes", "--prefill-lane-rows", "--headroom-gib", "--graph-budget-mib"]),
    ("GLM5_FLASH_PREFILL_LANES=4\nGLM5_FLASH_PREFILL_LANE_ROWS=2048\n",
     ["--prefill-lanes 4", "--prefill-lane-rows 2048"], []),
    ("GLM5_FLASH_PREFILL_LANES=1\n", ["--prefill-lanes 1"], ["--prefill-lane-rows"]),
    ("GLM5_FLASH_HEADROOM_GIB=1\n", ["--headroom-gib 1"], ["--prefill-lanes"]),
    ("GLM5_FLASH_GRAPH_BUDGET_MIB=512\n", ["--graph-budget-mib 512"], ["--headroom-gib"]),
])
def test_glmf_lanes_and_headroom_are_forwarded_only_when_set(tmp_path, keys, expected, absent):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    for option in expected:
        assert option in launch, launch
    for option in absent:
        assert option not in launch, launch


@pytest.mark.parametrize("key,value,message", [
    ("GLM5_FLASH_PREFILL_LANES", "0", "GLM5_FLASH_PREFILL_LANES must be 1 to 4"),
    ("GLM5_FLASH_PREFILL_LANES", "5", "GLM5_FLASH_PREFILL_LANES must be 1 to 4"),
    ("GLM5_FLASH_PREFILL_LANE_ROWS", "2000", "GLM5_FLASH_PREFILL_LANE_ROWS must be a multiple of 64"),
    ("GLM5_FLASH_PREFILL_LANE_ROWS", "8192", "GLM5_FLASH_PREFILL_LANE_ROWS must be a multiple of 64"),
    ("GLM5_FLASH_PREFILL_LANE_ROWS", "0", "GLM5_FLASH_PREFILL_LANE_ROWS must be a multiple of 64"),
    ("GLM5_FLASH_HEADROOM_GIB", "-1", "GLM5_FLASH_HEADROOM_GIB must be a non-negative size"),
    ("GLM5_FLASH_HEADROOM_GIB", "1GiB", "GLM5_FLASH_HEADROOM_GIB must be a non-negative size"),
    ("GLM5_FLASH_GRAPH_BUDGET_MIB", "0", "GLM5_FLASH_GRAPH_BUDGET_MIB must be a positive whole number"),
])
def test_glmf_invalid_lanes_or_headroom_fail_before_workers_launch(tmp_path, key, value, message):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2,
              "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{key}={value}\n")
    assert result.returncode == 2 and message in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


GLMF_CONFIG = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
               "layer_types": ["linear_attention", "deepseek_sparse_attention"]}


def test_glmf_default_launch_passes_no_admission_or_verify_option(tmp_path):
    """Packed admission prefill and the chain verify policy are opt-in: a launch without their
    keys passes none of their options (one prefill pass per prompt, the cost verify policy)."""
    result = _family_launch_result(tmp_path, GLMF_CONFIG, "test/glmf", "GLM5_FLASH_FP8_MODEL_ID=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    for option in ("--prefill-batch", "--verify-policy", "--spec-tau"):
        assert option not in launch, (option, launch)


@pytest.mark.parametrize("keys,expected,absent", [
    ("GLM5_FLASH_PREFILL_BATCH=off\n", (), ("--prefill-batch",)),
    ("GLM5_FLASH_PREFILL_BATCH=on\n", ("--prefill-batch",), ()),
    ("GLM5_FLASH_VERIFY_POLICY=cost\n", (), ("--verify-policy",)),
    ("GLM5_FLASH_VERIFY_POLICY=chain\n", ("--verify-policy chain",), ("--spec-tau",)),
    ("GLM5_FLASH_VERIFY_POLICY=chain\nGLM5_FLASH_SPEC_TAU=0.5\n", ("--verify-policy chain", "--spec-tau 0.5"), ()),
    ("GLM5_FLASH_SPEC_TAU=1\n", ("--spec-tau 1",), ("--verify-policy",)),
    ("GLM5_FLASH_PREFILL_BATCH=on\nGLM5_FLASH_VERIFY_POLICY=chain\n", ("--prefill-batch", "--verify-policy chain"), ()),
])
def test_glmf_admission_and_verify_keys_are_forwarded(tmp_path, keys, expected, absent):
    result = _family_launch_result(tmp_path, GLMF_CONFIG, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    for option in expected:
        assert option in launch, (option, launch)
    for option in absent:
        assert option not in launch, (option, launch)


@pytest.mark.parametrize("keys,message", [
    ("GLM5_FLASH_PREFILL_BATCH=yes\n", "GLM5_FLASH_PREFILL_BATCH must be"),
    ("GLM5_FLASH_VERIFY_POLICY=greedy\n", "GLM5_FLASH_VERIFY_POLICY must be"),
    ("GLM5_FLASH_SPEC_TAU=0\n", "GLM5_FLASH_SPEC_TAU must be"),
    ("GLM5_FLASH_SPEC_TAU=0.0\n", "GLM5_FLASH_SPEC_TAU must be"),
    ("GLM5_FLASH_SPEC_TAU=1.5\n", "GLM5_FLASH_SPEC_TAU must be"),
    ("GLM5_FLASH_SPEC_TAU=abc\n", "GLM5_FLASH_SPEC_TAU must be"),
])
def test_glmf_admission_and_verify_keys_reject_bad_values_before_launch(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, GLMF_CONFIG, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,expected,absent", [
    ("", ("CUTEAFD_GLMF_DRAFT_POLICY=cycle",), ("CUTEAFD_GLMF_ROUTE_RING_CHECK",)),
    ("GLM5_FLASH_DRAFT_POLICY=shared\n", ("CUTEAFD_GLMF_DRAFT_POLICY=shared",), ("CUTEAFD_GLMF_ROUTE_RING_CHECK",)),
    ("GLM5_FLASH_DRAFT_POLICY=shared\nGLM5_FLASH_ROUTE_RING_CHECK=on\n",
     ("CUTEAFD_GLMF_DRAFT_POLICY=shared", "CUTEAFD_GLMF_ROUTE_RING_CHECK=1"), ()),
])
def test_glmf_draft_policy_keys_reach_the_coordinator(tmp_path, keys, expected, absent):
    """The shared draft policy is opt-in (cycle by default) and the route ring check is diagnostic."""
    result = _family_launch_result(tmp_path, GLMF_CONFIG, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    for option in expected:
        assert option in launch, (option, launch)
    for option in absent:
        assert option not in launch, (option, launch)


@pytest.mark.parametrize("keys,message", [
    ("GLM5_FLASH_DRAFT_POLICY=buckets\n", "GLM5_FLASH_DRAFT_POLICY must be"),
    ("GLM5_FLASH_ROUTE_RING_CHECK=yes\n", "GLM5_FLASH_ROUTE_RING_CHECK must be"),
])
def test_glmf_draft_policy_keys_reject_bad_values_before_launch(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, GLMF_CONFIG, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,forwarded", [("", None), ("GLM5_FLASH_KDA_STATE=f32\n", None),
                                             ("GLM5_FLASH_KDA_STATE=bf16\n", "bf16"),
                                             ("GLM5_FLASH_KDA_STATE=bf16-tile\n", "bf16-tile")])
def test_glmf_kda_state_is_forwarded_with_bf16_kda_projections(tmp_path, keys, forwarded):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nRTX_GPUS=1\nGLM5_FLASH_KDA_FP8=off\n" + keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    if forwarded is None:
        assert "--kda-state" not in launch
    else:
        assert f"--kda-state {forwarded}" in launch and launch.count("--kda-state") == 1


@pytest.mark.parametrize("keys,message", [
    ("RTX_GPUS=1\nGLM5_FLASH_KDA_STATE=bf16\n", "GLM5_FLASH_KDA_FP8=off"),
    ("RTX_GPUS=2\nGLM5_FLASH_KDA_FP8=off\nGLM5_FLASH_KDA_STATE=bf16\n", "without a head split"),
    ("RTX_GPUS=1\nGLM5_FLASH_KDA_FP8=off\nGLM5_FLASH_KDA_STATE=fp16\n", "GLM5_FLASH_KDA_STATE must be"),
])
def test_glmf_kda_state_rejects_unsupported_layouts_before_launch(tmp_path, keys, message):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\n" + keys)
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
    result = _family_launch_result(tmp_path, config, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
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
    result = _family_launch_result(tmp_path, config, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,expected", [("", None), ("GLM5_FLASH_PREFIX_MARKS=arena\n", "arena"),
                                           ("GLM5_FLASH_PREFIX_MARKS=pool\n", "pool")])
def test_glmf_prefix_marks_are_forwarded(tmp_path, keys, expected):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
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
    ("GLM5_FLASH_PREFIX_MARKS=arena\n", None),
    ("", None),
])
def test_glmf_pool_marks_turn_the_host_tier_on(tmp_path, keys, expected):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf", f"GLM5_FLASH_FP8_MODEL_ID=off\n{keys}")
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
    planner = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line and "--layout" in line)
    for command in [launch, planner]:
        if expected is None:
            assert "--prefix-marks" not in command, command
        else:
            assert command.count("--prefix-marks") == 1 and f"--prefix-marks {expected}" in command, command


def test_glmf_prefix_marks_reject_unknown_stores_before_launch(tmp_path):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nGLM5_FLASH_PREFIX_MARKS=host\n")
    assert result.returncode == 2 and "GLM5_FLASH_PREFIX_MARKS must be arena or pool" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,shared", [("", True), ("GLM5_FLASH_REPLAY_RECORDS=own\n", False),
                                        ("GLM5_FLASH_REPLAY_RECORDS=shared\n", True)])
def test_glmf_replay_records_are_forwarded_only_when_shared(tmp_path, keys, shared):
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nRTX_GPUS=1\n" + keys)
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
                                  "GLM5_FLASH_FP8_MODEL_ID=off\n" + keys)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("keys,forwarded", [("", False), ("GLM5_FLASH_DECODE_ROWS=64\n", False),
                                             ("GLM5_FLASH_DECODE_ROWS=128\n", True)])
def test_glmf_decode_rows_are_forwarded_only_at_128(tmp_path, keys, forwarded):
    """GLM5_FLASH_DECODE_ROWS=128 reaches the coordinator as --decode-rows 128 (the wide decode
    programs); 64, the default, passes nothing; no Spark worker sees it."""
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nRTX_GPUS=1\n" + keys)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    launch = next(line for line in lines if "cuteafd serve-glmf" in line)
    assert ("--decode-rows 128" in launch) == forwarded and launch.count("--decode-rows") == int(forwarded)
    assert not any("--decode-rows" in line for line in lines if "cuteafd expertd-native" in line)


def test_glmf_wide_decode_rows_take_shared_replay_records(tmp_path):
    """128-row steps keep their records in the prefill scratch too: both keys reach the coordinator."""
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nRTX_GPUS=1\nGLM5_FLASH_DECODE_ROWS=128\n"
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
                                  "GLM5_FLASH_FP8_MODEL_ID=off\n" + keys)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


def test_probe_dump_root_is_mounted_for_remote_row_dumps(tmp_path):
    root = tmp_path / "dumps"
    root.mkdir()
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  f"GLM5_FLASH_FP8_MODEL_ID=off\nRTX_GPUS=1\nPROBE_DUMP_ROOT={root}\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert f"-v {root}:{root} -e CUTEAFD_PROBE_DUMP_ROOT={root}" in launch
    result = _family_launch_result(tmp_path / "unset", SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nRTX_GPUS=1\n")
    assert result.returncode == 0 and "CUTEAFD_PROBE_DUMP_ROOT" not in result.stderr
    result = _family_launch_result(tmp_path / "missing", SPLIT_CONFIGS["glm5_flash"], "test/glmf",
                                  f"GLM5_FLASH_FP8_MODEL_ID=off\nPROBE_DUMP_ROOT={tmp_path}/absent\n")
    assert result.returncode == 2 and "PROBE_DUMP_ROOT must be" in result.stderr
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
    planner = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line and "--layout" in line)
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
        expected = "/root/.cache/huggingface/hub/models--test--model/snapshots/abc"
    elif source_kind == "external_snapshot":
        source = str(tmp_path / "vendor")
        Path(source).mkdir()
        expected = source
    result = _family_launch_result(tmp_path, config, "test/model",
                                  f"SPECULATOR=off\nGLM5_FLASH_FP8_MODEL_ID=off\nVISION=auto\nCHAT_TEMPLATE_FROM={source}\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-glmf" in line)
    assert f"--chat-template-from {expected}" in launch
    if source_kind == "external_snapshot":
        assert f"-v {source}:{source}:ro" in launch
    worker = next(line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line)
    assert "--chat-template-from" not in worker and source not in worker


def test_text_only_glm_ignores_template_override(tmp_path):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2, "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/model",
                                  "SPECULATOR=off\nGLM5_FLASH_FP8_MODEL_ID=off\nVISION=off\nCHAT_TEMPLATE_FROM=missing/vendor\n")
    assert result.returncode == 0, result.stderr
    assert "--chat-template-from" not in result.stderr and "missing/vendor" not in result.stderr


@pytest.mark.parametrize("source", ["../vendor", "vendor/../flash", "vendor/..", "missing/vendor"])
def test_glm_template_override_invalid_source_refused_before_workers(tmp_path, source):
    config = {"model_type": "glm5_next", "num_hidden_layers": 2,
              "mlp_layer_types": ["sparse"] * 2, "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    result = _family_launch_result(tmp_path, config, "test/model",
                                  f"SPECULATOR=off\nGLM5_FLASH_FP8_MODEL_ID=off\nVISION=auto\nCHAT_TEMPLATE_FROM={source}\n")
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
    result = _family_launch_result(tmp_path, family_config, "test/model",
                                  "GLM5_FLASH_FP8_MODEL_ID=off\nSPECULATOR=off\n" + setting)
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
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
    assert "--device 1 --split-device 0" in launch and "device=0,1" in launch


def test_auto_uses_one_gpu_when_only_one_exists(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["mimo_pro"], "test/model", "", physical_gpus=(0,))
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
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


@pytest.mark.parametrize("host, expected", [("", "auto"), ("HOST_CACHE_BYTES=0\n", None),
                                           ("HOST_CACHE_BYTES=16GiB\n", "16GiB")])
def test_mimo_port_flags_preserve_explicit_host_budget(tmp_path: Path, host: str, expected: str | None) -> None:
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 3, "moe_layer_freq": [0, 1, 1]}
    keys = ("MIMO_HOST_CACHE=on\nMIMO_PREFIX_DRAFT=on\nMIMO_COPY_WINDOWS=on\n"
            "MIMO_SNAPSHOT_WAIT=on\nMIMO_PREFILL_CHUNK_S=4\nHTTP_QUEUE_DEPTH=32\nHTTP_QUEUE_WAIT_MS=1234\n") + host
    result = _family_launch_result(tmp_path, config, "XiaomiMiMo/MiMo-V2-Flash", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    for flag in ("--mimo-host-cache", "--mimo-prefix-draft", "--mimo-copy-windows",
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
    assert "--host-cache-bytes" not in launch[0] and "--pool-tokens 0" in launch[0]


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
    assert "--host-cache-bytes" not in launch[0] and "--pool-tokens 0" in launch[0]


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
    launch = next(line for line in result.stderr.splitlines() if "--coordinator-gpu-budget-gib" in line and "serve-" in line)
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
    assert "--coordinator-gpu-budget-gib 26.0" in preflight and "--coordinator-weight-budget-gib 26.0" in preflight
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
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
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
    assert f"--coordinator-gpu-budget-gib {admitted_gib}" in preflight


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
    launch = next(line for line in result.stderr.splitlines() if "serve-mimo --snapshot" in line)
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
    effective = concurrency if concurrency is not None else (16 if gpu_total_mib <= 32768 else 8)
    assert f"--concurrency {effective}" in preflight
    assert f"--max-sequences {effective}" in launch
    assert "unbound variable" not in result.stderr
    if warm:
        assert "--context-tokens 131072" in preflight


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
    assert ("cuteafd plan" in result.stderr) == (kind != "off")
    if kind != "off":
        preflight = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
        assert f"--vision {mode or 'auto'}" in preflight


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
    planner = next(line for line in lines if "cuteafd plan" in line and "--layout" in line)
    launch = next(line for line in lines if "cuteafd serve-glmf" in line)
    assert "--vision rtx" in planner and "--pool-tokens 262144" in planner, planner
    for command in [planner, launch]:
        assert ("--decode-rows 128" in command) == forwarded and command.count("--decode-rows") == int(forwarded), command


GLMF_VISION = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
               "layer_types": ["linear_attention", "deepseek_sparse_attention"], "vision_config": {"depth": 24}}
GLMF_RTX_ENCODER_PLAN = {"placement_supported": True, "fits": True, "spark_ranks": 1, "encoder_plan_hash": "ab" * 32,
                         "encoder": {"kind": {"kind": "rtx", "gpu": 0}, "replicas": []}}


@pytest.mark.parametrize("keys,serve,plan", [
    # Unset: serve-glmf takes the launcher's defaults, the plan the planner's (the same), and the
    # plan's arguments are work/p0's.
    ("", {"--max-sequences": "8", "--prefix-cache-entries": "20"}, {}),
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
    shared replay records. Unset keys reach neither command, so the plan's arguments are unchanged."""
    result = _family_launch_result(tmp_path, GLMF_VISION, "zai-org/GLM-5.3-Flash",
                                   "RTX_GPUS=1\nSPECULATOR=off\nVISION=rtx\nGLM5_FLASH_REPLAY_RECORDS=own\n" + keys,
                                   encoder_plan=GLMF_RTX_ENCODER_PLAN)
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    commands = {"plan": next(line for line in lines if "cuteafd plan" in line and "--layout" in line) + " ",
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
    assert "--host-cache-bytes auto" not in serve


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
    assert "--draft-head tensor" in serve
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
    assert "cuteafd plan" not in result.stderr
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
    assert "cuteafd plan" not in result.stderr and "--encoder-listen" not in result.stderr


_GLMF = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
         "layer_types": ["linear_attention", "deepseek_sparse_attention"]}


@pytest.mark.parametrize("slot", ["../x", "-s", "a b", ""])
def test_invalid_wip_slot_fails_before_any_container(tmp_path, slot):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "GLM5_FLASH_FP8_MODEL_ID=off\n",
                                   extra_args=("--wip", slot))
    assert result.returncode != 0
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


def test_restart_removes_only_own_worker_port(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf",
                                   "GLM5_FLASH_FP8_MODEL_ID=off\nEXPERT_PORT=19555\nINSTANCE=own\n", restart=True)
    assert result.returncode == 0, result.stderr
    assert "docker rm -f cuteafd-spark-expert-h0-19555" in result.stderr
    assert "docker ps -aq --filter" not in result.stderr
    assert "docker rm -f cuteafd-spark-expert-h0-19441" not in result.stderr
    assert "docker rm -f cuteafd-spark-expert-wip" not in result.stderr


def test_restart_all_sweeps_workers_and_keeps_the_wip_container(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "GLM5_FLASH_FP8_MODEL_ID=off\n",
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
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "GLM5_FLASH_FP8_MODEL_ID=off\n",
                                   extra_env={"CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP": device_map})
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    worker = next(line for line in lines if "docker run -d --name cuteafd-spark-expert-" in line)
    coordinator = next(line for line in lines if "cuteafd serve-glmf" in line)
    assert f"-e CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP={device_map}" in worker, worker
    assert f"-e CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP={device_map}" in coordinator, coordinator


def test_rdma_device_map_is_not_set_unless_given(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "GLM5_FLASH_FP8_MODEL_ID=off\n",
                                   extra_env={"CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP": ""})
    assert result.returncode == 0, result.stderr
    assert "CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP" not in result.stderr


def test_invalid_rdma_device_map_fails_before_workers_launch(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "GLM5_FLASH_FP8_MODEL_ID=off\n",
                                   extra_env={"CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP": "10.0.0.1=dev x"})
    assert result.returncode == 2
    assert "local-ip=device" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


def test_no_nest_launch_needs_no_sudo_page_cache_drop(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "GLM5_FLASH_FP8_MODEL_ID=off\n", with_nest=False)
    assert result.returncode == 0, result.stderr
    assert "drop_caches" not in result.stderr
    assert "fadvise(DONTNEED)" in result.stdout


def test_spark_page_caches_drop_over_ssh_without_nest_when_opted_in(tmp_path):
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "GLM5_FLASH_FP8_MODEL_ID=off\n", with_nest=False,
                                   extra_env={"CUTEAFD_GLOBAL_PAGE_CACHE_DROP": "1"})
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    drops = [i for i, line in enumerate(lines) if line.startswith("ssh -n h0 ") and "drop_caches" in line]
    assert len(drops) == 2, result.stderr  # before the workers start and once they are resident
    worker = next(i for i, line in enumerate(lines) if "docker run -d --name cuteafd-spark-expert-" in line)
    assert drops[0] < worker < drops[1]
    assert "could not drop" not in result.stderr


@pytest.mark.parametrize("keys,mode", [("", None), ("RDMA_BOND_BALANCE=off\n", None),
                                      ("RDMA_BOND_BALANCE=labels\n", "labels"),
                                      ("RDMA_BOND_BALANCE=probe\n", "probe")])
def test_rdma_bond_balance_reaches_only_the_coordinator(tmp_path, keys, mode):
    """The flow-label switch is the coordinator's: forwarded only when not off. Workers get
    nothing; they connect with whatever label each coordinator connection asks for."""
    result = _family_launch_result(tmp_path, _GLMF, "test/glmf", "GLM5_FLASH_FP8_MODEL_ID=off\n" + keys)
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
                                   "GLM5_FLASH_FP8_MODEL_ID=off\nRDMA_BOND_BALANCE=yes\n")
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
        assert "cuteafd plan" not in result.stderr


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
    keys = "SPECULATOR=off\nVISION=off\nAUDIO=off\nGLM5_FLASH_FP8_MODEL_ID=test/model\n"
    if context is not None:
        keys += f"MAX_CONTEXT_TOKENS={context}\n"
    result = _family_launch_result(tmp_path, config, "test/model", keys)
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if " --snapshot " in line and "serve-" in line)
    assert f"--max-context {context or 0}" in launch
