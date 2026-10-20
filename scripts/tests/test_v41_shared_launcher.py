"""DeepSeek V4.1 launches through the shared family launcher (#82).

The goldens in fixtures/v41-launch/ were recorded from run.sh's private V4.1
path (scripts/tests/v41_launch_capture.py, before #82). Today's launcher must
start the same coordinator and Spark containers for every V4.1 example config
and the gate kit's v41-flash-max/min: the same options, mounts, environment,
image and serve command. The only differences allowed are the intended
additions listed in INTENDED and the legacy GPU-request conversion to CDI: shared launch features that now reach V4.1 at
their defaults, with no V4.1 code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v41_launch_capture import CASES, FIXTURES, KIT_CARDS, ROOT, capture, parse_run  # noqa: E402

# Environment the shared launcher adds, by role, at its default values. Each
# comes from a shared helper in scripts/launch/run-family.sh:
INTENDED = {
    "coordinator": {

        # shared_coordinator_env_args (SPARK_INTAKE, CONSOLE_TEXT, ATTENTION_PLACEMENT,
        # FP8_EXPERT_PREFILL): the binary's own defaults, now explicit.
        "CUTEAFD_SPARK_INTAKE": "auto",
        "CUTEAFD_CONSOLE_TEXT": "true",
        "CUTEAFD_ATTENTION_PLACEMENT": "auto",
        "CUTEAFD_FP8_EXPERT_PREFILL": "auto",
        # Admission follows the launcher's bounded readiness deadline.
        "CUTEAFD_SPARK_ADMISSION_TIMEOUT_SECONDS": "30",
        # shared_trace_args: the shared draft policy switches, default off.
        "CUTEAFD_DRAFT_SURVIVAL_LENGTHS": "0",
        # shared_thermal_args: the GPU throttle sampler (#114), default on.
        "CUTEAFD_THERMAL": "on",
        # Dashboard fidelity reads the host's shared reference cache.
        "CUTEAFD_FIDELITY_CACHE": "/run/cuteafd/fidelity",
    },
    "worker": {

        # shared_worker_env_args: FP8 prefill form and the worker's sampler.
        "CUTEAFD_FP8_EXPERT_PREFILL": "auto",
        "CUTEAFD_THERMAL": "on",
    },
}


def flag_values(command: list[str]) -> list[tuple[str, str | None]]:
    """A clap command line as (flag, value) pairs, sorted: flag order is not semantic."""
    pairs, index = [], 0
    while index < len(command):
        token = command[index]
        if token.startswith("--") and index + 1 < len(command) and not command[index + 1].startswith("--"):
            pairs.append((token, command[index + 1]))
            index += 2
        else:
            pairs.append((token, None))
            index += 1
    return sorted(pairs, key=lambda pair: (pair[0], pair[1] or ""))


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("v41-capture")
    os.environ.setdefault("TMPDIR", str(root))
    return root


@pytest.mark.parametrize("case", CASES)
def test_shared_launcher_matches_the_private_v41_path(case: str, tmp_root) -> None:
    golden = json.loads((FIXTURES / f"{case}.json").read_text())
    current = capture(ROOT, case)
    assert [c["host"] for c in current["containers"]] == [c["host"] for c in golden["containers"]]
    for old, new in zip(golden["containers"], current["containers"]):
        before, after = parse_run(old["argv"]), parse_run(new["argv"])
        role = "coordinator" if old["host"] == "raptor" else "worker"
        where = f"{case} {old['host']}"
        expected_options = []
        for option in before["options"]:
            if option.startswith("--gpus "):
                request = option.removeprefix("--gpus ").strip('"')
                names = request.removeprefix("device=").split(",")
                expected_options.extend(f"--device nvidia.com/gpu={name}" for name in names)
            else:
                expected_options.append(option)
        assert after["options"] == sorted(expected_options), where
        # shared_thermal_args also mounts the host power governor's state read-only
        # when gpu-powerd runs on this host (#114).
        governor = ["/run/gpu-powerd:/run/gpu-powerd:ro"] if role == "coordinator" and os.path.isdir("/run/gpu-powerd") else []
        # The coordinator also shares the host fidelity reference cache with the dashboard.
        fidelity = [f"<root>/hosts/{old['host']}/.cache/cuteafd/fidelity:/run/cuteafd/fidelity"] \
            if role == "coordinator" else []
        storage_run = Path("/srv/sparknest/run")
        storage = [f"mount:type=bind,src={storage_run},dst={storage_run},readonly"] \
            if storage_run.is_dir() else []
        assert after["mounts"] == sorted(before["mounts"] + governor + fidelity + storage), where
        assert after["image"] == before["image"], where
        added = {key: value for key, value in after["env"].items() if key not in before["env"]}
        # Locked dashboard pages show how to print the unlock link (never the secret).
        link = added.pop("CUTEAFD_CONSOLE_LINK_COMMAND", None)
        if role == "coordinator":
            assert link and link.startswith('echo "http://') and "$(cat ~/.cache/cuteafd/console/secret)" in link, where
        assert added == INTENDED[role], where
        assert {key: after["env"][key] for key in before["env"]} == before["env"], where
        # The serve command is the same set of flags and values (clap ignores order;
        # --usage now precedes nothing it depends on).
        assert after["command"][:3] == before["command"][:3], where
        expected_command = before["command"] + (["--mapped-scale-preload-gib", "0"] if role == "coordinator" else [])
        assert flag_values(after["command"]) == flag_values(expected_command), where
    # S5 replaces report JSON with the planner-owned deployment. Only the
    # preflight argv changes; all serving commands and fingerprint env above
    # remain byte-identical to the original captures.
    expected_plans = []
    for old in golden["plans"]:
        expected = [value for value in old if value not in ("--json", "--layout")]
        expected.insert(expected.index("--require-ready"), "--deployment")
        if case in KIT_CARDS:
            expected.remove("--require-ready")
        coordinator = parse_run(golden["containers"][0]["argv"])
        command = coordinator["command"]
        additions = []
        for flag in ("--spark-tp", "--spark-ep", "--kv-pool-size", "--memory-reservation"):
            if flag in command:
                additions += [flag, command[command.index(flag) + 1]]
        cache = command[command.index("--host-cache-bytes") + 1] if "--host-cache-bytes" in command else "0"
        expected[expected.index("--prefix-cache-entries") + 2:expected.index("--prefix-cache-entries") + 2] = ["--host-cache-bytes", cache]
        additions += ["--speculator", "dspark" if "--dspark" in command else "off",
                      "--deployment-peers", command[command.index("--peers") + 1]]
        expected[expected.index("--rtx"):expected.index("--rtx")] = additions
        index = expected.index("--available-sparks") + 2
        expected[index:index] = ["--mapped-scale-preload-gib", "0"]
        expected_plans.append(expected)
    assert current["plans"] == expected_plans, case


@pytest.mark.parametrize("cap", ["0", "5", "8"])
def test_scale_preload_cap_reaches_plan_and_native_serve(tmp_root, monkeypatch, cap) -> None:
    import v41_launch_capture as capture_module
    original = capture_module.example_config

    def with_cap(name, directory):
        config = original(name, directory)
        config.write_text(config.read_text() + f"MAPPED_SCALE_PRELOAD_GIB={cap}\n")
        return config

    monkeypatch.setattr(capture_module, "example_config", with_cap)
    result = capture_module.capture(ROOT, "deepseek-v41-mxfp4-max")
    command = parse_run(result["containers"][0]["argv"])["command"]
    for args in [command, *result["plans"]]:
        assert args.count("--mapped-scale-preload-gib") == 1
        assert args[args.index("--mapped-scale-preload-gib") + 1] == cap


def test_goldens_cover_every_v41_example_and_gate_card() -> None:
    examples = {p.stem for p in (ROOT / "examples" / "configs").glob("deepseek-v41-*.config")}
    assert examples <= set(CASES)
    kit = json.loads((ROOT / "scripts/bench/kits/gate/matrix-gate.json").read_text())
    assert {e["name"] for e in kit["entries"] if e.get("family") == "deepseek_v41"} <= set(CASES)
    assert {p.stem for p in FIXTURES.glob("*.json")} == set(CASES)


def test_shared_features_reach_v41_with_no_v41_code(tmp_root, monkeypatch) -> None:
    """Keys added for every family (speculation trace, THERMAL, PROFILE switches,
    the shared draft policy, probe dumps) reach the V4.1 coordinator and workers."""
    import v41_launch_capture as capture_module

    trace = tmp_root / "trace" / "rounds.jsonl"
    probe = tmp_root / "probe"
    probe.mkdir()
    extra = (f"SPECULATION_TRACE={trace}\nTHERMAL=off\nPROFILE=transport-timing,expert-timing\n"
             f"DRAFT_SURVIVAL_LENGTHS=on\nPROBE_DUMP_ROOT={probe}\nSPARK_INTAKE=pinned\nCONSOLE_TEXT=off\n")
    original = capture_module.example_config

    def with_keys(name, directory):
        config = original(name, directory)
        config.write_text(config.read_text() + extra)
        return config

    monkeypatch.setattr(capture_module, "example_config", with_keys)
    result = capture_module.capture(ROOT, "deepseek-v41-mxfp4-max")
    coordinator = parse_run(result["containers"][0]["argv"])
    env = coordinator["env"]
    assert env["CUTEAFD_SPECULATION_TRACE"] == str(trace)
    assert f"{trace.parent}:{trace.parent}" in coordinator["mounts"]
    assert env["CUTEAFD_THERMAL"] == "off"
    assert env["CUTEAFD_PROTOCOL_V2_TCP_TIMING"] == "1" and env["CUTEAFD_PROFILE_HOST"]
    assert env["CUTEAFD_DRAFT_SURVIVAL_LENGTHS"] == "1"
    assert env["CUTEAFD_PROBE_DUMP_ROOT"] == str(probe)
    assert env["CUTEAFD_SPARK_INTAKE"] == "pinned" and env["CUTEAFD_CONSOLE_TEXT"] == "false"
    for container in result["containers"][1:]:
        worker = parse_run(container["argv"])["env"]
        assert worker["CUTEAFD_THERMAL"] == "off"
        assert worker["CUTEAFD_EXPERT_TIMING"] == "1"
        assert worker["CUTEAFD_PROFILE_HOST"] == container["host"]


def test_serving_containers_run_as_the_host_user_with_the_hub_at_hf() -> None:
    """#108 and #111: every V4.1 serving container and planner run as the host user
    with HF_HOME=/hf and the hub at /hf/hub, so binaries that resolve snapshots
    under HF_HOME (pre-#108 seals included) find drafters and checkpoints."""
    for case in CASES:
        golden = json.loads((FIXTURES / f"{case}.json").read_text())
        for container in golden["containers"]:
            run = parse_run(container["argv"])
            assert any(option.startswith("--user ") for option in run["options"]), case
            assert run["env"]["HF_HOME"] == "/hf", case
            assert any(mount.endswith(":/hf/hub:ro") for mount in run["mounts"]), case
            assert not any("/root/" in mount for mount in run["mounts"]), case


@pytest.mark.parametrize("value", [None, "off", "draft", "all"])
def test_v41_fp8_head_key_reaches_planner_and_coordinator(tmp_root, monkeypatch, value) -> None:
    """V41_FP8_HEAD=off|draft|all forwards CUTEAFD_V41_FP8_HEAD to the coordinator and
    the planner (both read it); unset forwards nothing (the engine default, all)."""
    import v41_launch_capture as capture_module

    original = capture_module.example_config

    def with_key(name, directory):
        config = original(name, directory)
        if value is not None:
            config.write_text(config.read_text() + f"V41_FP8_HEAD={value}\n")
        return config

    monkeypatch.setattr(capture_module, "example_config", with_key)
    result = capture_module.capture(ROOT, "deepseek-v41-mxfp4-max")
    coordinator = parse_run(result["containers"][0]["argv"])["env"]
    plan = result["plans"][0]
    if value is None:
        assert "CUTEAFD_V41_FP8_HEAD" not in coordinator
        assert not any("CUTEAFD_V41_FP8_HEAD" in arg for arg in plan)
    else:
        assert coordinator["CUTEAFD_V41_FP8_HEAD"] == value
        assert f"CUTEAFD_V41_FP8_HEAD={value}" in plan
        assert result["containers"][0]["argv"].count(f"CUTEAFD_V41_FP8_HEAD={value}") == 1
    for worker in result["containers"][1:]:
        assert "CUTEAFD_V41_FP8_HEAD" not in parse_run(worker["argv"])["env"]


def test_v41_rejects_kv_cache_as_an_unused_key(tmp_root, monkeypatch) -> None:
    """V4.1 has no KV-precision flag. KV_CACHE is a hard error before any container."""
    import v41_launch_capture as capture_module

    original = capture_module.example_config

    def with_key(name, directory):
        return capture_module.append_config(original(name, directory), "KV_CACHE=fp8\n")

    monkeypatch.setattr(capture_module, "example_config", with_key)
    with pytest.raises(RuntimeError, match=r"unused configuration key: KV_CACHE"):
        capture_module.capture(ROOT, "deepseek-v41-mxfp4-max")


def test_v41_fp8_head_key_rejects_bad_values_and_env_conflicts(tmp_root, monkeypatch) -> None:
    import subprocess
    from v41_launch_capture import example_config
    config = example_config("deepseek-v41-mxfp4-max", tmp_root)
    config.write_text(config.read_text() + "V41_FP8_HEAD=bf16\n")
    env = {**os.environ, "HOME": str(tmp_root), "PATH": "/usr/bin:/bin"}
    hf = tmp_root / "hf" / "hub" / "models--deepseek-ai--DeepSeek-V4.1-Flash" / "snapshots" / "dba1be0a40aa45a94ad051997016db3960a90277"
    hf.mkdir(parents=True, exist_ok=True)
    (hf / "config.json").write_text('{"model_type": "deepseek_v41"}')
    env["HF_HOME"] = str(tmp_root / "hf")
    result = subprocess.run(["bash", str(ROOT / "run.sh"), "--config", str(config), "--dry-run"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 2 and "V41_FP8_HEAD must be off, draft or all" in result.stderr
    result = subprocess.run(["bash", str(ROOT / "run.sh"), "--config", str(config), "--set", "V41_FP8_HEAD=off",
                             "--dry-run"], capture_output=True, text=True,
                            env={**env, "CUTEAFD_V41_FP8_HEAD": "draft"})
    assert result.returncode == 2 and "conflicts with CUTEAFD_V41_FP8_HEAD=draft" in result.stderr
    # The key belongs to the shared key grammar.
    known = subprocess.run(["bash", "-c", "source scripts/lib/release-common.sh; release_known_key V41_FP8_HEAD"],
                           cwd=ROOT)
    assert known.returncode == 0
