"""DeepSeek V4.1 launches through the shared family launcher (#82).

The goldens in fixtures/v41-launch/ were recorded from run.sh's private V4.1
path (scripts/tests/v41_launch_capture.py, before #82). Today's launcher must
start the same coordinator and Spark containers for every V4.1 example config
and the gate kit's v41-flash-max/min: the same options, mounts, environment,
image and serve command. The only differences allowed are the intended
additions listed in INTENDED: shared launch features that now reach V4.1 at
their defaults, with no V4.1 code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v41_launch_capture import CASES, FIXTURES, ROOT, capture, parse_run  # noqa: E402

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
        # shared_trace_args: the shared draft policy switches, default off.
        "CUTEAFD_DRAFT_COST_BUCKETS": "0",
        "CUTEAFD_DRAFT_CONFIDENCE": "0",
        "CUTEAFD_DRAFT_SURVIVAL_LENGTHS": "0",
        # shared_thermal_args: the GPU throttle sampler (#114), default on.
        "CUTEAFD_THERMAL": "on",
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
        assert after["options"] == before["options"], where
        # shared_thermal_args also mounts the host power governor's state read-only
        # when gpu-powerd runs on this host (#114).
        governor = ["/run/gpu-powerd:/run/gpu-powerd:ro"] if role == "coordinator" and os.path.isdir("/run/gpu-powerd") else []
        assert after["mounts"] == sorted(before["mounts"] + governor), where
        assert after["image"] == before["image"], where
        added = {key: value for key, value in after["env"].items() if key not in before["env"]}
        assert added == INTENDED[role], where
        assert {key: after["env"][key] for key in before["env"]} == before["env"], where
        # The serve command is the same set of flags and values (clap ignores order;
        # --usage now precedes nothing it depends on).
        assert after["command"][:3] == before["command"][:3], where
        assert flag_values(after["command"]) == flag_values(before["command"]), where
    # The planner preflight (`cuteafd plan`, HF_HOME=/hf) is unchanged too.
    assert current["plans"] == golden["plans"], case


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
