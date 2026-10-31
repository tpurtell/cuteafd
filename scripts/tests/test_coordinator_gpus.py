"""COORDINATOR_GPUS selects the coordinator GPUs, primary first, the same way for every family.

A fake three-GPU host (nvidia-smi reports GPUs 0, 1 and 2) launches each family
through the real run-family.sh (test_run_family_dispatch._family_launch_result):
an explicit pair reaches the coordinator container in that order, primary as
CUDA ordinal 0; a missing, repeated, busy or unfit GPU fails before any
container starts; unset keeps the automatic choice.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from launcher_sandbox import FAKE_GPU_PCI, FAKE_GPU_UUIDS, fake_gpus, fake_nvidia_smi
from test_run_family_dispatch import SPLIT_CONFIGS, _coordinator_gpus, _family_launch_result

ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / "scripts/lib/coordinator-gpus.py"
THREE = (0, 1, 2)


def _sparks(count: int) -> str:
    return f"SPARK_COUNT={count}\n" + "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\n" for i in range(count))


# Every family with a two-GPU coordinator layout: checkpoint, model id, extra keys
# and the serve command its coordinator runs.
FAMILIES = {
    "deepseek_v41": ({"model_type": "deepseek_v41"}, "deepseek-ai/DeepSeek-V4.1-Flash",
                     "MODEL_REVISION=\n" + _sparks(4), "serve-native"),
    "deepseek_v4": ({"model_type": "deepseek_v4", "num_hidden_layers": 61}, "test/V4", "", "serve-dsv4"),
    "glm5": ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}, "test/glm", "",
             "serve-glm"),
    "glm5_flash": (SPLIT_CONFIGS["glm5_flash"], "test/glmf", "", "serve-glmf"),
    "mimo_v2": (SPLIT_CONFIGS["mimo_pro"], "test/mimo", "", "serve-mimo"),
    "qwen4": (SPLIT_CONFIGS["qwen4"], "test/qwen4", "SPARK_COUNT=0\nEXPERT_BACKEND=local\n", "serve-qwen4"),
    "kimi_k3": ({"model_type": "kimi_k3", "text_config": {"model_type": "kimi_linear", "num_hidden_layers": 93,
                                                         "first_k_dense_replace": 1, "moe_layer_freq": 1}},
                "test/Kimi-K3", _sparks(6), "serve-kimi"),
    "step5": ({"model_type": "step5_vl", "text_config": {"model_type": "step5", "num_hidden_layers": 92,
                                                        "moe_layer_list": list(range(3, 91))}},
              "test/Step-5", "", "serve-step5"),
}
WIP_ONLY = {"kimi_k3", "step5"}


def _launch(tmp_path: Path, family: str, keys: str, **kwargs) -> tuple[subprocess.CompletedProcess[str], str | None]:
    config, model, base, serve = FAMILIES[family]
    extra = ("--wip", "gpus") if family in WIP_ONLY or family == "deepseek_v41" else ()
    plan = {"placement_supported": True, "fits": True, "spark_budget_bytes": 108 << 30} if family in WIP_ONLY else None
    result = _family_launch_result(tmp_path, config, model, base + "SPECULATOR=off\nVISION=off\nAUDIO=off\n" + keys,
                                   physical_gpus=kwargs.pop("physical_gpus", THREE), extra_args=extra,
                                   encoder_plan=plan, **kwargs)
    launch = next((line for line in result.stderr.splitlines()
                   if "docker run -d" in line and f"cuteafd {serve} " in line or
                   "docker run -d" in line and f" {serve} " in line), None)
    return result, launch


def _logical_devices(family: str, launch: str) -> tuple[int, int] | None:
    """The serve command's primary and second CUDA ordinals (V4.1: by --rtx-gpus)."""
    argv = shlex.split(launch)
    argv = argv[next(i for i, a in enumerate(argv) if a.startswith("serve-")):]  # the serve command's own flags
    if family == "deepseek_v41":
        return (0, 1) if argv[argv.index("--rtx-gpus") + 1] == "2" else None
    if "--split-device" not in argv:
        return None
    return int(argv[argv.index("--device") + 1]), int(argv[argv.index("--split-device") + 1])


def _no_containers(result: subprocess.CompletedProcess[str]) -> None:
    assert "docker run -d" not in result.stderr, result.stderr


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("pair", ["1,2", "2,1"])
def test_explicit_pair_is_the_container_device_set_primary_first(tmp_path: Path, family: str, pair: str) -> None:
    result, launch = _launch(tmp_path, family, f"COORDINATOR_GPUS={pair}\n")
    assert result.returncode == 0, result.stderr
    assert launch, result.stderr
    # Both GPUs, in the named order: CUDA ordinal 0 is the primary, 1 the second.
    assert _coordinator_gpus(launch) == [int(x) for x in pair.split(",")]
    assert _logical_devices(family, launch) == (0, 1)
    assert FAKE_GPU_UUIDS[0] not in launch


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("selector", ["index", "uuid", "pci"])
def test_pair_may_be_named_by_uuid_or_pci_bus_id(tmp_path: Path, family: str, selector: str) -> None:
    name = {"index": lambda i: str(i), "uuid": lambda i: FAKE_GPU_UUIDS[i],
            "pci": lambda i: FAKE_GPU_PCI[i].removeprefix("0000")}[selector]
    result, launch = _launch(tmp_path, family, f"COORDINATOR_GPUS={name(2)},{name(1)}\n")
    assert result.returncode == 0, result.stderr
    assert _coordinator_gpus(launch) == [2, 1]


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("keys,diagnostic", [
    ("COORDINATOR_GPUS=1,3\n", "names GPU 3, but nvidia-smi reports no such physical GPU"),
    ("COORDINATOR_GPUS=2,2\n", "names GPU 2 twice"),
    ("COORDINATOR_GPUS=1,2,0\n", "must name one or two GPUs"),
    ("COORDINATOR_GPUS=1;2\n", "is not a host GPU index, PCI bus id or GPU UUID"),
    ("COORDINATOR_GPUS=1\nRTX_GPUS=2\n", "names 1 GPU(s), but RTX_GPUS/COORDINATOR_SPLIT ask for 2"),
    ("COORDINATOR_GPUS=1,2\nRTX_GPUS=1\n", "names 2 GPU(s), but RTX_GPUS/COORDINATOR_SPLIT ask for 1"),
])
def test_invalid_selection_fails_before_containers(tmp_path: Path, family: str, keys: str, diagnostic: str) -> None:
    result, _ = _launch(tmp_path, family, keys)
    assert result.returncode == 2, result.stderr
    assert diagnostic in result.stderr
    _no_containers(result)


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("busy", [1, 2])
def test_busy_explicit_gpu_fails_clearly(tmp_path: Path, family: str, busy: int) -> None:
    result, _ = _launch(tmp_path, family, "COORDINATOR_GPUS=1,2\n", busy_gpus=(busy,))
    assert result.returncode == 2, result.stderr
    assert f"GPU {busy} ({FAKE_GPU_UUIDS[busy]})" in result.stderr
    assert f"as logical GPU {busy - 1}" in result.stderr
    assert "COORDINATOR_GPUS" in result.stderr
    _no_containers(result)


@pytest.mark.parametrize("family", sorted(set(FAMILIES) - {"deepseek_v41"}))
def test_explicit_pair_too_small_for_the_plan_fails(tmp_path: Path, family: str) -> None:
    result, _ = _launch(tmp_path, family, "COORDINATOR_GPUS=2,1\n", plan_rtx_bytes=96 << 30)
    assert result.returncode == 2, result.stderr
    assert "coordinator GPU cannot hold this deployment" in result.stderr
    assert f"GPU 2 ({FAKE_GPU_UUIDS[2]}) as logical GPU 0" in result.stderr
    _no_containers(result)


def test_v41_explicit_pair_is_checked_not_replaced_by_the_free_memory_pick(tmp_path: Path) -> None:
    # GPU 0 has the most free memory; the automatic pick would pair it, the explicit pair never does.
    result, launch = _launch(tmp_path, "deepseek_v41", "COORDINATOR_GPUS=2,1\nRTX_GPUS=auto\n")
    assert result.returncode == 0, result.stderr
    assert _coordinator_gpus(launch) == [2, 1]
    # A pair whose second GPU cannot hold V4.1's logical GPU 1 fails; it is not swapped for GPU 0.
    result, _ = _launch(tmp_path / "unfit", "deepseek_v41", "COORDINATOR_GPUS=2,1\n", busy_gpus=(1,))
    assert result.returncode == 2
    assert "cannot serve two-RTX V4.1" in result.stderr and "as logical GPU 1" in result.stderr
    _no_containers(result)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_single_explicit_gpu_serves_alone(tmp_path: Path, family: str) -> None:
    if family == "kimi_k3":
        pytest.skip("Kimi K3 serves TP2 attention on two RTX GPUs only")
    result, launch = _launch(tmp_path, family, "COORDINATOR_GPUS=2\n")
    if family == "qwen4":
        # Spark-free Qwen owns whole layers on a pair; one GPU needs Spark experts.
        result, launch = _launch(tmp_path / "sparks", family, "COORDINATOR_GPUS=2\nSPARK_COUNT=1\nEXPERT_BACKEND=spark\n")
    assert result.returncode == 0, result.stderr
    assert _coordinator_gpus(launch) == [2]
    assert _logical_devices(family, launch) is None


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_unset_keeps_the_automatic_choice(tmp_path: Path, family: str) -> None:
    result, launch = _launch(tmp_path, family, "")
    assert result.returncode == 0, result.stderr
    if family == "deepseek_v41":
        # The free-memory pick: GPU 0 primary, the emptiest other card second (ties to the lower index).
        assert _coordinator_gpus(launch) == [0, 1]
    elif family == "step5":
        assert _coordinator_gpus(launch) in ([0], [0, 1])
    else:
        # The lowest-index GPU, plus the next one when the family splits by default.
        assert _coordinator_gpus(launch) in ([0], [0, 1])
        assert _coordinator_gpus(launch)[0] == 0


def test_v41_unset_pairs_the_emptiest_other_gpu(tmp_path: Path) -> None:
    result, launch = _launch(tmp_path, "deepseek_v41", "", busy_gpus=(1,))
    assert result.returncode == 0, result.stderr
    assert _coordinator_gpus(launch) == [0, 2]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_split_off_serves_the_named_primary_alone(tmp_path: Path, family: str) -> None:
    if family in {"deepseek_v41", "kimi_k3", "qwen4"}:
        pytest.skip("V4.1 has no COORDINATOR_SPLIT; Kimi needs two GPUs; Spark-free Qwen owns whole layers")
    result, launch = _launch(tmp_path, family, "COORDINATOR_GPUS=2\nCOORDINATOR_SPLIT=off\n")
    assert result.returncode == 0, result.stderr
    assert _coordinator_gpus(launch) == [2]


# --- the selector itself -----------------------------------------------------

def _selector(tmp_path: Path, *args: str, gpus=None) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake_nvidia_smi(tmp_path)
    env = {**os.environ, "FAKE_GPUS": json.dumps(gpus if gpus is not None else fake_gpus(3))}
    return subprocess.run([sys.executable, str(SELECTOR), "--nvidia-smi", str(tmp_path / "nvidia-smi"), *args],
                          capture_output=True, text=True, env=env, timeout=30)


@pytest.mark.parametrize("selectors,count,expected", [
    ("", "auto", [0, 1]), ("", "1", [0]), ("", "2", [0, 1]),
    ("2", "auto", [2]), ("2,0", "auto", [2, 0]), ("2,0", "2", [2, 0]),
    (f"{FAKE_GPU_UUIDS[1]},{FAKE_GPU_PCI[2]}", "auto", [1, 2]),
    (f"{FAKE_GPU_UUIDS[1].lower()}, {FAKE_GPU_PCI[2][4:].upper()}", "auto", [1, 2]),
])
def test_resolve_orders_the_named_gpus(tmp_path: Path, selectors: str, count: str, expected: list[int]) -> None:
    result = _selector(tmp_path, "resolve", "--selectors", selectors, "--count", count)
    assert result.returncode == 0, result.stderr
    rows = [line.split() for line in result.stdout.splitlines()]
    assert [int(row[0]) for row in rows] == expected
    assert [row[1] for row in rows] == [FAKE_GPU_UUIDS[i] for i in expected]


def test_resolve_on_a_one_gpu_host_returns_that_gpu(tmp_path: Path) -> None:
    result = _selector(tmp_path, "resolve", "--count", "auto", gpus=fake_gpus(1))
    assert result.returncode == 0 and result.stdout.split()[0] == "0"


def test_admit_reclaims_only_the_replaced_container(tmp_path: Path) -> None:
    deployment = tmp_path / "plan.json"
    deployment.write_text(json.dumps({"spark_ranks": 1, "memory_layout": {"devices": [
        {"kind": "rtx", "index": 0, "items": [{"category": "weights", "bytes": 40 << 30},
                                               {"category": "experts", "bytes": 30 << 30},
                                               {"category": "kv", "bytes": 20 << 30}]}]}}))
    gpus = fake_gpus(3, free_mib=20 << 10, apps=((1, 123, 60 << 10), (1, 456, 4 << 10)))
    args = ("admit", "--uuids", FAKE_GPU_UUIDS[1], "--deployment", str(deployment))
    refused = _selector(tmp_path, *args, gpus=gpus)
    assert refused.returncode == 2 and "needs 40960 MiB for its fixed owners" in refused.stderr
    admitted = _selector(tmp_path, *args, "--reclaim-pid", "123", gpus=gpus)
    assert admitted.returncode == 0, admitted.stderr


def test_admit_charges_onboarded_experts_without_sparks(tmp_path: Path) -> None:
    deployment = tmp_path / "plan.json"
    deployment.write_text(json.dumps({"spark_ranks": 0, "memory_layout": {"devices": [
        {"kind": "rtx", "index": 0, "items": [{"category": "weights", "bytes": 10 << 30},
                                               {"category": "experts", "bytes": 50 << 30}]}]}}))
    result = _selector(tmp_path, "admit", "--uuids", FAKE_GPU_UUIDS[0], "--deployment", str(deployment),
                       gpus=fake_gpus(1, free_mib=40 << 10))
    assert result.returncode == 2 and "needs 61440 MiB" in result.stderr


def test_removed_selector_keys_are_unknown(tmp_path: Path) -> None:
    for key in ("COORDINATOR_GPU", "COORDINATOR_SPLIT_GPU", "COORDINATOR_GPU_UUID", "COORDINATOR_GPU_PCI_BUS_ID"):
        result, _ = _launch(tmp_path / key, "glm5_flash", f"{key}=1\n")
        assert result.returncode != 0 and f"unknown configuration key: {key}" in result.stderr
        _no_containers(result)


def test_every_launch_path_resolves_through_the_shared_selector() -> None:
    family = (ROOT / "scripts/launch/run-family.sh").read_text()
    v41 = family.split("launch_deepseek_v41() {", 1)[1].split("\n}\n", 1)[0]
    generic = family.split("launch_deepseek_v41\n  exit 0\nfi", 1)[1]
    for body in (v41, generic):
        assert body.count("release_resolve_coordinator_gpus ") == 1
        assert "release_admit_coordinator_gpus " in body
        assert "release_coordinator_device_args " in body
        # Only Spark workers take GPUs any other way (all of their own host's).
        assert set(re.findall(r"cuteafd_cdi_gpu_args (\S+)", body)) <= {"all", "cuteafd_cdi_require", "release_serving_docker_args"}
    for name in ('"$coordinator"', '"$coordinator_name"'):
        start = family.index(f'--name {name} --restart no')
        assert family[start:start + 80].split("--restart no ", 1)[1].startswith('"${RELEASE_COORDINATOR_DEVICE_ARGS[@]}"')
    assert "coordinator-gpus.py\" select" in v41 and "--selectors \"$COORDINATOR_GPUS\"" in v41
    for script in ("wip.sh", "scripts/build/cuteafd-dev.sh"):
        assert 'release_resolve_coordinator_gpus "$COORDINATOR_GPUS" auto' in (ROOT / script).read_text()
    assert "nvidia-smi" not in generic.split("release_resolve_coordinator_gpus ", 1)[1].split("plan_json=", 1)[0] \
        .replace("nvidia-smi --id=\"$gpu\"", "")
