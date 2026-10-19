"""Kimi K3 launches through the shared launcher (run.sh -> run-family.sh), the
example configs and the gate kit, from WIP slots only.

CPU only: docker, ssh, nest and nvidia-smi are stubs that print their argv
(test_run_family_dispatch._family_launch_result).
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from test_run_family_dispatch import _family_launch_result

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("wip_cards_new_families", ROOT / "scripts/bench/wip-cards.py")
cards = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts/lib"))
SPEC.loader.exec_module(cards)

KIMI = {"model_type": "kimi_k3", "text_config": {"model_type": "kimi_linear", "num_hidden_layers": 93,
                                                 "first_k_dense_replace": 1, "moe_layer_freq": 1},
        "vision_config": {"depth": 2}}
KIMI_ID = "wrldsuksgo2mars/Kimi-K3-EXL3-K2"
SIX = "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\n" for i in range(6))
KIMI_KEYS = "SPARK_COUNT=6\nRTX_GPUS=2\nSPECULATOR=off\n" + SIX
PLANNED = {"placement_supported": True, "fits": True, "spark_budget_bytes": 108 << 30}


def _kimi(tmp_path: Path, keys: str = "", *, wip: bool = True, **kwargs):
    return _family_launch_result(tmp_path, KIMI, KIMI_ID, KIMI_KEYS + keys,
                                 extra_args=("--wip", "k3") if wip else (),
                                 encoder_plan=kwargs.pop("encoder_plan", PLANNED), **kwargs)


def _line(result, needle: str) -> str:
    assert result.returncode == 0, result.stderr
    return next(line for line in result.stderr.splitlines() if needle in line)


def _flag(line: str, flag: str) -> str | None:
    """FLAG's value in the cuteafd command of a docker run line (not docker's own options)."""
    tokens = line.split()
    if "cuteafd" in tokens:
        tokens = tokens[tokens.index("cuteafd"):]
    return tokens[tokens.index(flag) + 1] if flag in tokens else None


def test_kimi_dispatches_to_serve_kimi_with_its_own_defaults(tmp_path: Path) -> None:
    result = _kimi(tmp_path)
    serve = _line(result, "cuteafd serve-kimi --snapshot")
    assert "/hf/hub/models--wrldsuksgo2mars--Kimi-K3-EXL3-K2/snapshots/abc" in serve
    assert _flag(serve, "--peers") == ",".join(f"10.0.0.{i + 1}:19441" for i in range(6))
    assert (_flag(serve, "--device"), _flag(serve, "--split-device")) == ("0", "1")
    assert _flag(serve, "--max-sequences") == "16"
    assert _flag(serve, "--embedding-placement") == "gpu"
    assert _flag(serve, "--prefix-cache-entries") == "20"
    # serve-kimi's own defaults stay its own: compiled 1M context, automatic KV pool,
    # its onboarded-layer default; it takes no speculator or media options.
    for absent in ("--max-context", "--pool-tokens", "--rtx-expert-layers", "--vision", "--audio",
                   "--draft", "--mtp", "--dspark", "--local-experts"):
        assert absent not in serve.split(), absent
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert len(workers) == 6
    # No hard-coded Spark budget: the planner picks the family's and the workers get it.
    plan = _line(result, "cuteafd plan")
    assert "--spark-budget-gib" not in plan.split()
    assert all(f"--device-budget-bytes {108 << 30} " in worker for worker in workers)
    assert "--world 6" in workers[0] and "--vision off --audio off" in plan
    assert _flag(plan, "--spark-ranks") == "6" and _flag(plan, "--rtx") == "2"
    assert _flag(plan, "--speculator") == "off" and "--require-ready" not in plan.split()
    # The slot's binary is asked for serve-kimi before anything is planned.
    lines = result.stderr.splitlines()
    probe = next(i for i, line in enumerate(lines) if "dst=/opt/cuteafd-serve-probe" in line)
    assert lines[probe].endswith("serve-kimi --help ")
    assert probe < lines.index(plan)


def test_kimi_forwards_typed_keys_when_set(tmp_path: Path) -> None:
    serve = _line(_kimi(tmp_path, "RTX_EXPERT_LAYERS=5\nPOOL_TOKENS=1048576\nMAX_CONTEXT_TOKENS=262144\n"
                                  "CONCURRENCY=8\nSPARK_DEVICE_BUDGET_BYTES=107374182400\n"),
                  "cuteafd serve-kimi --snapshot")
    assert _flag(serve, "--rtx-expert-layers") == "5"
    assert _flag(serve, "--pool-tokens") == "1048576"
    assert _flag(serve, "--max-context") == "262144"
    assert _flag(serve, "--max-sequences") == "8"


def test_kimi_explicit_spark_budget_reaches_plan_and_workers(tmp_path: Path) -> None:
    result = _kimi(tmp_path, "SPARK_DEVICE_BUDGET_BYTES=107374182400\n")
    assert _flag(_line(result, "cuteafd plan"), "--spark-budget-gib") == "100.0"
    assert "--device-budget-bytes 107374182400 " in _line(result, "cuteafd expertd-native")


@pytest.mark.parametrize("keys,diagnostic", [
    ("SPECULATOR=dflash2\n", "SPECULATOR=dflash2 does not apply to kimi_k3"),
    ("SPECULATOR=mtp\n", "SPECULATOR=mtp does not apply to kimi_k3"),
    ("ATTENTION_PLACEMENT=heads\n", "kimi_k3 supports only ATTENTION_PLACEMENT=auto"),
    ("ATTENTION_PLACEMENT=context\n", "kimi_k3 supports only ATTENTION_PLACEMENT=auto"),
    ("VISION=rtx\n", "kimi_k3 bundled media execution is not qualified"),
    ("AUDIO=spark\n", "kimi_k3 bundled media execution is not qualified"),
    ("SPARK_COUNT=4\n", "exactly six Spark ranks"),
    ("SPARK_COUNT=0\n", "exactly six Spark ranks"),
    ("EXPERT_BACKEND=local\nSPARK_COUNT=0\n", "exactly six Spark ranks"),
    ("RTX_GPUS=1\n", "TP2 attention on two RTX GPUs"),
    ("COORDINATOR_SPLIT=off\n", "TP2 attention on two RTX GPUs"),
    ("EMBEDDING=host\n", "GPU token embeddings only"),
    ("RTX_EXPERT_LAYERS=auto\n", "fixed routed-layer count"),
    ("RTX_EXPERT_LAYERS=94\n", "RTX_EXPERT_LAYERS must be"),
    ("KV_CACHE=bf16\n", "unused configuration key: KV_CACHE"),
    ("SPECULATOR_MODEL_ID=x/y\n", "unused configuration key: SPECULATOR_MODEL_ID"),
    ("L2_PREFETCH=auto\n", "unused configuration key: L2_PREFETCH"),
])
def test_kimi_refuses_unsupported_configuration(tmp_path: Path, keys: str, diagnostic: str) -> None:
    result = _kimi(tmp_path, keys)
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert "docker run -d" not in result.stderr
    assert "Traceback" not in result.stderr


def test_kimi_refuses_release_images(tmp_path: Path) -> None:
    result = _kimi(tmp_path, wip=False)
    assert result.returncode != 0
    assert "kimi_k3 serving is under qualification; run it from a WIP slot" in result.stderr
    assert "docker run -d" not in result.stderr and "cuteafd plan" not in result.stderr


def test_kimi_refuses_a_slot_without_serve_kimi(tmp_path: Path) -> None:
    # The slot's binary predates serve-kimi: --help on it fails, nothing is planned or started.
    result = _kimi(tmp_path, probe_fails=True)
    assert result.returncode == 2, result.stderr
    assert "WIP slot k3 has no 'serve-kimi' command; build the slot from a commit that serves kimi_k3" in result.stderr
    assert "cuteafd plan" not in result.stderr
    assert "docker run -d" not in result.stderr and "expertd-native" not in result.stderr


def test_kimi_refuses_a_plan_that_is_not_wip_launchable(tmp_path: Path) -> None:
    result = _kimi(tmp_path, encoder_plan={**PLANNED, "fits": False})
    assert result.returncode != 0
    assert "invalid deployment specification" in result.stderr
    assert "docker run -d" not in result.stderr


def test_family_table_and_planner_fixtures_name_kimi() -> None:
    table = ROOT / "scripts/lib/checkpoint-family.py"
    out = subprocess.run(["python3", str(table), "/dev/stdin"], input=json.dumps(KIMI),
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["kimi_k3", "kimi_k3", "1", "92"]
    fixtures = json.loads((ROOT / "rust/crates/cuteafd-loader/tests/fixtures/launch-families.json").read_text())
    lines = {case["line"] for case in fixtures}
    assert "kimi_k3 kimi_k3 1 92" in lines
    launch = (ROOT / "rust/crates/cuteafd-loader/src/plan/launch.rs").read_text()
    assert '"kimi_k3" =>' in launch


# --- the example config --------------------------------------------------------

EXAMPLE = ROOT / "examples/configs/kimi-k3-exl3-max.config"
PRIVATE = ("raptor", "ostrich", "dodo", "emu", "kiwi", "rhea", "moa", "10.55.", "/home/", "/mnt/", "sparknest")


def _keys(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if re.match(r"^[A-Z_0-9]+=", line))


def test_kimi_example_config_is_the_max_card() -> None:
    values = _keys(EXAMPLE)
    assert values["MODEL_ID"] == KIMI_ID
    assert values["MODEL_REVISION"] == "acba35b61a672561652b16a96fca1be85c1bd875"
    assert (values["RTX_GPUS"], values["SPARK_COUNT"], values["RTX_EXPERT_LAYERS"]) == ("2", "6", "5")
    assert (values["SPECULATOR"], values["VISION"], values["AUDIO"], values["POOL_TOKENS"]) == ("off", "off", "off", "auto")
    for absent in ("MAX_CONTEXT_TOKENS", "SPARK_DEVICE_BUDGET_BYTES", "KV_CACHE", "CONCURRENCY"):
        assert absent not in values
    for rank in range(6):
        assert values[f"SPARK_{rank}_HOST"] == f"REPLACE-ME-spark-{rank}"
        assert values[f"SPARK_{rank}_LANE_A"] == f"192.0.2.{rank + 1}"
    text = EXAMPLE.read_text()
    assert not any(needle in text.lower() for needle in PRIVATE)
    assert "./run.sh --config examples/configs/kimi-k3-exl3-max.config --wip SLOT --dry-run" in text


def test_kimi_example_config_launches_through_run_family(tmp_path: Path) -> None:
    keys = "".join(f"{key}={value}\n" for key, value in _keys(EXAMPLE).items()
                   if not key.startswith("SPARK_") and key not in ("MODEL_ID", "MODEL_REVISION"))
    result = _family_launch_result(tmp_path, KIMI, KIMI_ID, keys + "SPARK_COUNT=6\n" + SIX,
                                   extra_args=("--wip", "k3"), encoder_plan=PLANNED)
    serve = _line(result, "cuteafd serve-kimi --snapshot")
    assert _flag(serve, "--rtx-expert-layers") == "5"
    assert "--pool-tokens" not in serve.split() and "--max-context" not in serve.split()


# --- the gate card -------------------------------------------------------------

def test_kimi_gate_card_lists_and_generates_a_wip_launch(tmp_path: Path) -> None:
    card = cards.load_cards(ROOT / "scripts/bench/kits/gate", ["kimi-k3-max"])[0]
    assert card["family"] == "kimi_k3" and card["gpus"] == [0, 1]
    assert card["sparks"] == ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"]
    assert card["profile"] == "smoke"
    assert cards.config(card["config"])["MODEL_ID"] == KIMI_ID
    assert cards.slot_needs(card)["coordinator"] == cards.slot_needs(card)["spark"] == ["exl3/exl3-kimik3-k23"]
    key = tmp_path / "key"
    key.write_text("k" * 64 + "\n")
    key.chmod(0o600)
    job = cards.generate(card, "cand", {"instance": "t", "slot": "s"}, tmp_path, key, {}, {"fabric"}, None, 1)
    values = cards.config(Path(job["entry"]["config"]))
    assert job["entry"]["run_args"] == ["--wip", "s"]
    assert (values["SPARK_COUNT"], values["RTX_GPUS"], values["SPECULATOR"]) == ("6", "2", "off")
    assert not any(key.endswith("_LANE_B") for key in values)


def test_gate_card_ports_stay_unique() -> None:
    matrix = json.loads((ROOT / "scripts/bench/kits/gate/matrix-gate.json").read_text())
    for key in ("ADDR", "EXPERT_PORT"):
        values = [entry["set"][key] for entry in matrix["entries"] if key in entry["set"]]
        assert len(values) == len(set(values)), key
