"""Kimi K3 and Step-5 launch through the shared launcher (run.sh -> run-family.sh), the
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

from test_run_family_dispatch import _family_launch_result, _snapshot

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
    assert str(tmp_path / "hf/hub/models--wrldsuksgo2mars--Kimi-K3-EXL3-K2/snapshots/abc") in serve
    assert _flag(serve, "--peers") == ",".join(f"10.0.0.{i + 1}:19441" for i in range(6))
    assert (_flag(serve, "--device"), _flag(serve, "--split-device")) == ("0", "1")
    assert _flag(serve, "--max-sequences") == "16"
    assert _flag(serve, "--embedding-placement") == "gpu"
    assert _flag(serve, "--prefix-cache-entries") == "20"
    # serve-kimi's own defaults stay its own: the checkpoint core picks the context (1M for a K6
    # core, 512K for BF16; 1M compiled), automatic KV pool;
    # the planner picks the onboarded layers; it takes no speculator or media options.
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    for absent in ("--max-context", "--pool-tokens", "--vision", "--audio",
                   "--draft", "--draft-fp8", "--mtp", "--dspark", "--local-experts"):
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
    assert "--drafter-snapshot" not in plan.split() and "--draft-fp8" not in plan.split()
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


@pytest.mark.parametrize("enabled", [False, True])
def test_kimi_dflash2_uses_one_resolved_fp8_snapshot_for_plan_and_serve(tmp_path: Path, enabled: bool) -> None:
    model, revision = "lightseekorg/kimi-k3-dflash2", "e77935fb4804e17eb55085bffd045eae1d779769"
    _snapshot(tmp_path / "hf", model, {}, revision=revision)
    result = _kimi(tmp_path, f"SPECULATOR={'dflash2' if enabled else 'off'}\n"
                            f"SPECULATOR_MODEL_ID={model}\nSPECULATOR_MODEL_REVISION={revision}\n"
                            + ("SPECULATOR_DRAFTS=7\n" if enabled else ""))
    serve, plan = _line(result, "cuteafd serve-kimi --snapshot"), _line(result, "cuteafd plan")
    assert _flag(plan, "--speculator") == ("dflash2" if enabled else "off")
    if enabled:
        snapshot = str(tmp_path / f"hf/hub/models--{model.replace('/', '--')}/snapshots/{revision}")
        assert _flag(serve, "--draft") == _flag(plan, "--drafter-snapshot") == snapshot
        assert _flag(serve, "--draft-fp8") == _flag(plan, "--draft-fp8") == "true"
        assert _flag(serve, "--draft-fixed") == "7"
    else:
        assert all(flag not in serve.split() for flag in ("--draft", "--draft-fp8", "--draft-fixed"))
        assert all(flag not in plan.split() for flag in ("--drafter-snapshot", "--draft-fp8"))


@pytest.mark.parametrize("cache", [None, "bf16", "fp8"])
def test_kimi_kv_format_reaches_both_plan_and_target_drafter(tmp_path: Path, cache) -> None:
    result = _kimi(tmp_path, "" if cache is None else f"KV_CACHE={cache}\n")
    expected = cache or "fp8"
    assert _flag(_line(result, "cuteafd serve-kimi --snapshot"), "--kv-cache") == expected
    assert _flag(_line(result, "cuteafd plan"), "--kv-cache") == expected


def test_kimi_explicit_spark_budget_reaches_plan_and_workers(tmp_path: Path) -> None:
    result = _kimi(tmp_path, "SPARK_DEVICE_BUDGET_BYTES=107374182400\n")
    assert _flag(_line(result, "cuteafd plan"), "--spark-budget-gib") == "100.0"
    assert "--device-budget-bytes 107374182400 " in _line(result, "cuteafd expertd-native")


@pytest.mark.parametrize("keys,diagnostic", [
    ("SPECULATOR=dflash2\n", "SPECULATOR=dflash2 needs SPECULATOR_MODEL_ID"),
    ("SPECULATOR=mtp\n", "SPECULATOR=mtp does not apply to kimi_k3"),
    ("ATTENTION_PLACEMENT=heads\n", "kimi_k3 supports only ATTENTION_PLACEMENT=auto"),
    ("ATTENTION_PLACEMENT=context\n", "kimi_k3 supports only ATTENTION_PLACEMENT=auto"),
    ("VISION=rtx\n", "kimi_k3 bundled media execution is not qualified"),
    ("AUDIO=spark\n", "kimi_k3 bundled media execution is not qualified"),
    ("RTX_GPUS=1\n", "TP2 attention on two RTX GPUs"),
    ("COORDINATOR_SPLIT=off\n", "TP2 attention on two RTX GPUs"),
    ("EMBEDDING=host\n", "GPU token embeddings only"),
    ("RTX_EXPERT_LAYERS=max\n", "auto or a fixed routed-layer count"),
    ("RTX_EXPERT_LAYERS=94\n", "RTX_EXPERT_LAYERS must be"),
    ("KV_CACHE=int8\n", "KV_CACHE must be bf16 or fp8"),
    ("L2_PREFETCH=auto\n", "unused configuration key: L2_PREFETCH"),
])
def test_kimi_refuses_unsupported_configuration(tmp_path: Path, keys: str, diagnostic: str) -> None:
    result = _kimi(tmp_path, keys)
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert "docker run -d" not in result.stderr
    assert "Traceback" not in result.stderr


def test_kimi_serves_from_release_images(tmp_path: Path) -> None:
    result = _kimi(tmp_path, wip=False)
    assert "under qualification" not in result.stderr
    assert "cuteafd serve-kimi --snapshot" in result.stderr


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
K6_EXAMPLE = ROOT / "examples/configs/kimi-k3-exl3-k6-max.config"
KIMI_K6_ID = "wrldsuksgo2mars/Kimi-K3-EXL3-K2-Core-K6"
PRIVATE = ("raptor", "ostrich", "dodo", "emu", "kiwi", "rhea", "moa", "10.55.", "/home/", "/mnt/", "sparknest")


def _keys(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if re.match(r"^[A-Z_0-9]+=", line))


def test_kimi_example_config_is_the_max_card() -> None:
    values = _keys(EXAMPLE)
    assert values["MODEL_ID"] == KIMI_ID
    assert values["MODEL_REVISION"] == "acba35b61a672561652b16a96fca1be85c1bd875"
    assert (values["RTX_GPUS"], values["SPARK_COUNT"]) == ("2", "6")
    assert "RTX_EXPERT_LAYERS" not in values
    assert (values["SPECULATOR"], values["VISION"], values["AUDIO"], values["POOL_TOKENS"]) == ("off", "off", "off", "auto")
    for absent in ("MAX_CONTEXT_TOKENS", "SPARK_DEVICE_BUDGET_BYTES", "KV_CACHE", "CONCURRENCY"):
        assert absent not in values
    for rank in range(6):
        assert values[f"SPARK_{rank}_HOST"] == f"REPLACE-ME-spark-{rank}"
        assert values[f"SPARK_{rank}_LANE_A"] == f"192.0.2.{rank + 1}"
    text = EXAMPLE.read_text()
    assert not any(needle in text.lower() for needle in PRIVATE)
    assert "./run.sh --config examples/configs/kimi-k3-exl3-max.config --wip SLOT --dry-run" in text


def test_kimi_k6_example_config_serves_the_k6_core_with_its_own_1m_default() -> None:
    values = _keys(K6_EXAMPLE)
    assert (values["MODEL_ID"], values["MODEL_REVISION"]) == (KIMI_K6_ID, "1a28050d2954426a43a3db705e53ca120e5f2b47")
    assert (values["SPECULATOR"], values["SPECULATOR_MODEL_ID"]) == ("dflash2", "lightseekorg/kimi-k3-dflash2")
    assert (values["RTX_GPUS"], values["SPARK_COUNT"], values["POOL_TOKENS"]) == ("2", "6", "auto")
    # Unset: serve-kimi takes the full compiled 1M from the checkpoint's K6 core.
    for absent in ("MAX_CONTEXT_TOKENS", "SPARK_DEVICE_BUDGET_BYTES", "KV_CACHE", "CONCURRENCY", "RTX_EXPERT_LAYERS"):
        assert absent not in values
    text = K6_EXAMPLE.read_text()
    assert not any(needle in text.lower() for needle in PRIVATE)
    assert "./run.sh --config examples/configs/kimi-k3-exl3-k6-max.config --wip SLOT --dry-run" in text
    assert _keys(EXAMPLE).keys() - {"SPECULATOR_MODEL_ID"} == values.keys() - {"SPECULATOR_MODEL_ID"}


def test_kimi_example_config_launches_through_run_family(tmp_path: Path) -> None:
    keys = "".join(f"{key}={value}\n" for key, value in _keys(EXAMPLE).items()
                   if not key.startswith("SPARK_") and key not in ("MODEL_ID", "MODEL_REVISION"))
    result = _family_launch_result(tmp_path, KIMI, KIMI_ID, keys + "SPARK_COUNT=6\n" + SIX,
                                   extra_args=("--wip", "k3"), encoder_plan=PLANNED)
    serve = _line(result, "cuteafd serve-kimi --snapshot")
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    assert "--pool-tokens" not in serve.split() and "--max-context" not in serve.split()


# --- the gate card -------------------------------------------------------------

def test_kimi_gate_card_lists_and_generates_a_wip_launch(tmp_path: Path) -> None:
    card = cards.load_cards(ROOT / "scripts/bench/kits/gate", ["kimi-k3-max"])[0]
    assert card["family"] == "kimi_k3" and card["gpus"] == [0, 1]
    assert card["sparks"] == ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"]
    assert card["profile"] == "smoke"
    assert cards.config(card["config"])["MODEL_ID"] == "wrldsuksgo2mars/Kimi-K3-EXL3-K2-Core-K6"
    assert cards.slot_needs(card)["coordinator"] == cards.slot_needs(card)["spark"] == ["exl3/exl3-kimik3-k23"]
    key = tmp_path / "key"
    key.write_text("k" * 64 + "\n")
    key.chmod(0o600)
    job = cards.generate(card, "cand", {"instance": "t", "slot": "s"}, tmp_path, key, {}, {"fabric"}, None, 1)
    values = cards.config(Path(job["entry"]["config"]))
    assert job["entry"]["run_args"] == ["--wip", "s"]
    assert (values["SPARK_COUNT"], values["RTX_GPUS"], values["SPECULATOR"]) == ("6", "2", "dflash2")
    assert not any(key.endswith("_LANE_B") for key in values)


def test_gate_card_ports_stay_unique() -> None:
    matrix = json.loads((ROOT / "scripts/bench/kits/gate/matrix-gate.json").read_text())
    for key in ("ADDR", "EXPERT_PORT"):
        values = [entry["set"][key] for entry in matrix["entries"] if key in entry["set"]]
        assert len(values) == len(set(values)), key


def test_kimi_k6_core_follows_checkpoint_headers_without_a_launch_key(tmp_path: Path) -> None:
    (tmp_path / "default").mkdir()
    (tmp_path / "removed").mkdir()
    result = _kimi(tmp_path / "default", "")
    assert "--native-k6" not in _line(result, "cuteafd serve-kimi --snapshot").split()
    assert "--native-k6" not in _line(result, "cuteafd plan").split()
    removed = _kimi(tmp_path / "removed", "NATIVE_K6=on\n")
    assert removed.returncode != 0 and "NATIVE_K6" in removed.stderr


# --- Step-5 (#274) ---------------------------------------------------------------

STEP5 = {"model_type": "step5_vl", "vision_config": {"depth": 2},
         "text_config": {"model_type": "step5", "num_hidden_layers": 92, "moe_layer_list": list(range(3, 91))}}
STEP5_FP8 = "stepfun-ai/Step-5-Preview"
STEP5_NVFP4 = "stepfun-ai/Step-5-Preview-NVFP4"
STEP5_PLANNED = {"placement_supported": True, "fits": True, "spark_budget_bytes": 100 << 30}


def _hosts(count: int) -> str:
    return "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\n" for i in range(count))


def _step5(tmp_path: Path, keys: str, *, wip: bool = True, model: str = STEP5_FP8, **kwargs):
    return _family_launch_result(tmp_path, STEP5, model, keys,
                                 extra_args=("--wip", "s5") if wip else (),
                                 encoder_plan=kwargs.pop("encoder_plan", STEP5_PLANNED), **kwargs)


@pytest.mark.parametrize("model,gpus,sparks", [(STEP5_FP8, 2, ranks) for ranks in (4, 5, 6)] + [(STEP5_NVFP4, 1, 4)])
def test_step5_dispatches_to_serve_step5(tmp_path: Path, model: str, gpus: int, sparks: int) -> None:
    result = _step5(tmp_path, f"SPARK_COUNT={sparks}\nRTX_GPUS={gpus}\n" + _hosts(sparks), model=model)
    serve = _line(result, "cuteafd serve-step5 --snapshot")
    assert f"/hf/hub/models--{model.replace('/', '--')}/snapshots/abc" in serve
    assert len(_flag(serve, "--peers").split(",")) == sparks
    assert _flag(serve, "--embedding-placement") == "gpu"
    assert _flag(serve, "--mtp") == "3"
    assert _flag(serve, "--prefix-cache-entries") == "20"
    assert ((_flag(serve, "--device"), _flag(serve, "--split-device")) == ("0", "1")) == (gpus == 2)
    for absent in ("--max-context", "--pool-tokens", "--rtx-expert-layers", "--vision", "--audio",
                   "--draft", "--draft-fp8", "--dspark", "--local-experts"):
        assert absent not in serve.split(), absent
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert len(workers) == sparks and f"--world {sparks}" in workers[0]
    plan = _line(result, "cuteafd plan")
    assert "--spark-budget-gib" not in plan.split()
    assert all(f"--device-budget-bytes {100 << 30} " in worker for worker in workers)
    assert _flag(plan, "--spark-ranks") == str(sparks) and _flag(plan, "--rtx") == str(gpus)
    assert "--vision off --audio off" in plan and "--require-ready" not in plan.split()
    lines = result.stderr.splitlines()
    probe = next(i for i, line in enumerate(lines) if "dst=/opt/cuteafd-serve-probe" in line)
    assert lines[probe].endswith("serve-step5 --help ") and probe < lines.index(plan)


@pytest.mark.parametrize("model,gpus,sparks", [(STEP5_FP8, 2, ranks) for ranks in (4, 5, 6)] + [(STEP5_NVFP4, 1, 4)])
@pytest.mark.parametrize("value,expected", [(None, "fp8"), ("bf16", "bf16"), ("fp8", "fp8")])
def test_step5_cache_format_matches_plan_and_serve(tmp_path: Path, model, gpus, sparks, value, expected) -> None:
    keys = f"SPARK_COUNT={sparks}\nRTX_GPUS={gpus}\n" + _hosts(sparks)
    if value is not None:
        keys += f"KV_CACHE={value}\n"
    result = _step5(tmp_path, keys, model=model)
    assert _flag(_line(result, "cuteafd serve-step5 --snapshot"), "--kv-cache") == expected
    assert _flag(_line(result, "cuteafd plan"), "--kv-cache") == expected


@pytest.mark.parametrize("keys,diagnostic", [
    ("SPECULATOR=dflash2\n", "SPECULATOR=dflash2 does not apply to step5"),
    ("ATTENTION_PLACEMENT=heads\n", "step5 supports only ATTENTION_PLACEMENT=auto"),
    ("VISION=rtx\n", "step5 bundled media execution is not qualified"),
    ("SPARK_COUNT=0\n", "Step-5 has no local-expert layout"),
    ("EXPERT_BACKEND=local\nSPARK_COUNT=0\n", "Step-5 has no local-expert layout"),
    ("EMBEDDING=host\n", "step5 serves GPU token embeddings only"),
    ("RTX_EXPERT_LAYERS=4\n", "unused configuration key: RTX_EXPERT_LAYERS"),
    ("SPECULATOR_MODEL_ID=x/y\n", "unused configuration key: SPECULATOR_MODEL_ID"),
    ("SPECULATOR_MODEL_REVISION=abc\n", "unused configuration key: SPECULATOR_MODEL_REVISION"),
    ("SPECULATOR_DRAFTS=adaptive\n", "unused configuration key: SPECULATOR_DRAFTS"),
    ("KV_CACHE=int8\n", "KV_CACHE must be bf16 or fp8"),
])
def test_step5_refuses_unsupported_configuration(tmp_path: Path, keys: str, diagnostic: str) -> None:
    result = _step5(tmp_path, "SPARK_COUNT=6\nRTX_GPUS=2\n" + _hosts(6) + keys)
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert "docker run -d" not in result.stderr and "Traceback" not in result.stderr


@pytest.mark.parametrize("keys,depth", [
    ("", "3"),
    ("SPECULATOR=mtp\n", "3"),
    *[(f"SPECULATOR=mtp\nSPECULATOR_DEPTH={depth}\n", str(depth)) for depth in (1, 2, 3)],
    ("SPECULATOR=off\n", "0"),
])
def test_step5_native_mtp_defaults_on_with_explicit_off(tmp_path: Path, keys: str, depth: str) -> None:
    result = _step5(tmp_path, "SPARK_COUNT=6\nRTX_GPUS=2\n" + _hosts(6) + keys)
    serve = _line(result, "cuteafd serve-step5 --snapshot")
    assert _flag(serve, "--mtp") == depth
    assert _flag(_line(result, "cuteafd plan"), "--native-mtp-layers") == depth
    for absent in ("--draft", "--dspark"):
        assert absent not in serve.split(), absent
    other = _family_launch_result(tmp_path / "other", {"model_type": "deepseek_v4"},
                                  "test/dsv4", "SPECULATOR=dspark\n")
    assert _flag(_line(other, "cuteafd plan"), "--native-mtp-layers") is None


def test_step5_serves_from_release_images_and_refuses_slots_without_serve_step5(tmp_path: Path) -> None:
    keys = "SPARK_COUNT=6\nRTX_GPUS=2\n" + _hosts(6)
    release = _step5(tmp_path / "release", keys, wip=False)
    assert "under qualification" not in release.stderr
    assert "cuteafd serve-step5 --snapshot" in release.stderr
    old = _step5(tmp_path / "old", keys, probe_fails=True)
    assert old.returncode == 2
    assert "WIP slot s5 has no 'serve-step5' command; build the slot from a commit that serves step5" in old.stderr
    assert "cuteafd plan" not in old.stderr and "docker run -d" not in old.stderr


def test_family_table_and_planner_fixtures_name_step5() -> None:
    out = subprocess.run(["python3", str(ROOT / "scripts/lib/checkpoint-family.py"), "/dev/stdin"],
                         input=json.dumps(STEP5), capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["step5", "step5_vl", "3", "90"]
    fixtures = json.loads((ROOT / "rust/crates/cuteafd-loader/tests/fixtures/launch-families.json").read_text())
    assert "step5 step5_vl 3 90" in {case["line"] for case in fixtures}


@pytest.mark.parametrize("name,model,revision,gpus,sparks", [
    ("step5-fp8-max.config", STEP5_FP8, "364b3b03fed2477aa0953fe855a2c4c8c48ee3cc", "2", 6),
    ("step5-nvfp4-min.config", STEP5_NVFP4, "bc246cc2848e17b37f377eda8fd825c5b3018d1f", "1", 4),
])
def test_step5_example_configs(tmp_path: Path, name, model, revision, gpus, sparks) -> None:
    path = ROOT / "examples/configs" / name
    values = _keys(path)
    assert (values["MODEL_ID"], values["MODEL_REVISION"], values["RTX_GPUS"]) == (model, revision, gpus)
    assert values["SPARK_COUNT"] == str(sparks)
    assert (values["SPECULATOR"], values["VISION"], values["AUDIO"], values["POOL_TOKENS"]) == ("mtp", "off", "off", "auto")
    for rank in range(sparks):
        assert values[f"SPARK_{rank}_HOST"] == f"REPLACE-ME-spark-{rank}"
        assert values[f"SPARK_{rank}_LANE_A"] == f"192.0.2.{rank + 1}"
    assert f"SPARK_{sparks}_HOST" not in values
    text = path.read_text()
    assert not any(needle in text.lower() for needle in PRIVATE)
    assert f"./run.sh --config examples/configs/{name} --wip SLOT --dry-run" in text
    keys = "".join(f"{k}={v}\n" for k, v in values.items()
                   if not k.startswith("SPARK_") and k not in ("MODEL_ID", "MODEL_REVISION", "SPECULATOR"))
    result = _step5(tmp_path, keys + f"SPARK_COUNT={sparks}\n" + _hosts(sparks), model=model)
    assert "cuteafd serve-step5 --snapshot" in _line(result, "cuteafd serve-step5 --snapshot")


@pytest.mark.parametrize("name,gpus,sparks,package", [
    ("step5-fp8-max", [0, 1], ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"], "fp8/fp8-step5"),
    ("step5-nvfp4-min", [1], ["ostrich", "dodo", "emu", "kiwi"], "fp8/fp8-step5-nvfp4a4"),
])
def test_step5_gate_cards_list_and_generate_wip_launches(tmp_path: Path, name, gpus, sparks, package) -> None:
    card = cards.load_cards(ROOT / "scripts/bench/kits/gate", [name])[0]
    assert card["family"] == "step5" and card["gpus"] == gpus and card["sparks"] == sparks
    assert card["profile"] == "smoke"
    assert cards.slot_needs(card)["spark"] == [package]
    key = tmp_path / "key"
    key.write_text("k" * 64 + "\n")
    key.chmod(0o600)
    job = cards.generate(card, "cand", {"instance": "t", "slot": "s"}, tmp_path, key, {}, {"fabric"}, None, 1)
    values = cards.config(Path(job["entry"]["config"]))
    assert job["entry"]["run_args"] == ["--wip", "s"]
    assert (values["SPARK_COUNT"], values["RTX_GPUS"]) == (str(len(sparks)), str(len(gpus)))
    assert values["COORDINATOR_GPUS"] == ",".join(map(str, gpus))


@pytest.mark.parametrize("family", ["kimi", "step5"])
def test_qualifying_family_capacity_uses_cuda_total(tmp_path, family):
    total = 101_973_491_712
    kwargs = dict(cuda_total_bytes=total, gpu_total_mib=97887, gpu_free_mib=97887)
    result = (_kimi(tmp_path, **kwargs) if family == "kimi"
              else _step5(tmp_path, "SPARK_COUNT=6\nRTX_GPUS=2\n" + _hosts(6), **kwargs))
    plan = _line(result, "cuteafd plan")
    assert float(_flag(plan, "--coordinator-gpu-budget-gib")) == total / 2**30
    assert float(_flag(plan, "--coordinator-weight-budget-gib")) == total / 2**30


@pytest.mark.parametrize("ranks", (6, 7, 8))
def test_kimi_launches_every_covered_rank_count_from_planner(tmp_path, ranks):
    hosts = "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\nSPARK_{i}_LANE_B=\n"
                    for i in range(ranks))
    result = _kimi(tmp_path, f"SPARK_COUNT={ranks}\n" + hosts)
    serve = _line(result, "cuteafd serve-kimi --snapshot")
    assert _flag(serve, "--peers") == ",".join(f"10.0.0.{i+1}:19441" for i in range(ranks))
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert len(workers) == ranks
    assert all(f"--world {ranks} " in worker for worker in workers)


@pytest.mark.parametrize("ranks", (1, 2, 3, 4, 5))
def test_kimi_delegates_unservable_counts_to_planner(tmp_path, ranks):
    result = _kimi(tmp_path, f"SPARK_COUNT={ranks}\n",
                   encoder_plan={**PLANNED, "fits": False, "placement_supported": False})
    assert result.returncode != 0
    assert "invalid deployment specification" in result.stderr
    assert "cuteafd expertd-native" not in result.stderr


# --- Muse Spark (#923) -----------------------------------------------------------

MUSE = {"model_type": "muse_spark", "vision_config": {"depth": 2},
        "audio_config": {"model_type": "muse_spark_audio"},
        "text_config": {"model_type": "muse_spark_text", "num_hidden_layers": 62,
                        "mlp_layer_types": ["sparse"] * 62}}
MUSE_ID = "meta-models/Muse-Spark-Open-816B-A42B"


def _muse(tmp_path: Path, keys: str = "", *, wip: bool = True, **kwargs):
    return _family_launch_result(tmp_path, MUSE, MUSE_ID, "SPARK_COUNT=6\nRTX_GPUS=2\n" + _hosts(6) + keys,
                                 extra_args=("--wip", "muse") if wip else (),
                                 encoder_plan=kwargs.pop("encoder_plan", {**PLANNED, "spark_ranks": 6,
                                     "encoder_plan_hash": "ab" * 32, "encoder": None,
                                     "audio_encoder": {"kind": {"kind": "off"}, "replicas": []}}), **kwargs)


def test_muse_dispatches_to_serve_muse_with_its_own_defaults(tmp_path: Path) -> None:
    result = _muse(tmp_path)
    serve = _line(result, "cuteafd serve-muse --snapshot")
    assert f"/hf/hub/models--{MUSE_ID.replace('/', '--')}/snapshots/abc" in serve
    assert _flag(serve, "--peers") == ",".join(f"10.0.0.{i + 1}:19441" for i in range(6))
    # The head-split pair every family's launcher passes; serve-muse defaults to one GPU.
    assert (_flag(serve, "--device"), _flag(serve, "--split-device")) == ("0", "1")
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    assert _flag(serve, "--prefix-cache-entries") == "20"
    assert _flag(serve, "--vision") == _flag(serve, "--audio") == "off"
    for absent in ("--max-context", "--pool-tokens", "--draft", "--mtp", "--kv-cache",
                   "--prefill-packing", "--expert-partial-bf16-max-rows", "--expert-partial-wide-format"):
        assert absent not in serve.split(), absent
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert len(workers) == 6 and all("--world 6 " in worker for worker in workers)
    plan = _line(result, "cuteafd plan")
    assert _flag(plan, "--speculator") == "off" and "--drafter-snapshot" not in plan.split()
    assert "--vision auto --audio auto" in plan
    lines = result.stderr.splitlines()
    probe = next(i for i, line in enumerate(lines) if "dst=/opt/cuteafd-serve-probe" in line)
    assert lines[probe].endswith("serve-muse --help ")


@pytest.mark.parametrize("vision,audio", [("spark", "spark"), ("rtx", "spark"), ("off", "spark"), ("spark", "rtx")])
def test_muse_generic_media_plan_handoff(tmp_path: Path, vision: str, audio: str) -> None:
    def placed(kind, index):
        return {"kind": {"kind": kind, **({"rank": index} if kind == "spark" else {"gpu": index})}, "replicas": []}
    plan = {**PLANNED, "spark_ranks": 6, "encoder_plan_hash": "ab" * 32,
            "encoder": None if vision == "off" else placed(vision, 0),
            "audio_encoder": placed(audio, 1)}
    result = _muse(tmp_path, f"VISION={'off' if vision == 'off' else 'auto'}\nAUDIO=auto\n", encoder_plan=plan)
    serve = _line(result, "cuteafd serve-muse --snapshot")
    assert _flag(serve, "--vision") == ("off" if vision == "off" else f"{vision}:0")
    assert _flag(serve, "--audio") == f"{audio}:1"
    assert "--audio-encoder-backend" not in serve.split()
    if vision == "spark":
        assert _flag(serve, "--vision-peers") == "10.0.0.1:19442"
        assert _flag(serve, "--encoder-plan-hash") == "ab" * 32
        assert _flag(serve, "--encoder-revision") == "abc"
    if audio == "spark":
        assert _flag(serve, "--audio-peers") == "10.0.0.2:19443"
        assert _flag(serve, "--audio-encoder-plan-hash") == "ab" * 32
        assert _flag(serve, "--audio-encoder-revision") == "abc"
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert sum("--encoder " in worker for worker in workers) == (vision == "spark")
    assert sum("--audio-encoder " in worker for worker in workers) == (audio == "spark")


def test_muse_forwards_typed_keys_when_set(tmp_path: Path) -> None:
    result = _muse(tmp_path, "RTX_EXPERT_LAYERS=3\nPOOL_TOKENS=1048576\nMAX_CONTEXT_TOKENS=262144\n"
                             "CONCURRENCY=8\nEXPERT_PARTIAL_BF16_MAX_ROWS=128\nEXPERT_PARTIAL_WIDE_FORMAT=fp8\n"
                             "SPARK_DEVICE_BUDGET_BYTES=107374182400\n")
    serve, plan = _line(result, "cuteafd serve-muse --snapshot"), _line(result, "cuteafd plan")
    # The planner admits the same partial-reply overrides the server runs with.
    assert _flag(plan, "--expert-partial-bf16-max-rows") == "128"
    assert _flag(plan, "--expert-partial-wide-format") == "fp8"
    assert _flag(serve, "--rtx-expert-layers") == "3"
    assert _flag(serve, "--pool-tokens") == "1048576"
    assert _flag(serve, "--max-context") == "262144"
    assert _flag(serve, "--max-sequences") == "8"
    assert _flag(serve, "--expert-partial-bf16-max-rows") == "128"
    assert _flag(serve, "--expert-partial-wide-format") == "fp8"
    assert _flag(serve, "--spark-budget-gib") == "100.0"


@pytest.mark.parametrize("drafter,revision", [
    ("RadixArk/Muse-Spark-Open-DFlash2", "46c90f41a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0"),
    ("nvidia/Muse-Spark-Open-816B-A42B-DFlash2", "22e2669ea0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0"),
])
@pytest.mark.parametrize("cache", [None, "bf16"])
def test_muse_dflash2_resolves_one_snapshot_for_plan_and_serve(tmp_path: Path, drafter, revision, cache) -> None:
    _snapshot(tmp_path / "hf", drafter, {}, revision=revision)
    keys = (f"SPECULATOR=dflash2\nSPECULATOR_MODEL_ID={drafter}\nSPECULATOR_MODEL_REVISION={revision}\n"
            "SPECULATOR_DRAFTS=5\n" + ("" if cache is None else f"DRAFT_KV_CACHE={cache}\n"))
    result = _muse(tmp_path, keys)
    serve, plan = _line(result, "cuteafd serve-muse --snapshot"), _line(result, "cuteafd plan")
    snapshot = str(tmp_path / f"hf/hub/models--{drafter.replace('/', '--')}/snapshots/{revision}")
    assert _flag(serve, "--draft") == _flag(plan, "--drafter-snapshot") == snapshot
    assert _flag(plan, "--speculator") == "dflash2"
    assert _flag(serve, "--draft-kv-cache") == _flag(plan, "--draft-kv-cache") == (cache or "fp8")
    assert _flag(serve, "--draft-fixed") == "5"
    assert "--draft-fp8" not in serve.split()


@pytest.mark.parametrize("keys,diagnostic", [
    ("ATTENTION_PLACEMENT=layers\n", "supports only ATTENTION_PLACEMENT=auto"),
    ("SPECULATOR=mtp\n", "SPECULATOR=mtp does not apply to muse_spark"),
    ("PREFILL_PACKING=unrestricted\n", "Muse Spark has no packed prefill yet (#923)"),
    ("RTX_EXPERT_LAYERS=max\n", "Muse Spark RTX_EXPERT_LAYERS is auto or a fixed routed-layer count"),
    ("SPECULATOR=dflash2\nSPECULATOR_MODEL_ID=RadixArk/Muse-Spark-Open-DFlash2\nDRAFT_KV_CACHE=fp16\n",
     "DRAFT_KV_CACHE must be bf16 or fp8"),
])
def test_muse_refuses_unsupported_configuration(tmp_path: Path, keys: str, diagnostic: str) -> None:
    if "SPECULATOR_MODEL_ID=" in keys:
        _snapshot(tmp_path / "hf", "RadixArk/Muse-Spark-Open-DFlash2", {})
    result = _muse(tmp_path, keys)
    assert result.returncode != 0
    assert diagnostic in result.stderr, result.stderr[-2000:]
    assert "docker run -d" not in result.stderr
    assert "Traceback" not in result.stderr


def test_muse_refuses_a_slot_without_serve_muse(tmp_path: Path) -> None:
    result = _muse(tmp_path, probe_fails=True)
    assert result.returncode == 2, result.stderr
    assert "WIP slot muse has no 'serve-muse' command; build the slot from a commit that serves muse_spark" in result.stderr
    assert "cuteafd plan" not in result.stderr and "docker run -d" not in result.stderr


def test_family_table_and_planner_fixtures_name_muse() -> None:
    out = subprocess.run(["python3", str(ROOT / "scripts/lib/checkpoint-family.py"), "/dev/stdin"],
                         input=json.dumps(MUSE), capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["muse_spark", "muse_spark", "0", "61"]
    fixtures = json.loads((ROOT / "rust/crates/cuteafd-loader/tests/fixtures/launch-families.json").read_text())
    assert "muse_spark muse_spark 0 61" in {case["line"] for case in fixtures}
    assert '"muse_spark" =>' in (ROOT / "rust/crates/cuteafd-loader/src/plan/launch.rs").read_text()
