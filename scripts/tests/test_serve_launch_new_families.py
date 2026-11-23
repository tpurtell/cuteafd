"""Kimi K3 and Step-5 launch through the shared launcher (run.sh -> run-family.sh), the
example configs and the gate kit, from WIP slots only.

CPU only: docker, ssh, nest and nvidia-smi are stubs that print their argv
(test_run_family_dispatch._family_launch_result).
"""
from __future__ import annotations

import importlib.util
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from test_run_family_dispatch import _coordinator_gpus, _family_launch_result, _snapshot

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
PLANNED = {"placement_supported": True, "fits": True, "spark_budget_bytes": 108 << 30, "encoder": None, "encoder_plan_hash": "ab" * 32}


def _kimi(tmp_path: Path, keys: str = "", *, wip: bool = True, **kwargs):
    return _family_launch_result(tmp_path, KIMI, KIMI_ID, KIMI_KEYS + keys,
                                 extra_args=("--wip", "k3") if wip else (),
                                 encoder_plan=kwargs.pop("encoder_plan", PLANNED), **kwargs)


def _line(result, needle: str) -> str:
    assert result.returncode == 0, result.stderr
    return next(line for line in result.stderr.splitlines() if needle in line and "--resolve-settings-only" not in line)


def _flag(line: str, flag: str) -> str | None:
    """FLAG's value in the cuteafd command of a docker run line (not docker's own options)."""
    tokens = line.split()
    if "cuteafd" in tokens:
        tokens = tokens[tokens.index("cuteafd"):]
    return tokens[tokens.index(flag) + 1] if flag in tokens else None


def test_hy4_flash_launcher_composes_0spark_and_spark_argv(tmp_path: Path) -> None:
    config = {
        "model_type": "hy_v4_flash",
        "num_hidden_layers": 42,
        "mlp_layer_types": ["dense"] + ["sparse"] * 41,
    }
    # 0-Spark 2x RTX max (original)
    keys_max = "SPARK_COUNT=0\nRTX_GPUS=2\nEXPERT_BACKEND=local\nSPECULATOR=off\nVISION=off\nAUDIO=off\nCORE_VIEW=original\n"
    res_max = _family_launch_result(
        tmp_path / "max", config, "wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC", keys_max,
        physical_gpus=(0, 1), extra_args=("--wip", "hy4f"), encoder_plan={**PLANNED, "spark_ranks": 0},
        deployment_overrides={"prefill_unit_rows": 2048, "prefill_waves": 1},
    )
    serve_max = _line(res_max, "serve-hy4f --snapshot")
    assert _flag(serve_max, "--prefill-packing") is None and _flag(serve_max, "--prefix-cache-mark-mib") is None
    marked = _family_launch_result(
        tmp_path / "marked", config, "wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC",
        keys_max + "PREFILL_PACKING=unrestricted\nPREFIX_CACHE_MARK_MIB=1024\nDECODE_ROWS=32\n",
        physical_gpus=(0, 1), extra_args=("--wip", "hy4f"), encoder_plan={**PLANNED, "spark_ranks": 0},
        deployment_overrides={"prefill_unit_rows": 2048, "prefill_waves": 1, "decode_rows": 32},
    )
    marked_serve, marked_plan = _line(marked, "serve-hy4f --snapshot"), _line(marked, "cuteafd plan")
    assert _flag(marked_serve, "--prefill-packing") == "unrestricted"
    assert _flag(marked_serve, "--prefix-cache-mark-mib") == _flag(marked_plan, "--prefix-cache-mark-mib") == "1024"
    assert _flag(marked_serve, "--decode-rows") == _flag(marked_plan, "--decode-rows") == "32"
    assert (_flag(serve_max, "--device"), _flag(serve_max, "--split-device")) == ("0", "1")
    assert _flag(serve_max, "--core-view") == "original"
    assert _flag(serve_max, "--kv-cache") == "fp8"
    assert _flag(serve_max, "--rtx-expert-layers") == "auto"
    assert "--peers" not in shlex.split(serve_max)
    # 1x RTX + 4 Sparks k6-compact
    four = "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\n" for i in range(4))
    keys_s4 = "SPARK_COUNT=4\nRTX_GPUS=1\nEXPERT_BACKEND=spark\nSPECULATOR=off\nVISION=off\nAUDIO=off\nCORE_VIEW=k6-compact\n" + four
    res_s4 = _family_launch_result(
        tmp_path / "s4", config, "wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC", keys_s4,
        physical_gpus=(0,), extra_args=("--wip", "hy4f"), encoder_plan={**PLANNED, "spark_ranks": 4},
        deployment_overrides={"prefill_unit_rows": 2048, "prefill_waves": 1},
    )
    serve_s4 = _line(res_s4, "serve-hy4f --snapshot")
    assert _flag(serve_s4, "--core-view") == "k6-compact"
    assert _flag(serve_s4, "--peers") == ",".join(f"10.0.0.{i + 1}:19441" for i in range(4))
    assert "--split-device" not in shlex.split(serve_s4)


def test_inkling_native_launcher_composes_single_and_paired_argv(tmp_path: Path) -> None:
    config = {"model_type": "inkling2", "text_config": {"num_hidden_layers": 48}}
    keys = "SPARK_COUNT=6\nRTX_GPUS=1\nSPECULATOR=off\nVISION=off\nAUDIO=off\n" + SIX
    result = _family_launch_result(tmp_path / "whole", config, "thinkingmachines/Inkling-2-Small-MXFP8", keys,
        physical_gpus=(0,), extra_args=("--wip", "inkling"), encoder_plan=PLANNED,
        deployment_overrides={"prefill_unit_rows": 2752, "prefill_waves": 3})
    serve = _line(result, "serve-inkling2 --snapshot")
    assert _flag(serve, "--peers") == ",".join(f"10.0.0.{i + 1}:19441" for i in range(6))
    assert _flag(serve, "--max-context-tokens") == "auto"
    assert _flag(serve, "--core-view") == "original"
    assert (_flag(serve, "--prefill-rows"), _flag(serve, "--prefill-waves")) == ("2752", "3")
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    assert "--dspark=false" in shlex.split(serve)
    for flag in ("--split-device", "--attention-placement", "--dspark", "--prefill-packing"):
        assert flag not in shlex.split(serve)
    pair = _family_launch_result(tmp_path / "pair", config, "thinkingmachines/Inkling-2-Small-MXFP8",
        keys.replace("RTX_GPUS=1", "RTX_GPUS=2").replace("SPECULATOR=off", "SPECULATOR=dspark"),
        physical_gpus=(0, 1), extra_args=("--wip", "inkling"), encoder_plan=PLANNED)
    pair_serve = _line(pair, "serve-inkling2 --snapshot")
    pair_plan = _line(pair, "cuteafd plan")
    assert (_flag(pair_serve, "--device"), _flag(pair_serve, "--split-device")) == ("0", "1")
    assert _flag(pair_plan, "--rtx") == "2"
    assert _flag(pair_serve, "--peers") == _flag(serve, "--peers")
    assert _flag(pair_plan, "--speculator") == "dspark"
    assert "--dspark" in shlex.split(pair_serve)


def test_inkling_packing_and_mark_budget_reach_plan_and_serve(tmp_path: Path) -> None:
    config = {"model_type": "inkling2", "text_config": {"num_hidden_layers": 48}}
    keys = ("SPARK_COUNT=6\nRTX_GPUS=1\nSPECULATOR=off\nVISION=off\nAUDIO=off\n"
            "PREFILL_PACKING=unrestricted\nPREFIX_CACHE_MARK_MIB=1024\n" + SIX)
    result = _family_launch_result(tmp_path, config, "thinkingmachines/Inkling-2-Small-MXFP8", keys,
        physical_gpus=(0,), extra_args=("--wip", "inkling"), encoder_plan=PLANNED)
    serve, plan = _line(result, "serve-inkling2 --snapshot"), _line(result, "cuteafd plan")
    assert _flag(serve, "--prefill-packing") == "unrestricted"
    assert _flag(serve, "--prefix-cache-mark-mib") == _flag(plan, "--prefix-cache-mark-mib") == "1024"


@pytest.mark.parametrize("setting, enabled", [("", True), ("auto", True), ("off", False), ("dspark", True)])
def test_inkling_bundled_drafter_defaults_on_and_honors_off(tmp_path: Path, setting: str, enabled: bool) -> None:
    config = {"model_type": "inkling2", "text_config": {"num_hidden_layers": 48}}
    keys = "SPARK_COUNT=6\nRTX_GPUS=1\nVISION=off\nAUDIO=off\n" + SIX
    if setting:
        keys += f"SPECULATOR={setting}\n"
    result = _family_launch_result(tmp_path, config, "thinkingmachines/Inkling-2-Small-MXFP8", keys,
        physical_gpus=(0,), extra_args=("--wip", "inkling"), encoder_plan=PLANNED)
    serve = _line(result, "serve-inkling2 --snapshot")
    plan = _line(result, "cuteafd plan")
    assert ("--dspark" in shlex.split(serve)) == enabled
    assert ("--dspark=false" in shlex.split(serve)) == (not enabled)
    assert _flag(plan, "--speculator") == ("dspark" if enabled else "off")


@pytest.mark.parametrize("cache", ["fp8", "bf16"])
def test_inkling_bundled_drafter_matches_plan_and_serve(tmp_path: Path, cache: str) -> None:
    config = {"model_type": "inkling2", "text_config": {"num_hidden_layers": 48}}
    keys = (f"SPARK_COUNT=6\nRTX_GPUS=1\nSPECULATOR=dspark\nDRAFT_KV_CACHE={cache}\n"
            "VISION=off\nAUDIO=off\n" + SIX)
    result = _family_launch_result(tmp_path, config, "thinkingmachines/Inkling-2-Small-MXFP8", keys,
        physical_gpus=(0,), extra_args=("--wip", "inkling"), encoder_plan=PLANNED)
    serve = _line(result, "serve-inkling2 --snapshot")
    plan = _line(result, "cuteafd plan")
    assert "--dspark" in shlex.split(serve)
    assert _flag(plan, "--speculator") == "dspark"
    assert _flag(serve, "--draft-kv-cache") == _flag(plan, "--draft-kv-cache") == cache
    assert "--draft-fp8" not in shlex.split(serve)


@pytest.mark.parametrize("setting, message", [
    ("DRAFT_KV_CACHE=fp16", "DRAFT_KV_CACHE must be bf16 or fp8"),
    ("SPECULATOR_FP8=on", "Inkling SPECULATOR_FP8 must be auto or off"),
    ("SPECULATOR=dflash2", "SPECULATOR=dflash2 does not apply to inkling2"),
])
def test_inkling_drafter_rejects_unsupported_modes(tmp_path: Path, setting: str, message: str) -> None:
    config = {"model_type": "inkling2", "text_config": {"num_hidden_layers": 48}}
    keys = "SPARK_COUNT=6\nRTX_GPUS=1\nSPECULATOR=dspark\nVISION=off\nAUDIO=off\n" + SIX + setting + "\n"
    result = _family_launch_result(tmp_path, config, "thinkingmachines/Inkling-2-Small-MXFP8", keys,
        physical_gpus=(0,), extra_args=("--wip", "inkling"), encoder_plan=PLANNED)
    assert result.returncode == 2 and message in result.stderr


def test_minimax_qualified_card_opens_one_rtx_three_sparks_public_launch(tmp_path: Path) -> None:
    config = {"model_type": "minimax_m3_vl", "text_config": {
        "num_hidden_layers": 60, "moe_layer_freq": [0] * 3 + [1] * 57}}
    three = "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\n" for i in range(3))
    keys = "SPARK_COUNT=3\nRTX_GPUS=1\nSPECULATOR=off\nVISION=off\nAUDIO=off\n" + three
    result = _family_launch_result(tmp_path / "wip", config, "MiniMaxAI/MiniMax-M3.1-Flash", keys,
        extra_args=("--wip", "minimax"), encoder_plan=PLANNED)
    serve = _line(result, "serve-minimax --snapshot")
    assert _flag(serve, "--peers") == ",".join(f"10.0.0.{i + 1}:19441" for i in range(3))
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    assert _flag(serve, "--split-wire") == "fp32"
    assert _flag(serve, "--host-cache-bytes") == "auto"
    assert "--split-device" not in serve.split()
    release = _family_launch_result(tmp_path / "release", config, "MiniMaxAI/MiniMax-M3.1-Flash", keys,
        encoder_plan=PLANNED)
    assert release.returncode == 0
    assert "--split-device" not in _line(release, "serve-minimax --snapshot").split()
    pair = _family_launch_result(tmp_path / "pair", config, "MiniMaxAI/MiniMax-M3.1-Flash",
        keys.replace("RTX_GPUS=1", "RTX_GPUS=2"), encoder_plan=PLANNED)
    assert pair.returncode == 0
    assert _flag(_line(pair, "serve-minimax --snapshot"), "--split-device") == "1"
    matrix = json.loads((ROOT / "scripts/bench/kits/gate/matrix-gate.json").read_text())
    entry = next(entry for entry in matrix["entries"] if entry["name"] == "minimax-flash-min")
    assert entry["family"] == "minimax_m3" and len(entry["gpus"]) == 1 and len(entry["sparks"]) == 3
    assert entry["config"] == "examples/configs/minimax-m31-flash-nvfp4-min.config"


def test_minimax_forwards_core_view_and_still_media_keys(tmp_path: Path) -> None:
    config = {"model_type": "minimax_m3_vl", "vision_config": {"depth": 2}, "text_config": {
        "num_hidden_layers": 60, "moe_layer_freq": [0] * 3 + [1] * 57}}
    three = "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\n" for i in range(3))
    keys = ("SPARK_COUNT=3\nRTX_GPUS=1\nSPECULATOR=off\nVISION=auto\nAUDIO=off\nCORE_VIEW=original\n"
            "VISION_REPLICAS=2\nMEDIA_CACHE_BYTES=2GiB\n" + three)
    result = _family_launch_result(tmp_path, config, "MiniMaxAI/MiniMax-M3.1-Flash", keys,
        extra_args=("--wip", "minimax"), encoder_plan=PLANNED)
    serve, plan = _line(result, "serve-minimax --snapshot"), _line(result, "cuteafd plan")
    assert _flag(serve, "--core-view") == _flag(plan, "--core-view") == "original"
    assert _flag(serve, "--vision-replicas") == _flag(plan, "--vision-replicas") == "2"
    assert _flag(serve, "--media-cache-bytes") == _flag(plan, "--media-cache-bytes") == "2GiB"
    marked = _family_launch_result(tmp_path / "marked", config, "MiniMaxAI/MiniMax-M3.1-Flash",
        keys + "PREFIX_CACHE_MARK_MIB=1024\n", extra_args=("--wip", "minimax"), encoder_plan=PLANNED)
    assert _flag(_line(marked, "serve-minimax --snapshot"), "--prefix-cache-mark-mib") == \
        _flag(_line(marked, "cuteafd plan"), "--prefix-cache-mark-mib") == "1024"


@pytest.mark.parametrize("gpus,placement", [(1, "auto"), (2, "layers"), (2, "heads")])
@pytest.mark.parametrize("selection", [None, "auto", "dspark", "off"])
def test_minimax_dspark_default_and_explicit_off_match_plan_and_serve(tmp_path, gpus, placement, selection):
    config = {"model_type": "minimax_m3_vl", "text_config": {
        "num_hidden_layers": 60, "moe_layer_freq": [0] * 3 + [1] * 57}}
    ranks = 3 if gpus == 1 else 4
    hosts = "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\n" for i in range(ranks))
    keys = (f"SPARK_COUNT={ranks}\nRTX_GPUS={gpus}\nATTENTION_PLACEMENT={placement}\n"
            "VISION=off\nAUDIO=off\n") + hosts
    if selection is not None:
        keys += f"SPECULATOR={selection}\n"
    result = _family_launch_result(tmp_path, config, "MiniMaxAI/MiniMax-M3.1-Flash", keys,
        encoder_plan=PLANNED)
    serve = _line(result, "serve-minimax --snapshot")
    plan = _line(result, "cuteafd plan ")
    enabled = selection != "off"
    assert ("--dspark" in shlex.split(serve)) == enabled
    assert ("--dspark=false" in shlex.split(serve)) == (not enabled)
    assert _flag(plan, "--speculator") == ("dspark" if enabled else "off")
    if gpus == 2:
        assert _flag(serve, "--split-device") == "1"


@pytest.mark.parametrize("mode", [None, "history", "clamped", "off"])
def test_minimax_passes_the_shared_draft_policy(tmp_path, mode):
    config = {"model_type": "minimax_m3_vl", "text_config": {
        "num_hidden_layers": 60, "moe_layer_freq": [0] * 3 + [1] * 57}}
    hosts = "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\n" for i in range(3))
    result = _family_launch_result(tmp_path, config, "MiniMaxAI/MiniMax-M3.1-Flash",
        "SPARK_COUNT=3\nRTX_GPUS=1\nVISION=off\nAUDIO=off\n" + (f"DRAFT_POLICY={mode}\n" if mode else "") + hosts,
        encoder_plan=PLANNED)
    serve = _line(result, "serve-minimax --snapshot")
    assert ("--draft-policy off" in serve) == (mode == "off")
    assert ("--draft-policy clamped" in serve) == (mode == "clamped")
    assert "CUTEAFD_DRAFT_POLICY" not in serve


@pytest.mark.parametrize("filter", ["", "info,cuteafd::shared::mapped_table=debug"])
def test_generic_log_filter_reaches_coordinator_and_workers(tmp_path: Path, filter: str) -> None:
    result = _kimi(tmp_path, extra_env={"RUST_LOG": filter})
    serve = _line(result, "serve-kimi --snapshot")
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    expected = f"RUST_LOG={filter or 'info'}"
    assert expected in serve.split()
    assert len(workers) == 6
    assert all(expected in shlex.split(worker) for worker in workers)


def test_kimi_dispatches_to_serve_kimi_with_its_own_defaults(tmp_path: Path) -> None:
    result = _kimi(tmp_path)
    serve = _line(result, "serve-kimi --snapshot")
    assert str(tmp_path / "hf/hub/models--wrldsuksgo2mars--Kimi-K3-EXL3-K2/snapshots/abc") in serve
    assert _flag(serve, "--peers") == ",".join(f"10.0.0.{i + 1}:19441" for i in range(6))
    assert (_flag(serve, "--device"), _flag(serve, "--split-device")) == ("0", "1")
    assert _flag(serve, "--max-sequences") == "16"
    assert _flag(serve, "--embedding-placement") == "gpu"
    assert _flag(serve, "--prefix-cache-entries") == "20"
    # Automatic context follows the model maximum and admitted KV pool for every core.
    # The planner picks onboarded layers; no speculator or media options are passed.
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    assert _flag(serve, "--max-context-tokens") == "auto"
    assert _flag(serve,"--vision")=="off"  # Fixture admits no tower; Auto falls back explicitly.
    for absent in ("--pool-tokens",
                   "--draft", "--draft-fp8", "--mtp", "--dspark", "--local-experts"):
        assert absent not in serve.split(), absent
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert len(workers) == 6
    # No hard-coded Spark budget: the planner picks the family's and the workers get it.
    plan = _line(result, "cuteafd plan")
    assert "--spark-memory-cap" not in plan.split()
    assert all(f"--device-budget-bytes {108 << 30} " in worker for worker in workers)
    assert "--world 6" in workers[0] and "--vision auto --audio off" in plan
    assert _flag(plan, "--spark-ranks") == "6" and _flag(plan, "--rtx") == "2"
    assert _flag(plan, "--speculator") == "off" and "--require-ready" not in plan.split()
    assert "--drafter-snapshot" not in plan.split() and "--draft-fp8" not in plan.split()
    # The slot's binary is asked for serve-kimi before anything is planned.
    lines = result.stderr.splitlines()
    probe = next(i for i, line in enumerate(lines) if "dst=/opt/cuteafd-serve-probe" in line)
    assert lines[probe].endswith("serve-kimi --help ")
    assert probe < lines.index(plan)


@pytest.mark.parametrize("gpus,split", [(1, "auto"), (1, "off"), (2, "auto"), (2, "off")])
def test_kimi_launch_uses_only_actual_physical_owners(tmp_path: Path, gpus: int, split: str) -> None:
    result = _kimi(tmp_path, f"RTX_GPUS={gpus}\nCOORDINATOR_SPLIT={split}\n",
                   physical_gpus=tuple(range(gpus)))
    serve = _line(result, "serve-kimi --snapshot")
    owners = gpus if split != "off" else 1
    assert _coordinator_gpus(serve) == list(range(owners))
    assert _flag(serve, "--split-device") == ("1" if owners == 2 else None)
    if owners == 2:
        assert _flag(serve, "--device") == "0"
    assert _flag(_line(result, "cuteafd plan"), "--rtx") == str(owners)


def test_kimi_forwards_typed_keys_when_set(tmp_path: Path) -> None:
    serve = _line(_kimi(tmp_path, "RTX_EXPERT_LAYERS=5\nPOOL_TOKENS=1048576\nMAX_CONTEXT_TOKENS=262144\n"
                                  "CONCURRENCY=8\nSPARK_MEMORY_CAP=107374182400\n"),
                  "serve-kimi --snapshot")
    assert _flag(serve, "--rtx-expert-layers") == "5"
    assert _flag(serve, "--pool-tokens") == "1048576"
    assert _flag(serve, "--max-context-tokens") == "262144"
    assert _flag(serve, "--max-sequences") == "8"


@pytest.mark.parametrize("placement", ["auto", "heads", "context", "layers"])
def test_kimi_attention_placement_reaches_plan_and_serve(tmp_path: Path, placement: str) -> None:
    result = _kimi(tmp_path, f"ATTENTION_PLACEMENT={placement}\n")
    serve = _line(result, "serve-kimi --snapshot")
    plan = _line(result, "cuteafd plan")
    assert _flag(plan, "--attention-placement") == placement
    assert f"CUTEAFD_ATTENTION_PLACEMENT={placement}" in serve


def test_kimi_forwards_all_onboard_pin(tmp_path: Path) -> None:
    serve = _line(_kimi(tmp_path, "RTX_EXPERT_LAYERS=all\n"), "serve-kimi --snapshot")
    assert _flag(serve, "--rtx-expert-layers") == "all"


@pytest.mark.parametrize("enabled", [False, True])
def test_kimi_dflash2_uses_one_resolved_fp8_snapshot_for_plan_and_serve(tmp_path: Path, enabled: bool) -> None:
    model, revision = "lightseekorg/kimi-k3-dflash2", "e77935fb4804e17eb55085bffd045eae1d779769"
    _snapshot(tmp_path / "hf", model, {}, revision=revision)
    result = _kimi(tmp_path, f"SPECULATOR={'dflash2' if enabled else 'off'}\n"
                            f"SPECULATOR_MODEL_ID={model}\nSPECULATOR_MODEL_REVISION={revision}\n"
                            + ("DRAFT_LENGTH=7\n" if enabled else ""))
    serve, plan = _line(result, "serve-kimi --snapshot"), _line(result, "cuteafd plan")
    assert _flag(plan, "--speculator") == ("dflash2" if enabled else "off")
    if enabled:
        snapshot = str(tmp_path / f"hf/hub/models--{model.replace('/', '--')}/snapshots/{revision}")
        assert _flag(serve, "--draft") == _flag(plan, "--drafter-snapshot") == snapshot
        assert _flag(serve, "--draft-fp8") == "true"
        assert "--draft-fp8" not in plan.split(), "Rust resolves Kimi FP8 from DFlash selection"
        assert _flag(serve, "--draft-length") == "7"
    else:
        assert all(flag not in serve.split() for flag in ("--draft", "--draft-fp8", "--draft-length"))
        assert all(flag not in plan.split() for flag in ("--drafter-snapshot", "--draft-fp8"))


@pytest.mark.parametrize("cache", ["bf16", "fp8"])
def test_kimi_dspark_uses_resolved_native_weights_and_independent_draft_kv(tmp_path: Path, cache) -> None:
    model, revision = "Inferact/Kimi-K3.1-Pro-DSpark", "c197a09072916cf63172df2d9d4baef0b77ed83a"
    _snapshot(tmp_path / "hf", model, {}, revision=revision)
    result = _kimi(tmp_path, f"SPECULATOR=dspark\nDRAFT_KV_CACHE={cache}\n")
    serve = _line(result,"serve-kimi --snapshot")
    plan = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line and "--deployment" in line)
    assert _flag(serve,"--draft") == _flag(plan,"--drafter-snapshot")
    assert _flag(serve,"--draft-fp8") == "false"
    assert _flag(serve,"--draft-kv-cache") == _flag(plan,"--draft-kv-cache") == cache
    assert _flag(plan,"--speculator") == "dspark"
    assert _flag(serve,"--tp2-dflash") != "true"


@pytest.mark.parametrize("selection,expected", [(None, "true"), ("off", "false"), ("on", "true")])
def test_kimi_tp2_dflash_plan_matches_serve_and_keeps_explicit_false(tmp_path: Path, selection, expected) -> None:
    model, revision = "lightseekorg/kimi-k3-dflash2", "e77935fb4804e17eb55085bffd045eae1d779769"
    _snapshot(tmp_path / "hf", model, {}, revision=revision)
    keys = f"SPECULATOR=dflash2\nSPECULATOR_MODEL_ID={model}\nSPECULATOR_MODEL_REVISION={revision}\n"
    if selection is not None:
        keys += f"TP2_DFLASH={selection}\n"
    result = _kimi(tmp_path, keys)
    assert _flag(_line(result, "serve-kimi --snapshot"), "--tp2-dflash") == expected
    assert _flag(_line(result, "cuteafd plan"), "--tp2-dflash") == (expected if selection is not None else None)


def test_kimi_tp2_dflash_refuses_invalid_selection(tmp_path: Path) -> None:
    result = _kimi(tmp_path, "TP2_DFLASH=auto\n")
    assert result.returncode != 0
    assert "TP2_DFLASH must be on or off" in result.stderr


@pytest.mark.parametrize("cache", [None, "bf16", "fp8"])
def test_kimi_kv_format_reaches_both_plan_and_target_drafter(tmp_path: Path, cache) -> None:
    result = _kimi(tmp_path, "" if cache is None else f"KV_CACHE={cache}\n")
    expected = cache or "fp8"
    assert _flag(_line(result, "serve-kimi --snapshot"), "--kv-cache") == expected
    assert _flag(_line(result, "cuteafd plan"), "--kv-cache") == expected


def test_kimi_explicit_spark_budget_reaches_plan_and_workers(tmp_path: Path) -> None:
    result = _kimi(tmp_path, "SPARK_MEMORY_CAP=107374182400\n")
    assert _flag(_line(result, "cuteafd plan"), "--spark-memory-cap") == "107374182400"
    assert "--device-budget-bytes 107374182400 " in _line(result, "expertd-native")
    # A global flag: the worker's own admission follows the same Spark cap.
    assert _flag(_line(result, "expertd-native"), "--spark-memory-cap") == "107374182400"
    assert _flag(_line(result, "serve-kimi --snapshot"), "--spark-memory-cap") == "107374182400"


def test_kimi_unset_spark_budget_leaves_serve_on_the_family_default(tmp_path: Path) -> None:
    assert "--spark-memory-cap" not in _line(_kimi(tmp_path, ""), "serve-kimi --snapshot")


def test_kimi_auto_places_bundled_tower_on_admitted_spark(tmp_path):
    plan={**PLANNED,"encoder":{"kind":{"kind":"spark","rank":1},"replicas":[],
        "weights":894718080,"scratch":1538887936},"max_image_tokens":16698}
    result=_kimi(tmp_path,encoder_plan=plan)
    serve=_line(result,"serve-kimi --snapshot")
    assert _flag(serve,"--vision")=="spark:1"
    assert _flag(serve,"--vision-peers")=="10.0.0.2:19442"
    assert "--encoder-max-tokens 16698" in result.stderr


def test_kimi_forwards_admitted_vision_replicas_and_media_cache(tmp_path):
    plan={**PLANNED,"encoder":{"kind":{"kind":"spark","rank":0},"replicas":[1],
        "weights":894718080,"scratch":1538887936},"max_image_tokens":16698}
    result=_kimi(tmp_path,"VISION_REPLICAS=2\nMEDIA_CACHE_BYTES=1GiB\n",encoder_plan=plan)
    serve=_line(result,"serve-kimi --snapshot")
    assert _flag(serve,"--vision-replicas")=="2"
    assert _flag(serve,"--media-cache-bytes")=="1GiB"
    assert _flag(serve,"--vision-peers")=="10.0.0.1:19442,10.0.0.2:19442"


@pytest.mark.parametrize("keys,diagnostic", [
    ("SPECULATOR=dflash2\n", "SPECULATOR=dflash2 needs SPECULATOR_MODEL_ID"),
    ("SPECULATOR=mtp\n", "SPECULATOR=mtp does not apply to kimi_k3"),
    ("ATTENTION_PLACEMENT=invalid\n", "ATTENTION_PLACEMENT must be auto, context, layers or heads"),

    ("AUDIO=spark\n", "kimi_k3 has no bundled audio execution owner"),
    ("EMBEDDING=host\n", "GPU token embeddings only"),
    ("RTX_EXPERT_LAYERS=max\n", "RTX_EXPERT_LAYERS must be auto, all or N"),
    ("RTX_EXPERT_LAYERS=50%\n", "RTX_EXPERT_LAYERS must be auto, all or N"),
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
    assert "serve-kimi --snapshot" in result.stderr


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
    assert '"kimi_k3" | "kimi_k31" =>' in launch
    assert "kimi_k3 kimi_k31 1 92" in lines


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
    for absent in ("MAX_CONTEXT_TOKENS", "SPARK_MEMORY_CAP", "KV_CACHE", "CONCURRENCY"):
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
    for absent in ("MAX_CONTEXT_TOKENS", "SPARK_MEMORY_CAP", "KV_CACHE", "CONCURRENCY", "RTX_EXPERT_LAYERS"):
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
    serve = _line(result, "serve-kimi --snapshot")
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    assert "--pool-tokens" not in serve.split()
    assert _flag(serve, "--max-context-tokens") == "auto"


@pytest.mark.parametrize("recipe", ["example", "release"])
def test_kimi_min_recipe_launches_one_physical_owner_with_pinned_prefill_shape(tmp_path: Path, recipe: str) -> None:
    # The gate kit retired the K3 min card in v6 (K3 keeps its max card only).
    path = ROOT / "examples/configs/kimi-k3-exl3-k6-min.config"
    values = _keys(path)
    if recipe != "example":
        matrix_path = ("scripts/bench/release-smoke-v4.json" if recipe == "release"
                       else "scripts/bench/kits/gate/matrix-gate.json")
        matrix = json.loads((ROOT / matrix_path).read_text())
        entry = next(entry for entry in matrix["entries"] if entry["name"] == "kimi-k3-min")
        assert entry["config"] == str(path.relative_to(ROOT))
        assert entry["family"] == "kimi_k3" and entry["gpus"] == [0]
        assert len(entry["sparks"]) == 6
        values.update(matrix.get("defaults", {}).get("set", {}))
        values.update(entry["set"])
    assert (values["MODEL_ID"], values["MODEL_REVISION"]) == (KIMI_K6_ID, "1a28050d2954426a43a3db705e53ca120e5f2b47")
    assert (values["RTX_GPUS"], values["COORDINATOR_GPUS"], values["SPARK_COUNT"]) == ("1", "0", "6")
    assert values["SPARK_MEMORY_CAP"] == str(110 << 30)
    assert values["POOL_TOKENS"] == "auto" and values["SPECULATOR"] == "dflash2"
    assert values["PREFILL_ROWS"] == "1024"
    assert not {"RTX_EXPERT_LAYERS", "PREFILL_WAVES", "PREFILL_SPAN_ROWS",
                "MAX_CONTEXT_TOKENS", "CONCURRENCY", "GPU_MEMORY_CAP"}.intersection(values)
    draft = values["SPECULATOR_MODEL_ID"]
    _snapshot(tmp_path / "hf", draft, {}, revision="e77935fb4804e17eb55085bffd045eae1d779769")
    config = "".join(f"{key}={value}\n" for key, value in values.items()
                     if key not in ("MODEL_ID", "MODEL_REVISION"))
    result = _family_launch_result(tmp_path, KIMI, KIMI_K6_ID, config,
                                   extra_args=("--wip", "k3"), physical_gpus=(0,),
                                   encoder_plan={**PLANNED, "spark_budget_bytes": 110 << 30})
    serve = _line(result, "serve-kimi --snapshot")
    plan = _line(result, "cuteafd plan")
    assert _coordinator_gpus(serve) == [0] and _flag(serve, "--split-device") is None
    assert _flag(plan, "--rtx") == "1" and _flag(plan, "--spark-ranks") == "6"
    assert _flag(plan, "--speculator") == "dflash2" and _flag(serve, "--draft") is not None
    assert _flag(plan, "--spark-memory-cap") == _flag(serve, "--spark-memory-cap") == str(110 << 30)
    assert _flag(plan, "--prefill-rows") == _flag(serve, "--prefill-rows") == "1024"
    assert (_flag(plan, "--prefill-waves") or "3") == _flag(serve, "--prefill-waves") == "3"
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    assert "--pool-tokens" not in serve.split()
    assert _flag(serve, "--max-context-tokens") == "auto"
    workers = [line for line in result.stderr.splitlines() if "expertd-native" in line]
    assert len(workers) == 6 and all(f"--device-budget-bytes {110 << 30} " in worker for worker in workers)


# --- the gate card -------------------------------------------------------------

KIMI_K31_ID = "wrldsuksgo2mars/Kimi-K3.1-Pro-EXL3-K2-MC"


@pytest.mark.parametrize("stem,size,gpus", [
    ("kimi-k3", "max", [0, 1]),
    ("kimi-k31-mc", "min", [0]),
    ("kimi-k31-mc", "max", [0, 1]),
])
def test_kimi_gate_card_lists_and_generates_a_wip_launch(tmp_path: Path, stem: str, size: str,
                                                         gpus: list[int]) -> None:
    card = cards.load_cards(ROOT / "scripts/bench/kits/gate", [f"{stem}-{size}"])[0]
    assert card["family"] == "kimi_k3" and card["gpus"] == gpus
    assert card["sparks"] == ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"]
    assert card["profile"] == "smoke"
    k3 = stem == "kimi-k3"
    assert cards.config(card["config"])["MODEL_ID"] == (KIMI_K6_ID if k3 else KIMI_K31_ID)
    assert cards.slot_needs(card)["coordinator"] == cards.slot_needs(card)["spark"] == ["exl3/exl3-kimik3-k23"]
    key = tmp_path / "key"
    key.write_text("k" * 64 + "\n")
    key.chmod(0o600)
    job = cards.generate(card, "cand", {"instance": "t", "slot": "s"}, tmp_path, key, {}, {"fabric"}, None, 1)
    values = cards.config(Path(job["entry"]["config"]))
    assert job["entry"]["run_args"] == ["--wip", "s"]
    assert (values["SPARK_COUNT"], values["RTX_GPUS"]) == ("6", str(len(gpus)))
    assert values["COORDINATOR_GPUS"] == ",".join(map(str, gpus))
    if k3:
        assert values["SPECULATOR"] == "dflash2"
        if size == "min":
            assert values["SPARK_MEMORY_CAP"] == str(110 << 30)
        else:
            assert "SPARK_MEMORY_CAP" not in values
    else:
        # The shipped K3.1 config selects the checkpoint's block DSpark.
        assert values["SPECULATOR"] == "auto"
        assert values["CORE_VIEW"] == ("k6-compact" if size == "min" else "original")
        assert "SPARK_MEMORY_CAP" not in values
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
    assert "--native-k6" not in _line(result, "serve-kimi --snapshot").split()
    assert "--native-k6" not in _line(result, "cuteafd plan").split()
    removed = _kimi(tmp_path / "removed", "NATIVE_K6=on\n")
    assert removed.returncode != 0 and "NATIVE_K6" in removed.stderr


# --- Step-5 (#274) ---------------------------------------------------------------

STEP5 = {"model_type": "step5_vl", "vision_config": {"depth": 2},
         "text_config": {"model_type": "step5", "num_hidden_layers": 92, "moe_layer_list": list(range(3, 91))}}
STEP5_FP8 = "stepfun-ai/Step-5-Preview"
STEP5_NVFP4 = "stepfun-ai/Step-5-Preview-NVFP4"
STEP5_PLANNED = {"placement_supported": True, "fits": True, "spark_budget_bytes": 100 << 30, "encoder": None, "encoder_plan_hash": "ab" * 32}


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
    assert _flag(serve, "--max-context-tokens") == "auto"
    for absent in ("--pool-tokens", "--rtx-expert-layers",
                   "--draft", "--draft-fp8", "--dspark", "--local-experts"):
        assert absent not in serve.split(), absent
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert len(workers) == sparks and f"--world {sparks}" in workers[0]
    plan = _line(result, "cuteafd plan")
    assert "--spark-memory-cap" not in plan.split()
    assert all(f"--device-budget-bytes {100 << 30} " in worker for worker in workers)
    assert _flag(plan, "--spark-ranks") == str(sparks) and _flag(plan, "--rtx") == str(gpus)
    assert "--vision auto --audio off" in plan and "--require-ready" not in plan.split()
    lines = result.stderr.splitlines()
    probe = next(i for i, line in enumerate(lines) if "dst=/opt/cuteafd-serve-probe" in line)
    assert lines[probe].endswith("serve-step5 --help ") and probe < lines.index(plan)


@pytest.mark.parametrize("value", ["auto", "all", "0", "10", "88"])
def test_step5_fixed_rtx_expert_layers_reach_serve(tmp_path: Path, value: str) -> None:
    keys = f"SPARK_COUNT=6\nRTX_GPUS=2\nRTX_EXPERT_LAYERS={value}\n" + _hosts(6)
    serve = _line(_step5(tmp_path / "fixed", keys), "cuteafd serve-step5 --snapshot")
    assert _flag(serve, "--rtx-expert-layers") == (None if value == "auto" else value)
    result = _step5(tmp_path / "invalid", "SPARK_COUNT=6\nRTX_GPUS=2\nRTX_EXPERT_LAYERS=half\n" + _hosts(6))
    assert result.returncode != 0


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
    ("ATTENTION_PLACEMENT=context\n", "Step-5 has no context/layers attention executor"),
    ("EMBEDDING=host\n", "step5 serves GPU token embeddings only"),
    ("SPECULATOR_MODEL_ID=x/y\n", "unused configuration key: SPECULATOR_MODEL_ID"),
    ("SPECULATOR_MODEL_REVISION=abc\n", "unused configuration key: SPECULATOR_MODEL_REVISION"),
    ("DRAFT_LENGTH=adaptive\n", "DRAFT_LENGTH must be a positive integer"),
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
    assert (values["SPECULATOR"], values.get("VISION", "auto"), values["AUDIO"], values["POOL_TOKENS"]) == ("mtp", "auto", "off", "auto")
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
    ("step5-mc-original-max", [0, 1], ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"], "exl3/exl3-step5-k46"),
    ("step5-mc-original-min", [1], ["ostrich", "dodo", "emu"], "exl3/exl3-step5-k46"),
])
def test_step5_gate_cards_list_and_generate_wip_launches(tmp_path: Path, name, gpus, sparks, package) -> None:
    # The gate kit cards the live K4-MC checkpoint; the official FP8 and NVFP4
    # releases keep their example configs (tested above) but no gate card.
    card = cards.load_cards(ROOT / "scripts/bench/kits/gate", [name])[0]
    assert card["family"] == "step5" and card["gpus"] == gpus and card["sparks"] == sparks
    values = {**cards.config(card["config"]), **card["set"]}
    assert values["MODEL_ID"] == "wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC"
    assert values["MODEL_REVISION"] == "aedda90d25bd02bb6520bcaf88e06e6c9bc932c4"
    assert values["CORE_VIEW"] == "original"
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
    assert int(_flag(plan, "--gpu-memory-cap")) == total
    assert "--coordinator-weight-budget-gib" not in plan.split()
    # Serve admits resident weights against the budget the plan was made with.
    serve = _line(result, "serve-kimi --snapshot" if family == "kimi" else "serve-step5 --snapshot")
    assert float(_flag(serve, "--coordinator-weight-budget-gib")) == total / 2**30


@pytest.mark.parametrize("ranks", (6, 7, 8))
def test_kimi_launches_every_covered_rank_count_from_planner(tmp_path, ranks):
    hosts = "".join(f"SPARK_{i}_HOST=h{i}\nSPARK_{i}_LANE_A=10.0.0.{i + 1}\nSPARK_{i}_LANE_B=\n"
                    for i in range(ranks))
    result = _kimi(tmp_path, f"SPARK_COUNT={ranks}\n" + hosts)
    serve = _line(result, "serve-kimi --snapshot")
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
MUSE_PLANNED = {**PLANNED, "spark_ranks": 6, "encoder_plan_hash": "ab" * 32,
                "encoder": None, "audio_encoder": {"kind": {"kind": "off"}, "replicas": []}}


def _muse(tmp_path: Path, keys: str = "", *, wip: bool = True, **kwargs):
    return _family_launch_result(tmp_path, MUSE, MUSE_ID, "SPARK_COUNT=6\nRTX_GPUS=2\nSPECULATOR=off\n" + _hosts(6) + keys,
                                 extra_args=("--wip", "muse") if wip else (),
                                 encoder_plan=kwargs.pop("encoder_plan", MUSE_PLANNED), **kwargs)


def test_muse_dispatches_to_serve_muse_with_its_own_defaults(tmp_path: Path) -> None:
    result = _muse(tmp_path)
    serve = _line(result, "serve-muse --snapshot")
    assert f"/hf/hub/models--{MUSE_ID.replace('/', '--')}/snapshots/abc" in serve
    assert _flag(serve, "--peers") == ",".join(f"10.0.0.{i + 1}:19441" for i in range(6))
    # The head-split pair every family's launcher passes; serve-muse defaults to one GPU.
    assert (_flag(serve, "--device"), _flag(serve, "--split-device")) == ("0", "1")
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    assert _flag(serve, "--prefix-cache-entries") == "20"
    assert _flag(serve, "--vision") == _flag(serve, "--audio") == "off"
    assert _flag(serve, "--max-context-tokens") == "auto"
    assert _flag(serve, "--kv-cache") == "bf16"
    for absent in ("--pool-tokens", "--draft", "--mtp",
                   "--prefill-packing", "--expert-partial-bf16-max-rows", "--spark-prefill-reply-format", "--spark-decode-reply-format"):
        assert absent not in serve.split(), absent
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert len(workers) == 6 and all("--world 6 " in worker for worker in workers)
    plan = _line(result, "cuteafd plan")
    assert _flag(plan, "--speculator") == "off" and "--drafter-snapshot" not in plan.split()
    assert _flag(serve, "--tp2-dflash") == _flag(plan, "--tp2-dflash") == "false"
    assert "--vision auto --audio auto" in plan
    lines = result.stderr.splitlines()
    probe = next(i for i, line in enumerate(lines) if "dst=/opt/cuteafd-serve-probe" in line)
    assert lines[probe].endswith("serve-muse --help ")


@pytest.mark.parametrize("selection,expected", [("", "bf16"), ("bf16", "bf16"), ("fp8", "fp8")])
def test_muse_target_kv_reaches_plan_and_serve(tmp_path: Path, selection: str, expected: str) -> None:
    keys = f"KV_CACHE={selection}\n" if selection else ""
    result = _muse(tmp_path, keys)
    serve, plan = _line(result, "serve-muse --snapshot"), _line(result, "cuteafd plan")
    assert _flag(serve, "--kv-cache") == _flag(plan, "--kv-cache") == expected


def test_muse_plan_and_serve_share_graph_reserve_and_media_cache(tmp_path: Path) -> None:
    result = _muse(tmp_path, "VISION=auto\nGPU_GRAPH_RESERVE=gpu0=300MB,gpu1=200MB\nMEDIA_CACHE_BYTES=2GiB\n")
    serve, plan = _line(result, "serve-muse --snapshot"), _line(result, "cuteafd plan")
    for line in (plan, serve):
        assert "CUTEAFD_GPU_GRAPH_RESERVE=gpu0=300MB,gpu1=200MB" in line
    assert _flag(serve, "--media-cache-bytes") == _flag(plan, "--media-cache-bytes") == "2GiB"


def test_muse_invalid_target_kv_is_refused_before_launch(tmp_path: Path) -> None:
    result = _muse(tmp_path, "KV_CACHE=int8\n")
    assert result.returncode == 2
    assert "KV_CACHE must be bf16 or fp8" in result.stderr
    assert "serve-muse --snapshot" not in result.stderr


@pytest.mark.parametrize("mode", [None, "dflash2", "off"])
def test_muse_default_drafter_is_pinned_and_off_is_explicit(tmp_path: Path, mode) -> None:
    drafter = "nvidia/Muse-Spark-Open-816B-A42B-DFlash2"
    revision = "22e2669e4c60689f9bff0749a34d9d8e59ce7cd6"
    _snapshot(tmp_path / "hf", drafter, {}, revision=revision)
    # The shared fixture creates a repository once; add a second cached revision.
    root = tmp_path / "hf/hub" / f"models--{drafter.replace('/', '--')}"
    moving = root / "snapshots/unqualified-main"
    moving.mkdir()
    (moving / "config.json").write_text("{}")
    (root / "refs/main").write_text("unqualified-main")
    keys = "SPARK_COUNT=6\nRTX_GPUS=2\n" + _hosts(6)
    if mode is not None:
        keys += f"SPECULATOR={mode}\n"
    result = _family_launch_result(tmp_path, MUSE, MUSE_ID, keys,
                                  extra_args=("--wip", "muse"), encoder_plan=MUSE_PLANNED)
    serve, plan = _line(result, "serve-muse --snapshot"), _line(result, "cuteafd plan")
    assert _flag(plan, "--speculator") == ("off" if mode == "off" else "dflash2")
    if mode == "off":
        assert "--draft" not in serve.split() and "--drafter-snapshot" not in plan.split()
    else:
        snapshot = str(tmp_path / f"hf/hub/models--{drafter.replace('/', '--')}/snapshots/{revision}")
        assert _flag(serve, "--draft") == _flag(plan, "--drafter-snapshot") == snapshot


@pytest.mark.parametrize("selection,expected", [(None, "true"), ("off", "false"), ("on", "true")])
def test_muse_tp2_dflash_reaches_plan_and_serve(tmp_path: Path, selection, expected) -> None:
    drafter = "nvidia/Muse-Spark-Open-816B-A42B-DFlash2"
    revision = "22e2669e4c60689f9bff0749a34d9d8e59ce7cd6"
    _snapshot(tmp_path / "hf", drafter, {}, revision=revision)
    keys = "SPECULATOR=dflash2\n"
    if selection is not None:
        keys += f"TP2_DFLASH={selection}\n"
    result = _muse(tmp_path, keys)
    serve, plan = _line(result, "serve-muse --snapshot"), _line(result, "cuteafd plan")
    assert _flag(serve, "--tp2-dflash") == _flag(plan, "--tp2-dflash") == expected


def test_muse_prefill_scoring_uses_ordinary_launch(tmp_path: Path) -> None:
    result = _muse(tmp_path, "")
    serve, plan = _line(result, "serve-muse --snapshot"), _line(result, "cuteafd plan")
    assert "--full-prefill-logits" not in serve
    assert "--full-prefill-logits" not in plan


def test_step5_forwards_shared_speculation_trace(tmp_path: Path) -> None:
    trace = tmp_path / "trace" / "step5.jsonl"
    result = _step5(tmp_path, f"SPARK_COUNT=4\nRTX_GPUS=2\nSPECULATION_TRACE={trace}\n" + _hosts(4))
    serve = _line(result, "serve-step5 --snapshot")
    assert f"CUTEAFD_SPECULATION_TRACE={trace}" in serve
    assert f"{trace.parent}:{trace.parent}" in serve


def test_muse_forwards_speculation_trace(tmp_path: Path) -> None:
    trace = tmp_path / "trace" / "speculation-trace.jsonl"
    result = _muse(tmp_path, f"SPECULATION_TRACE={trace}\n")
    serve = _line(result, "serve-muse --snapshot")
    assert f"CUTEAFD_SPECULATION_TRACE={trace}" in serve
    assert f"{trace.parent}:{trace.parent}" in serve


def test_muse_trace_registration_still_rejects_unknown_keys(tmp_path: Path) -> None:
    trace = tmp_path / "trace" / "speculation-trace.jsonl"
    result = _muse(tmp_path, f"SPECULATION_TRACE={trace}\nSPECULATION_TRACE_TYPO=on\n")
    assert result.returncode == 2, result.stderr
    assert "unknown configuration key: SPECULATION_TRACE_TYPO" in result.stderr
    assert "docker run -d" not in result.stderr

@pytest.mark.parametrize("vision,audio", [("spark", "spark"), ("rtx", "spark"), ("off", "spark"), ("spark", "rtx")])
def test_muse_generic_media_plan_handoff(tmp_path: Path, vision: str, audio: str) -> None:
    def placed(kind, index):
        return {"kind": {"kind": kind, **({"rank": index} if kind == "spark" else {"gpu": index})}, "replicas": []}
    plan = {**PLANNED, "spark_ranks": 6, "encoder_plan_hash": "ab" * 32,
            "encoder": None if vision == "off" else placed(vision, 0),
            "audio_encoder": placed(audio, 1)}
    result = _muse(tmp_path, f"VISION={'off' if vision == 'off' else 'auto'}\nAUDIO=auto\n", encoder_plan=plan)
    serve = _line(result, "serve-muse --snapshot")
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
                             "CONCURRENCY=8\nEXPERT_PARTIAL_BF16_MAX_ROWS=128\nSPARK_PREFILL_REPLY_FORMAT=fp8\nSPARK_DECODE_REPLY_FORMAT=nvfp4\n"
                             "SPARK_MEMORY_CAP=107374182400\n")
    serve, plan = _line(result, "serve-muse --snapshot"), _line(result, "cuteafd plan")
    # The planner admits the same partial-reply overrides the server runs with.
    assert _flag(plan, "--expert-partial-bf16-max-rows") == "128"
    assert _flag(plan, "--spark-prefill-reply-format") == "fp8"
    assert _flag(plan, "--spark-decode-reply-format") == "nvfp4"
    assert _flag(serve, "--rtx-expert-layers") == "3"
    assert _flag(serve, "--pool-tokens") == "1048576"
    assert _flag(serve, "--max-context-tokens") == "262144"
    assert _flag(serve, "--max-sequences") == "8"
    assert _flag(serve, "--expert-partial-bf16-max-rows") == "128"
    assert "CUTEAFD_SPARK_PREFILL_REPLY_FORMAT=fp8" in serve
    assert "CUTEAFD_SPARK_DECODE_REPLY_FORMAT=nvfp4" in serve
    assert _flag(serve, "--spark-memory-cap") == "107374182400"


@pytest.mark.parametrize("drafter,revision", [
    ("RadixArk/Muse-Spark-Open-DFlash2", "46c90f41a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0"),
    ("nvidia/Muse-Spark-Open-816B-A42B-DFlash2", "22e2669ea0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0"),
])
@pytest.mark.parametrize("cache", [None, "bf16"])
def test_muse_dflash2_resolves_one_snapshot_for_plan_and_serve(tmp_path: Path, drafter, revision, cache) -> None:
    _snapshot(tmp_path / "hf", drafter, {}, revision=revision)
    keys = (f"SPECULATOR=dflash2\nSPECULATOR_MODEL_ID={drafter}\nSPECULATOR_MODEL_REVISION={revision}\n"
            "DRAFT_LENGTH=5\n" + ("" if cache is None else f"DRAFT_KV_CACHE={cache}\n"))
    result = _muse(tmp_path, keys)
    serve, plan = _line(result, "serve-muse --snapshot"), _line(result, "cuteafd plan")
    snapshot = str(tmp_path / f"hf/hub/models--{drafter.replace('/', '--')}/snapshots/{revision}")
    assert _flag(serve, "--draft") == _flag(plan, "--drafter-snapshot") == snapshot
    assert _flag(plan, "--speculator") == "dflash2"
    assert _flag(serve, "--draft-kv-cache") == _flag(plan, "--draft-kv-cache") == (cache or "fp8")
    assert _flag(serve, "--draft-length") == "5"
    assert "--draft-fp8" not in serve.split()


@pytest.mark.parametrize("mode", ["unrestricted", "exact"])
def test_muse_forwards_supported_packing_modes(tmp_path: Path, mode: str) -> None:
    serve = _line(_muse(tmp_path, f"PREFILL_PACKING={mode}\n"), "serve-muse --snapshot")
    assert _flag(serve, "--prefill-packing") == mode


@pytest.mark.parametrize("keys,diagnostic", [
    ("ATTENTION_PLACEMENT=layers\n", "Muse has no context/layers attention executor"),
    ("SPECULATOR=mtp\n", "SPECULATOR=mtp does not apply to muse_spark"),
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


@pytest.mark.parametrize("size,gpus,ranks", [("min", [0], 4), ("max", [0, 1], 6)])
def test_muse_examples_and_smoke_pin_the_qualified_drafter(tmp_path: Path, size, gpus, ranks) -> None:
    path = ROOT / f"examples/configs/muse-spark-exl3-{size}.config"
    values = _keys(path)
    assert values["MODEL_ID"] == "wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-Core-K6"
    assert values["MODEL_REVISION"] == "89998fb177acd9a862ac8bd4fb79bfd50e940c97"
    assert values["SPECULATOR"] == "dflash2"
    assert values["SPECULATOR_MODEL_ID"] == "nvidia/Muse-Spark-Open-816B-A42B-DFlash2"
    assert values["SPECULATOR_MODEL_REVISION"] == "22e2669e4c60689f9bff0749a34d9d8e59ce7cd6"
    assert (values["RTX_GPUS"], values["SPARK_COUNT"], values["POOL_TOKENS"]) == (str(len(gpus)), str(ranks), "auto")
    assert values["VISION"] == ("spark" if size == "min" else "auto")
    assert values["AUDIO"] == "auto"
    if size == "min":
        assert int(values["MAX_CONTEXT_TOKENS"]) > 262144
        assert int(values["MAX_CONTEXT_TOKENS"]) % 64 == 0
    else:
        assert "MAX_CONTEXT_TOKENS" not in values
    assert not any(key in values for key in ("RTX_EXPERT_LAYERS", "CONCURRENCY", "SPARK_MEMORY_CAP"))
    assert not any(needle in path.read_text().lower() for needle in PRIVATE)
    _snapshot(tmp_path / "hf", values["SPECULATOR_MODEL_ID"], {}, revision=values["SPECULATOR_MODEL_REVISION"])
    keys = "".join(f"{k}={v}\n" for k, v in values.items()
                   if k not in ("MODEL_ID", "MODEL_REVISION") and not k.startswith("SPARK_"))
    keys += f"SPARK_COUNT={ranks}\n" + _hosts(ranks)
    plan = {**MUSE_PLANNED, "spark_ranks": ranks,
            "encoder": {"kind": {"kind": "spark", "rank": 0}, "replicas": []},
            "audio_encoder": {"kind": {"kind": "spark", "rank": 1}, "replicas": []}}
    result = _family_launch_result(tmp_path, MUSE, values["MODEL_ID"], keys,
                                  extra_args=("--wip", "muse"), encoder_plan=plan)
    serve = _line(result, "serve-muse --snapshot")
    assert "--draft" in serve.split() and "--pool-tokens" not in serve.split()
    matrix = json.loads((ROOT / "scripts/bench/release-smoke-muse.json").read_text())
    entry = next(e for e in matrix["entries"] if e["name"] == f"muse-spark-exl3-{size}")
    assert entry["gpus"] == gpus and len(entry["sparks"]) == ranks
    assert entry["config"] == str(path.relative_to(ROOT)) and entry["run_args"] == ["--wip", "muse"]
    for key in ("MODEL_ID", "MODEL_REVISION", "SPECULATOR", "SPECULATOR_MODEL_ID", "SPECULATOR_MODEL_REVISION"):
        assert entry["set"][key] == values[key]


@pytest.mark.parametrize('size,ranks', [('min', 4), ('max', 6)])
def test_muse_mc_configs_do_not_replace_standalone_support(size, ranks) -> None:
    path = ROOT / f'examples/configs/muse-spark-exl3-k4-mc-{size}.config'
    values = _keys(path)
    standalone = _keys(ROOT / f'examples/configs/muse-spark-exl3-{size}.config')
    assert values['MODEL_ID'] == 'wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-MC'
    assert values['MODEL_REVISION'] == 'cc5dc2537083177e3f02ea0b2d9dfc179d40425c'
    assert values['CORE_VIEW'] == 'original'
    assert values['SPARK_COUNT'] == str(ranks)
    for key in ('SPECULATOR', 'SPECULATOR_MODEL_ID', 'SPECULATOR_MODEL_REVISION', 'VISION', 'AUDIO'):
        assert values[key] == standalone[key]
    if size == 'min':
        assert values['MAX_CONTEXT_TOKENS'] == 'auto'
        assert standalone['MAX_CONTEXT_TOKENS'] == '634304'
    matrix = json.loads((ROOT / 'scripts/bench/release-smoke-v5.json').read_text())
    entry = next(row for row in matrix['entries'] if row['name'] == f'muse-spark-exl3-k4-mc-{size}')
    assert entry['config'] == str(path.relative_to(ROOT))
    assert entry['set']['MODEL_ID'] == values['MODEL_ID']
    assert entry['set']['CORE_VIEW'] == 'original'
    current = json.loads((ROOT / 'benchmarks/current-matrix.json').read_text())
    assert any(row['checkpoint'] == values['MODEL_ID'] for row in current['cells'])
    assert not any(row['checkpoint'] == standalone['MODEL_ID'] for row in current['cells'])
    retired = next(row for row in current['retired'] if row['name'] == f'muse-spark-exl3-{size}')
    assert retired['config'] == f'examples/configs/muse-spark-exl3-{size}.config'


def test_family_table_and_planner_fixtures_name_muse() -> None:
    out = subprocess.run(["python3", str(ROOT / "scripts/lib/checkpoint-family.py"), "/dev/stdin"],
                         input=json.dumps(MUSE), capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["muse_spark", "muse_spark", "0", "61"]
    fixtures = json.loads((ROOT / "rust/crates/cuteafd-loader/tests/fixtures/launch-families.json").read_text())
    assert "muse_spark muse_spark 0 61" in {case["line"] for case in fixtures}
    assert '"muse_spark" =>' in (ROOT / "rust/crates/cuteafd-loader/src/plan/launch.rs").read_text()

@pytest.mark.parametrize("family,model,ranks", [("inkling2", "thinkingmachines/Inkling-2-Small", 4),
    ("step5", "wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC", 3)])
@pytest.mark.parametrize("vision", ["spark", "rtx", "off"])
def test_pe_shared_encoder_plan_handoff(tmp_path, family, model, ranks, vision):
    config = {"model_type": "inkling2" if family == "inkling2" else "step5_vl", "vision_config": {"depth": 2},
        "text_config": {"model_type": "inkling2" if family == "inkling2" else "step5", "num_hidden_layers": 48 if family == "inkling2" else 92, "moe_layer_list": list(range(3, 91))}}
    owner = None if vision == "off" else {"kind": {"kind": vision, "rank" if vision == "spark" else "gpu": 0}, "replicas": []}
    plan = {**PLANNED, "encoder_plan_hash": "ab" * 32, "encoder": owner}
    keys = f"SPARK_COUNT={ranks}\nRTX_GPUS=1\nSPECULATOR=off\nVISION={vision}\nAUDIO=off\n" + _hosts(ranks)
    result = _family_launch_result(tmp_path, config, model, keys, physical_gpus=(0,),
        extra_args=("--wip", "pe"), encoder_plan=plan)
    serve = _line(result, f"serve-{'inkling2' if family == 'inkling2' else 'step5'} --snapshot")
    assert _flag(serve, "--vision") == ("off" if vision == "off" else f"{vision}:0")
    if vision == "spark":
        assert _flag(serve, "--vision-peers") == "10.0.0.1:19442"
        assert _flag(serve, "--encoder-plan-hash") == "ab" * 32
        assert _flag(serve, "--encoder-revision") == "abc"
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert sum("--encoder " in worker for worker in workers) == (vision == "spark")


@pytest.mark.parametrize("family,model,ranks", [("inkling2", "thinkingmachines/Inkling-2-Small", 4),
    ("step5", "wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC", 3)])
def test_pe_encoder_replicas_match_plan_and_runtime(tmp_path, family, model, ranks):
    config = {"model_type": "inkling2" if family == "inkling2" else "step5_vl", "vision_config": {"depth": 2},
        "text_config": {"model_type": "step5", "num_hidden_layers": 48 if family == "inkling2" else 92,
            "moe_layer_list": list(range(3,91))}}
    plan = {**PLANNED, "encoder": {"kind":{"kind":"spark","rank":0},"replicas":[1]},"encoder_plan_hash":"ab"*32}
    keys = f"SPARK_COUNT={ranks}\nRTX_GPUS=1\nSPECULATOR=off\nVISION=spark\nVISION_REPLICAS=2\nAUDIO=off\n"+_hosts(ranks)
    result = _family_launch_result(tmp_path, config, model, keys, physical_gpus=(0,),
        extra_args=("--wip","pe"), encoder_plan=plan)
    serve = _line(result,f"serve-{family} --snapshot")
    assert _flag(serve,"--vision-replicas")=="2"
    assert _flag(_line(result,"cuteafd plan"),"--vision-replicas")=="2"
    assert _flag(serve,"--vision-peers")=="10.0.0.1:19442,10.0.0.2:19442"


@pytest.mark.parametrize('wip', [True, False], ids=['wip-slot', 'release-image'])
@pytest.mark.parametrize('dry_run', [True, False], ids=['plan', 'serve'])
def test_missing_vision_attention_refuses_planned_startup(tmp_path, wip, dry_run):
    library = tmp_path / 'libcuteafd_native.so'
    # The unconditional C ABI is present; the generated init entrypoints are not.
    subprocess.run(['cc', '-shared', '-fPIC', '-x', 'c', '-', '-o', str(library)], check=True,
                   text=True, input='void cuteafd_vision_create(void) {}\n')
    plan = {**PLANNED, 'encoder': {'kind': {'kind': 'rtx', 'gpu': 0}, 'replicas': []}}
    result = _family_launch_result(tmp_path, KIMI, KIMI_ID, KIMI_KEYS + 'VISION=auto\n',
        native_fixture=library, encoder_plan=plan,
        extra_args=(('--wip', 'k3') if wip else ()) + (('--dry-run',) if dry_run else ()))
    scope = 'WIP' if wip else 'RELEASE'
    assert result.returncode == 2, result.stderr
    assert f'this build lacks vision attention AOT; rebuild with CUTEAFD_{scope}_VISION_ATTENTION_AOT=ON or set VISION=off' in result.stderr
    assert 'serve-kimik3 --snapshot' not in result.stderr and 'docker run -d' not in result.stderr


def test_missing_pcm_audio_refuses_planned_startup(tmp_path):
    library = tmp_path / 'libcuteafd_native.so'
    subprocess.run(['cc', '-shared', '-fPIC', '-x', 'c', '-', '-o', str(library)], check=True,
                   text=True, input='void cuteafd_pcm_create(void) {}\n')
    plan = {**PLANNED, 'audio_encoder': {'kind': {'kind': 'rtx', 'gpu': 0}, 'replicas': []}}
    result = _muse(tmp_path, 'VISION=off\nAUDIO=auto\n', native_fixture=library, encoder_plan=plan)
    assert result.returncode == 2, result.stderr
    assert 'this build lacks PCM audio AOT; rebuild with CUTEAFD_WIP_AUDIO_AOT=ON or set AUDIO=off' in result.stderr
    assert 'serve-muse --snapshot' not in result.stderr
