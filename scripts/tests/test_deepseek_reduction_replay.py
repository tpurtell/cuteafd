from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path[:0] = [str(TOOLS / d) for d in ("lib", "aot", "bench", "hf", "qualify/deepseek_v4", "qualify/deepseek_v41", "qualify/glm5_flash")]

import analyze_expert_reduction_replay as analyze  # noqa: E402
import collect_expert_route_bank as collect  # noqa: E402
import plan_expert_reduction_replay as plan  # noqa: E402


def slice_qualifier():
    spec = importlib.util.spec_from_file_location(
        "v41_slice_native", TOOLS / "qualify/deepseek_v41/qualify_v41_slice_native.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slice_repeat_equality_accepts_all_exported_capacities():
    qualifier = slice_qualifier()
    options, capacities = qualifier.parse_args([
        "--native-lib", "/native.so", "--width", "192",
        "--capacities", "1,16,80,256,1024,4096", "--repeat-equality", "3",
        "--spark-tp", "3", "--require-ordered",
    ])
    assert capacities == set(qualifier.CAPACITIES)
    assert options.repeat_equality == 3
    assert options.spark_tp == 3
    _, defaults = qualifier.parse_args([
        "--native-lib", "/native.so", "--width", "192", "--repeat-equality", "3",
    ])
    assert defaults == set(qualifier.CAPACITIES)


@pytest.mark.parametrize("extra", [
    ["--repeat-equality", "2"], ["--capacities", "256"],
    ["--capacities", "512", "--repeat-equality", "3"],
    ["--full-backbone", "--spark-tp", "3"],
])
def test_slice_repeat_equality_rejects_invalid_options(extra):
    with pytest.raises(SystemExit):
        slice_qualifier().parse_args(["--native-lib", "/native.so", "--width", "192", *extra])


@pytest.mark.parametrize("samples, exact", [
    ([b"\x00\x00\x00\x00"] * 3, True),
    ([b"\x00\x00\x00\x00", b"\x00\x00\x00\x80", b"\x00\x00\x00\x00"], False),
])
def test_slice_repeat_equality_checks_bytes_and_runs_every_replay(samples, exact):
    qualifier = slice_qualifier()
    calls = []
    values = iter(samples)
    assert qualifier.repeat_equal(lambda: calls.append(1), lambda: next(values), 3) is exact
    assert len(calls) == 3


def flash_manifest() -> dict[str, object]:
    return {
        "record": "manifest",
        "schema": "cuteafd-expert-route-bank-v2",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "hidden_size": 4096,
        "top_k": 6,
        "routed_experts": 256,
        "quantization_recipe": "deepseek_v4_native_fp4_fp8_mixed_v1",
        "routed_layers": [0, 1],
    }


def test_route_trace_uses_deepseek_top6_and_expert_bounds() -> None:
    line = (
        "protocol_v2_expert_queue_row_routes request_id_base=1 layer_id=0 "
        "rows=1 source_kinds=MtpVerifyBlock row_routes=0:0+1+2+3+4+5"
    )
    fragment = collect.parse_route_line(line, top_k=6, routed_experts=256)
    assert fragment is not None
    assert fragment.routes == ((0, 1, 2, 3, 4, 5),)

    with pytest.raises(ValueError, match="top-8"):
        collect.parse_route_line(line, top_k=8, routed_experts=256)
    with pytest.raises(ValueError, match="outside 0..4"):
        collect.parse_route_line(line, top_k=6, routed_experts=5)
    with pytest.raises(ValueError, match="repeats row index 0"):
        collect.parse_route_line(
            line + ",0:6+7+8+9+10+11", top_k=6, routed_experts=256
        )


def test_plan_preserves_explicit_deepseek_geometry(tmp_path: Path) -> None:
    bank = tmp_path / "routes.jsonl"
    records = [flash_manifest()]
    for layer_id in (0, 1):
        records.append(
            {
                "record": "fragment",
                "output_id": "code-000",
                "case": "code",
                "cycle": 0,
                "physical_m": 1,
                "layer_id": layer_id,
                "routes": [[0, 1, 2, 3, 4, 5]],
            }
        )
    bank.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    _, geometry, cycles = plan.load_cycles(bank)
    assert geometry.hidden_size == 4096
    assert geometry.top_k == 6
    assert geometry.routed_layers == (0, 1)
    chain = plan.chain_record(
        geometry,
        "semantic-m001-0000",
        "semantic",
        1,
        [plan.CyclePiece(cycles[0], 0, 1)],
    )
    assert [layer["layer_id"] for layer in chain["layers"]] == [0, 1]
    assert all(len(row) == 6 for layer in chain["layers"] for row in layer["routes"])


def test_plan_rejects_legacy_implicit_geometry() -> None:
    legacy = flash_manifest()
    legacy["schema"] = "cuteafd-expert-route-bank-v1"
    with pytest.raises(ValueError, match="explicit DeepSeek model geometry"):
        plan.geometry_from_manifest(legacy)


def test_plan_rejects_duplicate_experts_in_a_route(tmp_path: Path) -> None:
    bank = tmp_path / "duplicate-routes.jsonl"
    records = [flash_manifest()]
    for layer_id in (0, 1):
        records.append(
            {
                "record": "fragment",
                "output_id": "code-000",
                "case": "code",
                "cycle": 0,
                "physical_m": 1,
                "layer_id": layer_id,
                "routes": [[0, 1, 2, 3, 4, 4]],
            }
        )
    bank.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repeats an expert ID"):
        plan.load_cycles(bank)


def test_reduction_summary_uses_recorded_layer_samples() -> None:
    measurements = [
        {
            "physical_m": 12,
            "chain_id": "chain",
            "path": "coordinator",
            "path_order": 0,
            "dispatch_ms": 4.0,
            "layer_ms": [1.0, 3.0],
        },
        {
            "physical_m": 12,
            "chain_id": "chain",
            "path": "spark-row-sharded",
            "path_order": 1,
            "dispatch_ms": 6.0,
            "layer_ms": [2.0, 4.0],
        },
    ]
    pairs = analyze.paired_by_m(measurements)
    summary = analyze.summarize(
        {
            "schema": "cuteafd-expert-reduction-replay-result-v2",
            "cohort": "semantic",
            **{
                key: value
                for key, value in flash_manifest().items()
                if key not in {"record", "schema"}
            },
        },
        pairs,
        bootstrap_samples=8,
        seed=1,
    )
    row = summary["rows"][0]
    assert row["coordinator_layer_ms_mean"] == 2.0
    assert row["spark_layer_ms_mean"] == 3.0
    assert summary["schema"] == "cuteafd-expert-reduction-replay-summary-v2"
    assert summary["hidden_size"] == 4096
    assert summary["routed_layers"] == [0, 1]


@pytest.mark.parametrize("ranks", range(1, 10))
def test_native_qualifiers_share_covered_count_choices_without_cuda(ranks):
    import qualify_v41_replicated_native as replicated
    from spark_packages import native_spark_artifacts
    covered = ranks in native_spark_artifacts()
    slice_args = ["--native-lib", "/native.so", "--width", "192", "--spark-tp", str(ranks)]
    replicated_args = ["--native-lib", "/native.so", "--checkpoint", "--spark-tp", str(ranks)]
    if covered:
        assert slice_qualifier().parse_args(slice_args)[0].spark_tp == ranks
        assert replicated.parse_args(replicated_args).spark_tp == ranks
    else:
        with pytest.raises(SystemExit):
            slice_qualifier().parse_args(slice_args)
        with pytest.raises(SystemExit):
            replicated.parse_args(replicated_args)
    assert replicated.native_spark_artifacts() == native_spark_artifacts()


def test_native_tp8_role_gate_checks_loaded_geometry_without_cuda():
    import ctypes as C
    import qualify_v41_replicated_native as replicated
    from _v41_expert_native import Info
    class Library:
        def cuteafd_v41_expert_info(self, capacity, output):
            info = C.cast(output, C.POINTER(Info)).contents
            info.role, info.logical_intermediate, info.kernel_intermediate = 8, 288, 288
            info.experts, info.hidden_size, info.topk = 384, 5120, 6
            info.capacity_rows = capacity
            return 0
    replicated.assert_native_role_geometry(Library(), 80, artifact=replicated.native_spark_artifacts()[8][0])
    with pytest.raises(AssertionError):
        replicated.assert_native_role_geometry(Library(), 80, artifact=replicated.native_spark_artifacts()[6][0])
