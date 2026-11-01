"""CPU regression tests for the replicated-native qualifier's TP-degree -> role id.

Two independent code paths compute the native role family:

* `run_checkpoint` (the real-checkpoint oracle) and
* the synthetic/timing path.

Both previously used the legacy `3 + spark_tp` arithmetic, which is correct for
TP2 (5) and TP3 (6) but maps TP6 to 9 instead of the registered role 7, so a TP6
run died on `assert meta.role == role`. These tests assert the mapping is driven
by the role TABLE, not by arithmetic, and that no `3 + ...` formula survives.

They also assert the failure is loud rather than silent: the role assertion must
actually compare against the table entry.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "python" / "tools"
QUALIFIER = TOOLS / "qualify" / "deepseek_v41" / "qualify_v41_replicated_native.py"


def _load_qualifier():
    torch = pytest.importorskip("torch")
    assert torch is not None
    if str(TOOLS) not in sys.path:
        sys.path[:0] = [str(TOOLS / d) for d in ("lib", "aot", "bench", "hf", "qualify/deepseek_v4", "qualify/deepseek_v41", "qualify/glm5_flash")]
    try:
        spec = importlib.util.spec_from_file_location("cuteafd_qualifier_roles", QUALIFIER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"pinned SparkInfer/b12x unavailable: {error}")
    return module


def test_role_table_is_the_authority_for_every_degree() -> None:
    module = _load_qualifier()
    artifacts = module.native_spark_artifacts()
    assert artifacts[6][0].role == 7
    assert artifacts[2][0].role == 5 and artifacts[3][0].role == 6
    assert len(artifacts[5]) == len(artifacts[7]) == 2


def test_no_legacy_role_arithmetic_survives_in_the_source() -> None:
    source = QUALIFIER.read_text(encoding="utf-8")
    # Any `3 + options.spark_tp` (in either path) is the bug this prevents.
    assert not re.search(r"3\s*\+\s*options\.spark_tp", source), (
        "legacy 3 + spark_tp role arithmetic is back; TP6 would map to role 9")
    assert "native_shard(options.spark_tp, options.tp_rank, options.layer)" in source
    assert "role, intermediate = artifact.role, artifact.logical" in source
    assert "tp4_legacy" not in source


def test_both_role_paths_are_covered_by_executable_checks() -> None:
    """The two paths must contain an executed role assertion, not a comment."""
    tree = ast.parse(QUALIFIER.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    assert "run_checkpoint" in functions
    checkpoint_asserts = [
        node for node in ast.walk(functions["run_checkpoint"])
        if isinstance(node, ast.Assert)
    ]
    assert checkpoint_asserts, "run_checkpoint has no role assertion"

    # The timing path calls assert_native_role_geometry, which is the shared gate.
    timing_asserts = [
        node for node in ast.walk(functions.get("_run_timing", ast.Pass()))
        if isinstance(node, ast.Call)
    ]
    names = {getattr(node.func, "id", None) for node in timing_asserts}
    assert "assert_native_role_geometry" in names, (
        "_run_timing must gate on the shared geometry/role helper")


def test_role_gate_rejects_the_legacy_tp6_mapping() -> None:
    """A library publishing role 9 for a TP6 request must be refused."""
    module = _load_qualifier()
    import ctypes as C
    import ctypes.util  # noqa: F401

    fields = (["abi_version", "role", "experts", "hidden_size",
               "logical_intermediate", "kernel_intermediate", "topk",
               "capacity_rows"] + ["scratch_bytes"] +
              ["max_rows", "rows_padded", "max_tasks", "max_phys_tiles",
               "max_active_clusters"] + ["input_dtype"])

    class _Info(C.Structure):
        _fields_ = [(name, C.c_uint64 if name == "scratch_bytes" else C.c_int32)
                    for name in fields]

    def make(role):
        def fn(capacity, out):
            data = [2, role, 384, 5120, 384, 384, 6, capacity, 4096, 32, 32, 64, 64, 188, 7]
            C.memmove(out, (C.c_int32 * len(data))(*data), C.sizeof(_Info))
            return 0
        return type("L", (), {"cuteafd_v41_expert_info": staticmethod(fn)})()

    # role 9 is what the legacy formula would have expected: must be rejected.
    with pytest.raises(AssertionError):
        module.assert_native_role_geometry(make(9), 16, artifact=module.native_spark_artifacts()[6][0])
    # role 7 is the registered TP6 family: accepted.
    assert module.assert_native_role_geometry(make(7), 16, artifact=module.native_spark_artifacts()[6][0]).role == 7
