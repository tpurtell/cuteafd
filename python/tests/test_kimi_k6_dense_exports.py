"""CPU-only contracts for local Kimi core shapes and shared ABI exports."""
import ast
from pathlib import Path
import runpy
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "python/tools/aot/export_b12x_kimi_k6_dense_aot.py"


def load_exporter(monkeypatch):
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", ModuleType("_pinned_sparkinfer"))
    # The exporter prepends its lib path like every standalone AOT entry point.
    monkeypatch.setattr(sys, "path", list(sys.path))
    return runpy.run_path(str(EXPORTER))


def test_local_shapes_and_sides(monkeypatch):
    shapes = load_exporter(monkeypatch)["TRELLIS_SHAPES"]
    assert len({name for name, *_ in shapes}) == len(shapes)
    assert {(k, n, side) for _, k, n, side in shapes} == {
        (7168, 6144, 1), (6144, 7168, 2), (7168, 3072, 1), (3072, 7168, 2),
        (7168, 3584, 1), (3584, 7168, 1), (7168, 1536, 1), (7168, 640, 1),
        (1536, 18432, 1), (1536, 9216, 1), (7168, 12288, 1),
        (12288, 7168, 1), (7168, 163840, 1),
    }
    assert all(k % 128 == n % 128 == 0 for _, k, n, _ in shapes)
    assert not any(n == 576 or k == 1792 or n == 1792 for _, k, n, _ in shapes)


def test_only_shared_native_compiler_and_both_codebooks(monkeypatch):
    module = load_exporter(monkeypatch)
    fake = ModuleType("b12x.integration.cuteafd.trellis_linear")
    fake.compile_trellis_linear_aot = lambda **params: params
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    programs = module["programs"]()
    assert len(programs) == 2 * len(module["TRELLIS_SHAPES"])
    assert len({name for name, *_ in programs}) == len(programs)
    for _, op, params, compile_program in programs:
        assert op == "trellis_linear"
        assert params["bits"] == 6
        assert params["codebook"] in ("mcg", "mul1")
        assert params["max_rows"] == 4096
        assert params["rotation_count"] == 1
        assert compile_program() == params
    imports = [node.module for node in ast.walk(ast.parse(EXPORTER.read_text()))
               if isinstance(node, ast.ImportFrom)]
    assert "b12x.gemm.trellis_linear" not in imports
    assert "b12x.integration.cuteafd.trellis_linear" in imports


def test_comparison_group_is_opt_in():
    groups = runpy.run_path(str(ROOT / "scripts/build/compare-sm-exports.py"))["jobs"]("all")
    group = next(row for row in groups if row[0] == "kimi-k6-dense")
    assert group[1:] == ("export_b12x_dsv4_aot.py", ["--geometry", "kimi_k6_dense"])
