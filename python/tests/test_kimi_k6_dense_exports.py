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
    transforms = ModuleType("b12x.integration.cuteafd.trellis_transform")
    transforms.compile_trellis_input_aot = lambda **params: params
    transforms.compile_trellis_output_aot = lambda **params: params
    monkeypatch.setitem(sys.modules, transforms.__name__, transforms)
    dequant = ModuleType("b12x.integration.cuteafd.trellis_dequant")
    dequant.compile_trellis_dequant_aot = lambda **params: params
    projection = ModuleType("b12x.integration.cuteafd.pointwise")
    projection.compile_bf16_projection_aot = lambda **params: params
    torch = ModuleType("torch")
    torch.float32 = "f32"
    for fake_module in (dequant, projection, torch):
        monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    programs = module["programs"]()
    assert len(programs) == 87
    assert len({name for name, *_ in programs}) == len(programs)
    core = [item for item in programs if item[1] == "trellis_linear"]
    assert len(core) == 2 * len(module["TRELLIS_SHAPES"])
    for _, op, params, compile_program in core:
        assert op == "trellis_linear"
        assert params["bits"] == 6
        assert params["codebook"] in ("mcg", "mul1")
        assert params["max_rows"] == 4096
        assert params["rotation_count"] == 1
        assert compile_program() == params
    ingress = [item for item in programs if item[1] == "trellis_input"]
    egress = [item for item in programs if item[1] == "trellis_output"]
    assert len(ingress) == 12
    assert {params["features"] for _, _, params, _ in ingress} == {1536, 3072, 3584, 6144, 7168, 12288}
    assert {params["features"] for _, _, params, _ in egress} == {640, 1536, 3072, 3584, 6144, 7168, 9216, 12288, 18432, 163840}
    for name, op, params, compile_program in ingress + egress:
        assert params["max_rows"] == 4096
        assert compile_program() == params
        if op == "trellis_input":
            assert params["input_dtype"] == "bf16"
            suffix = "_rotated_bf16" if params.get("output_dtype") == "bf16" else ""
            assert name == f"trellis_input_k{params['features']}_bf16{suffix}"
        else:
            dtype = "f32" if params["features"] == 163840 else "bf16"
            assert params["output_dtype"] == dtype
            assert name == f"trellis_output_n{params['features']}_{dtype}"
    for _, op, params, compile_program in programs:
        assert compile_program() == params
    names = {name for name, *_ in programs}
    for policy in module["row_policies"]():
        for field in ("native_program", "native_input", "wide_input", "dequant_program", "projection_program", "output_program"):
            assert policy[field] in names
    fixture = module["programs_for_shapes"]((("s1", 128, 256, 1), ("s2", 128, 256, 2)),
        max_rows=64, include_chunk=False, f32_outputs=(256,))
    assert {name for name, *_ in fixture} == {
        "trellis_s1_mcg", "trellis_s1_mul1", "trellis_s2_mcg", "trellis_s2_mul1",
        "trellis_input_k128_bf16", "trellis_output_n256_f32"}
    assert all(params["max_rows"] == 64 for _, _, params, _ in fixture)
    imports = [node.module for node in ast.walk(ast.parse(EXPORTER.read_text()))
               if isinstance(node, ast.ImportFrom)]
    assert "b12x.gemm.trellis_linear" not in imports
    assert "b12x.integration.cuteafd.trellis_linear" in imports


def test_comparison_group_is_opt_in():
    groups = runpy.run_path(str(ROOT / "scripts/build/compare-sm-exports.py"))["jobs"]("all")
    group = next(row for row in groups if row[0] == "kimi-k6-dense")
    assert group[1:] == ("export_b12x_dsv4_aot.py", ["--geometry", "kimi_k6_dense"])


def test_chunk_route_policy_and_scratch(monkeypatch):
    module = load_exporter(monkeypatch)
    policies = module["row_policies"]()
    assert len(policies) == 26
    for policy in policies:
        k, n, c = policy["input_features"], policy["output_features"], policy["chunk_columns"]
        assert c % 128 == 0 and n % c == 0
        assert policy["weight_scratch_bytes"] == c*k*2 <= 256*1024*1024
        assert policy["threshold_rows"] == 64
        assert policy["below"] == "shared_native"
        assert policy["at_or_above"] == "chunk_dequant_bf16"
        assert policy["stitch_before_output_transform"]
        assert policy["chunk_output_dtype"] == "f32"
        assert policy["chunk_output_scratch_bytes_per_row"] == c*4
    head = next(p for p in policies if p["shape"] == "lm_head")
    assert head["chunk_columns"] == 16384
    assert head["output_program"].endswith("_f32")
