"""CPU contract tests for Hy4 and Hy4-flash skinny-GEMV + TMA router exports."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "python" / "tools" / "aot" / "export_b12x_hy4_aot.py"

HY4_ROUTER_TILES = ((16, 2, 4, 128), (64, 8, 4, 128))


class _Compilers(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("compile_"):
            return lambda *args, **kwargs: (self.__name__.rsplit(".", 1)[-1], name, args, kwargs)
        raise AttributeError(name)


class _FakeRoutedBf16Projection:
    def __init__(self, n, k, *, out_dtype, max_skinny_rows=None, skinny_tiles_by_rows=()):
        self.n = n
        self.k = k
        self.out_dtype = out_dtype
        self.max_skinny_rows = max_skinny_rows
        self.skinny_tiles_by_rows = tuple(skinny_tiles_by_rows)

    @property
    def exact_row_cap(self):
        return self.max_skinny_rows if self.max_skinny_rows > 0 else 2**31 - 1

    def key(self):
        return (self.n, self.k, self.out_dtype, self.max_skinny_rows, self.skinny_tiles_by_rows)


@pytest.fixture()
def hy4_exporter(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", types.SimpleNamespace(LOCK_DATA={}))
    torch_stub = types.SimpleNamespace(bfloat16="bf16", float32="f32")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    cutlass_mod = types.ModuleType("cutlass")
    cutlass_mod.Float32 = "cutlass.Float32"
    cutlass_mod.Int32 = "cutlass.Int32"
    cute_mod = types.ModuleType("cutlass.cute")
    cute_mod.Pointer = object
    cute_mod.jit = lambda fn: fn
    cutlass_mod.cute = cute_mod
    monkeypatch.setitem(sys.modules, "cutlass", cutlass_mod)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute_mod)

    cuda_mod = types.ModuleType("cuda")
    cuda_bindings = types.ModuleType("cuda.bindings")
    cuda_driver = types.ModuleType("cuda.bindings.driver")
    cuda_driver.CUstream = object
    cuda_mod.bindings = cuda_bindings
    cuda_bindings.driver = cuda_driver
    monkeypatch.setitem(sys.modules, "cuda", cuda_mod)
    monkeypatch.setitem(sys.modules, "cuda.bindings", cuda_bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", cuda_driver)

    skinny_mod = types.ModuleType("b12x.gemm.bf16_gemv._skinny")
    skinny_mod.ROUTER_TILES_BY_ROWS = {(256, 6144): HY4_ROUTER_TILES}
    skinny_mod.RoutedBf16Projection = _FakeRoutedBf16Projection

    common_mod = types.ModuleType("b12x.integration.cuteafd._common")
    common_mod.Operand = lambda name, dtype, shape, role="in", note="": (name, dtype, shape, role, note)
    common_mod.Scalar = lambda name: ("scalar", name)
    common_mod.compile_program = lambda launch, **kwargs: types.SimpleNamespace(launch=launch, **kwargs)

    package = types.ModuleType("b12x.integration.cuteafd")
    pointwise = _Compilers("b12x.integration.cuteafd.pointwise")
    package.pointwise = pointwise
    package._common = common_mod
    monkeypatch.setitem(sys.modules, "b12x", types.ModuleType("b12x"))
    monkeypatch.setitem(sys.modules, "b12x.gemm", types.ModuleType("b12x.gemm"))
    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv", types.ModuleType("b12x.gemm.bf16_gemv"))
    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv._skinny", skinny_mod)
    monkeypatch.setitem(sys.modules, "b12x.integration", types.ModuleType("b12x.integration"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd", package)
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd.pointwise", pointwise)
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd._common", common_mod)

    spec = importlib.util.spec_from_file_location("export_b12x_hy4_aot_under_test", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tp", [1, 2])
def test_hy4_router_exports_routed_bf16_projection_with_tuned_tiles(hy4_exporter, tp):
    projs = hy4_exporter.hy4_builders(tp, "projections")
    prog = projs["router"]()
    proj = prog.launch.proj
    assert isinstance(proj, _FakeRoutedBf16Projection)
    assert (proj.n, proj.k, proj.out_dtype, proj.max_skinny_rows) == (256, 6144, "cutlass.Float32", 64)
    assert proj.skinny_tiles_by_rows == HY4_ROUTER_TILES
    assert prog.name == "hy4_router_scores"
    assert [op[0] for op in prog.operands] == ["x", "w", "logits"]
    assert [op[1] for op in prog.operands] == ["bf16", "bf16", "f32"]
    assert prog.scalars == (("scalar", "rows"),)
    assert prog.geometry == {
        "hidden": 6144,
        "experts": 256,
        "skinny_max_rows": 64,
        "exact_row_cap": 64,
    }


@pytest.mark.parametrize("tp", [1, 2])
def test_hy4_flash_router_exports_routed_bf16_projection_with_generic_skinny(hy4_exporter, tp):
    projs = hy4_exporter.builders(tp, "projections")
    prog = projs["router"]()
    proj = prog.launch.proj
    assert isinstance(proj, _FakeRoutedBf16Projection)
    assert (proj.n, proj.k, proj.out_dtype, proj.max_skinny_rows) == (512, 3072, "cutlass.Float32", 64)
    assert proj.skinny_tiles_by_rows == ()
    assert prog.name == "hy4_router_scores"
    assert [op[0] for op in prog.operands] == ["x", "w", "logits"]
    assert [op[1] for op in prog.operands] == ["bf16", "bf16", "f32"]
    assert prog.scalars == (("scalar", "rows"),)
    assert prog.geometry == {
        "hidden": 3072,
        "experts": 512,
        "skinny_max_rows": 64,
        "exact_row_cap": 64,
    }


@pytest.mark.parametrize(
    "family,hidden,experts",
    [("hy4", 6144, 256), ("hy4_flash", 3072, 512)],
)
def test_hy4_program_loaders_require_router_logits_abi(family, hidden, experts):
    src = (ROOT / f"rust/crates/cuteafd-daemon/src/families/{family}/programs.rs").read_text()
    assert 'load("router", &["x", "w", "logits"])?' in src
    assert f"validate_router_spec(proj_router.spec(), {hidden}, {experts})?" in src
