"""CPU contract tests for bitwise-exact small-row TMA decode tile exports."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draft_bf16_projection_uses_decode_tma_tiles(monkeypatch):
    captured = {}

    class FakeTma:
        def __init__(self, n, k, *, out_dtype, tiles_by_rows):
            captured["n"] = n
            captured["k"] = k
            captured["out_dtype"] = out_dtype
            captured["tiles_by_rows"] = tiles_by_rows

        def key(self):
            return ("fake", captured["n"], captured["k"], captured["tiles_by_rows"])

    def fake_decode_tma_tiles(n, k):
        return ((16, 1, 2, 16, 16, 8), (64, 2, 2, 32, 32, 8)) if n <= 1024 else ()

    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16", float32="f32"))
    monkeypatch.setitem(sys.modules, "cutlass", SimpleNamespace(BFloat16="c_bf16", Float32="c_f32"))
    monkeypatch.setitem(sys.modules, "b12x", types.ModuleType("b12x"))
    monkeypatch.setitem(sys.modules, "b12x.gemm", types.ModuleType("b12x.gemm"))
    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv", types.ModuleType("b12x.gemm.bf16_gemv"))
    monkeypatch.setitem(
        sys.modules,
        "b12x.gemm.bf16_gemv._skinny",
        SimpleNamespace(TmaBf16Projection=FakeTma, decode_tma_tiles=fake_decode_tma_tiles),
    )
    monkeypatch.setitem(sys.modules, "b12x.integration", types.ModuleType("b12x.integration"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd", types.ModuleType("b12x.integration.cuteafd"))
    monkeypatch.setitem(
        sys.modules,
        "b12x.integration.cuteafd._common",
        SimpleNamespace(
            Operand=lambda *a: a,
            Scalar=lambda *a: a,
            compile_program=lambda launch, **kw: kw,
        ),
    )

    mod = _load("test_dsv4_small_row", "python/tools/aot/export_b12x_dsv4_aot.py")
    res_small = mod.compile_draft_bf16_projection(8192, 1024, "bf16")
    assert res_small["geometry"]["tiles_by_rows"] == ((16, 1, 2, 16, 16, 8), (64, 2, 2, 32, 32, 8))
    res_wide = mod.compile_draft_bf16_projection(8192, 16384, "f32")
    assert res_wide["geometry"]["tiles_by_rows"] == ((16, 1, 4, 16, 64, 6),)
