"""GLM 5.3 Flash's wide decode programs in the coordinator exporter
(python/tools/aot/export_b12x_dsv4_aot.py), listed without SparkInfer or a GPU: the compile
functions are stubs that return their call."""
from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "python" / "tools" / "aot" / "export_b12x_dsv4_aot.py"
GEOMETRY = types.SimpleNamespace(moe_inter=2048, dense_inter=12288, index_kpool=4, kda_width=8192, hidden=4096)
WIDE_STEMS = {
    "index_producer_m128", "index_producer_c_m128", "index_topk_decode_m128", "mhc_post_pre_m128", "kda_m128",
    "kda_w8_m128", "kda_s16_m128", "mla_producer_m128", "o_m128", "sparse_mla_decode_m128", "ffn_i2048_m128",
    "ffn_i12288_m128", "kda_commit_m128", "kda_commit_s16_m128", "kda_commit_c_m128", "kda_commit_c_s16_m128",
}


class _Compilers(types.ModuleType):
    """A SparkInfer module whose ``compile_*`` functions return (module, function, args, kwargs)."""

    def __getattr__(self, name):
        if name.startswith("compile_"):
            return lambda *args, **kwargs: (self.__name__.rsplit(".", 1)[-1], name, args, kwargs)
        raise AttributeError(name)


@pytest.fixture()
def exporter(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("B12X_COMPILE_DISK_CACHE", "B12X_COMPILE_MEMORY_CACHE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", types.SimpleNamespace(REVISION="test"))
    package = types.ModuleType("b12x.integration.cuteafd")
    for name in ("glmf", "dsv4_mhc", "glm_sparse_mla"):
        module = _Compilers(f"b12x.integration.cuteafd.{name}")
        setattr(package, name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    package.glmf.mhc_geometry = lambda g: ("mhc geometry", g)
    monkeypatch.setitem(sys.modules, "b12x", types.ModuleType("b12x"))
    monkeypatch.setitem(sys.modules, "b12x.integration", types.ModuleType("b12x.integration"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd", package)
    spec = importlib.util.spec_from_file_location("export_b12x_dsv4_aot_under_test", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_wide_programs_are_new_m128_stems(exporter):
    base = exporter.glmf_programs(GEOMETRY, 64, 4096, 131072)
    wide = exporter.glmf_wide_decode_programs(GEOMETRY, 64, 128, 131072)
    stems = [stem for stem, *_ in wide]
    assert len(stems) == len(set(stems)) and set(stems) == WIDE_STEMS
    assert not set(stems) & {stem for stem, *_ in base}, "an existing stem would change"
    # The 64-row decode programs they widen are all in the base list.
    assert {stem.replace("_m128", "_m64") for stem in stems if not stem.startswith("kda_commit")} \
        <= {stem for stem, *_ in base}
    for stem, op, params, thunk in wide:
        module, function, args, kwargs = thunk()
        assert args == (("mhc geometry", GEOMETRY) if module == "dsv4_mhc" else GEOMETRY,), stem
        if stem.startswith("kda_commit"):
            # Capacity rows stay the prefill rows' (no scratch); the records hold 128 rows.
            assert op == "kda_commit" and "max_rows" not in params and params["replay_rows"] == 128
            assert kwargs["replay_rows"] == 128 and kwargs["state_dtype"] == ("bfloat16" if "_s16" in stem else "float32")
            assert function == ("compile_glmf_kda_commit_c_aot" if stem.startswith("kda_commit_c")
                                else "compile_glmf_kda_commit_aot")
        else:
            assert params["max_rows"] == 128 and kwargs["max_rows"] == 128, stem


def test_the_wide_programs_keep_the_decode_routes(exporter):
    calls = {stem: thunk() for stem, _, _, thunk in exporter.glmf_wide_decode_programs(GEOMETRY, 64, 128, 131072)}
    kwargs = {stem: call[3] for stem, call in calls.items()}
    assert kwargs["kda_m128"] == {"max_rows": 128, "fp8": True}
    assert kwargs["kda_w8_m128"] == {"max_rows": 128, "fp8_only": "decode"}
    assert kwargs["kda_s16_m128"] == {"max_rows": 128, "fp8": True, "state_dtype": "bfloat16",
                                      "state_rounding": "window"}
    assert kwargs["mhc_post_pre_m128"] == {"max_rows": 128, "route": "decode"}
    # The 128-row bucket keeps one split where the planner would split it maximally.
    assert kwargs["sparse_mla_decode_m128"] == {"route": "decode", "max_rows": 128, "name": "glmf_sparse_mla",
                                                "fp32_partials": True, "full_launch_splits": 1}
    assert kwargs["index_topk_decode_m128"] == {"max_rows": 128, "max_pages": 512, "mode": "decode"}
    assert kwargs["index_producer_c_m128"] == {"max_rows": 128, "verify_rows": 128}
    for stem in ("mla_producer_m128", "o_m128"):
        assert kwargs[stem] == {"max_rows": 128, "fp8_only": "decode"}
    assert kwargs["ffn_i2048_m128"] == {"inter": 2048, "max_rows": 128, "fp8_only": "decode"}
    assert kwargs["ffn_i12288_m128"] == {"inter": 12288, "max_rows": 128, "fp8_only": "decode"}


def test_no_wide_programs_unless_wider_than_the_decode_programs(exporter):
    for decode_rows, wide_rows in ((64, 0), (64, 64), (128, 128), (128, 64)):
        assert exporter.glmf_wide_decode_programs(GEOMETRY, decode_rows, wide_rows, 131072) == []


def test_the_head_split_takes_no_wide_program(exporter):
    stems = [stem for stem, *_ in exporter.glmf_head_split_programs(GEOMETRY, 64, 4096, 131072)]
    assert stems and not any("m128" in stem for stem in stems)


def test_builds_export_the_wide_programs_unless_switched_off():
    """CMake's CUTEAFD_GLMF_WIDE_DECODE_ROWS (default 128) reaches the exporter and its stamp; WIP
    builds pass CUTEAFD_WIP_GLMF_WIDE_DECODE_ROWS through (0 leaves the wide programs out)."""
    cmake = (ROOT / "native" / "CMakeLists.txt").read_text()
    assert re.search(r'set\(CUTEAFD_GLMF_WIDE_DECODE_ROWS "128" CACHE STRING', cmake)
    programs = (ROOT / "native" / "cmake" / "shared" / "dsv4_programs.cmake").read_text()
    assert '--glmf-wide-decode-rows "${CUTEAFD_GLMF_WIDE_DECODE_ROWS}"' in programs
    # The stamp holds the export arguments, so changing the switch exports again.
    assert 'file(GENERATE OUTPUT "${stamp}" CONTENT "${CUTEAFD_DSV4_EXPORT_ARGS}\\n")' in programs
    wip = (ROOT / "scripts" / "build" / "build-wip-artifacts.sh").read_text()
    assert '-DCUTEAFD_GLMF_WIDE_DECODE_ROWS="${CUTEAFD_WIP_GLMF_WIDE_DECODE_ROWS:-128}"' in wip
    assert '"CUTEAFD_WIP_GLMF_WIDE_DECODE_ROWS=${CUTEAFD_WIP_GLMF_WIDE_DECODE_ROWS:-128}"' in (ROOT / "wip.sh").read_text()
    exporter = EXPORTER.read_text()
    assert 'parser.add_argument("--glmf-wide-decode-rows", type=int, default=128,' in exporter
