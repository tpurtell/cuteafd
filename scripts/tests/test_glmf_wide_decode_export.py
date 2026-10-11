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
    for name in ("glmf", "dsv4_mhc", "glm_sparse_mla", "context_split", "qwen4", "qwen4_gdn", "qwen4_attention"):
        module = _Compilers(f"b12x.integration.cuteafd.{name}")
        setattr(package, name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    package.glmf.mhc_geometry = lambda g: ("mhc geometry", g)
    package.context_split.sparse_mla_partial_split_plan = lambda g, **kw: 1 if kw["sm_count"] == 170 else 4
    monkeypatch.setitem(sys.modules, "b12x", types.ModuleType("b12x"))
    monkeypatch.setitem(sys.modules, "b12x.integration", types.ModuleType("b12x.integration"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd", package)
    spec = importlib.util.spec_from_file_location("export_b12x_dsv4_aot_under_test", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("geometry,checkpoint,limit", [
    ("flash", "deepseek-ai/DeepSeek-V4-Flash-0731", 1048576),
    ("pro", "deepseek-ai/DeepSeek-V4-Pro-0813", 1048576),
    ("glm", "zai-org/GLM-5.3", 1048576),
    ("glmf", "zai-org/GLM-5.3-Flash", 1048576),
    ("qwen4", "Qwen/Qwen3.8-Flash-Next", 262144),
])
def test_checkpoint_extent_table_and_head_splits(exporter, geometry, checkpoint, limit):
    assert exporter.CHECKPOINT_CONTEXTS[geometry] == (checkpoint, limit)
    for name in (geometry, geometry + "2") if geometry != "qwen4" else (geometry,):
        assert exporter.geometry_context(name, 131072) == 131072
        assert exporter.geometry_context(name, 1048576) == limit
        assert exporter.geometry_context(name, 2097152) == limit
    with pytest.raises(ValueError, match="positive"):
        exporter.geometry_context(geometry, 0)


def test_qwen_fp8_kv_exports_preserve_bf16_and_weight_variants(exporter):
    programs = exporter.qwen4_programs(GEOMETRY, 64, 4096, 262144)
    calls = {stem: thunk()[3] for stem, _, _, thunk in programs}
    assert len(calls) == len(programs)
    for cap in (64, 4096):
        for stem in ("attn_producer", "sparse_gqa"):
            assert "kv_format" not in calls[f"{stem}_m{cap}"]
            assert calls[f"{stem}_kv_fp8_m{cap}"] == {"max_rows": cap, "kv_format": "fp8"}
        mode = "decode" if cap == 64 else "prefill"
        assert calls[f"attn_producer_w8_kv_fp8_m{cap}"] == {
            "max_rows": cap, "fp8_only": mode, "kv_format": "fp8"}
    assert calls["attn_producer_fp8_kv_fp8_m64"] == {"max_rows": 64, "fp8": True, "kv_format": "fp8"}


def test_wide_index_uses_clamped_extent(exporter):
    extent = exporter.geometry_context("glmf", 2097152)
    base = exporter.glmf_programs(GEOMETRY, 64, 4096, extent)
    wide = exporter.glmf_wide_decode_programs(GEOMETRY, 64, 128, extent)
    for _, op, params, thunk in base + wide:
        if op == "index_topk":
            assert params["max_pages"] == 4096
            assert thunk()[3]["max_pages"] == 4096


def test_builds_pass_the_extent_explicitly(exporter):
    cmake = (ROOT / "native/CMakeLists.txt").read_text()
    assert 'set(CUTEAFD_DSV4_MAX_CONTEXT "1048576" CACHE STRING' in cmake
    assert exporter.DEFAULT_MAX_CONTEXT == 1048576
    for mode, entry in (("RELEASE", "build.sh"), ("WIP", "wip.sh")):
        variable = f'CUTEAFD_{mode}_DSV4_MAX_CONTEXT'
        assert f'-DCUTEAFD_DSV4_MAX_CONTEXT="${{{variable}:-1048576}}"' in (
            ROOT / f'scripts/build/build-{mode.lower()}-artifacts.sh').read_text()
        assert f'-e "{variable}=${{{variable}:-1048576}}"' in (ROOT / entry).read_text()
    source = EXPORTER.read_text()
    assert '"max_context": args.max_context' in source
    assert 'manifest["family_capacities"][family] = {"max_context": extent}' in source
    assert 'make(g, args.decode_rows, args.prefill_rows, extent)' in source


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


@pytest.mark.parametrize("family,heads,index_topk,pool,record_bytes", [
    ("glm", 64, 2048, 1, 656), ("glmf", 64, 2048, 4, 528),
    ("flash", 64, 512, 4, 584), ("pro", 128, 1024, 4, 584),
])
def test_context_split_opt_in_geometry(exporter, family, heads, index_topk, pool, record_bytes):
    g = types.SimpleNamespace(heads=heads, index_topk=index_topk, record_bytes=record_bytes,
                              page_rows=64, kv_page_bytes=record_bytes*64)
    if family == "glmf":
        g.index_kpool = pool
    if family in ("flash", "pro"):
        g.o_groups = 8
    work = exporter.context_split_programs(g, 64, 1048576)
    calls = {stem: (params, thunk()) for stem, _, params, thunk in work}
    assert len(calls) == len(work)
    assert calls["dsa_candidate_merge"][1][3]["topk"] == (index_topk//pool if family=="glmf" else index_topk)
    assert calls["lse_combine2"][1][3] == {"heads":heads//2,"has_sink":family in ("flash","pro")}
    assert calls["scored_index_topk_decode_m64"][1][3]["max_pages"] == 1048576//pool//64
    for begin in (0,heads//2):
        for splits, sms in ((1,170),(4,188)):
            params, call = calls[f"sparse_mla_partial_h{begin}_s{splits}_m64"]
            assert call[3] == {"max_rows":64,"head_begin":begin,"head_count":heads//2,"num_splits":splits}
            assert params["device_sm_counts"] == [sms]
    layouts = {stem: call[3]["page_bytes"] for stem, (_, call) in calls.items() if "gather" in stem}
    assert layouts["paged_staging_gather_index"] == 8448
    if family in ("flash","pro"):
        assert layouts["paged_staging_gather_c4"] == 37440
        assert layouts["paged_staging_gather_swa"] == 149760
    else:
        assert layouts["paged_staging_gather_kv"] == record_bytes*64
