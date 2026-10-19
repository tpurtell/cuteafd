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
    "kda_w8_m128", "kda_s16_m128", "kda_w8_s16_m128", "mla_producer_m128", "o_m128", "sparse_mla_decode_m128", "ffn_i2048_m128",
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
    for name in ("glmf", "glm_ffn", "glm_attention", "glm_indexer", "dsv4_mhc", "glm_sparse_mla", "context_split", "qwen4", "qwen4_gdn", "qwen4_attention", "qwen_dense", "kimi_mla", "kimi_residual", "pointwise"):
        module = _Compilers(f"b12x.integration.cuteafd.{name}")
        setattr(package, name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    package.kimi_mla.KimiMlaGeometry = lambda **kw: types.SimpleNamespace(**kw)
    package.glmf.mhc_geometry = lambda g: ("mhc geometry", g)
    package.dsv4_mhc.mhc_post_pre_route = lambda rows: "decode" if rows < 96 else "block_m"
    # The real planner's shape at 32 heads: more splits for fewer rows (188 SMs: s32/s11/s4).
    package.context_split.sparse_mla_partial_split_plan = lambda g, **kw: {
        (170, 1): 32, (170, 8): 8, (170, 64): 1, (188, 1): 32, (188, 8): 11, (188, 64): 4}[(kw["sm_count"], kw["max_rows"])]
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
    ("qwen35", "Qwen/Qwen3.5-9B", 262144),
    ("qwen38", "Qwen/Qwen3.8-27B", 262144),
    ("step5", "stepfun-ai/Step-5-Preview", 1048576),
])
def test_checkpoint_extent_table_and_head_splits(exporter, geometry, checkpoint, limit):
    assert exporter.CHECKPOINT_CONTEXTS[geometry] == (checkpoint, limit)
    for name in (geometry, geometry + "2") if not geometry.startswith("qwen") else (geometry,):
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
        mode = "decode" if cap == 64 else "prefill"
        for stem in ("attn_producer", "sparse_gqa"):
            assert "kv_format" not in calls[f"{stem}_m{cap}"]
            expected = {"max_rows": cap, "kv_format": "fp8"}
            if stem == "sparse_gqa":
                expected["route"] = mode
                assert calls[f"{stem}_m{cap}"]["route"] == mode
            assert calls[f"{stem}_kv_fp8_m{cap}"] == expected
        assert calls[f"attn_producer_w8_kv_fp8_m{cap}"] == {
            "max_rows": cap, "fp8_only": mode, "kv_format": "fp8"}
    assert calls["attn_producer_fp8_kv_fp8_m64"] == {"max_rows": 64, "fp8": True, "kv_format": "fp8"}


@pytest.mark.parametrize("norm_dtype", ["torch.float32", "torch.bfloat16"])
@pytest.mark.parametrize("decode_rows,prefill_rows", [(64, 4096), (64, 64)])
def test_dense_qwen_exports_only_native_dense_programs(exporter, norm_dtype, decode_rows, prefill_rows):
    programs = exporter.qwen_dense_programs(GEOMETRY, decode_rows, prefill_rows, 262144,
                                             norm_dtype=norm_dtype)
    calls = {stem: (op, params, thunk()) for stem, op, params, thunk in programs}
    caps = set((decode_rows, prefill_rows))
    assert len(calls) == len(programs) == 3 + 5 * len(caps)
    assert set(calls) == {"norm", "add", "gdn_commit"} | {
        f"{op}_m{cap}" for cap in caps for op in ("gdn", "attn_producer", "full_gqa", "attn_o", "ffn")}
    for stem, (op, params, call) in calls.items():
        module, function, args, kwargs = call
        assert module == "qwen_dense" and args == (GEOMETRY,)
        expected_function = "producer" if op == "attn_producer" else op
        assert function == f"compile_qwen_dense_{expected_function}_aot"
        assert params == {k: str(v) if k == "norm_dtype" else v for k, v in kwargs.items()}
    for cap in caps:
        assert calls[f"gdn_m{cap}"][2][3] == {"max_rows": cap, "norm_dtype": norm_dtype}
        assert calls[f"full_gqa_m{cap}"][2][3] == {"max_rows": cap, "max_splits": 32}


def test_glm_declared_static_w8a8_exports_decode_and_prefill(exporter):
    for make in (exporter.glm_programs, exporter.glm_head_split_programs):
        programs = make(GEOMETRY, 64, 4096, 1048576)
        calls = {stem: (params, thunk()[3]) for stem, _, params, thunk in programs}
        for rows in (64, 4096):
            params, kwargs = calls[f"ffn_i12288_pt_m{rows}"]
            expected = {"inter": 12288, "max_rows": rows, "fp8_only": "prefill", "tensor_scales": True}
            assert params == kwargs == expected


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
    assert kwargs["kda_m128"] == {"max_rows": 128, "fp8": True, "seq_ctas": 8, "decode_tiles": True}
    assert kwargs["kda_w8_m128"] == {"max_rows": 128, "fp8_only": "decode"}
    assert kwargs["kda_s16_m128"] == {"max_rows": 128, "fp8": True, "state_dtype": "bfloat16",
                                      "state_rounding": "window", "seq_ctas": 8, "decode_tiles": True}
    assert kwargs["mhc_post_pre_m128"] == {"max_rows": 128, "route": "decode",
                                            "wide_partials_per_cta": 25, "wide_min_rows": 64}
    # Wide execution shares the m64 numerical recipe domain.
    assert kwargs["sparse_mla_decode_m128"] == {"route": "decode", "max_rows": 128, "name": "glmf_sparse_mla",
                                                "fp32_partials": True, "plan_capacity_rows": 64, "sm_count": None}
    mla = next(thunk for stem, _, _, thunk in exporter.glmf_wide_decode_programs(
        GEOMETRY, 64, 128, 131072) if stem == "sparse_mla_decode_m128")
    for sms in (170, 188):
        planned = mla(sm_count=sms)[3]
        assert planned == {**kwargs["sparse_mla_decode_m128"], "sm_count": sms}
    assert kwargs["index_topk_decode_m128"] == {"max_rows": 128, "max_pages": 512, "mode": "decode"}
    assert kwargs["index_producer_c_m128"] == {"max_rows": 128, "verify_rows": 128}
    for stem in ("mla_producer_m128", "o_m128"):
        assert kwargs[stem] == {"max_rows": 128, "fp8_only": "decode"}
    assert kwargs["ffn_i2048_m128"] == {"inter": 2048, "max_rows": 128, "fp8_only": "decode"}
    assert kwargs["ffn_i12288_m128"] == {"inter": 12288, "max_rows": 128, "fp8_only": "decode"}


def test_glmf_decode_partial_fusion_is_opt_in_and_prefill_unchanged(exporter):
    groups = {"glmf": exporter.glmf_programs(GEOMETRY, 64, 4096, 1048576)
              + exporter.glmf_wide_decode_programs(GEOMETRY, 64, 128, 1048576),
              "glmf2": exporter.glmf_head_split_programs(GEOMETRY, 64, 4096, 1048576)}
    changed = set()
    for prefix, programs in groups.items():
        for stem, op, params, thunk in programs:
            if op != "mhc_post_pre":
                continue
            kwargs = thunk()[3]
            if stem in ("mhc_post_pre_m64", "mhc_post_pre_m128"):
                assert params["wide_partials_per_cta"] == kwargs["wide_partials_per_cta"] == 25
                assert params["wide_min_rows"] == kwargs["wide_min_rows"] == 64
                changed.add(f"{prefix}_{stem}")
            else:
                assert "wide_partials_per_cta" not in params
                assert "wide_partials_per_cta" not in kwargs
    assert changed == {"glmf_mhc_post_pre_m64", "glmf_mhc_post_pre_m128"}


def test_only_five_bf16_decode_stems_use_parallel_tiled_kda(exporter):
    groups = {
        "glmf": exporter.glmf_programs(GEOMETRY, 64, 4096, 1048576)
                + exporter.glmf_wide_decode_programs(GEOMETRY, 64, 128, 1048576),
        "glmf2": exporter.glmf_head_split_programs(GEOMETRY, 64, 4096, 1048576),
    }
    qualified = {"glmf_kda_m64", "glmf_kda_m128", "glmf_kda_s16_m64",
                 "glmf_kda_s16_m128", "glmf2_kda_m64"}
    changed = set()
    for prefix, programs in groups.items():
        for stem, _, params, thunk in programs:
            kwargs = thunk()[3]
            name = f"{prefix}_{stem}"
            for key in ("seq_ctas", "decode_tiles"):
                assert (key in params) == (key in kwargs), name
            if name in qualified:
                assert params["seq_ctas"] == kwargs["seq_ctas"] == 8, name
                assert params["decode_tiles"] is kwargs["decode_tiles"] is True, name
                changed.add(name)
            else:
                assert "seq_ctas" not in kwargs and "decode_tiles" not in kwargs, name
    assert changed == qualified


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
    # One variant per split count across (SM count, row cap); each records what it serves.
    for begin in (0,heads//2):
        variants = {stem for stem in calls if stem.startswith(f"sparse_mla_partial_h{begin}_s")}
        assert variants == {f"sparse_mla_partial_h{begin}_s{s}_m64" for s in (1, 4, 8, 11, 32)}
        for splits, caps in ((1, {"170": [64]}), (4, {"188": [64]}), (8, {"170": [8]}), (11, {"188": [8]}),
                             (32, {"170": [1], "188": [1]})):
            params, call = calls[f"sparse_mla_partial_h{begin}_s{splits}_m64"]
            assert call[3] == {"max_rows":64,"head_begin":begin,"head_count":heads//2,"num_splits":splits}
            assert params["row_caps"] == caps
            assert params["device_sm_counts"] == sorted(int(sm) for sm in caps)
    layouts = {stem: call[3]["page_bytes"] for stem, (_, call) in calls.items() if "gather" in stem}
    assert layouts["paged_staging_gather_index"] == 8448
    if family in ("flash","pro"):
        assert layouts["paged_staging_gather_c4"] == 37440
        assert layouts["paged_staging_gather_swa"] == 149760
    else:
        assert layouts["paged_staging_gather_kv"] == record_bytes*64
    # GLM Flash alone: one GPU's selection over its own units (glmf_index_expand_owned).
    assert ("index_expand_owned" in calls) == (family == "glmf")
    if family == "glmf":
        assert calls["index_expand_owned"][1][:2] == ("glmf", "compile_glmf_index_expand_owned_aot")


def test_context_split_programs_append_after_every_other_program_when_switched_on():
    """CUTEAFD_CONTEXT_SPLIT_AOT (default OFF) reaches the exporter; ON appends the primitives after
    the glmf wide programs so no existing stem or table index moves, ONLY exports nothing else."""
    cmake = (ROOT / "native" / "CMakeLists.txt").read_text()
    assert 'set(CUTEAFD_CONTEXT_SPLIT_AOT "OFF" CACHE STRING' in cmake
    programs = (ROOT / "native" / "cmake" / "shared" / "dsv4_programs.cmake").read_text()
    assert '--context-split "${context_split_aot}"' in programs
    source = EXPORTER.read_text()
    assert 'parser.add_argument("--context-split", choices=("off", "on", "only"), default="off",' in source
    wide = source.index("glmf_wide_decode_programs(GLM53_FLASH, args.decode_rows,")
    appended = source.index('if args.context_split == "on":')
    assert appended > wide, "context-split programs must come after every existing program"
    assert 'context_split_programs(\n                bases[name], args.decode_rows, geometry_context(name, args.max_context))' in source
    # glmf2's compact commit (context keeps the compact index cache on two GPUs) comes last too.
    assert source.index('work.append(("glmf2", "kda_commit_c", "kda_commit"') > appended


def test_glmf_w8_state_is_orthogonal_and_canonical(exporter):
    programs = exporter.glmf_programs(GEOMETRY, 64, 4096, 131072)
    programs += exporter.glmf_wide_decode_programs(GEOMETRY, 64, 128, 131072)
    calls = {stem: (params, thunk()[3]) for stem, _, params, thunk in programs}
    for cap, mode in ((64, "decode"), (128, "decode"), (4096, "prefill")):
        params, kw = calls[f"kda_w8_s16_m{cap}"]
        expected = {"max_rows": cap, "fp8_only": mode, "state_dtype": "bfloat16", "state_rounding": "window"}
        assert params == kw == expected
        # Decode and prefill differ only in their row capacity: one canonical FP8 route per
        # load covers the whole decode range, so no recipe key selects a second one.
        assert "decode_recipe" not in params
    assert calls["kda_w8_s16t_m4096"][1]["state_rounding"] == "tile"


@pytest.mark.parametrize("decode_rows,prefill_rows", [(64, 4096), (64, 64)])
def test_kimi_kda_exports_share_glmf_program_and_native_tp2_abi(exporter, decode_rows, prefill_rows):
    programs = exporter.kimi_kda_programs(GEOMETRY, decode_rows, prefill_rows, 1048576)
    kda = [p for p in programs if p[1] in ("kda", "kda_commit")]
    calls = {stem: (op, params, thunk()) for stem, op, params, thunk in kda}
    assert len(calls) == len(kda) == 1 + len({decode_rows, prefill_rows})
    for stem, (op, params, call) in calls.items():
        module, function, args, kwargs = call
        assert module == "glmf" and args == (GEOMETRY,)
        assert function == f"compile_glmf_{op}_aot"
        assert params == kwargs
        if op == "kda":
            assert kwargs["output_kind"] == "norm"
            assert kwargs["decode"] == (kwargs["max_rows"] == decode_rows)
    # KDA reuses the identical shared-FFN down projection, without a second export.
    output = [p for p in programs if p[0] == "ffn_down_i6144"]
    assert len(output) == 1
    _, op, params, thunk = output[0]
    assert op == "bf16_projection" and params == {"width": 6144, "outputs": 7168}
    assert thunk() == ("pointwise", "compile_bf16_projection_aot", (), params)


@pytest.mark.parametrize("decode_rows,prefill_rows", [(64, 4096), (64, 64)])
def test_kimi_nope_mla_exports_both_head_layouts_and_all_modes(exporter, decode_rows, prefill_rows):
    programs = exporter.kimi_kda_programs(GEOMETRY, decode_rows, prefill_rows, 1048576)
    calls = {stem: (op, params, thunk()) for stem, op, params, thunk in programs if op.startswith("mla_") or op == "dense_mla"}
    assert len(calls) == 2 * (2 * len({decode_rows, prefill_rows}) + 3)
    for heads in (48, 96):
        suffix = "" if heads == 48 else "_h96"
        for rows in {decode_rows, prefill_rows}:
            for op in ("mla_producer", "mla_output"):
                actual_op, params, (module, fn, args, kwargs) = calls[f"{op}{suffix}_m{rows}"]
                assert actual_op == op and module == "kimi_mla"
                assert fn == f"compile_kimi_{op}_aot"
                assert vars(args[0]) == {"heads": heads}
                assert kwargs == {"max_rows": rows}
                assert params == {"max_rows": rows, "heads": heads}
        for mode, rows in (("decode", decode_rows), ("verify", decode_rows), ("extend", prefill_rows)):
            op, params, (module, fn, args, kwargs) = calls[f"dense_mla_{mode}{suffix}_m{rows}"]
            assert op == "dense_mla" and module == "kimi_mla" and fn == "compile_kimi_dense_mla_aot"
            assert vars(args[0]) == {"heads": heads}
            assert kwargs == {"max_rows": rows, "max_context": 1048576, "mode": mode}
            assert params == {"mode": mode, "max_rows": rows, "heads": heads}


def test_kimi_w8_exports_every_residual_transition_and_situ_width(exporter):
    programs = exporter.kimi_w8_programs()
    calls = {stem: (op, params, thunk()) for stem, op, params, thunk in programs}
    assert len(calls) == len(programs) == 44
    for stem, (op, params, (module, fn, args, kwargs)) in calls.items():
        assert args == () and kwargs == params
        if op == "bf16_projection":
            assert module == "pointwise" and fn == "compile_bf16_projection_aot"
            width = int(stem.rsplit("_i", 1)[1])
            assert params == ({"width": 7168, "outputs": 2 * width, "compensated": True}
                              if "gate_up" in stem else {"width": width, "outputs": 7168})
            continue
        assert module == "kimi_residual"
        if op == "situ":
            assert fn == "compile_kimi_situ_aot"
            assert params == {"intermediate": int(stem.removeprefix("situ_i"))}
        else:
            assert op == "attn_res_rmsnorm" and fn == "compile_kimi_attn_res_rmsnorm_aot"
            assert params["hidden"] == 7168
            assert params["has_mix"] == (params["candidates"] != 1)
            assert params["append"] == stem.endswith("_append")
    for candidates in range(2, 10):
        for mode in ("keep", "add", "replace"):
            assert f"attn_res_c{candidates}_{mode}" in calls
    assert {params["intermediate"] for _, op, params, _ in programs if op == "situ"} == {
        3072, 6144, 16896, 33792}


def test_kimi_routes_reuse_shared_kernels_at_hidden_and_latent_widths(exporter):
    programs = exporter.kimi_kda_programs(GEOMETRY, 64, 4096, 1048576)
    calls = {stem: (op, params, thunk) for stem, op, params, thunk in programs}
    assert len(calls) == len(programs)
    for stem, geometry in (("router_scores", {"hidden": 7168, "routed_experts": 896, "top_k": 16}),
                           ("expert_input_quant", {"hidden": 3584})):
        op, params, thunk = calls[stem]
        module, function, args, kwargs = thunk()
        assert op == stem and params == {} and kwargs == {}
        assert module == "glmf" and function == f"compile_glmf_{stem}_aot"
        assert len(args) == 1 and vars(args[0]) == geometry
    # KDA output already uses this canonical BF16 projection; no duplicate export.
    op, params, thunk = calls["ffn_down_i6144"]
    assert op == "bf16_projection" and params == {"width": 6144, "outputs": 7168}
    assert thunk() == ("pointwise", "compile_bf16_projection_aot", (), params)


def test_kimi_latent_exports_native_norm_and_canonical_projections(exporter):
    programs = exporter.kimi_latent_programs()
    calls = {stem: (op, params, thunk()) for stem, op, params, thunk in programs}
    assert set(calls) == {"rmsnorm_h3584", "rmsnorm_h7168", "latent_down", "latent_up"}
    for hidden in (3584, 7168):
        assert calls[f"rmsnorm_h{hidden}"] == ("rmsnorm", {"hidden": hidden},
            ("kimi_residual", "compile_kimi_rmsnorm_aot", (), {"hidden": hidden}))
    for direction, width, outputs in (("down", 7168, 3584), ("up", 3584, 7168)):
        params = dict(width=width, outputs=outputs)
        assert calls[f"latent_{direction}"] == ("bf16_projection", params,
            ("pointwise", "compile_bf16_projection_aot", (), params))
