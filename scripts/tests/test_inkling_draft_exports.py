"""Recurrent draft export inventory without importing CUDA or the fork."""
import ast
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def catalog(monkeypatch):
    module = types.ModuleType("b12x.integration.cuteafd.inkling_drafter")
    for name in ("dflare_mix", "mask_init", "recurrent_gate", "head_fold", "logit_finish", "residual_norm",
                 "qk_norm_rope", "silu_mul", "bias_add", "kv_store"):
        setattr(module, f"compile_inkling_{name}_aot", lambda _name=name, **kwargs: (_name, kwargs))
    package = types.ModuleType("b12x.integration.cuteafd")
    package.inkling_drafter = module
    monkeypatch.setitem(sys.modules, package.__name__, package)
    path = ROOT / "python/tools/aot/export_b12x_dsv4_aot.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("inkling_draft_programs", "inkling_draft_projection_shapes")]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_recurrent_exports_cover_all_ten_primitives_and_rank_variants(monkeypatch):
    exports = catalog(monkeypatch)["inkling_draft_programs"]()
    assert len(exports) == 18
    by_name = {name: (op, params, compile_) for name, op, params, compile_ in exports}
    assert len(by_name) == len(exports)
    for q, k in ((48, 8), (24, 4), (0, 8), (0, 4)):
        op, params, compile_ = by_name[f"qk_norm_rope_q{q}_k{k}"]
        assert compile_() == ("qk_norm_rope", {"q_heads": q, "kv_heads": k})
        assert params == {"max_rows": 4096}
    for heads in (8, 4):
        for fmt in ("fp8", "bf16"):
            assert by_name[f"kv_store_k{heads}_{fmt}"][2]() == ("kv_store", {"kv_heads": heads, "kv_format": fmt})
    for name in ("mask_init", "recurrent_gate"):
        assert by_name[name][1] == {"max_rows": 64}
    for name in ("logit_finish", "bias_add_w768_f32", "bias_add_w201024_bf16"):
        assert by_name[name][1] == {"max_rows": 448}


def test_recurrent_projection_shapes_include_native_head_and_tp2_columns(monkeypatch):
    shapes = catalog(monkeypatch)["inkling_draft_projection_shapes"]
    whole, split = shapes(1), shapes(2)
    for expected in ((4608, 6144), (4608, 1024), (6144, 4608), (4608, 18432), (18432, 4608),
                     (9728, 768), (256, 201024), (4608, 201024)):
        assert expected in whole
    for expected in ((4608, 3072), (4608, 512), (3072, 4608), (4608, 9216), (9216, 4608)):
        assert expected in split
def test_fir_exports_keep_wide_prefill_separate_from_bounded_replay(monkeypatch):
    path = ROOT / "python/tools/aot/export_b12x_dsv4_aot.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "inkling_fir_programs")
    for module_name, compiler in (("fir_history", "compile_fir_history_commit_aot"),
                                  ("inkling2_convolution_replay", "compile_inkling2_convolution_replay_aot"),
                                  ("inkling2_convolution_strided", "compile_inkling2_convolution_strided_aot")):
        module = types.ModuleType("b12x.integration.cuteafd."+module_name)
        setattr(module, compiler, lambda _name=compiler, **kw: (_name, kw))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    exports = namespace["inkling_fir_programs"]()
    assert len(exports) == 17
    assert exports[0][0] == "fir_history_commit" and exports[0][2] == {"max_rows": 64}
    for name, op, params, compile_ in exports[1:11]:
        assert params["max_rows"] == (64 if name.endswith("replay") else 4096)
        assert compile_()[1]["width"] in (512, 1024, 4608)
        assert op in ("convolution_replay", "convolution_strided")
    assert exports[11][0] == "fir_history_commit_m128" and exports[11][2] == {"max_rows": 128}
    assert exports[11][3]() == ("compile_fir_history_commit_aot", {"max_rows": 128})
    for name, op, params, compile_ in exports[12:]:
        assert name.endswith("_replay_m128") and params == {"max_rows": 128}
        assert compile_()[1]["width"] in (512, 1024, 4608)
        assert compile_()[1]["max_rows"] == 128
        assert op == "convolution_replay"


def test_mini_exports_cover_gqa_dflash2():
    path = ROOT / "python/tools/aot/export_b12x_dsv4_aot.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "inkling2_mini_projection_shapes")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    bf16, _ = namespace["inkling2_mini_projection_shapes"]()
    assert {(2304, 13824), (6912, 2304), (9216, 2304), (2304, 576), (2304, 256),
            (2304, 6144), (2304, 2048), (4096, 2304)} <= bf16
