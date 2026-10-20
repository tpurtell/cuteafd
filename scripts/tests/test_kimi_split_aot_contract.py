"""Split Kimi AOT catalog is opt-in and independent of CUDA imports."""
import ast
from functools import partial
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def test_split_catalog_covers_whole_and_head_split(monkeypatch):
    path = ROOT / "python/tools/aot/export_b12x_kimi_split_aot.py"
    tree = ast.parse(path.read_text())
    for module, names in (
        ("_kda_contract", ("KDAGeometry",)),
        ("kimi_mla", ("KimiMlaGeometry",)),
        ("kimi_split", ("compile_kimi_kda_core_aot", "compile_kimi_mla_rank_norm_aot",
                        "compile_kimi_mla_query_aot", "compile_kimi_mla_gate_output_aot"))):
        fake = types.ModuleType("b12x.integration.cuteafd." + module)
        for name in names:
            setattr(fake, name, lambda *args, **kw: (args, kw))
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
    namespace = {"partial": partial}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                           type_ignores=[]), str(path), "exec"), namespace)
    entries = namespace["programs"]()
    assert len(entries) == 20
    assert len({name for name, _, _ in entries}) == 20
    for heads in (48, 96):
        for rows in (64, 4096):
            selected = {name: thunk for name, thunk, capacity in entries
                        if f"h{heads}_" in name and capacity == rows}
            assert len(selected) == 5
            assert selected[f"kda_core_h{heads}_m{rows}"]()[1] == {
                "max_rows": rows, "decode": rows == 64}
            for stride in (576, 640):
                assert selected[f"mla_rank_norm_h{heads}_kv{stride}_m{rows}"]()[1]["kv_stride"] == stride


def test_split_has_dedicated_sm_comparison_group():
    spec = importlib.util.spec_from_file_location("sm_exports", ROOT / "scripts/build/compare-sm-exports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert next(j for j in module.jobs("all") if j[0] == "kimi-split") == (
        "kimi-split", "export_b12x_kimi_split_aot.py", [])
