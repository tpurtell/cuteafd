"""CPU-only EXLR8 ABI/export registration checks, without importing CUDA."""
import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FORK = ROOT / "third_party/sparkinfer"


def contract():
    # The geometry helpers deliberately have no torch/CUDA dependence.
    tree = ast.parse((FORK / "b12x/integration/cuteafd/exlr8.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("geometry", "scratch_bytes")]
    namespace = {"ABI_VERSION": 3, "CTAS_PER_SM": 4}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "exlr8_contract", "exec"), namespace)
    return namespace


def test_capacity_scratch_matches_native_v3():
    api = contract()
    g = api["geometry"]()
    assert g == dict(abi_version=3, hidden=6144, units=16, top_k=8, max_groups=512,
                     max_rows=64, split_k_blocks=8, fc1_splits=6, ctas_per_sm=4)
    assert api["scratch_bytes"](g) == dict(fc1=805306368, intermediate=33554432, fc2=201326592)


@pytest.mark.parametrize("kwargs", [dict(hidden=127), dict(units=0), dict(top_k=-1),
    dict(max_groups=2**31), dict(hidden=True), dict(units=2**31-1,max_groups=2**31-1),
    dict(top_k=2**26), dict(max_groups=2**27), dict(units=2**24), dict(hidden=2**30)])
def test_invalid_geometry_rejected(kwargs):
    with pytest.raises(ValueError):
        contract()["geometry"](**kwargs)


def test_exlr8_has_separate_comparison_group():
    spec = importlib.util.spec_from_file_location("sm_exports", ROOT / "scripts/build/compare-sm-exports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs = [j for j in module.jobs("all") if j[0] == "exlr8"]
    assert jobs == [("exlr8", "export_b12x_dsv4_aot.py", ["--geometry", "exlr8"])]


def test_serving_and_glue_export_inventory():
    tree = ast.parse((FORK / "b12x/integration/cuteafd/exlr8_glue.py").read_text())
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "exlr8_export_programs")
    result = next(n.value for n in builder.body if isinstance(n, ast.Assign) and n.targets[0].id == "result")
    assert [item.elts[0].value for item in result.elts] == [
        "a16_decode", "a16_serving", "group", "shared_a16_serving", "shared_group"]
    for index in (3, 4):
        shared = {k.arg: k.value.value for k in result.elts[index].elts[-1].body.keywords
                  if isinstance(k.value, ast.Constant)}
        assert shared["shared_expert"] is True
        assert shared["top_k"] == 1 and shared["max_groups"] == 4
    assert shared["experts"] == 1
    namespace = {"compile_exlr8_group_aot": lambda **kwargs: None}
    # Drop the local CUDA-dependent import; constructing the lambdas compiles nothing.
    builder.body = [node for node in builder.body if not isinstance(node, ast.ImportFrom)]
    exec(compile(ast.Module(body=[builder], type_ignores=[]), "exlr8_inventory", "exec"), namespace)
    inventory = namespace["exlr8_export_programs"]
    assert [p[0] for p in inventory("spark")] == ["a16_serving", "group"]
    assert [p[0] for p in inventory("coordinator")] == ["shared_a16_serving", "shared_group"]
    with pytest.raises(ValueError):
        inventory("unknown")


def test_no_changes_to_existing_export_builders():
    # The new family remains opt-in; existing geometry lists are unchanged.
    spec = importlib.util.spec_from_file_location("sm_exports", ROOT / "scripts/build/compare-sm-exports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "exlr8" not in module.GEOMETRIES
    assert "exlr8" not in module.EXL3_GEOMETRIES
