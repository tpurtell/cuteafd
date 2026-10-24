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
    namespace = {"compile_exlr8_group_aot": lambda **kwargs: kwargs, "compile_exlr8_a16_aot": lambda **kwargs: kwargs}
    # Drop the local CUDA-dependent import; constructing the lambdas compiles nothing.
    builder.body = [node for node in builder.body if not isinstance(node, ast.ImportFrom)]
    exec(compile(ast.Module(body=[builder], type_ignores=[]), "exlr8_inventory", "exec"), namespace)
    inventory = namespace["exlr8_export_programs"]
    spark = inventory("spark")
    assert [p[0] for p in spark] == [
        "a16_serving", "group", "a16_grouped", "a8_grouped", "grouped_group",
        "a16_grouped64"]
    a16_grouped = dict(spark[2][2])
    a8_grouped = dict(spark[3][2])
    assert a16_grouped == {"max_rows": 4096} and a8_grouped == {"max_rows": 4096, "activations": "a8"}
    assert spark[2][3]()["grouped"] is True and "activations" not in spark[2][3]()
    assert spark[3][3]() == {**spark[2][3](), "activations": "a8"}
    # The 64-row grouped instance is the only export that must match K1's
    # F32 FC2 partial precision, and it pairs with the existing 64-row glue.
    assert dict(spark[5][2]) == {"max_rows": 64, "fc2_partials": "f32"}
    grouped64 = spark[5][3]()
    assert grouped64["grouped"] is True and grouped64["group_rows"] == 16
    assert grouped64["grouped_rows"] == 64 and grouped64["fc2_partials"] == "f32"
    assert grouped64["max_groups"] == 512
    assert [p[0] for p in inventory("coordinator")] == ["shared_a16_serving", "shared_group"]
    with pytest.raises(ValueError):
        inventory("unknown")


def test_native_worker_times_routing_experts_and_reduction_before_existing_drain():
    worker = (ROOT / "rust/crates/cuteafd-daemon/src/shared/experts/service/exlr8.rs").read_text()
    assert "WorkerTiming::from_env(library, config.rank)?" in worker
    execute = worker.split("    fn execute(", 1)[1].split("    /// # Safety", 1)[0]
    mapped = execute.split("if let Some(mapped) = mapped {", 1)[1].split("} else {", 1)[0]
    assert mapped.index("library_copy(") < mapped.index("timing.record(0,")
    assert mapped.index("timing.record(0,") < mapped.index("copy_route_records(")
    host = execute.split("} else {", 2)[2].split("// SAFETY: weights,", 1)[0]
    assert host.index("copy_h2d(execution.route_weights()?") < host.index("timing.record(0,")
    assert execute.count("timing.record(0,") == 2
    assert execute.index("execution.run(") < execute.index("timing.record(1,")
    assert execute.index("timing.record(1,") < execute.index("cuda_stream_synchronize(")
    assert execute.index("cuda_stream_synchronize(") < execute.index("timing.publish(")
    assert 'rows as u32, executor_id, "exlr8"' in execute
    # One existing per-chunk completion plus the unconditional failure drain; no timing wait.
    assert execute.count("cuda_stream_synchronize(") == 2


def test_no_changes_to_existing_export_builders():
    # The new family remains opt-in; existing geometry lists are unchanged.
    spec = importlib.util.spec_from_file_location("sm_exports", ROOT / "scripts/build/compare-sm-exports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "exlr8" not in module.GEOMETRIES
    assert "exlr8" not in module.EXL3_GEOMETRIES
