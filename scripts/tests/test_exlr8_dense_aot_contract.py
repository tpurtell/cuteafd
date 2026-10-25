"""Dense operand export catalog checks without importing CUDA."""
import ast
from functools import partial
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def catalog(monkeypatch):
    path = ROOT / "python/tools/aot/export_b12x_exlr8_dense_aot.py"
    tree = ast.parse(path.read_text())
    namespace = {"partial": partial}
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float32="torch.float32"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    for name, compilers in (
        ("trellis_linear", ("compile_trellis_linear_aot",)),
        ("scaled_linear", ("compile_scaled_linear_aot",)),
        ("dense_mxfp8", ("compile_dense_mxfp8_aot", "compile_dense_mxfp8_quant_aot")),
        ("fp8_row_embedding", ("compile_fp8_row_embedding_aot",)),
        ("trellis_unpack", ("compile_trellis_unpack_e4m3_aot",)),
        ("trellis_transform", ("compile_trellis_input_aot", "compile_trellis_output_aot")),
        ("native_dense", ("compile_trellis_fp8_prefill_aot", "compile_swiglu_split_aot",
                          "compile_f32_narrow_pack_aot")),
        ("pointwise", ("compile_bf16_projection_aot",)),
    ):
        fake = types.ModuleType("b12x.integration.cuteafd." + name)
        for compiler in compilers:
            setattr(fake, compiler, lambda **kwargs: kwargs)
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["programs"]()


def test_dense_trellis_catalog_has_native_full_shapes(monkeypatch):
    entries = catalog(monkeypatch)[:4]
    assert [name for name, _, _, _ in entries] == [
        "trellis_q_b", "trellis_o", "trellis_gate_up", "trellis_down"]
    assert [(p["input_features"], p["output_features"], p["sharded_side"])
            for _, _, p, _ in entries] == [
        (2048, 16384, 1), (16384, 6144, 2), (6144, 12288, 1), (12288, 6144, 2)]
    for _, op, params, thunk in entries:
        assert op == "trellis_linear"
        assert params["max_rows"] == 4096
        assert params["bits"] == 4 and params["codebook"] == "e4m3"
        assert params["rotation_count"] == 4
        assert thunk() == params


def test_dense_a8_catalog_is_additive_and_quantizes_each_width_once(monkeypatch):
    entries = catalog(monkeypatch)
    assert all(not name.endswith("_a8") for name, *_ in entries[:63])
    cores = [(name, p, thunk) for name, op, p, thunk in entries if op == "dense_mxfp8"]
    assert len(cores) == 8
    assert {p["input_features"] for _, p, _ in cores} == {2048, 6144, 8192, 12288, 16384}
    for name, params, thunk in cores:
        assert name.endswith("_a8") and params["input_dtype"] == "f16"
        assert params["encoding"] == "trellis" and params["rotation_count"] == 4
        assert thunk() == params
    quant = [(name, p, thunk) for name, op, p, thunk in entries if op == "dense_mxfp8_quant"]
    assert len(quant) == 5
    assert {p["input_features"] for _, p, _ in quant} == {p["input_features"] for _, p, _ in cores}
    assert all(thunk() == params for _, params, thunk in quant)


def test_dense_tp2_catalog_halves_only_the_owned_axis(monkeypatch):
    entries = {name: (op, params, thunk)
               for name, op, params, thunk in catalog(monkeypatch)}
    assert len(entries) == 76
    for name in ("trellis_q_b", "trellis_o", "trellis_gate_up", "trellis_down",
                 "int4_q_b", "int4_o"):
        op, full, _ = entries[name]
        half_op, half, thunk = entries[name + "_tp2"]
        axis = "output_features" if full["sharded_side"] == 1 else "input_features"
        expected = dict(full, **{axis: full[axis] // 2})
        assert half_op == op
        assert half == expected
        assert half["max_rows"] == 4096
        assert thunk() == half
    assert not any(name.endswith("_tp2") and name.startswith(("mxfp8", "fp8_row"))
                   for name in entries)


def test_dense_has_opt_in_sm_comparison_group(monkeypatch):
    spec = importlib.util.spec_from_file_location("sm_exports", ROOT / "scripts/build/compare-sm-exports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name, exporter, args = next(j for j in module.jobs("all") if j[0] == "exlr8-dense-trellis")
    assert exporter == "export_b12x_dsv4_aot.py"
    assert args[:2] == ["--geometry", "exlr8_dense"]
    assert args[3].split(",") == [name for name, _, _, _ in catalog(monkeypatch)[:4]]
    groups = [j for j in module.jobs("all") if j[0].startswith("exlr8-dense-")]
    selected = [name for _, _, args in groups for name in args[3].split(",")]
    assert sorted(selected) == sorted(name for name, _, _, _ in catalog(monkeypatch))
    assert "exlr8_dense" not in module.GEOMETRIES

def test_unpacked_prefill_only_covers_native_trellis(monkeypatch):
    entries = {name: (op, params) for name, op, params, _ in catalog(monkeypatch)}
    unpacked = [name.removeprefix("unpack_") for name in entries if name.startswith("unpack_")]
    assert len(unpacked) == 8
    for name in unpacked:
        _, core = entries["trellis_" + name]
        op, unpack = entries["unpack_" + name]
        assert op == "trellis_unpack_e4m3"
        assert unpack == {k: core[k] for k in ("input_features", "output_features", "sharded_side")}
        op, prefill = entries["prefill_" + name]
        assert op == "trellis_fp8_prefill"
        assert prefill == {k: core[k] for k in ("input_features", "output_features", "max_rows")}
    assert not any(name.startswith(("unpack_int4", "unpack_mxfp8")) for name in entries)
    assert entries["bf16_index_weights"][1] == dict(width=6144, outputs=32, output_dtype="torch.float32")


def test_exlr8_coordinator_enables_dense_and_both_stage_geometries():
    text = (ROOT / "native/cmake/shared/dsv4_programs.cmake").read_text()
    assert 'if(CUTEAFD_EXLR8_ROLE STREQUAL "coordinator")' in text
    assert '${CUTEAFD_PROGRAM_GEOMETRY},exlr8_dense,glm_stage,glm2_stage' in text


def test_bf16_core_is_static_and_matches_f16_geometry(monkeypatch):
    entries = {name: (op, params, thunk)
               for name, op, params, thunk in catalog(monkeypatch)}
    for name, (op, params, thunk) in entries.items():
        if name.startswith("trellis_") and name.endswith("_bf16") and op == "trellis_linear":
            base = entries[name.removesuffix("_bf16")][1]
            assert params == dict(base, input_dtype="bf16")
            assert thunk() == params
    assert sum(op == "trellis_linear" and p.get("input_dtype") == "bf16"
               for op, p, _ in entries.values()) == 8


def test_native_host_admission_uses_shared_working_set_budget():
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5/engine/exlr8.rs").read_text()
    family = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5/mod.rs").read_text()
    shared = "cuteafd_loader::formats::mapped_preload::host_memory()?"
    assert shared in engine.split("fn admit_host(", 1)[1].split("pub(crate) struct NativeExperts", 1)[0]
    assert shared in family.split("fn load_native<", 1)[1]
    assert "fn host_available(" not in engine
    assert 'read_to_string("/proc/meminfo")' not in engine
