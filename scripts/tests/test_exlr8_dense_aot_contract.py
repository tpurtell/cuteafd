"""Dense operand export catalog checks without importing CUDA."""
import ast
from functools import partial
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def catalog(monkeypatch, families=("glm", "glmf")):
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
        ("trellis_transform", ("compile_trellis_input_aot", "compile_trellis_output_aot")),
        ("native_dense", ("compile_swiglu_split_aot",
                          "compile_f32_narrow_pack_aot")),
        ("pointwise", ("compile_bf16_projection_aot",)),
    ):
        fake = types.ModuleType("b12x.integration.cuteafd." + name)
        for compiler in compilers:
            setattr(fake, compiler, lambda **kwargs: kwargs)
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["programs"](families=families)


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
    entries = catalog(monkeypatch, ("glm",))
    first_a8 = next(i for i, (_, op, _, _) in enumerate(entries) if op == "dense_mxfp8")
    assert all(not name.endswith("_a8") for name, *_ in entries[:first_a8])
    assert all(op in ("dense_mxfp8", "dense_mxfp8_quant") for _, op, _, _ in entries[first_a8:])
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
               for name, op, params, thunk in catalog(monkeypatch, ("glm",))}
    assert len(entries) == 60
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
    groups = [j for j in module.jobs("all") if j[0].startswith("exlr8-dense-")
              and not j[0].startswith("exlr8-dense-scope-")]
    for group, _, args in groups:
        assert args[-2:] == ["--dense-trellis-families",
                             "glm,glmf" if "-geometry-" in group else "glm"]
    scopes = [j for j in module.jobs("all") if j[0].startswith("exlr8-dense-scope-")]
    assert [args for _, _, args in scopes] == [
        ["--geometry", "exlr8_dense", "--dense-trellis-families", family] for family in ("glm", "glmf")]
    selected = [name for _, _, args in groups for name in args[3].split(",")]
    assert sorted(selected) == sorted(name for name, _, _, _ in catalog(monkeypatch))
    assert "exlr8_dense" not in module.GEOMETRIES

def test_exlr8_coordinator_enables_dense_and_both_stage_geometries(tmp_path):
    import subprocess

    cmake = ROOT / "native/cmake/shared/dsv4_programs.cmake"
    # The shared stages must be present exactly once with or without GLM.
    for glm in (False, True):
        source = tmp_path / str(glm)
        source.mkdir()
        (source / "CMakeLists.txt").write_text("\n".join([
            "cmake_minimum_required(VERSION 3.18)",
            "project(exlr8_scope NONE)",
            "set(CUTEAFD_CUDA_ARCHITECTURES 120)",
            "set(CUTEAFD_CONTEXT_SPLIT_AOT OFF)",
            'set(CUTEAFD_GLM_GEOMETRIES "glm;glm2")',
            f'set(CUTEAFD_ENABLE_GLM_AOT {"ON" if glm else "OFF"})',
            "set(CUTEAFD_ENABLE_EXLR8_AOT ON)",
            "set(CUTEAFD_EXLR8_FAMILIES glm)",
            "add_custom_target(cuteafd_verify_sparkinfer_source)",
            f'include("{cmake}")',
            'file(WRITE "${CMAKE_BINARY_DIR}/scope.txt" "${CUTEAFD_PROGRAM_GEOMETRY}")',
        ]))
        build = source / "build"
        result = subprocess.run(["cmake", "-S", str(source), "-B", str(build)],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        geometries = (build / "scope.txt").read_text().split(",")
        assert len(geometries) == len(set(geometries))
        for geometry in ("exlr8", "exlr8_dense", "glm_stage", "glm2_stage"):
            assert geometries.count(geometry) == 1
        assert ("glm" in geometries) == glm


def test_bf16_core_is_static_and_matches_f16_geometry(monkeypatch):
    entries = {name: (op, params, thunk)
               for name, op, params, thunk in catalog(monkeypatch)}
    for name, (op, params, thunk) in entries.items():
        if name.startswith("trellis_") and name.endswith("_bf16") and op == "trellis_linear":
            base = entries[name.removesuffix("_bf16")][1]
            assert params == dict(base, input_dtype="bf16")
            assert thunk() == params
    assert sum(op == "trellis_linear" and p.get("input_dtype") == "bf16"
               for op, p, _ in entries.values()) == 17


def test_geometry_catalog_is_deduplicated_and_a16_only(monkeypatch):
    import hashlib
    import json

    entries = catalog(monkeypatch)
    legacy = [(name, op, params) for name, op, params, _ in entries[:60]]
    assert hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest() == (
        "a35a432bf52b58c0611ae59aa05bddcc27eadbacc83e0d199077cd53fef6addd")
    assert len(entries) == 102
    assert len({name for name, *_ in entries}) == len(entries)
    geometries = []
    for name, op, params, thunk in entries:
        if op in ("trellis_linear", "scaled_linear", "fp8_row_embedding"):
            geometries.append((op, tuple(sorted(params.items()))))
        assert thunk() == params
    assert len(geometries) == len(set(geometries))
    new = entries[60:]
    assert len([1 for _, op, _, _ in new if op == "trellis_linear"]) == 18
    assert len([1 for _, op, p, _ in new if op == "scaled_linear" and p["encoding"] == "mxfp8"]) == 10
    assert not any("glmf" in name or name.endswith("_a8") for name, *_ in new)
    assert not any(op in ("dense_mxfp8", "dense_mxfp8_quant") for _, op, _, _ in new)
    shapes = {(p["input_features"], p["output_features"], p["sharded_side"])
              for _, op, p, _ in new if op == "scaled_linear"}
    assert (4096, 32, 1) in shapes and (128, 8192, 1) in shapes
    assert sum(name == "trellis_k8192_n4096_s2" for name, *_ in new) == 1


def test_dense_family_scope_preserves_legacy_and_unions_shared_geometry(monkeypatch):
    import hashlib
    import json
    import pytest

    glm = catalog(monkeypatch, ("glm",))
    flash = catalog(monkeypatch, ("glmf",))
    union = catalog(monkeypatch, ("glm", "glmf"))
    triples = lambda entries: [(name, op, params) for name, op, params, _ in entries]
    assert len(glm) == 60
    assert hashlib.sha256(json.dumps(triples(glm), sort_keys=True).encode()).hexdigest() == (
        "a35a432bf52b58c0611ae59aa05bddcc27eadbacc83e0d199077cd53fef6addd")
    assert triples(union[:len(glm)]) == triples(glm)
    assert len(flash) == 58
    assert len(union) == 102
    glm_names = {name for name, *_ in glm}
    flash_names = {name for name, *_ in flash}
    assert glm_names | flash_names == {name for name, *_ in union}
    reused = glm_names & flash_names
    assert len(reused) == 16
    assert {"swiglu_split_i6144", "swiglu_split_i12288",
            "f32_narrow_pack_n128_p160", "f32_narrow_pack_n32_p160"} <= reused
    assert not any(op in ("dense_mxfp8", "dense_mxfp8_quant") for _, op, _, _ in flash)
    assert triples(catalog(monkeypatch, ("glmf", "glm", "glmf"))) == triples(union)
    for families in ((), ("kimik3",), ("glm", "unknown")):
        with pytest.raises(ValueError, match="families"):
            catalog(monkeypatch, families)


def test_flash_index_companions_pack_native_bf16_weights_without_quantization(monkeypatch):
    entries = {name: (op, params, thunk)
               for name, op, params, thunk in catalog(monkeypatch, ("glmf",))}
    for outputs in (32, 128):
        op, params, thunk = entries[f"bf16_projection_k4096_n{outputs}"]
        assert op == "bf16_projection"
        assert params == {"width": 4096, "outputs": outputs, "output_dtype": "torch.float32"}
        assert thunk() == params
    # MXFP8 key projection occupies columns0:128; native BF16 weights/gate
    # projections fill128:160 and160:288 without touching those key columns.
    for width, column in ((32, 128), (128, 160)):
        op, params, thunk = entries[f"f32_narrow_pack_n{width}_p288"]
        assert op == "f32_narrow_pack"
        assert params == {"width": width, "pitch": 288}
        assert column + width <= params["pitch"]
        assert thunk() == params
    glm_names = {name for name, *_ in catalog(monkeypatch, ("glm",))}
    assert "bf16_projection_k4096_n128" not in glm_names
    assert "f32_narrow_pack_n128_p288" not in glm_names
    assert "f32_narrow_pack_n32_p288" not in glm_names


def test_shared_scope_knob_keeps_k6_default_and_native_family_fallback():
    import pytest

    path = ROOT / "python/tools/aot/export_b12x_dsv4_aot.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "dense_family_scopes")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    scope = namespace["dense_family_scopes"]
    assert scope(None, "glm") == (("kimik3",), ("glm",))
    assert scope(None, "glmf") == (("kimik3",), ("glmf",))
    assert scope("glm", "glmf") == (("glm",), ("glm",))
    assert scope("glmf", "glm") == (("glmf",), ("glmf",))
    assert scope("glmf,kimik3,glm,glm", "glm") == (("kimik3", "glm", "glmf"), ("glm", "glmf"))
    assert scope("kimik3", "glm") == (("kimik3",), ())
    assert scope("step5", "glm") == (("step5",), ())
    assert scope("ml4", "glm") == (("ml4",), ())
    assert scope("ml4,glmf,step5,glm", "glm") == (("glm", "glmf", "step5", "ml4"), ("glm", "glmf"))
    assert scope("step5,glmf,glm", "glm") == (("glm", "glmf", "step5"), ("glm", "glmf"))
    for families in ("", "glm,", "other"):
        with pytest.raises(ValueError, match="dense-trellis-families"):
            scope(families, "glm")


def test_native_dense_scope_preserves_existing_table_metadata():
    path = ROOT / "python/tools/aot/export_b12x_dsv4_aot.py"
    tree = ast.parse(path.read_text())
    branch = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                  and ast.unparse(node.test) == "name == 'exlr8_dense'")
    assignment = next(node for node in branch.body if isinstance(node, ast.Assign))
    assert ast.literal_eval(assignment.value) == {"abi_version": 1}


def test_native_host_admission_uses_shared_working_set_budget():
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/shared/experts/exlr8/shared.rs").read_text()
    family = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5/mod.rs").read_text()
    shared = "cuteafd_loader::formats::mapped_preload::host_memory()?"
    assert shared in engine.split("fn admit_host(", 1)[1].split("pub(crate) struct NativeExperts", 1)[0]
    assert shared in family.split("fn load_native<", 1)[1]
    assert "fn host_available(" not in engine
    assert 'read_to_string("/proc/meminfo")' not in engine


def test_native_shared_owner_is_family_neutral_and_glm_keeps_its_buffers():
    shared = (ROOT / "rust/crates/cuteafd-daemon/src/shared/experts/exlr8/shared.rs").read_text()
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5/engine.rs").read_text()
    family = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5/mod.rs").read_text()
    assert "family: KernelFamily" in shared
    assert "shared_workspace_bytes(programs, family, ROWS)" in shared
    assert "programs, self.family, ROWS, self.workspace" in shared
    assert "Workspace" not in shared and "KernelFamily::Glm" not in shared
    assert "execution.seal_graph_storage()?" in shared
    assert "w.x.buffer,\n                w.shared.buffer" in engine
    assert "cuteafd_ffi::exlr8::KernelFamily::Glm,\n                devices.len(), rank" in family
