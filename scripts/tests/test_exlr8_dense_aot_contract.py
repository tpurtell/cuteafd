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


def test_minimax_native_mxfp8_a8_catalog_preserves_shipping_exlr8(monkeypatch):
    triples = lambda entries: [(name, op, params) for name, op, params, _ in entries]
    shipping = catalog(monkeypatch)
    native = catalog(monkeypatch, ("minimax",))
    assert len(native) == 12
    assert len({name for name, *_ in native}) == len(native)
    gemms = [(name, params) for name, op, params, _ in native if op == "dense_mxfp8"]
    assert {(p["input_features"], p["output_features"]) for _, p in gemms} == {
        (6144, 8192), (6144, 512), (6144, 128), (8192, 6144),
        (6144, 12288), (12288, 6144), (6144, 3072), (3072, 6144)}
    for name, op, params, thunk in native:
        assert params["input_dtype"] == "bf16"
        assert params["rotation_count"] == 1 and params["max_rows"] == 4096
        assert thunk() == params
        if op == "dense_mxfp8":
            assert params["encoding"] == "mxfp8" and params["sharded_side"] == 1
            assert name == f"mxfp8_k{params['input_features']}_n{params['output_features']}_s1_a8"
        else:
            assert op == "dense_mxfp8_quant"
            assert name == f"quant_k{params['input_features']}_bf16_a8"
    assert {p["input_features"] for _, op, p, _ in native if op == "dense_mxfp8_quant"} == {
        3072, 6144, 8192, 12288}
    union = catalog(monkeypatch, ("minimax", "glmf", "glm", "minimax"))
    assert triples(union) == triples(shipping) + triples(native)
    assert len({name for name, *_ in union}) == len(union)


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


def test_minimax_msa_fixture_matches_pinned_fork_without_cuda():
    import json

    base = ROOT / "third_party/sparkinfer/b12x/integration/cuteafd"
    expected = []
    for module, compiler, extent_name, modes, operation in (
        ("minimax_indexer", "compile_minimax_index_aot", "max_pages",
         ((64, 16384, "decode"), (4096, 16384, "prefill")), "minimax_index"),
        ("minimax_attention", "compile_minimax_attention_aot", "max_batch",
         ((64, 64, "decode"), (4096, 64, "extend")), "minimax_attention"),
    ):
        tree = ast.parse((base / (module + ".py")).read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_layout", compiler)]
        namespace = {
            "torch": types.SimpleNamespace(**{name: name for name in
                     ("float8_e4m3fn", "float32", "uint8", "int32", "bfloat16")}),
            "Operand": lambda name, dtype, shape, role="in", **kwargs:
                dict(name=name, dtype=dtype, shape=shape, role=role),
            "Scalar": lambda name, kind="int32", *args: dict(name=name, type=kind),
            "compile_program": lambda launch, **kwargs: kwargs,
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), module, "exec"), namespace)
        def launch(rows, extent, mode, *args):
            return types.SimpleNamespace(nbytes=namespace["_layout"](rows, extent, mode)[2])
        namespace.update(_Decode=launch, _Prefill=launch, _Attention=launch)
        for rows, extent, mode in modes:
            params = dict(max_rows=rows, mode=mode, **{extent_name: extent})
            program = namespace[compiler](**params)
            expected.append(dict(name=program["name"], family="minimax", op=operation,
                params=params, capacity_rows=rows, geometry=program["geometry"],
                pointers=list(program["operands"]), scalars=list(program["scalars"]),
                scratch_bytes_at_capacity={key: fn(rows) for key, fn in program["scratch"].items()}))
    fixture = ROOT / "rust/crates/cuteafd-ffi/tests/fixtures/minimax_msa_programs.json"
    assert json.loads(fixture.read_text()) == {"programs": expected}


def test_minimax_dispatch_cmake_release_and_placement_scopes_agree(monkeypatch, tmp_path):
    import json
    import subprocess
    import pytest
    from test_wip_scope_exports import aot, cmake_flags

    native = catalog(monkeypatch, ("minimax",))
    tree = ast.parse((ROOT / "python/tools/aot/export_b12x_dsv4_aot.py").read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "'minimax' in geometries")
    fake = types.ModuleType("export_b12x_exlr8_dense_aot")
    fake.minimax_programs = lambda: native
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    for module_name, compiler in (("minimax_indexer", "compile_minimax_index_aot"),
                                  ("minimax_attention", "compile_minimax_attention_aot")):
        module = types.ModuleType("b12x.integration.cuteafd." + module_name)
        setattr(module, compiler, lambda **kwargs: kwargs)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    module = types.ModuleType("b12x.integration.cuteafd.minimax_producer")
    for compiler in ("compile_minimax_norm_rope_aot", "compile_minimax_index_producer_aot",
                     "compile_minimax_oai_split_aot"):
        setattr(module, compiler, lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module = types.ModuleType("b12x.integration.cuteafd.minimax")
    for compiler in ("compile_minimax_norm_aot", "compile_minimax_router_scores_aot",
                     "compile_minimax_kv_pack_aot"):
        setattr(module, compiler, lambda: None)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module = types.ModuleType("b12x.integration.cuteafd.index_cache")
    module.compile_index_cache_gather_aot = lambda: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = {"geometries": ["minimax"], "manifest": {"families": {}, "family_capacities": {}},
                 "work": [], "CHECKPOINT_CONTEXTS": {},
                 "args": types.SimpleNamespace(decode_rows=64, prefill_rows=4096, max_context=1048576)}
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ("geometry_context", "minimax_sparse_programs", "minimax_producer_programs",
                                 "minimax_forward_programs")]
    exec(compile(ast.Module(body=helpers + [branch], type_ignores=[]), "dispatch", "exec"), namespace)
    assert namespace["manifest"]["families"] == {"minimax": {"abi_version": 1}}
    assert namespace["manifest"]["family_capacities"] == {"minimax": {"max_context": 1048576}}
    sparse = namespace["work"][len(native):len(native) + 4]
    assert [(suffix, op, params, thunk()) for _, suffix, op, params, thunk in sparse] == [
        ("index_decode", "minimax_index", dict(max_rows=64, max_pages=16384, mode="decode"),
         dict(max_rows=64, max_pages=16384, mode="decode")),
        ("index_prefill", "minimax_index", dict(max_rows=4096, max_pages=16384, mode="prefill"),
         dict(max_rows=4096, max_pages=16384, mode="prefill")),
        ("attention_decode", "minimax_attention", dict(max_rows=64, max_batch=64, mode="decode"),
         dict(max_rows=64, max_batch=64, mode="decode")),
        ("attention_extend", "minimax_attention", dict(max_rows=4096, max_batch=64, mode="extend"),
         dict(max_rows=4096, max_batch=64, mode="extend")),
    ]
    # Non-default capacities must flow all the way into the pinned compilers.
    custom = namespace["minimax_sparse_programs"](16, 2048, 2049)
    assert custom[0][3]() == dict(max_rows=16, max_pages=33, mode="decode")
    assert custom[1][3]() == dict(max_rows=2048, max_pages=33, mode="prefill")
    assert custom[3][3]() == dict(max_rows=2048, max_batch=16, mode="extend")
    stem = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "stem" for target in node.targets)
                and "kimi_split" in ast.unparse(node.value))
    names = []
    for family, suffix, op, *_ in namespace["work"]:
        namespace.update(family=family, suffix=suffix, op=op)
        exec(compile(ast.Module(body=[stem], type_ignores=[]), "stem", "exec"), namespace)
        names.append(namespace["stem"])
    assert names[:len(native)] == ["exlr8_dense_" + name for name, *_ in native]
    assert names[len(native):] == ["minimax_index_decode", "minimax_index_prefill",
                                  "minimax_attention_decode", "minimax_attention_extend",
                                  "minimax_norm_rope_h64", "minimax_norm_rope_h4",
                                  "minimax_index_producer", "minimax_oai_split_i3072",
                                  "minimax_oai_split_i12288", "minimax_residual_norm",
                                  "minimax_router_scores", "minimax_kv_pack",
                                  "minimax_index_cache_gather"]
    assert all(params == {"max_rows": 4096} and thunk() is None
               for _, _, _, params, thunk in namespace["work"][-4:])
    namespace.update(family="glmf", suffix="sentinel", op="attention")
    exec(compile(ast.Module(body=[stem], type_ignores=[]), "stem", "exec"), namespace)
    assert namespace["stem"] == "glmf_sentinel"
    # MiniMax cannot accidentally enter either pre-existing family scope.
    scope_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "dense_family_scopes")
    exec(compile(ast.Module(body=[scope_node], type_ignores=[]), "scope", "exec"), namespace)
    with pytest.raises(ValueError):
        namespace["dense_family_scopes"]("minimax", None)

    _, scopes = aot.config_scopes(dict(FAMILY="minimax_m3", MODEL_ID="fixture/minimax", SPARK_COUNT="2"))
    assert scopes["CUTEAFD_WIP_MINIMAX_AOT"] == "ON"
    assert scopes["CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES"] == "minimax:nvfp4a4"
    assert scopes.get("CUTEAFD_WIP_EXLR8_AOT", "OFF") == "OFF"
    assert cmake_flags("coordinator", scopes)["CUTEAFD_ENABLE_MINIMAX_AOT"] == "ON"
    assert cmake_flags("expert", scopes)["CUTEAFD_ENABLE_MINIMAX_AOT"] == "OFF"

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    release = load("minimax_release_scope", "scripts/build/release-scope.py")
    monkeypatch.setattr(release, "scopes_module", lambda _: aot)
    (tmp_path / "minimax.config").write_text("FAMILY=minimax_m3\nMODEL_ID=fixture/minimax\nSPARK_COUNT=2\n")
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"entries": [{"name": "minimax", "family": "minimax_m3", "config": "minimax.config"}]}))
    derived, _ = release.derive(tmp_path, matrix, "minimax", {})
    assert derived["CUTEAFD_RELEASE_MINIMAX_AOT"] == "ON"
    assert derived["CUTEAFD_RELEASE_EXLR8_AOT"] == "OFF"
    assert release.requirements(tmp_path, matrix, "minimax")[0]["programs"] == ["minimax"]
    placement = load("minimax_placement_inputs", "scripts/build/export-placement-inputs.py")
    args = next(args for name, _, args in placement.export_jobs() if name == "programs")
    assert "minimax" in args[args.index("--geometry") + 1].split(",")
    assert "minimax" not in args[args.index("--dense-trellis-families") + 1].split(",")
    assert "minimax" not in args[args.index("--exlr8-family") + 1].split(",")

    (tmp_path / "CMakeLists.txt").write_text("\n".join([
        "cmake_minimum_required(VERSION 3.18)", "project(minimax_scope NONE)",
        "set(CUTEAFD_CUDA_ARCHITECTURES 120)", "set(CUTEAFD_CONTEXT_SPLIT_AOT OFF)",
        "set(CUTEAFD_ENABLE_MINIMAX_AOT ON)",
        "add_custom_target(cuteafd_verify_sparkinfer_source)",
        f'include("{ROOT / "native/cmake/shared/dsv4_programs.cmake"}")',
        'file(WRITE "${CMAKE_BINARY_DIR}/scope.json" "${CUTEAFD_PROGRAM_GEOMETRY}|${CUTEAFD_DENSE_TRELLIS_FAMILIES}|${CUTEAFD_EXLR8_FAMILIES}")',
    ]))
    build = tmp_path / "build"
    result = subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(build)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build / "scope.json").read_text() == "minimax,expert_reply||"


def minimax_forward_fixture():
    base = ROOT / "third_party/sparkinfer/b12x/integration/cuteafd"
    namespace = {
        "torch": types.SimpleNamespace(**{name: name for name in
                 ("bfloat16", "int64", "int32", "float32", "uint8")}),
        "Operand": lambda name, dtype, shape, role="in", **kwargs:
            dict(name=name, dtype=dtype, shape=shape, role=role),
        "Scalar": lambda name, kind="int32": dict(name=name, type=kind),
        "compile_program": lambda launch, **kwargs: kwargs,
        "_dtype": lambda dtype: dtype, "AffineNorm": lambda *args: None,
        "_ResidualNorm": lambda **kwargs: None,
        "_RouterScores": lambda: None, "_KvPack": lambda: None, "_IndexCacheGather": lambda: None,
    }
    modules = {}
    for source, names in (("pointwise", ("compile_affine_norm_aot",)),
                          ("minimax", ("compile_minimax_norm_aot", "compile_minimax_router_scores_aot",
                                       "compile_minimax_kv_pack_aot")),
                          ("index_cache", ("compile_index_cache_gather_aot",))):
        path = base / (source + ".py")
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        module = types.ModuleType("b12x.integration.cuteafd." + source)
        for name in names:
            setattr(module, name, namespace[name])
        modules[module.__name__] = module
    exporter = ast.parse((ROOT / "python/tools/aot/export_b12x_dsv4_aot.py").read_text())
    helper = next(node for node in exporter.body if isinstance(node, ast.FunctionDef)
                  and node.name == "minimax_forward_programs")
    dispatch = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "forward", "exec"), dispatch)
    import unittest.mock
    expected = []
    with unittest.mock.patch.dict(sys.modules, modules):
        for suffix, op, params, thunk in dispatch["minimax_forward_programs"](4096):
            program = thunk()
            expected.append(dict(name="minimax_" + suffix, family="minimax", op=op,
                params=params, capacity_rows=4096, geometry=program["geometry"],
                pointers=list(program["operands"]), scalars=list(program["scalars"]),
                scratch_bytes_at_capacity={}))
    return {"programs": expected}


def test_minimax_forward_fixture_matches_pinned_fork_without_cuda():
    import json
    fixture = ROOT / "rust/crates/cuteafd-ffi/tests/fixtures/minimax_forward_programs.json"
    assert json.loads(fixture.read_text()) == minimax_forward_fixture()


def test_minimax_producer_fixture_matches_pinned_fork_without_cuda():
    import json

    path = ROOT / "third_party/sparkinfer/b12x/integration/cuteafd/minimax_producer.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and (n.name.startswith("compile_minimax_") or n.name == "_positive_int")]
    namespace = {
        "torch": types.SimpleNamespace(**{name: name for name in
                 ("bfloat16", "int64", "float32", "uint8", "float8_e4m3fn")}),
        "Operand": lambda name, dtype, shape, role="in", **kwargs:
            dict(name=name, dtype=dtype, shape=shape, role=role),
        "Scalar": lambda name, kind="int32": dict(name=name, type=kind),
        "compile_program": lambda launch, **kwargs: kwargs,
        "_NormRope": lambda heads: None, "_Index": lambda: None, "_OaiSplit": lambda width: None,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    exporter = ast.parse((ROOT / "python/tools/aot/export_b12x_dsv4_aot.py").read_text())
    helper = next(n for n in exporter.body if isinstance(n, ast.FunctionDef)
                  and n.name == "minimax_producer_programs")
    dispatch = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "producers", "exec"), dispatch)
    # The catalog selects exactly these compiler specializations, in append-only order.
    import unittest.mock
    module = types.ModuleType("b12x.integration.cuteafd.minimax_producer")
    for name, fn in namespace.items():
        if name.startswith("compile_minimax_"): setattr(module, name, fn)
    expected = []
    with unittest.mock.patch.dict(sys.modules, {module.__name__: module}):
        for suffix, op, params, thunk in dispatch["minimax_producer_programs"](4096):
            program = thunk()
            expected.append(dict(name="minimax_" + suffix, family="minimax", op=op,
                params=params, capacity_rows=4096, geometry=program["geometry"],
                pointers=list(program["operands"]), scalars=list(program["scalars"]),
                scratch_bytes_at_capacity={key: fn(4096) for key, fn in program.get("scratch", {}).items()}))
    fixture = ROOT / "rust/crates/cuteafd-ffi/tests/fixtures/minimax_producer_programs.json"
    assert json.loads(fixture.read_text()) == {"programs": expected}
