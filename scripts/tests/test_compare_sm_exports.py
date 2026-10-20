"""CPU-only classification and offline export coverage for the SM gate."""
import ast
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("compare_sm_exports", ROOT / "scripts/build/compare-sm-exports.py")
compare_sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_sm)


def cuda_elf(payload=b"kernel"):
    # Minimal CUDA ELF64 with one file-backed section, matching CuTe's images.
    data = bytearray(128 + len(payload))
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", data, 18, 190)
    struct.pack_into("<QQ", data, 32, 0, 64)
    struct.pack_into("<HHHH", data, 54, 56, 0, 64, 1)
    struct.pack_into("<I", data, 68, 1)
    struct.pack_into("<QQ", data, 88, 128, len(payload))
    data[128:] = payload
    return bytes(data)


def exports(tmp_path, host=(b"host", b"host"), kernels=(b"kernel", b"kernel"), scratch=(64, 64)):
    directories = [tmp_path / "188", tmp_path / "170"]
    for i, directory in enumerate(directories):
        directory.mkdir()
        (directory / "test.o").write_bytes(host[i] + cuda_elf(kernels[i]))
        (directory / "test.h").write_text("unchanged ABI\n")
        (directory / "comparison-context.json").write_text(json.dumps({
            "sms": (188, 170)[i], "capability": [12, 0], "cuda_initialized": False,
            "sparkinfer_revision": "pin", "source_tree_sha256": "tree", "exporter": "export.py", "args": [],
            "toolchain": {"torch": "test", "triton": "test", "nvidia-cutlass-dsl": "test"},
            "environment": {"CUTEAFD_EXPORT_NARROW_AOT": "1", "CUTEAFD_EXPORT_HC_LAGGED": "1"},
            "exporter_sha256": "exporter", "comparison_script_sha256": "script",
            "export_sources_sha256": {"export.py": "source"},
            "max_shared_memory_per_block": 101376, "max_shared_memory_per_multiprocessor": 102400,
        }))
        (directory / "table.json").write_text(json.dumps({
            "physical_sms": (188, 170)[i], "device": "offline", "artifacts": {"hash": "varies"},
            "programs": [{"name": "test", "scratch_bytes_at_capacity": scratch[i], "object_sha256": "varies"}],
        }))
    return directories


@pytest.mark.parametrize("value", ["188", "188,188", "0,170", "-1,170", "188,170,84", "x,170"])
def test_invalid_sms(value):
    with pytest.raises(Exception, match="SM counts|two distinct"):
        compare_sm.parse_sms(value)


def test_two_target_counts():
    assert compare_sm.parse_sms("188,170") == (188, 170)


@pytest.mark.parametrize("group,names", [
    ("residual", ["attn_res_c1_keep_append", "attn_res_c2_add_append",
                  "attn_res_c4_add", "attn_res_c5_add",
                  "attn_res_c9_add", "attn_res_c9_replace", "attn_res_c9_keep"]),
    ("latent", ["rmsnorm_h3584", "rmsnorm_h7168", "latent_down", "latent_up"]),
    ("situ", [f"situ_i{width}" for width in (3072, 6144, 16896, 33792)]),
])
def test_kimi_w8_cross_sm_groups(group, names):
    job = next(job for job in compare_sm.jobs("suspects") if job[0] == f"kimi-{group}")
    assert job == (f"kimi-{group}", "export_b12x_dsv4_aot.py",
                   ["--geometry", "kimik3", "--only", ",".join(names)])


def test_mimop_k32_tp6_uses_production_exact_slice_export():
    job = next(job for job in compare_sm.jobs("all") if job[0] == "routed-mimop-tp6")
    assert job == ("routed-mimop-tp6", "package_fp8_moe_aot.py",
                   ["--geometry", "mimop", "--capacities", compare_sm.CAPACITIES,
                    "--layouts", "tp6", "--exact-slices"])


@pytest.mark.parametrize("geometry,bits", [
    ("qwen4", "45"), ("glm", "45"), ("glmf", "34"), ("dsv4f", "23"),
    ("v41", "23"), ("v41", "34"), ("v41", "45"),
])
def test_compact_gate_matches_packager_spark_companions(tmp_path, monkeypatch, geometry, bits):
    name = f"exl3-compact-{geometry}-k{bits}"
    _, exporter, args = next(job for job in compare_sm.jobs("all") if job[0] == name)
    assert exporter == "package_exl3_aot.py"
    package = compare_sm.load_file("test_compact_package", ROOT / "python/tools/aot/package_exl3_aot.py")
    profiles = package.compact_profiles(geometry, package.build_profiles("spark", geometry))
    capacities = package.package_capacities(geometry, compare_sm.CAPACITIES)
    recorded = []
    monkeypatch.setitem(sys.modules, "package_exl3_aot", package)
    monkeypatch.setitem(sys.modules, "export_b12x_exl3_aot", SimpleNamespace(
        export=lambda *values, **options: recorded.append((values, options))))
    compare_sm._export_exl3(args, tmp_path)
    assert [(values[0], values[1], values[2], values[3], values[5], values[6], values[7])
            for values, _ in recorded] == [
        (tmp_path / profile / f"m{rows}", width, experts, rows, "auto", topk, dtype)
        for profile, width, experts, topk, dtype, _ in profiles for rows in capacities]
    for values, options in recorded:
        assert values[4] == tuple(map(int, bits))
        assert options["compile_only"] and options["compact_weights"]
        assert options["tile"] == (64, 128, 64, 128)
        assert not options.get("warp_specialized") and not options.get("wire_input")
        rows = values[3]
        assert options.get("route_block", 8) == package.route_block(geometry, rows)
        assert options.get("swiglu_limit", 10.0) == package.swiglu_limit(geometry)
    widths = {profile[1] for profile in profiles}
    assert widths == ({256, 384} if geometry == "qwen4" else
                      {640} if geometry == "v41" else {384, 768})


def test_identical_object_ignores_context_metadata(tmp_path):
    report = compare_sm.compare(*exports(tmp_path))
    assert report["programs"][0]["classification"] == "identical"
    assert report["manifest_changed_fields"] == []


def test_host_only_difference_is_not_device_code(tmp_path):
    report = compare_sm.compare(*exports(tmp_path, host=(b"grid=188", b"grid=170")))
    record = report["programs"][0]
    assert record["classification"] == "host-only"
    assert record["cubins_identical"]
    assert not record["object_identical"]


def test_cubin_difference_classified_separately(tmp_path):
    report = compare_sm.compare(*exports(tmp_path, kernels=(b"kernel188", b"kernel170"), scratch=(64, 128)))
    record = report["programs"][0]
    assert record["classification"] == "cubin-bytes"
    assert not record["cubins_identical"]
    assert "test.scratch_bytes_at_capacity" in record["metadata_changed_fields"]


def test_missing_objects_fail_closed(tmp_path):
    left, right = exports(tmp_path)
    (right / "test.o").unlink()
    with pytest.raises(ValueError, match="object set mismatch"):
        compare_sm.compare(left, right)


def test_missing_cubins_fail_closed(tmp_path):
    left, right = exports(tmp_path)
    (left / "test.o").write_bytes(b"no CUDA image")
    with pytest.raises(ValueError, match="no CUDA ELF"):
        compare_sm.compare(left, right)


def test_changed_export_arguments_fail_closed(tmp_path):
    left, right = exports(tmp_path)
    path = right / "comparison-context.json"
    context = json.loads(path.read_text())
    context["args"] = ["different-capacity"]
    path.write_text(json.dumps(context))
    with pytest.raises(ValueError, match="beyond SM count: args"):
        compare_sm.compare(left, right)


@pytest.mark.parametrize("field", ["toolchain", "export_sources_sha256", "max_shared_memory_per_block"])
def test_missing_provenance_on_both_targets_fails_closed(tmp_path, field):
    left, right = exports(tmp_path)
    for directory in (left, right):
        path = directory / "comparison-context.json"
        context = json.loads(path.read_text())
        del context[field]
        path.write_text(json.dumps(context))
    with pytest.raises(ValueError, match=f"missing provenance: {field}"):
        compare_sm.compare(left, right)


def test_same_sm_count_does_not_prove_a_cross_device_comparison(tmp_path):
    left, right = exports(tmp_path)
    path = right / "comparison-context.json"
    context = json.loads(path.read_text())
    context["sms"] = 188
    path.write_text(json.dumps(context))
    with pytest.raises(ValueError, match="two distinct SM counts"):
        compare_sm.compare(left, right)


def test_initialized_cuda_context_fails_closed(tmp_path):
    left, right = exports(tmp_path)
    path = right / "comparison-context.json"
    context = json.loads(path.read_text())
    context["cuda_initialized"] = True
    path.write_text(json.dumps(context))
    with pytest.raises(ValueError, match="CPU-only"):
        compare_sm.compare(left, right)


def test_nested_objects_and_standalone_cubins(tmp_path):
    left, right = exports(tmp_path)
    for directory in (left, right):
        nested = directory / "tp1/m16/routes"
        nested.mkdir(parents=True)
        (nested / "sort.cubin").write_bytes(cuda_elf())
        (nested / "routes.json").write_text(json.dumps({
            "objects": [{"stage": "sort", "grid_x": "(live + 255) / 256"}]}))
        for name in ("test.o", "test.h", "table.json"):
            (directory / name).rename(nested.parent / name)
    report = compare_sm.compare(left, right)
    records = {row["program"]: row for row in report["programs"]}
    assert set(records) == {"tp1/m16/test", "tp1/m16/routes/sort"}
    assert records["tp1/m16/routes/sort"]["artifact_kind"] == "cubin"
    assert records["tp1/m16/test"]["classification"] == "identical"


def test_missing_abi_header_fails_closed(tmp_path):
    left, right = exports(tmp_path)
    (right / "test.h").unlink()
    with pytest.raises(ValueError, match="missing exported ABI header"):
        compare_sm.compare(left, right)


def test_list_includes_dense_qwen_export_groups():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/build/compare-sm-exports.py"), "--list"],
                            check=True, capture_output=True, text=True)
    names = result.stdout.splitlines()
    for geometry in ("qwen35", "qwen38"):
        assert names.count(geometry) == 1
        job = next(job for job in compare_sm.jobs("all") if job[0] == geometry)
        assert job == (geometry, "export_b12x_dsv4_aot.py", ["--geometry", geometry])


@pytest.mark.parametrize("heads,suffix", [(64, ""), (32, "-tp2")])
@pytest.mark.parametrize("group", ["pointwise", "projections", "attention", "metadata", "region-scores"])
def test_step5_export_groups_bind_native_geometry(heads, suffix, group):
    job = next(job for job in compare_sm.jobs("all") if job[0] == f"step5-{group}{suffix}")
    expected = ["--heads", str(heads)]
    if group != "pointwise":
        expected += ["--group", group]
    assert job == (f"step5-{group}{suffix}", "export_b12x_step5_aot.py", expected)


@pytest.mark.parametrize("heads", [32, 64])
def test_step5_projection_operands_keep_checkpoint_materialization_boundaries(monkeypatch, heads):
    import types
    module_path = ROOT / "python/tools/aot/export_b12x_step5_aot.py"
    spec = importlib.util.spec_from_file_location("step5_projection_exports", module_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16", float32="f32"))
    pointwise = SimpleNamespace(compile_bf16_projection_aot=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "b12x", types.ModuleType("b12x"))
    monkeypatch.setitem(sys.modules, "b12x.integration", types.ModuleType("b12x.integration"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd", SimpleNamespace(pointwise=pointwise))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = {name: build() for name, build in module.builders(heads, "projections").items()}
    assert len(calls) == 12
    for name, (width, outputs, dtype) in {
        "attn_q": (4096, heads * 192, "bf16"),
        "attn_kv": (4096, heads // 16 * 192, "bf16"),
        "attn_gate": (4096, heads, "bf16"),
        "attn_out": (heads * 192, 4096, "bf16"),
        "attn_out_partial": (heads * 192, 4096, "f32"),
        "index_q": (4096, heads // 16 * 4 * 256, "bf16"),
        "index_kz": (4096, 256, "bf16"),
        "router": (4096, 352, "f32"),
        "dense_gate_up": (4096, 27648, "bf16"),
        "dense_down": (13824, 4096, "bf16"),
        "shared_gate_up": (4096, 3072, "bf16"),
        "shared_down": (1536, 4096, "bf16"),
    }.items():
        assert calls[name] == dict(width=width, outputs=outputs, output_dtype=dtype)


def test_catalog_includes_all_geometries_and_fused_suspects():
    jobs = compare_sm.jobs("all")
    names = [name for name, _, _ in jobs]
    assert len(names) == len(set(names))
    assert set(compare_sm.GEOMETRIES) <= set(names)
    assert {f"routed-{name}" for name in compare_sm.ROUTED_GEOMETRIES} <= set(names)
    for geometry in compare_sm.ROUTED_GEOMETRIES:
        if geometry.startswith("glmfdense"):
            assert f"routed-{geometry}-tp2" not in names
            continue
        args = next(args for name, _, args in jobs if name == f"routed-{geometry}-tp2")
        assert args[args.index("--layouts") + 1] == "tp2"
        assert "--exact-slices" in args
    assert {f"exl3-{name}" for name in compare_sm.EXL3_GEOMETRIES} <= set(names)
    assert {f"exl3-routes-{name}" for name in compare_sm.EXL3_GEOMETRIES} <= set(names)
    assert {"audio", "vision-attention"} <= set(names)
    assert {f"exl3-{name}-k23" for name in compare_sm.EXL3_GEOMETRIES} <= set(names)
    assert {"exl3-glm-k45", "exl3-qwen4-k45", "exl3-qwen4-k345", "exl3-qwen4-k2345"} <= set(names)
    assert {f"context-split-{name}" for name in ("flash", "pro", "glm", "glmf")} <= set(names)
    assert {"v41-fp8", "v41-experts", "v41-index", "v41-router", "v41-attention"} <= set(names)
    assert {"suspect-flash-m1", "suspect-flash-m16", "suspect-glm-mla", "suspect-glmf-mla"} <= set(names)
    assert all("--only" in args for _, _, args in compare_sm.jobs("suspects"))
    expert_args = next(args for name, _, args in jobs if name == "v41-experts")
    assert expert_args[expert_args.index("--input-format") + 1] == "bf16"
    for family in ("dsv4f", "dsv4p"):
        for suffix, role in (("", "rtx_backbone"), ("-rtx_tp2", "rtx_tp2")):
            args = next(args for name, _, args in jobs if name == f"{family}-slices{suffix}")
            assert args[args.index("--geometry") + 1] == family
            assert args[args.index("--role") + 1] == role
            assert args[args.index("--atomic-min-capacity") + 1] == "256"


@pytest.mark.parametrize("geometry", ["glmf", "glmf2", "kimik3"])
@pytest.mark.parametrize("label,stem", [("decode", "kda_m64"), ("prefill", "kda_m4096"), ("commit", "kda_commit")])
def test_kda_groups_export_only_the_shared_kda_programs(geometry, label, stem):
    job = next(job for job in compare_sm.jobs("suspects") if job[0] == f"kda-{geometry}-{label}")
    assert job == (f"kda-{geometry}-{label}", "export_b12x_dsv4_aot.py",
                   ["--geometry", geometry, "--only", stem])


def test_hybrid_group_covers_compact_and_grouped_live_rows():
    _, exporter, args = next(job for job in compare_sm.jobs("all")
                             if job[0] == "v41-slices-rtx_tp2-hybrid")
    assert exporter == "export_b12x_slices_aot.py"
    assert args[args.index("--role") + 1] == "rtx_tp2"
    assert args[args.index("--compact-max-capacity") + 1] == "80"
    assert args[args.index("--compact-live-rows") + 1] == "8"
    assert args[args.index("--rows") + 1] == "1,2,16,80"


def test_hybrid_objects_are_identical_at_188_and_170_sms(tmp_path, monkeypatch):
    # The normal host script gate has no DSL compiler. In the coordinator
    # compiler image this executes the real offline exporter, without a GPU.
    pytest.importorskip("cutlass")
    monkeypatch.setattr(compare_sm, "ROOT", ROOT)
    monkeypatch.setenv("CUTEAFD_SOURCE_ROOT", str(ROOT))
    for sms in (188, 170):
        output = tmp_path / str(sms)
        output.mkdir()
        payload = dict(output=str(output), sms=sms,
                       exporter="export_b12x_slices_aot.py",
                       args=["--role", "rtx_tp2", "--rows", "1,16", "--width", "192",
                             "--compact-max-capacity", "16", "--compact-live-rows", "8",
                             "--standard-names"])
        subprocess.run([sys.executable, str(ROOT / "scripts/build/compare-sm-exports.py"),
                        "--_worker", json.dumps(payload)], check=True, timeout=300)
    report = compare_sm.compare(tmp_path / "188", tmp_path / "170")
    assert report["programs"]
    assert all(program["object_identical"] and program["header_identical"]
               and program["cubins_identical"] for program in report["programs"])
    assert not report["manifest_changed_fields"]


def test_offline_exporters_do_not_initialize_or_allocate_cuda():
    for name in ("v41_fp8", "v41_nvfp4", "v41_experts", "slices"):
        source = (ROOT / f"python/tools/aot/export_b12x_{name}_aot.py").read_text()
        assert "torch.cuda.init()" not in source
        assert 'torch.empty(1, dtype=torch.uint8, device="cuda")' not in source


@pytest.mark.parametrize("bits", [(3, 4), (3, 4, 5), (2, 3, 4, 5)])
@pytest.mark.parametrize("rotation", ["warp_specialized", "fused_input_rotation", "token_major_rotation"])
def test_multitier_exports_do_not_use_two_tier_only_schedules(tmp_path, monkeypatch, bits, rotation):
    recorded = []
    package = SimpleNamespace(
        GEOMETRIES={"qwen4": (4096,)},
        profiles_for_role=lambda *_: [("tp1", 512, 128, 8, "bf16", [])],
        route_block=lambda *_: 8, swiglu_limit=lambda *_: 10.0,
        wire_input=lambda *_: False, ws_input_stages=lambda *_: None,
        ws_dynamic_tiles=lambda *_: False, ws_tile=lambda *_: None,
        decode_tile=lambda *_: None,
    )
    for name in ("warp_specialized", "fused_input_rotation", "token_major_rotation"):
        setattr(package, name, lambda *_, name=name: name == rotation)
    monkeypatch.setitem(sys.modules, "package_exl3_aot", package)
    monkeypatch.setitem(sys.modules, "export_b12x_exl3_aot", SimpleNamespace(
        export=lambda *args, **kwargs: recorded.append((args, kwargs))))
    compare_sm._export_exl3(["--geometry", "qwen4", "--capacities", "1024",
                            "--bits", ",".join(map(str, bits))], tmp_path)
    assert len(recorded) == 1
    assert recorded[0][0][4] == bits
    assert recorded[0][1].get(rotation, False) == (len(bits) == 2)


def test_cpu_lut_override_covers_all_bindings_and_restores_on_error():
    first = SimpleNamespace(_trellis256_execution_lut=lambda device, codebook: ("first", device, codebook))
    fourth = SimpleNamespace(_trellis256_execution_lut=lambda device, codebook: ("fourth", device, codebook))
    originals = [module._trellis256_execution_lut for module in (first, fourth)]
    with pytest.raises(RuntimeError, match="compile failed"):
        with compare_sm.cpu_trellis_luts((first, fourth), "cpu"):
            assert first._trellis256_execution_lut("cuda", "codebook") == ("first", "cpu", "codebook")
            assert fourth._trellis256_execution_lut("cuda", "codebook") == ("fourth", "cpu", "codebook")
            raise RuntimeError("compile failed")
    assert [module._trellis256_execution_lut for module in (first, fourth)] == originals


def test_production_package_limits_special_rotation_to_two_tiers():
    source = (ROOT / "python/tools/aot/package_exl3_aot.py").read_text()
    for policy in ("warp_specialized", "fused_input_rotation", "token_major_rotation"):
        matches = [line for line in source.splitlines()
                   if f"{policy}(geometry," in line and line.startswith("                ")]
        assert matches and all("len(args.bits) == 2" in line for line in matches)


def residency_gate(name="validate_table_residency"):
    # Load the pure gate without importing the CUDA compiler on the host.
    path = ROOT / "python/tools/aot/export_b12x_dsv4_aot.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("minimum", [188, 170, 11, 0])
def test_diagnostic_minimum_sms_contract_exported(minimum):
    assert residency_gate("program_minimum_sms")("diagnostic", {"route": "paged_fused", "minimum_sms": minimum}) == minimum


@pytest.mark.parametrize("geometry", [{"route": "paged_fused"}, {"minimum_sms": -1}, {"minimum_sms": True}])
def test_invalid_diagnostic_minimum_sms_contract_rejected(geometry):
    with pytest.raises(ValueError, match="minimum_sms"):
        residency_gate("program_minimum_sms")("diagnostic", geometry)


@pytest.mark.parametrize("rows", [1, 16, 64, 128, 4096])
def test_fused_cannot_enter_default_table_at_any_capacity(rows):
    with pytest.raises(ValueError, match="residency gate.*fallback"):
        residency_gate()(f"glmf_index_topk_decode_m{rows}", {"route": "paged_fused", "max_rows": rows})


@pytest.mark.parametrize("route", ["paged_tiled", "packed_contiguous", "decode"])
def test_grid_agnostic_default_table_routes_allowed(route):
    residency_gate()("program", {"route": route})


def test_fused_diagnostics_are_not_serving_tables():
    residency_gate()("suspect", {"route": "paged_fused"}, diagnostic=True)


def test_comparison_rejects_existing_ungated_default_table(tmp_path):
    left, right = exports(tmp_path)
    for directory in (left, right):
        context = json.loads((directory / "comparison-context.json").read_text())
        context["exporter"] = "export_b12x_dsv4_aot.py"
        (directory / "comparison-context.json").write_text(json.dumps(context))
        (directory / "comparison-plans.json").write_text(json.dumps({
            "test": {"geometry": {"route": "paged_fused", "max_rows": 128}}}))
    with pytest.raises(ValueError, match="default serving table.*residency"):
        compare_sm.compare(left, right)
    for directory in (left, right):
        context = json.loads((directory / "comparison-context.json").read_text())
        context["args"] = ["--only", "test"]
        (directory / "comparison-context.json").write_text(json.dumps(context))
    assert compare_sm.compare(left, right)["programs"][0]["object_identical"]


def test_table_export_calls_gate_before_writing_object():
    source = (ROOT / "python/tools/aot/export_b12x_dsv4_aot.py").read_text()
    assert source.index("validate_table_residency(stem,") < source.index("program.export_to_c(")


def test_failed_group_does_not_discard_other_completed_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_sm, "jobs", lambda _: [("bad", "export.py", []), ("good", "export.py", [])])
    monkeypatch.setattr(compare_sm, "load_file", lambda *_: type("Filesystem", (), {"check_path": lambda _: None}))
    monkeypatch.setattr(compare_sm.sys, "argv", ["compare-sm-exports.py", "--compare-only",
                        "--workers", "2", "--output-dir", str(tmp_path)])
    (tmp_path / "good").mkdir()

    def compare(left, right):
        if left.parent.name == "bad":
            raise ValueError("missing pair")
        return {"programs": []}

    monkeypatch.setattr(compare_sm, "compare", compare)
    assert compare_sm.main() == 1
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert [group["group"] for group in report["groups"]] == ["good"]
    assert report["failures"] == [{"group": "bad", "error": "missing pair"}]


@pytest.mark.parametrize("heads", [32, 64])
def test_step5_norms_take_runtime_epsilon_and_keep_final_bf16_boundary(monkeypatch, heads):
    import types
    spec = importlib.util.spec_from_file_location("step5_norm_exports",
        ROOT / "python/tools/aot/export_b12x_step5_aot.py")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16", float32="f32"))
    pointwise = SimpleNamespace(compile_affine_norm_aot=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "b12x", types.ModuleType("b12x"))
    monkeypatch.setitem(sys.modules, "b12x.integration", types.ModuleType("b12x.integration"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd", SimpleNamespace(pointwise=pointwise))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = {name: build() for name, build in module.builders(heads, "pointwise").items()
             if "norm" in name}
    assert set(calls) == {"hidden_norm", "final_norm", "index_q_norm", "index_k_norm",
        "swa_q_norm_rope", "swa_k_norm_rope", "csa_q_norm_rope", "csa_k_norm_rope"}
    assert all("eps" in call and call["eps"] is None for call in calls.values())
    assert calls["hidden_norm"].get("input_dtype", "f32") == "f32"
    assert calls["final_norm"] == dict(width=4096, zero_centered=True, eps=None, input_dtype="bf16")
    for name in ("index_q_norm", "index_k_norm"):
        assert calls[name]["input_dtype"] == calls[name]["parameter_dtype"] == "bf16"
    assert calls["index_k_norm"]["subtract_mean"] and calls["index_k_norm"]["bias"]
    assert calls["index_q_norm"]["zero_centered"]
    for kind, rotary in (("swa", 192), ("csa", 64)):
        for name, count in (("q", heads), ("k", heads // 16)):
            assert calls[f"{kind}_{name}_norm_rope"] == dict(width=192, heads=count,
                rotary=rotary, zero_centered=True, eps=None, input_dtype="bf16")


@pytest.mark.parametrize("heads", [32, 64])
def test_step5_rope_builders_cover_all_policy_modes_without_recipe_specialization(monkeypatch, heads):
    import types
    spec = importlib.util.spec_from_file_location("step5_rope_exports",
        ROOT / "python/tools/aot/export_b12x_step5_aot.py")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16", float32="f32"))
    monkeypatch.setitem(sys.modules, "b12x", types.ModuleType("b12x"))
    monkeypatch.setitem(sys.modules, "b12x.integration", types.ModuleType("b12x.integration"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd", SimpleNamespace(pointwise=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd.rope", SimpleNamespace(
        compile_rope_tables_aot=lambda **kwargs: kwargs))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builders = module.builders(heads, "pointwise")
    calls = {name: build() for name, build in builders.items() if name.startswith("rope_s")}
    assert calls == {f"rope_s{span}_{'llama3' if scaling else 'plain'}":
                     dict(rotary_span=span, llama3=scaling)
                     for span in (32, 64, 192) for scaling in (False, True)}
def test_source_comparison_requires_same_sm_and_allows_only_source_changes(tmp_path):
    left, right = exports(tmp_path)
    with pytest.raises(ValueError, match="same SM count"):
        compare_sm.compare(left, right, source_comparison=True)
    context_file = right / "comparison-context.json"
    context = json.loads(context_file.read_text())
    context.update(sms=188, sparkinfer_revision="candidate", source_tree_sha256="new-tree",
                   exporter_sha256="new-exporter", export_sources_sha256={"export.py": "new"})
    context_file.write_text(json.dumps(context))
    assert compare_sm.compare(left, right, source_comparison=True)["programs"][0]["object_identical"]
    context["toolchain"] = {"torch": "other"}
    context_file.write_text(json.dumps(context))
    with pytest.raises(ValueError, match="toolchain"):
        compare_sm.compare(left, right, source_comparison=True)
