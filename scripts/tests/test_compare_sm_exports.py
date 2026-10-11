"""CPU-only classification and offline export coverage for the SM gate."""
import ast
import importlib.util
import json
from pathlib import Path
import struct

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


def test_catalog_includes_all_geometries_and_fused_suspects():
    jobs = compare_sm.jobs("all")
    names = [name for name, _, _ in jobs]
    assert len(names) == len(set(names))
    assert set(compare_sm.GEOMETRIES) <= set(names)
    assert {f"routed-{name}" for name in compare_sm.ROUTED_GEOMETRIES} <= set(names)
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


def test_offline_exporters_do_not_initialize_or_allocate_cuda():
    for name in ("v41_fp8", "v41_nvfp4", "v41_experts", "slices"):
        source = (ROOT / f"python/tools/aot/export_b12x_{name}_aot.py").read_text()
        assert "torch.cuda.init()" not in source
        assert 'torch.empty(1, dtype=torch.uint8, device="cuda")' not in source


def residency_gate():
    # Load the pure gate without importing the CUDA compiler on the host.
    path = ROOT / "python/tools/aot/export_b12x_dsv4_aot.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "validate_table_residency")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["validate_table_residency"]


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
