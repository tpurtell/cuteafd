"""Opt-in BF16 input packages must survive staging and retain their build identity."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import ctypes
import shutil
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "python/tools/aot/package_fp8_moe_aot.py"
REVISION = json.loads((REPO / "third_party/sparkinfer.lock.json").read_text())["revision"]
spec = importlib.util.spec_from_file_location("fp8_package", TOOL)
package_tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = package_tool
spec.loader.exec_module(package_tool)
preflight_spec = importlib.util.spec_from_file_location("bf16_preflight", REPO / "scripts/launch/preflight-fp8-bf16.py")
preflight = importlib.util.module_from_spec(preflight_spec)
assert preflight_spec.loader is not None
preflight_spec.loader.exec_module(preflight)


@pytest.mark.parametrize("intermediate", [640, 2048, 2304, 3072])
def test_all_spark_counts_own_unequal_whole_blocks(intermediate):
    layouts = package_tool.spark_layouts(intermediate)
    for tp in range(1, 9):
        widths = package_tool.exact_widths(intermediate, tp)
        if tp > 6 or tp > intermediate // 128:
            assert not widths
            assert f"tp{tp}" not in layouts
            continue
        assert len(widths) == tp
        assert sum(widths) == intermediate
        assert all(width > 0 and width % 128 == 0 for width in widths)
        assert max(widths) - min(widths) <= 128
        assert f"tp{tp}" in layouts
        if len(set(widths)) > 1:
            assert all(f"tp{tp}-w{width}" in layouts for width in widths)
    assert package_tool.exact_widths(intermediate, 0) == []
    assert package_tool.exact_widths(intermediate, 9) == []


@pytest.mark.parametrize("tp", [7, 8])
def test_packager_rejects_deleted_world_before_geometry_or_compilation(tp):
    assert package_tool.exact_widths(2048, tp) == []
    assert not any(layout.startswith(f"tp{tp}") for layout in package_tool.spark_layouts(2048))
    with pytest.raises(SystemExit, match="Spark TP must be in 1..6"):
        package_tool.layout_geometry(None, f"tp{tp}")


def test_qwen_tp5_exact_slice_has_no_padding():
    assert package_tool.exact_widths(640, 5) == [128] * 5
    assert "tp5" in package_tool.spark_layouts(640)
    assert not any(layout.startswith(("tp6", "tp7", "tp8")) for layout in package_tool.spark_layouts(640))


def test_mxfp4_exact_k32_widths_reuse_existing_tail_abi():
    assert package_tool.exact_widths(2048, 3) == [768, 640, 640]
    assert package_tool.exact_widths(2048, 3, 32) == [704, 672, 672]
    assert package_tool.exact_widths(2048, 6, 32) == [352, 352, 352, 352, 320, 320]
    assert package_tool.exact_widths(2048, 6, 16) == []
    # Importing geometry is CPU-only; no exporter or CUDA context is involved.
    from dataclasses import dataclass, replace

    @dataclass(frozen=True)
    class Geometry:
        intermediate: int = 2048
        weights: str = "mxfp4"
        tp: int = 1

        def with_tp(self, tp):
            return replace(self, tp=tp)

    base = Geometry()
    for width in (352, 320, 384, 256):
        tp, geometry = package_tool.layout_geometry(base, f"tp6-w{width}")
        assert tp == 6
        assert geometry.slice == width
    with pytest.raises(SystemExit, match="no rank"):
        package_tool.layout_geometry(base, "tp6-w336")
    for width in (768, 640, 704, 672):
        tp, geometry = package_tool.layout_geometry(base, f"tp3-w{width}")
        assert tp == 3
        assert geometry.slice == width
    for width in (656, 688, 736):
        with pytest.raises(SystemExit, match="no rank"):
            package_tool.layout_geometry(base, f"tp3-w{width}")


@pytest.mark.parametrize("counts,expected", [("", ""), ("5", "--layouts;tp5"),
                                            ("1;5;6", "--layouts;tp1,tp5,tp6")])
def test_cmake_requested_counts_preserve_defaults_and_select_both_input_forms(tmp_path, counts, expected):
    source = (REPO / "native/cmake/shared/fp8_moe.cmake").read_text()
    body = source.split("  set(requested_layouts)", 1)[1].split("  set(stamp ", 1)[0]
    script = tmp_path / "select.cmake"
    script.write_text('set(CUTEAFD_FP8_MOE_ROLE spark)\n'
                      f'set(CUTEAFD_GENERIC_SPARK_COUNTS "{counts}")\n'
                      'set(requested_layouts)\n' + body + '\nmessage("LAYOUTS=${requested_layouts}")\n')
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert f"LAYOUTS={expected}" in result.stderr
    assert source.count('${exact_slices} ${requested_layouts}') >= 2


def test_step5_nvfp4_keeps_exporter_geometry_and_canonical_a4_package(tmp_path):
    source = (REPO / "native/cmake/shared/fp8_moe.cmake").read_text()
    body = source.split("set(CUTEAFD_FP8_MOE_ENTRIES)\n", 1)[1].split(
        "foreach(entry IN LISTS CUTEAFD_FP8_MOE_ENTRIES)", 1)[0]
    script = tmp_path / "entries.cmake"
    script.write_text('set(CUTEAFD_EXPERT_FAMILIES step5:nvfp4 glm:nvfp4)\n' + body
                      + '\nmessage("ENTRIES=${CUTEAFD_FP8_MOE_ENTRIES}")\n')
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "ENTRIES=step5:nvfp4;glm:nvfp4;glm:nvfp4a4" in result.stderr
    assert "step5_nvfp4a4" not in TOOL.read_text()
    branch = source.split('elseif(entry STREQUAL "step5:nvfp4")', 1)[1].split("elseif(", 1)[0]
    assert 'set(geometry "step5_nvfp4")' in branch
    assert '/fp8/fp8-step5-nvfp4a4")' in branch


def test_step5_bf16_alone_enables_package_exporter(tmp_path):
    source = (REPO / "native/CMakeLists.txt").read_text()
    body = source.split("set(CUTEAFD_FP8_EXPERT_FAMILIES OFF)", 1)[1].split("endif()", 1)[0]
    script = tmp_path / "bf16.cmake"
    script.write_text('cmake_minimum_required(VERSION 3.24)\n'
                      'set(CUTEAFD_EXPERT_FAMILIES step5:bf16)\n' + body + 'endif()\n'
                      'message("ENABLED=${CUTEAFD_FP8_EXPERT_FAMILIES}")\n')
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "ENABLED=ON" in result.stderr


def test_step5_bf16_bypasses_native_slice_exporter(tmp_path):
    source = REPO / "native/cmake/shared/expert_families.cmake"
    script = tmp_path / "native.cmake"
    script.write_text('set(CUTEAFD_EXPERT_FAMILIES step5:bf16 step5:fp8 step5:nvfp4)\n'
                      f'include("{source}")\n'
                      'message("NATIVE=${CUTEAFD_NATIVE_EXPERT_FAMILIES}")\n')
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "NATIVE=\n" in result.stderr


@pytest.mark.parametrize("counts", ["0", "7", "8", "9", "5;bad"])
def test_cmake_requested_counts_fail_before_compilation(tmp_path, counts):
    source = (REPO / "native/cmake/shared/fp8_moe.cmake").read_text()
    body = source.split("  set(requested_layouts)", 1)[1].split("  set(stamp ", 1)[0]
    script = tmp_path / "select.cmake"
    script.write_text('set(CUTEAFD_FP8_MOE_ROLE spark)\n'
                      f'set(CUTEAFD_GENERIC_SPARK_COUNTS "{counts}")\n' + body)
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "requires counts 1..6" in result.stderr


@pytest.mark.parametrize("geometry,blocks", [("qwen4", 5), ("glm", 16), ("glmf", 16),
                                             ("dsv4f", 16), ("dsv4p", 24)])
@pytest.mark.parametrize("count", range(1, 9))
def test_cmake_exl3_requested_counts_match_packager_profiles(tmp_path, geometry, blocks, count):
    source = (REPO / "native/cmake/shared/exl3.cmake").read_text()
    body = source.split("  set(CUTEAFD_EXL3_PROFILE_ARGS)", 1)[1].split(
        '  list(JOIN CUTEAFD_EXL3_GEOMETRY_LAYOUTS', 1)[0]
    script = tmp_path / "select.cmake"
    script.write_text('set(CUTEAFD_EXL3_ROLE spark)\n'
                      f'set(CUTEAFD_EXL3_GEOMETRY {geometry})\n'
                      f'set(CUTEAFD_GENERIC_SPARK_COUNTS {count})\n' + body
                      + '\nmessage("LAYOUTS=${CUTEAFD_EXL3_GEOMETRY_LAYOUTS}")\n'
                      + 'message("PROFILES=${CUTEAFD_EXL3_PROFILE_ARGS}")\n')
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=30)
    if count > 6 or count > blocks:
        assert result.returncode != 0
        assert "requires a nonempty H128 slice" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    assert 'LAYOUTS=' + ';'.join(f'tp{count}-rank{rank}' for rank in range(count)) in result.stderr
    widths = sorted({(blocks // count + (rank < blocks % count)) * 128 for rank in range(count)}, reverse=True)
    profiles = ';'.join(part for width in widths for part in ('--profile', f'tp{count}-width{width}'))
    assert f'PROFILES={profiles}' in result.stderr


def test_coordinator_exports_tp2_for_every_routed_weight_format():
    for layouts in (package_tool.ROLE_LAYOUTS, package_tool.MXFP4_ROLE_LAYOUTS,
                    package_tool.NVFP4_ROLE_LAYOUTS):
        assert layouts["coordinator"] == ("tp1", "tp2")
    assert package_tool.ROLE_INPUT["coordinator"] == "bf16"
    # Unequal coordinator slices must not pass through Spark's default filter.
    assert 'elif args.role == "spark" and not args.layouts' in TOOL.read_text()
    cmake = (REPO / "native/cmake/shared/fp8_moe.cmake").read_text()
    assert 'set(exact_slices "--exact-slices")' in cmake


def test_capacity_scratch_reserves_largest_retained_form_like_runtime_abi():
    programs = [
        {"capacity": 4096, "form": "w8a8", "scratch": 470028288},
        {"capacity": 1024, "form": "auto", "scratch": 117510144},
        {"capacity": 16, "form": "auto", "scratch": 1841152},
        {"capacity": 4096, "form": "auto", "scratch": 734346240},
        {"capacity": 1024, "form": "w8a8", "scratch": 163779584},
    ]
    assert package_tool.capacity_scratch(programs) == [
        {"capacity": 16, "scratch_bytes": 1841152},
        {"capacity": 1024, "scratch_bytes": 163779584},
        {"capacity": 4096, "scratch_bytes": 734346240},
    ]
    source = TOOL.read_text()
    assert '"capacities": capacity_scratch(programs)' in source
    assert "kPrograms[i].scratch > *bytes" in source


@pytest.mark.parametrize("geometry", [
    "mimo", "mimop", "mimof", "glm", "glmf", "qwen4", "step5_bf16",
    "glm_nvfp4", "glmf_nvfp4", "qwen4_nvfp4", "glmfdense_nvfp4",
    "glm_nvfp4a4", "glmf_nvfp4a4", "qwen4_nvfp4a4", "glmfdense_nvfp4a4",
])
def test_bridge_shares_aot_handles_and_device_optins_until_last_rank(tmp_path, geometry):
    compiler = shutil.which("c++")
    assert compiler, "CPU package regression requires a C++ compiler"
    (tmp_path / "cuda_runtime.h").write_text("""
#pragma once
using cudaLibrary_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1;
extern "C" int cudaGetDevice(int*);
extern "C" int cudaLibraryUnload(void*);
""")
    stem = f"mock_{geometry}"
    (tmp_path / f"{stem}.h").write_text(f"""
#pragma once
#include <set>
#include <cstdint>
static int device = 0, initialized = 0, unloaded = 0, fail_device = -1;
static uintptr_t serial = 0;
static void* kernel_library = nullptr;
static std::set<std::pair<void*, int>> optins;
extern "C" int cudaGetDevice(int* out) {{ *out = device; return 0; }}
extern "C" int cudaLibraryUnload(void*) {{ ++unloaded; return 0; }}
extern "C" void mock_device(int value) {{ device = value; }}
extern "C" void mock_fail(int value) {{ fail_device = value; }}
extern "C" int mock_initialized() {{ return initialized; }}
extern "C" int mock_unloaded() {{ return unloaded; }}
static void _mlir_cuteafd_{stem}_cuda_init(void** args) {{
  auto* library = *static_cast<void***>(args[0]);
  *library = reinterpret_cast<void*>(++serial);
  ++initialized;
  *static_cast<int*>(args[1]) = 0;
}}
static void _mlir_cuteafd_{stem}_cuda_load_to_device(void** args) {{
  auto library = **static_cast<void***>(args[0]);
  int rank = *static_cast<int*>(args[1]);
  kernel_library = library;
  if (rank == fail_device) {{ *static_cast<int*>(args[2]) = 1; return; }}
  optins.emplace(library, rank);
  *static_cast<int*>(args[2]) = 0;
}}
static void mock_launch(void** args, int) {{
  *static_cast<int*>(args[13]) = optins.count({{kernel_library, device}}) ? 0 : 1;
}}
""")
    source = package_tool.bridge_source(
        [{"stem": stem, "capacity": 80, "scratch": 65536, "form": "auto", "entry": "mock_launch",
          "weight_read_grouping": ({"contract": "conservative_route_upper_bound",
              "segments": [{"end_rows": 80, "group_rows": 1}]} if "nvfp4" in geometry else
              {"contract": "exact_grouping", "segments": [
                  {"end_rows": 16, "group_rows": 16}, {"end_rows": 80, "group_rows": 64}]})}],
        [5 | (5 if geometry == "step5_bf16" else 4 if "nvfp4" in geometry else 1) << 8 | 1 << 16 | 1 << 24, 6144, 1024, 384, 8, 2048, 2, 1, 0, 1, 80, 0, 0, 0, 0, 0])
    (tmp_path / "bridge.cc").write_text(source)
    library = tmp_path / "bridge.so"
    compiled = subprocess.run([compiler, "-shared", "-fPIC", "-std=c++17", "-pthread",
                               f"-I{tmp_path}", str(tmp_path / "bridge.cc"), "-o", str(library)],
                              text=True, capture_output=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    lib = ctypes.CDLL(str(library))
    lib.cuteafd_fp8moe_destroy.argtypes = [ctypes.c_void_p]
    lib.cuteafd_fp8moe_launch.argtypes = [ctypes.c_void_p, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_void_p]
    lib.cuteafd_fp8moe_weight_read_group.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
    group, contract = ctypes.c_uint(), ctypes.c_uint()
    for form in (0, 1, 2):
        for rows, expected in ((0, 64), (1, 16), (16, 16), (17, 64), (80, 64)):
            assert lib.cuteafd_fp8moe_weight_read_group(80, form, rows, ctypes.byref(group), ctypes.byref(contract)) == 0
            assert (group.value, contract.value) == ((1, 2) if "nvfp4" in geometry else (expected, 1))
    assert lib.cuteafd_fp8moe_weight_read_group(80, 0, 81, ctypes.byref(group), ctypes.byref(contract)) != 0
    assert lib.mock_initialized() == 0  # The contract query must not initialize CUDA.
    contexts = []
    for rank in (0, 1):
        lib.mock_device(rank)
        context = ctypes.c_void_p()
        assert lib.cuteafd_fp8moe_create(ctypes.byref(context)) == 0
        contexts.append(context)
    assert lib.mock_initialized() == 1
    pointers = (ctypes.c_void_p * 11)(*([1] * 11))
    for rank in (0, 1):
        lib.mock_device(rank)
        assert lib.cuteafd_fp8moe_launch(contexts[rank], 80, pointers, 80, None) == 0
    for index in range(11):
        pointers[index] = None
        status = lib.cuteafd_fp8moe_launch(contexts[1], 80, pointers, 80, None)
        assert (status == 0) == (geometry == "step5_bf16" and index in (4, 6, 8))
        pointers[index] = 1
    lib.cuteafd_fp8moe_destroy(contexts[0])
    assert lib.mock_unloaded() == 0
    lib.mock_fail(2)
    lib.mock_device(2)
    failed = ctypes.c_void_p()
    assert lib.cuteafd_fp8moe_create(ctypes.byref(failed)) == 1
    assert not failed.value and lib.mock_unloaded() == 0
    lib.mock_device(1)
    assert lib.cuteafd_fp8moe_launch(contexts[1], 80, pointers, 80, None) == 0
    lib.cuteafd_fp8moe_destroy(contexts[1])
    assert lib.mock_unloaded() == 1
    lib.mock_device(2)
    assert lib.cuteafd_fp8moe_create(ctypes.byref(failed)) == 1
    assert lib.mock_unloaded() == 2
    lib.mock_fail(-1)
    lib.mock_device(0)
    assert lib.cuteafd_fp8moe_create(ctypes.byref(failed)) == 0
    assert lib.mock_initialized() == 3
    assert lib.cuteafd_fp8moe_launch(failed, 80, pointers, 80, None) == 0
    lib.cuteafd_fp8moe_destroy(failed)
    assert lib.mock_unloaded() == 3


def write_package(path: Path, *, input_kind="wire", role="spark", revision=REVISION, weights="fp8"):
    layout = "tp4" if role == "spark" else "tp1"
    library = path / layout / "libcuteafd_fp8moe.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"fixture library")
    manifest = {
        "schema": "cuteafd.fp8moe-package.v1", "role": role, "geometry": "mimo",
        "sparkinfer_revision": revision,
        "compute": [12, 1],
        "layouts": {layout: {"input": input_kind, "capacities": [{"capacity": 80}],
                             "weights": weights, "abi_version": 5, "activation": "silu_bf16_product",
                             "projection": "bf16", "hidden": 4096, "experts": 256, "top_k": 8,
                             "intermediate": 2048, "tp": 4, "slice": 512, "swiglu_limit": 0.0}},
        "files": {str(library.relative_to(path)): {
            "bytes": library.stat().st_size, "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def stage(tmp_path: Path, kind: str, *, requested="mimo", role="expert", revision=REVISION,
          families="mimo:fp8;glm:nvfp4"):
    built = tmp_path / "build"
    native = built / "native/fp8"
    output = tmp_path / "output"
    built.mkdir()
    # Release compiles/stages a source copy, WIP runs directly from its source tree.
    (built / "source").symlink_to(REPO, target_is_directory=True)
    wire = "wire" if role == "expert" else "bf16"
    owner = "spark" if role == "expert" else "coordinator"
    write_package(native / "fp8-mimo", input_kind=wire, role=owner)
    for name in ("fp8-glm-nvfp4", "fp8-glm-nvfp4a4", "fp8-qwen4-nvfp4", "fp8-qwen4-nvfp4a4"):
        write_package(native / name, input_kind="bf16", role=owner, weights="nvfp4")
    write_package(native / "fp8-step5", input_kind=wire, role=owner)
    write_package(native / "fp8-step5-bf16-weights", input_kind="bf16", role=owner, weights="bf16")
    write_package(native / "fp8-step5-nvfp4a4", input_kind="bf16", role=owner, weights="nvfp4")
    if role == "coordinator":
        write_package(native / "fp8-qwen4", input_kind=wire, role=owner)
    write_package(native / "fp8-mimo-bf16", input_kind="bf16", revision=revision)
    # An earlier opt-in in the same slot must disappear after opting out.
    write_package(output / "fp8/fp8-mimo-bf16", input_kind="bf16")
    script = (REPO / f"scripts/build/build-{kind}-artifacts.sh").read_text()
    body = script.split("# Exact FP8 expert packages", 1)[1].split("\ninstall -m 0644", 1)[0]
    body = "# Exact FP8 expert packages" + body
    # The release script initializes this array in its preceding EXL3 staging block.
    preamble = "set -euo pipefail\nIFS=';' read -ra release_family_list <<<\"$expert_families\"\n"
    result = subprocess.run(["bash", "-c", preamble + body], capture_output=True, text=True,
                            env={**os.environ, "build_root": str(built), "build_dir": str(built),
                                 "source_dir": str(REPO), "output_dir": str(output), "role": role,
                                 "expert_families": families, "bf16_families": requested},
                            timeout=30)
    return result, output / "fp8"


@pytest.mark.parametrize("kind", ["release", "wip"])
@pytest.mark.parametrize("role", ["coordinator", "expert"])
def test_step5_stages_native_weights_and_one_nvfp4_package(tmp_path, kind, role):
    result, output = stage(tmp_path, kind, requested="", role=role,
                           families="step5:fp8;step5:bf16;step5:nvfp4")
    assert result.returncode == 0, result.stderr
    assert {p.name for p in output.iterdir()} == {
        "fp8-step5", "fp8-step5-bf16-weights", "fp8-step5-nvfp4a4", "fp8-mimo-bf16"}
    package_tool.verify(output / "fp8-step5-bf16-weights", revision=REVISION,
                        role=role, input_kind="bf16")


@pytest.mark.parametrize("role", ["spark", "coordinator"])
def test_nvfp4_primary_input_is_bf16_only(role):
    assert package_tool.input_kind(role, "nvfp4") == "bf16"
    assert package_tool.input_kind(role, "nvfp4", "bf16") == "bf16"
    with pytest.raises(ValueError, match="NVFP4 requires BF16"):
        package_tool.input_kind(role, "nvfp4", "wire")
    for weights in ("fp8", "mxfp4"):
        assert package_tool.input_kind(role, weights) == ("wire" if role == "spark" else "bf16")


def test_legacy_nvfp4_wire_artifact_cannot_be_staged(tmp_path):
    legacy = write_package(tmp_path / "legacy", weights="nvfp4")
    with pytest.raises(ValueError, match="NVFP4 requires BF16"):
        package_tool.verify(legacy)


@pytest.mark.parametrize("role", ["coordinator", "expert"])
def test_release_stages_nvfp4_mtp_sibling_only_on_coordinator(tmp_path, role):
    result, output = stage(tmp_path, "release", requested="", role=role,
                           families="qwen4:nvfp4")
    assert result.returncode == 0, result.stderr
    expected = {"fp8-qwen4-nvfp4", "fp8-qwen4-nvfp4a4", "fp8-mimo-bf16"}
    if role == "coordinator":
        expected.add("fp8-qwen4")
    assert {p.name for p in output.iterdir()} == expected


@pytest.mark.parametrize("kind", ["release", "wip"])
def test_requested_sibling_is_verified_and_staged_with_existing_a8_packages(tmp_path, kind):
    result, output = stage(tmp_path, kind)
    assert result.returncode == 0, result.stderr
    assert {p.name for p in output.iterdir()} == {
        "fp8-mimo", "fp8-mimo-bf16", "fp8-glm-nvfp4", "fp8-glm-nvfp4a4"}
    package_tool.verify(output / "fp8-mimo-bf16", revision=REVISION, role="expert", input_kind="bf16",
                        layout="tp4", min_capacity=64)


@pytest.mark.parametrize("kind", ["release", "wip"])
def test_default_opt_out_removes_stale_sibling(tmp_path, kind):
    result, output = stage(tmp_path, kind, requested="")
    assert result.returncode == 0, result.stderr
    assert not (output / "fp8-mimo-bf16").exists()
    assert (output / "fp8-mimo").is_dir()


@pytest.mark.parametrize("kind", ["release", "wip"])
def test_sibling_from_another_pin_cannot_be_shipped(tmp_path, kind):
    result, _ = stage(tmp_path, kind, revision="a" * 40)
    assert result.returncode != 0
    assert "SparkInfer revision" in result.stderr and "fp8-mimo-bf16" in result.stderr


@pytest.mark.parametrize("kwargs, error", [
    ({"role": "coordinator"}, "role spark differs from coordinator"),
    ({"input_kind": "wire"}, "input bf16 differs from wire"),
    ({"layout": "tp2"}, "no tp2 layout"),
    ({"min_capacity": 4096}, "no capacity for 4096 rows"),
    ({"min_capacity": 0}, "minimum capacity must be positive"),
])
def test_capability_checks_reject_an_unusable_package(tmp_path, kwargs, error):
    package = write_package(tmp_path / "package", input_kind="bf16")
    with pytest.raises(ValueError, match=error):
        package_tool.verify(package, **kwargs)


def test_cli_admits_only_the_selected_same_pin_decode_layout(tmp_path):
    package = write_package(tmp_path / "package", input_kind="bf16")
    result = subprocess.run([
        "python3", str(TOOL), "verify", "--package", str(package), "--sparkinfer-revision", REVISION,
        "--role", "expert", "--input", "bf16", "--layout", "tp4", "--min-capacity", "64",
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verified"] is True


@pytest.mark.parametrize("bf16_arg, expected", [
    (None, ""), ("__legacy__", ""), ("mimo,glm", "mimo;glm"),
])
def test_remote_opt_in_preserves_preceding_optional_arguments(bf16_arg, expected):
    text = (REPO / "build.sh").read_text()
    decoder = 'set -euo pipefail\nremote_dir="$1"' + text.split(
        'set -euo pipefail\nremote_dir="$1"', 1)[1].split('\ncd "$remote_dir"', 1)[0]
    arguments = ["/source", "dev", "inference", "engine", REVISION, "version",
                 "__legacy__", "tp2,tp6", "__legacy__", "mimo:fp8,glm:fp8"]
    if bf16_arg is not None:
        arguments.append(bf16_arg)
    result = subprocess.run(["bash", "-c", decoder +
                             '\nprintf "%s\\n" "$spark_tp_roles" "$expert_families" "$bf16_families"',
                             "decoder", *arguments], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["tp2;tp6", "mimo:fp8;glm:fp8", expected]


def test_preflight_checks_real_package_files_and_matching_native_contract(tmp_path):
    write_package(tmp_path / "fp8-mimo")
    write_package(tmp_path / "fp8-mimo-bf16", input_kind="bf16")
    preflight.verify_sibling(tmp_path, "mimo", "tp4", 64, REVISION, package_tool.verify)
    library = tmp_path / "fp8-mimo-bf16/tp4/libcuteafd_fp8moe.so"
    library.write_bytes(b"corrupt library")
    with pytest.raises(ValueError):
        preflight.verify_sibling(tmp_path, "mimo", "tp4", 64, REVISION, package_tool.verify)


@pytest.mark.parametrize("field, value, error", [
    ("sparkinfer_revision", "a" * 40, "source pin"),
    ("compute", [12, 0], "SM121"),
    ("geometry", "glm", "geometry mimo"),
    ("role", "coordinator", "Spark SM121"),
    ("input", "wire", "bf16 tp4"),
    ("slice", 384, "differs from primary in slice"),
    ("swiglu_limit", 7.0, "differs from primary in swiglu_limit"),
    ("capacities", [{"capacity": 16}], "capacity for 64"),
])
def test_preflight_rejects_incompatible_sibling_before_cuda(tmp_path, field, value, error):
    write_package(tmp_path / "fp8-mimo")
    sibling = write_package(tmp_path / "fp8-mimo-bf16", input_kind="bf16")
    manifest = json.loads((sibling / "manifest.json").read_text())
    target = manifest if field in ("sparkinfer_revision", "compute", "geometry", "role") else manifest["layouts"]["tp4"]
    target[field] = value
    (sibling / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=error):
        preflight.verify_sibling(tmp_path, "mimo", "tp4", 64, REVISION, package_tool.verify)


@pytest.mark.parametrize("layout, capacity, revision, error", [
    ("tp6", 64, REVISION, "wire tp6"),
    ("tp4", 4096, REVISION, "capacity for 4096"),
    ("tp4", 64, "unknown", "does not identify"),
])
def test_preflight_requires_requested_layout_workspace_and_image_identity(tmp_path, layout, capacity, revision, error):
    write_package(tmp_path / "fp8-mimo")
    write_package(tmp_path / "fp8-mimo-bf16", input_kind="bf16")
    with pytest.raises(ValueError, match=error):
        preflight.verify_sibling(tmp_path, "mimo", layout, capacity, revision, package_tool.verify)


@pytest.mark.parametrize("counts,geometry,expected", [
    ("glm=6;qwen4=4", "qwen4", "LAYOUTS=tp4-rank0;tp4-rank1;tp4-rank2;tp4-rank3"),
    ("glm=6", "qwen4", "LAYOUTS=tp1-rank0"),  # no count for qwen4: packaged defaults
])
def test_cmake_exl3_family_counts_only_apply_to_their_family(tmp_path, counts, geometry, expected):
    source = (REPO / "native/cmake/shared/exl3.cmake").read_text()
    body = source.split("  set(CUTEAFD_EXL3_PROFILE_ARGS)", 1)[1].split('  list(JOIN CUTEAFD_EXL3_GEOMETRY_LAYOUTS', 1)[0]
    script = tmp_path / "select.cmake"
    script.write_text('set(CUTEAFD_EXL3_ROLE spark)\n'
                      f'set(CUTEAFD_EXL3_GEOMETRY {geometry})\nset(CUTEAFD_EXL3_GEOMETRY_LAYOUTS tp1-rank0)\n'
                      f'set(CUTEAFD_GENERIC_SPARK_COUNTS "{counts}")\n' + body
                      + '\nmessage("LAYOUTS=${CUTEAFD_EXL3_GEOMETRY_LAYOUTS}")\n')
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert expected in result.stderr


def test_cmake_exl3_own_family_invalid_count_still_fails(tmp_path):
    source = (REPO / "native/cmake/shared/exl3.cmake").read_text()
    body = source.split("  set(CUTEAFD_EXL3_PROFILE_ARGS)", 1)[1].split('  list(JOIN CUTEAFD_EXL3_GEOMETRY_LAYOUTS', 1)[0]
    script = tmp_path / "select.cmake"
    script.write_text('set(CUTEAFD_EXL3_ROLE spark)\nset(CUTEAFD_EXL3_GEOMETRY qwen4)\n'
                      'set(CUTEAFD_GENERIC_SPARK_COUNTS "qwen4=6")\n' + body)
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "requires a nonempty H128 slice" in result.stderr


def test_cmake_fp8_family_counts_only_apply_to_their_family(tmp_path):
    source = (REPO / "native/cmake/shared/fp8_moe.cmake").read_text()
    body = source.split("  set(requested_layouts)", 1)[1].split("  set(stamp ", 1)[0]
    for family, expected in (("glm", "LAYOUTS=--layouts;tp6"), ("qwen4", "LAYOUTS=--layouts;tp4"), ("mimof", "LAYOUTS=")):
        script = tmp_path / "select.cmake"
        script.write_text('set(CUTEAFD_FP8_MOE_ROLE spark)\n'
                          f'set(count_family {family})\nset(CUTEAFD_GENERIC_SPARK_COUNTS "glm=6;qwen4=4")\n'
                          'set(requested_layouts)\n' + body + '\nmessage("LAYOUTS=${requested_layouts}")\n')
        result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert expected in result.stderr and (family != "mimof" or "tp" not in result.stderr)


@pytest.mark.parametrize("kind,storage", [("fp8", 1), ("mxfp4", 2), ("nvfp4a4", 4), ("bf16", 5)])
@pytest.mark.parametrize("activation,code", [("silu_bf16_product", 1), ("silu_post_clamp_fp32_product", 2)])
def test_explicit_activation_info_words(kind, storage, activation, code):
    from types import SimpleNamespace
    g = SimpleNamespace(kind=kind, weights="nvfp4" if kind == "nvfp4a4" else kind,
                        activation=activation, projection="bf16", swiglu_limit=7.0,
                        hidden=4096, slice=256, experts=352, top_k=8, intermediate=1536, tp=6)
    if kind == "mxfp4" and code == 2:
        with pytest.raises(ValueError, match="MXFP4"):
            package_tool.info_words(g, [16], False)
        return
    words = package_tool.info_words(g, [1, 16, 80, 256, 1024, 4096], False)
    assert len(words) == 16
    assert words[0] == 5 | storage << 8 | code << 16 | 1 << 24
    assert words[9:] == [6, 1, 16, 80, 256, 1024, 4096]
    assert words[0] not in (1, 2, 3, 4)  # old runtimes reject rather than guess
    for field, value in [("activation", "unknown"), ("projection", "fp32"), ("swiglu_limit", float("nan"))]:
        bad = SimpleNamespace(**vars(g))
        setattr(bad, field, value)
        with pytest.raises(ValueError):
            package_tool.info_words(bad, [16], False)


def test_mxfp4_package_rejects_post_activation(tmp_path):
    path = write_package(tmp_path / "package", weights="mxfp4")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["layouts"]["tp4"]["activation"] = "silu_post_clamp_fp32_product"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="MXFP4 supports only"):
        package_tool.verify(path)


@pytest.mark.parametrize("field,value", [("activation", None), ("activation", "unknown"),
                                         ("projection", "fp32"), ("abi_version", 4)])
def test_package_requires_explicit_activation_contract(tmp_path, field, value):
    path = write_package(tmp_path / "package")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["layouts"]["tp4"][field] = value
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="activation/projection ABI"):
        package_tool.verify(path)
