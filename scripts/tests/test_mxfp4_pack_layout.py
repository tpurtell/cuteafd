"""Compile the real native inverse map and compare an independent tensor permutation."""
import ast
import ctypes
import importlib.util
from pathlib import Path
import shutil
import sys
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mxfp4_layouts", ROOT / "python/tools/lib/mxfp4_layouts.py")
LAYOUTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAYOUTS)
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "mxfp4_native_packages", ROOT / "python/tools/lib/spark_packages.py")
PACKAGES = importlib.util.module_from_spec(PACKAGE_SPEC)
PACKAGE_SPEC.loader.exec_module(PACKAGES)
WIDTHS = sorted({artifact.logical for family in ("v41", "dsv4f", "dsv4p")
                 for artifacts in PACKAGES.native_spark_artifacts(family).values()
                 for artifact in artifacts} | {32, 64, 96})


@pytest.fixture(scope="module")
def native_map(tmp_path_factory):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    directory = tmp_path_factory.mktemp("mxfp4-pack-map")
    (directory / "cuteafd_mxfp4_layouts.h").write_text(LAYOUTS.header())
    source = directory / "map.c"
    source.write_text(r'''
#include "expert_pack_layout.h"
#include "cuteafd_mxfp4_layouts.h"
void coordinates(uint64_t* output, uint64_t count, uint32_t n, uint32_t k,
    uint32_t half, int scales, int compact) {
  for (uint64_t i = 0; i < count; ++i) {
    uint32_t row, col;
    cuteafd_expert_pack_layout_coordinates(i, n, k, half, scales, compact, &row, &col);
    output[i] = ((uint64_t)row << 32) | col;
  }
}
uint32_t stored(uint32_t hidden, uint32_t width) {
  return cuteafd_mxfp4_stored_width(hidden, width);
}
uint32_t ownership(uint32_t intermediate, uint32_t world) {
  return cuteafd_mxfp4_ownership_unit(intermediate, world);
}
''')
    module = directory / "map.so"
    subprocess.run([compiler, "-std=c11", "-shared", "-fPIC", "-O2", "-Wall", "-Wextra",
                    "-Werror", "-I", str(ROOT / "native/shared/cuda"), "-I", str(directory),
                    str(source), "-o", str(module)], check=True, capture_output=True)
    library = ctypes.CDLL(str(module))
    library.coordinates.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
                                   ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                   ctypes.c_int, ctypes.c_int]
    library.coordinates.restype = None
    library.stored.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    library.stored.restype = ctypes.c_uint32
    library.ownership.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    library.ownership.restype = ctypes.c_uint32
    return library


def tensor_permutation(n, k, half, scales, compact):
    columns = k // (32 if scales else 8)
    source = ((np.arange(n, dtype=np.uint64)[:, None] << np.uint64(32))
              | np.arange(columns, dtype=np.uint64)[None, :])
    planes = np.split(source, 2, axis=0) if compact and half else [source]
    result = []
    for plane in planes:
        for nt in range(0, plane.shape[0], 128 if compact else 256):
            for kt in range(0, columns, 4 if scales else 16):
                piece = plane[nt:nt + (128 if compact else 256), kt:kt + (4 if scales else 16)]
                if not scales:
                    rows, cols = piece.shape
                    piece = piece.reshape(rows // 32, 4, 8, cols // 4, 4).transpose(3, 0, 2, 4, 1)
                result.append(piece.ravel())
    return np.concatenate(result)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_exact_k32_inverse_mapping_is_a_complete_bijection(native_map, width, scales, gated):
    # The native loader deliberately retains the N256 layout for aligned widths.
    compact = bool(width % 128)
    n, k, half = (2 * width, 512, width) if gated else (512, width, 0)
    expected = tensor_permutation(n, k, half, scales, compact)
    actual = np.empty_like(expected)
    native_map.coordinates(actual.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                           actual.size, n, k, half, scales, compact)
    np.testing.assert_array_equal(actual, expected)
    assert np.unique(actual).size == n * (k // (32 if scales else 8))


def test_measured_storage_holds_and_generated_native_header_agree(native_map):
    for hidden in (4096, 5120, 7168, 8192):
        for width in WIDTHS:
            assert native_map.stored(hidden, width) == LAYOUTS.stored_width(hidden, width)
    assert LAYOUTS.stored_width(5120, 576) == 576
    for hidden, width in [(0, 32), (255, 32), (5120, 0), (5120, 33), (5120, 8224)]:
        assert native_map.stored(hidden, width) == 0
        with pytest.raises(ValueError):
            LAYOUTS.stored_width(hidden, width)


@pytest.mark.parametrize("tile", [32, 64, 128, 192])
@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("compact", [False, True])
def test_exporter_fc1_partial_matches_kernel_compact_activation_contract(tile, hidden, compact):
    source = ROOT / "python/tools/aot/export_b12x_slices_aot.py"
    tree = ast.parse(source.read_text())
    assignment, = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "partial_cols"
                           for target in node.targets)]
    expression = compile(ast.Expression(assignment.value), str(source), "eval")
    # Evaluate the exporter's actual pure expression, without importing CUDA.
    # family is intentionally absent: raw/coded weight scales share FC1 storage.
    cols = eval(expression, {"__builtins__": {}},
                dict(selected_width=tile, hidden=hidden, compact=compact))
    assert cols == (hidden if compact else tile // 4 + tile // 32)
    kernel = (ROOT / "third_party/sparkinfer/b12x/moe/_shared/kernels/w4a8_v41_slice.py").read_text()
    assert kernel.count("self.slices * self.routes * (self.width // 4 + self.width // 32)") >= 2
    assert "qs_offset = Int64(self.slices * self.routes * (self.width // 4))" in kernel


def test_native_plan_rejects_mixed_decode_tilings():
    data = LAYOUTS.layouts()
    data["native_slices"][0]["slices"][1] = 64
    with pytest.raises(ValueError, match="decode capacities must share"):
        LAYOUTS.validate(data)


def test_only_qualified_hardware_recipes_override_matching_arithmetic():
    assert LAYOUTS.native_slice_width(5120, 288, 121, 1, 192) == 128
    assert LAYOUTS.native_slice_width(5120, 288, 121, 16, 192) == 128
    assert LAYOUTS.native_slice_width(5120, 288, 121, 256, 192) == 64
    assert LAYOUTS.native_slice_width(5120, 288, 120, 1, 192) == 192
    assert LAYOUTS.native_slice_width(4096, 352, 121, 16, 128) == 32
    assert LAYOUTS.native_slice_width(4096, 352, 121, 256, 128) == 128
    assert LAYOUTS.native_slice_width(4096, 352, 120, 16, 128) == 128
    assert LAYOUTS.native_slice_width(5120, 576, 121, 80, 192) == 128
    assert LAYOUTS.native_slice_width(5120, 576, 121, 256, 192) == 192
    assert LAYOUTS.ownership_unit(2304, 5) == 64
    assert LAYOUTS.ownership_unit(2304, 7) == 32
    assert LAYOUTS.generic_stored_width(6144, 352) == 384
    assert LAYOUTS.generic_stored_width(4096, 352) == 352


def test_native_plan_rejects_duplicate_capacity_keys():
    data = LAYOUTS.layouts()
    data["native_slices"][0]["capacities"][1] = 1
    with pytest.raises(ValueError, match="pair uniquely"):
        LAYOUTS.validate(data)


@pytest.mark.parametrize("family,hidden,intermediate", [
    ("v41", 5120, 2304), ("dsv4f", 4096, 2048), ("dsv4p", 7168, 3072),
])
def test_world_artifacts_match_every_absolute_layer_source_interval(family, hidden, intermediate):
    spec = importlib.util.spec_from_file_location("mxfp4_native_packages", ROOT / "python/tools/lib/spark_packages.py")
    packages = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packages)
    worlds = packages.native_spark_artifacts(family)
    assert set(worlds) == set(range(2, 9))
    if family == "v41":
        assert [a.logical for a in worlds[5]] == [512, 448]
    for world, artifacts in worlds.items():
        unequal = (intermediate // LAYOUTS.ownership_unit(intermediate, world)) % world != 0
        assert len(artifacts) == (2 if unequal else 1)
        for artifact in artifacts:
            assert artifact.stored == LAYOUTS.stored_width(hidden, artifact.logical)
            assert artifact.stored <= -(-artifact.logical // 128) * 128
            if unequal:
                assert artifact.role == 65536 + artifact.logical
                assert artifact.name == f"spark_w{artifact.logical}"
            else:
                assert artifact.role == {2: 5, 3: 6, 4: 1, 6: 7, 8: 8}[world]
        for layer in (0, 1, 17, 63):
            shards = sorted(packages.native_shard(world, rank, layer, family) for rank in range(world))
            cursor = 0
            for first, artifact in shards:
                assert first == cursor
                assert artifact in artifacts
                cursor += artifact.logical
            assert cursor == intermediate


def test_generated_ownership_policy_matches_independent_native_fixture(native_map):
    for intermediate in (2048, 2304, 3072):
        for world in range(1, 9):
            assert native_map.ownership(intermediate, world) == LAYOUTS.ownership_unit(intermediate, world)


@pytest.mark.parametrize("family", ["v41", "dsv4f", "dsv4p"])
@pytest.mark.parametrize("world", range(2, 9))
def test_cmake_native_artifact_selection_uses_world_inventory(tmp_path, family, world):
    generated = tmp_path / "artifacts.cmake"
    subprocess.run([sys.executable, str(ROOT / "python/tools/lib/spark_packages.py"),
                    "--cmake", str(generated)], check=True, capture_output=True, text=True)
    source = (ROOT / "native/cmake/shared/mxfp4_widths.cmake").read_text()
    selection = source[:source.index("foreach(key IN LISTS native_width_keys)")]
    tp_source = (ROOT / "native/cmake/families/deepseek_v41/v41_spark_tp_experts.cmake").read_text()
    tp_selection = tp_source[tp_source.index("set(CUTEAFD_V41_SPARK_TP_SELECTED)"):
                             tp_source.index("set(CUTEAFD_V41_SPARK_TP_EXPERT_TARGETS)")]
    family_source = (ROOT / "native/cmake/shared/expert_families.cmake").read_text()
    family_selection = family_source[family_source.index("foreach(entry IN LISTS CUTEAFD_NATIVE_EXPERT_FAMILIES)"):
                                     family_source.index('  set(device_handles "")')]
    family_selection += '\nlist(APPEND selected_family_roles "${family}:${role}")\nendforeach()\n'
    script = tmp_path / "select.cmake"
    script.write_text('cmake_minimum_required(VERSION 3.24)\n'
                      f'include("{generated}")\n'
                      'set(CUTEAFD_ENABLE_V41_EXPERT_AOT ON)\n'
                      'set(CUTEAFD_V41_EXPERT_ROLE spark)\n'
                      f'set(CUTEAFD_SPARK_TP_ROLES tp{world})\n'
                      f'set(CUTEAFD_EXPERT_FAMILIES {family}:spark_tp{world})\n'
                      + selection + '\n' + tp_selection
                      + f'\nset(CUTEAFD_NATIVE_EXPERT_FAMILIES dsv4f:spark_tp{world} dsv4p:spark_tp{world})\n'
                      + family_selection
                      + '\nmessage(STATUS "keys=${native_width_keys}")\n'
                      + 'message(STATUS "equal=${CUTEAFD_V41_SPARK_TP_SELECTED}")\n'
                      + 'message(STATUS "families=${selected_family_roles}")\n')
    result = subprocess.run(["cmake", "-P", str(script)], check=True, capture_output=True,
                            text=True, timeout=30)
    keys = {f"{f}:{a.logical}" for f in {"v41", family}
            for a in PACKAGES.native_spark_artifacts(f)[world] if a.name.startswith("spark_w")}
    actual = next(line.removeprefix("-- keys=") for line in result.stdout.splitlines()
                  if line.startswith("-- keys="))
    assert set(actual.split(";")) - {""} == keys
    equal = next(line.removeprefix("-- equal=") for line in result.stdout.splitlines()
                 if line.startswith("-- equal="))
    expected = [f"tp{world}" for a in PACKAGES.native_spark_artifacts()[world]
                if a.name == f"spark_tp{world}"]
    assert (equal.split(";") if equal else []) == expected
    selected = next(line.removeprefix("-- families=") for line in result.stdout.splitlines()
                    if line.startswith("-- families="))
    expected_families = {f"{f}:{a.name}" for f in ("dsv4f", "dsv4p")
                         for a in PACKAGES.native_spark_artifacts(f)[world]
                         if a.name == f"spark_tp{world}"}
    assert set(selected.split(";")) - {""} == expected_families


def test_exporter_admission_matches_only_world_artifacts():
    spec = importlib.util.spec_from_file_location(
        "native_export_artifacts", ROOT / "python/tools/aot/export_b12x_slices_aot.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    for family, shape in exporter.FAMILY_GEOMETRY.items():
        artifacts = {artifact.name: artifact for group in PACKAGES.native_spark_artifacts(family).values()
                     for artifact in group}
        for name, artifact in artifacts.items():
            assert exporter.role_geometry(family, name) == (
                shape["experts"], artifact.logical, artifact.stored, shape["topk"])
        for width in range(32, shape["intermediate"] + 1, 32):
            name = f"spark_w{width}"
            if name not in artifacts:
                with pytest.raises(ValueError, match="no native Spark artifact"):
                    exporter.role_geometry(family, name)


@pytest.fixture(scope="module")
def native_forward_map(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("mxfp4-forward-map")
    text = (ROOT / "native/tests/v41_expert_pack_geometry_selftest.cc").read_text()
    compact = text[text.index("uint64_t compact_index("):text.index("// Direct port of the device")]
    payload = text[text.index("uint64_t payload_index("):text.index("void claim_w13(")]
    source = directory / "forward.cc"
    source.write_text('#include <algorithm>\n#include <cstdint>\n' + compact + payload + r'''
extern "C" void forward(uint64_t* output, uint32_t n, uint32_t k, uint32_t half,
                        int scales, int compact) {
  const uint32_t columns = k / (scales ? 32 : 8);
  for (uint32_t row = 0; row < n; ++row)
    for (uint32_t col = 0; col < columns; ++col) {
      uint64_t index;
      if (compact) index = compact_index(row, col, n, columns, half, scales);
      else if (!scales) index = payload_index(row, col, n, k, half, false);
      else index = (uint64_t(row / 256) * ((k + 127) / 128) + col / 4) * 1024
          + (row % 256) * 4 + col % 4;
      output[index] = (uint64_t(row) << 32) | col;
    }
}
''')
    module = directory / "forward.so"
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O2", "-Wall", "-Wextra",
                    "-Werror", str(source), "-o", str(module)], check=True, capture_output=True)
    library = ctypes.CDLL(str(module))
    library.forward.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.c_uint32, ctypes.c_int, ctypes.c_int]
    library.forward.restype = None
    return library


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_native_gpu_selftest_forward_oracle_matches_tensor_permutation(
        native_forward_map, width, scales, gated):
    compact = bool(width % 128)
    n, k, half = (2 * width, 512, width) if gated else (512, width, 0)
    expected = tensor_permutation(n, k, half, scales, compact)
    actual = np.full_like(expected, np.iinfo(np.uint64).max)
    native_forward_map.forward(actual.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                               n, k, half, scales, compact)
    np.testing.assert_array_equal(actual, expected)
