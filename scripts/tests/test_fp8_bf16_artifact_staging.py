"""Opt-in BF16 input packages must survive staging and retain their build identity."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "python/tools/aot/package_fp8_moe_aot.py"
REVISION = json.loads((REPO / "third_party/sparkinfer.lock.json").read_text())["revision"]
spec = importlib.util.spec_from_file_location("fp8_package", TOOL)
package_tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
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
        if tp > intermediate // 128:
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


def test_qwen_tp5_exact_slice_has_no_padding():
    assert package_tool.exact_widths(640, 5) == [128] * 5
    assert "tp5" in package_tool.spark_layouts(640)
    assert not any(layout.startswith(("tp6", "tp7", "tp8")) for layout in package_tool.spark_layouts(640))


@pytest.mark.parametrize("counts,expected", [("", ""), ("5", "--layouts;tp5"),
                                            ("1;5;7;8", "--layouts;tp1,tp5,tp7,tp8")])
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


@pytest.mark.parametrize("counts", ["0", "9", "5;bad"])
def test_cmake_requested_counts_fail_before_compilation(tmp_path, counts):
    source = (REPO / "native/cmake/shared/fp8_moe.cmake").read_text()
    body = source.split("  set(requested_layouts)", 1)[1].split("  set(stamp ", 1)[0]
    script = tmp_path / "select.cmake"
    script.write_text('set(CUTEAFD_FP8_MOE_ROLE spark)\n'
                      f'set(CUTEAFD_GENERIC_SPARK_COUNTS "{counts}")\n' + body)
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "requires counts 1..8" in result.stderr


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
    if count > blocks:
        assert result.returncode != 0
        assert "requires a nonempty H128 slice" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    assert 'LAYOUTS=' + ';'.join(f'tp{count}-rank{rank}' for rank in range(count)) in result.stderr
    widths = sorted({(blocks // count + (rank < blocks % count)) * 128 for rank in range(count)}, reverse=True)
    profiles = ';'.join(part for width in widths for part in ('--profile', f'tp{count}-width{width}'))
    assert f'PROFILES={profiles}' in result.stderr


def write_package(path: Path, *, input_kind="wire", role="spark", revision=REVISION):
    layout = "tp4" if role == "spark" else "tp1"
    library = path / layout / "libcuteafd_fp8moe.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"fixture library")
    manifest = {
        "schema": "cuteafd.fp8moe-package.v1", "role": role, "geometry": "mimo",
        "sparkinfer_revision": revision,
        "compute": [12, 1],
        "layouts": {layout: {"input": input_kind, "capacities": [{"capacity": 80}],
                             "weights": "fp8", "hidden": 4096, "experts": 256, "top_k": 8,
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
    for name in ("fp8-mimo", "fp8-glm-nvfp4", "fp8-glm-nvfp4a4"):
        write_package(native / name, input_kind=wire, role=owner)
    for name in ("fp8-qwen4-nvfp4", "fp8-qwen4-nvfp4a4"):
        write_package(native / name, input_kind=wire, role=owner)
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
    arguments = ["/source", "dev", "inference", "engine", REVISION, "version", "OFF",
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
