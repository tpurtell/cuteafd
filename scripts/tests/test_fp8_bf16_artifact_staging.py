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


def test_coordinator_exports_tp2_for_every_routed_weight_format():
    for layouts in (package_tool.ROLE_LAYOUTS, package_tool.MXFP4_ROLE_LAYOUTS,
                    package_tool.NVFP4_ROLE_LAYOUTS):
        assert layouts["coordinator"] == ("tp1", "tp2")
    assert package_tool.ROLE_INPUT["coordinator"] == "bf16"
    cmake = (REPO / "native/cmake/shared/fp8_moe.cmake").read_text()
    assert 'set(exact_slices "--exact-slices")' in cmake


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
