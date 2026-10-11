#!/usr/bin/env python3
"""Executable coverage for the Spark TP role plan of `build.sh` and `wip.sh`, plus
the `write-v41-expert-tp-manifest.py` role table.

No hardware, Docker, SSH, cmake or cargo is touched. The test extracts the real
role-selection block from each script by its boundary comments, sources the real
`scripts/lib/release-common.sh` for `release_die`/`release_spark_topology_explicit`,
and runs the block in bash under a matrix of configurations. A change to the
allowlist, the default set or the multi-role syntax fails here rather than at
release time.

The two producers resolve roles differently on purpose, and both halves are
asserted:

* `build.sh` resolves the universal release set (`tp2;tp3;tp6`) regardless of the
  configured topology, so one published image pair serves every approved native
  topology. Detailed malformed-input coverage lives in
  `test_release_universal_images.py`.
* `wip.sh` derives its default from the same shared release role configuration.
  Shared slots cover the actual TP3 minimum and TP6 reference card. An explicit
  subset override (including empty for legacy TP4) remains available.

The manifest half writes a synthetic role export for a requested role and checks
that the writer accepts the geometry it was told to expect (tp6: intermediate
384, no padding) and rejects a mismatched one, so a future geometry disagreement
cannot pass silently.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build.sh"
WIP = REPO / "wip.sh"
RELEASE_COMMON = REPO / "scripts" / "lib" / "release-common.sh"
MANIFEST = REPO / "scripts" / "build" / "write-v41-expert-tp-manifest.py"

UNIVERSAL = "tp2;tp3;tp6"

# Each script keeps its role plan in one contiguous, marker-delimited block before
# its dry-run exit. wip.sh names the WIP role variable in its usage text earlier in
# the file, so its block is located from the right.
BUILD_START = "# release-spark-tp-roles:start"
WIP_START = "# Opt-in replicated-group Spark expert roles"
BLOCK_END = "if ((dry_run)); then"


def _role_block(script: Path = BUILD, *, variable: str = "spark_tp_roles",
                last: bool = False, marker: str = BUILD_START) -> str:
    text = script.read_text(encoding="utf-8")
    finder = text.rindex if last else text.index
    start = finder(marker)
    end = text.index(BLOCK_END, start)
    block = text[start:end]
    assert f"{variable}=" in block, f"{script.name} role block moved"
    assert "release_spark_tp_roles_canonical" in block, f"{script.name} role validator moved"
    assert "release_spark_tp_roles_default" in block, f"{script.name} role default moved"
    return block


def _run(script: Path, spark_tp: str, override: str | None, *, variable: str,
         env_name: str, block: str) -> subprocess.CompletedProcess:
    harness = f"""
set -euo pipefail
source "{RELEASE_COMMON}"
{block}
printf 'ROLES=%s\\n' "${{{variable}}}"
"""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "SPARK_TP": spark_tp,
        # `release_spark_topology_explicit` reads SPARK_EP under `set -u`; an
        # explicit topology always sets it, so mirror that here.
        "SPARK_EP": "1" if spark_tp else "",
    }
    # An unset override means "use the script's own default"; a set override is the
    # documented escape hatch, and an empty one is the explicit legacy request.
    if override is not None:
        env[env_name] = override
    return subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, env=env,
        timeout=60, check=False,
    )


def _run_roles(spark_tp: str, override: str | None) -> subprocess.CompletedProcess:
    return _run(BUILD, spark_tp, override, variable="spark_tp_roles",
                env_name="CUTEAFD_RELEASE_SPARK_TP_ROLES", block=_role_block())


def _run_wip_roles(spark_tp: str, override: str | None) -> subprocess.CompletedProcess:
    return _run(WIP, spark_tp, override, variable="wip_spark_tp_roles",
                env_name="CUTEAFD_WIP_SPARK_TP_ROLES",
                block=_role_block(WIP, variable="wip_spark_tp_roles", last=True,
                                  marker=WIP_START))


def _roles(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 0, result.stderr
    return result.stdout.rstrip().splitlines()[-1].removeprefix("ROLES=")


@pytest.mark.parametrize("spark_tp", ["", "2", "3", "4", "6"])
def test_build_release_default_is_universal(spark_tp) -> None:
    """No configured topology may narrow what a published image can serve."""
    assert _roles(_run_roles(spark_tp, None)) == UNIVERSAL


@pytest.mark.parametrize(
    "override,expected",
    [
        ("tp2;tp3;tp6", UNIVERSAL),
        ("tp6;tp2;tp3", UNIVERSAL),
        ("tp6", "tp6"),
        ("tp2;tp6", "tp2;tp6"),
        ("", ""),
    ],
)
def test_build_release_subset_escape_hatch(override, expected) -> None:
    assert _roles(_run_roles("4", override)) == expected


def test_build_rejects_an_unknown_role_with_a_clear_error() -> None:
    result = _run_roles("", "tp5")
    assert result.returncode == 2, result.stdout
    assert "accepts only tp2, tp3 and tp6" in result.stderr
    assert "tp5" in result.stderr


@pytest.mark.parametrize(
    "spark_tp,override,expected",
    [
        ("", None, UNIVERSAL),
        ("4", None, UNIVERSAL),
        ("6", None, UNIVERSAL),
        ("2", None, UNIVERSAL),
        ("3", None, UNIVERSAL),
        ("3", "", ""),
        ("6", "", ""),
        ("", UNIVERSAL, UNIVERSAL),
        ("", "tp6", "tp6"),
        ("4", UNIVERSAL, UNIVERSAL),
    ],
)
def test_wip_role_selection_matrix(spark_tp, override, expected) -> None:
    """WIP shares release defaults and validation, with a subset escape hatch."""
    result = _run_wip_roles(spark_tp, override)
    assert _roles(result) == expected


def test_wip_rejects_an_unknown_role_and_names_the_wip_variable() -> None:
    result = _run_wip_roles("", "tp5")
    assert result.returncode == 2, result.stdout
    assert "CUTEAFD_WIP_SPARK_TP_ROLES accepts only tp2, tp3 and tp6" in result.stderr


def test_wip_subset_is_always_contained_in_the_release_default() -> None:
    """A WIP slot may build less than a release, never something else.

    The parity that matters is containment: every role a slot resolves must also be
    baked into the published image, so a slot result transfers to the release path.
    """
    release = set(_roles(_run_roles("", None)).split(";")) - {""}
    assert release == set(UNIVERSAL.split(";"))
    for spark_tp in ("2", "3", "4", "6"):
        slot = set(_roles(_run_wip_roles(spark_tp, None)).split(";")) - {""}
        assert slot <= release, (spark_tp, slot, release)


@pytest.mark.parametrize("tp,width", [(3, 768), (6, 384)])
def test_nvfp4_launcher_admission_matches_resident_planes(tp: int, width: int) -> None:
    result = subprocess.run(
        ["bash", "-c", f'source "{RELEASE_COMMON}"; release_spark_layer_bytes {tp} nvfp4'],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == 384 * (3 * 5120 * width * 9 // 16 + 16)


def test_release_and_wip_artifact_scripts_share_the_allowlist() -> None:
    for path, variable in (
        (REPO / "scripts" / "build" / "build-release-artifacts.sh", "CUTEAFD_RELEASE_SPARK_TP_ROLES"),
        (REPO / "scripts" / "build" / "build-wip-artifacts.sh", "CUTEAFD_WIP_SPARK_TP_ROLES"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "tp2|tp3|tp6)" in text, path
        assert f"{variable} accepts only tp2, tp3 and tp6" in text, path


def _role_geometry(role: str) -> tuple:
    sys.path.insert(0, str(REPO / "python" / "tools"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_tp6_export_tables", REPO / "python" / "tools" / "aot" / "export_b12x_slices_aot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ROLE_GEOMETRY[role]


CAPACITIES = (1, 16, 80, 256, 1024, 4096)


def _write_export(directory: Path, role: str) -> Path:
    import hashlib

    experts, intermediate, kernel_intermediate, topk = _role_geometry(role)
    directory.mkdir(parents=True, exist_ok=True)
    variants = []
    artifact_sha256 = {}
    for capacity in CAPACITIES:
        header = directory / f"v41_{role}_m{capacity}.h"
        obj = directory / f"v41_{role}_m{capacity}.o"
        header.write_text("// stub\n", encoding="utf-8")
        obj.write_bytes(f"stub-object-{capacity}\n".encode())
        variants.append({"name": f"v41_{role}_m{capacity}", "capacity_rows": capacity})
        artifact_sha256[header.name] = hashlib.sha256(header.read_bytes()).hexdigest()
        artifact_sha256[obj.name] = hashlib.sha256(obj.read_bytes()).hexdigest()
    manifest = {
        "schema": 1,
        "role": role,
        "spark_tp_degree": {"spark": 4, "spark_tp2": 2, "spark_tp3": 3, "spark_tp6": 6}[role],
        "capability": [12, 1],
        "sparkinfer_revision": "0" * 40,
        "geometry": {
            "experts": experts,
            "hidden": 5120,
            "intermediate": intermediate,
            "kernel_intermediate": kernel_intermediate,
            "topk": topk,
        },
        "variants": variants,
        "artifact_sha256": artifact_sha256,
    }
    path = directory / "v41_experts.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_writer_accepts_tp6_and_rejects_a_wrong_extent(tmp_path: Path) -> None:
    role = "tp6"
    export_dir = tmp_path / f"v41_spark_{role}_experts"
    _write_export(export_dir, "spark_tp6")
    output = tmp_path / "V41_EXPERT_TP_AOT.json"
    # The writer requires the built library whenever a role is requested; a stub
    # file exercises the hash/validation path without any native build.
    library = tmp_path / "libcuteafd_native.so"
    library.write_bytes(b"\x7fELF-stub\n")

    def invoke() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(MANIFEST), "--role", "expert", "--requested", role,
             "--native-build-dir", str(tmp_path), "--native-library", str(library),
             "--output", str(output)],
            capture_output=True, text=True, timeout=120, check=False,
        )

    result = invoke()
    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["spark_tp_roles"] == ["tp6"]
    assert document["manifests"]["tp6"]["spark_tp_degree"] == 6
    assert document["manifests"]["tp6"]["geometry"]["intermediate"] == 384
    assert document["manifests"]["tp6"]["geometry"]["kernel_intermediate"] == 384

    # A geometry that disagrees with the official 384 must fail closed.
    manifest_path = export_dir / "v41_experts.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["geometry"]["kernel_intermediate"] = 640
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    failed = invoke()
    assert failed.returncode == 2
    assert "kernel_intermediate" in failed.stderr


@pytest.mark.parametrize("role", ["tp3", "tp6"])
def test_nvfp4_manifest_requires_exact_export_and_bf16_input(tmp_path: Path, role: str) -> None:
    native_dir = tmp_path / f"v41_spark_{role}_experts"
    native_path = _write_export(native_dir, f"spark_{role}")
    nvfp4_dir = tmp_path / f"v41_nvfp4_spark_{role}"
    nvfp4_dir.mkdir()
    for artifact in native_dir.iterdir():
        if artifact.suffix in (".o", ".h"):
            (nvfp4_dir / artifact.name).write_bytes(artifact.read_bytes())
    payload = json.loads(native_path.read_text())
    payload.update(quant_mode="nvfp4", input_format="bf16")
    path = nvfp4_dir / "v41_nvfp4_experts.json"
    path.write_text(json.dumps(payload))
    library = tmp_path / "libcuteafd_native.so"
    library.write_bytes(b"stub")
    output = tmp_path / "V41_EXPERT_TP_AOT.json"
    command = [sys.executable, str(MANIFEST), "--role", "expert", "--requested", role,
               "--nvfp4", "--native-build-dir", str(tmp_path), "--native-library", str(library),
               "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["nvfp4_spark_tp_roles"] == [role]
    payload["input_format"] = "fp8_k32"
    path.write_text(json.dumps(payload))
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert "BF16 fabric input" in result.stderr


def test_manifest_writer_rejects_an_unknown_role_before_touching_files(
    tmp_path: Path,
) -> None:
    output = tmp_path / "V41_EXPERT_TP_AOT.json"
    result = subprocess.run(
        [sys.executable, str(MANIFEST), "--role", "expert", "--requested", "tp5",
         "--native-build-dir", str(tmp_path), "--output", str(output)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 2
    assert "expected tp2, tp3 or tp6" in result.stderr
    assert not output.exists()
