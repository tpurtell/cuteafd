from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).parents[2]
PYTHON_ROOTS = (ROOT / "benchmarks", ROOT / "python", ROOT / "scripts")
IGNORED_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
B12X_REQUIREMENT = re.compile(
    r"(?i)(?<![\w.-])b12x(?:\[[^\]\r\n]*\])?\s*"
    r"(?:===|==|~=|!=|<=|>=|<|>|@)"
)
CUTLASS_PACKAGES = (
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
)
QUALIFIED_CUTLASS_VERSION = "4.6.2"
METADATA_FREE_PYTHON_CACHE_MARKERS = (
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "*.pyc",
    "*.pyo",
)


def active_python_sources() -> list[Path]:
    return sorted(
        path
        for root in PYTHON_ROOTS
        for path in root.rglob("*.py")
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def is_retired_sparkinfer_module(module: str | None) -> bool:
    return module == "sparkinfer" or bool(
        module and module.startswith("sparkinfer.")
    )


def test_active_python_does_not_import_retired_sparkinfer_package() -> None:
    violations: list[str] = []
    for path in active_python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module]
            else:
                continue
            for module in modules:
                if is_retired_sparkinfer_module(module):
                    relative = path.relative_to(ROOT)
                    violations.append(f"{relative}:{node.lineno}: {module}")

    assert not violations, (
        "active Python sources must import b12x, not the retired sparkinfer "
        "package:\n" + "\n".join(violations)
    )




def test_standalone_tools_bootstrap_pinned_source_before_b12x_imports() -> None:
    violations: list[str] = []
    explicit_source_tools = {
        "compare_v41_expert_upstream.py",
        "compare_v41_index_topk.py",
        "compare_v41_mhc_upstream.py",
        "compare_v41_narrow_projection.py",
        "compare_v41_upstream_attention.py",
        "compare_v41_vocab_upstream.py",
        "compare_v41_wo_b_fusion.py",
    }
    tools_root = ROOT / "python" / "tools"
    for path in sorted(tools_root.rglob("*.py")):
        if path.name == "_pinned_sparkinfer.py":
            continue
        text = path.read_text(encoding="utf-8")
        if path.name in explicit_source_tools:
            assert "--b12x-root" in text or "PYTHONPATH pointing at" in text
            continue
        tree = ast.parse(text, filename=str(path))
        external_imports = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Import)
                and any(
                    alias.name == "b12x"
                    or alias.name.startswith("b12x.")
                    for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (
                    node.module == "b12x"
                    or bool(node.module and node.module.startswith("b12x."))
                )
            )
        ]
        if not external_imports:
            continue
        bootstrap_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            and any(alias.name == "_pinned_sparkinfer" for alias in node.names)
        ]
        relative = path.relative_to(ROOT)
        if len(bootstrap_imports) != 1:
            violations.append(f"{relative}: expected one _pinned_sparkinfer import")
            continue
        bootstrap_line = bootstrap_imports[0].lineno
        first_external_line = min(node.lineno for node in external_imports)
        if bootstrap_line >= first_external_line:
            violations.append(
                f"{relative}:{bootstrap_line}: bootstrap follows SparkInfer "
                f"import at line {first_external_line}"
            )

    assert not violations, (
        "standalone tools must verify and prepend CUTEAFD's pinned b12x/SparkInfer "
        "tree before importing it:\n" + "\n".join(violations)
    )


def test_build_metadata_does_not_pin_retired_b12x_package() -> None:
    package_files = [
        ROOT / "python" / "pyproject.toml",
        ROOT / "python" / "uv.lock",
        *sorted((ROOT / "docker").glob("Dockerfile*")),
        *sorted(ROOT.glob("requirements*.txt")),
    ]
    violations: list[str] = []
    for path in package_files:
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if B12X_REQUIREMENT.search(line):
                violations.append(
                    f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}"
                )

    assert not violations, (
        "build metadata must not install or pin an external b12x package:\n"
        + "\n".join(violations)
    )


def test_images_validate_the_pinned_b12x_import_namespace() -> None:
    dev = (ROOT / "docker/Dockerfile.dev").read_text()
    assert "COPY third_party/sparkinfer" not in dev
    assert "io.cuteafd.sparkinfer.revision" not in dev
    for relative in ("docker/Dockerfile.release",):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "pathlib, b12x" in text
        assert 'importlib.metadata.version("b12x")' in text
        assert "pathlib, sparkinfer" not in text
        assert 'importlib.metadata.version("sparkinfer")' not in text


def test_metadata_free_release_copies_filter_and_reject_python_caches() -> None:
    for relative in ("build.sh", "scripts/build/build-release-artifacts.sh"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        exclude_lines = [
            line for line in text.splitlines() if "--exclude" in line
        ]
        for marker in METADATA_FREE_PYTHON_CACHE_MARKERS:
            assert any(marker in line for line in exclude_lines), (
                f"{relative} must exclude {marker} from metadata-free "
                "SparkInfer source copies"
            )
        assert text.count("--require-no-python-cache") == 1, (
            f"{relative} must guard its copied SparkInfer source exactly once"
        )

    for relative in (
        "build.sh",
        "wip.sh",
        "scripts/build/build-release-artifacts.sh",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        exclude_lines = [line for line in text.splitlines() if "--exclude" in line]
        assert any(".venv*" in line for line in exclude_lines), (
            f"{relative} must exclude every local named Python environment"
        )

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for marker in METADATA_FREE_PYTHON_CACHE_MARKERS:
        assert marker in dockerignore, (
            f".dockerignore must exclude SparkInfer cache marker {marker}"
        )
    for relative in ("docker/Dockerfile.dev", "docker/Dockerfile.release"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "ENV PYTHONDONTWRITEBYTECODE=1" in text
        if relative.endswith(".dev"):
            continue
        assert "--require-no-python-cache" in text, (
            f"{relative} must reject cached Python artifacts after COPY"
        )


def test_legacy_ds4_aot_bridge_is_gone() -> None:
    # The legacy DS4 Flash/Pro, B12X and W8A16 AOT bridges were removed; the
    # dsv4 programs and V4.1 exports are the only native AOT paths.
    cmake = (ROOT / "native/CMakeLists.txt").read_text(encoding="utf-8")
    for option in (
        "CUTEAFD_ENABLE_DS4_FLASH_AOT",
        "CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT",
        "CUTEAFD_ENABLE_W8A16_AOT",
    ):
        assert option not in cmake, f"{option} must stay removed"


def test_fork_and_images_share_qualified_cutlass_pin() -> None:
    fork_metadata = ROOT / "third_party" / "sparkinfer" / "pyproject.toml"
    assert fork_metadata.is_file(), (
        "initialize the pinned third_party/sparkinfer source before testing "
        "dependency agreement"
    )
    dependencies = tomllib.loads(fork_metadata.read_text(encoding="utf-8"))[
        "project"
    ]["dependencies"]
    versions: dict[str, str] = {}
    for requirement in dependencies:
        for package in CUTLASS_PACKAGES:
            prefix = f"{package}=="
            if requirement.startswith(prefix):
                versions[package] = requirement.removeprefix(prefix)

    assert versions == {
        package: QUALIFIED_CUTLASS_VERSION for package in CUTLASS_PACKAGES
    }, (
        "the SparkInfer fork must pin every CUTLASS DSL package to the "
        f"qualified {QUALIFIED_CUTLASS_VERSION} set, found {versions}"
    )

    image_pin = re.compile(r'nvidia-cutlass-dsl\[cu13\]==([^"]+)"')
    sparkinfer_images = []
    for dockerfile in sorted((ROOT / "docker").glob("Dockerfile*")):
        content = dockerfile.read_text(encoding="utf-8")
        if "third_party/sparkinfer" not in content:
            continue
        sparkinfer_images.append(dockerfile.name)
        match = image_pin.search(content)
        assert match is not None, f"{dockerfile.relative_to(ROOT)} has no CUTLASS DSL pin"
        assert match.group(1) == QUALIFIED_CUTLASS_VERSION, (
            f"{dockerfile.relative_to(ROOT)} pins CUTLASS DSL {match.group(1)}, "
            f"expected {QUALIFIED_CUTLASS_VERSION} to match the fork"
        )
    assert sparkinfer_images == ["Dockerfile.dev", "Dockerfile.release"]


def test_fork_accepts_the_ngc_base_torch_prerelease() -> None:
    fork_metadata = ROOT / "third_party" / "sparkinfer" / "pyproject.toml"
    assert fork_metadata.is_file(), (
        "initialize the pinned third_party/sparkinfer source before testing "
        "the image dependency contract"
    )
    dependencies = tomllib.loads(fork_metadata.read_text(encoding="utf-8"))[
        "project"
    ]["dependencies"]
    torch_requirements = [
        Requirement(requirement)
        for requirement in dependencies
        if Requirement(requirement).name == "torch"
    ]
    assert len(torch_requirements) == 1
    assert Version("2.12.0a0") in torch_requirements[0].specifier, (
        "nvcr.io/nvidia/pytorch:26.05-py3 contains torch 2.12.0a0; "
        f"the fork requirement {torch_requirements[0]} rejects that base and "
        "makes the image's `uv pip check` fail"
    )


def test_standalone_bootstrap_imports_verified_submodule() -> None:
    env = os.environ.copy()
    tools_path = os.fspath(ROOT / "python" / "tools" / "lib")
    env["PYTHONPATH"] = tools_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import _pinned_sparkinfer as pinned; "
                "print(pinned.IMPORTED_MODULE); print(pinned.REVISION); "
                "print(pinned.VERSION)"
            ),
        ],
        check=False,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    imported_module, revision, version = result.stdout.strip().splitlines()
    assert Path(imported_module).resolve().is_relative_to(
        (ROOT / "third_party" / "sparkinfer").resolve()
    )
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert Version(version) == Version("1.3.0")










def test_release_preflight_requires_matching_engine_revisions() -> None:
    # DeepSeek V4.1's release launch (run-family.sh's launch_deepseek_v41).
    release = (ROOT / "scripts/launch/run-family.sh").read_text(encoding="utf-8")
    release = release[release.index("launch_deepseek_v41() {"):]

    assert (
        """engine_commit="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}'"""
        in release
    )
    assert "coordinator image has no engine revision" in release
    # The per-host preflight compares the Spark image's engine-revision label to
    # the coordinator's, and each failure names the host it ran on.
    assert """[[ "$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")" == "$engine" ]]""" in release
    assert 'spark preflight on $host' in release
    assert 'Spark host preflight failed on $host' in release
    fingerprint = release.split('fingerprint="$(', maxsplit=1)[1].split(
        'coordinator="$RELEASE_COORDINATOR_CONTAINER_NAME"', maxsplit=1
    )[0]
    assert '"$engine_commit"' in fingerprint


def test_release_build_overrides_the_base_image_version_label() -> None:
    build = (ROOT / "build.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker" / "Dockerfile.release").read_text(
        encoding="utf-8"
    )

    assert 'ARG CUTEAFD_RELEASE_VERSION=unknown' in dockerfile
    assert 'LABEL org.opencontainers.image.version=${CUTEAFD_RELEASE_VERSION}' in dockerfile
    assert 'spark_release_version="${SPARK_EXPERT_DOCKER_INFERENCE##*:}"' in build
    assert '[[ "$spark_release_version" == "$release_version" ]]' in build
    assert 'release_version="$6"' in build
    assert 'source_manifest_sha256="${8-__legacy__}"' in build
    assert 'spark_tp_roles="${9-__legacy__}"' in build
    assert build.count('--build-arg CUTEAFD_RELEASE_VERSION="$release_version"') == 2
    assert build.count('org.opencontainers.image.version') == 2
    remote_revision_label = next(
        line
        for line in build.splitlines()
        if "org.opencontainers.image.revision" in line
        and "SPARK_EXPERT_DOCKER_INFERENCE" in line
    )
    remote_version_label = next(
        line
        for line in build.splitlines()
        if "org.opencontainers.image.version" in line
        and "SPARK_EXPERT_DOCKER_INFERENCE" in line
    )
    assert remote_version_label == remote_revision_label.replace("revision", "version")




