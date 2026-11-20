"""Make standalone tools use CUTEAFD's verified SparkInfer source tree."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tomllib
from types import ModuleType


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(
    os.environ.get(
        "CUTEAFD_SPARKINFER_SOURCE_DIR",
        ROOT / "third_party" / "sparkinfer",
    )
).expanduser().resolve()
LOCK = Path(
    os.environ.get(
        "CUTEAFD_SPARKINFER_LOCK_FILE",
        ROOT / "third_party" / "sparkinfer.lock.json",
    )
).expanduser().resolve()
VERIFIER_PATH = ROOT / "scripts" / "build" / "verify-sparkinfer-source.py"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_cuteafd_verify_sparkinfer_source",
        VERIFIER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SparkInfer source verifier: {VERIFIER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepend_source() -> None:
    source = os.fspath(SOURCE)
    retained_paths: list[str] = []
    for entry in sys.path:
        if entry and Path(entry).resolve() == SOURCE:
            continue
        retained_paths.append(entry)
    sys.path[:] = [source, *retained_paths]


def _export_cache_paused():
    # The export cache records what an export reads (python/tools/lib/_export_cache.py).
    # Verification hashes the whole tree against the lock: that proves the tree
    # is the locked one, it does not consume it, so only the lock is an input.
    recorder = sys.modules.get("_export_cache")
    if recorder is None:
        from contextlib import nullcontext
        return nullcontext()
    recorder.note_read(LOCK)
    return recorder.paused()


try:
    _verifier = _load_verifier()
    with _export_cache_paused():
        LOCK_DATA = _verifier.verify(SOURCE, LOCK)
    _prepend_source()
    IMPORTED_MODULE = _verifier.verify_import_source(SOURCE)
except Exception as exc:
    raise RuntimeError(
        "standalone SparkInfer tools require the verified "
        f"{SOURCE} tree locked by {LOCK}: {exc}"
    ) from exc

REVISION = str(LOCK_DATA["revision"])
VERSION = str(
    tomllib.loads((SOURCE / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
)


# Shared with standalone SparkInfer AOT programs. This bootstrap also covers
# engine-owned exporters that call CuTe's raw export_to_c API.
from b12x._lib.aot_host import pin_aot_host_target

try:
    pin_aot_host_target()
except ModuleNotFoundError as exc:
    if exc.name != "cutlass":
        raise


def _install_backend_cache():
    # One implementation and identity for CMake, offline republishes and spawned
    # variant workers. Runtime JIT caches cannot retain export_to_c's compiler IR.
    name = "_cuteafd_kernel_records"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/build/kernel-records.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    sys.modules[name].install_backend_cache()


_install_backend_cache()


def export_target(properties) -> str:
    """What an AOT manifest records as its device: the compiled target
    ("sm120"), never the export host's GPU marketing name, which made two hosts'
    identical exports differ (RTX PRO 6000 vs its Max-Q edition, #127)."""
    return f"sm{properties.major}{properties.minor}"


def supports_routed_mul1(source: Path = SOURCE) -> bool:
    """True when the verified SparkInfer tree exports routed-expert MUL1 kernels."""
    codebooks = source / "b12x" / "moe" / "_shared" / "trellis_codebooks.py"
    mixed = source / "b12x" / "moe" / "_shared" / "kernels" / "w4a16" / "mixed_trellis.py"
    if not codebooks.is_file() or not mixed.is_file():
        return False
    with _export_cache_paused():
        spec = importlib.util.spec_from_file_location("_cuteafd_trellis_codebooks_probe", codebooks)
        if spec is None or spec.loader is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        has_mul1 = getattr(module, "MUL1", None) == "mul1" and "mul1" in getattr(module, "CODEBOOKS", ())
        return bool(has_mul1 and "MUL1" in mixed.read_text(encoding="utf-8"))


def require_routed_mul1(source: Path = SOURCE, revision: str = REVISION) -> None:
    if not supports_routed_mul1(source):
        raise ValueError(
            f"EXL3 routed-expert MUL1 requires a newer SparkInfer pin (current revision {revision})"
        )


# Every exporter imports this module before first CUDA use. Admission holds a
# host RAM/context reservation until exit, including sequential FP8 packages.
from _export_jobs import install_context_admission
install_context_admission()
