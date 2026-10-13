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


try:
    _verifier = _load_verifier()
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


# Portable host code for every AOT export (#127). CuTe DSL compiles each
# exported object's host loader for the *build host's* CPU unless told
# otherwise, so a Zen 5 build host emitted AVX-512 that a coordinator without
# it cannot run, and two build hosts never produced identical objects. Every
# exporter imports this module before compiling, so the default HostTarget is
# pinned here once. Coordinator stubs (launch wrappers, not hot code) use
# x86-64-v3 (AVX2). Every Spark is a GB10 (Cortex-X925 + Cortex-A725, one
# feature set), so Spark exports are tuned for its fast core: CuTe DSL 4.6.2's
# LLVM accepts cortex-x925, and on a GB10 "native" already resolved to it
# (identical objects), so the pin keeps today's code and makes it reproducible.
# An explicit per-call --host-target still wins; CUTEAFD_CUTE_HOST_TARGET
# overrides the default ("native" keeps the build host).
HOST_TARGETS = {
    "x86_64": "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v3",
    "aarch64": "llvm -mtriple=aarch64-unknown-linux-gnu -mcpu=cortex-x925",
}


def host_target() -> str:
    """The CuTe DSL host target AOT exports use on this machine ("" = native)."""
    import platform

    value = os.environ.get("CUTEAFD_CUTE_HOST_TARGET")
    if value is not None:
        return "" if value == "native" else value
    machine = platform.machine()
    if machine not in HOST_TARGETS:
        raise RuntimeError(f"no portable CuTe DSL host target for {machine}; set CUTEAFD_CUTE_HOST_TARGET")
    return HOST_TARGETS[machine]


def _pin_cute_host_target() -> None:
    target = host_target()
    if not target:
        return
    try:
        from cutlass.base_dsl import compiler
    except ModuleNotFoundError:
        return  # no CuTe DSL here (host tools, verifiers): nothing can compile

    cls = compiler.HostTarget
    if getattr(cls, "_cuteafd_host_target", None) == target:
        return
    # Read-side pin: option objects are built in several places, some before
    # this module runs (the DSL singleton's CompileOptions at `import cutlass`),
    # and the string parser assigns "" for an absent --host-target. The object
    # export reads only .triple/.cpu/.features, so an empty spec resolves to the
    # pinned target there; a non-empty explicit spec is used as given.
    pinned = compiler._parse_host_target(target)

    def resolved(self) -> tuple[str, str, str]:
        return (self._triple, self._cpu, self._features) if self._triple else pinned

    cls.triple = property(lambda self: resolved(self)[0])
    cls.cpu = property(lambda self: resolved(self)[1])
    cls.features = property(lambda self: resolved(self)[2])
    cls._cuteafd_host_target = target


_pin_cute_host_target()
HOST_TARGET = host_target()


def export_target(properties) -> str:
    """What an AOT manifest records as its device: the compiled target
    ("sm120"), never the export host's GPU marketing name, which made two hosts'
    identical exports differ (RTX PRO 6000 vs its Max-Q edition, #127)."""
    return f"sm{properties.major}{properties.minor}"
