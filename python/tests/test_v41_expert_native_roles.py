"""CPU-only contract checks for the replicated-group native expert wrapper.

Covers the pure symbol/geometry selection added for Spark TP2/TP3 (roles 5/6)
without loading a library or touching a GPU: the ABI structs, the family prefix
mapping, and mutual exclusion between families.
"""
from __future__ import annotations

import ctypes as C
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python" / "tools" / "lib" / "_v41_expert_native.py"


def _load():
    torch = pytest.importorskip("torch")
    assert torch is not None
    spec = importlib.util.spec_from_file_location("_v41_expert_native_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_abi_struct_layout_is_unchanged() -> None:
    module = _load()
    assert C.sizeof(module.Info) == 64
    assert C.sizeof(module.Launch) == 392
    assert module.Launch.stream.offset == 384


def test_symbol_prefix_selects_new_spark_families() -> None:
    module = _load()
    assert module.expert_symbol_prefix() == "cuteafd_v41_expert_"
    assert module.expert_symbol_prefix(local=True) == "cuteafd_v41_local_expert_"
    assert module.expert_symbol_prefix(tp2=True) == "cuteafd_v41_tp2_expert_"
    for family in ("v41", "dsv4f", "dsv4p"):
        for artifacts in module.native_spark_artifacts(family).values():
            for artifact in artifacts:
                assert module.expert_symbol_prefix(artifact=artifact, family=family) == artifact.prefix(family)


def test_only_launch_abi_symbols_are_namespaced() -> None:
    module = _load()
    prefix = "cuteafd_v41_spark_tp2_expert_"
    assert module.namespaced_symbol("cuteafd_v41_expert_launch", prefix) == \
        "cuteafd_v41_spark_tp2_expert_launch"
    assert module.namespaced_symbol("cuteafd_v41_expert_info", prefix) == \
        "cuteafd_v41_spark_tp2_expert_info"
    # The packer size query and the packer are canonical for every family.
    assert module.namespaced_symbol("cuteafd_expert_packed_sizes", prefix) == \
        "cuteafd_expert_packed_sizes"
    assert module.namespaced_symbol("cuteafd_pack_expert_async", prefix) == \
        "cuteafd_pack_expert_async"
    # The canonical prefix is an identity mapping.
    assert module.namespaced_symbol("cuteafd_v41_expert_launch",
                                    "cuteafd_v41_expert_") == "cuteafd_v41_expert_launch"


def test_symbol_families_are_mutually_exclusive() -> None:
    module = _load()
    artifact = module.native_spark_artifacts()[2][0]
    for kwargs in ({"local": True, "tp2": True}, {"artifact": artifact, "tp2": True},
                   {"artifact": artifact, "local": True}):
        with pytest.raises(AssertionError):
            module.expert_symbol_prefix(**kwargs)
    from spark_packages import NativeArtifact
    with pytest.raises(AssertionError):
        module.expert_symbol_prefix(artifact=NativeArtifact(9, 384, 384, "spark_tp6"))


@pytest.mark.parametrize("family", ("dsv4f", "dsv4p"))
def test_native_v4_family_symbols_keep_shared_packer(family) -> None:
    module = _load()
    prefix = module.expert_symbol_prefix(family=family)
    assert prefix == f"cuteafd_{family}_expert_"
    assert module.namespaced_symbol("cuteafd_v41_expert_launch", prefix) == f"{prefix}launch"
    assert module.namespaced_symbol("cuteafd_pack_expert_async", prefix) == "cuteafd_pack_expert_async"
    assert module.namespaced_symbol("cuteafd_set_expert_hidden", prefix) == "cuteafd_set_expert_hidden"
    assert module.expert_symbol_prefix(family=family, tp2=True) == f"cuteafd_{family}_tp2_expert_"
    for artifacts in module.native_spark_artifacts(family).values():
        for artifact in artifacts:
            assert module.expert_symbol_prefix(family=family, artifact=artifact) == artifact.prefix(family)
