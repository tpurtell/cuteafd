"""CPU-only grouping contracts use the actual pinned route/compute authority."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("expert_grouping", ROOT / "python/tools/lib/expert_grouping.py")
grouping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grouping)


def test_slice_group_rows_come_from_pinned_planner_and_compute():
    assert grouping.slice_group_rows(ROOT / "third_party/sparkinfer") == 16


@pytest.mark.parametrize("capacity", [1, 80])
def test_group_macro_covers_each_compiled_capacity(capacity):
    macro = grouping.native_group_macros([{"capacity_rows": capacity, "group_rows": 16}])[0]
    assert f"((capacity)=={capacity}) ? 16u" in macro
    assert macro.endswith("0u)")


def test_hybrid_macro_keeps_compiled_tile_and_compact_live_cutoff():
    macro = grouping.native_group_macros([{"capacity_rows": 80, "group_rows": 16,
        "compact_group_rows": 1, "compact_live_rows": 8}])[0]
    assert "((rows)>0 && (rows)<=8 ? 1u : 16u)" in macro


@pytest.mark.parametrize("record", [
    {"capacity_rows": 1}, {"capacity_rows": 1, "group_rows": 0},
    {"capacity_rows": 80, "group_rows": 16, "compact_live_rows": 8},
    {"capacity_rows": 80, "group_rows": 16, "compact_live_rows": 81, "compact_group_rows": 1},
])
def test_missing_or_unknown_grouping_is_rejected(record):
    with pytest.raises((KeyError, ValueError)):
        grouping.native_group_macros([record])


def test_every_native_wrapper_namespaces_the_additive_query():
    for wrapper in (ROOT / "native/families/deepseek_v41/src").glob("*experts.cc"):
        text = wrapper.read_text()
        if "#define cuteafd_expert_output_kind" in text:
            assert "#define cuteafd_expert_group_rows" in text, wrapper
    cmake = (ROOT / "native/cmake/shared/expert_families.cmake").read_text()
    assert "set(expert_ops info initialize output_kind group_rows" in cmake
