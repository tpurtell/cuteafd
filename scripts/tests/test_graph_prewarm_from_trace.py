"""Trace-derived startup lists preserve hot shapes and per-device ownership."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "dev" / "graph-prewarm-from-trace.py"
SPEC = importlib.util.spec_from_file_location("graph_prewarm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trace(device=0, charged=8 << 20):
    def graph(rows, width, count):
        return {"device": device, "site": "glm5.segment", "bank": 1,
                "shape": f"GlmGraphKey {{ layer: 0, rows: {rows}, table_width: {width}, table_stride: {width}, previous: First, head: false }}",
                "count": count}
    return {"schema_version": 1, "cache": {"device": device, "charged": charged},
            "sites": {"glm5.segment": {"largest_bytes": 3 << 20}},
            "graphs": [graph(1, 16, 100), graph(16, 32, 99), graph(64, 128, 1)]}


def test_frequency_and_low_parameter_pin_candidate():
    result = MODULE.generate([trace()], "glm5", coverage=0.99)
    assert result["entries"] == [{"rows": 1, "table_width": 16, "pin": True},
                                 {"rows": 16, "table_width": 32, "pin": False}]
    assert result["reserve_base_bytes"] == 8 << 20
    assert result["reserve_extra_bytes"] == 4 << 20


def test_devices_are_not_added_and_repeated_snapshots_do_not_duplicate_shapes():
    result = MODULE.generate([trace(), trace(), trace(1, 12 << 20)], "glm5")
    assert len(result["entries"]) == 3
    assert result["reserve_base_bytes"] == 12 << 20


def test_startup_prewarm_counts_do_not_select_cold_shapes():
    data = trace()
    for graph in data["graphs"]:
        graph["cycles"] = {"other": 100, "verify": graph["count"]}
    data["graphs"][2]["cycles"] = {"other": 1000}
    result = MODULE.generate([data], "glm5")
    assert len(result["entries"]) == 2
    assert all(entry["rows"] != 64 for entry in result["entries"])


def test_cold_low_rank_shape_is_not_pinned():
    data = trace()
    data["graphs"][0]["count"] = 1
    result = MODULE.generate([data], "glm5")
    assert not any(entry["pin"] for entry in result["entries"])


@pytest.mark.parametrize("change", [lambda t: t.update(schema_version=2),
                                    lambda t: t.update(graphs=[]),
                                    lambda t: t["graphs"][0].update(shape="bad")])
def test_invalid_trace_fails(change):
    data = trace()
    change(data)
    with pytest.raises(ValueError):
        MODULE.generate([data], "glm5")


def v41_trace(counts=(32, 32, 1), prefill=False):
    graphs = [{"device": 0, "site": "deepseek_v41.lane_row", "bank": 1,
               "shape": f"(7, {rows})", "count": count} for rows, count in zip((1, 16, 128), counts)]
    if prefill:
        graphs.append({"device": 0, "site": "deepseek_v41.lane_prefill", "bank": 1,
                       "shape": "(7, 2752)", "count": 3})
    return {"schema_version": 1, "cache": {"device": 0, "charged": 5 << 20},
            "sites": {"deepseek_v41.lane_row": {"largest_bytes": 2 << 20}}, "graphs": graphs}


def test_v41_rows_pin_the_hot_decode_width_and_keep_replay():
    result = MODULE.generate([v41_trace()], "deepseek_v41", coverage=0.99)
    assert result["family"] == "deepseek_v41"
    # 1 and 16 are the hot widths; 128 is retained for the decoder replay.
    assert [entry["rows"] for entry in result["entries"]] == [1, 16, 128]
    assert [entry["pin"] for entry in result["entries"]] == [True, False, False]
    assert "table_width" not in result["entries"][0]
    assert result["reserve_base_bytes"] == 6 << 20
    assert result["reserve_extra_bytes"] == 2 << 20


def test_v41_prefill_chunk_is_admitted_only_when_declared():
    with pytest.raises(ValueError):
        MODULE.generate([v41_trace(prefill=True)], "deepseek_v41")
    result = MODULE.generate([v41_trace(prefill=True)], "deepseek_v41", prefill_rows=2752)
    assert any(entry["rows"] == 2752 for entry in result["entries"])


def test_v41_selection_vector_key_needs_a_site_rule():
    data = v41_trace()
    data["graphs"].append({"device": 0, "site": "deepseek_v41.sparse", "bank": 1,
                           "shape": "(7, [1, 2, 3])", "count": 9})
    with pytest.raises(ValueError):
        MODULE.generate([data], "deepseek_v41")
