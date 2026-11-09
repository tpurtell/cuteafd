"""Trace-derived startup lists preserve hot shapes and per-device ownership."""
import importlib.util
import json
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
    assert result["reserve_extra_bytes"] == 2 << 20


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


@pytest.mark.parametrize("change", [lambda t: t.update(schema_version=3),
                                    lambda t: t.update(graphs=[]),
                                    lambda t: t["graphs"][0].update(shape="bad")])
def test_invalid_trace_fails(change):
    data = trace()
    change(data)
    with pytest.raises(ValueError):
        MODULE.generate([data], "glm5")


def modern_trace():
    data = trace()
    data["schema_version"] = 2
    data["cache"]["reserve"] = 16 << 20
    for graph in data["graphs"]:
        graph.update(hits=graph["count"], cycles={"verify": graph["count"]},
                     evictions=0, recaptures=0, ghost_hits=0, first_seen_ms=0,
                     hit_histogram={"rolling": True, "interval_ms": 10_000,
                                    "start_ms": 0, "counts": [5, 0, 0, 0, 0]},
                     permanent_bytes=1000, capture_bytes=0, nodes=10,
                     executables=1, basis={"kind": "nodes", "node_rate": 10, "graph_rate": 900})
    return data


def test_sustained_heat_pins_nonminimum_geometry_and_reports_cost():
    data = modern_trace()
    data["graphs"][1]["hit_histogram"]["counts"] = [1, 1, 1, 1, 0]
    report = []
    result = MODULE.generate([data, data], "glm5", pin_report=report)
    assert [entry["rows"] for entry in result["entries"] if entry["pin"]] == [16]
    assert len(report) == 1
    assert report[0]["permanent_bytes"] is None
    assert report[0]["observed_permanent_lower_bound_bytes"] == 1000
    assert report[0]["capture_bytes"] == 0
    assert report[0]["reserve_fraction"] is None
    assert report[0]["observed_reserve_fraction_lower_bound"] == 1000 / (16 << 20)
    assert report[0]["reasons"] == ["sustained_heat"]


def test_repeated_recapture_survives_frequency_coverage():
    data = modern_trace()
    data["graphs"][2].update(evictions=3, recaptures=2, ghost_hits=4)
    result = MODULE.generate([data], "glm5", coverage=0.99)
    assert {"rows": 64, "table_width": 128, "pin": True} in result["entries"]
    assert not result["entries"][0]["pin"], "early-only frequency is not sustained heat"


def test_partial_and_preidentity_intervals_do_not_prove_sustained_heat():
    graph = modern_trace()["graphs"][0]
    graph["hit_histogram"]["counts"] = [1, 1, 1]
    assert not MODULE.sustained(graph)
    graph["hit_histogram"]["counts"] = [0, 0, 1, 1, 1, 0]
    graph["first_seen_ms"] = 20_000
    assert MODULE.sustained(graph)


def test_rolling_window_not_lifetime_heat():
    graph = modern_trace()["graphs"][0]
    graph["hit_histogram"].update(start_ms=600_000, counts=[1] * 60)
    assert MODULE.sustained(graph)
    graph["hit_histogram"]["counts"] = [0] * 60
    assert not MODULE.sustained(graph)


def test_latest_rolling_snapshot_overrides_old_heat_even_in_reverse_order():
    import copy
    hot = modern_trace()
    hot["graphs"][0]["hit_histogram"]["counts"] = [1] * 5
    cold = copy.deepcopy(hot)
    for graph in cold["graphs"]:
        graph["hit_histogram"].update(start_ms=600_000, counts=[0] * 60)
    for traces in ([hot, cold], [cold, hot]):
        assert not any(entry["pin"] for entry in MODULE.generate(traces, "glm5")["entries"])


def test_pin_cost_includes_startup_only_siblings_of_same_geometry():
    import copy
    data = modern_trace()
    graph = data["graphs"][0]
    graph["hit_histogram"]["counts"] = [1] * 5
    sibling = copy.deepcopy(graph)
    sibling.update(bank=2, cycles={"other": 1}, count=1)
    data["graphs"].append(sibling)
    report = []
    MODULE.generate([data, data], "glm5", pin_report=report)
    assert report[0]["observed_permanent_lower_bound_bytes"] == 2000
    assert report[0]["executables"] == 2


def test_cost_keeps_distinct_key_ids_with_identical_display_shapes():
    import copy
    data = modern_trace()
    graph = data["graphs"][0]
    graph.update(key_id=1)
    graph["hit_histogram"]["counts"] = [1] * 5
    sibling = copy.deepcopy(graph)
    sibling["key_id"] = 2
    data["graphs"].append(sibling)
    report = []
    MODULE.generate([data], "glm5", pin_report=report)
    assert report[0]["observed_permanent_lower_bound_bytes"] == 2000
    assert report[0]["executables"] == 2


def test_new_snapshot_forgets_omitted_identity_heat_and_cost():
    import copy
    old = modern_trace()
    old["cache"]["sampled_at_ms"] = 40_000
    old["graphs"][0]["hit_histogram"]["counts"] = [1, 1, 1, 1, 0]
    new = copy.deepcopy(old)
    new["cache"]["sampled_at_ms"] = 700_000
    new["graphs"] = new["graphs"][1:]
    for graph in new["graphs"]:
        graph["hit_histogram"].update(start_ms=110_000, counts=[0] * 60)
    for inputs in ([old, new], [new, old]):
        report = []
        result = MODULE.generate(inputs, "glm5", pin_report=report)
        assert not any(entry["pin"] for entry in result["entries"])
        assert report == []


def test_partial_geometry_inventory_is_only_a_cost_lower_bound():
    data = modern_trace()
    data["graphs"] = data["graphs"][:1]
    data["graphs"][0]["hit_histogram"]["counts"] = [1] * 5
    report = []
    MODULE.generate([data], "glm5", pin_report=report)
    assert report[0]["observed_permanent_lower_bound_bytes"] == 1000
    assert report[0]["permanent_bytes"] is None
    assert report[0]["reserve_fraction"] is None
    assert report[0]["cost_basis"] == "observed_inventory_lower_bound_unknown_total"


def test_latest_snapshot_row_wins_within_the_same_histogram_interval():
    import copy
    old = modern_trace()
    old["cache"]["sampled_at_ms"] = 40_001
    old["graphs"][0]["hit_histogram"]["counts"] = [1] * 5
    new = copy.deepcopy(old)
    new["cache"]["sampled_at_ms"] = 40_002
    new["graphs"][0]["hit_histogram"]["counts"] = [0] * 5
    for inputs in ([old, new], [new, old]):
        assert not any(entry["pin"] for entry in MODULE.generate(inputs, "glm5")["entries"])


def context_trace(device=0):
    """A two-rank trace: the context placement records each rank's half table."""
    data = trace(device=device)
    for graph in data["graphs"]:
        rows, width, _ = map(int, MODULE.GEOMETRY.search(graph["shape"]).groups())
        graph["shape"] = graph["shape"].replace(
            f"table_width: {width}, table_stride: {width}",
            f"table_width: {width // 2}, table_stride: {width // 2}")
    return data


def test_mistral_large4_selects_its_site_prefix_and_layout():
    one = MODULE.generate([trace()], "mistral_large4", coverage=0.99, layout=1)
    assert one["family"] == "mistral_large4" and one["layout"] == 1
    assert [(entry["rows"], entry["table_width"]) for entry in one["entries"]] == [(1, 16), (16, 32)]
    # Checked-in entries hold the width a step is built with, not the rank's half.
    two = MODULE.generate([context_trace()], "mistral_large4", coverage=0.99, layout=2)
    assert two["layout"] == 2
    assert [(entry["rows"], entry["table_width"]) for entry in two["entries"]] == [(1, 16), (16, 32)]
    assert two["reserve_base_bytes"] == one["reserve_base_bytes"]


def test_layout_is_required_for_mistral_large4_and_reserved_for_it():
    with pytest.raises(ValueError):
        MODULE.generate([trace()], "mistral_large4")
    with pytest.raises(ValueError):
        MODULE.generate([trace()], "glm5", layout=1)


def test_odd_half_table_width_is_refused():
    data = context_trace()
    data["graphs"][0]["shape"] = data["graphs"][0]["shape"].replace(
        "table_width: 8, table_stride: 8", "table_width: 9, table_stride: 9")
    with pytest.raises(ValueError):
        MODULE.generate([data], "mistral_large4", layout=2)


@pytest.mark.parametrize("family", ["glm5", "mistral_large4"])
def test_generator_prefix_matches_shared_engine_graph_site(family):
    import re
    engine = SCRIPT.parents[2] / "rust/crates/cuteafd-daemon/src/families/glm5/engine.rs"
    sites = re.findall(r'name:\s*"([^"]+)"', engine.read_text())
    assert "glm5.segment" in sites
    assert all(site.startswith(MODULE.SITE_PREFIX[family]) for site in sites)


@pytest.mark.parametrize("family", ["glm5_flash", "deepseek_v41"])
def test_native_family_schemas_are_not_advertised_as_glm_consumers(family):
    assert family not in MODULE.SITE_PREFIX
    with pytest.raises(ValueError, match="no checked-in shape-list consumer"):
        MODULE.generate([trace()], family)


@pytest.mark.parametrize("family", ["glm5", "mistral_large4"])
def test_every_consumer_uses_fixed_allocation_tail_not_largest_graph(family):
    data = trace()
    for graph in data["graphs"]:
        graph["site"] = MODULE.SITE_PREFIX[family] + "segment"
    data["sites"] = {MODULE.SITE_PREFIX[family] + "segment": {"largest_bytes": 80 << 20}}
    result = MODULE.generate([data], family, **({"layout": 1} if family == "mistral_large4" else {}))
    assert result["reserve_base_bytes"] == 8 << 20
    assert result["reserve_extra_bytes"] == 2 << 20


@pytest.mark.parametrize("filename", ["glm_prewarm.json", "ml4_prewarm_1.json", "ml4_prewarm_2.json"])
def test_regeneration_reproduces_checked_in_base_and_tail(filename):
    import json
    root = SCRIPT.parents[2] / "rust/crates/cuteafd-loader/src/serving_capacity"
    checked_in = json.loads((root / filename).read_text())
    data = trace(charged=checked_in["reserve_base_bytes"])
    # The measured arena, not a largest capture or the old reservation, is the base.
    data["sites"]["glm5.segment"]["largest_bytes"] = 80 << 20
    result = MODULE.generate([data], checked_in["family"],
                             **({"layout": checked_in["layout"]} if checked_in.get("layout") else {}))
    for key in ("reserve_base_bytes", "reserve_extra_bytes"):
        assert result[key] == checked_in[key]


def kimi_trace(device=0, charged=600 << 20, rows=1, speculative=False, cycles=None):
    shape = (f"lane=0 rank=0 rows={rows} op="
             + (f"attention layer=0 seq=1 spec={int(speculative)}" if speculative
                else "residual site=Norm"))
    return {"schema_version": 2, "cache": {"device": device, "charged": charged},
            "sites": {"kimi_k3.step": {"largest_bytes": 3 << 20}},
            "graphs": [{"device": device, "site": "kimi_k3.step", "bank": 0, "shape": shape,
                        "count": 100, "cycles": cycles if cycles is not None else {"decode": 100},
                        "hits": 100, "evictions": 0, "recaptures": 0, "first_seen_ms": 0,
                        "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                                          "counts": [25, 25, 25, 25, 0]}}]}


def test_kimi_layout_row_keeps_replay_widths_and_skips_prefill():
    trace = kimi_trace()
    trace["graphs"].append(dict(trace["graphs"][0],
                                shape="lane=0 rank=0 rows=4096 op=residual site=Norm",
                                cycles={"prefill": 50}))
    row = MODULE.generate_kimi([trace], 1, "kimi-k3-exl3-k6-min", coverage=0.999)
    assert row["entries"] == [{"rows": 1, "speculative": False, "pin": True}]
    assert row["reserve_base_bytes"] == 600 << 20
    assert row["reserve_extra_bytes"] == 4 << 20


def test_kimi_verify_width_is_its_own_pin(tmp_path):
    trace = kimi_trace(rows=4, speculative=True)
    row = MODULE.generate_kimi([trace], 2, "kimi-k3-exl3-k6-max")
    assert row["entries"] == [{"rows": 4, "speculative": True, "pin": True}]
    out = tmp_path / "kimi_prewarm.json"
    MODULE.merge_kimi_layout(out, MODULE.generate_kimi([kimi_trace()], 1, "kimi-k3-exl3-k6-min"))
    MODULE.merge_kimi_layout(out, row)
    layouts = json.loads(out.read_text())["layouts"]
    assert [layout["ranks"] for layout in layouts] == [1, 2]
    assert layouts[1]["layout"] == "kimi-k3-exl3-k6-max"


def test_kimi_cold_shape_is_listed_but_unpinned():
    trace = kimi_trace()
    trace["graphs"] = [dict(graph, cycles={"decode": 1}, hits=1,
                           hit_histogram={"interval_ms": 10_000, "start_ms": 0,
                                          "counts": [1, 0, 0, 0, 0]})
                       for graph in trace["graphs"]]
    row = MODULE.generate_kimi([trace], 1, "kimi-k3-exl3-k6-min")
    assert row["entries"] == [{"rows": 1, "speculative": False, "pin": False}]


@pytest.mark.parametrize("shape", ["bad", "lane=0 rank=0 rows=0 op=residual site=Norm"])
def test_kimi_invalid_shape_fails(shape):
    trace = kimi_trace()
    trace["graphs"][0]["shape"] = shape
    with pytest.raises(ValueError):
        MODULE.generate_kimi([trace], 1, "kimi-k3-exl3-k6-min")
