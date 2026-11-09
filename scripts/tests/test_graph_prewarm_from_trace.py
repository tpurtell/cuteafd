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
    shape = (f"lane=0 rank={device} rows={rows} op="
             + (f"attention layer=0 seq=1 spec={int(speculative)}" if speculative
                else "residual site=Norm"))
    return {"schema_version": 2, "cache": {"device": device, "charged": charged},
            "sites": {"kimi_k3.step": {"largest_bytes": 3 << 20}},
            "graphs": [{"device": device, "site": "kimi_k3.step", "bank": 0, "shape": shape,
                        "count": 100, "cycles": cycles if cycles is not None else {"decode": 100},
                        "hits": 100, "evictions": 0, "recaptures": 0, "first_seen_ms": 0,
                        "capture_bytes": 3 << 20,
                        "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                                          "counts": [25, 25, 25, 25, 0]}}]}


def test_kimi_layout_row_keeps_replay_widths_and_skips_prefill():
    trace = kimi_trace()
    trace["graphs"].append(dict(trace["graphs"][0],
                                shape="lane=0 rank=0 rows=4096 op=residual site=Norm",
                                cycles={"prefill": 50}))
    row = MODULE.generate_kimi([trace], 1, "kimi-k3-exl3-k6-min", coverage=0.999)
    assert row["entries"] == [{"rows": 1, "speculative": False, "op": "residual", "pin": True, "capture_bytes": [3 << 20]}]
    assert row["reserve_base_bytes"] == [4 << 20]
    assert row["reserve_extra_bytes"] == 2 << 20


def test_kimi_verify_width_is_its_own_pin(tmp_path):
    trace = kimi_trace(rows=4, speculative=True)
    row = MODULE.generate_kimi([trace, kimi_trace(device=1, rows=4, speculative=True)], 2, "kimi-k3-exl3-k6-max")
    assert row["entries"] == [{"rows": 4, "speculative": True, "op": "attention", "sequences": 1, "pin": True, "capture_bytes": [3 << 20, 3 << 20]}]
    out = tmp_path / "kimi_prewarm.json"
    MODULE.merge_kimi_layout(out, MODULE.generate_kimi([kimi_trace()], 1, "kimi-k3-exl3-k6-min"))
    MODULE.merge_kimi_layout(out, row)
    layouts = json.loads(out.read_text())["layouts"]
    assert [layout["ranks"] for layout in layouts] == [1, 2]
    assert layouts[1]["layout"] == "kimi-k3-exl3-k6-max"


def test_kimi_startup_only_geometry_is_neither_selected_nor_pinned():
    trace = kimi_trace()
    trace["graphs"].append(dict(trace["graphs"][0],
        shape="lane=0 rank=0 rows=16 op=attention layer=0 seq=16 spec=0",
        cycles={"other": 1000}, count=1000, capture_bytes=None,
        evictions=3, recaptures=3))
    row = MODULE.generate_kimi([trace], 1, "min")
    assert row["entries"] == [
        {"rows": 1, "speculative": False, "op": "residual", "pin": True, "capture_bytes": [3 << 20]}]
    assert row["reserve_base_bytes"] == [4 << 20]


def test_kimi_cold_shape_outside_lifetime_tier_is_listed_but_unpinned():
    trace = kimi_trace()
    trace["graphs"].append(dict(trace["graphs"][0],
        shape="lane=0 rank=0 rows=16 op=residual site=Norm", cycles={"decode": 1}, hits=1,
        hit_histogram={"interval_ms": 10_000, "start_ms": 0,
                       "counts": [1, 0, 0, 0, 0]}))
    row = MODULE.generate_kimi([trace], 1, "kimi-k3-exl3-k6-min", coverage=1)
    assert row["entries"] == [
        {"rows": 1, "speculative": False, "op": "residual", "pin": True, "capture_bytes": [3 << 20]},
        {"rows": 16, "speculative": False, "op": "residual", "pin": False, "capture_bytes": [3 << 20]}]


@pytest.mark.parametrize("shape", ["bad", "lane=0 rank=0 rows=0 op=residual site=Norm"])
def test_kimi_invalid_shape_fails(shape):
    trace = kimi_trace()
    trace["graphs"][0]["shape"] = shape
    with pytest.raises(ValueError):
        MODULE.generate_kimi([trace], 1, "kimi-k3-exl3-k6-min")


def test_kimi_unreachable_rows_are_not_proposed_even_with_decode_counts():
    trace = kimi_trace()
    trace["graphs"].append(dict(trace["graphs"][0],
                                shape="lane=0 rank=0 rows=4096 op=ffn layer=0"))
    assert MODULE.generate_kimi([trace], 1, "min")["entries"] == [
        {"rows": 1, "speculative": False, "op": "residual", "pin": True, "capture_bytes": [3 << 20]}]


def test_kimi_geometry_preserves_operation_and_attention_sequence_count():
    trace = kimi_trace(rows=20, speculative=True)
    trace["graphs"][0]["shape"] = "lane=0 rank=0 rows=20 op=attention layer=0 seq=4 spec=1"
    trace["graphs"].append(dict(trace["graphs"][0],
        shape="lane=0 rank=0 rows=20 op=context layer=1 spec=1"))
    row = MODULE.generate_kimi([trace], 1, "max")
    assert row["entries"] == [
        {"rows": 20, "speculative": True, "op": "attention", "sequences": 4, "pin": True, "capture_bytes": [3 << 20]},
        {"rows": 20, "speculative": True, "op": "context", "pin": True, "capture_bytes": [3 << 20]}]


def test_kimi_base_sums_selected_latest_identity_shares_per_rank():
    import copy
    first = kimi_trace()
    first["cache"]["sampled_at_ms"] = 40_000
    graph = first["graphs"][0]
    graph.update(key_id=1, capture_bytes=1)
    sibling = dict(graph, bank=2, cycles={"other": 1}, count=1, capture_bytes=2)
    first["graphs"] += [dict(graph), sibling,
                        dict(graph, key_id=3, shape="lane=0 rank=0 rows=64 op=ffn layer=0",
                             cycles={"other": 1}, capture_bytes=900 << 20)]
    old = copy.deepcopy(first)
    old["cache"]["sampled_at_ms"] = 39_000
    old["graphs"][0]["capture_bytes"] = 100 << 20
    second = kimi_trace(device=1)
    second["graphs"][0]["capture_bytes"] = (2 << 20) + 1
    for inputs in ([old, first, second], [second, first, old]):
        row = MODULE.generate_kimi(inputs, 2, "max")
        assert row["reserve_base_bytes"] == [2 << 20, 4 << 20]
        assert row["entries"][0]["capture_bytes"] == [3, (2 << 20) + 1]


def test_kimi_bases_follow_logical_rank_not_cuda_ordinal():
    rank0 = kimi_trace(device=7)
    rank0["graphs"][0].update(shape="lane=0 rank=0 rows=1 op=residual site=Norm",
                               capture_bytes=1)
    rank1 = kimi_trace(device=3)
    rank1["graphs"][0].update(shape="lane=0 rank=1 rows=1 op=residual site=Norm",
                               capture_bytes=(2 << 20) + 1)
    row = MODULE.generate_kimi([rank1, rank0], 2, "max")
    assert row["reserve_base_bytes"] == [2 << 20, 4 << 20]
    assert row["entries"][0]["capture_bytes"] == [1, (2 << 20) + 1]
    rank1["graphs"][0]["shape"] = rank0["graphs"][0]["shape"]
    with pytest.raises(ValueError, match="one-to-one"):
        MODULE.generate_kimi([rank0, rank1], 2, "max")


@pytest.mark.parametrize("share", [None, -1, "0", True])
def test_kimi_missing_or_invalid_selected_capture_share_fails(share):
    data = kimi_trace()
    data["graphs"][0]["capture_bytes"] = share
    with pytest.raises(ValueError, match="capture_bytes"):
        MODULE.generate_kimi([data], 1, "min")
    del data["graphs"][0]["capture_bytes"]
    with pytest.raises(ValueError, match="capture_bytes"):
        MODULE.generate_kimi([data], 1, "min")


def test_kimi_lifetime_hot_tier_survives_aged_out_rolling_window():
    data = kimi_trace()
    data["cache"]["sampled_at_ms"] = 900_000
    seed = data["graphs"][0]
    data["graphs"] = [dict(seed, key_id=index,
        shape=f"lane=0 rank=0 rows={rows} op=residual site=Norm",
        hits=hits, count=hits, cycles={"decode": hits},
        hit_histogram={"interval_ms": 10_000, "start_ms": 310_000, "counts": [0] * 60})
        for index, (rows, hits) in enumerate([(1, 1), (4, 60), (8, 25), (16, 14)])]
    report = []
    row = MODULE.generate_kimi([data], 1, "min", coverage=1, pin_report=report)
    assert [entry["rows"] for entry in row["entries"] if entry["pin"]] == [1, 4]
    assert not any(entry["rolling_pin"] for entry in report)
    assert {entry["rows"] for entry in report if entry["lifetime_pin"]} == {1, 4}
    assert all(entry["whole_run_interval_spread"] is None for entry in report)


def test_kimi_lifetime_tier_uses_per_key_counts_and_keeps_cutoff_ties():
    data = kimi_trace()
    seed = data["graphs"][0]
    data["graphs"] = [dict(seed, key_id=index,
        shape=f"lane=0 rank=0 rows={rows} op=ffn layer={index}",
        hits=hits, cycles={"verify": hits},
        hit_histogram={"interval_ms": 10_000, "start_ms": 0, "counts": [0] * 5})
        for index, (rows, hits) in enumerate([(4, 40), (8, 30), (16, 30), (32, 1)])]
    row = MODULE.generate_kimi([data], 1, "min", coverage=1)
    assert [entry["rows"] for entry in row["entries"] if entry["pin"]] == [4, 8, 16]


def test_kimi_dominant_verify_pins_exact_attention_and_flagless_siblings():
    data = kimi_trace()
    seed = data["graphs"][0]
    shapes = [
        ("rows=20 op=attention layer=0 seq=4 spec=1", 100),
        ("rows=20 op=attention layer=0 seq=5 spec=1", 10),
        ("rows=20 op=context layer=1 spec=1", 100),
        ("rows=20 op=ffn layer=1", 1),
        ("rows=20 op=residual site=Norm", 1),
        ("rows=1 op=residual site=Norm", 1),
        ("rows=4 op=ffn layer=2", 100),
        ("rows=4 op=residual site=Norm", 100),
    ]
    data["graphs"] = [dict(seed, key_id=index, shape="lane=0 rank=0 " + shape,
        hits=count, cycles={"decode" if "rows=1 " in shape else "verify": count},
        hit_histogram={"interval_ms": 10_000, "start_ms": 0, "counts": [0] * 5})
        for index, (shape, count) in enumerate(shapes)]
    report = []
    MODULE.generate_kimi([data], 1, "min", coverage=1, pin_report=report)
    dominant = [entry for entry in report if "dominant_verify" in entry["reasons"]]
    assert {(entry["op"], entry["sequences"]) for entry in dominant} == {
        ("attention", 4), ("context", None), ("ffn", None), ("residual", None)}
    assert all(entry["rows"] == 20 for entry in dominant)
    assert any(entry["rows"] == 1 and "plain_c1" in entry["reasons"] for entry in report)


def test_kimi_lifetime_tiers_are_independent_per_device_and_operation():
    traces = [kimi_trace(device=rank) for rank in (0, 1)]
    for rank, data in enumerate(traces):
        seed = data["graphs"][0]
        data["graphs"] = [dict(seed, key_id=index,
            shape=f"lane=0 rank={rank} rows={rows} op={op} " +
                  ("site=Norm" if op == "residual" else "layer=0"),
            hits=hits, cycles={"verify": hits},
            hit_histogram={"interval_ms": 10_000, "start_ms": 0, "counts": [0] * 5})
            for index, (rows, op, hits) in enumerate([
                (4, "ffn", 100 if rank == 0 else 1),
                (8, "ffn", 1 if rank == 0 else 100),
                (16, "residual", 2)])]
    row = MODULE.generate_kimi(traces, 2, "max", coverage=1)
    assert {(entry["rows"], entry["op"]) for entry in row["entries"] if entry["pin"]} == {
        (4, "ffn"), (8, "ffn"), (16, "residual")}


def test_kimi_zero_replay_geometry_stays_lazy_even_with_hot_request_counts():
    data = kimi_trace()
    seed = data["graphs"][0]
    data["graphs"].append(dict(seed, key_id=2,
        shape="lane=0 rank=0 rows=4 op=attention layer=0 seq=1 spec=1",
        hits=0, cycles={"verify": 10000}, capture_bytes=None,
        evictions=10, recaptures=10))
    row = MODULE.generate_kimi([data], 1, "min", coverage=1)
    assert len(row["entries"]) == 1
    assert row["entries"][0]["rows"] == 1
    data["graphs"][0]["hits"] = 0
    with pytest.raises(ValueError, match="no replayed"):
        MODULE.generate_kimi([data], 1, "min")


def test_kimi_one_replay_selects_whole_geometry_and_charges_zero_hit_sibling():
    data = kimi_trace()
    graph = data["graphs"][0]
    graph.update(hits=1, cycles={"decode": 2}, capture_bytes=3)
    data["graphs"].append(dict(graph, bank=2, hits=0, capture_bytes=7))
    report = []
    row = MODULE.generate_kimi([data], 1, "min", pin_report=report)
    assert row["entries"][0]["capture_bytes"] == [10]
    assert report[0]["zero_hit_sibling_identities"] == 1
    assert report[0]["zero_hit_sibling_bytes"] == 7


def test_kimi_old_replayed_geometry_cannot_survive_latest_zero_replay_inventory():
    import copy
    old = kimi_trace(rows=4, speculative=True)
    old["cache"]["sampled_at_ms"] = 100
    new = copy.deepcopy(old)
    new["cache"]["sampled_at_ms"] = 200
    new["graphs"][0].update(hits=0, cycles={"verify": 1}, capture_bytes=None)
    plain = kimi_trace()["graphs"][0]
    new["graphs"].append(plain)
    for traces in ([old, new], [new, old]):
        row = MODULE.generate_kimi(traces, 1, "min")
        assert [entry["rows"] for entry in row["entries"]] == [1]


def test_kimi_entry_costs_sum_to_each_once_rounded_base():
    traces = [kimi_trace(device=rank) for rank in (0, 1)]
    for rank, data in enumerate(traces):
        seed = data["graphs"][0]
        data["graphs"] = [dict(seed, key_id=index,
            shape=f"lane=0 rank={rank} rows={rows} op=ffn layer=0",
            hits=hits, cycles={"decode": hits}, capture_bytes=cost)
            for index, (rows, hits, cost) in enumerate([(1, 100, 0), (4, 10, 13 + rank)])]
    row = MODULE.generate_kimi(traces, 2, "max", coverage=1)
    assert [entry["capture_bytes"] for entry in row["entries"]] == [[0, 0], [13, 14]]
    assert row["reserve_base_bytes"] == [MODULE.packed(sum(entry["capture_bytes"][rank]
        for entry in row["entries"])) for rank in (0, 1)]
    traces[1]["graphs"] = traces[1]["graphs"][:1]
    row = MODULE.generate_kimi(traces, 2, "max", coverage=1)
    assert row["entries"][1]["capture_bytes"] == [13, None]


def test_kimi_seeded_grouped_trace_reproduces_checked_in_layouts():
    fixture = json.loads((SCRIPT.parents[1] / "tests/fixtures/kimi-prewarm-seeded.json").read_text())
    checked = json.loads((SCRIPT.parents[2] /
        "rust/crates/cuteafd-loader/src/serving_capacity/kimi_prewarm.json").read_text())
    for layout in fixture["layouts"]:
        traces = []
        for rank, device in enumerate(layout["devices"]):
            data = kimi_trace(device=device["device"])
            seed = data["graphs"][0]
            data["graphs"] = []
            for index, group in enumerate(device["groups"]):
                rows, spec, op, seq, hits, decode, verify, other, identities, cost = group
                shape = f"lane=0 rank={rank} rows={rows} op={op}"
                if op == "attention":
                    shape += f" layer=0 seq={seq} spec={int(spec)}"
                elif op == "context":
                    shape += f" layer=0 spec={int(spec)}"
                else:
                    shape += " site=Norm" if op == "residual" else " layer=0"
                share, tail = divmod(cost, identities)
                for sibling in range(identities):
                    data["graphs"].append(dict(seed, key_id=f"{index}-{sibling}", shape=shape,
                        hits=hits, cycles={"decode": decode, "verify": verify, "other": other},
                        capture_bytes=share + (sibling < tail),
                        hit_histogram={"interval_ms": 10000, "start_ms": 0, "counts": [0] * 60}))
            traces.append(data)
        result = MODULE.generate_kimi(traces, layout["ranks"], layout["layout"])
        expected = next(row for row in checked["layouts"] if row["ranks"] == layout["ranks"])
        assert result == expected
        for rank in range(layout["ranks"]):
            assert MODULE.packed(sum(entry["capture_bytes"][rank] or 0
                for entry in result["entries"])) == result["reserve_base_bytes"][rank]
        assert result["reserve_extra_bytes"] == MODULE.CHUNK


def test_kimi_zero_share_is_valid_and_missing_rank_fails():
    data = kimi_trace()
    data["graphs"][0]["capture_bytes"] = 0
    row = MODULE.generate_kimi([data], 1, "min")
    assert row["reserve_base_bytes"] == [0]
    assert row["entries"][0]["capture_bytes"] == [0]
    with pytest.raises(ValueError, match="missing a physical rank"):
        MODULE.generate_kimi([data], 2, "max")
