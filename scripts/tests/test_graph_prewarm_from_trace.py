"""Trace-derived startup lists preserve hot shapes and per-device ownership."""
import copy
import importlib.util
import hashlib
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
                "count": count, "hits": count}
    return {"schema_version": 1, "cache": {"device": device, "charged": charged},
            "sites": {"glm5.segment": {"largest_bytes": 3 << 20}},
            "graphs": [graph(1, 16, 100), graph(16, 32, 99), graph(64, 128, 1)]}


@pytest.mark.parametrize("family", ["deepseek_v4", "deepseek_v41", "qwen4", "mimo_v2", "kimi_k3"])
def test_non_glm_generators_keep_existing_fixture_outputs(family):
    expected = json.loads((Path(__file__).parent / "fixtures/non-glm-prewarm-expected.json").read_text())
    if family == "deepseek_v4":
        result = MODULE.generate([v4_trace(0, 5 << 20)], family, layout="rtx1-local", existing=v4_list())
    elif family == "deepseek_v41":
        result = MODULE.generate([v41_trace()], family, layout=1)
    elif family == "qwen4":
        result = MODULE.generate_qwen(fixture_traces(), "single_0spark", coverage=0.99)
    elif family == "mimo_v2":
        result = MODULE.generate([mimo_trace()], family, coverage=0.99, ranks=1, variant="mimof-48", sparks=True)
    else:
        result = MODULE.generate_kimi([kimi_trace()], 1, "kimi-k3-exl3-k6-min", core="k6")
    assert result == expected[family]


def test_step5_real_hqa_traces_reproduce_installed_inventories_byte_for_byte():
    import gzip

    fixture = SCRIPT.parents[1] / "tests/fixtures/step5-prewarm-trace.json.gz"
    evidence = json.loads(gzip.decompress(fixture.read_bytes()))
    assert evidence["checkpoint"] == "wrldsuksgo2mars/Step-5-Preview-EXL3-K4-Core-K6-HQA"
    assert evidence["revision"] == "61f9275e7f604cb11a5ae9a64a07ff1c600174b7"
    generated = {"layouts": [MODULE.generate_step5(layout["traces"], layout["layout"])
                             for layout in evidence["layouts"]]}
    installed = SCRIPT.parents[2] / "rust/crates/cuteafd-loader/src/serving_capacity/step5_prewarm.json"
    assert json.dumps(generated, indent=2) + "\n" == installed.read_text()
    expected = [(58, [3_407_872_000], [46]), (59, [2_835_349_504, 2_629_828_608], [])]
    for layout, (rows, bases, pins) in zip(generated["layouts"], expected, strict=True):
        assert [rank["reserve_base_bytes"] for rank in layout["ranks"]] == bases
        for rank in layout["ranks"]:
            assert len(rank["entries"]) == rows
            assert [entry["rows"] for entry in rank["entries"] if entry["pin"]] == pins
            assert rank["reserve_extra_bytes"] == MODULE.CHUNK
            assert rank["zero_replay_sibling_bytes"] == 0
            assert rank["lazy_zero_replay_bytes"] > 0
    for layout in evidence["layouts"]:
        for trace in layout["traces"]:
            assert all(graph["nodes"] > 2 and not graph["spans_devices"]
                       and graph["recaptures"] == 0 for graph in trace["graphs"])
    assert fixture.stat().st_size < 1_000_000


def step5_trace(device=0):
    def graph(layer, rows, section, hits, capture, pin=False):
        return {"site": "step5.section", "bank": 1,
                "shape": f"Key {{ layer: {layer}, rows: {rows}, section: {section} }}",
                "nodes": 4, "spans_devices": False, "hits": hits, "capture_bytes": capture,
                "hit_histogram": {"counts": [1, 1, 1, 1] if pin else [hits], "interval_ms": 10000}}
    return {"schema_version": 2, "cache": {"device": device, "sampled_at_ms": 40000},
            "graphs": [graph(0, 1, "Projection", 100, 1000),
                       graph(0, 4, "Attention", 2, 2000, True),
                       graph(1, 16, "Finish", 1, 0),
                       graph(2, 64, "Shared", 0, 999999)]}


@pytest.mark.parametrize("layout", ["fp8-2rtx-6spark", "exl3-k4-core-k6-2rtx-6spark"])
def test_step5_selects_every_replayed_section_and_heat_orders_each_rank(layout):
    result = MODULE.generate_step5([step5_trace(1), step5_trace(0)], layout)
    assert [rank["rank"] for rank in result["ranks"]] == [0, 1]
    for rank in result["ranks"]:
        assert [entry["rows"] for entry in rank["entries"]] == [4, 1, 16]
        assert [entry["pin"] for entry in rank["entries"]] == [True, False, False]
        assert rank["entries"][-1]["capture_bytes"] == 1500
        assert rank["reserve_base_bytes"] == 2 << 20
        assert rank["reserve_extra_bytes"] == 2 << 20
    assert result == MODULE.generate_step5([step5_trace(0), step5_trace(1), step5_trace(0)],
                                           layout)


def test_step5_exl3_min_and_max_have_independent_rank_identities():
    layout = "exl3-k4-core-k6-1rtx-3spark"
    result = MODULE.generate_step5([step5_trace()], layout)
    assert result["layout"] == layout
    assert len(result["ranks"]) == 1
    with pytest.raises(ValueError, match="requires 2 device traces"):
        MODULE.generate_step5([step5_trace()], "exl3-k4-core-k6-2rtx-6spark")


@pytest.mark.parametrize("change", [lambda t: t["graphs"][0].update(nodes=1),
    lambda t: t["graphs"][0].update(spans_devices=True),
    lambda t: t["graphs"][0].update(capture_bytes=None),
    lambda t: t["graphs"][0].update(shape="Key { layer: 0, rows: 1, sequence: 2 }"),
    lambda t: t["graphs"].append(copy.deepcopy(t["graphs"][0]))])
def test_step5_rejects_unmeasured_or_unsafe_section_inventory(change):
    data = step5_trace()
    change(data)
    with pytest.raises(ValueError):
        MODULE.generate_step5([data], "nvfp4-1rtx-4spark")


def test_step5_row_selection_prices_zero_replay_siblings_on_every_rank():
    data = step5_trace()
    sibling = copy.deepcopy(data["graphs"][0])
    sibling.update(shape="Key { layer: 1, rows: 1, section: Attention }", hits=0, capture_bytes=3000)
    data["graphs"].append(sibling)
    result = MODULE.generate_step5([data], "nvfp4-1rtx-4spark")["ranks"][0]
    row = next(entry for entry in result["entries"] if entry["rows"] == 1)
    assert row["capture_bytes"] == 4000
    assert len(row["sections"]) == 2
    assert result["zero_replay_sibling_bytes"] == 3000
    assert result["lazy_zero_replay_bytes"] == 999999


def test_step5_never_borrows_another_layout_or_selects_capture_only():
    with pytest.raises(ValueError):
        MODULE.generate_step5([step5_trace()], "fp8-2rtx-6spark")
    data = step5_trace()
    for graph in data["graphs"]:
        graph["hits"] = 0
    with pytest.raises(ValueError, match="no replayed"):
        MODULE.generate_step5([data], "nvfp4-1rtx-4spark")


def test_step5_extension_preserves_every_existing_family_output_byte_for_byte():
    # Hashes from the generator at 2cdcc757a3, before Step-5 dispatch was added.
    cases = [
        (MODULE.generate([trace()], "glm5"), "3081a5619f1997899f40c27fc72a392f0a0a7c6219f52dc360e41671b0cb5cc8"),
        (MODULE.generate([flash_trace()], "glm5_flash"), "135dc28da122a25b94de1cc0e525d571f006ae1ad2eee33fe55fff7984b0823a"),
        (MODULE.generate([context_trace()], "mistral_large4", layout=2), "5b9886faf0eeea59230dc82c7f60229feae0b90479576ec9438767479206bf73"),
        (MODULE.generate_qwen([qwen_trace()], "single_sparks"), "4c18df79e34426c11074a1f9f82785f916f2cccf73475d8d2f476bc811e45735"),
        (MODULE.generate([v4_trace(0, 5 << 20)], "deepseek_v4", layout="rtx1-local", existing=v4_list()),
         "fc9483c5180001c782b40e200cb5138fc3d03c7ecf71b85af34b1d77c4a71032"),
        (MODULE.generate([v41_trace()], "deepseek_v41", layout=1), "8e9f32b829849a0f3fc4a2a01c595346c80606c7ee866e50cc5b2601982df754"),
        (MODULE.generate([mimo_trace()], "mimo_v2", ranks=1, variant="mimof-48", sparks=True),
         "51a36a9ea8fe52f6b62bd8a7de62c7ff5e984cae8b499e5f3c37088d72576ce2"),
    ]
    for result, expected in cases:
        assert hashlib.sha256((json.dumps(result, indent=2) + "\n").encode()).hexdigest() == expected


def test_frequency_and_low_parameter_pin_candidate():
    result = MODULE.generate([trace()], "glm5", coverage=0.99)
    assert result["entries"] == [{"rows": 1, "table_width": 16, "pin": True},
                                 {"rows": 16, "table_width": 32, "pin": False}]
    assert result["reserve_base_bytes"] == 8 << 20
    assert result["reserve_extra_bytes"] == 2 << 20


@pytest.mark.parametrize("family", ["glm5", "mistral_large4"])
@pytest.mark.parametrize("missing", ["all", "one"])
def test_glm_legacy_trace_without_replay_hits_names_the_required_upgrade(family, missing):
    data = trace()
    for graph in data["graphs"] if missing == "all" else data["graphs"][:1]:
        del graph["hits"]
    kwargs = {"layout": 1} if family == "mistral_large4" else {}
    with pytest.raises(ValueError, match="legacy trace has no replay hits; regenerate with schema 2"):
        MODULE.generate([data], family, **kwargs)


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


@pytest.mark.parametrize("family", ["glm5", "mistral_large4", "glm5_flash"])
def test_glm_capture_only_geometry_never_enters_coverage_or_pins(family):
    if family == "glm5_flash":
        data = flash_trace()
    else:
        data = modern_trace()
    # A huge lazy-capture count must not dilute the replay denominator. Even
    # churn/heat cannot turn a never-replayed geometry into startup demand.
    hot, cold = data["graphs"][0], data["graphs"][-1]
    hot.update(hits=1000, cycles={"decode": 1001})
    cold.update(hits=0, count=100_000, cycles={"decode": 100_000},
                evictions=3, recaptures=3,
                hit_histogram={"interval_ms": 10_000, "start_ms": 0, "counts": [1] * 5})
    data["graphs"] = [hot, cold]
    kwargs = {"layout": 1} if family == "mistral_large4" else {}
    result = MODULE.generate([data], family, **kwargs)
    assert len(result["entries"]) == 1
    assert result["entries"][0]["rows"] == (4 if family == "glm5_flash" else 1)


@pytest.mark.parametrize("family", ["glm5", "mistral_large4", "glm5_flash"])
def test_glm_trace_without_replays_cannot_propose_prewarm(family):
    data = flash_trace() if family == "glm5_flash" else modern_trace()
    for graph in data["graphs"]:
        graph["hits"] = 0
    kwargs = {"layout": 1} if family == "mistral_large4" else {}
    with pytest.raises(ValueError, match="no matching graph identities"):
        MODULE.generate([data], family, **kwargs)


@pytest.mark.parametrize("family", ["glm5", "mistral_large4", "glm5_flash"])
def test_glm_startup_hits_are_capped_by_request_cycles(family):
    data = flash_trace() if family == "glm5_flash" else modern_trace()
    hot, startup = data["graphs"][0], data["graphs"][-1]
    hot.update(hits=1000, cycles={"verify": 1001})
    startup.update(hits=1_000_000, cycles={"other": 1_000_000, "decode": 1})
    data["graphs"] = [hot, startup]
    kwargs = {"layout": 1} if family == "mistral_large4" else {}
    result = MODULE.generate([data], family, coverage=0.99, **kwargs)
    assert len(result["entries"]) == 1
    assert result["entries"][0]["rows"] == (4 if family == "glm5_flash" else 1)


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








@pytest.mark.parametrize("site,shape,expected", [
    ("window", "(16, 987654321)", ("row", 16, 0)),
    ("compressor", "(1, 998)", ("row", 1, 0)),
    ("index_selection", "[7, 16, 100, 200, 300, 1, 500, 512, 0, 1]", ("row", 16, 0)),
    ("sparse_attention", "(7, [7, 16, 100, 200, 300])", ("row", 16, 0)),
    ("sparse_attention", "(7, [18446744073709551615, 7, 16, 100, 200, 300])", ("row", 16, 0)),
    ("dual_attention_tail", "(7, [100, 200, 300, 16, 500])", ("row", 16, 0)),
    ("target_head", "112", ("row", 112, 0)),
    ("dspark_chain", "(16, 7)", ("draft", 16, 7)),
    ("dspark_distributed_terminal", "(5, 1, 16)", ("draft", 16, 5)),
    ("dspark_cache", "()", None),
])
def test_v41_native_site_axes(site, shape, expected):
    assert MODULE.v41_geometry(shape, set(range(1, 65)) | {128}, "deepseek_v41." + site) == expected


@pytest.mark.parametrize("site,shape", [
    ("dspark_chain", "(5, 16)"),
    ("dspark_distributed_terminal", "(7, 2, 16)"),
    ("window", "(987654321, 16)"),
    ("index_selection", "[7, 16]"),
    ("sparse_attention", "(7, [1, 2, 3])"),
    ("unknown", "16"),
])
def test_v41_unknown_or_invalid_axes_fail(site, shape):
    with pytest.raises(ValueError):
        MODULE.v41_geometry(shape, set(range(1, 65)), "deepseek_v41." + site)






def flash_graph(rows, spec, long, width, stride, pool_stride, count, segment=0,
                site="glm5_flash.decode", capture_bytes=1 << 20):
    return {"device": 0, "site": site, "bank": 1,
            "shape": f"GraphKey {{ segment: {segment}, rows: {rows}, spec: {spec}, long: {long}, "
                     f"pool_width: {width}, page_stride: {stride}, pool_stride: {pool_stride} }}",
            "count": count, "hits": count, "capture_bytes": capture_bytes, "executables": 1}


def flash_trace(graphs=None, charged=16 << 20, capture_bytes=1 << 20, device=0):
    return {"schema_version": 2, "cache": {"device": device, "charged": charged},
            "sites": {"glm5_flash.decode": {"largest_bytes": 3 << 20}},
            "graphs": graphs if graphs is not None else
            [flash_graph(4, "false", "false", 0, 64, 0, 100, segment=0,
                         capture_bytes=capture_bytes),
             flash_graph(4, "false", "false", 0, 64, 0, 90, segment=3,
                         capture_bytes=capture_bytes),
             flash_graph(16, "true", "false", 0, 128, 0, 30, segment=0,
                         capture_bytes=capture_bytes),
             flash_graph(1, "true", "true", 2048, 64, 16, 1, segment=2,
                         capture_bytes=capture_bytes)]}


def test_flash_checked_in_list_regenerates_byte_for_byte_from_source_traces(tmp_path):
    import gzip
    fixture = Path(__file__).parent / "fixtures/glmf-prewarm-traces.json.gz"
    traces = json.loads(gzip.decompress(fixture.read_bytes()))
    groups = {}
    for layout, snapshots in traces.items():
        groups[layout] = []
        for index, snapshot in enumerate(snapshots):
            path = tmp_path / f"{layout}-{index}.json"
            path.write_text(json.dumps(snapshot))
            groups[layout].append(path)
    generated = MODULE.generate_layouts(groups, "glm5_flash")
    checked_in = SCRIPT.parents[2] / "rust/crates/cuteafd-loader/src/serving_capacity/glmf_prewarm.json"
    assert json.dumps(generated, indent=2) + "\n" == checked_in.read_text()


def test_flash_shapes_aggregate_segments_and_pin_the_hot_ones():
    result = MODULE.generate([flash_trace()], "glm5_flash", coverage=0.99)
    assert result["family"] == "glm5_flash"
    # The base is the selected entries' measured capture bytes (three rows of
    # 1 MiB on device 0), not the 16 MiB whole-session arena the trace held.
    assert result["measured_bytes"] == [4 << 20]
    assert result["measured_executables"] == [3]
    # Startup prewarms every traced shape, so the reserve base holds the whole
    # request demand; the tail is the fixed 2 MiB allocation headroom.
    assert result["reserve_extra_bytes"] == 2 << 20
    # 99% coverage keeps the two hot shapes (190 and 30 hits); only the peak is pinned.
    assert result["entries"] == [
        {"rows": 4, "spec": False, "long": False, "pool_width": 0,
         "page_stride": 64, "pool_stride": 0, "pin": True},
        {"rows": 16, "spec": True, "long": False, "pool_width": 0,
         "page_stride": 128, "pool_stride": 0, "pin": False},
    ]


def test_flash_normalized_mla_shapes_project_kda_and_head_once():
    def graph(segment, stride=0, spec="false", head="false", hits=1000, share=1 << 20):
        item = flash_graph(4, spec, "false", 0, stride, 0, hits, segment=segment,
                           capture_bytes=share)
        item["shape"] = item["shape"].replace(f"spec: {spec},", f"spec: {spec}, head: {head},")
        return item
    data = flash_trace([graph(0), graph(1, stride=64, hits=100, share=2 << 20),
                        graph(1, stride=128, hits=50, share=3 << 20),
                        graph(2, head="true", share=4 << 20), graph(3, hits=0),
                        graph(4, spec="true", share=99 << 20)])
    result = MODULE.generate([data], "glm5_flash", coverage=1)
    assert [entry["page_stride"] for entry in result["entries"]] == [64, 128]
    assert result["measured_bytes"] == [12 << 20]
    assert result["measured_executables"] == [5]
    assert result == MODULE.generate([data, data], "glm5_flash", coverage=1)
    first = MODULE.generate([data], "glm5_flash", coverage=0.6)
    assert len(first["entries"]) == 1
    assert first["measured_bytes"] == [8 << 20]
    assert first["measured_executables"] == [4]
    missing = copy.deepcopy(data)
    missing["graphs"][0]["capture_bytes"] = None
    with pytest.raises(ValueError, match="no measured capture bytes"):
        MODULE.generate([missing], "glm5_flash")
    mixed = copy.deepcopy(data)
    mixed["graphs"].append(flash_graph(4, "false", "false", 0, 64, 0, 10))
    with pytest.raises(ValueError, match="one graph-key format"):
        MODULE.generate([mixed], "glm5_flash")


@pytest.mark.parametrize("family", ["qwen4", "glm5_flash"])
@pytest.mark.parametrize("nodes", [0, 1, 2])
def test_eager_records_do_not_affect_prewarm_style_or_selected_sibling_cost(family, nodes):
    data = qwen_trace() if family == "qwen4" else flash_trace()

    def generate(trace):
        return (MODULE.generate_qwen([trace], "single_0spark") if family == "qwen4"
                else MODULE.generate([trace], family))

    expected = generate(data)
    for shape in [data["graphs"][0]["shape"], "unparseable eager identity"]:
        eager = copy.deepcopy(data["graphs"][0])
        eager.update(key_id=9999, shape=shape, permanently_eager=True, executables=0,
                     permanent_bytes=0, nodes=nodes, capture_bytes=None, hits=0, count=1,
                     cycles={"other": 1})
        candidate = copy.deepcopy(data)
        candidate["graphs"].append(eager)
        assert generate(candidate) == expected


def test_flash_short_steps_collapse_the_unkeyed_pool_geometry():
    graphs = [flash_graph(4, "false", "false", 1, 64, 16, 10),
              flash_graph(4, "false", "false", 4, 64, 32, 5)]
    result = MODULE.generate([flash_trace(graphs)], "glm5_flash")
    assert result["entries"] == [{"rows": 4, "spec": False, "long": False, "pool_width": 0,
                                  "page_stride": 64, "pool_stride": 0, "pin": True}]


def test_flash_startup_prewarm_is_not_request_demand():
    graphs = flash_trace()["graphs"]
    for graph in graphs:
        graph["cycles"] = {"other": 1000}
    graphs[0]["cycles"] = {"other": 1000, "verify": 5}
    result = MODULE.generate([flash_trace(graphs)], "glm5_flash")
    assert [(entry["rows"], entry["pin"]) for entry in result["entries"]] == [(4, True)]


def test_flash_each_device_keeps_its_own_measured_base():
    first = flash_trace([flash_graph(4, "false", "false", 0, 64, 0, 10)],
                        charged=8 << 20)
    second = flash_trace([flash_graph(4, "false", "false", 0, 64, 0, 10,
                                      capture_bytes=3 << 20)],
                         charged=24 << 20, device=1)
    result = MODULE.generate([first, second], "glm5_flash")
    # One selected shape per device, 1 MiB and 3 MiB; each packs once to its
    # own 2 MiB / 4 MiB base, never scaled from the other rank's price.
    assert (result["measured_bytes"], result["measured_executables"]) == ([2 << 20, 4 << 20], [1, 1])


def test_flash_a_selected_shape_without_a_measured_share_fails():
    missing = flash_trace()
    missing["graphs"][0]["capture_bytes"] = None
    with pytest.raises(ValueError, match="no measured capture bytes"):
        MODULE.generate([missing], "glm5_flash")
    # A device trace that never captured one of the layout's selected shapes
    # cannot price that shape; the base would be short by an unknown share.
    incomplete = flash_trace([flash_graph(4, "false", "false", 0, 64, 0, 100)])
    with pytest.raises(ValueError, match="lacks selected shapes"):
        MODULE.generate([flash_trace(), incomplete], "glm5_flash")


@pytest.mark.parametrize("change", [
    lambda t: t["graphs"][0].update(shape="GlmGraphKey { rows: 1 }"),
    lambda t: t["graphs"][0].update(capture_bytes=None),
    lambda t: t["cache"].update(device=1),
    lambda t: t.update(graphs=[flash_graph(0, "false", "false", 0, 64, 0, 5)]),
    lambda t: t.update(graphs=[flash_graph(4, "false", "true", 1, 64, 0, 5)]),
])
def test_flash_invalid_trace_fails(change):
    data = flash_trace()
    change(data)
    with pytest.raises(ValueError):
        MODULE.generate([data], "glm5_flash")


def flash_trace_path(tmp_path, name, capture_bytes):
    path = tmp_path / name
    path.write_text(json.dumps(flash_trace(capture_bytes=capture_bytes)))
    return path


def test_flash_shapes_are_priced_separately_by_resolved_shape(tmp_path):
    one = flash_trace_path(tmp_path, "one.json", 1 << 20)
    two = flash_trace_path(tmp_path, "two.json", 3 << 20)
    groups = MODULE.layout_groups([f"1-sparks={one}", f"2={two}"], "glm5_flash")
    result = MODULE.generate_layouts(groups, "glm5_flash")
    assert result["schema_version"] == 3
    assert result["family"] == "glm5_flash"
    assert set(result["layouts"]) == {"1-sparks", "2"}
    # Each layout prices its own selected entries: the same four rows of
    # captures, 4 MiB at the 1 MiB row price and 12 MiB at the 3 MiB row price.
    assert result["layouts"]["1-sparks"]["measured_bytes"] == [4 << 20]
    assert result["layouts"]["2"]["measured_bytes"] == [12 << 20]
    # A shape entry carries only its own trace's prices, never the document's
    # schema or another shape's arena.
    assert "schema_version" not in result["layouts"]["1-sparks"]
    assert result["layouts"]["1-sparks"]["entries"] == result["layouts"]["2"]["entries"]


def test_flash_prices_a_layout_from_the_costlier_quant(tmp_path):
    # One layout traced twice, as an EXL3 and an FP8 checkpoint would be. The
    # entries are the same request shapes either way; the price is the costlier
    # quant's selected-entry sum, taken as that trace's own (bytes, executables)
    # pair, never a mix of the two.
    cheap = flash_trace([flash_graph(4, "false", "false", 0, 64, 0, 100, segment=0),
                         flash_graph(4, "false", "false", 0, 64, 0, 90, segment=1)], charged=16 << 20)
    dear = flash_trace([flash_graph(4, "false", "false", 0, 64, 0, 100, segment=0,
                                    capture_bytes=3 << 20),
                        flash_graph(4, "false", "false", 0, 64, 0, 90, segment=1,
                                    capture_bytes=3 << 20)], charged=20 << 20)
    dear["sites"]["glm5_flash.decode"]["largest_bytes"] = 5 << 20
    paths = []
    for name, data in (("exl3.json", cheap), ("fp8.json", dear)):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        paths.append(path)
    groups = MODULE.layout_groups([f"1-sparks={path}" for path in paths], "glm5_flash")
    layout = MODULE.generate_layouts(groups, "glm5_flash")["layouts"]["1-sparks"]
    assert (layout["measured_bytes"], layout["measured_executables"]) == ([6 << 20], [2])
    assert layout["reserve_extra_bytes"] == 2 << 20
    assert [(entry["rows"], entry["spec"]) for entry in layout["entries"]] == [(4, False)]


def test_flash_accepts_a_subset_of_traced_shapes_and_no_unknown_keys(tmp_path):
    path = flash_trace_path(tmp_path, "trace.json", 16 << 20)
    # An untraced shape falls back to the engine's bootstrap set, so a partial
    # document is valid; an unknown key is still refused.
    groups = MODULE.layout_groups([f"2-sparks={path}"], "glm5_flash")
    assert set(MODULE.generate_layouts(groups, "glm5_flash")["layouts"]) == {"2-sparks"}
    with pytest.raises(ValueError, match="unknown GLM Flash layout"):
        MODULE.layout_groups([f"min={path}"], "glm5_flash")
    with pytest.raises(ValueError, match="unknown GLM Flash layout"):
        MODULE.layout_groups([f"1-sparks={path}", f"mid={path}"], "glm5_flash")


def test_single_layout_families_keep_the_flat_document(tmp_path):
    data = trace()
    path = tmp_path / "glm5.json"
    path.write_text(json.dumps(data))
    result = MODULE.generate_layouts(MODULE.layout_groups([str(path)], "glm5"), "glm5")
    assert "layouts" not in result
    assert result["family"] == "glm5"

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


def test_layout_is_required_for_mistral_large4_and_optional_for_glm5():
    with pytest.raises(ValueError):
        MODULE.generate([trace()], "mistral_large4")
    assert MODULE.generate([trace()], "glm5", layout=1)["layout"] == 1
    two = MODULE.generate([context_trace()], "glm5", coverage=0.99, layout=2)
    assert two["layout"] == 2
    assert [(entry["rows"], entry["table_width"]) for entry in two["entries"]] == [(1, 16), (16, 32)]


def test_glm_two_rank_cli_keeps_built_widths(tmp_path):
    import subprocess
    import sys

    paths = []
    for device in (0, 1):
        path = tmp_path / f"trace.device-{device}.json"
        path.write_text(json.dumps(context_trace(device)))
        paths.extend(["--trace", str(path)])
    output = tmp_path / "glm_prewarm_2.json"
    subprocess.run([sys.executable, str(SCRIPT), "--family", "glm5", "--layout", "2",
                    *paths, "--output", str(output)], check=True, capture_output=True, text=True)
    expected = MODULE.generate([context_trace(0), context_trace(1)], "glm5", layout=2)
    assert output.read_text() == json.dumps(expected, indent=2) + "\n"


def test_glm_two_rank_trace_reproduces_checked_in_bytes():
    import gzip

    fixture = Path(__file__).parent / "fixtures/glm-prewarm-two-rank-trace.json.gz"
    traces = json.loads(gzip.decompress(fixture.read_bytes()))
    assert {trace["cache"]["device"] for trace in traces} == {0, 1}
    generated = MODULE.generate(traces, "glm5", layout=2)
    installed = SCRIPT.parents[2] / "rust/crates/cuteafd-loader/src/serving_capacity/glm_prewarm_2.json"
    assert json.dumps(generated, indent=2) + "\n" == installed.read_text()
    assert generated["layout"] == 2
    assert generated["reserve_extra_bytes"] == MODULE.CHUNK
    replayed = {MODULE.geometry("glm5", graph["shape"])
                for trace in traces for graph in trace["graphs"]
                if MODULE.glm_request_replays(graph) > 0}
    assert all((entry["rows"], entry["table_width"] // 2) in replayed
               for entry in generated["entries"])
    assert fixture.stat().st_size < 1_000_000


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


def test_v41_uses_its_layout_schema_not_a_glm_prefix_consumer():
    assert "deepseek_v41" not in MODULE.SITE_PREFIX
    with pytest.raises(ValueError, match="deepseek_v41 requires --layout 1 or 2"):
        MODULE.generate([trace()], "deepseek_v41")


def test_flash_is_not_advertised_as_a_glm_prefix_consumer():
    assert "glm5_flash" not in MODULE.SITE_PREFIX


@pytest.mark.parametrize("family", ["glm5", "mistral_large4"])
def test_every_consumer_uses_fixed_allocation_tail_not_largest_graph(family):
    data = trace()
    for graph in data["graphs"]:
        graph["site"] = MODULE.SITE_PREFIX[family] + "segment"
    data["sites"] = {MODULE.SITE_PREFIX[family] + "segment": {"largest_bytes": 80 << 20}}
    result = MODULE.generate([data], family, **({"layout": 1} if family == "mistral_large4" else {}))
    assert result["reserve_base_bytes"] == 8 << 20
    assert result["reserve_extra_bytes"] == 2 << 20


@pytest.mark.parametrize("filename", ["glm_prewarm_1.json", "ml4_prewarm_1.json", "ml4_prewarm_2.json"])
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


def qwen_identity(segment, rows=64, spec=False, long=False, width=64, page_stride=64, pool_stride=128):
    return (f"GraphKey {{ segment: {segment}, rows: {rows}, spec: {str(spec).lower()}, "
            f"long: {str(long).lower()}, pool_width: {width}, page_stride: {page_stride}, "
            f"pool_stride: {pool_stride} }}")


def qwen_trace(device=0, charged=8 << 20, capture=(1 << 20, 2 << 20, 4 << 20)):
    def graph(segment, rows, count, share, hits=None):
        return {"device": device, "site": "qwen4.segment", "bank": 0,
                "key_id": segment * 1000 + rows,
                "shape": qwen_identity(segment, rows=rows),
                "count": count, "cycles": {"decode": count},
                "hits": count if hits is None else hits,
                "hit_histogram": {"rolling": True, "interval_ms": 10_000,
                                  "start_ms": 0, "counts": [5, 0, 0, 0, 0]},
                "evictions": 0, "recaptures": 0,
                "capture_bytes": share, "permanent_bytes": share * 2,
                "nodes": 10, "executables": 1, "basis": {"kind": "bytes"}}
    return {"schema_version": 2,
            "cache": {"device": device, "charged": charged, "sampled_at_ms": 10_000},
            "sites": {"qwen4.segment": {"largest_bytes": 3 << 20}},
            "graphs": [graph(0, 64, 100, capture[0]), graph(1, 64, 100, capture[1]),
                       graph(1, 128, 1, capture[2])]}


def test_qwen_shape_list_is_keyed_by_resolved_shape_and_device():
    result = MODULE.generate_qwen([qwen_trace(0, 8 << 20), qwen_trace(1, 12 << 20)],
                                  "dual_0spark", coverage=0.99)
    assert result["shape"] == "dual_0spark"
    # One 2 MiB chunk each: the selected (rows=64) identities' measured shares
    # 1 MiB + 2 MiB on rank 0, 1 MiB + 2 MiB on rank 1, rounded once. The
    # trace's charged high-water is never the base.
    assert result["reserve_base_bytes"] == [4 << 20, 4 << 20]
    assert result["reserve_extra_bytes"] == [2 << 20, 2 << 20]
    # Layer segments share one capture shape: one entry, not one per segment.
    assert result["trace"]["entries"] == [
        {"rows": 64, "spec": False, "long": False, "pool_width": 64,
         "page_stride": 64, "pool_stride": 128, "pin": False}]
    assert "dual_0spark" in result["trace"]["provenance"]


def test_qwen_base_ignores_charged_and_counts_startup_only_siblings():
    quiet = qwen_trace(0, 64 << 20, capture=(1 << 20, 2 << 20, 4 << 20))
    # A startup-only sibling is captured but never replayed; it still costs its
    # share in the arena. Moving the charged counter must not move the base.
    quiet["graphs"][1]["hits"] = 0
    result = MODULE.generate_qwen([quiet], "single_0spark", coverage=0.99)
    assert result["reserve_base_bytes"] == [4 << 20]
    loud = copy.deepcopy(quiet)
    loud["cache"]["charged"] = 5 << 30
    assert MODULE.generate_qwen([loud], "single_0spark", coverage=0.99)["reserve_base_bytes"] == [4 << 20]


def test_qwen_zero_hit_bootstrap_shapes_are_never_selected():
    # The provisional startup set captures many shapes the traffic never
    # replays. A startup capture is a miss and an event, never demand: ranking
    # by replay hits keeps every zero-hit shape out of the list (#1652), even
    # though their captured bytes are real.
    data = qwen_trace()
    hot = data["graphs"][0]
    hot["hits"] = 100
    cold = []
    for segment in range(1, 12):
        graph = dict(data["graphs"][1])
        graph.update(key_id=segment, hits=0, count=1, misses=1, cycles={"decode": 1},
                     capture_bytes=1 << 20, shape=qwen_identity(segment, rows=4))
        cold.append(graph)
    data["graphs"] = [hot] + cold
    summary = {}
    result = MODULE.generate_qwen([data], "single_0spark", summary=summary)
    assert result["trace"]["entries"] == [
        {"rows": 64, "spec": False, "long": False, "pool_width": 64,
         "page_stride": 64, "pool_stride": 128, "pin": False}]
    assert result["reserve_base_bytes"] == [2 << 20]
    # Captured once, never replayed: never selected, and reported rather than
    # hidden (11 cold identities, 1 MiB each).
    assert summary == {"zero_replay_identities": len(cold), "zero_replay_bytes": len(cold) << 20,
                       "zero_share_identities": 0, "zero_share_bytes": 0}
    assert f"{len(cold)} zero-replay identities" in result["trace"]["provenance"]


def test_qwen_zero_share_identity_is_charged_the_device_mean():
    # A capture boundary with no new physical growth is allocator reuse, not a
    # free graph. The shared cache keeps such a record's estimate; the base
    # must not treat the zero as a measured zero, or a selected shape's share
    # vanishes from the reserve.
    data = qwen_trace()
    data["graphs"][0]["capture_bytes"] = 0
    data["graphs"][2]["capture_bytes"] = 1 << 20
    summary = {}
    result = MODULE.generate_qwen([data], "single_0spark", coverage=0.99, summary=summary)
    # Device mean over three captures: (0 + 2 + 1) MiB / 3 = 1 MiB; the zero
    # selected identity is charged it, so 1 + 2 = 3 MiB -> one 2 MiB chunk.
    assert result["reserve_base_bytes"] == [4 << 20]
    assert summary["zero_share_identities"] == 1
    assert summary["zero_share_bytes"] == 1 << 20
    assert "1 zero-share selected identities" in result["trace"]["provenance"]
    # A measured zero outside the selected set is not filled.
    data = qwen_trace()
    data["graphs"][2]["capture_bytes"] = 0
    assert MODULE.generate_qwen([data], "single_0spark", coverage=0.99)["reserve_base_bytes"] == [4 << 20]


def test_qwen_selected_identity_without_a_measured_share_fails():
    data = qwen_trace()
    data["graphs"][1]["capture_bytes"] = None
    with pytest.raises(ValueError, match="without a measured capture share"):
        MODULE.generate_qwen([data], "single_0spark", coverage=0.99)
    # A shape outside the selected set may stay unmeasured.
    data = qwen_trace()
    data["graphs"][2]["capture_bytes"] = None
    assert MODULE.generate_qwen([data], "single_0spark", coverage=0.99)["trace"]["entries"]


def test_qwen_regeneration_reproduces_a_checked_in_generation_exactly():
    # A committed fixture trace reproduces byte-identical base and extra: the
    # checked-in lists must never be hand-edited.
    fixture = Path(__file__).resolve().parent / "fixtures" / "qwen-prewarm-trace"
    traces = [json.loads(path.read_text()) for path in sorted(fixture.glob("trace.device-*.json"))]
    generated = MODULE.generate_qwen(traces, "single_0spark", coverage=0.99)
    expected = json.loads((fixture / "expected-single_0spark.json").read_text())
    assert generated == expected



def normalized_qwen_identity(segment, rows=64, spec=False, head=False, long=False,
                             page_width=0, page_stride=0, pool_stride=0):
    return (f"GraphKey {{ segment: {segment}, rows: {rows}, spec: {str(spec).lower()}, "
            f"head: {str(head).lower()}, long: {str(long).lower()}, page_width: {page_width}, "
            f"page_stride: {page_stride}, pool_stride: {pool_stride} }}")


def test_qwen_normalized_full_shapes_project_gdn_and_explicit_head_once():
    data = qwen_trace(capture=(1 << 20, 2 << 20, 4 << 20))
    template = data["graphs"][0]
    data["graphs"] = []
    for identity, hits, share in [
            (normalized_qwen_identity(0), 1000, 1 << 20),
            (normalized_qwen_identity(1, page_width=64, page_stride=64), 100, 2 << 20),
            (normalized_qwen_identity(1, long=True, page_width=128, page_stride=128, pool_stride=32), 50, 3 << 20),
            (normalized_qwen_identity(2, head=True), 1000, 4 << 20),
            # A startup-only GDN sibling belongs to selected Full steps too.
            (normalized_qwen_identity(3), 0, 1 << 20),
            # Same rows but different spec is not a compatible GDN identity.
            (normalized_qwen_identity(4, spec=True), 1000, 99 << 20)]:
        graph = copy.deepcopy(template)
        graph.update(shape=identity, key_id=len(data["graphs"]), hits=hits, capture_bytes=share)
        data["graphs"].append(graph)
    result = MODULE.generate_qwen([data], "single_0spark", coverage=1)
    assert result["trace"]["entries"] == [
        {"rows": 64, "spec": False, "long": False, "pool_width": 0,
         "page_stride": 64, "pool_stride": 0, "pin": False},
        {"rows": 64, "spec": False, "long": True, "pool_width": 0,
         "page_stride": 128, "pool_stride": 32, "pin": False}]
    assert result["reserve_base_bytes"] == [12 << 20]  # 1+2+3+4+1, rounded once.
    # Ranking uses Full demand, not the much hotter zero-geometry identities.
    first = MODULE.generate_qwen([data], "single_0spark", coverage=0.6)
    assert len(first["trace"]["entries"]) == 1
    assert first["reserve_base_bytes"] == [8 << 20]
    repeated = MODULE.generate_qwen([data, data], "single_0spark", coverage=1)
    assert repeated["trace"]["entries"] == result["trace"]["entries"]
    assert repeated["reserve_base_bytes"] == result["reserve_base_bytes"]
    assert repeated["reserve_extra_bytes"] == result["reserve_extra_bytes"]


def test_qwen_normalized_trace_rejects_unrepresentable_and_mixed_geometries():
    data = qwen_trace()
    data["graphs"][0]["shape"] = normalized_qwen_identity(0, page_width=32, page_stride=64)
    with pytest.raises(ValueError, match="one graph-key format"):
        MODULE.generate_qwen([data], "single_0spark")
    data["graphs"] = [data["graphs"][0]]
    with pytest.raises(ValueError, match="unsupported normalized"):
        MODULE.generate_qwen([data], "single_0spark")


def qwen_list():
    def shape(name, owners):
        return {"shape": name, "reserve_base_bytes": [5 << 30] * owners,
                "reserve_extra_bytes": [2 << 20] * owners, "trace": None}
    return {"schema_version": 1, "family": "qwen4", "provenance": "bootstrap",
            "bootstrap": {"entries": []},
            "shapes": [shape("single_0spark", 1), shape("single_sparks", 1),
                       shape("dual_0spark", 2), shape("dual_sparks", 2)]}


def fixture_traces():
    fixture = Path(__file__).resolve().parent / "fixtures" / "qwen-prewarm-trace"
    return [json.loads(path.read_text()) for path in sorted(fixture.glob("trace.device-*.json"))]


def test_qwen_merge_replaces_only_the_traced_shape():
    generated = MODULE.generate_qwen(fixture_traces(), "single_0spark", coverage=0.99)
    merged = MODULE.merge_qwen(qwen_list(), generated)
    installed = {prewarm["shape"]: prewarm for prewarm in merged["shapes"]}
    assert installed["single_0spark"] == generated
    assert installed["dual_0spark"]["trace"] is None
    assert installed["dual_0spark"]["reserve_base_bytes"] == [5 << 30, 5 << 30]
    assert "single_0spark" in merged["provenance"]
    assert "dual_sparks" in merged["provenance"], "untraced shapes are named"
    assert MODULE.merge_qwen(merged, generated) == merged, "merging is idempotent"


def test_qwen_merge_installs_the_generated_base_and_extra_exactly():
    fixture = Path(__file__).resolve().parent / "fixtures" / "qwen-prewarm-trace"
    generated = MODULE.generate_qwen(fixture_traces(), "single_0spark", coverage=0.99)
    merged = MODULE.merge_qwen(qwen_list(), generated)
    installed = next(prewarm for prewarm in merged["shapes"] if prewarm["shape"] == "single_0spark")
    assert installed == json.loads((fixture / "expected-single_0spark.json").read_text())


def test_qwen_merge_rejects_a_list_without_the_shape():
    generated = MODULE.generate_qwen(fixture_traces(), "single_0spark", coverage=0.99)
    document = qwen_list()
    document["shapes"] = [prewarm for prewarm in document["shapes"]
                          if prewarm["shape"] != "single_0spark"]
    with pytest.raises(ValueError, match="no single_0spark"):
        MODULE.merge_qwen(document, generated)


def test_qwen_single_shape_needs_exactly_one_device_trace():
    assert len(MODULE.generate_qwen([qwen_trace()], "single_0spark")["reserve_base_bytes"]) == 1
    with pytest.raises(ValueError):
        MODULE.generate_qwen([qwen_trace()], "dual_sparks")


def test_qwen_unparseable_identity_fails():
    data = qwen_trace()
    data["graphs"][0]["shape"] = "bad"
    with pytest.raises(ValueError):
        MODULE.generate_qwen([data], "single_0spark")


def test_qwen_family_requires_shape_argument():
    with pytest.raises(ValueError):
        MODULE.generate([qwen_trace()], "qwen4")


def v4_shape(rows, width, attention="c4"):
    return (f'GraphKey {{ layer: 3, rows: {rows}, chunked: false, attention: "{attention}", '
            f'table_width: {width}, table_stride: {width}, previous: 0, '
            'tp2_broadcast: false, exchange_f32: false }')


def v4_list():
    shapes = {"rtx1-spark": 1, "rtx1-local": 1, "rtx2-spark": 2, "rtx2-local": 2}
    return {"schema_version": 1, "family": "deepseek_v4", "layouts": {
        key: {"traced": False, "reserve_base_bytes": [0] * ranks,
              "reserve_extra_bytes": [0] * ranks,
              "bootstrap_bytes": [1 << 20] * ranks, "entries": []}
        for key, ranks in shapes.items()}}


def v4_trace(device, charged, capture=2 << 20):
    """One rank's snapshot: a regular verify geometry, a startup row and a rare
    verify geometry used once inside one sampling interval."""
    def graph(rows, width, count, attention="c4", heat="regular"):
        counts = {"rare": [count, 0, 0, 0, 0, 0],
                  "regular": [1, 0, 0, count - 2, 0, 0],
                  "sustained": [count // 6] * 6}[heat]
        hits = 0 if heat == "rare" else count - 1
        return {"device": device, "site": "deepseek_v4.segment", "bank": 1,
                "shape": v4_shape(rows, width, attention), "count": count, "key_id": (rows, width),
                "hits": hits, "cycles": {"decode": count}, "capture_bytes": capture,
                "hit_histogram": {"interval_ms": 10_000, "start_ms": 0, "counts": counts}}
    return {"schema_version": 2, "cache": {"device": device, "charged": charged},
            "sites": {"deepseek_v4.segment": {"largest_bytes": 3 << 20}},
            "graphs": [graph(16, 512, 100), graph(4, 0, 90, "local-experts"),
                       graph(1, 16, 1, heat="rare")]}


def test_v4_geometry_allows_the_width_zero_startup_set():
    assert MODULE.geometry("deepseek_v4", v4_shape(4, 0, "local-experts")) == (4, 0)
    assert MODULE.geometry("deepseek_v4", v4_shape(16, 512)) == (16, 512)
    with pytest.raises(ValueError):
        MODULE.geometry("deepseek_v4", "GraphKey { rows: 1 }")


def test_v4_eager_tails_never_become_prewarm_entries():
    assert MODULE.prewarmable("deepseek_v4", v4_shape(4, 0, "local-experts"))
    # A 2-RTX shape's startup set is the TP2 router pair at width 0.
    assert MODULE.prewarmable("deepseek_v4", v4_shape(4, 0, "tp2-broadcast"))
    assert MODULE.prewarmable("deepseek_v4", v4_shape(4, 0, "tp2-replicated"))
    assert MODULE.prewarmable("deepseek_v4", v4_shape(16, 512))
    assert not MODULE.prewarmable("deepseek_v4", v4_shape(5, 0, "tail"))
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].append({"device": 0, "site": "deepseek_v4.segment", "bank": 1,
                            "shape": v4_shape(5, 0, "tail"), "count": 4, "key_id": (5, 0),
                            "cycles": {"decode": 4}, "capture_bytes": 2 << 20})
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list())
    assert (5, 0) not in {(row["rows"], row["table_width"])
                          for row in entry["layouts"]["rtx1-local"]["entries"]}


def test_v4_replaces_one_shape_with_per_rank_arrays():
    result = MODULE.generate([v4_trace(0, 5 << 20, 6 << 20), v4_trace(1, 7 << 20, 4 << 20)],
                             "deepseek_v4", layout="rtx2-spark", existing=v4_list())
    assert result["layouts"]["rtx1-spark"] == v4_list()["layouts"]["rtx1-spark"]
    assert result["layouts"]["rtx2-local"] == v4_list()["layouts"]["rtx2-local"]
    entry = result["layouts"]["rtx2-spark"]
    assert entry["traced"] is True
    # The one-use geometry is rare and stays lazy; the base is the warmed set's
    # own captures: two geometries of 6 MiB on device 0, 4 MiB on 1.
    assert {(row["rows"], row["table_width"]) for row in entry["entries"]} == {(4, 0), (16, 512)}
    assert entry["reserve_base_bytes"] == [12 << 20, 8 << 20]
    # The arena tail only: capture headroom is a shared launcher policy.
    assert entry["reserve_extra_bytes"] == [2 << 20, 2 << 20]
    assert entry["bootstrap_bytes"] == [6 << 20, 4 << 20]


def test_v4_heat_split_pins_sustained_prewarms_regular_and_leaves_rare_lazy():
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].append({
        "device": 0, "site": "deepseek_v4.segment", "bank": 1,
        "shape": v4_shape(8, 64), "count": 60, "key_id": (8, 64), "hits": 59,
        "cycles": {"verify": 60}, "capture_bytes": 3 << 20,
        "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                          "counts": [10, 10, 10, 10, 10, 10]}})
    heat = []
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list(),
                            heat_report=heat)["layouts"]["rtx1-local"]
    entries = {(row["rows"], row["table_width"]): row["pin"] for row in entry["entries"]}
    assert entries[(8, 64)] is True, "sustained heat is pinned"
    assert entries[(16, 512)] is True, "the dominant verify geometry always pins"
    assert entries[(4, 0)] is False, "a recurring startup row is prewarmed, not pinned"
    assert (1, 16) not in entries, "a one-use geometry stays lazy"
    classes = {(row["device"], row["heat"]): (row["keys"], row["arena_bytes"], row["selected"])
               for row in heat}
    assert classes[(0, "startup")] == (1, 2 << 20, True)
    assert classes[(0, "sustained")] == (1, 3 << 20, True)
    assert classes[(0, "regular")] == (1, 2 << 20, True)
    assert classes[(0, "rare")] == (1, 2 << 20, False)
    # The rare geometry's bytes are not charged in the base: 2 + 2 + 3 MiB of
    # warmed captures, rounded up to whole 2 MiB chunks.
    assert entry["reserve_base_bytes"] == [8 << 20]


def test_v4_startup_rows_prewarm_and_only_sustained_heat_pins():
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].append({"device": 0, "site": "deepseek_v4.segment", "bank": 1,
                            "shape": v4_shape(64, 0, "local-experts"), "count": 90,
                            "key_id": (64, 0), "hits": 89, "cycles": {"decode": 90},
                            "capture_bytes": 2 << 20,
                            "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                                              "counts": [18, 18, 18, 18, 18, 0]}})
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list())
    entries = {(row["rows"], row["table_width"]): row["pin"]
               for row in entry["layouts"]["rtx1-local"]["entries"]}
    # A width-0 row the trace replayed is a startup row at every row count;
    # only the one whose own heat is sustained earns permanent residency.
    assert entries[(4, 0)] is False
    assert entries[(64, 0)] is True
    # The dominant verify geometry always pins; the other rare one stays lazy.
    assert entries[(16, 512)] is True
    assert (1, 16) not in entries


def test_v4_single_interval_startup_row_is_prewarmed_unpinned():
    # A width-0 row replayed inside one sampling interval is reachable, so it
    # is warmed; a pin otherwise needs sustained heat, and ARC must be able to
    # reclaim a row serving never revisits (TJ, 2026-11-09).
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].append({"device": 0, "site": "deepseek_v4.segment", "bank": 1,
                            "shape": v4_shape(32, 0, "local-experts"), "count": 90,
                            "key_id": (32, 0), "hits": 90, "cycles": {"decode": 90},
                            "capture_bytes": 2 << 20,
                            "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                                              "counts": [0, 0, 0, 0, 90, 0]}})
    heat = []
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list(),
                            heat_report=heat)["layouts"]["rtx1-local"]
    entries = {(row["rows"], row["table_width"]): row["pin"] for row in entry["entries"]}
    assert entries[(32, 0)] is False
    assert ("startup", True) in {(row["heat"], row["selected"]) for row in heat}
    # The rare one-use verify geometry stays lazy; the three warmed captures
    # are the base.
    assert entry["reserve_base_bytes"] == [6 << 20]


def test_v4_c1_and_dominant_verify_pin_whatever_their_heat():
    # Plain C1 (rows 1, width 0) and the layout's highest-lifetime-hit verify
    # geometry cost every step, so they pin even when their heat is regular;
    # the other regular verify geometry stays prewarmed and evictable.
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].extend([
        {"device": 0, "site": "deepseek_v4.segment", "bank": 1,
         "shape": v4_shape(1, 0, "local-experts"), "count": 60, "key_id": (1, 0),
         "hits": 59, "cycles": {"decode": 60}, "capture_bytes": 2 << 20,
         "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                           "counts": [1, 0, 0, 57, 0, 0]}},
        {"device": 0, "site": "deepseek_v4.segment", "bank": 1,
         "shape": v4_shape(12, 64), "count": 40, "key_id": (12, 64),
         "hits": 39, "cycles": {"verify": 40}, "capture_bytes": 2 << 20,
         "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                           "counts": [1, 0, 0, 38, 0, 0]}},
    ])
    report = []
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list(),
                            pin_report=report)["layouts"]["rtx1-local"]
    entries = {(row["rows"], row["table_width"]): row["pin"] for row in entry["entries"]}
    assert entries[(1, 0)] is True, "plain C1 pins whatever its heat"
    # (16, 512) has 99 lifetime hits against (12, 64)'s 39.
    assert entries[(16, 512)] is True
    assert entries[(12, 64)] is False
    assert entries[(4, 0)] is False
    reasons = {(row["rows"], row["table_width"]): row["reasons"] for row in report}
    assert reasons[(1, 0)] == ["c1_decode"]
    assert reasons[(16, 512)] == ["dominant_verify"]
    assert (12, 64) not in reasons
    assert entry["reserve_base_bytes"] == [8 << 20]


def test_v4_entries_order_pins_first_then_lifetime_hits():
    # A shrunken reserve warms entries in list order, so the fan-out order puts
    # the pinned set first and then the geometries serving replays most; order
    # never changes the selected set, base or pin flags.
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].extend([
        {"device": 0, "site": "deepseek_v4.segment", "bank": 1,
         "shape": v4_shape(1, 0, "local-experts"), "count": 60, "key_id": (1, 0),
         "hits": 59, "cycles": {"decode": 60}, "capture_bytes": 2 << 20,
         "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                           "counts": [1, 0, 0, 57, 0, 0]}},
        {"device": 0, "site": "deepseek_v4.segment", "bank": 1,
         "shape": v4_shape(32, 0, "local-experts"), "count": 8, "key_id": (32, 0),
         "hits": 7, "cycles": {"decode": 8}, "capture_bytes": 2 << 20,
         "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                           "counts": [0, 0, 0, 7, 0, 0]}},
    ])
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list())["layouts"]["rtx1-local"]
    order = [(row["rows"], row["table_width"]) for row in entry["entries"]]
    # Pins: (16, 512) has 99 lifetime hits, plain C1 59. Unpinned rows follow
    # in hit order: (4, 0) 89, then (32, 0) 7; the one-use (1, 16) stays lazy.
    assert order == [(16, 512), (1, 0), (4, 0), (32, 0)]
    assert [row["pin"] for row in entry["entries"]] == [True, True, False, False]
    assert entry["reserve_base_bytes"] == [8 << 20]


def test_v4_startup_row_captured_once_is_not_warmed_or_pinned():
    # One miss and zero hits is a capture with no replay: warming it would
    # charge the startup arena for a graph the workload never reuses, and
    # pinning it would make that one-off permanent (TJ, 2026-11-09).
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].append({"device": 0, "site": "deepseek_v4.segment", "bank": 1,
                            "shape": v4_shape(50, 0, "local-experts"), "count": 18,
                            "key_id": (50, 0), "hits": 0, "misses": 18,
                            "cycles": {"verify": 18}, "capture_bytes": 3 << 20,
                            "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                                              "counts": [0, 0, 0, 0, 0, 0]}})
    heat = []
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list(),
                            heat_report=heat)["layouts"]["rtx1-local"]
    entries = {(row["rows"], row["table_width"]): row["pin"] for row in entry["entries"]}
    assert (50, 0) not in entries
    # It is a rare leftover, not startup residency, and stays lazy.
    classes = {row["heat"]: row for row in heat}
    assert classes["rare"]["keys"] == 2  # the one-use verify geometry and this row
    assert classes["rare"]["selected"] is False
    assert entry["reserve_base_bytes"] == [4 << 20]


def test_v4_shape_needs_one_trace_per_rank():
    with pytest.raises(ValueError):
        MODULE.generate([v4_trace(0, 5 << 20)], "deepseek_v4", layout="rtx2-spark", existing=v4_list())
    single = MODULE.generate([v4_trace(0, 5 << 20)], "deepseek_v4", layout="rtx1-local", existing=v4_list())
    # The 5 MiB charge is mostly lazy captures; only the warmed set is the base.
    assert single["layouts"]["rtx1-local"]["reserve_base_bytes"] == [4 << 20]


def test_v4_unknown_layout_fails():
    with pytest.raises(ValueError):
        MODULE.generate([v4_trace(0, 5 << 20)], "deepseek_v4", layout="rtx4", existing=v4_list())


def v41_graph(rows=1, *, hits=10, cost=1234, bank=1, key_id=1, site="lane_row", cycles=None):
    return {"site": "deepseek_v41." + site, "shape": str((7, rows)), "hits": hits,
            "capture_bytes": cost, "bank": bank, "key_id": key_id,
            "cycles": {"verify": hits + 1} if cycles is None else cycles}


def v41_trace(device=0, graphs=None, sampled=1):
    return {"schema_version": 2, "cache": {"device": device, "charged": 999999999,
            "sampled_at_ms": sampled}, "graphs": [v41_graph()] if graphs is None else graphs}


def test_v41_replays_select_whole_geometry_but_not_capture_only_geometry():
    hot = v41_graph()
    sibling = v41_graph(hits=0, cost=4000, bank=2)
    cold = v41_graph(rows=16, hits=0, cost=999999, key_id=2)
    report = []
    result = MODULE.generate([v41_trace(graphs=[hot, hot, sibling, cold])], "deepseek_v41", layout=1, pin_report=report)
    rank = result["ranks"][0]
    assert rank["entries"] == [{"rows": 1, "pin": False}]
    assert rank["reserve_base_bytes"] == 2 << 20
    assert rank["reserve_extra_bytes"] == 2 << 20
    assert report[0]["capture_bytes"] == 5234
    assert report[0]["zero_replay_sibling_bytes"] == 4000
    assert report[0]["zero_replay_sibling_identities"] == 1


def test_v41_latest_inventory_and_exact_ids_win_independently_per_rank():
    old = v41_trace(graphs=[v41_graph(cost=99999999)], sampled=1)
    new = v41_trace(graphs=[v41_graph(cost=0), v41_graph(cost=2345, key_id=2)], sampled=2)
    second = v41_trace(device=1, graphs=[v41_graph(cost=5 << 20)])
    for inputs in [[old, new, second], [second, new, old]]:
        result = MODULE.generate(inputs, "deepseek_v41", layout=2)
        assert [rank["reserve_base_bytes"] for rank in result["ranks"]] == [2 << 20, 6 << 20]


@pytest.mark.parametrize("value", [None, True, False, -1, 1.5, "1"])
def test_v41_capture_shares_are_strict_nonnegative_integers(value):
    with pytest.raises(ValueError, match="capture_bytes"):
        MODULE.generate([v41_trace(graphs=[v41_graph(cost=value)])], "deepseek_v41", layout=1)


def test_v41_missing_share_and_missing_replay_count_fail():
    for field in ["capture_bytes", "hits"]:
        graph = v41_graph()
        del graph[field]
        with pytest.raises(ValueError, match=field):
            MODULE.generate([v41_trace(graphs=[graph])], "deepseek_v41", layout=1)


def test_v41_unreachable_full_chunk_is_not_a_pin():
    replay = v41_graph(128, hits=1, cycles={"prefill": 2})
    chunk = v41_graph(2752, hits=0, key_id=2, cycles={"other": 1})
    rank = MODULE.generate([v41_trace(graphs=[replay, chunk])], "deepseek_v41", layout=1, prefill_rows=2752)["ranks"][0]
    assert rank["prefill_entries"] == [{"rows": 128, "pin": True}]
    chunk["cycles"] = {"prefill": 1}
    rank = MODULE.generate([v41_trace(graphs=[replay, chunk])], "deepseek_v41", layout=1, prefill_rows=2752)["ranks"][0]
    assert rank["prefill_entries"] == [{"rows": 128, "pin": True}, {"rows": 2752, "pin": True}]


def test_v41_unknown_layout_or_missing_rank_has_no_fallback():
    for layout in [None, 0, 2, 3]:
        with pytest.raises(ValueError):
            MODULE.generate([v41_trace()], "deepseek_v41", layout=layout)


@pytest.mark.parametrize("width", [5, 7])
def test_v41_flattened_draft_heads_keep_their_owner_and_cost(width):
    graphs = []
    for count in range(1, 17):
        draft = v41_graph(hits=1, key_id=count, site="dspark_chain")
        draft["shape"] = str((count, width))
        head = v41_graph(hits=1000, key_id=count, site="target_head")
        head["shape"] = str(count * width)
        graphs.extend([draft, head])
    rank = MODULE.generate([v41_trace(graphs=graphs)], "deepseek_v41", layout=1, coverage=0.01)["ranks"][0]
    owners = {entry["requests"] * entry["width"] for entry in rank["draft_entries"]}
    assert {count * width for count in range(1, 17) if count * width > 64} <= owners
    head = v41_graph(site="target_head")
    head["shape"] = "112"
    with pytest.raises(ValueError, match="lack an observed"):
        MODULE.generate([v41_trace(graphs=[head])], "deepseek_v41", layout=1)


def test_v41_stale_histogram_cannot_pin_and_current_histogram_can():
    graph = v41_graph()
    graph["first_seen_ms"] = 0
    graph["hit_histogram"] = {"start_ms": 0, "interval_ms": 10_000, "counts": [1] * 5}
    for sampled, pinned in [(40_000, True), (500_000, False)]:
        rank = MODULE.generate([v41_trace(graphs=[graph], sampled=sampled)],
                               "deepseek_v41", layout=1)["ranks"][0]
        assert rank["entries"][0]["pin"] is pinned


def mimo_trace(device=0, charged=640 << 20, largest=12 << 20):
    def graph(layer, rows, count, head):
        return {"device": device, "site": "mimo_v2.segment", "bank": 1,
                "shape": f"GraphKey {{ layer: {layer}, rows: {rows}, table_stride: 128, previous: First, head: {str(head).lower()} }}",
                "count": count, "hits": count, "capture_bytes": (device + 1) * (3 << 20)}
    return {"schema_version": 1, "cache": {"device": device, "charged": charged},
            "sites": {"mimo_v2.segment": {"largest_bytes": largest}},
            "graphs": [graph(0, 1, 100, True), graph(1, 1, 100, False),
                       graph(0, 16, 99, True), graph(0, 64, 1, True)]}



def test_mimo_rows_are_the_only_prewarm_dimension():
    result = MODULE.generate([mimo_trace()], "mimo_v2", coverage=0.99, ranks=1, variant="mimof-48", sparks=True)
    layout = result["layouts"][0]
    assert layout["ranks"] == 1
    assert layout["variant"] == "mimof-48"
    assert layout["sparks"] is True
    assert result["schema_version"] == 2
    assert layout["entries"] == [{"rows": 1, "pin": True}, {"rows": 16, "pin": False}]
    assert layout["reserve_base_bytes"] == [10 << 20]
    assert layout["reserve_extra_bytes"] == [2 << 20]



def test_mimo_layout_carries_one_reserve_per_physical_rank():
    result = MODULE.generate([mimo_trace(0), mimo_trace(1, 578 << 20, 16 << 20)],
                             "mimo_v2", ranks=2, variant="mimop-70", sparks=True)
    layout = result["layouts"][0]
    assert layout["ranks"] == 2
    assert layout["reserve_base_bytes"] == [12 << 20, 24 << 20]
    assert layout["reserve_extra_bytes"] == [2 << 20, 2 << 20]



def test_mimo_selected_arena_includes_cold_siblings_not_rare_tail_or_high_water():
    import copy
    data = mimo_trace()
    sibling = copy.deepcopy(data["graphs"][0])
    sibling.update(key_id=7, cycles={"other": 2}, hits=0, capture_bytes=1 << 20)
    data["graphs"].append(sibling)
    data["graphs"][-2]["capture_bytes"] = 100 << 20
    arenas = []
    result = MODULE.generate([data, data], "mimo_v2", coverage=0.99, ranks=1,
                             variant="mimof-48", sparks=True, arena_report=arenas)
    assert result["layouts"][0]["reserve_base_bytes"] == [10 << 20]
    assert arenas == [{"device": 0, "prewarm_capture_bytes": 10 << 20,
                       "prewarm_base_bytes": 10 << 20, "lazy_capture_bytes": 100 << 20,
                       "lazy_unknown_identities": 0, "zero_replay_keys": 1,
                       "zero_replay_capture_bytes": 1 << 20, "zero_replay_unknown_identities": 0,
                       "selected_zero_hit_siblings": 1, "selected_zero_hit_sibling_bytes": 1 << 20,
                       "always_zero_hit_sibling_classes": [], "device_high_water_bytes": 640 << 20}]



def test_mimo_capture_only_rows_stay_lazy_even_with_heat_or_recaptures():
    data = mimo_trace()
    cold = data["graphs"][-1]
    cold.update(hits=0, misses=1, captures=1, count=10000, cycles={"verify": 10000},
                evictions=3, recaptures=3, first_seen_ms=0,
                hit_histogram={"interval_ms": 10000, "start_ms": 0, "counts": [1, 1, 1, 1, 0]})
    arenas = []
    result = MODULE.generate([data], "mimo_v2", coverage=1, ranks=1,
                             variant="mimof-48", sparks=True, arena_report=arenas)
    assert [entry["rows"] for entry in result["layouts"][0]["entries"]] == [1, 16]
    assert result["layouts"][0]["reserve_base_bytes"] == [10 << 20]
    assert arenas[0]["zero_replay_keys"] == 1
    assert arenas[0]["zero_replay_capture_bytes"] == 3 << 20
    assert arenas[0]["selected_zero_hit_siblings"] == 0
    assert arenas[0]["lazy_capture_bytes"] == 3 << 20



def test_mimo_all_zero_replay_rows_refuse_a_prewarm_proposal():
    data = mimo_trace()
    for graph in data["graphs"]:
        graph.update(hits=0, misses=1, captures=1)
    with pytest.raises(ValueError, match="no matching graph identities"):
        MODULE.generate([data], "mimo_v2", ranks=1, variant="mimof-48", sparks=True)



def test_mimo_consistently_zero_hit_head_siblings_are_flagged_not_omitted():
    import copy
    data = mimo_trace()
    data["graphs"] = [data["graphs"][0], data["graphs"][2]]
    for graph in list(data["graphs"]):
        sibling = copy.deepcopy(graph)
        sibling.update(key_id=graph["count"], hits=0, count=1, cycles={"other": 1}, capture_bytes=1 << 20)
        sibling["shape"] = sibling["shape"].replace("head: true", "head: false")
        data["graphs"].append(sibling)
    arenas = []
    result = MODULE.generate([data], "mimo_v2", ranks=1, variant="mimof-48", sparks=True, arena_report=arenas)
    assert result["layouts"][0]["reserve_base_bytes"] == [8 << 20]
    assert arenas[0]["selected_zero_hit_siblings"] == 2
    assert arenas[0]["selected_zero_hit_sibling_bytes"] == 2 << 20
    assert len(arenas[0]["always_zero_hit_sibling_classes"]) == 1
    assert "head: false" in arenas[0]["always_zero_hit_sibling_classes"][0]["shape"]



def test_mimo_latest_inventory_replaces_capture_shares_in_either_input_order():
    import copy
    old = mimo_trace()
    old["cache"]["sampled_at_ms"] = 1
    new = copy.deepcopy(old)
    new["cache"]["sampled_at_ms"] = 2
    for graph in new["graphs"]:
        graph["capture_bytes"] = 1 << 20
    for traces in ([old, new, new], [new, old]):
        result = MODULE.generate(traces, "mimo_v2", ranks=1, variant="mimof-48", sparks=True)
        assert result["layouts"][0]["reserve_base_bytes"] == [4 << 20]



@pytest.mark.parametrize("share", [None, -1, True, "1024"])
def test_mimo_selected_identity_requires_measured_nonnegative_capture_share(share):
    data = mimo_trace()
    data["graphs"][0]["capture_bytes"] = share
    with pytest.raises(ValueError, match="lacks measured capture_bytes"):
        MODULE.generate([data], "mimo_v2", ranks=1, variant="mimof-48", sparks=True)



def test_mimo_zero_capture_share_is_valid_allocator_reuse():
    data = mimo_trace()
    data["graphs"][0]["capture_bytes"] = 0
    result = MODULE.generate([data], "mimo_v2", ranks=1, variant="mimof-48", sparks=True)
    assert result["layouts"][0]["reserve_base_bytes"] == [10 << 20]



def test_mimo_forgotten_selected_sibling_refuses_incomplete_arena():
    import copy
    old = mimo_trace()
    old["cache"]["sampled_at_ms"] = 1
    new = copy.deepcopy(old)
    new["cache"]["sampled_at_ms"] = 2
    new["graphs"] = new["graphs"][1:]
    with pytest.raises(ValueError, match="forgotten"):
        MODULE.generate([old, new], "mimo_v2", ranks=1, variant="mimof-48", sparks=True)



def test_mimo_missing_selected_geometry_on_a_peer_fails():
    peer = mimo_trace(1)
    peer["graphs"] = peer["graphs"][:2]
    with pytest.raises(ValueError, match="lacks selected MiMo geometries"):
        MODULE.generate([mimo_trace(), peer], "mimo_v2", ranks=2, variant="mimof-48", sparks=True)



def test_mimo_sustained_verify_rows_outside_startup_join_prewarm_and_pin():
    data = mimo_trace()
    data["schema_version"] = 2
    for graph in data["graphs"]:
        graph["cycles"] = {"verify": graph["count"]}
    data["graphs"][-1].update(cycles={"verify": 1}, first_seen_ms=0,
        hit_histogram={"interval_ms": 10000, "start_ms": 0, "counts": [1, 1, 1, 1, 0]})
    result = MODULE.generate([data], "mimo_v2", coverage=0.99, ranks=1,
                             variant="mimof-48", sparks=True)
    assert {"rows": 64, "pin": True} in result["layouts"][0]["entries"]
    assert result["layouts"][0]["reserve_base_bytes"] == [12 << 20]



def test_mimo_lifetime_pins_c1_dominant_verify_and_half_hit_tier():
    import copy
    data = mimo_trace()
    template = data["graphs"][0]
    data["schema_version"] = 2
    data["graphs"] = []
    for rows, hits, cycle in [(1, 10, "decode"), (2, 40, "verify"),
                              (4, 30, "verify"), (8, 20, "verify")]:
        graph = copy.deepcopy(template)
        graph.update(shape=template["shape"].replace("rows: 1,", f"rows: {rows},"),
                     hits=hits, count=hits, cycles={cycle: hits},
                     hit_histogram={"interval_ms": 10000, "counts": [0, 0, 0, 0]})
        data["graphs"].append(graph)
    report = []
    result = MODULE.generate([data], "mimo_v2", ranks=1, variant="mimof-48",
                             sparks=True, pin_report=report)
    assert {entry["rows"] for entry in result["layouts"][0]["entries"] if entry["pin"]} == {1, 2, 4}
    reasons = {entry["rows"]: entry["reasons"] for entry in report}
    assert "plain_c1" in reasons[1]
    assert "dominant_verify" in reasons[2]
    assert "lifetime_hit_tier" in reasons[2]
    assert "lifetime_hit_tier" in reasons[4]
    assert all(entry["capture_bytes"] == 3 << 20 for entry in report)



def test_mimo_lifetime_pins_keep_cutoff_ties_and_ignore_startup_hits():
    import copy
    data = mimo_trace()
    data["schema_version"] = 2
    for graph in data["graphs"]:
        graph.update(hits=20, cycles={"verify": 20})
    cold = copy.deepcopy(data["graphs"][-1])
    cold["shape"] = cold["shape"].replace("rows: 64,", "rows: 32,")
    cold.update(hits=10000, cycles={"other": 10000})
    data["graphs"].append(cold)
    result = MODULE.generate([data], "mimo_v2", ranks=1, variant="mimof-48", sparks=True)
    assert {entry["rows"] for entry in result["layouts"][0]["entries"] if entry["pin"]} == {1}
    # The two C1 layer siblings aggregate to 40 hits, exactly half the total.
    data["graphs"][0]["hits"] = 10
    data["graphs"][0]["cycles"] = {"verify": 10}
    result = MODULE.generate([data], "mimo_v2", ranks=1, variant="mimof-48", sparks=True)
    assert {entry["rows"] for entry in result["layouts"][0]["entries"] if entry["pin"]} == {1, 16, 64}
    assert 32 not in {entry["rows"] for entry in result["layouts"][0]["entries"]}



def test_mimo_lifetime_pins_use_each_device_latest_inventory():
    import copy
    old = mimo_trace()
    old["cache"]["sampled_at_ms"] = 1
    new = copy.deepcopy(old)
    new["cache"]["sampled_at_ms"] = 2
    peer = mimo_trace(1)
    for data, dominant in [(old, 64), (new, 16), (peer, 64)]:
        data["schema_version"] = 2
        for graph in data["graphs"]:
            rows = MODULE.key_of("mimo_v2", graph["shape"])[0]
            hits = 1000 if rows == dominant else 1
            graph.update(hits=hits, cycles={"verify": hits})
    result = MODULE.generate([old, new, peer], "mimo_v2", ranks=2, variant="mimop-70", sparks=True)
    assert {entry["rows"] for entry in result["layouts"][0]["entries"] if entry["pin"]} == {1, 16, 64}



def test_mimo_rank_count_must_match_the_trace():
    with pytest.raises(ValueError):
        MODULE.generate([mimo_trace(0)], "mimo_v2", ranks=2, variant="mimop-70", sparks=True)
    with pytest.raises(ValueError):
        MODULE.generate([mimo_trace(0)], "mimo_v2")



def test_mimo_rejects_rows_outside_the_decode_geometry():
    data = mimo_trace()
    data["graphs"][0]["shape"] = "GraphKey { layer: 0, rows: 65, table_stride: 128, previous: First, head: true }"
    with pytest.raises(ValueError):
        MODULE.generate([data], "mimo_v2", ranks=1, variant="mimof-48", sparks=True)



def test_mimo_checked_in_row_shares_reproduce_bases_and_tail():
    import json
    checked_in = json.loads((SCRIPT.parents[2] /
        "rust/crates/cuteafd-loader/src/serving_capacity/mimo_prewarm.json").read_text())
    assert {(layout["variant"], layout["ranks"], layout["sparks"])
            for layout in checked_in["layouts"]} == {("mimof-48", 1, True), ("mimop-70", 2, True)}
    for layout in checked_in["layouts"]:
        traces = []
        for device, shares in enumerate(layout["entry_capture_bytes"]):
            graphs = []
            for entry, share in zip(layout["entries"], shares, strict=True):
                graphs.append({"site": "mimo_v2.segment", "bank": 1,
                    "shape": f"GraphKey {{ layer: 0, rows: {entry['rows']}, head: true }}",
                    "hits": 1, "count": 1, "cycles": {"verify": 1}, "capture_bytes": share})
            traces.append({"schema_version": 2, "cache": {"device": device, "charged": sum(shares)},
                           "graphs": graphs})
        regenerated = MODULE.generate(traces, "mimo_v2", coverage=1, ranks=layout["ranks"],
            variant=layout["variant"], sparks=layout["sparks"])["layouts"][0]
        for key in ("reserve_base_bytes", "reserve_extra_bytes"):
            assert regenerated[key] == layout[key]
        for original, rebuilt in zip(layout["entry_capture_bytes"], regenerated["entry_capture_bytes"], strict=True):
            assert dict(zip((entry["rows"] for entry in layout["entries"]), original, strict=True)) == \
                dict(zip((entry["rows"] for entry in regenerated["entries"]), rebuilt, strict=True))
        assert 1 in {entry["rows"] for entry in layout["entries"] if entry["pin"]}



def test_mimo_order_is_pinned_then_latest_request_replay_hits_with_row_ties():
    import copy
    old = mimo_trace()
    old["cache"]["sampled_at_ms"] = 1
    latest = copy.deepcopy(old)
    latest["cache"]["sampled_at_ms"] = 2
    template = latest["graphs"][0]
    latest["schema_version"] = 2
    latest["graphs"] = []
    for rows, hits, share in [(1, 1, 1), (2, 100, 2), (8, 20, 3), (4, 20, 4), (16, 10, 5)]:
        graph = copy.deepcopy(template)
        graph.update(shape=template["shape"].replace("rows: 1,", f"rows: {rows},"),
            hits=hits + 10000, count=hits + 10000,
            cycles={"verify": hits, "other": 10000}, capture_bytes=share << 20)
        latest["graphs"].append(graph)
    # Obsolete lifetime heat and startup Other must not change ordering.
    old = copy.deepcopy(latest)
    old["cache"]["sampled_at_ms"] = 1
    old["graphs"][-1].update(hits=100000, count=100000, cycles={"verify": 100000})
    for traces in ([old, latest], [latest, old]):
        layout = MODULE.generate(traces, "mimo_v2", coverage=1, ranks=1,
            variant="mimof-48", sparks=True)["layouts"][0]
        assert [(entry["rows"], entry["pin"]) for entry in layout["entries"]] == \
            [(2, True), (1, True), (4, False), (8, False), (16, False)]
        assert layout["entry_capture_bytes"] == [[value << 20 for value in [2, 1, 4, 3, 5]]]

def kimi_trace(device=0, charged=600 << 20, rows=1, speculative=False, cycles=None):
    shape = (f"lane=0 rank={device} rows={rows} op="
             + (f"attention layer=0 spec={int(speculative)}" if speculative
                else "ffn layer=0"))
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
                                shape="lane=0 rank=0 rows=4096 op=ffn layer=0",
                                cycles={"prefill": 50}))
    row = MODULE.generate_kimi([trace], 1, "kimi-k3-exl3-k6-min", coverage=0.999, core="k6")
    assert row["entries"] == [{"rows": 1, "speculative": False, "op": "ffn", "pin": True, "capture_bytes": [3 << 20]}]
    assert row["reserve_base_bytes"] == [4 << 20]
    assert row["reserve_extra_bytes"] == 2 << 20


@pytest.mark.parametrize("core", ["bf16", "k6"])
def test_kimi_cli_merges_core_key_and_prints_summary(tmp_path, core):
    import subprocess
    import sys
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(kimi_trace()))
    output = tmp_path / "kimi_prewarm.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--family", "kimi_k3",
        "--variant", core, "--ranks", "1", "--layout", "cli", "--coverage", "1",
        "--trace", str(trace), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "1 shapes; layout=cli;" in result.stdout
    assert json.loads(output.read_text())["layouts"][0]["core"] == core


def test_kimi_verify_width_is_its_own_pin(tmp_path):
    trace = kimi_trace(rows=4, speculative=True)
    row = MODULE.generate_kimi([trace, kimi_trace(device=1, rows=4, speculative=True)], 2, "kimi-k3-exl3-k6-max", core="k6")
    assert row["entries"] == [{"rows": 4, "speculative": True, "op": "attention", "pin": True, "capture_bytes": [3 << 20, 3 << 20]}]
    out = tmp_path / "kimi_prewarm.json"
    MODULE.merge_kimi_layout(out, MODULE.generate_kimi([kimi_trace()], 1, "kimi-k3-exl3-k6-min", core="k6"))
    MODULE.merge_kimi_layout(out, row)
    layouts = json.loads(out.read_text())["layouts"]
    assert [layout["ranks"] for layout in layouts] == [1, 2]
    assert layouts[1]["layout"] == "kimi-k3-exl3-k6-max"


def test_kimi_startup_only_geometry_is_neither_selected_nor_pinned():
    trace = kimi_trace()
    trace["graphs"].append(dict(trace["graphs"][0],
        shape="lane=0 rank=0 rows=16 op=attention layer=0 spec=0",
        cycles={"other": 1000}, count=1000, capture_bytes=None,
        evictions=3, recaptures=3))
    row = MODULE.generate_kimi([trace], 1, "min", core="k6")
    assert row["entries"] == [
        {"rows": 1, "speculative": False, "op": "ffn", "pin": True, "capture_bytes": [3 << 20]}]
    assert row["reserve_base_bytes"] == [4 << 20]


def test_kimi_cold_shape_outside_lifetime_tier_is_listed_but_unpinned():
    trace = kimi_trace()
    trace["graphs"].append(dict(trace["graphs"][0],
        shape="lane=0 rank=0 rows=16 op=ffn layer=0", cycles={"decode": 1}, hits=1,
        hit_histogram={"interval_ms": 10_000, "start_ms": 0,
                       "counts": [1, 0, 0, 0, 0]}))
    row = MODULE.generate_kimi([trace], 1, "kimi-k3-exl3-k6-min", coverage=1, core="k6")
    assert row["entries"] == [
        {"rows": 1, "speculative": False, "op": "ffn", "pin": True, "capture_bytes": [3 << 20]},
        {"rows": 16, "speculative": False, "op": "ffn", "pin": False, "capture_bytes": [3 << 20]}]


@pytest.mark.parametrize("shape", ["bad", "lane=0 rank=0 rows=0 op=ffn layer=0"])
def test_kimi_invalid_shape_fails(shape):
    trace = kimi_trace()
    trace["graphs"][0]["shape"] = shape
    with pytest.raises(ValueError):
        MODULE.generate_kimi([trace], 1, "kimi-k3-exl3-k6-min", core="k6")


def test_kimi_unreachable_rows_are_not_proposed_even_with_decode_counts():
    trace = kimi_trace()
    trace["graphs"].append(dict(trace["graphs"][0],
                                shape="lane=0 rank=0 rows=4096 op=ffn layer=0"))
    assert MODULE.generate_kimi([trace], 1, "min", core="k6")["entries"] == [
        {"rows": 1, "speculative": False, "op": "ffn", "pin": True, "capture_bytes": [3 << 20]}]


def test_kimi_row_only_attention_trace_has_no_sequence_dimension():
    data = kimi_trace(rows=20, speculative=True)
    row = MODULE.generate_kimi([data], 1, "min", core="k6")
    assert row["entries"] == [{"rows": 20, "speculative": True, "op": "attention",
                               "pin": True, "capture_bytes": [3 << 20]}]


def test_kimi_geometry_preserves_operation_and_verify_variant():
    data = kimi_trace(rows=20, speculative=True)
    data["graphs"].append(dict(data["graphs"][0],
        shape="lane=0 rank=0 rows=20 op=context layer=1 spec=1"))
    row = MODULE.generate_kimi([data], 1, "max", core="k6")
    assert row["entries"] == [
        {"rows": 20, "speculative": True, "op": "attention", "pin": True, "capture_bytes": [3 << 20]},
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
        row = MODULE.generate_kimi(inputs, 2, "max", core="k6")
        assert row["reserve_base_bytes"] == [2 << 20, 4 << 20]
        assert row["entries"][0]["capture_bytes"] == [3, (2 << 20) + 1]


def test_kimi_bases_follow_logical_rank_not_cuda_ordinal():
    rank0 = kimi_trace(device=7)
    rank0["graphs"][0].update(shape="lane=0 rank=0 rows=1 op=ffn layer=0",
                               capture_bytes=1)
    rank1 = kimi_trace(device=3)
    rank1["graphs"][0].update(shape="lane=0 rank=1 rows=1 op=ffn layer=0",
                               capture_bytes=(2 << 20) + 1)
    row = MODULE.generate_kimi([rank1, rank0], 2, "max", core="k6")
    assert row["reserve_base_bytes"] == [2 << 20, 4 << 20]
    assert row["entries"][0]["capture_bytes"] == [1, (2 << 20) + 1]
    rank1["graphs"][0]["shape"] = rank0["graphs"][0]["shape"]
    with pytest.raises(ValueError, match="one-to-one"):
        MODULE.generate_kimi([rank0, rank1], 2, "max", core="k6")


@pytest.mark.parametrize("share", [None, -1, "0", True])
def test_kimi_missing_or_invalid_selected_capture_share_fails(share):
    data = kimi_trace()
    data["graphs"][0]["capture_bytes"] = share
    with pytest.raises(ValueError, match="capture_bytes"):
        MODULE.generate_kimi([data], 1, "min", core="k6")
    del data["graphs"][0]["capture_bytes"]
    with pytest.raises(ValueError, match="capture_bytes"):
        MODULE.generate_kimi([data], 1, "min", core="k6")


def test_kimi_lifetime_hot_tier_survives_aged_out_rolling_window():
    data = kimi_trace()
    data["cache"]["sampled_at_ms"] = 900_000
    seed = data["graphs"][0]
    data["graphs"] = [dict(seed, key_id=index,
        shape=f"lane=0 rank=0 rows={rows} op=ffn layer=0",
        hits=hits, count=hits, cycles={"decode": hits},
        hit_histogram={"interval_ms": 10_000, "start_ms": 310_000, "counts": [0] * 60})
        for index, (rows, hits) in enumerate([(1, 1), (4, 60), (8, 25), (16, 14)])]
    report = []
    row = MODULE.generate_kimi([data], 1, "min", coverage=1, pin_report=report, core="k6")
    assert [entry["rows"] for entry in row["entries"] if entry["pin"]] == [4, 1]
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
    row = MODULE.generate_kimi([data], 1, "min", coverage=1, core="k6")
    assert [entry["rows"] for entry in row["entries"] if entry["pin"]] == [4, 8, 16]


def test_kimi_prewarm_orders_pins_then_lifetime_replay_heat_not_requests():
    data = kimi_trace()
    seed = data["graphs"][0]
    data["graphs"] = [dict(seed, key_id=index,
        shape=f"lane=0 rank=0 rows={rows} op=ffn layer=0",
        hits=hits, cycles={"verify": requests}, capture_bytes=rows,
        hit_histogram={"interval_ms": 10000, "start_ms": 0, "counts": [0] * 5})
        for index, (rows, hits, requests) in enumerate([
            (1, 1, 1), (4, 100, 100), (8, 2, 1000), (16, 20, 20), (32, 20, 20)])]
    row = MODULE.generate_kimi([data], 1, "min", coverage=1, core="k6")
    assert [(entry["rows"], entry["pin"]) for entry in row["entries"]] == [
        (4, True), (1, True), (16, False), (32, False), (8, False)]
    assert [entry["capture_bytes"] for entry in row["entries"]] == [[4], [1], [16], [32], [8]]
    data["graphs"].reverse()
    assert MODULE.generate_kimi([data], 1, "min", coverage=1, core="k6") == row


def test_kimi_dominant_verify_pins_attention_and_flagless_siblings():
    data = kimi_trace()
    seed = data["graphs"][0]
    shapes = [
        ("rows=20 op=attention layer=0 spec=1", 100),
        ("rows=24 op=attention layer=0 spec=1", 10),
        ("rows=20 op=context layer=1 spec=1", 100),
        ("rows=20 op=ffn layer=1", 1),
        ("rows=1 op=ffn layer=0", 1),
        ("rows=4 op=ffn layer=2", 100),
    ]
    data["graphs"] = [dict(seed, key_id=index, shape="lane=0 rank=0 " + shape,
        hits=count, cycles={"decode" if "rows=1 " in shape else "verify": count},
        hit_histogram={"interval_ms": 10_000, "start_ms": 0, "counts": [0] * 5})
        for index, (shape, count) in enumerate(shapes)]
    report = []
    MODULE.generate_kimi([data], 1, "min", coverage=1, pin_report=report, core="k6")
    dominant = [entry for entry in report if "dominant_verify" in entry["reasons"]]
    assert {entry["op"] for entry in dominant} == {"attention", "context", "ffn"}
    assert all(entry["rows"] == 20 for entry in dominant)
    assert any(entry["rows"] == 1 and "plain_c1" in entry["reasons"] for entry in report)


def test_kimi_lifetime_tiers_are_independent_per_device_and_operation():
    traces = [kimi_trace(device=rank) for rank in (0, 1)]
    for rank, data in enumerate(traces):
        seed = data["graphs"][0]
        data["graphs"] = [dict(seed, key_id=index,
            shape=f"lane=0 rank={rank} rows={rows} op={op} layer=0" +
                  (" spec=1" if op == "context" else ""),
            hits=hits, cycles={"verify": hits},
            hit_histogram={"interval_ms": 10_000, "start_ms": 0, "counts": [0] * 5})
            for index, (rows, op, hits) in enumerate([
                (4, "ffn", 100 if rank == 0 else 1),
                (8, "ffn", 1 if rank == 0 else 100),
                (16, "context", 2)])]
    row = MODULE.generate_kimi(traces, 2, "max", coverage=1, core="k6")
    assert {(entry["rows"], entry["op"]) for entry in row["entries"] if entry["pin"]} == {
        (4, "ffn"), (8, "ffn"), (16, "context")}


def test_kimi_zero_replay_geometry_stays_lazy_even_with_hot_request_counts():
    data = kimi_trace()
    seed = data["graphs"][0]
    data["graphs"].append(dict(seed, key_id=2,
        shape="lane=0 rank=0 rows=4 op=attention layer=0 spec=1",
        hits=0, cycles={"verify": 10000}, capture_bytes=None,
        evictions=10, recaptures=10))
    row = MODULE.generate_kimi([data], 1, "min", coverage=1, core="k6")
    assert len(row["entries"]) == 1
    assert row["entries"][0]["rows"] == 1
    data["graphs"][0]["hits"] = 0
    with pytest.raises(ValueError, match="no replayed"):
        MODULE.generate_kimi([data], 1, "min", core="k6")


def test_kimi_one_replay_selects_whole_geometry_and_charges_zero_hit_sibling():
    data = kimi_trace()
    graph = data["graphs"][0]
    graph.update(hits=1, cycles={"decode": 2}, capture_bytes=3)
    data["graphs"].append(dict(graph, bank=2, hits=0, capture_bytes=7))
    report = []
    row = MODULE.generate_kimi([data], 1, "min", pin_report=report, core="k6")
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
        row = MODULE.generate_kimi(traces, 1, "min", core="k6")
        assert [entry["rows"] for entry in row["entries"]] == [1]


def test_kimi_entry_costs_sum_to_each_once_rounded_base():
    traces = [kimi_trace(device=rank) for rank in (0, 1)]
    for rank, data in enumerate(traces):
        seed = data["graphs"][0]
        data["graphs"] = [dict(seed, key_id=index,
            shape=f"lane=0 rank={rank} rows={rows} op=ffn layer=0",
            hits=hits, cycles={"decode": hits}, capture_bytes=cost)
            for index, (rows, hits, cost) in enumerate([(1, 100, 0), (4, 10, 13 + rank)])]
    row = MODULE.generate_kimi(traces, 2, "max", coverage=1, core="k6")
    assert [entry["capture_bytes"] for entry in row["entries"]] == [[0, 0], [13, 14]]
    assert row["reserve_base_bytes"] == [MODULE.packed(sum(entry["capture_bytes"][rank]
        for entry in row["entries"])) for rank in (0, 1)]
    traces[1]["graphs"] = traces[1]["graphs"][:1]
    row = MODULE.generate_kimi(traces, 2, "max", coverage=1, core="k6")
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
            data["cache"]["sampled_at_ms"] = device["sampled_at_ms"]
            for index, group in enumerate(device["groups"]):
                rows, spec, op, hits, cycles, histogram_index, first_bin, identities, cost = group
                histogram = device["histograms"][histogram_index]
                shape = f"lane=0 rank={rank} rows={rows} op={op} layer=0"
                if op != "ffn":
                    shape += f" spec={int(spec)}"
                share, tail = divmod(cost, identities)
                # Exact histogram/cycle classes preserve selection and pin heat;
                # summed raw shares preserve the once-rounded physical base.
                first_seen = (0 if first_bin == 0 else
                    histogram["start_ms"] + first_bin * histogram["interval_ms"])
                for sibling in range(identities):
                    data["graphs"].append(dict(seed, key_id=f"{index}-{sibling}", shape=shape,
                        hits=hits, cycles=cycles, first_seen_ms=first_seen,
                        capture_bytes=share + (sibling < tail), hit_histogram=histogram))
            traces.append(data)
        core = layout.get("core", "k6")
        result = MODULE.generate_kimi(traces, layout["ranks"], layout["layout"], coverage=fixture["coverage"], core=core)
        expected = next(row for row in checked["layouts"]
                        if row["core"] == core and row["ranks"] == layout["ranks"])
        assert result == expected
        for rank in range(layout["ranks"]):
            assert MODULE.packed(sum(entry["capture_bytes"][rank] or 0
                for entry in result["entries"])) == result["reserve_base_bytes"][rank]
        assert result["reserve_extra_bytes"] == MODULE.CHUNK


def test_kimi_cleaned_fixture_prewarms_replayed_late_widths_without_pinning():
    checked = json.loads((SCRIPT.parents[2] /
        "rust/crates/cuteafd-loader/src/serving_capacity/kimi_prewarm.json").read_text())
    minimum, maximum = [row for row in checked["layouts"] if row["core"] == "k6"]
    entries = {(entry["rows"], entry["op"], entry["speculative"]): entry
               for entry in minimum["entries"]}
    for rows, op in [(50, "attention"), (56, "attention"),
                     (55, "ffn"), (56, "ffn"), (58, "ffn")]:
        assert not entries[(rows, op, op == "attention")]["pin"]
    for rows in [52, 55, 58]:
        assert (rows, "attention", True) not in entries
    assert not any(entry["rows"] == 62 and entry["op"] in ("attention", "ffn")
                   for entry in maximum["entries"])
    assert all(entry["op"] != "residual" and "sequences" not in entry
               for layout in checked["layouts"] for entry in layout["entries"])


def test_kimi_zero_share_is_valid_and_missing_rank_fails():
    data = kimi_trace()
    data["graphs"][0]["capture_bytes"] = 0
    row = MODULE.generate_kimi([data], 1, "min", core="k6")
    assert row["reserve_base_bytes"] == [0]
    assert row["entries"][0]["capture_bytes"] == [0]
    with pytest.raises(ValueError, match="missing a physical rank"):
        MODULE.generate_kimi([data], 2, "max", core="k6")


def test_dense_qwen_replays_select_geometry_siblings_and_order_by_heat():
    def graph(segment, rows, stride, spec, hits, cost=1024):
        return {"site": "qwen3_5.segment", "bank": 1, "key_id": segment * 100 + rows,
                "shape": f"Key {{ segment: {segment}, rows: {rows}, table_stride: {stride}, spec: {str(spec).lower()} }}",
                "hits": hits, "nodes": 8, "capture_bytes": cost}
    data = {"schema_version": 2, "cache": {"device": 1}, "graphs": [
        graph(0, 1, 16, False, 10), graph(1, 1, 16, False, 0),
        graph(0, 4, 32, True, 100), graph(1, 4, 32, True, 50),
        graph(0, 64, 128, True, 0, 10 << 20)]}
    summary = {}
    result = MODULE.generate_qwen35([data], "h5120-l64-fp8-kvfp8-rtx1", summary)
    assert summary["zero_replay_identities"] == 1
    assert summary["zero_replay_bytes"] == 10 << 20
    assert summary["selected_zero_hit_siblings"] == 1
    assert summary["selected_zero_hit_sibling_bytes"] == 1024
    assert summary["always_zero_hit_segments"] == []
    assert summary["zero_share_identities"] == 0
    layout = result["layouts"][0]
    assert layout["entries"] == [
        {"rows": 4, "table_stride": 32, "spec": True, "pin": False},
        {"rows": 1, "table_stride": 16, "spec": False, "pin": False}]
    assert layout["reserve_base_bytes"] == 2 << 20
    assert layout["reserve_extra_bytes"] == 2 << 20
    # Every selected sibling is charged, including the zero-hit sibling.
    data["graphs"][1]["capture_bytes"] = 3 << 20
    assert MODULE.generate_qwen35([data], layout["variant"])["layouts"][0]["reserve_base_bytes"] == 4 << 20
    cold = graph(0, 16, 64, True, 0)
    cold.update(nodes=None, capture_bytes=None, captures=0)
    data["graphs"].append(cold)
    assert MODULE.generate_qwen35([data], layout["variant"]) == MODULE.generate_qwen35(
        [{**data, "graphs": data["graphs"][:-1]}], layout["variant"])
    data["graphs"][0]["nodes"] = 1
    with pytest.raises(ValueError, match="singleton"):
        MODULE.generate_qwen35([data], layout["variant"])


def test_step5_launcher_defaults_graphs_on_for_plan_and_serve():
    launcher = (SCRIPT.parents[2] / "scripts/launch/run-family.sh").read_text()
    assert 'step5_decode_graphs="$(get DECODE_GRAPHS on)"' in launcher
    assert '--step5-decode-graphs "$step5_decode_graphs"' in launcher
    step5 = launcher.split('step5_decode_graphs="$(get DECODE_GRAPHS on)"', 1)[1].split("esac", 1)[0]
    assert 'on) family_args+=(--decode-graphs true)' in step5
    assert 'off) family_args+=(--decode-graphs false)' in step5


def test_dense_qwen_launcher_defaults_graphs_on_for_plan_and_serve():
    launcher = (SCRIPT.parents[2] / "scripts/launch/run-family.sh").read_text()
    for section in ("shared_graph_census_args()", "plan_profile_env=()"):
        dense = launcher.split(section, 1)[1].split('if [[ "$family" == qwen3_5 ]]; then', 1)[1].split("esac", 1)[0]
        assert 'case "$(get DECODE_GRAPHS on)" in' in dense
        assert "CUTEAFD_QWEN35_GRAPHS=1" in dense
        assert "CUTEAFD_QWEN35_GRAPHS=0" in dense


def test_dense_qwen_engine_graph_lifetime_reconciliation_contract():
    root = SCRIPT.parents[2]
    engine = (root / "rust/crates/cuteafd-daemon/src/families/qwen3_5/engine.rs").read_text()
    graph = (root / "rust/crates/cuteafd-daemon/src/families/qwen3_5/graphs.rs").read_text()
    scheduler = (root / "rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs").read_text()
    assert engine.index("stream: Stream") < engine.index("graphs: graphs::Graphs") < engine.index("weights: DenseWeights")
    assert '"memory_reconciliation_provenance"' in engine
    assert '"devices": [{"device": device.id}]' in engine
    assert 'capture("first_request")' in scheduler
    assert "nodes < 2" in graph
    assert "spans_devices: false" in graph
    assert graph.count("device::observe_completed(") == 1
    assert "cache_ticket" in graph and "cache_commit" in graph
    assert "GraphOwner" in graph and "struct Executable" not in graph
    assert "golden_decode_graphs" in graph
    assert "off_digest == on_digest" in graph
    assert "off_digest == warm_digest" in graph
    assert graph.index("let (_, warm_digest) = score(true)?") < graph.index("let before = replay_hits()")
    assert graph.index("let before = replay_hits()") < graph.index("let (on, on_digest) = score(true)?")
    assert "after > before" in graph
    golden = (root / "rust/crates/cuteafd-daemon/src/families/qwen3_5/mod.rs").read_text()
    assert "published_reference(golden, cfg.hidden, cfg.vocab)?" in golden
    assert 'let consumer = crate::shared::probe::CallbackRows::new(' in golden
    assert 'reference.row(if reference_rows == tokens.len() { position } else { index })?' in golden
    assert 'saved.write_all(&value.to_le_bytes())?' in golden
    assert "external top-1024 quality is qualified separately" in golden
    verify = (root / "rust/crates/cuteafd-daemon/src/families/qwen3_5/verify.rs").read_text()
    assert "if self.graphs.enabled && plan.stride > 0" in verify
    admission = (root / "rust/crates/cuteafd-loader/src/placement/families/qwen3_5/admission.rs").read_text()
    assert 'id: "rtx0".into()' in admission
    assert "graph_capture_stats" not in graph


def test_dense_qwen_trimmed_trace_reproduces_checked_in_bytes():
    import gzip

    fixture = SCRIPT.parents[1] / "tests/fixtures/qwen35-prewarm-trace.json.gz"
    trace = json.loads(gzip.decompress(fixture.read_bytes()))
    summary = {}
    generated = MODULE.generate_qwen35([trace], "h5120-l64-bf16-kvfp8-rtx1", summary)
    installed = SCRIPT.parents[2] / "rust/crates/cuteafd-loader/src/serving_capacity/qwen35_prewarm.json"
    assert json.dumps(generated, indent=2) + "\n" == installed.read_text()
    layout = generated["layouts"][0]
    assert layout["traced"]
    assert layout["reserve_extra_bytes"] == MODULE.CHUNK
    assert len(layout["entries"]) == 67
    assert not any(entry["pin"] for entry in layout["entries"])
    assert layout["reserve_base_bytes"] == 599_785_472
    assert summary["zero_replay_identities"] == 896
    assert summary["always_zero_hit_segments"] == []
    assert {int(MODULE.DENSE_QWEN_GEOMETRY.fullmatch(graph["shape"])[1])
            for graph in trace["graphs"]} == set(range(64))
    assert all(graph["nodes"] == 13 for graph in trace["graphs"])
    assert fixture.stat().st_size < 1_000_000


def test_dense_qwen_diagnostic_verify_bypasses_capture_and_replay():
    root = SCRIPT.parents[2] / "rust/crates/cuteafd-daemon/src/families/qwen3_5"
    graph = (root / "graphs.rs").read_text()
    serve = (root / "serve.rs").read_text()
    golden = (root / "mod.rs").read_text()
    diagnostic = serve.split("    fn verify_device_ungraphed(", 1)[1].split("    fn verify_bucket_rows(", 1)[0]
    assert diagnostic.index("let _bypass = self.engine.borrow().bypass_graphs()") < diagnostic.index("self.verify_device(")
    assert "impl Drop for GraphBypass" in graph
    assert "self.ready.set(self.previous)" in graph
    assert "Rc<Cell<bool>>" in graph
    assert "diagnostic_bypass_restores_on_error_nested_calls_and_unwind" in graph
    assert "if !self.graphs.enabled || !self.graphs.capture_ready.get() { return segment(); }" in graph
    assert "ensure!(counters() == before" in serve
    assert "(true, scheduler::UngraphedKind::Decode)" in serve
    assert "(false, scheduler::UngraphedKind::Scoring)" in serve
    assert "check_diagnostic_graph_bypass(&tokens)?" in golden



def muse_trace(device=0):
    def graph(rows, layer, hits, cost):
        return {"site": "muse_spark.segment", "bank": 1, "key_id": f"{rows}-{layer}",
                "shape": f"Key {{ layer: {layer}, rows: {rows}, stride: 2048, section: Attention }}",
                "hits": hits, "count": hits + 1, "misses": 1, "capture_bytes": cost}
    return {"schema_version": 2, "cache": {"device": device, "sampled_at_ms": 10},
            "graphs": [graph(1, 0, 20, 100), graph(1, 1, 0, 50),
                       graph(4, 0, 10, 200), graph(8, 0, 0, 300)]}


def test_muse_prices_capture_only_siblings_but_not_zero_replay_geometries():
    result = MODULE.generate_muse([muse_trace()], 1, 2)
    assert result["reserve_base_bytes"] == [2 << 20]
    assert result["reserve_extra_bytes"] == [2 << 20]
    assert result["entries"] == [
        {"rows": 1, "pin": False, "capture_bytes": [150]},
        {"rows": 4, "pin": False, "capture_bytes": [200]}]
    assert "1 zero-replay siblings included" in result["provenance"]
    assert "1 zero-replay identities stay lazy" in result["provenance"]
    assert result == MODULE.generate_muse([muse_trace(), muse_trace()], 1, 2)


def test_muse_latest_snapshot_and_physical_rank_costs_are_not_borrowed():
    old = muse_trace()
    old["cache"]["sampled_at_ms"] = 1
    old["graphs"][0]["capture_bytes"] = 9999999
    other = muse_trace(1)
    other["graphs"][0]["capture_bytes"] = 300
    result = MODULE.generate_muse([old, muse_trace(), other], 2, 2)
    assert result["entries"][0]["capture_bytes"] == [150, 350]
    assert result == MODULE.generate_muse([other, muse_trace(), old], 2, 2)
    with pytest.raises(ValueError, match="needs 2 device traces"):
        MODULE.generate_muse([muse_trace()], 2, 2)
    missing = muse_trace()
    missing["graphs"][1].pop("capture_bytes")
    with pytest.raises(ValueError, match="lacks measured capture share"):
        MODULE.generate_muse([missing], 1, 2)


def test_muse_measured_min_inventory_prices_its_selected_rows():
    checked = json.loads((SCRIPT.parents[2] /
        "rust/crates/cuteafd-loader/src/serving_capacity/muse_prewarm.json").read_text())
    assert {(row["ranks"], row["onboard_layers"]) for row in checked["layouts"]} == {(1, 2), (2, 0)}
    layout = checked["layouts"][0]
    assert len(layout["entries"]) == 55
    assert [entry["rows"] for entry in layout["entries"] if entry["pin"]] == [56]
    assert len({entry["rows"] for entry in layout["entries"]}) == 55
    assert 1 in {entry["rows"] for entry in layout["entries"]}
    assert layout["reserve_base_bytes"] == [1056964608]
    assert layout["reserve_extra_bytes"] == [2 << 20]
    assert MODULE.packed(sum(entry["capture_bytes"][0] for entry in layout["entries"])) == 1056964608


def test_muse_trimmed_source_traces_reproduce_both_layouts_byte_for_byte():
    import gzip

    fixture = SCRIPT.parents[1] / "tests/fixtures/muse-prewarm-traces.json.gz"
    traces = json.loads(gzip.decompress(fixture.read_bytes()))
    generated = {"schema_version": 1, "family": "muse_spark", "layouts": [
        MODULE.generate_muse(traces["min"], 1, 2),
        MODULE.generate_muse(traces["max"], 2, 0)]}
    installed = SCRIPT.parents[2] / "rust/crates/cuteafd-loader/src/serving_capacity/muse_prewarm.json"
    assert json.dumps(generated, indent=2) + "\n" == installed.read_text()
    maximum = generated["layouts"][1]
    assert len(maximum["entries"]) == 53
    assert not any(entry["pin"] for entry in maximum["entries"])
    assert maximum["reserve_base_bytes"] == [1287651328, 1302331392]
    assert maximum["reserve_extra_bytes"] == [MODULE.CHUNK] * 2
    assert "2984 zero-replay identities stay lazy" in maximum["provenance"]
    for snapshots in traces.values():
        for trace in snapshots:
            assert all(graph["nodes"] >= 2 for graph in trace["graphs"])
    assert fixture.stat().st_size < 1_000_000


def test_muse_synthetic_trace_regeneration_is_byte_stable():
    expected = json.dumps(MODULE.generate_muse([muse_trace()], 1, 2), indent=2) + "\n"
    assert json.dumps(MODULE.generate_muse([muse_trace()], 1, 2), indent=2) + "\n" == expected


def test_muse_cli_reports_onboard_geometry_and_reproduces_output(tmp_path):
    import subprocess
    import sys

    trace_path = tmp_path / "trace.json"
    output = tmp_path / "prewarm.json"
    trace_path.write_text(json.dumps(muse_trace()))
    expected = json.dumps(MODULE.generate_muse([muse_trace()], 1, 2), indent=2) + "\n"
    result = subprocess.run([sys.executable, str(SCRIPT), "--family", "muse_spark",
                             "--trace", str(trace_path), "--ranks", "1", "--onboard-layers", "2",
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Muse 1 ranks onboard_layers=2: 2 geometries" in result.stdout
    assert output.read_text() == expected


def test_capture_ticket_native_failures_always_report_backoff():
    """New V4.1 capture sites must not silently cancel OOM ghosts by Drop."""
    import re

    families = SCRIPT.parents[2] / "rust/crates/cuteafd-daemon/src/families"
    checked = 0
    for path in (families / "deepseek_v41").rglob("*.rs"):
        # Test modules may precede production impls (notably the router).
        source = re.sub(r"^#\[cfg\(test\)\]\s*mod \w+\s*\{.*?^\}", "",
                        path.read_text(), flags=re.MULTILINE | re.DOTALL)
        if ".cache_ticket(" not in source and path.name != "dual.rs":
            continue
        for native in re.finditer(r"(?:decode_graph::begin_capture|\.cuda_graph_end_capture_topology)\(", source):
            prefix = source[max(0, native.start() - 180):native.start()]
            assert re.search(r"(?:plan\.)?ticket\.capture_result\([^;{}]*$", prefix), path
            checked += 1
    assert checked >= 44, "the remaining V4.1 capture inventory must stay covered"
    for family, file, api in [
        ("qwen4", "engine.rs", "cache_ticket_failed"),
        ("qwen3_5", "graphs.rs", "cache_ticket_failed"),
        ("muse_spark", "graphs.rs", "cache_ticket_failed"),
        ("kimi_k3", "graphs.rs", "cache_ticket_failed"),
        ("step5", "graphs.rs", "cache_ticket_failed"),
        ("deepseek_v4", "engine.rs", "cache_ticket_failed"),
        ("mimo_v2", "engine.rs", "cache_ticket_failed"),
        ("glm5", "engine.rs", "cache_ticket_failed"),
        ("glm5_flash", "engine.rs", "cache_ticket_failed"),
    ]:
        source = (families / family / file).read_text()
        warnings = [warning for warning in re.finditer(r'tracing::warn!\([^\n]*\);', source)
                    if "unavailable" in warning.group()
                    and ("eager" in warning.group() or "fallback" in warning.group())]
        assert len(warnings) == 2, (family, "begin and end capture failure warnings")
        for warning in warnings:
            prefix = source[max(0, warning.start() - 180):warning.start()]
            assert api + "(ticket, &error)" in prefix, (family, warning.group())


def test_kimi_core_key_merge_preserves_other_formats_and_layouts(tmp_path):
    out = tmp_path / "kimi_prewarm.json"
    k6 = MODULE.generate_kimi([kimi_trace()], 1, "k6-min", core="k6")
    bf16 = MODULE.generate_kimi([kimi_trace()], 1, "bf16-min", core="bf16")
    MODULE.merge_kimi_layout(out, k6)
    MODULE.merge_kimi_layout(out, bf16)
    assert json.loads(out.read_text())["layouts"] == [bf16, k6]
    replacement = dict(bf16, layout="bf16-new")
    MODULE.merge_kimi_layout(out, replacement)
    assert json.loads(out.read_text())["layouts"] == [replacement, k6]
    for core in (None, "fp8", ""):
        with pytest.raises(ValueError, match="checkpoint core key"):
            MODULE.generate_kimi([kimi_trace()], 1, "min", core=core)
