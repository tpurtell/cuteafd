#!/usr/bin/env python3
"""Propose a family prewarm list from GRAPH_TRACE snapshots.

Request frequency selects coverage. Schema 2 pins sustained rolling-window
heat or repeated eviction/recapture, and reports permanent pin cost separately
from measured step-boundary allocator growth. Review and qualify before use.
"""
import argparse
import json
import re
from pathlib import Path

CHUNK = 2 << 20
GEOMETRY = re.compile(r"rows: (\d+), table_width: (\d+), table_stride: (\d+)")
# Families sharing one engine share its site names.
SITE_PREFIX = {"glm5": "glm5.", "mistral_large4": "glm5."}
KIMI_ROWS = re.compile(r"rows=(\d+)")
KIMI_OP = re.compile(r"op=(\w+)")
KIMI_SPEC = re.compile(r"spec=(\d)")


def packed(value):
    return (value + CHUNK - 1) // CHUNK * CHUNK


def sustained(graph):
    histogram = graph.get("hit_histogram", {})
    counts = histogram.get("counts", [])
    interval_ms = int(histogram.get("interval_ms", 10_000))
    start_ms = int(histogram.get("start_ms", 0))
    first = max(0, (int(graph.get("first_seen_ms", 0)) - start_ms + interval_ms - 1) // interval_ms)
    # The current (partial) interval cannot prove sustained heat.
    complete = counts[first:-1]
    return len(complete) >= 3 and sum(int(count) > 0 for count in complete) >= len(complete) * 0.6


def kimi_geometry(shape):
    """Kimi keys print `rows=.. op=.. spec=..`; the prewarm identity is the
    replay width and whether it is a verify capture."""
    rows = KIMI_ROWS.search(shape)
    op = KIMI_OP.search(shape)
    if rows is None or op is None:
        raise ValueError(f"cannot parse kimi_k3 graph geometry: {shape!r}")
    if int(rows.group(1)) <= 0:
        raise ValueError("unsupported prewarm geometry")
    spec = KIMI_SPEC.search(shape)
    return int(rows.group(1)), op.group(1) != "residual" and op.group(1) != "ffn" and spec is not None and spec.group(1) == "1"


def generate_kimi(traces, ranks, layout, coverage=0.999):
    """One layout row for the checked-in list: shapes at or below decode row
    coverage, pinned only when the trace shows sustained heat or repeated
    recapture. Prefill rows stay eager and never replay (#1646)."""
    if ranks < 1 or not layout:
        raise ValueError("kimi_k3 needs a coordinator rank count and layout name")
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    counts = {}
    arenas = []
    reasons = {}
    identities = {}
    latest = {}
    for trace in traces:
        version = trace.get("schema_version")
        if version not in (1, 2):
            raise ValueError("expected GRAPH_TRACE schema_version=1 or 2")
        cache = trace["cache"]
        device = int(cache["device"])
        arenas.append(int(cache["charged"]))
        inferred_end = max((int(graph.get("hit_histogram", {}).get("start_ms", 0)) +
                            len(graph.get("hit_histogram", {}).get("counts", [])) *
                            int(graph.get("hit_histogram", {}).get("interval_ms", 0))
                            for graph in trace["graphs"]), default=0)
        snapshot_time = int(cache.get("sampled_at_ms", inferred_end))
        if device not in latest or snapshot_time >= latest[device][0]:
            latest[device] = (snapshot_time, trace)
        device_counts = {}
        for graph in trace["graphs"]:
            if not graph["site"].startswith("kimi_k3."):
                continue
            key = kimi_geometry(graph["shape"])
            if key[0] > 64:
                continue  # Row-window prefill keys cannot replay (#1646).
            cycles = graph.get("cycles")
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) \
                if cycles is not None else int(graph["count"])
            if count > 0:
                device_counts[key] = device_counts.get(key, 0) + count
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
    if not counts:
        raise ValueError("trace contains no matching graph identities")
    for device, (_, trace) in latest.items():
        for graph in trace["graphs"]:
            if not graph["site"].startswith("kimi_k3."):
                continue
            identity = (device, graph["site"], graph.get("bank", 0),
                        graph.get("key_id", graph["shape"]))
            key = kimi_geometry(graph["shape"])
            if key[0] <= 64:
                identities[identity] = (key, graph)
    for identity, (key, graph) in identities.items():
        cycles = graph.get("cycles", {})
        if not any(int(cycles.get(cycle, 0)) > 0 for cycle in ("decode", "verify")):
            continue
        why = reasons.setdefault(key, set())
        histogram = graph.get("hit_histogram", {})
        histogram_end = int(histogram.get("start_ms", 0)) + \
            len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
        if histogram_end >= latest[identity[0]][0] and sustained(graph):
            why.add("sustained_heat")
        if int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2:
            why.add("repeated_recapture")
    ranked = sorted(counts, key=lambda key: (-counts[key], key[0], key[1]))
    total = sum(counts.values())
    covered = 0
    selected = set()
    for key in ranked:
        selected.add(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    selected.update(key for key, why in reasons.items() if why)
    entries = [{"rows": rows, "speculative": speculative, "pin": bool(reasons.get((rows, speculative)))}
               for rows, speculative in sorted(selected)]
    return {"ranks": ranks, "layout": layout,
            "reserve_base_bytes": packed(max(arenas)), "reserve_extra_bytes": CHUNK,
            "entries": entries}


def merge_kimi_layout(path, row):
    """Replace one layout row so a min and a max trace update one list."""
    data = {"schema_version": 1, "family": "kimi_k3", "layouts": []}
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("family") != "kimi_k3" or data.get("schema_version") != 1:
            raise ValueError(f"{path} is not a Kimi K3 prewarm list")
    layouts = [layout for layout in data["layouts"] if int(layout["ranks"]) != row["ranks"]] + [row]
    data["layouts"] = sorted(layouts, key=lambda layout: int(layout["ranks"]))
    path.write_text(json.dumps(data, indent=2) + "\n")


def generate(traces, family, coverage=0.999, pin_frequency=0.9, pin_report=None, layout=None):
    prefix = SITE_PREFIX.get(family)
    if prefix is None:
        raise ValueError(f"no checked-in shape-list consumer for {family}")
    if family == "mistral_large4":
        if layout not in (1, 2):
            raise ValueError("mistral_large4 keys its list by layout: pass --layout 1 or 2")
    elif layout is not None:
        raise ValueError("--layout applies to mistral_large4 only")
    # A two-rank trace is recorded on the context placement, whose per-rank key
    # carries the half table; a checked-in entry holds the width a step is
    # built with, which is what the engine warms.
    scale = 2 if layout == 2 else 1
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    counts = {}
    arenas = []
    reserves = {}
    reasons = {}
    # Costs/heat use only the latest complete inventory per device.
    identities = {}
    legacy = set()
    latest = {}
    for trace in traces:
        version = trace.get("schema_version")
        if version not in (1, 2):
            raise ValueError("expected GRAPH_TRACE schema_version=1 or 2")
        cache = trace["cache"]
        device = int(cache["device"])
        arenas.append(int(cache["charged"]))
        reserves[device] = max(reserves.get(device, 0), int(cache.get("reserve", 0)))
        # Snapshots enumerate the entire bounded cache. Missing identities in a
        # newer snapshot have been forgotten; old heat/cost must not survive.
        inferred_end = max((int(graph.get("hit_histogram", {}).get("start_ms", 0)) +
                            len(graph.get("hit_histogram", {}).get("counts", [])) *
                            int(graph.get("hit_histogram", {}).get("interval_ms", 0))
                            for graph in trace["graphs"]), default=0)
        snapshot_time = int(cache.get("sampled_at_ms", inferred_end))
        if device not in latest or snapshot_time >= latest[device][0]:
            latest[device] = (snapshot_time, trace)

        device_counts = {}
        for graph in trace["graphs"]:
            if not graph["site"].startswith(prefix):
                continue
            match = GEOMETRY.search(graph["shape"])
            if match is None:
                raise ValueError(f"cannot parse {family} geometry: {graph['shape']!r}")
            rows, width, stride = map(int, match.groups())
            if rows <= 0 or width <= 0 or stride != width:
                raise ValueError("unsupported prewarm geometry")
            if scale != 1 and width % 2:
                raise ValueError(f"{family} layout {layout} key width {width} is not an even half table")
            key = (rows, width * scale)
            cycles = graph.get("cycles")
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) if cycles is not None else int(graph["count"])
            if count > 0:
                device_counts[key] = device_counts.get(key, 0) + count
                if version == 1:
                    legacy.add(key)
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
    if not counts:
        raise ValueError("trace contains no matching graph identities")
    for device, (_, trace) in latest.items():
        for graph in trace["graphs"]:
            if graph["site"].startswith(prefix):
                rows, width, _ = map(int, GEOMETRY.search(graph["shape"]).groups())
                identity = (device, graph["site"], graph.get("bank", 0), graph.get("key_id", graph["shape"]))
                identities[identity] = ((rows, width * scale), graph)
    for identity, (key, graph) in identities.items():
        cycles = graph.get("cycles", {})
        if not any(int(cycles.get(cycle, 0)) > 0 for cycle in ("decode", "verify")):
            continue
        why = reasons.setdefault(key, set())
        histogram = graph.get("hit_histogram", {})
        histogram_end = int(histogram.get("start_ms", 0)) + len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
        if histogram_end >= latest[identity[0]][0] and sustained(graph):
            why.add("sustained_heat")
        if int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2:
            why.add("repeated_recapture")
    ranked = sorted(counts, key=lambda key: (-counts[key], key[1], key[0]))
    total = sum(counts.values())
    covered = 0
    selected = set()
    for key in ranked:
        selected.add(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    # A thrashing low-frequency key can fall outside frequency coverage.
    selected.update(key for key, why in reasons.items() if why)
    peak = max(counts.values())
    minimum = min(counts, key=lambda key: (key[0] * key[1], key))
    if minimum in legacy and counts[minimum] >= peak * pin_frequency:
        reasons.setdefault(minimum, set()).add("legacy_frequency")
    entries = [{"rows": rows, "table_width": width, "pin": bool(reasons.get((rows, width)))}
               for rows, width in sorted(selected, key=lambda key: (key[1], key[0]))]
    if pin_report is not None:
        for key in sorted(selected):
            if not reasons.get(key):
                continue
            for device in sorted(reserves):
                graphs = [graph for identity, (geometry, graph) in identities.items()
                          if identity[0] == device and geometry == key]
                if not graphs:
                    continue
                known = all("permanent_bytes" in graph for graph in graphs)
                cost = sum(int(graph["permanent_bytes"]) for graph in graphs) if known else None
                pin_report.append({"device": device, "rows": key[0], "table_width": key[1],
                                   "reasons": sorted(reasons[key]), "permanent_bytes": None,
                                   "observed_permanent_lower_bound_bytes": cost,
                                   "reserve_bytes": reserves[device], "reserve_fraction": None,
                                   "observed_reserve_fraction_lower_bound": cost / reserves[device] if cost is not None and reserves[device] else None,
                                   "capture_bytes": sum(int(graph.get("capture_bytes") or 0) for graph in graphs),
                                   "nodes": sum(int(graph.get("nodes") or 0) for graph in graphs),
                                   "executables": sum(int(graph.get("executables", 1)) for graph in graphs),
                                   "basis": graphs[0].get("basis"),
                                   "cost_basis": "observed_inventory_lower_bound_unknown_total" if known else "unknown_legacy_trace"})
    base = packed(max(arenas))
    # Runtime headroom is resolved by GPU_GRAPH_RESERVE_EXTRA, not the trace.
    extra = CHUNK
    result = {"schema_version": 1, "family": family}
    if layout is not None:
        result["layout"] = layout
    result.update({"reserve_base_bytes": base, "reserve_extra_bytes": extra, "entries": entries})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=[*SITE_PREFIX, "kimi_k3"])
    parser.add_argument("--trace", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage", type=float, default=0.999)
    parser.add_argument("--pin-frequency", type=float, default=0.9, help="schema 1 legacy frequency threshold")
    parser.add_argument("--ranks", type=int, help="kimi_k3 coordinator rank count for this trace")
    parser.add_argument("--layout", help="kimi_k3 layout name for this trace, e.g. kimi-k3-exl3-k6-min")
    args = parser.parse_args()
    report = []
    try:
        traces = [json.loads(path.read_text()) for path in args.trace]
        if args.family == "kimi_k3":
            if args.ranks is None or args.layout is None:
                raise ValueError("kimi_k3 needs --ranks and --layout")
            result = generate_kimi(traces, args.ranks, args.layout, args.coverage)
            merge_kimi_layout(args.output, result)
        else:
            result = generate(traces, args.family, args.coverage, args.pin_frequency, report, int(args.layout) if args.layout is not None else None)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        parser.error(str(error))
    print(f"{len(result['entries'])} shapes; base={result['reserve_base_bytes']} "
          f"extra={result['reserve_extra_bytes']} bytes; "
          f"pinned={sum(entry['pin'] for entry in result['entries'])}")
    for pin in report:
        print("pin " + json.dumps(pin, sort_keys=True))


if __name__ == "__main__":
    main()
