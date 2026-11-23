#!/usr/bin/env python3
"""Propose a family prewarm list from GRAPH_TRACE snapshots.

Request frequency selects the traced working set. Schema 2 pins sustained
rolling-window heat or repeated eviction/recapture. Qwen4's reserve base is
the sum of the selected identities' measured `capture_bytes` per device,
rounded once, with a fixed 2 MiB capture headroom; the trace's charged
high-water is never used. Review and qualify before use.

Examples:
  python3 scripts/dev/graph-prewarm-from-trace.py --family glm5 \
    --trace trace.device-0.json --output prewarm.json
  python3 scripts/dev/graph-prewarm-from-trace.py --family glm5_flash \
    --trace 1-sparks=trace-1-sparks.json --trace 2=trace-2.json \
    --trace 2-sparks=trace-2-sparks.json --output prewarm.json

Input is the versioned GRAPH_TRACE JSON snapshot, not arbitrary log text.
Frequencies are aggregated by geometry, not bank id or layer. Pin candidates
must have both dominant frequency and the lowest row/table parameter rank.
Reserve suggestions use the maximum per-device arena, never a sum of ranks.
GLM Flash prices the base from the selected entries' measured capture bytes per
device (the maximum across devices and quants), not the trace's whole-session
arena: shapes outside the coverage set never prewarm and never charge.
GLM Flash keys its list by resolved card shape, `<devices>[-sparks]`: onboarding
and TP2 each add captured segments, so each shape's own capture demand prices
its own reserve and a launch reads only its own entry. A shape's entries are
the same for every quant of the checkpoint (a shape is a pure function of the
launch geometry); when a shape is traced on more than one quant, the costlier
quant's measured arena and largest capture price it. A shape with no trace
warms the engine's bootstrap set, never another shape's list.

Qwen4 keys its list by resolved capture shape (RTX count, Sparks attached):
pass `--shape` with `--merge qwen_prewarm.json` to install the generated object
into that shape's entry, replacing its `trace: null`; every other shape keeps
its own reserve and list. `--output` writes the generated object alone for
review. A traced shape's base is the per-device sum of its selected prewarm
identities' measured capture bytes (startup-only siblings included,
deduplicated by identity, rounded once to 2 MiB); selection ranks by replayed
hits only, so a zero-hit startup shape is never selected and a
captured-but-never-replayed identity stays lazy.

DeepSeek V4 keys a shape by the resolved coordinator GPU count and Spark
attachment (`rtx1-spark`, `rtx1-local`, `rtx2-spark`, `rtx2-local`), and
captures different graphs per rank, so `--layout` plus `--list` replaces that
one shape's per-rank arrays in the checked-in `v4_prewarm.json` and marks it
traced: devices map to ranks in ascending device order, `reserve_base_bytes` is
the emitted prewarm set's own measured captures (the trace's whole charged
arena also holds lazy captures, which the shared reserve-extra policy covers),
`reserve_extra_bytes` is the arena's 2 MiB allocation tail (the shared capture
headroom is separate), and `bootstrap_bytes` is the largest measured capture
unless the trace has none.
Every verify geometry the trace observed becomes an entry unless it is rare -
used once, or only inside one sampling interval, such as one panel or the 32K
context pass - which stays lazy under the shared capture headroom. A width-0
startup row follows a stricter test: it is warmed only when the trace replayed
it, so a row captured once and never looked up again stays lazy. Pins follow
sustained heat, plus two keys that cost every step: plain C1 (rows 1, width 0)
and the layout's dominant verify geometry pin whatever their heat class. A row
hot in one sampling interval, or merely recurring, is otherwise prewarmed but
stays evictable for ARC. Entries follow the fan-out order - pinned first, then
lifetime replay hits descending - so a reduced reserve warms the hottest rows
first. A sustained geometry is pinned, a regular one prewarmed unpinned, and
the heat report prints each class's keys and arena bytes. The engine warms the
entries its pool can express and skips the rest.
"""
import argparse
import ast
import json
import math
import re
from pathlib import Path

CHUNK = 2 << 20
GLM5_FLASH_LAYOUTS = {"1", "1-sparks", "2", "2-sparks"}
EXTRA_BYTES = 2 << 20
GEOMETRY = re.compile(r"rows: (\d+), table_width: (\d+), table_stride: (\d+)")
# The prewarm arena's allocation tail; capture headroom is a shared policy key.
V4_TAIL = 2 << 20
# GraphKey's display shape; width 0 is the local-expert startup set or the tail.
V4_GEOMETRY = re.compile(r'rows: (\d+), chunked: \w+, attention: "([^"]+)", '
                         r"table_width: (\d+), table_stride: (\d+)")
# Families sharing one engine share its site names.
SITE_PREFIX = {"glm5": "glm5.", "mistral_large4": "glm5.",
               "deepseek_v4": "deepseek_v4."}
GLMF_GEOMETRY = re.compile(
    r"GraphKey \{ segment: \d+, rows: (\d+), spec: (true|false), long: (true|false), "
    r"pool_width: (\d+), page_stride: (\d+), pool_stride: (\d+) \}")
GLMF_NORMALIZED_GEOMETRY = re.compile(
    r"GraphKey \{ segment: \d+, rows: (\d+), spec: (true|false), head: (true|false), "
    r"long: (true|false), pool_width: (\d+), page_stride: (\d+), pool_stride: (\d+) \}")
QWEN_GEOMETRY = re.compile(
    r"GraphKey \{ segment: (\d+), rows: (\d+), spec: (true|false), long: (true|false), "
    r"pool_width: (\d+), page_stride: (\d+), pool_stride: (\d+) \}")
QWEN_NORMALIZED_GEOMETRY = re.compile(
    r"GraphKey \{ segment: (\d+), rows: (\d+), spec: (true|false), head: (true|false), "
    r"long: (true|false), page_width: (\d+), page_stride: (\d+), pool_stride: (\d+) \}")
QWEN_SHAPES = ("single_0spark", "single_sparks", "dual_0spark", "dual_sparks")
KIMI_ROWS = re.compile(r"rows=(\d+)")
KIMI_RANK = re.compile(r"rank=(\d+)")
KIMI_OP = re.compile(r"op=(\w+)")
KIMI_SPEC = re.compile(r"spec=(\d)")


V41_DECODE_ROWS = 64
V41_REPLAY_ROWS = 128
V41_PRODUCERS = {"window", "compressor"}
V41_HEADS = {"target_head", "target_norm", "vocabulary_logits", "vocabulary_greedy"}
V41_LAYER_ROWS = {
    "lane_row", "lane_prefill", "attention.query", "attention.query.tp2_prefix",
    "attention.output", "backbone.shared", "router.remote", "router.local", "index.query",
    "projection.query.rank", "projection.query.gather",
    "projection.output.rank", "projection.output.gather", "tp2_ffn",
}


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


# Hottest class wins when ranks disagree, because one prewarm set is shared.
HEAT_ORDER = {"rare": 0, "regular": 1, "sustained": 2, "startup": 3}


def heat(graph):
    """The startup class of one traced geometry.

    `rare` is a key the trace used once - a capture with no later hit - or only
    inside a single sampling interval, which is one point or panel such as the
    32K context pass; it stays lazy under the shared capture headroom.
    `sustained` is the rolling-window heat a pin is earned by, and `regular` is
    everything between: worth prewarming, not worth permanent residency.
    """
    histogram = graph.get("hit_histogram", {})
    counts = histogram.get("counts", [])
    intervals = sum(int(count) > 0 for count in counts) if counts else int(int(graph.get("count", 0)) > 0)
    if int(graph.get("hits", 0)) == 0 or intervals <= 1:
        return "rare"
    return "sustained" if sustained(graph) else "regular"


def geometry(family, shape):
    """`(rows, table_width)` of a captured segment's display shape."""
    if family != "deepseek_v4":
        match = GEOMETRY.search(shape)
        if match is None:
            raise ValueError(f"cannot parse {family} geometry: {shape!r}")
        rows, width, stride = map(int, match.groups())
        if rows <= 0 or width <= 0 or stride != width:
            raise ValueError("unsupported prewarm geometry")
        return rows, width
    match = V4_GEOMETRY.search(shape)
    if match is None:
        raise ValueError(f"cannot parse deepseek_v4 geometry: {shape!r}")
    rows, width, stride = int(match[1]), int(match[3]), int(match[4])
    if rows <= 0 or width < 0 or stride != width:
        raise ValueError("unsupported prewarm geometry")
    return rows, width


def prewarmable(family, shape):
    """Whether a captured key can stand as a prewarm entry.

    V4 keeps tails eager (#1638), so a width-0 tail capture must not become a
    startup entry; every other width-0 key is the local-expert/TP2 startup set
    (a 1-RTX shape captures `local-experts`, a 2-RTX one `tp2-broadcast` and
    `tp2-replicated`), and their geometry is the same `(rows, 0)` key.
    """
    if family != "deepseek_v4":
        return True
    match = V4_GEOMETRY.search(shape)
    return match is None or int(match[3]) > 0 or match[2] != "tail"


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
    kind = op.group(1)
    if kind not in ("ffn", "attention", "context"):
        raise ValueError(f"unknown Kimi operation: {kind}")
    return (int(rows.group(1)), kind != "ffn" and spec is not None and spec.group(1) == "1", kind)


def kimi_lifetime_pins(identities):
    """Project per-key lifetime demand onto geometry, without treating the final
    rolling window as a whole-run history. Startup-only counts never qualify."""
    reasons = {}
    classes = {}
    verify = {}
    for (device, *_), (key, graph) in identities.items():
        cycles = graph.get("cycles", {})
        request_count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify"))
        if request_count <= 0:
            continue
        if key[0] == 1 and not key[1]:
            reasons.setdefault(key, set()).add("plain_c1")
        # Capture misses are not hits; cap lifetime hits by request activity so
        # startup Other replays cannot manufacture a hot tier.
        hits = min(int(graph.get("hits", 0)), request_count)
        if hits > 0:
            classes.setdefault((device, key[2]), []).append((hits, key))
        if key[1] and key[2] in ("attention", "context"):
            geometry = (device, key[2], key[0])
            verify[geometry] = verify.get(geometry, 0) + int(cycles.get("verify", 0))
    for rows in classes.values():
        rows.sort(key=lambda row: -row[0])
        total = sum(hits for hits, _ in rows)
        covered = 0
        cutoff = None
        for hits, key in rows:
            if cutoff is not None and hits < cutoff:
                break
            reasons.setdefault(key, set()).add("lifetime_hit_tier")
            covered += hits
            if covered * 2 >= total and cutoff is None:
                cutoff = hits
    # Pick a dominant verify width per device, preferring attention over Context,
    # then pin its flagless FFN and peer Context siblings.
    for device in {owner[0] for owner in verify}:
        candidates = [(count, rows) for (owner, op, rows), count in verify.items()
                      if owner == device and op == "attention" and count > 0]
        if not candidates:
            candidates = [(count, rows) for (owner, _, rows), count in verify.items()
                          if owner == device and count > 0]
        if not candidates:
            continue
        _, rows = min(candidates, key=lambda row: (-row[0], row[1]))
        for (owner, *_), (key, graph) in identities.items():
            if owner != device or key[0] != rows or not graph.get("cycles", {}).get("verify", 0):
                continue
            if key[2] == "ffn" or key[1]:
                reasons.setdefault(key, set()).add("dominant_verify")
    return reasons


def generate_kimi(traces, ranks, layout, coverage=0.999, pin_report=None, *, core):
    """One layout row: lifetime-hot and rolling/churn pins plus unpinned request
    coverage. Prefill rows stay eager and never replay (#1646)."""
    if core not in ("bf16", "k6"):
        raise ValueError("kimi_k3 needs --variant bf16 or k6 for the checkpoint core key")
    if ranks < 1 or not layout:
        raise ValueError("kimi_k3 needs a coordinator rank count and layout name")
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    counts = {}
    lifetime_hits = {}
    reasons = {}
    identities = {}
    latest = {}
    for trace in traces:
        version = trace.get("schema_version")
        if version not in (1, 2):
            raise ValueError("expected GRAPH_TRACE schema_version=1 or 2")
        cache = trace["cache"]
        device = int(cache["device"])
        inferred_end = max((int(graph.get("hit_histogram", {}).get("start_ms", 0)) +
                            len(graph.get("hit_histogram", {}).get("counts", [])) *
                            int(graph.get("hit_histogram", {}).get("interval_ms", 0))
                            for graph in trace["graphs"]), default=0)
        snapshot_time = int(cache.get("sampled_at_ms", inferred_end))
        if device not in latest or snapshot_time >= latest[device][0]:
            latest[device] = (snapshot_time, trace)
    for device, (_, trace) in latest.items():
        for graph in trace["graphs"]:
            if not graph["site"].startswith("kimi_k3."):
                continue
            identity = (device, graph["site"], graph.get("bank", 0),
                        graph.get("key_id", graph["shape"]))
            key = kimi_geometry(graph["shape"])
            if key[0] <= 64:
                identities[identity] = (key, graph)
    replayed = {}
    for identity, (key, graph) in identities.items():
        cycles = graph.get("cycles", {})
        request_count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify"))
        if request_count <= 0 or int(graph.get("hits", 0)) < 1:
            continue
        replayed[identity] = (key, graph)
        device_counts = counts.setdefault(identity[0], {})
        device_counts[key] = device_counts.get(key, 0) + request_count
        device_hits = lifetime_hits.setdefault(identity[0], {})
        device_hits[key] = device_hits.get(key, 0) + min(int(graph.get("hits", 0)), request_count)
        why = reasons.setdefault(key, set())
        histogram = graph.get("hit_histogram", {})
        histogram_end = int(histogram.get("start_ms", 0)) + \
            len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
        if histogram_end >= latest[identity[0]][0] and sustained(graph):
            why.add("sustained_heat")
        if int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2:
            why.add("repeated_recapture")
    counts = {key: max(rows.get(key, 0) for rows in counts.values())
              for key in {key for rows in counts.values() for key in rows}}
    if not counts:
        raise ValueError("trace contains no replayed Kimi request identities")
    lifetime_reasons = kimi_lifetime_pins(replayed)
    for key, why in lifetime_reasons.items():
        reasons.setdefault(key, set()).update(why)
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    total = sum(counts.values())
    covered = 0
    selected = set()
    for key in ranked:
        selected.add(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    selected.update(key for key, why in reasons.items() if why)
    heat = {key: max(rows.get(key, 0) for rows in lifetime_hits.values()) for key in selected}
    ordered = sorted(selected, key=lambda key: (not bool(reasons.get(key)), -heat[key], key))
    entries = [{"rows": rows, "speculative": speculative, "op": op,
                "pin": bool(reasons.get((rows, speculative, op)))}
               for rows, speculative, op in ordered]
    # Charge only emitted geometry, including startup-only siblings. The
    # completed-step shares are physical growth; zero is valid allocator reuse.
    bases = [None] * ranks
    costs = {key: [None] * ranks for key in selected}
    for device, (_, trace) in latest.items():
        owners = {int(match.group(1)) for graph in trace["graphs"]
                  if graph["site"].startswith("kimi_k3.")
                  and (match := KIMI_RANK.search(graph["shape"])) is not None}
        if len(owners) != 1:
            raise ValueError(f"Kimi device {device} needs exactly one physical rank")
        rank = owners.pop()
        if not 0 <= rank < ranks or bases[rank] is not None:
            raise ValueError("Kimi trace device/rank mapping is not one-to-one")
        shares = []
        for (owner, *_), (geometry, graph) in identities.items():
            if owner != device or geometry not in selected:
                continue
            share = graph.get("capture_bytes")
            if not isinstance(share, int) or isinstance(share, bool) or share < 0:
                raise ValueError(f"Kimi emitted geometry has missing or invalid capture_bytes: {graph['shape']}")
            shares.append(share)
            costs[geometry][rank] = (costs[geometry][rank] or 0) + share
        bases[rank] = packed(sum(shares))
    if any(base is None for base in bases):
        raise ValueError("Kimi trace is missing a physical rank")
    for entry, key in zip(entries, ordered):
        entry["capture_bytes"] = costs[key]
    if pin_report is not None:
        for key in sorted(selected):
            for device in sorted(latest):
                graphs = [graph for (owner, *_), (geometry, graph) in identities.items()
                          if owner == device and geometry == key]
                if not graphs:
                    continue
                known = all("permanent_bytes" in graph for graph in graphs)
                pin_report.append({"device": device, "rows": key[0], "speculative": key[1],
                                   "op": key[2],
                                   "pin": bool(reasons.get(key)), "reasons": sorted(reasons.get(key, [])),
                                   "rolling_pin": "sustained_heat" in reasons.get(key, set()),
                                   "lifetime_pin": key in lifetime_reasons,
                                   "whole_run_interval_spread": None,
                                   "lifetime_request_hits": sum(min(int(graph.get("hits", 0)),
                                       sum(int(graph.get("cycles", {}).get(cycle, 0))
                                           for cycle in ("decode", "verify"))) for graph in graphs),
                                   "zero_hit_sibling_identities": sum(int(graph.get("hits", 0)) == 0 for graph in graphs),
                                   "zero_hit_sibling_bytes": sum(int(graph.get("capture_bytes") or 0)
                                       for graph in graphs if int(graph.get("hits", 0)) == 0),
                                   "executables": sum(int(graph.get("executables", 1)) for graph in graphs),
                                   "observed_permanent_lower_bound_bytes":
                                       sum(int(graph["permanent_bytes"]) for graph in graphs) if known else None,
                                   "observed_capture_bytes": sum(int(graph.get("capture_bytes") or 0) for graph in graphs),
                                   "full_geometry_cost_bytes": None})
    return {"core": core, "ranks": ranks, "layout": layout,
            "reserve_base_bytes": bases, "reserve_extra_bytes": CHUNK,
            "entries": entries}



def merge_kimi_layout(path, row):
    """Replace one core/layout row, preserving every other key's evidence."""
    data = {"schema_version": 1, "family": "kimi_k3", "layouts": []}
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("family") != "kimi_k3" or data.get("schema_version") != 1:
            raise ValueError(f"{path} is not a Kimi K3 prewarm list")
    layouts = [layout for layout in data["layouts"]
               if (layout["core"], int(layout["ranks"])) != (row["core"], row["ranks"])] + [row]
    data["layouts"] = sorted(layouts, key=lambda layout: (layout["core"], int(layout["ranks"])))
    path.write_text(json.dumps(data, indent=2) + "\n")



def generate_deepseek_v4(traces, layout, existing, coverage=0.999, pin_frequency=0.9,
                         pin_report=None, heat_report=None):
    """Replace one deepseek_v4 layout in the checked-in per-rank prewarm list."""
    if layout is None or existing is None:
        raise ValueError("deepseek_v4 needs --layout and --list")
    if existing.get("family") != "deepseek_v4":
        raise ValueError(f"list is for {existing.get('family')}, not deepseek_v4")
    prior = existing.get("layouts", {}).get(layout)
    if prior is None:
        raise ValueError(f"list has no layout {layout!r}")
    ranks = len(prior["reserve_base_bytes"])
    if len(prior["bootstrap_bytes"]) != ranks:
        raise ValueError(f"layout {layout} has inconsistent rank arrays")
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    prefix = SITE_PREFIX["deepseek_v4"]
    counts = {}
    reserves = {}
    captures = {}
    # Per device and geometry, the largest capture total any one snapshot saw.
    spreads = {}
    matched = set()
    reasons = {}
    # The startup class of every observed geometry, hottest rank wins.
    classes = {}
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
        device_bytes = {}
        for graph in trace["graphs"]:
            if not graph["site"].startswith(prefix):
                continue
            matched.add(device)
            if not prewarmable("deepseek_v4", graph["shape"]):
                continue
            captures[device] = max(captures.get(device, 0), int(graph.get("capture_bytes") or 0))
            key = geometry("deepseek_v4", graph["shape"])
            device_bytes[key] = device_bytes.get(key, 0) + int(graph.get("capture_bytes") or 0)
            cycles = graph.get("cycles")
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) if cycles is not None else int(graph["count"])
            if count > 0:
                device_counts[key] = device_counts.get(key, 0) + count
                if version == 1:
                    legacy.add(key)
        for key, value in device_bytes.items():
            spreads[(device, key)] = max(spreads.get((device, key), 0), value)
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
    if not counts:
        raise ValueError("trace contains no matching graph identities")
    for device, (_, trace) in latest.items():
        for graph in trace["graphs"]:
            if graph["site"].startswith(prefix) and prewarmable("deepseek_v4", graph["shape"]):
                key = geometry("deepseek_v4", graph["shape"])
                identity = (device, graph["site"], graph.get("bank", 0), graph.get("key_id", graph["shape"]))
                identities[identity] = (key, graph)
    for identity, (key, graph) in identities.items():
        cycles = graph.get("cycles", {})
        if not any(int(cycles.get(cycle, 0)) > 0 for cycle in ("decode", "verify")):
            continue
        # A width-0 row is a startup row once the trace proves a replay; a
        # capture with no later hit stays an ordinary lazy graph. Interval
        # concentration is a verify-geometry test: a short panel still proves a
        # decode row is reachable at that layout. The hottest rank decides a
        # shared entry's class.
        observed = heat(graph)
        if key[1] == 0:
            observed = "startup" if int(graph.get("hits", 0)) > 0 else "rare"
        if key not in classes or HEAT_ORDER[observed] > HEAT_ORDER[classes[key]]:
            classes[key] = observed
        why = reasons.setdefault(key, set())
        histogram = graph.get("hit_histogram", {})
        histogram_end = int(histogram.get("start_ms", 0)) + len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
        if histogram_end >= latest[identity[0]][0] and sustained(graph):
            why.add("sustained_heat")
        if int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2:
            why.add("repeated_recapture")
    # Startup residency is for heat the trace proves. A sustained verify
    # geometry is pinned, a regular one is prewarmed unpinned, and a rare
    # one - captured once, or used only inside one sampling interval such
    # as the 32K context pass - stays lazy under the shared capture
    # headroom. A width-0 startup row is kept when the trace replayed it,
    # but it earns a pin only on sustained heat: a row hot in one interval,
    # or merely recurring, lives in the prewarmed set and stays evictable
    # so ARC can reclaim it if serving never revisits it. One captured once
    # and never replayed is neither warmed nor pinned. A thrashing key is
    # kept even when rare, because the trace shows the churn rather than a
    # single use. Two keys are the cost of every step, so they pin whatever
    # their heat class: plain C1 (rows 1, width 0) and the layout's
    # dominant verify geometry, the width>0 key with the most lifetime hits
    # in the latest device snapshots. Entries keep the fan-out order -
    # pinned first, then lifetime replay hits descending - so a reduced
    # reserve warms the hottest rows before the list ends. Order never
    # changes selection, base or pin flags.
    selected = {key for key in counts
                if classes.get(key, "rare") != "rare" or reasons.get(key)}
    if (1, 0) in counts:
        selected.add((1, 0))
        reasons.setdefault((1, 0), set()).add("c1_decode")
    lifetime_hits = {}
    for key, graph in identities.values():
        lifetime_hits[key] = lifetime_hits.get(key, 0) + int(graph.get("hits", 0))
    verify_hits = {key: hits for key, hits in lifetime_hits.items() if key[1] > 0}
    if verify_hits:
        dominant = max(verify_hits, key=lambda key: (verify_hits[key], key))
        selected.add(dominant)
        reasons.setdefault(dominant, set()).add("dominant_verify")
    peak = max(counts.values())
    minimum = min(counts, key=lambda key: (key[0] * key[1], key))
    if minimum in legacy and counts[minimum] >= peak * pin_frequency:
        reasons.setdefault(minimum, set()).add("legacy_frequency")
    # Fan-out order: the pinned set first, then lifetime replay hits
    # descending, so a shrunken reserve keeps what serving replays most.
    order = sorted(selected, key=lambda key: (not reasons.get(key),
                                              -lifetime_hits.get(key, 0), key[1], key[0]))
    entries = [{"rows": rows, "table_width": width, "pin": bool(reasons.get((rows, width)))}
               for rows, width in order]
    if pin_report is not None:
        for key in sorted(selected):
            if not reasons.get(key):
                continue
            for device in sorted(reserves):
                graphs = [graph for identity, (shape, graph) in identities.items()
                          if identity[0] == device and shape == key]
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
    devices = sorted(matched)
    if len(devices) != ranks:
        raise ValueError(f"layout {layout} traces {len(devices)} matching devices, needs {ranks}")
    if heat_report is not None:
        for device in devices:
            totals = {}
            for key, observed in classes.items():
                bucket = totals.setdefault(observed, [0, 0, False])
                bucket[0] += 1
                bucket[1] += spreads.get((device, key), 0)
                bucket[2] = bucket[2] or key in selected
            for observed in ("startup", "sustained", "regular", "rare"):
                if observed in totals:
                    observed_keys, arena, kept = totals[observed]
                    heat_report.append({"device": device, "heat": observed,
                                        "keys": observed_keys, "arena_bytes": arena,
                                        "packed_bytes": packed(arena), "selected": kept})
    base = [packed(sum(spreads.get((device, key), 0) for key in selected))
            for device in devices]
    # The prewarm arena's tail only; runtime capture headroom is charged by
    # the shared GPU_GRAPH_RESERVE_EXTRA/_CAP policy, never hand-picked here.
    extra = [V4_TAIL] * ranks
    # One measured capture estimates the next; a trace without capture
    # shares keeps the checked-in number.
    bootstrap = [min(packed(captures.get(device, 0)) or prior["bootstrap_bytes"][rank], base[rank] + extra[rank])
                 for rank, device in enumerate(devices)]
    layouts = dict(existing["layouts"])
    layouts[layout] = {"traced": True,
                       "reserve_base_bytes": base, "reserve_extra_bytes": extra,
                       "bootstrap_bytes": bootstrap, "entries": entries}
    return {"schema_version": existing.get("schema_version", 1), "family": "deepseek_v4",
            "layouts": layouts}


def v41_geometry(shape, allowed, site):
    """Decode native Debug keys by site, never by guessing an integer axis.

    Pointer/fingerprint axes identify executables, but startup consumes row
    preferences and (request count, draft width) preferences, not addresses.
    """
    def integers(value, length=None):
        return isinstance(value, (tuple, list)) and (length is None or len(value) == length) and all(type(v) is int and v >= 0 for v in value)

    try:
        value = ast.literal_eval(shape)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"cannot parse V4.1 geometry at site {site}: {shape!r}") from error
    name = site.removeprefix("deepseek_v41.")
    if name == "dspark_cache" and value == ():
        return None
    if name in {"dspark_chain", "dspark_distributed_terminal"}:
        if name == "dspark_chain" and integers(value, 2):
            count, width = value
        elif name == "dspark_distributed_terminal" and integers(value, 3) and value[1] in (0, 1):
            width, _, count = value
        else:
            raise ValueError(f"invalid V4.1 draft geometry at {site}: {shape!r}")
        if not 1 <= count <= V41_DECODE_ROWS or width not in (5, 7):
            raise ValueError(f"invalid V4.1 draft axes at {site}: {shape!r}")
        return ("draft", count, width)
    if name in V41_PRODUCERS and integers(value, 2):
        rows = value[0]
    elif name in V41_LAYER_ROWS and integers(value, 2):
        rows = value[1]
    elif name in V41_HEADS | {"target_embedding", "engram.gate"} and type(value) is int:
        rows = value
    elif name == "index_selection" and integers(value) and len(value) >= 9:
        rows = value[1]
    elif name in {"sparse_attention", "dual_attention_tail"} and isinstance(value, tuple) and len(value) == 2 and type(value[0]) is int and integers(value[1]):
        fingerprint = value[1]
        if name == "dual_attention_tail" and len(fingerprint) >= 2:
            rows = fingerprint[-2]
        elif name == "sparse_attention" and len(fingerprint) >= 5:
            rows = fingerprint[2] if fingerprint[0] in (2**32 - 1, 2**64 - 1) else fingerprint[1]
        else:
            raise ValueError(f"invalid V4.1 fingerprint at {site}: {shape!r}")
    else:
        raise ValueError(f"cannot parse V4.1 geometry at site {site}: {shape!r}")
    # TP2 draft heads flatten count * width into a row axis.
    head_rows = {count * width for count in range(1, V41_DECODE_ROWS + 1) for width in (5, 7)}
    if rows not in allowed and not (name in V41_HEADS and rows in head_rows):
        raise ValueError(f"V4.1 row width {rows} at site {site} is outside the prewarm domain")
    return ("row", rows, 0)



def glm_request_replays(graph):
    """Startup hits and lazy capture misses are not request replay demand."""
    cycles = graph.get("cycles")
    requests = (sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify"))
                if cycles is not None else int(graph["count"]))
    return min(int(graph.get("hits", 0)), requests)


def generate_glm5(traces, family, coverage=0.999, pin_frequency=0.9, pin_report=None, layout=None):
    prefix = SITE_PREFIX.get(family)
    if prefix is None:
        raise ValueError(f"no checked-in shape-list consumer for {family}; qwen4 is keyed by --shape")
    if family == "mistral_large4" or layout is not None:
        if layout not in (1, 2):
            raise ValueError(f"{family} keys its list by layout: pass --layout 1 or 2")
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
            if version == 1 and "hits" not in graph:
                raise ValueError("legacy trace has no replay hits; regenerate with schema 2")
            match = GEOMETRY.search(graph["shape"])
            if match is None:
                raise ValueError(f"cannot parse {family} geometry: {graph['shape']!r}")
            rows, width, stride = map(int, match.groups())
            if rows <= 0 or width <= 0 or stride != width:
                raise ValueError("unsupported prewarm geometry")
            if scale != 1 and width % 2:
                raise ValueError(f"{family} layout {layout} key width {width} is not an even half table")
            key = (rows, width * scale)
            count = glm_request_replays(graph)
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
    # A thrashing low-frequency key can fall outside replay coverage, but a
    # geometry with no request replays must remain lazy even if it churns.
    selected.update(key for key, why in reasons.items() if why and key in counts)
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


def generate_v41(traces, coverage, pin_frequency, prefill_rows, pin_report, layout):
    if layout not in (1, 2):
        raise ValueError("deepseek_v41 requires --layout 1 or 2")
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    if prefill_rows is not None and not 1 <= prefill_rows <= 4096:
        raise ValueError("prefill rows must be in 1..4096")
    prefill = {V41_REPLAY_ROWS}
    if prefill_rows is not None:
        prefill.add(prefill_rows)
    allowed = set(range(1, V41_DECODE_ROWS + 1)) | prefill
    latest = {}
    identities = {}
    drafts = set()
    exercised_prefill = set()
    for trace in traces:
        if trace.get("schema_version") not in (1, 2):
            raise ValueError("expected GRAPH_TRACE schema_version=1 or 2")
        device = trace["cache"]["device"]
        if type(device) is not int or not 0 <= device < layout:
            raise ValueError("trace device is outside the declared layout")
        sampled = trace["cache"].get("sampled_at_ms", 0)
        if device not in latest or sampled >= latest[device][0]:
            latest[device] = (sampled, trace)
        occurrences = {}
        snapshot = {(graph["site"], graph.get("bank", 0), graph.get("key_id", graph["shape"])): graph
                    for graph in trace["graphs"]}
        for graph in sorted(snapshot.values(), key=lambda graph: (graph["site"], graph.get("bank", 0), graph.get("key_id", 0), graph["shape"])):
            if not graph["site"].startswith("deepseek_v41."):
                continue
            cost = graph.get("capture_bytes")
            if cost is not None and (type(cost) is not int or cost < 0):
                raise ValueError("capture_bytes must be a nonnegative integer for every identity")
            hits = graph.get("hits")
            if type(hits) is not int or hits < 0:
                raise ValueError("hits must be a nonnegative integer (capture counts are not replays)")
            geometry = v41_geometry(graph["shape"], allowed, graph["site"])
            # key_id and pointer fingerprints change across launches. Pair each
            # bank/geometry one-to-one in key order; duplicates remain distinct.
            shape_identity = (device, graph["site"], graph.get("bank", 0), geometry)
            ordinal = occurrences.get(shape_identity, 0)
            occurrences[shape_identity] = ordinal + 1
            identity = (*shape_identity, ordinal)
            previous = identities.get(identity)
            if previous is None:
                merged = {**graph, "cycles": dict(graph.get("cycles", {}))}
            else:
                old = previous[1]
                merged = {**(graph if sampled >= old["_sampled_at_ms"] else old)}
                merged["hits"] = old["hits"] + hits
                merged["cycles"] = {key: old.get("cycles", {}).get(key, 0) + graph.get("cycles", {}).get(key, 0)
                                    for key in old.get("cycles", {}).keys() | graph.get("cycles", {}).keys()}
                measured = [value for value in (old.get("capture_bytes"), cost) if value is not None]
                merged["capture_bytes"] = max(measured) if measured else None
            merged["_sampled_at_ms"] = max(sampled, previous[1]["_sampled_at_ms"] if previous else sampled)
            identities[identity] = (geometry, merged)
            if geometry and geometry[0] == "draft":
                drafts.add(geometry)
            if geometry and geometry[0] == "row" and geometry[1] in prefill:
                if graph.get("cycles", {}).get("prefill", 0) > 0:
                    exercised_prefill.add(geometry[1])
    if set(latest) != set(range(layout)):
        raise ValueError("need a complete trace inventory for every layout rank")
    for identity, (_, graph) in identities.items():
        if graph.get("capture_bytes") is None:
            raise ValueError(f"capture_bytes missing for identity {identity!r} in every trace")
    if not identities:
        raise ValueError("trace contains no matching graph identities")

    def geometries(geometry, graph):
        if geometry is None:
            return set()
        kind, rows, _ = geometry
        if kind == "draft":
            return {geometry}
        name = graph["site"].removeprefix("deepseek_v41.")
        keys = set()
        if rows <= V41_DECODE_ROWS:
            keys.add(geometry)
        if rows in exercised_prefill:
            keys.add(("prefill", rows, 0))
        if name in V41_HEADS:
            owners = {key for key in drafts if key[1] * key[2] == rows}
            if rows > V41_DECODE_ROWS and rows not in exercised_prefill and not owners and graph["hits"]:
                raise ValueError(f"flattened draft-head rows {rows} lack an observed draft_entries owner")
            keys.update(owners)
        return keys

    counts, reasons = {}, {}
    for identity, (geometry, graph) in identities.items():
        # A startup capture is not demand. Only actual replay selects geometry.
        if not graph["hits"] or not any(graph.get("cycles", {}).get(c, 0) for c in ("decode", "verify", "prefill")):
            continue
        for key in geometries(geometry, graph):
            counts[key] = counts.get(key, 0) + graph["hits"]
            why = reasons.setdefault(key, set())
            histogram = graph.get("hit_histogram", {})
            histogram_end = int(histogram.get("start_ms", 0)) + len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
            if histogram_end >= latest[identity[0]][0] and sustained(graph):
                why.add("sustained_heat")
            if graph.get("evictions", 0) >= 2 and graph.get("recaptures", 0) >= 2:
                why.add("repeated_recapture")
    selected, covered = set(), 0
    for key in sorted(counts, key=lambda key: (-counts[key], key)):
        selected.add(key)
        covered += counts[key]
        if covered >= sum(counts.values()) * coverage:
            break
    selected.update(key for key, why in reasons.items() if why)
    # Exact repeated chunk/replay captures may have no hit in a short trace,
    # but only when the workload actually exercised their prefill geometry.
    selected.update(("prefill", rows, 0) for rows in exercised_prefill)
    for geometry, graph in identities.values():
        if graph["hits"] and geometry and geometry[0] == "row" and geometry[1] > V41_DECODE_ROWS:
            selected.update(key for key in geometries(geometry, graph) if key[0] == "draft")
    ranks = []
    for device in range(layout):
        inventory = [(geometry, graph) for identity, (geometry, graph) in identities.items() if identity[0] == device]
        present = set().union(*(geometries(geometry, graph) for geometry, graph in inventory))
        keys = selected & present
        chosen = [graph for geometry, graph in inventory if geometries(geometry, graph) & keys
                  or (geometry is None and any(key[0] == "draft" for key in keys))]
        capture = sum(graph["capture_bytes"] for graph in chosen)
        zero = [graph for graph in chosen if not graph["hits"]]
        rank = {"rank": device, "reserve_base_bytes": packed(capture), "reserve_extra_bytes": CHUNK,
                "entries": [{"rows": rows, "pin": bool(reasons.get((kind, rows, width)))}
                            for kind, rows, width in sorted(keys) if kind == "row"],
                "prefill_entries": [{"rows": rows, "pin": True} for kind, rows, _ in sorted(keys) if kind == "prefill"],
                "draft_entries": [{"requests": rows, "width": width, "pin": bool(reasons.get((kind, rows, width)))}
                                  for kind, rows, width in sorted(keys) if kind == "draft"]}
        ranks.append(rank)
        if pin_report is not None:
            pin_report.append({"device": device, "capture_bytes": capture, "identities": len(chosen),
                               "zero_replay_sibling_bytes": sum(graph["capture_bytes"] for graph in zero),
                               "zero_replay_sibling_identities": len(zero),
                               "cost_basis": "measured step growth, split by estimate within shared steps"})
    return {"schema_version": 2, "family": "deepseek_v41", "layout": layout, "ranks": ranks}


MIMO_GEOMETRY = re.compile(r"rows: (\d+)")
MIMO_DECODE_ROWS = 64


def key_of(family, shape):
    """The family's prewarm key from a trace shape string."""
    if family != "mimo_v2":
        match = GEOMETRY.search(shape)
        if match is None:
            raise ValueError(f"cannot parse {family} geometry: {shape!r}")
        rows, width, stride = map(int, match.groups())
        if rows <= 0 or width <= 0 or stride != width:
            raise ValueError("unsupported prewarm geometry")
        return (rows, width)
    match = MIMO_GEOMETRY.search(shape)
    if match is None:
        raise ValueError(f"cannot parse {family} geometry: {shape!r}")
    (rows,) = map(int, match.groups())
    if not 1 <= rows <= MIMO_DECODE_ROWS:
        raise ValueError("unsupported prewarm geometry")
    return (rows,)



def entry_of(family, key):
    if family != "mimo_v2":
        rows, width = key
        return {"rows": rows, "table_width": width, "pin": False}
    (rows,) = key
    return {"rows": rows, "pin": False}



def mimo_lifetime_pins(identities):
    """Pin indivisible rows using request replay demand, never startup heat."""
    reasons, devices = {}, {}
    for (device, *_), (key, graph) in identities.items():
        cycles = graph.get("cycles")
        requests = (sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify"))
                    if cycles is not None else int(graph["count"]))
        hits = min(int(graph.get("hits", 0)), requests)
        if hits <= 0:
            continue
        rows = devices.setdefault(device, {})
        demand = rows.setdefault(key, [0, 0])
        demand[0] += hits
        demand[1] += min(hits, int((cycles or {}).get("verify", 0)))
        if key == (1,):
            reasons.setdefault(key, set()).add("plain_c1")
    for rows in devices.values():
        total = sum(demand[0] for demand in rows.values())
        covered, cutoff = 0, None
        for key in sorted(rows, key=lambda key: (-rows[key][0], key)):
            hits = rows[key][0]
            if cutoff is not None and hits < cutoff:
                break
            reasons.setdefault(key, set()).add("lifetime_hit_tier")
            covered += hits
            if cutoff is None and covered * 2 >= total:
                cutoff = hits
        verify = [key for key in rows if rows[key][1] > 0]
        if verify:
            dominant = min(verify, key=lambda key: (-rows[key][1], key))
            reasons.setdefault(dominant, set()).add("dominant_verify")
    return reasons



def generate_mimo(traces, family, coverage=0.999, pin_frequency=0.9, pin_report=None, ranks=None, variant=None, sparks=None, arena_report=None, layout=None):
    prefix = "mimo_v2." if family == "mimo_v2" else None
    if prefix is None:
        raise ValueError(f"no checked-in shape-list consumer for {family}")
    if family == "mimo_v2" and ranks is None:
        raise ValueError("mimo_v2 needs --ranks: 1x and 2x layouts capture different graphs")
    if family == "mimo_v2" and (not variant or sparks is None):
        raise ValueError("mimo_v2 needs --variant and --sparks for the checkpoint/layout key")
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
    def geometry(shape):
        key = key_of(family, shape)
        if family != "mimo_v2":
            rows, width = key
            if scale != 1 and width % 2:
                raise ValueError(f"{family} layout {layout} key width {width} is not an even half table")
            return (rows, width * scale)
        return key

    counts = {}
    charged = {}
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
        charged[device] = max(charged.get(device, 0), int(cache["charged"]))
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
            key = geometry(graph["shape"])
            cycles = graph.get("cycles")
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) if cycles is not None else int(graph["count"])
            # MiMo prewarms a whole row geometry. A capture-only identity is
            # not replay demand; replayed siblings can still select this row.
            if family == "mimo_v2" and int(graph.get("hits", 0)) <= 0:
                continue
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
                identity = (device, graph["site"], graph.get("bank", 0), graph.get("key_id", graph["shape"]))
                identities[identity] = (geometry(graph["shape"]), graph)
    for identity, (key, graph) in identities.items():
        cycles = graph.get("cycles", {})
        if family == "mimo_v2" and int(graph.get("hits", 0)) <= 0:
            continue
        if not any(int(cycles.get(cycle, 0)) > 0 for cycle in ("decode", "verify")):
            continue
        why = reasons.setdefault(key, set())
        histogram = graph.get("hit_histogram", {})
        histogram_end = int(histogram.get("start_ms", 0)) + len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
        if histogram_end >= latest[identity[0]][0] and sustained(graph):
            why.add("sustained_heat")
        if int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2:
            why.add("repeated_recapture")
    ranked = sorted(counts, key=lambda key: (-counts[key], tuple(reversed(key))))
    total = sum(counts.values())
    covered = 0
    selected = set()
    for key in ranked:
        selected.add(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    if family == "mimo_v2":
        for key, why in mimo_lifetime_pins(identities).items():
            reasons.setdefault(key, set()).update(why)
    # A protected low-frequency key can fall outside frequency coverage.
    selected.update(key for key, why in reasons.items() if why)
    peak = max(counts.values())
    minimum = min(counts, key=lambda key: (math.prod(key), key))
    if minimum in legacy and counts[minimum] >= peak * pin_frequency:
        reasons.setdefault(minimum, set()).add("legacy_frequency")
    replay_hits = {}
    if family == "mimo_v2":
        device_hits = {}
        for (device, *_), (key, graph) in identities.items():
            cycles = graph.get("cycles")
            requests = (sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify"))
                        if cycles is not None else int(graph["count"]))
            hits = min(int(graph.get("hits", 0)), requests)
            row = (device, key)
            device_hits[row] = device_hits.get(row, 0) + hits
        for (_, key), hits in device_hits.items():
            replay_hits[key] = max(replay_hits.get(key, 0), hits)
    entries = []
    order = (lambda key: (not bool(reasons.get(key)), -replay_hits.get(key, 0), key)) \
        if family == "mimo_v2" else lambda key: tuple(reversed(key))
    for key in sorted(selected, key=order):
        entry = entry_of(family, key)
        entry["pin"] = bool(reasons.get(key))
        entries.append(entry)
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
                pin_report.append({"device": device, "rows": key[0],
                                   "table_width": key[1] if len(key) > 1 else None,
                                   "reasons": sorted(reasons[key]), "permanent_bytes": None,
                                   "observed_permanent_lower_bound_bytes": cost,
                                   "reserve_bytes": reserves[device], "reserve_fraction": None,
                                   "observed_reserve_fraction_lower_bound": cost / reserves[device] if cost is not None and reserves[device] else None,
                                   "capture_bytes": sum(int(graph.get("capture_bytes") or 0) for graph in graphs),
                                   "nodes": sum(int(graph.get("nodes") or 0) for graph in graphs),
                                   "executables": sum(int(graph.get("executables", 1)) for graph in graphs),
                                   "basis": graphs[0].get("basis"),
                                   "cost_basis": "observed_inventory_lower_bound_unknown_total" if known else "unknown_legacy_trace"})
    if family == "mimo_v2":
        devices = sorted(charged)
        if len(devices) != ranks:
            raise ValueError(f"trace has {len(devices)} device(s) for a {ranks}-rank layout")
        # Latest complete inventories contain each owner once, including cold
        # layer/head siblings of selected rows. Whole-cache high-water includes
        # unselected startup shapes and is not the proposed prewarm arena.
        for trace in traces:
            device = int(trace["cache"]["device"])
            for graph in trace["graphs"]:
                if not graph["site"].startswith(family + "."):
                    continue
                identity = (device, graph["site"], graph.get("bank", 0), graph.get("key_id", graph["shape"]))
                if key_of(family, graph["shape"]) in selected and identity not in identities:
                    raise ValueError(f"selected MiMo identity forgotten from latest inventory: {identity}")
        bases = []
        for device in devices:
            prewarm_bytes = lazy_bytes = lazy_unknown = 0
            zero_replay_keys = zero_replay_bytes = zero_replay_unknown = 0
            selected_zero_keys = selected_zero_bytes = 0
            sibling_classes = {}
            geometries = set()
            for identity, (geometry, graph) in identities.items():
                if identity[0] != device:
                    continue
                share = graph.get("capture_bytes")
                known = isinstance(share, int) and not isinstance(share, bool) and share >= 0
                replayed = int(graph.get("hits", 0)) > 0
                if not replayed:
                    zero_replay_keys += 1
                    if known:
                        zero_replay_bytes += share
                    else:
                        zero_replay_unknown += 1
                if geometry in selected:
                    # Row siblings are an indivisible physical capture unit.
                    sibling = re.sub(r"rows: \d+", "rows: *", graph["shape"])
                    group = sibling_classes.setdefault((graph["site"], sibling), [])
                    group.append((geometry, replayed))
                    if not replayed:
                        selected_zero_keys += 1
                        if known:
                            selected_zero_bytes += share
                    if not known:
                        raise ValueError(f"selected MiMo identity lacks measured capture_bytes: {identity}")
                    geometries.add(geometry)
                    prewarm_bytes += share
                elif known:
                    lazy_bytes += share
                else:
                    lazy_unknown += 1
            if selected - geometries:
                raise ValueError(f"device {device} lacks selected MiMo geometries: {sorted(selected - geometries)}")
            bases.append(packed(prewarm_bytes))
            if arena_report is not None:
                arena_report.append({"device": device, "prewarm_capture_bytes": prewarm_bytes,
                                     "prewarm_base_bytes": bases[-1], "lazy_capture_bytes": lazy_bytes,
                                     "lazy_unknown_identities": lazy_unknown,
                                     "zero_replay_keys": zero_replay_keys,
                                     "zero_replay_capture_bytes": zero_replay_bytes,
                                     "zero_replay_unknown_identities": zero_replay_unknown,
                                     "selected_zero_hit_siblings": selected_zero_keys,
                                     "selected_zero_hit_sibling_bytes": selected_zero_bytes,
                                     "always_zero_hit_sibling_classes": [
                                         {"site": site, "shape": shape}
                                         for (site, shape), values in sorted(sibling_classes.items())
                                         if {geometry for geometry, _ in values} == selected
                                         and not any(replayed for _, replayed in values)],
                                     "device_high_water_bytes": charged[device]})
        return {"schema_version": 2, "family": family, "layouts": [{
            "variant": variant, "sparks": sparks, "ranks": len(devices),
            "reserve_base_bytes": bases,
            "reserve_extra_bytes": [CHUNK for device in devices],
            "entry_capture_bytes": [[sum(graph["capture_bytes"]
                for identity, (geometry, graph) in identities.items()
                if identity[0] == device and geometry == (entry["rows"],))
                for entry in entries] for device in devices],
            "entries": entries}]}
    base = packed(max(charged.values()))
    # Runtime headroom is resolved by GPU_GRAPH_RESERVE_EXTRA, not the trace.
    extra = CHUNK
    result = {"schema_version": 1, "family": family}
    if layout is not None:
        result["layout"] = layout
    result.update({"reserve_base_bytes": base, "reserve_extra_bytes": extra, "entries": entries})
    return result



def generate(traces, family, coverage=0.999, pin_frequency=0.9, prefill_rows=None, pin_report=None, layout=None, existing=None, heat_report=None, *, ranks=None, variant=None, sparks=None, arena_report=None):
    if family == "mimo_v2":
        return generate_mimo(traces, family, coverage, pin_frequency, pin_report, ranks, variant, sparks, arena_report, layout)
    if family == "deepseek_v4":
        return generate_deepseek_v4(traces, layout, existing, coverage, pin_frequency, pin_report, heat_report)
    if family == "glm5_flash":
        if layout is not None:
            raise ValueError("--layout applies to mistral_large4 or deepseek_v41 only")
        return generate_glm5_flash(traces, coverage, pin_frequency)
    if family in SITE_PREFIX:
        return generate_glm5(traces, family, coverage, pin_frequency, pin_report, layout)
    if family != "deepseek_v41":
        raise ValueError("only glm5 and deepseek_v41 have a checked-in shape-list consumer")
    return generate_v41(traces, coverage, pin_frequency, prefill_rows, pin_report, layout)


def glmf_identity(shape):
    """A GLM Flash graph identity as its stable key tuple.

    Short steps key neither the pool top-k's width nor the pool table stride
    (`GraphGeometry::keyed`), so those identities collapse to one shape.
    """
    normalized = GLMF_NORMALIZED_GEOMETRY.fullmatch(shape.strip())
    if normalized is not None:
        rows, spec, head, long, width, page_stride, pool_stride = normalized.groups()
        if int(page_stride) == 0:
            if int(rows) <= 0 or long != "false" or int(width) != 0 or int(pool_stride) != 0 \
                    or (head == "true" and spec != "false"):
                raise ValueError("unsupported normalized GLM Flash KDA/head geometry")
            return (int(rows), spec == "true", False, 0, 0, 0)
        if head == "true":
            raise ValueError("unsupported normalized GLM Flash head geometry")
    else:
        match = GLMF_GEOMETRY.fullmatch(shape.strip())
        if match is None:
            raise ValueError(f"cannot parse glm5_flash graph identity: {shape!r}")
        rows, spec, long, width, page_stride, pool_stride = match.groups()
    rows, width, page_stride, pool_stride = (int(value) for value in (rows, width, page_stride, pool_stride))
    if rows <= 0 or page_stride <= 0:
        raise ValueError("unsupported prewarm geometry")
    if long == "false":
        width = pool_stride = 0
    elif pool_stride <= 0:
        raise ValueError("unsupported prewarm geometry")
    return (rows, spec == "true", long == "true", width, page_stride, pool_stride)


def generate_glm5_flash(traces, coverage=0.999, pin_frequency=0.9):
    """A GLM Flash startup list: every selected shape prewarms at startup, the
    always-hot ones are pinned, and the reserve base is what those selected
    shapes measured, device by device, plus the fixed 2 MiB allocation tail.
    Startup prewarm itself runs as Other, never as request demand. The
    whole-session arena in `cache.charged` also holds captures outside the
    coverage set (startup warmups and one-off shapes this launch never
    prewarms), so it is not the base. Runtime capture headroom comes from the
    shared extra keys, not the trace."""
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    traces = [{**trace, "graphs": [graph for graph in trace["graphs"]
                                  if not graph.get("permanently_eager", False)]}
              for trace in traces]
    styles = {GLMF_NORMALIZED_GEOMETRY.fullmatch(graph["shape"].strip()) is not None
              for trace in traces for graph in trace["graphs"]
              if graph["site"].startswith("glm5_flash.")}
    if len(styles) > 1:
        raise ValueError("GLM Flash traces must use one graph-key format")
    normalized = True in styles
    if normalized:
        latest = {}
        for trace in traces:
            device = int(trace["cache"].get("device", 0))
            stamp = int(trace["cache"].get("sampled_at_ms", 0))
            if device not in latest or stamp >= latest[device][0]:
                latest[device] = (stamp, trace)
        traces = [trace for _, trace in latest.values()]

    def compatible(key, graph, step):
        if not normalized or key[4] != 0:
            return key == step
        head = GLMF_NORMALIZED_GEOMETRY.fullmatch(graph["shape"].strip())[3] == "true"
        return key[0] == step[0] and (head or key[1] == step[1])

    counts = {}
    for trace in traces:
        if trace.get("schema_version") != 2:
            raise ValueError("expected GRAPH_TRACE schema_version=2")
        matching = [graph for graph in trace["graphs"] if graph["site"].startswith("glm5_flash.")]
        device_counts = {}
        for graph in matching:
            key = glmf_identity(graph["shape"])
            count = glm_request_replays(graph)
            if count > 0 and (not normalized or key[4] > 0):
                device_counts[key] = device_counts.get(key, 0) + count
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
    if not counts:
        raise ValueError("trace contains no matching graph identities")
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    total = sum(counts.values())
    covered = 0
    selected = []
    for key in ranked:
        selected.append(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    peak = max(counts.values())
    pinned = {key for key in selected if counts[key] >= peak * pin_frequency}
    entries = [{"rows": key[0], "spec": key[1], "long": key[2], "pool_width": key[3],
                "page_stride": key[4], "pool_stride": key[5], "pin": key in pinned}
               for key in sorted(selected)]
    # Price exactly the entries this list prewarms: sum each selected shape's
    # measured capture bytes on one device, take the costliest quant traced for
    # that device, and keep one base per coordinator device. A selected shape
    # with no measured share means the trace cannot price this list; it is
    # missing data, never a zero.
    wanted = set(selected)
    shares = {}
    for trace in traces:
        device = int(trace["cache"].get("device", 0))
        seen = set()
        captured = 0
        executables = 0
        identities = set()
        for graph in trace["graphs"]:
            if not graph["site"].startswith("glm5_flash."):
                continue
            key = glmf_identity(graph["shape"])
            if not any(compatible(key, graph, step) for step in wanted):
                continue
            if normalized:
                identity = (graph["site"], graph.get("bank", 0), graph.get("key_id", graph["shape"]))
                if identity in identities:
                    continue
                identities.add(identity)
            capture = graph.get("capture_bytes")
            if capture is None:
                raise ValueError(f"GLM Flash selected shape {key} has no measured capture bytes "
                                 f"on device {device}")
            if key in wanted:
                seen.add(key)
            captured += int(capture)
            executables += int(graph.get("executables", 1) or 0)
        if seen != wanted:
            raise ValueError(f"GLM Flash trace for device {device} lacks selected shapes "
                             f"{sorted(wanted - seen)}; its measured base is unknown")
        if captured > 0 and executables > 0:
            previous = shares.get(device)
            if previous is None or (packed(captured), executables) > previous:
                shares[device] = (packed(captured), executables)
    if not shares:
        raise ValueError("trace has no measured GLM Flash arena")
    devices = sorted(shares)
    if devices != list(range(len(devices))):
        raise ValueError(f"GLM Flash traces must cover devices 0..n-1; saw {devices}")
    measured_bytes = [shares[device][0] for device in devices]
    measured_executables = [shares[device][1] for device in devices]
    # Startup prewarms every selected shape, so the reserve base already holds
    # the whole request demand. The fixed allocation tail is the arena's own
    # headroom for its single physical allocation; runtime capture headroom is
    # the shared extra keys' (never a trace-derived largest capture).
    extra = CHUNK
    return {"schema_version": 1, "family": "glm5_flash",
            "measured_bytes": measured_bytes, "measured_executables": measured_executables,
            "reserve_extra_bytes": extra, "entries": entries}


MUSE_GEOMETRY = re.compile(r"Key \{ layer: \d+, rows: (\d+), stride: \d+, section: (\w+) \}")


def generate_muse(traces, ranks, onboard_layers):
    """Select every replayed row geometry and price all its captured siblings."""
    if ranks not in (1, 2) or type(onboard_layers) is not int or not 0 <= onboard_layers <= 62:
        raise ValueError("Muse needs --ranks 1|2 and --onboard-layers 0..62")
    latest = {}
    for trace in traces:
        if trace.get("schema_version") not in (1, 2):
            raise ValueError("expected GRAPH_TRACE schema_version=1 or 2")
        device = int(trace["cache"]["device"])
        timestamp = int(trace["cache"].get("sampled_at_ms", 0))
        if device not in latest or timestamp >= latest[device][0]:
            latest[device] = (timestamp, trace)
    devices = sorted(latest)
    if len(devices) != ranks:
        raise ValueError(f"Muse {ranks} ranks needs {ranks} device traces, found {devices}")
    identities = {}
    counts = {}
    pins = set()
    for device, (_, trace) in latest.items():
        for graph in trace["graphs"]:
            if graph["site"] != "muse_spark.segment":
                continue
            match = MUSE_GEOMETRY.fullmatch(graph["shape"])
            if match is None or not 1 <= int(match[1]) <= 64:
                raise ValueError(f"cannot parse Muse identity: {graph['shape']!r}")
            rows = int(match[1])
            identity = (device, graph.get("bank", 0), graph.get("key_id", graph["shape"]))
            if identity in identities:
                continue
            identities[identity] = (rows, graph)
            replay = qwen_replay(graph)
            if replay > 0:
                counts[rows] = max(counts.get(rows, 0), replay)
                if sustained(graph):
                    pins.add(rows)
    if not counts:
        raise ValueError("trace records no replayed Muse graph identities")
    selected = set(counts)
    costs = {rows: [0] * ranks for rows in selected}
    zero_siblings = zero_lazy = 0
    for (device, _, _), (rows, graph) in identities.items():
        if rows not in selected:
            zero_lazy += 1
            continue
        share = graph.get("capture_bytes")
        if share is None or int(share) < 0:
            raise ValueError(f"Muse selected identity lacks measured capture share: {graph['shape']!r}")
        costs[rows][devices.index(device)] += int(share)
        zero_siblings += qwen_replay(graph) == 0
    # The emitted startup step captures all layer/head siblings at this row count.
    # Missing rank costs indicate an incomplete trace, not free graphs.
    if any(cost <= 0 for costs_by_rank in costs.values() for cost in costs_by_rank):
        raise ValueError("Muse selected geometry has no measured capture cost on a rank")
    entries = [{"rows": rows, "pin": rows in pins, "capture_bytes": costs[rows]}
               for rows in sorted(selected, key=lambda rows: (rows not in pins, -counts[rows], rows))]
    return {"ranks": ranks, "onboard_layers": onboard_layers,
            "provenance": f"GRAPH_TRACE latest inventories on devices {devices}; every replayed row geometry; "
                          f"{zero_siblings} zero-replay siblings included; {zero_lazy} zero-replay identities stay lazy",
            "reserve_base_bytes": [packed(sum(cost[rank] for cost in costs.values())) for rank in range(ranks)],
            "reserve_extra_bytes": [CHUNK] * ranks, "entries": entries}


def qwen_geometry(graph):
    match = QWEN_NORMALIZED_GEOMETRY.fullmatch(graph["shape"])
    if match is not None:
        segment, rows, spec, head, long, page_width, page_stride, pool_stride = match.groups()
        rows, page_width, page_stride, pool_stride = map(int, (rows, page_width, page_stride, pool_stride))
        if rows <= 0 or page_width != page_stride:
            raise ValueError("unsupported normalized Qwen prewarm geometry")
        if page_stride == 0:
            if long != "false" or pool_stride != 0 or (head == "true" and spec != "false"):
                raise ValueError("invalid normalized Qwen GDN/head geometry")
        elif head == "true" or (long == "true") != (pool_stride > 0):
            raise ValueError("invalid normalized Qwen Full geometry")
        return (rows, spec == "true", long == "true", 0, page_stride, pool_stride)
    match = QWEN_GEOMETRY.search(graph["shape"])
    if match is None:
        raise ValueError(f"cannot parse qwen4 identity: {graph['shape']!r}")
    segment, rows, spec, long, width, page_stride, pool_stride = match.groups()
    if int(rows) <= 0 or int(page_stride) <= 0 or int(pool_stride) <= 0:
        raise ValueError("unsupported prewarm geometry")
    return (int(rows), spec == "true", long == "true", int(width),
            int(page_stride), int(pool_stride))


def qwen_replay(graph):
    """Replayed launches for one qwen4 graph identity (0 = captured, never used)."""
    replay = graph.get("hits")
    if replay is None:
        cycles = graph.get("cycles")
        replay = (sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify"))
                  if cycles is not None else int(graph["count"])) - int(graph.get("misses", 0))
    return int(replay)


def generate_qwen(traces, shape, coverage=0.999, pin_report=None, summary=None):
    """One `QwenShapePrewarm` object for a resolved capture shape.

    Only identities with at least one replay can be selected: a capture is an
    event, not demand, so a shape captured once and never replayed stays lazy
    and its count and bytes are reported. The reserve base is the traced
    working set's arena: each identity of a selected shape's measured
    `capture_bytes` (every segment row, replayed or a startup-only sibling),
    summed per device, deduplicated by identity, rounded once to the 2 MiB
    chunk, and a selected identity without a measured share fails. The trace's
    charged high-water is never the base; `reserve_extra_bytes` is the fixed
    2 MiB capture headroom.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    traces = [{**trace, "graphs": [graph for graph in trace["graphs"]
                                  if not graph.get("permanently_eager", False)]}
              for trace in traces]
    counts = {}
    reserves = {}
    reasons = {}
    identities = {}
    latest = {}
    devices = set()
    for trace in traces:
        version = trace.get("schema_version")
        if version not in (1, 2):
            raise ValueError("expected GRAPH_TRACE schema_version=1 or 2")
        cache = trace["cache"]
        device = int(cache["device"])
        devices.add(device)
        reserves[device] = max(reserves.get(device, 0), int(cache.get("reserve", 0)))
        inferred_end = max((int(graph.get("hit_histogram", {}).get("start_ms", 0)) +
                            len(graph.get("hit_histogram", {}).get("counts", [])) *
                            int(graph.get("hit_histogram", {}).get("interval_ms", 0))
                            for graph in trace["graphs"]), default=0)
        snapshot_time = int(cache.get("sampled_at_ms", inferred_end))
        if device not in latest or snapshot_time >= latest[device][0]:
            latest[device] = (snapshot_time, trace)
    owners = 1 if shape.startswith("single") else 2
    devices = sorted(devices)
    if len(devices) != owners:
        raise ValueError(f"{shape} needs {owners} device traces, found {devices}")
    styles = {QWEN_NORMALIZED_GEOMETRY.fullmatch(graph["shape"]) is not None
              for _, trace in latest.values() for graph in trace["graphs"]
              if graph["site"].startswith("qwen4.")}
    if len(styles) != 1:
        raise ValueError("Qwen traces must use one graph-key format")
    normalized = True in styles

    def compatible(key, graph, step):
        if not normalized or key[4] != 0:
            return key == step
        # GDN has only rows/spec; the explicit terminal head ignores spec too.
        head = QWEN_NORMALIZED_GEOMETRY.fullmatch(graph["shape"])[4] == "true"
        return key[0] == step[0] and (head or key[1] == step[1])

    # Frequency and heat use only each device's latest complete inventory:
    # identities missing from it have been forgotten and must not survive.
    # Only replayed graphs rank the working set: a startup capture is an event
    # (and a miss), never demand, so a zero-hit bootstrap shape can never be
    # selected (#1652).
    for device, (_, trace) in latest.items():
        for graph in trace["graphs"]:
            if not graph["site"].startswith("qwen4."):
                continue
            geometry = qwen_geometry(graph)
            identity = (device, graph["site"], graph.get("bank", 0),
                        graph.get("key_id", graph["shape"]))
            previous = identities.get(identity)
            if previous is None or (previous[1].get("capture_bytes") is None
                                    and graph.get("capture_bytes") is not None):
                identities[identity] = (geometry, graph)
            cycles = graph.get("cycles")
            replay = qwen_replay(graph)
            # Every layer segment runs once per decode step: rank a shape by its
            # busiest segment rather than summing all of them.
            if replay > 0 and (not normalized or geometry[4] > 0):
                counts[geometry] = max(counts.get(geometry, 0), replay)
    if not counts:
        raise ValueError("trace records no replayed graph identities")
    # A captured-but-never-replayed identity is first touch, not demand: it is
    # never selected, and its count and bytes are reported rather than hidden.
    zero_replay = [graph for (_, graph) in identities.values() if qwen_replay(graph) == 0]
    zero_replay_bytes = sum(int(graph.get("capture_bytes") or 0) for graph in zero_replay)
    if summary is not None:
        summary["zero_replay_identities"] = len(zero_replay)
        summary["zero_replay_bytes"] = zero_replay_bytes
    # A completed capture boundary with no new physical growth is allocator
    # reuse, not a free graph: the shared cache keeps such a record's estimate
    # for exactly this case. Charge a selected identity whose measured share is
    # zero the launch's measured mean capture on its device, so the traced
    # working set's base is never undercounted by the proportional split.
    measured = {}
    for (device, _, _, _), (_, graph) in identities.items():
        share = graph.get("capture_bytes")
        if share is None:
            continue
        captures, total = measured.get(device, (0, 0))
        measured[device] = (captures + max(int(graph.get("captures") or 0), 1),
                            total + int(share))
    mean = {device: total / captures for device, (captures, total) in measured.items()}
    for identity, (key, graph) in identities.items():
        cycles = graph.get("cycles", {})
        if not any(int(cycles.get(cycle, 0)) > 0 for cycle in ("decode", "verify")):
            continue
        histogram = graph.get("hit_histogram", {})
        histogram_end = int(histogram.get("start_ms", 0)) + \
            len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
        if histogram_end >= latest[identity[0]][0] and sustained(graph):
            reasons.setdefault(key, set()).add("sustained_heat")
        if int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2:
            reasons.setdefault(key, set()).add("repeated_recapture")
    if normalized:
        for _, (key, graph) in identities.items():
            for step in counts:
                if key in reasons and compatible(key, graph, step):
                    reasons.setdefault(step, set()).update(reasons[key])
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    total = sum(counts.values())
    covered = 0
    selected = set()
    for key in ranked:
        selected.add(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    # A thrashing low-frequency key can fall outside frequency coverage, but a
    # zero-hit startup shape is never demand and must not re-enter this way.
    selected.update(key for key, why in reasons.items() if why and key in counts)
    entries = [{"rows": rows, "spec": spec, "long": long, "pool_width": width,
                "page_stride": page_stride, "pool_stride": pool_stride,
                "pin": bool(reasons.get(key))}
               for key in sorted(selected)
               for rows, spec, long, width, page_stride, pool_stride in [key]]
    if pin_report is not None:
        for key in sorted(selected):
            if not reasons.get(key):
                continue
            for device in devices:
                graphs = [graph for identity, (geometry, graph) in identities.items()
                          if identity[0] == device and compatible(geometry, graph, key)]
                if not graphs:
                    continue
                cost = sum(int(graph["permanent_bytes"]) for graph in graphs) \
                    if all("permanent_bytes" in graph for graph in graphs) else None
                pin_report.append({"device": device, "shape": qwen_shape_text(key),
                                   "reasons": sorted(reasons[key]), "permanent_bytes": None,
                                   "observed_permanent_lower_bound_bytes": cost,
                                   "reserve_bytes": reserves[device], "reserve_fraction": None,
                                   "observed_reserve_fraction_lower_bound": cost / reserves[device] if cost is not None and reserves[device] else None,
                                   "capture_bytes": sum(int(graph["capture_bytes"]) for graph in graphs
                                                        if graph.get("capture_bytes") is not None),
                                   "null_capture_bytes": sum(1 for graph in graphs
                                                             if graph.get("capture_bytes") is None),
                                   "nodes": sum(int(graph.get("nodes") or 0) for graph in graphs),
                                   "executables": sum(int(graph.get("executables", 1)) for graph in graphs),
                                   "basis": graphs[0].get("basis"),
                                   "cost_basis": "observed_inventory_lower_bound_unknown_total" if cost is not None else "unknown_legacy_trace"})
    base = []
    zero_share = []
    for device in devices:
        total = 0
        charged = set()
        for selected_key in sorted(selected):
            for identity, (geometry, graph) in identities.items():
                if (identity[0] != device or identity in charged
                        or not compatible(geometry, graph, selected_key)):
                    continue
                charged.add(identity)
                share = graph.get("capture_bytes")
                if share is None:
                    raise ValueError(
                        f"{shape} device {device} selects {qwen_shape_text(selected_key)} "
                        f"without a measured capture share: {graph['shape']!r}")
                share = int(share)
                if share == 0:
                    share = round(mean.get(device, 0.0))
                    zero_share.append(identity)
                total += share
        base.append(packed(total))
    zero_share_bytes = sum(round(mean.get(identity[0], 0.0)) for identity in zero_share)
    if summary is not None:
        summary["zero_share_identities"] = len(zero_share)
        summary["zero_share_bytes"] = zero_share_bytes
    extra = [EXTRA_BYTES for _ in devices]
    provenance = (f"GRAPH_TRACE {len(traces)} snapshots at {shape}; devices {devices}; "
                  f"{len(selected)} selected shapes; capture base {base} + {EXTRA_BYTES} extra bytes; "
                  f"{len(zero_replay)} zero-replay identities ({zero_replay_bytes} bytes) stay lazy; "
                  f"{len(zero_share)} zero-share selected identities ({zero_share_bytes} bytes) "
                  f"charged at the device's measured mean capture")
    return {"shape": shape, "reserve_base_bytes": base,
            "reserve_extra_bytes": extra,
            "trace": {"provenance": provenance, "entries": entries}}


def qwen_shape_text(key):
    rows, spec, long, width, page_stride, pool_stride = key
    return (f"rows={rows} spec={spec} long={long} pool_width={width} "
            f"page_stride={page_stride} pool_stride={pool_stride}")


def qwen_provenance(shapes):
    """The list document's provenance sentence for its current traced set."""
    traced = [prewarm["shape"] for prewarm in shapes if prewarm.get("trace") is not None]
    untraced = [prewarm["shape"] for prewarm in shapes if prewarm.get("trace") is None]
    text = ("Traced shapes regenerate from their own GRAPH_TRACE snapshots ("
            + ", ".join(traced) + ") with scripts/dev/graph-prewarm-from-trace.py; "
            "a traced shape's base is the per-device sum of its selected prewarm "
            "identities' measured capture bytes, including startup-only siblings, "
            "deduplicated by identity and rounded once to 2 MiB; selection ranks by "
            "replayed hits only, so a zero-hit startup shape is never selected and a "
            "captured-but-never-replayed identity stays lazy. Extra "
            "is the fixed 2 MiB capture headroom; the shared graph reserve extra adds "
            "room for runtime captures.")
    if untraced:
        text += (" Untraced (" + ", ".join(untraced) + ") keeps the bootstrap reserve "
                 "above and prewarms the 682-shape bootstrap startup set at launch with "
                 "a warning; an untraced shape never borrows another shape's list.")
    return text


def merge_qwen(document, generated):
    """Install one generated shape object into a checked-in prewarm list.

    Every other shape keeps its reserve and trace untouched, so a shape is only
    ever replaced by its own trace. The provenance sentence is derived from the
    installed set, which keeps the document reproducible from the generator.
    """
    merged = json.loads(json.dumps(document))
    shape = generated["shape"]
    if not any(prewarm["shape"] == shape for prewarm in merged["shapes"]):
        raise ValueError(f"prewarm list has no {shape} shape")
    merged["shapes"] = [generated if prewarm["shape"] == shape else prewarm
                        for prewarm in merged["shapes"]]
    merged["provenance"] = qwen_provenance(merged["shapes"])
    return merged


def layout_groups(specs, family):
    """Group `--trace` arguments by layout: `LAYOUT=PATH` for GLM Flash, a bare
    path (the one layout) for GLM. A GLM Flash key names a resolved card shape,
    `<devices>[-sparks]`; a shape left out warms the engine's bootstrap set, so
    a trace is required only for the shapes this run measured."""
    groups = {}
    for spec in specs:
        layout, path = spec.split("=", 1) if "=" in spec else ("min", spec)
        if family != "glm5_flash" and layout != "min":
            raise ValueError(f"{family} has one layout, not {layout!r}")
        if family == "glm5_flash" and layout not in GLM5_FLASH_LAYOUTS:
            raise ValueError(f"unknown GLM Flash layout {layout!r}; expected one of {sorted(GLM5_FLASH_LAYOUTS)}")
        groups.setdefault(layout, []).append(Path(path))
    return groups


def generate_layouts(groups, family, coverage=0.999, pin_frequency=0.9, pin_report=None, layout=None):
    """The checked-in document: one entry per traced shape, each priced from its
    own traces. A shape's pins and arena describe that shape's traffic; absent
    keys fall back to the engine's bootstrap set."""
    layouts = {name: generate([json.loads(path.read_text()) for path in paths],
                              family, coverage, pin_frequency, pin_report=pin_report, layout=layout)
               for name, paths in sorted(groups.items())}
    if family != "glm5_flash":
        return layouts.pop("min")
    for values in layouts.values():
        for key in ("schema_version", "family"):
            values.pop(key)
    return {"schema_version": 3, "family": family, "layouts": layouts}


DENSE_QWEN_GEOMETRY = re.compile(
    r"Key \{ segment: (\d+), rows: (\d+), table_stride: (\d+), spec: (true|false) \}")


def generate_qwen35(traces, variant, summary=None):
    """Every replayed complete-layer geometry, heat ordered, single RTX only."""
    if not variant or not variant.endswith("-rtx1"):
        raise ValueError("dense Qwen needs --variant ending in -rtx1")
    latest = {}
    for trace in traces:
        if trace.get("schema_version") != 2:
            raise ValueError("dense Qwen requires trace schema 2")
        device = int(trace["cache"]["device"])
        time = int(trace["cache"].get("sampled_at_ms", 0))
        if device not in latest or time >= latest[device][0]:
            latest[device] = (time, trace)
    if len(latest) != 1:
        raise ValueError("dense Qwen supports exactly one RTX")
    identities = {}
    counts = {}
    pins = set()
    for _, trace in latest.values():
        for graph in trace["graphs"]:
            if graph["site"] != "qwen3_5.segment":
                continue
            match = DENSE_QWEN_GEOMETRY.fullmatch(graph["shape"])
            if match is None:
                raise ValueError(f"unknown dense Qwen key: {graph['shape']}")
            _, rows, stride, spec = match.groups()
            key = (int(rows), int(stride), spec == "true")
            if not 1 <= key[0] <= 64 or key[1] < 1 or key[1] & (key[1] - 1):
                raise ValueError(f"invalid dense Qwen geometry: {key}")
            if graph.get("nodes") is None and not graph.get("captures", 0) and not graph.get("hits", 0):
                continue  # A denied cold miss has no executable topology or capture share.
            if graph.get("nodes") is None or int(graph["nodes"]) < 2:
                raise ValueError("dense Qwen must not retain singleton/unknown graphs")
            identity = (graph.get("bank", 0), graph.get("key_id", graph["shape"]))
            identities[identity] = (key, graph)
            hits = int(graph.get("hits", 0))
            if hits > 0:
                counts[key] = max(counts.get(key, 0), hits)
                if sustained(graph):
                    pins.add(key)
    if not counts:
        raise ValueError("dense Qwen trace has no replayed geometry")
    selected = [(key, graph) for key, graph in identities.values() if key in counts]
    if any(graph.get("capture_bytes") is None for _, graph in selected):
        raise ValueError("selected dense Qwen geometry has no measured capture share")
    measured = [int(graph["capture_bytes"]) for _, graph in identities.values()
                if graph.get("capture_bytes") is not None and int(graph["capture_bytes"]) > 0]
    mean = math.ceil(sum(measured) / len(measured)) if measured else 0
    base = packed(sum(int(graph["capture_bytes"]) or mean for _, graph in selected))
    if summary is not None:
        lazy = [graph for key, graph in identities.values() if key not in counts]
        siblings = [graph for _, graph in selected if not int(graph.get("hits", 0))]
        zero_share = [graph for _, graph in selected if not int(graph["capture_bytes"])]
        segments = {int(DENSE_QWEN_GEOMETRY.fullmatch(graph["shape"])[1])
                    for _, graph in selected}
        replayed_segments = {int(DENSE_QWEN_GEOMETRY.fullmatch(graph["shape"])[1])
                             for _, graph in selected if int(graph.get("hits", 0)) > 0}
        summary.update(
            zero_replay_identities=len(lazy),
            zero_replay_bytes=sum(int(graph.get("capture_bytes") or 0) for graph in lazy),
            selected_zero_hit_siblings=len(siblings),
            selected_zero_hit_sibling_bytes=sum(int(graph["capture_bytes"]) or mean for graph in siblings),
            zero_share_identities=len(zero_share), zero_share_bytes=len(zero_share) * mean,
            always_zero_hit_segments=sorted(segments - replayed_segments))
    keys = sorted(counts, key=lambda key: (key not in pins, -counts[key], key))
    return {"schema_version": 1, "family": "qwen3_5", "layouts": [{
        "variant": variant, "traced": True, "reserve_base_bytes": base,
        "reserve_extra_bytes": EXTRA_BYTES,
        "entries": [{"rows": rows, "table_stride": stride, "spec": spec,
                     "pin": (rows, stride, spec) in pins} for rows, stride, spec in keys]}]}


STEP5_LAYOUTS = {"fp8-2rtx-6spark": 2, "nvfp4-1rtx-4spark": 1,
                 "exl3-k4-core-k6-1rtx-3spark": 1, "exl3-k4-core-k6-2rtx-6spark": 2}
STEP5_GEOMETRY = re.compile(r"Key \{ layer: (\d+), rows: (\d+), section: "
                            r"(Projection|Attention|Dense|Frontend|Shared|Finish) \}")

HY4_LAYOUTS = {
    "exl3-k4-mc-1rtx-0spark": 1,
    "exl3-k4-mc-2rtx-0spark": 2,
    "exl3-k4-mc-1rtx-spark": 1,
    "exl3-k4-mc-2rtx-spark": 2,
}
HY4_GEOMETRY = re.compile(
    r"Key \{ layer: (\d+), rows: (\d+), section: "
    r"(SwaLocal|SwaAttn|SwaExtendPre|SwaExtendPost|DsaPre|DsaIndexQ|DsaLocalPost|DsaAttnOut|FfnFrontend|FfnExpert|FfnFinish) \}"
)


def generate_hy4_flash(traces, layout):
    """Whole replayed row geometries, per rank; never borrow another layout."""
    if layout not in HY4_LAYOUTS:
        raise ValueError(f"unknown Hy4-flash layout {layout!r}")
    latest = {}
    for trace in traces:
        if trace.get("schema_version") != 2:
            raise ValueError("Hy4-flash requires GRAPH_TRACE schema_version=2")
        device = int(trace["cache"]["device"])
        sampled = int(trace["cache"]["sampled_at_ms"])
        if device not in latest or sampled >= latest[device][0]:
            latest[device] = (sampled, trace)
    if len(latest) != HY4_LAYOUTS[layout]:
        raise ValueError(f"{layout} requires {HY4_LAYOUTS[layout]} device traces")
    rank_identities = []
    for rank, device in enumerate(sorted(latest)):
        _, trace = latest[device]
        identities = {}
        for graph in trace["graphs"]:
            if graph["site"] != "hy4_flash.section":
                continue
            match = HY4_GEOMETRY.fullmatch(graph["shape"])
            if match is None:
                raise ValueError(f"cannot parse Hy4-flash identity: {graph['shape']!r}")
            layer, rows, section = match.groups()
            key = (int(layer), int(rows), section)
            if key[0] > 42 or key[1] <= 0:
                raise ValueError(f"unsupported Hy4-flash geometry {key}")
            if int(graph.get("nodes") or 0) < 2 or graph.get("spans_devices", False):
                raise ValueError(f"invalid local Hy4-flash section topology {key}")
            capture = graph.get("capture_bytes")
            if capture is None or int(capture) < 0:
                raise ValueError(f"Hy4-flash section {key} lacks measured capture bytes")
            if key in identities:
                raise ValueError(f"duplicate Hy4-flash exact geometry {key} on device {device}")
            identities[key] = graph
        rank_identities.append(identities)
    all_graphs = [(key, graph) for identities in rank_identities for key, graph in identities.items()]
    rows = {key[1] for key, graph in all_graphs if qwen_replay(graph) >= 1}
    if not rows:
        raise ValueError("Hy4-flash layout has no replayed row geometries")
    heat = {row: sum(qwen_replay(graph) for key, graph in all_graphs if key[1] == row) for row in rows}
    hot = {
        row: any(
            sustained(graph)
            or (int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2)
            for key, graph in all_graphs
            if key[1] == row and qwen_replay(graph) >= 1
        )
        for row in rows
    }
    ordered_rows = sorted(rows, key=lambda row: (not hot[row], -heat[row], row))
    ranks = []
    for rank, inventory in enumerate(rank_identities):
        identities = {key: graph for key, graph in inventory.items() if key[1] in rows}
        if {key[1] for key in identities} != rows:
            raise ValueError(f"Hy4-flash rank {rank} lacks a selected row geometry")
        total_measured = sum(int(g["capture_bytes"]) for g in identities.values())
        if total_measured <= 0:
            raise ValueError(f"Hy4-flash rank {rank} has no positive measured capture shares")
        # CUDA allocates graph storage in 2 MiB slabs; completed_capture records per-section
        # slab jumps (0 or 2 MiB), so sum(capture_bytes) is the exact physical arena cost of
        # the selected set rather than multiplying a 2 MiB jump across zero-delta siblings.
        amortized = math.ceil(total_measured / len(identities))
        costs = {key: int(graph["capture_bytes"]) for key, graph in identities.items()}
        entries = [
            {
                "rows": row,
                "pin": hot[row],
                "replay_hits": heat[row],
                "capture_bytes": sum(costs[key] for key in identities if key[1] == row),
                "sections": [
                    {
                        "layer": key[0],
                        "section": key[2],
                        "capture_bytes": costs[key],
                        "replay_hits": qwen_replay(identities[key]),
                    }
                    for key in sorted(identities)
                    if key[1] == row
                ],
            }
            for row in ordered_rows
        ]
        ranks.append(
            {
                "rank": rank,
                "reserve_base_bytes": packed(total_measured),
                "reserve_extra_bytes": CHUNK,
                "entries": entries,
                "zero_replay_sibling_bytes": sum(
                    costs[key] or amortized for key, graph in identities.items() if qwen_replay(graph) == 0
                ),
                "lazy_zero_replay_bytes": sum(
                    int(graph["capture_bytes"]) for key, graph in inventory.items() if key[1] not in rows
                ),
            }
        )
    return {"schema_version": 1, "family": "hy4_flash", "layout": layout, "ranks": ranks}


INKLING_GEOMETRY = re.compile(
    r"Key \{ layer: (\d+), rows: (\d+), section: "
    r"(WholeAttention|RankAttention|AfterAttention|Routes|ExpertFinish), mode: (Decode|Verify) \}"
)


def generate_inkling2(bands, layout, ranks):
    """Each band is an isolated same-layout launch with its own exact cycle denominator.

    A snapshot has cumulative hits, not concurrency. Supply one final snapshot per
    device in a band and the number of target decode+verify cycles in that launch.
    A donor re-key or ARC eviction can discard earlier hits: the retained counts
    are lower bounds, suitable for conservative pins, not exact frequencies.
    """
    if not layout or ranks not in (1, 2) or not bands:
        raise ValueError("Inkling needs --layout, --ranks 1|2, and one or more bands")
    rank_costs = [{} for _ in range(ranks)]
    rank_hits = [{} for _ in range(ranks)]
    rank_pins = [set() for _ in range(ranks)]
    seen_bands = set()
    for band, cycles, traces in bands:
        if not band or band in seen_bands or type(cycles) is not int or cycles <= 0:
            raise ValueError(f"invalid or duplicate Inkling band {band!r}")
        seen_bands.add(band)
        ordered = sorted(traces, key=lambda t: int(t["cache"]["device"]))
        if len(ordered) != ranks or len({int(t["cache"]["device"]) for t in ordered}) != ranks:
            raise ValueError(f"Inkling band {band} needs one snapshot per rank")
        for rank, trace in enumerate(ordered):
            if trace.get("schema_version") != 2:
                raise ValueError("Inkling needs GRAPH_TRACE schema_version=2")
            seen = set()
            for graph in trace["graphs"]:
                if graph["site"] != "inkling2.section" or qwen_replay(graph) == 0:
                    continue
                match = INKLING_GEOMETRY.fullmatch(graph["shape"])
                if match is None:
                    raise ValueError(f"invalid Inkling graph identity {graph['shape']!r}")
                layer, rows, section, mode = match.groups()
                key = (int(layer), int(rows), section, mode)
                if (not 0 <= key[0] < 48 or not 1 <= key[1] <= 64 or key in seen
                        or int(graph.get("nodes") or 0) < 3 or graph.get("spans_devices", False)):
                    raise ValueError(f"invalid or duplicate Inkling graph {key} on rank {rank}")
                seen.add(key)
                hits = qwen_replay(graph)
                capture = graph.get("capture_bytes")
                if not 0 < hits <= cycles or type(capture) is not int or capture < 0:
                    raise ValueError(f"invalid Inkling capture/replay accounting for {key} in {band}")
                rank_costs[rank][key] = max(capture, rank_costs[rank].get(key, 0))
                rank_hits[rank][key] = rank_hits[rank].get(key, 0) + hits
                if 2 * hits >= cycles:
                    rank_pins[rank].add(key)
    devices = []
    for rank, (costs, hits, pins) in enumerate(zip(rank_costs, rank_hits, rank_pins)):
        if not costs or not sum(costs.values()):
            raise ValueError(f"Inkling rank {rank} needs measured replayed captures")
        entries = [dict(layer=key[0], rows=key[1], section=key[2], mode=key[3], pin=key in pins,
                        capture_bytes=costs[key], replay_hits=hits[key])
                   for key in sorted(costs, key=lambda key: (key not in pins, -hits[key], key))]
        devices.append(dict(rank=rank, reserve_base_bytes=packed(sum(costs.values())),
                            reserve_extra_bytes=CHUNK, entries=entries))
    return dict(schema_version=1, family="inkling2", layout=layout, ranks=devices)


def inkling2_bands(specs):
    """BAND:CYCLES=RANK0.json[,RANK1.json]; never infer bands from time windows."""
    for spec in specs:
        header, paths = spec.split("=", 1)
        band, cycles = header.rsplit(":", 1)
        yield band, int(cycles), [json.loads(Path(path).read_text()) for path in paths.split(",")]


def generate_step5(traces, layout):
    """Whole replayed row geometries, per rank; never borrow another layout."""
    if layout not in STEP5_LAYOUTS:
        raise ValueError(f"unknown Step-5 layout {layout!r}")
    latest = {}
    for trace in traces:
        if trace.get("schema_version") != 2:
            raise ValueError("Step-5 requires GRAPH_TRACE schema_version=2")
        device = int(trace["cache"]["device"])
        sampled = int(trace["cache"]["sampled_at_ms"])
        if device not in latest or sampled >= latest[device][0]:
            latest[device] = (sampled, trace)
    if len(latest) != STEP5_LAYOUTS[layout]:
        raise ValueError(f"{layout} requires {STEP5_LAYOUTS[layout]} device traces")
    rank_identities = []
    for rank, device in enumerate(sorted(latest)):
        _, trace = latest[device]
        identities = {}
        for graph in trace["graphs"]:
            if graph["site"] != "step5.section":
                continue
            match = STEP5_GEOMETRY.fullmatch(graph["shape"])
            if match is None:
                raise ValueError(f"cannot parse Step-5 identity: {graph['shape']!r}")
            layer, rows, section = match.groups()
            key = (int(layer), int(rows), section)
            if key[0] >= 92 or key[1] <= 0:
                raise ValueError(f"unsupported Step-5 geometry {key}")
            if int(graph.get("nodes") or 0) < 2 or graph.get("spans_devices", False):
                raise ValueError(f"invalid local Step-5 section topology {key}")
            capture = graph.get("capture_bytes")
            if capture is None or int(capture) < 0:
                raise ValueError(f"Step-5 section {key} lacks measured capture bytes")
            if key in identities:
                raise ValueError(f"duplicate Step-5 exact geometry {key} on device {device}")
            identities[key] = graph
        rank_identities.append(identities)
    all_graphs = [(key, graph) for identities in rank_identities for key, graph in identities.items()]
    rows = {key[1] for key, graph in all_graphs if qwen_replay(graph) >= 1}
    if not rows:
        raise ValueError("Step-5 layout has no replayed row geometries")
    heat = {row: sum(qwen_replay(graph) for key, graph in all_graphs if key[1] == row) for row in rows}
    hot = {row: any(sustained(graph) or (int(graph.get("evictions", 0)) >= 2
                    and int(graph.get("recaptures", 0)) >= 2)
                    for key, graph in all_graphs if key[1] == row and qwen_replay(graph) >= 1) for row in rows}
    ordered_rows = sorted(rows, key=lambda row: (not hot[row], -heat[row], row))
    ranks = []
    for rank, inventory in enumerate(rank_identities):
        identities = {key: graph for key, graph in inventory.items() if key[1] in rows}
        if {key[1] for key in identities} != rows:
            raise ValueError(f"Step-5 rank {rank} lacks a selected row geometry")
        positive = [int(g["capture_bytes"]) for g in identities.values() if int(g["capture_bytes"]) > 0]
        if not positive:
            raise ValueError(f"Step-5 rank {rank} has no positive measured capture shares")
        mean = math.ceil(sum(positive) / len(positive))
        # Zero growth can be allocator reuse, not a free executable. Use this
        # device's measured mean rather than undercounting its selected set.
        costs = {key: int(graph["capture_bytes"]) or mean for key, graph in identities.items()}
        entries = [{"rows": row, "pin": hot[row], "replay_hits": heat[row],
                    "capture_bytes": sum(cost for key, cost in costs.items() if key[1] == row),
                    "sections": [{"layer": key[0], "section": key[2],
                                  "capture_bytes": costs[key], "replay_hits": qwen_replay(identities[key])}
                                 for key in sorted(identities) if key[1] == row]}
                   for row in ordered_rows]
        ranks.append({"rank": rank, "reserve_base_bytes": packed(sum(costs.values())),
                      "reserve_extra_bytes": CHUNK, "entries": entries,
                      "zero_replay_sibling_bytes": sum(costs[key] for key, graph in identities.items()
                                                       if qwen_replay(graph) == 0),
                      "lazy_zero_replay_bytes": sum(int(graph["capture_bytes"]) for key, graph in inventory.items()
                                                    if key[1] not in rows)})
    return {"schema_version": 1, "family": "step5", "layout": layout, "ranks": ranks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True,
                        choices=sorted((*SITE_PREFIX, "glm5_flash", "qwen4", "deepseek_v41", "mimo_v2", "kimi_k3", "qwen3_5", "muse_spark", "step5", "hy4_flash", "inkling2")))
    parser.add_argument("--layout",
                        help="coordinator layout the trace was recorded on (glm5, mistral_large4, deepseek_v41), "
                             "or the deepseek_v4 layout key to replace in --list")
    parser.add_argument("--list", type=Path,
                        help="deepseek_v4 only: checked-in prewarm list the layout is replaced in")
    parser.add_argument("--shape", choices=QWEN_SHAPES,
                        help="qwen4 resolved capture shape (owner count, Sparks attached)")
    parser.add_argument("--prefill-rows", type=int, help="V4.1 exact current full prefill chunk width")
    parser.add_argument("--trace", action="append", required=True,
                        help="PATH, LAYOUT=PATH for GLM Flash, or BAND:CYCLES=RANK0.json[,RANK1.json] for Inkling (repeatable)")
    parser.add_argument("--output", type=Path,
                        help="write the generated object here (one shape for qwen4, the list otherwise)")
    parser.add_argument("--merge", type=Path,
                        help="qwen4 only: install the generated shape into this prewarm list in place")
    parser.add_argument("--coverage", type=float, default=0.999)
    parser.add_argument("--pin-frequency", type=float, default=0.9, help="schema 1 legacy frequency threshold")
    parser.add_argument("--ranks", type=int, help="MiMo/Kimi/Inkling physical coordinator ranks")
    parser.add_argument("--variant", help="checkpoint variant: MiMo program/layers (mimof-48), Kimi core (bf16 or k6)")
    parser.add_argument("--onboard-layers", type=int, help="Muse resolved onboarded layer prefix")
    parser.add_argument("--sparks", choices=("yes", "no"), help="MiMo Spark attachment")
    args = parser.parse_args()
    report = []
    heat_rows = []
    summary = {}
    arenas = []
    if args.family == "qwen4" and args.shape is None:
        parser.error("--shape is required for qwen4")
    if args.family != "qwen4" and args.shape is not None:
        parser.error("--shape applies only to qwen4")
    if args.family != "qwen4" and args.merge is not None:
        parser.error("--merge applies only to qwen4")
    if (args.family == "deepseek_v4") != (args.list is not None):
        parser.error("--list applies only to deepseek_v4")
    if args.family == "deepseek_v4" and args.layout is None:
        parser.error("--layout is required for deepseek_v4")
    if args.family != "deepseek_v4" and args.list is not None:
        parser.error("--list applies only to deepseek_v4")
    layout = args.layout
    if args.family in ("mistral_large4", "deepseek_v41") or (args.family == "glm5" and layout is not None):
        if layout not in ("1", "2"):
            parser.error(f"{args.family} keys its list by layout: pass --layout 1 or 2")
        layout = int(layout)
    elif args.family == "step5":
        if layout not in STEP5_LAYOUTS:
            parser.error("Step-5 requires its exact --layout")
    elif args.family == "hy4_flash":
        if layout not in HY4_LAYOUTS:
            parser.error("Hy4-flash requires its exact --layout")
    elif args.family == "inkling2":
        if not layout or args.ranks not in (1, 2):
            parser.error("Inkling requires its exact --layout and --ranks 1 or 2")
    elif layout is not None and args.family not in ("deepseek_v4", "kimi_k3"):
        parser.error("--layout applies to glm5, mistral_large4, deepseek_v4, deepseek_v41, kimi_k3, step5, hy4_flash and inkling2 only")
    if (args.output is None) == (args.merge is None):
        parser.error("exactly one of --output or --merge is required")
    try:
        if args.family == "qwen3_5":
            result = generate_qwen35([json.loads(Path(path).read_text()) for path in args.trace], args.variant, summary)
        elif args.family == "muse_spark":
            result = generate_muse([json.loads(Path(path).read_text()) for path in args.trace],
                                   args.ranks, args.onboard_layers)
        elif args.family == "step5":
            result = generate_step5([json.loads(Path(path).read_text()) for path in args.trace], layout)
        elif args.family == "hy4_flash":
            result = generate_hy4_flash([json.loads(Path(path).read_text()) for path in args.trace], layout)
        elif args.family == "inkling2":
            result = generate_inkling2(list(inkling2_bands(args.trace)), layout, args.ranks)
        elif args.family == "mimo_v2":
            result = generate([json.loads(Path(path).read_text()) for path in args.trace],
                              args.family, args.coverage, args.pin_frequency, pin_report=report,
                              ranks=args.ranks, variant=args.variant,
                              sparks=None if args.sparks is None else args.sparks == "yes",
                              arena_report=arenas)
        elif args.family == "kimi_k3":
            if args.ranks is None or args.layout is None:
                raise ValueError("kimi_k3 needs --ranks and --layout")
            result = generate_kimi([json.loads(Path(path).read_text()) for path in args.trace],
                                   args.ranks, args.layout, args.coverage, report, core=args.variant)
        elif args.family == "qwen4":
            traces = [json.loads(Path(spec).read_text()) for spec in args.trace]
            result = generate_qwen(traces, args.shape, args.coverage, report, summary)
        elif args.family == "deepseek_v4":
            result = generate_deepseek_v4([json.loads(Path(spec).read_text()) for spec in args.trace],
                                          layout, json.loads(args.list.read_text()),
                                          args.coverage, args.pin_frequency, report, heat_rows)
        elif args.family == "deepseek_v41":
            result = generate([json.loads(Path(path).read_text()) for path in args.trace],
                              args.family, args.coverage, args.pin_frequency, args.prefill_rows, report, layout)
        else:
            result = generate_layouts(layout_groups(args.trace, args.family),
                                      args.family, args.coverage, args.pin_frequency, report, layout)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        parser.error(str(error))
    if args.family == "kimi_k3":
        merge_kimi_layout(args.output, result)
    elif args.merge is not None:
        document = merge_qwen(json.loads(args.merge.read_text()), result)
        args.merge.write_text(json.dumps(document, indent=2) + "\n")
    else:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.family == "muse_spark":
        print(f"Muse {result['ranks']} ranks onboard_layers={result['onboard_layers']}: {len(result['entries'])} geometries; "
              f"base={result['reserve_base_bytes']} extra={result['reserve_extra_bytes']}")
        return
    if args.family == "deepseek_v4":
        entry = result["layouts"][layout]
        verify = sum(1 for row in entry["entries"] if row["table_width"] > 0)
        print(f"{len(entry['entries'])} shapes ({verify} verify geometries, "
              f"{len(entry['entries']) - verify} width-0 startup rows); "
              f"base={entry['reserve_base_bytes']} "
              f"extra={entry['reserve_extra_bytes']} bootstrap={entry['bootstrap_bytes']} bytes; "
              f"pinned={sum(entry['pin'] for entry in entry['entries'])}")
        for row in heat_rows:
            print("heat " + json.dumps(row, sort_keys=True))
        for pin in report:
            print("pin " + json.dumps(pin, sort_keys=True))
        return
    if args.family == "qwen4":
        print(f"{result['shape']}: {len(result['trace']['entries'])} shapes; "
              f"base={result['reserve_base_bytes']} extra={result['reserve_extra_bytes']} bytes; "
              f"pinned={sum(entry['pin'] for entry in result['trace']['entries'])}; "
              f"zero_replay={summary.get('zero_replay_identities', 0)} "
              f"({summary.get('zero_replay_bytes', 0)} bytes); "
              f"zero_share={summary.get('zero_share_identities', 0)} "
              f"({summary.get('zero_share_bytes', 0)} bytes filled)")
        for pin in report:
            print("pin " + json.dumps(pin, sort_keys=True))
        return
    if args.family in ("mimo_v2", "qwen3_5"):
        if args.family == "qwen3_5":
            print("selection " + json.dumps(summary, sort_keys=True))
        for layout in result["layouts"]:
            print(f"{len(layout['entries'])} shapes; ranks={layout.get('ranks', 1)} "
                  f"base={layout['reserve_base_bytes']} extra={layout['reserve_extra_bytes']} bytes; "
                  f"pinned={sum(entry['pin'] for entry in layout['entries'])}")
        for arena in arenas:
            print("arena " + json.dumps(arena, sort_keys=True))
        for pin in report:
            print("pin " + json.dumps(pin, sort_keys=True))
        return
    values = (result["layouts"].values() if "layouts" in result else
              [result] if args.family == "kimi_k3" else result.get("ranks", [result]))
    for values in values:
        pinned = sum(entry["pin"] for entry in values["entries"])
        if "measured_bytes" in values:
            print(f"{len(values['entries'])} shapes; measured={values['measured_bytes']} bytes "
                  f"for {values['measured_executables']} executables; "
                  f"extra={values['reserve_extra_bytes']} bytes; pinned={pinned}")
        else:
            layout = f" layout={values['layout']};" if "layout" in values else ""
            print(f"{len(values['entries'])} shapes;{layout} base={values['reserve_base_bytes']} "
                  f"extra={values['reserve_extra_bytes']} bytes; pinned={pinned}")

    for pin in report:
        print("pin " + json.dumps(pin, sort_keys=True))


if __name__ == "__main__":
    main()
