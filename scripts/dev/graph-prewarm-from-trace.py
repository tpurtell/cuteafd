#!/usr/bin/env python3
"""Turn device graph traces into a family shape prewarm list and reserve suggestion.

Example:
  python3 scripts/dev/graph-prewarm-from-trace.py --family glm5 \
    --trace trace.device-0.json --output prewarm.json
  python3 scripts/dev/graph-prewarm-from-trace.py --family deepseek_v41 \
    --trace trace.device-0.json --prefill-rows 2752 --output prewarm.json

Input is the versioned GRAPH_TRACE JSON snapshot, not arbitrary log text.
Frequencies are aggregated by geometry, not bank id or layer. Pin candidates
must have both dominant frequency and the lowest row/table parameter rank.
Reserve suggestions use the maximum per-device arena, never a sum of ranks.
"""
import argparse
import json
import re
from pathlib import Path

CHUNK = 2 << 20
GEOMETRY = re.compile(r"rows: (\d+), table_width: (\d+), table_stride: (\d+)")
V41_DECODE_ROWS = 64
V41_REPLAY_ROWS = 128
V41_TUPLE = re.compile(r"^\(\d+, (\d+)\)$")
V41_BARE = re.compile(r"^(\d+)$")


def packed(value):
    return (value + CHUNK - 1) // CHUNK * CHUNK


def v41_rows(shape, allowed, site):
    """V4.1 keys are row widths: `rows` or `(layer, rows)`. A key that carries a
    selection vector or a non-row axis is a site rule this generator does not
    have; fail loudly rather than aggregate the wrong integer."""
    match = V41_TUPLE.match(shape) or V41_BARE.match(shape)
    if match is None:
        raise ValueError(f"cannot parse V4.1 geometry at site {site}: {shape!r}")
    rows = int(match.groups()[-1])
    if rows not in allowed:
        raise ValueError(f"V4.1 row width {rows} at site {site} is outside the prewarm domain")
    return rows


def generate(traces, family, coverage=0.999, pin_frequency=0.9, prefill_rows=None):
    if family not in ("glm5", "deepseek_v41"):
        raise ValueError("only glm5 and deepseek_v41 have a checked-in shape-list consumer")
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    counts = {}
    arenas = []
    largest = 0
    for trace in traces:
        if trace.get("schema_version") != 1:
            raise ValueError("expected GRAPH_TRACE schema_version=1")
        cache = trace["cache"]
        arenas.append(int(cache["charged"]))
        for site, values in trace.get("sites", {}).items():
            if site.startswith(family + "."):
                largest = max(largest, int(values["largest_bytes"]))
        # Snapshots for the same device may overlap; max, not sum, prevents
        # double-counting repeated snapshots from one trace session.
        device_counts = {}
        allowed = set(range(1, V41_DECODE_ROWS + 1)) | {V41_REPLAY_ROWS}
        if prefill_rows is not None:
            allowed.add(prefill_rows)
        for graph in trace["graphs"]:
            if not graph["site"].startswith(family + "."):
                continue
            if family == "deepseek_v41":
                key = (v41_rows(graph["shape"], allowed, graph["site"]), 0)
            else:
                match = GEOMETRY.search(graph["shape"])
                if match is None:
                    raise ValueError(f"cannot parse {family} geometry: {graph['shape']!r}")
                rows, width, stride = map(int, match.groups())
                if rows <= 0 or width <= 0 or stride != width:
                    raise ValueError("unsupported prewarm geometry")
                key = (rows, width)
            cycles = graph.get("cycles")
            # Startup prewarm is attributed to Other, not request frequency.
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) if cycles is not None else int(graph["count"])
            if count > 0:
                device_counts[key] = device_counts.get(key, 0) + count
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
    if not counts:
        raise ValueError("trace contains no matching graph identities")
    ranked = sorted(counts, key=lambda key: (-counts[key], key[1], key[0]))
    total = sum(counts.values())
    covered = 0
    selected = []
    for key in ranked:
        selected.append(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    peak = max(counts.values())
    minimum = min(counts, key=lambda key: (key[0] * key[1], key))
    if family == "deepseek_v41":
        entries = [{"rows": rows, "pin": (rows, 0) == minimum and counts[(rows, 0)] >= peak * pin_frequency}
                   for rows, _ in sorted(selected)]
    else:
        entries = [{"rows": rows, "table_width": width,
                    "pin": (rows, width) == minimum and counts[(rows, width)] >= peak * pin_frequency}
                   for rows, width in sorted(selected, key=lambda key: (key[1], key[0]))]
    base = packed(max(arenas))
    # One largest measured capture is the explicit capture-before-reuse
    # headroom; keep it separate from the observed arena base.
    extra = packed(largest)
    return {"schema_version": 1, "family": family,
            "reserve_base_bytes": base, "reserve_extra_bytes": extra,
            "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["glm5", "deepseek_v41"])
    parser.add_argument("--prefill-rows", type=int,
                        help="V4.1 full prefill chunk width, an allowed large geometry")
    parser.add_argument("--trace", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage", type=float, default=0.999)
    parser.add_argument("--pin-frequency", type=float, default=0.9)
    args = parser.parse_args()
    try:
        result = generate([json.loads(path.read_text()) for path in args.trace],
                          args.family, args.coverage, args.pin_frequency, args.prefill_rows)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['entries'])} shapes; base={result['reserve_base_bytes']} "
          f"extra={result['reserve_extra_bytes']} bytes; "
          f"pinned={sum(entry['pin'] for entry in result['entries'])}")


if __name__ == "__main__":
    main()
