#!/usr/bin/env python3
"""Issue #928 evidence only: integer ownership/storage/compute ceilings, not timings."""
import csv
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FAMILIES = (("v41", 5120, 2304, "native-coded"),
            ("dsv4f", 4096, 2048, "native-raw"),
            ("dsv4p", 7168, 3072, "native-raw"),
            ("mimof", 4096, 2048, "grouped-raw"),
            ("mimop", 6144, 2048, "grouped-raw"))
TILES = (64, 128, 192)


def up(value, unit):
    return (value + unit - 1) // unit * unit


def balanced(intermediate, world, unit, layer):
    q, remainder = divmod(intermediate // unit, world)
    result = []
    for rank in range(world):
        index = (rank - layer) % world if remainder else rank
        first = (index * q + min(index, remainder)) * unit
        logical = (q + (index < remainder)) * unit
        result.append((first, logical, logical))
    return result


def ownership(intermediate, world, unit, scheme, layer):
    if scheme == "rotation":
        return balanced(intermediate, world, unit, layer)
    if scheme == "pad-a":
        physical = up((intermediate + world - 1) // world, unit)
        return [(min(rank * physical, intermediate),
                 max(0, min(physical, intermediate - rank * physical)), physical)
                for rank in range(world)]
    return [(first, logical, up(logical, unit))
            for first, logical, _ in balanced(intermediate, world, 32, layer)]


def check(intermediate, world, unit, scheme, layers):
    cap = up(up((intermediate + world - 1) // world, 32), 128)
    for ranks in layers:
        intervals = sorted((first, first + logical) for first, logical, _ in ranks if logical)
        assert intervals[0][0] == 0 and intervals[-1][1] == intermediate
        assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
        assert sum(logical for _, logical, _ in ranks) == intermediate
        assert all(0 <= logical <= physical <= cap for _, logical, physical in ranks)
        assert all(first % 32 == logical % 32 == physical % 32 == 0
                   for first, logical, physical in ranks)
    if scheme != "pad-a":
        # A full absolute-layer rotation balances source columns, not layer barriers.
        assert len({sum(layer[rank][1] for layer in layers) for rank in range(world)}) == 1
    return cap


def main():
    cases, summaries, inventory = [], [], {}
    for family, hidden, intermediate, backend in FAMILIES:
        for world in range(2, 9):
            control = balanced(intermediate, world, 32, 0)
            control = [(first, logical, up(logical, 128)) for first, logical, _ in control]
            control_tiles = [192 if family == "v41" and (logical % 192 == 0 or logical == 288)
                             else 128 for _, logical, _ in control]
            control_max = max(up(rank[1], tile) for rank, tile in zip(control, control_tiles))
            seen = {}
            for unit in (32, 64, 128):
                for scheme in ("rotation", "pad-a", "pad-b"):
                    layers = [ownership(intermediate, world, unit, scheme, layer)
                              for layer in range(world)]
                    cap = check(intermediate, world, unit, scheme, layers)
                    signature = tuple(tuple((logical, physical) for _, logical, physical in ranks)
                                      for ranks in layers)
                    duplicate = seen.setdefault(signature, f"{scheme}-u{unit}")
                    # Keep every arm in the evidence; only exact compute+storage duplicates
                    # are eligible for timing deduplication. Source offsets remain in JSON.
                    for tile in TILES:
                        measured = [[up(logical, tile) for _, logical, _ in ranks] for ranks in layers]
                        maxima = [max(values) for values in measured]
                        mean_sum = sum(sum(values) / world for values in measured)
                        small_max = sum(max(up(logical, 32) for _, logical, _ in ranks)
                                        for ranks in layers)
                        item = dict(family=family, hidden=hidden, intermediate=intermediate,
                                    backend=backend, world=world, scheme=scheme, unit=unit,
                                    tile=tile, period=world, storage_cap=cap,
                                    duplicate_of=None if duplicate == f"{scheme}-u{unit}" else duplicate,
                                    zero_rank=any(logical == 0 for ranks in layers for _, logical, _ in ranks),
                                    logical_widths=sorted({s for ranks in layers for _, s, _ in ranks}),
                                    stored_widths=sorted({p for ranks in layers for _, _, p in ranks}),
                                    max_columns=max(maxima), cycle_max_columns=sum(maxima),
                                    cycle_mean_columns=mean_sum,
                                    max_over_mean=sum(maxima) / mean_sum,
                                    logical_max_over_mean=max(s for ranks in layers for _, s, _ in ranks) * world / intermediate,
                                    max_rank_cycle_stored=max(sum(layer[rank][2] for layer in layers)
                                                              for rank in range(world)),
                                    control_max_columns=control_max,
                                    control_compute_ratio=control_max * world / sum(maxima),
                                    tile32_ceiling_ratio=sum(maxima) / small_max,
                                    probe="compute-step" if max(maxima) < control_max else "staging-or-traffic",
                                    layers=[dict(layer=layer,
                                                 ranks=[dict(rank=rank, first=first, logical=logical,
                                                             stored=physical, computed=measured[layer][rank],
                                                             physical_bound_computed=up(physical, tile))
                                                        for rank, (first, logical, physical) in enumerate(ranks)])
                                            for layer, ranks in enumerate(layers)])
                        cases.append(item)
                        summaries.append({key: value for key, value in item.items() if key != "layers"})
                        key = f"{family}:{hidden}:{intermediate}"
                        entry = inventory.setdefault(key, dict(max_width=0, pairs=set()))
                        for ranks in layers:
                            for _, logical, physical in ranks:
                                if logical:
                                    entry["max_width"] = max(entry["max_width"], logical)
                                    entry["pairs"].add((logical, physical))
    for entry in inventory.values():
        entry["pairs"] = sorted(entry["pairs"])
        entry["curve_widths"] = list(range(32, entry["max_width"] + 1, 32))
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    data = dict(schema=1, engine=head, kind="analytic-not-measured",
                assumptions=["All ranks see identical routed rows; each layer waits for every rank.",
                             "Primary ceiling sums each layer's maximum; rank mean is secondary.",
                             "Rotation period is a full world layers, including equal splits.",
                             "Computed columns use logical bounds; stored extent affects traffic, not slice count.",
                             "physical_bound_computed separately shows kernels that visit every padded column.",
                             "MiMo tile ceilings are hypothetical native-slice ceilings, not its GEMV cost model.",
                             "Zero-logical Pad-A ranks need a no-op role; current native ABI requires positive width.",
                             "No timings, winner selections, allocation totals, or hardware admission are inferred."],
                cases=cases, inventory=inventory)
    (OUT / "ownership.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    with (OUT / "ownership.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUT / "width-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Validated {len(cases)} family/world/scheme/tile cases with exact coverage and storage caps")
    for intermediate in (2304, 2048, 3072):
        family = {2304: "v41", 2048: "dsv4f", 3072: "dsv4p"}[intermediate]
        for world in (3, 4, 5, 6, 7, 8):
            rows = [row for row in summaries if row["family"] == family and row["world"] == world
                    and row["scheme"] == "rotation"]
            text = "; ".join(f"u{unit}: " + "/".join(str(next(row["max_columns"] for row in rows
                                     if row["unit"] == unit and row["tile"] == tile)) for tile in TILES)
                             for unit in (32, 64, 128))
            print(f"I{intermediate}/TP{world}, max computed tile64/128/192: {text}")
    for key, value in inventory.items():
        print(f"{key}: curve32..{value['max_width']}, {len(value['curve_widths'])} widths, "
              f"{len(value['pairs'])} distinct logical/stored pairs")


if __name__ == "__main__":
    main()
