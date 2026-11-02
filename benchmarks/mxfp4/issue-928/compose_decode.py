#!/usr/bin/env python3
"""Issue #928 evidence only: compose recorded kernels at per-layer rank barriers."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics

CAPACITIES = (1, 16, 80)
TILES = (64, 128, 192)


def median(values):
    return statistics.median(values)


def spread(values):
    return dict(mean=statistics.mean(values), median=median(values),
                minimum=min(values), maximum=max(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.)


def key(rank):
    return rank["logical"], rank["stored"]


def record(index, family, logical, capacity):
    return index[family, logical, capacity]


def samples(index, family, pair, tile, capacity):
    logical, stored = pair
    if family.startswith("mimo"):
        # The grouped GEMV has no logical bound: zero tails run all P columns.
        row = record(index, family, stored, capacity)
        return row["samples_us"][tile]
    return record(index, family, logical, capacity)["samples_us"][f"s{logical}p{stored}t{tile}"]


def choices(index, family, pair):
    logical, stored = pair
    probe = record(index, family, stored if family.startswith("mimo") else logical, 1)
    labels = [name for name in probe["samples_us"] if not name.startswith("padded")] if family.startswith("mimo") else TILES
    result = []
    for tile in labels:
        label = tile if family.startswith("mimo") else f"s{logical}p{stored}t{tile}"
        if all(label in record(index, family, stored if family.startswith("mimo") else logical, capacity)["samples_us"]
               for capacity in CAPACITIES):
            result.append(tile)
    return result


def qualified(index, family, pair, tile):
    logical, stored = pair
    label = tile if family.startswith("mimo") else f"s{logical}p{stored}t{tile}"
    return all(all(record(index, family, stored if family.startswith("mimo") else logical, c)[field].get(label, False)
                   for field in ("quality_eligible", "within_version_invariants")) for c in CAPACITIES)


def blend(layers, get_samples):
    timed = [[get_samples(rank) for rank in layer["ranks"]] for layer in layers]
    count = min(len(values) for layer in timed for values in layer)
    maxima = [sum(max(values[i] for values in layer) for layer in timed) for i in range(count)]
    means = [sum(statistics.mean(values[i] for values in layer) for layer in timed) for i in range(count)]
    point_max = sum(max(median(values) for values in layer) for layer in timed)
    point_mean = sum(statistics.mean(median(values) for values in layer) for layer in timed)
    return dict(sum_layer_max_us=point_max, sum_layer_mean_us=point_mean,
                max_over_mean=point_max / point_mean, samples_sum_layer_max_us=maxima,
                samples_sum_layer_mean_us=means, paired_sample_summary=spread(maxima))


def curve_slopes(index):
    curves = []
    for family in sorted({k[0] for k in index if not k[0].startswith("mimo")}):
        widths = sorted({k[1] for k in index if k[0] == family})
        for capacity, tile in itertools.product(CAPACITIES, TILES):
            pairs = []
            for width in widths:
                previous = width - 32
                if previous not in widths or (width + tile - 1) // tile != (previous + tile - 1) // tile:
                    continue
                left, right = record(index, family, previous, capacity), record(index, family, width, capacity)
                a, b = f"s{previous}p{previous}t{tile}", f"s{width}p{width}t{tile}"
                if a not in left["medians_us"] or b not in right["medians_us"]:
                    continue
                t0, t1 = left["medians_us"][a], right["medians_us"][b]
                matched = (left["experts"] == right["experts"] and left["route_histogram"] == right["route_histogram"]
                           and previous % 128 != 0 and width % 128 != 0)
                pairs.append(dict(start=previous, end=width, delta_us=t1 - t0, ratio=t1 / t0,
                                  same_population_routes_compact=matched))
            stable = [p for p in pairs if p["same_population_routes_compact"]]
            curves.append(dict(family=family, capacity=capacity, tile=tile, pairs=pairs,
                               matched_pairs=len(stable),
                               matched_delta_us_per_32=spread([p["delta_us"] for p in stable]) if stable else None,
                               matched_ratio_per_32=spread([p["ratio"] for p in stable]) if stable else None))
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analytic = json.loads(Path(__file__).with_name("ownership.json").read_text())
    sources = sorted(args.results.glob("*/*-decode.json"))
    index = {}
    for source in sources:
        for row in json.loads(source.read_text())["results"]:
            identity = row["geometry"], row["width"], row["rows"]
            assert identity not in index, identity
            index[identity] = row
    rows, failures = [], []
    for case in analytic["cases"]:
        if case["tile"] != 128 or case["duplicate_of"] is not None or case["zero_rank"]:
            continue
        family, world = case["family"], case["world"]
        if family not in {k[0] for k in index}:
            continue
        pairs = sorted({key(rank) for layer in case["layers"] for rank in layer["ranks"]})
        control_case = next(c for c in analytic["cases"] if c["family"] == family and c["world"] == world
                            and c["tile"] == 128 and c["unit"] == 32 and c["scheme"] == "rotation")
        controls = {}
        for capacity in CAPACITIES:
            controls[capacity] = [blend(control_case["layers"], lambda rank, label=label:
                record(index, family, rank["logical"], capacity)["samples_us"][label])
                for label in ("padded", "padded-control-1", "padded-control-2")]
        try:
            options = [choices(index, family, pair) for pair in pairs]
        except KeyError as error:
            failures.append(dict(family=family, world=world, scheme=case["scheme"], unit=case["unit"], missing=str(error)))
            continue
        for variant in itertools.product(*options):
            selected = dict(zip(pairs, variant))
            eligible = all(qualified(index, family, pair, tile) for pair, tile in selected.items())
            pad_qualification = family.startswith("mimo") and any(s != p for s, p in pairs)
            measurements = {}
            for capacity in CAPACITIES:
                candidate = blend(case["layers"], lambda rank: samples(index, family, key(rank), selected[key(rank)], capacity))
                reference = controls[capacity][0]
                paired = [a / b for a, b in zip(reference["samples_sum_layer_max_us"], candidate["samples_sum_layer_max_us"])]
                control_points = [c["sum_layer_max_us"] for c in controls[capacity]]
                measurements[capacity] = dict(**candidate,
                    control_sum_layer_max_us=reference["sum_layer_max_us"],
                    control_min_us=min(control_points), control_max_us=max(control_points),
                    control_spread_ratio=max(control_points) / min(control_points),
                    median_paired_speedup=median(paired),
                    point_speedup=reference["sum_layer_max_us"] / candidate["sum_layer_max_us"],
                    noninferior_within_control_spread=candidate["sum_layer_max_us"] <= max(control_points))
            rows.append(dict(family=family, world=world, scheme=case["scheme"], unit=case["unit"],
                period=case["period"], logical_max_over_mean=case["logical_max_over_mean"],
                selected=[dict(logical=s, stored=p, tile=t) for (s, p), t in selected.items()],
                quality_and_invariants=eligible, zero_tail_oracle_pending=pad_qualification,
                pad_b_timing=("Pad-B = P-width curve (zero tail computed)" if case["scheme"] == "pad-b"
                              else "Pad-A = P-width curve (zero tail computed)") if pad_qualification else None,
                noninferior_all_capacities=all(m["noninferior_within_control_spread"] for m in measurements.values()),
                provisional_only=True, measurements=measurements))
    best = []
    for family, world in sorted({(r["family"], r["world"]) for r in rows}):
        candidates = [r for r in rows if r["family"] == family and r["world"] == world
                      and r["quality_and_invariants"] and r["noninferior_all_capacities"]]
        if candidates:
            winner = min(candidates, key=lambda r: r["measurements"][1]["sum_layer_max_us"])
            best.append({k: v for k, v in winner.items() if k != "measurements"} | dict(
                speedups=[winner["measurements"][c]["point_speedup"] for c in CAPACITIES],
                max_over_mean=[winner["measurements"][c]["max_over_mean"] for c in CAPACITIES]))
    output = dict(kind="provisional-composed-kernel-evidence-not-engine-timing",
        seal=dict(engine="615afa4e6715f5463c7c2bb3d740999ae11eff11",
                  fork="bcab711a0106dabe3fbbc8231a80b96103b4b215",
                  candidate_sha256="650c8f1e3fd30aff5a549e7bbb93265c0ad9149292ff6d9d078df1a84fe09065"),
        sources=[dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                 for source in sources],
        assumptions=["Every layer waits for all ranks; full rotation sums per-layer maxima.",
            "Mean rank time is secondary; V4.1 two-lane overlap is not modeled.",
            "One tiling per (logical,stored) width across C1/C16/C80; no product policy is inferred.",
            "Source populations vary with width to exceed L2; across-width routing groups may differ.",
            "Paired sample indices compose independent measurements, not simultaneous hardware ranks.",
            "Grouped padded kernels run physical P; zero-tail oracle eligibility needs separate qualification.",
            "Only bitwise-compute-and-storage duplicates and zero-logical Pad-A arms were omitted.",
            "Point noninferiority uses the three identical-control medians; raw paired samples remain included."],
        slopes=curve_slopes(index), candidates=rows, provisional_fastest_noninferior=best, missing=failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    print(f"Composed {len(rows)} variants; {len(best)} provisional noninferior worlds; {len(failures)} missing cases")
    for item in best:
        print(item["family"], item["world"], item["scheme"], item["unit"],
              "/".join(f"{v:.4f}" for v in item["speedups"]), item["selected"])


if __name__ == "__main__":
    main()
