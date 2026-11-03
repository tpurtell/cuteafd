"""Whole-expert (expert-parallel) Spark load from real Muse Spark routes.

Input: the router's selected expert ids per layer, as recorded by
`model.capture_routes` during the golden (one array per layer, [tokens, 8],
in token order; .npy, .pt or .safetensors with key `ids`). Output: for each
ownership policy and step width, the busiest rank's work relative to the
intermediate-sliced TP split, which gives every rank 1/N of every route.

Work of one rank in one step = routed rows it runs (activation work) and
distinct experts it reads (weight bytes; dominant at decode). Policies:

- fixed: the engine's `ExpertOwners::staggered` rule, rank (e + layer) % N;
- balanced: experts sorted by popularity (half the routes), snake-dealt over
  ranks per layer, so each rank's expected load is equal;
- replicated: balanced, plus the H hottest experts of each layer held on every
  rank's spare budget, each of their routes running on the least-loaded rank
  holding them (whole-expert LPT, as the replicated group scheduler).

Steps: contiguous token windows of `width` rows (decode/verify widths 1..8,
the 2752-row prefill unit), drawn from the second half of the routes so the
popularity table is not fit on the measured steps.

    python3 ep_balance.py ROUTES_DIR --ranks 6 --hot 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

EXPERTS = 256


def load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".pt":
        import torch
        return torch.load(path, map_location="cpu").numpy()
    from safetensors.numpy import load_file
    return load_file(str(path))["ids"]


def staggered(layer: int, ranks: int) -> np.ndarray:
    return (np.arange(EXPERTS) + layer) % ranks


def balanced(counts: np.ndarray, ranks: int) -> np.ndarray:
    owner = np.empty(EXPERTS, dtype=np.int64)
    for position, expert in enumerate(np.argsort(-counts, kind="stable")):
        cycle, offset = divmod(position, ranks)
        owner[expert] = offset if cycle % 2 == 0 else ranks - 1 - offset
    return owner


def step_loads(ids: np.ndarray, owner: np.ndarray, ranks: int, hot: np.ndarray | None):
    """(max rank routed rows, max rank distinct experts) of one step."""
    rows = np.zeros(ranks)
    experts = [set() for _ in range(ranks)]
    hot_set = set() if hot is None else set(int(e) for e in hot)
    flat = ids.reshape(-1)
    cold = [int(e) for e in flat if int(e) not in hot_set]
    for e in cold:
        rows[owner[e]] += 1
        experts[owner[e]].add(e)
    # Hot experts: whole-expert LPT over every rank (each holds a replica).
    for e in sorted(hot_set, key=lambda e: -int((flat == e).sum())):
        n = int((flat == e).sum())
        if n == 0:
            continue
        target = int(np.argmin([len(x) + rows[r] / 64 for r, x in enumerate(experts)]))
        rows[target] += n
        experts[target].add(e)
    return rows.max(), max(len(x) for x in experts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("routes", type=Path)
    parser.add_argument("--ranks", type=int, default=6)
    parser.add_argument("--hot", type=int, default=8)
    parser.add_argument("--widths", default="1,4,8,2752")
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()
    files = sorted(p for p in args.routes.iterdir() if p.suffix in (".npy", ".pt", ".safetensors"))
    rng = np.random.default_rng(0)
    widths = [int(w) for w in args.widths.split(",")]
    results = {(policy, w): [] for policy in ("fixed", "balanced", "replicated") for w in widths}
    for layer, path in enumerate(files):
        ids = load(path).astype(np.int64)
        half = len(ids) // 2
        counts = np.bincount(ids[:half].reshape(-1), minlength=EXPERTS)
        owners = {"fixed": staggered(layer, args.ranks), "balanced": balanced(counts, args.ranks),
                  "replicated": balanced(counts, args.ranks)}
        hot = np.argsort(-counts)[:args.hot]
        test = ids[half:]
        for w in widths:
            if len(test) < w:
                continue
            starts = rng.integers(0, len(test) - w + 1, size=min(args.samples, max(1, len(test) - w + 1)))
            for policy, owner in owners.items():
                for s in starts:
                    step = test[s:s + w]
                    rows, experts = step_loads(step, owner, args.ranks,
                                               hot if policy == "replicated" else None)
                    tp_rows = step.size / args.ranks
                    tp_experts = len(np.unique(step)) / args.ranks
                    results[(policy, w)].append((rows / tp_rows, experts / tp_experts))
    print(f"{len(files)} layers, {args.ranks} ranks; max-rank / TP (routed rows, weight bytes): mean, p99")
    for w in widths:
        for policy in ("fixed", "balanced", "replicated"):
            values = np.array(results[(policy, w)])
            if not len(values):
                continue
            mean, p99 = values.mean(0), np.percentile(values, 99, axis=0)
            print(f"  rows {w:5d} {policy:10s} rows {mean[0]:.2f}/{p99[0]:.2f}  weights {mean[1]:.2f}/{p99[1]:.2f}")


if __name__ == "__main__":
    main()
