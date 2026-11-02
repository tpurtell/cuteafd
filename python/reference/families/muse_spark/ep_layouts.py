"""Muse Spark Spark-expert layouts: per-layer time and memory over the width curve.

Analysis only (no engine change). Every arm keeps one residency (single
residency rule); fidelity is reported separately (``ep_numerics.py``).

Arms (6 Sparks unless --ranks):

- ``ep``: whole experts, owner (e + layer) % N (the current design; baseline).
  A rank reads its owned selected experts whole and returns one combined
  E-wide BF16 row per row (post_expert_norm and routing weight applied).
- ``tp-bf16`` / ``tp-fp8`` / ``tp-nvfp4``: every expert intermediate-sliced
  1/N on every rank (the other families' TP). A rank returns its partial
  down-proj output per selected expert (top-k x E per row), quantized for
  the wire (FP8 E4M3 per-32 scales: 8.25 bits; NVFP4 E2M1 per-16 E4M3
  scales: 4.5 bits). The coordinator dequantizes, sums the N partials per
  expert, applies post_expert_norm and the routing weight, then sums.
- ``colsplit``: TP gate/up over the intermediate, an all-gather of the
  intermediate among the Sparks (FP8 wire), then each rank computes complete
  output values of a 1/N column slice of each selected expert's down
  projection: exact per-element dots, no partial sums. Coordinator ingress is
  one full top-k x E set per row across all ranks (BF16).
- ``hybrid-W``: tp-<wire> partials up to W rows; above W the ranks
  reduce-scatter each expert's partials among themselves (each rank owns 1/N
  of the E columns of every selected expert) and return combined-normed
  slices. One residency (TP slices) at every width.

Model (calibrated on this cluster's measured EXL3 K4 Spark kernels):

- decode/verify (critical path, one step in flight): max over ranks of the
  weight bytes it reads at READ_GBS (GB10 LPDDR5X; a fit of GLM 5.3 EXL3 TP6
  real decode waves gives 280 GB/s) plus per-row compute, plus a fixed wave
  overhead, plus wire time (Spark egress at LINK_SPARK, coordinator ingress at
  LINK_COORD), plus the coordinator combine; summed (nothing overlaps).
- prefill (2752-row unit, the shared scheduler keeps 3 units in flight): the
  unit's time is the max of its resources (max-rank Spark compute at
  PREFILL_TFLOPS, Spark egress, Spark-to-Spark exchange, coordinator ingress,
  coordinator combine); utilization of each resource against that max is
  reported so the binding resource is visible.

    python3 ep_layouts.py [--routes ROUTES_DIR] [--ranks 6]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

E_IN = 4096          # expert input width (pre_expert_proj stream)
INTER = 4096         # expert intermediate
EXPERTS, TOPK, LAYERS = 256, 8, 62
K_BITS = 4.0         # EXL3 K4 experts (trellis payload; vectors are ~0.1%)
EXPERT_BYTES = 3 * E_IN * INTER * K_BITS / 8   # one whole expert: 25.2 MB
READ_GBS = 273e9     # GB10 LPDDR5X; GLM 5.3 EXL3 TP6 decode fit: 280 GB/s
FIXED_US = 100.0     # measured wave overhead at 1 row (168 us GLM TP6 minus its read)
ROW_US = 3.7         # measured per-row decode cost (same fit)
PREFILL_TFLOPS = 38e12  # GLM TP5 w512 K45 m2752: 415 GFLOP in 10.8 ms per rank
LINK_SPARK = 200e9 / 8   # bytes/s per Spark port
LINK_COORD = 400e9 / 8   # bytes/s coordinator port
COORD_BYTES_PER_S = 1.2e12   # RTX PRO elementwise combine (read-bound HBM, conservative)

WIRE_BITS = {"bf16": 16.0, "fp8": 8.25, "nvfp4": 4.5}


def zipf_routes(rows: int, layer: int, rng, alpha: float = 0.6) -> np.ndarray:
    """[rows, TOPK] distinct ids per row from a per-layer Zipf popularity."""
    weights = 1.0 / np.arange(1, EXPERTS + 1) ** alpha
    perm = np.random.default_rng(1000 + layer).permutation(EXPERTS)
    p = np.empty(EXPERTS)
    p[perm] = weights / weights.sum()
    return np.stack([rng.choice(EXPERTS, TOPK, replace=False, p=p) for _ in range(rows)])


UNITS = INTER // 128   # H128 units per expert; TP deals them round-robin from offset e


def tp_units(ranks: int) -> np.ndarray:
    """[EXPERTS, ranks] H128 units of each expert on each rank (staggered remainder)."""
    out = np.zeros((EXPERTS, ranks), dtype=np.int64)
    for e in range(EXPERTS):
        for u in range(UNITS):
            out[e, (u + e) % ranks] += 1
    return out


OWNERS: dict[int, np.ndarray] = {}   # layer -> popularity-balanced owners (--owner balanced)


def ep_owner(layer: int, ranks: int) -> np.ndarray:
    if layer in OWNERS:
        return OWNERS[layer]
    return (np.arange(EXPERTS) + layer) % ranks


def snake_owner(counts: np.ndarray, ranks: int) -> np.ndarray:
    """Experts by popularity, snake-dealt over the ranks (ep_balance's balanced owner)."""
    owner = np.empty(EXPERTS, dtype=np.int64)
    for position, expert in enumerate(np.argsort(-counts, kind="stable")):
        cycle, offset = divmod(position, ranks)
        owner[expert] = offset if cycle % 2 == 0 else ranks - 1 - offset
    return owner


class RealRoutes:
    """Contiguous token windows of the golden's real routes (``muse_routes``). Popularity
    (for the balanced owner and the skew report) comes from each window's first half;
    steps are drawn from the second half, so a balanced owner is never fit to its test."""

    def __init__(self, root: Path, rng):
        from muse_routes import layer_dirs, load_layer
        self.rng = rng
        self.layers: dict[int, list[np.ndarray]] = {}
        self.counts: dict[int, np.ndarray] = {}
        for path in layer_dirs(root):
            layer, windows = load_layer(path)
            fit, test = [], []
            for ids in windows.values():
                half = ids.shape[0] // 2
                fit.append(ids[:half])
                test.append(ids[half:].astype(np.int64))
            self.counts[layer] = np.bincount(np.concatenate(fit).reshape(-1), minlength=EXPERTS)
            self.layers[layer] = [t for t in test if t.shape[0] > 0]
        if not self.layers:
            raise SystemExit(f"{root}: no layerNNN directories")

    def step(self, rows: int, index: int) -> tuple[int, np.ndarray]:
        layers = sorted(self.layers)
        layer = layers[index % len(layers)]
        windows = [w for w in self.layers[layer] if w.shape[0] >= rows]
        if not windows:
            raise SystemExit(f"layer {layer}: no window of {rows} test tokens")
        window = windows[int(self.rng.integers(len(windows)))]
        start = int(self.rng.integers(window.shape[0] - rows + 1))
        return layer, window[start:start + rows]


def skew_report(routes: RealRoutes, ranks: int) -> None:
    """Per-expert popularity skew and whole-expert load imbalance (route shares)."""
    print(f"\nPer-expert popularity (fit half of every window; {len(routes.counts)} layers)")
    print(f"{'stat':34} {'min':>7} {'median':>7} {'max':>7}")
    stats = {"hottest expert / mean": [], "top-8 share of routes": [], "experts with < 0.1x mean": [],
             "Gini": [], f"EP staggered: max rank / mean ({ranks})": [],
             f"EP balanced: max rank / mean ({ranks})": []}
    for layer, counts in sorted(routes.counts.items()):
        p = np.sort(counts / counts.sum())
        mean = 1.0 / EXPERTS
        stats["hottest expert / mean"].append(p[-1] / mean)
        stats["top-8 share of routes"].append(p[-8:].sum())
        stats["experts with < 0.1x mean"].append(float((p < 0.1 * mean).sum()))
        cum = np.cumsum(p)
        stats["Gini"].append(1 - 2 * cum.sum() / EXPERTS + 1 / EXPERTS)
        test = np.bincount(np.concatenate(routes.layers[layer]).reshape(-1), minlength=EXPERTS)
        for name, owner in ((f"EP staggered: max rank / mean ({ranks})", (np.arange(EXPERTS) + layer) % ranks),
                            (f"EP balanced: max rank / mean ({ranks})", snake_owner(counts, ranks))):
            load = np.bincount(owner, weights=test, minlength=ranks)
            stats[name].append(load.max() / load.mean())
    for name, values in stats.items():
        v = np.array(values)
        print(f"{name:34} {v.min():7.3f} {np.median(v):7.3f} {v.max():7.3f}")


def decode_us(arm: str, ids: np.ndarray, layer: int, ranks: int) -> dict:
    rows = ids.shape[0]
    distinct = np.unique(ids)
    out = {}
    if arm == "ep":
        owner = ep_owner(layer, ranks)
        per_rank = np.bincount(owner[distinct], minlength=ranks)
        rows_rank = np.bincount(owner[ids.reshape(-1)], minlength=ranks)
        read = per_rank.max() * EXPERT_BYTES / READ_GBS * 1e6
        compute = read + ROW_US * rows_rank.max() / TOPK * 1.0
        reply_rank = rows * E_IN * 2
        ingress = rows * E_IN * 2 * ranks
        combine_bytes = ingress
        s2s = 0.0
        imbalance = per_rank.max() / (len(distinct) / ranks)
    else:
        wire = arm.split("-", 1)[1] if arm.startswith("tp-") else "bf16"
        units = tp_units(ranks)[distinct].sum(0)
        read = units.max() * EXPERT_BYTES / UNITS / READ_GBS * 1e6
        compute = read + ROW_US * rows
        imbalance = units.max() / units.mean()
        if arm == "colsplit":
            # all-gather of the intermediate (FP8 wire): each rank sends its 1/N
            # slice of every selected (row, expert) intermediate to every peer.
            inter_rank = rows * TOPK * INTER / ranks * 8.25 / 8
            s2s = inter_rank * (ranks - 1) / LINK_SPARK * 1e6 + 10.0
            reply_rank = rows * TOPK * E_IN / ranks * 2
            ingress = rows * TOPK * E_IN * 2
        else:
            s2s = 0.0
            reply_rank = rows * TOPK * E_IN * WIRE_BITS[wire] / 8
            ingress = reply_rank * ranks
        combine_bytes = ingress + rows * TOPK * E_IN * 4
    wire_us = max(reply_rank / LINK_SPARK, ingress / LINK_COORD) * 1e6
    combine_us = combine_bytes / COORD_BYTES_PER_S * 1e6 + (3.0 if arm != "ep" else 0.0)
    total = FIXED_US + compute + s2s + wire_us + combine_us
    out.update(compute_us=compute, read_us=read, s2s_us=s2s, reply_kb=reply_rank / 1024, ingress_kb=ingress / 1024,
               wire_us=wire_us, combine_us=combine_us, total_us=total, imbalance=imbalance)
    return out


def prefill_us(arm: str, ids: np.ndarray, layer: int, ranks: int) -> dict:
    rows = ids.shape[0]
    flops_route = 3 * E_IN * INTER * 2
    if arm == "ep":
        owner = ep_owner(layer, ranks)
        routes_rank = np.bincount(owner[ids.reshape(-1)], minlength=ranks)
        compute = routes_rank.max() * flops_route / PREFILL_TFLOPS * 1e6
        reply_rank = rows * E_IN * 2
        ingress = reply_rank * ranks
        s2s = 0.0
        combine_bytes = ingress
    else:
        wire = arm.split("-", 1)[1] if arm.startswith("tp-") else "bf16"
        units = tp_units(ranks)
        work = units[ids.reshape(-1)].sum(0)
        compute = work.max() / UNITS * flops_route / PREFILL_TFLOPS * 1e6
        if arm == "colsplit":
            inter_rank = rows * TOPK * INTER / ranks * 8.25 / 8
            s2s = inter_rank * (ranks - 1) / LINK_SPARK * 1e6
            reply_rank = rows * TOPK * E_IN / ranks * 2
            ingress = rows * TOPK * E_IN * 2
        elif arm.startswith("hybrid"):
            # above the cut: per-expert reduce-scatter among the ranks (FP8 wire),
            # then each rank returns its 1/N columns of every routed (row, expert), normed.
            part = rows * TOPK * E_IN * 8.25 / 8
            s2s = part * (ranks - 1) / ranks / LINK_SPARK * 1e6
            reply_rank = rows * TOPK * E_IN / ranks * 2
            ingress = rows * TOPK * E_IN * 2
        else:
            s2s = 0.0
            reply_rank = rows * TOPK * E_IN * WIRE_BITS[wire] / 8
            ingress = reply_rank * ranks
        combine_bytes = ingress + rows * TOPK * E_IN * 4
    egress = reply_rank / LINK_SPARK * 1e6
    coord = ingress / LINK_COORD * 1e6
    combine = combine_bytes / COORD_BYTES_PER_S * 1e6
    parts = dict(compute=compute, egress=egress, s2s=s2s, ingress=coord, combine=combine)
    unit = max(parts.values())
    return dict(unit_us=unit, binding=max(parts, key=parts.get),
                **{f"{k}_util": v / unit for k, v in parts.items()},
                reply_mb=reply_rank / 2**20, ingress_mb=ingress / 2**20)


def hybrid(arm: str, ids, layer, ranks, cut: int, phase):
    wire = arm.split("-")[-1]
    return phase(f"tp-{wire}" if ids.shape[0] <= cut else "hybrid", ids, layer, ranks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranks", type=int, default=6)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--widths", default="1,4,8,16,64,2752")
    parser.add_argument("--routes", type=Path, help="real route ids per layer (as ep_balance.py)")
    parser.add_argument("--zipf", type=float, default=0.6)
    parser.add_argument("--owner", choices=("staggered", "balanced"), default="staggered",
                        help="EP owner table: (e + layer) % N, or popularity snake-dealt (real routes only)")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    real = RealRoutes(args.routes, rng) if args.routes else None
    if args.owner == "balanced":
        if real is None:
            raise SystemExit("--owner balanced needs --routes")
        OWNERS.update({layer: snake_owner(counts, args.ranks) for layer, counts in real.counts.items()})
    sample = (lambda rows, s: real.step(rows, s)) if real else (lambda rows, s: (s % LAYERS, zipf_routes(rows, s % LAYERS, rng, args.zipf)))
    widths = [int(w) for w in args.widths.split(",")]
    arms = ["ep", "tp-bf16", "tp-fp8", "tp-nvfp4", "colsplit"]
    print(f"Muse Spark, {args.ranks} Sparks, EXL3 K4 expert {EXPERT_BYTES / 1e6:.1f} MB, "
          f"read {READ_GBS / 1e9:.0f} GB/s, prefill {PREFILL_TFLOPS / 1e12:.0f} TFLOP/s; routes: "
          f"{f'real ({len(real.layers)} layers, EP owner {args.owner})' if real else f'synthetic Zipf a={args.zipf}'}")
    if real:
        skew_report(real, args.ranks)
    print("\nDecode/verify (critical path, per layer; per token x62 layers at rows 1)")
    print(f"{'rows':>5} {'arm':10} {'imbal':>5} {'read':>7} {'s2s':>6} {'reply KB':>9} {'ingr KB':>8} "
          f"{'wire':>6} {'comb':>5} {'layer us':>8} {'ms/tok':>7}")
    for w in [w for w in widths if w <= 64]:
        for arm in arms:
            acc = []
            for s in range(args.samples):
                layer, ids = sample(w, s)
                acc.append(decode_us(arm, ids, layer, args.ranks))
            m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
            print(f"{w:5d} {arm:10} {m['imbalance']:5.2f} {m['read_us']:7.1f} {m['s2s_us']:6.1f} "
                  f"{m['reply_kb']:9.1f} {m['ingress_kb']:8.1f} {m['wire_us']:6.1f} {m['combine_us']:5.1f} "
                  f"{m['total_us']:8.1f} {m['total_us'] * LAYERS / 1e3:7.2f}")
    print("\nPrefill unit (rows 2752; time = max of the resources, 3 units in flight)")
    print(f"{'arm':10} {'unit ms':>8} {'binding':>8} {'comp':>5} {'egr':>5} {'s2s':>5} {'ingr':>5} {'comb':>5} "
          f"{'reply MB':>9} {'ingr MB':>8} {'62L s':>6}")
    for w in [w for w in widths if w > 64]:
        for arm in arms + ["hybrid-fp8"]:
            acc = []
            for s in range(max(4, args.samples // 6)):
                layer, ids = sample(w, s)
                acc.append(prefill_us("hybrid" if arm.startswith("hybrid") else arm, ids, layer, args.ranks))
            m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0] if k != "binding"}
            binding = max(set(a["binding"] for a in acc), key=[a["binding"] for a in acc].count)
            print(f"{arm:10} {m['unit_us'] / 1e3:8.2f} {binding:>8} {m['compute_util']:5.2f} {m['egress_util']:5.2f} "
                  f"{m['s2s_util']:5.2f} {m['ingress_util']:5.2f} {m['combine_util']:5.2f} {m['reply_mb']:9.1f} "
                  f"{m['ingress_mb']:8.1f} {m['unit_us'] * LAYERS / 1e6:6.2f}")
    mixes()


def mixes(unit: int = 2752, cutoff: int = 64, ranks: int = 6) -> None:
    """The BF16 cutoff's cost: one tp-<wide> prefill unit carrying `k` decode/verify
    segments of up to `cutoff` rows (BF16 partials, lossless speculation) and the
    rest as one wide segment, the worst mixes admission charges
    (`replies::max_rank_bytes`). Ingress-bound once BF16 rows dominate: a
    quality/speed knob (TJ), not a defect."""
    print(f"\nPrefill unit {unit} rows with k BF16 segments of {cutoff} rows (cutoff), the rest wide; {ranks} ranks")
    print(f"{'wide':6} {'k':>3} {'bf16 rows':>9} {'reply MB':>9} {'ingr MB':>8} {'ingr ms':>8} {'comp ms':>8} {'binding':>8}")
    flops_route = 3 * E_IN * INTER * 2
    compute = unit * TOPK * flops_route / ranks / PREFILL_TFLOPS * 1e3
    for wide in ("nvfp4", "fp8"):
        for k in (0, 4, 8, 15):
            bf16 = min(k * cutoff, unit)
            reply = TOPK * E_IN * (bf16 * 2 + (unit - bf16) * WIRE_BITS[wide] / 8)
            ingress = reply * ranks / LINK_COORD * 1e3
            print(f"{wide:6} {k:3d} {bf16:9d} {reply / 2**20:9.1f} {reply * ranks / 2**20:8.1f} {ingress:8.2f} "
                  f"{compute:8.2f} {'ingress' if ingress > compute else 'compute':>8}")


if __name__ == "__main__":
    main()
