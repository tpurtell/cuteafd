#!/usr/bin/env python3
"""Balanced TP expert placement prototype (v4): placement rules, per-rank unit
maps, an offline scorer, a held-out route-aware fit and a numerics model.

  scripts/bench/tp-placement.py score LOG.jsonl [LOG ...] --experts 256 --tp 6 [--json OUT]
  scripts/bench/tp-placement.py synthetic --experts 384 --unique 8,18,60,130 --tp 6
  scripts/bench/tp-placement.py memory --experts 384 --tp 6 --hidden 6144 --bpw 4.25 --layers 11 70
  scripts/bench/tp-placement.py numerics [--trials 64]

A rule maps unit u (an `unit`-column block of an expert's intermediate) of
expert e in layer L to a rank of an N-rank Spark world. Units per expert:
U = intermediate / unit. Rules (all checkpoint-independent closed forms
except `fitted`):

  contiguous     offline contiguous reference: rank r owns one contiguous run of whole
                 units, the first U mod N ranks one extra (TP6 of 16: 3,3,3,3,2,2
                 for every expert of every layer). With unit=32 (MXFP4 K32
                 scale groups, MiMo) this is 352,352,352,352,320,320.
  layer-rotation contiguous runs shifted by L: the heavy ranks move per layer,
                 every expert of a layer shares them.
  rotated        rank = (U*e + u + L) mod N for uneven worlds (#408).
                 Static policies keep divisible worlds contiguous, unlike the
                 raw EXLR8 checkpoint ownership primitive in the Rust loader.
  fitted         offline experiment only: per (layer, expert), the set of heavy
                 ranks chosen by local search on fit-half routes.

The load of a rank in one wave (one layer-step) is the units it reads: one per
unit of each distinct routed expert (decode and verify read each expert's
weights once per wave; measured per-rank time is proportional to width, #500).
The busiest share is the busiest rank's load over the wave total (ideal 1/N).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATIC_RULES = ('contiguous', 'layer-rotation', 'rotated')


@dataclass(frozen=True)
class Geometry:
    experts: int
    intermediate: int = 2048
    unit: int = 128

    @property
    def units(self) -> int:
        if self.unit <= 0 or self.intermediate % self.unit:
            raise ValueError(f'intermediate {self.intermediate} is not whole {self.unit}-column units')
        return self.intermediate // self.unit


def unit_owner(rule: str, units: int, expert, unit, layer: int, world: int):
    """Rank owning `unit` of `expert` in `layer` (numpy-broadcasting)."""
    if not 1 <= world <= units:
        raise ValueError(f'world {world} needs 1..{units} ranks')
    expert, unit = np.asarray(expert), np.asarray(unit)
    if rule not in STATIC_RULES:
        raise ValueError('unknown rule ' + rule)
    if units % world == 0:
        rule = 'contiguous'
    if rule == 'rotated':
        return (units * expert + unit + layer) % world
    base, extra = divmod(units, world)
    if rule in ('contiguous', 'layer-rotation'):
        # Contiguous runs: ranks < extra own base + 1 units, then base each.
        cut = extra * (base + 1)
        rank = np.where(unit < cut, unit // (base + 1), extra + (unit - cut) // max(base, 1))
        if rule == 'layer-rotation':
            rank = (rank + layer) % world
        return rank + 0 * expert
    raise ValueError('unknown rule ' + rule)


def count_matrix(rule: str, geometry: Geometry, layer: int, world: int) -> np.ndarray:
    """[experts, world] units each rank holds of each expert in `layer`."""
    e = np.arange(geometry.experts)[:, None]
    u = np.arange(geometry.units)[None, :]
    owner = unit_owner(rule, geometry.units, e, u, layer, world)
    counts = np.zeros((geometry.experts, world), dtype=np.int32)
    np.add.at(counts, (np.broadcast_to(e, owner.shape), owner), 1)
    return counts


def heavy_patterns(units: int, world: int) -> list[tuple[int, ...]]:
    """Every choice of the U mod N heavy ranks (TP6 of 16: 15 choices)."""
    return list(itertools.combinations(range(world), units % world))


def pattern_counts(units: int, world: int, heavy: tuple[int, ...]) -> np.ndarray:
    base = units // world
    counts = np.full(world, base, dtype=np.int32)
    counts[list(heavy)] += 1
    return counts


@dataclass(frozen=True)
class RankUnits:
    """One rank's resident units of one layer: what a loader slices and a
    worker groups into launch classes."""
    layer: int
    world: int
    rank: int
    units: dict[int, tuple[int, ...]]  # expert -> sorted global unit indices

    def width_classes(self) -> dict[int, list[int]]:
        """Unit count -> experts, ascending. Each class is one contiguous
        expert table (class-local id = position) served by one fixed-width
        launch, its routes remapped and every other route masked to -1."""
        classes: dict[int, list[int]] = {}
        for expert, owned in sorted(self.units.items()):
            if owned:
                classes.setdefault(len(owned), []).append(expert)
        return dict(sorted(classes.items()))

    def column_ranges(self, expert: int, unit: int) -> list[tuple[int, int]]:
        """Merged [start, end) intermediate columns the loader reads for one
        expert (gate/up rows and down K columns use the same ranges)."""
        spans: list[list[int]] = []
        for u in self.units.get(expert, ()):
            if spans and spans[-1][1] == u * unit:
                spans[-1][1] += unit
            else:
                spans.append([u * unit, (u + 1) * unit])
        return [tuple(s) for s in spans]

    def resident_units(self) -> int:
        return sum(len(v) for v in self.units.values())


def rank_units(rule: str, geometry: Geometry, layer: int, world: int, rank: int) -> RankUnits:
    e = np.arange(geometry.experts)[:, None]
    u = np.arange(geometry.units)[None, :]
    owner = unit_owner(rule, geometry.units, e, u, layer, world)
    return RankUnits(layer, world, rank, {int(x): tuple(int(v) for v in np.nonzero(owner[x] == rank)[0])
                                          for x in range(geometry.experts)})


# ---------------------------------------------------------------- route logs

@dataclass
class Waves:
    """Decode/verify waves of a route log: CSR expert sets per wave, plus the
    same waves restricted to each request-parity half (when the log's
    per-request row counts attribute rows to requests)."""
    layer: np.ndarray
    rows: np.ndarray
    requests: np.ndarray
    time_half: np.ndarray
    attributable: np.ndarray
    offsets: dict[str, np.ndarray]
    ids: dict[str, np.ndarray]

    def experts(self, view: str, index: int) -> np.ndarray:
        o = self.offsets[view]
        return self.ids[view][o[index]:o[index + 1]]


def parse_log(paths: list[Path], kinds=('decode', 'mtp_verify')) -> Waves:
    layer, rows, requests, times, attributable = [], [], [], [], []
    sets = {'all': [], 'half0': [], 'half1': []}
    for path in paths:
        with open(path, errors='replace') as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get('event') != 'expert_routes' or record.get('kind') not in kinds or 'topk' not in record:
                    continue
                topk = record['topk']
                reqs = record.get('requests') or []
                layer.append(int(record['layer']))
                rows.append(len(topk))
                requests.append(len(reqs))
                times.append(int(record.get('unix_ns', 0)))
                sets['all'].append(sorted({int(x) for row in topk for x in row}))
                ok = bool(reqs) and sum(int(n) for _, n in reqs) == len(topk)
                attributable.append(ok)
                halves = ([], [])
                if ok:
                    at = 0
                    for request, n in reqs:
                        halves[int(request) % 2].extend(int(x) for row in topk[at:at + int(n)] for x in row)
                        at += int(n)
                sets['half0'].append(sorted(set(halves[0])))
                sets['half1'].append(sorted(set(halves[1])))
    times_a = np.asarray(times, dtype=np.int64)
    median = np.median(times_a) if len(times_a) else 0
    offsets, ids = {}, {}
    for view, lists in sets.items():
        offsets[view] = np.concatenate([[0], np.cumsum([len(x) for x in lists])]).astype(np.int64)
        ids[view] = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int32)
    return Waves(np.asarray(layer, np.int32), np.asarray(rows, np.int32), np.asarray(requests, np.int32),
                 (times_a > median).astype(np.int8), np.asarray(attributable, bool), offsets, ids)


def save_waves(waves: Waves, path: Path):
    arrays = {'layer': waves.layer, 'rows': waves.rows, 'requests': waves.requests,
              'time_half': waves.time_half, 'attributable': waves.attributable}
    for view in waves.offsets:
        arrays[f'off_{view}'], arrays[f'ids_{view}'] = waves.offsets[view], waves.ids[view]
    np.savez_compressed(path, **arrays)


def load_waves(path: Path) -> Waves:
    z = np.load(path)
    views = [k[4:] for k in z.files if k.startswith('off_')]
    return Waves(z['layer'], z['rows'], z['requests'], z['time_half'], z['attributable'],
                 {v: z[f'off_{v}'] for v in views}, {v: z[f'ids_{v}'] for v in views})


def wave_groups(waves: Waves) -> dict[str, np.ndarray]:
    """Named wave masks: one-row C1 waves (top-k experts), single-request
    verify waves, C2-C11 and C12-C16 multi-request waves. Logs whose family
    declares no serving step (no request ids) fall back to row buckets."""
    if not waves.requests.any():
        rows = waves.rows
        return {'rows=1': rows == 1, 'rows 2-8': (rows > 1) & (rows <= 8), 'rows 9-32': (rows > 8) & (rows <= 32),
                'rows>32': rows > 32, 'all decode': np.ones(len(rows), bool)}
    single = waves.requests == 1
    return {
        'C1 rows=1': single & (waves.rows == 1),
        'C1 verify': single & (waves.rows > 1),
        'C2-C11': (waves.requests >= 2) & (waves.requests < 12),
        'C12-C16': waves.requests >= 12,
        'all decode': np.ones(len(waves.layer), bool),
    }


def indicator(waves: Waves, view: str, select: np.ndarray, experts: int) -> np.ndarray:
    """[selected waves, experts] 0/1 matrix of the view's expert sets."""
    index = np.nonzero(select)[0]
    o = waves.offsets[view]
    lengths = o[index + 1] - o[index]
    out = np.zeros((len(index), experts), dtype=np.int32)
    if len(index):
        row = np.repeat(np.arange(len(index)), lengths)
        col = np.concatenate([waves.ids[view][o[i]:o[i + 1]] for i in index]) if lengths.sum() else np.zeros(0, int)
        out[row, col] = 1
    return out


def shares(waves: Waves, view: str, select: np.ndarray, matrices: dict[int, np.ndarray], experts: int):
    """Per selected wave: busiest share, wave units, and the granularity bound
    ceil(total / N) / total (no placement of these units can do better)."""
    busiest, totals, bound = [], [], []
    for layer in np.unique(waves.layer[select]):
        chosen = select & (waves.layer == layer)
        ind = indicator(waves, view, chosen, experts)
        keep = ind.sum(1) > 0
        loads = ind[keep] @ matrices[int(layer)]
        total = loads.sum(1)
        world = loads.shape[1]
        busiest.append(loads.max(1) / total)
        totals.append(total)
        bound.append(np.ceil(total / world) / total)
    if not busiest:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(busiest), np.concatenate(totals), np.concatenate(bound)


def summarize(values: np.ndarray) -> dict:
    if not len(values):
        return {'waves': 0}
    return {'waves': int(len(values)), 'mean': float(values.mean()), 'p50': float(np.percentile(values, 50)),
            'p99': float(np.percentile(values, 99)), 'max': float(values.max())}


def static_matrices(rule: str, geometry: Geometry, world: int, layers) -> dict[int, np.ndarray]:
    return {int(L): count_matrix(rule, geometry, int(L), world) for L in layers}


def fit_layer(ind: np.ndarray, start: np.ndarray, units: int, world: int, passes: int = 3, seed: int = 0) -> np.ndarray:
    """Local search over each expert's heavy-rank set minimizing the mean
    busiest share of the fit waves `ind` [waves, experts]; starts from `start`."""
    patterns = np.stack([pattern_counts(units, world, p) for p in heavy_patterns(units, world)])
    counts = start.copy()
    loads = ind @ counts
    total = loads.sum(1, keepdims=True).astype(np.float64)
    total[total == 0] = 1
    rng = np.random.default_rng(seed)
    experts = np.nonzero(ind.sum(0))[0]
    for _ in range(passes):
        changed = 0
        for e in rng.permutation(experts):
            waves_e = np.nonzero(ind[:, e])[0]
            base = loads[waves_e] - counts[e]
            # [patterns, waves_e, world]
            trial = base[None] + patterns[:, None, :]
            cost = (trial.max(2) / total[waves_e, 0][None]).sum(1)
            best = int(cost.argmin())
            current = cost[np.nonzero((patterns == counts[e]).all(1))[0][0]]
            if cost[best] < current - 1e-12:
                counts[e] = patterns[best]
                loads[waves_e] = trial[best]
                changed += 1
        if not changed:
            break
    return counts


def fit_matrices(waves: Waves, view: str, select: np.ndarray, geometry: Geometry, world: int,
                 passes: int = 3) -> dict[int, np.ndarray]:
    out = {}
    for layer in np.unique(waves.layer):
        start = count_matrix('rotated', geometry, int(layer), world)
        ind = indicator(waves, view, select & (waves.layer == layer), geometry.experts)
        out[int(layer)] = fit_layer(ind[ind.sum(1) > 0], start, geometry.units, world, passes, seed=int(layer))
    return out


def score(waves: Waves, geometry: Geometry, world: int, fit: bool = True, passes: int = 3,
          fine_unit: int | None = None) -> dict:
    """In-sample shares per wave group for every static rule (plus, with
    `fine_unit`, the contiguous and layer-rotated slices at that finer unit,
    e.g. 32 for MXFP4's K32 scale groups), then the held-out fit."""
    layers = np.unique(waves.layer)
    groups = wave_groups(waves)
    rules = {r: static_matrices(r, geometry, world, layers) for r in STATIC_RULES}
    if fine_unit:
        fine = Geometry(geometry.experts, geometry.intermediate, fine_unit)
        for r in ('contiguous', 'layer-rotation'):
            rules[f'{r} u{fine_unit}'] = static_matrices(r, fine, world, layers)
    result = {'geometry': geometry.__dict__, 'world': world, 'ideal': 1 / world, 'in_sample': {}, 'held_out': {}}
    for name, mask in groups.items():
        row = {}
        for rule, mats in rules.items():
            row[rule] = summarize(shares(waves, 'all', mask, mats, geometry.experts)[0])
        row['granularity bound'] = summarize(shares(waves, 'all', mask, rules['rotated'], geometry.experts)[2])
        result['in_sample'][name] = row
    if not fit:
        return result
    # Held-out: fit on request-parity half 0, score half 1's rows; then the
    # time split (fit on the first half of the log, score the second).
    attributable = waves.attributable
    splits = {
        'request parity': ('half0', attributable, 'half1', attributable),
        'time halves': ('all', waves.time_half == 0, 'all', waves.time_half == 1),
    }
    for split, (fit_view, fit_mask, eval_view, eval_mask) in splits.items():
        fitted = fit_matrices(waves, fit_view, fit_mask, geometry, world, passes)
        per = {}
        for name, mask in groups.items():
            select = mask & eval_mask
            per[name] = {
                'rotated': summarize(shares(waves, eval_view, select, rules['rotated'], geometry.experts)[0]),
                'fitted': summarize(shares(waves, eval_view, select, fitted, geometry.experts)[0]),
                'fitted in-sample': summarize(shares(waves, fit_view, mask & fit_mask, fitted, geometry.experts)[0]),
                'granularity bound': summarize(shares(waves, eval_view, select, rules['rotated'],
                                                      geometry.experts)[2]),
            }
        result['held_out'][split] = per
    return result


def synthetic(geometry: Geometry, world: int, unique: list[int], waves: int = 4000, seed: int = 408,
              layers=range(3, 78)) -> dict:
    """Busiest share of uniformly random expert sets of each size (no route
    log): a routing-independent reference for families without one."""
    rng = np.random.default_rng(seed)
    layers = list(layers)
    out = {}
    for u in unique:
        row = {}
        picks = [(layers[i % len(layers)], rng.choice(geometry.experts, u, replace=False)) for i in range(waves)]
        for rule in STATIC_RULES:
            mats = {}
            values, bounds = [], []
            for layer, experts in picks:
                if layer not in mats:
                    mats[layer] = count_matrix(rule, geometry, layer, world)
                loads = mats[layer][experts].sum(0)
                values.append(loads.max() / loads.sum())
                bounds.append(math.ceil(loads.sum() / world) / loads.sum())
            row[rule] = summarize(np.asarray(values))
        row['granularity bound'] = summarize(np.asarray(bounds))
        out[str(u)] = row
    return out


# ---------------------------------------------------------------- memory

def memory(geometry: Geometry, world: int, hidden: int, bpw: float, layers) -> dict:
    """Resident units per rank summed over `layers`, and GiB at `bpw` bits per
    weight over the three projections of each unit."""
    unit_bytes = 3 * geometry.unit * hidden * bpw / 8
    out = {}
    for rule in STATIC_RULES:
        per_rank = np.zeros(world, dtype=np.int64)
        for L in layers:
            per_rank += count_matrix(rule, geometry, int(L), world).sum(0)
        out[rule] = {'units': per_rank.tolist(), 'gib': [round(float(x * unit_bytes / 2**30), 2) for x in per_rank],
                     'max_over_mean': float(per_rank.max() / per_rank.mean())}
    return out


# ---------------------------------------------------------------- numerics

def bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (nearest even), returned as float32."""
    bits = np.asarray(x, np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def rank_output(act: np.ndarray, w2: np.ndarray, weights: np.ndarray, owned: list[np.ndarray],
                two_class: bool = False, f32_partials: bool = False) -> np.ndarray:
    """Coordinator output of one row under a unit ownership.

    act [topk, I] (BF16 SwiGLU, placement-independent), w2 [topk, H, I],
    weights [topk]; owned[r][k] is a column mask of route k on rank r. Each
    rank's kernel rounds every route's down output to BF16 (y), sums routes in
    FP32 slot order and rounds the partial to BF16; the coordinator sums the
    BF16 partials in FP32 rank order and rounds once (route_reduce.cu).
    `two_class` models two width-class launches combined separately, then
    added in FP32 and rounded (a second BF16 rounding per rank).
    `f32_partials` keeps rank partials in FP32 on the wire (EXLR8's reply)."""
    out = np.zeros(w2.shape[1], np.float32)
    for mask in owned:
        partial = np.zeros(w2.shape[1], np.float32)
        classes = {}
        for k in range(len(weights)):
            if mask[k].any():
                y = bf16((w2[k][:, mask[k]].astype(np.float32) @ act[k][mask[k]].astype(np.float32)))
                classes.setdefault(int(mask[k].sum()) if two_class else 0, []).append(weights[k] * y)
        for parts in classes.values():
            partial = partial + bf16(np.sum(parts, axis=0, dtype=np.float32)) if two_class else \
                partial + np.sum(parts, axis=0, dtype=np.float32)
        out += partial if f32_partials else bf16(partial)
    return bf16(out)


def numerics(trials: int = 32, hidden: int = 512, seed: int = 500) -> dict:
    """Error of the coordinator output against an FP64 reference under each
    placement (TP6, 16 units of 128, top-8 of 256 experts)."""
    geometry, world, topk = Geometry(256, 2048, 128), 6, 8
    rng = np.random.default_rng(seed)
    errors = {name: [] for name in ('contiguous', 'rotated', 'rotated two-class', 'contiguous-k32',
                                    'contiguous f32 partials', 'rotated f32 partials')}
    differs = {name: 0 for name in errors}
    for trial in range(trials):
        layer = int(rng.integers(3, 78))
        experts = rng.choice(geometry.experts, topk, replace=False)
        act = bf16(rng.standard_normal((topk, geometry.intermediate)).astype(np.float32) * 0.5)
        w2 = bf16(rng.standard_normal((topk, hidden, geometry.intermediate)).astype(np.float32) * 0.02)
        weights = rng.random(topk).astype(np.float32)
        weights /= weights.sum()
        exact = np.einsum('k,khi,ki->h', weights.astype(np.float64), w2.astype(np.float64), act.astype(np.float64))

        def masks(rule, g):
            owner = unit_owner(rule, g.units, experts[:, None], np.arange(g.units)[None, :], layer, world)
            cols = np.repeat(owner, g.unit, axis=1)
            return [cols == r for r in range(world)]

        base = rank_output(act, w2, weights, masks('contiguous', geometry))
        outputs = {
            'contiguous': base,
            'rotated': rank_output(act, w2, weights, masks('rotated', geometry)),
            'rotated two-class': rank_output(act, w2, weights, masks('rotated', geometry), two_class=True),
            'contiguous-k32': rank_output(act, w2, weights, masks('contiguous', Geometry(256, 2048, 32))),
            'contiguous f32 partials': rank_output(act, w2, weights, masks('contiguous', geometry), f32_partials=True),
            'rotated f32 partials': rank_output(act, w2, weights, masks('rotated', geometry), f32_partials=True),
        }
        scale = np.abs(exact).mean()
        for name, out in outputs.items():
            errors[name].append(np.abs(out - exact).mean() / scale)
            differs[name] += int((out.view(np.uint32) != base.view(np.uint32)).any())
    return {name: {'mean_rel_abs_error': float(np.mean(v)), 'trials': trials,
                   'trials_bitwise_different_from_contiguous': differs[name]} for name, v in errors.items()}


# ---------------------------------------------------------------- CLI

def render_score(result: dict) -> str:
    lines = [f"world={result['world']} ideal={result['ideal']:.4f} geometry={result['geometry']}", '',
             '| waves | rule | n | mean | p50 | p99 |', '|---|---|---:|---:|---:|---:|']
    for name, row in result['in_sample'].items():
        for rule, s in row.items():
            if s['waves']:
                lines.append(f"| {name} | {rule} | {s['waves']} | {s['mean']:.4f} | {s['p50']:.4f} | {s['p99']:.4f} |")
    for split, per in result['held_out'].items():
        lines += ['', f'held-out ({split})', '', '| waves | rule | n | mean | p50 | p99 |', '|---|---|---:|---:|---:|---:|']
        for name, row in per.items():
            for rule, s in row.items():
                if s['waves']:
                    lines.append(f"| {name} | {rule} | {s['waves']} | {s['mean']:.4f} | {s['p50']:.4f} "
                                 f"| {s['p99']:.4f} |")
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    s.add_argument('logs', nargs='+', type=Path)
    s.add_argument('--experts', type=int, required=True)
    s.add_argument('--intermediate', type=int, default=2048)
    s.add_argument('--unit', type=int, default=128)
    s.add_argument('--tp', type=int, default=6)
    s.add_argument('--no-fit', action='store_true')
    s.add_argument('--fine-unit', type=int, help='also score contiguous slices at this unit (32: MXFP4 352/320)')
    s.add_argument('--passes', type=int, default=3)
    s.add_argument('--cache', type=Path, help='parsed-wave .npz cache (written when missing)')
    s.add_argument('--json', type=Path)
    m = sub.add_parser('memory')
    m.add_argument('--experts', type=int, required=True)
    m.add_argument('--intermediate', type=int, default=2048)
    m.add_argument('--unit', type=int, default=128)
    m.add_argument('--tp', type=int, default=6)
    m.add_argument('--hidden', type=int, required=True)
    m.add_argument('--bpw', type=float, required=True)
    m.add_argument('--layers', type=int, nargs=2, metavar=('FIRST', 'END'), default=(3, 78))
    y = sub.add_parser('synthetic')
    y.add_argument('--experts', type=int, required=True)
    y.add_argument('--intermediate', type=int, default=2048)
    y.add_argument('--unit', type=int, default=128)
    y.add_argument('--tp', type=int, default=6)
    y.add_argument('--unique', type=lambda t: [int(x) for x in t.split(',') if x], required=True)
    y.add_argument('--waves', type=int, default=4000)
    n = sub.add_parser('numerics')
    n.add_argument('--trials', type=int, default=32)
    args = parser.parse_args(argv)
    if args.command == 'score':
        geometry = Geometry(args.experts, args.intermediate, args.unit)
        if args.cache and args.cache.exists():
            waves = load_waves(args.cache)
        else:
            waves = parse_log(args.logs)
            if args.cache:
                save_waves(waves, args.cache)
        if not len(waves.layer):
            print('no decode waves with per-row routes', file=sys.stderr)
            return 2
        result = score(waves, geometry, args.tp, fit=not args.no_fit, passes=args.passes, fine_unit=args.fine_unit)
        print(render_score(result))
    elif args.command == 'synthetic':
        result = synthetic(Geometry(args.experts, args.intermediate, args.unit), args.tp, args.unique, args.waves)
        for u, row in result.items():
            print(f'U={u} ' + ' '.join(f"{rule}={s['mean']:.4f}" for rule, s in row.items()))
    elif args.command == 'memory':
        result = memory(Geometry(args.experts, args.intermediate, args.unit), args.tp, args.hidden, args.bpw,
                        range(*args.layers))
        print(json.dumps(result, indent=1))
    else:
        result = numerics(args.trials)
        print(json.dumps(result, indent=1))
    if getattr(args, 'json', None):
        args.json.write_text(json.dumps(result, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
