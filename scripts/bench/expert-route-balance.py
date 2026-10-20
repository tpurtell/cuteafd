#!/usr/bin/env python3
"""Score Spark load balance offline from an expert route log (PROFILE=expert-routes).

  scripts/bench/expert-route-balance.py LOG.jsonl [LOG ...] --tp 4,6 [--rule exl3-k4,exlr8-unit]
      [--kind decode,mtp_verify] [--weight unique|rows] [--shared] [--json OUT]
  scripts/bench/expert-route-balance.py --synthetic-uniform 8,23,43 --tp 6

Each record of the log is one layer-step (one Spark wave: a step's MoE layer).
An expert is split into UNITS (16) 128-channel units. A rule places every unit
of every expert on a rank (#408 design comment):

  exl3-k4      Offline fixed EXL3 reference: unit u
               on rank u mod N (every expert puts ceil(16/N) units on rank 0).
  exlr8-unit   EXLR8: unit u of expert e in layer L on rank (16e + u + L) mod N.
               With --shared the shared expert is e = EXPERTS (256) and is
               routed on every layer-step.

A layer-step's load on a rank is the units it reads: with --weight unique (the
default; decode and verify read each routed expert's weights once per step)
one per unit of each distinct routed expert, with --weight rows the units
times that expert's routed rows (compute-bound prefill). The busiest rank's
share is its load over the total. Reported per rule and TP size: mean, p50 and
p99 of that share over layer-steps, the mean share in units of 16 (ideal
16/N), the unit histogram (how many units each routed expert puts on its
layer-step's busiest rank) and each rank's share of the total load; then the
same per wave kind and row bucket.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPERTS = 256
UNITS = 16
RULES = ('exl3-k4', 'exlr8-unit')


def unit_rank(rule: str, expert: int, unit: int, layer: int, tp: int) -> int:
    if rule == 'exl3-k4':
        return unit % tp
    if rule == 'exlr8-unit':
        return (UNITS * expert + unit + layer) % tp
    raise ValueError('unknown rule ' + rule)


def rank_units(rule: str, expert: int, layer: int, tp: int) -> list[int]:
    """Units of `expert` on each rank."""
    counts = [0] * tp
    for unit in range(UNITS):
        counts[unit_rank(rule, expert, unit, layer, tp)] += 1
    return counts


def step_loads(experts: dict[int, int], layer: int, rule: str, tp: int, weight: str, shared: bool):
    """Per-rank load of one layer-step and each placed expert's units per rank;
    `experts` maps expert id -> routed rows."""
    loads = [0] * tp
    placed = []
    items = list(experts.items())
    if shared:
        items.append((EXPERTS, max(experts.values(), default=1)))
    for expert, rows in items:
        factor = rows if weight == 'rows' else 1
        units = rank_units(rule, expert, layer, tp)
        placed.append(units)
        for rank, count in enumerate(units):
            loads[rank] += count * factor
    return loads, placed


def record_experts(record: dict) -> dict[int, int]:
    """Expert id -> routed rows of one record (topk rows or expert_rows counts)."""
    if 'topk' in record:
        counts: Counter = Counter()
        for row in record['topk']:
            counts.update(int(e) for e in row)
        return dict(counts)
    if 'expert_rows' in record:
        return {int(e): int(n) for e, n in record['expert_rows']}
    raise ValueError('record has neither topk nor expert_rows')


def read_log(paths: list[Path]):
    for path in paths:
        with open(path, errors='replace') as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A container stopped mid-write leaves at most one partial last line.
                    print(f'{path}:{number}: skipping an unparsable line', file=sys.stderr)
                    continue
                if record.get('event') == 'expert_routes':
                    yield record


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[index]


def row_bucket(rows: int) -> str:
    for limit in (1, 4, 8, 16, 64, 512):
        if rows <= limit:
            return f'<={limit}'
    return '>512'


class Score:
    def __init__(self):
        self.shares: list[float] = []
        # Units each placed expert puts on its layer-step's busiest rank.
        self.unit_histogram: Counter = Counter()
        self.rank_load: Counter = Counter()
        self.total = 0
        self.unique: list[int] = []

    def add(self, loads: list[int], placed: list[list[int]]):
        total = sum(loads)
        if not total:
            return
        busiest = max(range(len(loads)), key=loads.__getitem__)
        self.shares.append(loads[busiest] / total)
        self.unit_histogram.update(units[busiest] for units in placed)
        for rank, load in enumerate(loads):
            self.rank_load[rank] += load
        self.total += total
        self.unique.append(len(placed))

    def summary(self, tp: int) -> dict:
        n = len(self.shares)
        mean = sum(self.shares) / n if n else math.nan
        return {
            'layer_steps': n,
            'mean': mean, 'p50': percentile(self.shares, 50), 'p99': percentile(self.shares, 99),
            'max': max(self.shares) if n else math.nan,
            'busiest_of_16': UNITS * mean if n else math.nan, 'ideal_of_16': UNITS / tp, 'ideal': 1 / tp,
            'mean_unique_experts': sum(self.unique) / n if n else math.nan,
            'busiest_rank_units_per_expert': {str(k): v for k, v in sorted(self.unit_histogram.items())},
            'rank_load_share': [self.rank_load[r] / self.total if self.total else math.nan for r in range(tp)],
        }


def score(records, tps: list[int], rules: list[str], weight: str, shared: bool, kinds: set[str] | None,
          layer_offset: int) -> dict:
    overall = {(rule, tp): Score() for rule in rules for tp in tps}
    grouped = defaultdict(Score)
    kinds_seen: Counter = Counter()
    for record in records:
        kind = record.get('kind', 'decode')
        kinds_seen[kind] += 1
        if kinds is not None and kind not in kinds:
            continue
        experts = record_experts(record)
        if not experts:
            continue
        layer = int(record['layer']) + layer_offset
        bucket = row_bucket(int(record.get('rows', 1)))
        for rule in rules:
            for tp in tps:
                loads, placed = step_loads(experts, layer, rule, tp, weight, shared)
                overall[rule, tp].add(loads, placed)
                grouped[rule, tp, kind, bucket].add(loads, placed)
    result = {'weight': weight, 'shared': shared, 'kinds_seen': dict(kinds_seen), 'rules': {}}
    for (rule, tp), item in overall.items():
        result['rules'].setdefault(rule, {})[str(tp)] = item.summary(tp)
    result['groups'] = [
        {'rule': rule, 'tp': tp, 'kind': kind, 'rows': bucket, **item.summary(tp)}
        for (rule, tp, kind, bucket), item in sorted(grouped.items())
    ]
    return result


def synthetic_uniform(unique_counts: list[int], steps: int, seed: int, layers=range(3, 78)):
    """Records of `u` distinct uniformly random experts per layer-step (one row each)."""
    generator = random.Random(seed)
    layers = list(layers)
    for unique in unique_counts:
        for index in range(steps):
            yield {'event': 'expert_routes', 'kind': f'uniform-U{unique}', 'layer': layers[index % len(layers)],
                   'rows': unique, 'expert_rows': [[e, 1] for e in generator.sample(range(EXPERTS), unique)]}


def render(result: dict) -> str:
    lines = [f"weight={result['weight']} shared={result['shared']} kinds={result['kinds_seen']}", '',
             '| rule | TP | layer-steps | mean | p50 | p99 | busiest /16 | ideal /16 | unique experts |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rule, by_tp in result['rules'].items():
        for tp, s in by_tp.items():
            lines.append(f"| {rule} | {tp} | {s['layer_steps']} | {s['mean']:.4f} | {s['p50']:.4f} | {s['p99']:.4f} "
                         f"| {s['busiest_of_16']:.3f} | {s['ideal_of_16']:.3f} | {s['mean_unique_experts']:.1f} |")
    lines += ['', '| rule | TP | kind | rows | layer-steps | mean | p50 | p99 | unique experts |',
              '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for g in result['groups']:
        lines.append(f"| {g['rule']} | {g['tp']} | {g['kind']} | {g['rows']} | {g['layer_steps']} | {g['mean']:.4f} "
                     f"| {g['p50']:.4f} | {g['p99']:.4f} | {g['mean_unique_experts']:.1f} |")
    lines.append('')
    for rule, by_tp in result['rules'].items():
        for tp, s in by_tp.items():
            ranks = ' '.join(f'{x:.4f}' for x in s['rank_load_share'])
            lines.append(f"{rule} TP{tp}: units per expert on the busiest rank {s['busiest_rank_units_per_expert']}; "
                         f"rank load shares {ranks}")
    return '\n'.join(lines)


def int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(',') if x]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', nargs='*', type=Path)
    parser.add_argument('--tp', type=int_list, default=[4, 6], help='Spark TP sizes, e.g. 4,6')
    parser.add_argument('--rule', default=','.join(RULES), help='comma list of ' + ', '.join(RULES))
    parser.add_argument('--kind', help='only these wave kinds (comma list: decode, mtp_verify, prefill)')
    parser.add_argument('--weight', choices=('unique', 'rows'), default='unique')
    parser.add_argument('--shared', action='store_true', help='also place the shared expert (e = 256) on the Sparks')
    parser.add_argument('--layer-offset', type=int, default=0, help='added to each record layer to give the rule L')
    parser.add_argument('--synthetic-uniform', type=int_list, metavar='U,U',
                        help='score uniform random routes with U distinct experts per layer-step instead of logs')
    parser.add_argument('--steps', type=int, default=20000, help='synthetic layer-steps per U')
    parser.add_argument('--seed', type=int, default=408)
    parser.add_argument('--json', type=Path, help='write the full result here')
    args = parser.parse_args(argv)
    rules = [r for r in args.rule.split(',') if r]
    if not rules or any(r not in RULES for r in rules):
        parser.error('--rule must be a comma list of ' + ', '.join(RULES))
    if not args.tp or any(not 1 <= tp <= 16 for tp in args.tp):
        parser.error('--tp must list sizes 1..16')
    if bool(args.logs) == bool(args.synthetic_uniform):
        parser.error('give route logs or --synthetic-uniform, not both')
    records = synthetic_uniform(args.synthetic_uniform, args.steps, args.seed) if args.synthetic_uniform \
        else read_log(args.logs)
    kinds = set(args.kind.split(',')) if args.kind else None
    result = score(records, args.tp, rules, args.weight, args.shared, kinds, args.layer_offset)
    if not any(s['layer_steps'] for by_tp in result['rules'].values() for s in by_tp.values()):
        print('no layer-steps matched (kinds seen: %s)' % result['kinds_seen'], file=sys.stderr)
        return 2
    print(render(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
