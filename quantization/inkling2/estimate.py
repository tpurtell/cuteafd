"""Operational wall-clock forecasts from receipts; never import numerical code."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo('Asia/Taipei')


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def stamp(value):
    return datetime.fromisoformat(value).timestamp()


def estimate(root, now=None):
    """Return a provisional range and timezone-explicit completion timestamps.

    Before one whole layer completes, extrapolate actual successive K4 expert
    cycle times, add observed attention/setup time and a declared three-minute
    successor/export allowance per layer. Replace that model with measured
    complete-layer durations as soon as those exist. Tail allowances remain
    separately visible until their own completion receipts arrive.
    """
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    layers = [load(p) for p in sorted((root/'layers').glob('*/complete.json'))]
    events = []
    path = root/'events.jsonl'
    if path.exists():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines)-1:
                    raise
                # The producer may be appending its last line right now.
    complete = load(root/'quantization-complete.json').get('status') == 'passed'
    published = load(root/'publication-complete.json').get('status') == 'passed'
    handed = load(root/'handoff-complete.json').get('status') == 'passed'
    cleaned = load(root/'handoff-cleanup-complete.json').get('status') == 'passed'
    failure = load(root/'failure.json')
    result = dict(as_of=now.astimezone(TAIPEI).isoformat(), timezone='Asia/Taipei',
        provisional=True, completed_layers=len(layers), blocked=bool(failure),
        assumption='continuous progress; failure or hardware downtime moves the calendar estimate')
    remaining_layers = max(0, 48-len(layers))
    base_seconds = None
    if layers:
        samples = [r['seconds'] for r in layers[-6:]]
        base_seconds = statistics.mean(samples)
        margins = (.80, 1.25) if len(samples) < 3 else (.90, 1.15)
        result.update(basis='mean of recent completed layers', sample_layers=len(samples),
                      seconds_per_layer=base_seconds)
    else:
        downs = [e for e in events if e.get('bits') == 4 and
                 e.get('module', '').endswith('.down_proj') and '.mlp.experts.' in e['module']]
        cycles = [stamp(b['time'])-stamp(a['time']) for a,b in zip(downs, downs[1:])
                  if a['module'].split('.layers.')[1].split('.')[0] ==
                     b['module'].split('.layers.')[1].split('.')[0]]
        if len(cycles) >= 5:
            cycle = statistics.mean(cycles)
            entries = [e for e in events if e.get('event') == 'attention_input_capture']
            first = stamp(entries[0]['time']) if entries else stamp(downs[0]['time'])
            setup = max(0, stamp(downs[0]['time'])-first-cycle)
            export_allowance = 180
            base_seconds = cycle*384+setup+export_allowance
            margins = (.75, 1.30)
            result.update(basis='measured partial-layer expert throughput plus observed setup',
                sample_expert_cycles=len(cycles), seconds_per_expert=cycle,
                observed_setup_seconds=setup, unmeasured_successor_export_allowance_seconds=export_allowance,
                seconds_per_layer=base_seconds)
    starts = [e for e in events if e.get('event') == 'layer_complete']
    if starts:
        started = stamp(starts[-1]['time'])
    else:
        entries = [e for e in events if e.get('event') == 'attention_input_capture']
        started = stamp(entries[0]['time']) if entries else now.timestamp()
    elapsed = max(0, now.timestamp()-started)
    if remaining_layers == 0 or complete or published or handed:
        layer_range = [0., 0.]
    elif base_seconds is not None:
        layer_range = [max(0., remaining_layers*base_seconds*factor-elapsed) for factor in margins]
    else:
        result.update(basis='insufficient measured work; retain the previous forecast until more receipts exist',
                      remaining_seconds_range=None, completion_time_range=None)
        return result
    tails = dict(head_mtp=[1800., 10800.], validation_upload=[1800., 7200.], placement_cleanup=[900., 2700.])
    if complete or published or handed:
        tails['head_mtp'] = [0., 0.]
    if published or handed:
        tails['validation_upload'] = [0., 0.]
    if handed:
        tails['placement_cleanup'] = [180., 900.]
    if cleaned:
        tails = {k:[0., 0.] for k in tails}
        result['provisional'] = False
    total = [layer_range[i]+sum(v[i] for v in tails.values()) for i in (0,1)]
    layer_point = 0. if layer_range == [0., 0.] else max(0., remaining_layers*base_seconds-elapsed)
    point = layer_point+sum(statistics.mean(v) for v in tails.values())
    result.update(layer_remaining_seconds_range=layer_range, tail_allowance_seconds=tails,
        tail_basis='planning allowances, not measured Inkling phase durations; upload allowance informed by prior Kimi publication',
        remaining_seconds_range=total,
        remaining_seconds_estimate=point,
        layer_remaining_seconds_estimate=layer_point,
        completion_time_estimate=(now+timedelta(seconds=point)).astimezone(TAIPEI).isoformat(),
        layer_completion_time_range=[(now+timedelta(seconds=s)).astimezone(TAIPEI).isoformat() for s in layer_range],
        completion_time_range=[(now+timedelta(seconds=s)).astimezone(TAIPEI).isoformat() for s in total])
    return result
