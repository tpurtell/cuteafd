#!/usr/bin/env python3
"""Assert zero CUDA captures in every timed sweep point, for any family.

Usage: check-graph-census.py --log coordinator.log --sweep sweep.json
Requires PROFILE=census. Thermal INFO timestamps anchor sweep's process-relative
clock to UTC; wrapper cache-miss messages are deliberately not counted.
"""
import argparse
import collections
import datetime
import json
import math
from pathlib import Path
import re
import statistics
import sys

STAMP = re.compile(r'^(\d{4}-\d\d-\d\dT\S+Z)\s')
SITE = re.compile(r'\bsite=(?:"([^"]+)"|(\S+))')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
# Sweep edges round to milliseconds. Keep boundary cases conservative rather
# than certifying a capture clean because the clocks differ by a fraction of ms.
EDGE_S = 0.002
MAX_SKEW_S = 0.050


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('clock value must be a finite number')
    return float(value)


def check(log, sweep):
    anchors, captures, stamps = [], [], []
    evidence = False
    decoder = json.JSONDecoder()
    for raw in Path(log).open():
        line = ANSI.sub('', raw)
        stamp = STAMP.match(line)
        utc = datetime.datetime.fromisoformat(stamp[1].replace('Z', '+00:00')).timestamp() if stamp else None
        if utc is not None:
            stamps.append(utc)
        if 'cuteafd::thermal:' in line and 'current=' in line:
            if utc is None:
                raise ValueError('thermal anchor lacks UTC timestamp')
            current, _ = decoder.raw_decode(line.split('current=', 1)[1])
            anchors.append(utc - finite(current['queried_s']))
        if 'graph census executable site' in line:
            evidence = True
        if 'graph census executable site' in line:
            site = SITE.search(line)
            if utc is None or site is None:
                raise ValueError('capture record lacks timestamp or site')
            captures.append((utc, site[1] or site[2]))
        if 'capture_write_failed' in line:
            raise ValueError('native census capture records were lost')
    if not anchors:
        raise ValueError('missing thermal UTC/queried_s clock anchors')
    if not evidence:
        raise ValueError('missing PROFILE=census evidence; absence of logs is not zero captures')
    skew = max(anchors) - min(anchors)
    if skew > MAX_SKEW_S:
        raise ValueError(f'inconsistent thermal clock anchors: {skew:.6f} s spread')
    epoch = statistics.median(anchors)
    # Use the full observed offset range, plus rounded-edge slack.
    low, high = min(anchors) - EDGE_S, max(anchors) + EDGE_S
    data = json.loads(Path(sweep).read_text())
    points = [('decode', p) for p in data.get('points', [])]
    points += [('prefill', p) for p in data.get('prefill', [])]
    if not points:
        raise ValueError('sweep contains no timed points')
    windows = []
    for kind, point in points:
        thermal = point.get('thermal')
        if not isinstance(thermal, dict):
            raise ValueError('timed point lacks thermal window')
        start, end = finite(thermal['from_s']), finite(thermal['to_s'])
        if start < 0 or end <= start:
            raise ValueError('invalid timed window')
        if not stamps or min(stamps) > low + start or max(stamps) < high + end:
            raise ValueError('coordinator log does not bracket timed window')
        sites = collections.Counter(site for utc, site in captures if low + start <= utc <= high + end)
        windows.append({
            'kind': kind, 'phase': point.get('phase', kind), 'concurrency': point.get('c'),
            'context': point.get('context', point.get('length')),
            'from_unix_s': epoch + start, 'to_unix_s': epoch + end,
            'captures': sum(sites.values()), 'sites': dict(sorted(sites.items())),
        })
    return {'status': 'fail' if any(w['captures'] for w in windows) else 'pass',
            'clock_anchors': len(anchors), 'clock_spread_s': skew,
            'edge_slack_s': EDGE_S, 'total_logged_captures': len(captures), 'windows': windows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--sweep', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = check(args.log, args.sweep)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'error', 'error': str(error)}))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
