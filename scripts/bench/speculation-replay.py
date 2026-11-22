#!/usr/bin/env python3
"""Offline speculation replay over fixed-length draft traces (#2460).

Reads wip-cards job directories whose server ran `DRAFT_LENGTH=<block>` with
`SPECULATION_TRACE`, and writes one tidy row per (round, request) plus the
analysis the brief asks for. Analysis only: it never launches anything.

  speculation-replay.py tidy JOB_DIR... --out DIR      # tidy.jsonl + manifests
  speculation-replay.py analyze DIR --out DIR          # cycle fit, acceptance, replay

Each tidy row carries the family, card, content and phase, the request's
context at the round, the batch's rows and requests, every drafted position's
score / history estimate / outcome, and the round's host-clock times. Times come
from the round record itself (`round_us`/`verify_us`/`draft_us`, or the shared
scheduler's `cycle_ms`/`verify_ms`/`draft_ms`, or GLM 5.3's per-request
`verify_ms`).

Replay. A fixed-L trace reveals the first rejection, so for any rule that
verifies a prefix of the block, the accepted drafts are min(rule length,
accepted run) and the round commits that plus one target token. The round's
time is the family's fitted cycle time at the replayed row count, context
and concurrency. A rule never sees outcomes past the first rejection, and
only a prefix rule is replayable, which every candidate here is.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

CONTENT_KEYS = ('code', 'prose', 'json')


# ---------------------------------------------------------------- loading

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # one unterminated final record when the writer was cut


def request_index(report):
    """HTTP request id -> (panel, content, phase, c, prompt_tokens) from a smoke report.json."""
    index = {}

    def visit(node, panel, content, c, phase):
        if isinstance(node, dict):
            content = node.get('content', content) if isinstance(node.get('content'), str) else content
            c = node.get('c', c) if isinstance(node.get('c'), int) else c
            ctx = node.get('context') if isinstance(node.get('context'), int) else None
            spec = node.get('speculation')
            if isinstance(spec, dict) and spec.get('request_id'):
                index[spec['request_id']] = dict(panel=panel, content=content, c=c, phase=phase,
                    prompt_tokens=node.get('prompt_tokens'), context=ctx)
            timing = node.get('timing')
            if isinstance(timing, dict) and isinstance(timing.get('speculation'), dict):
                rid = timing['speculation'].get('request_id')
                if rid:
                    index[rid] = dict(panel=panel, content=content, c=c, phase=phase,
                        prompt_tokens=timing.get('prompt_tokens'), context=ctx)
            for key, value in node.items():
                visit(value, panel, content, c, key if key in ('warm', 'timed', 'runs', 'requests') else phase)
        elif isinstance(node, list):
            for value in node:
                visit(value, panel, content, c, phase)

    for panel in report.get('panels', []):
        visit(panel, panel.get('id'), None, None, None)
    if isinstance(report.get('baseline'), dict):  # smoke profile: the basic card's decode rows
        visit(report['baseline'], 'baseline', None, None, None)
    # Context points carry the context on the point; propagate to their requests.
    for panel in report.get('panels', []):
        if panel.get('id') != 'context_load':
            continue
        for p in panel.get('passes', []):
            for point in (p.get('points') or []):
                waves = [('timed', [r.get('timing') or {} for r in point.get('requests') or []]),
                         ('warm', point.get('warm_requests') or [])]
                for phase, timings in waves:
                    for timing in timings:
                        rid = (timing.get('speculation') or {}).get('request_id')
                        if rid:
                            index[rid] = dict(panel='context_load', content='context', c=point.get('c'),
                                phase=phase, prompt_tokens=timing.get('prompt_tokens'), context=point.get('context'))
    return index


def round_times(record):
    """(round_us, verify_us, draft_us) from whichever fields the record has."""
    if 'round_us' in record:
        return record.get('round_us'), record.get('verify_us'), record.get('draft_us')
    if 'cycle_ms' in record:
        ms = lambda k: None if record.get(k) is None else int(1000 * record[k])
        return ms('cycle_ms'), ms('verify_ms'), ms('draft_ms')
    return None, None, None


def tidy_job(job_dir):
    """Rows of one job directory: one per (round, request)."""
    job_dir = Path(job_dir)
    job = json.loads((job_dir / 'job.json').read_text())
    entry = job['entry']
    reports = sorted(job_dir.glob('reports/**/report.json'))
    index = request_index(json.loads(reports[0].read_text())) if reports else {}
    family = entry['family']
    rounds, cycles, glm_cycles = [], [], collections.defaultdict(list)
    for record in read_jsonl(job_dir / 'speculation-trace.jsonl'):
        kind = record.get('type') or record.get('kind')
        if kind == 'draft_round':
            rounds.append(record)
            cycles.append(None)
        elif kind == 'cycle':  # GLM 5.3: one line per verified sequence
            glm_cycles[record['id']].append(record)
        elif rounds and cycles[-1] is None and ('cycle_ms' in record or 'neural_features' in record):
            # The family's own round record (shared scheduler `cycle`, GLM Flash/MiMo) follows
            # observe's round record and names the same members.
            ids = record.get('ids') or record.get('request_ids') or []
            if sorted(ids) == sorted(q['id'] for q in rounds[-1]['requests']):
                cycles[-1] = record
    paired = True
    seen = collections.Counter()  # per-request round counter, for GLM 5.3 cycle lines
    generated = collections.Counter()
    rows = []
    for k, record in enumerate(rounds):
        cycle = cycles[k]
        times = round_times(record) if 'round_us' in record else (
            round_times(cycle) if cycle and 'cycle_ms' in cycle else (None, None, None))
        requests = record['requests']
        contexts = record.get('context')
        if contexts is None and cycle and cycle.get('contexts'):
            contexts = [c[0] - 1 if c else None for c in cycle['contexts']]
        batch_rows = sum(q['proposed'] + 1 for q in requests)
        member = {}
        if cycle and cycle.get('neural_features') is not None:
            ids = cycle.get('ids') or cycle.get('request_ids') or []
            member = {rid: f for rid, f in zip(ids, cycle['neural_features']) if f}
        for i, q in enumerate(requests):
            meta = index.get(q.get('request_id') or '', {})
            context = contexts[i] if contexts and i < len(contexts) else None
            verify_ms = None
            features = member.get(q['id']) or []
            if glm_cycles.get(q['id']):
                n = seen[q['id']]
                if n < len(glm_cycles[q['id']]):
                    line = glm_cycles[q['id']][n]
                    context = line.get('position', context)
                    verify_ms = line.get('verify_ms')
                    features = line.get('features') or []
                seen[q['id']] += 1
            if context is None and meta.get('prompt_tokens') is not None:
                context = meta['prompt_tokens'] + generated[q['id']]
            positions = [p for p in q['positions'] if p.get('track', 'neural') == 'neural']
            panel, content, c = meta.get('panel'), meta.get('content'), meta.get('c')
            if not meta:
                # Concurrency waves carry no per-request ids: label short unmatched rounds by batch width.
                panel = 'concurrency' if context is not None and context < 4096 else 'unlabeled'
                content, c = ('distinct-v1', len(requests)) if panel == 'concurrency' else (None, None)
            rows.append(dict(
                family=family, card=job['card'], round=record['round'], lane=record.get('lane'),
                request=q['id'], request_id=q.get('request_id'),
                panel=panel, content=content, c=c, phase=meta.get('phase'),
                context=context, batch_requests=len(requests), batch_rows=batch_rows,
                block=record['block'], policy=record['policy'], room=q['room'],
                proposed=q['proposed'], accepted=q['accepted_drafts'], censored=q['censored'],
                score=[p.get('score') if p.get('score') is not None else p.get('calibrated') if p.get('calibrated') is not None
                       else (features[k][1] if k < len(features) and len(features[k]) > 1 else None)
                       for k, p in enumerate(positions)],
                history=[p.get('p') for p in positions],
                outcome=[p.get('outcome') for p in positions],
                round_us=times[0], verify_us=times[1] if times[1] is not None else (
                    None if verify_ms is None else int(1000 * verify_ms)), draft_us=times[2]))
            rows[-1]['score_source'] = score_source(family, positions, features)
            generated[q['id']] += q['accepted_drafts'] + 1
    return rows


def score_source(family, positions, features):
    """Which per-position signal a row's `score` holds."""
    if any(p.get('head') is not None for p in positions):
        return 'head (dSpark/trained, calibrated)'
    if any(p.get('score') is not None for p in positions):
        return 'raw candidate p (DFlash2 selector)'
    if features:
        return 'raw candidate p (cycle features)'
    return 'history p (position acceptance)'


def manifest(job_dir):
    job_dir = Path(job_dir)
    job = json.loads((job_dir / 'job.json').read_text())
    result = json.loads((job_dir / 'result.json').read_text()) if (job_dir / 'result.json').is_file() else {}
    keep = ('MODEL_ID', 'MODEL_REVISION', 'SPECULATOR', 'SPECULATOR_MODEL_ID', 'CORE_VIEW', 'DRAFT_LENGTH',
            'RTX_GPUS', 'SPARK_COUNT', 'GLM5_FLASH_DECODE_ROWS', 'COORDINATOR_DOCKER_INFERENCE')
    thermal, context_points = [], []
    for report in sorted(job_dir.glob('reports/**/report.json')):
        for panel in json.loads(report.read_text()).get('panels', []):
            if panel.get('id') == 'context_load':
                for p in panel.get('passes', []):
                    for point in p.get('points') or []:
                        context_points.append(dict(context=point.get('context'), c=point.get('c'),
                            status=point.get('status'), aggregate_tok_s=point.get('aggregate_tok_s'),
                            reason=point.get('reason') or ((point.get('warm_errors') or [None])[0])))
            for p in panel.get('passes', []):
                if isinstance(p, dict) and p.get('thermal'):
                    thermal.append(dict(panel=panel.get('id'), thermal=summarize_thermal(p['thermal'])))
    measured = []
    for report in sorted(job_dir.glob('reports/**/report.json')):
        for panel in json.loads(report.read_text()).get('panels', []):
            if panel.get('id') == 'concurrency':
                for p in panel.get('passes', []):
                    for point in p.get('points') or []:
                        measured.append(dict(context='0', c=point.get('c'), tok_s=point.get('aggregate_tok_s')))
    for point in context_points:
        if point['status'] == 'ok':
            measured.append(dict(context=context_band(point['context']), c=point['c'], tok_s=point['aggregate_tok_s']))
    highest = {}
    for point in context_points:
        if point['status'] == 'ok':
            highest[point['context']] = max(highest.get(point['context'], 0), point['c'] or 0)
    return dict(card=job['card'], family=job['entry']['family'], status=result.get('status'),
        error=result.get('error'), launcher=result.get('launcher'), artifacts=result.get('artifacts'),
        settings={k: v for k, v in job['entry']['set'].items() if k in keep}, thermal=thermal,
        context_points=context_points, highest_concurrency_by_context=highest, measured_fixed=measured,
        notes=KNOWN_NOTES.get(job['card'], []))


# Known per-card caveats recorded in the dataset manifest.
KNOWN_NOTES = {
    'qwen38-exl3-min': ['128K C16 failed: packed prefill of 16 prompts charged full per-member page tables '
                        '(shared estimate 143K of a 1.12M pool) and left the engine unusable; #2570. '
                        'Highest level at 128K is C8.'],
    'muse-spark-exl3-k4-mc-max': ['context_load requests often stop at EOS after 8-100 of 256 tokens; '
                                  'long-context cells have fewer rounds.'],
}


def summarize_thermal(window):
    """Per-GPU power-cap and thermal seconds, peak temperature and min T.Limit headroom."""
    out = []
    for gpu in (window.get('gpus') or []) if isinstance(window, dict) else []:
        throttle = gpu.get('throttle_s') or {}
        out.append(dict(index=gpu.get('index'), power_cap_s=throttle.get('sw_power_cap'),
            sw_thermal_s=throttle.get('sw_thermal'), hw_thermal_s=throttle.get('hw_thermal'),
            power_brake_s=throttle.get('hw_power_brake'), max_temperature_c=(gpu.get('temperature_c') or {}).get('max'),
            min_tlimit_margin_c=(gpu.get('tlimit_margin_c') or {}).get('min'),
            busy_sm_clock_mean_mhz=(gpu.get('busy') or {}).get('sm_clock_mean_mhz'),
            min_power_limit_w=(gpu.get('power_limit_w') or {}).get('min')))
    return out


# ---------------------------------------------------------------- analysis

def accepted_run(row):
    """Leading accepted drafts the fixed trace proves (outcomes stop at the first rejection)."""
    n = 0
    for outcome in row['outcome']:
        if outcome is True:
            n += 1
        else:
            break
    return n


def full_block(row):
    """Whether the round verified the whole block (C16 can be cut by the row cap)."""
    return row['proposed'] >= min(row['block'], row['room'])


def context_band(context):
    if context is None:
        return 'unknown'
    for edge, name in ((2048, '0'), (12288, '8K'), (49152, '32K'), (1 << 30, '128K')):
        if context < edge:
            return name
    return '128K'


def concurrency_band(n):
    return 'C1' if n <= 1 else 'C2-4' if n <= 4 else 'C5-8' if n <= 8 else 'C9-16'


def fit_cycle(rows):
    """Linear model, the measured rows table (see `fit_table`) and the per-band
    two-term fit a + b*requests + c*rows; `predict` prices from the table, or
    from the two-term fit when `fit['pricing'] == 'two-term'`."""
    fit = fit_linear(rows)
    if fit is not None:
        fit['table'] = fit_table(rows)
        model = OnlineTwoTerm()
        for t, rws, n, band in round_samples(rows, with_requests=True):
            model.observe(band, rws, t, n)
        fit['two_term'] = {band: model.coef(band) for band in model.count if model.coef(band)}
    return fit


def fit_linear(rows):
    """Least squares round_us ~ a + b*rows + c*rows*ctx_k + d*requests*ctx_k + e*requests per family.

    ctx_k is the batch's mean context in thousands of tokens. Returns
    coefficients and the R^2 over rounds (one sample per round, not per request).
    """
    samples = [(t, rws, n, BAND_K[band]) for t, rws, n, band in round_samples(rows)]
    if len(samples) < 8:
        return None
    X = [[1., rws, rws * ck, n * ck, n] for (_, rws, n, ck) in samples]
    y = [t for (t, *_rest) in samples]
    coef = lstsq(X, y)
    pred = [sum(c * x for c, x in zip(coef, xs)) for xs in X]
    mean = statistics.fmean(y)
    ss_tot = sum((v - mean) ** 2 for v in y) or 1.
    ss_res = sum((v - p) ** 2 for v, p in zip(y, pred))
    return dict(coef=coef, r2=1 - ss_res / ss_tot, rounds=len(samples),
                terms=['1', 'rows', 'rows*ctx_k', 'requests*ctx_k', 'requests'])


def lstsq(X, y, ridge=1e-6):
    """Normal equations with a tiny ridge (pure Python; five columns)."""
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) + (ridge if i == j else 0.) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
    for col in range(k):  # Gauss-Jordan with partial pivoting
        pivot = max(range(col, k), key=lambda r: abs(A[r][col]))
        A[col], A[pivot], b[col], b[pivot] = A[pivot], A[col], b[pivot], b[col]
        d = A[col][col] or 1e-12
        for j in range(col, k):
            A[col][j] /= d
        b[col] /= d
        for r in range(k):
            if r != col and A[r][col]:
                f = A[r][col]
                for j in range(col, k):
                    A[r][j] -= f * A[col][j]
                b[r] -= f * b[col]
    return b


def round_samples(rows, timed_only=True, with_requests=False):
    """One (round_us, rows, requests, context band) per batch round."""
    rounds = {}
    for r in rows:
        if timed_only and r.get('phase') == 'warm':
            continue
        t = r['round_us'] if r['round_us'] is not None else r['verify_us']
        if t is None:
            continue
        key = (r['card'], r['round'], r['lane'])
        e = rounds.setdefault(key, dict(t=t, rows=r['batch_rows'], n=r['batch_requests'], ctx=[]))
        if r['context'] is not None:
            e['ctx'].append(r['context'])
    return [(e['t'], e['rows'], e['n'], context_band(statistics.fmean(e['ctx']) if e['ctx'] else None))
            for e in rounds.values()]


def fit_table(rows):
    """Measured median round time by (context band, rows): the curve the replay prices.

    Rounds at every concurrency contribute to their row count, so a C8 round
    of 64 rows and a C16 round of 64 rows share one point; the request count is
    kept as a column for review but the replay prices rows within a band.
    """
    cells = collections.defaultdict(list)
    for t, rws, n, band in round_samples(rows):
        cells[(band, rws)].append(t)
    table = {}
    for (band, rws), ts in cells.items():
        if len(ts) >= MIN_POINT_ROUNDS:
            table.setdefault(band, {})[rws] = statistics.median(ts)
    return table


MIN_POINT_ROUNDS = 10


def interpolate(curve, rows):
    """Piecewise-linear in rows over a measured {rows: us} curve, flat beyond its ends."""
    xs = sorted(curve)
    if rows <= xs[0]:
        return curve[xs[0]]
    if rows >= xs[-1]:
        return curve[xs[-1]]
    lo = max(x for x in xs if x <= rows)
    hi = min(x for x in xs if x >= rows)
    return curve[lo] if hi == lo else curve[lo] + (curve[hi] - curve[lo]) * (rows - lo) / (hi - lo)


def predict(fit, rows, requests, ctx_k):
    """Round microseconds at `rows`: interpolate the measured table of the
    nearest context band when present, else the linear model."""
    two = fit.get('two_term') if isinstance(fit, dict) and fit.get('pricing') == 'two-term' else None
    if two:
        band = context_band(ctx_k * 1000)
        a, b, c = two.get(band) or two.get('0')
        return max(1., a + b * requests + max(c, 0.) * rows)
    table = fit.get('table') if isinstance(fit, dict) else None
    if table:
        band = context_band(ctx_k * 1000)
        curve = table.get(band) or table.get(min(table, key=lambda b: abs(BAND_K[b] - ctx_k)))
        base = table.get('0')
        if base is None or curve is base or rows >= min(curve):
            return max(1., interpolate(curve, rows))
        # Below the band's smallest measured row count, take the 0-context curve's
        # shape scaled by the band's measured ratio at that row count.
        anchor = min(curve)
        return max(1., interpolate(base, rows) * curve[anchor] / interpolate(base, anchor))
    a, b, c, d, e = fit['coef']
    return max(1., a + b * rows + c * rows * ctx_k + d * requests * ctx_k + e * requests)


BAND_K = {'0': 0., '8K': 8., '32K': 32., '128K': 128., 'unknown': 0.}


# -- rules: each maps a row to the number of drafts it would verify (a prefix length).

def rule_fixed(n):
    return lambda row: min(n, row['proposed'])


class Platt:
    """Python mirror of the engine's online calibrator (draft_history.rs `Platt`):
    p = sigmoid(a*logit + b), Newton step on log-loss with a decayed information
    matrix, theta from [1, 0], prior 2, decay 0.998, slope clamped to [0, 3]."""

    DECAY, PRIOR = 0.998, 2.

    def __init__(self):
        self.theta = [1., 0.]
        self.info = [[self.PRIOR, 0.], [0., self.PRIOR]]

    def apply(self, x):
        z = self.theta[0] * x + self.theta[1]
        return 1 / (1 + math.exp(-max(min(z, 60.), -60.)))

    def observe(self, x, accepted):
        v = (x, 1.)
        p = self.apply(x)
        curvature = max(p * (1 - p), 0.02)
        for i in range(2):
            for j in range(2):
                self.info[i][j] = self.DECAY * self.info[i][j] + curvature * v[i] * v[j]
            self.info[i][i] = max(self.info[i][i], 1e-3)
        (a, b), (c, d) = self.info
        det = a * d - b * c
        if not (math.isfinite(det) and det > 1e-12):
            return
        g = ((float(accepted) - p) * v[0], float(accepted) - p)
        step = ((d * g[0] - b * g[1]) / det, (a * g[1] - c * g[0]) / det)
        self.theta[0] = min(max(self.theta[0] + min(max(step[0], -1.), 1.), 0.), 3.)
        self.theta[1] = min(max(self.theta[1] + min(max(step[1], -2.), 2.), -8.), 8.)


def logit_of(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def attach_calibration(rows):
    """One calibrated acceptance probability per drafted position, `row['prob']`,
    computed once per family in round order (never calibrated twice):

    - engine-calibrated head scores (`score_source` head): used as recorded;
    - raw candidate p: an online per-position Platt over logit(p), the engine's
      own calibrator, applied before it learns from the round (causal; every
      round is held out from its own fit) and trained only on reached positions;
    - no per-token score: the history estimate as recorded.

    Also sets `row['raw']`, the uncalibrated per-token score (or history p)."""
    per_family = collections.defaultdict(list)
    for r in rows:
        per_family[r['family']].append(r)
    for family, fam_rows in per_family.items():
        fam_rows.sort(key=lambda r: (r['card'], r['lane'] or 0, r['round'], r['request']))
        platts = collections.defaultdict(Platt)
        for r in fam_rows:
            source = r.get('score_source') or ''
            scored = any(v is not None for v in r['score'])
            r['raw'] = [v if v is not None else 0. for v in (r['score'] if scored else r['history'])]
            if not scored or source.startswith('head') or source.startswith('history'):
                r['prob'] = list(r['raw'])
                r['calibration'] = 'engine' if source.startswith('head') else 'history'
                continue
            r['prob'] = [platts[k].apply(logit_of(v)) for k, v in enumerate(r['raw'])]
            r['calibration'] = 'online platt (replay)'
            for k, (v, outcome) in enumerate(zip(r['raw'], r['outcome'])):
                if outcome is None:
                    break
                platts[k].observe(logit_of(v), outcome is True)
    return rows


def keep_prefix(values, cut):
    keep = 0
    for v in values:
        if v > cut:
            keep += 1
        else:
            break
    return keep


def rule_history(cut=None, raw_cut=0.5, head_cut=0.3, history_cut=0.3, calibrated=False):
    """`history`: keep while the per-token value clears its cut, plus one explore.

    calibrated=False reproduces dev: raw DFlash2 candidate p against 0.5, the
    engine-calibrated dSpark head against 0.3, history p against 0.3.
    calibrated=True keys every scored family on its one calibrated probability
    against `cut` (the sweep TJ asked for); token-only families keep history p.

    Replay caveat: the history estimate in a fixed-length trace saw every position
    of every round, so for token-only drafters it is what a fixed policy would
    estimate, not what `history`'s own shorter rounds would have produced.
    """
    def length(row):
        source = row.get('score_source') or ''
        if source.startswith('history'):
            c, vals = history_cut, row['raw']
        elif calibrated:
            c, vals = cut, row['prob']
        elif source.startswith('head'):
            c, vals = head_cut, row['prob']
        else:
            c, vals = raw_cut, row['raw']
        return min(keep_prefix(vals, c) + 1, row['proposed'])
    return length


def rule_product(threshold):
    """Keep while the running product of the calibrated probabilities stays above
    `threshold` (the prefix's expected survival), plus one explore."""
    def length(row):
        product, keep = 1., 0
        for v in row['prob']:
            product *= v
            if product > threshold:
                keep += 1
            else:
                break
        return min(keep + 1, row['proposed'])
    return length


def settle(lengths, members, max_rows=64):
    """Dev `settle`: fit the round into `max_rows` (anchors included), granting
    positions across requests by highest next score, each keeping a prefix."""
    budget = max_rows - len(members)
    if sum(lengths) <= budget:
        return lengths
    scores = [m['prob'] for m in members]
    granted = [0] * len(members)
    for _ in range(max(budget, 0)):
        best = None
        for i, m in enumerate(members):
            if granted[i] >= lengths[i]:
                continue
            v = scores[i][granted[i]] if granted[i] < len(scores[i]) else 0.
            key = (v, -granted[i], -m['request'])
            if best is None or key > best[0]:
                best = (key, i)
        if best is None:
            break
        granted[best[1]] += 1
    return granted


class OnlineCost:
    """Round time learned online from earlier rounds of the same trace: a
    running median per (context band, verify rows) with linear interpolation in
    rows; until a band has points, the 0-context band scaled by its first point,
    and before any observation, the trace's first measured round."""

    def __init__(self):
        self.samples = collections.defaultdict(lambda: collections.defaultdict(list))

    def observe(self, band, rows, us):
        bucket = self.samples[band][rows]
        bucket.append(us)
        if len(bucket) > 64:
            bucket.pop(0)

    def curve(self, band):
        return {r: statistics.median(v) for r, v in self.samples.get(band, {}).items() if v}

    def predict(self, band, rows):
        curve = self.curve(band)
        if curve:
            return max(1., interpolate(curve, rows))
        base = self.curve('0')
        if base:
            return max(1., interpolate(base, rows))
        return None


class OnlineRatio:
    """TJ's ratio-only cost: round time ~ base + rows*per_row, fitted online per
    context band by least squares over earlier rounds only, exposing just
    r = per_row/base. The marginal test is scale-free, so predict() returns time
    in units of `base`: 1 + r*rows. Records r after every round for convergence."""

    def __init__(self):
        self.sums = collections.defaultdict(lambda: [0.] * 5)  # n, sx, sy, sxx, sxy
        self.history = collections.defaultdict(list)

    def observe(self, band, rows, us):
        n, sx, sy, sxx, sxy = self.sums[band]
        self.sums[band] = [n + 1, sx + rows, sy + us, sxx + rows * rows, sxy + rows * us]
        self.history[band].append(self.ratio(band))

    def ratio(self, band):
        n, sx, sy, sxx, sxy = self.sums.get(band, [0.] * 5)
        if n < 3 or n * sxx - sx * sx <= 0:
            return None
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        base = (sy - slope * sx) / n
        return slope / base if base > 0 else None

    def predict(self, band, rows):
        r = self.ratio(band)
        if r is None:
            r = self.ratio('0')
        return None if r is None else 1. + max(r, 0.) * rows

    def convergence(self, tolerance=0.1):
        """Rounds until the band's online r stays within `tolerance` of its final value."""
        out = {}
        for band, values in self.history.items():
            final = values[-1] if values else None
            if not final:
                continue
            last_bad = max((i for i, v in enumerate(values) if v is None or abs(v - final) > tolerance * abs(final)),
                           default=-1)
            out[band] = dict(final_ratio=final, rounds_to_10pct=last_bad + 1, rounds=len(values))
        return out


class OnlineTwoTerm:
    """Round time t = a + b*requests + c*rows, fitted online per context band by
    least squares over earlier rounds only. A marginal draft row joins a request
    already in the batch, so its cost is c (per row at fixed requests and
    context), not the rows-only curve's slope, which mixes in per-request cost
    when rows grew with concurrency. `ratio=True` exposes only c / (a + b*n),
    TJ's scale-free variant. Until a band identifies c (rank 3), the 0-context
    band's fit is used; with neither, no prediction."""

    def __init__(self, ratio=False):
        self.ratio = ratio
        self.xtx = collections.defaultdict(lambda: [[0.] * 3 for _ in range(3)])
        self.xty = collections.defaultdict(lambda: [0.] * 3)
        self.count = collections.Counter()
        self.history = collections.defaultdict(list)

    def observe(self, band, rows, us, requests=1):
        x = (1., float(requests), float(rows))
        for i in range(3):
            self.xty[band][i] += x[i] * us
            for j in range(3):
                self.xtx[band][i][j] += x[i] * x[j]
        self.count[band] += 1
        coef = self.coef(band)
        self.history[band].append(None if coef is None else coef[2] / max(coef[0] + coef[1] * requests, 1.))

    def coef(self, band):
        if self.count[band] < 8:
            return None
        A = [row[:] for row in self.xtx[band]]
        b = self.xty[band][:]
        for i in range(3):
            A[i][i] += 1e-6 * (1 + A[i][i])
        try:
            sol = solve3(A, b)
        except ZeroDivisionError:
            return None
        return sol if sol[0] + sol[1] > 0 else None

    def predict(self, band, rows, requests=1):
        coef = self.coef(band) or self.coef('0')
        if coef is None:
            return None
        a, b, c = coef
        c = max(c, 0.)
        base = max(a + b * requests, 1.)
        return 1. + (c / base) * rows if self.ratio else base + c * rows

    def convergence(self, tolerance=0.1):
        out = {}
        for band, values in self.history.items():
            final = next((v for v in reversed(values) if v is not None), None)
            if not final:
                continue
            last_bad = max((i for i, v in enumerate(values) if v is None or abs(v - final) > tolerance * abs(final)),
                           default=-1)
            out[band] = dict(final_ratio=final, rounds_to_10pct=last_bad + 1, rounds=len(values))
        return out


def solve3(A, b):
    """Gauss-Jordan for a 3x3 system (raises ZeroDivisionError when singular)."""
    A = [row[:] + [v] for row, v in zip(A, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(A[r][col]))
        if abs(A[pivot][col]) < 1e-12:
            raise ZeroDivisionError
        A[col], A[pivot] = A[pivot], A[col]
        for r in range(3):
            if r != col:
                f = A[r][col] / A[col][col]
                A[r] = [x - f * y for x, y in zip(A[r], A[col])]
    return [A[i][3] / A[i][i] for i in range(3)]


def replay(rows, fit, rule, batch_rule=None, max_rows=64, cost_model=None):
    """Expected emitted tok/s of `rule` over the rows' batches, with time from `fit`.

    A batch round is the set of rows sharing (card, round, lane). Its replayed
    rows are sum(1 + L_i); committed tokens are sum(1 + min(L_i, accepted_i)).
    `batch_rule`, when given, maps a round's rows to their lengths jointly and is
    given an online cost model that only ever saw earlier rounds; the round is
    still *priced* from `fit` like every other rule.
    """
    rounds = collections.defaultdict(list)
    for r in rows:
        rounds[(r['card'], r['round'], r['lane'])].append(r)
    online = (cost_model or OnlineCost)()
    tokens = seconds = verify_rows = 0.
    for key in sorted(rounds, key=lambda k: (k[0], k[1], k[2] or 0)):
        members = rounds[key]
        ctx = [m['context'] for m in members if m['context'] is not None]
        ck = statistics.fmean(ctx) / 1000 if ctx else 0.
        band = context_band(ck * 1000)
        lengths = batch_rule(members, online, band) if batch_rule else settle([rule(m) for m in members], members, max_rows)
        committed = sum(1 + min(L, accepted_run(m)) for L, m in zip(lengths, members))
        nrows = sum(1 + L for L in lengths)
        t = predict(fit, nrows, len(members), ck)
        measured = members[0]['round_us'] if members[0]['round_us'] is not None else members[0]['verify_us']
        if measured is not None:
            if isinstance(online, OnlineTwoTerm):
                online.observe(band, members[0]['batch_rows'], measured, members[0]['batch_requests'])
            else:
                online.observe(band, members[0]['batch_rows'], measured)
        tokens += committed
        seconds += t / 1e6
        verify_rows += nrows
    out = dict(tok_s=tokens / seconds if seconds else 0., tokens=tokens, rounds=len(rounds),
               rows_per_round=verify_rows / max(len(rounds), 1), tokens_per_round=tokens / max(len(rounds), 1))
    if isinstance(online, (OnlineRatio, OnlineTwoTerm)):
        out['convergence'] = online.convergence()
    return out


def rule_marginal(members, cost, band, max_rows=64):
    """Greedy over the whole batch: add the next row with the highest expected
    acceptance (the calibrated product along its prefix) while expected tokens
    per expected round time still rise, with round time from `cost`, an online
    model of earlier rounds only. With no cost observed yet, verify full blocks
    (the trace's own shape) for this round."""
    n = len(members)
    price = (lambda rows: cost.predict(band, rows, n)) if isinstance(cost, OnlineTwoTerm) else (lambda rows: cost.predict(band, rows))
    if price(n) is None:
        return settle([m['proposed'] for m in members], members, max_rows)
    lengths, cum = [0] * n, [1.] * n
    rows, expected = n, float(n)
    t = price(rows)
    while rows < max_rows:
        best, gain = None, 0.
        for i, m in enumerate(members):
            if lengths[i] >= m['proposed']:
                continue
            g = cum[i] * m['prob'][lengths[i]]
            if g > gain:
                best, gain = i, g
        if best is None:
            break
        t_next = price(rows + 1)
        if (expected + gain) / t_next <= expected / t:
            break
        cum[best] *= members[best]['prob'][lengths[best]]
        lengths[best] += 1
        rows += 1
        expected += gain
        t = t_next
    return lengths


def acceptance_table(rows):
    """Per position: conditional and cumulative acceptance, by family x context band x content."""
    table = collections.defaultdict(lambda: [[0, 0, 0] for _ in range(16)])  # reached, accepted, rounds
    for r in rows:
        key = (r['family'], context_band(r['context']), r['content'] or 'unknown')
        run = accepted_run(r)
        for k, outcome in enumerate(r['outcome'][:16]):
            cell = table[key][k]
            cell[2] += 1
            if outcome is None:
                continue
            cell[0] += 1
            cell[1] += outcome is True
    out = []
    for (family, band, content), cells in sorted(table.items()):
        cumulative = 1.
        for k, (reached, accepted, total) in enumerate(cells):
            if not total:
                continue
            conditional = accepted / reached if reached else None
            cumulative = cumulative * conditional if conditional is not None else None
            out.append(dict(family=family, context=band, content=content, position=k + 1, reached=reached,
                accepted=accepted, conditional=conditional, cumulative=cumulative))
            if cumulative is None:
                break
    return out


def reliability(rows, field='score', bins=10):
    """Observed acceptance per score bin over reached positions, by family."""
    table = collections.defaultdict(lambda: [[0, 0] for _ in range(bins)])
    for r in rows:
        for value, outcome in zip(r[field], r['outcome']):
            if value is None or outcome is None:
                continue
            b = min(int(value * bins), bins - 1)
            table[r['family']][b][0] += 1
            table[r['family']][b][1] += outcome is True
    return {f: [dict(bin=f'{i/bins:.1f}-{(i+1)/bins:.1f}', n=n, accepted=a / n if n else None)
                for i, (n, a) in enumerate(cells)] for f, cells in table.items()}


CUTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)
RULES = {
    **{f'fixed-{n}': rule_fixed(n) for n in range(0, 8)},
    'history': rule_history(),
    **{f'history@{t}': rule_history(cut=t, calibrated=True) for t in CUTS},
    **{f'product>{t}': rule_product(t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)},
}
BATCH_RULES = {'marginal-rowsonly': OnlineCost, 'marginal-ratio-rowsonly': OnlineRatio,
               'marginal': OnlineTwoTerm, 'marginal-ratio': lambda: OnlineTwoTerm(ratio=True)}
RULE_PREFIXES = ('fixed-', 'history', 'product', 'marginal')


def analyze(rows, weights=None, manifests=(), pricing='table'):
    """Everything the report needs, from tidy rows (and, for calibration, the
    manifests' measured fixed-length tok/s per context x concurrency)."""
    measured = {}
    for m in manifests:
        for point in m.get('measured_fixed') or []:
            measured[(m['family'], point['context'], point['c'])] = point['tok_s']
    attach_calibration(rows)
    fits = {}
    by_family = collections.defaultdict(list)
    for r in rows:
        by_family[r['family']].append(r)
    for family, fam_rows in by_family.items():
        fits[family] = fit_cycle(fam_rows)
        if fits[family] is not None:
            fits[family]['pricing'] = pricing
    cells = []
    acceptance_rows = acceptance_table(rows)
    for family, fam_rows in by_family.items():
        fit = fits[family]
        if fit is None:
            continue
        groups = collections.defaultdict(list)
        for r in fam_rows:
            groups[(context_band(r['context']), r['c'] or r['batch_requests'])].append(r)
        for (band, c), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            rounds = {(g['card'], g['round'], g['lane']) for g in group}
            # Replay only on batches where every member is in this cell: a round mixing cells is split.
            row = dict(family=family, context=band, c=c, rounds=len(rounds),
                       truncated=sum(not full_block(g) for g in group) / len(group))
            for name, rule in RULES.items():
                row[name] = replay(group, fit, rule)['tok_s']
            for name, model in BATCH_RULES.items():
                result = replay(group, fit, None, rule_marginal, cost_model=model)
                row[name] = result['tok_s']
                if 'convergence' in result:
                    row['ratio_convergence'] = result['convergence'].get(band)
            cut = max(CUTS, key=lambda t: row[f'history@{t}'])
            row['best_cut'], row['best_cut_tok_s'] = cut, row[f'history@{cut}']
            # The replay prices verify rounds only; scale every rule by measured / replayed
            # fixed-length tok/s in this cell so fixed-7 reproduces the measurement.
            fixed_name = f"fixed-{max(g['block'] for g in group)}"
            got = measured.get((family, band, c))
            row['calibration'] = got / row[fixed_name] if got and row.get(fixed_name) else None
            best = max((k for k in row if k.startswith(RULE_PREFIXES) and k != 'fixed-0'
                        and isinstance(row[k], float)), key=lambda k: row[k])
            row['best'] = best
            cells.append(row)
    sources = {}
    for r in rows:
        sources.setdefault(r['family'], collections.Counter())[r.get('score_source') or '?'] += 1
    sources = {f: c.most_common(1)[0][0] for f, c in sources.items()}
    for row in acceptance_rows:
        row['score_source'] = sources.get(row['family'])
    return dict(fits=fits, cells=cells, sources=sources, acceptance=acceptance_rows,
                reliability=reliability(rows, 'prob'), mix=traffic_mix(cells, weights),
                cut_sweep=cut_sweep(cells, weights), ratio=ratio_summary(by_family, fits))


def cut_sweep(cells, weights=None):
    """Best calibrated history cut per family x context band (mean tok/s over that
    band's concurrency cells), and per family in the traffic mix; plus one shared
    cut chosen to maximize the sum over families of mix tok/s relative to each
    family's own best."""
    weights = weights or DEFAULT_MIX
    out = {'by_band': {}, 'by_family': {}}
    families = sorted({c['family'] for c in cells})
    for family in families:
        for band in weights:
            group = [c for c in cells if c['family'] == family and c['context'] == band]
            if not group:
                continue
            means = {t: statistics.fmean(c[f'history@{t}'] for c in group) for t in CUTS}
            best = max(means, key=means.get)
            out['by_band'][f'{family}/{band}'] = dict(best_cut=best, tok_s=means[best],
                at_0_3=means[0.3], dev_history=statistics.fmean(c['history'] for c in group))
    mix = traffic_mix(cells, weights)
    for family in families:
        means = {t: mix[family].get(f'history@{t}') for t in CUTS}
        best = max((t for t in CUTS if means[t] is not None), key=lambda t: means[t])
        out['by_family'][family] = dict(best_cut=best, tok_s=means[best], relative={t: means[t] / means[best] for t in CUTS})
    shared = {t: statistics.fmean(out['by_family'][f]['relative'][t] for f in families) for t in CUTS}
    out['shared_cut'] = max(shared, key=shared.get)
    out['shared_relative'] = {f: out['by_family'][f]['relative'][out['shared_cut']] for f in families}
    return out


def ratio_summary(by_family, fits):
    """Online r = per_row/base per family x band over each family's full trace order,
    against the offline least-squares r of its measured table; and rounds to 10%."""
    out = {}
    for family, fam_rows in by_family.items():
        rounds = {}
        for r in sorted(fam_rows, key=lambda r: (r['card'], r['round'], r['lane'] or 0)):
            if r.get('phase') == 'warm':
                continue
            t = r['round_us'] if r['round_us'] is not None else r['verify_us']
            if t is not None:
                rounds.setdefault((r['card'], r['round'], r['lane']), (context_band(r['context']), r['batch_rows'], t))
        model = OnlineRatio()
        for band, rows_, t in rounds.values():
            model.observe(band, rows_, t)
        conv = model.convergence()
        out[family] = {band: dict(conv[band], offline_ratio=offline_ratio(fits.get(family), band))
                       for band in conv}
    return out


def offline_ratio(fit, band):
    curve = ((fit or {}).get('table') or {}).get(band)
    if not curve or len(curve) < 2:
        return None
    xs = sorted(curve)
    n = len(xs); sx = sum(xs); sy = sum(curve[x] for x in xs)
    sxx = sum(x * x for x in xs); sxy = sum(x * curve[x] for x in xs)
    den = n * sxx - sx * sx
    if den <= 0:
        return None
    slope = (n * sxy - sx * sy) / den
    base = (sy - slope * sx) / n
    return slope / base if base > 0 else None


DEFAULT_MIX = {'0': 0.25, '8K': 0.25, '32K': 0.3, '128K': 0.2}


def traffic_mix(cells, weights=None):
    """Context-weighted mean tok/s per family and rule (half or more of the weight at 32K+)."""
    weights = weights or DEFAULT_MIX
    assert sum(w for k, w in weights.items() if k in ('32K', '128K')) >= 0.5 - 1e-9, 'brief: >= half the weight at 32K+'
    out = {}
    for family in {c['family'] for c in cells}:
        rules = [k for k in cells[0] if k.startswith(RULE_PREFIXES) and isinstance(cells[0][k], float)]
        row = {}
        for rule in rules:
            total = weight = 0.
            for band, w in weights.items():
                values = [c[rule] for c in cells if c['family'] == family and c['context'] == band]
                if values:
                    total += w * statistics.fmean(values)
                    weight += w
            row[rule] = total / weight if weight else None
        out[family] = row
    return out


def render(result):
    """Markdown tables: cycle curves, replay cells and the traffic mix."""
    src = result.get('sources', {})
    label = lambda f: f"{f} [{src.get(f, '?')}]"
    lines = ['## Score source per family', ''] + [f'- {f}: {s}' for f, s in sorted(src.items())]
    lines += ['', '## Cycle time (median ms per batch round, by verify rows)', '']
    for family, fit in sorted(result['fits'].items()):
        if not fit:
            continue
        for band, curve in sorted(fit['table'].items(), key=lambda kv: BAND_K.get(kv[0], 0)):
            pts = ', '.join(f'{r}:{t / 1000:.1f}' for r, t in sorted(curve.items()) if r in (1, 2, 4, 8, 16, 24, 32, 48, 56, 64, 96, 128))
            lines.append(f'- {family} {band}: {pts}')
    rules = ['history', 'fixed-1', 'fixed-2', 'fixed-3', 'fixed-4', 'fixed-7', 'product>0.3', 'product>0.5', 'marginal']
    lines += ['', '## Replay tok/s by family x context x concurrency', '',
              'Replayed verify-round tok/s. `calib` is measured / replayed fixed-length tok/s for the cell (prefill, ramp and tail',
              'overheads the replay does not price); multiply a rule by it for an absolute estimate. Ratios between rules need no calibration.', '',
              '| Family | Context | C | rounds | trunc | calib | ' + ' | '.join(rules) + ' | best |',
              '|' + '---|' * (len(rules) + 7)]
    for c in result['cells']:
        if c['rounds'] < 20:
            continue
        cal = '' if c.get('calibration') is None else f"{c['calibration']:.2f}"
        lines.append(f"| {label(c['family'])} | {c['context']} | {c['c']} | {c['rounds']} | {c['truncated']:.2f} | {cal} | "
                     + ' | '.join(f"{c[r]:.1f}" for r in rules) + f" | {c['best']} |")
    lines += ['', '## Traffic mix (25% 0, 25% 8K, 30% 32K, 20% 128K)', '', '| Family | ' + ' | '.join(rules) + ' |',
              '|' + '---|' * (len(rules) + 1)]
    for family, row in sorted(result['mix'].items()):
        lines.append(f'| {label(family)} | ' + ' | '.join('' if row.get(r) is None else f'{row[r]:.1f}' for r in rules) + ' |')
    return '\n'.join(lines) + '\n'


CONTEXT_STEPS = {'0': '#86b6ef', '8K': '#5598e7', '32K': '#256abf', '128K': '#104281'}  # one hue, light -> dark


def plot_cycles(result, path):
    """Small multiples, one panel per family on shared axes: median ms per batch round
    vs verify rows, one line per context band (sequential, light = short)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fits = {f: fit for f, fit in sorted(result['fits'].items()) if fit and fit.get('table')}
    n = len(fits)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.7 * rows + 0.6), sharex=False, sharey=True,
                             squeeze=False, facecolor='#fcfcfb')
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (family, fit) in zip(axes.flat, fits.items()):
        ax.set_visible(True)
        ax.set_facecolor('#fcfcfb')
        for band in ('0', '8K', '32K', '128K'):
            curve = fit['table'].get(band)
            if not curve:
                continue
            xs = sorted(x for x in curve if x <= 64)
            ax.plot(xs, [curve[x] / 1000 for x in xs], color=CONTEXT_STEPS[band], lw=2, marker='o', ms=3,
                    label=f'{band} context')
        ax.set_title(family, fontsize=10, color='#0b0b0b', loc='left')
        ax.grid(True, color='#e6e5e0', lw=0.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#b7b6b0')
        ax.tick_params(colors='#52514e', labelsize=8)
        ax.set_xlim(0, 68)
        ax.set_xticks([0, 16, 32, 48, 64])
        ax.set_xlabel('verify rows per batch round', fontsize=8, color='#52514e')
    for ax in axes[:, 0]:
        ax.set_ylabel('ms per round (median)', fontsize=8, color='#52514e')
    handles, labels = next(ax for ax in axes.flat if ax.get_visible()).get_legend_handles_labels()
    fig.suptitle('Round time vs verify rows, by context (DRAFT_LENGTH = native block, dev)', y=0.99,
                 fontsize=11, color='#0b0b0b')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tidy')
    t.add_argument('jobs', nargs='+')
    t.add_argument('--out', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('dataset')
    a.add_argument('--out', required=True)
    a.add_argument('--plot', action='store_true', help='also write cycle-curves.png (needs matplotlib)')
    a.add_argument('--pricing', choices=('table', 'two-term'), default='table',
                   help='price replayed rounds from the measured rows table or the per-band a + b*requests + c*rows fit')
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == 'tidy':
        with open(out / 'tidy.jsonl', 'a') as f:
            for job in args.jobs:
                rows = tidy_job(job)
                for row in rows:
                    f.write(json.dumps(row) + '\n')
                m = manifest(job)
                (out / 'manifests').mkdir(exist_ok=True)
                (out / 'manifests' / f"{m['card']}.json").write_text(json.dumps(m, indent=1))
                timed = sum(r['round_us'] is not None for r in rows)
                print(f"{job}: {len(rows)} request-rounds, {timed} timed, "
                      f"{sum(r['context'] is not None for r in rows)} with context")
    else:
        rows = list(read_jsonl(Path(args.dataset) / 'tidy.jsonl'))
        manifests = [json.loads(p.read_text()) for p in sorted((Path(args.dataset) / 'manifests').glob('*.json'))]
        result = analyze(rows, manifests=manifests, pricing=args.pricing)
        (out / 'analysis.json').write_text(json.dumps(result, indent=1, default=str))
        (out / 'report.md').write_text(render(result))
        if args.plot:
            plot_cycles(result, out / 'cycle-curves.png')
        print(json.dumps({'families': list(result['fits']), 'cells': len(result['cells'])}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
