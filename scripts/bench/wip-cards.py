#!/usr/bin/env python3
"""Task-private WIP cards. Build/seal first, then hold one fair serving turn.

The union of the smoke jobs' locks stays held across every arm and repeat, until
the last smoke has torn down. Children inherit CUTEAFD_HW_LOCKS_HELD, so smoke
does not take those locks again. Builds (including legacy builds and seed/GPU
exports) finish before this turn; dry-run prints the turn and is CPU-only.

Stopping: SIGTERM, SIGINT or SIGHUP stops the active build or reaches every running smoke, which
tears down its coordinator and Spark workers (bounded, 60 s) before it exits;
wip-cards waits for that (75 s, up to SMOKE_STOP_GRACE_S), starts no further job
and exits 128 + the signal. This fits hwlock.py's 80 s outer stop grace (if
wrapped) and systemd's 90 s stop timeout. Launch it as a systemd user unit with
the defaults (KillMode=control-group, TimeoutStopSec=90 s), e.g.
  systemd-run --user --unit=TASK --collect --setenv=PATH="$PATH" --setenv=HOME="$HOME" \\
    --working-directory=REPO scripts/bench/wip-cards.py ...
and stop it with `systemctl --user stop TASK`: the locks are released only
after the containers are gone. Never pass KillMode=mixed or KillMode=process
(only the main process would be signalled, so smoke would not tear down), a
TimeoutStopSec below 90 s (systemd would SIGKILL the teardown), and never
`systemctl kill -s KILL` or `kill -9` a running card.
"""
from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import contextlib
import datetime
import math
import functools
import itertools
import hashlib
from http.client import HTTPException
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import signal
import statistics
import socket
import threading
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib

REPO = Path(__file__).resolve().parents[2]
_cdi_spec = importlib.util.spec_from_file_location('cdi_gpus', REPO / 'scripts/lib/cdi_gpus.py')
_cdi = importlib.util.module_from_spec(_cdi_spec)
_cdi_spec.loader.exec_module(_cdi)
cdi_gpu_args = _cdi.cdi_gpu_args
sys.path.insert(0, str(REPO / 'scripts/lib'))
from wip_aot_scopes import exl3_spark_layouts, family_scopes
_hw_spec = importlib.util.spec_from_file_location('hwlock', REPO / 'scripts/hw/hwlock.py')
hwlock = importlib.util.module_from_spec(_hw_spec)
_hw_spec.loader.exec_module(hwlock)
# Profilers that wrap a serving process (PROFILE on the coordinator, PROFILE_SPARK
# on Spark workers), and their report suffix.
PROFILER_REPORTS = {'nsys': '.nsys-rep', 'ncu': '.ncu-rep'}
VERIFY_SHAPE = re.compile(r'\d+(x\d+)?(,\d+(x\d+)?)*(@\d+)?')
NSYS_SESSION = 'cuteafd-profile'  # RELEASE_NSYS_SESSION in scripts/lib/release-common.sh
HOSTS = ('ostrich', 'dodo', 'emu', 'kiwi', 'rhea', 'moa')
# LAN-only build seeds: compile, export and seal Spark slots, never serve. They have
# no RoCE (10 GbE only): their staging lanes are their LAN address and wip.sh moves
# their sealed slots through sparknest. Keep in sync with release_lan_seed_hosts
# (scripts/lib/release-common.sh). kakapo passed its gate 2026-10-13 (cold build
# sealed, distributed through sparknest, served at C1 with 0 errors).
SEED_HOSTS = ('kakapo', 'cassowary')
SEEDS = ('auto', 'rhea', 'moa', *SEED_HOSTS)
# 'auto': wip.sh picks the less-loaded LAN-only seed, then rhea/moa. A named seed
# is a placement request (wip.sh --seed).
DEFAULT_SEED = 'auto'
CORRECTNESS = {'sim-5090', 'fidelity', 'cache', 'image', 'no-fit', 'packed-check'}
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
METRICS = ('C1', 'C8', '8K', 'kl', 'top1', 'pool', 'readiness_s')
# Before -> after table: baseline card metrics, then the client sweep's levels
# (sweep C<n>) and post-load C1, which are separate measurements.
SUMMARY_METRICS = ('C1', 'C8', 'sweep C1', 'sweep C4', 'sweep C8', 'sweep C16', 'sustained C1', 'sustained C4', 'sustained C8', 'sustained C16', 'post-load C1', '8K', '8K w/tail', 'kl', 'top1', 'pool', 'readiness_s')
FIDELITY_PANELS = {'fidelity', 'fidelity_full'}
IMAGE_PREFIX_PROBES = {'image-prefix', 'image-prefix-partial'}
IMAGE_PROBES = {'image'} | IMAGE_PREFIX_PROBES
SCOPED_PANELS = {'hardware', 'configuration', 'decode_content', 'concurrency', 'spec_lossless_set'}
LOWER_IS_BETTER = {'kl', 'readiness_s'}  # also every 'prefill N TTFT s'
FAILURE = re.compile(r'(?i)(\berror\b|\bpanicked\b|\bfatal\b|\bfailed\b|undefined symbol|no such file|refus|exited:|timed out|out of memory|traceback)')
@functools.lru_cache(maxsize=None)
def bench_panels(binary='cuteafd'):
    """Ask the host cuteafd binary (`cuteafd bench panels --json`) for the selectable panel ids."""
    cmd = str(binary or shutil.which('cuteafd') or 'cuteafd')
    try:
        result = subprocess.run([cmd, 'bench', 'panels', '--json'],
                                capture_output=True, text=True, timeout=30)
    except OSError as error:
        raise ValueError(f'{cmd} bench panels --json failed: {error}') from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()[-1:]
        raise ValueError(f'{cmd} bench panels --json failed: {detail}')
    try:
        panels = json.loads(result.stdout)
    except ValueError as error:
        raise ValueError(f'{cmd} bench panels --json returned invalid JSON: {error}') from error
    if not isinstance(panels, list) or not panels or not all(isinstance(p, str) and p for p in panels):
        raise ValueError(f'{cmd} bench panels --json returned invalid panel list')
    return frozenset(panels)


def short_instance(value):
    if not NAME.fullmatch(value):
        raise ValueError('invalid instance: ' + value)
    return value if len(value) <= 41 else value[:32] + '-' + hashlib.sha256(value.encode()).hexdigest()[:8]


def launcher_defaults(repo=REPO):
    """Read the launcher's default EXPERT_PORT and ADDR from scripts/launch/run-family.sh."""
    text = (Path(repo) / 'scripts/launch/run-family.sh').read_text()
    defaults = {match[1]: match[2] for match in re.finditer(r'\$\(get\s+(EXPERT_PORT|ADDR)\s+([^\s")]+)\)', text)}
    if {'EXPERT_PORT', 'ADDR'} - defaults.keys():
        raise ValueError('missing EXPERT_PORT/ADDR defaults in scripts/launch/run-family.sh')
    return defaults


LAUNCHER_DEFAULTS = launcher_defaults()


def resolve_container_keys(entry):
    """Resolve and validate the container-name and endpoint keys a launch uses."""
    values = entry.get('set', {})
    instance = str(values.get('INSTANCE', ''))
    if not instance or len(instance) > 41 or not NAME.fullmatch(instance):
        raise ValueError(f'INSTANCE must be [A-Za-z0-9_.-] (1..41 chars): {instance!r}')
    port = str(values['EXPERT_PORT']) if 'EXPERT_PORT' in values else LAUNCHER_DEFAULTS['EXPERT_PORT']
    if not port.isascii() or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(f'EXPERT_PORT must be in 1..65535: {port!r}')
    addr = str(values['ADDR']) if 'ADDR' in values else LAUNCHER_DEFAULTS['ADDR']
    sparks = list(entry.get('sparks', []))
    for host in sparks:
        if not isinstance(host, str) or not NAME.fullmatch(host) or host not in HOSTS:
            raise ValueError(f'invalid Spark host: {host!r}')
    return dict(instance=instance, expert_port=port, addr=addr, sparks=sparks)


def container_targets(job):
    resolved = resolve_container_keys(job['entry'])
    targets = [('raptor', 'cuteafd-coordinator-' + resolved['instance'])]
    targets += [(host, f"cuteafd-spark-expert-{host}-{resolved['expert_port']}") for host in job.get('spark_role_hosts', resolved['sparks'])]
    return targets


def load_lossless_set(path):
    """An absolute, checked spec_lossless_set case file; the server re-validates."""
    path = Path(path).expanduser().resolve()
    cases = json.loads(path.read_text())
    if not isinstance(cases, list) or not cases or len(cases) > 64:
        raise ValueError(f'{path}: a lossless set is a JSON list of 1 to 64 cases')
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) - {'id', 'prompt', 'nonce', 'source', 'reconstructed'} \
                or not all(isinstance(case.get(k), str) and case[k].strip() for k in ('id', 'prompt', 'nonce')):
            raise ValueError(f'{path}: each case needs string id, prompt and nonce (optional source, reconstructed)')
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', case['id']) or case['id'] in ids:
            raise ValueError(f"{path}: invalid or duplicate case id {case['id']!r}")
        if re.search(r'[\[\]\x00-\x1f]', case['nonce']):
            raise ValueError(f"{path}: case {case['id']}: nonce has brackets or control characters")
        ids.add(case['id'])
    return str(path)


def parse_lossless_dump(text, set_path=None):
    """`CASE[,CASE...]:START-END` for the spec_lossless_set panel's full-vocabulary row dumps:
    both arms, rows predicting generated tokens START..END (inclusive, 0..127)."""
    match = re.fullmatch(r'([A-Za-z0-9_.,-]+):(\d+)-(\d+)', text.strip())
    if not match:
        raise ValueError(f'{text!r}: expected CASE[,CASE...]:START-END, e.g. code-2:70-80')
    cases = match.group(1).split(',')
    start, end = int(match.group(2)), int(match.group(3))
    if not all(cases) or len(set(cases)) != len(cases) or not start <= end <= 127:
        raise ValueError(f'{text!r}: distinct case ids and a window 0 <= START <= END <= 127')
    known = {c['id'] for c in json.loads(Path(set_path or REPO / 'scripts/bench/fixtures/spec-lossless-set-v1.json').read_text())}
    unknown = [c for c in cases if c not in known]
    if unknown:
        raise ValueError(f"{text!r}: cases not in the lossless set: {', '.join(unknown)}")
    return dict(cases=cases, start=start, end=end)


def requested_panels(profile, binary=None):
    if not profile.startswith('panels:'):
        return []
    panels = [p.strip() for p in profile[7:].split(',') if p.strip()]
    if not panels:
        return []
    allowed = bench_panels(binary or shutil.which('cuteafd') or 'cuteafd')
    bad = set(panels) - allowed - {'baseline'}
    if bad:
        raise ValueError('unknown requested panel: ' + ', '.join(sorted(bad)))
    return panels


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def config(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    return values


def run(cmd, **kwargs):
    timeout = kwargs.pop('timeout', 120)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=timeout, **kwargs)


def card_layout(card, overrides=None):
    """(coordinator GPUs, Spark hosts) a card launches on: its resolved layout.

    The single source every lock derivation follows (#2364): COORDINATOR_GPUS
    (defaulting to the card's gpus) and the first SPARK_COUNT ranks' hosts
    (defaulting to the card's sparks). `overrides` are arm config values,
    applied on top exactly as generate() does.
    """
    values = card.get('resolved_values')
    if values is None:
        values = {**config(card['config']), **card.get('set', {})}
    if overrides:
        values = {**values, **overrides}
    try:
        gpus = [int(x) for x in str(values.get('COORDINATOR_GPUS', '')).split(',') if x]
    except ValueError:
        raise ValueError('invalid coordinator GPUs') from None
    gpus = gpus or list(card.get('gpus', [0]))
    if not 1 <= len(gpus) <= 2 or len(set(gpus)) != len(gpus) or not set(gpus) <= {0, 1}:
        raise ValueError('invalid coordinator GPUs')
    hosts = list(card.get('sparks', []))
    try:
        count = int(values.get('SPARK_COUNT', len(hosts)))
    except ValueError:
        raise ValueError('invalid Spark count') from None
    if not 0 <= count <= len(HOSTS):
        raise ValueError('invalid Spark count')
    hosts = [values.get(f'SPARK_{i}_HOST', hosts[i] if i < len(hosts) else HOSTS[i]) for i in range(count)]
    if len(set(hosts)) != count or not set(hosts) <= set(HOSTS):
        raise ValueError('invalid Spark inventory')
    return gpus, hosts


def arms_from(values, releases=()):
    arms = {}
    for value in values:
        name, address = value.split('=', 1)
        instance, slot = address.split(':', 1)
        if not all(NAME.fullmatch(x) for x in (name, instance, slot)) or len(instance) > 41 or len(slot) > 64:
            raise ValueError('invalid arm, WIP instance or slot')
        if name in arms or instance in [a['instance'] for a in arms.values()]:
            raise ValueError('arm names and WIP instances must be unique')
        arms[name] = {'instance': instance, 'slot': slot}
    for value in releases:
        name, address = value.split('=', 1)
        instance, tag = address.split(':', 1)
        if not all(NAME.fullmatch(x) for x in (name, instance, tag)) or len(instance) > 41 or len(tag) > 128:
            raise ValueError('invalid release arm, instance or tag')
        if name in arms or instance in [a['instance'] for a in arms.values()]:
            raise ValueError('arm names and instances must be unique')
        arms[name] = {'instance': instance, 'slot': None, 'release': tag}
    return arms


def shared_wip(arms, mappings, builds, parallel):
    if mappings and (builds or parallel != 1):
        raise ValueError('--arm-wip refuses --build and parallel runs')
    for value in mappings:
        name, instance = value.split('=', 1)
        if name not in arms or 'release' in arms[name] or not NAME.fullmatch(instance) or len(instance) > 41:
            raise ValueError('invalid --arm-wip mapping')
        if 'wip_instance' in arms[name]:
            raise ValueError('duplicate --arm-wip mapping')
        arms[name]['wip_instance'] = instance
    # Arms of one WIP instance may name different sealed slots (e.g. a base and a
    # candidate source built in the same instance): launches stage per slot and
    # serving instance (release_wip_serving_layout), and --arm-wip runs one arm
    # at a time, so slots never swap files under each other.


def load_cards(kit, names):
    cards = {}
    for matrix in sorted(kit.glob('matrix-*.json')):
        data = json.loads(matrix.read_text())
        for raw in data['entries']:
            entry = {**data.get('defaults', {}), **raw}
            entry['set'] = {**data.get('defaults', {}).get('set', {}), **raw.get('set', {})}
            entry['profile'] = entry.get('profile', data.get('profile', 'smoke'))
            entry['source_matrix'] = str(matrix)
            if entry['name'] in cards:
                raise ValueError('duplicate card name: ' + entry['name'])
            if not NAME.fullmatch(entry['name']):
                raise ValueError('unsafe card name')
            path = Path(entry['config']).expanduser()
            if not path.is_absolute():
                # Kit-relative, else repository-relative (in-repo kits name examples/configs/*).
                path = kit / path if (kit / path).is_file() else REPO / path
            if not path.is_file():
                path = kit / 'cfg' / path.name
            entry['config'] = str(path.resolve())
            cards[entry['name']] = entry
    missing = set(names) - cards.keys()
    if missing:
        raise ValueError('unknown cards: ' + ', '.join(sorted(missing)))
    return [cards[name] for name in names]


def overrides_from(items):
    overrides = {}
    for item in items:
        card, setting = item.split(':', 1)
        key, value = setting.split('=', 1)
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', key) or '\n' in value or '\r' in value:
            raise ValueError('invalid override')
        if key in {'API_KEY_FILE', 'CUTEAFD_API_KEY', 'WIP_INSTANCE', 'WIP_ROOT', 'INSTANCE', 'BENCH_SIMULATED'}:
            raise ValueError('driver owns override ' + key)
        overrides.setdefault(card, {})[key] = value
    return overrides


def arm_env_from(items, arms):
    """Per-arm private diagnostics, so one sealed binary can run an off/on pair.
    Only CUTEAFD_DIAG_* names: the launcher forwards exactly those, and nothing
    else may differ between arms through the environment."""
    out = {}
    for item in items:
        name, setting = item.split(':', 1)
        key, value = setting.split('=', 1)
        if name not in arms:
            raise ValueError('unknown --arm-env arm: ' + name)
        if not re.fullmatch(r'CUTEAFD_DIAG_[A-Z0-9_]+', key) or '\n' in value or '\r' in value:
            raise ValueError('--arm-env takes CUTEAFD_DIAG_* names only: ' + key)
        out.setdefault(name, {})[key] = value
    return out


def probe_selection(items, cards, arms):
    probes = {card['name']: set(card.get('probes', [])) | {'fabric'} for card in cards}
    scoped = {}
    for item in items:
        selector = item.split(':')
        if len(selector) == 2:
            card, values = selector
            arm = None
        elif len(selector) == 3:
            arm, card, values = selector
            if arm not in arms:
                raise ValueError('probe arm must be selected')
        else:
            raise ValueError('probe needs CARD:probes or ARM:CARD:probes')
        if card not in probes:
            raise ValueError('probe card must be selected')
        values = set(filter(None, values.split(',')))
        if not values <= IMAGE_PROBES | {'memory', 'console', 'fabric'}:
            raise ValueError('unknown probe')
        if arm is None:
            probes[card] = values
        else:
            scoped.setdefault(arm, {})[card] = values
    return probes, scoped


# Default --client limit; smoke bounds every hook at 7200 s.
CLIENT_TIMEOUT_S = 3600


def generate(entry, arm_name, arm, state, key_file, overrides, probes, expected_pool, repeat, trace=False, exact_state=None,
             cold_hosts=None, binary=None, client=None, client_timeout=CLIENT_TIMEOUT_S):
    entry = json.loads(json.dumps(entry))
    requested_panels(entry.get('profile', 'smoke'), binary=binary)
    entry['fabric'] = 'fabric' in probes
    gpus, hosts = card_layout(entry, overrides)
    values = entry.pop('resolved_values', None)
    if values is None:
        values = config(entry['config'])
        values.update(entry.pop('set', {}))
    else:
        entry.pop('set', None)
    simulated = values.pop('BENCH_SIMULATED', '') or 'sim5090' in entry['name'] or 'sim-5090' in entry['name']
    values.pop('BENCH_HARDWARE_CLASS', None)
    values.update(overrides)
    ring_expectation = values.pop('BENCH_EXPECT_GLM_RING', None)
    if ring_expectation is not None:
        observer = entry.get('observer')
        if not observer or not observer.get('argv'):
            raise ValueError('BENCH_EXPECT_GLM_RING requires an observer')
        if probes & {'memory', 'console'}:
            raise ValueError('BENCH_EXPECT_GLM_RING requires its own observer, not a probe replacement')
        observer['argv'] = ['env', 'BENCH_EXPECT_GLM_RING=' + ring_expectation, *observer['argv']]
    if 'CUTEAFD_API_KEY' in values:
        raise ValueError('inline API keys are forbidden')
    # The sanitized base also removes publication-only keys from original configs.
    values.pop('SPARK_HOSTS', None)
    count = len(hosts)
    for key in list(values):
        if re.fullmatch(r'SPARK_[0-7]_(HOST|LANE_A|LANE_B)', key):
            del values[key]
    # Only V4.1's launcher reads a secondary rail; other families reject the key.
    lane_b = entry.get('family') == 'deepseek_v41'
    for i, host in enumerate(hosts):
        index = HOSTS.index(host) + 1
        values.update({f'SPARK_{i}_HOST': host, f'SPARK_{i}_LANE_A': f'10.55.0.{index}'})
        if lane_b:
            values[f'SPARK_{i}_LANE_B'] = f'10.55.1.{index}'
    values.update(API_KEY_FILE=str(key_file), ENABLE_BENCH='on', WIP_INSTANCE=arm.get('wip_instance', arm['instance']), INSTANCE=arm['instance'], SPARK_COUNT=str(count), RTX_GPUS=str(len(gpus)), COORDINATOR_GPUS=','.join(map(str, gpus)))
    run_args = ['--wip', arm['slot']]
    if 'release' in arm:
        values.pop('WIP_INSTANCE', None)
        values.pop('WIP_ROOT', None)
        values['COORDINATOR_DOCKER_INFERENCE'] = 'ghcr.io/tpurtell/cuteafd-coordinator:' + arm['release']
        if count > 0:
            values['SPARK_EXPERT_DOCKER_INFERENCE'] = 'ghcr.io/tpurtell/cuteafd-spark-expert:' + arm['release']
        run_args = []
    elif arm.get('root'):
        values['WIP_ROOT'] = arm['root']
    card = entry['name']
    values['INSTANCE'] = short_instance(f"{arm['instance']}-{card}-r{repeat}")
    if len(values['INSTANCE']) > 41:
        raise ValueError('serving INSTANCE exceeds 41 characters')
    label = f'{card}-{arm_name}-r{repeat}' + ('-simulated' if simulated else '')
    dest = state / label
    dest.mkdir(parents=True, exist_ok=True)
    if values.get('PROFILE') or values.get('PROFILE_SPARK') or 'memory' in probes:
        # One per-job directory, mounted on the coordinator and every Spark at
        # the same path; Spark-side files are collected back after the run.
        values['PROFILE_DIR'] = str(dest / 'profile')
        (dest / 'profile').mkdir(exist_ok=True)
        if 'memory' in probes:
            save(dest / 'profile' / 'observer-context.json', {'pending': True})
        # nsys sessions start paused (the coordinator's under PROFILE=nsys, each
        # selected Spark rank's under PROFILE_SPARK=nsys); smoke opens them after
        # the sweep's first warm wave and closes them (writing the reports) at
        # sweep end, or before teardown if the sweep stopped in between.
        sessions = []
        resolved = resolve_container_keys(dict(set=values, sparks=hosts))
        if 'nsys' in values.get('PROFILE', '').split(','):
            # PROFILE_NSYS_ROOT runs its own nsys; the window must talk to the same one.
            nsys = values['PROFILE_NSYS_ROOT'] + '/nsys' if values.get('PROFILE_NSYS_ROOT') else '/usr/local/cuda/bin/nsys'
            sessions.append(['docker', 'exec', 'cuteafd-coordinator-' + resolved['instance'], nsys])
        if values.get('PROFILE_SPARK') == 'nsys':
            for rank in spark_profile_ranks(values, hosts):
                worker = f"cuteafd-spark-expert-{hosts[rank]}-{resolved['expert_port']}"
                sessions.append(['ssh', '-o', 'BatchMode=yes', hosts[rank], 'docker', 'exec', worker, '/usr/local/cuda/bin/nsys'])
        if sessions:
            if not sweeping(entry):
                raise ValueError(card + ': PROFILE=nsys and PROFILE_SPARK=nsys capture the timed sweep; pass --levels, --prefill-lengths or --verify-shapes')
            # A cold window drops the clean page cache first, so the profiled
            # timed wave reads cold (warm-up already ran before the window).
            purge = [cold_purge_argv(cold_hosts)] if cold_hosts else []
            entry['window'] = dict(
                open=dict(argv=window_argv(purge + [s + ['start', '--session=' + NSYS_SESSION] for s in sessions]), timeout_s=300, required=True),
                close=dict(argv=window_argv([s + ['stop', '--session=' + NSYS_SESSION] for s in sessions]), timeout_s=1800, required=True))
    elif 'PROFILE_DIR' in values:
        raise ValueError(card + ': PROFILE_DIR is owned by the driver; set PROFILE only')
    if client and (exact_state or cold_hosts):
        raise ValueError(card + ': --client excludes --exact-state and --cold-window')
    if cold_hosts and entry.get('prefill_lengths'):
        # Every prefill point starts cold: the purge runs before its warm-up and
        # again before its timed request. Decode levels after them are
        # post-prefill, not cold.
        entry['prefill_purge'] = dict(argv=cold_purge_argv(cold_hosts), timeout_s=300, required=True)
    elif cold_hosts and not entry.get('window'):
        if not entry.get('levels'):
            raise ValueError(card + ': --cold-window drops the page cache before the timed sweep; pass --levels or --prefill-lengths')
        if exact_state:
            raise ValueError(card + ': --cold-window and --exact-state both use the measurement window')
        entry['window'] = dict(open=dict(argv=cold_purge_argv(cold_hosts), timeout_s=300, required=True),
                               close=dict(argv=['true'], timeout_s=10, required=True))
    if exact_state:
        # A family's exact prefix-state golden, run against the live server while
        # smoke holds the card's locks (CUTEAFD_SMOKE_URL); output in the job dir.
        if entry.get('window'):
            raise ValueError(card + ': --exact-state uses the measurement window; drop PROFILE=nsys')
        if not (entry.get('levels') or entry.get('prefill_lengths')):
            raise ValueError(card + ': --exact-state runs at the end of the sweep; pass --levels')
        # Smoke exports the live server's address as CUTEAFD_SMOKE_URL to every hook.
        argv = [*exact_state, '--output', str(dest / 'exact-state')]
        entry['window'] = dict(open=dict(argv=['true'], timeout_s=10, required=True),
                               close=dict(argv=['sh', '-c', 'exec "$@" --resume-url "$CUTEAFD_SMOKE_URL"', 'exact-state', *argv],
                                          timeout_s=3600, required=True))
    if entry.get('lossless_dump') or 'image-prefix-partial' in probes:
        # The server writes each dumped probe under this job's directory (mounted at the same path).
        if values.get('PROBE_DUMP_ROOT'):
            raise ValueError(card + ': the selected row dump owns PROBE_DUMP_ROOT; drop the explicit setting')
        root = dest / ('lossless-rows' if entry.get('lossless_dump') else 'image-prefix-rows')
        root.mkdir(exist_ok=True)
        values['PROBE_DUMP_ROOT'] = str(root)
    if trace:
        if values.get('SPECULATION_TRACE'):
            raise ValueError(card + ': --trace owns SPECULATION_TRACE; drop the explicit setting')
        values['SPECULATION_TRACE'] = str(dest / 'speculation-trace.jsonl')
    base = dest / 'base.config'
    base.write_text(''.join(f'{k}={v}\n' for k, v in sorted(values.items())))
    metadata = {k: entry.pop(k) for k in ('source_matrix', 'correctness_only', 'published', 'compared', 'kind', 'probes', 'expected_pool') if k in entry}
    entry.update(name=label, config=str(base), set=values, gpus=gpus, sparks=hosts, run_args=run_args)
    job = dict(card=card, arm=arm_name, repeat=repeat, simulated=bool(simulated), state=str(dest), entry=entry, metadata=metadata, probes=probes, expected_pool=expected_pool, trace=bool(trace))
    hook = [sys.executable, str(Path(__file__).resolve())]
    if 'census' in values.get('PROFILE', '').split(',') and sweeping(entry):
        sample = dict(argv=hook + ['--census-final-sample', str(dest / 'job.json')], timeout_s=180, required=True)
        window = entry.setdefault('window', dict(open=dict(argv=['true'], timeout_s=10, required=True)))
        previous = window.get('close')
        if previous:
            sample['argv'] = window_argv([previous['argv'], sample['argv']])
            sample['timeout_s'] += previous.get('timeout_s', 0)
        window['close'] = sample
    timeout = int(entry.get('timeout_s', 900)) + int(entry.get('run_timeout_s', 900)) * entry_passes(entry) + 120
    if probes & {'memory', 'console'} or trace:
        entry['observer'] = dict(argv=hook + ['--observe', str(dest / 'job.json')], timeout_s=min(timeout, 7200),
                                 finish_timeout_s=OBSERVER_FINISH_TIMEOUT_S, required=True)
        if 'memory' in probes:
            entry['observer']['cleanup'] = sampler_cleanup(job)
    # Attest the deployment against the live server before any existing hook or timing.
    previous = entry.get('precheck')
    check = hook + ['--precheck', str(dest / 'job.json')]
    check_timeout = 1800 if probes & IMAGE_PREFIX_PROBES else 180
    if previous:
        check = precheck_argv(check, previous['argv'])
        check_timeout += previous.get('timeout_s', 0)
    entry['precheck'] = dict(argv=check, timeout_s=check_timeout, required=True)
    if 'memory' in probes and hosts and (entry.get('levels') or entry.get('prefill_lengths')):
        sample = dict(argv=hook + ['--spark-memory-boundary', str(dest / 'job.json')], timeout_s=20, required=True)
        window = entry.setdefault('window', dict(open=dict(argv=['true'], timeout_s=10, required=True)))
        previous = window.get('close')
        if previous:
            sample['argv'] = window_argv([previous['argv'], sample['argv']])
            sample['timeout_s'] += previous.get('timeout_s', 0)
        window['close'] = sample
    if client:
        # A client workload against the live server once it is ready, inside the
        # card's turn (smoke's precheck hook: after readiness, before teardown,
        # required). It runs in JOB/client with CUTEAFD_SMOKE_URL, the server's
        # API key file (CUTEAFD_API_KEY_FILE) and CUTEAFD_CLIENT_OUT exported.
        if entry.get('profile') != 'panels:' or entry.get('levels') or entry.get('prefill_lengths') or entry.get('window'):
            raise ValueError(card + ': --client needs --panels none and no sweep, levels or measurement window')
        if not 1 <= client_timeout <= 7200:
            raise ValueError(card + ': --client-timeout must be 1..7200 seconds')
        out = dest / 'client'
        out.mkdir(exist_ok=True)
        argv = ['env', f'CUTEAFD_CLIENT_OUT={out}', f'CUTEAFD_API_KEY_FILE={key_file}',
                'sh', '-c', 'cd "$CUTEAFD_CLIENT_OUT" && exec "$@"', 'client', *client]
        previous = entry.get('precheck')
        entry['precheck'] = dict(argv=precheck_argv(previous['argv'], argv) if previous else argv,
                                 timeout_s=client_timeout + (previous['timeout_s'] if previous else 0), required=True)
        if entry['precheck']['timeout_s'] > 7200:
            raise ValueError(card + ': --client-timeout plus the card\'s own precheck exceeds smoke\'s 7200 s hook limit')
        job['client'] = argv
    # Re-checked at grant: smoke runs prelaunch once it holds this card's locks.
    entry['prelaunch'] = dict(argv=hook + ['--preflight-job', str(dest / 'job.json')], timeout_s=300, required=True)
    save(dest / 'job.json', {**job, 'probes': sorted(probes)})
    save(dest / 'matrix.json', dict(build='wip-cards', profile=entry.get('profile', 'smoke'), entries=[entry]))
    return job


def apply_arm_config(job, overrides, unsets):
    """Apply explicit launch-schema edits after generated defaults, on both copies."""
    if not overrides and not unsets:
        return
    if overrides.keys() & unsets:
        raise ValueError('arm config key is both set and unset')
    values = job['entry']['set']
    values.update(overrides)
    for key in unsets:
        values.pop(key, None)
    dest = Path(job['state'])
    Path(job['entry']['config']).write_text(''.join(f'{k}={v}\n' for k, v in sorted(values.items())))
    save(dest / 'matrix.json', dict(build='wip-cards', profile=job['entry'].get('profile', 'smoke'), entries=[job['entry']]))


def parallel_safe(jobs, parallel):
    if parallel <= 1:
        return
    for job in jobs:
        e = job['entry']
        meta = job['metadata']
        kind = meta.get('kind', 'sim-5090' if job['simulated'] else None)
        if not (meta.get('correctness_only') is True or kind in CORRECTNESS) or meta.get('published') or meta.get('compared'):
            raise ValueError('--parallel is correctness-only; performance card: ' + job['card'])
        profile = e.get('profile', 'smoke')
        # Smoke always includes a baseline, even for panels:. Those incidental
        # timings are discarded for correctness-only parallel cards, not compared.
        if meta.get('profile_published'):
            raise ValueError('parallel profile is published')
    for a, b in itertools.combinations(jobs, 2):
        ea, eb = a['entry'], b['entry']
        ra, rb = resolve_container_keys(ea), resolve_container_keys(eb)
        if set(ea['gpus']) & set(eb['gpus']) or set(ra['sparks']) & set(rb['sparks']):
            raise ValueError('parallel cards overlap GPUs or Sparks')
        for key, field in (('ADDR', 'addr'), ('EXPERT_PORT', 'expert_port')):
            if ra[field] == rb[field]:
                raise ValueError('parallel cards overlap ' + key)
        if ra['instance'] == rb['instance']:
            raise ValueError('parallel cards need separate instances')


def schedule(cards, arms, repeats, interleave):
    for repeat in range(1, repeats + 1):
        for card in cards:
            order = list(arms.items())
            if interleave and repeat % 2 == 0:
                order.reverse()
            for name, arm in order:
                yield card, name, arm, repeat


def smoke_groups(jobs, parallel):
    if parallel > 1 and len({j['arm'] for j in jobs}) == 1:
        return [jobs]
    return [[job] for job in jobs]


def layout_locks(layouts):
    """The locks a serving turn needs, from resolved (GPUs, Spark hosts) layouts.

    Mirrors smoke.rs locks_for: every coordinator GPU the layout resolves to,
    `sparks` only when a pool host (ostrich/dodo/emu/kiwi) is attached, and
    rhea/moa only when those out-of-pool hosts are used (they have their own
    locks). `exclusive` is a scheduling rule inside one smoke run (V4.1 runs
    alone), not a claim on both GPUs: it never widens this set (#2364). GPU0
    no longer implies `sparks`; card_layout resolves SPARK_COUNT and the hosts
    explicitly, so a zero-Spark layout takes no pool lock.
    """
    names = set()
    for gpus, hosts in layouts:
        if set(hosts) & set(HOSTS[:4]):
            names.add('sparks')
        names.update(f'gpu{gpu}' for gpu in gpus)
        names.update(set(hosts) & {'rhea', 'moa'})
    return hwlock.canonical(names)


def serving_locks(jobs):
    """The serving turn's lock set; mirror of smoke.rs locks_for (parity test)."""
    layouts = []
    for job in jobs:
        entry = job['entry']
        gpus = entry.get('gpus')
        layouts.append(([0] if gpus is None else gpus, entry.get('sparks') or []))
    return layout_locks(layouts)


# Driver work per job inside the turn beyond smoke's own budgets: the at-grant
# job preflight, the leftover-container check and result collection.
JOB_DRIVER_SLACK_S = 600


def sweeping(entry):
    """Whether smoke runs a client-side sweep: decode levels, prefill lengths or verify shapes."""
    return bool(entry.get('levels') or entry.get('prefill_lengths') or entry.get('sweep_verify'))


# Same-server passes: smoke caps them like the server caps panel passes.
MAX_SWEEP_PASSES = 20


def entry_passes(entry):
    """Same-server passes of the sweep and context_load (`sweep_passes`, default 1)."""
    return int(entry.get('sweep_passes') or 1)


def runs_context_load(profile):
    """Whether a profile runs the server's context_load panel (explicit panels, or the
    built-in profiles that plan it)."""
    if profile.startswith('panels:'):
        return 'context_load' in [p.strip() for p in profile[7:].split(',')]
    return profile in ('long', 'full')


def pass_snapshots(sweep, context_records):
    """Every same-server pass's per-band evidence, in order: the sweep's points
    (speed), prefill points, graph bands (#2649 deltas, #2670 sites and thermals)
    and phase thermals, plus that pass's context_load points and bands. Bands
    carry their pass number when there is more than one pass."""
    sweeps = (sweep.get('passes') or [sweep]) if sweep else []
    count = max(len(sweeps), len(context_records))
    if count <= 1:
        return []
    snapshots = []
    for index in range(count):
        number = index + 1
        snapshot = dict(passes=count, **{'pass': number})
        if index < len(sweeps):
            one = sweeps[index]
            snapshot['sweep'] = dict(points=one.get('points', []), prefill=one.get('prefill', []),
                                     thermal=one.get('thermal', {}),
                                     graph_bands=[{**band, 'pass': number} for band in one.get('graph_bands', [])])
            if one.get('verify'):
                snapshot['sweep']['verify'] = one['verify']
        if index < len(context_records):
            record = context_records[index]
            snapshot['context_load'] = dict(points=record.get('points', []), thermal=record.get('thermal'),
                                            graph_bands=[{**band, 'pass': number} for band in record.get('graph_bands', [])])
        snapshots.append(snapshot)
    return snapshots


def job_max_run(job):
    """Seconds one job may hold the serving turn: smoke's launch and run budget
    (timeout_s + run_timeout_s + 120 s, the observer's bound), every hook's own
    timeout, smoke's teardown and the driver's checks around it."""
    entry = job['entry']
    # run_timeout_s bounds each same-server pass (smoke scales the panel deadline).
    run = int(entry.get('run_timeout_s', 900)) * entry_passes(entry)
    bound = int(entry.get('timeout_s', 900)) + run + 120
    if sweeping(entry) and entry.get('profile') != 'panels:':
        bound += run  # panels and a sweep: smoke bounds only the panels by run_timeout_s
    hooks = [entry.get('prelaunch'), entry.get('precheck'), *(entry.get('window') or {}).values()]
    bound += sum(int(hook['timeout_s']) for hook in hooks if hook)
    if entry.get('prefill_purge'):
        # Before each prefill point's warm-up and again before its timed request.
        points = len(entry.get('prefill_lengths') or []) * (2 if entry.get('prefill_repeat') else 1) * entry_passes(entry)
        bound += 2 * points * int(entry['prefill_purge']['timeout_s'])
    return bound + SMOKE_STOP_GRACE_S + JOB_DRIVER_SLACK_S


def held_by_ancestor(env=None):
    """Locks an enclosing hwlock hold already has for this process (`CUTEAFD_HW_LOCKS_HELD`)."""
    return hwlock.canonical(filter(None, (os.environ if env is None else env).get('CUTEAFD_HW_LOCKS_HELD', '').split(',')))


@contextlib.contextmanager
def serving_turn(jobs, env, wait):
    """Hold one post-build turn until every smoke has exited and torn down.

    The caller drains detached smokes before leaving, even on a stop signal:
    60 s smoke teardown, 75 s here, 80 s outer hwlock if used, 90 s systemd.
    Never put a build or export inside this scope. The turn is bounded by the
    sum of its jobs' budgets (`job_max_run`, so --run-timeout raises it), which
    hwlock records and enforces, and which lets it backfill ahead of an older
    ticket it cannot delay.

    Under an enclosing hwlock hold (`CUTEAFD_HW_LOCKS_HELD`), the turn's set
    must already be covered: an inner ticket for a lock the outer hold does not
    cover queues behind tickets waiting on the outer hold itself and cannot be
    granted (#2364). Fail at once instead.
    """
    parent = held_by_ancestor(env)
    locks = serving_locks(jobs)
    if parent:
        missing = [name for name in locks if name not in parent]
        if missing:
            raise ValueError(
                f'serving turn needs {",".join(locks)}, but the enclosing hwlock holds {",".join(parent)}: '
                f'an inner ticket for {",".join(missing)} cannot be granted while the outer hold waits. '
                f'Rerun under: hwlock.py --locks {",".join(locks)}')
    wanted = [name for name in locks if name not in parent]
    with contextlib.ExitStack() as stack:
        if wanted:
            stack.enter_context(hwlock.hold(wanted, wait, 'wip-cards arms/repeats',
                                            cancelled=lambda: STOP['signal'] is not None,
                                            max_run=sum(job_max_run(job) for job in jobs)))
        env['CUTEAFD_HW_LOCKS_HELD'] = ','.join(hwlock.canonical([*parent, *locks]))
        yield


def adopt_roots(arms):
    """An existing (not --build) arm's root is its coordinator container's /wip
    mount source, the same root the seal check reads, so the smoke binary and the
    launch config's WIP_ROOT name the slot that was checked. A build that used a
    custom WIP_ROOT otherwise resolved to the default wip-<instance> layout."""
    for arm in arms.values():
        if 'release' in arm or arm.get('root'):
            continue
        container = 'cuteafd-coordinator-wip-' + arm.get('wip_instance', arm['instance'])
        try:
            found = subprocess.run(['docker', 'inspect', '-f',
                                    '{{range .Mounts}}{{if eq .Destination "/wip"}}{{.Source}}{{end}}{{end}}', container],
                                   capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if found.returncode == 0 and found.stdout.strip():
            arm['root'] = found.stdout.strip()


def slot_binary(arm):
    """The smoke runner a sealed WIP slot ships: the arm's own coordinator build."""
    root = arm.get('root') or str(Path.home() / '.cache/cuteafd/builds' / ('wip-' + arm.get('wip_instance', arm['instance'])))
    return Path(root) / 'slots' / arm['slot'] / 'coordinator/workspace/.cuteafd-wip/cuteafd'


def resolve_binary(binary, arm, dry=False):
    """The cuteafd that runs one arm's smoke: --binary as given, else the one on
    PATH, else the arm's own sealed slot binary. An agent shell rarely has cuteafd
    on PATH, and the slot binary is the runner that arm's build shipped, so it is
    the right default per arm (an A/B of two builds keeps each arm's own smoke)."""
    if binary:
        if not dry and not (Path(binary).is_file() or shutil.which(binary)):
            raise ValueError(f'--binary {binary} is neither a file nor on PATH')
        return binary
    found = shutil.which('cuteafd')
    if found:
        return found
    if dry:
        return 'cuteafd'
    if 'release' in arm:
        raise ValueError('--binary is required for a release arm: it serves published images and ships no host binary')
    path = slot_binary(arm)
    if not path.is_file():
        raise ValueError(f'--binary: cuteafd is not on PATH and this arm\'s sealed slot has no binary at {path}; '
                         'build the arm (--build) or pass --binary')
    return str(path)


def panel_binary(binary, arms):
    """The host cuteafd binary for panel queries: --binary, else cuteafd on PATH,
    else the first arm's sealed slot binary."""
    if binary:
        return binary
    found = shutil.which('cuteafd')
    if found:
        return found
    for arm in (arms or {}).values():
        if 'release' not in arm and slot_binary(arm).is_file():
            return str(slot_binary(arm))
    return 'cuteafd'


def binary_supports(binary, repo, job, needs):
    """Asks the smoke binary itself (a CPU-only --dry-run of this job's matrix)
    whether it knows each needed entry field; an older binary would ignore them."""
    if not needs:
        return
    result = subprocess.run([binary, 'bench', 'smoke', '--matrix', str(Path(job['state']) / 'matrix.json'), '--repo', str(repo),
                             '--state', str(Path(job['state']) / 'smoke-dry'), '--dry-run'],
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise ValueError(f"{binary} bench smoke --dry-run failed: {(result.stderr or result.stdout).strip().splitlines()[-1:]}")
    if 'matched_panel_window' in needs and 'matched_panel_window=true' not in result.stderr:
        raise ValueError(f'{binary} lacks matched panel windows; pass a current shared host runner with --binary')
    if 'warm_prelude' in needs and 'warm_prelude=server-v1' not in result.stderr:
        raise ValueError(f'{binary} lacks the server-owned warm prelude; build a current cuteafd and pass --binary before using hardware')
    missing = [field for field in needs if field + '=' not in result.stderr]
    if missing:
        raise ValueError(f"{binary} predates smoke fields {', '.join(missing)}; build cuteafd from this checkout and pass --binary")


def launcher_forwards_v41_trace(repo):
    """Whether the checkout's V4.1 launch forwards SPECULATION_TRACE: run.sh's own
    path before #82, run-family.sh's shared trace helper since."""
    family = repo / 'scripts/launch/run-family.sh'
    if family.is_file() and 'launch_deepseek_v41' in family.read_text():
        return 'CUTEAFD_SPECULATION_TRACE' in family.read_text()
    return 'CUTEAFD_SPECULATION_TRACE' in (repo / 'run.sh').read_text()


def speculating(values):
    """Whether a launch resolves a speculator (traces are written only then)."""
    return values.get('SPECULATOR', 'auto') != 'off' and values.get('DSPARK', 'on') != 'off'


def planned_speculator(job, deployment):
    """Attest the identity, not just that deployment admission succeeded."""
    selected = deployment.get('speculator')
    if selected not in {'off', 'mtp', 'dspark', 'dflash2', 'dfly2'}:
        raise ValueError('launcher deployment preflight returned no resolved speculator identity')
    values = job['entry']['set']
    requested = values.get('SPECULATOR', 'auto')
    if requested != 'auto' and requested != selected:
        raise ValueError(f'planned speculator {selected} differs from requested {requested}')
    previous = job.get('planned_speculator')
    if previous is not None and previous != selected:
        raise ValueError(f'planned speculator changed since enqueue: {previous} -> {selected}')
    if selected == 'off' and values.get('DRAFT_LENGTH') and speculating(values):
        raise ValueError('DRAFT_LENGTH requested but the planned launch has no speculator')
    job['planned_speculator'] = selected


def served_speculator(settings):
    """Normalize the server's startup settings, not the launch config request."""
    options = {item['name']: item.get('value') for item in settings if 'name' in item}
    resolved = next((item.get('value') for item in settings
                     if item.get('name') == 'speculator' and item.get('source') == 'resolved'), None)
    if resolved is not None:
        return resolved
    if options.get('dspark') == 'true':
        return 'dspark'
    draft = options.get('draft')
    if draft and draft != 'false':
        return 'dspark' if 'dspark' in draft.lower() else 'dflash2'
    if str(options.get('mtp', '')).isdigit() and int(options['mtp']) > 0:
        return 'mtp'
    return 'off'


def attest_served_speculator(job, settings):
    selected = job.get('planned_speculator')
    if selected is None:
        return None
    if not settings:
        raise ValueError(f'no served startup settings for planned speculator {selected}')
    served = served_speculator(settings)
    if selected != served:
        raise ValueError(f'planned speculator {selected} differs from served {served}')
    return served


def preflight_at_grant(job, arm):
    """Smoke's prelaunch hook, run once it holds this card's locks: the absence of
    our containers is only meaningful then. Checked up front it would refuse a job
    queued behind a running task that still holds a container of the same name."""
    artifacts = preflight_job(job, arm)
    assert_absent(job)
    assert_port_free(job)
    if 'memory' in job['probes']:
        save(Path(job['entry']['set']['PROFILE_DIR']) / 'observer-context.json', {'pending': True})
    return artifacts


def preflight_deployment(job, arm):
    """Use the launcher's CPU deployment admission, not a second family policy."""
    repo = Path(arm.get('repo', REPO))
    entry = job['entry']
    cmd = [str(repo / 'run.sh'), '--config', entry['config'],
           *entry.get('run_args', []), '--dry-run', '--total-capacity']
    for key, value in sorted(entry.get('set', {}).items()):
        cmd += ['--set', f'{key}={value}']
    try:
        result = run(cmd, timeout=300)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise ValueError('launcher deployment preflight refused: ' + detail) from None
    except subprocess.TimeoutExpired:
        raise ValueError('launcher deployment preflight timed out after 300 seconds') from None
    decoder = json.JSONDecoder()
    for offset, char in enumerate(result.stdout):
        if char != '{':
            continue
        try:
            deployment, _ = decoder.raw_decode(result.stdout[offset:])
        except ValueError:
            continue
        if isinstance(deployment, dict) and isinstance(deployment.get('spark_roles'), list):
            planned_speculator(job, deployment)
            return deployment
    raise ValueError('launcher deployment preflight returned no typed Spark roles')


def preflight_job(job, arm):
    """Everything about one job that is checkable before its turn: the arm is
    sealed on every host it runs on, the slot holds the packages and audio entry
    the card needs, and the job's trace/profile directories are writable. Container
    absence is asserted at grant (preflight_at_grant), not here. Returns the
    artifact record."""
    values = job['entry']['set']
    if 'image-prefix-partial' in job['probes']:
        try:
            run([sys.executable, '-c', 'import numpy'], timeout=10)
        except subprocess.SubprocessError:
            raise ValueError('image-prefix-partial needs NumPy; run wip-cards with the gate venv Python') from None
        root = values.get('PROBE_DUMP_ROOT')
        if not root or not Path(root).is_dir() or not os.access(root, os.W_OK):
            raise ValueError('image-prefix-partial requires its writable PROBE_DUMP_ROOT')
    needs = job.get('needs') or {'coordinator': [], 'spark': [], 'audio': False}
    artifacts, problems = {}, []
    try:
        container_targets(job)
    except ValueError as error:
        problems.append(str(error))
    if problems:
        raise ValueError('; '.join(problems))
    deployment = preflight_deployment(job, arm)
    required_packages = deployment.get('artifact_requirements', {}).get('spark_packages')
    if required_packages is None:
        # Older sealed binaries register recipes but do not attest installed variants.
        tp = deployment.get('topology', {}).get('tp') or int(values.get('SPARK_COUNT', 0))
        required_packages = {package: exl3_spark_layouts(
            package.removeprefix('exl3/exl3-').rsplit('-k', 1)[0] + ':exl3-k' + package.rsplit('-k', 1)[1], tp)
            for package in needs['spark'] if package.startswith('exl3/') and tp}
    inventory = resolve_container_keys(job['entry'])['sparks']
    active_roles = {}
    for role in deployment['spark_roles']:
        rank = role.get('rank')
        if type(rank) is not int or not 0 <= rank < len(inventory):
            raise ValueError(f'planner Spark role rank has no configured host: {rank!r}')
        active_roles[inventory[rank]] = role
    job['spark_role_hosts'] = list(active_roles)
    for host in ('raptor', *active_roles):
        meta = json.loads(slot_check(arm, host))
        release = 'release' in arm
        keys = ('image', 'image_id', 'repo_digests') if release else ('slot', 'seal_sha256', 'artifact_manifest_sha256')
        artifacts[host] = {k: meta[k] for k in keys}
        contents = meta.get('contents')
        if host != 'raptor' and active_roles[host]['experts'] and required_packages and release:
            command = ['docker', 'run', '--rm', '--network', 'none', '-i', '--entrypoint', 'python3',
                       meta['image'], '-', json.dumps(required_packages),
                       f"{job['card']} {job['entry'].get('family', '<unknown>')} {host} image {meta['image_id']}"]
            try:
                run(['ssh', '-o', 'BatchMode=yes', host, shlex.join(command)],
                    input=(REPO / 'scripts/launch/preflight-spark-packages.py').read_text())
            except subprocess.SubprocessError as error:
                problems.append((getattr(error, 'stderr', None) or str(error)).strip())
        if host != 'raptor' and active_roles[host]['experts'] and required_packages and not release:
            present = set((contents or {}).get('spark_variants', []))
            for package, layouts in required_packages.items():
                for layout in layouts:
                    variant = f'{package}/{layout}'
                    if variant not in present:
                        problems.append(f"{job['card']} {job['entry'].get('family', '<unknown>')}: {host}: "
                                        f"missing Spark package {variant}; seal {meta['seal_sha256']} "
                                        f"slot {arm['slot']}")
        if release or contents is None:
            continue
        role = active_roles.get(host)
        wanted = needs['coordinator'] if host == 'raptor' else (needs['spark'] if role['experts'] else [])
        missing = sorted(set(wanted) - set(contents['packages']))
        if missing:
            problems.append(f"{host}: slot {arm['slot']} lacks {', '.join(missing)} (build with the card's scopes: wip-cards --build)")
        needs_audio = needs['audio'] if host == 'raptor' else bool(role.get('audio_encoder'))
        if needs_audio and contents.get('audio') is False:
            problems.append(f"{host}: slot {arm['slot']} was built without the audio tower (AUDIO={values.get('AUDIO', 'auto')}); "
                            'rebuild with CUTEAFD_WIP_AUDIO_AOT=ON or set AUDIO=off')
        if needs_audio and job['entry'].get('family') == 'muse_spark' and contents.get('pcm_audio') is not True:
            problems.append(f"{host}: slot {arm['slot']} lacks the Muse audio tower (cuteafd_pcm_required); "
                            'rebuild with CUTEAFD_WIP_AUDIO_AOT=ON')
        tp_role = deployment.get('artifact_requirements', {}).get('spark_tp_role')
        # The seal check always reports tp_roles (None: no TP manifest); a record
        # without the field is not a seal check result and carries no attestation.
        if tp_role and host != 'raptor' and role['experts'] and 'tp_roles' in contents:
            problems.extend(tp_role_problems(host, arm['slot'], tp_role, needs.get('nvfp4', False), contents))
    for key in ('SPECULATION_TRACE', 'PROFILE_DIR'):
        if values.get(key):
            directory = Path(values[key]) if key == 'PROFILE_DIR' else Path(values[key]).parent
            if not os.access(directory, os.W_OK):
                problems.append(f'{key} directory {directory} is not writable')
    if problems:
        raise ValueError('; '.join(problems))
    return artifacts


def tp_role_problems(host, slot, role, nvfp4, contents):
    """The launcher's fail-closed V4.1 role check (#98): every Spark rank's slot
    attests the role, with verified symbols and matching hashes."""
    attested = contents.get('tp_roles')
    rebuild = f'rebuild with CUTEAFD_WIP_SPARK_TP_ROLES={role} (wip-cards --build derives it from the card)'
    if attested is None:
        return [f'{host}: slot {slot} has no V41_EXPERT_TP_AOT.json; it cannot attest Spark TP role {role}; {rebuild}']
    if attested.get('error'):
        return [f"{host}: slot {slot} cannot attest Spark TP role {role}: {attested['error']}; {rebuild}"]
    if role not in attested.get('roles', []):
        return [f"{host}: slot {slot} attests Spark TP roles {';'.join(attested.get('roles')) or '<none>'}, not {role}; {rebuild}"]
    if not attested.get('verified'):
        return [f'{host}: slot {slot}: TP symbols are not verified for {role}; {rebuild}']
    if nvfp4 and role not in attested.get('nvfp4', []):
        return [f'{host}: slot {slot} lacks verified NVFP4 {role} exports; {rebuild} with the NVFP4 scope']
    return []


def git_head(repo):
    result = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=60)
    return result.stdout.strip() if result.returncode == 0 else None


def explicit_launchers(arms, arm_repos):
    """The --arm-repo ARM=PATH mappings, validated."""
    explicit = {}
    for item in arm_repos:
        name, _, path = item.partition('=')
        if name not in arms or not path:
            raise ValueError('invalid --arm-repo mapping: ' + item)
        if name in explicit:
            raise ValueError('duplicate --arm-repo mapping: ' + name)
        explicit[name] = Path(path).expanduser()
    return explicit


def submodule_problems(repo):
    """How a launcher checkout's submodules differ from what run.sh verifies: every
    submodule (recursively) initialized at the commit its superproject records, and
    each third_party/<name> at the revision third_party/<name>.lock.json pins."""
    try:
        status = run(['git', '-C', str(repo), 'submodule', 'status', '--recursive']).stdout
    except subprocess.CalledProcessError as error:
        return ['git submodule status failed: ' + (error.stderr or error.stdout or str(error)).strip()]
    except subprocess.TimeoutExpired:
        return ['git submodule status timed out']
    problems, commits = [], {}
    for line in status.splitlines():
        if not line.strip():
            continue
        flag, (commit, path, *_rest) = line[0], line[1:].split()
        if flag == '-':
            problems.append(f'{path} is not initialized')
        elif flag == '+':
            problems.append(f'{path} is checked out at {commit[:12]}, not the commit the superproject records')
        elif flag == 'U':
            problems.append(f'{path} has merge conflicts')
        else:
            commits[path] = commit
    for lock in sorted(Path(repo).glob('third_party/*.lock.json')):
        path = 'third_party/' + lock.name.removesuffix('.lock.json')
        try:
            pinned = json.loads(lock.read_text())['revision']
        except (OSError, ValueError, KeyError, TypeError):
            problems.append(f'{lock} has no readable revision')
            continue
        if path in commits and commits[path] != pinned:
            problems.append(f'{path} is at {commits[path][:12]}, not {pinned[:12]} pinned by {lock.name}')
    return problems


def check_launcher_submodules(arms, arm_repos, builds, global_repo):
    """Refuses, before any build, seed lease or hardware ticket, an arm whose launcher
    checkout run.sh would reject at launch (#2722): run.sh verifies the live arm
    worktree's SparkInfer source, not the sealed build copy. --build arms serve from
    the source checkout wip-source-checkout.sh creates and verifies against the locks.
    Refuses rather than initializing: the worktree belongs to its owner."""
    explicit = explicit_launchers(arms, arm_repos)
    problems = []
    for repo, names in itertools.groupby(sorted((str(explicit.get(name, global_repo)), name) for name in arms if name not in builds),
                                         key=lambda item: item[0]):
        found = submodule_problems(repo)
        if not found:
            continue
        script = Path(repo) / 'scripts/dev/submodule-init-local'
        fix = shlex.quote(str(script)) if script.is_file() else f'git -C {shlex.quote(repo)} submodule update --init --recursive'
        problems.append(f"arm {', '.join(name for _repo, name in names)}: launcher worktree {repo}: {'; '.join(found)}; "
                        f'fix with {fix} (wip-cards does not change an arm worktree)')
    if problems:
        raise ValueError('launcher submodules are not at their pinned commits, refused before taking any lock:\n  '
                         + '\n  '.join(problems))


def resolve_launchers(arms, arm_repos, builds, global_repo, state, task, dry):
    """Gives every arm the launcher checkout that serves it (recorded in the plan as
    arm['repo'] and arm['repo_commit']): a --build arm its own <builds>/<task>/source-<arm>
    (which must be at the built revision), any other arm its --arm-repo ARM=PATH, else the
    global --repo. Explicit and built launchers must be git checkouts holding run.sh."""
    explicit = explicit_launchers(arms, arm_repos)
    if set(explicit) & set(builds):
        raise ValueError('--arm-repo refuses --build arms (their source checkout is the launcher): ' + ', '.join(sorted(set(explicit) & set(builds))))
    for name, arm in arms.items():
        if name in builds:
            # The commit pinned by resolve_builds, never a ref resolved again here.
            repo = Path.home() / '.cache/cuteafd/builds' / task / ('source-' + name)
            wanted = builds[name]
        elif name in explicit:
            repo, wanted = explicit[name], None
        else:
            arm['repo'], arm['repo_commit'] = str(global_repo), git_head(global_repo)
            continue
        if not (dry and name in builds):
            if not (repo / 'run.sh').is_file():
                raise ValueError(f'{name}: launcher {repo} has no run.sh')
            head = git_head(repo)
            if (repo / '.git').exists() is False or head is None:
                raise ValueError(f'{name}: launcher {repo} is not a git checkout')
            if wanted and head != wanted:
                raise ValueError(f'{name}: source checkout {repo} is at {head[:12]}, not its built revision {wanted[:12]}')
        arm['repo'], arm['repo_commit'] = str(repo), (git_head(repo) if not (dry and name in builds) else wanted)
    return arms


def smoke_command(binary, repo, job):
    dest = Path(job['state'])
    return [binary, 'bench', 'smoke', '--matrix', str(dest / 'matrix.json'), '--repo', str(repo), '--state', str(dest / 'smoke'), '--out-root', str(dest / 'reports'), '--parallel', '1']


# How long a signalled smoke may take to tear down before it is killed: above its
# own 60 s bound, under hwlock.py's 80 s grace and systemd's 90 s TimeoutStopSec.
SMOKE_STOP_GRACE_S = 75.0
# Matches HookChild::spawn's default finish_timeout_s in rust/crates/cuteafd-bench/src/smoke.rs.
OBSERVER_FINISH_TIMEOUT_S = 5


def observer_teardown_budget(job):
    """Total observer teardown budget, derived from and strictly shorter than finish_timeout_s."""
    observer = (job.get('entry') or {}).get('observer') or {}
    finish_timeout = float(observer.get('finish_timeout_s') or OBSERVER_FINISH_TIMEOUT_S)
    margin = 1.5 if finish_timeout > 2.0 else finish_timeout * 0.3
    return max(0.1, min(finish_timeout * 0.7, finish_timeout - margin))
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
# The first stop signal received, and the detached smoke process groups it reaches.
STOP = dict(signal=None)
_LIVE = {}  # id(process) -> process
_LIVE_LOCK = threading.RLock()  # re-entrant: the handler runs on the main thread


def _signal_live(signum):
    with _LIVE_LOCK:
        live = list(_LIVE.values())
    for process in live:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signum)


def forward_stop(signum, _frame=None):
    """Signal handler: remember the first stop signal and pass it to every
    running smoke's process group (smoke ignores repeats)."""
    if STOP['signal'] is None:
        STOP['signal'] = signum
    _signal_live(signum)


def install_stop_handlers():
    for sig in STOP_SIGNALS:
        signal.signal(sig, forward_stop)


def stopped_code():
    return 128 + STOP['signal'] if STOP['signal'] is not None else None


def detached(cmd, log, exit_file, env):
    # Shell remains in the detached group and writes a numeric status even if the
    # CLI disappears. It records stop signals but waits for smoke's teardown;
    # even a child that exits zero after termination cannot publish success. The
    # parent's serving turn outlives this process and its container teardown.
    # setsid may fork when its caller is a process-group leader; --wait must
    # propagate the child status rather than report the successful fork.
    script = ('terminated=0; trap \'terminated=143\' TERM; trap \'terminated=130\' INT; '
              'trap \'terminated=129\' HUP; "$@"; rc=$?; '
              'if ((terminated)); then rc=$terminated; echo "terminated (exit $rc)" >&2; fi; '
              'printf "%s\\n" "$rc" > "$CARD_EXIT"; exit "$rc"')
    with Path(log).open('w') as output:
        process = subprocess.Popen(['setsid', '--wait', 'bash', '-c', script, 'wip-card', *cmd], stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, env={**env, 'CARD_EXIT': str(exit_file)}, start_new_session=False)
    with _LIVE_LOCK:
        _LIVE[id(process)] = process
    if STOP['signal'] is not None:  # arrived while it started
        _signal_live(STOP['signal'])
    return process


def wait_detached(process, grace=SMOKE_STOP_GRACE_S):
    """Waits for a detached smoke. After a stop signal it gets `grace` seconds
    to tear down, then its group is killed."""
    deadline = None
    try:
        while True:
            try:
                return process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            if STOP['signal'] is not None:
                deadline = deadline or time.monotonic() + grace
                if time.monotonic() >= deadline:
                    print(f'wip-cards: smoke did not stop within {grace:.0f} s; killing it', file=sys.stderr)
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(process.pid, signal.SIGKILL)
                    return process.wait()
    finally:
        with _LIVE_LOCK:
            _LIVE.pop(id(process), None)


def http(url, path, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url + path, data=data, headers={'Authorization': 'Bearer ' + os.environ['CUTEAFD_API_KEY'], 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")}') from None


def census_final_sample(job):
    """Close census evidence on the server's report clock, outside timed work."""
    sample = http(os.environ['CUTEAFD_SMOKE_URL'], '/v1/thermal')
    validate_census_clock(sample['now_s'])
    save(Path(job['state']) / 'census-final-thermal.json', sample)


def validate_census_clock(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid census server clock')
    return value


def merge_census_marker(job):
    """Translate server now_s through daemon clock anchors, never client time."""
    dest = Path(job['state'])
    sample = json.loads((dest / 'census-final-thermal.json').read_text())
    now = validate_census_clock(sample['now_s'])
    log = dest / 'smoke' / 'logs' / (job['entry']['name'] + '.coordinator.log')
    text = log.read_text()
    offsets, queries = [], []
    decoder = json.JSONDecoder()
    for raw in text.splitlines():
        line = ANSI.sub('', raw)
        if 'cuteafd::thermal:' not in line or 'current=' not in line:
            continue
        stamp = datetime.datetime.fromisoformat(line.split()[0].replace('Z', '+00:00')).timestamp()
        queried = validate_census_clock(decoder.raw_decode(line.split('current=', 1)[1])[0]['queried_s'])
        # Every anchor, including teardown queries logged after the final
        # sample, only fixes the server-to-wall clock offset; ordering does not.
        offsets.append(stamp - queried)
        queries.append(queried)
    if not offsets or max(offsets) - min(offsets) > .050:
        raise ValueError('missing or inconsistent daemon thermal clock anchors')
    if now < min(queries):
        raise ValueError('final census sample precedes every daemon thermal anchor')
    stamp = datetime.datetime.fromtimestamp(statistics.median(offsets) + now, datetime.timezone.utc)
    marker = (stamp.isoformat(timespec='microseconds').replace('+00:00', 'Z')
              + ' INFO cuteafd::bench_census: final server thermal sample'
              + f' server_now_s={now} source=census-final-thermal.json\n')
    if marker not in text:
        with log.open('a') as output:
            output.write('\n' + marker)


def red_square():
    def chunk(name, data):
        return struct.pack('>I', len(data)) + name + data + struct.pack('>I', zlib.crc32(name + data))
    raw = (b'\x00' + b'\xff\x00\x00' * 224) * 224
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 224, 224, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def image_prefix_probe(url, dest, dump_root=None):
    """Keep every attempt and its assertion, including a smoke hook's teardown retry."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for number in itertools.count(1):
        attempt = dest / f'image-prefix-attempt-{number}'
        try:
            attempt.mkdir()
            break
        except FileExistsError:
            continue
    try:
        result = image_prefix_attempt(url, attempt, dump_root)
    except Exception as error:
        save(attempt / 'verdict.json', dict(status='failed', error=str(error)))
        raise
    save(attempt / 'verdict.json', dict(status='pass', requests=result))
    return result


def image_prefix_tokens(url, model, prefix, image, question):
    counted = http(url, '/v1/responses/input_tokens', dict(model=model,
        input=[dict(role='user', content=[dict(type='input_text', text=prefix),
            dict(type='input_image', image_url=image), dict(type='input_text', text=question)])],
        reasoning=dict(effort='none')), timeout=120)
    if counted.get('exact_token_count') is not True:
        raise ValueError('image-prefix boundary probe requires exact token counts')
    return counted['input_tokens']


def image_prefix_attempt(url, dest, dump_root=None):
    """Diverge after a cached image, then restore the exact next-turn ancestor (#2340)."""
    model = http(url, '/v1/models')['data'][0]['id']
    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data))
    raw = (b'\x00' + b'\xff\x00\x00' * 160 + b'\x00\x00\xff' * 160) * 200
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 320, 200, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
    image = 'data:image/png;base64,' + base64.b64encode(png).decode()
    nonce = str(time.time_ns())
    results, quality = [], []
    question_a = 'Which colours are in the image? Brief.'
    question_b = 'Is there red in the image? yes/no.'
    for variant, entries, padding in [('even', 120, ''), ('odd', 120, ' x'), ('long', 4096, '')]:
        prefix = f'Run {nonce} {variant}. Reference notes. ' + ' '.join(
            f'Entry {i} is code {i * 7}.' for i in range(entries)) + padding
        # Count the same image/question template, without generation, to exercise actual
        # odd/even prompt frontiers rather than guessing from a character's tokenization.
        desired = 1 if variant == 'odd' else 0
        for pad in range(16):
            candidate = prefix + ' x' * pad
            count = image_prefix_tokens(url, model, candidate, image, question_b)
            if count % 2 == desired:
                prefix, prompt_tokens = candidate, count
                break
        else:
            raise ValueError(f'{variant}: could not select an exact prompt frontier parity')
        save(dest / f'image-prefix-{variant}-boundary.json',
             dict(input_tokens=prompt_tokens, parity=desired, padding=pad))
        requests = [
            ('text-a', False, 'What is the code of entry 3? Number only.', r'\b21\b'),
            ('text-b', False, 'What is the code of entry 5? Number only.', r'\b35\b'),
            ('image-a', True, question_a, r'(?is)(?=.*\bred\b)(?=.*\bblue\b)'),
            ('image-b', True, question_b, r'\byes\b'),
        ]
        previous_content = previous_answer = None
        exact_bodies = []
        for label, with_image, question, expected in requests + [('next-turn', True, question_b, r'\byes\b')]:
            content = [dict(type='text', text=prefix)]
            if with_image:
                content.append(dict(type='image_url', image_url=dict(url=image)))
            content.append(dict(type='text', text=question))
            messages = [dict(role='user', content=content)]
            if label == 'next-turn':
                messages = [dict(role='user', content=previous_content),
                            dict(role='assistant', content=previous_answer),
                            dict(role='user', content=question)]
            receipt = dest / f'image-prefix-{variant}-{label}.json'
            try:
                response = http(url, '/v1/chat/completions', dict(model=model,
                    messages=messages, max_tokens=32,
                    reasoning_effort='none', temperature=0, stream=False), timeout=600)
            except Exception as error:
                save(receipt, dict(error=str(error)))
                raise
            save(receipt, response)
            answer = response['choices'][0]['message'].get('content') or ''
            if not re.search(expected, answer, re.I):
                raise ValueError(f'{variant}/{label}: incorrect answer {answer!r}')
            usage = response['usage']
            cached = usage.get('prompt_cache_hit_tokens', 0)
            source = usage.get('prefix_cache_source')
            if label == 'image-b' and usage['prompt_tokens'] != prompt_tokens:
                raise ValueError(f'{variant}: count and serving frontiers differ: {usage} != {prompt_tokens}')
            # Exact-only families deliberately miss on divergence, while page/replay
            # families may reuse the common prefix. Both are valid if provenance agrees.
            if label == 'text-a' and (cached != 0 or source != 'miss'):
                raise ValueError(f'{variant}/{label}: unique prefix was not a miss: {usage}')
            if label != 'text-a' and (source not in {'miss', 'device', 'host'}
                    or (cached > 0) != (source != 'miss')):
                raise ValueError(f'{variant}/{label}: inconsistent prefix provenance: {usage}')
            if dump_root and label == 'image-b' and not (0 < cached < prompt_tokens):
                raise ValueError(f'{variant}/{label}: request 4 did not partially hit: {usage}')
            if label == 'next-turn' and not (cached >= prompt_tokens and source in {'device', 'host'}):
                raise ValueError(f'{variant}/{label}: exact image ancestor did not hit: {usage}')
            if label == 'image-b':
                previous_content, previous_answer = content, answer
            results.append(dict(variant=variant, request=label, answer=answer, cached=cached, source=source))
            if label in {'image-b', 'next-turn'}:
                exact_bodies.append((label, dict(model=model, messages=messages, max_tokens=32,
                    reasoning_effort='none', temperature=0, stream=False)))
        for label, body in exact_bodies:
            # A turn ancestor inherits decode-produced KV, and image-b may have
            # approximate SWA state. Exact restores must match the capture, not cold.
            probes = []
            for name, cold in [('reference', False), ('cold', True), ('warm', False), ('warm-repeat', False)]:
                reply = http(url, '/v1/bench/probe', dict(body=body,
                    spec=dict(cold=cold, no_speculation=True, pair_exempt=True,
                              record_state=True, record_rows=1, top_k=5)), timeout=600)
                save(dest / f'image-prefix-{variant}-{label}-{name}.json', reply)
                record = reply.get('probe') or {}
                if record.get('cold') is not cold or record.get('no_speculation') is not True:
                    raise ValueError(f'{variant}/{label}: engine did not honor exact probe switches')
                if not record.get('generated'):
                    raise ValueError(f'{variant}/{label}: exact probe omitted generated token IDs')
                probes.append(record)
            reference, cold, warm, repeat = probes
            image_next_turn_fidelity(reference, warm, repeat)
            if warm.get('cached_tokens', 0) < len(cold.get('prompt_ids') or []):
                raise ValueError(f'{variant}/{label}: exact image prompt was not fully restored')
        if dump_root:
            # This namespace has only the divergent seed, never an exact image-b hit.
            score_prefix = prefix.replace(f'Run {nonce}', f'Quality {nonce}', 1)
            quality.append(image_partial_fidelity(url, dest, Path(dump_root), variant, model,
                score_prefix, image, question_a, question_b))
    if quality:
        rows = sum(q['rows'] for q in quality)
        kl = sum(q['mean_kl'] * q['rows'] for q in quality) / rows
        top1 = sum(q['top1_agreement'] * q['rows'] for q in quality) / rows
        # Same strict quality bars as the fidelity panel, pooled over the fixed fixture.
        passed = round(kl, 6) <= 0.03 and round(top1, 6) >= 0.95
        save(dest / 'partial-fidelity.json', dict(status='pass' if passed else 'failed',
             rows=rows, mean_kl=kl, top1_agreement=top1, variants=quality))
        if not passed:
            raise ValueError(f'image partial fidelity failed: KL {kl:.6f} (max 0.03), top-1 {top1:.2%} (min 95%) over {rows} rows')
    return results


IMAGE_CONTINUATION = ('Yes, there is red on the left half of the image, and blue on the right half. '
                      'The two colors fill equal rectangular regions and meet at a straight vertical boundary.')


def image_partial_fidelity(url, dest, dump_root, variant, model, prefix, image, seed_question, question):
    def body(question):
        content = [dict(type='text', text=prefix), dict(type='image_url', image_url=dict(url=image)),
                   dict(type='text', text=question)]
        return dict(model=model, messages=[dict(role='user', content=content)],
                    reasoning_effort='none', temperature=0, max_tokens=32, stream=False)

    start = image_prefix_tokens(url, model, prefix, image, question)
    positions = list(range(start, start + 24))
    score_body = body(question)
    score_body['messages'].append(dict(role='assistant', content=IMAGE_CONTINUATION))
    stem = f'{dest.name}-{variant}'
    paths = {arm: dump_root / f'{stem}-{arm}' for arm in ('cold', 'partial')}
    records = {}
    for arm, path in paths.items():
        cold = arm == 'cold'
        if not cold:
            seed = http(url, '/v1/chat/completions', body(seed_question), timeout=600)
            save(dest / f'{variant}-partial-seed.json', seed)
            usage = seed['usage']
            if usage.get('prompt_cache_hit_tokens') != 0 or usage.get('prefix_cache_source') != 'miss':
                raise ValueError(f'{variant}: quality seed did not prefill cold: {usage}')
        reply = http(url, '/v1/bench/probe', dict(body=score_body, spec=dict(cold=cold,
            no_speculation=True, pair_exempt=True, score_from=start, score_reuse_prefix=not cold, score_path='decode',
            score_positions=positions, dump_rows=str(path), top_k=1)), timeout=600)
        save(dest / f'{variant}-score-{arm}.json', reply)
        record = records[arm] = reply['probe']
        rows = record['rows']
        if (record['cold'] is not cold or record['no_speculation'] is not True
                or record['score_path'] != 'decode' or record['error'] or record['generated']
                or record['scored'] != len(positions) or [r['position'] for r in rows] != positions
                or len(record['prompt_ids']) < start + len(positions) or not all(r['finite'] for r in rows)):
            raise ValueError(f'{variant}/{arm}: incomplete fixed-continuation scoring')
        cached = record['cached_tokens']
        if not (cached == 0 if cold else 0 < cached < start):
            raise ValueError(f'{variant}/{arm}: unexpected scoring cache hit of {cached} tokens')
        restore = (record.get('provenance') or {}).get('prefix_restore')
        if not cold and restore is not None and restore.get('partial') is not True:
            raise ValueError(f'{variant}/{arm}: scoring used an exact ancestor, not partial replay')
        if not cold and record.get('engine') == 'deepseek_v41':
            plan = (restore or {}).get('plan') or {}
            lag = plan.get('lag') or {}
            if (plan.get('fidelity') != 'ApproximateReplay' or plan.get('target_end') != start
                    or lag.get('replay_start') != cached or not cached < lag.get('source_end', 0) < start):
                raise ValueError(f'{variant}/{arm}: scoring did not attest bounded replay: {restore}')
    if records['cold']['prompt_ids'] != records['partial']['prompt_ids']:
        raise ValueError(f'{variant}: scored prompt IDs differ across cold and partial arms')
    receipt = dest / f'{variant}-partial-kl.json'
    run([sys.executable, str(REPO / 'scripts/bench/compare-logits.py'), str(paths['cold']),
         str(paths['partial']), '--json', str(receipt)])
    metrics = json.loads(receipt.read_text())
    if (metrics['rows'] != len(positions) or any(metrics[k] for k in ('only_a', 'only_b', 'nonfinite_rows'))
            or not all(math.isfinite(metrics[k]) for k in ('mean_kl', 'top1_agreement'))):
        raise ValueError(f'{variant}: full-vocabulary dumps do not cover the fixed continuation')
    return dict(variant=variant, cached=records['partial']['cached_tokens'], **{
        k: metrics[k] for k in ('rows', 'mean_kl', 'max_kl', 'top1_agreement')})


def image_next_turn_fidelity(reference, warm, repeat):
    """Restore identity is relative to the capture, not a differently rounded cold prefill."""
    def state(record, kind):
        rows = [s for s in record.get('states', []) if s['kind'] == kind]
        if len(rows) != 1:
            raise ValueError(f'next-turn probe requires one {kind} state')
        return {k: rows[0][k] for k in ('end', 'sha256', 'bytes')}
    captured = state(reference, 'prompt')
    for record in (warm, repeat):
        if state(record, 'restore') != captured or state(record, 'prompt') != captured:
            raise ValueError('next-turn restored state differs from captured state')
        if record.get('prompt_ids') != reference.get('prompt_ids'):
            raise ValueError('next-turn probe prompt IDs differ')
    if any(warm.get(k) != repeat.get(k) for k in ('generated', 'rows', 'states')):
        raise ValueError('next-turn warm repeat is not byte-identical')
    if warm.get('generated') != reference.get('generated') or warm.get('rows') != reference.get('rows'):
        raise ValueError('next-turn warm output differs from fixed reference')


def ready_log(job):
    name = 'cuteafd-coordinator-' + job['entry']['set']['INSTANCE']
    result = run(['docker', 'logs', name])
    text = result.stdout + result.stderr
    (Path(job['state']) / 'ready.log').write_text(text)
    return ANSI.sub('', text)


def admitted_pool(text):
    # key=value forms, and GLM's admission line ("...; pool 2097152 tokens").
    values = re.findall(r'\b(?:admitted_pool_tokens|admitted_tokens|pool_tokens)\s*=\s*(\d+)|\bpool (\d+) tokens\b',
                        ANSI.sub('', text))
    values = [a or b for a, b in values]
    if not values:
        raise ValueError('ready.log has no admitted pool value')
    return int(values[-1])


def precheck(job):
    dest = Path(job['state'])
    result = {}
    try:
        if job.get('planned_speculator') is not None:
            status = http(os.environ['CUTEAFD_SMOKE_URL'], '/v1/bench/status', timeout=10)
            result['speculator'] = attest_served_speculator(job, status.get('settings'))
            result['drafter'] = status.get('drafter')
        if 'memory' in job['probes'] and job['entry'].get('sparks'):
            spark_memory_boundary(job, 'ready')
        text = ready_log(job) if job['expected_pool'] is not None or set(job['probes']) & IMAGE_PROBES else ''
        if job['expected_pool'] is not None:
            result['pool'] = admitted_pool(text)
            if result['pool'] != job['expected_pool']:
                raise ValueError(f"admitted pool {result['pool']} != expected {job['expected_pool']}")
        if 'image' in job['probes']:
            url = os.environ['CUTEAFD_SMOKE_URL']
            model = http(url, '/v1/models')['data'][0]['id']
            response = http(url, '/v1/chat/completions', dict(model=model, messages=[dict(role='user', content=[dict(type='text', text='What color is this square? Answer with just the color.'), dict(type='image_url', image_url=dict(url=red_square()))])], max_tokens=512, temperature=0, stream=False))
            save(dest / 'image-response.json', response)
            answer = response['choices'][0]['message'].get('content') or ''
            result['image'] = {'model': model, 'answer': answer}
            if not re.search(r'\bred\b', answer, re.I):
                raise ValueError('image answer does not identify red')
        if set(job['probes']) & IMAGE_PREFIX_PROBES:
            dump_root = job['entry']['set']['PROBE_DUMP_ROOT'] if 'image-prefix-partial' in job['probes'] else None
            result['image-prefix'] = [image_prefix_probe(os.environ['CUTEAFD_SMOKE_URL'], dest, dump_root)
                                      for _ in range(1 if dump_root else 3)]
        result['status'] = 'pass'
    except Exception as error:
        result.update(status='failed', error=str(error))
        save(dest / 'precheck.json', result)
        raise
    save(dest / 'precheck.json', result)


# Executed inside the architecture-matching image, with host PID namespace so
# NVML sees the sampler's own PID. Never use nvidia-smi totals for GB10 memory.
CUDA_SAMPLER = r'''
import ctypes as c, ctypes.util, json, os, time
cuda=c.CDLL(ctypes.util.find_library('cudart') or 'libcudart.so.13')
nv=c.CDLL('libnvidia-ml.so.1')
class Process(c.Structure):
    _fields_=[('pid',c.c_uint),('usedGpuMemory',c.c_ulonglong),('gpuInstanceId',c.c_uint),('computeInstanceId',c.c_uint)]
def check(rc):
    if rc: raise RuntimeError('CUDA/NVML status '+str(rc))
check(nv.nvmlInit_v2())
devices=[int(x) for x in os.environ['CARD_GPUS'].split(',')]
# Both CUDA and NVML numeric indices are filtered by CDI inside this container.
os.environ['CUDA_VISIBLE_DEVICES']=os.environ['CARD_GPU_UUIDS']
for i in range(len(devices)):
    check(cuda.cudaSetDevice(i)); check(cuda.cudaFree(c.c_void_p()))
while True:
    rows=[]
    for i,physical in enumerate(devices):
        check(cuda.cudaSetDevice(i)); free=c.c_size_t(); total=c.c_size_t()
        check(cuda.cudaMemGetInfo(c.byref(free),c.byref(total)))
        pci=c.create_string_buffer(32); check(cuda.cudaDeviceGetPCIBusId(pci,32,i))
        handle=c.c_void_p(); check(nv.nvmlDeviceGetHandleByPciBusId_v2(pci,c.byref(handle)))
        count=c.c_uint(128); procs=(Process*128)()
        check(nv.nvmlDeviceGetComputeRunningProcesses_v3(handle,c.byref(count),procs))
        own=[p.usedGpuMemory for p in procs[:count.value] if p.pid==os.getpid()]
        if len(own)!=1 or own[0]==2**64-1: raise RuntimeError('sampler NVML context unavailable')
        raw=total.value-free.value
        rows.append(dict(gpu=physical,raw_bytes=raw,sampler_bytes=own[0],corrected_bytes=max(0,raw-own[0])))
    print(json.dumps(dict(time=time.time(),devices=rows)),flush=True)
    time.sleep(.05)
'''


# A persistent SSH stream avoids a connection per sample and creates no CUDA
# context. EOF or a lost observer heartbeat ends the remote process too.
SPARK_MEMORY_SAMPLER = r'''
import json, os, select, sys, time
from pathlib import Path
rank, host, lifetime = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
fields = {'MemAvailable', 'MemFree', 'Cached', 'SReclaimable', 'AnonPages',
          'PageTables', 'SecPageTables', 'Slab', 'VmallocUsed'}
started = heartbeat = time.monotonic()
phase = 'loading'
pending = ''
once = len(sys.argv) > 4 and sys.argv[4] == 'once'
while time.monotonic() - started < lifetime:
    began = time.monotonic()
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if readable and not once:
        control = os.read(sys.stdin.fileno(), 65536)
        if not control:
            break
        pending += control.decode()
        while '\n' in pending:
            line, pending = pending.split('\n', 1)
            phase = json.loads(line)['phase']
        heartbeat = began
    if began - heartbeat > 5:
        break
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in fields:
            parts = value.split()
            if parts[1:] != ['kB']:
                raise RuntimeError(key + ': unexpected meminfo unit: ' + line)
            mem[key] = int(parts[0]) * 1024
    missing = fields - mem.keys()
    # SecPageTables is not present on older kernels; retain that fact as null.
    if missing - {'SecPageTables'}:
        raise RuntimeError('missing meminfo fields: ' + str(sorted(missing)))
    mem.setdefault('SecPageTables', None)
    print(json.dumps(dict(time=time.time(), rank=rank, host=host, phase=phase, mem=mem)), flush=True)
    if once:
        break
    time.sleep(max(0, .05 - (time.monotonic() - began)))
'''


def spark_memory_command(rank, host, lifetime):
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
            '-o', 'ServerAliveInterval=2', '-o', 'ServerAliveCountMax=2', host,
            shlex.join(['python3', '-I', '-u', '-c', SPARK_MEMORY_SAMPLER,
                        str(rank), host, str(lifetime)])]


def spark_memory_read(child, output, guard_bytes, job, rank, host, errors):
    try:
        for line in child.stdout:
            row = json.loads(line)
            if row['rank'] != rank or row['host'] != host:
                raise ValueError('Spark sampler identity mismatch')
            if guard_bytes and row['mem']['MemAvailable'] < guard_bytes:
                row['guard'] = 'MemAvailable below 512 MiB'
                output.write(json.dumps(row) + '\n')
                output.flush()
                stop_memory_worker(job, host)
                raise RuntimeError(f'{host}: memory guard stopped this card worker')
            output.write(json.dumps(row) + '\n')
            output.flush()
    except Exception as error:
        errors.append(str(error))


def stop_memory_worker(job, host):
    values = job['entry']['set']
    slot = next(iter(job['entry']['run_args'][1:]), None)
    if not slot or '--wip' not in job['entry']['run_args']:
        raise ValueError('memory guard requires a WIP card with verifiable worker mounts')
    layout = '/.cache/cuteafd/wip-run/' + (values.get('WIP_INSTANCE', '') + '/' if values.get('WIP_INSTANCE') else '')
    layout += slot + '/serve-' + values['INSTANCE'] + '/bin'
    name = 'cuteafd-spark-expert-' + host + '-' + values['EXPERT_PORT']
    program = '''import json,subprocess,sys
name,layout=sys.argv[1:]
p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=2)
if p.returncode: raise RuntimeError('card worker unavailable: '+name)
info=json.loads(p.stdout)[0]
if not any(m.get('Destination')=='/opt/cuteafd/bin' and m.get('Source','').endswith(layout) for m in info.get('Mounts',[])):
    raise RuntimeError('refusing foreign card worker '+name)
subprocess.run(['docker','stop','-t','1',info['Id']],check=True,timeout=3)
'''
    run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=2', host,
         shlex.join(['python3', '-I', '-c', program, name, layout])], timeout=6)


def spark_memory_phase(url, diagnostics, timeout=.5):
    def telemetry(endpoint):
        began = time.monotonic()
        record = dict(endpoint=endpoint, timeout_s=timeout, exception_type=None)
        try:
            result = http(url, endpoint, timeout=timeout)
            record['status'] = 'ok'
            return result
        except Exception as error:
            record.update(status='unavailable', exception_type=type(error).__name__, error=str(error))
            raise
        finally:
            record['elapsed_s'] = time.monotonic() - began
            diagnostics.append(record)

    try:
        snapshot = telemetry('/v1/console/snapshot')
        if snapshot.get('starting'):
            return 'loading'
        gauges = snapshot.get('g', {})
        if gauges.get('prefilling_n', 0):
            return 'prefill'
        if gauges.get('active', 0):
            return 'decode'
        active = telemetry('/v1/bench/status').get('active')
        return 'panel:' + str(active['panel']) if active else 'ready-idle'
    except (OSError, HTTPException, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as error:
        diagnostics.append(dict(status='unavailable', source='phase-classification',
                                exception_type=type(error).__name__, error=str(error)))
        return 'unavailable'


def spark_memory_boundary(job, boundary):
    """Capture real host state at readiness/sweep completion, even for subsecond sweeps."""
    diagnostics = []
    # Boundary hooks can tolerate busy-server latency. Continuous observation
    # keeps its short timeout so telemetry cannot starve Spark heartbeats.
    phase = spark_memory_phase(os.environ['CUTEAFD_SMOKE_URL'], diagnostics, timeout=3.)
    # Both hooks run after smoke has confirmed health; an older starting
    # snapshot cannot make this known-ready boundary a loading sample.
    if phase == 'loading':
        phase = 'ready-idle'
    evidence = dict(phase=phase, boundary=boundary, ready=True, telemetry=diagnostics)
    save(Path(job['state']) / f'memory-phase-{boundary}.json', evidence)
    hosts = job['entry'].get('sparks', [])
    def sample(item):
        rank, host = item
        command = spark_memory_command(rank, host, 5)
        command[-1] += ' once'
        result = run(command, timeout=8)
        row = json.loads(result.stdout)
        if row['rank'] != rank or row['host'] != host:
            raise ValueError('Spark boundary sampler identity mismatch')
        row.update(evidence)
        path = Path(job['state']) / f'memory-spark-rank{rank}-{boundary}.jsonl'
        with path.open('w') as output:
            output.write(json.dumps(row) + '\n')
        if job.get('memory_guard_bytes', 0) and row['mem']['MemAvailable'] < job['memory_guard_bytes']:
            stop_memory_worker(job, host)
            raise RuntimeError(f'{host}: memory guard stopped this card worker at {boundary}')
    if hosts:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            list(pool.map(sample, enumerate(hosts)))


def spark_memory_report(dest):
    ranks = {}
    for path in sorted(dest.glob('memory-spark-rank*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rank = ranks.setdefault(str(row['rank']), dict(host=row['host'], samples=0, phases={}))
            available = row['mem']['MemAvailable']
            rank['samples'] += 1
            rank['min_mem_available_bytes'] = min(rank.get('min_mem_available_bytes', available), available)
            phase = rank['phases'].setdefault(row['phase'], dict(samples=0, min_mem_available_bytes=available, peak_bytes={}))
            phase['samples'] += 1
            phase['min_mem_available_bytes'] = min(phase['min_mem_available_bytes'], available)
            for key, value in row['mem'].items():
                if value is not None:
                    phase['peak_bytes'][key] = max(phase['peak_bytes'].get(key, 0), value)
            if row.get('guard'):
                rank['guard'] = row['guard']
    return ranks


def console_frames(url, stop, output):
    from urllib.parse import urlparse
    parsed = urlparse(url)
    def exact(sock, size):
        data = b''
        while len(data) < size:
            part = sock.recv(size - len(data))
            if not part:
                raise EOFError
            data += part
        return data
    while not stop.exists():
        try:
            with socket.create_connection((parsed.hostname, parsed.port), timeout=1) as sock:
                nonce = base64.b64encode(os.urandom(16)).decode()
                request = ('GET /v1/console HTTP/1.1\r\nHost: ' + parsed.netloc + '\r\nAuthorization: Bearer ' + os.environ['CUTEAFD_API_KEY'] + '\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: ' + nonce + '\r\nSec-WebSocket-Version: 13\r\n\r\n')
                sock.sendall(request.encode())
                header = b''
                while not header.endswith(b'\r\n\r\n'):
                    header += exact(sock, 1)
                    if len(header) > 16384:
                        raise ValueError('oversized websocket header')
                if b' 101 ' not in header:
                    raise ValueError('console websocket not ready')
                while not stop.exists():
                    a, b = exact(sock, 2)
                    size = b & 127
                    if size == 126:
                        size = struct.unpack('>H', exact(sock, 2))[0]
                    elif size == 127:
                        size = struct.unpack('>Q', exact(sock, 8))[0]
                    if size > 16 * 1024 * 1024:
                        raise ValueError('oversized console frame')
                    mask = exact(sock, 4) if b & 128 else None
                    body = exact(sock, size)
                    if mask:
                        body = bytes(x ^ mask[i % 4] for i, x in enumerate(body))
                    opcode = a & 15
                    if opcode == 8:
                        break
                    if opcode == 9 and size <= 125:
                        mask = os.urandom(4)
                        sock.sendall(bytes([0x8a, 0x80 | size]) + mask + bytes(x ^ mask[i % 4] for i, x in enumerate(body)))
                    elif opcode == 1:
                        output.write(json.dumps(json.loads(body)) + '\n')
                        output.flush()
        except (OSError, EOFError, ValueError):
            time.sleep(.25)


def sampler_cleanup(job):
    name = 'cuteafd-card-memory-' + job['entry']['set']['INSTANCE']
    owner = hashlib.sha256(job['state'].encode()).hexdigest()
    program = '''import json,subprocess,sys
name,owner=sys.argv[1:]
ids=subprocess.check_output(['docker','ps','-aq','--filter','name=^'+name+'$'],text=True).split()
for ident in ids:
    info=json.loads(subprocess.check_output(['docker','inspect',ident],text=True))[0]
    if info['Config'].get('Labels',{}).get('cuteafd.card.owner')!=owner:
        raise RuntimeError('refusing foreign memory sampler '+name)
    subprocess.run(['docker','rm','-f',ident],check=True)
'''
    return [sys.executable, '-c', program, name, owner]


def publish_observer_context(job):
    """Publish a complete measured sample indexed by container CUDA ordinal."""
    dest = Path(job['state'])
    with (dest / 'memory.jsonl').open() as samples:
        line = samples.readline()
    if not line.endswith('\n'):
        return False
    sample = json.loads(line)
    physical = job['entry']['gpus']
    measured = {row['gpu']: row['sampler_bytes'] for row in sample['devices']}
    if set(measured) != set(physical) or any(type(n) is not int or n <= 0 for n in measured.values()):
        raise ValueError('invalid measured observer context bytes')
    path = Path(job['entry']['set']['PROFILE_DIR']) / 'observer-context.json'
    temporary = path.with_suffix('.tmp')
    save(temporary, {str(i): measured[gpu] for i, gpu in enumerate(physical)})
    temporary.replace(path)
    return True


def observer_gpu_uuids(devices):
    """Use the launcher's host-side identity resolver before CDI filtering."""
    script = 'source "$1"\nrelease_resolve_coordinator_gpus "$2" "$3"\nprintf "%s\\n" "${RELEASE_COORDINATOR_GPU_UUIDS[@]}"'
    resolved = run(['bash', '-c', script, 'observer-gpus',
                    str(REPO / 'scripts/lib/release-common.sh'),
                    ','.join(map(str, devices)), str(len(devices))], timeout=30)
    uuids = resolved.stdout.splitlines()
    if len(uuids) != len(devices) or any(not re.fullmatch(r'GPU-[A-Za-z0-9-]+', uuid) for uuid in uuids):
        raise ValueError('launcher returned invalid observer GPU UUIDs')
    return ','.join(uuids)


def observe(job):
    dest = Path(job['state'])
    stop = Path(os.environ['CUTEAFD_SMOKE_STOP'])
    sampler = None
    spark_samplers, spark_errors = [], []
    memory_phase, phase_sample_at = 'loading', 0.
    name = 'cuteafd-card-memory-' + job['entry']['set']['INSTANCE']
    memory = (dest / 'memory.jsonl').open('w')
    console = (dest / 'console.jsonl').open('w')
    frames = (dest / 'console-frames.jsonl').open('w')
    policy = (dest / 'policy-stats.jsonl').open('w')
    policy_sample_at = 0.
    reader = None
    observer_published = False
    if 'console' in job['probes']:
        reader = threading.Thread(target=console_frames, args=(os.environ['CUTEAFD_SMOKE_URL'], stop, frames), daemon=True)
        reader.start()
    try:
        if 'memory' in job['probes']:
            e = job['entry']
            gpu_list = ','.join(map(str, e['gpus']))
            image = e['set'].get('COORDINATOR_DOCKER_DEV', 'cuteafd-coordinator-dev')
            owner = hashlib.sha256(job['state'].encode()).hexdigest()
            cmd = ['docker', 'run', '--rm', '--init', '-i', '--name', name, '--label', 'cuteafd.card.owner=' + owner, '--pid=host', '--network=none', *cdi_gpu_args(gpu_list.split(',')), '-e', 'CARD_GPUS=' + gpu_list, '-e', 'CARD_GPU_UUIDS=' + observer_gpu_uuids(e['gpus']), '--entrypoint', 'python3', image, '-u', '-c', CUDA_SAMPLER]
            sampler = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=memory, stderr=subprocess.PIPE, text=True)
            lifetime = min(float(e['observer']['timeout_s']), 7200)
            for rank, host in enumerate(e.get('sparks', [])):
                output = (dest / f'memory-spark-rank{rank}.jsonl').open('w')
                error = (dest / f'memory-spark-rank{rank}.stderr').open('w')
                try:
                    child = subprocess.Popen(spark_memory_command(rank, host, lifetime), stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=error, text=True)
                except Exception:
                    output.close()
                    error.close()
                    raise
                reader_thread = threading.Thread(target=spark_memory_read,
                    args=(child, output, job.get('memory_guard_bytes', 0), job, rank, host, spark_errors), daemon=True)
                spark_samplers.append((child, reader_thread, output, error))
                reader_thread.start()
        trace_path = Path(job['entry']['set']['SPECULATION_TRACE']) if job.get('trace') else None
        trace_due = None
        while not stop.exists():
            if sampler and sampler.poll() is not None:
                raise RuntimeError('memory sampler exited: ' + sampler.stderr.read())
            if sampler and not observer_published:
                observer_published = publish_observer_context(job)
            if spark_errors:
                raise RuntimeError('Spark memory sampler: ' + '; '.join(spark_errors))
            if spark_samplers:
                if time.monotonic() >= phase_sample_at:
                    diagnostics = []
                    memory_phase = spark_memory_phase(os.environ['CUTEAFD_SMOKE_URL'], diagnostics)
                    with (dest / 'memory-phase.jsonl').open('a') as output:
                        output.write(json.dumps(dict(time=time.time(), phase=memory_phase,
                                                     telemetry=diagnostics)) + '\n')
                    phase_sample_at = time.monotonic() + 1
                for child, _, _, _ in spark_samplers:
                    if child.poll() is not None:
                        raise RuntimeError('Spark memory sampler exited: ' + str(child.returncode))
                    child.stdin.write(json.dumps(dict(phase=memory_phase)) + '\n')
                    child.stdin.flush()
            if trace_path and not (dest / 'trace-check.json').exists():
                # After the baseline's warm-up (the first part of the run), the
                # trace must have records. Diagnostic only: the row decides.
                try:
                    active = (http(os.environ['CUTEAFD_SMOKE_URL'], '/v1/bench/status', timeout=.5).get('active') or {})
                    # Past the first warm-up: 5% into the run, or (a run that skips
                    # the baseline starts its first panel at once) 60 s into it.
                    if active and (active.get('fraction', 0) >= .05 or active.get('elapsed_s', 0) >= 60):
                        trace_due = trace_due or time.monotonic() + 5
                except (OSError, RuntimeError, ValueError):
                    pass
                if trace_due and time.monotonic() >= trace_due:
                    size = trace_path.stat().st_size if trace_path.is_file() else 0
                    save(dest / 'trace-check.json', {'after': 'warm-up', 'bytes': size, 'path': str(trace_path),
                                                     'ok': bool(size) or not speculating(job['entry']['set'])})
            if 'console' in job['probes']:
                try:
                    snapshot = http(os.environ['CUTEAFD_SMOKE_URL'], '/v1/console/snapshot', timeout=.5)
                    console.write(json.dumps({'time': time.time(), 'snapshot': snapshot}) + '\n')
                    console.flush()
                    if time.monotonic() >= policy_sample_at:
                        stats = http(os.environ['CUTEAFD_SMOKE_URL'], '/v1/stats', timeout=.5)
                        policy.write(json.dumps({'time': time.time(), 'stats': stats}) + '\n')
                        policy.flush()
                        policy_sample_at = time.monotonic() + 1.
                except (OSError, RuntimeError):
                    pass
            if not stop.exists():
                time.sleep(.25)
    finally:
        teardown_start = time.monotonic()
        finish_timeout_s = float(((job.get('entry') or {}).get('observer') or {}).get('finish_timeout_s') or OBSERVER_FINISH_TIMEOUT_S)
        budget_s = observer_teardown_budget(job)
        deadline = teardown_start + budget_s
        phases = []
        stop.touch()
        for child, _, _, _ in spark_samplers:
            with contextlib.suppress(OSError):
                child.stdin.close()
        if sampler:
            phase_start = time.monotonic()
            cleanup_timeout = min(budget_s * 0.6, max(0.0, deadline - phase_start))
            try:
                run(sampler_cleanup(job), timeout=cleanup_timeout)
                phases.append(dict(phase='sampler_cleanup', status='ok', elapsed_s=round(time.monotonic() - phase_start, 4)))
            except subprocess.TimeoutExpired:
                phases.append(dict(phase='sampler_cleanup', status='timeout', elapsed_s=round(time.monotonic() - phase_start, 4)))
                print(f'wip-cards: observer sampler_cleanup timed out after {cleanup_timeout:.2f}s (hook cleanup will retry)', file=sys.stderr, flush=True)
            phase_start = time.monotonic()
            wait_timeout = min(budget_s * 0.25, max(0.0, deadline - phase_start))
            try:
                sampler.wait(timeout=wait_timeout)
                phases.append(dict(phase='sampler_wait', status='ok', elapsed_s=round(time.monotonic() - phase_start, 4)))
            except subprocess.TimeoutExpired:
                phases.append(dict(phase='sampler_wait', status='timeout', elapsed_s=round(time.monotonic() - phase_start, 4)))
                with contextlib.suppress(OSError):
                    sampler.kill()
                with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                    sampler.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        # Give EOF a brief shared grace, then kill remaining SSH clients together
        # before reaping, all against the same overall deadline.
        eof_deadline = min(deadline, time.monotonic() + .15)
        for child, _, _, _ in spark_samplers:
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=max(0, eof_deadline - time.monotonic()))
        killed_sparks = set()
        for child, _, _, _ in spark_samplers:
            if child.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    child.kill()
                    killed_sparks.add(child.pid)
        for child, spark_reader, output, error in spark_samplers:
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                spark_errors.append('Spark memory sampler teardown timed out')
            if child.returncode not in (None, 0) and child.pid not in killed_sparks:
                spark_errors.append(f'Spark memory sampler exit {child.returncode}; see {error.name}')
            spark_reader.join(timeout=max(0, deadline - time.monotonic()))
            if spark_reader.is_alive():
                spark_errors.append('Spark memory reader teardown timed out')
            else:
                output.close()
            error.close()
        if reader:
            phase_start = time.monotonic()
            join_timeout = max(0.0, deadline - phase_start)
            reader.join(timeout=join_timeout)
            phases.append(dict(phase='reader_join', status='timeout' if reader.is_alive() else 'ok',
                               elapsed_s=round(time.monotonic() - phase_start, 4)))
        frames.close()
        policy.close()
        memory.close()
        console.close()
        save(dest / 'observer-teardown.json', dict(
            finish_timeout_s=finish_timeout_s,
            budget_s=round(budget_s, 4),
            elapsed_s=round(time.monotonic() - teardown_start, 4),
            phases=phases,
        ))
    if 'memory' in job['probes'] and not (dest / 'memory.jsonl').stat().st_size:
        raise ValueError('memory probe produced no samples')
    if spark_errors:
        raise RuntimeError('Spark memory sampler: ' + '; '.join(spark_errors))
    for rank, host in enumerate(job['entry'].get('sparks', []) if 'memory' in job['probes'] else []):
        if not (dest / f'memory-spark-rank{rank}.jsonl').stat().st_size:
            raise ValueError(f'Spark memory probe produced no samples for rank {rank} ({host})')
    if 'console' in job['probes'] and not (dest / 'console.jsonl').stat().st_size:
        raise ValueError('console probe produced no authenticated snapshots')


def peaks(dest):
    result = {}
    path = dest / 'memory.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            for row in json.loads(line)['devices']:
                values = result.setdefault(str(row['gpu']), {})
                for key in ('raw_bytes', 'corrected_bytes', 'sampler_bytes'):
                    values[key] = max(values.get(key, 0), row[key])
    return result


def ledger(dest):
    lines, rings, device_peaks = [], {}, {}
    for path in (dest / 'smoke' / 'logs').glob('*.log'):
        if (dest / 'smoke').is_symlink() and not path.name.startswith(dest.name + '.'):
            continue
        text = ANSI.sub('', path.read_text(errors='replace'))
        lines.extend(line for line in text.splitlines() if 'memory ledger' in line or 'device_occupied_bytes' in line or 'Spark unified memory state' in line)
        for line in text.splitlines():
            if 'memory ledger' not in line or 'report=' not in line:
                continue
            try:
                report = json.loads(line.split('report=', 1)[1])
            except ValueError:
                continue
            for device in report.get('devices', []):
                key = path.name + ':' + str(device['device'])
                device_peaks[key] = max(device_peaks.get(key, 0), device.get('used', 0))
            ring = report.get('pinned', {}).get('scopes', {}).get('transport/rdma-rings', 0)
            if ring:
                rings[path.name] = max(rings.get(path.name, 0), ring)
        for match in re.finditer(r'\b(?:ring_budget_peak_bytes_point_in_time|budget_peak)=(\d+)', text):
            rings[path.name] = max(rings.get(path.name, 0), int(match[1]))
    return {'lines': lines, 'spark_ring_peak_bytes': rings, 'device_used_peak_bytes': device_peaks}


def spark_profile_ranks(values, hosts):
    """The Spark ranks PROFILE_SPARK wraps (PROFILE_SPARK_RANKS, default rank 0),
    checked against this card's Spark count; the launcher checks them again."""
    text = values.get('PROFILE_SPARK_RANKS') or '0'
    if not re.fullmatch(r'(0|[1-9][0-9]?)(,(0|[1-9][0-9]?))*', text):
        raise ValueError('PROFILE_SPARK_RANKS must be a comma list of Spark ranks (e.g. 0 or 0,3)')
    ranks = [int(r) for r in text.split(',')]
    if len(set(ranks)) != len(ranks):
        raise ValueError('PROFILE_SPARK_RANKS lists a rank twice')
    if not hosts:
        raise ValueError('PROFILE_SPARK needs a card with Spark expert workers')
    if max(ranks) >= len(hosts):
        raise ValueError(f'PROFILE_SPARK_RANKS names rank {max(ranks)}; this card has ranks 0..{len(hosts) - 1}')
    return ranks


def cold_purge_argv(hosts):
    """SparkNest's clean page-cache drop on these hosts (its installed setuid
    helper; other workloads on them lose cached pages too). The CLI is resolved
    here, so a hook running under a systemd unit's minimal PATH still finds it."""
    nest = shutil.which('nest')
    if not nest:
        raise ValueError('--cold-window needs the SparkNest `nest` CLI on PATH')
    return [nest, 'drop-caches', *[arg for host in hosts for arg in ('--host', host)]]


def precheck_argv(first, second):
    """Stop on failed readiness evidence; preserve the client's actual exit."""
    return ['sh', '-c', shlex.join(first) + ' && exec ' + shlex.join(second), 'card-precheck']


def window_argv(commands):
    """One hook argv for one or more window commands: a single command runs as
    is; several run in order, every one attempted, failing if any failed."""
    if len(commands) == 1:
        return commands[0]
    script = 'rc=0; ' + ' '.join(shlex.join(c) + ' || rc=1;' for c in commands) + ' exit $rc'
    return ['sh', '-c', script, 'profile-window']


def expected_spark_reports(values, hosts):
    """Report files PROFILE_SPARK writes, one per selected rank, named by host and rank."""
    tool = values.get('PROFILE_SPARK')
    if tool not in PROFILER_REPORTS:
        return []
    return [f'{tool}-{hosts[rank]}-rank{rank}{PROFILER_REPORTS[tool]}' for rank in spark_profile_ranks(values, hosts)]


def collect_profile(job):
    """Copies Spark-side PROFILE files back into the job's profile directory
    (the coordinator writes there directly). Missing hosts are recorded, not fatal.
    Profiler reports (nsys/ncu) are listed with their sizes, not read."""
    values = job['entry']['set']
    directory = values.get('PROFILE_DIR')
    if not directory:
        return None
    result = {'dir': directory, 'switches': [s for s in values.get('PROFILE', '').split(',') if s], 'hosts': {}}
    if values.get('PROFILE_SPARK'):
        result['spark'] = {'tool': values['PROFILE_SPARK'], 'expected': expected_spark_reports(values, job['entry']['sparks'])}
    for host in job['entry']['sparks']:
        try:
            subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes', f'{host}:{directory}/', directory + '/'],
                           check=True, capture_output=True, text=True, timeout=300)
            result['hosts'][host] = 'collected'
        except (subprocess.SubprocessError, OSError) as error:
            result['hosts'][host] = 'failed: ' + (getattr(error, 'stderr', '') or str(error)).strip()[:200]
    result['files'] = sorted(p.name for p in Path(directory).glob('*') if p.is_file())
    result['reports'] = [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(Path(directory).glob('*'))
                         if p.is_file() and p.suffix in PROFILER_REPORTS.values()]
    return result


def fail(row, message, kind=None):
    """Fails a row keeping the first real error: later derived checks are
    appended to `errors`, never replace it."""
    errors = row.setdefault('errors', [row['error']] if row.get('error') else [])
    if message not in errors:
        errors.append(message)
    row['status'] = 'failed'
    row.setdefault('error', message)
    if not row['error']:
        row['error'] = message
    if kind and not row.get('failure_kind'):
        row['failure_kind'] = kind


def first_log_error(dest, entry_name):
    """The first real error line in a job's smoke logs: the launcher/worker
    output before smoke's own wrapper line ("error: ./run.sh exited ...")."""
    logs = dest / 'smoke' / 'logs'
    wrapper = None
    for name in (entry_name + '.log', entry_name + '.coordinator.log'):
        path = logs / name
        if not path.is_file():
            continue
        for line in ANSI.sub('', path.read_text(errors='replace')).splitlines():
            line = line.strip()
            if not FAILURE.search(line) or line.startswith('Error response from daemon: No such container'):
                continue
            if line.startswith('error: ./run.sh exited') or line.startswith('error: not ready'):
                wrapper = wrapper or line
                continue
            return line[:400]
    return wrapper


def classify(row, launched):
    """launch: the server never served the card; quality: it served and a
    correctness check failed; measurement: it served and a measurement or
    evidence check failed; preflight: refused before any hardware."""
    if row['status'] == 'pass':
        return None
    if row.get('exit_code') == 82:
        return 'preflight'
    if not launched:
        return 'launch'
    if row.get('quality') in ('fail', 'failed') or any(isinstance(row.get(k), dict) and row[k].get('status') in ('fail', 'failed') for k in ('cache', 'spec', 'lossless_set')):
        return 'quality'
    return 'measurement'


def host_load_advisories(phases):
    """Coordinator host load during timed phases, reported as advisory only.

    A 2026-11-16 calibration over same-config repeats found these bars miss most
    real slowdowns and flag many unaffected runs, so they never force a rerun on
    their own; a paired arm must also show the slowdown."""
    advisories = []
    for phase in phases:
        if not phase.get('timed'):
            continue
        name = phase.get('name', '?')
        load = phase.get('host_load') or {}
        cpu = load.get('cpu_some_avg10')
        if cpu is not None and cpu > 10:
            advisories.append(f"{name}: host load (advisory): coordinator CPU pressure {cpu:.1f}% (>10%)")
        memory = load.get('memory_some_avg10')
        if memory is not None and memory > 5:
            advisories.append(f"{name}: host load (advisory): coordinator memory pressure {memory:.1f}% (>5%)")
        scan = load.get('pgscan_pages_per_s')
        if scan is not None and scan > 1000:
            advisories.append(f"{name}: host load (advisory): coordinator page reclaim {scan:.0f} pages/s (>1000 pages/s)")
    return advisories


def fabric_rerun_reasons(phases):
    """Reconcile wire bytes with both endpoints, including older RX meters."""
    reasons = []
    for phase in phases:
        if not phase.get('timed'):
            continue
        ports = phase.get('ports', [])
        for p in ports:
            label = f"{p.get('host')} {p.get('device')}/{p.get('port')}"
            if any((p.get(k) or 0) > 0 for k in ('engine_sent_bytes', 'engine_received_bytes')) and 'DOWN' in (p.get('state') or ''):
                reasons.append(f"{phase.get('name', '?')}: port down while used: {label}")
            seconds = p.get('seconds') or 0
            if seconds <= 0:
                continue
            # Coordinator GIDs identify each Spark; never borrow unrelated bytes.
            import ipaddress
            try:
                address = p['host'].rsplit(':', 1)[0]
                gid = '00000000000000000000ffff' + ipaddress.IPv4Address(address).packed.hex()
            except (ValueError, KeyError):
                gid = None
            local_gids = {peer['local_gid'].replace(':', '').lower() for peer in p.get('peers', []) if peer.get('local_gid')}
            unique_port = sum(bool(other.get('peers')) for other in ports if other.get('host') == p.get('host')) == 1
            peers = [peer for other in ports if other.get('host') == 'coordinator'
                     for peer in other.get('peers', [])
                     if p.get('peers') and ((peer.get('peer', '').replace(':', '').lower() in local_gids) if local_gids
                         else unique_port and gid and peer.get('peer', '').replace(':', '').lower() == gid)]
            nic = sum(p.get(k) or 0 for k in ('sent_bytes', 'received_bytes'))
            storage_bytes = sum(p.get(k) or 0 for k in ('sparknest_sent_bytes', 'sparknest_received_bytes'))
            if storage_bytes * 8 / seconds / 1e9 > 1 and storage_bytes > nic * 0.05:
                reasons.append(f"{phase.get('name', '?')}: SparkNest traffic: {label} {storage_bytes * 8 / seconds / 1e9:.2f} Gb/s (>5% and >1 Gb/s)")
            for side, opposite in [('sent', 'received'), ('received', 'sent')]:
                physical, local = p.get(side + '_bytes'), p.get('engine_' + side + '_bytes')
                if physical is None or local is None:
                    continue
                storage = p.get('sparknest_' + side + '_bytes')
                if p.get('sparknest_status') not in (None, '', 'unregistered') and storage is None:
                    continue
                own = max(local, sum(peer.get(opposite + '_bytes', 0) for peer in peers))
                residual = max(0, physical - own - int(own * 0.65) - (p.get('estimated_header_' + side + '_bytes') or 0) - (storage or 0))
                if residual * 8 / seconds / 1e9 > 1 and residual > nic * 0.05:
                    reasons.append(f"{phase.get('name', '?')}: foreign traffic: {label} {residual * 8 / seconds / 1e9:.2f} Gb/s {side} (>5% and >1 Gb/s)")
    return list(dict.fromkeys(reasons))


def fabric_assessment(phases):
    """Fabric recovery is informational; raw counters, not old labels, decide."""
    caveats = fabric_rerun_reasons(phases) + host_load_advisories(phases)
    for phase in phases:
        if not phase.get('timed'):
            continue
        for p in phase.get('ports', []):
            h = p.get('health') or {}
            share = h.get('retransmit_pct')
            if h.get('retransmit_judged') and share is not None and share > (h.get('retransmit_warn_pct') or 0.5):
                caveats.append(f"{phase.get('name', '?')}: retransmits {share:.2g}% of packets; results remain valid, throughput may be lower")
        caveats.extend(f"{phase.get('name', '?')}: {a}; informational" for a in phase.get('assessments', [])
                       if a.startswith('storage attribution unavailable:'))
    return list(dict.fromkeys(caveats))


FALLBACK_BACKEND = 'mmap-fallback'


def table_backend_phases(report):
    """Every measured phase's mapped tables in run order: each panel pass and
    the sub-phases it recorded. A pass's own mapped_tables are added after the
    panel ran, so they read last."""
    phases = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, child in node.items():
                if key != 'mapped_tables' and isinstance(child, (dict, list)):
                    walk(child, f'{path}/{key}')
            tables = node.get('mapped_tables')
            if isinstance(tables, list):
                fabric = node.get('fabric')
                name = fabric.get('name') if isinstance(fabric, dict) else None
                phases.append((name or path, tables))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f'{path}/{index}')

    for panel in report.get('panels') or []:
        for index, pass_value in enumerate(panel.get('passes') or []):
            walk(pass_value, f"{panel.get('id', '?')}/pass{index + 1}")
    return phases


def table_backend_findings(phases):
    """(caveats, contaminated) from raw per-phase stats: a backend change after
    the table's first phase is contamination, while mmap from startup is one
    configuration note (io_uring was never available)."""
    first, caveats, contaminated = {}, [], False
    for phase, tables in phases:
        for table in tables or []:
            name, backend = table.get('name'), table.get('backend')
            if not name or not backend:
                continue
            if name not in first:
                first[name] = backend
                if backend == FALLBACK_BACKEND:
                    caveats.append(f'{name}: read backend {FALLBACK_BACKEND} from startup '
                                   '(io_uring unavailable); mapped gathers run slower, configuration only')
                continue
            if backend != first[name]:
                contaminated = True
                caveats.append(f'{phase}: {name} read backend changed from {first[name]} to {backend}; '
                               'mapped gathers are slower, compare in a quiet window')
    return caveats, contaminated


def timing_instrumentation(job):
    values = {**config(job['entry']['config']), **job['entry'].get('set', {})} if job['entry'].get('config') else job['entry'].get('set', {})
    profiles = set(filter(None, values.get('PROFILE', '').split(',')))
    layer = bool(profiles & {'draft-timing', 'transport-timing'})
    return {'per_layer_timing': 'on (perturbing)' if layer else 'off',
            'profiling': bool(profiles), 'source': 'launch configuration'}


def verify_rows(verify):
    """One summary row per verify shape: R, B, medians and graph state."""
    rows = []
    for sample in (verify.get('record') or {}).get('samples', []):
        timed = sorted(sample.get('verify_ms') or [])
        select = sorted(sample.get('select_ms') or [])
        rows.append(dict(shape=sample['shape'], R=sample['real_rows'], B=sample['bucket'], natural_B=sample['natural_bucket'],
                         verify_ms=statistics.median(timed) if timed else None, select_ms=statistics.median(select) if select else None,
                         verify_min_ms=timed[0] if timed else None, repeats=len(timed), graphed=sample.get('graphed'),
                         graph=sample.get('graph'), contexts=sorted(set(sample.get('contexts') or []))))
    return rows


def summarize_job(job, exit_code):
    dest = Path(job['state'])
    row = {k: job[k] for k in ('card', 'arm', 'repeat', 'simulated')}
    row.update(exit_code=exit_code, status='failed' if exit_code else 'pass', instrumentation=timing_instrumentation(job))
    if 'census' in job['entry'].get('set', {}).get('PROFILE', '').split(',') and (
            job['entry'].get('levels') or job['entry'].get('prefill_lengths')):
        try:
            merge_census_marker(job)
        except (OSError, ValueError, KeyError, TypeError) as error:
            fail(row, f'final census evidence: {error}', 'measurement')
    if 'artifacts' in job: row['artifacts'] = job['artifacts']
    if 'launcher' in job: row['launcher'] = job['launcher']
    report = None
    context_records, sweep = [], None
    reports = list((dest / 'reports').glob('**/report.json'))
    if job.get('grouped'):
        reports = [path for path in reports if path.parent.name.endswith('-' + job['entry']['name'])]
    state = dest / 'smoke' / 'state.json'
    if state.exists():
        row['smoke'] = json.loads(state.read_text())
    outcome = (row.get('smoke') or {}).get(job['entry']['name']) or {}
    row['fabric'] = list(outcome.get('fabric') or [])
    # The first real error wins: the launcher's own words, then smoke's outcome
    # (its wrapper "./run.sh exited ..." last), then checks derived below.
    found = [e for e in (first_log_error(dest, job['entry']['name']), outcome.get('error')) if e]
    found.sort(key=lambda e: e.startswith(('error: ./run.sh exited', './run.sh exited', 'error: not ready', 'not ready')))
    if exit_code or outcome.get('status') not in (None, 'done'):
        for message in found:
            fail(row, message)
    launched = outcome.get('launch_s') is not None or bool(reports)
    sweep_only = job['entry'].get('profile') == 'panels:' and sweeping(job['entry'])
    readiness_only = job['entry'].get('profile') == 'panels:' and job['entry'].get('baseline') is False and not sweep_only
    if (sweep_only or readiness_only) and not reports:
        pass  # No server-side profile ran; the sweep or the probes carry the results.
    elif len(reports) != 1:
        fail(row, f'expected one report, found {len(reports)}')
    else:
        report = json.loads(reports[0].read_text())
        server = report.get('server', {})
        row['drafter'] = server.get('drafter')
        settings = server.get('configuration', {}).get('settings', [])
        try:
            row['served_speculator'] = attest_served_speculator(job, settings)
        except ValueError as error:
            fail(row, str(error), 'measurement')
        for setting in settings:
            if setting.get('name') == 'per-layer-timing':
                row['instrumentation'].update(per_layer_timing=setting['value'], source='server resolved')
        row['table_caveats'], row['table_contaminated'] = table_backend_findings(table_backend_phases(report))
        if not row['fabric']:
            row['fabric'].extend(report.get('fabric') or [])
        present = {p['id'] for p in report.get('panels', [])} | ({'baseline'} if report.get('baseline') else set())
        missing = set(requested_panels(job['entry'].get('profile', 'smoke'), binary=job.get('binary'))) - present
        if missing:
            fail(row, 'requested panels absent from report: ' + ', '.join(sorted(missing)))
        baseline = report.get('baseline') or {}
        card = baseline.get('card') or {}
        quality = baseline.get('quality') or {}
        checks = quality.get('checks', [])
        fidelity = next((c for c in checks if c['id'] == 'fidelity'), {})
        row.update(C1=next((d['tok_s'] for d in card.get('decode', []) if d['content'] == 'code'), None), C8=next((c['aggregate_tok_s'] for c in card.get('concurrency', []) if c['width'] == 8), None), **{'8K': (card.get('prefill') or {}).get('tok_s'), '8K w/tail': (card.get('prefill_tail') or {}).get('tok_s')}, kl=fidelity.get('metrics', {}).get('kl'), top1=fidelity.get('metrics', {}).get('top1'), cache=next((c for c in checks if c['id'] in ('cache_exact', 'cache')), None), spec=next((c for c in checks if 'spec' in c['id']), None), pool=(card.get('capacity') or {}).get('kv_tokens'), readiness_s=report.get('server', {}).get('readiness_s'), report=str(reports[0]), quality=quality.get('status'))
        # Report/panel statuses serialize as 'failed'; Rust CheckStatus as 'fail'.
        failed = {'fail', 'failed'}
        bad = [c['id'] + (': ' + c['summary'] if c.get('summary') else '') for c in checks if c.get('status') in failed]
        if bad:
            fail(row, 'quality: ' + '; '.join(bad)[:400], 'quality')
        elif report.get('status') in failed or quality.get('status') in failed:
            fail(row, 'report ' + str(report.get('status')) + ': ' + str(report.get('error') or '')[:300])
        for panel in report.get('panels', []):
            if panel.get('status') in failed:
                fail(row, f"panel {panel['id']}: {panel.get('error') or panel['status']}", 'measurement')
        context_panel = next((p for p in report.get('panels', []) if p.get('id') == 'context_load'), None)
        if context_panel:
            records = context_panel.get('passes', [])
            context_records = records
            latest = records[-1] if records else {}
            if any(p.get('status') == 'failed' for record in records for p in record.get('points', [])):
                fail(row, 'context_load: request or prefix-reuse failure', 'measurement')
            row['context_graph_bands'] = latest.get('graph_bands', [])
        lossless = lossless_set_result(report)
        if lossless:
            row['lossless_set'] = lossless
            if lossless['status'] == 'fail':
                fail(row, 'quality: spec_lossless_set: ' + lossless['summary'][:400], 'quality')
            if job['entry'].get('lossless_dump') and lossless['status'] != 'skipped' and not lossless.get('row_dump'):
                # A server image without the option ignores it; the requested rows must exist.
                fail(row, 'spec_lossless_set: lossless_dump requested but the server wrote no row dump '
                     '(build the serving arm with row-dump support)', 'measurement')
    if job.get('correctness_parallel'):
        for metric in ('C1', 'C8', '8K'):
            row[metric] = None
    profile = collect_profile(job)
    if profile:
        row['profile'] = profile
        for switch, suffix in PROFILER_REPORTS.items():
            # The coordinator's report is named by its host alone; Spark reports add the rank.
            if switch in profile['switches'] and not any(r['name'].endswith(suffix) and r['bytes'] and '-rank' not in r['name']
                                                         for r in profile['reports']):
                fail(row, f'PROFILE={switch} produced no {suffix} report in {profile["dir"]}', 'measurement')
        if profile.get('spark'):
            sizes = {r['name']: r['bytes'] for r in profile['reports']}
            missing = [name for name in profile['spark']['expected'] if not sizes.get(name)]
            if missing:
                fail(row, f"PROFILE_SPARK={profile['spark']['tool']} produced no {', '.join(missing)} in {profile['dir']}", 'measurement')
        if 'expert-routes' in profile['switches']:
            logs = sorted(Path(profile['dir']).glob('expert-routes-*.jsonl'))
            profile['expert_routes'] = [{'name': p.name, 'bytes': p.stat().st_size} for p in logs]
            profile['expert_inputs'] = len(list(Path(profile['dir']).glob('expert-input-*.safetensors')))
            if not any(p.stat().st_size for p in logs):
                fail(row, f'PROFILE=expert-routes wrote no route log in {profile["dir"]}', 'measurement')
    trace_path = job['entry']['set'].get('SPECULATION_TRACE')
    if trace_path and job.get('trace'):
        path = Path(trace_path)
        records, proposed, cycles = 0, 0, 0
        if path.is_file():
            for line in path.open(errors='replace'):
                if not line.strip():
                    continue
                records += 1
                try:
                    observation = json.loads(line)
                except ValueError:
                    continue
                if 'depths' in observation and 'cycle_ms' in observation:
                    cycles += 1
                    proposed += sum(max(0, n) for n in observation['depths'])
                elif observation.get('type') == 'draft_round':
                    proposed += sum(r.get('proposed', 0) for r in observation.get('requests', []))
        expected = speculating(job['entry']['set']) or job.get('planned_speculator', 'off') != 'off'
        row['trace'] = {'path': trace_path, 'bytes': path.stat().st_size if path.is_file() else 0, 'records': records,
                        'expected': expected, 'proposal_evidence_rows': proposed, 'cycles': cycles}
        if expected and not records:
            fail(row, 'speculation trace is empty after the run (' + trace_path + ')', 'measurement')
        elif expected and cycles >= 8 and not proposed:
            fail(row, 'speculation requested but nonempty trace contains zero proposed tokens', 'measurement')
        if expected:
            for log in (dest / 'smoke' / 'logs').glob('*coordinator.log'):
                if re.search(r'DRAFT_LENGTH=.*ignored: no speculator|DRAFT_LENGTH.*no speculator', log.read_text(errors='replace')):
                    fail(row, 'DRAFT_LENGTH ignored: launch has no speculator', 'measurement')
    sweeps = [p for p in (dest / 'reports').glob('**/sweep.json')
              if not job.get('grouped') or p.parent.name.endswith('-' + job['entry']['name']) or p.parent.name == job['entry']['name']]
    if sweeping(job['entry']):
        if len(sweeps) != 1:
            fail(row, f'expected one sweep.json, found {len(sweeps)}')
        else:
            sweep = json.loads(sweeps[0].read_text())
            if not outcome.get('fabric'):
                row['fabric'].extend(p['fabric'] for one in (sweep.get('passes') or [sweep])
                                     for p in [*one.get('points', []), *one.get('prefill', [])] if p.get('fabric'))
            row['sweep'] = sweep.get('points', [])
            for point in row['sweep']:
                key = (f"sustained C{point['c']}" if point['phase'] == 'sustained' else
                       'post-load C1' if point['phase'] == 'post_load' else f"sweep C{point['c']}")
                row[key] = point['aggregate_tok_s']
            verify = sweep.get('verify')
            if verify:
                row['verify_shapes'] = verify_rows(verify)
                if not (verify.get('isolation') or {}).get('passed'):
                    fail(row, f"verify timing isolation failed: {verify.get('isolation')}", 'measurement')
            for point in sweep.get('prefill', []):
                row[f"prefill {point['length']}"] = point['tok_s']
                row[f"prefill {point['length']} TTFT s"] = point['ttft_s']
                if point.get('label') == '8K w/tail':
                    row['8K w/tail'] = point['tok_s']
            row['mapped_tables'] = {**{f"prefill {p['length']}": p['mapped_tables'] for p in sweep.get('prefill', [])
                                       if p.get('mapped_tables')},
                                    **{f"{p['phase']} C{p['c']}": p['mapped_tables'] for p in sweep.get('points', [])
                                       if p.get('mapped_tables')}}
            row['graph_bands'] = sweep.get('graph_bands', [])
    # context_load's timed points join the sweep's bands in one per-band table.
    row['graph_bands'] = [*row.get('graph_bands', []), *row.pop('context_graph_bands', [])]
    # Same-server passes: every pass's snapshot, and every pass's bands in the
    # table; the headline metrics above are the final pass's.
    snapshots = pass_snapshots(sweep, context_records)
    if snapshots:
        row['passes'] = snapshots
        row['headline_pass'] = len(snapshots)
        row['graph_bands'] = [band for snapshot in snapshots for part in ('sweep', 'context_load')
                              for band in (snapshot.get(part) or {}).get('graph_bands', [])]
    expected = entry_passes(job['entry'])
    recorded = [(name, n) for name, n in (('sweep', len(sweep.get('passes') or [sweep]) if sweep else 0),
                                          ('context_load', len(context_records))) if n]
    for name, count in recorded:
        if expected > 1 and count != expected:
            fail(row, f'sweep_passes={expected} but {name} recorded {count} pass(es) on the server '
                 '(build the smoke binary from a checkout with --sweep-passes)', 'measurement')
    row['peaks'] = peaks(dest)
    row['spark_memory'] = spark_memory_report(dest)
    row['memory_probe'] = dict(cadence_s=.05, spark_phase_refresh_s=1,
                               spark_guard_bytes=job.get('memory_guard_bytes', 0),
                               spark_context='none; host /proc/meminfo, bytes',
                               spark_phase_note='sampled labels; 1 s polling and up to 2 s console snapshot age; explicit ready/post-sweep samples')
    row['ledger'] = ledger(dest)
    coordinator = {k: v for k, v in row['ledger']['device_used_peak_bytes'].items() if '.coordinator.log:' in k}
    row['memory_crosscheck'] = {'ledger_device_used_peak_bytes': coordinator, 'cuda_corrected_peak_bytes': {k: v['corrected_bytes'] for k, v in row['peaks'].items()}, 'note': 'ledger point-in-time samples vs CUDA 50ms high-water; differences are retained, not treated as identical samples'}
    row['spark_ring_peaks'] = row['ledger']['spark_ring_peak_bytes']
    frames = dest / 'console-frames.jsonl'
    timings = {}
    if frames.exists():
        for line in frames.read_text().splitlines():
            message = json.loads(line)
            for event in message.get('ev', []):
                for stage, micros in (event.get('s') or {}).items():
                    timings.setdefault(stage, []).append(micros)
    row['stage_timings_us'] = {k: {'samples': len(v), 'median': statistics.median(v), 'max': max(v)} for k, v in timings.items()}
    for key in ('precheck',):
        path = dest / (key + '.json')
        if path.exists():
            row[key] = json.loads(path.read_text())
    if row.get('drafter') is None and (row.get('precheck') or {}).get('drafter') is not None:
        row['drafter'] = row['precheck']['drafter']
    if job.get('planned_speculator') is not None and not row.get('served_speculator'):
        evidence = row.get('precheck', {})
        if evidence.get('speculator') != job['planned_speculator']:
            fail(row, evidence.get('error') or 'missing served speculator attestation', 'measurement')
        else:
            row['served_speculator'] = evidence['speculator']
    if job.get('nonce_seed'):
        text = '\n'.join(ANSI.sub('', p.read_text(errors='replace')) for p in (dest / 'smoke' / 'logs').glob('*.coordinator.log'))
        hashes = prompt_token_hashes(text)
        all_hashes = hashes
        window = outcome.get('matched_panel_window')
        if job['entry'].get('matched_panel_window'):
            try:
                measured, untimed = measured_prompt_hashes(text, window, report)
                hashes = [h for wave in measured for h in wave]
                row['measured_prompt_waves'] = measured
                row['matched_panel_window'] = window
                row['untimed_prompt_token_hashes'] = untimed
            except (ValueError, KeyError, TypeError) as error:
                hashes = []
                fail(row, f'matched timed prompt attestation: {error}', 'measurement')
        row['all_prompt_token_hashes'] = all_hashes
        # Concurrent admission reorders requests, so pairs match on the whole
        # set of shared input-determined prompts; follow-up and arm-local
        # diagnostic prompts are reported but not compared.
        row.update(nonce_seed=job['nonce_seed'], first_prompt_token_hash=hashes[0] if hashes else None,
                   prompt_token_hashes=sorted(hashes), follow_up_prompts=follow_up_prompts(text),
                   arm_local_prompts=arm_local_prompts(text),
                   requests=matched_requests(dest), state=str(dest))
        if not hashes:
            fail(row, 'matched prompts produced no prompt token hashes')
    row['fabric_caveats'] = fabric_assessment(row['fabric'])
    row.setdefault('table_caveats', [])
    row.setdefault('table_contaminated', False)
    row['ab_verdict'] = 'rerun' if fabric_rerun_reasons(row['fabric']) or row['table_contaminated'] else 'eligible'
    row['failure_kind'] = row.get('failure_kind') or classify(row, launched)
    save(dest / 'result.json', row)
    return row


def lossless_set_result(report):
    """The spec_lossless_set panel's verdict: its gate over every case, or a
    failure when the panel errored or produced no pass. Case evidence stays in
    report.json."""
    panel = next((p for p in report.get('panels', []) if p.get('id') == 'spec_lossless_set'), None)
    if panel is None:
        return None
    passes = panel.get('passes') or []
    if panel.get('status') == 'unsupported':
        return {'status': 'skipped', 'summary': panel.get('error') or 'unsupported'}
    if not passes:
        return {'status': 'fail', 'summary': panel.get('error') or 'no pass recorded'}
    last = passes[-1]
    result = {'status': 'pass' if last.get('gate') == 'pass' else 'fail', 'summary': last.get('summary', ''),
              'passed': last.get('passed'), 'total': last.get('total'), 'set': last.get('set'),
              'failing': [r['id'] for r in last.get('rows', []) if not r.get('pass')]}
    if last.get('row_dump'):
        result['row_dump'] = last['row_dump']
    return result


# A matched pair compares the prompts its inputs decide. A prompt that
# re-renders model output (a follow-up turn, or ids taken from a response)
# logs as an output-dependent prompt with no hash, because two engines
# legitimately tokenize model text differently.
# Panel-local diagnostics also log without hashes: not every arm runs them.
MATCHED_PROMPT = re.compile(r'matched benchmark prompt.*?prompt_token_hash\s*=\s*"?([0-9a-f]{16})')
FOLLOW_UP_PROMPT = re.compile(r'benchmark output-dependent prompt')
ARM_LOCAL_PROMPT = re.compile(r'benchmark arm-local prompt')


def prompt_token_hashes(text):
    """The input-determined prompt hashes of a coordinator log, in log order."""
    return MATCHED_PROMPT.findall(text)


def measured_prompt_hashes(text, window, report):
    """Legacy servers lack per-request phase tags. The host brackets the run;
    the two measured panel protocols provide exact counts and order within it.
    Fail closed if a panel, an admission, or its timestamp is missing."""
    if not window or not report:
        raise ValueError('missing host measurement window or report')
    if window['prelude_path'] not in {'server-v1', 'client-v1'}:
        raise ValueError('unknown warm prelude path')
    lo, hi = window['start_unix_s'], window['end_unix_s']
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi):
        raise ValueError('invalid host measurement window')
    inside, untimed = [], []
    for line in text.splitlines():
        match = MATCHED_PROMPT.search(line)
        if not match:
            continue
        stamp = datetime.datetime.fromisoformat(line.split()[0].replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            raise ValueError('missing coordinator timestamp time zone')
        (inside if lo <= stamp.timestamp() <= hi else untimed).append(match[1])
    phases = []
    for panel in report['panels']:
        name = panel['id']
        if name not in SCOPED_PANELS:
            raise ValueError(f'cannot attest measurement panel {name!r}')
        if name not in {'decode_content', 'concurrency'}:
            continue
        if panel['status'] != 'done' or not panel['passes']:
            raise ValueError(f'incomplete measurement panel {name!r}')
        for result in panel['passes']:
            if name == 'decode_content':
                phases.extend((0, 1) for _ in result['rows'])
            else:
                for point in result['points']:
                    c = point['c']
                    if type(c) is not int or c < 1 or point.get('errors') != 0 or point.get('prompts') != 'distinct-v1' or point.get('phase'):
                        raise ValueError('concurrency point is not a successful distinct-v1 closed batch')
                    phases.append((c, c))
    if not phases or any(n <= 0 for _, n in phases) or sum(a + b for a, b in phases) != len(inside):
        raise ValueError(f'expected {sum(a + b for a, b in phases)} panel admissions, got {len(inside)}')
    measured, start = [], 0
    for warm, timed in phases:
        untimed.extend(inside[start:start + warm])
        start += warm
        measured.append(sorted(inside[start:start + timed]))
        start += timed
    return measured, untimed


def scoped_panel_window(card):
    profile = card.get('profile', '')
    panels = set(profile[7:].split(',')) if profile.startswith('panels:') else set()
    return (card.get('baseline') is False and panels <= SCOPED_PANELS
            and bool(panels & {'decode_content', 'concurrency'})
            and not card.get('levels') and not card.get('prefill_lengths')
            and not card.get('panel_config', {}).get('concurrency', {}).get('sustained'))


def follow_up_prompts(text):
    """Output-dependent prompts reported, which pairs do not compare."""
    return len(FOLLOW_UP_PROMPT.findall(text))


def arm_local_prompts(text):
    """Panel-local diagnostic prompts reported, which pairs do not compare."""
    return len(ARM_LOCAL_PROMPT.findall(text))


def matched_requests(dest):
    requests = {}
    path = dest / 'console-frames.jsonl'
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        for event in json.loads(line).get('ev', []):
            if event.get('e') == 'admit':
                requests.setdefault(str(event['id']), {}).update(prompt_tokens=event['prompt'], max_tokens=event['max'])
            if event.get('e') == 'first':
                requests.setdefault(str(event['id']), {})['first_ms'] = event['t']
            if event.get('e') == 'retire':
                requests.setdefault(str(event['id']), {}).update(retire_ms=event['t'], generated=event.get('gen'))
            if event.get('e') != 'round':
                continue
            for member in event.get('req', []):
                row = requests.setdefault(str(member[0]), {})
                row['steps'] = row.get('steps', 0) + 1
                row['emitted_tokens'] = row.get('emitted_tokens', 0) + member[4]
                row['step_ms'] = row.get('step_ms', 0) + event['t1'] - event['t0']
    output = []
    for id, row in requests.items():
        if not row.get('steps'):
            continue
        decode_ms = row.get('retire_ms', 0) - row.get('first_ms', 0) if 'first_ms' in row and 'retire_ms' in row else 0
        output.append(dict(request=id, **row, emitted_tok_s=1000 * max(0, (row.get('generated') or row['emitted_tokens']) - 1) / decode_ms if decode_ms > 0 else None, round_service_tok_s=1000 * row['emitted_tokens'] / row['step_ms'] if row['step_ms'] > 0 else None, tokens_per_step=row['emitted_tokens'] / row['steps'], ms_per_step=row['step_ms'] / row['steps']))
    return output


def validate_matched_pairs(rows, arm_names):
    for card, repeat in {(r['card'], r['repeat']) for r in rows if r.get('nonce_seed')}:
        pair = [r for r in rows if r['card'] == card and r['repeat'] == repeat]
        if {r['arm'] for r in pair} != set(arm_names):
            continue  # Not all arms have completed yet.
        hashes = [r.get('measured_prompt_waves', r.get('prompt_token_hashes')) for r in pair]
        if not all(hashes) or any(h != hashes[0] for h in hashes[1:]):
            for row in pair:
                fail(row, 'matched pair missing or mismatched prompt token hashes', 'measurement')
                row['exit_code'] = row.get('exit_code') or 81
                (Path(row['state']) / 'exit').write_text('81\n')
                save(Path(row['state']) / 'result.json', row)


def arm_pairs(arm_names):
    """(candidate, baseline) for every arm after the first.

    Arms named `<role><tag>` in exactly two roles, the first arm's role being
    the baseline and each tag having one arm of each role (base1/cand1,
    base2/cand2), pair within their tag: a four-arm lane study compares each
    lane mode against its own baseline, never cand2 against base1. Any other
    naming compares every arm against the first."""
    split = [re.fullmatch(r'(.*?)(\d+)', name) for name in arm_names]
    if len(arm_names) > 2 and all(split):
        roles = list(dict.fromkeys(m.group(1) for m in split))
        by_tag = {}
        for name, m in zip(arm_names, split):
            by_tag.setdefault(m.group(2), {})[m.group(1)] = name
        if len(roles) == 2 and all(set(arms) == set(roles) for arms in by_tag.values()):
            base, cand = roles
            return [(arms[cand], arms[base]) for arms in by_tag.values()]
    return [(candidate, arm_names[0]) for candidate in arm_names[1:]]


def summary_math(rows, arm_names, metrics=METRICS):
    medians, paired = [], []
    for card in sorted({r['card'] for r in rows}):
        for arm in arm_names:
            group = [r for r in rows if r['card'] == card and r['arm'] == arm and r['status'] == 'pass' and not r['simulated']]
            medians.append(dict(card=card, arm=arm, **{m: statistics.median([r[m] for r in group if isinstance(r.get(m), (float, int))]) if any(isinstance(r.get(m), (float, int)) for r in group) else None for m in metrics}))
        for candidate, baseline in arm_pairs(arm_names):
            for m in metrics:
                deltas, percentages = [], []
                for repeat in sorted({r['repeat'] for r in rows}):
                    pair = {r['arm']: r for r in rows if r['card'] == card and r['repeat'] == repeat and r['status'] == 'pass' and not r['simulated']}
                    a, b = pair.get(baseline, {}).get(m), pair.get(candidate, {}).get(m)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        deltas.append(b - a)
                        if a:
                            percentages.append(100 * (b - a) / a)
                paired.append(dict(card=card, baseline=baseline, candidate=candidate, metric=m, pairs=len(deltas), median_delta=statistics.median(deltas) if deltas else None, median_percent=statistics.median(percentages) if percentages else None))
    return medians, paired


def status_cell(check):
    if not isinstance(check, dict):
        return ''
    return str(check.get('status', '')) + (' (' + check['summary'][:60] + ')' if check.get('summary') and check.get('status') != 'pass' else '')


def spec_cell(row):
    """The baseline's single spec check, then the lossless set's k/N when it ran."""
    text = status_cell(row.get('spec'))
    lossless = row.get('lossless_set')
    if isinstance(lossless, dict):
        text = (text + '; ' if text else '') + f"set {lossless.get('passed', '?')}/{lossless.get('total', '?')} {lossless['status']}"
        if lossless.get('failing'):
            text += ' (' + ', '.join(lossless['failing'])[:80] + ')'
    return text


def drafter_differences(rows, arms):
    """Compare resolved records, not the launch options that requested them."""
    pairs = {}
    for row in rows:
        if row['status'] == 'pass':
            pairs.setdefault((row['card'], row['repeat']), {})[row['arm']] = row.get('drafter')
    differences = []
    for (card, repeat), layouts in sorted(pairs.items()):
        if len(layouts) < 2 or any(arm not in layouts for arm in arms):
            continue
        if any(layout != layouts[arms[0]] for layout in layouts.values()):
            differences.append(dict(card=card, repeat=repeat, arms=layouts))
    return differences


def write_summary(state, rows, arms):
    validate_matched_pairs(rows, list(arms))
    names = list(arms)
    drafter_changes = drafter_differences(rows, names)
    extra = sorted({k for r in rows for k in r if k.startswith('prefill ') and isinstance(r.get(k), (int, float))},
                   key=lambda k: (int(k.split()[1]), k))
    present = [m for m in (*SUMMARY_METRICS, *extra) if any(isinstance(r.get(m), (int, float)) for r in rows)]
    medians, paired = summary_math(rows, names, present or METRICS)
    for row in rows:
        row['fabric_caveats'] = fabric_assessment(row.get('fabric', []))
        row.setdefault('table_caveats', [])
        row.setdefault('table_contaminated', False)
        row['ab_verdict'] = 'rerun' if fabric_rerun_reasons(row.get('fabric', [])) or row['table_contaminated'] else 'eligible'
    rerun = any(row['ab_verdict'] == 'rerun' for row in rows)
    caveats = [dict(card=r['card'], arm=r['arm'], repeat=r['repeat'], caveat=c)
               for r in rows for c in [*r.get('fabric_caveats', []), *r.get('table_caveats', [])]]
    passes = [dict(card=r['card'], arm=r['arm'], repeat=r['repeat'], **{'pass': p['pass']}, passes=p['passes'],
                   headline=p['pass'] == r.get('headline_pass'),
                   graph_bands=[b for part in ('sweep', 'context_load') for b in (p.get(part) or {}).get('graph_bands', [])])
              for r in rows for p in r.get('passes', [])]
    save(state / 'summary.json', dict(rows=rows, medians=medians, paired_deltas=paired,
                                    ab_verdict='rerun' if rerun else 'eligible', caveats=caveats, drafter_differences=drafter_changes,
                                    **({'passes': passes} if passes else {})))
    def cell(value):
        if isinstance(value, float):
            return f'{value:.4g}'
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(',', ':'))
        return str(value if value is not None else '').replace('|', '\\|').replace('\n', ' ')
    text = f'# wip-cards summary: {state.name}\n\n'
    failed = [r for r in rows if r['status'] != 'pass']
    text += f'{len(rows) - len(failed)} of {len(rows)} jobs passed.\n\n'
    warm_paths = [f"{r['card']}/{r['arm']}/r{r['repeat']}: {r['matched_panel_window']['prelude_path']}"
                  for r in rows if r.get('matched_panel_window')]
    if warm_paths:
        text += 'Matched prelude paths: ' + ', '.join(warm_paths) + '.\n\n'
    multipass = sorted({r['headline_pass'] for r in rows if r.get('headline_pass')})
    if multipass:
        text += ('Headline speed is the final same-server pass (pass ' + '/'.join(map(str, multipass)) + ' of each '
                 'launch); every pass is in summary.json `passes` / `rows[].passes` and sweep.json `passes`.\n\n')
    # Before -> after per card and metric: medians of passing real-hardware rows,
    # each candidate's ratio to its own baseline (arm_pairs), and the paired
    # median where pairs exist. Two arms keep the single Ratio column.
    if len(names) > 1:
        pairs = arm_pairs(names)
        labels = ['Ratio'] if len(pairs) == 1 else [f'{cand}/{base}' for cand, base in pairs]
        text += ('## Before -> after\n\n| Card | Metric | ' + ' | '.join(names) + ' | '
                 + ' | '.join(f'{label} | Paired % | Pairs' for label in labels) + ' |\n')
        text += '| --- | --- |' + ' ---: |' * (len(names) + 3 * len(pairs)) + '\n'
        for card in sorted({r['card'] for r in rows}):
            by_arm = {m['arm']: m for m in medians if m['card'] == card}
            for metric in present:
                values = [by_arm.get(arm, {}).get(metric) for arm in names]
                if all(v is None for v in values):
                    continue
                cells = []
                for cand, base in pairs:
                    before, after = by_arm.get(base, {}).get(metric), by_arm.get(cand, {}).get(metric)
                    ratio = after / before if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before else None
                    pair = next((p for p in paired if p['card'] == card and p['candidate'] == cand
                                 and p['baseline'] == base and p['metric'] == metric), {})
                    cells.append(f"{cell(ratio)} | {cell(pair.get('median_percent'))} | {pair.get('pairs', 0)}")
                note = ' (lower is better)' if metric in LOWER_IS_BETTER or metric.endswith('TTFT s') else ''
                text += f"| {card} | {metric}{note} | " + ' | '.join(cell(v) for v in values) + ' | ' + ' | '.join(cells) + ' |\n'
        text += '\n'
    else:
        text += '## Results\n\n| Card | ' + ' | '.join(present) + ' |\n| --- |' + ' ---: |' * len(present) + '\n'
        for m in medians:
            text += f"| {m['card']} | " + ' | '.join(cell(m.get(k)) for k in present) + ' |\n'
        text += '\n'
    if drafter_changes:
        text += '## Resolved Drafter Differences\n\n'
        for change in drafter_changes:
            layouts = '; '.join(f"{arm}: {cell(layout) if layout is not None else 'not reported'}"
                                for arm, layout in change['arms'].items())
            text += f"- {change['card']} / r{change['repeat']}: {layouts}\n"
        text += '\n'
    if caveats:
        text += ('## A/B Verdict: Rerun\n\nTimed foreign traffic, a used port down, or a mapped-table backend change: compare in a quiet window.\n\n' if rerun
                 else '## Measurement Notes\n\nResults remain valid; retransmits may reduce throughput, and mmap from startup is configuration.\n\n')
        for c in caveats:
            text += f"- {c['card']} / {c['arm']} / r{c['repeat']}: {c['caveat']}\n"
        text += '\n'
    if any(r.get('fabric') for r in rows):
        text += '## Fabric By Arm And Phase\n\nPer-port tx+rx bytes; not unique-byte fabric totals. Engine headers are ceil(bytes/MTU)*62 lower bounds; SparkNest uses per-op packets*62. Residual includes ACK/retry and sample skew. Unknown is not zero.\n\n'
        text += '| Card | Arm | Repeat | Phase | Host / port | NIC | Engine | SparkNest | Headers est. | Residual |\n| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |\n'
        for r in rows:
            for phase in r.get('fabric', []):
                for p in phase.get('ports', []):
                    def pair(tx, rx):
                        a, b = p.get(tx), p.get(rx)
                        return a + b if isinstance(a, int) and isinstance(b, int) else 'unknown'
                    values = [pair(a,b) for a,b in [('sent_bytes','received_bytes'), ('engine_sent_bytes','engine_received_bytes'),
                              ('sparknest_sent_bytes','sparknest_received_bytes'), ('estimated_header_sent_bytes','estimated_header_received_bytes'),
                              ('foreign_sent_bytes','foreign_received_bytes')]]
                    text += f"| {r['card']} | {r['arm']} | {r['repeat']} | {cell(phase.get('name'))} | {cell(p.get('host'))} {cell(p.get('device'))}/{p.get('port')} | " + ' | '.join(map(cell, values)) + ' |\n'
        text += '\n### Fabric Health By Arm And Phase\n\n| Card | Arm | Phase | Host / port | Verdict | Retransmits (% of packets) | Judged | Reason |\n| --- | --- | --- | --- | --- | ---: | --- | --- |\n'
        for r in rows:
            for phase in r.get('fabric', []):
                for p in phase.get('ports', []):
                    h = p.get('health') or {}
                    values = [r['card'], r['arm'], phase.get('name'), f"{p.get('host')} {p.get('device')}/{p.get('port')}",
                              ('warn' if h.get('retransmit_judged') and (h.get('retransmit_pct') or 0) > (h.get('retransmit_warn_pct') or 0.5) else 'ok' if h.get('retransmit_judged') and h.get('retransmit_pct') is not None else 'idle / unavailable'), h.get('retransmit_pct', 'unknown'),
                              'yes' if h.get('retransmit_judged') else 'no (under 2 GB)',
                              'results remain valid']
                    text += '| ' + ' | '.join(map(cell, values)) + ' |\n'
        pauses = [p.get('health', {}).get('pause_counters', {}) for r in rows for phase in r.get('fabric', []) for p in phase.get('ports', []) if p.get('health')]
        text += '\nPFC pause: ' + ('per-priority deltas retained in JSON' if any(pauses) else 'unavailable in this report') + '. Retransmits cost goodput; no data is lost and results remain valid.\n\n'
    if any(r.get('graph_bands') for r in rows):
        # Steady state captures no graphs (AGENTS.md): every timed band should be capture-free.
        text += ('## Graph Captures Per Timed Band\n\n/v1/stats and /v1/thermal read just before and after each timed band '
                 '(sweep bands and context_load points). Throttle columns are the worst GPU: power-cap = software cap + power brake '
                 'seconds, thermal = software + hardware slowdown seconds. Eager sites: eager runs per site, spanning sites marked *.\n\n')
        text += ('Pass: same-server pass (`--sweep-passes`); the first captures shapes for the first time, '
                 'later passes should be capture-free.\n\n')
        text += ('| Card | Arm | Repeat | Pass | Band | Process captures | Device-cache captures | Misses | Eager runs | Capture-free '
                 '| Power-cap s | Thermal s | Busy SM MHz | Max C | Min headroom C | Eager sites |\n'
                 '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |\n')
        for r in rows:
            for band in r.get('graph_bands', []):
                free = {True: 'yes', False: 'NO'}.get(band.get('zero_captures'), 'unavailable: ' + str(band.get('reason')))
                thermal = band.get('thermal') or {}
                sites = ', '.join(f"{s['site']}{'*' if s.get('spans_devices') else ''} {s.get('eager_runs', 0)}"
                                  for s in band.get('sites') or [] if s.get('eager_runs'))
                text += '| ' + ' | '.join(map(cell, [r['card'], r['arm'], r['repeat'], band.get('pass', 1), band.get('band'), band.get('captures'),
                                                   band.get('device_captures'), band.get('misses'), band.get('eager_runs'), free,
                                                   thermal.get('power_cap_s'), thermal.get('thermal_slowdown_s'),
                                                   thermal.get('busy_sm_clock_mhz'), thermal.get('max_temperature_c'),
                                                   thermal.get('min_tlimit_headroom_c'), sites])) + ' |\n'
        text += '\n'
    if any(r.get('spark_memory') for r in rows):
        text += '## Spark Host Memory\n\nHost /proc/meminfo; no Spark CUDA context. Sampled phase labels, not exact wave boundaries. Minima are bytes.\n\n'
        text += '| Card | Arm | Repeat | Rank | Host | Phase | Samples | Minimum MemAvailable bytes |\n| --- | --- | ---: | ---: | --- | --- | ---: | ---: |\n'
        for r in rows:
            for rank, memory in r.get('spark_memory', {}).items():
                for phase, values in [('all', memory), *memory['phases'].items()]:
                    text += '| ' + ' | '.join(map(cell, [r['card'], r['arm'], r['repeat'], rank, memory['host'], phase,
                                                       values['samples'], values['min_mem_available_bytes']])) + ' |\n'
        text += '\n'
    text += '## Jobs\n\n| Card | Arm | Repeat | Status | Failure | Spec | Cache | Trace | Per-layer timing | Profile | Error |\n| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |\n'
    for r in rows:
        trace = r.get('trace') or {}
        trace_cell = (f"{trace.get('records', 0)} records" if trace.get('expected') else 'not expected') if trace else ''
        profile = r.get('profile') or {}
        profile_cell = f"{len(profile.get('files', []))} files" if profile else ''
        if profile.get('reports'):
            profile_cell += ': ' + ', '.join(f"{r['name']} ({r['bytes'] / 2**20:.1f} MiB)" for r in profile['reports'])
        text += f"| {r['card']} | {r['arm']} | {r['repeat']} | {r['status']}{' (simulated)' if r.get('simulated') else ''} | {r.get('failure_kind') or ''} | {cell(spec_cell(r))} | {cell(status_cell(r.get('cache')))} | {trace_cell} | {cell((r.get('instrumentation') or {}).get('per_layer_timing', 'unknown'))} | {profile_cell} | {cell((r.get('error') or '')[:240])} |\n"
    if failed:
        text += '\n## Failures\n\nlaunch = the server never served the card; quality = it served and a correctness check failed; measurement = it served and a measurement or evidence check failed; preflight = refused before hardware.\n\n'
        for r in failed:
            text += f"- **{r['card']} / {r['arm']} / r{r['repeat']}** ({r.get('failure_kind') or 'unknown'}): {r.get('error') or 'exit ' + str(r.get('exit_code'))}\n"
            for extra in (r.get('errors') or [])[1:]:
                text += f'  - then: {extra}\n'
    (state / 'summary.md').write_text(text)


def bound_ports(stdout):
    """Local TCP listen ports from `ss -Hltn` output."""
    ports = set()
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[3].rpartition(':')[2].isdigit():
            ports.add(int(fields[3].rpartition(':')[2]))
    return ports


def assert_absent(job):
    for host, name in container_targets(job):
        cmd = ['docker', 'ps', '-a', '--format', '{{.Names}}']
        if host != 'raptor':
            cmd = ['ssh', '-o', 'BatchMode=yes', host, shlex.join(cmd)]
        if name in run(cmd).stdout.splitlines():
            raise ValueError('refusing existing (possibly foreign) container: ' + name)


def listening_ports():
    """This host's TCP listen ports."""
    return bound_ports(run(['ss', '-Hltn']).stdout)


def assert_port_free(job):
    """Two separate jobs on disjoint GPUs can share a kit entry's fixed ADDR port;
    the second would then measure the first job's server. Refuse a bound port."""
    port = int(resolve_container_keys(job['entry'])['addr'].rpartition(':')[2])
    if port in listening_ports():
        raise ValueError(f'refusing card: ADDR port {port} is already bound on raptor (another server is listening)')


def slot_check(arm, host):
    if 'release' in arm:
        image = 'ghcr.io/tpurtell/cuteafd-' + ('coordinator' if host == 'raptor' else 'spark-expert') + ':' + arm['release']
        cmd = ['docker', 'image', 'inspect', image]
        info = json.loads(run(cmd if host == 'raptor' else ['ssh', '-o', 'BatchMode=yes', host, shlex.join(cmd)]).stdout)[0]
        return json.dumps({'image': image, 'image_id': info['Id'], 'repo_digests': info.get('RepoDigests', [])})
    instance = arm.get('wip_instance', arm['instance'])
    container = ('cuteafd-coordinator-wip-' if host == 'raptor' else 'cuteafd-spark-expert-wip-') + instance
    role = 'coordinator' if host == 'raptor' else 'spark-expert'
    # Reads the sealed slot from the host side of the /wip bind mount: builds stop
    # their WIP containers, and a check must never start one. The root is the
    # task's own build root, else the (possibly stopped) container's /wip mount
    # source, else the default WIP_ROOT layout. Only a legacy root slot the host
    # user cannot read falls back to exec in an already running container.
    root = arm.get('root') or '-'
    cmd = ['python3', '-c', SLOT_SEAL_CHECK, arm['slot'], role, instance, container, root]
    try:
        return run(cmd if host == 'raptor' else ['ssh', '-o', 'BatchMode=yes', host, shlex.join(cmd)]).stdout
    except subprocess.SubprocessError as error:
        detail = (getattr(error, 'stderr', None) or str(error)).strip().splitlines()
        raise ValueError(f'{host}: missing or unsealed WIP slot {arm["instance"]}:{arm["slot"]}: {detail[-1] if detail else error}') from None


SLOT_SEAL_CHECK = '''import hashlib,json,pathlib,subprocess,sys
slot,role,instance,container,root=sys.argv[1:6]
def check(p,read):
    meta=read(p/'META.json')
    assert hashlib.sha256(meta).hexdigest()==read(p/'FINGERPRINT').decode().strip(), 'META seal mismatch'
    m=json.loads(meta)
    assert m['slot']==slot and m['role']==role and m['wip_instance']==instance, 'slot identity mismatch'
    for relative,key in [('SOURCE_SHA256SUMS','source_manifest_sha256'),('workspace/.cuteafd-wip/ARTIFACT_SHA256SUMS','artifact_manifest_sha256')]:
        assert hashlib.sha256(read(p/relative)).hexdigest()==m[key], relative+' seal mismatch'
    m['seal_sha256']=hashlib.sha256(meta).hexdigest()
    return m
if root=='-':
    found=subprocess.run(['docker','inspect','-f','{{range .Mounts}}{{if eq .Destination "/wip"}}{{.Source}}{{end}}{{end}}',container],capture_output=True,text=True)
    root=found.stdout.strip() if found.returncode==0 and found.stdout.strip() else str(pathlib.Path.home()/'.cache/cuteafd/builds'/('wip-'+instance))
host=pathlib.Path(root)/'slots'/slot/role
def contents(p):
    art=p/'workspace/.cuteafd-wip'
    packages=sorted(str(d.relative_to(art)) for kind in ('exl3','fp8') if (art/kind).is_dir() for d in (art/kind).iterdir() if d.is_dir())
    audio=pcm_audio=None
    try:
        out=subprocess.run(['nm','-D','--defined-only',str(art/'libcuteafd_native.so')],capture_output=True,text=True,timeout=60)
        if out.returncode==0:
            symbols={line.split()[-1] for line in out.stdout.splitlines() if line.split()}
            audio='cuteafd_audio_required' in symbols
            pcm_audio='cuteafd_pcm_required' in symbols
    except (OSError,subprocess.SubprocessError):
        pass
    variants=sorted(str(lib.parent.relative_to(art)) for pattern in
        ('exl3/*/tp*-rank*/m*/libcuteafd_exl3.so','fp8/*/tp*/libcuteafd_fp8moe.so')
        for lib in art.glob(pattern) if lib.is_file())
    return dict(packages=packages,spark_variants=variants,audio=audio,pcm_audio=pcm_audio,tp_roles=tp_roles(p,art))
def tp_roles(p,art):
    # The V4.1 Spark TP roles this slot attests, verified as run-family.sh's
    # launch_deepseek_v41 verifies them; None when the slot carries no manifest.
    try:
        meta=json.loads((p/'META.json').read_bytes())
        data=(art/'V41_EXPERT_TP_AOT.json').read_bytes()
    except (OSError,ValueError):
        return None
    try:
        assert hashlib.sha256(data).hexdigest()==meta.get('v41_expert_tp_manifest_sha256'),'TP manifest hash mismatch'
        manifest=json.loads(data)
        roles=manifest.get('spark_tp_roles')
        assert manifest.get('schema')==1 and isinstance(roles,list) and all(r in ('tp2','tp3','tp5','tp6','tp7','tp8') for r in roles),'invalid TP roles'
        assert roles==meta.get('spark_tp_roles'),'slot and artifact roles differ'
        lib=hashlib.sha256((art/'libcuteafd_native.so').read_bytes()).hexdigest()
        assert lib==manifest.get('native_library_sha256'),'native library hash mismatch'
        return dict(roles=roles,verified=manifest.get('symbols_verified') is True,nvfp4=manifest.get('nvfp4_spark_tp_roles',[]),error=None)
    except (OSError,ValueError,AssertionError) as error:
        return dict(roles=[],verified=False,nvfp4=[],error=str(error))
try:
    m=check(host,lambda path: path.read_bytes())
    m['seal_source']='host:'+str(host)
    m['contents']=contents(host)
except PermissionError:
    state=subprocess.run(['docker','inspect','-f','{{.State.Running}}',container],capture_output=True,text=True).stdout.strip()
    if state!='true':
        raise
    m=check(pathlib.Path('/wip/slots')/slot/role,lambda path: subprocess.run(['docker','exec',container,'cat',str(path)],check=True,capture_output=True).stdout)
    m['seal_source']='exec:'+container
print(json.dumps(m))
'''


def cleanup_script(arm, task):
    # Ownership is durable and explicit. Existing WIP arms are never adopted by
    # running cards; only roots created by this task's build helper can be erased.
    # An instance may hold several slots (e.g. gate/pre891/at891/devclone), so a
    # cleanup removes one slot's build root, overlay and registrations, and only
    # tears down the instance once no slot remains on that host.
    instance, slot = arm['instance'], arm['slot']
    root = arm.get('root', str(Path.home() / '.cache/cuteafd/builds' / ('wip-' + instance)))
    owner = json.dumps({'task': task, 'instance': instance}, sort_keys=True)
    script = '''import json, pathlib, shutil, subprocess, sys
home=pathlib.Path.home()
root=home/'.cache/cuteafd/builds'/ROOT_NAME
owner=OWNER
instance=INSTANCE
slot=SLOT
names=NAMES
runs=home/'.cache/cuteafd/wip-run'/instance
slots=home/'.cache/cuteafd/wip-slots'/instance
overlay=runs/slot
registration=slots/(slot+'.json')
legacy=home/'.cache/cuteafd/wip-slots'/(slot+'.json')
def remaining_slots():
    # Any build slot, runtime overlay or per-instance registration still on
    # this host means the instance root must survive.
    remaining=set()
    build_slots=root/'slots'
    if build_slots.is_dir(): remaining.update(p.name for p in build_slots.iterdir())
    if runs.is_dir(): remaining.update(p.name for p in runs.iterdir())
    if slots.is_dir(): remaining.update(p.stem for p in slots.glob('*.json'))
    remaining.discard(slot)
    return remaining
present=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
residual=[overlay,registration]
if legacy.is_file() and json.loads(legacy.read_text()).get('wip_instance')==instance: residual.append(legacy)
if not root.exists():
    if any(name in present for name in names) or any(p.exists() or p.is_symlink() for p in residual):
        raise RuntimeError('unowned cleanup residue without WIP root '+str(root))
    print('verified absent', root); sys.exit(0)
if root.is_symlink(): raise RuntimeError('refusing root symlink')
marker=root/'.wip-card-owner.json'
if not marker.is_file() or json.loads(marker.read_text())!=owner:
    raise RuntimeError('refusing unowned WIP root '+str(root))
for name in names:
    ids=subprocess.check_output(['docker','ps','-aq','--filter','name=^'+name+'$'],text=True).split()
    for ident in ids:
        info=json.loads(subprocess.check_output(['docker','inspect',ident],text=True))[0]
        env=info['Config'].get('Env',[])
        mounts=info.get('Mounts',[])
        if 'WIP_INSTANCE='+instance not in env or not any(m['Source']==str(root) for m in mounts):
            raise RuntimeError('refusing foreign container '+name)
        subprocess.run(['docker','rm','-f',ident],check=True)
    remaining=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
    if name in remaining: raise RuntimeError('container still present '+name)
if overlay.is_symlink(): raise RuntimeError('refusing overlay symlink')
if overlay.exists(): shutil.rmtree(overlay)
for registration, shared in [(slots/(slot+'.json'), False),(legacy, True)]:
    if registration.exists():
        data=json.loads(registration.read_text())
        if data.get('wip_instance')==instance and data.get('slot')==slot: registration.unlink()
        # The legacy slot-name file is shared by every instance using that slot
        # name; another instance's entry there is not ours to remove or refuse.
        elif not shared: raise RuntimeError('refusing foreign registration '+str(registration))
build_slot=root/'slots'/slot
if build_slot.is_symlink(): raise RuntimeError('refusing slot root symlink')
removed=[overlay,slots/(slot+'.json'),build_slot]
if legacy.is_file() and json.loads(legacy.read_text()).get('wip_instance')==instance: removed.append(legacy)
if build_slot.exists(): shutil.rmtree(build_slot)
if not remaining_slots():
    # The last slot on this host also releases the instance root.
    if root.is_symlink(): raise RuntimeError('refusing root symlink')
    shutil.rmtree(root)
for path in removed:
    if path.exists() or path.is_symlink(): raise RuntimeError('cleanup failed '+str(path))
print('verified absent', root, overlay)
'''
    replacements = {'ROOT_NAME': repr(Path(root).name), 'OWNER': repr(json.loads(owner)), 'NAMES': repr(['cuteafd-coordinator-wip-' + instance, 'cuteafd-spark-expert-wip-' + instance]), 'INSTANCE': repr(instance), 'SLOT': repr(slot)}
    return re.sub(r'\b(ROOT_NAME|OWNER|NAMES|INSTANCE|SLOT)\b', lambda m: replacements[m[0]], script)


def cleanup(arms, task, dry, seed=None):
    for arm in arms.values():
        if 'wip_instance' in arm or 'release' in arm:
            continue  # External sealed builds and release images are never owned by this task.
        script = cleanup_script(arm, task)
        hosts = dict.fromkeys(('raptor', *HOSTS, *SEED_HOSTS, *arm.get('seed_candidates', []), *arm.get('cleanup_hosts', []), *([seed] if seed and seed != 'auto' else [])))
        for host in hosts:
            cmd = ['python3', '-c', script] if host == 'raptor' else ['ssh', '-o', 'BatchMode=yes', host, shlex.join(['python3', '-c', script])]
            print(shlex.join(cmd))
            if not dry:
                print(run(cmd).stdout.strip())
        # The instance's sparknest build area (sealed slot archives a non-RoCE seed
        # staged, wip.sh wip_nest_slot_dir) goes with it; it only ever holds ours.
        area = NEST_BUILD_AREA / arm['instance']
        cmd = ['bash', '-c', 'if [ -e "$1" ]; then nest rm -r "$1"; else echo "verified absent $1"; fi', 'nest-area', str(area)]
        print(shlex.join(cmd))
        if not dry and NEST_BUILD_AREA.parent.is_dir():
            print(run(cmd).stdout.strip())


NEST_BUILD_AREA = Path('/mnt/sparknest/build-cache/wip')


def resolved_card(card, overrides, unsets=()):
    """One effective configuration for scope derivation and package preflight."""
    values = {**config(card['config']), **card['set'], **overrides}
    for key in unsets:
        values.pop(key, None)
    count = int(values.get('SPARK_COUNT', len(card.get('sparks', []))))
    if not 0 <= count <= len(HOSTS):
        raise ValueError('invalid Spark count')
    sparks = [values.get(f'SPARK_{i}_HOST', card['sparks'][i] if i < len(card['sparks']) else HOSTS[i]) for i in range(count)]
    return {**card, 'set': values, 'resolved_values': values, 'sparks': sparks}


def build_scopes(cards):
    env = family_scopes(cards, config)
    env.update(CUTEAFD_KACHE_SPARK='1', CUTEAFD_KACHE=str(Path.home() / '.cache/cuteafd/tools/x86_64/kache'), CARGO_BUILD_JOBS=os.environ.get('CARGO_BUILD_JOBS', '16'), RUST_TEST_THREADS=os.environ.get('RUST_TEST_THREADS', '16'))
    return env


def package_dir(scope):
    """The slot directory (under .cuteafd-wip/) a FAMILY:FORMAT scope ships as
    (scripts/build/build-wip-artifacts.sh), or None for scopes with no package."""
    family, fmt = scope.split(':', 1)
    if fmt == 'fp8':
        return 'fp8/fp8-' + family
    if fmt in ('nvfp4', 'nvfp4a4'):
        return f'fp8/fp8-{family}-nvfp4a4'
    if fmt == 'mxfp8':
        return f'fp8/fp8-{family}-mxfp8'
    if fmt.startswith('exl3-k'):
        return f'exl3/exl3-{family}-{fmt[5:]}'
    return None


def slot_needs(card):
    """What a sealed slot must contain to launch `card`, per role: the expert
    packages its scopes ship (the same derivation --build uses) and, when the
    card places the audio tower, the native audio entry point."""
    scopes = build_scopes([card])
    coordinator = {package_dir(s) for s in scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';') if s} - {None}
    spark = {package_dir(s) for s in scopes['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';') if s} - {None}
    values = card['resolved_values'] if 'resolved_values' in card else {**config(card['config']), **card.get('set', {})}
    if int(values.get('SPARK_COUNT', len(card.get('sparks', [])))) == 0:
        spark = set()
    return dict(coordinator=sorted(coordinator), spark=sorted(spark),
                audio=scopes.get('CUTEAFD_WIP_AUDIO_AOT') == 'ON', **v41_tp_role(card, values))


def v41_tp_role(card, values):
    """The SM121 expert role a DeepSeek V4.1 card's Spark TP requires, as the
    planner names it (`artifact_requirements.spark_tp_role`): a native non-EXL3
    count other than 0 and 4 (TP = count). TP4 is the built-in default role.
    NVFP4 checkpoints need the role's NVFP4 exports too."""
    if card.get('family') != 'deepseek_v41':
        return dict(tp_role=None, nvfp4=False)
    count = int(values.get('SPARK_COUNT', len(card.get('sparks', []))))
    role = f'tp{count}' if 2 <= count <= 8 and count != 4 else None
    return dict(tp_role=role, nvfp4=checkpoint_is_nvfp4(values))


def checkpoint_is_nvfp4(values):
    """Whether the card's checkpoint ships NVFP4 routed experts (its own
    config.json's moe_quant_algo, as the launcher reads it)."""
    model = values.get('MODEL_ID', 'deepseek-ai/DeepSeek-V4.1-Flash-1017')
    hub = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub' / ('models--' + model.replace('/', '--'))
    revision = values.get('MODEL_REVISION') or ((hub / 'refs/main').read_text().strip() if (hub / 'refs/main').is_file() else '')
    try:
        quant = json.loads((hub / 'snapshots' / revision / 'config.json').read_text()).get('quantization_config') or {}
        return quant.get('moe_quant_algo') == 'NVFP4'
    except (OSError, ValueError):
        return 'nvfp4' in model.lower()


def toolchain_hash(revision):
    """The dev toolchain identity of `revision` (scripts/build/dev-toolchain.py
    over that revision's own files), so a build runs in its pinned dev image."""
    def read_blob(name):
        shown = subprocess.run(['git', '-C', str(REPO), 'show', f'{revision}:{name}'], capture_output=True, timeout=120)
        if shown.returncode:
            raise ValueError(f'cannot read {name} at {revision}: ' + shown.stderr.decode(errors='replace').strip())
        return shown.stdout

    script = 'scripts/build/dev-toolchain.py'
    source = read_blob(script)
    # Older revisions predate the shared INPUTS declaration. Never execute a
    # historical toolchain script just to discover its inputs.
    inputs = ('docker/Dockerfile.dev', 'docker/entrypoint.sh', 'scripts/build/install-dev-cache-tools.sh', script)
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'INPUTS' for target in node.targets):
            inputs = ast.literal_eval(node.value)
            break
    if not isinstance(inputs, tuple) or not inputs or not all(isinstance(name, str) for name in inputs):
        raise ValueError(f'invalid toolchain INPUTS at {revision}')
    digest = hashlib.sha256()
    for name in inputs:
        blob = source if name == script else read_blob(name)
        digest.update(name.encode() + b'\0' + blob + b'\0')
    return digest.hexdigest()[:16]


# Serving-only keys never belong in a build (staging) config.
BUILD_DROP = {'SPARK_HOSTS', 'BENCH_SIMULATED', 'BENCH_HARDWARE_CLASS', 'BENCH_EXPECT_GLM_RING',
              'API_KEY_FILE', 'INSTANCE',
              'WIP_INSTANCE', 'WIP_ROOT', 'ADDR', 'EXPERT_PORT', 'PROFILE', 'PROFILE_DIR', 'SPECULATION_TRACE',
              'PROFILE_NCU_KERNEL', 'PROFILE_NCU_COUNT', 'PROFILE_NCU_SET', 'PROFILE_NSYS_GPU_METRICS', 'PROFILE_NSYS_ROOT',
              'PROFILE_NCU_SKIP', 'PROFILE_SPARK', 'PROFILE_SPARK_RANKS', 'PROFILE_SPARK_NCU_KERNEL', 'PROFILE_SPARK_NCU_COUNT',
              'PROFILE_SPARK_NCU_SET', 'PROFILE_SPARK_NCU_SKIP', 'PROFILE_SPARK_NSYS_GPU_METRICS',
              'PROFILE_EXPERT_INPUT_SAMPLE', 'PROFILE_EXPERT_ROUTE_WEIGHTS', 'COORDINATOR_DOCKER_INFERENCE', 'SPARK_EXPERT_DOCKER_INFERENCE'}


def build_lock_mode(revision):
    text = run(['git', '-C', str(REPO), 'show', revision + ':wip.sh']).stdout
    native = run(['git', '-C', str(REPO), 'show', revision + ':scripts/build/build-wip-artifacts.sh']).stdout
    # CPU-export sources run all AOT exports on CPU without GPU export locks.
    if 'CUTEAFD_WIP_EXPORT_LOCKS' not in text and 'CUDA_VISIBLE_DEVICES=' in text:
        return 'cpu-export'
    # Historical native-phase sources lock only their GPU export phase.
    if 'CUTEAFD_WIP_EXPORT_LOCKS' in text and ('wip_capped_export()' in text or 'CUTEAFD_WIP_HOST_EXPORT_TURN' in text or 'CUTEAFD_WIP_EXPORT_LOCK_FILES' in native):
        return 'native-phase'
    return 'legacy-full-build'


def build_slot_capable(revision):
    """Whether `revision`'s wip.sh takes a build slot around Cargo itself (CUTEAFD_WIP_BUILD_SLOT)."""
    shown = subprocess.run(['git', '-C', str(REPO), 'show', revision + ':wip.sh'], capture_output=True, text=True, timeout=120)
    return shown.returncode == 0 and 'CUTEAFD_WIP_BUILD_SLOT' in shown.stdout


def seed_option_capable(revision):
    """Whether `revision`'s wip.sh takes its build seed from WIP_SEED_HOST. An older
    one builds on SPARK_0_HOST, so its staging config must lead with the seed."""
    shown = subprocess.run(['git', '-C', str(REPO), 'show', revision + ':wip.sh'], capture_output=True, text=True, timeout=120)
    return shown.returncode == 0 and 'WIP_SEED_HOST' in shown.stdout


def staging_hosts(union, seed, separate):
    """The Spark hosts a staging config lists. With a separate seed (WIP_SEED_HOST)
    they are only the serving union, filled to six when its size is not a rank count
    config validation admits (2/3/4/6), preferring a pool-Spark seed as filler; a seed
    outside the pool (kakapo) is never named. A legacy wip.sh builds on SPARK_0_HOST,
    so there the seed leads."""
    if separate:
        hosts = [h for h in HOSTS if h in union]
        spare = [*([seed] if seed in HOSTS and seed not in hosts else []), *(h for h in HOSTS if h not in hosts and h != seed)]
    else:
        hosts = [seed, *[h for h in HOSTS if h in union and h != seed]]
        spare = [h for h in HOSTS if h not in hosts]
    # Config validation admits 2/3/4/6 ranks, not arbitrary staging unions.
    return hosts if len(hosts) in (2, 3, 4, 6) else hosts + spare[:6 - len(hosts)]


def lan_seeds_known(revision):
    """The LAN-only seeds `revision`'s tooling knows: its release_lan_seed_hosts, or
    kakapo alone before that set existed (such a wip.sh neither balances nor
    detects cassowary's down RoCE ports)."""
    shown = subprocess.run(['git', '-C', str(REPO), 'show', revision + ':scripts/lib/release-common.sh'],
                           capture_output=True, text=True, timeout=120)
    found = re.search(r'^release_lan_seed_hosts=\(([^)]*)\)', shown.stdout, re.M) if shown.returncode == 0 else None
    return tuple(found[1].split()) if found else ('kakapo',)


def resolve_seed(seed, revision, separate):
    """(seed wip.sh starts from, whether it is a placement request, hosts it may build
    on). 'auto' starts from kakapo and lets wip.sh balance over the LAN-only seeds
    this revision knows, then fall back to rhea/moa; a named seed is passed as
    `--seed`, a placement request."""
    known = lan_seeds_known(revision)
    if seed == 'auto':
        return 'kakapo', False, (*known, 'rhea', 'moa') if separate else ('kakapo',)
    if seed in SEED_HOSTS and seed not in known:
        raise ValueError(f'{revision} predates the LAN-only seed {seed}; use --seed auto, '
                         f'{", ".join(known)}, rhea or moa for this arm')
    return seed, True, (seed,)


def seed_env(arm, seed, dry):
    """wip.sh's seed parameters for a Spark build: WIP_SEED_HOST, plus the bulk
    transfer address of a named seed off the RoCE fabric (an automatic one is
    resolved by wip.sh after it balances). Legacy wip.sh reads neither."""
    if not arm.get('seed_separate') or arm.get('build_role') != 'both':
        return {}
    if seed in HOSTS or not arm.get('seed_explicit', True):
        return dict(WIP_SEED_HOST=seed)
    return dict(WIP_SEED_HOST=seed, WIP_SEED_ADDR=seed_lanes(seed, dry)[0])


def seed_lanes(seed, dry):
    """LANE_A/LANE_B of a build seed. A pool Spark has its RoCE rails; an
    out-of-pool seed without RoCE uses its directly attached LAN address
    (scripts/build/seed-transfer.py), which wip.sh uses for every bulk transfer
    (WIP_SEED_ADDR; never a SPARK_* key)."""
    if seed in HOSTS:
        idx = HOSTS.index(seed) + 1
        return f'10.55.0.{idx}', f'10.55.1.{idx}'
    if dry:
        return '0.0.0.0', '0.0.0.0'
    addr = run([sys.executable, str(REPO / 'scripts/build/seed-transfer.py'), 'lan-addr', seed]).stdout.strip()
    return addr, addr


def resolve_builds(items):
    """Pins every --build ARM=REV to a commit once, before any build runs: a symbolic
    ref that moves during hours of builds must not disagree with the revision that was
    built or with the launcher checkout the launcher phase checks it against (#315)."""
    builds = {}
    for item in items:
        name, revision = item.split('=', 1)
        if name in builds:
            raise ValueError('duplicate --build arm ' + name)
        builds[name] = run(['git', '-C', str(REPO), 'rev-parse', revision + '^{commit}']).stdout.strip()
    return builds


def owner_marker_script(root, task, instance, slot, revision):
    # A direct WIP retry may predate the cards owner marker. Adopt only a single
    # slot whose seals, identity and frozen source all match this build task.
    return """import hashlib,json,pathlib
p=pathlib.Path.home()/'.cache/cuteafd/builds'/ROOT_NAME
assert not p.is_symlink(), 'refusing root symlink'
p.mkdir(parents=True,exist_ok=True)
m=p/'.wip-card-owner.json'
owner=OWNER
if m.exists():
    assert json.loads(m.read_text())==owner, 'foreign WIP owner'
elif any(p.iterdir()):
    slots=p/'slots'
    assert slots.is_dir() and {d.name for d in slots.iterdir()}=={SLOT}, 'unowned slot set'
    roles=list((slots/SLOT).iterdir())
    assert roles, 'no sealed roles'
    for d in roles:
        assert d.name in ('coordinator','spark-expert') and not d.is_symlink(), 'invalid role'
        raw=(d/'META.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==(d/'FINGERPRINT').read_text().strip(), 'META seal mismatch'
        meta=json.loads(raw)
        assert (meta['slot'],meta['role'],meta['wip_instance'])==(SLOT,d.name,INSTANCE), 'slot identity mismatch'
        for relative,key in [('SOURCE_SHA256SUMS','source_manifest_sha256'),('workspace/.cuteafd-wip/ARTIFACT_SHA256SUMS','artifact_manifest_sha256')]:
            assert hashlib.sha256((d/relative).read_bytes()).hexdigest()==meta[key], 'manifest seal mismatch'
        revision_path=d/'workspace/.cuteafd-source-revision'
        revision_bytes=revision_path.read_bytes()
        assert revision_bytes.decode().strip()==REVISION, 'source revision mismatch'
        assert hashlib.sha256(revision_bytes).hexdigest()+'  ./.cuteafd-source-revision' in (d/'SOURCE_SHA256SUMS').read_text().splitlines(), 'unsealed source revision'
m.write_text(json.dumps(owner))
""".replace('ROOT_NAME', repr(Path(root).name)).replace('OWNER', repr(dict(task=task, instance=instance))).replace('SLOT', repr(slot)).replace('INSTANCE', repr(instance)).replace('REVISION', repr(revision))


def claim_root(root, task, arm, revision, seed_candidates, dry=False):
    """Write this task's owner marker before any build work touches the root: on
    raptor and, for a both-role build, every host the slot is staged to plus any
    seed that may compile it. A coordinator-only export claims raptor alone. An
    unreachable automatic seed is skipped (wip.sh would not choose it either);
    any other failure stops the build. Claiming first keeps a root that a failed
    or interrupted run leaves behind owned by this task, so a later --resume never
    has to adopt an unowned root."""
    seed_hosts = [h for h in seed_candidates if h not in HOSTS]
    hosts = ('raptor', *HOSTS, *seed_hosts) if arm['build_role'] == 'both' else ('raptor',)
    for host in hosts:
        if dry:
            continue
        marker = owner_marker_script(root, task, arm['instance'], arm['slot'], revision)
        command = ['python3', '-c', marker]
        try:
            run(command if host == 'raptor' else ['ssh', '-o', 'BatchMode=yes', host, shlex.join(command)])
        except subprocess.CalledProcessError as error:
            # An unreachable automatic seed is skipped by wip.sh too; anything else stops.
            if arm['seed_explicit'] or host not in seed_hosts or error.returncode != 255:
                raise
            print(f'# build seed {host} unreachable; wip.sh will not choose it')


def build_arms(builds, arms, cards, state, task, dry, seed=DEFAULT_SEED):
    scopes = build_scopes(cards)
    for name, revision in builds.items():
        if name not in arms:
            raise ValueError('unknown build arm ' + name)
        arm = arms[name]
        if 'release' in arm:
            raise ValueError('--build refuses release arm ' + name)
        source = Path.home() / '.cache/cuteafd/builds' / task / ('source-' + name)
        if not dry:
            source.parent.mkdir(parents=True, exist_ok=True)
        root = Path.home() / '.cache/cuteafd/builds' / ('wip-' + arm['instance'])
        arm['root'] = str(root)
        arm['build_revision'] = revision
        arm['build_lock_mode'] = build_lock_mode(revision)
        # 0-Spark cards only (#90): a coordinator-only slot, no Spark stage or seed turn.
        spark_cards = [card for card in cards if int((card['resolved_values'] if 'resolved_values' in card else {**config(card['config']), **card.get('set', {})}).get('SPARK_COUNT', len(card.get('sparks', [])))) > 0]
        build_role = 'both' if spark_cards else 'coordinator'
        arm['build_role'] = build_role
        build_cmd = ['nice', '-n', '19', str(source / 'wip.sh'), '--slot', arm['slot'], '--role', build_role, '--config', str(state / ('build-' + name + '.config'))]
        # A wip.sh that takes a build slot around Cargo itself needs no outer build.lock:
        # staging, reuse, GPU export turns and distribution then never block other builds.
        arm['build_slot'] = build_slot_capable(revision)
        if arm['build_lock_mode'] == 'legacy-full-build':
            print('# legacy arm: GPU lock held for the full build')
            # hwlock refuses to nest inside an enclosing hold (exit 77); one that covers gpu1 already holds it for us.
            if 'gpu1' not in held_by_ancestor():
                build_cmd = [sys.executable, str(REPO / 'scripts/hw/hwlock.py'), '--locks', 'gpu1', '--wait', '1800', '--', *build_cmd]
        print('# build locking ' + name + ': ' + arm['build_lock_mode'])
        # Seed on an out-of-pool host; stage the sealed slot to the whole serving union.
        # The seed is wip.sh's own parameter (WIP_SEED_HOST), never a SPARK_* rank,
        # unless this revision's wip.sh predates it.
        union = {h for card in cards for h in card.get('sparks', [])}
        arm['seed_separate'] = seed_option_capable(revision)
        requested = seed
        seed, arm['seed_explicit'], seed_candidates = resolve_seed(requested, revision, arm['seed_separate'])
        arm['seed_candidates'] = list(seed_candidates)
        arm['cleanup_hosts'] = list(dict.fromkeys(('raptor', *HOSTS, *seed_candidates, seed)))
        if not dry:
            # Preserve ownership inventory even if compiling/sealing later fails.
            save(state / 'plan.json', {'task': task, 'arms': arms, 'jobs': []})
        if build_role == 'both' and arm['seed_separate'] and arm['seed_explicit']:
            build_cmd += ['--seed', seed]
        # A coordinator-only slot stages to no Spark: SPARK_COUNT=0, no ranks named.
        hosts = staging_hosts(union, seed, arm['seed_separate']) if build_role == 'both' else []
        if build_role == 'both' and len(hosts) > 6:
            raise ValueError(f'{name}: {revision} predates WIP_SEED_HOST, so its wip.sh builds on SPARK_0_HOST and '
                             f'{seed} plus the {len(union)}-Spark union makes {len(hosts)} ranks; use --seed rhea|moa '
                             'for this arm or build a revision with the separate seed (#117)')
        if build_role == 'both':
            via = ('--seed' if arm['seed_explicit'] else 'WIP_SEED_HOST') if arm['seed_separate'] else 'legacy SPARK_0_HOST'
            shown = f"auto: {seed} or least loaded of {','.join(seed_candidates)}" if requested == 'auto' and arm['seed_separate'] else seed
            print(f"# build seed {name}: {shown} ({via}); Spark targets ({len(hosts)}): {','.join(hosts)}")
        build_config = state / ('build-' + name + '.config')
        # A staging config: the first card's model and its --set/--arm-set
        # overrides, the union of Spark hosts, and no serving topology. Export
        # scopes (TP counts, families, audio) come from every card through
        # build_scopes, never from how many hosts receive the slot.
        values = dict(cards[0]['resolved_values']) if 'resolved_values' in cards[0] else {**config(cards[0]['config']), **cards[0].get('set', {})}
        for key in list(values):
            if key in BUILD_DROP or re.fullmatch(r'SPARK_[0-7]_(HOST|LANE_A|LANE_B)', key):
                del values[key]
        # The staging container binds to the card's primary coordinator GPU
        # (default 1), so a build can stay off a GPU another session holds.
        gpu = values.get('COORDINATOR_GPUS', '1').split(',')[0]
        if gpu not in ('0', '1'):
            raise ValueError(f'staging COORDINATOR_GPUS must start with GPU 0 or 1, got {gpu}')
        values.update(SPARK_COUNT=str(len(hosts)), COORDINATOR_GPUS=gpu, RTX_GPUS='1')
        tag = 'tc-' + toolchain_hash(revision)
        values.update(COORDINATOR_DOCKER_DEV='cuteafd-coordinator-dev:' + tag, SPARK_EXPERT_DOCKER_DEV='cuteafd-spark-expert-dev:' + tag)
        arm['dev_image_tag'] = tag
        for i, host in enumerate(hosts):
            lane_a, lane_b = seed_lanes(host, dry)
            values.update({f'SPARK_{i}_HOST': host, f'SPARK_{i}_LANE_A': lane_a, f'SPARK_{i}_LANE_B': lane_b})
        build_config.write_text(''.join(f'{k}={v}\n' for k, v in sorted(values.items())))
        claim_root(root, task, arm, revision, seed_candidates, dry)
        # Only the submodules a WIP build reads, from the local module stores
        # (scripts/build/wip-source-checkout.sh, verified against the tree locks).
        checkout = REPO / 'scripts/build/wip-source-checkout.sh'
        if checkout.is_file():
            prepare = [['bash', str(checkout), str(REPO), revision, str(source)]]
        else:
            prepare = [['git', '-C', str(REPO), 'worktree', 'add', '--detach', str(source), revision],
                       ['git', '-C', str(source), 'submodule', 'update', '--init', '--recursive']]
        if not dry and (source / '.git').exists():
            # A rerun of this task (e.g. after a failed build) reuses the checkout of the same commit.
            head = run(['git', '-C', str(source), 'rev-parse', 'HEAD']).stdout.strip()
            dirty = run(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no']).stdout.strip()
            if head != revision or dirty:
                raise ValueError(f'{source} holds {head[:12]}{" (dirty)" if dirty else ""}, not {revision}; remove it or use a new --task')
            prepare = []
        commands = [
            *prepare,
            [sys.executable, str(REPO / 'scripts/build/assert-build-filesystem.py'), str(root), str(source)],
            build_cmd if arm['build_slot'] else ['flock', '-w', '1800', str(Path.home() / '.cache/cuteafd/build.lock'), *build_cmd],
        ]
        for cmd in commands:
            print(shlex.join(cmd))
            if not dry:
                if cmd is commands[-1]:
                    env = {**os.environ, **scopes, 'WIP_INSTANCE': arm['instance'], 'WIP_ROOT': str(root)}
                    env.pop('WIP_SEED_HOST', None)
                    env.pop('WIP_SEED_ADDR', None)
                    env.update(seed_env(arm, seed, dry))
                    if arm['build_slot']:
                        env['CUTEAFD_WIP_BUILD_SLOT'] = 'on'
                    if arm['build_lock_mode'] == 'native-phase':
                        env['CUTEAFD_WIP_EXPORT_LOCKS'] = 'on'
                    else:
                        env.pop('CUTEAFD_WIP_EXPORT_LOCKS', None)
                    save(state / ('build-' + name + '.json'), dict(revision=revision, lock_mode=arm['build_lock_mode'], scopes=scopes, command=cmd))
                    if run_build(cmd, env=env, cwd=source, timeout=14400):
                        raise ValueError('build failed: ' + name)
                else:
                    # Recursive submodule checkout of a fresh worktree outlasts run()'s 120 s.
                    status = run_build(cmd, timeout=1800)
                    if status:
                        raise subprocess.CalledProcessError(status, cmd)
        if dry:
            print('# identical build scopes ' + json.dumps(scopes, sort_keys=True))


def run_build(cmd, *, env=None, cwd=None, timeout):
    """Forward cancellation during build (including historical arm checkouts)."""
    if stopped_code():
        raise hwlock.Cancelled()
    supervisor = REPO / 'scripts/build/build-process.py'
    process = subprocess.Popen([sys.executable, str(supervisor), '--grace', '85',
                                '--timeout', str(timeout), '--', *cmd],
                               env=env, cwd=cwd, start_new_session=True)
    with _LIVE_LOCK:
        _LIVE[id(process)] = process
    try:
        if STOP['signal'] is not None:
            _signal_live(STOP['signal'])
        status = process.wait()
        if stopped_code():
            raise hwlock.Cancelled()
        return status
    finally:
        with _LIVE_LOCK:
            _LIVE.pop(id(process), None)


def parse_panel_config(text):
    """Keep arbitrary panel objects intact; the bench binary owns their schema."""
    panel, sep, body = text.partition('=')
    if not sep or not panel:
        raise ValueError('--panel-config needs PANEL=JSON')
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError('--panel-config JSON must be an object')
    return panel, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', default='gate',
                        help='kit directory, or "gate" (default) for the shared in-repo gate kit (scripts/bench/kits/gate)')
    parser.add_argument('--cards', nargs='+', default=[])
    parser.add_argument('--arm', action='append', default=[], metavar='NAME=INSTANCE:SLOT')
    parser.add_argument('--arm-release', action='append', default=[], metavar='NAME=INSTANCE:TAG', help='use published image pair with the same launcher, without --wip')
    parser.add_argument('--arm-wip', action='append', default=[], metavar='NAME=WIP_INSTANCE', help='reuse an external sealed WIP build; serving instances stay distinct')
    parser.add_argument('--card-arm', action='append', default=[], metavar='CARD=ARM', help='matrix mode: bind a card to one arm (useful for disjoint correctness pairs)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--interleave', action='store_true')
    mode.add_argument('--matrix', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--matched-prompts', action='store_true', help='interleave with one shared nonce sequence per pair and verified prompt token hashes')
    parser.add_argument('--nonce-seed', help='explicit shared nonce seed for every job, overriding matched-prompts auto-seeds')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--wait', type=float, default=float(os.environ.get('CUTEAFD_HW_LOCK_WAIT_S', 21600)),
                        help='maximum wait for the single serving turn, in seconds (default 21600)')
    parser.add_argument('--task', default='cards-' + time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--state', type=Path)
    parser.add_argument('--repo', type=Path, default=REPO, help='launcher checkout for arms that are neither --build nor --arm-repo')
    parser.add_argument('--arm-repo', action='append', default=[], metavar='ARM=PATH', help='launcher checkout serving this non-built arm '
                        '(a reused slot names its matching source); --build arms serve through their own source-ARM checkout')
    parser.add_argument('--binary', help='the cuteafd that runs smoke; default: cuteafd on PATH, else the arm\'s sealed '
                        'slot binary (<WIP_ROOT>/slots/<slot>/coordinator/workspace/.cuteafd-wip/cuteafd)')
    parser.add_argument('--set', action='append', default=[], metavar='CARD:KEY=VALUE')
    parser.add_argument('--arm-set', action='append', default=[], metavar='ARM:[CARD:]KEY=VALUE', help='arm config override; without CARD, applied after generation')
    parser.add_argument('--arm-unset', action='append', default=[], metavar='ARM:KEY', help='remove an arm config key after generation')
    parser.add_argument('--arm-env', action='append', default=[], metavar='ARM:CUTEAFD_DIAG_NAME=VALUE',
                        help='arm-specific private diagnostic in the launch environment (CUTEAFD_DIAG_* only; '
                        'the launcher forwards those to the coordinator unchanged)')
    parser.add_argument('--probe', action='append', default=[], metavar='[ARM:]CARD:PROBES',
                        help='image, image-prefix, image-prefix-partial, memory, console, fabric; '
                             'image-prefix-partial additionally requires a partial hit and scores fixed continuation KL/top-1 (NumPy)')
    parser.add_argument('--spark-memory-guard', action='store_true',
                        help='with --probe memory: stop only a verified card WIP worker if host MemAvailable falls below 512 MiB')
    parser.add_argument('--panels', metavar='LIST', help='run these server panels on every card instead of its profile '
                        '(comma list; "baseline" for the basic card + quick quality alone; "none" for no server profile)')
    parser.add_argument('--panel-config', action='append', default=[], metavar='PANEL=JSON',
                        help='explicit per-panel configuration; repeat for different panels (default greedy)')
    parser.add_argument('--lossless-set', type=Path, metavar='FILE', help='case file for the spec_lossless_set panel '
                        '(JSON [{id, prompt, nonce}]); default: the shipped scripts/bench/fixtures/spec-lossless-set-v1.json')
    parser.add_argument('--lossless-concurrency', type=int, metavar='N', help='spec_lossless_set cases in flight per arm (1..64)')
    parser.add_argument('--dump-lossless-rows', metavar='CASES:START-END', help='spec_lossless_set: dump full-vocabulary F32 '
                        'log-prob rows of both arms for these cases over generated tokens START..END (e.g. code-2,code-5:70-90) '
                        'to <job>/lossless-rows/spec-lossless-rows-*/ with a manifest.json; needs --panels with spec_lossless_set')
    parser.add_argument('--levels', metavar='C,C', help='client-side code-decode sweep at these concurrency levels after the profile, e.g. 1,16')
    parser.add_argument('--verify-shapes', metavar='SHAPES', help='warmed verify-only exact-shape timing instead of a decode sweep: '
                        '";"-separated per-request row shapes, e.g. "4x8;5,4x7;8x8;4x8@64" (C8 at R=32/33/64 and R=32 forced to '
                        'bucket 64); composes with PROFILE=nsys (the window frames the timed pass). See docs/wip-cards.md')
    parser.add_argument('--verify-context', type=int, metavar='N', help='--verify-shapes history tokens per request (default 1024)')
    parser.add_argument('--verify-repeats', type=int, metavar='N', help='--verify-shapes timed repeats per shape (default 20)')
    parser.add_argument('--post-load-c1', action='store_true', help='re-measure C1 after the widest --levels level')
    parser.add_argument('--sweep-passes', type=int, default=1, metavar='N',
                        help=f'run the client sweep (--levels/--prefill-lengths/--verify-shapes) and context_load N times '
                        f'against one server launch (1..{MAX_SWEEP_PASSES}, default 1); every pass is kept, the headline is the '
                        'final pass. For graph-cache convergence: pass 1 captures, later passes should not')
    parser.add_argument('--sweep-tokens', type=int, metavar='N', help='output tokens per --levels decode request '
                        '(default 192); longer runs give profiled waves more decode steps at every width, and 0 omits '
                        'max_tokens so the server output default bounds each request (KV-reservation admission probes)')
    parser.add_argument('--post-load-repeat', action='store_true', help='make that post-load C1 re-send the levels C1 prompts '
                        '(needs PREFIX_CACHE_ENTRIES=0; the sweep fails if a repeat hits the cache)')
    parser.add_argument('--context-lengths', metavar='N,N', help='shared base contexts for context_load (256..1048576; default 32768,131072)')
    parser.add_argument('--context-levels', metavar='C,C', help='context_load concurrency levels (distinct 1..256; default 1,8,16)')
    parser.add_argument('--context-suffix', type=int, metavar='N', help='additional fresh context_load suffix tokens per request (0..1048576; default 0)')
    parser.add_argument('--prefill-repeat', action='store_true', help='each prefill point\'s timed request re-sends its '
                        'warm-up\'s corpus behind a fresh nonce (nominal warm: the same mapped-table rows, no prefix-cache reuse)')
    parser.add_argument('--prefill-idle-seconds', type=float, help='idle after each ordinary prefill warm-up before the timed request (default 0)')
    parser.add_argument('--prefill-lengths', metavar='N,N', help='client-side cold prefill at these prompt lengths (one untimed warm '
                        'request each, then one measured), e.g. 8192')
    parser.add_argument('--trace', action='store_true', help='every job writes SPECULATION_TRACE to <job>/speculation-trace.jsonl; '
                        'checked after warm-up and recorded in result.json')
    parser.add_argument('--cold-window', metavar='HOSTS', help='drop the clean page cache on these comma-separated hosts '
                        '(nest drop-caches): with --prefill-lengths before every prefill point\'s warm-up and again before '
                        'its timed request (decode levels after them are post-prefill, not cold); otherwise once when the '
                        'timed sweep opens, after warm-up. Under the card\'s held locks, so only hosts the card owns; '
                        'composes with PROFILE=nsys')
    parser.add_argument('--exact-state', metavar='ARGV', help='run this exact prefix-state golden (shell-split argv; gets '
                        '--resume-url URL --output JOB/exact-state) against the live server at the end of the sweep; needs --levels')
    parser.add_argument('--client', metavar='ARGV', help='run this client workload (shell-split argv) against the live server once '
                        'the card is ready, inside its turn, in JOB/client with CUTEAFD_SMOKE_URL, CUTEAFD_API_KEY_FILE and '
                        'CUTEAFD_CLIENT_OUT exported; a failing client fails the card. Needs --panels none (no sweep); '
                        'excludes --exact-state and --cold-window')
    parser.add_argument('--client-timeout', type=int, default=CLIENT_TIMEOUT_S, metavar='SECONDS',
                        help=f'--client limit in seconds (1..7200, default {CLIENT_TIMEOUT_S})')
    parser.add_argument('--resume', action='store_true', help='reuse --state (or --task): keep passed jobs, rerun failed or missing '
                        'ones; a rerun job keeps its earlier attempt under attempt-N/')
    parser.add_argument('--preflight-only', action='store_true', help='run every CPU check (slots, packages, audio, containers, '
                        'binary, trace dirs), write the plan and exit; for hwlock --preflight')
    parser.add_argument('--expect-pool', action='append', default=[], metavar='CARD=N')
    parser.add_argument('--run-timeout', action='append', default=[], metavar='CARD=SECONDS',
                        help="override a card's kit run_timeout_s (the timed sweep's limit), e.g. for a long-context panel")
    parser.add_argument('--build', action='append', default=[], metavar='ARM=REV')
    parser.add_argument('--build-only', action='store_true', help='build and seal selected arms without generating or launching smoke jobs')
    parser.add_argument('--seed', '--seed-host', dest='seed_host', default=DEFAULT_SEED, choices=SEEDS,
                        help=f'out-of-serving Spark that compiles and seals --build arms (default {DEFAULT_SEED}: the '
                        'less-loaded LAN-only seed, kakapo or cassowary, then rhea/moa). The LAN-only seeds are 10 GbE '
                        'build-only hosts whose slots reach the serving Sparks through sparknest; a named seed is a placement request')
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--print-locks', action='store_true',
                        help='print the serving turn lock set this run would take and exit, without building or running')
    parser.add_argument('--observe', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--precheck', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--preflight-job', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--census-final-sample', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--spark-memory-boundary', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.spark_memory_boundary:
        spark_memory_boundary(json.loads(args.spark_memory_boundary.read_text()), 'post-sweep')
        return 0
    if args.census_final_sample:
        census_final_sample(json.loads(args.census_final_sample.read_text()))
        return 0
    if not 0 < args.wait < float('inf'):
        parser.error('--wait must be finite and positive')
    if args.observe or args.precheck:
        job = json.loads((args.observe or args.precheck).read_text())
        (observe if args.observe else precheck)(job)
        return 0
    if args.preflight_job:
        # Smoke's prelaunch hook: the same checks again, now that smoke holds the locks.
        job = json.loads(args.preflight_job.read_text())
        plan = json.loads((args.preflight_job.resolve().parent.parent / 'plan.json').read_text())
        artifacts = preflight_at_grant(job, plan['arms'][job['arm']])
        if job.get('artifacts') and artifacts != job['artifacts']:
            raise ValueError('sealed artifacts changed since enqueue: ' + json.dumps(artifacts, sort_keys=True))
        print('preflight at grant: pass')
        return 0
    if not NAME.fullmatch(args.task) or args.parallel < 1 or args.repeats < 1:
        parser.error('invalid task, repeats or parallelism')
    if args.build_only and (not args.build or args.cleanup):
        parser.error('--build-only requires --build and refuses --cleanup')
    arms = arms_from(args.arm, args.arm_release)
    # Pin each --build ref once, before any build or launcher check reads it again.
    builds = resolve_builds(args.build)
    shared_wip(arms, args.arm_wip, builds, args.parallel)
    if args.nonce_seed is not None and (not args.nonce_seed or '\n' in args.nonce_seed or '\r' in args.nonce_seed):
        parser.error('--nonce-seed must be non-empty and single-line')
    if not arms:
        parser.error('at least one --arm or --arm-release is required')
    if args.matched_prompts and not args.interleave:
        parser.error('--matched-prompts requires --interleave')
    if args.interleave and (len(arms) < 2 or args.parallel != 1):
        parser.error('interleave needs two or more arms and --parallel 1')
    if len(arms) > 1 and not (args.interleave or args.matrix or args.cleanup or args.build_only):
        parser.error('several arms need --interleave (an A/B: alternating pairs) or --matrix (every card on every arm, once)')
    if args.resume and args.cleanup:
        parser.error('--resume refuses --cleanup')
    state = (args.state or Path.home() / '.cache/cuteafd/builds' / args.task).expanduser().resolve()
    if not (args.resume or args.cleanup or args.build_only) and (state / 'plan.json').exists() and \
            any(state.glob('*/exit')):
        if args.state is None and '--task' not in sys.argv:
            # The default task name is time-stamped; a same-second collision gets a suffix.
            for n in itertools.count(2):
                candidate = state.with_name(f'{state.name}-{n}')
                if not candidate.exists():
                    state, args.task = candidate, candidate.name
                    break
        else:
            raise ValueError(f'{state} already holds a run; pass --resume to rerun its failed or missing jobs, '
                             'or choose a fresh --task/--state')
    state.mkdir(parents=True, exist_ok=True)
    if args.cleanup:
        plan = state / 'plan.json'
        if plan.exists():
            recorded = json.loads(plan.read_text())
            if recorded['task'] != args.task or set(recorded['arms']) != set(arms) or any(recorded['arms'][name]['instance'] != arm['instance'] or recorded['arms'][name]['slot'] != arm['slot'] or recorded['arms'][name].get('release') != arm.get('release') for name, arm in arms.items()):
                raise ValueError('cleanup does not match this task plan')
            arms = recorded['arms']
        cleanup(arms, args.task, args.dry_run, args.seed_host)
        return 0
    if not args.cards or not (args.matrix or args.interleave):
        parser.error('--cards and --matrix or --interleave required')
    kit = REPO / 'scripts/bench/kits' / args.kit if NAME.fullmatch(args.kit) and (REPO / 'scripts/bench/kits' / args.kit).is_dir() else Path(args.kit)
    cards = load_cards(kit.expanduser().resolve(), args.cards)
    overrides = overrides_from(args.set)
    arm_overrides = {}
    arm_config = {}
    arm_unsets = {}
    for item in args.arm_unset:
        name, key = item.split(':', 1)
        if name not in arms:
            raise ValueError('unknown override arm: ' + name)
        overrides_from(['config:' + key + '='])
        arm_unsets.setdefault(name, set()).add(key)
    for item in args.arm_set:
        name, setting = item.split(':', 1)
        if name not in arms:
            raise ValueError('unknown override arm: ' + name)
        if ':' not in setting.split('=', 1)[0]:
            arm_config.setdefault(name, {}).update(overrides_from(['config:' + setting])['config'])
            continue
        parsed = overrides_from([setting])
        if set(parsed) - set(args.cards):
            raise ValueError('arm override names must be selected cards')
        for card, values in parsed.items():
            arm_overrides.setdefault(name, {}).setdefault(card, {}).update(values)
    arm_env = arm_env_from(args.arm_env, arms)
    if args.post_load_c1 and not args.levels:
        parser.error('--post-load-c1 needs --levels')
    if args.post_load_repeat and not args.post_load_c1:
        parser.error('--post-load-repeat needs --post-load-c1')
    if args.prefill_repeat and not args.prefill_lengths:
        parser.error('--prefill-repeat needs --prefill-lengths')
    if args.prefill_idle_seconds is not None and (not args.prefill_lengths or not math.isfinite(args.prefill_idle_seconds)
                                                or not 0 <= args.prefill_idle_seconds < 2**64):
        parser.error('--prefill-idle-seconds needs --prefill-lengths and finite nonnegative representable seconds')
    if args.sweep_tokens is not None and not (args.levels and 0 <= args.sweep_tokens <= 8192):
        parser.error('--sweep-tokens needs --levels and 0 (omit max_tokens) or 1..8192 tokens')
    if not 1 <= args.sweep_passes <= MAX_SWEEP_PASSES:
        parser.error(f'--sweep-passes takes 1..{MAX_SWEEP_PASSES} same-server passes')
    levels = None
    if args.levels:
        levels = [int(x) for x in args.levels.split(',') if x.strip()]
        if not levels or len(set(levels)) != len(levels) or not all(1 <= x <= 256 for x in levels):
            parser.error('--levels takes distinct concurrency levels in 1..256, e.g. 1,16')
    prefill_lengths = None
    if args.prefill_lengths:
        prefill_lengths = [int(x) for x in args.prefill_lengths.split(',') if x.strip().isdigit()]
        if not prefill_lengths or len(prefill_lengths) != len([x for x in args.prefill_lengths.split(',') if x.strip()]) \
                or not all(256 <= x <= 1 << 20 for x in prefill_lengths):
            parser.error('--prefill-lengths takes prompt lengths in 256..1048576, e.g. 8192')
    context_lengths = None
    if args.context_lengths is not None:
        parts = [x.strip() for x in args.context_lengths.split(',') if x.strip()]
        if not parts or not all(x.isdigit() and 256 <= int(x) <= 1 << 20 for x in parts):
            parser.error('--context-lengths takes prompt lengths in 256..1048576, e.g. 32768,131072')
        context_lengths = [int(x) for x in parts]
    if args.context_suffix is not None and not 0 <= args.context_suffix <= 1 << 20:
        parser.error('--context-suffix takes tokens in 0..1048576')
    context_levels = None
    if args.context_levels is not None:
        parts = [x.strip() for x in args.context_levels.split(',') if x.strip()]
        if not parts or not all(x.isdigit() and 1 <= int(x) <= 256 for x in parts) or len(set(map(int, parts))) != len(parts):
            parser.error('--context-levels takes distinct concurrency levels in 1..256, e.g. 1,8,16')
        context_levels = [int(x) for x in parts]
    verify = None
    if (args.verify_context is not None or args.verify_repeats is not None) and not args.verify_shapes:
        parser.error('--verify-context and --verify-repeats need --verify-shapes')
    if args.verify_shapes:
        if args.levels or args.prefill_lengths:
            parser.error('--verify-shapes runs alone: drop --levels and --prefill-lengths')
        shapes = [shape.strip() for shape in args.verify_shapes.split(';') if shape.strip()]
        if not shapes or not all(VERIFY_SHAPE.fullmatch(shape) for shape in shapes):
            parser.error('--verify-shapes takes ";"-separated shapes like 4x8;5,4x7;4x8@64')
        verify = dict(shapes=shapes)
        if args.verify_context is not None:
            verify['context'] = args.verify_context
        if args.verify_repeats is not None:
            verify['repeats'] = args.verify_repeats
    if args.client and (args.panels is None or args.panels.strip() != 'none'):
        parser.error('--client needs --panels none')
    if args.panels is not None:
        selection = '' if args.panels.strip() == 'none' else args.panels.strip()
        if not selection and not (levels or prefill_lengths or verify or args.probe or args.client):
            parser.error('--panels none runs nothing without --levels, --prefill-lengths, --verify-shapes, --probe or --client')
        profile = 'panels:' + selection
        listed = requested_panels(profile, binary=panel_binary(args.binary, arms))
        for card in cards:
            card['profile'] = profile
            if (listed or not (levels or prefill_lengths or verify)) and 'baseline' not in listed:
                card['baseline'] = False  # Explicit panels alone, or a probe-only readiness run; list baseline to keep it.
    try:
        panel_config = dict(parse_panel_config(text) for text in args.panel_config)
    except ValueError as error:
        parser.error(str(error))
    lossless_set = None
    if args.lossless_concurrency is not None:
        if not 1 <= args.lossless_concurrency <= 64:
            parser.error('--lossless-concurrency must be 1-64')
        if not any('spec_lossless_set' in requested_panels(card.get('profile', 'smoke'), binary=panel_binary(args.binary, arms)) for card in cards):
            parser.error('--lossless-concurrency needs --panels with spec_lossless_set')
    if args.lossless_set is not None:
        lossless_set = load_lossless_set(args.lossless_set)
        if not any('spec_lossless_set' in requested_panels(card.get('profile', 'smoke'), binary=panel_binary(args.binary, arms)) for card in cards):
            parser.error('--lossless-set needs --panels with spec_lossless_set')
    lossless_dump = None
    if args.dump_lossless_rows is not None:
        try:
            lossless_dump = parse_lossless_dump(args.dump_lossless_rows, lossless_set)
        except ValueError as error:
            parser.error(f'--dump-lossless-rows: {error}')
        if not any('spec_lossless_set' in requested_panels(card.get('profile', 'smoke'), binary=panel_binary(args.binary, arms)) for card in cards):
            parser.error('--dump-lossless-rows needs --panels with spec_lossless_set')
    for card in cards:
        if panel_config:
            card['panel_config'] = {**card.get('panel_config', {}), **panel_config}
        if context_lengths is not None:
            card['context_lengths'] = context_lengths
        if context_levels is not None:
            card['context_levels'] = context_levels
        if args.context_suffix is not None:
            card['context_suffix'] = args.context_suffix
        if lossless_set:
            card['lossless_set'] = lossless_set
        if args.lossless_concurrency is not None:
            card['lossless_concurrency'] = args.lossless_concurrency
        if lossless_dump:
            card['lossless_dump'] = lossless_dump
        if prefill_lengths:
            card['prefill_lengths'] = prefill_lengths
            if args.prefill_idle_seconds is not None:
                card['prefill_idle_seconds'] = args.prefill_idle_seconds
            if args.prefill_repeat:
                card['prefill_repeat'] = True
        if verify:
            card['sweep_verify'] = verify
        if levels:
            card['levels'] = levels
            if args.sweep_tokens is not None:
                card['sweep_tokens'] = args.sweep_tokens
            if args.post_load_c1:
                card['post_load_c1'] = True
                if args.post_load_repeat:
                    card['post_load_repeat'] = True
    if args.sweep_passes > 1:
        for card in cards:
            if not (sweeping(card) or runs_context_load(card.get('profile', 'smoke'))):
                parser.error('--sweep-passes repeats the client sweep and context_load: pass --levels, --prefill-lengths, '
                             '--verify-shapes or --panels with context_load')
            card['sweep_passes'] = args.sweep_passes
    probes, arm_probes = probe_selection(args.probe, cards, arms)
    expected = {card['name']: card.get('expected_pool') for card in cards}
    for item in args.expect_pool:
        name, value = item.split('=', 1)
        expected[name] = int(value)
    run_timeouts = {}
    for item in args.run_timeout:
        name, value = item.split('=', 1)
        if not value.isdigit() or int(value) < 1:
            raise ValueError('--run-timeout needs CARD=SECONDS with a positive integer')
        run_timeouts[name] = int(value)
    if (set(overrides) | set(probes) | set(expected) | set(run_timeouts)) - set(args.cards):
        raise ValueError('override/probe names must be selected cards')
    for card in cards:
        if args.matched_prompts or args.nonce_seed is not None:
            if not scoped_panel_window(card) and args.matched_prompts and any(
                    (probes[card['name']] | arm_probes.get(arm, {}).get(card['name'], set()))
                    & IMAGE_PROBES for arm in arms):
                raise ValueError('--matched-prompts cannot precede its first benchmark request with an image probe')
        requested_panels(card.get('profile', 'smoke'), binary=panel_binary(args.binary, arms))
        card['set'].update(overrides.get(card['name'], {}))
        if card['name'] in run_timeouts:
            card['run_timeout_s'] = run_timeouts[card['name']]
        values = {**config(card['config']), **card['set']}
        count = int(values.get('SPARK_COUNT', len(card.get('sparks', []))))
        if not 0 <= count <= len(HOSTS):
            raise ValueError('invalid Spark count')
        card['sparks'] = [values.get(f'SPARK_{i}_HOST', card.get('sparks', [])[i] if i < len(card.get('sparks', [])) else HOSTS[i]) for i in range(count)]
    bindings = dict(item.split('=', 1) for item in args.card_arm)
    if bindings and (args.interleave or set(bindings) - set(args.cards) or set(bindings.values()) - set(arms)):
        raise ValueError('--card-arm needs selected cards/arms in matrix mode')
    if args.print_locks:
        layouts = []
        for name in arms:
            for card in cards:
                if card['name'] in bindings and bindings[card['name']] != name:
                    continue
                layouts.append(card_layout(resolved_card(card, {**arm_overrides.get(name, {}).get(card['name'], {}),
                    **arm_config.get(name, {})}, arm_unsets.get(name, set()))))
        print(','.join(layout_locks(layouts)))
        return 0
    # Build and stage every arm's layout, even when only one arm requests it.
    build_cards = []
    for name in arms:
        for card in cards:
            variant = resolved_card(card, {**arm_overrides.get(name, {}).get(card['name'], {}),
                **arm_config.get(name, {})}, arm_unsets.get(name, set()))
            build_cards.append(variant)
    if not args.build_only:
        check_launcher_submodules(arms, args.arm_repo, builds, args.repo)
    install_stop_handlers()
    try:
        build_arms(builds, arms, build_cards, state, args.task, args.dry_run, args.seed_host)
    except hwlock.Cancelled:
        return stopped_code() or 1
    if stopped_code():
        return stopped_code()
    adopt_roots(arms)
    if args.build_only:
        save(state / 'plan.json', {'task': args.task, 'arms': arms, 'jobs': []})
        return 0
    resolve_launchers(arms, args.arm_repo, builds, args.repo, state, args.task, args.dry_run)
    if args.trace:
        for name, arm in arms.items():
            if args.dry_run and name in builds:
                continue
            for card in cards:
                if card.get('family') == 'deepseek_v41' and not launcher_forwards_v41_trace(Path(arm['repo'])):
                    raise ValueError(f"{card['name']}: arm {name}'s launcher {arm['repo']} does not forward SPECULATION_TRACE on the V4.1 path")
    key_file = state / 'api-key'
    if not key_file.exists():
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as key:
            key.write(secrets.token_hex(32) + '\n')
    if key_file.is_symlink() or key_file.stat().st_mode & 0o077:
        raise ValueError('API key file must be private (0600), not a symlink')
    jobs = [generate(resolved_card(card, {**arm_overrides.get(name, {}).get(card['name'], {}), **arm_config.get(name, {})}, arm_unsets.get(name, set())), name, arm, state, key_file, {}, probes[card['name']] | arm_probes.get(name, {}).get(card['name'], set()), expected[card['name']], repeat, args.trace, shlex.split(args.exact_state) if args.exact_state else None, [h for h in (args.cold_window or '').split(',') if h] or None, binary=panel_binary(args.binary, arms), client=shlex.split(args.client) if args.client else None, client_timeout=args.client_timeout) for card, name, arm, repeat in schedule(cards, arms, args.repeats if args.interleave else 1, args.interleave) if card['name'] not in bindings or bindings[card['name']] == name]
    variants = {(c['name'], n): resolved_card(c, {**arm_overrides.get(n, {}).get(c['name'], {}),
        **arm_config.get(n, {})}, arm_unsets.get(n, set())) for c in cards for n in arms}
    for job in jobs:
        apply_arm_config(job, arm_config.get(job['arm'], {}), arm_unsets.get(job['arm'], set()))
        if args.spark_memory_guard:
            if 'memory' not in job['probes'] or '--wip' not in job['entry']['run_args'] or not job['entry']['sparks']:
                raise ValueError('--spark-memory-guard requires --probe memory on Spark-backed WIP cards')
            job['memory_guard_bytes'] = 512 * 1024 * 1024
        job['launcher'] = dict(repo=arms[job['arm']]['repo'], commit=arms[job['arm']]['repo_commit'])
        job['needs'] = slot_needs(variants[(job['card'], job['arm'])])
        # Per arm: an A/B of two builds runs each arm's own smoke runner.
        job['binary'] = resolve_binary(args.binary, arms[job['arm']], args.dry_run)
        if arm_env.get(job['arm']):
            job['arm_env'] = arm_env[job['arm']]
        if (args.matched_prompts or args.nonce_seed is not None) and scoped_panel_window(job['entry']):
            job['entry']['matched_panel_window'] = True
            save(Path(job['state']) / 'matrix.json', dict(build='wip-cards', entries=[job['entry']]))
        save(Path(job['state']) / 'job.json', {**job, 'probes': sorted(job['probes'])})
    parallel_safe(jobs, args.parallel)
    for job in jobs:
        if args.nonce_seed is not None or args.matched_prompts:
            job['nonce_seed'] = args.nonce_seed if args.nonce_seed is not None else hashlib.sha256(f"{args.task}:{job['card']}:{job['repeat']}".encode()).hexdigest()[:16]
            save(Path(job['state']) / 'job.json', {**job, 'probes': sorted(job['probes'])})
        job['correctness_parallel'] = args.parallel > 1
        if args.parallel > 1:
            job['entry']['exclusive'] = False
            save(Path(job['state']) / 'matrix.json', dict(build='wip-cards', profile=job['entry'].get('profile', 'smoke'), entries=[job['entry']]))
    groups = smoke_groups(jobs, args.parallel)
    if groups and len(groups[0]) > 1:
        group_root = state / 'parallel-smoke'
        group_root.mkdir(exist_ok=True)
        for job in jobs:
            job['grouped'] = True
            dest = Path(job['state'])
            for name in ('smoke', 'reports'):
                (dest / name).symlink_to(group_root / name, target_is_directory=True)
        save(group_root / 'matrix.json', dict(build='wip-cards', entries=[job['entry'] for job in jobs]))
    save(state / 'plan.json', {'task': args.task, 'arms': arms, 'jobs': [{**j, 'probes': sorted(j['probes'])} for j in jobs]})
    if args.dry_run:
        print('# single serving turn after all builds/seals: hwlock --locks ' +
              ','.join(serving_locks(jobs)) + ' --wait ' + str(args.wait) +
              ' --max-run ' + str(round(sum(job_max_run(job) for job in jobs))) +
              '; held through all arms/repeats and final teardown')
        if groups and len(groups[0]) > 1:
            cmd = smoke_command(jobs[0]['binary'], arms[jobs[0]['arm']]['repo'], dict(state=str(state / 'parallel-smoke')))
            cmd[-1] = str(args.parallel)
            prefix = ('CUTEAFD_BENCH_NONCE_SEED=' + shlex.quote(args.nonce_seed) + ' ') if args.nonce_seed is not None else ''
            print(prefix + 'setsid ' + shlex.join(cmd) + ' # grouped smoke scheduler; numeric exits per card')
            return 0
        for job in jobs:
            cmd = smoke_command(job['binary'], job['launcher']['repo'], job)
            prefix = ('CUTEAFD_BENCH_NONCE_SEED=' + shlex.quote(job['nonce_seed']) + ' ') if job.get('nonce_seed') else ''
            print('# settings ' + job['arm'] + '/' + job['card'] + ': ' + json.dumps(job['entry']['set'], sort_keys=True))
            if args.context_suffix is not None:
                print('# context_suffix=' + str(job['entry']['context_suffix']))
            if args.sweep_passes > 1:
                print(f"# sweep_passes={args.sweep_passes}: one server launch, {args.sweep_passes} passes of the sweep and context_load")
            print(prefix + 'setsid ' + shlex.join(cmd) + ' # numeric exit -> ' + job['state'] + '/exit')
            if job.get('client'):
                print('# smoke client (after readiness, in the card\'s turn): ' + shlex.join(job['client']))
            for kind in ('observer', 'precheck'):
                if kind in job['entry']:
                    print('# smoke ' + kind + ': ' + shlex.join(job['entry'][kind]['argv']))
        return 0
    kept = {}
    if args.resume:
        for job in jobs:
            dest = Path(job['state'])
            result = dest / 'result.json'
            if result.exists():
                previous = json.loads(result.read_text())
                if previous.get('status') == 'pass' and (previous.get('precheck') or {}).get('speculator') is not None:
                    kept[job['state']] = previous
                    continue
            leftovers = [p for p in dest.iterdir() if p.name in ('exit', 'result.json', 'driver.log', 'smoke', 'reports',
                         'trace-check.json', 'speculation-trace.jsonl', 'precheck.json', 'console.jsonl',
                         'console-frames.jsonl', 'policy-stats.jsonl', 'memory.jsonl', 'ready.log', 'profile')]
            if leftovers and not job.get('grouped'):
                attempt = next(dest / f'attempt-{n}' for n in itertools.count(1) if not (dest / f'attempt-{n}').exists())
                attempt.mkdir()
                for path in leftovers:
                    path.rename(attempt / path.name)
                if job['entry']['set'].get('PROFILE_DIR'):
                    Path(job['entry']['set']['PROFILE_DIR']).mkdir(exist_ok=True)
                    if 'memory' in job['probes']:
                        save(Path(job['entry']['set']['PROFILE_DIR']) / 'observer-context.json', {'pending': True})
        if kept:
            print(f'wip-cards: resume keeps {len(kept)} passed job(s), reruns {len(jobs) - len(kept)}', file=sys.stderr)
    fields = ['fabric', 'warm_prelude'] + (['levels', 'post_load_c1'] + (['post_load_repeat'] if args.post_load_repeat else []) if args.levels else []) + (['prefill_lengths'] if prefill_lengths else []) + \
        (['prefill_repeat'] if args.prefill_repeat else []) + \
        (['prefill_idle_seconds'] if args.prefill_idle_seconds is not None else []) + \
        (['prefill_purge'] if any(job['entry'].get('prefill_purge') for job in jobs) else []) + \
        (['sweep_tokens'] if args.sweep_tokens is not None else []) + \
        (['baseline'] if any(job['entry'].get('baseline') is False for job in jobs) else []) + \
        (['window'] if any(job['entry'].get('window') for job in jobs) else []) + \
        (['lossless_set'] if any(job['entry'].get('lossless_set') for job in jobs) else []) + \
        (['lossless_concurrency'] if args.lossless_concurrency is not None else []) + \
        (['lossless_dump'] if any(job['entry'].get('lossless_dump') for job in jobs) else []) + \
        (['context_lengths'] if context_lengths is not None else []) + (['context_levels'] if context_levels is not None else []) + \
        (['context_suffix'] if args.context_suffix is not None else []) + \
        (['panel_config'] if panel_config else []) + (['sweep_verify'] if verify else []) + \
        (['worker_stop'] if any(job['entry']['set'].get('PROFILE_SPARK') for job in jobs) else []) + \
        (['sweep_passes'] if args.sweep_passes > 1 else [])
    for job in {(j['launcher']['repo'], bool(j['entry'].get('matched_panel_window'))): j for j in reversed(jobs)}.values():
        requested_panels(job['entry'].get('profile', 'smoke'), binary=job['binary'])
        binary_supports(job['binary'], job['launcher']['repo'], job,
                        fields + (['matched_panel_window'] if job['entry'].get('matched_panel_window') else []))
    # Everything checkable without hardware, before any queue ticket.
    problems = []
    for job in jobs:
        if job['state'] in kept:
            continue
        try:
            job['artifacts'] = preflight_job(job, arms[job['arm']])
            save(Path(job['state']) / 'job.json', {**job, 'probes': sorted(job['probes'])})
        except (ValueError, subprocess.SubprocessError) as error:
            problems.append(f"{job['card']}/{job['arm']}/r{job['repeat']}: {error}")
    if problems:
        save(state / 'preflight.json', {'status': 'failed', 'problems': problems})
        raise ValueError('preflight failed before queuing:\n  ' + '\n  '.join(problems))
    save(state / 'preflight.json', {'status': 'pass', 'jobs': len(jobs) - len(kept)})
    if args.preflight_only:
        print(f'wip-cards: preflight pass for {len(jobs) - len(kept)} job(s); plan {state / "plan.json"}')
        return 0
    env = {**os.environ, 'CUTEAFD_API_KEY': key_file.read_text().strip()}
    env.pop('CUTEAFD_BENCH_NONCE_SEED', None)
    if args.nonce_seed is not None:
        env['CUTEAFD_BENCH_NONCE_SEED'] = args.nonce_seed
    def cancelled_row(job):
        row = {k: job[k] for k in ('card', 'arm', 'repeat', 'simulated')}
        row.update(status='failed', exit_code=stopped_code(), error=f"stopped by signal {STOP['signal']} before it started",
                   failure_kind='cancelled', state=job['state'])
        return row

    def execute(job):
        dest = Path(job['state'])
        if job['state'] in kept:
            return kept[job['state']]
        if STOP['signal'] is not None:
            return cancelled_row(job)
        try:
            # Re-checked right before this job's turn: an earlier job may have
            # left a container, or a slot may have been rebuilt meanwhile.
            artifacts = preflight_job(job, arms[job['arm']])
            if artifacts != job.get('artifacts', artifacts):
                raise ValueError('sealed artifacts changed since preflight: ' + json.dumps(artifacts, sort_keys=True))
        except (ValueError, subprocess.SubprocessError) as error:
            row = {k: job[k] for k in ('card', 'arm', 'repeat', 'simulated')}
            row.update(status='failed', exit_code=82, error=str(error), failure_kind='preflight', state=str(dest))
            (dest / 'exit').write_text('82\n')
            save(dest / 'result.json', row)
            return row
        exit_file = dest / 'exit'
        if exit_file.exists():
            raise ValueError('run already exists; use --resume or a fresh task/state: ' + str(dest))
        job_env = dict(env)
        job_env.pop('CUTEAFD_BENCH_NONCE_SEED', None)
        if job.get('nonce_seed'):
            job_env['CUTEAFD_BENCH_NONCE_SEED'] = job['nonce_seed']
        for key in [k for k in job_env if k.startswith('CUTEAFD_DIAG_')]:
            del job_env[key]
        job_env.update(job.get('arm_env', {}))
        if STOP['signal'] is not None:
            return cancelled_row(job)
        process = detached(smoke_command(job['binary'], job['launcher']['repo'], job), dest / 'driver.log', exit_file, job_env)
        rc = wait_detached(process)
        # Cancellation already uses smoke's bounded teardown; remote probes and
        # profile collection must not extend the 60/75/80/90 s stop chain.
        if STOP['signal'] is not None:
            row = cancelled_row(job)
            row['error'] = f"stopped by signal {STOP['signal']}; smoke exited"
            exit_file.write_text(str(stopped_code()) + '\n')
            save(dest / 'result.json', row)
            return row
        assert_absent(job)
        code = int(exit_file.read_text()) if exit_file.exists() else (rc if rc >= 0 else 128 - rc)
        exit_file.write_text(str(code) + '\n')
        row = summarize_job(job, code)
        row['state'] = str(dest)
        if row['status'] != 'pass' and not code:
            exit_file.write_text('81\n')
            row['exit_code'] = 81
            save(dest / 'result.json', row)
        return row
    try:
        # build_arms (including legacy/export turns) completed before job creation;
        # sealed-slot preflight rejects missing artifacts instead of building lazily.
        with serving_turn([job for job in jobs if job['state'] not in kept], env, args.wait):
            if STOP['signal'] is not None:
                return stopped_code()
            if groups and len(groups[0]) > 1:
                group_root = state / 'parallel-smoke'
                try:
                    for job in jobs:
                        preflight_job(job, arms[job['arm']])
                except (ValueError, subprocess.SubprocessError) as error:
                    rows = []
                    for job in jobs:
                        row = {k: job[k] for k in ('card', 'arm', 'repeat', 'simulated')}
                        row.update(status='failed', exit_code=82, error=str(error))
                        (Path(job['state']) / 'exit').write_text('82\n')
                        save(Path(job['state']) / 'result.json', row)
                        rows.append(row)
                    write_summary(state, rows, arms)
                    return 1
                command = smoke_command(jobs[0]['binary'], arms[jobs[0]['arm']]['repo'], dict(state=str(group_root)))
                command[-1] = str(args.parallel)
                if STOP['signal'] is not None:
                    return stopped_code()
                process = detached(command, group_root / 'driver.log', group_root / 'exit', env)
                rc = wait_detached(process)
                if STOP['signal'] is not None:
                    for job in jobs:
                        (Path(job['state']) / 'exit').write_text(str(stopped_code()) + '\n')
                        save(Path(job['state']) / 'result.json', cancelled_row(job))
                    return stopped_code()
                for job in jobs:
                    assert_absent(job)
                rows = []
                for job in jobs:
                    row = summarize_job(job, rc)
                    # A different entry's failure does not erase this entry's evidence.
                    outcomes = row.get('smoke', {})
                    if isinstance(outcomes, dict) and outcomes.get(job['entry']['name'], {}).get('status') == 'done':
                        row = summarize_job(job, 0)
                    code = 0 if row['status'] == 'pass' else (rc or 81)
                    (Path(job['state']) / 'exit').write_text(str(code) + '\n')
                    row['exit_code'] = code
                    rows.append(row)
                write_summary(state, rows, arms)
                return stopped_code() or int(any(row['status'] != 'pass' for row in rows))
            rows = []
            with contextlib.ExitStack() as stack:
                # Sequential comparisons stop immediately if teardown left containers.
                if args.parallel > 1:
                    executor = stack.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel))
                    outcomes = executor.map(execute, jobs)
                else:
                    outcomes = map(execute, jobs)
                for row in outcomes:
                    rows.append(row)
                    write_summary(state, rows, arms)
            return stopped_code() or int(any(r['status'] != 'pass' for r in rows))
    except hwlock.Cancelled:
        return stopped_code() or 1


def cli():
    try:
        code = main()
    except (ValueError, RuntimeError, subprocess.SubprocessError, OSError) as error:
        print('wip-cards: ' + str(error), file=sys.stderr)
        return 2
    if hwlock.OVERRAN[0]:
        print("wip-cards: the serving turn's --max-run bound was reached; smokes stopped and torn down", file=sys.stderr)
        return 124
    return code


if __name__ == '__main__':
    raise SystemExit(cli())
