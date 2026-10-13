#!/usr/bin/env python3
"""Task-private WIP cards. Smoke alone owns serving locks; dry-run is CPU-only."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import itertools
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
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
# PROFILE switches that wrap the coordinator in a profiler, and their report suffix.
PROFILER_REPORTS = {'nsys': '.nsys-rep', 'ncu': '.ncu-rep'}
NSYS_SESSION = 'cuteafd-profile'  # RELEASE_NSYS_SESSION in scripts/lib/release-common.sh
HOSTS = ('ostrich', 'dodo', 'emu', 'kiwi', 'rhea', 'moa')
# Out-of-pool build seeds: compile, export and seal Spark slots, never serve.
# kakapo has no RoCE (10 GbE only): its staging lanes are its LAN address and
# wip.sh moves its sealed slots through sparknest.
SEED_HOSTS = ('kakapo',)
SEEDS = ('rhea', 'moa', *SEED_HOSTS)
# kakapo passed its gate 2026-10-13 (mimo26-flash-max: cold build sealed, distributed
# through sparknest, served at C1 with 0 errors); it isolates seed CPU load from serving.
DEFAULT_SEED = 'kakapo'
CORRECTNESS = {'sim-5090', 'fidelity', 'cache', 'image', 'no-fit', 'packed-check'}
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
METRICS = ('C1', 'C8', '8K', 'kl', 'top1', 'pool', 'readiness_s')
# Before -> after table: baseline card metrics, then the client sweep's levels
# (sweep C<n>) and post-load C1, which are separate measurements.
SUMMARY_METRICS = ('C1', 'C8', 'sweep C1', 'sweep C4', 'sweep C8', 'sweep C16', 'post-load C1', '8K', 'kl', 'top1', 'pool', 'readiness_s')
LOWER_IS_BETTER = {'kl', 'readiness_s'}  # also every 'prefill N TTFT s'
FAILURE = re.compile(r'(?i)(\berror\b|\bpanicked\b|\bfatal\b|\bfailed\b|undefined symbol|no such file|refus|exited:|timed out|out of memory|traceback)')
PANELS = {'baseline', 'hardware', 'configuration', 'decode_content', 'concurrency', 'draft_mix', 'prefill', 'retained', 'prefix_cache', 'startup', 'agentic', 'tool_eval', 'structured', 'ifeval', 'code', 'math', 'needle', 'fidelity', 'fidelity_full', 'reasoning_effort', 'spec_lossless_set'}


def short_instance(value):
    if not NAME.fullmatch(value):
        raise ValueError('invalid instance: ' + value)
    return value if len(value) <= 41 else value[:32] + '-' + hashlib.sha256(value.encode()).hexdigest()[:8]


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


def requested_panels(profile):
    if not profile.startswith('panels:'):
        return []
    panels = [p.strip() for p in profile[7:].split(',') if p.strip()]
    bad = set(panels) - PANELS - {'baseline'}
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
    return subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=120, **kwargs)


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
    slots = {}
    for arm in arms.values():
        if 'release' in arm:
            continue
        instance = arm.get('wip_instance', arm['instance'])
        if instance in slots and slots[instance] != arm['slot']:
            raise ValueError('shared WIP instance requires identical slots')
        slots[instance] = arm['slot']


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


def generate(entry, arm_name, arm, state, key_file, overrides, probes, expected_pool, repeat, trace=False):
    entry = json.loads(json.dumps(entry))
    requested_panels(entry.get('profile', 'smoke'))
    values = config(entry['config'])
    values.update(entry.pop('set', {}))
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
    gpus = [int(x) for x in values.get('COORDINATOR_GPUS', '').split(',') if x]
    if not gpus:
        first = int(values.get('COORDINATOR_GPU', entry.get('gpus', [0])[0]))
        gpus = [first] if int(values.get('RTX_GPUS', len(entry.get('gpus', [first])))) == 1 else [0, 1]
    if len(set(gpus)) != len(gpus) or not set(gpus) <= {0, 1}:
        raise ValueError('invalid coordinator GPUs')
    hosts = list(entry.get('sparks', []))
    values.pop('SPARK_HOSTS', None)
    count = int(values.get('SPARK_COUNT', len(hosts)))
    if not 0 <= count <= len(HOSTS):
        raise ValueError('invalid Spark count')
    hosts = [values.get(f'SPARK_{i}_HOST', hosts[i] if i < len(hosts) else HOSTS[i]) for i in range(count)]
    if len(set(hosts)) != count or not set(hosts) <= set(HOSTS):
        raise ValueError('invalid Spark inventory')
    for key in list(values):
        if re.fullmatch(r'SPARK_[0-5]_(HOST|LANE_A|LANE_B)', key):
            del values[key]
    for i, host in enumerate(hosts):
        index = HOSTS.index(host) + 1
        values.update({f'SPARK_{i}_HOST': host, f'SPARK_{i}_LANE_A': f'10.55.0.{index}', f'SPARK_{i}_LANE_B': f'10.55.1.{index}'})
    values.update(API_KEY_FILE=str(key_file), ENABLE_BENCH='on', WIP_INSTANCE=arm.get('wip_instance', arm['instance']), INSTANCE=arm['instance'], SPARK_COUNT=str(count), RTX_GPUS=str(len(gpus)), COORDINATOR_GPU=str(gpus[0]))
    run_args = ['--wip', arm['slot']]
    if 'release' in arm:
        values.pop('WIP_INSTANCE', None)
        values.pop('WIP_ROOT', None)
        values['COORDINATOR_DOCKER_INFERENCE'] = 'ghcr.io/tpurtell/cuteafd-coordinator:' + arm['release']
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
    if values.get('PROFILE'):
        # One per-job directory, mounted on the coordinator and every Spark at
        # the same path; Spark-side files are collected back after the run.
        values['PROFILE_DIR'] = str(dest / 'profile')
        (dest / 'profile').mkdir(exist_ok=True)
        if 'nsys' in values['PROFILE'].split(','):
            # The coordinator's nsys session starts paused; smoke opens it after
            # the sweep's first warm wave and closes it (writing the report) at
            # sweep end, or before teardown if the sweep stopped in between.
            if not (entry.get('levels') or entry.get('prefill_lengths')):
                raise ValueError(card + ': PROFILE=nsys captures the timed sweep; pass --levels or --prefill-lengths')
            exec_nsys = ['docker', 'exec', 'cuteafd-coordinator-' + values['INSTANCE'], '/usr/local/cuda/bin/nsys']
            entry['window'] = dict(
                open=dict(argv=exec_nsys + ['start', '--session=' + NSYS_SESSION], timeout_s=120, required=True),
                close=dict(argv=exec_nsys + ['stop', '--session=' + NSYS_SESSION], timeout_s=1800, required=True))
    elif 'PROFILE_DIR' in values:
        raise ValueError(card + ': PROFILE_DIR is owned by the driver; set PROFILE only')
    if trace:
        if values.get('SPECULATION_TRACE') or values.get('COORDINATOR_TRACE'):
            raise ValueError(card + ': --trace owns SPECULATION_TRACE; drop the explicit setting')
        values['SPECULATION_TRACE'] = str(dest / 'speculation-trace.jsonl')
    base = dest / 'base.config'
    base.write_text(''.join(f'{k}={v}\n' for k, v in sorted(values.items())))
    metadata = {k: entry.pop(k) for k in ('source_matrix', 'correctness_only', 'published', 'compared', 'kind', 'probes', 'expected_pool') if k in entry}
    entry.update(name=label, config=str(base), set=values, gpus=gpus, sparks=hosts, run_args=run_args)
    job = dict(card=card, arm=arm_name, repeat=repeat, simulated=bool(simulated), state=str(dest), entry=entry, metadata=metadata, probes=probes, expected_pool=expected_pool, trace=bool(trace))
    hook = [sys.executable, str(Path(__file__).resolve())]
    timeout = int(entry.get('timeout_s', 900)) + int(entry.get('run_timeout_s', 900)) + 120
    if probes & {'memory', 'console'} or trace:
        entry['observer'] = dict(argv=hook + ['--observe', str(dest / 'job.json')], timeout_s=min(timeout, 7200), required=True)
        if 'memory' in probes:
            entry['observer']['cleanup'] = sampler_cleanup(job)
    if expected_pool is not None or 'image' in probes:
        entry['precheck'] = dict(argv=hook + ['--precheck', str(dest / 'job.json')], timeout_s=180, required=True)
    # Re-checked at grant: smoke runs prelaunch once it holds this card's locks.
    entry['prelaunch'] = dict(argv=hook + ['--preflight-job', str(dest / 'job.json')], timeout_s=300, required=True)
    save(dest / 'job.json', {**job, 'probes': sorted(probes)})
    save(dest / 'matrix.json', dict(build='wip-cards', profile=entry.get('profile', 'smoke'), entries=[entry]))
    return job


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
        if set(ea['gpus']) & set(eb['gpus']) or set(ea['sparks']) & set(eb['sparks']):
            raise ValueError('parallel cards overlap GPUs or Sparks')
        for key in ('ADDR', 'EXPERT_PORT'):
            if ea['set'].get(key) == eb['set'].get(key):
                raise ValueError('parallel cards overlap ' + key)
        if ea['set']['INSTANCE'] == eb['set']['INSTANCE']:
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


def preflight_job(job, arm):
    """Everything about one job that is checkable without hardware: no serving
    container of ours exists, the arm is sealed on every host it runs on, the
    slot holds the packages and audio entry the card needs, and the job's
    trace/profile directories are writable. Returns the artifact record."""
    values = job['entry']['set']
    assert_absent(job)
    needs = job.get('needs') or {'coordinator': [], 'spark': [], 'audio': False}
    artifacts, problems = {}, []
    for host in ('raptor', *job['entry']['sparks']):
        meta = json.loads(slot_check(arm, host))
        release = 'release' in arm
        keys = ('image', 'image_id', 'repo_digests') if release else ('slot', 'seal_sha256', 'artifact_manifest_sha256')
        artifacts[host] = {k: meta[k] for k in keys}
        contents = meta.get('contents')
        if release or contents is None:
            continue
        wanted = needs['coordinator'] if host == 'raptor' else needs['spark']
        missing = sorted(set(wanted) - set(contents['packages']))
        if missing:
            problems.append(f"{host}: slot {arm['slot']} lacks {', '.join(missing)} (build with the card's scopes: wip-cards --build)")
        if needs['audio'] and contents.get('audio') is False:
            problems.append(f"{host}: slot {arm['slot']} was built without the audio tower (AUDIO={values.get('AUDIO', 'auto')}); "
                            'rebuild with CUTEAFD_WIP_AUDIO_AOT=ON or set AUDIO=off')
        role = needs.get('tp_role')
        # The seal check always reports tp_roles (None: no TP manifest); a record
        # without the field is not a seal check result and carries no attestation.
        if role and host != 'raptor' and 'tp_roles' in contents:
            problems.extend(tp_role_problems(host, arm['slot'], role, needs.get('nvfp4', False), contents))
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


def smoke_command(binary, repo, job):
    dest = Path(job['state'])
    return [binary, 'bench', 'smoke', '--matrix', str(dest / 'matrix.json'), '--repo', str(repo), '--state', str(dest / 'smoke'), '--out-root', str(dest / 'reports'), '--parallel', '1']


def detached(cmd, log, exit_file, env):
    # Shell remains in the detached group and writes a numeric status even if the
    # CLI disappears. No flock here: smoke owns all serving locks.
    script = '"$@"; rc=$?; printf "%s\\n" "$rc" > "$CARD_EXIT"; exit "$rc"'
    with Path(log).open('w') as output:
        return subprocess.Popen(['setsid', 'bash', '-c', script, 'wip-card', *cmd], stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, env={**env, 'CARD_EXIT': str(exit_file)}, start_new_session=False)


def http(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url + path, data=data, headers={'Authorization': 'Bearer ' + os.environ['CUTEAFD_API_KEY'], 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")}') from None


def red_square():
    def chunk(name, data):
        return struct.pack('>I', len(data)) + name + data + struct.pack('>I', zlib.crc32(name + data))
    raw = (b'\x00' + b'\xff\x00\x00' * 224) * 224
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 224, 224, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def ready_log(job):
    name = 'cuteafd-coordinator-' + job['entry']['set']['INSTANCE']
    result = run(['docker', 'logs', name])
    text = result.stdout + result.stderr
    (Path(job['state']) / 'ready.log').write_text(text)
    return ANSI.sub('', text)


def admitted_pool(text):
    values = re.findall(r'\b(?:admitted_pool_tokens|admitted_tokens|pool_tokens)\s*=\s*(\d+)', ANSI.sub('', text))
    if not values:
        raise ValueError('ready.log has no admitted pool value')
    return int(values[-1])


def precheck(job):
    dest = Path(job['state'])
    result = {}
    try:
        text = ready_log(job)
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


def observe(job):
    dest = Path(job['state'])
    stop = Path(os.environ['CUTEAFD_SMOKE_STOP'])
    sampler = None
    name = 'cuteafd-card-memory-' + job['entry']['set']['INSTANCE']
    memory = (dest / 'memory.jsonl').open('w')
    console = (dest / 'console.jsonl').open('w')
    frames = (dest / 'console-frames.jsonl').open('w')
    policy = (dest / 'policy-stats.jsonl').open('w')
    policy_sample_at = 0.
    reader = None
    if 'console' in job['probes']:
        reader = threading.Thread(target=console_frames, args=(os.environ['CUTEAFD_SMOKE_URL'], stop, frames), daemon=True)
        reader.start()
    try:
        if 'memory' in job['probes']:
            e = job['entry']
            gpu_list = ','.join(map(str, e['gpus']))
            image = e['set'].get('COORDINATOR_DOCKER_DEV', 'cuteafd-coordinator-dev')
            owner = hashlib.sha256(job['state'].encode()).hexdigest()
            cmd = ['docker', 'run', '--rm', '-i', '--name', name, '--label', 'cuteafd.card.owner=' + owner, '--pid=host', '--network=none', '--gpus', '"device=' + gpu_list + '"', '-e', 'CARD_GPUS=' + gpu_list, '--entrypoint', 'python3', image, '-u', '-c', CUDA_SAMPLER]
            sampler = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=memory, stderr=subprocess.PIPE, text=True)
        trace_path = Path(job['entry']['set']['SPECULATION_TRACE']) if job.get('trace') else None
        trace_due = None
        while not stop.exists():
            if sampler and sampler.poll() is not None:
                raise RuntimeError('memory sampler exited: ' + sampler.stderr.read())
            if trace_path and not (dest / 'trace-check.json').exists():
                # After the baseline's warm-up (the first part of the run), the
                # trace must have records. Diagnostic only: the row decides.
                try:
                    active = (http(os.environ['CUTEAFD_SMOKE_URL'], '/v1/bench/status').get('active') or {})
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
                    snapshot = http(os.environ['CUTEAFD_SMOKE_URL'], '/v1/console/snapshot')
                    console.write(json.dumps({'time': time.time(), 'snapshot': snapshot}) + '\n')
                    console.flush()
                    if time.monotonic() >= policy_sample_at:
                        stats = http(os.environ['CUTEAFD_SMOKE_URL'], '/v1/stats')
                        policy.write(json.dumps({'time': time.time(), 'stats': stats}) + '\n')
                        policy.flush()
                        policy_sample_at = time.monotonic() + 1.
                except (OSError, RuntimeError):
                    pass
            time.sleep(.25)
    finally:
        if sampler:
            run(sampler_cleanup(job))
            sampler.wait(timeout=10)
        stop.touch()
        if reader:
            reader.join(timeout=3)
        frames.close()
        policy.close()
        memory.close()
        console.close()
    if 'memory' in job['probes'] and not (dest / 'memory.jsonl').stat().st_size:
        raise ValueError('memory probe produced no samples')
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


def collect_profile(job):
    """Copies Spark-side PROFILE files back into the job's profile directory
    (the coordinator writes there directly). Missing hosts are recorded, not fatal.
    Profiler reports (nsys/ncu) are listed with their sizes, not read."""
    values = job['entry']['set']
    directory = values.get('PROFILE_DIR')
    if not directory:
        return None
    result = {'dir': directory, 'switches': values.get('PROFILE', '').split(','), 'hosts': {}}
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


def summarize_job(job, exit_code):
    dest = Path(job['state'])
    row = {k: job[k] for k in ('card', 'arm', 'repeat', 'simulated')}
    row.update(exit_code=exit_code, status='failed' if exit_code else 'pass')
    if 'artifacts' in job: row['artifacts'] = job['artifacts']
    reports = list((dest / 'reports').glob('**/report.json'))
    if job.get('grouped'):
        reports = [path for path in reports if path.parent.name.endswith('-' + job['entry']['name'])]
    state = dest / 'smoke' / 'state.json'
    if state.exists():
        row['smoke'] = json.loads(state.read_text())
    outcome = (row.get('smoke') or {}).get(job['entry']['name']) or {}
    # The first real error wins: the launcher's own words, then smoke's outcome
    # (its wrapper "./run.sh exited ..." last), then checks derived below.
    found = [e for e in (first_log_error(dest, job['entry']['name']), outcome.get('error')) if e]
    found.sort(key=lambda e: e.startswith(('error: ./run.sh exited', './run.sh exited', 'error: not ready', 'not ready')))
    if exit_code or outcome.get('status') not in (None, 'done'):
        for message in found:
            fail(row, message)
    launched = outcome.get('launch_s') is not None or bool(reports)
    sweep_only = job['entry'].get('profile') == 'panels:' and bool(job['entry'].get('levels') or job['entry'].get('prefill_lengths'))
    if sweep_only and not reports:
        pass  # No server-side profile ran; the sweep below carries the numbers.
    elif len(reports) != 1:
        fail(row, f'expected one report, found {len(reports)}')
    else:
        report = json.loads(reports[0].read_text())
        present = {p['id'] for p in report.get('panels', [])} | ({'baseline'} if report.get('baseline') else set())
        missing = set(requested_panels(job['entry'].get('profile', 'smoke'))) - present
        if missing:
            fail(row, 'requested panels absent from report: ' + ', '.join(sorted(missing)))
        baseline = report.get('baseline') or {}
        card = baseline.get('card') or {}
        quality = baseline.get('quality') or {}
        checks = quality.get('checks', [])
        fidelity = next((c for c in checks if c['id'] == 'fidelity'), {})
        row.update(C1=next((d['tok_s'] for d in card.get('decode', []) if d['content'] == 'code'), None), C8=(card.get('concurrent') or {}).get('aggregate_tok_s'), **{'8K': (card.get('prefill') or {}).get('tok_s')}, kl=fidelity.get('metrics', {}).get('kl'), top1=fidelity.get('metrics', {}).get('top1'), cache=next((c for c in checks if c['id'] in ('cache_exact', 'cache')), None), spec=next((c for c in checks if 'spec' in c['id']), None), pool=(card.get('capacity') or {}).get('kv_tokens'), readiness_s=report.get('server', {}).get('readiness_s'), report=str(reports[0]), quality=quality.get('status'))
        # Report/panel statuses serialize as 'failed'; Rust CheckStatus as 'fail'.
        failed = {'fail', 'failed'}
        bad = [c['id'] + (': ' + c['summary'] if c.get('summary') else '') for c in checks if c.get('status') in failed]
        if bad:
            fail(row, 'quality: ' + '; '.join(bad)[:400], 'quality')
        elif report.get('status') in failed or quality.get('status') in failed:
            fail(row, 'report ' + str(report.get('status')) + ': ' + str(report.get('error') or '')[:300])
        lossless = lossless_set_result(report)
        if lossless:
            row['lossless_set'] = lossless
            if lossless['status'] == 'fail':
                fail(row, 'quality: spec_lossless_set: ' + lossless['summary'][:400], 'quality')
    if job.get('correctness_parallel'):
        for metric in ('C1', 'C8', '8K'):
            row[metric] = None
    profile = collect_profile(job)
    if profile:
        row['profile'] = profile
        for switch, suffix in PROFILER_REPORTS.items():
            if switch in profile['switches'] and not any(r['name'].endswith(suffix) and r['bytes'] for r in profile['reports']):
                fail(row, f'PROFILE={switch} produced no {suffix} report in {profile["dir"]}', 'measurement')
    trace_path = job['entry']['set'].get('SPECULATION_TRACE')
    if trace_path and job.get('trace'):
        path = Path(trace_path)
        records = sum(1 for line in path.open(errors='replace') if line.strip()) if path.is_file() else 0
        row['trace'] = {'path': trace_path, 'bytes': path.stat().st_size if path.is_file() else 0, 'records': records,
                        'expected': speculating(job['entry']['set'])}
        if row['trace']['expected'] and not records:
            fail(row, 'speculation trace is empty after the run (' + trace_path + ')', 'measurement')
    sweeps = [p for p in (dest / 'reports').glob('**/sweep.json')
              if not job.get('grouped') or p.parent.name.endswith('-' + job['entry']['name']) or p.parent.name == job['entry']['name']]
    if job['entry'].get('levels') or job['entry'].get('prefill_lengths'):
        if len(sweeps) != 1:
            fail(row, f'expected one sweep.json, found {len(sweeps)}')
        else:
            sweep = json.loads(sweeps[0].read_text())
            row['sweep'] = sweep.get('points', [])
            for point in row['sweep']:
                key = 'post-load C1' if point['phase'] == 'post_load' else f"sweep C{point['c']}"
                row[key] = point['aggregate_tok_s']
            for point in sweep.get('prefill', []):
                row[f"prefill {point['length']}"] = point['tok_s']
                row[f"prefill {point['length']} TTFT s"] = point['ttft_s']
    row['peaks'] = peaks(dest)
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
    if job.get('nonce_seed'):
        text = '\n'.join(ANSI.sub('', p.read_text(errors='replace')) for p in (dest / 'smoke' / 'logs').glob('*.coordinator.log'))
        hashes = re.findall(r'prompt_token_hash\s*=\s*"?([0-9a-f]{16})', text)
        row.update(nonce_seed=job['nonce_seed'], first_prompt_token_hash=hashes[0] if hashes else None, requests=matched_requests(dest), state=str(dest))
        # Prefill-only sweeps (max_tokens 1) emit no decode rounds; the prompt
        # token hash above is what matches their requests.
        prefill_only = bool(job['entry'].get('prefill_lengths')) and not job['entry'].get('levels') \
            and job['entry'].get('profile') == 'panels:'
        if not row['requests'] and not prefill_only:
            fail(row, 'matched prompts produced no per-request console rounds')
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
    return {'status': 'pass' if last.get('gate') == 'pass' else 'fail', 'summary': last.get('summary', ''),
            'passed': last.get('passed'), 'total': last.get('total'), 'set': last.get('set'),
            'failing': [r['id'] for r in last.get('rows', []) if not r.get('pass')]}


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
        hashes = [r.get('first_prompt_token_hash') for r in pair]
        if not all(hashes) or len(set(hashes)) != 1:
            for row in pair:
                fail(row, 'matched pair missing or mismatched first request prompt token hash', 'measurement')
                row['exit_code'] = row.get('exit_code') or 81
                (Path(row['state']) / 'exit').write_text('81\n')
                save(Path(row['state']) / 'result.json', row)


def summary_math(rows, arm_names, metrics=METRICS):
    medians, paired = [], []
    for card in sorted({r['card'] for r in rows}):
        for arm in arm_names:
            group = [r for r in rows if r['card'] == card and r['arm'] == arm and r['status'] == 'pass' and not r['simulated']]
            medians.append(dict(card=card, arm=arm, **{m: statistics.median([r[m] for r in group if isinstance(r.get(m), (float, int))]) if any(isinstance(r.get(m), (float, int)) for r in group) else None for m in metrics}))
        for candidate in arm_names[1:]:
            for m in metrics:
                deltas, percentages = [], []
                for repeat in sorted({r['repeat'] for r in rows}):
                    pair = {r['arm']: r for r in rows if r['card'] == card and r['repeat'] == repeat and r['status'] == 'pass' and not r['simulated']}
                    a, b = pair.get(arm_names[0], {}).get(m), pair.get(candidate, {}).get(m)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        deltas.append(b - a)
                        if a:
                            percentages.append(100 * (b - a) / a)
                paired.append(dict(card=card, baseline=arm_names[0], candidate=candidate, metric=m, pairs=len(deltas), median_delta=statistics.median(deltas) if deltas else None, median_percent=statistics.median(percentages) if percentages else None))
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


def write_summary(state, rows, arms):
    validate_matched_pairs(rows, list(arms))
    names = list(arms)
    extra = sorted({k for r in rows for k in r if k.startswith('prefill ') and isinstance(r.get(k), (int, float))},
                   key=lambda k: (int(k.split()[1]), k))
    present = [m for m in (*SUMMARY_METRICS, *extra) if any(isinstance(r.get(m), (int, float)) for r in rows)]
    medians, paired = summary_math(rows, names, present or METRICS)
    save(state / 'summary.json', dict(rows=rows, medians=medians, paired_deltas=paired))
    def cell(value):
        if isinstance(value, float):
            return f'{value:.4g}'
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(',', ':'))
        return str(value if value is not None else '').replace('|', '\\|').replace('\n', ' ')
    text = f'# wip-cards summary: {state.name}\n\n'
    failed = [r for r in rows if r['status'] != 'pass']
    text += f'{len(rows) - len(failed)} of {len(rows)} jobs passed.\n\n'
    # Before -> after per card and metric: medians of passing real-hardware rows,
    # ratio after/before, and the paired median where pairs exist.
    if len(names) > 1:
        text += '## Before -> after\n\n| Card | Metric | ' + names[0] + ' | ' + ' | '.join(names[1:]) + ' | Ratio | Paired % | Pairs |\n'
        text += '| --- | --- |' + ' ---: |' * len(names) + ' ---: | ---: | ---: |\n'
        for card in sorted({r['card'] for r in rows}):
            by_arm = {m['arm']: m for m in medians if m['card'] == card}
            for metric in present:
                values = [by_arm.get(arm, {}).get(metric) for arm in names]
                if all(v is None for v in values):
                    continue
                before, after = values[0], values[-1]
                ratio = after / before if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before else None
                pair = next((p for p in paired if p['card'] == card and p['candidate'] == names[-1] and p['metric'] == metric), {})
                note = ' (lower is better)' if metric in LOWER_IS_BETTER or metric.endswith('TTFT s') else ''
                text += f"| {card} | {metric}{note} | " + ' | '.join(cell(v) for v in values) + f" | {cell(ratio)} | {cell(pair.get('median_percent'))} | {pair.get('pairs', 0)} |\n"
        text += '\n'
    else:
        text += '## Results\n\n| Card | ' + ' | '.join(present) + ' |\n| --- |' + ' ---: |' * len(present) + '\n'
        for m in medians:
            text += f"| {m['card']} | " + ' | '.join(cell(m.get(k)) for k in present) + ' |\n'
        text += '\n'
    text += '## Jobs\n\n| Card | Arm | Repeat | Status | Failure | Spec | Cache | Trace | Profile | Error |\n| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |\n'
    for r in rows:
        trace = r.get('trace') or {}
        trace_cell = (f"{trace.get('records', 0)} records" if trace.get('expected') else 'not expected') if trace else ''
        profile = r.get('profile') or {}
        profile_cell = f"{len(profile.get('files', []))} files" if profile else ''
        if profile.get('reports'):
            profile_cell += ': ' + ', '.join(f"{r['name']} ({r['bytes'] / 2**20:.1f} MiB)" for r in profile['reports'])
        text += f"| {r['card']} | {r['arm']} | {r['repeat']} | {r['status']}{' (simulated)' if r.get('simulated') else ''} | {r.get('failure_kind') or ''} | {cell(spec_cell(r))} | {cell(status_cell(r.get('cache')))} | {trace_cell} | {profile_cell} | {cell((r.get('error') or '')[:240])} |\n"
    if failed:
        text += '\n## Failures\n\nlaunch = the server never served the card; quality = it served and a correctness check failed; measurement = it served and a measurement or evidence check failed; preflight = refused before hardware.\n\n'
        for r in failed:
            text += f"- **{r['card']} / {r['arm']} / r{r['repeat']}** ({r.get('failure_kind') or 'unknown'}): {r.get('error') or 'exit ' + str(r.get('exit_code'))}\n"
            for extra in (r.get('errors') or [])[1:]:
                text += f'  - then: {extra}\n'
    (state / 'summary.md').write_text(text)


def assert_absent(job):
    e = job['entry']
    targets = [('raptor', 'cuteafd-coordinator-' + e['set']['INSTANCE'])]
    targets += [(h, f"cuteafd-spark-expert-{h}-{e['set']['EXPERT_PORT']}") for h in e['sparks']]
    for host, name in targets:
        cmd = ['docker', 'ps', '-a', '--format', '{{.Names}}']
        if host != 'raptor':
            cmd = ['ssh', '-o', 'BatchMode=yes', host, shlex.join(cmd)]
        if name in run(cmd).stdout.splitlines():
            raise ValueError('refusing existing (possibly foreign) container: ' + name)


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
    audio=None
    try:
        out=subprocess.run(['nm','-D','--defined-only',str(art/'libcuteafd_native.so')],capture_output=True,text=True,timeout=60)
        if out.returncode==0: audio=any(line.split()[-1:]==['cuteafd_audio_required'] for line in out.stdout.splitlines())
    except (OSError,subprocess.SubprocessError):
        pass
    return dict(packages=packages,audio=audio,tp_roles=tp_roles(p,art))
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
        assert manifest.get('schema')==1 and isinstance(roles,list) and all(r in ('tp2','tp3','tp6') for r in roles),'invalid TP roles'
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
    instance, slot = arm['instance'], arm['slot']
    root = arm.get('root', str(Path.home() / '.cache/cuteafd/builds' / ('wip-' + instance)))
    owner = json.dumps({'task': task, 'instance': instance}, sort_keys=True)
    script = '''import json, pathlib, shutil, subprocess, sys
home=pathlib.Path.home()
root=home/'.cache/cuteafd/builds'/ROOT_NAME
owner=OWNER
if not root.exists():
    names=NAMES
    present=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
    residual=[home/'.cache/cuteafd/wip-run'/INSTANCE/SLOT,home/'.cache/cuteafd/wip-slots'/INSTANCE/(SLOT+'.json')]
    legacy=home/'.cache/cuteafd/wip-slots'/(SLOT+'.json')
    if legacy.is_file() and json.loads(legacy.read_text()).get('wip_instance')==INSTANCE: residual.append(legacy)
    if any(name in present for name in names) or any(p.exists() or p.is_symlink() for p in residual):
        raise RuntimeError('unowned cleanup residue without WIP root '+str(root))
    print('verified absent', root); sys.exit(0)
if root.is_symlink(): raise RuntimeError('refusing root symlink')
marker=root/'.wip-card-owner.json'
if not marker.is_file() or json.loads(marker.read_text())!=owner:
    raise RuntimeError('refusing unowned WIP root '+str(root))
names=NAMES
for name in names:
    ids=subprocess.check_output(['docker','ps','-aq','--filter','name=^'+name+'$'],text=True).split()
    for ident in ids:
        info=json.loads(subprocess.check_output(['docker','inspect',ident],text=True))[0]
        env=info['Config'].get('Env',[])
        mounts=info.get('Mounts',[])
        if 'WIP_INSTANCE='+INSTANCE not in env or not any(m['Source']==str(root) for m in mounts):
            raise RuntimeError('refusing foreign container '+name)
        subprocess.run(['docker','rm','-f',ident],check=True)
    remaining=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
    if name in remaining: raise RuntimeError('container still present '+name)
overlay=home/'.cache/cuteafd/wip-run'/INSTANCE/SLOT
if overlay.is_symlink(): raise RuntimeError('refusing overlay symlink')
if overlay.exists(): shutil.rmtree(overlay)
for registration, shared in [(home/'.cache/cuteafd/wip-slots'/INSTANCE/(SLOT+'.json'), False),(home/'.cache/cuteafd/wip-slots'/(SLOT+'.json'), True)]:
    if registration.exists():
        data=json.loads(registration.read_text())
        if data.get('wip_instance')==INSTANCE and data.get('slot')==SLOT: registration.unlink()
        # The legacy slot-name file is shared by every instance using that slot
        # name; another instance's entry there is not ours to remove or refuse.
        elif not shared: raise RuntimeError('refusing foreign registration '+str(registration))
if root.is_symlink(): raise RuntimeError('refusing root symlink')
shutil.rmtree(root)
for path in [root,overlay,home/'.cache/cuteafd/wip-slots'/INSTANCE/(SLOT+'.json')]:
    if path.exists() or path.is_symlink(): raise RuntimeError('cleanup failed '+str(path))
print('verified absent', root, overlay)
'''
    replacements = {'ROOT_NAME': repr(Path(root).name), 'OWNER': repr(json.loads(owner)), 'NAMES': repr(['cuteafd-coordinator-wip-' + instance, 'cuteafd-spark-expert-wip-' + instance]), 'INSTANCE': repr(instance), 'SLOT': repr(slot)}
    return re.sub(r'\b(ROOT_NAME|OWNER|NAMES|INSTANCE|SLOT)\b', lambda m: replacements[m[0]], script)


def cleanup(arms, task, dry):
    for arm in arms.values():
        if 'wip_instance' in arm or 'release' in arm:
            continue  # External sealed builds and release images are never owned by this task.
        script = cleanup_script(arm, task)
        for host in ('raptor', *HOSTS, *SEED_HOSTS):
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


def build_scopes(cards):
    # Derived per card: a card with no Sparks (local experts) contributes only
    # coordinator scopes, and a Spark card's TP count is keyed to its own family
    # (FAMILY=COUNT, read by native/cmake/shared/{exl3,fp8_moe}.cmake), so one
    # card's geometry is never exported at another card's TP.
    scopes, spark_scopes, roles, counts, env = set(), set(), set(), set(), {}
    for card in cards:
        family = card.get('family')
        values = {**config(card['config']), **card.get('set', {})}
        count = int(values.get('SPARK_COUNT', len(card.get('sparks', []))))
        scopes_before = set(scopes)
        scopes = set()
        if family == 'deepseek_v41' and count in (2, 3, 6):
            roles.add('tp' + str(count))
        if family == 'deepseek_v4':
            tag = 'dsv4p' if 'pro' in card['name'] else 'dsv4f'
            scopes.update({tag + ':rtx_backbone', tag + (':spark_tp2' if count == 2 else ':spark')})
            if int(values.get('RTX_GPUS', len(card.get('gpus', [0])))) == 2:
                scopes.add(tag + ':rtx_tp2')
            env['CUTEAFD_WIP_DSV4_AOT'] = 'ON'
            if 'exl3' in card['name']:
                scopes.add(tag + ':exl3-k23')
        for fam, tag, switch in [('glm5', 'glm', 'GLM'), ('glm5_flash', 'glmf', 'GLMF'), ('mimo_v2', 'mimop' if 'pro' in card['name'] else 'mimof', 'MIMO'), ('qwen4', 'qwen4', 'QWEN4')]:
            if family == fam:
                env['CUTEAFD_WIP_' + switch + '_AOT'] = 'ON'
                if count:
                    counts.add(f"{tag}={int(values.get('SPARK_TP', count))}")
                # Qwen's FP8 package has no Spark TP layout (intermediate 640): request the NVFP4 family,
                # which CMake expands to nvfp4, nvfp4a4 and, on the coordinator only, the FP8 MTP sibling
                # (fp8_moe.cmake), as the release image ships them.
                checkpoint_id = values.get('MODEL_ID', values.get('MODEL', ''))
                nvfp4 = any('nvfp4' in str(value).lower() for value in
                            (card['name'], checkpoint_id, values.get('EXPERT_FORMAT', '')))
                scopes.add(tag + (':nvfp4' if tag == 'qwen4' or nvfp4 else ':fp8'))
                if fam == 'glm5_flash' and nvfp4:
                    scopes.add('glmfdense:nvfp4')
                if 'exl3' in card['name'] or 'tr3' in card['name']:
                    scopes.add(tag + (':exl3-k34' if tag == 'glmf' else ':exl3-k45'))
                if fam == 'mimo_v2':
                    # MiMo places its audio tower unless the card turns audio off.
                    if values.get('AUDIO', 'auto') != 'off':
                        env['CUTEAFD_WIP_AUDIO_AOT'] = 'ON'
                    # Two RTX cards run the head-split programs (mimof2/mimop2).
                    wanted = {tag, tag + '2'} if int(values.get('RTX_GPUS', len(card.get('gpus', [0])))) == 2 else {tag}
                    env['CUTEAFD_WIP_MIMO_GEOMETRIES'] = ';'.join(sorted(set(env.get('CUTEAFD_WIP_MIMO_GEOMETRIES', '').split(';')) - {''} | wanted))
        card_scopes, scopes = scopes, scopes_before | scopes
        if count:
            spark_scopes |= {scope for scope in card_scopes if not scope.startswith('glmfdense:')}
    env.update(CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES=';'.join(sorted(scopes)),
               CUTEAFD_WIP_GENERIC_SPARK_COUNTS=';'.join(sorted(counts)))
    env.update(CUTEAFD_KACHE_SPARK='1', CUTEAFD_WIP_EXPERT_FAMILIES=';'.join(sorted(spark_scopes)), CUTEAFD_WIP_SPARK_TP_ROLES=';'.join(sorted(roles)), CUTEAFD_WIP_EXPORT_LOCKS='on', CUTEAFD_KACHE=str(Path.home() / '.cache/cuteafd/tools/x86_64/kache'), CARGO_BUILD_JOBS='16', RUST_TEST_THREADS='16', CMAKE_BUILD_PARALLEL_LEVEL='16')
    return env


def package_dir(scope):
    """The slot directory (under .cuteafd-wip/) a FAMILY:FORMAT scope ships as
    (scripts/build/build-wip-artifacts.sh), or None for scopes with no package."""
    family, fmt = scope.split(':', 1)
    if fmt == 'fp8':
        return 'fp8/fp8-' + family
    if fmt in ('nvfp4', 'nvfp4a4'):
        return f'fp8/fp8-{family}-{fmt}'
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
    values = {**config(card['config']), **card.get('set', {})}
    if int(values.get('SPARK_COUNT', len(card.get('sparks', [])))) == 0:
        spark = set()
    return dict(coordinator=sorted(coordinator), spark=sorted(spark),
                audio=scopes.get('CUTEAFD_WIP_AUDIO_AOT') == 'ON', **v41_tp_role(card, values))


def v41_tp_role(card, values):
    """The SM121 expert role a DeepSeek V4.1 card's Spark TP requires, as the
    launcher derives it (scripts/launch/run-family.sh, launch_deepseek_v41): an
    explicit SPARK_TP/SPARK_EP topology with TP 2, 3 or 6, or a native non-EXL3
    count other than 0 and 4 (TP = count). TP4 is the built-in default role.
    NVFP4 checkpoints need the role's NVFP4 exports too."""
    if card.get('family') != 'deepseek_v41':
        return dict(tp_role=None, nvfp4=False)
    count = int(values.get('SPARK_COUNT', len(card.get('sparks', []))))
    explicit = bool(values.get('SPARK_TP') or values.get('SPARK_EP'))
    exl3 = values.get('EXPERT_FORMAT', 'native') == 'exl3'
    tp = int(values['SPARK_TP']) if explicit and values.get('SPARK_TP') else count
    role = None
    if explicit and tp in (2, 3, 6):
        role = f'tp{tp}'
    elif not explicit and not exl3 and count not in (0, 4) and count in (2, 3, 6):
        role = f'tp{count}'
    return dict(tp_role=role, nvfp4=checkpoint_is_nvfp4(values))


def checkpoint_is_nvfp4(values):
    """Whether the card's checkpoint ships NVFP4 routed experts (its own
    config.json's moe_quant_algo, as the launcher reads it)."""
    model = values.get('MODEL_ID', 'deepseek-ai/DeepSeek-V4.1-Flash')
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
    digest = hashlib.sha256()
    for name in ('docker/Dockerfile.dev', 'docker/entrypoint.sh', 'scripts/build/install-dev-cache-tools.sh',
                 'scripts/build/dev-toolchain.py'):
        shown = subprocess.run(['git', '-C', str(REPO), 'show', f'{revision}:{name}'], capture_output=True, timeout=120)
        if shown.returncode:
            raise ValueError(f'cannot read {name} at {revision}: ' + shown.stderr.decode(errors='replace').strip())
        blob = shown.stdout
        digest.update(name.encode() + b'\0' + blob + b'\0')
    return digest.hexdigest()[:16]


# Serving-only keys never belong in a build (staging) config.
BUILD_DROP = {'SPARK_HOSTS', 'BENCH_SIMULATED', 'BENCH_HARDWARE_CLASS', 'BENCH_EXPECT_GLM_RING', 'COORDINATOR_GPUS',
              'COORDINATOR_GPU_UUID', 'COORDINATOR_GPU_PCI_BUS_ID', 'SPARK_TP', 'SPARK_EP', 'API_KEY_FILE', 'INSTANCE',
              'WIP_INSTANCE', 'WIP_ROOT', 'ADDR', 'EXPERT_PORT', 'PROFILE', 'PROFILE_DIR', 'SPECULATION_TRACE',
              'PROFILE_NCU_KERNEL', 'PROFILE_NCU_COUNT', 'PROFILE_NCU_SET', 'PROFILE_NSYS_GPU_METRICS',
              'COORDINATOR_DOCKER_INFERENCE', 'SPARK_EXPERT_DOCKER_INFERENCE'}


def build_lock_mode(revision):
    text = run(['git', '-C', str(REPO), 'show', revision + ':wip.sh']).stdout
    native = run(['git', '-C', str(REPO), 'show', revision + ':scripts/build/build-wip-artifacts.sh']).stdout
    # Native-phase sources lock only their export: on the host under a held fair
    # turn (CUTEAFD_WIP_HOST_EXPORT_TURN) or, older, through a mounted lock file.
    if 'CUTEAFD_WIP_EXPORT_LOCKS' in text and ('CUTEAFD_WIP_HOST_EXPORT_TURN' in text or 'CUTEAFD_WIP_EXPORT_LOCK_FILES' in native):
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


def seed_env(arm, seed, dry):
    """wip.sh's seed parameters for a Spark build: WIP_SEED_HOST, plus the bulk
    transfer address of a seed off the RoCE fabric. Legacy wip.sh reads neither."""
    if not arm.get('seed_separate') or arm.get('build_role') != 'both':
        return {}
    return dict(WIP_SEED_HOST=seed, **({} if seed in HOSTS else {'WIP_SEED_ADDR': seed_lanes(seed, dry)[0]}))


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


def build_arms(items, arms, cards, state, task, dry, seed=DEFAULT_SEED):
    scopes = build_scopes(cards)
    for item in items:
        name, revision = item.split('=', 1)
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
        spark_cards = [card for card in cards if int({**config(card['config']), **card.get('set', {})}.get('SPARK_COUNT', len(card.get('sparks', [])))) > 0]
        build_role = 'both' if spark_cards else 'coordinator'
        arm['build_role'] = build_role
        build_cmd = ['nice', '-n', '19', str(source / 'wip.sh'), '--slot', arm['slot'], '--role', build_role, '--config', str(state / ('build-' + name + '.config'))]
        # A wip.sh that takes a build slot around Cargo itself needs no outer build.lock:
        # staging, reuse, GPU export turns and distribution then never block other builds.
        arm['build_slot'] = build_slot_capable(revision)
        if arm['build_lock_mode'] == 'legacy-full-build':
            print('# legacy arm: GPU lock held for the full build')
            build_cmd = [sys.executable, str(REPO / 'scripts/hw/hwlock.py'), '--locks', 'gpu1', '--wait', '1800', '--', *build_cmd]
        print('# build locking ' + name + ': ' + arm['build_lock_mode'])
        # Seed on an out-of-pool host; stage the sealed slot to the whole serving union.
        # The seed is wip.sh's own parameter (WIP_SEED_HOST), never a SPARK_* rank,
        # unless this revision's wip.sh predates it.
        union = {h for card in cards for h in card.get('sparks', [])}
        arm['seed_separate'] = seed_option_capable(revision)
        hosts = staging_hosts(union, seed, arm['seed_separate'])
        if build_role == 'both' and len(hosts) > 6:
            raise ValueError(f'{name}: {revision} predates WIP_SEED_HOST, so its wip.sh builds on SPARK_0_HOST and '
                             f'{seed} plus the {len(union)}-Spark union makes {len(hosts)} ranks; use --seed rhea|moa '
                             'for this arm or build a revision with the separate seed (#117)')
        if build_role == 'both':
            print(f"# build seed {name}: {seed} ({'WIP_SEED_HOST' if arm['seed_separate'] else 'legacy SPARK_0_HOST'}); "
                  f"Spark targets ({len(hosts)}): {','.join(hosts)}")
        build_config = state / ('build-' + name + '.config')
        # A staging config: the first card's model and its --set/--arm-set
        # overrides, the union of Spark hosts, and no serving topology. Export
        # scopes (TP counts, families, audio) come from every card through
        # build_scopes, never from how many hosts receive the slot.
        values = {**config(cards[0]['config']), **cards[0].get('set', {})}
        for key in list(values):
            if key in BUILD_DROP or re.fullmatch(r'SPARK_[0-7]_(HOST|LANE_A|LANE_B)', key):
                del values[key]
        values.update(SPARK_COUNT=str(len(hosts)), COORDINATOR_GPU='1', RTX_GPUS='1')
        if len(hosts) in (2, 3):
            # Two or three ranks without explicit keys are the compact EXL3 layout.
            values.update(EXPERT_FORMAT='exl3', SPARKINFER_EXL3='force', EXL3_PAIRED_TP4='off')
        tag = 'tc-' + toolchain_hash(revision)
        values.update(COORDINATOR_DOCKER_DEV='cuteafd-coordinator-dev:' + tag, SPARK_EXPERT_DOCKER_DEV='cuteafd-spark-expert-dev:' + tag)
        arm['dev_image_tag'] = tag
        for i, host in enumerate(hosts):
            lane_a, lane_b = seed_lanes(host, dry)
            values.update({f'SPARK_{i}_HOST': host, f'SPARK_{i}_LANE_A': lane_a, f'SPARK_{i}_LANE_B': lane_b})
        build_config.write_text(''.join(f'{k}={v}\n' for k, v in sorted(values.items())))
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
            wanted = run(['git', '-C', str(REPO), 'rev-parse', revision + '^{commit}']).stdout.strip()
            dirty = run(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no']).stdout.strip()
            if head != wanted or dirty:
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
                    # A native-phase wip.sh takes the seed's fair turn itself, around the export only:
                    # Cargo, slot reuse and native-cache hits need no Spark lock, so they never queue.
                    if build_role == 'both' and arm['build_lock_mode'] != 'native-phase':
                        # Wait (not fail) while another job's export holds the seed Spark; wip.sh
                        # takes the lock itself for the native phase, so release it at once.
                        seed_wait = int(os.environ.get('CUTEAFD_WIP_SEED_WAIT_S', '14400'))
                        # A fair turn (hwlock), so a seed wait never overtakes an older queued job.
                        subprocess.run([sys.executable, str(REPO / 'scripts/hw/hwlock.py'), '--locks', seed,
                                        '--wait', str(seed_wait), '--', 'true'], check=True, timeout=seed_wait + 60)
                    for host in (('raptor', *HOSTS, *([seed] if seed not in HOSTS else [])) if build_role == 'both' else ('raptor',)):
                        marker = 'import pathlib,json; p=pathlib.Path.home()/".cache/cuteafd/builds"/' + repr(root.name) + '; p.mkdir(parents=True,exist_ok=True); m=p/".wip-card-owner.json"; owner=' + repr(dict(task=task, instance=arm['instance'])) + '; assert not m.exists() or json.loads(m.read_text())==owner; assert m.exists() or not any(p.iterdir()); m.write_text(json.dumps(owner))'
                        mark = ['python3', '-c', marker]
                        run(mark if host == 'raptor' else ['ssh', '-o', 'BatchMode=yes', host, shlex.join(mark)])
                    env = {**os.environ, **scopes, 'WIP_INSTANCE': arm['instance'], 'WIP_ROOT': str(root)}
                    env.pop('WIP_SEED_HOST', None)
                    env.pop('WIP_SEED_ADDR', None)
                    env.update(seed_env(arm, seed, dry))
                    if arm['build_slot']:
                        env['CUTEAFD_WIP_BUILD_SLOT'] = 'on'
                    if arm['build_lock_mode'] == 'legacy-full-build':
                        env.pop('CUTEAFD_WIP_EXPORT_LOCKS', None)
                    save(state / ('build-' + name + '.json'), dict(revision=revision, lock_mode=arm['build_lock_mode'], scopes=scopes, command=cmd))
                    result = subprocess.run(cmd, env=env, cwd=source, timeout=14400)
                    if result.returncode:
                        raise ValueError('build failed: ' + name)
                else:
                    # Recursive submodule checkout of a fresh worktree outlasts run()'s 120 s.
                    subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=1800)
        if dry:
            print('# identical build scopes ' + json.dumps(scopes, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', default=str(Path.home() / '.cache/cuteafd/builds/release-v2-rc2/kit'),
                        help='kit directory, or "gate" for the shared gate kit (scripts/bench/kits/gate: mimo26-flash-max, '
                        'qwen38-exl3-min, glmf-exl3-max, v41-flash-max)')
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
    parser.add_argument('--task', default='cards-' + time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--state', type=Path)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--binary', default='cuteafd')
    parser.add_argument('--set', action='append', default=[], metavar='CARD:KEY=VALUE')
    parser.add_argument('--arm-set', action='append', default=[], metavar='ARM:CARD:KEY=VALUE', help='arm-specific setting, overriding --set')
    parser.add_argument('--probe', action='append', default=[], metavar='CARD:image,memory,console')
    parser.add_argument('--panels', metavar='LIST', help='run these server panels on every card instead of its profile '
                        '(comma list; "baseline" for the basic card + quick quality alone; "none" for no server profile)')
    parser.add_argument('--lossless-set', type=Path, metavar='FILE', help='case file for the spec_lossless_set panel '
                        '(JSON [{id, prompt, nonce}]); default: the shipped scripts/bench/fixtures/spec-lossless-set-v1.json')
    parser.add_argument('--levels', metavar='C,C', help='client-side code-decode sweep at these concurrency levels after the profile, e.g. 1,16')
    parser.add_argument('--post-load-c1', action='store_true', help='re-measure C1 after the widest --levels level')
    parser.add_argument('--prefill-lengths', metavar='N,N', help='client-side cold prefill at these prompt lengths (one untimed warm '
                        'request each, then one measured), e.g. 8192')
    parser.add_argument('--trace', action='store_true', help='every job writes SPECULATION_TRACE to <job>/speculation-trace.jsonl; '
                        'checked after warm-up and recorded in result.json')
    parser.add_argument('--resume', action='store_true', help='reuse --state (or --task): keep passed jobs, rerun failed or missing '
                        'ones; a rerun job keeps its earlier attempt under attempt-N/')
    parser.add_argument('--preflight-only', action='store_true', help='run every CPU check (slots, packages, audio, containers, '
                        'binary, trace dirs), write the plan and exit; for hwlock --preflight')
    parser.add_argument('--expect-pool', action='append', default=[], metavar='CARD=N')
    parser.add_argument('--build', action='append', default=[], metavar='ARM=REV')
    parser.add_argument('--build-only', action='store_true', help='build and seal selected arms without generating or launching smoke jobs')
    parser.add_argument('--seed', '--seed-host', dest='seed_host', default=DEFAULT_SEED, choices=SEEDS,
                        help=f'out-of-serving Spark that compiles and seals --build arms (default {DEFAULT_SEED}); '
                        'kakapo is a 10 GbE build-only seed whose slots reach the serving Sparks through sparknest')
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--observe', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--precheck', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--preflight-job', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.observe or args.precheck:
        job = json.loads((args.observe or args.precheck).read_text())
        (observe if args.observe else precheck)(job)
        return 0
    if args.preflight_job:
        # Smoke's prelaunch hook: the same checks again, now that smoke holds the locks.
        job = json.loads(args.preflight_job.read_text())
        plan = json.loads((args.preflight_job.resolve().parent.parent / 'plan.json').read_text())
        artifacts = preflight_job(job, plan['arms'][job['arm']])
        if job.get('artifacts') and artifacts != job['artifacts']:
            raise ValueError('sealed artifacts changed since enqueue: ' + json.dumps(artifacts, sort_keys=True))
        print('preflight at grant: pass')
        return 0
    if not NAME.fullmatch(args.task) or args.parallel < 1 or args.repeats < 1:
        parser.error('invalid task, repeats or parallelism')
    if args.build_only and (not args.build or args.cleanup):
        parser.error('--build-only requires --build and refuses --cleanup')
    arms = arms_from(args.arm, args.arm_release)
    shared_wip(arms, args.arm_wip, args.build, args.parallel)
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
        cleanup(arms, args.task, args.dry_run)
        return 0
    if not args.cards or not (args.matrix or args.interleave):
        parser.error('--cards and --matrix or --interleave required')
    kit = REPO / 'scripts/bench/kits' / args.kit if NAME.fullmatch(args.kit) and (REPO / 'scripts/bench/kits' / args.kit).is_dir() else Path(args.kit)
    cards = load_cards(kit.expanduser().resolve(), args.cards)
    overrides = overrides_from(args.set)
    arm_overrides = {}
    for item in args.arm_set:
        name, setting = item.split(':', 1)
        if name not in arms:
            raise ValueError('unknown override arm: ' + name)
        parsed = overrides_from([setting])
        if set(parsed) - set(args.cards):
            raise ValueError('arm override names must be selected cards')
        for card, values in parsed.items():
            arm_overrides.setdefault(name, {}).setdefault(card, {}).update(values)
    if args.post_load_c1 and not args.levels:
        parser.error('--post-load-c1 needs --levels')
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
    if args.panels is not None:
        selection = '' if args.panels.strip() == 'none' else args.panels.strip()
        if not selection and not (levels or prefill_lengths):
            parser.error('--panels none runs nothing without --levels or --prefill-lengths')
        profile = 'panels:' + selection
        listed = requested_panels(profile)
        for card in cards:
            card['profile'] = profile
            if listed and 'baseline' not in listed:
                card['baseline'] = False  # Explicit panels alone; list baseline to keep it.
    lossless_set = None
    if args.lossless_set is not None:
        lossless_set = load_lossless_set(args.lossless_set)
        if not any('spec_lossless_set' in requested_panels(card.get('profile', 'smoke')) for card in cards):
            parser.error('--lossless-set needs --panels with spec_lossless_set')
    for card in cards:
        if lossless_set:
            card['lossless_set'] = lossless_set
        if prefill_lengths:
            card['prefill_lengths'] = prefill_lengths
        if levels:
            card['levels'] = levels
            if args.post_load_c1:
                card['post_load_c1'] = True
        if args.trace and card.get('family') == 'deepseek_v41' and not launcher_forwards_v41_trace(args.repo):
            raise ValueError(f"{card['name']}: this checkout's launcher does not forward SPECULATION_TRACE on the V4.1 path")
    probes = {card['name']: set(card.get('probes', [])) for card in cards}
    expected = {card['name']: card.get('expected_pool') for card in cards}
    for item in args.probe:
        name, values = item.split(':', 1)
        probes[name] = set(values.split(','))
        if not probes[name] <= {'image', 'memory', 'console'}:
            raise ValueError('unknown probe')
    for item in args.expect_pool:
        name, value = item.split('=', 1)
        expected[name] = int(value)
    if (set(overrides) | set(probes) | set(expected)) - set(args.cards):
        raise ValueError('override/probe names must be selected cards')
    for card in cards:
        if args.matched_prompts or args.nonce_seed is not None:
            probes[card['name']].add('console')
            if args.matched_prompts and 'image' in probes[card['name']]:
                raise ValueError('--matched-prompts cannot precede its first benchmark request with an image probe')
        requested_panels(card.get('profile', 'smoke'))
        card['set'].update(overrides.get(card['name'], {}))
        values = {**config(card['config']), **card['set']}
        count = int(values.get('SPARK_COUNT', len(card.get('sparks', []))))
        if not 0 <= count <= len(HOSTS):
            raise ValueError('invalid Spark count')
        card['sparks'] = [values.get(f'SPARK_{i}_HOST', card.get('sparks', [])[i] if i < len(card.get('sparks', [])) else HOSTS[i]) for i in range(count)]
    # Build and stage every arm's layout, even when only one arm requests it.
    build_cards = []
    for name in arms:
        for card in cards:
            variant = {**card, 'set': {**card['set'], **arm_overrides.get(name, {}).get(card['name'], {})}}
            values = {**config(variant['config']), **variant['set']}
            count = int(values.get('SPARK_COUNT', len(variant['sparks'])))
            if not 0 <= count <= len(HOSTS):
                raise ValueError('invalid Spark count')
            variant['sparks'] = [values.get(f'SPARK_{i}_HOST', card['sparks'][i] if i < len(card['sparks']) else HOSTS[i]) for i in range(count)]
            build_cards.append(variant)
    build_arms(args.build, arms, build_cards, state, args.task, args.dry_run, args.seed_host)
    if args.build_only:
        save(state / 'plan.json', {'task': args.task, 'arms': arms, 'jobs': []})
        return 0
    key_file = state / 'api-key'
    if not key_file.exists():
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as key:
            key.write(secrets.token_hex(32) + '\n')
    if key_file.is_symlink() or key_file.stat().st_mode & 0o077:
        raise ValueError('API key file must be private (0600), not a symlink')
    bindings = dict(item.split('=', 1) for item in args.card_arm)
    if bindings and (args.interleave or set(bindings) - set(args.cards) or set(bindings.values()) - set(arms)):
        raise ValueError('--card-arm needs selected cards/arms in matrix mode')
    jobs = [generate(card, name, arm, state, key_file, {**overrides.get(card['name'], {}), **arm_overrides.get(name, {}).get(card['name'], {})}, probes[card['name']], expected[card['name']], repeat, args.trace) for card, name, arm, repeat in schedule(cards, arms, args.repeats if args.interleave else 1, args.interleave) if card['name'] not in bindings or bindings[card['name']] == name]
    variants = {(c['name'], n): {**c, 'set': {**c['set'], **arm_overrides.get(n, {}).get(c['name'], {})}} for c in cards for n in arms}
    for job in jobs:
        job['needs'] = slot_needs(variants[(job['card'], job['arm'])])
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
        if groups and len(groups[0]) > 1:
            cmd = smoke_command(args.binary, args.repo, dict(state=str(state / 'parallel-smoke')))
            cmd[-1] = str(args.parallel)
            prefix = ('CUTEAFD_BENCH_NONCE_SEED=' + shlex.quote(args.nonce_seed) + ' ') if args.nonce_seed is not None else ''
            print(prefix + 'setsid ' + shlex.join(cmd) + ' # grouped smoke scheduler; numeric exits per card')
            return 0
        for job in jobs:
            cmd = smoke_command(args.binary, args.repo, job)
            prefix = ('CUTEAFD_BENCH_NONCE_SEED=' + shlex.quote(job['nonce_seed']) + ' ') if job.get('nonce_seed') else ''
            print('# settings ' + job['arm'] + '/' + job['card'] + ': ' + json.dumps(job['entry']['set'], sort_keys=True))
            print(prefix + 'setsid ' + shlex.join(cmd) + ' # numeric exit -> ' + job['state'] + '/exit')
            for kind in ('observer', 'precheck'):
                if kind in job['entry']:
                    print('# smoke ' + kind + ': ' + shlex.join(job['entry'][kind]['argv']))
        return 0
    kept = {}
    if args.resume:
        for job in jobs:
            dest = Path(job['state'])
            result = dest / 'result.json'
            if result.exists() and json.loads(result.read_text()).get('status') == 'pass':
                kept[job['state']] = json.loads(result.read_text())
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
        if kept:
            print(f'wip-cards: resume keeps {len(kept)} passed job(s), reruns {len(jobs) - len(kept)}', file=sys.stderr)
    fields = (['levels', 'post_load_c1'] if args.levels else []) + (['prefill_lengths'] if prefill_lengths else []) + \
        (['baseline'] if any(job['entry'].get('baseline') is False for job in jobs) else []) + \
        (['window'] if any(job['entry'].get('window') for job in jobs) else []) + \
        (['lossless_set'] if any(job['entry'].get('lossless_set') for job in jobs) else [])
    for job in jobs[:1]:
        binary_supports(args.binary, args.repo, job, fields)
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
    def execute(job):
        dest = Path(job['state'])
        if job['state'] in kept:
            return kept[job['state']]
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
        process = detached(smoke_command(args.binary, args.repo, job), dest / 'driver.log', exit_file, job_env)
        rc = process.wait()
        code = int(exit_file.read_text()) if exit_file.exists() else (rc if rc >= 0 else 128 - rc)
        exit_file.write_text(str(code) + '\n')
        row = summarize_job(job, code)
        row['state'] = str(dest)
        if row['status'] != 'pass' and not code:
            exit_file.write_text('81\n')
            row['exit_code'] = 81
            save(dest / 'result.json', row)
        return row
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
        command = smoke_command(args.binary, args.repo, dict(state=str(group_root)))
        command[-1] = str(args.parallel)
        process = detached(command, group_root / 'driver.log', group_root / 'exit', env)
        rc = process.wait()
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
        return int(any(row['status'] != 'pass' for row in rows))
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        for row in executor.map(execute, jobs):
            rows.append(row)
            write_summary(state, rows, arms)
    return int(any(r['status'] != 'pass' for r in rows))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, subprocess.SubprocessError, OSError) as error:
        print('wip-cards: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
