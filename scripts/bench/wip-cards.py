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
HOSTS = ('ostrich', 'dodo', 'emu', 'kiwi', 'rhea', 'moa')
CORRECTNESS = {'sim-5090', 'fidelity', 'cache', 'image', 'no-fit', 'packed-check'}
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
METRICS = ('C1', 'C8', '8K', 'kl', 'top1', 'pool', 'readiness_s')
PANELS = {'baseline', 'hardware', 'configuration', 'decode_content', 'concurrency', 'prefill', 'retained', 'prefix_cache', 'startup', 'agentic', 'tool_eval', 'structured', 'ifeval', 'code', 'math', 'needle', 'fidelity', 'fidelity_full', 'reasoning_effort'}


def short_instance(value):
    if not NAME.fullmatch(value):
        raise ValueError('invalid instance: ' + value)
    return value if len(value) <= 41 else value[:32] + '-' + hashlib.sha256(value.encode()).hexdigest()[:8]


def requested_panels(profile):
    if not profile.startswith('panels:'):
        return []
    panels = [p.strip() for p in profile[7:].split(',') if p.strip()]
    bad = set(panels) - PANELS
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


def arms_from(values):
    arms = {}
    for value in values:
        name, address = value.split('=', 1)
        instance, slot = address.split(':', 1)
        if not all(NAME.fullmatch(x) for x in (name, instance, slot)) or len(instance) > 41 or len(slot) > 64:
            raise ValueError('invalid arm, WIP instance or slot')
        if name in arms or instance in [a['instance'] for a in arms.values()]:
            raise ValueError('arm names and WIP instances must be unique')
        arms[name] = {'instance': instance, 'slot': slot}
    return arms


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
                path = kit / path
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


def generate(entry, arm_name, arm, state, key_file, overrides, probes, expected_pool, repeat):
    entry = json.loads(json.dumps(entry))
    requested_panels(entry.get('profile', 'smoke'))
    values = config(entry['config'])
    values.update(entry.pop('set', {}))
    simulated = values.pop('BENCH_SIMULATED', '') or 'sim5090' in entry['name'] or 'sim-5090' in entry['name']
    values.pop('BENCH_HARDWARE_CLASS', None)
    values.update(overrides)
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
    values.update(API_KEY_FILE=str(key_file), ENABLE_BENCH='on', WIP_INSTANCE=arm['instance'], INSTANCE=arm['instance'], SPARK_COUNT=str(count), RTX_GPUS=str(len(gpus)), COORDINATOR_GPU=str(gpus[0]))
    if arm.get('root'):
        values['WIP_ROOT'] = arm['root']
    card = entry['name']
    values['INSTANCE'] = short_instance(f"{arm['instance']}-{card}-r{repeat}")
    if len(values['INSTANCE']) > 41:
        raise ValueError('serving INSTANCE exceeds 41 characters')
    label = f'{card}-{arm_name}-r{repeat}' + ('-simulated' if simulated else '')
    dest = state / label
    dest.mkdir(parents=True, exist_ok=True)
    base = dest / 'base.config'
    base.write_text(''.join(f'{k}={v}\n' for k, v in sorted(values.items())))
    metadata = {k: entry.pop(k) for k in ('source_matrix', 'correctness_only', 'published', 'compared', 'kind', 'probes', 'expected_pool') if k in entry}
    entry.update(name=label, config=str(base), set=values, gpus=gpus, sparks=hosts, run_args=['--wip', arm['slot']])
    job = dict(card=card, arm=arm_name, repeat=repeat, simulated=bool(simulated), state=str(dest), entry=entry, metadata=metadata, probes=probes, expected_pool=expected_pool)
    hook = [sys.executable, str(Path(__file__).resolve())]
    timeout = int(entry.get('timeout_s', 900)) + int(entry.get('run_timeout_s', 900)) + 120
    if probes & {'memory', 'console'}:
        entry['observer'] = dict(argv=hook + ['--observe', str(dest / 'job.json')], timeout_s=min(timeout, 7200), required=True)
        if 'memory' in probes:
            entry['observer']['cleanup'] = sampler_cleanup(job)
    if expected_pool is not None or 'image' in probes:
        entry['precheck'] = dict(argv=hook + ['--precheck', str(dest / 'job.json')], timeout_s=180, required=True)
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
        while not stop.exists():
            if sampler and sampler.poll() is not None:
                raise RuntimeError('memory sampler exited: ' + sampler.stderr.read())
            if 'console' in job['probes']:
                try:
                    snapshot = http(os.environ['CUTEAFD_SMOKE_URL'], '/v1/console/snapshot')
                    console.write(json.dumps({'time': time.time(), 'snapshot': snapshot}) + '\n')
                    console.flush()
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


def summarize_job(job, exit_code):
    dest = Path(job['state'])
    row = {k: job[k] for k in ('card', 'arm', 'repeat', 'simulated')}
    row.update(exit_code=exit_code, status='failed' if exit_code else 'pass')
    reports = list((dest / 'reports').glob('**/report.json'))
    if job.get('grouped'):
        reports = [path for path in reports if path.parent.name.endswith('-' + job['entry']['name'])]
    if len(reports) != 1:
        row.update(status='failed', error=f'expected one report, found {len(reports)}')
    else:
        report = json.loads(reports[0].read_text())
        present = {p['id'] for p in report.get('panels', [])} | ({'baseline'} if report.get('baseline') else set())
        missing = set(requested_panels(job['entry'].get('profile', 'smoke'))) - present
        if missing:
            row.update(status='failed', error='requested panels absent from report: ' + ', '.join(sorted(missing)))
        baseline = report.get('baseline') or {}
        card = baseline.get('card') or {}
        quality = baseline.get('quality') or {}
        checks = quality.get('checks', [])
        fidelity = next((c for c in checks if c['id'] == 'fidelity'), {})
        row.update(C1=next((d['tok_s'] for d in card.get('decode', []) if d['content'] == 'code'), None), C8=(card.get('concurrent') or {}).get('aggregate_tok_s'), **{'8K': (card.get('prefill') or {}).get('tok_s')}, kl=fidelity.get('metrics', {}).get('kl'), top1=fidelity.get('metrics', {}).get('top1'), cache=next((c for c in checks if c['id'] in ('cache_exact', 'cache')), None), spec=next((c for c in checks if 'spec' in c['id']), None), pool=(card.get('capacity') or {}).get('kv_tokens'), readiness_s=report.get('server', {}).get('readiness_s'), report=str(reports[0]), quality=quality.get('status'))
        if report.get('status') == 'failed' or quality.get('status') == 'failed' or any(c.get('status') == 'failed' for c in checks):
            row['status'] = 'failed'
    if job.get('correctness_parallel'):
        for metric in ('C1', 'C8', '8K'):
            row[metric] = None
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
    state = dest / 'smoke' / 'state.json'
    if state.exists():
        row['smoke'] = json.loads(state.read_text())
    if job.get('nonce_seed'):
        text = '\n'.join(ANSI.sub('', p.read_text(errors='replace')) for p in (dest / 'smoke' / 'logs').glob('*.coordinator.log'))
        hashes = re.findall(r'prompt_token_hash\s*=\s*"?([0-9a-f]{16})', text)
        row.update(nonce_seed=job['nonce_seed'], first_prompt_token_hash=hashes[0] if hashes else None, requests=matched_requests(dest), state=str(dest))
        if not row['requests']:
            row.update(status='failed', error='matched prompts produced no per-request console rounds')
    save(dest / 'result.json', row)
    return row


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
                row.update(status='failed', error='matched pair missing or mismatched first request prompt token hash', exit_code=81)
                (Path(row['state']) / 'exit').write_text('81\n')
                save(Path(row['state']) / 'result.json', row)


def summary_math(rows, arm_names):
    medians, paired = [], []
    for card in sorted({r['card'] for r in rows}):
        for arm in arm_names:
            group = [r for r in rows if r['card'] == card and r['arm'] == arm and r['status'] == 'pass' and not r['simulated']]
            medians.append(dict(card=card, arm=arm, **{m: statistics.median([r[m] for r in group if isinstance(r.get(m), (float, int))]) if any(isinstance(r.get(m), (float, int)) for r in group) else None for m in METRICS}))
        for candidate in arm_names[1:]:
            for m in METRICS:
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


def write_summary(state, rows, arms):
    validate_matched_pairs(rows, list(arms))
    medians, paired = summary_math(rows, list(arms))
    save(state / 'summary.json', dict(rows=rows, medians=medians, paired_deltas=paired))
    columns = ['card', 'arm', 'repeat', 'status', 'simulated', *METRICS, 'cache', 'spec', 'peaks', 'quality', 'nonce_seed', 'first_prompt_token_hash', 'requests']
    def cell(value):
        if isinstance(value, float):
            return f'{value:.4g}'
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(',', ':'))
        return str(value if value is not None else '').replace('|', '\\|').replace('\n', ' ')
    text = '| ' + ' | '.join(columns) + ' |\n|' + '|'.join([' --- '] * len(columns)) + '|\n'
    text += ''.join('| ' + ' | '.join(cell(r.get(c)) for c in columns) + ' |\n' for r in rows)
    text += '\n## Medians (real hardware, passing rows only)\n\n| Card | Arm | C1 | C8 | 8K |\n| --- | --- | ---: | ---: | ---: |\n'
    text += ''.join('| ' + ' | '.join(cell(r.get(c)) for c in ('card', 'arm', 'C1', 'C8', '8K')) + ' |\n' for r in medians)
    text += '\n## Paired Deltas\n\n| Card | Candidate | Metric | Pairs | Delta | Percent |\n| --- | --- | --- | ---: | ---: | ---: |\n'
    text += ''.join('| ' + ' | '.join(cell(r.get(c)) for c in ('card', 'candidate', 'metric', 'pairs', 'median_delta', 'median_percent')) + ' |\n' for r in paired if r['pairs'])
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
    container = ('cuteafd-coordinator-wip-' if host == 'raptor' else 'cuteafd-spark-expert-wip-') + arm['instance']
    role = 'coordinator' if host == 'raptor' else 'spark-expert'
    program = '''import hashlib,json,pathlib,sys
p=pathlib.Path('/wip/slots')/sys.argv[1]/sys.argv[2]
meta=(p/'META.json').read_bytes()
assert hashlib.sha256(meta).hexdigest()==(p/'FINGERPRINT').read_text().strip(), 'META seal mismatch'
m=json.loads(meta)
assert m['slot']==sys.argv[1] and m['role']==sys.argv[2] and m['wip_instance']==sys.argv[3], 'slot identity mismatch'
for relative,key in [('SOURCE_SHA256SUMS','source_manifest_sha256'),('workspace/.cuteafd-wip/ARTIFACT_SHA256SUMS','artifact_manifest_sha256')]:
    assert hashlib.sha256((p/relative).read_bytes()).hexdigest()==m[key], relative+' seal mismatch'
print(json.dumps(m))
'''
    cmd = ['docker', 'exec', container, 'python3', '-c', program, arm['slot'], role, arm['instance']]
    try:
        return run(cmd if host == 'raptor' else ['ssh', '-o', 'BatchMode=yes', host, shlex.join(cmd)]).stdout
    except subprocess.SubprocessError as error:
        raise ValueError(f'{host}: missing or unsealed WIP slot {arm["instance"]}:{arm["slot"]}: {getattr(error, "stderr", str(error))}') from None


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
for registration in [home/'.cache/cuteafd/wip-slots'/INSTANCE/(SLOT+'.json'),home/'.cache/cuteafd/wip-slots'/(SLOT+'.json')]:
    if registration.exists():
        data=json.loads(registration.read_text())
        if data.get('wip_instance')==INSTANCE and data.get('slot')==SLOT: registration.unlink()
        else: raise RuntimeError('refusing foreign registration '+str(registration))
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
        script = cleanup_script(arm, task)
        for host in ('raptor', *HOSTS):
            cmd = ['python3', '-c', script] if host == 'raptor' else ['ssh', '-o', 'BatchMode=yes', host, shlex.join(['python3', '-c', script])]
            print(shlex.join(cmd))
            if not dry:
                print(run(cmd).stdout.strip())


def build_scopes(cards):
    scopes, roles, env = set(), set(), {}
    for card in cards:
        family = card.get('family')
        values = {**config(card['config']), **card.get('set', {})}
        count = int(values.get('SPARK_COUNT', len(card.get('sparks', []))))
        if family == 'deepseek_v41' and count in (2, 3, 6):
            roles.add('tp' + str(count))
        if family == 'deepseek_v4':
            tag = 'dsv4p' if 'pro' in card['name'] else 'dsv4f'
            scopes.update({tag + ':rtx_backbone', tag + (':spark_tp2' if count == 2 else ':spark')})
            env['CUTEAFD_WIP_DSV4_AOT'] = 'ON'
            if 'exl3' in card['name']:
                scopes.add(tag + ':exl3-k23')
        for fam, tag, switch in [('glm5', 'glm', 'GLM'), ('glm5_flash', 'glmf', 'GLMF'), ('mimo_v2', 'mimop' if 'pro' in card['name'] else 'mimof', 'MIMO'), ('qwen4', 'qwen4', 'QWEN4')]:
            if family == fam:
                env['CUTEAFD_WIP_' + switch + '_AOT'] = 'ON'
                scopes.add(tag + ':fp8')
                if 'exl3' in card['name'] or 'tr3' in card['name']:
                    scopes.add(tag + (':exl3-k34' if tag == 'glmf' else ':exl3-k45'))
                if fam == 'mimo_v2':
                    # MiMo places its audio tower unless the card turns audio off.
                    if values.get('AUDIO', 'auto') != 'off':
                        env['CUTEAFD_WIP_AUDIO_AOT'] = 'ON'
                    # Two RTX cards run the head-split programs (mimof2/mimop2).
                    wanted = {tag, tag + '2'} if int(values.get('RTX_GPUS', len(card.get('gpus', [0])))) == 2 else {tag}
                    env['CUTEAFD_WIP_MIMO_GEOMETRIES'] = ';'.join(sorted(set(env.get('CUTEAFD_WIP_MIMO_GEOMETRIES', '').split(';')) - {''} | wanted))
    env.update(CUTEAFD_KACHE_SPARK='1', CUTEAFD_WIP_EXPERT_FAMILIES=';'.join(sorted(scopes)), CUTEAFD_WIP_SPARK_TP_ROLES=';'.join(sorted(roles)), CUTEAFD_WIP_EXPORT_LOCKS='on', CUTEAFD_KACHE=str(Path.home() / '.cache/cuteafd/tools/x86_64/kache'), CARGO_BUILD_JOBS='16', RUST_TEST_THREADS='16', CMAKE_BUILD_PARALLEL_LEVEL='16')
    return env


def build_lock_mode(revision):
    text = run(['git', '-C', str(REPO), 'show', revision + ':wip.sh']).stdout
    native = run(['git', '-C', str(REPO), 'show', revision + ':scripts/build/build-wip-artifacts.sh']).stdout
    return 'native-phase' if 'CUTEAFD_WIP_EXPORT_LOCKS' in text and 'CUTEAFD_WIP_EXPORT_LOCK_FILES' in native else 'legacy-full-build'


def build_arms(items, arms, cards, state, task, dry):
    scopes = build_scopes(cards)
    for item in items:
        name, revision = item.split('=', 1)
        if name not in arms:
            raise ValueError('unknown build arm ' + name)
        arm = arms[name]
        source = Path.home() / '.cache/cuteafd/builds' / task / ('source-' + name)
        if not dry:
            source.parent.mkdir(parents=True, exist_ok=True)
        root = Path.home() / '.cache/cuteafd/builds' / ('wip-' + arm['instance'])
        arm['root'] = str(root)
        arm['build_revision'] = revision
        arm['build_lock_mode'] = build_lock_mode(revision)
        build_cmd = ['nice', '-n', '19', str(source / 'wip.sh'), '--slot', arm['slot'], '--role', 'both', '--config', str(state / ('build-' + name + '.config'))]
        if arm['build_lock_mode'] == 'legacy-full-build':
            print('# legacy arm: GPU lock held for the full build')
            build_cmd = ['flock', '-w', '1800', str(Path.home() / '.cache/cuteafd/gpu1.lock'), *build_cmd]
        print('# build locking ' + name + ': ' + arm['build_lock_mode'])
        # Seed on an out-of-pool Spark; stage the sealed slot to the whole union.
        union = {h for card in cards for h in card.get('sparks', [])}
        hosts = ['moa', *[h for h in HOSTS if h in union and h != 'moa']]
        # Config validation admits 2/3/4/6 ranks, not arbitrary staging unions.
        if len(hosts) not in (2, 3, 4, 6):
            hosts += [h for h in HOSTS if h not in hosts][:6 - len(hosts)]
        build_config = state / ('build-' + name + '.config')
        values = config(cards[0]['config'])
        values.pop('SPARK_HOSTS', None)
        values.pop('BENCH_SIMULATED', None)
        values.pop('BENCH_HARDWARE_CLASS', None)
        values.pop('COORDINATOR_GPUS', None)
        values.pop('COORDINATOR_GPU_UUID', None)
        values.pop('COORDINATOR_GPU_PCI_BUS_ID', None)
        for key in list(values):
            if re.fullmatch(r'SPARK_[0-5]_(HOST|LANE_A|LANE_B)', key):
                del values[key]
        values.update(SPARK_COUNT=str(len(hosts)), COORDINATOR_GPU='1', RTX_GPUS='1')
        if len(hosts) == 2:
            values.update(EXPERT_FORMAT='exl3', SPARKINFER_EXL3='force', EXL3_PAIRED_TP4='off')
            values.pop('SPARK_TP', None)
            values.pop('SPARK_EP', None)
        else:
            values.update(EXPERT_FORMAT='native', SPARKINFER_EXL3='disable', EXL3_PAIRED_TP4='off', SPARK_TP=str(len(hosts)), SPARK_EP='1', MODEL_VARIANT='flash')
        for i, host in enumerate(hosts):
            idx = HOSTS.index(host) + 1
            values.update({f'SPARK_{i}_HOST': host, f'SPARK_{i}_LANE_A': f'10.55.0.{idx}', f'SPARK_{i}_LANE_B': f'10.55.1.{idx}'})
        build_config.write_text(''.join(f'{k}={v}\n' for k, v in sorted(values.items())))
        commands = [
            ['git', '-C', str(REPO), 'worktree', 'add', '--detach', str(source), revision],
            ['git', '-C', str(source), 'submodule', 'update', '--init', '--recursive'],
            [sys.executable, str(REPO / 'scripts/build/assert-build-filesystem.py'), str(root), str(source)],
            ['flock', '-w', '1800', str(Path.home() / '.cache/cuteafd/build.lock'), *build_cmd],
        ]
        for cmd in commands:
            print(shlex.join(cmd))
            if not dry:
                if cmd[0] == 'flock':
                    run(['flock', '-n', str(Path.home() / '.cache/cuteafd/moa.lock'), 'true'])
                    for host in ('raptor', *HOSTS):
                        marker = 'import pathlib,json; p=pathlib.Path.home()/".cache/cuteafd/builds"/' + repr(root.name) + '; p.mkdir(parents=True,exist_ok=True); m=p/".wip-card-owner.json"; owner=' + repr(dict(task=task, instance=arm['instance'])) + '; assert not m.exists() or json.loads(m.read_text())==owner; assert m.exists() or not any(p.iterdir()); m.write_text(json.dumps(owner))'
                        mark = ['python3', '-c', marker]
                        run(mark if host == 'raptor' else ['ssh', '-o', 'BatchMode=yes', host, shlex.join(mark)])
                    env = {**os.environ, **scopes, 'WIP_INSTANCE': arm['instance'], 'WIP_ROOT': str(root)}
                    if arm['build_lock_mode'] == 'legacy-full-build':
                        env.pop('CUTEAFD_WIP_EXPORT_LOCKS', None)
                    save(state / ('build-' + name + '.json'), dict(revision=revision, lock_mode=arm['build_lock_mode'], scopes=scopes, command=cmd))
                    result = subprocess.run(cmd, env=env, cwd=source, timeout=14400)
                    if result.returncode:
                        raise ValueError('build failed: ' + name)
                else:
                    run(cmd)
        if dry:
            print('# identical build scopes ' + json.dumps(scopes, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, default=Path.home() / '.cache/cuteafd/builds/release-v2-rc2/kit')
    parser.add_argument('--cards', nargs='+', default=[])
    parser.add_argument('--arm', action='append', default=[], metavar='NAME=INSTANCE:SLOT')
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
    parser.add_argument('--expect-pool', action='append', default=[], metavar='CARD=N')
    parser.add_argument('--build', action='append', default=[], metavar='ARM=REV')
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--observe', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--precheck', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.observe or args.precheck:
        job = json.loads((args.observe or args.precheck).read_text())
        (observe if args.observe else precheck)(job)
        return 0
    if not NAME.fullmatch(args.task) or args.parallel < 1 or args.repeats < 1:
        parser.error('invalid task, repeats or parallelism')
    arms = arms_from(args.arm)
    if args.nonce_seed is not None and (not args.nonce_seed or '\n' in args.nonce_seed or '\r' in args.nonce_seed):
        parser.error('--nonce-seed must be non-empty and single-line')
    if not arms:
        parser.error('at least one --arm is required')
    if args.matched_prompts and not args.interleave:
        parser.error('--matched-prompts requires --interleave')
    if args.interleave and (len(arms) < 2 or args.parallel != 1):
        parser.error('interleave needs two or more arms and --parallel 1')
    state = (args.state or Path.home() / '.cache/cuteafd/builds' / args.task).expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    if args.cleanup:
        plan = state / 'plan.json'
        if plan.exists():
            recorded = json.loads(plan.read_text())
            if recorded['task'] != args.task or set(recorded['arms']) != set(arms) or any(recorded['arms'][name]['instance'] != arm['instance'] or recorded['arms'][name]['slot'] != arm['slot'] for name, arm in arms.items()):
                raise ValueError('cleanup does not match this task plan')
            arms = recorded['arms']
        cleanup(arms, args.task, args.dry_run)
        return 0
    if not args.cards or not (args.matrix or args.interleave):
        parser.error('--cards and --matrix or --interleave required')
    cards = load_cards(args.kit.expanduser().resolve(), args.cards)
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
    build_arms(args.build, arms, build_cards, state, args.task, args.dry_run)
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
    jobs = [generate(card, name, arm, state, key_file, {**overrides.get(card['name'], {}), **arm_overrides.get(name, {}).get(card['name'], {})}, probes[card['name']], expected[card['name']], repeat) for card, name, arm, repeat in schedule(cards, arms, args.repeats if args.interleave else 1, args.interleave) if card['name'] not in bindings or bindings[card['name']] == name]
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
    env = {**os.environ, 'CUTEAFD_API_KEY': key_file.read_text().strip()}
    env.pop('CUTEAFD_BENCH_NONCE_SEED', None)
    if args.nonce_seed is not None:
        env['CUTEAFD_BENCH_NONCE_SEED'] = args.nonce_seed
    def execute(job):
        dest = Path(job['state'])
        try:
            assert_absent(job)
            for host in ('raptor', *job['entry']['sparks']):
                slot_check(arms[job['arm']], host)
        except (ValueError, subprocess.SubprocessError) as error:
            row = {k: job[k] for k in ('card', 'arm', 'repeat', 'simulated')}
            row.update(status='failed', exit_code=82, error=str(error))
            (dest / 'exit').write_text('82\n')
            save(dest / 'result.json', row)
            return row
        exit_file = dest / 'exit'
        if exit_file.exists():
            raise ValueError('run already exists; use a fresh task/state: ' + str(dest))
        job_env = dict(env)
        job_env.pop('CUTEAFD_BENCH_NONCE_SEED', None)
        if job.get('nonce_seed'):
            job_env['CUTEAFD_BENCH_NONCE_SEED'] = job['nonce_seed']
        process = detached(smoke_command(args.binary, args.repo, job), dest / 'driver.log', exit_file, job_env)
        rc = process.wait()
        code = int(exit_file.read_text()) if exit_file.exists() else (rc if rc >= 0 else 128 - rc)
        exit_file.write_text(str(code) + '\n')
        row = summarize_job(job, code)
        if row['status'] != 'pass' and not code:
            exit_file.write_text('81\n')
            row['exit_code'] = 81
        return row
    if groups and len(groups[0]) > 1:
        group_root = state / 'parallel-smoke'
        try:
            for job in jobs:
                assert_absent(job)
                for host in ('raptor', *job['entry']['sparks']):
                    slot_check(arms[job['arm']], host)
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
