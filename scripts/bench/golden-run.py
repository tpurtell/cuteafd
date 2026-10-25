#!/usr/bin/env python3
"""Run a family golden subcommand, a native test or a command from sealed WIP slots
on raptor's RTX GPUs, optionally against real expert workers on Sparks.

  scripts/bench/golden-run.py --task NAME --family dsv4|glm|glmf|mimo|qwen4|qwen35|step5 \\
      --model HF_ID_OR_SNAPSHOT_DIR --arm A=WIP_INSTANCE:SLOT [--arm B=...] \\
      [--arm-env A:KEY=VALUE] [--arm-arg A:TOKEN] [--env KEY=VALUE] [--gpus 0,1] [--repeats N] \\
      [--timeout SECONDS] [--preflight CMD] [--compare-logits A:B] \\
      [--native-test TEST_EXECUTABLE | --command] [--fixtures DIR [--fixtures-writable]] [--rdma] \\
      [--spark-hosts H1,H2,... [--expert-layers 3,5-7] [--worker-env KEY=VALUE] [--arm-worker-env A:KEY=VALUE] [--worker-arg ARG]] \\
      -- ARGS...

Each arm runs its slot's `cuteafd`, `libcuteafd_native.so`, `PROGRAMS.json` and expert
packages, mounted read-only as /opt/cuteafd/{bin,lib,share} in the slot's own dev image
(META base_image_id), with CDI `--device nvidia.com/gpu=N`, `--ipc host`, memlock
unlimited, the host uid (release_serving_docker_args), the full namespace read-only at /mnt/sparknest
(HF_HOME=/mnt/sparknest/hf-home). What runs:

- default: the family's golden subcommand (`glm-golden ARGS`, ...). The slot supplies
  --native-lib and --manifest; ARGS may not override them.
- --native-test EXE: a test executable built in the coordinator dev image from the
  slot's commit (e.g. `cargo test --no-run` via scripts/build/cuteafd-dev.sh), mounted
  read-only and run as `EXE ARGS` (e.g. `NAME --exact --ignored --nocapture`); green only
  when its `test result:` lines show at least one test passed and none failed.
- --command: ARGS verbatim (e.g. `python3 -m pytest -q python/tests/...`).
--rdma gives the container host networking and /dev/infiniband without Sparks (a
same-host verbs loopback test); --spark-hosts implies it.
Both raw modes run in the slot's sealed source tree, mounted read-only at
/workspace/cuteafd (the working directory, PYTHONPATH with the reference packages,
PYTHONDONTWRITEBYTECODE=1), with CUTEAFD_NATIVE_LIB and CUTEAFD_PROGRAM_MANIFEST set.

Placeholders in ARGS and in --env/--arm-env values: {snapshot} (the checkpoint's
container path), {hub} (the resolved host hub path), {out} (the run's writable output directory, e.g.
`--save-logits {out}/logits`), {fixtures}, {peers} and {expert_layers}. Other existing
absolute host paths in ARGS are mounted read-only at the same path. --fixtures DIR is
mounted at its own path (read-only; --fixtures-writable for a fixture-generation run,
which must write under ~/.cache/cuteafd/builds/TASK/ outside golden/ and never holds
Sparks) and exported as CUTEAFD_GOLDEN_FIXTURES.

Real experts (--spark-hosts, ranks in the given order): before every run the arm's
sealed spark-expert slot is checked on each host (seal, artifact checksums, SparkInfer
lock, the same source manifest as the coordinator slot, dev image, CDI, a free
--expert-port, no leftover container, the checkpoint visible), then each host runs the
serving path's `cuteafd expertd-native --rank R --world N` from that slot (read-only
mounts, no staging copy), bounded by an in-container timeout. The coordinator waits for
every worker's "native worker bootstrap ready" (--worker-ready-timeout; a worker that
exits fails the run at once), runs with host networking and /dev/infiniband, and gets
CUTEAFD_GOLDEN_PEERS (lane-A IP:PORT by rank, from --config), CUTEAFD_GOLDEN_SPARK_WORLD,
CUTEAFD_GOLDEN_EXPERT_LAYERS (expanded --expert-layers) and CUTEAFD_GOLDEN_SPARK_PACKAGES
(the workers' package manifest.json files, identical on every host, read-only under
/opt/cuteafd/spark). A golden subcommand must take them as `--peers {peers}`. Workers are
removed (logs kept) after every run, on failure, timeout and SIGTERM/SIGINT too, before
the locks are released. oliver (reserved) and the LAN-only build seeds are refused.

Hardware: the locks (sparks for dodo/emu, gpu0, gpu1, rhea, moa, in that order) are
taken once through the fair queue (scripts/hw/hwlock.py) and every arm and repeat runs
back to back inside that one turn, in interleaved order (B,A on even repeats), so A/B
pairs share conditions. Slots, images, Spark hosts and container names are checked
before enqueue and again at grant; --preflight adds a shell check at both points, like
hwlock's. Each container (golden-TASK-ARM) first checks it sees its GPUs (exit 86 when
not), is removed on any exit before the locks are released, and runs under a per-arm
timeout. --dry-run runs the checks and prints every docker and worker command without
taking locks.

Output, under ~/.cache/cuteafd/builds/TASK/golden/: ARM/rN/{log,exit,argv.json,
workers/HOST.log} and whatever the run wrote to {out}; argv.json holds the exact docker
and worker commands, image IDs, seal SHAs, env and the GPUs' power context (gpu-powerd
state, sampled power limit and draw). summary.md lists exit codes and the key log lines
(admission, logits digests, agreement/top-1, resume verdicts, test results, errors),
the workers' readiness and errors, plus each --compare-logits A:B result: byte
equality, then max |diff|, KL and top-1 over rows (vocab from the checkpoint's
config.json, or --vocab). Both layouts a golden writes are compared: a flat
`*logits*.bin` dump and the per-row safetensors directory (row-NNNNNNNN.safetensors
plus manifest.jsonl) that Kimi and the shared probe write; for the row layout the
table names the row carrying the largest KL.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUBCOMMANDS = {'dsv4': 'dsv4-golden', 'v41': 'v41-golden', 'glm': 'glm-golden', 'glmf': 'glmf-golden',
               'mimo': 'mimo-golden', 'qwen4': 'qwen4-golden', 'qwen35': 'qwen35-golden',
               'step5': 'step5-golden'}
# Families without a golden subcommand yet: native tests and commands only.
RAW_FAMILIES = ('kimi',)
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')
ROW_FILE = re.compile(r'row-[0-9]+\.safetensors\Z')
ENV_KEY = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
# Key result lines across the families' goldens.
KEY_LINE = re.compile(r'(?i)(admission|prefill logits|decode logits|logits digest|top-1|agreement|mean nll|mean kl|'
                      r'resume at|resume check|vs one straight|byte-identical|rows:|verify|layer +[0-9]+: cosine|'
                      r'test result|^test |^running [0-9]+ test|'
                      r'\berror\b|panicked|failed|refus|out of memory|golden-run:)')
TEST_RESULT = re.compile(r'test result: (\w+)\. (\d+) passed; (\d+) failed')
GPU_CHECK_EXIT = 86
# Golden flags that write files: only the arm's {out} directory is writable.
WRITERS = ('--save-logits', '--bench-dump', '--replay-dump', '--geometry-trace')
CONTAINER_HOME = '/tmp/cuteafd-home'
OUT = '/out'
SLOT_LIB, SLOT_BIN, SLOT_SHARE = '/opt/cuteafd/lib', '/opt/cuteafd/bin', '/opt/cuteafd/share'
ENTRYPOINT = '/opt/cuteafd/entry/release-entrypoint.sh'
POWER_STATE = Path('/run/gpu-powerd/state.json')
POWER_SAMPLE_S = 5.0
# Raw modes: the slot's sealed source tree, read-only, is the working directory.
SOURCE = '/workspace/cuteafd'
TEST_DIR = '/opt/cuteafd/test'
SPARK_PACKAGES = '/opt/cuteafd/spark'
def namespace_root():
    return str(Path(os.environ.get('CUTEAFD_NAMESPACE_ROOT', '/mnt/sparknest')).expanduser().resolve())


HUB = '/mnt/sparknest/hf-home/hub'
# Real experts. Hosts refused outright, with the reason (lift a refusal here once cleared).
REFUSED_HOSTS = {
    'oliver': 'reserved for the Kimi F32 golden',
    'kakapo': 'build seed for the fidelity goldens, not a serving rank',
    'cassowary': 'LAN-only build seed, not a serving rank',
    'raptor': 'the coordinator, not a Spark rank',
}
# Which hardware lock covers a Spark (scripts/hw/hwlock.py order; as `cuteafd bench smoke`).
HOST_LOCKS = {'ostrich': 'sparks', 'dodo': 'sparks', 'emu': 'sparks', 'kiwi': 'sparks', 'rhea': 'rhea', 'moa': 'moa'}
WORKER_READY = 'native worker bootstrap ready'
WORKER_LABEL = 'cuteafd.golden-run'
DEFAULT_EXPERT_PORT = 19451  # beside serving's 19441, so a stray serving worker is detected, not reused
DEFAULT_WORKER_BUDGET = 107374182400
# Forwarded to the coordinator and every worker when set (as run-family.sh does).
RDMA_ENV = ('CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP', 'CUTEAFD_VERBS_APP_IB_PORT_NUM',
            'CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hwlock = _load('hwlock', REPO / 'scripts/hw/hwlock.py')
cards = _load('wip_cards', REPO / 'scripts/bench/wip-cards.py')


class Interrupted(Exception):
    pass


def _interrupt(signum, _frame):
    raise Interrupted(f'signal {signum}')


def run(cmd, timeout=120, **kwargs):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=timeout, **kwargs)


# --------------------------------------------------------------------------- parsing

def parse_arms(values):
    arms = {}
    for value in values:
        name, sep, address = value.partition('=')
        instance, sep2, slot = address.partition(':')
        if not (sep and sep2) or not all(NAME.fullmatch(x) for x in (name, instance, slot)) \
                or len(instance) > 41 or len(slot) > 64:
            raise ValueError(f'invalid --arm {value!r} (NAME=WIP_INSTANCE:SLOT)')
        if name in arms:
            raise ValueError(f'duplicate arm {name}')
        arms[name] = {'instance': instance, 'slot': slot, 'env': {}, 'args': []}
    return arms


def parse_env(value):
    key, sep, val = value.partition('=')
    if not sep or not ENV_KEY.fullmatch(key) or '\n' in val:
        raise ValueError(f'invalid environment setting {value!r} (KEY=VALUE)')
    return key, val


def apply_env(arms, common, per_arm):
    for value in common:
        key, val = parse_env(value)
        for arm in arms.values():
            arm['env'][key] = val
    for value in per_arm:
        name, sep, setting = value.partition(':')
        if not sep or name not in arms:
            raise ValueError(f'--arm-env {value!r} names no arm (ARM:KEY=VALUE)')
        key, val = parse_env(setting)
        arms[name]['env'][key] = val


def parse_gpus(value):
    gpus = sorted({int(g) for g in value.split(',') if g.strip()})
    if not gpus or any(g not in (0, 1) for g in gpus):
        raise ValueError('--gpus takes raptor GPU indices 0 and/or 1')
    return gpus


def locks_for(gpus):
    return hwlock.canonical([f'gpu{g}' for g in gpus])


def schedule(arms, repeats, interleave=True):
    for repeat in range(1, repeats + 1):
        order = list(arms)
        if interleave and repeat % 2 == 0:
            order.reverse()
        for name in order:
            yield name, repeat


def golden_args(family, args, sparks=False):
    if family not in SUBCOMMANDS:
        raise ValueError(f'family {family} has no golden subcommand; use --native-test or --command')
    for i, arg in enumerate(args):
        flag, sep, value = arg.partition('=')
        if flag in ('--native-lib', '--manifest'):
            raise ValueError(f'{flag} comes from the slot; drop it from the golden args')
        if flag == '--peers' or flag.startswith('--sparks'):
            value = value if sep else (args[i + 1] if i + 1 < len(args) else '')
            if not sparks:
                raise ValueError('Spark peers come from --spark-hosts; without it run coordinator-only '
                                 '(e.g. --skip-routed-experts)')
            if flag != '--peers' or value != '{peers}':
                raise ValueError('with --spark-hosts pass the workers as `--peers {peers}`')
    if sparks and not any(a in ('--peers', '--peers={peers}') for a in args):
        raise ValueError('--spark-hosts starts workers the golden must use: add `--peers {peers}`')
    for flag, value in zip(args, args[1:]):
        if flag in WRITERS and not value.startswith('{out}'):
            raise ValueError(f'{flag} {value}: write outputs under the arm directory ({flag} {{out}}/NAME)')
    for arg in args:
        flag, sep, value = arg.partition('=')
        if sep and flag in WRITERS and not value.startswith('{out}'):
            raise ValueError(f'{arg}: write outputs under the arm directory ({flag}={{out}}/NAME)')
    return [SUBCOMMANDS[family], *args]


PLACEHOLDERS = ('snapshot', 'hub', 'out', 'fixtures', 'peers', 'expert_layers')


def substitute(args, snapshot, out=OUT, **values):
    """Replace {snapshot}, {hub}, {out}, {fixtures}, {peers} and {expert_layers}."""
    table = {'snapshot': snapshot, 'hub': HUB, 'out': out, **values}
    result = []
    for arg in args:
        for key in PLACEHOLDERS:
            token = '{' + key + '}'
            if token in arg:
                if table.get(key) is None:
                    raise ValueError(f'{token} in {arg!r} has no value in this run '
                                     f'({"--fixtures" if key == "fixtures" else "--spark-hosts"})')
                arg = arg.replace(token, table[key])
        result.append(arg)
    return result


def placeholder_values(state):
    return dict(hub=str(state['hub']), fixtures=state.get('fixtures'), peers=state['sparks']['peers_csv'] if state.get('sparks') else None,
                expert_layers=','.join(map(str, state['sparks']['layers'])) if state.get('sparks')
                and state['sparks']['layers'] else None)


def container_name(task, arm):
    return f'golden-{task}-{arm}'


def worker_name(task, arm, rank):
    return f'golden-{task}-{arm}-expert{rank}'


# --------------------------------------------------------------------------- Spark workers

def parse_hosts(value):
    hosts = [h.strip() for h in value.split(',') if h.strip()]
    if not hosts:
        raise ValueError('--spark-hosts names no host')
    if len(set(hosts)) != len(hosts):
        raise ValueError(f'--spark-hosts repeats a host: {value}')
    for host in hosts:
        if host in REFUSED_HOSTS:
            raise ValueError(f'--spark-hosts: {host} is refused ({REFUSED_HOSTS[host]})')
        if host not in HOST_LOCKS:
            raise ValueError(f'--spark-hosts: {host} is not a pool Spark ({", ".join(sorted(HOST_LOCKS))})')
    return hosts


def parse_layers(value):
    """'3,5-7' -> [3, 5, 6, 7] (sorted, unique)."""
    layers = set()
    for part in value.split(','):
        part = part.strip()
        first, sep, last = part.partition('-')
        if not first.isdigit() or (sep and not last.isdigit()):
            raise ValueError(f'invalid --expert-layers {value!r} (e.g. 3 or 3,5-7)')
        lo, hi = int(first), int(last) if sep else int(first)
        if hi < lo:
            raise ValueError(f'invalid --expert-layers range {part}')
        layers.update(range(lo, hi + 1))
    return sorted(layers)


def spark_lanes(config_path, hosts):
    """Each host's lane-A fabric address from the configuration's SPARK_i_HOST/SPARK_i_LANE_A."""
    values = cards.config(config_path)
    lanes = {}
    for key, host in values.items():
        match = re.fullmatch(r'SPARK_(\d+)_HOST', key)
        if match and values.get(f'SPARK_{match.group(1)}_LANE_A'):
            lanes[host] = values[f'SPARK_{match.group(1)}_LANE_A']
    missing = [h for h in hosts if h not in lanes]
    if missing:
        raise ValueError(f'{config_path} has no SPARK_i_HOST/SPARK_i_LANE_A for {", ".join(missing)}')
    for host in hosts:
        if not re.fullmatch(r'[0-9A-Fa-f.:]+', lanes[host]):
            raise ValueError(f'{config_path}: lane address of {host} is not an IP: {lanes[host]}')
    return [lanes[h] for h in hosts]


def locks_for(gpus, hosts=()):
    names = {f'gpu{g}' for g in gpus} | {HOST_LOCKS[h] for h in hosts}
    return hwlock.canonical(sorted(names))


def ssh(host, command, timeout=120, check=True):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command],
                          check=check, text=True, capture_output=True, timeout=timeout)


# Runs on each Spark (python3 -c, argv: root slot role instance image port worker_names_json
# snapshot_rel snapshot_abs): the sealed spark-expert slot, its artifacts, SparkInfer lock,
# dev image, CDI, a free port, no leftover worker, the checkpoint, the workers' serving
# identity and package manifests, as JSON.
REMOTE_CHECK = r'''import hashlib,json,os,pathlib,re,subprocess,sys
root,slot,instance,image,port,names,snap_rel,snap_abs=sys.argv[1:9]
home=pathlib.Path.home()
if root=='-':
    found=subprocess.run(['docker','inspect','-f','{{range .Mounts}}{{if eq .Destination "/wip"}}{{.Source}}{{end}}{{end}}','cuteafd-spark-expert-wip-'+instance],capture_output=True,text=True)
    root=found.stdout.strip() if found.returncode==0 and found.stdout.strip() else str(home/'.cache/cuteafd/builds'/('wip-'+instance))
p=pathlib.Path(root)/'slots'/slot/'spark-expert'
def fail(m):
    print(json.dumps({'error':m})); sys.exit(0)
try:
    meta=(p/'META.json').read_bytes()
except OSError as e:
    fail(f'no sealed spark-expert slot {instance}:{slot} under {root} ({e.strerror})')
if hashlib.sha256(meta).hexdigest()!=(p/'FINGERPRINT').read_text().strip(): fail('spark-expert META seal mismatch')
m=json.loads(meta)
if (m.get('slot'),m.get('role'),m.get('wip_instance'))!=(slot,'spark-expert',instance): fail('spark-expert slot identity mismatch')
art=p/'workspace/.cuteafd-wip'
for rel,key in (('SOURCE_SHA256SUMS','source_manifest_sha256'),('workspace/.cuteafd-wip/ARTIFACT_SHA256SUMS','artifact_manifest_sha256')):
    if hashlib.sha256((p/rel).read_bytes()).hexdigest()!=m.get(key): fail(rel+' seal mismatch')
listed={}
for line in (art/'ARTIFACT_SHA256SUMS').read_text().splitlines():
    if line.strip():
        d,n=line.split(None,1); listed[n.strip().removeprefix('./')]=d
for n in ('cuteafd','libcuteafd_native.so'):
    if n not in listed: fail('ARTIFACT_SHA256SUMS does not list '+n)
for n,d in listed.items():
    h=hashlib.sha256()
    with open(art/n,'rb') as f:
        while c:=f.read(1<<24): h.update(c)
    if h.hexdigest()!=d: fail(n+' does not match ARTIFACT_SHA256SUMS')
ws=p/'workspace'
for rel in ('docker/release-entrypoint.sh','third_party/sparkinfer','scripts/lib/release-common.sh'):
    if not (ws/rel).exists(): fail('slot lacks '+rel)
v=subprocess.run([sys.executable,str(ws/'scripts/build/verify-sparkinfer-source.py'),'--source',str(ws/'third_party/sparkinfer'),'--lock',str(ws/'third_party/sparkinfer.lock.json')],capture_output=True,text=True)
if v.returncode: fail('SparkInfer source does not match its lock: '+(v.stderr.strip().splitlines() or ['?'])[-1])
if m.get('base_image_id')!=image and image!='-': fail(f'slot dev image {m.get("base_image_id")} differs from the expected {image}')
i=subprocess.run(['docker','image','inspect','-f','{{.Id}}',m.get('base_image_id','')],capture_output=True,text=True)
if i.returncode or i.stdout.strip()!=m.get('base_image_id'): fail(f'dev image {m.get("base_image_id")} is missing')
if os.environ.get('CUTEAFD_CDI_CHECK')!='off':  # off: script tests only
    cdi=any(re.search(r'^\s*-?\s*name:\s*\W?all\W?\s*$',pathlib.Path(s).read_text(),re.M) for s in ('/var/run/cdi/nvidia.yaml','/etc/cdi/nvidia.yaml') if os.access(s,os.R_OK))
    if not cdi: fail('CDI device nvidia.com/gpu=all is not in /var/run/cdi/nvidia.yaml')
    if not os.path.exists('/dev/infiniband'): fail('no /dev/infiniband')
if port!='-':  # checked at grant: an earlier queued run may hold it until then
    s=subprocess.run(['ss','-Hltn','sport = :'+port],capture_output=True,text=True)
    if s.returncode==0 and s.stdout.strip(): fail('port '+port+' is already in use')
ps=subprocess.run(['docker','ps','-a','--format','{{.Names}}'],capture_output=True,text=True,check=True).stdout.split()
left=[n for n in json.loads(names) if n in ps]
if left: fail('container '+left[0]+' already exists (a previous run left it; inspect, save its logs and docker rm -f it)')
hub=os.path.realpath(home/'.cache/huggingface/hub')
snapshot=snap_abs if snap_rel=='-' else os.path.join(hub,snap_rel)
if not os.path.isfile(os.path.join(snapshot,'config.json')): fail('checkpoint not visible: '+snapshot)
sv=subprocess.run(['bash','-c','source "$1"; release_serving_docker_args; printf "%s\\0" "${RELEASE_SERVING_ARGS[@]}"','x',str(ws/'scripts/lib/release-common.sh')],capture_output=True)
if sv.returncode: fail('release_serving_docker_args failed: '+sv.stderr.decode(errors='replace').strip()[-300:])
packages={}
for kind in ('fp8','exl3'):
    for d in sorted((art/kind).iterdir()) if (art/kind).is_dir() else ():
        if (d/'manifest.json').is_file(): packages[f'{kind}/{d.name}']=(d/'manifest.json').read_text()
variants={k:sorted(x.name for x in (art/k).iterdir() if x.is_dir()) for k in ('bin-variants','lib-variants') if (art/k).is_dir()}
print(json.dumps({'root':str(p),'seal_sha256':hashlib.sha256(meta).hexdigest(),'source_manifest_sha256':m['source_manifest_sha256'],
  'artifact_manifest_sha256':m['artifact_manifest_sha256'],'artifacts':listed,'image_id':m['base_image_id'],'image':m.get('base_image'),
  'hub':hub,'snapshot':snapshot,'serving_args':sv.stdout.decode().rstrip('\0').split('\0'),'packages':packages,'variants':variants}))
'''


def check_spark(host, arm, state, coordinator, at_grant=True):
    """The arm's spark-expert slot on HOST (raises ValueError with the host named).
    The worker port is checked only at grant: before enqueue the run ahead of us
    in the fair queue may legitimately hold it."""
    root = coordinator.get('wip_root') or '-'
    snap = state['snapshot_container']
    snap_rel = '-'
    names = json.dumps([worker_name(state['task'], arm, r) for r in range(len(state['sparks']['hosts']))])
    argv = ['python3', '-c', REMOTE_CHECK, root, coordinator['slot'], coordinator['instance'], '-',
            str(state['sparks']['port']) if at_grant else '-', names, snap_rel, snap if snap_rel == '-' else '-']
    try:
        out = ssh(host, shlex.join(argv), timeout=600).stdout
        record = json.loads(out.strip().splitlines()[-1])
    except (subprocess.SubprocessError, json.JSONDecodeError, IndexError) as error:
        detail = (getattr(error, 'stderr', None) or str(error)).strip().splitlines()
        raise ValueError(f'{host}: Spark check failed: {detail[-1] if detail else error}') from None
    if record.get('error'):
        raise ValueError(f"{host}: {record['error']}")
    if record['source_manifest_sha256'] != coordinator['source_manifest_sha256']:
        raise ValueError(f"{host}: spark-expert slot {coordinator['instance']}:{coordinator['slot']} was built from "
                         'different sources than the coordinator slot')
    return record


def check_sparks(state, arm, coordinator, at_grant=True):
    records = {host: check_spark(host, arm, state, coordinator, at_grant) for host in state['sparks']['hosts']}
    first = next(iter(records.values()))
    for host, record in records.items():
        if record['packages'] != first['packages'] or record['artifact_manifest_sha256'] != first['artifact_manifest_sha256']:
            raise ValueError(f'{host}: spark-expert slot differs from {state["sparks"]["hosts"][0]}\'s '
                             '(artifacts or package manifests)')
    return records


def worker_command(state, arm, rank, record):
    """The serving path's expertd-native for one rank, from the sealed spark-expert slot."""
    root = Path(record['root'])
    art = root / 'workspace/.cuteafd-wip'
    sparks = state['sparks']
    mounts = [(art / 'cuteafd', f'{SLOT_BIN}/cuteafd'), (art / 'libcuteafd_native.so', f'{SLOT_LIB}/libcuteafd_native.so')]
    for kind, target in (('bin-variants', SLOT_BIN), ('lib-variants', SLOT_LIB)):
        for variant in record['variants'].get(kind, ()):
            mounts.append((art / kind / variant, f'{target}/{variant}'))
    for kind in sorted({p.split('/')[0] for p in record['packages']}):
        mounts.append((art / kind, f'{SLOT_LIB}/{kind}'))
    mounts += [(art, SLOT_SHARE), (root / 'workspace/docker/release-entrypoint.sh', ENTRYPOINT),
               (root / 'workspace/third_party/sparkinfer', '/source/third_party/sparkinfer'),
               (namespace_root(), namespace_root())]
    snapshot = record['snapshot']
    hub = Path(record['hub'])
    if not hub.is_relative_to(namespace_root()):
        mounts.append((hub, hub))
    if not Path(snapshot).is_relative_to(namespace_root()) and not Path(snapshot).is_relative_to(hub):
        mounts.append((snapshot, snapshot))
    cmd = ['docker', 'run', '-d', '--name', worker_name(state['task'], arm, rank), '--restart', 'no',
           '--label', f"{WORKER_LABEL}={state['task']}/{arm}", '--device', 'nvidia.com/gpu=all', '--network', 'host',
           '--ipc', 'host', '--ulimit', 'memlock=-1:-1', '--device=/dev/infiniband', *record['serving_args']]
    if sparks.get('memory'):  # a shared Spark's unified memory: cap the container, swap equally
        cmd += ['--memory', str(sparks['memory']), '--memory-swap', str(sparks['memory'])]
    for source, target in mounts:
        cmd += ['-v', f'{source}:{target}:ro']
    env = {'PATH': f'{SLOT_BIN}:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
           'PYTHONPATH': '/source/third_party/sparkinfer', 'CUTEAFD_NATIVE_LIB': f'{SLOT_LIB}/libcuteafd_native.so',
           # As the coordinator gets it: workers resolve installed programs and packages from it.
           'CUTEAFD_PROGRAM_MANIFEST': f'{SLOT_SHARE}/PROGRAMS.json',
           'RUST_LOG': 'info', 'HF_HOME': namespace_root() + '/hf-home',
           **{k: os.environ[k] for k in ('CUTEAFD_PUSH_SOCKET',) if os.environ.get(k)}, **{k: os.environ[k] for k in RDMA_ENV if os.environ.get(k)}, **sparks['env'],
           **sparks.get('arm_env', {}).get(arm, {})}
    for key, value in env.items():
        cmd += ['-e', f'{key}={value}']
    # Bounded by itself: a killed driver still frees the Spark after the run's budget.
    lifetime = state['timeout'] + sparks['ready_timeout'] + 120
    cmd += ['--entrypoint', 'timeout', record['image_id'], '--kill-after=30', str(lifetime), ENTRYPOINT,
            'cuteafd', 'expertd-native', '--snapshot', snapshot, '--native-lib', f'{SLOT_LIB}/libcuteafd_native.so',
            '--rank', str(rank), '--world', str(len(sparks['hosts'])), '--device-budget-bytes', str(sparks['budget']),
            '--listen', f"0.0.0.0:{sparks['port']}", *sparks['args']]
    return cmd


WORKER_STATE = ('n={name}; if docker logs "$n" 2>&1 | grep -qF {marker}; then echo ready; '
                'elif [ "$(docker inspect -f {{{{.State.Running}}}} "$n" 2>/dev/null)" = true ]; then echo starting; '
                'else echo exited; fi')
WORKER_TEARDOWN = ('n={name}; docker container inspect "$n" >/dev/null 2>&1 || exit 0; '
                   'l=$(docker inspect -f \'{{{{index .Config.Labels "{label}"}}}}\' "$n"); '
                   '[ "$l" = {value} ] || {{ echo "refusing to remove foreign container $n ($l)" >&2; exit 3; }}; '
                   'docker logs -t "$n" 2>&1; docker rm -f "$n" >/dev/null; '
                   '! docker container inspect "$n" >/dev/null 2>&1')


def start_worker(host, cmd):
    ssh(host, shlex.join(cmd), timeout=180)


def worker_state(host, name):
    try:
        out = ssh(host, WORKER_STATE.format(name=shlex.quote(name), marker=shlex.quote(WORKER_READY)), timeout=60).stdout
    except subprocess.SubprocessError:
        return 'unknown'
    return out.strip() or 'unknown'


def teardown_worker(host, name, label, log):
    """Save the worker's log and remove it (only with our label). Returns a problem or None."""
    command = WORKER_TEARDOWN.format(name=shlex.quote(name), label=WORKER_LABEL, value=shlex.quote(label))
    for attempt in range(3):
        try:
            proc = ssh(host, command, timeout=180, check=False)
        except subprocess.SubprocessError as error:
            problem = f'{host}: worker teardown failed: {error}'
            continue
        if proc.stdout:
            Path(log).write_text(proc.stdout)
        if proc.returncode == 0:
            return None
        problem = f'{host}: worker {name} not removed: {proc.stderr.strip() or proc.returncode}'
        if proc.returncode == 3:
            return problem
    return problem


def start_workers(state, arm, records, out_dir):
    """Start every rank, then wait for all to be ready; raises on exit or timeout."""
    hosts = state['sparks']['hosts']
    for rank, host in enumerate(hosts):
        print(f'golden-run: starting expert rank {rank} on {host}', flush=True)
        start_worker(host, worker_command(state, arm, rank, records[host]))
    deadline = time.monotonic() + state['sparks']['ready_timeout']
    pending = dict(enumerate(hosts))
    while pending:
        for rank, host in list(pending.items()):
            status = worker_state(host, worker_name(state['task'], arm, rank))
            if status == 'ready':
                del pending[rank]
            elif status == 'exited':
                raise RuntimeError(f'expert rank {rank} on {host} exited before readiness '
                                   f'(log: {out_dir}/workers/{host}.log)')
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"expert ranks {sorted(pending)} not ready after {state['sparks']['ready_timeout']} s")
        time.sleep(state['sparks']['poll_s'])
    print(f"golden-run: {len(hosts)} expert ranks ready", flush=True)


def stop_workers(state, arm, out_dir):
    workers = Path(out_dir) / 'workers'
    workers.mkdir(parents=True, exist_ok=True)
    problems = []
    for rank, host in enumerate(state['sparks']['hosts']):
        problem = teardown_worker(host, worker_name(state['task'], arm, rank), f"{state['task']}/{arm}",
                                  workers / f'{host}.log')
        if problem:
            problems.append(problem)
    return problems


# --------------------------------------------------------------------------- checkpoint

def hub_root():
    default = Path(namespace_root()) / 'hf-home/hub' if os.environ.get('CUTEAFD_NAMESPACE_ROOT') else Path.home() / '.cache/huggingface/hub'
    return Path(os.environ.get('CUTEAFD_HF_HUB', default)).resolve()


def resolve_model(model, hub=None):
    """(host snapshot dir, container path, extra read-only mounts)."""
    hub = hub or hub_root()
    path = Path(model).expanduser()
    if not path.is_absolute():
        repo_id, _, revision = model.partition('@')
        if repo_id.count('/') != 1:
            raise ValueError(f'--model {model!r} is neither a snapshot directory nor ORG/NAME[@REVISION]')
        root = hub / ('models--' + repo_id.replace('/', '--'))
        if not revision:
            ref = root / 'refs/main'
            if not ref.is_file():
                raise ValueError(f'{repo_id}: no {ref}; pass the snapshot directory')
            revision = ref.read_text().strip()
        path = root / 'snapshots' / revision
    path = path.resolve()
    if not (path / 'config.json').is_file():
        raise ValueError(f'{path} has no config.json')
    if path.is_relative_to(namespace_root()):
        return path, str(path), []
    return path, str(path), [str(path)]


def vocab_size(snapshot):
    config = json.loads((Path(snapshot) / 'config.json').read_text())
    for scope in (config, config.get('text_config') or {}, config.get('language_config') or {}):
        if isinstance(scope.get('vocab_size'), int):
            return scope['vocab_size']
    raise ValueError(f'{snapshot}/config.json has no vocab_size; pass --vocab')


def path_mounts(args, covered):
    """Existing absolute host paths in ARGS, outside the paths already mounted."""
    mounts = []
    for arg in args:
        value = arg.split('=', 1)[1] if arg.startswith('--') and '=' in arg else arg
        if not value.startswith('/') or value.startswith(OUT) or '{' in value:
            continue
        path = Path(value)
        if not path.exists() or any(path.resolve().is_relative_to(c) for c in covered):
            continue
        if str(path) not in mounts:
            mounts.append(str(path))
    return mounts


# --------------------------------------------------------------------------- slot

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def verify_slot(arm):
    """Seal (META/FINGERPRINT, source and artifact manifests, as wip-cards'
    slot_check), then every listed artifact's own sha256. Returns the record."""
    probe = {'instance': arm['instance'], 'slot': arm['slot']}
    if arm.get('root'):
        probe['root'] = arm['root']
    meta = json.loads(cards.slot_check(probe, 'raptor'))
    source = meta.get('seal_source', '')
    if not source.startswith('host:'):
        raise ValueError(f"{arm['instance']}:{arm['slot']}: slot is readable only inside its container ({source}); "
                         'rebuild it with ./wip.sh as the host user')
    root = Path(source[5:])
    artifacts = root / 'workspace/.cuteafd-wip'
    listed = {}
    for line in (artifacts / 'ARTIFACT_SHA256SUMS').read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            listed[name.strip().removeprefix('./')] = digest
    for name in ('cuteafd', 'libcuteafd_native.so', 'PROGRAMS.json'):
        if name not in listed:
            raise ValueError(f'{root}: ARTIFACT_SHA256SUMS does not list {name}')
    for name, digest in listed.items():
        if sha256_file(artifacts / name) != digest:
            raise ValueError(f'{root}: {name} does not match ARTIFACT_SHA256SUMS')
    for relative in ('workspace/docker/release-entrypoint.sh', 'workspace/third_party/sparkinfer'):
        if not (root / relative).exists():
            raise ValueError(f'{root}: slot lacks {relative}')
    identity = {}
    try:
        identity = json.loads((root / 'workspace/BUILD_IDENTITY.json').read_text())
    except (OSError, json.JSONDecodeError):
        pass
    image_id = meta.get('base_image_id')
    if not image_id:
        raise ValueError(f'{root}: META.json has no base_image_id')
    return dict(instance=arm['instance'], slot=arm['slot'], root=str(root), seal_sha256=meta['seal_sha256'],
                artifact_manifest_sha256=meta['artifact_manifest_sha256'], artifacts=listed,
                packages=(meta.get('contents') or {}).get('packages'), image=meta.get('base_image'),
                image_id=image_id, source_commit=identity.get('commit'), source_dirty=identity.get('dirty'),
                sparkinfer_revision=meta.get('sparkinfer_revision'), wip_root=meta.get('wip_root'),
                source_manifest_sha256=meta['source_manifest_sha256'])


def verify_sparkinfer(slot):
    workspace = Path(slot['root']) / 'workspace'
    run([sys.executable, str(workspace / 'scripts/build/verify-sparkinfer-source.py'),
         '--source', str(workspace / 'third_party/sparkinfer'),
         '--lock', str(workspace / 'third_party/sparkinfer.lock.json')])


def image_record(image_id):
    try:
        info = json.loads(run(['docker', 'image', 'inspect', image_id]).stdout)[0]
    except (subprocess.SubprocessError, IndexError, json.JSONDecodeError):
        raise ValueError(f'dev image {image_id} is missing (scripts/build/build-dev-images.sh, or rebuild the slot)') from None
    if info['Id'] != image_id:
        raise ValueError(f'dev image {image_id} resolved to {info["Id"]}')
    return dict(id=info['Id'], tags=info.get('RepoTags') or [])


def container_exists(name):
    names = run(['docker', 'ps', '-a', '--format', '{{.Names}}']).stdout.split()
    return name in names


def preflight(state, arms, at_grant=True):
    """CPU-only checks, before enqueue and again at grant: every slot sealed and
    unchanged, its image present, no container of ours left over and, with
    --spark-hosts, every Spark's spark-expert slot (see check_spark)."""
    records = {}
    for name, arm in arms.items():
        slot = verify_slot(arm)
        verify_sparkinfer(slot)
        slot['image_record'] = image_record(slot['image_id'])
        if container_exists(container_name(state['task'], name)):
            raise ValueError(f"container {container_name(state['task'], name)} already exists "
                             '(a previous run of this task left it; inspect and docker rm it)')
        if state.get('sparks'):
            slot['sparks'] = check_sparks(state, name, slot, at_grant)
        records[name] = slot
    previous = state.get('slots')
    if previous:
        for name, slot in records.items():
            if (slot['seal_sha256'], slot['artifacts']) != (previous[name]['seal_sha256'], previous[name]['artifacts']):
                raise ValueError(f'arm {name}: sealed slot changed since enqueue')
            for host, record in (slot.get('sparks') or {}).items():
                before = previous[name]['sparks'][host]
                if (record['seal_sha256'], record['artifacts']) != (before['seal_sha256'], before['artifacts']):
                    raise ValueError(f'arm {name}: sealed spark-expert slot on {host} changed since enqueue')
    return records


def spark_package_files(state, name, slot):
    """The workers' package manifests (identical on every host), written under the
    task for a read-only mount; returns {relative path: sha256}."""
    record = next(iter(slot['sparks'].values()))
    root = Path(state['root']) / name / 'spark-packages'
    digests = {}
    for package, text in sorted(record['packages'].items()):
        path = root / package / 'manifest.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        digests[f'{package}/manifest.json'] = hashlib.sha256(text.encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    return root, digests


# --------------------------------------------------------------------------- docker

def serving_args():
    """The shared serving identity (uid/gid, groups, USER, HOME, HF_HOME, caches)."""
    script = 'source scripts/lib/release-common.sh; release_serving_docker_args; printf "%s\\0" "${RELEASE_SERVING_ARGS[@]}"'
    out = subprocess.run(['bash', '-c', script], cwd=REPO, check=True, capture_output=True, timeout=60).stdout
    return out.decode().rstrip('\0').split('\0')


def docker_command(state, name, arm, slot, out_dir, golden, packages=None):
    root = Path(slot['root'])
    art = root / 'workspace/.cuteafd-wip'
    gpus = state['gpus']
    mode = state['mode']
    mounts = [(str(art / 'cuteafd'), f'{SLOT_BIN}/cuteafd'),
              (str(art / 'libcuteafd_native.so'), f'{SLOT_LIB}/libcuteafd_native.so')]
    for source, target in (('bin-variants', SLOT_BIN), ('lib-variants', SLOT_LIB)):
        variants = art / source
        if variants.is_dir():
            for variant in sorted(variants.iterdir()):
                if variant.is_dir():
                    mounts.append((str(variant), f'{target}/{variant.name}'))
    for package in ('exl3', 'fp8'):
        if (art / package).is_dir():
            mounts.append((str(art / package), f'{SLOT_LIB}/{package}'))
    mounts += [(str(art), SLOT_SHARE),
               (str(root / 'workspace/docker/release-entrypoint.sh'), ENTRYPOINT),
               (str(root / 'workspace/third_party/sparkinfer'), '/source/third_party/sparkinfer'),
               (namespace_root(), namespace_root())]
    mounts += [(p, p) for p in state['extra_mounts']]
    if mode != 'golden':
        mounts.append((str(root / 'workspace'), SOURCE))
    if mode == 'native-test':
        mounts.append((state['native_test'], f"{TEST_DIR}/{Path(state['native_test']).name}"))
    if state.get('fixtures') and not state.get('fixtures_writable'):
        mounts.append((state['fixtures'], state['fixtures']))
    if packages is not None:
        mounts.append((str(packages), SPARK_PACKAGES))
    cmd = ['docker', 'run', '--rm', '--name', container_name(state['task'], name),
           '--label', f"cuteafd.golden-run={state['task']}/{name}",
           *state['serving_args'], *cards.cdi_gpu_args(gpus), '--ipc', 'host',
           '--ulimit', 'memlock=-1:-1']
    if state.get('memory'):  # cap the coordinator container's host memory, swap equally
        cmd += ['--memory', str(state['memory']), '--memory-swap', str(state['memory'])]
    if state.get('sparks') or state.get('rdma'):
        cmd += ['--network', 'host', '--device=/dev/infiniband']
    for source, target in mounts:
        cmd += ['-v', f'{source}:{target}:ro']
    cmd += ['-v', f'{out_dir}:{OUT}']
    if state.get('fixtures_writable'):
        cmd += ['-v', f"{state['fixtures']}:{state['fixtures']}"]
    if mode != 'golden':
        cmd += ['-w', SOURCE]
    # The sealed program manifest in every mode: family goldens read it from the
    # environment as well as from --manifest.
    env = {'PATH': f'{SLOT_BIN}:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
           'PYTHONPATH': '/source/third_party/sparkinfer', 'CUTEAFD_NATIVE_LIB': f'{SLOT_LIB}/libcuteafd_native.so',
           'CUTEAFD_PROGRAM_MANIFEST': f'{SLOT_SHARE}/PROGRAMS.json', 'RUST_LOG': 'info',
           'HF_HOME': namespace_root() + '/hf-home'}
    if os.environ.get('CUTEAFD_PUSH_SOCKET'):
        env['CUTEAFD_PUSH_SOCKET'] = os.environ['CUTEAFD_PUSH_SOCKET']
    if mode != 'golden':
        env.update(PYTHONPATH=f'/source/third_party/sparkinfer:{SOURCE}/python/reference',
                   PYTHONDONTWRITEBYTECODE='1')
    env['CUTEAFD_GOLDEN_SNAPSHOT'] = state['snapshot_container']
    env['CUTEAFD_GOLDEN_OUT'] = OUT
    if state.get('fixtures'):
        env['CUTEAFD_GOLDEN_FIXTURES'] = state['fixtures']
    if state.get('sparks'):
        sparks = state['sparks']
        env.update(CUTEAFD_GOLDEN_PEERS=sparks['peers_csv'], CUTEAFD_GOLDEN_SPARK_WORLD=str(len(sparks['hosts'])),
                   CUTEAFD_GOLDEN_SPARK_HOSTS=','.join(sparks['hosts']))
        if sparks['layers']:
            env['CUTEAFD_GOLDEN_EXPERT_LAYERS'] = ','.join(map(str, sparks['layers']))
        if packages is not None:
            env['CUTEAFD_GOLDEN_SPARK_PACKAGES'] = SPARK_PACKAGES
        env.update({k: os.environ[k] for k in RDMA_ENV if os.environ.get(k)})
    if state.get('sparks'):
        env['CUTEAFD_SPARK_ADMISSION_TIMEOUT_SECONDS'] = str(state['sparks']['ready_timeout'])
    values = placeholder_values(state)
    for key, value in arm['env'].items():
        env[key] = substitute([value], state['snapshot_container'], **values)[0]
    for key, value in env.items():
        cmd += ['-e', f'{key}={value}']
    # The container first proves it sees its GPUs (a systemd daemon-reload on the
    # host strips GPU access silently), then execs the golden through the slot's
    # release entrypoint (library path for the native lib and CuTe runtime).
    check = (f'mkdir -p "$HOME"; n=$(nvidia-smi -L 2>&1 | grep -c "^GPU "); nvidia-smi -L >&2; '
             f'if [ "$n" -ne {len(gpus)} ]; then echo "golden-run: GPU access check failed: '
             f'saw $n of {len(gpus)} GPUs" >&2; exit {GPU_CHECK_EXIT}; fi; exec "$@"')
    program = (['cuteafd'] if mode == 'golden' else
               [f"{TEST_DIR}/{Path(state['native_test']).name}"] if mode == 'native-test' else [])
    cmd += ['--entrypoint', 'bash', slot['image_id'], '-c', check, 'golden-run', ENTRYPOINT, *program, *golden]
    return cmd, env


def remove_container(name):
    """Stop and remove our container; never one without our label."""
    try:
        found = run(['docker', 'ps', '-aq', '--filter', f'name=^{name}$']).stdout.split()
    except subprocess.SubprocessError as error:
        return f'could not list containers: {error}'
    for ident in found:
        labels = json.loads(run(['docker', 'inspect', '-f', '{{json .Config.Labels}}', ident]).stdout or 'null') or {}
        if 'cuteafd.golden-run' not in labels:
            return f'refusing to remove foreign container {name}'
        subprocess.run(['docker', 'rm', '-f', ident], capture_output=True, timeout=120)
    if container_exists(name):
        return f'container {name} is still present'
    return None


# --------------------------------------------------------------------------- power

def governor_summary():
    try:
        state = json.loads(POWER_STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return dict(source=str(POWER_STATE), updated=state.get('updated'), budget_w=state.get('budget_w'),
                reason=state.get('reason'),
                gpus=[dict(index=g.get('index'), limit_w=g.get('limit_w'), max_w=g.get('max_w'),
                           bound=(g.get('plan') or {}).get('bound'), reason=(g.get('plan') or {}).get('reason'))
                      for g in state.get('gpus', [])])


class PowerSampler:
    """nvidia-smi power limit/draw of the arm's GPUs every few seconds while it runs."""

    def __init__(self, gpus, period=POWER_SAMPLE_S):
        self.gpus, self.period, self.samples = gpus, period, []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while True:
            try:
                out = run(['nvidia-smi', '-i', ','.join(map(str, self.gpus)), '--format=csv,noheader,nounits',
                           '--query-gpu=index,power.limit,power.draw'], timeout=20).stdout
                for line in out.splitlines():
                    index, limit, draw = (x.strip() for x in line.split(','))
                    self.samples.append((int(index), float(limit), float(draw)))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            if self.stop.wait(self.period):
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=30)

    def summary(self):
        result = {}
        for gpu in self.gpus:
            rows = [(limit, draw) for index, limit, draw in self.samples if index == gpu]
            if rows:
                result[str(gpu)] = dict(samples=len(rows), mean_limit_w=round(sum(r[0] for r in rows) / len(rows), 1),
                                        min_limit_w=min(r[0] for r in rows), max_limit_w=max(r[0] for r in rows),
                                        mean_draw_w=round(sum(r[1] for r in rows) / len(rows), 1),
                                        max_draw_w=max(r[1] for r in rows))
        return result


# --------------------------------------------------------------------------- running

def run_args(state, arm=None):
    """The golden argv, plus ARM's own --arm-arg tokens appended."""
    return substitute(state['golden'] + (arm or {}).get('args', []), state['snapshot_container'],
                      **placeholder_values(state))


def run_arm(state, name, repeat, arms):
    arm, slot = arms[name], state['slots'][name]
    out_dir = Path(state['root']) / name / f'r{repeat}'
    out_dir.mkdir(parents=True, exist_ok=False)
    golden = run_args(state, arm)
    packages, package_digests = spark_package_files(state, name, slot) if state.get('sparks') else (None, None)
    cmd, env = docker_command(state, name, arm, slot, out_dir, golden, packages)
    workers = [dict(host=host, rank=rank, container=worker_name(state['task'], name, rank),
                    argv=worker_command(state, name, rank, slot['sparks'][host]),
                    seal_sha256=slot['sparks'][host]['seal_sha256'], image_id=slot['sparks'][host]['image_id'])
               for rank, host in enumerate(state['sparks']['hosts'])] if state.get('sparks') else []
    record = dict(arm=name, repeat=repeat, argv=cmd, golden=golden, container=container_name(state['task'], name),
                  mode=state['mode'], native_test=state.get('native_test'), native_test_sha256=state.get('native_test_sha256'),
                  fixtures=state.get('fixtures'), workers=workers, spark_packages=package_digests,
                  expert_layers=state['sparks']['layers'] if state.get('sparks') else None,
                  image=slot['image'], image_id=slot['image_id'], image_tags=slot['image_record']['tags'],
                  seal_sha256=slot['seal_sha256'], artifact_manifest_sha256=slot['artifact_manifest_sha256'],
                  slot=f"{slot['instance']}:{slot['slot']}", slot_root=slot['root'], source_commit=slot['source_commit'],
                  source_dirty=slot['source_dirty'], env=env, arm_env=arm['env'], arm_args=arm['args'], gpus=state['gpus'],
                  locks=state['locks'], snapshot=state['snapshot'], timeout_s=state['timeout'],
                  power_before=governor_summary())
    print(f"golden-run: {name} r{repeat} -> {out_dir}", flush=True)
    started = time.time()
    code, problem, proc, timed_out = None, None, None, False
    with open(out_dir / 'log', 'w') as log, PowerSampler(state['gpus']) as sampler:
        try:
            if workers:
                started_workers = time.time()
                start_workers(state, name, slot['sparks'], out_dir)
                record['workers_ready_s'] = round(time.time() - started_workers, 1)
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = proc.wait(timeout=state['timeout'])
        except subprocess.TimeoutExpired:
            problem, timed_out = f"timed out after {state['timeout']} s", True
        except (RuntimeError, subprocess.SubprocessError) as error:
            problem = f'expert workers: {error}'
        finally:
            removal = remove_container(record['container'])
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            # Workers go before the locks do, whatever happened (logs kept).
            teardown = stop_workers(state, name, out_dir) if workers else []
            problem = '; '.join(filter(None, [problem, removal, *teardown])) or None
    if code == GPU_CHECK_EXIT:
        problem = 'GPU access check failed inside the container'
    elif code == 0 and problem is None and state['mode'] == 'native-test':
        problem = test_verdict(out_dir / 'log')
    # A problem never leaves exit 0: timeouts are 124, other failures keep a nonzero code or become 1.
    exit_code = code if problem is None else 124 if timed_out else code if code not in (None, 0) else 1
    record.update(started=started, seconds=round(time.time() - started, 1), exit=exit_code, problem=problem,
                  power_after=governor_summary(),
                  power_samples=sampler.summary())
    (out_dir / 'exit').write_text(f"{record['exit']}\n")
    (out_dir / 'argv.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def test_verdict(log):
    """None when the test binary's summaries show tests ran and none failed."""
    try:
        text = ANSI.sub('', Path(log).read_text(errors='replace'))
    except OSError as error:
        return f'no test log: {error}'
    results = TEST_RESULT.findall(text)
    if not results:
        return 'no `test result:` summary in the log'
    if any(int(failed) for _, _, failed in results) or any(status != 'ok' for status, _, _ in results):
        return 'tests failed'
    if not sum(int(passed) for _, passed, _ in results):
        return 'no test ran (filter matched nothing, or the test is ignored without --ignored)'
    return None


def worker_lines(directory, limit=6):
    lines = []
    for log in sorted(Path(directory).glob('*.log')):
        text = ANSI.sub('', log.read_text(errors='replace'))
        ready = [l for l in text.splitlines() if WORKER_READY in l]
        # Error-level lines only: workers log `WARN ... peer closed` when the coordinator exits.
        errors = [l.strip()[:300] for l in text.splitlines()
                  if re.search(r'\bERROR\b|^\S*\s*[Ee]rror:|panicked|NV_ERR|out of memory', l)]
        lines.append(f"{log.stem}: {'ready' if ready else 'NOT ready'}; {len(errors)} error lines")
        lines += [f'  {l}' for l in errors[-limit:]]
    return lines


def key_lines(log, limit=16):
    try:
        text = ANSI.sub('', Path(log).read_text(errors='replace'))
    except OSError:
        return []
    lines = [line.strip() for line in text.splitlines() if KEY_LINE.search(line)]
    return [line[:400] for line in lines[-limit:]]


# --------------------------------------------------------------------------- comparison

def numpy_python():
    git_common = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                                capture_output=True, text=True)
    candidates = [Path(sys.executable), REPO / '.venv/bin/python']
    if git_common.returncode == 0:
        candidates.append(Path(git_common.stdout.strip()).parent / '.venv/bin/python')
    for python in candidates:
        if python.exists() and subprocess.run([str(python), '-c', 'import numpy'], capture_output=True).returncode == 0:
            return [str(python)]
    return None


def logits_files(directory):
    return sorted(str(p.relative_to(directory)) for p in Path(directory).rglob('*logits*.bin') if p.is_file())


def row_dump_dirs(directory):
    """Relative directories holding a per-row safetensors dump (row-NNNNNNNN.safetensors).

    Kimi and the shared probe write their logits this way: one F32 tensor per
    row plus manifest.jsonl, instead of a single flat `.bin` file.
    """
    return sorted({str(p.parent.relative_to(directory)) for p in Path(directory).rglob('row-*.safetensors')
                   if p.is_file() and ROW_FILE.fullmatch(p.name)})


def compare_pairs(state, pairs, repeats):
    """Compare saved logits of arm pairs, per repeat, file by file."""
    results = []
    if not pairs:
        return results
    python = numpy_python()
    tool = REPO / 'scripts/bench/compare-logits.py'
    for a, b in pairs:
        for repeat in range(1, repeats + 1):
            da, db = Path(state['root']) / a / f'r{repeat}', Path(state['root']) / b / f'r{repeat}'
            dumped = False
            files = [f for f in logits_files(da) if (db / f).is_file()]
            for name in files:
                dumped = True
                out = Path(state['root']) / f'compare-{a}-{b}-r{repeat}-{name.replace("/", "_")}.json'
                extra = []
                tokens = (da / name).parent / 'tokens.bin'
                if Path(name).name == 'logits.bin' and tokens.is_file():
                    extra = ['--tokens', str(tokens)]
                args = [str(da / name), str(db / name), str(state['vocab']), *extra, '--json', str(out)]
                results.append(compare_one(state, python, tool, args, out, dict(a=a, b=b, repeat=repeat, file=name)))
            # Per-row dumps: each directory of row-NNNNNNNN.safetensors compares as
            # one set, and the tool reports the row holding the largest KL.
            dirs_b = set(row_dump_dirs(db)) if db.is_dir() else set()
            for name in row_dump_dirs(da) if da.is_dir() else []:
                dumped = True
                if name not in dirs_b:
                    results.append(dict(a=a, b=b, repeat=repeat, file=name,
                                        error='row dump present in one arm only'))
                    continue
                out = Path(state['root']) / f'compare-{a}-{b}-r{repeat}-{name.replace("/", "_")}-rows.json'
                args = [str(da / name), str(db / name), '--json', str(out)]
                results.append(compare_one(state, python, tool, args, out, dict(a=a, b=b, repeat=repeat, file=name)))
            if not dumped:
                results.append(dict(a=a, b=b, repeat=repeat, file=None,
                                    error='no saved logits common to both arms'))
    return results


def compare_one(state, python, tool, args, out, record):
    """Run compare-logits.py once and fold its JSON into a summary row."""
    if python:
        cmd = [*python, str(tool), *args]
    else:  # the dev image has numpy; read-only views of the outputs
        image = state['slots'][record['a']]['image_id']
        cmd = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
               '-v', f'{tool}:{tool}:ro', '-v', f"{state['root']}:{state['root']}", '--entrypoint',
               'python3', image, str(tool), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode or not out.is_file():
        return dict(record, error=(proc.stderr.strip().splitlines() or ['compare failed'])[-1])
    measured = json.loads(out.read_text())
    return dict(measured, **record, path_a=measured['a'], path_b=measured['b'])


def write_summary(state, records, comparisons):
    lines = [f"# golden-run {state['task']}", '',
             f"family `{state['family']}`, checkpoint `{state['snapshot']}`, GPUs {state['gpus']}, locks "
             f"{','.join(state['locks'])}", '', f"golden args: `{shlex.join(state['golden'])}`", '',
             '| arm | repeat | exit | seconds | slot | seal | image | env | GPU limit W (mean) |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in records:
        env = ' '.join(f'{k}={v}' for k, v in r['arm_env'].items()) or '-'
        power = ', '.join(f"gpu{g} {p['mean_limit_w']:g}" for g, p in sorted(r['power_samples'].items())) or '-'
        exit_cell = str(r['exit']) + (f" ({r['problem']})" if r['problem'] else '')
        lines.append(f"| {r['arm']} | {r['repeat']} | {exit_cell} | {r['seconds']} | {r['slot']} | "
                     f"{r['seal_sha256'][:12]} | {r['image_id'][7:19]} | {env} | {power} |")
    if state.get('sparks'):
        sparks = state['sparks']
        lines += ['', f"Spark expert ranks: {', '.join(f'{r}={h}' for r, h in enumerate(sparks['hosts']))} "
                  f"(port {sparks['port']}, peers `{sparks['peers_csv']}`), expert layers "
                  f"{','.join(map(str, sparks['layers'])) or 'unspecified'}"]
    if state['mode'] != 'golden':
        lines += ['', f"mode `{state['mode']}`" + (f", test `{state['native_test']}` sha256 "
                                                    f"{state['native_test_sha256'][:12]}" if state.get('native_test') else '')]
    for r in records:
        run_dir = Path(state['root']) / r['arm'] / f"r{r['repeat']}"
        lines += ['', f"## {r['arm']} r{r['repeat']} (exit {r['exit']})", '', '```',
                  *(key_lines(run_dir / 'log') or ['(no key lines)']), '```']
        if r.get('workers'):
            lines += ['', f"workers (ready after {r.get('workers_ready_s', '-')} s):", '', '```',
                      *(worker_lines(run_dir / 'workers') or ['(no worker logs)']), '```']
    if comparisons:
        lines += ['', '## Logits comparisons', '', '| a | b | repeat | file | byte-identical | rows | max abs diff | mean KL | max KL | top-1 | worst row |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for c in comparisons:
            if c.get('error'):
                lines.append(f"| {c['a']} | {c['b']} | {c['repeat']} | {c['file'] or '-'} | error: {c['error']} | | | | | | |")
                continue
            worst = '-' if c['byte_identical'] or not c.get('worst_row') else c['worst_row']
            if not c['rows']:
                lines.append(f"| {c['a']} | {c['b']} | {c['repeat']} | {c['file']} | {c['byte_identical']} | 0 | | | | | {worst} |")
                continue
            lines.append(f"| {c['a']} | {c['b']} | {c['repeat']} | {c['file']} | {c['byte_identical']} | {c['rows']} | "
                         f"{c['max_abs_diff']:.4e} | {c['mean_kl']:.6f} | {c['max_kl']:.6f} | {100 * c['top1_agreement']:.2f}% | {worst} |")
    text = '\n'.join(lines) + '\n'
    (Path(state['root']) / 'summary.md').write_text(text)
    return text


# --------------------------------------------------------------------------- main

def resolve_fixtures(args, task_root):
    if not args.fixtures:
        if args.fixtures_writable:
            raise ValueError('--fixtures-writable needs --fixtures DIR')
        return None
    path = Path(args.fixtures).expanduser()
    if not path.is_absolute():
        raise ValueError('--fixtures takes an absolute directory')
    path = path.resolve()
    if args.fixtures_writable:
        if args.spark_hosts:
            raise ValueError('fixture generation runs GPU-only: drop --spark-hosts (the Sparks would idle)')
        if not path.is_relative_to(task_root) or path.is_relative_to(task_root / 'golden') or path == task_root:
            raise ValueError(f'--fixtures-writable directory must be under {task_root}/ and outside golden/')
        subprocess.run([sys.executable, str(REPO / 'scripts/build/assert-build-filesystem.py'), str(task_root.parent)],
                       check=True, capture_output=True, timeout=60)
    elif not path.is_dir():
        raise ValueError(f'--fixtures {path} is not a directory')
    if not re.fullmatch(r'/[A-Za-z0-9._/+-]+', str(path)):
        raise ValueError(f'--fixtures {path}: path characters must be [A-Za-z0-9._/+-]')
    return str(path)


def build_state(args, arms):
    task_root = Path.home() / '.cache/cuteafd/builds' / args.task
    snapshot, container_path, extra = resolve_model(args.model)
    hub = hub_root()
    sparks = None
    if args.spark_hosts:
        hosts = parse_hosts(args.spark_hosts)
        if args.worker_ready_timeout < 1:
            raise ValueError('--worker-ready-timeout must be positive seconds')
        if not 1 <= args.expert_port <= 65535:
            raise ValueError('--expert-port out of range')
        if args.worker_memory_bytes is not None and args.worker_memory_bytes <= args.worker_budget_bytes:
            raise ValueError('--worker-memory-bytes must exceed --worker-budget-bytes')
        lanes = spark_lanes(args.config, hosts)
        worker_env = {}
        for value in args.worker_env:
            key, val = parse_env(value)
            worker_env[key] = val
        arm_worker_env = {name: {} for name in arms}
        for value in args.arm_worker_env:
            name, sep, setting = value.partition(':')
            if not sep or name not in arms:
                raise ValueError(f'--arm-worker-env {value!r} names no arm (ARM:KEY=VALUE)')
            key, val = parse_env(setting)
            arm_worker_env[name][key] = val
        sparks = dict(hosts=hosts, lanes=lanes, port=args.expert_port,
                      peers_csv=','.join(f'{lane}:{args.expert_port}' for lane in lanes),
                      layers=parse_layers(args.expert_layers) if args.expert_layers else [],
                      env=worker_env, arm_env=arm_worker_env, args=list(args.worker_arg), budget=args.worker_budget_bytes,
                      memory=args.worker_memory_bytes,
                      ready_timeout=args.worker_ready_timeout, poll_s=args.worker_poll)
    elif args.expert_layers or args.worker_env or args.arm_worker_env or args.worker_arg or args.worker_memory_bytes is not None:
        raise ValueError('--expert-layers, --worker-env, --arm-worker-env, --worker-arg and --worker-memory-bytes need --spark-hosts')
    mode = 'native-test' if args.native_test else 'command' if args.command else 'golden'
    if mode == 'golden':
        golden = golden_args(args.family, args.golden, sparks=sparks is not None)
    else:
        golden = list(args.golden)
    native_test = native_digest = None
    if args.native_test:
        test = Path(args.native_test).expanduser()
        if not test.is_absolute() or not test.is_file() or not os.access(test, os.X_OK):
            raise ValueError(f'--native-test {args.native_test} is not an executable file (an absolute path)')
        native_test, native_digest = str(test.resolve()), sha256_file(test)
    fixtures = resolve_fixtures(args, task_root)
    if not hub.is_relative_to(namespace_root()):
        extra = [str(hub)] + [path for path in extra if not Path(path).is_relative_to(hub)]
    covered = [hub, Path(namespace_root()), snapshot] + ([Path(fixtures)] if fixtures else [])
    state = dict(task=args.task, family=args.family, root=str(task_root / 'golden'), gpus=parse_gpus(args.gpus), memory=args.memory_bytes,
                 snapshot=str(snapshot), snapshot_container=container_path, hub=str(hub), timeout=args.timeout,
                 golden=golden, extra_mounts=extra + path_mounts(golden, covered + [Path(p) for p in extra]),
                 mode=mode, native_test=native_test, native_test_sha256=native_digest, fixtures=fixtures,
                 fixtures_writable=bool(args.fixtures_writable), sparks=sparks,
                 rdma=bool(args.rdma))
    state['locks'] = locks_for(state['gpus'], sparks['hosts'] if sparks else ())
    state['vocab'] = args.vocab or (vocab_size(snapshot) if args.compare_logits else None)
    for arm in arms.values():
        run_args(state, arm)  # every placeholder resolves, per arm
    for arm in arms.values():
        for value in arm['env'].values():
            substitute([value], container_path, **placeholder_values(state))
    return state


def parse_pairs(values, arms):
    pairs = []
    for value in values:
        a, sep, b = value.partition(':')
        if not sep or a not in arms or b not in arms or a == b:
            raise ValueError(f'--compare-logits {value!r} needs two different arms (A:B)')
        pairs.append((a, b))
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, help='output under ~/.cache/cuteafd/builds/TASK/golden')
    parser.add_argument('--family', required=True, choices=sorted({*SUBCOMMANDS, *RAW_FAMILIES}),
                        help='family (its golden subcommand; raw-only families need --native-test/--command)')
    parser.add_argument('--model', required=True, help='HF id (ORG/NAME[@REV]) or snapshot directory; {snapshot} in ARGS')
    parser.add_argument('--arm', action='append', default=[], metavar='NAME=WIP_INSTANCE:SLOT')
    parser.add_argument('--arm-env', action='append', default=[], metavar='ARM:KEY=VALUE')
    parser.add_argument('--arm-arg', action='append', default=[], metavar='ARM:TOKEN',
                        help='one extra argv token appended for that arm only (repeatable, in order)')
    parser.add_argument('--env', action='append', default=[], metavar='KEY=VALUE', help='environment for every arm')
    parser.add_argument('--gpus', default='0,1', help='raptor GPUs, locked gpu0 then gpu1 (default 0,1)')
    raw = parser.add_mutually_exclusive_group()
    raw.add_argument('--native-test', metavar='EXE', help='run this test executable (built in the coordinator dev '
                     'image from the slot commit) with ARGS instead of a golden subcommand')
    raw.add_argument('--command', action='store_true', help='run ARGS verbatim in the slot source tree')
    parser.add_argument('--fixtures', metavar='DIR', help='fixture directory, mounted at its own path; {fixtures}')
    parser.add_argument('--fixtures-writable', action='store_true',
                        help='mount --fixtures writable (a GPU-only fixture-generation run under the task)')
    parser.add_argument('--rdma', action='store_true',
                        help='host networking and /dev/infiniband without Sparks (same-host verbs loopback)')
    parser.add_argument('--spark-hosts', metavar='H1,H2,...', help='start real expert workers, rank = position')
    parser.add_argument('--expert-layers', metavar='LIST', help='expert layers the run exercises (e.g. 3 or 3,5-7); '
                        'CUTEAFD_GOLDEN_EXPERT_LAYERS and {expert_layers}')
    parser.add_argument('--expert-port', type=int, default=DEFAULT_EXPERT_PORT)
    parser.add_argument('--worker-env', action='append', default=[], metavar='KEY=VALUE', help='worker environment')
    parser.add_argument('--arm-worker-env', action='append', default=[], metavar='ARM:KEY=VALUE',
                        help='worker environment for one arm (repeatable; over --worker-env)')
    parser.add_argument('--worker-arg', action='append', default=[], metavar='ARG',
                        help='extra expertd-native argument (repeatable; e.g. --worker-arg=--exl3-schedule)')
    parser.add_argument('--memory-bytes', type=int, metavar='N',
                        help='host memory cap for the coordinator container (docker --memory/--memory-swap)')
    parser.add_argument('--worker-budget-bytes', type=int, default=DEFAULT_WORKER_BUDGET)
    parser.add_argument('--worker-memory-bytes', type=int, metavar='N',
                        help='container memory cap per worker (docker --memory/--memory-swap); above the device budget')
    parser.add_argument('--worker-ready-timeout', type=int, default=1500, help='seconds for worker bootstrap and coordinator model admission')
    parser.add_argument('--worker-poll', type=float, default=5.0, help=argparse.SUPPRESS)
    parser.add_argument('--config', default=str(REPO / 'cuteafd.config'),
                        help='configuration with SPARK_i_HOST/SPARK_i_LANE_A (default cuteafd.config)')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--no-interleave', action='store_true', help='keep arm order on every repeat')
    parser.add_argument('--timeout', type=int, default=3600, help='seconds per arm (default 3600)')
    parser.add_argument('--wait', type=float, default=float(os.environ.get('CUTEAFD_HW_LOCK_WAIT_S', 21600)))
    parser.add_argument('--preflight', default='', help='shell check before enqueue and at grant (as hwlock)')
    parser.add_argument('--preflight-timeout', type=float, default=300)
    parser.add_argument('--preflight-at-grant', action='store_true',
                        help='run --preflight only once the locks are granted (free-memory floors and the like)')
    parser.add_argument('--compare-logits', action='append', default=[], metavar='A:B')
    parser.add_argument('--vocab', type=int, help='logits row width (default: the checkpoint config.json vocab_size)')
    parser.add_argument('--overwrite', action='store_true', help='replace an earlier golden/ of this task')
    parser.add_argument('--dry-run', action='store_true', help='run the checks, print the commands, take no locks')
    parser.add_argument('--preflight-only', action='store_true', help='run the CPU checks and exit (0 = pass)')
    parser.add_argument('golden', nargs=argparse.REMAINDER, help='-- GOLDEN ARGS (verbatim, after the subcommand)')
    args = parser.parse_args(argv)
    if args.golden[:1] == ['--']:
        args.golden = args.golden[1:]
    try:
        if not NAME.fullmatch(args.task) or args.repeats < 1 or args.timeout < 1:
            raise ValueError('invalid --task, --repeats or --timeout')
        if not args.golden and not args.native_test:
            raise ValueError('missing args after --')
        if args.family in RAW_FAMILIES and not (args.native_test or args.command):
            raise ValueError(f'family {args.family} has no golden subcommand yet; use --native-test or --command')
        arms = parse_arms(args.arm)
        if not arms:
            raise ValueError('at least one --arm is required')
        apply_env(arms, args.env, args.arm_env)
        for value in args.arm_arg:
            name, sep, token = value.partition(':')
            if not sep or name not in arms or not token or '\n' in token:
                raise ValueError(f'--arm-arg {value!r} names no arm (ARM:TOKEN)')
            arms[name]['args'].append(token)
        pairs = parse_pairs(args.compare_logits, arms)
        state = build_state(args, arms)
        for name in arms:
            if len(container_name(args.task, name)) > 128:
                raise ValueError('task and arm names make a container name over 128 characters')
        state['slots'] = preflight(state, arms, at_grant=False)
        state['serving_args'] = serving_args()
        if not args.preflight_at_grant:
            hwlock.check_preflight(args.preflight, args.preflight_timeout)
    except (ValueError, OSError, subprocess.SubprocessError, hwlock.PreflightFailed) as error:
        print(f'golden-run: {error}', file=sys.stderr)
        return 78 if isinstance(error, hwlock.PreflightFailed) else 2
    order = list(schedule(arms, args.repeats, not args.no_interleave))
    if args.preflight_only or args.dry_run:
        if args.dry_run:
            print(f"# locks (fair queue, in order): {','.join(state['locks'])}")
            for name, repeat in order:
                out = Path(state['root']) / name / f'r{repeat}'
                packages = Path(state['root']) / name / 'spark-packages' if state.get('sparks') else None
                for rank, host in enumerate(state['sparks']['hosts'] if state.get('sparks') else ()):
                    print(f'# {name} r{repeat} expert rank {rank} on {host}\n'
                          f"ssh {host} {shlex.quote(shlex.join(worker_command(state, name, rank, state['slots'][name]['sparks'][host])))}")
                cmd, _ = docker_command(state, name, arms[name], state['slots'][name], out, run_args(state, arms[name]), packages)
                print(f'# {name} r{repeat}\n{shlex.join(cmd)}')
        print('golden-run: preflight pass')
        return 0
    root = Path(state['root'])
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            print(f'golden-run: {root} exists; pass --overwrite or use another --task', file=sys.stderr)
            return 2
        import shutil
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    if state.get('fixtures_writable'):
        Path(state['fixtures']).mkdir(parents=True, exist_ok=True)  # ours, before Docker would create it as root
    (root / 'plan.json').write_text(json.dumps(dict(state, order=order, arms=arms), indent=2, default=str) + '\n')

    held = set(filter(None, os.environ.get('CUTEAFD_HW_LOCKS_HELD', '').split(',')))
    records = []
    previous = {signum: signal.signal(signum, _interrupt) for signum in (signal.SIGTERM, signal.SIGHUP)}

    def body():
        # At grant: the slots are still what was checked at enqueue.
        state['slots'] = preflight(state, arms)
        print(f"golden-run: locks {','.join(state['locks'])} held; {len(order)} runs", flush=True)
        for name, repeat in order:
            records.append(run_arm(state, name, repeat, arms))

    code = 0
    try:
        if set(state['locks']) <= held:
            body()  # an enclosing hwlock already holds our locks
        else:
            with hwlock.hold(state['locks'], args.wait, f'golden-run {args.task}', preflight=args.preflight,
                             preflight_timeout=args.preflight_timeout,
                             preflight_at_grant_only=args.preflight_at_grant):
                body()
    except hwlock.PreflightFailed as error:
        print(f'golden-run: {error}', file=sys.stderr)
        code = 78
    except TimeoutError as error:
        print(f'golden-run: {error}', file=sys.stderr)
        code = 75
    except (Interrupted, KeyboardInterrupt) as error:
        print(f'golden-run: interrupted ({error}); containers removed, locks released', file=sys.stderr)
        code = 130
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'golden-run: {error}', file=sys.stderr)
        code = 2
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    comparisons = compare_pairs(state, pairs, args.repeats) if records and len(records) == len(order) else []
    print(write_summary(state, records, comparisons))
    if code:
        return code
    if any(r['exit'] != 0 for r in records) or any(c.get('error') for c in comparisons):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
