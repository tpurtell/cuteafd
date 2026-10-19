import fcntl
import shlex
import signal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('golden_run', ROOT / 'scripts/bench/golden-run.py')
golden = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(golden)
COMPARE = ROOT / 'scripts/bench/compare-logits.py'
CSPEC = importlib.util.spec_from_file_location('compare_logits', COMPARE)
compare = importlib.util.module_from_spec(CSPEC)
CSPEC.loader.exec_module(compare)
IMAGE = 'sha256:' + 'ab' * 32

STUB_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
state = Path(os.environ['STUB_DOCKER_STATE'])
names = json.loads(state.read_text()) if state.exists() else {}
with open(os.environ['STUB_DOCKER_LOG'], 'a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\n')
args = sys.argv[1:]
def save():
    state.write_text(json.dumps(names))
host = os.environ.get('STUB_HOST', 'raptor')
def find(name):
    return next((i for i, n in names.items() if n['name'] == name or i == name), None)
if args[:2] == ['image', 'inspect']:
    if args[2] == '-f':
        print(args[4]); sys.exit(0)
    print(json.dumps([{'Id': args[2], 'RepoTags': ['cuteafd-coordinator-dev:test']}])); sys.exit(0)
if args[:2] == ['container', 'inspect']:
    sys.exit(0 if find(args[2]) else 1)
if args[:1] == ['logs']:
    ident = find(args[-1])
    print(names[ident].get('log', '') if ident else ''); sys.exit(0 if ident else 1)
if args[:2] == ['inspect', '-f']:
    ident = find(args[-1])
    if not ident:
        sys.exit(1)
    if '.State.Running' in args[2]:
        print('true' if names[ident].get('running') else 'false')
    elif 'index .Config.Labels' in args[2]:
        print(names[ident]['labels'].get('cuteafd.golden-run', ''))
    else:
        print(json.dumps(names[ident]['labels']))
    sys.exit(0)
if args[:2] == ['run', '-d']:
    name = args[args.index('--name') + 1]
    key, _, value = args[args.index('--label') + 1].partition('=')
    fail = os.environ.get('STUB_WORKER_FAIL') == host
    log = 'expert worker loading\n' + ('error: CUDA context creation failed NV_ERR_NO_MEMORY\n' if fail else
                                        'INFO native worker bootstrap ready rank=0\n'
                                        'WARN native RoCE peer removed error=native RoCE peer closed\n')
    if os.environ.get('STUB_WORKER_SLOW') == host:
        log = 'expert worker loading\n'
    names['id-' + name] = {'name': name, 'labels': {key: value}, 'log': log, 'running': not fail}; save()
    print('id-' + name); sys.exit(0)
if args[:1] == ['ps']:
    if '-q' in args[1] or '-aq' in args:
        wanted = args[args.index('--filter') + 1].split('=', 1)[1].strip('^$')
        print('\n'.join(ident for ident, n in names.items() if n['name'] == wanted))
    else:
        print('\n'.join(n['name'] for n in names.values()))
    sys.exit(0)
if args[:1] == ['inspect']:
    print(json.dumps(names[args[-1]]['labels'])); sys.exit(0)
if args[:2] == ['rm', '-f']:
    names.pop(find(args[2]), None); save(); sys.exit(0)
if args[:1] == ['run']:
    name = args[args.index('--name') + 1]
    labels = {}
    if '--label' in args:
        key, _, value = args[args.index('--label') + 1].partition('=')
        labels[key] = value
    ident = 'id-' + name
    names[ident] = {'name': name, 'labels': labels}; save()
    out = next(v.split(':')[0] for v in (args[i + 1] for i, a in enumerate(args) if a == '-v') if v.endswith(':/out'))
    env = dict(args[i + 1].split('=', 1) for i, a in enumerate(args) if a == '-e')
    print('GLM admission placement=attention ' + env.get('CUTEAFD_ATTENTION_PLACEMENT', 'auto'))
    print('prefill logits: 4 rows | mean NLL engine 1.0 | logits digest 00ff')
    if os.environ.get('STUB_RUN_OUTPUT'):
        print(os.environ['STUB_RUN_OUTPUT'])
    if os.environ.get('STUB_RUN_ENV'):
        Path(os.environ['STUB_RUN_ENV']).write_text(json.dumps(env))
    if os.environ.get('STUB_LOGITS'):
        Path(out, 'logits').mkdir(parents=True, exist_ok=True)
        Path(out, 'logits', 'logits.bin').write_bytes(Path(os.environ['STUB_LOGITS']).read_bytes())
    time.sleep(float(os.environ.get('STUB_RUN_SLEEP', '0')))
    code = int(os.environ.get('STUB_RUN_EXIT', '0'))
    if os.environ.get('STUB_LEAVE') != '1':
        names.pop(ident, None); save()
    sys.exit(code)
sys.exit(0)
'''


def write_slot(home, instance, slot='s1', tamper=False, variants=()):
    root = home / '.cache/cuteafd/builds' / f'wip-{instance}' / 'slots' / slot / 'coordinator'
    art = root / 'workspace/.cuteafd-wip'
    (art / 'fp8/fp8-glm').mkdir(parents=True)
    files = {'cuteafd': b'bin', 'libcuteafd_native.so': b'lib', 'PROGRAMS.json': b'{}'}
    for variant in variants:
        files[f'bin-variants/{variant}/cuteafd'] = b'variant-bin'
        files[f'lib-variants/{variant}/libcuteafd_native.so'] = b'variant-lib'
    for name, data in files.items():
        (art / name).parent.mkdir(parents=True, exist_ok=True)
        (art / name).write_bytes(data)
    sums = ''.join(f'{hashlib.sha256(d).hexdigest()}  {n}\n' for n, d in files.items()).encode()
    (art / 'ARTIFACT_SHA256SUMS').write_bytes(sums)
    (root / 'workspace/docker').mkdir(parents=True)
    (root / 'workspace/docker/release-entrypoint.sh').write_text('#!/bin/sh\nexec "$@"\n')
    (root / 'workspace/third_party/sparkinfer').mkdir(parents=True)
    (root / 'workspace/third_party/sparkinfer.lock.json').write_text('{}')
    (root / 'workspace/scripts/build').mkdir(parents=True)
    (root / 'workspace/scripts/build/verify-sparkinfer-source.py').write_text('print("verified")\n')
    (root / 'workspace/BUILD_IDENTITY.json').write_text(json.dumps({'commit': 'c0ffee', 'dirty': False}))
    source = b'abc  ./x\n'
    (root / 'SOURCE_SHA256SUMS').write_bytes(source)
    meta = json.dumps({'slot': slot, 'role': 'coordinator', 'wip_instance': instance, 'base_image': 'dev:tc',
                       'base_image_id': IMAGE, 'source_manifest_sha256': hashlib.sha256(source).hexdigest(),
                       'artifact_manifest_sha256': hashlib.sha256(sums).hexdigest()}).encode()
    (root / 'META.json').write_bytes(meta)
    (root / 'FINGERPRINT').write_text(hashlib.sha256(meta + (b'x' if tamper else b'')).hexdigest() + '\n')
    return root


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    (bindir / 'docker').write_text(STUB_DOCKER)
    (bindir / 'docker').chmod(0o755)
    hub = tmp_path / 'hub'
    snap = hub / 'models--org--model/snapshots/rev1'
    snap.mkdir(parents=True)
    (snap / 'config.json').write_text(json.dumps({'vocab_size': 8}))
    (hub / 'models--org--model/refs').mkdir()
    (hub / 'models--org--model/refs/main').write_text('rev1\n')
    locks = tmp_path / 'locks'
    locks.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('PATH', f'{bindir}:{os.environ["PATH"]}')
    monkeypatch.setenv('STUB_DOCKER_STATE', str(tmp_path / 'docker-state.json'))
    monkeypatch.setenv('STUB_DOCKER_LOG', str(tmp_path / 'docker.log'))
    monkeypatch.setenv('CUTEAFD_HF_HUB', str(hub))
    monkeypatch.delenv('CUTEAFD_HW_LOCKS_HELD', raising=False)
    monkeypatch.delenv('CUTEAFD_HW_GATE_TICKETS', raising=False)
    monkeypatch.setattr(golden.hwlock, 'LOCK_DIR', locks)
    monkeypatch.setattr(golden.PowerSampler, '_loop', lambda self: None)
    monkeypatch.setattr(golden, 'POWER_STATE', tmp_path / 'no-powerd.json')
    return dict(home=home, tmp=tmp_path, locks=locks, log=tmp_path / 'docker.log', state=tmp_path / 'docker-state.json')


def docker_calls(env):
    return [json.loads(line) for line in env['log'].read_text().splitlines()] if env['log'].exists() else []


def runs(env):
    return [c for c in docker_calls(env) if c[:1] == ['run']]


BASE = ['--task', 'gr', '--family', 'glm', '--model', 'org/model']
GOLDEN = ['--', '--snapshot', '{snapshot}', '--device', '0', '--split-device', '1', '--skip-routed-experts', '--nll']


@pytest.mark.parametrize('family,sub', sorted(golden.SUBCOMMANDS.items()))
def test_argv_per_family_passes_args_verbatim(family, sub):
    args = ['--snapshot', '{snapshot}', '--golden', '/g', '--resume-at', '100,256', '--save-logits', '{out}/l']
    built = golden.substitute(golden.golden_args(family, args), '/hf/hub/m')
    assert built == [sub, '--snapshot', '/hf/hub/m', '--golden', '/g', '--resume-at', '100,256', '--save-logits', '/out/l']


@pytest.mark.parametrize('args,match', [
    (['--native-lib', '/x.so'], 'comes from the slot'),
    (['--manifest=/p.json'], 'comes from the slot'),
    (['--peers', 'ostrich:1'], 'come from --spark-hosts'),
    (['--save-logits', '/tmp/x'], 'under the arm directory'),
    (['--save-logits=/tmp/x'], 'under the arm directory'),
])
def test_argv_refuses_slot_spark_and_unwritable_paths(args, match):
    with pytest.raises(ValueError, match=match):
        golden.golden_args('dsv4', args)


def test_lock_order_is_canonical_gpu0_then_gpu1():
    assert golden.locks_for(golden.parse_gpus('1,0')) == ['gpu0', 'gpu1']
    assert golden.locks_for(golden.parse_gpus('1')) == ['gpu1']
    with pytest.raises(ValueError):
        golden.parse_gpus('2')


def test_interleaved_schedule():
    arms = {'a': {}, 'b': {}}
    assert list(golden.schedule(arms, 3)) == [('a', 1), ('b', 1), ('b', 2), ('a', 2), ('a', 3), ('b', 3)]
    assert list(golden.schedule(arms, 2, interleave=False)) == [('a', 1), ('b', 1), ('a', 2), ('b', 2)]




def test_unsealed_slot_is_refused_before_any_lock(env, capsys, monkeypatch):
    write_slot(env['home'], 'i1', tamper=True)
    taken = []
    monkeypatch.setattr(golden.hwlock, 'hold', lambda *a, **k: taken.append(a))
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 2
    assert 'unsealed' in capsys.readouterr().err
    assert not taken and not runs(env)


def test_tampered_artifact_is_refused(env, capsys):
    root = write_slot(env['home'], 'i1')
    (root / 'workspace/.cuteafd-wip/libcuteafd_native.so').write_bytes(b'other')
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 2
    assert 'does not match ARTIFACT_SHA256SUMS' in capsys.readouterr().err


@pytest.mark.parametrize('variants', [(), ('x86-64-v4',)])
def test_host_isa_variants_are_mounted_and_overrides_forwarded(env, variants):
    root = write_slot(env['home'], 'i1', variants=variants)
    argv = BASE + ['--arm', 'v2=i1:s1', '--arm', 'v4=i1:s1',
                   '--arm-env', 'v2:CUTEAFD_HOST_ISA=x86-64-v2',
                   '--arm-env', 'v4:CUTEAFD_HOST_ISA=x86-64-v4'] + GOLDEN
    assert golden.main(argv) == 0
    for call, isa in zip(runs(env), ('x86-64-v2', 'x86-64-v4')):
        envs = [call[i + 1] for i, arg in enumerate(call) if arg == '-e']
        assert f'CUTEAFD_HOST_ISA={isa}' in envs
        mounts = [call[i + 1] for i, arg in enumerate(call) if arg == '-v']
        variant_mounts = [mount for mount in mounts if '-variants/' in mount]
        assert len(variant_mounts) == 2 * len(variants)
        for variant in variants:
            art = root / 'workspace/.cuteafd-wip'
            assert f'{art}/bin-variants/{variant}:/opt/cuteafd/bin/{variant}:ro' in mounts
            assert f'{art}/lib-variants/{variant}:/opt/cuteafd/lib/{variant}:ro' in mounts


def test_env_per_arm_and_container_mounts(env, monkeypatch):
    monkeypatch.setattr(golden.cards._cdi, 'cdi_names', lambda specs=(): {'0', '1', 'all'})
    write_slot(env['home'], 'i1')
    order = []
    real_hold = golden.hwlock.hold

    def hold(names, *args, **kwargs):
        order.append(list(names))
        return real_hold(names, *args, **kwargs)
    monkeypatch.setattr(golden.hwlock, 'hold', hold)
    argv = BASE + ['--arm', 'heads=i1:s1', '--arm', 'context=i1:s1', '--env', 'RUST_LOG=warn',
                   '--arm-env', 'heads:CUTEAFD_ATTENTION_PLACEMENT=heads',
                   '--arm-env', 'context:CUTEAFD_ATTENTION_PLACEMENT=context', '--arm-arg', 'heads:--trace',
                   '--arm-arg', 'heads:{out}/t', '--gpus', '1,0'] + GOLDEN
    assert golden.main(argv) == 0
    assert order == [['gpu0', 'gpu1']]
    calls = runs(env)
    assert [c[c.index('--name') + 1] for c in calls] == ['golden-gr-heads', 'golden-gr-context']
    for call, mode in zip(calls, ['heads', 'context']):
        envs = [call[i + 1] for i, a in enumerate(call) if a == '-e']
        assert f'CUTEAFD_ATTENTION_PLACEMENT={mode}' in envs and 'RUST_LOG=warn' in envs
        assert 'CUTEAFD_PROGRAM_MANIFEST=/opt/cuteafd/share/PROGRAMS.json' in envs
        assert call[call.index('--user') + 1] == f'{os.getuid()}:{os.getgid()}'
        assert '--gpus' not in call
        assert [call[i + 1] for i, a in enumerate(call) if a == '--device'
                and call[i + 1].startswith('nvidia.com/gpu=')] == ['nvidia.com/gpu=0', 'nvidia.com/gpu=1']
        assert call[call.index('--ipc') + 1] == 'host' and 'memlock=-1:-1' in call
        mounts = [call[i + 1] for i, a in enumerate(call) if a == '-v']
        assert f"{env['tmp']}/hub:/hf/hub:ro" in mounts
        assert any(m.endswith(':/opt/cuteafd/lib/libcuteafd_native.so:ro') for m in mounts)
        assert any(m.endswith(':/opt/cuteafd/share:ro') for m in mounts)
        assert any(m.endswith(f'/gr/golden/{mode}/r1:/out') for m in mounts)
        assert IMAGE in call
        tail = call[call.index(IMAGE) + 1:]
        assert (tail[-2:] == ['--trace', '/out/t']) == (mode == 'heads')
        if mode == 'heads':
            tail = tail[:-2]
        assert tail[-8:] == ['glm-golden', '--snapshot', '/hf/hub/models--org--model/snapshots/rev1', '--device', '0',
                             '--split-device', '1', '--skip-routed-experts', '--nll'][-8:]
    out = env['home'] / '.cache/cuteafd/builds/gr/golden'
    record = json.loads((out / 'heads/r1/argv.json').read_text())
    assert record['arm_env'] == {'RUST_LOG': 'warn', 'CUTEAFD_ATTENTION_PLACEMENT': 'heads'}
    assert record['arm_args'] == ['--trace', '{out}/t']
    assert record['image_id'] == IMAGE and len(record['seal_sha256']) == 64
    assert record['source_commit'] == 'c0ffee'
    assert 'power_before' in record and 'power_samples' in record
    assert (out / 'heads/r1/exit').read_text() == '0\n'
    summary = (out / 'summary.md').read_text()
    assert 'attention heads' in summary and 'logits digest 00ff' in summary
    # Locks are free again afterwards.
    for name in ('gpu0', 'gpu1'):
        with open(env['locks'] / f'{name}.lock', 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_failure_removes_container_before_releasing_locks(env, monkeypatch):
    write_slot(env['home'], 'i1')
    monkeypatch.setenv('STUB_RUN_EXIT', '3')
    monkeypatch.setenv('STUB_LEAVE', '1')  # as if --rm had not cleaned up
    released = []
    real_hold = golden.hwlock.hold

    class Hold:
        def __init__(self, *a, **k):
            self.inner = real_hold(*a, **k)

        def __enter__(self):
            return self.inner.__enter__()

        def __exit__(self, *exc):
            released.append(json.loads(env['state'].read_text()))
            return self.inner.__exit__(*exc)
    monkeypatch.setattr(golden.hwlock, 'hold', Hold)
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 1
    assert released == [{}]
    assert any(c[:2] == ['rm', '-f'] for c in docker_calls(env))
    assert (env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/exit').read_text() == '3\n'


def test_timeout_kills_and_removes(env, monkeypatch):
    write_slot(env['home'], 'i1')
    monkeypatch.setenv('STUB_RUN_SLEEP', '30')
    assert golden.main(BASE + ['--arm', 'a=i1:s1', '--timeout', '1'] + GOLDEN) == 1
    record = json.loads((env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/argv.json').read_text())
    assert record['exit'] == 124 and 'timed out' in record['problem']
    assert json.loads(env['state'].read_text()) == {}


def test_gpu_access_failure_is_named(env, monkeypatch):
    write_slot(env['home'], 'i1')
    monkeypatch.setenv('STUB_RUN_EXIT', str(golden.GPU_CHECK_EXIT))
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 1
    record = json.loads((env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/argv.json').read_text())
    assert record['problem'] == 'GPU access check failed inside the container'


def test_foreign_container_is_refused(env, capsys):
    write_slot(env['home'], 'i1')
    env['state'].write_text(json.dumps({'id-x': {'name': 'golden-gr-a', 'labels': {}}}))
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 2
    assert 'already exists' in capsys.readouterr().err
    assert not runs(env)


def test_failed_preflight_hook_runs_nothing(env):
    write_slot(env['home'], 'i1')
    assert golden.main(BASE + ['--arm', 'a=i1:s1', '--preflight', 'exit 3'] + GOLDEN) == 78
    assert not runs(env)


def test_compare_logits_byte_identical_arms(env, monkeypatch):
    write_slot(env['home'], 'i1')
    logits = env['tmp'] / 'logits.bin'
    np.arange(32, dtype='<f4').tofile(logits)
    monkeypatch.setenv('STUB_LOGITS', str(logits))
    argv = BASE + ['--arm', 'a=i1:s1', '--arm', 'b=i1:s1', '--compare-logits', 'a:b'] + GOLDEN + \
        ['--save-logits', '{out}/logits']
    assert golden.main(argv) == 0
    summary = (env['home'] / '.cache/cuteafd/builds/gr/golden/summary.md').read_text()
    assert '| a | b | 1 | logits/logits.bin | True | 4 | 0.0000e+00 |' in summary


def test_compare_tool_on_synthetic_logits(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(130, 50)).astype('<f4')
    b = a.copy()
    b[7] += 0.5 * rng.normal(size=50).astype('<f4')
    b[100, a[100].argmax()] -= 100
    pa, pb, tokens = tmp_path / 'a.bin', tmp_path / 'b.bin', tmp_path / 'tokens.bin'
    a.tofile(pa)
    b.tofile(pb)
    rng.integers(0, 50, size=131).astype('<u4').tofile(tokens)
    same = compare.compare(pa, pa, 50)
    assert same['byte_identical'] and same['max_abs_diff'] == 0 and same['mean_kl'] == 0 and same['top1_agreement'] == 1
    diff = compare.compare(pa, pb, 50, tokens)
    assert not diff['byte_identical'] and diff['rows'] == 130
    assert diff['max_abs_diff'] == pytest.approx(100, rel=1e-5)
    assert diff['top1_agreement'] < 1 and diff['top1_agreement'] >= 128 / 130
    assert diff['max_kl'] > diff['mean_kl'] > 0
    assert diff['nll_rows'] == 130 and diff['nll_a'] > 0
    with pytest.raises(ValueError):
        compare.compare(pa, pb, 7)
    cli = subprocess.run([sys.executable, str(COMPARE), str(pa), str(pa), '50'], capture_output=True, text=True)
    assert cli.returncode == 0 and 'byte-identical: True' in cli.stdout


def test_resolve_model_uses_hub_paths(env):
    host, container, extra = golden.resolve_model('org/model')
    assert container == '/hf/hub/models--org--model/snapshots/rev1' and not extra
    assert golden.vocab_size(host) == 8


# --------------------------------------------------------------------------- real Spark experts

STUB_SSH = r'''#!/usr/bin/env bash
# ssh -o ... HOST COMMAND: run COMMAND locally as that host (docker stub sees STUB_HOST).
while [[ "$1" == -o ]]; do shift 2; done
host="$1"; shift
python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$host" "$*" >> "$STUB_SSH_LOG"
export STUB_HOST="$host"
exec bash -c "$*"
'''

SPARKS = ['dodo', 'emu', 'rhea', 'moa']
SPARK_BASE = ['--task', 'gr', '--family', 'step5']
CONFIG = ''.join(f'SPARK_{i}_HOST={h}\nSPARK_{i}_LANE_A=10.9.0.{i + 1}\n' for i, h in enumerate(SPARKS + ['ostrich']))


def write_spark_slot(home, instance, slot='s1', source=b'abc  ./x\n', packages=None):
    root = home / '.cache/cuteafd/builds' / f'wip-{instance}' / 'slots' / slot / 'spark-expert'
    art = root / 'workspace/.cuteafd-wip'
    art.mkdir(parents=True)
    files = {'cuteafd': b'arm-bin', 'libcuteafd_native.so': b'arm-lib'}
    for name, data in files.items():
        (art / name).write_bytes(data)
    for package, text in (packages or {'fp8/fp8-step5': '{"layouts": {"tp4": {"capacity": 64}}}'}).items():
        (art / package).mkdir(parents=True)
        (art / package / 'manifest.json').write_text(text)
    sums = ''.join(f'{hashlib.sha256(d).hexdigest()}  {n}\n' for n, d in files.items()).encode()
    (art / 'ARTIFACT_SHA256SUMS').write_bytes(sums)
    ws = root / 'workspace'
    (ws / 'docker').mkdir()
    (ws / 'docker/release-entrypoint.sh').write_text('#!/bin/sh\nexec "$@"\n')
    (ws / 'third_party/sparkinfer').mkdir(parents=True)
    (ws / 'third_party/sparkinfer.lock.json').write_text('{}')
    (ws / 'scripts/build').mkdir(parents=True)
    (ws / 'scripts/build/verify-sparkinfer-source.py').write_text('print("verified")\n')
    (ws / 'scripts/lib').mkdir(parents=True)
    (ws / 'scripts/lib/release-common.sh').write_text(
        'release_serving_docker_args() { RELEASE_SERVING_ARGS=(--user 1001:1001 -e HF_HOME=/hf); }\n')
    (root / 'SOURCE_SHA256SUMS').write_bytes(source)
    meta = json.dumps({'slot': slot, 'role': 'spark-expert', 'wip_instance': instance, 'base_image': 'spark:tc',
                       'base_image_id': 'sha256:' + 'cd' * 32, 'source_manifest_sha256': hashlib.sha256(source).hexdigest(),
                       'artifact_manifest_sha256': hashlib.sha256(sums).hexdigest()}).encode()
    (root / 'META.json').write_bytes(meta)
    (root / 'FINGERPRINT').write_text(hashlib.sha256(meta).hexdigest() + '\n')
    return root


@pytest.fixture
def spark_env(env, monkeypatch):
    bindir = env['tmp'] / 'bin'
    (bindir / 'ssh').write_text(STUB_SSH)
    (bindir / 'ssh').chmod(0o755)
    (env['home'] / '.cache/huggingface').mkdir(parents=True)
    (env['home'] / '.cache/huggingface/hub').symlink_to(env['tmp'] / 'hub')
    config = env['tmp'] / 'cuteafd.config'
    config.write_text(CONFIG)
    monkeypatch.setenv('STUB_SSH_LOG', str(env['tmp'] / 'ssh.log'))
    monkeypatch.setattr(golden.cards._cdi, 'cdi_names', lambda specs=(): {'0', '1', 'all'})
    write_slot(env['home'], 'i1')
    write_spark_slot(env['home'], 'i1')
    return dict(env, config=config, ssh=env['tmp'] / 'ssh.log')


def ssh_calls(env):
    return [json.loads(line) for line in env['ssh'].read_text().splitlines()] if env['ssh'].exists() else []


def spark_argv(env, *extra, hosts='dodo,emu,rhea,moa'):
    test = env['tmp'] / 'chain-test'
    if not test.exists():
        test.write_text('#!/bin/sh\n')
        test.chmod(0o755)
    return SPARK_BASE + ['--model', 'org/model', '--arm', 'a=i1:s1', '--gpus', '1', '--config', str(env['config']),
                         '--spark-hosts', hosts, '--expert-layers', '3', '--worker-poll', '0.05',
                         '--native-test', str(test), *extra,
                         '--', 'families::step5::chain::tests::native_step5_layer_chain', '--exact', '--ignored']


PASSED = 'running 1 test\ntest families::step5::chain::tests::native_step5_layer_chain ... ok\n' \
         'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1271 filtered out'


@pytest.mark.parametrize('hosts,match', [
    ('oliver', 'Kimi'),
    ('kakapo', 'build seed'), ('dodo,dodo', 'repeats'), ('emu,nosuch', 'not a pool Spark'), (' , ', 'no host'),
])
def test_spark_hosts_refused(hosts, match):
    with pytest.raises(ValueError, match=match):
        golden.parse_hosts(hosts)


def test_expert_layers_and_lock_order():
    assert golden.parse_layers('3') == [3]
    assert golden.parse_layers('5-7,3,6') == [3, 5, 6, 7]
    for bad in ('', 'x', '7-5', '3,'):
        with pytest.raises(ValueError):
            golden.parse_layers(bad)
    # Canonical order: sparks (pool Sparks) -> gpu0 -> gpu1 -> rhea -> moa, whatever the given order.
    assert golden.locks_for([1], ['moa', 'rhea', 'emu', 'dodo']) == ['sparks', 'gpu1', 'rhea', 'moa']
    assert golden.locks_for([0, 1], ['rhea']) == ['gpu0', 'gpu1', 'rhea']
    assert golden.locks_for([0], ['dodo']) == ['sparks', 'gpu0']


def test_golden_subcommand_needs_peers_placeholder_with_sparks():
    with pytest.raises(ValueError, match='add `--peers {peers}`'):
        golden.golden_args('glm', ['--layers', '5'], sparks=True)
    with pytest.raises(ValueError, match='--peers {peers}'):
        golden.golden_args('glm', ['--peers', '10.0.0.1:1'], sparks=True)
    assert golden.golden_args('glm', ['--peers', '{peers}'], sparks=True)[-2:] == ['--peers', '{peers}']
    built = golden.substitute(['--peers', '{peers}', '--x={expert_layers}'], '/s', peers='a:1,b:1', expert_layers='3')
    assert built == ['--peers', 'a:1,b:1', '--x=3']
    with pytest.raises(ValueError, match='no value'):
        golden.substitute(['{fixtures}'], '/s')


@pytest.mark.parametrize('seconds', ['0', '-1'])
def test_worker_admission_rejects_nonpositive_launch_bound(spark_env, seconds, capsys):
    assert golden.main(spark_argv(spark_env, '--worker-ready-timeout', seconds)) == 2
    assert '--worker-ready-timeout must be positive seconds' in capsys.readouterr().err


def test_real_experts_start_before_and_stop_before_locks_release(spark_env, monkeypatch):
    env = spark_env
    monkeypatch.setenv('STUB_RUN_OUTPUT', PASSED)
    monkeypatch.setenv('STUB_RUN_ENV', str(env['tmp'] / 'run-env.json'))
    order, released = [], []
    real_hold = golden.hwlock.hold

    class Hold:
        def __init__(self, names, *a, **k):
            order.append(list(names))
            self.inner = real_hold(names, *a, **k)

        def __enter__(self):
            return self.inner.__enter__()

        def __exit__(self, *exc):
            released.append(json.loads(env['state'].read_text()))
            return self.inner.__exit__(*exc)
    monkeypatch.setattr(golden.hwlock, 'hold', Hold)
    assert golden.main(spark_argv(env, '--worker-ready-timeout', '2400', '--worker-memory-bytes', '115964116992',
                                  '--worker-env', 'CUTEAFD_X=base', '--arm-worker-env', 'a:CUTEAFD_X=arm')) == 0
    assert order == [['sparks', 'gpu1', 'rhea', 'moa']]
    assert released == [{}]  # every worker and the coordinator removed before the locks went
    calls = docker_calls(env)
    starts = [i for i, c in enumerate(calls) if c[:2] == ['run', '-d']]
    coordinator = next(i for i, c in enumerate(calls) if c[:2] == ['run', '--rm'])
    assert len(starts) == 4 and max(starts) < coordinator
    # Rank = position in --spark-hosts, the serving path's worker from the sealed slot.
    for rank, (host, command) in enumerate([c for c in ssh_calls(env) if 'expertd-native' in c[1]]):
        assert host == SPARKS[rank]
        argv = shlex.split(command)
        assert argv[argv.index('--rank') + 1] == str(rank) and argv[argv.index('--world') + 1] == '4'
        assert argv[argv.index('--listen') + 1] == '0.0.0.0:19451'
        assert argv[argv.index('--name') + 1] == f'golden-gr-a-expert{rank}'
        assert '--network' in argv and 'nvidia.com/gpu=all' in argv and '--device=/dev/infiniband' in argv
        assert argv[argv.index('--entrypoint') + 1] == 'timeout'
        mounts = [argv[i + 1] for i, a in enumerate(argv) if a == '-v']
        assert all(m.endswith(':ro') for m in mounts)
        assert any('/spark-expert/workspace/.cuteafd-wip/libcuteafd_native.so:/opt/cuteafd/lib/libcuteafd_native.so:ro' in m
                   for m in mounts)
        assert any(m.endswith(':/opt/cuteafd/lib/fp8:ro') for m in mounts)
        assert 'CUTEAFD_PROGRAM_MANIFEST=/opt/cuteafd/share/PROGRAMS.json' in argv
        assert argv[argv.index('--memory') + 1] == argv[argv.index('--memory-swap') + 1] == '115964116992'
        assert 'CUTEAFD_X=arm' in argv and 'CUTEAFD_X=base' not in argv
    run_env = json.loads((env['tmp'] / 'run-env.json').read_text())
    assert run_env['CUTEAFD_GOLDEN_PEERS'] == '10.9.0.1:19451,10.9.0.2:19451,10.9.0.3:19451,10.9.0.4:19451'
    assert run_env['CUTEAFD_GOLDEN_SPARK_WORLD'] == '4' and run_env['CUTEAFD_GOLDEN_EXPERT_LAYERS'] == '3'
    assert run_env['CUTEAFD_GOLDEN_SNAPSHOT'] == '/hf/hub/models--org--model/snapshots/rev1'
    assert run_env['CUTEAFD_PROGRAM_MANIFEST'] == '/opt/cuteafd/share/PROGRAMS.json'
    assert run_env['CUTEAFD_GOLDEN_SPARK_PACKAGES'] == '/opt/cuteafd/spark'
    assert run_env['CUTEAFD_SPARK_ADMISSION_TIMEOUT_SECONDS'] == '2400'
    call = calls[coordinator]
    assert '--network' in call and '--device=/dev/infiniband' in call
    assert call[call.index('-w') + 1] == '/workspace/cuteafd'
    mounts = [call[i + 1] for i, a in enumerate(call) if a == '-v']
    assert any(m.endswith('/coordinator/workspace:/workspace/cuteafd:ro') for m in mounts)
    assert f"{env['tmp']}/chain-test:/opt/cuteafd/test/chain-test:ro" in mounts
    out = env['home'] / '.cache/cuteafd/builds/gr/golden/a'
    assert any(m == f'{out}/spark-packages:/opt/cuteafd/spark:ro' for m in mounts)
    assert (out / 'spark-packages/fp8/fp8-step5/manifest.json').read_text() == '{"layouts": {"tp4": {"capacity": 64}}}'
    tail = call[call.index(IMAGE) + 1:]
    assert tail[-4:] == ['/opt/cuteafd/test/chain-test', 'families::step5::chain::tests::native_step5_layer_chain',
                         '--exact', '--ignored']
    for host in SPARKS:
        assert golden.WORKER_READY in (out / f'r1/workers/{host}.log').read_text()
    record = json.loads((out / 'r1/argv.json').read_text())
    assert [w['host'] for w in record['workers']] == SPARKS and record['native_test_sha256']
    assert set(record['spark_packages']) == {'fp8/fp8-step5/manifest.json'}
    summary = (out.parent / 'summary.md').read_text()
    assert 'Spark expert ranks: 0=dodo, 1=emu, 2=rhea, 3=moa' in summary
    assert 'test result: ok. 1 passed' in summary and 'dodo: ready; 0 error lines' in summary


@pytest.mark.parametrize('output,problem', [
    ('running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 1272 filtered out', 'no test ran'),
    ('test result: FAILED. 0 passed; 1 failed; 0 ignored', 'tests failed'),
    ('no summary at all', 'no `test result:`'),
])
def test_native_test_needs_a_passing_summary(spark_env, monkeypatch, output, problem):
    monkeypatch.setenv('STUB_RUN_OUTPUT', output)
    argv = [a for a in spark_argv(spark_env) if a]
    assert golden.main(argv) == 1
    record = json.loads((spark_env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/argv.json').read_text())
    assert problem in record['problem']
    assert json.loads(spark_env['state'].read_text()) == {}


def test_worker_exit_before_ready_fails_fast_and_tears_down(spark_env, monkeypatch):
    monkeypatch.setenv('STUB_WORKER_FAIL', 'rhea')
    assert golden.main(spark_argv(spark_env)) == 1
    calls = docker_calls(spark_env)
    assert not any(c[:2] == ['run', '--rm'] for c in calls)  # the coordinator never ran
    record = json.loads((spark_env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/argv.json').read_text())
    assert 'rank 2 on rhea exited before readiness' in record['problem']
    assert json.loads(spark_env['state'].read_text()) == {}
    log = spark_env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/workers/rhea.log'
    assert 'NV_ERR_NO_MEMORY' in log.read_text()
    summary = (spark_env['home'] / '.cache/cuteafd/builds/gr/golden/summary.md').read_text()
    assert 'rhea: NOT ready; 1 error lines' in summary


def test_worker_readiness_timeout_tears_down(spark_env, monkeypatch):
    monkeypatch.setenv('STUB_WORKER_SLOW', 'moa')
    assert golden.main(spark_argv(spark_env, '--worker-ready-timeout', '1')) == 1
    record = json.loads((spark_env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/argv.json').read_text())
    assert 'not ready after 1 s' in record['problem'] and '[3]' in record['problem']
    assert not any(c[:2] == ['run', '--rm'] for c in docker_calls(spark_env))
    assert json.loads(spark_env['state'].read_text()) == {}


def test_run_timeout_tears_down_workers(spark_env, monkeypatch):
    monkeypatch.setenv('STUB_RUN_SLEEP', '30')
    assert golden.main(spark_argv(spark_env, '--timeout', '1')) == 1
    record = json.loads((spark_env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/argv.json').read_text())
    assert record['exit'] == 124 and 'timed out' in record['problem']
    assert json.loads(spark_env['state'].read_text()) == {}
    # The workers bound themselves too: in-container timeout beyond the run's budget.
    command = next(c[1] for c in ssh_calls(spark_env) if 'expertd-native' in c[1])
    argv = shlex.split(command)
    image = argv.index('sha256:' + 'cd' * 32)
    assert argv[argv.index('--entrypoint') + 1] == 'timeout' and argv[image + 1:image + 3] == ['--kill-after=30', str(1 + 1500 + 120)]


def test_sigterm_tears_down_workers_and_releases_locks(spark_env, monkeypatch):
    import threading
    import time
    monkeypatch.setenv('STUB_RUN_SLEEP', '30')
    # Signal once the coordinator container exists (under load a fixed delay can
    # fire before the workers start, which tests nothing).
    def signal_when_running():
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if spark_env['state'].exists() and any(
                    n.get('name') == 'golden-gr-a' for n in json.loads(spark_env['state'].read_text() or '{}').values()):
                break
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=signal_when_running, daemon=True).start()
    assert golden.main(spark_argv(spark_env)) == 130
    assert json.loads(spark_env['state'].read_text()) == {}
    for name in ('sparks', 'gpu1', 'rhea', 'moa'):
        with open(spark_env['locks'] / f'{name}.lock', 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_spark_slot_from_other_sources_is_refused_before_locks(spark_env, monkeypatch, capsys):
    import shutil
    root = spark_env['home'] / '.cache/cuteafd/builds/wip-i1/slots/s1/spark-expert'
    shutil.rmtree(root)
    write_spark_slot(spark_env['home'], 'i1', source=b'other  ./y\n')
    taken = []
    monkeypatch.setattr(golden.hwlock, 'hold', lambda *a, **k: taken.append(a))
    assert golden.main(spark_argv(spark_env)) == 2
    assert 'different sources than the coordinator slot' in capsys.readouterr().err
    assert not taken and not runs(spark_env)


def test_leftover_worker_is_refused(spark_env, capsys):
    spark_env['state'].write_text(json.dumps({'id-x': {'name': 'golden-gr-a-expert2', 'labels': {}}}))
    assert golden.main(spark_argv(spark_env)) == 2
    assert 'golden-gr-a-expert2 already exists' in capsys.readouterr().err


def test_dry_run_prints_workers_and_takes_nothing(spark_env, monkeypatch, capsys):
    taken = []
    monkeypatch.setattr(golden.hwlock, 'hold', lambda *a, **k: taken.append(a))
    assert golden.main(spark_argv(spark_env, '--dry-run')) == 0
    out = capsys.readouterr().out
    assert '# locks (fair queue, in order): sparks,gpu1,rhea,moa' in out
    for rank, host in enumerate(SPARKS):
        assert f'# a r1 expert rank {rank} on {host}' in out
    assert 'native_step5_layer_chain' in out and 'golden-run: preflight pass' in out
    assert not taken and not runs(spark_env)
    assert not any(c[:2] == ['run', '-d'] for c in docker_calls(spark_env))
    assert not (spark_env['home'] / '.cache/cuteafd/builds/gr/golden').exists()


def test_spark_hosts_without_lanes_in_config(spark_env, capsys):
    spark_env['config'].write_text('SPARK_0_HOST=dodo\nSPARK_0_LANE_A=10.9.0.1\n')
    assert golden.main(spark_argv(spark_env)) == 2
    assert 'no SPARK_i_HOST/SPARK_i_LANE_A for emu, rhea, moa' in capsys.readouterr().err


def test_fixture_generation_command_mode(env, monkeypatch):
    write_slot(env['home'], 'i1')
    monkeypatch.setenv('STUB_RUN_ENV', str(env['tmp'] / 'run-env.json'))
    fixtures = env['home'] / '.cache/cuteafd/builds/gr/fixtures/chain'
    argv = SPARK_BASE + ['--model', 'org/model', '--arm', 'a=i1:s1', '--gpus', '1', '--command',
                         '--fixtures', str(fixtures), '--fixtures-writable',
                         '--env', 'CUTEAFD_STEP5_CHAIN_FIXTURE_DIR={fixtures}', '--env', 'CARGO_TARGET_DIR={fixtures}',
                         '--env', 'STEP5_QUANT_SNAPSHOT={snapshot}', '--env', 'STEP5_REFERENCE_SNAPSHOT={hub}/models--x/snapshots/r',
                         '--', 'python3', '-m', 'pytest', '-q', 'python/tests/test_step5_chain_fixture.py']
    assert golden.main(argv) == 0
    assert fixtures.is_dir()
    call = runs(env)[0]
    mounts = [call[i + 1] for i, a in enumerate(call) if a == '-v']
    assert f'{fixtures}:{fixtures}' in mounts  # writable
    assert any(m.endswith('/coordinator/workspace:/workspace/cuteafd:ro') for m in mounts)
    assert '--network' not in call and call[call.index('-w') + 1] == '/workspace/cuteafd'
    assert call[-5:] == ['python3', '-m', 'pytest', '-q', 'python/tests/test_step5_chain_fixture.py']
    run_env = json.loads((env['tmp'] / 'run-env.json').read_text())
    assert run_env['CUTEAFD_STEP5_CHAIN_FIXTURE_DIR'] == str(fixtures) == run_env['CUTEAFD_GOLDEN_FIXTURES']
    assert run_env['STEP5_QUANT_SNAPSHOT'] == '/hf/hub/models--org--model/snapshots/rev1'
    assert run_env['STEP5_REFERENCE_SNAPSHOT'] == '/hf/hub/models--x/snapshots/r'
    assert run_env['PYTHONDONTWRITEBYTECODE'] == '1' and '/workspace/cuteafd/python/reference' in run_env['PYTHONPATH']
    assert 'CUTEAFD_GOLDEN_PEERS' not in run_env


@pytest.mark.parametrize('extra,match', [
    (['--fixtures-writable'], 'needs --fixtures'),
    (['--fixtures', '/tmp/elsewhere', '--fixtures-writable'], 'must be under'),
    (['--fixtures', 'TASK/golden/x', '--fixtures-writable'], 'outside golden/'),
    (['--fixtures', 'TASK/fx', '--fixtures-writable', '--spark-hosts', 'dodo'], 'GPU-only'),
    (['--expert-layers', '3'], 'need --spark-hosts'),
    (['--arm-worker-env', 'a:K=V'], 'need --spark-hosts'),
    (['--spark-hosts', 'dodo', '--arm-worker-env', 'zz:K=V'], 'names no arm'),
    (['--worker-memory-bytes', '115964116992'], 'need --spark-hosts'),
    (['--arm-arg', 'zz:--x'], 'names no arm'),
    (['--spark-hosts', 'dodo', '--worker-memory-bytes', '1'], 'must exceed --worker-budget-bytes'),
    (['--fixtures', 'relative/dir'], 'absolute'),
])
def test_fixture_and_spark_argument_refusals(env, capsys, extra, match):
    write_slot(env['home'], 'i1')
    task = str(env['home'] / '.cache/cuteafd/builds/gr')
    extra = [e.replace('TASK', task) for e in extra]
    argv = SPARK_BASE + ['--model', 'org/model', '--arm', 'a=i1:s1', '--command', *extra, '--', 'true']
    assert golden.main(argv) == 2
    assert match in capsys.readouterr().err
    assert not runs(env)


def test_dense_native_test_without_sparks(env, monkeypatch):
    write_slot(env['home'], 'i1')
    monkeypatch.setenv('STUB_RUN_OUTPUT', PASSED.replace('layer_chain', 'dense_chain'))
    test = env['tmp'] / 'chain-test'
    test.write_text('#!/bin/sh\n')
    test.chmod(0o755)
    fixtures = env['tmp'] / 'fixtures'
    fixtures.mkdir()
    order = []
    real_hold = golden.hwlock.hold
    monkeypatch.setattr(golden.hwlock, 'hold', lambda names, *a, **k: (order.append(list(names)), real_hold(names, *a, **k))[1])
    argv = SPARK_BASE + ['--model', 'org/model', '--arm', 'a=i1:s1', '--gpus', '1', '--native-test', str(test),
                         '--fixtures', str(fixtures), '--', 'families::step5::chain::tests::native_step5_dense_chain',
                         '--exact', '--ignored']
    assert golden.main(argv) == 0
    assert order == [['gpu1']]
    call = runs(env)[0]
    mounts = [call[i + 1] for i, a in enumerate(call) if a == '-v']
    assert f'{fixtures}:{fixtures}:ro' in mounts and '--network' not in call


def test_spark_port_checked_only_at_grant():
    """A run queued behind another golden must not fail preflight on its port."""
    import inspect
    source = inspect.getsource(golden.check_spark)
    assert "if at_grant else '-'" in source
    assert "if port!='-':" in golden.REMOTE_CHECK
    assert inspect.signature(golden.preflight).parameters['at_grant'].default is True
