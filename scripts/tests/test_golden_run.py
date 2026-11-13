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
def images():
    path = os.environ.get('STUB_DOCKER_IMAGES')
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}
def image(ref):
    table = images().get(host) or images().get('*')
    if table is None:  # no per-host table: every image exists with one common identity
        return {'id': ref, 'layers': ['sha256:default']}
    return table.get(ref) or next((entry for entry in table.values() if entry.get('id') == ref), None)
if args[:2] == ['image', 'inspect']:
    if args[2] == '-f':
        ref = args[4]
        if '.RootFS.Layers' in args[3]:
            entry = image(ref)
            if not entry:
                sys.exit(1)
            print(json.dumps(entry.get('layers', []))); sys.exit(0)
        entry = image(ref)
        print(entry['id'] if entry else ref); sys.exit(0)
    print(json.dumps([{'Id': args[2], 'RepoTags': ['cuteafd-coordinator-dev:test']}])); sys.exit(0)
if args[:2] == ['image', 'ls']:
    table = images().get(host) or images().get('*') or {}
    print('\n'.join(entry['id'] for entry in table.values())); sys.exit(0)
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
if args[:1] == ['run'] and args[-1:] == ['--help']:
    print(os.environ['STUB_GOLDEN_HELP']); sys.exit(0)
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
    monkeypatch.setenv('STUB_GOLDEN_HELP', CLI_HELP)
    return dict(home=home, tmp=tmp_path, locks=locks, log=tmp_path / 'docker.log', state=tmp_path / 'docker-state.json')


def docker_calls(env):
    return [json.loads(line) for line in env['log'].read_text().splitlines()] if env['log'].exists() else []


def runs(env):
    return [c for c in docker_calls(env) if c[:1] == ['run'] and c[-1:] != ['--help']]


BASE = ['--task', 'gr', '--family', 'glm', '--model', 'org/model']
GOLDEN = ['--', '--snapshot', '{snapshot}', '--device', '0', '--split-device', '1', '--skip-routed-experts', '--nll']


def test_minimax_golden_uses_shared_runner_subcommand():
    assert golden.golden_args('minimax', ['--snapshot', '{snapshot}', '--nll']) == [
        'minimax-golden', '--snapshot', '{snapshot}', '--nll']


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
        assert call[call.index('--security-opt') + 1].endswith('/docker/seccomp-code-bench.json')
        mounts = [call[i + 1] for i, a in enumerate(call) if a == '-v']
        assert "/mnt/sparknest:/mnt/sparknest:ro" in mounts
        assert f"{env['tmp'] / 'hub'}:{env['tmp'] / 'hub'}:ro" in mounts
        assert any(m.endswith(':/opt/cuteafd/lib/libcuteafd_native.so:ro') for m in mounts)
        assert any(m.endswith(':/opt/cuteafd/share:ro') for m in mounts)
        assert any(m.endswith(f'/gr/golden/{mode}/r1:/out') for m in mounts)
        assert IMAGE in call
        tail = call[call.index(IMAGE) + 1:]
        assert (tail[-2:] == ['--trace', '/out/t']) == (mode == 'heads')
        if mode == 'heads':
            tail = tail[:-2]
        assert tail[-8:] == ['glm-golden', '--snapshot', str(env['tmp'] / 'hub/models--org--model/snapshots/rev1'), '--device', '0',
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


def test_require_identical_fails_on_differing_logits(env, monkeypatch):
    write_slot(env['home'], 'i1')
    logits = env['tmp'] / 'logits.bin'
    np.arange(32, dtype='<f4').tofile(logits)
    monkeypatch.setattr(golden, 'compare_pairs',
                        lambda *a: [{'a': 'a', 'b': 'b', 'repeat': 1, 'file': 'logits.bin', 'byte_identical': False, 'rows': 0}])
    monkeypatch.setenv('STUB_LOGITS', str(logits))
    argv = BASE + ['--arm', 'a=i1:s1', '--arm', 'b=i1:s1', '--compare-logits', 'a:b'] + GOLDEN + \
        ['--save-logits', '{out}/logits']
    assert golden.main(argv) == 0  # report-only by default
    assert golden.main(argv[:len(BASE)] + ['--require-identical', '--overwrite'] + argv[len(BASE):]) == 1


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


def log_probs(values) -> np.ndarray:
    """Row-wise log-softmax: the probe's `log_probs` tensor is already normalized."""
    values = np.asarray(values, dtype=np.float64)
    top = values.max(axis=-1, keepdims=True)
    return (values - top - np.log(np.exp(values - top).sum(axis=-1, keepdims=True))).astype('<f4')


def write_row(path: Path, values) -> None:
    """One row as the probe writes it: the safetensors header from rows.rs."""
    data = np.asarray(values, dtype='<f4').tobytes()
    header = json.dumps({'log_probs': {'dtype': 'F32', 'shape': [len(values)],
                                       'data_offsets': [0, len(data)]}}).encode()
    header += b' ' * (-len(header) % 8)
    path.write_bytes(len(header).to_bytes(8, 'little') + header + data)


def test_compare_tool_reads_safetensors_rows(tmp_path):
    rng = np.random.default_rng(1)
    rows = log_probs(rng.normal(size=(3, 40)))
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    for index, row in enumerate(rows):
        write_row(a / f'row-{index:08}.safetensors', row)
        write_row(b / f'row-{index:08}.safetensors', row)
    same = compare.compare_paths(a, b)
    assert same['layout'] == 'rows' and same['rows'] == 3 and same['byte_identical']
    assert same['max_abs_diff'] == 0 and same['mean_kl'] == 0 and same['top1_agreement'] == 1
    # Perturb one row into a new top-1: the row is named and its metrics are its own.
    perturbed = rows[1].copy()
    perturbed[3] = perturbed.max() + 2
    write_row(b / 'row-00000001.safetensors', log_probs(perturbed))
    diff = compare.compare_paths(a, b)
    assert not diff['byte_identical'] and diff['rows'] == 3 and diff['vocab'] == 40
    assert diff['worst_row'] == 'row-00000001.safetensors'
    assert diff['worst_abs_diff_row'] == 'row-00000001.safetensors'
    assert diff['top1_agreement'] == pytest.approx(2 / 3)
    # Only one row moved, so its KL is the mean's three times over.
    assert diff['mean_kl'] == pytest.approx(diff['max_kl'] / 3)
    assert diff['max_kl'] > 0 and diff['max_abs_diff'] > 0
    # A row only one side wrote is a difference, and a width mismatch names its row.
    (b / 'row-00000002.safetensors').unlink()
    missing = compare.compare_paths(a, b)
    assert missing['only_a'] == 1 and missing['rows'] == 2 and not missing['byte_identical']
    write_row(b / 'row-00000001.safetensors', np.zeros(39))
    with pytest.raises(ValueError, match='widths differ'):
        compare.compare_paths(a, b)
    cli = subprocess.run([sys.executable, str(COMPARE), str(a), str(b)], capture_output=True, text=True)
    assert cli.returncode == 2 and 'widths differ' in cli.stderr


def test_compare_pairs_reads_safetensors_row_dumps(tmp_path):
    root = tmp_path / 'golden'
    rows = log_probs(np.random.default_rng(2).normal(size=(2, 24)))
    for arm in ('a', 'b'):
        (root / arm / 'r1' / 'logits').mkdir(parents=True)
        for index, row in enumerate(rows):
            write_row(root / arm / 'r1' / 'logits' / f'row-{index:08}.safetensors', row)
    state = dict(root=str(root), task='t', family='kimi', snapshot='org/model', gpus='0', locks='gpu0',
                 golden=['kimi-golden'], mode='golden', vocab=24, slots={})
    results = golden.compare_pairs(state, [('a', 'b')], 1)
    assert len(results) == 1 and not results[0].get('error')
    assert results[0]['file'] == 'logits' and results[0]['rows'] == 2 and results[0]['byte_identical']
    golden.write_summary(state, [], results)
    assert '| a | b | 1 | logits | True | 2 |' in (root / 'summary.md').read_text()
    bad = rows[1].copy()
    bad[0] = bad.max() + 3
    write_row(root / 'b/r1/logits/row-00000001.safetensors', log_probs(bad))
    results = golden.compare_pairs(state, [('a', 'b')], 1)
    assert results[0]['worst_row'] == 'row-00000001.safetensors'
    golden.write_summary(state, [], results)
    table = next(line for line in (root / 'summary.md').read_text().splitlines()
                 if line.startswith('| a | b |') and 'repeat' not in line)
    assert '| False |' in table and 'row-00000001.safetensors' in table


@pytest.mark.parametrize('positions', [None, [1, 2], [4, 5]])
def test_compare_pairs_aligns_bin_nll_with_scored_positions(tmp_path, positions):
    root = tmp_path / 'golden'
    logits = np.array([[8, 0, -2], [-2, 0, 8]], dtype='<f4')
    tokens = np.array([2, 0, 2] if positions is None or positions[0] == 1
                      else [2, 1, 1, 1, 0, 2], dtype='<u4')
    for arm in ('a', 'b'):
        out = root / arm / 'r1' / 'window'
        out.mkdir(parents=True)
        logits.tofile(out / 'logits.bin')
        tokens.tofile(out / 'tokens.bin')
        if positions is not None:
            (out / 'meta.json').write_text(json.dumps(dict(positions=positions)))
    state = dict(root=str(root), vocab=3, slots={})
    result, = golden.compare_pairs(state, [('a', 'b')], 1)
    assert not result.get('error') and result['nll_rows'] == 2
    expected = -float(log_probs(logits)[np.arange(2), [0, 2]].mean())
    assert result['nll_a'] == pytest.approx(expected, abs=1e-7)
    assert result['nll_b'] == pytest.approx(expected, abs=1e-7)
    assert result['mean_kl'] == 0 and result['top1_agreement'] == 1


@pytest.mark.parametrize('with_metadata', [False, True])
def test_compare_pairs_preserves_full_prompt_bin_final_unscored_row(tmp_path, with_metadata):
    root = tmp_path / 'golden'
    logits = np.array([[8, 0, -2], [-2, 0, 8], [1, 2, 3]], dtype='<f4')
    for arm in ('a', 'b'):
        out = root / arm / 'r1'
        out.mkdir(parents=True)
        logits.tofile(out / 'logits.bin')
        np.array([2, 0, 2], dtype='<u4').tofile(out / 'tokens.bin')
        if with_metadata:
            (out / 'meta.json').write_text(json.dumps(dict(positions=[1, 2, 3])))
    result, = golden.compare_pairs(dict(root=str(root), vocab=3, slots={}), [('a', 'b')], 1)
    assert not result.get('error') and result['rows'] == 3 and result['nll_rows'] == 2
    assert result['nll_a'] == pytest.approx(-float(log_probs(logits[:2])[np.arange(2), [0, 2]].mean()), abs=1e-7)


@pytest.mark.parametrize('positions', [None, [0, 1], [4], [4, 6], [4, 6.0], [5, 6]])
def test_compare_pairs_refuses_ambiguous_bin_nll_alignment(tmp_path, positions):
    root = tmp_path / 'golden'
    for arm in ('a', 'b'):
        out = root / arm / 'r1'
        out.mkdir(parents=True)
        np.zeros((2, 3), dtype='<f4').tofile(out / 'logits.bin')
        np.zeros(6, dtype='<u4').tofile(out / 'tokens.bin')
        if positions is not None:
            (out / 'meta.json').write_text(json.dumps(dict(positions=positions)))
    result, = golden.compare_pairs(dict(root=str(root), vocab=3, slots={}), [('a', 'b')], 1)
    assert result.get('error')


@pytest.mark.parametrize('mismatch', ['positions', 'tokens'])
def test_compare_pairs_refuses_misaligned_bin_arms(tmp_path, mismatch):
    root = tmp_path / 'golden'
    for arm in ('a', 'b'):
        out = root / arm / 'r1'
        out.mkdir(parents=True)
        np.zeros((2, 3), dtype='<f4').tofile(out / 'logits.bin')
        np.array([0, 0, int(arm == 'b' and mismatch == 'tokens'), 0, 0, 0],
                 dtype='<u4').tofile(out / 'tokens.bin')
        positions = [3, 4] if arm == 'b' and mismatch == 'positions' else [4, 5]
        (out / 'meta.json').write_text(json.dumps(dict(positions=positions)))
    result, = golden.compare_pairs(dict(root=str(root), vocab=3, slots={}), [('a', 'b')], 1)
    assert 'different' in result['error']


def test_resolve_model_uses_hub_paths(env):
    host, container, extra = golden.resolve_model('org/model')
    assert container == str(host) and extra == [str(host)]
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


SEALED_LAYERS = ['sha256:' + '11' * 32, 'sha256:' + '22' * 32]
SEALED_ID = 'sha256:' + 'cd' * 32


def local_id(host):
    return 'sha256:' + hashlib.sha256(host.encode()).hexdigest()


def seal_spark_layers(home, layers, instance='i1', slot='s1'):
    """Reseal the spark-expert slot's META with its store-independent layer list."""
    root = home / '.cache/cuteafd/builds' / f'wip-{instance}' / 'slots' / slot / 'spark-expert'
    meta = json.loads((root / 'META.json').read_text())
    meta['base_image_layers'] = layers
    data = json.dumps(meta).encode()
    (root / 'META.json').write_bytes(data)
    (root / 'FINGERPRINT').write_text(hashlib.sha256(data).hexdigest() + '\n')


def stub_images(env, monkeypatch, per_host):
    """The docker stub's per-host images: {host: {ref: {'id': ..., 'layers': [...]}}}."""
    path = env['tmp'] / 'spark-images.json'
    path.write_text(json.dumps(per_host))
    monkeypatch.setenv('STUB_DOCKER_IMAGES', str(path))


def worker_images(env):
    images = []
    for host, command in ssh_calls(env):
        if 'expertd-native' not in command:
            continue
        argv = shlex.split(command)
        images.append((host, argv[argv.index('--entrypoint') + 2]))
    return images


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
    ('maxq', 'not a pool Spark'),
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
    assert run_env['CUTEAFD_GOLDEN_SNAPSHOT'] == str(env['tmp'] / 'hub/models--org--model/snapshots/rev1')
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


def test_spark_dev_image_is_matched_by_layers_not_store_ids(spark_env, monkeypatch):
    """The seed's store-local .Id differs from each target's: only layers identify the image."""
    env = spark_env
    monkeypatch.setenv('STUB_RUN_OUTPUT', PASSED)
    seal_spark_layers(env['home'], SEALED_LAYERS)
    stub_images(env, monkeypatch, {host: {'spark:tc': {'id': local_id(host), 'layers': SEALED_LAYERS}}
                                    for host in SPARKS})
    assert golden.main(spark_argv(env)) == 0
    assert worker_images(env) == [(host, local_id(host)) for host in SPARKS]


def test_spark_dev_image_with_different_layers_is_refused(spark_env, monkeypatch, capsys):
    env = spark_env
    seal_spark_layers(env['home'], SEALED_LAYERS)
    table = {host: {'spark:tc': {'id': local_id(host), 'layers': SEALED_LAYERS}} for host in SPARKS}
    table['emu']['spark:tc']['layers'] = ['sha256:' + 'ff' * 32]
    stub_images(env, monkeypatch, table)
    assert golden.main(spark_argv(env)) == 2
    assert 'emu: dev image spark:tc does not match the sealed spark-expert image layers' in capsys.readouterr().err
    assert not runs(env)


def test_spark_dev_image_accepts_an_identical_local_copy_under_another_name(spark_env, monkeypatch):
    env = spark_env
    monkeypatch.setenv('STUB_RUN_OUTPUT', PASSED)
    seal_spark_layers(env['home'], SEALED_LAYERS)
    stub_images(env, monkeypatch, {host: {'loaded-' + host: {'id': local_id(host), 'layers': SEALED_LAYERS}}
                                    for host in SPARKS})
    assert golden.main(spark_argv(env)) == 0
    assert worker_images(env) == [(host, local_id(host)) for host in SPARKS]


def test_older_slot_without_layers_uses_the_seed_image_layers(spark_env, monkeypatch):
    """A slot sealed before base_image_layers: the coordinator looks the sealed ID's
    layers up on the build seed (kakapo); the targets carry them under other IDs."""
    env = spark_env
    monkeypatch.setenv('STUB_RUN_OUTPUT', PASSED)
    table = {'kakapo': {SEALED_ID: {'id': SEALED_ID, 'layers': SEALED_LAYERS}}}
    for host in SPARKS:
        table[host] = {'spark:tc': {'id': local_id(host), 'layers': SEALED_LAYERS}}
    stub_images(env, monkeypatch, table)
    assert golden.main(spark_argv(env)) == 0
    assert any(host == 'kakapo' for host, _ in ssh_calls(env))
    assert worker_images(env) == [(host, local_id(host)) for host in SPARKS]


def test_older_slot_without_layers_and_no_seed_image_is_refused(spark_env, monkeypatch, capsys):
    env = spark_env
    table = {'kakapo': {'other:tag': {'id': 'sha256:' + 'ee' * 32, 'layers': ['sha256:other']}}}
    for host in SPARKS:
        table[host] = {'spark:tc': {'id': local_id(host), 'layers': SEALED_LAYERS}}
    stub_images(env, monkeypatch, table)
    assert golden.main(spark_argv(env)) == 2
    assert f'dodo: dev image {SEALED_ID} is missing' in capsys.readouterr().err


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
    image = argv.index('spark:tc')
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


def test_dry_run_reports_locks_without_argv_and_takes_nothing(spark_env, monkeypatch, capsys):
    taken = []
    monkeypatch.setattr(golden.hwlock, 'hold', lambda *a, **k: taken.append(a))
    assert golden.main(spark_argv(spark_env, '--dry-run')) == 0
    out = capsys.readouterr().out
    assert '# locks (fair queue, in order): sparks,gpu1,rhea,moa' in out
    assert 'native_step5_layer_chain' not in out
    assert 'golden-run: preflight pass' in out
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
    assert run_env['STEP5_QUANT_SNAPSHOT'] == str(env['tmp'] / 'hub/models--org--model/snapshots/rev1')
    assert run_env['STEP5_REFERENCE_SNAPSHOT'] == str(env['tmp'] / 'hub/models--x/snapshots/r')
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


def test_kimi_golden_routes_to_its_subcommand():
    args = ['--snapshot', '{snapshot}', '--windows', '/panel', '--peers', '{peers}']
    assert golden.golden_args('kimi', args, sparks=True) == ['kimi-golden', *args]


def test_muse_golden_routes_packing_and_worker_arguments():
    args = ['--snapshot', '{snapshot}', '--golden', '/reference', '--packed-check', '4',
            '--prefill-packing', 'unrestricted', '--peers', '{peers}']
    assert golden.golden_args('muse', args, sparks=True) == ['muse-golden', *args]
    with pytest.raises(ValueError, match='Spark peers come from'):
        golden.golden_args('muse', args, sparks=False)


def test_muse_command_runs_from_sealed_slot_without_sparks(env):
    write_slot(env['home'], 'i1')
    command = ['python3', '-m', 'pytest', '-q',
               'python/tests/test_dense_trellis_k6_exports.py::test_muse_f32_head_chunk_row_invariance']
    argv = ['--task', 'gr', '--family', 'muse', '--model', 'org/model', '--arm', 'a=i1:s1',
            '--gpus', '0', '--command', '--env', 'CUTEAFD_MUSE_HEAD_GPU_TEST=1', '--', *command]
    assert golden.main(argv) == 0
    call = runs(env)[0]
    assert call[-len(command):] == command
    assert call[call.index('-w') + 1] == '/workspace/cuteafd'
    assert '--network' not in call
    assert any('/coordinator/workspace:/workspace/cuteafd:ro' in arg for arg in call)
    record = json.loads((env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/argv.json').read_text())
    assert record['mode'] == 'command' and record['gpus'] == [0]
    assert record['arm_env']['CUTEAFD_MUSE_HEAD_GPU_TEST'] == '1'


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


def test_rdma_loopback_native_test_without_sparks(env, monkeypatch):
    write_slot(env['home'], 'i1')
    monkeypatch.setenv('STUB_RUN_OUTPUT', PASSED)
    test = env['tmp'] / 'loopback-test'
    test.write_text('#!/bin/sh\n')
    test.chmod(0o755)
    argv = SPARK_BASE + ['--model', 'org/model', '--arm', 'a=i1:s1', '--gpus', '1', '--native-test', str(test),
                         '--rdma', '--', 'transport::tests::loopback', '--exact', '--ignored']
    assert golden.main(argv) == 0
    call = runs(env)[0]
    assert '--network' in call and '--device=/dev/infiniband' in call


def test_spark_port_checked_only_at_grant():
    """A run queued behind another golden must not fail preflight on its port."""
    import inspect
    source = inspect.getsource(golden.check_spark)
    assert "if at_grant else '-'" in source
    assert "if port!='-':" in golden.REMOTE_CHECK
    assert inspect.signature(golden.preflight).parameters['at_grant'].default is True


def test_namespace_root_keeps_mount_loading_paths_identical(env, monkeypatch):
    root = env['tmp'] / 'namespace'
    snapshot = root / 'hf-home/hub/models--org--model/snapshots/rev1'
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{"vocab_size": 8}')
    monkeypatch.setenv('CUTEAFD_NAMESPACE_ROOT', str(root))
    monkeypatch.setenv('CUTEAFD_HF_HUB', str(root / 'hf-home/hub'))
    host, container, extra = golden.resolve_model(str(snapshot))
    assert str(host) == container == str(snapshot)
    assert not extra
    assert (Path(container) / 'config.json').read_bytes() == (host / 'config.json').read_bytes()
    assert golden.hub_root() == root / 'hf-home/hub'


def test_command_mode_does_not_auto_mount_host_executables_and_prefers_usr_local_bin(env):
    write_slot(env['home'], 'i1')
    host_exe = env['tmp'] / 'host-nsys'
    host_exe.write_text('#!/bin/sh\n')
    host_exe.chmod(0o755)
    data_file = env['tmp'] / 'inputs.bin'
    data_file.write_bytes(b'data')
    argv = BASE + ['--arm', 'a=i1:s1', '--gpus', '1', '--command', '--',
                   '/usr/local/bin/nsys', '/usr/bin/env', str(host_exe), f'--input={data_file}']
    assert golden.main(argv) == 0
    call = runs(env)[0]
    mounts = [call[i + 1] for i, a in enumerate(call) if a == '-v']
    assert not any('/usr/local/bin/nsys:' in m for m in mounts)
    assert not any('/usr/bin/env:' in m for m in mounts)
    assert f'{host_exe}:{host_exe}:ro' not in mounts
    assert f'{data_file}:{data_file}:ro' in mounts
    assert call[-4:] == ['/usr/local/bin/nsys', '/usr/bin/env', str(host_exe), f'--input={data_file}']
    path_env = next(call[i + 1].split('=', 1)[1] for i, a in enumerate(call)
                    if a == '-e' and call[i + 1].startswith('PATH=')).split(':')
    assert path_env.index('/usr/local/bin') < path_env.index('/usr/local/cuda/bin')

    # Explicit --mount mounts a host executable when the caller asks for it.
    explicit = BASE + ['--arm', 'a=i1:s1', '--gpus', '1', '--command', '--overwrite',
                       '--mount', str(host_exe), '--', str(host_exe)]
    assert golden.main(explicit) == 0
    mounts = [runs(env)[-1][i + 1] for i, a in enumerate(runs(env)[-1]) if a == '-v']
    assert f'{host_exe}:{host_exe}:ro' in mounts


CLI_HELP = """Usage: cuteafd glm-golden --snapshot <SNAPSHOT>
Options:
      --snapshot <SNAPSHOT>
      --device <DEVICE> [default: 0]
      --split-device <SPLIT_DEVICE>
      --skip-routed-experts
      --nll
      --trace <TRACE>
      --golden <GOLDEN>
      --save-logits <SAVE_LOGITS>
      --resume-at <RESUME_AT>
      --resume-span <RESUME_SPAN>
      --resume-decode <RESUME_DECODE>
      --peers <PEERS>
      --prefill-rows <PREFILL_ROWS>
      --prefill-waves <PREFILL_WAVES>
"""


def test_sealed_help_has_no_gpu_or_host_network(monkeypatch, tmp_path):
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, CLI_HELP, '')
    monkeypatch.setattr(golden, 'run', run)
    assert golden.sealed_help({'root': str(tmp_path), 'image_id': IMAGE}, 'glm-golden') == CLI_HELP
    cmd = commands[0]
    assert '--gpus' not in cmd and '--device' not in cmd
    assert cmd[cmd.index('--network') + 1] == 'none'
    assert cmd[-2:] == ['glm-golden', '--help']
    assert cmd[cmd.index('--entrypoint') + 1] == '/usr/bin/timeout'
    assert cmd[cmd.index(IMAGE) + 1] == '45'


@pytest.mark.parametrize('args,message', [
    (['--snapshot', '{snapshot}', '--max-context', '4096'], 'unrecognized arguments'),
    (['--nll'], '--dump-dir'),
    (['--dump-dir', '{out}/dump', '--device', '0'], 'unrecognized arguments'),
])
def test_sealed_cli_refusals_precede_lock(env, monkeypatch, capsys, args, message):
    write_slot(env['home'], 'i1')
    help_text = CLI_HELP if '--max-context' in args else (
        'Usage: cuteafd dsv4-golden --dump-dir <DUMP_DIR>\nOptions:\n      --dump-dir <DUMP_DIR>\n      --nll\n')
    monkeypatch.setattr(golden, 'sealed_help', lambda *a: help_text)
    taken = []
    monkeypatch.setattr(golden.hwlock, 'hold', lambda *a, **k: taken.append(a))
    assert golden.main(BASE + ['--arm', 'a=i1:s1', '--', *args]) == 2
    assert message in capsys.readouterr().err
    assert not taken and not runs(env)


def test_v41_uses_rtx_gpus_not_device(env, monkeypatch, capsys):
    write_slot(env['home'], 'i1')
    text = 'Usage: cuteafd v41-golden\nOptions:\n      --rtx-gpus <RTX_GPUS> [default: 1]\n'
    monkeypatch.setattr(golden, 'sealed_help', lambda *a: text)
    args = BASE.copy()
    args[3] = 'v41'
    assert golden.main(args + ['--arm', 'a=i1:s1', '--preflight-only', '--', '--device', '0']) == 2
    assert 'unrecognized arguments' in capsys.readouterr().err
    assert golden.main(args + ['--arm', 'a=i1:s1', '--preflight-only', '--', '--rtx-gpus', '1']) == 0


def test_physical_gpu_one_is_logical_zero(env, capsys):
    write_slot(env['home'], 'i1')
    base = BASE + ['--arm', 'a=i1:s1', '--gpus', '1', '--preflight-only', '--', '--snapshot', '{snapshot}']
    assert golden.main(base + ['--device', '1']) == 2
    assert 'container-visible logical ordinal' in capsys.readouterr().err
    assert golden.main(base + ['--device', '0']) == 0


@pytest.mark.parametrize('flag,value', [('--prefill-rows', '80'), ('--prefill-rows', '4160'),
                                       ('--prefill-waves', '5')])
def test_invalid_prefill_shape_precedes_lock(env, monkeypatch, capsys, flag, value):
    write_slot(env['home'], 'i1')
    taken = []
    monkeypatch.setattr(golden.hwlock, 'hold', lambda *a, **k: taken.append(a))
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN + [flag, value]) == 2
    assert flag in capsys.readouterr().err
    assert not taken and not runs(env)


def test_sparknest_requires_rdma_before_lock(env, monkeypatch, capsys):
    write_slot(env['home'], 'i1')
    monkeypatch.setenv('CUTEAFD_NAMESPACE_ROOT', str(env['tmp'] / 'hub'))
    base = BASE + ['--arm', 'a=i1:s1', '--preflight-only']
    assert golden.main(base + GOLDEN) == 2
    assert 'add --rdma' in capsys.readouterr().err
    assert golden.main(base + ['--rdma'] + GOLDEN) == 0


def test_missing_reference_env_derived_from_sealed_image(env, monkeypatch):
    write_slot(env['home'], 'i1')
    monkeypatch.delenv('CUTEAFD_REFERENCE_IMAGE', raising=False)
    monkeypatch.delenv('CUTEAFD_REFERENCE_IMAGE_SHA256', raising=False)
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 0
    record = json.loads((env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/argv.json').read_text())
    assert record['env']['CUTEAFD_REFERENCE_IMAGE'] == IMAGE
    assert record['env']['CUTEAFD_REFERENCE_IMAGE_SHA256'] == IMAGE


def test_v41_prefill_uses_live_row_floor(env, monkeypatch, capsys):
    write_slot(env['home'], 'i1')
    text = 'Usage: cuteafd v41-golden\nOptions:\n      --prefill-batch-tokens <ROWS>\n        [default: 80]\n'
    monkeypatch.setattr(golden, 'sealed_help', lambda *a: text)
    args = BASE.copy()
    args[3] = 'v41'
    args += ['--arm', 'a=i1:s1', '--preflight-only', '--', '--prefill-batch-tokens']
    assert golden.main(args + ['80']) == 2
    assert 'page-aligned' in capsys.readouterr().err
    assert golden.main(args + ['128']) == 2
    assert 'live rows' in capsys.readouterr().err
    assert golden.main(args + ['256']) == 0


def test_arm_specific_cli_options_are_validated(env, capsys):
    write_slot(env['home'], 'i1')
    assert golden.main(BASE + ['--arm', 'a=i1:s1', '--arm-arg', 'a:--max-context',
                               '--arm-arg', 'a:4096', '--preflight-only'] + GOLDEN) == 2
    assert '--max-context' in capsys.readouterr().err


def test_dump_dir_is_required_and_writable(env, monkeypatch):
    write_slot(env['home'], 'i1')
    text = 'Usage: cuteafd dsv4-golden --dump-dir <DUMP_DIR>\nOptions:\n      --dump-dir <DUMP_DIR>\n'
    monkeypatch.setattr(golden, 'sealed_help', lambda *a: text)
    assert golden.main(BASE + ['--arm', 'a=i1:s1', '--preflight-only', '--',
                               '--dump-dir', '{out}/dump']) == 0


def test_clap_multiline_defaults_and_optional_values():
    text = """Usage: cuteafd glmf-golden [OPTIONS] --snapshot <SNAPSHOT>

Options:
      --snapshot <SNAPSHOT>
          Checkpoint snapshot directory

      --expert-usage[=<EXPERT_USAGE>]
          Count routes
          [default: true]
          [possible values: true, false]

      --prefill-rows <PREFILL_ROWS>
          Rows; zero selects automatic resolution
          [default: 0]

      --prefill-waves <PREFILL_WAVES>
          Waves; zero selects automatic resolution
          [default: 0]

      --resume-at <RESUME_AT>
          --resume-at accepts multiple positions.

  -h, --help
          Print help
"""
    state = dict(golden=['glmf-golden', '--snapshot', '{snapshot}', '--expert-usage=false'],
                 family='glmf', gpus=[1], snapshot_container='/model', hub='/hub', sparks=None)
    golden.validate_golden_cli(state, {}, text)
    parser, defaults = golden.cli_from_help(text)
    assert defaults['--prefill-rows'] == '0'
    assert parser.parse_args(['--snapshot', '/model', '--expert-usage=false']).expert_usage == 'false'
    assert parser.parse_args(['--snapshot', '/model', '--expert-usage']).expert_usage is None


def test_arm_dump_dir_must_be_writable(env, monkeypatch, capsys):
    write_slot(env['home'], 'i1')
    text = 'Usage: cuteafd dsv4-golden --dump-dir <DUMP_DIR>\nOptions:\n      --dump-dir <DUMP_DIR>\n      --nll\n'
    monkeypatch.setattr(golden, 'sealed_help', lambda *a: text)
    assert golden.main(BASE + ['--arm', 'a=i1:s1', '--arm-arg', 'a:--dump-dir',
                               '--arm-arg', 'a:/readonly/dump', '--preflight-only', '--', '--nll']) == 2
    assert 'write outputs under the arm directory' in capsys.readouterr().err


def test_sealed_help_failure_does_not_print_argv(env, monkeypatch, capsys):
    write_slot(env['home'], 'i1')
    def failed(*args):
        raise subprocess.CalledProcessError(2, ['private-program', 'private-argument'])
    monkeypatch.setattr(golden, 'sealed_help', failed)
    assert golden.main(BASE + ['--arm', 'a=i1:s1', '--preflight-only'] + GOLDEN) == 2
    error = capsys.readouterr().err
    assert 'sealed CLI/container check failed' in error
    assert 'private-program' not in error and 'private-argument' not in error


def test_required_native_library_is_supplied_by_slot_environment():
    text = 'Usage: cuteafd golden --native-lib <NATIVE_LIB> --golden <GOLDEN>\nOptions:\n' + (
        '      --native-lib <NATIVE_LIB>\n      --golden <GOLDEN>\n')
    parser, _ = golden.cli_from_help(text)
    assert parser.parse_args(['--golden', '/golden']).native_lib is None
    with pytest.raises(ValueError, match='--golden'):
        parser.parse_args([])


def test_hidden_diagnostic_flags_are_accepted():
    text = 'Usage: cuteafd glm-golden --snapshot <SNAPSHOT>\nOptions:\n      --snapshot <SNAPSHOT>\n      --nll\n'
    parser, _ = golden.cli_from_help(text)
    assert parser.parse_args(['--snapshot', '/m', '--skip-routed-experts']).skip_routed_experts
    with pytest.raises(ValueError, match='--not-a-flag'):
        parser.parse_args(['--snapshot', '/m', '--not-a-flag'])


def test_hidden_flag_lists_cover_every_hidden_daemon_arg():
    """Every `#[arg(long, hide = true)]` daemon field must be in exactly one list
    (or be a launcher-only option golden runs never pass)."""
    import re as _re
    root = Path(__file__).resolve().parents[2] / 'rust/crates/cuteafd-daemon/src'
    launcher_only = {'--peer-expert-ranges', '--deprecated-peer-expert-ranges'}
    found = {}
    for path in root.rglob('*.rs'):
        text = path.read_text()
        for m in _re.finditer(r'#\[arg\(([^\]]*hide = true[^\]]*)\)\]\s*(?:///[^\n]*\n\s*)*pub (\w+): ([^,\n]+)', text):
            attrs, name, ty = m.groups()
            explicit = _re.search(r'long = "([\w-]+)"', attrs)
            flag = '--' + (explicit[1] if explicit else name.replace('_', '-'))
            found[flag] = ty.strip()
    assert found, 'no hidden daemon args found (pattern drift)'
    for flag, ty in found.items():
        if flag in launcher_only:
            continue
        expected = golden.HIDDEN_FLAGS if ty == 'bool' else golden.HIDDEN_OPTIONS
        assert flag in expected, f'{flag} ({ty}) is hidden in the daemon but missing from golden-run'
    for flag in golden.HIDDEN_FLAGS + golden.HIDDEN_OPTIONS:
        assert flag in found, f'{flag} is listed in golden-run but no daemon declares it hidden'


def test_hidden_options_take_an_operand():
    text = 'Usage: cuteafd glmf-golden --snapshot <SNAPSHOT>\nOptions:\n      --snapshot <SNAPSHOT>\n'
    parser, _ = golden.cli_from_help(text)
    parsed = parser.parse_args(['--snapshot', '/m', '--save-logits', '/out/l', '--skip-experts'])
    assert parsed.save_logits == '/out/l' and parsed.skip_experts
